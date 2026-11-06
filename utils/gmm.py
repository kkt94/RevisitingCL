# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

import logging
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features

logger = logging.getLogger()

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

def evaluate_gmm_on_all_task(params, task_id, CL_dataset, test_loader_list, model, tokenizer, accelerator) -> dict:
    """
    Cluster backbone features with a Gaussian mixture model and evaluate the clustering.
    The classifier is not used; only how well the features cluster by class is evaluated.

    Args:
        - params: experiment configuration
        - task_id: current task id
        - CL_dataset: continual learning dataset
        - test_loader_list: test loaders of each task
        - model: model to evaluate
        - tokenizer: tokenizer of the model
        - accelerator: accelerate.Accelerator

    Returns:
        - gmm_result_dict: clustering accuracy, ARI and NMI per task (and overall for CIL)
    """
    il_mode = params.il_mode
    assert il_mode in ['CIL', 'TIL'], 'NotImplemented for il_mode %s' % (il_mode)

    total_num_task = CL_dataset.continual_config['NUM_TASK']
    cur_num_class_list = CL_dataset.continual_config['CUR_NUM_CLASS']

    if il_mode == 'CIL':
        # CIL: pool the data of all tasks
        num_task = len(test_loader_list)
        cur_num_class = cur_num_class_list  # per-task class counts
        num_class = sum(cur_num_class)
    else:
        # TIL: evaluate each task separately
        num_task = len(test_loader_list)
        cur_num_class = cur_num_class_list[:num_task]
        num_class = cur_num_class  # per-task class counts

    # Unwrap the backbone
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, WrapModel):
        model = model.model
    if hasattr(model, 'backbone_model'):
        model = model.backbone_model

    # Step 1: extract features
    with torch.no_grad():
        test_feature_list = []
        test_label_idx_list = []
        for t_id in range(num_task):
            _test_feature_list = []
            _test_label_idx_list = []
            for lm_input in test_loader_list[t_id]:
                extracted_features = obtain_features(params=params,
                                                     model=model,
                                                     lm_input=lm_input,
                                                     tokenizer=tokenizer)
                if il_mode == 'CIL':
                    label_idx = lm_input['label_idx_cil']
                else:
                    label_idx = lm_input['label_idx_til']

                extracted_features, label_idx = accelerator.gather_for_metrics((extracted_features, label_idx))
                extracted_features = extracted_features.detach().cpu()
                label_idx = label_idx.cpu()

                # Word-level tasks: flatten tokens and keep labelled positions
                if extracted_features.dim() == 3:
                    batch_size, seq_length, feature_dim = extracted_features.shape
                    extracted_features = extracted_features.view(-1, feature_dim)
                    label_idx = label_idx.view(-1)
                    mask = label_idx != -100  # drop padding positions
                    extracted_features = extracted_features[mask]
                    label_idx = label_idx[mask]
                else:
                    # Sentence-level tasks need no reshaping
                    pass

                _test_feature_list.append(extracted_features)
                _test_label_idx_list.append(label_idx)

            _test_feature_list = torch.cat(_test_feature_list, dim=0)
            _test_label_idx_list = torch.cat(_test_label_idx_list, dim=0)

            test_feature_list.append(_test_feature_list)
            test_label_idx_list.append(_test_label_idx_list)

        if il_mode == 'CIL':
            test_features_all = torch.cat(test_feature_list, dim=0)
            test_label_idx_all = torch.cat(test_label_idx_list, dim=0)
        else:
            test_features_all = test_feature_list
            test_label_idx_all = test_label_idx_list

        del test_feature_list, test_label_idx_list
        torch.cuda.empty_cache()

    # GMM clustering and evaluation
    if accelerator.is_main_process:
        gmm_result_dict = {}
        if il_mode == 'CIL':
            features = test_features_all
            labels = test_label_idx_all.numpy()
            num_clusters = sum(cur_num_class)
            print(f"CIL_num_data: {len(labels)}")
            print(f"CIL_num_clusters: {num_clusters}")

            # Fit GMM
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            cluster_assignments = gmm.fit_predict(features.float().numpy())

            # Encode labels
            le_labels = LabelEncoder()
            le_clusters = LabelEncoder()
            labels_encoded = le_labels.fit_transform(labels)
            clusters_encoded = le_clusters.fit_transform(cluster_assignments)

            # Hungarian matching between clusters and labels
            contingency_matrix = metrics.cluster.contingency_matrix(labels_encoded, clusters_encoded)
            row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

            # Map back to the original label ids
            label_mapping = {}
            for row, col in zip(row_ind, col_ind):
                cluster_label = le_clusters.inverse_transform([col])[0]
                true_label = le_labels.inverse_transform([row])[0]
                label_mapping[cluster_label] = true_label

            # Unmatched clusters
            cluster_labels = np.unique(cluster_assignments)
            missing_labels = set(cluster_labels) - set(label_mapping.keys())
            for missing_label in missing_labels:
                label_mapping[missing_label] = -1

            # Apply the mapping
            mapped_clusters = np.array([label_mapping.get(c, -1) for c in cluster_assignments])

            # Metrics over samples with a matched cluster
            valid_indices = mapped_clusters != -1
            accuracy = np.mean(mapped_clusters[valid_indices] == labels[valid_indices]) * 100
            accuracy = np.round(accuracy, 3)
            ari = metrics.adjusted_rand_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
            ari = np.round(ari, 4)
            nmi = metrics.normalized_mutual_info_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
            nmi = np.round(nmi, 4)

            gmm_result_dict['Overall_Accuracy'] = accuracy
            gmm_result_dict['Overall_ARI'] = ari
            gmm_result_dict['Overall_NMI'] = nmi
            print(gmm_result_dict)

            # Class-index range of each task
            class_ranges = []
            start_idx = 0
            for num_classes in cur_num_class_list:
                end_idx = start_idx + num_classes - 1
                class_ranges.append((start_idx, end_idx))
                start_idx += num_classes

            # Per-task accuracy, ARI and NMI
            for t_id in range(total_num_task):
                class_start, class_end = class_ranges[t_id]
                task_class_labels = np.arange(class_start, class_end + 1)
                task_mask = np.isin(labels, task_class_labels)

                num_samples = task_mask.sum()
                if num_samples == 0:
                    task_accuracy = -1
                    task_ari = -1
                    task_nmi = -1
                else:
                    task_labels = labels[task_mask]
                    task_mapped_clusters = mapped_clusters[task_mask]
                    valid_indices = task_mapped_clusters != -1
                    task_labels = task_labels[valid_indices]
                    task_mapped_clusters = task_mapped_clusters[valid_indices]
                    if len(task_labels) == 0:
                        task_accuracy = -1
                        task_ari = -1
                        task_nmi = -1
                    else:
                        task_accuracy = np.mean(task_mapped_clusters == task_labels) * 100
                        task_accuracy = np.round(task_accuracy, 3)
                        task_ari = metrics.adjusted_rand_score(task_labels, task_mapped_clusters) * 100
                        task_ari = np.round(task_ari, 4)
                        task_nmi = metrics.normalized_mutual_info_score(task_labels, task_mapped_clusters) * 100
                        task_nmi = np.round(task_nmi, 4)
                gmm_result_dict[f'Task_{t_id}_Accuracy'] = task_accuracy
                gmm_result_dict[f'Task_{t_id}_ARI'] = task_ari
                gmm_result_dict[f'Task_{t_id}_NMI'] = task_nmi
                print(gmm_result_dict)
        else:
            # TIL: cluster each task separately
            for t_id in range(total_num_task):
                if t_id >= num_task:
                    gmm_result_dict[f'Task_{t_id}_Accuracy'] = -1
                    gmm_result_dict[f'Task_{t_id}_ARI'] = -1
                    gmm_result_dict[f'Task_{t_id}_NMI'] = -1
                    continue

                features = test_features_all[t_id]
                labels = test_label_idx_all[t_id].numpy()
                num_clusters = cur_num_class[t_id]
                print(f"TIL_num_clusters: {num_clusters}")

                # Fit clustering
                gmm = GaussianMixture(n_components=num_clusters, random_state=42)
                cluster_assignments = gmm.fit_predict(features.float().numpy())

                # Encode labels
                le_labels = LabelEncoder()
                le_clusters = LabelEncoder()
                labels_encoded = le_labels.fit_transform(labels)
                clusters_encoded = le_clusters.fit_transform(cluster_assignments)

                # Hungarian matching between clusters and labels
                contingency_matrix = metrics.cluster.contingency_matrix(labels_encoded, clusters_encoded)
                row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

                # Map back to the original label ids
                label_mapping = {}
                for row, col in zip(row_ind, col_ind):
                    cluster_label = le_clusters.inverse_transform([col])[0]
                    true_label = le_labels.inverse_transform([row])[0]
                    label_mapping[cluster_label] = true_label

                # Unmatched clusters
                cluster_labels = np.unique(cluster_assignments)
                missing_labels = set(cluster_labels) - set(label_mapping.keys())
                for missing_label in missing_labels:
                    label_mapping[missing_label] = -1

                # Apply the mapping
                mapped_clusters = np.array([label_mapping.get(c, -1) for c in cluster_assignments])

                # Metrics over samples with a matched cluster
                valid_indices = mapped_clusters != -1
                accuracy = np.mean(mapped_clusters[valid_indices] == labels[valid_indices]) * 100
                accuracy = np.round(accuracy, 3)
                ari = metrics.adjusted_rand_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
                ari = np.round(ari, 4)
                nmi = metrics.normalized_mutual_info_score(labels[valid_indices], mapped_clusters[valid_indices]) * 100
                nmi = np.round(nmi, 4)

                gmm_result_dict[f'Task_{t_id}_Accuracy'] = accuracy
                gmm_result_dict[f'Task_{t_id}_ARI'] = ari
                gmm_result_dict[f'Task_{t_id}_NMI'] = nmi

    else:
        # Non-main processes return placeholder results
        num_tasks_for_results = total_num_task
        gmm_result_dict = {}
        for t_id in range(num_tasks_for_results):
            gmm_result_dict[f'Task_{t_id}_Accuracy'] = -1
            gmm_result_dict[f'Task_{t_id}_ARI'] = -1
            gmm_result_dict[f'Task_{t_id}_NMI'] = -1
        if il_mode == 'CIL':
            gmm_result_dict['Overall_Accuracy'] = -1
            gmm_result_dict['Overall_ARI'] = -1
            gmm_result_dict['Overall_NMI'] = -1

    return gmm_result_dict
