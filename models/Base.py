# Derived from the incremental learning codebase of Zheng et al. (ACL 2024),
# https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm
# Modifications copyright (c) 2026 Keuntae Kim.
# See NOTICE for details.

import os
import numpy as np
import pandas as pd
import logging
import torch
from abc import abstractmethod
from copy import deepcopy
from accelerate.utils import gather_object

from utils.evaluation import compute_backward_transfer, compute_forward_transfer, compute_average_acc, compute_average_inc_acc, compute_forgetting
from utils.probing import probing_on_all_task, save_all_features_labels
from utils.kmeans import evaluate_kmeans_on_all_task
from utils.gmm import evaluate_gmm_on_all_task
from utils.spectral import evaluate_spectral_clustering_on_all_task
from utils.agglomerative import evaluate_agglomerative_clustering_on_all_task
from utils.deep_clustering import evaluate_deep_clustering_on_all_task
from utils.featdump import dump_test_features
from utils.posthoc_jlt import posthoc_align_head, posthoc_restore_head

logger = logging.getLogger()

# Online clustering diagnostics: name -> (function, log prefix, summary attribute, display name)
CLUSTERING_ALGOS = {
    'spectral': (evaluate_spectral_clustering_on_all_task, 'Spectral', 'spectral_result_summary', 'Spectral Clustering'),
    'kmeans': (evaluate_kmeans_on_all_task, 'KMeans', 'kmeans_result_summary', 'KMeans'),
    'gmm': (evaluate_gmm_on_all_task, 'GMM', 'gmm_result_summary', 'GMM'),
    'agglomerative': (evaluate_agglomerative_clustering_on_all_task, 'Agglomerative', 'agglomerative_result_summary', 'Agglomerative Clustering'),
    'deep': (evaluate_deep_clustering_on_all_task, 'DeepClustering', 'deep_clustering_result_summary', 'Deep Clustering'),
}


def get_online_clustering_algos(params):
    """Parse --online_clustering, e.g. 'spectral' or 'spectral,kmeans,gmm,agglomerative,deep'."""
    names = [n.strip() for n in str(getattr(params, 'online_clustering', 'spectral')).split(',') if n.strip()]
    if names == ['all']:
        return list(CLUSTERING_ALGOS)
    for n in names:
        assert n in CLUSTERING_ALGOS, 'Unknown clustering algorithm %s' % n
    return names

class BaseLearner(object):
    def __init__(self, params, CL_dataset, accelerator):
        # parameters
        self.params = params
        self.best_model_ckpt_name = None
        self.global_step = 0
        self.CL_dataset = CL_dataset
        self.accelerator = accelerator

        # initialization
        self.model = None
        self.classifier = None
        self.optimizer = None
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = [], [], []
        self.build_metric()
        self.build_backbone()
        self.build_classifier()
        self.build_optimizer()
        self.build_dataloader()
        self.build_buffer()
        self.accelerate_prepare()

    # ================================= Prepare model, data and optimizer =======================================
    @abstractmethod
    def build_metric(self):
        pass

    @abstractmethod
    def build_backbone(self):
        pass

    @abstractmethod
    def build_classifier(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass
        
    @abstractmethod
    def build_dataloader(self):
        pass

    @abstractmethod
    def build_buffer(self):
        pass

    @abstractmethod
    def accelerate_prepare(self):
        pass
    # ==============================================================================================

    # ================================= Task-Level Functions =======================================
    def incremental_training(self):

        num_task = self.CL_dataset.continual_config['NUM_TASK']

        # Learn Tasks Incrementally
        for task_id in range(num_task):
            if self.accelerator.is_main_process: 
                logger.info("============================================================================")   
                logger.info("Beggin training the task %d (total %d tasks)" % (task_id + 1, num_task))     
                logger.info("============================================================================")
            self.begin_task(task_id)
            self.train_epochs(task_id)
            self.end_task(task_id)  

    def begin_task(self, task_id):
        
        self.best_score = -1
        self.step = 0

        # save features
        if task_id == 0 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'BeforeIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)
        
    def end_task(self, task_id):

        # testing
        if self.accelerator.is_main_process:
            logger.info("Testing...")

        result_dict = self.evaluate_model(task_id=task_id)
        il_mode = self.params.il_mode

        # Update current task result
        for t_id, _acc in enumerate(result_dict['Test_Acc_List']):
            self.result_summary.update(task_id, t_id, _acc)

        # Update next task result if available
        if 'Next_Acc_List' in result_dict:
            for t_id, _acc_next in enumerate(result_dict['Next_Acc_List']):
                self.result_summary.update(task_id, t_id + 1, _acc_next)

        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Result Summary Test After Task %d = \n%s' % (il_mode, 
                                                                                 task_id,
                                                                                 self.result_summary.print_format()))

        # Save checkpoint
        if self.params.save_ckpt:
            torch.save(
                {
                    'model': deepcopy(self.model).cpu(),
                    'classifier_list': deepcopy(self.classifier_list).cpu(),
                }
                , os.path.join(self.params.dump_path, 'last_ckpt_task%d.pth' % (task_id)))

        # Save features
        num_task = self.CL_dataset.continual_config['NUM_TASK']
        if task_id == num_task - 1 and self.params.save_features_before_after_IL:
            save_all_features_labels(self.params, 
                                     'AfterIL',
                                     self.train_loader_list, 
                                     self.model, 
                                     self.tokenizer, 
                                     self.accelerator)

        # Save GPU memory
        torch.cuda.empty_cache()
    # ===========================================================================================


    # ================== Evaluation, Logging, Saving and Loading Functions ======================
    def evaluate_model(self, task_id: int) -> dict:
        '''
        Evaluate the model and log the result.

        Args:
            - task_id: the id of the task the model has just learned

        Return:
            - result_dict: dict with Test_Acc_List and, if available, Next_Acc_List
        '''
        result_dict = {}
        log_dict = {}

        cur_task_id = task_id
        il_mode = self.params.il_mode

        # Deployable (non-oracle) post-hoc LM-head re-alignment, applied once before evaluation
        _posthoc_state = None
        if getattr(self.params, 'jlt_posthoc', False):
            _posthoc_state = posthoc_align_head(self, task_id)

        # Evaluate the current task and all previously learned tasks
        try:
            acc_list, acc_next_list = self.evaluate_all_seen_task_tc(cur_task_id, 'test', il_mode)
        finally:
            posthoc_restore_head(_posthoc_state)
        result_dict['Test_Acc_List'] = acc_list

        # Log per-task accuracy
        for t_id in range(cur_task_id + 1):
            log_dict['Test_Acc_Task_%d' % (t_id)] = acc_list[t_id]

        # Update the result summary for the current task
        logger.info(f'Updating task {cur_task_id}, eval_task_id {cur_task_id}, with value {acc_list[cur_task_id]}')
        self.result_summary.update(cur_task_id, cur_task_id, acc_list[cur_task_id])

        # Also record the accuracy on the next (not yet learned) task, if available
        if acc_next_list is not None and cur_task_id + 1 < len(self.result_summary.result_summary):
            if acc_next_list[cur_task_id] is not None:
                logger.info(f'Updating next task: task {cur_task_id}, eval_task_id {cur_task_id + 1}, with value {acc_next_list[cur_task_id]}')
                self.result_summary.update(cur_task_id, cur_task_id + 1, acc_next_list[cur_task_id])

        # Log results
        log_dict['Test_Acc_Task_Seen'] = np.round(np.mean(acc_list[:cur_task_id + 1]), 3)
        if self.params.classifier == 'None':
            log_dict['Test_Acc_Task_All'] = np.round(np.mean(acc_list), 3)

        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Test Result = %s' % (il_mode, log_dict))
            logger.info(f'Result Summary Test After Task {cur_task_id} =\n{self.result_summary.print_format()}')
        self.accelerator.log(log_dict, step=self.global_step)

        # Dump backbone test features for the offline diagnostic grid
        if getattr(self.params, 'dump_task_features', False):
            dump_test_features(
                params=self.params,
                task_id=task_id,
                CL_dataset=self.CL_dataset,
                test_loader_list=self.test_loader_list,
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator
            )

        # Classifier-free clustering diagnostics of the backbone
        total_num_task = self.CL_dataset.continual_config['NUM_TASK']
        for algo in get_online_clustering_algos(self.params):
            evaluate_fn, prefix, summary_name, display_name = CLUSTERING_ALGOS[algo]
            if getattr(self.params, 'skip_online_clustering', False):
                # Skipped when diagnostics are computed offline from dumped features
                cluster_result_dict = {}
            else:
                cluster_result_dict = evaluate_fn(
                    params=self.params,
                    task_id=task_id,
                    CL_dataset=self.CL_dataset,
                    test_loader_list=self.test_loader_list,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator
                )
            self._log_clustering_result(task_id, total_num_task, cluster_result_dict,
                                        result_dict, prefix, summary_name, display_name)

        # Probing: evaluate probing accuracy on all tasks and log LinearProb results
        if self.params.is_probing:
            total_num_task = len(self.train_loader_list)

            if self.params.il_mode == 'CIL':
                # CIL: use the data loaders of all tasks
                probing_train_loader_list = self.train_loader_list
                probing_test_loader_list = self.test_loader_list
                num_task = total_num_task
            else:
                # TIL: use the data loaders up to the next task
                num_task = task_id + 2  # current task plus the next one
                num_task = min(num_task, total_num_task)

                probing_train_loader_list = self.train_loader_list[:num_task]
                probing_test_loader_list = self.test_loader_list[:num_task]

            prob_acc_dict = probing_on_all_task(
                params=self.params,
                task_id=task_id,
                CL_dataset=self.CL_dataset,
                train_loader_list=probing_train_loader_list,
                test_loader_list=probing_test_loader_list,
                model=self.model,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator
            )

            # Only LinearProb results are logged and tracked
            if 'LinearProb' in prob_acc_dict:
                prob_acc_list = prob_acc_dict['LinearProb']
                result_dict['LinearProb_List'] = prob_acc_list
                prob_log_dict = {
                    'LinearProb_Acc_Task_Seen': np.round(np.mean(prob_acc_list[:task_id + 1]), 3),
                    'LinearProb_Acc_Task_All': np.round(np.mean(prob_acc_list[:total_num_task]), 3)
                }

                # Update the probing summary for each task
                for t_id in range(total_num_task):
                    acc = prob_acc_list[t_id]
                    if acc != -1:
                        logger.info(f'Updating probing result for LinearProb: task {task_id}, eval_task_id {t_id}, with value {acc}')
                        self.probing_result_summary.update(task_id, t_id, acc)

                # Print the probing summary
                if self.accelerator.is_main_process:
                    logger.info(f'Result Summary Probing After Task {task_id} (LinearProb) =\n{self.probing_result_summary.print_format()}')
                    logger.info('Probing Result (LinearProb) = %s' % (prob_log_dict))

                self.accelerator.log(prob_log_dict, step=self.global_step)

        return result_dict

    def _log_clustering_result(self, task_id, total_num_task, cluster_result_dict,
                               result_dict, prefix, summary_name, display_name):
        """Store, log and summarize the per-task scores of one clustering algorithm."""
        acc_list, ari_list, nmi_list = [], [], []
        for t_id in range(total_num_task):
            acc_list.append(cluster_result_dict.get(f'Task_{t_id}_Accuracy', -1))
            ari_list.append(cluster_result_dict.get(f'Task_{t_id}_ARI', -1))
            nmi_list.append(cluster_result_dict.get(f'Task_{t_id}_NMI', -1))

        result_dict[f'{prefix}_Acc_List'] = acc_list
        result_dict[f'{prefix}_ARI_List'] = ari_list
        result_dict[f'{prefix}_NMI_List'] = nmi_list

        # Mean over valid entries (-1 marks a missing result)
        def compute_mean(values):
            valid_values = [v for v in values if v != -1]
            return np.round(np.mean(valid_values), 4) if valid_values else -1

        log_dict = {
            f'{prefix}_Acc_Task_Seen': compute_mean(acc_list[:task_id + 1]),
            f'{prefix}_Acc_Task_All': compute_mean(acc_list),
            f'{prefix}_ARI_Task_Seen': compute_mean(ari_list[:task_id + 1]),
            f'{prefix}_ARI_Task_All': compute_mean(ari_list),
            f'{prefix}_NMI_Task_Seen': compute_mean(nmi_list[:task_id + 1]),
            f'{prefix}_NMI_Task_All': compute_mean(nmi_list),
        }
        if self.params.il_mode == 'CIL':
            log_dict[f'{prefix}_Overall_Accuracy'] = cluster_result_dict.get('Overall_Accuracy', -1)
            log_dict[f'{prefix}_Overall_ARI'] = cluster_result_dict.get('Overall_ARI', -1)
            log_dict[f'{prefix}_Overall_NMI'] = cluster_result_dict.get('Overall_NMI', -1)

        summary = getattr(self, summary_name, None)
        for t_id in range(total_num_task):
            acc, ari, nmi = acc_list[t_id], ari_list[t_id], nmi_list[t_id]
            if acc != -1 and summary is not None:
                logger.info(f'Updating {display_name} result: task {task_id}, eval_task_id {t_id}, with Accuracy {acc}, ARI {ari}, NMI {nmi}')
                summary.update(task_id, t_id, Accuracy=acc, ARI=ari, NMI=nmi)

        if self.accelerator.is_main_process and summary is not None:
            logger.info(f'Result Summary {display_name} After Task {task_id} =\n{summary.print_format()}')
            logger.info(f'{display_name} Result = %s' % (log_dict))

        self.accelerator.log(log_dict, step=self.global_step)

    def evaluate_all_seen_task_tc(self, cur_task_id: int, phase: str, il_mode: str) -> tuple:
        '''
        Evaluate the model on all seen tasks, including current and next tasks

        Params:
            - cur_task_id: the ID of the current task
            - phase: 'train', 'dev', or 'test'
            - il_mode: 'CIL' or 'TIL'

        Return:
            - acc_list: list of accuracies for all tasks evaluated
            - acc_next_task_list: list of accuracies for the next task (if exists)
        '''
        assert phase in ['train', 'test', 'dev']

        acc_list = []
        acc_next_task_list = None

        # Evaluate on all seen tasks
        save_dict_all = None

        for eval_t_id in range(cur_task_id + 1):
            # Evaluate current task
            if self.params.classification_type == 'sentence-level':
                acc, acc_next_task = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
            elif self.params.classification_type == 'word-level':
                acc, acc_next_task = self.evaluate_current_task(eval_t_id, cur_task_id, phase, il_mode)
            else:
                raise ValueError(f"Unsupported classification type: {self.params.classification_type}")

            acc_list.append(acc)

            # If next task exists, store its result
            if acc_next_task is not None:
                if acc_next_task_list is None:
                    acc_next_task_list = []
                acc_next_task_list.append(acc_next_task)

        save_dict_all = [save_dict_all]  # transform to list for gather_object() to collect correctly
        gathered_save_dict_all = gather_object(save_dict_all)

        if self.accelerator.is_main_process:
            if gathered_save_dict_all is not None:
                with open(os.path.join(self.params.dump_path, f'{phase}_cur_task_{cur_task_id}_save_result.npy'), 'wb') as f:
                    np.save(f, gathered_save_dict_all)

        return acc_list, acc_next_task_list

    @abstractmethod
    def evaluate_current_task(self, eval_task_id: int, cur_task_id: int, phase: str, il_mode: str) -> tuple:
        '''
        Evaluate the model on the current task

        Params:
            - eval_task_id: the ID of the task to be evaluated
            - cur_task_id: the ID of the current task
            - phase: 'train', 'dev', or 'test'
            - il_mode: 'CIL' or 'TIL'

        Return:
            - acc: accuracy for the evaluated task
            - acc_next_task: accuracy for the next task (if exists), otherwise None
        '''
        pass
    # ===========================================================================================


    # ================================= Epoch-Level Functions ====================================
    @abstractmethod
    def train_epochs(self, task_id):
        pass
    # ===========================================================================================
    
    def finish_training(self):
        '''
            Finish training: print the result
        '''
        log_dict = {}
        il_mode = self.params.il_mode
        num_tasks = self.CL_dataset.continual_config['NUM_TASK']
        if self.accelerator.is_main_process:
            logger.info('Mode = %s, Summary Test Acc = \n%s' % (il_mode, self.result_summary.print_format()))
            result_dict = self.result_summary.get_value()
            result_matrix = result_dict  # numpy array
            bwt_acc = compute_backward_transfer(result_matrix)
            fwt_acc = compute_forward_transfer(result_matrix)
            fgt_acc = compute_forgetting(result_matrix)
            aver_acc = compute_average_acc(result_matrix)
            aver_inc_acc = compute_average_inc_acc(result_matrix)
            log_dict['Test_Aver_ACC'] = aver_acc
            log_dict['Test_Bwt_ACC'] = bwt_acc
            log_dict['Test_Fwt_ACC'] = fwt_acc
            log_dict['Test_Fgt_ACC'] = fgt_acc
            log_dict['Test_Aver_Inc_ACC'] = aver_inc_acc
            logger.info('Mode = %s, Summary Result = \n%s'%(il_mode,log_dict))    

        # Summaries for the clustering results
        clustering_methods = ['kmeans', 'gmm', 'spectral', 'agglomerative', 'deep_clustering']
        metrics_list = ['Accuracy', 'ARI', 'NMI']
        for method in clustering_methods:
            result_summary_attr = f'{method}_result_summary'
            if hasattr(self, result_summary_attr):
                result_summary = getattr(self, result_summary_attr)
                if self.accelerator.is_main_process:
                    logger.info(f'Mode = {il_mode}, Summary {method.capitalize()} Results:')
                    # Result matrix for each metric
                    for metric in metrics_list:
                        result_matrix = result_summary.get_metric_matrix(metric)
                        if result_matrix is None:
                            continue  # no results for this metric

                        # Compute metrics
                        method_bwt = compute_backward_transfer(result_matrix)
                        method_fwt = compute_forward_transfer(result_matrix)
                        method_fgt = compute_forgetting(result_matrix)
                        method_aver = compute_average_acc(result_matrix)
                        method_aver_inc = compute_average_inc_acc(result_matrix)

                        method_log_dict = {}
                        method_log_dict[f'{method.capitalize()}_{metric}_Aver'] = method_aver
                        method_log_dict[f'{method.capitalize()}_{metric}_Bwt'] = method_bwt
                        method_log_dict[f'{method.capitalize()}_{metric}_Fwt'] = method_fwt
                        method_log_dict[f'{method.capitalize()}_{metric}_Fgt'] = method_fgt
                        method_log_dict[f'{method.capitalize()}_{metric}_Aver_Inc'] = method_aver_inc

                        logger.info(f'{method.capitalize()} {metric} Results: {method_log_dict}')
                        self.accelerator.log(method_log_dict, step=self.global_step)
                        print(f'{method.capitalize()} {metric} Result Matrix:')
                        print(result_matrix)

        # Probing summary
        if hasattr(self, 'probing_result_summary'):
            probing_log_dict = {}
            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Summary Probing Acc = \n%s' % (il_mode, self.probing_result_summary.print_format()))
            # Compute Forward and Backward Transfer according to Probing Result Summary for the whole learning process
            probing_bwt_acc = compute_backward_transfer(self.probing_result_summary.get_value())
            probing_fwt_acc = compute_forward_transfer(self.probing_result_summary.get_value()) 
            probing_fgt_acc = compute_forgetting(self.probing_result_summary.get_value()) 
            probing_aver_acc = compute_average_acc(self.probing_result_summary.get_value()) 
            probing_aver_inc_acc = compute_average_inc_acc(self.probing_result_summary.get_value()) 
            probing_log_dict['Probing_Aver_ACC'] = probing_aver_acc
            probing_log_dict['Probing_Bwt_ACC'] = probing_bwt_acc
            probing_log_dict['Probing_Fwt_ACC'] = probing_fwt_acc
            probing_log_dict['Probing_Fgt_ACC'] = probing_fgt_acc
            probing_log_dict['Probing_Aver_Inc_ACC'] = probing_aver_inc_acc

            if self.accelerator.is_main_process:
                logger.info('Mode = %s, Summary Probing Result = \n%s' % (il_mode, probing_log_dict))    
            self.accelerator.log(probing_log_dict, step=self.global_step)
            print(self.probing_result_summary.get_value())
        
        # End Wandb
        self.accelerator.end_training()

