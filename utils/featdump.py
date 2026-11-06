# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Per-task backbone feature dumping for offline diagnostics.

At every task boundary, the backbone features of the test split of every task
(seen, current and future) are extracted and saved. Future-task features are
needed for the forward-transfer term a_{i-1,i}. All offline diagnostics (e.g.
other clustering configurations, feature normalizations, or clustering-free
measures) are computed from these dumps, so they all see identical
representations without re-running incremental learning.

Layout
------
{dump_path}/featdump/after_task{K}.npz
    feat_t{j} : float16 [N_j, d]   backbone features of test task j after task K
    lab_t{j}  : int64   [N_j]      CIL label indices of test task j
    meta      : json string (backbone, method, dataset, task counts, ...)
"""

import os
import json
import logging

import numpy as np
import torch

from utils.wrapmodel import WrapModel
from utils.backbone import obtain_features

logger = logging.getLogger()


def _unwrap(model):
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, WrapModel):
        model = model.model
    if hasattr(model, 'backbone_model'):
        model = model.backbone_model
    return model


def dump_test_features(params, task_id, CL_dataset, test_loader_list,
                       model, tokenizer, accelerator) -> None:
    """Extract and persist test features of all tasks at checkpoint `task_id`."""

    max_per_task = int(getattr(params, 'dump_feat_max_per_task', 3000))
    out_dir = os.path.join(params.dump_path, 'featdump')
    out_path = os.path.join(out_dir, 'after_task%d.npz' % task_id)

    model_ = _unwrap(model)
    num_task = len(test_loader_list)

    arrays = {}
    with torch.no_grad():
        for t_id in range(num_task):
            feats, labs = [], []
            n_seen = 0
            for lm_input in test_loader_list[t_id]:
                extracted = obtain_features(params=params,
                                            model=model_,
                                            lm_input=lm_input,
                                            tokenizer=tokenizer)
                label_idx = lm_input['label_idx_cil'] if params.il_mode == 'CIL' \
                    else lm_input['label_idx_til']

                extracted, label_idx = accelerator.gather_for_metrics((extracted, label_idx))
                extracted = extracted.detach().cpu()
                label_idx = label_idx.cpu()

                if extracted.dim() == 3:  # word-level task: flatten and drop padding
                    b, s, d = extracted.shape
                    extracted = extracted.view(-1, d)
                    label_idx = label_idx.view(-1)
                    keep = label_idx != -100
                    extracted, label_idx = extracted[keep], label_idx[keep]

                feats.append(extracted.float().numpy().astype(np.float16))
                labs.append(label_idx.numpy().astype(np.int64))
                n_seen += extracted.shape[0]
                # Deterministic prefix cap: test loaders are unshuffled, so the
                # same subset is dumped at every checkpoint of a given run.
                if max_per_task > 0 and n_seen >= max_per_task:
                    break

            if len(feats) == 0:
                continue
            f = np.concatenate(feats, axis=0)
            l = np.concatenate(labs, axis=0)
            if max_per_task > 0:
                f, l = f[:max_per_task], l[:max_per_task]
            arrays['feat_t%d' % t_id] = f
            arrays['lab_t%d' % t_id] = l

    if not accelerator.is_main_process:
        return

    os.makedirs(out_dir, exist_ok=True)
    cont = CL_dataset.continual_config
    meta = {
        'backbone': params.backbone,
        'method': params.method,
        'dataset': params.dataset,
        'il_mode': params.il_mode,
        'checkpoint_task_id': int(task_id),
        'num_task': int(cont['NUM_TASK']),
        'cur_num_class': [int(c) for c in cont['CUR_NUM_CLASS']],
        'seed': getattr(params, 'seed', None),
        'max_per_task': max_per_task,
        'sizes': {k: list(v.shape) for k, v in arrays.items() if k.startswith('feat_')},
    }
    np.savez_compressed(out_path, meta=json.dumps(meta), **arrays)
    logger.info('[featdump] wrote %s (%d eval tasks)' %
                (out_path, sum(1 for k in arrays if k.startswith('feat_'))))
