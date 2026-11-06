# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Post-hoc Just LM-Head Tuning (JLT), deployable variant.

Task-adaptive JLT re-fits the LM head on the full training split of the task
being evaluated. This module implements a variant that can be used as a method:

  * after training task k, freeze the backbone;
  * re-align only the LM head, once per checkpoint, on a small replay buffer
    drawn from the training loaders of the seen tasks 0..k (including the
    current task k). Per task, the buffer takes `jlt_posthoc_per_class`
    (default 1) x the number of classes examples from the shuffled training
    loader, rounded up to whole batches;
  * evaluate all tasks with this single head;
  * restore the original head afterwards, so the incremental-learning
    trajectory is unchanged.

Future-task data is never used, and the full training split of an evaluated
task is never used (unlike task-adaptive JLT). The resulting A_T / BWT / FWT
are therefore standard CIL numbers.
"""

import logging
from copy import deepcopy

import torch

from utils.wrapmodel import WrapModel

logger = logging.getLogger()


def _backbone(learner):
    m = getattr(learner, 'wrap_model', None)
    if m is None:
        m = learner.model
    if hasattr(m, 'module'):
        m = m.module
    if isinstance(m, WrapModel):
        m = m.model
    if hasattr(m, 'backbone_model'):
        m = m.backbone_model
    return m


def _collect_buffer(learner, task_id, per_class):
    """Leading batches of each seen task's train loader, until per_class * #classes samples are reached."""
    cont = learner.CL_dataset.continual_config
    buf = []
    for t in range(task_id + 1):
        budget = int(per_class * cont['CUR_NUM_CLASS'][t])
        taken = 0
        for lm_input in learner.train_loader_list[t]:
            buf.append({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in lm_input.items()})
            taken += len(lm_input['input_ids'])
            if taken >= budget:
                break
    return buf


def posthoc_align_head(learner, task_id):
    """Re-align the LM head on the replay buffer. Returns state for restoration."""
    params = learner.params
    model = _backbone(learner)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        return None

    orig_state = deepcopy(lm_head.state_dict())
    orig_rg = {n: p.requires_grad for n, p in model.named_parameters()}

    buf = _collect_buffer(learner, task_id, getattr(params, 'jlt_posthoc_per_class', 1))
    n_samples = sum(len(b['input_ids']) for b in buf)

    for p in model.parameters():
        p.requires_grad = False
    for p in lm_head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW([p for p in lm_head.parameters() if p.requires_grad],
                            lr=float(getattr(params, 'lm_head_lr', 5e-4)))
    dev = next(model.parameters()).device
    model.eval()

    epochs = int(getattr(params, 'lm_head_epochs', 3))
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for b in buf:
            ids = b['input_ids_with_ans'].to(dev)
            att = b['attention_mask_with_ans'].to(dev)
            lab = b['labels_with_ans'].to(dev)

            # The backbone is frozen: compute hidden states without grad so that
            # only the head receives gradients. Restricting the loss to answer-token
            # positions keeps the vocabulary projection small in memory.
            with torch.no_grad():
                hs = model(input_ids=ids, attention_mask=att,
                           output_hidden_states=True).hidden_states[-1]
            tgt = lab[:, 1:]
            h = hs[:, :-1, :]
            sel = tgt != -100
            if sel.sum() == 0:
                continue
            feats = h[sel].to(lm_head.weight.dtype)
            targets = tgt[sel]

            logits = lm_head(feats)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
            del hs, h, feats, logits, loss
        if learner.accelerator.is_main_process:
            logger.info('[posthoc-JLT] task %d epoch %d/%d loss %.4f (buffer %d samples, %d batches)'
                        % (task_id, ep + 1, epochs, tot / max(nb, 1), n_samples, len(buf)))

    for n, p in model.named_parameters():
        p.requires_grad = orig_rg.get(n, True)
    torch.cuda.empty_cache()
    return (lm_head, orig_state)


def posthoc_restore_head(state):
    if state is None:
        return
    lm_head, orig_state = state
    lm_head.load_state_dict(orig_state)
    torch.cuda.empty_cache()
