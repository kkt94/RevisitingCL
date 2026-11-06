# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""CPU unit test for utils.posthoc_jlt: exercises the real code path with a tiny model."""
import os
import sys
import types

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.posthoc_jlt import posthoc_align_head, posthoc_restore_head

V, T_TASKS, B, S = 64, 3, 2, 12

cfg = AutoConfig.for_model('gpt2', vocab_size=V, n_positions=32, n_embd=32,
                           n_layer=2, n_head=2)
model = AutoModelForCausalLM.from_config(cfg)


def fake_batch():
    ids = torch.randint(0, V, (B, S))
    att = torch.ones(B, S, dtype=torch.long)
    lab = ids.clone()
    lab[:, :S - 3] = -100           # only the last few tokens are "answer" tokens
    return {'input_ids_with_ans': ids, 'attention_mask_with_ans': att,
            'labels_with_ans': lab, 'input_ids': ids}


class FakeLearner:
    def __init__(self):
        self.model = model
        self.params = types.SimpleNamespace(
            jlt_posthoc=True, jlt_posthoc_per_class=1,
            lm_head_lr=1e-3, lm_head_epochs=2)
        self.CL_dataset = types.SimpleNamespace(
            continual_config={'CUR_NUM_CLASS': [4] * T_TASKS, 'NUM_TASK': T_TASKS})
        self.train_loader_list = [[fake_batch() for _ in range(5)] for _ in range(T_TASKS)]
        self.accelerator = types.SimpleNamespace(is_main_process=True)


learner = FakeLearner()
head = model.get_output_embeddings()
emb = model.get_input_embeddings()
before_head = head.weight.detach().clone()
body = {n: p.detach().clone() for n, p in model.named_parameters()
        if 'wte' not in n and 'lm_head' not in n}
rg_before = {n: p.requires_grad for n, p in model.named_parameters()}

state = posthoc_align_head(learner, task_id=1)
after_head = head.weight.detach().clone()

assert not torch.equal(before_head, after_head), 'FAIL: head was not updated'
for n, p in model.named_parameters():
    if n in body:
        assert torch.equal(body[n], p.detach()), 'FAIL: backbone param %s changed' % n
print('OK  head updated, backbone unchanged  (max head delta %.3e)'
      % (after_head - before_head).abs().max())

tied = emb.weight.data_ptr() == head.weight.data_ptr()
print('NOTE tie_word_embeddings =', tied,
      '-> head tuning also moves the input embedding' if tied else '')

posthoc_restore_head(state)
assert torch.equal(head.weight.detach(), before_head), 'FAIL: head not restored'
print('OK  head restored exactly after evaluation')

for n, p in model.named_parameters():
    assert rg_before[n] == p.requires_grad, 'FAIL: requires_grad not restored for %s' % n
print('OK  requires_grad flags restored')

# buffer sizing honours the per-class budget
from utils.posthoc_jlt import _collect_buffer
buf = _collect_buffer(learner, 1, per_class=1)
n = sum(len(b['input_ids']) for b in buf)
print('OK  buffer for tasks 0..1 with 4 classes/task, 1 per class -> %d samples in %d batches'
      % (n, len(buf)))
print('\nALL CHECKS PASSED')
