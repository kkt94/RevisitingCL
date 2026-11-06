# Derived from the incremental learning codebase of Zheng et al. (ACL 2024),
# https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm
# Modifications copyright (c) 2026 Keuntae Kim.
# See NOTICE for details.

import numpy as np
import logging
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from datasets import Dataset
from contextlib import nullcontext
import os

from utils.metric import ResultSummary, ResultSummary2
from utils.backbone import get_backbone, obtain_features
from utils.classifier import get_classifier
from utils.optimizer import get_optimizer
from utils.dataloader import get_dataloader
from utils.buffer import get_buffer
from utils.wrapmodel import WrapModel
from utils.evaluation import evaluate_sent_level_acc_with_generation, evaluate_sent_level_acc_with_classifier, evaluate_word_level_acc_with_classifier, evaluate_sent_level_acc_with_lmhead
from models.Base import BaseLearner
from torch.optim import AdamW

def log_memory_usage(stage=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logging.info(f"[Memory {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

logger = logging.getLogger()

def get_SEQ_params(parser):
    '''
        The parameters of model SEQ and WarmupAndFix
    '''
    parser.add_argument("--SEQ_peft_type", type=str, default='None', choices=['None','PromptTuning','LoRA'], help="The PEFT type")
    parser.add_argument("--SEQ_fix_old_classifier", default=False, type=bool, help="if fix the old classifiers")
    parser.add_argument("--SEQ_fix_encoder", default=False, type=bool, help="if fix the encoder")
    parser.add_argument("--SEQ_warmup_epoch_before_fix_encoder", type=int, default=0, help="The number of fine-tuning epoch before fixing the encoder (only for the first task). NOTE: Only used when SEQ_fix_encoder is True")
    parser.add_argument("--SEQ_warmup_target", type=str, default='causal-lm', choices=['causal-lm','ce-loss'], help="The training target during warmup epochs (only for the first task). NOTE: Only used when SEQ_fix_encoder is True")
    parser.add_argument("--SEQ_preallocated_classifier", type=bool, default=False, help="If pre-allocating classifiers for optimization?")
    parser.add_argument("--SEQ_use_prototype_for_prediction", type=bool, default=False, help="If using prototype as classifier weights")

    parser.add_argument("--PEFT_type", type=str, default='None', choices=['None','PromptTuning','LoRA'], help="The peft type")

    # For PromptTuning (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_num_virtual_tokens", type=int, default=10, help="The number of tokens for prompt tuning")
    parser.add_argument("--PEFT_prompt_tuning_init_text", type=str, default='auto', help="The initialization words for prompt tuning")
    # For LoRA (more details can be found in this (url)[https://huggingface.co/docs/peft]
    parser.add_argument("--PEFT_lora_r", type=int, default=8, help="The rank of lora")
    parser.add_argument("--PEFT_lora_alpha", type=int, default=16, help="The scaling of lora")
    parser.add_argument("--PEFT_lora_bias", type=str, default="none", help="The bias of lora")
    parser.add_argument("--PEFT_lora_dropout", type=float, default=0.05, help="The dropout rate of lora")
    parser.add_argument("--PEFT_lora_target_modules", type=str, nargs='+', default=None, help="The target_modules of lora")

    # LM head learning options
    parser.add_argument("--use_logit_matching", action='store_true')
    parser.add_argument("--target_token_logit", type=float, default=10.0)
    parser.add_argument("--other_token_logit", type=float, default=0.0)
    parser.add_argument("--logit_match_weight", type=float, default=0.5)
    parser.add_argument("--use_deep_supervision", action='store_true')
    parser.add_argument("--supervision_layers", type=int, nargs='+', default=[4,8,12])
    parser.add_argument("--ds_weight", type=float, default=0.2)
    parser.add_argument("--use_distillation", action='store_true')
    parser.add_argument("--kd_weight", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--use_ewc", action='store_true')
    parser.add_argument("--ewc_importance", type=float, default=0.1)


class SEQ(BaseLearner):
    '''
        Training a model sequentially to a number of tasks
    '''
    def __init__(self, params, CL_dataset, accelerator):
        # Map SEQ_peft_type to PEFT_type
        if hasattr(params, 'SEQ_peft_type') and params.SEQ_peft_type != 'None':
            params.PEFT_type = params.SEQ_peft_type
        elif not hasattr(params, 'PEFT_type'):
            params.PEFT_type = 'None'

        super().__init__(params, CL_dataset, accelerator)
        assert params.classifier in ['None','Linear','CosineLinear'], 'NotImplemented for classifier %s and model %s'%(params.classifier,'SEQ')
        assert params.il_mode in ['IIL','CIL','TIL'], 'NotImplemented for il mode %s and model %s'%(params.il_mode,'SEQ')
        assert params.classification_type in ['sentence-level','word-level'], 'NotImplemented for classification type %s'%(params.classification_type)
        if params.SEQ_use_prototype_for_prediction:
            assert params.classifier in ['Linear','CosineLinear']

    # ================================= Initialization =======================================
    def build_metric(self):
        self.result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.probing_result_summary = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.kmeans_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.gmm_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.spectral_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.deep_clustering_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])
        self.agglomerative_result_summary = ResultSummary2(num_task=self.CL_dataset.continual_config['NUM_TASK'])

        if self.params.il_mode == 'IIL':
            self.result_summary_train = ResultSummary(num_task=self.CL_dataset.continual_config['NUM_TASK'])

    def build_backbone(self):
        # Do not reload the backbone if it is already loaded
        if hasattr(self, 'model') and self.model is not None:
            log_memory_usage("Backbone Already Loaded")
            return

        num_task = self.CL_dataset.continual_config['NUM_TASK']
        log_memory_usage("Before Backbone Loading")
        self.model, self.tokenizer = get_backbone(self.params, num_task)
        log_memory_usage("After Backbone Loading")

    def build_classifier(self):
        feature_dim = self.model.module.config.hidden_size if hasattr(self.model,'module') else self.model.config.hidden_size
        out_dim_list = self.CL_dataset.continual_config['CUR_NUM_CLASS']
        self.classifier_list = get_classifier(self.params, feature_dim, out_dim_list)
        if self.classifier_list is not None:
            self.ce_loss = nn.CrossEntropyLoss()

    def build_optimizer(self):
        """
        Build the optimizer for the backbone and (optional) classifiers, and a
        separate LM-head optimizer when lm_head_finetune is set.
        """
        # 1) Backbone / classifier optimizer
        self.optimizer = get_optimizer(self.params, self.model, self.classifier_list)

        # 2) LM-head optimizer
        self.lm_head_optimizer = None
        if self.params.lm_head_finetune:
            lm_head = self.model.get_output_embeddings()
            # Explicitly enable LM-head gradients (LoRA freezes them otherwise)
            for param in lm_head.parameters():
                param.requires_grad = True
            self.lm_head_optimizer = torch.optim.AdamW(
                [lm_head.weight], lr=self.params.lm_head_lr
            )

        # 3) param_groups mapping
        num_task = self.CL_dataset.continual_config["NUM_TASK"]
        if self.params.classifier != "None" and not (
            self.params.SEQ_fix_encoder and self.params.SEQ_warmup_epoch_before_fix_encoder == 0
        ):
            assert len(self.optimizer.param_groups) == 1 + num_task
            self.param_groups_mapping = {
                "encoder": 0,
                "classifier": {i: 1 + i for i in range(num_task)},
            }
        elif self.params.classifier != "None":
            assert len(self.optimizer.param_groups) == num_task
            self.param_groups_mapping = {
                "classifier": {i: i for i in range(num_task)},
            }
        else:
            assert len(self.optimizer.param_groups) == 1
            self.param_groups_mapping = {"encoder": 0}

    def build_dataloader(self):
        self.train_loader_list, self.dev_loader_list, self.test_loader_list = get_dataloader(self.params, self.CL_dataset, self.tokenizer)

    def build_buffer(self):
        self.buffer = None
        if self.params.is_replay:
            self.buffer = get_buffer(self.params,self.CL_dataset.continual_config,self.accelerator)

    def accelerate_prepare(self):
        self.wrap_model = WrapModel(self.model, self.classifier_list)
        self.wrap_model, self.optimizer, *self.train_loader_list = self.accelerator.prepare(self.wrap_model, self.optimizer, *self.train_loader_list)
        if len(self.dev_loader_list)>1:
            self.dev_loader_list = self.accelerator.prepare(*self.dev_loader_list)
            self.test_loader_list = self.accelerator.prepare(*self.test_loader_list)
        else:
            self.dev_loader_list = [self.accelerator.prepare(*self.dev_loader_list)]
            self.test_loader_list = [self.accelerator.prepare(*self.test_loader_list)]

    # =============================================================================================

    # ================================= Task-Level Functions =======================================
    def begin_task(self, task_id):
        log_memory_usage(f"Task {task_id} Begin")

        # From the second task on, free memory left over from the previous task
        if task_id > 0:
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.optimizer.zero_grad()
            if hasattr(self, 'lm_head_optimizer') and self.lm_head_optimizer is not None:
                self.lm_head_optimizer.zero_grad()

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            log_memory_usage(f"Task {task_id} After Cleanup")

        super().begin_task(task_id)

        # Fix a component by setting its learning rate as zero
        if task_id>0 and self.params.classifier!='None' and self.params.SEQ_fix_old_classifier:
            self.optimizer.param_groups[self.param_groups_mapping['classifier'][task_id-1]]['lr']=0

    def end_task(self, task_id):
        log_memory_usage(f"Task {task_id} End Before Cleanup")

        super().end_task(task_id)

        if self.params.is_replay:
            # For Distributed Data Parallel
            if hasattr(self.wrap_model,'module'):
                wrap_model = self.wrap_model.module
            else:
                wrap_model = self.wrap_model
            model = wrap_model.model

            self.buffer.update_buffer(task_id, self.train_loader_list[task_id], model, self.tokenizer)

        self._cleanup_memory()
        log_memory_usage(f"Task {task_id} End After Cleanup")

    def _cleanup_memory(self):
        """Release gradients, optimizer state and cached GPU memory between tasks."""
        import gc

        # 1. Per-epoch loss lists
        if hasattr(self, 'backbone_loss_list'):
            del self.backbone_loss_list
        if hasattr(self, 'lm_head_loss_list'):
            del self.lm_head_loss_list
        if hasattr(self, 'loss_list'):
            del self.loss_list

        # 2. Optimizer gradients and state
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        del v
            self.optimizer.state.clear()

        # 3. LM-head optimizer gradients and state
        if hasattr(self, 'lm_head_optimizer') and self.lm_head_optimizer is not None:
            for group in self.lm_head_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()
            for state in self.lm_head_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        del v
            self.lm_head_optimizer.state.clear()

        # 4. LoRA adapter gradients
        if hasattr(self.model, 'peft_config'):
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

        # 5. Model gradients
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

        # 6. Classifier gradients
        if hasattr(self, 'classifier_list') and self.classifier_list is not None:
            for classifier in self.classifier_list:
                if classifier is not None:
                    for param in classifier.parameters():
                        if param.grad is not None:
                            param.grad.detach_()
                            param.grad.zero_()

        # 7. Accelerator
        if hasattr(self, 'accelerator'):
            self.accelerator.free_memory()
            if hasattr(self.accelerator, 'clear_memory'):
                self.accelerator.clear_memory()

        # 8. CUDA cache and garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for _ in range(3):
                torch.cuda.empty_cache()

        gc.collect()

        # 9. Drop references to CUDA tensors that are not model/classifier parameters
        if torch.cuda.is_available():
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        if obj.device.type == 'cuda' and not obj.requires_grad:
                            is_model_param = any(obj is param for param in self.model.parameters()) if hasattr(self, 'model') else False
                            is_classifier_param = False
                            if hasattr(self, 'classifier_list') and self.classifier_list:
                                is_classifier_param = any(obj is param for classifier in self.classifier_list
                                                        if classifier is not None
                                                        for param in classifier.parameters())

                            if not is_model_param and not is_classifier_param:
                                try:
                                    del obj
                                except:
                                    pass
                except:
                    pass

            torch.cuda.empty_cache()

        # 10. cuBLAS workspaces
        try:
            torch._C._cuda_clearCublasWorkspaces()
        except:
            pass

        # 11. Final garbage collection
        for _ in range(3):
            gc.collect()

    # ================================= Epoch-Level Functions =================================
    def train_epochs(self, task_id):
        """
        Train the backbone and (optionally) the LM head jointly in a single loop.
        """

        # ----------------------------------------------------------------------
        # 1) Training data loader (optionally merged with the replay buffer)
        # ----------------------------------------------------------------------
        if task_id > 0 and self.params.is_replay and not self.params.Replay_batch_level:
            train_dataset = self.train_loader_list[task_id].dataset
            buf_dataset   = Dataset.from_dict(self.buffer.get_all_data())
            buf_dataset.set_format(type="torch")

            base_dataset    = ConcatDataset((train_dataset, buf_dataset))
            base_dataloader = DataLoader(
                base_dataset,
                batch_size=self.params.batch_size,
                shuffle=True,
                drop_last=False,
            )
        else:
            base_dataloader = self.train_loader_list[task_id]

        cur_train_loader = self.accelerator.prepare(base_dataloader)

        # ----------------------------------------------------------------------
        # 2) Training loop (backbone and LM head updated together)
        # ----------------------------------------------------------------------
        backbone_epochs = self.params.training_epochs
        if task_id == 0 and self.params.SEQ_fix_encoder:
            backbone_epochs += self.params.SEQ_warmup_epoch_before_fix_encoder

        self.backbone_update = True
        self.lm_head_update  = bool(self.params.lm_head_finetune)

        for epoch_id in range(backbone_epochs):
            if self.accelerator.is_main_process:
                logger.info(
                    f"------ [Train] epoch {epoch_id + 1}/{backbone_epochs} ------"
                )
                log_memory_usage(f"Task {task_id} Epoch {epoch_id + 1} Begin")

            self.begin_epoch(task_id, epoch_id)

            for lm_input in cur_train_loader:
                self.observe_batch(task_id, epoch_id, lm_input)

            self.end_epoch(task_id, epoch_id)

            if self.accelerator.is_main_process:
                log_memory_usage(f"Task {task_id} Epoch {epoch_id + 1} End")

    def begin_epoch(self, task_id, epoch_id):
        '''
            Start of each epoch
        '''

        self.backbone_loss_list = []
        self.lm_head_loss_list  = []

        # Fix a component by setting its learning rate as zero
        if task_id==0 and self.params.SEQ_fix_encoder and epoch_id==self.params.SEQ_warmup_epoch_before_fix_encoder:
            if 'encoder' in self.param_groups_mapping.keys():
                self.optimizer.param_groups[self.param_groups_mapping['encoder']]['lr']=0
                for name, param in self.model.named_parameters():
                    param.requires_grad = False

    def end_epoch(self, task_id, epoch_id):
        '''
            End of each epoch
        '''
        # Print training information
        if hasattr(self, 'loss_list') and len(self.loss_list)>0:
            mean_loss = np.mean(self.loss_list)
            if self.accelerator.is_main_process:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f"%(
                            epoch_id+1, self.step, mean_loss
                    ))
            self.accelerator.log({'loss':mean_loss},step=self.global_step)

        # Update the classifier weight using class mean features
        if self.params.SEQ_use_prototype_for_prediction and \
            (not (task_id==0 and self.params.SEQ_fix_encoder and epoch_id<self.params.SEQ_warmup_epoch_before_fix_encoder)):
            # NOTE: We do not update the classifier weight during warmup stage!
            self.update_prototype_weight(task_id)

        # Clear gradients and the CUDA cache after each epoch
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad()
        if hasattr(self, 'lm_head_optimizer') and self.lm_head_optimizer is not None:
            self.lm_head_optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def observe_batch(self, task_id, epoch_id, lm_input):
        """
        Observe a batch of data
        """
        # ----------------------------------------------------------
        # 0) bookkeeping
        # ----------------------------------------------------------
        self.step += 1
        self.global_step += 1

        # ----------------------------------------------------------
        # 0-A) Append a replay batch (batch-level replay)
        # ----------------------------------------------------------
        if task_id > 0 and self.params.is_replay and self.params.Replay_batch_level:
            buffer_lm_input = self.buffer.get_one_batch()
            lm_input = {
                k: torch.cat(
                        [lm_input[k], buffer_lm_input[k].to(lm_input[k].device)],
                        dim=0
                    ) if isinstance(lm_input[k], torch.Tensor) else
                    lm_input[k] + buffer_lm_input[k]
                for k in lm_input.keys() if k in buffer_lm_input.keys()
            }

        # ----------------------------------------------------------
        # 0-B) unwrap accelerator
        # ----------------------------------------------------------
        wrap_model = self.wrap_model.module if hasattr(self.wrap_model, "module") else self.wrap_model
        model            = wrap_model.model
        classifier_list  = wrap_model.classifier_list

        # ----------------------------------------------------------
        # 1) Causal-LM loss (no classifier)
        # ----------------------------------------------------------
        if self.params.classifier == "None" or (
            task_id == 0 and self.params.SEQ_fix_encoder
            and epoch_id < self.params.SEQ_warmup_epoch_before_fix_encoder
            and self.params.SEQ_warmup_target == "causal-lm"
        ):
            total_loss = model(
                input_ids      = lm_input["input_ids_with_ans"],
                attention_mask = lm_input["attention_mask_with_ans"],
                labels         = lm_input["labels_with_ans"]
            ).loss

        # ----------------------------------------------------------
        # 2) Cross-entropy loss with a classifier (Linear / CosineLinear)
        # ----------------------------------------------------------
        elif self.params.classifier in ["Linear", "CosineLinear"] and (
            (not self.params.SEQ_use_prototype_for_prediction) or
            (self.params.SEQ_use_prototype_for_prediction and task_id == 0
             and self.params.SEQ_fix_encoder
             and epoch_id < self.params.SEQ_warmup_epoch_before_fix_encoder
             and self.params.SEQ_warmup_target == "ce-loss")
        ):
            extracted_feature = obtain_features(
                params   = self.params,
                model    = model,
                lm_input = lm_input,
                tokenizer= self.tokenizer
            )

            # ------------------------------------------------------
            # 2-A) CIL
            # ------------------------------------------------------
            if self.params.il_mode == "CIL":
                feat_device = extracted_feature.device

                # Cast classifiers to the feature dtype
                for i, classifier in enumerate(classifier_list):
                    if hasattr(classifier, 'weight') and classifier.weight.dtype != extracted_feature.dtype:
                        classifier_list[i] = classifier.to(extracted_feature.dtype)

                logits = torch.cat([
                    classifier_list[i].to(feat_device)(extracted_feature)
                    for i in (
                        range(len(classifier_list)) if self.params.SEQ_preallocated_classifier
                        else range(task_id + 1)
                    )
                ], dim=-1)

                if self.params.classification_type == "sentence-level":
                    label_idx = lm_input["label_idx_cil"]
                else:  # word-level
                    logits    = logits.reshape(-1, logits.shape[-1])
                    label_idx = lm_input["label_idx_cil"].reshape(-1)
                    label_idx = label_idx.masked_fill(
                        label_idx >= self.CL_dataset.continual_config["ACCUM_NUM_CLASS"][task_id], 0
                    )
                    if not self.params.is_replay:
                        label_idx = label_idx.masked_fill(
                            torch.logical_and(
                                label_idx > 0,
                                label_idx < self.CL_dataset.continual_config["PRE_ACCUM_NUM_CLASS"][task_id]
                            ),
                            0
                        )

                label_idx = label_idx.to(logits.device, non_blocking=True)
                total_loss = self.ce_loss(logits, label_idx)

            # ------------------------------------------------------
            # 2-B) TIL
            # ------------------------------------------------------
            elif self.params.il_mode == "TIL":
                feat_device = extracted_feature.device

                # Cast classifiers to the feature dtype
                for i, classifier in enumerate(classifier_list):
                    if hasattr(classifier, 'weight') and classifier.weight.dtype != extracted_feature.dtype:
                        classifier_list[i] = classifier.to(extracted_feature.dtype)

                total_loss = torch.tensor(0.0, device=feat_device)
                for t_id in range(task_id + 1):
                    bg   = self.CL_dataset.continual_config["PRE_ACCUM_NUM_CLASS"][t_id]
                    ed   = self.CL_dataset.continual_config["ACCUM_NUM_CLASS"][t_id]
                    mask = (lm_input["label_idx_cil"] >= bg) & (lm_input["label_idx_cil"] < ed)
                    if mask.sum():
                        logits    = classifier_list[t_id].to(feat_device)(extracted_feature[mask])
                        label_idx = lm_input["label_idx_til"][mask]
                        label_idx = label_idx.to(logits.device, non_blocking=True)
                        total_loss += self.ce_loss(logits, label_idx)
            else:
                raise NotImplementedError()

        # ----------------------------------------------------------
        # 3) Prototype classifier (prediction only): nothing to train
        # ----------------------------------------------------------
        elif self.params.SEQ_use_prototype_for_prediction:
            return None
        else:
            raise NotImplementedError()

        # 4) Backbone + classifier update
        backbone_loss = None
        if getattr(self, "backbone_update", True):
            model.train()
            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            scalar_loss = total_loss.item()
            if not (np.isnan(scalar_loss) or np.isinf(scalar_loss)):
                self.optimizer.step()
                self.backbone_loss_list.append(scalar_loss)
                backbone_loss = scalar_loss

        # 5) Optional LM-head update on detached features
        lm_loss_value = None
        if getattr(self, "lm_head_update", False) and getattr(self, "lm_head_optimizer", None) is not None:
            # Explicitly enable LM-head gradients (LoRA freezes them otherwise)
            lm_head = model.get_output_embeddings()
            for param in lm_head.parameters():
                param.requires_grad = True

            with torch.no_grad():
                feat_detached = extracted_feature.detach()

            enc = self.tokenizer(lm_input["target"], add_special_tokens=False, padding=False)
            label_token_ids = torch.tensor([ids[0] for ids in enc["input_ids"]],
                                           device=feat_detached.device)

            vocab_logits    = lm_head(feat_detached)
            label_token_ids = label_token_ids.to(vocab_logits.device, non_blocking=True)

            lm_ce_loss = self.ce_loss(vocab_logits, label_token_ids)

            self.lm_head_optimizer.zero_grad()
            self.accelerator.backward(lm_ce_loss)
            self.lm_head_optimizer.step()

            lm_loss_value = lm_ce_loss.item()
            if not (np.isnan(lm_loss_value) or np.isinf(lm_loss_value)):
                self.lm_head_loss_list.append(lm_loss_value)

        # ----------------------------------------------------------
        # 6) Step-level logging
        # ----------------------------------------------------------
        if self.params.info_per_steps and self.step % self.params.info_per_steps == 0:
            if self.accelerator.is_main_process:
                if self.backbone_loss_list:
                    mean_b_loss = np.mean(self.backbone_loss_list)
                    logger.info(f"[Backbone]  Epoch {epoch_id+1}, Step {self.step}: mean_loss = {mean_b_loss:.3f}")
                    self.accelerator.log({"backbone_loss": mean_b_loss}, step=self.global_step)
                if self.lm_head_loss_list:
                    mean_lm_loss = np.mean(self.lm_head_loss_list)
                    logger.info(f"[LM-Head]  Epoch {epoch_id+1}, Step {self.step}: mean_loss = {mean_lm_loss:.3f}")
                    self.accelerator.log({"lm_head_loss": mean_lm_loss}, step=self.global_step)

    # ===========================================================================================


    # ======================== Evaluation ========================
    def evaluate_current_task(
        self,
        eval_task_id: int,
        cur_task_id: int,
        phase: str,
        il_mode: str,
    ):
        """
        Evaluate one task. If lm_head_finetune is set, the LM head is fine-tuned
        for lm_head_epochs on the evaluated task's training data with the backbone
        frozen, used for evaluation, and then restored to its original weights.
        """
        assert phase in ["train", "dev", "test"]
        assert il_mode in ["IIL", "CIL", "TIL"]

        loaders = (
            self.train_loader_list
            if phase == "train"
            else self.dev_loader_list
            if phase == "dev"
            else self.test_loader_list
        )
        next_loader = (
            loaders[eval_task_id + 1] if eval_task_id + 1 < len(loaders) else None
        )

        # Unwrap DDP
        wrap_model = (
            self.wrap_model.module if hasattr(self.wrap_model, "module") else self.wrap_model
        )
        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        # 0. No classifier: evaluate via generation
        if classifier_list is None:
            acc_cur, acc_next = evaluate_sent_level_acc_with_generation(
                model=model,
                eval_data_loader=loaders[eval_task_id],
                next_eval_data_loader=next_loader,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )
            return acc_cur, acc_next

        # 1. Optional LM-head fine-tuning before evaluation
        if self.params.lm_head_finetune:
            lm_head = model.get_output_embeddings()
            orig_state = deepcopy(lm_head.state_dict())

            # (1) Freeze the backbone
            for p in model.parameters():
                p.requires_grad = False
            # Keep LM-head gradients enabled (also when LoRA is used)
            for p in lm_head.parameters():
                p.requires_grad = True

            # (2) Fine-tuning loader and optimizer
            ft_loader = self.train_loader_list[eval_task_id]
            ft_opt = torch.optim.AdamW(lm_head.parameters(), lr=self.params.lm_head_lr)

            autocast_ctx = (
                torch.cuda.amp.autocast
                if self.accelerator.mixed_precision in ("fp16", "bf16")
                else nullcontext
            )
            model.eval()
            for _ in range(self.params.lm_head_epochs):
                for lm_input in ft_loader:
                    with torch.no_grad(), autocast_ctx():
                        feats = obtain_features(
                            self.params, model, lm_input, self.tokenizer
                        )
                    dev = lm_head.weight.device
                    if feats.device != dev:
                        feats = feats.to(dev, non_blocking=True)

                    # Label token: first token of the target string
                    if "target" in lm_input:
                        enc = self.tokenizer(
                            lm_input["target"], add_special_tokens=False, padding=False
                        )
                        label_tok = torch.tensor(
                            [ids[0] for ids in enc["input_ids"]], device=dev
                        )
                    else:
                        first = []
                        for row in lm_input["labels_with_ans"]:
                            nz = (row != -100).nonzero(as_tuple=False)
                            first.append(
                                row[nz[0]].item() if nz.numel() else self.tokenizer.eos_token_id
                            )
                        label_tok = torch.tensor(first, device=dev)

                    # Forward and update
                    logits = lm_head(feats)
                    loss = F.cross_entropy(logits, label_tok)
                    ft_opt.zero_grad()
                    loss.backward()
                    ft_opt.step()

                    del feats, logits, label_tok, loss
                    torch.cuda.empty_cache()

            # (3) Evaluate with the fine-tuned LM head
            acc_cur, acc_next = evaluate_sent_level_acc_with_lmhead(
                model=model,
                cur_task_id=cur_task_id,
                eval_data_loader=loaders[eval_task_id],
                next_eval_data_loader=next_loader,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )

            # (4) Restore the LM head and requires_grad
            lm_head.load_state_dict(orig_state)
            for p in model.parameters():
                p.requires_grad = True
            for p in lm_head.parameters():
                p.requires_grad = True
            torch.cuda.empty_cache()
            return acc_cur, acc_next

        # 2. Standard classifier evaluation
        if self.params.classification_type == "sentence-level":
            acc_cur, acc_next = evaluate_sent_level_acc_with_classifier(
                model=model,
                classifier_list=classifier_list,
                cur_task_id=(cur_task_id if il_mode == "CIL" else eval_task_id),
                eval_data_loader=loaders[eval_task_id],
                next_eval_data_loader=next_loader,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )
            return acc_cur, acc_next

        if self.params.classification_type == "word-level":
            acc_cur, acc_next = evaluate_word_level_acc_with_classifier(
                model=model,
                classifier_list=classifier_list,
                cur_task_id=cur_task_id,
                eval_data_loader=loaders[eval_task_id],
                next_eval_data_loader=next_loader,
                tokenizer=self.tokenizer,
                accelerator=self.accelerator,
                params=self.params,
                idx2label=self.CL_dataset.continual_config["idx2label"],
            )
            return acc_cur, acc_next

        raise NotImplementedError(
            f"Unknown classification_type {self.params.classification_type}"
        )

    # ======================== Other Model-Specific Functions ===================================
    def update_prototype_weight(self, task_id):
        '''
            Compute the class mean features and use them as the weight of the CURRENT classifier.

            NOTE: The old classifier is fix since prototype classifier can not be trained.
        '''
        # For Distributed Data Parallel
        if hasattr(self.wrap_model,'module'):
            wrap_model = self.wrap_model.module
        else:
            wrap_model = self.wrap_model

        model = wrap_model.model
        classifier_list = wrap_model.classifier_list

        cur_class_prototypes = torch.zeros_like(classifier_list[task_id].weight.data).to(model.device)
        cur_class_idx = self.CL_dataset.continual_config['CUR_CLASS'][task_id]
        cnt_class_samples = {class_idx:0 for class_idx in cur_class_idx}

        # Compute the class mean of all training samples
        with torch.no_grad():
            for lm_input in self.train_loader_list[task_id]:
                extracted_feature = obtain_features(params=self.params,
                                                    model=model,
                                                    lm_input=lm_input,
                                                    tokenizer=self.tokenizer)
                # bf16 backbones emit bf16 features while classifiers are fp32
                extracted_feature = extracted_feature.to(
                    next(classifier_list[0].parameters()).dtype)
                logits = torch.concatenate([classifier(extracted_feature) for classifier in classifier_list[:task_id+1]],dim=-1)
                if self.params.classification_type == 'sentence-level':
                    label_idx = lm_input['label_idx_cil']
                elif self.params.classification_type == 'word-level':
                    extracted_feature = extracted_feature.reshape(-1,extracted_feature.shape[-1])
                    logits = logits.reshape(-1,logits.shape[-1])
                    label_idx = lm_input['label_idx_cil'].reshape(-1)
                    # NOTE: Mask the unseen labels to O. This line is useful only when one task contains the labels from unseen tasks (mainly for probing study).
                    label_idx = label_idx.masked_fill(label_idx>=logits.shape[-1],0)
                    if not self.params.is_replay:
                        label_idx = label_idx.masked_fill(torch.logical_and(label_idx>0,label_idx<self.CL_dataset.continual_config['PRE_ACCUM_NUM_CLASS'][task_id]),0)
                else:
                    raise NotImplementedError()

                for class_idx in cur_class_idx:
                    if class_idx not in label_idx:
                        continue
                    class_mask = (label_idx==class_idx)
                    cnt_class_samples[class_idx] += 1
                    cur_class_prototypes[class_idx-cur_class_idx[0]] += extracted_feature[class_mask].sum(dim=0)

        # Set the weight of the current classifier
        for class_idx in cur_class_idx:
            if cnt_class_samples[class_idx]==0:
                continue
            classifier_list[task_id].weight.data[class_idx-cur_class_idx[0]] = cur_class_prototypes[class_idx-cur_class_idx[0]]/cnt_class_samples[class_idx]

