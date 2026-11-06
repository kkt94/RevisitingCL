# Derived from the incremental learning codebase of Zheng et al. (ACL 2024),
# https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm
# Modifications copyright (c) 2026 Keuntae Kim.
# See NOTICE for details.

import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import (
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)

from utils.prompt import get_auto_prompt_tuning_init_text

# -----------------------------------------------------------------------------
#  Supported backbone names and their types
# -----------------------------------------------------------------------------
BACKBONE2TYPE = {
    # Generative
    "gpt2": "generative",
    "gpt2-large": "generative",
    "EleutherAI/pythia-70m-deduped": "generative",
    "EleutherAI/pythia-160m-deduped": "generative",
    "EleutherAI/pythia-410m-deduped": "generative",
    "EleutherAI/pythia-1b-deduped": "generative",
    "EleutherAI/pythia-1.4b-deduped": "generative",
    "EleutherAI/pythia-2.8b-deduped": "generative",
    "baffo32/decapoda-research-llama-7B-hf": "generative",
    "meta-llama/Llama-3.2-1B": "generative",
    "lmsys/vicuna-7b-v1.1": "generative",
    "llama2-13b-orca-8k-3319": "generative",
    # Qwen local checkpoints
    "Qwen3-0.6B": "generative",
    "Qwen3-1.7B": "generative",
    "Qwen2-0.5B": "generative",
    "Qwen2.5-0.5B": "generative",
    # Discriminative
    "roberta-base": "discriminative",
    "roberta-large": "discriminative",
    "bert-base-cased": "discriminative",
    "bert-base-uncased": "discriminative",
    "bert-large-cased": "discriminative",
    "bert-large-uncased": "discriminative",
}

# -----------------------------------------------------------------------------
#  Shared helper: keep tokenizer and embeddings in sync
# -----------------------------------------------------------------------------

def _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task: int = 1):
    """Ensure pad/special tokens exist and resize model embeddings if needed."""
    import torch
    import logging
    
    # 1) Make sure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

    # 2) Extra special tokens
    add_tokens = ["__ans__"]
    if getattr(params, "LAMOL_use_task_specific_gen_token", True):
        add_tokens += [f"__{i}__" for i in range(num_task)]
    else:
        add_tokens.append("__gen__")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})

    # 3) Resize embeddings
    if num_added > 0:
        try:
            # On the meta device, disable mean_resizing to avoid linalg_eig
            embed_tokens = model.get_input_embeddings()
            lm_head = model.get_output_embeddings()
            
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Meta device detected. Using safe resize method.")
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            else:
                model.resize_token_embeddings(len(tokenizer))
                
            model.config.vocab_size = len(tokenizer)  # keep generate() consistent
            logging.info(
                f"[Backbone] Added {num_added} tokens, new vocab_size = {len(tokenizer)}"
            )
        except Exception as e:
            logging.error(f"[Backbone] Failed to resize embeddings: {e}")
            
            # On the meta device, skip the resize and only update the config
            if embed_tokens.weight.device.type == 'meta':
                logging.warning("[Backbone] Skipping embedding resize for Meta device. Config updated only.")
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")
            else:
                # On a real device, resize the embedding matrices manually
                old_vocab_size = embed_tokens.weight.shape[0]
                new_vocab_size = len(tokenizer)
                
                embed_dim = embed_tokens.weight.shape[1]
                
                # Build the new weights on CPU
                new_embed_weight = torch.zeros(new_vocab_size, embed_dim)
                new_embed_weight[:old_vocab_size] = embed_tokens.weight.to('cpu')
                
                # Initialize new token rows from a normal distribution
                if new_vocab_size > old_vocab_size:
                    std = embed_tokens.weight.std().item()
                    new_embed_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                
                embed_tokens.weight = torch.nn.Parameter(new_embed_weight)
                
                # Untied LM head
                if lm_head is not embed_tokens:
                    new_lm_weight = torch.zeros(new_vocab_size, embed_dim)
                    new_lm_weight[:old_vocab_size] = lm_head.weight.to('cpu')
                    if new_vocab_size > old_vocab_size:
                        new_lm_weight[old_vocab_size:].normal_(mean=0.0, std=std)
                    lm_head.weight = torch.nn.Parameter(new_lm_weight)
                
                model.config.vocab_size = new_vocab_size
                logging.info(f"[Backbone] Manual resize completed: {old_vocab_size} -> {new_vocab_size}")


# -----------------------------------------------------------------------------
#  Main builder
# -----------------------------------------------------------------------------

def get_backbone(params, num_task: int = 1):
    """load backbone model & tokenizer and apply PEFT if needed"""

    # Resolve backbone type
    if params.backbone_type == "auto":
        assert (
            params.backbone in BACKBONE2TYPE
        ), f"Not implemented for backbone {params.backbone}"
        setattr(params, "backbone_type", BACKBONE2TYPE[params.backbone])
    else:
        assert params.backbone_type in (
            "generative",
            "discriminative",
        ), f"Invalid backbone_type {params.backbone_type}"

    # ======================================================================
    #  1) Local Llama checkpoints (Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B)
    # ======================================================================
    if params.backbone in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B"]:
        model_path = f"./{params.backbone.split('/')[-1]}"
        print(model_path)

        # Load the tokenizer first and add special tokens
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        add_tokens = ["__ans__"]
        if getattr(params, "LAMOL_use_task_specific_gen_token", True):
            add_tokens += [f"__{i}__" for i in range(num_task)]
        else:
            add_tokens.append("__gen__")
        
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        
        # Load config (keeps the original vocab_size)
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if isinstance(config.rope_scaling, dict) and "rope_type" in config.rope_scaling:
                factor = config.rope_scaling.get("factor", 1.0)
                config.rope_scaling = {"type": "linear", "factor": factor}
        
        # 1B/3B: bf16 + flash attention + Liger kernels; 8B uses 8-bit quantization.
        # 3B is kept in bf16 (int8 weights cannot require grad, so full
        # fine-tuning is impossible); bf16 + gradient checkpointing + an 8-bit
        # optimizer fits 3B full fine-tuning in 24GB.
        is_1b = ("1B" in params.backbone) or ("3B" in params.backbone)
        if is_1b:
            try:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                print("Applied Liger kernel patches to Llama")
            except Exception as e:
                print(f"Liger patch skipped: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
                attn_implementation="flash_attention_2",
            ).to("cuda")
            if "3B" in params.backbone and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                if hasattr(model.config, "use_cache"):
                    model.config.use_cache = False
                print("3B: gradient checkpointing enabled")
            print("Loaded Llama in bf16 with flash_attention_2")
        else:
            try:
                # Try 8-bit quantization to save memory
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=False,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                print("Loaded with 8bit quantization")
            except Exception as e:
                print(f"8bit quantization failed: {e}")
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=False,
                        quantization_config=quantization_config,
                        device_map="auto",
                    )
                    print("Loaded with 4bit quantization")
                except Exception as e2:
                    print(f"4bit quantization also failed: {e2}")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        trust_remote_code=False,
                    )
                    print("Loaded on CPU with float32")
        
        # Resize embeddings
        if num_added > 0:
            try:
                # mean_resizing=False avoids linalg_eig
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Resized embeddings: +{num_added} tokens, vocab_size = {len(tokenizer)}")
            except Exception as e:
                logging.warning(f"[Backbone] Embedding resize failed: {e}")
                # Fall back to updating the config only
                model.config.vocab_size = len(tokenizer)
                logging.info(f"[Backbone] Config updated: vocab_size = {len(tokenizer)}")

        # Optional PEFT
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  2) Local Qwen checkpoints
    # ======================================================================
    elif params.backbone in ["Qwen3-0.6B", "Qwen2-0.5B", "Qwen2.5-0.5B", "Qwen3-1.7B"]:
        model_path = f"./{params.backbone}"
        print(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            padding_side="left" if params.backbone_type == "generative" else "right",
        )

        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    # ======================================================================
    #  3) Other HuggingFace Hub models (online or cached)
    # ======================================================================
    else:
        try:
            config = AutoConfig.from_pretrained(params.backbone)
            config.return_dict = True
        except Exception:
            config = AutoConfig.from_pretrained(
                os.path.join(params.backbone_cache_path, params.backbone)
            )
            config.return_dict = True

        # Load model
        def _load_model_from_pretrained(path, **kwargs):
            # torch_dtype="auto" follows the checkpoint dtype (bf16 for recent
            # releases). Otherwise transformers loads fp32, and a 1B backbone
            # no longer fits a reasonable batch size in 24GB.
            kwargs.setdefault("torch_dtype", "auto")
            return AutoModelForCausalLM.from_pretrained(
                path, low_cpu_mem_usage=True, device_map="auto", **kwargs
            )

        if params.backbone_revision:
            # A specific revision was requested
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                model = _load_model_from_pretrained(
                    params.backbone,
                    config=config,
                    revision=params.backbone_revision,
                    cache_dir=cache_dir,
                )
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(cache_dir), config=config
                )
        else:
            # Use the latest revision
            try:
                model = _load_model_from_pretrained(params.backbone, config=config)
            except Exception:
                model = _load_model_from_pretrained(
                    os.path.join(params.backbone_cache_path, params.backbone),
                    config=config,
                )

        # Optional PEFT
        if (
            hasattr(params, "PEFT_type")
            and params.PEFT_type is not None
            and params.PEFT_type != "None"
        ):
            if params.PEFT_type == "PromptTuning":
                if params.PEFT_prompt_tuning_init_text and params.PEFT_prompt_tuning_init_text != "":
                    prompt_text = (
                        get_auto_prompt_tuning_init_text(dataset=params.dataset)
                        if params.PEFT_prompt_tuning_init_text == "auto"
                        else params.PEFT_prompt_tuning_init_text
                    )
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        prompt_tuning_init=PromptTuningInit.TEXT,
                        prompt_tuning_init_text=prompt_text,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
                else:
                    peft_config = PromptTuningConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        num_virtual_tokens=params.PEFT_num_virtual_tokens,
                        token_dim=model.config.hidden_size,
                        tokenizer_name_or_path=params.backbone,
                    )
            elif params.PEFT_type == "LoRA":
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    target_modules=params.PEFT_lora_target_modules,
                    r=params.PEFT_lora_r,
                    lora_alpha=params.PEFT_lora_alpha,
                    bias=params.PEFT_lora_bias,
                    lora_dropout=params.PEFT_lora_dropout,
                )
            else:
                raise NotImplementedError()

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Load tokenizer
        def _load_tokenizer(path, **kwargs):
            return AutoTokenizer.from_pretrained(
                path,
                padding_side="left" if params.backbone_type == "generative" else "right",
                **kwargs,
            )

        if params.backbone_revision:
            cache_dir = os.path.join(
                params.backbone_cache_path,
                os.path.join(os.path.basename(params.backbone), params.backbone_revision),
            )
            try:
                tokenizer = _load_tokenizer(
                    params.backbone, revision=params.backbone_revision, cache_dir=cache_dir
                )
            except Exception:
                tokenizer = _load_tokenizer(os.path.join(cache_dir))
        else:
            try:
                tokenizer = _load_tokenizer(params.backbone)
            except Exception:
                tokenizer = _load_tokenizer(
                    os.path.join(params.backbone_cache_path, params.backbone)
                )

    # ======================================================================
    #  Common post-processing (pad token, special tokens, embedding sync)
    # ======================================================================
    # Llama checkpoints were already handled above
    if params.backbone_type == "generative" and params.backbone not in ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
        _sync_tokenizer_and_embeddings(tokenizer, model, params, num_task)

    # Fallback pad token (also for discriminative backbones)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(model)
    return model, tokenizer

# -----------------------------------------------------------------------------
#  Feature extraction and generation
# -----------------------------------------------------------------------------

def _validate_lm_input_for_features(lm_input, model, context="obtain_features"):
    """Validate and clamp inputs before a forward pass."""
    logger = logging.getLogger()
    validated_input = {}
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if "input_ids" in lm_input:
        input_ids = lm_input["input_ids"]
        if input_ids.max() >= vocab_size or input_ids.min() < 0:
            logger.warning(
                f"[{context}] input_ids out of range (0-{vocab_size-1}). Clamping to unk_token."
            )
            unk_id = getattr(model.config, "unk_token_id", 0) or 0
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        validated_input["input_ids"] = input_ids
    if "attention_mask" in lm_input:
        validated_input["attention_mask"] = lm_input["attention_mask"]
    for k in lm_input:
        if k not in validated_input:
            validated_input[k] = lm_input[k]
    return validated_input


def obtain_features(params, model, lm_input, tokenizer):
    """Extract last hidden state features for classification"""
    safe_input = _validate_lm_input_for_features(lm_input, model)
    if params.backbone_type == "generative":
        assert params.classification_type == "sentence-level"
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
        ).hidden_states
        assert params.backbone_extract_token == "last_token"
        feat = all_hs[-1][:, -1, :].contiguous()
    elif params.backbone_type == "discriminative":
        all_hs = model(
            input_ids=safe_input["input_ids"],
            attention_mask=safe_input["attention_mask"],
            output_hidden_states=True,
        ).hidden_states
        if params.classification_type == "sentence-level":
            if params.backbone_extract_token == "last_token":
                idx = safe_input["attention_mask"].sum(dim=-1) - 1
                last = all_hs[-1]
                batch, seq, dim = last.size()
                idx = idx.view(-1, 1, 1).expand(-1, 1, dim)
                feat = last.gather(1, idx).squeeze(1).contiguous()
            elif params.backbone_extract_token == "cls_token":
                feat = all_hs[-1][:, 0, :].contiguous()
            else:
                raise NotImplementedError()
        elif params.classification_type == "word-level":
            feat = all_hs[-1]
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return feat


def obtain_generate_ids(params, model, lm_input, tokenizer):
    """Generate continuation ids"""
    safe_input = _validate_lm_input_for_features(lm_input, model, "obtain_generate_ids")
    in_len = safe_input["input_ids"].shape[1]
    gen_all = model.generate(
        input_ids=safe_input["input_ids"],
        attention_mask=safe_input["attention_mask"],
        max_new_tokens=params.backbone_max_new_token,
        pad_token_id=tokenizer.eos_token_id,
    )
    return gen_all[:, in_len:].contiguous()
