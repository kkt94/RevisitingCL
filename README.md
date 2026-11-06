# [NeurIPS 2026] Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity

> Keuntae Kim\*, Beomseok Lee\*, Hyunwoo Kim\*, Yong Suk Choi&dagger;
>
> Hanyang University, LG Electronics, KT Corporation
>
> \*: Equal contribution, &dagger;: Corresponding author

**[[Paper (PDF)]](paper/RevisitingCL_NeurIPS2026.pdf)**

This is the official code of the NeurIPS 2026 paper
**"Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity"**.
The camera-ready paper is included in this repository as
[paper/RevisitingCL_NeurIPS2026.pdf](paper/RevisitingCL_NeurIPS2026.pdf).

## Abstract

Continual learning in language models is commonly studied in the incremental learning (IL) setting,
where a model learns a sequence of tasks and is evaluated through its final classifier or
language-model (LM) head. This interface, however, does not indicate where forgetting occurs: a drop
in accuracy may reflect representation degradation in the backbone, readout mismatch in the head, or
both. We therefore argue that backward transfer (BWT) and forward transfer (FWT) should be
interpreted as interface-dependent quantities rather than as single model-level properties. We study
three evaluation interfaces: the trained head, which is updated sequentially with the backbone; a
probe newly trained on frozen backbone features; and a classifier-free backbone diagnostic. They
measure deployed performance, recoverability under a new readout, and representation separability,
respectively. Across class-incremental experiments with encoder and decoder language backbones, plus
task-incremental sanity checks, conclusions about forgetting and plasticity change with the
interface: the trained head often indicates severe forgetting, the probe near-complete
recoverability, and the backbone diagnostic intermediate representation drift with a consistent
negative relation between stability and plasticity. This relation holds for clustering-free
diagnostics such as nearest-class-mean accuracy and across five clustering algorithms and three
metrics. Finally, Just LM-Head Tuning (JLT), a head-only intervention on a fixed incrementally
trained backbone, recovers 96-99% of the gap to joint training for replay- and distillation-based
methods but only about three quarters for plain sequential fine-tuning, suggesting that many
failures attributed to catastrophic forgetting reflect head-backbone mismatch plus partial
representation drift. Our results call for reporting where forgetting is measured, not only how
much.

## Overview

This repository measures backward transfer (BWT) and operational forward transfer (FWT) through
three evaluation interfaces and a head-only intervention:

| Interface | Frozen | Readout | Measures |
|---|---|---|---|
| Trained head | nothing | classifier / LM head trained during IL | deployed IL performance |
| Probe | backbone | newly trained linear classifier | information recoverable by a fresh readout |
| Backbone diagnostic | backbone | none (clustering, or clustering-free NCM / kNN / silhouette / Fisher) | classifier-free representation separability |
| JLT (Just LM-Head Tuning) | backbone | LM head only | performance recoverable by head re-alignment |

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Backbones](#backbones)
- [Quick start](#quick-start)
- [Evaluation interfaces and options](#evaluation-interfaces-and-options)
- [Reproducing the paper](#reproducing-the-paper)
- [Analysis](#analysis)
- [Notes](#notes)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Repository structure

```
main_CL.py                  entry point for every incremental learning run
models/                     IL methods: SEQ (Base, SEQ*), DER++, CLSER, L2KD, LAMOL, LAMOL_KD, PCLL,
                            PEFT (LoRA, prompt tuning)
utils/
  backbone.py, dataloader.py, evaluation.py, ...    training and evaluation utilities
  probing.py                probe interface
  spectral.py, kmeans.py, gmm.py, agglomerative.py, deep_clustering.py
                            online backbone diagnostics
  featdump.py               per-task backbone feature dumps for offline diagnostics
  posthoc_jlt.py            post-hoc JLT
config/{CIL,TIL}/{discriminative,generative}_backbones/<stream>/<method>.yaml
scripts/                    experiment and analysis scripts
analysis/                   offline diagnostics and statistics
tests/                      unit test for post-hoc JLT
paper/                      camera-ready paper (PDF)
dataset/                    raw data and task streams (not distributed, see dataset/README.md)
```

## Installation

```bash
git clone https://github.com/kkt94/RevisitingCL.git
cd RevisitingCL
conda create -n revisitingcl python=3.11 -y
conda activate revisitingcl
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA
pip install -r requirements.txt

# Llama-3.2-1B/3B are loaded with FlashAttention-2
pip install flash-attn --no-build-isolation
pip install liger-kernel   # optional
```

All experiments in the paper were run on a server with eight NVIDIA RTX A5000 (24GB) GPUs.

## Data

No data is distributed with this repository. Place the raw files under `dataset/` as described in
[dataset/README.md](dataset/README.md) and build the incremental task streams:

```bash
bash scripts/preprocess_dataset.sh
```

This creates `dataset/<stream>/continual_data.json` and `dataset/<stream>/continual_config.json`
for the five streams used in the paper:

| Stream | Tasks | Classes | Task type |
|---|---|---|---|
| `topic3datasets_task5` | 5 | 25 | topic classification (AGNews, DBPedia, Yahoo Answers) |
| `clinc150_task15` | 15 | 150 | intent classification |
| `banking77_task7` | 7 | 77 | intent classification |
| `fewrel_task8` | 8 | 80 | relation extraction |
| `tacred_task8` | 8 | 40 | relation extraction |

## Backbones

Hugging Face Hub names are used directly for BERT, RoBERTa and Pythia
(e.g. `bert-base-cased`, `roberta-large`, `EleutherAI/pythia-410m-deduped`).
Llama and Qwen checkpoints are loaded from a local directory in the repository root named after
the model:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir Llama-3.2-1B   # --backbone meta-llama/Llama-3.2-1B
huggingface-cli download Qwen/Qwen3-0.6B --local-dir Qwen3-0.6B             # --backbone Qwen3-0.6B
```

## Quick start

Every run is a call to `main_CL.py` with a method configuration file. Command-line options
override the values in the configuration file.

```bash
# Smoke test: Pythia-70M, sequential fine-tuning on TACRED (8 tasks), one epoch per task,
# evaluated through the LM head, the probe and the spectral-clustering diagnostic
python main_CL.py --exp_prefix smoke \
    --cfg ./config/CIL/generative_backbones/tacred_task8/SEQ_full.yaml \
    --backbone EleutherAI/pythia-70m-deduped --classifier None \
    --training_epochs 1 --seed 1 --is_probing True
```

Each run writes its log to `experiments/<exp_prefix>-<config>/<timestamp>/train.log`. After every
task the log reports the accuracy matrix of each interface, for example

```
Result Summary Test After Task 7                      # trained head (LM head or classifier)
Result Summary Probing After Task 7 (LinearProb)      # probe
Result Summary Spectral Clustering After Task 7       # backbone diagnostic
```

and `analysis/online_interfaces.py` turns these matrices into BWT and FWT (see [Analysis](#analysis)).
Logging to Weights & Biases is offline by default; pass `--is_wandb True --wandb_project <name>`
to log online.

## Evaluation interfaces and options

| Option | Effect |
|---|---|
| `--is_probing True` | probe interface: a linear probe is trained on frozen features after every task |
| `--online_clustering spectral` | backbone diagnostic during training (default `spectral`); any comma-separated subset of `spectral,kmeans,gmm,agglomerative,deep`, or `all` |
| `--skip_online_clustering` | skip the online clustering diagnostic (when it is computed offline) |
| `--dump_task_features` | dump backbone test features of all tasks at every task boundary for the offline diagnostics |
| `--lm_head_finetune` | task-adaptive JLT: before each task is evaluated, the backbone is frozen and only the LM head is re-aligned on that task's training split (recoverability diagnostic) |
| `--jlt_posthoc` | post-hoc JLT: the LM head is re-aligned once per checkpoint on a small replay buffer that takes `--jlt_posthoc_per_class` x (number of classes) examples from the training loader of every seen task, rounded up to whole batches, then restored (standard CIL evaluation) |
| `--lm_head_lr`, `--lm_head_epochs` | JLT learning rate (default 5e-4) and epochs (default 3) |
| `--is_replay True --Replay_fix_budge_each_class True --Replay_buffer_size <#classes>` | ER with one stored example per class (use with `SEQ_full.yaml`) |
| `--il_mode TIL` | task-incremental evaluation (use the configurations under `config/TIL/`) |

Boolean options of the form `--flag True` are parsed with `type=bool`; omit them instead of
passing `False`.

### Examples

```bash
# Generative CIL with Llama-3.2-1B, all five clustering diagnostics online
python main_CL.py --exp_prefix llama \
    --cfg ./config/CIL/generative_backbones/banking77_task7/DERpp.yaml \
    --backbone meta-llama/Llama-3.2-1B --classifier None \
    --training_epochs 5 --seed 1 --is_probing True --online_clustering all

# Discriminative CIL with BERT and a linear classifier
python main_CL.py --exp_prefix bert \
    --cfg ./config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml \
    --backbone bert-base-cased --classifier Linear \
    --training_epochs 5 --seed 1 --is_probing True

# ER with one example per class (CLINC150 has 150 classes)
python main_CL.py --exp_prefix er \
    --cfg ./config/CIL/discriminative_backbones/clinc150_task15/SEQ_full.yaml \
    --backbone bert-base-cased --classifier Linear --training_epochs 5 --seed 1 \
    --is_replay True --Replay_fix_budge_each_class True --Replay_buffer_size 150

# Task-adaptive JLT
python main_CL.py --exp_prefix jlt \
    --cfg ./config/CIL/generative_backbones/tacred_task8/SEQ_full.yaml \
    --backbone meta-llama/Llama-3.2-1B --classifier None --training_epochs 5 --seed 1 \
    --lm_head_finetune --lm_head_lr 5e-4 --lm_head_epochs 3

# Post-hoc JLT (buffer of roughly one example per class)
python main_CL.py --exp_prefix posthoc \
    --cfg ./config/CIL/generative_backbones/tacred_task8/SEQ_full.yaml \
    --backbone meta-llama/Llama-3.2-1B --classifier None --training_epochs 5 --seed 1 \
    --jlt_posthoc --jlt_posthoc_per_class 1

# Feature dumps for the offline backbone diagnostics (LoRA unit)
python main_CL.py --exp_prefix unit \
    --cfg ./config/CIL/generative_backbones/clinc150_task15/LoRA.yaml \
    --backbone meta-llama/Llama-3.2-1B --classifier None --training_epochs 5 --seed 42 \
    --batch_size 4 --dump_task_features --skip_online_clustering
```

## Reproducing the paper

The scripts below run every benchmark stream for one backbone and seed. Unless otherwise noted in the
paper, experiments were repeated with three seeds.

| Script | Paper |
|---|---|
| `scripts/run_cil_discriminative.sh <backbone> <seed>` | CIL with BERT/RoBERTa: Base, ER, DER++, CLSER, SEQ* (Figures 2a, 7, 8; Tables 2, 8, 12, 13) |
| `scripts/run_cil_discriminative.sh <backbone> <seed> TIL` | TIL sanity check (Figures 4-6; Table 11) |
| `scripts/run_cil_generative.sh <backbone> <seed>` | CIL with Pythia/Qwen/Llama: Base, DER++, CLSER, L2KD, LAMOL_g, LAMOL_t, LAMOL_KD, PCLL, SEQ* (Figures 2b, 3, 9, 10; Tables 2, 9, 12, 13) |
| `scripts/run_jlt.sh <backbone> <seed>` | task-adaptive JLT (Tables 3, 16-18) and post-hoc JLT (Table 20) |
| `scripts/run_diagnostic_units.sh <seed>` | 27 backbone x dataset x method units on Llama-3.2-1B and Qwen3-0.6B, including LoRA and prompt tuning, with feature dumps (Sections 5.2-5.3; Tables 10, 14, 15) |
| `scripts/run_analysis.sh <experiments_dir> <output_dir>` | all offline analyses below |

```bash
bash scripts/run_cil_discriminative.sh bert-base-cased 1
bash scripts/run_cil_discriminative.sh bert-base-cased 1 TIL
bash scripts/run_cil_generative.sh EleutherAI/pythia-410m-deduped 1
CLUSTERING=all bash scripts/run_cil_generative.sh Qwen3-0.6B 1
bash scripts/run_jlt.sh meta-llama/Llama-3.2-1B 1
bash scripts/run_diagnostic_units.sh 42
bash scripts/run_analysis.sh experiments outputs
```

`CLUSTERING` selects the online clustering diagnostics of the experiment scripts (default
`spectral`).

## Analysis

| Script | Output |
|---|---|
| `analysis/online_interfaces.py` | BWT/FWT of the trained head, the probe and the clustering-based backbone diagnostic from the training logs, and their BWT-FWT correlation per backbone |
| `analysis/run_diag.py` | every clustering configuration (algorithm, normalization, cluster count, seed, matching, class balance, sample size) and the clustering-free diagnostics (NCM, kNN-1/10, silhouette, Fisher ratio), computed from the feature dumps |
| `analysis/export_tables.py` | per-unit BWT/FWT for every interface, BWT-FWT correlations per unit subset, and the sensitivity grid |
| `analysis/collect_metrics.py` | A_T, BWT and FWT of the trained head for every run, and post-hoc JLT against the matching standard run |
| `analysis/jlt_decomposition.py` | readout-recoverable gap and residual of the JLT decomposition (Tables 3 and 18) |

```bash
bash scripts/run_analysis.sh experiments outputs
python tests/test_posthoc_jlt.py   # CPU check of post-hoc JLT (head updated, backbone untouched, head restored)
```

## Notes

- Generation for the LM-head interface uses each backbone's default generation configuration.
- In JLT, the original LM-head weights are restored after evaluation.
- SEQ* keeps the backbone fixed after the first-task warm-up, so its BWT-backbone and FWT-backbone
  are zero by construction and it is excluded from the backbone-level correlation analysis.
- KLDA (Momeni et al., 2025), reported as KLDA-E in Appendix Tables 16-17, is not part of this
  repository.

## Citation

```bibtex
@inproceedings{kim2026revisiting,
  title     = {Revisiting Continual Learning in Language Models: A Three-Interface Diagnosis of Stability and Plasticity},
  author    = {Kim, Keuntae and Lee, Beomseok and Kim, Hyunwoo and Choi, Yong Suk},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

This code builds on the incremental learning codebase of Junhao Zheng, Shengjie Qiu and Qianli Ma
([zzz47zzz/codebase-for-incremental-learning-with-llm](https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm)).
Please also cite their work:

```bibtex
@inproceedings{zheng2024learn,
  title     = {Learn or Recall? Revisiting Incremental Learning with Pre-trained Language Models},
  author    = {Zheng, Junhao and Qiu, Shengjie and Ma, Qianli},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {14848--14877},
  year      = {2024}
}
```

The method implementations follow the official code of
DER++ ([mammoth](https://github.com/aimagelab/mammoth)),
CLS-ER ([NeurAI-Lab/CLS-ER](https://github.com/NeurAI-Lab/CLS-ER)),
LAMOL ([jojotenya/LAMOL](https://github.com/jojotenya/LAMOL)),
L2KD ([voidism/L2KD](https://github.com/voidism/L2KD)) and
PCLL ([DAMO-ConvAI/pcll](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/pcll)),
and the FewRel/TACRED label mapping follows [VAG](https://github.com/shaoyijia/VAG).
See [NOTICE](NOTICE) for details.

This work was supported by the Institute of Information & Communications Technology Planning &
Evaluation (IITP) grant (No. RS-2025-25422680, No. RS-2020-II201373), and the National Research
Foundation of Korea (NRF) grant (No. RS-2025-00520618) funded by the Korean Government (MSIT).

## License

The original contributions of this repository are released under the [MIT License](LICENSE).
Files derived from third-party code remain subject to the terms of their original authors;
see [NOTICE](NOTICE).
