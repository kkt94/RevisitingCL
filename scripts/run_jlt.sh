#!/bin/bash
# Just LM-Head Tuning (Section 6, Appendix F).
#
#   task-adaptive JLT : before evaluating task t, the backbone is frozen and only the
#                       LM head is re-aligned on the training split of task t
#                       (recoverability diagnostic, not standard CIL accuracy).
#   post-hoc JLT      : the LM head is re-aligned once per checkpoint on a replay buffer
#                       with one example per class of every seen task (standard CIL).
#
# Usage: bash scripts/run_jlt.sh <backbone> [seed]
#   e.g. bash scripts/run_jlt.sh meta-llama/Llama-3.2-1B 1
set -e
source "$(dirname "$0")/common.sh"

BACKBONE=${1:?backbone}
SEED=${2:-1}
NAME=$(basename "$BACKBONE")

# Task-adaptive JLT (Table 3, Table 14)
for DS in "${DATASETS[@]}"; do
  for M in SEQ_full L2KD LAMOL_KD PCLL; do
    run --exp_prefix "jlt-${NAME}-s${SEED}" \
        --cfg "./config/CIL/generative_backbones/${DS}/${M}.yaml" \
        --backbone "$BACKBONE" --classifier None \
        --training_epochs "$(epochs_of "$DS")" --seed "$SEED" \
        --lm_head_finetune --lm_head_lr 5e-4 --lm_head_epochs 3
  done
done

# Post-hoc JLT (Table 17)
for PAIR in banking77_task7:SEQ_full clinc150_task15:SEQ_full tacred_task8:SEQ_full tacred_task8:DERpp; do
  DS=${PAIR%%:*}; M=${PAIR##*:}
  run --exp_prefix "posthoc-jlt-${NAME}-s${SEED}" \
      --cfg "./config/CIL/generative_backbones/${DS}/${M}.yaml" \
      --backbone "$BACKBONE" --classifier None \
      --training_epochs "$(epochs_of "$DS")" --seed "$SEED" \
      --jlt_posthoc --jlt_posthoc_per_class 1 --lm_head_lr 5e-4 --lm_head_epochs 3
done
