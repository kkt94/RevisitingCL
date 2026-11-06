#!/bin/bash
# Class-incremental learning with a discriminative (encoder) backbone and a
# linear classifier: Base, ER (one example per class), DER++, CLSER and SEQ*.
# The online clustering diagnostic is constant for SEQ* after the warm-up task.
#
# Usage: bash scripts/run_cil_discriminative.sh <backbone> [seed] [il_mode]
#   e.g. bash scripts/run_cil_discriminative.sh bert-base-cased 1
#        bash scripts/run_cil_discriminative.sh roberta-base 1 TIL   # TIL sanity check
set -e
source "$(dirname "$0")/common.sh"

BACKBONE=${1:?backbone}
SEED=${2:-1}
MODE=${3:-CIL}
PREFIX="$(echo "$MODE" | tr 'A-Z' 'a-z')-disc-$(basename "$BACKBONE")-s${SEED}"

for DS in "${DATASETS[@]}"; do
  CFG_DIR="./config/${MODE}/discriminative_backbones/${DS}"
  COMMON=(--backbone "$BACKBONE" --classifier Linear --training_epochs "$(epochs_of "$DS")"
          --seed "$SEED" --is_probing True --online_clustering "$CLUSTERING")

  run --exp_prefix "$PREFIX" --cfg "${CFG_DIR}/SEQ_full.yaml" "${COMMON[@]}"
  run --exp_prefix "${PREFIX}-ER" --cfg "${CFG_DIR}/SEQ_full.yaml" "${COMMON[@]}" \
      --is_replay True --Replay_fix_budge_each_class True --Replay_buffer_size "$(num_class_of "$DS")"
  run --exp_prefix "$PREFIX" --cfg "${CFG_DIR}/DERpp.yaml" "${COMMON[@]}"
  run --exp_prefix "$PREFIX" --cfg "${CFG_DIR}/CLSER.yaml" "${COMMON[@]}"
  run --exp_prefix "$PREFIX" --cfg "${CFG_DIR}/SEQ_pre_warm_fix.yaml" "${COMMON[@]}"
done
