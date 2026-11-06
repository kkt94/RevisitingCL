#!/bin/bash
# Class-incremental learning with a generative (decoder) backbone, evaluated
# through the LM head, a probing classifier and the backbone diagnostic.
#
# Usage: bash scripts/run_cil_generative.sh <backbone> [seed]
#   e.g. bash scripts/run_cil_generative.sh EleutherAI/pythia-410m-deduped 1
#        bash scripts/run_cil_generative.sh Qwen3-0.6B 1
#        bash scripts/run_cil_generative.sh meta-llama/Llama-3.2-1B 1
set -e
source "$(dirname "$0")/common.sh"

BACKBONE=${1:?backbone}
SEED=${2:-1}
METHODS=(SEQ_full DERpp CLSER L2KD LAMOL_g LAMOL_t LAMOL_KD PCLL)
PREFIX="cil-gen-$(basename "$BACKBONE")-s${SEED}"

for DS in "${DATASETS[@]}"; do
  for M in "${METHODS[@]}"; do
    run --exp_prefix "$PREFIX" \
        --cfg "./config/CIL/generative_backbones/${DS}/${M}.yaml" \
        --backbone "$BACKBONE" --classifier None \
        --training_epochs "$(epochs_of "$DS")" --seed "$SEED" \
        --is_probing True --online_clustering "$CLUSTERING"
  done
  # SEQ*: backbone frozen after the first-task warm-up, linear classifier on top
  run --exp_prefix "$PREFIX" \
      --cfg "./config/CIL/generative_backbones/${DS}/SEQ_pre_warm_fix.yaml" \
      --backbone "$BACKBONE" --classifier Linear \
      --training_epochs "$(epochs_of "$DS")" --seed "$SEED"
done
