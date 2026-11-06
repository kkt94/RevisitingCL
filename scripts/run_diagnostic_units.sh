#!/bin/bash
# Backbone x dataset x method units used for the clustering-free diagnostics,
# the sensitivity analysis and the PEFT study (Sections 5.2-5.3, Appendices B.3 and E.2-E.3).
# Backbone features of all tasks are dumped at every task boundary, and all
# diagnostics are computed offline (see scripts/run_analysis.sh).
#
#   Llama-3.2-1B : {TACRED, Banking77, CLINC150, FewRel} x {Base, DER++, CLSER}
#                  {TACRED, Banking77, CLINC150} x {LoRA, prompt tuning}
#   Qwen3-0.6B   : {TACRED, Banking77, CLINC150} x {Base, DER++, CLSER}
#
# Usage: bash scripts/run_diagnostic_units.sh [seed]
set -e
source "$(dirname "$0")/common.sh"

SEED=${1:-42}

unit() {  # <backbone> <dataset> <config>
  run --exp_prefix "unit-$(basename "$1")-s${SEED}" \
      --cfg "./config/CIL/generative_backbones/$2/$3.yaml" \
      --backbone "$1" --classifier None \
      --training_epochs "$(epochs_of "$2")" --seed "$SEED" --batch_size 4 \
      --dump_task_features --skip_online_clustering
}

for DS in tacred_task8 banking77_task7 clinc150_task15 fewrel_task8; do
  for M in SEQ_full DERpp CLSER; do unit meta-llama/Llama-3.2-1B "$DS" "$M"; done
done
for DS in tacred_task8 banking77_task7 clinc150_task15; do
  for M in LoRA PromptTuning; do unit meta-llama/Llama-3.2-1B "$DS" "$M"; done
  for M in SEQ_full DERpp CLSER; do unit Qwen3-0.6B "$DS" "$M"; done
done
