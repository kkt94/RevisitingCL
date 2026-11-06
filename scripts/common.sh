#!/bin/bash
# Shared settings for the experiment scripts. Run every script from the repository root.

DATASETS=(topic3datasets_task5 clinc150_task15 banking77_task7 fewrel_task8 tacred_task8)

# Training epochs per incremental task (Appendix A.2).
epochs_of() {
  case "$1" in
    topic3datasets_task5) echo 3 ;;
    *) echo 5 ;;
  esac
}

# Total number of classes, used for the one-example-per-class ER buffer.
num_class_of() {
  case "$1" in
    topic3datasets_task5) echo 25 ;;
    clinc150_task15) echo 150 ;;
    banking77_task7) echo 77 ;;
    fewrel_task8) echo 80 ;;
    tacred_task8) echo 40 ;;
  esac
}

# Online clustering diagnostics: spectral (default) or a comma-separated list,
# e.g. CLUSTERING=spectral,kmeans,gmm,agglomerative,deep (or CLUSTERING=all).
CLUSTERING=${CLUSTERING:-spectral}

run() {
  echo "+ python main_CL.py $*"
  python main_CL.py "$@"
}
