#!/bin/bash
# Offline analysis of the dumped backbone features and the training logs.
#
# Usage: bash scripts/run_analysis.sh [experiments_dir] [output_dir] [num_workers]
set -e

EXP=${1:-experiments}
OUT=${2:-outputs}
WORKERS=${3:-16}
mkdir -p "$OUT"

# 1) every clustering and clustering-free diagnostic, per checkpoint and task
python analysis/run_diag.py --exp_root "$EXP" --out "$OUT/diag.csv" --workers "$WORKERS"

# 2) per-unit BWT/FWT, BWT-FWT correlations per unit subset, sensitivity grid
python analysis/export_tables.py --diag "$OUT/diag.csv" --exp_root "$EXP" --outdir "$OUT/tables"

# 3) head-interface metrics and post-hoc JLT comparison
python analysis/collect_metrics.py --exp_root "$EXP" --out "$OUT/metrics.csv"

# 4) JLT decomposition from Appendix Tables 16-17
python analysis/jlt_decomposition.py | tee "$OUT/jlt_decomposition.txt"

# 5) online interfaces (head, probe, clustering) and their BWT-FWT correlation per backbone
python analysis/online_interfaces.py --exp_root "$EXP" --outdir "$OUT/online"
