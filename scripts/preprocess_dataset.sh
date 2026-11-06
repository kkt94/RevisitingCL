#!/bin/bash
# Build the incremental task streams from the raw datasets in ./dataset/<name>.
# Run from the repository root. The fixed seed reproduces the task splits used in the paper.
set -e
python utils/dataformat_preprocess.py --dataset topic3datasets --seed 1 --num_sample_train_per_class 3000 --num_sample_test_per_class 2000
python utils/dataformat_preprocess.py --dataset clinc150 --seed 1
python utils/dataformat_preprocess.py --dataset banking77 --seed 1
python utils/dataformat_preprocess.py --dataset fewrel --seed 1
python utils/dataformat_preprocess.py --dataset tacred --seed 1
