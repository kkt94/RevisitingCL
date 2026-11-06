# Datasets

No data is distributed with this repository. The raw files follow the layout of the incremental
learning codebase of Zheng et al., whose README links to a copy of the raw data
(https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm). Please obtain each
dataset under its own terms (TACRED must be obtained from the LDC).

Expected layout:

```
dataset/
  topic3datasets/
    agnews/   classes.txt  train.csv  test.csv
    dbpedia/  classes.txt  train.csv  test.csv
    yahoo/    classes.txt  train.csv  test.csv
  clinc150/   data_full.json  label_dict.json
  banking77/  categories.json  train.csv  test.csv
  fewrel/     FewRel-2021.pkl
  tacred/     TACRED-2021.pkl
```

Build the incremental task streams from the repository root with

```bash
bash scripts/preprocess_dataset.sh
```

which creates `dataset/<stream>/continual_data.json` and `dataset/<stream>/continual_config.json`.
The fixed seed (`--seed 1`) reproduces the task splits used in the paper.

| Raw folder | Stream | Tasks | Classes | Source and terms |
|---|---|---|---|---|
| `topic3datasets` (AGNews, DBPedia, Yahoo Answers) | `topic3datasets_task5` | 5 | 25 | Zhang et al. (2015); research-use terms of the original sources, DBpedia under CC-BY-SA |
| `clinc150` | `clinc150_task15` | 15 | 150 | UCI Machine Learning Repository, CC BY 4.0 |
| `banking77` | `banking77_task7` | 7 | 77 | PolyAI, CC BY 4.0 |
| `fewrel` | `fewrel_task8` | 8 | 80 | THUNLP, MIT License |
| `tacred` | `tacred_task8` | 8 | 40 | LDC2018T24, LDC user agreement (must be obtained from the LDC) |
