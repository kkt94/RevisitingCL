# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Compute every backbone diagnostic configuration from the dumped features.

Clustering is evaluated one factor at a time around the base configuration
(Spectral, no normalization, k = number of classes, seed 0, Hungarian
matching, natural class distribution, 1500 samples), plus the full
algorithm x normalization cross and the full test set. The clustering-free
diagnostics are computed with and without L2 normalization.

Usage:
    python analysis/run_diag.py --exp_root experiments --out outputs/diag.csv --workers 16
"""

import os
import sys
import csv
import glob
import time
import argparse
import itertools
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import find_runs
from analysis.diag_core import load_checkpoint, clustering_scores, clusterfree_scores

BASE = dict(algo='spectral', norm='none', k_mult=1.0, seed=0,
            matching='hungarian', balance='natural', nmax=1500)

AXES = {
    'algo':     ['kmeans', 'gmm', 'spectral', 'agglomerative'],
    'norm':     ['none', 'l2', 'standardize', 'whiten'],
    'k_mult':   [0.5, 1.0, 2.0],
    'seed':     [0, 1, 2],
    'matching': ['hungarian', 'majority'],
    'balance':  ['natural', 'balanced'],
    'nmax':     [750, 1500, 3000],
}


def build_configs():
    cfgs, seen = [], set()

    def add(c, family):
        key = '|'.join('%s=%s' % (k, c[k]) for k in sorted(c))
        if key in seen:
            return
        seen.add(key)
        d = dict(c)
        d['_key'] = key
        d['_family'] = family
        cfgs.append(d)

    # one-factor-at-a-time around the paper's configuration
    for ax, vals in AXES.items():
        for v in vals:
            c = dict(BASE)
            c[ax] = v
            add(c, 'ofat:%s' % ax)
    # algorithm x normalization cross
    for a, n in itertools.product(AXES['algo'], AXES['norm']):
        c = dict(BASE)
        c['algo'], c['norm'] = a, n
        add(c, 'cross:algo-norm')
    # base configuration on the full dumped test set
    c = dict(BASE)
    c['nmax'] = 0
    add(c, 'full-test-set')
    return cfgs


CONFIGS = build_configs()


def job(args):
    tag, run_dir, k = args
    rows = []
    try:
        feats, labs, meta = load_checkpoint(run_dir, k)
    except Exception as e:
        return [('__error__', tag, k, str(e), '', '', '')]

    for c in CONFIGS:
        kw = {kk: c[kk] for kk in BASE}
        try:
            sc = clustering_scores(feats, labs, meta, **kw)
        except Exception as e:
            rows.append(('__error__', tag, k, c['_key'] + ':' + str(e), '', '', ''))
            continue
        for metric, arr in sc.items():
            for i, v in enumerate(arr):
                if np.isfinite(v):
                    rows.append(('cluster', tag, k, c['_key'], metric, i, v))

    for norm in ('l2', 'none'):
        try:
            sc = clusterfree_scores(feats, labs, meta, norm=norm)
        except Exception as e:
            rows.append(('__error__', tag, k, 'clusterfree/%s:%s' % (norm, e), '', '', ''))
            continue
        for metric, arr in sc.items():
            for i, v in enumerate(arr):
                if np.isfinite(v):
                    rows.append(('clusterfree', tag, k, 'norm=%s' % norm, metric, i, v))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default='experiments')
    ap.add_argument('--tag_glob', default='*')
    ap.add_argument('--out', default='outputs/diag.csv')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    runs = find_runs(a.exp_root, a.tag_glob)
    print('complete runs found: %d' % len(runs))
    for t in sorted(runs):
        print('  ', t)
    if not runs:
        return

    jobs = []
    for tag, run_dir in sorted(runs.items()):
        n_ck = len(glob.glob(os.path.join(run_dir, 'featdump', 'after_task*.npz')))
        jobs += [(tag, run_dir, k) for k in range(n_ck)]
    print('checkpoints: %d | configs/checkpoint: %d' % (len(jobs), len(CONFIGS)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'tag', 'ckpt', 'config', 'metric', 'eval_task', 'score'])
        with mp.Pool(a.workers) as pool:
            for n, rows in enumerate(pool.imap_unordered(job, jobs), 1):
                for r in rows:
                    w.writerow(r)
                if n % 10 == 0:
                    el = time.time() - t0
                    print('  %d/%d checkpoints  %.1f min elapsed  eta %.1f min'
                          % (n, len(jobs), el / 60, el / n * (len(jobs) - n) / 60),
                          flush=True)
    print('wrote %s in %.1f min' % (a.out, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
