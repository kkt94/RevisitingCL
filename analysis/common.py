# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Shared helpers for the offline analysis scripts.

A *run* is one experiment directory written by main_CL.py
(``experiments/<exp_prefix>-<cfg>/<timestamp>/``) that contains ``train.log``
and, when ``--dump_task_features`` was used, ``featdump/after_task{k}.npz``.
A *unit* is one (backbone, dataset, method, PEFT type) combination.
"""

import os
import re
import glob
import json

import numpy as np
from scipy import stats

NUM_RE = r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?'


# --------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------
def find_runs(exp_root, tag_glob='*', require_featdump=True):
    """Return {tag: run_dir}. With require_featdump, only runs whose feature
    dump covers every task are returned."""
    runs = {}
    for log in sorted(glob.glob(os.path.join(exp_root, tag_glob, '*', 'train.log'))):
        run_dir = os.path.dirname(log)
        tag = os.path.basename(os.path.dirname(run_dir))
        if require_featdump:
            ckpts = sorted(glob.glob(os.path.join(run_dir, 'featdump', 'after_task*.npz')))
            if not ckpts:
                continue
            meta = json.loads(str(np.load(ckpts[0], allow_pickle=True)['meta']))
            if len(ckpts) != meta['num_task']:
                continue
        runs[tag] = run_dir
    return runs


def _logged_param(txt, name):
    m = re.search(r"'%s': ('([^']*)'|[^,}]+)" % name, txt)
    if m is None:
        return None
    return m.group(2) if m.group(2) is not None else m.group(1).strip()


def run_info(run_dir):
    """Backbone, dataset, method, PEFT type, post-hoc JLT flag and task count of a run."""
    txt = open(os.path.join(run_dir, 'train.log'), errors='ignore').read(50000)
    peft = _logged_param(txt, 'PEFT_type')
    info = dict(
        backbone=_logged_param(txt, 'backbone'),
        dataset=_logged_param(txt, 'dataset'),
        method=_logged_param(txt, 'method'),
        peft=peft if peft not in (None, 'None') else None,
        jlt_posthoc=_logged_param(txt, 'jlt_posthoc') == 'True',
        num_task=None,
    )
    ckpts = sorted(glob.glob(os.path.join(run_dir, 'featdump', 'after_task*.npz')))
    if ckpts:
        meta = json.loads(str(np.load(ckpts[0], allow_pickle=True)['meta']))
        info['num_task'] = meta['num_task']
    if info['peft'] is not None:
        info['method'] = info['peft']
    return info


def unit_key(info):
    return (info['backbone'], info['dataset'], info['method'])


# --------------------------------------------------------------------------
# accuracy matrices
# --------------------------------------------------------------------------
def parse_head_matrix(run_dir, last_completed=False):
    """Matrix a_{k,i} of the sequentially trained head, as logged by Base.py.

    Entries that were never evaluated are -1 in the log and NaN here. With
    last_completed=True the index of the last checkpoint is also returned.
    """
    log = os.path.join(run_dir, 'train.log')
    if not os.path.exists(log):
        return (None, None) if last_completed else None
    txt = open(log, errors='ignore').read()
    blocks = re.findall(r'Result Summary Test After Task (\d+) =\s*\n(.*?\]\])', txt, flags=re.S)
    if not blocks:
        return (None, None) if last_completed else None
    k_last, body = blocks[-1]
    v = np.array([float(x) for x in re.findall(NUM_RE, body)], dtype=np.float64)
    T = int(round(np.sqrt(len(v))))
    if T * T != len(v):
        return (None, None) if last_completed else None
    M = v.reshape(T, T)
    M[M == -1] = np.nan
    return (M, int(k_last)) if last_completed else M


def load_diag(path):
    """Read the CSV written by run_diag.py.

    Returns {(tag, kind, config, metric): {(ckpt, eval_task): score}}.
    """
    import csv
    from collections import defaultdict
    d = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['kind'] == '__error__':
                continue
            d[(r['tag'], r['kind'], r['config'], r['metric'])][
                (int(r['ckpt']), int(r['eval_task']))] = float(r['score'])
    return d


def to_matrix(cells, T):
    M = np.full((T, T), np.nan)
    for (k, i), v in cells.items():
        if k < T and i < T:
            M[k, i] = v
    return M


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def bwt_fwt(M):
    """BWT (Eq. 1) and operational FWT (Eq. 2) of an a_{k,i} matrix."""
    T = M.shape[0]
    if T < 2:
        return np.nan, np.nan
    bwt = np.nanmean([M[T - 1, i] - M[i, i] for i in range(T - 1)])
    fwt = np.nanmean([M[i, i] - M[i - 1, i] for i in range(1, T)])
    return float(bwt), float(fwt)


def terms(M):
    """A = mean final row, B = mean diagonal, C = mean pre-task row.

    BWT = A - B and FWT = B - C share B; A and C share nothing with FWT and
    BWT respectively.
    """
    T = M.shape[0]
    A = np.nanmean([M[T - 1, i] for i in range(T - 1)])
    B1 = np.nanmean([M[i, i] for i in range(T - 1)])
    B2 = np.nanmean([M[i, i] for i in range(1, T)])
    C = np.nanmean([M[i - 1, i] for i in range(1, T)])
    return A, (B1 + B2) / 2, C


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def _rowcorr(X, Y):
    X = X - X.mean(1, keepdims=True)
    Y = Y - Y.mean(1, keepdims=True)
    return (X * Y).sum(1) / np.sqrt((X * X).sum(1) * (Y * Y).sum(1))


def partial_corr(x, y, z):
    """Pearson correlation of x and y after removing the linear effect of z."""
    Z = np.vstack([np.ones_like(z), z]).T

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, np.nan
    return stats.pearsonr(rx, ry)


def perm_null(A, B, C, n=20000, seed=0):
    """Null distribution of corr(A - B, B - C) when only the shared term B is
    permuted across units. Returns (observed r, null mean, null 5th percentile,
    P(null <= observed))."""
    rng = np.random.RandomState(seed)
    obs = stats.pearsonr(A - B, B - C)[0]
    P = np.stack([rng.permutation(len(B)) for _ in range(n)])
    Bp = B[P]
    null = _rowcorr(A[None, :] - Bp, Bp - C[None, :])
    return obs, float(null.mean()), float(np.percentile(null, 5)), float((null <= obs).mean())


def boot_ci(x, y, n=20000, seed=0):
    """95% percentile bootstrap CI of the Pearson correlation, resampling units."""
    rng = np.random.RandomState(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    idx = np.stack([rng.randint(0, len(x), len(x)) for _ in range(n)])
    X, Y = x[idx], y[idx]
    good = (X.std(1) >= 1e-9) & (Y.std(1) >= 1e-9)
    r = _rowcorr(X[good], Y[good])
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
