# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Backbone-level diagnostics computed offline from the feature dumps written by
utils/featdump.py. Every configuration therefore sees identical backbone
representations, and no configuration requires re-training.

clustering  : unsupervised clustering + cluster-to-label matching, with options
              for algorithm, feature normalization, cluster count, seed,
              matching scheme, class balance and sample size
clusterfree : NCM, kNN-1/10, ridge probe, silhouette and Fisher ratio; no
              cluster count, no random initialization, no label matching.
              NCM, kNN and the ridge probe use a fixed stratified 50/50 split
              of the test features.
"""

import os
import json
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skmetrics
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_checkpoint(run_dir, k):
    """Features and labels of every task, extracted after training task k."""
    z = np.load(os.path.join(run_dir, 'featdump', 'after_task%d.npz' % k),
                allow_pickle=True)
    meta = json.loads(str(z['meta']))
    T = meta['num_task']
    feats = [z['feat_t%d' % t].astype(np.float32) for t in range(T)]
    labs = [z['lab_t%d' % t] for t in range(T)]
    return feats, labs, meta


# --------------------------------------------------------------------------
# feature preprocessing
# --------------------------------------------------------------------------
def apply_norm(X, norm):
    if norm == 'none':
        return X
    if norm == 'l2':
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-8)
    if norm == 'standardize':
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
        return (X - mu) / np.maximum(sd, 1e-8)
    if norm == 'whiten':
        Xc = X - X.mean(0, keepdims=True)
        # economical PCA whitening on the top components
        d = min(Xc.shape[0] - 1, Xc.shape[1], 128)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :d] * np.sqrt(Xc.shape[0] - 1))
    raise ValueError(norm)


def subsample(y, nmax, rng, balance='natural'):
    """Return indices, either a natural-distribution or class-balanced subset."""
    n = len(y)
    if balance == 'balanced':
        classes = np.unique(y)
        per = max(1, int(nmax // len(classes))) if nmax > 0 else None
        idx = []
        for c in classes:
            ci = np.where(y == c)[0]
            rng.shuffle(ci)
            idx.append(ci[:per] if per is not None else ci)
        idx = np.concatenate(idx)
    else:
        idx = np.arange(n)
        if nmax > 0 and n > nmax:
            idx = rng.choice(n, size=nmax, replace=False)
    idx.sort()
    return idx


# --------------------------------------------------------------------------
# clustering diagnostics
# --------------------------------------------------------------------------
def _cluster(X, k, algo, seed):
    if algo == 'kmeans':
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    if algo == 'gmm':
        return GaussianMixture(n_components=k, covariance_type='diag',
                               random_state=seed, reg_covar=1e-4,
                               max_iter=100).fit_predict(X)
    if algo == 'spectral':
        return SpectralClustering(n_clusters=k, random_state=seed,
                                  affinity='nearest_neighbors', n_neighbors=10,
                                  assign_labels='kmeans').fit_predict(X)
    if algo == 'agglomerative':
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    raise ValueError(algo)


def _match(y, c, matching):
    """Map cluster ids to label ids."""
    cm = skmetrics.cluster.contingency_matrix(y, c)
    ys = np.unique(y)
    cs = np.unique(c)
    if matching == 'hungarian':
        r, col = linear_sum_assignment(-cm)
        m = {cs[j]: ys[i] for i, j in zip(r, col)}
    elif matching == 'majority':
        m = {cs[j]: ys[cm[:, j].argmax()] for j in range(len(cs))}
    else:
        raise ValueError(matching)
    return np.array([m.get(x, -1) for x in c])


def clustering_scores(feats, labs, meta, algo='spectral', norm='none',
                      k_mult=1.0, seed=0, matching='hungarian',
                      balance='natural', nmax=1500):
    """Global clustering over pooled test features; per-task scores."""
    T = meta['num_task']
    X = np.concatenate(feats, 0)
    y = np.concatenate(labs, 0)
    task_of = np.concatenate([np.full(len(labs[t]), t) for t in range(T)])

    rng = np.random.RandomState(1234 + seed)
    idx = subsample(y, nmax, rng, balance)
    X, y, task_of = X[idx], y[idx], task_of[idx]

    X = apply_norm(X, norm)
    n_class = int(sum(meta['cur_num_class']))
    k = max(2, int(round(k_mult * n_class)))
    k = min(k, len(X) - 1)

    c = _cluster(X, k, algo, seed)
    mapped = _match(y, c, matching)

    out = {}
    for m in ('ACC', 'ARI', 'NMI'):
        out[m] = np.full(T, np.nan)
    for t in range(T):
        sel = task_of == t
        if sel.sum() < 2:
            continue
        out['ACC'][t] = float(np.mean(mapped[sel] == y[sel]) * 100)
        out['ARI'][t] = float(skmetrics.adjusted_rand_score(y[sel], c[sel]) * 100)
        out['NMI'][t] = float(
            skmetrics.normalized_mutual_info_score(y[sel], c[sel]) * 100)
    return out


# --------------------------------------------------------------------------
# clustering-free diagnostics
# --------------------------------------------------------------------------
def _split_half(y, rng):
    """Stratified 50/50 split (fit half / eval half)."""
    a, b = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        h = max(1, len(ci) // 2)
        a.append(ci[:h])
        b.append(ci[h:] if len(ci) > 1 else ci[:1])
    return np.concatenate(a), np.concatenate(b)


def clusterfree_scores(feats, labs, meta, norm='l2', seed=0, ridge_lambda=1e-2):
    """NCM / kNN-1 / kNN-10 / ridge-probe / silhouette / Fisher ratio."""
    T = meta['num_task']
    X = np.concatenate(feats, 0)
    y = np.concatenate(labs, 0)
    task_of = np.concatenate([np.full(len(labs[t]), t) for t in range(T)])
    X = apply_norm(X, norm)

    rng = np.random.RandomState(999 + seed)
    fit, ev = _split_half(y, rng)
    classes = np.unique(y)
    cidx = {c: i for i, c in enumerate(classes)}

    # nearest class mean
    M = np.stack([X[fit][y[fit] == c].mean(0) for c in classes])
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-8)
    Xe = X[ev]
    Xen = Xe / np.maximum(np.linalg.norm(Xe, axis=1, keepdims=True), 1e-8)
    ncm_pred = classes[np.argmax(Xen @ Mn.T, axis=1)]

    # kNN on the same split
    sim = Xen @ (X[fit] / np.maximum(np.linalg.norm(X[fit], axis=1, keepdims=True), 1e-8)).T
    order = np.argsort(-sim, axis=1)
    yf = y[fit]

    def knn_pred(kk):
        kk = min(kk, order.shape[1])
        nb = yf[order[:, :kk]]
        return np.array([np.bincount(r, minlength=int(classes.max()) + 1).argmax()
                         for r in nb])

    knn1 = knn_pred(1)
    knn10 = knn_pred(10)

    # ridge (least-squares) linear probe, closed form -> deterministic
    Xf = X[fit]
    Yf = np.zeros((len(Xf), len(classes)), dtype=np.float32)
    Yf[np.arange(len(Xf)), [cidx[c] for c in yf]] = 1.0
    d = Xf.shape[1]
    A = Xf.T @ Xf + ridge_lambda * len(Xf) * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xf.T @ Yf)
    probe_pred = classes[np.argmax(Xe @ W, axis=1)]

    te = task_of[ev]
    out = {k: np.full(T, np.nan) for k in
           ('NCM', 'kNN1', 'kNN10', 'RidgeProbe', 'Silhouette', 'Fisher')}
    for t in range(T):
        sel = te == t
        if sel.sum() >= 1:
            out['NCM'][t] = float(np.mean(ncm_pred[sel] == y[ev][sel]) * 100)
            out['kNN1'][t] = float(np.mean(knn1[sel] == y[ev][sel]) * 100)
            out['kNN10'][t] = float(np.mean(knn10[sel] == y[ev][sel]) * 100)
            out['RidgeProbe'][t] = float(np.mean(probe_pred[sel] == y[ev][sel]) * 100)

        # label-aware, training-free geometry of task t
        s = task_of == t
        Xt, yt = X[s], y[s]
        if len(np.unique(yt)) >= 2 and len(Xt) >= 3:
            out['Silhouette'][t] = float(
                skmetrics.silhouette_score(Xt, yt, metric='cosine') * 100)
            gm = Xt.mean(0, keepdims=True)
            sb = sw = 0.0
            for c in np.unique(yt):
                Xc = Xt[yt == c]
                mc = Xc.mean(0, keepdims=True)
                sb += len(Xc) * float(((mc - gm) ** 2).sum())
                sw += float(((Xc - mc) ** 2).sum())
            out['Fisher'][t] = float(100.0 * sb / max(sb + sw, 1e-8))
    return out
