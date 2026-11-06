# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Per-unit BWT / operational FWT under every evaluation interface, and the
statistics of the BWT-FWT relation across units.

Inputs : the CSV written by analysis/run_diag.py and the experiment logs.
Outputs (in --outdir):
  units.csv              one row per unit; BWT/FWT for the trained head, the
                         base Spectral configuration and every clustering-free
                         diagnostic (both normalizations)
  units_long.csv         the same in long format, for every diagnostic series
  coupling.csv           for every series and unit subset: Pearson / Spearman r,
                         T1 = corr(A, FWT), T2 = partial corr(BWT, FWT | B),
                         the shared-term permutation null and a bootstrap CI
  sensitivity_grid.csv   per-unit BWT/FWT for every clustering configuration
  summary.txt            the main tables in readable form

A unit is one (backbone, dataset, method) combination. Post-hoc JLT runs share
the backbone trajectory of their standard run and are excluded. When the same
unit was run more than once, the first run (by experiment tag) is used.

Usage:
    python analysis/export_tables.py --diag outputs/diag.csv --exp_root experiments
"""

import os
import sys
import csv
import argparse

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import (find_runs, run_info, unit_key, parse_head_matrix, load_diag,
                             to_matrix, bwt_fwt, terms, partial_corr, perm_null, boot_ci)

SPECTRAL_BASE = ('algo=spectral|balance=natural|k_mult=1.0|'
                 'matching=hungarian|nmax=1500|norm=none|seed=0')
CLUSTERFREE = ('NCM', 'kNN1', 'kNN10', 'Silhouette', 'Fisher', 'RidgeProbe')
PEFT_METHODS = ('LoRA', 'PromptTuning')


def build_units(exp_root, tag_glob, diag):
    units = {}
    for tag, run_dir in sorted(find_runs(exp_root, tag_glob).items()):
        info = run_info(run_dir)
        if info['jlt_posthoc']:
            continue
        if (tag, 'cluster', SPECTRAL_BASE, 'ACC') not in diag:
            continue
        if any(unit_key(u) == unit_key(info) for u in units.values()):
            continue
        info['run_dir'] = run_dir
        info['family'] = 'PEFT' if info['method'] in PEFT_METHODS else 'fullFT'
        units[tag] = info
    return units


def build_series(units, diag):
    """{(kind, config, metric): {tag: a_{k,i} matrix}}, including the trained head."""
    series = {('head', '-', 'accuracy'): {}}
    for tag, u in units.items():
        M = parse_head_matrix(u['run_dir'])
        if M is not None and M.shape[0] == u['num_task']:
            series[('head', '-', 'accuracy')][tag] = M
    for (tag, kind, cfg, metric), cells in diag.items():
        if tag in units and kind in ('cluster', 'clusterfree'):
            series.setdefault((kind, cfg, metric), {})[tag] = to_matrix(cells, units[tag]['num_task'])
    return series


def subset_stats(mats, tags, nperm, nboot):
    rows = []
    for t in tags:
        if t not in mats:
            continue
        b, f = bwt_fwt(mats[t])
        A, B, C = terms(mats[t])
        if np.all(np.isfinite([b, f, A, B, C])):
            rows.append((b, f, A, B, C))
    out = dict(n=len(rows))
    if len(rows) < 5:
        return out
    bwt, fwt, A, B, C = np.array(rows).T
    if min(bwt.std(), fwt.std(), A.std(), B.std()) < 1e-9:
        return out
    out['r'], out['p_r'] = stats.pearsonr(bwt, fwt)
    out['spearman'] = stats.spearmanr(bwt, fwt)[0]
    out['T1'], out['p_T1'] = stats.pearsonr(A, fwt)
    out['T2'], out['p_T2'] = partial_corr(bwt, fwt, B)
    obs, null_mean, null_p5, p = perm_null(A, B, C, n=nperm)
    out.update(perm_r_obs=obs, perm_null_mean=null_mean, perm_null_p5=null_p5, perm_p=p)
    out['boot_lo'], out['boot_hi'] = boot_ci(bwt, fwt, n=nboot)
    return out


def define_subsets(units, order):
    subsets = {}
    for bb in sorted({u['backbone'] for u in units.values()}):
        name = bb.split('/')[-1]
        base = [t for t in order if units[t]['backbone'] == bb]
        subsets['%s:all' % name] = base
        for fam in ('fullFT', 'PEFT'):
            sel = [t for t in base if units[t]['family'] == fam]
            if sel and len(sel) != len(base):
                subsets['%s:%s' % (name, fam)] = sel
    subsets['all'] = list(order)
    return subsets


def fmt(v, spec='%.4f'):
    return spec % v if isinstance(v, (float, np.floating)) and np.isfinite(v) else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', default='outputs/diag.csv')
    ap.add_argument('--exp_root', default='experiments')
    ap.add_argument('--tag_glob', default='*')
    ap.add_argument('--outdir', default='outputs/tables')
    ap.add_argument('--nperm', type=int, default=20000)
    ap.add_argument('--nboot', type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    diag = load_diag(a.diag)
    units = build_units(a.exp_root, a.tag_glob, diag)
    series = build_series(units, diag)
    order = sorted(units, key=lambda t: (units[t]['backbone'], units[t]['family'],
                                         units[t]['dataset'], units[t]['method']))
    print('units: %d   series: %d' % (len(units), len(series)))

    # ---------------- per-unit tables --------------------------------------
    main_ifaces = [('head', '-', 'accuracy', 'Head'), ('cluster', SPECTRAL_BASE, 'ACC', 'Spectral')]
    for norm in ('l2', 'none'):
        main_ifaces += [('clusterfree', 'norm=%s' % norm, m, '%s_%s' % (m, norm)) for m in CLUSTERFREE]

    head_key = ('head', '-', 'accuracy')
    with open(os.path.join(a.outdir, 'units.csv'), 'w', newline='') as fw:
        w = csv.writer(fw)
        hdr = ['tag', 'backbone', 'dataset', 'method', 'family', 'num_task', 'A_T_head']
        for *_, name in main_ifaces:
            hdr += ['BWT_' + name, 'FWT_' + name]
        w.writerow(hdr)
        for t in order:
            u = units[t]
            Mh = series[head_key].get(t)
            row = [t, u['backbone'], u['dataset'], u['method'], u['family'], u['num_task'],
                   fmt(np.nanmean(Mh[-1]) if Mh is not None else np.nan, '%.2f')]
            for kind, cfg, met, _ in main_ifaces:
                M = series.get((kind, cfg, met), {}).get(t)
                b, f = bwt_fwt(M) if M is not None else (np.nan, np.nan)
                row += [fmt(b, '%.2f'), fmt(f, '%.2f')]
            w.writerow(row)

    with open(os.path.join(a.outdir, 'units_long.csv'), 'w', newline='') as fl:
        w = csv.writer(fl)
        w.writerow(['tag', 'backbone', 'dataset', 'method', 'family',
                    'kind', 'config', 'metric', 'A_last', 'BWT', 'FWT'])
        for key in sorted(series, key=str):
            for t in order:
                M = series[key].get(t)
                if M is None:
                    continue
                u = units[t]
                b, f = bwt_fwt(M)
                w.writerow([t, u['backbone'], u['dataset'], u['method'], u['family'], *key,
                            fmt(np.nanmean(M[-1]), '%.2f'), fmt(b, '%.2f'), fmt(f, '%.2f')])

    # ---------------- coupling statistics ----------------------------------
    subsets = define_subsets(units, order)
    fields = ['subset', 'kind', 'config', 'metric', 'n', 'r', 'p_r', 'spearman',
              'T1', 'p_T1', 'T2', 'p_T2', 'perm_r_obs', 'perm_null_mean',
              'perm_null_p5', 'perm_p', 'boot_lo', 'boot_hi']
    results = {}
    with open(os.path.join(a.outdir, 'coupling.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sname, tags in subsets.items():
            for key in sorted(series, key=str):
                st = subset_stats(series[key], tags, a.nperm, a.nboot)
                results[(sname,) + key] = st
                rec = dict(subset=sname, kind=key[0], config=key[1], metric=key[2])
                rec.update({k: (fmt(v) if isinstance(v, (float, np.floating)) else v)
                            for k, v in st.items()})
                w.writerow(rec)

    # ---------------- sensitivity grid -------------------------------------
    with open(os.path.join(a.outdir, 'sensitivity_grid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tag', 'backbone', 'dataset', 'method', 'family', 'algo', 'norm',
                    'k_mult', 'seed', 'matching', 'balance', 'nmax', 'metric', 'BWT', 'FWT'])
        for key in sorted(k for k in series if k[0] == 'cluster'):
            kv = dict(p.split('=') for p in key[1].split('|'))
            for t in order:
                M = series[key].get(t)
                if M is None:
                    continue
                u = units[t]
                b, fw = bwt_fwt(M)
                w.writerow([t, u['backbone'], u['dataset'], u['method'], u['family'],
                            kv['algo'], kv['norm'], kv['k_mult'], kv['seed'], kv['matching'],
                            kv['balance'], kv['nmax'], key[2], fmt(b, '%.2f'), fmt(fw, '%.2f')])

    # ---------------- readable summary --------------------------------------
    lines = []
    emit = lines.append
    ncm = ('clusterfree', 'norm=l2', 'NCM')
    spectral = ('cluster', SPECTRAL_BASE, 'ACC')
    emit('BWT-FWT Pearson r by unit subset')
    emit('%-36s %4s %10s %10s %10s' % ('subset', 'n', 'Spectral', 'NCM', 'head'))
    for sname in subsets:
        r = [results[(sname,) + k].get('r', np.nan) for k in (spectral, ncm, head_key)]
        n = results[(sname,) + spectral]['n']
        emit('%-36s %4d %10s %10s %10s' % (sname, n, *[fmt(v, '%+.3f') for v in r]))
    for sname in subsets:
        emit('')
        emit('[%s] T1 / T2 / permutation null / bootstrap CI' % sname)
        emit('%-64s %4s %7s %7s %7s %8s %17s' % ('series', 'n', 'r', 'T1', 'T2', 'null', '95% CI'))
        for key in sorted(series, key=str):
            if key[0] == 'cluster' and key[1] != SPECTRAL_BASE:
                continue
            st = results[(sname,) + key]
            if 'r' not in st:
                continue
            emit('%-64s %4d %7.3f %7.3f %7.3f %8.3f   [%6.3f, %6.3f]' %
                 (('%s %s %s' % key)[:64], st['n'], st['r'], st['T1'], st['T2'],
                  st['perm_null_mean'], st['boot_lo'], st['boot_hi']))
        cl = [results[(sname,) + k] for k in series if k[0] == 'cluster']
        cl = [s for s in cl if 'r' in s]
        if cl:
            emit('clustering series: %d | r<0: %d | T1<0: %d (p<.05: %d) | T2<0: %d (p<.05: %d)' % (
                len(cl), sum(s['r'] < 0 for s in cl),
                sum(s['T1'] < 0 for s in cl), sum(s['p_T1'] < 0.05 for s in cl),
                sum(s['T2'] < 0 for s in cl), sum(s['p_T2'] < 0.05 for s in cl)))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(a.outdir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')
    print('\nwrote %s' % a.outdir)


if __name__ == '__main__':
    main()
