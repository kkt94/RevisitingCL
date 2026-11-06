# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
BWT and operational FWT under the interfaces evaluated online during training
(trained head, probe, and clustering-based backbone diagnostics), and
their BWT-FWT correlation per backbone (Figures 2-3, Tables 2 and 11-13).

Each run's train.log is parsed for
  head        : the last "Result Summary Test After Task k" matrix
  probe       : the "Summary Probing Acc" matrix (runs with --is_probing True)
  clustering  : the "Updating <algorithm> result: ..." lines (--online_clustering)

A unit is one (il_mode, backbone, dataset, method) combination; BWT and FWT
are averaged over the seeds of a unit before correlating across the units of
one backbone.

Usage:
    python analysis/online_interfaces.py --exp_root experiments --outdir outputs/online
"""

import os
import re
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import NUM_RE, find_runs, parse_head_matrix, bwt_fwt, _logged_param

UPDATE_RE = re.compile(r'Updating (.+?) result: task (\d+), eval_task_id (\d+), with '
                       r'Accuracy (%s), ARI (%s), NMI (%s)' % (NUM_RE, NUM_RE, NUM_RE))


def parse_matrix_block(txt, header):
    blocks = re.findall(header + r'\s*\n(.*?\]\])', txt, flags=re.S)
    if not blocks:
        return None
    v = np.array([float(x) for x in re.findall(NUM_RE, blocks[-1])])
    T = int(round(np.sqrt(len(v))))
    if T * T != len(v):
        return None
    M = v.reshape(T, T)
    M[M == -1] = np.nan
    return M


def parse_run(run_dir):
    """{interface name: a_{k,i} matrix} for one run."""
    txt = open(os.path.join(run_dir, 'train.log'), errors='ignore').read()
    out = {}
    head = parse_head_matrix(run_dir)
    if head is None:
        return out
    T = head.shape[0]
    out['head'] = head
    probe = parse_matrix_block(txt, r'Summary Probing Acc =')
    if probe is not None and probe.shape[0] == T and np.isfinite(probe).any():
        out['probe'] = probe
    clus = defaultdict(lambda: np.full((T, T), np.nan))
    for name, k, i, acc, ari, nmi in UPDATE_RE.findall(txt):
        k, i = int(k), int(i)
        if k < T and i < T:
            for metric, v in (('ACC', acc), ('ARI', ari), ('NMI', nmi)):
                clus['%s|%s' % (name, metric)][k, i] = float(v)
    out.update(clus)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default='experiments')
    ap.add_argument('--tag_glob', default='*')
    ap.add_argument('--outdir', default='outputs/online')
    ap.add_argument('--min_units', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    per_unit = defaultdict(lambda: defaultdict(list))   # unit -> interface -> [(bwt, fwt)]
    for tag, run_dir in sorted(find_runs(a.exp_root, a.tag_glob, require_featdump=False).items()):
        txt = open(os.path.join(run_dir, 'train.log'), errors='ignore').read(50000)
        if _logged_param(txt, 'jlt_posthoc') == 'True' or _logged_param(txt, 'lm_head_finetune') == 'True':
            continue
        unit = (_logged_param(txt, 'il_mode'), _logged_param(txt, 'backbone'),
                _logged_param(txt, 'dataset'), _logged_param(txt, 'method'),
                _logged_param(txt, 'is_replay') == 'True')
        for iface, M in parse_run(run_dir).items():
            b, f = bwt_fwt(M)
            if np.isfinite(b) and np.isfinite(f):
                per_unit[unit][iface].append((b, f))

    with open(os.path.join(a.outdir, 'units.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['il_mode', 'backbone', 'dataset', 'method', 'replay', 'interface',
                    'n_seeds', 'BWT', 'FWT'])
        for unit in sorted(per_unit, key=str):
            for iface, vals in sorted(per_unit[unit].items()):
                b, f = np.mean(vals, 0)
                w.writerow([*unit, iface, len(vals), '%.2f' % b, '%.2f' % f])

    # correlation per (il_mode, backbone) across its dataset-method units
    groups = defaultdict(lambda: defaultdict(list))
    for unit, ifaces in per_unit.items():
        for iface, vals in ifaces.items():
            groups[(unit[0], unit[1])][iface].append(np.mean(vals, 0))

    rows = []
    for (mode, bb), ifaces in sorted(groups.items(), key=str):
        for iface, pts in sorted(ifaces.items()):
            pts = np.array(pts)
            if len(pts) < a.min_units or pts[:, 0].std() < 1e-9 or pts[:, 1].std() < 1e-9:
                continue
            pr, pp = stats.pearsonr(pts[:, 0], pts[:, 1])
            sr = stats.spearmanr(pts[:, 0], pts[:, 1])[0]
            rows.append(dict(il_mode=mode, backbone=bb, interface=iface, n_units=len(pts),
                             pearson=pr, p=pp, spearman=sr))

    with open(os.path.join(a.outdir, 'correlations.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['il_mode', 'backbone', 'interface', 'n_units',
                                           'pearson', 'p', 'spearman'])
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.4f' % v if isinstance(v, float) else v) for k, v in r.items()})

    print('%-5s %-34s %-34s %5s %9s %9s' % ('mode', 'backbone', 'interface', 'n', 'pearson', 'spearman'))
    for r in rows:
        print('%-5s %-34s %-34s %5d %9.4f %9.4f' % (r['il_mode'], r['backbone'][:34],
                                                   r['interface'][:34], r['n_units'],
                                                   r['pearson'], r['spearman']))

    # aggregate over backbones (Table 2): head, probe, and all clustering configurations
    print('\nAggregate over backbones')
    for mode in sorted({r['il_mode'] for r in rows}):
        sel = [r for r in rows if r['il_mode'] == mode]
        for fam, pick in (('trained head', lambda i: i == 'head'),
                          ('probe', lambda i: i == 'probe'),
                          ('backbone diagnostic', lambda i: '|' in i)):
            rs = np.array([r['pearson'] for r in sel if pick(r['interface'])])
            if len(rs):
                print('%-4s %-20s n=%-4d mean r=%+.2f std=%.2f range %+.2f to %+.2f  negative %d/%d'
                      % (mode, fam, len(rs), rs.mean(), rs.std(), rs.min(), rs.max(),
                         (rs < 0).sum(), len(rs)))
    print('\nwrote %s' % a.outdir)


if __name__ == '__main__':
    main()
