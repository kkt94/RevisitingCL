# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
Standard CIL metrics (A_T, BWT, operational FWT) of the sequentially trained
head for every run, and the comparison between post-hoc JLT runs and the
matching standard runs (Appendix Table 17).

Usage:
    python analysis/collect_metrics.py --exp_root experiments --out outputs/metrics.csv
"""

import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import find_runs, run_info, parse_head_matrix


def metrics_at(M, k):
    """A_k, BWT and operational FWT at checkpoint k (the last completed task)."""
    A = float(np.nanmean(M[k, :k + 1]))
    if k == 0:
        return A, np.nan, np.nan
    bwt = np.nanmean([M[k, i] - M[i, i] for i in range(k)])
    fwt = np.nanmean([M[i, i] - M[i - 1, i] for i in range(1, k + 1)])
    return A, float(bwt), float(fwt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_root', default='experiments')
    ap.add_argument('--tag_glob', default='*')
    ap.add_argument('--out', default='outputs/metrics.csv')
    a = ap.parse_args()

    rows = []
    for tag, run_dir in sorted(find_runs(a.exp_root, a.tag_glob, require_featdump=False).items()):
        M, k = parse_head_matrix(run_dir, last_completed=True)
        if M is None:
            continue
        info = run_info(run_dir)
        A, bwt, fwt = metrics_at(M, k)
        rows.append(dict(tag=tag, backbone=info['backbone'], dataset=info['dataset'],
                         method=info['method'], posthoc_jlt=info['jlt_posthoc'],
                         tasks_done=k + 1, tasks_total=M.shape[0],
                         A_T=round(A, 2), BWT=round(bwt, 2), FWT=round(fwt, 2),
                         run_dir=run_dir))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['tag'])
        w.writeheader()
        w.writerows(rows)

    print('%-34s %-18s %-14s %-5s %-6s %8s %8s %8s' %
          ('tag', 'dataset', 'method', 'JLT', 'done', 'A_T', 'BWT', 'FWT'))
    for r in sorted(rows, key=lambda z: (z['backbone'], z['dataset'], z['method'], z['posthoc_jlt'])):
        print('%-34s %-18s %-14s %-5s %-6s %8.2f %8.2f %8.2f' %
              (r['tag'][:34], r['dataset'], r['method'], 'yes' if r['posthoc_jlt'] else '-',
               '%d/%d' % (r['tasks_done'], r['tasks_total']), r['A_T'], r['BWT'], r['FWT']))

    # post-hoc JLT vs. the standard run of the same unit
    best = {}
    for r in rows:
        key = (r['backbone'], r['dataset'], r['method'], r['posthoc_jlt'])
        if key not in best or r['tasks_done'] > best[key]['tasks_done']:
            best[key] = r
    pairs = []
    for (bb, ds, me, jlt), r in sorted(best.items()):
        j = best.get((bb, ds, me, True))
        if jlt or j is None or j['tasks_done'] != r['tasks_done'] or r['tasks_done'] != r['tasks_total']:
            continue
        pairs.append((bb, ds, me, r, j))
    if pairs:
        print('\nPost-hoc JLT vs. standard evaluation')
        print('%-22s %-18s %-8s %8s %8s %8s %8s %8s %8s' %
              ('backbone', 'dataset', 'method', 'A_T', '+JLT', 'BWT', '+JLT', 'FWT', '+JLT'))
        for bb, ds, me, r, j in pairs:
            print('%-22s %-18s %-8s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f' %
                  (bb.split('/')[-1], ds, me, r['A_T'], j['A_T'], r['BWT'], j['BWT'], r['FWT'], j['FWT']))
        avg = np.array([[r['A_T'], j['A_T'], r['BWT'], j['BWT'], r['FWT'], j['FWT']]
                        for *_, r, j in pairs]).mean(0)
        print('%-50s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f' % ('average', *avg))
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
