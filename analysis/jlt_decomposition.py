# Copyright (c) 2026 Keuntae Kim.
# Licensed under the MIT License. See LICENSE for details.

"""
JLT decomposition of Section 6, computed from the per-stream results of
Appendix Tables 16-17 (three decoder backbones x five streams).

  A_joint - A_std = (A_JLT - A_std) + (A_joint - A_JLT)
                    readout-recoverable   residual

Prints the aggregate decomposition (Table 3), the residual per stream and the
change in operational FWT under task-adaptive JLT (Table 18).

Usage:
    python analysis/jlt_decomposition.py
"""

import numpy as np

DATASETS = ['Tacred', 'Banking77', 'Clinc150', 'Fewrel', 'Topic3']

# (A_T, BWT, FWT) per stream, from Appendix Tables 16-17. "Joint" is A_T of joint fine-tuning.
TABLE14 = {
    'Pythia-410M': {
        'Joint': [98.99, 95.66, 96.33, 95.35, 95.50],
        'Base':        [(11.97, -91.26, 91.28), (7.56, -75.42, 68.86), (4.67, -89.69, 87.62), (6.17, -80.57, 73.59), (20.53, -91.02, 93.08)],
        'L2KD':        [(24.68, -79.27, 94.39), (52.32, -35.08, 85.72), (26.67, -73.17, 94.74), (30.18, -53.23, 74.57), (58.17, 10.27, 40.25)],
        'LAMOL_KD':    [(32.81, -69.55, 93.68), (49.48, -54.92, 97.08), (42.09, -60.48, 97.50), (27.70, -75.60, 95.43), (50.08, -54.98, 92.97)],
        'PCLL':        [(24.30, -65.29, 74.37), (45.91, -56.97, 84.41), (44.16, -57.02, 91.81), (29.79, -68.41, 82.84), (55.80, -42.23, 92.66)],
        'Base+JLT':    [(98.93, -0.55, 11.38), (75.07, -19.92, 84.58), (72.47, -25.50, 92.29), (51.31, -36.10, 77.50), (72.21, -14.80, 79.92)],
        'L2KD+JLT':    [(98.04, -0.25, 4.92), (95.23, -2.99, 88.56), (96.20, -3.38, 94.52), (95.18, -1.04, 93.52), (93.91, -0.16, 92.95)],
        'LAMOL_KD+JLT': [(98.24, 0.25, 9.59), (95.03, -3.33, 88.98), (96.38, -2.95, 93.40), (95.73, -0.54, 92.82), (94.00, -0.08, 92.99)],
        'PCLL+JLT':    [(96.70, -0.58, 10.29), (93.31, -3.60, 87.48), (96.27, -3.29, 91.36), (94.70, -1.06, 92.15), (93.46, -0.26, 93.51)],
    },
    'Llama3.2-1B': {
        'Joint': [98.33, 96.26, 97.33, 95.65, 95.50],
        'Base':        [(12.23, -91.60, 92.01), (11.56, -91.10, 89.28), (5.89, -93.21, 92.45), (8.42, -81.39, 77.03), (19.50, -88.74, 97.84)],
        'L2KD':        [(31.83, -68.77, 81.20), (51.56, -32.46, 76.52), (26.44, -66.62, 87.88), (32.75, -54.67, 78.49), (63.17, 17.27, 38.09)],
        'LAMOL_KD':    [(32.75, -62.04, 86.04), (45.85, -64.91, 87.01), (44.80, -54.05, 92.93), (29.96, -71.79, 89.10), (48.10, -56.19, 92.72)],
        'PCLL':        [(31.18, -77.22, 93.48), (49.03, -67.69, 87.40), (40.69, -61.93, 93.45), (28.46, -74.38, 93.88), (48.84, -55.42, 92.89)],
        'Base+JLT':    [(98.63, -0.74, 21.56), (79.47, -17.08, 87.05), (74.91, -23.33, 93.45), (51.29, -36.30, 79.57), (73.06, -13.23, 79.85)],
        'L2KD+JLT':    [(97.06, -1.76, 17.26), (96.10, -2.20, 87.46), (96.36, -2.95, 95.02), (95.35, -1.00, 94.38), (93.82, -0.16, 92.79)],
        'LAMOL_KD+JLT': [(97.15, -0.83, 8.44), (95.36, -2.42, 84.13), (87.84, -8.10, 89.05), (94.72, -0.99, 92.16), (90.68, -1.03, 89.74)],
        'PCLL+JLT':    [(96.30, -0.32, 18.01), (94.32, -2.54, 87.33), (96.29, -3.29, 91.43), (95.42, -0.97, 94.94), (92.24, -0.10, 91.98)],
    },
    'Qwen3-0.6B': {
        'Joint': [98.66, 93.36, 96.66, 95.65, 96.50],
        'Base':        [(11.30, -90.94, 90.14), (12.73, -92.39, 91.44), (6.60, -90.57, 90.64), (6.98, -79.11, 73.01), (19.99, -89.91, 89.94)],
        'L2KD':        [(34.67, -67.41, 90.38), (55.78, -25.99, 75.11), (27.31, -63.83, 86.02), (32.90, -58.13, 73.79), (62.32, 9.90, 44.41)],
        'LAMOL_KD':    [(42.20, -59.61, 94.17), (52.44, -52.31, 97.42), (42.71, -59.83, 93.07), (26.45, -79.12, 95.30), (42.09, -60.48, 93.50)],
        'PCLL':        [(49.40, -55.33, 93.50), (52.99, -50.68, 97.78), (43.27, -59.81, 90.10), (31.29, -74.50, 96.14), (49.40, -55.33, 92.50)],
        'Base+JLT':    [(97.49, -1.73, 15.76), (75.26, -22.58, 86.36), (77.24, -21.31, 93.24), (52.55, -35.27, 78.87), (71.03, -18.32, 82.20)],
        'L2KD+JLT':    [(98.21, -0.15, 10.69), (94.93, -3.67, 86.40), (95.27, -4.40, 93.12), (95.58, -1.15, 94.98), (92.16, -1.21, 91.32)],
        'LAMOL_KD+JLT': [(97.88, -0.53, 13.95), (92.08, -6.82, 84.89), (90.47, -9.50, 92.07), (95.01, -1.87, 94.67), (91.47, -1.53, 90.21)],
        'PCLL+JLT':    [(96.23, -0.07, 17.69), (92.66, -5.00, 87.93), (95.22, -4.36, 92.36), (94.96, -1.13, 96.55), (89.56, -1.75, 90.11)],
    },
}

METHODS = ['Base', 'L2KD', 'LAMOL_KD', 'PCLL']
MODELS = list(TABLE14.keys())


def col(model, method, j):
    return np.array([TABLE14[model][method][d][j] for d in range(5)])


def line(c='-', n=96):
    print(c * n)


# Table 3
print('\n### Table 3: readout-recoverable fraction of the gap to joint training')
print('    total gap   G = A_joint - A_std ;  readout-recoverable  R = A_JLT - A_std ;  residual = A_joint - A_JLT')
line()
print('%-10s %-10s %8s %8s %8s %10s %10s %9s' %
      ('method', 'model', 'A_std', 'A_JLT', 'A_joint', 'gap G', 'residual', 'R/G %'))
line()
agg = {m: [] for m in METHODS}
for method in METHODS:
    for model in MODELS:
        std = col(model, method, 0).mean()
        jlt = col(model, method + '+JLT', 0).mean()
        joint = np.mean(TABLE14[model]['Joint'])
        g, res = joint - std, joint - jlt
        frac = 100 * (jlt - std) / g
        agg[method].append((std, jlt, joint, g, res, frac))
        print('%-10s %-12s %8.2f %8.2f %8.2f %10.2f %10.2f %8.1f%%' %
              (method, model, std, jlt, joint, g, res, frac))
    a = np.array(agg[method])
    print('%-10s %-12s %8.2f %8.2f %8.2f %10.2f %10.2f %8.1f%%   (mean)' %
          ('', 'ALL', a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
           a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean()))
    line('.')

# Table 18, top
print('\n### Table 18 (top): residual gap A_joint - A_JLT per stream')
print('    A large residual is not recovered by head re-alignment alone.')
line()
for method in METHODS:
    print('%s:' % method)
    print('  %-14s %s' % ('model', ' '.join('%10s' % d for d in DATASETS)))
    for model in MODELS:
        res = np.array(TABLE14[model]['Joint']) - col(model, method + '+JLT', 0)
        print('  %-14s %s' % (model, ' '.join('%10.2f' % v for v in res)))
    line('.')

# Table 18, bottom
print('\n### Table 18 (bottom): change in operational FWT under task-adaptive JLT')
print('    Under JLT both a_{i,i} and a_{i-1,i} are measured with an aligned head, so the')
print('    gain from creating an output interface for unseen labels is excluded.')
line()
for method in METHODS:
    print('%s:' % method)
    print('  %-14s %s' % ('model', ' '.join('%16s' % d for d in DATASETS)))
    for model in MODELS:
        f0 = col(model, method, 2)
        f1 = col(model, method + '+JLT', 2)
        print('  %-14s %s' % (model, ' '.join('%7.1f->%-7.1f' % (a, b)
                                              for a, b in zip(f0, f1))))
    allf0 = np.concatenate([col(m, method, 2) for m in MODELS])
    allf1 = np.concatenate([col(m, method + '+JLT', 2) for m in MODELS])
    print('  %-14s mean FWT %.2f -> %.2f  (delta %+.2f)' %
          ('', allf0.mean(), allf1.mean(), allf1.mean() - allf0.mean()))
    line('.')

print('\n  Relation extraction (TACRED, FewRel) vs. intent and topic streams:')
for method in METHODS:
    re_d, ot_d = [], []
    for model in MODELS:
        f0, f1 = col(model, method, 2), col(model, method + '+JLT', 2)
        re_d += [f1[0] - f0[0]]
        ot_d += [f1[1] - f0[1], f1[2] - f0[2], f1[4] - f0[4]]
        re_d += [f1[3] - f0[3]]
    print('   %-10s Tacred+Fewrel  %+7.2f     Banking+Clinc+Topic3 %+7.2f' %
          (method, np.mean(re_d), np.mean(ot_d)))
