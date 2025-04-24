"""
Synopsis: Script to test computation of fixed-point equations using FPT
for subordination.
Author: Elvis Dohmatob, Arjun Subramonian

Run as `python -m examples.subordination' from within auto-fpt.
"""

from sympy import Symbol, MatrixSymbol, Identity, pprint, latex
import numpy as np
from fpt import calc
import pickle as pkl

# Input dimension and sample sizes
d = Symbol("d", integer=True, positive=True)
n = Symbol("n", integer=True, positive=True)
lambd = Symbol(r'\lambda', integer=True)
n1 = Symbol("n_1", integer=True, positive=True)  # p_1 * n
n2 = Symbol("n_2", integer=True, positive=True)  # p_2 * n
p1 = Symbol(r"p_1", positive=True)
p2 = Symbol(r"p_2", positive=True)
phi = Symbol(r"\phi", positive=True)

# Design Matrices
Z1 = MatrixSymbol("Z_1", n1, d)
Z2 = MatrixSymbol("Z_2", n2, d)
S1 = MatrixSymbol("S_1", d , d)  # Sqrt of covariance matrix for group 1
S2 = MatrixSymbol("S_2", d , d)  # Sqrt of covariance matrix for group 2
X1 = Z1@S1
X2 = Z2@S2

# Empirical Covariance Matrices
M1 = X1.T@X1
M2 = X2.T@X2
M = M1 + M2

# Load minimal linear pencil of resolvent R of M precomputed using NCAlgebra
R = (M + Identity(d)).inv()
with open('examples/pencils/subordination-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)

# Compute fixed-point equations for expected trace of R
row_idx = np.flatnonzero(u)[0]
col_idx = np.flatnonzero(v)[0]

variances = {Z1: 1 / (n * lambd), Z2: 1 / (n * lambd)}
eqns = calc(Q, random_matrices=["Z_1", "Z_2"], row_idx=row_idx, col_idx=col_idx,
            variances=variances, subs={n1: n * p1, n2: n * p2, d: n * phi})
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))
