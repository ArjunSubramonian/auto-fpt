"""
Synopsis: Script to test computation of fixed-point equations using free probability theory for anisotropic Marchenko-Pastur (MP) law.
Author: Elvis Dohmatob, Arjun Subramonian

Run as `python -m examples.anisotropic-MP' from within auto-fpt.
"""

from sympy import Symbol, MatrixSymbol, Identity, pprint, latex
import numpy as np
from fpt import calc
import pickle as pkl

# Form the design matrix.
n = Symbol("n", integer=True, positive=True)
d = Symbol("d", integer=True, positive=True)
Z = MatrixSymbol("Z", n, d)
Sigma_sqrt = MatrixSymbol("\Sigma_{sqrt}",d , d)  # Covariance matrix
phi = Symbol(r"\phi", positive=True)

X = Z@Sigma_sqrt  # Design matrix
expr = (X.T@X + Identity(d)).inv()  # Resolvent matrix

# Load minimal linear pencil precomputed using NCAlgebra
with open('examples/pencils/anisotropic-MP-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)

# Get the index of the one-hot entry in u, v.
row_idx, col_idx = np.argmax(u), np.argmax(v)

# Get free probability equations defining the limiting value of the
# trace of the resolvent.
eqns = calc(Q, random_matrices=["Z"], row_idx=row_idx, col_idx=col_idx, normalize="full", subs={d: n * phi})
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))
