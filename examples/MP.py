"""
Synopsis: Script to test computation of fixed-point equations using free probability theory for Marchenko-Pastur (MP) law.
Author: Arjun Subramonian, Elvis Dohmatob

Run as `python -m examples.MP' from within auto-fpt.
"""


from sympy import Symbol, MatrixSymbol, Identity, pprint, latex
import numpy as np
from fpt import calc
import pickle as pkl

# Form the design matrix.
n = Symbol(r'n', integer=True, positive=True)
d = Symbol(r'd', integer=True, positive=True)
Z = MatrixSymbol(r'Z', n, d)
phi = Symbol(r"\phi", positive=True)

# Form the expression of the resolvent.
MP_expr = (Z.T * Z + Identity(d)).inv()

# Compute a minimal linear pencil using NCAlgebra.
# Q, (u, v) = compute_minimal_pencil(MP_expr)
with open('examples/pencils/MP-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)
print(latex(Q))

# Get the index of the one-hot entry in u, v.
i = np.flatnonzero(u)[0]
j = np.flatnonzero(v)[0]

# Get free probability equations defining the limiting value of the
# trace of the resolvent. The normalize="full" option allows us to apply
# the variance 1/(n*lambda) of the entries of the random matrix Z.
eqns = calc(Q, random_matrices='Z', row_idx=i, col_idx=j, normalize='full', subs={d: n * phi}, verbose=1)
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))