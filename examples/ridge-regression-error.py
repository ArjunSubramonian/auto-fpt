"""
Synopsis: Script to test computation of fixed-point equations using FPT
for generalization error of ridge regression.
Author: Arjun Subramonian

Run as `python -m examples.ridge-regression-error' from within auto-fpt.
"""

from sympy import Symbol, MatrixSymbol, Identity, pprint, latex
import numpy as np
from fpt import calc
import pickle as pkl

# Define the relevant random and deterministic matrices and scalars.
n = Symbol(r'n', integer=True, positive=True)
d = Symbol(r'd', integer=True, positive=True)
Z = MatrixSymbol(r'Z', n, d)
sqrt_S = MatrixSymbol(r'\Sigma_{sqrt}', d, d)
S = MatrixSymbol(r'\Sigma', d, d)
T = MatrixSymbol(r'\Theta', d, d)
phi = Symbol(r"\phi", positive=True)

# Form the resolvent for the bias component of the generalization error.
inv_term = (sqrt_S * Z.T * Z * sqrt_S + Identity(d)).inv()
bias_expr = inv_term * T * inv_term * S

# Compute a minimal linear pencil using NCAlgebra.
# Q, (u, v) = compute_minimal_pencil(bias_expr)
with open('examples/pencils/ridge-regression-error-bias-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)
print(latex(Q))

# Get the index of the one-hot entry in u, v.
i = np.flatnonzero(u)[0]
j = np.flatnonzero(v)[0]

# Compute the fixed-point equations for the bias term using FPT
eqns = calc(Q, random_matrices='Z', row_idx=i, col_idx=j, normalize='full',
            subs={d: n * phi})
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))

# Form the resolvent for the variance component of the generalization error.
var_expr = inv_term * (sqrt_S * Z.T * Z * sqrt_S) * inv_term * S

# Compute a minimal linear pencil using NCAlgebra.
# Q, (u, v) = compute_minimal_pencil(var_expr)
with open('examples/pencils/ridge-regression-error-var-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)
print(latex(Q))

# Get the index of the one-hot entry in u, v.
i = np.flatnonzero(u)[0]
j = np.flatnonzero(v)[0]

# Compute the fixed-point equations for variance term using FPT.
eqns = calc(Q, random_matrices='Z', row_idx=i, col_idx=j, normalize='full',
            subs={d: n * phi})
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))
