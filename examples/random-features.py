"""
Synopsis: Script to test computation of fixed-point equations using free probability theory for generalization error of random features model.
Author: Arjun Subramonian

Run as `python -m examples.random-features' from within auto-fpt.
"""

from sympy import Symbol, MatrixSymbol, Identity, pprint, latex, \
    Eq, solve, symbols, fraction, simplify, expand, factor, together
import numpy as np
from fpt import calc
import pickle as pkl

# Define the relevant random and deterministic matrices and scalars.
n = Symbol("n", integer=True, positive=True)
d = Symbol("d", integer=True, positive=True)
m = Symbol("m", integer=True, positive=True)

phi = Symbol(r"\phi", positive=True)
psi = Symbol(r"\psi", positive=True)
lambd = Symbol(r"\lambda", positive=True)
d = phi * n
m = d / psi

zeta = Symbol(r"\zeta", positive=True)
eta = Symbol(r"\eta", positive=True)
beta = Symbol(r"\beta", positive=True)

W0 = MatrixSymbol("W_0", m, d)
X = MatrixSymbol("X", d, n)
T0 = MatrixSymbol(r"\Theta_0", m, n)
variances = {X: 1 / d, W0: zeta / (m * lambd), T0: beta / (m * lambd)}

# Form the resolvent for Kinv.
F0 = W0 @ X + T0
Kinv = (F0.T @ F0 + Identity(n)).inv()

# Compute a minimal linear pencil for X^T * X * Kinv + Kinv using NCAlgebra.
# Q, (u, v) = compute_minimal_pencil(X.T @ X @ Kinv + Kinv)
with open('examples/pencils/random-features-pencil.pkl', 'rb') as f:
    Q, (u, v) = pkl.load(f)

pprint(Q)
print('u:', u)
print('v:', v)

# Get the index of the entry in u, v corresponding to X^T * X * Kinv.
XTXKinv_i = np.flatnonzero(u)[0]
XTXKinv_j = np.flatnonzero(v)[0]

# Compute the fixed-point equations for Kinv and X^T * X * Kinv using free probability theory.
eqns = calc(Q, random_matrices=[X, W0, T0], row_idx=XTXKinv_i, col_idx=XTXKinv_j, subs={}, variances=variances)
print(',\\\\\n'.join([latex(eqn) for eqn in eqns]))

# Get the index of the entry in u, v corresponding to Kinv.
Kinv_i = np.flatnonzero(u)[1]
Kinv_j = np.flatnonzero(v)[0]

# Empirical validation of fixed-point equations
print()
phi_val = 1.5
print('phi:', phi_val)
psi_val = 0.8
print('psi:', psi_val)
lambd_val = 0.1
print('lambd:', lambd_val)

# Empirical estimates of nonlinearity-related constants
z = np.random.normal(0, 1, 2000)
r = np.maximum(z, 0) - 1 / np.sqrt(2 * np.pi)
zeta_val = np.mean(z > 0) ** 2
print('zeta:', zeta_val)
eta_val = np.mean(r ** 2)
print('eta:', eta_val)
beta_val = eta_val - zeta_val
print('beta:', beta_val)

subbed_eqns = []
for eqn in eqns:
    subbed_eqns.append(eqn.subs(phi, phi_val).subs(psi, psi_val).subs(lambd, lambd_val).subs(zeta, zeta_val).subs(beta, beta_val))

# Iteratively solve fixed-point equations
eqn_sol = {eqn.lhs: 0 for eqn in eqns}
for _ in range(100):
    new_eqn_sol = {}
    for eqn in subbed_eqns:
        new_eqn_sol[eqn.lhs] = eqn.rhs.evalf(subs=eqn_sol)
    eqn_sol = new_eqn_sol

# Empirical estimates of tau1 and tau2
n = 200
d = int(phi_val * n)
m = int(d / psi_val)

X = np.random.normal(0, 1, (d, n))
W1 = np.random.normal(0, 1, (m, d))
F = np.maximum(W1 @ X / np.sqrt(d), 0) - 1 / np.sqrt(2 * np.pi)
Kinv = np.linalg.inv(F.T @ F / m + lambd_val * np.eye(n))
tau1_sol = np.trace(Kinv) / n
tau2_sol = np.trace(X.T @ X @ Kinv / d) / n

# Error of fixed-point equations
print()
for g, sol in eqn_sol.items():
    if (g.i, g.j) == (Kinv_i, Kinv_j):
        print('Error of tau1 estimate:', abs(sol / lambd_val - tau1_sol))
    elif (g.i, g.j) == (XTXKinv_i, XTXKinv_j):
        print('Error of tau2 estimate:', abs(sol / lambd_val - tau2_sol))

# Verify solutions for tau1 and tau2 satisfy coupled equations from Adlam and Pennington (2020)
print()
def eval_tau_rel(phi, psi, eta, zeta, lambd, tau_1, tau_2):
    return phi * (zeta * tau_2 * tau_1 + phi * (tau_2 - tau_1)) + zeta * tau_1 * tau_2 * psi * (lambd * tau_1 - 1), \
    (tau_2 - tau_1) * phi * (zeta * (tau_2 - tau_1) + eta * tau_1) - zeta * tau_1 * tau_2 * (lambd * tau_1 - 1)

eq1_err, eq2_err = eval_tau_rel(phi=phi_val, psi=psi_val, eta=eta_val, zeta=zeta_val, lambd=lambd_val, tau_1=tau1_sol, tau_2=tau2_sol)
print('Error for (S67):', eq1_err)
print('Error for (S68):', eq2_err)

# Analytical validation of fixed-point equations

# Solve symbolically for tau1 and tau2
tau_1 = Symbol(r"\tau_1", positive=True)
tau_2 = Symbol(r"\tau_2", positive=True)

# Replace entries of G with new symbols to make free_symbols
# return all variables in equation
var_replacements = symbols(["v" + str(i) for i in range(1, len(eqns) + 1)])
G = {}
subs = {}
eliminate = []
for it, eqn in enumerate(eqns):
    i = eqn.lhs.i
    j = eqn.lhs.j
    G[(i, j)] = eqn

    if (i, j) == (Kinv_i, Kinv_j):
        subs[eqn.lhs] = lambd * tau_1
    elif (i, j) == (XTXKinv_i, XTXKinv_j):
        subs[eqn.lhs] = lambd * tau_2
    else:
        subs[eqn.lhs] = var_replacements[it]
    
    if (i, j) not in [(Kinv_i, Kinv_j), (XTXKinv_i, XTXKinv_j)]:
        eliminate.append(subs[eqn.lhs])

# Rewrite equations by sorting, moving all variables to lhs, and expanding
print('Rewriting system of equations...')
new_eqns = []
for it, eqn in enumerate([G[(Kinv_i, Kinv_j)], G[(XTXKinv_i, XTXKinv_j)]] + sorted(eqns, key=lambda x: (x.lhs.i, x.lhs.j))):
    i = eqn.lhs.i
    j = eqn.lhs.j
    
    if it > 1 and (i, j) in [(Kinv_i, Kinv_j), (XTXKinv_i, XTXKinv_j)]:
        continue
    
    n, d = fraction(eqn.rhs)
    re_eqn = (d * eqn.lhs - n).subs(subs)
    new_eqns.append(expand(simplify(re_eqn)))
    pprint(Eq(new_eqns[-1], 0))
print()

# Eliminate all variables except tau1 and tau2
print('Eliminating variables...')
subbed_eqns = new_eqns
for eqn in new_eqns:
    elim_cands = [v for v in eliminate if v in eqn.free_symbols]
    if len(elim_cands) == 0:
        continue
    
    tgt = elim_cands[0]
    subs = {
        tgt: solve(eqn, tgt)[0]
    }
    for it, seqn in enumerate(subbed_eqns):
        subbed_eqns[it] = simplify(seqn.subs(subs))

nonzero_eqns = []
for eqn in subbed_eqns:
    if eqn != 0:
        nonzero_eqns.append(fraction(together(factor(expand(eqn))))[0])
        pprint(Eq(nonzero_eqns[-1], 0))
        print(latex(Eq(nonzero_eqns[-1], 0)))
print()

print('Demonstrating equivalence to (S67) and (S68)...')
p1 = simplify(expand(nonzero_eqns[0].subs(beta, eta - zeta)))
p1_new = p1 / phi
h1 = expand((tau_2 - tau_1) * phi * (zeta * (tau_2 - tau_1) + eta * tau_1) - zeta * tau_1 * tau_2 * (lambd * tau_1 - 1))
pprint(p1_new)
print(latex(p1_new))
print("Matches (S68):", p1_new - h1 == 0)

p2 = simplify(expand(nonzero_eqns[1].subs(beta, eta - zeta)))
p2_new = simplify(p1 * (-psi / phi) + p2 / lambd)
h2 = expand(phi * (zeta * tau_2 * tau_1 + phi * (tau_2 - tau_1)) + zeta * tau_1 * tau_2 * psi * (lambd * tau_1 - 1))
pprint(p2_new)
print(latex(p2_new))
print("Matches (S67):", p2_new - h2 == 0)