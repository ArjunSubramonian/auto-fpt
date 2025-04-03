"""
Synopsis: Helper functions for FPT calculations (e.g R-transform, etc.).
Author: Elvis Dohmatob, Arjun Subramonian
"""
from sympy import (Matrix, Identity, ZeroMatrix, BlockMatrix,
                   block_collapse, inv_quick, simplify, factor, sqrt, 
                   MatrixExpr, Pow, Inverse, MatPow, Add, Mul, MatAdd, MatMul, Transpose)
from sympy.matrices.expressions.matexpr import Transpose
import numpy as np


def is_zero(stuff):
    if hasattr(stuff, "blocks"):
        for block in stuff.blocks:
            if not is_zero(block):
                return False
        return True
    else:
        return 0 * stuff == stuff

def scalarize_block_matrix(M, var_bindings):
    """Converts a given block matrix in to scalar form in which every named
    variable matrix (block of M) is replaced / bound to a given named scalar
    variable, i.e Z1 --> z1, C1 --> c1, ..."""
    blocks = M.blocks
    A = np.zeros(blocks.shape, dtype=object)
    matrices = [var for var in var_bindings if isinstance(var, Transpose)]
    matrices += [var for var in var_bindings if not isinstance(var, Transpose)]
    for i in range(blocks.shape[0]):
        for j in range(M.blocks.shape[1]):
            block = blocks[i, j]
            p, q = block.shape
            I = Identity(p)
            if is_zero(block):
                continue
            elif block == I:
                A[i, j] = 1
            elif -block == I:
                A[i, j] = -1
            else:
                for var in matrices:
                    binding = var_bindings[var]
                    if var.shape != block.shape:
                        continue
                    elif var in block.free_symbols or var.T in block.free_symbols:
                        if p == q:
                            A[i, j] = block[0, 0].subs(sqrt(var)[0, 0], sqrt(binding))
                            A[i, j] = A[i, j].subs(var[0, 0], binding)
                        else:
                            A[i, j] = block[0, 0].subs(var[0, 0], binding)
    return Matrix(A)


def decompose_Q(Q, rands):
    """rands should contain all the random quantities (symbolic matrices) that
    appear as blocks of Q."""
    rands = rands + [item.T for item in rands]
    rands = rands + [-item for item in rands]
    rows = []
    for i in range(Q.blocks.shape[0]):
        row = []
        for j in range(Q.blocks.shape[1]):
                block = Q.blocks[i, j]
                # if block in rands:
                if len([rand for rand in rands if rand in block.free_symbols]):
                    item = -block
                else:
                    item = ZeroMatrix(*block.shape)
                row.append(item)
        rows.append(row)
    Qx = BlockMatrix(rows)
    F_ = block_collapse(Q + Qx)
    return Qx, F_

def R_transform(Qx, G, rands, variances={}):
    """
    Computes the R-transform of Qx evaluated at G, i.e

        varphi(G \otimes 1 - Qx)^{-1}.

    The output is an n0xn0 complex matrix with

      Rij = sum_{k,L} sigma(i,k;L,j) * alpha_k * G[k,L],

    where the scalar sigma(i,k;L,j) is the covariance between the entries of
    block (i, k) and block (L, j) of Qx, and alpha_k is the length of block
    (k, L).

    We work in the setting where sigma(i,k;L,j) in {-var, 0, var}. For example,
    typically, var = 1 / (n * lambd)
    """
    from sympy.stats import Normal, E
    mapping = {}
    for i, Z in enumerate(rands):
        std = sqrt(variances.get(Z, 1))
        z = Normal("z_%s" % i, 0, std)
        mapping[Z] = z
        mapping[Z.T] = z
    qx = scalarize_block_matrix(Qx, mapping)
    n0 = Qx.blocks.shape[0]
    assert G.shape == (n0, n0)
    R = np.zeros((n0, n0), dtype="object")
    for i in range(n0):
        for j in range(n0):
            for k in range(n0):
                if k == j:
                    continue
                block1 = Qx.blocks[i, k]
                if is_zero(block1):
                    continue
                for L in range(n0):
                    if L == i:
                        continue
                    if (i, k) == (L, j):
                        continue
                    block2 = Qx.blocks[L, j]
                    if is_zero(block2):
                        continue
                    if block2.shape != block1.shape[::-1]:
                        continue
                    cov = E(qx[i, k] * qx[L, j])
                    assert block2 not in [block1, -block1]
                    shape = Qx.blocks[k, L].shape
                    assert shape[1] == shape[0]
                    m = shape[0]
                    R[i, j] += cov * G[k, L] * m
    return Matrix(R)


def symmetrize_matrix(q):
    n, m = q.shape
    q = np.array(q)
    q_ = np.zeros((n + m, n + m), dtype=object)
    q_[:m, -n:] = q
    q_[-n:, :m] = q.T
    return Matrix(q_)


def symmetrize_block_matrix(Q):
    adj = []
    for i in range(Q.T.blocks.shape[0]):
        adj_row = []
        for j in range(Q.blocks.shape[1]):
            adj_row.append(ZeroMatrix(Q.T.blocks[i, 0].shape[0],
            Q.blocks[0, j].shape[1]))
        for j in range(Q.T.blocks.shape[1]):
            adj_row.append(Q.T.blocks[i, j])
        adj.append(adj_row)

    for i in range(Q.blocks.shape[0]):
        adj_row = []
        for j in range(Q.blocks.shape[1]):
            adj_row.append(Q.blocks[i, j])
        for j in range(Q.T.blocks.shape[1]):
            adj_row.append(ZeroMatrix(Q.blocks[i, 0].shape[0],
            Q.T.blocks[0, j].shape[1]))
        adj.append(adj_row)
    return BlockMatrix(adj)


def inv_heuristic(diff):
    """Heuristic computation of the inverse of a of a matrix of the form

        X = [A B]
            [C D]

    where A and D are sparse.
    """
    n0 = diff.shape[0]
    if n0 <= 5:
        return diff.inv()
    assert diff.shape[1] == n0
    m = n0 // 2
    A = diff[:m, :m]
    B = diff[:m, m:]
    C = diff[m:, :m]
    D = diff[m:, m:]
    Ainv = simplify(inv_quick(A)).applyfunc(factor)
    sc = D - C@Ainv@B
    scinv = simplify(inv_quick(sc)).applyfunc(factor)
    A_ = Ainv + Ainv@B@scinv@C@Ainv
    B_ = -Ainv@B@scinv
    C_ = -scinv@C@Ainv
    D_ = scinv
    inv = BlockMatrix([[A_, B_], [C_, D_]]).as_explicit()
    return simplify(inv).applyfunc(factor)


def matricize_expr(expr, conv_map):
    if expr.func == Pow:
        subexpr = matricize_expr(expr.args[0], conv_map)
        if issubclass(type(subexpr), MatrixExpr):
            if expr.args[1] == -1:
                return Inverse(subexpr, -1)
            return MatPow(subexpr, expr.args[1])
        return expr
    elif expr.func == Add:
        mat_args = []
        scal_args = []
        for a in expr.args:
            subexpr = matricize_expr(a, conv_map)
            if issubclass(type(subexpr), MatrixExpr):
                mat_args.append(subexpr)
            else:
                scal_args.append(subexpr)
        if len(mat_args) == 0:
            return expr
        scal_args = [a * Identity(mat_args[0].shape[0]) for a in scal_args]
        return MatAdd(*(mat_args + scal_args))
    elif expr.func == Mul:
        args = [matricize_expr(a, conv_map) for a in expr.args]
        return MatMul(*args)
    elif expr in conv_map:
        new_expr = conv_map[expr]
        if new_expr.func == Transpose:
            return new_expr.args[0]
        return new_expr
    else:
        return expr