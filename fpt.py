"""
Synopsis: Generic tools for free probability theory of rectangular random
Gaussian matrices.
Author: Elvis Dohmatob, Arjun Subramonian
"""
import os
import warnings
import argparse
from importlib import reload

import pickle
import numpy as np

from sympy import (Function, symbols, Matrix, matrices, MatrixSymbol,
                   simplify, simplify, factor, Eq, pretty_print)
from sympy.matrices.expressions.matexpr import MatrixElement

import free_proba_utils
reload(free_proba_utils)

def find_duplicates(q_inv, Q):
    """Gather entries of given matrix into equivalence classes of pairs of
    indices with same value q_inv[i, j]."""
    assert q_inv.shape == Q.blocks.shape
    n0 = q_inv.shape[0]
    classes = {}
    for i in range(n0):
        for j in range(n0):
            term = simplify(q_inv[i, j])
            shape = Q.blocks[i, j].shape
            if len(term.free_symbols) > 0:
                key = term, shape
                classes[key] = classes.get(key, []) + [(i, j)]
    return dict((term, indices) for (term, _), indices in classes.items())


def remove_duplicates(G, expr, classes):
    if not classes:
        return expr
    for indices in classes.values():
        i0, j0 = indices[0]
        for i, j in indices[1:]:
            expr = expr.subs(G[i, j], G[i0, j0])
    return simplify(expr)

def construct_fixed_point_equations(G, diff_inv, row_idx, col_idx, classes=None, subs=None):
    expr = diff_inv[row_idx, col_idx]
    expr = remove_duplicates(G=G, expr=expr, classes=classes)
    eqs = [Eq(G[row_idx, col_idx], expr)]

    # Gather subtree of fixed-point equations contraining the target expr
    prev_terms = set()
    while True:
        flag = False
        for a in expr.atoms(MatrixElement):
            if (a.args[1], a.args[2]) not in prev_terms:
                eqs.append(Eq(G[a.args[1], a.args[2]],
                              diff_inv[a.args[1], a.args[2]]))
                expr = expr.subs(G[a.args[1], a.args[2]],
                                 diff_inv[a.args[1], a.args[2]])
                prev_terms.add((a.args[1], a.args[2]))
                flag = True
        if not flag:
            break
    tmp = list(set(eqs[1:]))
    if eqs[0] in tmp:
        tmp.remove(eqs[0])
    eqs = eqs[:1] + tmp

    subs = {k: v for k, v in subs.items() if k is not None}
    if subs is not None:
        eqs = [eq.subs(subs) for eq in eqs]
    
    eqs = [factor(simplify(eq)) for eq in eqs]
    return expr, eqs


def calc(Q, random_matrices=None, row_idx=None, col_idx=None, 
         verbose=0, normalize=None, subs=None, variances=None,
         display=pretty_print):
    """
    Parameters
    ----------
    Q: BlockMatrix
        Linear pencil
    row_idx: int (required)
        Row index of sought-for quantity withing pencil
    col_idx: int (required)
        Column index of sought-for quantity withing pencil
    """
    if None in [row_idx, col_idx]:
        raise ValueError("Both row_idx and col_idx required in function call!")

    n0 = Q.blocks.shape[0]
    if verbose:
        print("Q = ")
        display(Q)
        print("Size of pencil: %i x %i" % (n0, n0))

    free_symbols = list(Q.free_symbols)
    free_symbols = dict(zip(map(str, free_symbols), free_symbols))
    random_matrices = list(random_matrices)
    rands = []
    for rmat in random_matrices:
        if isinstance(rmat, str) and rmat in free_symbols:
            rands.append(free_symbols[rmat])
        else:
            rands.append(rmat)

    # This corrective piece of code only exist because the input pencils
    # are usually as if n = lambda = 1
    if variances is None:
        n, lambd, d = [free_symbols.get(symb, symbols(symb)) for symb in "n lambda d".split()]
        variances = {}
        if normalize is not None:
            for rmat in rands:
                rmat_name = str(rmat)
                if rmat_name in ["Z_1", "Z_2", "Z", "S"]:
                    if normalize == "full":
                        if rmat_name == "S":
                            var = 1 / d
                        else:
                            var = 1 / (n * lambd)
                    elif normalize == "sample size":
                        if rmat_name == "S":
                            var = 1 / d
                        else:
                            var = 1 / n
                    else:
                        raise ValueError(normalize)
                    variances[rmat] = var

    if variances and verbose:
        print("Variance of entries of each random matrix:")
        for k, v in variances.items():
            print("\t%s: %s" % (k, v))

    if verbose >= 2:
        for rand in rands:
            if rand not in free_symbols.values():
                warnings.warn(
                "Random matrix %s doesn't appear in pencil" % rand)
    rands = list(set(rands).intersection(free_symbols.values()))

    matrix_symbols = dict((k, v)
    for k, v in free_symbols.items() if isinstance(v, MatrixSymbol))
    matrix_symbols.update(dict(("%s_T" % k, v.T if v in random_matrices else v)
    for k, v in matrix_symbols.items()))

    print("\nPreparing matrix-to-scalar identification...")
    matrix_to_scalar_map = dict((v, symbols("x%d" % idx))
    for idx, v in enumerate(matrix_symbols.values()))
    tmp = {}
    for k, v in matrix_symbols.items():
        if k.endswith("_T"):
            if v.T not in random_matrices:
                tmp[v.T] = matrix_to_scalar_map[v]
    matrix_to_scalar_map.update(tmp)

    def _print_identification(exclude_random=False):
        treated = []
        for k, v in matrix_to_scalar_map.items():
            if v in treated:
                continue
            if exclude_random and (k in rands or k.T in rands):
                continue
            print("\t%s --> %s" % (v, k))
            treated.append(v)

    if verbose:
        _print_identification()
    print("\nSplitting the input pencil...")
    Qx, F = free_proba_utils.decompose_Q(Q, rands=rands)
    if verbose:
        print("Qx = ")
        display(Qx)
        print("F = ")
        display(F)

    print("\nScalarizing matrices...")
    q = free_proba_utils.scalarize_block_matrix(Q, matrix_to_scalar_map)
    qx = free_proba_utils.scalarize_block_matrix(Qx, matrix_to_scalar_map)
    f = free_proba_utils.scalarize_block_matrix(F, matrix_to_scalar_map)
    if verbose:
        print("qx = ")
        display(qx)
        print("f = ")
        display(f)

    print("\nForming symmetrization of block matrices...")
    Q_ = free_proba_utils.symmetrize_block_matrix(Q)
    Qx_ = free_proba_utils.symmetrize_block_matrix(Qx)
    if verbose:
        print("Q_ = ")
        display(Q_)

    print("\nChecking that pencil makes sense...")
    q_inv = free_proba_utils.inv_heuristic(q)
    print("q_inv[%i, %i] = " % (row_idx, col_idx))
    display(q_inv[row_idx, col_idx])
    print("\nWhere")
    _print_identification()

    print("\nIdentifying indices of entries which must be equal in G...")
    classes = find_duplicates(q_inv, Q)
    if verbose:
        for val, indices in classes.items():
            print("\t%s: %s" % (indices, val))

    # Setup sough-for matrix of normalized traces
    print("\nInferring relevant mask...")
    mask = np.ones(q.shape, dtype=int)
    mask[np.array(q_inv, dtype=object) == 0] = 0
    mask = Matrix(mask)  # sparsity structure of G, infered from that of Q
    if verbose:
        display(mask)

    print("\nPreparing G matrix...")
    G = MatrixSymbol("G", n0, n0).as_explicit()
    G = matrices.dense.matrix_multiply_elementwise(G, mask)
    G_ = free_proba_utils.symmetrize_matrix(G)

    # Compute R-transform
    print("\nComputing R-transform R = R(G) matrix...")
    r_ = free_proba_utils.R_transform(Qx_, G_, rands=rands,
                                          variances=variances)
    r_.simplify()
    if verbose:
        print("r_ = ")
        display(r_)

    # Invert difference of R-transform and constant matrices
    print("\nComputing F - R(G) matrix...")
    r = r_[:n0, n0:].T
    diff = f - r
    diff = simplify(diff)

    print("\nComputing inverse...")
    diff_inv = free_proba_utils.inv_heuristic(diff)

    print("\nComputing fixed-point equations...")
    _, eqs = construct_fixed_point_equations(G=G, diff_inv=diff_inv,
                                                row_idx=row_idx,
                                                col_idx=col_idx,
                                                classes=classes,
                                                subs=subs)

    print("\nMatricizing equations...")
    
    print("\nThe fixed-point equations are:\n")
    returned_eqs = []
    for eq in eqs:
        scalar_to_matrix_map = {v : k for k, v in matrix_to_scalar_map.items()}
        if eq.rhs.free_symbols.intersection(set(matrix_to_scalar_map.values())):
            matrix_rhs = simplify(free_proba_utils.matricize_expr(eq.rhs, scalar_to_matrix_map))
            returned_eqs.append(Eq(eq.lhs, Function(r'trbar')(matrix_rhs)))
            display(returned_eqs[-1])
        else:
            returned_eqs.append(eq)
            display(eq)
        print("\n")

    return returned_eqs

if __name__ == "__main__":
    usage1 = 'python fpt.py --pencil-file test.pkl --i 1 --j -1'
    usage1 += ' --normalize "full" --verbose 1'
    usage2 = 'python fpt.py --pencil-file test.pkl --i 1'
    usage2 += ' --j 9 --normalize "sample size" --verbose 1'
    usage3  = 'python fpt.py --pencil-file test.pkl --i 1 --j 10'
    usage3 += ' --random-matrix Z_1 --random-matrix Z_2 --random-matrix S '
    usage3 += '--verbose 1 --normalize "full"'
    parser = argparse.ArgumentParser(
    description='Command-line OVPTR',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''Example usage:
    (1) %s
    (2) %s
    (3) %s
    ''' % (usage1, usage2, usage3))
    parser.add_argument("--pencil-file", type=str, required=True,
                        help="path to .pkl file containing precomputed pencil")
    parser.add_argument("--i", type=int, required=True,
                        help="sought-for normalized trace corresponds to entry"
                             " (i, j) of inverse of pencil")
    parser.add_argument("--j", type=int, required=True,
                        help="sought-for normalized trace corresponds to entry"
                             " (i, j) of inverse of pencil")
    parser.add_argument("--random-matrix", type=str, action="append",
                        default=[],
                        help="specify a random matrix appearing in the pencil")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2],
                        help="specify verbosity level")
    parser.add_argument("--normalize", default=None,
                        choices=["full", "sample size"])
    args = parser.parse_args()
    args_dict = vars(args)

    # Read-in the input file containing the linear pencil
    pencil_file = os.path.abspath(args.pencil_file)
    print("\nLoading input pencil file %s ..." % pencil_file)
    with open(pencil_file, "rb") as inf:
        Q = pickle.load(inf)

    args_dict.pop("pencil_file")
    args_dict["random_matrices"] = args_dict.pop("random_matrix")
    args_dict["row_idx"] = args_dict.pop("i")
    args_dict["col_idx"] = args_dict.pop("j")
    calc(Q=Q, **args_dict)
