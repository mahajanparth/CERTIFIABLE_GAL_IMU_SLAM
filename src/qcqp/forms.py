"""Linear and quadratic forms over the lifted vector z.

A *linear-form tensor* has shape ``(..., n)``: entry ``[..., :]`` is a coefficient row ``f``
with value ``f @ z``.  A *quadratic-form tensor* has shape ``(..., n, n)`` and every trailing
``n x n`` slice is symmetric, with value ``z^T A z``.  Products of linear forms are
symmetrised so every quadratic form is stored in the homogeneous symmetric form the claims
use (``z^T A_k z = b_k``).
"""
from __future__ import annotations

import numpy as np

_EPS = np.zeros((3, 3, 3))
for _a, _b, _c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPS[_a, _b, _c] = 1.0
    _EPS[_a, _c, _b] = -1.0


def sym(Q):
    return 0.5 * (Q + np.swapaxes(Q, -1, -2))


def outer(f, g):
    """Entrywise product of two linear-form tensors of the same leading shape."""
    f, g = np.broadcast_arrays(f, g)
    return sym(f[..., :, None] * g[..., None, :])


def matmul(A, B):
    """(p,q,n) x (q,r,n) -> (p,r,n,n)."""
    return sym(np.einsum("pqn,qrm->prnm", A, B))


def matvec(A, x):
    """(p,q,n) x (q,n) -> (p,n,n)."""
    return sym(np.einsum("pqn,qm->pnm", A, x))


def dot(a, b):
    """(3,n) . (3,n) -> (n,n)."""
    return sym(np.einsum("an,am->nm", a, b))


def cross(a, b):
    """(3,n) x (3,n) -> (3,n,n)."""
    return sym(np.einsum("ijk,jn,km->inm", _EPS, a, b))


def hat(w):
    """Skew matrix of a linear (3,n) or quadratic (3,n,n) form: [w]_x[a,b] = -eps_abc w_c."""
    return -np.einsum("abc,c...->ab...", _EPS, w)


def cmat_right(A, C):
    """linear (p,q,n) times constant (q,r) -> (p,r,n)."""
    return np.einsum("pqn,qr->prn", A, C)


def cvec(C, x):
    """constant (p,q) times linear (q,n) -> (p,n)."""
    return np.einsum("pq,qn->pn", C, x)


def lin_T(A):
    """transpose of a (p,q,n) linear matrix form."""
    return np.swapaxes(A, 0, 1)

