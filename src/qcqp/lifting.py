"""Lifted vector z, its index map, the encoder z(state) and the decoder state(z).

Index map (claim M3-C01), 0-based half-open::

    k = 0,1,2:  vec(R_k) column-major [15k, 15k+9), v_k [15k+9, 15k+12), p_k [15k+12, 15k+15)
    k = 0,1,2:  b_k = [b_g; b_a]      [45+6k, 51+6k)
    e = 0,1, o = 63+30e:  vec(Y) [o,o+9), vec(Y') [o+9,o+18), V [o+18,o+21),
                          V' [o+21,o+24), P [o+24,o+27), P' [o+27,o+30)
    s                                 123
    anisotropic only:  rho_v,e [124+6e, 127+6e), rho_p,e [127+6e, 130+6e)   (n = 136)

Edge e joins keyframes (e, e+1) and uses bias b_e.  With M = R_i dRbar, w = J_R b_g,
c^v = J_nu b, c^p = J_rho b:  Y = M [w]_x, Y' = R_j [w]_x, V = M c^v, V' = R_j c^v,
P = M c^p, P' = R_j c^p, and (anisotropic) rho_v = M^T r_v, rho_p = M^T r_p.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixture import hat


@dataclass(frozen=True)
class Layout:
    n_kf: int = 3
    n_edges: int = 2
    aniso: bool = False

    # -- per-keyframe blocks ------------------------------------------------------------
    def R(self, k):
        return slice(15 * k, 15 * k + 9)

    def v(self, k):
        return slice(15 * k + 9, 15 * k + 12)

    def p(self, k):
        return slice(15 * k + 12, 15 * k + 15)

    @property
    def bias0(self):
        return 15 * self.n_kf

    def b(self, k):
        o = self.bias0 + 6 * k
        return slice(o, o + 6)

    def bg(self, k):
        o = self.bias0 + 6 * k
        return slice(o, o + 3)

    def ba(self, k):
        o = self.bias0 + 6 * k + 3
        return slice(o, o + 3)

    # -- per-edge liftings --------------------------------------------------------------
    def _o(self, e):
        return self.bias0 + 6 * self.n_kf + 30 * e

    def Y(self, e):
        return slice(self._o(e), self._o(e) + 9)

    def Yp(self, e):
        return slice(self._o(e) + 9, self._o(e) + 18)

    def V(self, e):
        return slice(self._o(e) + 18, self._o(e) + 21)

    def Vp(self, e):
        return slice(self._o(e) + 21, self._o(e) + 24)

    def P(self, e):
        return slice(self._o(e) + 24, self._o(e) + 27)

    def Pp(self, e):
        return slice(self._o(e) + 27, self._o(e) + 30)

    @property
    def s(self):
        return self._o(self.n_edges)

    @property
    def n_base(self):
        return self.s + 1

    def rho_v(self, e):
        assert self.aniso
        o = self.n_base + 6 * e
        return slice(o, o + 3)

    def rho_p(self, e):
        assert self.aniso
        o = self.n_base + 6 * e + 3
        return slice(o, o + 3)

    @property
    def n(self):
        return self.n_base + (6 * self.n_edges if self.aniso else 0)

    @property
    def bias_indices(self):
        return np.arange(self.bias0, self.bias0 + 6 * self.n_kf)

    # -- linear-form tensors of each block --------------------------------------------
    def vec_form(self, sl):
        idx = np.arange(self.n)[sl]
        f = np.zeros((len(idx), self.n))
        f[np.arange(len(idx)), idx] = 1.0
        return f

    def mat_form(self, sl):
        """(3,3,n) form of a column-major vec'd 3x3 block: entry (a,b) at start + a + 3b."""
        f = self.vec_form(sl)                     # 9 x n, row a + 3b
        return f.reshape(3, 3, self.n).transpose(1, 0, 2)

    def s_form(self):
        f = np.zeros(self.n)
        f[self.s] = 1.0
        return f


BASE = Layout()
ANISO = Layout(aniso=True)


@dataclass
class State:
    R: list
    v: list
    p: list
    b: list           #: project order [b_g; b_a] per keyframe

    def copy(self):
        return State([np.array(x) for x in self.R], [np.array(x) for x in self.v],
                     [np.array(x) for x in self.p], [np.array(x) for x in self.b])


def truth_state(inst) -> State:
    return State([t[0].copy() for t in inst.truth], [t[1].copy() for t in inst.truth],
                 [t[2].copy() for t in inst.truth], [inst.b_true.copy() for _ in inst.truth])


def edge_liftings(state: State, edge):
    """``(M, Y, Y', V, V', P, P')`` of one edge, from the state."""
    i, j = edge.i, edge.j
    b = state.b[i]
    M = state.R[i] @ edge.dR
    W = hat(edge.J_R @ b[0:3])
    cv, cp = edge.J_nu @ b, edge.J_rho @ b
    Rj = state.R[j]
    return M, M @ W, Rj @ W, M @ cv, Rj @ cv, M @ cp, Rj @ cp


def z_of_state(state: State, inst, layout: Layout = BASE, s: float = 1.0) -> np.ndarray:
    z = np.zeros(layout.n)
    for k in range(layout.n_kf):
        z[layout.R(k)] = state.R[k].flatten(order="F")
        z[layout.v(k)] = state.v[k]
        z[layout.p(k)] = state.p[k]
        z[layout.b(k)] = state.b[k]
    for e, edge in enumerate(inst.edges):
        M, Y, Yp, V, Vp, P, Pp = edge_liftings(state, edge)
        z[layout.Y(e)] = Y.flatten(order="F")
        z[layout.Yp(e)] = Yp.flatten(order="F")
        z[layout.V(e)] = V
        z[layout.Vp(e)] = Vp
        z[layout.P(e)] = P
        z[layout.Pp(e)] = Pp
    z[layout.s] = 1.0
    if layout.aniso:
        from .cost import edge_residual_forms
        for e, edge in enumerate(inst.edges):
            _, rv, rp = edge_residual_forms(inst, layout, e)
            M = state.R[edge.i] @ edge.dR
            z[layout.rho_v(e)] = M.T @ (rv @ z)
            z[layout.rho_p(e)] = M.T @ (rp @ z)
    return s * z


def project_so3(A):
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def state_of_z(z, layout: Layout = BASE, project: bool = False) -> State:
    """Decode by dividing every block by s (z and -z decode to the same state)."""
    z = np.asarray(z, dtype=float)
    s = z[layout.s]
    R = [z[layout.R(k)].reshape(3, 3, order="F") / s for k in range(layout.n_kf)]
    if project:
        R = [project_so3(x) for x in R]
    return State(R, [z[layout.v(k)] / s for k in range(layout.n_kf)],
                 [z[layout.p(k)] / s for k in range(layout.n_kf)],
                 [z[layout.b(k)] / s for k in range(layout.n_kf)])

