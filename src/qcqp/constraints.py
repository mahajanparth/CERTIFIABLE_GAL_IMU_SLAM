"""Every constraint family of the M3 QCQP as explicit symmetric matrices ``z^T A_k z = b_k``.

Notation per edge e = (i, j), bias b = b_i:  M = R_i dRbar, w = J_R b_g, c^v = J_nu b,
c^p = J_rho b.  "Both sides" means the i-side liftings (M; Y, V, P) and the j-side liftings
(R_j; Y', V', P').

Primary
  P0  s^2 = 1                                                       (b = 1, the only b != 0)
  P1  R_k^T R_k = s^2 I                     (upper triangle, 6 per keyframe)
  P2  R^(a) x R^(b) = s R^(c), (a,b,c) cyclic (column cross products, 9 per keyframe)
  P3  M^T Y = [w]_x s, R_j^T Y' = [w]_x s, M^T V = c^v s, R_j^T V' = c^v s,
      M^T P = c^p s, R_j^T P' = c^p s      (30 per edge)
Redundant
  RA1 R_k R_k^T = s^2 I                     (6 per keyframe)
  RB1 Y w = 0, Y' w = 0                     (6 per edge)
  RB2 Y^T Y = |w|^2 I - w w^T, also Y'      (12 per edge)
  RC1 |V|^2 = |c^v|^2, |P|^2 = |c^p|^2, both sides       (4 per edge)
  RG  Y^T V = [w]_x^T c^v, Y^T P = [w]_x^T c^p, V.P = c^v.c^p, both sides (14 per edge)
  RL  s Y = M [w]_x, s V = M c^v, s P = M c^p, both sides (30 per edge)
  RH  (A e_a) x L = A (e_a x E) for each lifted column L = A E, A in {M, R_j} (90 per edge)
  RS  Y M^T = [M w]_x, Y' R_j^T = [R_j w]_x (18 per edge)
Anisotropic only (n = 136)
  P4  M rho_v = s r_v, M rho_p = s r_p      (6 per edge)
  RR  |rho_v|^2 = |r_v|^2, |rho_p|^2 = |r_p|^2, rho_v.rho_p = r_v.r_p (3 per edge)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import forms as Fm
from .cost import edge_residual_forms
from .lifting import BASE, Layout

PRIMARY = ("P0", "P1", "P2", "P3")
REDUNDANT = ("RA1", "RB1", "RB2", "RC1", "RG", "RL", "RH", "RS")
ANISO_PRIMARY = ("P4",)
ANISO_REDUNDANT = ("RR",)
CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_UPPER = [(a, b) for a in range(3) for b in range(a, 3)]


@dataclass(frozen=True)
class Constraint:
    family: str
    name: str
    A: sp.csr_matrix
    b: float

    def residual(self, z) -> float:
        return float(z @ (self.A @ z)) - self.b


def _mk(family, name, Q, b=0.0):
    Q = 0.5 * (Q + Q.T)
    return Constraint(family, name, sp.csr_matrix(Q), float(b))


def _emit(out, family, prefix, Qt, b=0.0):
    """Emit every entry of a quadratic-form tensor of shape (..., n, n)."""
    lead = Qt.shape[:-2]
    for idx in np.ndindex(*lead):
        out.append(_mk(family, f"{prefix}{list(idx)}", Qt[idx], b))


class _Edge:
    """Linear-form tensors of one edge."""

    def __init__(self, inst, layout: Layout, e: int):
        ed = inst.edges[e]
        L = layout
        self.e = e
        bf = L.vec_form(L.b(ed.i))
        self.M = Fm.cmat_right(L.mat_form(L.R(ed.i)), ed.dR)
        self.Rj = L.mat_form(L.R(ed.j))
        self.w = Fm.cvec(ed.J_R, bf[0:3])
        self.cv = Fm.cvec(ed.J_nu, bf)
        self.cp = Fm.cvec(ed.J_rho, bf)
        self.W = Fm.hat(self.w)
        self.sides = {
            "i": (self.M, L.mat_form(L.Y(e)), L.vec_form(L.V(e)), L.vec_form(L.P(e))),
            "j": (self.Rj, L.mat_form(L.Yp(e)), L.vec_form(L.Vp(e)), L.vec_form(L.Pp(e))),
        }


def build(inst, layout: Layout = BASE) -> dict[str, list[Constraint]]:
    """All families, keyed by family name, in a fixed order."""
    L = layout
    s = L.s_form()
    fam: dict[str, list[Constraint]] = {f: [] for f in PRIMARY + REDUNDANT}
    if L.aniso:
        fam.update({f: [] for f in ANISO_PRIMARY + ANISO_REDUNDANT})
    I3 = np.eye(3)

    Q = np.zeros((L.n, L.n))
    Q[L.s, L.s] = 1.0
    fam["P0"].append(_mk("P0", "s^2", Q, 1.0))
    ss = Fm.outer(s, s)
    for k in range(L.n_kf):
        R = L.mat_form(L.R(k))
        RtR = Fm.matmul(Fm.lin_T(R), R)
        RRt = Fm.matmul(R, Fm.lin_T(R))
        for a, b in _UPPER:
            fam["P1"].append(_mk("P1", f"R{k}^T R{k}[{a},{b}]", RtR[a, b] - I3[a, b] * ss))
            fam["RA1"].append(_mk("RA1", f"R{k} R{k}^T[{a},{b}]", RRt[a, b] - I3[a, b] * ss))
        for a, b, c in CYCLIC:
            cr = Fm.cross(R[:, a], R[:, b]) - Fm.outer(s, R[:, c])
            _emit(fam["P2"], "P2", f"R{k} col{a}x col{b}-s col{c}", cr)

    for e in range(L.n_edges):
        E = _Edge(inst, L, e)
        w, cv, cp, W = E.w, E.cv, E.cp, E.W
        for side, (A, Y, V, P) in E.sides.items():
            tag = f"e{e}.{side}"
            # P3
            _emit(fam["P3"], "P3", f"{tag} A^T Y-[w]s",
                  Fm.matmul(Fm.lin_T(A), Y) - Fm.outer(W, s))
            _emit(fam["P3"], "P3", f"{tag} A^T V-c^v s", Fm.matvec(Fm.lin_T(A), V) - Fm.outer(cv, s))
            _emit(fam["P3"], "P3", f"{tag} A^T P-c^p s", Fm.matvec(Fm.lin_T(A), P) - Fm.outer(cp, s))
            # RB1
            _emit(fam["RB1"], "RB1", f"{tag} Y w", Fm.matvec(Y, w))
            # RB2
            YtY = Fm.matmul(Fm.lin_T(Y), Y)
            ww = Fm.dot(w, w)
            for a, b in _UPPER:
                fam["RB2"].append(_mk("RB2", f"{tag} Y^TY[{a},{b}]",
                                      YtY[a, b] - I3[a, b] * ww + Fm.outer(w[a], w[b])))
            # RC1
            fam["RC1"].append(_mk("RC1", f"{tag} |V|^2", Fm.dot(V, V) - Fm.dot(cv, cv)))
            fam["RC1"].append(_mk("RC1", f"{tag} |P|^2", Fm.dot(P, P) - Fm.dot(cp, cp)))
            # RG
            _emit(fam["RG"], "RG", f"{tag} Y^T V",
                  Fm.matvec(Fm.lin_T(Y), V) - Fm.matvec(Fm.lin_T(W), cv))
            _emit(fam["RG"], "RG", f"{tag} Y^T P",
                  Fm.matvec(Fm.lin_T(Y), P) - Fm.matvec(Fm.lin_T(W), cp))
            fam["RG"].append(_mk("RG", f"{tag} V.P", Fm.dot(V, P) - Fm.dot(cv, cp)))
            # RL
            _emit(fam["RL"], "RL", f"{tag} sY", Fm.outer(s, Y) - Fm.matmul(A, W))
            _emit(fam["RL"], "RL", f"{tag} sV", Fm.outer(s, V) - Fm.matvec(A, cv))
            _emit(fam["RL"], "RL", f"{tag} sP", Fm.outer(s, P) - Fm.matvec(A, cp))
            # RH
            cols = [(f"Ycol{k}", Y[:, k], W[:, k]) for k in range(3)] + [("V", V, cv), ("P", P, cp)]
            for a in range(3):
                ea_hat = Fm.hat(I3[a][:, None] * np.ones((1, 1)))[..., 0]   # constant [e_a]_x
                for nm, Lc, Ec in cols:
                    lhs = Fm.cross(A[:, a], Lc)
                    rhs = Fm.matvec(A, Fm.cvec(ea_hat, Ec))
                    _emit(fam["RH"], "RH", f"{tag} A e{a} x {nm}", lhs - rhs)
            # RS
            _emit(fam["RS"], "RS", f"{tag} Y A^T",
                  Fm.matmul(Y, Fm.lin_T(A)) - Fm.hat(Fm.matvec(A, w)))
        if L.aniso:
            _, rv, rp = edge_residual_forms(inst, L, e)
            rhov, rhop = L.vec_form(L.rho_v(e)), L.vec_form(L.rho_p(e))
            _emit(fam["P4"], "P4", f"e{e} M rho_v-s r_v", Fm.matvec(E.M, rhov) - Fm.outer(s, rv))
            _emit(fam["P4"], "P4", f"e{e} M rho_p-s r_p", Fm.matvec(E.M, rhop) - Fm.outer(s, rp))
            fam["RR"].append(_mk("RR", f"e{e} |rho_v|^2", Fm.dot(rhov, rhov) - Fm.dot(rv, rv)))
            fam["RR"].append(_mk("RR", f"e{e} |rho_p|^2", Fm.dot(rhop, rhop) - Fm.dot(rp, rp)))
            fam["RR"].append(_mk("RR", f"e{e} rho_v.rho_p", Fm.dot(rhov, rhop) - Fm.dot(rv, rp)))
    return fam


def flat(families: dict, names=None) -> list[Constraint]:
    names = list(families) if names is None else names
    return [c for f in names for c in families[f]]


def family_residuals(z, families: dict) -> dict[str, np.ndarray]:
    return {f: np.array([c.residual(z) for c in cs]) for f, cs in families.items()}


def family_max(z, families: dict) -> dict[str, float]:
    return {f: float(np.abs(r).max()) for f, r in family_residuals(z, families).items()}


def vech(A) -> np.ndarray:
    """Upper-triangular entries (diagonal included) of a symmetric matrix, row-major."""
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    return A[np.triu_indices(A.shape[0])]
