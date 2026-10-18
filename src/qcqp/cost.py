"""Cost matrix Q = sum_t F_t^T W_t F_t of the M3 QCQP.

Residual linear forms of edge (i, j) with M = R_i dRbar, Gam = [J_R bbar_g]_x,
cbar^v = J_nu bbar, cbar^p = J_rho bbar (claim M3-C07)::

    r_R = R_j (I + Gam/2) - M (I - Gam/2) - 1/2 (Y + Y')
    r_v = v_j - v_i - g Dt s - R_i dvbar + 1/2 (R_j + M) cbar^v - 1/2 (V + V')
    r_p = p_j - p_i - Dt v_i - 1/2 g Dt^2 s - R_i dpbar + 1/2 (R_j + M) cbar^p - 1/2 (P + P')

These are the top 3x5 block of the cleared Gal(3) residual
``A (I - x^/2) - B (I + x^/2)``, ``A = G^-1 X_j``, ``B = X_i Ubar``, ``x = J (b - bbar)``.

Weightings (``variant``):

* ``"iso"`` (base, n = 124): 1/2 tau_R ||r_R||_F^2 + tau_v ||r_v||^2 + tau_p ||r_p||^2;
* ``"aniso"`` (n = 136): 1/2 tau_R ||r_R||_F^2 + [rho_v; rho_p]^T W_vp [rho_v; rho_p]
  (W_vp from preintMeasCov, or a caller-supplied weight);
* ``"partial"`` (n = 124, claim M3-C33): tr(rt G rt^T), G = blkdiag(D, G2).

Plus, in every variant: chordal R_0 prior 1/(2 sigma^2) ||R_0 - R0bar s||_F^2, v_0 / p_0
priors 1/sigma^2 ||. - . s||^2, GNSS 1/sigma_gnss^2 ||p_k - z_k s||^2 (k = 1, 2), and bias
random walk ||b_{k+1} - b_k||^2_{Lambda_rw}.  The 1/2 on the Frobenius terms makes them match
tau |theta|^2 to first order, because ||R Exp(theta) - R||_F^2 = 2 |theta|^2 + O(theta^4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import forms
from .fixture import hat
from .lifting import BASE, Layout


@dataclass(frozen=True)
class Term:
    name: str
    F: np.ndarray        #: m x n
    W: np.ndarray        #: m x m, symmetric PSD


def vec_rows(A):
    """(3,3,n) linear matrix form -> (9,n) rows of vec(.) column-major."""
    return A.transpose(1, 0, 2).reshape(9, A.shape[-1])


def edge_residual_forms(inst, layout: Layout, e: int):
    edge = inst.edges[e]
    i, j = edge.i, edge.j
    s = layout.s_form()
    Ri, Rj = layout.mat_form(layout.R(i)), layout.mat_form(layout.R(j))
    vi, vj = layout.vec_form(layout.v(i)), layout.vec_form(layout.v(j))
    pi, pj = layout.vec_form(layout.p(i)), layout.vec_form(layout.p(j))
    Y, Yp = layout.mat_form(layout.Y(e)), layout.mat_form(layout.Yp(e))
    V, Vp = layout.vec_form(layout.V(e)), layout.vec_form(layout.Vp(e))
    P, Pp = layout.vec_form(layout.P(e)), layout.vec_form(layout.Pp(e))
    I3 = np.eye(3)
    Gam = hat(edge.J_R @ inst.bbar[0:3])
    cv, cp = edge.J_nu @ inst.bbar, edge.J_rho @ inst.bbar
    Dt, g = edge.Dt, inst.g
    M = forms.cmat_right(Ri, edge.dR)                         # R_i dRbar
    rR = (forms.cmat_right(Rj, I3 + 0.5 * Gam) - forms.cmat_right(M, I3 - 0.5 * Gam)
          - 0.5 * (Y + Yp))
    Rcv = lambda A, c: np.einsum("abn,b->an", A, c)            # noqa: E731  (lin) x const vec
    rv = (vj - vi - np.outer(g * Dt, s) - Rcv(Ri, edge.dv)
          + 0.5 * (Rcv(Rj, cv) + Rcv(M, cv)) - 0.5 * (V + Vp))
    rp = (pj - pi - Dt * vi - np.outer(0.5 * g * Dt**2, s) - Rcv(Ri, edge.dp)
          + 0.5 * (Rcv(Rj, cp) + Rcv(M, cp)) - 0.5 * (P + Pp))
    return rR, rv, rp


def terms(inst, layout: Layout = BASE, variant: str = "iso", W_vp=None) -> list[Term]:
    """``W_vp`` (aniso only): a 6x6 weight on [rho_v; rho_p] used for every edge instead of
    the instance's preintMeasCov-derived one (claim M3-C34's synthetic weight)."""
    if variant == "aniso" and not layout.aniso:
        raise ValueError(f"the anisotropic cost variant needs the n = 136 layout, got n = {layout.n}")
    out = []
    for e, edge in enumerate(inst.edges):
        rR, rv, rp = edge_residual_forms(inst, layout, e)
        if variant == "partial":
            out.append(Term(f"imu{e}.rot", vec_rows(rR), np.kron(edge.D, np.eye(3))))
            out.append(Term(f"imu{e}.vp", np.vstack([rv, rp]), np.kron(edge.G2, np.eye(3))))
            continue
        out.append(Term(f"imu{e}.rot", vec_rows(rR), 0.5 * edge.tau_R * np.eye(9)))
        if variant == "iso":
            out.append(Term(f"imu{e}.vel", rv, edge.tau_v * np.eye(3)))
            out.append(Term(f"imu{e}.pos", rp, edge.tau_p * np.eye(3)))
        elif variant == "aniso":
            rho = np.vstack([layout.vec_form(layout.rho_v(e)), layout.vec_form(layout.rho_p(e))])
            out.append(Term(f"imu{e}.rho", rho, edge.W_vp if W_vp is None else np.asarray(W_vp, float)))
        else:
            raise ValueError(f"unknown cost variant {variant!r}")
    s = layout.s_form()
    R0bar, v0bar, p0bar = inst.prior
    w_prior = 1.0 / inst.sigma_prior**2
    out.append(Term("prior.R0", layout.vec_form(layout.R(0)) - np.outer(R0bar.flatten(order="F"), s),
                    0.5 * w_prior * np.eye(9)))
    out.append(Term("prior.v0", layout.vec_form(layout.v(0)) - np.outer(v0bar, s), w_prior * np.eye(3)))
    out.append(Term("prior.p0", layout.vec_form(layout.p(0)) - np.outer(p0bar, s), w_prior * np.eye(3)))
    for k, zk in sorted(inst.gnss.items()):
        out.append(Term(f"gnss{k}", layout.vec_form(layout.p(k)) - np.outer(zk, s),
                        np.eye(3) / inst.sigma_gnss**2))
    for k in range(layout.n_kf - 1):
        out.append(Term(f"rw{k}", layout.vec_form(layout.b(k + 1)) - layout.vec_form(layout.b(k)),
                        inst.rw_precision))
    return out


def assemble_Q(term_list) -> np.ndarray:
    Q = sum(t.F.T @ t.W @ t.F for t in term_list)
    return 0.5 * (Q + Q.T)


def factored_cost(z, term_list) -> float:
    total = 0.0
    for t in term_list:
        r = t.F @ z
        total += float(r @ t.W @ r)
    return total


def whitened_residual(z, term_list) -> np.ndarray:
    """Stack of sqrt(W_t) F_t z (symmetric square root)."""
    out = []
    for t in term_list:
        w, U = np.linalg.eigh(t.W)
        out.append((U * np.sqrt(np.clip(w, 0, None))) @ U.T @ (t.F @ z))
    return np.concatenate(out)
