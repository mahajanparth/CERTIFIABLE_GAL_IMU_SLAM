"""Shor relaxation of the M3 QCQP and the measurements of Rai et al. Sec. VII-A.

Only what the OPEN-HYPOTHESIS claims M3-C36..C38 describe; certification proper is M4.

* relaxation:  min <Q, Z>  s.t.  <A_k, Z> = b_k,  Z >= 0   (Clarabel; never SCS here);
* numerical rank: #{lambda_i(Z) > 1e-6 lambda_max(Z)}  (Rai Sec. VII-A);
* sigma_1 / sigma_2 of Z  (Barfoot's rank-one test, threshold 1e5 per CLAUDE.md);
* relative gap eta = |x^T Q x - p*| / |x^T Q x|, p* the dual optimum, x the rounded point:
  rotation blocks projected onto SO(3) by SVD, liftings rebuilt from the decoded state.

The cost is divided by max|Q| before solving (max|Q| ~ 1.1e10 on this instance); p* and
the primal value are reported in the original units.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import cost as CO
from .lifting import Layout, State, state_of_z, z_of_state


@dataclass
class ShorResult:
    status: str
    primal: float
    dual: float
    Z: np.ndarray
    eig: np.ndarray
    seconds: float

    @property
    def rank(self) -> int:
        return int((self.eig > 1e-6 * self.eig[0]).sum())

    @property
    def sv_ratio(self) -> float:
        return float(self.eig[0] / max(abs(self.eig[1]), 1e-300))


def solve_shor(Q, constraints, solver: str = "CLARABEL", **opts) -> ShorResult:
    import cvxpy as cp

    n = Q.shape[0]
    scale = float(np.abs(Q).max())
    Abig = sp.vstack([c.A.reshape(1, n * n) for c in constraints]).tocsr()
    b = np.array([c.b for c in constraints])
    Z = cp.Variable((n, n), PSD=True)
    con = Abig @ cp.vec(Z, order="C") == b
    prob = cp.Problem(cp.Minimize(cp.trace((Q / scale) @ Z)), [con])
    t0 = time.time()
    prob.solve(solver=solver, **opts)
    Zv = 0.5 * (Z.value + Z.value.T)
    eig = np.linalg.eigvalsh(Zv)[::-1]
    # cvxpy's equality dual y satisfies  Q - sum_k y_k A_k >= 0 with dual objective -b^T y
    dual = -float(b @ con.dual_value) * scale
    return ShorResult(prob.status, float(prob.value) * scale, dual, Zv, eig, time.time() - t0)


def round_solution(res: ShorResult, inst, layout: Layout) -> State:
    w, U = np.linalg.eigh(res.Z)
    z = U[:, -1] * np.sqrt(max(w[-1], 0.0))
    if z[layout.s] < 0:
        z = -z
    return state_of_z(z, layout, project=True)


def relative_gap(res: ShorResult, inst, layout: Layout, term_list) -> tuple[float, float]:
    """(eta, x^T Q x) at the rounded point."""
    st = round_solution(res, inst, layout)
    f = CO.factored_cost(z_of_state(st, inst, layout), term_list)
    return abs(f - res.dual) / abs(f), f
