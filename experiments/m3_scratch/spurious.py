"""M3 scratch: lifting equivalence in both directions and spurious feasible points.

(1) Hand-built spurious points: which families vanish / which exclude it.
(2) Reverse direction: trust-region least squares from random z0 onto {primary constraints = 0};
    every converged point must be +-z(state) for a genuine state (R_k in SO(3), liftings = defs).

Run:  python -m experiments.m3_scratch.spurious
"""
import numpy as np
from scipy.optimize import least_squares
from .instance import build, so3_exp, sk
from .qcqp import (Problem, PRIMARY, REDUNDANT, eval_family, idx_R, idx_e, idx_b, IDX_S, NKF)
from .check_constraints import truth_state

rng = np.random.default_rng(7)


def fam_report(F, z, fams):
    return {f: float(np.abs(eval_family(F[f], z)).max()) for f in fams if f in F}


def lifted_consistency(prob, z):
    """|z - s * z_of_state(state_of_z(z))|_inf : zero iff z is a genuine (+-) lifted state."""
    s = z[IDX_S]; Rs, vs, ps, bs = prob.state_of_z(z)
    return float(np.abs(z - s * prob.z_of_state(Rs, vs, ps, bs)).max()), [float(np.linalg.det(R)) for R in Rs]


def spurious_points(prob, F):
    fams = PRIMARY + REDUNDANT
    st = truth_state(prob.inst); Rs, vs, ps, bs = st
    out = {}
    # (a) improper rotation at KF0, liftings recomputed consistently from it
    Rr = [Rs[0] @ np.diag([1, 1, -1.0])] + Rs[1:]
    out["det-1 R0 (consistent liftings)"] = prob.z_of_state(Rr, vs, ps, bs)
    # (b) R0 = 0 with b0 = 0 (so every edge-0 lifting is 0 and P3 holds trivially)
    out["R0 = 0, b0 = 0"] = prob.z_of_state([np.zeros((3, 3))] + Rs[1:], vs, ps, [np.zeros(6)] + bs[1:])
    # (c) edge-0 i-side liftings built with a WRONG rotation Q (same Q for Y, V, P)
    z = prob.z_of_state(*st); d = prob.ed[0]; o = idx_e(0); Q = so3_exp(np.array([0.4, -0.3, 0.9]))
    w = d["JR"] @ bs[0][0:3]; cv = d["Jv"] @ bs[0]; cp = d["Jp"] @ bs[0]
    zc = z.copy(); zc[o:o + 9] = (Q @ sk(w)).flatten(order="F"); zc[o + 18:o + 21] = Q @ cv; zc[o + 24:o + 27] = Q @ cp
    out["edge0 i-liftings with wrong rotation Q"] = zc
    # (d) sign-flipped i-side liftings
    zd = z.copy(); zd[o:o + 9] *= -1; zd[o + 18:o + 21] *= -1; zd[o + 24:o + 27] *= -1
    out["edge0 i-liftings sign-flipped"] = zd
    # (e) edge-0 liftings computed from the bias of KF1 instead of KF0
    b_wrong = bs[0] + np.array([0.02, -0.01, 0.03, 0.05, 0.02, -0.04])
    ze = z.copy(); ze_alt = prob.z_of_state(Rs, vs, ps, [b_wrong] + bs[1:]); ze[o:o + 30] = ze_alt[o:o + 30]
    out["edge0 liftings from a different bias"] = ze
    # (f) global sign flip (NOT spurious: maps to the same state)
    out["global sign flip -z"] = -z
    Qm = prob.Q()
    for name, zz in out.items():
        rep = fam_report(F, zz, fams)
        viol = [f for f, v in rep.items() if v > 1e-9]
        print(f"\n[{name}]  violated families: {viol}")
        print("   satisfied:", [f for f, v in rep.items() if v <= 1e-9])
    zt = prob.z_of_state(*st)
    print("\nsign flip: cost(-z) - cost(z) =", (-zt) @ Qm @ (-zt) - zt @ Qm @ zt,
          " state_of_z(-z) == state_of_z(z):",
          all(np.allclose(a, b) for A, B in zip(prob.state_of_z(-zt), prob.state_of_z(zt)) for a, b in zip(A, B)))


def reverse_direction(prob, F, fams, n_starts=200, scale=1.0):
    As = [(A, b) for f in fams if f in F for _, A, b in F[f]]
    Ast = np.stack([A for A, _ in As]); bv = np.array([b for _, b in As])
    fun = lambda z: np.einsum("i,kij,j->k", z, Ast, z) - bv
    jac = lambda z: 2 * np.einsum("kij,j->ki", Ast, z)
    conv = 0; worst = 0.0; dets = []
    for t in range(n_starts):
        z0 = rng.normal(size=prob.n) * scale
        sol = least_squares(fun, z0, jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        if np.abs(fun(sol.x)).max() < 1e-10:
            conv += 1; c, d = lifted_consistency(prob, sol.x); worst = max(worst, c); dets += d
    return conv, worst, (min(dets), max(dets)) if dets else None


def main():
    inst = build(np.zeros(6)); prob = Problem(inst); F = prob.constraints()
    spurious_points(prob, F)
    print("\n--- reverse direction: TRF least squares onto primary constraints from 200 random z0 ---")
    conv, worst, dr = reverse_direction(prob, F, PRIMARY)
    print(f"primary only: converged {conv}/200, max |z - s z(state(z))| = {worst:.1e}, det(R_k) range {dr}")
    for drop in ["P2_hand_cols", "P3_lift", "P1_orth_cols"]:
        fams = [f for f in PRIMARY if f != drop and f in F]
        conv, worst, dr = reverse_direction(prob, F, fams, n_starts=60)
        print(f"primary minus {drop}: converged {conv}/60, max |z - s z(state(z))| = {worst:.1e}, det range {dr}")


if __name__ == "__main__":
    main()
