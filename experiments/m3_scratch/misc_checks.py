"""M3 scratch: small checks backing individual claims.

  (1) r = X_i Ubar Cay(x) (Exp(eps^) - I)(I - x^/2) exactly, when X_i^-1 G^-1 X_j = Ubar Cay(x) Exp(eps^)
  (2) cleared residual vanishes iff X_j = G X_i Ubar Cay(x)
  (3) rank(P1 u RA1) per keyframe (trace dependency)
  (4) rotation weight: 0.5 tau_R (used) vs tau_R (Rai Eq.(20) literal) against GTSAM near truth
  (5) P2 alone (s=1): TRF solutions are SO(3) or 0
  (6) lifted form is load-bearing: perturbing Y entries of z changes the cost; Eq.(19) recomputed does not

Run:  python -m experiments.m3_scratch.misc_checks
"""
import numpy as np
from scipy.optimize import least_squares
from .instance import build, so3_exp, gal, gal_hat, gal_cay, gal_exp, gamma_left, sk, B_TRUE
from .qcqp import Problem, idx_R, idx_e, IDX_S, Lin, quad, NKF
from .check_constraints import rand_state, truth_state
from .check_cost import gtsam_cost, perturb, factored_cost

rng = np.random.default_rng(11)


def residual5(E, bbar, Ri, vi, pi, Rj, vj, pj, b):
    A = np.linalg.inv(gamma_left(E.Dt)) @ gal(Rj, vj, pj, 0.0); B = gal(Ri, vi, pi, 0.0) @ E.Ubar
    x = gal_hat(E.J @ (b - bbar)); I5 = np.eye(5)
    return A @ (I5 - x / 2) - B @ (I5 + x / 2), A, B, x


def main():
    inst = build(np.zeros(6)); E = inst["edges"][0]; bbar = inst["bbar"]
    w1 = w2 = w2b = 0.0
    for _ in range(2000):
        Ri = so3_exp(rng.normal(size=3) * 2); vi, pi = rng.normal(size=3), rng.normal(size=3); b = rng.normal(size=6) * 0.3
        x10 = E.J @ (b - bbar); eps = rng.normal(size=10) * 0.3; eps[9] = 0.0
        Xj = gamma_left(E.Dt) @ gal(Ri, vi, pi, 0.0) @ E.Ubar @ gal_cay(x10) @ gal_exp(eps)
        r, A, B, x = residual5(E, bbar, Ri, vi, pi, Xj[:3, :3], Xj[:3, 3], Xj[:3, 4], b)
        rhs = B @ gal_cay(x10) @ (gal_exp(eps) - np.eye(5)) @ (np.eye(5) - x / 2)
        w1 = max(w1, np.abs(r - rhs).max())
        Xj0 = gamma_left(E.Dt) @ gal(Ri, vi, pi, 0.0) @ E.Ubar @ gal_cay(x10)
        r0, A0, B0, _ = residual5(E, bbar, Ri, vi, pi, Xj0[:3, :3], Xj0[:3, 3], Xj0[:3, 4], b)
        w2 = max(w2, np.abs(r0).max(), abs(Xj0[3, 4]), abs(A0[3, 4] - E.Dt), abs(B0[3, 4] - E.Dt))
        w2b = max(w2b, np.abs(r[3:, :]).max())
    print(f"(1) max |r - X_i Ubar Cay(x)(Exp(eps)-I)(I-x/2)| over 2000 = {w1:.2e}")
    print(f"(2) at X_j = G X_i Ubar Cay(x): max |r|, |t_j|, |A_t - Dt|, |B_t - Dt| = {w2:.2e};  bottom rows of r at general states: {w2b:.1e}")
    # (3)
    prob = Problem(inst); F = prob.constraints()
    for k in range(NKF):
        vs = [A[np.triu_indices(prob.n)] for fam in ("P1_orth_cols", "RA1_orth_rows") for lab, A, _ in F[fam] if lab.startswith(f"R{k}")]
        print(f"(3) KF{k}: |P1|+|RA1| = {len(vs)}, rank = {np.linalg.matrix_rank(np.array(vs), tol=1e-9)}")
    # (4)
    inst_t = build(B_TRUE.copy()); T = truth_state(inst_t)
    for fac, label in [(0.5, "0.5 tau_R (used)"), (1.0, "tau_R (Rai Eq.(20) literal)")]:
        p = Problem(inst_t)
        def cost(z, fac=fac, p=p):
            c = 0.0
            for nm, W, Fm in p.cost_terms():
                Wm = W * (fac / 0.5) if nm.endswith("_R") and nm.startswith("imu") else W
                c += (Fm @ z) @ Wm @ (Fm @ z)
            return c
        rr = [abs(cost(p.z_of_state(*st)) / gtsam_cost(inst_t, *st) - 1) for st in (perturb(*T, 1e-4) for _ in range(30))]
        print(f"(4) rotation weight {label:28s}: max |ratio - 1| at eps=1e-4 over 30 = {max(rr):.2e}")
    # (5) P2 only (with s fixed to 1)
    def p2(rv):
        R = rv.reshape(3, 3, order="F"); out = []
        for (a, b_, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            out.append(np.cross(R[:, a], R[:, b_]) - R[:, c])
        return np.concatenate(out)
    kinds = {"SO(3)": 0, "zero": 0, "other": 0}
    for _ in range(300):
        sol = least_squares(p2, rng.normal(size=9) * 2, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if np.abs(p2(sol.x)).max() > 1e-10: continue
        R = sol.x.reshape(3, 3, order="F")
        if np.abs(R).max() < 1e-6: kinds["zero"] += 1
        elif np.abs(R.T @ R - np.eye(3)).max() < 1e-8 and abs(np.linalg.det(R) - 1) < 1e-8: kinds["SO(3)"] += 1
        else: kinds["other"] += 1
    print("(5) converged solutions of P2 (s=1) from 300 random starts:", kinds)
    # (6)
    st = rand_state(); z = prob.z_of_state(*st); c0 = factored_cost(prob, z)
    zp = z.copy(); o = idx_e(0); zp[o:o + 9] += 1e-3 * rng.normal(size=9)
    inst_t = build(B_TRUE.copy()); pt = Problem(inst_t); zt = pt.z_of_state(*truth_state(inst_t))
    zp = zt.copy(); o = idx_e(0); zp[o:o + 9] += 1e-6 * rng.normal(size=9)
    print(f"(6) at truth (bbar=b_true): cost(z) = {factored_cost(pt, zt):.3e}; with Y_0 entries of z perturbed by 1e-6: "
          f"{factored_cost(pt, zp):.3e}  (an Eq.(19) residual recomputed from (R,b) would be unchanged)")

if __name__ == "__main__":
    main()
