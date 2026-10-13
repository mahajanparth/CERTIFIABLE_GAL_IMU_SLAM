"""M3 scratch: the isotropy question.

(a) Scaling obstruction: on the feasible set, v_j enters z only as itself, so for any quadratic form
    the coefficient of lam^2 in z(v_j + lam d)^T Q z(v_j + lam d) is d^T Q_vv d -- independent of R_i.
    A body-frame weight rho^T W rho, rho = (R_i dRbar)^T r_v, has lam^2 coefficient d^T M W M^T d.
(b) LSQ fit of rho_v^T W rho_v over all degree-2 monomials of the edge-0 sub-vector of base z.
(c) Lifted variant (n = 136): rho = M^T r lifted, M rho = s r.  z^T Q z equals the direct anisotropic
    cost; compared with GTSAM under the SAME 6x6 (vel,pos) weight (and a world-frame contrast).
(d) Free variant: right-multiplied weight tr(r G r^T) on the 3x5 top block of the 5x5 residual.

Run:  python -m experiments.m3_scratch.isotropy_fit
"""
import numpy as np
from .instance import build, so3_exp, sk, B_TRUE
from .qcqp import Problem, direct_cost, idx_R, idx_v, idx_p, idx_b, idx_e, IDX_S
from .check_cost import gtsam_cost, perturb, factored_cost
from .check_constraints import rand_state, truth_state

rng = np.random.default_rng(2)
VP = [6, 7, 8, 3, 4, 5]          # NavState (rot,pos,vel) rows -> (vel,pos)


def scaling_obstruction():
    inst = build(np.zeros(6)); prob = Problem(inst); d = np.array([0.3, -1.0, 0.5])
    for name, W in [("iso 3I", 3 * np.eye(3)), ("diag(1,2,5)", np.diag([1.0, 2.0, 5.0]))]:
        coefs = []
        for _ in range(200):
            Rs, vs, ps, bs = rand_state()
            M = Rs[0] @ prob.ed[0]["Rc"]
            vals = []
            for lam in (0.0, 1.0, 2.0):          # exact quadratic in lam -> 2nd difference / 2
                v2 = list(vs); v2[1] = vs[1] + lam * d
                _, rv, _ = prob.residual_values(prob.z_of_state(Rs, v2, ps, bs), 0)
                rho = M.T @ rv; vals.append(rho @ W @ rho)
            coefs.append((vals[2] - 2 * vals[1] + vals[0]) / 2)
        print(f"  (a) W={name:12s}: lam^2 coefficient over 200 random R_0: min {min(coefs):.6f} max {max(coefs):.6f}")


def lsq_fit():
    inst = build(np.zeros(6)); prob = Problem(inst)
    sub = np.r_[idx_R(0):idx_R(0) + 15, idx_R(1):idx_R(1) + 15, idx_b(0):idx_b(0) + 6, idx_e(0):idx_e(0) + 30, IDX_S]
    iu = np.triu_indices(len(sub)); N = 6000
    Z = np.zeros((N, len(iu[0]))); RV = np.zeros((N, 3)); RP = np.zeros((N, 3))
    for t in range(N):
        Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(3)]
        st = (Rs, [rng.normal(size=3) for _ in range(3)], [rng.normal(size=3) for _ in range(3)], [rng.normal(size=6) * 0.3 for _ in range(3)])
        z = prob.z_of_state(*st); zs = z[sub]; Z[t] = np.outer(zs, zs)[iu]
        _, rv, rp = prob.residual_values(z, 0); M = Rs[0] @ prob.ed[0]["Rc"]; RV[t] = M.T @ rv; RP[t] = M.T @ rp
    print(f"  (b) monomials={Z.shape[1]} samples={N} rank(sample matrix)={np.linalg.matrix_rank(Z, tol=1e-8)}")
    A = np.random.default_rng(3).normal(size=(3, 3))
    for name, f in [("iso 3I on rho_v", 3 * np.einsum("ti,ti->t", RV, RV)),
                    ("diag(1,2,5) on rho_v", np.einsum("ti,ij,tj->t", RV, np.diag([1.0, 2, 5]), RV)),
                    ("diag(1,1,1.01) on rho_v", np.einsum("ti,ij,tj->t", RV, np.diag([1, 1, 1.01]), RV)),
                    ("random SPD on rho_v", np.einsum("ti,ij,tj->t", RV, A @ A.T + np.eye(3), RV)),
                    ("cross rho_v . rho_p", np.einsum("ti,ti->t", RV, RP))]:
        c, *_ = np.linalg.lstsq(Z, f, rcond=None)
        print(f"      {name:26s} relative LSQ residual = {np.linalg.norm(Z @ c - f) / np.linalg.norm(f):.2e}")


def W_vp_of(E):
    S = E.Sigma9[np.ix_(VP, VP)]; return np.linalg.inv(0.5 * (S + S.T))


def lifted_variant():
    for label, bbar in [("bbar=b_true", B_TRUE.copy()), ("bbar=0", np.zeros(6))]:
        inst = build(bbar); Wvp = [W_vp_of(E) for E in inst["edges"]]
        prob = Problem(inst, aniso=True, W_vp=Wvp); Q = prob.Q(); T = truth_state(inst)
        worst = 0
        for _ in range(300):
            st = rand_state(); z = prob.z_of_state(*st); jd = direct_cost(prob, *st, aniso=True)[0]
            worst = max(worst, abs(z @ Q @ z - jd) / max(1, abs(jd)))
        print(f"  (c) {label}: n={prob.n}, random states (300): max |zQz - J_direct_aniso|/max(1,J) = {worst:.2e}")
        body = lambda e, r: inst["edges"][e].tR * r[0:3] @ r[0:3] + r[VP] @ Wvp[e] @ r[VP]
        probw = Problem(inst)                              # world-frame contrast: W applied to r itself
        for eps in [1e-2, 1e-3, 1e-4]:
            rb, rw = [], []
            for _ in range(30):
                st = perturb(*T, eps); g = gtsam_cost(inst, *st, imu_weight=body)
                rb.append(abs(factored_cost(prob, prob.z_of_state(*st)) / g - 1))
                zw = probw.z_of_state(*st); cw = 0.0
                for nm, W, F in probw.cost_terms():
                    if nm.endswith("_v") or nm.endswith("_p"): continue
                    cw += (F @ zw) @ W @ (F @ zw)
                for e in range(2):
                    _, rv, rp = probw.residual_values(zw, e); r6 = np.concatenate([rv, rp]); cw += r6 @ Wvp[e] @ r6
                rw.append(abs(cw / g - 1))
            print(f"      eps={eps:.0e}: body-frame lifted max|ratio-1| = {max(rb):.2e};  world-frame W on r: max|ratio-1| = {max(rw):.2e}")


def right_weighted():
    """tr(r G r^T), G = blkdiag(D, G2), r = top 3x5 block [r_R | r_v | r_p]; target W_R = tr(D) I - D."""
    inst = build(B_TRUE.copy()); prob = Problem(inst); T = truth_state(inst); n = prob.n
    Gs, WR, G2s = [], [], []
    for E in inst["edges"]:
        Wr = np.linalg.inv(E.Sigma9[0:3, 0:3]); D = 0.5 * np.trace(Wr) * np.eye(3) - Wr
        S2 = np.array([[np.trace(E.Sigma9[6:9, 6:9]), np.trace(E.Sigma9[6:9, 3:6])],
                       [np.trace(E.Sigma9[3:6, 6:9]), np.trace(E.Sigma9[3:6, 3:6])]]) / 3
        G2 = np.linalg.inv(S2); G = np.zeros((5, 5)); G[:3, :3] = D; G[3:, 3:] = G2
        Gs.append(G); WR.append(Wr); G2s.append(G2)
        print(f"  (d) eig(D) = {np.round(np.linalg.eigvalsh(D), 1)} (>=0 needed), G2 offdiag/sqrt(diag prod) = "
              f"{G2[0, 1] / np.sqrt(G2[0, 0] * G2[1, 1]):.3f}, eig(G) min {np.linalg.eigvalsh(G).min():.2e}")
    def cost_rw(z):
        c = 0.0
        for nm, W, F in prob.cost_terms():
            if nm.startswith("imu"): continue
            c += (F @ z) @ W @ (F @ z)
        for e in range(2):
            rR, rv, rp = prob.residual_values(z, e); top = np.hstack([rR, rv[:, None], rp[:, None]])
            c += np.trace(top @ Gs[e] @ top.T)
        return c
    def wfun(e, r):
        rvp = np.concatenate([r[6:9], r[3:6]]); return r[0:3] @ WR[e] @ r[0:3] + rvp @ np.kron(G2s[e], np.eye(3)) @ rvp
    for eps in [1e-2, 1e-3, 1e-4]:
        rr = [abs(cost_rw(prob.z_of_state(*st)) / gtsam_cost(inst, *st, imu_weight=wfun) - 1)
              for st in (perturb(*T, eps) for _ in range(30))]
        print(f"      eps={eps:.0e}: right-weighted (no extra lifting) vs GTSAM with W_R full, (vel,pos) G2 x I3: max|ratio-1| = {max(rr):.2e}")


def main():
    print("(a) scaling obstruction"); scaling_obstruction()
    print("(b) least-squares fit in the base lifting"); lsq_fit()
    print("(c) residual-lifted anisotropic variant"); lifted_variant()
    print("(d) right-multiplied weight in the base lifting"); right_weighted()


if __name__ == "__main__":
    main()
