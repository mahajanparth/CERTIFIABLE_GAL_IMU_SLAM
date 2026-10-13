"""M3 scratch: the cost.  z^T Q z vs the direct (unlifted) 5x5 Galilean-Cayley cost, vs the GTSAM
GalileanImuFactor factor graph under the same weighting; lifted-form structure of the rotation rows;
truth-cost discrepancy vs bias offset; residual-Jacobian rank at truth.

Run:  python -m experiments.m3_scratch.check_cost
"""
import numpy as np
import gtsam
from gtsam import Rot3, Pose3
from .instance import build, so3_exp, _gt_bias, gal_exp, gal_cay, B_TRUE
from .qcqp import Problem, direct_cost, NKF, idx_R, idx_v, idx_p, idx_b, idx_e, IDX_S
from .check_constraints import rand_state, truth_state

rng = np.random.default_rng(1)


def gtsam_cost(inst, Rs, vs, ps, bs, imu_weight=None):
    """GTSAM reference.  IMU: GalileanImuFactor.unwhitenedError (NavState order rot,pos,vel) weighted by
    imu_weight(e, r9) (default: the same scalar taus, tau_R|r_rot|^2 + tau_p|r_pos|^2 + tau_v|r_vel|^2).
    Prior / GPS / bias between factors are GTSAM factors; returned as a sum of squares (= 2 * error)."""
    J = 0.0; vals = gtsam.Values()
    X = lambda k: gtsam.symbol("x", k); V = lambda k: gtsam.symbol("v", k); B = lambda k: gtsam.symbol("b", k)
    for k in range(NKF):
        vals.insert(X(k), Pose3(Rot3(Rs[k]), ps[k])); vals.insert(V(k), vs[k]); vals.insert(B(k), _gt_bias(bs[k]))
    for e, E in enumerate(inst["edges"]):
        r = gtsam.GalileanImuFactor(X(e), V(e), X(e + 1), V(e + 1), B(e), E.pim).unwhitenedError(vals)
        J += (E.tR * r[0:3] @ r[0:3] + E.tp * r[3:6] @ r[3:6] + E.tv * r[6:9] @ r[6:9]) if imu_weight is None \
            else imu_weight(e, r)
    g = gtsam.NonlinearFactorGraph()
    for k, y in inst["gnss"].items():
        g.add(gtsam.GPSFactor(X(k), y, gtsam.noiseModel.Isotropic.Precision(3, inst["w_gnss"])))
    wR, wv, wp = inst["w_prior"]; pr = inst["prior"]
    g.add(gtsam.PriorFactorPose3(X(0), Pose3(Rot3(pr["R"]), pr["p"]),
                                 gtsam.noiseModel.Diagonal.Precisions(np.array([wR] * 3 + [wp] * 3))))
    g.add(gtsam.PriorFactorVector(V(0), pr["v"], gtsam.noiseModel.Isotropic.Precision(3, wv)))
    for e in range(2):
        Lm = inst["lam_rw"][e]; Lg = np.diag(np.concatenate([np.diag(Lm)[3:6], np.diag(Lm)[0:3]]))  # (a,g) order
        g.add(gtsam.BetweenFactorConstantBias(B(e), B(e + 1), gtsam.imuBias.ConstantBias(),
                                              gtsam.noiseModel.Gaussian.Information(Lg)))
    return J + 2 * g.error(vals)


def perturb(Rs, vs, ps, bs, eps, rng=rng):
    return ([R @ so3_exp(eps * rng.normal(size=3)) for R in Rs], [v + eps * rng.normal(size=3) for v in vs],
            [p + eps * rng.normal(size=3) for p in ps], [b + eps * 0.1 * rng.normal(size=6) for b in bs])


def factored_cost(prob, z):
    return sum((F @ z) @ W @ (F @ z) for _, W, F in prob.cost_terms())


def lifted_form_report(prob):
    """Coefficients of the rotation-residual linear forms on the b, Y, Y' columns."""
    out = {}
    for e in range(2):
        rR, _, _ = prob.residual_forms(e); FR = rR.reshape(9, prob.n, order="F")   # rows = vec(r_R) col-major
        o = idx_e(e); bi = idx_b(e)
        out[e] = dict(b=np.abs(FR[:, bi:bi + 6]).max(),
                      Y=np.abs(FR[:, o:o + 9] + 0.5 * np.eye(9)).max(),
                      Yp=np.abs(FR[:, o + 9:o + 18] + 0.5 * np.eye(9)).max())
    return out


def residual_vector(prob, z):
    out = []
    for _, W, F in prob.cost_terms():
        w, U = np.linalg.eigh(W); out.append(np.sqrt(np.clip(w, 0, None)) * (U.T @ (F @ z)))
    return np.concatenate(out)


def jac_rank(prob, st, h=1e-6):
    Rs, vs, ps, bs = st
    def zof(d):
        R2 = [Rs[k] @ so3_exp(d[15 * k:15 * k + 3]) for k in range(NKF)]
        v2 = [vs[k] + d[15 * k + 3:15 * k + 6] for k in range(NKF)]; p2 = [ps[k] + d[15 * k + 6:15 * k + 9] for k in range(NKF)]
        b2 = [bs[k] + d[15 * k + 9:15 * k + 15] for k in range(NKF)]
        return prob.z_of_state(R2, v2, p2, b2)
    cols = []
    for i in range(45):
        d = np.zeros(45); d[i] = h
        cols.append((residual_vector(prob, zof(d)) - residual_vector(prob, zof(-d))) / (2 * h))
    Jm = np.array(cols).T; sv = np.linalg.svd(Jm, compute_uv=False)
    return int((sv > sv[0] * 1e-12).sum()), sv[0] / sv[-1]


def main():
    for label, bbar in [("bbar = b_true (model-consistent)", B_TRUE.copy()), ("bbar = 0 (bias correction active)", np.zeros(6))]:
        inst = build(bbar); prob = Problem(inst); Q = prob.Q(); T = truth_state(inst)
        print(f"\n### {label}")
        worst = 0
        for _ in range(500):
            st = rand_state(); z = prob.z_of_state(*st)
            worst = max(worst, abs(z @ Q @ z - direct_cost(prob, *st)[0]) / max(1.0, abs(direct_cost(prob, *st)[0])))
        print(f"random states (500): max |zQz - J_direct|/max(1,J) = {worst:.2e}")
        zt = prob.z_of_state(*T)
        print(f"truth: zQz = {zt @ Q @ zt:.3e} (dense form; cancellation)  factored = {factored_cost(prob, zt):.3e}"
              f"  J_direct = {direct_cost(prob, *T)[0]:.3e}  J_gtsam = {gtsam_cost(inst, *T):.3e}")
        for eps in [1e-2, 1e-3, 1e-4]:
            rats = []
            for _ in range(50):
                st = perturb(*T, eps); z = prob.z_of_state(*st)
                rats.append(abs(factored_cost(prob, z) / gtsam_cost(inst, *st) - 1))
            print(f"  eps={eps:.0e}: max |cost/J_gtsam - 1| over 50 = {max(rats):.2e}  median {np.median(rats):.2e}")
        rk, cond = jac_rank(prob, T)
        print(f"  whitened residual Jacobian at truth: rank {rk}/45, cond {cond:.2e}")
    # Q structure and lifted form
    inst = build(np.zeros(6)); prob = Problem(inst); Q = prob.Q()
    ev = np.linalg.eigvalsh(Q); bc = np.arange(45, 63); nb = np.setdiff1d(np.arange(prob.n), bc)
    print(f"\nQ: max asym {np.abs(Q - Q.T).max():.1e}, min eig / max eig {ev[0] / ev[-1]:.1e}, max|Q| {np.abs(Q).max():.2e}")
    print(f"   max |Q[b, non-b]| = {np.abs(Q[np.ix_(bc, nb)]).max():.1e};  Q[b,b] - RW part = "
          f"{np.abs(Q[np.ix_(bc, bc)] - sum(F.T @ W @ F for nm, W, F in prob.cost_terms() if nm.startswith('rw'))[np.ix_(bc, bc)]).max():.1e}")
    print("   lifted form of r_R (max |coef - expected|; b: expected 0, Y/Y': expected -1/2 I9):", lifted_form_report(prob))
    # truth-cost discrepancy vs bias offset (bbar = b_true - lam * b_true)
    print("\ntruth-cost discrepancy vs GTSAM as the bias offset shrinks:")
    for lam in [1.0, 0.5, 0.25, 0.125]:
        inst = build(B_TRUE * (1 - lam)); prob = Problem(inst); T = truth_state(inst); zt = prob.z_of_state(*T)
        c, g = factored_cost(prob, zt), gtsam_cost(inst, *T)
        print(f"  |b_true - bbar| = {lam * np.linalg.norm(B_TRUE):.4f}: cost = {c:.4e}, J_gtsam = {g:.4e}, rel diff = {abs(c / g - 1):.2e}")
    inst = build(np.zeros(6)); E = inst["edges"][0]
    print("\nCay vs Exp of J db (edge 0):")
    for sc in [1.0, 0.5, 0.25, 0.125]:
        x = E.J @ (sc * B_TRUE)
        print(f"  |J db|={np.linalg.norm(x):.3e}  max|Ubar(Cay-Exp)| = {np.abs(E.Ubar @ (gal_cay(x) - gal_exp(x))).max():.3e}")


if __name__ == "__main__":
    main()
