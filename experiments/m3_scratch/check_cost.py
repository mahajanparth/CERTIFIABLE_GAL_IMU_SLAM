"""M3 scratch: z^T Q z vs the direct 5x5 cost (any state), vs GTSAM factor graph (near truth)."""
import numpy as np
import gtsam
from gtsam import NavState, Rot3, Pose3
from .instance import build, so3_exp, _gt_bias, gal_exp, gal_cay, B_TRUE
from .qcqp import Problem, direct_cost, NKF
from .check_constraints import rand_state

rng = np.random.default_rng(1)


def gtsam_iso_cost(inst, Rs, vs, ps, bs):
    """Reference: GTSAM GalileanImuFactor unwhitened errors weighted with the SAME scalar taus,
    plus GTSAM prior/GPS/between factors with isotropic models.  Returned as sum of squares (2*error)."""
    J = 0.0
    vals = gtsam.Values()
    X = lambda k: gtsam.symbol("x", k); V = lambda k: gtsam.symbol("v", k); B = lambda k: gtsam.symbol("b", k)
    for k in range(NKF):
        vals.insert(X(k), Pose3(Rot3(Rs[k]), ps[k])); vals.insert(V(k), vs[k]); vals.insert(B(k), _gt_bias(bs[k]))
    for e, E in enumerate(inst["edges"]):
        f = gtsam.GalileanImuFactor(X(e), V(e), X(e + 1), V(e + 1), B(e), E.pim)
        r = f.unwhitenedError(vals)                 # (rot, pos, vel)
        J += E.tR * r[0:3] @ r[0:3] + E.tp * r[3:6] @ r[3:6] + E.tv * r[6:9] @ r[6:9]
    g = gtsam.NonlinearFactorGraph()
    for k, y in inst["gnss"].items():
        g.add(gtsam.GPSFactor(X(k), y, gtsam.noiseModel.Isotropic.Precision(3, inst["w_gnss"])))
    wR, wv, wp = inst["w_prior"]; pr = inst["prior"]
    g.add(gtsam.PriorFactorPose3(X(0), Pose3(Rot3(pr["R"]), pr["p"]),
                                 gtsam.noiseModel.Diagonal.Precisions(np.array([wR] * 3 + [wp] * 3))))
    g.add(gtsam.PriorFactorVector(V(0), pr["v"], gtsam.noiseModel.Isotropic.Precision(3, wv)))
    for e in range(2):
        L = inst["lam_rw"][e]; Lg = np.diag(np.concatenate([np.diag(L)[3:6], np.diag(L)[0:3]]))   # (a,g) order
        g.add(gtsam.BetweenFactorConstantBias(B(e), B(e + 1), gtsam.imuBias.ConstantBias(),
                                              gtsam.noiseModel.Gaussian.Information(Lg)))
    return J + 2 * g.error(vals)


def perturb(Rs, vs, ps, bs, eps):
    Rs2 = [R @ so3_exp(eps * rng.normal(size=3)) for R in Rs]
    vs2 = [v + eps * rng.normal(size=3) for v in vs]; ps2 = [p + eps * rng.normal(size=3) for p in ps]
    bs2 = [b + eps * 0.1 * rng.normal(size=6) for b in bs]
    return Rs2, vs2, ps2, bs2


def main():
    for label, bbar in [("bbar = b_true (model-consistent)", B_TRUE.copy()), ("bbar = 0 (bias correction active)", np.zeros(6))]:
        inst = build(bbar)
        prob = Problem(inst); Q = prob.Q()
        X = inst["truth"]; T = ([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], [B_TRUE.copy()] * 3)
        print(f"\n### {label}")
        # (1) exact identity at arbitrary states
        worst = 0
        for _ in range(500):
            st = rand_state()
            z = prob.z_of_state(*st); jq = z @ Q @ z; jd, _ = direct_cost(prob, *st)
            worst = max(worst, abs(jq - jd) / max(1.0, abs(jd)))
        print(f"random states (500): max |zQz - J_direct|/max(1,J) = {worst:.2e}")
        zt = prob.z_of_state(*T)
        print(f"truth: zQz = {zt @ Q @ zt:.6e}   J_direct = {direct_cost(prob, *T)[0]:.6e}   J_gtsam_iso = {gtsam_iso_cost(inst, *T):.6e}")
        for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
            rats = []
            for _ in range(50):
                st = perturb(*T, eps); z = prob.z_of_state(*st)
                rats.append(abs(z @ Q @ z / gtsam_iso_cost(inst, *st) - 1))
            print(f"  eps={eps:.0e}: max |zQz/J_gtsam_iso - 1| over 50 = {max(rats):.2e}  median {np.median(rats):.2e}")
    # Cay vs Exp correction (third order)
    inst = build(np.zeros(6)); E = inst["edges"][0]
    print("\nCay vs Exp of J db (edge 0):")
    for sc in [1.0, 0.5, 0.25, 0.125]:
        db = sc * B_TRUE; x = E.J @ db
        print(f"  |J db|={np.linalg.norm(x):.3e}  max|Ubar(Cay-Exp)| = {np.abs(E.Ubar @ (gal_cay(x) - gal_exp(x))).max():.3e}")


if __name__ == "__main__":
    main()
