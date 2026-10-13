"""M3 scratch (follow-up to the implementer report): noise-model-independent restatements.

  (7) truth-cost discrepancy vs GTSAM split by component (rot / vel / pos), weight-free ratios
      0.5|r_R|_F^2 / |r_rot|^2, |r_v|^2 / |r_vel|^2, |r_p|^2 / |r_pos|^2 summed over edges
  (8) body- vs world-frame anisotropic weighting with a SYNTHETIC, Sigma-independent W_vp
  (9) C04 support: the Gal(3) right Jacobian J_R(x) = sum_n (-ad_x)^n/(n+1)! has equal
      (omega,omega), (nu,nu), (rho,rho) diagonal blocks (= SO(3) right Jacobian), checked
      against a finite-difference right Jacobian of Gal3.Expmap
  (10) printed instance fingerprint (taus, Dt) for the noise model

Run:  python -m experiments.m3_scratch.followup_checks
"""
import numpy as np
import gtsam
from gtsam import Rot3, Pose3, Gal3
from .instance import build, gal_exp, B_TRUE
from .instance import gal_hat
from .qcqp import Problem, NKF
from .check_cost import gtsam_cost, perturb, factored_cost
from .check_constraints import truth_state

rng = np.random.default_rng(21)
VP = [6, 7, 8, 3, 4, 5]


def gtsam_imu_residuals(inst, Rs, vs, ps, bs):
    vals = gtsam.Values(); X = lambda k: gtsam.symbol("x", k); V = lambda k: gtsam.symbol("v", k); B = lambda k: gtsam.symbol("b", k)
    from .instance import _gt_bias as gb
    for k in range(NKF):
        vals.insert(X(k), Pose3(Rot3(Rs[k]), ps[k])); vals.insert(V(k), vs[k]); vals.insert(B(k), gb(bs[k]))
    return [gtsam.GalileanImuFactor(X(e), V(e), X(e + 1), V(e + 1), B(e), E.pim).unwhitenedError(vals)
            for e, E in enumerate(inst["edges"])]


def component_split():
    print("(7) truth-cost ratio - 1 per component (weight-free), bbar = (1 - lam) b_true:")
    prev = None
    for lam in [1.0, 0.5, 0.25, 0.125]:
        inst = build(B_TRUE * (1 - lam)); prob = Problem(inst); T = truth_state(inst); z = prob.z_of_state(*T)
        rg = gtsam_imu_residuals(inst, *T)
        ours = np.zeros(3); ref = np.zeros(3)
        for e in range(2):
            rR, rv, rp = prob.residual_values(z, e)
            ours += [0.5 * np.sum(rR**2), rv @ rv, rp @ rp]
            ref += [rg[e][0:3] @ rg[e][0:3], rg[e][6:9] @ rg[e][6:9], rg[e][3:6] @ rg[e][3:6]]
        d = np.abs(ours / ref - 1)
        rat = "" if prev is None else "  ratios " + " ".join(f"{x:.3f}" for x in d / prev)
        print(f"   offset {lam * np.linalg.norm(B_TRUE):.4f}: rot {d[0]:.3e} vel {d[1]:.3e} pos {d[2]:.3e}{rat}")
        prev = d


def synthetic_aniso():
    print("(8) synthetic W_vp = 1e6 * blkdiag(diag(1,2,5), diag(1,2,5)), bbar = b_true:")
    inst = build(B_TRUE.copy()); Wvp = 1e6 * np.diag([1, 2, 5, 1, 2, 5.0])
    prob = Problem(inst, aniso=True, W_vp=[Wvp, Wvp]); probw = Problem(inst); T = truth_state(inst)
    body = lambda e, r: inst["edges"][e].tR * r[0:3] @ r[0:3] + r[VP] @ Wvp @ r[VP]
    for eps in [1e-2, 1e-3, 1e-4]:
        rb, rw = [], []
        for _ in range(30):
            st = perturb(*T, eps); g = gtsam_cost(inst, *st, imu_weight=body)
            rb.append(abs(factored_cost(prob, prob.z_of_state(*st)) / g - 1))
            zw = probw.z_of_state(*st); cw = sum((F @ zw) @ W @ (F @ zw) for nm, W, F in probw.cost_terms()
                                                 if not (nm.endswith("_v") or nm.endswith("_p")))
            for e in range(2):
                _, rv, rp = probw.residual_values(zw, e); r6 = np.concatenate([rv, rp]); cw += r6 @ Wvp @ r6
            rw.append(abs(cw / g - 1))
        print(f"   eps={eps:.0e}: body-frame lifted max|ratio-1| = {max(rb):.2e}; world-frame min|ratio-1| = {min(rw):.2e} max {max(rw):.2e}")


def ad_matrix(x):
    """ad_x as a 10x10 matrix in (omega, nu, rho, alpha) order, from the hat bracket."""
    A = np.zeros((10, 10)); X = gal_hat(x)
    for k in range(10):
        ek = np.zeros(10); ek[k] = 1; Y = gal_hat(ek); C = X @ Y - Y @ X
        A[:, k] = np.concatenate([[C[2, 1], C[0, 2], C[1, 0]], C[:3, 3], C[:3, 4], [C[3, 4]]])
    return A


def right_jacobian_blocks():
    print("(9) Gal(3) right Jacobian diagonal blocks:")
    worst_series = worst_blocks = worst_lower = 0.0
    for _ in range(200):
        x = rng.normal(size=10) * 0.8
        ad = ad_matrix(x); Js = np.zeros((10, 10)); term = np.eye(10); f = 1.0
        for n_ in range(40):
            Js += term / f; term = term @ (-ad); f *= (n_ + 2)
        h = 1e-6; Jfd = np.zeros((10, 10)); E0inv = np.linalg.inv(gal_exp(x))
        for k in range(10):
            d = np.zeros(10); d[k] = h
            Jfd[:, k] = (Gal3.Logmap(Gal3(E0inv @ gal_exp(x + d))) - Gal3.Logmap(Gal3(E0inv @ gal_exp(x - d)))) / (2 * h)
        worst_series = max(worst_series, np.abs(Js - Jfd).max())
        worst_blocks = max(worst_blocks, np.abs(Js[3:6, 3:6] - Js[0:3, 0:3]).max(), np.abs(Js[6:9, 6:9] - Js[0:3, 0:3]).max())
        worst_lower = max(worst_lower, np.abs(ad[0:3, 3:]).max(), np.abs(ad[3:6, 6:]).max())
    print(f"   max|series - FD| = {worst_series:.1e}; max|J_nunu - J_ww|, |J_rhorho - J_ww| = {worst_blocks:.1e}; "
          f"ad_x: omega row beyond omega cols and nu row beyond nu cols, max = {worst_lower:.1e}")


def fingerprint():
    inst = build(np.zeros(6))
    for e, E in enumerate(inst["edges"]):
        print(f"(10) edge {e}: Dt={E.Dt!r} tau_R={E.tR:.4e} tau_v={E.tv:.4e} tau_p={E.tp:.4e} "
              f"Lambda_rw diag = {np.diag(inst['lam_rw'][e])[[0, 3]]}")


def main():
    component_split(); synthetic_aniso(); right_jacobian_blocks(); fingerprint()


if __name__ == "__main__":
    main()
