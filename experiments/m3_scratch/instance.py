"""M3 scratch: the 3-keyframe / 2-edge GNSS-anchored test instance (throwaway, not src/).

Trajectory = M2 fixture M2_TRAJ_V1 (docs/02_baseline_reproduction.md section 0 on branch
stage-b/m2-baselines): piecewise-constant body twist, 400 Hz, 0.5 s per edge.
Preintegration is GTSAM 4.3.0 PreintegratedImuMeasurementsG at a NOMINAL bias bbar.

Single boundary conversion point: `_gt_bias` / `_perm_cols` below.  Project bias order is
[b_g; b_a]; GTSAM ConstantBias is (accel, gyro).  Gal3 tangent rows are (omega, nu, rho, alpha)
= (rot, VEL, POS, time).  NavState tangent / ImuFactor error order is (rot, POS, VEL).
"""
from __future__ import annotations
import numpy as np
import gtsam
from gtsam import Gal3, NavState, Rot3, Pose3
from gtsam.imuBias import ConstantBias

G_W = np.array([0.0, 0.0, -9.81])
DT = 2.5e-3
N_STEPS = 200
B_TRUE = np.array([0.0100, -0.0150, 0.0080, 0.0600, -0.0400, 0.0900])  # [b_g; b_a]
EDGES_W = [np.array([0.35, -0.22, 0.17]), np.array([-0.28, 0.41, -0.33])]
EDGES_F = [np.array([0.60, 0.45, 9.95]), np.array([-0.75, 0.90, 9.60])]
R0_LOG = np.array([0.10, -0.20, 0.30])
P0 = np.zeros(3)
V0 = np.array([0.50, -0.30, 0.20])

# noise densities used ONLY to produce GTSAM's covariance (weights); data stay noise-free
SIG_G = 1.7e-4 * np.sqrt(1 / DT)   # rad/s per sample (white-noise density 1.7e-4 rad/s/sqrt(Hz))
SIG_A = 2.0e-3 * np.sqrt(1 / DT)   # m/s^2 per sample
SIG_BG_RW = 1.9e-5                 # rad/s^2/sqrt(Hz)
SIG_BA_RW = 3.0e-3                 # m/s^3/sqrt(Hz)
SIG_GNSS = 0.05                    # m
SIG_PRIOR_R, SIG_PRIOR_V, SIG_PRIOR_P = 1e-3, 1e-3, 1e-3


def sk(a):
    a = np.asarray(a, float)
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def vee3(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def so3_exp(w):
    return Rot3.Expmap(np.asarray(w, float)).matrix()


def gal(R, v, p, t):
    """X = [[R, v, p],[0,1,t],[0,0,1]]  (velocity column BEFORE position column)."""
    M = np.eye(5); M[:3, :3] = R; M[:3, 3] = v; M[:3, 4] = p; M[3, 4] = t
    return M


def gal_hat(x10):
    """gal(3) hat, tangent order (phi, u=vel, w=pos, alpha) == GTSAM (omega, nu, rho, alpha)."""
    A = np.zeros((5, 5)); A[:3, :3] = sk(x10[0:3]); A[:3, 3] = x10[3:6]; A[:3, 4] = x10[6:9]
    A[3, 4] = x10[9]; return A


def gal_cay(x10):
    A = gal_hat(x10); I = np.eye(5)
    return np.linalg.solve(I - A / 2, I + A / 2)


def gal_exp(x10):
    return Gal3.Expmap(np.asarray(x10, float)).matrix()


def gamma_left(Dt):
    """Delama Eq.(18) Gamma_ij = T(-Dt) H' (pre-flight F2 corrected relation), left factor."""
    return gal(np.eye(3), G_W * Dt, -0.5 * G_W * Dt**2, -Dt)


# ---------------- boundary conversion (the ONE place) ----------------
def _gt_bias(b_proj):
    b = np.asarray(b_proj, float)
    return ConstantBias(b[3:6], b[0:3])          # ConstantBias(accel, gyro)


def _perm_cols(J_gtsam):
    """Jacobian columns (a, g) -> project (g, a)."""
    return np.concatenate([J_gtsam[:, 3:6], J_gtsam[:, 0:3]], axis=1)


NAV_FROM_GAL = [0, 1, 2, 6, 7, 8, 3, 4, 5]       # P_N of GTSAM doc Eq.(38): (rot,pos,vel) <- Gal3 rows
# ---------------------------------------------------------------------


def params():
    pr = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pr.setGyroscopeCovariance(SIG_G**2 * DT * np.eye(3))      # continuous density
    pr.setAccelerometerCovariance(SIG_A**2 * DT * np.eye(3))
    pr.setIntegrationCovariance(1e-10 * np.eye(3))
    return pr


def truth_states():
    """Closed-form exact chaining (M2 section 0.3 / component relation 1.1)."""
    R, v, p = so3_exp(R0_LOG), V0.copy(), P0.copy()
    out = [(R.copy(), v.copy(), p.copy())]
    T = DT * N_STEPS
    for w, f in zip(EDGES_W, EDGES_F):
        # exact ZOH increment via Gal3 exponential of the held input (GTSAM doc Eq.(12)-style)
        U = gal_exp(np.concatenate([w * T, f * T, np.zeros(3), [T]]))
        dR, dv, dp = U[:3, :3], U[:3, 3], U[:3, 4]
        p = p + v * T + 0.5 * G_W * T**2 + R @ dp
        v = v + G_W * T + R @ dv
        R = R @ dR
        out.append((R.copy(), v.copy(), p.copy()))
    return out


class Edge:
    pass


def preintegrate(bbar, b_meas=B_TRUE):
    """Return per-edge constants: Ubar (5x5, at bbar), J (10x6, project bias order), Sigma9."""
    edges = []
    pr = params()
    for w, f in zip(EDGES_W, EDGES_F):
        pim = gtsam.PreintegratedImuMeasurementsG(pr, _gt_bias(bbar))
        for _ in range(N_STEPS):
            pim.integrateMeasurement(f + b_meas[3:6], w + b_meas[0:3], DT)
        e = Edge(); e.pim = pim; e.Dt = pim.deltaTij()
        e.dR = pim.deltaRij().matrix(); e.dv = np.array(pim.deltaVij()); e.dp = np.array(pim.deltaPij())
        e.Ubar = gal(e.dR, e.dv, e.dp, e.Dt)
        e.Sigma9 = np.array(pim.preintMeasCov())   # NavState order (rot, pos, vel)
        e.J = jacobian(pim, bbar, e.Ubar)
        edges.append(e)
    return edges


def increment_from_predict(pim, b_proj):
    """U(b) = X_i^{-1} Gamma^{-1} X_j with X_j = pim.predict(X_i=identity-ish, b)."""
    Xi = NavState(Rot3(), np.zeros(3), np.zeros(3))
    Xj = pim.predict(Xi, _gt_bias(b_proj))
    Dt = pim.deltaTij()
    A = np.linalg.inv(gamma_left(Dt)) @ gal(Xj.attitude().matrix(), Xj.velocity(), Xj.position(), 0.0)
    return A  # X_i = I5 so X_i^{-1} Gamma^{-1} X_j = Gamma^{-1} X_j


def jacobian(pim, bbar, Ubar, h=1e-6):
    """Right-chart bias Jacobian J (10x6, columns [b_g; b_a]) of U(b) = Ubar Exp(J (b - bbar))."""
    J = np.zeros((10, 6))
    Ui = np.linalg.inv(Ubar)
    for k in range(6):
        d = np.zeros(6); d[k] = h
        Xp = Gal3(increment_from_predict(pim, bbar + d)); Xm = Gal3(increment_from_predict(pim, bbar - d))
        lp = Gal3.Logmap(Gal3(Ui @ Xp.matrix())); lm = Gal3.Logmap(Gal3(Ui @ Xm.matrix()))
        J[:, k] = (lp - lm) / (2 * h)
    return J


def taus(Sigma9):
    """Rai Eq.(23) trace reduction, applied to GTSAM's (rot, pos, vel) covariance."""
    tR = 3.0 / np.trace(Sigma9[0:3, 0:3]); tp = 3.0 / np.trace(Sigma9[3:6, 3:6]); tv = 3.0 / np.trace(Sigma9[6:9, 6:9])
    return tR, tv, tp


def lam_rw(Dt):
    return np.diag([1 / (SIG_BG_RW**2 * Dt)] * 3 + [1 / (SIG_BA_RW**2 * Dt)] * 3)


def build(bbar=np.zeros(6)):
    """The instance dict used by every scratch script."""
    X = truth_states()
    edges = preintegrate(bbar)
    inst = dict(truth=X, edges=edges, bbar=np.asarray(bbar, float),
                b_true=B_TRUE.copy(),
                gnss={1: X[1][2].copy(), 2: X[2][2].copy()},
                prior=dict(R=X[0][0].copy(), v=X[0][1].copy(), p=X[0][2].copy()),
                w_gnss=1 / SIG_GNSS**2, w_prior=(1 / SIG_PRIOR_R**2, 1 / SIG_PRIOR_V**2, 1 / SIG_PRIOR_P**2),
                lam_rw=[lam_rw(e.Dt) for e in edges])
    for e in edges:
        e.tR, e.tv, e.tp = taus(e.Sigma9)
    return inst


if __name__ == "__main__":
    inst = build()
    for k, e in enumerate(inst["edges"]):
        print("edge", k, "Dt", e.Dt, "taus", e.tR, e.tv, e.tp)
        print("  J rot/accel block max", np.abs(e.J[0:3, 3:6]).max(), " J time row max", np.abs(e.J[9]).max())
    # sanity: model A (right Exp) reproduces predict at b_true
    for e in inst["edges"]:
        db = B_TRUE - inst["bbar"]
        Ut = increment_from_predict(e.pim, B_TRUE)
        print("  |Ubar Exp(J db) - U(b)| =", np.abs(e.Ubar @ gal_exp(e.J @ db) - Ut).max(),
              " |Ubar Cay(J db) - U(b)| =", np.abs(e.Ubar @ gal_cay(e.J @ db) - Ut).max())
