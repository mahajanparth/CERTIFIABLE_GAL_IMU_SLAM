"""M3 -- one or more tests per claim of ``docs/claims/M3.yaml`` (M3-C01..C38).

Written from the claim list alone.  Every reference below is computed INDEPENDENTLY of
``src/qcqp``:

* the preintegration data (Ubar, J, covariances) are re-integrated here with GTSAM 4.3.0
  directly (:func:`ref`), with the bias-order conversion written out at each call;
* ground-truth keyframes are re-derived here by chaining ``pim.predict`` at the true bias;
* the lifted vector is re-encoded here from the claim's index map (:func:`z_ref`);
* residuals are recomputed from 5x5 Gal(3) matrices (:func:`resid5`), costs from those
  residuals (:func:`J_direct`) and from a GTSAM factor graph (:func:`gtsam_cost`);
* spurious points are constructed explicitly.

The only things imported from ``src.qcqp.fixture`` are the trajectory LITERALS (rates,
forces, initial state, noise densities); everything derived from them is recomputed.
"""
from __future__ import annotations

import functools
import gzip
import html as _html
import pathlib
import re

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import least_squares

import gtsam
from gtsam import Pose3, Rot3, noiseModel
from gtsam.imuBias import ConstantBias
from gtsam.symbol_shorthand import B, V, X

from src.qcqp import constraints as CN
from src.qcqp import cost as CO
from src.qcqp import fixture as SRC_FX   # used ONLY to pin src's literals against LIT
import src.qcqp.instance as INST
from src.qcqp import lifting as LF

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class LIT:
    """Instance literals, RETYPED here (not imported from src) so that the reference does not
    move with a corrupted src literal.  Trajectory: M2 fixture M2_TRAJ_V1 as defined on branch
    stage-b/m2-baselines (src/trajectory.py); sigmas: the M3 claims' domain text."""
    B_G = np.array([0.0100, -0.0150, 0.0080])
    B_A = np.array([0.0600, -0.0400, 0.0900])
    BIAS_TRUE = np.concatenate([B_G, B_A])            # project order [b_g; b_a]
    OMEGA_BODY = (np.array([0.35, -0.22, 0.17]), np.array([-0.28, 0.41, -0.33]))
    F_BODY = (np.array([0.60, 0.45, 9.95]), np.array([-0.75, 0.90, 9.60]))
    ROT_VEC_0 = np.array([0.10, -0.20, 0.30])
    P_0 = np.zeros(3)
    V_0 = np.array([0.50, -0.30, 0.20])
    DT, STEPS = 2.5e-3, 200
    SIGMA_PRIOR = 1e-3
    SIGMA_GNSS = 0.05
    SIGMA_BG_RW = 1.9e-5
    SIGMA_BA_RW = 3.0e-3


#: M2_TRAJ_V1 reference values (M2 src/trajectory.py, transcribed at 15 significant digits):
#: these pin the ground-truth trajectory independently of any literal above
M2_REF = dict(
    V_KF1=np.array([-3.092347809743656e-01, -1.121267155202117e+00, 1.401383484057339e-01]),
    P_KF1=np.array([5.715027067778627e-02, -3.148113358346768e-01, 9.348023290152563e-02]),
    V_KF2=np.array([-1.520564406279614e+00, -1.903478183400628e+00, -1.641944548085670e-01]),
    P_KF2=np.array([-4.312534482841444e-01, -1.111986889144151e+00, 7.234717707859262e-02]),
    R_KF2=np.array([[9.677181225278726e-01, -2.412143599148208e-01, -7.305660751781723e-02],
                    [2.301509945763355e-01, 9.638954740062700e-01, -1.339254825854730e-01],
                    [1.027236828908148e-01, 1.127880656856546e-01, 9.882948432589516e-01]]),
    DR_01=np.array([[9.903776532493884e-01, -9.387914277777526e-02, -1.016799414611560e-01],
                    [7.470913760838102e-02, 9.811536442685306e-01, -1.782023319050390e-01],
                    [1.164931272738697e-01, 1.688911865370475e-01, 9.787262734840944e-01]]),
)

# ======================================================================================
# Independent references
# ======================================================================================

I3, I5 = np.eye(3), np.eye(5)
G_W = np.array([0.0, 0.0, -9.81])
N_BASE, N_ANISO = 124, 136
S_IDX = 123


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=float)


def rexp(w):
    return Rot3.Expmap(np.asarray(w, dtype=float)).matrix()


def Xmat(R, v, p, t):
    """5x5 Gal(3) matrix [[R, v, p], [0, 1, t], [0, 0, 1]] (velocity column BEFORE position)."""
    M = np.eye(5)
    M[0:3, 0:3], M[0:3, 3], M[0:3, 4], M[3, 4] = R, v, p, t
    return M


def xhat(x):
    """10-vector in GTSAM's (omega, nu, rho, alpha) order -> 5x5 algebra element."""
    M = np.zeros((5, 5))
    M[0:3, 0:3], M[0:3, 3], M[0:3, 4], M[3, 4] = skew(x[0:3]), x[3:6], x[6:9], x[9]
    return M


def cay5(x):
    A = xhat(x)
    return np.linalg.solve(I5 - 0.5 * A, I5 + 0.5 * A)


def gexp(x):
    return gtsam.Gal3.Expmap(np.asarray(x, dtype=float)).matrix()


def Gmat(Dt):
    return Xmat(I3, G_W * Dt, -0.5 * G_W * Dt**2, -Dt)


def cb(b):
    """project [b_g; b_a] -> ConstantBias(accel, gyro), written out here on purpose."""
    return ConstantBias(np.asarray(b[3:6], float), np.asarray(b[0:3], float))


#: the claims' stated IMU noise model (domain of M3-C01.., fingerprint M3-C39), written here
#: as literals rather than read from src
GYRO_COV, ACC_COV, INT_COV = (1.7e-4)**2, (2.0e-3)**2, 1e-10


def make_pim(gyro, acc, dt, bbar):
    p = gtsam.PreintegrationParams.MakeSharedU(9.81)
    p.setGyroscopeCovariance(GYRO_COV * I3)
    p.setAccelerometerCovariance(ACC_COV * I3)
    p.setIntegrationCovariance(INT_COV * I3)
    pim = gtsam.PreintegratedImuMeasurementsG(p, cb(bbar))
    for w, a in zip(gyro, acc):
        pim.integrateMeasurement(np.asarray(a, float), np.asarray(w, float), dt)
    return pim


def fixture_pim(e, bbar):
    n = LIT.STEPS
    gyro = np.tile(LIT.OMEGA_BODY[e] + LIT.B_G, (n, 1))
    acc = np.tile(LIT.F_BODY[e] + LIT.B_A, (n, 1))
    return make_pim(gyro, acc, LIT.DT, bbar)


def U_of(pim, b):
    """U(b) = G^-1 X_j for X_i = identity, from pim.predict."""
    ns = pim.predict(gtsam.NavState(), cb(b))
    Xj = Xmat(ns.attitude().matrix(), np.asarray(ns.velocity()), np.asarray(ns.position()), 0.0)
    return np.linalg.solve(Gmat(pim.deltaTij()), Xj)


def fd_J(pim, bbar, h):
    """Central differences of Log(Ubar^-1 U(b)); columns in PROJECT order [b_g; b_a]."""
    bbar = np.asarray(bbar, float)
    Ui = np.linalg.inv(U_of(pim, bbar))
    J = np.zeros((10, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        lp = gtsam.Gal3.Logmap(gtsam.Gal3(Ui @ U_of(pim, bbar + d)))
        lm = gtsam.Gal3.Logmap(gtsam.Gal3(Ui @ U_of(pim, bbar - d)))
        J[:, k] = (lp - lm) / (2 * h)
    return J


def ls_J(pim, bbar, n=60, seed=0):
    """J as the least-squares LINEAR fit of Log(Ubar^-1 U(bbar + d)) over n random |d_k| <= 0.1
    (project order).  The map is exactly linear (predict applies Ubar Exp(J d)), so this has no
    step-size trade-off: the fit residual is returned and is ~1e-15."""
    bbar = np.asarray(bbar, float)
    Ui = np.linalg.inv(U_of(pim, bbar))
    D = np.random.default_rng(seed).uniform(-0.1, 0.1, (n, 6))
    L = np.array([gtsam.Gal3.Logmap(gtsam.Gal3(Ui @ U_of(pim, bbar + d))) for d in D])
    J = np.linalg.lstsq(D, L, rcond=None)[0].T
    return J, float(np.abs(D @ J.T - L).max())


@functools.lru_cache(maxsize=None)
def ref(bbar_key):
    """Independent per-edge data at nominal bias.  ``bbar_key`` is "true", "zero" or a tuple."""
    bbar = {"true": LIT.BIAS_TRUE, "zero": np.zeros(6)}.get(bbar_key) if isinstance(bbar_key, str) \
        else np.array(bbar_key, float)
    edges = []
    for e in range(2):
        pim = fixture_pim(e, bbar)
        nav = pim.deltaXij()                                    # NavState (R, p, v)
        Dt = pim.deltaTij()
        Ubar = Xmat(nav.attitude().matrix(), np.asarray(nav.velocity()),
                    np.asarray(nav.position()), Dt)
        C = np.asarray(pim.preintMeasCov())                      # NavState order (rot, pos, vel)
        rot, pos, vel = slice(0, 3), slice(3, 6), slice(6, 9)
        S2 = np.array([[np.trace(C[vel, vel]), np.trace(C[vel, pos])],
                       [np.trace(C[pos, vel]), np.trace(C[pos, pos])]]) / 3
        Svp = np.block([[C[vel, vel], C[vel, pos]], [C[pos, vel], C[pos, pos]]])
        edges.append(dict(
            i=e, j=e + 1, Dt=Dt, Ubar=Ubar, dR=Ubar[0:3, 0:3], dv=Ubar[0:3, 3], dp=Ubar[0:3, 4],
            J=ls_J(pim, bbar)[0], pim=pim, C=C,
            tau_R=3 / np.trace(C[rot, rot]), tau_v=3 / np.trace(C[vel, vel]),
            tau_p=3 / np.trace(C[pos, pos]),
            W_R=np.linalg.inv(C[rot, rot]), G2=np.linalg.inv(S2), W_vp=np.linalg.inv(Svp)))
    return dict(bbar=bbar, edges=edges)


@functools.lru_cache(maxsize=None)
def truth_ref():
    """Ground truth re-derived by chaining GTSAM's prediction at the true bias."""
    ns = gtsam.NavState(Rot3.Expmap(LIT.ROT_VEC_0), LIT.P_0, LIT.V_0)
    out = [ns]
    for e in range(2):
        out.append(fixture_pim(e, LIT.BIAS_TRUE).predict(out[-1], cb(LIT.BIAS_TRUE)))
    R = [s.attitude().matrix() for s in out]
    return LF.State(R, [np.asarray(s.velocity()) for s in out],
                    [np.asarray(s.position()) for s in out], [LIT.BIAS_TRUE.copy() for _ in out])


def lift_ref(state, ed):
    i, j = ed["i"], ed["j"]
    b = state.b[i]
    M = state.R[i] @ ed["dR"]
    W = skew(ed["J"][0:3, 0:3] @ b[0:3])
    cv, cp = ed["J"][3:6] @ b, ed["J"][6:9] @ b
    return dict(M=M, Y=M @ W, Yp=state.R[j] @ W, V=M @ cv, Vp=state.R[j] @ cv,
                P=M @ cp, Pp=state.R[j] @ cp, w=ed["J"][0:3, 0:3] @ b[0:3], cv=cv, cp=cp)


def resid5(state, ed, bbar):
    """Cleared residual A (I - x^/2) - B (I + x^/2) from 5x5 matrices."""
    i, j = ed["i"], ed["j"]
    x = ed["J"] @ (state.b[i] - bbar)
    Xi = Xmat(state.R[i], state.v[i], state.p[i], 0.0)
    Xj = Xmat(state.R[j], state.v[j], state.p[j], 0.0)
    A = np.linalg.solve(Gmat(ed["Dt"]), Xj)
    Bm = Xi @ ed["Ubar"]
    return A @ (I5 - 0.5 * xhat(x)) - Bm @ (I5 + 0.5 * xhat(x))


def z_ref(state, r, aniso=False, s=1.0):
    """The claim's index map, written out literally (M3-C01)."""
    z = np.zeros(N_ANISO if aniso else N_BASE)
    for k in range(3):
        z[15 * k:15 * k + 9] = state.R[k].flatten(order="F")
        z[15 * k + 9:15 * k + 12] = state.v[k]
        z[15 * k + 12:15 * k + 15] = state.p[k]
        z[45 + 6 * k:51 + 6 * k] = state.b[k]
    for e, ed in enumerate(r["edges"]):
        o = 63 + 30 * e
        L = lift_ref(state, ed)
        z[o:o + 9] = L["Y"].flatten(order="F")
        z[o + 9:o + 18] = L["Yp"].flatten(order="F")
        z[o + 18:o + 21], z[o + 21:o + 24] = L["V"], L["Vp"]
        z[o + 24:o + 27], z[o + 27:o + 30] = L["P"], L["Pp"]
        if aniso:
            rr = resid5(state, ed, r["bbar"])
            z[124 + 6 * e:127 + 6 * e] = L["M"].T @ rr[0:3, 3]
            z[127 + 6 * e:130 + 6 * e] = L["M"].T @ rr[0:3, 4]
    z[S_IDX] = 1.0
    return s * z


def decode_ref(z):
    s = z[S_IDX]
    return LF.State([z[15 * k:15 * k + 9].reshape(3, 3, order="F") / s for k in range(3)],
                    [z[15 * k + 9:15 * k + 12] / s for k in range(3)],
                    [z[15 * k + 12:15 * k + 15] / s for k in range(3)],
                    [z[45 + 6 * k:51 + 6 * k] / s for k in range(3)])


RW_PREC = np.diag([1 / (LIT.SIGMA_BG_RW**2 * 0.5)] * 3 + [1 / (LIT.SIGMA_BA_RW**2 * 0.5)] * 3)


def J_direct(state, r, imu="iso", rot_weight=0.5, W_vp=None):
    """Cost from 5x5 residuals (M3-C24): IMU + chordal R0 + v0/p0 + GNSS + random walk."""
    tr = truth_ref()
    J = 0.0
    for ed in r["edges"]:
        rr = resid5(state, ed, r["bbar"])
        rR, rv, rp = rr[0:3, 0:3], rr[0:3, 3], rr[0:3, 4]
        M = state.R[ed["i"]] @ ed["dR"]
        if imu == "partial":
            D = 0.5 * np.trace(ed["W_R"]) * I3 - ed["W_R"]
            rt = rr[0:3, :]
            G = np.zeros((5, 5))
            G[0:3, 0:3], G[3:5, 3:5] = D, ed["G2"]
            J += np.trace(rt @ G @ rt.T)
            continue
        J += rot_weight * ed["tau_R"] * np.sum(rR**2)
        if imu == "iso":
            J += ed["tau_v"] * rv @ rv + ed["tau_p"] * rp @ rp
        elif imu == "aniso":
            rho = np.r_[M.T @ rv, M.T @ rp]
            J += rho @ (ed["W_vp"] if W_vp is None else W_vp) @ rho
        elif imu == "world":
            q = np.r_[rv, rp]
            J += q @ (ed["W_vp"] if W_vp is None else W_vp) @ q
    sp_ = LIT.SIGMA_PRIOR
    J += 0.5 / sp_**2 * np.sum((state.R[0] - tr.R[0])**2)
    J += np.sum((state.v[0] - tr.v[0])**2) / sp_**2 + np.sum((state.p[0] - tr.p[0])**2) / sp_**2
    for k in (1, 2):
        J += np.sum((state.p[k] - tr.p[k])**2) / LIT.SIGMA_GNSS**2
    for k in (0, 1):
        d = state.b[k + 1] - state.b[k]
        J += d @ RW_PREC @ d
    return float(J)


def imu_info(ed, kind="iso", W_vp=None):
    """9x9 information in the IMU factor's NavState order (rot, pos, vel)."""
    Info = np.zeros((9, 9))
    rot, pos, vel = slice(0, 3), slice(3, 6), slice(6, 9)
    if kind == "iso":
        return np.diag([ed["tau_R"]] * 3 + [ed["tau_p"]] * 3 + [ed["tau_v"]] * 3)
    if kind == "partial":
        Info[rot, rot] = ed["W_R"]
        G2 = ed["G2"]                                   # (vel, pos) order
        Info[vel, vel], Info[pos, pos] = G2[0, 0] * I3, G2[1, 1] * I3
        Info[vel, pos], Info[pos, vel] = G2[0, 1] * I3, G2[1, 0] * I3
        return Info
    if kind == "aniso":
        Info[rot, rot] = ed["tau_R"] * I3
        W = ed["W_vp"] if W_vp is None else W_vp       # (vel, pos) order
        Info[vel, vel], Info[vel, pos] = W[0:3, 0:3], W[0:3, 3:6]
        Info[pos, vel], Info[pos, pos] = W[3:6, 0:3], W[3:6, 3:6]
        return Info
    raise ValueError(kind)


def gtsam_cost(state, r, kind="iso", W_vp=None):
    """2 * graph.error of the GTSAM reference graph (M3-C26)."""
    tr = truth_ref()
    graph = gtsam.NonlinearFactorGraph()
    for ed in r["edges"]:
        f = gtsam.GalileanImuFactor(X(ed["i"]), V(ed["i"]), X(ed["j"]), V(ed["j"]), B(ed["i"]),
                                    ed["pim"])
        graph.add(f.cloneWithNewNoiseModel(noiseModel.Gaussian.Information(imu_info(ed, kind, W_vp))))
    graph.add(gtsam.PriorFactorPose3(X(0), Pose3(Rot3(tr.R[0]), tr.p[0]),
                                     noiseModel.Isotropic.Sigma(6, LIT.SIGMA_PRIOR)))
    graph.add(gtsam.PriorFactorVector(V(0), tr.v[0], noiseModel.Isotropic.Sigma(3, LIT.SIGMA_PRIOR)))
    for k in (1, 2):
        graph.add(gtsam.GPSFactor(X(k), tr.p[k], noiseModel.Isotropic.Sigma(3, LIT.SIGMA_GNSS)))
    rw_sig = np.r_[[LIT.SIGMA_BA_RW * np.sqrt(0.5)] * 3, [LIT.SIGMA_BG_RW * np.sqrt(0.5)] * 3]
    for k in (0, 1):                                    # GTSAM (accel, gyro) order
        graph.add(gtsam.BetweenFactorConstantBias(B(k), B(k + 1), ConstantBias(),
                                                  noiseModel.Diagonal.Sigmas(rw_sig)))
    vals = gtsam.Values()
    for k in range(3):
        vals.insert(X(k), Pose3(Rot3(state.R[k]), state.p[k]))
        vals.insert(V(k), np.asarray(state.v[k], float))
        vals.insert(B(k), cb(state.b[k]))
    return 2.0 * graph.error(vals)


def random_state(rng, R_scale=2.0, b_norm=0.2):
    """R ~ Exp(N(0, R_scale^2 I)), v, p ~ N(0, I), |b| ~ b_norm."""
    return LF.State([rexp(R_scale * rng.standard_normal(3)) for _ in range(3)],
                    [rng.standard_normal(3) for _ in range(3)],
                    [rng.standard_normal(3) for _ in range(3)],
                    [b_norm / np.sqrt(6) * rng.standard_normal(6) for _ in range(3)])


def claim_sampler_state(rng):
    """The C32/C35 sampler: R_k = Exp(2 N(0,I3)), v_k, p_k ~ N(0,I3), b_k ~ 0.3 N(0,I6)."""
    R = [rexp(2 * rng.standard_normal(3)) for _ in range(3)]
    v = [rng.standard_normal(3) for _ in range(3)]
    p = [rng.standard_normal(3) for _ in range(3)]
    b = [0.3 * rng.standard_normal(6) for _ in range(3)]
    return LF.State(R, v, p, b)


def perturbed(state, eps, rng):
    return LF.State([R @ rexp(eps * rng.standard_normal(3)) for R in state.R],
                    [v + eps * rng.standard_normal(3) for v in state.v],
                    [p + eps * rng.standard_normal(3) for p in state.p],
                    [b + 0.1 * eps * rng.standard_normal(6) for b in state.b])


# -- shortcuts to the code under test ---------------------------------------------------

def inst(which):
    return INST.instance(which)


@functools.lru_cache(maxsize=None)
def fams(which, aniso=False):
    return CN.build(inst(which), LF.ANISO if aniso else LF.BASE)


@functools.lru_cache(maxsize=None)
def terms(which, variant="iso"):
    layout = LF.ANISO if variant == "aniso" else LF.BASE
    return CO.terms(inst(which), layout, variant)


def fmax(z, families, names=None):
    names = list(families) if names is None else names
    return {f: max(abs(c.residual(z)) for c in families[f]) for f in names}


def m3_cost(state, which, variant="iso"):
    layout = LF.ANISO if variant == "aniso" else LF.BASE
    return CO.factored_cost(LF.z_of_state(state, inst(which), layout), terms(which, variant))


def term(which, name, variant="iso"):
    return next(t for t in terms(which, variant) if t.name == name)


# ======================================================================================
# Instance data are what GTSAM says (binds the instance to the independent reference)
# ======================================================================================

@pytest.mark.claim("M3-C01")
@pytest.mark.claim("M3-C24")
@pytest.mark.parametrize("which", ["true", "zero"])
def test_instance_matches_independent_gtsam_reference(which):
    r, I = ref(which), inst(which)
    assert np.array_equal(I.bbar, r["bbar"])
    for ed, er in zip(I.edges, r["edges"]):
        assert (ed.i, ed.j) == (er["i"], er["j"])
        assert abs(ed.Dt - er["Dt"]) == 0.0
        assert np.abs(ed.dR - er["dR"]).max() <= 1e-13
        assert np.abs(ed.dv - er["dv"]).max() <= 1e-13
        assert np.abs(ed.dp - er["dp"]).max() <= 1e-13
        assert np.abs(ed.J - er["J"]).max() <= 1e-12
        for k in ("tau_R", "tau_v", "tau_p"):
            assert abs(getattr(ed, k) / er[k] - 1) <= 1e-12, k
        assert np.abs(ed.W_vp - er["W_vp"]).max() <= 1e-9 * np.abs(er["W_vp"]).max()
        assert np.abs(ed.G2 - er["G2"]).max() <= 1e-9 * np.abs(er["G2"]).max()
        assert np.abs(ed.W_R - er["W_R"]).max() <= 1e-9 * np.abs(er["W_R"]).max()
    tr = truth_ref()
    for k in range(3):
        assert np.abs(I.truth[k][0] - tr.R[k]).max() <= 1e-12
        assert np.abs(I.truth[k][1] - tr.v[k]).max() <= 1e-12
        assert np.abs(I.truth[k][2] - tr.p[k]).max() <= 1e-12
    assert np.array_equal(I.b_true, LIT.BIAS_TRUE)
    assert np.allclose(I.rw_precision, RW_PREC, rtol=1e-15, atol=0)
    # factor-graph literals of the domain text
    assert I.sigma_prior == LIT.SIGMA_PRIOR and I.sigma_gnss == LIT.SIGMA_GNSS
    for k in (1, 2):
        assert np.abs(I.gnss[k] - tr.p[k]).max() <= 1e-12
    for got, want in zip(I.prior, (tr.R[0], tr.v[0], tr.p[0])):
        assert np.abs(got - want).max() <= 1e-15
    # the ground-truth trajectory is M2_TRAJ_V1's (independent transcription of M2's reference)
    for k, (key_v, key_p) in ((1, ("V_KF1", "P_KF1")), (2, ("V_KF2", "P_KF2"))):
        for st in (tr, LF.truth_state(I)):
            assert np.abs(st.v[k] - M2_REF[key_v]).max() <= 1e-12
            assert np.array_equal(st.b[k], LIT.BIAS_TRUE)       # truth is b_true whatever bbar
            assert np.abs(st.p[k] - M2_REF[key_p]).max() <= 1e-12
    assert np.abs(I.truth[2][0] - M2_REF["R_KF2"]).max() <= 1e-12
    if which == "true":                       # at bbar = b_true the PIM rotation is the exact dR
        assert np.abs(I.edges[0].dR - M2_REF["DR_01"]).max() <= 1e-12


@pytest.mark.claim("M3-C01")
def test_bias_boundary_converters():
    """Both directions of the single bias-order conversion (CLAUDE.md footgun 4)."""
    import src.qcqp.gtsam_boundary as GB
    b = np.array([1.0, 2, 3, 4, 5, 6])                    # [b_g; b_a]
    cbias = GB.to_gtsam_bias(b)
    assert np.array_equal(cbias.gyroscope(), b[0:3]) and np.array_equal(cbias.accelerometer(), b[3:6])
    assert np.array_equal(GB.from_gtsam_bias(ConstantBias(b[3:6], b[0:3])), b)
    assert np.array_equal(GB.from_gtsam_bias(cbias), b)


# ======================================================================================
# C01 - C03: the lifted vector and its size
# ======================================================================================

@pytest.mark.claim("M3-C01")
@pytest.mark.parametrize("which", ["true", "zero"])
def test_c01_index_map(which):
    rng = np.random.default_rng(101)
    # The defining expressions use the INSTANCE's own (dRbar, J): this test is about the
    # encoder and the index map; the instance data themselves are bound to the independent
    # GTSAM reference in test_instance_matches_independent_gtsam_reference (J to 1e-12, which
    # would move V = M J_nu b by ~1e-13 -- above this claim's 1e-14).
    r = dict(bbar=inst(which).bbar, edges=[dict(i=ed.i, j=ed.j, dR=ed.dR, J=ed.J, Dt=ed.Dt,
                                                Ubar=Xmat(ed.dR, ed.dv, ed.dp, ed.Dt))
                                           for ed in inst(which).edges])
    for _ in range(20):
        st = random_state(rng)
        z = LF.z_of_state(st, inst(which), LF.BASE)
        za = LF.z_of_state(st, inst(which), LF.ANISO)
        assert len(z) == 124 and LF.BASE.n == 124
        assert len(za) == 136 and LF.ANISO.n == 136
        assert z[123] == 1.0 and za[123] == 1.0
        for k in range(3):
            assert np.array_equal(z[15 * k:15 * k + 9], st.R[k].flatten(order="F"))
            assert np.array_equal(z[15 * k + 9:15 * k + 12], st.v[k])
            assert np.array_equal(z[15 * k + 12:15 * k + 15], st.p[k])
            assert np.array_equal(z[45 + 6 * k:51 + 6 * k], st.b[k])
        for e, ed in enumerate(r["edges"]):
            o = 63 + 30 * e
            L = lift_ref(st, ed)
            blocks = [(o, o + 9, L["Y"].flatten(order="F")), (o + 9, o + 18, L["Yp"].flatten(order="F")),
                      (o + 18, o + 21, L["V"]), (o + 21, o + 24, L["Vp"]),
                      (o + 24, o + 27, L["P"]), (o + 27, o + 30, L["Pp"])]
            for lo, hi, val in blocks:
                assert np.abs(z[lo:hi] - val).max() <= 1e-14, (e, lo)
                assert np.abs(za[lo:hi] - val).max() <= 1e-14, (e, lo)
            rr = resid5(st, ed, r["bbar"])
            rho_v, rho_p = L["M"].T @ rr[0:3, 3], L["M"].T @ rr[0:3, 4]
            # rho is |r|-sized (|r| up to ~10 here): 1e-14 relative to max(1, |r|)
            scale = max(1.0, np.abs(rr).max())
            assert np.abs(za[124 + 6 * e:127 + 6 * e] - rho_v).max() <= 1e-14 * scale
            assert np.abs(za[127 + 6 * e:130 + 6 * e] - rho_p).max() <= 1e-14 * scale
        # z_of_state's homogenising scale
        assert np.array_equal(LF.z_of_state(st, inst(which), LF.BASE, s=-2.0), -2.0 * z)
        # decoder round trip at several scales, every block (one multiply and one divide:
        # at most 2 ulp relative -> 4.5e-16 * max(1, |x|))
        for sc in (1.0, -1.0, -2.5, 0.3):
            for layout, zz in ((LF.BASE, z), (LF.ANISO, za)):
                d = LF.state_of_z(sc * zz, layout)
                for k in range(3):
                    for got, want in ((d.R[k], st.R[k]), (d.v[k], st.v[k]), (d.p[k], st.p[k]),
                                      (d.b[k], st.b[k])):
                        assert np.abs(got - want).max() <= 4.5e-16 * max(1.0, np.abs(want).max()), sc
    assert np.array_equal(LF.BASE.bias_indices, np.arange(45, 63))
    assert np.array_equal(LF.ANISO.bias_indices, np.arange(45, 63))


@pytest.mark.claim("M3-C01")
def test_decoder_projection_is_nearest_rotation():
    """state_of_z(project=True) returns the nearest PROPER rotation, also when the nearest
    orthogonal matrix is a reflection (reference: SVD with the det correction, written here)."""
    rng = np.random.default_rng(111)
    for _ in range(200):
        A = rexp(2 * rng.standard_normal(3)) @ np.diag([1.0, 1.0, -1.0]) + 0.1 * rng.standard_normal((3, 3))
        U, _, Vt = np.linalg.svd(A)
        assert np.linalg.det(U @ Vt) < 0                       # nearest orthogonal is a reflection
        ref_R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        z = np.zeros(124)
        z[0:9], z[S_IDX] = A.flatten(order="F"), 1.0
        R = LF.state_of_z(z, project=True).R[0]
        assert abs(np.linalg.det(R) - 1) <= 1e-12
        assert np.abs(R - ref_R).max() <= 1e-12
        assert np.abs(LF.project_so3(A) - ref_R).max() <= 1e-12
        # the rounding path at arbitrary sign / scale of s: an SDP eigenvector sc * z
        for sc in (-1.0, -2.5, 0.3):
            Rs = LF.state_of_z(sc * z, project=True).R[0]
            assert np.abs(Rs - ref_R).max() <= 1e-12, sc


@pytest.mark.claim("M3-C01")
def test_decoder_rounding_of_perturbed_lifted_states():
    """project=True on perturbed lifted z at s in {1, -1, -2.5, 0.3}: every rotation block is the
    nearest proper rotation of (block / s), computed here by SVD; v, p, b are block / s."""
    rng = np.random.default_rng(112)
    for _ in range(30):
        st = random_state(rng)
        z = z_ref(st, ref("zero")) + 1e-2 * rng.standard_normal(124)
        z[S_IDX] = 1.0
        for sc in (1.0, -1.0, -2.5, 0.3):
            d = LF.state_of_z(sc * z, project=True)
            for k in range(3):
                B = z[15 * k:15 * k + 9].reshape(3, 3, order="F")
                U, _, Vt = np.linalg.svd(B)
                Rref = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
                assert np.abs(d.R[k] - Rref).max() <= 1e-12, (sc, k)
                assert np.abs(d.v[k] - z[15 * k + 9:15 * k + 12]).max() <= 1e-14
                assert np.abs(d.b[k] - z[45 + 6 * k:51 + 6 * k]).max() <= 1e-14


@pytest.mark.claim("M3-C01")
def test_decoder_degenerate_inputs():
    """Degenerate rotation blocks still project to a proper rotation; s ~ 0 raises."""
    rng = np.random.default_rng(113)
    u, v = rng.standard_normal(3), rng.standard_normal(3)
    blocks = {"zero": np.zeros((3, 3)), "rank1": np.outer(u, v), "minusI": -I3,
              "rank2": rexp([0.3, -0.2, 0.9]) @ np.diag([2.0, 0.5, 0.0])}
    for name, A in blocks.items():
        for sc in (1.0, -1.0, -2.5):
            z = np.zeros(124)
            z[0:9], z[S_IDX] = sc * A.flatten(order="F"), sc
            R = LF.state_of_z(z, project=True).R[0]
            assert np.abs(R.T @ R - I3).max() <= 1e-12, (name, sc)
            assert abs(np.linalg.det(R) - 1) <= 1e-12, (name, sc, np.linalg.det(R))
    z = z_ref(truth_ref(), ref("zero"))
    for bad in (0.0, 1e-14, -1e-14, np.nan):
        zb = z.copy()
        zb[S_IDX] = bad
        with pytest.raises(ValueError):
            LF.state_of_z(zb)
        with pytest.raises(ValueError):
            LF.state_of_z(zb, project=True)
    zs = 1e-6 * z                                             # small but meaningful scale decodes
    assert np.abs(LF.state_of_z(zs).v[1] - truth_ref().v[1]).max() <= 1e-12


#: the decoder's documented relative floor, RETYPED here (lifting.py docstring of state_of_z:
#: raise iff |s| <= 1e-12 * max(max|z|, 1e-300)); not imported from src
S_FLOOR_DOC = 1e-12


def _decodes(z, **kw):
    try:
        LF.state_of_z(z, **kw)
    except ValueError:
        return False
    return True


@pytest.mark.claim("M3-C01")
def test_decoder_s_guard_threshold_tie_and_scale():
    """The s ~ 0 guard: relative threshold (0.5x raises, 2x decodes, at two scales), the exact
    tie |s| = 1e-12 max|z| raises and the next double decodes, and the guard is scale invariant
    (c z decodes to the state of z for c down to 1e-300, both signs)."""
    zt = z_ref(truth_ref(), ref("zero"))
    for project in (False, True):
        # threshold, at max|z| ~ 1.9 and at max|z| ~ 1.9e6 (an absolute floor fails one of the two)
        for c in (1.0, 1e6):
            zb = c * zt
            m = np.abs(np.delete(zb, S_IDX)).max()            # |s| stays below this, so max|z| = m
            for sgn in (1.0, -1.0):
                zb[S_IDX] = sgn * 0.5 * S_FLOOR_DOC * m
                assert not _decodes(zb, project=project), (c, sgn)
                zb[S_IDX] = sgn * 2.0 * S_FLOOR_DOC * m
                d = LF.state_of_z(zb, project=project)
                assert np.abs(d.v[1] * zb[S_IDX] - zb[24:27]).max() <= 1e-15 * c, (c, sgn)
        # exact tie: max|z| = 1 exactly (a power of two keeps 1e-12 * max|z| exact)
        for scale in (1.0, 2.0**-900, 2.0**600):
            z = np.zeros(124)
            z[9] = scale                                       # v_0[0], the max entry
            z[30:39] = 0.25 * scale * np.arange(1, 10) / 9      # some smaller entries
            tie = S_FLOOR_DOC * scale
            assert tie == 1e-12 * scale and S_FLOOR_DOC * np.abs(z).max() == tie
            z[S_IDX] = tie
            assert not _decodes(z, project=project), scale
            z[S_IDX] = -tie
            assert not _decodes(z, project=project), scale
            z[S_IDX] = np.nextafter(tie, np.inf)
            d = LF.state_of_z(z, project=project)
            assert d.v[0][0] == scale / z[S_IDX]
            z[S_IDX] = -np.nextafter(tie, np.inf)
            assert _decodes(z, project=project), scale
    # scale sweep: c z decodes to the same state (v, p, b and R exactly up to rounding)
    st = truth_ref()
    for c in (1e-13, 1e-200, 1e-300, -1e-13, -1e-300):
        d = LF.state_of_z(c * zt)
        for k in range(3):
            for got, want in ((d.R[k], st.R[k]), (d.v[k], st.v[k]), (d.p[k], st.p[k]),
                              (d.b[k], LIT.BIAS_TRUE)):
                assert np.abs(got - want).max() <= 1e-12 * max(1.0, np.abs(want).max()), c
        assert np.abs(LF.state_of_z(c * zt, project=True).R[1] - st.R[1]).max() <= 1e-12, c


@pytest.mark.claim("M3-C01")
def test_decoder_rejects_non_finite_z():
    """Any NaN / +-inf entry of z (not only s) raises ValueError naming non-finite entries,
    never a misleading 's is zero' message and never a NaN-carrying state."""
    zt = z_ref(truth_ref(), ref("zero"))
    for idx in (9, 26, 50, 100, S_IDX):                     # v_0, p_1, b_0, a lifting, s
        for bad in (np.nan, np.inf, -np.inf):
            zb = zt.copy()
            zb[idx] = bad
            for project in (False, True):
                with pytest.raises(ValueError, match="non-finite") as ei:
                    LF.state_of_z(zb, project=project)
                assert "zero" not in str(ei.value), (idx, bad)
                assert str(idx) in str(ei.value)
    # the aniso layout too (a NaN in a rho slot)
    za = z_ref(truth_ref(), ref("zero"), aniso=True)
    za[130] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        LF.state_of_z(za, LF.ANISO)


@pytest.mark.claim("M3-C01")
def test_state_copy_is_independent():
    """State.copy() is a deep copy of every block: an in-place edit of the copy leaves the
    original unchanged (and vice versa)."""
    rng = np.random.default_rng(114)
    st = random_state(rng)
    saved = {f: [np.array(x, copy=True) for x in getattr(st, f)] for f in ("R", "v", "p", "b")}
    c = st.copy()
    for f in ("R", "v", "p", "b"):
        for k in range(3):
            assert getattr(c, f)[k] is not getattr(st, f)[k], (f, k)
            assert np.array_equal(getattr(c, f)[k], saved[f][k]), (f, k)
            getattr(c, f)[k].flat[0] += 1.0                   # in-place edit of the copy
            assert np.array_equal(getattr(st, f)[k], saved[f][k]), (f, k)
    c2 = st.copy()
    st.b[0][0] += 7.0                                         # edit of the original
    assert c2.b[0][0] == saved["b"][0][0]


@pytest.mark.claim("M3-C24")
def test_cost_terms_rejects_unknown_variant():
    for layout in (LF.BASE, LF.ANISO):
        for bad in ("bogus", "ISO", "isotropic", ""):
            with pytest.raises(ValueError, match="variant"):
                CO.terms(inst("zero"), layout, bad)


@pytest.mark.claim("M3-C01")
def test_package_exports():
    """src.qcqp re-exports the layouts and entry points under the documented names (sizes are
    the claim's literals 124 / 136)."""
    import src.qcqp as PKG
    assert PKG.BASE.n == 124 and not PKG.BASE.aniso
    assert PKG.ANISO.n == 136 and PKG.ANISO.aniso
    for name in ("build_instance", "BASE", "ANISO", "Layout", "State", "z_of_state",
                 "state_of_z", "truth_state"):
        assert hasattr(PKG, name), name
    assert PKG.build_instance is INST.build_instance
    for name in ("BASE", "ANISO", "Layout", "State", "z_of_state", "state_of_z", "truth_state"):
        assert getattr(PKG, name) is getattr(LF, name), name
    I = PKG.build_instance(np.zeros(6))
    assert np.array_equal(I.bbar, np.zeros(6)) and len(I.edges) == 2


@pytest.mark.claim("M3-C01")
def test_truth_state_blocks_are_independent_arrays():
    """truth_state hands out a mutable State: every block is its own array, sharing memory with
    neither another keyframe's block nor the Instance (an in-place edit of the returned state
    must not reach b_1, b_2, Instance.b_true, Instance.truth or a later truth_state call).
    A FRESH instance is used so that a regression cannot poison the lru-cached one."""
    I = INST.build_instance(LIT.BIAS_TRUE.copy())
    tr = truth_ref()
    st = LF.truth_state(I)
    for f in ("R", "v", "p", "b"):
        blocks = getattr(st, f)
        for k in range(3):
            for k2 in range(k + 1, 3):
                assert not np.shares_memory(blocks[k], blocks[k2]), (f, k, k2)
    for k in range(3):
        assert not np.shares_memory(st.b[k], I.b_true), k
        for q, f in enumerate(("R", "v", "p")):
            assert not np.shares_memory(getattr(st, f)[k], I.truth[k][q]), (f, k)
    for k in range(3):                                      # in-place edits of the returned state
        st.b[k][0] += 1.0
        st.v[k][0] += 1.0
        st.p[k][0] += 1.0
        st.R[k][0, 0] += 1.0
    assert np.array_equal(I.b_true, LIT.BIAS_TRUE)
    fresh = LF.truth_state(I)
    for k in range(3):
        assert np.array_equal(fresh.b[k], LIT.BIAS_TRUE), k
        assert st.b[k][0] == LIT.BIAS_TRUE[0] + 1.0, k      # each edit applied exactly once
        for got, want in ((fresh.R[k], tr.R[k]), (fresh.v[k], tr.v[k]), (fresh.p[k], tr.p[k]),
                          (I.truth[k][0], tr.R[k]), (I.truth[k][1], tr.v[k]), (I.truth[k][2], tr.p[k])):
            assert np.abs(got - want).max() <= 1e-9 * max(1.0, np.abs(want).max()), k


#: the iso (base) term names, RETYPED from the M3-C24 term list (not read from src)
ISO_TERM_NAMES = ["imu0.rot", "imu0.vel", "imu0.pos", "imu1.rot", "imu1.vel", "imu1.pos",
                  "prior.R0", "prior.v0", "prior.p0", "gnss1", "gnss2", "rw0", "rw1"]


@pytest.mark.claim("M3-C24")
def test_cost_terms_default_variant_is_iso():
    """CO.terms(inst) with no variant / layout is the iso base cost: the 13 iso terms, and its Q
    reproduces the test-side iso 5x5-residual cost J_direct (not the partial weighting)."""
    for which in ("true", "zero"):
        T0 = CO.terms(inst(which))
        assert [t.name for t in T0] == ISO_TERM_NAMES, which
        Q = CO.assemble_Q(T0)
        assert Q.shape == (124, 124)
        rng = np.random.default_rng(241)
        r = ref(which)
        worst = sep = 0.0
        for _ in range(50):
            st = random_state(rng)
            z = z_ref(st, r)
            Jd, Jp = J_direct(st, r, imu="iso"), J_direct(st, r, imu="partial")
            worst = max(worst, abs(z @ Q @ z - Jd) / max(1.0, Jd))
            sep = max(sep, abs(Jp - Jd) / max(1.0, Jd))
        assert worst <= 1e-12, (which, worst)
        assert sep > 1e-6, (which, sep)                     # the check can tell iso from partial


# NOT claim-bound: no M3 claim states make_params' input validation (the claims only fix the
# domain MakeSharedU(9.81)); binding it to a claim would misreport what that claim's test checks.
def test_make_params_rejects_non_z_up_gravity():
    """make_params guards GTSAM's MakeSharedU convention (gravity (0, 0, -g), finite g > 0) with
    an explicit ValueError (not an assert, which python -O strips): wrong sign / x-y components,
    wrong shape (length 2 or 4, 2-D, 0-d) and +-inf / NaN in any component all raise with the
    guard's own message; z-up gravity is passed through."""
    import src.qcqp.gtsam_boundary as GB
    bad = [[0.0, 0.0, 9.81], [0.1, 0.0, -9.81], [0.0, -0.1, -9.81], [0.0, 0.0, 0.0],
           [0.0, 0.0, -9.81, 0.0], [0.0, 0.0, -9.81, -9.81], [0.0, -9.81], [[0.0, 0.0, -9.81]],
           [[0.0], [0.0], [-9.81]], -9.81]
    for c in range(3):
        for v in (np.inf, -np.inf, np.nan):
            g = [0.0, 0.0, -9.81]
            g[c] = v
            bad.append(g)
    for g in bad:
        with pytest.raises(ValueError, match="z-up"):
            GB.make_params(g, 1e-3, 1e-2, 1e-4)
    params = GB.make_params([0.0, 0.0, -9.81], 1e-3, 1e-2, 1e-4)
    assert np.array_equal(np.asarray(params.n_gravity), np.array([0.0, 0.0, -9.81]))


@pytest.mark.claim("M3-C01")
def test_decoder_s_guard_floor_below_1e_minus_300():
    """The documented floor max(max|z|, 1e-300): below max|z| = 1e-300 the threshold stays at
    1e-12 * 1e-300 = 1e-312 (literals retyped), so tiny-but-nonzero s still raises there."""
    for project in (False, True):
        z = np.zeros(124)
        z[9] = 1e-301                                          # max|z| = 1e-301 < floor
        z[S_IDX] = 5e-313                                      # > 1e-12 max|z|, but < 1e-312
        assert not _decodes(z, project=project)
        z[S_IDX] = 2e-312                                      # above the floored threshold
        assert _decodes(z, project=project)
        z = np.zeros(124)
        z[S_IDX] = 1e-320                                      # only s is nonzero
        assert not _decodes(z, project=project)
        z[30] = 1e-310
        z[S_IDX] = 1e-312 / 4                                  # max|z| = 1e-310 < floor
        assert not _decodes(z, project=project)


@pytest.mark.claim("M3-C01")
def test_decoder_s_guard_counts_aniso_rho_entries():
    """The guard's max|z| runs over ALL n = 136 entries of an aniso z, the rho slots
    [124, 136) included: with every other entry O(1), s = 1 and the rho entries +-1e13
    (so max|z| = 1e13 lives only there) the decoder must raise; at 1e11 it decodes
    (threshold 1e-12 * 1e11 = 0.1 < |s| = 1)."""
    zt = z_ref(truth_ref(), ref("zero"), aniso=True)
    assert zt.shape == (N_ANISO,) and zt[S_IDX] == 1.0 and np.abs(zt[:N_BASE]).max() < 10.0
    for mag in (1e13, -1e13):
        for idx in (slice(N_BASE, N_ANISO), slice(N_BASE, N_BASE + 1), slice(N_ANISO - 1, N_ANISO)):
            za = zt.copy()
            za[idx] = mag
            for project in (False, True):
                with pytest.raises(ValueError, match="zero relative"):
                    LF.state_of_z(za, LF.ANISO, project=project)
    za = zt.copy()
    za[N_BASE:] = 1e11
    for project in (False, True):
        assert _decodes(za, layout=LF.ANISO, project=project)


@pytest.mark.claim("M3-C01")
def test_decoder_large_scales_decode():
    """c z decodes to the state of z at very large |c| too (no upper cut-off on s)."""
    zt = z_ref(truth_ref(), ref("zero"))
    st = truth_ref()
    assert np.abs(zt).max() < 10.0                             # so 1e307 * zt stays finite
    for c in (1e100, 1e300, 1e305, -1e305, 1e307):
        for project in (False, True):
            d = LF.state_of_z(c * zt, project=project)
            for k in range(3):
                for got, want in ((d.R[k], st.R[k]), (d.v[k], st.v[k]), (d.p[k], st.p[k]),
                                  (d.b[k], LIT.BIAS_TRUE)):
                    assert np.abs(got - want).max() <= 1e-12 * max(1.0, np.abs(want).max()), c


@pytest.mark.claim("M3-C01")
def test_project_so3_det_correction_is_exact():
    """project_so3(A) = U diag(1, 1, +-1) V^T with the correction EXACTLY +-1 (chosen here from
    the sign of det(U V^T), test-side SVD): over 1e4 random A it matches to <= 2 ulp of 1 (a
    correction of det(U V^T) = +-(1 + few eps) instead of +-1 differs by ~10 ulp; this pin is what
    kills that defect).  The orthogonality / det bounds are a loose sanity check only, set to
    2e-14: measured clean max over 41 seeds (115, 0-39) is 3.3e-15 / 3.4e-15, i.e. ~6x headroom
    (the old 4e-15 had 1.2x and depended on the LAPACK build)."""
    rng = np.random.default_rng(115)
    worst_ref = worst_orth = worst_det = 0.0
    for _ in range(10000):
        A = rng.standard_normal((3, 3))
        U, _, Vt = np.linalg.svd(A)
        d3 = 1.0 if np.linalg.det(U @ Vt) > 0 else -1.0
        Rref = U @ np.diag([1.0, 1.0, d3]) @ Vt
        R = LF.project_so3(A)
        worst_ref = max(worst_ref, np.abs(R - Rref).max())
        worst_orth = max(worst_orth, np.abs(R.T @ R - I3).max())
        worst_det = max(worst_det, abs(np.linalg.det(R) - 1.0))
    assert worst_ref <= 4.5e-16, worst_ref
    assert worst_orth <= 2e-14, worst_orth
    assert worst_det <= 2e-14, worst_det


# NOT claim-bound: no M3 claim states that the Instance is immutable (M4 risk R4 of verifier
# pass 5); this guards the lru-cached instance against poisoning by an in-place write.
def test_cached_instance_arrays_are_read_only():
    """Every numpy array of the cached Instance and of its edges is read-only (an in-place write
    raises ValueError and leaves the value unchanged), the dicts are read-only mappings, and a
    later instance() call still returns the reference data."""
    for which in ("true", "zero"):
        I = INST.instance(which)
        arrays = [("bbar", I.bbar), ("g", I.g), ("b_true", I.b_true), ("rw", I.rw_precision)]
        arrays += [(f"truth{k}.{q}", I.truth[k][q]) for k in range(3) for q in range(3)]
        arrays += [(f"prior{q}", I.prior[q]) for q in range(3)]
        arrays += [(f"gnss{k}", I.gnss[k]) for k in (1, 2)]
        for e, ed in enumerate(I.edges):
            for name in ("dR", "dv", "dp", "J", "W_R", "G2", "W_vp"):
                arrays.append((f"e{e}.{name}", getattr(ed, name)))
            arrays += [(f"e{e}.cov{key}", blk) for key, blk in ed.cov.items()]
            arrays += [(f"e{e}.J_R", ed.J_R), (f"e{e}.J_nu", ed.J_nu)]   # views inherit the flag
        assert len(arrays) == 4 + 9 + 3 + 2 + 2 * (7 + 9 + 2)
        for name, a in arrays:
            assert isinstance(a, np.ndarray) and not a.flags.writeable, (which, name)
            before = a.copy()
            with pytest.raises(ValueError, match="read-only"):
                a.flat[0] = a.flat[0] + 1.0
            with pytest.raises(ValueError, match="read-only"):
                a += 1.0
            assert np.array_equal(a, before), (which, name)
        with pytest.raises(TypeError):
            I.gnss[1] = np.zeros(3)
        with pytest.raises(TypeError):
            I.edges[0].cov[("rot", "rot")] = np.zeros((3, 3))
        assert INST.instance(which) is I
    assert np.array_equal(INST.instance("true").b_true, LIT.BIAS_TRUE)
    assert np.array_equal(INST.instance("zero").bbar, np.zeros(6))


@pytest.mark.claim("M3-C02")
def test_c02_count_and_jside_velocity_lifting_is_needed():
    assert 15 * 3 + 6 * 3 + 24 * 2 + 1 == 112
    assert 124 - 112 == 2 * (3 + 3)
    for which in ("true", "zero"):
        for e in range(2):
            F = term(which, f"imu{e}.vel").F
            o = 63 + 30 * e
            assert np.array_equal(F[:, o + 21:o + 24], -0.5 * I3)   # V'
            assert np.array_equal(F[:, o + 18:o + 21], -0.5 * I3)   # V


def _rai_text():
    raw = gzip.decompress((FIXTURES / "rai_2609.38048v1.html.gz").read_bytes()).decode("utf8")
    raw = re.sub(r"<annotation[^>]*>(.*?)</annotation>", r" \1 ", raw, flags=re.S)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", _html.unescape(raw))


@pytest.mark.claim("M3-C03")
def test_c03_rai_published_sizes():
    """MATH CHECK on published numbers; touches no src by nature (the claim is about Rai's paper)."""
    triples = [(8, 2, 340), (16, 2, 652), (8, 1, 334), (16, 1, 646), (16, 2, 652), (24, 1, 958),
               (24, 3, 970)]
    text = _rai_text()
    # the triples are the paper's own: Sec. VII-B sentence and Table II rows
    assert re.search(r"K\s*=\s*8.{0,80}M\s*=\s*2.{0,80}n\s*=\s*340", text)
    assert re.search(r"K\s*=\s*16.{0,120}n\s*=\s*652", text)
    table = text[text.index("EuRoC MH_01"):]
    for K, M, n in triples[2:]:
        assert re.search(rf"\b{K} {M} {n}\b", table), (K, M, n)
    for K, M, n in triples:
        assert n == 39 * K + 6 * M + 16
        direct = 15 * K + 24 * (K - 1) + 6 * M + 1
        assert direct == 39 * K + 6 * M - 23
        assert n - direct == 39


# ======================================================================================
# C04 - C08: bias Jacobian, cleared residual, affine forms
# ======================================================================================

def _time_varying(seed, n=150, dt=2.5e-3):
    rng = np.random.default_rng(seed)
    k = np.arange(n)[:, None] * dt
    a0, w0 = rng.normal(0, 1, 3) + [0, 0, 9.8], rng.normal(0, 0.4, 3)
    fa, fw, ph = rng.uniform(0.5, 3, 3), rng.uniform(0.5, 3, 3), rng.uniform(0, 6, 3)
    acc = a0 + 0.8 * np.sin(fa * k + ph)
    gyro = w0 + 0.5 * np.cos(fw * k + ph)
    return gyro, acc, dt


@pytest.mark.claim("M3-C04")
def test_c04_bias_jacobian_structure():
    cases = [(fixture_pim(e, bb), bb) for e in range(2) for bb in (LIT.BIAS_TRUE, np.zeros(6))]
    for seed in range(5):
        gyro, acc, dt = _time_varying(seed)
        bb = np.random.default_rng(50 + seed).normal(0, 0.05, 6)
        cases.append((make_pim(gyro, acc, dt, bb), bb))
    for pim, bb in cases:
        J = fd_J(pim, bb, 1e-6)
        assert np.abs(J[0:3, 3:6]).max() == 0.0
        assert np.abs(J[9, :]).max() == 0.0
        assert np.abs(J[3:6, 3:6] - J[0:3, 0:3]).max() <= 1e-8
        assert 0.05 <= np.abs(J[0:3, 0:3]).max() <= 1.0
        # predict's correction is exactly linear in the Log chart, which is why the instance
        # may use a larger step: h = 1e-2 agrees with h = 1e-6 to FD round-off (eps/h ~ 1e-10)
        assert np.abs(fd_J(pim, bb, 1e-2) - J).max() <= 1e-8
    # proof step: the Gal(3) right Jacobian series has equal diagonal blocks, and agrees with a
    # finite-difference right Jacobian of Gal3.Expmap
    rng = np.random.default_rng(404)
    for _ in range(200):
        x = rng.standard_normal(10)
        x *= 0.8 / np.linalg.norm(x)
        Jr = _right_jacobian_series(x)
        h = 1e-6
        E0inv = np.linalg.inv(gexp(x))
        Jfd = np.zeros((10, 10))
        for k in range(10):
            d = np.zeros(10)
            d[k] = h
            Jfd[:, k] = (gtsam.Gal3.Logmap(gtsam.Gal3(E0inv @ gexp(x + d)))
                         - gtsam.Gal3.Logmap(gtsam.Gal3(E0inv @ gexp(x - d)))) / (2 * h)
        assert np.abs(Jr - Jfd).max() <= 1e-8
        assert np.array_equal(Jr[0:3, 0:3], Jr[3:6, 3:6])
        assert np.array_equal(Jr[0:3, 0:3], Jr[6:9, 6:9])
    # the instance's J is this J (project column order)
    for which in ("true", "zero"):
        for e, ed in enumerate(inst(which).edges):
            assert np.abs(ed.J - fd_J(fixture_pim(e, ref(which)["bbar"]), ref(which)["bbar"],
                                      1e-6)).max() <= 1e-8


@pytest.mark.claim("M3-C04")
def test_c04_instance_J_structural_zeros_exact_and_J_accuracy():
    """C04's structural zeros hold EXACTLY in the shipped instance J (not to a tolerance), and
    the instance J matches the exact linear map (least-squares fit, residual ~1e-15) to 5e-14.
    IMPLEMENTATION PIN beyond C04's 1e-8: h = 1e-3 central FD is only 2.3e-13 .. 6.1e-13
    accurate here, and it sets the floor on family residuals at independently lifted states."""
    for which in ("true", "zero"):
        for e, ed in enumerate(inst(which).edges):
            assert np.all(ed.J[0:3, 3:6] == 0.0), (which, e, np.abs(ed.J[0:3, 3:6]).max())
            assert np.all(ed.J[9, :] == 0.0), (which, e)
            Jx, fit = ls_J(fixture_pim(e, ref(which)["bbar"]), ref(which)["bbar"], seed=10 + e)
            assert fit <= 1e-14, fit                            # the map is linear
            assert np.abs(ed.J - Jx).max() <= 5e-14, (which, e, np.abs(ed.J - Jx).max())


def _vee5(A):
    return np.r_[A[2, 1], A[0, 2], A[1, 0], A[0:3, 3], A[0:3, 4], A[3, 4]]


def _ad(x):
    """ad_x y = vee([x^, y^]), built column by column from the matrix commutator."""
    X_ = xhat(x)
    cols = []
    for k in range(10):
        y = np.zeros(10)
        y[k] = 1.0
        Y_ = xhat(y)
        cols.append(_vee5(X_ @ Y_ - Y_ @ X_))
    return np.array(cols).T


def _right_jacobian_series(x, terms_=40):
    A = -_ad(x)
    out, P, fact = np.zeros((10, 10)), np.eye(10), 1.0
    for n in range(terms_):
        fact *= (n + 1)
        out = out + P / fact                   # (-ad_x)^n / (n+1)!
        P = P @ A
    return out


def _random_Xi(rng):
    return rexp(2 * rng.standard_normal(3)), rng.standard_normal(3), rng.standard_normal(3)


@pytest.mark.claim("M3-C05")
def test_c05_cleared_residual():
    """MATH CHECK (test-side 5x5 algebra only, no src); src-bound twin:
    test_c05_shipped_residual_vanishes_on_cayley_endpoints."""
    rng = np.random.default_rng(5)
    worst, worst_t = 0.0, 0.0
    for which in ("true", "zero"):
        for ed in ref(which)["edges"]:
            Dt = ed["Dt"]
            Gi = np.linalg.inv(Gmat(Dt))
            for _ in range(1000):
                R, v, p = _random_Xi(rng)
                b = 0.2 / np.sqrt(6) * rng.standard_normal(6)
                x = ed["J"] @ (b - ref(which)["bbar"])
                Xi = Xmat(R, v, p, 0.0)
                Xj = Gmat(Dt) @ Xi @ ed["Ubar"] @ cay5(x)
                worst_t = max(worst_t, abs(Xj[3, 4]))
                A, Bm = Gi @ Xj, Xi @ ed["Ubar"]
                r = A @ (I5 - 0.5 * xhat(x)) - Bm @ (I5 + 0.5 * xhat(x))
                worst = max(worst, np.abs(r).max())
                assert A[3, 4] == pytest.approx(Dt, abs=1e-15) and Bm[3, 4] == pytest.approx(Dt, abs=1e-15)
                # arbitrary X_j (t = 0): bottom rows exactly 0
                Rj, vj, pj = _random_Xi(rng)
                Xj2 = Xmat(Rj, vj, pj, 0.0)
                A2 = Gi @ Xj2
                r2 = A2 @ (I5 - 0.5 * xhat(x)) - Bm @ (I5 + 0.5 * xhat(x))
                assert np.all(r2[3:, :] == 0.0)
                assert A2[3, 4] == Dt
                assert np.abs(r2).max() > 1e-6
    assert worst <= 1e-12, worst
    assert worst_t <= 1e-12


def _inst_edge(which, e):
    ed = inst(which).edges[e]
    return dict(i=ed.i, j=ed.j, Dt=ed.Dt, J=ed.J, Ubar=Xmat(ed.dR, ed.dv, ed.dp, ed.Dt))


def _src_top_block(st, which, e):
    """The shipped affine forms evaluated on the shipped encoder: the 3x5 top block."""
    z = LF.z_of_state(st, inst(which))
    rR, rv, rp = CO.edge_residual_forms(inst(which), LF.BASE, e)
    return np.column_stack([np.einsum("abn,n->ab", rR, z), rv @ z, rp @ z])


@pytest.mark.claim("M3-C05")
def test_c05_shipped_residual_vanishes_on_cayley_endpoints():
    """Binds C05 to src: on X_j := G X_i Ubar Cay(x) the shipped forms vanish, off it they don't."""
    rng = np.random.default_rng(55)
    for which in ("true", "zero"):
        bbar = inst(which).bbar
        for e in range(2):
            ed = _inst_edge(which, e)
            for _ in range(100):
                st = random_state(rng)
                x = ed["J"] @ (st.b[e] - bbar)
                Xj = Gmat(ed["Dt"]) @ Xmat(st.R[e], st.v[e], st.p[e], 0.0) @ ed["Ubar"] @ cay5(x)
                st.R[e + 1], st.v[e + 1], st.p[e + 1] = Xj[0:3, 0:3], Xj[0:3, 3], Xj[0:3, 4]
                assert np.abs(_src_top_block(st, which, e)).max() <= 1e-12 * max(1.0, np.abs(Xj).max())
                st2 = st.copy()
                st2.v[e + 1] = st.v[e + 1] + 1e-3 * rng.standard_normal(3)
                assert np.abs(_src_top_block(st2, which, e)).max() > 1e-6


@pytest.mark.claim("M3-C06")
def test_c06_residual_factorisation():
    """MATH CHECK (no src); src-bound twin: test_c06_shipped_residual_factorisation."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for which in ("true", "zero"):
        for ed in ref(which)["edges"]:
            for _ in range(1000):
                R, v, p = _random_Xi(rng)
                b = 0.2 / np.sqrt(6) * rng.standard_normal(6)
                x = ed["J"] @ (b - ref(which)["bbar"])
                eps = np.r_[0.3 / np.sqrt(9) * rng.standard_normal(9), 0.0]
                Xi = Xmat(R, v, p, 0.0)
                Xj = Gmat(ed["Dt"]) @ Xi @ ed["Ubar"] @ cay5(x) @ gexp(eps)
                A, Bm = np.linalg.solve(Gmat(ed["Dt"]), Xj), Xi @ ed["Ubar"]
                r = A @ (I5 - 0.5 * xhat(x)) - Bm @ (I5 + 0.5 * xhat(x))
                pred = Xi @ ed["Ubar"] @ cay5(x) @ (gexp(eps) - I5) @ (I5 - 0.5 * xhat(x))
                worst = max(worst, np.abs(r - pred).max())
    assert worst <= 1e-12, worst


@pytest.mark.claim("M3-C06")
def test_c06_shipped_residual_factorisation():
    """Binds C06 to src: the shipped forms equal the top block of X_i Ubar Cay(x)(Exp(eps)-I)(I-x^/2)."""
    rng = np.random.default_rng(66)
    worst = 0.0
    for which in ("true", "zero"):
        bbar = inst(which).bbar
        for e in range(2):
            ed = _inst_edge(which, e)
            for _ in range(100):
                st = random_state(rng)
                x = ed["J"] @ (st.b[e] - bbar)
                eps = np.r_[0.1 * rng.standard_normal(9), 0.0]
                Xi = Xmat(st.R[e], st.v[e], st.p[e], 0.0)
                Xj = Gmat(ed["Dt"]) @ Xi @ ed["Ubar"] @ cay5(x) @ gexp(eps)
                st.R[e + 1], st.v[e + 1], st.p[e + 1] = Xj[0:3, 0:3], Xj[0:3, 3], Xj[0:3, 4]
                pred = Xi @ ed["Ubar"] @ cay5(x) @ (gexp(eps) - I5) @ (I5 - 0.5 * xhat(x))
                worst = max(worst, np.abs(_src_top_block(st, which, e) - pred[0:3, :]).max())
    assert worst <= 1e-12, worst


@pytest.mark.claim("M3-C07")
@pytest.mark.parametrize("which", ["true", "zero"])
def test_c07_affine_forms_equal_5x5_residual(which):
    rng = np.random.default_rng(7)
    I, r = inst(which), ref(which)
    worst = 0.0
    for _ in range(500):
        st = random_state(rng)
        z = z_ref(st, r)
        for e, ed in enumerate(r["edges"]):
            rR, rv, rp = CO.edge_residual_forms(I, LF.BASE, e)
            direct = resid5(st, ed, r["bbar"])
            worst = max(worst, np.abs(np.einsum("abn,n->ab", rR, z) - direct[0:3, 0:3]).max(),
                        np.abs(rv @ z - direct[0:3, 3]).max(), np.abs(rp @ z - direct[0:3, 4]).max())
    assert worst <= 1e-12, worst


@pytest.mark.claim("M3-C08")
def test_c08_lifted_coefficients_and_lifting_dependence():
    bias_cols = np.arange(45, 63)
    for which in ("true", "zero"):
        for e in range(2):
            o = 63 + 30 * e
            Fr = term(which, f"imu{e}.rot").F
            Fv = term(which, f"imu{e}.vel").F
            Fp = term(which, f"imu{e}.pos").F
            assert Fr.shape == (9, 124)
            assert np.array_equal(Fr[:, o:o + 9], -0.5 * np.eye(9))
            assert np.array_equal(Fr[:, o + 9:o + 18], -0.5 * np.eye(9))
            assert np.array_equal(Fv[:, o + 18:o + 21], -0.5 * I3)
            assert np.array_equal(Fv[:, o + 21:o + 24], -0.5 * I3)
            assert np.array_equal(Fp[:, o + 24:o + 27], -0.5 * I3)
            assert np.array_equal(Fp[:, o + 27:o + 30], -0.5 * I3)
            for F in (Fr, Fv, Fp):
                assert np.all(F[:, bias_cols] == 0.0)
    zt = z_ref(truth_ref(), ref("true"))
    T = terms("true")
    base = CO.factored_cost(zt, T)
    assert base <= 1e-15
    rng = np.random.default_rng(8)
    for _ in range(5):
        zp = zt.copy()
        zp[63:72] += 1e-6 * rng.standard_normal(9)
        assert CO.factored_cost(zp, T) >= 1e-6


# ======================================================================================
# C09 - C17: constraint families
# ======================================================================================

def _states_for_families(which, n_random=200, seed=9):
    rng = np.random.default_rng(seed)
    yield "truth", truth_ref()
    for k in range(n_random):
        yield f"random{k}", random_state(rng)


def _family_check(names, tol_truth, tol_random, aniso=False):
    for which in ("true", "zero"):
        fam = fams(which, aniso)
        r = ref(which)
        worst = {"truth": 0.0, "random": 0.0}
        for label, st in _states_for_families(which):
            z = z_ref(st, r, aniso=aniso)
            m = max(fmax(z, fam, names).values())
            key = "truth" if label == "truth" else "random"
            worst[key] = max(worst[key], m)
        assert worst["truth"] <= tol_truth, (which, worst)
        assert worst["random"] <= tol_random, (which, worst)


@pytest.mark.claim("M3-C09")
def test_c09_primary_vanish_on_liftings():
    _family_check(["P0", "P1", "P2", "P3"], 1e-12, 1e-12)


def _orth_forms(k, rows):
    """Hand-built P1 (columns, R^T R = s^2 I) or RA1 (rows, R R^T = s^2 I) forms of rotation k."""
    def idx(r_, c_):
        return 15 * k + r_ + 3 * c_
    out = []
    for a in range(3):
        for b in range(a, 3):
            if rows:   # (R R^T)_ab = sum_c R_ac R_bc
                tt = [(1.0, idx(a, c), idx(b, c)) for c in range(3)]
            else:      # (R^T R)_ab = sum_c R_ca R_cb
                tt = [(1.0, idx(c, a), idx(c, b)) for c in range(3)]
            if a == b:
                tt.append((-1.0, S_IDX, S_IDX))
            out.append(_bilinear(N_BASE, tt))
    return out


@pytest.mark.claim("M3-C09")
@pytest.mark.claim("M3-C10")
def test_P1_is_column_and_RA1_is_row_orthonormality():
    """Pins each family's MEANING, not only the union (P1 <-> RA1 swapped would keep the union)."""
    fam = fams("true")
    for k in range(3):
        for name, rows in (("P1", False), ("RA1", True)):
            mine = [c.A.toarray() for c in fam[name][6 * k:6 * k + 6]]
            for A in _orth_forms(k, rows):
                assert min(np.abs(A - M).max() for M in mine) == 0.0, (name, k)
    # and the two differ off the variety (so the check above is not vacuous): at a non-orthogonal
    # R the column and row Gram matrices differ
    z = np.zeros(124)
    z[0:9], z[S_IDX] = np.arange(1.0, 10.0), 1.0
    assert max(abs(c.residual(z) - d.residual(z)) for c, d in zip(fam["P1"][:6], fam["RA1"][:6])) > 1


@pytest.mark.claim("M3-C10")
def test_c10_rai_table1_analogues_vanish():
    _family_check(["RA1", "RB1", "RB2", "RC1"], 1e-12, 1e-12)


@pytest.mark.claim("M3-C11")
def test_c11_galilean_redundant_vanish():
    _family_check(["RG", "RL", "RH", "RS"], 1e-12, 1e-12)


@pytest.mark.claim("M3-C12")
def test_c12_aniso_families_vanish():
    _family_check(["P4", "RR"], 1e-10, 1e-10, aniso=True)


@pytest.mark.claim("M3-C13")
@pytest.mark.parametrize("aniso", [False, True])
def test_c13_families_nontrivial_off_variety(aniso):
    rng = np.random.default_rng(13)
    for which in ("true", "zero"):
        fam = fams(which, aniso)
        z0 = z_ref(truth_ref(), ref(which), aniso=aniso)
        mins = {f: np.inf for f in fam}
        for _ in range(20):
            zp = z0 + 1e-3 * rng.standard_normal(len(z0))
            own = {f: max(abs(float(zp @ (c.A.toarray() @ zp)) - c.b) for c in cs) for f, cs in fam.items()}
            src_max = CN.family_max(zp, fam)          # the src helper must report |g|, not g
            src_res = CN.family_residuals(zp, fam)     # and family_residuals the SIGNED g
            for f, cs in fam.items():
                g_own = np.array([float(zp @ (c.A.toarray() @ zp)) - c.b for c in cs])
                tol_f = 1e-13 * (zp @ zp) * max(abs(c.A).max() for c in cs)
                assert np.abs(src_res[f] - g_own).max() <= tol_f, f
            for f, m in own.items():
                # round-off of z^T A z is ~ eps |z|^2 max|A| (RR cancels: |A| ~ 1e4 vs g ~ 1e-4)
                tol = 1e-13 * (zp @ zp) * max(abs(c.A).max() for c in fam[f])
                assert abs(src_max[f] - m) <= tol, (f, src_max[f], m)
                mins[f] = min(mins[f], m)
        assert all(v > 1e-7 for v in mins.values()), (which, mins)


EXPECTED_SIZES = {"P0": 1, "P1": 18, "P2": 27, "P3": 60, "RA1": 18, "RB1": 12, "RB2": 24, "RC1": 8,
                  "RG": 28, "RL": 60, "RH": 180, "RS": 36}


def _vech_stack(cs):
    return np.array([CN.vech(c.A) for c in cs])


@pytest.mark.claim("M3-C14")
def test_c14_family_sizes_and_ranks():
    fam = fams("true")
    assert {f: len(v) for f, v in fam.items()} == EXPECTED_SIZES
    # the primary / redundant labelling (C09 lists P0-P3 as primary; C10, C11 the redundant ones)
    assert tuple(CN.PRIMARY) == ("P0", "P1", "P2", "P3")
    assert set(CN.REDUNDANT) == {"RA1", "RB1", "RB2", "RC1", "RG", "RL", "RH", "RS"}
    assert tuple(CN.ANISO_PRIMARY) == ("P4",) and tuple(CN.ANISO_REDUNDANT) == ("RR",)
    for f, cs in fam.items():
        assert np.linalg.matrix_rank(_vech_stack(cs), tol=1e-9) == len(cs), f
    prim = [c for f in ("P0", "P1", "P2", "P3") for c in fam[f]]
    red = [c for f in ("RA1", "RB1", "RB2", "RC1", "RG", "RL", "RH", "RS") for c in fam[f]]
    assert len(prim) == 106 and len(red) == 366
    assert np.linalg.matrix_rank(_vech_stack(prim), tol=1e-9) == 106
    assert np.linalg.matrix_rank(_vech_stack(prim + red), tol=1e-9) == 453
    fa = fams("true", True)
    assert len(fa["P4"]) == 12 and len(fa["RR"]) == 6
    for f in ("P4", "RR"):
        assert np.linalg.matrix_rank(_vech_stack(fa[f]), tol=1e-9) == len(fa[f])
    prim_a = [c for f in ("P0", "P1", "P2", "P3", "P4") for c in fa[f]]
    assert len(prim_a) == 118
    assert np.linalg.matrix_rank(_vech_stack(CN.flat(fa)), tol=1e-9) == 471


def _bilinear(n, terms_):
    """Symmetric A with z^T A z = sum coeff * z_a z_b."""
    A = np.zeros((n, n))
    for c, a, b in terms_:
        A[a, b] += 0.5 * c
        A[b, a] += 0.5 * c
    return A


def _cross_forms(k, rows):
    """Handedness forms of rotation k, built by hand: entry (comp, c) for cyclic (a, b, c)."""
    def idx(r_, c_):
        return 15 * k + r_ + 3 * c_          # column-major vec
    out = {}
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for comp in range(3):
            j, l = (comp + 1) % 3, (comp + 2) % 3
            if rows:   # (R_(a) x R_(b))_comp - s R_{c, comp}
                t = [(1.0, idx(a, j), idx(b, l)), (-1.0, idx(a, l), idx(b, j)), (-1.0, S_IDX, idx(c, comp))]
            else:      # (R^(a) x R^(b))_comp - s R_{comp, c}
                t = [(1.0, idx(j, a), idx(l, b)), (-1.0, idx(l, a), idx(j, b)), (-1.0, S_IDX, idx(comp, c))]
            out[(comp, c)] = _bilinear(N_BASE, t)
    return out


@pytest.mark.claim("M3-C15")
def test_c15_row_handedness_equals_column_handedness():
    P2 = fams("true")["P2"]
    for k in range(3):
        col, row = _cross_forms(k, rows=False), _cross_forms(k, rows=True)
        for (comp, c), Acol in col.items():
            assert np.abs(Acol - row[(c, comp)]).max() == 0.0
        # and the shipped P2 of keyframe k is exactly this set
        mine = [c_.A.toarray() for c_ in P2[9 * k:9 * k + 9]]
        for Acol in col.values():
            assert min(np.abs(Acol - M).max() for M in mine) == 0.0


@pytest.mark.claim("M3-C16")
def test_c16_P1_RA1_rank_11_per_keyframe():
    fam = fams("true")
    for k in range(3):
        P1, RA1 = fam["P1"][6 * k:6 * k + 6], fam["RA1"][6 * k:6 * k + 6]
        assert np.linalg.matrix_rank(_vech_stack(P1 + RA1), tol=1e-9) == 11
        diag = [0, 3, 5]                        # (0,0), (1,1), (2,2) in the upper-triangle order
        S = sum(P1[d].A.toarray() for d in diag) - sum(RA1[d].A.toarray() for d in diag)
        assert np.abs(S).max() == 0.0


@pytest.mark.claim("M3-C17")
@pytest.mark.parametrize("aniso", [False, True])
def test_c17_homogeneous_symmetric(aniso):
    rng = np.random.default_rng(17)
    cs = CN.flat(fams("true", aniso))
    n = 136 if aniso else 124
    nonzero_b = [c for c in cs if c.b != 0]
    assert len(nonzero_b) == 1 and nonzero_b[0].b == 1.0 and nonzero_b[0].family == "P0"
    A0 = nonzero_b[0].A.toarray()
    assert A0[S_IDX, S_IDX] == 1.0 and np.count_nonzero(A0) == 1
    for c in cs:
        A = c.A.toarray()
        assert A.shape == (n, n)
        assert np.array_equal(A, A.T)
        # Homogeneity: the shipped residual is evaluated as z^T A z - b with nothing else
        # (checked against an own evaluation, so a stray linear/constant term would show).
        # The identity lam^2 z^T A z == (lam z)^T A (lam z) was deleted: it holds for EVERY A
        # and could not fail (verifier pass 1, Sec. 7).
        z = rng.standard_normal(n)
        for lam in (1.7, -0.3):
            own = float((lam * z) @ (A @ (lam * z))) - c.b
            assert abs(c.residual(lam * z) - own) <= 1e-12 * max(1.0, abs(own))


# ======================================================================================
# C18 - C23: spurious points
# ======================================================================================

def _stacked(cs):
    n = cs[0].A.shape[0]
    Big = sp.vstack([c.A for c in cs]).tocsr()
    b = np.array([c.b for c in cs])
    m = len(cs)

    def fun(z):
        Az = (Big @ z).reshape(m, n)
        return Az @ z - b

    def jac(z):
        return 2.0 * (Big @ z).reshape(m, n)
    return fun, jac


def _solve_feasible(cs, z0):
    fun, jac = _stacked(cs)
    sol = least_squares(fun, z0, jac=jac, method="trf", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=2000)
    return sol.x, np.abs(fun(sol.x)).max()


def _consistency(z, r):
    st = decode_ref(z)
    return np.abs(z - z[S_IDX] * z_ref(st, r)).max(), [np.linalg.det(R) for R in st.R]


@pytest.mark.claim("M3-C18")
@pytest.mark.slow
def test_c18_primary_feasible_set_is_lifted_states():
    fam, r = fams("zero"), ref("zero")
    prim = CN.flat(fam, ["P0", "P1", "P2", "P3"])
    rng = np.random.default_rng(7)
    converged = 0
    for _ in range(200):
        z, res = _solve_feasible(prim, rng.standard_normal(124))
        if res < 1e-10:
            converged += 1
            err, dets = _consistency(z, r)
            assert err <= 1e-9, err
            assert max(abs(d - 1) for d in dets) <= 1e-9, dets
    assert converged >= 150, converged

    def drop(name, starts=60):
        cs = CN.flat(fam, [f for f in ("P0", "P1", "P2", "P3") if f != name])
        out = []
        for _ in range(starts):
            z, res = _solve_feasible(cs, rng.standard_normal(124))
            if res < 1e-10:
                out.append(_consistency(z, r))
        return out

    noP2 = drop("P2")
    assert any(min(d) < -1 + 1e-6 for _, d in noP2), "no det = -1 solution without P2"
    noP3 = drop("P3")
    assert any(err > 1e-3 for err, _ in noP3), "no inconsistent lifting without P3"
    noP1 = drop("P1")
    assert any(min(abs(x) for x in d) < 1e-6 for _, d in noP1), "no det ~ 0 solution without P1"


@pytest.mark.claim("M3-C19")
def test_c19_P2_alone_implies_SO3_or_zero():
    P2 = fams("zero")["P2"][0:9]                      # rotation 0
    rng = np.random.default_rng(19)
    rows = list(range(9))
    n_so3 = n_zero = n_other = 0
    for _ in range(300):
        def fun(x):
            z = np.zeros(124)
            z[0:9], z[S_IDX] = x, 1.0
            return np.array([c.residual(z) for c in P2])

        def jac(x):
            z = np.zeros(124)
            z[0:9], z[S_IDX] = x, 1.0
            return np.array([2 * (c.A @ z)[rows] for c in P2])
        sol = least_squares(fun, 2 * rng.standard_normal(9), jac=jac, method="trf",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        if np.abs(fun(sol.x)).max() >= 1e-10:
            continue
        R = sol.x.reshape(3, 3, order="F")
        if np.abs(R).max() <= 1e-6:
            n_zero += 1
        elif np.abs(R.T @ R - I3).max() <= 1e-8 and abs(np.linalg.det(R) - 1) <= 1e-8:
            n_so3 += 1
        else:
            n_other += 1
    assert n_other == 0 and n_so3 > 0 and n_so3 + n_zero >= 200, (n_so3, n_zero, n_other)


@pytest.mark.claim("M3-C20")
def test_c20_improper_rotation_spurious_point():
    r = ref("zero")
    st = truth_ref().copy()
    st.R[0] = st.R[0] @ np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(st.R[0]) == pytest.approx(-1.0)
    m = fmax(z_ref(st, r), fams("zero"))
    for f in ("P0", "P1", "P3", "RA1", "RB1", "RB2", "RC1", "RG", "RL"):
        assert m[f] <= 1e-12, (f, m[f])
    for f in ("P2", "RH", "RS"):
        assert m[f] >= 1e-3, (f, m[f])


@pytest.mark.claim("M3-C21")
@pytest.mark.parametrize("which", ["true", "zero"])
def test_c21_zero_rotation_spurious_point(which):
    r = ref(which)
    st = truth_ref().copy()
    st.R[0] = np.zeros((3, 3))
    st.b[0] = np.zeros(6)
    m = fmax(z_ref(st, r), fams(which))
    for f in ("P0", "P2", "P3", "RB1", "RB2", "RC1", "RG", "RL", "RH", "RS"):
        assert m[f] <= 1e-12, (f, m[f])
    for f in ("P1", "RA1"):
        assert m[f] >= 0.5, (f, m[f])


@pytest.mark.claim("M3-C22")
@pytest.mark.parametrize("mode", ["rotated", "negated"])
def test_c22_gram_identities_do_not_pin_lifting(mode):
    r = ref("zero")
    st = truth_ref()
    z = z_ref(st, r)
    L = lift_ref(st, r["edges"][0])
    Qm = rexp([0.4, -0.3, 0.9]) if mode == "rotated" else -L["M"]
    if mode == "rotated":
        assert np.abs(Qm - L["M"]).max() > 0.1
    z[63:72] = (Qm @ skew(L["w"])).flatten(order="F")
    z[81:84] = Qm @ L["cv"]
    z[87:90] = Qm @ L["cp"]
    m = fmax(z, fams("zero"))
    for f in ("P0", "P1", "P2", "RA1", "RB1", "RB2", "RC1", "RG"):
        assert m[f] <= 1e-12, (f, m[f])
    for f in ("P3", "RL", "RH", "RS"):
        assert m[f] >= 1e-6, (f, m[f])


@pytest.mark.claim("M3-C23")
@pytest.mark.parametrize("aniso", [False, True])
def test_c23_global_sign_flip(aniso):
    """-z is the same state: bound to independent references.  (The bitwise identities
    g(-z) == g(z), (-z)^T Q (-z) == z^T Q z hold for ANY quadratic form and could not fail; they
    were replaced, verifier pass 1 Sec. 7.)"""
    layout = LF.ANISO if aniso else LF.BASE
    rng = np.random.default_rng(23)
    for which in ("true", "zero"):
        r = ref(which)
        for st in [truth_ref()] + [random_state(rng) for _ in range(20)]:
            zm = -z_ref(st, r, aniso=aniso)
            # Constraint residuals and cost at -z are bitwise those at z (IEEE negation is exact),
            # so asserting them only duplicates C09-C12 / C24 (verifier pass 2 Sec. 5): removed.
            # What C23 adds is that -z DECODES to the same state:
            d = LF.state_of_z(zm, layout)
            for k in range(3):
                for got, want in ((d.R[k], st.R[k]), (d.v[k], st.v[k]), (d.p[k], st.p[k]), (d.b[k], st.b[k])):
                    assert np.abs(got - want).max() <= 4.5e-16 * max(1.0, np.abs(want).max())
            dp = LF.state_of_z(zm, layout, project=True)        # and on the rounding path
            for k in range(3):
                assert np.abs(dp.R[k] - st.R[k]).max() <= 1e-14


# ======================================================================================
# C24 - C30: the cost
# ======================================================================================

@pytest.mark.claim("M3-C24")
@pytest.mark.parametrize("which", ["true", "zero"])
@pytest.mark.parametrize("variant,count", [("iso", 500), ("aniso", 300)])
def test_c24_cost_equals_direct(which, variant, count):
    rng = np.random.default_rng(24)
    r = ref(which)
    Q = CO.assemble_Q(terms(which, variant))
    worst = 0.0
    for _ in range(count):
        st = random_state(rng)
        z = z_ref(st, r, aniso=(variant == "aniso"))
        Jd = J_direct(st, r, imu=variant)
        worst = max(worst, abs(z @ Q @ z - Jd) / max(1.0, Jd))
    assert worst <= 1e-12, worst


@pytest.mark.claim("M3-C25")
@pytest.mark.parametrize("variant", ["iso", "aniso", "partial"])
def test_c25_Q_psd_and_bias_block(variant):
    for which in ("true", "zero"):
        Q = CO.assemble_Q(terms(which, variant))
        assert np.array_equal(Q, Q.T)
        ev = np.linalg.eigvalsh(Q)
        assert ev[0] / ev[-1] >= -1e-12, ev[0] / ev[-1]
        b = np.arange(45, 63)
        nb = np.setdiff1d(np.arange(Q.shape[0]), b)
        assert np.all(Q[np.ix_(b, nb)] == 0.0)
        rw = np.zeros((18, 18))
        for k in (0, 1):
            Dk = np.zeros((6, 18))
            Dk[:, 6 * k:6 * k + 6], Dk[:, 6 * k + 6:6 * k + 12] = -np.eye(6), np.eye(6)
            rw += Dk.T @ RW_PREC @ Dk
        assert np.array_equal(Q[np.ix_(b, b)], rw)
        # instance reconstruction check against the claim's domain note (max|Q| ~ 1.1e10)
        assert 1.0e10 <= np.abs(Q).max() <= 1.2e10


@pytest.mark.claim("M3-C26")
def test_c26_truth_cost_and_first_order_agreement_with_gtsam():
    r = ref("true")
    zt = z_ref(truth_ref(), r)
    assert CO.factored_cost(zt, terms("true")) <= 1e-15
    assert gtsam_cost(truth_ref(), r) <= 1e-15
    rng = np.random.default_rng(26)
    for eps in (1e-2, 1e-3, 1e-4):
        worst = 0.0
        for _ in range(50):
            st = perturbed(truth_ref(), eps, rng)
            worst = max(worst, abs(m3_cost(st, "true") / gtsam_cost(st, r) - 1))
        assert worst <= 0.05 * eps, (eps, worst)


def _gtsam_unwhitened(state, ed):
    f = gtsam.GalileanImuFactor(X(0), V(0), X(1), V(1), B(0), ed["pim"])
    vals = gtsam.Values()
    i, j = ed["i"], ed["j"]
    vals.insert(X(0), Pose3(Rot3(state.R[i]), state.p[i]))
    vals.insert(V(0), np.asarray(state.v[i], float))
    vals.insert(X(1), Pose3(Rot3(state.R[j]), state.p[j]))
    vals.insert(V(1), np.asarray(state.v[j], float))
    vals.insert(B(0), cb(state.b[i]))
    n = np.asarray(f.unwhitenedError(vals))                 # NavState order (rot, pos, vel)
    return n[0:3], n[6:9], n[3:6]                           # -> rot, vel, pos


@pytest.mark.claim("M3-C27")
def test_c27_truth_discrepancy_per_component():
    st = truth_ref()
    dev = {"rot": [], "vel": [], "pos": []}
    for lam in (1, 0.5, 0.25, 0.125):
        bbar = tuple((1 - lam) * LIT.BIAS_TRUE)
        I = INST.build_instance(np.array(bbar))            # code under test: the M3 residuals
        z = LF.z_of_state(st, I)
        m = {"rot": 0.0, "vel": 0.0, "pos": 0.0}
        g = {"rot": 0.0, "vel": 0.0, "pos": 0.0}
        for e, ed in enumerate(ref(bbar)["edges"]):          # independent: GTSAM's residuals
            rR, rv, rp = CO.edge_residual_forms(I, LF.BASE, e)
            m["rot"] += 0.5 * np.sum(np.einsum("abn,n->ab", rR, z) ** 2)
            m["vel"] += np.sum((rv @ z) ** 2)
            m["pos"] += np.sum((rp @ z) ** 2)
            nr, nv, npos = _gtsam_unwhitened(st, ed)
            g["rot"] += nr @ nr
            g["vel"] += nv @ nv
            g["pos"] += npos @ npos
        for k in dev:
            dev[k].append(m[k] / g[k] - 1)
    for k, d in dev.items():
        assert max(abs(x) for x in d) <= 3e-2, (k, d)
    shrink = {k: [abs(d[q + 1]) / abs(d[q]) for q in range(3)] for k, d in dev.items()}
    assert all(0.2 <= f <= 0.3 for f in shrink["rot"]), (dev, shrink)
    for k in ("vel", "pos"):
        assert all(0.45 <= f <= 0.55 for f in shrink[k]), (k, dev, shrink)


@pytest.mark.claim("M3-C28")
def test_c28_cayley_vs_exp_is_cubic():
    """MATH CHECK of the surrogate (test-side Cay / GTSAM Exp).  Its only src contact is the J
    equality below, already covered by the instance test; the shipped residual's use of Cay is
    bound by test_c05_shipped / test_c06_shipped."""
    ed = ref("zero")["edges"][0]
    dev = []
    for s in (1, 0.5, 0.25, 0.125):
        x = ed["J"] @ (s * LIT.BIAS_TRUE)
        dev.append(np.abs(ed["Ubar"] @ (cay5(x) - gexp(x))).max())
    factors = [dev[k] / dev[k + 1] for k in range(3)]
    assert all(7.5 <= f <= 8.5 for f in factors), (dev, factors)
    # the instance's surrogate is this Cayley map: the edge-0 rotation residual at truth equals
    # the 5x5 one built with cay5
    assert np.abs(inst("zero").edges[0].J - ed["J"]).max() <= 1e-12


@pytest.mark.claim("M3-C29")
def test_c29_half_frobenius_weight():
    r = ref("true")
    rng = np.random.default_rng(29)
    half, full = 0.0, 0.0
    T = terms("true")
    for _ in range(30):
        st = perturbed(truth_ref(), 1e-4, rng)
        Jg = gtsam_cost(st, r)
        z = LF.z_of_state(st, inst("true"))
        Jm = CO.factored_cost(z, T)
        extra = sum(CO.factored_cost(z, [t]) for t in T if t.name.endswith(".rot"))
        half = max(half, abs(Jm / Jg - 1))
        full = max(full, abs((Jm + extra) / Jg - 1))     # tau_R ||r_R||_F^2, Rai Eq. (20)
    assert half <= 5e-6, half
    assert full >= 0.1, full


def _local_param(state, d):
    out = state.copy()
    for k in range(3):
        q = d[15 * k:15 * k + 15]
        out.R[k] = state.R[k] @ rexp(q[0:3])
        out.v[k] = state.v[k] + q[3:6]
        out.p[k] = state.p[k] + q[6:9]
        out.b[k] = state.b[k] + q[9:15]
    return out


@pytest.mark.claim("M3-C30")
@pytest.mark.parametrize("which", ["true", "zero"])
def test_c30_whitened_jacobian_full_rank(which):
    I, T = inst(which), terms(which)
    st = truth_ref()
    h = 1e-6
    cols = []
    for k in range(45):
        d = np.zeros(45)
        d[k] = h
        fp = CO.whitened_residual(LF.z_of_state(_local_param(st, d), I), T)
        fm = CO.whitened_residual(LF.z_of_state(_local_param(st, -d), I), T)
        cols.append((fp - fm) / (2 * h))
    Jw = np.array(cols).T
    sv = np.linalg.svd(Jw, compute_uv=False)
    # NB the rank is invariant to any positive per-term rescaling, so it cannot see a wrong
    # whitening weight; test_whitened_residual_is_symmetric_root_whitening pins the weights.
    assert int((sv > 1e-12 * sv[0]).sum()) == 45


def _sym_root(W):
    w, U = np.linalg.eigh(W)
    return (U * np.sqrt(np.clip(w, 0.0, None))) @ U.T


@pytest.mark.claim("M3-C30")
def test_whitened_residual_is_symmetric_root_whitening():
    """||sqrt(W) F z||^2 must equal the cost, with the SYMMETRIC root, also for a singular PSD
    weight (eigh then returns eigenvalues ~ -1e-10, whose sqrt would be NaN without a clip)."""
    rng = np.random.default_rng(300)
    Qr = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    W_sing = Qr @ np.diag([1e6, 2e6, 5e6, 1e6, 0.0, 0.0]) @ Qr.T
    assert np.linalg.eigvalsh(W_sing).min() < 0           # round-off makes it "slightly negative"
    cases = [(terms("true"), LF.BASE), (terms("zero", "aniso"), LF.ANISO),
             (CO.terms(inst("true"), LF.ANISO, "aniso", W_vp=W_sing), LF.ANISO)]
    for T, layout in cases:
        for _ in range(10):
            st = random_state(rng)
            z = LF.z_of_state(st, inst("true"), layout)
            wr = CO.whitened_residual(z, T)
            assert np.all(np.isfinite(wr))
            own = np.concatenate([_sym_root(t.W) @ (t.F @ z) for t in T])
            assert np.abs(wr - own).max() <= 1e-9 * max(1.0, np.abs(own).max())
            cost = CO.factored_cost(z, T)
            assert abs(wr @ wr - cost) <= 1e-9 * max(1.0, cost)


# ======================================================================================
# C31 - C35: isotropy, anisotropy, AutoTight
# ======================================================================================

@pytest.mark.claim("M3-C31")
def test_c31_isotropy_obstruction():
    r = ref("zero")
    ed = r["edges"][0]
    d = np.array([0.3, -1.0, 0.5])
    rng = np.random.default_rng(31)

    def coeff(st, W):
        vals = []
        for lam in (0, 1, 2):
            s2 = st.copy()
            s2.v[1] = st.v[1] + lam * d
            rho = (s2.R[0] @ ed["dR"]).T @ resid5(s2, ed, r["bbar"])[0:3, 3]
            vals.append(rho @ W @ rho)
        return 0.5 * (vals[2] - 2 * vals[1] + vals[0])

    def coeff_cost(st, f):
        vals = []
        for lam in (0, 1, 2):
            s2 = st.copy()
            s2.v[1] = st.v[1] + lam * d
            vals.append(f(s2))
        return 0.5 * (vals[2] - 2 * vals[1] + vals[0])

    def m3(s2):
        return CO.factored_cost(LF.z_of_state(s2, inst("zero")), terms("zero"))

    def ref_cost(s2):
        return J_direct(s2, r)

    iso, aniso, qz = [], [], []
    for _ in range(200):
        st = truth_ref().copy()
        st.R[0] = rexp(2 * rng.standard_normal(3))
        iso.append(coeff(st, 3 * I3))
        aniso.append(coeff(st, np.diag([1.0, 2.0, 5.0])))
        qz.append((coeff_cost(st, m3), coeff_cost(st, ref_cost)))
    assert np.ptp(iso) <= 1e-9 and abs(iso[0] - 3 * d @ d) <= 1e-9
    assert np.ptp(aniso) > 1.0, np.ptp(aniso)
    # the shipped base cost's lam^2 coefficient equals the independent 5x5 cost's, and is
    # R_0-independent (the claim's proof step); the former "Q vs Q[24:27,24:27]" check compared
    # Q with itself and was replaced (verifier pass 1, Sec. 7)
    qm = np.array([a for a, _ in qz])
    qr = np.array([b for _, b in qz])
    assert np.abs(qm - qr).max() <= 1e-6 * np.abs(qr).max(), np.abs(qm - qr).max()
    # (ptp(qm) ~ 0 and ptp(qr) ~ 0 hold for ANY Q / by construction, since v_1 enters with
    # constant coefficients; both cannot fail and were removed -- verifier passes 2 and 3, Sec. 5.)
    # The shipped lam^2 coefficient also equals its closed form from the REFERENCE weights:
    # v_1 enters r_v of edge 0 as +I, r_v of edge 1 as -I and r_p of edge 1 as -Dt_1 I.
    e0, e1 = r["edges"]
    closed = (e0["tau_v"] + e1["tau_v"] + e1["tau_p"] * e1["Dt"]**2) * (d @ d)
    assert np.abs(qm - closed).max() <= 1e-6 * closed, (qm[:3], closed)


SUB = np.r_[np.arange(0, 30), np.arange(45, 51), np.arange(63, 93), [S_IDX]]   # 67 entries


@functools.lru_cache(maxsize=None)
def _samples(seed, shipped=False):
    """6000 i.i.d. lifted states at bbar = 0 from the claims' sampler (numpy default_rng(seed)):
    the degree-2 monomial matrix over the edge-0 sub-vector, and rho_v, rho_p of edge 0."""
    assert len(SUB) == 67
    r = ref("zero")
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(67)
    ed = r["edges"][0]
    rows, rhov, rhop = [], [], []
    for _ in range(6000):
        st = claim_sampler_state(rng)
        zs = (LF.z_of_state(st, inst("zero")) if shipped else z_ref(st, r))[SUB]
        rows.append(np.outer(zs, zs)[iu])
        rr = resid5(st, ed, r["bbar"])
        M = st.R[0] @ ed["dR"]
        rhov.append(M.T @ rr[0:3, 3])
        rhop.append(M.T @ rr[0:3, 4])
    return np.array(rows), np.array(rhov), np.array(rhop), iu


@pytest.mark.claim("M3-C32")
@pytest.mark.slow
def test_c32_monomial_fit():
    # the monomials come from the SHIPPED encoder (binds C32 to src); f from the 5x5 residual
    Phi, rv, rp, iu = _samples(2, shipped=True)
    assert Phi.shape == (6000, 2278)

    def relres(f):
        c, *_ = np.linalg.lstsq(Phi, f, rcond=None)
        return np.linalg.norm(Phi @ c - f) / np.linalg.norm(f)
    W = np.random.default_rng(320).standard_normal((3, 3))
    W = W @ W.T + 0.5 * I3
    assert relres(3 * np.einsum("ka,ka->k", rv, rv)) <= 1e-12
    assert relres(np.einsum("ka,ka->k", rv, rp)) <= 1e-12
    assert relres(np.einsum("ka,ab,kb->k", rv, np.diag([1.0, 2, 5]), rv)) >= 1e-4
    assert relres(np.einsum("ka,ab,kb->k", rv, W, rv)) >= 1e-4


@pytest.mark.claim("M3-C33")
def test_c33_partial_anisotropy():
    r = ref("true")
    for ed in r["edges"]:
        D = 0.5 * np.trace(ed["W_R"]) * I3 - ed["W_R"]
        assert np.linalg.eigvalsh(D).min() >= 0
        assert np.linalg.eigvalsh(ed["W_R"]).max() <= 0.5 * np.trace(ed["W_R"])
        G2 = ed["G2"]
        # instance reconstruction check against the claim's measured note (corr -0.867)
        assert abs(G2[0, 1] / np.sqrt(G2[0, 0] * G2[1, 1]) + 0.867) <= 2e-3
    # the M3 partial cost is quadratic in base z and equals the direct tr(rt G rt^T) cost
    Qp = CO.assemble_Q(terms("true", "partial"))
    rng = np.random.default_rng(33)
    for _ in range(50):
        st = random_state(rng)
        Jd = J_direct(st, r, imu="partial")
        z = z_ref(st, r)
        assert abs(z @ Qp @ z - Jd) <= 1e-12 * max(1.0, Jd)
    for eps in (1e-2, 1e-3, 1e-4):
        worst = 0.0
        for _ in range(30):
            st = perturbed(truth_ref(), eps, rng)
            worst = max(worst, abs(m3_cost(st, "true", "partial") / gtsam_cost(st, r, "partial") - 1))
        assert worst <= 0.05 * eps, (eps, worst)


W_SYN = 1e6 * np.diag([1.0, 2.0, 5.0, 1.0, 2.0, 5.0])     #: M3-C34's synthetic (vel, pos) weight


@pytest.mark.claim("M3-C34")
def test_c34_frame_of_anisotropic_lifting():
    r = ref("true")
    T = CO.terms(inst("true"), LF.ANISO, "aniso", W_vp=W_SYN)
    rng = np.random.default_rng(34)
    for eps in (1e-2, 1e-3, 1e-4):
        body = world = 0.0
        for _ in range(30):
            st = perturbed(truth_ref(), eps, rng)
            Jg = gtsam_cost(st, r, "aniso", W_vp=W_SYN)
            Jm = CO.factored_cost(LF.z_of_state(st, inst("true"), LF.ANISO), T)
            body = max(body, abs(Jm / Jg - 1))
            world = max(world, abs(J_direct(st, r, imu="world", W_vp=W_SYN) / Jg - 1))
        assert body <= 1.0 * eps, (eps, body)
        assert world >= 1e-2, (eps, world)


def _mono_coeffs(A):
    """Coefficient vector of z_S^T A_SS z_S over the 2278 monomials z_a z_b, a <= b."""
    A = A.toarray()[np.ix_(SUB, SUB)]
    C = 2 * A
    C[np.diag_indices(67)] = np.diag(A)
    return C[np.triu_indices(67)]


def _restricted(cs):
    keep = np.zeros(124, bool)
    keep[SUB] = True
    out = []
    for c in cs:
        rows, cols = c.A.nonzero()
        if c.b == 0 and keep[rows].all() and keep[cols].all():
            out.append(_mono_coeffs(c.A))
    return np.array(out)


@functools.lru_cache(maxsize=None)
def _nullspace():
    """Cut at the largest relative singular-value gap (claim M3-C35)."""
    Phi = _samples(4)[0]
    U, sv, Vt = np.linalg.svd(Phi, full_matrices=True)
    pos = sv[sv > 0]
    k = int(np.argmax(pos[:-1] / pos[1:]))           # gap between sv[k] and sv[k+1]
    rank = k + 1
    return Vt[rank:].T, sv, rank


@pytest.mark.claim("M3-C35")
@pytest.mark.slow
def test_c35_autotight_cross_check():
    N, sv, rank = _nullspace()
    gap_hi, gap_lo = sv[rank - 1] / sv[0], sv[rank] / sv[0]
    assert rank == 2015 and N.shape == (2278, 263), (rank, gap_hi, gap_lo)
    Phi = _samples(4)[0]
    fam = fams("zero")
    stages = [("P1", "P2", "P3"), ("RA1", "RB1", "RB2", "RC1"), ("RG",), ("RL",), ("RH",), ("RS",)]
    expected = [60, 92, 106, 136, 226, 236]
    acc = []
    for names, exp in zip(stages, expected):
        Cf = _restricted(CN.flat(fam, list(names)))
        acc.append(Cf)
        S = np.vstack(acc)
        s = np.linalg.svd(S, compute_uv=False)
        assert int((s > 1e-10 * s[0]).sum()) == exp, (names, int((s > 1e-10 * s[0]).sum()))
        # projection onto the computed null space (the claim's "inside it to 1e-10")
        proj = Cf - (Cf @ N) @ N.T
        assert (np.linalg.norm(proj, axis=1) / np.linalg.norm(Cf, axis=1)).max() <= 1e-10, names
        # and basis-free: annihilated by the samples
        ann = np.linalg.norm(Phi @ Cf.T, axis=0) / (sv[0] * np.linalg.norm(Cf, axis=1))
        assert ann.max() <= 1e-10, (names, ann.max())


# ======================================================================================
# C39: instance fingerprint
# ======================================================================================

FINGERPRINT = {0: (6.9204e7, 4.8049e5, 5.8906e6), 1: (6.9204e7, 4.8167e5, 5.8972e6)}


def _sig4(a, b):
    return float(f"{a:.4e}") == float(f"{b:.4e}")


@pytest.mark.claim("M3-C39")
def test_c39_instance_fingerprint():
    r = ref("zero")                      # independent PIMs with the stated params, bbar = 0
    I = inst("zero")
    for e in range(2):
        ed, ei = r["edges"][e], I.edges[e]
        assert ed["Dt"] == 0.5000000000000003 and ei.Dt == 0.5000000000000003
        for got_ref, got_src, want in zip((ed["tau_R"], ed["tau_v"], ed["tau_p"]),
                                          (ei.tau_R, ei.tau_v, ei.tau_p), FINGERPRINT[e]):
            assert _sig4(got_ref, want), (e, got_ref, want)
            assert _sig4(got_src, want), (e, got_src, want)
    d = np.diag(I.rw_precision)
    assert np.all(I.rw_precision == np.diag(d))
    assert all(_sig4(x, 5.5402e9) for x in d[0:3]) and all(_sig4(x, 2.2222e5) for x in d[3:6]), d
    dr = np.diag(RW_PREC)
    assert all(_sig4(x, 5.5402e9) for x in dr[0:3]) and all(_sig4(x, 2.2222e5) for x in dr[3:6])


# ======================================================================================
# The form helpers on their own outputs (verifier pass 2: symmetrisation downstream hides
# defects here, so each helper is checked directly against numbers computed in the test)
# ======================================================================================

@pytest.mark.claim("M3-C17")
def test_form_helpers_are_symmetric_and_evaluate_correctly():
    from src.qcqp import forms as FM
    rng = np.random.default_rng(170)
    n = 7
    z = rng.standard_normal(n)
    a, b = rng.standard_normal((3, n)), rng.standard_normal((3, n))
    A, Bm = rng.standard_normal((3, 3, n)), rng.standard_normal((3, 3, n))
    C = rng.standard_normal((3, 3))
    ev = lambda Q: np.einsum("...nm,n,m->...", Q, z, z)       # noqa: E731
    lv = lambda L: L @ z                                        # noqa: E731

    def symmetric(Q):
        return np.array_equal(Q, np.swapaxes(Q, -1, -2))
    M = rng.standard_normal((n, n))
    S = FM.sym(M)
    assert symmetric(S) and np.abs(S - 0.5 * (M + M.T)).max() <= 1e-15
    cases = {
        "outer": (FM.outer(a, b), lv(a) * lv(b)),
        "matmul": (FM.matmul(A, Bm), lv(A) @ lv(Bm)),
        "matvec": (FM.matvec(A, a), lv(A) @ lv(a)),
        "dot": (FM.dot(a, b), lv(a) @ lv(b)),
        "cross": (FM.cross(a, b), np.cross(lv(a), lv(b))),
    }
    for name, (Q, want) in cases.items():
        assert symmetric(Q), name
        assert np.abs(ev(Q) - want).max() <= 1e-12 * max(1.0, np.abs(want).max()), name
    # the entry-wise outer coefficients themselves (not only their value): 0.5 (f g^T + g f^T)
    assert np.abs(FM.outer(a, b)[1] - 0.5 * (np.outer(a[1], b[1]) + np.outer(b[1], a[1]))).max() <= 1e-15
    # linear helpers
    assert np.abs(lv(FM.hat(a)) - skew(lv(a))).max() <= 1e-14
    assert np.abs(ev(FM.hat(FM.matvec(A, a))) - skew(lv(A) @ lv(a))).max() <= 1e-12
    assert np.abs(lv(FM.cmat_right(A, C)) - lv(A) @ C).max() <= 1e-13
    assert np.abs(lv(FM.cvec(C, a)) - C @ lv(a)).max() <= 1e-13
    assert np.abs(lv(FM.lin_T(A)) - lv(A).T).max() == 0.0
