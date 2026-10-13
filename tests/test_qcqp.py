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
from src.qcqp import fixture as FX
import src.qcqp.instance as INST
from src.qcqp import lifting as LF

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

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


def make_pim(gyro, acc, dt, bbar):
    p = gtsam.PreintegrationParams(G_W)
    p.setGyroscopeCovariance(FX.SIGMA_G**2 * I3)
    p.setAccelerometerCovariance(FX.SIGMA_A**2 * I3)
    p.setIntegrationCovariance(FX.SIGMA_INT**2 * I3)
    pim = gtsam.PreintegratedImuMeasurementsG(p, cb(bbar))
    for w, a in zip(gyro, acc):
        pim.integrateMeasurement(np.asarray(a, float), np.asarray(w, float), dt)
    return pim


def fixture_pim(e, bbar):
    n = 200
    gyro = np.tile(FX.OMEGA_BODY[e] + FX.B_G, (n, 1))
    acc = np.tile(FX.F_BODY[e] + FX.B_A, (n, 1))
    return make_pim(gyro, acc, 2.5e-3, bbar)


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


@functools.lru_cache(maxsize=None)
def ref(bbar_key):
    """Independent per-edge data at nominal bias.  ``bbar_key`` is "true", "zero" or a tuple."""
    bbar = {"true": FX.BIAS_TRUE, "zero": np.zeros(6)}.get(bbar_key) if isinstance(bbar_key, str) \
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
            J=fd_J(pim, bbar, 1e-3), pim=pim, C=C,
            tau_R=3 / np.trace(C[rot, rot]), tau_v=3 / np.trace(C[vel, vel]),
            tau_p=3 / np.trace(C[pos, pos]),
            W_R=np.linalg.inv(C[rot, rot]), G2=np.linalg.inv(S2), W_vp=np.linalg.inv(Svp)))
    return dict(bbar=bbar, edges=edges)


@functools.lru_cache(maxsize=None)
def truth_ref():
    """Ground truth re-derived by chaining GTSAM's prediction at the true bias."""
    ns = gtsam.NavState(Rot3.Expmap(FX.ROT_VEC_0), FX.P_0, FX.V_0)
    out = [ns]
    for e in range(2):
        out.append(fixture_pim(e, FX.BIAS_TRUE).predict(out[-1], cb(FX.BIAS_TRUE)))
    R = [s.attitude().matrix() for s in out]
    return LF.State(R, [np.asarray(s.velocity()) for s in out],
                    [np.asarray(s.position()) for s in out], [FX.BIAS_TRUE.copy() for _ in out])


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


RW_PREC = np.diag([1 / (FX.SIGMA_BG_RW**2 * 0.5)] * 3 + [1 / (FX.SIGMA_BA_RW**2 * 0.5)] * 3)


def J_direct(state, r, imu="iso", rot_weight=0.5):
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
            J += rho @ ed["W_vp"] @ rho
        elif imu == "world":
            q = np.r_[rv, rp]
            J += q @ ed["W_vp"] @ q
    sp_ = FX.SIGMA_PRIOR
    J += 0.5 / sp_**2 * np.sum((state.R[0] - tr.R[0])**2)
    J += np.sum((state.v[0] - tr.v[0])**2) / sp_**2 + np.sum((state.p[0] - tr.p[0])**2) / sp_**2
    for k in (1, 2):
        J += np.sum((state.p[k] - tr.p[k])**2) / FX.SIGMA_GNSS**2
    for k in (0, 1):
        d = state.b[k + 1] - state.b[k]
        J += d @ RW_PREC @ d
    return float(J)


def imu_info(ed, kind="iso"):
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
        W = ed["W_vp"]                                  # (vel, pos) order
        Info[vel, vel], Info[vel, pos] = W[0:3, 0:3], W[0:3, 3:6]
        Info[pos, vel], Info[pos, pos] = W[3:6, 0:3], W[3:6, 3:6]
        return Info
    raise ValueError(kind)


def gtsam_cost(state, r, kind="iso"):
    """2 * graph.error of the GTSAM reference graph (M3-C26)."""
    tr = truth_ref()
    graph = gtsam.NonlinearFactorGraph()
    for ed in r["edges"]:
        f = gtsam.GalileanImuFactor(X(ed["i"]), V(ed["i"]), X(ed["j"]), V(ed["j"]), B(ed["i"]),
                                    ed["pim"])
        graph.add(f.cloneWithNewNoiseModel(noiseModel.Gaussian.Information(imu_info(ed, kind))))
    graph.add(gtsam.PriorFactorPose3(X(0), Pose3(Rot3(tr.R[0]), tr.p[0]),
                                     noiseModel.Isotropic.Sigma(6, FX.SIGMA_PRIOR)))
    graph.add(gtsam.PriorFactorVector(V(0), tr.v[0], noiseModel.Isotropic.Sigma(3, FX.SIGMA_PRIOR)))
    for k in (1, 2):
        graph.add(gtsam.GPSFactor(X(k), tr.p[k], noiseModel.Isotropic.Sigma(3, FX.SIGMA_GNSS)))
    rw_sig = np.r_[[FX.SIGMA_BA_RW * np.sqrt(0.5)] * 3, [FX.SIGMA_BG_RW * np.sqrt(0.5)] * 3]
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
    assert np.array_equal(I.b_true, FX.BIAS_TRUE)
    assert np.allclose(I.rw_precision, RW_PREC, rtol=1e-15, atol=0)


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
        # decoder round trip
        d = LF.state_of_z(-2.5 * z)
        for k in range(3):
            assert np.abs(d.R[k] - st.R[k]).max() <= 1e-15
            assert np.abs(d.b[k] - st.b[k]).max() <= 1e-15


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
    cases = [(fixture_pim(e, bb), bb) for e in range(2) for bb in (FX.BIAS_TRUE, np.zeros(6))]
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
    # the instance's J is this J (project column order)
    for which in ("true", "zero"):
        for e, ed in enumerate(inst(which).edges):
            assert np.abs(ed.J - fd_J(fixture_pim(e, ref(which)["bbar"]), ref(which)["bbar"],
                                      1e-6)).max() <= 1e-8


def _random_Xi(rng):
    return rexp(2 * rng.standard_normal(3)), rng.standard_normal(3), rng.standard_normal(3)


@pytest.mark.claim("M3-C05")
def test_c05_cleared_residual():
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


@pytest.mark.claim("M3-C06")
def test_c06_residual_factorisation():
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
            for f, m in fmax(zp, fam).items():
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
    lam = 1.7
    nonzero_b = [c for c in cs if c.b != 0]
    assert len(nonzero_b) == 1 and nonzero_b[0].b == 1.0 and nonzero_b[0].family == "P0"
    A0 = nonzero_b[0].A.toarray()
    assert A0[S_IDX, S_IDX] == 1.0 and np.count_nonzero(A0) == 1
    for c in cs:
        A = c.A.toarray()
        assert A.shape == (n, n)
        assert np.array_equal(A, A.T)
        z = rng.standard_normal(n)
        g1, g2 = z @ A @ z, (lam * z) @ A @ (lam * z)
        assert abs(lam**2 * g1 - g2) <= 1e-12 * (z @ z) * max(np.abs(A).max(), 1e-300)


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
    rng = np.random.default_rng(18)
    prim = CN.flat(fam, ["P0", "P1", "P2", "P3"])
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
    for which in ("true", "zero"):
        z = z_ref(truth_ref(), ref(which), aniso=aniso)
        for c in CN.flat(fams(which, aniso)):
            assert c.residual(-z) == c.residual(z)
        variant = "aniso" if aniso else "iso"
        Q = CO.assemble_Q(terms(which, variant))
        assert (-z) @ Q @ (-z) - z @ Q @ z == 0.0
        a = LF.state_of_z(z, LF.ANISO if aniso else LF.BASE)
        b = LF.state_of_z(-z, LF.ANISO if aniso else LF.BASE)
        for k in range(3):
            for x, y in ((a.R, b.R), (a.v, b.v), (a.p, b.p), (a.b, b.b)):
                assert np.array_equal(x[k], y[k])


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


@pytest.mark.claim("M3-C27")
def test_c27_bbar_offset_difference_halves():
    rel = []
    for lam in (1, 0.5, 0.25, 0.125):
        bbar = tuple((1 - lam) * FX.BIAS_TRUE)
        I = INST.build_instance(np.array(bbar))
        z = LF.z_of_state(truth_ref(), I)
        Jm = CO.factored_cost(z, CO.terms(I, LF.BASE))
        Jg = gtsam_cost(truth_ref(), ref(bbar))
        assert Jg > 0
        rel.append(abs(Jm - Jg) / Jg)
    assert rel[0] <= 1e-2, rel
    ratios = [rel[k + 1] / rel[k] for k in range(3)]
    assert all(0.45 <= q <= 0.55 for q in ratios), (rel, ratios)


@pytest.mark.claim("M3-C28")
def test_c28_cayley_vs_exp_is_cubic():
    ed = ref("zero")["edges"][0]
    dev = []
    for s in (1, 0.5, 0.25, 0.125):
        x = ed["J"] @ (s * FX.BIAS_TRUE)
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
    assert int((sv > 1e-12 * sv[0]).sum()) == 45


# ======================================================================================
# C31 - C35: isotropy, anisotropy, AutoTight
# ======================================================================================

@pytest.mark.claim("M3-C31")
def test_c31_isotropy_obstruction():
    r = ref("zero")
    ed = r["edges"][0]
    d = np.array([0.3, -1.0, 0.5])
    rng = np.random.default_rng(31)
    Q = CO.assemble_Q(terms("zero"))

    def coeff(st, W):
        vals = []
        for lam in (0, 1, 2):
            s2 = st.copy()
            s2.v[1] = st.v[1] + lam * d
            rho = (s2.R[0] @ ed["dR"]).T @ resid5(s2, ed, r["bbar"])[0:3, 3]
            vals.append(rho @ W @ rho)
        return 0.5 * (vals[2] - 2 * vals[1] + vals[0])

    def coeff_Q(st):
        vals = []
        for lam in (0, 1, 2):
            s2 = st.copy()
            s2.v[1] = st.v[1] + lam * d
            z = z_ref(s2, r)
            vals.append(z @ Q @ z)
        return 0.5 * (vals[2] - 2 * vals[1] + vals[0])

    iso, aniso, qz = [], [], []
    for _ in range(200):
        st = truth_ref().copy()
        st.R[0] = rexp(2 * rng.standard_normal(3))
        iso.append(coeff(st, 3 * I3))
        aniso.append(coeff(st, np.diag([1.0, 2.0, 5.0])))
        qz.append(coeff_Q(st))
    assert np.ptp(iso) <= 1e-9 and abs(iso[0] - 3 * d @ d) <= 1e-9
    assert np.ptp(aniso) > 1.0, np.ptp(aniso)
    # the base cost's lam^2 coefficient is R_0-independent (it is d^T Q_vv d)
    assert np.ptp(qz) <= 1e-6 * abs(np.mean(qz)), np.ptp(qz)
    assert abs(np.mean(qz) - d @ Q[24:27, 24:27] @ d) <= 1e-6 * abs(np.mean(qz))


SUB = np.r_[np.arange(0, 30), np.arange(45, 51), np.arange(63, 93), [S_IDX]]   # 67 entries


@functools.lru_cache(maxsize=None)
def _samples():
    """6000 i.i.d. lifted states at bbar = 0: monomial matrix over the edge-0 sub-vector."""
    assert len(SUB) == 67
    r = ref("zero")
    rng = np.random.default_rng(32)
    iu = np.triu_indices(67)
    rows, rhov, rhop = [], [], []
    for _ in range(6000):
        st = random_state(rng)
        zs = z_ref(st, r)[SUB]
        rows.append(np.outer(zs, zs)[iu])
        ed = r["edges"][0]
        rr = resid5(st, ed, r["bbar"])
        M = st.R[0] @ ed["dR"]
        rhov.append(M.T @ rr[0:3, 3])
        rhop.append(M.T @ rr[0:3, 4])
    return np.array(rows), np.array(rhov), np.array(rhop), iu


@pytest.mark.claim("M3-C32")
@pytest.mark.slow
def test_c32_monomial_fit():
    Phi, rv, rp, iu = _samples()
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


@pytest.mark.claim("M3-C34")
def test_c34_full_vel_pos_anisotropy_via_rho():
    r = ref("true")
    rng = np.random.default_rng(34)
    for eps in (1e-2, 1e-3, 1e-4):
        worst = 0.0
        for _ in range(30):
            st = perturbed(truth_ref(), eps, rng)
            worst = max(worst, abs(m3_cost(st, "true", "aniso") / gtsam_cost(st, r, "aniso") - 1))
        assert worst <= 0.05 * eps, (eps, worst)
    worst_world = 0.0
    for _ in range(30):
        st = perturbed(truth_ref(), 1e-4, rng)
        worst_world = max(worst_world, abs(J_direct(st, r, imu="world") / gtsam_cost(st, r, "aniso") - 1))
    assert worst_world >= 1e-3, worst_world


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
    Phi = _samples()[0]
    U, sv, Vt = np.linalg.svd(Phi, full_matrices=True)
    rank = int((sv > 1e-10 * sv[0]).sum())
    return Vt[rank:].T, sv, rank


@pytest.mark.claim("M3-C35")
@pytest.mark.slow
def test_c35_autotight_cross_check():
    N, sv, rank = _nullspace()
    assert rank == 2015 and N.shape == (2278, 263)
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
        # (a) basis-free: every hand form is annihilated by the samples to 1e-10 relative
        Phi = _samples()[0]
        ann = np.linalg.norm(Phi @ Cf.T, axis=0) / (sv[0] * np.linalg.norm(Cf, axis=1))
        assert ann.max() <= 1e-10, (names, ann.max())
        # (b) projection onto the COMPUTED null-space basis.  That basis is only as accurate as
        # the singular-value gap allows (Wedin: sin(angle) <~ eps * sigma_0 / sigma_{rank-1});
        # with these samples sigma_{rank-1}/sigma_0 = 1.4e-8, so the bound is ~1.6e-8 and
        # 1e-10 cannot be resolved (measured up to 4.8e-10).  Tolerance = max(1e-10, bound).
        tol = max(1e-10, np.finfo(float).eps * sv[0] / sv[rank - 1])
        proj = Cf - (Cf @ N) @ N.T
        assert (np.linalg.norm(proj, axis=1) / np.linalg.norm(Cf, axis=1)).max() <= tol, names


# ======================================================================================
# C36 - C38: OPEN-HYPOTHESIS -- the Shor relaxation (the measurement the claims describe)
# ======================================================================================
#
# These claims say "settle in M4".  The measurement is implemented here exactly as the
# claims describe it (Clarabel, numerical rank with the 1e-6 relative floor, sigma_1/sigma_2,
# Rai's eta) and the tests assert only what the claims assert.  A failure is a finding.

from src.qcqp import sdp as SDP  # noqa: E402


@functools.lru_cache(maxsize=None)
def _shor(which, aniso=False, extra_key=None):
    layout = LF.ANISO if aniso else LF.BASE
    T = terms(which, "aniso" if aniso else "iso")
    cs = CN.flat(fams(which, aniso))
    if extra_key is not None:
        cs = cs + list(_discovered(which))
    res = SDP.solve_shor(CO.assemble_Q(T), cs, solver="CLARABEL")
    eta, f = SDP.relative_gap(res, inst(which), layout, T)
    return res, eta, f


def _edge_sub(e):
    """Edge-e sub-vector: R_i, v_i, p_i, R_j, v_j, p_j, b_i, the 30 liftings, s (67 entries)."""
    return np.r_[np.arange(15 * e, 15 * e + 30), np.arange(45 + 6 * e, 51 + 6 * e),
                 np.arange(63 + 30 * e, 93 + 30 * e), [S_IDX]]


@functools.lru_cache(maxsize=None)
def _discovered(which):
    """Null-space directions per edge not spanned by the hand families, as Constraints."""
    r = ref(which)
    rng = np.random.default_rng(36)
    out = []
    for e in range(2):
        sub = _edge_sub(e)
        iu = np.triu_indices(67)
        Phi = np.array([np.outer(zs, zs)[iu] for zs in
                        (z_ref(random_state(rng), r)[sub] for _ in range(6000))])
        _, sv, Vt = np.linalg.svd(Phi, full_matrices=True)
        rank = int((sv > 1e-10 * sv[0]).sum())
        N = Vt[rank:].T
        keep = np.zeros(124, bool)
        keep[sub] = True
        H = []
        for c in CN.flat(fams(which)):
            rows, cols = c.A.nonzero()
            if c.b == 0 and keep[rows].all() and keep[cols].all():
                A = c.A.toarray()[np.ix_(sub, sub)]
                C = 2 * A
                C[np.diag_indices(67)] = np.diag(A)
                H.append(C[iu])
        Hn = np.array(H) @ N
        _, s2, V2 = np.linalg.svd(Hn, full_matrices=True)
        k = int((s2 > 1e-10 * s2[0]).sum())
        for u in V2[k:]:
            d = N @ u
            Cm = np.zeros((67, 67))
            Cm[iu] = d
            Asub = 0.5 * (Cm + Cm.T)                  # off-diagonal coefficient 2 A_ab -> A_ab
            Asub[np.diag_indices(67)] = np.diag(Cm)
            A = np.zeros((124, 124))
            A[np.ix_(sub, sub)] = Asub
            out.append(CN.Constraint("AT", f"e{e} autotight", sp.csr_matrix(A), 0.0))
    return tuple(out)


@pytest.mark.claim("M3-C36")
@pytest.mark.slow
def test_c36_discovered_directions_tighten():
    disc = _discovered("true")
    assert len(disc) == 2 * 27, len(disc)
    base, eta0, _ = _shor("true")
    more, eta1, _ = _shor("true", extra_key="autotight")
    assert more.rank < base.rank or eta1 < eta0, (base.rank, eta0, more.rank, eta1)


@pytest.mark.claim("M3-C37")
@pytest.mark.slow
def test_c37_shor_tight_noise_free():
    res, eta, f = _shor("true")
    assert res.sv_ratio >= 1e5 and eta <= 1e-4, (res.status, res.primal, res.dual, res.sv_ratio,
                                                 res.rank, eta, f)


@pytest.mark.claim("M3-C38")
@pytest.mark.slow
def test_c38_aniso_keeps_tightness():
    res, eta, f = _shor("true", aniso=True)
    assert res.sv_ratio >= 1e5 and eta <= 1e-4, (res.status, res.primal, res.dual, res.sv_ratio,
                                                 res.rank, eta, f)
