"""One test per M2 claim, bound with ``@pytest.mark.claim``.

Read ``docs/claims/M2.yaml`` alongside this file: each test asserts what its claim says,
at the tolerance its claim states, and nothing is relaxed to make a baseline look better
than it is.

The single most important structural fact encoded here is that **the three baselines do
not share a tolerance**:

* Tier 0 (``1e-10``) -- against the exact closed form.  Only Baseline 3 qualifies; it is
  exact zero-order-hold and *is* ground truth on this fixture.
* Tier 1 (``1e-6``)  -- Baselines 1 vs 2, which share a discretization but not a
  retraction.  The ``1e-6`` covers ``R`` as well as ``v`` and ``p``: Baseline 2's raw
  Cayley retraction loses ``||omega||^3 T dt^2 / 12`` by construction, 3.0e-8 on this
  fixture, so Tier 0 in ``R`` is not merely unmet between 1 and 2 -- it is unattainable
  without a 64x refinement of a ``dt`` that M2-C01 pins.  That 3.0e-8 is a property of
  the method and is asserted *positively* in
  ``test_c22b_baseline2_retraction_really_is_cayley``.
* Tier 2 (``5e-3``)  -- Baselines 1/2 vs Baseline 3, which do not.  A magnitude bound
  alone would pass a baseline that is wrong by a constant ``1e-3``, so Tier 2 is always
  paired with a convergence-order test.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gtsam
from gtsam import Gal3, NavState, Pose3, Rot3, Values
from gtsam import NonlinearFactorGraph, Ordering, noiseModel
from gtsam.imuBias import ConstantBias
from gtsam.symbol_shorthand import B, V, X
from scipy.integrate import quad
from scipy.linalg import expm

from src import trajectory as T
from src import preintegration_cayley as B2
from src import preintegration_forster as B1
from src import preintegration_galilean as B3

REPO = Path(__file__).resolve().parents[1]

TIER0 = 1e-10      # vs the exact closed form
TIER1 = 1e-6       # B1 <-> B2, same discretization
TIER2 = 5e-3       # B1/B2 <-> B3, different discretization
ORDER_BAND = (0.45, 0.55)   # a first-order method halves its error when dt halves

STEP_SWEEP = (100, 200, 400, 800, 1600)


def _max_abs(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def _rot_err(A, B_) -> float:
    """Chart-consistent rotation distance, ``||Log(A^T B)||_inf``."""
    return float(np.abs(T.so3_log(np.asarray(A).T @ np.asarray(B_))).max())


def _zoh_increment_expm(w, a, Tt):
    """Exact constant-input increment ``(dR, dv, dp)`` from ONE 5x5 matrix exponential.

    ``X = [[R, v, p], [0, 1, t], [0, 0, 1]]`` obeys ``dX/dt = X A`` with
    ``A = [[w^, a, 0], [0, 0, 1], [0, 0, 0]]`` (``dR = R w^``, ``dv = R a``, ``dp = v``,
    ``dt = 1``), so ``X(T) = expm(T A)``.  ``scipy.linalg.expm`` and nothing from
    ``src`` beyond ``hat`` -- an independent reference for the GTSAM side, not a
    re-reading of ``T.M1`` / ``T.M2``.
    """
    A = np.zeros((5, 5))
    A[0:3, 0:3] = T.hat(np.asarray(w, dtype=float))
    A[0:3, 3] = np.asarray(a, dtype=float)
    A[3, 4] = 1.0
    X = expm(Tt * A)
    return X[0:3, 0:3], X[0:3, 3], X[0:3, 4]


# ======================================================================================
# Fixture
# ======================================================================================


@pytest.mark.claim("M2-C01")
def test_c01_fixture_is_deterministic_bitwise():
    """The fixture has no RNG, so two *separate processes* agree bitwise, not merely
    to within a tolerance."""
    script = (
        "import numpy as np, sys;"
        "sys.path.insert(0, %r);" % str(REPO) +
        "from src import trajectory as T;"
        "s=T.measurement_streams();"
        "out={'gyro':[a.tolist() for a,_ in s],'accel':[b.tolist() for _,b in s],"
        "'g':T.G_W.tolist(),'dt':T.DT,'T_edge':T.T_EDGE,'steps':T.STEPS_PER_EDGE,"
        "'n_edges':T.N_EDGES,'b':T.BIAS_TRUE.tolist(),'R0':T.R_0.tolist(),"
        "'p0':T.P_0.tolist(),'v0':T.V_0.tolist()};"
        "import json;print(json.dumps(out))"
    )
    runs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", script], cwd=str(REPO),
                              capture_output=True, text=True, check=True)
        runs.append(json.loads(proc.stdout))
    a, b = runs

    for e in range(T.N_EDGES):
        g_a, g_b = np.array(a["gyro"][e]), np.array(b["gyro"][e])
        f_a, f_b = np.array(a["accel"][e]), np.array(b["accel"][e])
        assert g_a.shape == (T.STEPS_PER_EDGE, 3)
        assert np.array_equal(g_a, g_b), f"gyro stream of edge {e} is not reproducible"
        assert np.array_equal(f_a, f_b), f"accel stream of edge {e} is not reproducible"
        assert g_a.tobytes() == g_b.tobytes()
        assert f_a.tobytes() == f_b.tobytes()
        # omega_k = w_e + b_g and a_k = f_e + b_a EXACTLY, every step
        assert np.array_equal(g_a, np.tile(T.OMEGA_BODY[e] + T.B_G, (T.STEPS_PER_EDGE, 1)))
        assert np.array_equal(f_a, np.tile(T.F_BODY[e] + T.B_A, (T.STEPS_PER_EDGE, 1)))

    for key in ("g", "dt", "T_edge", "steps", "n_edges", "b", "R0", "p0", "v0"):
        assert np.array_equal(np.array(a[key]), np.array(b[key])), f"{key} differs"

    assert a["dt"] == 2.5e-3 and a["T_edge"] == 0.5
    assert a["steps"] == 200 and a["n_edges"] == 2

    # -- the constants the claim STATES, pinned against literals retyped from its text --
    # The stream checks above compare the streams with tile(T.OMEGA_BODY + T.B_G), i.e.
    # against the module's own constants, so they are self-referential: a file-level edit
    # of B_G / B_A (verifier mutant N29) changes both sides and passes.  These literals are
    # typed from the M2-C01 statement, not printed from src, and are checked both in the
    # generating subprocesses (a file-level edit) and in this process (a monkeypatch).
    claimed = {
        "b_g": [0.0100, -0.0150, 0.0080], "b_a": [0.0600, -0.0400, 0.0900],
        "w_0": [0.35, -0.22, 0.17], "f_0": [0.60, 0.45, 9.95],
        "w_1": [-0.28, 0.41, -0.33], "f_1": [-0.75, 0.90, 9.60],
        "rot_0": [0.10, -0.20, 0.30], "v_0": [0.50, -0.30, 0.20],
    }
    assert np.array_equal(np.array(a["b"]), claimed["b_g"] + claimed["b_a"]), \
        f"generated b = {a['b']} is not the claimed [b_g; b_a]"
    assert np.array_equal(np.array(a["v0"]), claimed["v_0"])
    for e, (w_key, f_key) in enumerate((("w_0", "f_0"), ("w_1", "f_1"))):
        # every generated sample is exactly w_e + b_g / f_e + b_a with the CLAIMED values
        assert np.array_equal(np.array(a["gyro"][e]), np.tile(
            np.array(claimed[w_key]) + np.array(claimed["b_g"]), (200, 1))), \
            f"gyro stream of edge {e} is not the claimed w_{e} + b_g"
        assert np.array_equal(np.array(a["accel"][e]), np.tile(
            np.array(claimed[f_key]) + np.array(claimed["b_a"]), (200, 1))), \
            f"accel stream of edge {e} is not the claimed f_{e} + b_a"
    # R_0 = Exp(rot_0), against scipy's matrix exponential rather than src's Rodrigues
    assert _max_abs(np.array(a["R0"]), expm(T.hat(claimed["rot_0"]))) <= 1e-15
    # and the in-process module constants
    for name, value in (("B_G", claimed["b_g"]), ("B_A", claimed["b_a"]),
                        ("ROT_VEC_0", claimed["rot_0"]), ("V_0", claimed["v_0"])):
        assert np.array_equal(getattr(T, name), value), f"T.{name} = {getattr(T, name)}"
    assert np.array_equal(T.BIAS_TRUE, claimed["b_g"] + claimed["b_a"])
    for e, (w_key, f_key) in enumerate((("w_0", "f_0"), ("w_1", "f_1"))):
        assert np.array_equal(T.OMEGA_BODY[e], claimed[w_key])
        assert np.array_equal(T.F_BODY[e], claimed[f_key])
    assert a["steps"] * a["dt"] == a["T_edge"]
    assert np.array_equal(np.array(a["g"]), [0.0, 0.0, -9.81])
    assert np.array_equal(np.array(a["p0"]), np.zeros(3))


@pytest.mark.claim("M2-C02")
def test_c02_closed_form_increment_is_the_exact_reference():
    """(a) ``M1``/``M2`` against adaptive quadrature of the same integrals;
    (b) the chained states against the 15-digit Sec. 0.4 literals."""
    # -- (a) ---------------------------------------------------------------------------
    worst_M1 = worst_M2 = 0.0
    for edge in T.edges():
        w, Tedge = edge.omega_body, edge.duration
        assert np.linalg.norm(w) > 1e-3, "closed form needs ||w|| > 0"
        M1c, M2c = T.M1(w, Tedge), T.M2(w, Tedge)
        M1n = np.empty((3, 3))
        M2n = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                M1n[i, j] = quad(lambda s, i=i, j=j: T.so3_exp(w * s)[i, j],
                                 0.0, Tedge, epsabs=1e-14, epsrel=1e-14, limit=400)[0]
                M2n[i, j] = quad(lambda s, i=i, j=j: T.M1(w, s)[i, j],
                                 0.0, Tedge, epsabs=1e-14, epsrel=1e-14, limit=400)[0]
        worst_M1 = max(worst_M1, _max_abs(M1c, M1n))
        worst_M2 = max(worst_M2, _max_abs(M2c, M2n))
    assert worst_M1 <= 1e-14, f"M1 vs quadrature {worst_M1:.3e}"
    assert worst_M2 <= 1e-14, f"M2 vs quadrature {worst_M2:.3e}"

    # -- (b) ---------------------------------------------------------------------------
    for edge, (dRr, dvr, dpr) in zip(T.edges(), T.REF_INCREMENTS):
        dR, dv, dp = edge.exact_increment()
        assert _max_abs(dR, dRr) <= 1e-12
        assert _max_abs(dv, dvr) <= 1e-12
        assert _max_abs(dp, dpr) <= 1e-12

    states = T.exact_states()
    assert _max_abs(states[1].v, T.REF_V_KF1) <= 1e-12
    assert _max_abs(states[1].p, T.REF_P_KF1) <= 1e-12
    assert _max_abs(states[2].v, T.REF_V_KF2) <= 1e-12
    assert _max_abs(states[2].p, T.REF_P_KF2) <= 1e-12
    assert _max_abs(states[2].R, T.REF_R_KF2) <= 1e-12


# ======================================================================================
# Baseline 1 -- Forster
# ======================================================================================


@pytest.mark.claim("M2-C03")
def test_c03_component_endpoint_relation_matches_gtsam_predict():
    """``R_j = R_i dR``, ``v_j = v_i + g Dt + R_i dv``,
    ``p_j = p_i + v_i Dt + 0.5 g Dt^2 + R_i dp`` reproduces ``predict()``.

    ``g_W`` is the gravity VECTOR (GTSAM's ``n_gravity``), ``[0, 0, -9.81]`` for z-up.
    Holds for any state -- there is no time-coordinate restriction here; that only
    appears once the relation is lifted to Gal(3) (claim M2-C18).
    """
    state_i = T.NavTriple(R=T.so3_exp([0.3, 0.1, -0.4]),
                          v=np.array([-0.7, 0.2, 1.1]),
                          p=np.array([1.5, -2.5, 3.5]))
    for edge in T.edges():
        gal = B3.integrate_edge(edge, T.BIAS_TRUE)
        predicted = gal.predict(state_i)
        formula = T.chain(state_i, gal.dR, gal.dv, gal.dp, gal.dt_total)
        assert _max_abs(predicted.R, formula.R) <= 1e-12
        assert _max_abs(predicted.v, formula.v) <= 1e-12
        assert _max_abs(predicted.p, formula.p) <= 1e-12


@pytest.mark.claim("M2-C04")
def test_c04_forster_increment_is_first_order_in_dt():
    """Baseline 1's Euler sums are O(dt) in ``dv`` and ``dp``.

    ``dR`` is exact here *only* because the fixture's body rate is constant per edge, so
    the product of per-step exponentials telescopes to ``Exp(w T)``.  Do not generalize.

    Part (c) is the half with teeth.  A convergence *ratio* cannot see which rotation the
    Eq. (9b)/(9c) sums use: ``dR_ik`` (the partial increment, correct) and ``dR_{i,k+1}``
    (this step's rotation already folded in, wrong) are a left and a right rectangle rule
    for the same integral, so both are O(dt) and both halve.  Worse, their *magnitudes*
    agree to four digits -- at n = 200 the correct sum is off by 2.1709e-3 and the folded
    one by 2.1710e-3 -- so asserting the magnitude cannot separate them either.  What
    separates them is the **sign**: by Euler-Maclaurin the left rule's leading truncation
    is ``-(dt/2)(Exp(w T) - I) a`` and the right rule's is ``+`` that.  Asserted below.
    """
    edge_dv, edge_dp, edge_dR = [], [], []
    for n in STEP_SWEEP:
        edge = T.edges(n)[0]
        pim = B1.integrate_edge(edge, T.BIAS_TRUE)
        dR, dv, dp = edge.exact_increment()
        edge_dv.append(_max_abs(pim.dv, dv))
        edge_dp.append(_max_abs(pim.dp, dp))
        edge_dR.append(_max_abs(pim.dR, dR))

    assert max(edge_dR) <= 1e-12, f"dR should telescope exactly here: {edge_dR}"
    for name, errs in (("dv", edge_dv), ("dp", edge_dp)):
        ratios = [errs[k + 1] / errs[k] for k in range(len(errs) - 1)]
        assert all(ORDER_BAND[0] <= r <= ORDER_BAND[1] for r in ratios), \
            f"{name} convergence ratios {ratios} (errors {errs})"

    # -- (b) magnitude at the fixture's own dt, so the O(dt) constant is pinned too -----
    for index, expected in ((0, 2.170879e-3), (1, 2.760238e-3)):
        edge = T.edges()[index]
        pim = B1.integrate_edge(edge, T.BIAS_TRUE)
        _, dv, _ = edge.exact_increment()
        got = _max_abs(pim.dv, dv)
        assert abs(got - expected) <= 0.10 * expected, \
            f"edge {index}: |dv - dv_exact| = {got:.6e}, expected {expected:.6e} +-10%"

    # -- (c) the SIGNED leading truncation: this is what pins dR_ik vs dR_{i,k+1} -------
    for n in STEP_SWEEP:
        for edge in T.edges(n):
            pim = B1.integrate_edge(edge, T.BIAS_TRUE)
            _, dv, _ = edge.exact_increment()
            a, w, duration = edge.f_body, edge.omega_body, edge.duration
            predicted = -(edge.dt / 2.0) * (T.so3_exp(w * duration) - np.eye(3)) @ a
            relative = _max_abs(pim.dv - dv, predicted) / np.abs(predicted).max()
            assert relative <= 1e-3, (
                f"edge {edge.index} n={n}: signed dv truncation {pim.dv - dv} does not "
                f"match the LEFT rectangle rule's -(dt/2)(Exp(wT)-I)a = {predicted} "
                f"(relative {relative:.4f}).  A relative gap of ~2 means the sums use "
                f"dR_{{i,k+1}} instead of dR_ik.")

    # -- (d) the SIGNED dp truncation, to second order: pins dR_ik in the dp sum ----------
    # Part (c) pins the rotation used in dv only.  A dp-only fold (``0.5 dR_{i,k+1} a dt^2``,
    # verifier mutant F05a3) shifts dp by exactly 0.5 dt^2 (Exp(wT) - I) a -- 1% of the
    # O(dt) truncation, so neither the ratios nor a first-order signed check can see it.
    # With f(t) = Exp(wt) a and F(t) = int_0^t f, the left-rule Euler-Maclaurin expansion
    # dv_k = F(t_k) - (dt/2)(f(t_k) - a) + (dt^2/12)(f'(t_k) - f'(0)) + O(dt^4), summed once
    # more for sum_k dv_k dt and added to 0.5 dt^2 sum_k f_k, gives
    #   dp_euler - dp_exact = -(dt/2)(dv_exact - T a)
    #                         + dt^2 [ (Exp(wT) - I) a / 6 - T hat(w) a / 12 ] + O(dt^3).
    # The O(dt^3) remainder is measured at 1.2e-4 (n=200) of the mutant's shift and halves
    # with dt; the dp-only fold moves the dt^2 coefficient of (Exp(wT) - I) a from 1/6 to
    # 2/3, i.e. by exactly 1.0 of the shift.  Bound: 1e-2 of the shift.
    for n in STEP_SWEEP:
        for edge in T.edges(n):
            pim = B1.integrate_edge(edge, T.BIAS_TRUE)
            dR_ex, dv_ex, dp_ex = edge.exact_increment()
            a, w, duration, dt = edge.f_body, edge.omega_body, edge.duration, edge.dt
            predicted = (-(dt / 2.0) * (dv_ex - duration * a)
                         + dt**2 * ((dR_ex - np.eye(3)) @ a / 6.0
                                    - duration * (T.hat(w) @ a) / 12.0))
            fold_shift = np.abs(0.5 * dt**2 * (dR_ex - np.eye(3)) @ a).max()
            relative = _max_abs(pim.dp - dp_ex, predicted) / fold_shift
            assert relative <= 1e-2, (
                f"edge {edge.index} n={n}: signed dp truncation {pim.dp - dp_ex} misses the "
                f"second-order Euler-Maclaurin prediction {predicted} by {relative:.3e} of "
                f"0.5 dt^2 (Exp(wT)-I) a.  ~1.0 means the dp sum folds dR_{{i,k+1}}.")


@pytest.mark.claim("M2-C05")
def test_c05_forster_bias_correction_is_mixed_chart_and_quadratic():
    """Forster Eq. (44): right group composition for ``dR``, additive for ``dv``/``dp``,
    all five Jacobians frozen at ``bbar``.

    The Jacobians come from finite differences on re-integration because GTSAM 4.3.0
    exposes **no** ``delRdelBiasOmega`` / ``delVdelBias*`` / ``delPdelBias*`` on either
    PIM class in Python -- the two classes have identical member sets.
    """
    pimg_members = {m for m in dir(gtsam.PreintegratedImuMeasurementsG)
                    if not m.startswith("_")}
    pim_members = {m for m in dir(gtsam.PreintegratedImuMeasurements)
                   if not m.startswith("_")}
    assert pimg_members == pim_members
    for absent in ("delRdelBiasOmega", "delVdelBiasOmega", "delVdelBiasAcc",
                   "delPdelBiasOmega", "delPdelBiasAcc"):
        assert absent not in pimg_members, f"{absent} unexpectedly available"

    # ``increment_error`` itself, on hand-computed cases (verification pass 3, P22: a
    # version that ignored dp left this whole test blind to dp-correction errors).  Each
    # case perturbs exactly one component by a known amount.
    identity = (np.eye(3), np.zeros(3), np.zeros(3))
    c, s = np.cos(0.25), np.sin(0.25)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # 0.25 rad about z
    for case, other, expected in (
            ("dR", (Rz, np.zeros(3), np.zeros(3)), 0.25),
            ("dv", (np.eye(3), np.array([0.0, -3e-4, 1e-4]), np.zeros(3)), 3e-4),
            ("dp", (np.eye(3), np.zeros(3), np.array([2e-5, 0.0, -7e-5])), 7e-5)):
        for a, b in ((other, identity), (identity, other)):
            got = B1.increment_error(a, b)
            assert abs(got - expected) <= 1e-15, \
                f"increment_error on a {case}-only difference: {got!r}, expected {expected!r}"
    # the max over components, not a sum
    assert B1.increment_error((Rz, np.array([0.0, -3e-4, 1e-4]),
                               np.array([2e-5, 0.0, -7e-5])), identity) == \
        B1.increment_error((Rz, np.zeros(3), np.zeros(3)), identity)

    direction = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    direction /= np.linalg.norm(direction)
    for edge in T.edges():
        base = B1.integrate_edge(edge, T.BIAS_TRUE)
        jac = B1.bias_jacobians_fd(edge, T.BIAS_TRUE)
        J_R, J_vg, J_va, J_pg, J_pa = jac
        assert J_R.shape == J_vg.shape == J_pa.shape == (3, 3)

        errors = {}
        components = {"dR": {}, "dv": {}, "dp": {}}
        for mag in (1e-3, 1e-2):
            db = mag * direction
            corrected = B1.bias_corrected_increment(base.increment(), jac, db)
            # the rotation correction really is a RIGHT group composition
            assert np.allclose(corrected[0],
                               base.dR @ T.so3_exp(J_R @ db[0:3]), atol=0, rtol=0)
            reintegrated = B1.integrate_edge(edge, T.BIAS_TRUE + db).increment()
            errors[mag] = B1.increment_error(corrected, reintegrated)
            # ... and component by component, without increment_error: dp's residual is
            # ~7x below dv's (3.0e-9 vs 2.1e-8 at 1e-3), so a max over components can
            # hide a first-order dp error behind dv's second-order one
            components["dR"][mag] = _rot_err(reintegrated[0], corrected[0])
            components["dv"][mag] = _max_abs(corrected[1], reintegrated[1])
            components["dp"][mag] = _max_abs(corrected[2], reintegrated[2])

        ratio = errors[1e-2] / errors[1e-3]
        assert 50.0 <= ratio <= 200.0, \
            f"edge {edge.index}: not quadratic, ratio {ratio:.1f} (errors {errors})"
        # measured 100.04 / 100.01 / 100.005 (edge 0), 100.10 / 100.01 / 100.01 (edge 1)
        for name, errs in components.items():
            ratio = errs[1e-2] / errs[1e-3]
            assert 50.0 <= ratio <= 200.0, (
                f"edge {edge.index}: the {name} correction is not quadratic, ratio "
                f"{ratio:.1f} (errors {errs})")


class _StreamEdge:
    """Duck-typed ``Edge`` over an arbitrary stream, so the baselines' own ``integrate_edge``
    and ``bias_jacobians_fd`` can run on the time-varying stream."""

    def __init__(self, gyro, accel, dt):
        self._gyro, self._accel, self.dt = gyro, accel, dt

    def gyro_stream(self):
        return self._gyro

    def accel_stream(self):
        return self._accel


def _cayley_equivalent_gyro(gyro, dt, b_g):
    """The gyro stream whose EXACT ZOH rotation equals Baseline 2's product of ``Cay``.

    ``Cay(theta u) = Exp(2 arctan(theta/2) u)`` (the angle identity of M2-C22 (a)), so
    rescaling each bias-corrected rate by ``2 arctan(theta_k/2) / theta_k`` with
    ``theta_k = ||(omega_k - b_g) dt||`` makes GTSAM's ``prod Exp`` reproduce ``prod Cay``.
    """
    w = gyro - b_g
    theta = np.linalg.norm(w * dt, axis=1)
    return w * (2.0 * np.arctan(theta / 2.0) / theta)[:, None] + b_g


def _gtsam_euler_reference(gyro, accel, dt, bias):
    """``(dR, dv, dp)`` that the Eq. (9) Euler sums MUST produce on ``(gyro, accel)``,
    built from GTSAM's exact-ZOH Galilean preintegration and nothing in ``src`` but
    the bias-block adapter and the fixture's closed-form ``M1``/``M2``.

    GTSAM's rotation is ``prod Exp((omega_k - b_g) dt)`` -- the same product as Baseline 1,
    for ANY stream.  Its velocity and position are the ZOH sums
    ``dv += dR_k M1(w_k, dt) a_k`` and ``dp += dv_k dt + dR_k M2(w_k, dt) a_k``, whereas the
    Euler sums use ``a_k dt`` and ``a_k dt^2 / 2``.  So the difference is known in closed
    form step by step, with GTSAM's own running ``dR_k``:
    ``e_v += dR_k (M1 - dt I) a_k``, ``e_p += e_v dt + dR_k (M2 - dt^2/2 I) a_k`` -- an exact
    identity, not an expansion, so the comparison holds to round-off.
    """
    params = gtsam.PreintegrationParams(np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurementsG(params, B3.to_gtsam_bias(bias))
    e_v = np.zeros(3)
    e_p = np.zeros(3)
    for k in range(len(gyro)):
        R_k = pim.deltaRij().matrix()
        w = gyro[k] - bias[0:3]
        a = accel[k] - bias[3:6]
        e_p = e_p + e_v * dt + R_k @ (T.M2(w, dt) - 0.5 * dt**2 * np.eye(3)) @ a
        e_v = e_v + R_k @ (T.M1(w, dt) - dt * np.eye(3)) @ a
        pim.integrateMeasurement(accel[k], gyro[k], dt)
    return (pim.deltaRij().matrix(), np.array(pim.deltaVij()) - e_v,
            np.array(pim.deltaPij()) - e_p), pim


@pytest.mark.claim("M2-F08")
def test_c05b_baselines_1_and_2_on_a_time_varying_stream():
    """Baselines 1 and 2 against GTSAM on the M2-C19/C21 time-varying stream.

    Every other Baseline-1/2 test runs on the fixture, whose twist is constant per edge.
    There ``Exp(w dt)`` commutes with the running ``dR`` and ``J_R`` commutes with ``dR``, so
    accumulating the rotation on the LEFT (verifier mutant N24, 1.7e-15 on the fixture) and
    conjugating ``J_R`` by ``dR`` (N11) were both invisible.

    (a) Rotation: Baseline 1's ``prod Exp`` against GTSAM's, and Baseline 2's ``prod Cay``
        against GTSAM fed the Cayley-equivalent rates.  Measured 4.6e-18 / 2.7e-17; the
        left-accumulated product is 4.5e-5 away.  Bound 1e-12.
    (b) ``dv``/``dp``: against GTSAM's ZOH increment minus the step-by-step closed-form
        ZOH-vs-Euler difference (see :func:`_gtsam_euler_reference`).  Measured
        <= 5.2e-15; a dp-only ``dR_{i,k+1}`` fold moves dp by 2.2e-6.  Bound 1e-12.
    (c) ``J_R`` from Baseline 1's finite differences against the rotation block of GTSAM's
        bias correction, which is ``dR Exp(J_R db_g)`` exactly (M2-C20).  Measured 1.8e-11
        against |J_R| = 0.10; ``dR J_R dR^T`` is 2.3e-6 away.  Bound 1e-9.
    """
    gyro, accel, dt = T.time_varying_stream()
    bias = T.BIAS_TRUE
    stream = _StreamEdge(gyro, accel, dt)

    # the stream really is non-commuting: the two products differ (scipy, not src)
    right = left = np.eye(3)
    for k in range(len(gyro)):
        step = expm(T.hat((gyro[k] - bias[0:3]) * dt))
        right, left = right @ step, step @ left
    assert _rot_err(right, left) >= 1e-5, "time-varying stream should not commute"

    # -- (a) + (b) --------------------------------------------------------------------------
    cases = (("B1", B1.ForsterPreintegration, gyro, {}),
             ("B2-raw", B2.CayleyPreintegration,
              _cayley_equivalent_gyro(gyro, dt, bias[0:3]), {"variant": B2.VARIANT_RAW}))
    for name, cls, reference_gyro, kwargs in cases:
        pim = B1.integrate_edge(stream, bias, cls=cls, **kwargs)
        (dR_ref, dv_ref, dp_ref), _ = _gtsam_euler_reference(reference_gyro, accel, dt, bias)
        errs = {"R": _rot_err(dR_ref, pim.dR), "v": _max_abs(pim.dv, dv_ref),
                "p": _max_abs(pim.dp, dp_ref)}
        bad = {k: f"{v:.3e}" for k, v in errs.items() if v > 1e-12}
        assert not bad, f"{name} on the time-varying stream: {bad}"

    # -- (c) --------------------------------------------------------------------------------
    _, gtsam_pim = _gtsam_euler_reference(gyro, accel, dt, bias)
    R_hat = gtsam_pim.deltaRij().matrix()
    eps = B1.FD_EPS
    J_ref = np.zeros((3, 3))
    for m in range(3):
        step = np.zeros(6)
        step[m] = eps
        plus = gtsam_pim.predict(NavState(), B3.to_gtsam_bias(bias + step))
        minus = gtsam_pim.predict(NavState(), B3.to_gtsam_bias(bias - step))
        J_ref[:, m] = (_indep_so3_log(R_hat.T @ plus.attitude().matrix())
                       - _indep_so3_log(R_hat.T @ minus.attitude().matrix())) / (2.0 * eps)
    J_R = B1.bias_jacobians_fd(stream, bias)[0]
    assert np.abs(J_ref).max() >= 1e-2
    assert _max_abs(J_R, J_ref) <= 1e-9, \
        f"J_R on the time-varying stream is {_max_abs(J_R, J_ref):.3e} from GTSAM's"


@pytest.mark.claim("M2-C06")
def test_c06_forster_covariance_is_symmetric_psd_and_monotone():
    """Forster Eq. (63), checked at *every* step, not just the last one."""
    sigma_g, sigma_a = 1.7e-4, 1e-2
    # NavState block order (rot, pos, vel) vs this module's (rot, vel, pos)
    P = np.zeros((9, 9))
    P[0:3, 0:3] = np.eye(3)
    P[3:6, 6:9] = np.eye(3)
    P[6:9, 3:6] = np.eye(3)

    for edge in T.edges():
        pim = B1.ForsterPreintegration(bias_hat=T.BIAS_TRUE,
                                       sigma_g=sigma_g, sigma_a=sigma_a)
        previous_trace = -np.inf
        for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
            pim.integrate_measurement(omega, acc, edge.dt)
            S = pim.cov
            assert S.shape == (9, 9)
            assert np.abs(S - S.T).max() <= 1e-14
            assert np.linalg.eigvalsh(0.5 * (S + S.T)).min() >= -1e-12
            trace = float(np.trace(S))
            assert trace >= previous_trace, "tr(Sigma) decreased"
            previous_trace = trace

        # cross-check shape and magnitude against GTSAM, like with like
        gtsam_cov = B3.integrate_edge(edge, T.BIAS_TRUE,
                                      sigma_g=sigma_g, sigma_a=sigma_a).preint_cov()
        assert gtsam_cov.shape == (9, 9)
        reordered = P @ pim.cov @ P.T
        relative = _max_abs(reordered, gtsam_cov) / np.abs(gtsam_cov).max()
        assert relative <= 1e-2, f"magnitude cross-check {relative:.3e}"


# Anisotropic densities for the covariance reference.  With ISOTROPIC noise dR Q dR^T = Q,
# so every orientation error in the noise-input or rotation-transition blocks of Eq. (63)
# is invisible (verifier pass 2, M2-F07: N06, N10, N28 survived for exactly that reason).
# Per-axis ratios 1 : 1.7 : 3 (gyro) and 1 : 2 : 4 (accel).
ANISO_SIGMA_G = np.array([1.0e-4, 1.7e-4, 3.0e-4])
ANISO_SIGMA_A = np.array([5.0e-3, 1.0e-2, 2.0e-2])
#: central-difference step on the per-step noise samples; truncation O(eps^2) ~ 1e-10 and
#: round-off ~1e-16/eps ~ 1e-11, against covariance entries that are O(1) in relative terms
NOISE_FD_EPS = 1e-5


def _blocks(S):
    return {(i, j): S[3 * i:3 * i + 3, 3 * j:3 * j + 3] for i in range(3) for j in range(3)}


def _block_relative(A, B_):
    """Per 3x3 block, ``max|A - B| / max|B|`` -- one number per block, upper triangle."""
    a, b = _blocks(A), _blocks(B_)
    return {k: float(np.abs(a[k] - b[k]).max() / np.abs(b[k]).max())
            for k in a if k[0] <= k[1]}


def _indep_so3_log(R):
    """Rotation vector of ``R`` from the trace and the skew part.  Retyped here so that the
    covariance reference shares no code with ``src`` beyond ``hat``."""
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = np.arccos(c)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return skew if th < 1e-12 else (th / np.sin(th)) * skew


def _euler_step(w, a, dt):
    """One step of the noise-free Eq. (9) scheme as a gravity-free Gal(3)-style increment
    ``(R, v, p, t)``: ``(Exp(w dt), a dt, a dt^2 / 2, dt)``, Exp via ``scipy.linalg.expm``."""
    return (expm(T.hat(w * dt)), a * dt, 0.5 * a * dt**2, dt)


def _compose(X, Y):
    """``(R1, v1, p1, t1) . (R2, v2, p2, t2) = (R1 R2, v1 + R1 v2, p1 + v1 t2 + R1 p2,
    t1 + t2)``.  Composing ``_euler_step``s with this law IS the Eq. (9a)-(9c) recursion:
    ``dv += dR a dt`` and ``dp += dv dt + 0.5 dR a dt^2``."""
    R1, v1, p1, t1 = X
    R2, v2, p2, t2 = Y
    return (R1 @ R2, v1 + R1 @ v2, p1 + v1 * t2 + R1 @ p2, t1 + t2)


def _covariance_reference(edge_index, sigma_g=ANISO_SIGMA_G, sigma_a=ANISO_SIGMA_A):
    """Covariance of the DISCRETE noisy Eq. (9) integrator, by linearising it numerically.

    The noisy increment is a function of the 6n per-step noise samples ``eta_k`` (added to
    the bias-corrected rate and specific force of step k).  Its error in Forster's chart,
    ``(Log(dR0^T dR), dv - dv0, dp - dp0)``, has Jacobian ``J_k`` with respect to
    ``eta_k``, obtained here by central differences on ``prefix_k . step_k(eta) .
    suffix_k`` -- nothing from ``src.preintegration_forster`` is called.  With independent
    samples of covariance ``diag(sigma^2) / dt``, ``Sigma = sum_k J_k Q J_k^T`` to first
    order.  Eq. (63) is the exact first-order recursion of this same scheme, so the two
    must agree to finite-difference precision, block by block.  The ``J_k`` are cached:
    they depend on the fixture constants only, never on ``src`` code a mutant could
    change.  The densities default to the anisotropic pair; either may be zero.
    """
    dt = T.edges()[edge_index].dt
    Q = np.diag(np.concatenate([np.broadcast_to(np.asarray(sigma_g, float), (3,))**2,
                                np.broadcast_to(np.asarray(sigma_a, float), (3,))**2])) / dt
    return sum(J @ Q @ J.T for J in _noise_jacobians(edge_index))


@functools.lru_cache(maxsize=None)
def _noise_jacobians(edge_index):
    """Per-step noise Jacobians ``J_k`` (9x6, w.r.t. ``[eta_g; eta_a]``) of the discrete
    noisy Eq. (9) integrator; see :func:`_covariance_reference`."""
    edge = T.edges()[edge_index]
    w = edge.gyro_stream() - T.BIAS_TRUE[0:3]
    a = edge.accel_stream() - T.BIAS_TRUE[3:6]
    dt, n = edge.dt, edge.n_steps
    steps = [_euler_step(w[k], a[k], dt) for k in range(n)]
    identity = (np.eye(3), np.zeros(3), np.zeros(3), 0.0)
    prefix = [identity]
    for k in range(n):
        prefix.append(_compose(prefix[-1], steps[k]))
    suffix = [identity] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix[k] = _compose(steps[k], suffix[k + 1])
    R0, v0, p0, _ = prefix[n]

    def error(X):
        return np.concatenate([_indep_so3_log(R0.T @ X[0]), X[1] - v0, X[2] - p0])

    jacobians = []
    eps = NOISE_FD_EPS
    for k in range(n):
        J = np.zeros((9, 6))
        for m in range(6):
            e = np.zeros(6)
            e[m] = eps
            plus = _compose(_compose(prefix[k], _euler_step(w[k] + e[0:3], a[k] + e[3:6], dt)),
                            suffix[k + 1])
            minus = _compose(_compose(prefix[k], _euler_step(w[k] - e[0:3], a[k] - e[3:6], dt)),
                             suffix[k + 1])
            J[:, m] = (error(plus) - error(minus)) / (2.0 * eps)
        jacobians.append(J)
    return tuple(jacobians)


def _gtsam_anisotropic_params():
    params = gtsam.PreintegrationParams(T.G_W)
    params.setGyroscopeCovariance(np.diag(ANISO_SIGMA_G**2))
    params.setAccelerometerCovariance(np.diag(ANISO_SIGMA_A**2))
    params.setIntegrationCovariance(np.zeros((3, 3)))
    return params


@pytest.mark.claim("M2-F07")
def test_c06b_forster_covariance_against_an_independent_reference():
    """Eq. (63) is checked for CORRECTNESS, not only for structure.

    ``test_c06`` asserts symmetry, PSD and a monotone trace.  Those hold for every
    covariance mutant the verifier tried (8/8), including a sign flip of the rotation
    coupling, and its single normalised GTSAM cross-check is dominated by the vv block.

    (a) Block by block against :func:`_covariance_reference`, a numerical linearisation of
        the discrete noisy integrator with ANISOTROPIC gyro and accel densities.  Measured
        agreement <= 2.7e-9 relative in every block on both edges; the bound is 1e-6.  The
        weakest of the verifier's mutants (``Jr -> I``) moves a block by 4.3e-4, the
        others by 5e-3 (noise scaled by 0.5%) up to 2.0 (sign of the coupling).

    (b) Against GTSAM's Galilean ``preintMeasCov()``, with the CHART made explicit.  GTSAM
        perturbs the increment on the right in Gal(3), ``U Exp(xi)``, whose velocity and
        position components are ``dR nu`` and ``dR rho``; Forster perturbs ``dv`` and
        ``dp`` additively.  So ``xi_Gal = M xi_Forster`` with ``M = diag(I, dR^T, dR^T)``
        (rot, vel, pos), and the covariances relate by ``M Sigma M^T``.  Untransformed,
        the rotation/velocity cross block differs by 0.11 / 0.20 on the two edges (0.17 /
        0.19 with isotropic noise -- the "17%" observation of pass 2); that is the
        rotation of the cross block by ``dR`` (0.22 / 0.30 rad), not a bug, and it does
        not shrink with dt.  Transformed, the rotation block agrees to round-off and every
        other block agrees to an O(dt) residual -- GTSAM integrates the noise through the
        exact ZOH step, Forster through the Euler step -- that halves exactly with dt
        (measured ratios 0.4998..0.5009 over n = 100..1600).
    """
    # -- (a) ------------------------------------------------------------------------------
    for edge in T.edges():
        pim = B1.integrate_edge(edge, T.BIAS_TRUE,
                                sigma_g=ANISO_SIGMA_G, sigma_a=ANISO_SIGMA_A)
        rel = _block_relative(pim.cov, _covariance_reference(edge.index))
        bad = {k: f"{v:.3e}" for k, v in rel.items() if v > 1e-6}
        assert not bad, (
            f"edge {edge.index}: Eq. (63) disagrees with the linearised discrete integrator "
            f"in blocks (rot=0, vel=1, pos=2) {bad}")

    # -- (b) ------------------------------------------------------------------------------
    params = _gtsam_anisotropic_params()
    # NavState order (rot, pos, vel) -> this module's (rot, vel, pos); P is an involution
    P = np.zeros((9, 9))
    P[0:3, 0:3] = np.eye(3)
    P[3:6, 6:9] = np.eye(3)
    P[6:9, 3:6] = np.eye(3)
    for index in range(T.N_EDGES):
        history = []
        for n in STEP_SWEEP:
            edge = T.edges(n)[index]
            pim = B1.integrate_edge(edge, T.BIAS_TRUE,
                                    sigma_g=ANISO_SIGMA_G, sigma_a=ANISO_SIGMA_A)
            ref = P @ B3.integrate_edge(edge, T.BIAS_TRUE, params=params).preint_cov() @ P
            M = np.eye(9)
            M[3:6, 3:6] = pim.dR.T
            M[6:9, 6:9] = pim.dR.T
            raw = _block_relative(pim.cov, ref)
            charted = _block_relative(M @ pim.cov @ M.T, ref)
            history.append(charted)
            assert raw[(0, 1)] >= 0.1, (
                f"edge {index} n={n}: untransformed R-v cross block within "
                f"{raw[(0, 1)]:.3e} -- the chart difference should be visible")
            assert charted[(0, 0)] <= 1e-12, \
                f"edge {index} n={n}: rotation block {charted[(0, 0)]:.3e}"
            if n == T.STEPS_PER_EDGE:
                worst = max(charted.values())
                assert worst <= 2e-2, \
                    f"edge {index}: chart-transformed blocks differ by {worst:.3e}"
        for k in history[0]:
            if k == (0, 0):
                continue
            ratios = [history[m + 1][k] / history[m][k] for m in range(len(history) - 1)]
            assert all(ORDER_BAND[0] <= r <= ORDER_BAND[1] for r in ratios), (
                f"edge {index} block {k}: the chart-transformed residual is not O(dt) "
                f"discretisation (ratios {ratios})")


@pytest.mark.claim("M2-F11")
def test_c06c_forster_covariance_with_a_single_noise_source():
    """Eq. (63) with exactly ONE nonzero density, against the same independent reference.

    Every other covariance check sets both densities, so the ``has_noise`` gate that
    decides whether Eq. (63) runs at all was untested where it matters: an ``or -> and``
    slip (verification pass 3, P17) silently returned a zero covariance for a gyro-only or
    accel-only configuration and left the whole suite green.

    * gyro only: every block is nonzero -- gyro noise reaches ``dv`` and ``dp`` through
      the rotation coupling ``A[3:9, 0:3]`` (the reason M2-C26 was refuted);
    * accel only: the rotation rows/columns are EXACTLY zero in both, since accel noise
      never enters ``dR``; the other blocks are checked relatively.

    Measured worst block: 5.2e-10 / 1.8e-9 (gyro-only, edges 0 / 1; tr = 2.3e-7) and
    2.7e-9 / 8.3e-10 (accel-only; tr = 2.8e-4), against the 1e-6 bound of test_c06b.
    """
    zero = np.zeros(3)
    for label, sigma_g, sigma_a in (("gyro-only", ANISO_SIGMA_G, zero),
                                    ("accel-only", zero, ANISO_SIGMA_A)):
        for edge in T.edges():
            pim = B1.integrate_edge(edge, T.BIAS_TRUE, sigma_g=sigma_g, sigma_a=sigma_a)
            ref = _covariance_reference(edge.index, sigma_g, sigma_a)
            assert np.trace(ref) > 0.0
            assert np.trace(pim.cov) >= 0.5 * np.trace(ref), (
                f"{label}, edge {edge.index}: tr(Sigma) = {np.trace(pim.cov):.3e} against "
                f"{np.trace(ref):.3e} -- the has_noise gate skipped Eq. (63)")
            if label == "accel-only":
                # src's covariance, not the reference's, carries the claim: the reference
                # zeroes these rows by construction, so asserting it here would only test
                # the reference.  The relative comparison below skips the rotation blocks
                # because they are asserted to be exactly zero on src here.
                assert not np.any(pim.cov[0:3, :]) and not np.any(pim.cov[:, 0:3]), \
                    f"accel-only, edge {edge.index}: accel noise leaked into the rotation"
                a, b = _blocks(pim.cov), _blocks(ref)
                rel = {k: float(np.abs(a[k] - b[k]).max() / np.abs(b[k]).max())
                       for k in ((1, 1), (1, 2), (2, 2))}
            else:
                rel = _block_relative(pim.cov, ref)
            bad = {k: f"{v:.3e}" for k, v in rel.items() if v > 1e-6}
            assert not bad, (
                f"{label}, edge {edge.index}: Eq. (63) disagrees with the linearised "
                f"discrete integrator in blocks (rot=0, vel=1, pos=2) {bad}")


# ======================================================================================
# Baseline 2 -- Rai Cayley (FROM-PAPER REIMPLEMENTATION, no public code exists)
# ======================================================================================


@pytest.mark.claim("M2-C07")
def test_c07_cayley_map_and_its_right_jacobian():
    """Rai Eq. (3) and Eq. (4).  Eq. (4) is the RIGHT Jacobian; the left convention
    fails by O(1), which is the kind of thing a from-paper reimplementation gets wrong
    silently."""
    rng = np.random.default_rng(20250212)

    # -- (a) Cay(2 tan(theta/2) a) == Exp(theta a) --------------------------------------
    axes = rng.standard_normal((2000, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    thetas = rng.uniform(1e-4, 3.0, size=2000)   # theta < 3: see the claim's domain
    worst = 0.0
    for a, th in zip(axes, thetas):
        worst = max(worst, _max_abs(B2.cay(B2.cayley_vector(th * a)), T.so3_exp(th * a)))
    assert worst <= 1e-13, f"Cay vs Exp {worst:.3e}"

    # -- (b) right vs left ---------------------------------------------------------------
    eps = 1e-7
    worst_right, worst_left = 0.0, 0.0
    for _ in range(300):
        u = rng.standard_normal(3)
        phi = 0.4 * u / np.linalg.norm(u)
        d = rng.standard_normal(3)
        numeric = (B2.cay(phi + eps * d) - B2.cay(phi)) / eps
        Jd = B2.cay_right_jacobian(phi) @ d
        worst_right = max(worst_right, _max_abs(numeric, B2.cay(phi) @ T.hat(Jd)))
        worst_left = max(worst_left, _max_abs(numeric, T.hat(Jd) @ B2.cay(phi)))
    assert worst_right <= 1e-5, f"right convention {worst_right:.3e}"
    assert worst_left >= 1e-1, f"left convention should fail, got {worst_left:.3e}"


@pytest.mark.claim("M2-C08")
def test_c08_cayley_denominator_is_never_singular():
    """Rai Eq. (17).  ``det(I - hat(xi)/2) = 1 + ||xi||^2/4 >= 1``, so Eq. (16) clears
    into Eq. (18) with no case analysis.  Totality on so(3)."""
    rng = np.random.default_rng(20250213)
    directions = rng.standard_normal((5000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    norms = rng.uniform(0.0, 5.0, size=5000)
    worst = 0.0
    for d, n in zip(directions, norms):
        xi = n * d
        actual = B2.cay_denominator_det(xi)
        closed = B2.cay_denominator_det_closed_form(xi)
        worst = max(worst, abs(actual - closed))
        assert closed >= 1.0
        assert actual >= 1.0 - 1e-12
    assert worst <= 1e-12, f"Eq. (17) deviation {worst:.3e}"


@pytest.mark.claim("M2-C09")
def test_c09_rotation_residual_is_an_exact_restatement_of_eq16():
    """Rai Eq. (19) vanishes **iff** Eq. (16) holds -- it is not a truncation."""
    rng = np.random.default_rng(20250214)
    worst_on_manifold, weakest_off_manifold = 0.0, np.inf
    for _ in range(500):
        R_i = T.so3_exp(rng.uniform(-np.pi / 2, np.pi / 2, 3))
        dRbar = T.so3_exp(rng.uniform(-0.5, 0.5, 3))
        xi = rng.uniform(-0.2, 0.2, 3)           # within the C07 chart restriction
        R_j = B2.rotation_from_eq16(R_i, dRbar, xi)

        on = np.abs(B2.rotation_residual_eq19(R_i, R_j, dRbar, xi)).max()
        worst_on_manifold = max(worst_on_manifold, float(on))

        u = rng.standard_normal(3)
        R_j_bad = R_j @ T.so3_exp(1e-3 * u / np.linalg.norm(u))
        off = np.abs(B2.rotation_residual_eq19(R_i, R_j_bad, dRbar, xi)).max()
        weakest_off_manifold = min(weakest_off_manifold, float(off))

    assert worst_on_manifold <= 1e-12, f"on-manifold residual {worst_on_manifold:.3e}"
    assert weakest_off_manifold >= 1e-4, \
        f"off-manifold residual too small: {weakest_off_manifold:.3e}"


@pytest.mark.claim("M2-C10")
def test_c10_velocity_and_position_residuals_at_both_state_sets():
    """Rai Eq. (21b)/(21c) at ``db = 0``, evaluated at BOTH state sets.

    The distinction is the whole content of this test, and the claim used to get it
    wrong.  Part (a) is **self-consistency** and is an algebraic tautology: ``chain()``
    produces ``v_j = v_i + g Dt + R_i dvbar``, so ``r^v = R_i^T(v_j - v_i - g Dt) - dvbar``
    reduces to ``dvbar - dvbar``.  Its measured 1.65e-13 is round-off, not accuracy, and
    it can only fail on a sign error that ``chain()`` does not mirror.  It is kept as that
    sign guard and labelled as nothing more.

    Part (b) is the measurement.  At the fixture's **ground-truth** states the residual
    does not vanish -- it *is* the O(dt) truncation of M2-C04, 2.760e-3, six orders above
    the 1e-9 the claim used to assert there.

    The ``|r^v_gt| == |dvbar - dv_exact|`` to 1e-12 assertion is NOT a test of the Eq. (9)
    sums (an earlier version of this docstring said it was): at ground truth ``chain()``
    gives ``R_i^T(v_j - v_i - g Dt) = dv_exact`` identically, so both sides contain
    ``factor.dvbar`` and any change to the sums moves both equally.  It tests only how the
    residual handles the state (``R_i^T``, gravity, ``Dt``), and it is sign-blind.  The part
    that tests the sums is the 2.760e-3 +-10% magnitude; ``test_c04`` (c)/(d) carry the
    signed checks.
    """
    self_states = B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, T.initial_state(),
                                           cls=B2.CayleyPreintegration)
    ground_truth = T.exact_states()
    worst_self = 0.0
    worst_ground_truth = 0.0
    for k, edge in enumerate(T.edges()):
        factor = B2.CayleyFactor.from_edge(edge, T.BIAS_TRUE)

        # -- (a) self-consistent states: a tautology, kept only as a sign guard ---------
        s_i, s_j = self_states[k], self_states[k + 1]
        r_v = factor.velocity_residual(s_i.R, s_i.v, s_j.v, T.G_W)
        r_p = factor.position_residual(s_i.R, s_i.p, s_j.p, s_i.v, T.G_W)
        worst_self = max(worst_self, float(np.abs(r_v).max()), float(np.abs(r_p).max()))

        # -- (b) the fixture's ACTUAL ground-truth states ------------------------------
        g_i, g_j = ground_truth[k], ground_truth[k + 1]
        r_v_gt = factor.velocity_residual(g_i.R, g_i.v, g_j.v, T.G_W)
        r_p_gt = factor.position_residual(g_i.R, g_i.p, g_j.p, g_i.v, T.G_W)
        worst_ground_truth = max(worst_ground_truth, float(np.abs(r_v_gt).max()),
                                 float(np.abs(r_p_gt).max()))

        # and it is exactly the Eq. (9) truncation, not something else that happens to
        # be the same size
        _, dv_ex, dp_ex = edge.exact_increment()
        assert _max_abs(np.abs(r_v_gt), np.abs(factor.dvbar - dv_ex)) <= 1e-12, \
            f"edge {k}: r^v at ground truth is not |dvbar - dv_exact|"
        assert _max_abs(np.abs(r_p_gt), np.abs(factor.dpbar - dp_ex)) <= 1e-12, \
            f"edge {k}: r^p at ground truth is not |dpbar - dp_exact|"

    assert worst_self <= 1e-9, \
        f"self-consistent residual {worst_self:.3e} (tautological; sign guard only)"
    assert 1e-4 <= worst_ground_truth <= 1e-2, \
        f"at ground truth this must be the O(dt) truncation, got {worst_ground_truth:.3e}"
    assert abs(worst_ground_truth - 2.760341e-3) <= 0.10 * 2.760341e-3, \
        f"ground-truth residual {worst_ground_truth:.6e}, expected 2.760e-3 +-10%"


@pytest.mark.claim("M2-F09")
def test_c10b_velocity_and_position_residuals_at_nonzero_bias():
    """Rai Eq. (21b)/(21c) evaluated at ``db != 0`` -- the ``A^{v.}``/``A^{p.}`` paths.

    ``test_c10`` passes ``db = None``, so flipping the sign of the bias terms (verifier
    mutant N04) survived the suite: the factor's whole bias model was exercised by nothing.

    The states here are what the residual is meant to explain: KF_i is the fixture's KF0
    (``R_0 != I``, so ``R_i^T`` is exercised) and KF_j is chained from a RE-INTEGRATION of
    the Eq. (9) sums at ``bbar + db``.  Then ``r^v = dv(bbar + db) - dvbar - A^v db``, the
    first-order Taylor remainder, so it must be O(||db||^2): the ratio between
    ``||db|| = 1e-2`` and ``1e-3`` lies in [50, 200] (measured 100.0 on both edges, both
    residuals) and the bias terms must cancel at least 99% of the bias-induced change
    (measured remainder at 1e-2: r^v 4.6e-4 / 4.9e-4, r^p 3.2e-4 / 3.4e-4 of it on edges
    0 / 1).  With either sign flipped the
    residual is ``2 A db``: ratio 10, remainder 2.0.  The direction has distinct gyro and
    accel blocks, so a ``db_g``/``db_a`` exchange is seen too.
    """
    direction = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    direction /= np.linalg.norm(direction)
    s_i = T.initial_state()
    for edge in T.edges():
        factor = B2.CayleyFactor.from_edge(edge, T.BIAS_TRUE)
        residual, uncorrected = {}, {}
        for mag in (1e-3, 1e-2):
            db = mag * direction
            moved = B1.integrate_edge(edge, T.BIAS_TRUE + db, cls=B2.CayleyPreintegration)
            s_j = T.chain(s_i, moved.dR, moved.dv, moved.dp, moved.dt_total)
            residual[mag] = (
                factor.velocity_residual(s_i.R, s_i.v, s_j.v, T.G_W, db),
                factor.position_residual(s_i.R, s_i.p, s_j.p, s_i.v, T.G_W, db))
            uncorrected[mag] = (
                factor.velocity_residual(s_i.R, s_i.v, s_j.v, T.G_W),
                factor.position_residual(s_i.R, s_i.p, s_j.p, s_i.v, T.G_W))
        for m, name in ((0, "r^v (21b)"), (1, "r^p (21c)")):
            small = np.abs(residual[1e-3][m]).max()
            large = np.abs(residual[1e-2][m]).max()
            ratio = large / small
            assert 50.0 <= ratio <= 200.0, (
                f"edge {edge.index} {name}: residual at db != 0 is not O(||db||^2), ratio "
                f"{ratio:.1f} ({small:.3e} -> {large:.3e}); ~10 means the bias terms enter "
                f"with the wrong sign or the wrong block")
            remainder = large / np.abs(uncorrected[1e-2][m]).max()
            assert remainder <= 1e-2, (
                f"edge {edge.index} {name}: the bias terms leave {remainder:.3e} of the "
                f"bias-induced change uncorrected")


@pytest.mark.claim("M2-C11")
def test_c11_affine_lift_is_algebraically_identical_to_eq19():
    """Rai Eq. (22b) vs Eq. (19) under ``xi = J_R (b_g - bbar_g)``.

    An exact identity, so this is a cheap and very sharp test of a correct transcription:
    a sign slip shows up immediately.

    This docstring used to add "or a swapped ``Y``/``Y'``".  That was false and has been
    struck.  Eq. (22b) combines them as ``-(1/2)(Y + Y')``; a sum is symmetric, so
    exchanging the two is **literally a no-op** -- no test can catch it, and claiming one
    does invites a reader to trust a guard that does not exist.  ``tests/test_mutation.py``
    asserts the no-op rather than pretending otherwise.
    """
    rng = np.random.default_rng(20250215)
    worst = 0.0
    for _ in range(2000):
        R_i = T.so3_exp(rng.uniform(-np.pi / 2, np.pi / 2, 3))
        R_j = T.so3_exp(rng.uniform(-np.pi / 2, np.pi / 2, 3))
        dRbar = T.so3_exp(rng.uniform(-0.5, 0.5, 3))
        J_R = rng.uniform(-1.0, 1.0, (3, 3))
        b_g = rng.uniform(-0.05, 0.05, 3)
        bbar_g = rng.uniform(-0.05, 0.05, 3)
        xi = J_R @ (b_g - bbar_g)
        worst = max(worst, _max_abs(B2.rotation_residual_eq19(R_i, R_j, dRbar, xi),
                                    B2.rotation_residual_eq22b(R_i, R_j, dRbar, J_R,
                                                               b_g, bbar_g)))
    assert worst <= 1e-13, f"Eq. (22b) vs Eq. (19) {worst:.3e}"


@pytest.mark.claim("M2-C12")
def test_c12_isotropic_precision_is_structurally_forced():
    """Rai Eq. (20)/(23).

    The forcing mechanism is SO(3)-invariance of the 2-norm under the ``R_i``
    premultiplication that removes rotation-state products from Eq. (21b)/(21c) -- not
    the SDP structure, which the paper does not claim.  The paper does acknowledge that
    the isotropic weight drops cross-correlations and directional weighting.
    """
    rng = np.random.default_rng(20250216)
    W = np.diag([1.0, 2.0, 5.0])
    worst_isotropic, best_weighted = 0.0, 0.0
    for _ in range(2000):
        R = T.so3_exp(rng.uniform(-np.pi, np.pi, 3))
        r = rng.standard_normal(3)
        worst_isotropic = max(worst_isotropic,
                              abs(B2.weighted_square(r) - B2.weighted_square(R @ r)))
        best_weighted = max(best_weighted,
                            abs(B2.weighted_square(r, W) - B2.weighted_square(R @ r, W)))
    assert worst_isotropic <= 1e-12, f"isotropic invariance {worst_isotropic:.3e}"
    # the isotropic half above is invariant under ANY function of ||r||, so it cannot see
    # a weighted_square that forgot to square (verification pass 3, P33).  Pin the value.
    r = np.array([1.0, -2.0, 3.0])
    assert B2.weighted_square(r) == 14.0, \
        f"weighted_square([1, -2, 3]) = {B2.weighted_square(r)!r}, expected 1 + 4 + 9 = 14"
    assert B2.weighted_square(r, W) == 1.0 + 8.0 + 45.0, \
        f"weighted_square([1, -2, 3], diag(1, 2, 5)) = {B2.weighted_square(r, W)!r}, expected 54"
    assert best_weighted >= 1.0, \
        f"a non-isotropic weight must break the identity, got {best_weighted:.3e}"

    Sigma = np.diag([0.3, 0.6, 0.9])
    assert abs(B2.isotropic_precision(Sigma) - 3.0 / 1.8) <= 1e-15


@pytest.mark.claim("M2-C13")
def test_c13_frozen_jacobians_with_an_exact_retraction():
    """The heart of Baseline 2: the *Jacobian* is frozen, the *retraction* is exact."""
    edge = T.edges()[0]
    factor = B2.CayleyFactor.from_edge(edge, T.BIAS_TRUE)

    # (a) the bias map really is the frozen AFFINE map xi = J_R (b_g - bbar_g), and the
    #     retraction really is driven by it rather than by re-integration.
    #
    #     The byte-equality check that used to live here ("J_R is bitwise unchanged after
    #     a predict at a perturbed bias") was unfalsifiable: no code path writes J_R after
    #     construction, so it could not fail for any input, and it left the claim's actual
    #     content -- the affine map -- exercised by nothing.  Dropping the bbar_g offset is
    #     a 10x error in xi and used to survive the whole suite.
    before = factor.J_R.copy()
    before_bytes = before.tobytes()
    R_i = T.R_0
    delta = np.array([2e-3, -1e-3, 5e-4])

    # the offset is there: xi vanishes EXACTLY at the linearization point
    assert np.array_equal(factor.xi(T.B_G), np.zeros(3)), \
        f"xi(bbar_g) must be exactly zero, got {factor.xi(T.B_G)} -- the bbar_g offset " \
        f"has been dropped from CayleyFactor.xi"
    # ... and the map is linear in the OFFSET, not in the bias
    assert _max_abs(factor.xi(T.B_G + delta), factor.J_R @ delta) <= 1e-15
    assert _max_abs(factor.xi(T.B_G - delta), -(factor.J_R @ delta)) <= 1e-15
    assert np.linalg.norm(factor.J_R @ T.B_G) >= 1e-3, \
        "the fixture's bbar_g must make the dropped offset detectable at all"

    # the retraction consumes exactly that xi, and is NOT a re-integration
    for scale, floor in ((1.0, None), (25.0, 1e-6)):
        b_g = T.B_G + scale * delta
        frozen = factor.predicted_rotation(R_i, b_g)
        assert _max_abs(frozen, R_i @ factor.dRbar @ B2.cay(factor.J_R @ (scale * delta))) \
            <= 1e-15, "predicted_rotation is not R_i dRbar Cay(J_R db_g)"
        relinearized = R_i @ B1.integrate_edge(
            edge, np.concatenate([b_g, T.B_A]), cls=B2.CayleyPreintegration).dR
        gap = _rot_err(frozen, relinearized)
        if floor is not None:
            assert gap >= floor, \
                f"frozen and re-linearized predictions agree to {gap:.3e} at |db_g|=" \
                f"{np.linalg.norm(scale * delta):.2e}: the factor is re-linearizing"

    assert factor.J_R.tobytes() == before_bytes, "J_R was mutated in passing"
    assert np.array_equal(factor.J_R, before)

    # (b) the retraction is kept exact: truncating Cay to (I + hat(xi)) is a different
    #     factor, by >= 1e-6 already at ||xi|| = 1e-2.
    rng = np.random.default_rng(20250217)
    worst = 0.0
    for _ in range(200):
        u = rng.standard_normal(3)
        xi = 1e-2 * u / np.linalg.norm(u)
        assert abs(np.linalg.norm(xi) - 1e-2) < 1e-12
        dRbar = factor.dRbar
        R_j = B2.rotation_from_eq16(R_i, dRbar, xi)
        exact = B2.rotation_residual_eq19(R_i, R_j, dRbar, xi)
        truncated = B2.rotation_residual_truncated(R_i, R_j, dRbar, xi)
        assert np.abs(exact).max() <= 1e-12
        worst = max(worst, _max_abs(exact, truncated))
    assert worst >= 1e-6, f"truncation difference only {worst:.3e}"
    # the lower bound above passes for ANY contrast at least that far from Eq. (19), even
    # one that is worse than the first-order truncation (verification pass 3, P29: the
    # contrast built on (I - hat(xi)) survived).  Pin the contrast itself: at
    # R_i = R_j = dRbar = I it is exactly -hat(xi), typed out here.
    xi = np.array([0.03, -0.02, 0.01])
    minus_hat_xi = -np.array([[0.0, -0.01, -0.02],
                              [0.01, 0.0, -0.03],
                              [0.02, 0.03, 0.0]])
    got = B2.rotation_residual_truncated(np.eye(3), np.eye(3), np.eye(3), xi)
    assert _max_abs(got, minus_hat_xi) <= 1e-17, \
        f"truncated contrast at the identity is not -hat(xi):\n{got}"


def _indep_hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _indep_cay(phi):
    """``Cay(phi)`` in its closed polynomial form ``I + 4/(4 + |phi|^2) (K + K^2/2)``,
    ``K = hat(phi)`` -- algebraically Eq. (3), but no linear solve, nothing from ``src``."""
    K = _indep_hat(phi)
    return np.eye(3) + (4.0 / (4.0 + float(phi @ phi))) * (K + 0.5 * (K @ K))


def _indep_cayley_increment(gyro, accel, dt, bias):
    """Baseline 2's Eq. (9) sums, retyped: ``dp += dv dt + dR a dt^2/2``, ``dv += dR a dt``,
    ``dR <- dR Cay((omega - b_g) dt)``."""
    R, v, p = np.eye(3), np.zeros(3), np.zeros(3)
    for omega, acc in zip(gyro, accel):
        w = omega - bias[0:3]
        a = acc - bias[3:6]
        p = p + v * dt + 0.5 * (R @ a) * dt**2
        v = v + (R @ a) * dt
        R = R @ _indep_cay(w * dt)
    return R, v, p


#: the reference's own FD step -- deliberately not src's FD_EPS (1e-6): truncation is
#: below 1e-11 relative here and round-off ~1e-14/eps ~ 1e-9 absolute
CAYLEY_REF_FD_EPS = 1e-5


def _indep_cayley_bias_jacobians(edge, bias):
    """``(J_R, A_vg, A_va, A_pg, A_pa)`` of the retyped CAYLEY integrator, by central
    differences; J_R in the right chart ``dR(b + db) ~ dR(b) Exp(J_R db_g)``."""
    gyro, accel, dt = edge.gyro_stream(), edge.accel_stream(), edge.dt
    R0 = _indep_cayley_increment(gyro, accel, dt, bias)[0]
    eps = CAYLEY_REF_FD_EPS
    J_R = np.zeros((3, 3))
    J_v = np.zeros((3, 6))
    J_p = np.zeros((3, 6))
    for m in range(6):
        e = np.zeros(6)
        e[m] = eps
        Rp, vp, pp = _indep_cayley_increment(gyro, accel, dt, bias + e)
        Rm, vm, pm = _indep_cayley_increment(gyro, accel, dt, bias - e)
        J_v[:, m] = (vp - vm) / (2.0 * eps)
        J_p[:, m] = (pp - pm) / (2.0 * eps)
        if m < 3:
            J_R[:, m] = (_indep_so3_log(R0.T @ Rp) - _indep_so3_log(R0.T @ Rm)) / (2.0 * eps)
    return J_R, J_v[:, 0:3], J_v[:, 3:6], J_p[:, 0:3], J_p[:, 3:6]


@pytest.mark.claim("M2-F11")
def test_c13b_cayley_factor_jacobians_are_the_cayley_integrators():
    """Rai Sec. IV-A: the frozen Jacobians are those of the CAYLEY retraction.

    ``CayleyFactor.from_edge`` taking ``J_R`` and ``A`` from the Forster (Exp) integrator
    instead survived the whole suite (verification pass 3, P35): the two differ by only
    ~2.3e-7 relative, O(dt^2).  Pinned two-sidedly against an independent finite
    difference of a retyped Cayley integrator (:func:`_indep_cayley_bias_jacobians`):

    measured, relative to max|block| (edge 0 / edge 1)

    =====  ====================  ====================  =========
    block  factor vs reference   Forster vs reference  bound
    =====  ====================  ====================  =========
    J_R    1.6e-10 / 3.0e-10     2.32e-7 / 3.59e-7     1e-8
    A_vg   2.9e-8 / 1.9e-8       2.32e-7 / 3.81e-7     1e-7
    A_pg   3.1e-8 / 2.0e-8       2.32e-7 / 3.81e-7     1e-7
    A_va   5.1e-9 / 1.4e-9       9.5e-9 / 2.0e-8       1e-7
    A_pa   4.3e-9 / 5.1e-9       8.8e-9 / 1.3e-8       1e-7
    =====  ====================  ====================  =========

    The A floor (~3e-8) is round-off in the factor's own 1e-6-step differences, not a
    Cayley/Exp effect; J_R is clean, and it alone separates the integrators by >20x on
    both sides of its bound.  A_va / A_pa cannot separate them (the gap is at the noise).
    """
    names = ("J_R", "A_vg", "A_va", "A_pg", "A_pa")
    bounds = {"J_R": 1e-8, "A_vg": 1e-7, "A_va": 1e-7, "A_pg": 1e-7, "A_pa": 1e-7}
    for edge in T.edges():
        factor = B2.CayleyFactor.from_edge(edge, T.BIAS_TRUE)
        ref = dict(zip(names, _indep_cayley_bias_jacobians(edge, T.BIAS_TRUE)))
        got = {"J_R": factor.J_R, "A_vg": factor.A_vg, "A_va": factor.A_va,
               "A_pg": factor.A_pg, "A_pa": factor.A_pa}
        rel = {k: float(np.abs(got[k] - ref[k]).max() / np.abs(ref[k]).max()) for k in names}
        bad = {k: f"{v:.3e}" for k, v in rel.items() if v > bounds[k]}
        assert not bad, (
            f"edge {edge.index}: CayleyFactor's frozen Jacobians are not the Cayley "
            f"integrator's {bad} (bounds {bounds})")

        # the other side: the check resolves the retraction at all.  Forster's (Exp)
        # J_R is >= 1e-7 from the Cayley reference, 10x outside the bound above.
        forster_J_R = B1.bias_jacobians_fd(edge, T.BIAS_TRUE)[0]
        gap = float(np.abs(forster_J_R - ref["J_R"]).max() / np.abs(ref["J_R"]).max())
        assert gap >= 1e-7, \
            f"edge {edge.index}: Exp and Cay bias Jacobians agree to {gap:.3e}"


def _indep_eq19(R_i, R_j, dRbar, xi):
    """Rai Eq. (19), retyped: ``(R_j - R_i dRbar) - 1/2 (R_j + R_i dRbar) hat(xi)``."""
    A = R_i @ dRbar
    return (R_j - A) - 0.5 * (R_j + A) @ _indep_hat(xi)


def _indep_eq22b(R_i, R_j, dRbar, J_R, b_g, bbar_g):
    """Rai Eq. (22b), retyped: ``R_j (I + G/2) - R_i dRbar (I - G/2) - 1/2 (Y + Y')`` with
    ``G = hat(J_R bbar_g)``, ``Y = R_i dRbar hat(J_R b_g)``, ``Y' = R_j hat(J_R b_g)``."""
    G = _indep_hat(J_R @ bbar_g)
    X = _indep_hat(J_R @ b_g)
    I3 = np.eye(3)
    return (R_j @ (I3 + 0.5 * G) - R_i @ dRbar @ (I3 - 0.5 * G)
            - 0.5 * (R_i @ dRbar @ X + R_j @ X))


def _indep_truncated(R_i, R_j, dRbar, xi):
    """The first-order contrast ``R_j - R_i dRbar (I + hat(xi))``, retyped."""
    return R_j - R_i @ dRbar @ (np.eye(3) + _indep_hat(xi))


def _indep_rot(axis_angle):
    """Rodrigues, retyped (no ``src``): ``I + sin(t) K + (1 - cos(t)) K^2``."""
    t = float(np.linalg.norm(axis_angle))
    K = _indep_hat(np.asarray(axis_angle) / t)
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


@pytest.mark.claim("M2-F12")
def test_c13c_cayley_factor_residual_methods_are_wired_to_eq19_and_eq22b():
    """The FACTOR-LEVEL residuals -- what the QCQP/SDP lifting will consume.

    Every Eq. (19)/(22b) check elsewhere calls the module-level functions, so a wiring slip
    inside ``CayleyFactor.rotation_residual`` / ``rotation_residual_affine`` /
    ``rotation_residual_truncated`` passed the whole suite (verification pass 4, M2-F12:
    Q25 accel offset in the affine lift, Q45 ``-xi``, Q46 ``dRbar^T``).  Each method is
    checked here against Eq. (19) / (22b) retyped from the paper, with ``xi`` formed in the
    test from the factor's frozen fields, on
      (a) self-consistent states ``R_j = R_i dRbar Cay(xi)`` -- Eq. (19) and (22b) vanish;
      (b) perturbed states with a nonzero bias offset -- the nonzero value must match.
    All rotations are non-identity and non-symmetric, so a transpose is visible, and the
    nominal gyro and accel biases differ, so an exchanged offset is visible.

    Measured worst (edge-1 / synthetic factor), against the 1e-13 bound: Eq. (19) method
    1.7e-16 / 1.7e-16 on-manifold, 1.4e-17 / 1.4e-17 perturbed; Eq. (22b) method
    3.1e-16 / 3.7e-16 and 2.2e-16 / 3.3e-16; truncated method <= 1.1e-16.  Smallest
    perturbed Eq. (19) residual 2.4e-2 / 2.1e-2.  Q25 / Q45 / Q46 give 4.7e-2 / 6.6e-2 /
    5.2e-1 on-manifold.
    """
    rng = np.random.default_rng(20250219)
    edge_factor = B2.CayleyFactor.from_edge(T.edges()[1], T.BIAS_TRUE)
    synthetic = B2.CayleyFactor(
        bbar=np.array([0.012, -0.007, 0.004, 0.31, -0.22, 0.17]),
        dRbar=_indep_rot(np.array([0.4, -0.3, 0.25])),
        dvbar=np.zeros(3), dpbar=np.zeros(3), Dt=1.0,
        J_R=np.array([[-0.9, 0.2, 0.1], [0.05, -1.1, 0.3], [-0.2, 0.15, -0.8]]),
        A_vg=np.zeros((3, 3)), A_va=np.zeros((3, 3)),
        A_pg=np.zeros((3, 3)), A_pa=np.zeros((3, 3)))

    for name, f in (("edge-1 factor", edge_factor), ("synthetic factor", synthetic)):
        dRbar, J_R = np.asarray(f.dRbar), np.asarray(f.J_R)
        bbar_g, bbar_a = np.asarray(f.bbar[0:3]), np.asarray(f.bbar[3:6])
        assert np.abs(dRbar - dRbar.T).max() >= 1e-2, f"{name}: dRbar is ~symmetric"
        assert np.linalg.norm(J_R @ (bbar_a - bbar_g)) >= 1e-3, \
            f"{name}: an exchanged bias offset would be invisible"
        worst_on = {"eq19": 0.0, "eq22b": 0.0, "trunc": 0.0}
        worst_off = {"eq19": 0.0, "eq22b": 0.0, "trunc": 0.0}
        smallest_off = np.inf
        for _ in range(50):
            R_i = _indep_rot(rng.uniform(-1.2, 1.2, 3))
            assert np.abs(R_i - R_i.T).max() >= 1e-3
            db_g = rng.uniform(-0.05, 0.05, 3)
            assert np.linalg.norm(db_g) >= 1e-3
            b_g = bbar_g + db_g
            xi = J_R @ db_g                      # Rai: xi = J_R (b_g - bbar_g), frozen J_R

            # (a) self-consistent: R_j = R_i dRbar Cay(xi), Cay in closed polynomial form
            R_j = R_i @ dRbar @ _indep_cay(xi)
            assert np.abs(_indep_eq19(R_i, R_j, dRbar, xi)).max() <= 1e-13
            worst_on["eq19"] = max(worst_on["eq19"], float(np.abs(
                f.rotation_residual(R_i, R_j, b_g)).max()))
            worst_on["eq22b"] = max(worst_on["eq22b"], float(np.abs(
                f.rotation_residual_affine(R_i, R_j, b_g)).max()))
            worst_on["trunc"] = max(worst_on["trunc"], _max_abs(
                f.rotation_residual_truncated(R_i, R_j, b_g),
                _indep_truncated(R_i, R_j, dRbar, xi)))

            # (b) perturbed: R_j off Eq. (16), the expected nonzero residual retyped
            R_j_bad = R_j @ _indep_rot(rng.uniform(-0.1, 0.1, 3))
            ref19 = _indep_eq19(R_i, R_j_bad, dRbar, xi)
            ref22 = _indep_eq22b(R_i, R_j_bad, dRbar, J_R, b_g, bbar_g)
            smallest_off = min(smallest_off, float(np.abs(ref19).max()))
            worst_off["eq19"] = max(worst_off["eq19"], _max_abs(
                f.rotation_residual(R_i, R_j_bad, b_g), ref19))
            worst_off["eq22b"] = max(worst_off["eq22b"], _max_abs(
                f.rotation_residual_affine(R_i, R_j_bad, b_g), ref22))
            worst_off["trunc"] = max(worst_off["trunc"], _max_abs(
                f.rotation_residual_truncated(R_i, R_j_bad, b_g),
                _indep_truncated(R_i, R_j_bad, dRbar, xi)))

        assert smallest_off >= 1e-3, f"{name}: perturbation too small ({smallest_off:.3e})"
        assert worst_on["eq19"] <= 1e-13, (
            f"{name}: CayleyFactor.rotation_residual is {worst_on['eq19']:.3e} on a "
            f"self-consistent state -- it is not Eq. (19) at xi = J_R (b_g - bbar_g)")
        assert worst_off["eq19"] <= 1e-13, (
            f"{name}: CayleyFactor.rotation_residual differs from the retyped Eq. (19) by "
            f"{worst_off['eq19']:.3e} at a perturbed state")
        assert worst_on["eq22b"] <= 1e-13, (
            f"{name}: CayleyFactor.rotation_residual_affine is {worst_on['eq22b']:.3e} on a "
            f"self-consistent state -- it is not Eq. (22b) with offset bbar_g")
        assert worst_off["eq22b"] <= 1e-13, (
            f"{name}: CayleyFactor.rotation_residual_affine differs from the retyped "
            f"Eq. (22b) by {worst_off['eq22b']:.3e} at a perturbed state")
        assert worst_on["trunc"] <= 1e-13 and worst_off["trunc"] <= 1e-13, (
            f"{name}: CayleyFactor.rotation_residual_truncated differs from the retyped "
            f"truncation by {max(worst_on['trunc'], worst_off['trunc']):.3e}")


# ======================================================================================
# Baseline 3 -- GTSAM 4.3.0 Galilean
# ======================================================================================


@pytest.mark.claim("M2-C14")
def test_c14_gal3_matrix_and_constructor_orders_disagree():
    """``matrix()`` is ``[R | v | p]``; the constructor is ``Gal3(R, r, v, t)``.

    Velocity before position in the matrix, position before velocity in the constructor.
    Prefer the named accessors over positional unpacking everywhere.
    """
    R = Rot3.Expmap(np.array([0.10, -0.20, 0.30]))
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([10.0, 20.0, 30.0])
    t = 7.0
    Xg = Gal3(R, position, velocity, t)
    M = Xg.matrix()

    assert M.shape == (5, 5)
    assert np.allclose(M[0:3, 0:3], R.matrix(), atol=1e-15)
    assert np.allclose(M[0:3, 3], velocity, atol=1e-15), "column 3 must be VELOCITY"
    assert np.allclose(M[0:3, 4], position, atol=1e-15), "column 4 must be POSITION"
    assert M[3, 4] == t
    assert np.allclose(M[3, 0:3], 0.0) and M[3, 3] == 1.0
    assert np.allclose(M[4, 0:4], 0.0) and M[4, 4] == 1.0

    # the accessors return the 2nd/3rd constructor arguments, NOT the swapped reading
    assert np.allclose(Xg.position(), position, atol=1e-15)
    assert np.allclose(Xg.velocity(), velocity, atol=1e-15)
    assert not np.allclose(Xg.position(), velocity)
    assert np.allclose(Xg.translation(), Xg.position(), atol=1e-15)
    assert Xg.time() == t

    round_trip = Gal3(M)
    assert np.allclose(round_trip.matrix(), M, atol=1e-15)
    assert np.allclose(round_trip.position(), position, atol=1e-15)
    assert np.allclose(round_trip.velocity(), velocity, atol=1e-15)
    assert round_trip.time() == t


@pytest.mark.claim("M2-C15")
def test_c15_gal3_tangent_order_and_the_navstate_remap():
    """Tangent is ``(omega, nu, rho, alpha)`` = (rotation, VELOCITY, POSITION, time)."""
    blocks = {
        0: [(1, 2), (2, 1)], 1: [(0, 2), (2, 0)], 2: [(0, 1), (1, 0)],   # rotation
        3: [(0, 3)], 4: [(1, 3)], 5: [(2, 3)],                            # velocity column
        6: [(0, 4)], 7: [(1, 4)], 8: [(2, 4)],                            # position column
        9: [(3, 4)],                                                       # time
    }
    for k in range(10):
        e_k = np.zeros(10)
        e_k[k] = 1.0
        H = Gal3.Hat(e_k)
        nonzero = {tuple(ij) for ij in np.argwhere(np.abs(H) > 1e-15)}
        assert nonzero == set(blocks[k]), f"Hat(e_{k}) occupies {sorted(nonzero)}"

        Xg = Gal3.Expmap(1e-6 * e_k)
        moved_v = np.abs(Xg.velocity()).max() > 1e-9
        moved_p = np.abs(Xg.position()).max() > 1e-9
        moved_t = abs(Xg.time()) > 1e-9
        if 3 <= k < 6:
            assert moved_v and not moved_p and not moved_t
        elif 6 <= k < 9:
            assert moved_p and not moved_v and not moved_t
        elif k == 9:
            assert moved_t and not moved_v and not moved_p

    xi = np.array([0.03, -0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert _max_abs(Gal3.Expmap(xi).matrix(), expm(Gal3.Hat(xi))) <= 1e-14

    # NavState block order is (rot, pos, vel), so the remap is {0, 6, 3}
    assert B3.GAL3_BLOCK_FOR_NAVSTATE == (0, 6, 3)
    remapped = B3.gal3_tangent_to_navstate_order(np.arange(10.0))
    assert np.array_equal(remapped, [0, 1, 2, 6, 7, 8, 3, 4, 5])


@pytest.mark.claim("M2-C16")
def test_c16_remap_is_a_permutation_not_a_chart_identification():
    """Getting the ``{0, 6, 3}`` remap right and then assuming the charts agree is the
    obvious next mistake.  ``NavState``'s *retraction* is not the Gal(3) exponential.

    Its *group* is, though, and that is the sharper statement: the remapped Gal(3)
    logarithm equals ``NavState.Logmap`` **exactly**, so the only thing that disagrees is
    ``localCoordinates``.  Both halves are asserted, because the exact-equality half is
    what pins which object the 0.1076 is a gap against.

    The bound is near-tight: 0.107628 against ``>= 0.1``, a 7.6% margin.  The previously
    recorded 0.763 is not reproducible here, so it is asserted away rather than left to
    suggest 660% of headroom that does not exist.
    """
    states = T.exact_states()
    s_i, s_j = states[0], states[2]

    X_i = B3.gal3_lift(s_i, 0.0)
    X_j = B3.gal3_lift(s_j, 0.0)
    assert X_i.time() == 0.0 and X_j.time() == 0.0

    gal_log = np.asarray(Gal3.Logmap(X_i.inverse().compose(X_j)))
    remapped = B3.gal3_tangent_to_navstate_order(gal_log)

    ns_i = NavState(Rot3(s_i.R), s_i.p, s_i.v)
    ns_j = NavState(Rot3(s_j.R), s_j.p, s_j.v)
    local = np.asarray(ns_i.localCoordinates(ns_j))

    # the rotation block DOES agree -- the disagreement is in the translation/velocity
    assert _max_abs(remapped[0:3], local[0:3]) <= 1e-12
    gap = _max_abs(remapped, local)
    assert gap >= 0.1, f"charts unexpectedly close: {gap:.3e}"
    assert abs(gap - 0.107628) <= 1e-5, \
        f"recorded measurement is 0.107628, got {gap:.6f}"

    # the GROUP logarithms agree exactly: NavState's group IS Gal(3) at t = 0, for all
    # three keyframe pairs.  Nothing here is a tolerance -- it is bitwise zero.
    for i, j in ((0, 1), (1, 2), (0, 2)):
        a, b = states[i], states[j]
        A, Bg = B3.gal3_lift(a, 0.0), B3.gal3_lift(b, 0.0)
        remap = B3.gal3_tangent_to_navstate_order(
            np.asarray(Gal3.Logmap(A.inverse().compose(Bg))))
        na = NavState(Rot3(a.R), a.p, a.v)
        nb = NavState(Rot3(b.R), b.p, b.v)
        group_log = np.asarray(NavState.Logmap(na.inverse().compose(nb)))
        assert _max_abs(remap, group_log) == 0.0, \
            f"KF{i}->KF{j}: group logs differ by {_max_abs(remap, group_log):.3e}"


@pytest.mark.claim("M2-C17")
def test_c17_constant_bias_is_accel_first():
    """``ConstantBias(a, g).vector() == [b_a; b_g]``, opposite the project's ``[b_g; b_a]``.
    Every crossing of the GTSAM boundary swaps the two 3-blocks -- Jacobian COLUMNS too."""
    a = np.array([1.0, 2.0, 3.0])
    g = np.array([4.0, 5.0, 6.0])
    cb = ConstantBias(a, g)
    assert np.array_equal(cb.accelerometer(), a)
    assert np.array_equal(cb.gyroscope(), g)
    assert np.array_equal(cb.vector(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    project = np.concatenate([T.B_G, T.B_A])        # [b_g; b_a]
    crossed = B3.to_gtsam_bias(project)
    assert np.array_equal(crossed.vector(), np.concatenate([T.B_A, T.B_G]))
    assert np.array_equal(B3.from_gtsam_bias(crossed), project)

    # a single application of the swap is NOT the identity; two applications are
    assert not np.array_equal(B3.swap_bias_blocks(project), project)
    assert np.array_equal(B3.swap_bias_blocks(B3.swap_bias_blocks(project)), project)

    pim = B3.GalileanPreintegration(project)
    assert np.array_equal(pim.pim.biasHatVector(), np.concatenate([T.B_A, T.B_G]))


@pytest.mark.claim("M2-C18")
def test_c18_gal3_endpoint_relation_holds_only_with_the_t_i_term():
    """``X_j = W_ij(t_i) X_i U_ij`` with ``W_ij(t_i) = Gal3(I, 0.5 g Dt^2 - g Dt (t_i + Dt),
    g Dt, 0)``.

    Pre-flight F2's ``T(-Dt) H' X_i U_ij`` is correct only at ``t_i = 0`` and is wrong by
    exactly ``|g| Dt t_i`` otherwise.  Benign under GTSAM (``NavState`` carries no clock),
    a live hazard wherever Gal(3) states are decision variables.
    """
    state_i = T.initial_state()
    cases = []
    for edge in T.edges():
        pim = B3.integrate_edge(edge, T.BIAS_TRUE)
        cases.append((pim.increment_gal3(), pim.increment(), pim.dt_total))
    # also the composed two-edge span, Dt = 1.0.  NB gal3_components returns
    # (R, p, v, t) -- position SECOND; reading it as (R, v, p) is exactly the M2-C14
    # footgun, so the unpack is spelled out rather than positional-by-habit.
    U_full = cases[0][0].compose(cases[1][0])
    dR_f, dp_f, dv_f, _ = B3.gal3_components(U_full)
    cases.append((U_full, (dR_f, dv_f, dp_f), 2 * T.T_EDGE))

    for U, (dR, dv, dp), Dt in cases:
        reference = T.chain(state_i, dR, dv, dp, Dt)
        for t_i in (0.0, 3.7):
            X_i = B3.gal3_lift(state_i, t_i)

            R_w, p_w, v_w, t_w = B3.gal3_components(
                B3.endpoint_W_form(X_i, U, Dt, t_i))
            err_w = max(_max_abs(R_w, reference.R), _max_abs(v_w, reference.v),
                        _max_abs(p_w, reference.p))
            assert err_w <= 1e-12, f"W-form at t_i={t_i}, Dt={Dt}: {err_w:.3e}"
            assert abs(t_w - (t_i + Dt)) <= 1e-12, "W-form must advance the clock"

            R_f, p_f, v_f, t_f = B3.gal3_components(B3.endpoint_F2_form(X_i, U, Dt))
            err_f = max(_max_abs(R_f, reference.R), _max_abs(v_f, reference.v),
                        _max_abs(p_f, reference.p))
            assert abs(t_f - t_i) <= 1e-12, "F2-form leaves the clock at t_i"
            if t_i == 0.0:
                assert err_f <= 1e-12, f"F2-form should be exact at t_i=0: {err_f:.3e}"
            else:
                expected = float(np.linalg.norm(T.G_W)) * Dt * t_i
                assert abs(err_f - expected) <= 1e-6, \
                    f"F2-form error {err_f:.6f}, expected |g| Dt t_i = {expected:.6f}"

            R_s, p_s, v_s, _ = B3.gal3_components(B3.endpoint_F2_swapped(X_i, U, Dt))
            err_s = max(_max_abs(R_s, reference.R), _max_abs(v_s, reference.v),
                        _max_abs(p_s, reference.p))
            assert err_s >= 1e-3, f"swapped order should be wrong, got {err_s:.3e}"


@pytest.mark.claim("M2-C19")
def test_c19_galilean_increment_is_exact_zoh_and_right_accumulating():
    """(a) exact ZOH against the closed form, where TangentPreintegration is not;
    (b) ``U_{k+1} = U_k V_k``, tested on a TIME-VARYING stream.

    Part (b) must not use the fixture's own per-edge constant inputs: constant-input
    increments commute, so a constant stream passes under either accumulation order and
    the test would have no teeth at all.
    """
    # -- (a) -----------------------------------------------------------------------------
    params = B3.make_params()
    for edge in T.edges():
        dR, dv, dp = edge.exact_increment()

        galilean = B3.integrate_edge(edge, T.BIAS_TRUE)
        gal_err = max(_max_abs(galilean.dR, dR), _max_abs(galilean.dv, dv),
                      _max_abs(galilean.dp, dp))
        assert gal_err <= 1e-12, f"Galilean is not exact ZOH: {gal_err:.3e}"

        tangent = gtsam.PreintegratedImuMeasurements(
            params, B3.to_gtsam_bias(T.BIAS_TRUE))
        for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
            tangent.integrateMeasurement(acc, omega, edge.dt)
        tangent_dv_err = _max_abs(np.asarray(tangent.deltaVij()), dv)
        assert tangent_dv_err >= 1e-4, \
            f"TangentPreintegration should NOT be exact here: {tangent_dv_err:.3e}"

    # -- (b) -----------------------------------------------------------------------------
    gyro, accel, dt = T.time_varying_stream()
    n = T.TIME_VARYING_STEPS
    assert not np.allclose(gyro[0], gyro[-1]), "part (b) needs a time-varying stream"

    reference = []
    running = B3.GalileanPreintegration(T.BIAS_TRUE)
    for k in range(n):
        running.integrate_measurement(gyro[k], accel[k], dt)
        reference.append(running.increment_gal3())

    U_right = Gal3.Identity()
    U_left = Gal3.Identity()
    right_err = left_err = 0.0
    for k in range(n):
        V_k = B3.single_step(gyro[k], accel[k], dt, T.BIAS_TRUE)
        U_right = U_right.compose(V_k)
        U_left = V_k.compose(U_left)
        right_err = max(right_err, _max_abs(U_right.matrix(), reference[k].matrix()))
        left_err = max(left_err, _max_abs(U_left.matrix(), reference[k].matrix()))

    assert right_err <= 1e-14, f"U_k V_k should reproduce the backend: {right_err:.3e}"
    assert left_err >= 1e-5, f"V_k U_k should deviate, got {left_err:.3e}"


@pytest.mark.claim("M2-C20")
def test_c20_galilean_bias_correction_is_right_composed_in_gal3():
    """``U(db) = Uhat . Expmap_Gal3(J db)`` with ``J`` frozen: first-order Jacobian,
    **exact** retraction.  The component-wise form with the SAME ``J`` is a different and
    worse model -- it drops the ``dR``-rotation of the velocity/position tangent blocks.
    """
    rng = np.random.default_rng(20250218)
    directions = rng.standard_normal((30, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    for edge in T.edges():
        # zero gravity + identity NavState => predict() IS the corrected increment
        pim = B3.integrate_edge(edge, T.BIAS_TRUE, g_W=np.zeros(3))

        def increment_at(db, _pim=pim):
            return B3.corrected_increment_gal3(_pim, db)

        U_hat = increment_at(np.zeros(6))

        # -- the bias boundary, against an INDEPENDENT reference ---------------------------
        # Everything after this block is self-referential: J is estimated about U_hat and
        # both models are scored against U(db) from the same adapter.  An adapter that
        # skips the [b_g; b_a] <-> (accel, gyro) swap (mutant M07) integrates the edge at
        # a wrong nominal bias AND perturbs in a wrong order, and every GTSAM-against-GTSAM
        # comparison cancels it.  So pin the GTSAM side to the 5x5-expm closed form:
        # (i) U_hat at bbar = BIAS_TRUE must be the bias-free increment, to Tier 0; and
        # (ii) a perturbation whose accel and gyro blocks are DISTINCT (and of different
        # size) must shift the accel input by db[0:3] and the gyro input by db[3:6].
        # GTSAM's predict() applies a first-order correction, not a re-integration, so (ii)
        # holds to O(||db||^2) -- measured 2.26e-6 / 3.29e-6 on the two edges, quartering
        # exactly (4.00) when db is halved -- while the block-swapped reading is off by
        # 4.3e-2 / 4.5e-2.  The bound 1e-5 sits between them with ~3x / ~4000x margin.
        def reference(db_gtsam, _edge=edge):
            db_a, db_g = np.asarray(db_gtsam[0:3]), np.asarray(db_gtsam[3:6])
            return _zoh_increment_expm(_edge.omega_body - db_g, _edge.f_body - db_a,
                                       _edge.duration)

        def boundary_err(db_gtsam, ref_db=None):
            R_, p_, v_, _ = B3.gal3_components(increment_at(db_gtsam))
            ref = reference(db_gtsam if ref_db is None else ref_db)
            return max(_max_abs(R_, ref[0]), _max_abs(v_, ref[1]), _max_abs(p_, ref[2]))

        anchor_err = boundary_err(np.zeros(6))
        assert anchor_err <= TIER0, \
            f"edge {edge.index}: U_hat is not the bias-free increment: {anchor_err:.3e}"

        db_probe = np.array([3.0e-2, -2.0e-2, 1.0e-2,       # accel block (GTSAM cols 0:3)
                             -4.0e-3, 2.0e-3, 5.0e-3])      # gyro block  (GTSAM cols 3:6)
        probe_err = boundary_err(db_probe)
        assert probe_err <= 1e-5, (
            f"edge {edge.index}: U(db) does not apply db in (accel, gyro) column order: "
            f"{probe_err:.3e}")
        quarter = probe_err / boundary_err(db_probe / 2.0)
        assert 3.5 <= quarter <= 4.5, \
            f"edge {edge.index}: boundary error is not O(||db||^2): ratio {quarter:.3f}"
        # and the probe really discriminates: the block-swapped reading is far away
        swapped_gap = boundary_err(db_probe, ref_db=np.r_[db_probe[3:6], db_probe[0:3]])
        assert swapped_gap >= 1e-2, \
            f"edge {edge.index}: probe cannot see a bias block swap ({swapped_gap:.3e})"

        J = B3.bias_jacobian_fd(increment_at, eps=B3.J_FD_EPS)
        assert J.shape == (10, 6)
        # columns are GTSAM's (accel, gyro); rotation cannot see accel bias, and the
        # clock cannot see bias at all
        assert np.abs(J[0:3, 0:3]).max() <= 1e-10, "J[0:3, 0:3] must vanish"
        assert np.abs(J[9, :]).max() <= 1e-10, "J[9, :] must vanish"

        # The claim asserts the component-wise model "is wrong by AT LEAST 1e-2", i.e. a
        # statement about the supremum over directions.  30 random directions in 6-D
        # undersample that supremum badly: the true sup here is 1.4e-2 (edge 0) and
        # 1.6e-2 (edge 1), while a 30-sample max lands around 0.9e-2 -- below the stated
        # bound for sampling reasons alone.  The leading model difference is the linear
        # map db -> [(dR - I) J_v ; (dR - I) J_p] db, so its top right-singular vector is
        # added to the direction set.  This measures the sup, it does not relax it.
        R_hat, _, _, _ = B3.gal3_components(U_hat)
        difference_map = np.vstack([(R_hat - np.eye(3)) @ J[3:6, :],
                                    (R_hat - np.eye(3)) @ J[6:9, :]])
        worst_direction = np.linalg.svd(difference_map)[2][0]
        probe = np.vstack([directions, worst_direction, -worst_direction])

        magnitudes = (1e-4, 1e-3, 1e-2, 5e-2)
        right, component = {}, {}
        for mag in magnitudes:
            r_worst = c_worst = 0.0
            for d in probe:
                db = mag * d
                U_true = increment_at(db)
                r_worst = max(r_worst, B3.gal3_distance(
                    U_true, B3.right_composition_model(U_hat, J, db)))
                c_worst = max(c_worst, B3.gal3_distance(
                    U_true, B3.component_wise_model(U_hat, J, db)))
            right[mag], component[mag] = r_worst, c_worst

        for mag in magnitudes:
            assert right[mag] <= 1e-8, \
                f"edge {edge.index}: right composition at |db|={mag}: {right[mag]:.3e}"
        assert component[5e-2] >= 1e-2, \
            f"edge {edge.index}: component-wise should be wrong: {component[5e-2]:.3e}"

        # The lower bound above cannot fail on a contrast model that is buggy in the
        # "worse" direction -- exchanging its p and v tangent blocks (verifier mutant N13)
        # passed.  So pin WHAT the contrast's error is, to first order, from its definition:
        # with xi = J db, U_true = Uhat Exp(xi) and U_comp = (R Exp(xi_R), p + xi_rho,
        # v + xi_nu), the Gal(3) log of U_true^-1 U_comp has velocity block
        # (R^T - I) xi_nu and position block (R^T - I) xi_rho, rotation and time 0, up to
        # O(||db||^2).  Measured relative deviation 5.8e-4 / 2.4e-4 at ||db|| = 1e-4,
        # growing x10 per decade (it is the second-order remainder); bound 1e-2.  The p/v
        # exchanged contrast deviates by 23 / 14.
        for d in directions:
            db = 1e-4 * d
            xi = J @ db
            predicted = np.zeros(10)
            predicted[3:6] = (R_hat.T - np.eye(3)) @ xi[3:6]
            predicted[6:9] = (R_hat.T - np.eye(3)) @ xi[6:9]
            got = np.asarray(Gal3.Logmap(
                increment_at(db).inverse().compose(B3.component_wise_model(U_hat, J, db))))
            deviation = _max_abs(got, predicted) / np.abs(predicted).max()
            assert deviation <= 1e-2, (
                f"edge {edge.index}: the component-wise contrast's error is not the dropped "
                f"dR-rotation of the (v, p) tangent blocks (relative {deviation:.3e})")

        # the right-composition residual scales LINEARLY with ||db||, i.e. it is FD noise
        # in J rather than model error
        span = right[5e-2] / right[1e-4]
        assert 100.0 <= span <= 2000.0, f"residual scaling {span:.1f} (expected ~500)"


@pytest.mark.claim("M2-C21")
def test_c21_gal3_bias_jacobian_recursion():
    """``J_{k+1} = Ad_{V_k^-1} J_k + C_k`` with ``J_i = 0``.

    ``C_k`` is the derivative with respect to the **bias**.  The ``-C_k`` form in the
    literature differentiates with respect to the **measurement**; the two differ by the
    sign hidden in ``(a_k - b_a)``.  Fixing the convention before comparing signs is the
    whole content of the sign question.
    """
    gyro, accel, dt = T.time_varying_stream()
    n = T.TIME_VARYING_STEPS
    recursion = B3.bias_jacobian_recursion(gyro, accel, dt, T.BIAS_TRUE, n)

    worst_relative = 0.0
    for k in range(n):
        def reintegrate(db, _k=k):
            return B3.integrate_stream(gyro, accel, dt,
                                       T.BIAS_TRUE + B3.swap_bias_blocks(db),
                                       n_steps=_k + 1).increment_gal3()

        J_fd = B3.bias_jacobian_fd(reintegrate, eps=B3.C_FD_EPS)
        scale = np.abs(J_fd).max()
        assert scale > 0.0
        worst_relative = max(worst_relative, _max_abs(recursion[k], J_fd) / scale)

    assert worst_relative <= 1e-6, f"recursion vs re-integration {worst_relative:.3e}"


# ======================================================================================
# Cross-baseline agreement -- the three tiers
# ======================================================================================


def _kf2_states():
    """KF2 from every baseline, at the fixture's own discretization."""
    s0 = T.initial_state()
    return {
        "B1": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0)[2],
        "B2-raw": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0,
                                           cls=B2.CayleyPreintegration,
                                           variant=B2.VARIANT_RAW)[2],
        "B2-reparam": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0,
                                               cls=B2.CayleyPreintegration,
                                               variant=B2.VARIANT_REPARAM)[2],
        "B3": B3.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0)[2],
        "exact": T.exact_states()[2],
    }


@pytest.mark.claim("M2-C22")
def test_c22_tier1_same_discretization():
    """TIER 1, ``1e-6``, in ``v``, ``p`` AND ``R``.

    Baselines 1 and 2 share Rai Eq. (9) and differ only in the rotation retraction, so
    their gap is pure retraction, not discretization.  The reading is the raw one, which
    Rai Sec. IV-A fixes (see ``src/preintegration_cayley.py``'s module docstring); there
    is no second admissible reading to exercise.  The reparameterized variant is run here
    only to *exhibit* that it collapses onto Baseline 1, which is exactly why it cannot
    be the spec.

    This bound is an upper bound and nothing more.  The content of the gap -- that it is
    3.0e-8 and that it is the Cayley deficit -- lives in
    ``test_c22b_baseline2_retraction_really_is_cayley``, which is where a maintainer
    would notice if the retraction were quietly replaced.
    """
    states = _kf2_states()
    b1 = states["B1"]
    raw = states["B2-raw"]
    gaps = {"v": _max_abs(raw.v, b1.v), "p": _max_abs(raw.p, b1.p),
            "R": _max_abs(raw.R, b1.R)}
    failures = [f"{k}: {gaps[k]:.3e} > {TIER1:.0e}"
                for k in ("v", "p", "R") if gaps[k] > TIER1]
    assert not failures, f"Tier-1 agreement (B2-raw vs B1): {failures}; measured {gaps}"
    assert max(gaps.values()) >= TIER1 / 100.0, (
        f"B2-raw agrees with B1 to {max(gaps.values()):.3e} -- that is Exp's answer, not "
        f"Cay's.  See test_c22b.")

    # the contrast: the Eq. (3) reparameterization IS Exp, so it collapses onto B1
    reparam = states["B2-reparam"]
    collapse = max(_max_abs(reparam.v, b1.v), _max_abs(reparam.p, b1.p),
                   _max_abs(reparam.R, b1.R))
    assert collapse <= 1e-12, (
        f"the reparameterized variant should be numerically indistinguishable from "
        f"Baseline 1, got {collapse:.3e}")


@pytest.mark.claim("M2-C22")
@pytest.mark.claim("M2-F01")
def test_c22b_baseline2_retraction_really_is_cayley():
    """The one property that makes Baseline 2 a *distinct* baseline, asserted positively.

    Before this test existed, replacing ``CayleyPreintegration.retract`` with ``Exp``
    survived the whole suite -- and turned it green, because the only three tests that
    could see the difference were the three that were failing.  A maintainer clearing a
    red suite would have deleted Baseline 2's reason to exist and been rewarded.  So the
    deficit is asserted as the measurement it is, from four independent directions:

    (a) the retraction itself: ``Cay(theta a)`` rotates by ``2 arctan(theta/2)``;
    (b) per edge, the gap against ``Exp`` equals ``||omega||^3 T dt^2 / 12`` -- the
        per-step loss ``theta^3/12`` times the step count, exact here because the
        fixture's body rate is constant per edge so the per-step deficits share an axis
        and add;
    (c) at KF2 the accumulated gap is 3.0e-8, two-sided;
    (d) it is ``O(dt^2)``: the ratio is 0.25 per halving, not 0.5 and not 1.

    Substituting ``Exp`` for ``Cay`` fails (a), (b), (c) and (d).
    """
    # -- (a) the retraction, directly ---------------------------------------------------
    pim = B2.CayleyPreintegration(bias_hat=T.BIAS_TRUE, variant=B2.VARIANT_RAW)
    for theta in (1e-3, 1e-2, 0.1, 0.5, 1.0):
        axis = np.array([0.3, -0.5, 0.8])
        axis = axis / np.linalg.norm(axis)
        R = pim.retract(theta * axis)
        angle = float(np.linalg.norm(T.so3_log(R)))
        assert abs(angle - 2.0 * np.arctan(theta / 2.0)) <= 1e-13, (
            f"retract({theta}) rotates by {angle:.12f}; Cay gives "
            f"{2.0 * np.arctan(theta / 2.0):.12f}, Exp gives {theta:.12f}")
        # it under-rotates, and for small theta the deficit IS theta^3/12
        deficit = theta - angle
        assert deficit > 0.0, f"retract({theta}) does not under-rotate -- it is not Cay"
        if theta <= 0.1:
            assert abs(deficit / (theta**3 / 12.0) - 1.0) <= 1e-2, (
                f"retract({theta}) deficit {deficit:.6e} is not theta^3/12 = "
                f"{theta**3 / 12.0:.6e}")
        else:
            assert deficit >= 0.5 * theta**3 / 12.0, \
                f"retract({theta}) is indistinguishable from Exp -- it is not Cay"

    # -- (b) the per-edge deficit is ||omega||^3 T dt^2 / 12 ----------------------------
    for edge in T.edges():
        forster = B1.integrate_edge(edge, T.BIAS_TRUE)
        cayley = B1.integrate_edge(edge, T.BIAS_TRUE, cls=B2.CayleyPreintegration,
                                   variant=B2.VARIANT_RAW)
        deficit = T.so3_log(forster.dR.T @ cayley.dR)
        rate = float(np.linalg.norm(edge.omega_body))
        predicted = rate**3 * edge.duration * edge.dt**2 / 12.0
        measured = float(np.linalg.norm(deficit))
        assert abs(measured - predicted) <= 1e-3 * predicted, (
            f"edge {edge.index}: Cayley deficit {measured:.6e} vs predicted "
            f"||omega||^3 T dt^2/12 = {predicted:.6e}")
        # and it is a pure LAG about the rotation axis: Cay under-rotates
        cosine = float(deficit @ edge.omega_body / (measured * rate))
        assert cosine <= -1.0 + 1e-6, \
            f"edge {edge.index}: deficit is not anti-parallel to omega ({cosine:.9f})"

    # -- (c) the accumulated KF2 gap, two-sided ----------------------------------------
    states = _kf2_states()
    gap = _max_abs(states["B2-raw"].R, states["B1"].R)
    assert abs(gap - 2.9965e-8) <= 0.10 * 2.9965e-8, (
        f"KF2 rotation gap {gap:.6e}; the Cayley retraction gives 3.0e-8 +-10%.  A gap "
        f"near 1e-14 means Cay has been replaced by Exp.")

    # -- (d) it is O(dt^2), not O(dt) and not zero -------------------------------------
    gaps = []
    for n in STEP_SWEEP:
        s0 = T.initial_state()
        b1 = B1.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0)[2]
        b2 = B1.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0,
                                      cls=B2.CayleyPreintegration,
                                      variant=B2.VARIANT_RAW)[2]
        gaps.append(_max_abs(b2.R, b1.R))
    ratios = [gaps[k + 1] / gaps[k] for k in range(len(gaps) - 1)]
    assert all(abs(r - 0.25) <= 0.01 for r in ratios), (
        f"the Cayley deficit must scale as dt^2 (ratio 0.25 per halving); got {ratios} "
        f"from gaps {gaps}.  0.5 would be O(dt); ~1 would be round-off, i.e. no deficit.")


@pytest.mark.claim("M2-C23")
def test_c23_tier2_magnitude_and_convergence_order():
    """TIER 2, ``5e-3``, **plus** a convergence-order test.

    Both halves are required.  The magnitude bound alone would pass a baseline that is
    wrong by a constant ``1e-3``; the order test is the one with teeth.  Baseline 3 is
    exact ZOH and is the reference side of this comparison, so the gap is the O(dt)
    truncation of Baselines 1-2 and nothing else.

    The rotations are held to Tier 1, not Tier 0.  The claim used to say ``1e-10`` here;
    that holds for Baseline 1 (3.1e-15) but not for Baseline 2 under the raw Sec. IV-A
    reading, whose 3.0e-8 is the Cayley retraction deficit and is asserted as such below
    rather than forgiven.
    """
    # -- (a) magnitude at dt = 2.5e-3 ---------------------------------------------------
    states = _kf2_states()
    b3 = states["B3"]
    measured = {}
    for name in ("B1", "B2-raw", "B2-reparam"):
        s = states[name]
        measured[name] = {"v": _max_abs(s.v, b3.v), "p": _max_abs(s.p, b3.p),
                          "R": _max_abs(s.R, b3.R)}
    failures = []
    for name, gaps in measured.items():
        if gaps["v"] > TIER2:
            failures.append(f"{name} v: {gaps['v']:.3e} > {TIER2:.0e}")
        if gaps["p"] > TIER2:
            failures.append(f"{name} p: {gaps['p']:.3e} > {TIER2:.0e}")
        if gaps["R"] > TIER1:
            failures.append(f"{name} R: {gaps['R']:.3e} > {TIER1:.0e}")
    # Baseline 1 and the reparameterized variant are Exp and clear Tier 0 in R; the raw
    # Cayley reading does not and must not -- its 3.0e-8 is the M2-C22 retraction deficit.
    assert measured["B1"]["R"] <= TIER0, \
        f"B1 rotation vs B3: {measured['B1']['R']:.3e}"
    assert measured["B2-reparam"]["R"] <= TIER0, \
        f"B2-reparam rotation vs B3: {measured['B2-reparam']['R']:.3e}"
    assert abs(measured["B2-raw"]["R"] - 2.9965e-8) <= 0.10 * 2.9965e-8, (
        f"B2-raw rotation vs B3 is {measured['B2-raw']['R']:.3e}; the Cayley deficit is "
        f"3.0e-8 +-10% (see test_c22b)")

    # -- (b) the gap is first order in dt, for Baseline 1 AND Baseline 2 ----------------
    # Computed unconditionally: the order test is the half with teeth, so it must not be
    # skipped just because the magnitude half already failed.  The claim covers both
    # Euler baselines; the sweep used to run Baseline 1 only.
    sweeps = {"B1": {}, "B2-raw": {"cls": B2.CayleyPreintegration,
                                   "variant": B2.VARIANT_RAW}}
    for baseline, kwargs in sweeps.items():
        gaps_v, gaps_p = [], []
        for n in STEP_SWEEP:
            s0 = T.initial_state()
            kf2_b = B1.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0, **kwargs)[2]
            kf2_b3 = B3.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0)[2]
            gaps_v.append(_max_abs(kf2_b.v, kf2_b3.v))
            gaps_p.append(_max_abs(kf2_b.p, kf2_b3.p))

        for name, gaps in (("v", gaps_v), ("p", gaps_p)):
            ratios = [gaps[k + 1] / gaps[k] for k in range(len(gaps) - 1)]
            if not all(ORDER_BAND[0] <= r <= ORDER_BAND[1] for r in ratios):
                failures.append(f"{baseline} {name} gap ratios {ratios} (gaps {gaps})")

    assert not failures, f"Tier 2: {failures}; magnitudes {measured}"


@pytest.mark.claim("M2-C24")
def test_c24_tier0_against_exact_ground_truth():
    """TIER 0, ``1e-10``.  Baseline 3 is ground truth on this fixture to machine
    precision, which is exactly why the three baselines cannot share one tolerance.

    Baselines 1 and 2 match the reference ROTATIONS only because the fixture's body rate
    is constant per edge; in ``v`` and ``p`` they are held to Tier 2.

    The three baselines do not share a rotation tolerance either.  Baseline 1 and the
    reparameterized Cayley variant are both ``Exp`` and clear Tier 0 (9.8e-15 / 2.4e-14).
    Baseline 2 under the raw Sec. IV-A reading clears only Tier 1, at 3.0e-8, because its
    retraction loses ``||omega||^3 T dt^2 / 12`` by construction.  M2-C24 used to record
    "B1 and B2 rotation: 2.4e-14" and demand 1e-10 of both -- but 2.4e-14 is the
    *reparameterized* number, i.e. that line was measured on a Baseline 2 that is
    Baseline 1.  Each variant now carries its own tolerance, and the raw one carries a
    two-sided assertion so the deficit cannot silently vanish.
    """
    s0 = T.initial_state()
    exact = T.exact_states()

    galilean = B3.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0)
    for k in (1, 2):
        assert _max_abs(galilean[k].R, exact[k].R) <= TIER0
        assert _max_abs(galilean[k].v, exact[k].v) <= TIER0
        assert _max_abs(galilean[k].p, exact[k].p) <= TIER0
    # and against the Sec. 0.4 literals directly
    assert _max_abs(galilean[2].R, T.REF_R_KF2) <= TIER0
    assert _max_abs(galilean[2].v, T.REF_V_KF2) <= TIER0
    assert _max_abs(galilean[2].p, T.REF_P_KF2) <= TIER0

    euler = {
        "B1": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0),
        "B2-raw": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0,
                                           cls=B2.CayleyPreintegration,
                                           variant=B2.VARIANT_RAW),
        "B2-reparam": B1.integrate_edge_states(T.edges(), T.BIAS_TRUE, s0,
                                               cls=B2.CayleyPreintegration,
                                               variant=B2.VARIANT_REPARAM),
    }
    rotation_tolerance = {"B1": TIER0, "B2-reparam": TIER0, "B2-raw": TIER1}
    failures, measured = [], {}
    for name, states in euler.items():
        rot = max(_max_abs(states[k].R, exact[k].R) for k in (1, 2))
        vel = max(_max_abs(states[k].v, exact[k].v) for k in (1, 2))
        pos = max(_max_abs(states[k].p, exact[k].p) for k in (1, 2))
        measured[name] = {"R": rot, "v": vel, "p": pos}
        tol = rotation_tolerance[name]
        if rot > tol:
            failures.append(f"{name} R: {rot:.3e} > {tol:.0e}")
        if vel > TIER2 or pos > TIER2:
            failures.append(f"{name} v/p: {vel:.3e}/{pos:.3e} > {TIER2:.0e}")
    assert not failures, f"rotations against ground truth: {failures}; measured {measured}"
    # two-sided on the raw Cayley reading: 3.0e-8 is the measurement, not a budget
    assert abs(measured["B2-raw"]["R"] - 2.9965e-8) <= 0.10 * 2.9965e-8, (
        f"B2-raw rotation vs ground truth is {measured['B2-raw']['R']:.3e}; the Cayley "
        f"deficit is 3.0e-8 +-10% (see test_c22b)")


# ======================================================================================
# Observability
# ======================================================================================


def _observability_graph():
    bias = T.BIAS_TRUE
    exact = T.exact_states()
    params = B3.make_params(T.G_W, sigma_g=1.7e-4, sigma_a=1e-2, sigma_int=1e-6,
                            combined=True, sigma_bg=1e-5, sigma_ba=1e-4)

    pims = []
    for edge in T.edges():
        pim = gtsam.PreintegratedCombinedMeasurementsG(params, B3.to_gtsam_bias(bias))
        for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
            pim.integrateMeasurement(acc, omega, edge.dt)
        pims.append(pim)

    values = Values()
    keys = []
    for i, s in enumerate(exact):
        values.insert(X(i), Pose3(Rot3(s.R), s.p))
        values.insert(V(i), s.v)
        values.insert(B(i), B3.to_gtsam_bias(bias))
        keys += [X(i), V(i)]
    keys += [B(i) for i in range(T.N_KEYFRAMES)]
    return pims, values, Ordering(keys), exact


def _rank(graph, values, ordering):
    A, _ = graph.linearize(values).jacobian(ordering)
    s = np.linalg.svd(A, compute_uv=False)
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    return A, int((s > tol).sum())


@pytest.mark.claim("M2-C25")
def test_c25_fixture_is_bias_observable_without_a_bias_prior():
    """State dim ``3 x (6 pose + 3 vel + 6 bias) = 45``.

    Rank is invariant to positive reweighting, so none of this depends on the noise
    model.  The CONDITION number does depend on it and is deliberately not claimed.
    """
    pims, values, ordering, exact = _observability_graph()

    imu_only = NonlinearFactorGraph()
    for k, pim in enumerate(pims):
        imu_only.add(gtsam.GalileanCombinedImuFactor(
            X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim))

    # residuals vanish at ground truth
    for k, pim in enumerate(pims):
        factor = gtsam.GalileanCombinedImuFactor(
            X(k), V(k), X(k + 1), V(k + 1), B(k), B(k + 1), pim)
        residual = np.asarray(factor.unwhitenedError(values))
        assert np.abs(residual).max() <= 1e-12, \
            f"IMU factor {k} unwhitened residual {np.abs(residual).max():.3e}"

    A, rank_imu = _rank(imu_only, values, ordering)
    assert A.shape[1] == 45, f"state dimension is {A.shape[1]}, expected 45"
    assert rank_imu == 30, f"IMU edges alone: rank {rank_imu}/45, expected 30"
    # NB rank 30 on a 30x45 matrix is FULL ROW RANK -- the maximum a 30-row system can
    # have.  This assertion can only fail on row degeneracy; it says nothing whatever
    # about observability.  It is a structural non-degeneracy check and is labelled as
    # one so no reader counts it as evidence for the headline.
    assert A.shape[0] == 30 and rank_imu == min(A.shape), \
        "rank 30 is full ROW rank here, not an observability result"

    with_priors = imu_only.clone()
    with_priors.add(gtsam.PriorFactorPose3(
        X(0), Pose3(Rot3(exact[0].R), exact[0].p), noiseModel.Isotropic.Sigma(6, 1e-3)))
    with_priors.add(gtsam.PriorFactorVector(
        V(0), exact[0].v, noiseModel.Isotropic.Sigma(3, 1e-3)))
    A_priors, rank_priors = _rank(with_priors, values, ordering)
    assert rank_priors == 39, f"IMU + KF0 priors: rank {rank_priors}/45, expected 39"
    # Again full ROW rank (39 rows), so again structural and not observability.
    assert A_priors.shape[0] == 39 and rank_priors == min(A_priors.shape), \
        "rank 39 is full ROW rank here, not an observability result"

    # The 6 remaining directions are NOT "exactly the bias block" -- the claim used to
    # say so and it is false.  Measure where they actually live.  Principal cosines are
    # used rather than a per-direction energy split because the SVD nullspace basis is
    # only defined up to an orthogonal mixing, so per-direction numbers are not
    # reproducible while the cosines are.
    singular = np.linalg.svd(A_priors, compute_uv=False)
    nullspace = np.linalg.svd(A_priors)[2][rank_priors:].T      # 45 x 6
    assert nullspace.shape == (45, 6)
    assert np.abs(A_priors @ nullspace).max() <= 1e-6 * singular[0], \
        "extracted nullspace does not annihilate the Jacobian"
    # column layout under this ordering: X0 V0 X1 V1 X2 V2 B0 B1 B2 -> bias is cols 27:45
    bias_columns = nullspace[27:45, :]
    cosines = np.linalg.svd(bias_columns, compute_uv=False)
    expected_cosines = np.array([0.978227, 0.978041, 0.839290, 0.815739,
                                 0.299838, 0.299389])
    assert _max_abs(cosines, expected_cosines) <= 1e-3, \
        f"principal cosines to the bias block: {np.array2string(cosines, precision=6)}"
    assert cosines.min() <= 0.35, \
        "at least two null directions sit ~72 degrees off the bias block; if they were " \
        "'exactly the bias block' every cosine would be 1.0"
    pose_velocity_energy = float((nullspace[0:27, :] ** 2).sum())
    assert abs(pose_velocity_energy - 2.537) <= 1e-2, \
        f"pose/velocity energy in the nullspace {pose_velocity_energy:.3f} / 6"

    # A bias prior on ONE keyframe does close the gap, and that is STRUCTURAL, not luck:
    # the random-walk residuals force B0 = B1 = B2, so every null vector has the form
    # (M delta; delta, delta, delta) with delta != 0 (a nonzero pose/velocity part M delta
    # alone would be observable through the KF0 priors and the IMU edges).  No null
    # direction can therefore be orthogonal to the bias block, and a 6-dim prior on B0
    # removes all six.  This is strictly weaker than the "not the bias block" assertion
    # above and must not be read as evidence for it.  The cosines themselves are facts
    # about GTSAM's tangent coordinates (rescaling the bias units moves them), not about
    # geometry; the coordinate-free content is that every null vector has a nonzero
    # pose/velocity part.
    bias_pinned = with_priors.clone()
    bias_pinned.add(gtsam.PriorFactorConstantBias(
        B(0), B3.to_gtsam_bias(T.BIAS_TRUE), noiseModel.Isotropic.Sigma(6, 1e-3)))
    _, rank_bias_pinned = _rank(bias_pinned, values, ordering)
    assert rank_bias_pinned == 45, \
        f"a 6-dim bias prior should close exactly the gap, got {rank_bias_pinned}/45"

    full = with_priors.clone()
    for k in (1, 2):
        full.add(gtsam.GPSFactor(X(k), exact[k].p, noiseModel.Isotropic.Sigma(3, 1e-2)))
    A_full, rank_full = _rank(full, values, ordering)
    assert rank_full == 45, f"full graph: rank {rank_full}/45, expected 45"
    # THIS is the observability result: 45 on a square 45x45 system is not forced by the
    # row count, so unlike the two above it could have come out lower.
    assert A_full.shape == (45, 45), f"full Jacobian is {A_full.shape}, expected square"

    assert full.error(values) <= 1e-18, f"graph error {full.error(values):.3e}"
