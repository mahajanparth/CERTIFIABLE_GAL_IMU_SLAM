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
    # REF_R_KF1 used to be *defined* as R_0 @ REF_DR_01, which made this comparison
    # circular -- it could not fail for any input.  It is now a transcribed literal, so
    # the chaining relation genuinely has to reproduce it.
    assert T.REF_R_KF1.flags.owndata, "REF_R_KF1 must be a literal, not a derived view"
    assert _max_abs(states[1].R, T.REF_R_KF1) <= 1e-12
    assert _max_abs(T.R_0 @ T.REF_DR_01, T.REF_R_KF1) <= 1e-12
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

    direction = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    direction /= np.linalg.norm(direction)
    for edge in T.edges():
        base = B1.integrate_edge(edge, T.BIAS_TRUE)
        jac = B1.bias_jacobians_fd(edge, T.BIAS_TRUE)
        J_R, J_vg, J_va, J_pg, J_pa = jac
        assert J_R.shape == J_vg.shape == J_pa.shape == (3, 3)

        errors = {}
        for mag in (1e-3, 1e-2):
            db = mag * direction
            corrected = B1.bias_corrected_increment(base.increment(), jac, db)
            # the rotation correction really is a RIGHT group composition
            assert np.allclose(corrected[0],
                               base.dR @ T.so3_exp(J_R @ db[0:3]), atol=0, rtol=0)
            reintegrated = B1.integrate_edge(edge, T.BIAS_TRUE + db).increment()
            errors[mag] = B1.increment_error(corrected, reintegrated)

        ratio = errors[1e-2] / errors[1e-3]
        assert 50.0 <= ratio <= 200.0, \
            f"edge {edge.index}: not quadratic, ratio {ratio:.1f} (errors {errors})"


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
    the 1e-9 the claim used to assert there.  Asserting it equals ``|dvbar - dv_exact|``
    to 1e-12 is sharp: any change to the Eq. (9) sums moves one side and not the other.
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

    # -- (b) the gap is first order in dt ------------------------------------------------
    # Computed unconditionally: the order test is the half with teeth, so it must not be
    # skipped just because the magnitude half already failed.
    gaps_v, gaps_p = [], []
    for n in STEP_SWEEP:
        s0 = T.initial_state()
        kf2_b1 = B1.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0)[2]
        kf2_b3 = B3.integrate_edge_states(T.edges(n), T.BIAS_TRUE, s0)[2]
        gaps_v.append(_max_abs(kf2_b1.v, kf2_b3.v))
        gaps_p.append(_max_abs(kf2_b1.p, kf2_b3.p))

    for name, gaps in (("v", gaps_v), ("p", gaps_p)):
        ratios = [gaps[k + 1] / gaps[k] for k in range(len(gaps) - 1)]
        if not all(ORDER_BAND[0] <= r <= ORDER_BAND[1] for r in ratios):
            failures.append(f"{name} gap ratios {ratios} (gaps {gaps})")

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

    # A bias prior on ONE keyframe does close the gap -- but only because no null
    # direction is ORTHOGONAL to the bias block, which is strictly weaker than the
    # sub-assertion above and must not be read as evidence for it.
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
