"""Claims M1-C37, M1-C41..M1-C45 and M1-C53..M1-C55: the IMU model, the increment,
and the gtsam 4.3.0 preintegration / bias cross-validation.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.spatial.transform import Rotation

from src import gal3
from src import imu_dynamics as imu
from src.gal3 import compose, exp, gal, unpack, wedge

from . import ode_truth
from .conftest import (G, assert_discrete_converges, assert_stream_non_commuting, rand_axis,
                       rand_rot, rng)

I3 = np.eye(3)
I5 = np.eye(5)


# ==========================================================================
# the NavState endpoint form
# ==========================================================================
def _navstate_form_b(R_i, v_i, p_i, dR, dv, dp, Dt, g):
    """W_ij . Phi_Dt(X_i) . U_ij using gtsam NavState objects."""
    import gtsam

    W = gtsam.NavState(gtsam.Rot3(I3), 0.5 * g * Dt ** 2, g * Dt)
    X_i = gtsam.NavState(gtsam.Rot3(R_i), p_i, v_i)
    phi_X = gtsam.NavState(X_i.attitude(), X_i.position() + X_i.velocity() * Dt,
                           X_i.velocity())
    U = gtsam.NavState(gtsam.Rot3(dR), dp, dv)
    return W.compose(phi_X).compose(U)


@pytest.mark.claim("M1-C37")
def test_gtsam_navstate_endpoint_form(endpoint_dataset):
    """X_j = W_ij Phi_Dt(X_i) U_ij reproduces (R_j, p_j, v_j) on gtsam 4.3.0."""
    d = endpoint_dataset
    Dt, g = d["Dt"], d["g"]

    def residual(src):
        worst = 0.0
        for k in range(d["n"]):
            N = _navstate_form_b(d["R_i"][k], d["v_i"][k], d["p_i"][k],
                                 src["dR"][k], src["dv"][k], src["dp"][k], Dt, g)
            worst = max(worst,
                        float(np.abs(N.attitude().matrix() - src["R_j"][k]).max()),
                        float(np.abs(N.position() - src["p_j"][k]).max()),
                        float(np.abs(N.velocity() - src["v_j"][k]).max()))
        return worst

    # ACCURACY: against the DOP853 continuous truth (time-varying inputs)
    worst = residual(d)
    assert worst <= 1e-9, worst
    # IDENTITY CHECK on the project's discrete recursion (not an accuracy check)
    worst_id = residual(d["disc"])
    assert worst_id <= 1e-12, worst_id
    assert_discrete_converges(d)
    assert_stream_non_commuting(d)

    # 2000 algebraic instances against the Forster endpoint
    r = rng(370)
    worst_alg = 0.0
    for _ in range(2000):
        Dt2 = float(r.uniform(0.05, 1.0))
        R_i, v_i, p_i = rand_rot(r), 2.0 * r.normal(size=3), 5.0 * r.normal(size=3)
        dR, dv, dp = rand_rot(r, scale=0.4), r.normal(size=3), r.normal(size=3)
        N = _navstate_form_b(R_i, v_i, p_i, dR, dv, dp, Dt2, G)
        R_f, v_f, p_f = gal3.forster_endpoint(R_i, v_i, p_i, dR, dv, dp, Dt2, G)
        worst_alg = max(worst_alg,
                        float(np.abs(N.attitude().matrix() - R_f).max()),
                        float(np.abs(N.position() - p_f).max()),
                        float(np.abs(N.velocity() - v_f).max()))
    # W_ij's position entry is +0.5 g Dt^2 (PLUS).  A former check that a MINUS-sign
    # W misses forster_endpoint by >= 1e-3 was deleted: it is a fact about gtsam
    # arithmetic (the two W differ by |g| Dt^2), and no change to src could make it
    # fail.  The sign is pinned by worst_alg above and by the ODE comparisons.
    assert worst_alg <= 1e-12, worst_alg


# ==========================================================================
# the increment
# ==========================================================================
@pytest.mark.claim("M1-C41")
def test_increment_left_invariant_ode():
    """Ddot = D xi^ with xi = (omega, a, 0, 1); D(Dt) = exp(Dt xi) for constant inputs."""
    r = rng(41)

    # (a) the continuous claim, against an INDEPENDENT ODE solve: DOP853 on
    #     Ddot = D xi^ (rtol = atol = 1e-13) vs the closed form exp(Dt xi).
    worst_ode_const = 0.0
    for Dt in (0.02, 0.1, 0.4, 1.0, 2.0):
        omega = 0.7 * r.normal(size=(200, 3))
        acc = 2.0 * r.normal(size=(200, 3))
        dR, dv, dp = ode_truth.solve_increment_constant(omega, acc, Dt)
        for k in range(200):
            ref = imu.preintegrate_constant(omega[k], acc[k], Dt)
            worst_ode_const = max(worst_ode_const,
                                  float(np.abs(gal(dR[k], dv[k], dp[k], Dt) - ref).max()))
    assert worst_ode_const <= 1e-11, worst_ode_const

    # (b) the project's Euler scheme, constant inputs, at the WORST Dt of the stated
    #     range (the first-order error grows with Dt): 1000 configurations at Dt = 0.4.
    #     The worst error is 2.970 dt on these configurations, so tolerance 1e-5 needs
    #     dt <= 3.37e-6; the recorded rule is dt <= 2.5e-6, tested at that boundary.
    dt = 2.5e-6
    n = int(round(0.4 / dt))
    omega = 0.7 * r.normal(size=(1000, 3))
    acc = 2.0 * r.normal(size=(1000, 3))
    dR, dv, dp, t = imu.euler_increment(omega, acc, dt, n)
    assert abs(t - 0.4) <= 1e-12
    worst = 0.0
    for k in range(1000):
        ref = imu.preintegrate_constant(omega[k], acc[k], t)
        worst = max(worst, float(np.abs(gal(dR[k], dv[k], dp[k], t) - ref).max()))
    assert worst <= 1e-5, worst

    # (b') "explicit Euler" pinned, on the same 1000 configurations.  Explicit Euler on
    #     Ddot = D A (A = xi^ constant) is D_n = (I5 + dt A)^n, so
    #     n log(I + dt A) = T A - (T dt / 2) A^2 + O(dt^2) gives the SIGNED first-order
    #     error  D_n - exp(T A) = -(T dt / 2) A^2 exp(T A) + O(dt^2).  A semi-implicit
    #     update (dp from the NEW dv) is also first order but has a different
    #     coefficient (mismatch 2.9 here), which the err/dt magnitude checks cannot see.
    #     A is written by hand and exp(T A) is scipy expm -- nothing from src.
    def hat_by_hand(om, a):
        A = np.zeros((5, 5))
        A[0:3, 0:3] = [[0.0, -om[2], om[1]], [om[2], 0.0, -om[0]], [-om[1], om[0], 0.0]]
        A[0:3, 3] = a
        A[3, 4] = 1.0
        return A

    dt_c, T_c = 1e-4, 0.4
    n_c = int(round(T_c / dt_c))
    cR, cv, cp, cT = imu.euler_increment(omega, acc, dt_c, n_c)
    worst_coef = worst_pred = worst_pow = 0.0
    for k in range(1000):
        A = hat_by_hand(omega[k], acc[k])
        E = sla.expm(T_c * A)
        pred = -(T_c / 2.0) * (A @ A @ E)
        coef = (gal(cR[k], cv[k], cp[k], cT) - E) / dt_c
        worst_coef = max(worst_coef, float(np.abs(coef - pred).max()))
        worst_pred = max(worst_pred, float(np.abs(pred).max()))
        if k < 20:   # and literally the matrix power (I5 + dt A)^n
            P = np.linalg.matrix_power(I5 + dt_c * A, n_c)
            worst_pow = max(worst_pow,
                            float(np.abs(gal(cR[k], cv[k], cp[k], cT) - P).max()))
    assert worst_coef <= 1e-2, worst_coef
    # the predicted coefficient also accounts for part (b)'s worst error (2.970 dt)
    assert 0.99 <= (worst / dt) / worst_pred <= 1.01, (worst / dt, worst_pred)
    assert worst_pow <= 1e-11, worst_pow

    # (c) TIME-VARYING, non-commuting xi(t): the left-invariant ODE (D xi^, not
    #     xi^ D) against DOP853.  Euler is first order, so check the error and the
    #     order; with constant inputs a swapped product dR @ W -> W @ dR is invisible.
    st = ode_truth.Stream.random(r, 50)
    Dt_tv = 0.4
    dR_ref, dv_ref, dp_ref = ode_truth.solve_increment(st, Dt_tv)
    errs = []
    for h in (2e-5, 1e-5):
        m = int(round(Dt_tv / h))
        om_s, acc_s = st.sampled(h, m)
        out = imu.euler_increment(om_s, acc_s, h, m, per_step=True)
        errs.append(max(float(np.abs(out[0] - dR_ref).max()),
                        float(np.abs(out[1] - dv_ref).max()),
                        float(np.abs(out[2] - dp_ref).max())))
    assert errs[1] <= 1e-4, errs
    assert 1.8 <= errs[0] / errs[1] <= 2.2, errs
    # INPUT GUARD (a property of the test's stream, not evidence about src): the
    # stream genuinely distinguishes D xi^ from xi^ D -- the right-invariant rotation
    # solution (R^T solves Sdot = S [-omega]x) is far from the left one.  Without it
    # a commuting stream would silently let the order check above pass for a swapped
    # product (mutation harness: ode_truth_constant_omega).
    dR_neg, _, _ = ode_truth.solve_increment(st.negated_rate(), Dt_tv)
    order_gap = float(np.abs(dR_ref - np.swapaxes(dR_neg, -1, -2)).max())
    assert order_gap >= 100.0 * 1e-4, order_gap

    # the ODE itself, by central difference on expm(s xi^)
    worst_ode = 0.0
    h = 1e-6
    for _ in range(50):
        xi = imu.increment_generator(0.7 * r.normal(size=3), 2.0 * r.normal(size=3))
        A = wedge(xi)
        s = float(r.uniform(0.02, 0.4))
        deriv = (sla.expm((s + h) * A) - sla.expm((s - h) * A)) / (2.0 * h)
        rhs = sla.expm(s * A) @ A
        worst_ode = max(worst_ode,
                        float(np.abs(deriv - rhs).max()) / max(1.0, float(np.abs(rhs).max())))
    assert worst_ode <= 1e-8, worst_ode


@pytest.mark.claim("M1-C42")
def test_increment_closed_form_blocks():
    """dR = Exp(omega Dt), dv = Dt G1 a, dp = Dt^2 G2 a, time block Dt."""
    r = rng(42)
    worst = 0.0
    for _ in range(500):
        omega = 0.7 * r.normal(size=3)
        acc = 2.0 * r.normal(size=3)
        Dt = float(r.uniform(0.02, 0.5))
        dR, dv, dp, t = imu.increment_blocks(omega, acc, Dt)
        ref = imu.preintegrate_constant(omega, acc, Dt)
        worst = max(worst, float(np.abs(gal(dR, dv, dp, t) - ref).max()))
        assert abs(t - Dt) <= 1e-15
        assert abs(float(ref[3, 4]) - Dt) <= 1e-15
    assert worst <= 1e-12, worst

    # dp carries no Gamma1 * w term: the generator's w slot is zero, so position
    # comes entirely from the alpha * Gamma2 * u coupling.
    omega, acc, Dt = 0.3 * np.ones(3), np.array([0.4, -0.2, 1.1]), 0.37
    xi = imu.increment_generator(omega, acc)
    assert np.abs(xi[6:9]).max() == 0.0
    _, _, dp, _ = imu.increment_blocks(omega, acc, Dt)
    assert np.abs(dp - Dt ** 2 * (gal3.gamma2(omega * Dt) @ acc)).max() <= 1e-15
    assert np.abs(dp).max() >= 1e-3


@pytest.mark.claim("M1-C43")
def test_forster_component_equations(endpoint_dataset):
    """R_j = R_i dR, v_j = v_i + g Dt + R_i dv, p_j = p_i + v_i Dt + 0.5 g Dt^2 + R_i dp."""
    d = endpoint_dataset
    Dt, g = d["Dt"], d["g"]

    def residuals(src):
        worst = 0.0
        worst_block = 0.0
        for k in range(d["n"]):
            R_f, v_f, p_f = gal3.forster_endpoint(
                d["R_i"][k], d["v_i"][k], d["p_i"][k],
                src["dR"][k], src["dv"][k], src["dp"][k], Dt, g)
            worst = max(worst,
                        float(np.abs(R_f - src["R_j"][k]).max()),
                        float(np.abs(v_f - src["v_j"][k]).max()),
                        float(np.abs(p_f - src["p_j"][k]).max()))
            # the 5x5 endpoint relation decomposes into exactly these three equations
            X_i = gal(d["R_i"][k], d["v_i"][k], d["p_i"][k], 0.0)
            D = gal(src["dR"][k], src["dv"][k], src["dp"][k], Dt)
            R_l, v_l, p_l, t_l = unpack(gal3.endpoint(X_i, D, g))
            worst_block = max(worst_block,
                              float(np.abs(R_l - R_f).max()),
                              float(np.abs(v_l - v_f).max()),
                              float(np.abs(p_l - p_f).max()),
                              abs(t_l))
        return worst, worst_block

    # ACCURACY: against the DOP853 continuous truth (time-varying inputs)
    worst, worst_block = residuals(d)
    assert worst <= 1e-9, worst
    assert worst_block <= 1e-12, worst_block

    # IDENTITY CHECK on the project's discrete recursion (not an accuracy check)
    worst_id, _ = residuals(d["disc"])
    assert worst_id <= 1e-12, worst_id
    assert_discrete_converges(d)
    assert_stream_non_commuting(d)


@pytest.mark.claim("M1-C44")
def test_bias_correction_sign():
    """omega = omega_m - b_g, a = a_m - b_a, with b = [b_g ; b_a]."""
    r = rng(44)
    worst = 0.0
    for _ in range(1000):
        omega = r.normal(size=3)
        acc = r.normal(size=3)
        b = r.normal(size=6)
        omega_m = omega + b[0:3]
        acc_m = acc + b[3:6]
        o_hat, a_hat = imu.correct_measurements(omega_m, acc_m, b)
        worst = max(worst, float(np.abs(o_hat - omega).max()),
                    float(np.abs(a_hat - acc).max()))
        # the sign: applying the routine with -b must ADD the bias back
        o_neg, a_neg = imu.correct_measurements(omega_m, acc_m, -b)
        worst = max(worst, float(np.abs(o_neg - (omega_m + b[0:3])).max()),
                    float(np.abs(a_neg - (acc_m + b[3:6])).max()))
    assert worst <= 1e-14, worst


@pytest.mark.claim("M1-C45")
def test_preintegration_chaining_is_the_group_product():
    """D_ij D_jk = D_ik, with the FIRST segment's velocity and the SECOND's duration."""
    r = rng(45)
    n_inst, n1, n2, dt = 300, 800, 1200, 5e-5
    T1, T2 = n1 * dt, n2 * dt

    # IDENTITY CHECK on the project's recursion, with TIME-VARYING per-step inputs
    # (constant inputs would let an accumulation-order defect hide).
    st = ode_truth.Stream.random(r, n_inst)
    om1, a1 = st.sampled(dt, n1)
    om2, a2 = st.shifted(T1).sampled(dt, n2)
    D_ij = imu.integrate_increment(om1, a1, dt, n1, per_step=True)
    D_jk = imu.integrate_increment(om2, a2, dt, n2, per_step=True)
    D_ik = imu.integrate_increment(om2, a2, dt, n2, D0=D_ij, per_step=True)
    D_full = imu.integrate_increment(np.concatenate([om1, om2]), np.concatenate([a1, a2]),
                                     dt, n1 + n2, per_step=True)
    worst = 0.0
    for k in range(n_inst):
        A = gal(D_ij[0][k], D_ij[1][k], D_ij[2][k], D_ij[3])
        B = gal(D_jk[0][k], D_jk[1][k], D_jk[2][k], D_jk[3])
        C = gal(D_ik[0][k], D_ik[1][k], D_ik[2][k], D_ik[3])
        F = gal(D_full[0][k], D_full[1][k], D_full[2][k], D_full[3])
        worst = max(worst, float(np.abs(imu.chain(A, B) - C).max()),
                    float(np.abs(imu.chain(A, B) - F).max()))
    assert worst <= 1e-12, worst

    # The recursion used above is a consistent discretisation of the increment ODE,
    # checked against DOP853 (the identity check alone holds for ANY per-step Gal(3)
    # element, e.g. one with 1.5 a dt^2 in position).
    #  * omega = 0, constant a: zero-order hold is then exact, so integrate_increment
    #    must reproduce the ODE solution to round-off for any dt;
    #  * the time-varying stream above: first order, err(2 dt) / err(dt) ~ 2.
    acc0 = 2.0 * r.normal(size=(n_inst, 3))
    z = np.zeros((n_inst, 3))
    xR, xv, xp, xT = imu.integrate_increment(z, acc0, dt, n1)
    tR, tv, tp = ode_truth.solve_increment_constant(z, acc0, xT)
    worst_w0 = max(float(np.abs(xR - tR).max()), float(np.abs(xv - tv).max()),
                   float(np.abs(xp - tp).max()))
    assert worst_w0 <= 1e-12, worst_w0
    eR, ev, ep = ode_truth.solve_increment(st, T1)
    errs = []
    for h, m in ((2 * dt, n1 // 2), (dt, n1)):
        om_h, a_h = st.sampled(h, m)
        o = imu.integrate_increment(om_h, a_h, h, m, per_step=True)
        errs.append(max(float(np.abs(o[0] - eR).max()), float(np.abs(o[1] - ev).max()),
                        float(np.abs(o[2] - ep).max())))
    assert errs[1] <= 1e-4, errs
    assert 1.8 <= errs[0] / errs[1] <= 2.2, errs

    # The per-step element IS a Gal(3) element (which is why the increments chain as
    # the product): imu_step_element(omega, a, dt) = exp(dt (0, a, 0, 1)) . (Exp(omega
    # dt), 0, 0, 0), reference by scipy expm and scipy Rotation; and integrate_increment
    # is the ordered product of these elements.
    worst_step = 0.0
    for _ in range(200):
        om_k, a_k, h = 2.0 * r.normal(size=3), 3.0 * r.normal(size=3), float(r.uniform(1e-3, 0.2))
        A = np.zeros((5, 5))
        A[0:3, 3] = h * a_k
        A[3, 4] = h
        Rk = np.eye(5)
        Rk[0:3, 0:3] = Rotation.from_rotvec(om_k * h).as_matrix()
        worst_step = max(worst_step,
                         float(np.abs(imu.imu_step_element(om_k, a_k, h) - sla.expm(A) @ Rk).max()))
    assert worst_step <= 1e-12, worst_step
    om_s, a_s = st.sampled(dt, 50)
    Dp = imu.integrate_increment(om_s, a_s, dt, 50, per_step=True)
    worst_prod = 0.0
    for k in range(20):
        P = np.eye(5)
        for j in range(50):
            P = P @ imu.imu_step_element(om_s[j, k], a_s[j, k], dt)
        worst_prod = max(worst_prod, float(np.abs(gal(Dp[0][k], Dp[1][k], Dp[2][k], Dp[3]) - P).max()))
    assert worst_prod <= 1e-12, worst_prod

    # ACCURACY: the continuous increments from three separate DOP853 solves chain
    # as the group product (eR, ev, ep = the [0, T1] solve above).
    fR, fv, fp = ode_truth.solve_increment(st.shifted(T1), T2)
    gR, gv, gp = ode_truth.solve_increment(st, T1 + T2)
    worst_ode = 0.0
    for k in range(n_inst):
        C = imu.chain(gal(eR[k], ev[k], ep[k], T1), gal(fR[k], fv[k], fp[k], T2))
        worst_ode = max(worst_ode, float(np.abs(C - gal(gR[k], gv[k], gp[k], T1 + T2)).max()))
    assert worst_ode <= 1e-10, worst_ode

    # 2000 purely algebraic triples, plus the swapped-duration form which must fail
    worst_alg = 0.0
    worst_wrong = 0.0
    for _ in range(2000):
        Dt_ij, Dt_jk = float(r.uniform(0.05, 1.0)), float(r.uniform(0.05, 1.0))
        A = gal(rand_rot(r, scale=0.5), r.normal(size=3), r.normal(size=3), Dt_ij)
        B = gal(rand_rot(r, scale=0.5), r.normal(size=3), r.normal(size=3), Dt_jk)
        C = imu.chain(A, B)
        dR_ij, dv_ij, dp_ij, _ = unpack(A)
        dR_jk, dv_jk, dp_jk, _ = unpack(B)
        worst_alg = max(
            worst_alg,
            float(np.abs(C[0:3, 0:3] - dR_ij @ dR_jk).max()),
            float(np.abs(C[0:3, 3] - (dv_ij + dR_ij @ dv_jk)).max()),
            float(np.abs(C[0:3, 4] - (dp_ij + dR_ij @ dp_jk + dv_ij * Dt_jk)).max()),
            abs(float(C[3, 4]) - (Dt_ij + Dt_jk)),
        )
        wrong = dp_ij + dR_ij @ dp_jk + dv_jk * Dt_ij
        worst_wrong = max(worst_wrong, float(np.abs(C[0:3, 4] - wrong).max()))
    assert worst_alg <= 1e-14, worst_alg
    assert worst_wrong >= 1e-3, worst_wrong


# ==========================================================================
# gtsam preintegration and bias
# ==========================================================================
@pytest.mark.claim("M1-C53")
def test_gtsam_pim_predict_matches_both_forms():
    """gtsam's PreintegratedImuMeasurements.predict agrees with both endpoint forms."""
    import gtsam

    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    assert np.abs(np.asarray(params.n_gravity) - np.array([0.0, 0.0, -9.81])).max() <= 1e-12
    g = np.asarray(params.n_gravity, dtype=float).reshape(3)

    bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurements(params, bias)
    omega = np.array([0.3, -0.2, 0.5])
    acc = np.array([0.4, 0.2, 10.11])
    for _ in range(500):
        pim.integrateMeasurement(acc, omega, 1e-3)

    Dt = float(pim.deltaTij())
    dR = pim.deltaRij().matrix()
    dv = np.asarray(pim.deltaVij(), dtype=float).reshape(3)
    dp = np.asarray(pim.deltaPij(), dtype=float).reshape(3)
    assert abs(Dt - 0.5) <= 1e-9

    r = rng(53)
    worst_a = worst_b = worst_t = 0.0
    for _ in range(25):
        R_i, v_i, p_i = rand_rot(r), 2.0 * r.normal(size=3), 5.0 * r.normal(size=3)
        state_i = gtsam.NavState(gtsam.Rot3(R_i), p_i, v_i)
        pred = pim.predict(state_i, bias)

        # form (a): T(-Dt) H' X_i D_ij in Gal(3)
        form_a = gal3.endpoint(gal(R_i, v_i, p_i, 0.0), gal(dR, dv, dp, Dt), g)
        R_a, v_a, p_a, t_a = unpack(form_a)
        worst_a = max(worst_a,
                      float(np.abs(R_a - pred.attitude().matrix()).max()),
                      float(np.abs(v_a - pred.velocity()).max()),
                      float(np.abs(p_a - pred.position()).max()))
        worst_t = max(worst_t, abs(t_a))

        # form (b): W_ij Phi_Dt(X_i) U_ij on NavState
        N = _navstate_form_b(R_i, v_i, p_i, dR, dv, dp, Dt, g)
        worst_b = max(worst_b,
                      float(np.abs(N.attitude().matrix() - pred.attitude().matrix()).max()),
                      float(np.abs(N.velocity() - pred.velocity()).max()),
                      float(np.abs(N.position() - pred.position()).max()))
    assert worst_a <= 1e-9, worst_a
    assert worst_b <= 1e-9, worst_b
    assert worst_t <= 1e-12, worst_t


@pytest.mark.claim("M1-C54")
def test_gtsam_constant_bias_is_accel_first():
    """ConstantBias(biasAcc, biasGyro) and .vector() = [b_a ; b_g]."""
    import gtsam

    r = rng(54)
    worst = 0.0
    margin = np.inf
    for _ in range(1000):
        b_a = r.normal(size=3)
        b_g = 10.0 * r.normal(size=3) + 50.0
        cb = gtsam.imuBias.ConstantBias(b_a, b_g)
        worst = max(worst, float(np.abs(cb.accelerometer() - b_a).max()))
        worst = max(worst, float(np.abs(cb.gyroscope() - b_g).max()))
        worst = max(worst, float(np.abs(cb.vector() - np.concatenate([b_a, b_g])).max()))
        margin = min(margin, float(np.abs(cb.vector()[0:3] - b_g).max()))
    assert worst <= 1e-14, worst
    assert margin >= 1e-3, margin


@pytest.mark.claim("M1-C55")
def test_bias_convention_swap():
    """b = [b_g ; b_a] <-> ConstantBias is the involutive block swap S."""
    S = imu.BIAS_SWAP
    assert np.array_equal(S @ S, np.eye(6))

    r = rng(55)
    worst = 0.0
    worst_naive = 0.0
    for _ in range(1000):
        b = r.normal(size=6)
        cb = imu.bias_to_gtsam(b)
        worst = max(worst, float(np.abs(np.asarray(cb.vector()) - S @ b).max()))
        worst = max(worst, float(np.abs(imu.bias_from_gtsam(cb) - b).max()))
        worst = max(worst, float(np.abs(
            np.concatenate([cb.gyroscope(), cb.accelerometer()]) - b).max()))
        # the naive conversion (gyro passed as the accel slot) does not round-trip
        import gtsam

        naive = gtsam.imuBias.ConstantBias(b[0:3], b[3:6])
        worst_naive = max(worst_naive, float(np.abs(
            np.concatenate([naive.gyroscope(), naive.accelerometer()]) - b).max()))
    assert worst <= 1e-14, worst
    assert worst_naive >= 1e-3, worst_naive
