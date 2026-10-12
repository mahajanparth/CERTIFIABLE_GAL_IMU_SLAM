"""Claims M1-C37, M1-C41..M1-C45 and M1-C53..M1-C55: the IMU model, the increment,
and the gtsam 4.3.0 preintegration / bias cross-validation.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from src import gal3
from src import imu_dynamics as imu
from src.gal3 import compose, exp, gal, unpack, wedge

from .conftest import G, rand_axis, rand_rot, rng

I3 = np.eye(3)


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
    worst = 0.0
    for k in range(d["n"]):
        N = _navstate_form_b(d["R_i"][k], d["v_i"][k], d["p_i"][k],
                             d["dR"][k], d["dv"][k], d["dp"][k], Dt, g)
        worst = max(worst,
                    float(np.abs(N.attitude().matrix() - d["R_j"][k]).max()),
                    float(np.abs(N.position() - d["p_j"][k]).max()),
                    float(np.abs(N.velocity() - d["v_j"][k]).max()))
    assert worst <= 1e-5, worst

    # 2000 algebraic instances against the Forster endpoint
    r = rng(370)
    worst_alg = 0.0
    worst_minus = 0.0
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
        # W_ij's position entry is +0.5 g Dt^2: the minus sign is wrong here
        import gtsam

        W_bad = gtsam.NavState(gtsam.Rot3(I3), -0.5 * G * Dt2 ** 2, G * Dt2)
        X_i = gtsam.NavState(gtsam.Rot3(R_i), p_i, v_i)
        phi_X = gtsam.NavState(X_i.attitude(), X_i.position() + X_i.velocity() * Dt2,
                               X_i.velocity())
        N_bad = W_bad.compose(phi_X).compose(gtsam.NavState(gtsam.Rot3(dR), dp, dv))
        worst_minus = max(worst_minus, float(np.abs(N_bad.position() - p_f).max()))
    assert worst_alg <= 1e-12, worst_alg
    assert worst_minus >= 1e-3, worst_minus


# ==========================================================================
# the increment
# ==========================================================================
@pytest.mark.claim("M1-C41")
def test_increment_left_invariant_ode():
    """Ddot = D xi^ with xi = (omega, a, 0, 1); D(Dt) = exp(Dt xi) for constant inputs."""
    r = rng(41)
    dt = 2.5e-6                       # the claim requires dt <= 5e-6
    worst = 0.0
    for _ in range(10):
        omega = 0.7 * r.normal(size=3)
        acc = 2.0 * r.normal(size=3)
        Dt = float(r.uniform(0.02, 0.4))
        n = int(round(Dt / dt))
        Dt = n * dt                      # keep Dt an exact multiple of dt
        dR, dv, dp, t = imu.euler_increment(omega, acc, dt, n)
        assert abs(t - Dt) <= 1e-12
        ref = imu.preintegrate_constant(omega, acc, Dt)
        worst = max(worst, float(np.abs(gal(dR, dv, dp, t) - ref).max()))
    assert worst <= 1e-5, worst

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
    worst = 0.0
    worst_block = 0.0
    for k in range(d["n"]):
        R_f, v_f, p_f = gal3.forster_endpoint(
            d["R_i"][k], d["v_i"][k], d["p_i"][k],
            d["dR"][k], d["dv"][k], d["dp"][k], Dt, g)
        worst = max(worst,
                    float(np.abs(R_f - d["R_j"][k]).max()),
                    float(np.abs(v_f - d["v_j"][k]).max()),
                    float(np.abs(p_f - d["p_j"][k]).max()))
        # the 5x5 endpoint relation decomposes into exactly these three equations
        X_i = gal(d["R_i"][k], d["v_i"][k], d["p_i"][k], 0.0)
        D = gal(d["dR"][k], d["dv"][k], d["dp"][k], Dt)
        lhs = gal3.endpoint(X_i, D, g)
        R_l, v_l, p_l, t_l = unpack(lhs)
        worst_block = max(worst_block,
                          float(np.abs(R_l - R_f).max()),
                          float(np.abs(v_l - v_f).max()),
                          float(np.abs(p_l - p_f).max()),
                          abs(t_l))
    assert worst <= 1e-5, worst
    assert worst_block <= 1e-12, worst_block


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
    om1 = 0.7 * r.normal(size=(n_inst, 3))
    a1 = 2.0 * r.normal(size=(n_inst, 3))
    om2 = 0.7 * r.normal(size=(n_inst, 3))
    a2 = 2.0 * r.normal(size=(n_inst, 3))
    D_ij = imu.integrate_increment(om1, a1, dt, n1)
    D_jk = imu.integrate_increment(om2, a2, dt, n2)
    D_ik = imu.integrate_increment(om2, a2, dt, n2, D0=D_ij)
    worst = 0.0
    for k in range(n_inst):
        A = gal(D_ij[0][k], D_ij[1][k], D_ij[2][k], D_ij[3])
        B = gal(D_jk[0][k], D_jk[1][k], D_jk[2][k], D_jk[3])
        C = gal(D_ik[0][k], D_ik[1][k], D_ik[2][k], D_ik[3])
        worst = max(worst, float(np.abs(imu.chain(A, B) - C).max()))
    assert worst <= 1e-12, worst

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
