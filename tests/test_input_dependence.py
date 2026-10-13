"""Input dependence: every input of every public ``src`` callable, varied off the fixture.

Seven verification passes found the same class of defect seven times: an input of some
function that no test ever moves away from its fixture value, so replacing the input by
the fixture constant passes the whole suite (pass 7: U03-U07, U10-U13, U20, U23, U23b,
U26, U30-U34, U38-U40).  This module closes the class rather than the instances.  The
table below lists every public function, method and constructor in ``src`` and every
input it reads -- explicit arguments, dataclass fields, and the module constants it
reads implicitly -- and names the test here that runs it at TWO values, both different
from the fixture value and from each other, against a reference that does not go
through the code under test (retyped equations, ``scipy.linalg.expm`` / ``logm``,
independent central differences, or a GTSAM call that ``src`` does not wrap).

Values used throughout (none is a fixture value):

* edges: ``EDGE_A`` (dt 1e-3, 200 steps, Dt 0.2 s) and ``EDGE_B`` (dt 5e-3, 146 steps,
  Dt 0.73 s), with body rates / forces that are not the fixture's;
* gravity: ``G_A`` (|g| 9.79, tilted off -z) and ``G_B`` (|g| 9.83, tilted elsewhere);
* biases: ``BIAS_GYRO`` (accel half zero), ``BIAS_ACCEL`` (gyro half zero), ``BIAS_BOTH``;
* states: ``STATE_A``, ``STATE_B`` (R, v, p all non-fixture, p != 0);
* densities: sigma_g 3.1e-4 / 8.3e-4 (or per-axis), sigma_a 7.3e-3 / 2.6e-2, sigma_int
  2.2e-3 / 4.0e-4, sigma_bg 3e-5 / 7e-6, sigma_ba 2e-4 / 6e-4 -- gyro != accel always.

INVENTORY  (function / constructor  x  input  ->  test in this module)
=====================================================================================

src/trajectory.py
-----------------
hat(a)                       a                     test_traj_so3_helpers
vee(A)                       A                     test_traj_so3_helpers
so3_exp(w)                   w (both branches)     test_traj_so3_helpers
so3_log(R)                   R (both branches)     test_traj_so3_helpers
M1(w, T)                     w (both branches), T  test_traj_M1_M2
M2(w, T)                     w (both branches), T  test_traj_M1_M2
NavTriple(R, v, p)           R, v, p               test_traj_navtriple_and_edge_fields,
                                                   test_f18_constructors_do_not_alias
Edge(index, omega_body,      omega_body, f_body,   test_traj_navtriple_and_edge_fields,
     f_body, dt, n_steps)    dt, n_steps           test_f18_constructors_do_not_alias
Edge.duration                dt, n_steps           test_traj_navtriple_and_edge_fields
Edge.gyro_stream             omega_body, n_steps,  test_traj_navtriple_and_edge_fields
                             [B_G module const]    (B_G: test_traj_streams_read_module_bias)
Edge.accel_stream            f_body, n_steps,      test_traj_navtriple_and_edge_fields
                             [B_A module const]    (B_A: test_traj_streams_read_module_bias)
Edge.exact_increment         omega_body, f_body,   test_traj_navtriple_and_edge_fields
                             dt, n_steps
edges(n_steps)               n_steps               test_traj_edges_exact_states_streams
initial_state()              [R_0, V_0, P_0]       test_traj_initial_state_and_chain
chain(state, dR, dv, dp, Dt) state, dR, dv, dp,    test_traj_initial_state_and_chain
                             Dt, [G_W module]
exact_states(n_steps)        n_steps               test_traj_edges_exact_states_streams
measurement_streams(n_steps) n_steps               test_traj_edges_exact_states_streams
time_varying_stream(n, dt)   n_steps, dt           test_traj_time_varying_stream

src/preintegration_forster.py
-----------------------------
so3_right_jacobian(phi)      phi (both branches)   test_b1_so3_right_jacobian
ForsterPreintegration(...)   bias_hat (b_g, b_a    test_b1_single_step_reads_every_field,
                             each), sigma_g,       test_f18_constructors_do_not_alias
                             sigma_a, dR, dv, dp,
                             dt_total, cov
.b_g / .b_a                  bias_hat halves       test_b1_single_step_reads_every_field
.integrate_measurement       omega, acc, dt        test_b1_single_step_reads_every_field
.has_noise                   sigma_g, sigma_a      test_b1_single_step_reads_every_field
.noise_covariances(dt)       dt, sigma_g, sigma_a  test_b1_noise_covariances
.retract(phi)                phi                   test_b1_single_step_reads_every_field
._propagate_covariance       (via integrate_*)     test_b1_single_step_reads_every_field,
                                                   test_b1_covariance_over_a_stream
.predict(state_i, g_W)       state_i, g_W,         test_b1_predict
                             dt_total (Dt), dR/dv/dp
.increment()                 dR, dv, dp            test_b1_single_step_reads_every_field
integrate_edge(edge, bias,   edge (dt, n_steps,    test_b1_integrate_edge
  sigma_g, sigma_a, cls,     rates), bias halves,
  **kwargs)                  sigma_g, sigma_a, cls,
                             variant kwarg
integrate_edge_states(edges, edges, bias halves,   test_b1_integrate_edge_states
  bias, state0, g_W, cls,    state0, g_W, cls,
  **kwargs)                  variant kwarg
bias_jacobians_fd(edge,      edge, bias halves,    test_b1_bias_jacobians_fd
  bias, eps, cls, **kwargs)  eps, cls, variant
bias_corrected_increment(    base, jacobians,      test_b1_bias_corrected_increment
  base, jacobians, db)       db_g, db_a
increment_error(a, b)        dR, dv, dp of a, b    test_b1_increment_error

src/preintegration_cayley.py
----------------------------
cay(phi)                     phi                   test_b2_cayley_maps
cayley_vector(phi)           phi (both branches)   test_b2_cayley_maps
cay_right_jacobian(phi)      phi                   test_b2_cayley_maps
cay_denominator_det(xi)      xi                    test_b2_cayley_maps
cay_denominator_det_closed_form(xi)  xi            test_b2_cayley_maps
CayleyPreintegration(...,    variant (+ all of     test_b2_variant_flag,
  variant)                   ForsterPreintegration) test_b1_integrate_edge(_states)
.retract(phi)                phi, variant          test_b2_variant_flag
rotation_from_eq16           R_i, dRbar, xi        test_b2_rotation_residual_functions
rotation_residual_eq19       R_i, R_j, dRbar, xi   test_b2_rotation_residual_functions
rotation_residual_truncated  R_i, R_j, dRbar, xi   test_b2_rotation_residual_functions
rotation_residual_eq22b      R_i, R_j, dRbar, J_R, test_b2_rotation_residual_functions
                             b_g, bbar_g
velocity_residual_eq21b      R_i, v_i, v_j, dvbar, test_b2_eq21b_eq21c
                             Dt, g_W, A_vg, A_va,
                             db_g, db_a (+ None)
position_residual_eq21c      R_i, p_i, p_j, v_i,   test_b2_eq21b_eq21c
                             dpbar, Dt, g_W, A_pg,
                             A_pa, db_g, db_a (+ None)
isotropic_precision(Sigma)   Sigma                 test_b2_precision_and_weighted_square
weighted_square(r, W)        r, W (+ None)         test_b2_precision_and_weighted_square
CayleyFactor(...)            every field           test_b2_factor_methods_read_own_fields,
                                                   test_f18_constructors_do_not_alias
CayleyFactor.from_edge(edge, edge (dt, n_steps,    test_b2_factor_from_edge
  bbar, variant)             rates), bbar halves,
                             variant
.xi(b_g)                     b_g, bbar[0:3], J_R   test_b2_factor_methods_read_own_fields
.predicted_rotation(R_i,b_g) R_i, b_g, bbar,       test_b2_factor_methods_read_own_fields,
                             dRbar, J_R            test_b2_factor_from_edge
.rotation_residual*          R_i, R_j, b_g, fields test_b2_factor_methods_read_own_fields
.velocity_residual(R_i, v_i, R_i, v_i, v_j, g_W,   test_b2_factor_methods_read_own_fields
  v_j, g_W, db)              db halves (+ None),
                             Dt, dvbar, A_v*
.position_residual(...)      ditto (p, Dt, A_p*)   test_b2_factor_methods_read_own_fields

src/preintegration_galilean.py
------------------------------
to_gtsam_bias(bias)          bias halves           test_b3_bias_adapters
from_gtsam_bias(cb)          cb halves             test_b3_bias_adapters
swap_bias_blocks(v)          v halves              test_b3_bias_adapters
gal3_tangent_to_navstate_order(xi)  xi             test_b3_gal3_adapters
gal3(R, p, v, t)             R, p, v, t            test_b3_gal3_adapters
gal3_components(X)           X                     test_b3_gal3_adapters
navstate(state)/navtriple(ns) state / ns           test_b3_gal3_adapters
gal3_from_navstate(ns, t)    ns, t                 test_b3_gal3_adapters
gal3_lift(state, t)          state, t              test_b3_gal3_adapters
make_params(g_W, sigma_g,    g_W, sigma_g, sigma_a,test_b3_make_params_stores_every_input,
  sigma_a, sigma_int,        sigma_int (incl. 0),  test_b3_densities_reach_the_covariance
  combined, sigma_bg,        combined, sigma_bg,
  sigma_ba)                  sigma_ba
GalileanPreintegration(      bias_hat halves,      test_b3_constructor_reads_every_input,
  bias_hat, params, g_W,     params, g_W, sigma_g, test_f18_constructors_do_not_alias
  sigma_g, sigma_a, combined) sigma_a, combined
.integrate_measurement       omega, acc, dt        test_b3_single_step_and_accessors
.dR/.dv/.dp/.dt_total/       (the pim)             test_b3_single_step_and_accessors
  .increment/.increment_gal3
.predict(state_i)            state_i, bias_hat,    test_b3_predict
                             g_W (via params), Dt
.preint_cov()                (the pim)             test_b3_densities_reach_the_covariance
integrate_edge(edge, bias,   edge, bias halves,    test_b3_integrate_edge_states
  **kwargs)                  kwargs (g_W)
integrate_edge_states(edges, edges, bias halves,   test_b3_integrate_edge_states
  bias, state0, **kwargs)    state0, g_W kwarg
integrate_stream(gyro,       gyro, accel, dt,      test_b3_integrate_stream_and_single_step
  accel, dt, bias, n_steps)  bias halves, n_steps
single_step(omega, acc, dt,  omega, acc, dt,       test_b3_integrate_stream_and_single_step
  bias)                      bias halves
endpoint_W(Dt, t_i, g_W)     Dt, t_i, g_W          test_b3_endpoint_relations
endpoint_Hprime(Dt, g_W)     Dt, g_W               test_b3_endpoint_relations
time_shift(dt)               dt                    test_b3_endpoint_relations
endpoint_W_form(X_i, U, Dt,  X_i, U, Dt, t_i, g_W  test_b3_endpoint_relations
  t_i, g_W)
endpoint_F2_form(X_i,U,Dt,g) X_i, U, Dt, g_W       test_b3_endpoint_relations
endpoint_F2_swapped(...)     X_i, U, Dt, g_W       test_b3_endpoint_relations
corrected_increment_gal3(    pim (bias_hat halves, test_b3_corrected_increment_gal3
  pim, db_gtsam, Dt)         dt_total), db halves,
                             Dt (None and given)
bias_jacobian_fd(            increment_at, eps     test_b3_bias_jacobian_fd_reads_eps
  increment_at, eps)
right_composition_model      Uhat, J, db           test_b3_models_and_distance
component_wise_model         Uhat, J, db           test_b3_models_and_distance
gal3_distance(A, B)          A, B                  test_b3_models_and_distance
bias_jacobian_recursion(     gyro, accel, dt,      test_b3_bias_jacobian_recursion
  gyro, accel, dt, bias,     bias halves, n_steps,
  n_steps, eps, **kwargs)    eps (see its docstring)

Notes.
* ``CayleyPreintegration`` has no inputs beyond ``ForsterPreintegration``'s and
  ``variant``; it is exercised through ``integrate_edge(cls=CayleyPreintegration)``.
* The module constants ``SMALL_ANGLE``, ``FD_EPS``, ``J_FD_EPS``, ``C_FD_EPS``,
  ``GAL3_BLOCK_FOR_NAVSTATE`` and the fixture literals are DEFINITIONS (C01 pins the
  literals); the defaults that read them are covered by passing other values explicitly.
* ``bias_jacobian_recursion``'s ``eps`` and its ``**kwargs`` (params / gravity) do not
  change the result measurably: one step's increment is linear in the bias to ~1e-12 at
  any ``eps <= 0.1``, and gravity does not enter an increment.  Hard-coding either is an
  EQUIVALENT mutant; the test runs at two values and checks the reference regardless.

Every test collects its failures and reports all of them, each prefixed by the name of
the function and the input it was varied in.
"""
from __future__ import annotations

import numpy as np
import pytest

import gtsam
from gtsam import Gal3, NavState, Rot3
from gtsam.imuBias import ConstantBias
from gtsam.symbol_shorthand import B, V, X
from scipy.linalg import expm, logm

from src import trajectory as T
from src import preintegration_cayley as B2
from src import preintegration_forster as B1
from src import preintegration_galilean as B3


# ======================================================================================
# Independent references -- nothing below imports or calls ``src``
# ======================================================================================

#: fixture literals, RETYPED (claim M2-C01's text), never read from ``src``
FIX_G = np.array([0.0, 0.0, -9.81])
FIX_BG = np.array([0.0100, -0.0150, 0.0080])
FIX_BA = np.array([0.0600, -0.0400, 0.0900])
FIX_BIAS = np.concatenate([FIX_BG, FIX_BA])
FIX_DT = 2.5e-3


def ihat(v):
    v = np.asarray(v, dtype=float)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def ivee(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def iexp(phi):
    """Rodrigues, retyped (``expm`` is used for the 5x5 references; this is faster)."""
    phi = np.asarray(phi, dtype=float)
    th = float(np.linalg.norm(phi))
    K = ihat(phi)
    if th < 1e-6:
        return np.eye(3) + K + 0.5 * (K @ K) + (K @ K @ K) / 6.0
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th**2) * (K @ K)


def ilog(R):
    """``Log`` on SO(3), retyped; the small-angle branch keeps full relative precision."""
    R = np.asarray(R, dtype=float)
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = float(np.arccos(c))
    w = ivee(R - R.T) / 2.0
    if th < 1e-4:
        return w * (1.0 + th**2 / 6.0)
    return (th / np.sin(th)) * w


def icay(phi):
    """Eq. (3) in its closed polynomial form ``I + 4/(4 + |phi|^2) (K + K^2/2)``."""
    phi = np.asarray(phi, dtype=float)
    K = ihat(phi)
    return np.eye(3) + (4.0 / (4.0 + float(phi @ phi))) * (K + 0.5 * (K @ K))


def ieuler(gyro, accel, dt, bias, retract=iexp, start=None):
    """Eq. (9a)-(9c) Euler sums, retyped, with a pluggable retraction."""
    R, v, p = (np.eye(3), np.zeros(3), np.zeros(3)) if start is None else start
    for omega, acc in zip(gyro, accel):
        w = np.asarray(omega) - bias[0:3]
        a = np.asarray(acc) - bias[3:6]
        p = p + v * dt + 0.5 * (R @ a) * dt**2
        v = v + (R @ a) * dt
        R = R @ retract(w * dt)
    return R, v, p


def ireparam(phi):
    """``Cay(2 tan(theta/2) a)`` -- the Eq. (3) contrast; retyped, equal to ``Exp``."""
    phi = np.asarray(phi, dtype=float)
    th = float(np.linalg.norm(phi))
    return icay((2.0 * np.tan(th / 2.0) / th) * phi if th > 0 else phi)


def izoh(w, a, Tt):
    """Exact constant-input increment from one 5x5 ``expm``."""
    A = np.zeros((5, 5))
    A[0:3, 0:3] = ihat(w)
    A[0:3, 3] = a
    A[3, 4] = 1.0
    Xm = expm(Tt * A)
    return Xm[0:3, 0:3], Xm[0:3, 3], Xm[0:3, 4]


def ichain(state, inc, Dt, g):
    R, v, p = state
    dR, dv, dp = inc
    return (R @ dR, v + g * Dt + R @ dv, p + v * Dt + 0.5 * g * Dt**2 + R @ dp)


def igal3(R, p, v, t):
    """The Gal(3) matrix ``[[R, v, p], [0, 1, t], [0, 0, 1]]`` (claim M2-C14)."""
    M = np.eye(5)
    M[0:3, 0:3] = R
    M[0:3, 3] = v
    M[0:3, 4] = p
    M[3, 4] = t
    return M


def igal3_hat(xi):
    """Tangent ``(omega, nu = velocity, rho = position, alpha = time)`` (claim M2-C15)."""
    A = np.zeros((5, 5))
    A[0:3, 0:3] = ihat(xi[0:3])
    A[0:3, 3] = xi[3:6]
    A[0:3, 4] = xi[6:9]
    A[3, 4] = xi[9]
    return A


def igal3_log(M):
    L = np.real(logm(np.asarray(M, dtype=float)))
    return np.concatenate([ivee(L[0:3, 0:3]), L[0:3, 3], L[0:3, 4], [L[3, 4]]])


def izoh_stream(gyro, accel, dt, bias, n=None):
    """Product of exact per-step ZOH increments as one 5x5 matrix (GTSAM order of
    accumulation, ``U_{k+1} = U_k V_k``; claim M2-C19)."""
    n = len(gyro) if n is None else n
    U = np.eye(5)
    for k in range(n):
        dR, dv, dp = izoh(gyro[k] - bias[0:3], accel[k] - bias[3:6], dt)
        U = U @ igal3(dR, dp, dv, dt)
    return U


def iswap(v):
    return np.concatenate([v[3:6], v[0:3]])


def mabs(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())


def rel(a, b) -> float:
    return mabs(a, b) / float(np.abs(np.asarray(b, dtype=float)).max())


def rot_err(A, B_) -> float:
    return float(np.abs(ilog(np.asarray(A).T @ np.asarray(B_))).max())


def _check(failures, name, err, bound):
    if not err <= bound:
        failures.append(f"{name} ({err:.3e} > {bound:.0e})")


# ======================================================================================
# Non-fixture inputs
# ======================================================================================

#: two edges; neither is 0.5 s long, neither has dt = 2.5e-3 or 200 steps both
EDGE_A = T.Edge(index=0, omega_body=np.array([0.52, -0.11, 0.29]),
                f_body=np.array([0.30, -0.70, 9.40]), dt=1e-3, n_steps=200)
EDGE_B = T.Edge(index=1, omega_body=np.array([-0.18, 0.44, -0.61]),
                f_body=np.array([-0.90, 0.25, 10.30]), dt=5e-3, n_steps=146)
#: their durations, typed as literals (not ``dt * n_steps``)
EDGE_DURATION = {0: 0.2, 1: 0.73}
EDGES = (EDGE_A, EDGE_B)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


G_A = 9.79 * _unit([0.10, -0.20, -1.0])
G_B = 9.83 * _unit([-0.30, 0.15, -0.94])

BIAS_GYRO = np.array([0.031, -0.027, 0.012, 0.0, 0.0, 0.0])
BIAS_ACCEL = np.array([0.0, 0.0, 0.0, -0.08, 0.11, 0.05])
BIAS_BOTH = np.array([-0.020, 0.040, -0.033, 0.13, -0.06, -0.10])
BIASES = {"gyro-half only": BIAS_GYRO, "accel-half only": BIAS_ACCEL, "both halves": BIAS_BOTH}

STATE_A = (iexp([0.30, 0.10, -0.40]), np.array([1.20, -0.40, 0.30]), np.array([3.0, -1.0, 2.0]))
STATE_B = (iexp([-0.70, 0.25, 0.90]), np.array([-0.60, 0.85, -1.10]), np.array([-2.0, 0.5, 4.0]))

#: tolerances.  TIER0 for exact arithmetic paths; the others are measured headroom
TIER0 = 1e-12
EULER_PIN = 1e-12          # retyped Euler sums vs src's (same arithmetic up to the retraction)
JAC_FD_REL = 1e-7          # src's FD (eps 1e-6) vs the reference FD (eps 1e-5), relative
COV_FD_REL = 1e-7          # Eq. (63) vs the reference FD noise linearisation, relative
#: GTSAM's single-source covariance blocks vs sigma^2 Dt I: the reference is first order in
#: the per-step angle (measured 1.5e-8 at dt 1e-3, 9.8e-7 at dt 5e-3); still 5000x below
#: the smallest density error it must see (U40, accel variance x1.005)
GTSAM_PROPAGATED_REL = 2e-6
GETTER_REL = 1e-12         # make_params' stored covariances vs sigma^2 I


def _assert_no_failures(failures):
    assert not failures, "\n".join(failures)


def _rng(seed):
    return np.random.default_rng(seed)


def _rand_rot(rng):
    return iexp(rng.normal(size=3))


# ======================================================================================
# src/trajectory.py
# ======================================================================================


@pytest.mark.claim("M2-F17")
def test_traj_so3_helpers():
    """``hat``, ``vee``, ``so3_exp``, ``so3_log`` at two generic inputs and one inside the
    small-angle branch, against ``cross``, ``expm`` and ``logm``."""
    failures = []
    for phi in (np.array([0.7, -0.2, 1.1]), np.array([-2.0, 0.5, 0.3]),
                np.array([3e-9, -1e-9, 2e-9])):
        b = np.array([0.4, -1.3, 2.2])
        _check(failures, f"hat({phi}) b is not cross(phi, b)", mabs(T.hat(phi) @ b,
                                                                    np.cross(phi, b)), TIER0)
        _check(failures, f"vee(hat({phi})) is not phi", mabs(T.vee(ihat(phi)), phi), TIER0)
        R = iexp(phi)
        _check(failures, f"so3_exp({phi}) is not expm(hat)", mabs(T.so3_exp(phi), R), TIER0)
        _check(failures, f"so3_log(Exp({phi})) is not phi", mabs(T.so3_log(R), phi), 1e-12)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_M1_M2():
    """``M1(w, T)``, ``M2(w, T)`` at two rates and two durations (0.2, 0.73 s), and in the
    small-angle branch, against the 5x5 ``expm`` (columns = unit specific forces)."""
    failures = []
    for w, Tt in ((np.array([0.52, -0.11, 0.29]), 0.2), (np.array([-0.18, 0.44, -0.61]), 0.73),
                  (np.array([2e-9, 1e-9, -3e-9]), 0.37)):
        M1_ref = np.column_stack([izoh(w, e, Tt)[1] for e in np.eye(3)])
        M2_ref = np.column_stack([izoh(w, e, Tt)[2] for e in np.eye(3)])
        _check(failures, f"M1(w={w}, T={Tt}) is not the expm integral",
               mabs(T.M1(w, Tt), M1_ref), TIER0)
        _check(failures, f"M2(w={w}, T={Tt}) is not the expm double integral",
               mabs(T.M2(w, Tt), M2_ref), TIER0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_navtriple_and_edge_fields():
    """``NavTriple`` stores its inputs; ``Edge`` at two (dt, n_steps, rates) that are not
    the fixture's: duration, both streams and the exact increment."""
    failures = []
    for (R, v, p) in (STATE_A, STATE_B):
        nt = T.NavTriple(R=R, v=v, p=p)
        for name, got, want in zip("Rvp", nt.as_tuple(), (R, v, p)):
            if not np.array_equal(got, want):
                failures.append(f"NavTriple.{name} is not its input")
    for edge in EDGES:
        where = f"Edge(dt={edge.dt}, n_steps={edge.n_steps})"
        _check(failures, f"{where}.duration is not the literal {EDGE_DURATION[edge.index]}",
               abs(edge.duration - EDGE_DURATION[edge.index]), 1e-15)
        gyro, accel = edge.gyro_stream(), edge.accel_stream()
        if gyro.shape != (edge.n_steps, 3) or accel.shape != (edge.n_steps, 3):
            failures.append(f"{where} stream shape is {gyro.shape}/{accel.shape}, not "
                            f"({edge.n_steps}, 3)")
        _check(failures, f"{where}.gyro_stream is not omega_body + b_g",
               mabs(gyro, np.tile(edge.omega_body + FIX_BG, (edge.n_steps, 1))), 0.0)
        _check(failures, f"{where}.accel_stream is not f_body + b_a",
               mabs(accel, np.tile(edge.f_body + FIX_BA, (edge.n_steps, 1))), 0.0)
        ref = izoh(edge.omega_body, edge.f_body, EDGE_DURATION[edge.index])
        for name, got, want in zip(("dR", "dv", "dp"), edge.exact_increment(), ref):
            _check(failures, f"{where}.exact_increment {name} is not the ZOH increment",
                   mabs(got, want), TIER0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_streams_read_module_bias(monkeypatch):
    """``gyro_stream`` / ``accel_stream`` read the module constants ``B_G`` / ``B_A``
    (implicit inputs): set them to two other values and the streams must follow."""
    failures = []
    for bias in (BIAS_BOTH, np.array([0.05, 0.02, -0.04, -0.2, 0.3, 0.07])):
        monkeypatch.setattr(T, "B_G", bias[0:3].copy())
        monkeypatch.setattr(T, "B_A", bias[3:6].copy())
        _check(failures, f"Edge.gyro_stream ignores the module B_G = {bias[0:3]}",
               mabs(EDGE_A.gyro_stream()[0], EDGE_A.omega_body + bias[0:3]), 0.0)
        _check(failures, f"Edge.accel_stream ignores the module B_A = {bias[3:6]}",
               mabs(EDGE_A.accel_stream()[0], EDGE_A.f_body + bias[3:6]), 0.0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_edges_exact_states_streams():
    """``edges(n)``, ``exact_states(n)``, ``measurement_streams(n)`` at n = 80 and 333."""
    rates = ((np.array([0.35, -0.22, 0.17]), np.array([0.60, 0.45, 9.95])),
             (np.array([-0.28, 0.41, -0.33]), np.array([-0.75, 0.90, 9.60])))
    s = (iexp([0.10, -0.20, 0.30]), np.array([0.50, -0.30, 0.20]), np.zeros(3))
    ref_states = [s]
    for w, f in rates:
        ref_states.append(ichain(ref_states[-1], izoh(w, f, 0.5), 0.5, FIX_G))
    failures = []
    for n in (80, 333):
        es = T.edges(n)
        for e, (w, f) in zip(es, rates):
            if e.n_steps != n:
                failures.append(f"edges({n}) edge {e.index} has n_steps {e.n_steps}")
            _check(failures, f"edges({n}) edge {e.index} dt is not 0.5/{n}",
                   abs(e.dt - 0.5 / n), 1e-18)
            _check(failures, f"edges({n}) edge {e.index} rates are not the fixture's",
                   max(mabs(e.omega_body, w), mabs(e.f_body, f)), 0.0)
        for k, (got, want) in enumerate(zip(T.exact_states(n), ref_states)):
            for name, g_, w_ in zip("Rvp", got.as_tuple(), want):
                _check(failures, f"exact_states({n}) KF{k} {name}", mabs(g_, w_), TIER0)
        streams = T.measurement_streams(n)
        for k, ((gy, ac), (w, f)) in enumerate(zip(streams, rates)):
            if gy.shape != (n, 3):
                failures.append(f"measurement_streams({n}) edge {k} has shape {gy.shape}")
            _check(failures, f"measurement_streams({n}) edge {k} values",
                   max(mabs(gy, np.tile(w + FIX_BG, (n, 1))),
                       mabs(ac, np.tile(f + FIX_BA, (n, 1)))), 0.0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_initial_state_and_chain(monkeypatch):
    """``chain`` at two (state, increment, Dt) sets with Dt = 0.2 / 0.73, and with its
    implicit module gravity ``G_W`` moved to two other vectors; ``initial_state`` with
    its implicit ``R_0`` / ``V_0`` / ``P_0`` moved."""
    rng = _rng(11)
    failures = []
    for Dt, state in ((0.2, STATE_A), (0.73, STATE_B)):
        inc = (_rand_rot(rng), rng.normal(size=3), rng.normal(size=3))
        got = T.chain(T.NavTriple(*state), *inc, Dt)
        for name, g_, w_ in zip("Rvp", got.as_tuple(), ichain(state, inc, Dt, FIX_G)):
            _check(failures, f"chain(Dt={Dt}) {name}", mabs(g_, w_), TIER0)
        for g in (G_A, G_B):
            monkeypatch.setattr(T, "G_W", g.copy())
            got = T.chain(T.NavTriple(*state), *inc, Dt)
            for name, g_, w_ in zip("Rvp", got.as_tuple(), ichain(state, inc, Dt, g)):
                _check(failures, f"chain ignores the module gravity G_W = {g} ({name})",
                       mabs(g_, w_), TIER0)
            monkeypatch.setattr(T, "G_W", FIX_G.copy())
    for (R, v, p) in (STATE_A, STATE_B):
        monkeypatch.setattr(T, "R_0", R.copy())
        monkeypatch.setattr(T, "V_0", v.copy())
        monkeypatch.setattr(T, "P_0", p.copy())
        s0 = T.initial_state()
        for name, g_, w_ in zip("Rvp", s0.as_tuple(), (R, v, p)):
            _check(failures, f"initial_state ignores the module {name}_0", mabs(g_, w_), 0.0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_traj_time_varying_stream():
    """``time_varying_stream(n, dt)`` at (17, 1e-3) and (55, 4e-3), retyped."""
    failures = []
    phase = np.array([0.0, 1.0, 2.0])
    for n, dt in ((17, 1e-3), (55, 4e-3)):
        gyro, accel, dt_out = T.time_varying_stream(n, dt)
        k = np.arange(n, dtype=float)[:, None]
        a_ref = np.array([0.5, 0.4, 9.81]) + 0.6 * np.sin(0.7 * k * dt + phase)
        w_ref = np.array([0.3, -0.2, 0.15]) + 0.4 * np.cos(1.1 * k * dt + phase)
        if gyro.shape != (n, 3) or dt_out != dt:
            failures.append(f"time_varying_stream({n}, {dt}) shape {gyro.shape}, dt {dt_out}")
        else:
            _check(failures, f"time_varying_stream({n}, {dt}) values",
                   max(mabs(gyro, w_ref), mabs(accel, a_ref)), 1e-15)
    _assert_no_failures(failures)


# ======================================================================================
# src/preintegration_forster.py
# ======================================================================================


@pytest.mark.claim("M2-F17")
def test_b1_so3_right_jacobian():
    """``Jr(phi)`` at two angles and in the small-angle branch, against central
    differences of ``Exp(phi + d) = Exp(phi) Exp(Jr d)`` with ``expm`` / ``logm``."""
    failures = []
    eps = 1e-6
    for phi in (np.array([0.7, -0.2, 1.1]), np.array([-1.9, 0.6, 0.4]),
                np.array([3e-9, -1e-9, 2e-9])):
        R0 = iexp(phi)
        J = np.column_stack([(ilog(R0.T @ iexp(phi + eps * e)) - ilog(R0.T @ iexp(phi - eps * e)))
                             / (2 * eps) for e in np.eye(3)])
        _check(failures, f"so3_right_jacobian({phi})", mabs(B1.so3_right_jacobian(phi), J), 1e-8)
    _assert_no_failures(failures)


def _fd_step_jacobians(state, omega, acc, dt, bias, retract=iexp, eps=1e-6, eps_noise=1e-3):
    """Linearisation of ONE Euler step in Forster's chart (``dR Exp(d)``, additive v, p):
    ``A`` (9x9) w.r.t. the state, ``Bg`` / ``Ba`` (9x3) w.r.t. additive measurement noise.
    The noise step is larger (the noise enters multiplied by dt, and round-off in the
    rotation chart is absolute)."""
    R, v, p = state

    def step(dphi, dv, dp, eg, ea):
        return ieuler([omega + eg], [acc + ea], dt, bias, retract,
                      start=(R @ iexp(dphi), v + dv, p + dp))

    Rn, vn, pn = step(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

    def chart(out):
        return np.concatenate([ilog(Rn.T @ out[0]), out[1] - vn, out[2] - pn])

    cols = []
    for slot in range(5):
        for i in range(3):
            args_p = [np.zeros(3) for _ in range(5)]
            args_m = [np.zeros(3) for _ in range(5)]
            h = eps if slot < 3 else eps_noise
            args_p[slot][i] = h
            args_m[slot][i] = -h
            cols.append((chart(step(*args_p)) - chart(step(*args_m))) / (2 * h))
    J = np.column_stack(cols)
    return J[:, 0:9], J[:, 9:12], J[:, 12:15]


@pytest.mark.claim("M2-F17")
def test_b1_single_step_reads_every_field():
    """One ``integrate_measurement`` from a NON-default starting increment, at two
    configurations: every constructor field (``bias_hat`` halves, ``sigma_g``,
    ``sigma_a``, ``dR``, ``dv``, ``dp``, ``dt_total``, ``cov``) and every argument
    (``omega``, ``acc``, ``dt`` = 1e-3 / 4.7e-3) must reach the result.

    The covariance reference is the FD linearisation of the retyped step
    (:func:`_fd_step_jacobians`), ``A C0 A^T + Bg (sg^2/dt) Bg^T + Ba (sa^2/dt) Ba^T``,
    checked with the noise alone (``C0 = 0``, one source at a time, so a density error
    cannot hide behind the other source or behind ``C0``) and with ``C0`` alone.
    """
    rng = _rng(5)
    failures = []
    configs = ((BIAS_BOTH, 3.1e-4, 7.3e-3, 1e-3, 0.11),
               (BIAS_GYRO, np.array([2e-4, 5e-4, 9e-4]), np.array([4e-3, 1.1e-2, 2.6e-2]),
                4.7e-3, 0.37))
    for bias, sg, sa, dt, t0 in configs:
        for half_bias in (bias, BIAS_ACCEL):
            R0, v0, p0 = _rand_rot(rng), rng.normal(size=3), rng.normal(size=3)
            omega, acc = rng.normal(size=3), rng.normal(size=3) + np.array([0, 0, 9.8])
            where = f"ForsterPreintegration(bias={half_bias}, dt={dt})"
            pim = B1.ForsterPreintegration(bias_hat=half_bias.copy(), dR=R0.copy(),
                                           dv=v0.copy(), dp=p0.copy(), dt_total=t0)
            if pim.has_noise:
                failures.append(f"{where}.has_noise is True with zero densities")
            _check(failures, f"{where}.b_g / b_a are not the bias halves",
                   max(mabs(pim.b_g, half_bias[0:3]), mabs(pim.b_a, half_bias[3:6])), 0.0)
            pim.integrate_measurement(omega, acc, dt)
            ref = ieuler([omega], [acc], dt, half_bias, start=(R0, v0, p0))
            for name, got, want in zip(("dR", "dv", "dp"), pim.increment(), ref):
                _check(failures, f"{where}.integrate_measurement {name} from the given "
                       f"start", mabs(got, want), EULER_PIN)
            _check(failures, f"{where}.dt_total is not t0 + dt", abs(pim.dt_total - (t0 + dt)),
                   1e-15)
            _check(failures, f"{where}.retract is not Exp", mabs(pim.retract(omega * dt),
                                                                 iexp(omega * dt)), TIER0)

        A, Bg, Ba = _fd_step_jacobians((R0, v0, p0), omega, acc, dt, bias)
        sgv = np.broadcast_to(np.asarray(sg, float), (3,))
        sav = np.broadcast_to(np.asarray(sa, float), (3,))
        for label, kw, want in (
                ("gyro noise alone", dict(sigma_g=sg), Bg @ np.diag(sgv**2 / dt) @ Bg.T),
                ("accel noise alone", dict(sigma_a=sa), Ba @ np.diag(sav**2 / dt) @ Ba.T)):
            pim = B1.ForsterPreintegration(bias_hat=bias.copy(), dR=R0.copy(), dv=v0.copy(),
                                           dp=p0.copy(), dt_total=t0, **kw)
            if not pim.has_noise:
                failures.append(f"has_noise is False with {label}")
            pim.integrate_measurement(omega, acc, dt)
            _check(failures, f"Eq. (63) one step, {label} (sigma_g={sg}, sigma_a={sa}, "
                   f"dt={dt}, bias={bias})", rel(pim.cov, want), COV_FD_REL)
        L = rng.normal(size=(9, 9))
        C0 = 1e-4 * (L @ L.T)
        pim = B1.ForsterPreintegration(bias_hat=bias.copy(), dR=R0.copy(), dv=v0.copy(),
                                       dp=p0.copy(), dt_total=t0, cov=C0.copy(),
                                       sigma_g=1e-30, sigma_a=1e-30)
        pim.integrate_measurement(omega, acc, dt)
        _check(failures, f"Eq. (63) one step does not propagate the initial cov (dt={dt})",
               rel(pim.cov, A @ C0 @ A.T), COV_FD_REL)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_noise_covariances():
    """``noise_covariances(dt)`` at two dt and two (scalar, per-axis) density pairs."""
    failures = []
    for sg, sa, dt in ((3.1e-4, 7.3e-3, 1e-3),
                       (np.array([2e-4, 5e-4, 9e-4]), np.array([4e-3, 1.1e-2, 2.6e-2]), 4.7e-3)):
        pim = B1.ForsterPreintegration(bias_hat=np.zeros(6), sigma_g=sg, sigma_a=sa)
        Qg, Qa = pim.noise_covariances(dt)
        sgv = np.broadcast_to(np.asarray(sg, float), (3,))
        sav = np.broadcast_to(np.asarray(sa, float), (3,))
        _check(failures, f"noise_covariances({dt}) Qg with sigma_g={sg}",
               rel(Qg, np.diag(sgv**2) / dt), 1e-15)
        _check(failures, f"noise_covariances({dt}) Qa with sigma_a={sa}",
               rel(Qa, np.diag(sav**2) / dt), 1e-15)
    _assert_no_failures(failures)


def _fd_noise_covariance(gyro, accel, dt, bias, sg, sa, eps=1e-3):
    """First-order covariance of the retyped Euler sums by central differences in every
    per-step noise channel (``Q = sigma^2 / dt``), Forster's chart.  No Eq. (63)."""
    R0, v0, p0 = ieuler(gyro, accel, dt, bias)
    sgv = np.broadcast_to(np.asarray(sg, float), (3,))
    sav = np.broadcast_to(np.asarray(sa, float), (3,))
    q = np.concatenate([sgv**2, sav**2]) / dt
    S = np.zeros((9, 9))
    for k in range(len(gyro)):
        Jk = np.zeros((9, 6))
        for m in range(6):
            outs = []
            for s in (eps, -eps):
                g, a = gyro.copy(), accel.copy()
                if m < 3:
                    g[k, m] += s
                else:
                    a[k, m - 3] += s
                R, v, p = ieuler(g, a, dt, bias)
                outs.append(np.concatenate([ilog(R0.T @ R), v - v0, p - p0]))
            Jk[:, m] = (outs[0] - outs[1]) / (2 * eps)
        S += Jk @ np.diag(q) @ Jk.T
    return S


@pytest.mark.claim("M2-F16")
def test_b1_covariance_over_a_stream():
    """Eq. (63) over a time-varying stream at two (dt, n, bias, densities), gyro != accel,
    one noise source at a time, against :func:`_fd_noise_covariance`."""
    failures = []
    for n, dt, bias, sg, sa in ((14, 1e-3, BIAS_BOTH, 3.1e-4, 7.3e-3),
                                (11, 4.7e-3, BIAS_GYRO, np.array([2e-4, 5e-4, 9e-4]),
                                 np.array([4e-3, 1.1e-2, 2.6e-2]))):
        gyro, accel, _ = T.time_varying_stream(n, dt)
        for label, kw_src, (g_, a_) in (("gyro", dict(sigma_g=sg), (sg, 0.0)),
                                        ("accel", dict(sigma_a=sa), (0.0, sa))):
            pim = B1.ForsterPreintegration(bias_hat=bias.copy(), **kw_src)
            for omega, acc in zip(gyro, accel):
                pim.integrate_measurement(omega, acc, dt)
            want = _fd_noise_covariance(gyro, accel, dt, bias, g_, a_)
            _check(failures, f"Eq. (63) over {n} steps, dt={dt}, {label} noise alone "
                   f"(sigma_g={sg}, sigma_a={sa})", rel(pim.cov, want), COV_FD_REL)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_predict():
    """``predict(state, g_W)`` on the 0.2 s and 0.73 s edges at non-true biases, from
    two non-fixture states and under two non-fixture gravities."""
    failures = []
    for edge, bias, state, g in ((EDGE_A, BIAS_BOTH, STATE_A, G_A),
                                 (EDGE_B, BIAS_ACCEL, STATE_B, G_B)):
        pim = B1.integrate_edge(edge, bias.copy())
        inc = ieuler(edge.gyro_stream(), edge.accel_stream(), edge.dt, bias)
        want = ichain(state, inc, EDGE_DURATION[edge.index], g)
        got = pim.predict(T.NavTriple(*state), g)
        for name, g_, w_ in zip("Rvp", got.as_tuple(), want):
            _check(failures, f"Forster predict on the {EDGE_DURATION[edge.index]} s edge, "
                   f"g={g}: {name}", mabs(g_, w_), 1e-11)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_integrate_edge():
    """``integrate_edge`` on both non-fixture edges, every bias half, both classes and
    both Cayley variants; densities stored as given."""
    failures = []
    cases = ((B1.ForsterPreintegration, {}, iexp), (B2.CayleyPreintegration, {}, icay),
             (B2.CayleyPreintegration, {"variant": B2.VARIANT_REPARAM}, ireparam))
    for edge in EDGES:
        for label, bias in BIASES.items():
            for cls, kw, retract in cases:
                where = (f"integrate_edge({cls.__name__}{kw}, Dt "
                         f"{EDGE_DURATION[edge.index]}, bias {label})")
                pim = B1.integrate_edge(edge, bias.copy(), sigma_g=8.3e-4, sigma_a=2.6e-2,
                                        cls=cls, **kw)
                if type(pim) is not cls:
                    failures.append(f"{where} returned a {type(pim).__name__}")
                if not np.array_equal(pim.bias_hat, bias):
                    failures.append(f"{where}: bias_hat is not its input")
                if not (float(pim.sigma_g) == 8.3e-4 and float(pim.sigma_a) == 2.6e-2):
                    failures.append(f"{where}: densities not stored")
                ref = ieuler(edge.gyro_stream(), edge.accel_stream(), edge.dt, bias, retract)
                for name, got, want in zip(("dR", "dv", "dp"), pim.increment(), ref):
                    _check(failures, f"{where} {name}", mabs(got, want), 1e-11)
                _check(failures, f"{where} dt_total", abs(pim.dt_total -
                                                          EDGE_DURATION[edge.index]), 1e-13)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_integrate_edge_states():
    """``integrate_edge_states`` chained over the two non-fixture edges at three
    (bias, state0, g_W, cls) combinations, none fixture-valued."""
    failures = []
    for bias, state, g, cls, kw, retract in (
            (BIAS_BOTH, STATE_A, G_A, B1.ForsterPreintegration, {}, iexp),
            (BIAS_GYRO, STATE_B, G_B, B2.CayleyPreintegration, {}, icay),
            (BIAS_ACCEL, STATE_A, G_B, B2.CayleyPreintegration,
             {"variant": B2.VARIANT_REPARAM}, ireparam)):
        got = B1.integrate_edge_states(EDGES, bias.copy(), T.NavTriple(*state), g,
                                       cls=cls, **kw)
        want = [state]
        for edge in EDGES:
            inc = ieuler(edge.gyro_stream(), edge.accel_stream(), edge.dt, bias, retract)
            want.append(ichain(want[-1], inc, EDGE_DURATION[edge.index], g))
        for k in range(3):
            for name, g_, w_ in zip("Rvp", got[k].as_tuple(), want[k]):
                _check(failures, f"Baseline 1/2 integrate_edge_states({cls.__name__}{kw}, "
                       f"bias {bias}) KF{k} {name}", mabs(g_, w_), 1e-10)
    _assert_no_failures(failures)


def _ifd_jacobians(edge, bias, eps, retract):
    gyro, accel, dt = edge.gyro_stream(), edge.accel_stream(), edge.dt
    R0 = ieuler(gyro, accel, dt, bias, retract)[0]
    J_R, J_v, J_p = np.zeros((3, 3)), np.zeros((3, 6)), np.zeros((3, 6))
    for m in range(6):
        e = np.zeros(6)
        e[m] = eps
        Rp, vp, pp = ieuler(gyro, accel, dt, bias + e, retract)
        Rm, vm, pm = ieuler(gyro, accel, dt, bias - e, retract)
        J_v[:, m] = (vp - vm) / (2 * eps)
        J_p[:, m] = (pp - pm) / (2 * eps)
        if m < 3:
            J_R[:, m] = (ilog(R0.T @ Rp) - ilog(R0.T @ Rm)) / (2 * eps)
    return J_R, J_v[:, 0:3], J_v[:, 3:6], J_p[:, 0:3], J_p[:, 3:6]


JAC_NAMES = ("J_R", "J_vg", "J_va", "J_pg", "J_pa")


@pytest.mark.claim("M2-F17")
def test_b1_bias_jacobians_fd():
    """``bias_jacobians_fd(edge, bias, eps, cls)`` on the non-fixture edges, at non-true
    biases, at the default eps (vs a reference FD at eps 1e-5) and at two LARGE eps
    (3e-2, 1e-1) where the FD truncation is visible, so ``eps`` must really be used
    (vs a reference FD at the same eps)."""
    failures = []
    for edge, bias, cls, kw, retract in ((EDGE_A, BIAS_BOTH, B1.ForsterPreintegration, {}, iexp),
                                         (EDGE_B, BIAS_GYRO, B2.CayleyPreintegration, {}, icay)):
        where = f"bias_jacobians_fd({cls.__name__}, Dt {EDGE_DURATION[edge.index]})"
        got = B1.bias_jacobians_fd(edge, bias.copy(), cls=cls, **kw)
        ref = _ifd_jacobians(edge, bias, 1e-5, retract)
        for name, g_, r_ in zip(JAC_NAMES, got, ref):
            _check(failures, f"{where} {name} at the default eps", rel(g_, r_), JAC_FD_REL)
        for eps in (3e-2, 1e-1):
            got = B1.bias_jacobians_fd(edge, bias.copy(), eps=eps, cls=cls, **kw)
            ref = _ifd_jacobians(edge, bias, eps, retract)
            small = _ifd_jacobians(edge, bias, 1e-5, retract)
            visible = max(rel(r_, s_) for r_, s_ in zip(ref, small))
            assert visible >= 1e-6, f"eps = {eps} is not visible in the FD ({visible:.2e})"
            for name, g_, r_ in zip(JAC_NAMES, got, ref):
                _check(failures, f"{where} {name} ignores eps = {eps}", rel(g_, r_), 1e-9)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_bias_corrected_increment():
    """Eq. (44) at random increments / Jacobians and a db in each half separately."""
    rng = _rng(7)
    failures = []
    for db in (np.array([0.03, -0.01, 0.02, 0, 0, 0]), np.array([0, 0, 0, -0.05, 0.04, 0.09])):
        base = (_rand_rot(rng), rng.normal(size=3), rng.normal(size=3))
        J = tuple(rng.normal(size=(3, 3)) for _ in range(5))
        got = B1.bias_corrected_increment(base, J, db)
        want = (base[0] @ iexp(J[0] @ db[0:3]), base[1] + J[1] @ db[0:3] + J[2] @ db[3:6],
                base[2] + J[3] @ db[0:3] + J[4] @ db[3:6])
        for name, g_, w_ in zip(("dR", "dv", "dp"), got, want):
            _check(failures, f"bias_corrected_increment {name} at db = {db}", mabs(g_, w_),
                   TIER0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b1_increment_error():
    """``increment_error`` with each of dR, dv, dp in turn the dominant difference."""
    failures = []
    R = iexp([0.2, 0.1, -0.3])
    base = (R, np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.5, 0.2]))
    for k, (other, want) in enumerate((
            ((R @ iexp([0.0, 0.07, 0.0]), base[1], base[2]), 0.07),
            ((R, base[1] + np.array([0, 0, -0.11]), base[2]), 0.11),
            ((R, base[1], base[2] + np.array([0.13, 0, 0])), 0.13))):
        _check(failures, f"increment_error with component {k} dominant",
               abs(B1.increment_error(other, base) - want), 1e-13)
    _assert_no_failures(failures)


# ======================================================================================
# src/preintegration_cayley.py
# ======================================================================================


@pytest.mark.claim("M2-F17")
def test_b2_cayley_maps():
    """``cay``, ``cayley_vector`` (both branches), ``cay_right_jacobian`` (vs Eq. (4)
    retyped and vs FD), and both denominator functions, at two generic inputs."""
    failures = []
    eps = 1e-6
    for phi in (np.array([0.7, -0.2, 1.1]), np.array([-1.9, 0.6, 0.4])):
        _check(failures, f"cay({phi})", mabs(B2.cay(phi), icay(phi)), TIER0)
        th = np.linalg.norm(phi)
        _check(failures, f"cayley_vector({phi})",
               mabs(B2.cayley_vector(phi), 2 * np.tan(th / 2) / th * phi), TIER0)
        J4 = (np.eye(3) - ihat(phi) / 2) / (1 + phi @ phi / 4)
        C0 = icay(phi)
        Jfd = np.column_stack([(ilog(C0.T @ icay(phi + eps * e)) - ilog(C0.T @ icay(phi - eps * e)))
                               / (2 * eps) for e in np.eye(3)])
        _check(failures, f"cay_right_jacobian({phi}) vs Eq. (4)",
               mabs(B2.cay_right_jacobian(phi), J4), TIER0)
        _check(failures, f"cay_right_jacobian({phi}) vs FD",
               mabs(B2.cay_right_jacobian(phi), Jfd), 1e-8)
        _check(failures, f"cay_denominator_det({phi})",
               abs(B2.cay_denominator_det(phi) - np.linalg.det(np.eye(3) - ihat(phi) / 2)), 1e-14)
        _check(failures, f"cay_denominator_det_closed_form({phi})",
               abs(B2.cay_denominator_det_closed_form(phi) - (1 + phi @ phi / 4)), 1e-15)
    tiny = np.array([1e-13, -2e-13, 3e-13])
    _check(failures, "cayley_vector small branch", mabs(B2.cayley_vector(tiny), tiny), 1e-25)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b2_variant_flag():
    """``CayleyPreintegration(variant)``: RAW retracts by ``Cay(phi)``, REPARAM by
    ``Exp(phi)``, at two angles; any other variant is refused."""
    failures = []
    for phi in (np.array([0.3, -0.1, 0.2]), np.array([-1.2, 0.8, 0.5])):
        raw = B2.CayleyPreintegration(bias_hat=np.zeros(6))
        rep = B2.CayleyPreintegration(bias_hat=np.zeros(6), variant=B2.VARIANT_REPARAM)
        _check(failures, f"variant raw retract({phi})", mabs(raw.retract(phi), icay(phi)), TIER0)
        _check(failures, f"variant reparam retract({phi})", mabs(rep.retract(phi), iexp(phi)),
               1e-14)
    with pytest.raises(ValueError):
        B2.CayleyPreintegration(bias_hat=np.zeros(6), variant="exp")
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b2_rotation_residual_functions():
    """Eq. (16), (19), the truncation and Eq. (22b) at two random argument sets."""
    rng = _rng(13)
    failures = []
    for trial in range(2):
        R_i, R_j, dRbar = _rand_rot(rng), _rand_rot(rng), _rand_rot(rng)
        xi, J_R = 0.1 * rng.normal(size=3), rng.normal(size=(3, 3))
        b_g, bbar_g = 0.05 * rng.normal(size=3), 0.05 * rng.normal(size=3)
        A = R_i @ dRbar
        _check(failures, f"rotation_from_eq16 (trial {trial})",
               mabs(B2.rotation_from_eq16(R_i, dRbar, xi), A @ icay(xi)), TIER0)
        _check(failures, f"rotation_residual_eq19 (trial {trial})",
               mabs(B2.rotation_residual_eq19(R_i, R_j, dRbar, xi),
                    (R_j - A) - 0.5 * (R_j + A) @ ihat(xi)), TIER0)
        _check(failures, f"rotation_residual_truncated (trial {trial})",
               mabs(B2.rotation_residual_truncated(R_i, R_j, dRbar, xi),
                    R_j - A @ (np.eye(3) + ihat(xi))), TIER0)
        G = ihat(J_R @ bbar_g)
        Xi = ihat(J_R @ b_g)
        want = (R_j @ (np.eye(3) + G / 2) - R_i @ dRbar @ (np.eye(3) - G / 2)
                - 0.5 * (R_i @ dRbar @ Xi + R_j @ Xi))
        _check(failures, f"rotation_residual_eq22b (trial {trial})",
               mabs(B2.rotation_residual_eq22b(R_i, R_j, dRbar, J_R, b_g, bbar_g), want), TIER0)
    _assert_no_failures(failures)


def _ieq21b(R_i, v_i, v_j, dvbar, Dt, g, A_vg, A_va, db):
    return R_i.T @ (v_j - v_i - g * Dt) - dvbar - A_vg @ db[0:3] - A_va @ db[3:6]


def _ieq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A_pg, A_pa, db):
    return (R_i.T @ (p_j - p_i - v_i * Dt - 0.5 * g * Dt**2) - dpbar
            - A_pg @ db[0:3] - A_pa @ db[3:6])


@pytest.mark.claim("M2-F17")
def test_b2_eq21b_eq21c():
    """Eq. (21b) / (21c) at Dt = 0.2 / 0.73, gravity G_A / G_B, db in each half, and
    with the optional bias terms omitted (``None``)."""
    rng = _rng(17)
    failures = []
    for Dt, g, db in ((0.2, G_A, BIAS_GYRO), (0.73, G_B, BIAS_ACCEL)):
        R_i = _rand_rot(rng)
        v_i, v_j, p_i, p_j, dvbar, dpbar = (rng.normal(size=3) for _ in range(6))
        A = [rng.normal(size=(3, 3)) for _ in range(4)]
        where = f"(Dt={Dt}, g={g}, db={db})"
        _check(failures, f"velocity_residual_eq21b {where}",
               mabs(B2.velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g, A[0], A[1],
                                               db[0:3], db[3:6]),
                    _ieq21b(R_i, v_i, v_j, dvbar, Dt, g, A[0], A[1], db)), TIER0)
        _check(failures, f"position_residual_eq21c {where}",
               mabs(B2.position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A[2], A[3],
                                               db[0:3], db[3:6]),
                    _ieq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A[2], A[3], db)), TIER0)
        z = np.zeros((3, 3))
        _check(failures, f"velocity_residual_eq21b without bias terms {where}",
               mabs(B2.velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g),
                    _ieq21b(R_i, v_i, v_j, dvbar, Dt, g, z, z, db)), TIER0)
        _check(failures, f"position_residual_eq21c without bias terms {where}",
               mabs(B2.position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g),
                    _ieq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, z, z, db)), TIER0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b2_precision_and_weighted_square():
    """Eq. (23) ``tau = 3 / tr Sigma`` and ``r' W r`` / ``|r|^2`` at two inputs."""
    rng = _rng(19)
    failures = []
    for _ in range(2):
        L = rng.normal(size=(3, 3))
        S = L @ L.T + 0.1 * np.eye(3)
        r = rng.normal(size=3)
        _check(failures, "isotropic_precision", abs(B2.isotropic_precision(S) - 3 / np.trace(S)),
               1e-14)
        _check(failures, "weighted_square(r)", abs(B2.weighted_square(r) - r @ r), 1e-14)
        _check(failures, "weighted_square(r, W)", abs(B2.weighted_square(r, S) - r @ S @ r),
               1e-13)
    _assert_no_failures(failures)


def _random_factor(rng, Dt, bbar):
    fields = dict(bbar=bbar.copy(), dRbar=_rand_rot(rng), dvbar=rng.normal(size=3),
                  dpbar=rng.normal(size=3), Dt=Dt, J_R=rng.normal(size=(3, 3)),
                  A_vg=rng.normal(size=(3, 3)), A_va=rng.normal(size=(3, 3)),
                  A_pg=rng.normal(size=(3, 3)), A_pa=rng.normal(size=(3, 3)))
    return B2.CayleyFactor(**fields), fields


@pytest.mark.claim("M2-F17")
def test_b2_factor_methods_read_own_fields():
    """A ``CayleyFactor`` built DIRECTLY from random fields (``Dt`` 0.2 / 0.73, ``bbar``
    not the true bias), so that no method can lean on a fixture value: ``xi``,
    ``predicted_rotation``, the three rotation residuals, and the velocity / position
    residuals under two gravities with ``db`` in each half and ``db = None``."""
    rng = _rng(23)
    failures = []
    for Dt, bbar, g, b_g, db in ((0.2, BIAS_BOTH, G_A, np.array([0.07, -0.02, 0.05]), BIAS_GYRO),
                                 (0.73, BIAS_ACCEL + BIAS_GYRO, G_B,
                                  np.array([-0.04, 0.09, 0.01]), BIAS_ACCEL)):
        f, F = _random_factor(rng, Dt, bbar)
        R_i, R_j = _rand_rot(rng), _rand_rot(rng)
        v_i, v_j, p_i, p_j = (rng.normal(size=3) for _ in range(4))
        where = f"CayleyFactor(Dt={Dt}, bbar={bbar})"
        xi = F["J_R"] @ (b_g - bbar[0:3])
        A = R_i @ F["dRbar"]
        _check(failures, f"{where}.xi", mabs(f.xi(b_g), xi), TIER0)
        _check(failures, f"{where}.predicted_rotation ignores its bbar/fields",
               mabs(f.predicted_rotation(R_i, b_g), A @ icay(xi)), TIER0)
        _check(failures, f"{where}.rotation_residual",
               mabs(f.rotation_residual(R_i, R_j, b_g), (R_j - A) - 0.5 * (R_j + A) @ ihat(xi)),
               TIER0)
        _check(failures, f"{where}.rotation_residual_truncated",
               mabs(f.rotation_residual_truncated(R_i, R_j, b_g), R_j - A @ (np.eye(3) + ihat(xi))),
               TIER0)
        G = ihat(F["J_R"] @ bbar[0:3])
        Xi = ihat(F["J_R"] @ b_g)
        want = (R_j @ (np.eye(3) + G / 2) - A @ (np.eye(3) - G / 2) - 0.5 * (A @ Xi + R_j @ Xi))
        _check(failures, f"{where}.rotation_residual_affine",
               mabs(f.rotation_residual_affine(R_i, R_j, b_g), want), TIER0)
        for dbx in (db, None):
            dbv = np.zeros(6) if dbx is None else dbx
            _check(failures, f"{where}.velocity_residual (g={g}, db={dbx}) does not use "
                   f"its Dt / g_W / db",
                   mabs(f.velocity_residual(R_i, v_i, v_j, g, db=dbx),
                        _ieq21b(R_i, v_i, v_j, F["dvbar"], Dt, g, F["A_vg"], F["A_va"], dbv)),
                   TIER0)
            _check(failures, f"{where}.position_residual (g={g}, db={dbx}) does not use "
                   f"its Dt / g_W / db",
                   mabs(f.position_residual(R_i, p_i, p_j, v_i, g, db=dbx),
                        _ieq21c(R_i, p_i, p_j, v_i, F["dpbar"], Dt, g, F["A_pg"], F["A_pa"],
                                dbv)), TIER0)
    _assert_no_failures(failures)


FACTOR_JAC_NAMES = ("J_R", "A_vg", "A_va", "A_pg", "A_pa")


@pytest.mark.claim("M2-F17")
def test_b2_factor_from_edge():
    """``CayleyFactor.from_edge(edge, bbar, variant)`` on the 0.2 s / 0.73 s edges, at a
    non-true bbar, for both variants: every stored field against the retyped sums (Cay for
    RAW, ``Cay(2 tan(theta/2) a)`` for REPARAM) and the reference FD Jacobians, ``Dt``
    against the literal duration, ``variant`` stored; then ``predicted_rotation`` at a
    gyro bias away from ``bbar`` against ``R_i dRbar_ref Cay(J_ref (b_g - bbar_g))``."""
    rng = _rng(29)
    failures = []
    for edge, bbar in ((EDGE_A, BIAS_BOTH), (EDGE_B, BIAS_GYRO + BIAS_ACCEL)):
        gyro, accel, dt = edge.gyro_stream(), edge.accel_stream(), edge.dt
        for variant, retract in ((B2.VARIANT_RAW, icay), (B2.VARIANT_REPARAM, ireparam)):
            where = f"from_edge(Dt {EDGE_DURATION[edge.index]}, bbar {bbar}, {variant})"
            f = B2.CayleyFactor.from_edge(edge, bbar.copy(), variant=variant)
            if not np.array_equal(f.bbar, bbar):
                failures.append(f"{where}: bbar is not its input")
            if f.variant != variant:
                failures.append(f"{where}: variant {f.variant!r} stored")
            ref = ieuler(gyro, accel, dt, bbar, retract)
            for name, got, want in zip(("dRbar", "dvbar", "dpbar"), (f.dRbar, f.dvbar, f.dpbar),
                                       ref):
                _check(failures, f"{where}: {name}", mabs(got, want), 1e-11)
            _check(failures, f"{where}: Dt is not the literal edge duration",
                   abs(f.Dt - EDGE_DURATION[edge.index]), 1e-13)
            jref = _ifd_jacobians(edge, bbar, 1e-5, retract)
            for name, want in zip(FACTOR_JAC_NAMES, jref):
                _check(failures, f"{where}: {name}", rel(getattr(f, name), want), JAC_FD_REL)
            R_i = _rand_rot(rng)
            b_g = bbar[0:3] + np.array([0.02, -0.015, 0.01])
            want = R_i @ ref[0] @ icay(jref[0] @ (b_g - bbar[0:3]))
            _check(failures, f"{where}: predicted_rotation at b_g != bbar_g",
                   mabs(f.predicted_rotation(R_i, b_g), want), 1e-10)
    _assert_no_failures(failures)


# ======================================================================================
# src/preintegration_galilean.py
# ======================================================================================


@pytest.mark.claim("M2-F17")
def test_b3_bias_adapters():
    """``to_gtsam_bias`` / ``from_gtsam_bias`` / ``swap_bias_blocks`` at two biases with
    distinct halves, against ``ConstantBias``'s own accessors."""
    failures = []
    for b in (BIAS_BOTH, np.array([0.05, 0.02, -0.04, -0.2, 0.3, 0.07])):
        cb = B3.to_gtsam_bias(b)
        _check(failures, f"to_gtsam_bias({b})",
               max(mabs(cb.accelerometer(), b[3:6]), mabs(cb.gyroscope(), b[0:3])), 0.0)
        _check(failures, f"from_gtsam_bias(ConstantBias(a, g)) for {b}",
               mabs(B3.from_gtsam_bias(ConstantBias(b[3:6], b[0:3])), b), 0.0)
        _check(failures, f"swap_bias_blocks({b})", mabs(B3.swap_bias_blocks(b), iswap(b)), 0.0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_gal3_adapters():
    """``gal3``, ``gal3_components``, ``navstate`` / ``navtriple``, ``gal3_from_navstate``,
    ``gal3_lift`` and the tangent remap at two states and two clock values."""
    rng = _rng(31)
    failures = []
    for (R, v, p), t in ((STATE_A, 1.37), (STATE_B, -0.42)):
        Xg = B3.gal3(R, p, v, t)
        _check(failures, f"gal3(R, p, v, t={t}) matrix", mabs(Xg.matrix(), igal3(R, p, v, t)),
               1e-15)
        comps = B3.gal3_components(Xg)
        _check(failures, f"gal3_components (t={t})",
               max(mabs(comps[0], R), mabs(comps[1], p), mabs(comps[2], v), abs(comps[3] - t)),
               1e-15)
        ns = B3.navstate(T.NavTriple(R=R, v=v, p=p))
        _check(failures, "navstate(state)", max(mabs(ns.attitude().matrix(), R),
                                                mabs(ns.position(), p), mabs(ns.velocity(), v)),
               1e-15)
        back = B3.navtriple(NavState(Rot3(R), p, v))
        _check(failures, "navtriple(ns)", max(mabs(back.R, R), mabs(back.v, v),
                                              mabs(back.p, p)), 1e-15)
        _check(failures, f"gal3_from_navstate(ns, t={t})",
               mabs(B3.gal3_from_navstate(NavState(Rot3(R), p, v), t).matrix(),
                    igal3(R, p, v, t)), 1e-15)
        _check(failures, f"gal3_lift(state, t={t})",
               mabs(B3.gal3_lift(T.NavTriple(R=R, v=v, p=p), t).matrix(), igal3(R, p, v, t)),
               1e-15)
        xi = rng.normal(size=10)
        _check(failures, "gal3_tangent_to_navstate_order",
               mabs(B3.gal3_tangent_to_navstate_order(xi),
                    np.concatenate([xi[0:3], xi[6:9], xi[3:6]])), 0.0)
    _assert_no_failures(failures)


#: two density sets; gyro != accel, every entry != the c06 / c25b fixture densities
DENSITIES = ({"sigma_g": 3.1e-4, "sigma_a": 7.3e-3, "sigma_int": 2.2e-3,
              "sigma_bg": 3e-5, "sigma_ba": 2e-4},
             {"sigma_g": 8.3e-4, "sigma_a": 2.6e-2, "sigma_int": 4.0e-4,
              "sigma_bg": 7e-6, "sigma_ba": 6e-4})


@pytest.mark.claim("M2-F16")
def test_b3_make_params_stores_every_input():
    """``make_params`` at two density sets, both ``combined`` values and two gravities:
    each stored covariance read back with GTSAM's own getters must equal ``sigma^2 I``
    to ``1e-12`` relative, ``n_gravity`` must be the given vector, and the params class
    must follow ``combined``."""
    failures = []
    for d, g in zip(DENSITIES, (G_A, G_B)):
        for combined in (False, True):
            where = f"make_params(g={g}, combined={combined}, {d})"
            kw = dict(d) if combined else {k: d[k] for k in ("sigma_g", "sigma_a", "sigma_int")}
            p = B3.make_params(g, combined=combined, **kw)
            want_cls = gtsam.PreintegrationCombinedParams if combined else gtsam.PreintegrationParams
            if type(p) is not want_cls:
                failures.append(f"{where}: params class {type(p).__name__}")
            _check(failures, f"{where}: n_gravity", mabs(p.n_gravity, g), 0.0)
            getters = [("gyroscope", p.getGyroscopeCovariance, "sigma_g"),
                       ("accelerometer", p.getAccelerometerCovariance, "sigma_a"),
                       ("integration", p.getIntegrationCovariance, "sigma_int")]
            if combined:
                getters += [("bias-omega", p.getBiasOmegaCovariance, "sigma_bg"),
                            ("bias-acc", p.getBiasAccCovariance, "sigma_ba")]
            for name, get, key in getters:
                _check(failures, f"{where}: stored {name} covariance is not {key}^2 I",
                       rel(np.asarray(get()), d[key]**2 * np.eye(3)), GETTER_REL)
    # sigma_int = 0 must store zero, whatever the other densities are
    for d in DENSITIES:
        p = B3.make_params(G_A, sigma_g=d["sigma_g"], sigma_a=d["sigma_a"], sigma_int=0.0)
        _check(failures, f"make_params(sigma_int=0) stores a nonzero integration covariance",
               float(np.abs(np.asarray(p.getIntegrationCovariance())).max()), 0.0)
    _assert_no_failures(failures)


def _b3_cov(edge, params, combined=False, bias=BIAS_BOTH):
    pim = (gtsam.PreintegratedCombinedMeasurementsG if combined
           else gtsam.PreintegratedImuMeasurementsG)(params, ConstantBias(bias[3:6], bias[0:3]))
    for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
        pim.integrateMeasurement(acc, omega, edge.dt)
    return pim


@pytest.mark.claim("M2-F16")
def test_b3_densities_reach_the_covariance():
    """Each density of ``make_params`` ALONE, at two values, on the 0.2 s and 0.73 s edges,
    through GTSAM's propagated covariance (``preintMeasCov`` order: rot, pos, vel):

    * gyro alone: rotation block ``sigma_g^2 Dt I`` (white rotation noise, isotropic, is
      carried by rotations only; the deviation is the right Jacobian, measured <= 2e-8);
    * accel alone: velocity block ``sigma_a^2 Dt I`` (same argument, measured <= 2e-8);
    * integration alone: covariance EXACTLY ``sigma_int^2 Dt I`` on position, zero
      elsewhere; and every density zero (``sigma_int = 0``) gives exactly zero;
    * random walks: the combined factor's bias blocks ``sigma_ba^2 Dt I`` at [9:12] and
      ``sigma_bg^2 Dt I`` at [12:15].
    """
    failures = []
    for edge, d in zip(EDGES, DENSITIES):
        Dt = EDGE_DURATION[edge.index]
        I3 = np.eye(3)
        where = f"Dt {Dt}"
        C = np.asarray(_b3_cov(edge, B3.make_params(G_A, sigma_g=d["sigma_g"])).preintMeasCov())
        _check(failures, f"{where}: gyro-only rotation block is not sigma_g^2 Dt I "
               f"(sigma_g={d['sigma_g']})", rel(C[0:3, 0:3], d["sigma_g"]**2 * Dt * I3),
               GTSAM_PROPAGATED_REL)
        C = np.asarray(_b3_cov(edge, B3.make_params(G_A, sigma_a=d["sigma_a"])).preintMeasCov())
        _check(failures, f"{where}: accel-only velocity block is not sigma_a^2 Dt I "
               f"(sigma_a={d['sigma_a']})", rel(C[6:9, 6:9], d["sigma_a"]**2 * Dt * I3),
               GTSAM_PROPAGATED_REL)
        _check(failures, f"{where}: accel-only rotation block is not zero",
               float(np.abs(C[0:3, 0:3]).max()), 0.0)
        for combined in (False, True):
            C = np.asarray(_b3_cov(edge, B3.make_params(G_A, sigma_int=d["sigma_int"],
                                                        combined=combined),
                                   combined).preintMeasCov())
            want = np.zeros_like(C)
            want[3:6, 3:6] = d["sigma_int"]**2 * Dt * I3
            _check(failures, f"{where}: integration-only covariance is not sigma_int^2 Dt I "
                   f"on position (sigma_int={d['sigma_int']}, combined={combined})",
                   rel(C, want), 1e-9)
            C0 = np.asarray(_b3_cov(edge, B3.make_params(G_A, sigma_int=0.0, combined=combined),
                                    combined).preintMeasCov())
            _check(failures, f"{where}: sigma_int = 0 (all densities 0) gives a nonzero "
                   f"covariance (combined={combined})", float(np.abs(C0).max()), 0.0)
        params = B3.make_params(G_A, combined=True, sigma_bg=d["sigma_bg"], sigma_ba=d["sigma_ba"])
        pim = _b3_cov(edge, params, combined=True)
        cov = np.asarray(gtsam.GalileanCombinedImuFactor(X(0), V(0), X(1), V(1), B(0), B(1),
                                                         pim).noiseModel().covariance())
        _check(failures, f"{where}: accel random-walk block is not sigma_ba^2 Dt I "
               f"(sigma_ba={d['sigma_ba']})", rel(cov[9:12, 9:12], d["sigma_ba"]**2 * Dt * I3),
               1e-9)
        _check(failures, f"{where}: gyro random-walk block is not sigma_bg^2 Dt I "
               f"(sigma_bg={d['sigma_bg']})", rel(cov[12:15, 12:15], d["sigma_bg"]**2 * Dt * I3),
               1e-9)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_constructor_reads_every_input():
    """``GalileanPreintegration(bias_hat, params, g_W, sigma_g, sigma_a, combined)``:
    each bias half reaches GTSAM's ``biasHat``; a given ``params`` object is used as is;
    otherwise ``g_W`` / ``sigma_g`` / ``sigma_a`` reach the params; ``combined`` picks the
    PIM class."""
    failures = []
    for (label, b), g, d in zip(list(BIASES.items())[:2], (G_A, G_B), DENSITIES):
        for combined in (False, True):
            where = f"GalileanPreintegration(bias {label}, combined={combined})"
            gal = B3.GalileanPreintegration(b.copy(), g_W=g, sigma_g=d["sigma_g"],
                                            sigma_a=d["sigma_a"], combined=combined)
            _check(failures, f"{where}: biasHat is not (bbar_a, bbar_g)",
                   mabs(np.asarray(gal.pim.biasHat().vector()), iswap(b)), 0.0)
            want_cls = (gtsam.PreintegratedCombinedMeasurementsG if combined
                        else gtsam.PreintegratedImuMeasurementsG)
            if type(gal.pim) is not want_cls or gal.combined is not combined:
                failures.append(f"{where}: PIM class {type(gal.pim).__name__}")
            _check(failures, f"{where}: g_W did not reach the params",
                   mabs(gal.params.n_gravity, g), 0.0)
            _check(failures, f"{where}: sigma_g did not reach the params",
                   rel(np.asarray(gal.params.getGyroscopeCovariance()), d["sigma_g"]**2 * np.eye(3)),
                   GETTER_REL)
            _check(failures, f"{where}: sigma_a did not reach the params",
                   rel(np.asarray(gal.params.getAccelerometerCovariance()),
                       d["sigma_a"]**2 * np.eye(3)), GETTER_REL)
        own = gtsam.PreintegrationParams(G_B)
        gal = B3.GalileanPreintegration(b.copy(), params=own, g_W=G_A)
        if gal.params is not own:
            failures.append("GalileanPreintegration ignores a given params object")
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_single_step_and_accessors():
    """``integrate_measurement(omega, acc, dt)`` at dt = 1e-3 / 4.7e-3 and two non-true
    biases; ``dR`` / ``dv`` / ``dp`` / ``dt_total`` / ``increment`` / ``increment_gal3``
    against the 5x5 ``expm`` step."""
    failures = []
    for dt, b, omega, acc in ((1e-3, BIAS_BOTH, np.array([0.4, -0.3, 0.9]),
                               np.array([0.2, 0.1, 9.7])),
                              (4.7e-3, BIAS_GYRO, np.array([-1.1, 0.2, 0.3]),
                               np.array([-0.6, 0.8, 10.1]))):
        gal = B3.GalileanPreintegration(b.copy())
        gal.integrate_measurement(omega, acc, dt)
        dR, dv, dp = izoh(omega - b[0:3], acc - b[3:6], dt)
        where = f"Baseline 3 one step (dt={dt}, bias={b})"
        for name, got, want in (("dR", gal.dR, dR), ("dv", gal.dv, dv), ("dp", gal.dp, dp)):
            _check(failures, f"{where}: {name}", mabs(got, want), 1e-14)
        _check(failures, f"{where}: dt_total", abs(gal.dt_total - dt), 1e-18)
        _check(failures, f"{where}: increment()", max(mabs(x, y) for x, y in
                                                     zip(gal.increment(), (dR, dv, dp))), 1e-14)
        _check(failures, f"{where}: increment_gal3", mabs(gal.increment_gal3().matrix(),
                                                          igal3(dR, dp, dv, dt)), 1e-14)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_predict():
    """``predict(state)`` at the INTEGRATION bias, which is not the true bias, on the 0.2 s
    and 0.73 s edges, from two states, under two gravities (``g_W`` of the constructor):
    against the exact ZOH increment at ``(omega - bbar_g, a - bbar_a)`` chained with the
    same gravity."""
    failures = []
    for edge, b, state, g in ((EDGE_A, BIAS_BOTH, STATE_A, G_A),
                              (EDGE_B, BIAS_ACCEL + 0.5 * BIAS_GYRO, STATE_B, G_B)):
        gal = B3.integrate_edge(edge, b.copy(), g_W=g)
        Dt = EDGE_DURATION[edge.index]
        inc = izoh(edge.gyro_stream()[0] - b[0:3], edge.accel_stream()[0] - b[3:6], Dt)
        want = ichain(state, inc, Dt, g)
        got = gal.predict(T.NavTriple(*state))
        for name, g_, w_ in zip("Rvp", got.as_tuple(), want):
            _check(failures, f"Baseline 3 predict at bias_hat {b} (Dt {Dt}, g {g}): {name}",
                   mabs(g_, w_), 1e-11)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_integrate_edge_states():
    """``integrate_edge(_states)`` over the two non-fixture edges at two (bias, state0,
    g_W) combinations."""
    failures = []
    for b, state, g in ((BIAS_GYRO, STATE_A, G_B), (BIAS_BOTH, STATE_B, G_A)):
        got = B3.integrate_edge_states(EDGES, b.copy(), T.NavTriple(*state), g_W=g)
        want = [state]
        for edge in EDGES:
            Dt = EDGE_DURATION[edge.index]
            inc = izoh(edge.gyro_stream()[0] - b[0:3], edge.accel_stream()[0] - b[3:6], Dt)
            want.append(ichain(want[-1], inc, Dt, g))
        for k in range(3):
            for name, g_, w_ in zip("Rvp", got[k].as_tuple(), want[k]):
                _check(failures, f"Baseline 3 integrate_edge_states(bias {b}) KF{k} {name}",
                       mabs(g_, w_), 1e-10)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_integrate_stream_and_single_step():
    """``integrate_stream`` (``n_steps`` given and ``None``) and ``single_step`` on a
    time-varying stream at dt = 1.7e-3 / 4.1e-3 and two non-true biases."""
    failures = []
    for n, dt, b, part in ((23, 1.7e-3, BIAS_BOTH, 9), (17, 4.1e-3, BIAS_ACCEL, 13)):
        gyro, accel, _ = T.time_varying_stream(n, dt)
        for n_steps, n_eff in ((None, n), (part, part)):
            gal = B3.integrate_stream(gyro, accel, dt, b.copy(), n_steps=n_steps)
            _check(failures, f"integrate_stream(dt={dt}, bias={b}, n_steps={n_steps})",
                   mabs(gal.increment_gal3().matrix(), izoh_stream(gyro, accel, dt, b, n_eff)),
                   1e-13)
        for k in (0, n - 1):
            dR, dv, dp = izoh(gyro[k] - b[0:3], accel[k] - b[3:6], dt)
            _check(failures, f"single_step(dt={dt}, bias={b}) at k={k}",
                   mabs(B3.single_step(gyro[k], accel[k], dt, b.copy()).matrix(),
                        igal3(dR, dp, dv, dt)), 1e-14)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_endpoint_relations():
    """``endpoint_W``, ``endpoint_Hprime``, ``time_shift`` and the three endpoint forms at
    two (Dt, t_i, g_W), all non-fixture.  ``endpoint_W_form`` is checked SEMANTICALLY:
    it must reproduce the component chain with the same gravity and advance the clock to
    ``t_i + Dt``; the others against retyped 5x5 products."""
    rng = _rng(37)
    failures = []
    for Dt, t_i, g, state in ((0.2, 1.3, G_A, STATE_A), (0.73, 3.7, G_B, STATE_B)):
        I3 = np.eye(3)
        W = igal3(I3, 0.5 * g * Dt**2 - g * Dt * (t_i + Dt), g * Dt, 0.0)
        H = igal3(I3, -0.5 * g * Dt**2, g * Dt, 0.0)
        S = igal3(I3, np.zeros(3), np.zeros(3), -Dt)
        where = f"(Dt={Dt}, t_i={t_i}, g={g})"
        _check(failures, f"endpoint_W {where}", mabs(B3.endpoint_W(Dt, t_i, g).matrix(), W), 1e-14)
        _check(failures, f"endpoint_Hprime {where}", mabs(B3.endpoint_Hprime(Dt, g).matrix(), H),
               1e-14)
        _check(failures, f"time_shift(-{Dt})", mabs(B3.time_shift(-Dt).matrix(), S), 0.0)
        R, v, p = state
        Xi_m = igal3(R, p, v, t_i)
        inc = (_rand_rot(rng), rng.normal(size=3), rng.normal(size=3))
        U_m = igal3(inc[0], inc[2], inc[1], Dt)
        Xi, U = Gal3(Xi_m), Gal3(U_m)
        Rj, vj, pj = ichain(state, inc, Dt, g)
        _check(failures, f"endpoint_W_form {where} is not the component chain",
               mabs(B3.endpoint_W_form(Xi, U, Dt, t_i, g).matrix(), igal3(Rj, pj, vj, t_i + Dt)),
               1e-12)
        _check(failures, f"endpoint_F2_form {where}",
               mabs(B3.endpoint_F2_form(Xi, U, Dt, g).matrix(), S @ H @ Xi_m @ U_m), 1e-12)
        _check(failures, f"endpoint_F2_swapped {where}",
               mabs(B3.endpoint_F2_swapped(Xi, U, Dt, g).matrix(), H @ S @ Xi_m @ U_m), 1e-12)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_corrected_increment_gal3():
    """``corrected_increment_gal3(pim, db, Dt)`` for a PIM integrated at a non-true bias
    on the 0.2 s and 0.73 s edges:

    * ``db = 0`` gives the exact ZOH increment at the PIM's own ``bias_hat`` (Tier 0),
      with the clock equal to the literal edge duration when ``Dt is None`` and to the
      given value otherwise;
    * a small ``db`` in each half (GTSAM order) agrees with the exact ZOH increment at
      ``bias_hat + swap(db)`` to second order: the gap is below ``1e-6`` at
      ``|db| = 1e-4`` while the increment itself moves by more than ``1e-5``.
    """
    failures = []
    for edge, b in ((EDGE_A, BIAS_BOTH), (EDGE_B, BIAS_GYRO)):
        Dt = EDGE_DURATION[edge.index]
        gal = B3.integrate_edge(edge, b.copy())
        w0, a0 = edge.gyro_stream()[0], edge.accel_stream()[0]
        dR, dv, dp = izoh(w0 - b[0:3], a0 - b[3:6], Dt)
        where = f"corrected_increment_gal3(bias_hat {b}, Dt {Dt})"
        _check(failures, f"{where}, db = 0, is not the increment at bias_hat",
               mabs(B3.corrected_increment_gal3(gal, np.zeros(6)).matrix(),
                    igal3(dR, dp, dv, Dt)), 1e-11)
        _check(failures, f"{where}: clock with Dt = 1.37 given",
               abs(B3.corrected_increment_gal3(gal, np.zeros(6), Dt=1.37).time() - 1.37), 0.0)
        for db_gtsam in (np.array([1e-4, -0.5e-4, 0.8e-4, 0, 0, 0]),
                         np.array([0, 0, 0, -0.7e-4, 1e-4, 0.3e-4])):
            bb = b + iswap(db_gtsam)
            eR, ev, ep = izoh(w0 - bb[0:3], a0 - bb[3:6], Dt)
            got = B3.corrected_increment_gal3(gal, db_gtsam).matrix()
            moved = mabs(igal3(eR, ep, ev, Dt), igal3(dR, dp, dv, Dt))
            assert moved >= 1e-5, f"db = {db_gtsam} does not move the increment ({moved:.2e})"
            _check(failures, f"{where}, db = {db_gtsam} (GTSAM order)",
                   mabs(got, igal3(eR, ep, ev, Dt)), 1e-6)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_bias_jacobian_fd_reads_eps():
    """``bias_jacobian_fd(increment_at, eps)`` on an increment that is visibly nonlinear in
    ``db``, at eps = 3e-2 and 1e-1, against a central FD of :func:`igal3_log` at the SAME
    eps; the two eps give results that differ from each other and from eps = 1e-6."""
    w0, a0, Tt = np.array([0.9, -0.6, 1.3]), np.array([0.4, -2.1, 9.6]), 0.73

    def inc_matrix(db):
        dR, dv, dp = izoh(w0 - 3 * db[3:6], a0 - 3 * db[0:3], Tt)
        return igal3(dR, dp, dv, Tt)

    def increment_at(db):
        return Gal3(inc_matrix(np.asarray(db, dtype=float)))

    def ref(eps):
        U0inv = np.linalg.inv(inc_matrix(np.zeros(6)))
        J = np.zeros((10, 6))
        for m in range(6):
            e = np.zeros(6)
            e[m] = eps
            J[:, m] = (igal3_log(U0inv @ inc_matrix(e)) - igal3_log(U0inv @ inc_matrix(-e))) / (2 * eps)
        return J

    failures = []
    small = ref(1e-6)
    for eps in (3e-2, 1e-1):
        want = ref(eps)
        assert rel(want, small) >= 1e-5, f"eps = {eps} is not visible ({rel(want, small):.2e})"
        _check(failures, f"bias_jacobian_fd ignores eps = {eps}",
               rel(B3.bias_jacobian_fd(increment_at, eps=eps), want), 1e-9)
    _check(failures, "bias_jacobian_fd at the default eps", rel(B3.bias_jacobian_fd(increment_at),
                                                                small), 1e-7)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_models_and_distance():
    """``right_composition_model``, ``component_wise_model`` and ``gal3_distance`` at two
    random (Uhat, J, db)."""
    rng = _rng(41)
    failures = []
    for trial in range(2):
        R, v, p = _rand_rot(rng), rng.normal(size=3), rng.normal(size=3)
        t = float(rng.normal())
        Um = igal3(R, p, v, t)
        U = Gal3(Um)
        J = 0.3 * rng.normal(size=(10, 6))
        db = 0.1 * rng.normal(size=6)
        xi = J @ db
        _check(failures, f"right_composition_model (trial {trial})",
               mabs(B3.right_composition_model(U, J, db).matrix(), Um @ expm(igal3_hat(xi))),
               1e-13)
        _check(failures, f"component_wise_model (trial {trial})",
               mabs(B3.component_wise_model(U, J, db).matrix(),
                    igal3(R @ iexp(xi[0:3]), p + xi[6:9], v + xi[3:6], t)), 1e-13)
        Bm = Um @ expm(igal3_hat(0.2 * rng.normal(size=10)))
        _check(failures, f"gal3_distance (trial {trial})",
               abs(B3.gal3_distance(U, Gal3(Bm)) -
                   np.abs(igal3_log(np.linalg.inv(Um) @ Bm)).max()), 1e-12)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F17")
def test_b3_bias_jacobian_recursion():
    """``bias_jacobian_recursion`` at two non-true biases, dt = 1.7e-3 / 4.1e-3,
    ``n_steps`` shorter than the stream, two eps: every ``J_k`` against an independent
    central FD (eps 1e-6) of ``Log(U_k(b)^-1 U_k(b + swap(db)))``, ``U_k`` the product of
    per-step 5x5 ``expm`` increments (no ``src``, no GTSAM)."""
    failures = []
    for n, dt, b, n_steps, eps in ((12, 1.7e-3, BIAS_BOTH, 9, 1e-6),
                                   (10, 4.1e-3, BIAS_ACCEL + BIAS_GYRO, 7, 1e-4)):
        gyro, accel, _ = T.time_varying_stream(n, dt)
        Js = B3.bias_jacobian_recursion(gyro, accel, dt, b.copy(), n_steps, eps=eps)
        if len(Js) != n_steps:
            failures.append(f"bias_jacobian_recursion returned {len(Js)} J_k, not {n_steps}")
            continue
        h = 1e-6
        for k in (0, n_steps // 2, n_steps - 1):
            U0inv = np.linalg.inv(izoh_stream(gyro, accel, dt, b, k + 1))
            J = np.zeros((10, 6))
            for m in range(6):
                e = np.zeros(6)
                e[m] = h
                J[:, m] = (igal3_log(U0inv @ izoh_stream(gyro, accel, dt, b + iswap(e), k + 1))
                           - igal3_log(U0inv @ izoh_stream(gyro, accel, dt, b - iswap(e), k + 1))
                           ) / (2 * h)
            _check(failures, f"bias_jacobian_recursion(bias {b}, dt {dt}, eps {eps}) J_{k + 1}",
                   rel(Js[k], J), 1e-6)
    _assert_no_failures(failures)


# ======================================================================================
# M2-F18 -- constructors must not alias the caller's arrays
# ======================================================================================


@pytest.mark.claim("M2-F18")
def test_f18_constructors_do_not_alias():
    """Build every ``src`` object from caller arrays, then overwrite the caller's arrays in
    place: no stored field may change (pass 7 measured ``xi`` moving by 7.3e-3 and
    Baseline 3 ``predict`` by 4.2e-2 under exactly this)."""
    rng = _rng(43)
    failures = []

    def snapshot(obj, names):
        return {n: np.array(getattr(obj, n), dtype=float, copy=True) for n in names}

    def compare(label, obj, snap):
        for n, before in snap.items():
            if not np.array_equal(np.asarray(getattr(obj, n), dtype=float), before):
                failures.append(f"{label}.{n} aliases the caller's array")

    # Baseline 1 / 2 preintegrator (all array fields)
    for cls in (B1.ForsterPreintegration, B2.CayleyPreintegration):
        args = dict(bias_hat=BIAS_BOTH.copy(), sigma_g=np.array([2e-4, 5e-4, 9e-4]),
                    sigma_a=np.array([4e-3, 1.1e-2, 2.6e-2]), dR=_rand_rot(rng),
                    dv=rng.normal(size=3), dp=rng.normal(size=3), cov=np.eye(9))
        pim = cls(**args)
        names = ("bias_hat", "sigma_g", "sigma_a", "dR", "dv", "dp", "cov")
        snap = snapshot(pim, names)
        for a in args.values():
            a[...] = 7.0
        compare(cls.__name__, pim, snap)
    b = BIAS_BOTH.copy()
    pim = B1.integrate_edge(EDGE_A, b)
    snap = snapshot(pim, ("bias_hat",))
    b[:] = FIX_BIAS
    compare("integrate_edge(...)", pim, snap)

    # Baseline 2 factor: from_edge and direct construction
    b = BIAS_BOTH.copy()
    f = B2.CayleyFactor.from_edge(EDGE_A, b)
    xi_before = f.xi(FIX_BG)
    b[:] = FIX_BIAS
    if not np.array_equal(f.bbar, BIAS_BOTH):
        failures.append("CayleyFactor.from_edge(edge, b).bbar aliases b")
    if not np.array_equal(f.xi(FIX_BG), xi_before):
        failures.append("CayleyFactor.xi moves when the caller's bbar is overwritten")
    f, fields = _random_factor(rng, 0.73, BIAS_BOTH)
    names = B2.CayleyFactor._ARRAY_FIELDS
    snap = snapshot(f, names)
    for n in names:
        fields[n][...] = 3.0
    compare("CayleyFactor", f, snap)

    # Baseline 3 wrapper
    b = BIAS_BOTH.copy()
    gal = B3.integrate_edge(EDGE_A, b)
    before = gal.predict(T.NavTriple(*STATE_A)).as_tuple()
    b[:] = FIX_BIAS
    if not np.array_equal(gal.bias_hat, BIAS_BOTH):
        failures.append("GalileanPreintegration.bias_hat aliases the caller's array")
    after = gal.predict(T.NavTriple(*STATE_A)).as_tuple()
    if max(mabs(x, y) for x, y in zip(before, after)) != 0.0:
        failures.append("Baseline 3 predict moves when the caller's bias is overwritten")

    # trajectory value objects
    R, v, p = (np.array(x, copy=True) for x in STATE_A)
    nt = T.NavTriple(R=R, v=v, p=p)
    snap = snapshot(nt, "Rvp")
    for a in (R, v, p):
        a[...] = 5.0
    compare("NavTriple", nt, snap)
    w, fb = np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 9.8])
    e = T.Edge(index=0, omega_body=w, f_body=fb, dt=1e-3, n_steps=10)
    snap = snapshot(e, ("omega_body", "f_body"))
    w[:] = 9.0
    fb[:] = 9.0
    compare("Edge", e, snap)
    _assert_no_failures(failures)
