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
                             (+ the cov gate,      test_f23_initial_covariance_propagates_
                             M2-F23)                 at_zero_density
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
* ``bias_jacobian_recursion``'s ``eps`` DOES change the result outside the fixture
  regime: at dt = 0.3 s it moves ``J`` by 1.9e-7 (eps 5e-2) and 3.1e-6 (eps 0.2)
  relative.  The note that stood here ("does not change the result measurably") was
  false (pass 9, W12); ``test_f27_recursion_reads_eps_at_large_steps`` pins it.
* Pass 9 (M2-F26, M2-F27) added: complex-step covariance and bias-Jacobian references
  (``_cs_*``; the covariance bound is now 1e-10, was 1e-7); Baseline 2's covariance with
  gyro noise on (``test_f26_*``); per-axis-zero and 1e-21 densities; ``M1`` / ``M2`` on
  fast twists against ``mpmath`` quadrature; ``navstate`` on a drifted ``R``; and the
  FD-step defaults on high-rate edges (``FD_EPS`` is now 3e-5).
* The ``**kwargs`` of ``single_step`` / ``bias_jacobian_recursion`` are NOT inert, and
  the note that used to stand here ("gravity does not enter an increment", so dropping
  them is equivalent) was false: ``params`` can carry ``body_P_sensor``, which moves
  ``single_step`` by ~1e-3 and the recursion's ``J`` by 30% (pass 8, M2-F21: W13, W14).
  ``test_f21_kwargs_reach_gtsam_body_p_sensor`` checks both against the retyped
  sensor->body correction.
* Pass 8 (M2-F21 - M2-F25) added: ``corrected_increment_gal3(Dt=...)`` checked in R, v
  and p under gravity (``Dt`` overrides the clock ONLY); ``isotropic_precision`` at
  tr Sigma = 4.3e-8 ... 1e-200; ``gal3``'s ``Rot3`` branch; the ``so3_log`` near-pi
  branch (``test_f24_so3_log_near_pi``); the series branches of ``so3_exp``, ``Jr``,
  ``M1``, ``M2``, ``cayley_vector`` and ``so3_log`` against 60-digit ``mpmath``
  (``test_f25_*``); zero-density propagation of an initial ``cov``
  (``test_f23_initial_covariance_propagates_at_zero_density``); and outward aliasing
  (``test_f22_no_outward_aliasing``).

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
JAC_FD_REL = 1e-7          # src's FD (eps FD_EPS = 3e-5) vs the reference FD (eps 1e-5), relative
#: Eq. (63) vs the COMPLEX-STEP linearisation of the retyped noisy Euler step, relative.
#: Pass 9 (M2-F27 a): the old FD reference was good to only ~2.5e-9, so its bound (1e-7)
#: could not see a 4e-9 Ba error (X31) or a 9.5e-9 transition error in B2 (X24).  The
#: complex-step reference is exact to rounding; measured agreement 1.9e-15 - 1.8e-12
#: (block-relative), so 1e-10 keeps a >= 50x margin on the clean code.
COV_CS_REL = 1e-10
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


@pytest.mark.claim("M2-F17")
def test_b1_single_step_reads_every_field():
    """One ``integrate_measurement`` from a NON-default starting increment, at two
    configurations: every constructor field (``bias_hat`` halves, ``sigma_g``,
    ``sigma_a``, ``dR``, ``dv``, ``dp``, ``dt_total``, ``cov``) and every argument
    (``omega``, ``acc``, ``dt`` = 1e-3 / 4.7e-3) must reach the result.

    The covariance reference is the COMPLEX-STEP linearisation of the retyped step
    (:func:`_cs_step_jacobians`), ``A C0 A^T + Bg (sg^2/dt) Bg^T + Ba (sa^2/dt) Ba^T``,
    checked with the noise alone (``C0 = 0``, one source at a time, so a density error
    cannot hide behind the other source or behind ``C0``) and with ``C0`` alone at
    EXACTLY zero densities (M2-F23).
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
            if not np.array_equal(pim.cov, np.zeros((9, 9))):
                failures.append(f"{where}: zero densities and zero cov give a nonzero cov")

        A, Bg, Ba = _cs_step_jacobians((R0, v0, p0), omega, acc, dt, bias)
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
                   f"dt={dt}, bias={bias})", rel(pim.cov, want), COV_CS_REL)
        L = rng.normal(size=(9, 9))
        C0 = 1e-4 * (L @ L.T)
        # C0 alone, at EXACTLY zero densities (M2-F23: the old has_noise gate skipped
        # Eq. (63) here and froze C0; this check used sigma = 1e-30 and could not see it)
        pim = B1.ForsterPreintegration(bias_hat=bias.copy(), dR=R0.copy(), dv=v0.copy(),
                                       dp=p0.copy(), dt_total=t0, cov=C0.copy())
        pim.integrate_measurement(omega, acc, dt)
        _check(failures, f"Eq. (63) one step does not propagate the initial cov at zero "
               f"densities (dt={dt})", rel(pim.cov, A @ C0 @ A.T), COV_CS_REL)
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


@pytest.mark.claim("M2-F16")
def test_b1_covariance_over_a_stream():
    """Eq. (63) over a time-varying stream at two (dt, n, bias, densities), gyro != accel,
    one noise source at a time, against :func:`_cs_covariance` (complex step)."""
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
            want = _cs_covariance(gyro, accel, dt, bias, g_, a_)
            _check(failures, f"Eq. (63) over {n} steps, dt={dt}, {label} noise alone "
                   f"(sigma_g={sg}, sigma_a={sa})", rel(pim.cov, want), COV_CS_REL)
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
    # M2-F21 (W46, U47): realistic and tiny traces.  The fixture's rotation-block trace is
    # ~4e-8; nothing may clamp or regularise tr Sigma at any positive scale.
    for scale in (4.3e-8, 1e-12, 1e-30, 1e-200):
        L = rng.normal(size=(3, 3))
        S = L @ L.T + 0.1 * np.eye(3)
        S = S * (scale / np.trace(S))
        want = 3.0 / (S[0, 0] + S[1, 1] + S[2, 2])
        _check(failures, f"isotropic_precision at tr Sigma = {scale:.1e}",
               abs(B2.isotropic_precision(S) - want) / want, 1e-14)
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
        _check(failures, f"gal3(Rot3(R), p, v, t={t}) matrix (the Rot3 branch)",
               mabs(B3.gal3(Rot3(R), p, v, t).matrix(), igal3(R, p, v, t)), 1e-15)
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
        _check(failures, f"{where}: clock with Dt = None is not the literal edge duration",
               abs(B3.corrected_increment_gal3(gal, np.zeros(6)).time() - Dt), 1e-13)
        _check(failures, f"{where}: clock with Dt = 1.37 given",
               abs(B3.corrected_increment_gal3(gal, np.zeros(6), Dt=1.37).time() - 1.37), 0.0)
        # M2-F21 (W02): ``Dt`` overrides the CLOCK only.  The gravity terms that predict()
        # adds are g * dt_total and g * dt_total^2 / 2 -- the PIM's own duration -- so R, v
        # and p must be the increment whatever Dt is, under two non-zero gravities.
        for g in (G_A, G_B):
            gal_g = B3.integrate_edge(edge, b.copy(), g_W=g)
            _check(failures, f"{where}, g = {g}, Dt = 1.37 given: (R, v, p) is not the "
                   f"increment with clock 1.37",
                   mabs(B3.corrected_increment_gal3(gal_g, np.zeros(6), Dt=1.37).matrix(),
                        igal3(dR, dp, dv, 1.37)), 1e-11)
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


#: every array-valued field of ``CayleyFactor``, typed here (Rai Sec. IV-A's factor data:
#: bbar, the three preintegrated increments and the five frozen Jacobians)
CAYLEY_FACTOR_ARRAY_FIELDS = ("bbar", "dRbar", "dvbar", "dpbar", "J_R",
                              "A_vg", "A_va", "A_pg", "A_pa")


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
    # M2-F22 (W07, W08): the field list is the TEST's, never src's ``_ARRAY_FIELDS``
    names = CAYLEY_FACTOR_ARRAY_FIELDS
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


# ======================================================================================
# Verification pass 8 (docs/verification/stage-b_m2-baselines_pass8.md): M2-F21 - M2-F25
# ======================================================================================


@pytest.mark.claim("M2-F23")
def test_f23_initial_covariance_propagates_at_zero_density():
    """``ForsterPreintegration(cov=C0)`` with sigma_g = sigma_a = 0 EXACTLY must carry C0
    through every step: ``C_n = A_n ... A_1 C0 A_1^T ... A_n^T``, each ``A_k`` the FD
    COMPLEX-STEP linearisation of the retyped Euler step (:func:`_cs_step_jacobians`) along the
    retyped trajectory.  Pass 8 measured the shipped gate leaving ``cov`` bit-identical
    to C0 after 200 steps, 92% from the propagated value."""
    failures = []
    C0 = np.diag(np.arange(1.0, 10.0)) * 1e-4
    gyro_tv, accel_tv, _ = T.time_varying_stream(40, 4.7e-3)
    cases = (("EDGE_A (200 steps, dt 1e-3)", EDGE_A.gyro_stream(), EDGE_A.accel_stream(),
              EDGE_A.dt, BIAS_BOTH),
             ("time-varying stream (40 steps, dt 4.7e-3)", gyro_tv, accel_tv, 4.7e-3,
              BIAS_GYRO))
    for label, gyro, accel, dt, bias in cases:
        for cls in (B1.ForsterPreintegration, B2.CayleyPreintegration):
            retract = iexp if cls is B1.ForsterPreintegration else icay
            cretract = _cexp if cls is B1.ForsterPreintegration else _ccay
            pim = cls(bias_hat=bias.copy(), cov=C0.copy())
            state = (np.eye(3), np.zeros(3), np.zeros(3))
            C = C0.copy()
            for omega, acc in zip(gyro, accel):
                pim.integrate_measurement(omega, acc, dt)
                A, _, _ = _cs_step_jacobians(state, omega, acc, dt, bias, retract=cretract)
                C = A @ C @ A.T
                state = ieuler([omega], [acc], dt, bias, retract, start=state)
            moved = rel(C, C0)
            assert moved >= 1e-2, f"{label}: the reference does not move C0 ({moved:.1e})"
            _check(failures, f"{cls.__name__} at zero densities does not propagate the "
                   f"initial cov over {label}", rel(pim.cov, C), COV_CS_REL)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F24")
def test_f24_so3_log_near_pi():
    """``so3_log`` approaching and AT ``pi``, 300 random axes per angle, against
    ``scipy.spatial.transform.Rotation.as_rotvec`` of the same matrix (quaternion route,
    no ``src``).  At ``theta = pi`` the sign of the axis is not unique, so there the check
    is ``Exp(so3_log(R)) == R`` and ``|so3_log(R)| == pi``.  Pass 8 measured the shipped
    function wrong by 2.7e-3 at ``pi - 1e-6``, 2.6e8 at ``pi - 1e-8`` and by a norm-15
    vector at ``pi`` for 36% of axes.  No floating-point warning may be raised."""
    import warnings
    from scipy.spatial.transform import Rotation

    rng = _rng(53)
    failures = []
    for gap in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0):
        th = np.pi - gap
        worst = 0.0
        for _ in range(300):
            a = _unit(rng.normal(size=3))
            R = Rotation.from_rotvec(th * a).as_matrix()
            # warnings-as-errors around the src call ONLY (pass 9): the scipy reference
            # calls stay outside the filter, so a scipy deprecation warning cannot turn
            # this test into an error attributed to nothing in src
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    got = np.asarray(T.so3_log(R), dtype=float)
                except Warning as w:
                    failures.append(f"so3_log at pi - {gap:.0e} raises {w!r}")
                    break
            if gap == 0.0:
                err = max(mabs(Rotation.from_rotvec(got).as_matrix(), R),
                          abs(float(np.linalg.norm(got)) - np.pi))
            else:
                err = mabs(got, Rotation.from_matrix(R).as_rotvec())
            worst = max(worst, err if np.isfinite(err) else np.inf)
        _check(failures, f"so3_log at theta = pi - {gap:.0e}", worst, 1e-13)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F25: the small-angle branches against high-precision (mpmath) references
# --------------------------------------------------------------------------------------

def _mp():
    import mpmath
    mpmath.mp.dps = 60
    return mpmath


def _mp_hat(mpmath, v):
    return mpmath.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _mp_norm(mpmath, v):
    return mpmath.sqrt(sum(x * x for x in v))


def _mp_to_np(M):
    return np.array([[float(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])


def _mp_exp(mpmath, phi):
    th = _mp_norm(mpmath, phi)
    K = _mp_hat(mpmath, phi)
    return mpmath.eye(3) + (mpmath.sin(th) / th) * K + ((1 - mpmath.cos(th)) / th**2) * (K * K)


def _mp_jr(mpmath, phi):
    th = _mp_norm(mpmath, phi)
    K = _mp_hat(mpmath, phi)
    return (mpmath.eye(3) - ((1 - mpmath.cos(th)) / th**2) * K
            + ((th - mpmath.sin(th)) / th**3) * (K * K))


def _mp_M1_M2(mpmath, w, Tt):
    """``int_0^T Exp(w s) ds`` and its integral, exact in ``mp`` (unit-axis form)."""
    th = _mp_norm(mpmath, w)
    K = _mp_hat(mpmath, [x / th for x in w])
    I3 = mpmath.eye(3)
    M1 = Tt * I3 + ((1 - mpmath.cos(th * Tt)) / th) * K + (Tt - mpmath.sin(th * Tt) / th) * (K * K)
    M2 = (Tt**2 / 2 * I3 + ((Tt - mpmath.sin(th * Tt) / th) / th) * K
          + (Tt**2 / 2 - (1 - mpmath.cos(th * Tt)) / th**2) * (K * K))
    return M1, M2


def _offdiag_rel(got, want):
    """max over OFF-diagonal entries of |got - want| / |want|: the entries where a K^2
    coefficient is a relative, not a sub-ulp, effect at small angles."""
    m = ~np.eye(3, dtype=bool)
    return float((np.abs(np.asarray(got) - want)[m] / np.abs(want)[m]).max())


#: small-angle test vectors: all components comparable, so every off-diagonal entry is a
#: ~|theta| quantity; norms inside the 1e-8 series branch, and (for so3_log / W19) between
#: 1e-8 and 1e-6
_SMALL_DIRS = (np.array([0.62, -0.48, 0.55]), np.array([-0.41, 0.70, 0.52]))


@pytest.mark.claim("M2-F25")
def test_f25_series_branches_against_mpmath():
    """``so3_exp``, ``so3_right_jacobian``, ``M1``, ``M2`` (series branch, |theta| = 9e-9
    and 4e-9) and ``cayley_vector`` (its ``1e-12`` branch boundary, |theta| = 9.9e-7 and
    3e-7) against 60-digit ``mpmath`` closed forms, input rounded to double first.

    Bounds and what each separates (pass-8 mutants):
    * Exp, M1, M2 off-diagonal entries, RELATIVE 1e-12: a dropped ``K^2`` term is
      ~|theta| relative (Q39 3.7e-9, W17 9.3e-10, W18 6.9e-10); the series' own
      truncation is ~theta^2 ~ 1e-16;
    * Jr, ABSOLUTE 1e-15: a flipped ``K`` sign is |theta| ~ 9e-9 (Q01);
    * cayley_vector, RELATIVE 1e-14: ``2 tan(theta/2) / theta - 1 = theta^2 / 12`` is
      8.2e-14 at 9.9e-7 (W15: threshold 1e-6); the double evaluation is a few ulp."""
    mpmath = _mp()
    failures = []
    for d in _SMALL_DIRS:
        for nrm in (9e-9, 4e-9):
            phi = nrm * d / np.linalg.norm(d)
            mphi = [mpmath.mpf(float(x)) for x in phi]
            _check(failures, f"so3_exp series branch |phi| = {nrm:.0e} (off-diagonal, rel)",
                   _offdiag_rel(T.so3_exp(phi), _mp_to_np(_mp_exp(mpmath, mphi))), 1e-12)
            _check(failures, f"so3_right_jacobian series branch |phi| = {nrm:.0e} (abs)",
                   mabs(B1.so3_right_jacobian(phi), _mp_to_np(_mp_jr(mpmath, mphi))), 1e-15)
            for Tt in (0.37, 0.73):
                M1w, M2w = _mp_M1_M2(mpmath, mphi, mpmath.mpf(Tt))
                _check(failures, f"M1 series branch |w| = {nrm:.0e}, T = {Tt} (off-diag, rel)",
                       _offdiag_rel(T.M1(phi, Tt), _mp_to_np(M1w)), 1e-12)
                _check(failures, f"M2 series branch |w| = {nrm:.0e}, T = {Tt} (off-diag, rel)",
                       _offdiag_rel(T.M2(phi, Tt), _mp_to_np(M2w)), 1e-12)
        for nrm in (9.9e-7, 3e-7):
            phi = nrm * d / np.linalg.norm(d)
            mphi = [mpmath.mpf(float(x)) for x in phi]
            th = _mp_norm(mpmath, mphi)
            want = np.array([float(2 * mpmath.tan(th / 2) / th * x) for x in mphi])
            _check(failures, f"cayley_vector |phi| = {nrm:.1e} (rel)",
                   rel(B2.cayley_vector(phi), want), 1e-14)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F25")
def test_f25_so3_log_small_angles_against_mpmath():
    """``so3_log(R)`` for ``R`` = the 60-digit ``mpmath`` Exp of ``phi``, rounded to
    double, against ``phi``:

    * 1e-8 < |phi| < 1e-6 (9e-7, 3e-7), RELATIVE 1e-14.  This is the GENERIC branch,
      whose ``theta / (2 sin theta)`` is insensitive to the rounding of ``arccos``; the
      series value ``vee((R - R^T)/2)`` is low by ``theta^2/6`` = 1.35e-13 at 9e-7 (W19:
      ``SMALL_ANGLE = 1e-6``).

    The series branch itself (|phi| < 1e-8) stays exercised by :func:`test_traj_so3_helpers`
    and by every right-chart FD Jacobian (removing it gives NaN there).  A per-component 0.75-ulp check of
    it against the 60-digit log stood here until pass 9 and was REMOVED by orchestrator
    decision: it pinned correct rounding, which no claim states, with a 1.18x margin on
    the clean code (0.633-0.639 ulp), so an evaluation-order or BLAS change could fail it.
    The one mutant it killed, Q44 (``arccos`` clamped at ``1 - 1e-15``), is EQUIVALENT
    under the tolerance rule: 3.3e-24 absolute / 2.96e-16 relative on ``so3_log``
    (theta 1e-12 .. 1), against C22's 2.3e-11 per-edge gap / 10 = 2.3e-12."""
    mpmath = _mp()
    failures = []
    for d in _SMALL_DIRS:
        for nrm in (9e-7, 3e-7):
            phi = nrm * d / np.linalg.norm(d)
            mphi = [mpmath.mpf(float(x)) for x in phi]
            R = _mp_to_np(_mp_exp(mpmath, mphi))
            _check(failures, f"so3_log at |phi| = {nrm:.0e} (generic branch, rel)",
                   rel(T.so3_log(R), phi), 1e-14)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F21: ``**kwargs`` of single_step / bias_jacobian_recursion reach GTSAM
# --------------------------------------------------------------------------------------

#: a sensor-to-body pose that is not the identity: rotation Rz(0.3)-ish about a tilted
#: axis and a lever arm
BPS_R = iexp([0.05, -0.08, 0.30])
BPS_ARM = np.array([0.10, 0.20, -0.04])


def _bps_params(g):
    """A ``gtsam.PreintegrationParams`` built WITHOUT ``src`` (raw GTSAM)."""
    params = gtsam.PreintegrationParams(np.asarray(g, dtype=float))
    params.setBodyPSensor(gtsam.Pose3(Rot3(BPS_R), np.array(BPS_ARM)))
    return params


def _bps_correct(omega, acc, bias):
    """The sensor->body measurement correction, retyped: unbias in the sensor frame,
    rotate by ``bRs``, remove the lever-arm centripetal term ``w x (w x arm)``."""
    w = BPS_R @ (np.asarray(omega) - bias[0:3])
    a = BPS_R @ (np.asarray(acc) - bias[3:6]) - np.cross(w, np.cross(w, BPS_ARM))
    return w, a


def _bps_stream(gyro, accel, dt, bias, n):
    U = np.eye(5)
    for k in range(n):
        w, a = _bps_correct(gyro[k], accel[k], bias)
        dR, dv, dp = izoh(w, a, dt)
        U = U @ igal3(dR, dp, dv, dt)
    return U


@pytest.mark.claim("M2-F21")
def test_f21_kwargs_reach_gtsam_body_p_sensor():
    """``single_step(..., params=P)`` and ``bias_jacobian_recursion(..., params=P)`` with
    ``P.body_P_sensor`` non-identity, against the retyped sensor->body correction and a 5x5
    ``expm`` increment (no ``src``).  Pass 8: dropping the kwargs moves ``single_step``
    by 1.7e-4 (W14) and the recursion's ``J`` by 30% relative (W13); the inventory's old
    "equivalent" note covered gravity only."""
    failures = []
    for n, dt, b, g, n_steps in ((12, 1.7e-3, BIAS_BOTH, G_A, 9),
                                 (10, 4.1e-3, BIAS_ACCEL + BIAS_GYRO, G_B, 7)):
        gyro, accel, _ = T.time_varying_stream(n, dt)
        for k in (0, n - 1):
            w, a = _bps_correct(gyro[k], accel[k], b)
            dR, dv, dp = izoh(w, a, dt)
            _check(failures, f"single_step(dt={dt}, bias={b}, params with body_P_sensor) "
                   f"at k={k}", mabs(B3.single_step(gyro[k], accel[k], dt, b.copy(),
                                                    params=_bps_params(g)).matrix(),
                                     igal3(dR, dp, dv, dt)), 1e-14)
        Js = B3.bias_jacobian_recursion(gyro, accel, dt, b.copy(), n_steps,
                                        params=_bps_params(g))
        h = 1e-6
        for k in (0, n_steps - 1):
            U0inv = np.linalg.inv(_bps_stream(gyro, accel, dt, b, k + 1))
            J = np.zeros((10, 6))
            for m in range(6):
                e = np.zeros(6)
                e[m] = h
                J[:, m] = (igal3_log(U0inv @ _bps_stream(gyro, accel, dt, b + iswap(e), k + 1))
                           - igal3_log(U0inv @ _bps_stream(gyro, accel, dt, b - iswap(e), k + 1))
                           ) / (2 * h)
            _check(failures, f"bias_jacobian_recursion(bias {b}, dt {dt}, params with "
                   f"body_P_sensor) J_{k + 1}", rel(Js[k], J), 1e-6)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F22: outward aliasing and the Baseline 3 bias
# --------------------------------------------------------------------------------------


@pytest.mark.claim("M2-F22")
def test_f22_no_outward_aliasing():
    """Arrays HANDED OUT by ``src`` must not be views of its state:

    * ``ForsterPreintegration.increment()`` (and the Cayley subclass): writing into the
      returned ``dR`` / ``dv`` / ``dp`` must not move the PIM or its ``predict`` (W09);
    * ``.b_g`` / ``.b_a``: writing into them must not move ``bias_hat`` or the next step;
    * ``GalileanPreintegration.bias_hat`` is read-only, and is GTSAM's own ``biasHat``:
      an in-place edit must raise, and must never leave ``predict`` at a bias GTSAM is not
      integrating at (S39: pass 8 measured 4.2e-2)."""
    failures = []
    rng = _rng(61)
    state = T.NavTriple(*STATE_A)
    for cls in (B1.ForsterPreintegration, B2.CayleyPreintegration):
        pim = B1.integrate_edge(EDGE_A, BIAS_BOTH.copy(), cls=cls)
        before = [x.copy() for x in (pim.dR, pim.dv, pim.dp)]
        pred = pim.predict(state, G_A).as_tuple()
        for x in pim.increment():
            x[...] = 4.0
        if max(mabs(x, y) for x, y in zip((pim.dR, pim.dv, pim.dp), before)) != 0.0:
            failures.append(f"{cls.__name__}.increment() returns views of the PIM's state")
        if max(mabs(x, y) for x, y in zip(pim.predict(state, G_A).as_tuple(), pred)) != 0.0:
            failures.append(f"{cls.__name__}.predict moves after the caller writes into "
                            f"increment()")
        b0 = pim.bias_hat.copy()
        pim.b_g[...] = 0.5
        pim.b_a[...] = 0.5
        if not np.array_equal(pim.bias_hat, b0):
            failures.append(f"{cls.__name__}.b_g / .b_a are views into bias_hat")
    omega, acc = rng.normal(size=3), rng.normal(size=3)
    ref = ieuler([omega], [acc], 1e-3, BIAS_BOTH)
    pim = B1.ForsterPreintegration(bias_hat=BIAS_BOTH.copy())
    pim.b_g[...] = 0.0
    pim.integrate_measurement(omega, acc, 1e-3)
    if max(mabs(x, y) for x, y in zip(pim.increment(), ref)) > EULER_PIN:
        failures.append("a write into ForsterPreintegration.b_g moves the next step")

    for b, g in ((BIAS_BOTH, G_A), (BIAS_GYRO, G_B)):
        gal = B3.integrate_edge(EDGE_B, b.copy(), g_W=g)
        before = gal.predict(state).as_tuple()
        try:
            gal.bias_hat[:] = FIX_BIAS
            failures.append("GalileanPreintegration.bias_hat accepts an in-place write")
        except ValueError:
            pass
        if not np.array_equal(np.asarray(gal.bias_hat), b):
            failures.append("GalileanPreintegration.bias_hat is no longer its input after "
                            "an attempted in-place edit")
        if not np.array_equal(iswap(np.asarray(gal.pim.biasHat().vector())), np.asarray(
                gal.bias_hat)):
            failures.append("GalileanPreintegration.bias_hat desynchronises from GTSAM's "
                            "biasHat")
        after = gal.predict(state).as_tuple()
        if max(mabs(x, y) for x, y in zip(before, after)) != 0.0:
            failures.append("Baseline 3 predict moves after an in-place edit of bias_hat")
    _assert_no_failures(failures)


# ======================================================================================
# Verification pass 9 (docs/verification/stage-b_m2-baselines_pass9.md): M2-F26, M2-F27
# ======================================================================================
#
# The pass-9 survivors (X15, X21, X22, X24, X31, X41, W12, P23, S21) all moved an output
# by less than the old references could resolve: the covariance references were
# finite differences good to ~2.5e-9, and the bias-Jacobian references finite differences
# good to ~2e-8.  The references below are COMPLEX-STEP derivatives of the retyped
# discrete integrators: ``f'(x) = Im f(x + i h) / h`` has no subtractive cancellation, so
# with ``h = 1e-20`` it is exact to rounding (~1e-15 relative), and it needs no derivation
# of any Jacobian block.  Every map is written so that it is complex-ANALYTIC: no
# ``abs``/``norm``/``arccos``, the rotation angle enters only through ``phi . phi``.

CS_H = 1e-20


def _chat(v):
    """``hat`` that keeps a complex argument complex (``ihat`` casts to float)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
                    dtype=complex)


def _cexp(phi):
    """``Exp`` as the power series ``I + a(t) K + b(t) K^2``, ``t = phi . phi``,
    ``a = sum (-t)^n / (2n+1)!``, ``b = sum (-t)^n / (2n+2)!``: analytic in ``phi``, no
    ``sin(th)/th`` cancellation, and no ``src`` code.  With 30 terms it converges to
    rounding for ``|phi| <= 4`` (M2-F29 widened the guard from ``|phi| <= 2`` so the
    per-step angle sweep can reach 3 rad): the largest series term is ``t^n / (2n+1)!``
    <= 2.7 at ``t = 16``, so cancellation costs < 1e-15 absolute, and the truncation
    ``16^30 / 61!`` is ~1e-47.  Pinned against 50-digit mpmath ``expm`` up to 3.14 rad by
    :func:`test_f29_complex_series_exp_against_mpmath`."""
    phi = np.asarray(phi, dtype=complex)
    t = phi @ phi
    assert abs(t) <= 16.0, "series Exp used outside its accuracy range"
    a = b = 0.0
    term_a, term_b = 1.0 + 0j, 0.5 + 0j          # 1/1!, 1/2!
    for n in range(30):
        a = a + term_a
        b = b + term_b
        term_a = term_a * (-t) / ((2 * n + 2) * (2 * n + 3))
        term_b = term_b * (-t) / ((2 * n + 3) * (2 * n + 4))
    K = _chat(phi)
    return np.eye(3) + a * K + b * (K @ K)


def _ccay(phi):
    """Eq. (3) in its closed polynomial form, complex-analytic (cf. :func:`icay`)."""
    phi = np.asarray(phi, dtype=complex)
    K = _chat(phi)
    return np.eye(3) + (4.0 / (4.0 + phi @ phi)) * (K + 0.5 * (K @ K))


def _ceuler(gyro, accel, dt, bias, retract=_cexp, start=None):
    """:func:`ieuler` in complex arithmetic (the same Eq. (9a)-(9c) sums)."""
    if start is None:
        R, v, p = np.eye(3, dtype=complex), np.zeros(3, complex), np.zeros(3, complex)
    else:
        R, v, p = (np.asarray(x, dtype=complex) for x in start)
    bias = np.asarray(bias, dtype=complex)
    for omega, acc in zip(gyro, accel):
        w = np.asarray(omega, dtype=complex) - bias[0:3]
        a = np.asarray(acc, dtype=complex) - bias[3:6]
        p = p + v * dt + 0.5 * (R @ a) * dt**2
        v = v + (R @ a) * dt
        R = R @ retract(w * dt)
    return R, v, p


def _cs_chart(R0, out):
    """Forster-chart derivative ``(vee(R0^-1 dR), dv, dp)`` of a complex-stepped output."""
    dR = np.imag(out[0]) / CS_H
    # ``R0^-1``, not ``R0^T`` (M2-F29): over 1000 retyped steps at 3e-9 rad the reference
    # state drifts 1.1e-14 off orthogonality, and with ``R0^T`` that drift entered the
    # chart, putting the reference itself 7.0e-12 block-relative off a 40-digit mpmath
    # covariance (src: 9.0e-15).  ``R0^-1 dR`` is the exact right-chart derivative of
    # ``R0 Exp(d)`` whether or not ``R0`` is orthogonal.
    X = np.linalg.solve(R0, dR)
    return np.concatenate([ivee(0.5 * (X - X.T)), np.imag(out[1]) / CS_H,
                           np.imag(out[2]) / CS_H])


def _cs_step_jacobians(state, omega, acc, dt, bias, retract=_cexp):
    """``A`` (9x9, w.r.t. ``(dphi, dv, dp)`` in ``R Exp(dphi), v + dv, p + dp``) and
    ``Bg``, ``Ba`` (9x3, w.r.t. additive measurement noise) of ONE retyped Euler step,
    by complex step.  Exact to rounding; no Jacobian block is derived."""
    R, v, p = (np.asarray(x, dtype=float) for x in state)
    Rn = np.real(_ceuler([omega], [acc], dt, bias, retract, start=(R, v, p))[0])
    cols = []
    for slot in range(5):
        for i in range(3):
            d = np.zeros(3, complex)
            d[i] = 1j * CS_H
            Rs, vs, ps = R.astype(complex), v.astype(complex), p.astype(complex)
            om, ac = np.asarray(omega, complex), np.asarray(acc, complex)
            if slot == 0:
                Rs = Rs @ _cexp(d)
            elif slot == 1:
                vs = vs + d
            elif slot == 2:
                ps = ps + d
            elif slot == 3:
                om = om + d
            else:
                ac = ac + d
            cols.append(_cs_chart(Rn, _ceuler([om], [ac], dt, bias, retract,
                                              start=(Rs, vs, ps))))
    J = np.column_stack(cols)
    return J[:, 0:9], J[:, 9:12], J[:, 12:15]


def _cs_covariance(gyro, accel, dt, bias, sg=0.0, sa=0.0, retract=_cexp, C0=None):
    """First-order covariance of the noisy retyped Euler sums: the product of the
    complex-step per-step linearisations, ``C <- A C A^T + Bg Qg Bg^T + Ba Qa Ba^T`` with
    ``Q = diag(sigma^2) / dt`` (per-axis densities allowed, zeros allowed)."""
    sgv = np.broadcast_to(np.asarray(sg, float), (3,))
    sav = np.broadcast_to(np.asarray(sa, float), (3,))
    Qg, Qa = np.diag(sgv**2) / dt, np.diag(sav**2) / dt
    C = np.zeros((9, 9)) if C0 is None else np.array(C0, dtype=float)
    state = (np.eye(3), np.zeros(3), np.zeros(3))
    for omega, acc in zip(gyro, accel):
        A, Bg, Ba = _cs_step_jacobians(state, omega, acc, dt, bias, retract)
        C = A @ C @ A.T + Bg @ Qg @ Bg.T + Ba @ Qa @ Ba.T
        state = tuple(np.real(x) for x in _ceuler([omega], [acc], dt, bias, retract,
                                                  start=state))
    return C


def _cs_bias_jacobians(gyro, accel, dt, bias, retract=_cexp):
    """``(J_R, J_vg, J_va, J_pg, J_pa)`` of the retyped Euler sums by complex step in the
    bias, ``J_R`` in the right chart.  Exact to rounding (the FD references are ~2e-8)."""
    R0 = np.real(_ceuler(gyro, accel, dt, bias, retract)[0])
    J = np.zeros((9, 6))
    for m in range(6):
        b = np.asarray(bias, complex).copy()
        b[m] += 1j * CS_H
        J[:, m] = _cs_chart(R0, _ceuler(gyro, accel, dt, b, retract))
    return J[0:3, 0:3], J[3:6, 0:3], J[3:6, 3:6], J[6:9, 0:3], J[6:9, 3:6]


def _block_rel(A, B_):
    """max over the six upper 3x3 blocks of ``max|A - B| / max|B|`` (blocks of an
    all-zero reference are compared absolutely and must be exactly zero)."""
    worst = 0.0
    for i in range(3):
        for j in range(i, 3):
            a = A[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            b = B_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            scale = float(np.abs(b).max())
            err = float(np.abs(a - b).max())
            worst = max(worst, err / scale if scale > 0 else (np.inf if err else 0.0))
    return worst


# --------------------------------------------------------------------------------------
# M2-F26: Baseline 2's covariance with gyro noise ON
# --------------------------------------------------------------------------------------


@pytest.mark.claim("M2-F26")
def test_f26_cayley_covariance_against_complex_step():
    """``CayleyPreintegration``'s Eq. (63) with gyro noise on, against the complex-step
    linearisation of the noisy CAYLEY Euler integrator (:func:`_cs_covariance` with
    :func:`_ccay`), block by block.

    Pass 9 (M2-F26) found the covariance feeding gyro noise through ``Exp``'s right
    Jacobian while integrating with ``Cay``; the noisy step is ``Cay(w dt) Exp(J_Cay(w dt)
    n dt)`` (Rai Eq. (4)), so the rotation block was off by 4.6e-7 / 7.2e-7 relative on
    the fixture edges.  Every earlier B2 covariance check ran at zero densities, where the
    noise input does not enter.  The reparameterized contrast integrates with
    ``Cay(2 tan(theta/2) a) = Exp(theta a)``, so its reference retraction is ``Exp``.
    Measured agreement (clean): <= 7.6e-13 block-relative; with Exp's right Jacobian
    (F26) 1.1e-7 .. 6.2e-6; with an Exp transition (X24) 2.0e-9 .. 4.2e-7."""
    failures = []
    gyro_tv, accel_tv, _ = T.time_varying_stream(40, 4.7e-3)
    streams = (("EDGE_A", EDGE_A.gyro_stream(), EDGE_A.accel_stream(), EDGE_A.dt, BIAS_BOTH),
               ("EDGE_B", EDGE_B.gyro_stream(), EDGE_B.accel_stream(), EDGE_B.dt, BIAS_GYRO),
               ("time-varying stream", gyro_tv, accel_tv, 4.7e-3, BIAS_BOTH))
    sg = np.array([2e-4, 5e-4, 9e-4])
    sa = np.array([4e-3, 1.1e-2, 2.6e-2])
    for label, gyro, accel, dt, bias in streams:
        variants = ((B2.VARIANT_RAW, _ccay),) + (
            ((B2.VARIANT_REPARAM, _cexp),) if label == "EDGE_B" else ())
        for variant, cretract in variants:
            for noise, kw in (("gyro noise alone", dict(sigma_g=sg)),
                              ("gyro and accel noise", dict(sigma_g=sg, sigma_a=sa))):
                want = _cs_covariance(gyro, accel, dt, bias, kw.get("sigma_g", 0.0),
                                      kw.get("sigma_a", 0.0), cretract)
                pim = B2.CayleyPreintegration(bias_hat=bias.copy(), variant=variant, **kw)
                for omega, acc in zip(gyro, accel):
                    pim.integrate_measurement(omega, acc, dt)
                _check(failures, f"CayleyPreintegration({variant}) Eq. (63), {noise}, on "
                       f"{label} (block-relative)", _block_rel(pim.cov, want), COV_CS_REL)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F27 (c): the noise gate with per-axis zeros and tiny densities
# --------------------------------------------------------------------------------------


@pytest.mark.claim("M2-F27")
def test_f27_has_noise_per_axis_and_tiny_densities():
    """Per-axis densities with ZERO axes, and tiny positive densities (1e-21), for both
    integrators, against :func:`_cs_covariance`.  ``has_noise`` is "some density is
    positive": a single positive gyro axis, or an accel density of 1e-21, is noise and
    must be propagated (pass 9: X21 required every gyro axis to be positive and X22
    ignored densities <= 1e-20; both returned a ZERO covariance, 100% off)."""
    failures = []
    gyro, accel, _ = T.time_varying_stream(14, 1e-3)
    configs = (("sigma_g = [3e-4, 0, 0] alone", np.array([3e-4, 0.0, 0.0]), 0.0),
               ("sigma_g = [0, 0, 5e-4], sigma_a = [0, 2e-2, 0]", np.array([0.0, 0.0, 5e-4]),
                np.array([0.0, 2e-2, 0.0])),
               ("sigma_a = 1e-21 alone", 0.0, 1e-21),
               ("sigma_a = [0, 0, 1e-21] alone", 0.0, np.array([0.0, 0.0, 1e-21])),
               ("sigma_g = 1e-21 alone", 1e-21, 0.0))
    for cls, cretract in ((B1.ForsterPreintegration, _cexp), (B2.CayleyPreintegration, _ccay)):
        for label, sg, sa in configs:
            where = f"{cls.__name__} with {label}"
            pim = cls(bias_hat=BIAS_BOTH.copy(), sigma_g=sg, sigma_a=sa)
            if not pim.has_noise:
                failures.append(f"{where}: has_noise is False")
            for omega, acc in zip(gyro, accel):
                pim.integrate_measurement(omega, acc, 1e-3)
            want = _cs_covariance(gyro, accel, 1e-3, BIAS_BOTH, sg, sa, cretract)
            _check(failures, f"{where}: Eq. (63) over 14 steps (block-relative)",
                   _block_rel(pim.cov, want), COV_CS_REL)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F27 (d): M1 / M2 on FAST twists against high-precision quadrature
# --------------------------------------------------------------------------------------


def _mp_quad_M1_M2(w, Tt):
    """``M1 = int_0^T Exp(w s) ds`` and ``M2 = int_0^T (T - s) Exp(w s) ds`` (Cauchy's
    repeated-integral formula) by 30-digit ``mpmath.quad`` of the Rodrigues coefficients
    ``sin(th s)`` and ``1 - cos(th s)`` -- numerical integration, not the closed-form
    antiderivatives under test."""
    mpmath = _mp()
    mpmath.mp.dps = 30
    w = [mpmath.mpf(float(x)) for x in w]
    Tt = mpmath.mpf(Tt)
    th = _mp_norm(mpmath, w)
    K = _mp_hat(mpmath, [x / th for x in w])
    I3 = mpmath.eye(3)
    s1 = mpmath.quad(lambda s: mpmath.sin(th * s), [0, Tt])
    c1 = mpmath.quad(lambda s: 1 - mpmath.cos(th * s), [0, Tt])
    s2 = mpmath.quad(lambda s: (Tt - s) * mpmath.sin(th * s), [0, Tt])
    c2 = mpmath.quad(lambda s: (Tt - s) * (1 - mpmath.cos(th * s)), [0, Tt])
    out = (_mp_to_np(Tt * I3 + s1 * K + c1 * (K * K)),
           _mp_to_np(Tt**2 / 2 * I3 + s2 * K + c2 * (K * K)))
    mpmath.mp.dps = 60
    return out


@pytest.mark.claim("M2-F27")
def test_f27_M1_M2_fast_twists_against_quadrature():
    """``M1``, ``M2`` at |w| = 5, 2.55 and 4.05 rad/s, T = 1 and 0.9 s, against
    :func:`_mp_quad_M1_M2`, at C02's own 1e-14.  On the slow fixture-like twists a
    1e-13-relative error in the ``K^2`` coefficient is ~5e-16 absolute (pass 9: X15
    survived); at |w| = 5 it is 1.2e-13.  Measured clean: <= 2.5e-16."""
    failures = []
    for w, Tt in ((np.array([3.0, -4.0, 0.0]), 1.0), (np.array([1.5, 0.5, -2.0]), 1.0),
                  (np.array([-2.2, 3.1, 1.4]), 0.9)):
        M1w, M2w = _mp_quad_M1_M2(w, Tt)
        _check(failures, f"M1(w={w}, T={Tt}) vs mpmath quadrature", mabs(T.M1(w, Tt), M1w),
               1e-14)
        _check(failures, f"M2(w={w}, T={Tt}) vs mpmath quadrature", mabs(T.M2(w, Tt), M2w),
               1e-14)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F27 (e): navstate on a rotation carrying orthogonality drift
# --------------------------------------------------------------------------------------


@pytest.mark.claim("M2-F27")
def test_f27_navstate_keeps_a_drifted_rotation():
    """``navstate`` passes ``R`` to GTSAM as given (``Rot3(matrix)`` stores the matrix; no
    re-orthonormalisation), so Baseline 3's ``predict`` from a state whose ``R`` has
    drifted off SO(3) -- as chained floating-point states do -- is the component endpoint
    relation (C03, "holds for any state") applied to THAT ``R``, to 1e-12.  Projecting
    (pass 9: X41, ``Rot3.ClosestTo``) moves the result by about the drift: 1.4e-11 /
    5.0e-11 in R at orthogonality errors 2.9e-11 / 1.1e-10.  Measured clean: 5.2e-15."""
    rng = _rng(67)
    failures = []
    for (R0, v0, p0), edge, bias, g, drift in ((STATE_A, EDGE_A, BIAS_BOTH, G_A, 1e-11),
                                               (STATE_B, EDGE_B, BIAS_ACCEL, G_B, 3e-11)):
        R = R0 + drift * rng.normal(size=(3, 3))
        # orthogonality error 2.9e-11 / 1.1e-10 (deterministic seed).  No precondition
        # assert: it could not fail (M2-F28); that the drift is visible is shown by X41's
        # kill in tests/test_mutation.py, not asserted here.
        orth = float(np.abs(R.T @ R - np.eye(3)).max())
        where = f"(orthogonality error {orth:.1e})"
        if not np.array_equal(B3.navstate(T.NavTriple(R=R, v=v0, p=p0)).attitude().matrix(), R):
            failures.append(f"navstate does not keep a drifted R as given {where}")
        got = B3.integrate_edge(edge, bias.copy(), g_W=g).predict(T.NavTriple(R=R, v=v0, p=p0))
        U = izoh_stream(edge.gyro_stream(), edge.accel_stream(), edge.dt, bias)
        want = ichain((R, v0, p0), (U[0:3, 0:3], U[0:3, 3], U[0:3, 4]),
                      EDGE_DURATION[edge.index], g)
        for name, g_, w_ in zip("Rvp", got.as_tuple(), want):
            _check(failures, f"Baseline 3 predict from a drifted R {where}: {name}",
                   mabs(g_, w_), TIER0)
    _assert_no_failures(failures)


# --------------------------------------------------------------------------------------
# M2-F27 (f): FD-step choices, on high-rate edges and at large eps
# --------------------------------------------------------------------------------------

#: src's default-eps bias Jacobians vs complex step, relative.  Measured clean (FD_EPS =
#: 3e-5): <= 4.2e-10 on the high-rate edges below; FD_EPS = 1e-3 (P23) / a factor eps of
#: 1e-3 (S21) give 8.6e-8 and 1.8e-7; the old FD_EPS = 1e-6 gave up to 2.0e-8.
JAC_CS_REL = 1e-8

#: high-rate edges: |omega| = 5 rad/s over 1 s and |omega| = 2.55 rad/s over 1.5 s
HIGH_RATE_EDGES = (
    ("|w| = 5, T = 1 s", T.Edge(index=0, omega_body=np.array([3.0, -4.0, 0.0]),
                                f_body=np.array([0.4, -1.2, 9.7]), dt=5e-3, n_steps=200)),
    ("|w| = 2.55, T = 1.5 s", T.Edge(index=1, omega_body=np.array([1.5, 0.5, -2.0]),
                                     f_body=np.array([-0.6, 0.8, 10.1]), dt=5e-3, n_steps=300)))


@pytest.mark.claim("M2-F27")
def test_f27_bias_jacobians_on_high_rate_edges_against_complex_step():
    """``bias_jacobians_fd`` (default eps; B1 and B2) and ``CayleyFactor.from_edge`` (both
    variants) on high-rate edges, against :func:`_cs_bias_jacobians` of the retyped Euler
    sums.  The old checks ran only on slow edges against FD references good to ~2e-8, so
    an FD step of 1e-3 (P23, S21: truncation 1.8e-7 here) survived."""
    failures = []
    for label, edge in HIGH_RATE_EDGES:
        gyro, accel, dt = edge.gyro_stream(), edge.accel_stream(), edge.dt
        refs = {"exp": _cs_bias_jacobians(gyro, accel, dt, BIAS_BOTH, _cexp),
                "cay": _cs_bias_jacobians(gyro, accel, dt, BIAS_BOTH, _ccay)}
        for cls, key in ((B1.ForsterPreintegration, "exp"), (B2.CayleyPreintegration, "cay")):
            got = B1.bias_jacobians_fd(edge, BIAS_BOTH.copy(), cls=cls)
            for name, g_, r_ in zip(JAC_NAMES, got, refs[key]):
                _check(failures, f"bias_jacobians_fd({cls.__name__}) {name} at the default "
                       f"eps on the high-rate edge {label}", rel(g_, r_), JAC_CS_REL)
        for variant, key in ((B2.VARIANT_RAW, "cay"), (B2.VARIANT_REPARAM, "exp")):
            f = B2.CayleyFactor.from_edge(edge, BIAS_BOTH.copy(), variant=variant)
            for name, r_ in zip(FACTOR_JAC_NAMES, refs[key]):
                _check(failures, f"CayleyFactor.from_edge({variant}) {name} on the high-rate "
                       f"edge {label}", rel(getattr(f, name), r_), JAC_CS_REL)
    _assert_no_failures(failures)


def _igal3_vee(A):
    return np.concatenate([ivee(A[0:3, 0:3]), A[0:3, 3], A[0:3, 4], [A[3, 4]]])


def _irecursion(gyro, accel, dt, bias, n, eps):
    """``J_{k+1} = Ad_{V_k^-1} J_k + C_k`` retyped: ``V_k`` from one 5x5 ``expm`` (exact
    ZOH), ``C_k`` by central differences at step ``eps`` of :func:`igal3_log`, and the
    adjoint by conjugating :func:`igal3_hat`.  No ``src``, no GTSAM."""
    def V(k, db):
        b = bias + iswap(db)
        dR, dv, dp = izoh(gyro[k] - b[0:3], accel[k] - b[3:6], dt)
        return igal3(dR, dp, dv, dt)

    J, out = np.zeros((10, 6)), []
    for k in range(n):
        V0 = V(k, np.zeros(6))
        Xinv = np.linalg.inv(V0)
        C = np.zeros((10, 6))
        for m in range(6):
            e = np.zeros(6)
            e[m] = eps
            C[:, m] = (igal3_log(Xinv @ V(k, e)) - igal3_log(Xinv @ V(k, -e))) / (2 * eps)
        Ad = np.column_stack([_igal3_vee(Xinv @ igal3_hat(c) @ V0) for c in np.eye(10)])
        J = Ad @ J + C
        out.append(J)
    return out


@pytest.mark.claim("M2-F27")
def test_f27_recursion_reads_eps_at_large_steps():
    """``bias_jacobian_recursion(eps=...)`` with LARGE steps (dt = 0.3 s, 10 steps) and
    large eps (5e-2, 0.2), where one step's increment is visibly nonlinear in the bias,
    against :func:`_irecursion` at the SAME eps.  The inventory used to say ``eps`` "does
    not change the result measurably"; at these inputs it moves ``J`` by 1.9e-7 and
    3.1e-6 relative, so a recursion that drops ``eps`` (pass 9: W12) is wrong.  Measured
    clean: <= 1.9e-14."""
    failures = []
    dt, n = 0.3, 10
    gyro, accel, _ = T.time_varying_stream(n, dt)
    for eps in (5e-2, 0.2):
        want = _irecursion(gyro, accel, dt, BIAS_BOTH, n, eps)
        # eps moves the reference by 1.9e-7 / 3.1e-6 relative vs eps = 1e-6 (measured, see
        # the docstring).  No precondition assert: it could not fail (M2-F28); that eps is
        # visible is shown by W12's kill in tests/test_mutation.py.
        got = B3.bias_jacobian_recursion(gyro, accel, dt, BIAS_BOTH.copy(), n, eps=eps)
        _check(failures, f"bias_jacobian_recursion at dt = {dt}, eps = {eps}",
               max(rel(g_, w_) for g_, w_ in zip(got, want)), 1e-9)
    _assert_no_failures(failures)


# ======================================================================================
# M2-F28 (verification pass 10): per-step angle band, has_noise semantics, Eq. (21b) /
# (21c) bias-half combinations and large v_i Dt
# ======================================================================================


def _mp_expm_hat(mpmath, phi):
    """``Exp`` as ``mpmath.expm`` of the 3x3 hat matrix: the general matrix exponential,
    not Rodrigues, so it shares no closed form with the code under test."""
    return mpmath.expm(_mp_hat(mpmath, phi))


def _mp_cay(mpmath, phi):
    """Rai Eq. (3) as its DEFINITION, ``(I - K/2)^-1 (I + K/2)``, in ``mpmath``."""
    K = _mp_hat(mpmath, phi)
    I3 = mpmath.eye(3)
    return mpmath.inverse(I3 - K / 2) * (I3 + K / 2)


def _mp_right_jacobian(mpmath, retract, phi, h):
    """The right Jacobian of ``retract`` from its DEFINITION, ``retract(phi + t e)
    = retract(phi) Exp(t J e) + O(t^2)``: column ``i`` is ``vee(retract(phi)^T
    d/dt retract(phi + t e_i))`` by a central difference at step ``h`` in ``mpmath``
    (truncation ``~h^2``, round-off ``~10^-dps / h``).  No Jacobian formula is used."""
    R0T = retract(mpmath, phi).T
    cols = []
    for i in range(3):
        up = [x + (h if j == i else 0) for j, x in enumerate(phi)]
        dn = [x - (h if j == i else 0) for j, x in enumerate(phi)]
        D = R0T * (retract(mpmath, up) - retract(mpmath, dn)) / (2 * h)
        cols.append([float((D[2, 1] - D[1, 2]) / 2), float((D[0, 2] - D[2, 0]) / 2),
                     float((D[1, 0] - D[0, 1]) / 2)])
    return np.array(cols).T


#: per-step / argument angles from just above the 1e-8 series branch to 3 rad
F28_ANGLES = (1.1e-8, 1e-7, 1e-6, 1e-5, 9e-5, 1e-4, 1.1e-4, 1e-3, 1e-2, 1.1e-2, 0.1, 0.5,
              1.0, 2.0, 3.0)

#: Jr / J_Cay vs the mpmath definitional derivative, ABSOLUTE.  Measured clean <= 1.7e-16
#: on every angle (entries are O(1)); 60x margin.
F28_JACOBIAN_ABS = 1e-14


@pytest.mark.claim("M2-F28")
def test_f28_right_jacobians_against_mpmath_across_angles():
    """``so3_right_jacobian`` (B1) and ``cay_right_jacobian`` (Rai Eq. (4)) at |phi| from
    1.1e-8 to 3 rad, two directions each, against :func:`_mp_right_jacobian` of
    ``mpmath.expm`` / the Eq. (3) definition at 50 digits (h = 1e-22), at 1e-14 absolute.

    Pass 10 (M2-F28): every earlier Jacobian check sat at one or two angles.  What this
    separates, measured:
    * Y18 (``Jr``'s K^2 coefficient x (1 + 1e-8)): 7.8e-9 at |phi| = 3, 4.5e-9 at 2,
      1.3e-9 at 1 (vs the 1e-14 bound);
    * the textbook ``(1 - cos th) / th^2`` K coefficient (src before M2-F28): 3.0e-9 /
      3.3e-9 at 1.1e-8, 2.6e-11 / 2.9e-11 at 1e-7, 2.9e-11 / 3.2e-11 at 1e-6 --
      cancellation, up to ~1e-16 / th.
    """
    mpmath = _mp()
    mpmath.mp.dps = 50
    h = mpmath.mpf("1e-22")
    failures = []
    try:
        for d in _SMALL_DIRS:
            for nrm in F28_ANGLES:
                phi = nrm * d / np.linalg.norm(d)
                mphi = [mpmath.mpf(float(x)) for x in phi]
                _check(failures, f"so3_right_jacobian at |phi| = {nrm:.1e} vs mpmath (abs)",
                       mabs(B1.so3_right_jacobian(phi),
                            _mp_right_jacobian(mpmath, _mp_expm_hat, mphi, h)),
                       F28_JACOBIAN_ABS)
                _check(failures, f"cay_right_jacobian at |phi| = {nrm:.1e} vs mpmath (abs)",
                       mabs(B2.cay_right_jacobian(phi),
                            _mp_right_jacobian(mpmath, _mp_cay, mphi, h)),
                       F28_JACOBIAN_ABS)
    finally:
        mpmath.mp.dps = 60
    _assert_no_failures(failures)


def _unit_rows(X):
    return X / np.linalg.norm(X, axis=1)[:, None]


#: ``(per-step angle |(omega - b_g) dt|, dt[, steps])`` (12 steps unless given): angles
#: from 3e-9 to 3 rad on both sides of every branch point in the noise-input / transition
#: code (1e-8, 1e-4, 1e-2), at body rates from 1.2e-6 rad/s (a stationary IMU) to 10 rad/s.
#: M2-F29 added the two ends: 3e-9 (over 1000 steps) and 9e-9 rad sit INSIDE the 1e-8
#: series branch of ``Exp`` / ``Jr``, the normal regime of a stationary, well-calibrated
#: IMU; 0.7, 1, 2 and 3 rad reach the domain edge (nothing restricts B1's or B2's per-step
#: angle to <= 0.5 rad).
F28_STEP_ANGLES = ((3e-9, 2.5e-3, 1000), (9e-9, 1e-3),
                   (1e-7, 1e-3), (1e-6, 1e-3), (1e-5, 1e-4), (5e-5, 1e-4), (9e-5, 1e-3),
                   (2e-2, 5e-3), (0.1, 1e-2), (0.25, 2.5e-2), (0.5, 5e-2),
                   (0.7, 7e-2), (1.0, 0.1), (2.0, 0.2), (3.0, 0.3))


def _f28_stream(angle, dt, n=12):
    """A time-varying stream whose bias-corrected per-step angle is EXACTLY ``angle``
    (rotation axis and specific force vary step to step, so the steps do not commute)."""
    gyro0, accel, _ = T.time_varying_stream(n, dt)
    gyro = BIAS_BOTH[0:3] + _unit_rows(gyro0) * (angle / dt)
    return gyro, accel


@pytest.mark.claim("M2-F28")
def test_f28_covariance_across_per_step_angles_against_complex_step():
    """Eq. (63) for Baseline 1, Baseline 2 (raw) and the reparameterized contrast, at
    per-step angles 3e-9 .. 3 rad (:data:`F28_STEP_ANGLES`), anisotropic gyro noise
    alone and gyro + accel, against :func:`_cs_covariance` (complex step), block-relative
    :data:`COV_CS_REL`.

    Pass 10 (M2-F28 b/f): every covariance reference used per-step angles in [4e-4,
    4e-3] rad, so a branch on ``|w dt|`` outside it survived.  Measured clean: <= 5.4e-15
    (after the M2-F28 ``Jr`` fix; with the old K coefficient B1 is 8.9e-11 at 1e-7 rad and
    9.95e-11 at 1e-6 -- inside the bound, which is why the ``Jr`` fix is pinned by
    :func:`test_f28_right_jacobians_against_mpmath_across_angles`, not here).  Mutants and
    their worst effect here (block-relative, vs 1e-10):
    * Y24 (B2 raw uses ``J_r`` below 1e-4): 1.35e-9 at 9e-5 rad, 4.2e-10 at 5e-5;
    * Y55 (B2 raw uses ``J_r`` above 1e-2): 6.7e-5 (2e-2 rad) .. 5.2e-2 (0.5 rad);
      Y56 (B1 ``I - K/2`` above 1e-2): 1.3e-4 .. 8.6e-2;
    * Y57 (``A_RR = I`` below 1e-4, B1 and B2): 3.8e-7 (1e-7 rad) .. 3.4e-4 (9e-5 rad);
    * Y18 (``Jr`` K^2 coefficient x (1 + 1e-8)): 8.2e-10 at 0.5 rad, 2.1e-10 at 0.25.

    Pass 11 (M2-F29): the band was [1e-7, 0.5] rad, so branches below the 1e-8 series
    threshold and above 0.5 rad survived.  Added 3e-9 rad (x 1000 steps), 9e-9 rad, and
    0.7, 1, 2, 3 rad.  Measured clean at the new angles (worst of the three classes and
    both noise settings): 1.1e-14 (3e-9 x 1000), 9.2e-16 (9e-9), 1.4e-15 (0.7), 2.4e-15
    (1), 5.4e-15 (2), 7.7e-15 (3); every older angle unchanged or lower (the ``R0^-1``
    chart fix in :func:`_cs_chart`).  Mutants (vs 1e-10):
    * Z07 (B1 ``I - K/2`` above 0.6): 0.18 (0.7 rad) .. 8.2 (3 rad);
    * Z09 (``A_RR = I`` above 0.6): 2.3 .. 34;  Z11 (B2 raw ``J_r`` above 0.6): 0.092 .. 1.7;
    * Z08 (``A_RR = I`` below 1e-8): 1.9e-6 (3e-9 x 1000), 3.4e-8 (9e-9);
    * Z10 (gyro noise input ``I`` below 1e-8): 2.8e-9 (3e-9 x 1000), 1.0e-8 (9e-9)."""
    failures = []
    sg = np.array([2e-4, 5e-4, 9e-4])
    sa = np.array([4e-3, 1.1e-2, 2.6e-2])
    cases = (("ForsterPreintegration", B1.ForsterPreintegration, {}, _cexp),
             ("CayleyPreintegration(raw)", B2.CayleyPreintegration,
              {"variant": B2.VARIANT_RAW}, _ccay),
             ("CayleyPreintegration(reparam)", B2.CayleyPreintegration,
              {"variant": B2.VARIANT_REPARAM}, _cexp))
    for angle, dt, *steps in F28_STEP_ANGLES:
        n = steps[0] if steps else 12
        gyro, accel = _f28_stream(angle, dt, n)
        for name, cls, kw, cretract in cases:
            for noise, dens in (("gyro noise alone", dict(sigma_g=sg)),
                                ("gyro and accel noise", dict(sigma_g=sg, sigma_a=sa))):
                want = _cs_covariance(gyro, accel, dt, BIAS_BOTH, dens.get("sigma_g", 0.0),
                                      dens.get("sigma_a", 0.0), cretract)
                pim = cls(bias_hat=BIAS_BOTH.copy(), **kw, **dens)
                for omega, acc in zip(gyro, accel):
                    pim.integrate_measurement(omega, acc, dt)
                _check(failures, f"{name} Eq. (63), {noise}, per-step angle {angle:.2g} "
                       f"(dt = {dt:.2g}) over {n} steps (block-relative)",
                       _block_rel(pim.cov, want), COV_CS_REL)
    _assert_no_failures(failures)


# ======================================================================================
# M2-F29 (verification pass 11): the per-step angle sweep's two ends, the complex series
# ``Exp`` at large angles, and Eq. (22b) at |Gamma| ~ 1
# ======================================================================================


@pytest.mark.claim("M2-F29")
def test_f29_complex_series_exp_against_mpmath():
    """:func:`_cexp` (the reference ``Exp`` of every complex-step covariance) at |phi| from
    1e-9 to 3.14 rad, against 50-digit ``mpmath.expm`` of the hat matrix, at 1e-14 absolute.
    M2-F29 widened its guard to ``|phi| <= 4`` so the sweep can reach 3 rad; this pins that
    the series is still exact there.  Measured clean: <= 3.3e-16."""
    mpmath = _mp()
    mpmath.mp.dps = 50
    failures = []
    for d in _SMALL_DIRS:
        for nrm in (1e-9, 9e-9, 1e-3, 0.5, 1.0, 2.0, 3.0, 3.14):
            phi = nrm * d / np.linalg.norm(d)
            want = _mp_to_np(_mp_expm_hat(mpmath, [mpmath.mpf(float(x)) for x in phi]))
            _check(failures, f"_cexp at |phi| = {nrm:.2g} vs mpmath expm (abs)",
                   mabs(np.real(_cexp(phi)), want), 1e-14)
    _assert_no_failures(failures)


@pytest.mark.slow
@pytest.mark.claim("M2-F29")
def test_f29_covariance_of_a_stationary_imu_over_4000_steps():
    """Eq. (63) for B1, B2 (raw) and the reparameterized contrast over 4000 steps at a
    per-step angle of 5e-9 rad (dt = 2.5e-3: a 10 s stationary IMU at 400 Hz with
    ``|w - b_g|`` = 2e-6 rad/s, i.e. inside the 1e-8 series branch at every step), gyro
    noise alone, against :func:`_cs_covariance`, block-relative :data:`COV_CS_REL`.

    Measured clean: 6.8e-14 (all three).  Before M2-F29 the complex-step chart used
    ``R0^T`` and was itself 7.0e-12 off a 40-digit mpmath covariance at 3e-9 rad x 1000
    steps (src: 9.0e-15), see :func:`_cs_chart`.  Z08 (``A_RR = I`` below 1e-8):
    7.7e-6; Z10 (gyro noise input ``I`` below 1e-8): 3.5e-9 (both vs 1e-10)."""
    failures = []
    sg = np.array([2e-4, 5e-4, 9e-4])
    angle, dt, n = 5e-9, 2.5e-3, 4000
    gyro, accel = _f28_stream(angle, dt, n)
    for name, cls, kw, cretract in (
            ("ForsterPreintegration", B1.ForsterPreintegration, {}, _cexp),
            ("CayleyPreintegration(raw)", B2.CayleyPreintegration,
             {"variant": B2.VARIANT_RAW}, _ccay),
            ("CayleyPreintegration(reparam)", B2.CayleyPreintegration,
             {"variant": B2.VARIANT_REPARAM}, _cexp)):
        want = _cs_covariance(gyro, accel, dt, BIAS_BOTH, sg, 0.0, cretract)
        pim = cls(bias_hat=BIAS_BOTH.copy(), sigma_g=sg, **kw)
        for omega, acc in zip(gyro, accel):
            pim.integrate_measurement(omega, acc, dt)
        _check(failures, f"{name} Eq. (63), stationary IMU, per-step angle 5e-09 over "
               f"{n} steps (block-relative)", _block_rel(pim.cov, want), COV_CS_REL)
    _assert_no_failures(failures)


#: Eq. (22b) vs a retyped Eq. (19), absolute: the claim's own bound (M2-C11).
F29_EQ22B_ABS = 1e-13


@pytest.mark.claim("M2-F29")
def test_f29_eq22b_matches_a_retyped_eq19_at_large_gamma():
    """Rai Eq. (22b) (``src``) against Eq. (19) RETYPED here,
    ``(R_j - R_i dRbar) - (1/2)(R_j + R_i dRbar) hat(J_R (b_g - bbar_g))``, over 2000 draws
    with ``J_R ~ N(0, 1)``, ``b_g, bbar_g ~ U(-0.5, 0.5)``, ``|dRbar| <= 1.7`` rad and
    ``R_i, R_j`` anywhere, at C11's own 1e-13 absolute.

    Pass 11 (M2-F29, Z30): ``test_c11`` draws ``|Gamma| = |J_R bbar_g| <~ 0.09``, so a
    1e-12 relative slip in ``Rbreve``'s ``Gamma/2`` moves it by <~ 7e-14, under its bound.
    Here ``|Gamma|`` reaches ~2.  Measured clean: 8.9e-16 (three seeds); Z30
    (``Gamma/2 x (1 + 1e-12)`` in ``Rbreve``): 1.2e-12."""
    rng = _rng(20261015)
    worst = 0.0
    for _ in range(2000):
        R_i = iexp(rng.uniform(-np.pi, np.pi, 3))
        R_j = iexp(rng.uniform(-np.pi, np.pi, 3))
        dRbar = iexp(rng.uniform(-1.0, 1.0, 3))
        J_R = rng.standard_normal((3, 3))
        b_g = rng.uniform(-0.5, 0.5, 3)
        bbar_g = rng.uniform(-0.5, 0.5, 3)
        A = R_i @ dRbar
        want = (R_j - A) - 0.5 * (R_j + A) @ ihat(J_R @ (b_g - bbar_g))
        worst = max(worst, mabs(B2.rotation_residual_eq22b(R_i, R_j, dRbar, J_R, b_g,
                                                           bbar_g), want))
    assert worst <= F29_EQ22B_ABS, (
        f"rotation_residual_eq22b at large Gamma vs retyped Eq. (19): {worst:.3e} "
        f"> {F29_EQ22B_ABS:.0e}")


#: positive densities down to the smallest positive normal float and the smallest
#: denormal.  ``has_noise`` is ``sigma > 0`` on any axis, so DENORMALS COUNT AS NOISE
#: (a denormal is a strictly positive float; nothing in Eq. (63) needs it to be normal).
#: The covariance is checked down to 1e-150, where its largest entry is 1.4e-302 (still
#: normal; measured <= 4.1e-15 block-relative).  Below that it goes denormal (1.4e-306 at
#: 1e-152) and then underflows to zero (1e-156), so there only the flag is pinned -- and
#: the flag at the smallest denormal rules out ANY positive threshold.
F28_DENSITIES_PROPAGATED = (1e-21, 1e-60, 1e-61, 1e-100, 1e-101, 1e-140, 1e-150)
F28_DENSITIES_FLAG_ONLY = (1e-200, float(np.finfo(float).tiny), 5e-324)


def _cs_step_sequence(gyro, accel, dt, bias, retract):
    """The complex-step ``(A, Bg, Ba)`` of every step of :func:`_cs_covariance`, so many
    density settings can reuse one linearisation."""
    out, state = [], (np.eye(3), np.zeros(3), np.zeros(3))
    for omega, acc in zip(gyro, accel):
        out.append(_cs_step_jacobians(state, omega, acc, dt, bias, retract))
        state = tuple(np.real(x) for x in _ceuler([omega], [acc], dt, bias, retract,
                                                  start=state))
    return out


def _cs_cov_from_steps(steps, dt, sg, sa):
    sgv = np.broadcast_to(np.asarray(sg, float), (3,))
    sav = np.broadcast_to(np.asarray(sa, float), (3,))
    Qg, Qa = np.diag(sgv**2) / dt, np.diag(sav**2) / dt
    C = np.zeros((9, 9))
    for A, Bg, Ba in steps:
        C = A @ C @ A.T + Bg @ Qg @ Bg.T + Ba @ Qa @ Ba.T
    return C


@pytest.mark.claim("M2-F28")
def test_f28_has_noise_is_any_positive_density_on_any_axis():
    """For B1 and B2, each noise source alone, each axis alone, and every density in
    :data:`F28_DENSITIES_PROPAGATED` + :data:`F28_DENSITIES_FLAG_ONLY`: ``has_noise`` is
    True, and (where the covariance is representable) Eq. (63) matches the complex-step
    reference block-relative at :data:`COV_CS_REL`.

    Pass 10 (M2-F28 c): ``has_noise`` was pinned only at 1e-21, so a threshold anywhere
    in (1e-21, ~1.5e-154] survived (Y20: <= 1e-100 ignored; Y58: accel <= 1e-60 ignored);
    each returns a ZERO covariance where the reference is nonzero (block-relative error
    1.0, i.e. 100% off), and ``has_noise`` False.  Measured clean: <= 3.0e-15
    block-relative at every propagated density."""
    failures = []
    gyro, accel, _ = T.time_varying_stream(14, 1e-3)
    dt = 1e-3
    for cls, cretract in ((B1.ForsterPreintegration, _cexp), (B2.CayleyPreintegration, _ccay)):
        steps = _cs_step_sequence(gyro, accel, dt, BIAS_BOTH, cretract)
        for source in ("sigma_g", "sigma_a"):
            for axis in range(3):
                for dens in F28_DENSITIES_PROPAGATED + F28_DENSITIES_FLAG_ONLY:
                    sig = np.zeros(3)
                    sig[axis] = dens
                    where = f"{cls.__name__} with {source}[{axis}] = {dens:.1e} alone"
                    pim = cls(bias_hat=BIAS_BOTH.copy(), **{source: sig})
                    if not pim.has_noise:
                        failures.append(f"{where}: has_noise is False")
                    if dens not in F28_DENSITIES_PROPAGATED:
                        continue
                    for omega, acc in zip(gyro, accel):
                        pim.integrate_measurement(omega, acc, dt)
                    want = _cs_cov_from_steps(steps, dt, sig if source == "sigma_g" else 0.0,
                                              sig if source == "sigma_a" else 0.0)
                    _check(failures, f"{where}: Eq. (63) over 14 steps (block-relative)",
                           _block_rel(pim.cov, want), COV_CS_REL)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F28")
def test_f28_eq21b_eq21c_every_combination_of_bias_halves():
    """Eq. (21b) / (21c) with the bias terms supplied for the gyro half only, the accel
    half only, both, and neither, against the retyped residuals (an omitted half is a
    zero term), at TIER0.

    Pass 10 (M2-F28 d): every call supplied both halves or neither, so applying the
    ``A_va`` / ``A_pa`` term only when ``db_g`` is given (Y30 / Y59) survived; with only
    the accel half supplied they drop ``A db_a``: 0.18 (Y30) / 0.29 (Y59) absolute here.
    Measured clean: 0 (the retyped residual performs the same operations)."""
    rng = _rng(281)
    failures = []
    for Dt, g in ((0.2, G_A), (0.73, G_B)):
        R_i = _rand_rot(rng)
        v_i, v_j, p_i, p_j, dvbar, dpbar = (rng.normal(size=3) for _ in range(6))
        A_vg, A_va, A_pg, A_pa = (rng.normal(size=(3, 3)) for _ in range(4))
        db = BIAS_BOTH
        z3 = np.zeros(3)
        for label, use_g, use_a in (("gyro half only", True, False),
                                    ("accel half only", False, True),
                                    ("both halves", True, True), ("neither half", False, False)):
            kw_v = dict(A_vg=A_vg if use_g else None, db_g=db[0:3] if use_g else None,
                        A_va=A_va if use_a else None, db_a=db[3:6] if use_a else None)
            kw_p = dict(A_pg=A_pg if use_g else None, db_g=db[0:3] if use_g else None,
                        A_pa=A_pa if use_a else None, db_a=db[3:6] if use_a else None)
            db_ref = np.concatenate([db[0:3] if use_g else z3, db[3:6] if use_a else z3])
            where = f"(Dt={Dt}, {label})"
            _check(failures, f"velocity_residual_eq21b {where}",
                   mabs(B2.velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g, **kw_v),
                        _ieq21b(R_i, v_i, v_j, dvbar, Dt, g, A_vg, A_va, db_ref)), TIER0)
            _check(failures, f"position_residual_eq21c {where}",
                   mabs(B2.position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, **kw_p),
                        _ieq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A_pg, A_pa, db_ref)),
                   TIER0)
    _assert_no_failures(failures)


@pytest.mark.claim("M2-F28")
def test_f28_eq21c_at_large_velocity_and_duration():
    """Eq. (21b) / (21c) at |v_i| = 43 / 47 m/s and Dt = 2 s, against the retyped residuals at
    C10's own 1e-12, with ``p_j`` / ``v_j`` built so the residual itself is O(1) while
    ``v_i Dt`` is ~60 m per component and ``g Dt^2 / 2`` ~20 m.

    Pass 10 (M2-F28 e): on the fixture (|v| <= 1.2, Dt = 0.5) a 1e-13 relative error in
    ``v_i Dt`` (Y31) is <= 6e-14 absolute; here it is 8.3e-12 (8x the bound).  Measured
    clean: 0 (same operations as the retyped residual, so the bound's margin is not eaten
    by rounding)."""
    rng = _rng(283)
    failures = []
    Dt = 2.0
    for g, v_i in ((G_A, np.array([30.0, -25.0, 18.0])), (G_B, np.array([-22.0, 31.0, -27.0]))):
        R_i = _rand_rot(rng)
        p_i = np.array([120.0, -45.0, 8.0]) + rng.normal(size=3)
        dvbar, dpbar = rng.normal(size=3), rng.normal(size=3)
        A_vg, A_va, A_pg, A_pa = (rng.normal(size=(3, 3)) for _ in range(4))
        v_j = v_i + g * Dt + R_i @ (dvbar + 0.5 * rng.normal(size=3))
        p_j = p_i + v_i * Dt + 0.5 * g * Dt**2 + R_i @ (dpbar + 0.5 * rng.normal(size=3))
        db = BIAS_BOTH
        where = f"(|v_i| = {np.linalg.norm(v_i):.0f} m/s, Dt = {Dt})"
        _check(failures, f"velocity_residual_eq21b at large v_i Dt {where}",
               mabs(B2.velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g, A_vg, A_va,
                                               db[0:3], db[3:6]),
                    _ieq21b(R_i, v_i, v_j, dvbar, Dt, g, A_vg, A_va, db)), TIER0)
        _check(failures, f"position_residual_eq21c at large v_i Dt {where}",
               mabs(B2.position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A_pg, A_pa,
                                               db[0:3], db[3:6]),
                    _ieq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g, A_pg, A_pa, db)), TIER0)
    _assert_no_failures(failures)
