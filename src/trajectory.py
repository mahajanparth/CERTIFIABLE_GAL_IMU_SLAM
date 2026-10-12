"""Fixture ``M2_TRAJ_V1`` -- the shared trajectory every M2 baseline is measured on.

Three keyframes, two edges, piecewise-constant *body* twist on each edge.  Because the
twist is constant over an edge the noise-free measurement stream is literally constant,
so there is **no RNG anywhere in this module**: two independent generations agree
*bitwise*, not merely to within a tolerance (claim M2-C01).

The constant-twist assumption also buys a closed form for the gravity-free increment
(claim M2-C02)::

    dR = Exp(w T)        dv = M1(T) a        dp = M2(T) a

with ``M1(T) = int_0^T Exp(w s) ds`` and ``M2(T) = int_0^T M1(s) ds``.  That closed form
is the *exact-arithmetic* reference for the whole milestone: no discretization enters it,
so it is the only object in M2 that may be compared against at Tier 0.

Chaining (claim M2-C02 / M2-C03)::

    R_j = R_i dR
    v_j = v_i + g_W Dt + R_i dv
    p_j = p_i + v_i Dt + 0.5 g_W Dt^2 + R_i dp
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# SO(3) helpers.  Kept here rather than in a baseline module because all three baselines
# and the fixture itself need them, and because the exact ground truth must not be
# routed through any one baseline's conventions.
# --------------------------------------------------------------------------------------

#: below this rotation angle the closed forms are evaluated by their Taylor series
SMALL_ANGLE = 1e-8


def hat(a) -> np.ndarray:
    """The 3x3 skew-symmetric matrix with ``hat(a) @ b == cross(a, b)``."""
    a = np.asarray(a, dtype=float)
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def vee(A) -> np.ndarray:
    """Inverse of :func:`hat`."""
    A = np.asarray(A, dtype=float)
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def so3_exp(w) -> np.ndarray:
    """Rodrigues' formula, ``Exp: so(3) -> SO(3)``."""
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < SMALL_ANGLE:
        # series: I + K + K^2/2 is already exact to double precision at this scale
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th**2) * (K @ K)


def so3_log(R) -> np.ndarray:
    """``Log: SO(3) -> so(3)``, returning the rotation vector."""
    R = np.asarray(R, dtype=float)
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = float(np.arccos(c))
    if th < SMALL_ANGLE:
        return vee(0.5 * (R - R.T))
    return (th / (2.0 * np.sin(th))) * vee(R - R.T)


def M1(w, T: float) -> np.ndarray:
    """``int_0^T Exp(w s) ds`` in closed form (claim M2-C02).

    ``M1(T) = T I + ((1 - cos(th T)) / th) K + (T - sin(th T) / th) K^2`` with
    ``th = ||w||`` and ``K = hat(w) / th`` the *unit-axis* skew.
    """
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < SMALL_ANGLE:
        K = hat(w)
        return T * np.eye(3) + 0.5 * T**2 * K + (T**3 / 6.0) * (K @ K)
    K = hat(w) / th
    return (T * np.eye(3)
            + ((1.0 - np.cos(th * T)) / th) * K
            + (T - np.sin(th * T) / th) * (K @ K))


def M2(w, T: float) -> np.ndarray:
    """``int_0^T M1(s) ds`` in closed form (claim M2-C02).

    ``M2(T) = (T^2/2) I + ((T - sin(th T)/th)/th) K + (T^2/2 - (1 - cos(th T))/th^2) K^2``.
    """
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    if th < SMALL_ANGLE:
        K = hat(w)
        return 0.5 * T**2 * np.eye(3) + (T**3 / 6.0) * K + (T**4 / 24.0) * (K @ K)
    K = hat(w) / th
    return (0.5 * T**2 * np.eye(3)
            + ((T - np.sin(th * T) / th) / th) * K
            + (0.5 * T**2 - (1.0 - np.cos(th * T)) / th**2) * (K @ K))


# --------------------------------------------------------------------------------------
# Fixture constants.  Every one of these is a literal; nothing is sampled.
# --------------------------------------------------------------------------------------

G_W = np.array([0.0, 0.0, -9.81])      #: gravity VECTOR in the world frame (z-up)
DT = 2.5e-3                            #: nominal IMU sample period
T_EDGE = 0.5                           #: duration of one keyframe-to-keyframe edge
STEPS_PER_EDGE = 200                   #: == T_EDGE / DT
N_EDGES = 2
N_KEYFRAMES = N_EDGES + 1

B_G = np.array([0.0100, -0.0150, 0.0080])   #: true gyro bias,  project order b = [b_g; b_a]
B_A = np.array([0.0600, -0.0400, 0.0900])   #: true accel bias
BIAS_TRUE = np.concatenate([B_G, B_A])      #: the project convention is [b_g; b_a]

#: per-edge constant body angular rate
OMEGA_BODY = (np.array([0.35, -0.22, 0.17]),
              np.array([-0.28, 0.41, -0.33]))
#: per-edge constant body specific force
F_BODY = (np.array([0.60, 0.45, 9.95]),
          np.array([-0.75, 0.90, 9.60]))

ROT_VEC_0 = np.array([0.10, -0.20, 0.30])   #: R_0 = Exp(ROT_VEC_0)
R_0 = so3_exp(ROT_VEC_0)
P_0 = np.zeros(3)
V_0 = np.array([0.50, -0.30, 0.20])


# --------------------------------------------------------------------------------------
# Reference values, transcribed at 15 significant digits from the fixture definition
# (docs/02_baseline_reproduction.md Sec. 0.4).  These are the Tier-0 ground truth.
# --------------------------------------------------------------------------------------

REF_DR_01 = np.array([
    [9.903776532493884e-01, -9.387914277777526e-02, -1.016799414611560e-01],
    [7.470913760838102e-02, 9.811536442685306e-01, -1.782023319050390e-01],
    [1.164931272738697e-01, 1.688911865370475e-01, 9.787262734840944e-01]])
REF_DV_01 = np.array([2.860848012625113e-02, -2.059158118715534e-01, 4.976091490259237e+00])
REF_DP_01 = np.array([2.924832251685720e-02, -1.526611626991823e-02, 1.245394361998341e+00])

REF_DR_12 = np.array([
    [9.656306111940006e-01, 1.483233923728816e-01, 2.134424840562463e-01],
    [-1.768115211449302e-01, 9.767603374607809e-01, 1.211483766045474e-01],
    [-1.905130145567926e-01, -1.547236712287475e-01, 9.694149662791377e-01]])
REF_DV_12 = np.array([1.707829111988740e-01, 7.855781189893493e-01, 4.753841859545298e+00])
REF_DP_12 = np.array([-3.379126843155931e-03, 1.697985087871268e-01, 1.194510739753956e+00])

REF_V_KF1 = np.array([-3.092347809743656e-01, -1.121267155202117e+00, 1.401383484057339e-01])
REF_P_KF1 = np.array([5.715027067778627e-02, -3.148113358346768e-01, 9.348023290152563e-02])

REF_V_KF2 = np.array([-1.520564406279614e+00, -1.903478183400628e+00, -1.641944548085670e-01])
REF_P_KF2 = np.array([-4.312534482841444e-01, -1.111986889144151e+00, 7.234717707859262e-02])
REF_R_KF2 = np.array([
    [9.677181225278726e-01, -2.412143599148208e-01, -7.305660751781723e-02],
    [2.301509945763355e-01, 9.638954740062700e-01, -1.339254825854730e-01],
    [1.027236828908148e-01, 1.127880656856546e-01, 9.882948432589516e-01]])

#: ``REF_R_KF1`` is not tabulated in Sec. 0.4; it is pinned by the tabulated ``REF_DR_01``.
REF_R_KF1 = R_0 @ REF_DR_01

REF_INCREMENTS = ((REF_DR_01, REF_DV_01, REF_DP_01),
                  (REF_DR_12, REF_DV_12, REF_DP_12))


# --------------------------------------------------------------------------------------
# Fixture objects
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NavTriple:
    """A navigation state ``(R, v, p)``.  No clock -- the Gal(3) time coordinate is
    carried separately, because only Baseline 3 has one (claim M2-C18)."""

    R: np.ndarray
    v: np.ndarray
    p: np.ndarray

    def as_tuple(self):
        return self.R, self.v, self.p


@dataclass(frozen=True)
class Edge:
    """One keyframe-to-keyframe edge of the fixture."""

    index: int
    omega_body: np.ndarray      #: true (bias-free) body rate, constant over the edge
    f_body: np.ndarray          #: true (bias-free) body specific force, constant
    dt: float
    n_steps: int

    @property
    def duration(self) -> float:
        return self.dt * self.n_steps

    # -- the noise-free measurement stream -------------------------------------------
    def gyro_stream(self) -> np.ndarray:
        """``omega_k = w_e + b_g`` for every step.  Exactly constant -- no RNG."""
        return np.tile(self.omega_body + B_G, (self.n_steps, 1))

    def accel_stream(self) -> np.ndarray:
        """``a_k = f_e + b_a`` for every step.  Exactly constant -- no RNG."""
        return np.tile(self.f_body + B_A, (self.n_steps, 1))

    # -- the exact increment ----------------------------------------------------------
    def exact_increment(self):
        """``(dR, dv, dp)`` in closed form (claim M2-C02).  No discretization."""
        T = self.duration
        dR = so3_exp(self.omega_body * T)
        dv = M1(self.omega_body, T) @ self.f_body
        dp = M2(self.omega_body, T) @ self.f_body
        return dR, dv, dp


def edges(n_steps: int = STEPS_PER_EDGE) -> tuple[Edge, ...]:
    """The fixture's edges, optionally re-discretized.

    ``T_EDGE`` is held fixed and ``dt = T_EDGE / n_steps``, which is what makes the
    ``n = 100, 200, 400, 800, 1600`` sweep of claims M2-C04 and M2-C23 a *refinement* of
    one and the same continuous trajectory rather than five different trajectories.
    """
    dt = T_EDGE / n_steps
    return tuple(Edge(index=e, omega_body=OMEGA_BODY[e], f_body=F_BODY[e],
                      dt=dt, n_steps=n_steps)
                 for e in range(N_EDGES))


def initial_state() -> NavTriple:
    """KF0.  ``p_0 = 0`` exactly."""
    return NavTriple(R=R_0.copy(), v=V_0.copy(), p=P_0.copy())


def chain(state: NavTriple, dR, dv, dp, Dt: float) -> NavTriple:
    """The component endpoint relation of claims M2-C02 / M2-C03.

    ``g_W`` is the gravity *vector*; for z-up it is ``[0, 0, -9.81]``.
    """
    R_i, v_i, p_i = state.as_tuple()
    return NavTriple(R=R_i @ dR,
                     v=v_i + G_W * Dt + R_i @ dv,
                     p=p_i + v_i * Dt + 0.5 * G_W * Dt**2 + R_i @ dp)


def exact_states(n_steps: int = STEPS_PER_EDGE) -> list[NavTriple]:
    """The exact keyframe states KF0, KF1, KF2 from the closed-form increments.

    Independent of ``n_steps`` up to round-off -- the closed form has no discretization.
    """
    states = [initial_state()]
    for edge in edges(n_steps):
        dR, dv, dp = edge.exact_increment()
        states.append(chain(states[-1], dR, dv, dp, edge.duration))
    return states


def measurement_streams(n_steps: int = STEPS_PER_EDGE):
    """``[(gyro, accel), ...]`` per edge.  Deterministic by construction (claim M2-C01)."""
    return [(e.gyro_stream(), e.accel_stream()) for e in edges(n_steps)]


# --------------------------------------------------------------------------------------
# A time-varying stream.  Claim M2-C19's domain note is the reason this exists: a
# CONSTANT-input stream cannot distinguish U_{k+1} = U_k V_k from U_{k+1} = V_k U_k,
# because constant-input increments commute.  Any test of accumulation ORDER must use
# this, not the fixture's own per-edge constant inputs.
# --------------------------------------------------------------------------------------

TIME_VARYING_STEPS = 40
_PHASE = np.array([0.0, 1.0, 2.0])


def time_varying_stream(n_steps: int = TIME_VARYING_STEPS, dt: float = DT):
    """``(gyro, accel, dt)``; closed-form sinusoids, still no RNG.

    ``a_k = [0.5, 0.4, 9.81] + 0.6 sin(0.7 k dt + [0,1,2])``
    ``w_k = [0.3, -0.2, 0.15] + 0.4 cos(1.1 k dt + [0,1,2])``
    """
    k = np.arange(n_steps, dtype=float)[:, None]
    accel = np.array([0.5, 0.4, 9.81]) + 0.6 * np.sin(0.7 * k * dt + _PHASE)
    gyro = np.array([0.3, -0.2, 0.15]) + 0.4 * np.cos(1.1 * k * dt + _PHASE)
    return gyro, accel, dt
