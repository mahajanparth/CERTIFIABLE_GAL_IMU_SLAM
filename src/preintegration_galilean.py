"""Baseline 3 -- a thin wrapper over GTSAM 4.3.0's Galilean preintegration.

This is the only baseline backed by shipped, third-party code; the wrapper exists to pin
the conventions at the boundary, not to re-derive anything.  On this fixture Baseline 3
*is* ground truth: its increment is exact zero-order-hold and reproduces the closed form
of :mod:`src.trajectory` to ~1e-14 (claim M2-C19), which is precisely why the three
baselines cannot share one tolerance.

Four ordering footguns are handled here and nowhere else:

1. ``Gal3.matrix()`` is ``[[R, v, r], [0, 1, t], [0, 0, 1]]`` -- **velocity in column 3,
   position in column 4** -- while the constructor is ``Gal3(R, r, v, t)`` with
   **position first**.  The two orders are opposite (claim M2-C14).  Everything below
   goes through the named accessors.
2. The ``Gal3`` tangent is ``(omega, nu, rho, alpha)`` = (rotation, VELOCITY, POSITION,
   time); ``NavState``'s is (rotation, position, velocity).  Hence
   :data:`GAL3_BLOCK_FOR_NAVSTATE` (claim M2-C15).
3. That remap is a block permutation **only**.  It does not make the two charts agree --
   ``NavState``'s retraction is not the Gal(3) exponential, and the reordered Gal(3)
   logarithm differs from ``localCoordinates`` by 0.1076 on this fixture (claim M2-C16).
   It does NOT differ from ``NavState.Logmap``: against the *group* logarithm the gap is
   exactly 0.0, because NavState's group is Gal(3) at ``t = 0``.  The disagreement is with
   the retraction, and only with the retraction.
4. ``imuBias.ConstantBias`` orders ``(accelerometer, gyroscope)``, so ``.vector()`` is
   ``[b_a; b_g]`` -- opposite the project's ``[b_g; b_a]``.  Every crossing swaps the two
   3-blocks, including Jacobian *columns* (claim M2-C17).
"""
from __future__ import annotations

import numpy as np

import gtsam
from gtsam import Gal3, NavState, Rot3
from gtsam.imuBias import ConstantBias

from .trajectory import G_W, NavTriple

#: ``GAL3_BLOCK_FOR_NAVSTATE[n]`` is the Gal(3) tangent start index of NavState block
#: ``n`` (0 = rotation, 1 = position, 2 = velocity).  Reconstructed from the measured
#: block semantics: GTSAM does not export ``kGal3BlockForNavState`` to Python.
GAL3_BLOCK_FOR_NAVSTATE = (0, 6, 3)


# --------------------------------------------------------------------------------------
# Boundary adapters
# --------------------------------------------------------------------------------------


def to_gtsam_bias(bias) -> ConstantBias:
    """Project ``[b_g; b_a]`` -> ``ConstantBias(accel, gyro)``."""
    b = np.asarray(bias, dtype=float)
    return ConstantBias(b[3:6], b[0:3])


def from_gtsam_bias(cb: ConstantBias) -> np.ndarray:
    """``ConstantBias`` -> project ``[b_g; b_a]``."""
    return np.concatenate([cb.gyroscope(), cb.accelerometer()])


def swap_bias_blocks(v) -> np.ndarray:
    """Swap the two 3-blocks of a 6-vector.  An involution, and **not** the identity."""
    v = np.asarray(v, dtype=float)
    return np.concatenate([v[3:6], v[0:3]])


def gal3_tangent_to_navstate_order(xi) -> np.ndarray:
    """Reorder a 10-vector Gal(3) tangent into NavState's 9-vector block order.

    A permutation of blocks and nothing more -- it does **not** reconcile the charts.
    """
    xi = np.asarray(xi, dtype=float)
    out = np.empty(9)
    for n, start in enumerate(GAL3_BLOCK_FOR_NAVSTATE):
        out[3 * n:3 * n + 3] = xi[start:start + 3]
    return out


def gal3(R, p, v, t: float) -> Gal3:
    """``Gal3`` from components, via the *named* constructor order ``(R, r, v, t)``."""
    R = R.matrix() if isinstance(R, Rot3) else np.asarray(R, dtype=float)
    return Gal3(Rot3(R), np.asarray(p, dtype=float), np.asarray(v, dtype=float), float(t))


def gal3_components(X: Gal3):
    """``(R, p, v, t)`` via the named accessors -- never positional matrix unpacking."""
    return (X.rotation().matrix(), np.array(X.position()), np.array(X.velocity()),
            float(X.time()))


def navstate(state: NavTriple) -> NavState:
    return NavState(Rot3(state.R), state.p, state.v)


def navtriple(ns: NavState) -> NavTriple:
    return NavTriple(R=ns.attitude().matrix(),
                     v=np.array(ns.velocity()),
                     p=np.array(ns.position()))


def gal3_from_navstate(ns: NavState, t: float = 0.0) -> Gal3:
    """Lift a ``NavState`` to Gal(3) with an explicit clock value."""
    return gal3(ns.attitude().matrix(), ns.position(), ns.velocity(), t)


def gal3_lift(state: NavTriple, t: float = 0.0) -> Gal3:
    return gal3(state.R, state.p, state.v, t)


# --------------------------------------------------------------------------------------
# Parameters and the preintegrator
# --------------------------------------------------------------------------------------


def make_params(g_W=G_W, sigma_g: float = 0.0, sigma_a: float = 0.0,
                sigma_int: float = 0.0, combined: bool = False,
                sigma_bg: float = 0.0, sigma_ba: float = 0.0):
    """Preintegration params with ``n_gravity`` set to the gravity **vector**."""
    g_W = np.asarray(g_W, dtype=float)
    cls = (gtsam.PreintegrationCombinedParams if combined
           else gtsam.PreintegrationParams)
    params = cls(g_W)
    params.setGyroscopeCovariance(sigma_g**2 * np.eye(3))
    params.setAccelerometerCovariance(sigma_a**2 * np.eye(3))
    params.setIntegrationCovariance(sigma_int**2 * np.eye(3))
    if combined:
        params.setBiasOmegaCovariance(sigma_bg**2 * np.eye(3))
        params.setBiasAccCovariance(sigma_ba**2 * np.eye(3))
    return params


class GalileanPreintegration:
    """Thin wrapper over ``gtsam.PreintegratedImuMeasurementsG``.

    Holds the project's ``[b_g; b_a]`` bias convention on the outside and GTSAM's
    ``[b_a; b_g]`` on the inside.
    """

    def __init__(self, bias_hat, params=None, g_W=G_W,
                 sigma_g: float = 0.0, sigma_a: float = 0.0, combined: bool = False):
        self.bias_hat = np.array(bias_hat, dtype=float, copy=True)   # M2-F18: no aliasing
        self.combined = combined
        self.params = params if params is not None else make_params(
            g_W, sigma_g, sigma_a, combined=combined)
        pim_cls = (gtsam.PreintegratedCombinedMeasurementsG if combined
                   else gtsam.PreintegratedImuMeasurementsG)
        self.pim = pim_cls(self.params, to_gtsam_bias(self.bias_hat))

    def integrate_measurement(self, omega_meas, acc_meas, dt: float) -> None:
        """NB GTSAM's argument order is ``(measuredAcc, measuredOmega, dt)``."""
        self.pim.integrateMeasurement(np.asarray(acc_meas, dtype=float),
                                      np.asarray(omega_meas, dtype=float), float(dt))

    # -- increment accessors ----------------------------------------------------------
    @property
    def dR(self) -> np.ndarray:
        return self.pim.deltaRij().matrix()

    @property
    def dv(self) -> np.ndarray:
        return np.array(self.pim.deltaVij())

    @property
    def dp(self) -> np.ndarray:
        return np.array(self.pim.deltaPij())

    @property
    def dt_total(self) -> float:
        return float(self.pim.deltaTij())

    def increment(self):
        return self.dR, self.dv, self.dp

    def increment_gal3(self) -> Gal3:
        """``U_ij = Gal3(dR, dp, dv, Dt)`` -- position argument second (claim M2-C14)."""
        return gal3(self.dR, self.dp, self.dv, self.dt_total)

    def predict(self, state_i: NavTriple) -> NavTriple:
        """GTSAM's ``predict`` at the integration bias ``bias_hat``."""
        return navtriple(self.pim.predict(navstate(state_i), to_gtsam_bias(self.bias_hat)))

    def preint_cov(self) -> np.ndarray:
        """GTSAM's propagated covariance, in **NavState** (rot, pos, vel) block order."""
        return np.asarray(self.pim.preintMeasCov())


def integrate_edge(edge, bias_hat, **kwargs) -> GalileanPreintegration:
    pim = GalileanPreintegration(bias_hat, **kwargs)
    for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
        pim.integrate_measurement(omega, acc, edge.dt)
    return pim


def integrate_edge_states(edges_, bias_hat, state0, **kwargs):
    states = [state0]
    for edge in edges_:
        states.append(integrate_edge(edge, bias_hat, **kwargs).predict(states[-1]))
    return states


def integrate_stream(gyro, accel, dt: float, bias_hat, n_steps=None,
                     **kwargs) -> GalileanPreintegration:
    """Integrate an arbitrary (possibly time-varying) stream."""
    pim = GalileanPreintegration(bias_hat, **kwargs)
    n = len(gyro) if n_steps is None else n_steps
    for k in range(n):
        pim.integrate_measurement(gyro[k], accel[k], dt)
    return pim


def single_step(omega, acc, dt: float, bias_hat, **kwargs) -> Gal3:
    """The one-step Gal(3) increment ``V_k`` at ``bias_hat``."""
    pim = GalileanPreintegration(bias_hat, **kwargs)
    pim.integrate_measurement(omega, acc, dt)
    return pim.increment_gal3()


# --------------------------------------------------------------------------------------
# The Gal(3) endpoint relation -- claim M2-C18
# --------------------------------------------------------------------------------------


def endpoint_W(Dt: float, t_i: float = 0.0, g_W=G_W) -> Gal3:
    """``W_ij(t_i) = Gal3(I, 0.5 g Dt^2 - g Dt (t_i + Dt), g Dt, 0)``.

    The gravity/clock element of the endpoint relation ``X_j = W_ij(t_i) X_i U_ij``.
    It depends on ``t_i`` -- that dependence is the whole point.  The pre-flight F2 form
    :func:`endpoint_Hprime` drops it and is therefore correct only at ``t_i = 0``, wrong
    by exactly ``g Dt t_i`` elsewhere.  Benign under GTSAM, where ``NavState`` carries no
    clock; a live hazard wherever Gal(3) states are decision variables.
    """
    g_W = np.asarray(g_W, dtype=float)
    return gal3(np.eye(3), 0.5 * g_W * Dt**2 - g_W * Dt * (t_i + Dt), g_W * Dt, 0.0)


def endpoint_Hprime(Dt: float, g_W=G_W) -> Gal3:
    """Pre-flight F2's ``H' = Gal3(I, -0.5 g Dt^2, g Dt, 0)``.  Kept for the C18 contrast."""
    g_W = np.asarray(g_W, dtype=float)
    return gal3(np.eye(3), -0.5 * g_W * Dt**2, g_W * Dt, 0.0)


def time_shift(dt: float) -> Gal3:
    """``T(dt) = Gal3(I, 0, 0, dt)``."""
    return gal3(np.eye(3), np.zeros(3), np.zeros(3), float(dt))


def endpoint_W_form(X_i: Gal3, U_ij: Gal3, Dt: float, t_i: float, g_W=G_W) -> Gal3:
    """``X_j = W_ij(t_i) . X_i . U_ij``.  Correct for any ``t_i``; yields ``t_j = t_i + Dt``."""
    return endpoint_W(Dt, t_i, g_W).compose(X_i).compose(U_ij)


def endpoint_F2_form(X_i: Gal3, U_ij: Gal3, Dt: float, g_W=G_W) -> Gal3:
    """``X_j = T(-Dt) . H' . X_i . U_ij``.  Correct only at ``t_i = 0``; yields ``t_j = t_i``."""
    return (time_shift(-Dt).compose(endpoint_Hprime(Dt, g_W))
            .compose(X_i).compose(U_ij))


def endpoint_F2_swapped(X_i: Gal3, U_ij: Gal3, Dt: float, g_W=G_W) -> Gal3:
    """``H' . T(-Dt) . X_i . U_ij`` -- the swapped order.  Wrong by ``g Dt^2`` even at ``t_i = 0``."""
    return (endpoint_Hprime(Dt, g_W).compose(time_shift(-Dt))
            .compose(X_i).compose(U_ij))


# --------------------------------------------------------------------------------------
# Bias Jacobians -- claims M2-C20, M2-C21
# --------------------------------------------------------------------------------------

#: FD step for the 10x6 Gal(3) bias Jacobian (claim M2-C20's measured setting)
J_FD_EPS = 1e-7
#: FD step for the per-step C_k of the recursion (claim M2-C21's measured setting)
C_FD_EPS = 1e-6


def corrected_increment_gal3(pim: GalileanPreintegration, db_gtsam, Dt=None) -> Gal3:
    """GTSAM's bias-corrected increment as a ``Gal3``.

    ``predict()`` from the identity ``NavState`` returns the corrected increment plus the
    gravity terms of the PIM's own params, ``v = dv + g Dt`` and ``p = dp + g Dt^2 / 2``
    (the endpoint relation, claim M2-C03, at ``R_i = I``, ``v_i = p_i = 0``).  Those terms
    are removed here, so the result is the increment for ANY ``n_gravity``.  (Before
    pass 8 they were not: the function was correct only for a zero-gravity PIM -- every
    caller in the suite built one -- and off by ``|g| Dt`` in velocity otherwise.)
    ``db_gtsam`` is in GTSAM's ``[b_a; b_g]`` column order.
    """
    b = pim.bias_hat + swap_bias_blocks(db_gtsam)
    ns = pim.pim.predict(NavState(), to_gtsam_bias(b))
    g = np.asarray(pim.params.n_gravity, dtype=float)
    Dt_ij = pim.dt_total
    return gal3(ns.attitude().matrix(),
                np.asarray(ns.position(), dtype=float) - 0.5 * g * Dt_ij**2,
                np.asarray(ns.velocity(), dtype=float) - g * Dt_ij,
                pim.dt_total if Dt is None else Dt)


def bias_jacobian_fd(increment_at, eps: float = J_FD_EPS) -> np.ndarray:
    """10x6 right-chart Jacobian ``J = d/d(db) Log(Uhat^-1 U(db))`` by central differences.

    ``increment_at(db)`` must return the ``Gal3`` increment at bias offset ``db``, with
    ``db`` in GTSAM's ``[b_a; b_g]`` column order.  Structurally the result has
    ``J[0:3, 0:3] = 0`` (rotation does not see accel bias) and ``J[9, :] = 0`` (the clock
    does not see bias at all).
    """
    Uhat_inv = increment_at(np.zeros(6)).inverse()
    J = np.zeros((10, 6))
    for m in range(6):
        step = np.zeros(6)
        step[m] = eps
        plus = Gal3.Logmap(Uhat_inv.compose(increment_at(step)))
        minus = Gal3.Logmap(Uhat_inv.compose(increment_at(-step)))
        J[:, m] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    return J


def right_composition_model(Uhat: Gal3, J: np.ndarray, db) -> Gal3:
    """``U(db) = Uhat . Expmap_Gal3(J db)`` -- first-order Jacobian, **exact** retraction."""
    return Uhat.compose(Gal3.Expmap(np.asarray(J, dtype=float) @ np.asarray(db, dtype=float)))


def component_wise_model(Uhat: Gal3, J: np.ndarray, db) -> Gal3:
    """``(dR Exp(J_R db), dv + J_v db, dp + J_p db)`` using the SAME ``J``.

    The Forster-style componentwise correction.  It is *not* what the Galilean factor
    does: it drops the ``dR``-rotation of the velocity/position tangent blocks that the
    right Gal(3) composition supplies (claim M2-C20).
    """
    from .trajectory import so3_exp
    J = np.asarray(J, dtype=float)
    db = np.asarray(db, dtype=float)
    xi = J @ db
    R, p, v, t = gal3_components(Uhat)
    return gal3(R @ so3_exp(xi[0:3]), p + xi[6:9], v + xi[3:6], t)


def gal3_distance(A: Gal3, B: Gal3) -> float:
    """``||Log(A^-1 B)||_inf`` -- a chart-consistent distance on Gal(3)."""
    return float(np.abs(np.asarray(Gal3.Logmap(A.inverse().compose(B)))).max())


def bias_jacobian_recursion(gyro, accel, dt: float, bias_hat, n_steps: int,
                            eps: float = C_FD_EPS, **kwargs):
    """``J_{k+1} = Ad_{V_k^-1} J_k + C_k`` with ``J_i = 0`` (claim M2-C21).

    ``C_k = d/d(db) Log(Vhat_k^-1 V_k(db))`` is that step's own right-chart bias
    Jacobian.  **The sign convention matters**: ``C_k`` is the derivative with respect to
    the BIAS.  The ``-C_k`` form seen in the literature differentiates with respect to the
    MEASUREMENT, and the two differ by the sign hidden in ``(a_k - b_a)``.  Columns are in
    GTSAM's ``[b_a; b_g]`` order.

    Returns the list of ``J_k`` after each of the ``n_steps`` steps.
    """
    bias_hat = np.asarray(bias_hat, dtype=float)
    J = np.zeros((10, 6))
    out = []
    for k in range(n_steps):
        Vhat = single_step(gyro[k], accel[k], dt, bias_hat, **kwargs)

        def step_increment(db, _k=k):
            return single_step(gyro[_k], accel[_k], dt,
                               bias_hat + swap_bias_blocks(db), **kwargs)

        C = bias_jacobian_fd(step_increment, eps=eps)
        J = np.asarray(Vhat.inverse().AdjointMap()) @ J + C
        out.append(J.copy())
    return out
