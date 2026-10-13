"""The ONE place where GTSAM 4.3.0's conventions are converted to the project's.

Footguns handled here and nowhere else (CLAUDE.md "Ordering footguns"):

1. ``imuBias.ConstantBias`` is (accel, gyro); the project is ``[b_g; b_a]``.  Both bias
   vectors AND bias-Jacobian columns are swapped here.
2. ``Gal3(R, r, v, t)`` takes POSITION before velocity although the matrix stores velocity in
   column 3 and position in column 4.  Every Gal3 is built through :func:`gal3` with named
   arguments and read through the named accessors.
3. The Gal3 tangent is ``(omega, nu, rho, alpha)`` = (rot, VEL, POS, time); ``NavState`` (and
   therefore ``preintMeasCov`` and the IMU factor residual) is (rot, POS, VEL).  The Jacobian
   J returned here keeps the Gal3 row order; covariance blocks are returned by name.
"""
from __future__ import annotations

import numpy as np

import gtsam
from gtsam.imuBias import ConstantBias

#: rows of a NavState-ordered 9-vector / 9x9 covariance, by block name
NAV_ROWS = {"rot": slice(0, 3), "pos": slice(3, 6), "vel": slice(6, 9)}
#: rows of a Gal3-ordered 10-vector, by block name
GAL_ROWS = {"rot": slice(0, 3), "vel": slice(3, 6), "pos": slice(6, 9), "time": slice(9, 10)}


def to_gtsam_bias(b) -> ConstantBias:
    """Project ``[b_g; b_a]`` -> ``ConstantBias(accel, gyro)``."""
    b = np.asarray(b, dtype=float)
    return ConstantBias(b[3:6], b[0:3])


def from_gtsam_bias(cb: ConstantBias) -> np.ndarray:
    """``ConstantBias`` -> project ``[b_g; b_a]``."""
    return np.concatenate([np.asarray(cb.gyroscope()), np.asarray(cb.accelerometer())])


def gal3(R, p, v, t: float) -> gtsam.Gal3:
    """Gal3 from named components; note the GTSAM constructor order (R, r=p, v, t)."""
    return gtsam.Gal3(gtsam.Rot3(np.asarray(R, dtype=float)), np.asarray(p, dtype=float),
                      np.asarray(v, dtype=float), float(t))


def nav_to_gal3(ns: gtsam.NavState, t: float = 0.0) -> gtsam.Gal3:
    return gal3(ns.attitude().matrix(), ns.position(), ns.velocity(), t)


def make_params(g_w, sigma_g, sigma_a, sigma_int):
    params = gtsam.PreintegrationParams(np.asarray(g_w, dtype=float))
    params.setGyroscopeCovariance(sigma_g**2 * np.eye(3))
    params.setAccelerometerCovariance(sigma_a**2 * np.eye(3))
    params.setIntegrationCovariance(sigma_int**2 * np.eye(3))
    return params


def integrate(params, bbar, gyro, acc, dt) -> gtsam.PreintegratedImuMeasurementsG:
    """Galilean PIM at nominal bias ``bbar`` (project order).  GTSAM's argument order is
    ``integrateMeasurement(measuredAcc, measuredOmega, dt)``."""
    pim = gtsam.PreintegratedImuMeasurementsG(params, to_gtsam_bias(bbar))
    for w, a in zip(gyro, acc):
        pim.integrateMeasurement(np.asarray(a, dtype=float), np.asarray(w, dtype=float), dt)
    return pim


def gravity_left(g_w, Dt) -> gtsam.Gal3:
    """``G = X(I, v = g Dt, p = -1/2 g Dt^2, t = -Dt)`` (CLAUDE.md corrected endpoint)."""
    g_w = np.asarray(g_w, dtype=float)
    return gal3(np.eye(3), p=-0.5 * g_w * Dt**2, v=g_w * Dt, t=-Dt)


def increment(pim, g_w, b) -> gtsam.Gal3:
    """``U(b) = G^-1 X_j`` with ``X_j = pim.predict(identity, b)`` lifted at t = 0."""
    Dt = pim.deltaTij()
    Xj = nav_to_gal3(pim.predict(gtsam.NavState(), to_gtsam_bias(b)), 0.0)
    return gravity_left(g_w, Dt).inverse().compose(Xj)


def bias_jacobian(pim, g_w, bbar, h: float = 1e-3) -> np.ndarray:
    """J (10x6): rows Gal3 (omega, nu, rho, alpha), columns PROJECT order [b_g; b_a].

    Central differences of ``Log(Ubar^-1 U(bbar + d))``.  ``pim.predict`` applies the
    first-order right correction ``Ubar Exp(J d)`` (GTSAM doc Eq. (36)), so this map is
    exactly linear in ``d`` (measured: 1e-6 vs 1e-2 steps agree to 7e-10, the 1e-6 error being
    pure round-off), and h = 1e-3 has round-off ~1e-13 and no truncation error.
    The columns are produced by perturbing project-order bias entries, so the swap to
    GTSAM's (accel, gyro) happens inside :func:`to_gtsam_bias` only.
    """
    bbar = np.asarray(bbar, dtype=float)
    Ubar_inv = increment(pim, g_w, bbar).inverse()
    J = np.zeros((10, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = h
        lp = gtsam.Gal3.Logmap(Ubar_inv.compose(increment(pim, g_w, bbar + d)))
        lm = gtsam.Gal3.Logmap(Ubar_inv.compose(increment(pim, g_w, bbar - d)))
        J[:, k] = (lp - lm) / (2 * h)
    return J


def cov_blocks(pim):
    """``preintMeasCov`` blocks by name (NavState order (rot, pos, vel) resolved here)."""
    C = np.asarray(pim.preintMeasCov())
    return {(a, b): C[NAV_ROWS[a], NAV_ROWS[b]] for a in NAV_ROWS for b in NAV_ROWS}
