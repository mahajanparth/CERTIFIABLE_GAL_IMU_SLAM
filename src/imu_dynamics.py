"""IMU measurement model, ground-truth integration, and the preintegrated increment.

Conventions fixed for this project:

* measurement model   omega_m = omega + b_g + n_g,   a_m = a + b_a + n_a
* bias vector         b = [b_g ; b_a]   -- GYRO FIRST (gtsam's ConstantBias is the
  other way round; the one conversion lives here, in :func:`bias_to_gtsam` /
  :func:`bias_from_gtsam`, and nowhere else)
* gravity             g in the world frame, default (0, 0, -9.81)

The gravity-free increment D is a genuine Gal(3) element: D = (dR, dv, dp, Dt), and
chaining increments IS the Gal(3) product (M1-C45).
"""

from __future__ import annotations

import numpy as np

from . import gal3
from .gal3 import G_DEFAULT, compose, exp, gal, skew

__all__ = [
    "G_DEFAULT",
    "BIAS_SWAP",
    "correct_measurements",
    "bias_to_gtsam",
    "bias_from_gtsam",
    "imu_step_element",
    "integrate_truth",
    "integrate_increment",
    "euler_increment",
    "increment_blocks",
    "preintegrate_constant",
    "chain",
]

_I3 = np.eye(3)

#: the involutive block swap between b = [b_g ; b_a] and gtsam's [b_a ; b_g].
BIAS_SWAP = np.block([[np.zeros((3, 3)), _I3], [_I3, np.zeros((3, 3))]])


# --------------------------------------------------------------------------
# measurement model
# --------------------------------------------------------------------------
def correct_measurements(omega_m, a_m, b):
    """Bias-corrected, noise-free IMU inputs (M1-C44).

    omega = omega_m - b_g,  a = a_m - b_a,  with b = [b_g ; b_a].
    """
    b = np.asarray(b, dtype=float).reshape(6)
    omega_m = np.asarray(omega_m, dtype=float).reshape(3)
    a_m = np.asarray(a_m, dtype=float).reshape(3)
    return omega_m - b[0:3], a_m - b[3:6]


def bias_to_gtsam(b):
    """b = [b_g ; b_a] -> gtsam.imuBias.ConstantBias (which takes ACCEL FIRST)."""
    import gtsam

    b = np.asarray(b, dtype=float).reshape(6)
    return gtsam.imuBias.ConstantBias(b[3:6], b[0:3])


def bias_from_gtsam(cb) -> np.ndarray:
    """gtsam.imuBias.ConstantBias -> b = [b_g ; b_a]."""
    return np.concatenate([np.asarray(cb.gyroscope(), dtype=float),
                           np.asarray(cb.accelerometer(), dtype=float)])


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------
def _so3_exp_batch(phi: np.ndarray) -> np.ndarray:
    """Rodrigues on a stack of axis-angle vectors, shape (..., 3) -> (..., 3, 3)."""
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    th = np.linalg.norm(phi, axis=-1)
    K = np.zeros(phi.shape[:-1] + (3, 3))
    K[..., 0, 1] = -phi[..., 2]
    K[..., 0, 2] = phi[..., 1]
    K[..., 1, 0] = phi[..., 2]
    K[..., 1, 2] = -phi[..., 0]
    K[..., 2, 0] = -phi[..., 1]
    K[..., 2, 1] = phi[..., 0]
    t2 = th * th
    small = th < gal3.THETA_SMALL
    s = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0,
                 np.sin(np.where(small, 1.0, th)) / np.where(small, 1.0, th))
    c = np.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0,
                 (1.0 - np.cos(np.where(small, 1.0, th))) / np.where(small, 1.0, t2))
    return _I3 + s[..., None, None] * K + c[..., None, None] * (K @ K)


def imu_step_element(omega, a, dt: float) -> np.ndarray:
    """One integration step as a Gal(3) element: (Exp(omega dt), a dt, a dt^2/2, dt).

    The increment recursion D <- D . S(omega, a, dt) is exactly left multiplication
    by this element, which is why the increments chain as a Gal(3) product.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    a = np.asarray(a, dtype=float).reshape(3)
    return gal(gal3.so3_exp(omega * dt), a * dt, 0.5 * a * dt * dt, dt)


def integrate_truth(R0, v0, p0, omega, acc, dt, n, g=G_DEFAULT):
    """Forward-integrate the navigation state AND the matching gravity-free increment.

    Rdot = R [omega]x,  vdot = g + R a,  pdot = v, stepped as

        p <- p + v dt + (g + R a) dt^2 / 2
        v <- v + (g + R a) dt
        R <- R Exp(omega dt)

    and the increment with the same stepping and g = 0.  Batched over the leading
    axis; constant (omega, acc) per instance.

    Returns ``((R, v, p), (dR, dv, dp, Dt))``.
    """
    R = np.array(R0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    p = np.array(p0, dtype=float, copy=True)
    omega = np.asarray(omega, dtype=float)
    acc = np.asarray(acc, dtype=float)
    g = np.asarray(g, dtype=float).reshape(3)

    E = _so3_exp_batch(omega * dt)
    dR = np.broadcast_to(_I3, R.shape).copy()
    dv = np.zeros_like(v)
    dp = np.zeros_like(p)
    half = 0.5 * dt * dt
    for _ in range(int(n)):
        f = g + np.einsum("...ij,...j->...i", R, acc)
        p = p + v * dt + half * f
        v = v + f * dt
        R = R @ E
        df = np.einsum("...ij,...j->...i", dR, acc)
        dp = dp + dv * dt + half * df
        dv = dv + df * dt
        dR = dR @ E
    return (R, v, p), (dR, dv, dp, int(n) * dt)


def integrate_increment(omega, acc, dt, n, D0=None):
    """The gravity-free increment alone, optionally continuing from ``D0``.

    ``D0`` is ``(dR, dv, dp, t0)``; continuing from the end of one segment is
    bit-identical to integrating both segments from the identity.
    """
    omega = np.asarray(omega, dtype=float)
    acc = np.asarray(acc, dtype=float)
    E = _so3_exp_batch(omega * dt)
    if D0 is None:
        dR = np.broadcast_to(_I3, E.shape).copy()
        dv = np.zeros(E.shape[:-1])
        dp = np.zeros(E.shape[:-1])
        t0 = 0.0
    else:
        dR = np.array(D0[0], dtype=float, copy=True)
        dv = np.array(D0[1], dtype=float, copy=True)
        dp = np.array(D0[2], dtype=float, copy=True)
        t0 = float(D0[3])
    half = 0.5 * dt * dt
    for _ in range(int(n)):
        df = np.einsum("...ij,...j->...i", dR, acc)
        dp = dp + dv * dt + half * df
        dv = dv + df * dt
        dR = dR @ E
    return dR, dv, dp, t0 + int(n) * dt


def euler_increment(omega, acc, dt, n):
    """Plain explicit Euler on the increment ODE Ddot = D xi^ (M1-C41).

    Every block uses the pre-update value, so the error is FIRST order in dt.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    acc = np.asarray(acc, dtype=float).reshape(3)
    W = skew(omega)
    dR = _I3.copy()
    dv = np.zeros(3)
    dp = np.zeros(3)
    for _ in range(int(n)):
        dRa = dR @ acc
        dp = dp + dv * dt
        dv = dv + dRa * dt
        dR = dR + (dR @ W) * dt
    return dR, dv, dp, int(n) * dt


# --------------------------------------------------------------------------
# closed-form preintegration
# --------------------------------------------------------------------------
def increment_blocks(omega, acc, Dt):
    """(dR, dv, dp, Dt) for constant (omega, a) over [0, Dt] (M1-C42).

    dR = Exp(omega Dt),  dv = Dt Gamma1(omega Dt) a,  dp = Dt^2 Gamma2(omega Dt) a.
    dp has NO Gamma1 w term: the generator's w slot is zero, so position is driven
    entirely by the alpha * Gamma2 * u coupling of the exponential.
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    acc = np.asarray(acc, dtype=float).reshape(3)
    Dt = float(Dt)
    phi = omega * Dt
    return (
        gal3.so3_exp(phi),
        Dt * (gal3.gamma1(phi) @ acc),
        Dt * Dt * (gal3.gamma2(phi) @ acc),
        Dt,
    )


def increment_generator(omega, acc) -> np.ndarray:
    """xi = (omega, a, 0, 1), the left-invariant generator of the increment ODE."""
    omega = np.asarray(omega, dtype=float).reshape(3)
    acc = np.asarray(acc, dtype=float).reshape(3)
    return np.concatenate([omega, acc, np.zeros(3), [1.0]])


def preintegrate_constant(omega, acc, Dt) -> np.ndarray:
    """D(Dt) = exp(Dt * (omega, a, 0, 1)) as a 5x5 Gal(3) element (M1-C41)."""
    return exp(float(Dt) * increment_generator(omega, acc))


def chain(D_ij, D_jk) -> np.ndarray:
    """D_ik = D_ij D_jk -- preintegration chaining IS the Gal(3) product (M1-C45)."""
    return compose(D_ij, D_jk)
