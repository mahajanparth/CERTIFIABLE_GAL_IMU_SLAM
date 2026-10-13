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
    # sin(th)/th carries its th^6 Taylor term: truncating after th^4 leaves th^7/5040
    # = 1.5e-13 in the matrix just below the switch (M1-C42), whereas the next omitted
    # term, th^9/9!, is 5e-18.  (1 - cos th)/th^2 stops at th^4: its first omitted
    # term th^6/8! enters multiplied by |K K| <= th^2, i.e. <= 1e-15 at the switch.
    s = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0,
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


def _skew_batch(a: np.ndarray) -> np.ndarray:
    """[a]x on a stack of 3-vectors, shape (..., 3) -> (..., 3, 3)."""
    a = np.asarray(a, dtype=float)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1] = -a[..., 2]
    K[..., 0, 2] = a[..., 1]
    K[..., 1, 0] = a[..., 2]
    K[..., 1, 2] = -a[..., 0]
    K[..., 2, 0] = -a[..., 1]
    K[..., 2, 1] = a[..., 0]
    return K


def _inputs(omega, acc, n, per_step):
    """Return (omega_k, acc_k) as functions of the step index k.

    ``per_step=False``: (omega, acc) are constant, shape (..., 3).
    ``per_step=True``:  (omega, acc) are sampled streams, shape (n, ..., 3); step k
    uses sample k (zero-order hold on [k dt, (k+1) dt)).
    """
    omega = np.asarray(omega, dtype=float)
    acc = np.asarray(acc, dtype=float)
    if per_step:
        if omega.shape[0] != int(n) or acc.shape[0] != int(n):
            raise ValueError("per-step streams must have leading length n")
        return omega, acc
    return (np.broadcast_to(omega, (1,) + omega.shape),
            np.broadcast_to(acc, (1,) + acc.shape))


def integrate_truth(R0, v0, p0, omega, acc, dt, n, g=G_DEFAULT, per_step=False):
    """Forward-integrate the navigation state AND the matching gravity-free increment.

    Rdot = R [omega]x,  vdot = g + R a,  pdot = v, stepped as

        p <- p + v dt + (g + R a) dt^2 / 2
        v <- v + (g + R a) dt
        R <- R Exp(omega dt)

    and the increment with the same stepping and g = 0.  Batched over the leading
    axis.  With ``per_step=True`` the inputs are sampled streams of shape
    (n, batch, 3) -- the only way an accumulation-order defect (R Exp vs Exp R)
    becomes visible, since constant-input factors all commute.

    This is a FIRST-ORDER scheme: it is not an accurate ODE solution, and the
    endpoint relation holds for it as an exact algebraic identity of the recursion
    for any dt.  Accuracy claims must be checked against an independent ODE solve.

    Returns ``((R, v, p), (dR, dv, dp, Dt))``.
    """
    R = np.array(R0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    p = np.array(p0, dtype=float, copy=True)
    om_s, acc_s = _inputs(omega, acc, n, per_step)
    g = np.asarray(g, dtype=float).reshape(3)

    E_s = _so3_exp_batch(om_s * dt).reshape(om_s.shape + (3,))
    dR = np.broadcast_to(_I3, R.shape).copy()
    dv = np.zeros_like(v)
    dp = np.zeros_like(p)
    half = 0.5 * dt * dt
    for k in range(int(n)):
        E = E_s[k if per_step else 0]
        a_k = acc_s[k if per_step else 0]
        f = g + np.einsum("...ij,...j->...i", R, a_k)
        p = p + v * dt + half * f
        v = v + f * dt
        R = R @ E
        df = np.einsum("...ij,...j->...i", dR, a_k)
        dp = dp + dv * dt + half * df
        dv = dv + df * dt
        dR = dR @ E
    return (R, v, p), (dR, dv, dp, int(n) * dt)


def integrate_increment(omega, acc, dt, n, D0=None, per_step=False):
    """The gravity-free increment alone, optionally continuing from ``D0``.

    ``D0`` is ``(dR, dv, dp, t0)``; continuing from the end of one segment is
    bit-identical to integrating both segments from the identity.  ``per_step``
    as in :func:`integrate_truth`.
    """
    om_s, acc_s = _inputs(omega, acc, n, per_step)
    E_s = _so3_exp_batch(om_s * dt).reshape(om_s.shape + (3,))
    if D0 is None:
        dR = np.broadcast_to(_I3, E_s.shape[1:]).copy()
        dv = np.zeros(E_s.shape[1:-1])
        dp = np.zeros(E_s.shape[1:-1])
        t0 = 0.0
    else:
        dR = np.array(D0[0], dtype=float, copy=True)
        dv = np.array(D0[1], dtype=float, copy=True)
        dp = np.array(D0[2], dtype=float, copy=True)
        t0 = float(D0[3])
    half = 0.5 * dt * dt
    for k in range(int(n)):
        E = E_s[k if per_step else 0]
        a_k = acc_s[k if per_step else 0]
        df = np.einsum("...ij,...j->...i", dR, a_k)
        dp = dp + dv * dt + half * df
        dv = dv + df * dt
        dR = dR @ E
    return dR, dv, dp, t0 + int(n) * dt


def euler_increment(omega, acc, dt, n, per_step=False):
    """Plain explicit Euler on the increment ODE Ddot = D xi^ (M1-C41).

    Every block uses the pre-update value, so the error is FIRST order in dt.
    Batched over leading axes of (omega, acc), shape (..., 3); with
    ``per_step=True`` they are sampled streams of shape (n, ..., 3).
    """
    om_s, acc_s = _inputs(omega, acc, n, per_step)
    W_s = _skew_batch(om_s)
    dR = np.broadcast_to(_I3, W_s.shape[1:]).copy()
    dv = np.zeros(W_s.shape[1:-1])
    dp = np.zeros(W_s.shape[1:-1])
    for k in range(int(n)):
        W = W_s[k if per_step else 0]
        dRa = np.einsum("...ij,...j->...i", dR, acc_s[k if per_step else 0])
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
