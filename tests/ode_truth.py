"""Independent continuous-time ground truth for the IMU claims.

Everything here is built from ``scipy`` and ``numpy`` only -- nothing from ``src`` --
so that accuracy claims are checked against an authority that cannot share a defect
with the code under test:

* the navigation ODE  Rdot = R [omega(t)]x,  vdot = g + R a(t),  pdot = v
* the increment ODE   Ddot = D xi(t)^,  xi(t) = (omega(t), a(t), 0, 1),  D(0) = I5
  (identical to the navigation ODE with R0 = I, v0 = p0 = 0, g = 0)

are integrated with ``scipy.integrate.solve_ivp`` (DOP853, rtol = atol = 1e-13).

The inputs are deliberately TIME-VARYING, smooth, and non-commuting
(``omega(t)`` changes direction), because with constant inputs every per-step
factor Exp(omega dt) commutes with every other and an accumulation-order defect
(``dR @ E`` vs ``E @ dR``) is invisible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

RTOL = ATOL = 1e-13


def _skew(a: np.ndarray) -> np.ndarray:
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1] = -a[..., 2]
    K[..., 0, 2] = a[..., 1]
    K[..., 1, 0] = a[..., 2]
    K[..., 1, 2] = -a[..., 0]
    K[..., 2, 0] = -a[..., 1]
    K[..., 2, 1] = a[..., 0]
    return K


@dataclass
class Stream:
    """omega(t) = w0 + w1 * sin(fw t + pw),  a(t) = a0 + a1 * cos(fa t + pa).

    All parameters have shape (batch, 3); products are elementwise, so the three
    components oscillate at different frequencies and the DIRECTION of omega(t)
    changes -- consecutive rotations do not commute.
    """

    w0: np.ndarray
    w1: np.ndarray
    fw: np.ndarray
    pw: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    fa: np.ndarray
    pa: np.ndarray

    @classmethod
    def random(cls, r: np.random.Generator, n: int, w_scale=0.8, w_amp=1.5,
               a_scale=2.0, a_amp=2.0, f_lo=2.0, f_hi=12.0) -> "Stream":
        return cls(
            w0=w_scale * r.normal(size=(n, 3)),
            w1=w_amp * r.normal(size=(n, 3)),
            fw=r.uniform(f_lo, f_hi, size=(n, 3)),
            pw=r.uniform(0.0, 2 * np.pi, size=(n, 3)),
            a0=a_scale * r.normal(size=(n, 3)),
            a1=a_amp * r.normal(size=(n, 3)),
            fa=r.uniform(f_lo, f_hi, size=(n, 3)),
            pa=r.uniform(0.0, 2 * np.pi, size=(n, 3)),
        )

    @property
    def n(self) -> int:
        return self.w0.shape[0]

    def omega(self, t):
        t = np.asarray(t, dtype=float)[..., None, None]
        return self.w0 + self.w1 * np.sin(self.fw * t + self.pw)

    def acc(self, t):
        t = np.asarray(t, dtype=float)[..., None, None]
        return self.a0 + self.a1 * np.cos(self.fa * t + self.pa)

    def shifted(self, t0: float) -> "Stream":
        """The same signal started at t0: s.shifted(t0).omega(t) == s.omega(t0 + t)."""
        return Stream(self.w0, self.w1, self.fw, self.pw + self.fw * t0,
                      self.a0, self.a1, self.fa, self.pa + self.fa * t0)

    def negated_rate(self) -> "Stream":
        """omega -> -omega (used to build the RIGHT-invariant rotation solution)."""
        return Stream(-self.w0, -self.w1, self.fw, self.pw,
                      self.a0, self.a1, self.fa, self.pa)

    def sampled(self, dt: float, n_steps: int):
        """Zero-order-hold samples at t_k = k dt, shape (n_steps, batch, 3)."""
        t = dt * np.arange(int(n_steps))
        return self.omega(t), self.acc(t)


def solve_nav(stream: Stream, R0, v0, p0, Dt: float, g):
    """DOP853 solution of the navigation ODE at t = Dt, batched over the stream.

    ``g`` is one gravity vector (3,) or one per instance (m, 3).  ``Dt`` may be
    negative: solve_ivp then integrates backward from t = 0.
    """
    m = stream.n
    g = np.broadcast_to(np.asarray(g, dtype=float), (m, 3))
    y0 = np.concatenate([np.asarray(R0, float).reshape(m, 9),
                         np.asarray(v0, float).reshape(m, 3),
                         np.asarray(p0, float).reshape(m, 3)], axis=1).ravel()

    def rhs(t, y):
        Y = y.reshape(m, 15)
        R = Y[:, 0:9].reshape(m, 3, 3)
        v = Y[:, 9:12]
        Rd = R @ _skew(stream.omega(t))
        vd = g + np.einsum("kij,kj->ki", R, stream.acc(t))
        return np.concatenate([Rd.reshape(m, 9), vd, v], axis=1).ravel()

    sol = solve_ivp(rhs, (0.0, float(Dt)), y0, method="DOP853", rtol=RTOL, atol=ATOL)
    assert sol.success, sol.message
    Y = sol.y[:, -1].reshape(m, 15)
    return Y[:, 0:9].reshape(m, 3, 3), Y[:, 9:12], Y[:, 12:15]


def solve_increment(stream: Stream, Dt: float):
    """DOP853 solution of Ddot = D xi(t)^ from D(0) = I5: (dR, dv, dp) at Dt."""
    m = stream.n
    return solve_nav(stream, np.broadcast_to(np.eye(3), (m, 3, 3)),
                     np.zeros((m, 3)), np.zeros((m, 3)), Dt, np.zeros(3))


def solve_increment_constant(omega, acc, Dt: float):
    """DOP853 solution of Ddot = D xi^ for CONSTANT (omega, a), batched (m, 3)."""
    omega = np.atleast_2d(np.asarray(omega, float))
    acc = np.atleast_2d(np.asarray(acc, float))
    m = omega.shape[0]
    z = np.zeros((m, 3))
    st = Stream(w0=omega, w1=z, fw=z, pw=z, a0=acc, a1=z, fa=z, pa=z)
    return solve_increment(st, Dt)
