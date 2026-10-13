"""Baseline 1 -- conventional on-manifold IMU preintegration (Forster et al., T-RO 2017).

Scope of this module, stated precisely because the three M2 baselines deliberately do
*not* share a tolerance:

* the increment recursion here is the **first-order Euler sum** of Forster Eq. (35)-(37)
  (identical discretization to Rai Eq. (9a)-(9c)).  It is ``O(dt)`` accurate in ``dv``
  and ``dp`` -- claim M2-C04 -- and is therefore *not* comparable against the exact
  closed form of :mod:`src.trajectory` at better than the Tier-2 tolerance;
* ``dR`` happens to be exact on *this* fixture only because the fixture's body rate is
  constant per edge, so the product of per-step exponentials telescopes to ``Exp(w T)``.
  That is a property of the fixture, not of the recursion.  Do not generalize it.

Bias handling follows Forster Eq. (44) and is deliberately **mixed-chart**: the rotation
correction is a right group composition, the velocity and position corrections are
additive, and all five Jacobians are frozen at ``bbar`` (claim M2-C05).

GTSAM 4.3.0 exposes **no** ``delRdelBiasOmega`` / ``delVdelBias*`` / ``delPdelBias*``
accessor on either PIM class in Python, so the reference Jacobians cannot be read off a
GTSAM object; :func:`bias_jacobians_fd` obtains them by finite differences on
re-integration instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trajectory import (G_W, NavTriple, SMALL_ANGLE, hat, so3_exp, so3_log)


def so3_right_jacobian(phi) -> np.ndarray:
    """``Jr(phi)``, the right Jacobian of ``Exp`` on SO(3).  Needed by Eq. (63) only."""
    phi = np.asarray(phi, dtype=float)
    th = float(np.linalg.norm(phi))
    K = hat(phi)
    if th < SMALL_ANGLE:
        return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)
    return (np.eye(3)
            - ((1.0 - np.cos(th)) / th**2) * K
            + ((th - np.sin(th)) / th**3) * (K @ K))


@dataclass
class ForsterPreintegration:
    """Forster-style preintegrated increment, accumulated at a frozen ``bias_hat``.

    ``bias_hat`` is in the **project** convention ``[b_g; b_a]``.  Crossing the GTSAM
    boundary swaps the two 3-blocks (claim M2-C17); that swap lives in
    :mod:`src.preintegration_galilean`, never here.
    """

    bias_hat: np.ndarray
    #: gyro noise DENSITY (rad/s/sqrt(Hz)); a scalar (isotropic) or a per-axis 3-vector
    sigma_g: float | np.ndarray = 0.0
    #: accel noise DENSITY (m/s^2/sqrt(Hz)); a scalar (isotropic) or a per-axis 3-vector
    sigma_a: float | np.ndarray = 0.0

    dR: np.ndarray = field(default_factory=lambda: np.eye(3))
    dv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dp: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt_total: float = 0.0
    #: 9x9 propagated covariance in (rotation, velocity, position) block order
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))

    def __post_init__(self) -> None:
        # M2-F18: never alias a caller's array.  ``np.asarray`` of a float array is the
        # caller's object, so an in-place update of the caller's bias estimate would move
        # ``bias_hat`` while the increment accumulated at the old value stayed frozen.
        self.bias_hat = np.array(self.bias_hat, dtype=float, copy=True)
        self.sigma_g = np.array(self.sigma_g, dtype=float, copy=True)
        self.sigma_a = np.array(self.sigma_a, dtype=float, copy=True)
        self.dR = np.array(self.dR, dtype=float, copy=True)
        self.dv = np.array(self.dv, dtype=float, copy=True)
        self.dp = np.array(self.dp, dtype=float, copy=True)
        self.dt_total = float(self.dt_total)
        self.cov = np.array(self.cov, dtype=float, copy=True)

    @property
    def b_g(self) -> np.ndarray:
        """A COPY of ``bias_hat[0:3]`` (M2-F22: never a view a caller could write through)."""
        return np.array(self.bias_hat[0:3], dtype=float, copy=True)

    @property
    def b_a(self) -> np.ndarray:
        """A COPY of ``bias_hat[3:6]`` (M2-F22)."""
        return np.array(self.bias_hat[3:6], dtype=float, copy=True)

    # -- Eq. (35)-(37) -----------------------------------------------------------------
    def integrate_measurement(self, omega_meas, acc_meas, dt: float) -> None:
        """One step of ``dR_ij = prod Exp(w dt)``, ``dv_ij = sum dR_ik a dt``,
        ``dp_ij = sum [dv_ik dt + 0.5 dR_ik a dt^2]``.

        The order matters: ``dp`` and ``dv`` are updated with the increment ``dR_ik``
        *before* this step's rotation is folded in.
        """
        w = np.asarray(omega_meas, dtype=float) - self.b_g
        a = np.asarray(acc_meas, dtype=float) - self.b_a

        # M2-F23: the gate is "is there anything to propagate", not "is there noise".  A
        # nonzero initial ``cov`` must be carried through ``A cov A^T`` even at zero
        # densities.  Skipping is exact only when both are zero (then Eq. (63) gives 0).
        if self.has_noise or np.any(self.cov != 0.0):
            self._propagate_covariance(w, a, dt)

        dR_ik = self.dR
        self.dp = self.dp + self.dv * dt + 0.5 * (dR_ik @ a) * dt**2
        self.dv = self.dv + (dR_ik @ a) * dt
        self.dR = dR_ik @ self.retract(w * dt)
        self.dt_total += dt

    @property
    def has_noise(self) -> bool:
        return bool(np.any(np.asarray(self.sigma_g) > 0.0)
                    or np.any(np.asarray(self.sigma_a) > 0.0))

    def noise_covariances(self, dt: float):
        """Discrete-time ``(Qg, Qa)`` from the continuous densities: ``diag(sigma^2) / dt``.

        Per-axis densities are allowed so that a covariance test can use ANISOTROPIC noise:
        with isotropic noise ``dR Q dR^T = Q``, which hides every orientation error in the
        noise-input and rotation-transition blocks (verification pass 2, M2-F07).
        """
        sg = np.broadcast_to(np.asarray(self.sigma_g, dtype=float), (3,))
        sa = np.broadcast_to(np.asarray(self.sigma_a, dtype=float), (3,))
        return np.diag(sg**2) / dt, np.diag(sa**2) / dt

    @staticmethod
    def retract(phi) -> np.ndarray:
        """The rotation retraction.  Baseline 2 overrides this hook and, with it,
        :meth:`retract_right_jacobian`."""
        return so3_exp(phi)

    @staticmethod
    def retract_right_jacobian(phi) -> np.ndarray:
        """The right Jacobian of :meth:`retract`: ``retract(phi + d) = retract(phi)
        Exp(J d) + O(|d|^2)``.  It is the noise-input Jacobian of Eq. (63), so it must
        belong to the SAME map as :meth:`retract` (M2-F26: Baseline 2 used to inherit
        ``Exp``'s here while integrating with ``Cay``)."""
        return so3_right_jacobian(phi)

    # -- Eq. (63) ----------------------------------------------------------------------
    def _propagate_covariance(self, w, a, dt: float) -> None:
        """Exact first-order covariance recursion of the Euler scheme above.

        The chart is Forster's: ``dR_noisy = dR Exp(dphi)``, ``dv_noisy = dv + ddv``,
        ``dp_noisy = dp + ddp`` (additive in v and p).  GTSAM's Galilean ``preintMeasCov``
        lives in the right Gal(3) chart ``U Exp(xi)``, whose velocity/position components
        are ``dR^T ddv`` and ``dR^T ddp``; compare via ``M = diag(I, dR^T, dR^T)``, not
        directly (test_c06b).
        """
        dR_step = self.retract(w * dt)
        Jr = self.retract_right_jacobian(w * dt)
        dR_ik = self.dR
        A = np.zeros((9, 9))
        A[0:3, 0:3] = dR_step.T
        A[3:6, 0:3] = -dR_ik @ hat(a) * dt
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 0:3] = -0.5 * dR_ik @ hat(a) * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 6:9] = np.eye(3)

        Bg = np.zeros((9, 3))
        Bg[0:3, :] = Jr * dt
        Ba = np.zeros((9, 3))
        Ba[3:6, :] = dR_ik * dt
        Ba[6:9, :] = 0.5 * dR_ik * dt**2

        # discrete-time noise covariances from the continuous densities
        Qg, Qa = self.noise_covariances(dt)
        self.cov = A @ self.cov @ A.T + Bg @ Qg @ Bg.T + Ba @ Qa @ Ba.T

    # -- Eq. (30)-(33) -----------------------------------------------------------------
    def predict(self, state_i: NavTriple, g_W=G_W) -> NavTriple:
        """The component endpoint relation (claim M2-C03)."""
        g_W = np.asarray(g_W, dtype=float)
        R_i, v_i, p_i = state_i.as_tuple()
        Dt = self.dt_total
        return NavTriple(R=R_i @ self.dR,
                         v=v_i + g_W * Dt + R_i @ self.dv,
                         p=p_i + v_i * Dt + 0.5 * g_W * Dt**2 + R_i @ self.dp)

    def increment(self):
        return self.dR.copy(), self.dv.copy(), self.dp.copy()


def integrate_edge(edge, bias_hat, sigma_g=0.0, sigma_a=0.0,
                   cls=ForsterPreintegration, **kwargs):
    """Run ``cls`` over one fixture edge's noise-free stream."""
    pim = cls(bias_hat=np.asarray(bias_hat, dtype=float),
              sigma_g=sigma_g, sigma_a=sigma_a, **kwargs)
    for omega, acc in zip(edge.gyro_stream(), edge.accel_stream()):
        pim.integrate_measurement(omega, acc, edge.dt)
    return pim


def integrate_edge_states(edges_, bias_hat, state0, g_W=G_W,
                          cls=ForsterPreintegration, **kwargs):
    """Chain ``cls`` across every edge, returning ``[KF0, KF1, ...]``."""
    states = [state0]
    for edge in edges_:
        pim = integrate_edge(edge, bias_hat, cls=cls, **kwargs)
        states.append(pim.predict(states[-1], g_W))
    return states


# --------------------------------------------------------------------------------------
# Bias correction -- Forster Eq. (44)
# --------------------------------------------------------------------------------------

#: finite-difference step for the reference bias Jacobians.  Central differences, so the
#: truncation is O(eps^2) ~ 1e-12 and does not pollute the quadratic-growth test of C05.
FD_EPS = 1e-6


def bias_jacobians_fd(edge, bias_hat, eps: float = FD_EPS,
                      cls=ForsterPreintegration, **kwargs):
    """``(J_R, J_vg, J_va, J_pg, J_pa)`` by central finite differences on re-integration.

    ``J_R`` is the **right**-chart rotation Jacobian: ``dR(bbar + db_g) ~
    dR(bbar) Exp(J_R db_g)``.  The other four are ordinary additive Jacobians.

    This route is not a convenience.  GTSAM 4.3.0's Python bindings expose no bias
    Jacobian accessor on ``PreintegratedImuMeasurements`` or
    ``PreintegratedImuMeasurementsG`` -- the two classes have identical member sets and
    neither carries ``delRdelBiasOmega`` / ``delVdelBias*`` / ``delPdelBias*``.
    """
    bias_hat = np.asarray(bias_hat, dtype=float)
    base = integrate_edge(edge, bias_hat, cls=cls, **kwargs)
    dR0 = base.dR

    J_R = np.zeros((3, 3))
    J_v = np.zeros((3, 6))
    J_p = np.zeros((3, 6))
    for m in range(6):
        step = np.zeros(6)
        step[m] = eps
        plus = integrate_edge(edge, bias_hat + step, cls=cls, **kwargs)
        minus = integrate_edge(edge, bias_hat - step, cls=cls, **kwargs)
        J_v[:, m] = (plus.dv - minus.dv) / (2.0 * eps)
        J_p[:, m] = (plus.dp - minus.dp) / (2.0 * eps)
        if m < 3:
            J_R[:, m] = (so3_log(dR0.T @ plus.dR) - so3_log(dR0.T @ minus.dR)) / (2.0 * eps)
    return J_R, J_v[:, 0:3], J_v[:, 3:6], J_p[:, 0:3], J_p[:, 3:6]


def bias_corrected_increment(base_increment, jacobians, db):
    """Forster Eq. (44), the mixed-chart first-order correction.

    ``dR(b) = dR(bbar) Exp(J_R db_g)``   -- **right group composition**
    ``dv(b) = dv(bbar) + J_vg db_g + J_va db_a``   -- additive
    ``dp(b) = dp(bbar) + J_pg db_g + J_pa db_a``   -- additive

    The asymmetry is Forster's, not an oversight: see claim M2-C05.
    """
    dR0, dv0, dp0 = base_increment
    J_R, J_vg, J_va, J_pg, J_pa = jacobians
    db = np.asarray(db, dtype=float)
    db_g, db_a = db[0:3], db[3:6]
    return (dR0 @ so3_exp(J_R @ db_g),
            dv0 + J_vg @ db_g + J_va @ db_a,
            dp0 + J_pg @ db_g + J_pa @ db_a)


def increment_error(a, b) -> float:
    """Chart-consistent distance between two ``(dR, dv, dp)`` triples."""
    dRa, dva, dpa = a
    dRb, dvb, dpb = b
    return max(float(np.abs(so3_log(dRb.T @ dRa)).max()),
               float(np.abs(dva - dvb).max()),
               float(np.abs(dpa - dpb).max()))
