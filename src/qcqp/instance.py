"""The M3 test instance: 3 keyframes, 2 Galilean IMU edges, GNSS-anchored.

Factor graph (claims M3-C01 domain):

* KF0: chordal rotation prior, velocity prior, position prior, all sigma 1e-3;
* KF1, KF2: GNSS position, sigma 0.05 (noise-free measurement = true position);
* edge (i, i+1), i = 0, 1: Galilean IMU factor at nominal bias ``bbar`` using bias ``b_i``;
* bias random walk b_k -> b_{k+1}, k = 0, 1, precision diag(1/(sigma_bg^2 Dt_b) I3,
  1/(sigma_ba^2 Dt_b) I3) (Rai et al. Sec. VI-A Lambda_rw with Dt_b = 0.5 s); no bias prior.

All numerical inputs from GTSAM come through :mod:`.gtsam_boundary`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from . import fixture as F
from . import gtsam_boundary as GB


@dataclass(frozen=True)
class EdgeData:
    i: int
    j: int
    Dt: float
    dR: np.ndarray            #: dRbar, rotation of Ubar
    dv: np.ndarray            #: dvbar
    dp: np.ndarray            #: dpbar
    J: np.ndarray             #: 10x6, rows (omega, nu, rho, alpha), cols [b_g; b_a]
    cov: dict                 #: preintMeasCov blocks by name
    tau_R: float
    tau_v: float
    tau_p: float
    W_R: np.ndarray           #: Sigma_rot^-1
    G2: np.ndarray            #: (trace-reduced 2x2 (vel, pos) covariance)^-1
    W_vp: np.ndarray          #: (6x6 (vel, pos) covariance block)^-1

    @property
    def J_R(self):
        return self.J[0:3, 0:3]

    @property
    def J_nu(self):
        return self.J[3:6, :]

    @property
    def J_rho(self):
        return self.J[6:9, :]

    @property
    def D(self):
        """``tr(W_R)/2 I - W_R`` (claim M3-C33)."""
        return 0.5 * np.trace(self.W_R) * np.eye(3) - self.W_R


@dataclass(frozen=True)
class Instance:
    bbar: np.ndarray
    g: np.ndarray
    edges: tuple
    truth: tuple                     #: ((R, v, p), ...) per keyframe
    b_true: np.ndarray
    gnss: dict                       #: keyframe -> measured position
    prior: tuple                     #: (R0bar, v0bar, p0bar)
    sigma_prior: float
    sigma_gnss: float
    rw_precision: np.ndarray         #: 6x6, project order
    pims: tuple = field(repr=False, default=())

    @property
    def n_kf(self):
        return len(self.truth)


def _edge(pim, e, g, bbar) -> EdgeData:
    U = GB.increment(pim, g, bbar)
    cov = GB.cov_blocks(pim)
    tau = {k: 3.0 / np.trace(cov[(k, k)]) for k in ("rot", "vel", "pos")}
    S2 = np.array([[np.trace(cov[("vel", "vel")]), np.trace(cov[("vel", "pos")])],
                   [np.trace(cov[("pos", "vel")]), np.trace(cov[("pos", "pos")])]]) / 3.0
    Svp = np.block([[cov[("vel", "vel")], cov[("vel", "pos")]],
                    [cov[("pos", "vel")], cov[("pos", "pos")]]])
    return EdgeData(
        i=e, j=e + 1, Dt=float(pim.deltaTij()),
        dR=U.rotation().matrix(), dv=np.asarray(U.velocity()), dp=np.asarray(U.position()),
        J=GB.bias_jacobian(pim, g, bbar), cov=cov,
        tau_R=tau["rot"], tau_v=tau["vel"], tau_p=tau["pos"],
        W_R=np.linalg.inv(cov[("rot", "rot")]), G2=np.linalg.inv(S2), W_vp=np.linalg.inv(Svp))


def build_instance(bbar) -> Instance:
    bbar = np.asarray(bbar, dtype=float).copy()
    g = F.G_W.copy()
    params = GB.make_params(g, F.SIGMA_G, F.SIGMA_A, F.SIGMA_INT)
    pims, edges = [], []
    for e in range(F.N_EDGES):
        gyro, acc = F.streams(e)
        pim = GB.integrate(params, bbar, gyro, acc, F.DT)
        pims.append(pim)
        edges.append(_edge(pim, e, g, bbar))
    truth = tuple(F.true_states())
    dtb = F.T_EDGE
    rw = np.diag(np.r_[np.full(3, 1.0 / (F.SIGMA_BG_RW**2 * dtb)),
                       np.full(3, 1.0 / (F.SIGMA_BA_RW**2 * dtb))])
    return Instance(bbar=bbar, g=g, edges=tuple(edges), truth=truth,
                    b_true=F.BIAS_TRUE.copy(),
                    gnss={k: truth[k][2].copy() for k in (1, 2)},
                    prior=tuple(x.copy() for x in truth[0]),
                    sigma_prior=F.SIGMA_PRIOR, sigma_gnss=F.SIGMA_GNSS,
                    rw_precision=rw, pims=tuple(pims))


@functools.lru_cache(maxsize=None)
def instance(which: str = "true") -> Instance:
    """Cached instance at ``bbar = b_true`` (``"true"``) or ``bbar = 0`` (``"zero"``)."""
    bbar = {"true": F.BIAS_TRUE, "zero": np.zeros(6)}[which]
    return build_instance(bbar)
