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

import dataclasses
import functools
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from . import fixture as F
from . import gtsam_boundary as GB


class FrozenDict(Mapping):
    """A read-only mapping (item assignment raises TypeError) that, unlike
    ``types.MappingProxyType``, survives ``pickle`` / ``copy.deepcopy`` (verifier pass 6, M3-F09 a).
    ``__reduce__`` is load-bearing: the default slot-state restore goes through ``__setattr__``,
    which is blocked."""
    __slots__ = ("_d",)

    def __init__(self, *args, **kw):
        object.__setattr__(self, "_d", dict(*args, **kw))

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f"FrozenDict({self._d!r})"

    def __setattr__(self, name, value):
        raise AttributeError("FrozenDict is read-only")

    def __reduce__(self):
        return (FrozenDict, (self._d,))


def _rebuild(cls, kw):
    """Unpickle / deepcopy target of EdgeData and Instance: rebuild, then re-freeze (unpickled
    numpy arrays come back writable)."""
    if cls is Instance:
        return _frozen(cls(**kw))
    return cls(**{k: _freeze(v) for k, v in kw.items()})


class _Rebuilt:
    def __reduce__(self):
        return (_rebuild, (type(self), {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}))


@dataclass(frozen=True)
class EdgeData(_Rebuilt):
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
class Instance(_Rebuilt):
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
    return _frozen(Instance(bbar=bbar, g=g, edges=tuple(edges), truth=truth,
                            b_true=F.BIAS_TRUE.copy(),
                            gnss={k: truth[k][2].copy() for k in (1, 2)},
                            prior=tuple(x.copy() for x in truth[0]),
                            sigma_prior=F.SIGMA_PRIOR, sigma_gnss=F.SIGMA_GNSS,
                            rw_precision=rw, pims=tuple(pims)))


def _freeze(x):
    """Read-only view of the numpy content of ``x`` (arrays flagged write=False, dicts wrapped
    in a read-only mapping, tuples rebuilt); other objects are returned unchanged.  A view (an
    array with a ``.base``, e.g. a preintMeasCov block) is replaced by an independent copy first,
    so that no writable base stays reachable (verifier pass 6, M3-F09 b)."""
    if isinstance(x, np.ndarray):
        if x.base is not None:
            x = x.copy()
        x.setflags(write=False)
        return x
    if isinstance(x, Mapping):
        return FrozenDict({k: _freeze(v) for k, v in x.items()})
    if isinstance(x, tuple):
        return tuple(_freeze(v) for v in x)
    return x


def _frozen(inst: Instance) -> Instance:
    """Every numpy array reachable from the instance and its edges is made read-only, so that an
    in-place write by a caller cannot poison the lru-cached :func:`instance` (the GTSAM PIM
    objects in ``pims`` are not covered)."""
    edges = tuple(dataclasses.replace(ed, **{f.name: _freeze(getattr(ed, f.name))
                                            for f in dataclasses.fields(ed)}) for ed in inst.edges)
    kw = {f.name: _freeze(getattr(inst, f.name)) for f in dataclasses.fields(inst)
          if f.name not in ("edges", "pims")}
    return dataclasses.replace(inst, edges=edges, **kw)


@functools.lru_cache(maxsize=None)
def instance(which: str = "true") -> Instance:
    """Cached instance at ``bbar = b_true`` (``"true"``) or ``bbar = 0`` (``"zero"``)."""
    bbar = {"true": F.BIAS_TRUE, "zero": np.zeros(6)}[which]
    return build_instance(bbar)
