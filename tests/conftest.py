"""Shared fixtures and sampling helpers for the M1 claim tests.

Every sample count in docs/claims/M1.yaml is an independent draw from
``numpy.random.default_rng`` with the fixed seed recorded here.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src import gal3
from src import imu_dynamics as imu

from . import ode_truth

#: the fixed, recorded seed for every randomized claim test
SEED = 20241012

G = gal3.G_DEFAULT


def rng(offset: int = 0) -> np.random.Generator:
    """A fresh generator; ``offset`` keeps independent tests from sharing a stream."""
    return np.random.default_rng(SEED + offset)


def _rotvec_to_matrix(phi):
    """Exp_SO3 by scipy -- an authority independent of the module under test."""
    return Rotation.from_rotvec(np.asarray(phi, dtype=float)).as_matrix()


def rand_rot(r, n=None, scale=1.0):
    """Exp_SO3 of scale * N(0, I3), built with scipy (never with ``gal3.so3_exp``)."""
    if n is None:
        return _rotvec_to_matrix(scale * r.normal(size=3))
    return np.array([_rotvec_to_matrix(scale * r.normal(size=3)) for _ in range(n)])


def assert_term_exercised(mags, margin=1e-3, min_frac=0.99):
    """Coverage guard: the load-bearing term had magnitude >= margin on >= min_frac
    of the draws (and a median well above it).  ``min(|t|) > 0`` on a continuous
    draw cannot fail; this can.
    """
    mags = np.asarray(mags, dtype=float)
    frac = float(np.mean(mags >= margin))
    assert frac >= min_frac, (frac, float(mags.min()))
    assert float(np.median(mags)) >= 100.0 * margin, float(np.median(mags))
    return frac


def rand_axis(r, n=None):
    a = r.normal(size=3 if n is None else (n, 3))
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def rand_rot_angle(r, lo, hi, n=None):
    """Random rotation with ||phi|| uniform in (lo, hi)."""
    if n is None:
        return _rotvec_to_matrix(r.uniform(lo, hi) * rand_axis(r))
    th = r.uniform(lo, hi, size=n)
    ax = rand_axis(r, n)
    return np.array([_rotvec_to_matrix(t * a) for t, a in zip(th, ax)])


def rand_gal(r, v_scale=1.0, p_scale=3.0, t_scale=1.0, rot_scale=1.0):
    """X(R, v, p, t) with R = Exp(N(0,I)), v ~ N(0,I), p ~ 3 N(0,I), t ~ N(0,1)."""
    return gal3.gal(
        rand_rot(r, scale=rot_scale),
        v_scale * r.normal(size=3),
        p_scale * r.normal(size=3),
        t_scale * r.normal(),
    )


def rand_xi(r, scale=1.0):
    return scale * r.normal(size=gal3.DIM)


def rand_xi_bounded_phi(r, lo=0.05, hi=2.9, scale=1.0):
    """xi with ||phi|| uniform in (lo, hi) and the rest scale * N(0, 1)."""
    xi = scale * r.normal(size=gal3.DIM)
    xi[0:3] = r.uniform(lo, hi) * rand_axis(r)
    return xi


# --------------------------------------------------------------------------
# the shared endpoint dataset (M1-C31, C34, C37, C43)
# --------------------------------------------------------------------------
#: first-order bound on the discrete recursion vs the ODE truth at dt = 1e-4
#: (measured 7.7e-4 on the navigation state, 8.6e-4 on the increment).
DISCRETE_ACCURACY_TOL = 2e-3


def _build_endpoint_dataset(r, n_inst, n_steps, dt, g, label):
    """One endpoint dataset; ``g`` is (3,) or one gravity vector per instance (n_inst, 3).

    integrate_truth takes a single g, so the discrete recursion is run once per
    distinct gravity vector and reassembled in instance order.
    """
    Dt = n_steps * dt
    R_i = rand_rot(r, n_inst)
    v_i = 2.0 * r.normal(size=(n_inst, 3))
    p_i = 5.0 * r.normal(size=(n_inst, 3))
    stream = ode_truth.Stream.random(r, n_inst)
    g_inst = np.broadcast_to(np.asarray(g, dtype=float), (n_inst, 3)).copy()

    info_nav, info_inc = {}, {}
    R_j, v_j, p_j = ode_truth.solve_nav(stream, R_i, v_i, p_i, Dt, g_inst, info=info_nav)
    dR, dv, dp = ode_truth.solve_increment(stream, Dt, info=info_inc)
    groups = [np.flatnonzero((g_inst == gk).all(axis=1)) for gk in np.unique(g_inst, axis=0)]

    def discrete(n, h):
        om_s, acc_s = stream.sampled(h, n)
        out = {k: np.empty((n_inst, 3, 3)) for k in ("R_j", "dR")}
        out.update({k: np.empty((n_inst, 3)) for k in ("v_j", "p_j", "dv", "dp")})
        for idx in groups:
            (Rd, vd, pd), (dRd, dvd, dpd, Dt_out) = imu.integrate_truth(
                R_i[idx], v_i[idx], p_i[idx], om_s[:, idx], acc_s[:, idx], h, n,
                g_inst[idx[0]], per_step=True)
            for k, val in zip(("R_j", "v_j", "p_j", "dR", "dv", "dp"),
                              (Rd, vd, pd, dRd, dvd, dpd)):
                out[k][idx] = val
        out["Dt"] = Dt_out
        return out

    # INPUT GUARD data: the right-invariant rotation solution of the same stream
    # (R^T of the omega -> -omega solve); far from the left-invariant one only if
    # the stream is genuinely non-commuting.  See assert_stream_non_commuting.
    dR_neg, _, _ = ode_truth.solve_increment(stream.negated_rate(), Dt)
    order_gap = float(np.abs(dR - np.swapaxes(dR_neg, -1, -2)).max())

    disc = discrete(n_steps, dt)
    disc_half = discrete(max(n_steps // 2, 1), Dt / max(n_steps // 2, 1))
    return dict(
        label=label, n=n_inst, Dt=Dt, g=g_inst, dt=dt,
        R_i=R_i, v_i=v_i, p_i=p_i,
        R_j=R_j, v_j=v_j, p_j=p_j,
        dR=dR, dv=dv, dp=dp,
        stream=stream, disc=disc, disc_half=disc_half, order_gap=order_gap,
        ode_t_nav=info_nav["t"], ode_t_inc=info_inc["t"],
    )


def rand_gravity(r, n=None):
    """Gravity of random DIRECTION and MAGNITUDE: |g| uniform in (2, 20) m/s^2."""
    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1)) * rand_axis(r, n)


def assert_gravity_directions_diverse(gs, min_angle=0.5, min_perm_gap=1.0, min_component=1.0,
                                      min_span=0.1):
    """INPUT GUARD on a block of random-g draws (not evidence about src).

    The random-g checks exist to catch defects that are invisible for g parallel to z,
    e.g. a src function that swaps g's x and y components (pass-6 Q50..Q53).  A
    ``rand_gravity`` that silently returned |g| (0, 0, -1) passed every claim test and
    reopened those mutants (pass-7 F1).  Require, over the block ``gs`` (shape (m, 3)):

    * some draw at >= ``min_angle`` rad from the z axis (``off_z``);
    * some draw with |g - g[[1, 0, 2]]|_inf >= ``min_perm_gap`` (an x/y swap is visible);
    * for m > 1, two draws >= ``min_angle`` rad apart as lines (``spread``): the block
      is not all parallel to ONE direction.
    * EVERY component is exercised: max over the draws of |g_k| >= ``min_component``
      for k = x, y, z (``comp``).  This applies to a single draw too.
    * for m >= 3, the unit directions SPAN R^3: smallest singular value of the stacked
      (m, 3) unit vectors / sqrt(m) >= ``min_span`` (``span``).

    The last two close pass-8 M1-F21: a PLANAR distribution (g in the x-z plane, >= 0.6
    rad off z) passed off_z, perm_gap and spread, and under it a g_y sign error in
    gravity_lift, gravity_element, integrate_truth or forster_endpoint survived the
    suite.  ``min_span`` is 0.1, not 0.2: one recorded 4-g dataset (Dt = +0.4) has
    span 0.122 (Dt = -0.3: 0.305; the 400..2000-draw blocks 0.549..0.570); a planar block
    has span <= 2e-16.  comp is >= 4.2 on every recorded block (the one-draw C45 g: 4.20)
    and exactly 0 for g_y = 0.
    """
    gs = np.atleast_2d(np.asarray(gs, dtype=float))
    u = gs / np.linalg.norm(gs, axis=1, keepdims=True)
    off_z = float(np.arccos(np.clip(np.abs(u[:, 2]), 0.0, 1.0)).max())
    perm_gap = float(np.abs(gs - gs[:, [1, 0, 2]]).max())
    assert off_z >= min_angle, off_z
    assert perm_gap >= min_perm_gap, perm_gap
    spread = None
    if len(gs) > 1:
        spread = float(np.arccos(np.clip(np.abs(u @ u.T), 0.0, 1.0)).max())
        assert spread >= min_angle, spread
    comp = float(np.abs(gs).max(axis=0).min())
    assert comp >= min_component, comp
    span = None
    if len(gs) >= 3:
        span = float(np.linalg.svd(u, compute_uv=False)[-1] / np.sqrt(len(gs)))
        assert span >= min_span, span
    return off_z, perm_gap, spread, comp, span


def assert_rand_g_datasets(datasets):
    """INPUT GUARD on ``endpoint_datasets_rand_g`` (not evidence about src).

    * Each dataset's sign of Dt is the one its label states, and exactly one dataset
      has Dt < 0: they are the endpoint claims' only Dt < 0 coverage, and making the
      second one forward (dt = +1e-4) passed every claim test (pass-7 F2).
    * For each dataset the integration ran in the direction of Dt -- for the Dt < 0
      one, BACKWARD: both DOP853 solves (navigation and increment) have a strictly
      monotone time grid from 0 to Dt, and the recursion's step dt and its
      accumulated Dt have the sign of Dt.
    * Each dataset has 4 distinct g, of diverse direction
      (``assert_gravity_directions_diverse``; pass-7 F1).
    """
    for d in datasets:
        assert (d["Dt"] < 0) == ("Dt=-" in d["label"]), (d["label"], d["Dt"])
    n_neg = sum(d["Dt"] < 0 for d in datasets)
    assert n_neg == 1, n_neg
    for d in datasets:
        sgn = np.sign(d["Dt"])
        assert np.sign(d["dt"]) == sgn and np.sign(d["disc"]["Dt"]) == sgn, d["label"]
        for t in (d["ode_t_nav"], d["ode_t_inc"]):
            assert t[0] == 0.0 and t[-1] == d["Dt"], (d["label"], t[0], t[-1])
            assert bool(np.all(np.sign(np.diff(t)) == sgn)), d["label"]
        g_distinct = np.unique(d["g"], axis=0)
        assert len(g_distinct) == 4, len(g_distinct)
        assert_gravity_directions_diverse(g_distinct)


@pytest.fixture(scope="session")
def endpoint_dataset():
    """200 instances with TIME-VARYING, non-commuting (omega(t), a(t)), Dt = 0.5.

    Two independent sources of (X_j, D_ij):

    * ``R_j, v_j, p_j, dR, dv, dp`` -- the CONTINUOUS truth: the navigation ODE
      and the increment ODE solved SEPARATELY by scipy DOP853 at rtol = atol =
      1e-13 (``tests/ode_truth.py``; nothing from ``src``).  Accuracy claims are
      checked against these.
    * ``disc`` -- the project's first-order recursion ``integrate_truth`` on the
      same stream (zero-order hold), 5000 steps at dt = 1e-4.  The endpoint
      relation holds for it as an exact ALGEBRAIC IDENTITY of the recursion for any
      dt, so it supports identity checks only.  It is also checked against the ODE
      truth (first order, see ``assert_discrete_converges``) so that a degraded
      fixture -- e.g. 2 steps at dt = 0.25 -- is caught.

    ``g`` is stored per instance, shape (n, 3); here every row is (0, 0, -9.81).
    """
    r = rng(1031)
    n_inst, n_steps, dt = 200, 5000, 1e-4
    return _build_endpoint_dataset(r, n_inst, n_steps, dt, G, "g=G_DEFAULT, Dt=+0.5")


@pytest.fixture(scope="session")
def endpoint_datasets_rand_g():
    """Two more endpoint datasets for the claims that hold for ARBITRARY g and Dt of
    either sign (M1-C31, C34, C37, C43): 40 instances each, 4 random gravity vectors
    (random direction, |g| in (2, 20)) of 10 instances each, built exactly like
    ``endpoint_dataset``:

    * Dt = +0.4 (4000 steps of dt = 1e-4);
    * Dt = -0.3 (3000 steps of dt = -1e-4): the ODEs are solved BACKWARD from t = 0
      and the recursion steps backward.
    """
    out = []
    for off, n_steps, dt, label in ((3101, 4000, 1e-4, "random g, Dt=+0.4"),
                                    (3102, 3000, -1e-4, "random g, Dt=-0.3")):
        r = rng(off)
        g = np.repeat(rand_gravity(r, 4), 10, axis=0)
        out.append(_build_endpoint_dataset(r, 40, n_steps, dt, g, label))
    return out


def _discrete_error(d, disc):
    return max(float(np.abs(disc[k] - d[k]).max())
               for k in ("R_j", "v_j", "p_j", "dR", "dv", "dp"))


def assert_discrete_converges(d):
    """The discrete recursion is a first-order discretisation of the ODE truth.

    err(dt) <= DISCRETE_ACCURACY_TOL, err(2 dt) / err(dt) in [1.8, 2.2], and the
    recursion's Dt equals the ODE's.  Fails for a degraded fixture.
    """
    assert abs(d["disc"]["Dt"] - d["Dt"]) <= 1e-12
    e1 = _discrete_error(d, d["disc"])
    e2 = _discrete_error(d, d["disc_half"])
    assert e1 <= DISCRETE_ACCURACY_TOL, e1
    assert 1.8 <= e2 / e1 <= 2.2, (e1, e2)
    return e1


def assert_stream_non_commuting(d):
    """INPUT GUARD on the endpoint dataset (not evidence about src).

    The endpoint tests catch accumulation-order defects (R @ E vs E @ R) only if
    the stream's rotations do not commute.  Require the left- and right-invariant
    rotation solutions of the stream to differ by >= 0.1, i.e. 50 x the discrete
    accuracy tolerance DISCRETE_ACCURACY_TOL; a constant-omega stream gives
    round-off only.
    """
    assert d["order_gap"] >= 0.1, d["order_gap"]
    return d["order_gap"]


@pytest.fixture(scope="session")
def algebraic_instances():
    """2000 purely algebraic (X_i, D_ij) pairs on the t_i = 0 slice."""
    r = rng(2027)
    n = 2000
    out = []
    for _ in range(n):
        Dt = r.uniform(0.05, 1.0)
        X_i = gal3.gal(rand_rot(r), 2.0 * r.normal(size=3), 5.0 * r.normal(size=3), 0.0)
        D = gal3.gal(rand_rot(r, scale=0.4), r.normal(size=3), r.normal(size=3), Dt)
        out.append((X_i, D, Dt))
    return out
