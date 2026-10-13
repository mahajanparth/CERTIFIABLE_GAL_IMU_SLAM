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
    """
    r = rng(1031)
    n_inst, n_steps, dt = 200, 5000, 1e-4
    Dt = n_steps * dt
    R_i = rand_rot(r, n_inst)
    v_i = 2.0 * r.normal(size=(n_inst, 3))
    p_i = 5.0 * r.normal(size=(n_inst, 3))
    stream = ode_truth.Stream.random(r, n_inst)

    R_j, v_j, p_j = ode_truth.solve_nav(stream, R_i, v_i, p_i, Dt, G)
    dR, dv, dp = ode_truth.solve_increment(stream, Dt)

    def discrete(n, h):
        om_s, acc_s = stream.sampled(h, n)
        (Rd, vd, pd), (dRd, dvd, dpd, Dt_out) = imu.integrate_truth(
            R_i, v_i, p_i, om_s, acc_s, h, n, G, per_step=True)
        return dict(R_j=Rd, v_j=vd, p_j=pd, dR=dRd, dv=dvd, dp=dpd, Dt=Dt_out)

    # INPUT GUARD data: the right-invariant rotation solution of the same stream
    # (R^T of the omega -> -omega solve); far from the left-invariant one only if
    # the stream is genuinely non-commuting.  See assert_stream_non_commuting.
    dR_neg, _, _ = ode_truth.solve_increment(stream.negated_rate(), Dt)
    order_gap = float(np.abs(dR - np.swapaxes(dR_neg, -1, -2)).max())

    disc = discrete(n_steps, dt)
    disc_half = discrete(max(n_steps // 2, 1), Dt / max(n_steps // 2, 1))
    return dict(
        n=n_inst, Dt=Dt, g=G, dt=dt,
        R_i=R_i, v_i=v_i, p_i=p_i,
        R_j=R_j, v_j=v_j, p_j=p_j,
        dR=dR, dv=dv, dp=dp,
        stream=stream, disc=disc, disc_half=disc_half, order_gap=order_gap,
    )


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
