"""Shared fixtures and sampling helpers for the M1 claim tests.

Every sample count in docs/claims/M1.yaml is an independent draw from
``numpy.random.default_rng`` with the fixed seed recorded here.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import gal3
from src import imu_dynamics as imu

#: the fixed, recorded seed for every randomized claim test
SEED = 20241012

G = gal3.G_DEFAULT


def rng(offset: int = 0) -> np.random.Generator:
    """A fresh generator; ``offset`` keeps independent tests from sharing a stream."""
    return np.random.default_rng(SEED + offset)


def rand_rot(r, n=None, scale=1.0):
    """Exp_SO3 of scale * N(0, I3)."""
    if n is None:
        return gal3.so3_exp(scale * r.normal(size=3))
    return np.array([gal3.so3_exp(scale * r.normal(size=3)) for _ in range(n)])


def rand_axis(r, n=None):
    a = r.normal(size=3 if n is None else (n, 3))
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def rand_rot_angle(r, lo, hi, n=None):
    """Random rotation with ||phi|| uniform in (lo, hi)."""
    if n is None:
        return gal3.so3_exp(r.uniform(lo, hi) * rand_axis(r))
    th = r.uniform(lo, hi, size=n)
    ax = rand_axis(r, n)
    return np.array([gal3.so3_exp(t * a) for t, a in zip(th, ax)])


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
# the shared integrator ground truth (M1-C31, C34, C37, C43)
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def endpoint_dataset():
    """200 instances integrated with 5000 Euler steps at dt = 1e-4, Dt = 0.5.

    The navigation state and the gravity-free increment come from the SAME
    stepping scheme, so the endpoint relation should hold to roundoff rather
    than to the integrator's truncation error.
    """
    r = rng(1031)
    n_inst, n_steps, dt = 200, 5000, 1e-4
    Dt = n_steps * dt
    R_i = rand_rot(r, n_inst)
    v_i = 2.0 * r.normal(size=(n_inst, 3))
    p_i = 5.0 * r.normal(size=(n_inst, 3))
    omega = 0.8 * r.normal(size=(n_inst, 3))
    acc = 2.0 * r.normal(size=(n_inst, 3))
    (R_j, v_j, p_j), (dR, dv, dp, Dt_out) = imu.integrate_truth(
        R_i, v_i, p_i, omega, acc, dt, n_steps, G
    )
    assert abs(Dt_out - Dt) < 1e-12
    return dict(
        n=n_inst, Dt=Dt, g=G,
        R_i=R_i, v_i=v_i, p_i=p_i,
        R_j=R_j, v_j=v_j, p_j=p_j,
        dR=dR, dv=dv, dp=dp,
        omega=omega, acc=acc,
    )


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
