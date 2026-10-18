"""Shared fixtures and sampling helpers for the M1 claim tests.

Every sample count in docs/claims/M1.yaml is an independent draw from
``numpy.random.default_rng`` with the fixed seed recorded here.
"""
from __future__ import annotations

import os

# Pin the BLAS / OpenMP thread pools to ONE thread, BEFORE numpy (and with it OpenBLAS)
# is imported anywhere in the test process.  On a loaded multi-core host the default
# (one thread per core in every process) oversubscribes the CPU: the mutation harness
# runs several pytest subprocesses at once, each doing many tiny matrix products, and a
# run that takes ~40 s pinned was observed to crawl for hours (pass-11, mutant S08).
# Assigned here, not only in a shell, so every run of the suite is protected; the
# harness subprocesses inherit os.environ and so are pinned too.  The suite result is
# unchanged by the pins (impl-M1-012: 235 passed with them, with and without the shell
# exports; the claim tests' operands are 3x3 / 5x5, below any BLAS threading threshold).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402  (must follow the thread pins above)
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from src import gal3
from src import imu_dynamics as imu

from . import ode_truth

#: the fixed, recorded seed for every randomized claim test
SEED = 20241012

#: the project's stated gravity (M1.yaml conventions), RETYPED as a literal: it must not be
#: read from ``gal3.G_DEFAULT``, or every test that uses it follows a mutation of the
#: default (pass-9 M1-F22: G_DEFAULT -> (0, 0, +9.81) or -9.80665 survived the suite).
#: ``gal3.G_DEFAULT`` is pinned against it (and against gtsam) in the M1-C53 test.
G = np.array([0.0, 0.0, -9.81])

#: RETYPED literal gravity set (pass-10 M1-F26 / M1-F27): one vector in EACH of the 8
#: sign octants, with magnitudes 3.97 .. 14.6 (none equal to 9.81).  Unlike
#: ``rand_gravity`` draws, no fixture edit can change these without a visible change to
#: a literal, so a per-draw nonlinear g map in src (hard-coded |g|, sign flip in some
#: octants) is caught regardless of how the random draws are distributed.
G_LITERAL_SET = np.array([
    [1.2, 2.3, 3.1],        # + + +
    [4.0, 1.1, -6.5],       # + + -
    [2.5, -7.0, 1.8],       # + - +
    [9.0, -3.3, -11.0],     # + - -
    [-1.4, 0.9, 3.6],       # - + +
    [-5.5, 8.2, -1.7],      # - + -
    [-0.8, -2.2, 3.5],      # - - +
    [-6.0, -4.0, -9.81],    # - - -
])

#: RETYPED literal gravity set for the FREE-FALL BACKSTOPS (pass-11 M1-F30 / M1-F31).
#: Three passes running, a narrower random-g fixture (fixed |g|, an octant flip, |g| in
#: (2, 11.5)) passed every statistical guard and masked a g-dependent src defect (hard
#: coded |g|, sign flip, saturation ``np.clip(g, -11.5, 11.5)``).  Guards on a random
#: block's statistics cannot rule out every per-draw map, so EVERY g-consuming src
#: function (gravity_lift, gravity_lift_uncorrected, gravity_element,
#: gravity_generator, integrate_truth, forster_endpoint, and the endpoint forms of
#: M1-C31..C38 / C43) is also checked against exact closed-form constant-specific-force
#: kinematics on THIS set, which no edit to ``rand_gravity`` or to the datasets reaches.
#: It spans the claims' stated |g| range (2, 20): G_LITERAL_SET's 8 octant vectors
#: (|g| 3.97 .. 14.6), four of |g| ~ 2 (1.98 .. 2.08), one per axis AND sign with a
#: component of magnitude 18.9 .. 19.9, two with every component of magnitude >= 11.3
#: (|g| ~ 20), and three more of |g| 18.7 .. 19.6.  Every sign octant occurs at least twice.
#: (No finite set excludes a defect acting only beyond its largest entries; the set
#: covers the stated range up to its top, |g_k| = 19.9.)
G_BACKSTOP_SET = np.vstack([G_LITERAL_SET, np.array([
    [1.1, -1.2, 1.3],       # + - +   |g| 2.08
    [-1.0, 1.4, -1.0],      # - + -   |g| 1.99
    [1.2, 1.0, -1.3],       # + + -   |g| 2.03
    [-1.3, -1.2, -0.9],     # - - -   |g| 1.99
    [19.0, 0.5, -0.8],      # +x large
    [-19.9, 0.7, 1.1],      # -x large
    [0.6, 19.4, 0.9],       # +y large
    [-0.7, -18.9, -1.2],    # -y large
    [0.3, -0.9, 19.6],      # +z large
    [1.3, 0.4, -19.3],      # -z large
    [11.6, -11.5, 11.4],    # every component large, |g| 19.92
    [-11.7, 11.3, -11.6],   # every component large, |g| 19.98
    [-15.2, -9.1, 6.0],     # - - +   |g| 18.7
    [7.3, 13.1, 12.2],      # + + +   |g| 19.3
    [16.5, -4.2, -9.7],     # + - -   |g| 19.6
])])

#: Dt values of the free-fall backstops: both signs, and a long one (g Dt^2 large)
BACKSTOP_DT = (0.4, -0.3, 1.1)


def assert_backstop_set_covers(gs=None):
    """INPUT GUARD on the literal backstop set (data only, nothing from src).

    Deliberately independent of ``assert_gravity_directions_diverse`` (a mutant that
    switches the random-g guards off leaves this one in place).  Requires: every sign
    octant at least once (``bs_octants``); min |g| <= 2.1 and max |g| >= 19.9 (the
    stated range (2, 20), ``bs_mag``); and for EACH axis a component >= +18.9 and one
    <= -18.9 (``bs_axis``, the smallest of those six margins).
    """
    gs = G_BACKSTOP_SET if gs is None else np.asarray(gs, dtype=float)
    codes = ((gs > 0) * np.array([4, 2, 1])).sum(axis=1)
    bs_octants = int(np.bincount(codes, minlength=8).min())
    assert bs_octants >= 1, bs_octants
    mags = np.linalg.norm(gs, axis=1)
    bs_mag = (float(mags.min()), float(mags.max()))
    assert bs_mag[0] <= 2.1 and bs_mag[1] >= 19.9, bs_mag
    bs_axis = float(min(gs.max(axis=0).min(), (-gs.min(axis=0)).min()))
    assert bs_axis >= 18.9, bs_axis
    return bs_octants, bs_mag, bs_axis


def mat5(R, v, p, t) -> np.ndarray:
    """The 5x5 Gal(3) matrix [[R, v, p], [0, 1, t], [0, 0, 1]], built here (not src)."""
    X = np.eye(5)
    X[0:3, 0:3] = np.asarray(R, dtype=float)
    X[0:3, 3] = np.asarray(v, dtype=float).reshape(3)
    X[0:3, 4] = np.asarray(p, dtype=float).reshape(3)
    X[3, 4] = float(t)
    return X


def freefall_cases(r):
    """The free-fall backstop instances: for every g of ``G_BACKSTOP_SET``, every Dt of
    ``BACKSTOP_DT`` and a body acceleration a in {0, N(0, I)}: a random keyframe
    (R_i, v_i, p_i) and the EXACT closed-form solution of the navigation ODE with
    omega = 0 and constant a, i.e. constant specific force f = g + R_i a:

        R_j = R_i,  v_j = v_i + f Dt,  p_j = p_i + v_i Dt + f Dt^2 / 2,

    and the matching exact gravity-free increment D = (I, a Dt, a Dt^2 / 2, Dt).
    Everything is numpy arithmetic written here; nothing comes from src.
    Yields dicts with keys g, Dt, R_i, v_i, p_i, a, D, R_j, v_j, p_j.
    """
    for g in G_BACKSTOP_SET:
        for Dt in BACKSTOP_DT:
            for a_scale in (0.0, 1.0):
                R_i = rand_rot(r)
                v_i, p_i = 2.0 * r.normal(size=3), 5.0 * r.normal(size=3)
                a = a_scale * r.normal(size=3)
                f = g + R_i @ a
                yield dict(g=g, Dt=Dt, R_i=R_i, v_i=v_i, p_i=p_i, a=a,
                           D=mat5(np.eye(3), a * Dt, 0.5 * a * Dt * Dt, Dt),
                           R_j=R_i, v_j=v_i + f * Dt,
                           p_j=p_i + v_i * Dt + 0.5 * f * Dt * Dt)


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
                                      min_span=0.1, min_signed=1.0, min_mag_ratio=2.0,
                                      min_max_mag=14.0):
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

    * for m >= 3, EVERY component takes BOTH signs with magnitude >= ``min_signed``:
      max_k g_k >= min_signed and min_k g_k <= -min_signed for k = x, y, z (``signed``,
      the smallest of those six margins).

    This closes pass-9 M1-F23: span and comp only rule out LINEAR defects (g -> M g); a
    HALF-SPACE distribution (g_y := |g_y|) passed every guard, and under it a
    sign-dependent defect (g_y -> |g_y| in gravity_lift or integrate_truth) survived the
    suite.  Measured ``signed`` on the recorded blocks: 2.102 and 4.321 (the 4-g datasets,
    Dt = +0.4 / -0.3), 16.7..18.8 (the 400..2000-draw blocks); a half-space block has
    signed <= 0.  The single-draw C45 block is exempt (m < 3).

    * for m >= 3, the MAGNITUDES vary: max |g| / min |g| >= ``min_mag_ratio`` (2.0)
      (``mag_ratio``).
    * for m >= 8, EVERY one of the 8 sign octants of g occurs at least once
      (``octants``, the smallest octant count).

    These close pass-10 M1-F26 / M1-F27, two fixtures that preserve every guard above
    by construction: H_MAG2 (|g| = 9.81 on every draw, directions unchanged) masked a
    src defect hard-coding the standard magnitude (g -> 9.81 g / |g|), and H_ODD
    (g -> -g wherever g_x g_y g_z < 0; only four octants remain, each component still
    takes both signs) masked the matching per-draw sign flip.  Measured on the
    recorded blocks: mag_ratio 2.537 and 3.430 (the 4-g datasets, Dt = +0.4 / -0.3),
    9.88..9.97 (the 400..2000-draw blocks); octants 38..229 (the 400..2000-draw
    blocks; the 4-g datasets are exempt).  Under H_MAG2 mag_ratio is 1.0; under H_ODD
    four octants are empty (octants = 0).  Guards on marginal statistics cannot exclude
    every per-draw nonlinear map, so the behavioural backstop is ``G_LITERAL_SET``,
    which no edit to ``rand_gravity`` reaches.

    * for m >= 3, the TOP of the stated range (2, 20) is reached: max |g| >=
      ``min_max_mag`` (14.0) (``mag_max``).

    This closes the FIXTURE half of pass-11 M1-F30: H_LOW (|g| ~ U(2, 11.5)) has
    mag_max <= 11.5.  Measured: 19.82 and 18.47 (the 4-g datasets), >= 19.9 on the
    400..2000-draw blocks, 14.59 on G_LITERAL_SET (hence 14.0, not 15).  The fixture
    half of M1-F31 is ``assert_rand_g_datasets_parity``.  The SRC half of both -- any
    g-dependent defect a degenerate fixture could hide -- is closed by the free-fall
    backstops on ``G_BACKSTOP_SET`` (``freefall_cases``), which do not depend on these
    guards at all.
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
    signed = None
    if len(gs) >= 3:
        signed = float(min(gs.max(axis=0).min(), (-gs.min(axis=0)).min()))
        assert signed >= min_signed, signed
    mag_ratio = None
    if len(gs) >= 3:
        mags = np.linalg.norm(gs, axis=1)
        mag_ratio = float(mags.max() / mags.min())
        assert mag_ratio >= min_mag_ratio, mag_ratio
    octants = None
    if len(gs) >= 8:
        codes = ((gs > 0) * np.array([4, 2, 1])).sum(axis=1)
        octants = int(np.bincount(codes, minlength=8).min())
        assert octants >= 1, octants
    mag_max = None
    if len(gs) >= 3:
        mag_max = float(np.linalg.norm(gs, axis=1).max())
        assert mag_max >= min_max_mag, mag_max
    return off_z, perm_gap, spread, comp, span, signed, mag_ratio, octants, mag_max


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


def assert_rand_g_datasets_parity(datasets):
    """INPUT GUARD on ``endpoint_datasets_rand_g`` (pass-11 M1-F31), called at the END of
    the dataset tests: in each dataset, g_x g_y g_z takes BOTH signs among its 4 distinct
    g (``parity``, the smaller count).  The octant guard exempts m = 4 blocks, so a
    dataset-local odd-octant flip (H_ODD_DS) passed every guard.  Measured parity: 1 and 1
    (Dt = +0.4 / -0.3); under H_ODD_DS 0.  It is called LAST so that every older mutant
    (e.g. pass-10's PFLIP + H_ODD, killed by the literal backstop of M1-C31) still dies on
    its original assertion; the src half (PFLIP in gravity_lift_uncorrected) is caught
    earlier, by the free-fall backstop.
    """
    for d in datasets:
        prod = np.prod(np.unique(d["g"], axis=0), axis=1)
        parity = int(min((prod > 0).sum(), (prod < 0).sum()))
        assert parity >= 1, (d["label"], parity)


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
