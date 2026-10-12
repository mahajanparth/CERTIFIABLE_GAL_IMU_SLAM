"""One test per claim in docs/claims/M1.yaml, claims M1-C01..M1-C36, C38..C40, C46..C52.

Each test is bound to its claim with @pytest.mark.claim; the orchestrator discovers
claims through that marker and nothing else.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.spatial.transform import Rotation

from src import gal3
from src.gal3 import (
    Ad,
    Ad_apply,
    ad,
    cay,
    cay_matrix,
    compose,
    exp,
    gal,
    gamma1,
    gamma1_inv,
    gamma2,
    inverse,
    log,
    skew,
    so3_exp,
    so3_log,
    unpack,
    vee,
    wedge,
)

from .conftest import (  # noqa: F401  (relative import keeps the helpers in one place)
    G,
    _rotvec_to_matrix,
    rand_axis,
    rand_gal,
    rand_rot,
    rand_rot_angle,
    rand_xi,
    rand_xi_bounded_phi,
    rng,
    assert_discrete_converges,
    assert_term_exercised,
)

I3 = np.eye(3)
I5 = np.eye(5)
I10 = np.eye(10)


# ==========================================================================
# the group
# ==========================================================================
@pytest.mark.claim("M1-C01")
def test_product_closed_form():
    """X1 X2 = (R1 R2, v1 + R1 v2, p1 + R1 p2 + v1 t2, t1 + t2)."""
    r = rng(1)
    worst = 0.0
    term = []
    for _ in range(2000):
        X1, X2 = rand_gal(r), rand_gal(r)
        R1, v1, p1, t1 = unpack(X1)
        R2, v2, p2, t2 = unpack(X2)
        closed = gal(R1 @ R2, v1 + R1 @ v2, p1 + R1 @ p2 + v1 * t2, t1 + t2)
        worst = max(worst, float(np.abs(X1 @ X2 - closed).max()))
        worst = max(worst, float(np.abs(compose(X1, X2) - closed).max()))
        term.append(float(np.abs(v1 * t2).max()))
    # the v1*t2 term is load-bearing: it must be non-negligible on the sample set
    assert_term_exercised(term)
    assert worst <= 1e-12, worst


@pytest.mark.claim("M1-C02")
def test_closure():
    """The product stays in Gal(3): SO(3) block, zero rows, unit diagonal."""
    r = rng(2)
    worst = 0.0
    for _ in range(2000):
        P = compose(rand_gal(r), rand_gal(r))
        worst = max(worst, gal3.structural_residual(P))
    assert worst <= 1e-12, worst


@pytest.mark.claim("M1-C03")
def test_associativity():
    """(X1 X2) X3 = X1 (X2 X3), through the implementation's compose()."""
    r = rng(3)
    worst = 0.0
    for _ in range(2000):
        X1, X2, X3 = rand_gal(r), rand_gal(r), rand_gal(r)
        left = compose(compose(X1, X2), X3)
        right = compose(X1, compose(X2, X3))
        worst = max(worst, float(np.abs(left - right).max()))
    assert worst <= 1e-10, worst


@pytest.mark.claim("M1-C04")
def test_identity_two_sided():
    """I5 is a two-sided identity."""
    r = rng(4)
    E = gal3.identity()
    worst = 0.0
    for _ in range(2000):
        X = rand_gal(r)
        worst = max(worst, float(np.abs(compose(E, X) - X).max()))
        worst = max(worst, float(np.abs(compose(X, E) - X).max()))
    assert worst <= 1e-14, worst


@pytest.mark.claim("M1-C05")
def test_inverse_two_sided():
    """X^-1 = (R^T, -R^T v, -R^T (p - t v), -t), with (X X^-1)[3,4] exactly 0."""
    r = rng(5)
    worst = 0.0
    worst_time = 0.0
    term = []
    for _ in range(2000):
        X = rand_gal(r, t_scale=3.0)
        Xi = inverse(X)
        A, B = compose(X, Xi), compose(Xi, X)
        worst = max(worst, float(np.abs(A - I5).max()), float(np.abs(B - I5).max()))
        worst_time = max(worst_time, abs(float(A[3, 4])))
        _, v, _, t = unpack(X)
        term.append(float(np.abs(t * v).max()))
    # the -R^T (p - t v) term is only exercised when t v is non-negligible
    assert_term_exercised(term)
    assert worst <= 1e-10, worst
    assert worst_time <= 1e-12, worst_time


@pytest.mark.claim("M1-C06")
def test_non_abelian():
    """There exist X1, X2 with max|X1 X2 - X2 X1| >= 1e-3."""
    r = rng(6)
    worst = 0.0
    witness = None
    for _ in range(200):
        X1, X2 = rand_gal(r), rand_gal(r)
        d = float(np.abs(compose(X1, X2) - compose(X2, X1)).max())
        if d > worst:
            worst, witness = d, (X1, X2)
    assert witness is not None
    assert worst >= 1e-3, worst


@pytest.mark.claim("M1-C07")
def test_time_block_additive():
    """t(X1 X2) = t1 + t2 and t(X^-1) = -t, both exactly."""
    r = rng(7)
    worst = 0.0
    for _ in range(2000):
        X1, X2 = rand_gal(r), rand_gal(r)
        t1, t2 = float(X1[3, 4]), float(X2[3, 4])
        worst = max(worst, abs(float(compose(X1, X2)[3, 4]) - (t1 + t2)))
        worst = max(worst, abs(float(inverse(X1)[3, 4]) + t1))
    assert worst <= 1e-12, worst


# ==========================================================================
# the algebra
# ==========================================================================
@pytest.mark.claim("M1-C08")
def test_wedge_vee_inverse():
    """wedge and vee are mutually inverse; u is column 3, w is column 4."""
    r = rng(8)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi(r)
        worst = max(worst, float(np.abs(vee(wedge(xi)) - xi).max()))
        A = wedge(xi)
        worst = max(worst, float(np.abs(wedge(vee(A)) - A).max()))
    assert worst <= 1e-14, worst
    # pin the (vel, pos) block order: 0-indexed e3..e5 -> velocity, e6..e8 -> position
    e = np.eye(gal3.DIM)
    assert np.array_equal(wedge(e[3])[0:3, 3], np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(wedge(e[6])[0:3, 4], np.array([1.0, 0.0, 0.0]))
    assert np.abs(wedge(e[3])[0:3, 4]).max() == 0.0
    assert np.abs(wedge(e[6])[0:3, 3]).max() == 0.0


@pytest.mark.claim("M1-C09")
def test_ad_blocks_and_commutator():
    """ad_xi has the stated blocks and wedge(ad_xi eta) = xi^ eta^ - eta^ xi^."""
    r = rng(9)
    worst = 0.0
    for _ in range(2000):
        xi, eta = rand_xi(r), rand_xi(r)
        lhs = wedge(ad(xi) @ eta)
        X, E = wedge(xi), wedge(eta)
        worst = max(worst, float(np.abs(lhs - (X @ E - E @ X)).max()))
    assert worst <= 1e-10, worst
    xi = rand_xi(r)
    A = ad(xi)
    assert np.abs(A[9, :]).max() == 0.0
    assert np.array_equal(A[6:9, 9], xi[3:6])
    # and the block structure itself
    phi, u, w, al = xi[0:3], xi[3:6], xi[6:9], xi[9]
    blocks = np.zeros((10, 10))
    blocks[0:3, 0:3] = skew(phi)
    blocks[3:6, 0:3] = skew(u)
    blocks[3:6, 3:6] = skew(phi)
    blocks[6:9, 0:3] = skew(w)
    blocks[6:9, 3:6] = -al * I3
    blocks[6:9, 6:9] = skew(phi)
    blocks[6:9, 9] = u
    assert np.abs(A - blocks).max() == 0.0


@pytest.mark.claim("M1-C10")
def test_ad_homomorphism_and_jacobi():
    """ad_{ad_xi eta} = [ad_xi, ad_eta] and the Jacobi identity."""
    r = rng(10)
    worst_hom = worst_jac = 0.0
    for _ in range(2000):
        xi, eta = rand_xi(r), rand_xi(r)
        A, B = ad(xi), ad(eta)
        worst_hom = max(worst_hom, float(np.abs(ad(A @ eta) - (A @ B - B @ A)).max()))
        x, y, z = rand_xi(r), rand_xi(r), rand_xi(r)
        jac = ad(x) @ (ad(y) @ z) + ad(y) @ (ad(z) @ x) + ad(z) @ (ad(x) @ y)
        worst_jac = max(worst_jac, float(np.abs(jac).max()))
    assert worst_hom <= 1e-12, worst_hom
    assert worst_jac <= 1e-12, worst_jac


@pytest.mark.claim("M1-C11")
def test_Ad_closed_form():
    """Ad_X(xi) = vee(X xi^ X^-1) with the stated closed form; needs t != 0."""
    r = rng(11)
    worst = 0.0
    term = []
    for _ in range(2000):
        X = rand_gal(r, t_scale=1.5)
        xi = rand_xi(r)
        term.append(float(np.abs(float(X[3, 4]) * xi[3:6]).max()))   # the -t R u term
        lhs = wedge(Ad_apply(X, xi))
        rhs = X @ wedge(xi) @ inverse(X)
        worst = max(worst, float(np.abs(lhs - rhs).max()))
        # the matrix form must agree with the componentwise one
        worst = max(worst, float(np.abs(Ad(X) @ xi - Ad_apply(X, xi)).max()))
    assert_term_exercised(term)
    assert worst <= 1e-9, worst


@pytest.mark.claim("M1-C12")
def test_Ad_group_homomorphism():
    """Ad_{XY} = Ad_X Ad_Y."""
    r = rng(12)
    worst = 0.0
    for _ in range(2000):
        X, Y = rand_gal(r), rand_gal(r)
        worst = max(worst, float(np.abs(Ad(compose(X, Y)) - Ad(X) @ Ad(Y)).max()))
    assert worst <= 1e-8, worst


@pytest.mark.claim("M1-C13")
def test_Ad_of_inverse():
    """Ad_{X^-1} = (Ad_X)^-1."""
    r = rng(13)
    worst = 0.0
    for _ in range(2000):
        X = rand_gal(r)
        worst = max(worst, float(np.abs(Ad(inverse(X)) - np.linalg.inv(Ad(X))).max()))
    assert worst <= 1e-8, worst


@pytest.mark.claim("M1-C14")
def test_Ad_ad_compatibility():
    """Ad_X ad_xi Ad_X^-1 = ad_{Ad_X xi}, in both the matrix and applied forms."""
    r = rng(14)
    worst_m = worst_a = 0.0
    for _ in range(2000):
        X = rand_gal(r)
        xi, eta = rand_xi(r, 0.5), rand_xi(r, 0.5)
        AX = Ad(X)
        AXi = np.linalg.inv(AX)
        worst_m = max(worst_m, float(np.abs(AX @ ad(xi) @ AXi - ad(AX @ xi)).max()))
        lhs = AX @ (ad(xi) @ eta)
        rhs = ad(AX @ xi) @ (AX @ eta)
        worst_a = max(worst_a, float(np.abs(lhs - rhs).max()))
    assert worst_m <= 1e-7, worst_m
    assert worst_a <= 1e-7, worst_a


@pytest.mark.claim("M1-C15")
def test_Ad_exp_equals_expm_ad():
    """Ad_{exp(xi)} = expm(ad_xi)."""
    r = rng(15)
    worst = 0.0
    for _ in range(500):
        xi = rand_xi_bounded_phi(r, 0.05, 2.9, scale=0.6)
        worst = max(worst, float(np.abs(Ad(exp(xi)) - sla.expm(ad(xi))).max()))
    assert worst <= 1e-8, worst


@pytest.mark.claim("M1-C16")
def test_time_is_Ad_invariant():
    """The last row of Ad_X is e_10^T for every X."""
    r = rng(16)
    e10 = np.zeros(gal3.DIM)
    e10[9] = 1.0
    worst = 0.0
    for _ in range(2000):
        worst = max(worst, float(np.abs(Ad(rand_gal(r))[9, :] - e10).max()))
    assert worst <= 1e-14, worst


@pytest.mark.claim("M1-C17")
def test_unimodular():
    """det(Ad_X) = 1 and trace(ad_xi) = 0."""
    r = rng(17)
    worst_det = worst_tr = 0.0
    for _ in range(500):
        worst_det = max(worst_det, abs(float(np.linalg.det(Ad(rand_gal(r)))) - 1.0))
    for _ in range(500):
        worst_tr = max(worst_tr, abs(float(np.trace(ad(rand_xi(r))))))
    assert worst_det <= 1e-12, worst_det
    assert worst_tr <= 1e-14, worst_tr


# ==========================================================================
# Gamma coefficients, exp and log
# ==========================================================================
def _gamma_series(phi, m, terms=41):
    """sum_{n=0}^{terms-1} [phi]x^n / (n + m)!."""
    from math import factorial

    P = skew(phi)
    acc = np.zeros((3, 3))
    Pn = I3.copy()
    for n in range(terms):
        acc = acc + Pn / float(factorial(n + m))
        Pn = Pn @ P
    return acc


@pytest.mark.claim("M1-C18")
def test_gamma_kappa_form_matches_series():
    """Gamma1, Gamma2 in kappa form equal the truncated series sum Phi^n/(n+m)!."""
    r = rng(18)
    worst = 0.0
    for _ in range(2000):
        phi = r.uniform(1e-3, 3.0) * rand_axis(r)
        worst = max(worst, float(np.abs(gamma1(phi) - _gamma_series(phi, 1)).max()))
        worst = max(worst, float(np.abs(gamma2(phi) - _gamma_series(phi, 2)).max()))
    assert worst <= 1e-12, worst
    # the limits at theta = 0
    assert np.abs(gamma1(np.zeros(3)) - I3).max() <= 1e-15
    assert np.abs(gamma2(np.zeros(3)) - 0.5 * I3).max() <= 1e-15


@pytest.mark.claim("M1-C19")
def test_exp_closed_form():
    """exp(xi) = (Exp(phi), G1 u, G1 w + alpha G2 u, alpha) = expm(xi^)."""
    r = rng(19)
    worst = 0.0
    term = []
    for _ in range(2000):
        xi = rand_xi_bounded_phi(r, 0.05, 2.9)
        phi, u, w, al = xi[0:3], xi[3:6], xi[6:9], xi[9]
        term.append(float(np.abs(al * (gamma2(phi) @ u)).max()))
        closed = gal(so3_exp(phi), gamma1(phi) @ u,
                     gamma1(phi) @ w + al * (gamma2(phi) @ u), al)
        worst = max(worst, float(np.abs(exp(xi) - closed).max()))
        worst = max(worst, float(np.abs(exp(xi) - sla.expm(wedge(xi))).max()))
    assert_term_exercised(term)   # the alpha*G2*u coupling is exercised
    assert worst <= 1e-9, worst


@pytest.mark.claim("M1-C20")
def test_exp_time_block_exact():
    """exp(xi)[3,4] = alpha exactly."""
    r = rng(20)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi(r)
        xi[9] = 3.0 * r.normal()
        worst = max(worst, abs(float(exp(xi)[3, 4]) - float(xi[9])))
    assert worst <= 1e-12, worst


@pytest.mark.claim("M1-C21")
def test_log_and_gamma1_inverse():
    """Gamma1^-1 = I - Phi/2 + c2 Phi^2 with the non-singular c2; log has the stated form."""
    r = rng(21)
    theta = np.exp(r.uniform(np.log(1e-9), np.log(np.pi), size=20000))
    axes = rand_axis(r, 20000)
    worst = 0.0
    for th, ax in zip(theta, axes):
        phi = th * ax
        worst = max(worst, float(np.abs(gamma1_inv(phi) @ gamma1(phi) - I3).max()))
    assert worst <= 1e-13, worst

    # the stated log formula, componentwise
    worst_log = 0.0
    for _ in range(500):
        X = rand_gal(r, t_scale=2.0, rot_scale=0.9)
        R, v, p, t = unpack(X)
        phi = so3_log(R)
        u = gamma1_inv(phi) @ v
        w = gamma1_inv(phi) @ (p - t * (gamma2(phi) @ u))
        worst_log = max(worst_log,
                        float(np.abs(log(X) - np.concatenate([phi, u, w, [t]])).max()))
        worst_log = max(worst_log,
                        float(np.abs(gamma1_inv(phi) - np.linalg.inv(gamma1(phi))).max()))
    assert worst_log <= 1e-12, worst_log

    # the equivalent form 1/th^2 - (1+cos th)/(2 th sin th) must NOT be used:
    # it is the one that blows up on the chart edge.  Ours stays exact there.
    import mpmath as mp

    mp.mp.dps = 60

    def exact_c2(th):
        T = mp.mpf(th)
        return (1 - T * mp.sin(T) / (2 * (1 - mp.cos(T)))) / T ** 2

    def forbidden(th):
        with np.errstate(all="ignore"):
            return 1.0 / th ** 2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))

    assert np.isfinite(gal3.gamma_coefficients(np.pi)[3])
    ours_worst = forbidden_worst = 0.0
    for th in [np.pi] + [np.pi - 10.0 ** -k for k in range(3, 10)]:
        e = exact_c2(th)
        ours = gal3.gamma_coefficients(th)[3]
        f = forbidden(th)
        ours_worst = max(ours_worst, float(abs(mp.mpf(ours) - e) / abs(e)))
        forbidden_worst = max(
            forbidden_worst,
            float(abs(mp.mpf(float(f)) - e) / abs(e)) if np.isfinite(f) else np.inf,
        )
    assert ours_worst <= 1e-13, ours_worst
    assert forbidden_worst >= 1e-10, forbidden_worst


@pytest.mark.claim("M1-C22")
def test_log_exp_roundtrip():
    """log(exp(xi)) = xi for ||phi|| < pi; above pi, log returns the wrapped branch."""
    r = rng(22)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi_bounded_phi(r, 0.05, np.pi - 1e-3)
        worst = max(worst, float(np.abs(log(exp(xi)) - xi).max()))
    assert worst <= 1e-9, worst

    # above pi the SO(3) log wraps: |phi| = 3.5 comes back as 2 pi - 3.5
    for _ in range(50):
        ax = rand_axis(r)
        xi = np.zeros(gal3.DIM)
        xi[0:3] = 3.5 * ax
        back = log(exp(xi))
        assert abs(float(np.linalg.norm(back[0:3])) - (2 * np.pi - 3.5)) <= 1e-9
        assert np.abs(back[0:3] + (2 * np.pi - 3.5) * ax).max() <= 1e-9
        assert float(np.abs(back[0:3] - xi[0:3]).max()) >= 1e-3


@pytest.mark.claim("M1-C23")
def test_exp_surjective():
    """exp(log(X)) = X for every X in Gal(3), including |phi| arbitrarily close to pi."""
    r = rng(23)
    worst = 0.0
    worst_theta = 0.0
    for _ in range(5000):
        X = gal3.gal(rand_rot(r, scale=5.0), r.normal(size=3), r.normal(size=3),
                     r.normal())
        worst = max(worst, float(np.abs(exp(log(X)) - X).max()))
        worst_theta = max(worst_theta, float(np.linalg.norm(so3_log(X[0:3, 0:3]))))
    assert worst_theta > np.pi - 0.05, worst_theta   # the sample set reaches the edge
    assert worst <= 1e-9, worst


@pytest.mark.claim("M1-C24")
def test_gamma1_invertibility_and_taylor_branch():
    """det/cond of Gamma1 and the mandatory Taylor branch below theta = 5e-2."""
    import mpmath as mp

    mp.mp.dps = 60

    ax = np.array([0.0, 0.0, 1.0])
    det0 = float(np.linalg.det(gamma1(np.zeros(3))))
    det_pi = float(np.linalg.det(gamma1(np.pi * ax)))
    det_6 = float(np.linalg.det(gamma1(6.0 * ax)))
    det_2pi = float(np.linalg.det(gamma1(2 * np.pi * ax)))
    cond_pi = float(np.linalg.cond(gamma1(np.pi * ax)))
    assert abs(det0 - 1.0) <= 5e-5, det0
    assert abs(det_pi - 0.4053) <= 5e-5, det_pi
    assert abs(det_6 - 2.21e-3) <= 5e-6, det_6
    assert abs(det_2pi) <= 1e-12, det_2pi
    assert abs(cond_pi - 1.5708) <= 5e-5, cond_pi
    assert abs(cond_pi - np.pi / 2) <= 1e-10, cond_pi

    def exact(th):
        T = mp.mpf(th)
        s, c = mp.sin(T), mp.cos(T)
        return [(1 - c) / T ** 2, (T - s) / T ** 3,
                (T ** 2 + 2 * c - 2) / (2 * T ** 4),
                (1 - T * s / (2 * (1 - c))) / T ** 2]

    thetas = [1e-9, 1e-6, 1e-4, 1e-2, 5e-2, 1e-1, 1.0, 3.0]
    worst = [0.0, 0.0, 0.0, 0.0]
    for th in thetas:
        e = exact(th)
        got = gal3.gamma_coefficients(th)
        for i in range(4):
            worst[i] = max(worst[i], float(abs(mp.mpf(got[i]) - e[i]) / abs(e[i])))
    assert max(worst) <= 1e-11, worst

    # and the same assertion WITHOUT the branch fails, in the documented way
    e = exact(1e-4)
    bad = gal3.gamma_coefficients_closed_form(1e-4)
    rel_k3 = float(abs(mp.mpf(bad[2]) - e[2]) / abs(e[2]))
    rel_c2 = float(abs(mp.mpf(bad[3]) - e[3]) / abs(e[3]))
    assert rel_k3 > 1e-11 and rel_c2 > 1e-11
    assert 0.5 <= rel_k3 <= 2.0, rel_k3        # reported 1.0
    assert 5.0 <= rel_c2 <= 8.0, rel_c2        # reported 6.3

    # the switch point itself
    assert gal3.THETA_SMALL == 5e-2

    # Dense sweep against a SERIES-defined 50-digit reference (independent of every
    # closed form): 1e-11 must hold everywhere, not only at the eight tabulated
    # theta -- in particular on both sides of the switch.
    mp.mp.dps = 50

    def series_ref(th):
        T = mp.mpf(th)
        T2 = T * T
        k1 = mp.nsum(lambda k: (-1) ** k * T2 ** k / mp.factorial(2 * k + 2), [0, mp.inf])
        k2 = mp.nsum(lambda k: (-1) ** k * T2 ** k / mp.factorial(2 * k + 3), [0, mp.inf])
        k3 = mp.nsum(lambda k: (-1) ** k * T2 ** k / mp.factorial(2 * k + 4), [0, mp.inf])
        if th < 1.0:   # c2 = sum_{n>=1} (-1)^(n+1) B_2n th^(2n-2) / (2n)!
            c2 = mp.fsum((-1) ** (n + 1) * mp.bernoulli(2 * n) * T2 ** (n - 1)
                         / mp.factorial(2 * n) for n in range(1, 40))
        else:          # well away from any cancellation at 50 digits
            c2 = (1 - T * mp.sin(T) / (2 * (1 - mp.cos(T)))) / T2
        return k1, k2, k3, c2

    sweep = np.concatenate([np.logspace(-12, 1, 400), np.linspace(0.045, 0.2, 400),
                            np.linspace(3.0, 6.2, 60),
                            [np.pi, np.nextafter(5e-2, 0.0), 5e-2]])
    worst_sweep = [0.0] * 4
    for th in sweep:
        e = series_ref(float(th))
        got = gal3.gamma_coefficients(float(th))
        for i in range(4):
            worst_sweep[i] = max(worst_sweep[i],
                                 float(abs(mp.mpf(got[i]) - e[i]) / abs(e[i])))
    assert max(worst_sweep) <= 1e-11, worst_sweep

    # THETA_SMALL = 5e-2 is LOAD-BEARING.  Just below the switch the binding error
    # is the 3-term Taylor truncation (relative error ~ theta^6), not the 1 - cos
    # cancellation: kappa1's truncation is 7.75e-13 at the switch and would be
    # 1.30e-11 at 8e-2 -- moving the switch there would break the 1e-11 bound.
    def taylor_k1(th):
        return 0.5 - th ** 2 / 24.0 + th ** 4 / 720.0

    below = float(np.nextafter(gal3.THETA_SMALL, 0.0))
    trunc_at_switch = float(abs(mp.mpf(gal3.gamma_coefficients(below)[0])
                                - series_ref(below)[0]) / series_ref(below)[0])
    trunc_at_8e2 = float(abs(mp.mpf(taylor_k1(8e-2)) - series_ref(8e-2)[0])
                         / series_ref(8e-2)[0])
    assert 5e-13 <= trunc_at_switch <= 1e-12, trunc_at_switch
    assert trunc_at_8e2 > 1e-11, trunc_at_8e2


# ==========================================================================
# Cayley
# ==========================================================================
@pytest.mark.claim("M1-C25")
def test_cayley_closed_form():
    """Cay(xi) = (S^-1(I+[phi]x/2), S^-1 u, S^-1(w + (alpha/2) u), alpha)."""
    r = rng(25)
    worst = 0.0
    term = []
    for _ in range(2000):
        xi = rand_xi_bounded_phi(r, 0.05, 2.9)
        term.append(float(np.abs(0.5 * xi[9] * xi[3:6]).max()))
        worst = max(worst, float(np.abs(cay(xi) - cay_matrix(xi)).max()))
        # and the stated blocks, written out
        phi, u, w, al = xi[0:3], xi[3:6], xi[6:9], xi[9]
        S = I3 - skew(phi) / 2.0
        Si = np.linalg.inv(S)
        ref = gal(Si @ (I3 + skew(phi) / 2.0), Si @ u, Si @ (w + 0.5 * al * u), al)
        worst = max(worst, float(np.abs(cay(xi) - ref).max()))
    assert_term_exercised(term)     # the (alpha/2) u term is exercised
    assert worst <= 1e-9, worst

    # the MINUS form is wrong, and by a lot
    r2 = rng(2500)
    worst_minus = 0.0
    for _ in range(200):
        xi = rand_xi_bounded_phi(r2, 0.05, 2.9)
        phi, u, w, al = xi[0:3], xi[3:6], xi[6:9], xi[9]
        S = I3 - skew(phi) / 2.0
        minus = np.linalg.solve(S, w - 0.5 * al * u)
        worst_minus = max(worst_minus, float(np.abs(cay(xi)[0:3, 4] - minus).max()))
    assert worst_minus >= 1e-3, worst_minus


@pytest.mark.claim("M1-C26")
def test_cayley_denominator_never_vanishes():
    """det(I5 - xi^/2) = 1 + ||phi||^2/4 > 0, so Cayley is total."""
    r = rng(26)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi(r)
        xi[3:6] *= 50.0
        xi[6:9] *= 50.0
        xi[9] = 3.0 * r.normal()
        d = float(np.linalg.det(gal3.cay_denominator(xi)))
        worst = max(worst, abs(d - (1.0 + float(np.dot(xi[0:3], xi[0:3])) / 4.0)))
        assert d > 0.0
    assert worst <= 1e-10, worst

    # Totality, in a form that can fail: on 20000 draws with |phi| up to 50 and
    # |u|, |w| ~ 1e3 (alpha != 0), cay returns a finite Gal(3) element equal to the
    # defining (I5 - xi^/2)^-1 (I5 + xi^/2).  ("No LinAlgError" alone is vacuous:
    # det(I - [phi]x/2) = 1 + theta^2/4 >= 1 by construction.)
    r2 = rng(2600)
    worst_rel = worst_struct = 0.0
    min_det = np.inf
    for _ in range(20000):
        xi = r2.normal(size=gal3.DIM)
        xi[0:3] *= r2.uniform(0.0, 50.0) / np.linalg.norm(xi[0:3])
        xi[3:9] *= 1e3
        xi[9] = r2.normal() + np.sign(r2.normal()) * 0.5   # alpha != 0
        C = cay(xi)
        assert np.all(np.isfinite(C))
        M = cay_matrix(xi)
        worst_rel = max(worst_rel, float(np.abs(C - M).max()) / max(1.0, float(np.abs(M).max())))
        worst_struct = max(worst_struct, gal3.structural_residual(C))
        min_det = min(min_det, float(np.linalg.det(gal3.cay_denominator(xi))))
    assert worst_rel <= 1e-12, worst_rel
    assert worst_struct <= 1e-12, worst_struct
    assert min_det >= 1.0 - 1e-12, min_det


@pytest.mark.claim("M1-C27")
def test_cayley_lands_in_gal3():
    """Cay(xi) is in Gal(3) automatically; its time block equals alpha exactly."""
    r = rng(27)
    worst = 0.0
    worst_time = 0.0
    for _ in range(2000):
        xi = rand_xi(r)
        C = cay(xi)
        worst = max(worst, gal3.structural_residual(C))
        worst_time = max(worst_time, abs(float(C[3, 4]) - float(xi[9])))
    assert worst <= 1e-10, worst
    assert worst_time <= 1e-12, worst_time


@pytest.mark.claim("M1-C28")
def test_cayley_rotation_tangent_half_angle():
    """Cay([c]x) has angle theta with ||c|| = 2 tan(theta/2)."""
    r = rng(28)
    worst = 0.0
    for _ in range(2000):
        c = r.uniform(0.01, 5.0) * rand_axis(r)
        xi = np.zeros(gal3.DIM)
        xi[0:3] = c
        Re = cay(xi)[0:3, 0:3]
        phi = so3_log(Re)
        theta = float(np.linalg.norm(phi))
        worst = max(worst, abs(float(np.linalg.norm(c)) - 2.0 * np.tan(theta / 2.0)))
        # the axis is preserved
        assert np.abs(phi / theta - c / np.linalg.norm(c)).max() <= 1e-9
    assert worst <= 1e-8, worst

    # the sampler above covers theta in (0.01, 2.38) only; the large-angle regime,
    # theta up to 3.1415, through REAL cay round trips (angle and axis read back
    # with scipy, independently of so3_log): ||c|| log-uniform in (5, 5e4)
    worst_big = 0.0
    for _ in range(2000):
        cn = float(np.exp(r.uniform(np.log(5.0), np.log(5e4))))
        ax = rand_axis(r)
        xi = np.zeros(gal3.DIM)
        xi[0:3] = cn * ax
        rv = Rotation.from_matrix(cay(xi)[0:3, 0:3]).as_rotvec()
        th = float(np.linalg.norm(rv))
        worst_big = max(worst_big, abs(th - 2.0 * np.arctan(cn / 2.0)))
        assert np.abs(rv / th - ax).max() <= 1e-9
    assert worst_big <= 1e-9, worst_big

    # the tabulated values: transcription check, then each pushed through cay
    for theta, tabulated in [(0.5, 0.5107), (1.0, 1.0926), (2.0, 3.1148),
                             (3.0, 28.203), (3.1415, 4.3172e4)]:
        assert abs(2.0 * np.tan(theta / 2.0) - tabulated) <= 5e-5 * max(1.0, abs(tabulated))
        # the tabulation tolerance dc = 5e-5 max(1, c) pins theta to 2 dc / (4 + c^2)
        tol_theta = 2.0 * 5e-5 * max(1.0, tabulated) / (4.0 + tabulated ** 2) + 1e-12
        for _ in range(20):
            ax = rand_axis(r)
            for cn, tol in ((tabulated, tol_theta), (2.0 * np.tan(theta / 2.0), 1e-9)):
                xi = np.zeros(gal3.DIM)
                xi[0:3] = cn * ax
                Re = cay(xi)[0:3, 0:3]
                # orthogonality degrades as cond(S) ~ ||c||^2/4: ~2e-15 at theta = 3.0
                # but ~3e-12 at theta = 3.1415 -- the 'unusable above ~3 rad' domain
                assert np.abs(Re.T @ Re - I3).max() <= 1e-10
                rv = Rotation.from_matrix(Re).as_rotvec()
                th = float(np.linalg.norm(rv))
                assert abs(th - theta) <= tol, (theta, cn, th)
                assert np.abs(rv / th - ax).max() <= 1e-9


@pytest.mark.claim("M1-C29")
def test_cayley_matches_exp_to_second_order():
    """log(Cay(s xi)) = s xi + (s^3/12) vee((xi^)^3) + O(s^5): the SCALING is the claim."""
    r = rng(29)
    xis = []
    for _ in range(300):
        xi = r.normal(size=gal3.DIM)
        xis.append(xi / np.linalg.norm(xi))

    worst3 = 0.0
    for s in (1e-1, 5e-2):
        for xi in xis:
            d = log(cay(s * xi)) - s * xi
            worst3 = max(worst3, float(np.abs(d).max()) / s ** 3)
    assert worst3 <= 1.0, worst3

    s = 1e-2
    worst5 = 0.0
    for xi in xis:
        cube = vee(np.linalg.matrix_power(wedge(xi), 3))
        d = log(cay(s * xi)) - s * xi - (s ** 3 / 12.0) * cube
        worst5 = max(worst5, float(np.abs(d).max()) / s ** 5)
    assert worst5 <= 1.0, worst5


@pytest.mark.claim("M1-C30")
def test_cayley_constraints_degree_two():
    """The Cayley residual is a degree-<=2 block in the lifted variables."""
    import sympy as sp

    r = rng(30)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi_bounded_phi(r, 0.05, 2.9)
        xi[9] = r.normal() + np.sign(r.normal()) * 0.5     # alpha != 0
        phi, u, w, al = xi[0:3], xi[3:6], xi[6:9], xi[9]
        Re, ve, pe, _ = unpack(cay(xi))
        for res in gal3.cayley_constraint_residuals(phi, u, w, al, Re, ve, pe):
            worst = max(worst, float(np.abs(res).max()))
    assert worst <= 1e-10, worst

    exprs, syms = gal3.cayley_constraints_symbolic()
    assert len(exprs) == 15
    degrees = [sp.Poly(e, *syms).total_degree() for e in exprs]
    assert max(degrees) == 2, degrees


# ==========================================================================
# the endpoint relation
# ==========================================================================
def _endpoint_residuals(d, src):
    """max |endpoint(X_i, D) - X_j| and max |endpoint - Forster| over the dataset."""
    worst_xj = worst_forster = worst_time = 0.0
    for k in range(d["n"]):
        X_i = gal(d["R_i"][k], d["v_i"][k], d["p_i"][k], 0.0)
        D = gal(src["dR"][k], src["dv"][k], src["dp"][k], d["Dt"])
        lhs = gal3.endpoint(X_i, D, d["g"])
        X_j = gal(src["R_j"][k], src["v_j"][k], src["p_j"][k], 0.0)
        worst_xj = max(worst_xj, float(np.abs(lhs - X_j).max()))
        R_f, v_f, p_f = gal3.forster_endpoint(
            d["R_i"][k], d["v_i"][k], d["p_i"][k],
            src["dR"][k], src["dv"][k], src["dp"][k], d["Dt"], d["g"])
        worst_forster = max(worst_forster,
                            float(np.abs(lhs - gal(R_f, v_f, p_f, 0.0)).max()))
        worst_time = max(worst_time, abs(float(lhs[3, 4])))
    return worst_xj, worst_forster, worst_time


@pytest.mark.claim("M1-C31")
def test_endpoint_relation(endpoint_dataset):
    """X_j = T(-Dt) H' X_i D_ij, exactly, on all 25 entries."""
    d = endpoint_dataset

    # ACCURACY: against the continuous truth -- X_j and D_ij from two separate
    # DOP853 solves (rtol = atol = 1e-13) with time-varying inputs.
    worst_ode, worst_alg, worst_time = _endpoint_residuals(d, d)
    assert worst_ode <= 1e-9, worst_ode
    assert worst_alg <= 1e-12, worst_alg
    assert worst_time <= 1e-12, worst_time

    # IDENTITY CHECK (not an accuracy check): the relation is an exact algebraic
    # identity of the project's discrete recursion, for any dt.
    worst_id, _, worst_time_id = _endpoint_residuals(d, d["disc"])
    assert worst_id <= 1e-12, worst_id
    assert worst_time_id <= 1e-12, worst_time_id

    # and that recursion really is a first-order discretisation of the ODE
    assert_discrete_converges(d)


@pytest.mark.claim("M1-C32")
def test_gravity_element_is_one_exponential():
    """T(-Dt) H' = G(Dt) = exp(Dt (0, g, 0, -1)); the opposite fusion order differs."""
    r = rng(32)
    worst_fuse = worst_exp = 0.0
    worst_other = 0.0
    for _ in range(200):
        Dt = float(r.uniform(0.05, 1.0))
        G_Dt = gal3.gravity_element(Dt, G)
        fused = compose(gal3.time_shift(-Dt), gal3.gravity_lift(Dt, G))
        worst_fuse = max(worst_fuse, float(np.abs(G_Dt - fused).max()))
        worst_exp = max(worst_exp,
                        float(np.abs(G_Dt - exp(Dt * gal3.gravity_generator(G))).max()))
        other = compose(gal3.gravity_lift(Dt, G), gal3.time_shift(-Dt))
        expect = gal(I3, G * Dt, -1.5 * G * Dt ** 2, -Dt)
        worst_other = max(worst_other, float(np.abs(other - expect).max()))
        assert float(np.abs(other - G_Dt).max()) >= 1e-6
    assert worst_fuse <= 1e-12, worst_fuse
    assert worst_exp <= 1e-10, worst_exp
    assert worst_other <= 1e-12, worst_other


@pytest.mark.claim("M1-C33")
def test_gravity_one_parameter_subgroup():
    """G(s1) G(s2) = G(s1 + s2)."""
    r = rng(33)
    worst = 0.0
    for _ in range(500):
        s1, s2 = 2.0 * r.normal(), 2.0 * r.normal()
        lhs = compose(gal3.gravity_element(s1, G), gal3.gravity_element(s2, G))
        worst = max(worst, float(np.abs(lhs - gal3.gravity_element(s1 + s2, G)).max()))
    assert worst <= 1e-10, worst


@pytest.mark.claim("M1-C34")
def test_uncorrected_endpoint_is_wrong_twice(endpoint_dataset):
    """H X_i D_ij has time block Dt (not 0) and position p_j + g Dt^2 (not p_j)."""
    d = endpoint_dataset
    Dt = d["Dt"]

    def residuals(src):
        worst_t = worst_p = worst_rv = 0.0
        least_wrong = np.inf
        for k in range(d["n"]):
            X_i = gal(d["R_i"][k], d["v_i"][k], d["p_i"][k], 0.0)
            D = gal(src["dR"][k], src["dv"][k], src["dp"][k], Dt)
            rhs = gal3.endpoint_uncorrected(X_i, D, d["g"])
            worst_t = max(worst_t, abs(float(rhs[3, 4]) - Dt))
            worst_p = max(worst_p, float(np.abs(
                rhs[0:3, 4] - (src["p_j"][k] + d["g"] * Dt ** 2)).max()))
            worst_rv = max(worst_rv, float(np.abs(rhs[0:3, 0:3] - src["R_j"][k]).max()),
                           float(np.abs(rhs[0:3, 3] - src["v_j"][k]).max()))
            # and it really is wrong: the position error is O(g Dt^2)
            least_wrong = min(least_wrong,
                              float(np.abs(rhs[0:3, 4] - src["p_j"][k]).max()))
        return worst_t, worst_p, worst_rv, least_wrong

    # ACCURACY: against the DOP853 continuous truth
    worst_t, worst_p, worst_rv, least_wrong = residuals(d)
    assert worst_t <= 1e-12, worst_t
    assert worst_p <= 1e-9, worst_p
    assert worst_rv <= 1e-9, worst_rv
    assert least_wrong >= 1e-3, least_wrong

    # IDENTITY CHECK on the discrete recursion (not an accuracy check)
    worst_t, worst_p, worst_rv, _ = residuals(d["disc"])
    assert worst_t <= 1e-12, worst_t
    assert worst_p <= 1e-12, worst_p
    assert worst_rv <= 1e-12, worst_rv


@pytest.mark.claim("M1-C35")
def test_right_time_shift_corrupts_position(algebraic_instances):
    """H' X_i D_ij T(-Dt) clears time but shifts position by exactly -v_j Dt."""
    worst_p = worst_t = 0.0
    for X_i, D, Dt in algebraic_instances[:500]:
        rhs = gal3.endpoint_right_time_shift(X_i, D, G)
        X_j = gal3.endpoint(X_i, D, G)
        v_j, p_j = X_j[0:3, 3], X_j[0:3, 4]
        worst_p = max(worst_p, float(np.abs(rhs[0:3, 4] - (p_j - v_j * Dt)).max()))
        worst_t = max(worst_t, abs(float(rhs[3, 4])))
    assert worst_p <= 1e-10, worst_p
    assert worst_t <= 1e-12, worst_t


@pytest.mark.claim("M1-C36")
def test_endpoint_only_valid_on_zero_time_slice(algebraic_instances):
    """For t_i != 0 the position is off by g Dt t_i and the time block is t_i."""
    r = rng(36)
    worst_p = worst_t = 0.0
    for X_i, D, Dt in algebraic_instances[:500]:
        t_i = 2.0 * r.normal()
        X_t = X_i.copy()
        X_t[3, 4] = t_i
        rhs = gal3.endpoint(X_t, D, G, check_time=False)
        X_j = gal3.endpoint(X_i, D, G)
        worst_p = max(worst_p,
                      float(np.abs(rhs[0:3, 4] - (X_j[0:3, 4] + G * Dt * t_i)).max()))
        worst_t = max(worst_t, abs(float(rhs[3, 4]) - t_i))
        # R and v are unaffected
        assert float(np.abs(rhs[0:3, 0:3] - X_j[0:3, 0:3]).max()) <= 1e-12
        assert float(np.abs(rhs[0:3, 3] - X_j[0:3, 3]).max()) <= 1e-12
    assert worst_p <= 1e-10, worst_p
    assert worst_t <= 1e-12, worst_t

    # the implementation must refuse a keyframe with t_i != 0 rather than go quiet
    X_bad = algebraic_instances[0][0].copy()
    X_bad[3, 4] = 0.25
    with pytest.raises(ValueError):
        gal3.endpoint(X_bad, algebraic_instances[0][1], G)


# ==========================================================================
# the NavState (SE_2(3)) side
# ==========================================================================
@pytest.mark.claim("M1-C38")
def test_two_endpoint_forms_agree(algebraic_instances):
    """W_g = T(-Dt) H' T(Dt), T(Dt) U_g = D_ij, and the two forms agree."""
    r = rng(38)
    worst_w = worst_u = 0.0
    for _ in range(500):
        Dt = float(r.uniform(0.05, 1.0))
        W_g = gal3.lift_navstate(
            gal3.navstate(I3, 0.5 * G * Dt ** 2, G * Dt))
        conj = compose(compose(gal3.time_shift(-Dt), gal3.gravity_lift(Dt, G)),
                       gal3.time_shift(Dt))
        worst_w = max(worst_w, float(np.abs(conj - W_g).max()))
        dR = rand_rot(r, scale=0.4)
        dv, dp = r.normal(size=3), r.normal(size=3)
        U_g = gal3.lift_navstate(gal3.navstate(dR, dp, dv))
        D = gal(dR, dv, dp, Dt)
        worst_u = max(worst_u, float(np.abs(compose(gal3.time_shift(Dt), U_g) - D).max()))
    assert worst_w <= 1e-12, worst_w
    assert worst_u <= 1e-12, worst_u

    worst_form = 0.0
    for X_i, D, Dt in algebraic_instances[:200]:
        dR, dv, dp, _ = unpack(D)
        form_a = gal3.endpoint(X_i, D, G)
        W_g = gal3.lift_navstate(gal3.navstate(I3, 0.5 * G * Dt ** 2, G * Dt))
        U_g = gal3.lift_navstate(gal3.navstate(dR, dp, dv))
        phi_Xi = compose(compose(gal3.time_shift(-Dt), X_i), gal3.time_shift(Dt))
        form_b = compose(compose(W_g, phi_Xi), U_g)
        worst_form = max(worst_form, float(np.abs(form_a[0:3, 0:5] - form_b[0:3, 0:5]).max()))
        assert abs(float(form_a[3, 4])) <= 1e-12
        assert abs(float(form_b[3, 4])) <= 1e-12
    assert worst_form <= 1e-9, worst_form


@pytest.mark.claim("M1-C39")
def test_gtsam_navstate_compose_has_no_time_coupling():
    """gtsam.NavState composition is (R1R2, p1 + R1 p2, v1 + R1 v2) -- SE_2(3)."""
    import gtsam

    r = rng(39)
    worst = 0.0
    for _ in range(2000):
        R1, R2 = rand_rot(r), rand_rot(r)
        p1, v1, p2, v2 = (r.normal(size=3) for _ in range(4))
        N1 = gtsam.NavState(gtsam.Rot3(R1), p1, v1)
        N2 = gtsam.NavState(gtsam.Rot3(R2), p2, v2)
        Nc = N1.compose(N2)
        worst = max(worst, float(np.abs(Nc.attitude().matrix() - R1 @ R2).max()))
        worst = max(worst, float(np.abs(Nc.position() - (p1 + R1 @ p2)).max()))
        worst = max(worst, float(np.abs(Nc.velocity() - (v1 + R1 @ v2)).max()))
        worst = max(worst, float(np.abs(Nc.matrix()
                                        - gal3.navstate_compose(gal3.navstate(R1, p1, v1),
                                                                gal3.navstate(R2, p2, v2))).max()))
    assert worst <= 1e-12, worst


@pytest.mark.claim("M1-C40")
def test_phi_dt_is_an_automorphism_not_a_translation():
    """Phi_Dt is no fixed translation, is a NavState automorphism, and is conjugation in Gal(3)."""
    import gtsam

    r = rng(40)
    Dt = 0.5

    def phi(N):
        return gtsam.NavState(N.attitude(), N.position() + N.velocity() * Dt, N.velocity())

    # (i) no fixed translation.  Phi_Dt fixes the identity, so if Phi_Dt(X) = N X
    #     (or X N) for all X then N = Phi_Dt(I) = I -- and Phi_Dt(X) - X is exactly
    #     v Dt in position.  (A factor FITTED on one element, N_L = Phi(X) X^-1, is
    #     exact at X and need not fail on a nearby Y by any fixed amount, so no such
    #     lower bound is asserted.)
    I_nav = gtsam.NavState()
    assert np.array_equal(phi(I_nav).matrix(), I_nav.matrix())
    worst_shift = 0.0
    max_gap = 0.0
    for _ in range(2000):
        R, p, v = rand_rot(r), r.normal(size=3), r.normal(size=3)
        X = gtsam.NavState(gtsam.Rot3(R), p, v)
        Y = phi(X)
        worst_shift = max(worst_shift,
                          float(np.abs(Y.attitude().matrix() - R).max()),
                          float(np.abs(Y.velocity() - v).max()),
                          float(np.abs(Y.position() - (p + v * Dt)).max()))
        # the only candidate factor (the identity) misses by |v| Dt
        gap = float(np.abs(I_nav.compose(X).matrix() - Y.matrix()).max())
        assert abs(gap - float(np.abs(v).max()) * Dt) <= 1e-12, gap
        max_gap = max(max_gap, gap)
    assert worst_shift <= 1e-12, worst_shift
    assert max_gap >= 0.5, max_gap
    # the fitted factor is exact at its own fitting point (so "fails on a second
    # element by > 0.5" is NOT a property of Phi_Dt)
    X = gtsam.NavState(gtsam.Rot3(rand_rot(r)), r.normal(size=3), r.normal(size=3))
    N_L = phi(X).compose(X.inverse())
    assert float(np.abs(N_L.compose(X).matrix() - phi(X).matrix()).max()) <= 1e-12

    # (ii) but it IS an automorphism
    worst = 0.0
    for _ in range(2000):
        N1 = gtsam.NavState(gtsam.Rot3(rand_rot(r)), r.normal(size=3), r.normal(size=3))
        N2 = gtsam.NavState(gtsam.Rot3(rand_rot(r)), r.normal(size=3), r.normal(size=3))
        worst = max(worst, float(np.abs(phi(N1).compose(phi(N2)).matrix()
                                        - phi(N1.compose(N2)).matrix()).max()))
    assert worst <= 1e-12, worst

    # (iii) in Gal(3) it is conjugation by T(Dt), restricted to the t = 0 slice
    worst_c = 0.0
    for _ in range(2000):
        R, v, p = rand_rot(r), r.normal(size=3), r.normal(size=3)
        X_g = gal(R, v, p, 0.0)
        conj = compose(compose(gal3.time_shift(-Dt), X_g), gal3.time_shift(Dt))
        worst_c = max(worst_c, float(np.abs(conj - gal(R, v, p + v * Dt, 0.0)).max()))
    assert worst_c <= 1e-12, worst_c


# ==========================================================================
# gtsam 4.3.0 cross-validation
# ==========================================================================
@pytest.mark.claim("M1-C46")
def test_gtsam_gal3_matrix_layout():
    """gtsam.Gal3.matrix() is [R | v | p] with time at (3,4)."""
    import gtsam

    r = rng(46)
    worst = 0.0
    for _ in range(2000):
        R, v, p, t = rand_rot(r), r.normal(size=3), 3.0 * r.normal(size=3), r.normal()
        M = gtsam.Gal3(gtsam.Rot3(R), p, v, t).matrix()
        worst = max(worst, float(np.abs(M - gal(R, v, p, t)).max()))
        assert M[3, 3] == 1.0
        assert np.array_equal(M[4, :], np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert worst <= 1e-12, worst

    # column 3 is velocity, NOT position
    R = rand_rot(r)
    p = np.array([10.0, 20.0, 30.0])
    v = np.array([1.0, 2.0, 3.0])
    M = gtsam.Gal3(gtsam.Rot3(R), p, v, 0.7).matrix()
    assert float(np.abs(M[0:3, 3] - p).max()) >= 1e-3
    assert float(np.abs(M[0:3, 3] - v).max()) <= 1e-12
    assert float(np.abs(M[0:3, 4] - p).max()) <= 1e-12


@pytest.mark.claim("M1-C47")
def test_gtsam_gal3_constructor_order():
    """Gal3(R, r, v, t) takes POSITION before velocity; accessors agree."""
    import gtsam

    r = rng(47)
    worst = 0.0
    margin = np.inf
    for _ in range(2000):
        R = rand_rot(r)
        p = 3.0 * r.normal(size=3) + 10.0
        v = r.normal(size=3)
        t = float(r.normal())
        X = gtsam.Gal3(gtsam.Rot3(R), p, v, t)
        worst = max(worst, float(np.abs(X.position() - p).max()))
        worst = max(worst, float(np.abs(X.translation() - p).max()))
        worst = max(worst, float(np.abs(X.velocity() - v).max()))
        worst = max(worst, abs(float(X.time()) - t))
        worst = max(worst, float(np.abs(X.rotation().matrix() - R).max()))
        worst = max(worst, float(np.abs(X.attitude().matrix() - R).max()))
        margin = min(margin, float(np.abs(X.position() - v).max()))
    assert worst <= 1e-12, worst
    assert margin >= 1e-3, margin


@pytest.mark.claim("M1-C48")
def test_gtsam_gal3_tangent_order():
    """Gal3.Hat == wedge bit-for-bit; tangent is (omega, nu, rho, alpha)."""
    import gtsam

    r = rng(48)
    worst = 0.0
    for _ in range(2000):
        xi = rand_xi(r)
        worst = max(worst, float(np.abs(gtsam.Gal3.Hat(xi) - wedge(xi)).max()))
        worst = max(worst, float(np.abs(gtsam.Gal3.Vee(wedge(xi)) - xi).max()))
    assert worst <= 1e-14, worst

    e = np.eye(gal3.DIM)
    assert np.array_equal(gtsam.Gal3.Hat(e[3])[0:3, 3], np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(gtsam.Gal3.Hat(e[6])[0:3, 4], np.array([1.0, 0.0, 0.0]))
    X = gtsam.Gal3.Expmap(1e-4 * e[3])
    assert float(np.abs(X.velocity() - 1e-4 * np.array([1.0, 0.0, 0.0])).max()) <= 1e-12
    assert float(np.abs(X.position()).max()) <= 1e-12


@pytest.mark.claim("M1-C49")
def test_gtsam_navstate_opposite_ordering():
    """NavState.matrix() is [R | p | v] and its tangent is (rot, pos, vel)."""
    import gtsam

    r = rng(49)
    worst = 0.0
    margin = np.inf
    for _ in range(500):
        R = rand_rot(r)
        p = 3.0 * r.normal(size=3) + 10.0
        v = r.normal(size=3)
        M = gtsam.NavState(gtsam.Rot3(R), p, v).matrix()
        worst = max(worst, float(np.abs(M[0:3, 3] - p).max()))
        worst = max(worst, float(np.abs(M[0:3, 4] - v).max()))
        worst = max(worst, float(np.abs(M - gal3.navstate(R, p, v)).max()))
        margin = min(margin, float(np.abs(M[0:3, 3] - v).max()))
    assert worst <= 1e-12, worst
    assert margin >= 1e-3, margin

    e9 = np.eye(9)
    e10 = np.eye(gal3.DIM)
    N = gtsam.NavState().retract(1e-4 * e9[3])
    assert float(np.abs(N.position() - 1e-4 * np.array([1.0, 0.0, 0.0])).max()) <= 1e-12
    assert float(np.abs(N.velocity()).max()) <= 1e-12
    X = gtsam.Gal3.Expmap(1e-4 * e10[3])
    assert float(np.abs(X.velocity() - 1e-4 * np.array([1.0, 0.0, 0.0])).max()) <= 1e-12
    assert float(np.abs(X.position()).max()) <= 1e-12
    # and the discriminating witness at a visible scale
    Nb = gtsam.NavState().retract(1e-3 * e9[3])
    Xb = gtsam.Gal3.Expmap(1e-3 * e10[3])
    assert float(np.abs(Nb.position() - Xb.position()).max()) >= 1e-4


@pytest.mark.claim("M1-C50")
def test_gal3_navstate_tangent_permutation():
    """The 9x10 permutation P converts a Gal(3) tangent to NavState ordering."""
    import gtsam

    r = rng(50)
    P = gal3.NAVSTATE_PERM
    assert P.shape == (9, gal3.DIM)
    d_probe = np.arange(1.0, 11.0)
    assert np.array_equal(P @ d_probe, np.array([1., 2., 3., 7., 8., 9., 4., 5., 6.]))

    def permuted_error(scale, n):
        worst = worst_un = 0.0
        for _ in range(n):
            d = scale * r.normal(size=gal3.DIM)
            A = gtsam.Gal3.Expmap(d)
            B = gtsam.NavState.Expmap(P @ d)
            worst = max(worst,
                        float(np.abs(A.rotation().matrix() - B.attitude().matrix()).max()),
                        float(np.abs(A.position() - B.position()).max()),
                        float(np.abs(A.velocity() - B.velocity()).max()))
            C = gtsam.NavState.Expmap(d[0:9])
            worst_un = max(worst_un,
                           float(np.abs(A.position() - C.position()).max()),
                           float(np.abs(A.velocity() - C.velocity()).max()))
        return worst, worst_un

    # inside the stated domain, d ~ 1e-5 N(0, I10): 20000 draws (the worst case
    # grows with the sample count -- 50 draws understated it ~16x)
    worst, worst_un = permuted_error(1e-5, 20000)
    assert worst <= 1e-9, worst
    assert worst_un >= 1e-6, worst_un

    # the agreement is first order only, so the 1e-9 bound does NOT extend to
    # d ~ 1e-4 N(0, I10): the error grows ~100x for a 10x larger d
    worst_big, _ = permuted_error(1e-4, 2000)
    assert worst_big >= 1e-8, worst_big
    assert 30.0 <= worst_big / worst <= 300.0, (worst, worst_big)


@pytest.mark.claim("M1-C51")
def test_closed_forms_match_gtsam():
    """Expmap, Logmap, AdjointMap and adjointMap all agree with gtsam 4.3.0."""
    import gtsam

    r = rng(51)
    w_exp = w_log = w_Ad = w_ad = 0.0
    for _ in range(2000):
        xi = rand_xi_bounded_phi(r, 0.05, 2.9)
        w_exp = max(w_exp, float(np.abs(gtsam.Gal3.Expmap(xi).matrix() - exp(xi)).max()))
        w_ad = max(w_ad, float(np.abs(gtsam.Gal3.adjointMap(xi) - ad(xi)).max()))

        R, v, p, t = (rand_rot(r), r.normal(size=3), 3.0 * r.normal(size=3),
                      float(r.normal()))
        X = gtsam.Gal3(gtsam.Rot3(R), p, v, t)      # POSITION SECOND
        w_Ad = max(w_Ad, float(np.abs(X.AdjointMap() - Ad(gal(R, v, p, t))).max()))

        Rl = _rotvec_to_matrix(r.uniform(0.05, np.pi - 1e-3) * rand_axis(r))
        Xl = gtsam.Gal3(gtsam.Rot3(Rl), p, v, t)
        w_log = max(w_log, float(np.abs(gtsam.Gal3.Logmap(Xl) - log(gal(Rl, v, p, t))).max()))
    assert w_exp <= 1e-9, w_exp
    assert w_log <= 1e-9, w_log
    assert w_Ad <= 1e-10, w_Ad
    assert w_ad <= 1e-12, w_ad


@pytest.mark.claim("M1-C52")
def test_gtsam_gal3_group_operations():
    """compose/__mul__/inverse/between/localCoordinates/retract all match."""
    import gtsam

    r = rng(52)
    w_comp = w_inv = w_bet = w_loc = w_ret = w_wrong = 0.0
    w_loc_edge = 0.0
    n_loc = 0
    for _ in range(2000):
        Ra, va, pa, ta = rand_rot(r), r.normal(size=3), 3.0 * r.normal(size=3), float(r.normal())
        Rb, vb, pb, tb = rand_rot(r), r.normal(size=3), 3.0 * r.normal(size=3), float(r.normal())
        A = gtsam.Gal3(gtsam.Rot3(Ra), pa, va, ta)
        B = gtsam.Gal3(gtsam.Rot3(Rb), pb, vb, tb)
        Am, Bm = gal(Ra, va, pa, ta), gal(Rb, vb, pb, tb)
        w_comp = max(w_comp, float(np.abs(A.compose(B).matrix() - compose(Am, Bm)).max()))
        w_comp = max(w_comp, float(np.abs((A * B).matrix() - compose(Am, Bm)).max()))
        w_inv = max(w_inv, float(np.abs(A.inverse().matrix() - inverse(Am)).max()))
        w_bet = max(w_bet, float(np.abs(A.between(B).matrix()
                                        - compose(inverse(Am), Bm)).max()))
        d = 0.1 * r.normal(size=gal3.DIM)
        w_ret = max(w_ret, float(np.abs(A.retract(d).matrix() - compose(Am, exp(d))).max()))
        # Exp(d) X is the WRONG convention
        w_wrong = max(w_wrong, float(np.abs(A.retract(d).matrix()
                                            - compose(exp(d), Am)).max()))

        # localCoordinates(B) == Logmap(A^-1 B), with Logmap the project's
        # INDEPENDENT closed form log (M1-C21), on M1-C51's Logmap domain (the
        # relative rotation off the |phi| = pi edge).  Comparing against
        # gtsam.Gal3.Logmap(A.between(B)) instead would be gtsam against itself
        # (exactly 0.0 always) and could not fail.
        rel = compose(inverse(Am), Bm)
        if float(np.linalg.norm(gal3.so3_log(rel[0:3, 0:3]))) < np.pi - 1e-3:
            n_loc += 1
            w_loc = max(w_loc, float(np.abs(A.localCoordinates(B) - log(rel)).max()))

    # the domain edge itself: |phi_rel| in (pi - 1e-2, pi - 1.001e-3)
    for _ in range(400):
        Ra, va, pa, ta = rand_rot(r), r.normal(size=3), 3.0 * r.normal(size=3), float(r.normal())
        Rrel = _rotvec_to_matrix((np.pi - r.uniform(1.001e-3, 1e-2)) * rand_axis(r))
        Rb, vb, pb, tb = Ra @ Rrel, r.normal(size=3), 3.0 * r.normal(size=3), float(r.normal())
        A = gtsam.Gal3(gtsam.Rot3(Ra), pa, va, ta)
        B = gtsam.Gal3(gtsam.Rot3(Rb), pb, vb, tb)
        rel = compose(inverse(gal(Ra, va, pa, ta)), gal(Rb, vb, pb, tb))
        w_loc_edge = max(w_loc_edge, float(np.abs(A.localCoordinates(B) - log(rel)).max()))
    assert w_comp <= 1e-10, w_comp
    assert w_inv <= 1e-10, w_inv
    assert w_bet <= 1e-12, w_bet
    assert n_loc >= 1000, n_loc
    assert w_loc <= 1e-12, w_loc
    assert w_loc_edge <= 1e-12, w_loc_edge
    assert w_ret <= 1e-12, w_ret
    assert w_wrong >= 1e-6, w_wrong
