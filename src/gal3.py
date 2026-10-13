"""Gal(3), the Galilean group of 3D space, in the layout fixed for this project.

Group element (5x5)::

    X(R, v, p, t) = [[ R   , v, p ],
                     [ 0_1x3, 1, t ],
                     [ 0_1x3, 0, 1 ]]

column 3 is VELOCITY, column 4 is POSITION, entry (3, 4) is TIME.

Algebra element (10-vector) ``xi = (phi, u, w, alpha)``, block order
(rot, vel, pos, time)::

    xi^ = [[ [phi]x, u, w     ],
           [ 0_1x3 , 0, alpha ],
           [ 0_1x3 , 0, 0     ]]

Everything in this module is plain numpy; the gtsam cross-validation lives in the
tests and the bias convention lives in :mod:`src.imu_dynamics`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DIM",
    "THETA_SMALL",
    "G_DEFAULT",
    "KEYFRAME_TIME_TOL",
    "skew",
    "unskew",
    "so3_exp",
    "so3_log",
    "gal",
    "unpack",
    "identity",
    "compose",
    "inverse",
    "structural_residual",
    "wedge",
    "vee",
    "ad",
    "Ad",
    "Ad_apply",
    "gamma_coefficients",
    "gamma1",
    "gamma2",
    "gamma1_inv",
    "exp",
    "log",
    "cay",
    "cay_matrix",
    "cay_denominator",
    "cayley_constraint_residuals",
    "cayley_constraints_symbolic",
    "time_shift",
    "gravity_lift",
    "gravity_lift_uncorrected",
    "gravity_element",
    "gravity_generator",
    "endpoint",
    "endpoint_uncorrected",
    "endpoint_right_time_shift",
    "forster_endpoint",
    "navstate",
    "navstate_compose",
    "navstate_phi",
    "lift_navstate",
    "NAVSTATE_PERM",
]

#: dimension of gal(3)
DIM = 10

#: below this rotation angle the Gamma coefficients switch to their Taylor branch.
#: Mandatory: the closed forms for kappa3 and c2 lose *all* precision below
#: theta ~ 1e-4 (M1-C24).
THETA_SMALL = 5e-2

#: gravity used by the tests; every formula here takes ``g`` as an argument.
G_DEFAULT = np.array([0.0, 0.0, -9.81])

#: a keyframe state must sit on the t = 0 slice (M1-C31 / M1-C36).
KEYFRAME_TIME_TOL = 1e-12

_I3 = np.eye(3)
_I5 = np.eye(5)


# --------------------------------------------------------------------------
# so(3)
# --------------------------------------------------------------------------
def skew(a) -> np.ndarray:
    """[a]x, the 3x3 skew-symmetric matrix with [a]x b = a x b."""
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
    )


def unskew(A) -> np.ndarray:
    """Inverse of :func:`skew`; reads the three independent entries."""
    A = np.asarray(A, dtype=float)
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def so3_exp(phi) -> np.ndarray:
    """Exp_SO3: Rodrigues' formula, with a Taylor branch at small angle."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < THETA_SMALL:
        t2 = theta * theta
        t4 = t2 * t2
        s = 1.0 - t2 / 6.0 + t4 / 120.0          # sin(th)/th
        c = 0.5 - t2 / 24.0 + t4 / 720.0         # (1 - cos th)/th^2
    else:
        s = np.sin(theta) / theta
        c = 2.0 * np.sin(0.5 * theta) ** 2 / (theta * theta)
    return _I3 + s * K + c * (K @ K)


def so3_log(R) -> np.ndarray:
    """Log_SO3 returning the principal branch, ||phi|| in [0, pi].

    Goes through the unit quaternion (Shepperd's method) so that the result stays
    accurate all the way to theta = pi, where (R - R^T) carries no information.
    """
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qv = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qv = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qv = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qv = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    if qw < 0.0:                      # pick the hemisphere with theta <= pi
        qw, qv = -qw, -qv
    n = float(np.linalg.norm(qv))
    if n < 1e-10:
        # theta ~ 0: 2*atan2(n, qw)/n -> 2/qw with relative error O(n^2).
        return 2.0 * qv / qw
    return (2.0 * np.arctan2(n, qw) / n) * qv


# --------------------------------------------------------------------------
# the group
# --------------------------------------------------------------------------
def gal(R, v, p, t) -> np.ndarray:
    """Assemble X(R, v, p, t). Velocity in column 3, position in column 4."""
    X = np.eye(5)
    X[0:3, 0:3] = np.asarray(R, dtype=float)
    X[0:3, 3] = np.asarray(v, dtype=float).reshape(3)
    X[0:3, 4] = np.asarray(p, dtype=float).reshape(3)
    X[3, 4] = float(t)
    return X


def unpack(X):
    """(R, v, p, t) from a 5x5 Gal(3) element."""
    X = np.asarray(X, dtype=float)
    return X[0:3, 0:3].copy(), X[0:3, 3].copy(), X[0:3, 4].copy(), float(X[3, 4])


def identity() -> np.ndarray:
    return np.eye(5)


def compose(X1, X2) -> np.ndarray:
    """The Gal(3) product, in closed form (M1-C01).

    (R1 R2, v1 + R1 v2, p1 + R1 p2 + v1 * t2, t1 + t2).  Implemented from the
    block formula rather than as a 5x5 matmul so that the associativity and
    homomorphism tests exercise this code and not numpy's.
    """
    R1, v1, p1, t1 = unpack(X1)
    R2, v2, p2, t2 = unpack(X2)
    return gal(R1 @ R2, v1 + R1 @ v2, p1 + R1 @ p2 + v1 * t2, t1 + t2)


def inverse(X) -> np.ndarray:
    """X^-1 = (R^T, -R^T v, -R^T (p - t v), -t) (M1-C05)."""
    R, v, p, t = unpack(X)
    Rt = R.T
    return gal(Rt, -Rt @ v, -Rt @ (p - t * v), -t)


def structural_residual(X) -> float:
    """max violation of the Gal(3) structure: SO(3) block, zero rows, unit diagonal."""
    X = np.asarray(X, dtype=float)
    R = X[0:3, 0:3]
    res = [
        np.abs(R.T @ R - _I3).max(),
        abs(float(np.linalg.det(R)) - 1.0),
        np.abs(X[3, 0:3]).max(),
        np.abs(X[4, 0:4]).max(),
        abs(X[3, 3] - 1.0),
        abs(X[4, 4] - 1.0),
    ]
    return float(max(res))


# --------------------------------------------------------------------------
# the algebra
# --------------------------------------------------------------------------
def _split(xi):
    xi = np.asarray(xi, dtype=float).reshape(DIM)
    return xi[0:3], xi[3:6], xi[6:9], float(xi[9])


def wedge(xi) -> np.ndarray:
    """xi = (phi, u, w, alpha) -> xi^ in gal(3). u is column 3, w is column 4."""
    phi, u, w, alpha = _split(xi)
    A = np.zeros((5, 5))
    A[0:3, 0:3] = skew(phi)
    A[0:3, 3] = u
    A[0:3, 4] = w
    A[3, 4] = alpha
    return A


def vee(A) -> np.ndarray:
    """Inverse of :func:`wedge`."""
    A = np.asarray(A, dtype=float)
    return np.concatenate([unskew(A[0:3, 0:3]), A[0:3, 3], A[0:3, 4], [A[3, 4]]])


def ad(xi) -> np.ndarray:
    """ad_xi as a 10x10 matrix (M1-C09)."""
    phi, u, w, alpha = _split(xi)
    P = skew(phi)
    A = np.zeros((DIM, DIM))
    A[0:3, 0:3] = P
    A[3:6, 0:3] = skew(u)
    A[3:6, 3:6] = P
    A[6:9, 0:3] = skew(w)
    A[6:9, 3:6] = -alpha * _I3
    A[6:9, 6:9] = P
    A[6:9, 9] = u
    return A


def Ad(X) -> np.ndarray:
    """Ad_X as a 10x10 matrix, from the closed form of M1-C11."""
    R, v, p, t = unpack(X)
    A = np.zeros((DIM, DIM))
    A[0:3, 0:3] = R
    A[3:6, 0:3] = skew(v) @ R
    A[3:6, 3:6] = R
    A[6:9, 0:3] = skew(p - t * v) @ R
    A[6:9, 3:6] = -t * R
    A[6:9, 6:9] = R
    A[6:9, 9] = v
    A[9, 9] = 1.0
    return A


def Ad_apply(X, xi) -> np.ndarray:
    """Ad_X(xi) componentwise (M1-C11)."""
    R, v, p, t = unpack(X)
    phi, u, w, alpha = _split(xi)
    Rphi = R @ phi
    return np.concatenate(
        [
            Rphi,
            R @ u + skew(v) @ Rphi,
            R @ w - t * (R @ u) + skew(p - t * v) @ Rphi + alpha * v,
            [alpha],
        ]
    )


# --------------------------------------------------------------------------
# Gamma coefficients  (M1-C18, M1-C21, M1-C24)
# --------------------------------------------------------------------------
def gamma_coefficients(theta: float):
    """(kappa1, kappa2, kappa3, c2) at angle ``theta``.

    Above :data:`THETA_SMALL` the closed forms of M1-C18/M1-C21 are used; below it
    the mandatory 3-term Taylor branch of M1-C24.  Without that branch kappa3 and
    c2 lose all precision below theta ~ 1e-4.

    In the closed-form branch ``1 - cos(theta)`` is evaluated as the identically
    equal ``2 sin(theta/2)^2``.  Same expression, but it keeps the subtractive
    cancellation out of kappa3 and c2 just above the switch point, where
    ``1 - cos(theta)`` would already cost ~5 digits.

    c2 is deliberately the form ``(1 - theta sin(theta) / (2 (1 - cos theta))) / theta^2``
    and NOT the algebraically identical ``1/theta^2 - (1+cos theta)/(2 theta sin theta)``,
    whose denominator vanishes at theta = pi -- exactly the edge of the chart.
    """
    theta = float(theta)
    if theta < 0.0:
        raise ValueError("theta must be non-negative")
    if theta < THETA_SMALL:
        t2 = theta * theta
        t4 = t2 * t2
        kappa1 = 0.5 - t2 / 24.0 + t4 / 720.0
        kappa2 = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0
        kappa3 = 1.0 / 24.0 - t2 / 720.0 + t4 / 40320.0
        c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0
        return kappa1, kappa2, kappa3, c2
    t2 = theta * theta
    s = np.sin(theta)
    omc = 2.0 * np.sin(0.5 * theta) ** 2            # == 1 - cos(theta)
    kappa1 = omc / t2
    kappa2 = (theta - s) / (t2 * theta)
    kappa3 = (t2 - 2.0 * omc) / (2.0 * t2 * t2)     # == (th^2 + 2 cos th - 2)/(2 th^4)
    c2 = (1.0 - theta * s / (2.0 * omc)) / t2
    return float(kappa1), float(kappa2), float(kappa3), float(c2)


def gamma1(phi) -> np.ndarray:
    """Gamma1(phi) = I + kappa1 [phi]x + kappa2 [phi]x^2 = sum_n [phi]x^n/(n+1)!."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    P = skew(phi)
    k1, k2, _, _ = gamma_coefficients(float(np.linalg.norm(phi)))
    return _I3 + k1 * P + k2 * (P @ P)


def gamma2(phi) -> np.ndarray:
    """Gamma2(phi) = I/2 + kappa2 [phi]x + kappa3 [phi]x^2 = sum_n [phi]x^n/(n+2)!."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    P = skew(phi)
    _, k2, k3, _ = gamma_coefficients(float(np.linalg.norm(phi)))
    return 0.5 * _I3 + k2 * P + k3 * (P @ P)


def gamma1_inv(phi) -> np.ndarray:
    """Gamma1(phi)^-1 = I - [phi]x/2 + c2 [phi]x^2."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    P = skew(phi)
    _, _, _, c2 = gamma_coefficients(float(np.linalg.norm(phi)))
    return _I3 - 0.5 * P + c2 * (P @ P)


# --------------------------------------------------------------------------
# exp / log
# --------------------------------------------------------------------------
def exp(xi) -> np.ndarray:
    """exp(phi,u,w,alpha) = (Exp(phi), G1 u, G1 w + alpha G2 u, alpha) (M1-C19)."""
    phi, u, w, alpha = _split(xi)
    G1 = gamma1(phi)
    G2 = gamma2(phi)
    return gal(so3_exp(phi), G1 @ u, G1 @ w + alpha * (G2 @ u), alpha)


def log(X) -> np.ndarray:
    """log(R,v,p,t) = (Log R, G1^-1 v, G1^-1 (p - t G2 u), t) (M1-C21)."""
    R, v, p, t = unpack(X)
    phi = so3_log(R)
    G1i = gamma1_inv(phi)
    G2 = gamma2(phi)
    u = G1i @ v
    w = G1i @ (p - t * (G2 @ u))
    return np.concatenate([phi, u, w, [t]])


# --------------------------------------------------------------------------
# Cayley
# --------------------------------------------------------------------------
def cay(xi) -> np.ndarray:
    """Cay(xi) in closed form (M1-C25).

    (S^-1 (I + [phi]x/2), S^-1 u, S^-1 (w + (alpha/2) u), alpha), S = I - [phi]x/2.
    The ``+ (alpha/2) u`` coupling in the position block is a PLUS; it vanishes at
    alpha = 0, so an alpha = 0 test does not exercise it.
    """
    phi, u, w, alpha = _split(xi)
    P = skew(phi)
    S = _I3 - 0.5 * P
    rhs = np.empty((3, 5))
    rhs[:, 0:3] = _I3 + 0.5 * P
    rhs[:, 3] = u
    rhs[:, 4] = w + 0.5 * alpha * u
    sol = np.linalg.solve(S, rhs)
    return gal(sol[:, 0:3], sol[:, 3], sol[:, 4], alpha)


def cay_matrix(xi) -> np.ndarray:
    """Cay(xi) by the defining 5x5 expression (I5 - xi^/2)^-1 (I5 + xi^/2)."""
    A = wedge(xi)
    return np.linalg.solve(_I5 - 0.5 * A, _I5 + 0.5 * A)


def cay_denominator(xi) -> np.ndarray:
    """I5 - xi^/2, the sole denominator of the Cayley map (M1-C26)."""
    return _I5 - 0.5 * wedge(xi)


def cayley_constraint_residuals(phi, u, w, alpha, Re, ve, pe):
    """The three degree-<=2 Cayley constraint blocks of M1-C30, as residuals.

    (I - [phi]x/2) Re - (I + [phi]x/2),  (I - [phi]x/2) ve - u,
    (I - [phi]x/2) pe - (w + (alpha/2) u).
    """
    P = skew(phi)
    S = _I3 - 0.5 * P
    u = np.asarray(u, dtype=float).reshape(3)
    w = np.asarray(w, dtype=float).reshape(3)
    return (
        S @ np.asarray(Re, dtype=float) - (_I3 + 0.5 * P),
        S @ np.asarray(ve, dtype=float).reshape(3) - u,
        S @ np.asarray(pe, dtype=float).reshape(3) - (w + 0.5 * float(alpha) * u),
    )


def cayley_constraints_symbolic():
    """The 15 scalar Cayley equalities as sympy expressions in the lifted variables.

    Returns ``(exprs, symbols)``; ``exprs`` are the residuals (= 0) and ``symbols``
    are the independently lifted (phi, u, w, alpha, Re, ve, pe).
    """
    import sympy as sp

    phi = sp.symbols("phi0:3", real=True)
    u = sp.symbols("u0:3", real=True)
    w = sp.symbols("w0:3", real=True)
    alpha = sp.Symbol("alpha", real=True)
    Re = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"Re{i}{j}", real=True))
    ve = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"ve{i}", real=True))
    pe = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"pe{i}", real=True))

    P = sp.Matrix([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    S = sp.eye(3) - P / 2
    uv = sp.Matrix(3, 1, lambda i, j: u[i])
    wv = sp.Matrix(3, 1, lambda i, j: w[i])

    E1 = S * Re - (sp.eye(3) + P / 2)
    E2 = S * ve - uv
    E3 = S * pe - (wv + alpha / 2 * uv)

    exprs = [sp.expand(e) for e in list(E1) + list(E2) + list(E3)]
    syms = (
        list(phi)
        + list(u)
        + list(w)
        + [alpha]
        + list(Re)
        + list(ve)
        + list(pe)
    )
    return exprs, syms


# --------------------------------------------------------------------------
# gravity, time shifts, and the endpoint relation
# --------------------------------------------------------------------------
def time_shift(dt: float) -> np.ndarray:
    """T(dt) = (I, 0, 0, dt)."""
    return gal(_I3, np.zeros(3), np.zeros(3), dt)


def gravity_lift(Dt: float, g=G_DEFAULT) -> np.ndarray:
    """H'(Dt) = (I, g Dt, -0.5 g Dt^2, 0) -- position block NEGATED (M1-C31)."""
    g = np.asarray(g, dtype=float).reshape(3)
    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)


def gravity_lift_uncorrected(Dt: float, g=G_DEFAULT) -> np.ndarray:
    """H(Dt) = (I, g Dt, +0.5 g Dt^2, 0) -- the form M1-C34 shows to be wrong."""
    g = np.asarray(g, dtype=float).reshape(3)
    return gal(_I3, g * Dt, 0.5 * g * Dt * Dt, 0.0)


def gravity_element(Dt: float, g=G_DEFAULT) -> np.ndarray:
    """G(Dt) = (I, g Dt, -0.5 g Dt^2, -Dt) = T(-Dt) H'(Dt) (M1-C32)."""
    g = np.asarray(g, dtype=float).reshape(3)
    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, -Dt)


def gravity_generator(g=G_DEFAULT) -> np.ndarray:
    """zeta_g = (0, g, 0, -1) in gal(3); G(Dt) = exp(Dt zeta_g) (M1-C32)."""
    g = np.asarray(g, dtype=float).reshape(3)
    return np.concatenate([np.zeros(3), g, np.zeros(3), [-1.0]])


def _check_keyframe_time(X_i) -> float:
    t_i = float(np.asarray(X_i)[3, 4])
    if abs(t_i) > KEYFRAME_TIME_TOL:
        raise ValueError(
            "keyframe state must lie on the t = 0 slice: got t_i = "
            f"{t_i!r}. The endpoint relation X_j = T(-Dt) H' X_i D_ij is valid only "
            "there; for t_i != 0 the position block is off by exactly g*Dt*t_i and "
            "the time block comes out as t_i rather than 0 (M1-C36)."
        )
    return t_i


def endpoint(X_i, D_ij, g=G_DEFAULT, check_time: bool = True) -> np.ndarray:
    """X_j = T(-Dt) . H'(Dt) . X_i . D_ij (M1-C31).

    ``X_i`` MUST carry t_i = 0; pass ``check_time=False`` only to demonstrate the
    failure mode of M1-C36.
    """
    if check_time:
        _check_keyframe_time(X_i)
    Dt = float(np.asarray(D_ij)[3, 4])
    left = compose(time_shift(-Dt), gravity_lift(Dt, g))
    return compose(compose(left, X_i), D_ij)


def endpoint_uncorrected(X_i, D_ij, g=G_DEFAULT) -> np.ndarray:
    """H(Dt) . X_i . D_ij -- wrong in the time and position blocks (M1-C34)."""
    Dt = float(np.asarray(D_ij)[3, 4])
    return compose(compose(gravity_lift_uncorrected(Dt, g), X_i), D_ij)


def endpoint_right_time_shift(X_i, D_ij, g=G_DEFAULT) -> np.ndarray:
    """H'(Dt) . X_i . D_ij . T(-Dt) -- clears time, corrupts position (M1-C35)."""
    Dt = float(np.asarray(D_ij)[3, 4])
    inner = compose(compose(gravity_lift(Dt, g), X_i), D_ij)
    return compose(inner, time_shift(-Dt))


def forster_endpoint(R_i, v_i, p_i, dR, dv, dp, Dt, g=G_DEFAULT):
    """The Forster component equations (M1-C43)."""
    g = np.asarray(g, dtype=float).reshape(3)
    R_i = np.asarray(R_i, dtype=float)
    v_i = np.asarray(v_i, dtype=float).reshape(3)
    p_i = np.asarray(p_i, dtype=float).reshape(3)
    R_j = R_i @ np.asarray(dR, dtype=float)
    v_j = v_i + g * Dt + R_i @ np.asarray(dv, dtype=float).reshape(3)
    p_j = p_i + v_i * Dt + 0.5 * g * Dt * Dt + R_i @ np.asarray(dp, dtype=float).reshape(3)
    return R_j, v_j, p_j


# --------------------------------------------------------------------------
# NavState (SE_2(3)) side -- pure numpy mirrors of the gtsam objects
# --------------------------------------------------------------------------
def navstate(R, p, v) -> np.ndarray:
    """NavState as a 5x5: [R | p | v] -- POSITION in column 3, velocity in 4."""
    N = np.eye(5)
    N[0:3, 0:3] = np.asarray(R, dtype=float)
    N[0:3, 3] = np.asarray(p, dtype=float).reshape(3)
    N[0:3, 4] = np.asarray(v, dtype=float).reshape(3)
    return N


def navstate_compose(N1, N2) -> np.ndarray:
    """(R1 R2, p1 + R1 p2, v1 + R1 v2) -- no v1*Dt coupling (M1-C39)."""
    N1 = np.asarray(N1, dtype=float)
    N2 = np.asarray(N2, dtype=float)
    R1 = N1[0:3, 0:3]
    return navstate(
        R1 @ N2[0:3, 0:3],
        N1[0:3, 3] + R1 @ N2[0:3, 3],
        N1[0:3, 4] + R1 @ N2[0:3, 4],
    )


def navstate_phi(N, Dt: float) -> np.ndarray:
    """Phi_Dt(R, p, v) = (R, p + v Dt, v) (M1-C40)."""
    N = np.asarray(N, dtype=float)
    return navstate(N[0:3, 0:3], N[0:3, 3] + N[0:3, 4] * Dt, N[0:3, 4])


def lift_navstate(N) -> np.ndarray:
    """Lift a NavState into Gal(3) on the t = 0 slice."""
    N = np.asarray(N, dtype=float)
    return gal(N[0:3, 0:3], N[0:3, 4], N[0:3, 3], 0.0)


def _navstate_perm() -> np.ndarray:
    P = np.zeros((9, DIM))
    P[0:3, 0:3] = _I3      # rot  <- rot
    P[3:6, 6:9] = _I3      # pos  <- pos
    P[6:9, 3:6] = _I3      # vel  <- vel
    return P


#: 9x10 permutation taking a Gal(3) tangent (rot, vel, pos, time) to a NavState
#: tangent (rot, pos, vel).  Omitting it silently swaps position and velocity.
NAVSTATE_PERM = _navstate_perm()
