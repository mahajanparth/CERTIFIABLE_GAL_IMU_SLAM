"""Reproduces the pre-flight findings F2, F3, F4, F8 and F9.

These were established before any milestone work and are the ground the Stage B triad
builds on. They are kept runnable so that nothing downstream has to take them on trust:

    python -m experiments.preflight.gal3_checks

F2 is the important one -- it is a reproducible error in the project's own math contract.
"""
from __future__ import annotations

import numpy as np

G = np.array([0.0, 0.0, -9.81])


def sk(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    K = sk(w)
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def gal(R, v, p, t):
    """X = [[R, v, p], [0, 1, t], [0, 0, 1]] -- the convention fixed for this project."""
    M = np.eye(5)
    M[0:3, 0:3] = R
    M[0:3, 3] = v
    M[0:3, 4] = p
    M[3, 4] = t
    return M


def gal_inv(R, v, p, t):
    return gal(R.T, -R.T @ v, -R.T @ (p - t * v), -t)


def unpack(M):
    return M[0:3, 0:3], M[0:3, 3].copy(), M[0:3, 4].copy(), M[3, 4]


def integrate(Ri, vi, pi, omega, acc_b, dt, n):
    """Ground-truth navigation integration and the matching gravity-free increment."""
    R, v, p = Ri.copy(), vi.copy(), pi.copy()
    for _ in range(n):
        p = p + v * dt + 0.5 * (G + R @ acc_b) * dt**2
        v = v + (G + R @ acc_b) * dt
        R = R @ so3_exp(omega * dt)
    dR, dv, dp = np.eye(3), np.zeros(3), np.zeros(3)
    for _ in range(n):
        dp = dp + dv * dt + 0.5 * (dR @ acc_b) * dt**2
        dv = dv + (dR @ acc_b) * dt
        dR = dR @ so3_exp(omega * dt)
    return (R, v, p), (dR, dv, dp)


def endpoint_as_written(Ri, vi, pi, dR, dv, dp, Dt):
    """X_j = H X_i D exactly as the context file writes it. Known to be wrong twice over."""
    H = gal(np.eye(3), G * Dt, 0.5 * G * Dt**2, 0.0)
    return H @ gal(Ri, vi, pi, 0.0) @ gal(dR, dv, dp, Dt)


def endpoint_corrected(Ri, vi, pi, dR, dv, dp, Dt):
    """X_j = T(-Dt) H' X_i D, with H' position block NEGATED and T applied on the LEFT.

    Left-multiplying by T(-Dt) zeroes the time block without perturbing (R, v, p);
    right-multiplying would corrupt p by -v*Dt.
    """
    Hp = gal(np.eye(3), G * Dt, -0.5 * G * Dt**2, 0.0)
    T = gal(np.eye(3), np.zeros(3), np.zeros(3), -Dt)
    return T @ Hp @ gal(Ri, vi, pi, 0.0) @ gal(dR, dv, dp, Dt)


def adjoint_closed_form(R, v, p, t, phi, u, w, alpha):
    """Ad_X(xi) in closed form. Raw degree 3; closes to degree 2 under staged lifting."""
    Rphi = R @ phi
    return (Rphi,
            R @ u + sk(v) @ Rphi,
            R @ w - t * (R @ u) + sk(p - t * v) @ Rphi + alpha * v,
            alpha)


def cayley(A):
    n = A.shape[0]
    return np.linalg.solve(np.eye(n) - A / 2, np.eye(n) + A / 2)


def gal_wedge(phi, u, w, alpha):
    X = np.zeros((5, 5))
    X[0:3, 0:3] = sk(phi)
    X[0:3, 3] = u
    X[0:3, 4] = w
    X[3, 4] = alpha
    return X
