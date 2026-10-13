"""Fixture ``M2_TRAJ_V1``, re-transcribed for M3 (no import of the unmerged M2 branch).

Three keyframes, two edges of 0.5 s, constant body twist per edge, 200 IMU samples per edge
at dt = 2.5e-3, noise-free streams with a constant true bias.  Every number below is a
literal copied from the M2 fixture definition; nothing is sampled.

Bias vectors in this module are in the PROJECT order ``[b_g; b_a]``.

Ground truth is the closed form ``dR = Exp(w T)``, ``dv = M1(T) f``, ``dp = M2(T) f`` with
``M1 = int_0^T Exp(w s) ds``, ``M2 = int_0^T M1``, chained by
``R_j = R_i dR``, ``v_j = v_i + g T + R_i dv``, ``p_j = p_i + v_i T + 1/2 g T^2 + R_i dp``.
"""
from __future__ import annotations

import numpy as np

G_W = np.array([0.0, 0.0, -9.81])          #: gravity vector, world frame, z-up
DT = 2.5e-3                                #: IMU sample period
T_EDGE = 0.5                               #: edge duration
STEPS_PER_EDGE = 200
N_EDGES = 2
N_KF = 3

B_G = np.array([0.0100, -0.0150, 0.0080])  #: true gyro bias
B_A = np.array([0.0600, -0.0400, 0.0900])  #: true accel bias
BIAS_TRUE = np.concatenate([B_G, B_A])     #: project order [b_g; b_a]

OMEGA_BODY = (np.array([0.35, -0.22, 0.17]), np.array([-0.28, 0.41, -0.33]))
F_BODY = (np.array([0.60, 0.45, 9.95]), np.array([-0.75, 0.90, 9.60]))

ROT_VEC_0 = np.array([0.10, -0.20, 0.30])
P_0 = np.zeros(3)
V_0 = np.array([0.50, -0.30, 0.20])

# -- noise model of the M3 instance (see README of this package for provenance) ---------
SIGMA_G = 1.7e-4        #: gyro white-noise density (rad/s/sqrt(Hz)); M2 baseline value
SIGMA_A = 1.0e-2        #: accel white-noise density (m/s^2/sqrt(Hz)); M2 baseline value
SIGMA_INT = 1.0e-6      #: integration noise density; M2 baseline value
SIGMA_PRIOR = 1.0e-3    #: KF0 R / v / p prior sigma (M3 domain)
SIGMA_GNSS = 0.05       #: GNSS position sigma on KF1, KF2 (M3 domain)
SIGMA_BG_RW = 1.9e-5    #: gyro-bias random-walk density (M3 domain)
SIGMA_BA_RW = 3.0e-3    #: accel-bias random-walk density (M3 domain)


def hat(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def so3_exp(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    th = float(np.linalg.norm(w))
    K = hat(w)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def _m1(w, T):
    th = float(np.linalg.norm(w))
    K = hat(w) / th
    return T * np.eye(3) + (1 - np.cos(th * T)) / th * K + (T - np.sin(th * T) / th) * K @ K


def _m2(w, T):
    th = float(np.linalg.norm(w))
    K = hat(w) / th
    return (0.5 * T**2 * np.eye(3) + (T - np.sin(th * T) / th) / th * K
            + (0.5 * T**2 - (1 - np.cos(th * T)) / th**2) * K @ K)


def exact_increment(e: int):
    """Closed-form gravity-free increment ``(dR, dv, dp)`` of edge ``e``."""
    w, f, T = OMEGA_BODY[e], F_BODY[e], T_EDGE
    return so3_exp(w * T), _m1(w, T) @ f, _m2(w, T) @ f


def true_states():
    """``[(R_k, v_k, p_k)]`` for k = 0, 1, 2 from the closed form."""
    out = [(so3_exp(ROT_VEC_0), V_0.copy(), P_0.copy())]
    for e in range(N_EDGES):
        R, v, p = out[-1]
        dR, dv, dp = exact_increment(e)
        T = T_EDGE
        out.append((R @ dR, v + G_W * T + R @ dv, p + v * T + 0.5 * G_W * T**2 + R @ dp))
    return out


def streams(e: int):
    """Noise-free ``(gyro, accel)`` sample arrays of edge ``e`` (true bias added)."""
    gyro = np.tile(OMEGA_BODY[e] + B_G, (STEPS_PER_EDGE, 1))
    acc = np.tile(F_BODY[e] + B_A, (STEPS_PER_EDGE, 1))
    return gyro, acc
