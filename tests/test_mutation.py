"""Mutation harness: every defect below must be CAUGHT by the claim tests.

Each mutation is a single textual edit applied to a throw-away copy of the repo
(``src/``, ``tests/``, ``conftest.py``, ``pyproject.toml``); the named tests are
then run in a subprocess and every one of them must FAIL.  An unmutated copy is
run the same way and must PASS, so a failure is attributable to the mutation and
not to the harness.

The first group are the defects the M1 verifier showed SURVIVING the suite:

* the "5000-step integrator ground truth" replaced by 2 steps at dt = 0.25 -- it
  survived because the endpoint relation is an exact identity of the discrete
  recursion for any dt;
* swapped accumulation order ``dR @ E -> E @ dR`` (integrate_truth) and
  ``dR @ W -> W @ dR`` (euler_increment) -- they survived because constant
  (omega, a) makes every per-step factor commute.

The second group are positive controls the verifier showed were already caught.

The third group (M1-F07 / M1-F09) are the defects the second-pass verifier showed
SURVIVING, plus mutants that stand in for deleted test-local checks.  Each carries
an ``expect`` map: for every named test, a substring of the assertion that must be
the one that fails (pytest runs with ``--tb=short``, so a failure section shows only
the lines on the failing path).  This proves the mutant is killed by the check
written for it, not incidentally by an earlier assertion.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

G3 = "tests/test_gal3.py::"
IMU = "tests/test_imu_dynamics.py::"
ENDPOINT_TESTS = [
    G3 + "test_endpoint_relation",                    # M1-C31
    G3 + "test_uncorrected_endpoint_is_wrong_twice",  # M1-C34
    IMU + "test_gtsam_navstate_endpoint_form",        # M1-C37
    IMU + "test_forster_component_equations",         # M1-C43
]

C40 = G3 + "test_phi_dt_is_an_automorphism_not_a_translation"
C26 = G3 + "test_cayley_denominator_never_vanishes"
C24 = G3 + "test_gamma1_invertibility_and_taylor_branch"
C21 = G3 + "test_log_and_gamma1_inverse"
C41 = IMU + "test_increment_left_invariant_ode"
C45 = IMU + "test_preintegration_chaining_is_the_group_product"
C30 = G3 + "test_cayley_constraints_degree_two"
C36 = G3 + "test_endpoint_only_valid_on_zero_time_slice"
C42 = IMU + "test_increment_closed_form_blocks"
C02 = G3 + "test_closure"
C22 = G3 + "test_log_exp_roundtrip"
C31 = G3 + "test_endpoint_relation"
C32 = G3 + "test_gravity_element_is_one_exponential"
C33 = G3 + "test_gravity_one_parameter_subgroup"
C34 = G3 + "test_uncorrected_endpoint_is_wrong_twice"
C35 = G3 + "test_right_time_shift_corrupts_position"
C37 = IMU + "test_gtsam_navstate_endpoint_form"
C38 = G3 + "test_two_endpoint_forms_agree"
C43 = IMU + "test_forster_component_equations"
C55 = IMU + "test_bias_convention_swap"
# every test that draws random g (rand_gravity) or uses endpoint_datasets_rand_g
RAND_G_TESTS = [C31, C32, C33, C34, C35, C36, C37, C38, C43, C45]

# pass-8 F1W: rand_gravity planar in the x-z plane, >= 0.6 rad off z (g_y = 0 exactly)
RAND_G_OLD = ("    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1))"
              " * rand_axis(r, n)\n")
F1W = ("tests/conftest.py", RAND_G_OLD,
       "    b = r.uniform(0.6, np.pi - 0.6, size=n)\n"
       "    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1))"
       " * np.stack([np.sin(b), 0.0 * b, -np.cos(b)], axis=-1)\n")
# tests whose random-g guards all see >= 3 draws (the span check needs m >= 3); C45's
# only random g is ONE draw, guarded by off_z / perm_gap / comp alone
SPAN_TESTS = [n for n in RAND_G_TESTS if n != C45]
# pass-8 Y_LIFT / Y_ELEM / Y_TRUTH / Y_FORST: g -> g * (1, -1, 1) in one src function.
# name -> ((file, old, new), tests that must fail, {test: failing assertion})
_FLIP_Y = " * np.array([1.0, -1.0, 1.0])"
Y_SIGN = {
    "gravity_lift": (
        ("src/gal3.py",
         "    g = np.asarray(g, dtype=float).reshape(3)\n"
         "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n",
         "    g = np.asarray(g, dtype=float).reshape(3)" + _FLIP_Y + "\n"
         "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n"),
        [C31, C32, C35, C36, C38, C43],
        {C31: 'assert worst_ode <= 1e-9, (d["label"], worst_ode)',
         C32: "assert worst_fuse_rg <= 1e-12, worst_fuse_rg",
         C35: "assert worst_p_rg <= 1e-10, worst_p_rg",
         C36: "assert worst_p_rg <= 1e-10, worst_p_rg",
         C38: "assert worst_w_rg <= 1e-12, worst_w_rg",
         C43: 'assert worst_block <= 1e-12, (d["label"], worst_block)'}),
    "gravity_element": (
        ("src/gal3.py",
         "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, -Dt)\n",
         "    g = g" + _FLIP_Y + "\n    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, -Dt)\n"),
        [C32, C33], {C32: "assert worst_def_rg <= 1e-12, worst_def_rg",
                     C33: "assert worst_def_rg <= 1e-10, worst_def_rg"}),
    "integrate_truth": (
        ("src/imu_dynamics.py",
         "    g = np.asarray(g, dtype=float).reshape(3)\n\n    E_s",
         "    g = np.asarray(g, dtype=float).reshape(3)" + _FLIP_Y + "\n\n    E_s"),
        [C31, C34, C37, C43, C45],
        {C31: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C34: 'assert worst_p <= 1e-12, (d["label"], worst_p)',
         C37: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C43: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C45: "assert worst_large_nav <= 1e-12, worst_large_nav"}),
    "forster_endpoint": (
        ("src/gal3.py",
         "    R_i = np.asarray(R_i, dtype=float)\n    v_i = np.asarray(v_i, dtype=float).reshape(3)\n",
         "    g = g" + _FLIP_Y + "\n"
         "    R_i = np.asarray(R_i, dtype=float)\n    v_i = np.asarray(v_i, dtype=float).reshape(3)\n"),
        [C31, C37, C43], {C31: 'assert worst_alg <= 1e-12, (d["label"], worst_alg)',
                          C37: "assert worst_alg <= 1e-12, worst_alg",
                          C43: 'assert worst <= 1e-9, (d["label"], worst)'}),
}

C53 = IMU + "test_gtsam_pim_predict_matches_both_forms"
# pass-9 M1-F22: the default gravity mutated (N30: points UP)
G_DEFAULT_OLD = "G_DEFAULT = np.array([0.0, 0.0, -9.81])\n"
N30 = ("src/gal3.py", G_DEFAULT_OLD, "G_DEFAULT = np.array([0.0, 0.0, 9.81])\n")
# pass-9 M1-F23: a HALF-SPACE gravity fixture (g_y := |g_y|; passes off_z/perm_gap/spread/
# comp/span) and the sign-dependent g_y -> |g_y| defects it masked
H_HEMI = ("tests/conftest.py", RAND_G_OLD,
          "    g = r.uniform(2.0, 20.0, size=None if n is None else (n, 1)) * rand_axis(r, n)\n"
          "    g[..., 1] = np.abs(g[..., 1])\n"
          "    return g\n")
YABS_LIFT = ("src/gal3.py",
             "    g = np.asarray(g, dtype=float).reshape(3)\n"
             "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n",
             "    g = np.asarray(g, dtype=float).reshape(3)\n"
             "    g = np.array([g[0], abs(g[1]), g[2]])\n"
             "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n")
YABS_TRUTH = ("src/imu_dynamics.py",
              "    g = np.asarray(g, dtype=float).reshape(3)\n\n    E_s",
              "    g = np.asarray(g, dtype=float).reshape(3)\n"
              "    g = np.array([g[0], abs(g[1]), g[2]])\n\n    E_s")

# id -> (file, old, new, tests that must ALL fail[, {test: failing-assertion substring}])
MUTATIONS = {
    # ---- previously SURVIVING ------------------------------------------------
    "fixture_2_steps_dt_0.25": (
        "tests/conftest.py",
        "n_inst, n_steps, dt = 200, 5000, 1e-4",
        "n_inst, n_steps, dt = 200, 2, 0.25",
        ENDPOINT_TESTS,
    ),
    "integrate_truth_increment_dR@E->E@dR": (
        "src/imu_dynamics.py",
        "        dR = dR @ E\n    return (R, v, p)",
        "        dR = E @ dR\n    return (R, v, p)",
        ENDPOINT_TESTS,
    ),
    "euler_increment_dR@W->W@dR": (
        "src/imu_dynamics.py",
        "dR = dR + (dR @ W) * dt",
        "dR = dR + (W @ dR) * dt",
        [IMU + "test_increment_left_invariant_ode"],  # M1-C41
    ),
    # ---- order mutations already caught (kept as regression guards) ---------
    "integrate_truth_navstate_R@E->E@R": (
        "src/imu_dynamics.py",
        "        R = R @ E\n",
        "        R = E @ R\n",
        ENDPOINT_TESTS,
    ),
    "integrate_increment_dR@E->E@dR": (
        "src/imu_dynamics.py",
        "        dR = dR @ E\n    return dR, dv, dp, t0",
        "        dR = E @ dR\n    return dR, dv, dp, t0",
        [IMU + "test_preintegration_chaining_is_the_group_product"],  # M1-C45
    ),
    # ---- positive controls ---------------------------------------------------
    "cay_position_sign_flip": (
        "src/gal3.py",
        "rhs[:, 4] = w + 0.5 * alpha * u",
        "rhs[:, 4] = w - 0.5 * alpha * u",
        [G3 + "test_cayley_closed_form",                  # M1-C25
         G3 + "test_cayley_denominator_never_vanishes"],  # M1-C26 totality vs definition
    ),
    "gamma_2sin2->1-cos": (
        "src/gal3.py",
        "omc = 2.0 * np.sin(0.5 * theta) ** 2",
        "omc = 1.0 - np.cos(theta)",
        [G3 + "test_gamma1_invertibility_and_taylor_branch"],  # M1-C24
    ),
    "H_prime_position_sign_flip": (
        "src/gal3.py",
        "return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)",
        "return gal(_I3, g * Dt, +0.5 * g * Dt * Dt, 0.0)",
        [G3 + "test_endpoint_relation",              # M1-C31
         IMU + "test_forster_component_equations"],  # M1-C43
    ),
    "compose_v1*t2->v2*t1": (
        "src/gal3.py",
        "p1 + R1 @ p2 + v1 * t2, t1 + t2)",
        "p1 + R1 @ p2 + v2 * t1, t1 + t2)",
        [G3 + "test_product_closed_form",   # M1-C01
         G3 + "test_associativity"],        # M1-C03
    ),
    # ---- second-pass SURVIVORS (M1-F07), killed on the intended assertion ------
    "navstate_phi_sign_flip": (                                   # verifier M27
        "src/gal3.py",
        "N[0:3, 3] + N[0:3, 4] * Dt, N[0:3, 4])",
        "N[0:3, 3] - N[0:3, 4] * Dt, N[0:3, 4])",
        [C40], {C40: "assert worst_shift <= 1e-12"},
    ),
    "navstate_phi_no_shift": (                                    # verifier T4
        "src/gal3.py",
        "N[0:3, 3] + N[0:3, 4] * Dt, N[0:3, 4])",
        "N[0:3, 3], N[0:3, 4])",
        [C40], {C40: "assert abs(gap - float(np.abs(v).max()) * Dt) <= 1e-12"},
    ),
    "cay_denominator_minus->plus": (                                   # verifier M32
        "src/gal3.py",
        "return _I5 - 0.5 * wedge(xi)",
        "return _I5 + 0.5 * wedge(xi)",
        [C26], {C26: "assert worst_mat <= 1e-12"},
    ),
    "integrate_increment_dp_uses_new_dv": (                       # verifier M45
        "src/imu_dynamics.py",
        "        dp = dp + dv * dt + half * df\n        dv = dv + df * dt\n"
        "        dR = dR @ E\n    return dR, dv, dp, t0",
        "        dv = dv + df * dt\n        dp = dp + dv * dt + half * df\n"
        "        dR = dR @ E\n    return dR, dv, dp, t0",
        [C45], {C45: "assert worst_w0 <= 1e-12"},
    ),
    "imu_step_element_drop_half": (                               # verifier M49
        "src/imu_dynamics.py",
        "a * dt, 0.5 * a * dt * dt, dt)",
        "a * dt, a * dt * dt, dt)",
        [C45], {C45: "assert worst_step <= 1e-12"},
    ),
    "euler_increment_semi_implicit": (                            # verifier M51
        "src/imu_dynamics.py",
        "        dp = dp + dv * dt\n        dv = dv + dRa * dt\n",
        "        dv = dv + dRa * dt\n        dp = dp + dv * dt\n",
        [C41], {C41: "assert worst_coef <= 1e-2"},
    ),
    # ---- stand-ins for deleted test-local checks (M1-F09) ----------------------
    "fixture_constant_omega": (            # the endpoint stream made commuting
        "tests/conftest.py",
        "stream = ode_truth.Stream.random(r, n_inst)",
        "stream = ode_truth.Stream.random(r, n_inst, w_amp=0.0)",
        ENDPOINT_TESTS, {n: 'assert d["order_gap"] >= 0.1' for n in ENDPOINT_TESTS},
    ),
    "ode_truth_constant_omega": (          # verifier T3: C41's stream made commuting
        "tests/ode_truth.py",
        "            w1=w_amp * r.normal(size=(n, 3)),",
        "            w1=0.0 * r.normal(size=(n, 3)),",
        [C41], {C41: "assert order_gap >= 100.0 * 1e-4"},
    ),
    "taylor_branch_removed": (             # replaces the gamma_coefficients_closed_form strawman
        "src/gal3.py",
        "    if theta < THETA_SMALL:\n        t2 = theta * theta\n        t4 = t2 * t2\n"
        "        kappa1",
        "    if theta == 0.0:\n        t2 = theta * theta\n        t4 = t2 * t2\n"
        "        kappa1",
        [C24], {C24: "assert max(worst) <= 1e-11"},
    ),
    "theta_small_8e-2": (                  # replaces the test-local trunc_at_8e2 check
        "src/gal3.py",
        "THETA_SMALL = 5e-2",
        "THETA_SMALL = 8e-2",
        [C24], {C24: "assert max(worst_sweep) <= 1e-11"},
    ),
    "c2_forbidden_form": (                 # replaces the test-local forbidden_worst check
        "src/gal3.py",
        "    c2 = (1.0 - theta * s / (2.0 * omc)) / t2\n",
        "    c2 = 1.0 / t2 - (1.0 + np.cos(theta)) / (2.0 * theta * s)\n",
        [C21], {C21: "assert ours_worst <= 1e-13"},
    ),
    # ---- third-pass SURVIVORS (M1-F11), killed on the intended assertion -------
    "cayley_symbolic_E3_no_alpha_u": (                            # verifier V22
        "src/gal3.py",
        "    E3 = S * pe - (wv + alpha / 2 * uv)\n",
        "    E3 = S * pe - wv\n",
        [C30], {C30: "assert worst_sym_cay <= 1e-12"},
    ),
    "cayley_symbolic_S_plus": (                                   # verifier V23
        "src/gal3.py",
        "    S = sp.eye(3) - P / 2\n",
        "    S = sp.eye(3) + P / 2\n",
        [C30], {C30: "assert worst_sym_cay <= 1e-12"},
    ),
    "cayley_symbolic_alpha_fresh_symbol": (                       # pass-4 verifier N01
        "src/gal3.py",
        "    return exprs, syms\n",
        "    syms = [sp.Symbol('alpha', real=True) if str(x) == 'alpha' else x for x in syms]\n    syms[9] = sp.Symbol('alpha', positive=True)\n    return exprs, syms\n",
        [C30], {C30: "assert not sym_mismatch, sym_mismatch"},
    ),
    "keyframe_time_tol_1e-3": (                                   # verifier V26
        "src/gal3.py",
        "KEYFRAME_TIME_TOL = 1e-12",
        "KEYFRAME_TIME_TOL = 1e-3",
        [C36], {C36: "assert not near_bad, near_bad"},
    ),
    "so3_exp_batch_c_over_theta": (                               # verifier V34
        "src/imu_dynamics.py",
        "(1.0 - np.cos(np.where(small, 1.0, th))) / np.where(small, 1.0, t2))",
        "(1.0 - np.cos(np.where(small, 1.0, th))) / np.where(small, 1.0, th))",
        [C42, C45], {C42: "assert worst_batch <= 1e-14, worst_batch",
                     C45: "assert worst_large <= 1e-12, worst_large"},
    ),
    "so3_exp_batch_taylor_no_th4": (                              # verifier V33 (low)
        "src/imu_dynamics.py",
        "1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0,",
        "1.0 - t2 / 6.0 - t2 * t2 * t2 / 5040.0,",
        [C42], {C42: "assert worst_batch <= 1e-14, worst_batch"},
    ),
    "structural_residual_no_det": (                               # verifier V06 (low)
        "src/gal3.py",
        "        abs(float(np.linalg.det(R)) - 1.0),\n",
        "",
        [C02], {C02: "assert not missed, missed"},
    ),
    # ---- fifth-pass SURVIVORS (M1-F13) and the advisory items of passes 3-4 (M1-F14) --
    "cayley_symbolic_expr_foreign_alpha": (                       # stray-symbol pin itself
        "src/gal3.py",
        "    E3 = S * pe - (wv + alpha / 2 * uv)\n",
        "    E3 = S * pe - (wv + sp.Symbol('alpha') / 2 * uv)\n",
        [C30], {C30: "assert not stray, stray"},
    ),
    "cayley_symbolic_Re_column_major": (                          # pass-4 N06
        "src/gal3.py",
        "        + list(Re)\n",
        "        + list(Re.T)\n",
        [C30], {C30: "assert [str(x) for x in syms] == expected_names"},
    ),
    "cayley_symbolic_alpha_positive": (                           # pass-5 P10
        "src/gal3.py",
        '    alpha = sp.Symbol("alpha", real=True)\n',
        '    alpha = sp.Symbol("alpha", positive=True)\n',
        [C30], {C30: "assert not sym_mismatch, sym_mismatch"},
    ),
    "cayley_symbolic_not_real": (                                 # pass-5 P13
        "src/gal3.py",
        '    phi = sp.symbols("phi0:3", real=True)\n'
        '    u = sp.symbols("u0:3", real=True)\n'
        '    w = sp.symbols("w0:3", real=True)\n'
        '    alpha = sp.Symbol("alpha", real=True)\n'
        '    Re = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"Re{i}{j}", real=True))\n'
        '    ve = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"ve{i}", real=True))\n'
        '    pe = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"pe{i}", real=True))\n',
        '    phi = sp.symbols("phi0:3")\n'
        '    u = sp.symbols("u0:3")\n'
        '    w = sp.symbols("w0:3")\n'
        '    alpha = sp.Symbol("alpha")\n'
        '    Re = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"Re{i}{j}"))\n'
        '    ve = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"ve{i}"))\n'
        '    pe = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"pe{i}"))\n',
        [C30], {C30: "assert not sym_mismatch, sym_mismatch"},
    ),
    "cayley_symbolic_alpha_subclass": (                           # pass-5 P03
        "src/gal3.py",
        '    alpha = sp.Symbol("alpha", real=True)\n',
        '    alpha = type("SymAlpha", (sp.Symbol,), {})("alpha", real=True)\n',
        [C30], {C30: "assert not sym_mismatch, sym_mismatch"},
    ),
    "cayley_symbolic_alpha_printer_spoof": (                      # pass-5 P16
        "src/gal3.py",
        '    alpha = sp.Symbol("alpha", real=True)\n',
        '    alpha = type("Spoof", (sp.Symbol,), {"_sympystr": lambda self, p: "alpha"})'
        '("beta", real=True)\n',
        [C30], {C30: "assert not sym_mismatch, sym_mismatch"},
    ),
    "cayley_symbolic_float_half": (                               # pass-5 P11
        "src/gal3.py",
        "    E3 = S * pe - (wv + alpha / 2 * uv)\n",
        "    E3 = S * pe - (wv + 0.5 * alpha * uv)\n",
        [C30], {C30: "assert not bad_domains, bad_domains"},
    ),
    "cayley_symbolic_opaque_half": (                              # pass-5 P08
        "src/gal3.py",
        "    E3 = S * pe - (wv + alpha / 2 * uv)\n",
        "    E3 = S * pe - (wv + __import__('sympy.utilities.lambdify', fromlist=['x'])"
        ".implemented_function('h', lambda: 0.5)() * alpha * uv)\n",
        [C30], {C30: "assert not bad_domains, bad_domains"},
    ),
    "cayley_symbolic_returns_poly": (                             # pass-5 P15
        "src/gal3.py",
        "    return exprs, syms\n",
        "    return [sp.Poly(e, *syms) for e in exprs], syms\n",
        [C30], {C30: "assert not not_expr, not_expr"},
    ),
    "preintegrate_constant_wrong_above_2": (                      # pass-5 P19
        "src/imu_dynamics.py",
        "    return exp(float(Dt) * increment_generator(omega, acc))\n",
        "    C = gal(gal3.so3_exp([0.01, 0.0, 0.0]), np.zeros(3), np.zeros(3), 0.0)\n"
        "    return (C if float(Dt) > 2.0 else np.eye(5)) @ "
        "exp(float(Dt) * increment_generator(omega, acc))\n",
        [C41], {C41: "assert worst_long <= 1e-12, worst_long"},
    ),
    "preintegrate_constant_abs_dt": (                             # pass-5 P22
        "src/imu_dynamics.py",
        "    return exp(float(Dt) * increment_generator(omega, acc))\n",
        "    return exp(abs(float(Dt)) * increment_generator(omega, acc))\n",
        [C41], {C41: "assert worst_long <= 1e-12, worst_long"},
    ),
    "keyframe_time_tol_1.9e-12": (                                # pass-4 N14
        "src/gal3.py",
        "KEYFRAME_TIME_TOL = 1e-12",
        "KEYFRAME_TIME_TOL = 1.9e-12",
        [C36], {C36: "assert not edge_bad, edge_bad"},
    ),
    "keyframe_time_tol_9.9e-13": (
        "src/gal3.py",
        "KEYFRAME_TIME_TOL = 1e-12",
        "KEYFRAME_TIME_TOL = 9.9e-13",
        [C36], {C36: "assert not edge_ok, edge_ok"},
    ),
    "keyframe_guard_admits_nan": (                                # pass-4 NaN observation
        "src/gal3.py",
        "    if not abs(t_i) <= KEYFRAME_TIME_TOL:\n",
        "    if abs(t_i) > KEYFRAME_TIME_TOL:\n",
        [C36], {C36: "assert not nonfinite_bad, nonfinite_bad"},
    ),
    "so3_exp_batch_taylor_no_th6": (
        "src/imu_dynamics.py",
        "1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0,",
        "1.0 - t2 / 6.0 + t2 * t2 / 120.0,",
        [C42], {C42: "assert worst_batch <= 1e-14, worst_batch"},
    ),
    "so3_exp_batch_c_th4_over_700": (                             # pass-4 N19
        "src/imu_dynamics.py",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0",
        "0.5 - t2 / 24.0 + t2 * t2 / 700.0",
        [C42], {C42: "assert worst_batch <= 1e-14, worst_batch"},
    ),
    "so3_exp_batch_c_no_th4": (                                   # pass-4 N18
        "src/imu_dynamics.py",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0",
        "0.5 - t2 / 24.0",
        [C42], {C42: "assert worst_batch <= 1e-14, worst_batch"},
    ),
    "so3_log_trace_branch_below_0": (                             # pass-3 V44
        "src/gal3.py",
        "    if tr > 0.0:\n",
        "    if tr > -0.99:\n",
        [C22], {C22: "assert worst_band <= 2.5e-15, worst_band"},
    ),
    # ---- sixth-pass SURVIVORS (M1-F15 / M1-F16) and the M1-F15 regression itself ----
    "so3_exp_taylor_no_th6": (                                    # M1-F15 regression
        "src/gal3.py",
        "s = 1.0 - t2 / 6.0 + t4 / 120.0 - t4 * t2 / 5040.0",
        "s = 1.0 - t2 / 6.0 + t4 / 120.0",
        [C42, C45], {C42: "assert worst_so3_scipy <= 1e-14, worst_so3_scipy",
                     C45: "assert worst_prod_switch <= 1e-12, worst_prod_switch"},
    ),
    "so3_exp_c_th4_over_700": (                                   # pass-6 Q05
        "src/gal3.py",
        "        c = 0.5 - t2 / 24.0 + t4 / 720.0         # (1 - cos th)/th^2\n",
        "        c = 0.5 - t2 / 24.0 + t4 / 700.0         # (1 - cos th)/th^2\n",
        [C42], {C42: "assert worst_so3_scipy <= 1e-14, worst_so3_scipy"},
    ),
    "structural_residual_row3_short": (                           # pass-6 Q25
        "src/gal3.py",
        "        np.abs(X[3, 0:3]).max(),\n",
        "        np.abs(X[3, 0:2]).max(),\n",
        [C02], {C02: "assert not missed, missed"},
    ),
    "gravity_lift_ignores_g": (                                   # pass-6 Q49
        "src/gal3.py",
        "    g = np.asarray(g, dtype=float).reshape(3)\n"
        "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n",
        "    g = G_DEFAULT\n"
        "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n",
        [C31, C32, C35, C36, C38, C43],
        {C31: 'assert worst_ode <= 1e-9, (d["label"], worst_ode)',
         C32: "assert worst_fuse_rg <= 1e-12, worst_fuse_rg",
         C35: "assert worst_p_rg <= 1e-10, worst_p_rg",
         C36: "assert worst_p_rg <= 1e-10, worst_p_rg",
         C38: "assert worst_w_rg <= 1e-12, worst_w_rg",
         C43: 'assert worst_block <= 1e-12, (d["label"], worst_block)'},
    ),
    "gravity_element_g_permuted": (                               # pass-6 Q50
        "src/gal3.py",
        "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, -Dt)\n",
        "    g = g[[1, 0, 2]]\n    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, -Dt)\n",
        [C32, C33], {C32: "assert worst_def_rg <= 1e-12, worst_def_rg",
                     C33: "assert worst_def_rg <= 1e-10, worst_def_rg"},
    ),
    "forster_endpoint_g_permuted": (                              # pass-6 Q51
        "src/gal3.py",
        "    R_i = np.asarray(R_i, dtype=float)\n    v_i = np.asarray(v_i, dtype=float).reshape(3)\n",
        "    g = g[[1, 0, 2]]\n"
        "    R_i = np.asarray(R_i, dtype=float)\n    v_i = np.asarray(v_i, dtype=float).reshape(3)\n",
        [C31, C37, C43], {C31: 'assert worst_alg <= 1e-12, (d["label"], worst_alg)',
                          C37: "assert worst_alg <= 1e-12, worst_alg",
                          C43: 'assert worst <= 1e-9, (d["label"], worst)'},
    ),
    "integrate_truth_g_permuted": (                               # pass-6 Q52
        "src/imu_dynamics.py",
        "    g = np.asarray(g, dtype=float).reshape(3)\n\n    E_s",
        "    g = np.asarray(g, dtype=float).reshape(3)[[1, 0, 2]]\n\n    E_s",
        [C31, C34, C37, C43, C45],
        {C31: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C34: 'assert worst_p <= 1e-12, (d["label"], worst_p)',
         C37: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C43: 'assert worst_id <= 1e-12, (d["label"], worst_id)',
         C45: "assert worst_large_nav <= 1e-12, worst_large_nav"},
    ),
    "gravity_generator_g_permuted": (                             # pass-6 Q53
        "src/gal3.py",
        "    return np.concatenate([np.zeros(3), g, np.zeros(3), [-1.0]])\n",
        "    return np.concatenate([np.zeros(3), g[[1, 0, 2]], np.zeros(3), [-1.0]])\n",
        [C32], {C32: "assert worst_gen_rg == 0.0, worst_gen_rg"},
    ),
    "gravity_lift_uncorrected_ignores_g": (                       # pass-6 Q54
        "src/gal3.py",
        "    g = np.asarray(g, dtype=float).reshape(3)\n"
        "    return gal(_I3, g * Dt, 0.5 * g * Dt * Dt, 0.0)\n",
        "    g = G_DEFAULT\n"
        "    return gal(_I3, g * Dt, 0.5 * g * Dt * Dt, 0.0)\n",
        [C34], {C34: 'assert worst_p <= 1e-9, (d["label"], worst_p)'},
    ),
    "endpoint_abs_dt": (                                          # pass-6 Q28
        "src/gal3.py",
        "    Dt = float(np.asarray(D_ij)[3, 4])\n    left = compose(",
        "    Dt = abs(float(np.asarray(D_ij)[3, 4]))\n    left = compose(",
        [C31, C36, C38, C43],
        {C31: 'assert worst_ode <= 1e-9, (d["label"], worst_ode)',
         C36: "assert worst_p_rg <= 1e-10, worst_p_rg",
         C38: "assert worst_form_rg <= 1e-9, worst_form_rg",
         C43: 'assert worst_block <= 1e-12, (d["label"], worst_block)'},
    ),
    "gravity_lift_abs_dt": (                                      # pass-6 Q29
        "src/gal3.py",
        "    return gal(_I3, g * Dt, -0.5 * g * Dt * Dt, 0.0)\n",
        "    return gal(_I3, g * abs(Dt), -0.5 * g * Dt * Dt, 0.0)\n",
        [C31, C32, C38, C43],
        {C31: 'assert worst_ode <= 1e-9, (d["label"], worst_ode)',
         C32: "assert worst_fuse_rg <= 1e-12, worst_fuse_rg",
         C38: "assert worst_w_rg <= 1e-12, worst_w_rg",
         C43: 'assert worst_block <= 1e-12, (d["label"], worst_block)'},
    ),
    "forster_endpoint_abs_dt": (                                  # pass-6 Q30
        "src/gal3.py",
        "0.5 * g * Dt * Dt + R_i @",
        "0.5 * g * Dt * abs(Dt) + R_i @",
        [C31, C37, C43], {C31: 'assert worst_alg <= 1e-12, (d["label"], worst_alg)',
                          C37: "assert worst_alg <= 1e-12, worst_alg",
                          C43: 'assert worst <= 1e-9, (d["label"], worst)'},
    ),
    "navstate_phi_abs_dt": (                                      # pass-6 Q31
        "src/gal3.py",
        "N[0:3, 3] + N[0:3, 4] * Dt, N[0:3, 4])",
        "N[0:3, 3] + N[0:3, 4] * abs(Dt), N[0:3, 4])",
        [C40], {C40: "assert worst_shift_neg <= 1e-12, worst_shift_neg"},
    ),
    "increment_blocks_dp_abs_dt": (                               # pass-6 Q35
        "src/imu_dynamics.py",
        "        Dt * Dt * (gal3.gamma2(phi) @ acc),\n",
        "        Dt * abs(Dt) * (gal3.gamma2(phi) @ acc),\n",
        [C42], {C42: "assert worst_expm <= 1e-12, worst_expm"},
    ),
    "imu_step_element_abs_dt": (                                  # pass-6 Q36
        "src/imu_dynamics.py",
        "a * dt, 0.5 * a * dt * dt, dt)",
        "a * dt, 0.5 * a * dt * abs(dt), dt)",
        [C45], {C45: "assert worst_step_neg <= 1e-12, worst_step_neg"},
    ),
    "so3_log_small_branch_1e-4": (                                # pass-6 Q07
        "src/gal3.py",
        "    if n < 1e-10:\n",
        "    if n < 1e-4:\n",
        [C22], {C22: "assert worst_small <= 1e-15, worst_small"},
    ),
    "so3_exp_batch_switch_0.03_no_th6": (                         # pass-6 Q41 (two edits)
        "src/imu_dynamics.py",
        '    small = th < gal3.THETA_SMALL\n    # sin(th)/th carries its th^6 Taylor term: truncating after th^4 leaves th^7/5040\n    # = 1.5e-13 in the matrix just below the switch (M1-C42), whereas the next omitted\n    # term, th^9/9!, is 5e-18.  (1 - cos th)/th^2 stops at th^4: its first omitted\n    # term th^6/8! enters multiplied by |K K| <= th^2, i.e. <= 1e-15 at the switch.\n    s = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0 - t2 * t2 * t2 / 5040.0,',
        '    small = th < 0.03\n    # sin(th)/th carries its th^6 Taylor term: truncating after th^4 leaves th^7/5040\n    # = 1.5e-13 in the matrix just below the switch (M1-C42), whereas the next omitted\n    # term, th^9/9!, is 5e-18.  (1 - cos th)/th^2 stops at th^4: its first omitted\n    # term th^6/8! enters multiplied by |K K| <= th^2, i.e. <= 1e-15 at the switch.\n    s = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0,',
        [C42], {C42: "assert worst_batch_lo <= 2.5e-15, worst_batch_lo"},
    ),
    # ---- stand-ins for the checks deleted in pass 6 ("cannot fail independently") ----
    "bias_to_gtsam_naive": (               # replaces C55's gtsam-only worst_naive check
        "src/imu_dynamics.py",
        "    return gtsam.imuBias.ConstantBias(b[3:6], b[0:3])\n",
        "    return gtsam.imuBias.ConstantBias(b[0:3], b[3:6])\n",
        [C55], {C55: "assert worst <= 1e-14, worst"},
    ),
    "chain_swapped_duration": (            # replaces C45's data-only worst_wrong check
        "src/imu_dynamics.py",
        "    return compose(D_ij, D_jk)\n",
        "    X = compose(D_ij, D_jk)\n"
        "    X[0:3, 4] += D_jk[0:3, 3] * D_ij[3, 4] - D_ij[0:3, 3] * D_jk[3, 4]\n"
        "    return X\n",
        [C45], {C45: "assert worst <= 1e-12, worst"},
    ),
    # ---- seventh-pass SURVIVORS (M1-F17), killed on the intended assertion ----------
    "so3_exp_c_th6_over_10000": (                                 # pass-7 R01 (4.9e-15)
        "src/gal3.py",
        "        c = 0.5 - t2 / 24.0 + t4 / 720.0         # (1 - cos th)/th^2\n",
        "        c = 0.5 - t2 / 24.0 + t4 / 720.0 + t4 * t2 / 10000.0  # (1 - cos th)/th^2\n",
        [C42], {C42: "assert worst_sw_scalar <= 1.3e-15, worst_sw_scalar"},
    ),
    "so3_exp_batch_c_th6_over_10000": (                           # pass-7 R32 (4.9e-15)
        "src/imu_dynamics.py",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0,",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0 + t2 * t2 * t2 / 10000.0,",
        [C42], {C42: "assert worst_sw_batch <= 1.3e-15, worst_sw_batch"},
    ),
    "so3_log_small_branch_1e-5": (                                # pass-7 R03 (3.2e-11 rel)
        "src/gal3.py",
        "    if n < 1e-10:\n",
        "    if n < 1e-5:\n",
        [C22], {C22: "assert worst_small_rel <= 1e-15, worst_small_rel"},
    ),
    "c2_taylor_th4_over_30200": (                                 # pass-7 R05 (3.1e-12)
        "src/gal3.py",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30200.0",
        [C24], {C24: "assert c2_below <= 3e-13, c2_below"},
    ),
    "fixture_rand_gravity_minus_z": (                             # pass-7 F1
        "tests/conftest.py",
        "    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1)) * rand_axis(r, n)\n",
        "    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1))"
        " * np.array([0.0, 0.0, -1.0])\n",
        RAND_G_TESTS, {n: "assert off_z >= min_angle, off_z" for n in RAND_G_TESTS},
    ),
    "fixture_neg_dt_dataset_forward": (                           # pass-7 F2
        "tests/conftest.py",
        '(3102, 3000, -1e-4, "random g, Dt=-0.3")',
        '(3102, 3000, 1e-4, "random g, Dt=-0.3")',
        ENDPOINT_TESTS,
        {n: 'assert (d["Dt"] < 0) == ("Dt=-" in d["label"])' for n in ENDPOINT_TESTS},
    ),
    # ---- eighth-pass SURVIVORS (M1-F18..F21), killed on the intended assertion ------
    "gamma_kappa1_th4_over_719.993": (                            # pass-8 S01 (1.69e-13)
        "src/gal3.py",
        "kappa1 = 0.5 - t2 / 24.0 + t4 / 720.0",
        "kappa1 = 0.5 - t2 / 24.0 + t4 / 719.993",
        [C24], {C24: "assert poly_k1 <= 1e-15, poly_k1"},
    ),
    "gamma_kappa3_th4_over_40319": (                              # pass-8 S02 (9.2e-14)
        "src/gal3.py",
        "kappa3 = 1.0 / 24.0 - t2 / 720.0 + t4 / 40320.0",
        "kappa3 = 1.0 / 24.0 - t2 / 720.0 + t4 / 40319.0",
        [C24], {C24: "assert poly_k3 <= 1e-15, poly_k3"},
    ),
    "gamma_c2_th4_over_30239": (                                  # pass-8 S03 (8.2e-14)
        "src/gal3.py",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30239.0",
        [C24], {C24: "assert poly_c2 <= 1e-15, poly_c2"},
    ),
    "gamma_kappa1_4_terms": (                                     # pass-7 R07 (A3)
        "src/gal3.py",
        "kappa1 = 0.5 - t2 / 24.0 + t4 / 720.0",
        "kappa1 = 0.5 - t2 / 24.0 + t4 / 720.0 - t4 * t2 / 40320.0",
        [C24], {C24: "assert poly_k1 <= 1e-15, poly_k1"},
    ),
    "gamma_kappa2_4_terms": (                                     # pass-8 S04 (A3)
        "src/gal3.py",
        "kappa2 = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0",
        "kappa2 = 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0 - t4 * t2 / 362880.0",
        [C24], {C24: "assert poly_k2 <= 1e-15, poly_k2"},
    ),
    "gamma_c2_4_terms": (                                         # pass-8 S05 (A3)
        "src/gal3.py",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0",
        "c2 = 1.0 / 12.0 + t2 / 720.0 + t4 / 30240.0 + t4 * t2 / 1209600.0",
        [C24], {C24: "assert poly_c2 <= 1e-15, poly_c2"},
    ),
    "so3_exp_c_th6_over_60000": (                                 # pass-8 S06 (1.665e-15)
        "src/gal3.py",
        "        c = 0.5 - t2 / 24.0 + t4 / 720.0         # (1 - cos th)/th^2\n",
        "        c = 0.5 - t2 / 24.0 + t4 / 720.0 + t4 * t2 / 60000.0  # (1 - cos th)/th^2\n",
        [C42], {C42: "assert worst_sw_scalar <= 1.3e-15, worst_sw_scalar"},
    ),
    "so3_exp_batch_c_th6_over_60000": (                           # pass-8 S07 (1.665e-15)
        "src/imu_dynamics.py",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0,",
        "0.5 - t2 / 24.0 + t2 * t2 / 720.0 + t2 * t2 * t2 / 60000.0,",
        [C42], {C42: "assert worst_sw_batch <= 1.3e-15, worst_sw_batch"},
    ),
    "so3_log_small_branch_1e-7": (                                # pass-8 S08 (3.3e-15 rel)
        "src/gal3.py",
        "    if n < 1e-10:\n",
        "    if n < 1e-7:\n",
        [C22], {C22: "assert worst_small_rel <= 1e-15, worst_small_rel"},
    ),
    "fixture_rand_gravity_xz_plane": (                            # pass-8 F1W
        *F1W,
        RAND_G_TESTS, {n: "assert comp >= min_component, comp" for n in RAND_G_TESTS},
    ),
    "fixture_rand_gravity_tilted_plane": (   # planar, every component nonzero: span only
        "tests/conftest.py", RAND_G_OLD,
        "    b = np.asarray(r.uniform(0.6, np.pi - 0.6, size=n))[..., None]\n"
        "    e1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\n"
        "    e2 = np.array([0.0, 0.5, -1.0]) / np.sqrt(1.25)\n"
        "    return r.uniform(2.0, 20.0, size=None if n is None else (n, 1))"
        " * (np.cos(b) * e1 + np.sin(b) * e2)\n",
        SPAN_TESTS, {n: "assert span >= min_span, span" for n in SPAN_TESTS},
    ),
    "euler_increment_dp_abs_dt": (                                # pass-8 S30 (A1)
        "src/imu_dynamics.py",
        "        dp = dp + dv * dt\n",
        "        dp = dp + dv * abs(dt)\n",
        [C41], {C41: "assert worst_pow_neg <= 1e-11, worst_pow_neg"},
    ),
    "inputs_no_acc_length_check": (                               # pass-8 M06 (A2)
        "src/imu_dynamics.py",
        "        if omega.shape[0] != int(n) or acc.shape[0] != int(n):\n",
        "        if omega.shape[0] != int(n):\n",
        [C45], {C45: "assert not not_rejected, not_rejected"},
    ),
    **{f"{name}_g_y_sign": (edit[0], edit[1], edit[2], nodes, expect)
       for name, (edit, nodes, expect) in Y_SIGN.items()},
    **{f"{name}_g_y_sign_under_xz_plane_g": ((F1W, edit), None, None, RAND_G_TESTS,
                                             {n: "assert comp >= min_component, comp"
                                              for n in RAND_G_TESTS})
       for name, (edit, _, _) in Y_SIGN.items()},
    # ---- ninth-pass SURVIVORS (M1-F22, M1-F23) and advisory A5 -----------------
    "G_DEFAULT_points_up": (                                      # pass-9 N30 (M1-F22)
        *N30, [C53], {C53: "assert worst_default <= 1e-9, worst_default"},
    ),
    "G_DEFAULT_9.80665": (                                        # pass-9 N29 (M1-F22)
        "src/gal3.py", G_DEFAULT_OLD, "G_DEFAULT = np.array([0.0, 0.0, -9.80665])\n",
        [C53], {C53: "assert worst_default <= 1e-9, worst_default"},
    ),
    # N30 with the conftest gravity read back from src (the pre-F22 coupling): C53 does
    # not use conftest's G, so the default is still pinned
    "G_DEFAULT_points_up_conftest_coupled": (
        (N30, ("tests/conftest.py", "G = np.array([0.0, 0.0, -9.81])\n", "G = gal3.G_DEFAULT\n")),
        None, None, [C53], {C53: "assert worst_default <= 1e-9, worst_default"},
    ),
    "fixture_rand_gravity_half_space_y": (                        # pass-9 H_HEMI (M1-F23)
        *H_HEMI, SPAN_TESTS, {n: "assert signed >= min_signed, signed" for n in SPAN_TESTS},
    ),
    "gravity_lift_abs_g_y": (                                     # pass-9 YABS_LIFT alone
        *YABS_LIFT, [C31], {C31: 'assert worst_ode <= 1e-9, (d["label"], worst_ode)'},
    ),
    "integrate_truth_abs_g_y": (                                  # pass-9 YABS_TRUTH alone
        *YABS_TRUTH, [C31], {C31: 'assert worst_id <= 1e-12, (d["label"], worst_id)'},
    ),
    "gravity_lift_abs_g_y_under_half_space_g": (                  # pass-9 YABS_LIFT + H_HEMI
        (H_HEMI, YABS_LIFT), None, None,
        SPAN_TESTS, {n: "assert signed >= min_signed, signed" for n in SPAN_TESTS},
    ),
    "integrate_truth_abs_g_y_under_half_space_g": (               # pass-9 YABS_TRUTH + H_HEMI
        (H_HEMI, YABS_TRUTH), None, None,
        SPAN_TESTS, {n: "assert signed >= min_signed, signed" for n in SPAN_TESTS},
    ),
    "gamma_coefficients_accepts_theta_ge_-1": (                   # pass-9 N13 (A5)
        "src/gal3.py", "    if theta < 0.0:\n", "    if theta < -1.0:\n",
        [C24], {C24: "assert not not_rejected, not_rejected"},
    ),
    "gamma_coefficients_no_negative_check": (                     # pass-6 Q18 (A5)
        "src/gal3.py",
        "    if theta < 0.0:\n        raise ValueError(\"theta must be non-negative\")\n", "",
        [C24], {C24: "assert not not_rejected, not_rejected"},
    ),
}


def _copy_repo(dst: Path) -> None:
    for name in ("src", "tests"):
        shutil.copytree(ROOT / name, dst / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("conftest.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, dst / name)


def _run(workdir: Path, nodes) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rf", "--tb=short", "-p", "no:cacheprovider",
         *sorted(set(nodes))],
        cwd=workdir, capture_output=True, text=True, timeout=900,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _edits(mid: str):
    """The (file, old, new) edits of a mutation: one, or for a COMPOUND mutant (first
    field a tuple of edits, old/new None) several, each applied exactly once."""
    path, old, new = MUTATIONS[mid][:3]
    return list(path) if isinstance(path, tuple) else [(path, old, new)]


def _mutate_and_run(base: Path, mid: str):
    nodes = MUTATIONS[mid][3]
    work = base / mid.replace("*", "x").replace(">", "").replace("@", "at")
    _copy_repo(work)
    for path, old, new in _edits(mid):
        target = work / path
        text = target.read_text()
        count = text.count(old)
        if count != 1:
            return mid, None, f"mutation target occurs {count} times in {path}"
        target.write_text(text.replace(old, new))
    return (mid, *_run(work, nodes))


@pytest.fixture(scope="module")
def mutation_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("mutants")
    all_nodes = sorted({n for m in MUTATIONS.values() for n in m[3]})
    clean = base / "unmutated"
    _copy_repo(clean)
    # bounded concurrency: one subprocess per mutant at once can exhaust memory on a
    # shared machine; M1_MUTATION_WORKERS overrides the default (the CPU count)
    workers = int(os.environ.get("M1_MUTATION_WORKERS", os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(MUTATIONS) + 1))) as pool:
        baseline = pool.submit(_run, clean, all_nodes)
        futures = [pool.submit(_mutate_and_run, base, mid) for mid in MUTATIONS]
        results = {mid: (code, out) for mid, code, out in (f.result() for f in futures)}
        return baseline.result(), results


@pytest.mark.claim("M1-F01")
def test_unmutated_copy_passes(mutation_results):
    """The harness itself: an unmutated copy passes every test the mutants use."""
    (code, out), _ = mutation_results
    assert code == 0, out[-3000:]


@pytest.mark.claim("M1-F01")
@pytest.mark.parametrize("mid", list(MUTATIONS))
def test_mutation_is_killed(mutation_results, mid):
    """Each mutation makes EVERY one of its named tests fail (pytest exit code 1)."""
    _, results = mutation_results
    code, out = results[mid]
    assert code is not None, out
    assert code == 1, f"{mid}: pytest exit code {code} (expected 1 = tests failed)\n{out[-3000:]}"
    survivors = [n for n in MUTATIONS[mid][3] if f"FAILED {n}" not in out]
    assert not survivors, f"{mid} SURVIVED in {survivors}\n{out[-3000:]}"
    expect = MUTATIONS[mid][4] if len(MUTATIONS[mid]) > 4 else {}
    for node, needle in expect.items():
        section = _failure_section(out, node.split("::")[-1])
        stmt, elines = _failing_statement(section)
        assert elines, f"{mid}: {node} has no 'E ' failure lines\n{section[-3000:]}"
        assert needle in stmt, (
            f"{mid}: {node} failed, but not on `{needle}` (failing statement: {stmt!r})"
            f"\n{section[-3000:]}")


def _failing_statement(section: str) -> tuple[str, list[str]]:
    """The source statement --tb=short prints directly above the FIRST block of 'E '
    lines (the line that raised), and those 'E ' lines.  Matching the needle there,
    not anywhere in the section, rules out a match on echoed source of an assertion
    that did not fail."""
    lines = section.splitlines()
    first_e = next((i for i, ln in enumerate(lines) if ln.startswith("E ")), None)
    if first_e is None:
        return "", []
    start = first_e
    while start > 0 and not (": in " in lines[start - 1] and not lines[start - 1].startswith(" ")):
        start -= 1
    elines = []
    for ln in lines[first_e:]:
        if not ln.startswith("E "):
            break
        elines.append(ln)
    return "\n".join(lines[start:first_e]), elines


def _failure_section(out: str, test_name: str) -> str:
    """The --tb=short failure section of one test: from its '___ name ___' header to
    the next test header or '=====' banner."""
    lines = out.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("_") and f" {test_name} " in line:
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if (lines[j].startswith("___") and lines[j].rstrip().endswith("___")) or \
                lines[j].startswith("====="):
            end = j
            break
    return "\n".join(lines[start:end])
