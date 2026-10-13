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
        [C42, C45], {C42: "assert worst_batch <= 1e-12, worst_batch",
                     C45: "assert worst_large <= 1e-12, worst_large"},
    ),
    "so3_exp_batch_taylor_no_th4": (                              # verifier V33 (low)
        "src/imu_dynamics.py",
        "s = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0,",
        "s = np.where(small, 1.0 - t2 / 6.0,",
        [C42], {C42: "assert worst_batch <= 1e-12, worst_batch"},
    ),
    "structural_residual_no_det": (                               # verifier V06 (low)
        "src/gal3.py",
        "        abs(float(np.linalg.det(R)) - 1.0),\n",
        "",
        [C02], {C02: "assert not missed, missed"},
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


def _mutate_and_run(base: Path, mid: str):
    path, old, new, nodes = MUTATIONS[mid][:4]
    work = base / mid.replace("*", "x").replace(">", "").replace("@", "at")
    _copy_repo(work)
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
    with ThreadPoolExecutor(max_workers=len(MUTATIONS) + 1) as pool:
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
        assert needle in section, (
            f"{mid}: {node} failed, but not on `{needle}`\n{section[-3000:]}")


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
