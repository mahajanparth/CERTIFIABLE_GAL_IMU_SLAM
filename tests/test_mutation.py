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

# id -> (file, old, new, tests that must ALL fail)
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
}


def _copy_repo(dst: Path) -> None:
    for name in ("src", "tests"):
        shutil.copytree(ROOT / name, dst / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("conftest.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, dst / name)


def _run(workdir: Path, nodes) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
         *sorted(set(nodes))],
        cwd=workdir, capture_output=True, text=True, timeout=900,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _mutate_and_run(base: Path, mid: str):
    path, old, new, nodes = MUTATIONS[mid]
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
