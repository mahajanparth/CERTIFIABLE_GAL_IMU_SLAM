"""M0-F01 -- the claim suite must FAIL when the conventions table is wrong.

The verifier's mutation run (``docs/verification/m0-literature-matrix.md``, F-6) corrupted
41 fields of ``CONVENTIONS`` / ``INCOMPATIBILITIES`` and 40 of them still passed every test.
This file re-runs that experiment against the rewritten ``tests/test_conventions.py`` and
requires every mutant to be killed.

Mutants are deliberately *plausible*: an equation label is moved to a real neighbouring
equation of the same source (so the format guard ``^\\d+[a-z]?$`` still passes), an ordering
is permuted into another source's ordering, a sign or a coefficient is flipped, a figure or
instance is swapped for one the same paper also reports.  A mutant that only a format check
would catch proves nothing.

Mechanics.  Every ``test_*`` function of ``tests/test_conventions.py`` is called in-process
with one field of the live table replaced; any exception is a kill.  The table is restored
from a deep snapshot after each mutant and the restoration itself is checked.
"""
from __future__ import annotations

import copy
import inspect

import pytest

import tests.test_conventions as suite
from src.conventions import CONVENTIONS, INCOMPATIBILITIES

#: (path, corrupted value, what the corruption models).  The first 37 paths are the
#: survivors listed in F-6; the last 4 are the verifier's four positive-control fields.
MUTANTS: list[tuple[tuple, object, str]] = [
    (("delama", "inverse_formula"), "X^-1 = (A^T, -A^T a, -A^T (b + c a), -c)",
     "one sign inside the position slot"),
    (("delama", "semidirect_inverse_law"), "X^-1 = (A^-1, Ad_{A^-1}[a])", "dropped minus"),
    (("delama", "ctor_order"), ("dR", "dp", "dv", "dt"), "gtsam-style position-first tuple"),
    (("delama", "element_slots"), ("rotation", "position", "velocity", "time"),
     "velocity/position slots swapped"),
    (("delama", "eq", "exp"), "9", "exp moved to the log equation"),
    (("delama", "eq", "left_jacobian"), "11", "neighbouring equation"),
    (("delama", "eq", "endpoint"), "18", "endpoint moved onto its own left factor"),
    (("rai", "eq", "cayley_determinant"), "18", "neighbouring equation"),
    (("rai", "eq", "u_definitions"), "25", "neighbouring equation"),
    (("rai", "eq", "state_set"), "28", "the lifted vector, which also lists R_i, v_i, p_i"),
    (("rai", "eq", "lifting_constraints"), "29", "neighbouring equation"),
    (("rai", "rank1_criterion", "kind"), "singular-value ratio above a relative floor",
     "barfoot's kind of rule"),
    (("rai", "rank1_criterion", "statistic"), "#{lambda_i(Z_hat) > 1e-5 * lambda_max(Z_hat)}",
     "barfoot's floor inside rai's statistic"),
    (("rai", "rank1_criterion", "relation"), ">=", "rank >= 1, true of every nonzero Z"),
    (("rai", "tightness_degrades_with"), ("measurement sparsity",), "barfoot's cause"),
    (("rai", "ablation", "instance"), "synthetic K=16, M=2, n=652",
     "the paper's OTHER synthetic instance"),
    (("barfoot", "eq", "cayley"), "14b", "the inverse Cayley map of the same block"),
    (("barfoot", "eq", "se3_element"), "5", "the SO(3) definition"),
    (("barfoot", "eq", "shor_relaxation"), "19", "the rank-constrained SDP, not its relaxation"),
    (("barfoot", "element_order"), ("r", "C"), "translation-first element"),
    (("barfoot", "rank1_criterion", "kind"), "eigenvalue count above a relative floor",
     "rai's kind of rule"),
    (("barfoot", "rank1_criterion", "relative_floor"), 1e-6, "rai's floor"),
    (("barfoot", "tightness_degrades_evidence"),
     "Sec. 5 discussion prose: the log SVR values are lower in Fig. 9 than Fig. 12 because the "
     "continuous-time experiments used pose measurements only at the start, middle and end "
     "(Holmes and Barfoot, 2023)", "figures swapped"),
    (("gtsam", "retraction"), "X (+) delta == Exp(delta) X  (left perturbation)",
     "left retraction"),
    (("gtsam", "eq", "residual"), "41", "neighbouring equation"),
    (("gtsam", "eq", "rotating_frame"), "17", "the non-rotating endpoint, same shape"),
    (("gtsam", "eq", "adjoint"), "8", "neighbouring equation"),
    (("gtsam", "eq", "held_input_composition"), "12", "the general composition"),
    (("gtsam", "eq", "bias_lift_E"), "24", "the C matrix, which also contains E"),
    (("rai", "state_dim"), 15, "state with the 6 biases counted in"),
    (("barfoot", "state_dim"), 12, "vec(C) + r instead of the tangent dimension"),
    (("gtsam", "state_dim"), 15, "NavState 9 + bias 6"),
    (("O1", "runtime_check"),
     "Gal3(Rot3.Identity(), [1,2,3], [10,20,30], 7.0).matrix() puts [10,20,30] in column "
     "index 4 and [1,2,3] in column index 3", "the swap O1 warns about"),
    (("O2", "P_N_blocks"), "Gal3 rows [0:3, 3:6, 6:9]", "selection without the permutation"),
    (("O3", "a", "eq"), "2", "neighbouring equation"),
    (("O4", "b", "eq"), "8b", "the accelerometer half of the pair"),
    (("O7", "a", "says"),
     "Gamma_ij = [[I, g*dt, +0.5*g*dt**2], [0,1,-dt], [0,0,1]], applied on the LEFT of a "
     "Gal(3) product", "pre-flight defect F2 itself"),
    # -- the verifier's four positive-control fields --------------------------------------
    (("rai", "eq", "rotation_residual"), "21a", "the restatement of the same residual"),
    (("delama", "eq", "inverse"), "4", "neighbouring equation"),
    (("gtsam", "bias_order"), ("b_g", "b_a"), "this project's order instead of gtsam's"),
    (("O7", "a", "gravity_position_coefficient"), +0.5, "gtsam's coefficient"),
]


def _root(path):
    return CONVENTIONS if path[0] in CONVENTIONS else INCOMPATIBILITIES


def _get(path):
    node = _root(path)
    for key in path:
        node = node[key]
    return node


def _set(path, value):
    node = _root(path)
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def _suite_tests():
    return [(name, fn) for name, fn in inspect.getmembers(suite, inspect.isfunction)
            if name.startswith("test_") and fn.__module__ == suite.__name__]


def run_suite(stop_at_first: bool = True) -> list[tuple[str, str]]:
    """Run every test of the claim suite against the live table; return the failures."""
    failures = []
    for name, fn in _suite_tests():
        try:
            fn()
        except Exception as exc:  # any failure of the suite kills the mutant
            failures.append((name, f"{type(exc).__name__}: {str(exc)[:160]}"))
            if stop_at_first:
                break
    return failures


def run_mutant(path, value) -> list[tuple[str, str]]:
    snapshot = copy.deepcopy((CONVENTIONS, INCOMPATIBILITIES))
    original = _get(path)
    try:
        _set(path, value)
        return run_suite()
    finally:
        _set(path, original)
        assert (CONVENTIONS, INCOMPATIBILITIES) == snapshot, "the harness did not restore the table"


# -- positive controls: the harness itself works ---------------------------------------

@pytest.mark.claim("M0-F01")
def test_harness_null_control_unmutated_table_passes_whole_suite():
    """With no mutation the in-process run must be clean -- otherwise every 'kill' below
    could be a harness artefact."""
    assert len(_suite_tests()) >= 16
    assert run_suite(stop_at_first=False) == []


@pytest.mark.claim("M0-F01")
@pytest.mark.parametrize("path", [m[0] for m in MUTANTS[:3]] + [("O7", "reason")])
def test_harness_identity_mutation_survives(path):
    """Writing a field back to an equal copy of itself must NOT be reported as a kill:
    the set/restore mechanics are not what fails the suite."""
    assert run_mutant(path, copy.deepcopy(_get(path))) == []


@pytest.mark.claim("M0-F01")
def test_harness_gross_corruption_is_killed():
    """A corruption every version of the suite caught must still be caught here."""
    assert run_mutant(("rai", "eq", "rotation_residual"), "nonsense")


@pytest.mark.claim("M0-F01")
def test_mutant_list_is_the_verifiers_41():
    paths = [m[0] for m in MUTANTS]
    assert len(paths) == len(set(paths)) == 41
    for path, value, _ in MUTANTS:
        assert _get(path) != value, f"{path} mutant equals the real value"
        assert type(_get(path)) is type(value) or isinstance(_get(path), float), path


# -- the 41 mutants --------------------------------------------------------------------

@pytest.mark.claim("M0-F01")
@pytest.mark.parametrize("path,value,why", MUTANTS, ids=[".".join(map(str, m[0])) for m in MUTANTS])
def test_mutant_is_killed(path, value, why):
    failures = run_mutant(path, value)
    assert failures, (
        f"SURVIVOR: {'.'.join(map(str, path))} = {value!r} ({why}) passes the whole claim suite"
    )
