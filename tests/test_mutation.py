"""M0-F01 / M0-F06 -- the claim suite must FAIL when the conventions table is wrong.

The verifier's mutation run (``docs/verification/m0-literature-matrix.md``, F-6) corrupted
41 fields of ``CONVENTIONS`` / ``INCOMPATIBILITIES`` and 40 of them still passed every test.
Pass 2 (``docs/verification/m0-literature-matrix_pass2.md``, M0-F06) found 18 more plausible
corruptions that survived.  This file re-runs both experiments against
``tests/test_conventions.py`` and requires every mutant to be killed.

One pass-2 "corruption" turned out to be the true value: ``rai.bias_correction_side ->
"right"``.  Rai Eqs. (14)-(16) apply the bias correction Exp(xi_ij) / Cay(xi_ij) on the RIGHT
of the pre-integrated rotation, so the table's former "none" was the error; the table now
records "right" and the mutants below are the old "none" and "left".  The fields listed in
``src.conventions.UNVERIFIED_COMMENTARY`` are not mutated: no source can decide them.

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

#: Pass 2 (M0-F06): the 18 corruptions that survived the d8257e5 suite, in the report's order.
PASS2_SURVIVORS: list[tuple[tuple, object, str]] = [
    (("rai", "error_handedness"), "left-invariant", "rai has no group error"),
    # the report's mutant "right" IS the source's value (see module docstring); its two
    # alternatives are mutated instead
    (("rai", "bias_correction_side"), "none", "the table's pre-2026-10-12 value"),
    (("rai", "bias_correction_side"), "left", "the other side"),
    (("rai", "eq", "u_definitions"), "26a", "only the velocity half"),
    (("rai", "exact"), ("cayley rotation residual Eq. (20)", "denominator clearing Eq. (18)",
                        "lifted vector Eq. (28)"), "Eq. (20) for the residual"),
    (("rai", "linearized"), ("first-order bias Jacobians A^{vg},A^{va},A^{pg},A^{pa} at bbar in "
                             "the velocity and position residuals Eqs. (21b), (21c)",
                             "cost weighting Eq. (22)"), "Eq. (22) for the weighting"),
    (("delama", "exact"), ("held-input mean propagation Eq. (22)",
                           "gravity left factor Gamma_ij Eq. (17)"), "Eq. (17) for Gamma"),
    (("barfoot", "exact"), ("cayley lifting Eqs. (14), (16)", "homogeneous QCQP Eq. (18)"),
     "Eqs. (14), (16)"),
    (("gtsam", "linearized"), ("transition Eqs. (24)-(29)", "bias correction Eq. (34), first order"),
     "Eqs. (24)-(29)"),
    (("barfoot", "citation"),
     "arXiv:2308.12418v2 [cs.RO], 31 May 2024; IJRR 43(3):366-387, 2024, "
     "DOI 10.1177/02783649241269337 (Barfoot, Holmes, Dumbgen -- Certifiably Optimal Rotation "
     "and Pose Estimation Based on the Cayley Map)", "IJRR 43(3) ... 2024"),
    (("rai", "citation"),
     "arXiv:2609.38048v1 [cs.RO], 30 Sep 2026 (Rai, Xu, Chen, Rosen -- A QCQP-Representable "
     "IMU Pre-Integration Factor for Certifiable State Estimation)", "date 30 Sep 2026"),
    (("rai", "citation"),
     "arXiv:2609.38048v1 [cs.RO], 29 Sep 2026 (Rai, Xu, Chen, Carlone -- A QCQP-Representable "
     "IMU Pre-Integration Factor for Certifiable State Estimation)", "author Carlone for Rosen"),
    (("delama", "citation"),
     "arXiv:2411.05548v8 [cs.RO], 21 Aug 2025; RA-L, DOI 10.1109/LRA.2024.3511424 (Delama, "
     "Fornasier, Mahony, Weiss -- Equivariant IMU Preintegration with Biases: an SE_2(3) Group "
     "Approach)", "title"),
    (("gtsam", "endpoint_rejected_form"), "H_ij X_i D_ij  (gravity factor dropped)",
     "the wrong dropped factor"),
    (("O1", "consequence"), "nothing swaps; the constructor and matrix agree", "denies O1"),
    (("O2", "consequence"), "a selection only; no permutation is needed", "denies O2"),
    (("O4", "consequence"), "the same swap as O3 but on the measurements; the lift matrix E "
                            "(gtsam Eq. 24, R^{10x6}) is exactly this permutation", "Eq. 24 for E"),
    (("O6", "consequence"), "the letters agree with gtsam by luck, not by convention. Ordering "
                            "may be inferred from a letter", "inverts the warning"),
    (("O7", "reason"),
     "not a contradiction: the sign difference is exactly the Galilean t2*v1 coupling term of "
     "the gtsam group product, eq 3. With t2 = dt and v1 = g*dt the left Gal(3) factor "
     "re-contributes +g*dt**2, so the stored entry must be -0.5*g*dt**2 to land on "
     "+0.5*g*dt**2. SE_2(3) has the same coupling.", "last sentence inverted"),
]

#: Fields this pass added or corrected, and the remaining consequence/reason clauses.
NEW_FIELDS: list[tuple[tuple, object, str]] = [
    (("barfoot", "error_handedness"), "none", "the table's pre-2026-10-12 value"),
    (("barfoot", "error_handedness"), "left-invariant", "the other side"),
    (("barfoot", "eq", "rotation_averaging_cost"), "22", "the exp/ln rotation-averaging cost"),
    (("barfoot", "eq", "pose_averaging_cost"), "29", "its lifted QCQP"),
    (("barfoot", "eq", "cayley"), "14", "the whole block, inverse map included"),
    (("rai", "eq", "bias_correction"), "16", "the noise-free relation"),
    (("rai", "eq", "lifting_definitions_Gamma"), "22c", "the Y definitions"),
    (("gtsam", "eq", "transition_first_order"), "26", "the exact error update"),
    (("barfoot", "citation"),
     "arXiv:2308.12418v2 [cs.RO], 31 May 2024; IJRR 44(3):366-387, 2025, "
     "DOI 10.1177/02783649241269338 (Barfoot, Holmes, Dumbgen -- Certifiably Optimal Rotation "
     "and Pose Estimation Based on the Cayley Map)", "DOI off by one"),
    (("gtsam", "citation"),
     "GTSAM docs, 'Galilean IMU preintegration with a right-invariant error', "
     "borglab.github.io/gtsam/galileanimufactor/; behaviour re-verified against the installed "
     "gtsam 4.3.0 wheel", "title"),
    (("O3", "consequence"), "the two 3-blocks swap: accelerometer bias gets applied to the "
                            "accelerometer", "no swap"),
    (("O5", "consequence"), "any constraint template copied from gtsam into a Gal(3) lifting "
                            "lands in the wrong block", "wrong source"),
    (("O7", "consequence"), "using +0.5*g*dt**2 inside a Gal(3) left factor is off by exactly "
                            "-g*dt**2", "sign of the offset"),
    (("O7", "reason"),
     "a contradiction: the sign difference is exactly the Galilean t2*v1 coupling term of "
     "the gtsam group product, eq 3. With t2 = dt and v1 = g*dt the left Gal(3) factor "
     "re-contributes +g*dt**2, so the stored entry must be -0.5*g*dt**2 to land on "
     "+0.5*g*dt**2. SE_2(3) has no such coupling because it carries no time coordinate.",
     "verdict inverted"),
    (("delama", "citation"),
     "arXiv:2411.05548v8 [cs.RO], 21 Aug 2025; RA-L, DOI 10.1109/LRA.2024.3511424 (Delama, "
     "Fornasier, Mahony, Weiss -- Equivariant IMU Preintegration with Biases: a Special Galilean "
     "Group Approach)", "title, citation shape intact"),
    (("O2", "blocks"), ("nu",), "one of the two swapped blocks"),
    (("O1", "b", "says"), "Gal3(R, r, v, t) -- position is the 3rd argument, velocity the 2nd",
     "argument positions swapped"),
    (("O4", "a", "says"), "u = [a; omega]; integrateMeasurement(measuredOmega, measuredAcc, deltaT)",
     "API argument order swapped"),
    (("O7", "b", "says"), "W_ij = NavState(I, +0.5*g*dt**2, g*dt), applied on the RIGHT of an "
                          "SE_2(3) product AFTER the coasting map phi_dt", "wrong side"),
    (("barfoot", "relaxation"), "Shor Eq. (20); tightness == rank(X) == 2", "rank 2"),
]

ALL_MUTANTS = MUTANTS + PASS2_SURVIVORS + NEW_FIELDS


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
def test_mutant_lists_are_complete_and_distinct():
    from src.conventions import UNVERIFIED_COMMENTARY
    assert len({m[0] for m in MUTANTS}) == len(MUTANTS) == 41
    assert len(PASS2_SURVIVORS) == 18 + 1          # bias_correction_side contributes two
    pairs = [(m[0], repr(m[1])) for m in ALL_MUTANTS]
    assert len(pairs) == len(set(pairs))
    for path, value, _ in ALL_MUTANTS:
        assert _get(path) != value, f"{path} mutant equals the real value"
        assert type(_get(path)) is type(value) or isinstance(_get(path), float), path
        assert path not in UNVERIFIED_COMMENTARY, path


# -- the mutants -----------------------------------------------------------------------

@pytest.mark.claim("M0-F01")
@pytest.mark.claim("M0-F06")
@pytest.mark.parametrize("path,value,why", ALL_MUTANTS,
                         ids=[f"{'.'.join(map(str, m[0]))}[{i}]" for i, m in enumerate(ALL_MUTANTS)])
def test_mutant_is_killed(path, value, why):
    failures = run_mutant(path, value)
    assert failures, (
        f"SURVIVOR: {'.'.join(map(str, path))} = {value!r} ({why}) passes the whole claim suite"
    )
