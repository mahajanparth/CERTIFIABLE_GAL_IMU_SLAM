"""Mutation harness -- the suite is tested against ``src``, not the other way round.

A claim test proves that correct code passes.  It does **not** prove that incorrect code
fails, and the two are not the same property.  The Stage-B verifier made that concrete:
it applied 18 single-point mutations to ``src/`` and ran the whole suite against each.
Fourteen died.  **Four survived**, and one of the survivors -- replacing Baseline 2's
Cayley retraction with ``Exp`` -- did worse than survive: it turned a 3-failed suite
*green*.  The suite was rewarding the deletion of the one property that made Baseline 2
a distinct baseline.

This file replays all 18 mutations and asserts, for each, that the tests the verifier
named as its killers really do fail.  It is the regression guard on the guards: if a
future edit relaxes an assertion far enough that a mutant walks through, a test here goes
red and names the mutant and the test that stopped catching it.

Three of the four survivors are now killed, by assertions added in this pass:

====  ==========================================================  =========================
M02   Eq. (9b)/(9c) fold the step rotation into ``dR_ik`` first   ``test_c04`` part (c)
M12   Cayley retraction replaced by ``Exp``                       ``test_c22b``
M15   ``CayleyFactor.xi`` drops the ``bbar_g`` offset             ``test_c13`` part (a)
====  ==========================================================  =========================

The fourth, **M04**, is not killed and cannot be: Eq. (22b) combines ``Y`` and ``Y'`` as
``-(1/2)(Y + Y')``, and a sum is symmetric, so exchanging them is bit-for-bit the
identity.  ``test_m04_swapping_y_and_yprime_is_a_genuine_no_op`` asserts the no-op
*directly* rather than pretending a test catches it -- an undetectable mutation is a fact
about the algebra, and a harness that quietly skipped it would be hiding the same kind of
thing this file exists to expose.

Verification pass 2 ran its own 39-mutant sweep and found **14 more survivors**
(``docs/verification/stage-b_m2-baselines_pass2.md`` Sec. 5): eight covariance defects
that C06's structural checks cannot see, two order errors invisible on the commuting
constant-twist fixture (N24, N11), the dp-only ``dR_{i,k+1}`` fold (F05a3), the sign of
the Eq. (21b) bias terms (N04), a p/v swap in C20's contrast model (N13) and a change of
the fixture bias constants (N29).  ``PASS2_SURVIVORS`` replays all 14 and
``test_pass2_survivor_is_killed`` asserts each now dies; N29 is additionally replayed as
a real on-disk edit by ``test_n29_file_level_bias_edit_is_killed``.

Verification pass 3 (``docs/verification/stage-b_m2-baselines_pass3.md`` Sec. 4, finding
M2-F11) left P17, P35 and P22 gating, plus P33 and P29.  ``PASS3_SURVIVORS`` replays them
(and P22b, a dp-correction error the old c05 metric hid) and
``test_pass3_survivor_is_killed`` checks that each dies on the assertion written for it.

Verification pass 4 (``docs/verification/stage-b_m2-baselines_pass4.md``, finding M2-F12)
found that no test called ``CayleyFactor``'s own rotation-residual methods.
``PASS4_SURVIVORS`` replays Q25, Q45 and Q46 against ``test_c13c``.

Mutations are applied with ``monkeypatch`` against the imported modules rather than by
editing files, so each one is undone automatically and the mutants cannot leak into
another test.  Every mutation patches a *module-level* name or a *class attribute*, which
is what the production call sites look up, so the patch really is in force.
"""
from __future__ import annotations

import numpy as np
import pytest

import test_baselines as suite

from src import trajectory as T
from src import preintegration_cayley as B2
from src import preintegration_forster as B1
from src import preintegration_galilean as B3

from gtsam import Gal3


# ======================================================================================
# The 18 mutations
# ======================================================================================


def _m01_dv_before_dp(monkeypatch):
    """Forster: update ``dv`` before ``dp`` within a step."""
    def integrate_measurement(self, omega_meas, acc_meas, dt):
        w = np.asarray(omega_meas, dtype=float) - self.b_g
        a = np.asarray(acc_meas, dtype=float) - self.b_a
        if self.sigma_g > 0.0 or self.sigma_a > 0.0:
            self._propagate_covariance(w, a, dt)
        dR_ik = self.dR
        self.dv = self.dv + (dR_ik @ a) * dt            # MUTANT: dv first
        self.dp = self.dp + self.dv * dt + 0.5 * (dR_ik @ a) * dt**2
        self.dR = dR_ik @ self.retract(w * dt)
        self.dt_total += dt
    monkeypatch.setattr(B1.ForsterPreintegration, "integrate_measurement",
                        integrate_measurement)


def _m02_fold_rotation_first(monkeypatch):
    """Forster: fold the step's rotation in *before* using ``dR_ik`` for ``dv``/``dp``.

    A real misreading of Rai Eq. (9b)/(9c), which use the *partial* increment
    ``dR_ik``, not ``dR_{i,k+1}``.  Left rectangle rule -> right rectangle rule.
    """
    def integrate_measurement(self, omega_meas, acc_meas, dt):
        w = np.asarray(omega_meas, dtype=float) - self.b_g
        a = np.asarray(acc_meas, dtype=float) - self.b_a
        if self.sigma_g > 0.0 or self.sigma_a > 0.0:
            self._propagate_covariance(w, a, dt)
        self.dR = self.dR @ self.retract(w * dt)        # MUTANT: fold first
        dR_ik = self.dR
        self.dp = self.dp + self.dv * dt + 0.5 * (dR_ik @ a) * dt**2
        self.dv = self.dv + (dR_ik @ a) * dt
        self.dt_total += dt
    monkeypatch.setattr(B1.ForsterPreintegration, "integrate_measurement",
                        integrate_measurement)


def _m03_left_bias_composition(monkeypatch):
    """Forster Eq. (44): LEFT group composition for the rotation correction."""
    def bias_corrected_increment(base_increment, jacobians, db):
        dR0, dv0, dp0 = base_increment
        J_R, J_vg, J_va, J_pg, J_pa = jacobians
        db = np.asarray(db, dtype=float)
        db_g, db_a = db[0:3], db[3:6]
        return (T.so3_exp(J_R @ db_g) @ dR0,             # MUTANT: left
                dv0 + J_vg @ db_g + J_va @ db_a,
                dp0 + J_pg @ db_g + J_pa @ db_a)
    monkeypatch.setattr(B1, "bias_corrected_increment", bias_corrected_increment)


def _m04_swap_y_and_yprime(monkeypatch):
    """Eq. (22b): exchange ``Y`` and ``Y'``.  Structurally a no-op -- see the module
    docstring."""
    def rotation_residual_eq22b(R_i, R_j, dRbar, J_R, b_g, bbar_g):
        R_i, R_j, dRbar, J_R = map(np.asarray, (R_i, R_j, dRbar, J_R))
        Gamma = T.hat(J_R @ np.asarray(bbar_g))
        Rbreve = dRbar @ (np.eye(3) - Gamma / 2.0)
        Xi = T.hat(J_R @ np.asarray(b_g))
        Y = R_j @ Xi                                     # MUTANT: Y and Y' exchanged
        Yp = R_i @ dRbar @ Xi
        return R_j @ (np.eye(3) + Gamma / 2.0) - R_i @ Rbreve - 0.5 * (Y + Yp)
    monkeypatch.setattr(B2, "rotation_residual_eq22b", rotation_residual_eq22b)


def _m05_gamma_sign_flip(monkeypatch):
    """Eq. (22b): sign flip on ``Gamma`` inside ``Rbreve``."""
    def rotation_residual_eq22b(R_i, R_j, dRbar, J_R, b_g, bbar_g):
        R_i, R_j, dRbar, J_R = map(np.asarray, (R_i, R_j, dRbar, J_R))
        Gamma = T.hat(J_R @ np.asarray(bbar_g))
        Rbreve = dRbar @ (np.eye(3) + Gamma / 2.0)       # MUTANT: + instead of -
        Xi = T.hat(J_R @ np.asarray(b_g))
        Y = R_i @ dRbar @ Xi
        Yp = R_j @ Xi
        return R_j @ (np.eye(3) + Gamma / 2.0) - R_i @ Rbreve - 0.5 * (Y + Yp)
    monkeypatch.setattr(B2, "rotation_residual_eq22b", rotation_residual_eq22b)


def _m06_gal3_remap_036(monkeypatch):
    """Gal(3) -> NavState block remap ``(0, 6, 3)`` -> ``(0, 3, 6)``."""
    blocks = (0, 3, 6)

    def gal3_tangent_to_navstate_order(xi):
        xi = np.asarray(xi, dtype=float)
        out = np.empty(9)
        for n, start in enumerate(blocks):
            out[3 * n:3 * n + 3] = xi[start:start + 3]
        return out
    monkeypatch.setattr(B3, "GAL3_BLOCK_FOR_NAVSTATE", blocks)
    monkeypatch.setattr(B3, "gal3_tangent_to_navstate_order",
                        gal3_tangent_to_navstate_order)


def _m07_bias_adapter_no_swap(monkeypatch):
    """GTSAM bias boundary: drop the ``[b_g; b_a]`` <-> ``[b_a; b_g]`` block swap."""
    from gtsam.imuBias import ConstantBias
    monkeypatch.setattr(B3, "to_gtsam_bias",
                        lambda b: ConstantBias(np.asarray(b, float)[0:3],
                                               np.asarray(b, float)[3:6]))
    monkeypatch.setattr(B3, "from_gtsam_bias",
                        lambda cb: np.concatenate([cb.accelerometer(), cb.gyroscope()]))
    monkeypatch.setattr(B3, "swap_bias_blocks", lambda v: np.asarray(v, dtype=float))


def _m08_endpoint_w_drops_t_i(monkeypatch):
    """``W_ij(t_i)``: drop the ``t_i`` term, collapsing it to the pre-flight F2 form."""
    def endpoint_W(Dt, t_i=0.0, g_W=T.G_W):
        g_W = np.asarray(g_W, dtype=float)
        return B3.gal3(np.eye(3), 0.5 * g_W * Dt**2 - g_W * Dt * Dt, g_W * Dt, 0.0)
    monkeypatch.setattr(B3, "endpoint_W", endpoint_W)


def _m09_forward_adjoint(monkeypatch):
    """Bias Jacobian recursion: ``Ad_{V_k}`` instead of ``Ad_{V_k^-1}``.

    This is the planted claim M2-S02, which asserted the substitution changes nothing
    to 1e-10.  It changes the accumulated Jacobian by 9.99e-2 -- the same order as the
    Jacobian itself.
    """
    def bias_jacobian_recursion(gyro, accel, dt, bias_hat, n_steps,
                                eps=B3.C_FD_EPS, **kwargs):
        bias_hat = np.asarray(bias_hat, dtype=float)
        J = np.zeros((10, 6))
        out = []
        for k in range(n_steps):
            Vhat = B3.single_step(gyro[k], accel[k], dt, bias_hat, **kwargs)

            def step_increment(db, _k=k):
                return B3.single_step(gyro[_k], accel[_k], dt,
                                      bias_hat + B3.swap_bias_blocks(db), **kwargs)

            C = B3.bias_jacobian_fd(step_increment, eps=eps)
            J = np.asarray(Vhat.AdjointMap()) @ J + C    # MUTANT: forward adjoint
            out.append(J.copy())
        return out
    monkeypatch.setattr(B3, "bias_jacobian_recursion", bias_jacobian_recursion)


def _m10_jcay_sign_flip(monkeypatch):
    """Rai Eq. (4): sign flip on the skew term of the Cayley right Jacobian."""
    def cay_right_jacobian(phi):
        phi = np.asarray(phi, dtype=float)
        return (np.eye(3) + T.hat(phi) / 2.0) / (1.0 + float(phi @ phi) / 4.0)
    monkeypatch.setattr(B2, "cay_right_jacobian", cay_right_jacobian)


def _m11_isotropic_precision_inverted(monkeypatch):
    """Rai Eq. (23): ``tr(Sigma)/3`` instead of ``3/tr(Sigma)``."""
    monkeypatch.setattr(
        B2, "isotropic_precision",
        lambda S: float(np.trace(np.asarray(S, dtype=float))) / 3.0)


def _m12_cayley_becomes_exp(monkeypatch):
    """**The dangerous one.**  Baseline 2's retraction replaced by ``Exp``.

    Before this pass it survived the whole suite *and* turned it green, because the only
    tests that could see it were the three that were already failing.
    """
    monkeypatch.setattr(B2.CayleyPreintegration, "retract",
                        lambda self, phi: T.so3_exp(phi))


def _m13_chain_drops_half_gravity(monkeypatch):
    """``chain()``: drop the ``1/2`` on the gravity position term."""
    def chain(state, dR, dv, dp, Dt):
        R_i, v_i, p_i = state.as_tuple()
        return T.NavTriple(R=R_i @ dR,
                           v=v_i + T.G_W * Dt + R_i @ dv,
                           p=p_i + v_i * Dt + T.G_W * Dt**2 + R_i @ dp)
    monkeypatch.setattr(T, "chain", chain)


def _m14_m2_closed_form_sign_flip(monkeypatch):
    """``M2(T)``: sign flip on the ``K`` term of the closed form."""
    def M2_(w, Tt):
        w = np.asarray(w, dtype=float)
        th = float(np.linalg.norm(w))
        if th < T.SMALL_ANGLE:
            K = T.hat(w)
            return (0.5 * Tt**2 * np.eye(3) - (Tt**3 / 6.0) * K
                    + (Tt**4 / 24.0) * (K @ K))
        K = T.hat(w) / th
        return (0.5 * Tt**2 * np.eye(3)
                - ((Tt - np.sin(th * Tt) / th) / th) * K       # MUTANT: sign
                + (0.5 * Tt**2 - (1.0 - np.cos(th * Tt)) / th**2) * (K @ K))
    monkeypatch.setattr(T, "M2", M2_)


def _m15_xi_drops_bbar_offset(monkeypatch):
    """``CayleyFactor.xi``: ``J_R b_g`` instead of ``J_R (b_g - bbar_g)``.

    A 10x error in ``xi`` at the fixture's nominal bias, and it used to survive the
    whole suite -- M2-C13's substantive content was exercised by nothing.
    """
    monkeypatch.setattr(B2.CayleyFactor, "xi",
                        lambda self, b_g: self.J_R @ np.asarray(b_g, dtype=float))


def _m16_covariance_drops_transition(monkeypatch):
    """Forster Eq. (63): drop the ``A Sigma A^T`` propagation term."""
    def _propagate_covariance(self, w, a, dt):
        dR_step = self.retract(w * dt)
        Jr = B1.so3_right_jacobian(w * dt)
        dR_ik = self.dR
        Bg = np.zeros((9, 3))
        Bg[0:3, :] = Jr * dt
        Ba = np.zeros((9, 3))
        Ba[3:6, :] = dR_ik * dt
        Ba[6:9, :] = 0.5 * dR_ik * dt**2
        Qg = (self.sigma_g**2 / dt) * np.eye(3)
        Qa = (self.sigma_a**2 / dt) * np.eye(3)
        self.cov = Bg @ Qg @ Bg.T + Ba @ Qa @ Ba.T       # MUTANT: no A P A^T
    monkeypatch.setattr(B1.ForsterPreintegration, "_propagate_covariance",
                        _propagate_covariance)


def _m17_eq19_xi_sign_flip(monkeypatch):
    """Rai Eq. (19): sign flip on the ``xi`` term."""
    def rotation_residual_eq19(R_i, R_j, dRbar, xi):
        A = np.asarray(R_i) @ np.asarray(dRbar)
        R_j = np.asarray(R_j)
        return (R_j - A) + 0.5 * (R_j + A) @ T.hat(xi)   # MUTANT: + instead of -
    monkeypatch.setattr(B2, "rotation_residual_eq19", rotation_residual_eq19)


def _m18_corrected_increment_no_swap(monkeypatch):
    """``corrected_increment_gal3``: drop the bias block swap on ``db``."""
    def corrected_increment_gal3(pim, db_gtsam, Dt=None):
        b = pim.bias_hat + np.asarray(db_gtsam, dtype=float)   # MUTANT: no swap
        from gtsam import NavState
        ns = pim.pim.predict(NavState(), B3.to_gtsam_bias(b))
        return B3.gal3(ns.attitude().matrix(), ns.position(), ns.velocity(),
                       pim.dt_total if Dt is None else Dt)
    monkeypatch.setattr(B3, "corrected_increment_gal3", corrected_increment_gal3)


# ======================================================================================
# The sweep table.  ``killers`` are the claim tests the verifier recorded, plus the
# three added in this pass.  Every one of them must go red.
# ======================================================================================

SURVIVED_BEFORE = "survived the pre-remediation suite"

MUTANTS = [
    ("M01", _m01_dv_before_dp,               ["c10"],                       None),
    ("M02", _m02_fold_rotation_first,        ["c04"],                       SURVIVED_BEFORE),
    ("M03", _m03_left_bias_composition,      ["c05"],                       None),
    # M04 is the genuine no-op; it has its own test below and is not in this table.
    ("M05", _m05_gamma_sign_flip,            ["c11"],                       None),
    ("M06", _m06_gal3_remap_036,             ["c15"],                       None),
    ("M07", _m07_bias_adapter_no_swap,       ["c17", "c19", "c20", "c25"],  None),
    ("M08", _m08_endpoint_w_drops_t_i,       ["c18"],                       None),
    ("M09", _m09_forward_adjoint,            ["c21"],                       None),
    ("M10", _m10_jcay_sign_flip,             ["c07"],                       None),
    ("M11", _m11_isotropic_precision_inverted, ["c12"],                     None),
    ("M12", _m12_cayley_becomes_exp,         ["c22b"],                      SURVIVED_BEFORE),
    ("M13", _m13_chain_drops_half_gravity,   ["c02", "c03", "c18", "c25"],  None),
    ("M14", _m14_m2_closed_form_sign_flip,   ["c02", "c04", "c10", "c19", "c25"], None),
    ("M15", _m15_xi_drops_bbar_offset,       ["c13"],                       SURVIVED_BEFORE),
    ("M16", _m16_covariance_drops_transition, ["c06"],                      None),
    ("M17", _m17_eq19_xi_sign_flip,          ["c09", "c11", "c13"],         None),
    ("M18", _m18_corrected_increment_no_swap, ["c20"],                      None),
]


def _test_function(short_name):
    """``"c22b"`` -> the single ``test_c22b_*`` function in ``test_baselines``."""
    prefix = f"test_{short_name}_"
    matches = [n for n in dir(suite) if n.startswith(prefix)]
    assert len(matches) == 1, f"{short_name!r} matches {matches} in test_baselines"
    return matches[0], getattr(suite, matches[0])


def _outcome(short_name):
    """Run a claim test in-process.  Returns ``None`` if it passed, else the reason."""
    name, fn = _test_function(short_name)
    try:
        fn()
    except AssertionError as exc:
        return f"AssertionError: {(str(exc).splitlines() or ['<no message>'])[0][:160]}"
    except Exception as exc:                                  # noqa: BLE001
        return f"{type(exc).__name__}: {(str(exc).splitlines() or ['<no message>'])[0][:160]}"
    return None


@pytest.mark.claim("M2-F01")
@pytest.mark.parametrize("mutant_id,apply_mutation,killers,note",
                         MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(monkeypatch, mutant_id, apply_mutation, killers, note):
    """Every mutation in the verifier's sweep must make its named tests fail."""
    # the unmutated suite agrees with itself first -- otherwise a test that was already
    # red would be miscounted as a kill
    for short_name in killers:
        assert _outcome(short_name) is None, \
            f"{_test_function(short_name)[0]} fails BEFORE {mutant_id} is applied"

    apply_mutation(monkeypatch)

    survivors = [_test_function(s)[0] for s in killers if _outcome(s) is None]
    assert not survivors, (
        f"{mutant_id} SURVIVED {survivors}"
        + (f" -- and it {note}, so this is a regression in the test suite, not in src"
           if note else ""))


@pytest.mark.claim("M2-F01")
def test_m04_swapping_y_and_yprime_is_a_genuine_no_op(monkeypatch):
    """M04 is not killed, and asserting that honestly is the point.

    Rai Eq. (22b) is ``r^R = R_j(I + Gamma/2) - R_i Rbreve - (1/2)(Y + Y')``.  ``Y`` and
    ``Y'`` enter only through their sum, so exchanging them is the identity map on the
    residual -- not "hard to detect", not "detected weakly", but **bitwise identical**
    for every input.  No assertion anywhere could catch it.

    ``test_c11``'s docstring used to claim "a sign slip or a swapped Y/Y' shows up
    immediately".  The sign-slip half is true and M05 above proves it.  The swap half was
    false; it has been struck from that docstring, and this test records why.
    """
    rng = np.random.default_rng(20250219)
    unmutated = []
    arguments = []
    for _ in range(200):
        args = (T.so3_exp(rng.uniform(-np.pi / 2, np.pi / 2, 3)),
                T.so3_exp(rng.uniform(-np.pi / 2, np.pi / 2, 3)),
                T.so3_exp(rng.uniform(-0.5, 0.5, 3)),
                rng.uniform(-1.0, 1.0, (3, 3)),
                rng.uniform(-0.05, 0.05, 3),
                rng.uniform(-0.05, 0.05, 3))
        arguments.append(args)
        unmutated.append(B2.rotation_residual_eq22b(*args))

    _m04_swap_y_and_yprime(monkeypatch)

    for args, reference in zip(arguments, unmutated):
        mutated = B2.rotation_residual_eq22b(*args)
        assert mutated.tobytes() == reference.tobytes(), (
            "swapping Y and Y' changed the residual -- Eq. (22b) no longer combines them "
            "as -(1/2)(Y + Y'), so the no-op argument in test_c11's docstring is stale")

    # and therefore the test that would notice stays green under the mutation
    assert _outcome("c11") is None, \
        "test_c11 reacts to a swap that is algebraically the identity"


@pytest.mark.claim("M2-F01")
def test_the_sweep_covers_every_mutation_in_the_verifier_report():
    """All 18, none quietly dropped.  A harness that shrinks is a harness that lies."""
    covered = {m[0] for m in MUTANTS} | {"M04"}
    expected = {f"M{k:02d}" for k in range(1, 19)}
    assert covered == expected, f"missing {sorted(expected - covered)}"
    # and the three ex-survivors are explicitly flagged, so nobody has to re-derive which
    # mutations the suite was once blind to
    flagged = {m[0] for m in MUTANTS if m[3] == SURVIVED_BEFORE}
    assert flagged == {"M02", "M12", "M15"}, flagged


@pytest.mark.claim("M2-F01")
def test_replacing_cay_by_exp_cannot_make_the_suite_green(monkeypatch):
    """The specific failure mode this whole file exists for, stated as one assertion.

    The verifier's headline finding was that ``M12`` did not merely survive -- it took
    the suite from ``3 failed, 22 passed`` to ``25 passed``.  A maintainer clearing red
    tests would have been rewarded for deleting Baseline 2.  After this pass, the three
    formerly-red tests are green on the *correct* code and ``Exp`` makes a test red.
    """
    for short_name in ("c22", "c22b", "c23", "c24"):
        assert _outcome(short_name) is None, \
            f"{_test_function(short_name)[0]} is red on unmutated src"

    _m12_cayley_becomes_exp(monkeypatch)

    assert _outcome("c22b") is not None, (
        "substituting Exp for Cay leaves the whole suite green -- Baseline 2's one "
        "distinguishing property is pinned by nothing again")


@pytest.mark.claim("M2-F01")
def test_gal3_import_is_used():
    """``Gal3`` is imported for the recursion mutant; keep the import honest."""
    assert hasattr(Gal3, "Logmap")


# ======================================================================================
# The 14 survivors of the verifier's pass-2 sweep (docs/verification/
# stage-b_m2-baselines_pass2.md Sec. 5), plus three close neighbours named in the M2-F07
# / M2-F09 remediation brief.  Each is replayed exactly as the report describes it.
# ======================================================================================


def _integrate_measurement_variant(rotation_on_left=False, dp_uses_next_rotation=False):
    """Baseline 1's (and so Baseline 2's) ``integrate_measurement`` with one defect."""
    def integrate_measurement(self, omega_meas, acc_meas, dt):
        w = np.asarray(omega_meas, dtype=float) - self.b_g
        a = np.asarray(acc_meas, dtype=float) - self.b_a
        if self.has_noise:
            self._propagate_covariance(w, a, dt)
        dR_ik = self.dR
        step = self.retract(w * dt)
        dR_next = step @ dR_ik if rotation_on_left else dR_ik @ step
        dR_for_dp = dR_next if dp_uses_next_rotation else dR_ik
        self.dp = self.dp + self.dv * dt + 0.5 * (dR_for_dp @ a) * dt**2
        self.dv = self.dv + (dR_ik @ a) * dt
        self.dR = dR_next
        self.dt_total += dt
    return integrate_measurement


def _n24_rotation_accumulated_on_left(monkeypatch):
    """B1/B2: ``dR <- Retract(w dt) dR`` instead of ``dR Retract(w dt)``."""
    monkeypatch.setattr(B1.ForsterPreintegration, "integrate_measurement",
                        _integrate_measurement_variant(rotation_on_left=True))


def _f05a3_dp_only_fold(monkeypatch):
    """Eq. (9c) uses ``dR_{i,k+1}`` in the ``0.5 dR a dt^2`` term; dv keeps ``dR_ik``."""
    monkeypatch.setattr(B1.ForsterPreintegration, "integrate_measurement",
                        _integrate_measurement_variant(dp_uses_next_rotation=True))


def _n04_eq21b_bias_sign(monkeypatch):
    """Eq. (21b): ``+ A^vg db_g + A^va db_a`` instead of ``-``."""
    def velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g_W,
                                A_vg=None, A_va=None, db_g=None, db_a=None):
        r = np.asarray(R_i).T @ (np.asarray(v_j) - np.asarray(v_i) - np.asarray(g_W) * Dt)
        r = r - np.asarray(dvbar)
        if A_vg is not None and db_g is not None:
            r = r + np.asarray(A_vg) @ np.asarray(db_g)          # MUTANT: sign
        if A_va is not None and db_a is not None:
            r = r + np.asarray(A_va) @ np.asarray(db_a)          # MUTANT: sign
        return r
    monkeypatch.setattr(B2, "velocity_residual_eq21b", velocity_residual_eq21b)


def _n04p_eq21c_bias_sign(monkeypatch):
    """Eq. (21c): ``+ A^pg db_g + A^pa db_a`` instead of ``-`` (neighbour of N04)."""
    def position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g_W,
                                A_pg=None, A_pa=None, db_g=None, db_a=None):
        r = np.asarray(R_i).T @ (np.asarray(p_j) - np.asarray(p_i)
                                 - np.asarray(v_i) * Dt - 0.5 * np.asarray(g_W) * Dt**2)
        r = r - np.asarray(dpbar)
        if A_pg is not None and db_g is not None:
            r = r + np.asarray(A_pg) @ np.asarray(db_g)          # MUTANT: sign
        if A_pa is not None and db_a is not None:
            r = r + np.asarray(A_pa) @ np.asarray(db_a)          # MUTANT: sign
        return r
    monkeypatch.setattr(B2, "position_residual_eq21c", position_residual_eq21c)


def _covariance_variant(defect):
    """Forster Eq. (63) as in ``src``, with exactly one named defect."""
    def _propagate_covariance(self, w, a, dt):
        dR_step = self.retract(w * dt)
        Jr = B1.so3_right_jacobian(w * dt)
        dR_ik = self.dR
        A = np.zeros((9, 9))
        A[0:3, 0:3] = dR_step.T
        A[3:6, 0:3] = -dR_ik @ T.hat(a) * dt
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 0:3] = -0.5 * dR_ik @ T.hat(a) * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 6:9] = np.eye(3)
        Bg = np.zeros((9, 3))
        Bg[0:3, :] = Jr * dt
        Ba = np.zeros((9, 3))
        Ba[3:6, :] = dR_ik * dt
        Ba[6:9, :] = 0.5 * dR_ik * dt**2
        Qg, Qa = self.noise_covariances(dt)
        if defect == "coupling_sign":          # N27
            A[3:9, 0:3] = -A[3:9, 0:3]
        elif defect == "coupling_sign_v":      # N27v: velocity row only
            A[3:6, 0:3] = -A[3:6, 0:3]
        elif defect == "jr_identity":          # N05
            Bg[0:3, :] = np.eye(3) * dt
        elif defect == "a00_identity":         # N06
            A[0:3, 0:3] = np.eye(3)
        elif defect == "a00_untransposed":     # N10
            A[0:3, 0:3] = dR_step
        elif defect == "ba_position_no_half":  # N07
            Ba[6:9, :] = dR_ik * dt**2
        elif defect == "a_position_no_half":   # N09
            A[6:9, 0:3] = -dR_ik @ T.hat(a) * dt**2
        elif defect == "ba_transposed":        # N28
            Ba[3:6, :] = dR_ik.T * dt
            Ba[6:9, :] = 0.5 * dR_ik.T * dt**2
        elif defect == "noise_up":             # N08
            Qg, Qa = 1.005 * Qg, 1.005 * Qa
        elif defect == "noise_down":           # N08d
            Qg, Qa = 0.995 * Qg, 0.995 * Qa
        else:
            raise ValueError(defect)
        self.cov = A @ self.cov @ A.T + Bg @ Qg @ Bg.T + Ba @ Qa @ Ba.T
    return _propagate_covariance


def _covariance_mutant(defect):
    def apply(monkeypatch):
        monkeypatch.setattr(B1.ForsterPreintegration, "_propagate_covariance",
                            _covariance_variant(defect))
    apply.__doc__ = f"Forster Eq. (63) covariance defect: {defect}."
    return apply


def _n11_conjugated_j_r(monkeypatch):
    """``J_R`` replaced by ``dR J_R dR^T`` (commutes with ``dR`` on constant twist)."""
    original = B1.bias_jacobians_fd

    def bias_jacobians_fd(edge, bias_hat, eps=B1.FD_EPS, cls=B1.ForsterPreintegration,
                          **kwargs):
        J_R, J_vg, J_va, J_pg, J_pa = original(edge, bias_hat, eps=eps, cls=cls, **kwargs)
        dR0 = B1.integrate_edge(edge, bias_hat, cls=cls, **kwargs).dR
        return dR0 @ J_R @ dR0.T, J_vg, J_va, J_pg, J_pa
    monkeypatch.setattr(B1, "bias_jacobians_fd", bias_jacobians_fd)


def _n13_contrast_pv_swapped(monkeypatch):
    """M2-C20's component-wise contrast model with its p and v tangent blocks exchanged."""
    def component_wise_model(Uhat, J, db):
        J = np.asarray(J, dtype=float)
        xi = J @ np.asarray(db, dtype=float)
        R, p, v, t = B3.gal3_components(Uhat)
        return B3.gal3(R @ T.so3_exp(xi[0:3]), p + xi[3:6], v + xi[6:9], t)  # MUTANT
    monkeypatch.setattr(B3, "component_wise_model", component_wise_model)


def _n29_bias_constants_changed(monkeypatch):
    """``B_G -> [0.02, -0.015, 0.008]``, ``B_A -> [0.06, -0.04, 0.19]``, consistently, as
    a file-level edit of ``src/trajectory.py`` would leave them."""
    b_g = np.array([0.0200, -0.0150, 0.0080])
    b_a = np.array([0.0600, -0.0400, 0.1900])
    monkeypatch.setattr(T, "B_G", b_g)
    monkeypatch.setattr(T, "B_A", b_a)
    monkeypatch.setattr(T, "BIAS_TRUE", np.concatenate([b_g, b_a]))


#: ``(id, mutation, killers)``; every killer must go red.  ``killers`` lists the tests
#: whose assertions were written for the mutant; others may die too.
PASS2_SURVIVORS = [
    ("N24",   _n24_rotation_accumulated_on_left,         ["c05b"]),
    ("F05a3", _f05a3_dp_only_fold,                       ["c04", "c05b"]),
    ("N04",   _n04_eq21b_bias_sign,                      ["c10b"]),
    ("N27",   _covariance_mutant("coupling_sign"),       ["c06b"]),
    ("N05",   _covariance_mutant("jr_identity"),         ["c06b"]),
    ("N06",   _covariance_mutant("a00_identity"),        ["c06b"]),
    ("N10",   _covariance_mutant("a00_untransposed"),    ["c06b"]),
    ("N07",   _covariance_mutant("ba_position_no_half"), ["c06b"]),
    ("N09",   _covariance_mutant("a_position_no_half"),  ["c06b"]),
    ("N28",   _covariance_mutant("ba_transposed"),       ["c06b"]),
    ("N08",   _covariance_mutant("noise_up"),            ["c06b"]),
    ("N11",   _n11_conjugated_j_r,                       ["c05b"]),
    ("N13",   _n13_contrast_pv_swapped,                  ["c20"]),
    ("N29",   _n29_bias_constants_changed,               ["c01"]),
]
#: neighbours from the remediation brief, not in the verifier's table
PASS2_NEIGHBOURS = [
    ("N04p",  _n04p_eq21c_bias_sign,                     ["c10b"]),
    ("N27v",  _covariance_mutant("coupling_sign_v"),     ["c06b"]),
    ("N08d",  _covariance_mutant("noise_down"),          ["c06b"]),
]


@pytest.mark.claim("M2-F07")
@pytest.mark.claim("M2-F08")
@pytest.mark.claim("M2-F09")
@pytest.mark.claim("M2-F10")
@pytest.mark.parametrize("mutant_id,apply_mutation,killers",
                         PASS2_SURVIVORS + PASS2_NEIGHBOURS,
                         ids=[m[0] for m in PASS2_SURVIVORS + PASS2_NEIGHBOURS])
def test_pass2_survivor_is_killed(monkeypatch, mutant_id, apply_mutation, killers):
    """Every pass-2 survivor must now turn its named tests red."""
    for short_name in killers:
        assert _outcome(short_name) is None, \
            f"{_test_function(short_name)[0]} fails BEFORE {mutant_id} is applied"
    apply_mutation(monkeypatch)
    survivors = [_test_function(s)[0] for s in killers if _outcome(s) is None]
    assert not survivors, f"{mutant_id} (survived pass 2) still SURVIVES {survivors}"


@pytest.mark.claim("M2-F07")
def test_pass2_sweep_covers_all_fourteen_survivors():
    """The 14 ids of the pass-2 report's survivor table, none dropped."""
    expected = {"N24", "F05a3", "N04", "N27", "N05", "N06", "N10", "N07", "N09", "N28",
                "N08", "N11", "N13", "N29"}
    assert {m[0] for m in PASS2_SURVIVORS} == expected
    assert len(PASS2_SURVIVORS) == 14


def _run_c01_in_copy(tmp_path, edit=None):
    """Run ``test_c01`` against a COPY of the tree, optionally with ``src/trajectory.py``
    edited on disk.  This is a true file-level mutation: the generating subprocesses of
    ``test_c01`` import the edited file, which a monkeypatch cannot reach."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / ("mutated" if edit else "control")
    shutil.copytree(repo / "src", root / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "tests").mkdir(parents=True)
    shutil.copy(repo / "tests" / "test_baselines.py", root / "tests")
    shutil.copy(repo / "conftest.py", root)
    shutil.copy(repo / "pyproject.toml", root)
    if edit:
        path = root / "src" / "trajectory.py"
        text = path.read_text()
        for old, new in edit:
            assert text.count(old) == 1, f"edit anchor {old!r} not unique"
            text = text.replace(old, new)
        path.write_text(text)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_baselines.py::test_c01_fixture_is_deterministic_bitwise"],
        cwd=str(root), capture_output=True, text=True)
    return proc.returncode, proc.stdout


@pytest.mark.claim("M2-F10")
def test_n29_file_level_bias_edit_is_killed(tmp_path):
    """N29 exactly as the verifier applied it: an on-disk edit of ``B_G`` and ``B_A``.

    Before this pass every baseline subtracted the same edited ``BIAS_TRUE`` and
    ``test_c01`` compared the streams against the module's own constants, so the edit
    passed.  ``test_c01`` now checks the generated streams and constants against literals
    retyped from the M2-C01 statement.
    """
    code, out = _run_c01_in_copy(tmp_path)
    assert code == 0, f"control copy of test_c01 fails:\n{out[-2000:]}"
    code, out = _run_c01_in_copy(tmp_path, edit=[
        ("B_G = np.array([0.0100, -0.0150, 0.0080])",
         "B_G = np.array([0.0200, -0.0150, 0.0080])"),
        ("B_A = np.array([0.0600, -0.0400, 0.0900])",
         "B_A = np.array([0.0600, -0.0400, 0.1900])"),
    ])
    assert code != 0, "a file-level edit of B_G/B_A leaves test_c01 green"
    assert "is not the claimed [b_g; b_a]" in out, \
        f"test_c01 died, but not on the bias pin:\n{out[-2000:]}"


# ======================================================================================
# Survivors of the verifier's pass-3 sweep (docs/verification/stage-b_m2-baselines_pass3.md
# Sec. 4, finding M2-F11).  Each kill is checked against the ASSERTION written for it: the
# failure message must contain the given text, so a mutant that dies of something
# incidental is not counted as pinned.
# ======================================================================================


def _p17_has_noise_and(monkeypatch):
    """``has_noise``: ``or`` -> ``and``.  Eq. (63) is skipped when only one density is
    nonzero, and the covariance is silently zero."""
    monkeypatch.setattr(
        B1.ForsterPreintegration, "has_noise",
        property(lambda self: bool(np.any(np.asarray(self.sigma_g) > 0.0)
                                   and np.any(np.asarray(self.sigma_a) > 0.0))))


def _p35_cayley_factor_takes_forster_jacobians(monkeypatch):
    """``CayleyFactor.from_edge``: ``J_R`` and ``A`` from the Forster (Exp) integrator."""
    def from_edge(cls, edge, bbar, variant=B2.DEFAULT_VARIANT):
        bbar = np.asarray(bbar, dtype=float)
        pim = B1.integrate_edge(edge, bbar, cls=B2.CayleyPreintegration, variant=variant)
        J_R, A_vg, A_va, A_pg, A_pa = B1.bias_jacobians_fd(
            edge, bbar, cls=B1.ForsterPreintegration)                 # MUTANT
        return cls(bbar=bbar, dRbar=pim.dR, dvbar=pim.dv, dpbar=pim.dp,
                   Dt=pim.dt_total, J_R=J_R, A_vg=A_vg, A_va=A_va,
                   A_pg=A_pg, A_pa=A_pa, variant=variant)
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", classmethod(from_edge))


def _p22_increment_error_ignores_dp(monkeypatch):
    """``increment_error``: the dp term dropped from the max."""
    def increment_error(a, b):
        dRa, dva, _ = a
        dRb, dvb, _ = b
        return max(float(np.abs(T.so3_log(dRb.T @ dRa)).max()),
                   float(np.abs(dva - dvb).max()))
    monkeypatch.setattr(B1, "increment_error", increment_error)


def _p22b_dp_correction_slightly_wrong(monkeypatch):
    """Eq. (44) dp correction with ``J_pg`` scaled by ``1 + 3e-4``: a FIRST-order dp error
    that the max over components hides behind dv's second-order residual (the old c05
    ratio stays 100.01 on both edges; the dp-only ratio drops to 28)."""
    original = B1.bias_corrected_increment

    def bias_corrected_increment(base_increment, jacobians, db):
        J_R, J_vg, J_va, J_pg, J_pa = jacobians
        return original(base_increment,
                        (J_R, J_vg, J_va, (1.0 + 3e-4) * J_pg, J_pa), db)   # MUTANT
    monkeypatch.setattr(B1, "bias_corrected_increment", bias_corrected_increment)


def _p33_weighted_square_unsquared(monkeypatch):
    """``weighted_square(r)`` returns ``||r||``, not ``||r||^2``."""
    def weighted_square(r, W=None):
        r = np.asarray(r, dtype=float)
        if W is None:
            return float(np.sqrt(r @ r))                              # MUTANT
        return float(r @ np.asarray(W) @ r)
    monkeypatch.setattr(B2, "weighted_square", weighted_square)


def _p29_truncated_contrast_sign(monkeypatch):
    """M2-C13's truncated contrast built on ``(I - hat(xi))``."""
    def rotation_residual_truncated(R_i, R_j, dRbar, xi):
        A = np.asarray(R_i) @ np.asarray(dRbar)
        return np.asarray(R_j) - A @ (np.eye(3) - T.hat(xi))         # MUTANT
    monkeypatch.setattr(B2, "rotation_residual_truncated", rotation_residual_truncated)


#: ``(id, mutation, killer, text the killing assertion's message must contain)``
PASS3_SURVIVORS = [
    ("P17",  _p17_has_noise_and,                         "c06c",
     "the has_noise gate skipped Eq. (63)"),
    ("P35",  _p35_cayley_factor_takes_forster_jacobians, "c13b",
     "CayleyFactor's frozen Jacobians are not the Cayley integrator's {'J_R'"),
    ("P22",  _p22_increment_error_ignores_dp,            "c05",
     "increment_error on a dp-only difference"),
    ("P22b", _p22b_dp_correction_slightly_wrong,         "c05",
     "the dp correction is not quadratic"),
    ("P33",  _p33_weighted_square_unsquared,             "c12",
     "expected 1 + 4 + 9 = 14"),
    ("P29",  _p29_truncated_contrast_sign,               "c13",
     "truncated contrast at the identity is not -hat(xi)"),
]


@pytest.mark.claim("M2-F11")
@pytest.mark.parametrize("mutant_id,apply_mutation,killer,message", PASS3_SURVIVORS,
                         ids=[m[0] for m in PASS3_SURVIVORS])
def test_pass3_survivor_is_killed(monkeypatch, mutant_id, apply_mutation, killer, message):
    """Every pass-3 survivor must fail its killer, on the assertion written for it."""
    assert _outcome(killer) is None, \
        f"{_test_function(killer)[0]} fails BEFORE {mutant_id} is applied"
    apply_mutation(monkeypatch)
    outcome = _outcome(killer)
    assert outcome is not None, \
        f"{mutant_id} (survived pass 3) still SURVIVES {_test_function(killer)[0]}"
    assert outcome.startswith("AssertionError") and message in outcome, (
        f"{mutant_id} kills {_test_function(killer)[0]}, but not on the intended "
        f"assertion: {outcome}")


# ======================================================================================
# Survivors of the verifier's pass-4 sweep (docs/verification/stage-b_m2-baselines_pass4.md
# Sec. 4, finding M2-F12): the factor-level rotation residuals were called by no test.
# Same rule as pass 3 -- the kill must land on the assertion written for it.
# ======================================================================================


def _q25_affine_residual_takes_accel_offset(monkeypatch):
    """``CayleyFactor.rotation_residual_affine`` passes ``bbar_a`` as ``bbar_g``."""
    def rotation_residual_affine(self, R_i, R_j, b_g):
        return B2.rotation_residual_eq22b(R_i, R_j, self.dRbar, self.J_R,
                                          b_g, self.bbar[3:6])            # MUTANT
    monkeypatch.setattr(B2.CayleyFactor, "rotation_residual_affine",
                        rotation_residual_affine)


def _q45_rotation_residual_negated_xi(monkeypatch):
    """``CayleyFactor.rotation_residual`` evaluates Eq. (19) at ``-xi``."""
    def rotation_residual(self, R_i, R_j, b_g):
        return B2.rotation_residual_eq19(R_i, R_j, self.dRbar, -self.xi(b_g))  # MUTANT
    monkeypatch.setattr(B2.CayleyFactor, "rotation_residual", rotation_residual)


def _q46_rotation_residual_transposed_dRbar(monkeypatch):
    """``CayleyFactor.rotation_residual`` uses ``dRbar^T``."""
    def rotation_residual(self, R_i, R_j, b_g):
        return B2.rotation_residual_eq19(R_i, R_j, np.asarray(self.dRbar).T,
                                         self.xi(b_g))                     # MUTANT
    monkeypatch.setattr(B2.CayleyFactor, "rotation_residual", rotation_residual)


#: ``(id, mutation, killer, text the killing assertion's message must contain)``
PASS4_SURVIVORS = [
    ("Q25", _q25_affine_residual_takes_accel_offset,  "c13c",
     "rotation_residual_affine is"),
    ("Q45", _q45_rotation_residual_negated_xi,        "c13c",
     "CayleyFactor.rotation_residual is"),
    ("Q46", _q46_rotation_residual_transposed_dRbar,  "c13c",
     "CayleyFactor.rotation_residual is"),
]


@pytest.mark.claim("M2-F12")
@pytest.mark.parametrize("mutant_id,apply_mutation,killer,message", PASS4_SURVIVORS,
                         ids=[m[0] for m in PASS4_SURVIVORS])
def test_pass4_survivor_is_killed(monkeypatch, mutant_id, apply_mutation, killer, message):
    """Every pass-4 survivor must fail its killer, on the assertion written for it."""
    assert _outcome(killer) is None, \
        f"{_test_function(killer)[0]} fails BEFORE {mutant_id} is applied"
    apply_mutation(monkeypatch)
    outcome = _outcome(killer)
    assert outcome is not None, \
        f"{mutant_id} (survived pass 4) still SURVIVES {_test_function(killer)[0]}"
    assert outcome.startswith("AssertionError") and message in outcome, (
        f"{mutant_id} kills {_test_function(killer)[0]}, but not on the intended "
        f"assertion: {outcome}")
