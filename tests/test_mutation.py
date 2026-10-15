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

Verification pass 5 (``docs/verification/stage-b_m2-baselines_pass5.md`` Sec. 3, finding
M2-F13) found the factor's frozen rotation ``dRbar`` pinned by nothing.
``PASS5_SURVIVORS`` replays R01, R02, R04, R05 and R10 against ``test_c13d``, twice: once
as written (killed by the direct ``dRbar`` pin) and once with that pin disabled (killed by
the end-to-end Eq. (19) residual at ground truth).

Verification pass 6 (``docs/verification/stage-b_m2-baselines_pass6.md`` Sec. 4) found
that every ``from_edge`` call passed ``bbar = BIAS_TRUE`` (M2-F14: S46, S47, S48 leak the
fixture bias; S12, S13 corrupt ``bbar[3:6]``) and that Baseline 3's ``sigma_int`` /
``sigma_bg`` / ``sigma_ba`` were pinned by nothing (M2-F15: S35, S36).  Those seven are
replayed as REAL FILE EDITS in a scratch copy (``PASS6_F14_SURVIVORS``,
``PASS6_F15_SURVIVORS``) against ``test_c13e`` / ``test_c05c`` / ``test_c25b``.

Verification pass 7 (``docs/verification/stage-b_m2-baselines_pass7.md``) found the same
class a seventh time -- an input no test moves off its fixture value (M2-F16, M2-F17 and
the fixture-degenerate U-series) -- and that the file-level kill check matched the echoed
test source (item 15).  ``PASS7_MUTANTS`` replays its survivors, and the same edit at
every other input ``tests/test_input_dependence.py`` varies, as file edits; every
file-level kill is now matched on pytest's ``E `` lines only (M2-F19).

Verification pass 8 (``docs/verification/stage-b_m2-baselines_pass8.md``) found the
self-referential F18 field list (W07, W08), unpinned outward aliasing (W09, S39), inputs
still unvaried (W02, W13, W14, W46, U47, W10), small-angle branches no reference could
resolve (Q01, Q39, W15, W17-W20, Q44), and two latent ``src`` defects: a Forster
covariance frozen at zero densities (M2-F23) and ``so3_log`` near ``pi`` (M2-F24).
``PASS8_MUTANTS`` replays them -- and both defects -- as file edits, each killed
by a check against an INDEPENDENT reference on its intended ``E `` line; none relies on
a change detector (``test_pass5_replay_is_faithful`` or a pass-6/7 edit anchor).  Q44 is
no longer replayed: pass 9 ruled it equivalent and its ulp check was removed.

Verification pass 9 (``docs/verification/stage-b_m2-baselines_pass9.md``) found Baseline
2's covariance using Exp's right Jacobian (M2-F26) and nine survivors no reference could
resolve (M2-F27: X15, X21, X22, X24, X31, X41, W12, P23, S21).  ``PASS9_MUTANTS``
replays all of them and the defect as file edits, killed by complex-step, ``mpmath``
quadrature and retyped-recursion references.

Mutations before pass 6 are applied with ``monkeypatch`` against the imported modules
rather than by editing files, so each one is undone automatically and the mutants cannot leak into
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
        if self.has_noise or np.any(self.cov != 0.0):
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


# ======================================================================================
# Survivors of the verifier's pass-5 sweep (docs/verification/stage-b_m2-baselines_pass5.md
# Sec. 3, finding M2-F13): the factor's frozen rotation ``dRbar`` was pinned by no test.
# Each is replayed as a ``from_edge`` that stores a wrong ``dRbar`` and nothing else.
# ======================================================================================


def _from_edge_with_dRbar(make_dRbar):
    """A ``CayleyFactor.from_edge`` identical to src's except that ``dRbar`` is
    ``make_dRbar(edge, bbar, pim)``."""
    def from_edge(cls, edge, bbar, variant=B2.DEFAULT_VARIANT):
        bbar = np.asarray(bbar, dtype=float)
        pim = B1.integrate_edge(edge, bbar, cls=B2.CayleyPreintegration, variant=variant)
        J_R, A_vg, A_va, A_pg, A_pa = B1.bias_jacobians_fd(
            edge, bbar, cls=B2.CayleyPreintegration, variant=variant)
        return cls(bbar=bbar, dRbar=make_dRbar(edge, bbar, pim), dvbar=pim.dv,
                   dpbar=pim.dp, Dt=pim.dt_total, J_R=J_R, A_vg=A_vg, A_va=A_va,
                   A_pg=A_pg, A_pa=A_pa, variant=variant)
    return classmethod(from_edge)


def _r01_dRbar_transposed(monkeypatch):
    """``from_edge``: ``dRbar = pim.dR.T``."""
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: pim.dR.T))                                 # MUTANT


def _r02_dRbar_from_exp_integrator(monkeypatch):
    """``from_edge``: ``dRbar`` from the Forster (Exp) integrator."""
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: B1.integrate_edge(edge, bbar).dR))         # MUTANT


def _r04_dRbar_at_zero_gyro_bias(monkeypatch):
    """``from_edge``: ``dRbar`` integrated at ``b_g = 0``."""
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: B1.integrate_edge(
            edge, np.concatenate([np.zeros(3), bbar[3:6]]),
            cls=B2.CayleyPreintegration).dR))                              # MUTANT


def _r05_dRbar_from_reparam_variant(monkeypatch):
    """``from_edge``: ``dRbar`` from ``VARIANT_REPARAM`` (numerically Exp)."""
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: B1.integrate_edge(
            edge, bbar, cls=B2.CayleyPreintegration,
            variant=B2.VARIANT_REPARAM).dR))                               # MUTANT


def _r10_dRbar_times_small_cay(monkeypatch):
    """``from_edge``: ``dRbar Cay(1e-6 e1)``."""
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: pim.dR @ B2.cay(np.array([1e-6, 0.0, 0.0]))))  # MUTANT


#: ``(id, mutation, killer, text the killing assertion's message must contain)``
PASS5_SURVIVORS = [
    ("R01", _r01_dRbar_transposed,           "c13d",
     "CayleyFactor.dRbar is not the Cayley increment"),
    ("R02", _r02_dRbar_from_exp_integrator,  "c13d",
     "CayleyFactor.dRbar is not the Cayley increment"),
    ("R04", _r04_dRbar_at_zero_gyro_bias,    "c13d",
     "CayleyFactor.dRbar is not the Cayley increment"),
    ("R05", _r05_dRbar_from_reparam_variant, "c13d",
     "CayleyFactor.dRbar is not the Cayley increment"),
    ("R10", _r10_dRbar_times_small_cay,      "c13d",
     "CayleyFactor.dRbar is not the Cayley increment"),
]


@pytest.mark.claim("M2-F13")
def test_pass5_replay_is_faithful(monkeypatch):
    """The replayed ``from_edge`` with the CORRECT ``dRbar`` must be src's ``from_edge``
    field for field, and must pass c13d -- otherwise the pass-5 kills below could be
    killing a difference in the replay rather than in ``dRbar``."""
    fields = ("bbar", "dRbar", "dvbar", "dpbar", "Dt", "J_R", "A_vg", "A_va", "A_pg", "A_pa")
    src_factors = [B2.CayleyFactor.from_edge(e, T.BIAS_TRUE) for e in T.edges()]
    monkeypatch.setattr(B2.CayleyFactor, "from_edge", _from_edge_with_dRbar(
        lambda edge, bbar, pim: pim.dR))
    for edge, ref in zip(T.edges(), src_factors):
        got = B2.CayleyFactor.from_edge(edge, T.BIAS_TRUE)
        for f in fields:
            assert np.array_equal(getattr(got, f), getattr(ref, f)), \
                f"edge {edge.index}: replayed from_edge differs from src's in {f}"
        assert got.variant == ref.variant
    assert _outcome("c13d") is None, "the faithful replay fails c13d"


@pytest.mark.claim("M2-F13")
@pytest.mark.parametrize("mutant_id,apply_mutation,killer,message", PASS5_SURVIVORS,
                         ids=[m[0] for m in PASS5_SURVIVORS])
def test_pass5_survivor_is_killed(monkeypatch, mutant_id, apply_mutation, killer, message):
    """Every pass-5 survivor must fail its killer, on the assertion written for it."""
    assert _outcome(killer) is None, \
        f"{_test_function(killer)[0]} fails BEFORE {mutant_id} is applied"
    apply_mutation(monkeypatch)
    outcome = _outcome(killer)
    assert outcome is not None, \
        f"{mutant_id} (survived pass 5) still SURVIVES {_test_function(killer)[0]}"
    assert outcome.startswith("AssertionError") and message in outcome, (
        f"{mutant_id} kills {_test_function(killer)[0]}, but not on the intended "
        f"assertion: {outcome}")


@pytest.mark.claim("M2-F13")
@pytest.mark.parametrize("mutant_id,apply_mutation,killer,message", PASS5_SURVIVORS,
                         ids=[m[0] for m in PASS5_SURVIVORS])
def test_pass5_survivor_is_killed_end_to_end(monkeypatch, mutant_id, apply_mutation,
                                             killer, message):
    """With the direct ``dRbar`` pin disabled, c13d's end-to-end check -- the factor's
    Eq. (19) residual at ground truth equals the Cayley deficit, two-sided -- must kill
    every pass-5 survivor on its own."""
    monkeypatch.setattr(suite, "DRBAR_PIN_BOUND", np.inf)
    assert _outcome(killer) is None, \
        f"{_test_function(killer)[0]} fails BEFORE {mutant_id} is applied"
    apply_mutation(monkeypatch)
    outcome = _outcome(killer)
    assert outcome is not None, \
        f"{mutant_id} SURVIVES {_test_function(killer)[0]} with the dRbar pin disabled"
    assert outcome.startswith("AssertionError") and \
        "Eq. (19) residual at ground truth" in outcome, (
            f"{mutant_id} kills {_test_function(killer)[0]} end-to-end, but not on the "
            f"Eq. (19) ground-truth assertion: {outcome}")


# ======================================================================================
# Survivors of the verifier's pass-6 sweep (docs/verification/stage-b_m2-baselines_pass6.md
# Sec. 4, findings M2-F14 and M2-F15), replayed as REAL FILE EDITS: each mutant is a
# string edit of a ``src`` file in a scratch copy of the tree, and the killer tests run in
# a subprocess that imports the edited file.  Each kill is checked against the assertion
# written for it (the failure output must contain the given text).
# ======================================================================================


def _run_in_copy(root, nodeids, src_file=None, edit=None):
    """Copy ``src``, the two claim-test modules (``tests/test_baselines.py``,
    ``tests/test_input_dependence.py``), ``conftest.py`` and ``pyproject.toml`` into
    ``root``, optionally apply ``edit`` (``[(old, new), ...]``, each anchor unique) to
    ``src/<src_file>``, and run ``nodeids`` there.  A node id without ``::`` names a test
    in ``test_baselines.py``; otherwise it is a full ``tests/<file>::<test>`` id."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "src", root / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "tests").mkdir(parents=True)
    for name in ("test_baselines.py", "test_input_dependence.py"):
        shutil.copy(repo / "tests" / name, root / "tests")
    shutil.copy(repo / "conftest.py", root)
    shutil.copy(repo / "pyproject.toml", root)
    if edit:
        path = root / "src" / src_file
        text = path.read_text()
        for old, new in edit:
            assert text.count(old) == 1, f"edit anchor {old!r} not unique in {src_file}"
            text = text.replace(old, new)
        path.write_text(text)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + [n if "::" in n else f"tests/test_baselines.py::{n}" for n in nodeids],
        cwd=str(root), capture_output=True, text=True)
    return proc.returncode, proc.stdout


def _failure_lines(out):
    """Only pytest's ``E `` lines: the exception message and the repr of the failing
    expression.  pytest also ECHOES the failing test's source, so a message typed as a
    literal in the test is "present" in the raw output whichever assertion failed
    (verification pass 7, item 15 / M2-F19); matching only ``E `` lines removes that."""
    return "\n".join(line for line in out.splitlines() if line.startswith("E "))


_C25B = "test_c25b_baseline3_noise_model_is_built_from_the_stated_densities"

#: ``(id, src file, edit, killer test, text the killing assertion's message must contain)``
PASS6_F15_SURVIVORS = [
    ("S35", "preintegration_galilean.py",
     [("params.setIntegrationCovariance(sigma_int**2 * np.eye(3))",
       "params.setIntegrationCovariance(sigma_int * np.eye(3))")],
     _C25B, "integration covariance is not sigma_int^2 Dt I"),
    ("S36", "preintegration_galilean.py",
     [("params.setBiasOmegaCovariance(sigma_bg**2 * np.eye(3))",
       "params.setBiasOmegaCovariance(sigma_ba**2 * np.eye(3))"),
      ("params.setBiasAccCovariance(sigma_ba**2 * np.eye(3))",
       "params.setBiasAccCovariance(sigma_bg**2 * np.eye(3))")],
     _C25B, "bias random-walk covariance is not diag(sigma_ba^2, sigma_bg^2) Dt"),
]


_C13E = "test_c13e_cayley_factor_is_a_function_of_its_bbar"
_C05C = "test_c05c_forster_and_galilean_constructors_are_functions_of_their_bias"
#: the fixture's true bias as a literal -- what a ``from_edge`` that leaks it hard-codes
_FIXTURE_BIAS = "np.array([0.0100, -0.0150, 0.0080, 0.0600, -0.0400, 0.0900])"
_DRBAR_MSG = "CayleyFactor.dRbar is not the Cayley increment at the input bbar"
_BBAR_MSG = "CayleyFactor.bbar is not its input"
_CONSUMER_MSG = "factor Eq. (21b) residual with db = b - bbar at ground truth"

#: M2-F14.  ``message`` may be a tuple: then EVERY text must appear.  S12/S13 must die on
#: the Eq. (21b) consumer of ``bbar[3:6]``, not only on the direct ``bbar`` pin.
#: F14-B1 / F14-B3 are the same leak in the Baseline 1 / Baseline 3 constructors.
PASS6_F14_SURVIVORS = [
    ("S46", "preintegration_cayley.py",
     [("pim = integrate_edge(edge, bbar,", f"pim = integrate_edge(edge, {_FIXTURE_BIAS},")],
     _C13E, (_DRBAR_MSG, "CayleyFactor.dvbar is not the Cayley increment", _CONSUMER_MSG)),
    ("S47", "preintegration_cayley.py",
     [("bbar = np.asarray(bbar, dtype=float)\n        pim",
       f"bbar = {_FIXTURE_BIAS}\n        pim")],
     _C13E, (_BBAR_MSG, _DRBAR_MSG, "CayleyFactor.J_R is not the Cayley Jacobian",
             _CONSUMER_MSG)),
    ("S48", "preintegration_cayley.py",
     [("dRbar=pim.dR,",
       "dRbar=integrate_edge(edge, np.concatenate([[0.0100, -0.0150, 0.0080], bbar[3:6]]),"
       " cls=CayleyPreintegration, variant=variant).dR,")],
     _C13E, _DRBAR_MSG),
    ("S12", "preintegration_cayley.py",
     [("return cls(bbar=bbar,", "return cls(bbar=np.concatenate([bbar[0:3], np.zeros(3)]),")],
     _C13E, (_CONSUMER_MSG, _BBAR_MSG)),
    ("S13", "preintegration_cayley.py",
     [("return cls(bbar=bbar,", "return cls(bbar=np.concatenate([bbar[0:3], -bbar[3:6]]),")],
     _C13E, (_CONSUMER_MSG, _BBAR_MSG)),
    ("F14-B1", "preintegration_forster.py",
     [("pim = cls(bias_hat=np.asarray(bias_hat, dtype=float),",
       f"pim = cls(bias_hat={_FIXTURE_BIAS},")],
     _C05C, "Baseline 1 dR is not the Exp increment at the input bias"),
    ("F14-B3", "preintegration_galilean.py",
     [("        bias_hat = np.array(bias_hat, dtype=float, copy=True)",
       f"        bias_hat = {_FIXTURE_BIAS}")],
     _C05C, "Baseline 3 dR is not the exact ZOH increment at the input bias"),
]


@pytest.fixture(scope="module")
def pass6_control(tmp_path_factory):
    """The unmutated copy must pass every pass-6 killer, or no kill below means anything."""
    killers = sorted({m[3] for m in PASS6_F15_SURVIVORS + PASS6_F14_SURVIVORS})
    code, out = _run_in_copy(tmp_path_factory.mktemp("pass6_control"), killers)
    assert code == 0, f"control copy fails a pass-6 killer:\n{out[-3000:]}"
    return killers


def _assert_file_level_kill(root, mutant_id, src_file, edit, killer, message):
    code, out = _run_in_copy(root, [killer], src_file, edit)
    assert code != 0, f"{mutant_id} (file-level edit of {src_file}) SURVIVES {killer}"
    failed_on = _failure_lines(out)
    for text in ((message,) if isinstance(message, str) else message):
        assert text in failed_on, (
            f"{mutant_id} kills {killer}, but not on the intended assertion "
            f"({text!r} is on no 'E ' line):\n{out[-3000:]}")


@pytest.mark.claim("M2-F15")
@pytest.mark.parametrize("mutant_id,src_file,edit,killer,message", PASS6_F15_SURVIVORS,
                         ids=[m[0] for m in PASS6_F15_SURVIVORS])
def test_pass6_f15_survivor_is_killed_as_a_file_edit(tmp_path, pass6_control, mutant_id,
                                                     src_file, edit, killer, message):
    """S35 / S36: Baseline 3's noise-model plumbing, edited on disk."""
    _assert_file_level_kill(tmp_path, mutant_id, src_file, edit, killer, message)


@pytest.mark.claim("M2-F14")
@pytest.mark.parametrize("mutant_id,src_file,edit,killer,message", PASS6_F14_SURVIVORS,
                         ids=[m[0] for m in PASS6_F14_SURVIVORS])
def test_pass6_f14_survivor_is_killed_as_a_file_edit(tmp_path, pass6_control, mutant_id,
                                                     src_file, edit, killer, message):
    """S46 / S47 / S48 (the factor leaks the fixture bias) and S12 / S13 (``bbar[3:6]``
    zeroed / negated), edited on disk -- with no copy-of-``src`` guard involved."""
    _assert_file_level_kill(tmp_path, mutant_id, src_file, edit, killer, message)


@pytest.mark.claim("M2-F14")
def test_pass6_sweep_covers_every_medium_survivor():
    """The 7 medium survivors of the pass-6 report, none dropped."""
    ids = {m[0] for m in PASS6_F14_SURVIVORS + PASS6_F15_SURVIVORS}
    assert {"S12", "S13", "S35", "S36", "S46", "S47", "S48"} <= ids


# ======================================================================================
# Survivors of the verifier's pass-7 sweep (docs/verification/stage-b_m2-baselines_pass7.md
# Sec. 4, findings M2-F16 / M2-F17 and the fixture-degenerate low survivors), plus the
# same class -- an input replaced by its fixture value -- at every other site that
# tests/test_input_dependence.py now varies.  All are REAL FILE EDITS, run in parallel
# copies, and every kill must land on the intended assertion's ``E `` line (M2-F19).
# ======================================================================================

_ID = "tests/test_input_dependence.py::"
_FIX_G_LIT = "np.array([0.0, 0.0, -9.81])"
_FIX_BG_LIT = "np.array([0.0100, -0.0150, 0.0080])"

#: ``(id, src file, edit, killer node ids, texts that must ALL be on 'E ' lines)``
PASS7_MUTANTS = [
    # ---- pass-7 medium survivors (M2-F16, M2-F17) -----------------------------------
    ("U10", "preintegration_cayley.py",
     [("return rotation_from_eq16(R_i, self.dRbar, self.xi(b_g))",
       "return rotation_from_eq16(R_i, self.dRbar, self.J_R @ (np.asarray(b_g, dtype=float)"
       f" - {_FIX_BG_LIT}))")],
     (_ID + "test_b2_factor_methods_read_own_fields", _ID + "test_b2_factor_from_edge"),
     (".predicted_rotation ignores its bbar/fields", "predicted_rotation at b_g != bbar_g")),
    ("U13", "preintegration_forster.py",
     [("pim = integrate_edge(edge, bias_hat, cls=cls, **kwargs)",
       f"pim = integrate_edge(edge, {_FIXTURE_BIAS}, cls=cls, **kwargs)")],
     (_ID + "test_b1_integrate_edge_states",), ("Baseline 1/2 integrate_edge_states(",)),
    ("U20", "preintegration_galilean.py",
     [("params.setGyroscopeCovariance(sigma_g**2 * np.eye(3))",
       "params.setGyroscopeCovariance(0.0 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("stored gyroscope covariance is not sigma_g^2 I",
      "gyro-only rotation block is not sigma_g^2 Dt I")),
    ("U26", "preintegration_galilean.py",
     [("params.setGyroscopeCovariance(sigma_g**2 * np.eye(3))",
       "params.setGyroscopeCovariance(4.0 * sigma_g**2 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("stored gyroscope covariance is not sigma_g^2 I",
      "gyro-only rotation block is not sigma_g^2 Dt I")),
    ("U40", "preintegration_galilean.py",
     [("params.setAccelerometerCovariance(sigma_a**2 * np.eye(3))",
       "params.setAccelerometerCovariance(1.005 * sigma_a**2 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("stored accelerometer covariance is not sigma_a^2 I",
      "accel-only velocity block is not sigma_a^2 Dt I")),
    ("U23", "preintegration_galilean.py",
     [("params.setIntegrationCovariance(sigma_int**2 * np.eye(3))",
       "params.setIntegrationCovariance((1e-3 if sigma_int else 0.0)**2 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("stored integration covariance is not sigma_int^2 I",
      "integration-only covariance is not sigma_int^2 Dt I")),
    ("U23b", "preintegration_galilean.py",
     [("params.setIntegrationCovariance(sigma_int**2 * np.eye(3))",
       "params.setIntegrationCovariance(1e-3**2 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("make_params(sigma_int=0) stores a nonzero integration covariance",
      "sigma_int = 0 (all densities 0) gives a nonzero covariance")),
    ("U30", "preintegration_galilean.py",
     [("return navtriple(self.pim.predict(navstate(state_i), to_gtsam_bias(self.bias_hat)))",
       "return navtriple(self.pim.predict(navstate(state_i), "
       f"to_gtsam_bias({_FIXTURE_BIAS})))")],
     (_ID + "test_b3_predict",), ("Baseline 3 predict at bias_hat",)),
    ("U31", "preintegration_galilean.py",
     [("b = pim.bias_hat + swap_bias_blocks(db_gtsam)",
       f"b = {_FIXTURE_BIAS} + swap_bias_blocks(db_gtsam)")],
     (_ID + "test_b3_corrected_increment_gal3",), ("db = 0, is not the increment at bias_hat",)),
    ("U32", "preintegration_galilean.py",
     [("states.append(integrate_edge(edge, bias_hat, **kwargs).predict(states[-1]))",
       f"states.append(integrate_edge(edge, {_FIXTURE_BIAS}, **kwargs).predict(states[-1]))")],
     (_ID + "test_b3_integrate_edge_states",), ("Baseline 3 integrate_edge_states(",)),
    ("U38", "preintegration_galilean.py",
     [("Vhat = single_step(gyro[k], accel[k], dt, bias_hat, **kwargs)",
       f"Vhat = single_step(gyro[k], accel[k], dt, {_FIXTURE_BIAS}, **kwargs)")],
     (_ID + "test_b3_bias_jacobian_recursion",), ("bias_jacobian_recursion(bias",)),
    # ---- pass-7 fixture-degenerate survivors -----------------------------------------
    ("U03", "preintegration_cayley.py",
     [("self.dvbar, self.Dt, g_W,", "self.dvbar, 0.5, g_W,")],
     (_ID + "test_b2_factor_methods_read_own_fields",), (".velocity_residual (g=",)),
    ("U04", "preintegration_cayley.py",
     [("self.dpbar, self.Dt, g_W,", "self.dpbar, 0.5, g_W,")],
     (_ID + "test_b2_factor_methods_read_own_fields",), (".position_residual (g=",)),
    ("U05", "preintegration_cayley.py",
     [("Dt=pim.dt_total, J_R=J_R,", "Dt=0.5, J_R=J_R,")],
     (_ID + "test_b2_factor_from_edge",), ("Dt is not the literal edge duration",)),
    ("U06", "preintegration_cayley.py",
     [("- np.asarray(g_W) * Dt)", f"- {_FIX_G_LIT} * Dt)")],
     (_ID + "test_b2_eq21b_eq21c", _ID + "test_b2_factor_methods_read_own_fields"),
     ("velocity_residual_eq21b (Dt=", ".velocity_residual (g=")),
    ("U07", "preintegration_cayley.py",
     [("- 0.5 * np.asarray(g_W) * Dt**2)", f"- 0.5 * {_FIX_G_LIT} * Dt**2)")],
     (_ID + "test_b2_eq21b_eq21c", _ID + "test_b2_factor_methods_read_own_fields"),
     ("position_residual_eq21c (Dt=", ".position_residual (g=")),
    ("U11", "preintegration_forster.py",
     [("Dt = self.dt_total", "Dt = 0.5")],
     (_ID + "test_b1_predict",), ("Forster predict on the",)),
    ("U12", "preintegration_forster.py",
     [("g_W = np.asarray(g_W, dtype=float)\n        R_i, v_i, p_i",
       f"g_W = {_FIX_G_LIT}\n        R_i, v_i, p_i")],
     (_ID + "test_b1_predict", _ID + "test_b1_integrate_edge_states"),
     ("Forster predict on the", "Baseline 1/2 integrate_edge_states(")),
    ("U33", "preintegration_galilean.py",
     [("pim.integrate_measurement(omega, acc, dt)\n    return pim.increment_gal3()",
       "pim.integrate_measurement(omega, acc, 2.5e-3)\n    return pim.increment_gal3()")],
     (_ID + "test_b3_integrate_stream_and_single_step",), ("single_step(dt=",)),
    ("U34", "preintegration_galilean.py",
     [("g_W = np.asarray(g_W, dtype=float)\n    return gal3(np.eye(3), 0.5 * g_W",
       f"g_W = {_FIX_G_LIT}\n    return gal3(np.eye(3), 0.5 * g_W")],
     (_ID + "test_b3_endpoint_relations",),
     ("endpoint_W (Dt=", "endpoint_W_form (Dt=")),
    ("U39", "preintegration_galilean.py",
     [("pim.dt_total if Dt is None else Dt)", "0.5 if Dt is None else Dt)")],
     (_ID + "test_b3_corrected_increment_gal3",),
     ("clock with Dt = None is not the literal edge duration",)),
    # ---- the same class at the other sites the inventory varies (V series) ----------
    ("V01", "trajectory.py",
     [("v=v_i + G_W * Dt + R_i @ dv,\n                     p=p_i + v_i * Dt + 0.5 * G_W * Dt**2",
       f"v=v_i + {_FIX_G_LIT} * Dt + R_i @ dv,\n"
       f"                     p=p_i + v_i * Dt + 0.5 * {_FIX_G_LIT} * Dt**2")],
     (_ID + "test_traj_initial_state_and_chain",), ("chain ignores the module gravity",)),
    ("V02", "trajectory.py",
     [("0.6 * np.sin(0.7 * k * dt + _PHASE)", "0.6 * np.sin(0.7 * k * DT + _PHASE)")],
     (_ID + "test_traj_time_varying_stream",), ("time_varying_stream(17, 0.001) values",)),
    ("V03", "preintegration_forster.py",
     [("sigma_g=sigma_g, sigma_a=sigma_a, **kwargs)", "sigma_g=0.0, sigma_a=0.0, **kwargs)")],
     (_ID + "test_b1_integrate_edge",), ("densities not stored",)),
    ("V04", "preintegration_forster.py",
     [("    bias_hat = np.asarray(bias_hat, dtype=float)\n    base = integrate_edge(",
       "    eps = FD_EPS\n    bias_hat = np.asarray(bias_hat, dtype=float)\n"
       "    base = integrate_edge(")],
     (_ID + "test_b1_bias_jacobians_fd",), ("ignores eps = 0.03",)),
    ("V05", "preintegration_forster.py",
     [("        self.dR = np.array(self.dR, dtype=float, copy=True)\n",
       "        self.dR = np.eye(3)\n")],
     (_ID + "test_b1_single_step_reads_every_field",),
     ("integrate_measurement dR from the given start",)),
    ("V06", "preintegration_cayley.py",
     [("pim = integrate_edge(edge, bbar, cls=CayleyPreintegration, variant=variant)",
       "pim = integrate_edge(edge, bbar, cls=CayleyPreintegration, variant=VARIANT_RAW)")],
     (_ID + "test_b2_factor_from_edge",), ("reparam): dRbar",)),
    ("V07", "preintegration_cayley.py",
     [("            edge, bbar, cls=CayleyPreintegration, variant=variant)",
       "            edge, bbar, cls=CayleyPreintegration, variant=VARIANT_RAW)")],
     (_ID + "test_b2_factor_from_edge",), ("reparam): J_R",)),
    ("V08", "preintegration_galilean.py",
     [("g_W, sigma_g, sigma_a, combined=combined)", "G_W, sigma_g, sigma_a, combined=combined)")],
     (_ID + "test_b3_constructor_reads_every_input", _ID + "test_b3_predict"),
     ("g_W did not reach the params", "Baseline 3 predict at bias_hat")),
    ("V09", "preintegration_galilean.py",
     [("pim_cls = (gtsam.PreintegratedCombinedMeasurementsG if combined",
       "pim_cls = (gtsam.PreintegratedCombinedMeasurementsG if False")],
     (_ID + "test_b3_constructor_reads_every_input",), ("PIM class",)),
    ("V10", "preintegration_galilean.py",
     [("    Uhat_inv = increment_at(np.zeros(6)).inverse()",
       "    eps = J_FD_EPS\n    Uhat_inv = increment_at(np.zeros(6)).inverse()")],
     (_ID + "test_b3_bias_jacobian_fd_reads_eps",), ("bias_jacobian_fd ignores eps",)),
    ("V11", "preintegration_galilean.py",
     [("g_W = np.asarray(g_W, dtype=float)\n    return gal3(np.eye(3), -0.5 * g_W",
       f"g_W = {_FIX_G_LIT}\n    return gal3(np.eye(3), -0.5 * g_W")],
     (_ID + "test_b3_endpoint_relations",), ("endpoint_Hprime (Dt=",)),
    ("V12", "preintegration_galilean.py",
     [("                np.asarray(ns.position(), dtype=float) - 0.5 * g * Dt_ij**2,\n"
       "                np.asarray(ns.velocity(), dtype=float) - g * Dt_ij,",
       "                np.asarray(ns.position(), dtype=float),\n"
       "                np.asarray(ns.velocity(), dtype=float),")],
     (_ID + "test_b3_corrected_increment_gal3",), ("db = 0, is not the increment at bias_hat",)),
    ("V13", "preintegration_galilean.py",
     [("params.setBiasAccCovariance(sigma_ba**2 * np.eye(3))",
       "params.setBiasAccCovariance(1e-4**2 * np.eye(3))")],
     (_ID + "test_b3_make_params_stores_every_input",
      _ID + "test_b3_densities_reach_the_covariance"),
     ("stored bias-acc covariance is not sigma_ba^2 I", "accel random-walk block is not")),
    # ---- M2-F18 regressions: the defensive copies removed -----------------------------
    ("F18-B1", "preintegration_forster.py",
     [("self.bias_hat = np.array(self.bias_hat, dtype=float, copy=True)",
       "self.bias_hat = np.asarray(self.bias_hat, dtype=float)")],
     (_ID + "test_f18_constructors_do_not_alias",),
     ("ForsterPreintegration.bias_hat aliases the caller's array",)),
    ("F18-B2", "preintegration_cayley.py",
     [("setattr(self, name, np.array(getattr(self, name), dtype=float, copy=True))",
       "setattr(self, name, np.asarray(getattr(self, name), dtype=float))")],
     (_ID + "test_f18_constructors_do_not_alias",),
     ("CayleyFactor.from_edge(edge, b).bbar aliases b",
      "CayleyFactor.xi moves when the caller's bbar is overwritten")),
    # since pass 8 Baseline 3 keeps no numpy copy of the bias (``bias_hat`` is read off
    # GTSAM's ``biasHat``), so the aliasing defect is replayed as its pre-pass-8 form: a
    # stored ``np.asarray`` of the caller's array, read back by ``bias_hat``
    ("F18-B3", "preintegration_galilean.py",
     [("        bias_hat = np.array(bias_hat, dtype=float, copy=True)",
       "        bias_hat = np.asarray(bias_hat, dtype=float)\n        self._alias = bias_hat"),
      ("        b = from_gtsam_bias(self.pim.biasHat())\n        b.setflags(write=False)\n"
       "        return b", "        return self._alias")],
     (_ID + "test_f18_constructors_do_not_alias",),
     ("GalileanPreintegration.bias_hat aliases the caller's array",
      "Baseline 3 predict moves when the caller's bias is overwritten")),
]

#: the pass-7 report's survivors that this file must kill (medium + fixture-degenerate)
PASS7_REQUIRED = {"U10", "U13", "U20", "U23", "U23b", "U26", "U30", "U31", "U32", "U38",
                  "U40", "U03", "U04", "U05", "U06", "U07", "U11", "U12", "U33", "U34",
                  "U39"}


@pytest.fixture(scope="module")
def pass7_runs(tmp_path_factory):
    """Run the unmutated control and every PASS7 mutant, each in its own copy, in
    parallel.  The control must pass every killer, or no kill below means anything."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    killers = sorted({k for m in PASS7_MUTANTS for k in m[3]})
    code, out = _run_in_copy(tmp_path_factory.mktemp("pass7_control"), killers)
    assert code == 0, f"control copy fails a pass-7 killer:\n{out[-3000:]}"

    def run(m):
        mutant_id, src_file, edit, nodeids, _ = m
        root = tmp_path_factory.mktemp(f"pass7_{mutant_id}")
        return mutant_id, _run_in_copy(root, list(nodeids), src_file, edit)

    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2) // 2))) as pool:
        return dict(pool.map(run, PASS7_MUTANTS))


@pytest.mark.claim("M2-F17")
@pytest.mark.parametrize("mutant_id,src_file,edit,killers,messages", PASS7_MUTANTS,
                         ids=[m[0] for m in PASS7_MUTANTS])
def test_pass7_mutant_is_killed_as_a_file_edit(pass7_runs, mutant_id, src_file, edit,
                                               killers, messages):
    """Each pass-7 mutant, edited on disk, fails its killers on the intended assertions."""
    code, out = pass7_runs[mutant_id]
    assert code != 0, f"{mutant_id} (file-level edit of {src_file}) SURVIVES {killers}"
    failed_on = _failure_lines(out)
    for text in messages:
        assert text in failed_on, (
            f"{mutant_id} kills {killers}, but not on the intended assertion "
            f"({text!r} is on no 'E ' line):\n{out[-3000:]}")


@pytest.mark.claim("M2-F17")
def test_pass7_sweep_covers_every_listed_survivor():
    """The 21 pass-7 survivors named for this remediation, none dropped."""
    assert PASS7_REQUIRED <= {m[0] for m in PASS7_MUTANTS}


@pytest.mark.claim("M2-F19")
def test_f19_matcher_ignores_the_echoed_source(tmp_path):
    """Pass 7's demonstration, inverted: under the S35 edit only c25b part (b) fails, yet
    S36's part-(a) message is in the raw output (pytest echoes the test source).  The
    matcher must NOT find it on an ``E `` line, and must find S35's own message there."""
    s35 = next(m for m in PASS6_F15_SURVIVORS if m[0] == "S35")
    s36_message = next(m for m in PASS6_F15_SURVIVORS if m[0] == "S36")[4]
    code, out = _run_in_copy(tmp_path, [_C25B], s35[1], s35[2])
    assert code != 0
    assert s36_message in out, "pytest no longer echoes the source; revisit this test"
    assert s36_message not in _failure_lines(out), \
        "the matcher accepts a message that only appears in the echoed source"
    assert s35[4] in _failure_lines(out)


# ======================================================================================
# Survivors of the verifier's pass-8 sweep (docs/verification/stage-b_m2-baselines_pass8.md
# Sec. 5.4, findings M2-F21 - M2-F25) and the two latent src defects it found (M2-F23,
# M2-F24), replayed as REAL FILE EDITS.  Every killer is a check against an INDEPENDENT
# reference (retyped equations, 5x5 ``expm``, 60-digit ``mpmath``, ``scipy`` Rotation,
# raw GTSAM params) -- never a change detector (no copy-of-src comparison, no edit anchor)
# -- and every kill must land on the intended assertion's ``E `` line.
# ======================================================================================

_P8_SINGLE = _ID + "test_b1_single_step_reads_every_field"
_P8_F23 = _ID + "test_f23_initial_covariance_propagates_at_zero_density"
_P8_F24 = _ID + "test_f24_so3_log_near_pi"
_P8_SERIES = _ID + "test_f25_series_branches_against_mpmath"
_P8_LOG = _ID + "test_f25_so3_log_small_angles_against_mpmath"
_P8_BPS = _ID + "test_f21_kwargs_reach_gtsam_body_p_sensor"
_P8_ALIAS_OUT = _ID + "test_f22_no_outward_aliasing"
_P8_F18 = _ID + "test_f18_constructors_do_not_alias"

#: ``(id, src file, edit, killer node ids, texts that must ALL be on 'E ' lines)``
PASS8_MUTANTS = [
    # ---- the two latent defects, re-introduced ----------------------------------------
    ("F23", "preintegration_forster.py",
     [("if self.has_noise or np.any(self.cov != 0.0):", "if self.has_noise:")],
     (_P8_F23, _P8_SINGLE),
     ("at zero densities does not propagate the initial cov over",
      "does not propagate the initial cov at zero densities")),
    ("F24", "trajectory.py",
     [("    if c >= LOG_NEAR_PI_COS:\n", "    if True:\n")],
     (_P8_F24,), ("so3_log at theta = pi - 1e-06", "so3_log at theta = pi - 1e-08")),
    # ---- M2-F21 (medium + low) ---------------------------------------------------------
    ("W02", "preintegration_galilean.py",
     [("    Dt_ij = pim.dt_total\n", "    Dt_ij = pim.dt_total if Dt is None else Dt\n")],
     (_ID + "test_b3_corrected_increment_gal3",),
     ("Dt = 1.37 given: (R, v, p) is not the increment with clock 1.37",)),
    ("W13", "preintegration_galilean.py",
     [("Vhat = single_step(gyro[k], accel[k], dt, bias_hat, **kwargs)",
       "Vhat = single_step(gyro[k], accel[k], dt, bias_hat)")],
     (_P8_BPS,), ("params with body_P_sensor) J_",)),
    ("W14", "preintegration_galilean.py",
     [("    pim = GalileanPreintegration(bias_hat, **kwargs)\n"
       "    pim.integrate_measurement(omega, acc, dt)\n",
       "    pim = GalileanPreintegration(bias_hat)\n"
       "    pim.integrate_measurement(omega, acc, dt)\n")],
     (_P8_BPS,), ("params with body_P_sensor) at k=", "params with body_P_sensor) J_")),
    ("W46", "preintegration_cayley.py",
     [("return 3.0 / float(np.trace(Sigma))", "return 3.0 / max(float(np.trace(Sigma)), 1e-6)")],
     (_ID + "test_b2_precision_and_weighted_square",),
     ("isotropic_precision at tr Sigma = 4.3e-08",)),
    ("U47", "preintegration_cayley.py",
     [("return 3.0 / float(np.trace(Sigma))", "return 3.0 / (float(np.trace(Sigma)) + 1e-30)")],
     (_ID + "test_b2_precision_and_weighted_square",),
     ("isotropic_precision at tr Sigma = 1.0e-30",)),
    ("W10", "preintegration_galilean.py",
     [("R = R.matrix() if isinstance(R, Rot3)", "R = R.matrix().T if isinstance(R, Rot3)")],
     (_ID + "test_b3_gal3_adapters",), ("(the Rot3 branch)",)),
    # ---- M2-F22 ------------------------------------------------------------------------
    ("W07", "preintegration_cayley.py",
     [('"J_R", "A_vg", "A_va", "A_pg", "A_pa")', '"J_R", "A_vg", "A_va", "A_pg")')],
     (_P8_F18,), ("CayleyFactor.A_pa aliases the caller's array",)),
    ("W08", "preintegration_cayley.py",
     [('_ARRAY_FIELDS = ("bbar", "dRbar", "dvbar", "dpbar", "J_R", "A_vg", "A_va", "A_pg", "A_pa")',
       '_ARRAY_FIELDS = ("bbar",)')],
     (_P8_F18,), ("CayleyFactor.dRbar aliases the caller's array",
                  "CayleyFactor.A_pa aliases the caller's array")),
    ("W09", "preintegration_forster.py",
     [("return self.dR.copy(), self.dv.copy(), self.dp.copy()",
       "return self.dR, self.dv, self.dp")],
     (_P8_ALIAS_OUT,), ("ForsterPreintegration.increment() returns views",
                        "CayleyPreintegration.increment() returns views")),
    ("F22-bg", "preintegration_forster.py",
     [("return np.array(self.bias_hat[0:3], dtype=float, copy=True)",
       "return self.bias_hat[0:3]")],
     (_P8_ALIAS_OUT,), (".b_g / .b_a are views into bias_hat",
                        "a write into ForsterPreintegration.b_g moves the next step")),
    ("F22-ba", "preintegration_forster.py",
     [("return np.array(self.bias_hat[3:6], dtype=float, copy=True)",
       "return self.bias_hat[3:6]")],
     (_P8_ALIAS_OUT,), (".b_g / .b_a are views into bias_hat",)),
    # S39's own edit (``predict`` reads ``pim.biasHat()``) is EQUIVALENT since pass 8:
    # ``bias_hat`` IS ``from_gtsam_bias(pim.biasHat())``.  The defect S39 exposed -- a
    # writable second copy that can drift from GTSAM's -- is replayed here instead.
    ("S39-desync", "preintegration_galilean.py",
     [("        self.pim = pim_cls(self.params, to_gtsam_bias(bias_hat))\n",
       "        self.pim = pim_cls(self.params, to_gtsam_bias(bias_hat))\n"
       "        self._store = bias_hat\n"),
      ("        b = from_gtsam_bias(self.pim.biasHat())\n        b.setflags(write=False)\n"
       "        return b", "        return self._store")],
     (_P8_ALIAS_OUT,), ("GalileanPreintegration.bias_hat accepts an in-place write",
                        "desynchronises from GTSAM's biasHat",
                        "Baseline 3 predict moves after an in-place edit of bias_hat")),
    # ---- M2-F25: small-angle branches against mpmath -----------------------------------
    ("Q01", "preintegration_forster.py",
     [("return np.eye(3) - 0.5 * K + (1.0 / 6.0) * (K @ K)",
       "return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)")],
     (_P8_SERIES,), ("so3_right_jacobian series branch",)),
    ("Q39", "trajectory.py",
     [("return np.eye(3) + K + 0.5 * (K @ K)", "return np.eye(3) + K")],
     (_P8_SERIES,), ("so3_exp series branch",)),
    ("W17", "trajectory.py",
     [("return T * np.eye(3) + 0.5 * T**2 * K + (T**3 / 6.0) * (K @ K)",
       "return T * np.eye(3) + 0.5 * T**2 * K")],
     (_P8_SERIES,), ("M1 series branch",)),
    ("W18", "trajectory.py",
     [("return 0.5 * T**2 * np.eye(3) + (T**3 / 6.0) * K + (T**4 / 24.0) * (K @ K)",
       "return 0.5 * T**2 * np.eye(3) + (T**3 / 6.0) * K")],
     (_P8_SERIES,), ("M2 series branch",)),
    ("W15", "preintegration_cayley.py",
     [("    if th < 1e-12:\n", "    if th < 1e-6:\n")],
     (_P8_SERIES,), ("cayley_vector |phi| = 9.9e-07",)),
    ("W19", "trajectory.py",
     [("SMALL_ANGLE = 1e-8\n", "SMALL_ANGLE = 1e-6\n")],
     (_P8_LOG,), ("so3_log at |phi| = 9e-07 (generic branch, rel)",)),
    ("W20", "trajectory.py",
     [("    c = min(1.0, max(-1.0, c_raw))\n", "    c = min(1.0, c_raw)\n")],
     (_P8_F24,), ("so3_log at pi - 0e+00 raises",)),
    # Q44 (``arccos`` clamped at ``1 - 1e-15``) is no longer replayed here: it is
    # EQUIVALENT under the tolerance rule (pass 9: 3.3e-24 absolute, 2.96e-16 relative on
    # so3_log, against a 2.3e-12 threshold), and the 0.75-ulp correct-rounding check that
    # used to kill it was removed by orchestrator decision (it pinned unclaimed behaviour
    # with a 1.18x margin on the clean code).
]

#: the pass-8 survivors this file must kill on a reference check, plus the two defects
PASS8_REQUIRED = {"F23", "F24", "W02", "W07", "W08", "W09", "W10", "W13", "W14", "W46",
                  "U47", "Q01", "Q39", "W15", "W17", "W18", "W19", "W20"}


@pytest.fixture(scope="module")
def pass8_runs(tmp_path_factory):
    """The unmutated control and every PASS8 mutant, each in its own copy, in parallel.
    The control must pass every killer, or no kill below means anything."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    killers = sorted({k for m in PASS8_MUTANTS for k in m[3]})
    code, out = _run_in_copy(tmp_path_factory.mktemp("pass8_control"), killers)
    assert code == 0, f"control copy fails a pass-8 killer:\n{out[-3000:]}"

    def run(m):
        mutant_id, src_file, edit, nodeids, _ = m
        root = tmp_path_factory.mktemp(f"pass8_{mutant_id}")
        return mutant_id, _run_in_copy(root, list(nodeids), src_file, edit)

    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2) // 2))) as pool:
        return dict(pool.map(run, PASS8_MUTANTS))


@pytest.mark.claim("M2-F25")
@pytest.mark.parametrize("mutant_id,src_file,edit,killers,messages", PASS8_MUTANTS,
                         ids=[m[0] for m in PASS8_MUTANTS])
def test_pass8_mutant_is_killed_as_a_file_edit(pass8_runs, mutant_id, src_file, edit,
                                               killers, messages):
    """Each pass-8 mutant, edited on disk, fails its reference killers on the intended
    assertions."""
    code, out = pass8_runs[mutant_id]
    assert code != 0, f"{mutant_id} (file-level edit of {src_file}) SURVIVES {killers}"
    failed_on = _failure_lines(out)
    for text in messages:
        assert text in failed_on, (
            f"{mutant_id} kills {killers}, but not on the intended assertion "
            f"({text!r} is on no 'E ' line):\n{out[-3000:]}")


@pytest.mark.claim("M2-F25")
def test_pass8_killers_are_reference_checks():
    """No pass-8 killer is a change detector: every one lives in the claim-test modules
    (never this file, whose pass5/pass6/pass7 guards compare src to copies or anchors)."""
    for m in PASS8_MUTANTS:
        for k in m[3]:
            assert k.startswith(_ID), f"{m[0]}: killer {k} is not an input-dependence test"
    assert PASS8_REQUIRED <= {m[0] for m in PASS8_MUTANTS}


# ======================================================================================
# Survivors of the verifier's pass-9 sweep (docs/verification/stage-b_m2-baselines_pass9.md
# Sec. 5.4, findings M2-F26, M2-F27) and the latent defect M2-F26, replayed as REAL FILE
# EDITS.  Every killer is a check against an INDEPENDENT reference -- complex-step
# linearisations of the retyped integrators, 30-digit ``mpmath`` quadrature, the retyped
# endpoint relation with a 5x5 ``expm`` increment, a retyped Gal(3) recursion -- never a
# change detector, and every kill must land on the intended assertion's ``E `` line.
# ======================================================================================

_P9_F26 = _ID + "test_f26_cayley_covariance_against_complex_step"
_P9_NOISE = _ID + "test_f27_has_noise_per_axis_and_tiny_densities"
_P9_M1 = _ID + "test_f27_M1_M2_fast_twists_against_quadrature"
_P9_NAV = _ID + "test_f27_navstate_keeps_a_drifted_rotation"
_P9_JAC = _ID + "test_f27_bias_jacobians_on_high_rate_edges_against_complex_step"
_P9_REC = _ID + "test_f27_recursion_reads_eps_at_large_steps"
_P9_STREAM = _ID + "test_b1_covariance_over_a_stream"

#: ``(id, src file, edit, killer node ids, texts that must ALL be on 'E ' lines)``.
#: The measured effect of each mutant on its killer, against the bound, is in the comment.
PASS9_MUTANTS = [
    # M2-F26 re-introduced: B2's noise input through Exp's right Jacobian.
    # f26: 1.1e-7 (EDGE_A) .. 6.2e-6 (EDGE_B) block-relative vs 1e-10.
    ("F26", "preintegration_cayley.py",
     [("        return cay_right_jacobian(phi)\n", "        return so3_right_jacobian(phi)\n")],
     (_P9_F26,), ("CayleyPreintegration(raw) Eq. (63), gyro noise alone",)),
    # M1 closed-form K^2 coefficient x (1 + 1e-13): 1.19e-13 at |w| = 5 vs C02's 1e-14.
    ("X15", "trajectory.py",
     [("            + (T - np.sin(th * T) / th) * (K @ K))",
       "            + (T - np.sin(th * T) / th) * (K @ K) * (1.0 + 1e-13))")],
     (_P9_M1,), ("vs mpmath quadrature", "M1(w=")),
    # has_noise requires EVERY gyro axis > 0: covariance 0 (100% off) vs 1e-10.
    ("X21", "preintegration_forster.py",
     [("return bool(np.any(np.asarray(self.sigma_g) > 0.0)",
       "return bool(np.all(np.asarray(self.sigma_g) > 0.0)")],
     (_P9_NOISE,), ("with sigma_g = [3e-4, 0, 0] alone: has_noise is False",
                    "with sigma_g = [3e-4, 0, 0] alone: Eq. (63) over 14 steps")),
    # accel densities <= 1e-20 ignored: covariance 0 (100% off) vs 1e-10.
    ("X22", "preintegration_forster.py",
     [("or np.any(np.asarray(self.sigma_a) > 0.0))", "or np.any(np.asarray(self.sigma_a) > 1e-20))")],
     (_P9_NOISE,), ("with sigma_a = 1e-21 alone: has_noise is False",
                    "with sigma_a = 1e-21 alone: Eq. (63) over 14 steps")),
    # rotation transition always Exp (B2 integrates with Cay): f26 2.0e-9 (EDGE_A) ..
    # 4.2e-7 (EDGE_B) vs 1e-10.
    ("X24", "preintegration_forster.py",
     [("        A[0:3, 0:3] = dR_step.T\n", "        A[0:3, 0:3] = so3_exp(-w * dt)\n")],
     (_P9_F26,), ("CayleyPreintegration(raw) Eq. (63), gyro noise alone",)),
    # Ba velocity block x (1 + 2e-9): 4.0e-9 relative vs 1e-10.
    ("X31", "preintegration_forster.py",
     [("        Ba[3:6, :] = dR_ik * dt\n", "        Ba[3:6, :] = dR_ik * dt * (1.0 + 2e-9)\n")],
     (_P8_SINGLE, _P9_STREAM), ("Eq. (63) one step, accel noise alone",
                                "accel noise alone (sigma_g=")),
    # navstate re-orthonormalises R: 2.4e-12 .. 2.4e-10 at drifts 2.9e-11 / 1.1e-10 vs 1e-12.
    ("X41", "preintegration_galilean.py",
     [("    return NavState(Rot3(state.R), state.p, state.v)",
       "    return NavState(Rot3.ClosestTo(state.R), state.p, state.v)")],
     (_P9_NAV,), ("navstate does not keep a drifted R as given",
                  "Baseline 3 predict from a drifted R")),
    # the recursion drops eps: 1.9e-7 (eps 5e-2) and 3.1e-6 (eps 0.2) vs 1e-9.
    ("W12", "preintegration_galilean.py",
     [("C = bias_jacobian_fd(step_increment, eps=eps)",
       "C = bias_jacobian_fd(step_increment, eps=C_FD_EPS)")],
     (_P9_REC,), ("bias_jacobian_recursion at dt = 0.3, eps = 0.2",
                  "bias_jacobian_recursion at dt = 0.3, eps = 0.05")),
    # Forster FD_EPS = 1e-3: 8.6e-8 / 1.8e-7 relative vs 1e-8.
    ("P23", "preintegration_forster.py",
     [("FD_EPS = 3e-5\n", "FD_EPS = 1e-3\n")],
     (_P9_JAC,), ("at the default eps on the high-rate edge |w| = 2.55",
                  "CayleyFactor.from_edge(raw)")),
    # the pre-pass-9 default FD_EPS = 1e-6 (round-off-limited): 1.1e-8 .. 1.9e-8 vs 1e-8
    # in this test's metric; 1.0x-1.2x in the pass-10 verifier's.  EQUIVALENT under the
    # owner's rule (largest Jacobian change 4.4e-8 relative vs C21's 1e-7, pass 10) and
    # killed only by a round-off-margined assertion, so this kill is NOT relied on: it
    # is kept to document that the old default fails here, and is not in PASS9_REQUIRED.
    ("P23-old", "preintegration_forster.py",
     [("FD_EPS = 3e-5\n", "FD_EPS = 1e-6\n")],
     (_P9_JAC,), ("at the default eps on the high-rate edge",)),
    # CayleyFactor.from_edge with an FD eps of 1e-3: 8.6e-8 / 1.8e-7 vs 1e-8.
    ("S21", "preintegration_cayley.py",
     [("            edge, bbar, cls=CayleyPreintegration, variant=variant)",
       "            edge, bbar, eps=1e-3, cls=CayleyPreintegration, variant=variant)")],
     (_P9_JAC,), ("CayleyFactor.from_edge(raw)", "CayleyFactor.from_edge(reparam)")),
]

PASS9_REQUIRED = {"F26", "X15", "X21", "X22", "X24", "X31", "X41", "W12", "P23", "S21"}


@pytest.fixture(scope="module")
def pass9_runs(tmp_path_factory):
    """The unmutated control and every PASS9 mutant, each in its own copy, in parallel.
    The control must pass every killer, or no kill below means anything."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    killers = sorted({k for m in PASS9_MUTANTS for k in m[3]})
    code, out = _run_in_copy(tmp_path_factory.mktemp("pass9_control"), killers)
    assert code == 0, f"control copy fails a pass-9 killer:\n{out[-3000:]}"

    def run(m):
        mutant_id, src_file, edit, nodeids, _ = m
        root = tmp_path_factory.mktemp(f"pass9_{mutant_id}")
        return mutant_id, _run_in_copy(root, list(nodeids), src_file, edit)

    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2) // 2))) as pool:
        return dict(pool.map(run, PASS9_MUTANTS))


@pytest.mark.claim("M2-F27")
@pytest.mark.parametrize("mutant_id,src_file,edit,killers,messages", PASS9_MUTANTS,
                         ids=[m[0] for m in PASS9_MUTANTS])
def test_pass9_mutant_is_killed_as_a_file_edit(pass9_runs, mutant_id, src_file, edit,
                                               killers, messages):
    """Each pass-9 mutant, edited on disk, fails its reference killers on the intended
    assertions."""
    code, out = pass9_runs[mutant_id]
    assert code != 0, f"{mutant_id} (file-level edit of {src_file}) SURVIVES {killers}"
    failed_on = _failure_lines(out)
    for text in messages:
        assert text in failed_on, (
            f"{mutant_id} kills {killers}, but not on the intended assertion "
            f"({text!r} is on no 'E ' line):\n{out[-3000:]}")


@pytest.mark.claim("M2-F27")
def test_pass9_killers_are_reference_checks():
    """No pass-9 killer is a change detector: every one lives in the input-dependence
    module (never this file)."""
    for m in PASS9_MUTANTS:
        for k in m[3]:
            assert k.startswith(_ID), f"{m[0]}: killer {k} is not an input-dependence test"
    assert PASS9_REQUIRED <= {m[0] for m in PASS9_MUTANTS}


# ======================================================================================
# Survivors of the verifier's pass-10 sweep (docs/verification/stage-b_m2-baselines_pass10.md
# Sec. 5.4, finding M2-F28), replayed as REAL FILE EDITS.  Every killer is a check against
# an independent reference (complex step, 50-digit mpmath derivatives of ``expm`` and of
# the Cayley definition, retyped residuals), and every kill must land on the intended
# assertion's ``E `` line.  Measured effects (vs the bound) are in the comments.
# ======================================================================================

_P10_JR = _ID + "test_f28_right_jacobians_against_mpmath_across_angles"
_P10_COV = _ID + "test_f28_covariance_across_per_step_angles_against_complex_step"
_P10_NOISE = _ID + "test_f28_has_noise_is_any_positive_density_on_any_axis"
_P10_HALVES = _ID + "test_f28_eq21b_eq21c_every_combination_of_bias_halves"
_P10_LARGE_V = _ID + "test_f28_eq21c_at_large_velocity_and_duration"

_P10_RRJ_DOC = "        ``Exp``'s here while integrating with ``Cay``).\"\"\"\n"

#: ``(id, src file, edit, killer node ids, texts that must ALL be on 'E ' lines)``
PASS10_MUTANTS = [
    # Jr's K^2 coefficient x (1 + 1e-8): Jr 7.8e-9 abs at |phi| = 3 vs 1e-14; Eq. (63)
    # 8.2e-10 block-relative at 0.5 rad/step vs 1e-10.
    ("Y18", "preintegration_forster.py",
     [("+ ((th - np.sin(th)) / th**3) * (K @ K))",
       "+ ((th - np.sin(th)) / th**3) * (K @ K) * (1.0 + 1e-8))")],
     (_P10_JR, _P10_COV), ("so3_right_jacobian at |phi| = 3.0e+00 vs mpmath",
                           "ForsterPreintegration Eq. (63), gyro noise alone, "
                           "per-step angle 0.5 (dt = 0.05)")),
    # has_noise ignores densities <= 1e-100 (both sources): flag False at 1e-100 .. 5e-324
    # and covariance 0 at 1e-101 .. 1e-150 (block-relative 1.0 vs 1e-10).
    ("Y20", "preintegration_forster.py",
     [("return bool(np.any(np.asarray(self.sigma_g) > 0.0)",
       "return bool(np.any(np.asarray(self.sigma_g) > 1e-100)"),
      ("or np.any(np.asarray(self.sigma_a) > 0.0))", "or np.any(np.asarray(self.sigma_a) > 1e-100))")],
     (_P10_NOISE,), ("ForsterPreintegration with sigma_g[0] = 1.0e-101 alone: has_noise is False",
                     "CayleyPreintegration with sigma_a[2] = 1.0e-140 alone: Eq. (63) over 14 steps")),
    # accel densities <= 1e-60 ignored: flag False and covariance 0 (1.0 vs 1e-10).
    ("Y58", "preintegration_forster.py",
     [("or np.any(np.asarray(self.sigma_a) > 0.0))", "or np.any(np.asarray(self.sigma_a) > 1e-60))")],
     (_P10_NOISE,), ("ForsterPreintegration with sigma_a[1] = 1.0e-61 alone: has_noise is False",
                     "ForsterPreintegration with sigma_a[0] = 1.0e-61 alone: Eq. (63) over 14 steps")),
    # B2 raw noise input through Exp's Jr below |phi| = 1e-4: 1.35e-9 at 9e-5 rad/step vs 1e-10.
    ("Y24", "preintegration_cayley.py",
     [("        return cay_right_jacobian(phi)\n",
       "        if np.linalg.norm(phi) < 1e-4:\n            return so3_right_jacobian(phi)\n"
       "        return cay_right_jacobian(phi)\n")],
     (_P10_COV,), ("CayleyPreintegration(raw) Eq. (63), gyro noise alone, per-step angle 9e-05",)),
    # ... above |phi| = 1e-2: 6.7e-5 (2e-2 rad/step) .. 5.2e-2 (0.5) vs 1e-10.
    ("Y55", "preintegration_cayley.py",
     [("        return cay_right_jacobian(phi)\n",
       "        if np.linalg.norm(phi) > 1e-2:\n            return so3_right_jacobian(phi)\n"
       "        return cay_right_jacobian(phi)\n")],
     (_P10_COV,), ("CayleyPreintegration(raw) Eq. (63), gyro noise alone, per-step angle 0.02",)),
    # B1 noise input first-order I - K/2 above |phi| = 1e-2: 1.3e-4 .. 8.6e-2 vs 1e-10.
    ("Y56", "preintegration_forster.py",
     [(_P10_RRJ_DOC + "        return so3_right_jacobian(phi)\n",
       _P10_RRJ_DOC + "        if np.linalg.norm(phi) > 1e-2:\n"
       "            return np.eye(3) - 0.5 * hat(phi)\n        return so3_right_jacobian(phi)\n")],
     (_P10_COV,), ("ForsterPreintegration Eq. (63), gyro noise alone, per-step angle 0.02",)),
    # rotation transition A_RR = I below |w dt| = 1e-4 (B1 and B2): 3.8e-7 (1e-7 rad/step)
    # .. 3.4e-4 (9e-5) vs 1e-10.
    ("Y57", "preintegration_forster.py",
     [("        A[0:3, 0:3] = dR_step.T\n",
       "        A[0:3, 0:3] = np.eye(3) if np.linalg.norm(w * dt) < 1e-4 else dR_step.T\n")],
     (_P10_COV,), ("ForsterPreintegration Eq. (63), gyro noise alone, per-step angle 1e-07",
                   "CayleyPreintegration(raw) Eq. (63), gyro noise alone, per-step angle 9e-05")),
    # Eq. (21b) applies A_va db_a only when db_g is given: 0.18 abs vs 1e-12.
    ("Y30", "preintegration_cayley.py",
     [("    if A_va is not None and db_a is not None:", "    if A_va is not None and db_g is not None:")],
     (_P10_HALVES,), ("velocity_residual_eq21b (Dt=0.2, accel half only)",)),
    # Eq. (21c), same for A_pa: 0.29 abs vs 1e-12.
    ("Y59", "preintegration_cayley.py",
     [("    if A_pa is not None and db_a is not None:", "    if A_pa is not None and db_g is not None:")],
     (_P10_HALVES,), ("position_residual_eq21c (Dt=0.73, accel half only)",)),
    # Eq. (21c) v_i Dt x (1 + 1e-13): 8.3e-12 abs at |v_i| = 47 m/s, Dt = 2 s vs 1e-12
    # (clean is exactly 0 there).
    ("Y31", "preintegration_cayley.py",
     [("- np.asarray(v_i) * Dt - 0.5", "- np.asarray(v_i) * Dt * (1.0 + 1e-13) - 0.5")],
     (_P10_LARGE_V,), ("position_residual_eq21c at large v_i Dt",)),
]

PASS10_REQUIRED = {"Y18", "Y20", "Y24", "Y30", "Y31", "Y55", "Y56", "Y57", "Y58", "Y59"}


@pytest.fixture(scope="module")
def pass10_runs(tmp_path_factory):
    """The unmutated control and every PASS10 mutant, each in its own copy, in parallel.
    The control must pass every killer, or no kill below means anything."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    killers = sorted({k for m in PASS10_MUTANTS for k in m[3]})
    code, out = _run_in_copy(tmp_path_factory.mktemp("pass10_control"), killers)
    assert code == 0, f"control copy fails a pass-10 killer:\n{out[-3000:]}"

    def run(m):
        mutant_id, src_file, edit, nodeids, _ = m
        root = tmp_path_factory.mktemp(f"pass10_{mutant_id}")
        return mutant_id, _run_in_copy(root, list(nodeids), src_file, edit)

    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 2) // 2))) as pool:
        return dict(pool.map(run, PASS10_MUTANTS))


@pytest.mark.claim("M2-F28")
@pytest.mark.parametrize("mutant_id,src_file,edit,killers,messages", PASS10_MUTANTS,
                         ids=[m[0] for m in PASS10_MUTANTS])
def test_pass10_mutant_is_killed_as_a_file_edit(pass10_runs, mutant_id, src_file, edit,
                                                killers, messages):
    """Each pass-10 mutant, edited on disk, fails its reference killers on the intended
    assertions."""
    code, out = pass10_runs[mutant_id]
    assert code != 0, f"{mutant_id} (file-level edit of {src_file}) SURVIVES {killers}"
    failed_on = _failure_lines(out)
    for text in messages:
        assert text in failed_on, (
            f"{mutant_id} kills {killers}, but not on the intended assertion "
            f"({text!r} is on no 'E ' line):\n{out[-3000:]}")


@pytest.mark.claim("M2-F28")
def test_pass10_killers_are_reference_checks():
    """No pass-10 killer is a change detector: every one lives in the input-dependence
    module (never this file), and every listed survivor is replayed."""
    for m in PASS10_MUTANTS:
        for k in m[3]:
            assert k.startswith(_ID), f"{m[0]}: killer {k} is not an input-dependence test"
    assert PASS10_REQUIRED == {m[0] for m in PASS10_MUTANTS}
