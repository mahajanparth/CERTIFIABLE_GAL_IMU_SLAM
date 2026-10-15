"""Mutation harness for ``src/qcqp`` -- the M3 claim suite must FAIL on plausible defects.

Each mutant is a single, exactly-once textual substitution in one module of ``src/qcqp``,
applied to a SCRATCH COPY of the package (the working tree is never touched).  Every mutant
names the test(s) of ``tests/test_qcqp.py`` that must kill it; the harness runs exactly those
tests against the mutated copy in a fresh interpreter and requires every named test to fail.
An unmutated copy runs the union of all named tests and must pass them all, otherwise every
"kill" could be a harness artefact.

The defects are the ones a careless implementation of the claims would plausibly contain:
sign flips, transposes, swapped index ranges, dropped families and liftings, the wrong lifting
side, the GTSAM bias-order swap, NavState-vs-Gal3 block confusion, the Rai Eq. (20) literal
rotation weight, and a missing Dt in the random-walk precision.

A documented EQUIVALENT mutant (not in the list): building P2 from rows instead of columns
yields the identical set of quadratic forms (that is claim M3-C15), so no test can or should
kill it.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = "tests/test_qcqp.py::"

K_INSTANCE = T + "test_instance_matches_independent_gtsam_reference[true]"
K_C01 = T + "test_c01_index_map[true]"
K_C07T = T + "test_c07_affine_forms_equal_5x5_residual[true]"
K_C07Z = T + "test_c07_affine_forms_equal_5x5_residual[zero]"
K_C08 = T + "test_c08_lifted_coefficients_and_lifting_dependence"
K_C09 = T + "test_c09_primary_vanish_on_liftings"
K_C11 = T + "test_c11_galilean_redundant_vanish"
K_C14 = T + "test_c14_family_sizes_and_ranks"
K_C24 = T + "test_c24_cost_equals_direct[iso-500-true]"
K_C25 = T + "test_c25_Q_psd_and_bias_block[iso]"
K_C26 = T + "test_c26_truth_cost_and_first_order_agreement_with_gtsam"
K_C29 = T + "test_c29_half_frobenius_weight"
K_C24A = T + "test_c24_cost_equals_direct[aniso-300-true]"
K_C27 = T + "test_c27_truth_discrepancy_per_component"
K_C34 = T + "test_c34_frame_of_anisotropic_lifting"
K_C39 = T + "test_c39_instance_fingerprint"
K_INSTANCE_Z = T + "test_instance_matches_independent_gtsam_reference[zero]"
K_C13 = T + "test_c13_families_nontrivial_off_variety[False]"
K_PROJ = T + "test_decoder_projection_is_nearest_rotation"
K_WHITEN = T + "test_whitened_residual_is_symmetric_root_whitening"
K_P1RA1 = T + "test_P1_is_column_and_RA1_is_row_orthonormality"
K_BIASCONV = T + "test_bias_boundary_converters"
K_ROUND = T + "test_decoder_rounding_of_perturbed_lifted_states"
K_DEGEN = T + "test_decoder_degenerate_inputs"
K_FORMS = T + "test_form_helpers_are_symmetric_and_evaluate_correctly"
K_SGUARD = T + "test_decoder_s_guard_threshold_tie_and_scale"
K_NONFINITE = T + "test_decoder_rejects_non_finite_z"
K_COPY = T + "test_state_copy_is_independent"
K_VARIANT = T + "test_cost_terms_rejects_unknown_variant"
K_EXPORTS = T + "test_package_exports"
K_JEXACT = T + "test_c04_instance_J_structural_zeros_exact_and_J_accuracy"
K_TRUTHCOPY = T + "test_truth_state_blocks_are_independent_arrays"
K_DEFVARIANT = T + "test_cost_terms_default_variant_is_iso"
K_GRAVITY = T + "test_make_params_rejects_non_z_up_gravity"
K_FLOOR = T + "test_decoder_s_guard_floor_below_1e_minus_300"
K_LARGE = T + "test_decoder_large_scales_decode"
K_DETSIGN = T + "test_project_so3_det_correction_is_exact"

#: (id, module, old, new, what it models, killers)
MUTANTS = [
    ("rR-lifting-sign", "cost.py", "- 0.5 * (Y + Yp))", "+ 0.5 * (Y + Yp))",
     "sign flip on the Y, Y' liftings in r_R", [K_C08, K_C07T]),
    ("rR-gamma-sign", "cost.py", "forms.cmat_right(Rj, I3 + 0.5 * Gam)", "forms.cmat_right(Rj, I3 - 0.5 * Gam)",
     "sign flip of the nominal-bias Gam term on the R_j side", [K_C07T, K_C27]),
    ("rv-gravity-sign", "cost.py", "rv = (vj - vi - np.outer(g * Dt, s)", "rv = (vj - vi + np.outer(g * Dt, s)",
     "gravity sign in r_v", [K_C07Z, K_C26]),
    ("rp-gravity-sign", "cost.py", "np.outer(0.5 * g * Dt**2, s)", "np.outer(-0.5 * g * Dt**2, s)",
     "the pre-flight F2 position-block sign error", [K_C07Z]),
    ("M-transposed", "cost.py", "M = forms.cmat_right(Ri, edge.dR)", "M = forms.cmat_right(Ri, edge.dR.T)",
     "M = R_i dRbar^T in the residual forms", [K_C07Z]),
    ("rot-weight-rai-literal", "cost.py", "0.5 * edge.tau_R * np.eye(9)", "edge.tau_R * np.eye(9)",
     "Rai Eq. (20) literal tau_R ||r_R||_F^2", [K_C29, K_C24]),
    ("tau-v-p-swap", "cost.py", 'Term(f"imu{e}.vel", rv, edge.tau_v', 'Term(f"imu{e}.vel", rv, edge.tau_p',
     "velocity residual weighted with tau_p", [K_C24]),
    ("chordal-prior-weight", "cost.py", "0.5 * w_prior * np.eye(9)", "w_prior * np.eye(9)",
     "chordal R0 prior without the 1/2", [K_C26]),
    ("drop-jside-velocity-lifting", "cost.py", "- 0.5 * (V + Vp))", "- 1.0 * V)",
     "V' dropped from r_v (Rai's R_i-only lifting)", [K_C08, K_C07T]),
    ("P2-sign", "constraints.py", "Fm.cross(R[:, a], R[:, b]) - Fm.outer(s, R[:, c])",
     "Fm.cross(R[:, a], R[:, b]) + Fm.outer(s, R[:, c])", "left-handed P2", [K_C09]),
    ("P3-jside-uses-M", "constraints.py", '"j": (self.Rj, L.mat_form(L.Yp(e))', '"j": (self.M, L.mat_form(L.Yp(e))',
     "wrong lifting side: R_j-side liftings checked against M", [K_C09]),
    ("RS-transposed", "constraints.py", "Fm.matmul(Y, Fm.lin_T(A))", "Fm.matmul(Fm.lin_T(A), Y)",
     "RS written as A^T Y instead of Y A^T", [K_C11]),
    ("RH-sign", "constraints.py", "rhs = Fm.matvec(A, Fm.cvec(ea_hat, Ec))", "rhs = -Fm.matvec(A, Fm.cvec(ea_hat, Ec))",
     "RH with det(M) = -1", [K_C11]),
    ("drop-jside-families", "constraints.py", "for side, (A, Y, V, P) in E.sides.items():",
     "for side, (A, Y, V, P) in list(E.sides.items())[:1]:", "every j-side constraint dropped", [K_C14]),
    ("edge-uses-b_j-constraints", "constraints.py", "bf = L.vec_form(L.b(ed.i))", "bf = L.vec_form(L.b(ed.j))",
     "edge (i, j) constrained with b_j", [K_C09]),
    ("V-P-index-swap", "lifting.py", "return slice(self._o(e) + 18, self._o(e) + 21)",
     "return slice(self._o(e) + 24, self._o(e) + 27)", "V and P index ranges collide", [K_C01]),
    ("R-row-major", "lifting.py", 'z[layout.R(k)] = state.R[k].flatten(order="F")', "z[layout.R(k)] = state.R[k].flatten()",
     "vec(R) row-major", [K_C01]),
    ("lifting-uses-b_j", "lifting.py", "b = state.b[i]\n", "b = state.b[j]\n",
     "liftings built with the wrong keyframe's bias", [K_C01]),
    ("rho-wrong-side", "lifting.py", "z[layout.rho_v(e)] = M.T @ (rv @ z)", "z[layout.rho_v(e)] = M @ (rv @ z)",
     "rho_v = M r_v instead of M^T r_v", [K_C01]),
    ("bias-order-swap", "gtsam_boundary.py", "return ConstantBias(b[3:6], b[0:3])", "return ConstantBias(b[0:3], b[3:6])",
     "project bias passed to GTSAM as (gyro, accel)", [K_INSTANCE]),
    ("navstate-pos-vel-swap", "gtsam_boundary.py", 'NAV_ROWS = {"rot": slice(0, 3), "pos": slice(3, 6), "vel": slice(6, 9)}',
     'NAV_ROWS = {"rot": slice(0, 3), "pos": slice(6, 9), "vel": slice(3, 6)}',
     "preintMeasCov read in Gal3 (rot, vel, pos) order", [K_INSTANCE]),
    ("gravity-left-position-sign", "gtsam_boundary.py", "p=-0.5 * g_w * Dt**2", "p=+0.5 * g_w * Dt**2",
     "G with +1/2 g Dt^2 (the uncorrected contract)", [K_INSTANCE]),
    ("accel-noise-old-value", "fixture.py", "SIGMA_A = 2.0e-3", "SIGMA_A = 1.0e-2",
     "the pre-ad89dc4 accelerometer density", [K_C39, K_INSTANCE]),
    ("integration-noise-old-value", "fixture.py", "SIGMA_INT = 1.0e-5", "SIGMA_INT = 1.0e-6",
     "integration covariance 1e-12 instead of 1e-10", [K_C39]),
    ("gyro-noise-variance-not-squared", "gtsam_boundary.py", "setGyroscopeCovariance(sigma_g**2 * np.eye(3))",
     "setGyroscopeCovariance(sigma_g * np.eye(3))", "density passed where variance is expected", [K_C39]),
    ("gravity-NED-params", "gtsam_boundary.py", "gtsam.PreintegrationParams.MakeSharedU(", "gtsam.PreintegrationParams.MakeSharedD(",
     "MakeSharedD (gravity +z) for a z-up world", [K_INSTANCE]),
    ("tau-without-rai-3", "instance.py", "tau = {k: 3.0 / np.trace(cov[(k, k)])", "tau = {k: 1.0 / np.trace(cov[(k, k)])",
     "tau^-1 = tr(Sigma) instead of (1/3) tr(Sigma)", [K_C39]),
    ("rw-dt-one-second", "instance.py", "dtb = F.T_EDGE", "dtb = 1.0",
     "random-walk Dt taken as 1 s", [K_C39]),
    ("rv-nominal-bias-jside-dropped", "cost.py", "+ 0.5 * (Rcv(Rj, cv) + Rcv(M, cv)) - 0.5 * (V + Vp))",
     "+ 1.0 * Rcv(M, cv) - 0.5 * (V + Vp))", "cbar^v applied on the R_i side only", [K_C27, K_C07T]),
    ("aniso-weight-override-ignored", "cost.py", "edge.W_vp if W_vp is None else np.asarray(W_vp, float)",
     "edge.W_vp", "caller's W_vp silently ignored", [K_C34]),
    ("rho-v-p-order-swap", "cost.py",
     "rho = np.vstack([layout.vec_form(layout.rho_v(e)), layout.vec_form(layout.rho_p(e))])",
     "rho = np.vstack([layout.vec_form(layout.rho_p(e)), layout.vec_form(layout.rho_v(e))])",
     "W_vp applied in (pos, vel) order", [K_C24A]),
    # -- verifier pass 1 survivors (docs/verification/stage-b_m3-qcqp-derivation.md Sec. 6) --
    ("fx-sigma-prior", "fixture.py", "SIGMA_PRIOR = 1.0e-3", "SIGMA_PRIOR = 3.0e-3",
     "KF0 prior sigma changed", [K_INSTANCE, K_C24]),
    ("fx-sigma-gnss", "fixture.py", "SIGMA_GNSS = 0.05 ", "SIGMA_GNSS = 0.5  ",
     "GNSS sigma changed", [K_INSTANCE, K_C24]),
    ("fx-bg-true", "fixture.py", "B_G = np.array([0.0100, -0.0150, 0.0080])",
     "B_G = np.array([0.0100, -0.0150, 0.0090])", "true gyro bias changed", [K_INSTANCE, K_INSTANCE_Z]),
    ("fx-v0", "fixture.py", "V_0 = np.array([0.50, -0.30, 0.20])", "V_0 = np.array([0.50, -0.30, 0.25])",
     "initial velocity changed", [K_INSTANCE]),
    ("fx-rotvec0", "fixture.py", "ROT_VEC_0 = np.array([0.10, -0.20, 0.30])",
     "ROT_VEC_0 = np.array([0.10, -0.20, 0.35])", "initial attitude changed", [K_INSTANCE]),
    ("lf-decode-v-mul", "lifting.py", "[z[layout.v(k)] / s for", "[z[layout.v(k)] * s for",
     "decoder multiplies v by s", [K_C01]),
    ("lf-decode-p-mul", "lifting.py", "[z[layout.p(k)] / s for", "[z[layout.p(k)] * s for",
     "decoder multiplies p by s", [K_C01]),
    ("lf-project-nodet", "lifting.py", "D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])",
     "D = np.eye(3)", "projection may return a reflection", [K_PROJ]),
    ("lf-s-param-ignored", "lifting.py", "return s * z", "return z", "z_of_state ignores s", [K_C01]),
    ("lf-bias-indices", "lifting.py", "return np.arange(self.bias0, self.bias0 + 6 * self.n_kf)",
     "return np.arange(self.bias0, self.bias0 + 6 * (self.n_kf - 1))", "bias_indices drops a block", [K_C01]),
    ("co-whiten-W-not-sqrt", "cost.py", "(U * np.sqrt(np.clip(w, 0, None))) @ U.T",
     "(U * np.clip(w, 0, None)) @ U.T", "whitening by W instead of W^1/2", [K_WHITEN]),
    ("co-whiten-nonsym-root", "cost.py", "(U * np.sqrt(np.clip(w, 0, None))) @ U.T @ (t.F @ z)",
     "(U * np.sqrt(np.clip(w, 0, None))) @ (t.F @ z)", "non-symmetric root U Lambda^1/2", [K_WHITEN]),
    ("co-whiten-noclip", "cost.py", "np.sqrt(np.clip(w, 0, None))", "np.sqrt(w)",
     "NaN on a singular PSD weight", [K_WHITEN]),
    ("cn-P1-RA1-swap", "constraints.py",
     'fam["P1"].append(_mk("P1", f"R{k}^T R{k}[{a},{b}]", RtR[a, b] - I3[a, b] * ss))\n'
     '            fam["RA1"].append(_mk("RA1", f"R{k} R{k}^T[{a},{b}]", RRt[a, b] - I3[a, b] * ss))',
     'fam["P1"].append(_mk("P1", f"R{k}^T R{k}[{a},{b}]", RRt[a, b] - I3[a, b] * ss))\n'
     '            fam["RA1"].append(_mk("RA1", f"R{k} R{k}^T[{a},{b}]", RtR[a, b] - I3[a, b] * ss))',
     "P1 and RA1 contents exchanged (union unchanged)", [K_P1RA1]),
    ("cn-PRIMARY-incl-RA1", "constraints.py", 'PRIMARY = ("P0", "P1", "P2", "P3")',
     'PRIMARY = ("P0", "P1", "P2", "P3", "RA1")', "RA1 labelled primary", [K_C14]),
    ("cn-family-max-signed", "constraints.py", "float(np.abs(r).max())", "float(r.max())",
     "family_max reports signed max", [K_C13]),
    ("gb-from-bias-swap", "gtsam_boundary.py",
     "return np.concatenate([np.asarray(cb.gyroscope()), np.asarray(cb.accelerometer())])",
     "return np.concatenate([np.asarray(cb.accelerometer()), np.asarray(cb.gyroscope())])",
     "inverse bias converter in GTSAM order", [K_BIASCONV]),
    # -- verifier pass 2 survivors (docs/verification/stage-b_m3-qcqp-derivation_pass2.md Sec. 4.3).
    #    lf-bg-slice, lf-ba-slice, in-n_kf-edges, gb-GAL_ROWS-pos-vel, fm-cmat-left-transposed and
    #    fm-quadT-identity have no mutant here: their helpers were unused and are DELETED. --
    ("lf-project-before-divide", "lifting.py", "R = [project_so3(x) for x in R]",
     "R = [project_so3(x * s) for x in R]", "rotation block projected before dividing by s", [K_ROUND, K_PROJ]),
    ("lf-project-sign-detA", "lifting.py", "np.sign(np.linalg.det(U @ Vt))", "np.sign(np.linalg.det(A))",
     "det correction from det(A): singular blocks give a non-rotation", [K_DEGEN]),
    ("lf-decode-no-s-guard", "lifting.py", "if abs(s) <= S_FLOOR * max(zmax, 1e-300):", "if False:",
     "decoder divides by s ~ 0", [K_DEGEN, K_SGUARD]),
    ("lf-truth-state-bbar", "lifting.py", "[inst.b_true.copy() for _ in inst.truth]",
     "[inst.bbar.copy() for _ in inst.truth]", "truth bias taken from the linearisation point", [K_INSTANCE_Z]),
    ("cn-family-residuals-abs", "constraints.py", "np.array([c.residual(z) for c in cs])",
     "np.abs(np.array([c.residual(z) for c in cs]))", "family_residuals returns |g|", [K_C13]),
    ("fm-outer-nosym", "forms.py", "return sym(f[..., :, None] * g[..., None, :])",
     "return f[..., :, None] * g[..., None, :]", "outer product left unsymmetrised", [K_FORMS]),
    ("fm-sym-identity", "forms.py", "return 0.5 * (Q + np.swapaxes(Q, -1, -2))", "return Q",
     "sym is the identity", [K_FORMS]),
    ("rw-missing-dt", "instance.py", "np.full(3, 1.0 / (F.SIGMA_BG_RW**2 * dtb))", "np.full(3, 1.0 / F.SIGMA_BG_RW**2)",
     "random-walk precision without Dt_b", [K_C25]),
    # -- verifier pass 3 survivors #1-#10 (docs/verification/stage-b_m3-qcqp-derivation_pass3.md
    #    Sec. 4.3), finding M3-F05, plus the non-finite check and the FD-step change made for it --
    ("p3-lf-sfloor-1e-6", "lifting.py", "S_FLOOR = 1e-12", "S_FLOOR = 1e-6",
     "guard threshold 1e6 times too large", [K_SGUARD]),
    ("p3-lf-guard-strict", "lifting.py", "if abs(s) <= S_FLOOR", "if abs(s) < S_FLOOR",
     "the tie |s| = S_FLOOR max|z| decodes", [K_SGUARD]),
    ("p3-lf-guard-absolute", "lifting.py", "if abs(s) <= S_FLOOR * max(zmax, 1e-300):", "if abs(s) <= S_FLOOR:",
     "guard not scale invariant", [K_SGUARD]),
    ("p3-lf-guard-floor-1", "lifting.py", "max(zmax, 1e-300)", "max(zmax, 1.0)",
     "guard absolute below max|z| = 1", [K_SGUARD]),
    ("p3-lf-guard-uses-mean", "lifting.py", "zmax = float(np.abs(z).max())", "zmax = float(np.abs(z).mean())",
     "guard relative to mean|z|", [K_SGUARD]),
    ("p3-lf-copy-shallow-b", "lifting.py", "[np.array(x) for x in self.b])", "list(self.b))",
     "State.copy aliases the bias arrays", [K_COPY]),
    ("p3-co-unknown-variant-silent", "cost.py", 'raise ValueError(f"unknown cost variant {variant!r}")', "pass",
     "a mistyped variant silently drops the vel/pos IMU terms", [K_VARIANT]),
    ("p3-init-BASE-ANISO-swap", "__init__.py", "from .lifting import BASE, ANISO, Layout",
     "from .lifting import ANISO as BASE, BASE as ANISO, Layout", "package exports BASE/ANISO swapped",
     [K_EXPORTS]),
    ("p3-init-drop-build_instance", "__init__.py", "from .instance import build_instance  # noqa: F401\n", "",
     "build_instance not exported", [K_EXPORTS]),
    ("gb-fd-forward", "gtsam_boundary.py", "J[:, k] = (lp - lm) / (2 * h)", "J[:, k] = lp / h",
     "forward-difference J: structural zeros no longer exact", [K_JEXACT]),
    ("p3-lf-no-finite-check", "lifting.py", "if bad.size:", "if False:",
     "NaN outside s propagates; inf gives the 's is zero' message", [K_NONFINITE]),
    ("gb-fd-step-1e-3", "gtsam_boundary.py", "h: float = 0.1)", "h: float = 1e-3)",
     "the pre-F05 FD step (accurate only to ~6e-13)", [K_JEXACT]),
    # pass-4 survivors (M3-F06)
    ("p4-lf-truth-b-alias", "lifting.py", "[inst.b_true.copy() for _ in inst.truth]", "[inst.b_true for _ in inst.truth]",
     "truth_state's biases are one array, the instance's own b_true", [K_TRUTHCOPY]),
    ("p4-lf-truth-v-alias", "lifting.py", "[t[1].copy() for t in inst.truth]", "[t[1] for t in inst.truth]",
     "truth_state's velocities alias Instance.truth", [K_TRUTHCOPY]),
    ("p4-co-default-partial", "cost.py", 'variant: str = "iso"', 'variant: str = "partial"',
     "terms(inst) silently returns the partial (C33) weighting", [K_DEFVARIANT]),
    ("p4-gb-no-gravity-guard", "gtsam_boundary.py",
     'raise ValueError(f"MakeSharedU expects z-up gravity (0, 0, -g) with g > 0, got {g_w!r}")', "pass",
     "non-z-up gravity silently accepted (x/y dropped, sign lost)", [K_GRAVITY]),
    ("p4-lf-floor-zero", "lifting.py", "if abs(s) <= S_FLOOR * max(zmax, 1e-300):",
     "if abs(s) <= S_FLOOR * max(zmax, 0.0):", "decoder guard floor 1e-300 dropped", [K_FLOOR]),
    ("p4-lf-guard-ge", "lifting.py", "if abs(s) <= S_FLOOR * max(zmax, 1e-300):",
     "if abs(s) <= S_FLOOR * max(zmax, 1e-300) or abs(s) > 1e300:", "decoder rejects large |s|", [K_LARGE]),
    ("p4-lf-project-det-nosign", "lifting.py", "np.sign(np.linalg.det(U @ Vt))", "np.linalg.det(U @ Vt)",
     "det correction +-(1 + few eps) instead of exactly +-1", [K_DETSIGN]),
]

KILLERS = sorted({k for m in MUTANTS for k in m[5]})


def _copy_tree(dst: pathlib.Path) -> pathlib.Path:
    for name in ("conftest.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, dst / name)
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "tests").mkdir()
    shutil.copy2(ROOT / "tests" / "test_qcqp.py", dst / "tests" / "test_qcqp.py")
    shutil.copytree(ROOT / "tests" / "fixtures", dst / "tests" / "fixtures")
    return dst


def _run(root: pathlib.Path, nodeids):
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-o", "addopts=", "--no-header", "-rf", *nodeids],
                          cwd=root, capture_output=True, text=True, timeout=1800)
    failed = {ln.split(" ", 2)[1] for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")}
    return proc.returncode, failed, proc.stdout[-3000:]


def test_mutant_list_is_well_formed():
    assert len(MUTANTS) >= 15
    assert len({m[0] for m in MUTANTS}) == len(MUTANTS)
    for mid, mod, old, new, _, killers in MUTANTS:
        text = (ROOT / "src" / "qcqp" / mod).read_text()
        assert text.count(old) == 1, f"{mid}: pattern occurs {text.count(old)} times"
        assert old != new and killers


def test_control_unmutated_copy_passes_every_killer(tmp_path):
    root = _copy_tree(tmp_path)
    code, failed, out = _run(root, KILLERS)
    assert code == 0 and not failed, out


@pytest.mark.parametrize("mid,mod,old,new,why,killers", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_mutant_is_killed(tmp_path, mid, mod, old, new, why, killers):
    root = _copy_tree(tmp_path)
    path = root / "src" / "qcqp" / mod
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    code, failed, out = _run(root, killers)
    survivors = [k for k in killers if k not in failed]
    assert code != 0 and not survivors, f"SURVIVOR {mid} ({why}): {survivors} passed\n{out}"
