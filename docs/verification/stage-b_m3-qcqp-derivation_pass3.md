# Stage B verification, pass 3: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 3. Base commit 4931a68 (implementer's response to pass-2 finding M3-F04).
Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py, docs/claims/M3.yaml (M3-C01..C35, C39), and
one claim given in the brief only, M3-C42. I did not read docs/03_qcqp_derivation.md, experiments/m3_scratch/, or any
seeds file. I did not edit src/, tests/ or docs/claims/.

Scripts lived in a scratch directory inside the worktree (`_v3scratch/`, untracked, deleted before the final
commit). Mutant copies went to Python `tempfile` directories. The scripts were:

* `ref3.py`: a new independent reference. It imports neither src nor tests. It has its own PIMs (GTSAM, retyped
  literals); Ubar from `deltaXij()`; J from a 5-point stencil on the scipy `logm` chart (h = 0.05, since the map is
  exactly linear); the truth from Van Loan block exponentials; z by plain concatenation in claim order; its own
  5x5 residual and cost; its own decoder (SVD with det(U) det(V^T) correction); and its own GTSAM factor graph
  (`Diagonal.Precisions`, NavState-ordered taus, `BetweenFactorConstantBias` in (accel, gyro) order).
* `cmp3.py`, `dec3.py`, `claimsA..D.py`: src compared against `ref3`, plus the per-claim computations.
* `mutants3.py`, `harness3.py`: 78 new mutants and 31 replays. Each is a real, exactly-once textual edit on a
  tempfile copy of src + tests/test_qcqp.py + conftest + fixtures. The judge is the full tests/test_qcqp.py
  (slow tests included, `-x`).
* `dump3.py`, `diff3.py`, `measure3.py`: record 454 observable output groups of src/qcqp and diff a mutant against
  the clean tree. The inputs are fixed and include decoder inputs at seven scales, singular / reflected blocks,
  guard probes around the threshold, a singular PSD weight, aliasing probes, the package exports and every forms
  helper. These diffs supply the measured effect behind each equivalence ruling.

## Verdict summary

* **F04 (pass 2): FIXED.** All 6 pass-2 survivors whose code still exists are killed, replayed as real file
  edits. Each kill comes from a check against an independent reference. The other 6 survivors were in helpers
  that 4931a68 deleted. I confirmed those helpers were never used: at 96389b8, `git grep` over src and tests finds
  only their definitions, and at 4931a68 the only hit is a comment in test_mutation_m3.py. The cannot-fail
  assertions named in pass 2 were removed (C23 bitwise identities, C31 `ptp(qm)`) or labelled MATH CHECK (C03, C05,
  C06, C28). One residual tautology is left (Sec. 5).
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C42 REFUTED. 0 UNTESTABLE.**
* **Fresh sweep: 78 new mutants over all 8 files of src/qcqp, plus 31 replays of earlier mutants (109 real-file-edit
  runs).** New: 63 killed, 15 survive. Replays: 20 killed, 11 survive. Of the 26 survivors, **16 are equivalent**
  (measured or proved) and **10 are NON-EQUIVALENT (all low)**.
* **GATE (zero surviving non-equivalent mutants): FAILED.** Five of the 10 sit in the s ~ 0 guard that 4931a68
  added. Its threshold, its tie and its scale semantics are untested; the tests only bracket S_FLOOR between about
  5e-15 and 0.5. The other five are: `State.copy` aliasing of the bias arrays; `cost.terms` silently accepting an
  unknown variant; the package exports in `__init__.py` (untested); and the forward-difference J
  (pass-1 `gb-fd-forward`, re-ruled). Every mutant that changed z, Q, a family, the instance data, the GTSAM
  boundary, the projection or the decoder arithmetic was killed.

## 1. Finding M3-F04 ruling: FIXED

| pass-2 survivor | status at 4931a68 | replay result (first killing test, measured failure) |
|---|---|---|
| lf-project-before-divide (medium) | replayed | killed: test_decoder_projection_is_nearest_rotation (sc = -1: `\|R - ref_R\|` fails, assertion -1.0); also targeted by test_decoder_rounding_of_perturbed_lifted_states |
| lf-truth-state-bbar (medium) | replayed | killed: test_instance_matches_independent_gtsam_reference[zero] (`truth_state(I).b == LIT.BIAS_TRUE` is False) |
| lf-project-sign-detA | replayed | killed: test_decoder_degenerate_inputs (zero block: det 0, not 1) |
| cn-family-residuals-abs | replayed | killed: test_c13[False] (signed g vs own evaluation, family P0) |
| fm-outer-nosym | replayed | killed: test_form_helpers_are_symmetric_and_evaluate_correctly ("outer" not symmetric) |
| fm-sym-identity | replayed | killed: same test (sym check line 1671) |
| lf-bg-slice, lf-ba-slice, in-n_kf-edges, gb-GAL_ROWS-pos-vel, fm-cmat-left-transposed, fm-quadT-identity | helper deleted | nothing left to mutate. Before deletion (96389b8) `Layout.bg/ba`, `Instance.n_kf`, `GAL_ROWS`, `cmat_left` and `quad_T` had no use in src or tests: every `n_kf` hit is `Layout.n_kf` / `layout.n_kf` / `L.n_kf` / `self.n_kf`. The suite passes without them |

The s ~ 0 guard requested in pass 2 Sec. 6 exists now, and its existence is tested (s = 0, +/-1e-14 and NaN raise;
1e-6 z decodes). Its threshold semantics are not tested; see Sec. 4.3 #1-#5. This is a new gap that the fix itself
introduced, so it is filed as M3-F05 (Sec. 6). F04 itself is closed.

## 2. Independent re-checks (reference imports no src)

| check | result |
|---|---|
| instance vs ref3 PIMs (both bbar, both edges) | dR 1.0e-14 (predict vs deltaXij), dv 0, dp 4.4e-16; taus, W_R, G2, W_vp bitwise equal; Dt = 0.5000000000000003; truth (Van Loan) 2.2e-15; truth_state.b == b_true exactly for bbar = 0 and b_true |
| J: src (central FD h = 1e-3) vs ref3 (5-point logm chart) | 2.3e-13 to 6.2e-13. vs exact LS fit of the linear map (residual 8e-16 to 1.3e-15): clean 2.3e-13 to 6.2e-13, ref3 7e-15 to 1.3e-14 |
| index map: src z_of_state vs ref3 concatenation, 30 states x both bbar | 0 exactly (same J); 1.2e-13 / 2.0e-13 with ref3's J; aniso rho 3.2e-16 / 3.8e-16 relative |
| cost vs own 5x5 cost, 100 random states | factored 3.6e-15 / 5.2e-15, dense 3.4e-15 / 5.4e-15 (relative) |
| **cost vs own GTSAM graph**, bbar = b_true, 40 perturbed states per eps | factored/GTSAM - 1 = 2.31e-4, 5.38e-6, 3.16e-7 at eps 1e-2, 1e-3, 1e-4 (C26 bounds 5e-4, 5e-5, 5e-6); dense z^T Q z identical to 3 digits |
| truth costs | factored 2.1e-21, GTSAM 6.4e-22, **dense z^T Q z 4.3e-8** (cancellation, max\|Q\| ~ 1.1e10) |
| bbar = 0 at truth | M3 3.296776e-3 vs GTSAM 3.315734e-3 (0.57%, outside C26's domain) |
| **decoder, project = False**, s in {1, -1, -3.7, 0.05}, 60 states x noise {0, 1e-2, 0.3, 2.0}, some reflected R_0 | R, v, p, b vs ref3 decode of the s = 1 vector: <= 2.0e-16 relative |
| **decoder, project = True**, same inputs | v, p, b <= 2.0e-16. R, well-posed blocks: 0 at s = +/-1, 6.1e-14 at s = -3.7, 2.1e-14 at s = 0.05. For 36 ill-posed blocks (exact reflections, sigma2 = sigma3, where the nearest rotation is not unique) the two decoders choose different minimisers, but their Frobenius distances to the block agree to 4.4e-16. det in [1 - 2.7e-15, 1 + 2.4e-15] |
| SDP-style rounding: X = z z^T + noise N N^T, top eigenvector x sqrt(lambda), both signs | s = +/-1.0000 and identical results for both signs: \|R - R_true\| 2.8e-16 / 7.5e-7 / 1.4e-3 at noise 0 / 1e-6 / 1e-3; det 1 |
| **s ~ 0 guard semantics** (z_truth, max\|z\| = 1.9035) | raises iff s is non-finite or \|s\| <= 1e-12 * max(max\|z\|, 1e-300). Measured: s = 0, +/-1e-14, 0.999 x threshold and exactly the threshold raise; 1.001 x threshold decodes; NaN, +/-inf raise; tie (max\|z\| = 1, s = 1e-12) raises, next double decodes. Scale invariant from 1e+300 down to 1e-300 x z (all decode); 1e-320 x z raises (the 1e-300 floor makes the test absolute below 1e-300). z = 0 raises. **z with an inf entry and s = 1 raises** ("s is zero relative to max\|z\| = inf"); **z with a NaN entry and s = 1 decodes** (NaN propagates to the output) |

## 3. Claims

Each verdict rests on my own computation (ref3 + claims scripts), with src as the object under test. The
suite's claim tests also pass on the clean tree (Sec. 7).

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | Sec. 2 index map; n = 124 / 136, s at 123 |
| C02 | CONFIRMED | 112, 12; V' columns of imu0.vel exactly -1/2 I3 |
| C03 | CONFIRMED | all 7 triples n = 39K + 6M + 16, n - direct = 39 |
| C04 | CONFIRMED | own FD (GTSAM Logmap, h = 1e-6), 4 fixture cases + 6 random time-varying streams: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| 3.95e-10 (<= 1e-8); \|J_R\| 0.37-0.50. Own right-Jacobian series vs FD, 200 x: 1.45e-10; block differences 0 |
| C05, C06 | CONFIRMED | own algebra with own edges, 4000 draws: max\|r\| 1.8e-15, \|t_j\| 0, A34 = B34 = Dt exactly, bottom rows 0 exactly, arbitrary X_j min max\|r\| 0.843; C06 3.3e-15 |
| C07 | CONFIRMED | src forms vs own 5x5, 300 states: 3.2e-14 / 1.0e-14 (own J) |
| C08 | CONFIRMED | coefficient blocks exact; truth 2.1e-21; Y_0 perturbation min 3.6e-5 >= 1e-6 |
| C09-C12 | CONFIRMED | src families on own z: truth <= 5.1e-14; random (200): P1 1.7e-13, P3 2.3e-13, RL 2.3e-13 (<= 1e-12); P4 1.4e-12, RR 1.6e-11 (<= 1e-10) |
| C13 | CONFIRMED | min family max 1.0e-5 (RB1) to 1.3e-5 (RR) > 1e-7 |
| C14 | CONFIRMED | sizes as stated; every family full rank; primary 106; joint 472 -> rank 453 (sv 0.43 -> 6.4e-16); aniso 490 -> 471, primary 118 |
| C15 | CONFIRMED | own row-handedness forms match src column P2 exactly (mismatch 0) |
| C16 | CONFIRMED | rank 11 per KF; trace identity exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric (base and aniso); the only b != 0 is P0, b = 1 |
| C18 | CONFIRMED | own rerun (rng 1818): primary 60/60 converged, consistency 9.1e-12, \|det - 1\| 4.4e-16. Drop P2: det -1.000 found. Drop P3: consistency error 1.6e2. Drop P1: \|det\| 1.7e-72. Suite test (200 starts, rng 7) passes |
| C19 | CONFIRMED | 150 starts: 113 SO(3), 37 zero, 0 other |
| C20 | CONFIRMED | P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied <= 5.1e-14 |
| C21 | CONFIRMED | P1 = RA1 = 1.0; others <= 5.1e-14 (both bbar) |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: >= 1.3e-2; satisfied <= 4.7e-15 |
| C23 | CONFIRMED | decode(-z) == state exactly (project = False); <= 3.5e-14 with project = True (ref3's scipy-expm rotations are only orthogonal to 3.4e-14) |
| C24 | CONFIRMED | iso 3.6e-15 / 5.2e-15; aniso 3.2e-15 / 3.5e-15 (relative, own 5x5) |
| C25 | CONFIRMED | iso / aniso / partial at both bbar: exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0; bias block equal to own rw sum exactly |
| C26 | CONFIRMED | Sec. 2 |
| C27 | CONFIRMED | dev rot 2.011e-2 ... 3.1e-4, vel -1.223e-2 ..., pos -2.098e-3 ...; halving rot 0.250/0.249/0.249, vel 0.501/0.500/0.500, pos 0.504/0.502/0.501 |
| C28 | CONFIRMED | 2.088e-7, 2.605e-8, 3.253e-9, 4.064e-10; factors 8.02/8.01/8.00; \|x\| 0.060 |
| C29 | CONFIRMED | half 3.3e-7 (<= 5e-6), full 0.74 (>= 0.1), own graph |
| C30 | CONFIRMED | own symmetric root, own z, own local parametrisation: rank 45, cond 3.95e7 / 4.53e7 |
| C31 | CONFIRMED | iso 4.020000 (ptp 5.2e-13); diag(1,2,5) range [1.341, 6.693] |
| C32 | CONFIRMED | own sampler seed 2: 2.6e-14, 2.5e-14 (<= 1e-12); diag(1,2,5) 7.9e-2, random SPD 2.1e-1 (>= 1e-4); diag(1,1,1.01) 8.7e-4; rank 2015 |
| C33 | CONFIRMED | Q_partial vs own direct 3.3e-15; vs own GTSAM graph 8.8e-5, 2.1e-6, 1.6e-7 (bounds 0.05 eps); D min eig 3.46e7; G2 corr -0.867 |
| C34 | CONFIRMED | body 2.1e-3, 1.8e-4, 2.0e-5 (<= eps); world 6.2e-2, 7.9e-2, 6.1e-2 (>= 1e-2) |
| C35 | CONFIRMED (fragile tolerance) | own sampler seed 4: rank 2015, null 263, gap 1.9e-7 -> 4.5e-16; spans 60/92/106/136/226/236; max projection residual 2.6e-11 (<= 1e-10). **With scipy `expm` rotations (orthogonal only to ~3e-14) the same run gives a projection residual of 6.9e-10 > 1e-10** (gap 1.9e-7 -> 4.3e-15). The spans are unchanged, but "inside it to 1e-10" holds only for rotations orthogonal to about 1e-15 |
| C39 | CONFIRMED | Dt 0.5000000000000003 both; taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6, own = src; rw diag 5.5402e9 / 2.2222e5 |
| **C42** | **REFUTED** | below |

**M3-C42 (brief only): REFUTED.** The shipped decoder divides every block by s before projecting, so -z decodes to
the same state as z, not to its reflection. At the ground-truth lifted z (bbar = 0), decoding -z gives:

* project = False: \|R_k(-z) - R_k,true\| = 0 exactly, det = +1.000000000000000 for all k;
* project = True: \|R_k(-z) - R_k,true\| = 2.2e-16, det = +1;
* \|R_k(-z) + R_k,true\| = 1.98, so the output is not -R_k.

ref3's decoder gives the same results. For an SDP-style eigenvector of either sign (s = -1.0000 or +1.0000), the
recovered rotations are identical (7.45e-7 from truth at noise 1e-6, 1.35e-3 at noise 1e-3) and det = 1. Decoding
at s in {-1, -3.7} matches the s = +1 decode to <= 6.1e-14 (Sec. 2). So no sign fix of the eigenvector is needed
before calling `state_of_z`. The claimed behaviour (det -1 or -R) appears only for the pass-2 mutant
"project before dividing by s", and that mutant is now killed. (Forcing s > 0 is harmless but unnecessary.)

## 4. Mutation sweep

### 4.1 New mutants (78; none in pass 1, pass 2 or tests/test_mutation_m3.py)

By file: lifting 19, constraints 17, cost 13, fixture 8, instance 8, forms 7, gtsam_boundary 4, `__init__` 2.
Ids are `p3-*`. Each is one exactly-once substitution, applied on a tempfile copy.

**Killed (63), by first failing test.** Every killer checks against an independent reference: GTSAM PIMs and
retyped literals, M2_REF values, the test-side 5x5 residual/cost, hand-built constraint matrices, the claims'
literal sizes/labels, or a test-side SVD / eigh / numpy evaluation.
* test_decoder_degenerate_inputs: p3-lf-sfloor-1e-16, p3-lf-guard-no-isfinite.
* test_decoder_projection_is_nearest_rotation: p3-lf-project-UVt-transposed, p3-lf-project-D-first-axis.
* test_c01_index_map: p3-lf-decode-R-abs-s, p3-lf-z-s-entry-s, p3-lf-Yp-W-transposed, p3-lf-rho_p-uses-Rj,
  p3-lf-rho_v-offset, p3-lf-n-aniso-3, p3-lf-s_form-index, p3-co-rp-Dt-vi-sign, p3-co-rp-jside-cbar-cv,
  p3-co-Rcv-transposed (the cost mutants reach C01 through the aniso rho entries, checked against resid5).
* test_instance_matches_independent_gtsam_reference: p3-lf-truth-state-v-from-p, p3-fx-m2-T2-sign,
  p3-fx-hat-a0-sign, p3-fx-so3exp-cos-th, p3-fx-truth-p-no-vT, p3-fx-streams-acc-no-bias, p3-fx-fbody1-literal,
  p3-fx-n-edges-1, p3-fx-m1-K-coeff, p3-gb-predict-lift-t-Dt, p3-gb-J-no-Ubar-inv, p3-gb-gal3-t-negated,
  p3-gb-sharedU-sign, p3-gb-cov-blocks-transposed (W_vp, since the vel/pos block is not symmetric), p3-in-S2-vp-uses-pp,
  p3-in-edge-j-equals-i, p3-in-WR-from-pos, p3-in-J-at-zero-bias, p3-in-dv-dp-swap.
* test_c09: p3-cn-P0-b-2, p3-cn-P2-s-col-b, p3-cn-P3-W-transposed, p3-cn-residual-no-b, p3-cn-edge-cp-uses-Jnu,
  p3-cn-jside-uses-Y, p3-fm-matmul-B-transposed, p3-fm-linT-wrong-axes. test_c10: p3-cn-RB1-Y-transposed,
  p3-cn-RB2-ww-outer-sign. test_c11: p3-cn-RG-VP-cv-cv, p3-cn-RL-P-uses-cv, p3-cn-RH-ea-negated, p3-cn-RS-uses-cv.
  test_c12: p3-cn-P4-sign, p3-cn-RR-cross-rv-rv. test_c14: p3-cn-RC1-drop-P. test_c16: p3-cn-UPPER-lower (the
  diagonal positions in upper-triangle order move). test_P1_RA1: p3-cn-RA1-half-ss, p3-fm-sym-factor-1.
* test_c05_shipped: p3-co-rR-M-full-gam. test_c33: p3-co-partial-D-kron-swap. test_c08: p3-co-prior-v0-uses-p0bar.
  test_c24: p3-co-rw-identity, p3-co-assemble-noW, p3-in-gnss-kf0. test_whitened_residual: p3-co-whiten-eigh-WT.
  test_form_helpers: p3-fm-dot-nosym, p3-fm-matvec-nosym, p3-fm-cross-nosym.

**Survive (15):** see 4.3.

### 4.2 Replays of earlier mutants (31, real file edits)
* The 6 F04 survivors that still exist (Sec. 1): all killed.
* 14 earlier killed mutants: lf-decode-v-mul, lf-decode-b-abs, lf-matform-rowmajor (test_c01); lf-project-nodet
  (test_decoder_projection, 2.0 vs 1e-12); lf-decode-no-s-guard (test_decoder_degenerate_inputs, DID NOT RAISE);
  co-whiten-noclip (test_whitened_residual); cn-family-max-signed (test_c13); gb-from-bias-swap
  (test_bias_boundary_converters); rw-missing-dt and fx-sigma-gnss (instance test; fx-sigma-gnss is also killed by
  test_c24, pass 2); RA1-built-from-R^T R (test_P1_RA1); cn-PRIMARY-incl-RA1 (test_c14); co-gnss-sigma-not-squared
  and co-rw-range (test_c24, 1.7e-5 and 0.71 vs 1e-12). All killed.
* 11 earlier equivalents: fm-matvec-nm-swap, in-bbar-not-copied, co-aniso-Wvp-T, co-assemble-WT,
  co-whiten-clip-floor, fx-so3exp-branch, gb-fd-step, in-S2-offdiag-swap, cn-mk-nosym, cn-vech-tril (pass 2), and
  gb-fd-forward (pass 1: `J = lp / h`). All 11 survive. Ten are re-confirmed equivalent. **gb-fd-forward is
  re-ruled NON-EQUIVALENT** (4.3 #10).

### 4.3 COMPLETE survivor list (26) with rulings

Threshold: 1/10 of the tolerance in the relevant claim's test_hint / domain, never below the 8-ulp floor
(~1.8e-15 relative). A claim-free contract (exception vs return, aliasing) changes an output by definition, so it
is non-equivalent unless a proof shows that no src output changes.

| # | mutant | measured effect (diff3 vs clean, 454 groups) | ruling | class |
|---|---|---|---|---|
| 1 | **p3-lf-sfloor-1e-6** (`S_FLOOR = 1e-6`) | 7 groups: decoding of z_truth with s = 1.904e-12 ... 1e-6 (ratio 1e-12 ... 5e-7) and of e_0 + 1e-7 e_s now raises ValueError instead of returning z/s | **NON-EQUIVALENT** | low: the guard's threshold is not pinned. The tests bracket S_FLOOR only between ~5e-15 (from s = 1e-14, max\|z\| 1.9) and ~0.5 (from 1e-6 z) |
| 2 | **p3-lf-guard-strict** (`<` for `<=`) | 1 group: the tie \|s\| = S_FLOOR max\|z\| (e_0 + 1e-12 e_s) decodes instead of raising | **NON-EQUIVALENT** | low: boundary only; the documented `<=` is untested |
| 3 | **p3-lf-guard-absolute** (`abs(s) <= S_FLOOR`) | 5 groups: s = 1.9e-12 at max\|z\| 1.9 decodes (clean raises); 1e-13, 1e-200, 1e-300, 1e-305 x z_truth raise (clean decodes) | **NON-EQUIVALENT** | low: the guard is no longer scale invariant; 1e-6 z is the only scale tested |
| 4 | **p3-lf-guard-floor-1** (floor 1e-300 -> 1.0) | 4 groups: 1e-13 ... 1e-305 x z_truth raise | **NON-EQUIVALENT** | low: same |
| 5 | **p3-lf-guard-uses-mean** (mean\|z\| for max\|z\|) | 2 groups: s = 1.9e-12 and the tie decode | **NON-EQUIVALENT** | low |
| 6 | **p3-lf-copy-shallow-b** (`State.copy` keeps the b arrays) | values identical; `copy().b[k] is original.b[k]` True for all k (R, v, p still copied) | **NON-EQUIVALENT** | low: the method's purpose is an independent copy, and an in-place edit of a copy's bias corrupts the original. No test mutates a copy in place |
| 7 | **p3-co-unknown-variant-silent** (`raise ValueError` -> `pass`) | `terms(inst, BASE, "bogus")` returns 2 rotation terms + priors/GNSS/rw (no vel/pos) instead of raising | **NON-EQUIVALENT** | low: a mistyped variant in M4 silently drops the velocity/position IMU terms |
| 8 | **p3-init-BASE-ANISO-swap** | `src.qcqp.BASE.n` = 136, `src.qcqp.ANISO.n` = 124 | **NON-EQUIVALENT** | low: package-level exports are untested (all tests import submodules) |
| 9 | **p3-init-drop-build_instance** | `hasattr(src.qcqp, "build_instance")` False | **NON-EQUIVALENT** | low: same |
| 10 | **gb-fd-forward** (replay, `J[:, k] = lp / h`) | J changes by up to 1.24e-12. Against the exact linear map (LS fit, residual <= 1.3e-15): clean 2.3e-13 to 6.2e-13, **mutant 4.0e-13 to 1.06e-12 (worse)**. **J[0:3,3:6] becomes 3.5e-15 / 5.2e-15 instead of exactly 0** (C04 states exactly 0). Max family residual on exactly lifted states (61 states) rises from 1.3e-13 / 2.1e-13 to 4.1e-13 / 3.6e-13: a change of 2.8e-13, more than 1/10 of C09's 1e-12 | **NON-EQUIVALENT** (re-ruled; pass 1 had equivalent) | low: still within every claim tolerance, but worse than clean and above the rule-1 threshold. A test asserting `inst.edges[e].J[0:3,3:6] == 0` exactly (C04) would kill it |
| 11 | p3-co-partial-G2-T | partial terms rel 3.9e-17, Q rel 3.4e-19, wr <= 4.5e-16 rel; asymmetry of G2: 9.3e-10 abs = 3.9e-17 rel (edge 0, bbar = b_true), 0 elsewhere | EQUIVALENT | below the 8-ulp floor; G2 and G2^T are the same computed inverse of the exactly symmetric S2, equally accurate |
| 12 | p3-co-factored-abs (`abs(r W r)`) | identical on all groups, incl. 20 points x every term with a singular PSD W_vp | EQUIVALENT | proof: W_t is PSD, so the exact r^T W r >= 0, and the computed value differs from it by <= gamma_n \|r\|^T\|W\|\|r\|. abs changes only negative computed values, i.e. by at most twice that rounding bound (measured 0) |
| 13 | p3-co-aniso-assert-removed | identical; `terms(BASE, "aniso")` still raises AssertionError | EQUIVALENT | proof: the aniso branch calls `layout.rho_v(e)`, which asserts `self.aniso` before any term is returned. Same exception type, only the message differs |
| 14 | p3-fm-outer-no-broadcast | bitwise identical on all 454 groups (incl. outer of (3,3,n) x (n,) and (n,) x (3,n)) | EQUIVALENT | proof: `f[..., :, None] * g[..., None, :]` broadcasts the leading dims exactly as `broadcast_arrays` does (right-aligned, trailing axis appended to both). broadcast_arrays returns views, so the values are identical |
| 15 | p3-in-prior-not-copied | values identical; `prior[q] is truth[0][q]` True | EQUIVALENT | proof: every write in src goes to a freshly allocated local (complete list: z, f, Q, J, d, `_EPS`); nothing writes in place into an Instance field, and Instance is frozen. Aliasing two fields of the same frozen instance changes no src output |
| 16 | p3-in-zero-key-neg-zero (`0.0 * BIAS_TRUE`) | bitwise identical (bbar holds -0.0 entries, == 0.0) on every group, incl. J, dR, covariances, Q, families | EQUIVALENT | -0.0 and +0.0 behave the same in every arithmetic use here (x - (-0) = x, sums with +/-0) |
| 17 | fm-matvec-nm-swap (replay) | bitwise identical | EQUIVALENT | sym(X^T) = sym(X) bitwise |
| 18 | in-bbar-not-copied (replay) | values identical; `build_instance(a).bbar is a` | EQUIVALENT | proof as #15. The aliasing is with the caller's argument, and src never writes into it |
| 19 | co-aniso-Wvp-T (replay) | aniso terms rel 3.9e-17 (W_vp asymmetry 3.9e-17 rel); factored cost rel 3.2e-16; whitened residual 3.4e-11 / 5.3e-11 abs on entries ~4e4 (~1e-15 of the vector) | EQUIVALENT | at the floor; also << 1/10 of C30's 1e-12 relative cut and the whitening test's 1e-9 |
| 20 | co-assemble-WT (replay) | Q partial rel 3.4e-19 / 4.2e-20 | EQUIVALENT | below the floor |
| 21 | co-whiten-clip-floor (replay) | instance and C34 weights bitwise identical; singular caller W only: 4.2e-9 abs, 5.6e-11 rel per entry | EQUIVALENT | proof: every claim-domain weight has min eigenvalue >= 400 >> 1e-12, so clip(w, 1e-12) = clip(w, 0) = w bitwise. The singular weight lies outside every claim domain |
| 22 | fx-so3exp-branch (replay) | only the \|w\| = 1e-9 probe: 3.0e-18 abs | EQUIVALENT | the branch is never reached on the instance (\|ROT_VEC_0\| = 0.37, \|wT\| >= 0.2); at the probe, below the floor |
| 23 | gb-fd-step h = 0.1 (replay) | J changes 2.4e-13 to 5.1e-13 rel; error vs the exact map: clean 4.3e-13 / 6.2e-13, **mutant 1.9e-14 / 1.8e-14 (better)**. Exact zeros preserved. Family residual on exactly lifted states 1.3e-13 / 2.1e-13 -> 1.18e-13 / 1.18e-13 (change <= 9e-14 < 1e-13). Costs: rel 6e-16 on ~1e9; truth 2.5e-23 abs | EQUIVALENT | rule 1: every change is below 1/10 of the governing tolerance (C04 1e-8, C09 1e-12, C24 1e-12, C26 1e-15 abs), and the mutant is more accurate |
| 24 | in-S2-offdiag-swap (replay) | bitwise identical | EQUIVALENT | preintMeasCov traces bitwise symmetric here |
| 25 | cn-mk-nosym (replay) | bitwise identical | EQUIVALENT | every Q passed to `_mk` is already exactly symmetric |
| 26 | cn-vech-tril (replay) | identical on every constraint matrix; differs only on the non-symmetric probe (9.0) | EQUIVALENT | vech is documented for symmetric input; every A_k is exactly symmetric (C17 verified) |

Totals: 26 survivors. **Non-equivalent 10 (#1-#10), all low. Equivalent 16 (#11-#26).**

## 5. Tests that cannot fail or compare code with itself (after 4931a68)

1. **test_c31, `np.ptp(qr) <= 1e-6 mean(qr)`** (added in place of the removed `ptp(qm)`). qr is the lam^2
   coefficient of the test-side J_direct along v_1 + lam d. v_1 enters J_direct only through r_v of edges 0 and 1,
   each with coefficient +/-I, so qr = (tau_v,0 + tau_v,1)\|d\|^2 for every R_0. This assertion cannot fail and sees
   no src. The qm-vs-qr assertion remains a real check of Q's v_1 block.
2. **test_c02** `15*3+6*3+24*2+1 == 112` and `124-112 == 12`, and **test_c03**
   `direct == 39K + 6M - 23` (an algebraic identity for every K, M): constant arithmetic that cannot fail. Both
   are labelled or are claim arithmetic by nature.
3. Test-side-only parts that cannot fail because of src: test_c15's col-vs-row loop, test_c26
   `gtsam_cost(truth) <= 1e-15`, test_c33's D / G2 checks on `ref`, test_c39's `ref` asserts, the first half of
   test_c04, and test_c05/C06/C28 (now labelled MATH CHECK, with src-bound twins for C05/C06). These are legitimate
   claim tests with no mutation coverage.
4. test_c30's rank is invariant under per-term positive rescaling (already noted in the test).
5. **Untested surface (all of it shows up as survivors):** the guard's threshold / tie / scale semantics,
   `State.copy` independence of b, `cost.terms` rejection of unknown variants, package exports, and exactness of
   the instance J's structural zeros.

The pass-2 items are fixed: the C23 bitwise identities are removed (the decoder part, now also at project=True,
is real), C31 `ptp(qm)` is removed, and C03/C05/C06/C28 are labelled.

## 6. Findings and risks for M4

* **M3-F05 (new, gate-blocking, low): 10 non-equivalent survivors (Sec. 4.3 #1-#10).** Five are in the new s ~ 0
  guard. The other five are State.copy aliasing, silent unknown variant, `__init__` exports, and the
  forward-difference J. Tests that would kill each one: (a) the threshold at 0.5x / 2x of S_FLOOR max\|z\|, and the
  exact tie; (b) a scale sweep (1e-13 and 1e-300 x z decode); (c) mutating `st.copy().b[0]` in place leaves `st`
  unchanged; (d) `pytest.raises(ValueError)` for an unknown variant; (e) `src.qcqp.BASE.n == 124` /
  `src.qcqp.ANISO.n == 136` / `build_instance` exported; (f) `inst.edges[e].J[0:3,3:6] == 0` and `J[9] == 0`
  exactly (C04).
* **Rounding path is sound as shipped:** the s sign is handled (C42 refuted). Projection is correct for s in
  {1, -1, -3.7, 0.05} to <= 6.1e-14 on well-posed blocks. On exact reflections with repeated singular values the
  nearest rotation is not unique, and different but equally near rotations are returned. M4 should still report
  the constraint residual of the rounded point itself, since the decoder ignores the liftings.
* **Guard edge cases:** an inf entry anywhere in z raises with a misleading "s is zero" message. A NaN entry
  outside s does NOT raise and propagates NaN into the decoded state. M4 should check `np.isfinite(z).all()` on
  SDP output before rounding. The guard is absolute below max\|z\| = 1e-300 (irrelevant in practice).
* **Dense Q cancellation:** z^T Q z at truth is 4.3e-8 vs factored 2.1e-21 (max\|Q\| ~ 1.1e10). Use the factored
  cost for primal values and duality gaps, or rescale Q.
* **J accuracy floor:** the shipped central FD (h = 1e-3) is accurate only to 2.3e-13 to 6.2e-13, while
  h = 0.05-0.1 gives ~1e-14 (the map is exactly linear). This sets the floor on lifted V, P and on family residuals
  at independently lifted states (~1-2e-13). It is within all claim tolerances.
* **C35's 1e-10 projection margin depends on rotation orthogonality in the sampler:** 2.6e-11 with
  orthogonal-to-1e-15 rotations, 6.9e-10 with scipy expm (3e-14). Any M4 AutoTight-style rerun needs an exactly
  orthogonal sampler or a looser tolerance.
* **Surrogate gap at bbar = 0:** 0.57% vs GTSAM at truth, not shrinking with eps (outside C26's domain).
* **Redundant families are linearly dependent:** 472 rows, rank 453 (aniso 490, rank 471). RA1 is not redundant in
  the Shor relaxation (pass-2 C41).
