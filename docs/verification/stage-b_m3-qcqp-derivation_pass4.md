# Stage B verification, pass 4: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 4 (agent verify-M3-004). Base commit 9bbf3a3 (the implementer's response to
pass-3 finding M3-F05). Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py, docs/claims/M3.yaml
(M3-C01..C35, C39), and one claim given only in the brief, M3-C43. I did not read docs/03_qcqp_derivation.md,
experiments/m3_scratch/, or any seeds file. I did not edit src/, tests/ or docs/claims/.

Scripts lived in a scratch directory inside the worktree (`_v4scratch/`, untracked, deleted before the final commit).
Mutant copies went to Python `tempfile` directories. The scripts were:

* `ref4.py`: a new independent reference. It imports neither src nor tests. It has its own GTSAM PIMs built from
  retyped literals, and takes Ubar from `deltaXij()`. J is a least-squares linear fit of the **scipy `logm` chart**
  (not GTSAM's Logmap). The truth comes from Van Loan block exponentials, with an exactly orthogonal Rodrigues
  sampler. z is built by plain concatenation in claim order. The C07 affine residual forms are evaluated directly on
  the blocks of an arbitrary z. **Every constraint family is re-implemented as matrix equations on the blocks of z**
  (this is the re-implemented core routine). The reference also has its own cost on arbitrary z, its own 5x5 cost,
  its own decoder and its own GTSAM factor graph (`Diagonal.Precisions`, NavState-ordered information,
  `BetweenFactorConstantBias` in (accel, gyro) order).
* `cmp4.py`, `lin4.py`, `hcmp.py`, `claimsA/B/C.py`, `c43.py`: src compared against `ref4`, the per-claim
  computations, the linearity check on the J map and the FD-step study.
* `prop4.py`: property tests with 1e4 samples (families on lifted states, cost vs the 5x5 cost, decoder vs the own
  decoder).
* `mutants4.py`, `harness4.py`: 95 new mutants, 12 F05 replays and 15 replays of earlier equivalents. Each is a real,
  exactly-once textual edit on a tempfile copy of src + tests/test_qcqp.py + conftest + fixtures. The judge is the
  full tests/test_qcqp.py, slow tests included, run with `-x`.
* `dump4.py`, `measure4.py`: record 518 observable output groups of src/qcqp and diff a mutant against the clean
  tree. Two clean runs give identical dumps, so the dump is deterministic. The groups cover every instance field;
  z at 7 states x 2 layouts x 2 s values x 2 bbar; 120 decoder probes (scales 1e-320 to 1e300, threshold
  0.4x-1e6x, ties at 5 scales incl. sub-1e-300, NaN/inf at 6 positions, singular / reflected / rank-1 blocks, list
  input); every constraint matrix and residual; terms / Q / factored / whitened at 5 weightings incl. a singular
  W_vp; unknown variants; the default variant; every forms helper; the GTSAM boundary incl. non-z-up gravity;
  fixture literals; aliasing probes; and package exports.

## Verdict summary

* **F05 (pass 3): FIXED.** All 10 pass-3 survivors were replayed as real file edits judged by the full suite, and
  all 10 are killed. So are the two reverts of the 9bbf3a3 src changes. The src changes are correct: (a) the
  non-finite check, (b) h = 0.1, justified because the map is linear to rounding (measured), and (c) the message. No
  claim result moves beyond tolerance. `ls_J` is independent of src. The new C31 check is a real check (Sec. 1).
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C43 REFUTED. 0 UNTESTABLE.**
* **Fresh sweep: 95 new mutants over all 8 files of src/qcqp, plus 27 replays (122 real-file-edit runs).** Of the
  new mutants, 84 are killed and 11 survive. Of the 11 survivors, **4 are equivalent and 7 are NON-EQUIVALENT (all
  low)**. All 15 replays of earlier equivalents survive and are re-confirmed equivalent. The harness replay
  (`tests/test_mutation_m3.py`) passes 70/70, i.e. 68/68 mutants are killed by their named killers.
* **GATE (zero surviving non-equivalent mutants): FAILED**, on 7 low survivors (finding M3-F06, Sec. 6). These are:
  two aliasing defects in `truth_state`; the default `variant` of `cost.terms`; the gravity input assert in
  `make_params`; the guard's sub-1e-300 floor; an added upper-scale rejection clause in the guard; and the `sign()`
  of the projection's det correction (a rounding-level defect, about 10 ulp). Every mutant that changed z, Q, a
  family, the instance data, the GTSAM boundary arithmetic, the forms helpers, or the decoder's s / non-finite /
  threshold logic was killed.

## 1. Finding M3-F05 ruling: FIXED

| pass-3 survivor | replay result (full suite, first failing test) |
|---|---|
| #1 p3-lf-sfloor-1e-6 | killed: test_decoder_s_guard_threshold_tie_and_scale (ValueError at 2x threshold) |
| #2 p3-lf-guard-strict | killed: same test (exact tie decodes) |
| #3 p3-lf-guard-absolute | killed: same test (1e6-scale threshold / scale sweep) |
| #4 p3-lf-guard-floor-1 | killed: same test (scale sweep 1e-13 raises) |
| #5 p3-lf-guard-uses-mean | killed: same test |
| #6 p3-lf-copy-shallow-b | killed: test_state_copy_is_independent |
| #7 p3-co-unknown-variant-silent | killed: test_cost_terms_rejects_unknown_variant (DID NOT RAISE) |
| #8 p3-init-BASE-ANISO-swap | killed: test_package_exports (`136 == 124` false) |
| #9 p3-init-drop-build_instance | killed: test_package_exports |
| #10 gb-fd-forward | killed: test_c04_instance_J_structural_zeros_exact_and_J_accuracy (J[0:3,3:6] not exactly 0) |
| revert of the non-finite check (`if bad.size:` -> `if False:`) | killed: test_decoder_degenerate_inputs (DID NOT RAISE) |
| revert of h 0.1 -> 1e-3 | killed: test_c04_instance_J_structural_zeros_exact_and_J_accuracy (J vs LS fit > 5e-14) |

**src change (a), `state_of_z` rejects any non-finite z.** I checked this against my own decoder on 1e4 random z.
The inputs had s in {+/-1, -3.7, 0.05, 1e-6, 1e6}, noise up to 0.5, reflections, and s pushed below the threshold
on every 11th sample. There were 0 raise/no-raise disagreements (909 raised in both). The project=False outputs are
identical. NaN, +inf and -inf at 6 positions (incl. s and an aniso rho slot) all raise "non-finite" and never
"zero". The pass-3 hole is closed: a NaN outside s no longer decodes.

**src change (b), `bias_jacobian` h 1e-3 -> 0.1.** Linearity measured on both edges and both bbar, 200 random d per
amplitude:

* |Log(Ubar^-1 U(bbar + d)) - J d| stays at the round-off level (about 1e-15 absolute) for amplitudes from 1e-3 to
  2. The relative values 4.4e-12 at amp 1e-3 and 3e-14 at amp 0.1 are just that 1e-15 divided by |Log|.
* The additivity defect |L(d1 + d2) - L(d1) - L(d2)| is 1.6e-15 to 2.2e-15 for amp <= 0.1 and grows only with
  magnitude (4e-14 at amp 2). A quadratic term would scale as amp^2, which bounds its coefficient below about 1e-13.

So the map is linear to rounding.

J accuracy against a 400-sample LS fit of the scipy-logm chart (fit residual 0.7e-15 to 1.6e-15):

| h | 1e-3 | 1e-2 | 0.05 | **0.1 (shipped)** | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| max error, 4 edge/bbar cases | 2.3e-13 to 6.1e-13 | 3.3e-14 to 1.1e-13 | 9.1e-15 to 1.6e-14 | **5.7e-15 to 1.1e-14** | 0.9e-15 to 1.8e-15 | 0.7e-15 to 2.4e-15 |

This matches the docstring (4.4e-15 to 1.1e-14) and the commit message. The structural zeros J[0:3,3:6] and J[9]
are exactly 0 in the shipped instance (both bbar, both edges). The change h 0.1 vs 1e-3 moves only J-derived
outputs, as measured by the dump diff:

| output | change (abs) |
|---|---|
| J | 2.3e-13 to 6.1e-13 |
| z | <= 1.2e-13 |
| family residuals at src-lifted states | <= 1.4e-14 |
| Q | <= 8.9e-8, rel 8e-18 |
| factored cost at bbar = 0 states | <= 4e-12, rel <= 7e-10 of the cost, i.e. <= 4e-12 relative to max(1, J), the C24 measure |
| truth cost at bbar = b_true (C26) | <= 1e-22 |

My own claim measurements (Sec. 3) all moved in the better direction or not at all. Family residuals at
independently lifted states fall from 1.7e-13 to 2.3e-13 (pass 3) to <= 1.2e-14 (C09-C11, 1e4 samples). C30 cond
is unchanged (3.946e7 / 4.527e7). C35's projection residual is unchanged at 2.6e-11. **No claim result moves beyond
tolerance.**

**src change (c).** The cost.py message names the variant (`unknown cost variant 'bogus'`), and
test_cost_terms_rejects_unknown_variant's `match="variant"` binds to it. The KeyError mutant is killed.

**Test-side checks.**

* `ls_J` imports nothing from src. It uses `U_of` (GTSAM `predict`, test-side `Gmat`) and `gtsam.Gal3.Logmap`, so it
  is an independent reference.
* The replaced C31 check is `qm` (the lam^2 coefficient read from **src's Q**) against a closed form built from
  **reference** taus and Dt. This is not a tautology, because src weights enter qm and reference weights enter the
  closed form. Run alone, test_c31 kills p4-in-tau-v-p-swap and p4-co-rp-Dt2-vi. It does not see GNSS weights or the
  instance Dt (by design: v_1 does not enter them).
* My own value: qm = 3.264861e6 = closed form, exactly.

## 2. Independent re-checks (the reference imports no src)

| check | result |
|---|---|
| instance vs ref4 PIMs (both bbar, both edges) | dR 1.0e-14, dv 0, dp 4.4e-16; taus, W_R, G2, W_vp identical; Dt = 0.5000000000000003; truth (Van Loan) 3.3e-15; truth_state.b == b_true exactly |
| J: src (h = 0.1) vs ref4 (logm-chart LS fit) | 7.3e-15 to 1.1e-14 (h = 1e-3 would give 2.3e-13 to 6.2e-13) |
| z_of_state vs ref4 concatenation, 30 states x both bbar | 3.4e-15 abs (own J); aniso 1.1e-14 relative |
| **families: src A_k vs ref4 direct matrix equations, ARBITRARY z (off variety)**, base and aniso, both bbar | same order, same sign, every family: <= 2.5e-14 relative (P0 0; P1/P2/RA1 <= 3.3e-16; the J-dependent families 1e-15 to 2.5e-14, from the 1e-14 J difference) |
| **Q and factored cost vs ref4 cost on ARBITRARY z**, iso / partial / aniso, both bbar, 100 z each | dense <= 1.2e-15 relative, factored <= 8.9e-16 relative |
| cost vs own 5x5 cost, lifted states | 4.0e-15 to 5.9e-15 relative |
| decoder (1e4 samples, Sec. 1) | project = False identical. project = True, well-posed blocks: <= 2.4e-15 from own projection; blocks with repeated singular values: Frobenius distance agrees to 1.9e-15; det/orthogonality error <= 2.7e-15 |
| truth costs | factored 6.2e-22, GTSAM 6.2e-22, **dense z^T Q z 3.9e-9** (cancellation, max\|Q\| = 1.11e10) |

### Property tests (1e4 samples each, own encoder, src constraints and costs)

| instance | family max on lifted states | cost dense / factored vs own 5x5 |
|---|---|---|
| true, base | P0 0, P1/P2/RA1 4.4e-16, P3 7.2e-15, RB1 2.8e-16, RB2 1.3e-15, RC1 8.7e-15, RG 2.6e-15, RL 6.8e-15, RH 7.0e-15, RS 3.1e-15 | 4.7e-15 / 4.6e-15 |
| true, aniso | as base (<= 8.9e-15), P4 1.2e-13, RR 3.3e-12 (bound 1e-10) | 4.4e-15 / 4.6e-15 |
| zero, base | <= 1.2e-14 (RC1) | 4.5e-15 / 4.9e-15 |
| zero, aniso | <= 1.2e-14, P4 1.2e-13, RR 3.9e-12 | 4.3e-15 / 4.1e-15 |

### Spurious-feasible-point hunt (TRF, `xtol = ftol = gtol = 1e-15`, analytic Jacobian, src A_k; lifted-ness judged by ref4's encoder)

| family set | converged | max \|z - s z(state(z))\| | det R_k |
|---|---|---|---|
| primary P0-P3, bbar = 0 | 200/200 | 4.3e-13 | all +1 |
| all 12 families (primary + redundant), bbar = 0 | 40/40 | 6.7e-15 | all +1 |
| aniso primary P0-P4 (n = 136) | 60/60 | 2.3e-13 | all +1 |
| primary, bbar = b_true | 60/60 | 1.5e-13 | all +1 |
| drop P2 | 60/60 | 6.3e-14 | 97 of 180 blocks det -1 |
| drop P3 | 60/60 | 2.2e2 | +1 (liftings off their definitions) |
| drop P1 | 60/60 | 1.5e1 | min \|det\| 5.6e-73 |

The hunt also covered targeted constructions: reflections (C20), R_0 = 0 (C21), liftings with a wrong rotation or
the negative (C22), and the global sign flip (C23).

**No spurious point exists for the full primary set in any variant tried.**

## 3. Claims

Each verdict rests on my own computation, with src as the object under test. The suite's claim tests also pass on
the clean tree: 66/66, twice (8 min 44 s with default threads; 44 s with OMP_NUM_THREADS = 2).

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | n 124 / 136; s at 123; rho_v(1) = [130,133), rho_p(1) = [133,136); z[123] = 1; index map vs own concatenation 3.4e-15 |
| C02 | CONFIRMED | 112, 12; the V' columns of imu0.vel are exactly -1/2 I3 |
| C03 | CONFIRMED | all 7 triples satisfy n = 39K + 6M + 16 and n - direct = 39 |
| C04 | CONFIRMED | own FD (h = 1e-6, GTSAM Logmap) on 4 fixture cases + 6 random time-varying streams: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| 3.95e-10; \|J_R\| 0.37-0.50. ad(x) vs matrix commutator 0; right-Jacobian series vs FD 1.6e-10 (200 x); block differences exactly 0. Shipped J: zeros exact, \|J_nu,a - J_R\| <= 7.6e-15 |
| C05 | CONFIRMED | own algebra, 4000 draws: max\|r\| 2.2e-15; \|t_j\| 0; A34 = B34 = Dt exactly; bottom rows 0 exactly; arbitrary X_j min max\|r\| 1.04 |
| C06 | CONFIRMED | 3.1e-15 |
| C07 | CONFIRMED | src forms vs own 5x5, 500 states: 1.1e-14 / 1.0e-14 |
| C08 | CONFIRMED | coefficient blocks exact (deviation 0); truth cost 6.0e-22; Y_0 perturbation min 4.2e-5 >= 1e-6 |
| C09-C12 | CONFIRMED | truth <= 8.8e-15; 200 random: base <= 4.5e-15; P4 1.1e-13, RR 3.0e-12 (<= 1e-10); 1e4-sample run in Sec. 2 |
| C13 | CONFIRMED | min family max 1.2e-5 (RB1, base) / 1.4e-5 (aniso) > 1e-7 |
| C14 | CONFIRMED | sizes as stated; every family full rank; primary 106 (aniso 118); joint 472 -> 453 (aniso 490 -> 471) |
| C15 | CONFIRMED | own row-handedness matrices: each matches a src P2 matrix exactly (mismatch 0) |
| C16 | CONFIRMED | rank 11 for each KF; trace identity exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric (base and aniso); homogeneity 6.6e-17; the only b != 0 is s^2, b = 1 |
| C18 | CONFIRMED | Sec. 2: 200/200, consistency 4.3e-13, det +1; drop-P2 / P3 / P1 behave as claimed |
| C19 | CONFIRMED | 300 starts: 228 SO(3), 72 zero, 0 other |
| C20 | CONFIRMED | violated: P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied <= 6.5e-16 |
| C21 | CONFIRMED | P1 = RA1 = 1.0; all others <= 4.4e-16 |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: P3 9.6e-2, RL 8.4e-2, RH 9.0e-2, RS 1.3e-2; satisfied <= 3.3e-16 |
| C23 | CONFIRMED | residuals of -z equal those of z exactly; cost difference 0.0; decode(-z) == decode(z) exactly (project = True: difference 0) |
| C24 | CONFIRMED | iso 4.6e-15 / 4.0e-15; aniso 9.5e-15 / 8.9e-15 (500 / 300 states, own 5x5 cost) |
| C25 | CONFIRMED | iso / aniso / partial at both bbar: exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0 exactly; bias block equal to own rw sum exactly |
| C26 | CONFIRMED | truth factored 6.2e-22 (<= 1e-15); own GTSAM graph, 50 states per eps: 1.10e-4, 2.73e-6, 2.92e-7 (bounds 5e-4, 5e-5, 5e-6) |
| C27 | CONFIRMED | deviations rot / vel / pos 2.011e-2 / -1.223e-2 / -2.098e-3 at lam = 1, all <= 3e-2; halving factors rot 0.250 / 0.249 / 0.249, vel 0.501 / 0.500 / 0.500, pos 0.504 / 0.502 / 0.501 |
| C28 | CONFIRMED | 2.088e-7, 2.605e-8, 3.253e-9, 4.064e-10; factors 8.02 / 8.01 / 8.00; \|x\| 0.060 |
| C29 | CONFIRMED | half 2.1e-7 (<= 5e-6); full 0.75 (>= 0.1) |
| C30 | CONFIRMED | own symmetric root, own forms, own local parametrisation: rank 45, cond 3.946e7 / 4.527e7; src whitened residual: same rank and cond |
| C31 | CONFIRMED | src lam^2 coefficient equals the closed form exactly; 3I: 4.020000 (ptp 3.5e-14); diag(1,2,5): range [1.351, 6.694] |
| C32 | CONFIRMED | own exactly-orthogonal sampler, rng 2: 2.8e-15, 2.8e-15 (<= 1e-12); diag(1,2,5) 8.3e-2; random SPD 2.6e-1; rank 2015 |
| C33 | CONFIRMED | partial vs own GTSAM graph: 9.0e-5, 3.4e-6, 3.4e-7 (bounds 0.05 eps); D min eig 3.460e7; G2 corr -0.867 |
| C34 | CONFIRMED | body 1.2e-3, 1.5e-4, 2.5e-5 (<= eps); world 5.6e-2, 5.7e-2, 6.6e-2 (>= 1e-2) |
| C35 | CONFIRMED | rng 4: rank 2015, null 263, gap 2.1e-7 -> 4.5e-16; spans 60 / 92 / 106 / 136 / 226 / 236; projection residual 2.6e-11 (<= 1e-10). This uses rotations orthogonal to ~1e-16; the pass-3 fragility caveat stands |
| C39 | CONFIRMED | Dt 0.5000000000000003 for both edges; taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6 (own = src); rw diagonal 5.5402e9 / 2.2222e5 |
| **C43** | **REFUTED** | below |

**M3-C43 (brief only): REFUTED.** The exact closed form is Cay(phi) = I + ([phi]x + [phi]x^2 / 2) / (1 + |phi|^2/4).
The **[phi]x^2 term also carries the denominator.** The stated form I + [phi]x / (1 + |phi|^2/4) + [phi]x^2 / 2
drops it, so its error is ([phi]x^2 / 2) |phi|^2/4 / (1 + |phi|^2/4), of order |phi|^4 / 8. Measured:

* 1e4 samples, |phi| <= 0.1 (uniform in the ball, plus 200 on the sphere |phi| = 0.1): max entrywise error vs
  the Cayley solve **1.247e-5**. The lifted residual (I - [phi]x/2) Re - (I + [phi]x/2) is **1.247e-5**, and
  \|Re^T Re - I\| = 2.5e-5, so the stated matrix is not a rotation. Error / \|phi\|^4 lies in [0.084, 0.125].
* mpmath (50 digits) at |phi| = 0.1: error 1.24688e-5, which equals the predicted |phi|^4 / 8 / (1 + |phi|^2/4).
* The 1e-12 bound fails for every |phi| above about 1.7e-3. At |phi| = 2e-3 the error is 2.0e-12; at 1e-3 it is
  1.25e-13.
* With the denominator on both terms, the same samples give 2.2e-16 against the solve and 2.2e-16 on the lifted
  constraint.

The "without changing the feasible set" part also fails as stated: the stated expression violates the degree-2
constraint at every phi != 0.

## 4. Mutation sweep

### 4.1 New mutants (95; ids `p4-*`, none in passes 1-3 or in tests/test_mutation_m3.py)

By file: lifting 18, constraints 20, cost 16, forms 8, gtsam_boundary 12, instance 9, fixture 11, `__init__` 2. Each
is one exactly-once substitution on a tempfile copy, judged by the full tests/test_qcqp.py with `-x`.

**Killed (84), grouped by first failing test.**

* test_decoder_rejects_non_finite_z: lf-finite-only-s, lf-isnan-only.
* test_decoder_degenerate_inputs: lf-isinf-only.
* test_decoder_projection_is_nearest_rotation: lf-svd-of-AT, lf-project-D-middle.
* test_state_copy_is_independent: lf-copy-shallow-R.
* test_c01_index_map: lf-s-index-minus1, lf-s-entry-s, lf-W-from-ba, lf-Rj-is-M, lf-decode-R-rowmajor, lf-bias0-off,
  co-cp-uses-Jnu, co-rp-Dt2-vi, co-rv-dv-Rj, co-rp-dp-uses-dv, co-rp-g-Dt, fm-cmat-right-T. The cost mutants reach
  this test through the aniso rho entries, which are checked against resid5.
* test_c05_shipped: co-Gam-from-ba, co-rR-M-Gam-sign.
* test_c08: co-prior-R0-rowmajor, co-vec-rows-rowmajor.
* test_c09: cn-cyclic-wrong, cn-W-negated, cn-P3-P-uses-cv, cn-cv-from-Jrho, fm-eps-symmetric, fm-hat-sign,
  fm-cvec-transposed, fm-linT-noop, fm-matmul-out-T, fm-cross-sign.
* test_c10: cn-RB2-half-ww, cn-RC1-P-uses-cv, cn-RB1-uses-cv, fm-dot-allpairs.
* test_c11: cn-RG-YV-W-not-T, cn-RL-V-uses-M, cn-RH-ea-shift, cn-RS-hat-sign.
* test_c12: cn-P4-uses-Rj, cn-no-RR.
* test_c13 (signed g vs own evaluation): cn-residual-sign, cn-family-max-mean.
* test_c14: cn-RH-drop-P.
* test_c17: cn-P0-b-neg.
* test_c18: cn-flat-ignores-names, gb-h-1.0.
* test_P1_RA1: cn-upper-strict, cn-RA1-offdiag-ss.
* test_c24: co-gnss-weight-sigma.
* test_c25: co-assemble-nosym (F^T W F is not bitwise symmetric).
* test_c33: co-partial-vp-order.
* test_whitened_residual: co-whiten-no-F (a scale of 1 + 1e-7).
* test_cost_terms_rejects_unknown_variant: co-unknown-KeyError.
* test_c04_instance_J_...: gb-h-1e-2, in-J-step-1e-3.
* test_instance_matches_independent_gtsam_reference:
  * gtsam_boundary: gb-h-1e-5, gb-left-chart, gb-lm-plus, gb-nav-pv-swap, gb-integrate-swap,
    gb-int-cov-not-squared.
  * instance: in-S2-no-third, in-Svp-vv-pp, in-tau-v-p-swap, in-gnss-velocity, in-prior-kf1, in-Dt-literal,
    in-rw-bg-uses-ba, in-dR-transposed.
  * fixture: fx-so3exp-th, fx-m1-K-sign, fx-m2-T2, fx-true-v-gravity, fx-gyro-no-bias, fx-steps-199, fx-hat-T,
    fx-omega1-literal, fx-sigma-ba-rw, fx-so3exp-branch-1, fx-g-literal.
* test_package_exports: init-drop-truth_state, init-state_of_z-alias.

Every killer checks against an independent reference: GTSAM PIMs with retyped literals, M2_REF, the test-side 5x5
residual / cost / GTSAM graph, `ls_J`, the claims' literal sizes, or a test-side SVD / numpy evaluation. Many of the
new decoder, export and copy kills come from the 9bbf3a3 tests.

**Survive (11):** see 4.3.

### 4.2 Replays (27 real file edits)

* The 10 F05 survivors and the 2 reverts of 9bbf3a3: all 12 killed (Sec. 1).
* 15 earlier equivalents: p3-co-partial-G2-T, p3-co-factored-abs, p3-co-aniso-assert-removed,
  p3-fm-outer-no-broadcast, p3-in-prior-not-copied, p3-in-zero-key-neg-zero, in-bbar-not-copied, co-aniso-Wvp-T,
  co-assemble-WT, co-whiten-clip-floor, fx-so3exp-branch, in-S2-offdiag-swap, cn-mk-nosym, cn-vech-tril,
  fm-matvec-nm-swap. All 15 survive, and all 15 are re-confirmed equivalent with the 518-group dump (4.3 #12-#26).
  This replay drops pass 3's #23 (gb-fd-step h = 0.1), which is now the shipped code; its reverse is killed.
* `tests/test_mutation_m3.py`: 70 passed (68 mutants killed by their named killers, plus the well-formedness test
  and the clean control).

### 4.3 COMPLETE survivor list (26) with rulings

The threshold is 1/10 of the tolerance of the governing claim, never below the 8-ulp floor. As in pass 3, a change
in a claim-free contract (exception vs return, aliasing, a default argument) changes an output by definition. It is
therefore non-equivalent unless a proof shows that no src output changes.

| # | mutant | measured effect (518-group dump diff vs clean) | ruling | class |
|---|---|---|---|---|
| 1 | **p4-lf-truth-b-alias** (`[inst.b_true for _ in inst.truth]`) | values identical. The three `truth_state(I).b[k]` arrays are one object, and that object is the cached instance's `b_true` (shares_memory True) | **NON-EQUIVALENT** | low: `st = truth_state(I); st.b[0][0] += x` (the in-place edit test_state_copy_is_independent itself performs) also moves b_1, b_2 and corrupts the lru-cached instance's b_true for every later caller. No test edits a truth_state in place |
| 2 | **p4-lf-truth-v-alias** (`[t[1] for t in inst.truth]`) | values identical; `truth_state(I).v[k] is I.truth[k][1]` True | **NON-EQUIVALENT** | low: same mechanism; an in-place edit of the returned v corrupts `Instance.truth` (the instance is frozen, but its arrays are not) |
| 3 | **p4-co-default-partial** (`variant: str = "iso"` -> `"partial"`) | `CO.terms(I)` returns the 11 partial-variant terms instead of the 13 iso terms (both bbar) | **NON-EQUIVALENT** | low: every test passes `variant` explicitly. M4 code that calls `terms(inst)` for the base cost (the docstring's "iso (base)") would silently get the C33 weighting |
| 4 | **p4-gb-no-gravity-assert** (the z-up assert -> `pass`) | `make_params([0,0,+9.81])` and `make_params([0.1,0,-9.81])` return params instead of raising AssertionError (MakeSharedU(-9.81): gravity +z; the x component silently dropped) | **NON-EQUIVALENT** | low: outside the instance domain, but it is the only guard on a gravity convention (CLAUDE.md footgun territory). It is also an `assert`, which `python -O` strips |
| 5 | **p4-lf-floor-zero** (`max(zmax, 1e-300)` -> `max(zmax, 0.0)`) | 10 groups: 1e-320 x z_truth, the ties at max\|z\| = 1e-301 / 1e-310 and their prev-double neighbours now decode instead of raising | **NON-EQUIVALENT** | very low: the documented floor (`max(max\|z\|, 1e-300)`) is untested below 1e-300. The scale sweep stops at 1e-300 x z_truth, whose max\|z\| = 1.9e-300 is above the floor |
| 6 | **p4-lf-guard-ge** (adds `or abs(s) > 1e300`) | 0 groups (the largest probe is s = 1e300 exactly); by construction z with \|s\| in (1e300, 1.8e308] now raises | **NON-EQUIVALENT** | very low and artificial (an added clause, not a slip). It shows that the guard is tested only up to 1e6 x z. One `state_of_z(1e305 * z)` assertion kills it |
| 7 | **p4-lf-project-det-nosign** (`np.sign(det(U Vt))` -> `det(U Vt)`) | 31 groups, all project = True and `project_so3`: entries change by <= 8.9e-16 on probes, **<= 2.2e-15 (10 ulp) over 1e4 random blocks**; orthogonality error 6.1e-15 vs clean 2.9e-15; \|det - 1\| 5.8e-15 vs 2.8e-15; distance to the input up to 2.4e-15 larger | **NON-EQUIVALENT** | very low: the D entry becomes +/-(1 +/- few eps) instead of exactly +/-1. Rule 2 fails: 10 ulp, worse than clean. No claim tolerance covers project_so3 except C01's 1e-14 (threshold 1e-15), which the change exceeds. Killed by a test-side comparison with U diag(1, 1, sign) V^T, or by an orthogonality bound of 4e-15 over 1e4 random blocks |
| 8 | p4-lf-zmax-excl-s (max over z without s) | 0 groups | EQUIVALENT | proof: if \|s\| < max of the others, zmax is unchanged. If \|s\| >= every other entry and \|s\| >= 1e-300, both thresholds are <= 1e-12 \|s\| < \|s\|, so both decode. If \|s\| < 1e-300, every entry is < 1e-300 and both thresholds are 1e-312. Raise and return agree for every z; only the max printed in the error message can differ |
| 9 | p4-gb-h-0.5 | J changes <= 1.1e-14; its error vs the exact map **falls** from 5.7e-15..1.1e-14 to 0.9e-15..1.8e-15. z <= 1.8e-15; family residuals <= 1.8e-14; costs <= 2.5e-14 abs (<= 2.4e-11 of the cost) | EQUIVALENT | rule 1, and the mutant is more accurate: the change is <= 1/10 of every governing tolerance (C04 1e-8, C09 1e-12, C24 1e-12 relative to max(1, J), C26 1e-15 abs: 8e-23), and it passes the 5e-14 implementation pin |
| 10 | p4-gb-h-0.05 | J changes <= 1.8e-14; error vs exact 9.1e-15..1.6e-14 (clean 5.7e-15..1.1e-14, i.e. slightly worse). z <= 5.5e-15; family residuals <= 9.8e-15; costs <= 2.1e-14 abs | EQUIVALENT | rule 1 against the J-sensitive claims (C04, C09-C12, C24, C26), as in pass 3 #10/#23. C01's 1e-14 compares z with its defining expression using the same J, so a J change does not enter it |
| 11 | p4-gb-sharedU-abs (`MakeSharedU(norm(g))`) | 0 groups | EQUIVALENT | proof: on every input that passes the assert, g = (0, 0, -c) with c > 0, and norm(g) = sqrt(fl(c^2)) = c exactly in binary64 with round-to-nearest. Measured on 1e5 random c in [1e-3, 1e3] plus 9.81 / 9.80665 / 9.8: 0 mismatches |
| 12 | p3-co-partial-G2-T (replay) | partial terms rel 3.9e-17; Q rel 3.4e-19; costs / whitened <= 4.2e-16 relative | EQUIVALENT | below the 8-ulp floor (the G2 asymmetry is 3.9e-17 relative) |
| 13 | p3-co-factored-abs (replay) | 0 groups (incl. singular W_vp) | EQUIVALENT | proof as in pass 3 (PSD W: abs changes only rounding-negative values) |
| 14 | p3-co-aniso-assert-removed (replay) | the same AssertionError type from `layout.rho_v`; only the message text differs | EQUIVALENT | proof as in pass 3 |
| 15 | p3-fm-outer-no-broadcast (replay) | 0 groups | EQUIVALENT | proof as in pass 3 |
| 16 | p3-in-prior-not-copied (replay) | `prior[q] is truth[0][q]` | EQUIVALENT | proof as in pass 3: src never writes in place into an Instance field, and `prior` is not handed out by any src function (unlike #1/#2, where truth_state hands the arrays to the caller inside a mutable State) |
| 17 | p3-in-zero-key-neg-zero (replay) | 0 groups | EQUIVALENT | -0.0 behaves as +0.0 everywhere here |
| 18 | in-bbar-not-copied (replay) | `build_instance(a).bbar is a` | EQUIVALENT | proof as in pass 3. It aliases the caller's own argument; no src function writes into it or returns it inside a mutable container |
| 19 | co-aniso-Wvp-T (replay) | terms 3.9e-17 rel; whitened residual <= 5.3e-11 abs on entries ~2e4 (2.4e-15 rel), 5.9e-15 rel on a near-zero vector (abs 3e-16) | EQUIVALENT | at the floor; also << 1/10 of C30's relative cut and C24's 1e-12 |
| 20 | co-assemble-WT (replay) | Q rel 4.2e-20 / 3.4e-19 | EQUIVALENT | below the floor |
| 21 | co-whiten-clip-floor (replay) | only the singular caller W_vp: whitened residual 3.6e-10 rel per entry; every claim-domain weight unchanged | EQUIVALENT | proof as in pass 3 (claim weights have min eig >= 400 >> 1e-12) |
| 22 | fx-so3exp-branch (replay) | only the \|w\| = 2.2e-7 probe: 8.0e-18 | EQUIVALENT | the branch is never reached on the instance; below the floor at the probe |
| 23 | in-S2-offdiag-swap (replay) | 0 groups | EQUIVALENT | the preintMeasCov traces are bitwise symmetric |
| 24 | cn-mk-nosym (replay) | 0 groups | EQUIVALENT | every Q passed to `_mk` is already exactly symmetric |
| 25 | cn-vech-tril (replay) | only the non-symmetric probe | EQUIVALENT | vech is documented for symmetric input; every A_k is exactly symmetric |
| 26 | fm-matvec-nm-swap (replay) | 0 groups | EQUIVALENT | sym(X^T) = sym(X) bitwise |

Totals: 26 survivors. **Non-equivalent 7 (#1-#7), all low. Equivalent 19 (#8-#26).**

## 5. Tests that cannot fail or compare code with itself (at 9bbf3a3)

1. Fixed since pass 3: the cannot-fail `ptp(qr)` in test_c31 is replaced by a src-vs-reference check, which kills
   two weight mutants run alone (Sec. 1).
2. test_decoder_s_guard_threshold_tie_and_scale asserts `tie == 1e-12 * scale and S_FLOOR_DOC * max|z| == tie`.
   This is test-side arithmetic, a precondition that guards the exactness of the tie. It cannot fail because of
   src, and it is harmless.
3. test_c04_instance_J_...'s `fit <= 1e-14` checks the reference, not src. The src check is the next line (5e-14),
   which is labelled an IMPLEMENTATION PIN. It accepts any FD step from about 0.02 to 1 (Sec. 4.3 #9-#10, and the
   h = 1e-2 kill).
4. Unchanged from pass 3: test_c02 / test_c03 arithmetic; the test-side-only parts of C15, C26, C33, C39 and C04;
   MATH CHECK labels on C05 / C06 / C28; the scale invariance of C30's rank.
5. **Untested surface (all of it shows up as survivors):** in-place-edit independence of `truth_state`'s output;
   `cost.terms`' default variant; `make_params`' gravity guard; the decoder guard below max\|z\| = 1e-300 and above
   s = 1e6 x z; the exact +/-1 of the projection's det correction.

## 6. Findings and risks for M4

* **M3-F06 (new, gate-blocking, low): 7 non-equivalent survivors (Sec. 4.3 #1-#7).** Tests that would kill each
  one:
  * (a) `st = truth_state(I); st.b[0][0] += 1; st.v[0][0] += 1`, then assert `st.b[1]`, `I.b_true`, `I.truth[0][1]`
    and a fresh `truth_state(I)` are unchanged (#1, #2);
  * (b) `[t.name for t in CO.terms(I)]` equals the iso names, or `assemble_Q(terms(I))` equals the iso Q (#3);
  * (c) `pytest.raises(AssertionError)` (better: change the assert to a `ValueError`, which survives `-O`) for
    `make_params` with gravity (0, 0, +9.81) and (0.1, 0, -9.81) (#4);
  * (d) the documented floor: `z = 1e-301 e_0 + 1e-313 e_s` raises, and `1e-312` at max\|z\| = 1e-310 raises (#5);
  * (e) `state_of_z(1e305 * z_truth)` decodes to the state of z (#6);
  * (f) `project_so3(A)` equals `U diag(1, 1, sign(det(U V^T))) V^T` from a test-side SVD to <= 2 ulp, or its
    orthogonality error is <= 4e-15 over 1e4 random A (#7).
* **The rounding path is sound.** The decoder matches an independent decoder on 1e4 inputs (0 raise disagreements;
  project = False identical; project = True <= 2.4e-15). Non-finite input is now rejected. M4 should still report
  the constraint residual of the rounded point itself, since the decoder ignores the liftings.
* **Dense Q cancellation persists.** At truth, z^T Q z = 3.9e-9 vs factored 6.2e-22 (max\|Q\| = 1.11e10). Use the
  factored cost for primal values and duality gaps, or rescale Q, before any Clarabel solve. The constraint
  matrices have entries O(1) and the cost O(1e10). This scale disparity is the first thing to fix for SDP accuracy;
  Barfoot's sigma_1/sigma_2 >= 1e5 test will be meaningless if the solver's relative accuracy is swamped by the
  1e10 cost scale.
* **J floor.** J is accurate to <= 1.1e-14 now. The family residuals at independently lifted states are ~1e-14
  (P4 1.2e-13, RR 3.9e-12 scale with \|r\|^2).
* **C35's 1e-10 margin** still holds only for rotations orthogonal to ~1e-15 (2.6e-11 here). Any AutoTight rerun
  in M4 needs the same sampler, or a tolerance a decade above the gap edge.
* **Redundant families are linearly dependent** (472 -> rank 453; aniso 490 -> 471). Remove the dependencies (or
  let the solver handle rank-deficient equality blocks) before handing them to Clarabel.
* **Surrogate gap at bbar = 0** (0.57% vs GTSAM at truth, pass 3) is outside C26's domain and still applies.
