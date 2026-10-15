# Stage B verification, pass 5: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 5 (agent verify-M3-005). Base commit 519864f, the implementer's response to
the pass-4 finding M3-F06. Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py, docs/claims/M3.yaml
(M3-C01..C35, C39), plus one claim given only in the brief, M3-C44. I did not read docs/03_qcqp_derivation.md,
experiments/m3_scratch/ or any seeds file. I did not edit src/, tests/ or docs/claims/.

My scripts lived in a scratch directory inside the worktree (`_v5scratch/`, untracked, deleted before the commit).
Mutant trees went to Python `tempfile` directories. The scripts:

* `ref5.py`: a new independent reference. It imports neither src nor tests. It contains:
  * its own retyped literals and its own GTSAM PIMs;
  * Ubar from `deltaXij()`;
  * J as a 200-sample least-squares fit of the **scipy `logm` chart** (fit residual 1.1e-15 to 1.5e-15);
  * the truth from a **Van Loan block exponential** `expm([[W, f, 0], [0, 0, 1], [0, 0, 0]] T)`, which is not
    src's Rodrigues closed form, with an SVD-orthogonalised Rodrigues sampler;
  * z by plain concatenation.

  **Every constraint family is re-implemented as matrix equations on the blocks of an arbitrary z** (this is the
  re-implemented core routine). The reference also has the C07 residual forms on arbitrary z, its own cost on
  arbitrary z (iso / aniso / partial), its own 5x5 cleared-residual cost, its own decoder and projection, and (in
  `claimsB.py`) its own GTSAM factor graph using `Diagonal.Precisions`, NavState-ordered information and
  `BetweenFactorConstantBias` in (accel, gyro) order.
* `cmp5.py`, `prop5.py`, `hunt5*.py`, `claimsA/B/C.py`, `c44.py`, `gravity5.py`, `det5.py`: the src-vs-reference
  comparisons, the 1e4-sample property tests, the spurious-point hunt, the per-claim computations, C44, and the
  F06 checks.
* `mutants5.py`, `harness5.py`: 176 new mutants plus the 7 F06 replays. Each mutant is a real exactly-once textual
  edit on a tempfile copy of src + tests/test_qcqp.py + conftest + fixtures. The judge is the **full**
  tests/test_qcqp.py, slow tests included, run with `-x` (single-threaded BLAS, 5 workers). A clean control copy
  passes (72/72).
* `dump5.py`, `measure5.py`: record 487 observable output groups of src/qcqp and diff a mutant tree against the
  clean tree. The groups cover:
  * every instance field and the cov blocks;
  * z at 7 states x 2 layouts x 2 s values x 2 bbar;
  * every constraint matrix and its residuals on random z;
  * `family_max` / `flat` / `vech`;
  * terms / Q / factored cost / whitened residual at 5 weightings, including a singular W_vp;
  * the default variant, unknown variants and the aniso-on-base error;
  * about 100 decoder probes: scales from 1e-320 to 1e307, ties at 6 scales including sub-1e-300, NaN/inf at 5
    positions, degenerate blocks, list input, and aniso z with dominant rho;
  * 60 random `project_so3` inputs plus singular ones;
  * an aliasing vector (truth_state vs the Instance, Instance vs the fixture constants, State.copy, cache identity);
  * every forms helper, the GTSAM boundary (including 8 gravity inputs), the fixture literals and functions, the
    package exports and the layout.

  Two clean dumps are identical (0 of 487 groups differ).

## Verdict summary

* **F06 (pass 4): FIXED.** All 7 pass-4 survivors, replayed as real file edits and judged by the full suite, are
  killed (Sec. 1). The src change is correct. Under `python -O` it raises ValueError, and on valid input it is
  bitwise unchanged. Remaining caveats are in Sec. 1: the test bindings are nominal, the det test's reference is
  not independent, and its bounds have tight headroom.
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C44 REFUTED. 0 UNTESTABLE.**
* **Fresh sweep: 176 new mutants over all 8 files of src/qcqp + 7 replays = 183 real-file-edit runs.** 175 are
  killed (all 7 replays and 168 of the new mutants), and 8 new mutants survive. Of the 8 survivors, **5 are
  equivalent and 3 are NON-EQUIVALENT (all low)**. The replay of `tests/test_mutation_m3.py` passes **77/77** (75
  mutants killed by their named killers, plus the well-formedness test and the clean control).
* **GATE (zero surviving non-equivalent mutants): FAILED**, on 3 low survivors (new finding M3-F07, Sec. 6):
  * the decoder's max|z| ignores the rho entries (aniso layout);
  * make_params' shape check is untested;
  * make_params accepts g = -inf.

  Every mutant that changed z, Q, a family, the instance data, the GTSAM boundary arithmetic, the forms helpers,
  the fixture, the exports, or the decoder's s / threshold / floor / projection logic was killed.

## 1. Finding M3-F06 ruling: FIXED

### Replays of the 7 pass-4 survivors (real exactly-once edits on 519864f, full suite, `-x`)

| pass-4 survivor | replay edit | result: killer and failing assertion |
|---|---|---|
| #1 p4-lf-truth-b-alias | `[inst.b_true.copy() for _ in inst.truth]` -> `[inst.b_true for _ in inst.truth]` | killed by test_truth_state_blocks_are_independent_arrays (`shares_memory(b[0], b[1])`) |
| #2 p4-lf-truth-v-alias | `[t[1].copy() ...]` -> `[t[1] ...]` | killed by the same test (`shares_memory(v[k], I.truth[k][1])`) |
| #3 p4-co-default-partial | `variant: str = "iso"` -> `"partial"` | killed by test_cost_terms_default_variant_is_iso (term names) |
| #4 p4-gb-no-gravity-guard | the `raise ValueError(...)` -> `pass` | killed by test_make_params_rejects_non_z_up_gravity (DID NOT RAISE) |
| #5 p4-lf-floor-zero | `max(zmax, 1e-300)` -> `max(zmax, 0.0)` | killed by test_decoder_s_guard_floor_below_1e_minus_300 |
| #6 p4-lf-guard-ge | adds `or abs(s) > 1e300` | killed by test_decoder_large_scales_decode (ValueError at c = 1e305) |
| #7 p4-lf-project-det-nosign | `np.sign(det(U Vt))` -> `det(U Vt)` | killed by test_project_so3_det_correction_is_exact (`worst_ref` = 2.4e-15 > 4.5e-16) |

### The src change: `make_params` raises ValueError instead of asserting

| input g_w | normal | `python -O` |
|---|---|---|
| (0, 0, +9.81), (0.1, 0, -9.81), (0, -0.1, -9.81), (0, 0, 0), (0, 0, NaN), (NaN, 0, -9.81) | ValueError | ValueError |
| shape (4,), shape (1, 3), shape (3, 1), a scalar | ValueError | ValueError |
| (0, 0, -9.81), (-0.0, 0, -9.81), (0, 0, -10) [ints], (0, 0, -1e-300) | accepted | accepted |
| **(0, 0, -inf)** | **accepted** (n_gravity = (0, 0, -inf)) | **accepted** |

`sys.flags.optimize = 1` and `__debug__ = False` were confirmed under `-O`, so the guard survives `-O`.

**Valid input is unchanged.** For `make_params([0, 0, -9.81], ...)`, gravity and all three covariances are
array_equal to a direct `MakeSharedU(9.81)` with the same covariance settings. The full instance (dR, dv, dp, J,
W_R, G2, W_vp, Dt, taus) at both bbar has a **bitwise-identical SHA-256 digest at 9bbf3a3 and at 519864f**.

Two gaps remain in the guard:

* -inf is not rejected. The commit message claims only NaN and shape, but `g > 0` in the error text does not hold
  for an infinite g.
* The shape check has no test. See M3-F07 (b) and (c).

### truth_state independence test: real

It sees aliasing between keyframes: `shares_memory` is checked pairwise over k for R, v, p and b, and the
"each edit applied exactly once" assertion catches shared b blocks. It sees aliasing with Instance.b_true
(`shares_memory(st.b[k], I.b_true)`). It sees aliasing with Instance.truth through `shares_memory` and, after the
+1 in-place edits, through the 1e-9 comparison of `I.truth[k][q]` against the independent truth_ref.

Besides the two replays, I ran 5 targeted edits against this test alone, and the sweep ran 3 more. All are killed:

| edit | first assertion that fires |
|---|---|
| b_0 aliases b_true only | `shares_memory(st.b[0], I.b_true)` |
| b_2 aliases b_true only | `shares_memory(st.b[2], I.b_true)` |
| v_2 aliases I.truth only | `('v', 2)` |
| R_1 is a slice view of I.truth | `('R', 1)` |
| b blocks are views `b_true[:]` | `('b', 0, 1)` |
| sweep: p alias, R alias, list-multiplied b | killed |

It uses a fresh `build_instance`, so a regression cannot poison the cached instance. **Sound.**

### project_so3 det-correction test: a pin, not an independent reference

Its reference is `U diag(1, 1, d3) V^T` from the **same** `np.linalg.svd` (same LAPACK dgesdd). In the clean tree
it is therefore bitwise equal to src (worst_ref = 0). It checks "src uses an exact +/-1" and nothing about
correctness. That is the right tool for the target mutant, whose defect is defined relative to this algorithm.
Correctness (nearest proper rotation) comes from test_decoder_projection_is_nearest_rotation and from my own
decoder (1e4 samples, Sec. 2, difference 0).

The cross-implementation check: scipy `gesvd` (a different LAPACK driver) gives the same orthogonality / det
statistics.

**Fragility of the 4e-15 orthogonality / det bounds.** These are max over 1e4 random A; measured over 21 seeds
(115, 0-19):

| implementation | orth max over seeds | det max over seeds | seeds > 4e-15 |
|---|---|---|---|
| src (numpy gesdd) | 2.44e-15 to **3.33e-15** | 2.44e-15 to 3.11e-15 | 0 |
| scipy gesdd | same | same | 0 |
| scipy gesvd | 2.44e-15 to 3.11e-15 | 2.33e-15 to 3.11e-15 | 0 |
| target mutant | 5.33e-15 to 7.33e-15 | 4.88e-15 to 6.44e-15 | 21 / 21 |

The headroom over the worst seed is only 1.2x. A different BLAS/LAPACK build or FMA behaviour could plausibly
cross 4e-15. These two bounds are redundant for the kill: `worst_ref` alone kills the mutant with 5x margin
(2.4e-15 vs 4.5e-16). **Recommendation:** loosen the orth/det bounds to about 1e-14, or drop them, and keep the
pin. This is fragile but currently passes; it does not block the gate.

### Claim bindings of the new tests: nominal, not dishonest

No claim states the contracts these 6 new tests check:

* truth_state independence;
* the default variant;
* the gravity guard;
* the decoder floor below 1e-300 and decoding at large scale;
* the exact +/-1 det correction.

* The four lifting/decoder tests are bound to **M3-C01**. Passes 3 and 4 accepted C01 as the home of the decoder
  contract, because C01's test hint includes decoding.
* `test_cost_terms_default_variant_is_iso` is bound to **M3-C24**. This is partly honest: it does evaluate the C24
  identity zQz = J_direct (<= 1e-12) on the default code path. The default itself is not in C24.
* `test_make_params_rejects_non_z_up_gravity` is bound to **M3-C24** and contains **no C24 content**: it is GTSAM
  boundary input validation. C01's domain string (MakeSharedU(9.81)) or C39 would be closer.

These bindings can only add failure modes to a claim. They cannot make a claim falsely DONE, since every claim has
other bound tests with real content. The orchestrator will, however, count them as C01/C24 evidence they are not.
**Recommendation:** add an explicit contract claim (API / input-validation contracts of src/qcqp), or re-bind the
gravity test. Low; not gate-relevant.

## 2. Independent re-checks (the reference imports no src)

| check | result |
|---|---|
| instance vs ref5 PIMs (both bbar, both edges) | dR <= 1.0e-14, dv 0, dp 4.4e-16; taus, W_R, G2, W_vp identical; Dt 0.5000000000000003; truth (Van Loan) 3.3e-15; b_true exact; rw identical; gnss 3.3e-15; prior 2.5e-16 |
| J: src (h = 0.1) vs ref5 (scipy-logm LS fit) | 5.3e-15 to 1.04e-14 |
| z_of_state vs own concatenation, 30 states x both bbar | 7.5e-15 / 1.05e-14 abs (own J); aniso 1.2e-14 relative. With src's own instance data: **0** (base), 5.0e-16 relative (aniso) |
| **families: src A_k vs ref5 matrix equations on ARBITRARY z (20 random z ~ N(0, I))**, base and aniso, both bbar | **same order, same sign**, every family: P0 0; P1 / P2 / RA1 <= 2.6e-16; P3 / RB1 / RB2 / RC1 / RG / RL / RH / RS / P4 <= 1.4e-14 relative (from the 1e-14 J difference); RR <= 3.6e-16 |
| Q and factored cost vs ref5 cost on ARBITRARY z, iso / aniso / partial, both bbar, 50 z each | dense <= 1.6e-15 relative, factored <= 1.3e-15 relative |
| cost vs own 5x5 cost on lifted states (50 each) | <= 4.5e-15 |
| truth cost | factored 6.2e-22, own GTSAM 6.4e-22, own ref 1.1e-20; **dense z^T Q z 1.0e-8** (cancellation, as before) |

### Property tests (1e4 samples each, own encoder, src constraint matrices and costs)

The sampler is R = Exp(2 N), v, p ~ N(0, I) and b ~ 0.3 N. Its b is about 3.7x larger than the suite's
random_state.

| instance | family max on lifted states | cost dense / factored vs own 5x5 |
|---|---|---|
| true, base | P0 0, P1 1.8e-15, P2 1.0e-15, P3 1.6e-14, RA1 2.0e-15, RB1 1.2e-15, RB2 8.1e-15, RC1 5.3e-14, RG 1.7e-14, RL 1.6e-14, RH 1.7e-14, RS 6.2e-15 | 4.2e-15 / 3.3e-15 |
| true, aniso | as base (<= 5.5e-14), P4 1.2e-13, RR 3.6e-12 (bound 1e-10) | 3.3e-15 / 2.6e-15 |
| zero, base | <= 5.2e-14 (RC1) | 3.6e-15 / 2.8e-15 |
| zero, aniso | <= 5.5e-14, P4 1.2e-13, RR 3.5e-12 | 2.8e-15 / 2.5e-15 |
| partial (true) | n/a | factored vs own 5x5 3.0e-15 |

**Decoder vs own decoder, 1e4 random z:**

* The inputs had s in {1, -1, -3.7, 0.05, 1e-6, 1e6}, noise up to 0.5 |s|, reflected R_0 on every 7th sample, s
  near the 1e-12 threshold on every 11th and 13th sample, and both project values.
* There were **0 raise / no-raise disagreements**; 2470 inputs raised in both decoders.
* project = False: difference 0. project = True: difference 0. Orthogonality / det error <= 3.2e-15.

### Spurious-feasible-point hunt

TRF with `xtol = ftol = gtol = 1e-15`, an analytic Jacobian and src's A_k. Whether a point is lifted is judged by
ref5's encoder.

| family set | converged | max \|z - s z(state(z))\| | det R_k |
|---|---|---|---|
| primary P0-P3, bbar = 0 | 200/200 | 1.5e-13 | 600 / 600 are +1 |
| primary, bbar = b_true | 60/60 | 9.1e-14 | all +1 |
| aniso primary P0-P4 | 60/60 | 5.1e-13 | all +1 |
| all 12 families, bbar = 0 | 40/40 | 8.7e-15 | all +1 |
| drop P2 (P0, P1, P3) | 60/60 | n/a | **80 of 180 blocks det -1** |
| **P0, P1, RA1, P3 (no P2; the C44 set)** | 60/60 | 1.8e-13 (consistently lifted) | **83 of 180 blocks det -1** |
| P0, P1, RA1, P3, RB1, RB2, RC1, RG, RL (no P2, RH, RS) | 40/40 | 1.8e-14 | **66 of 120 blocks det -1** |
| drop P3 | 60/60 | 3.5e2 | +1 (liftings off their definitions) |
| drop P1 | 60/60 | 1.9e1 | min \|det\| 2.0e-73 |

The hunt also covered targeted constructions: reflections (C20 and -R_1), R_0 = 0 (C21), liftings with a wrong
rotation or the negative (C22), and the global sign flip (C23). **No spurious point exists for the full primary
set in any variant tried.**

## 3. Claims

Each verdict rests on my own computation, with src as the object under test. On the clean tree the suite passes
72/72 (71 s, OMP_NUM_THREADS = 2) and the clean control copy passes as well.

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | n 124 / 136; s at 123; rho_v(0) = [124,127), rho_p(1) = [133,136); z[123] = 1; src z vs own concatenation with src data: 0 (base), 5.0e-16 relative (aniso) |
| C02 | CONFIRMED | 112, 12; the V' columns of imu0.vel are exactly -1/2 I3 |
| C03 | CONFIRMED | all 7 triples satisfy n = 39K + 6M + 16, and the offset vs the direct count is 39 for each |
| C04 | CONFIRMED | own FD (h = 1e-6, GTSAM Logmap) on 4 fixture cases + 6 random time-varying streams with random bbar: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| 3.95e-10 (<= 1e-8); \|J_R\| 0.37-0.50. Right-Jacobian series vs FD of Gal3.Expmap, 200 x at \|x\| = 0.8: 1.6e-10; block differences exactly 0. Shipped J: zeros exact, \|J_nu,a - J_R\| <= 7.6e-15 |
| C05 | CONFIRMED | own algebra, 2000 draws: max\|r\| 1.8e-15; \|t_j\| 0; bottom rows exactly 0 for arbitrary X_j; arbitrary X_j min max\|r\| 1.15 (> 1e-6); A34 = B34 = Dt exactly |
| C06 | CONFIRMED | 2000 draws, \|eps\| = 0.3: 2.6e-15 |
| C07 | CONFIRMED | src forms vs own 5x5, 500 states per bbar: 1.07e-14 / 1.03e-14 (<= 1e-12) |
| C08 | CONFIRMED | coefficient blocks exact (deviation 0); bias columns exactly 0; truth cost 6.7e-22; Y_0 perturbation min over 20 draws 3.4e-5 (>= 1e-6) |
| C09-C12 | CONFIRMED | truth: <= 8.7e-16 (base), RR 8.7e-15; 1e4-sample property run in Sec. 2: base <= 5.5e-14 (<= 1e-12); P4 1.2e-13, RR 3.6e-12 (<= 1e-10) |
| C13 | CONFIRMED | min family max over 20 draws: 1.34e-5 (RB1, base), 1.22e-5 (aniso), both > 1e-7 |
| C14 | CONFIRMED | sizes exactly as stated; every family full rank; primary 106 / 106 (aniso 118 / 118); joint 472 -> 453 (aniso 490 -> 471) |
| C15 | CONFIRMED | own row-handedness matrices built entrywise from eps: each matches a src P2 matrix, mismatch 0 |
| C16 | CONFIRMED | rank 11 for each KF; the trace identity is exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric (base and aniso); homogeneity <= 6.5e-17; the only b != 0 is s^2, b = 1 |
| C18 | CONFIRMED | Sec. 2: 200/200 converged, consistency 1.5e-13, det +1. Drop P2: 80 of 180 blocks det -1. Drop P3: 3.5e2. Drop P1: \|det\| 2e-73 |
| C19 | CONFIRMED | 300 starts: 237 SO(3), 63 zero, 0 other |
| C20 | CONFIRMED | violated: P2 1.95, RH 8.5e-2, RS 1.5e-2; satisfied <= 8.7e-16 (both bbar) |
| C21 | CONFIRMED | P1 = RA1 = 1.0; all others <= 4.2e-16 |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: P3 9.6e-2, RL 8.4e-2, RH 9.0e-2, RS 1.3e-2; satisfied <= 4.4e-16 |
| C23 | CONFIRMED | residuals of -z equal those of z exactly (base and aniso, both bbar); cost difference 0.0 (dense and factored); decode(-z) == decode(z) exactly, both project values |
| C24 | CONFIRMED | 1e4 lifted states per (bbar, layout) vs own 5x5 cost: iso 4.2e-15 / 3.6e-15; aniso 3.3e-15 / 2.8e-15 (<= 1e-12) |
| C25 | CONFIRMED | iso / aniso / partial at both bbar: exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0 exactly; bias block equals the rw terms and own rw matrix exactly |
| C26 | CONFIRMED | truth factored 6.2e-22 (<= 1e-15); own GTSAM graph, 50 states per eps: 1.06e-4, 2.66e-6, 3.57e-7 (bounds 5e-4, 5e-5, 5e-6) |
| C27 | CONFIRMED | deviations at lam = 1: rot / vel / pos 2.011e-2 / -1.223e-2 / -2.098e-3 (<= 3e-2). Halving factors: rot 0.250 / 0.249 / 0.249; vel 0.501 / 0.500 / 0.500; pos 0.504 / 0.502 / 0.501. Own resid5 with own J at each bbar gives the same values to 4 digits |
| C28 | CONFIRMED | 2.088e-7, 2.605e-8, 3.253e-9, 4.064e-10; factors 8.02 / 8.01 / 8.00 (scipy expm, own J) |
| C29 | CONFIRMED | half 2.9e-7 (<= 5e-6); full 0.86 (>= 0.1) |
| C30 | CONFIRMED | own whitened residual (own cost pieces), own local parametrisation: rank 45, cond 3.946e7 / 4.527e7; src whitened residual: rank 45, cond 3.946e7 / 4.527e7 |
| C31 | CONFIRMED | 3I: 4.020000 (range 2.3e-13); diag(1,2,5): range [1.438, 6.576] (ptp 5.14 > 1.0); src lam^2 coefficient (Q block) equals the closed form from own weights, relative 0 |
| C32 | CONFIRMED | own exactly-orthogonal sampler, rng 2, own encoder: 2.3e-15, 2.8e-15 (<= 1e-12); diag(1,2,5) 7.9e-2; random SPD 1.3e-1; diag(1,1,1.01) 8.7e-4; rank 2015. Shipped encoder: identical to 3 digits |
| C33 | CONFIRMED | partial vs own GTSAM graph: 7.4e-5, 2.8e-6, 1.5e-7 (bounds 0.05 eps); D min eig 3.460e7 >= 0; G2 information correlation -0.867 |
| C34 | CONFIRMED | body 2.9e-3, 2.1e-4, 1.6e-5 (<= eps); world 1.0e-1, 1.0e-1, 6.6e-2 (>= 1e-2) |
| C35 | CONFIRMED | rng 4, own sampler and encoder: rank 2015, null 263, gap 1.9e-7 -> 4.5e-16. P0 homogenises to 0 (b -> b s^2). Spans 60 / 92 / 106 / 136 / 226 / 236. Projection residual <= 2.5e-11 (<= 1e-10). The pass-3/4 fragility caveat stands: margin 4x, rotations orthogonal to ~1e-16 |
| C39 | CONFIRMED | Dt 0.5000000000000003 for both edges; bbar = 0 taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6 (own = src); rw diagonal 5.5402e9 / 2.2222e5 |
| **C44** | **REFUTED** | below |

**M3-C44 (brief only): REFUTED.** For a square matrix, R^T R = I and R R^T = I are equivalent; both say "R is
orthogonal". Neither fixes the sign of det R, and c1 x c2 = c3 is exactly det R = +1. Measurements:

* **Exact counterexample** (mpmath, 50 digits): for R = diag(1, 1, -1), \|R^T R - I\| = \|R R^T - I\| = 0, while
  \|c1 x c2 - c3\| = 2.0 and det R = -1.
* **1e4 random improper orthogonal matrices** (QR, column-flipped): both orthogonality residuals <= 1.6e-15; the
  minimum of max\|c1 x c2 - c3\| is 1.17. Every one is P1- and RA1-feasible and violates handedness.
* **The QCQP feasible set changes.** With P2 removed and P1 + RA1 kept (P0, P1, RA1, P3), TRF from 60 random starts
  converges 60/60. The solutions are consistently lifted (\|z - s z(state(z))\| <= 1.8e-13), and **83 of 180
  rotation blocks have det = -1**. Adding RB1, RB2, RC1, RG and RL still leaves 66/120 blocks with det -1.
  Constructed points with R_0 -> R_0 diag(1,1,-1), or R_1 -> -R_1, with liftings recomputed, satisfy P0, P1, RA1,
  P3, RB1, RB2, RC1, RG and RL to <= 8.7e-16, and violate P2 (1.95 / 1.8), RH (8.5e-2) and RS (1.5e-2).

The handedness rows are therefore **not** redundant in the QCQP. They are what excludes the O(3) \ SO(3) component;
equivalently, C18's drop-P2 result and C20. Only RH / RS, which are themselves cross-product identities, also
remove the reflection.

## 4. Mutation sweep

### 4.1 New mutants (176; ids `p5-*`)

By file: lifting 42, constraints 32, cost 27, forms 9, gtsam_boundary 22, instance 18, fixture 23, `__init__` 3.
Each is one exactly-once substitution on a tempfile copy, judged by the full tests/test_qcqp.py with `-x`.

**Killed (168), grouped by first failing test:**

* **test_instance_matches_independent_gtsam_reference[true] (50)**
  * gtsam_boundary: to-bias-swap, gal3-tneg, nav-t1, gyrocov-nosq, intcov-zero, acc-uses-g, pim-bbar0,
    int-drop-first, G-p-plus, G-t-plus, inc-right, inc-negb, J-2x, J-leftchart-lp, navrows-swap, cov-T2.
  * instance: tau-1, S2-vv-pp, Svp-offT, dv-dp-swap, J-at-0, WR-noinv, Dt-0.5, params-swap, rw-nodt,
    gnss-kf1-only, sigma-swap, cache-swap.
  * fixture: DT, TEDGE, BA-digit, omega0-sign, f0-digit, v0, rv0, sg, sint, sgnss, sba-rw, hat-entry, exp-th,
    exp-branch-1, m1-sign, m2-KK, dp-m1, true-p-gminus, true-R-left, gyro-minus-bias, acc-BG, g-std.
* **test_c01_index_map[true] (40)**
  * lifting: v-reads-p, b-shift, V-is-Vp, Pp-is-P, s-minus1, rhop-eq-rhov, n-aniso-3, biasidx-short, vecform-neg,
    matform-rowmajor, sform-neg, lift-bias-j, lift-W-ba, lift-cv-cp-swap, lift-Yp-WT, z-R-rowmajor, z-Vp-is-V,
    z-s-s, z-abs-s, z-rhov-M, z-rhop-rv, z-rho-M-j, z-no-rho, s-abs, b-no-div, p-abs-s, project-inverted,
    project-default-true, zs-default-neg, nedges-1.
  * cost (through the aniso rho vs resid5): cp-Jnu, g-neg, rv-M-dv, rv-Vp-sign, rp-g-plus, rp-Mcp-sign, Rcv-T.
  * forms: cmat-CT. Instance: Jnu-is-Jrho, JR-cols.
* **test_c09 (17):** cn cyclic-anti, resid-plus-b, M-dRT, Rj-is-Ri, w-ba, cp-Jnu, jside-Vp-Pp-swap, P0-2, P2-plus,
  P3-AY, P3-V-cp; fm eps-sym, matmul-BT, matvec-AT, cross-perm, hat-T, cvec-CT.
* **test_cost_terms_default_variant_is_iso (12):** co rR-M-plus, rR-YY, rot-weight-full, vel-taup, wprior-nosq,
  R0-prior-full, v0-sign, gnss-drop2, gnss-nosq, rw-nodiff, rw-one, Q-noW. This new test is the first killer of
  most cost mutants because it runs early; later tests (C24 / C26) would also kill them.
* **test_c11 (8):** cn RG-VP-sign, RG-YP-W, RL-P-cv, RL-WA, RH-Wrow, RH-cross-swap, RH-ea-neg, RS-ATY.
* **test_c14 (4):** cn emit-drop-last, flat-drop-first, vech-nodiag; fm dot-aa.
* **test_c10 (3):** RB1-YT, RB2-sign, RC1-P-cv.
* **test_c12 (3):** P4-p-rv, RR-cross, rho-swap.
* **test_P1_RA1 (3):** upper-offdiag, P1-is-RA1, RA1-is-P1.
* **test_c33 (3):** co partial-kron-swap, partial-G2-diag; in D-full.
* **test_decoder_projection_is_nearest_rotation (3):** proj-D-first, proj-noD, proj-before-div.
* **test_make_params_rejects_non_z_up_gravity (2):** guard-le, guard-noy.
* **test_decoder_s_guard_threshold_tie_and_scale (2):** sfloor-1e-10, floor-1e-30.
* **test_package_exports (3):** drop-Layout, bi-is-instance, truth-is-State.
* **test_c08 (2):** co vecrows-rowmajor (the partial weighting sees the row order), fac-noW.
* **test_truth_state_blocks_are_independent_arrays (3):** truth-p-alias, truth-R-alias, truth-b-listmul.
* **One mutant each:**
  * test_bias_boundary_converters: from-bias-swap.
  * test_c13: fmax-signed.
  * test_c24[aniso]: co rho-swap.
  * test_c34: co ignore-Wvp.
  * test_cost_terms_rejects_unknown_variant: co unknown-pass.
  * test_decoder_degenerate_inputs: bad-gt1.
  * test_decoder_s_guard_floor_below_1e_minus_300: floor-1e-310.
  * test_form_helpers: fm sym-noop.
  * test_state_copy_is_independent: copy-v-shallow.
  * test_whitened_residual: whiten-nosqrt.

### 4.2 Replays

* The 7 F06 survivors: 7/7 killed (Sec. 1).
* `tests/test_mutation_m3.py`: **77 passed** (88 s). The 75 mutants, including the 7 new p4-* entries, are each
  killed by their named killers, plus the well-formedness test and the clean control.

### 4.3 Complete survivor list (8) with rulings

The threshold for equivalence is 1/10 of the governing claim's tolerance, never below the 8-ulp floor. As in passes
3-4, a change in a claim-free contract (exception vs return, aliasing) is non-equivalent unless a proof shows that
no src output changes.

| # | mutant | measured effect (487-group dump diff vs clean) | ruling | class |
|---|---|---|---|---|
| 1 | **p5-lf-zmax-pre-s** (`np.abs(z).max()` -> `np.abs(z[:layout.s]).max()`) | 0 of 487 groups. Targeted probe: an aniso z (n = 136) with rho entries ~2e13 and s = 1. Clean raises ("s ... is zero relative to max\|z\| = 2e13"); the mutant **decodes** (both project values) | **NON-EQUIVALENT** | low: in the aniso layout the max skips the 12 rho entries, so an SDP vector whose mass is in rho with s ~ 0 relative to it decodes instead of raising. In the base layout it only drops s, which is equivalent by the pass-4 #8 proof. No test exercises the guard on an aniso z with dominant rho |
| 2 | **p5-gb-guard-noshape** (drops `g_w.shape == (3,) and`) | `make_params([0, 0, -9.81, 0])`: clean ValueError, mutant **accepts** (4th entry silently ignored). Shapes (1,3) / (3,1) still raise a ValueError from numpy truth-value ambiguity | **NON-EQUIVALENT** | low: the shape check that 519864f advertises ("also rejects ... wrong shape") is untested; the test's inputs are all shape (3,) |
| 3 | **p5-gb-guard-finite** (adds `and np.isfinite(g_w[2])`) | `make_params([0, 0, -inf])`: clean **accepts** (n_gravity -inf), mutant raises | **NON-EQUIVALENT** | very low and an added clause, like pass-4 #6. It exposes a src gap: the guard's own contract "g > 0" does not reject g = inf, which would propagate inf/NaN into every PIM. Killing it requires a src change (isfinite) plus a test |
| 4 | p5-gb-cov-T (`C[NAV_ROWS[a], NAV_ROWS[b]]` -> `C[NAV_ROWS[b], NAV_ROWS[a]].T`) | 28 groups. preintMeasCov is not bitwise symmetric. cov blocks <= 1.0e-17 relative; W_R / D <= 3.2e-18; W_vp <= 2.9e-17; Q <= 4.2e-20; partial / aniso terms <= 2.9e-17; whitened residual <= 1.5e-15 relative to the group max; taus unchanged | EQUIVALENT | rule 1: the effect on every weight is <= 3e-17 relative, << 1/10 of C24's 1e-12, C33 / C34's 0.05 eps and C30's 1e-12 cut. Neither orientation of a rounding-asymmetric covariance is more correct |
| 5 | p5-in-cache-1 (`lru_cache(maxsize=None)` -> `maxsize=1`) | 0 groups (values and the consecutive-call identity are unchanged) | EQUIVALENT | proof: `build_instance` is deterministic (bitwise-identical digest across builds and across commits, Sec. 1), and no src code depends on instance identity. Only memoisation (runtime) changes; a caller that edits the cached instance in place is outside every contract (see #6-#8) |
| 6 | p5-in-btrue-alias (`b_true=F.BIAS_TRUE.copy()` -> `F.BIAS_TRUE`) | aliasing vector only: `shares_memory(I.b_true, F.BIAS_TRUE)` True; all values identical | EQUIVALENT | proof, as for pass-4 #16 (prior aliases truth): no src function writes in place into an Instance field, and `truth_state` copies b_true (now tested). The only observable change needs a caller's in-place write on a frozen instance's field. That would corrupt the fixture constant for later builds; see risk note R4 |
| 7 | p5-in-bbar-nocopy-g (`g = F.G_W.copy()` -> `F.G_W`) | aliasing only: `I.g is F.G_W` | EQUIVALENT | same proof: `inst.g` is only read (cost.py:61) |
| 8 | p5-fx-true-alias (`V_0.copy(), P_0.copy()` -> `V_0, P_0` in `true_states`) | aliasing only: `I.truth[0][1]` / `[0][2]` share memory with F.V_0 / F.P_0 | EQUIVALENT | same proof: `truth_state`, `prior` and `gnss` all copy, and no src function writes into `I.truth` |

Totals: 8 survivors. **Non-equivalent 3 (#1-#3), all low. Equivalent 5 (#4-#8).**

## 5. Tests that cannot fail or compare code with itself (at 519864f)

1. test_project_so3_det_correction_is_exact: its reference is the same LAPACK SVD as src (a pin, Sec. 1), and its
   orth/det bounds have 1.2x headroom over the worst of 21 seeds. They are redundant with the pin. This is fragile
   but not wrong.
2. test_make_params_rejects_non_z_up_gravity is bound to C24 but contains no C24 content (Sec. 1).
3. Unchanged from pass 4:
   * the tie-precondition arithmetic in test_decoder_s_guard_threshold_tie_and_scale;
   * the `fit <= 1e-14` reference check in test_c04_instance_J (the src check is the 5e-14 pin);
   * the test_c02 / test_c03 arithmetic;
   * the test-side-only parts of C15, C26, C33, C39 and C04;
   * the MATH CHECK labels on C05 / C06 / C28;
   * the scale invariance of C30's rank.
4. **Untested surface (all of it shows up as survivors):**
   * the decoder guard on aniso z with dominant rho;
   * make_params' shape check;
   * make_params on infinite gravity.

## 6. Findings and risks for M4

* **M3-F07 (new, gate-blocking, low): 3 non-equivalent survivors (Sec. 4.3 #1-#3).** Tests that would kill them:
  * (a) for #1: `za = z_of_state(truth, I, ANISO); za[124:] = 1e13; state_of_z(za, ANISO)` must raise (both
    project values);
  * (b) for #2: `make_params([0, 0, -9.81, 0], ...)` must raise ValueError (and, for robustness, `[[0, 0, -9.81]]`
    with `match="z-up"`);
  * (c) for #3: a **src fix** (add `np.isfinite(g_w).all()` to the guard), plus a test that `[0, 0, -inf]` and
    `[0, 0, np.inf]` raise.
* **Fragility (non-blocking):** loosen or drop the 4e-15 orthogonality / det bounds in
  test_project_so3_det_correction_is_exact; the `worst_ref` pin kills the target mutant with 5x margin.
* **Bindings (non-blocking):** add a contract claim for src/qcqp API / input validation, or re-bind the gravity test
  away from C24.
* **M4 risks:**
  * **R1, dense-Q cancellation persists.** At truth, z^T Q z = 1.0e-8 vs factored 6.2e-22 (max\|Q\| ~ 1e10). Use the
    factored cost for primal values and duality gaps, and rescale Q (or the variables) before any Clarabel solve.
    The constraint entries are O(1) and the cost is O(1e10). With this disparity, Barfoot's
    sigma_1/sigma_2 >= 1e5 rank test cannot be resolved reliably.
  * **R2, handedness is load-bearing (C44).** Dropping P2 admits det -1 points that are consistently lifted (83/180
    blocks in the hunt). Any M4 constraint pruning (for rank deficiency or for speed) must keep P2 or the RH / RS
    cross-product families.
  * **R3, redundant families are linearly dependent** (472 -> rank 453; aniso 490 -> 471). Remove the dependencies
    before handing them to Clarabel.
  * **R4, Instance arrays are mutable.** The instance is a frozen dataclass, but its numpy arrays are writable, and
    `instance()` is lru-cached. One in-place write by M4 code would poison every later caller. Consider
    `setflags(write=False)` on the Instance arrays.
  * **R5, C35's 1e-10 margin** holds with 4x headroom only for rotations orthogonal to ~1e-16. Any AutoTight rerun
    needs the same sampler, or a tolerance a decade above the gap edge.
  * **R6, the surrogate gap at bbar = 0** (truth cost 3.3e-3 vs ~1e-21 at bbar = b_true) lies outside C26's domain
    and still applies.
  * **R7, results are not bit-comparable.** No MOSEK is used, so results will not be bit-comparable with Rai's
    published tables; say so wherever they are compared.
