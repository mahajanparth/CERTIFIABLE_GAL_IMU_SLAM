# Stage B verification, pass 6: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 6 (agent verify-M3-006). Base commit 75c7908, the implementer's response to
the pass-5 finding M3-F07. Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py, docs/claims/M3.yaml
(M3-C01..C35, C39), plus one claim given only in the brief, M3-C45. I did not read docs/03_qcqp_derivation.md,
experiments/m3_scratch/ or any seeds file. I did not edit src/, tests/ or docs/claims/.

My scripts lived in a scratch directory inside the worktree (`_v6scratch/`, untracked, deleted before the commit).
Mutant trees went to Python `tempfile` directories. The scripts:

* `ref6.py`: a new independent reference. It imports neither src nor tests. It contains:
  * its own retyped literals and its own GTSAM PIMs;
  * Ubar as a 5x5 matrix built from `pim.predict(NavState(), b)` and its own G;
  * J as a 200-sample least-squares fit of the **scipy `logm` chart** of `Ubar^-1 U(bbar + d)` (fit residual
    1.2e-15 to 1.5e-15);
  * the truth from a **Van Loan block exponential** `expm([[W, f, 0], [0, 0, 1], [0, 0, 0]] T)`;
  * its own encoder, decoder and SVD projection (det sign from `det U * det V^T`, not `det(U V^T)`).

  **Every constraint family is re-implemented as matrix equations on the blocks of an arbitrary z** (this is the
  re-implemented core routine), as are the C07 residual forms on arbitrary z, the cost on arbitrary z (iso / aniso /
  partial) and the cost of a state from 5x5 cleared Gal(3) residuals.
* `claimsB.py` has its own GTSAM factor graph: GalileanImuFactor.unwhitenedError with its own weights, PriorFactorPose3 /
  PriorFactorVector / GPSFactor through `Diagonal.Precisions` / `Isotropic.Sigma`, and BetweenFactorConstantBias with
  its precisions in GTSAM's (accel, gyro) order; cost = 2 graph.error.
* `cmp6.py`, `prop6.py`, `hunt6*.py`, `claimsA/B/C.py`, `c35b.py`, `c45.py`: the src-vs-reference comparisons, the
  1e4-sample property tests, the spurious-point hunt, the per-claim computations, the C35 sampler study, and C45.
* `dump6.py`, `cmpdump.py`, `measure6.py`: record 750 observable output groups of src/qcqp and diff a tree against
  another. The groups cover:
  * every Instance / EdgeData field (both bbar), the cov blocks and their keys, the D / J_R / J_nu / J_rho properties;
  * z at 6 states x 2 layouts x 2 scales x 2 bbar;
  * every constraint (names, families, b, diagonals, A z on random z, residuals, signed family residuals,
    family_max, vech);
  * terms / Q / W / F / factored cost / whitened residual for iso, partial, aniso (3 weights including a singular
    one), an unknown variant and aniso on the base layout;
  * about 120 decoder probes (scales 1e-320 to 1e305, threshold ties at 4 magnitudes, NaN / inf at 3 positions,
    aniso z with dominant rho at 3 positions, zero / ones / reflected R_0, lists, random z, both project values);
  * 65 `project_so3` inputs including singular ones;
  * an aliasing vector and a **writeability vector** (every Instance / edge array, the dict types, truth_state
    blocks, build_instance on a caller's array, the fixture constants);
  * 19 gravity inputs to make_params, the boundary helpers, every forms helper, the fixture literals and functions,
    the exports, the layout.

  Two dumps of the same tree are identical (0 of 750 groups differ).
* `harness6.py`, `mutants6.py`, `replay6.py`: 217 new mutants plus the 7 F07 replays. Each mutant is a real
  exactly-once textual edit on a tempfile copy of src + tests/test_qcqp.py + conftest + fixtures; the pattern
  count is checked to be exactly 1 before the edit. The judge is the **full** tests/test_qcqp.py, slow tests
  included, run with `-x` (single-threaded BLAS, 10 workers). A clean control copy passes (74/74 in 68 s; 121 s for
  the full file run serially in a clean copy).

## Verdict summary

* **F07 (pass 5): FIXED.** The 3 pass-5 survivors and the 4 freeze mutants, replayed as real file edits and judged by
  the full suite, are all killed (Sec. 1). The src changes do not change any claim result: 735 of 750 output groups
  are bitwise identical between 519864f and 75c7908, and the 15 that differ are exactly the intended changes. The
  freeze does introduce a serialisation regression and leaves one writable path (new finding M3-F09).
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C45 REFUTED. 0 UNTESTABLE.**
* **Fresh sweep: 217 new mutants over all 8 files of src/qcqp + 7 replays = 224 real-file-edit runs.** 189 are
  killed (all 7 replays and 182 of the new mutants), and 35 new mutants survive. Of the 35 survivors, **31 are
  equivalent and 4 are NON-EQUIVALENT (all low)**. The replay of `tests/test_mutation_m3.py` passes **84/84**: the
  82 mutants are each killed by their named killers, plus the well-formedness test and the clean control.
* **GATE (zero surviving non-equivalent mutants): FAILED**, on 4 low survivors (new finding M3-F08, Sec. 6):
  * the decoder no longer forces float64 (`np.asarray(z)` without dtype);
  * `vech` returns the lower triangle (its documented row-major upper order is untested);
  * `nav_to_gal3`'s default time `t = 0.0` is untested;
  * `build_instance` freezes the caller's bbar array in place if the `.copy()` is dropped (the freeze made that copy
    load-bearing, and no test checks it).

  Every mutant that changed z, Q, a family's content, the instance data, the GTSAM boundary arithmetic, the forms
  helpers, the fixture, the exports, the freeze coverage, or the decoder's s / threshold / floor / projection logic
  was killed.

## 1. Finding M3-F07 ruling: FIXED

### Replays (real exactly-once edits on 75c7908, full suite, `-x`)

| mutant | edit | result: killer and failing assertion |
|---|---|---|
| p5-lf-zmax-pre-s | `np.abs(z).max()` -> `np.abs(z[:layout.s]).max()` | killed by test_decoder_s_guard_counts_aniso_rho_entries (DID NOT RAISE) |
| p5-gb-guard-noshape | drops `g_w.shape == (3,) and` | killed by test_make_params_rejects_non_z_up_gravity (DID NOT RAISE) |
| p5-gb-guard-finite | the isfinite clause removed (src fix reverted) | killed by test_make_params_rejects_non_z_up_gravity (DID NOT RAISE) |
| p7-in-no-setflags | `x.setflags(write=False)` -> `pass` | killed by test_cached_instance_arrays_are_read_only (`('true', 'bbar')`) |
| p7-in-not-frozen | `_frozen(Instance(` -> `(Instance(` | killed by the same test (`('true', 'bbar')`) |
| p7-in-edges-not-frozen | `replace(inst, edges=edges, **kw)` -> `replace(inst, **kw)` | killed by the same test (DID NOT RAISE TypeError on the cov dict) |
| p7-in-dict-writable | MappingProxyType dropped | killed by the same test (DID NOT RAISE TypeError) |

Note on p7-in-edges-not-frozen: `setflags` acts in place, so even with the frozen edges discarded the edge arrays stay
read-only and only the cov dict stays mutable. The kill comes from the dict check, which is still a valid kill.

### The src changes do not change any claim result

Dumps of the 750 groups at 519864f (= e6823a1; that commit only adds the pass-5 report) and at 75c7908: **735 groups
bitwise identical**. The 15 that differ are:

* the writeability vectors (`true.writeable`, `zero.writeable`, `bi.wr`): every array is now read-only;
* `make_params([0, 0, -inf])`: was accepted (n_gravity -inf), now raises;
* 11 gravity error messages, which gain the word "finite".

Every instance field, z, constraint, cost, decoder output, projection, GTSAM helper and fixture value is unchanged.

### Is anything in src writing to the frozen arrays?

No. The clean suite passes on read-only instances. A grep shows no in-place writes to instance fields. Any such
write would now raise.

### Freeze coverage hunt

| path | state at 75c7908 |
|---|---|
| bbar, g, b_true, rw_precision, truth[k][q], prior[q], gnss[k] | read-only |
| edge dR, dv, dp, J, W_R, G2, W_vp, cov[*] and the J_R / J_nu / J_rho views | read-only |
| gnss, cov | `mappingproxy` (assignment raises TypeError) |
| `EdgeData.D` | fresh writable array per call (harmless: it is not cached) |
| `truth_state(inst)` blocks | fresh writable copies (correct) |
| `build_instance(user_array)` | the caller's array is not frozen and not aliased (correct) |
| fixture constants F.BIAS_TRUE / G_W / V_0 / P_0 | still writable and not aliased (correct) |
| `Term.W` of aniso / rw terms | the instance's read-only W_vp / rw_precision, so an in-place write now raises instead of poisoning the cache |
| **cov block `.base`** | **writable**: every cov block is a view of the 9x9 `preintMeasCov()` copy, and `ed.cov[k].base[...] = x` succeeds and changes all 9 cached blocks |
| `a.setflags(write=True)` | allowed on every frozen array (numpy semantics: they own their data or have a writable base); this needs a deliberate call |
| `pims` (GTSAM PIM objects) | mutable (acknowledged); no src code reads them after build_instance |

**Regression introduced by the freeze.** At 519864f, `pickle.dumps(instance)` (16305 bytes, pims included),
`copy.deepcopy(instance)` and `dataclasses.asdict(instance)` all work. At 75c7908 all three raise
`TypeError: cannot pickle 'mappingproxy' object`, even with `pims=()`. No claim or test covers this, but it matters
for M4: process-pool solves, caching to disk and deepcopy-based perturbation all need it. See M3-F09.

### The new tests

* **test_decoder_s_guard_counts_aniso_rho_entries**: real. It uses the test-side reference z, puts +-1e13 in all of
  the rho slots, in the first one only and in the last one only, under both project values, and checks the decoding
  side at 1e11. It kills p5-lf-zmax-pre-s and my p6-lf-dec-zmax-base (`z[:layout.n_base]`).
* **Gravity test**: real. The 2-D and 0-d cases now require the guard's own "z-up" message, so the numpy
  truth-value ValueError no longer counts as a pass. It kills guard-noshape, guard-finite, guard-le, guard-nox and
  isfinite `.any()`. The only surviving guard mutant, `isfinite(g_w[2])`, is equivalent: the x and y clauses demand
  exactly 0.0, which is finite.
* **Read-only test**: real for the cached instances. It checks 40 arrays per instance (the count is asserted), both
  dicts, cache identity, and that the reference data is intact afterwards. Across the sweep it killed 12 freeze
  mutants: each top-level field skipped, the edge J / W_vp / cov skipped, tuple recursion dropped, dict values not
  recursed, and no cache. **Gaps:** it covers only `instance()`, not `build_instance()` on a caller's array, and not
  the `.base` path (Sec. 4.3 #24, M3-F09).

### Projection bounds loosened 4e-15 -> 2e-14: sound

* The pin compares src with `U diag(1, 1, d3) V^T` from the same `np.linalg.svd` on the same A. It is therefore 0 on
  any LAPACK build, because the call is deterministic, and it is strictly stronger than an orth / det bound against
  any deviation from that formula.
* The det-nosign mutant deviates by 2.4e-15 (pass 5), 5x above the 4.5e-16 pin. It stays killed (replayed in
  test_mutation_m3.py), and the mutant's orth / det values (<= 7.3e-15) no longer reach the loosened bounds, so the
  kill rests on the pin alone, as the docstring says.
* Correctness (the nearest proper rotation) is checked independently by test_decoder_projection_is_nearest_rotation.
  My own decoder agrees exactly (Sec. 2), and four projection mutants are killed (D-first, svd(A^T), det U only,
  sign of |det|).

**Accepted.**

### Unbinding the gravity and read-only tests: honest

Both tests carry an explicit `NOT claim-bound` comment with the reason (no M3 claim states input validation or
immutability). They therefore no longer inflate C24, or any claim, with content that claim does not have. This
answers the pass-5 binding remark. One remark remains: test_decoder_s_guard_counts_aniso_rho_entries is bound to
C01, which is consistent with the decoder-contract binding accepted in passes 3-5.

## 2. Independent re-checks (the reference imports no src)

| check | result |
|---|---|
| instance vs ref6 PIMs (both bbar, both edges) | dR, dv, dp **0**; cov blocks, taus, W_R, G2, W_vp identical; Dt 0.5000000000000003; truth (Van Loan) 3.3e-15; gnss 3.3e-15; g, b_true, bbar, rw identical |
| J: src (h = 0.1) vs ref6 (logm LS fit) | 5.1e-15 to 9.6e-15 |
| z_of_state vs own encoder, 30 states x both bbar x 2 layouts | <= 4.9e-15 relative |
| **families: src A_k vs ref6 matrix equations on ARBITRARY z (20 z ~ N(0, I))**, base and aniso, both bbar | **same order, same sign, same count**, every family: P0 0; P1 / P2 / RA1 <= 3.3e-16; P3 / RB1 / RB2 / RC1 / RG / RL / RH / RS <= 1.9e-14 relative (driven by the 1e-14 J difference); P4 <= 6.0e-16; RR <= 5.2e-16 |
| Q and factored cost vs own cost on ARBITRARY z, iso / partial / aniso, both bbar, 50 z each | dense <= 8.5e-16 relative, factored <= 2.2e-16 relative |
| factored cost vs own 5x5 cost on lifted states (50 per bbar) | <= 1.7e-16 relative |
| truth cost (bbar = b_true) | factored 6.7e-22, own 5x5 6.3e-22, own GTSAM 6.2e-22; **dense z^T Q z 1.2e-8** (cancellation, as before) |

### Property tests (1e4 samples each; own encoder, src constraint matrices and costs)

The sampler is R = scipy `expm(hat(2 N))`, v, p ~ N(0, I) and b ~ 0.3 N. Its rotations are orthogonal only to
**2.8e-13** at the largest angles, which accounts for the P1 / P2 / RA1 level below.

| instance | family max on lifted states | iso cost vs own 5x5 (dense / factored) |
|---|---|---|
| true, base | P0 0, P1 2.7e-13, P2 2.6e-13, P3 1.8e-13, RA1 2.7e-13, RB1 8.5e-16, RB2 4.4e-14, RC1 2.5e-13, RG 9.5e-14, RL 1.1e-14, RH 1.8e-13, RS 8.2e-14 | 1.9e-15 / 4.3e-16 |
| true, aniso | as base; P4 1.6e-12, RR 1.9e-11 (bound 1e-10); aniso factored vs own 3.8e-16 | 1.7e-15 / 4.3e-16 |
| zero, base | <= 2.8e-13 | 2.0e-15 / 4.0e-16 |
| zero, aniso | as base; P4 1.6e-12, RR 2.2e-11 | 1.6e-15 / 4.0e-16 |

All are within the C09-C12 bounds (1e-12 base, 1e-10 aniso). The base headroom is 3.6x with this deliberately less
orthogonal sampler.

**Decoder vs own decoder, 1e4 random z x 2 project values:**

* The inputs alternate base / aniso, use s in {1, -1, -3.7, 0.05, 1e-6, 1e6}, add noise on every 3rd sample, reflect
  R_0 on every 7th, put s at 1e-12 max|z| (+-1e-9 relative) on every 11th / 13th, and put a 1e14 entry (a rho entry
  for aniso) on every 17th.
* There are **0 raise / no-raise disagreements**; 3248 inputs raised in both decoders.
* The decoded states are bitwise equal (max difference 0). The projected rotations are orthogonal to 3.3e-15.

### Spurious-feasible-point hunt

TRF with `xtol = ftol = gtol = 1e-15`, an analytic Jacobian 2 A_k z and src's A_k. Whether a point is lifted is
judged by ref6's decoder and encoder.

| family set | converged | max \|z - s z(state(z))\| | det R_k |
|---|---|---|---|
| primary P0-P3, bbar = 0 (rng 7) | 200/200 | 2.0e-13 | 600 / 600 are +1 |
| primary, bbar = b_true | 60/60 | 1.9e-13 | 180 / 180 +1 |
| aniso primary P0-P4 | 60/60 | 1.6e-13 | 180 / 180 +1 |
| all 12 families, bbar = 0 | 40/40 | 6.5e-15 | all +1 |
| all 14 aniso families | 30/30 | 7.1e-15 | all +1 |
| drop P2 (P0, P1, P3) | 59/60 | 3.8e-14 (lifted) | **87 of 177 blocks det -1** |
| drop P2, RH, RS (keep every other redundant family) | 40/40 | 1.6e-14 | **57 of 120 blocks det -1** |
| drop P3 (P0, P1, P2) | 60/60 | **4.7e2** | +1 (liftings off their definitions) |
| P0, P1, P2 + Gram families RA1, RB1, RB2, RC1, RG (no P3, RL, RH, RS) | 60/60 | **7.5e-4** | +1 |
| drop P1 | 60/60 | 1.9e1 | min \|det\| **2.2e-73** |

The hunt also covered constructed points: reflection (C20), R_0 = 0 (C21), wrong-rotation and negated liftings
(C22), and the global sign flip (C23). **No spurious point exists for the full primary set in any variant.** The
Gram-only run confirms C22 at the solver level: without P3 / RL / RH / RS, the Gram identities alone leave the
liftings unpinned.

## 3. Claims

Each verdict rests on my own computation, with src as the object under test. On the clean tree the suite passes
74/74 (121 s, single-threaded).

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | n 124 / 136; s at 123, z[123] = 1; rho_v(0) = [124,127), rho_p(1) = [133,136); src z vs own encoder 4.6e-15 (base), 1.2e-15 relative (aniso) |
| C02 | CONFIRMED | 112, 12; the V' columns of imu0.vel are exactly -1/2 I3 (all other entries 0) |
| C03 | CONFIRMED | all 7 triples satisfy n = 39K + 6M + 16, and the offset vs the direct count is 39 for each |
| C04 | CONFIRMED | own FD (h = 1e-6, GTSAM Gal3 Logmap of own 5x5 U) on 4 fixture cases + 6 random time-varying streams with random bbar: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| 3.95e-10 (<= 1e-8); \|J_R\| 0.37-0.50. Own right-Jacobian series sum (-ad_x)^n/(n+1)! vs FD of Gal3.Expmap, 200 x at \|x\| = 0.8: 1.4e-10; block differences exactly 0. Shipped J: zeros exact, \|J_nu,a - J_R\| 7.6e-15 |
| C05 | CONFIRMED | own algebra, 2000 draws: max\|r\| 2.2e-15; \|t_j\| 0; bottom rows exactly 0 for arbitrary X_j; arbitrary X_j min max\|r\| 1.23 (> 1e-6); A34 = B34 = 0.5000000000000003 = Dt |
| C06 | CONFIRMED | 2000 draws, \|eps\| = 0.3, eps_alpha = 0: 3.1e-15 |
| C07 | CONFIRMED | src forms vs own 5x5 cleared residual, 500 states per bbar (b ~ 0.2 N): 5.8e-15 / 6.9e-15 (<= 1e-12) |
| C08 | CONFIRMED | coefficient blocks exact (deviation 0); bias columns exactly 0; truth cost 6.2e-22; perturbed Y_0 (1e-6 N), min over 20: 1.7e-5 (>= 1e-6) |
| C09-C12 | CONFIRMED | truth <= 1e-15; 1e4-sample property run (Sec. 2): base <= 2.8e-13 (<= 1e-12); P4 1.6e-12, RR 2.2e-11 (<= 1e-10) |
| C13 | CONFIRMED | own residuals at z_truth + 1e-3 N, min over 20 draws of each family max: 1.12e-5 (RB1, base), 1.28e-5 (RB1, aniso); > 1e-7 |
| C14 | CONFIRMED | sizes exactly as stated; every family full rank (own vech, tol 1e-9); primary 106 / 106 (aniso 118 / 118); joint 472 -> 453 (aniso 490 -> 471) |
| C15 | CONFIRMED | own row-handedness matrices built entrywise from eps: each matches a src P2 matrix, mismatch 0 (all 3 KF) |
| C16 | CONFIRMED | rank 11 for each KF; the trace identity is exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric (base and aniso); homogeneity <= 3.5e-17; exactly one b != 0, b = 1 |
| C18 | CONFIRMED | Sec. 2: 200/200 converged, consistency 2.0e-13, all det +1. Drop P2: 87 of 177 blocks det -1. Drop P3: 4.7e2. Drop P1: \|det\| 2.2e-73 |
| C19 | CONFIRMED | 300 starts: 241 SO(3), 59 zero, 0 other |
| C20 | CONFIRMED | violated: P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied <= 3.6e-16 |
| C21 | CONFIRMED | P1 = RA1 = 1.0 (both bbar); all others <= 2.9e-16 |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: P3 9.6e-2, RL 8.4e-2, RH 9.0e-2, RS 1.3e-2; satisfied <= 3.6e-16 |
| C23 | CONFIRMED | residuals of -z equal those of z exactly (base and aniso, both bbar); cost difference 0.0 (dense and factored); decode(-z) == decode(z) exactly, both project values |
| C24 | CONFIRMED | 1e4 lifted states per (bbar, layout) vs own 5x5 cost: dense <= 2.0e-15, factored <= 4.3e-16; aniso factored vs own rho cost 3.8e-16 (<= 1e-12) |
| C25 | CONFIRMED | iso / aniso / partial at both bbar: Q exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0 exactly; bias block equals own rw matrix exactly |
| C26 | CONFIRMED | truth factored 6.2e-22 (<= 1e-15); own GTSAM graph, 50 states per eps: 1.31e-4, 4.71e-6, 3.37e-7 (bounds 5e-4, 5e-5, 5e-6) |
| C27 | CONFIRMED | deviations at lam = 1: rot / vel / pos +2.011e-2 / -1.223e-2 / -2.098e-3 (<= 3e-2). Halving factors: rot 0.250 / 0.249 / 0.249; vel 0.501 / 0.500 / 0.500; pos 0.504 / 0.502 / 0.501 |
| C28 | CONFIRMED | own Cayley vs scipy expm on own Ubar: 2.088e-7, 2.605e-8, 3.253e-9, 4.064e-10; factors 8.02 / 8.01 / 8.00 |
| C29 | CONFIRMED | half 2.2e-7 (<= 5e-6); full 0.81 (>= 0.1) |
| C30 | CONFIRMED | src whitened residual, own local parametrisation (scipy expm), h = 1e-6: rank 45 / 45, cond 3.946e7 / 4.527e7 |
| C31 | CONFIRMED | own encoder, 200 random R_0: 3I constant 4.020000 (ptp 6.1e-13); diag(1,2,5) range [1.340, 6.596] (ptp 5.26 > 1.0) |
| C32 | CONFIRMED | own sampler (rng 2), own encoder, numpy lstsq on 2278 monomials: 2.6e-14, 2.5e-14 (<= 1e-12); diag(1,2,5) 7.9e-2; random SPD 1.5e-1; diag(1,1,1.01) 8.8e-4; rank 2015 |
| C33 | CONFIRMED | partial vs own GTSAM graph: 1.26e-4, 2.05e-6, 2.30e-7 (bounds 0.05 eps); D min eig 3.460e7 >= 0; G2 correlation -0.867 |
| C34 | CONFIRMED | body 2.1e-3, 2.5e-4, 2.4e-5 (<= eps); world 6.1e-2, 5.1e-2, 5.6e-2 (>= 1e-2) |
| C35 | CONFIRMED (caveat) | rng 4, own encoder: rank 2015, null 263, gap 1.7e-7 -> 4.5e-16. Spans 60 / 92 / 106 / 136 / 226 / 236. Projection residual: **2.7e-11** with exactly orthogonal rotations (Rodrigues or SVD-projected), within 1e-10; but **7.4e-10 (RA1)** with scipy-expm rotations (orthogonal to 2.8e-13, gap edge 4.3e-15). See M4 risk R5 |
| C39 | CONFIRMED | Dt 0.5000000000000003 for both edges; bbar = 0 taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6 (own = src); rw diagonal 5.5402e9 / 2.2222e5 |
| **C45** | **REFUTED** | below |

**M3-C45 (brief only): REFUTED.** With tau = 3 / tr(Sigma), tau is the reciprocal of the arithmetic mean of the
eigenvalues lambda_i of Sigma.

* tr(tau I - Sigma^-1) = 3 / AM(lambda) - 3 / HM(lambda) <= 0 by AM >= HM, with equality iff Sigma = cI. So
  tau I - Sigma^-1 can never be PSD unless Sigma is isotropic, and then it is 0.
* Its largest eigenvalue tau - 1/lambda_max is > 0 for every non-isotropic Sigma, so the reverse bound fails too.
  The isotropic cost is **neither** an upper nor a lower bound on r^T Sigma^-1 r.

Measurements:

* **Exact counterexample** (mpmath, 50 digits): Sigma = diag(1, 2, 5), tau = 0.375, eig(tau I - Sigma^-1) =
  (-0.625, -0.125, +0.175). For r = e1 the iso cost is 0.375 < aniso 1.0; for r = e3 the iso cost is 0.375 > aniso
  0.2.
* **1e4 random SPD Sigma**: tau I - Sigma^-1 >= 0 in **0 / 10000**, indefinite in 10000 / 10000 (max lambda_min
  -5.0e-2, min lambda_max +3.0e-2). It is 0 exactly for Sigma = cI.
* **On the M3 instance** (own PIMs, bbar = 0), eig(tau I - Sigma^-1) is indefinite on every block:
  * edge 0: rot (-2.6, -2.6, +5.2), tau_R 6.92e7; pos (-1.08e5, +5.23e4, +5.24e4); vel (-1.95e4, +9.19e3, +9.20e3);
  * edge 1: the same pattern.
* **blkdiag(tau_v I, tau_p I) - W_vp** has eigenvalues in [-2.0e7, +4.4e5].
* Over 1e4 random residuals, the frozen iso / anisotropic cost ratio ranges over [0.131, 1.64] (edge 0) and
  [0.133, 1.73] (edge 1), and the iso cost is *smaller* for 99.5% of them.

**Certifying the isotropic frozen-weight problem therefore gives no bound, conservative or otherwise, on the
anisotropic one.** It certifies only the frozen-weight problem itself (CLAUDE.md guardrail).

## 4. Mutation sweep

### 4.1 New mutants (217; ids `p6-*`)

By file: lifting 47, constraints 41, instance 37, cost 36, gtsam_boundary 25, fixture 17, forms 11, `__init__` 3.

**Killed (182), grouped by first failing test:**

* **test_c01_index_map[true] (39)**
  * lifting: n-else6, matform-T, v-range, bias-stride, rhov-stride, rhop-off, Yp-off, o-stride, biasidx,
    vecform-neg, sform-2, lift-cvcp, lift-bg-ba, lift-Yp-M, lift-Pp-M, lift-bj, lift-dRT, z-s-s, z-ret-abs,
    z-rhov-M, z-Y-C, z-R-C, z-b-zero, z-rho-Rj, z-rhop-rv, dec-s-abs, dec-notproj, dec-R-C, dec-b-mul, dec-v-abs.
  * cost (through the aniso rho vs the test's residual): rv-halfg, rp-vj, **rp-gminus** (the CLAUDE.md position-sign
    footgun), rv-Ri-cv, rp-dv, rv-VVp-minus, cv-bbar-ba.
  * forms: cmat-CT. Instance: Jrho-rows.
* **test_instance_matches_independent_gtsam_reference[true] (38)**
  * gtsam_boundary: to-swap, gal3-pv, gyro-nosq, int-sg, int-argswap, int-half-dt, gl-p-plus, gl-t-plus,
    inc-noinv, inc-right, J-leftchart, J-noh2, navrows-swap.
  * instance: tau-1, S2-no3, Svp-swap, dv-dp, J-at0, Dt-TE, rw-ba, rw-nodt, prior-kf1, sig-swap.
  * fixture: BG-digit, F1-digit, rv0, sprior, sgnss, sint, steps, hat-entry, m1-sign, m2-sign, true-gminus,
    true-vTj, streams-minus, acc-BG, bias-order.
* **test_c09 (20):**
  * constraints: P0-2, cyclic-anti, P1-plus, P0-b0, P2-plus, cv-Jrho, M-dRT, w-ba, P3-WT, bf-j, P3-V-cp, P3-P-noT,
    resid-plusb.
  * forms: eps-sym, matmul-AT, matvec-AT, cross-sign, cvec-CT, hat-pos, linT-noop.
* **test_cost_terms_default_variant_is_iso (14):** gam-neg, gam-ba, rR-swap-sign, rR-YY, rot-full, tauv-taup,
  prior-R-full, prior-R-C, prior-v-plus, gnss-nosq, rw-short, rw-sign, Q-noW, default-aniso.
* **test_cached_instance_arrays_are_read_only (12):** fz-tuple-shallow, fz-skip-{gnss, prior, truth, rw, g, btrue},
  fz-edge-skip-{cov, Wvp, J}, fz-dict-keys-only, nocache.
* **test_c11 (9):** RG-VP-plus, RL-P-cv, RL-Y-WA, RG-noWT, RH-eahat-neg, RH-P-cv, RH-Ycol-row, RS-AT,
  RS-YAT-swap.
* **test_decoder_s_guard_threshold_tie_and_scale (5):** sfloor-11, sfloor-13, dec-lt, dec-floor-1e-200,
  dec-zmax-mean.
* **test_decoder_projection_is_nearest_rotation (4):** proj-D-first, proj-svdT, proj-detU, proj-nosign-abs.
* **Three each:**
  * test_c10: RB2-sign, RB1-YT, RC1-P-cv.
  * test_c12: P4-Rj, P4-rhop-rv, RR-plus.
  * test_c14: RH-drop-a2, vech-k1, emit-skip0.
  * test_c33: partial-D-WR, partial-G2-kron, in-D-noh.
  * test_whitened_residual: wh-UT, wh-abs, wh-nosqrt.
  * gravity test: guard-le, guard-fin-any, guard-nox.
  * test_package_exports: init-drop-truth, init-drop-build, init-ANISO-BASE.
* **Two each:**
  * test_state_copy_is_independent: copy-R, copy-p.
  * test_truth_state_blocks_are_independent_arrays: ts-b-alias, ts-p-alias.
  * test_P1_RA1: upper-strict, RA1-is-P1.
  * test_c04_instance_J: J-h1e3, J-fwd.
* **One each:**
  * test_decoder_rejects_non_finite_z: dec-isnan.
  * test_decoder_degenerate_inputs: dec-bad-gt1.
  * test_decoder_s_guard_counts_aniso_rho_entries: dec-zmax-base.
  * test_c13: fmax-signed.
  * test_c08: vecrows-C.
  * test_c24[aniso]: rho-swap.
  * test_cost_terms_rejects_unknown_variant: unknown-iso.
  * test_c34: ignore-Wvp.
  * test_c25: Q-nosym.
  * test_bias_boundary_converters: from-swap.
  * test_form_helpers: sym-noop.
  * test_instance[zero]: in-zero-true.

### 4.2 Replays

* The 3 pass-5 survivors and the 4 freeze mutants: 7/7 killed (Sec. 1).
* `tests/test_mutation_m3.py`: **84 passed** (116 s). Its 82 mutants are each killed by their named killers, plus the
  well-formedness test and the clean control.

### 4.3 Complete survivor list (35) with rulings

Measured effect = the 750-group dump diff of the mutant tree against clean 75c7908, plus targeted probes where the
dump has no input that reaches the change. As in passes 3-5, a change in a claim-free contract (an output of a src
function on a valid input, or a side effect on a caller's data) is non-equivalent unless a proof shows that no such
output changes.

| # | mutant | measured effect | ruling | class |
|---|---|---|---|---|
| 1 | **p6-lf-dec-asarray-noflt** (`np.asarray(z, dtype=float)` -> `np.asarray(z)`) | 0 groups. Probe: a float32 z decodes to **float32** v / p / b, and project=True gives orthogonality **3.8e-8** (clean: float64, 8.9e-16); a float16 z raises **TypeError** (clean decodes, 1.8e-15). int64: unchanged | **NON-EQUIVALENT** | low: the decoder's float64 promotion is untested. A float32 SDP export would decode silently 7 orders less accurate |
| 2 | **p6-cn-vech-tril** (`np.triu_indices` -> `np.tril_indices`) | 8 groups: `vech` order changes; non-symmetric inputs give different values | **NON-EQUIVALENT** | low: vech's docstring promises "upper-triangular entries ... row-major", and no test pins the order (C14 / C16 use only ranks, which are invariant under a fixed permutation). An M4 svec builder pairing vech with `triu_indices` would be silently wrong |
| 3 | **p6-gb-nav-t1** (`nav_to_gal3(ns, t=0.0)` default -> `1.0`) | 0 groups (the only src caller passes t explicitly). Probe: `nav_to_gal3(NavState())` returns time 1.0 | **NON-EQUIVALENT** | very low: the public NavState lift's default time 0 (the C05 domain convention) is untested |
| 4 | **p6-in-bbar-nocopy** (`np.asarray(bbar, dtype=float).copy()` -> without `.copy()`) | 2 groups: `build_instance(user_array)` **freezes the caller's own array in place** (writeable True -> False) and aliases it; `instance("true")` freezes F.BIAS_TRUE | **NON-EQUIVALENT** | low: the freeze made this copy load-bearing. Without it, an M4 loop that refills one bbar buffer and calls build_instance would raise on the second fill. No test calls build_instance with an array it owns and then checks it |
| 5 | p6-cn-mk-nosym (drops `Q = 0.5 * (Q + Q.T)` in `_mk`) | 0 groups | EQUIVALENT | proof (rule 3): every Q reaching `_mk` is exactly symmetric. P0's Q is diagonal; every forms product ends in `sym`, which is exactly symmetric because fl(a+b) = fl(b+a); and sums / differences of exactly symmetric arrays stay exactly symmetric. 0.5 (Q + Q) = Q bitwise |
| 6 | p6-cn-csr-T (`csr_matrix(Q)` -> `csr_matrix(Q.T)`) | 0 groups | EQUIVALENT | proof: Q was just symmetrised exactly, so Q.T == Q bitwise |
| 7 | p6-cn-P2-rows (P2 built from rows) | 36 groups (order only): outside P2 bitwise identical; per KF, the multiset of P2 residuals identical (difference 0) | EQUIVALENT | proof: C15 (row forms = column forms as matrices). The constraint *set*, and hence the feasible set and any relaxation, is unchanged; only the intra-family order changes, and no claim fixes it. Documented equivalent in test_mutation_m3.py |
| 8 | p6-cn-RR-dot-swap (`dot(rhov, rhop) - dot(rv, rp)` -> both args swapped) | 0 groups | EQUIVALENT | proof: dot(a, b) = sym(a^T b); sym(X^T) = sym(X) bitwise |
| 9 | p6-fm-dot-swap (`"an,am->nm"` -> `"am,an->nm"`) | 0 groups | EQUIVALENT | same proof (the einsum yields the transpose; sym removes it bitwise) |
| 10 | p6-fm-outer-swap (`f g^T` -> `g f^T` inside sym) | 0 groups | EQUIVALENT | same proof |
| 11 | p6-cn-flat-rev (`flat` iterates the family names reversed) | 36 groups (the order of the flat list only) | EQUIVALENT | proof: the same constraint set in a different order; the feasible set, the ranks and every per-family result are unchanged; `flat` documents no order |
| 12 | p6-cn-aniso-noP4 (the P4 key is inserted after RR) | 24 groups (dict / flat order only; sorted family_max identical) | EQUIVALENT | same proof: each family's content is identical; only the key order of the aniso dict changes |
| 13 | p6-co-gnss-unsorted (`sorted(inst.gnss.items())` -> `.items()`) | 0 groups | EQUIVALENT | proof: build_instance inserts keys (1, 2) in order, so the iteration is identical |
| 14 | p6-in-gnss-order (gnss built over (2, 1)) | 2 groups (only my dump's key-iteration order) | EQUIVALENT | proof: the only consumer (cost.terms) sorts |
| 15 | p6-co-fac-abs (`abs(r W r)`) | 0 groups | EQUIVALENT | proof: every W is PSD, so r^T W r >= 0 up to a rounding of <= 3 eps \|W\| \|r\|^2; rule 4, 0 measured |
| 16 | p6-co-fac-WT (`r W^T r`) | 1 group: truth cost 9.2e-22 changes by 1.9e-37 (2.0e-16 relative) | EQUIVALENT | rule 4: <= 1 ulp of the value; C24's 1e-12 relative is 5e15x away |
| 17 | p6-co-Wvp-inv (`inv(inv(W_vp))`) | 11 groups: W 2.7e-18 / Q 1.3e-18 relative to max; costs <= 3.9e-16 relative | EQUIVALENT | rule 1: <= 1/10 of C24 (1e-12) and C34 (eps) |
| 18 | p6-co-aniso-noassert (the `assert layout.aniso` in terms -> `pass`) | 60 groups: the AssertionError message only (`Layout.rho_v`'s own `assert` still fires, same exception type); under `python -O` clean and mutant are identical | EQUIVALENT | the same exception at the same call. It exposes M3-F10 (below) |
| 19 | p6-gb-guard-fin-z (`isfinite(g_w).all()` -> `isfinite(g_w[2])`) | 0 groups | EQUIVALENT | proof: the remaining clauses require shape (3,) and g_x = g_y = 0.0 (finite), so the two are equivalent |
| 20 | p6-gb-g-abs (`-g_w[2]` -> `abs(g_w[2])`) | 0 groups | EQUIVALENT | proof: the guard requires g_w[2] < 0 |
| 21 | p6-gb-J-h005 (FD step 0.1 -> 0.05) | 179 groups: J moves 1.3e-14 to 1.8e-14 (its error vs the logm-fit reference goes from 5.1e-15 to 9.6e-15, to 9.0e-15 to 1.6e-14); Q <= 8.4e-20 relative; truth-cost absolute changes ~4e-24 | EQUIVALENT | rule 1: the claim tolerance on J is C04's 1e-8 (J_nu,a = J_R); downstream, C24 is 1e-12 relative and C26 truth is 1e-15 absolute, all >= 1e5x away. The test pin (5e-14) still holds; h = 1e-3 and the forward difference are killed |
| 22 | p6-gb-cov-T (`C[b, a].T`) | 46 groups: W_vp / Q <= 1.3e-19 relative to max; whitened residuals <= 1.2e-16 relative; taus unchanged | EQUIVALENT | rule 1 (as pass-5 #4) |
| 23 | p6-gb-cov-sym (symmetrise preintMeasCov) | 46 groups: W <= 6.4e-20, Q <= 4.9e-21 relative to max; whitened <= 1.8e-16 | EQUIVALENT | rule 1, same reasoning |
| 24 | p6-in-S2-offT (the trace of C_pv in place of C_vp) | 0 groups | EQUIVALENT | measured bitwise identical (the two traces agree in floating point here); rule 1 trivially |
| 25 | p6-in-WR-pinv (`inv` -> `pinv` for W_R) | 30 groups: W_R 8.3e-17 relative; partial cost <= 4.2e-16 relative; whitened <= 1.0e-15 relative | EQUIVALENT | rule 1 vs C33 (0.05 eps >= 5e-6) and C24 (1e-12) |
| 26 | p6-in-prior-alias (prior = truth[0] without copies) | aliasing vector only | EQUIVALENT | proof: both arrays are read-only, so the aliasing cannot be observed through any write; values identical |
| 27 | p6-in-gnss-alias | aliasing vector only | EQUIVALENT | same proof |
| 28 | p6-in-g-nocopy (`g = F.G_W`) | aliasing; F.G_W becomes read-only after the first build | EQUIVALENT | proof: no src function writes a fixture constant; the only new behaviour is that such a write, itself a defect that would poison every later build, now raises. No src output changes |
| 29 | p6-in-btrue-alias (`b_true = F.BIAS_TRUE`) | aliasing; F.BIAS_TRUE read-only | EQUIVALENT | same proof (pass-5 #6) |
| 30 | p6-fx-truth-V0-alias (true_states without V_0 / P_0 copies) | aliasing; F.V_0 / F.P_0 read-only | EQUIVALENT | same proof (pass-5 #8) |
| 31 | p6-in-fz-include-pims (`_freeze` also applied to pims) | 0 groups | EQUIVALENT | proof: `_freeze` returns non-array objects unchanged and rebuilds the tuple |
| 32 | p6-in-fz-copy-instead (copy each array before freezing) | 0 groups | EQUIVALENT | values identical; the only difference is that the cov blocks stop sharing a writable base, which is *safer* (see M3-F09 (b)) |
| 33 | p6-in-cache-1 (`maxsize=None` -> `1`) | 0 groups | EQUIVALENT | proof (pass-5 #5): build_instance is deterministic, and no src code depends on identity |
| 34 | p6-in-Jnu-cols (`J[3:6, :]` -> `J[3:6, 0:6]`) | 0 groups | EQUIVALENT | an identical slice |
| 35 | p6-fx-exp-thresh (so3_exp small-angle threshold 1e-8 -> 1e-6) | 0 groups | EQUIVALENT | proof: so3_exp is called only with ROT_VEC_0 (\|w\| = 0.374) and w T (\|w T\| >= 0.2); the branch is unreachable |

Totals: 35 survivors. **Non-equivalent 4 (#1-#4), all low. Equivalent 31 (#5-#35).**

## 5. Tests that cannot fail or compare code with itself (at 75c7908)

1. test_project_so3_det_correction_is_exact: still a same-LAPACK pin, now with honest loose sanity bounds (Sec. 1).
   Accepted.
2. test_cached_instance_arrays_are_read_only covers only the cached `instance()` path. The `build_instance` caller
   path and the `.base` path are untested (survivor #4, M3-F09).
3. Unchanged from pass 5:
   * the tie-precondition arithmetic in test_decoder_s_guard_threshold_tie_and_scale;
   * the `fit <= 1e-14` reference check in test_c04_instance_J (the src check is the 5e-14 pin);
   * the test_c02 / test_c03 arithmetic;
   * the test-side-only parts of C15, C26, C33, C39 and C04;
   * the MATH CHECK labels on C05 / C06 / C28;
   * the scale invariance of C30's rank.
4. **Untested surface (all of it shows up as survivors):**
   * the decoder's float64 promotion;
   * vech's documented order;
   * nav_to_gal3's default t;
   * build_instance's copy of its argument.

## 6. Findings and risks for M4

* **M3-F08 (new, gate-blocking, low): 4 non-equivalent survivors (Sec. 4.3 #1-#4).** Tests that would kill them:
  * (a) `state_of_z(z.astype(np.float32), project=True)`: assert every block is float64 and the projection is
    orthogonal to <= 1e-14; and `z.astype(np.float16)` decodes;
  * (b) `vech(A)` for a non-symmetric 4x4 `A = arange(16).reshape(4, 4)` equals `[0, 1, 2, 3, 5, 6, 7, 10, 11, 15]`
    (or `A[np.triu_indices(4)]` written out);
  * (c) `nav_to_gal3(gtsam.NavState()).matrix()[3, 4] == 0.0`;
  * (d) `b = np.zeros(6); I = build_instance(b)`: assert `b.flags.writeable`, `not np.shares_memory(b, I.bbar)`, and
    `b[0] = 1.0` leaves `I.bbar[0] == 0`.
* **M3-F09 (new, not gate-blocking, low): freeze side effects.**
  * (a) **Regression:** Instance is no longer picklable / deepcopy-able / `asdict`-able
    (`TypeError: cannot pickle 'mappingproxy' object`), where 519864f pickled to 16305 bytes. M4 process-pool
    solves, on-disk caching and deepcopy-based perturbation will hit this. Suggested fix: a small read-only dict
    subclass with `__reduce__`, or `copyreg.pickle(types.MappingProxyType, ...)`, plus a pickle round-trip test.
  * (b) **Writable path:** every `cov` block is a view whose `.base` is the writable 9x9 preintMeasCov copy. The write
    `ed.cov[k].base[...] = x` succeeds and changes all 9 cached blocks. Copying each block before freezing (my
    equivalent mutant #32) closes it.
  * (c) The GTSAM PIMs stay mutable (acknowledged); nothing in src reads them after the build.
* **M3-F10 (new, not gate-blocking, low): the aniso-layout guard is an `assert`.** Under `python -O`,
  `terms(inst, BASE, "aniso")` returns two `imu*.rho` terms with an **empty (0 x 124) F**, which silently drops the
  velocity / position IMU cost. It fails only later, with an opaque matmul shape error in assemble_Q /
  factored_cost. `Layout.rho_v` / `rho_p` rely on `assert` the same way. This is the same pattern F06 fixed in
  make_params; raise a ValueError instead.
* **M4 risks:**
  * **R1, dense-Q cancellation persists.** At truth, z^T Q z = 1.2e-8 vs factored 6.7e-22 (max\|Q\| ~ 1.1e10). Use
    the factored cost for primal values and gaps, and rescale before any Clarabel solve. Barfoot's
    sigma_1/sigma_2 >= 1e5 test cannot be resolved at this conditioning.
  * **R2, handedness is load-bearing.** Dropping P2 admits consistently lifted det -1 points (87/177 blocks; 57/120
    even with every other redundant family except RH / RS). Any constraint pruning must keep P2 or RH / RS.
  * **R3, redundant families are linearly dependent** (472 -> 453; aniso 490 -> 471). Remove the dependencies before
    handing them to Clarabel.
  * **R4, isotropic weights are not conservative (C45).** The frozen iso cost is neither an upper nor a lower bound
    on the anisotropic Mahalanobis cost (ratio range [0.13, 1.73] on this instance). A certificate for the iso
    problem certifies the iso problem only; no bound transfers.
  * **R5, C35's 1e-10 tolerance is sampler-fragile.** It holds with 3.7x headroom (2.7e-11) only for rotations
    orthogonal to ~1e-15. Rotations from scipy `expm` at \|theta\| up to ~7 rad (orthogonal to 2.8e-13) give 7.4e-10.
    Any AutoTight rerun on SDP-derived or solver-produced samples must re-orthogonalise first or widen the tolerance.
  * **R6, pickling (M3-F09 a)** blocks process-level parallelism over instances until it is fixed.
  * **R7, the surrogate gap at bbar = 0** (truth cost 3.3e-3 vs ~1e-21 at bbar = b_true) lies outside C26's domain
    and still applies.
  * **R8, results are not bit-comparable.** No MOSEK is used, so results will not be bit-comparable with Rai's
    published tables; say so wherever they are compared.
