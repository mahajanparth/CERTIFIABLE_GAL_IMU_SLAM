# Stage B verification, pass 7: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 7 (agent verify-M3-007). Base commit daace65, the implementer's response to the
pass-6 findings M3-F08 (blocking), M3-F09 and M3-F10. Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py,
docs/claims/M3.yaml (M3-C01..C35, C39), plus one claim given only in the brief, M3-C46. I did not read
docs/03_qcqp_derivation.md, experiments/m3_scratch/ or any seeds file. I did not edit src/, tests/ or docs/claims/.

Every Python run used `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` under a hard `timeout`. My scripts lived
in an untracked `_v7scratch/` inside the worktree (deleted before the commit). They were:

* `ref7.py`: a new independent reference. It imports neither src nor tests. It contains:
  * its own retyped literals, its own GTSAM PIMs and its own bias-order conversion;
  * Ubar as a 5x5 matrix from `pim.predict(NavState(), b)` and its own G (the CLAUDE.md corrected endpoint);
  * J as a 200-sample least-squares fit of the **scipy `logm` chart** of `Ubar^-1 U(bbar + d)` (fit residual 1.0e-15 to
    1.7e-15);
  * the ground truth from a **Van Loan block exponential** `expm([[W, f, 0], [0, 0, 1], [0, 0, 0]] T)` with Rodrigues R_0;
  * its own encoder (column-major via `R.T.reshape`), its own decoder and its own SVD projection (det sign from
    `det U * det V^T`).

  **Re-implemented core routines:** every constraint family as matrix equations on the blocks of an *arbitrary* z (in
  src's emission order, so the comparison is entry by entry); the C07 residual forms on arbitrary z; the cost on
  arbitrary z (iso / partial / aniso); and the cost of a state from 5x5 cleared Gal(3) residuals.
* `claimsB.py` has its own GTSAM factor graph: GalileanImuFactor on its own PIMs with `Gaussian.Information` weights,
  PriorFactorPose3 through `Diagonal.Precisions`, PriorFactorVector / GPSFactor through `Isotropic.Sigma`, and
  BetweenFactorConstantBias with precisions in GTSAM's (accel, gyro) order; cost = 2 graph.error.
* `cmp7.py`, `prop7.py`, `hunt7.py`, `hunt7g.py`, `claimsA.py`, `claimsD.py`, `c07.py`, `c46.py`, `c24meas.py`,
  `probe_fd.py`: the src-vs-reference comparisons, the 1e4-sample property tests, the spurious-point hunt, the
  per-claim computations, C46, the C24-budget measurement of survivors, and the FrozenDict probes.
* `dump7.py`, `cmpdump.py`, `measure7.py`: record **1383 observable output groups** of src/qcqp and diff two trees.
  The groups cover:
  * every Instance / EdgeData field (both bbar), cov blocks and keys, the D / J_R / J_nu / J_rho properties;
  * z at 6 states x 2 layouts x 2 scales x 2 bbar;
  * every constraint family (names, families, b, a hash of every A_k, the sparse class, residuals on 4 z, vech);
  * flat (full and named subset), family_residuals and family_max;
  * terms / Q / factored cost / whitened residual for iso, partial, aniso, aniso with 3 caller weights (including a
    singular one and a non-symmetric one), iso / partial on the aniso layout, an unknown variant, aniso on BASE, and
    iso with a stray W_vp;
  * about 110 decoder probes x 2 project values: scales 1e-320 to 1e305, threshold ties at 4 magnitudes, NaN / inf at
    3 positions, aniso rho entries, zero, ones, a reflected R_0, a list, float32 / float16 / longdouble / int / bool /
    complex / object dtypes, wrong shapes, and 20 random z;
  * 38 `project_so3` inputs, including singular and extreme-scale ones;
  * the layout (BASE, ANISO and a 4-KF / 3-edge layout), vech on 5 inputs, every forms helper, the boundary helpers,
    19 gravity inputs, the fixture literals and functions, and the exports;
  * **writeability and serialisation vectors:** the writable flag along every array's whole `.base` chain, the dict
    types, item and attribute assignment, the caller's bbar, the fixture constants, aliasing, the cache, `truth_state`,
    `Term.W`, and copy / deepcopy / pickle (protocols 2, 4, 5) of the Instance, of one edge and of one FrozenDict,
    `asdict`, `astuple` and `replace`.

  Two dumps of the same tree are identical (0 of 1383 groups differ).
* `harness7.py`, `mutants7.py`, `replay7.py`: 204 new mutants plus 15 replays. Each mutant is a real exactly-once
  textual edit on a tempfile copy of src + tests/test_qcqp.py + conftest + pyproject + fixtures. The pattern count is
  checked to be exactly 1 before the edit, and `PYTHONPATH` is stripped so the copy cannot import the worktree's src.
  The judge is the **full** tests/test_qcqp.py, slow tests included, run with `-x` (single-threaded BLAS, 8-10
  workers, 900 s timeout per mutant; no mutant timed out). The unmutated control copy passes (73.7 s).

## Verdict summary

* **Clean suite: 82/82 pass (77 s, single-threaded).** `tests/test_mutation_m3.py`: **99/99 pass (148 s)**. That is
  97 mutants, each killed by its named killers, plus the well-formedness test and the clean control.
* **M3-F08 (pass 6): FIXED.** The 4 pass-6 survivors, replayed as real edits and judged by the full suite, are all
  killed (Sec. 1).
* **M3-F09: FIXED as specified, with residual low items.**
  * (a) pickle / deepcopy / asdict work again. Protocols 2-5 round-trip the Instance with every one of the 50 arrays
    bitwise equal, all read-only along the whole base chain, and the PIMs intact.
  * (b) No array reachable from an Instance has a writable `.base`.
  * New low items go to M3-F12: `FrozenDict._d` is a plain mutable dict, a FrozenDict round-tripped on its own comes
    back with writable arrays, and protocol 0/1 pickling of an Instance aborts the interpreter inside GTSAM.
* **M3-F10: FIXED.** All three guards raise ValueError under `python -O` (measured in a subprocess with
  `sys.flags.optimize = 1`).
* **No claim-level output changed.** 1343 of 1383 groups are bitwise identical between 9ce5819 and daace65. The 40 that
  differ are exactly the intended changes (Sec. 1).
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C46 REFUTED** (the true relation is theta = 2 arctan(|phi|/2)). 0
  UNTESTABLE.
* **Fresh sweep: 204 new mutants over all 8 files of src/qcqp, plus 15 replays.** 194 are killed (179 new + 15 replays),
  and 25 new mutants survive. **20 survivors are equivalent and 5 are NON-EQUIVALENT (all low).**
* **GATE (zero surviving non-equivalent mutants): FAILED** (new finding M3-F11, Sec. 6). Four survivors sit in the new
  `FrozenDict`, whose own mapping contract is tested only through the Instance path:
  * a wrong `len`;
  * a missing key returns None instead of raising KeyError, so `0 in I.gnss` is True;
  * the constructor aliases its input dict;
  * a standalone copy / pickle becomes a plain mutable dict.

  The fifth: `increment` builds G from the nominal Dt = 0.5 instead of `pim.deltaTij()`. That shifts dv / dp by
  3.55e-15 (the test pin is 1e-13). At the bbar = 0 truth state, it moves the C24 deviation from 1.1e-13 to 2.6e-13
  (iso) and from 3.4e-15 to 3.6e-13 (aniso), against C24's 1e-12. That fails rules 1, 2 and 4.

## 1. Rulings on M3-F08, M3-F09, M3-F10

### Replays (real exactly-once edits on daace65, full suite, `-x`)

| mutant | result: first failing test |
|---|---|
| control (unmutated) | passes, 73.7 s |
| p6-lf-dec-asarray-noflt | killed by test_decoder_promotes_to_float64 |
| p6-cn-vech-tril | killed by test_vech_order_is_row_major_upper_triangle |
| p6-gb-nav-t1 | killed by test_nav_to_gal3_default_time_is_zero |
| p6-in-bbar-nocopy | killed by test_build_instance_does_not_freeze_or_alias_callers_bbar |
| p8-in-fz-view-nocopy | killed by test_instance_arrays_have_no_writable_base |
| p8-in-reduce-dropped, p8-fd-reduce-dropped, p8-in-rebuild-nofreeze, p8-in-rebuild-edge-nofreeze | killed by test_instance_pickle_and_deepcopy_round_trip (4/4) |
| p8-co-aniso-guard-pass / -assert, p8-lf-rho_v-guard-pass / -assert, p8-lf-rho_p-guard-pass / -assert | killed by test_aniso_guards_raise_value_error (6/6, in-process; the -O test also names them) |

**15/15 killed. M3-F08 FIXED.**

### No claim-level output changed (9ce5819 vs daace65, 1383 groups)

1343 groups are bitwise identical. The 40 that differ are:

* `Layout.rho_v` / `rho_p` on a non-aniso layout (7 groups) and `terms(..., BASE, "aniso")` (2 groups): AssertionError
  became ValueError with the new messages;
* serialisation (19 groups):
  * pickle (protocols 2 / 4 / 5), deepcopy, asdict and astuple of the Instance, an edge and a dict were
    `TypeError: cannot pickle 'mappingproxy'` and now work;
  * `copy.copy` and `replace` now show no writable base;
  * the dict repr changed;
* writeability (12 groups): the cov blocks' base chain went from `(False, True)` to `(False,)` (no base); the dict
  types went from mappingproxy to FrozenDict, with the same TypeError on item assignment.

Every instance value, z, constraint matrix, residual, cost, Q, whitened residual, decoder output, projection,
boundary helper, fixture value and export is unchanged. The decoder's float64 promotion was already present at 9ce5819;
F08 added only the test.

### M3-F09: serialisation and writable-path hunt

| path / operation | daace65 |
|---|---|
| `pickle` protocol 2, 3, 4, 5 of `instance("true")` | 50 / 50 arrays bitwise equal (dtype too); **0 writable anywhere in any base chain**; Q identical; sigmas, taus, Dt identical; PIMs intact (deltaTij, preintMeasCov, deltaXij R / p / v and biasHat bitwise equal; GalileanImuFactor error on a perturbed Values identical, which exercises the params and the bias Jacobians) |
| `copy.deepcopy`, `copy.copy`, deepcopy(pickle(deepcopy(I))), pickle of `[I, I]` | 0 writable; object identity within one pickle preserved |
| one `EdgeData` pickled / deep-copied on its own | read-only (re-frozen by `_rebuild`) |
| `dataclasses.asdict(I)` | works; arrays are writable deep copies (by design) |
| any cov block `.base` | none (copied before freezing) |
| **`I.gnss._d`, `ed.cov._d`** | **plain mutable `dict`.** `instance("true").gnss._d[1] = np.zeros(3)` succeeds, silently poisons the lru-cached instance, and changes the `gnss1` cost term. MappingProxyType had no such path (M3-F12 a) |
| a **FrozenDict on its own** (copy / deepcopy / pickle 4 / pickle 5) | stays a FrozenDict (item assignment raises), but its arrays come back **writable** (pickle 4 [True, True]; pickle 5 [True, False]). Only the Instance / EdgeData path re-freezes (M3-F12 b) |
| `pickle` protocol 0 or 1 | **the interpreter aborts (SIGABRT, core dump)**. A bare `gtsam.PreintegratedImuMeasurementsG` does the same, and so does 9ce5819's PIM, so this is GTSAM's pybind11 pickling, not a src regression. The default protocol (4) and multiprocessing are fine (M3-F12 c) |
| `dataclasses.replace(I, bbar=arr)` | returns an unfrozen Instance that holds the caller's array (not cached; by design) |
| `FrozenDict(user_dict)` | copies the dict (later writes to user_dict do not reach it); value arrays are not frozen by the class itself |
| `object.__setattr__(fd, "_d", ...)`, `a.setflags(write=True)` | deliberate bypasses, as accepted in pass 6 |

**Ruling M3-F09: FIXED** for (a) and (b) as specified. The residual items are low, not gate-blocking, and go to M3-F12.

### M3-F10

`terms(inst, BASE, "aniso")`, `BASE.rho_v(0)`, `BASE.rho_p(1)` and a 4-KF non-aniso `rho_p(2)` all raise ValueError
under `python -O` (checked with `sys.flags.optimize = 1`). The in-process test and the -O subprocess test both exist,
and 6 guard mutants (pass / assert / TypeError / inverted) are killed. **FIXED.**

### The new tests and their bindings

* **test_decoder_promotes_to_float64 (C01):** real. It compares float32 / float16 / int64 z bitwise against their
  float64 upcast and requires orthogonality <= 1e-14 after projection. It kills asarray-noflt, the float32 cast and the
  longdouble cast. Binding to C01 is consistent with the decoder-contract binding accepted in passes 3-6. **Accepted.**
* **test_vech_order / test_nav_to_gal3_default_time / test_build_instance_does_not_freeze_or_alias / no-writable-base /
  pickle round trip:** real, and unbound with an explicit `NOT claim-bound` reason. **Honest.**
* **test_aniso_guards_raise_value_error and the -O variant (C24):** acceptable. Without the guard, `-O` silently dropped
  the vel / pos IMU cost, which is exactly a C24 violation (cost != direct), so the guard protects C24's aniso
  statement. Remark: the `rho_v` / `rho_p` half is index-map content (C01: "the anisotropic variant has n = 136 with
  rho ..."), so C01 would be the more precise binding for that half. This does not inflate any claim.
* **Gaps:**
  * FrozenDict's own mapping contract (len, KeyError, input copy, standalone round-trip type) is untested
    (4 survivors).
  * The pickle test does not round-trip a FrozenDict on its own, and does not probe `_d`.
  * The dv / dp pin (1e-13) is 30x looser than the measured src-vs-reference agreement (dv bitwise 0, dp 4.4e-16), so a
    Dt-level defect survives.

## 2. Independent re-checks (the reference imports no src)

| check | result |
|---|---|
| instance vs ref7 PIMs (both bbar, both edges) | dR, dv, dp **0**; all 9 cov blocks, taus, W_R, G2, W_vp **0**; Dt 0.5000000000000003; truth (Van Loan) 3.3e-15; gnss 3.3e-15; rw, g, b_true, bbar 0 |
| J: src (h = 0.1) vs ref7 (logm LS fit) | 6.2e-15 to 1.0e-14 |
| z_of_state vs own encoder, 30 random states x 2 layouts | <= 3.7e-15 (true) / 5.1e-15 (zero) relative |
| **families: src A_k vs ref7 matrix equations on ARBITRARY z (20 z ~ N(0, I)), base and aniso, both bbar** | same family keys and order, same count and order within every family. Max relative difference: P0 0; P1 / P2 / RA1 <= 3.5e-16; RB2 / RS / P4 / RR <= 1.1e-15; P3 / RB1 / RC1 / RG / RL / RH <= 2.5e-14 (driven by the 1e-14 J difference) |
| cost on ARBITRARY z vs own (iso / partial / aniso, both bbar, 50 z each) | dense <= 7.8e-16 relative, factored <= 2.0e-16 relative |
| truth cost (bbar = b_true) | factored 6.17e-22, own 5x5 6.23e-22, own GTSAM graph 6.16e-22; **dense z^T Q z 5.2e-9** (cancellation; R1) |
| truth cost (bbar = 0) | factored 3.2967763857e-3, own 5x5 3.2967763858e-3 (the surrogate gap, R9) |

### Property tests (1e4 samples per configuration; own encoder; src constraint matrices and costs)

The sampler draws R = Rodrigues(2 N(0, I3)) (orthogonal to ~1e-16), v, p ~ N(0, I) and b ~ 0.3 N(0, I6).

| instance | family max on lifted states | C24: cost vs own 5x5 (dense / factored, / max(1, J)) |
|---|---|---|
| true, base | P0 0, P1 2.1e-15, P2 2.0e-15, P3 1.1e-14, RA1 2.2e-15, RB1 7.6e-16, RB2 1.6e-15, RC1 8.3e-15, RG 6.0e-15, RL 1.1e-14, RH 1.1e-14, RS 1.4e-15 | 1.0e-15 / 4.1e-16 |
| true, aniso | as base (<= 1.0e-14); P4 2.0e-14, RR 2.4e-13 | 7.6e-16 / 4.1e-16 |
| zero, base | <= 1.7e-14 | 9.0e-16 / 3.8e-16 |
| zero, aniso | <= 1.5e-14; P4 2.3e-14, RR 2.6e-13 | 8.6e-16 / 4.0e-16 |

These are within C09-C12 (1e-12 base; 1e-10 aniso) with >= 59x headroom, and within C24 (1e-12) with >= 1000x headroom.

**Decoder vs own decoder, 1e4 inputs x 2 project values:**

* The inputs alternate base / aniso, use scales {1, -1, -3.7, 0.05, 1e-6, 1e6}, add noise on every 3rd sample,
  reflect R_0 on every 7th, put s at 1e-12 max|z| (+-1e-9 relative) on every 11th / 13th, put a 1e14 entry (a rho
  entry for aniso) on every 17th, use float32 on every 19th, and put a NaN / +-inf on every 23rd.
* There are **0 raise / no-raise disagreements**; 3052 inputs raised in both decoders.
* The decoded blocks are bitwise equal (max difference 0) and all float64. The projected rotations are orthogonal to
  3.2e-15.

### Spurious-feasible-point hunt

TRF with `xtol = ftol = gtol = 1e-15`, an analytic Jacobian 2 A_k z and src's A_k. Whether a point is lifted is
judged by ref7's decoder and encoder: consistency = max|z - s z(state(z))|.

| family set | converged | consistency | det R_k |
|---|---|---|---|
| primary P0-P3, bbar = 0 (rng 7, C18) | 200/200 | 1.6e-13 | 600 / 600 are +1 |
| primary, bbar = b_true | 60/60 | 6.7e-14 | 180 / 180 +1 |
| aniso primary P0-P4 | 60/60 | 1.8e-13 | 180 / 180 +1 |
| all 12 families, bbar = 0 | 40/40 | 9.2e-15 | all +1 |
| all 14 aniso families | 30/30 | 9.6e-15 | all +1 |
| drop P2 (P0, P1, P3) | 60/60 | 4.1e-14 | **94 of 180 blocks det -1** |
| drop P3 | 60/60 | **4.45e2** | +1 (liftings off their definitions) |
| drop P1 | 60/60 | 1.5e1 | min \|det\| **4.3e-74** |
| **drop P2 only, keep all 8 redundant families (RH, RS included), bbar = 0** | 40/40 | 4.1e-15 | **49 of 120 blocks det -1** |
| same, bbar = b_true | 40/40 | 2.4e-15 | **37 of 120 det -1** |
| same, aniso (+P4, RR) | 30/30 | 7.8e-15 | **36 of 90 det -1** |
| drop P2, keep RH, RS only | 40/40 | 2.1e-14 | 50 of 120 det -1 |

**New observation (corrects pass-6 R2): RH and RS do not substitute for P2.**

* At every det -1 keyframe of these runs, the adjacent edge's w = J_R b_g is zero, to |w| ~ 1e-25 (`hunt7g.py`:
  12 of 12 det -1 solutions inspected).
* RH and RS encode handedness only through det(A) [A w]_x and det(A) A (e_a x E). Both vanish when w = c^v = c^p = 0,
  i.e. at a zero bias of that keyframe.
* Pass 6 R2 said "keep P2 or RH / RS". That is wrong: **P2 is the only unconditional handedness guard.** This does not
  contradict C18 or C20; C20's reflected point has w != 0.
* No spurious point exists for any set that contains P0-P3.

## 3. Claims

Each verdict rests on my own computation, with src as the object under test.

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | n 124 / 136; s 123, z[123] = 1; rho_v(0) = [124,127), rho_p(1) = [133,136); src z vs own encoder 3.6e-16 at truth, <= 5.1e-15 relative over 60 random states x 2 layouts |
| C02 | CONFIRMED | 112, 12; the V' columns of imu{0,1}.vel are exactly -1/2 I3 |
| C03 | CONFIRMED | all 7 triples satisfy n = 39K + 6M + 16; the offset vs the direct count is 39 for each |
| C04 | CONFIRMED | own FD (h = 1e-6, Gal3.Logmap of own 5x5 U) on 4 fixture cases + 6 random time-varying streams with random bbar: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| <= 3.95e-10 (<= 1e-8); \|J_R\| 0.36-0.50. Own right-Jacobian series sum (-ad_x)^n/(n+1)! vs FD of Gal3.Expmap, 200 x at \|x\| = 0.8: 1.45e-10; block differences exactly 0. Shipped J: zeros exact, \|J_nu,a - J_R\| <= 7.6e-15 |
| C05 | CONFIRMED | own algebra, 2000 draws per bbar: max\|r\| 2.2e-15; \|t_j\| 0; bottom rows exactly 0 for arbitrary X_j; arbitrary X_j min max\|r\| 0.89 (> 1e-6); A34 = B34 = 0.5000000000000003 |
| C06 | CONFIRMED | 2000 draws per bbar, \|eps\| = 0.3, eps_alpha = 0: 2.5e-15 |
| C07 | CONFIRMED | src forms on z(state) vs own 5x5 cleared residual, 500 states per bbar (\|b\| ~ 0.2): 3.6e-15 / 4.4e-15 (<= 1e-12) |
| C08 | CONFIRMED | coefficient blocks exact (deviation 0, both bbar), bias columns exactly 0; truth 6.17e-22; perturbed Y_0, min over 20: 2.6e-5 (>= 1e-6) |
| C09-C12 | CONFIRMED | truth: base <= 3.9e-16; aniso P4 <= 6e-16, RR <= 2e-14 (cancellation in the expanded \|r\|^2 form; src's own z gives 1.6e-14). Property run (Sec. 2): base <= 1.7e-14 (<= 1e-12); P4 2.3e-14, RR 2.6e-13 (<= 1e-10) |
| C13 | CONFIRMED | own z_truth + 1e-3 N, min over 20 draws of each family max: smallest 1.52e-5 (RB1, base), 1.07e-5 (RB1, aniso); > 1e-7 |
| C14 | CONFIRMED | sizes exactly as stated; every family full rank (own vech, tol 1e-9); primary 106 / 106 (aniso 118 / 118); joint 472 -> 453 (aniso 490 -> 471) |
| C15 | CONFIRMED | own row-handedness matrices built entrywise from eps: each equals a src P2 matrix, mismatch 0 (all 3 KF) |
| C16 | CONFIRMED | rank 11 for each KF; trace identity exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric (base and aniso); homogeneity <= 1.5e-16; exactly one b != 0, b = 1 |
| C18 | CONFIRMED | 200/200 converged, consistency 1.6e-13, 600 / 600 det +1. Drop P2: 94 / 180 det -1. Drop P3: 4.45e2. Drop P1: \|det\| 4.3e-74 |
| C19 | CONFIRMED | 300 starts: 237 SO(3), 63 zero, 0 other |
| C20 | CONFIRMED | violated: P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied <= 3.9e-16 (both bbar) |
| C21 | CONFIRMED | P1 = RA1 = 1.0 (both bbar); all others <= 3.9e-16 |
| C22 | CONFIRMED | rotated: P3 1.4e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: P3 9.6e-2, RL 8.4e-2, RH 9.0e-2, RS 1.3e-2; satisfied <= 3.3e-16 |
| C23 | CONFIRMED | residuals of -z equal those of z exactly (base / aniso, both bbar); cost difference 0.0 (dense and factored); decode(-z) == decode(z) exactly, both project values |
| C24 | CONFIRMED | 1e4 lifted states per (bbar, layout) vs own 5x5 cost: dense <= 1.0e-15, factored <= 4.1e-16 (<= 1e-12). Also 200 states + truth per variant: worst 1.1e-13 (iso, bbar = 0, truth); see R7 |
| C25 | CONFIRMED | iso / partial / aniso at both bbar: Q exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0 exactly; bias block equals own rw matrix exactly |
| C26 | CONFIRMED | truth factored 6.17e-22 (<= 1e-15); own GTSAM graph, 50 states per eps: 1.15e-4, 4.0e-6, 3.6e-7 (bounds 5e-4, 5e-5, 5e-6) |
| C27 | CONFIRMED | own PIMs / J / 5x5 residual vs GalileanImuFactor.unwhitenedError. Deviations at lam = 1: rot / vel / pos +2.011e-2 / -1.223e-2 / -2.098e-3 (<= 3e-2). Halving factors: rot 0.250 / 0.249 / 0.249; vel 0.501 / 0.500 / 0.500; pos 0.504 / 0.502 / 0.501 |
| C28 | CONFIRMED | own Cayley vs scipy expm on own Ubar: 2.0880e-7, 2.6048e-8, 3.2527e-9, 4.0639e-10; factors 8.02 / 8.01 / 8.00 |
| C29 | CONFIRMED | half 2.7e-7 (<= 5e-6); full 0.86 (>= 0.1) |
| C30 | CONFIRMED | src whitened residual, own local parametrisation (Rodrigues), h = 1e-6: rank 45 / 45; cond 3.946e7 / 4.527e7 |
| C31 | CONFIRMED | own encoder, 200 random R_0: 3I constant 4.020000 (ptp 2.1e-13); diag(1,2,5) range [1.365, 6.694] (ptp 5.33 > 1.0) |
| C32 | CONFIRMED | own sampler (rng 2), own encoder, numpy lstsq on 2278 monomials: 2.4e-15, 2.7e-15 (<= 1e-12); diag(1,2,5) 7.9e-2; random SPD 8.6e-2; diag(1,1,1.01) 8.7e-4; rank 2015 |
| C33 | CONFIRMED | partial vs own GTSAM graph: 1.24e-4, 1.96e-6, 2.33e-7 (bounds 0.05 eps); D eigenvalues >= 3.4602e7; G2 correlation -0.867 |
| C34 | CONFIRMED | body 2.2e-3, 3.4e-4, 1.3e-5 (<= eps); world 5.1e-2, 5.8e-2, 5.6e-2 (>= 1e-2) |
| C35 | CONFIRMED | rng 4, own encoder, Rodrigues rotations: rank 2015, null 263, gap 1.9e-7 -> 4.5e-16. Spans 60 / 92 / 106 / 136 / 226 / 236. Max projection residual 2.4e-11 (<= 1e-10). R5 still applies |
| C39 | CONFIRMED | Dt 0.5000000000000003 for both edges; bbar = 0 taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6 (own = src); rw diagonal 5.5402e9 / 2.2222e5 |
| **C46** | **REFUTED** | below |

**M3-C46 (brief only): REFUTED.** The Cayley map with the /2 scaling rotates by **theta = 2 arctan(|phi| / 2)**, not
2 arctan(|phi|). This is the CLAUDE.md relation |phi| = 2 tan(theta / 2).

* Over 1e4 random phi (|phi| from 1e-3 to 1e3; angle read with scipy `as_rotvec`):
  * the claimed theta = 2 arctan|phi| misses by up to **0.68 rad**;
  * 2 arctan(|phi| / 2) holds to 5.7e-14, and the axis is phi/|phi| to 3.2e-14.
* **Exact (mpmath, 50 digits):**
  * |phi| = 1 gives theta = 0.92729521800161223243 rad = 2 arctan(1/2), **not** pi/2;
  * the quarter turn is at |phi| = 2 (theta = 1.5707963267948966192).
* The part that pi is excluded only in the limit |phi| -> infinity is true (|phi| = 4.3e4 at 3.1415 rad), but the
  stated relation and the quarter-turn statement are false.

## 4. Mutation sweep

### 4.1 New mutants (204; ids `p7v-*`)

By file: lifting 51, constraints 43, cost 35, instance 30, gtsam_boundary 17, fixture 15, forms 10, `__init__` 3. The
emphasis was on the new code (FrozenDict, `_rebuild`, `_Rebuilt`, the `_freeze` base copy, the ValueError guards) and
on operators not used in passes 1-6: slice offsets and strides, side swaps, sign / transpose / format changes, dtype
casts, dropped loops, guard inversions and container-type changes.

**Killed (179), grouped by first failing test:**

* **test_c01_index_map[true] (42):**
  * lifting: v-off, p-short, bias0, b-stride, o-nkf, Vp-is-V, Pp-is-P, Yp-overlap, nbase, rhov-stride, rhop-off,
    n-aniso, biasidx, vecform-neg, matform-rowmajor, sform-2, el-b-j, el-W-ba, el-cv-cp, el-Vp-M, el-WT, z-v-p,
    z-abs-s, z-s-s, z-rho-Rj, z-rho-noT, z-rhop-rv, dec-s-last, dec-R-C, dec-R-abs, dec-b-mul, dec-p-v, dec-f32,
    dec-proj-always, rhov-guard-inv;
  * cost: Rcv-T, rv-Dt2, rp-half, rv-dp, rp-RjM-minus, rv-noV;
  * forms: cmat-CT.
* **test_instance_matches_independent_gtsam_reference[true] (28):**
  * gtsam_boundary: gal3-RT, gal3-negt, int-cov-a, acc-cov-g, gl-vneg, gl-tpos, J-5cols, J-lm-neg, J-noinv,
    to-gyro-ba, cov-T;
  * fixture: sint, sgnss, sba, sbg, m1-th2, m2-noth, true-RT, exp-th, acc-minus, gyro-nobias, V0, om1, SG, m1-sign;
  * instance: Dt-literal, rw-ba-dtb2, sig-swap.
* **test_c09 (19):**
  * constraints: P0-bneg, P0-Q2, P1-2ss, P2-ba, M-dRT, Rj-i, w-ba, cv-perm, cp-bj, side-j-Y, side-i-Pp, P3-plusW,
    P3-P-noT;
  * forms: sym-axes01, matvec-T, EPS-2, hat-noneg, matmul-noT, linT-noop.
* **test_cost_terms_default_variant_is_iso (17):** Gam-full, M-plusGam, Gam-btrue, Gam-zero, rR-YY, vel-rp, pos-tauR,
  wprior-nosq, prior-R-nohalf, prior-v-p, gnss-sigprior, gnss-kf0, rw-1, rw-I, Q-skip0, guard-or, in-gnss-kf012.
* **test_c11 (7):** RG-P-noWT, RG-VV, RL-sV-AT, RL-sY-WT, RH-eaT, RH-Pcv, RS-plus.
* **test_instance_pickle_and_deepcopy_round_trip (5):** rebuild-swap, reduce-nopims, fz-dict-only, fz-list,
  pims-dropped.
* **Four each:**
  * test_decoder_projection_is_nearest_rotation: proj-D-mid, proj-noVt, dec-ld, dec-proj-never;
  * test_c10: RB1-YT, RB2-minus, RC1-P-cv, fm-dot-all;
  * test_c14: RH-a2, RH-noP, emit-skiplast, edges-1;
  * test_c33: partial-kronswap, partial-rp-rv, partial-noD, in-D-trace.
* **Three each:**
  * test_decoder_s_guard_threshold_tie_and_scale: SFLOOR-1e10, SFLOOR-1e14, dec-floor-1;
  * test_c12: P4-Rj, RR-plus, RR-vv;
  * test_package_exports: drop-State, drop-Layout, drop-sofz;
  * test_cached_instance_arrays_are_read_only: frozen-truth, Jnu-rows (a writable `.copy()` property), truth-list.
* **Two each:**
  * test_state_copy_is_independent: copy-R-alias, copy-v-alias;
  * test_truth_state_blocks_are_independent_arrays: ts-v-alias, ts-b-shared;
  * test_aniso_guards_raise_value_error: rhop-guard-type, co-guard-type;
  * test_P1_RA1: UPPER-strict, kf-2;
  * test_c13[False]: fres-abs, fmax-mean;
  * test_vech_order: vech-todense, vech-k1;
  * test_c08: fc-absz, vecrows-noT;
  * test_whitened_residual: wh-abs, wh-noclip;
  * gravity test: guard-ge, guard-y;
  * test_instance_arrays_have_no_writable_base: fz-base-inverted, fz-view.
* **One each:**
  * test_decoder_rejects_non_finite_z: dec-isnan;
  * test_decoder_degenerate_inputs: dec-isinf;
  * test_decoder_s_guard_counts_aniso_rho_entries: dec-zmax-nolast;
  * test_decoder_s_guard_floor: dec-floor0;
  * test_c16: UPPER-lower;
  * test_c13[True]: aniso-extra-fam;
  * test_c18: flat-ignore-names;
  * test_c01[zero]: cbar-btrue;
  * test_c24[aniso]: rho-order;
  * test_cost_terms_rejects_unknown_variant: unknown-continue;
  * test_form_helpers: cross-sym;
  * test_bias_boundary_converters: from-acc;
  * test_c25[iso]: fx-Tedge-ulp.

### 4.2 Replays

* The 4 F08 survivors and the 11 p8 mutants: 15/15 killed (Sec. 1).
* `tests/test_mutation_m3.py`: **99 passed** (148 s).

### 4.3 Complete survivor list (25) with rulings

Measured effect = the 1383-group dump diff of the mutant tree against clean daace65, plus targeted probes
(`probe_fd.py`, `c24meas.py`) where the dump has no input that reaches the change. As in passes 3-6, a change in a
claim-free contract (an output of a src function on a valid input, or a side effect on a caller's data) is
non-equivalent unless a proof shows that no such output changes.

| # | mutant | measured effect | ruling | class |
|---|---|---|---|---|
| 1 | **p7v-gb-inc-Dt-literal** (`Dt = pim.deltaTij()` -> `Dt = 0.5` in `increment`) | 355 groups. dv / dp error vs the independent reference: **3.55e-15 / 2.0e-15** (clean 0 / 4.4e-16; 2 ulp of \|dv\| = 2.2e-15). J and families change at the same level. Cost error vs own 5x5 at the bbar = 0 truth state, / max(1, J): iso **2.6e-13** (clean 1.1e-13), aniso **3.6e-13** (clean 3.4e-15), synthetic 1e6 aniso weight **1.9e-12** (clean 6.7e-13) | **NON-EQUIVALENT** | low. G is built with the nominal edge time, while `predict` integrated deltaTij = 0.5000000000000003, so dv absorbs g (Dt_pim - 0.5). Rule 1 fails (3.5e-13 > 1/10 of C24's 1e-12); rules 2 and 4 fail (excess >> 2 ulp). Missed because the dv / dp pin is 1e-13 |
| 2 | **p7v-in-fd-reduce-dict** (FrozenDict `__reduce__` -> `(dict, ...)`) | copy / deepcopy / pickle of `I.gnss` or `ed.cov` returns a **plain dict, and item assignment succeeds**; `asdict(I)["gnss"]` is a dict. Inside Instance / EdgeData `_rebuild` re-freezes, so the tested path is unchanged | **NON-EQUIVALENT** | low: FrozenDict's documented "survives pickle / deepcopy" is tested only through the Instance |
| 3 | **p7v-in-fd-init-alias** (`dict(*args)` -> `args[0]`) | `u = {1: 1.0}; fd = FrozenDict(u); u[1] = 2.0; u[5] = 3.0` -> fd[1] = 2.0, len 2 (clean 1.0, 1). src always passes a fresh dict | **NON-EQUIVALENT** | low: a read-only mapping that a caller's later writes reach (same pattern as pass-6 #4, bbar-nocopy) |
| 4 | **p7v-in-fd-len** (`len(self._d)` -> `+ 1`) | `len(I.gnss)` = 3, `len(ed.cov)` = 10 (clean 2 / 9); iteration unchanged | **NON-EQUIVALENT** | low: the public mapping reports a wrong size; untested |
| 5 | **p7v-in-fd-getitem-get** (`self._d[key]` -> `self._d.get(key)`) | `I.gnss[0]` returns **None** instead of KeyError; **`0 in I.gnss` is True** (Mapping.__contains__ relies on KeyError); `I.gnss.get(7, "dflt")` returns None | **NON-EQUIVALENT** | low: a `k in inst.gnss` guard in M4 would accept every keyframe |
| 6 | p7v-cn-P2-rows (P2 built from rows) | 40 groups (order / names within P2 only) | EQUIVALENT | proof: C15 (the same set of forms); documented equivalent in test_mutation_m3.py |
| 7 | p7v-cn-csc (`csr_matrix(Q)` -> `csc_matrix(Q)`) | 52 groups: the class name only. Measured: for all 962 constraints (both layouts) the CSC `indptr / indices / data` are **bitwise identical** to the CSR ones | EQUIVALENT | proof (rule 3): A_k is exactly symmetric (pass-6 #5), so CSC(A) = CSR(A^T) = CSR(A) storage; every matvec accumulates in the same order, so `residual` is bitwise identical, and code that walks indptr as rows gets the same answer. Remark: a one-line `isinstance(c.A, sp.csr_matrix)` assert would pin the annotated type |
| 8 | p7v-cn-resid-noA (`self.A @ z` -> `self.A.T @ z`) | 0 groups | EQUIVALENT | proof: A exactly symmetric; the transposed CSC matvec accumulates the same products in the same order |
| 9 | p7v-cn-mk-noflt (`float(b)` dropped) | 0 groups | EQUIVALENT | proof: every `_mk` call passes the float literals 1.0 or 0.0 |
| 10 | p7v-co-partial-G2T (`kron(G2.T, I)`) | 10 groups: Q 3.4e-19 relative to max; partial cost change 0; whitened <= 5e-17 relative | EQUIVALENT | rule 1 (C33: 0.05 eps >= 5e-6) |
| 11 | p7v-co-Wvp-T (caller W_vp transposed) | 0 groups on the symmetric weights | EQUIVALENT | proof on the documented domain (Term.W is "symmetric PSD"): W^T == W bitwise |
| 12 | p7v-fm-outer-nobroadcast (`broadcast_arrays` dropped) | 0 groups | EQUIVALENT | proof: numpy broadcasting of `f[..., :, None] * g[..., None, :]` gives the same array |
| 13 | p7v-gb-J-h03 (FD step 0.1 -> 0.3) | 479 groups: J moves <= 8.2e-15, but its error vs the logm-fit reference **falls** from 1.0e-14 to 1.9e-15. Cost errors vs own 5x5 at the bbar = 0 truth state fall too (iso 1.1e-13 -> 7.9e-14; aniso synthetic 6.7e-13 -> 4.9e-13) | EQUIVALENT | rule 4: the mutant's error is no worse than clean anywhere (it is better); rule 1 vs C04 (1e-8). Cf. pass-6 #21 |
| 14 | p7v-gb-nav-defaultnone (default t = 0.0 -> -0.0) | 2 groups: the sign bit of the time entry only | EQUIVALENT | proof: -0.0 == 0.0; the only arithmetic on t is addition in compose, where x + -0.0 = x |
| 15 | p7v-in-fd-setattr-allowed (`__setattr__` permits) | 0 groups | EQUIVALENT | proof: `__slots__ = ("_d",)` and no `__dict__`, so the only assignment that can now succeed is to the private `_d`. Its contents are already reachable for writes in clean (`fd._d[k] = ...`, M3-F12 a). No public operation changes |
| 16 | p7v-in-fd-noslots (`__slots__` dropped) | 0 groups | EQUIVALENT | proof: `__setattr__` still blocks every name; the difference is reachable only through the dunder `fd.__dict__` |
| 17 | p7v-in-rebuild-inst-shallow (`_frozen(cls(**kw))` -> per-field `_freeze`) | 0 groups (pickle / deepcopy / copy all still read-only) | EQUIVALENT | proof: the edges in kw are already EdgeData rebuilt and frozen by their own `__reduce__`; `_freeze` leaves the pims tuple's objects unchanged |
| 18 | p7v-in-bbar-nodtype (`dtype=float` dropped in build_instance) | 0 groups; Instance.bbar keeps an int / float32 caller dtype | EQUIVALENT | proof: every consumer (`J @ bbar`, `to_gtsam_bias`, `bbar +- d`) promotes to float64; integer and float32 values convert exactly; the array is read-only, so no in-place truncation can occur |
| 19 | p7v-in-Svp-T (`C_vp.T` for `C_pv`) | 0 groups (bitwise) | EQUIVALENT | measured identical; preintMeasCov is exactly symmetric here |
| 20 | p7v-in-tau-trace-sum (`sum(diag)` for `trace`) | 0 groups | EQUIVALENT | the same summation |
| 21 | p7v-in-WR-T (`inv(Sigma_rot.T)`) | 38 groups: W_R 1.6e-18 relative to max; partial cost change 0; whitened <= 1.6e-16 relative | EQUIVALENT | rule 1 (C33 >= 5e-6, C24 1e-12) |
| 22 | p7v-in-G2-pinv | 38 groups: G2 7.9e-16 relative; partial cost change <= 5.2e-18 (/ max(1, J)) | EQUIVALENT | rule 1 (C33 >= 5e-6) |
| 23 | p7v-in-Jrho-6-9 (`J[6:10][:3]`) | 0 groups | EQUIVALENT | the identical slice |
| 24 | p7v-in-zero-key (instance() also accepts "0") | 0 groups | EQUIVALENT | the documented domain {"true", "zero"} is unchanged; instance() documents no validation of other keys |
| 25 | p7v-in-edges-tuple-list (`edges=edges` list) | 0 groups | EQUIVALENT | proof: `_frozen` rebuilds `edges` as a tuple |

Totals: 25 survivors. **Non-equivalent 5 (#1-#5), all low. Equivalent 20 (#6-#25).**

## 5. Tests that cannot fail, and untested surface (at daace65)

1. Unchanged from pass 6:
   * the same-LAPACK projection pin (accepted);
   * the tie-precondition arithmetic;
   * the C02 / C03 arithmetic;
   * the test-side-only parts of C04, C15, C26, C33, C39;
   * the MATH CHECK labels on C05 / C06 / C28;
   * the scale invariance of C30's rank.
2. **The dv / dp / dR pin in test_instance_matches_independent_gtsam_reference (1e-13)** is far looser than the measured
   agreement:
   * vs `deltaXij`: dv 0, dp 4.4e-16, dR 1.0e-14;
   * vs my predict-based reference: all 0.

   A Dt-level boundary defect survives because of it (#1).
3. **FrozenDict's mapping contract is untested** apart from item assignment and the Instance round trip (#2-#5).
4. The pickle test does not cover a FrozenDict round-tripped on its own, `_d`, or protocols other than the default.

## 6. Findings and risks for M4

* **M3-F11 (new, gate-blocking, low): 5 non-equivalent survivors (Sec. 4.3 #1-#5).** Tests that would kill them:
  * (a) `len(I.gnss) == 2` and `len(I.edges[0].cov) == 9`;
  * (b) `with pytest.raises(KeyError): I.gnss[0]`, `0 not in I.gnss`, and `I.gnss.get(0) is None`;
  * (c) `u = {1: x}; fd = FrozenDict(u); u[1] = y; u[5] = y`: assert `fd[1] is x and len(fd) == 1`;
  * (d) for `copy.copy`, `copy.deepcopy` and `pickle` of `I.gnss` and of `I.edges[0].cov`: `type(...) is FrozenDict`
    and item assignment raises TypeError;
  * (e) tighten the pin to `np.abs(ed.dv - er["dv"]).max() <= 1e-15` and dp <= 1.5e-15 (measured 0 / 4.4e-16), or
    assert `GB.increment(pim, g, bbar).matrix()[:3, 3]` equals `pim.deltaXij().velocity()` exactly.
* **M3-F12 (new, not gate-blocking, low): residual freeze / serialisation items.**
  * (a) `FrozenDict._d` is a plain dict. `instance("true").gnss._d[1] = ...` silently poisons the lru-cached instance and
    the GNSS cost term (measured). This is private-attribute access, but MappingProxyType had no such path. Suggested
    fix: store `types.MappingProxyType(dict(...))` in `_d` and reduce to `(FrozenDict, (dict(self._d),))`.
  * (b) A FrozenDict copied / pickled on its own keeps read-only item assignment but returns **writable arrays**; only
    the Instance / EdgeData path re-freezes. Freezing values in `FrozenDict.__init__` (or in its `__reduce__` target)
    would make the class self-contained.
  * (c) Pickle protocols 0 and 1 of an Instance **abort the interpreter** (SIGABRT) inside GTSAM's
    PreintegratedImuMeasurementsG pickling, as does a bare PIM. This is GTSAM behaviour and not a src regression, and
    protocols 2-5 work. Any M4 cache or legacy serialiser must not request protocol < 2, or must drop `pims`.
* **M4 risks:**
  * **R1, dense-Q cancellation persists.** At truth (bbar = b_true), z^T Q z = 5.2e-9 vs factored 6.2e-22. Use the
    factored cost for primal values and gaps, and rescale before any Clarabel solve. Barfoot's sigma_1/sigma_2 >= 1e5
    test cannot be resolved at this conditioning.
  * **R2, corrected: P2 is the only handedness guard.** Dropping P2 admits det -1 lifted points even with **all 8
    redundant families kept** (49 / 120 blocks at bbar = 0; 37 / 120 at b_true; 36 / 90 aniso). At every such
    keyframe the adjacent w = J_R b_g is 0 (|w| ~ 1e-25), where RH / RS degenerate. Pass-6 R2 ("keep P2 or RH / RS")
    was wrong. Since physical gyro biases are ~1e-2, a relaxation can sit near such points; never prune P2.
  * **R3, redundant families are linearly dependent** (472 -> 453; aniso 490 -> 471). Remove the dependencies before
    handing them to Clarabel.
  * **R4, isotropic weights are not conservative** (pass-6 C45). Certifying the iso problem bounds nothing about the
    anisotropic one.
  * **R5, C35's 1e-10 tolerance is sampler-fragile.** It holds at 2.4e-11 only with exactly orthogonal rotations.
    Re-orthogonalise SDP-derived samples first.
  * **R6, pickling (pass-6 R6) is resolved** for protocols >= 2; see M3-F12 c for 0/1.
  * **R7, the J finite-difference error uses part of the C24 budget.**
    * The shipped J (h = 0.1) is accurate to 1.0e-14. That alone gives a C24-style deviation of 1.1e-13 at the
      bbar = 0 truth state with the iso weights (1/9 of 1e-12), and 6.7e-13 with C34's synthetic 1e6 weight.
    * h = 0.3 (survivor #13) or a least-squares fit cuts J's error to ~2e-15.
    * If M4 compares primal costs to 1e-12 at large weights, refit J.
  * **R8, results are not bit-comparable** with Rai's published tables (no MOSEK); say so wherever they are compared.
  * **R9, the surrogate gap at bbar = 0** (truth cost 3.3e-3 vs ~6e-22 at bbar = b_true) lies outside C26's domain and
    still applies.
