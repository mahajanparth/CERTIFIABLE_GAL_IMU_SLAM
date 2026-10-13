# Stage-B verification, pass 7: M2 baselines (`stage-b/m2-baselines` @ `da672b8`)

Seventh independent adversarial pass. It follows `808614a` (F15 remediation: `test_c25b`, S35/S36
replayed as file edits) and `da672b8` (F14 remediation: `test_c13e`, `test_c05c`, S46/S47/S48/S12/S13
plus F14-B1/F14-B3 replayed as file edits through the new `_run_in_copy` helper).

Scope: rulings on M2-F14 and M2-F15; every claim in `docs/claims/M2.yaml` (C01-C25); one claim that
exists only in the pass brief, **M2-C30**; 46 new file-level mutants (U-series); a 27-mutant replay of
earlier mutants as real file edits.

Rules kept: I did not read `docs/02_*` or any derivation narrative, any `seeds*.yaml`, or anything under
`.cgis_SEEDS_BACKUP*`. All scratch code and mutant copies lived in the session scratchpad, outside the
worktree. I did not edit `src/`, `tests/` or `docs/claims/`. On the clean tree at `da672b8` the full
suite is **125 passed** (`tests/test_baselines.py` alone 36/36); `git status` was clean before and after
the sweep, and the final run on the clean tree is recorded in Sec. 7.

## 0. Summary

| | |
|---|---|
| **F14** | **PARTIALLY FIXED.** The constructors are now pinned as functions of their bias. `test_c13e` / `test_c05c` use independent references (retyped Cay/Rodrigues Euler sums, own central FD, one 5x5 `expm`) at four non-true biases, and all 9 file-level kills are genuine (checked on the `E` lines of the failure output, not only on the harness's text match). But the same leak, a bias input replaced by the hard-coded fixture bias, still survives the **whole suite** at six other bias-reading sites. All six are called only at `BIAS_TRUE`: `CayleyFactor.predicted_rotation` (U10, 7.9e-3), Baseline 3 `predict` (U30, Δv 4.2e-2), `corrected_increment_gal3` (U31, 4.8e-2), `bias_jacobian_recursion`'s V̂_k (U38, 4.5e-4 on \|J\| = 0.1), and both `integrate_edge_states` helpers (U13/U32, KF2 Δv 0.11). New finding **M2-F17 (medium)** |
| **F15** | **PARTIALLY FIXED.** σ_bg and σ_ba are pinned as functions: two operating points (nonzero in (a), zero in (b)), 100x-distinct values, and the (accel, gyro) block order, which I confirmed independently (an accel-only random walk lands in rows 9:12). Hard-coding either one is killed (U24, U25), and so is an anisotropic gyro random walk (U42). σ_int is pinned at **one** nonzero value only (1e-3), and σ_int = 0 → 0 is checked nowhere. `setIntegrationCovariance(1e-3**2 I)` (U23b) passes `test_baselines.py` 36/36: at the c06 settings it moves the position block by 12% and the whole matrix by 9.98e-3, just under c06's 1e-2. In the full suite it dies only on S35's edit anchor going missing, which is a change detector |
| **New, outside F15's named scope** | **M2-F16 (medium):** `make_params`' white **gyro** density is pinned by nothing. Dropping it (U20, rotation block → 0) or quadrupling it (U26) passes the full suite (125/125). The only check is c06's 1e-2 whole-matrix cross-check, and the rotation block is 2.9e-4 of the matrix scale. σ_a is pinned only to about 1% (U40, ×1.005, survives) |
| **New** | **M2-F18 (medium, latent):** all three constructors **alias** the caller's bias array (`np.asarray` of a float array is the same object). If a caller updates its estimate in place, the factor's `bbar` changes while ΔR̄/Δv̄/J stay frozen. `xi` then moves by 7.3e-3 and Baseline 3 `predict` by 4.2e-2. `test_c13e` / `test_c05c` pass `bias.copy()`, which hides this |
| Verdicts | **25 CONFIRMED (C01-C25), 1 REFUTED (C30), 0 UNTESTABLE** |
| **C30** | **REFUTED.** tr Σ_n / (n tr Σ_1) = **1.340497** at n = 400 (edge 0; 1.340018 on edge 1). It is already 1.000006 at n = 2, 6e-6 against the claimed 1e-12, and 1.0849 at n = 200. The increments are independent, but the transition A is not the identity: rotation noise feeds velocity, and velocity feeds position. So tr Σ_n is superlinear (the position block grows ~n³). My own retyped Eq. (63) recursion agrees with `src` bit-for-bit (relative 0.0) |
| Mutation sweep | **46 new file-level mutants.** Against `test_baselines.py`: 17 killed, 29 survived. In the full suite, 3 more die, but only on change detectors (U05 on `test_pass5_replay_is_faithful`; U23/U23b on the S35 replay's edit anchor). **Medium survivors: 10** (U10, U13, U20, U26, U30, U31, U32, U38 survive the full suite; U23/U23b are guard-only). **Low or equivalent: 19** |
| Regression replay | **27 earlier mutants** as real file edits: M02, M05, M06, M07, M08, M09, M12, M13, M15, M17, N04, P17, P29, P33, P35, Q25, Q45, Q46, R01, R04, S12, S13, S35, S36, S46, S47, S48. **27/27 killed** in `test_baselines.py`. Nothing regressed |
| `_run_in_copy` | Its control is **sound**: the unmutated copy must pass every killer, it imports `src` from the copy (there is no installed package; `cwd` comes first on `sys.path`), and every kill is real. But its "killed on the intended assertion" check is **partly vacuous**. pytest echoes the failing test's source, so any message typed literally in the test is "found" whichever assertion failed. Demonstrated: the S35 edit, which fails only part (b), "satisfies" S36's part-(a) message. The same holds for `_BBAR_MSG` in S12/S13/S47 (Sec. 5) |

---

## 1. Independent references (no `src` imports)

Scratch module `ind/core.py`: constants retyped from the M2-C01 text; `Exp = expm∘hat`, `Log = logm`;
`Cay = inv(I − φ^/2)(I + φ^/2)`; Eq. (4) typed from the paper; the Eq. (9) Euler sums with a pluggable
retraction; the exact increment from one 5x5 `expm`; central-FD bias Jacobians (eps 1e-5). Scratch scripts
add a retyped Rodrigues, a retyped Eq. (63) recursion (C30), a 200-step x 6-channel FD linearisation of
the noisy Euler integrator (C06), my own 5x5 Gal(3) ZOH steps and FD Jacobians (C19-C21), and my own factor
graph with densities 3e-4 / 2e-2 / 2e-6, random walks 3e-5 / 5e-4, prior sigmas 5e-2 / 2e-1 and GPS 0.5,
all different from the test's (C25). `src` is imported only as the system under test.

**Independence of the new test references.** `test_c13e` uses `_indep_cayley_increment` (retyped Euler
sums with the closed-form `_indep_cay`), `_indep_cayley_bias_jacobians` (own FD, eps 1e-5 ≠ src's 1e-6),
`_indep_eq21b/c` with a retyped gravity literal, and `_indep_ground_truth_keyframes` (5x5 `expm`).
`test_c05c` uses `_indep_exp_increment` (retyped Rodrigues) and `_zoh_increment_expm`. What they still take
from `src`: `T.hat` (inside `_zoh_increment_expm`); the fixture's streams, `omega_body`, `f_body` and
`duration` (from `src.trajectory`, which C01 pins against retyped literals); and `ROT_VEC_0`. None of these
is a code path under test in F14. `test_c25b` derives its expected values (σ²Δt) from the densities, not
from GTSAM; it takes `make_params` and `to_gtsam_bias` from `src` only as the system under test. **The
references are independent.**

## 2. Ruling on F14: PARTIALLY FIXED; new finding M2-F17 (medium)

What is fixed (all checked as real file edits in a scratch copy):

| mutant | `test_baselines.py` | killed on (E-line) |
|---|---|---|
| S46 increments at fixture bias | killed (c13e) | dRbar, dvbar, Eq. (21b) consumer |
| S47 `bbar` replaced by fixture literal | killed (c13e) | bbar, dRbar, J_R, consumer |
| S48 `dRbar` at fixture gyro bias | killed (c13e) | dRbar |
| S12 `bbar[3:6]` = 0 | killed (c13e) | Eq. (21b) consumer, bbar |
| S13 `bbar[3:6]` negated | killed (c13e) | Eq. (21b) consumer, bbar |
| F14-B1 / F14-B3 constructors at fixture bias | killed (c05c) | dR |

The new check also kills my U45 (Eq. (21b) method uses db_g for db_a), P35 and N04. `bbar[3:6]` is still
read by no `src` method. The new consumer is the test's own `db = b − f.bbar`, which pass 6 accepted as a
remedy ("pin bbar[3:6] directly").

What is not fixed. I hard-coded each remaining bias input to `BIAS_TRUE`, one at a time:

| id | site hard-coded to the fixture bias | full suite | effect at bbar = 0 |
|---|---|---|---|
| **U10** | `CayleyFactor.predicted_rotation` offset (Eq. (16)) | survives 125/125 | ΔR 7.92e-3 |
| **U30** | `GalileanPreintegration.predict` bias | survives | R 7.7e-3, v 4.22e-2, p 1.12e-2 |
| **U31** | `corrected_increment_gal3` linearisation point | survives | Gal(3) distance 4.80e-2 at db = 0 |
| **U38** | `bias_jacobian_recursion` V̂_k | survives | J off by 4.5e-4 (\|J\| = 0.10) |
| **U13** | Baseline 1/2 `integrate_edge_states` | survives | KF2 R 1.5e-2, v 0.109, p 4.2e-2 |
| **U32** | Baseline 3 `integrate_edge_states` | survives | KF2 R 1.5e-2, v 0.109, p 4.2e-2 |

U10 is a `CayleyFactor` method that ignores the factor's own `bbar`, so it is squarely inside F14's
statement. U01/U02 (the same leak in `xi` / `rotation_residual_affine`) are killed by c13c, but
`predicted_rotation` builds its own ξ. U30, U31 and U38 are the Baseline 3 analogue: c05c pins only the
*constructor's* increment, never `predict`, the bias correction or the recursion at a non-true bias. The
commit message's "Baselines 1 and 3 the same treatment" covers the constructors only. **M2-F17 (medium):**
every bias read outside the three constructors is exercised only at `BIAS_TRUE`. In M3/M4 the bias is
estimated, so each of these would silently evaluate at the true bias. **Suggested kill:** extend c05c/c13e
to call `predict`, `corrected_increment_gal3`, `bias_jacobian_recursion`, `integrate_edge_states` and
`predicted_rotation` at the four `F14_BIASES`.

**M2-F18 (medium, latent): aliasing.** `CayleyFactor.from_edge(edge, b).bbar is b`, and likewise for
`ForsterPreintegration.bias_hat` and `GalileanPreintegration.bias_hat` (all `True`, measured). After
`b[:] = BIAS_TRUE` on a factor built at b = 0: `xi` moves by 7.34e-3 while ΔR̄ and J_R stay frozen at 0.
Baseline 3 `predict` moves by 4.22e-2, and GTSAM's internal biasHat no longer matches `bias_hat`. The new
tests pass `bias.copy()` and so cannot see it. **Suggested fix:** `np.array(bbar, dtype=float, copy=True)`.

## 3. Ruling on F15: PARTIALLY FIXED; new finding M2-F16 (medium)

* **Order.** Independently of the test: with only the accel random walk on (1e-3), `preintMeasCov`
  diag[9:12] = 5.0e-7 = σ²Δt and diag[12:15] = 0. With only the gyro random walk on, the reverse. So
  `ConstantBias` (accel, gyro) is respected, and c25b indexes `[9:12]` = accel, `[12:15]` = gyro correctly.
* **σ_bg, σ_ba:** pinned. U24 and U25 (each hard-coded to its c25b value) are killed by c25b(b), which
  sets them to 0. U42 (gyro random walk z-axis ×1.5) is killed. S36 is killed on the part-(a) assertion
  (E-line).
* **σ_int:** single-point. c25b(b) checks σ_int = 1e-3 only. In (a), σ_int = 1e-6, but the position block
  is not asserted, and no test asserts that σ_int = 0 gives zero integration noise. U23 (`1e-3` whenever
  nonzero) and U23b (`1e-3` always) pass `test_baselines.py` 36/36. In the full suite they die only because
  S35's replay anchor text is gone, which is a change detector, not a check. U23b's effect at the default
  σ_int = 0 is the position block +11.99% and the whole matrix 9.98e-3 relative, inside c06's 1e-2 by 0.2%.
* **M2-F16 (medium):** the white-noise densities in `make_params` are not in c25b, and c06b builds its own
  `PreintegrationParams` (`_gtsam_anisotropic_params`). So `make_params`' σ_g / σ_a are checked only by c06's
  normalised cross-check at 1e-2 relative to max\|Σ\| = 5.0e-5. The rotation block's max is 1.44e-8, which
  is 2.9e-4 of that scale. U20 (gyro density dropped: rotation block 100% wrong, whole matrix 2.4e-3) and
  U26 (gyro variance ×4: rotation block +300%) **pass all 125 tests**. U40 (accel variance ×1.005) also
  survives; U27 (×1.015) is killed. U21/U22 (hard-coded densities) are killed, because c25b(b) runs with
  σ_g = σ_a = 0. **Suggested kill:** in c25b, set σ_g alone and assert the rotation block equals
  σ_g² Δt I. This holds to 1e-7 relative on this fixture (measured 4.99999955e-7 to 4.99999980e-7 vs σ²Δt = 5e-7 at σ_g = 1e-3).

## 4. Mutation sweep: 46 new file-level mutants

Each mutant was a string edit of a `src` file in a fresh scratch copy of `src/` + `tests/` (+
`orchestrator/`, `docs/`, `experiments/` for full-suite runs). `test_baselines.py` was run first. Every
survivor was then re-run against the full suite.

**Killed (17):** U01 `xi` offset hard-coded (c13c) · U02 affine bbar_g hard-coded (c13c) · U14 per-axis
gyro density reversed (c06b, c06c) · U15 `integrate_edge` dt = 2.5e-3 (c04, c06b, c22b, c23) · U16
`noise_covariances` dt hard-coded (c06b) · U17 σ_a := σ_g in `integrate_edge` (c06b, c06c) · U21 / U22
`make_params` σ_g / σ_a hard-coded (c25b) · U24 / U25 σ_bg / σ_ba hard-coded (c25b) · U27 accel variance
×1.015 (c06) · U35 `to_gtsam_bias` gyro block hard-coded (c05b, c05c, c20) · U37 `integrate_stream`
ignores n_steps (c21) · U41 gyro covariance takes σ_a (c06) · U42 gyro random walk anisotropic (c25b) ·
U43 wrapper ignores `params` (c06b) · U45 Eq. (21b) method uses db_g as db_a (c10b, c13e).

**Survived, medium (10):**

| id | mutant | full suite | effect |
|---|---|---|---|
| U10 | `predicted_rotation` offset at fixture b_g | survives | F17: 7.9e-3 |
| U13 | B1/B2 `integrate_edge_states` at fixture bias | survives | F17: KF2 v 0.109 |
| U30 | B3 `predict` at fixture bias | survives | F17: v 4.2e-2 |
| U31 | `corrected_increment_gal3` about fixture bias | survives | F17: 4.8e-2 |
| U32 | B3 `integrate_edge_states` at fixture bias | survives | F17: KF2 v 0.109 |
| U38 | recursion V̂_k at fixture bias | survives | F17: 4.5e-4 abs |
| U20 | `make_params` drops σ_g | survives | F16: rotation block → 0 |
| U26 | `make_params` gyro variance ×4 | survives | F16: rotation block ×4 |
| U23 | σ_int hard-coded 1e-3 when nonzero | guard-only (S35 anchor) | F15 single-point |
| U23b | σ_int hard-coded 1e-3 always | guard-only (S35 anchor) | F15: position block +12% at σ_int = 0 |

**Survived, low or equivalent (19):**

| id | mutant | full suite | classification |
|---|---|---|---|
| U03 / U04 | factor Eq. (21b)/(21c) Dt := 0.5 | survive | fixture-degenerate: every M2 edge is 0.5 s. On a 0.2 s edge, r^v shifts by 2.87. Promote to medium if M3 uses another edge length |
| U05 | `from_edge` stores Dt = 0.5 | guard-only (`test_pass5_replay_is_faithful`) | same; c13e's Dt pin compares against `n_steps·dt` = 0.5 on every edge |
| U11 | Forster `predict` Dt := 0.5 | survives | fixture-degenerate |
| U39 | `corrected_increment_gal3` clock := 0.5 | survives | fixture-degenerate |
| U06 / U07 / U12 / U34 | Eq. (21b)/(21c)/Forster `predict`/`endpoint_W` ignore g_W | survive | fixture-degenerate (one gravity in M2) |
| U33 | `single_step` dt := 2.5e-3 | survives | fixture-degenerate (the time-varying stream uses DT) |
| U08 / U09 | `from_edge` integrates / differentiates RAW whatever `variant` | survive | low: REPARAM is a contrast, never a factor in any test |
| U28 / U29 | wrapper's `combined` ignored | survive | low: the wrapper's combined path is untested, but misuse is loud (TypeError, measured) |
| U40 | accel variance ×1.005 | survives | low: inside c06's 1e-2 |
| U46 | Eq. (21c) swap of (A_pg, db_g) ↔ (A_pa, db_a) pairs | survives | equivalent (sum commutes) |
| U47 | τ = 3/(tr Σ + 1e-30) | survives | equivalent |
| U48 | `preint_cov` transposed | survives | equivalent (symmetric) |
| U49 | Forster B_a position block ×(1 + 1e-9) | survives | below tolerance |

Totals: 17 killed + 10 medium + 19 low = 46.

**Regression replay, 27/27 killed** (`test_baselines.py`, killers in brackets): M02 (c04, c05b, c05c,
c13b, c13e), M05 (c11, c13c, c13d), M06 (c15, c16), M07 (c05b, c05c, c06b, c17, c19, c20, c23, c24, c25),
M08 (c18), M09 (c21), M12 (c05b, c13b, c13d, c13e, c22, c22b, c23, c24), M13 (c02, c03, c10, c10b, c16,
c18, c24, c25), M15 (c13, c13c, c13d), M17 (c09, c11, c13, c13c), N04 (c10b, c13e), P17 (c06c), P29 (c13,
c13c), P33 (c12), P35 (c13b, c13e), Q25 (c13c, c13d), Q45 (c13c), Q46 (c13c, c13d), R01 (c13d, c13e), R04
(c13d, c13e), S12, S13, S46, S47, S48 (c13e), S35, S36 (c25b).

## 5. Tests and assertions that cannot fail, or that compare the code to itself

Pass-6 items 1-14 still hold. New in this pass:

15. **`_assert_file_level_kill`'s message check** searches the whole pytest stdout. pytest prints the failing
    test's source, so a message typed as a literal in the test is always "present" whenever the test fails at
    or after that line. Measured: under the S35 edit (only part (b) fails), the S36 message `"bias random-walk
    covariance is not diag(sigma_ba^2, sigma_bg^2) Dt"` is found, and it is found in the echoed source line.
    Under S48 (only `dRbar` fails), `_BBAR_MSG` is found the same way. The check is vacuous for S36 and for
    the `_BBAR_MSG` component of S12/S13/S47. It is effective only for messages built from f-string
    templates (`{name}`) or split across source lines. Every kill is nevertheless real: I re-checked all 9
    against the `E ` lines only. **Fix:** match against lines starting with `E `.
16. **The file-level replays are change detectors as well as kill checks**: any edit of an anchor's exact
    text fails `test_pass6_*` through `assert text.count(old) == 1`, as U23/U23b demonstrate. That catches
    a refactor, not a defect.
17. **`test_pass6_sweep_covers_every_medium_survivor`** compares two literal sets in the same file.
18. **c13e's `bbar is not the true bias` and c05c's `ptp(stream) == 0` preconditions** depend on literals
    and the fixture only.
19. **c13e's Dt pin** compares against `edge.n_steps * dt`, which is 0.5 on every M2 edge, so a literal 0.5
    passes it (U05).
20. **c25b(a)'s accel/gyro cross-term assertion**: GTSAM 4.3.0's `PreintegrationCombinedParams` exposes no
    initial or cross bias covariance setter (member list checked), and the random-walk propagation keeps the
    two blocks uncoupled. I found no `make_params` edit that makes it fail.

## 6. Per-claim verdicts

All numbers are mine. `src` was called only as the system under test.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations of the 400-sample stream are bytes-identical to each other and to my retyped streams. dt, T, N, bias and R_0 match the retyped literals (R_0 to 1.1e-16 against `expm`) |
| C02 | CONFIRMED | M1 vs `quad_vec` 7.6e-17 / 5.6e-17; M2 vs nested quad 1.8e-16 / 9.7e-17. Chained `expm` states vs REF: KF1 v 1.6e-15, p 2.0e-15; KF2 R 2.2e-16, v 2.6e-15, p 3.3e-15 |
| C03 | CONFIRMED | GTSAM `predict` from a nonzero NavState vs my chain: R 8.9e-15, v 8.9e-16, p 4.4e-16 |
| C04 | CONFIRMED | Ratios v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086, 0.50043, 0.50022, 0.50011. `src` vs my Euler ≤ 4.2e-17 |
| C05 | CONFIRMED | `src` J vs my FD ≤ 2.4e-8 relative. Error ratio (1e-2 vs 1e-3), R/v/p: 99.98 / 100.02 / 100.02 and 100.14 / 99.96 / 99.95 |
| C06 | CONFIRMED | Asymmetry ≤ 1.7e-20, min eig −4e-23, trace monotone on both edges. My noisy-Euler FD vs `src`: 7.7e-9 / 9.8e-9 (FD-limited; the 2.7e-9 is the test's own reference; the bound is 1e-6). Raw R-v block vs GTSAM: iso 0.1704 / 0.1945, aniso 0.1122 / 0.2049. Charted: rot 2.1e-14 / 1.7e-14, R-v 5.08e-3 / 5.04e-3, R-p 7.56e-3 / 7.54e-3, ratios 0.4998-0.5009 |
| C07 | CONFIRMED | (a) 2.7e-15 over 2000 axes, θ ∈ (0, 3). (b) right 6.1e-7 (FD-limited), left 1.2. `src` J_Cay equals my typed Eq. (4) exactly |
| C08 | CONFIRMED | 5000 draws, \|ξ\| ≤ 5: max deviation 2.7e-15; min det 1.0000 |
| C09 | CONFIRMED | On-manifold 3.3e-16; perturbed by Exp(1e-3 u): min 6.3e-4 ≥ 1e-4; `src` equals my typed Eq. (19) |
| C10 | CONFIRMED | Self-consistent states: 8.9e-16 / 1.7e-13. Ground truth: v 2.1709e-3 (edge 0) / **2.7603e-3** (edge 1); p 5.4e-4 / 6.8e-4; equals \|dv_exact − dv̄\| to 2.7e-15 |
| C11 | CONFIRMED | 2000 draws: max \|Eq. (22b) − Eq. (19)\| 4.4e-16 (`src` and my typed version) |
| C12 | CONFIRMED | Rodrigues rotations: iso 1.6e-14, W = diag(1, 2, 5) max 37.8 ≥ 1. A first run with `expm` of angles up to ~10 rad gave iso 1.4e-12, from my generator's orthogonality loss, not from `src`. Note that the 1e-12 is absolute and the r-scale is unstated |
| C13 | CONFIRMED | J_R bytes unchanged after evaluating at a perturbed bias. J_R vs my Cay FD 3.1e-10, vs my Exp FD 2.3e-7. Exact vs truncated at \|ξ\| = 1e-2: 4.1e-3 ≥ 1e-6 |
| C14 | CONFIRMED | col 3 = [10, 20, 30], col 4 = [1, 2, 3], M[3,4] = 7; accessors and the `Gal3(M)` round trip hold; `translation() == position()` |
| C15 | CONFIRMED | Hat block support correct for all 10 e_k; Expmap(1e-6 e_k) moves only the stated component; Expmap vs `expm(Hat)` 1.1e-16; table (0, 6, 3) |
| C16 | CONFIRMED | KF0-KF2 **0.107628** (KF0-KF1 0.1008, KF1-KF2 0.1678); against `NavState.Logmap` exactly 0.0 on all three pairs |
| C17 | CONFIRMED | `ConstantBias([1,2,3],[4,5,6])` accessors and `vector()` as stated; `biasHatVector` = [b_a; b_g]; the adapters round-trip, and one application is not the identity |
| C18 | CONFIRMED | W-form ≤ 1.8e-15 at t_i = 0 and 3.7, t_j = t_i + 0.5. F2 error 0 at t_i = 0 and **18.148500** = \|g\|·0.5·3.7 exactly. Swapped order 2.453 = g Dt² at t_i = 0 |
| C19 | CONFIRMED | Galilean vs ZOH: R 1.0e-14, v 2.0e-14, p 3.8e-15. Tangent dv 2.17e-3 / 2.76e-3 ≥ 1e-4. Time-varying stream: right 1.1e-16, left 6.2e-4 ≥ 1e-5 (my 5x5 `expm` steps) |
| C20 | CONFIRMED | J[0:3,0:3] = J[9,:] = 0 exactly. Right composition ≤ 3.9e-10, linear in \|db\|. Component-wise at 5e-2 over 30 directions: max 1.13e-2 / 1.46e-2, min 1.1e-3 / 8.4e-4; top singular direction 1.27e-2 / 1.62e-2. The ≥ 1e-2 holds as a sup, as the claim states |
| C21 | CONFIRMED | My recursion (own 5x5 V_k, own FD C_k) vs my FD re-integration: worst 3.4e-9 relative over 40 steps; `src` recursion vs mine 2.4e-10 |
| C22 | CONFIRMED | KF2 B1-B2: v 6.17e-8, p 4.38e-8, R max-abs **2.9965e-8** (\|\|Log\|\|∞ 2.810e-8, \|\|Log\|\|₂ 3.510e-8). Per edge 2.32574e-8 / 5.51753e-8 vs formula, relative −2.3e-7 / −3.4e-7. Ratios 0.25 ×4 (my own Exp/Cay integrators) |
| C23 | CONFIRMED | B1-B3 v 1.2867e-3, p 1.2031e-3, R 3.1e-15; B2-B3 R 3.0e-8. Ratios B1 v 0.50028..0.50003, p 0.50000; B2 v 0.50026..0.50003, p 0.49998..0.50000 |
| C24 | CONFIRMED | B3 KF1/KF2 ≤ 1.6e-14; B1 rotation ≤ 9.5e-15; B2 rotation 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | Own graph (own densities and sigmas) and test densities give identical results: shapes 30x45, 39x45, 45x45; ranks 30 / 39 / 45; bias-prior rank 45; cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]; pose/velocity energy 2.5371; error ≤ 6.7e-23; unwhitened residual 2.0e-14. Full-graph s_min/s_max 1.3e-9 to 4.0e-9, far above the 45·eps rank tolerance. The cosines are unchanged under reweighting, as expected (a row-scaled Jacobian has the same nullspace) |
| **C30** | **REFUTED** | Retyped Eq. (63), constant inputs, dt = 2.5e-3, σ_g = 1.7e-4, σ_a = 1e-2. tr Σ_n / (n tr Σ_1) = 1.000006 (n = 2), 1.000210 (10), 1.021222 (100), 1.084949 (200), **1.340497 (400)** on edge 0; 1.340018 at n = 400 on edge 1. The first deviation above 1e-12 is at n = 2. `src` agrees with my recursion to 0.0 relative. The noise increments are independent, but Σ_{n+1} = A Σ_n Aᵀ + B Q Bᵀ with A ≠ I, so earlier noise is amplified (δθ → δv → δp). Linear growth would require A orthogonal on the noise subspace, and it is not |

## 7. Final state

Report-only commit. The full suite on the clean tree is recorded below.
