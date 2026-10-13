# Stage-B verification, pass 9: M2 baselines (`stage-b/m2-baselines` @ `8f99912`)

This is the ninth independent adversarial pass. It follows three commits:
* `ee3146e`: the pass-8 remediation (F23 gate, F24 near-π branch, F22 copy/read-only semantics, and reference tests for F21, F22 and F25);
* `17bed7a`: `PASS8_MUTANTS`, which replays the pass-8 survivors and both latent defects as file edits;
* `8f99912`: the `corrected_increment_gal3` docstring.

Scope:
* rulings on M2-F21 to M2-F25 and on the implementer's two requests (W20, Q44);
* every claim in `docs/claims/M2.yaml` (C01-C25);
* one claim that exists only in the pass brief, **M2-C32**;
* **56 new file-level mutants** (X01-X56);
* **45 earlier mutants replayed as real file edits**;
* the owner's **tolerance-equivalence rule** (2026-10-13), applied explicitly to every sub-tolerance mutant (Sec. 5.3).

Rules kept:
* I did not read `docs/00_*` or `docs/02_*`, any derivation narrative, any `seeds*.yaml`, or anything under `.cgis_SEEDS_BACKUP*`.
* All references, scripts and mutant copies lived in the session scratchpad. The references import nothing from `src`. `src` is imported only as the system under test.
* I did not edit `src/`, `tests/` or `docs/claims/`. `git status` of the worktree was clean before and after every sweep.
* One procedural slip, recorded for completeness. A backgrounded `cd ... && ... &` left one of my scratch scripts (`meas_c05.py`, 754 B) in the session's launch directory `/home/parth/Desktop/CertifiableIMUSLAM` for under a minute. I moved it into the scratchpad, and that repository's `git status` is unchanged. Nothing else outside the worktree and the scratchpad was touched.

**Gate verdict: M2 does NOT pass.** The owner's rule allows zero surviving non-equivalent mutants. This sweep leaves **9**, all low: X15, X21, X22, X24, X31, X41, W12, P23 and S21 (Sec. 5.4). For each one I give a distinguishing input and a measured effect above 1/10 of the governing claim tolerance.

## 0. Summary

| | |
|---|---|
| **F21** | **FIXED.** W02, W13, W14, W46, U47 and W10 die as file edits. Each dies on a check against an independent reference: the retyped gravity removal, the retyped sensor-to-body correction plus a 5x5 `expm`, the retyped `3/tr`, or the retyped `Rot3` branch |
| **F22** | **FIXED.** The F18 field list is now typed in the test, and W07 and W08 die. W09, the `b_g`/`b_a` views (F22-bg, F22-ba), a writable `bias_hat` (X38), a GTSAM-order `bias_hat` (X37) and three new aliasing mutants (X16, X17, X54) all die. The S39 edit is now provably equivalent (X39). Mutants that make `predict` write into the caller's state die (X52, X56). Residual note: `NavTriple` and `Edge` arrays remain writable through their attributes and `as_tuple()`, because a frozen dataclass does not freeze numpy arrays. X51 (`chain` writes into the caller's arrays) is equivalent: it changes values only by rounding |
| **F23** | **FIXED.** I propagated C0 at *exactly* zero densities through my own FD-linearised Euler step (B1 and B2; fixture edge 0 and the time-varying stream). `src` agrees to **1.7e-10 / 9.2e-11** relative, while C0 itself moves by 552% / 83%. The F23 replay dies on two reference tests |
| **F24** | **FIXED.** I swept `so3_log(Exp(θa))` against 40-digit `mpmath` rotations: θ from 1e-9 to π, 300 axes per angle, both sides of 2π/3, θ = π, and coordinate or near-coordinate axes. The worst error is **8.9e-16**, and no floating-point warning is raised. The F24 replay and three value-changing branch mutants (X03, X05, X07) die |
| **F25** | **PARTIALLY FIXED.** All small-angle survivors die on 60-digit `mpmath` references: Q01, Q39, W15, W17, W18, W19, Q44, plus the new X11, X12, X14 and X53. S39's edit is now equivalent. Under the tolerance rule, R36, R37, S16, S02, U49, S31, W11 and P46 are **equivalent**. **W12, P23 and S21 are NOT equivalent**: at valid inputs they move a bias Jacobian by **3.1e-6, 2.1e-7 and 2.3e-7** relative, against a threshold of 1e-7 (C21's 1e-6 relative). Numbers are in Sec. 5.3 |
| **W20** | **Killed legitimately; also value-equivalent.** After the F24 fix, dropping the lower clamp gives values bit-identical to the clean code (0.0 change on 1500 near-π inputs). But on 179 of 2000 random π-rotations it feeds `arccos` an argument below −1. That raises the IEEE *invalid* flag: a `RuntimeWarning` by default, and a `FloatingPointError` under `np.errstate(invalid='raise')`. The clean code does neither. This is an observable behavioural difference caught by a non-change-detector check, so the kill counts. Fragility of the test's warning filter: Sec. 6 |
| **Q44** | **EQUIVALENT under the rule; the 0.75-ulp check is fragile.** The effect on `so3_log` is ≤ **3.3e-24** absolute and 2.96e-16 relative, against a threshold of 2.3e-12 (Sec. 5.3). The ulp check is an independent reference, not a change detector. But it pins correct rounding, which no claim states. The clean code reaches 0.633-0.639 ulp against the 0.75 bound, a **1.17-1.18x** margin (three seeds); Q44 lands at 2.47-2.49 ulp (3.3x the bound). Recommendation: move it out of the gating set, or keep it as advisory |
| **C32** | **REFUTED.** The double-precision antisymmetric-part log is off by **2.06e-4** at θ = π − 1e-6 (200 axes), 3.1e-6 at π − 1e-5, 2.6e-8 at π − 1e-4 and **2.66e-10 at π − 1e-3**. It meets 1e-10 only for gaps ≳ 2e-3 (1.09e-10 at 2e-3). The same formula in 50 digits is exact (0.0), so the failure is conditioning, not algebra. The near-π branch is therefore necessary |
| **Verdicts** | **25 CONFIRMED (C01-C25), 1 REFUTED (C32), 0 UNTESTABLE** |
| **Mutation sweep** | **New: 56.** 33 are killed by reference checks, 17 are equivalent (with numbers), and **6 survive non-equivalent** (X15, X21, X22, X24, X31, X41). **Replay: 45 earlier mutants as real edits.** All 34 non-sub-tolerance replays die on reference checks. Of the 11 sub-tolerance replays, 8 are equivalent and **3 survive non-equivalent** (W12, P23, S21) |
| **New findings** | **M2-F26 (low, latent):** `CayleyPreintegration`'s covariance propagates gyro noise through **Exp's** right Jacobian. Its own integrator needs the Cayley one (Rai Eq. (4)). The rotation block is off by **4.6e-7 / 7.2e-7** relative against an FD linearisation of the noisy Cayley integrator; B1 agrees to 2e-11 on the same reference. No claim covers B2's covariance. **M2-F27 (low):** the six test gaps behind the nine survivors (Sec. 4) |

---

## 1. Independent references (no `src` imports)

The scratch module `ind/ref.py` and the per-claim scripts contain:
* constants retyped from the C01 text;
* `Exp = expm∘hat`, `Log` via `scipy Rotation`, and `Cay = inv(I − φ^/2)(I + φ^/2)`;
* the Eq. (9) Euler sums with a pluggable retraction;
* the exact increment from one 5x5 `expm`;
* Gal(3) as my own 5x5 matrices with `logm`, and an adjoint obtained by conjugating my own hat;
* 40-60 digit `mpmath` rotations, logs and series;
* a Richardson-extrapolated FD bias Jacobian;
* an FD linearisation of the noisy Euler step in the Forster chart, for either retraction;
* my own factor graph. Its parameters differ from the test's: densities 3e-4 / 2e-2 / 2e-6, random walks 3e-5 / 5e-4, prior sigmas 5e-2 / 2e-1, GPS 0.5, bias prior 1e-1.

C14-C19, C25 and the C32 attack use raw GTSAM or pure numerics.

I spot-checked that each new kill site is an independent reference:
* `test_b3_models_and_distance`: retyped `igal3`, `expm(igal3_hat)` and `igal3_log`;
* `test_b1_increment_error`: retyped literals;
* `test_f21_*`: a retyped sensor-to-body correction. On the clean tree it agrees with GTSAM to 1e-14 (GTSAM subtracts ω×(ω×arm)), which confirms the retyping;
* `test_f24`: the scipy quaternion route;
* `test_f25_*`: 60-digit `mpmath`;
* `test_f18` and `test_f22`: a typed field list, and behavioural write-then-compare checks.

## 2. M2-C32: REFUTED

**Set-up.**
* Input: R is the 50-digit `mpmath` `Exp(θa)` rounded to double, for 200 random axes per angle.
* Formula under test: the claim's own. θ = arccos((tr R − 1)/2) and φ = θ·vee(R − Rᵀ)/(2 sin θ), with and without the arccos clamp.
* Reference: θa.

| gap π − θ | 1e-1 | 1e-2 | 3e-3 | 2e-3 | 1e-3 | 1e-4 | 1e-5 | **1e-6** |
|---|---|---|---|---|---|---|---|---|
| max abs error, double | 2.9e-14 | 3.0e-12 | 4.0e-11 | 1.09e-10 | **2.66e-10** | 2.58e-8 | 3.13e-6 | **2.06e-4** |
| same formula, 50 digits | 0.0 | 0.0 | 0.0 | — | 0.0 | 0.0 | 0.0 | 0.0 |

The claim says the formula is accurate to 1e-10 "for every θ ≤ π − 1e-6". That fails for every gap below about 2e-3. At the stated margin the error exceeds the bound by about 2e6. The clamp changes nothing at these gaps.

The mechanism is floating-point conditioning:
* the rounding of tr R gives δc ≈ 4e-16;
* this produces δθ = δc / sin θ;
* sin θ then carries a relative error δθ·cot θ ≈ δc / gap².

The claim's conclusion is false: a dedicated near-π branch *is* necessary at that margin. `src`'s new antipodal branch (switching at θ > 2π/3) is accurate to 8.9e-16 everywhere on my sweep (F24).

## 3. Rulings on F21-F25

**F21: FIXED.** Each of the six named mutants dies as a file edit in my own copies, on its reference assertion:

| mutant | killed by | reference |
|---|---|---|
| W02 | `b3_corrected_increment_gal3` | `igal3(dR, dp, dv, 1.37)` under G_A and G_B |
| W13, W14 | `f21_kwargs_reach_gtsam_body_p_sensor` | retyped body_P_sensor correction, 5x5 `expm` and its own FD |
| W46, U47 | `b2_precision_and_weighted_square` | tr Σ from 4.3e-8 down to 1e-200 |
| W10 | `b3_gal3_adapters` | the Rot3 branch |

The inventory's new note on `bias_jacobian_recursion`'s `eps` says it "does not change the result measurably". That is false outside the fixture regime: see the W12 survivor, ruled under F25.

**F22: FIXED.**
* `test_f18` now iterates the typed `CAYLEY_FACTOR_ARRAY_FIELDS`, so W07 and W08 die.
* W09, F22-bg and F22-ba die on `test_f22_no_outward_aliasing`.
* `bias_hat` is now a read-only view of GTSAM's `biasHat`. X37 (GTSAM order) dies in 10 tests, and X38 (writable copy) dies in f22.
* X39 is S39's own edit (`predict` reads `pim.biasHat()`). It is now provably equivalent: `bias_hat` *is* `from_gtsam_bias(pim.biasHat())`, and `to_gtsam_bias` rebuilds a `ConstantBias` from the same doubles. Measured: `predict` is bitwise identical over 40 (bias, state, edge) draws.
* New aliasing mutants: X16 (NavTriple aliases p), X17 (Edge aliases f_body) and X54 (`sigma_g` aliases the caller) die in f18. X52 and X56 (`predict` writes into the caller's state) die in 7 and 4 tests.
* Residual note, not a defect: `NavTriple` and `Edge` are frozen dataclasses whose numpy arrays stay writable. X51 makes `chain` write into the caller's arrays. It changes the caller's p by **2.8e-14** (rounding only) and the returned state by 0.0. That is equivalent under the rule: the tightest state tolerance is C02/C03's 1e-12, so the threshold is 1e-13.

**F23: FIXED.**
* The gate is now `has_noise or any(cov != 0)`.
* My reference propagates C0 = diag(1..9)·1e-4 at exactly zero densities, through an FD-linearised transition of my own Euler step (Exp or Cay retraction).
* `src` matches it to 1.7e-10 on fixture edge 0 (200 steps) and 9.2e-11 on the time-varying stream (40 steps), for B1 and B2 alike. C0 itself moves by 5.52 and 0.83 relative.
* The F23 replay dies on `test_f23_*` and `test_b1_single_step_reads_every_field`. The assertion that pinned the defective gate is gone.
* X19 (`if True:`) and X20 (gate on the diagonal only) are equivalent on the function's domain (Sec. 5.3).

**F24: FIXED.**
* Sweep: 18 angle settings × 300 axes, against 40-digit `mpmath` rotations rounded to double, plus 24 pathological axis/angle pairs. Worst errors:

  | θ | worst error |
  |---|---|
  | 1e-9 | 5.2e-26 |
  | 1e-6 | 2.1e-22 |
  | 0.5 | 5.6e-17 |
  | 2π/3 − 1e-9 | 8.9e-16 |
  | 2π/3 + 1e-9 | 4.4e-16 |
  | 3.0 | 4.4e-16 |
  | π − 1e-2 to π − 1e-12 | ≤ 8.9e-16 |
  | π, checked as Exp(log) = R and ‖log‖ = π | 8.9e-16 |

* Coordinate axes at π, π − 1e-7, 2.1 and 3.0 give an error of exactly 0.
* No warning was raised, with warnings set to error.
* Every value-changing branch mutant dies on `test_f24`: X03 (angle from arccos), X05 (axis from column 0), X07 (cos θ := −1).
* The branch mutants that survive (X01, X02, X04, X06, X09, X10) are equivalent under the rule (Sec. 5.3).

**F25: PARTIALLY FIXED.**
* **Small-angle branch survivors:** all die on independent 60-digit `mpmath` references, as file edits in my copies. Q01, Q39, W15, W17 and W18 die on `f25_series_branches`; W19 and Q44 on `f25_so3_log_small_angles`. The new variants X11, X12, X14 and X53 die too.
* **Change-detector-only kills from pass 8** (R36, R37, S02, S16, S21, S39):
  * S39 is now equivalent (above).
  * R36, R37, S16 and S02 still survive every reference test, but they are equivalent under the tolerance rule (Sec. 5.3).
  * **S21 is not equivalent.** On a high-rate edge its factor Jacobians move by 2.31e-7 relative, against a 1e-7 threshold.
* **Sub-tolerance class** (Sec. 5.3, one by one):
  * equivalent: R36, R37, S16, S02, U49, S31, W11, P46;
  * **not equivalent: W12, P23, S21.**

## 4. New findings

**M2-F26 (low, latent): Baseline 2's covariance uses the wrong noise-input Jacobian.**
* `CayleyPreintegration` inherits `_propagate_covariance`, which sets `Bg[0:3] = so3_right_jacobian(w dt) * dt`. That is the right Jacobian of **Exp**.
* For the Cayley step `dR_k Cay((ω − b − n) dt)`, the noise enters as `Cay(wdt) Exp(−J_Cay(wdt) n dt)`. The correct input is therefore Rai Eq. (4)'s `J_Cay(w dt)`.
* The transition, `A = Cay(w dt)ᵀ`, is correct.
* Against my FD linearisation of the noisy Cayley Euler step:

  | gyro-only densities | edge 0 | edge 1 |
  |---|---|---|
  | B2, rotation block | **4.63e-7** | **7.20e-7** |
  | B2, whole matrix | 4.63e-7 | 8.16e-7 |
  | B1, same reference | 2.0e-11 | 3.3e-11 |

* The discrepancy is O(‖ω dt‖²) per step.
* No claim covers B2's covariance. No test checks it with gyro noise on: `test_f23` runs B2 at zero densities, where Bg does not enter.
* Low severity at these rates. It starts to matter once Rai Eq. (23)'s τ = 3/tr Σ is fed B2's own Σ.

**M2-F27 (low): the gaps behind the nine survivors of Sec. 5.4.**
* (a) **Covariance tolerance (X31).** The covariance tests use `COV_FD_REL = 1e-7`, which is 37x looser than the 2.7e-9 agreement that C06 states. Their FD reference is itself only good to about 2.5e-9 (I measured 2.46e-9 isotropic and 2.8e-10 anisotropic). Killing X31 therefore needs an analytic, retyped Eq. (63) reference, not FD.
* (b) **B2's covariance (X24, F26).** It is unclaimed and checked only at zero density.
* (c) **`has_noise` semantics (X21, X22).** Per-axis densities with zero axes, and tiny positive densities, are not covered.
* (d) **M1/M2 against quadrature (X15).** They are checked only on slow edges, where a 1e-13-relative K² error shows up as 5.6e-16 absolute.
* (e) **`navstate` (X41).** It is never fed a state carrying the orthogonality drift that `src`'s own integrators produce.
* (f) **FD-step choices (W12, P23, S21).** They are checked only on slow, fixture-like edges and at small eps.

## 5. Mutation sweep

**Harness.**
* Each mutant is a string edit of a `src` file, applied in a fresh scratch copy of `src tests orchestrator experiments docs conftest.py pyproject.toml`. The anchor is asserted unique.
* The "fast set" is `test_baselines`, `test_input_dependence` and `test_orchestrator` (100 tests). It contains only reference checks.
* Every fast-set survivor that is not equivalent is re-run on the full suite (232 tests), to report change-detector hits.
* A comment-only control copy passes 100/100.
* "Guard-only" means a mutant dies only in `test_mutation`: `test_pass5_replay_is_faithful`, the pass-6/7/8 edit anchors, or the controls of the copy-runner fixtures. Under the owner's rule these do not count as kills.

### 5.1 New mutants X01-X56

| mutant | change | result |
|---|---|---|
| X01 | `LOG_NEAR_PI_COS` = −0.99 | survives; equivalent (5.3) |
| X02 | `LOG_NEAR_PI_COS` = 0.0 | survives; equivalent |
| X03 | near-π angle from arccos, not atan2 | killed: f24 |
| X04 | axis sign test `<=` | survives; equivalent (log set-valued at π) |
| X05 | axis always from column 0 | killed: f24 |
| X06 | symmetric part uses the clamped c | survives; equivalent (bitwise 0 on 3043 inputs) |
| X07 | symmetric part with cos θ := −1 | killed: f24 |
| X08 | generic sin θ from √(1 − c²) | killed: f25_so3_log_small_angles |
| X09 | atan2 with the clamped c | survives; equivalent (bitwise 0) |
| X10 | near-π result × (1 + 1e-15) | survives; equivalent |
| X11 | `SMALL_ANGLE` = 1e-10 | killed: f25_series, traj_M1_M2 |
| X12 | Exp series K² coefficient 1/3 | killed: f25_series |
| X13 | Exp K² coefficient in half-angle form (an identity) | survives; equivalent |
| X14 | M2 series K²/12 | killed: f25_series |
| X15 | M1 closed-form K² × (1 + 1e-13) | **survives; NOT equivalent** |
| X16 | NavTriple aliases p | killed: f18 |
| X17 | Edge aliases f_body | killed: f18 |
| X18 | time-varying stream phase + 1e-9 | killed: traj_time_varying_stream |
| X19 | covariance gate `if True:` | survives; equivalent on dt > 0 |
| X20 | gate on diag(cov) only | survives; equivalent on PSD cov |
| X21 | `has_noise` requires *all* gyro axes > 0 | **survives; NOT equivalent** |
| X22 | accel densities ≤ 1e-20 ignored | **survives; NOT equivalent** |
| X23 | Q/(dt + 1e-12) | killed: c06b, b1_noise_covariances |
| X24 | covariance transition always Exp (affects B2) | **survives; NOT equivalent** |
| X25 | Jr = I | killed: c06b, c06c, two b1 |
| X26 | Ba position block uses the next rotation | killed: c06b, c06c, two b1 |
| X27 | J_R in the left chart | killed: c05, c05b, c05c, c13b, c13e, two b1/b2 |
| X28 | forward-difference Jacobians | killed: c05c, c13b, c13e, two b1/b2 |
| X29 | `increment_error` in the left chart | killed: b1_increment_error |
| X30 | Forster `FD_EPS` = 3e-6 | survives; equivalent |
| X31 | Ba velocity block × (1 + 2e-9) | **survives; NOT equivalent** |
| X32 | truncated contrast to second order | killed: c13, c13c, two b2 |
| X33 | factor Jacobians ignore `variant` | killed: b2_factor_from_edge |
| X34 | det(I + ξ^/2) | survives; equivalent |
| X35 | tr Σ + Σ₀₁ − Σ₁₀ | survives; equivalent on symmetric Σ |
| X36 | factor fields through float32 | killed: 9 tests |
| X37 | `bias_hat` in GTSAM order | killed: 10 tests |
| X38 | `bias_hat` returns a writable copy | killed: f22 |
| X39 | S39's edit: `predict` reads `pim.biasHat()` | survives; equivalent (bitwise) |
| X40 | σ_int floored at 1e-8 | killed: two b3 |
| X41 | `navstate` uses `Rot3.ClosestTo` | **survives; NOT equivalent** |
| X42 | Gal(3) FD Jacobian in the left chart | killed: c20, c21, three b3/f21 |
| X43 | recursion output appended without `.copy()` | survives; equivalent (bitwise; `J` is rebound every step) |
| X44 | `gal3_distance` uses the 2-norm | killed: b3_models_and_distance |
| X45 | F2-swapped order = F2 order | killed: c18, b3_endpoint |
| X46 | `integrate_stream` ignores `n_steps` | killed: c21, b3_integrate_stream |
| X47 | B3 `integrate_edge_states` drops kwargs | killed: b3_integrate_edge_states |
| X48 | `preint_cov` reordered to (rot, vel, pos) | killed: c06, c06b |
| X49 | `J_FD_EPS` = 3e-7 | survives; equivalent |
| X50 | right model ξ × (1 + 1e-7) | killed: b3_models_and_distance |
| X51 | `chain` writes into the caller's arrays (rounding-level) | survives; equivalent |
| X52 | B1 `predict` corrupts the caller's p | killed: 7 tests |
| X53 | `so3_log` small-angle branch removed | killed: c05, b1_bias_jacobians_fd, f25, traj_so3 |
| X54 | `sigma_g` aliases the caller | killed: f18 |
| X55 | `as_tuple` copies v | survives; equivalent (outputs identical) |
| X56 | B3 `predict` overwrites the caller's p | killed: c03, c24, b3_states, f22 |

**Count (56).**
* **Killed by reference checks (33):** X03, X05, X07, X08, X11, X12, X14, X16, X17, X18, X23, X25-X29, X32, X33, X36-X38, X40, X42, X44-X48, X50, X52-X54, X56.
* **Equivalent (17):** X01, X02, X04, X06, X09, X10, X13, X19, X20, X30, X34, X35, X39, X43, X49, X51, X55.
* **Not equivalent, surviving (6):** X15, X21, X22, X24, X31, X41.

### 5.2 Replay of earlier mutants as real file edits (45)

Each edit is my own, applied in a fresh copy.

| group | mutants | result |
|---|---|---|
| pass-8 W/F survivors and defects (21) | W02, W07, W08, W09, W10, W13, W14, W46, U47, F22-bg, F22-ba, F23, F24, Q01, Q39, W15, W17, W18, W19, W20, Q44 | **21/21 killed**, each on its intended reference test (Sec. 3). W20 dies via the IEEE-invalid warning and is also value-equivalent. Q44 dies via the ulp check and is equivalent under the rule |
| earlier gating mutants (13) | M01, M06, M07, M08, M09, M12, M17, N27, S35, U20, V12, F18-B3, W42 | **13/13 killed**. M07 dies in 21 tests, M01 in 15, M12 in 13, M17 in 6, N27 in 5, S35 and U20 in 3 each, M06 and M09 in 3 each, M08 and F18-B3 in 2 each, V12 and W42 in 1 each. All are reference tests |
| sub-tolerance class (11) | R36, R37, S16, S02, S21, U49, P23, S31, W11, P46, W12 | none killed by a reference test. Under the tolerance rule (5.3): **8 equivalent; W12, P23, S21 NOT equivalent** |

Nothing regressed.

### 5.3 Equivalence rulings under the tolerance-equivalence rule

**How I apply the rule.** For each mutant:
* I list every output whose value it changes, Q.
* τ(Q) is the tightest tolerance that a claim in `M2.yaml` states for Q. Where no claim states a tolerance for Q itself, I use the claimed quantity Q is computed into, and name it.
* The effect E(Q) is measured over the claims' own inputs **and** over a broad sample of valid inputs to the function: other edges, high-rate edges, other eps, and all angles in (0, π].
* **Equivalent ⇔ E(Q) ≤ τ(Q)/10 for every Q, on the broad sample.**

Two choices of τ need stating:
* **Bias Jacobians.** The claim tolerance stated on a bias Jacobian is C21's "within 1e-6 relative". C20's "≤ 1e-8 at ‖db‖ ≤ 5e-2" applies to the Gal(3) right-composition model, so I apply it only to Baseline 3's J. For B1 and B2 I also report the C05 error ratio.
* **`so3_log`.** The only yaml tolerance on an SO(3) logarithm value is C22's per-edge gap ‖Log(dR_B1ᵀ dR_B2)‖ = formula to 1e-3 relative. On edge 0 that is 1e-3 × 2.3257e-8 = 2.3e-11, so the threshold is **2.3e-12**. If one instead took the 1e-13 that C07 and C11 state for rotation *matrices*, the threshold would be 1e-14. Under that alternative only X01 (1.78e-14) would flip; I flag it in its row.

| mutant | quantity Q | tightest claim τ | τ/10 | measured E (broad sample; claim inputs) | ruling |
|---|---|---|---|---|---|
| **R36** J_R ×(1+5e-9) | B2 factor J_R | C21 1e-6 rel | 1e-7 rel | 5.0e-9 rel (an exact scale, the same on every edge); \|δJ db\| at 5e-2 = 1.8e-10. Claim quantities: C10 (db = 0) 0.0, C13a bitwise unchanged, C13b min 3.1e-5 unchanged | **equivalent** |
| **R37** A_va ×(1+5e-8) | B2 factor A_va | C21 1e-6 rel | 1e-7 | 5.0e-8 rel; C10 0.0 | **equivalent** (2x margin) |
| **S16** A_pg ×(1+5e-8) | B2 factor A_pg | C21 1e-6 rel | 1e-7 | 5.0e-8 rel; C10 0.0 | **equivalent** (2x) |
| **S02** dR̄·Cay(3e-13 e₃) | B2 preintegrated rotation dR̄ | C22 per-edge gap 1e-3 rel × 2.3257e-8 = 2.3e-11 | 2.3e-12 | 3.0e-13 rad | **equivalent** (would not be under C07's 1e-13 matrix tolerance, which concerns Cay vs Exp, not dR̄) |
| **U49** Ba pos ×(1+1e-9) | B1/B2 covariance | C06 "agreement ≤ 2.7e-9 relative" | 2.7e-10 rel | 7.5e-12 (fixture, both edges, iso and aniso), 3.8e-11 (accel-only time-varying stream), per block | **equivalent** (7x margin) |
| **S31** `FD_EPS` 1e-4 | B1 J, B2 factor J | C21 1e-6 rel | 1e-7 | fixture and test edges 3.4e-8; high-rate edges (\|ω\| = 5 rad/s, 1 s and 1.5 s) 2.1e-8. C05 ratios 99.997 / 99.981 vs clean 100.009 / 99.981. It is *more* accurate: J_v error vs Richardson FD 2.6e-10 vs clean 2.35e-8 | **equivalent** |
| **X30** `FD_EPS` 3e-6 | same | 1e-6 rel | 1e-7 | 4.6e-8 (fixture and test edges), 2.0e-8 (high-rate); C05 99.999 / 99.984 | **equivalent** |
| **P23** `FD_EPS` 1e-3 | same | 1e-6 rel | 1e-7 | fixture 4.8e-8, but **high-rate edges 2.13e-7 (B1) / 2.31e-7 (B2)** | **NOT equivalent** |
| **S21** factor eps 1e-3 | B2 factor J | 1e-6 rel | 1e-7 | fixture 4.4e-8, **high-rate 2.31e-7** | **NOT equivalent** |
| **W11** `J_FD_EPS` 1e-5 | B3 J (and the C20 model) | C21 1e-6 rel; C20 1e-8 abs at ‖db‖ ≤ 5e-2 | 1e-7 rel; 1e-9 | J 5.4e-9 rel (fixture), 2.0e-9 (high-rate); \|δJ db\| at 5e-2: 5.2e-10; C20 residual 3.4e-10 → 3.5e-12 (change 3.9e-10) | **equivalent** |
| **X49** `J_FD_EPS` 3e-7 | same | same | same | 6.0e-9 rel, 1.9e-9 (high-rate); \|δJ db\| 5.4e-10; C20 change 2.5e-10 | **equivalent** |
| **P46** `C_FD_EPS` 1e-3 | B3 recursion J (default eps) | C21 1e-6 rel | 1e-7 | 1.1e-9 (claim stream), 1.9e-9 (dt = 0.05), 9.7e-10 (dt = 0.3). An explicit eps is unaffected (0.0) | **equivalent** |
| **W12** recursion ignores `eps` | B3 recursion J | C21 1e-6 rel | 1e-7 | 1.5e-8 at the claim's stream; **1.9e-7 at dt = 0.3, eps = 5e-2; 3.1e-6 at dt = 0.3, eps = 0.2** | **NOT equivalent** (a public input is dropped) |
| **Q44** clamp at 1 − 1e-15 | `so3_log` | C22 → 2.3e-11 | 2.3e-12 | 3.3e-24 abs, 2.96e-16 rel (θ from 1e-12 to 1) | **equivalent** |
| **W20** lower clamp dropped | `so3_log` value | same | 2.3e-12 | 0.0 on 1500 near-π inputs | **value-equivalent**, but it raises the IEEE invalid flag (179/2000 π-rotations): killed legitimately (Sec. 0) |
| **X01** −0.99 | `so3_log`, θ ∈ (2.09, 3.0) | C22 → 2.3e-11 | 2.3e-12 | 1.78e-14 (711/3043 inputs differ); 0.0 for θ < 0.3 | **equivalent** (*would flip* under a 1e-13 matrix-tolerance reading: 1.78e-14 > 1e-14) |
| **X02** 0.0 | `so3_log`, θ ∈ (π/2, 2π/3) | same | 2.3e-12 | 6.7e-16 | **equivalent** |
| **X04** `<=` | `so3_log` at exactly symmetric R (θ = π) | same | — | Differs only where vee(R − Rᵀ) = 0 exactly (4 of 3043 inputs, exact π-rotations): returns −πa instead of πa. Both are logarithms of R; ‖·‖ is identical, and Exp of the two differs by ≤ 1.7e-16. The docstring states "either sign is a valid logarithm" | **equivalent** (set-valued output) |
| **X06** clamped c in B | `so3_log` | same | 2.3e-12 | 0.0 (bitwise) | **equivalent** |
| **X09** atan2 with clamped c | `so3_log` | same | 2.3e-12 | 0.0 | **equivalent** |
| **X10** ×(1+1e-15) | `so3_log`, θ > 2π/3 | same | 2.3e-12 | 3.6e-15 | **equivalent** |
| **X13** half-angle K² coefficient | `so3_exp` | C07a 1e-13 (Exp matrix) | 1e-14 | 6.7e-16 (θ ∈ [1e-8, π]); 1.1e-16 below 0.3 rad | **equivalent** |
| **X19** gate `if True` | B1/B2 covariance | C06 2.7e-9 rel | 2.7e-10 | 0.0, bitwise, on 16 cases (iso, aniso, single source, C0, per-axis, tiny densities, non-PSD). At dt = 0 the mutant gives NaN where the clean code keeps 0, but dt = 0 is outside the shipped function's domain: the clean code already returns NaN at dt = 0 whenever a density or C0 is nonzero (Q = σ²/dt) | **equivalent** on dt > 0 |
| **X20** diag gate | covariance | same | 2.7e-10 | 0.0 on every PSD input. A PSD matrix with zero diagonal is zero, so the gates agree on all covariances. It differs (100%) only for the non-PSD input ones(9) − I | **equivalent** on the covariance domain |
| **X34** det(I + ξ^/2) | Eq. (17) determinant | C08 1e-12 | 1e-13 | 3.6e-15 (5000 ξ, ‖ξ‖ ≤ 5); det Mᵀ = det M | **equivalent** |
| **X35** tr + Σ₀₁ − Σ₁₀ | τ = 3/tr Σ | no claim on τ; nearest is C12 (1e-12 on the isotropic identity) | — | 2.2e-16 rel on symmetric Σ; 0.0 on `src`'s propagated blocks (asymmetry 1.7e-21) | **equivalent** on symmetric Σ |
| **X39** S39 edit | B3 `predict` | C03 1e-12 | 1e-13 | 0.0, bitwise | **equivalent** |
| **X43** no `.copy()` | recursion list | C21 | — | 0.0, bitwise. `J = Ad J + C` rebinds `J` each step, so no list element is ever mutated afterwards | **equivalent** |
| **X51** `chain` in place | KF states | C02/C03 1e-12 | 1e-13 | returned state 0.0; caller's p 2.8e-14 | **equivalent** |
| **X55** `as_tuple` copies v | NavTriple outputs | — | — | 0.0 (no `src` path writes into `as_tuple` results) | **equivalent** |

### 5.4 Complete survivor list (non-equivalent)

**Full-suite status.** All nine are run on the full suite (232 tests):
* X15, X21, X22, X24, X31, X41, W12 and P23 pass **232/232**, so not even a change detector fires.
* S21 dies only on `test_pass5_replay_is_faithful`, which is a change detector: it compares `src` with a copy of `src`. The same run also logged 37 fixture errors in the `test_mutation` copy-runner fixtures. They happened under a load average of 172-185, and no reference test failed.
* For information, among the equivalent replays: R36, R37 and S16 also die only on `test_pass5_replay_is_faithful`; R36's run had the same 37 load-induced fixture errors. S02 dies only on that test and on the `pass6` S48 anchor. All of these are change detectors.

**Owner clarification received during this pass (rounding floor).** The 1/10-of-tolerance threshold never goes below 8 ulp of the quantity, which is about 2e-15 at unit scale. A mutant whose effect is at rounding level, and which is no worse than the clean code, is equivalent. I re-applied this to every ruling:
* **It changes none of the nine survivors.** Their effects are 3.55e-15 to 1.19e-13 on unit-scale M1, which is worse than the clean code (3.3e-15 vs quad at \|w\| = 5); 2.2e-13 on unit-scale states; and relative changes of 4e-9 to 100% (covariances) or 1.9e-7 to 3.1e-6 (Jacobians). All of these are far above rounding level.
* **Among the equivalents**, it additionally covers X01 under the stricter 1e-13 matrix reading: 1.78e-14 is still above 8 ulp of π (3.6e-15), so the C22 mapping remains the operative one. It also covers X13 (6.7e-16), X10 (3.6e-15 ≈ 8 ulp of π), X34 (3.6e-15 at det ≈ 1 to 7) and X51 (2.8e-14 on \|p\| ≈ 300, i.e. < 1 ulp).

**Input-domain sensitivity, for the owner.** I measured the effects over a broad sample of valid inputs, not only over the claims' own inputs (fixture edges, the C21 stream, default eps). If equivalence is instead judged on the claims' own inputs only, six of the nine become equivalent:

| mutant | effect on claim inputs | threshold |
|---|---|---|
| X15 | fixture edges 3.9e-16 / 5.6e-16 | < 1e-15 |
| P23 | 4.8e-8 | < 1e-7 |
| S21 | 4.4e-8 | < 1e-7 |
| W12 | 1.5e-8 | < 1e-7 |
| X41 | ≤ 1.1e-14 on orthonormal states | < 1e-13 |
| X21, X22 | no claim input uses per-axis zero or 1e-21 densities | — |

X24 and X31 stay non-equivalent on the fixture itself: 9.5e-9 and 4.0e-9 relative, against 2.7e-10.


| mutant | change | distinguishing input and effect | governing τ/10 |
|---|---|---|---|
| **X15** | M1 closed-form K² × (1 + 1e-13) | M1 vs `quad_vec`: EDGE_B-like twist (\|w\| = 0.77, T = 0.73) moves by 3.55e-15; \|w\| = 5, T = 1 by **1.19e-13**, which breaks C02's own 1e-14 (the clean code is 3.3e-15 vs quad there). Fixture edges: 3.9e-16 | C02 1e-14 → 1e-15 |
| **X21** | `has_noise` = all gyro axes > 0 | σ_g = [3e-4, 0, 0], σ_a = 0, cov = 0: the covariance stays **0** instead of the propagated gyro noise (100% relative), for B1 and B2 | C06 2.7e-9 rel → 2.7e-10 |
| **X22** | accel densities ≤ 1e-20 ignored | σ_a = 1e-21, σ_g = 0: covariance 0 instead of the propagated value (100% relative) | 2.7e-10 rel |
| **X24** | covariance transition `so3_exp(−w dt)` | B2 covariance moves by **9.5e-9 to 5.6e-8** relative (fixture e0 iso 9.5e-9, e1 aniso 2.3e-8, gyro-only time-varying 5.6e-8). B1 is unchanged, since Exp(−x) = Exp(x)ᵀ bitwise. B2's covariance is unclaimed; the governing τ is C06's, for the same Eq. (63) recursion | 2.7e-10 rel |
| **X31** | Ba velocity block × (1 + 2e-9) | B1 and B2 covariance move by **4.0e-9** relative on every noisy case. The tests' `COV_FD_REL` is 1e-7 | 2.7e-10 rel |
| **X41** | `navstate` projects R with `Rot3.ClosestTo` | B3 `predict` from `src`'s own B1-chained KF states (n = 1600, orthogonality error 1.8e-13) moves by **2.2e-13**. On orthonormal-to-rounding R it moves by ≤ 1.1e-14 | C03 1e-12 → 1e-13 |
| **W12** | recursion ignores `eps` | dt = 0.3, 10 steps: eps = 5e-2 gives **1.9e-7**; eps = 0.2 gives **3.1e-6** relative | C21 1e-6 rel → 1e-7 |
| **P23** | Forster `FD_EPS` 1e-3 | high-rate edges (ω = [3, −4, 0], T = 1 s; ω = [1.5, 0.5, −2], T = 1.5 s): B1 J **2.13e-7**, B2 factor J **2.31e-7** relative | 1e-7 |
| **S21** | factor FD eps 1e-3 | the same high-rate edges: B2 factor J **2.31e-7** | 1e-7 |

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over from pass 8, item by item:**
* Items 1-11 (c10a tautology, c13 `J_R.tobytes()`, M04, c12 isotropic half, c08 `≥ 1`, c25 ranks 30/39, c06 structure, c20 lower bound, c01, c22 upper bound) are unchanged. They are structural and correctly labelled.
* Item 12 (the self-referential F18 field list): **fixed**.
* Item 13 (the assertion that pinned the F23 gate): **fixed**; it was removed.
* Item 14: still present, by design. Three preconditions are on references only (`visible`, `rel(want, small)`, `moved`). A fourth is new: `moved >= 1e-2` in `test_f23`.
* Item 15 (W02 read only `.time()`): **fixed**.
* Item 16 (tr Σ = O(1) only): **fixed**.
* Item 17: still present. It now also covers `test_pass8_killers_are_reference_checks`, which checks literal node-id prefixes and `PASS8_REQUIRED ⊆` a literal list in the same file. It cannot fail unless that file's own literals change.
* Item 18 (change detectors): still present. They were not needed for any kill in this pass.

**New in this pass:**
* **`test_f25_so3_log_small_angles_against_mpmath`, the 0.75-ulp bound (Q44).** It cannot fail spuriously on this machine; it is deterministic. But it asserts correct rounding with a **1.17-1.18x** margin (clean 0.633-0.639 ulp over three seeds). No claim states this property. A legitimate change in evaluation order, an FMA contraction, or a numpy or BLAS change could push the clean code past 0.75. Under the rule Q44 is equivalent, so this assertion buys nothing for the gate. Recommendation: make it advisory, or remove it.
* **`test_f24_so3_log_near_pi`.** `warnings.simplefilter("error")` wraps the scipy `Rotation.from_rotvec`, `as_matrix`, `from_matrix` and `as_rotvec` calls as well as `so3_log`. A future scipy deprecation warning would turn the test into an uncaught exception, attributed to nothing in `src`. The `except Warning` branch (`# pragma: no cover`) only covers `so3_log`. Recommendation: narrow the filter to `so3_log` and to `RuntimeWarning`.
* **`test_b1_single_step_reads_every_field`'s `np.array_equal(pim.cov, 0)`** at zero densities and zero C0. It can fail, and does on several covariance mutants, but it cannot distinguish X19 (always propagate), which is exact at 0.
* **The covariance checks** (c06b, c06c, `b1_covariance_over_a_stream`, `b1_single_step_reads_every_field`, `test_f23`) use `COV_FD_REL = 1e-7` against an FD reference that is only accurate to about 2.5e-9. They cannot see changes in the 1e-9 to 1e-7 range that C06's own stated agreement (2.7e-9) covers. This is the cause of X31 and X24.

## 7. Per-claim verdicts

All numbers are mine. `src` was called only as the system under test.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations of the 400-sample stream: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 11 scalar/vector constants are exactly equal to the retyped literals. R_0 matches `expm` to 1.1e-16 |
| C02 | CONFIRMED | My retyped M1 vs `quad_vec`: 7.6e-17 (e0), 5.6e-17 (e1); M2 vs nested quad: 1.8e-16 / 9.7e-17. `src`'s M1/M2 give the same. Chained closed form vs the REF literals: KF1 v 6.7e-16, p 5.6e-17; KF2 R 2.2e-16, v 6.7e-16, p 1.7e-16. Chained vs my 5x5 `expm` at KF2: 3.3e-15 |
| C03 | CONFIRMED | GTSAM `predict` from NavState(Exp([0.3, −0.5, 0.2]), [3, 1, −4], [1, −2, 0.5]) vs my chain: R 8.9e-15 / 7.8e-15, v 8.9e-16 / 8.9e-16, p 0 / 2.8e-17 |
| C04 | CONFIRMED | Edge 0 ratios: v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086, 0.50043, 0.50022, 0.50011. Edge 1 v 0.50003 to 0.50000, p 0.50088 to 0.50011. `src` vs my Euler sums: 2.2e-14 (Rodrigues vs `expm`) |
| C05 | CONFIRMED | Error ratio between \|db\| = 1e-2 and 1e-3 against my own re-integration, R / v / p: with my FD J 100.05 / 99.98 / 99.98 (e0) and 100.21 / 100.01 / 100.00 (e1); with `src`'s J and correction, 100.05 / 99.99 / 100.00 and 100.21 / 100.00 / 100.00. `src` J vs mine: 2.4e-8 relative |
| C06 | CONFIRMED | σ_g = 1.7e-4, σ_a = 1e-2, at every step: asymmetry ≤ 1.7e-20, minimum eigenvalue −4.0e-23, trace nondecreasing, on both edges. Against my FD-linearised noisy Euler step: anisotropic 2.8e-10 / 2.7e-10, isotropic **2.46e-9** / 9.0e-10, worst block relative. That is ≤ 2.7e-9, as stated, but with only 1.1x margin on edge 0 isotropic |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 2.7e-15 (`src` and mine). (b) 300 φ with \|φ\| = 0.4, ε = 1e-7: right 5.6e-7 (FD-limited); left max 1.14, min 1.9e-2. "Fails by ≥ 1e-1" holds as a sup over draws, as in pass 8. `src` J_Cay vs retyped: 0.0 |
| C08 | CONFIRMED | 5000 ξ with \|ξ\| ≤ 5: deviation 3.6e-15 (numpy det, `src` det and closed form); minimum 1.00000009 |
| C09 | CONFIRMED | 500 draws: on-manifold 3.3e-16; perturbed by Exp(1e-3 u): minimum 6.4e-4 ≥ 1e-4 |
| C10 | CONFIRMED | `src` factor dv̄/dp̄ vs my Cayley Euler: ≤ 5.3e-14. Self-consistent states: v 3.6e-15 / 5.6e-14, p 1.3e-15 / 1.0e-14. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.4e-4 / 6.8e-4, each in [1e-4, 1e-2]. \|r^v\| matches \|Δv̄ − Δv_exact\| to 2.8e-16 / 5.0e-15 |
| C11 | CONFIRMED | 2000 draws: Eq. (19) vs my Eq. (22b), and vs `src`'s: 6.7e-16 |
| C12 | CONFIRMED | 2000 draws with scipy-`Rotation` R: isotropic 1.2e-14; W = diag(1, 2, 5) max 29.9 ≥ 1. With my `expm` R the isotropic gap was 1.0e-12, because `expm` at angles up to about 6 rad is orthogonal only to 1.4e-13. That is a reference artefact, not a counterexample: R is then not in SO(3) to double precision |
| C13 | CONFIRMED | (a) `J_R` bytes unchanged after `predicted_rotation` and `rotation_residual` at a perturbed bias. (b) At the Eq. (16) point R_j = R_i dR̄ Cay(ξ), with \|ξ\| = 1e-2 over 200 directions, exact vs truncated: min **3.1e-5**, max 4.8e-5, ≥ 1e-6. Wording note: the difference is ½\|(R_i dR̄ − R_j) ξ^\|, so it depends on R_j. At the fixture ground-truth KF1 it is 9.7e-11, and at R_j = R_i dR̄ it is 0. The claim holds at the factor's own predicted state, which is what the hint builds; it is not universal over R_j |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7.0; `position()` = [1, 2, 3], `velocity()` = [10, 20, 30], `translation()` = `position()`; the `Gal3(M)` round trip gives 0.0 |
| C15 | CONFIRMED | `Gal3.Hat(e_k)` equals my 5x5 hat exactly for all 10 k. `Expmap(1e-6 e_k)` moves only the stated component, for all k. Expmap vs `expm(Hat)`: 1.4e-14 over 50 random ξ. `src` remap = (0, 6, 3) |
| C16 | CONFIRMED | KF0-KF2: **0.107628** (KF0-KF1 0.1008, KF1-KF2 0.1678). Remapped log vs `NavState.Logmap`: 0.0 on all three pairs. GTSAM Gal(3) log vs my `logm`: ≤ 1.8e-15 |
| C17 | CONFIRMED | `ConstantBias([1,2,3], [4,5,6])` accessors and `vector()` are as stated. `biasHatVector` = [b_a; b_g] exactly. The adapters round-trip exactly; one application moves the vector by 0.082; `swap` is an involution |
| C18 | CONFIRMED | My 5x5 products on both edges. W-form error ≤ 2.2e-16 at t_i = 0 and 3.7, with t_j = 0.5 / 4.2. F2 form error: ≤ 5.6e-17 at t_i = 0, and **18.1485 = \|g\|·0.5·3.7** exactly at t_i = 3.7. Swapped order: 2.4525 = \|g\| Δt² at t_i = 0. `src` forms vs mine: ≤ 8.9e-16 |
| C19 | CONFIRMED | Raw GTSAM PIM-G vs my ZOH: R 1.0e-14, v 2.0e-14, p 3.8e-15 (e0); 8.4e-15 / 1.7e-14 / 3.1e-15 (e1). Tangent dv: 2.17e-3 / 2.76e-3 ≥ 1e-4. `src` B3 vs ZOH: 2.0e-14. Time-varying stream: right 1.1e-16; left, one step at k = 40, 4.6e-5 ≥ 1e-5; GTSAM vs my ZOH chain 3.3e-16 |
| C20 | CONFIRMED | My FD J (eps 1e-7, my `logm`, GTSAM `predict` at g = 0): J[0:3,0:3] = J[9,:] = 0.0. Right composition: 2.8e-13 to 1.4e-10 (e0) and 3.1e-13 to 1.5e-10 (e1), linear in \|db\|, ≤ 1e-8. Component-wise at 5e-2: 30 directions 1.27e-2 / 1.25e-2; 200 random 1.24e-2 / 1.47e-2; top singular direction **1.40e-2 / 1.62e-2** ≥ 1e-2. `src` J vs mine: 3.6e-9 / 3.7e-9 relative |
| C21 | CONFIRMED | My recursion (my V_k, conjugation adjoint, FD C_k) vs my FD re-integration over 40 steps: 2.6e-8 relative (\|J\| = 0.10). `src` recursion vs my FD: 7.2e-8 |
| C22 | CONFIRMED | KF2 B1-B2 (`src`): v 6.17e-8, p 4.38e-8, R max-abs **2.9965e-8**; mine identical to 6e-14. Per-edge ‖Log‖₂ 2.325744e-8 / 5.517531e-8 vs formula 2.325744e-8 / 5.517533e-8 (rel 1.9e-7 / 3.3e-7). dt² ratios 0.25000, 0.25000, 0.25000, 0.25002. KF2 gap 0.12% from 3.0e-8 |
| C23 | CONFIRMED | B1-B3: v 1.2867e-3, p 1.2031e-3, R 3.1e-15. B2-B3: R 3.0e-8. My ratios: B1 v 0.50028, 0.50014, 0.50007, 0.50003 and p 0.50000 ×4; B2 v 0.50026 to 0.50003, p 0.49998 to 0.50000. `src` at n = 100 / 1600 matches mine to 1e-15 |
| C24 | CONFIRMED | B3 vs my 5x5 exact states: KF1 R/v/p 8.9e-16 / 1.6e-14 / 2.1e-15; KF2 1.1e-15 / 4.7e-15 / 5.1e-15. B1 R: 9.5e-15 / 2.7e-15. B2 R: 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My own graph and densities. Shapes and ranks: 30x45 → 30, 39x45 → 39, 45x45 → 45, plus bias prior 51x45 → 45. Principal cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**, pose/velocity energy 2.5371. Graph error 2.1e-23; unwhitened IMU residuals 1.7e-14 / 2.0e-14; s_min/s_max 4.0e-9 |
| **C32** | **REFUTED** | Sec. 2. At θ = π − 1e-6 the error is **2.06e-4**, 2e6 times the 1e-10 bound; it is already 2.66e-10 at π − 1e-3. In 50 digits the formula is exact, so this is floating-point conditioning |

**Counts: 25 CONFIRMED, 1 REFUTED (C32), 0 UNTESTABLE.**

## 8. Final state

This commit contains the report only.

Full suite on the clean tree (`7467b0c`, i.e. `8f99912` plus this report): **232 passed** in 364.4 s. At the start of the pass it was 232 passed in 64.5 s; the machine's load average later rose to 60-185.

`git status` of the worktree was clean before and after every sweep, and every mutant lived in a scratchpad copy.

**Gate: NOT passed.** 9 non-equivalent survivors remain, all low: X15, X21, X22, X24, X31, X41, W12, P23 and S21.
