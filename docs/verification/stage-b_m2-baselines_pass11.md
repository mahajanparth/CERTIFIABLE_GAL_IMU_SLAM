# Stage-B verification, pass 11: M2 baselines (`stage-b/m2-baselines` @ `23a73bf`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration. Agent: verify-M2-011.

**Scope of this pass:**
* a ruling on M2-F28: the pass-10 survivors and housekeeping items, answered by `2cc0762..23a73bf`. That range includes a **src change**: `so3_right_jacobian`'s K coefficient is now `2 sin^2(th/2)/th^2`;
* verdicts on M2-C01..C25, plus M2-C34, which this pass's brief supplies and the yaml does not contain;
* a check that the C18 edit in `23a73bf` touches only `test_hint`;
* the ten pass-10 survivors replayed as real file edits;
* 55 new mutants (Z01-Z55), each a real file edit;
* a replay of `tests/test_mutation.py`.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing named `*SEEDS*`. I did not browse the git history of `docs/claims`. The only history I diffed is the range the brief named (`d5a86ac..23a73bf`).

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`23a73bf`) | full suite **265 passed** in 242.6 s (load average 14-30) |
| `so3_right_jacobian` src change | **correct.** Error against mpmath is ≤ 3.5e-16 absolute at every angle from 1e-10 to 6 rad, including the 1e-8 series switch and near π. It moves no claim-level number beyond 2.2e-14 relative (Sec. 3) |
| C18 yaml edit | only `test_hint` changed; the statement is byte-identical. The hint numbers reproduce (Sec. 4) |
| **M2-F28** | **FIXED.** All 10 pass-10 survivors are killed as real file edits, each by its intended reference test. All housekeeping items are done |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C34), 0 UNTESTABLE |
| New mutants (55) | 32 killed by reference tests; 17 equivalent (numbers in Sec. 5.3); **6 non-equivalent survivors** |
| `tests/test_mutation.py` replay | 154/154 pass on the clean tree. This includes the 10 PASS10 file-edit replays and their reference-check guard |
| **Gate** | **NOT passed.** 6 non-equivalent survivors: Z07, Z08, Z09, Z10, Z11, Z30. They sit just outside the per-step angle band the F28 tests cover, plus one weak C11 sample (finding **M2-F29**) |

## 1. Independent references (no `src` imports)

My reference code lived in `results/p11` (gitignored) and was deleted before commit. It never imports `src` or `tests`; `src` was called only as the system under test.

* **Rotations:** `Exp` = scipy `expm(hat)` and a retyped Rodrigues; `Log` = scipy `Rotation`.
* **Jacobians:**
  * `J_r` as the power series Σ(−K)^k/(k+1)! in mpmath at 40-50 digits;
  * a central difference of `mpmath.expm` at h = 1e-20, as a cross-check (agrees with the series to 1e-31);
  * a numpy power series as a third copy.
* **Exact increments:** 5x5 `expm` ZOH.
* **Euler sums:** retyped Eq. (9a)-(9c).
* **Complex-step references (my own):** the complex covariance and bias Jacobians use scipy `expm` on a complex hat for `Exp`, a `solve`-based complex Cayley, and h = 1e-30. This is not the test file's scalar series.
* **GTSAM side:** my own 5x5 Gal(3) matrices, `logm` logarithms, and an FD bias Jacobian on `predict`.
* **C25:** a graph of my own, with densities different from the tests' (σ_g 3e-4, σ_a 2e-2, bias walks 3e-5 / 5e-4, integration 1e-5, prior σ 1e-2, GNSS σ 0.05).
* **C18:** rational arithmetic (`fractions`) for the rounding ruling.

**Re-implemented core routines:**
* `so3_right_jacobian`, three ways;
* the Eq. (63) covariance, by complex step;
* the Eq. (9) sums;
* Eq. (22b);
* the Gal(3) bias Jacobian and its recursion check.

## 2. M2-C34 (brief-only claim): REFUTED

**Claim:** at |φ| ≤ 4e-3, `Jr(φ) Jr(φ)ᵀ = I` within 1e-12 entrywise, so `Jr` leaves an isotropic rotation-noise covariance unchanged and "only reorients" it.

**Attack: `Jr` is not orthogonal, even to first order in the perturbation of the identity.** Write `Jr = I − aK + bK²` with `a = (1 − cos θ)/θ²` and `b = (θ − sin θ)/θ³`. Then `Jrᵀ = I + aK + bK²`, and

`Jr Jrᵀ = I + (2b − a²) K² + b² K⁴`, with `2b − a² = 1/12 + O(θ²)`.

So `Jr Jrᵀ ≈ I − (θ²/12)(I − â âᵀ)`. Along the axis the eigenvalue is 1. Transverse to it the eigenvalue is `2(1 − cos θ)/θ² = 1 − θ²/12 + …`. **This is a contraction, not a rotation.**

**Measured** (src's `so3_right_jacobian`; mpmath at 40 digits agrees):

| \|φ\| | max\|Jr Jrᵀ − I\| | θ²/12 |
|---|---|---|
| 4e-3 | **1.3333e-6** | 1.3333e-6 |
| 1.1e-3 (fixture per-step) | 1.0083e-7 | 1.0083e-7 |
| 1e-3 | 8.33e-8 | 8.33e-8 |
| 1e-4 | 8.33e-10 | 8.33e-10 |
| 1.2e-6 | 1.2e-13 | 1.2e-13 |

* **Property test:** 10,000 random φ with \|φ\| ≤ 4e-3. **All 10,000 violate the 1e-12 bound.** The worst case is 1.333e-6, 1.3e6 times the bound.
* The ratio to θ²/12 lies in [0.68, 1.00] (the extremes are entries vs eigenvalue).
* The bound holds only for \|φ\| ≲ 3.5e-6.
* **Consequence for the covariance claim:** `Jr σ²I Jrᵀ` differs from `σ²I` by 1.33e-6 relative at 4e-3 rad, and by 1.0e-7 at the fixture's 1.1e-3 rad. So `Jr` does change an isotropic covariance, at second order in the per-step angle. Dropping `Jr` (using `I`) is a 1e-7-relative error per step on the fixture, not a no-op. This is the same order as M2-C33's refuted gap in pass 10.
* Src's `Jr` agrees with my independent numpy series to 2.2e-16 on all 10,000 samples, so the refutation is about the claim, not about the code.

## 3. Ruling on M2-F28: FIXED

### 3.1 The src change: `Jr`'s K coefficient now `2 sin²(θ/2)/θ²`

**Correctness: 10,000-sample property test against mpmath (34 digits).** The angles are log-uniform in [1e-9, π), plus 400 samples within ±5% of the 1e-8 series switch and 400 at π − 10^[-12, -2].

| result | value |
|---|---|
| max abs error, all samples | **3.3e-16** (at θ = 3.1398) |
| around the switch | ≤ 8.9e-18 |

A second sweep used the 50-digit power series, 4 directions at each of 71 angles from 1e-9 to 6 rad (beyond π included). The new form is ≤ 3.5e-16 everywhere. The old form `(1 − cos θ)/θ²` is up to:

| decade (θ) | old-form error |
|---|---|
| 1e-8 | 4.7e-9 |
| 1e-7 | 1.3e-10 |
| 1e-6 | 3.5e-11 |
| 1e-5 | 2.3e-12 |
| 1e-4 | 1.2e-13 |
| ≥ 1e-2 | ≤ 1.6e-15 |

* Near π both forms are at rounding (3.5e-16). So the change cures a real cancellation and introduces nothing.
* The docstring's figures (3.0e-9 at 1.1e-8, ≤ 4.9e-13 at 1e-4) are consistent with mine.
* The `K²` coefficient `(θ − sin θ)/θ³` still cancels, but it multiplies \|K²\| = θ², so its absolute error stays at rounding. My sweep confirms this (no excess error at any angle). The docstring's argument is right.

**Claim-level effect, old form vs new form.** I patched the old form into the module namespace of both B1 and B2. Measured on:
* the fixture covariances (B1, B2-raw, B2-reparam; isotropic and anisotropic densities, both edges);
* the time-varying-stream covariances.

| quantity | effect |
|---|---|
| covariance | ≤ **2.2e-14** block-relative, against C06's 2.7e-9 |
| bias Jacobians, increments, factors | 0.0 (`Jr` does not enter them) |

**No claim number moves.**

**Is the fix pinned?**
* Z01 (revert to the old form) is killed by `test_f28_right_jacobians_against_mpmath_across_angles`.
* Z02 (old form only above 1e-3, where the two differ by ≤ 2.7e-14) survives, and is equivalent (Sec. 5.3).

### 3.2 The ten pass-10 survivors, replayed

Each is a real, exactly-once textual edit on a temp copy, retyped from the pass-10 description rather than copied from `PASS10_MUTANTS`, and judged by the fast set (111 reference tests):

| mutant | killed? | by |
|---|---|---|
| Y18 (`Jr` K² × (1 + 1e-8)) | **killed** | `test_f28_right_jacobians_against_mpmath_across_angles`, `test_f28_covariance_across_per_step_angles_against_complex_step` |
| Y20 (`has_noise` ≤ 1e-100 ignored) | **killed** | `test_f28_has_noise_is_any_positive_density_on_any_axis` |
| Y58 (accel ≤ 1e-60 ignored) | **killed** | `test_f28_has_noise_is_any_positive_density_on_any_axis` |
| Y24 (B2 raw `J_r` below 1e-4) | **killed** | `test_f28_covariance_across_per_step_angles_against_complex_step` |
| Y55 (B2 raw `J_r` above 1e-2) | **killed** | same |
| Y56 (B1 `I − K/2` above 1e-2) | **killed** | same |
| Y57 (`A_RR = I` below 1e-4) | **killed** | same |
| Y30 (A_va guarded by `db_g`) | **killed** | `test_f28_eq21b_eq21c_every_combination_of_bias_halves` |
| Y59 (A_pa guarded by `db_g`) | **killed** | same |
| Y31 (Eq. (21c) `v_i Dt` × (1 + 1e-13)) | **killed** | `test_f28_eq21c_at_large_velocity_and_duration` |

**All 10 are killed, and every killer is a reference check** (complex step, mpmath definitional derivative, or retyped residual). The margins are from the tests' own measurements, which my runs are consistent with:

| mutant | effect | bound | margin |
|---|---|---|---|
| Y18 | 7.8e-9 | 1e-14 | robust |
| Y24 | 1.35e-9 | 1e-10 | 13x |
| Y31 | 8.3e-12 | 1e-12 | 8x |
| Y20, Y58 | flag, and covariance 100% off | — | robust |
| Y55, Y56, Y57 | ≥ 3.8e-7 | 1e-10 | ≥ 3800x |
| Y30, Y59 | ≥ 0.18 abs | 1e-12 | robust |

### 3.3 Housekeeping items

| item | status |
|---|---|
| Stale FD_EPS comment (`test_baselines.py:1111`) | **fixed**. It now says 3e-5 and gives the reason |
| `test_c05b`(c) reference moving with `B1.FD_EPS` | **fixed**. It uses the literal `GTSAM_JR_REF_FD_EPS = 1e-5`, so an FD-step mutant can no longer move the reference |
| Cannot-fail preconditions `orth >= 1e-11` / `visible >= 1e-7` | **removed**. Replaced by comments that point to the X41 / W12 file-edit kills |
| P23-old margin | **labelled**. Now documented as equivalent (4.4e-8 < 1e-7), and not in `PASS9_REQUIRED` (checked) |
| C18 test hint | **fixed**; only the hint changed (Sec. 4) |

## 4. Per-claim verdicts

All numbers are mine.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 12 checked constants equal the retyped literals exactly. R_0 vs `expm`: 1.1e-16 |
| C02 | CONFIRMED | Retyped M1/M2 vs `quad_vec` (1e-14/1e-14): 7.6e-17 / 1.7e-16 (e0) and 5.6e-17 / 1.1e-16 (e1); `src` identical. Literals vs my 5x5 `expm` chain: ≤ 3.3e-15. `src` `exact_states` vs literals: KF2 R/v/p 1.1e-16 / 8.6e-16 / 2.2e-16, KF1 v/p 4.4e-16 / 3.5e-17 |
| C03 | CONFIRMED | GTSAM `predict` from NavState(Exp([0.3, −0.5, 0.2]), p, v) vs the formula: R 8.9e-15 / 7.8e-15, v 8.9e-16, p 0.0 |
| C04 | CONFIRMED | n = 100..1600. e0 v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086 → 0.50011. e1 v 0.50003 → 0.50000; p 0.50088 → 0.50011. `src` vs my Euler: ≤ 2.2e-14 |
| C05 | CONFIRMED | R/v/p ratios: 99.89 / 100.01 / 100.01 (e0), 99.88 / 99.97 / 99.97 (e1). FD J vs my complex step: 3.3e-10 / 6.7e-10 relative |
| C06 | CONFIRMED | σ_g 1.7e-4, σ_a 1e-2, every step: asymmetry ≤ 1.7e-20, min eigenvalue −4.0e-23, trace nondecreasing; anisotropic likewise. Vs my complex step: 1.6e-12 .. 2.1e-12 block-relative. Vs GTSAM after M: RR 2.3e-14 / 3.5e-15, Rv 5.04e-3 / 5.03e-3, Rp 7.53e-3 / 7.52e-3; untransformed Rv 0.170 / 0.195 |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 1.2e-15. (b) 300 φ with \|φ\| = 0.4, ε = 1e-7: right 5.0e-8; left max 0.38, min 1.6e-2. "Fails by ≥ 1e-1" holds as a sup over draws, as in passes 8-10 |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 3.6e-15; minimum value 1.00000004 |
| C09 | CONFIRMED | 500 draws: on-manifold 2.6e-16; perturbed min 6.1e-4 ≥ 1e-4 |
| C10 | CONFIRMED | Factor dv̄ vs my Cayley Euler: 0.0. Self-consistent: ≤ 8.9e-16. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.38e-4 / 6.81e-4, all in [1e-4, 1e-2]; \|r\| vs \|Δv̄ − Δv_exact\| ≤ 8e-16 |
| C11 | CONFIRMED | 2000 draws: 4.4e-16. But see Z30 (Sec. 5.4): the test's own sample is too narrow to see a 1e-12 relative slip in Eq. (22b) |
| C12 | CONFIRMED | 10,000 draws with Rodrigues rotations: isotropic 2.1e-14; W = diag(1, 2, 5) max 47.7 ≥ 1. (With scipy `expm` rotations at angles up to ~7 rad the isotropic gap reads 7.3e-13. That is `expm`'s own orthogonality error, not `src`'s) |
| C13 | CONFIRMED | (a) `J_R` bytes unchanged after `predicted_rotation` and `rotation_residual` at a perturbed bias. (b) 200 directions at \|ξ\| = 1e-2: min 3.06e-5, max 4.80e-5 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7; accessors as stated; `translation()` = `position()`; `Gal3(M)` round trip 0.0 |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. `Expmap(1e-6 e_k)` moves only the stated block. Expmap vs `expm(Hat)`: 3.1e-15 over 50 ξ |
| C16 | CONFIRMED | KF0-KF2 **0.107628**. Remapped `logm` vs `NavState.Logmap`: 1.8e-15 |
| C17 | CONFIRMED | Accessors and `vector()` as stated; `biasHatVector` = [b_a; b_g]. Adapters round-trip exactly; one swap moves the vector by 0.082 |
| C18 | CONFIRMED | My 5x5 products, against the component relation with the same increment. W form: ≤ 2.2e-16 (Dt = 0.5) and ≤ 1.3e-15 (span, Dt = 1.0). F2 form: 0 at t_i = 0; at t_i = 3.7, **18.148500** per edge and **36.297000** on the span, equal to \|g\| Dt t_i exactly. Swapped order: 2.4525 (Dt = 0.5) and 9.81 (Dt = 1.0) at t_i = 0. `src` (GTSAM increments): W ≤ 2.0e-15 per edge and 8.9e-16 on the span; F2 18.148500 / 36.297000. **Yaml diff `d5a86ac..23a73bf`: only `test_hint` lines change; the statement is byte-identical.** The new hint numbers reproduce. The hint's "≤ 8.0e-15 on the span" is a loose upper bound: I measure 8.9e-16 |
| C19 | CONFIRMED | PIM-G vs ZOH: R 1.0e-14, v 2.0e-14, p 3.8e-15 (e0); 8.4e-15 / 1.7e-14 / 3.1e-15 (e1). Tangent dv 2.171e-3 / 2.760e-3 ≥ 1e-4. Time-varying: right composition vs GTSAM 1.1e-16; left 6.2e-4 ≥ 1e-5 |
| C20 | CONFIRMED | My FD J (ε = 1e-7, `logm`, GTSAM `predict` at g = 0): J[0:3,0:3] = J[9,:] = 0.0. Right composition ≤ 1.9e-10 / 2.1e-10. Component-wise at 5e-2 over 30 random directions: 1.09e-2 / 1.31e-2 ≥ 1e-2. `src` J vs mine: 4.4e-9 |
| C21 | CONFIRMED | `src` recursion vs my FD re-integration at k = 1, 10, 20, 40: 4.6e-8 relative. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1-B2: v 6.1672e-8, p 4.3763e-8, R max-abs **2.9965e-8**. Per edge 2.325744e-8 / 5.517531e-8 vs the formula (relative 1.9e-7 / 3.3e-7). dt² ratios 0.25000 x3, 0.24999 |
| C23 | CONFIRMED | B1-B3: v 1.2867e-3, p 1.2031e-3, R 3.1e-15; B2-B3 R 3.0e-8. Ratios: B1 v 0.50028 → 0.50003, p 0.5000 x4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.50000 |
| C24 | CONFIRMED | B3 vs exact: KF1 8.9e-16 / 1.6e-14 / 2.1e-15; KF2 1.1e-15 / 4.7e-15 / 5.1e-15. B1 R 9.5e-15 / 2.7e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My graph and densities: ranks 30 (30x45), 39 (39x45), 45 (45x45). Principal cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**; pose/velocity energy 2.5371. Both are density-independent, as stated. Graph error 2.1e-23; unwhitened IMU residuals 1.7e-14 / 2.0e-14 |
| **C34** | **REFUTED** | Sec. 2. Max\|Jr Jrᵀ − I\| = **1.33e-6** at \|φ\| = 4e-3 (= θ²/12), against the 1e-12 bound; 10,000 of 10,000 samples violate it. `Jr` contracts the transverse directions; it does not merely reorient |

**Counts: 25 CONFIRMED, 1 REFUTED (C34), 0 UNTESTABLE.**

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh `tempfile` copy of `src tests orchestrator docs conftest.py pyproject.toml`. Every anchor is asserted to occur exactly once.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (111 tests). All of them are reference checks; the set excludes the change detectors in `test_mutation`.
* Every fast-set survivor that is not equivalent was re-run on the **full suite** (265 tests). An unmutated control copy passed 265/265.
* The worktree's tracked files never changed.

### 5.1 New mutants Z01-Z55

| id | change | result |
|---|---|---|
| Z01 | `Jr` K coefficient reverted to `(1 − cos θ)/θ²` | killed: f28_right_jacobians |
| Z02 | old K form only for θ > 1e-3 | survives; **equivalent** |
| Z03 | `Jr` K coefficient × (1 + 1e-12) | killed: f28_right_jacobians |
| Z04 | `Jr` series branch up to 1e-4 | killed: f28_right_jacobians |
| Z05 | `Jr` series branch up to 3e-6 | survives; **equivalent** |
| Z06 | `Jr` K term uses Kᵀ | killed: c06b |
| Z07 | B1 noise input `I − K/2` when \|w dt\| > 0.6 | **survives; NOT equivalent** |
| Z08 | A_RR = I when \|w dt\| < 1e-8 (B1 and B2) | **survives; NOT equivalent** |
| Z09 | A_RR = I when \|w dt\| > 0.6 | **survives; NOT equivalent** |
| Z10 | Bg uses I instead of `Jr`/`J_Cay` when \|w dt\| < 1e-8 | **survives; NOT equivalent** |
| Z11 | B2 raw uses `J_r` when \|φ\| > 0.6 | **survives; NOT equivalent** |
| Z12 | gyro `has_noise` threshold 1e-300 | killed: f28_has_noise |
| Z13 | gyro `has_noise` uses `np.all` | killed: f27_has_noise |
| Z14 | gyro `has_noise` uses `>= 0` (always true) | survives; **equivalent** |
| Z15 | covariance gate `np.all(cov != 0)` | killed: f23 |
| Z16 | Bg = Jrᵀ dt | killed: c06b |
| Z17 | Ba position block uses `dR_ik dR_step` | killed: c06b |
| Z18 | covariance uses `dR_{k+1}` in place of `dR_k` | killed: c06b |
| Z19 | Qa / (dt (1 + 1e-9)) | killed: b1_single_step |
| Z20 | A_pv × (1 + 1e-9) | killed: f23 |
| Z21 | J_Cay denominator ‖φ‖²/4 × (1 + 1e-8) | killed: b2_cayley_maps |
| Z22 | `cay` = solve(I + P, I − P) | killed: c05b |
| Z23 | Eq. (17) det(I + ξ̂/2) | survives; **equivalent** |
| Z24 | Eq. (19) ½ × (1 + 1e-12) | killed: c11 |
| Z25 | Eq. (21b) g Dt × (1 + 1e-13) | killed: f28_eq21c_large_v |
| Z26 | Eq. (21c) ½ g Dt² × (1 + 1e-13) | killed: f28_eq21c_large_v |
| Z27 | Eq. (21c) p_i × (1 + 1e-13) | killed: f28_eq21c_large_v |
| Z28 | `weighted_square` with Wᵀ | survives; **equivalent** |
| Z29 | `from_edge` Dt = `edge.duration` | survives; **equivalent** |
| Z30 | Eq. (22b) Rbreve's Γ/2 × (1 + 1e-12) | **survives (reference tests); NOT equivalent** |
| Z31 | `cayley_vector` × (1 + 1e-14) | killed: f25_series |
| Z32 | `cayley_vector` identity branch below 1e-5 | killed: f25_series |
| Z33 | `so3_log` near-π column index j = 0 | killed: f24_so3_log_near_pi |
| Z34 | `so3_log` sign test `<=` | survives; **equivalent** |
| Z35 | `LOG_NEAR_PI_COS` = −0.95 | survives; **equivalent** |
| Z36 | M1 series drops K² term | killed: f25_series |
| Z37 | M2 series drops K² term | killed: f25_series |
| Z38 | `so3_exp` K² coefficient as 2 sin²(θ/2)/θ² | survives; **equivalent** |
| Z39 | `so3_log` θ × (1 + 1e-13) | survives; **equivalent** |
| Z40 | B3 dt passed as float32 | killed: c06b |
| Z41 | `integrate_stream` drops last step | killed: b3_integrate_stream |
| Z42 | bias-acc covariance × (1 + 1e-10) | killed: b3_make_params |
| Z43 | B3 FD J in left chart (plus side) | killed: c20 |
| Z44 | `endpoint_W` rearranged −½gDt² − gDt t_i | survives; **equivalent** |
| Z45 | `gal3_distance` ignores time component | killed: b3_models_and_distance |
| Z46 | component-wise model ξ × (1 + 1e-9) | killed: b3_models_and_distance |
| Z47 | recursion C_k × (1 + 1e-8) | killed: f27_recursion |
| Z48 | B3 `predict` uses `pim.biasHat()` | survives; **equivalent** |
| Z49 | Eq. (44) rotation correction × (1 + 1e-7) | killed: c05 |
| Z50 | B1 `J_R` from Log(R₋ᵀR₊)/(2ε) | killed: b1_bias_jacobians_fd |
| Z51 | `predict` ½gDt·dt_total | survives; **equivalent** |
| Z52 | `increment_error` Log(R_aᵀR_b) | survives; **equivalent** |
| Z53 | `Jr` θ = sqrt(φ·φ) | survives; **equivalent** |
| Z54 | symmetrise `cov` after each step | survives; **equivalent** |
| Z55 | `Jr` K coefficient as ½ sinc²(θ/2) | survives; **equivalent** |

**Count (55):** 32 killed by reference tests, 17 equivalent, **6 not equivalent and surviving**.

### 5.2 `tests/test_mutation.py` replay

On the clean tree, all 154 tests in `test_mutation.py` pass (inside the 265). That includes:
* `test_pass10_mutant_is_killed_as_a_file_edit` [Y18 … Y31], which ran its control copy and the 10 file-level edits, each checking that the kill lands on the intended `E` line;
* `test_pass10_killers_are_reference_checks`.

My independent replay (Sec. 3.2) agrees with every one of them.

### 5.3 Equivalence rulings (owner's rule; threshold = max(τ/10, 8 ulp))

**How I measured E.** Each surviving mutant ran a probe over a broad input sample. The sample covered:
* B1, B2-raw and B2-reparameterized covariances and increments at per-step angles 0, 1e-10, 3e-9, 9e-9, 2e-8, 1e-7, 1e-5, 1e-3, 0.1, 0.5, 0.7, 1, 2 and 3 rad (12 steps), and at 5e-9 and 1e-9 rad over 4000 steps, with anisotropic gyro alone and gyro + accel noise;
* fixture covariances, predicts, factor fields and factor residuals;
* `Jr`, `Exp`, `Log`, `J_Cay` and `cay` at 82 angles from 1e-10 to π;
* 2000 `det` values, 2000 weighted squares, 2000 Eq. (22b) − (19) differences;
* 200 `increment_error` values;
* `endpoint_W` at Dt ∈ {0.5, 1, 2} and t_i ∈ {0, 3.7, 1e3};
* B3 `predict`.

E is the largest change over that sample.

| mutant | quantity Q | τ (claim) | threshold | measured E | ruling |
|---|---|---|---|---|---|
| Z02 | `Jr` → covariance | C06 2.7e-9 rel | 2.7e-10 | `Jr` 2.7e-14 abs; covariance 8.3e-15 block-rel (fixture 1.9e-15) | equivalent |
| Z05 | `Jr` → covariance | C06 | 2.7e-10 | `Jr` 1.1e-16 abs; covariance 6.8e-16 | equivalent |
| Z14 | covariance | C06 | 2.7e-10 | 0.0 on every output. **Proof:** with all densities 0 and `cov` = 0, `A·0·Aᵀ + Bg·0·Bgᵀ + Ba·0·Baᵀ` = 0 exactly; only the skip is lost | equivalent |
| Z23 | Eq. (17) det | C08 1e-12 | 1e-13 | 7.1e-15 abs. **Proof:** det(I + ξ̂/2) = det((I − ξ̂/2)ᵀ) | equivalent |
| Z28 | weighted square | C12 1e-12 | 1e-13 | 0.0. **Proof:** rᵀWᵀr = rᵀWr | equivalent |
| Z29 | factor Dt → residuals | C10 1e-12 | 1e-13 | residuals 3.6e-15 abs; Dt 3.3e-16 | equivalent |
| Z34 | `so3_log` | — | — | 0.0. **Proof:** differs only when a·s = 0, i.e. s = sin θ·a = 0 at θ = π, where both signs are valid logarithms | equivalent |
| Z35 | `so3_log` (generic branch now used for 2.09 < θ < 2.82) | C22 1e-3 rel | 1e-4 rel | 2.2e-16 abs | equivalent |
| Z38 | `Exp` → B1 R, covariance | C24 1e-10; C06 | 1e-11; 2.7e-10 | predicts 1.9e-14 abs; covariance 2.4e-14 | equivalent |
| Z39 | `Log` (and FD J_R through it) | C22 1e-3 rel; C21 1e-7 rel | 1e-4; 1e-8 | 1.0e-13 rel (1.7e-13 abs) | equivalent |
| Z44 | `endpoint_W` | C18 1e-12 | rounding | vs exact rationals: clean and mutant both ≤ 1 ulp, sup 1 ulp each (3.6e-12 at magnitude 2e4) | equivalent (rounding floor) |
| Z48 | B3 `predict` | C03 | — | 0.0. **Proof:** `bias_hat` is read from `pim.biasHat()` | equivalent |
| Z51 | B1/B2 `predict` | C03 1e-12 | 1e-13 | 2.2e-16 abs (Dt = dt_total, so Dt·dt_total = Dt²) | equivalent |
| Z52 | `increment_error` | C05 | — | 0.0. **Proof:** Log(Rᵀ) = −Log(R) for θ < π; the max-abs is sign-blind | equivalent |
| Z53 | `Jr` | — | — | 0.0 | equivalent |
| Z54 | covariance | C06 | 2.7e-10 | 1.8e-14 block-rel | equivalent |
| Z55 | `Jr` → covariance | C06 | 2.7e-10 | 7.4e-16 | equivalent |

### 5.4 Complete survivor list (non-equivalent)

All 6 were re-run on the full suite (Sec. 5.5). None changes any claim's recorded number on the claim's own inputs.

| mutant | change | distinguishing input and effect (block-relative covariance unless stated) | threshold |
|---|---|---|---|
| **Z07** | B1 noise input `I − K/2` when \|w dt\| > 0.6 | B1: **0.17** at 0.7 rad/step, 0.38 at 1.0, 1.8 at 2.0, 7.6 at 3.0. The largest tested per-step angle is 0.5 | C06 2.7e-10 |
| **Z09** | A_RR = I when \|w dt\| > 0.6 | B1, B2-reparam: **3.6** at 0.7 rad/step; B2-raw 3.1 | 2.7e-10 |
| **Z11** | B2 raw `J_r` when \|φ\| > 0.6 | B2-raw: **8.6e-2** at 0.7 rad/step, 0.79 at 3.0 | 2.7e-10 |
| **Z08** | A_RR = I when \|w dt\| < 1e-8 (B1 and B2) | **7.7e-6** at 5e-9 rad/step over 4000 steps (a 10 s stationary IMU at 400 Hz with \|w − b_g\| = 2e-6 rad/s); 1.5e-6 at 1e-9 x 4000; 1.1e-8 at 3e-9 x 12. The smallest tested per-step angle is 1e-7 | 2.7e-10 |
| **Z10** | Bg = I·dt in place of `Jr`·dt (`J_Cay`·dt for B2-raw) when \|w dt\| < 1e-8 | **1.0e-8** at 9e-9 rad/step; 3.4e-9 at 3e-9; 3.5e-9 at 5e-9 x 4000 | 2.7e-10 |
| **Z30** | Eq. (22b): `Rbreve = dRbar (I − Γ/2·(1 + 1e-12))` | `rotation_residual_eq22b` vs Eq. (19): **6.6e-14 abs on C11's own distribution** (20,000 draws, \|J_R\| ≤ 1, \|b̄_g\| ≤ 0.05; clean 6.7e-16), below `test_c11`'s 1e-13 bound. With \|J_R\| ~ N(0, 1) and \|b̄_g\| ~ 0.1: 4.8e-13. On the fixture factor: 3.9e-15 | C11 1e-13 → 1e-14 |

**Why these survive.**
* **Z07, Z09, Z11 and Z08, Z10** sit just outside `F28_STEP_ANGLES` = [1e-7, 0.5] rad. That band is exactly the band pass 10 recommended. The domain is wider at both ends:
  * nothing restricts the per-step angle of B1, or of B2's Cayley retraction, to ≤ 0.5 rad;
  * per-step angles below 1e-8 (the series branch of `Exp`/`Jr`) are the normal regime of a stationary, well-calibrated IMU.
* **Z30:** `test_c11` draws \|Γ\| = \|ĴR b̄_g\| ≲ 0.09, so a 1e-12 relative slip in Rbreve contributes ≲ 7e-14. That is below its 1e-13 bound, though the clean residual is 6.7e-16.

**What would kill them** (advisory; not part of this commit):
* **Z08, Z10:** add per-step angles inside the series branch, e.g. 3e-9 and 9e-9 rad, with ≥ 1000 steps at one of them, to `F28_STEP_ANGLES`. The complex-step `_cexp` series is valid there.
* **Z07, Z09, Z11:** extend `F28_STEP_ANGLES` to the domain edge. For example 1.0, 2.0 and 3.0 rad (B2-raw needs only finite φ; `_cexp`'s `|t| ≤ 4` guard would have to be widened, e.g. more series terms, or `_cexp` of φ/2 squared).
* **Z30:** tighten `test_c11` to 1e-14 (clean 6.7e-16, 15x margin), or widen its b̄_g range.

### 5.5 Full-suite runs of the survivors

| mutant | full suite (265) | notes |
|---|---|---|
| Z10, Z11 | **265 passed** | not even a change detector fires |
| Z07 | 255 passed, **10 errors** | all in `test_pass10_mutant_is_killed_as_a_file_edit[*]`: the edit rewrites Y56's anchor, so `pass10_runs`' anchor assertion errors the fixture |
| Z08, Z09 | 244 passed, **21 errors** | `test_pass9_…[*]` (11) and `test_pass10_…[*]` (10): the edit rewrites `A[0:3, 0:3] = dR_step.T`, the anchor of X24 and Y57 |
| Z30 | 264 passed, **1 failed** | `test_mutation.py::test_m04_swapping_y_and_yprime_is_a_genuine_no_op`, which asserts `src`'s Eq. (22b) is **bitwise equal to a copy of Eq. (22b) inside the test file** (with Y/Y' swapped) |

**None of these are kills.**
* The errors are edit-anchor change detectors.
* The Z30 failure is a test that compares `src` with a copy of `src` (Sec. 6).
* No reference test failed in any of the 6 runs.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over from pass 10, unchanged:** items 1-11, 14 and 17-18. None was needed for any kill in this pass.

**Resolved by `2cc0762..23a73bf`:**
* `test_c05b`(c) now uses a literal FD step;
* the stale comment is fixed;
* the `orth >= 1e-11` and `visible >= 1e-7` preconditions are removed;
* the P23-old kill is no longer relied on.

**Noted in this pass:**
* **`test_m04_swapping_y_and_yprime_is_a_genuine_no_op` compares `src` with a copy of `src`.** It retypes Eq. (22b) inside `test_mutation.py` and asserts the result is bitwise equal to `src`'s. So it fails on any edit of Eq. (22b), equivalent or not; it is a change detector. That is its documented purpose (proving M04 is a no-op), but it should not be read as coverage of Eq. (22b). It is the only test that reacts to Z30.
* **New F28 tests whose clean value is exactly 0.** `test_f28_eq21b_eq21c_every_combination_of_bias_halves` and `test_f28_eq21c_at_large_velocity_and_duration` compare against retyped residuals that perform the same floating-point operations, so the clean error is 0.0 by construction. They are still genuine reference checks (independent retypes, killing Y30, Y59, Y31, Z25, Z26 and Z27). But at 1e-12 they cannot distinguish a reordering of the same arithmetic, so their sensitivity is set by the input scale (\|v_i Dt\| ~ 60 m), not by rounding.
* **Coverage edge:** `F28_STEP_ANGLES` spans [1e-7, 0.5] rad. That is the cause of Z07-Z11.

## 7. Final state

This commit contains this report only. The worktree is clean:
* the scratch directory (`results/p11`, gitignored) was deleted;
* every mutant copy was a `tempfile` directory under it and was removed;
* no file outside the worktree was written;
* `src/`, `tests/` and `docs/claims/` are untouched.

Full suite on the clean tree at the end (`23a73bf` plus this report): **265 passed** in 187.0 s.
(`results/p10fix`, gitignored, pre-dates this pass and is not mine; I left it in place.)

**Gate: NOT passed.** There are 6 non-equivalent survivors: Z07, Z08, Z09, Z10, Z11 and Z30. They are recorded as **M2-F29**: per-step angle coverage at the domain edges (< 1e-8 and > 0.5 rad), and the C11 sample range. M2-C34 is REFUTED. M2-F28 is FIXED. The `Jr` src change is correct and moves no claim result.
