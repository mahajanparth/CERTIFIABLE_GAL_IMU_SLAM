# Stage-B verification, pass 12: M2 baselines (`stage-b/m2-baselines` @ `75f941e`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration. Agent: verify-M2-012.

**Scope of this pass:**
* a ruling on M2-F29 (the pass-11 survivors Z07, Z08, Z09, Z10, Z11 and Z30), answered by `75f941e`, which changes tests only;
* verdicts on M2-C01..C25, plus M2-C35, which this pass's brief supplies and the yaml does not contain;
* the six pass-11 survivors replayed as real file edits, retyped by me (not copied from `PASS11_MUTANTS`);
* 60 new mutants (A01-A60), each a real file edit;
* a replay of `tests/test_mutation.py`.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing named `*SEEDS*`. I did not browse the git history of `docs/claims`. The only history I diffed is `052ab7a..75f941e`.

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`75f941e`) | full suite **275 passed** in 226 s (load average 13-18; other jobs were running) |
| `src/` change in `052ab7a..75f941e` | **none.** `git diff --stat 052ab7a 75f941e -- src` is empty |
| **M2-F29** | **FIXED.** All six pass-11 survivors are killed as real file edits by reference tests, on the intended assertion lines. The references are valid at the new angles (Sec. 3) |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C35), 0 UNTESTABLE |
| New mutants (60) | 43 killed by reference tests; 15 equivalent (numbers in Sec. 5.3); **2 non-equivalent survivors** (A07, A11) |
| `tests/test_mutation.py` replay | 161/161 pass on the clean tree (inside the 275), including the 6 PASS11 file-edit replays and their guard |
| **Gate** | **NOT passed.** 2 non-equivalent survivors, both in the Eq. (19) / Eq. (22b) pair. They pass because the C11 bounds sit at the claim tolerance (1e-13) while the clean floor is 8.9e-16, and because nothing checks `src`'s Eq. (19) against a reference at large \|ξ\| (finding **M2-F30**) |

## 1. Independent references (no `src` imports)

My reference code lived in `results/p12` (gitignored) and was deleted before commit. It never imports `src` or `tests`. `src` was called only as the system under test. Where I evaluated a test helper (for example `_cs_covariance`), it was as an object under audit, never as a reference.

* **Rotations:**
  * `Exp` = scipy `expm(hat)`; `Log` = scipy `Rotation`;
  * in mpmath, Rodrigues at 60 digits and `mpmath.expm`.
* **Exact increments:** a 5x5 block `expm` (ZOH), and DOP853 at rtol = atol = 1e-13 (for C35).
* **Euler sums:** retyped Eq. (9a)-(9c).
* **Covariance, 60-digit mpmath (new; my core re-implementation for F29):**
  * Each step's linearisation (A, Bg, Ba) is a central difference at h = 1e-20 in 60-digit mpmath.
  * The chart is `R Exp(d)`, `v + d`, `p + d`, read off as `vee(skew(Rnᵀ R'))`. In 60 digits Rn is orthogonal to about 1e-58, so there is no chart ambiguity.
  * The steps are then combined by `C ← A C Aᵀ + Bg Qg Bgᵀ + Ba Qa Baᵀ`.
  * No Jacobian block is derived and no complex step is used.
  * Runs: per-step angles 3e-9 (x 1000 steps) and 5e-9 (x 4000 steps), plus 9e-9, 1e-7, 0.5, 0.7, 1, 2 and 3 rad (x 12 steps), for Exp and Cay, with gyro noise alone and with gyro + accel noise.
* **Complex-step bias Jacobians (mine):** scipy `expm` of a complex hat, a `solve`-based complex Cayley, and h = 1e-30.
* **GTSAM side:**
  * my own 5x5 Gal(3) matrices and `logm`;
  * my own Ad matrix from the contract's formula;
  * my own FD bias Jacobian and recursion;
  * my own factor graph for C25.
* **C25:** σ_g 3e-4, σ_a 2e-2, bias walks 3e-5 / 5e-4, integration 1e-5, prior σ 1e-2, GNSS σ 0.05. A second density set was 3-10x looser.

**Re-implemented core routines:**
* the Eq. (63) covariance (mpmath, above);
* the Eq. (9) sums;
* Eq. (19) and Eq. (22b);
* the Gal(3) bias Jacobian, Ad and the C21 recursion;
* the C18 endpoint products.

## 2. M2-C35 (brief-only claim): REFUTED

**Claim:** Baseline 1's position prediction is `p_j = p_i + v_i Dt − 0.5 g Dt² + R_i dp_ij` with g = (0, 0, −9.81), "the same minus sign as the −0.5 g Dt² entry of H'". The claim also says this reproduces DOP853 ground truth to 1e-6 on a Dt = 0.5 s edge.

**Both halves are false.**

1. **`src` does not use a minus sign.** `ForsterPreintegration.predict` (and `trajectory.chain`, and C02/C03's statements) use `+ 0.5 * g_W * Dt**2`. Physically `p̈ = g + R a`, so the gravity term is `+½ g Dt²`.
2. **The −½ g Dt² entry of H' comes from a different composition.** It is `T(−Dt) · H' · X_i · D_ij`. In the product, the clock coupling `v t` of the 5x5 group law contributes `+ g Dt · Dt` to the position block, and `−½ g Dt² + g Dt² = +½ g Dt²`. So the minus sign in the matrix does not carry over to the component relation.

**Measured.** DOP853 at rtol = atol = 1e-13 from the fixture KF0 over Dt = 0.5 s. The body increment comes from a second gravity-free DOP853 run.

| edge (body ω) | claim's −½ g Dt² vs DOP853 | `src` form +½ g Dt² vs DOP853 |
|---|---|---|
| fixture edge 0 | **2.452500** m | 8.9e-16 |
| fixture edge 1 | **2.452500** m | 6.2e-16 |
| ω = (1.3, 0.7, −2.1) | **2.452500** m | 1.4e-15 |

* The claim's error is exactly \|g\| Dt² = 2.4525 m, against the claimed 1e-6.
* `src`'s `B1.predict`, fed the DOP853 increments, reproduces DOP853 to **1.2e-15** in R, v and p.

## 3. Ruling on M2-F29: FIXED

`75f941e` changes `tests/test_input_dependence.py` and `tests/test_mutation.py` only. The `src` diff is empty.

### 3.1 The extended per-step angle sweep and the 4000-step stationary test

**Are the references valid at those angles?** Yes. My 60-digit mpmath covariance (Sec. 1) is the arbiter. Below, "cs" is the test's `_cs_covariance` with the new `R0⁻¹` chart and "cs-old" is the same with `R0ᵀ` swapped back in. All values are block-relative, worst of gyro noise alone and gyro + accel.

| per-step angle x steps | `src` B1 / B2-raw / B2-reparam vs mpmath | cs vs mpmath | cs-old vs mpmath |
|---|---|---|---|
| 3e-9 x 1000 | 9.0e-15 / 9.0e-15 / 9.0e-15 | **1.2e-14** | **7.0e-12** |
| 5e-9 x 4000 (stationary test) | 1.1e-13 / 1.1e-13 / 1.1e-13 | **7.8e-14** | **1.7e-10** |
| 9e-9 x 12 | 7.6e-16 (all three) | 5.1e-16 | 4.9e-15 |
| 1e-7 x 12 | 4.5e-16 / 1.0e-15 / 4.5e-16 | 8.5e-16 | 3.7e-15 |
| 0.5 x 12 | ≤ 1.1e-15 | 1.1e-15 | 5.2e-15 |
| 0.7 x 12 | ≤ 1.1e-15 | 1.4e-15 | 3.8e-15 |
| 1 x 12 | ≤ 1.4e-15 | 1.2e-15 | 7.7e-15 |
| 2 x 12 | ≤ 2.6e-15 | 3.8e-15 | 3.0e-14 |
| 3 x 12 | ≤ 4.7e-15 | 8.3e-15 | 7.3e-14 |

* At every new angle the complex-step reference agrees with 60-digit arithmetic to ≤ 1.2e-14 (≤ 7.8e-14 over 4000 steps). That is ≥ 1280x inside `COV_CS_REL` = 1e-10.
* The angles are where they are stated to be. `_f28_stream` sets `gyro = b_g + û·angle/dt`. The float subtraction `gyro − b_g` perturbs the per-step angle by about 1e-11 relative at 3e-9 rad, so 3e-9, 5e-9 and 9e-9 rad all sit inside the `|w dt| < 1e-8` series branch at every step.
* The stream inputs come from `src`'s `T.time_varying_stream`. That is input generation, not reference computation: `src` and the reference consume the same samples. A stream mutation (A43) is killed separately by the C19 and stream-literal tests.

### 3.2 The complex-step chart change (`R0ᵀ` → `np.linalg.solve(R0, dR)`)

* **The implementer's number reproduces.** With `R0ᵀ` the reference is 7.0e-12 off mpmath at 3e-9 x 1000, against `src`'s 9.0e-15. With `R0⁻¹` it is 1.2e-14. (The brief quotes 9.6e-15; on my run it is 1.2e-14 for the reference and 9.0e-15 for `src`.)
* **The change was necessary, not cosmetic.** On the new 4000-step stationary test the old chart is **1.7e-10** off mpmath. That exceeds `COV_CS_REL`, so the clean code would have failed against the old reference.
* **It shares no computation with `src`.** `R0` in `_cs_chart` is the reference's own propagated state (`_ceuler`, retyped `_cexp` / `_ccay`), and the solve is numpy. `_cs_chart`, `_cs_step_jacobians`, `_ceuler`, `_cexp`, `_ccay`, `ivee` and `_chat` import nothing from `src`.
* **It cannot hide a `src` defect.** For an orthogonal `R0` the two charts are identical. They differ only through the reference's own orthogonality drift, 1.1e-14, which is 1e4 times below the 1e-10 bound. The new reference is also more accurate at every angle in Sec. 3.1.

### 3.3 `_cexp` guard widened to `|t| = |φ|² ≤ 16`

I checked the series against 50-digit `mpmath.expm`, using 40 random directions per magnitude. The derivative is the complex-step value `Im _cexp(φ + i·1e-20·e)/1e-20`, compared with an mpmath central difference at h = 1e-25.

| \|φ\| | value error | derivative error |
|---|---|---|
| 1e-9 | 0 | 2.2e-16 |
| 1 | 1.1e-16 | 2.2e-16 |
| 2 | 5.6e-16 | 5.6e-16 |
| 3 | 7.8e-16 | 7.8e-16 |
| 3.14 | 5.0e-16 | 5.8e-16 |
| 3.5 | 1.1e-15 | 6.7e-16 |
| 3.9 | 1.7e-15 | 1.2e-15 |
| 3.9999 | 1.4e-15 | 1.3e-15 |

* The series is accurate to the guard's edge, and the sweep uses \|φ\| ≤ 3.
* Two docstring figures are slightly optimistic:
  * "cancellation costs < 1e-15 absolute" reads 1.7e-15 at 3.9 rad;
  * "≤ 3.3e-16" (two fixed directions) reads up to 7.8e-16 over random directions at 3 rad.
* Neither matters against the test's 1e-14 bound or the covariance's 1e-10.

### 3.4 Z30 and `test_c11`

* **Z30 is killed by a genuine reference test.** `test_f29_eq22b_matches_a_retyped_eq19_at_large_gamma` compares `src`'s Eq. (22b) with Eq. (19) retyped in the test, a different equation, at \|Γ\| up to about 2. Z30 reads **1.22e-12** against the 1e-13 bound.
* **But leaving `test_c11` and the new test at 1e-13 leaves a resolution gap.** The clean floor is 8.9e-16, so either test could be 1e-14 with ≥ 11x margin. Two new survivors sit in that gap (A07, A11; Sec. 5.4).
* A11 also exposes a separate hole: no test checks `src`'s **Eq. (19)** against a reference at large \|ξ\|. The new test checks `src`'s Eq. (22b) against a *retyped* Eq. (19). `test_c11` checks `src`'s 22b against `src`'s 19, but only at \|Γ\| ≲ 0.09.
* I rule Z30 itself killed. The gap is recorded as the new finding **M2-F30**, not as F29 left open.

### 3.5 Is the "every killer is a reference check" meta-test sound?

`test_pass11_killers_are_reference_checks` checks two things:
* that every killer node id starts with `tests/test_input_dependence.py::`;
* that `PASS11_REQUIRED` equals the replayed id set.

**This is a module-membership proxy, not a proof.** It would accept a change detector placed in `test_input_dependence.py`. On its own it is therefore not sound.

**In this instance the conclusion holds anyway.** I read the three killers:
* `test_f28_covariance_across_per_step_angles_against_complex_step` and `test_f29_covariance_of_a_stationary_imu_over_4000_steps` compare against the complex-step reference, which Sec. 3.1 validates against mpmath;
* `test_f29_eq22b_matches_a_retyped_eq19_at_large_gamma` compares against a retyped different equation.

None compares `src` with a copy of `src`. The companion `test_pass11_mutant_is_killed_as_a_file_edit` also requires each kill on the intended `E ` line, which closes the "wrong assertion fired" route.

**Advisory:** the meta-test's docstring should say that it checks location, and that the reference property was established by reading the killers.

### 3.6 Replay of Z07-Z11 and Z30 as real file edits (my retypes)

Each edit is exactly-once, on a temp copy, and judged by the fast set (114 reference tests, `test_mutation` excluded). The edit texts differ from `PASS11_MUTANTS`'s:
* **R07** edits `so3_right_jacobian` itself, so it also hits B2-reparam;
* **R07b** edits `retract_right_jacobian`;
* **R08 / R09 / R10** use conditional expressions in place of the implementer's forms;
* **R30** writes `(1 + 1e-12) * Gamma / 2`.

I then re-ran the four killer tests and read the `E ` lines.

| mutant | killed? | by | measured (vs bound) |
|---|---|---|---|
| R07 (Z07 via `Jr` above 0.6) | **killed** | f28 covariance sweep; f28 right-Jacobians vs mpmath; `test_b1_so3_right_jacobian` | B1 and B2-reparam 0.176 (0.7 rad) .. 8.2 (3 rad) vs 1e-10 |
| R07b (Z07, B1 only) | **killed** | f28 covariance sweep | B1 0.176 .. 8.2 |
| R08 (Z08) | **killed** | f28 sweep; f29 stationary | 1.92e-6 (3e-9 x 1000), 3.42e-8 (9e-9), 7.71e-6 (5e-9 x 4000) |
| R09 (Z09) | **killed** | f28 sweep | 2.3 .. 6.2 at 0.7-2 rad, all three classes |
| R10 (Z10) | **killed** | f28 sweep; f29 stationary | 2.80e-9 (3e-9 x 1000), 1.01e-8 (9e-9), 3.48e-9 (5e-9 x 4000); margin 28x |
| R11 (Z11) | **killed** | f28 sweep | B2-raw 0.092 (0.7 rad) .. 1.74 (3 rad) |
| R30 (Z30) | **killed** | f29 Eq. (22b) vs retyped (19) | 1.22e-12 vs 1e-13 |

* Every kill lands on the intended assertion (the angle-labelled `E ` lines above).
* The numbers agree with those the implementer recorded in `75f941e`.

**Edge probes** (A01-A06) move the thresholds to just inside the new sweep ends: 4e-9 at the bottom; 1.5 and 2.5 rad at the top; and a transposed `Jr` above 2.5 rad. **All are killed by the f28 sweep.**

## 4. Per-claim verdicts

All numbers are mine.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 11 constants checked equal the retyped literals exactly. R_0 vs `expm`: 1.1e-16 |
| C02 | CONFIRMED | Retyped M1/M2 vs `quad_vec` (1e-14): 7.6e-17 / 1.8e-16 (e0) and 5.6e-17 / 9.7e-17 (e1). `src` chain vs literals: KF1 v/p 4.4e-16 / 3.5e-17; KF2 R/v/p 1.1e-16 / 8.6e-16 / 2.2e-16. My 5x5 `expm` chain vs literals: ≤ 3.3e-15 |
| C03 | CONFIRMED | GTSAM `predict` from NavState(Exp([0.3, −0.5, 0.2]), p, v) vs the formula: R 8.9e-15, v 8.9e-16, p 4.4e-16 |
| C04 | CONFIRMED | n = 100..1600. e0: v 0.50002 → 0.50000, p 0.50086 → 0.50011. e1: v 0.50003 → 0.50000, p 0.50088 → 0.50011. `src` vs my Euler ≤ 1.0e-13 |
| C05 | CONFIRMED | R/v/p ratios 100.02 / 99.96 / 99.96 (e0) and 99.97 / 100.01 / 100.00 (e1) |
| C06 | CONFIRMED | σ_g 1.7e-4, σ_a 1e-2, every step: asymmetry ≤ 1.7e-20, min eigenvalue −4.0e-23, trace nondecreasing. Anisotropic densities likewise (6.8e-20 / −2.6e-23). Vs mpmath (Sec. 3.1): ≤ 4.7e-15 at 12 steps |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 2.9e-15. (b) 300 φ with \|φ\| = 0.4, ε = 1e-7: right 8.4e-10; left max 0.384, min 0.229 (≥ 1e-1 even as a minimum on this sample) |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 3.6e-15; closed form ≥ 1 (min 1.0000005) |
| C09 | CONFIRMED | 500 draws: on-manifold 2.5e-16; perturbed minimum 1.6e-4. **Spurious-point hunt:** the homogeneous part `R_j ↦ R_j(I − ξ̂/2)` has smallest singular value 1.0 over 10,000 ξ with \|ξ\| up to 28, so r^R = 0 has the unique solution `R_i dR̄ Cay(ξ)` among ALL 3x3 R_j (recovered to 1.4e-13, orthogonal). No spurious zero exists |
| C10 | CONFIRMED | Factor dv̄ vs my Cayley Euler: 0.0. Self-consistent: 2.7e-15. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.377e-4 / 6.810e-4 |
| C11 | CONFIRMED | 2000 draws, C11-test-like distribution: 4.4e-16. 20,000 draws with J_R ~ N(0, 1) and \|b\| ≤ 0.5: 8.9e-16. See F30 for the test resolution |
| C12 | CONFIRMED | 10,000 draws: isotropic 7.9e-15 (relative to max(1, \|r\|²)); W = diag(1, 2, 5) max 56.3 ≥ 1; τ(diag(0.5, 0.6, 0.7)) = 3/1.8 exactly |
| C13 | CONFIRMED | `J_R` bytes unchanged after `predicted_rotation` / `rotation_residual`. 200 directions at \|ξ\| = 1e-2: gap 3.45e-5 .. 5.00e-5 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7; accessors as stated; `Gal3(M)` round trip 0.0 |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. Expmap vs `expm(Hat)`: 1.1e-14 over 50 ξ. Remap (0, 6, 3) |
| C16 | CONFIRMED | Remapped `logm` vs `localCoordinates`: **0.107628**. Vs `NavState.Logmap`: 1.8e-15 |
| C17 | CONFIRMED | `ConstantBias(a, g)` accessors and `vector()` = [a; g]; `biasHatVector` = [b_a; b_g]; adapters round-trip exactly |
| C18 | CONFIRMED | My 5x5 products: W form ≤ 3.6e-15 (Dt = 0.5, t_i ∈ {0, 3.7}); t_j = t_i + Dt. F2 form: 0 at t_i = 0; **18.148500** (Dt = 0.5) and **36.297000** (Dt = 1.0) at t_i = 3.7, = \|g\| Dt t_i exactly. Swapped order at t_i = 0: 2.4525 / 9.81. `src` `endpoint_W_form` vs my product: ≤ 8.9e-16 |
| C19 | CONFIRMED | PIM-G vs ZOH: R/v/p 1.0e-14 / 2.0e-14 / 3.8e-15 (e0) and 8.4e-15 / 1.7e-14 / 3.1e-15 (e1). Tangent dv 2.171e-3 / 2.760e-3. Retyped time-varying stream equals `src`'s exactly; right composition 2.2e-16; left 6.2e-4 |
| C20 | CONFIRMED | My FD J (ε = 1e-7, `logm`, g = 0): J[0:3,0:3] = J[9,:] = 0.0. Right composition at 5e-2: ≤ 2.1e-10. Component-wise over 30 random directions: 1.7e-3 .. **1.33e-2**. That reads ≥ 1e-2 only as a sup, which the claim's hint documents. `src` J vs mine: 3.7e-9 |
| C21 | CONFIRMED | `src` recursion (40 steps) vs my FD re-integration: 2.3e-9 relative. My own recursion with my Ad: 1.0e-8. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1-B2: v 6.1672e-8, p 4.3763e-8, R **2.9965e-8**. Per edge 2.325744e-8 / 5.517531e-8 vs the formula 2.325744e-8 / 5.517533e-8. dt² ratios 0.25000 x3, 0.24999 |
| C23 | CONFIRMED | B1-B3: v 1.2867e-3, p 1.2031e-3, R 3.1e-15; B2-B3 R 3.0e-8. Ratios: B1 v 0.50028 → 0.50003, p 0.5000 x4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.50000 |
| C24 | CONFIRMED | B3 vs exact: KF1 8.9e-16 / 1.6e-14 / 2.1e-15; KF2 1.1e-15 / 4.7e-15 / 5.1e-15. B1 R 9.5e-15 / 2.7e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My graph, two density sets: ranks 30 (30x45), 39 (39x45), 45 (45x45). Principal cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**; pose/velocity energy 2.5371, identical for both density sets. Graph error at truth ≤ 2.1e-23 |
| **C35** | **REFUTED** | Sec. 2. The claim's −½ g Dt² is off DOP853 by **2.4525 m** = \|g\| Dt² on every edge tried; `src` uses +½ g Dt² and matches DOP853 to 1.2e-15 |

**Counts: 25 CONFIRMED, 1 REFUTED (C35), 0 UNTESTABLE.**

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh `tempfile` copy of `src tests orchestrator docs conftest.py pyproject.toml`. Every anchor is asserted to occur exactly once.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (114 tests, including the slow stationary test).
* **Every** fast-set survivor was then run on the **full suite** (275 tests, slow included). An unmutated control copy passed 114/114.
* The worktree's tracked files never changed.

### 5.1 New mutants A01-A60

| id | change | result |
|---|---|---|
| A01 | A_RR = I when \|w dt\| < 4e-9 | killed: f28 sweep (3e-9 x 1000) |
| A02 | Bg = I·dt when \|w\|dt < 4e-9 | killed: f28 sweep |
| A03 | A_RR = I when \|w dt\| > 2.5 | killed: f28 sweep (3 rad) |
| A04 | B2 raw uses `Jr` when \|φ\| > 2.5 | killed: f28 sweep |
| A05 | Bg uses `Jrᵀ` when \|w\|dt > 2.5 | killed: f28 sweep |
| A06 | A_RR = `dR_step` (not ᵀ) when \|w dt\| > 1.5 | killed: f28 sweep |
| **A07** | Eq. (22b) `Rbreve`: Γ/2 x (1 + 5e-14) | **survives; NOT equivalent** |
| A08 | Eq. (22b) `R_j(I + Γ/2·(1 + 1e-12))` | killed: f29 Eq. (22b) |
| A09 | Eq. (22b) `(0.5 + 1e-13)(Y + Y')` | killed: f29 Eq. (22b) |
| A10 | `Y' = R_j Ξ (1 + 1e-12)` | killed: f29 Eq. (22b) |
| **A11** | Eq. (19) ½ → ½ + 3e-14 | **survives; NOT equivalent** |
| A12 | Eq. (17) closed form x (1 + 1e-13) on the ‖ξ‖²/4 term | killed: b2_cayley_maps |
| A13 | `cay_denominator_det` x (1 + 1e-13) | killed: b2_cayley_maps |
| A14 | truncated residual uses I + ½ξ̂ | killed: c13, c13c, b2_rotation_residual_functions |
| A15 | `isotropic_precision` x (1 + 1e-14) | killed: c12, b2_precision |
| A16 | `cayley_vector` identity branch below 1e-8 | survives; **equivalent** |
| A17 | reparam `cayley_vector` x (1 + 1e-13) | killed: b2_variant_flag |
| A18 | `from_edge` FD Jacobians ignore `variant` | killed: b2_factor_from_edge, f27 high-rate |
| A19 | Eq. (16) `R_i Cay(ξ) dR̄` | killed: c09, c13, … |
| A20 | `J_Cay` denominator x (1 + 1e-11) | killed: b2_cayley_maps, f28 right Jacobians |
| A21 | `dv` updated before `dp` | killed: c04, c05b, c10, … (16 tests) |
| A22 | `Jr` series K² coefficient 1/3 | survives; **equivalent** |
| A23 | `Jr` series drops K² | survives; **equivalent** |
| A24 | B1 FD `J_R` in the left chart | killed: c05, c05b, c13b, … |
| A25 | FD `J_p` x (1 + 3e-8) | killed: b1_bias_jacobians_fd, f27 high-rate |
| A26 | `FD_EPS` = 1e-4 | survives; **equivalent** |
| A27 | `FD_EPS` = 1e-5 | survives; **equivalent** |
| A28 | Eq. (44) `J_va db_a` x (1 + 1e-9) | killed: b1_bias_corrected_increment |
| A29 | Eq. (44) left composition `Exp(J_R db_g) dR̄` | killed: c05, b1_bias_corrected_increment |
| A30 | A_pv x (1 + 1e-11) | survives; **equivalent** |
| A31 | A_pR as `−½ hat(dR a) dR dt²` | survives; **equivalent** (proof) |
| A32 | covariance gate on `diag(cov) != 0` | survives; **equivalent** (proof) |
| A33 | B1 `predict` with −½ g Dt² (the C35 form) | killed: c10, c23, c24, … |
| A34 | `increment_error` ignores dv | killed: c05, b1_increment_error |
| A35 | `bias_hat` aliases the caller | killed: f18 |
| A36 | per-axis σ_g reversed | killed: c06b, c06c, … (10 tests) |
| A37 | `SMALL_ANGLE` = 1e-6 | killed: f25 so3_log small angles vs mpmath |
| A38 | `so3_log` clamp removed | killed: c05, f24 near-π |
| A39 | M2 series K² coefficient 1/12 | killed: f25 series branches |
| A40 | M1 K² coefficient x (1 + 1e-12) | killed: f27 M1/M2 vs quadrature |
| A41 | near-π `atan2` uses clamped c | survives; **equivalent** |
| A42 | `so3_log` small branch x (1 + 1e-9) | survives; **equivalent** |
| A43 | time-varying gyro amplitude 0 | killed: c05b, c19, traj stream literals |
| A44 | `so3_exp` series drops K²/2 | killed: f25 series branches |
| A45 | `so3_exp` K² coefficient x (1 + 1e-14) | survives; **equivalent** |
| A46 | `J_FD_EPS` = 1e-6 | survives; **equivalent** |
| A47 | `C_FD_EPS` = 1e-5 | survives; **equivalent** |
| A48 | `gal3_distance` ignores the clock | killed: b3_models_and_distance |
| A49 | `integrate_stream` ignores `n_steps` | killed: c21, b3_integrate_stream |
| A50 | `time_shift` uses \|dt\| | killed: c18, b3_endpoint_relations |
| A51 | gravity removal in v x (1 + 1e-13) | survives; **equivalent** |
| A52 | component-wise model rotates its ξ by R | killed: c20, b3_models_and_distance |
| A53 | recursion C_k x (1 + 1e-8) | killed: f27 recursion |
| A54 | gyro covariance x (1 + 1e-9) | killed: b3_make_params, b3_constructor |
| A55 | `endpoint_W` t_i term x (1 + 1e-15) | killed: b3_endpoint_relations |
| A56 | H' position x (1 + 1e-12) | killed: c18, b3_endpoint_relations |
| A57 | `bias_hat` writable | killed: f22 |
| A58 | `from_gtsam_bias` unswapped | killed: c03, c05c, c17, … (13 tests) |
| A59 | `navstate` re-orthonormalises R | killed: f27 navstate drifted rotation |
| A60 | Eq. (21b) `A_va db_a` x (1 + 1e-13) | survives; **equivalent** |

**Count (60):** 43 killed by reference tests, 15 equivalent, **2 not equivalent and surviving**. Together with the 7 replays (Sec. 3.6, all killed), 67 file-level mutants were run in total.

### 5.2 `tests/test_mutation.py` replay

On the clean tree, all 161 tests in `test_mutation.py` pass (inside the 275). That includes:
* `test_pass11_mutant_is_killed_as_a_file_edit` [Z07 … Z30], which ran its control copy and the 6 file-level edits, each checked on its intended `E ` line;
* `test_pass11_killers_are_reference_checks`.

My independent retypes (Sec. 3.6) agree with every one of them.

### 5.3 Equivalence rulings (owner's rule; threshold = max(τ/10, 8 ulp))

| mutant | quantity Q | τ (claim) | threshold | measured E | ruling |
|---|---|---|---|---|---|
| A16 | `cayley_vector` → B2-reparam everything | — | — | **0.0** on `cayley_vector` (600 angles 1e-12..3 rad) and on B1/B2 covariances and increments (13 angle cases, up to 4000 steps). **Proof:** for θ < 1e-8, 2 tan(θ/2)/θ = 1 + θ²/12 + … with θ²/12 < 8.4e-18 < ulp(1)/2, so the factor is 1.0 | equivalent |
| A22 | `Jr` → covariance | C06 2.7e-9 rel | 2.7e-10 | `Jr` 7.7e-18 abs; covariance 1.9e-17 block-rel | equivalent |
| A23 | `Jr` → covariance | C06 | 2.7e-10 | `Jr` 7.7e-18; covariance 1.6e-18 | equivalent |
| A26 | B1/B2 FD Jacobians → C05 | C05 ratio ∈ [50, 200] | — | J vs my complex step ≤ 1.9e-9 rel (clean ≤ 1.2e-9) on fixture and 5 / 2.5 rad/s edges; C05 ratios move ≤ 2e-4 | equivalent |
| A27 | same | same | — | J ≤ 4.5e-9 rel; C05 ratios move ≤ 1.6e-3 | equivalent |
| A30 | covariance | C06 | 2.7e-10 | 2.0e-11 block-rel over the 13 angle cases (bounded by about 2δ, independent of step count) | equivalent |
| A31 | covariance | C06 | — | 3.3e-16. **Proof:** hat(Ra) R = R hat(a) Rᵀ R = R hat(a) | equivalent |
| A32 | covariance gate | C06 | — | **Proof:** for PSD `cov`, \|C_ij\| ≤ √(C_ii C_jj), so a zero diagonal implies `cov` = 0. The gate then decides identically. Differs only for a non-PSD "covariance", which is outside the domain | equivalent |
| A41 | `so3_log` near π | C22 1e-3 rel | — | 0.0 over 20,000 rotations with π − θ ∈ [1e-16, 0.5]. **Proof:** it differs only when c_raw < −1 by rounding | equivalent |
| A42 | `Log` (θ < 1e-8 branch) → FD J_R, `increment_error` | C22 1e-3 rel; FD `J_R` (C21-level 1e-7) | 1e-8 rel | 1.0e-9 relative / 1.0e-17 absolute on `Log`; FD bias Jacobians **0.0** (their Log arguments are ≥ 1e-5 rad, in the generic branch) | equivalent |
| A45 | `Exp` → B1 R, covariance | C24 1e-10 | 1e-11 | `Exp` 2.0e-14 abs; B1 KF2 R 4.4e-16, p 1.8e-15; covariance 0.0 | equivalent |
| A46 | B3 FD J | C20 1e-8; C21 1e-6 rel | 1e-9; 1e-7 | J vs Richardson FD (h = 2e-3, self-consistency 1.6e-12): clean 5.4e-9 / 4.4e-9, mutant 6.4e-10 / 3.6e-10. Change ≤ 5.4e-9 rel. Right-composition residual at 5e-2: clean 4.4e-10, mutant 4.5e-11 | equivalent (the mutant is **more** accurate) |
| A47 | C21 recursion | C21 1e-6 rel | 1e-7 | J₄₀ vs Richardson: clean 4.0e-10, mutant 3.2e-11 | equivalent (more accurate) |
| A51 | `corrected_increment_gal3` | C20 1e-8 | 1e-9 | 9.8e-12 abs up to Dt = 10 s with g = G_W (0.0 at g = 0, the C20 setting); FD J 3.6e-9 rel | equivalent |
| A60 | Eq. (21b) | C10 1e-9 | 1e-10 | 6.2e-13 abs (\|A_va\| ~ N(0, 1), \|db_a\| ≤ 1) | equivalent |

### 5.4 Complete survivor list (non-equivalent)

| mutant | change | distinguishing input and effect | threshold |
|---|---|---|---|
| **A07** | Eq. (22b): `Rbreve = dRbar (I − (1 + 5e-14) Γ/2)` | 20,000 draws: **7.9e-14 abs** with J_R ~ N(0, 1), \|b\| ≤ 0.5 (the F29 test's distribution); 3.2e-15 on the C11-test distribution. Under both tests' 1e-13 bound; clean is 8.9e-16 | C11 1e-13 → **1e-14** |
| **A11** | Eq. (19): `(0.5 + 3e-14)(R_j + A) ξ̂` | **2.2e-13 abs** on the C11 identity (`src` 22b vs `src` 19) at J_R ~ N(0, 1), \|b\| ≤ 0.5. That **exceeds the claim's own 1e-13**, yet no test sees it. 1.1e-14 on the C11-test distribution. C09 on-manifold residual with \|ξ\| up to 28 (inside C07's θ < 3 chart): 1.2e-13 | C11 1e-13 → 1e-14; C09 1e-12 → 1e-13 |

**Why these survive:**
* `test_c11` draws \|Γ\| ≲ 0.09.
* `test_f29_eq22b_matches_a_retyped_eq19_at_large_gamma` reaches \|Γ\| ~ 2, but both use the claim tolerance 1e-13 as the bound, ~110x above the clean floor.
* The F29 test checks `src`'s **22b** against a *retyped* 19, so `src`'s **19** is only ever checked at small \|ξ\| (`test_c11`, `test_c09`).

**What would kill them** (advisory; not part of this commit):
* bound the F29 Eq. (22b) test at 1e-14 (clean 8.9e-16, 11x margin), which kills A07 at 7.9e-14;
* add the mirror test: `src`'s `rotation_residual_eq19` against a retyped Eq. (22b), or against a retyped Eq. (16)-cleared form `R_j(I − ξ̂/2) − R_i dR̄ (I + ξ̂/2)`, at J_R ~ N(0, 1) / \|b\| ≤ 0.5 and at \|ξ\| up to the C07 chart edge, bound 1e-14. That kills A11 at 2.2e-13.

### 5.5 Full-suite runs of all 17 fast-set survivors

| mutant | full suite (275) | notes |
|---|---|---|
| A11, A30, A31, A41, A42, A45, A46, A47, A60 | **275 passed** | |
| A07 | 268 passed, **1 failed, 6 errors** | failed: `test_m04_swapping_y_and_yprime_is_a_genuine_no_op` (in-test copy of Eq. (22b), a change detector, pass-11 Sec. 6). Errors: `test_pass11_mutant_is_killed_as_a_file_edit[*]`, because Z30's edit anchor is no longer unique |
| A16, A22, A23, A32 | 254 passed, **21 errors** | A16, A32: `test_pass8_mutant_is_killed_as_a_file_edit[*]` anchor errors (classified). A22, A23: same count, not individually classified |
| A26, A27 | 264 passed, **11 errors** | A26: `test_pass9_mutant_is_killed_as_a_file_edit[*]`, `FD_EPS = 3e-5` anchor (classified). A27: same count |
| A51 | 238 passed, **37 errors** | `test_pass7_mutant_is_killed_as_a_file_edit[*]` |

**None of these are kills.** The errors are edit-anchor fixtures, and the one failure is a test that compares `src` with a copy of `src`. No reference test failed in any of the 17 runs.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over from pass 11, unchanged:**
* `test_m04_swapping_y_and_yprime_is_a_genuine_no_op` (src vs in-test copy of Eq. (22b));
* the two F28 retyped-residual tests whose clean value is 0.0 by construction;
* the pass-10 items.

**Noted in this pass:**
* **`test_pass11_killers_are_reference_checks` checks module membership only** (Sec. 3.5). It is sound here only because the three killers were read and are genuine references.
* **The `_cexp` docstring figures are slightly optimistic** (Sec. 3.3: 1.7e-15 against "< 1e-15"; 7.8e-16 against "≤ 3.3e-16"). This is harmless at the 1e-14 bound.
* **Bounds equal to the claim tolerance.** `test_c11` and the F29 Eq. (22b) test (both 1e-13, clean ≤ 8.9e-16), and `test_c09`'s on-manifold 1e-12, leave a band between τ/10 and τ in which mutants survive. A07 and A11 sit there.
* **Advisory: `J_FD_EPS` = 1e-7 is round-off-limited.** 1e-6 is 8x more accurate (A46). Likewise `C_FD_EPS` = 1e-5 beats 1e-6 by 12x (A47). This is not a defect: both are well inside C20 / C21.

## 7. Final state

This commit contains this report only. The worktree is clean:
* the scratch directory `results/p12` (gitignored) was deleted;
* every mutant copy was a temp directory under it and was removed;
* no file outside the worktree was written;
* `src/`, `tests/` and `docs/claims/` are untouched.

The full suite on the clean tree at the end (`75f941e`) passed: **275 passed** in 151.7 s.

**Gate: NOT passed.** There are 2 non-equivalent survivors, A07 and A11, recorded as **M2-F30**. The Eq. (19) / Eq. (22b) bounds are at the claim tolerance rather than τ/10, and `src`'s Eq. (19) is never checked against a reference at large \|ξ\| or \|Γ\|; A11 breaks the C11 identity by 2.2e-13, beyond the claim's own 1e-13, undetected. M2-C35 is REFUTED. M2-F29 is FIXED.
