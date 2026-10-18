# Stage-B verification, pass 15: M2 baselines (`stage-b/m2-baselines` @ `a5f87f2`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration. Agent: verify-M2-015.

**Scope of this pass:**
* rulings on M2-F33 (latent `src` defect), M2-F34 and M2-F35 (pass 14), all answered by `a5f87f2`. This is the first `src` change since pass 13: `trajectory.M1` / `M2` gain a 12-term even series below `th |T| = M_SERIES_SWITCH = 2.0`;
* verdicts on M2-C01..C25, plus M2-C38, which this pass's brief supplies and the yaml does not contain;
* the implementer's seven replays (Q01, Q01b, Q01c, Q03, Q05, G01, F33-revert), retyped by me with different text (RQ01, RQ01b, RQ01c, RQ03, RQ05, RG01, RF33);
* a fresh mutation sweep over `src/`: 45 new file-level mutants, aimed first at the new series code and at the edges of the new F34 / F35 tests;
* a replay of `tests/test_mutation.py`.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing named `*SEEDS*`. I did not browse the git history of `docs/claims`. The only history I diffed is `bc3a7f7..a5f87f2`.

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`a5f87f2`) | full suite **298 passed** in 175 s (single-thread BLAS, `timeout -k 30 1800`). An unmutated control copy in the mutation harness also passed 298/298 (223 s) |
| `src` / `docs/claims` change in `bc3a7f7..a5f87f2` | `src/trajectory.py` only (+45/-1: the M1/M2 series). `docs/claims` unchanged |
| **M2-F33** | **src defect FIXED.** The series coefficients are exact, the series is ≤ 0.98 ε relative on (0, 2.25), and `src` M1/M2 agree with an 80-digit reference to ≤ 4.6e-16 (off-diagonal relative) for \|w\| ∈ [1e-12, 10], \|T\| ≤ 10, both signs, across the switch. No claim-level output moved by more than 1.9e-15. **But the new test's grid leaves holes:** four series/switch mutants survive (new finding **M2-F38**). One residual regime remains: the old `SMALL_ANGLE` branch at large \|T\| (advisory **M2-F39**) |
| **M2-F34** | **The five named survivors are killed.** My retypes RQ01, RQ01b, RQ01c, RQ03 and RQ05 all fail the new chart-edge test on its intended assertion. Over 25 seeds the test is robust: clean headroom ≥ 1.91x (Eq. 19) and ≥ 1.51x (Eq. 22b); weakest mutant ≥ 1.48x over the bound. **But the check covers only \|ξ\| ∈ [20, 28.2].** The same defects moved inside \|ξ\| < 20 survive (Q09, Q10, Q11; new finding **M2-F37**) |
| **M2-F35** | **G01 is killed** (RG01 fails `test_c11` and `test_f35`; clean headroom 2.97x over 13 seeds). **The cause is not closed.** C11 still states no domain in Γ, and no finite sample can cover an unbounded one: G02 (\|Γ\| > 45) and G03 (\|Γ\| > 5 and \|ξ\| > 5 together) survive. The **clean** code itself breaks C11's 1e-13 at \|Γ\| = 1e3 (3.4e-13). G04, a 2-ulp shift of Γ/2 for \|Γ\| > 5, survives at the rule-4 resolution limit. New finding **M2-F36** |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C38), 0 UNTESTABLE |
| **C38** | **REFUTED.** Along a fixed axis Cayley is **not** a one-parameter group: Cay(au) Cay(bu) = Cay(((a + b)/(1 − ab/4)) u) (tan addition), not Cay((a + b)u). Over 10⁴ draws with \|a\|, \|b\| ≤ 0.01: sup **4.44e-7**, against the claimed 1e-12. At a = b = 0.01: **5.0e-7**. The corrected law holds to 1.1e-16. Merging Baseline 2's 200 fixture steps into one Cayley step moves dR by **9.2e-4 rad** |
| Mutants (52 file edits) | 7 retyped replays killed. 45 new: 28 killed, 6 equivalent, **10 non-equivalent survivors**, 1 (S05) equivalent on every claim domain but outside the remediation's declared domain (advisory) |
| Hangs | **None.** Every run finished well inside `timeout -k 30 1800`: fast set 97-146 s, full suite 218-512 s. No KILLED-BY-TIMEOUT |
| `tests/test_mutation.py` replay | 178/178 pass on the clean tree (inside the 298). That includes `test_pass14_mutant_is_killed_as_a_file_edit[Q01, Q01b, Q01c, Q03, Q05, G01, F33-revert]` and `test_pass14_killers_are_reference_checks` |
| **Gate** | **NOT passed.** 10 non-equivalent survivors: Q09, Q10, Q11 (M2-F37); G02, G03, G04 (M2-F36); S01, S02, S03, S04 (M2-F38) |

## 1. Independent references (no `src` imports)

My scratch directory was `results/p15` (gitignored). I deleted it before committing. No reference imports `src` or `tests`; `src` was called only as the system under test.

* **M1 / M2:** the closed-form scalar coefficients in mpmath at **80 digits**. I cross-checked them against my own mpmath exponential of the 9x9 block generator to ≤ 1.5e-56 at four points: a rate below 1e-8, negative T, x = 6.6, and T = 1000.
* **Series coefficients:** exact rationals (−1)^k/(2k + n)! in mpmath.
* **Exact increments:** a 5x5 block `scipy.linalg.expm` (ZOH), and `quad_vec` for C02.
* **Euler sums:**
  * Eq. (9a)-(9c) retyped, with scipy `Rotation` Exp and with a retyped Cayley;
  * a complex-step Euler (h = 1e-30) for the C05 Jacobians.
* **Covariance (C06):** my own FD-linearised one-step map in the chart (dR Exp(d), v + d, p + d), propagated as C ← A C Aᵀ + B Q Bᵀ. Isotropic and anisotropic densities, B1 and B2.
* **Gal(3):**
  * my own 5x5 matrices and `logm` chart, and FD re-integration for C21;
  * my own endpoint products for C18.
* **C25:** my own GTSAM graph at two density sets.
* **Eq. (19) / (22b):** a retyped Eq. (19) in 40-digit mpmath at the same double inputs. For (22b) the exact ξ = J_R(b_g − b̄_g) is formed in mpmath.
* **C38:** a retyped Cayley, plus mpmath arctan.

**Re-implemented core routines:** the M1/M2 coefficients, the Eq. (9) sums, Eq. (63), Eq. (19), Gal(3) endpoint products, the C21 Jacobian, and the C25 graph.

**Property tests with ≥ 1e4 samples:**

| test | samples |
|---|---|
| C38 | 10⁴ |
| C11 | 2·10⁴ |
| C12 | 10⁴ |
| C08 | 5000 |
| `cay` rule-4 | 10⁴ |
| F34 seeds | 25 × 2000 × 2 |
| F35 seeds | 13 × 2000 |
| rule-4 probes | 4000 each |
| M1/M2 sweeps | about 1.5·10⁴ (w, T) evaluations |

## 2. M2-C38 (brief-only claim): REFUTED

**Claim:** for unit u and \|a\|, \|b\| ≤ 0.01, Cay(au) Cay(bu) = Cay((a + b)u) to 1e-12. So Baseline 2's per-step Cayley retractions over a constant-ω interval could be merged into one step without changing dR.

**Why it is false.** On a fixed axis, Cay(φu) is the rotation by 2 arctan(φ/2). Angles add, so the Cayley parameters compose by the tangent addition law, φ = (a + b)/(1 − ab/4), not a + b. The angle deficit is ≈ ab(a + b)/4. Exp is a one-parameter group; Cay is not. This is the same O(θ³) retraction deficit that M2-C22 measures.

| measurement | value |
|---|---|
| sup over 10⁴ draws (random unit u, a, b ~ U(−0.01, 0.01)), max-abs matrix entry | **4.44e-7** (claim: 1e-12) |
| a = b = 0.01 | 5.00e-7; against Cay(((a + b)/(1 − ab/4))u): 1.1e-16 |
| 40-digit angle gap at a = b = 0.01 | 4.9996e-7 = ab(a + b)/4 to leading order |
| \|a\| = \|b\| = 1e-3 / 2e-4 / 1e-4 | 5.0e-10 / 4.0e-12 / 5.0e-13: the claim's 1e-12 holds only for \|a\|, \|b\| ≲ 1.5e-4 |
| Exp contrast (scipy), same a, b | 0.0 |
| fixture edge 0: 200 per-step Cay((ω − b_g) dt) vs one Cay(Σ) | \|Log\| = **9.23e-4 rad**, max-abs 7.4e-4 |

Merging the steps would change Baseline 2's dR by about 3·10⁴ times its whole Tier-1 gap to Baseline 1 (3.0e-8). Such a merge must not be done.

## 3. Rulings on M2-F33, M2-F34, M2-F35

### 3.1 M2-F33: src fix verified; test grid has holes

**Series algebra (checked by hand and numerically).**
* M1 = T I + T² c₂(θT) ŵ + T³ c₃(θT) ŵ², and M2 = (T²/2) I + T³ c₃ ŵ + T⁴ c₄ ŵ².
* Here c₂ = (1 − cos x)/x², c₃ = (x − sin x)/x³, c₄ = (x²/2 − 1 + cos x)/x⁴.
* Each matches C02's closed form term by term, with K = ŵ/θ. The cᵢ are even, so the sign of T is carried correctly by T², T³ and T⁴.

| check | measured |
|---|---|
| 12 coefficients per cₙ vs exact (−1)^k/(2k + n)! | max relative 1.29e-16 (the rounding of the rationals) |
| series vs 60-digit cₙ, 4000 x on (0, 2.25) | **2.17e-16 = 0.98 ε** relative (comment: ≤ 2.2e-16) ✓ |
| truncation at x = 2, first omitted term | c₃: x²⁴/27! = **1.54e-21** ✓ as commented. **c₂: x²⁴/26! = 4.16e-20** abs (1.2e-19 rel), which the comment omits. Harmless |
| closed form, coefficient relative error (the comment's table) | [1e-3, 1.1e-3): **2.3e-3** (comment: 1.9e-4); [0.1, 0.11): 2.2e-11 (1.3e-11); [0.5, 0.75): 3.7e-14 (2.3e-14); [1, 1.25): 2.0e-15; [1.75, 2): 4.1e-16; [2, 2.25): 5.1e-16. These describe the replaced code only. The small-x figures are understated by up to 12x (c₄ is the worst coefficient). Documentation only |

**`src` M1/M2 vs the 80-digit reference** (three directions; off-diagonal error over the largest off-diagonal reference entry, the test's metric):

| sweep | worst |
|---|---|
| \|w\| ∈ [1e-12, 10] (53 log-spaced), T ∈ {0.01, ±0.5, ±1, ±2.3, ±10} | **≤ 4.65e-16** |
| the same at T = ±100 / 1000 | 2.6e-14 / 2.6e-12, **all from the θ < SMALL_ANGLE branch** (M2-F39); the series and the closed form are ≤ 4.7e-16 there |
| x = θ\|T\| ∈ [1, 3] (201 points) plus 2(1 ± k ulp), k ≤ 4; T = ±1, 0.5, 4 | **≤ 3.4e-16** (series side ≤ 3.1e-16, closed side ≤ 3.4e-16 at x = 2.69). There is no step at the switch |
| \|w\| at 0.5e-8 .. 2e-8 (either side of SMALL_ANGLE) | T = ±0.5: 1.8e-16; ±10: 9.1e-16; 100: **8.3e-14**; 1e3: 8.3e-12; 1e4: 8.3e-10; 1e5: 8.3e-8. All of these are below SMALL_ANGLE. Above it (1.000001e-8, 1.1e-8, 2e-8) they are ≤ 2.6e-16 |
| negative T, x up to 56 | ≤ 4.65e-16. **No unstable regime at negative T** |

**No claim-level output moved.** I compared the old closed form (retyped from C02) with the new `src` on the fixture edges (x = 0.223 / 0.298):

| quantity | change, old to new |
|---|---|
| M1 | ≤ 6.9e-17 |
| M2 | ≤ 1.9e-16 |
| dv | ≤ 6.7e-16 |
| dp | ≤ 1.9e-15 |

* New vs mpmath: M1 ≤ 6.9e-18, M2 ≤ 8.7e-19. Old vs mpmath: ≤ 1.9e-16.
* `exact_states()` vs the literals at KF2: R / v / p = 1.1e-16 / 8.6e-16 / 2.9e-15, against C02's 1e-12.
* My 5x5-`expm` chain vs the literals: 2.2e-16 / 2.6e-15 / 3.3e-15.
* Every Tier-0 consumer (C02, C10, C22-C24) re-measured unchanged (Sec. 4).

**The test.** `test_f33_M1_M2_across_rates_against_mpmath_block_expm` is a genuine reference check: a 40-digit mpmath block exponential, never `src`.
* It kills my F33 revert **RF33** (both branch tests replaced by `if False:`).
* It also kills S08 (`x * abs(x)`), S11 (`abs(T)**3`), S16 (SMALL_ANGLE → 1e-6) and S14.

**The gap.** The grid is 41 rates, 0.25 decades apart, times T ∈ {±0.5, ±1, ±2.3}, so the largest series-side x it samples is 1.29. `test_f27` adds closed-side points at x = 2.55, 3.65 and 5. Nothing samples x ∈ (1.29, 2) or x ∈ (2.3, 2.55) ∪ (2.55, 3.65). Four mutants live there (Sec. 5.4, **M2-F38**).

**Ruling: M2-F33's src defect is FIXED.** Its test is necessary but not sufficient (M2-F38).

### 3.2 M2-F34: survivors killed and the test is robust, but only at the chart edge

**Replays.** My retypes are killed on the intended assertion of `test_f34`, with no other failures. The text differs from the implementer's: `(0.5 + 7e-16)`, `(0.5 + 3.3e-16)`, `(0.5 + 4.4e-16)`, `+ 3.5e-14 * float(norm(xi) > 20)`, and `(R_j @ Xi) * (1 + 1.5e-15 * float(...))`.

**Robustness.** I re-ran the test's distributions with my own code: scipy rotations, so the draws are not bit-identical to the test's. I used 25 seeds, including the test's own seed (20261020), with 2000 draws each against a 40-digit Eq. (19):

| quantity | over 25 seeds | bound | margin |
|---|---|---|---|
| clean Eq. (19) | max 6.82e-15 (5.04e-15 .. 6.82e-15) | 1.3e-14 | **1.91x** headroom |
| clean Eq. (22b) | max **9.30e-15** (seed 1018) | 1.4e-14 | **1.51x** headroom (the implementer stated 1.69x) |
| Q01 / Q01c / Q03, weakest seed | 3.49e-14 / 2.44e-14 / 3.995e-14 | 1.3e-14 | ≥ 1.88x over |
| Q01b, weakest seed | 1.93e-14 | 1.3e-14 | **1.48x** over |
| Q05, weakest seed | 2.40e-14 | 1.4e-14 | 1.71x over |
| max\|r\| on the band | 25.9 .. 27.5 | | |

* The bound sits under the rule-4 threshold for this domain (clean + 2ε·27.5 ≈ 1.9e-14), so anything the test lets through *on the band* is rule-4 equivalent.
* The test is in fact slightly stricter than rule 4. Q12 (½ → ½ + 2 ulp, rule-4 equivalent at about 9.5e-15 excess) is killed.
* **Verdict: robust.** No seed comes near a false failure, and no seed lets a named mutant through.

**The gap.** The new check draws only \|ξ\| ∈ [20, 28.2]. Below 20, the only Eq. (19) / (22b) bounds are still F30 (b) / F31 (c), at 2e-15 (1 + \|ξ\|). That is 4.2e-14 at \|ξ\| = 20, which is the same coarseness pass 14 objected to. Moving the pass-14 defects inward makes them survive the full suite (Sec. 5.4, **M2-F37**):

| mutant | change | excess | rule-4 allowance |
|---|---|---|---|
| Q09 | ½ → ½ + 6 ulp for \|ξ\| < 20 | 2.2e-14 | 8.3e-15 |
| Q10 | + 2e-14 for 12 < \|ξ\| < 20 | 2.1e-14 | 8.4e-15 |
| Q11 | Y′ x (1 + 1.5e-15) for 8 < \|J_R b_g\| < 18 | 1.1e-14 | 7.2e-15 |

**Ruling:** Q01, Q01b, Q01c, Q03 and Q05 are **FIXED**. The F34 class is **not closed** (M2-F37).

### 3.3 M2-F35: G01 killed; the domain question is still open

**Replay.** RG01 is my retype, `Gamma * (0.5 + 5e-13 * float(norm(J_R bbar_g) > 5))`. It is killed by `test_c11` (the new large-Γ loop) and by `test_f35`.

**Robustness over 13 seeds,** including the test's 20261021:

| quantity | measured | bound | margin |
|---|---|---|---|
| clean | 1.01e-14 .. **1.69e-14** | 5e-14 | **2.97x** headroom |
| G01 | ≥ 1.92e-11 at every seed | 5e-14 | about 380x over |

**Ruling: G01 FIXED.** But M2-F35's own first remedy, stating C11's Γ domain, was not taken. Sampling \|Γ\| ∈ [2, 40] cannot close an unbounded domain:
* **G02** (Γ/2 x (1 + 1e-12) for \|Γ\| > 45) survives the full suite. Excess 4.9e-11.
* **G03** (the same for \|Γ\| > 5 **and** \|ξ\| > 5) survives. No test draws large Γ and large ξ together: F35 and the `test_c11` loop keep \|ξ\| ≤ 1.7, and F34 keeps \|Γ\| ≤ 1.7. Excess 2.0e-11.
* **C11 as literally stated cannot hold on an unbounded Γ domain, even for the clean code.** Clean identity error over 2000 draws (`J_R` a scaled rotation, \|b − b̄\| ≤ 0.87):

  | \|Γ\| | clean identity error |
  |---|---|
  | 40 | 1.8e-14 |
  | 100 | 3.9e-14 |
  | 1e3 | **3.4e-13** |
  | 1e4 | **3.6e-12** |

  At \|Γ\| ≈ 1e3 and above, this exceeds C11's 1e-13. The cancellation is inherent in the lifted form: Γ/2 enters with both signs.
* **G04** (Γ/2 x (1 + 2 ulp) for \|Γ\| > 5) survives. Its excess is 4.8e-15 against 2ε·max\|r\| = 1.0e-15, because max\|r\| is only 2.24 on that domain while the intermediates are about 40. Under rule 4's literal "magnitude of the quantity" it is non-equivalent. If the owner reads the magnitude as the largest intermediate (about 40, allowance about 1.8e-14), it is equivalent. I record it as non-equivalent and flag it for that ruling.

New finding **M2-F36**.

## 4. Per-claim verdicts

All numbers are mine and were taken on `a5f87f2`.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. 12/12 constants equal the retyped literals. R_0 vs scipy: 5.6e-17 |
| C02 | CONFIRMED | M1 / M2 vs `quad_vec`: e0 5.6e-17 / 2.8e-17, e1 1.1e-16 / 2.8e-17. Chain vs literals: KF1 v/p 1.3e-15 / 2.0e-15; KF2 R/v/p 1.1e-16 / 8.6e-16 / 2.9e-15. My `expm` chain: 2.2e-16 / 2.6e-15 / 3.3e-15. The F33 change moved fixture outputs by ≤ 1.9e-15. **Domain caveats: M2-F38 (x ∈ (1.3, 2) and (3, 3.5) are unpinned), M2-F39 (θ < 1e-8 at large T)** |
| C03 | CONFIRMED | GTSAM PIM-G `predict` vs the formula, from a non-trivial state: R 7.3e-15 / 7.0e-15, v 0, p 0 |
| C04 | CONFIRMED | n = 100..1600. e0: v 0.50002 → 0.50000, p 0.50086 → 0.50011. e1: v 0.50003 → 0.50000, p 0.50088 → 0.50011. `src` vs my Euler ≤ 4.5e-13 |
| C05 | CONFIRMED | FD J vs my complex step, relative: J_R 6.7e-12; J_vg, J_va, J_pg, J_pa ≤ 6.7e-10. Error ratio over 20 directions per edge: [99.89, 100.07] |
| C06 | CONFIRMED | B1 and B2, isotropic and anisotropic, both edges: asymmetry ≤ 7.5e-20, min eigenvalue ≥ −4.0e-23, trace strictly increasing. Vs my FD-linearised reference: ≤ 1.4e-7 block-relative (limited by my FD at ε = 1e-7) |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 1.5e-15. (b) right 6.0e-7. Left: sup 1.37, minimum 0.022 over 300 unnormalised d. The ≥ 0.1 bound is a sup, as the test comment states |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 3.6e-15; minimum of the right-hand side 1.0000039 |
| C09 | CONFIRMED | 500 draws: on-manifold 2.6e-16; perturbed minimum 6.2e-4 |
| C10 | CONFIRMED | dv̄ / dp̄ vs my Cayley Euler: 0.0. Self-consistent: 2.7e-15. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.377e-4 / 6.811e-4. Matches \|dv̄ − dv_exact\| to ≤ 1.2e-15 |
| C11 | CONFIRMED | 2·10⁴ draws, J_R ~ N(0, 1), \|b\| ≤ 0.5: 1.8e-15 (vs `src` Eq. 19 and vs my retype). \|Γ\| ∈ [2, 40]: ≤ 1.7e-14. **On the test-hint distribution. C11's literal, unbounded-Γ reading fails for the clean code at \|Γ\| ≳ 300 (3.4e-13 at 1e3): M2-F36** |
| C12 | CONFIRMED | 10⁴ draws: 1.3e-15 relative; W = diag(1, 2, 5): 501 ≥ 1; τ(diag(0.5, 0.6, 0.7)) = 1.6666666666666665 |
| C13 | CONFIRMED | `J_R` bytes unchanged after predict / residual at a perturbed bias. 200 directions at \|ξ\| = 1e-2: minimum gap 3.12e-5 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3, 4] = 7; accessors as stated; `Gal3(M)` round trip exact |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. Expmap vs `expm` (100 ξ): 2.7e-15. Expmap(1e-6 e_k) moves only v (k = 3), p (6), t (9). Remap (0, 6, 3) |
| C16 | CONFIRMED | KF0→KF2 vs `localCoordinates`: **0.107628** (KF0→1 0.100839, KF1→2 0.167801). Vs `NavState.Logmap`: 0.0 on all three |
| C17 | CONFIRMED | Accessors and `vector()` as stated; `biasHatVector` = [b_a; b_g]; the adapters round-trip; a single swap is not the identity |
| C18 | CONFIRMED | My 5x5 products: W form ≤ 8.9e-16 at t_i ∈ {0, 3.7}, with t_j − t_i = Dt. F2 form: 0 at t_i = 0; **18.148500** / **36.297000** at t_i = 3.7. Swapped at t_i = 0: 2.4525 / 9.81. `src` W vs mine ≤ 1.8e-15 |
| C19 | CONFIRMED | PIM-G vs ZOH ≤ 2.0e-14; Tangent dv 2.171e-3 / 2.760e-3. Stream bit-equal to my retype. Right composition 3.3e-16; left at step 40: 6.17e-4 |
| C20 | CONFIRMED | My FD J: J[0:3, 0:3] = J[9, :] = 0. Right composition ≤ 4.8e-10. Component-wise sup at 5e-2, 30 directions: **1.05e-2** / **1.41e-2** |
| C21 | CONFIRMED | `src` recursion vs my `logm` FD re-integration: 3.8e-9 / 2.0e-9 / 2.9e-9 relative at steps 10 / 20 / 40. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1−B2: R **2.9965e-8**, v 6.1672e-8, p 4.3763e-8. Per edge, my Euler: 2.325744e-8 / 5.517531e-8 vs the formula 2.325744e-8 / 5.517533e-8. dt² ratios 0.25, 0.25, 0.25, 0.24999 |
| C23 | CONFIRMED | B1−B3: R 3.1e-15, v 1.2867e-3, p 1.2031e-3; B2−B3 R 3.0e-8. Ratios: B1 v 0.50028 → 0.50003, p 0.5 x4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.5 |
| C24 | CONFIRMED | B3 vs my ZOH chain: KF1 8.9e-16 / 1.6e-14 / 2.1e-15, KF2 1.1e-15 / 4.9e-15 / 5.1e-15. B1 R 9.7e-15 / 2.8e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My own graph, two density sets: ranks 30 (30x45), 39 (39x45), 45 (45x45). Cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]** at both; pose/velocity energy 2.5371; error at truth ≤ 6.9e-23 |
| **C38** | **REFUTED** | Sec. 2. **4.44e-7** vs 1e-12; Cayley composes by tan addition |

**Counts: 25 CONFIRMED, 1 REFUTED (C38), 0 UNTESTABLE.**

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh copy of `src tests orchestrator docs conftest.py pyproject.toml`. Every anchor is asserted to occur exactly once.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (120 tests). The unmutated control copy passed it.
* **Every** fast-set survivor was then run on the **full suite** (298 tests).
* All runs used `timeout -k 30 1800` and `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
* The worktree's tracked files never changed.

### 5.1 Mutants

| id | change | result |
|---|---|---|
| RQ01 | Eq. (19) `(0.5 + 7e-16)` (retype of Q01) | killed: f34 Eq. (19) |
| RQ01b | `(0.5 + 3.3e-16)` (Q01b) | killed: f34 Eq. (19) |
| RQ01c | `(0.5 + 4.4e-16)` (Q01c) | killed: f34 Eq. (19) |
| RQ03 | `+ 3.5e-14 * float(norm(xi) > 20)` (Q03) | killed: f34 Eq. (19) |
| RQ05 | `(R_j @ Xi) * (1 + 1.5e-15 * [\|J_R b_g\| > 8])` (Q05) | killed: f34 Eq. (22b) |
| RG01 | `Gamma * (0.5 + 5e-13 * [\|Γ\| > 5])` (G01) | killed: c11 large-Γ loop, f35 |
| RF33 | both M1/M2 series branches `if False:` (F33 revert) | killed: f33 |
| **Q09** | Eq. (19) ½ → ½ + 6 ulp when \|ξ\| < 20 | **survives; NOT equivalent** |
| **Q10** | Eq. (19) + 2e-14 when 12 < \|ξ\| < 20 | **survives; NOT equivalent** |
| **Q11** | Eq. (22b) Y′ x (1 + 1.5e-15) when 8 < \|J_R b_g\| < 18 | **survives; NOT equivalent** |
| Q12 | Eq. (19) ½ → ½ + 2 ulp everywhere | killed: f34 Eq. (19) |
| **G02** | Eq. (22b) Γ/2 x (1 + 1e-12) when \|Γ\| > 45 | **survives; NOT equivalent on C11's literal domain** |
| **G03** | the same when \|Γ\| > 5 and \|ξ\| > 5 | **survives; NOT equivalent on C11's literal domain** |
| **G04** | Γ/2 x (1 + 5e-16) when \|Γ\| > 5 | **survives; NOT equivalent (rule-4 resolution limit)** |
| **S01** | `_M_SERIES_TERMS = 9` | **survives; NOT equivalent** |
| **S02** | `_M_SERIES_TERMS = 10` | **survives; NOT equivalent** |
| **S03** | `M_SERIES_SWITCH = 3.0` | **survives; NOT equivalent** |
| **S04** | `M_SERIES_SWITCH = 3.5` | **survives; NOT equivalent** |
| S05 | switch test `th * T` (no `abs`) in both | survives; equivalent for T ≥ 0 (advisory: Sec. 5.3) |
| S06 | `M_SERIES_SWITCH = 1.0` | survives; equivalent |
| S07 | `M_SERIES_SWITCH = 1.5` | survives; equivalent |
| S08 | `x2 = x * abs(x)` | killed: f33 |
| S09 | M1 uses (c₃, c₂) swapped | killed: c02, c04, c05b, c10, c16, c19, … |
| S10 | Horner over `coeffs` not `reversed` | killed: c02, c04, … |
| S11 | M1 `abs(T)**3` | killed: f33 |
| S12 | SMALL_ANGLE branch disabled in M1 and M2 (`th < 0.0`) | survives; equivalent (better). The full suite shows only W17/W18 replay failures (Sec. 5.5) |
| S13 | `M_SERIES_SWITCH = 2.0000000000000004` | survives; equivalent |
| S14 | M2 series at θT(1 + 1e-9) | killed: c02, f33, … |
| S15 | c₅ instead of c₄ | killed: c02, … |
| S16 | M1 SMALL_ANGLE → 1e-6 | killed: f33 |
| F02 | A_pR coefficient 0.5 → 0.25 | killed: c06b, … |
| F03 | B_g uses Jrᵀ | killed: c06b, … |
| F04 | Q_a x dt instead of / dt | killed: c06, c06b, … |
| F05 | J_R one-sided difference | killed: c05b, c13b, c13e, … |
| F06 | Eq. (44) left composition | killed: c05, … |
| F08 | Jr K-term sign | killed: c06b, … |
| F09 | Jr series drops K²/6 (θ < 1e-8) | survives; equivalent |
| C01 | `cay` as `(I + P) inv(I − P)` | survives; equivalent (better) |
| C03 | `1 + \|ξ\|²/4.000001` | killed: c08, b2_cayley_maps |
| C04 | truncation I + ξ̂/2 | killed: c13, c13c, b2_rotation_residual_functions |
| C06 | τ numerator 3.0000001 | killed: c12, b2_precision |
| C08 | Rbreve = (I − Γ/2) dR̄ | killed: c11, … |
| C09 | J_Cay transposed in the noise input | killed: f26, f28, f29 (complex-step covariance references) |
| C10 | Eq. (21b) A_va term sign | killed: c10b, … |
| G06 | gravity removal ½ g Dt | killed: b3_corrected_increment_gal3 |
| G07 | J FD `/ eps` | killed: c20, … |
| G08 | left composition model | killed: c20, … |
| G09 | component-wise model v/p tangent swapped | killed: c20 (contrast pinned to first order) |
| G10 | `sigma_int` not squared | killed: c25b, … |
| G11 | bias random-walk covariances exchanged | killed: c25b, b3_densities, … |
| G12 | constructor passes (σ_a, σ_g) | killed: c06, b3_constructor |
| G13 | `integrate_stream` drops the last sample | killed: b3_integrate_stream |

**Counts (52 file-level mutants):**

| group | killed | equivalent | non-equivalent survivors |
|---|---|---|---|
| 7 retyped replays | 7 | | |
| 7 Eq. (19) / (22b) probes (Q09-Q12, G02-G04) | 1 (Q12) | | 6 |
| 16 trajectory series / switch (S01-S16) | 7 | 4 (S06, S07, S12, S13) + S05 (claim domain) | 4 |
| 22 other (F02-F09, C01-C10, G06-G13) | 20 | 2 (F09, C01) | |

### 5.2 `tests/test_mutation.py` replay

On the clean tree all 178 tests in `test_mutation.py` pass (inside the 298). That includes:
* `test_pass14_mutant_is_killed_as_a_file_edit[Q01, Q01b, Q01c, Q03, Q05, G01, F33-revert]` and its control copy;
* `test_pass14_killers_are_reference_checks`.

I read all three killers:
* `test_f34` and `test_f35` compare `src` with a retyped 40-digit mpmath Eq. (19), with the exact ξ formed in mpmath;
* `test_f33` compares `src` with a 40-digit mpmath 9x9 block exponential.

None compares `src` with a copy of `src`. My retypes RQ01-RG01 and RF33 agree with the replays' kills (Sec. 5.1).

### 5.3 Equivalence rulings

ε = 2.22e-16. "Excess" is mutant sup error minus clean sup error, both against an 80-digit (M1/M2) or 40-digit (Eq. 19, `cay`) mpmath reference, over the affected domain.

| mutant | quantity, domain | measured | rule | ruling |
|---|---|---|---|---|
| S06 | M1/M2, x ∈ [1, 2], T ∈ {0.5, 1, 2} | abs sup 3.86e-16 vs clean 3.90e-16 (not worse); allowance 2ε·1.96 = 8.7e-16 | 4 | equivalent |
| S07 | M1/M2, x ∈ [1.5, 2] | 3.905e-16 vs 3.901e-16; excess 4e-19 ≤ 8.5e-16 | 4 | equivalent |
| S12 | M1/M2, θ < 1e-8 | not worse anywhere: at T ≤ 10 identical to ≤ 1 ulp; at T = 1e2..1e5 off-diagonal 2.3e-16..2.8e-16 vs clean 8.3e-14..8.3e-8 (it fixes M2-F39) | 4 | equivalent (better) |
| S13 | M1/M2, x ∈ 2 ± 1e-10 | 4.262e-16 = clean, bit-identical sup | 4 | equivalent |
| F09 | `so3_right_jacobian`, θ < 1e-8 | dropped term ≤ θ²/6 ≤ 1.7e-17 ≤ 2ε | 4 | equivalent |
| C01 | `cay`, 10⁴ φ (half up to the chart edge, half \|φ\| ≤ 0.02) | mutant 1.23e-15 vs clean 1.77e-15 (better) | 4 | equivalent |
| S05 | M1/M2 | **Proof:** for T ≥ 0, `th * T == th * abs(T)` bitwise, so the edit is the identity. Every claim's T is an edge duration (C02: M1(T) = ∫₀ᵀ, chained with Dt > 0). For T < 0 with θ\|T\| > 2 the mutant evaluates the 12-term series far outside its range: 3.9e-2 abs at x ∈ [2, 10]. That is outside every claim, but **inside the domain the F33 remediation itself declares** (input-dependence table: "T (both signs)") | 3 | equivalent on claim domains; advisory (M2-F39) |

### 5.4 Complete survivor list (non-equivalent)

Rule-4 numbers are sup-norms over the affected domain against mpmath (4000 draws per Eq. (19) / (22b) probe; 540 (w, T) evaluations per series probe).

| mutant | change | sup error mutant (clean) | excess vs 2ε·max\|value\| | breaks a claim? |
|---|---|---|---|---|
| **Q09** | ½ → ½ + 6 ulp for \|ξ\| < 20 | 2.58e-14 (3.63e-15) | **2.2e-14** vs 8.3e-15 (max\|r\| 18.7) | no (C11 1e-13) |
| **Q10** | + 2e-14 for 12 < \|ξ\| < 20 | 2.57e-14 (4.77e-15) | **2.1e-14** vs 8.4e-15 | no |
| **Q11** | Y′ x (1 + 1.5e-15) for 8 < \|J_R b_g\| < 18 | 1.48e-14 (3.82e-15) | **1.1e-14** vs 7.2e-15 (1.5x; marginal) | no |
| **G02** | Γ/2 x (1 + 1e-12) for \|Γ\| > 45 | 4.93e-11 (3.0e-14) | **4.9e-11** vs 1.0e-15 | **yes**, C11's 1e-13, if C11's domain includes \|Γ\| > 45 (it states none) |
| **G03** | the same for \|Γ\| > 5 and \|ξ\| > 5 | 1.97e-11 (1.24e-14) | **2.0e-11** vs 1.2e-14 | **yes**, same proviso |
| **G04** | Γ/2 x (1 + 2 ulp) for \|Γ\| > 5 | 1.71e-14 (1.23e-14) | **4.8e-15** vs 1.0e-15 (max\|r\| 2.24; intermediates about 40) | no; resolution limit, owner ruling requested |
| **S01** | 9 series terms | M1/M2 at x ∈ [1.3, 2]: **3.29e-13** (3.22e-16) | 3.3e-13 vs 8.6e-16 | **yes**, C02's 1e-14 (for rates ≈ 1.3-2 rad/s at T = 1 s) |
| **S02** | 10 series terms | 2.78e-15 (3.22e-16) | 2.5e-15 vs 8.6e-16 | no (rule 1: > 1e-15 too) |
| **S03** | switch 3.0 | x ∈ [2, 3.5]: 3.06e-15 (5.36e-16) | 2.5e-15 vs 8.2e-16 | no |
| **S04** | switch 3.5 | x ∈ [2, 3.5]: **1.41e-13** (5.36e-16) | 1.4e-13 vs 8.2e-16 | **yes**, C02's 1e-14 |

**Why they survive.**
* **Q09-Q11:** F34's draws are confined to \|ξ\| ≥ 20. Below that, F30 (b) / F31 (c) bound 2e-15 (1 + \|ξ\|), which is about 4.5x the rule-4 allowance (M2-F37).
* **G02-G04:** no test samples \|Γ\| > 40, nor large Γ and large ξ together. F35's 5e-14 is about 3x clean but about 50x 2ε·max\|r\| on its own domain (M2-F36).
* **S01-S04:** the F33 grid's largest series-side x is 1.29, and its closed-side points are 2.3, 2.55, 3.65 and 5 (with f27). S01 and S02 act on (1.29, 2), and S03 and S04 on (2.3, 3.65) (M2-F38).

### 5.5 Full-suite runs of all fast-set survivors

| mutant | full suite (298) | notes |
|---|---|---|
| CTRL, S01, S02, S05, C01 | **298 passed** | |
| Q09, Q10 | 284 passed, **14 errors** | pass-12..14 replay anchor `return (R_j - A) - 0.5 * …` not unique (fixture, not a kill) |
| Q11 | 291 passed, 7 errors | pass-14 anchor `Yp = R_j @ Xi` |
| G02, G03, G04 | 283 passed, 15 errors | pass-11..14 anchor `Rbreve = …` |
| S03, S04, S06, S07, S13 | 291 passed, 7 errors | pass-14 anchor `M_SERIES_SWITCH = 2.0` |
| F09 | 277 passed, 21 errors | pass-8 anchors |
| S12 | 296 passed, **2 failed** | `test_pass8_mutant_is_killed_as_a_file_edit[W17, W18]`: those replays edit the SMALL_ANGLE branch that S12 disables, so they "survive". That is a replay-fixture interaction, not a reference test seeing S12 |

Every error is the replay harness's "edit anchor … not unique" assertion. **No failure is a kill by a reference test.** No run hung.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over, unchanged:**
* `test_m04` (an incidental bitwise reference);
* `test_pass5_replay_is_faithful`;
* the two F28 retyped-residual tests whose clean value is 0.0 by construction;
* the location-only `test_pass11..14_killers_are_reference_checks`. These are sound only because the killers were read (Sec. 5.2).

**Noted in this pass:**
* `test_f33`'s grid is a rate × duration lattice with 0.25-decade spacing. Its near-switch coverage is effectively one point each side (x = 1.29 and 2.3) (M2-F38).
* The pass-8 replays W17 / W18 target the M1/M2 SMALL_ANGLE branch. If that branch is removed as the M2-F39 fix, they must be retired with it, or they become unkillable.
* The `trajectory.py` comment's closed-form error table understates the small-x errors by up to 12x, and its truncation bound is c₃'s rather than c₂'s (Sec. 3.1). Documentation only.

## 7. New findings

* **M2-F36 (gate-blocking as C11 is written).** C11 still states no domain in Γ = J_R b̄_g.
  * The new large-Γ draws stop at \|Γ\| = 40, and nothing draws large Γ together with large ξ. G02 and G03 break the C11 identity by 2e-11 to 5e-11 unseen.
  * More fundamentally, C11's 1e-13 is unattainable on an unbounded Γ domain by the **clean** code: 3.4e-13 at \|Γ\| = 1e3 and 3.6e-12 at 1e4, from the inherent Γ/2 cancellation.
  * G04 sits at the rule-4 resolution limit (allowance 1.0e-15 on an output of magnitude 2.24, with intermediates of about 40).
  * **Remedy:** state C11's domain, for example \|J_R b̄_g\| ≤ 40 and \|ξ\| ≤ 28.2 (or a physical \|Γ\| ≤ 2). Add one joint large-Γ / large-ξ draw. Ask the owner whether rule 4's magnitude is the output's or the largest intermediate's (that decides G04).
* **M2-F37 (gate-blocking under rule 4).** The F34 check covers only \|ξ\| ∈ [20, 28.2]. Inside, the 2e-15 (1 + \|ξ\|) bound still admits Eq. (19) / (22b) defects 1.5x-2.7x the rule-4 allowance: Q09, Q10, Q11.
  * **Remedy:** extend the 40-digit-mpmath absolute check to the whole chart, \|ξ\| ∈ [0, 28.2], banded with a per-band absolute bound (for example 1e-14 on [5, 20)), or bounded by c·ε·max\|r\| per draw.
* **M2-F38 (gate-blocking).** The M2-F33 test leaves x = θ\|T\| ∈ (1.29, 2) and (2.3, 3.65) unsampled, apart from x = 2.55 in f27.
  * S01 (9 terms) and S04 (switch 3.5) break C02's 1e-14 (3.3e-13, 1.4e-13). S02 and S03 are rule-4 non-equivalent (2.5e-15 vs 8.6e-16).
  * **Remedy:** sample x densely on [1, 4] at fixed T. For example \|w\| = x/T for 30 x per unit at T = ±1, against the block exponential.
* **M2-F39 (advisory, latent).** M1 / M2's `th < SMALL_ANGLE` branch truncates after the T³/T⁴ term regardless of T. Its off-diagonal relative error is ≈ x²/12 with x = θ\|T\|:

  | T at θ ≈ 1e-8 | off-diagonal relative error |
  |---|---|
  | 100 s | 8.3e-14 |
  | 1000 s | 8.3e-12 |
  | 1e4 s | 8.3e-10 |
  | 1e5 s | 8.3e-8 |

  * No claim uses T > 2.3. A long stationary edge in M3/M4 would.
  * The new series already covers θ = 0 exactly, so deleting the branch (S12) fixes it with no loss anywhere. Retire W17 / W18 together with it.
  * Also advisory: the input-dependence table now declares "T (both signs)", but no test draws negative T with θ\|T\| > 2.3 (S05, 3.9e-2 there).

## 8. Final state

This commit contains this report only. The worktree is otherwise clean:
* the scratch directory `results/p15` (gitignored) was deleted, including every mutant copy and log;
* `src/`, `tests/` and `docs/claims/` are untouched.

The full suite on the clean tree (`a5f87f2`): **298 passed**.

**Gate: NOT passed.** There are 10 non-equivalent survivors:
* Q09, Q10 and Q11, recorded as **M2-F37**;
* G02, G03 and G04, recorded as **M2-F36**;
* S01, S02, S03 and S04, recorded as **M2-F38**.

The M2-F33 src defect is fixed. M2-F34's and M2-F35's named survivors (Q01, Q01b, Q01c, Q03, Q05, G01) are killed, and F33-revert is killed. M2-C38 is REFUTED: Cayley composes along an axis by tan addition, and the error is 4.44e-7 against 1e-12.
