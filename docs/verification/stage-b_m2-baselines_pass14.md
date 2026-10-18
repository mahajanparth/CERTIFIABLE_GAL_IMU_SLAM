# Stage-B verification, pass 14: M2 baselines (`stage-b/m2-baselines` @ `daebf66`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration. Agent: verify-M2-014.

**Scope of this pass:**
* a ruling on M2-F31 (pass-13 survivors P01-P03) and on the advisory M2-F32 (B07, B76, C07(b)), both answered by `daebf66`, which changes tests only;
* verdicts on M2-C01..C25, plus M2-C37, which this pass's brief supplies and the yaml does not contain;
* P01-P03, B07 and B76 replayed as real exactly-once file edits, retyped by me (R01, R02, R03, R07B, R76);
* seven probes (Q01, Q01b, Q01c, Q03, Q04, Q05, G01) aimed at the new F31 bounds and at Eq. (19) / Eq. (22b);
* 54 further new mutants (L01-L03, D01-D61 not contiguous, X01), each a real file edit;
* a replay of `tests/test_mutation.py`.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing named `*SEEDS*`. I did not browse the git history of `docs/claims`. The only history I diffed is `0e1d946..daebf66`.

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`daebf66`) | full suite **287 passed** in 159 s; a second clean control (single-thread BLAS, under a hard timeout) gave 287 passed in 412 s |
| `src/` and `docs/claims/` change in `0e1d946..daebf66` | **none.** `git diff --stat 0e1d946 daebf66 -- src docs/claims` is empty. Only `tests/test_baselines.py` (a comment), `tests/test_input_dependence.py` and `tests/test_mutation.py` changed |
| **M2-F31** | **P01-P03 are FIXED.** My retypes R01-R03 are killed on the intended part-(b) line. The new bounds stay strictly inside every claim tolerance at every \|ξ\| ≤ 28.2. **No Eq. (19) or Eq. (22b) defect below them breaks a claim's own tolerance inside the C07 chart.** But the bounds are coarser than the owner's equivalence rules. Five non-equivalent mutants survive under them (new finding **M2-F34**) |
| **M2-F32** | **Closed.** B07 (R07B) and B76 (R76) are killed by the two new tests against independent references. The so3_log bound sits just under the rule-4 threshold, so it is well calibrated rather than fragile (Sec. 3.3). The C07(b) wording is resolved by the new test comment |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C37), 0 UNTESTABLE |
| C37 | **REFUTED.** J_Cay = I − [φ]×/2 drops Eq. (4)'s scalar 1/(1 + \|φ\|²/4). Against a central-difference linearisation of Log(Cay(φ)ᵀCay(φ + d)), over 10⁴ φ with \|φ\| ≤ 0.1, the error is **2.49e-3**. The claim says 1e-9. With the denominator kept, the error is 1.8e-11. A 40-digit mpmath derivative gives the ratio 0.99750623 = 4/(4 + \|φ\|²) exactly at \|φ\| = 0.1. The claim holds to 1e-9 only for \|φ\| ≲ 6.3e-5 |
| Mutants (66 file edits) | 5 replays killed on intended lines; 32 new D-series mutants killed by reference tests; Q04 killed (see 5.5); **19 equivalent**; 3 L-series equivalent; **6 non-equivalent survivors (Q01, Q01b, Q01c, Q03, Q05, G01)** |
| Hangs | **None.** Three full-suite runs stalled under machine contention. On a single, timed re-run all three completed: L03 in 413 s, D04 in 231 s, D05 in 234 s. **No mutant is KILLED-BY-TIMEOUT** (Sec. 5.6) |
| `tests/test_mutation.py` replay | 170/170 pass on the clean tree (inside the 287). That includes `test_pass13_mutant_is_killed_as_a_file_edit[P01, P02, P03, B07, B76]` and `test_pass13_killers_are_reference_checks` |
| **Gate** | **NOT passed.** 6 non-equivalent survivors. Five sit inside the F30/F31 bounds (M2-F34). One is outside every tested Γ range, and C11 states no domain for Γ (M2-F35) |
| New advisory | **M2-F33**: `trajectory.M2(w, T)`'s closed form loses all accuracy for small \|w\| above the 1e-8 series switch. The dp error is **0.70 m** at \|w\| = 2e-8 rad/s over 0.5 s. No M2 claim reaches that range, but C02's domain note implies the closed form is valid there |

## 1. Independent references (no `src` imports)

My reference code lived in `results/p14` (gitignored) and was deleted before commit. It never imports `src` or `tests`. `src` was called only as the system under test.

* **Rotations:**
  * `Exp` and `Log` = scipy `Rotation`;
  * a 5x5 block `expm` (ZOH) for exact increments.
* **Euler sums:** retyped Eq. (9a)-(9c), with Exp and with a retyped Cayley.
* **C02 quadrature:** scipy `quad_vec` at 1e-14.
* **Bias Jacobians (C05):** my own complex-step Euler, h = 1e-30.
* **Covariance (C06):** my own FD-linearised one-step map in the chart `R Exp(d)`, `v + d`, `p + d`, propagated as `C ← A C Aᵀ + B Q Bᵀ`. Anisotropic as well as isotropic densities, for both B1 and B2.
* **Gal(3):**
  * my own 5x5 matrices and `logm` chart;
  * my own Ad, from the contract formula, checked against conjugation (1.8e-15) and against `AdjointMap` (2.2e-16);
  * my own FD Jacobians, recursion and endpoint products.
* **C25:** my own GTSAM graph, at two density sets.
* **C37:** a central-difference right Jacobian in double precision, and a 40-digit mpmath derivative.
* **F31:** a 40/50-digit mpmath Eq. (19) at the same double inputs, for rule-4 sup-norms.
* **F32 and so3_log:** a 50/60-digit mpmath Exp.
* **F33:** the closed-form M1/M2 in mpmath and the 5x5 `expm`.

**Re-implemented core routines:**
* Eq. (63) covariance;
* the Eq. (9) sums;
* Eq. (19) / cleared Eq. (16);
* Gal(3) Ad, the C20 Jacobian and the C21 recursion;
* the C18 endpoint products;
* the C25 graph.

**Property tests ≥ 1e4 samples:** C11 (2·10⁴), C09 chart edge (10⁴), C12 (10⁴), C37 (10⁴), D39 (10⁴), D61 (2·10⁴ per decade).

## 2. M2-C37 (brief-only claim): REFUTED

**Claim:** J_Cay(φ) = I − [φ]×/2, "the series terminates", agreeing with a central-difference linearisation to 1e-9 for \|φ\| ≤ 0.1.

**Why it is false:** the right Jacobian of the Cayley map is Rai Eq. (4), (I − [φ]×/2)/(1 + \|φ\|²/4). This is what M2-C07 states, and what `src.cay_right_jacobian` implements. C37 drops the scalar factor.

| measurement | value |
|---|---|
| C37's J vs central difference (h = 1e-6), 10⁴ φ, \|φ\| ≤ 0.1 | **2.49e-3** (claim: 1e-9) |
| Eq. (4) vs the same central difference | 1.79e-11 |
| 40-digit mpmath right Jacobian at \|φ\| = 0.1 vs C37's J | 2.49e-3; vs Eq. (4): 0.0 |
| mpmath J / C37's J (diagonal) | 0.99750623 = 4/(4 + \|φ\|²) |
| \|φ\|²/4 / (1 + \|φ\|²/4), the relative gap | 2.5e-11 at 1e-5, 9.0e-10 at 6e-5, 2.5e-9 at 1e-4, 2.5e-3 at 0.1 |

Mutant D34 makes C37's form the code. It is killed by `test_c07` ("right convention 1.704e-01 > 1e-5") and six input-dependence tests.

## 3. Ruling on M2-F31 and M2-F32

`daebf66` changes tests only. The `src` and `docs/claims` diffs are empty.

### 3.1 The F31 bounds hold inside every claim tolerance

* **Part (b)** is now 2e-15 (1 + \|ξ\|). **Part (c)** is the new chart-edge C11 identity check, also at 2e-15 (1 + \|ξ\|).
* Both bounds are monotone in \|ξ\|, so their maximum on the C07 chart is at \|ξ\| = 2 tan(1.5) = 28.20: **5.84e-14**. That is strictly inside C11's τ = 1e-13 and C09's τ/10 = 1e-13 at every \|ξ\| ≤ 28.2.
* The test asserts this at run time (three `assert ... < 1e-13` lines). I checked them by reading the code and by evaluating the constants.
* **Part (a)** stays a flat 1e-14.

### 3.2 Clean headroom, and the replays

**Clean headroom (my re-measurement, test distributions, 30 seeds × 2000):**

| bound | clean max | headroom |
|---|---|---|
| part (b) | **4.40e-16** per (1 + \|ξ\|) | 4.5x |
| part (c), against the retyped reference | 6.12e-16 | 3.3x |
| part (c), 22b vs 19 | 6.34e-16 | 3.2x |

At the tests' own seeds: (b) 4.05e-16, (c) 4.40e-16 / 4.44e-16.

**Replays.** These are my own retypes, with text different from the implementer's (Sec. 5.1). Each is killed on the intended part-(b) assertion line:

| retype | change | part (b) measured |
|---|---|---|
| R01 | `h = 0.5 + 4e-15 * (norm(xi) > 10)` | 7.016e-15 > 2e-15 |
| R02 | `r + 2e-13 if norm(xi) > 20` | 9.624e-15 > 2e-15 |
| R03 | `s = 1 + 8e-15 if norm(xi) > 5` | 7.119e-15 > 2e-15 |

**P01-P03 are killed, and the stated F31 gap (claim-breaking defects under the bound) is closed.**

**Can a defect below the bounds still break a claim?** I hunted for one and did not find any inside the chart. The largest Eq. (19) deviation that survives (b) and (c) is about (2e-15 − 4.4e-16)(1 + 28.2) ≈ 4.6e-14 at the edge. Added to the clean identity error of 7.1e-15, that is < 1e-13. For Eq. (22b), (c) and F29 together bound the same way.

**But the bounds are coarser than the equivalence rules.** Rule 1 needs ≤ τ/10 = 1e-14. Rule 4 allows clean + 2 eps max\|r\| = 7.1e-15 + 1.17e-14, measured with max\|r\| = 26.3. The test bound admits up to 5.8e-14 at the chart edge. Five mutants live in that window (Sec. 5.4, M2-F34).

### 3.3 M2-F32 (advisory)

**`test_f32_has_noise_is_false_at_zero_densities` is a genuine check against the stated rule.**
* It kills my retype R07B, the σ_g half (`sigma_g >= 0`), not the σ_a half the implementer replayed: "has_noise is True, not False".
* Pass 13's B07 equivalence ruling still stands for the covariance itself. Only the flag sees it, and the flag is now pinned.

**`test_f32_so3_log_between_2pi_3_and_162_degrees_against_mpmath` (2e-15).**
* It kills R76 (`LOG_NEAR_PI_COS = -0.9`) at 145.1° (2.220e-15 > 2e-15).
* **Clean, 12 seeds × 1600 draws:** 8.88e-16 at every seed (min 4.44e-16). The errors are quantised in steps of 4.44e-16, so clean would have to grow by more than two quanta to fail.
* **Fragility:** low. The test is deterministic and seeded, and 2.25x is in whole quanta.
* **Calibration:** the rule-4 threshold on this domain is clean + 2 eps × 2.83 = **2.14e-15**. That is just above the bound. Anything the test lets through is therefore essentially equivalent:
  * L03 (cut at −0.8) survives, with excess 6.7e-16 ≤ 1.11e-15, so it is equivalent;
  * L01 and L02 are equivalent;
  * R76's kill margin is one quantum.
* My verdict: well calibrated, not fragile.

**C07(b).** The new comment in `test_c07` states that the left-convention bound is a sup. I re-measured: sup 0.381, minimum 0.033 over 300 unit directions. The advisory is resolved.

## 4. Per-claim verdicts

All numbers are mine.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 14 constants equal the retyped literals exactly. R_0 vs scipy: 1.1e-16 |
| C02 | CONFIRMED | Retyped M1/M2 vs `quad_vec`: 7.6e-17 / 1.8e-16 (e0) and 5.6e-17 / 9.7e-17 (e1); `src` equals mine bitwise. 5x5-`expm` chain vs the literals: KF1 v/p 1.6e-15 / 2.0e-15; KF2 R/v/p 2.2e-16 / 2.6e-15 / 3.3e-15. **Domain caveat: M2-F33** |
| C03 | CONFIRMED | GTSAM PIM-G `predict` vs the formula: R 8.9e-15, v 8.9e-16, p 0 (e0); 7.8e-15 / 8.9e-16 / 2.8e-17 (e1) |
| C04 | CONFIRMED | Ratios for n = 100..1600. e0: v 0.50002 → 0.50000, p 0.50086 → 0.50011. e1: v 0.50003 → 0.50000, p 0.50088 → 0.50011. `src` vs my Euler ≤ 8.8e-15 |
| C05 | CONFIRMED | FD J vs my complex step, relative: J_R 4.4e-12 / 1.0e-11, J_v ≤ 7.4e-10, J_p ≤ 7.7e-10. Error ratio, 20 directions per edge: [99.89, 100.07] |
| C06 | CONFIRMED | B1 and B2, isotropic and anisotropic: asymmetry ≤ 2.5e-20, min eigenvalue ≥ −1.0e-22, trace strictly increasing. Vs my FD-linearised reference: 1.4e-9 .. 5.7e-9 relative |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 2.1e-15. (b) right 5.0e-8, left sup 0.381 (minimum 0.033) |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 3.6e-15; minimum 1.0000000 |
| C09 | CONFIRMED | 500 draws: on-manifold 4.4e-16; perturbed minimum 6.4e-4. σ_min(I − ξ̂/2) over 10⁴ chart ξ: 1.000000, so there is no spurious zero. Chart-edge on-manifold over 10⁴ draws: 1.2e-14 |
| C10 | CONFIRMED | dv̄/dp̄ vs my Cayley Euler: ≤ 5.3e-14. Self-consistent: 8.9e-16. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.377e-4 / 6.811e-4. Matches \|dv̄ − dv_exact\| to ≤ 3.6e-15 |
| C11 | CONFIRMED | 2·10⁴ draws, J_R ~ N(0, 1), \|b\| ≤ 0.5: 8.9e-16 (22b vs `src` 19, and vs my retype). Chart edge: 7.1e-15. **Domain in Γ unstated: M2-F35** |
| C12 | CONFIRMED | 10⁴ draws: 1.2e-15 relative; W = diag(1, 2, 5): 62.8 ≥ 1; τ(diag(0.5, 0.6, 0.7)) = 3/1.8 |
| C13 | CONFIRMED | `J_R` bytes unchanged after predict / residual at a perturbed bias. 200 directions at \|ξ\| = 1e-2, on-manifold: minimum gap 3.15e-5 / 3.48e-5 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3, 4] = 7; accessors as stated; `Gal3(M)` round trip exact |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. Expmap vs `expm`: 2.4e-15. Expmap(1e-6 e_k) moves only v (k = 3), p (6), t (9). Remap (0, 6, 3) |
| C16 | CONFIRMED | KF0→KF2 vs `localCoordinates` **0.107628** (KF0→1 0.100839, KF1→2 0.167801); vs `Logmap` ≤ 1.3e-15 |
| C17 | CONFIRMED | Accessors and `vector()` as stated; `biasHatVector` = [b_a; b_g]; the adapters round-trip; a single swap is not the identity |
| C18 | CONFIRMED | My 5x5 products: W form ≤ 7.1e-15. F2 form: 0 at t_i = 0; **18.148500** / **36.297000** at t_i = 3.7. Swapped at t_i = 0: 2.4525 / 9.81. `src` W vs mine ≤ 8.9e-16 |
| C19 | CONFIRMED | PIM-G vs ZOH ≤ 2.0e-14; Tangent dv 2.171e-3 / 2.760e-3. Stream retype equal. Right composition 4.4e-16; left at step 40: 6.26e-4 |
| C20 | CONFIRMED | My FD J: J[0:3, 0:3] = J[9, :] = 0. Right composition ≤ 2.3e-10. Component-wise at 5e-2, 30 directions: sup **1.19e-2** / **1.31e-2**. `src` J vs mine 4.4e-9 |
| C21 | CONFIRMED | `src` recursion vs my FD re-integration: 7.5e-9 / 5.6e-9 / 5.8e-9 relative at steps 10 / 20 / 40. My recursion with my Ad: ≤ 3.0e-8. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1−B2: R **2.9965e-8**, v 6.1672e-8, p 4.3763e-8. Per edge 2.325744e-8 / 5.517531e-8 vs the formula 2.325744e-8 / 5.517533e-8 (≤ 3.3e-7 relative). dt² ratios 0.25, 0.25, 0.25, 0.25001 |
| C23 | CONFIRMED | B1−B3: R 3.1e-15, v 1.2867e-3, p 1.2031e-3; B2−B3 R 3.0e-8. Ratios: B1 v 0.50028 → 0.50003, p 0.5 x4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.5 |
| C24 | CONFIRMED | B3 vs ZOH: KF1 9.2e-16 / 1.6e-14 / 2.1e-15, KF2 1.1e-15 / 4.4e-15 / 5.1e-15. B1 R 9.7e-15 / 2.7e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My own graph, two density sets: ranks 30 (30x45), 39 (39x45), 45 (45x45). Cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**; pose/velocity energy 2.5371; error at truth ≤ 2.2e-23 |
| **C37** | **REFUTED** | Sec. 2. Error **2.49e-3** vs 1e-9, missing 1/(1 + \|φ\|²/4) |

**Counts: 25 CONFIRMED, 1 REFUTED (C37), 0 UNTESTABLE.**

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh copy of `src tests orchestrator docs conftest.py pyproject.toml`. Every anchor is asserted to occur exactly once.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (117 tests). An unmutated control copy passed 117/117.
* **Every** fast-set survivor was then run on the **full suite** (287 tests).
* Later runs used a hard `timeout` and `OMP/OPENBLAS/MKL_NUM_THREADS=1` (Sec. 5.6).
* The worktree's tracked files never changed.

### 5.1 Mutants

| id | change | result |
|---|---|---|
| R01 | Eq. (19) `h = 0.5 + 4e-15 * (norm(xi) > 10)` (retype of P01) | killed: f30 (b) 7.016e-15 |
| R02 | Eq. (19) `r + 2e-13 if norm(xi) > 20` (P02) | killed: f30 (b) 9.624e-15 |
| R03 | Eq. (19) `hat(s * xi)`, s = 1 + 8e-15 for norm > 5 (P03) | killed: f30 (b) 7.119e-15 |
| R07B | `has_noise`: `sigma_g >= 0` (B07, other half) | killed: f32 has_noise |
| R76 | `LOG_NEAR_PI_COS = -0.9` (B76) | killed: f32 so3_log, 2.22e-15 at 145.1° |
| **Q01** | Eq. (19) 0.5 → 0.5000000000000007 | **survives; NOT equivalent** |
| **Q01b** | Eq. (19) 0.5 → 0.5000000000000003 | **survives; NOT equivalent** (marginal) |
| **Q01c** | Eq. (19) 0.5 → 0.5000000000000004 | **survives; NOT equivalent** |
| **Q03** | Eq. (19) + 3.5e-14 when \|ξ\| > 20 | **survives; NOT equivalent** |
| Q04 | Eq. (22b) 0.5 → 0.5000000000000006 in −½(Y + Y′) | killed: `test_m04` (bitwise vs a retyped 22b; Sec. 5.5) |
| **Q05** | Eq. (22b) Y′ x (1 + 1.5e-15) when \|J_R b_g\| > 8 | **survives; NOT equivalent** |
| **G01** | Eq. (22b) Γ/2 x (1 + 1e-12) when \|J_R b̄_g\| > 5 | **survives; NOT equivalent on C11's literal domain** (M2-F35) |
| L01 | `LOG_NEAR_PI_COS = -0.6` | survives; equivalent |
| L02 | `LOG_NEAR_PI_COS = -0.4` | survives; equivalent |
| L03 | `LOG_NEAR_PI_COS = -0.8` | survives; equivalent |
| D01 | `hat` sign slip | killed: c01, c02, … (58 tests) |
| D02 | `so3_exp` series drops K²/2 | killed: f25 series vs mpmath |
| D03 | `SMALL_ANGLE = 1e-6` | killed: f25 so3_log small angles |
| D04 | near-π sign test `<= 0` | survives; equivalent |
| D05 | M1 K coefficient as 2 sin²(θT/2)/θ | survives; equivalent (better than clean) |
| D06 | M2 K² `/ (th * th)` | survives; equivalent |
| D07 | `_PHASE` 2.0 → 2.0000001 | killed: traj_time_varying_stream |
| D09 | `chain` `Dt**2` → `(Dt * Dt)` | survives; equivalent |
| D10 | `B_G + omega_body` | survives; equivalent |
| D12 | so3_log generic `0.5 * th / sin th` | survives; equivalent |
| D13 | near-π angle from `arccos` | killed: f24 near π |
| D14 | no clamp on `c` | killed: c05 (nan), f24 |
| D15 | `Jr` series drops K²/6 | survives; equivalent |
| D16 | `Jr` K² coefficient x (1 + 1e-9) | killed: f28 right Jacobians vs mpmath |
| D17 | dv updated before dp | killed: c04, c05b, c10, c23, c24, … |
| D18 | dp uses `self.dR` for `dR_ik` | survives; equivalent |
| D19 | A_RR = dR_step | killed: c06b, c06c, … |
| D20 | A_pv x (1 + 1e-9) | killed: f23, f26, f27, f28, f29 |
| D21 | Ba position block uses `dR_step` | killed: c06b, c06c, … |
| D23 | `FD_EPS = 1e-5` | survives; equivalent |
| D24 | FD J_R as `Log(R₋ᵀR₊)/2ε` | killed: b1_bias_jacobians_fd |
| D26 | `increment_error` `Log(Raᵀ Rb)` | survives; equivalent |
| D27 | `has_noise` σ_g `np.all` | killed: f27, f28 |
| D29 | `dR` aliases the caller | killed: f18 |
| D30 | cov gate `np.all(cov != 0)` | killed: f23 |
| D31 | `cay` → Cay(−φ) | killed: c05b, c07, c09, c10, c13, … |
| D32 | `cayley_vector` identity below 1e-6 | killed: f25 series branches |
| D34 | C37's J_Cay (no denominator) | killed: c07, b2_cayley_maps, f26, f28, … |
| D36 | truncation `I + ξ̂/2` | killed: c13, c13c, … |
| D37 | Eq. (21b) gDt x (1 + 1e-12) | killed: c10 (4.79e-12 > 1e-12), … |
| D38 | Eq. (21c) ½gDt² x (1 + 1e-13) | killed: f28 eq21c at large v Dt |
| D39 | `weighted_square` Wᵀ | survives; equivalent |
| D40 | `xi` uses b̄_a | killed: c13, c13c, c13d, … |
| D43 | bias adapter not swapped | killed: c03, c17, c19, c20, … |
| D44 | remap (0, 3, 6) | killed: c15, c16, b3_gal3_adapters |
| D45 | accel covariance σ not σ² | killed: c06, b3_make_params, … |
| D46 | `endpoint_W` regrouped | survives; equivalent |
| D47 | `endpoint_Hprime` sign | killed: c18, b3_endpoint_relations |
| D48 | `time_shift(-dt)` | killed: c18, b3_endpoint_relations |
| D49 | `J_FD_EPS = 1e-6` | survives; equivalent |
| D50 | `C_FD_EPS = 1e-5` | survives; equivalent |
| D51 | `0.5 g Dt_ij · dt_total` | survives; equivalent |
| D52 | recursion uses Ad_V not Ad_{V⁻¹} | killed: c21, b3_recursion, f21, f27 |
| D53 | component-wise model left rotation | killed: c20, b3_models_and_distance |
| D55 | `navtriple` v/p swapped | killed: c03, c23, c24, … |
| D56 | `integrate_stream` ignores `n_steps` | killed: c21, b3_integrate_stream |
| D58 | `bias_hat` writable | killed: f22 |
| D59 | `gal3_distance(B, A)` | survives; equivalent |
| D60 | `Gamma * 0.5` | survives; equivalent |
| D61 | τ: `trace x (1 + 4e-16)` | survives; equivalent (on the rule-4 boundary) |
| X01 | `J_Cay` as `x 4/(4 + \|φ\|²)` | survives; equivalent |

**Counts (66 file-level mutants):**

| group | killed | equivalent | non-equivalent survivors |
|---|---|---|---|
| 5 replays | 5 | | |
| 7 probes | 1 (Q04) | | 6 |
| 3 L-series | | 3 | |
| 51 D/X | 32 | 19 | |

### 5.2 `tests/test_mutation.py` replay

On the clean tree all 170 tests in `test_mutation.py` pass (inside the 287). That includes:
* `test_pass13_mutant_is_killed_as_a_file_edit[P01, P02, P03, B07, B76]` and its control copy;
* `test_pass13_killers_are_reference_checks`.

I read all three killers:
* F30 (b) compares `src` Eq. (19) with the retyped cleared Eq. (16);
* the F32 so3_log test compares `src` with a 60-digit mpmath Exp;
* the F32 has_noise test checks the flag against the stated rule.

None compares `src` with a copy of `src`. My independent retypes (Sec. 3.2, 3.3) agree with the kills and their `E ` lines.

### 5.3 Equivalence rulings

ε = 2.22e-16. "Excess" is mutant sup error minus clean sup error, both against an mpmath or exact reference.

| mutant | quantity | measured | rule | ruling |
|---|---|---|---|---|
| L01 | `so3_log`, θ ∈ [120°, 126.9°] | excess 2.2e-16 ≤ 2ε·max\|φ\| = 9.8e-16 | 4 | equivalent |
| L02 | `so3_log`, θ ∈ [113.6°, 120°] | mutant 4.4e-16 vs clean 6.7e-16 (better) | 4 | equivalent |
| L03 | `so3_log`, θ ∈ [120°, 143.1°] | excess 6.7e-16 ≤ 1.11e-15 | 4 | equivalent |
| D04 | `so3_log` at exactly π | **Proof:** a·s_vec = 0 only when s_vec = 0, i.e. θ = π exactly, where ±a are both valid logarithms (the docstring says so; no claim fixes the sign at π) | 3 | equivalent |
| D05 | M1 | mutant 7.8e-16 vs clean **4.1e-10** (the mutant is better; see M2-F33). Fixture M1 change 6.2e-17 | 4 | equivalent |
| D06 | M2 | sup error identical to clean (3.86e-3, both; M2-F33); `th**2` and `th*th` differ by ≤ 1 ulp | 4 | equivalent |
| D09 | `chain` p | excess 0.0 ≤ 2.4e-14; bit-identical at the fixture Dt = 0.5 | 4 | equivalent |
| D10 | stream | **Proof:** IEEE addition is commutative; bit-identical | 3 | equivalent |
| D12 | `so3_log` generic branch | mutant 6.7e-16 = clean 6.7e-16 | 4 | equivalent |
| D15 | `Jr` series (θ < 1e-8) | dropped term ≤ θ²/6 = 1.7e-17 ≤ 2ε = 4.4e-16 | 4 | equivalent |
| D18 | dp update | **Proof:** `dR_ik` is `self.dR` (assigned on the line above, not yet updated) | 3 | equivalent |
| D23 | FD Jacobians (C05 band [50, 200]) | J change ≤ 3.3e-9 relative; C05 ratio [99.854, 100.084] vs clean [99.854, 100.084] | 1 | equivalent |
| D26 | `increment_error` | \|Log(Rᵀ)\| = \|Log R\|; rounding only (no test or claim difference) | 4 | equivalent |
| D39 | `weighted_square` | **Proof:** every claim and caller uses a symmetric (diagonal) W, for which `W.T` is bitwise W; measured diff 0 over 10⁴ diagonal W | 3 | equivalent |
| D46 | `endpoint_W` position | mutant 1.26ε relative vs clean 1.79ε (better) | 4 | equivalent |
| D49 | C20 J (τ = 1e-8) | right-composition residual 4.65e-11 / 3.10e-11 vs clean 4.65e-10 / 3.84e-10 (better) | 1 | equivalent |
| D50 | C21 recursion (τ = 1e-6 relative) | J₄₀ change 7.6e-10 relative ≤ τ/10 | 1 | equivalent |
| D51 | `corrected_increment_gal3` | **Proof:** `Dt_ij = pim.dt_total` on the line above | 3 | equivalent |
| D59 | `gal3_distance` (C20, τ = 1e-8) | asymmetry 6.3e-10 relative at distances 1e-6..0.3, i.e. ≤ 9e-12 absolute | 1 | equivalent |
| D60 | Eq. (22b) R̆ | **Proof:** ×0.5 and /2.0 are both exact in IEEE; bit-identical | 3 | equivalent |
| D61 | τ = 3/tr Σ | Fixture covariance blocks (432 density pairs): excess 8.43e-6 ≤ 2ε·max\|τ\| = 8.88e-6. **On the rule-4 boundary:** on synthetic trace decades the excess is 0.92x-1.04x of the allowance (1e-12: 4.40e-4 vs 4.21e-4; 1e-4: 4.32e-12 vs 4.29e-12). The edit is a 2ε relative shift of τ by construction | 4 | equivalent on the domain where τ is consumed; flagged as marginal |
| X01 | `J_Cay` | excess 0.0 (1.1e-16 both) | 4 | equivalent |

### 5.4 Complete survivor list (non-equivalent)

All rule-4 numbers are sup-norms against a 40-digit mpmath Eq. (19) over 4000 draws, 3000 of them with θ ∈ (2.5, 3). The magnitude is max\|r\| = 26.3, so the allowance is **2ε·max\|r\| = 1.17e-14**. Clean sup errors: Eq. (19) 7.1e-15, Eq. (22b) 3.6e-15.

| mutant | change | test metrics (bound) | sup error vs exact (clean) | excess vs 1.17e-14 | breaks a claim? |
|---|---|---|---|---|---|
| **Q01** | Eq. (19) ½ → 0.5000000000000007 | (a) 4.4e-15 (1e-14); (b) 1.26e-15 (2e-15); (c) 1.34e-15 (2e-15) | 3.91e-14 (7.1e-15) | **3.2e-14** | no: C11 identity 3.55e-14 < 1e-13 |
| **Q01b** | ½ → 0.5000000000000003 | (b) 7.8e-16; (c) 8.0e-16 | 2.13e-14 | **1.42e-14** (1.2x the allowance; marginal) | no |
| **Q01c** | ½ → 0.5000000000000004 | (b) 9.1e-16; (c) 9.4e-16 | 2.84e-14 | **2.1e-14** | no |
| **Q03** | Eq. (19) + 3.5e-14 for \|ξ\| > 20 | (b), (c) ≈ 1.8e-15 | ≈ 4.2e-14 (+4e-14 gives 4.6e-14) | **≈ 3.5e-14** | no: C11 identity ≈ 4.2e-14 |
| **Q05** | Eq. (22b) Y′ x (1 + 1.5e-15) for \|J_R b_g\| > 8 | (c) 9.5e-16; F29 8.9e-16 | 2.49e-14 (3.6e-15) | **2.1e-14** | no: C11 identity 2.49e-14 |
| **G01** | Eq. (22b) Γ/2 x (1 + 1e-12) for \|J_R b̄_g\| > 5 | every test samples \|Γ\| ≲ 2.8 | C11 identity **1.6e-11** at \|Γ\| ∈ (5, 36) | — | **yes**, C11's 1e-13, if C11's domain includes \|Γ\| > 5 (it states none) |

**Why Q01-Q05 survive.** The F30 (b) and F31 (c) bounds are 2e-15 (1 + \|ξ\|). That is 5.84e-14 at the chart edge: 5.8x Rule 1's τ/10 and about 4.5x the rule-4 allowance (clean + 1.17e-14 ≈ 1.9e-14). Errors that grow with \|ξ\| (a coefficient slip on the ξ̂ term), or that are confined to large \|ξ\|, pass up to that level.

**Why G01 survives.** `test_c11` draws \|Γ\| ≲ 0.09, F29 draws ≲ 2.8, and F31 (c) ≲ 1.7. Physically \|Γ\| = \|J_R b̄_g\| ≈ Dt \|b̄_g\| ~ 1e-2, but C11's statement and domain give no bound.

**Rule 4 at the resolution limit.** A flat bound at rule-4 resolution is hard to reach on this distribution. Normalised per draw by max(1, \|r\|), clean is up to 5.5ε (Eq. 19) and 11ε (Eq. 22b) over 40 seeds, which overlaps Q01b's 6.5ε. A remediation therefore needs either:
* a sharper test concentrated where these effects live: draws with \|ξ\| ∈ [20, 28.2] and an absolute bound of about 1.9e-14 (clean 7.1e-15 there). On my measurements that kills Q01, Q01c, Q03 and Q05, with Q01b borderline; or
* an owner ruling on whether rule 4 or the claim tolerance governs test resolution for this identity.

### 5.5 Q04: killed, by an incidental reference

Q04 fails only `test_m04_swapping_y_and_yprime_is_a_genuine_no_op` (286 passed, 1 failed). That test monkeypatches a **retyped** Eq. (22b) with Y and Y′ exchanged, and asserts it equals `src`'s Eq. (22b) **bitwise** on 200 draws with \|b\| ≤ 0.05. So it is a bitwise comparison of `src` against an independent retype, and it legitimately sees Q04.

I count Q04 as killed. Note that the kill is incidental to that test's stated purpose. Q05, which acts only at \|J_R b_g\| > 8, is not reached by it.

### 5.6 Full-suite runs of all fast-set survivors, and the stalled runs

| mutant | full suite (287) | notes |
|---|---|---|
| Q05, D04, D05, D06, D10, D12, D18, D26, D39, D49, D50, D59, X01 | **287 passed** | |
| Q01, Q01b, Q01c, Q03 | 280 passed, **7 errors** | pass-12/13 anchor not unique (fixture, not a kill) |
| L01, L02, L03 | 282 passed, **5 errors** | pass-13 anchor errors |
| G01, D60 | 279 passed, **8 errors** | pass-11/12 anchor errors |
| D15, D61 | 266 passed, **21 errors** | pass-8 anchor errors |
| D23 | 276 passed, **11 errors** | pass-9 anchor errors |
| D09, D46, D51 | 250 passed, **37 errors** | pass-7 anchor errors |
| Q04 | 286 passed, **1 failed** | `test_m04` (Sec. 5.5) |

**None of the errors is a kill by a reference test.**

**Stalled runs.** In my first full-suite batch, L03, D04 and D05 stalled at test 253 of 287 for about 25 hours, until the orchestrator stopped them.
* Test 253 is the `pass11_runs` module fixture, which starts its own thread pool of nested pytest copies.
* Three such suites ran concurrently on a machine at load ~33. The orchestrator traced suite stalls on this machine to OpenBLAS thread oversubscription.
* None of the three edits touches anything that fixture's replays depend on, and each mutant had passed the 117-test fast set in about 100 s.

I re-ran each alone, under `timeout -k 30 1800` with a faulthandler dump, beside an unmutated control:

| run | result |
|---|---|
| CTRL | 287 passed in 412 s |
| L03 | 282 passed + 5 anchor errors in 413 s |
| D04 | 287 passed in 231 s (single-thread BLAS) |
| D05 | 287 passed in 234 s (single-thread BLAS) |

**No mutant hangs, and none is recorded as KILLED-BY-TIMEOUT.** All later runs used hard timeouts and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over, unchanged:**
* `test_m04` (now also an incidental bitwise reference, Sec. 5.5);
* `test_pass5_replay_is_faithful`;
* the two F28 retyped-residual tests whose clean value is 0.0 by construction;
* the location-only `test_pass11/12/13_killers_are_reference_checks`. These are sound here only because the killers were read (Sec. 5.2).

**Noted in this pass:**
* F30 (b) and F31 (c) are sound against claim tolerances but coarser than the equivalence rules (M2-F34).
* No test samples \|Γ\| > ~2.8 (M2-F35).
* No test samples M1/M2 between \|w\| = 1e-8 and 0.29 (M2-F33).
* `test_c12` does not pin τ to the ulp: D61 moves τ(diag(0.5, 0.6, 0.7)) from 1.6666666666666665 to 1.6666666666666659 and passes. This is harmless under rule 4 (Sec. 5.3).

## 7. New findings

* **M2-F33 (advisory, latent defect).** `trajectory.M2(w, T)` (and M1, much more weakly) loses accuracy by cancellation in its K² coefficient (T²/2 − (1 − cos θT)/θ²) for 1e-8 < \|w\| ≲ 1e-2. Against the 5x5 `expm`, with f = fixture f₀ and T = 0.5:

  | \|w\| (rad/s) | dp error | dv error |
  |---|---|---|
  | 2e-8 | **0.70** | 1.7e-8 |
  | 1e-6 | 6.2e-5 | 7.7e-11 |
  | 1e-4 | 4.1e-9 | 5.1e-13 |

  Against mpmath, the M2 matrix is off by 1.2e-1 / 2.9e-3 / 1.1e-5 / 1.0e-8 / 7.3e-10 at \|w\| = 2e-8 / 1e-7 / 1e-6 / 1e-5 / 1e-4. The 1e-14 of C02 is met only for θT ≳ 5e-3.
  * No M2 claim is affected: C02 and every consumer use the fixture rates (\|w\| ≈ 0.4), so C02 stays CONFIRMED.
  * But C02's domain note ("use the series form for \|\|w\|\| < 1e-8") implies the closed form is valid above 1e-8, and that is false.
  * `Edge.exact_increment` is the Tier-0 ground truth, so any later use at low rates (stationary or slow-rotation edges in M3/M4) would be wrong by up to metres.
  * `test_traj_M1_M2` samples only \|w\| ∈ {≈ 3.7e-9, 0.6, 0.78}.
  * **Fix:** a stable series for the cancelling coefficients up to θT ~ 1e-2 (or the 5x5 `expm`), plus a test across 1e-8 .. 1 rad/s against mpmath. D05's 2 sin²(θT/2) form already fixes M1's term.
* **M2-F34 (gate-blocking under rules 1 and 4).** The F30 (b) / F31 (c) bound 2e-15 (1 + \|ξ\|) admits up to 5.84e-14 at the chart edge. Five mutants survive inside it: Q01, Q01b, Q01c, Q03 (Eq. 19) and Q05 (Eq. 22b). Their excess over clean is 1.4e-14 to 3.5e-14, against the rule-4 allowance of 1.17e-14 and Rule 1's τ/10 = 1e-14. None breaks C11's own 1e-13. **Remedy (advisory):** a chart-edge-concentrated check (\|ξ\| ∈ [20, 28.2]) at about 1.9e-14 absolute for Eq. (19) and Eq. (22b), or an owner ruling that for this identity the test resolution need only stay inside τ.
* **M2-F35 (gate-blocking as C11 is written).** C11 states no domain for Γ = J_R b̄_g, and no test draws \|Γ\| above ~2.8. G01 breaks the C11 identity by 1.6e-11 (> 1e-13) for \|Γ\| > 5. **Remedy:** state C11's domain (for example \|J_R b̄_g\| ≤ 2, which covers every physical use), or add a large-Γ draw to F29 / F31 (c).
* **Not a finding:** the "hung" mutants L03, D04 and D05 are not hangs (Sec. 5.6).

## 8. Final state

This commit contains this report only. The worktree is otherwise clean:
* the scratch directory `results/p14` (gitignored) was deleted;
* every mutant copy was a temp directory under it and was removed;
* `src/`, `tests/` and `docs/claims/` are untouched.

The full suite on the clean tree (`daebf66`): **287 passed**.

**Gate: NOT passed.** There are 6 non-equivalent survivors:
* Q01, Q01b, Q01c, Q03 and Q05, recorded as **M2-F34**;
* G01, recorded as **M2-F35**.

M2-F31's named survivors (P01-P03) are killed, and M2-F32 is closed. M2-C37 is REFUTED: J_Cay must keep 1/(1 + \|φ\|²/4); the error is 2.49e-3 at \|φ\| ≤ 0.1.
