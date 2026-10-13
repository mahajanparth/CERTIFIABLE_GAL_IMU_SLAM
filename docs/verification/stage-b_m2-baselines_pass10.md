# Stage-B verification, pass 10: M2 baselines (`stage-b/m2-baselines` @ `eb5011c`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration.

**Scope of this pass:**
* rulings on M2-F26 and M2-F27;
* verdicts on M2-C01..C25, plus M2-C33, which the pass-10 brief supplies and the yaml does not yet contain;
* 59 new mutants, plus 41 earlier mutants replayed as real file edits;
* tests that cannot fail.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing under `.cgis_SEEDS_BACKUP*`. For C02 I needed to know whether `src`'s `REF_*` literals match `docs/02_baseline_reproduction.md`. I counted occurrences of 13 literal strings (`grep -F -c`) and read nothing else in that file.

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`eb5011c`) | full suite **249 passed** in 79.0 s |
| **M2-F26** | **FIXED** |
| **M2-F27** | **PARTIALLY FIXED**. All nine named survivors (X15, X21, X22, X24, X31, X41, W12, P23, S21) are now killed by reference tests. But the gaps behind (b), (c) and (f) are only closed at the specific values the new tests use. 10 new survivors sit in exactly those gaps, or next to them (Sec. 5.4) |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C33), 0 UNTESTABLE |
| New mutants (59) | 38 killed by reference tests; 11 equivalent (numbers in Sec. 5.3); **10 non-equivalent survivors** |
| Replays (41) | 31 killed by reference tests, including all 9 pass-9 survivors, F26 and P23-old; 10 equivalent |
| **Gate** | **NOT passed.** 10 non-equivalent survivors, all low: Y18, Y20, Y24, Y30, Y31, Y55, Y56, Y57, Y58, Y59 |

**Machine load.** The load average reached 23 during the sweeps. No run timed out, and the comment-only control passed every time.

## 1. Independent references (no `src` imports)

My reference code lived in a scratch directory inside the worktree (`results/`, which is gitignored), deleted before commit. It never imports `src`.

* **Rotations:** `Exp` from scipy `expm(hat)` and a separately retyped Rodrigues; `Log` from scipy `Rotation`.
* **Exact increments:** 5x5 `expm` ZOH increments.
* **Euler sums:** retyped Eq. (9a)-(9c).
* **Covariance reference:** a complex-step covariance of my own.
  * It uses the matrix power series of `hat(phi)` for `Exp` (not the test file's scalar series), a `solve`-based complex Cayley map, and step `h = 1e-30`.
  * It linearises the noisy Euler step w.r.t. the state in the chart `(R Exp(d), v + d, p + d)` and w.r.t. additive measurement noise, with `Q = diag(sigma^2)/dt`.
* **Bias-Jacobian reference:** complex-step bias Jacobians, with the same exponential.
* **GTSAM side:** my own Gal(3) matrices, `logm` logarithms and conjugation adjoint.
* **C25:** my own `GalileanCombinedImuFactor` graph, with densities chosen independently of the tests.
* **C33:** mpmath at 40 digits.

`src` was called only as the system under test.

**The complex-step references in the test file (`_cexp`, `_ccay`, `_ceuler`, `_cs_step_jacobians`, `_cs_covariance`, `_cs_bias_jacobians`) are genuinely complex-analytic, and independent of `src`.** I checked them line by line:
* `phi @ phi` on 1-D complex arrays is the non-conjugating bilinear form, so it is analytic.
* `abs(t)` appears only inside an `assert` guard, never in a value.
* `np.asarray(..., dtype=float)` is applied only to real states.
* `ivee` is applied only to the real derivative matrix.
* There is no `norm`, `arccos`, branch or `abs` on any value path.

Against my own complex-step implementation, on a 30-step time-varying stream with anisotropic densities and a nonzero bias:

| reference | covariance (block-relative) | bias Jacobians (relative) |
|---|---|---|
| test `_cs_covariance` with `_cexp` | 4.3e-15 | 5.9e-16 |
| test `_cs_covariance` with `_ccay` | 1.3e-14 | 7.9e-16 |

They are exact to rounding, as stated.

## 2. M2-C33 (brief-only claim): REFUTED

**Claim:** `max|J_Cay(phi) - J_r(phi)| <= |phi|^3` for `|phi| <= 0.1`, so using `J_r` for B2's noise propagation is third order.

**Attack: the claim is false at second order.** Expanding both maps:
* `J_r = I - K/2 + K^2/6 + O(|phi|^3)`
* `J_Cay = (I - K/2)(1 - |phi|^2/4 + ...)`

With `K^2 = phi phi^T - |phi|^2 I`, this gives `J_Cay - J_r = -(phi phi^T / 6 + |phi|^2 I / 12) + O(|phi|^3)`, which is nonzero at **second** order. Its largest entry is `|phi|^2/4`, reached along a coordinate axis.

**Measured** (double precision; mpmath at 40 digits agrees):

| \|phi\| | max\|J_Cay − J_r\| | \|phi\|³ | \|phi\|²/4 |
|---|---|---|---|
| 0.1 | **2.494e-3** | 1.0e-3 | 2.500e-3 |
| 0.05 | 6.246e-4 | 1.25e-4 | 6.250e-4 |
| 0.01 | 2.500e-5 | 1.0e-6 | 2.500e-5 |
| 1e-3 | 2.500e-7 | 1.0e-9 | 2.500e-7 |

* The bound is violated on **10300 of 10300** samples, with |phi| from 1e-4 to 0.1 and random axes. The worst ratio to |phi|³ is 3.4e4.
* The mpmath residual after subtracting the second-order prediction is 3.5e-5 at |phi| = 0.085, which is O(|phi|³), as expected.

**Consequence.** Using `J_r` is a *second*-order (in the per-step angle) error in the noise input. That is exactly F26's 4.6e-7 / 7.2e-7 relative rotation-block error on the fixture: per-step angle about 1.1e-3, |phi|²/4 ≈ 3e-7, accumulated. The two maps `Cay` and `Exp` agree through second order (they differ at O(|phi|³)), so their right Jacobians agree only through **first** order.

## 3. Rulings on M2-F26 and M2-F27

### M2-F26: FIXED

`CayleyPreintegration.retract_right_jacobian` returns Rai Eq. (4)'s `J_Cay` for the raw variant and `Exp`'s `J_r` for the reparameterized contrast. `_propagate_covariance` reads the hook. Both choices are correct:
* `Cay(2 tan(theta/2) a) = Exp(theta a)`, so the reparameterized map's right Jacobian is `J_r`.
* My complex-step references agree.

Against my own complex-step covariance:

| case | clean | F26 replayed |
|---|---|---|
| B1, fixture iso/aniso, block-relative | 1.6e-12 .. 2.1e-12 | — |
| B2, `test_f26` streams (raw and reparam) | ≤ 1e-12 | killed by `test_f26` and `test_f27_has_noise` |

The transition `A_RR = Cay(w dt)^T` was already correct. X24 (an `Exp` transition) is now killed by three reference tests.

### M2-F27: PARTIALLY FIXED

Every item named in pass 9 is addressed, and each pass-9 survivor is killed by a reference test:

| pass-9 survivor | killed by |
|---|---|
| X15 | `test_f27_M1_M2_fast_twists_against_quadrature` |
| X21, X22 | `test_f27_has_noise_per_axis_and_tiny_densities` |
| X24 | `f23`, `f26`, `f27_has_noise` |
| X31 | `b1_covariance_over_a_stream`, `b1_single_step`, `f26`, `f27_has_noise` |
| X41 | `test_f27_navstate_keeps_a_drifted_rotation` |
| W12 | `test_f27_recursion_reads_eps_at_large_steps` |
| P23, P23-old, S21 | `test_f27_bias_jacobians_on_high_rate_edges_against_complex_step` |

Item (a) (the covariance bound 1e-7 → 1e-10 against a complex-step reference) is **fixed** in substance. Every 1e-9-scale block scaling I tried is killed: Y03-Y06 and the replays X31 and U49.

**Three gaps are closed only at the specific values the tests use:**
* **(b)/(f) Per-step angle band.** Every covariance reference test feeds per-step angles `|w dt|` in roughly [4e-4, 4e-3] rad: fixture 1.1e-3; EDGE_A 6e-4; EDGE_B 3.9e-3; the time-varying stream at dt = 4.7e-3 about 2.5e-3; the 14-step stream at dt = 1e-3 about 5e-4. A branch on `|w dt|` outside that band changes B1's or B2's covariance and survives. Survivors:
  * Y24: `J_r` below 1e-4;
  * Y55: `J_r` above 1e-2 (B2);
  * Y56: first-order `J_r` above 1e-2 (B1);
  * Y57: `A_RR = I` below 1e-4, which affects B1 too;
  * Y18: a 1e-8 relative error in `Jr`'s K² term, visible only at per-step angles ≳ 0.25 rad.
* **(c) Tiny densities.** `has_noise` is pinned at 1e-21 only. Any threshold `t` with 1e-21 < `t` ≲ 1.5e-154 still survives (Y20, Y58). A density of 1e-61 gives a covariance of 5.0e-123, and a density of 1e-150 gives 5.0e-301. Below about 1.5e-161, σ² underflows and the clean covariance is itself 0.
* **(f) FD step.** `FD_EPS = 3e-5` is a correct default; see the next part.

**FD_EPS = 3e-5: right default, no claim number depends on 1e-6.** `bias_jacobians_fd` error vs my complex step, relative:

| edge | 1e-6 | 3e-6 | 1e-5 | **3e-5** | 1e-4 | 3e-4 | 1e-3 |
|---|---|---|---|---|---|---|---|
| fixture e0, B1 | 2.4e-8 | 3.1e-9 | 2.4e-10 | **3.1e-10** | 2.6e-10 | 1.9e-9 | 2.1e-8 |
| fixture e1, B2 | 2.0e-8 | 1.1e-8 | 3.2e-9 | **7.3e-10** | 2.0e-10 | 1.9e-9 | 2.1e-8 |
| \|w\| = 5, T = 1, B1 | 1.0e-8 | 1.9e-9 | 6.9e-10 | **2.6e-10** | 8.2e-10 | 7.8e-9 | 8.7e-8 |
| \|w\| = 2.55, T = 1.5, B2 | 1.2e-8 | 5.9e-9 | 1.6e-9 | **9.4e-10** | 2.1e-9 | 1.8e-8 | 2.0e-7 |
| \|w\| = 8, dt = 1e-2, B2 (not in the tests) | 7.1e-9 | 4.0e-9 | 8.6e-10 | **4.5e-10** | 5.3e-10 | 4.8e-9 | 5.4e-8 |

* 3e-5 sits in the flat minimum (≤ 1.1e-9 on every edge I tried), between round-off and truncation. 1e-4 is equally good on slow edges and 2x worse on the 1.5 s edge.
* **No claim records a number that depends on the old 1e-6:**
  * C05 states only the [50, 200] ratio band. I measured R/v/p ratios of 99.92 / 100.00 / 100.00 (e0) and 99.99 / 100.03 / 99.995 (e1) at 3e-5. At 1e-6 they are 99.92 / 100.01 / 100.02 and 100.00 / 100.05 / 100.00. With exact complex-step J they are identical to 3e-5 to 4 digits.
  * C13a is a bitwise identity.
  * C20 and C21 use B3's own steps (`J_FD_EPS`, `C_FD_EPS`), which did not change.
* Two stale artefacts remain in tests, not in claims:
  * `tests/test_baselines.py:1111`'s comment still says "src's FD_EPS (1e-6)".
  * `test_c05b`(c) builds its GTSAM-based `J_R` reference with `eps = B1.FD_EPS`. The reference therefore moves with the mutant (Sec. 6).

**P23-old margin.** P23-old (`FD_EPS = 1e-6`) is killed only by `test_f27_bias_jacobians_on_high_rate_edges_against_complex_step`. Its error there is 1.0e-8 to 1.2e-8 relative against `JAC_CS_REL = 1e-8`, so the kill margin is 1.0x-1.2x in my metric (the test's comment says 1.9x in its own). That margin is round-off-dependent, i.e. fragile. **But P23-old is equivalent under the owner's rule.** Its largest change to any Jacobian over my broad sample is 4.4e-8 relative (fixture B2 reparameterized factor), against C21's 1e-6 / 10 = 1e-7. A robust kill is therefore not required for the gate. The clean code's own margin against `JAC_CS_REL` is ≥ 9x on the test's edges (≤ 1.1e-9 vs 1e-8), which is robust.

## 4. Per-claim verdicts

All numbers are mine.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 14 constants are exactly equal to the retyped literals. R_0 vs `expm`: 1.1e-16 |
| C02 | CONFIRMED | Retyped M1/M2 vs `quad_vec` (epsabs = epsrel = 1e-14): M1 7.6e-17 / 5.6e-17, M2 1.7e-16 / 1.1e-16 (e0/e1); `src` identical. Chained closed form vs the `REF_*` literals: KF1 v 4.4e-16, p 3.5e-17; KF2 R 1.1e-16, v 8.6e-16, p 2.2e-16. Literals vs my 5x5 `expm` chain: ≤ 3.3e-15. Each of the 13 checked literal strings occurs exactly once in `docs/02_baseline_reproduction.md` (count only) |
| C03 | CONFIRMED | GTSAM `predict` from NavState(Exp([0.3, −0.5, 0.2]), [3, 1, −4], [1, −2, 0.5]) vs my formula: R 8.9e-15 / 7.8e-15, v 8.9e-16, p 0 / 2.8e-17 |
| C04 | CONFIRMED | Error ratios over n = 100..1600: e0 v 0.50002, 0.50001, 0.50000, 0.50000 and p 0.50086 → 0.50011; e1 v 0.50003 → 0.50000 and p 0.50088 → 0.50011. `src` vs my `expm` Euler: ≤ 2.2e-14 |
| C05 | CONFIRMED | Ratios as in Sec. 3 (all in [99.9, 100.1]). `src` J vs complex step: 3.1e-10 / 6.4e-10 |
| C06 | CONFIRMED | σ_g = 1.7e-4, σ_a = 1e-2, at every step: asymmetry ≤ 1.7e-20 (iso), minimum eigenvalue −4.0e-23, trace nondecreasing; also anisotropic. Vs my complex step: 1.6e-12 .. 2.1e-12 block-relative (≤ 2.7e-9 as stated). Vs GTSAM after M: RR 2.3e-14 / 3.5e-15; Rv 5.04e-3 / 5.03e-3 and Rp 7.53e-3 / 7.52e-3 (the hint's 5.0e-3 / 7.5e-3 are these two blocks), halving at n = 400, 800 (2.52e-3, 1.26e-3; 3.77e-3, 1.88e-3). Untransformed R-v block 0.170 / 0.195 |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 1.6e-15. (b) 300 φ, \|φ\| = 0.4, ε = 1e-7: right 6.0e-7 (FD-limited); left max 1.37, min 2.2e-2. "Fails by ≥ 1e-1" holds as a sup over draws, as in passes 8-9. `src` J_Cay vs retyped: 0.0 |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 2.7e-15; minimum value 1.00000039 |
| C09 | CONFIRMED | 500 draws: on-manifold 4.4e-16; perturbed min 6.3e-4 ≥ 1e-4 |
| C10 | CONFIRMED | Factor dv̄/dp̄ vs my Cayley Euler: 0.0. Self-consistent states: ≤ 1.6e-13. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.4e-4 / 6.8e-4, all in [1e-4, 1e-2]. \|r\| vs \|Δv̄ − Δv_exact\|: ≤ 8e-16 |
| C11 | CONFIRMED | 2000 draws: 4.4e-16 |
| C12 | CONFIRMED | 2000 draws: isotropic 8.9e-15; W = diag(1, 2, 5) max 46.0 ≥ 1 |
| C13 | CONFIRMED | (a) `J_R` bytes unchanged after `predicted_rotation` and `rotation_residual` at a perturbed bias. (b) At R_j = R_i dR̄ Cay(ξ), \|ξ\| = 1e-2, 200 directions: exact vs truncated min 3.05e-5, max 4.8e-5 ≥ 1e-6 (the pass-9 wording note on the R_j dependence stands) |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7; accessors as stated; `translation()` = `position()`; `Gal3(M)` round trip 0.0 |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. `Expmap(1e-6 e_k)` moves only the stated block. Expmap vs `expm(Hat)`: 7.8e-16 over 50 ξ. Remap (0, 6, 3) |
| C16 | CONFIRMED | KF0-KF2 **0.107628** (KF0-KF1 0.100839, KF1-KF2 0.167801). Remapped `logm` vs `NavState.Logmap`: ≤ 1.8e-15 |
| C17 | CONFIRMED | Accessors, `vector()` and `biasHatVector` as stated. Adapters round-trip exactly; one application moves the vector by 1.67 |
| C18 | CONFIRMED | My 5x5 products on both edges (Dt = 0.5): W-form error ≤ 2.2e-16 at t_i = 0 and 3.7, with t_j = 0.5 / 4.2. F2 form: 0 at t_i = 0 and **18.148500 = \|g\|·0.5·3.7** exactly at t_i = 3.7. Swapped order: 2.4525 = \|g\| Dt² at t_i = 0. The hint's "36.3 vs 9.81·1.0·3.7" uses Dt = 1.0, which is not a fixture edge; the statement's "\|g\| Dt t_i" holds exactly at Dt = 0.5. Mutants: Y42 (rearranged W) is equivalent; Y44, Y45 and the replay X45 are killed by `test_c18` |
| C19 | CONFIRMED | PIM-G vs ZOH: R 1.0e-14, v 2.0e-14, p 3.8e-15 (e0); 8.4e-15 / 1.7e-14 / 3.1e-15 (e1). Tangent dv: 2.171e-3 / 2.760e-3 ≥ 1e-4. Time-varying: right 1.1e-16; left at k = 40 4.7e-5 ≥ 1e-5; GTSAM vs my ZOH chain 4.4e-16 |
| C20 | CONFIRMED | My FD J (eps 1e-7, `logm`, GTSAM `predict` at g = 0): J[0:3,0:3] = J[9,:] = 0.0. Right composition ≤ 1.7e-10 (e0) / 2.4e-10 (e1). Component-wise at 5e-2: 30 random directions 1.06e-2 / 1.35e-2; top singular direction of the leading model difference 1.27e-2 / 1.62e-2 ≥ 1e-2. `src` J vs mine: 3.6e-9 / 3.7e-9 |
| C21 | CONFIRMED | My recursion vs my FD re-integration: 2.6e-8 relative. `src` recursion vs my FD: 7.2e-8. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1-B2: v 6.167e-8, p 4.376e-8, R max-abs **2.9965e-8** (\|\|Log\|\|∞ 2.81e-8, \|\|Log\|\|₂ 3.51e-8); mine identical. Per-edge 2.325744e-8 / 5.517531e-8 vs formula (rel 1.9e-7 / 3.3e-7). dt² ratios 0.25000 ×3, 0.25001 |
| C23 | CONFIRMED | B1-B3: v 1.2867e-3, p 1.2031e-3, R 3.1e-15; B2-B3 R 3.0e-8. B1 v ratios 0.50028, 0.50014, 0.50007, 0.50003, p 0.5000 ×4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.50000 |
| C24 | CONFIRMED | B3 vs exact: KF1 8.9e-16 / 1.6e-14 / 2.1e-15, KF2 1.1e-15 / 4.7e-15 / 5.1e-15. B1 R 9.5e-15 / 2.7e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My graph and my densities (σ_g 1.7e-4, σ_a 1e-2, bias walks 1e-5 / 1e-4, integration 1e-4). Ranks 30/45 (30x45), 39/45 (39x45), 45/45 (45x45). Principal cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**, pose/velocity energy 2.5371 (density-independent, as the nullspace is invariant to per-factor whitening). Graph error 6.7e-23; unwhitened IMU residuals 1.7e-14 / 2.0e-14 |
| **C33** | **REFUTED** | Sec. 2. At \|φ\| = 0.1 the difference is **2.49e-3**, 2.5x the 1e-3 bound. It is second order, −(φφᵀ/6 + \|φ\|²I/12), and every sample violates the bound |

**Counts: 25 CONFIRMED, 1 REFUTED (C33), 0 UNTESTABLE.**

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh `tempfile` copy of `src tests orchestrator experiments docs conftest.py pyproject.toml`. Anchors are asserted unique.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (106 tests). All of them are reference checks; it excludes the change detectors in `test_mutation`.
* Every fast-set survivor that is not equivalent was re-run on the **full suite** (249 tests).
* A comment-only control passed 106/106.
* The worktree's tracked files never changed.

### 5.1 New mutants Y01-Y59

| id | change | result |
|---|---|---|
| Y01 | B1 `retract_right_jacobian` = `Jr(−φ)` (left Jacobian) | killed: c06b, c06c, b1_cov_stream, b1_single_step, f27_has_noise |
| Y02 | `Jr` at half the step | killed: 6 (c06b, c06c, f26, ...) |
| Y03 | A_vR × (1 + 1e-9) | killed: f23, f26, f27_has_noise |
| Y04 | A_pR × (1 + 1e-9) | killed: f27_has_noise |
| Y05 | A_pv × (1 + 1e-9) | killed: f23, f26, f27_has_noise |
| Y06 | Bg × (1 + 1e-9) | killed: c06b, b1 ×2, f26, f27_has_noise |
| Y07 | A_RR = `retract(−w dt)` instead of `dR_stepᵀ` | survives; **equivalent** (5.3) |
| Y08 | symmetrise `cov` after each step | survives; **equivalent** |
| Y09 | `FD_EPS` = 1e-5 | survives; **equivalent** |
| Y10 | `FD_EPS` = 3e-4 | killed: f27_bias_jacobians_high_rate |
| Y11 | J_v × (1 + 3e-7) | killed: 6 (c05c, c13b, c13e, b1_bias_jac, b2_factor, f27) |
| Y12 | J_p × (1 + 3e-7) | killed: 6 |
| Y13 | J_R × (1 + 3e-7) | killed: 7 |
| Y14 | J_R from `Log(R₋ᵀ R₊)/(2ε)` | killed: b1_bias_jacobians_fd |
| Y15 | Eq. (44) rotation applied on the left | killed: c05, b1_bias_corrected_increment |
| Y16 | `increment()` returns views | killed: f22_no_outward_aliasing |
| Y17 | `integrate_edge` drops `**kwargs` | killed: 7 (c22, c23, c24, ...) |
| Y18 | `Jr`'s K² coefficient × (1 + 1e-8) | **survives; NOT equivalent** (5.4) |
| Y19 | `Jr` series branch drops K²/6 | survives; **equivalent** |
| Y20 | `has_noise` thresholds 1e-100 | **survives; NOT equivalent** |
| Y21 | per-axis σ_g permuted [0, 2, 1] | killed: 7 |
| Y22 | B2 raw noise Jacobian `J_Cayᵀ` | killed: f26, f27_has_noise |
| Y23 | B2 reparam noise Jacobian `J_Cay(cayley_vector φ)` | killed: f26 |
| Y24 | B2 raw uses `J_r` when \|φ\| < 1e-4 | **survives; NOT equivalent** |
| Y25 | J_Cay numerator + φφᵀ/4 (= (I + K/2)⁻¹) | killed: c07, b2_cayley_maps, f26, f27_has_noise |
| Y26 | `cay` via `(I + P) inv(I − P)` | survives; **equivalent** |
| Y27 | `cayley_vector` small branch below 1e-6 | killed: f25_series |
| Y28 | Eq. (19) ξ term × (1 + 1e-10) | killed: c09, c11, c13c, b2 ×2 |
| Y29 | Eq. (22b) `(I + Γ/2) R_j` | killed: c11, c13c, c13d, b2 ×2 |
| Y30 | Eq. (21b) A_va guard tests `db_g` | **survives; NOT equivalent** |
| Y31 | Eq. (21c) `v_i Dt` × (1 + 1e-13) | **survives; NOT equivalent** |
| Y32 | `from_edge` Jacobians always raw | killed: b2_factor_from_edge, f27 |
| Y33 | `xi = J b − J b̄` | survives; **equivalent** |
| Y34 | B3 FD one-sided `plus/ε` | killed: b3_fd_reads_eps, f27_recursion |
| Y35 | B3 FD J × (1 + 3e-7) | killed: c20, b3_fd_reads_eps, f27_recursion |
| Y36 | recursion C × (1 + 3e-7) | killed: f27_recursion |
| Y37 | `J_FD_EPS` = 1e-4 | survives; **equivalent** |
| Y38 | `C_FD_EPS` = 1e-4 | survives; **equivalent** |
| Y39 | combined params: bias-ω covariance from σ_ba | killed: c25b, b3 ×2 |
| Y40 | integration covariance σ instead of σ² | killed: c25b, b3 ×2 |
| Y41 | component-wise model swaps p/v tangents | killed: c20, b3_models |
| Y42 | `endpoint_W` algebraically rearranged | survives; **equivalent** |
| Y43 | `gal3_from_navstate` ignores t | killed: b3_gal3_adapters |
| Y44 | `gal3_lift` ignores t | killed: c18, b3_gal3_adapters |
| Y45 | `time_shift` uses \|dt\| | killed: c18, b3_endpoint_relations |
| Y46 | M1 K coefficient × (1 + 1e-13) | killed: f27_M1_M2 |
| Y47 | M2 K coefficient × (1 + 1e-13) | killed: f27_M1_M2 |
| Y48 | M2 K² coefficient × (1 + 1e-13) | killed: f27_M1_M2 |
| Y49 | `so3_log` generic × (1 + 3e-12) | killed: c05, c22b, b1_increment_error, f25, traj_so3 |
| Y50 | `so3_log` small branch below 1e-6 | killed: f25_so3_log_small_angles |
| Y51 | `so3_exp` series drops K²/2 | killed: f25_series |
| Y52 | `dt_total += dt (1 + 1e-15)` | survives; **equivalent** |
| Y53 | Ba velocity uses `dR_ik dR_step` | killed: 6 |
| Y54 | `gal3_distance` uses Log(B⁻¹A) | survives; **equivalent** |
| Y55 | B2 raw uses `J_r` when \|φ\| > 1e-2 | **survives; NOT equivalent** |
| Y56 | B1 noise Jacobian `I − K/2` when \|φ\| > 1e-2 | **survives; NOT equivalent** |
| Y57 | A_RR = I when \|w dt\| < 1e-4 | **survives; NOT equivalent** |
| Y58 | accel `has_noise` threshold 1e-60 | **survives; NOT equivalent** |
| Y59 | Eq. (21c) A_pa guard tests `db_g` | **survives; NOT equivalent** |

**Count (59):** 38 killed by reference tests, 11 equivalent, **10 not equivalent and surviving**.

### 5.2 Replay of 41 earlier mutants as real file edits

I re-typed each edit from its description; I did not copy the `PASS9_MUTANTS` table.

**Killed by reference tests (31):**
* the pass-9 survivors X15, X21, X22, X24, X31, X41, W12, P23 and S21 (killing tests in Sec. 3);
* F26 and P23-old;
* X03, X05 (f24); X08 (f25); X12 (f25_series);
* X23, X25, X26 (6 tests each);
* X27 (8); X28 (7); X29 (b1_increment_error); X32 (4); X36 (11);
* X42 (6); X44 and X50 (b3_models); X45 (2); X46 (2);
* X53 (c05, traj_so3); X56 (4); U49 (f27_has_noise).

U49 (Ba position × (1 + 1e-9)) was equivalent in pass 9. It is now killed, which is fine.

**Surviving, all equivalent (10):** R36, S31, X30, X49, W11, P46, X19, X20, X01, X02 (Sec. 5.3).

### 5.3 Equivalence rulings (owner's rule; threshold = max(τ/10, 8 ulp))

**How I measured E.** Each surviving mutant ran a probe over a broad input sample (354 outputs):
* B1, B2 and B2-reparameterized covariances and increments, on the fixture edges and on time-varying streams with per-step angles from about 1e-5 to 0.5 rad, at six density settings (isotropic, anisotropic, gyro-only, accel-only, per-axis single, 1e-101);
* bias Jacobians and factor Jacobians on the fixture edges and two high-rate edges;
* B3 J, and the recursion at dt = 2.5e-3, 0.05 and 0.3;
* the residual functions on 50 random inputs with \|v\| ≈ 30 m/s and Dt up to 2 s;
* `Jr`, `Exp`, `Log` at angles from 1e-9 to 3, and M1/M2 at \|w\| ≤ 5, T ≤ 3;
* `endpoint_W`, `gal3_distance`, and B3 `predict`.

E is the largest change over that sample.

| mutant | quantity Q | τ (claim) | threshold | measured E | ruling |
|---|---|---|---|---|---|
| Y07 | B1/B2 covariance | C06 2.7e-9 rel | 2.7e-10 rel | B1 0.0 (Exp(−x) = Exp(x)ᵀ bitwise); B2 ≤ 1.7e-14 rel | equivalent |
| Y08 | covariance | C06 | 2.7e-10 | ≤ 1.5e-15 rel | equivalent |
| Y09 | B1/B2/factor J | C21 1e-6 rel | 1e-7 | 3.0e-9 rel (fixture), 1.8e-9 (high-rate) | equivalent |
| Y19 | `Jr` (unit scale) → covariance | C06; floor 8 ulp ≈ 2e-15 | 2e-15 abs | `Jr` 3.0e-19 abs; covariance 0.0 | equivalent |
| Y26 | `cay`, B2 J, B2 covariance | C07 1e-13; C21 | 1e-14; 1e-7 | `cay` and states at rounding; J 7.3e-10 rel (FD amplifies rounding) | equivalent |
| Y33 | ξ → r^R, predicted R | C11 1e-13 abs | 1e-14 | ξ 1.1e-18 abs; residuals at rounding | equivalent |
| Y37 | B3 J (C20 model, C21) | C21 1e-6 rel; C20 1e-8 abs at 5e-2 | 1e-7; 1e-9 | J 5.35e-9 rel; max_i \|δJ_i\|₂·5e-2 = **5.2e-10** | equivalent (1.9x) |
| Y38 | recursion J | C21 | 1e-7 | 3.9e-10 rel | equivalent |
| Y42 | `endpoint_W` form | C18 1e-12 | 1e-13 | 9.1e-13 abs on entries of scale 4.9e3 (t_i = 1e3), 1.9e-16 rel ≈ 1 ulp | equivalent (rounding floor) |
| Y52 | Dt → predicted states | C03 1e-12 | 1e-13 | 2.7e-14 abs (B3 predict from n = 1600 B1 states); increments 2e-16 rel | equivalent |
| Y54 | `gal3_distance` | C20 1e-8 | 1e-9 | 4.1e-14 abs | equivalent |
| R36 | factor J_R | C21 | 1e-7 | 5.0e-9 rel (ξ), 2.1e-9 rel over all factor J | equivalent |
| S31 | J (`FD_EPS` 1e-4) | C21 | 1e-7 | 2.6e-9 rel | equivalent |
| X30 | J (`FD_EPS` 3e-6) | C21 | 1e-7 | 1.1e-8 rel | equivalent |
| P23-old | J (`FD_EPS` 1e-6) | C21 | 1e-7 | 4.4e-8 rel | equivalent (2.3x), killed anyway |
| X49 | B3 J | C21; C20 | 1e-7; 1e-9 | 6.0e-9 rel; C20 effect 5.4e-10 | equivalent (1.8x) |
| W11 | B3 J | C21; C20 | 1e-7; 1e-9 | 5.4e-9 rel; C20 effect 5.2e-10 | equivalent |
| P46 | recursion J | C21 | 1e-7 | 4.0e-10 rel | equivalent |
| X19 | covariance | C06 | 2.7e-10 | 0.0 on all 354 outputs (differs only at dt = 0, outside the domain, as pass 9) | equivalent |
| X20 | covariance | C06 | 2.7e-10 | 0.0 (differs only for non-PSD `cov`) | equivalent |
| X01 | `so3_log` | C22 → 2.3e-11 | 2.3e-12 | 1.6e-14 abs | equivalent |
| X02 | `so3_log` | C22 | 2.3e-12 | 2.2e-16 abs | equivalent |

### 5.4 Complete survivor list (non-equivalent)

**Full-suite status.** All 10 were re-run on the full suite (249 tests); results are in Sec. 5.5. Every one is low severity. None changes any claim's recorded number on the claim's own inputs.

| mutant | change | distinguishing input and effect | threshold |
|---|---|---|---|
| **Y18** | `Jr` K² coefficient × (1 + 1e-8) | `Jr` itself moves by 1.19e-8 at \|φ\| = 3. **B1 and B2-reparameterized covariance 1.42e-9 rel** at per-step angle ≈ 0.5 rad (dt = 0.05, rate ×20); 6.3e-11 at 0.1 rad/step. `test_b1_so3_right_jacobian`'s 1e-8 absolute FD bound at \|φ\| ≈ 1 sees only 1.6e-9 | C06 2.7e-10 rel |
| **Y20** | `has_noise`: densities ≤ 1e-100 ignored | σ_g = σ_a = 1e-101: covariance 0 instead of 5.0e-203 (**100%**), for B1/B2/reparameterized | 2.7e-10 rel (8 ulp of 5e-203 ≈ 1e-217) |
| **Y58** | accel threshold 1e-60 | σ_g = 0, σ_a = 1e-61: covariance 0 instead of 5.0e-123 (**100%**) | same |
| **Y24** | B2 raw `J_r` when \|φ\| < 1e-4 | dt = 1e-4 (per-step angle ≈ 5e-5): B2 covariance **2.38e-9 rel**; dt = 2e-5: 9.5e-11 | 2.7e-10 rel |
| **Y55** | B2 raw `J_r` when \|φ\| > 1e-2 | dt = 0.05, rate ×20: **1.7e-1 rel**; dt = 0.02, rate ×10: 9.7e-3 | 2.7e-10 |
| **Y56** | B1 noise Jacobian `I − K/2` when \|φ\| > 1e-2 | same inputs: **1.6e-1 / 6.4e-3 rel** | 2.7e-10 |
| **Y57** | A_RR = I when \|w dt\| < 1e-4 (B1 and B2) | dt = 1e-4: **6.9e-4 rel** (gyro-only); dt = 2e-5: 1.4e-4 | 2.7e-10 |
| **Y30** | Eq. (21b): the A_va term is applied iff `db_g is not None` | `velocity_residual_eq21b(..., A_vg=None, A_va=A, db_g=None, db_a=db)`: drops A_va·db_a, **0.45 abs** on my sample (0.052 on a unit example). Every call in `src` passes both, so the factor is unaffected | C10 1e-12 → 1e-13 |
| **Y59** | Eq. (21c): same guard for A_pa | same pattern: **0.31 abs** | 1e-13 |
| **Y31** | Eq. (21c) `v_i Dt` × (1 + 1e-13) | random states with \|v\| ≈ 30 m/s, Dt ≤ 2 s: **1.27e-11 abs** on r^p of scale ≤ 1.7e2 (8 ulp ≈ 2.3e-13). On the fixture (\|v\| ≤ 1.2, Dt = 0.5): ≤ 6e-14, below threshold | C10 1e-12 → 1e-13; floor 2.3e-13 |

**What would kill them** (advisory; not part of this commit):
* **Y18, Y24, Y55, Y56, Y57:** run the existing complex-step covariance references at per-step angles of about 1e-5 and 0.1-0.5 rad, e.g. dt = 2e-5 and dt = 0.05 at 10 rad/s. For Y18, check `Jr` against an mpmath or complex-step reference at \|φ\| up to 3, at 1e-12.
* **Y20, Y58:** add a density near the underflow limit (σ = 1e-150, covariance about 5e-301).
* **Y30, Y59:** call Eq. (21b)/(21c) with only the accel pair supplied.
* **Y31:** test Eq. (21c) at large \|v\| Dt (tens of metres) at about 1e-13 relative.

### 5.5 Full-suite runs of the survivors

| mutant | full suite (249) | notes |
|---|---|---|
| Y18, Y24, Y30, Y31, Y55, Y56, Y59 | **249 passed** | not even a change detector fires |
| Y20, Y57, Y58 | 238 passed, **11 errors**, 0 failures | see below |

**The 11 errors are change detectors, not kills.** All of them are in `test_mutation.py::test_pass9_mutant_is_killed_as_a_file_edit[*]`. The cause is the module fixture `pass9_runs`: its `_run_in_copy` asserts that each PASS9 edit anchor occurs exactly once.
* Y20 and Y58 rewrite the `has_noise` lines, which are X21's and X22's anchors.
* Y57 rewrites `A[0:3, 0:3] = dR_step.T`, which is X24's anchor.

The assertion fails, the fixture errors, and all 11 parametrised cases error with it. This is an edit-anchor change detector, so under the owner's rule it is not a kill. No reference test failed in any of the 10 runs.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over from pass 9, unchanged:** items 1-11 (structural and labelled: the c10a tautology, the c13 `J_R.tobytes()` identity, M04, the c12 isotropic half, c08 `≥ 1`, the c25 ranks 30/39, c06 structure, the c20 lower bound, c01, the c22 upper bound), item 14 (reference-only preconditions) and items 17-18 (`test_mutation`'s literal-list checks and the change detectors). None was needed for any kill in this pass.

**New or changed in this pass:**
* **`test_c05b`(c)'s reference reads `src`'s step.** Its GTSAM-based `J_R` reference uses `eps = B1.FD_EPS`, i.e. `src`'s constant. A mutant of `FD_EPS` moves the reference with it, so this assertion cannot see an FD-step mutant (P23, Y10 and P23-old are caught elsewhere). The reference is otherwise independent (GTSAM `predict` plus a retyped log). Recommendation: a literal eps.
* **Stale comment.** `tests/test_baselines.py:1111` says "src's FD_EPS (1e-6)"; the value is now 3e-5. Documentation only.
* **`test_f27_bias_jacobians_on_high_rate_edges_against_complex_step`'s kill of P23-old is round-off-margined.** The error is 1.0e-8 to 1.2e-8 against `JAC_CS_REL = 1e-8`. It is not a can't-fail test, but the P23-old kill should not be relied on. It is not needed: P23-old is equivalent (4.4e-8 < 1e-7).
* **New reference-only preconditions (item-14 class):**
  * `assert orth >= 1e-11` in `test_f27_navstate_keeps_a_drifted_rotation`;
  * `assert visible >= 1e-7` in `test_f27_recursion_reads_eps_at_large_steps`;
  * `if not pim.has_noise` in `test_f27_has_noise` (this one can fail and does, on X21).

  The first two are deterministic and cannot fail; they guard the reference inputs only.
* **Coverage narrowness rather than a can't-fail test.** Every covariance reference (c06b, c06c, `b1_covariance_over_a_stream`, `b1_single_step`, f23, f26, f27_has_noise) uses per-step angles in [4e-4, 4e-3] rad. This is the cause of Y18, Y24, Y55, Y56 and Y57.

No new test compares `src` with a copy of `src`.

## 7. Final state

This commit contains this report only. The worktree is clean:
* the scratch directory (`results/p10`, gitignored) was deleted;
* every mutant copy was a `tempfile` directory and was removed;
* no file outside the worktree was written;
* `src/`, `tests/` and `docs/claims/` are untouched.

Full suite on the clean tree at the end (`eb5011c` plus this report): **249 passed** in 248.7 s, under load.

**Gate: NOT passed.** There are 10 non-equivalent survivors, all low: Y18, Y20, Y24, Y30, Y31, Y55, Y56, Y57, Y58 and Y59. M2-C33 is REFUTED. M2-F26 is FIXED; M2-F27 is PARTIALLY FIXED.
