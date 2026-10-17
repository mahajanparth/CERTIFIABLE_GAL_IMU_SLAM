# Stage-B verification, pass 13: M2 baselines (`stage-b/m2-baselines` @ `803f6e9`)

Adversarial verification of milestone M2. Baseline 1 is Forster preintegration. Baseline 2 is the from-paper Rai Cayley factor. Baseline 3 is GTSAM 4.3.0's Galilean preintegration. Agent: verify-M2-013.

**Scope of this pass:**
* a ruling on M2-F30 (the pass-12 survivors A07 and A11), answered by `803f6e9`, which changes tests only;
* verdicts on M2-C01..C25, plus M2-C36, which this pass's brief supplies and the yaml does not contain;
* A07 and A11 replayed as real exactly-once file edits, retyped by me (R07, R11, R11b);
* three probes (P01-P03) aimed at the new F30 test's scaled bound;
* 38 new mutants (B01-B83, not contiguous), each a real file edit;
* a replay of `tests/test_mutation.py`.

**What I did not read.** No derivation narratives, no `seeds*.yaml`, nothing named `*SEEDS*`. I did not browse the git history of `docs/claims`. The only history I diffed is `d6f52f8..803f6e9`.

**What I did not touch.** No file outside this worktree. No edits to `src/`, `tests/` or `docs/claims/`.

## 0. Summary

| item | result |
|---|---|
| Clean tree at start (`803f6e9`) | full suite **279 passed** in 419 s (load average 28-38; other jobs were running) |
| `src/` and `docs/claims/` change in `d6f52f8..803f6e9` | **none.** `git diff --stat d6f52f8 803f6e9 -- src docs/claims` is empty. Only `tests/test_input_dependence.py` and `tests/test_mutation.py` changed |
| **M2-F30** | **A07 and A11 are FIXED**: both are killed as real file edits by reference tests, on the intended assertion lines. **But the new test's part (b) bound, 1e-14 (1 + \|ξ\|), is too loose at the chart edge.** It allows 2.9e-13 absolute at \|ξ\| = 28.2, which is 2.9x C11's own tolerance. Three probes sit in that gap (new finding **M2-F31**) |
| Claims | **25 CONFIRMED** (C01-C25), **1 REFUTED** (C36), 0 UNTESTABLE |
| C36 | **REFUTED.** Σ scales by **k², not k**. At k = 2 the trace ratio is 4.000000000000. Deviation from k-linearity: 1.0 relative (k = 2) and 9.0 relative (k = 10). Deviation from k²: ≤ 4.3e-15 |
| Mutants (44 file edits: 3 replays + 3 probes + 38 new) | 3 replays killed; 30 new mutants killed by reference tests; 8 new mutants equivalent (Sec. 5.3); **3 non-equivalent survivors (P01, P02, P03)** |
| `tests/test_mutation.py` replay | 164/164 pass on the clean tree (inside the 279). That includes `test_pass12_mutant_is_killed_as_a_file_edit` [A07, A11] and `test_pass12_killers_are_reference_checks` |
| **Gate** | **NOT passed.** 3 non-equivalent survivors, all in `src`'s Eq. (19) at large \|ξ\|. Each breaks the C11 identity by 1.8e-13 to 2.0e-13, beyond the claim's own 1e-13, and none is seen by any test (M2-F31) |

## 1. Independent references (no `src` imports)

My reference code lived in `results/p13` (gitignored) and was deleted before commit. It never imports `src` or `tests`. `src` was called only as the system under test.

* **Rotations:**
  * `Exp` = scipy `expm(hat)`;
  * `Log` = scipy `Rotation`;
  * random rotations from `Rotation.random`. scipy `expm` of a hat at angles up to 5.2 rad is orthogonal only to 1.6e-13. That would have shown up as a false 1.3e-12 in C12, so I did not use it as an exact rotation there.
* **Exact increments:** a 5x5 block `expm` (ZOH), both per edge and per step.
* **Euler sums:** retyped Eq. (9a)-(9c), with Exp and with a retyped Cayley.
* **Covariance (C06, C36):** Each step's (A, B) is a central-difference linearisation (h = 2e-5) of my own one-step Euler map, in the chart `R Exp(d)`, `v + d`, `p + d`. The steps are combined by `C ← A C Aᵀ + B Q Bᵀ` with `Q = diag(σ²)/dt`. It agrees with `src` to **1.0e-9** relative on fixture edge 0.
* **Complex-step bias Jacobians (C05):** my own complex Euler with complex `expm`, h = 1e-30.
* **Gal(3):**
  * my own 5x5 matrices, inverse, `logm` chart and Ad (from the contract's formula);
  * my own FD Jacobians and recursion (C20, C21);
  * my own 5x5 endpoint products (C18);
  * my own factor graph (C25).
* **Eq. (16) cleared, in 40-digit mpmath** (for F30): `R_j(I − ξ̂/2) − R_i dR̄ (I + ξ̂/2)`.

**Re-implemented core routines:**
* the Eq. (63) covariance;
* the Eq. (9) sums;
* Eq. (19) / cleared Eq. (16) in mpmath;
* the Gal(3) bias Jacobian, Ad and the C21 recursion;
* the C18 endpoint products;
* the C25 graph.

## 2. M2-C36 (brief-only claim): REFUTED

**Claim:** scaling σ_g and σ_a by a common k scales every entry of Σ_ij by k (to 1e-12), "because the densities enter B as standard deviations".

**Why it is false:** the densities enter through `Q = diag(σ²)/dt`, as variances. `src`'s `noise_covariances` does this, and so does my independent reference. The propagation `Σ ← AΣAᵀ + B Q Bᵀ` from Σ₀ = 0 is linear in Q, so Σ scales by **k²**.

**Measured.** Fixture edge 0, σ_g = 1.7e-4, σ_a = 1e-2, 200 steps.

| k | max\|Σ(k) − kΣ(1)\| / max\|kΣ(1)\| | max\|Σ(k) − k²Σ(1)\| / max\|k²Σ(1)\| | tr Σ(k) / tr Σ(1) |
|---|---|---|---|
| 2 | **1.0** | 0.0 | 4.000000000000 |
| 10 | **9.0** | 2.8e-15 | 100.000000000000 |
| 0.5 | **0.5** | 0.0 | 0.250000000000 |
| 3.7 | **2.7** | 4.3e-15 | 13.690000000000 |

My independent reference gives the same trace ratio, 4.0 at k = 2. The claim's 1e-12 is missed by 12 orders of magnitude.

## 3. Ruling on M2-F30

`803f6e9` changes tests only. The `src` and `docs/claims` diffs are empty.

### 3.1 The F29 Eq. (22b) bound tightened to 1e-14

* **The headroom holds.** I re-measured the clean value over 10 seeds of the test's distribution (2000 draws each). The worst is **8.9e-16** (one seed 5.8e-16), so the bound has 11x headroom.
* **My retype R07** is `(1.0 + 5e-14) * Gamma / 2.0`, a different text from the implementer's `Gamma / 2.0 * (1.0 + 5e-14)`. It is **killed** on the intended line: `rotation_residual_eq22b at large Gamma vs retyped Eq. (19): 6.062e-14 > 1e-14`.

### 3.2 The new test `test_f30_eq19_matches_a_retyped_cleared_eq16_at_large_xi`

**Independence: genuine.**
* `_cleared_eq16` is `R_j (I − ihat(ξ)/2) − (R_i dR̄)(I + ihat(ξ)/2)`, built with the test module's own retyped `ihat`. It calls nothing from `src` (no `cay`, no `hat`, no `rotation_from_eq16`).
* It is algebraically Eq. (19), but arranged as two products against the Cayley factors rather than a difference minus a half-sum, so it is not a copy of `src`'s expression.
* `R_i`, `R_j` and `dR̄` come from the retyped `iexp`. `R_j` is drawn freely, not on-manifold, so the test checks the polynomial identity everywhere rather than only at its zero set.
* `src`'s Eq. (19) against my 40-digit mpmath cleared Eq. (16), over 600 chart-regime draws: **2.2e-16 x (1 + \|ξ\|)**.

**Part (a), flat 1e-14 at J_R ~ N(0, 1):**
* clean ≤ **8.9e-16** over 10 seeds, so 11x headroom;
* R11 (my retype, `hat(xi * (1.0 + 6e-14))`) and R11b (the implementer's `0.5 + 3e-14`) are both **killed** on the intended line, at **1.861e-13** > 1e-14.

**Part (b), scaled bound 1e-14 (1 + \|ξ\|) up to \|ξ\| = 28.2.** This is where the bound is too loose.

Clean, per (1 + \|ξ\|), is **≤ 3.84e-16** over 10 seeds, so the bound has 26x headroom. As an absolute number, though, the bound grows to **2.9e-13 at the chart edge**:
* that is **2.9x C11's own tolerance** (1e-13 absolute);
* it is 2.9x C09's τ/10 (1e-13);
* the clean absolute error there is only **3.55e-15** (3000 draws). So a flat 1e-14 would still leave 2.8x headroom, and 2e-15 (1 + \|ξ\|) would leave 5.2x.

A coefficient slip that is uniform in ξ is caught by part (a), because \|ξ\| there reaches about 6. Part (b) adds detection only for defects that are **concentrated at large \|ξ\|**, and those are exactly the defects its bound lets through. Three probes show it (Sec. 5.4):

| probe | edit to `rotation_residual_eq19` | part (b) metric | C11 identity (`src` 22b vs mutant 19), \|ξ\| ≤ 28.2 | C09 on-manifold |
|---|---|---|---|---|
| P01 | ½ → ½ + 4e-15 when \|ξ\|² > 100 | 7.2e-15 (< 1e-14) | **1.81e-13** | 2.2e-14 |
| P02 | + 2e-13 (scalar) when \|ξ\|² > 400 | 9.6e-15 (< 1e-14) | **2.04e-13** | **2.21e-13** |
| P03 | ξ x (1 + 8e-15) when \|ξ\|² > 25 | 7.2e-15 (< 1e-14) | **1.81e-13** | 2.2e-14 |
| clean | — | 3.5e-16 | 3.55e-15 | 2.1e-14 |

* All three pass the full suite as real file edits (Sec. 5.5).
* Each breaks the C11 identity by more than the claim's own 1e-13 inside the C07 chart, which is the domain the F30 test itself declares.
* P02 also breaks C09's on-manifold residual by 2.2e-13, against τ/10 = 1e-13.

**Is the "killers are reference checks" meta-test backed by real reference checks?** Yes.
* `test_pass12_killers_are_reference_checks` still checks location only, and its docstring now says so (the pass-12 advisory is addressed).
* I read both killers. The F29 test compares `src` Eq. (22b) with a retyped Eq. (19). The F30 test compares `src` Eq. (19) with the retyped cleared Eq. (16) (above). Neither compares `src` with a copy of `src`.
* `test_pass12_mutant_is_killed_as_a_file_edit` also requires each kill on the intended `E ` line, and I confirmed both kill lines independently (3.1, 3.2).

**Ruling.**
* **A07 and A11 are killed. M2-F30 is closed as stated.**
* The remediation leaves a gap in part (b): the bound exceeds the claim tolerance above \|ξ\| ≈ 9. I record that as **M2-F31**.
* **Suggested fix** (advisory; not part of this commit): bound part (b) at **2e-15 (1 + \|ξ\|)** or flat **1e-14** absolute. Either kills P01-P03 (7.2e-15 / 9.6e-15 / 7.2e-15 per (1 + \|ξ\|); 1.7e-13 to 2.0e-13 absolute). Alternatively, add a C11-identity check (`src` 22b vs `src` 19) on the chart-regime draws at 1e-14.

## 4. Per-claim verdicts

All numbers are mine.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations: 19200 B each, byte-identical to each other and to my retyped `tile`s. All 11 constants equal the retyped literals exactly. R_0 vs `expm`: 1.1e-16 |
| C02 | CONFIRMED | Retyped M1/M2 vs `quad_vec` (1e-14): 7.6e-17 / 1.7e-16 (e0) and 5.6e-17 / 1.1e-16 (e1); `src` M1/M2 equal mine exactly. My 5x5-`expm` chain vs the literals: KF1 v/p 1.6e-15 / 2.0e-15, KF2 R/v/p 2.2e-16 / 2.6e-15 / 3.3e-15. `src` `exact_states` vs mine: 3.3e-15 |
| C03 | CONFIRMED | GTSAM PIM-G `predict` from NavState(Exp([0.3, −0.5, 0.2]), p, v) vs the formula: R 8.8e-15, v 8.9e-16, p 1.1e-16 (e0); 7.8e-15 / 1.1e-16 / 0 (e1) |
| C04 | CONFIRMED | n = 100..1600. e0: v 0.50002 → 0.50000, p 0.50086 → 0.50011. e1: v 0.50003 → 0.50000, p 0.50088 → 0.50011. `src` vs my Euler ≤ 1.0e-13 |
| C05 | CONFIRMED | `src` FD J vs my complex step: 3.3e-10 / 6.7e-10 relative. Error ratio between \|db\| = 1e-3 and 1e-2 over 20 random directions per edge: [99.92, 100.06] |
| C06 | CONFIRMED | Isotropic: asymmetry 8.5e-21, min eigenvalue −4.0e-23, trace nondecreasing. Anisotropic densities: 5.4e-20 / −5.4e-23, nondecreasing. Vs my FD-linearised reference: 1.0e-9 relative |
| C07 | CONFIRMED | (a) 2000 axes, θ ∈ (0, 3): 2.8e-15. (b) 300 φ with \|φ\| = 0.4, ε = 1e-7, unit d: right 5.0e-8; left **max 0.382**. See the note below this table |
| C08 | CONFIRMED | 5000 ξ, \|ξ\| ≤ 5: 2.7e-15; closed form ≥ 1 (min 1.0000018) |
| C09 | CONFIRMED | 500 draws: on-manifold 3.3e-16; perturbed minimum 6.2e-4. **Spurious-point hunt:** the smallest singular value of (I − ξ̂/2) over 10,000 ξ with \|ξ\| ≤ 28.2 is 1.000000. So r^R = 0 is linear in R_j with an invertible operator, and has the unique solution `R_i dR̄ Cay(ξ)`. Chart-edge on-manifold residual over 10,000 draws: 4.4e-14 |
| C10 | CONFIRMED | Factor dv̄ vs my Cayley Euler: 0.0. Self-consistent: 2.7e-15. Ground truth: v **2.1709e-3 / 2.7603e-3**, p 5.377e-4 / 6.811e-4; it matches \|dv̄ − dv_exact\| to 2.7e-15 |
| C11 | CONFIRMED | 20,000 draws with J_R ~ N(0, 1), \|b\| ≤ 0.5: 8.9e-16. At \|ξ\| up to 28.2: 3.55e-15. See M2-F31 for the test resolution |
| C12 | CONFIRMED | 10,000 draws, exact rotations, r ~ N(0, 1): 1.3e-15 relative to max(1, \|r\|²) (\|r\|² up to 24). W = diag(1, 2, 5): max 275 ≥ 1. τ(diag(0.5, 0.6, 0.7)) = 3/1.8 exactly |
| C13 | CONFIRMED | `J_R` bytes unchanged after `predicted_rotation` / `rotation_residual` at a perturbed bias. 200 directions at \|ξ\| = 1e-2: minimum gap 1.19e-3 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7; accessors as stated; `Gal3(M)` round trip exact |
| C15 | CONFIRMED | `Hat(e_k)` equals my 5x5 hat exactly for all 10 k. Expmap vs `expm(Hat)`: 2.2e-15 over 50 ξ. Remap (0, 6, 3) |
| C16 | CONFIRMED | My remapped `logm` vs `localCoordinates`: **0.107628**. Vs `NavState.Logmap`: 1.8e-15 |
| C17 | CONFIRMED | `ConstantBias(a, g)` accessors and `vector()` = [a; g]; `biasHatVector` = [b_a; b_g]; the adapters round-trip exactly, and a single swap is not the identity |
| C18 | CONFIRMED | My 5x5 products: W form ≤ 1.3e-15 (Dt = 0.5 per edge, Dt = 1.0 span; t_i ∈ {0, 3.7}), with t_j = t_i + Dt. F2 form: 0 at t_i = 0; **18.148500** (Dt = 0.5) and **36.297000** (Dt = 1.0) at t_i = 3.7, = \|g\| Dt t_i exactly. Swapped order at t_i = 0: 2.4525 / 9.81. `src` `endpoint_W_form` vs my product: ≤ 8.9e-16 |
| C19 | CONFIRMED | PIM-G vs ZOH: R/v/p 1.0e-14 / 2.0e-14 / 3.8e-15 (e0) and 8.4e-15 / 1.7e-14 / 3.1e-15 (e1). Tangent dv 2.171e-3 / 2.760e-3. Retyped time-varying stream equals `src`'s exactly. Right composition: 3.3e-16 over 40 steps; left: 6.17e-4 at step 40 |
| C20 | CONFIRMED | My FD J (ε = 1e-7, my `logm` chart, g = 0): J[0:3,0:3] = J[9,:] = 0.0. Right composition: ≤ 3.8e-10 at \|db\| ≤ 5e-2. Component-wise at 5e-2, 30 directions: 8.7e-4 .. **1.18e-2** (e0) and 1.2e-3 .. **1.42e-2** (e1). That is ≥ 1e-2 as a sup, which is how the hint reads it. `src` J vs mine: 4.4e-9 |
| C21 | CONFIRMED | My recursion with my Ad vs my FD re-integration: 3.0e-8 relative at step 40. `src` recursion vs my FD: 1.3e-8 / 2.5e-9 / **2.9e-9** relative at steps 10 / 20 / 40. \|J₄₀\| = 0.100 |
| C22 | CONFIRMED | KF2 B1-B2: R **2.9965e-8**, v 6.1672e-8, p 4.3763e-8. Per edge 2.325744e-8 / 5.517531e-8 vs the formula 2.325744e-8 / 5.517533e-8. dt² ratios 0.25000 x3, 0.24999. `src` B1/B2 vs my own chains: 1.9e-14 / 7.1e-15 |
| C23 | CONFIRMED | B1-B3: R 3.1e-15, v 1.2867e-3, p 1.2031e-3; B2-B3 R 3.0e-8. Ratios: B1 v 0.50028 → 0.50003, p 0.5000 x4; B2 v 0.50026 → 0.50003, p 0.49998 → 0.50000 |
| C24 | CONFIRMED | B3 vs exact: KF1 8.9e-16 / 1.6e-14 / 2.1e-15; KF2 1.1e-15 / 4.7e-15 / 5.1e-15. B1 R 9.5e-15 / 2.7e-15. B2 R 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My own graph, two density sets (3e-4 / 2e-2 / 3e-5 / 5e-4 and 1e-3 / 1e-1 / 3e-4 / 2e-3): ranks 30 (30x45), 39 (39x45), 45 (45x45). Principal cosines **[0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]**; pose/velocity energy 2.5371, identical for both sets. Graph error at truth ≤ 2.1e-23 |
| **C36** | **REFUTED** | Sec. 2. Σ scales as **k²**: 1.0 relative off k-linear at k = 2, and ≤ 4.3e-15 off k² |

**Counts: 25 CONFIRMED, 1 REFUTED (C36), 0 UNTESTABLE.**

**Note on C07(b), advisory.** The left convention's error is not bounded below over directions. For d ∥ φ the two conventions coincide: I measured 4.0e-8 at d = φ/\|φ\|. On 300 unit directions the minimum is 0.025, and 96% are ≥ 0.1. The claim holds as a sup (0.382; the hint quotes a single maximum, 2.1e0). Pass 12's "≥ 1e-1 even as a minimum on this sample" does not generalise. The statement should say "for generic d" or "in sup". This is the same pattern as C20's documented sup.

## 5. Mutation sweep

**Harness.**
* Each mutant is a real string edit of a `src` file, in a fresh copy of `src tests orchestrator docs conftest.py pyproject.toml`. Every anchor is asserted to occur exactly once.
* The **fast set** is `test_baselines`, `test_input_dependence` and `test_orchestrator` (115 tests, including the slow stationary test). An unmutated control copy passed 115/115.
* **Every** fast-set survivor was then run on the **full suite** (279 tests).
* The worktree's tracked files never changed.

### 5.1 Mutants

| id | change | result |
|---|---|---|
| R07 | Eq. (22b) `Rbreve`: `(1 + 5e-14) Γ/2` | killed: f29 Eq. (22b), 6.062e-14 > 1e-14 |
| R11 | Eq. (19) `hat(ξ (1 + 6e-14))` | killed: f30 (a), 1.861e-13 > 1e-14 |
| R11b | Eq. (19) ½ → ½ + 3e-14 | killed: f30 (a), 1.861e-13 > 1e-14 |
| **P01** | Eq. (19) ½ → ½ + 4e-15 when \|ξ\|² > 100 | **survives; NOT equivalent** |
| **P02** | Eq. (19) + 2e-13 when \|ξ\|² > 400 | **survives; NOT equivalent** |
| **P03** | Eq. (19) ξ x (1 + 8e-15) when \|ξ\|² > 25 | **survives; NOT equivalent** |
| B01 | Qa x (1 + 1e-6) | killed: c06b, c06c, … |
| B02 | Qg x (1 + 1e-7) | killed: c06b, b1_noise_covariances, … |
| B03 | A_pR x (1 + 1e-6) | killed: b1_covariance_over_a_stream, f26, … |
| B04 | Ba position block x (1 + 1e-5) | killed: b1_covariance_over_a_stream, f26, … |
| B05 | Bg = `Jrᵀ dt` | killed: c06b, c06c, … |
| B06 | covariance gate `has_noise` only | killed: f23 |
| B07 | `has_noise`: `σ_a ≥ 0` | survives; **equivalent** (proof) |
| B09 | `dt_total += float32(dt)` | killed: c10, c13e, c05c, … |
| B11 | Eq. (44) `J_pa db_a` x (1 + 1e-10) | killed: b1_bias_corrected_increment |
| B15 | `so3_log` generic branch x (1 + 1e-12) | killed: c05, c22b, traj_so3_helpers |
| B17 | near-π `so3_log` axis column fixed to j = 0 | killed: f24 |
| B24 | stream `0.7 * (k * dt)` | survives; **equivalent** |
| B26 | `Edge.duration` returns `T_EDGE` | killed: traj_navtriple_and_edge_fields |
| B29 | `J_Cay` denominator + 2e-13 | killed: f28 right Jacobians vs mpmath |
| B37 | `from_edge` `Dt = edge.duration` | survives; **equivalent** |
| B40 | Eq. (22b) `Y' x (1 + 2e-14)` | killed: f29 Eq. (22b) |
| B42 | combined bias-ω covariance uses σ_ba | killed: c25b, b3_make_params, … |
| B43 | integration covariance σ (not σ²) | killed: c25b, b3_make_params, … |
| B45 | `gal3_distance` 2-norm | killed: b3_models_and_distance |
| B54 | GTSAM `integrateMeasurement` float32 dt | killed: c06b, c05c, c18, … |
| B56 | A_vR uses `dR_ik dR_step` | killed: c06b, c06c, … |
| B57 | Ba velocity block uses `dR_ik dR_step` | killed: c06b, c06c, … |
| B61 | `cay` via `inv(I − P)(I + P)` | survives; **equivalent** |
| B67 | Cayley noise input `J_Cayᵀ` | killed: f26, f27, f28 covariance |
| B70 | `gal3_from_navstate` ignores t | killed: b3_gal3_adapters |
| B71 | `gal3_lift` ignores t | killed: c18, b3_gal3_adapters |
| B72 | `endpoint_F2_swapped` unswapped | killed: c18, b3_endpoint_relations |
| B73 | component-wise model swaps v/p tangent blocks | killed: c20, b3_models_and_distance |
| B74 | dp update x (1 + 1e-9) | killed: c05b, c13e, c05c, … |
| B75 | `cayley_vector` identity below 1e-5 | killed: f25 series branches |
| B76 | `LOG_NEAR_PI_COS` = −0.95 | survives; **equivalent** |
| B77 | `so3_exp` K² coefficient as 2 sin²(θ/2)/θ² | survives; **equivalent** |
| B78 | M1 K² term x (1 + 1e-13) | killed: f27 M1/M2 vs quadrature |
| B79 | `corrected_increment_gal3`: `g Dt_ij` → `g pim.dt_total` | survives; **equivalent** (proof) |
| B80 | `isotropic_precision` x (1 + 1e-15) | killed: c12 |
| B81 | Eq. (21c) `A_pa db_a` x (1 + 1e-12) | survives; **equivalent** |
| B82 | Eq. (22b) Γ argument x (1 + 5e-15) | killed: f29 Eq. (22b) |
| B83 | FD `J_R` x (1 + 1e-8) | killed: c05b, c13e, c05c, … |

**Count (38 new mutants, B01-B83):** 30 killed by reference tests, 8 equivalent, 0 non-equivalent survivors. The 3 probes all survive and are not equivalent. The 3 replays are killed. In total, 44 file-level mutants were run.

### 5.2 `tests/test_mutation.py` replay

On the clean tree all 164 tests in `test_mutation.py` pass (inside the 279). That includes:
* the pass-12 file-edit replays [A07, A11] with their control copy;
* the location meta-test.

My independent retypes (R07, R11, R11b) agree with both kills and their `E ` lines.

### 5.3 Equivalence rulings

| mutant | quantity | τ (claim) | measured | rule | ruling |
|---|---|---|---|---|---|
| B07 | Eq. (63) covariance | C06 | **Proof:** with zero densities and Σ = 0, the mutant also runs `A·0·Aᵀ + B·0·Bᵀ`, which is exactly 0 in IEEE arithmetic for finite A, B. Positive densities are unaffected. Negative densities are outside the domain. Advisory: no test pins `has_noise == False` at zero densities | 3 | equivalent |
| B24 | time-varying stream → C19, C21 | C19 1e-14 | **Bit-identical** at the claimed (n = 40, dt = 2.5e-3) stream. U₄₀ is identical too. Elsewhere (n, dt ∈ {(14, 1e-3), (4000, 1e-3), (400, 7e-3)}) the max change is 1.78e-15 = 1 ulp of the largest \|a\| (10.36); both readings are the stated formula | 4 (2 ulp floor) | equivalent |
| B37 | Eq. (21b)/(21c) residuals | C10 (b) 1e-12 | Dt 0.5000000000000003 → 0.5. Residual change ≤ **3.6e-15** (C10 ground-truth states, both edges): 280x inside τ, 28x inside τ/10. The mutant's Dt is the exact edge length | 1 | equivalent |
| B61 | `cay`, Eq. (16), B2 states | C09 1e-12; C22 1e-6 | `cay` vs 40-digit mpmath (\|ξ\| ≤ 28.2): 7.8e-16 (clean 8.3e-16). C09 on-manifold over 1e4 chart draws: 2.3e-14 (clean 4.4e-14). KF2 B2 states move ≤ 1.9e-14 | 1, 4 (no worse than clean) | equivalent |
| B76 | `so3_log` at θ ∈ (2π/3, 161.8°) | consumers: C05 ratio band, C22 1e-3 relative, FD J (C21 1e-6 relative) | Log error vs 40-digit: **1.9e-14** (clean 8.9e-16). No claim reaches this angle range: FD `J_R` arguments are ~1e-5 rad, C22 gaps ~1e-8, and C05 errors ≤ 1e-4 | 1 | equivalent. Advisory: no test pins `so3_log` accuracy on (2π/3, ~162°) |
| B77 | `so3_exp` → B1 R | C24 1e-10 | `so3_exp` vs mpmath `expm` (θ ∈ [1e-8, 3.1]): 6.7e-16, same as clean. B1 KF2 R vs exact 1.7e-14 (clean 2.8e-15), 590x inside τ/10 | 1 | equivalent |
| B79 | `corrected_increment_gal3` | C20 | **Proof:** `Dt_ij = pim.dt_total` is assigned on the line above, so the two expressions are the same value | 3 | equivalent |
| B81 | Eq. (21c) | C10 1e-9 | 1.4e-12 absolute with A_pa ~ N(0, 1), \|db_a\| ≤ 1 (realistic \|A_pa\| ~ 0.1, db ~ 1e-2: ~1e-15) | 1 | equivalent |

### 5.4 Complete survivor list (non-equivalent)

| mutant | change | distinguishing input and effect | threshold it violates |
|---|---|---|---|
| **P01** | Eq. (19) ½ → ½ + 4e-15 for \|ξ\| > 10 | C11 identity broken by **1.81e-13** for \|ξ\| ∈ (10, 28.2); F30 (b) reads 7.2e-15 x (1 + \|ξ\|) < 1e-14 x (1 + \|ξ\|) | C11 τ = 1e-13 (Rule 1 needs ≤ 1e-14) |
| **P02** | Eq. (19) + 2e-13 for \|ξ\| > 20 | C11 identity **2.04e-13**; C09 on-manifold **2.21e-13**; F30 (b) reads 9.6e-15 per (1 + \|ξ\|) | C11 τ; C09 τ/10 = 1e-13 |
| **P03** | Eq. (19) ξ x (1 + 8e-15) for \|ξ\| > 5 | C11 identity **1.81e-13**; F30 (b) 7.2e-15 per (1 + \|ξ\|); F30 (a) does not reach \|ξ\| > 5 often enough to see it | C11 τ |

**Why they survive:**
* `test_c09` and `test_c11` draw small \|ξ\|.
* F30 (a) draws \|ξ\| ≲ 6.
* F30 (b) reaches the chart edge, but its bound grows with \|ξ\| to 2.9e-13, which is above the claim tolerance. A defect confined to large \|ξ\| (a branch, or an error that grows with \|ξ\|) of up to ~2.9x τ is invisible.

The probes are deliberately branch-shaped. They are not offered as plausible typos, but as witnesses that the test cannot resolve the claim tolerance on part of its own declared domain.

### 5.5 Full-suite runs of all 11 fast-set survivors

| mutant | full suite (279) | notes |
|---|---|---|
| B61, B76, B77, B81 | **279 passed** | |
| P01, P02, P03 | 277 passed, **2 errors** | `test_pass12_mutant_is_killed_as_a_file_edit[A07, A11]`: the A11 anchor is no longer unique (anchor fixture, not a kill) |
| B07 | 257 passed, **1 failed, 21 errors** | all are `test_pass8/9/10_mutant_is_killed_as_a_file_edit[*]` anchor errors. The "failure" is `[F23]` raising `edit anchor … not unique` |
| B24, B79 | 242 passed, **37 errors** | `test_pass7_mutant_is_killed_as_a_file_edit[*]` anchor errors |
| B37 | 241 passed, **1 failed, 37 errors** | failed: `test_pass5_replay_is_faithful`, which compares a test-local replay of `from_edge` with `src`'s `from_edge` field for field (a change detector; pass-11 Sec. 6 class). Errors: pass-7 anchors |

**None of these are kills by a reference test.**

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**Carried over, unchanged:**
* `test_m04_swapping_y_and_yprime_is_a_genuine_no_op`;
* `test_pass5_replay_is_faithful` (it fires on B37, which is equivalent);
* the two F28 retyped-residual tests whose clean value is 0.0 by construction;
* the location-only meta-tests (`test_pass11/12_killers_are_reference_checks`). These are sound here only because the killers were read (Sec. 3.2).

**Noted in this pass:**
* **F30 (b)'s scaled bound** exceeds the claim tolerance above \|ξ\| ≈ 9 (M2-F31).
* **Untested properties** (advisory; the mutants touching them are equivalent):
  * `has_noise == False` at zero densities (B07);
  * `so3_log` accuracy on (2π/3, ~162°) (B76).
* **C07(b) wording:** the left-convention failure is a sup, not a minimum (Sec. 4 note).

## 7. New findings

* **M2-F31 (gate-blocking).** The scaled bound of `test_f30_eq19_matches_a_retyped_cleared_eq16_at_large_xi` part (b), 1e-14 (1 + \|ξ\|), allows 2.9e-13 absolute at the C07 chart edge, while the clean absolute error is 3.55e-15. Three Eq. (19) probes (P01-P03) break the C11 identity by 1.8e-13 to 2.0e-13 (> τ = 1e-13), and P02 breaks C09's on-manifold residual by 2.2e-13 (> τ/10). All pass the full suite. **Suggested fix:** 2e-15 (1 + \|ξ\|), or flat 1e-14 absolute, in part (b) (clean headroom 5.2x / 2.8x), or a chart-regime C11-identity check at 1e-14.
* **M2-F32 (advisory).** C07(b)'s "LEFT convention fails by at least 1e-1" holds only in sup: at d ∥ φ the left error is 4.0e-8, and the minimum over 300 unit directions is 0.025. Also: `has_noise` False at zero densities is unpinned (B07), and so is `so3_log` on (2π/3, ~162°) (B76). Both are harmless to every claim.

## 8. Final state

This commit contains this report only. The worktree is otherwise clean:
* the scratch directory `results/p13` (gitignored) was deleted;
* every mutant copy was a temp directory under it and was removed;
* `src/`, `tests/` and `docs/claims/` are untouched.

The full suite on the clean tree (`803f6e9`): **279 passed**.

**Gate: NOT passed.** There are 3 non-equivalent survivors (P01, P02, P03), recorded as **M2-F31**. M2-F30's named survivors A07 and A11 are killed. M2-C36 is REFUTED (Σ scales as k², not k).
