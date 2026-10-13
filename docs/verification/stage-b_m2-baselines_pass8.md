# Stage-B verification, pass 8: M2 baselines (`stage-b/m2-baselines` @ `5de592a`)

This is the eighth independent adversarial pass. It follows these commits:
* `c52daff`: F18 defensive copies;
* `c2b75cb`: `tests/test_input_dependence.py`, its input inventory, and the bug fix in `corrected_increment_gal3`;
* `c05c605`: the `E `-line matcher;
* `663b89c` and `5de592a`.

Scope:
* rulings on M2-F16 to M2-F20;
* every claim in `docs/claims/M2.yaml` (C01-C25);
* one claim that exists only in the pass brief, **M2-C31**;
* 46 new file-level mutants (W-series);
* 53 advisory mutants from passes 2-7, replayed as real file edits (F20);
* 34 earlier gating mutants, replayed as real file edits.

Rules kept:
* I did not read `docs/00_*` or `docs/02_*`, any derivation narrative, any `seeds*.yaml`, or anything under `.cgis_SEEDS_BACKUP*`.
* All references, scripts and mutant copies lived in the session scratchpad.
* I did not edit `src/`, `tests/` or `docs/claims/`.
* `git status` was clean before and after every sweep.
* On the clean tree at `5de592a`, the full suite gives **203 passed** (53.3 s; the final run is in Sec. 8).

**Gate verdict: M2 does NOT pass.** The owner's rule allows zero surviving non-equivalent mutants, and this sweep leaves **29** of them (Sec. 5.4): 6 medium and 23 low.

## 0. Summary

| | |
|---|---|
| **F16** | **FIXED.** σ_g, σ_a, σ_int (including σ_int = 0), σ_bg and σ_ba are read back through GTSAM's getters at two density sets, to 1e-12 relative. They are also propagated one source at a time. U20, U26, U40, U23, U23b, Q28, Q29, P44, S35 and S36 all die as file edits. A ×(1+1e-6) scaling of `preint_cov` dies too (W42) |
| **F17** | **PARTIALLY FIXED.** The six named sites (U10, U13, U30, U31, U32, U38) and the fixture-degenerate U03-U07, U11, U12, U33, U34 and U39 die as file edits. But the inventory still leaves four inputs unvaried: (i) `corrected_increment_gal3(..., Dt=x)` under gravity. W02 survives, and its velocity error at Dt = 1.37 is **11.48**. (ii) `**kwargs` of `single_step` and `bias_jacobian_recursion`. The inventory calls dropping them EQUIVALENT, and that is **false**: a `params` with `body_P_sensor` moves `single_step` by 1.7e-4 and the recursion's J by **30% relative**. W13 and W14 survive. (iii) The scale of `isotropic_precision`'s input (W46). (iv) The `Rot3` branch of `gal3` (W10). New finding **M2-F21 (medium)** |
| **F18** | **PARTIALLY FIXED.** The code copies are present: every constructor stores copies, which I checked. But `test_f18_constructors_do_not_alias` takes the list of CayleyFactor fields from `src` itself (`B2.CayleyFactor._ARRAY_FIELDS`). As a result, W07 (drop `A_pa` from that list) and W08 (copy only `bbar`, so 8 of 9 fields alias the caller) **pass all 203 tests**. Outward aliasing is unpinned: W09 (`increment()` without copies) survives. An in-place edit of `GalileanPreintegration.bias_hat` desynchronises it from GTSAM's `biasHat`, and `predict` then moves by 4.2e-2 (S39). New finding **M2-F22 (medium)** |
| **F19** | **FIXED.** I checked this independently on my own S35 copy. S36's part-(a) message appears in the raw output (1 hit) but on no `E ` line (0 hits). S35's own message is on an `E ` line |
| **F20** | **NOT FIXED.** I replayed all 53 advisory mutants as file edits. 25 are killed, 14 are equivalent with written proofs (Sec. 5.2), 2 are N/A (the code they mutated has been removed), and **14 survive as non-equivalent**: P23, P46, Q01, Q39, Q44, R36, R37, S02, S16, S21, S31, S39, U47 and U49. Six of the 14 (R36, R37, S02, S16, S21, S39) die in the full suite, but only on change detectors (`test_pass5_replay_is_faithful`, the `pass6`/`pass7_runs` edit anchors). The pass-2 "tests that cannot fail" are ruled item by item in Sec. 6 |
| **C31** | **REFUTED.** With the fixture gravity, `predict(NavState(), b)` differs from the increment by **4.9050** in velocity (= \|g\| Δt) and **1.2263** in position (= \|g\| Δt²/2). This holds at db = 0 and db ≠ 0, on both edges. The claim is exact only at g = 0, where the difference from a zero-gravity PIM is 0.0. Once the gravity terms are removed, the difference is 3.6e-15. That is precisely the bug `c2b75cb` fixed |
| **Bug fix in `corrected_increment_gal3`** | **Correct.** At three gravities (zero, fixture, tilted), the fixed function equals a zero-gravity PIM's corrected increment to 3.6e-15. Its own new `Dt` path is unpinned (W02, see F21) |
| **New latent `src` defects** | **M2-F23 (medium):** `ForsterPreintegration(cov=C0)` with zero densities never propagates C0. After 200 steps it is still C0, while the propagated covariance differs by 4.96e-3 (**92% relative**). `test_b1_single_step_reads_every_field` sidesteps this with `sigma = 1e-30`, and it asserts the very gate that causes it. **M2-F24 (low):** `so3_log` has no near-π branch. Its error is 2.5e-7 at π − 1e-4, 2.7e-3 at π − 1e-6 and 2.6e8 at π − 1e-8. At θ = π it returns a vector of norm ~15 for 36% of axes |
| **Verdicts** | **25 CONFIRMED (C01-C25), 1 REFUTED (C31), 0 UNTESTABLE** |
| **Mutation sweep** | **46 new W-mutants.** 29 killed, 2 equivalent (W31, W32), **15 surviving non-equivalent** (W02, W07, W08, W09, W10, W11, W12, W13, W14, W15, W17, W18, W19, W20, W46; W13 dies only on a change detector). Advisory replay: 14 survivors (above). Earlier-mutant replay: **34/34 killed** |
| **Complete survivor list** | **29 non-equivalent** (Sec. 5.4): 6 medium (W02, W07, W08, W13, W14, W46) and 23 low |

---

## 1. Independent references (no `src` imports)

The scratch module `ind/core.py` and the scratch scripts contain:
* constants retyped from the M2-C01 text;
* `Exp = expm∘hat`, a retyped Rodrigues formula, and my own `Log`;
* `Cay = inv(I − φ^/2)(I + φ^/2)`;
* the Eq. (9) Euler sums with a pluggable retraction;
* the exact increment from one 5x5 `expm`;
* Gal(3) as my own 5x5 matrices, with `logm`;
* my own FD bias Jacobians;
* my own recursion, with a numeric adjoint obtained by conjugating my own hat;
* my own factor graph. Its parameters differ from the test's: densities 3e-4 / 2e-2 / 2e-6, random walks 3e-5 / 5e-4, prior sigmas 5e-2 / 2e-1, bias prior 1e-1, GPS 0.5.

`src` is imported only as the system under test. C31 used raw GTSAM only.

**Independence of the new references in `tests/test_input_dependence.py`.** These helpers are retyped, and none of them calls `src`: `ihat`, `iexp`, `ilog`, `icay`, `ieuler`, `izoh`, `izoh_stream`, `igal3*`, `_fd_step_jacobians`, `_fd_noise_covariance`, `_ifd_jacobians`, `_ieq21b/c`, and the gravity and bias literals.

The tests still take two things from `src`:
* the `Edge` streams and `time_varying_stream`. Both are pinned against retyped values in `test_traj_*`.
* `B3.make_params`, inside `_b3_cov`, as the system under test.

One reference is **self-referential**: `test_f18` iterates over `B2.CayleyFactor._ARRAY_FIELDS` (see F22).

## 2. M2-C31: REFUTED

Set-up: GTSAM `PreintegratedImuMeasurementsG` on each fixture edge at `BIAS_TRUE`. I compared its `predict` against my 5x5 `expm` ZOH increment and against a zero-gravity PIM's `predict`.

| n_gravity | db | predict − increment: v | p | after removing g Δt and ½ g Δt² |
|---|---|---|---|---|
| 0 | 0 / 7.1e-3 | 0.0 vs zero-g PIM (vs ZOH 2.0e-14 at db = 0) | 0.0 | — |
| [0, 0, −9.81] | 0 | **4.9050** | **1.2263** | 3.6e-15 / 1.6e-15 |
| [0, 0, −9.81] | 7.1e-3 | **4.9050** | **1.2263** | 3.6e-15 / 1.6e-15 |
| [1.3, −2.0, −9.5] | 0 and 7.1e-3 | 4.7500 (∞-norm of g Δt) | 1.1875 | 3.6e-15 / 1.6e-15 |

Rotation is unaffected (8.3e-16).

"Whatever gravity vector the PIM's params carry" is false for every g ≠ 0. From the identity, `predict` evaluates the M2-C03 endpoint relation with R_i = I and v_i = p_i = 0. That relation adds g Δt to velocity and ½ g Δt² to position. The increment can be read off `predict()` directly only at g = 0, which is the setting of the C20 test hint.

## 3. Rulings on F16-F19

**F16: FIXED.**
* `test_b3_make_params_stores_every_input` reads every stored covariance back through GTSAM's getters, to 1e-12 relative. It does this at two density sets, both `combined` values and two gravities, and it also checks σ_int = 0 → 0.
* `test_b3_densities_reach_the_covariance` propagates each source alone on the 0.2 s and 0.73 s edges.
* Killed as file edits: U20, U26, U40 (×1.005), U23, U23b, Q28, Q29, P44, S35, S36, and W42 (`preint_cov` ×(1+1e-6)).

**F17: PARTIALLY FIXED; new finding M2-F21 (medium).**

All six named leaks and all fixture-degenerate sites die as file edits. My own replay covers U10, U13, U30, U31, U32, U38, U03, U12, U34, U39 and V12; `PASS7_MUTANTS` covers the rest.

The inventory's promise (every input of every public callable is varied) is not met:

* **W02.** `corrected_increment_gal3`'s gravity removal uses `Dt` instead of `dt_total`.
  * Result: survives 203/203.
  * Effect: at Dt = 1.37 on a 0.2 s edge, velocity is wrong by **11.48**.
  * Why it survives: when `Dt` is given, the test checks only `.time()`.
* **W13 / W14.** `bias_jacobian_recursion` (W13) and `single_step` (W14) drop `**kwargs`.
  * The inventory docstring calls this "an EQUIVALENT mutant" because "gravity does not enter an increment". That argument covers gravity only.
  * `params` can carry `body_P_sensor`. With `Pose3(Rz(0.3), (0.1, 0.2, 0))`, `single_step` moves by **1.67e-4** and the 10-step recursion's J moves by **0.296 relative**.
  * W14 survives 203/203. W13 dies only on the `pass7_runs` anchor, which is a change detector.
* **W46.** `isotropic_precision` clamps tr Σ at 1e-6.
  * Result: survives. The test uses Σ = LLᵀ + 0.1 I, so tr Σ = O(1).
  * This fixture's realistic rotation-block trace is **4.3e-8**. There, τ would be 3.0e6 instead of 7e7, a 23x mis-weighting.
  * U47 (`+1e-30`) is the same gap at Σ = 1e-30 I: τ is 7.5e29 instead of 1e30.
* **W10.** `gal3(R, ...)` is never called with a `Rot3` argument, so its `isinstance` branch is never run. A transposed branch survives, with an effect of 0.94 in R.

**F18: PARTIALLY FIXED; new finding M2-F22 (medium).**

The code fix is real: `CayleyFactor(A_pa=A).A_pa is A` is `False`, as it is for every field I checked. Five gaps remain:

* **Self-referential field list.** The CayleyFactor regression test takes its field list from the code under test. Removing a field from `_ARRAY_FIELDS` therefore also removes it from the check. W07 (drop `A_pa`) and W08 (keep only `bbar`) **pass 203/203**.
* **Only `bbar` is pinned independently**, through the `from_edge` / `xi` checks. That is why F18-B2 dies.
* **Outward aliasing is not pinned.** W09 (`increment()` returns `self.dR` and friends without copies) survives.
* **`ForsterPreintegration.b_g` / `b_a` are views into `bias_hat`.** This is the shipped code, not a mutant: writing to `pim.b_g` changes `bias_hat`.
* **`GalileanPreintegration.bias_hat` is a public mutable array that duplicates GTSAM's `biasHat`.** After `gal.bias_hat[:] = BIAS_TRUE`, `predict` moves by 4.22e-2 while `pim.biasHatVector()` stays at 0 (S39).

**F19: FIXED.** `_failure_lines` keeps only lines that start with `E `. On my own S35 copy (claim suite, default traceback), the S36 message appears once in the raw output and on zero `E ` lines, and S35's own message is on an `E ` line. `test_f19_matcher_ignores_the_echoed_source` asserts the same.

## 4. New latent defects found while proving equivalences

* **M2-F23 (medium, latent): the Forster covariance is frozen at zero densities.**
  * `has_noise` gates the whole propagation, not just the noise term.
  * With σ_g = σ_a = 0 and C0 = diag(1..9)·1e-4, `ForsterPreintegration(cov=C0)` still holds a `cov` bit-identical to C0 after 200 fixture steps.
  * With densities of 1e-300 instead, it propagates to a matrix that differs by 4.96e-3 (92% relative).
  * `cov` is a public constructor input, and the inventory lists it.
  * `test_b1_single_step_reads_every_field` uses `sigma_g = sigma_a = 1e-30` for its "C0 alone" check. It also *asserts* `has_noise is False` at zero densities, which pins the defective gate.
* **M2-F24 (low, latent): `so3_log` near π.**
  * There is no antipodal branch. The error of `so3_log(Exp(θa))` is:

    | θ | π − 1e-2 | π − 1e-4 | π − 1e-6 | π − 1e-8 |
    |---|---|---|---|---|
    | error | 2.9e-11 | 2.5e-7 | 2.7e-3 | 2.6e8 |

  * At θ = π, rounding pushes (tr R − 1)/2 below −1 for 7206 of 20000 random axes. For those axes the function returns vectors of norm ~15, for example [−8.5, −11.4, −5.7].
  * Every M2 use is small-angle. The inventory's "both branches" does not cover this regime.
  * This is why W20 (drop the lower clamp, giving NaN instead of a garbage vector) survives.

## 5. Mutation sweep

**Harness.**
* Each mutant is a string edit of a `src` file in a fresh scratch copy of `src/ tests/ orchestrator/ experiments/ docs/ conftest.py pyproject.toml`. The edit anchor is asserted unique.
* `tests/test_baselines.py` and `tests/test_input_dependence.py` (75 tests) run first.
* Every survivor is then re-run on the **full** suite (203 tests).
* A comment-only control copy passes 75/75.

**"Guard-only"** means the mutant dies only on `test_pass5_replay_is_faithful`, a `test_pass6_*` test or the `pass7_runs` fixture. These are change detectors: they compare against a copy of `src`, or they fail when an edit anchor is missing.

### 5.1 New mutants (W01-W46)

**Killed (29):**

| mutant | change | killed by |
|---|---|---|
| W01 | `corrected_increment` gravity hard-coded to the fixture | c20 |
| W03 | position gravity term dropped | b3_corrected |
| W04 | velocity gravity hard-coded | c20 |
| W05 | db not block-swapped | c20, b3_corrected |
| W06 | gravity taken from `make_params()` | c20 |
| W16 | `so3_log` small branch ×(1+1e-4) | b1_bias_jacobians_fd |
| W21 | `dt_total` input dropped | b1_single_step |
| W22 | `cov` input dropped | b1_single_step |
| W23 | B1 `integrate_edge_states` drops g_W | b1_integrate_edge_states |
| W24 | `bias_jacobians_fd` minus side drops kwargs | b2_factor_from_edge |
| W25 | Jr K² ×1.0001 | c06b, b1_so3_right_jacobian |
| W26 | per-axis σ_g reversed | 5 tests |
| W27 | Qa += 1e-3 Qg | c06c, two b1 tests |
| W28 | factor db = None → bbar | c10, b2_factor_methods |
| W29 | J_Cay /4.0001 | b2_cayley_maps |
| W30 | Y′ = Ξ R_j | 5 tests |
| W33 | closed-form det + 1e-15 | b2_cayley_maps |
| W34 | remap (0, 3, 6) | c15, c16, b3_gal3_adapters |
| W35 | Ad_Vᵀ | c21, b3_recursion |
| W36 | `gal3_from_navstate` ignores t | b3_gal3_adapters |
| W37 | `time_shift` takes abs | c18, b3_endpoint |
| W38 | right model left-composed | c20, b3_models |
| W39 | wrapper dt = 2.5e-3 | 8 tests |
| W40 | B3 `integrate_edge` drops kwargs | c06, c06b, two b3 tests |
| W41 | random-walk setters on non-combined params | 23 tests (AttributeError) |
| W42 | `preint_cov` ×(1+1e-6) | c06b |
| W43 | `exact_increment` uses T = T_EDGE | traj_navtriple |
| W44 | NavTriple aliases its input | f18 |
| W45 | `b_a` ×(1+1e-9) | 9 tests |

**Equivalent (2):** W31 and W32. The proofs are in Sec. 5.2.

**Surviving non-equivalent (15):** W02, W07, W08, W09, W10, W11, W12, W13 (guard-only), W14, W15, W17, W18, W19, W20, W46. Sec. 5.4 lists each one with a distinguishing input.

### 5.2 Equivalence proofs (every mutant I call equivalent)

| mutant | change | proof |
|---|---|---|
| Q36 | `gal3_distance` → ‖Log(B⁻¹A)‖∞ | For θ < π, Log(X⁻¹) = −Log(X): Exp(−ξ) = Exp(ξ)⁻¹, and Log is the principal inverse. ‖−ξ‖∞ = ‖ξ‖∞. At θ = π both forms return the same value; the measured difference is 0.0 at θ = π, π − 1e-9 and π − 1e-6. Over 2000 random pairs the numerical gap reaches 1.5e-7. That gap comes from GTSAM's ill-conditioned Logmap near π, which affects both sides, not from the mutation |
| R13 | `J(b − b̄)` → `Jb − Jb̄` | distributivity |
| R18 | `rotation_residual` returns Eq. (22b) | Let ξ = J(b_g − b̄_g), Γ = (Jb̄_g)^ and X = (Jb_g)^. By linearity of hat, R_j(I + Γ/2) − A(I − Γ/2) − ½(AX + R_jX) = (R_j − A) − ½(R_j + A)(X − Γ), which is Eq. (19). This is the C11 identity |
| R21 | affine method returns Eq. (19) | the same identity, read in the other direction |
| R22 | affine method with arguments (b_g − b̄_g, 0) | Γ = 0 and X = ξ^, which gives Eq. (19) |
| R31 | Eq. (16) uses Cay(−ξ)ᵀ | Pᵀ = −P, so Cay(−ξ)ᵀ = [(I+P)⁻¹(I−P)]ᵀ = (I+P)(I−P)⁻¹. That equals (I−P)⁻¹(I+P) because rational functions of P commute |
| R33 | `predicted_rotation @ (I + 0·ξ^)` | For finite ξ, 0·ξ^ = ±0, so I + (±0) = I and M·I = M exactly (each entry is M_ij·1 + Σ M_ik·0). For non-finite ξ the original output is already non-finite |
| S03 | ΔR̄ ΔR̄ᵀ ΔR̄ | identity on SO(3), since RRᵀ = I |
| S11 | Δt = `edge.duration` | `integrate_edge` integrates exactly n_steps steps of `edge.dt`, and Σₙ dt = n·dt |
| S22 | factor preintegration with densities on | Data flow: `_propagate_covariance` writes only `self.cov`. `from_edge` reads `dR`, `dv`, `dp` and `dt_total`, and calls `bias_jacobians_fd` without densities, so its output is bitwise identical. The only full-suite failure is the `pass7_runs` anchor |
| S26 | Cay = (I+P)(I−P)⁻¹ | the factors commute, as in R31 |
| S43 | `1.0 * x` in `from_gtsam_bias` | IEEE 754 multiplication by 1.0 is exact for every value, including −0, ±inf and NaN |
| U46 | Eq. (21c) subtracts the bias terms in the other order | commutativity and associativity of addition |
| U48 | `preint_cov` transposed | P = APAᵀ + BQBᵀ is symmetric. The measured asymmetry is 1.2e-22 absolute, below 1 ulp at the 5.0e-5 scale |
| W31 | `r @ Wᵀ @ r` | rᵀWᵀr = (rᵀWr)ᵀ, and a scalar is its own transpose |
| W32 | `variant != RAW` instead of `== REPARAM` | `__init__` raises `ValueError` for any variant outside {raw, reparam}, and on that set the two predicates agree |
| M04 | Y ↔ Y′ (historical) | −½(Y + Y′) is symmetric in Y and Y′ |

All of these except S22, S43 and M04 are algebraic identities in exact arithmetic. In floating point, R13, R18, R21, R22, R31, S03, S11, S26, U46, U48 and W31 can differ in the last bits. I accept them as equivalent under the owner's "algebraic identity" clause.

### 5.3 Advisory replay (F20): 53 mutants as file edits

| result | mutants |
|---|---|
| **killed (25)** | N05, N06, N07, N08, N09, N10, N11, N13, N28 (by c06b, c06c and the b1 tests); P22 (c05, b1_increment_error); P40, U39 (b3_corrected); P44, Q28, Q29, U40 (c25b and the b3 tests); Q04, Q08 (b1 tests); Q16, R38 (same edit as S20), S45, U08 (same as U09), U28 (same as U29), U03 (b2 and b3 inventory tests); Q32 (b3_bias_jacobian_fd_reads_eps) |
| **equivalent (14)** | Q36, R13, R18, R21, R22, R31, R33, S03, S11, S22, S26, S43, U46, U48 (proofs in Sec. 5.2) |
| **N/A (2)** | P06 (`REF_R_KF1` no longer exists) and P45 (`predict` no longer takes a bias argument) |
| **survive (14)** | P23, P46, Q01, Q39, Q44, R36\*, R37\*, S02\*, S16\*, S21\*, S31, S39\*, U47, U49. \* = guard-only kill in the full suite |

### 5.4 Complete survivor list (non-equivalent; full suite or guard-only)

**Medium (6):**

| mutant | change | distinguishing input and effect |
|---|---|---|
| W02 | `corrected_increment_gal3` removes gravity using the given `Dt` | Dt = 1.37 on a 0.2 s edge at fixture g: Δv = **11.48** |
| W07 | `A_pa` dropped from `_ARRAY_FIELDS` | `CayleyFactor(A_pa=A)`, then `A[...] = 0`: `position_residual` moves by \|A_pa db_a\| |
| W08 | only `bbar` copied | the same as W07, for 8 fields |
| W13 | `bias_jacobian_recursion` V̂_k drops kwargs (guard-only) | `params` with `body_P_sensor`: J moves by **29.6% relative** |
| W14 | `single_step` drops kwargs | the same params: ΔV = 1.67e-4 |
| W46 | `isotropic_precision` clamps tr Σ ≥ 1e-6 | realistic Σ (tr 4.3e-8): τ = 3.0e6 instead of 7e7 |

**Low (23):**

| mutant | change | effect |
|---|---|---|
| S39 | Galilean `predict` uses `pim.biasHat()` (guard-only) | after an in-place edit of `bias_hat`: Δv = 4.22e-2 |
| W09 | `increment()` returns no copies | a caller's write corrupts the PIM |
| W10 | `gal3` `Rot3` branch transposed | 0.94 in R |
| U47 | τ = 3/(tr Σ + 1e-30) | at Σ = 1e-30 I: τ = 7.5e29 instead of 1e30 |
| W20 | `so3_log` lower clamp dropped | at Exp(πa): NaN instead of the already-wrong value (F24) |
| Q01 | Jr small-angle K sign | 6e-9 at θ = 8.8e-9; the test bound is 1e-8 |
| Q39 | `so3_exp` small branch drops K² | 1.5e-17 absolute; 3.7e-9 relative off-diagonal |
| W17 | M1 small branch drops K² | 2.5e-19 absolute; 9.3e-10 relative off-diagonal |
| W18 | M2 small branch drops K² | 2.3e-20 absolute; 6.9e-10 relative |
| W15 | `cayley_vector` threshold 1e-6 | 4.2e-14 relative at θ = 7e-7 |
| W19 | `SMALL_ANGLE` = 1e-6 | `so3_log` off by 3.3e-14 relative at θ = 5e-7 |
| Q44 | `so3_log` clamp at 1 − 1e-15 | ≤ 2.96e-16 relative, but bitwise different on 343 of 400 inputs |
| P23 | `FD_EPS` = 1e-3 | J changes by 2.6e-8 relative |
| S31 | `FD_EPS` = 1e-4 | 2.3e-8 |
| S21 | factor FD eps = 1e-3 (guard-only) | 3.2e-8 |
| W11 | `J_FD_EPS` = 1e-5 (C20's stated setting is 1e-7) | 5.4e-9 |
| P46 | `C_FD_EPS` = 1e-3 | 4.6e-10 at dt = 2.5e-3 |
| W12 | recursion ignores `eps` | 4.5e-8 relative at dt = 0.3, eps = 5e-2 |
| R36 | J_R ×(1+5e-9) (guard-only) | 5e-9 relative |
| R37 | A_va ×(1+5e-8) (guard-only) | 5e-8 |
| S16 | A_pg ×(1+5e-8) (guard-only) | 5e-8 |
| S02 | ΔR̄·Cay(3e-13 e₃) (guard-only) | 3e-13 rad |
| U49 | B_a position block ×(1+1e-9) | 1e-9 relative |

**A structural note for the gate.** Eighteen of the low survivors are perturbations below a test tolerance or choices of FD step: Q01, Q39, Q44, W15, W17, W18, W19, P23, S31, S21, W11, P46, W12, R36, R37, S16, S02 and U49. Their effects range from 1e-16 to 5e-8 relative, and three of them (P23, P46, S21) make the FD *more* accurate. No finite-tolerance test can kill an arbitrarily small perturbation, so adding tests cannot close this class under the gate as written. The owner has two options:
* state a tolerance-equivalence rule, for example "effect below the claim's tolerance and below the quantity's own approximation error counts as equivalent"; or
* accept that the gate cannot be met.

I have applied no such rule here: all eighteen are listed as survivors.

### 5.5 Regression replay of earlier gating mutants: 34/34 killed

Each one is my own file edit. "b1 ×2" means two b1 inventory tests, and so on.

| mutant | killed by |
|---|---|
| M02 | 19 tests |
| M05 | c11, c13c, c13d, b2 ×2 |
| M08 | c18, b3_endpoint |
| M09 | c21, b3_recursion |
| M12 | 13 tests |
| M15 | c13, c13c, c13d, b2 ×2 |
| M17 | c09, c11, c13, c13c, b2 ×2 |
| N04 | c10b, c13e, b2 ×2 |
| N24 | c05b, b1 ×2 |
| N27 | c06b, c06c, b1 ×2 |
| F05a3 | 11 tests |
| P17 | c06c, b1 ×2 |
| P35 | c13b, c13e, b2_from_edge |
| Q25 | c13c, c13d, b2 |
| Q45 | c13c, b2 |
| R01 | c13d, c13e, b2 |
| R10 | c13d, c13e, b2 |
| S35 | c25b, b3 ×2 |
| S36 | c25b, b3 ×2 |
| S46 | c13e, b2 |
| S47 | c13e, b2, f18 |
| U10 | b2 ×2 |
| U13 | b1_states |
| U20 | b3 ×3 |
| U26 | b3 ×3 |
| U30 | b3 ×2 |
| U31 | b3_corrected |
| U32 | b3_states |
| U38 | b3_recursion |
| U23b | b3 ×2 |
| U12 | b1 ×2 |
| U34 | b3_endpoint |
| V12 (the pre-fix `corrected_increment_gal3`) | b3_corrected |
| F18-B3 | f18 |

Nothing regressed.

## 6. Tests and assertions that cannot fail, or that compare the code to itself

**The pass-2 list, item by item:**

1. **c10 (a).** Still a tautology, and labelled as one. Not a defect.
2. **c10 (b), \|r\| = \|Δv̄ − Δv_exact\|.** Still an identity in the sums. The docstring is now correct.
3. **c13, `J_R.tobytes()`.** Still cannot fail: no code path writes `J_R`.
4. **c11 vs the literal Y/Y′ swap.** A genuine no-op (M04, proven).
5. **c12, isotropic half.** SO(3) orthogonality. P33 is now killed.
6. **c08, `closed >= 1`.** Still `1 + x²/4 ≥ 1`.
7. **c25, ranks 30 and 39.** Full row rank, and labelled as structural.
8. **c06, structural checks.** These still cannot see block errors on their own. The teeth now come from c06b, c06c, `test_b1_single_step` and `test_b1_covariance_over_a_stream`, and all 8 pass-2 covariance mutants die.
9. **c20, component-wise lower bound.** N13 now dies (c20, b3_models).
10. **c01, bitwise determinism.** Fails only on nondeterminism. The bias literals are now pinned: N29 dies in `test_mutation`, and the inventory retypes the `FIX_*` literals.
11. **c22, upper bound only.** The teeth are in c22b and in the lower bound.

Items 1, 3, 4, 6 and 7 are structural and correctly labelled. None can be "killed", and none hides a surviving mutant.

**New in this pass:**

12. **`test_f18_constructors_do_not_alias` reads `B2.CayleyFactor._ARRAY_FIELDS` from `src`.** Shrinking the list shrinks the check (W07, W08).
13. **`test_b1_single_step_reads_every_field` asserts `not pim.has_noise` at zero densities.** It therefore **pins the defective gate of M2-F23**. Its "C0 alone" check uses σ = 1e-30 and cannot see the frozen covariance.
14. **Three preconditions are on references only, so `src` cannot fail them:**
    * `visible >= 1e-6` in `test_b1_bias_jacobians_fd`;
    * `rel(want, small) >= 1e-5` in `test_b3_bias_jacobian_fd_reads_eps`;
    * `moved >= 1e-5` in `test_b3_corrected_increment_gal3`.
15. **`test_b3_corrected_increment_gal3`'s `Dt = 1.37` check reads only `.time()`** (W02).
16. **`test_b2_precision_and_weighted_square` uses only tr Σ = O(1)** (W46, U47).
17. **`test_pass7_sweep_covers_every_listed_survivor` and `test_pass6_sweep_covers_every_medium_survivor` compare literal sets within the same file.**
18. **`test_pass5_replay_is_faithful` and the `pass6_*` / `pass7_runs` anchor assertions are change detectors.** They "kill" R36, R37, S02, S16, S21, S22, S39 and W13, and they also fire on true no-ops (S03, S11, S22).

## 7. Per-claim verdicts

All numbers are mine. `src` was called only as the system under test.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | Two subprocess generations of the 400-sample stream are byte-identical (19200 B) to each other and to my retyped streams. All constants match the retyped literals, R_0 to 1.1e-16 against `expm` |
| C02 | CONFIRMED | M1 vs `quad_vec`: 7.6e-17 / 5.6e-17. M2 vs nested quad: 1.8e-16 / 9.7e-17. My `expm` chain vs REF: KF1 v 1.6e-15, p 2.0e-15; KF2 R 2.2e-16, v 2.6e-15, p 3.3e-15 |
| C03 | CONFIRMED | GTSAM `predict` from a nonzero NavState vs my chain: R 7.7e-15 / 6.8e-15, v 4.4e-16, p 4.4e-16 |
| C04 | CONFIRMED | Edge 0 ratios: v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086, 0.50043, 0.50022, 0.50011. Edge 1 is similar. `src` vs my Euler sums: 0.0 |
| C05 | CONFIRMED | Error ratio between \|db\| = 1e-2 and 1e-3, against my own re-integration, R / v / p: 99.96 / 100.00 / 100.00 (edge 0) and 99.82 / 100.05 / 100.04 (edge 1) |
| C06 | CONFIRMED | Asymmetry ≤ 1.7e-20, minimum eigenvalue −4.0e-23, trace monotone on both edges. The inventory's FD tests carry the correctness check, and every pass-2 covariance mutant dies there |
| C07 | CONFIRMED | (a) 1.6e-15 over 2000 axes, θ ∈ (0, 3). (b) Right convention 6.0e-7 (FD-limited). Left convention: max 1.37, min 2.2e-2, so the "fails by ≥ 1e-1" holds only as a sup over draws |
| C08 | CONFIRMED | 5000 draws with \|ξ\| ≤ 5: deviation 3.6e-15; minimum value 1.000004 |
| C09 | CONFIRMED | On-manifold 3.3e-16. Perturbed by Exp(1e-3 u): minimum 6.4e-4 ≥ 1e-4 |
| C10 | CONFIRMED | Self-consistent states: 3.6e-15 / 5.6e-14. Ground truth: v 2.1709e-3 / **2.7603e-3**, p 5.4e-4 / 6.8e-4; equal to my \|Δv̄ − Δv_exact\| to 5.3e-16 |
| C11 | CONFIRMED | 2000 draws: 4.4e-16 |
| C12 | CONFIRMED | With Rodrigues rotations: isotropic 8.9e-15; W = diag(1, 2, 5) max 32.5 ≥ 1 |
| C13 | CONFIRMED | `J_R` bytes unchanged after evaluating at a perturbed bias. Exact vs truncated residual at \|ξ\| = 1e-2: 4.6e-5 ≥ 1e-6 |
| C14 | CONFIRMED | Column 3 = [10, 20, 30], column 4 = [1, 2, 3], M[3,4] = 7. The accessors hold, the `Gal3(M)` round trip gives 0.0, and `translation() == position()` |
| C15 | CONFIRMED | Hat support is correct for all 10 e_k. Expmap(1e-6 e_k) moves only the stated component. Expmap vs `expm(Hat)`: 2.2e-16. GTSAM's Hat equals mine exactly |
| C16 | CONFIRMED | KF0-KF2 **0.107628** (KF0-KF1 0.1008, KF1-KF2 0.1678). Against `NavState.Logmap`: ≤ 1.8e-15 on all three pairs |
| C17 | CONFIRMED | `ConstantBias([1,2,3],[4,5,6])` accessors and `vector()` are as stated. `biasHatVector` = [b_a; b_g] exactly. The adapters round-trip, and one application moves the vector by 0.082 |
| C18 | CONFIRMED | My 5x5 product, at t_i = 0 and 3.7: the W-form error is ≤ 1.8e-15 and t_j = 0.5 / 4.2. The F2 error is 4.4e-16 at t_i = 0 and **18.148500** = \|g\|·0.5·3.7 at t_i = 3.7. The swapped order gives 2.4525 = \|g\| Δt² at t_i = 0. `src` W-form vs mine: 4.4e-16 |
| C19 | CONFIRMED | Galilean vs ZOH: R 1.0e-14, v 2.0e-14, p 3.8e-15. Tangent dv 2.17e-3 / 2.76e-3 ≥ 1e-4. Time-varying stream: right accumulation 4.4e-16, left 6.3e-4 ≥ 1e-5 |
| C20 | CONFIRMED | My FD J (my 5x5 logm, GTSAM predict at g = 0): J[0:3,0:3] = J[9,:] = 0 exactly. Right composition 3.4e-13 → 1.6e-10 (edge 0) and 3.9e-13 → 2.1e-10 (edge 1), linear in \|db\|. Component-wise at 5e-2 over 30 directions: max **1.32e-2** / 1.13e-2. `src` J vs mine: 3.6e-9 relative |
| C21 | CONFIRMED | My recursion (my V_k, numeric adjoint and FD C_k) vs my FD re-integration: 3.0e-8 relative over 40 steps (\|J\| = 0.10). `src` vs mine: 1.5e-8 |
| C22 | CONFIRMED | KF2 B1-B2: v 6.17e-8, p 4.38e-8, R max-abs **2.9965e-8** (mine 2.9965e-8). Per edge: 2.325744e-8 / 5.517531e-8 vs the formula's 2.325744e-8 / 5.517533e-8. Ratios 0.25000, 0.25000, 0.25000, 0.25002 |
| C23 | CONFIRMED | B1-B3: v 1.2867e-3, p 1.2031e-3, R 3.1e-15. B2-B3: R 3.0e-8. My ratios: B1 v 0.50028 → 0.50003, p 0.49998 → 0.50000; B2 v 0.50026 → 0.50003, p 0.49998 → 0.50000 |
| C24 | CONFIRMED | B3 at KF1 / KF2 vs my ZOH: ≤ 1.6e-14 (vs the REF literals: 7.1e-15). B1 rotation: 9.5e-15 / 2.7e-15. B2 rotation: 2.0e-8 / 3.0e-8 |
| C25 | CONFIRMED | My own graph, densities and sigmas. Shapes 30x45, 39x45, 45x45; ranks 30 / 39 / 45; rank with a bias prior 45. Cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]; pose/velocity energy 2.5371. Graph error 2.1e-23; unwhitened residual 2.0e-14; s_min/s_max 4.0e-9 |
| **C31** | **REFUTED** | Sec. 2. At fixture gravity, Δv = **4.9050** = \|g\| Δt and Δp = **1.2263** = ½ \|g\| Δt², at db = 0 and db ≠ 0, on both edges. Holds only at g = 0 |

**Counts: 25 CONFIRMED, 1 REFUTED (C31), 0 UNTESTABLE.**

## 8. Final state

This commit contains the report only.

Full suite on the clean tree (`b4a5bde`, i.e. `5de592a` plus the report skeleton): **203 passed** in 698.8 s. The run was slow because the machine was shared with other jobs; load average was 60-74. `git status` was clean before and after every sweep, and every mutant lived in a scratchpad copy.
