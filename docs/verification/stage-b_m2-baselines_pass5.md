# Stage-B verification, pass 5: M2 baselines (`stage-b/m2-baselines` @ `42c3660`)

This is the fifth independent adversarial pass. It follows commit `42c3660`, which says it
fixes pass-4 finding M2-F12 by adding `test_c13c`.

Scope:
- the ruling on F12;
- every claim in `docs/claims/M2.yaml` (C01–C25);
- one claim that exists only in the pass brief, **M2-C28**.

Every claim was judged by computation and treated the same way. I did not read
`docs/02_*`, any derivation narrative, any `seeds*.yaml`, or anything under
`.cgis_SEEDS_BACKUP*`. All scratch code and mutant copies lived outside the worktree, in
the session scratchpad. I did not edit `src/`, `tests/` or `docs/claims/`. `git status`
was clean before and after the sweep.

On the clean tree, the full suite at HEAD gives **100 passed**. `tests/test_baselines.py`
alone gives 32/32.

---

## 0. Summary

| | |
|---|---|
| **F12** | **PARTIALLY FIXED.** The *methods* are now pinned: Q25, Q45 and Q46 are dead as real file edits, and so is every non-equivalent wiring mutant of `rotation_residual`, `rotation_residual_affine`, `rotation_residual_truncated` and `xi` that I wrote (17 of 17). The factor's *frozen rotation input* `dRbar` is still pinned by nothing. `from_edge` storing `pim.dR.T` passes the whole suite, while the factor's Eq. (19) residual at ground truth goes from 2.0e-8 to **0.39 / 0.44**. That is new finding **M2-F13** (medium) |
| References in `test_c13c` | Independent. `_indep_eq19`, `_indep_eq22b`, `_indep_truncated`, `_indep_rot`, `_indep_cay` and `_indep_hat` are retyped and call nothing in `src`. But on the edge factor, `xi`, `dRbar`, `J_R` and `bbar` are read **from the factor under test**, so c13c checks wiring given the fields and never checks the fields |
| Synthetic-factor preconditions | Strong enough for transposes, signs and offset swaps. R12 (a hard-coded fixture `b_g` offset in `xi`) is killed **only** by the synthetic factor |
| Verdicts | **25 CONFIRMED (C01–C25), 1 REFUTED (C28), 0 UNTESTABLE** |
| C28 | **REFUTED.** max \|Cay(au)Cay(bu) − Cay((a+b)u)\| = **9.9e-3** against the claimed 1e-12. On the fixture, the B2 product vs Cay(n w dt) is **7.4e-4 / 1.6e-3** |
| Mutation sweep | **38 new file-level mutants (R01–R38): 24 killed, 14 survived. 5 survivors are medium (all on `dRbar`), 9 are low or equivalent** |
| Regression replay | 13 earlier mutants, replayed as real file edits (M02, M05, M12, M15, M17, N04, P17, P29, P33, P35, Q25, Q45, Q46): **13/13 killed**. Nothing regressed for F01–F11 |

---

## 1. Independent re-implementation

Scratch module `ind/core.py` imports nothing from `src/` or `tests/`. It contains:
- constants retyped from the M2-C01 text;
- `Exp = expm ∘ hat` and `Log = logm`;
- `Cay = inv(I − φ̂/2)(I + φ̂/2)`;
- the Eq. (9) Euler sums with a pluggable retraction;
- the exact increment from one 5x5 `expm`;
- central-FD bias Jacobians at eps = 1e-5.

It also has my own **Eq. (19)**, typed as `R_j − A − ½ R_j ξ̂ − ½ A ξ̂` (with A = R_i ΔR̄), and
my own **Eq. (22b)**: `R_j(I + Γ/2) − R_i ΔR̄(I − Γ/2) − ½(R_i ΔR̄ X + R_j X)`, with
Γ = (J_R b̄_g)^ and X = (J_R b_g)^.

### 1.1 F12: the factor methods against my Eq. (19) / (22b)

Setup:
- 300 random states per factor;
- R_i = Exp(U(−2, 2)³);
- b_g = b̄_g + U(−0.08, 0.08)³;
- the on-manifold state is R_j = R_i ΔR̄ Cay(ξ);
- the perturbed state is *left*-perturbed, Exp(U(−1, 1)³) R_j. The test perturbs on the right, so this is a different direction.

| factor | on-manifold \|r19\|, \|r22b\| | perturbed: method − my ref (19 / 22b / trunc) | J_R asym | ΔR̄ asym |
|---|---|---|---|---|
| edge 0 (`from_edge`) | 2.7e-16, 4.3e-16 | 2.2e-16 / 0 / 0 | 0.087 | 0.347 |
| edge 1 (`from_edge`) | 3.3e-16, 3.7e-16 | 2.2e-16 / 0 / 0 | 0.102 | 0.404 |
| my own synthetic factor (random J_R, b̄_g ≠ b̄_a) | 2.7e-16, 3.8e-16 | 2.2e-16 / 0 / 0 | 0.569 | 0.933 |

The methods are wired to Eq. (19), Eq. (22b) and the truncation today.

The factor's frozen *fields* also agree with my own Cayley integrator:

| field | edge 0 | edge 1 |
|---|---|---|
| ΔR̄ | 2.2e-14 | 2.2e-14 |
| Δv̄ | 1.2e-15 | 5.3e-14 |
| Δp̄ | 3.1e-16 | 8.9e-15 |
| J_R, relative | 1.7e-10 | 2.9e-10 |

For contrast, an Exp integrator differs by 1.9e-8 / 4.0e-8 in ΔR̄ and by 2.3e-7 / 3.6e-7
in J_R. So the code is correct. The problem is that nothing *requires* the ΔR̄ value (§2).

### 1.2 Per-claim measurements (B1, B2)

| claim | my result |
|---|---|
| C01 | Streams are `array_equal` to `src`. Two separate processes give an identical SHA-256 over both streams and all constants (`1245ca0f…`). My Exp(rot_vec_0) differs from `T.R_0` by 1.1e-16 because I used `expm`, not Rodrigues. That is a property of my reference, not of determinism |
| C02 | M1 vs quad 6.9e-17 / 5.6e-17; M2 vs quad 1.9e-16 / 1.1e-16. 5x5 expm vs closed form ≤ 2.0e-15. Chained vs the `REF_*` literals: KF1 v, p 1.6e-15 / 2.0e-15; KF2 R, v, p 2.2e-16 / 2.6e-15 / 3.3e-15 |
| C04 | Ratios: v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086, 0.50043, 0.50022, 0.50011. `src` B1 vs mine: 2.2e-14 / 6.3e-15 / 9.6e-16 |
| C05 | Quadratic ratios (R, v, p): 100.1 / 99.9 / 99.95 on edge 0 and 99.9 / 99.99 / 100.0 on edge 1. `src` J vs mine ≤ 2.9e-8 |
| C06 | My Eq. (63) (Jr by power series) vs `src`, anisotropic: **3.3e-15 / 1.9e-15** relative. My Eq. (63) vs my FD of the noisy integrator (200 steps × 6 noise directions): **4.0e-9 / 2.8e-9**. Symmetry and PSD hold at every step, and the trace is monotone. Vs GTSAM: magnitude 4.0e-4 / 4.9e-4; raw R–v 0.170 / 0.195. After M = diag(I, ΔRᵀ, ΔRᵀ): rot 2.3e-14 / 3.5e-15, R–v 5.0e-3, R–p 7.5e-3 |
| C07 | Cay(2tan(θ/2)a) vs Exp: 2.8e-15 over 2000 draws. Right FD 7.7e-7 (FD-limited); left **1.46**. `src` Eq. (4) vs mine: 0 |
| C08 | 2.7e-15 over 5000 draws. Minimum det 1.0000000001 |
| C09 | On-manifold 2.9e-16. Perturbed by Exp(1e-3 u): minimum 6.1e-4 (≥ 1e-4) |
| C10 | At ground truth, \|r^v\| = 2.1709e-3 / **2.7603e-3** and \|r^p\| = 5.38e-4 / 6.81e-4. `src` factor vs my residual ≤ 5.6e-14 |
| C11 | 4.4e-16 over 2000 draws |
| C12 | Isotropic 7.7e-13; weighted 44.4 (≥ 1.0). τ = 3/1.8 |
| C13 | J_R is bitwise unchanged after `predicted_rotation` at a perturbed bias. Truncation gap 4.8e-5 at \|ξ\| = 1e-2 (≥ 1e-6). Factor J_R is Cayley's (1.7e-10), not Exp's (2.3e-7) |
| C22 | KF2 B2 − B1: v 6.167e-8, p 4.376e-8. R is 2.81e-8 as \|Log\|∞, 2.996e-8 as a matrix ∞-norm, 3.51e-8 as \|Log\|₂. Per edge \|Log\| = 2.32574e-8 / 5.51753e-8, against ‖ω‖³T dt²/12 = 2.32574e-8 / 5.51753e-8 (relative 1.9e-7 / 3.3e-7). Ratio 0.2500 per halving |
| C23 | KF2 B1 − exact: v 1.2867e-3, p 1.2031e-3. Ratios: B1 v 0.50028→0.50003, p 0.5000; B2 v 0.50026→0.50003, p 0.49998→0.5000 |
| C24 | B1 rotation vs `REF_R_KF2`: 1.7e-14. B2: 3.0e-8. B3: see C19 |

Note on C22: the "3.0e-8 ± 10%" figure holds for the matrix ∞-norm (2.996e-8) and for
\|Log\|∞ (2.81e-8, −6.3%). It fails for \|Log\|₂ (3.51e-8). The claim is true under the
test's metric. The metric is not named in the statement, which is a hygiene gap and not a
refutation.

### 1.3 Baseline 3 and GTSAM facts (my own matrices, graph and FD)

| claim | my result |
|---|---|
| C03 | `predict` vs my chain from an arbitrary NavState: 7.9e-15 / 0 / 0 (edge 0) and 8.4e-15 / 8.9e-16 / 0 (edge 1) |
| C14 | col 3 = [10, 20, 30], col 4 = [1, 2, 3], M[3,4] = 7. `position` = `translation` = [1, 2, 3]. `velocity` = [10, 20, 30]. The round-trip through `Gal3(M)` holds |
| C15 | Hat(e_k) is nonzero only in the stated block for all 10 k. Expmap(1e-6 e_k) moves only v (k = 3..5), only p (6..8) and only t (9). Expmap vs expm(Hat): 2.2e-16 |
| C16 | Remapped Gal3 log vs `localCoordinates`: 0.1008 / 0.1678 / **0.10763** (0→1, 1→2, 0→2). Vs `NavState.Logmap`: 0.0 for all three |
| C17 | Accessors as claimed. `biasHatVector` = [b_a; b_g]. The adapters round-trip, and one swap is not the identity |
| C18 | My own 5x5 matrices. W-form 2.2e-16 / 1.6e-15, and t_j = t_i + Δt (4.2 at t_i = 3.7). F2 form 2.2e-16 at t_i = 0; at t_i = 3.7 it gives **18.1485 = 9.81·0.5·3.7**. Swapped form 2.45. `src` W vs mine 4.4e-16. (The hint's 36.3 uses Δt = 1; the statement's \|g\|Δt t_i is what holds) |
| C19 | Galilean vs exact ≤ 2.0e-14. Tangent dv 2.17e-3 / 2.76e-3. Time-varying stream with my ZOH V_k: right 1.1e-16, left 6.2e-4 |
| C20 | My FD J at eps = 1e-7. Right model ≤ 4.3e-10 (edge 0) and 3.4e-10 (edge 1). Component-wise over 30 random directions at 5e-2: **1.27e-2 / 1.28e-2**. J[0:3,0:3] = J[9,:] = 0 |
| C21 | My ZOH V_k, my FD C_k, GTSAM `AdjointMap`. vs FD of full re-integration: 1.9e-9 / 1.7e-9 / **2.6e-9** relative at steps 10, 20 and 40 |
| C25 | My own graph, two noise configurations, and my own prior sigmas (1e-2 and 1e-1, different from the test's). Ranks 30 / 39 / 45; +B0 prior 45. Cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]. pv energy 2.5371 in both configurations. Graph error ≤ 6.7e-23 |

---

## 2. Ruling on F12: **PARTIALLY FIXED**, and new finding **M2-F13 (medium)**

### What is fixed

`test_c13c` calls all three factor methods. It compares them, at both on-manifold and
perturbed states, against references that are genuinely retyped. A `grep` of the six
`_indep_*` helpers finds no `B2.`, `B1.` or `T.` call.

The synthetic factor's preconditions do their job:
- ΔR̄ is non-symmetric, so a transpose is visible;
- R_i is non-symmetric;
- \|db_g\| ≥ 1e-3, so a sign error is visible;
- \|J_R(b̄_a − b̄_g)\| ≥ 1e-3, so an exchanged offset is visible;
- J_R is non-symmetric, so a transposed J_R is visible.

Every non-equivalent method mutant dies, including two that are not obvious:
- R14 replaces Eq. (19) by the uncleared Eq. (16) residual R_j − R_i ΔR̄ Cay(ξ). It vanishes on the same set, but it is not polynomial. It differs off-manifold by the factor (I − ξ̂/2).
- R12 hard-codes the fixture's b_g in `xi`. It is invisible on the edge factor and is killed only by the synthetic factor.

### What is not fixed

On the real (edge) factor, c13c builds its reference from `f.dRbar`, `f.J_R` and
`f.bbar`, which are the fields of the object under test. It therefore checks the methods
*given* the fields. The fields are pinned elsewhere as follows:

| field | pinned by |
|---|---|
| `J_R`, `A_*` | c13b, against an independent Cayley FD |
| `bbar[0:3]` | c13: `xi(B_G) == 0` (R08 killed) |
| `dvbar`, `dpbar` | c10 magnitude (R03, R09 killed) |
| `Dt` | c10 gravity terms (R06 killed) |
| **`dRbar`** | **nothing** |

c13 uses `factor.dRbar` only self-referentially (`R_i @ factor.dRbar @ cay(...)`), plus a
*lower* bound on the frozen-vs-relinearised gap, which a wrong ΔR̄ only enlarges.

Impact on the factor-level residual at the fixture's ground-truth keyframes (`f13.py`),
where the correct value is the Cayley deficit:

| mutant (suite stays green) | \|Log(ΔR̄ᵀ ΔR̄_mut)\| | \|r19\| = \|r22b\| at GT, edge 0 / 1 |
|---|---|---|
| correct factor | 0 | 2.0e-8 / 4.1e-8 |
| R01 `dRbar = pim.dR.T` | 0.447 / 0.596 | **3.9e-1 / 4.4e-1** |
| R04 ΔR̄ integrated at b_g = 0 | 9.9e-3 | **7.7e-3 / 7.7e-3** |
| R02 ΔR̄ from the Forster (Exp) integrator | 2.3e-8 / 5.5e-8 | 9.7e-15 / 7.8e-15 (the Cayley deficit erased, the M12 pattern) |
| R05 ΔR̄ from `VARIANT_REPARAM` (= Exp) | same as R02 | same as R02 |
| R10 ΔR̄ · Cay(1e-6 e₁) | 1.0e-6 | 9.1e-7 / 1.0e-6 |

F12's own words name "the residuals the QCQP/SDP milestone will consume". A transposed
ΔR̄ makes those residuals wrong by 0.4 with the suite at 100/100, so the exposure F12
described is only half closed.

**Suggested kill** (one assertion). For each edge:
- pin `from_edge(edge, BIAS_TRUE).dRbar` against `_indep_cayley_increment(...)[0]` (already in the test file) to ≤ 1e-12;
- and/or assert the factor's \|r19\| at the ground-truth keyframes equals the per-edge deficit ‖ω‖³T dt²/12 within 10%.

Either one kills R01, R02, R04, R05 and R10. The first is ~2e-14 on the correct code, and
R02 / R05 sit at 1.9e-8.

---

## 3. Mutation sweep: 38 new mutants (R01–R38), focused on `CayleyFactor` and its inputs

Method:
- each mutant is a file-level edit in a fresh scratch copy of `src/`, `tests/`, `orchestrator/`, `experiments/`, `conftest.py` and `pyproject.toml`;
- each anchor string was asserted unique;
- each copy ran the whole of `tests/test_baselines.py`;
- the control copy gives 32/32.

None of these repeats an M-, N-, P- or Q-mutant.

**Killed (24):**

| id | mutant | killed by |
|---|---|---|
| R03 | `dvbar`, `dpbar` from the Exp integrator | c10, c10b |
| R06 | `Dt = dt_total + dt` | c10, c10b |
| R07 | `J_R` stored transposed | c13b |
| R08 | `bbar` stored block-swapped | c13 |
| R09 | `dvbar × (1 + 1e-3)` | c10, c10b |
| R11 | `xi` uses J_Rᵀ | c13, c13c |
| R12 | `xi` offset hard-coded to the fixture b_g | **c13c (synthetic only)** |
| R14 | `rotation_residual` = uncleared Eq. (16), R_j − R_i ΔR̄ Cay(ξ) | c13c |
| R15 | `rotation_residual` returns rᵀ | c13c |
| R16 | `rotation_residual` swaps R_i ↔ R_j | c13c |
| R17 | `rotation_residual` × 2 | c13c |
| R19 | `rotation_residual` inline ξ with the b̄_a offset | c13c |
| R20 | `rotation_residual_affine` passes J_Rᵀ | c13c |
| R23 | `rotation_residual_affine` passes −b̄_g | c13c |
| R24 | `rotation_residual_affine` passes ΔR̄ᵀ | c13c |
| R25 | `rotation_residual_truncated` method uses −ξ | c13c |
| R26 | `rotation_residual_truncated` method uses ΔR̄ᵀ | c13c |
| R27 | `rotation_residual_truncated` method returns Eq. (19) | c13c |
| R28 | Eq. (22b): R̆ = (I − Γ/2) ΔR̄ (order) | c11, c13c |
| R29 | Eq. (22b): Y = ΔR̄ R_i X | c11, c13c |
| R30 | Eq. (19): ξ̂ on the left | c09, c11, c13, c13c |
| R32 | `cay` = solve(I + P, I − P) | 10 tests |
| R34 | `velocity_residual` swaps db_g ↔ db_a | c10b |
| R35 | `position_residual` drops db_a | c10b |

**Survived, medium (5).** All five are M2-F13 (§2):

| id | mutant | measured effect |
|---|---|---|
| **R01** | `from_edge`: `dRbar = pim.dR.T` | factor r19 at GT 0.39 / 0.44 (correct 2e-8) |
| **R02** | `from_edge`: `dRbar` from the Forster/Exp integrator | erases the Cayley deficit; r19 at GT 1e-14 |
| **R04** | `from_edge`: `dRbar` integrated at b_g = 0 | r19 at GT 7.7e-3 |
| **R05** | `from_edge`: `dRbar` from `VARIANT_REPARAM` | same as R02 |
| **R10** | `from_edge`: `dRbar · Cay(1e-6 e₁)` | r19 at GT 1e-6, 25–50× the deficit |

**Survived, low or equivalent (9):**

| id | mutant | classification |
|---|---|---|
| R13 | `xi` = J_R b_g − J_R b̄_g | equivalent (distributivity, ≤ 1 ulp) |
| R18 | `rotation_residual` returns Eq. (22b) | equivalent in value (C11 identity, 4e-16) |
| R21 | `rotation_residual_affine` returns Eq. (19) | equivalent in **value**. It is not equivalent as a *structure*: it drops the Y/Y′ lifting that M3/M4 need. No numerical test can see it. Low, but worth noting for M3 |
| R22 | affine method passes (b_g − b̄_g, b̄ = 0), so Γ = 0 | same as R21: equal values, different lifting. Low |
| R31 | Eq. (16) uses Cay(−ξ)ᵀ | equivalent: Cay(−ξ) = Cay(ξ)ᵀ |
| R33 | `predicted_rotation` @ (I + 0·ξ̂) | no-op |
| R36 | `J_R × (1 + 5e-9)` | below c13b's 1e-8 bound. FD-level, low |
| R37 | `A_va × (1 + 5e-8)` | below c13b's 1e-7 bound (pass 4 Q26 used 1e-5). Low |
| R38 | `from_edge` stores `variant=VARIANT_REPARAM` (integration still raw) | equivalent: no code reads `CayleyFactor.variant` after construction. It is a dead field. Low hygiene |

### 3.1 Regression replay of earlier mutants as real file edits (13/13 killed)

| id | mutant | killed by |
|---|---|---|
| M02 | fold rotation first | c04, c05b, c13b |
| M05 | Γ sign in R̆ | c11, c13c |
| M12 | Cay → Exp | c05b, c13b, c22, c22b, c23, c24 |
| M15 | `xi` drops b̄_g | c13, c13c |
| M17 | Eq. (19) ξ sign | c09, c11, c13, c13c |
| N04 | Eq. (21b) A_vg sign | c10b |
| P17 | `has_noise` or → and | c06c |
| P29 | truncation with (I − ξ̂) | c13, c13c |
| P33 | `weighted_square` unsquared | c12 |
| P35 | factor Jacobians from Forster | c13b |
| Q25 | affine method takes b̄_a | c13c |
| Q45 | method uses −ξ | c13c |
| Q46 | method uses ΔR̄ᵀ | c13c |

---

## 4. Tests and assertions that cannot fail, or that compare the code to itself

Carried over and still true:
1. `test_c10` (a): the self-consistent residual is a tautology (labelled as one).
2. `test_c13`: `J_R.tobytes()` is unchanged, but nothing writes `J_R`. Also,
   `predicted_rotation == R_i ΔR̄ B2.cay(J_R δ)` re-states the code using
   `factor.dRbar` and `B2.cay`. Its gap check has a *lower* bound only, so a wrong ΔR̄
   (R01, R04) passes it more easily.
3. `test_c05`: `allclose(corrected[0], base.dR @ so3_exp(J_R @ db_g))` re-states
   `bias_corrected_increment`.
4. `test_c08`: `closed >= 1` is `1 + x²/4 ≥ 1`.
5. `test_c20`: `J[9,:] = 0` holds for any bias-only perturbation.
6. `test_c25`: ranks 30 and 39 are full row rank (labelled).
7. `test_c06`: the symmetry / PSD / monotone assertions are structural.
8. `test_c01`: `R0` vs `expm(T.hat(...))` shares `T.hat`.
9. `test_mutation.py::test_m04…` asserts a genuine no-op, by design.

The pass-4 item 10 (c06c's accel-only reference assertion) is resolved by `42c3660`.

New in `test_c13c`:

10. `assert np.abs(_indep_eq19(R_i, R_j, dRbar, xi)).max() <= 1e-13` checks the
    reference against the reference. It is an identity for any orthogonal `dRbar`, so
    `src` cannot make it fail.
11. `assert np.abs(R_i - R_i.T).max() >= 1e-3`, `norm(db_g) >= 1e-3` and
    `smallest_off >= 1e-3` are preconditions on RNG-generated test data. Only a change
    of seed can fail them.
12. For the **edge** factor, every c13c comparison takes `dRbar`, `J_R` and `bbar` from
    the factor itself, so no from-edge field error can fail c13c (M2-F13). The edge
    factor's `dRbar`-asymmetry precondition is the only line that reads `src`'s ΔR̄
    with any teeth, and R01 keeps ΔR̄ᵀ just as asymmetric.

---

## 5. Verdicts per claim

| id | verdict | attack |
|---|---|---|
| C01 | CONFIRMED | Retyped constants, streams `array_equal`. SHA-256 identical across two processes |
| C02 | CONFIRMED | quad ≤ 1.9e-16; 5x5 expm ≤ 2e-15; `REF_*` literals ≤ 3.3e-15 |
| C03 | CONFIRMED | My chain vs `predict`: 8.4e-15 |
| C04 | CONFIRMED | Own Euler: ratios v 0.5000, p 0.5009→0.5001 |
| C05 | CONFIRMED | Own FD J: ratios 99.9–100.1. M02 killed |
| C06 | CONFIRMED | Own Eq. (63) vs `src` 3.3e-15; vs own noise FD 4.0e-9 (anisotropic). Chart numbers reproduced. P17 killed |
| C07 | CONFIRMED | 2.8e-15; right 7.7e-7, left 1.46 |
| C08 | CONFIRMED | 2.7e-15; minimum 1.0000000001 |
| C09 | CONFIRMED | 2.9e-16 on-manifold, ≥ 6.1e-4 off. M17 and R30 killed |
| C10 | CONFIRMED | 2.7603e-3 / 2.1709e-3 at ground truth. N04, R34 and R35 killed |
| C11 | CONFIRMED | 4.4e-16. M05, R28 and R29 killed |
| C12 | CONFIRMED | 7.7e-13 / 44.4. P33 killed |
| C13 | CONFIRMED | Truncation gap 4.8e-5; J_R Cayley (1.7e-10 vs 2.3e-7); J_R frozen. M15, P29 and R11 killed. The code's ΔR̄ is right (2.2e-14), but see F13: nothing requires it |
| C14 | CONFIRMED | gtsam directly |
| C15 | CONFIRMED | All 10 Hat blocks and Expmap semantics; Expmap = expm(Hat) to 2.2e-16 |
| C16 | CONFIRMED | 0.10763 vs `localCoordinates`, 0.0 vs `Logmap` |
| C17 | CONFIRMED | gtsam directly; adapters round-trip |
| C18 | CONFIRMED | Own matrices: W 1.6e-15; F2 18.1485 = \|g\|Δt t_i; swapped 2.45 |
| C19 | CONFIRMED | ≤ 2.0e-14; Tangent 2.17e-3; right 1.1e-16 vs left 6.2e-4 |
| C20 | CONFIRMED | Right ≤ 4.3e-10; component-wise 1.27e-2 / 1.28e-2 at 5e-2 |
| C21 | CONFIRMED | Own recursion 2.6e-9 relative at step 40 |
| C22 | CONFIRMED | 6.167e-8 / 4.376e-8 / 3.0e-8 (matrix ∞-norm). Deficit equals ‖ω‖³T dt²/12 to 3e-7 relative. M12 killed. The metric is unstated (§1.2) |
| C23 | CONFIRMED | 1.2867e-3 / 1.2031e-3; ratios in band for B1 and B2 |
| C24 | CONFIRMED | B3 ≤ 2.0e-14; B1 rotation 1.7e-14; B2 rotation 3.0e-8 |
| C25 | CONFIRMED | Own graph, own prior sigmas, two noise configurations: 30 / 39 / 45 (+B0: 45), identical cosines, pv energy 2.5371 |
| **C28** | **REFUTED** | See below |

### M2-C28: REFUTED

Attack: 20 000 random unit axes u and a, b ∈ U(−0.3, 0.3). I compared
Cay(au) Cay(bu) with Cay((a+b)u) using my own `Cay`.

| quantity | measured | claimed |
|---|---|---|
| max \|Cay(au)Cay(bu) − Cay((a+b)u)\|∞ | **9.90e-3** | ≤ 1e-12 |
| at a = b = 0.3, u = 1/√3 | 8.41e-3 | |
| tan-half-angle law Cay(au)Cay(bu) = Cay(((a+b)/(1 − ab/4)) u) | 3.3e-16 | |

Why it fails:
- For parallel arguments, Cay(au) is a rotation by 2 arctan(a/2) about u.
- Angles add, so the composition is a rotation by 2arctan(a/2) + 2arctan(b/2).
- That is not 2 arctan((a+b)/2). The tangents of the half-angles compose by the addition law, not additively.
- Cay is not a one-parameter-subgroup map, because t ↦ Cay(tu) is not a homomorphism. Exp is one, which is exactly why the claim's analogy fails.

The "hence" part fails on the fixture as well:

| | edge 0 | edge 1 |
|---|---|---|
| B2 per-step product vs Cay(n w dt), matrix ∞-norm | **7.43e-4** | **1.60e-3** |
| rotation angle between them | 9.2e-4 | 2.2e-3 |
| `src` ΔR̄ vs Cay(n w dt) | 7.43e-4 | 1.60e-3 |

The product rotates by n·2arctan(θdt/2). Cay(nθdt) rotates by 2arctan(nθdt/2). The gap is
≈ (nθdt)³/12 − nθ³dt³/12 ≈ 9.3e-4 rad on edge 0, which matches the measurement.

The claim also contradicts M2-C22 (CONFIRMED). If the product were Cay(nwdt), the B1–B2
edge gap would be (‖ω‖T)³/12 ≈ 9e-4, not ‖ω‖³T dt²/12 = 2.3e-8.

**Counts: 25 CONFIRMED, 1 REFUTED (C28), 0 UNTESTABLE.**

---

## 6. Recommendations

1. **F13 (medium):** pin `CayleyFactor.from_edge(...).dRbar` against an independent
   Cayley integrator (`_indep_cayley_increment` is already in the test file) to
   ≤ 1e-12. Optionally, also assert the factor's ground-truth \|r19\| equals the
   per-edge Cayley deficit within 10%. This kills R01, R02, R04, R05 and R10.
2. Do not enter M2-C28 into the yaml. The true statement is the half-angle addition law
   Cay(au)Cay(bu) = Cay((a+b)/(1 − ab/4) u), for parallel arguments only.
3. Hygiene (low):
   - name the rotation metric in the M2-C22 "3.0e-8 ± 10%" statement;
   - either use or remove `CayleyFactor.variant`, which no code reads (R38);
   - note for M3 that no numeric test can distinguish the Eq. (22b) *lifting* from
     Eq. (19) (R21, R22), so the lifted structure needs a structural check (for
     example, affine dependence on (R_i, R_j, Y, Y′)) once it becomes a QCQP.
