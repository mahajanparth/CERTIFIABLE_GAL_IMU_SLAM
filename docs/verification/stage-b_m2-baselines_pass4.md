# Stage-B verification, pass 4: M2 baselines (`stage-b/m2-baselines` @ `0a0e66b`)

This is the fourth independent adversarial pass. It follows two commits:
- `61d1141`, which says it fixes pass-3 finding F11,
- `0a0e66b`, which rewords the M2-C25 statement (the F10 remainder).

Scope:
- every claim in `docs/claims/M2.yaml` (C01–C25),
- one claim that exists only in the pass brief (**M2-C27**).

Each claim was judged by computation, and every claim was treated the same way. I did not
read `docs/02_*`, any derivation narrative, any `seeds*.yaml`, or anything under
`.cgis_SEEDS_BACKUP*`. All scratch code and mutant copies lived outside the worktree. I
did not edit `src/`, `tests/` or `docs/claims/`. `git status` was clean before and after
the sweep.

The full suite at HEAD on the clean tree gives **96 passed**: 31 in `test_baselines.py`,
and the rest in `test_mutation.py` and `test_orchestrator.py`.

---

## 0. Summary

| | |
|---|---|
| F10 (C25 wording) | **FIXED**. I derived the new structural argument and confirmed it numerically (§3) |
| F11 (P17, P35, P22, dead code) | **FIXED**. All three medium survivors are killed by the new assertions, and so are my own neighbouring variants (Q05, Q24, Q26, Q10). Both dead-code items are gone |
| Verdicts | **25 CONFIRMED (C01–C25), 1 REFUTED (C27), 0 UNTESTABLE** |
| C27 | **REFUTED**. B2 − B1 increments differ by \|Δdv\| = **4.53e-8 / 1.03e-7** and \|Δdp\| = **7.48e-9 / 1.69e-8** (edge 0 / edge 1). The claimed bound is 1e-12 |
| My mutation sweep | **46 new file-level mutants: 33 killed, 13 survived**. 3 survivors are medium, 10 are low or equivalent |
| New finding | **M2-F12 (medium)**: `CayleyFactor.rotation_residual` and `CayleyFactor.rotation_residual_affine` are called by no test. These are the factor-level Eq. (19) / Eq. (22b) residuals that a QCQP/SDP would consume, so any wiring error in them passes the whole suite (Q25, Q45, Q46) |

---

## 1. Independent re-implementations (no import from `src/` or `tests/` for any reference)

The scratch module `indep_core.py` contains:
- constants retyped from the M2-C01 text;
- `Exp = scipy.linalg.expm ∘ hat` and `Log = logm`;
- exact ZOH from one 5x5 `expm`;
- the Forster Eq. (35)–(37) Euler sums with a pluggable retraction;
- **Eq. (63)** written out:
  - A = [[ΔR_stepᵀ, 0, 0], [−ΔR_ik â dt, I, 0], [−½ ΔR_ik â dt², I dt, I]]
  - B_g = [J_r(ω dt) dt; 0; 0], with J_r taken from its power series Σ(−ad)^k/(k+1)!
  - B_a = [0; ΔR_ik dt; ½ ΔR_ik dt²]
  - Q = diag(σ²)/dt;
- Cayley as an explicit `inv(I − φ̂/2)(I + φ̂/2)`;
- my own finite-difference linearisation of the noisy Euler integrator with respect to
  per-step noise (prefix · step(η) · suffix).

The `src` code was called only as the object under test.

### 1.1 Baselines 1 and 2

| item | independent result |
|---|---|
| constants / streams (C01) | identical to `src`. `array_equal` holds on both streams |
| quad of ∫Exp(ws)ds vs 5x5 expm (C02) | 8.9e-16 / 1.1e-16. `src` M1 vs quad: 6.9e-17. Sec. 0.4 literals: KF1 v, p 1.6e-15 / 2.0e-15; KF2 R, v, p 2.2e-16 / 2.6e-15 / 3.3e-15 |
| `src` B1 vs my Euler sums, n = 200 | R 2.2e-14, v 6.3e-15, p 9.6e-16 |
| C04 truncation at n = 200 | dv 2.170879e-3 / 2.760238e-3. Ratios: v 0.5000 ×4; p 0.5009, 0.5004, 0.5002, 0.5001 |
| C05, my FD Jacobians on my integrator | quadratic ratios (R, v, p) are 100.05 / 100.01 / 100.01 on edge 0 and 100.14 / 100.00 / 100.00 on edge 1. `src` J vs mine ≤ 2.8e-8 (J_vg, FD round-off) |
| C06, my Eq. (63) vs `src` cov (anisotropic) | 7.2e-14 / 5.7e-14 relative per block |
| C06, my Eq. (63) vs my FD of the noisy integrator | **2.7e-9 / 1.8e-9**. This is a correctness check, not only a structural one |
| C06, gyro-only / accel-only vs `src` | gyro-only 7.2e-14 / 5.7e-14 (tr 2.306e-7 / 2.254e-7). Accel-only 1.9e-15 / 2.3e-15 (tr 2.844e-4) |
| C06, structure (isotropic, every step) | asymmetry ≤ 1.7e-20, min eigenvalue −4.0e-23, trace monotone |
| C06 vs GTSAM | normalised magnitude 4.0e-4 / 4.9e-4. Raw R–v block 0.170 / 0.195 (isotropic) and 0.112 / 0.205 (anisotropic). After M = diag(I, ΔRᵀ, ΔRᵀ): rot 9e-16, R–v 5.0e-3, R–p 7.5e-3. All as the hint states |
| C07 | Cay(2tan(θ/2)a) vs Exp: 2.9e-15. Eq. (4) vs **right** FD Jacobian: 5.0e-10. Vs **left** FD: 0.384. `src` vs Eq. (4): 0 |
| C07, analytic derivation | Cay(φ) = Exp(ψ) with ψ = 2 arctan(\|φ\|/2) φ/\|φ\|, so J_right = J_r^Exp(ψ) · ∂ψ/∂φ. This equals Eq. (4) to **3.3e-16** for \|φ\| ∈ (0.01, 2). Eq. (4) is the right Jacobian *exactly*, not only to FD precision |
| C08 | 2.7e-15. The minimum of the determinant is 1.0000004 |
| C09 / C11 | 3.1e-16 on-manifold. Off-manifold minimum 1.5e-4 (≥ 1e-4). C11 2.7e-15 |
| C10 at ground truth | \|r^v\| 2.1709e-3 / **2.7602e-3**. \|r^p\| 5.38e-4 / 6.81e-4 |
| C12 | isotropic 6.2e-13, weighted 50.3 (≥ 1.0) |
| C13(b) | Cay vs (I + ξ̂) residual difference 5.0e-5 at \|ξ\| = 1e-2 |
| C13, factor J_R | vs my **Cayley** FD J_R: 3.1e-10 relative. Vs my **Exp** FD J_R: 2.3e-7. So the factor uses the Cayley Jacobians (P35 is fixed) |
| C22 | KF2 B2 − B1: v 6.167e-8, p 4.376e-8, R 2.9965e-8. Per-edge \|LogΔR\| 2.32574e-8 / 5.51753e-8, predicted ‖ω‖³T dt²/12 = 2.32574e-8 / 5.51753e-8 |
| C23 | KF2 B1 − exact: v 1.2867e-3, p 1.2031e-3. B1 order ratios: v 0.50028…0.50003, p 0.5000. B2: v 0.50026…0.50003, p 0.49998…0.5000. B2 R-gap ratio 0.25 |
| C24 | B1 rotation 1.7e-14. B2 rotation 2.9965e-8. v/p ≤ 2.43e-3 (Tier 2) |

### 1.2 Baseline 3 and the GTSAM facts

| item | independent result |
|---|---|
| C03 | `predict` vs my chain: 9.5e-15 / 0 / 4.4e-16 |
| C14 | matrix col 3 = [10, 20, 30], col 4 = [1, 2, 3], M[3,4] = 7. `position()` = `translation()` = [1, 2, 3] |
| C15 | Hat(e_k) blocks and Expmap(1e-6 e_k) semantics exactly as claimed. Expmap vs expm(Hat): 5.6e-17 |
| C16 | reordered Gal3 log vs `localCoordinates`: 0.1008 (0→1), 0.1678 (1→2), **0.10763** (0→2). Vs `NavState.Logmap`: 0.0 for all three |
| C17 | accelerometer = [1, 2, 3], gyroscope = [4, 5, 6], vector = [1…6]. `biasHatVector` = [b_a; b_g] |
| C18, my own 5x5 matrices | W-form 2.2e-16 / 1.6e-15, t_j = t_i + Δt. F2 form 18.1485 = \|g\|·0.5·3.7. Swapped form 2.45 at t_i = 0 |
| C19 | Galilean vs closed form: ≤ 2.0e-14. Tangent dv: 2.17e-3 / 2.76e-3. My ZOH V_k on the time-varying stream: right 1.1e-16, left 6.2e-4 |
| C20, my FD J (eps = 1e-7) | right composition ≤ 4.6e-10. Component-wise at 5e-2: 30-direction max 1.32e-2 / 1.35e-2, Nelder–Mead sup **1.419e-2 / 1.640e-2**. J[0:3,0:3] = J[9,:] = 0 |
| C21, my ZOH V_k, my FD C_k, GTSAM `AdjointMap` | 3.5e-9 relative at step 40 |
| C23 / C24 for B3 | B3 vs exact ≤ 1.6e-14. B1 − B3 at KF2: v 1.2867e-3, p 1.2031e-3, R 1.7e-14 |
| C25, my own graph (three noise configs, including anisotropic and different prior sigmas) | ranks 30 / 39 / 45 (+B0 prior: 45). Cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]. pv energy 2.5371 in all three configs. Graph error ≤ 1.1e-22 |

---

## 2. Ruling on F11: **FIXED**

| item | evidence |
|---|---|
| P17 (`has_noise` or→and) | `test_c06c` runs gyro-only and accel-only cases against the FD reference. My own gate mutant Q05 (`has_noise` ignores σ_a, so accel-only gives a zero covariance) is **killed by c06c**. Q14 (`integrate_edge` passes σ_a = σ_g) is killed by c06b and c06c. The c06c reference agrees with my FD to ~1e-9 |
| P35 (Forster Jacobians in CayleyFactor) | `test_c13b`'s reference (`_indep_cay`, own sums, own log) is independent of `src`. My variant Q24 (Jacobians from the *reparameterised* variant, which is Exp) is **killed**. Q26 (A_va × (1+1e-5)) is **also killed**, which is better than the c13b docstring claims for A_va. My own factor J_R vs Cayley FD: 3.1e-10 |
| P22 (`increment_error` ignores dp) | hand-computed single-component cases plus a per-component quadratic ratio. My Q10 (sum instead of max) is **killed by c05**. My per-component ratios are all ≈ 100 |
| Dead code | `REF_R_KF1` no longer exists anywhere. `GalileanPreintegration.predict` has no `bias` parameter |

Remaining hygiene (low): the M2-C25 `test_hint` still ends with *"the statement's
'happens to be orthogonal' wording undersells this"*. That is now a dangling reference,
because the statement no longer contains that wording.

## 3. Ruling on F10: **FIXED** (M2-C25's new wording, judged on its merits)

**Derivation.** Take the rank-39 system: two combined IMU factors plus the KF0 pose and
velocity priors.
- The last 6 rows of each `GalileanCombinedImuFactor` are the random walk b_{k+1} − b_k,
  with Jacobians (−I, +I). A null vector therefore has δ_B0 = δ_B1 = δ_B2 =: δ.
- The priors give δX0 = δV0 = 0.
- In each factor's 9 navigation rows, the Jacobian with respect to (X_{k+1}, V_{k+1}) is
  invertible (rank 30 is full row rank). So (δX_{k+1}, δV_{k+1}) is a linear function of
  (δX_k, δV_k, δ). By induction the pose/velocity part is M δ, with the KF0 rows of M
  equal to zero.
- If δ = 0, the whole vector is 0. So every nonzero null vector is (M δ; δ, δ, δ) with
  δ ≠ 0. It is never orthogonal to the bias block. In particular its B0 component is
  δ ≠ 0, which is exactly what makes a 6-dim B0 prior remove all six directions.

**Measurement**, in all three of my noise configurations:

| quantity | measured |
|---|---|
| KF0 part of the null basis | ≤ 2.2e-14 |
| \|δ_B1 − δ_B0\|, \|δ_B2 − δ_B0\| | ≤ 1.9e-15 |
| pv = M δ reconstruction | ≤ 3.3e-16 |
| singular values of the B0 part | [0.565, 0.565, 0.485, 0.471, 0.173, 0.173], all > 0 (so δ ↦ null vector is injective), and √3 × these = the claimed cosines |
| singular values of M | [5.52, 5.51, 1.23, 1.12, 0.369, 0.368] (so the pose/velocity part is nonzero for every null vector, which confirms the hint's coordinate-free sentence) |
| no pure-bias null vector | σ_min(A_bias · [I; I; I]) / σ_max = 0.017 / 0.060 / 0.0054 |
| rescaling the gyro-bias units ×10 / ×0.1 | cosines become [0.998 … 0.792] / [0.977 … 0.032]. They move, as stated, but never reach 0, as the structure requires |

The statement is correct. "(M δ; δ, δ, δ)" is schematic, since M has zero KF0 rows, but it
is not wrong.

---

## 4. My mutation sweep: 46 new mutants (Q01–Q46)

None of these is in `tests/test_mutation.py`, and none repeats a pass-3 P-mutant. Each one
is a file-level edit in a fresh scratch copy of `src/`, `tests/`, `orchestrator/`,
`experiments/`, `conftest.py` and `pyproject.toml`. Each anchor string was asserted
unique. Each copy ran the full `tests/test_baselines.py`. The control copy gives 31/31.

**Killed (33):**

| id | mutant | killed by |
|---|---|---|
| Q02 | Q = σ²·dt (not /dt) | c06, c06b, c06c |
| Q03 | accel Q uses σ (not σ²) | c06, c06b, c06c |
| Q05 | `has_noise` ignores σ_a | c06c |
| Q06 | Forster `predict` drops v_i Δt | c10, c23, c24 |
| Q07 | `dt_total = dt` (last step only) | c10, c23, c24 |
| Q09 | Eq. (44) rotation-correction sign | c05 |
| Q10 | `increment_error` sums instead of max | c05 |
| Q11 | cov B_a position block uses ΔR_{k+1} | c06b, c06c |
| Q12 | cov A_vR as −â ΔR_ik dt | c06b, c06c |
| Q13 | accel per-axis density order reversed | c06b, c06c |
| Q14 | `integrate_edge` passes σ_a = σ_g | c06b, c06c |
| Q15 | Cay without the ½ | 10 tests |
| Q17 | Eq. (16) uses ΔR̄ R_i | c09, c13 |
| Q18 | Eq. (19) (R_j + A) → (R_j − A) | c09, c11, c13 |
| Q19 | Eq. (22b) Y′ uses R_i | c11 |
| Q20 | Eq. (21b) R_i not transposed | c10, c10b |
| Q21 | Eq. (21c) A_pa applied to δb_g | c10b |
| Q22 | isotropic precision = 1/max diag | c12 |
| Q23 | `CayleyFactor.xi` offset uses b̄_a | c13 |
| Q24 | factor Jacobians from the reparam (= Exp) variant | c13b |
| Q26 | factor A_va × (1+1e-5) | c13b |
| Q27 | `navtriple` swaps v ↔ p | c03, c23, c24 |
| Q30 | W velocity sign | c18 |
| Q31 | `time_shift` sign | c18 |
| Q33 | recursion C_k taken w.r.t. the measurement (−C_k) | c21 |
| Q34 | `right_composition_model` left-composed | c20 |
| Q35 | component-wise contrast rotation sign | c20 |
| Q37 | `from_gtsam_bias` without the swap | c17 |
| Q38 | `make_params` accel covariance from σ_g | c06 |
| Q40 | time-varying stream phases collapsed to 0 | c05b |
| Q41 | `Edge.duration` = dt·(n−1) | 9 tests |
| Q42 | `so3_log` denominator + 1e-9 | c05, c05b, c13b, c22b |
| Q43 | cov A_pR sign only (v-row kept) | c06b, c06c |

**Survived (13), medium:**

| id | mutant | why it survives |
|---|---|---|
| **Q25** | `CayleyFactor.rotation_residual_affine` passes b̄_a as b̄_g | No test calls the method. At a self-consistent state the correct residual is 1.0e-16, while the mutant's is **4.2e-2**. Every Eq. (22b) check (c11, test_m04) exercises the *free function* only |
| **Q45** | `CayleyFactor.rotation_residual` uses −ξ | Same cause: c09 and c13 exercise `rotation_residual_eq19` and `predicted_rotation`, never the method |
| **Q46** | `CayleyFactor.rotation_residual` uses ΔR̄ᵀ | Same cause |

These three are **M2-F12**. The factor's two rotation-residual methods are the objects M3
and M4 would hand to the QCQP lifting. They are wired correctly today: correct affine
residual = Eq. (19) to 1.9e-16 at a random state. But nothing pins that wiring.

**Survived, low or equivalent (10):**

| id | mutant | classification |
|---|---|---|
| Q01 | `so3_right_jacobian` small-angle sign | dead branch (θ < 1e-8). The K term is ≤ 5e-9 there anyway. Low |
| Q04 | `has_noise` uses ≥ (always true) | equivalent: zero densities propagate to a zero covariance |
| Q08 | Forster J_R by one-sided FD | changes J_R by 7.3e-9 relative, under every bound. FD accuracy, low |
| Q16 | `cayley_vector` small-angle branch ×2 | dead path (θ < 1e-12, reparam contrast only). Low |
| Q28 | integration covariance σ (not σ²) | used only in C25, which is rank-invariant to weighting. Low |
| Q29 | bias-ω random-walk covariance σ (not σ²) | same as Q28. Low |
| Q32 | Gal(3) bias Jacobian by one-sided FD | FD accuracy only. c20/c21 bounds hold. Low |
| Q36 | `gal3_distance(A, B)` → Log(B⁻¹A) | equivalent: Log(X⁻¹) = −Log(X) |
| Q39 | `so3_exp` small-angle K² coefficient | dead at θ > 1e-8. Low |
| Q44 | `so3_log` clamp × (1−1e-15) | below resolution. Equivalent |

---

## 5. Tests and assertions that cannot fail, or that compare the code to itself

Carried over from pass 3 and still true:
1. `test_c10` (a): the self-consistent residual is a tautology (labelled as one).
2. `test_c13`: `J_R.tobytes()` is unchanged, but no code path writes `J_R`. Its
   `predicted_rotation == R_i ΔR̄ B2.cay(J_R δ)` re-states the implementation using
   `B2.cay`.
3. `test_c05`: `allclose(corrected[0], base.dR @ so3_exp(J_R @ db_g))` re-states
   `bias_corrected_increment`.
4. `test_c08`: `closed >= 1` is `1 + x²/4 ≥ 1`.
5. `test_c20`: `J[9,:] = 0` holds for any bias-only perturbation.
6. `test_c25`: ranks 30 and 39 are full row rank (labelled as such).
7. `test_c06`: the symmetry / PSD / monotone assertions are structural. Only its GTSAM
   magnitude line has teeth (it killed Q38).
8. `test_c01`: `R0` vs `expm(T.hat(...))` shares `T.hat`.
9. `test_mutation.py::test_m04…` asserts a genuine no-op, by design.

New in this pass:

10. `test_c06c`, accel-only: `not np.any(ref[0:3,:])` asserts a property of the
    *reference*, which is zero by construction. Only the `pim.cov` half of that line can
    fail.
11. **No test calls `CayleyFactor.rotation_residual` or
    `CayleyFactor.rotation_residual_affine`** (M2-F12; Q25, Q45, Q46 survive). Also,
    `CayleyFactor.rotation_residual_truncated` is called by no test; only the free
    function is pinned (P29's fix).

---

## 6. Verdicts per claim

| id | verdict | attack |
|---|---|---|
| C01 | CONFIRMED | Retyped constants. Streams `array_equal`. Q40 and Q41 die |
| C02 | CONFIRMED | quad and 5x5 expm vs closed form ≤ 8.9e-16. Sec. 0.4 literals ≤ 3.3e-15. Q41 is killed by c02 |
| C03 | CONFIRMED | gtsam `predict` vs my chain 9.5e-15. Q27 killed |
| C04 | CONFIRMED | Own Euler: ratios v 0.5000, p 0.5009→0.5001 |
| C05 | CONFIRMED | Own FD J: per-component ratios 100.0–100.14. Q09 and Q10 killed |
| C06 | CONFIRMED | Own Eq. (63) vs own noise-FD 2.7e-9. Structure holds at every step. Chart numbers reproduced. Q02, Q03, Q11–Q14 and Q43 killed |
| C07 | CONFIRMED | 2.9e-15. Right 5.0e-10, left 0.384. Eq. (4) derived analytically to 3.3e-16 |
| C08 | CONFIRMED | 2.7e-15, minimum 1.0000004 |
| C09 | CONFIRMED | 3.1e-16 on, ≥ 1.5e-4 off. Q17 and Q18 killed |
| C10 | CONFIRMED | 2.7602e-3 / 2.1709e-3 at ground truth. Q20 and Q21 killed |
| C11 | CONFIRMED | 2.7e-15. Q19 killed (but see F12 for the factor method) |
| C12 | CONFIRMED | 6.2e-13 / 50.3. Q22 killed |
| C13 | CONFIRMED | Truncation gap 5.0e-5 ≥ 1e-6. The factor's J_R is Cayley's (3.1e-10, vs Exp 2.3e-7). Q23 and Q24 killed |
| C14 | CONFIRMED | gtsam directly |
| C15 | CONFIRMED | Hat and Expmap block semantics. Expmap = expm(Hat) to 5.6e-17 |
| C16 | CONFIRMED | 0.10763 vs `localCoordinates`, 0.0 vs `Logmap` |
| C17 | CONFIRMED | gtsam directly. Q37 killed |
| C18 | CONFIRMED | Own matrices: W 1.6e-15, F2 18.1485 = \|g\|Δt t_i. Q30 and Q31 killed |
| C19 | CONFIRMED | ≤ 2.0e-14; Tangent 2.17e-3; right 1.1e-16 vs left 6.2e-4 |
| C20 | CONFIRMED | Right ≤ 4.6e-10. Component-wise sup 1.419e-2 / 1.640e-2. Q34 and Q35 killed |
| C21 | CONFIRMED | Own recursion 3.5e-9 relative. Q33 killed |
| C22 | CONFIRMED | 6.167e-8 / 4.376e-8 / 2.9965e-8. Deficit = ‖ω‖³T dt²/12 to 6 digits |
| C23 | CONFIRMED | 1.2867e-3 / 1.2031e-3. Ratios in band for B1 and B2 |
| C24 | CONFIRMED | B3 ≤ 1.6e-14. B1 rotation 1.7e-14. B2 rotation 3.0e-8 |
| C25 | CONFIRMED | Own graph, three weightings: 30 / 39 / 45, identical cosines. The new structural sentence was derived and measured (§3) |
| **C27** | **REFUTED** | See below |

**M2-C27: REFUTED.** Attack: integrate each fixture edge with my own Euler sums under Exp
and under Cay, then compare dv and dp. `src` reproduces my numbers to ≤ 5e-14.

| | edge 0 | edge 1 |
|---|---|---|
| \|dv_B2 − dv_B1\|_∞ | **4.529e-8** | **1.028e-7** |
| \|dp_B2 − dp_B1\|_∞ | **7.480e-9** | **1.693e-8** |
| \|Log(dR_B1ᵀ dR_B2)\| | 2.326e-8 | 5.518e-8 |

These exceed the claimed 1e-12 by 3.9 to 5.0 orders of magnitude. The gap is O(dt²): at
n = 100, 200, 400, 800 and 1600, edge-0 \|Δdv\| is 1.80e-7, 4.53e-8, 1.14e-8, 2.84e-9 and
7.11e-10 (ratio 0.25 per halving).

The stated mechanism is false. dv and dp are accumulated from *each integrator's own*
running ΔR_ik. That rotation is the product of Exp steps in B1 and of Cay steps in B2, so
it differs by k·θ³/12 at step k. It is not a "same frozen ΔR_ik". The retraction therefore
changes dv and dp as well as dR.

The claim also contradicts M2-C22, which the yaml records as CONFIRMED with a KF2 velocity
gap of 6.2e-8.

**Counts: 25 CONFIRMED, 1 REFUTED (C27), 0 UNTESTABLE.**

---

## 7. Recommendations

1. **F12 (medium):** test the factor-level `CayleyFactor.rotation_residual` and
   `rotation_residual_affine` directly. For example:
   - it vanishes at `predicted_rotation(R_i, b_g)`;
   - it equals the free Eq. (19) / Eq. (22b) functions under the factor's own
     ξ = J_R(b_g − b̄_g);
   - it is nonzero for a perturbed R_j.

   This kills Q25, Q45 and Q46.
2. Do not enter M2-C27 into the yaml. The true statement is that B1 and B2 increments
   differ in dv and dp by O(dt²) (4.5e-8 / 1.0e-7 in dv), which is what M2-C22 already
   implies.
3. Drop the dangling "'happens to be orthogonal' wording" reference from the M2-C25
   `test_hint`.
