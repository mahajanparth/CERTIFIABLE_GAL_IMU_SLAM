# Stage-B verification, pass 6: M2 baselines (`stage-b/m2-baselines` @ `b4e8420`)

Sixth independent adversarial pass. It follows `3ce49be` (WIP) and `3c28d06`, which say they
fix pass-5 finding M2-F13 (CayleyFactor.dRbar pinned by no test) by adding `test_c13d` and
replaying R01/R02/R04/R05/R10 in `tests/test_mutation.py`, and `b4e8420`, which edits the
M2-C22 statement to name the metric of each quoted rotation gap.

Scope: the F13 ruling; every claim in `docs/claims/M2.yaml` (C01-C25); one claim that exists
only in the pass brief, **M2-C29**; a field-by-field mutation sweep of the three baseline
constructors; a replay of earlier mutants as real file edits.

Rules kept: I did not read `docs/02_*` or any derivation narrative, any `seeds*.yaml`, or
anything under `.cgis_SEEDS_BACKUP*`. All scratch code and mutant copies lived in the session
scratchpad, outside the worktree. I did not edit `src/`, `tests/` or `docs/claims/`.

On the clean tree at `b4e8420` (plus this report), the full suite is **112 passed**;
`tests/test_baselines.py` alone is 33/33. `git status` was clean before and after the sweep.

## 1. Independent references (no `src` imports)

Scratch module `ind/core.py`: constants retyped from the M2-C01 text; `Exp = expm o hat`,
`Log = logm`; `Cay = inv(I - phi^/2)(I + phi^/2)`; `J_Cay` typed from Eq. (4); the Eq. (9)
Euler sums with a pluggable retraction; the exact increment from ONE 5x5 `expm`; central-FD
bias Jacobians at eps = 1e-5; my own Eq. (19) typed as `R_j - A - 1/2 R_j xi^ - 1/2 A xi^`.
`ind/c06.py`: a Rodrigues-based noisy Euler integrator and a 200-step x 6-direction central FD
of it (eps = 1e-5) for the covariance. `ind/claims2.py`: my own 5x5 Gal(3) matrices, my own
ZOH step `V_k = expm(dt [[w^, a, 0],[0,0,1],[0,0,0]])`, my own FD Jacobians, and my own
factor graph (prior sigmas 5e-2 / 2e-1, GPS sigma 0.5, densities 3e-4 / 2e-2, random-walk
3e-5 / 5e-4 -- all different from the test's). `src` is imported only as the system under
test.

---

## 0. Summary

| | |
|---|---|
| **F13** | **PARTIALLY FIXED.** `test_c13d` is independent (a retyped Cayley integrator and retyped Rodrigues, with no `src` maths) and tight on both sides. It kills R01, R02, R04, R05 and R10 as real file edits, along with every `dRbar` value mutant down to Cay(1e-11) (S01). But every `from_edge` call in the suite uses `bbar = BIAS_TRUE`. As a result, a `from_edge` whose `dRbar` (S48), whose increments (S46), or whose entire input (S47) ignore `bbar` and use the hard-coded fixture bias **passes all 112 tests**. `dRbar` is pinned at one point, not as a function of its input. That is new finding **M2-F14 (medium)** |
| **C22 (edited statement)** | **CONFIRMED.** Both metric labels are right: KF2 max-abs gap **2.99648e-8** (the same gap is 2.810e-8 in \|\|Log\|\|∞ and 3.510e-8 in \|\|Log\|\|₂); per-edge \|\|Log\|\|₂ **2.32574e-8 / 5.51753e-8** against \|\|ω\|\|³T dt²/12 = 2.32574e-8 / 5.51753e-8 (relative −2.3e-7 / −3.4e-7). "Exactly" holds to leading order only, well inside the hint's 1e-3 |
| Verdicts | **25 CONFIRMED (C01–C25), 1 REFUTED (C29), 0 UNTESTABLE** |
| **C29** | **REFUTED.** J_Cayᵀ J_Cay = ((1+s)I − φφᵀ/4)/(1+s)², with s = \|φ\|²/4. Its singular values are 1/√(1+s) (twice, perpendicular to φ) and 1/(1+s) (along φ), so J_Cay is orthogonal only at φ = 0. Measured max \|J_Cayᵀ J_Cay − I\| = **0.997** (5000 draws), **0.0754** at \|φ\| = 0.4, and **5.0e-7** at the fixture's own per-step φ (\|φ\| = 1.12e-3), 5e5 times the claimed 1e-12. The frozen factor J_R has singular values 0.4990–0.5000 (\|J_Rᵀ J_R − I\| = 0.751), so it does rescale bias perturbations, by about ½ |
| Mutation sweep | **51 new file-level mutants (S01–S51).** Against `test_baselines.py`: 32 killed, 19 survived. 8 of the 19 die in the full suite, but only on `test_pass5_replay_is_faithful`, which compares `src` against a copy of `src`'s own `from_edge` (§4). **7 survivors are medium** (S12, S13, S35, S36, S46, S47, S48); 12 are low or equivalent |
| Regression replay | **20 earlier mutants** replayed as real file edits: M02, M05, M08, M09, M12, M15, M17, N04, P17, P29, P33, P35, Q25, Q45, Q46, R01, R02, R04, R05, R10. **20/20 killed.** Nothing regressed |
| New findings | **M2-F14 (medium):** the `bbar`-dependence of `CayleyFactor.from_edge` is pinned by no test, and neither is `bbar[3:6]` (S12, S13, S46–S48). **M2-F15 (medium, noise model of Baseline 3):** the `sigma_int`, `sigma_bg` and `sigma_ba` plumbing in `make_params` is pinned by no test (S35, S36) |

---

## 2. Per-claim measurements

All numbers below are mine (`ind/claims1.py`, `ind/claims2.py`, `ind/c06.py`).

### 2.1 Baselines 1 and 2, the fixture, and the Cayley algebra

| claim | my result |
|---|---|
| C01 | My retyped streams and constants are `array_equal` to `src` on both edges. dt, T_EDGE and N are identical |
| C02 | M1 vs `quad_vec` (1e-14): 7.6e-17 / 5.6e-17. M2 vs nested quad: 1.8e-16 / 9.7e-17. Chained 5x5-expm states vs the `REF_*` literals: KF1 v, p 1.6e-15 / 2.0e-15; KF2 R, v, p 2.2e-16 / 2.6e-15 / 3.3e-15 |
| C04 | Own Euler, n = 100..1600. Ratios: v 0.50002, 0.50001, 0.50000, 0.50000; p 0.50086, 0.50043, 0.50022, 0.50011. `src` B1 vs mine: 2.2e-14 / 6.3e-15 / 9.6e-16 |
| C05 | Own FD J (eps 1e-5), random unit direction. Error ratio between 1e-2 and 1e-3: (R, v, p) 100.09 / 99.98 / 99.98 on edge 0 and 100.07 / 100.06 / 100.00 on edge 1. `src` J vs mine ≤ 2.9e-8 |
| C06 | My noisy-integrator FD (anisotropic) vs `src` Eq. (63): 7.3e-9 / 9.9e-9 relative per block. The claim's 2.7e-9 is the test's own reference; my FD at eps = 1e-5 is noise-limited at about 1e-8, against the test bound of 1e-6. Asymmetry ≤ 1.7e-20, min eig −4e-23, trace monotone. GTSAM, raw R–v block: iso 0.1704 / 0.1945; aniso 0.1122 / 0.2049. After M = diag(I, ΔRᵀ, ΔRᵀ): rot 2.3e-14 / 3.5e-15; R–v 5.04e-3; R–p 7.53e-3. Ratios 0.4998–0.5009 |
| C07 | Cay(2tan(θ/2)a) vs Exp: 2.8e-15 over 2000 draws. Right FD (e = 1e-7): 5.0e-8. Left: **0.384** (≥ 1e-1; the hint's 2.1 is a different draw). `src` Eq. (4) vs mine: 0 |
| C08 | 4.4e-15 over 5000 draws, \|ξ\| ≤ 5. min(1 + \|ξ\|²/4) = 1.00000002 |
| C09 | On-manifold 2.6e-16. Perturbed by Exp(1e-3 u): min 5.9e-4 |
| C10 | Ground truth: \|r^v\| 2.1709e-3 / **2.7603e-3**, \|r^p\| 5.38e-4 / 6.81e-4. Equal to \|Δv̄ − Δv_exact\| to 5e-16. `src` factor vs my residual ≤ 5.6e-14. Self-consistent ≤ 5.6e-14 (tautology) |
| C11 | 4.4e-16 over 2000 draws, with random (non-orthogonal) J_R |
| C12 | Isotropic, with scipy-uniform R and unit-scale r: 1.4e-14. Weighted diag(1, 2, 5): 219 (≥ 1). τ = 3/tr. Note: with my `expm`-built R at angles up to 5 rad, the isotropic gap rose to 1.3e-12. That is the 1.5e-13 orthogonality defect of my `expm`, not the claim |
| C13 | J_R is bitwise unchanged after `predicted_rotation`. Truncation gap 4.79e-5 at \|ξ\| = 1e-2. Factor J_R vs my Cayley FD: 3.1e-10 / 4.5e-10 relative. Exp FD: 2.3e-7 / 3.6e-7 |
| C22 | See §0. KF2 B1 − B2: v 6.1672e-8, p 4.3763e-8. dt² ratios of the KF2 gap: 0.25000, 0.25000, 0.25000, 0.25002 |
| C23 | KF2 vs exact: B1 v 1.28671e-3, p 1.20308e-3; B2 v 1.28674e-3, p 1.20313e-3. Ratios: B1 v 0.50028→0.50003, p 0.49999→0.50000; B2 v 0.50026→0.50003, p 0.49998→0.50000 |
| C24 | Rotations vs exact at KF2: B1 1.7e-14, B2 2.996e-8; KF1 B1 9.4e-15 |

### 2.2 Baseline 3 and GTSAM facts (my own 5x5 matrices, ZOH, FD and graph)

| claim | my result |
|---|---|
| C03 | `predict` vs my chain from a random NavState: 6.7e-15 / 4.4e-16 / 0 (edge 0) and 7.7e-15 / 4.4e-16 / 2.2e-16 (edge 1) |
| C14 | col 3 = [10, 20, 30], col 4 = [1, 2, 3], M[3,4] = 7. `position` = `translation` = [1, 2, 3]; `velocity` = [10, 20, 30]. The `Gal3(M)` round-trip holds |
| C15 | For all 10 k, Hat(e_k) is nonzero only in the stated block. Expmap(1e-6 e_k) moves only R (k = 0..2), v (3..5), p (6..8) or t (9). Expmap vs expm(Hat): 1.1e-16 |
| C16 | Remapped log vs `localCoordinates`: 0.1008 / 0.1678 / **0.10763** (0→1, 1→2, 0→2). Vs `NavState.Logmap`: 0.0, 0.0, 0.0 |
| C17 | As claimed. `biasHatVector` = [b_a; b_g]. The adapters round-trip, and one swap is not the identity |
| C18 | My 5x5 matrices. W-form 2.2e-16 / 1.6e-15, t_j = 4.2 at t_i = 3.7. F2: 2.2e-16 at t_i = 0 and **18.1485 = 9.81·0.5·3.7** at t_i = 3.7. Swapped 2.45. `src` W vs mine: 0 |
| C19 | Galilean vs exact: 2.0e-14 / 1.7e-14. Tangent dv: 2.17e-3 / 2.76e-3. Time-varying stream with my ZOH V_k: right 1.1e-16, left 6.2e-4 |
| C20 | My FD J at eps = 1e-7. Right model ≤ 4.4e-10 / 2.8e-10 over 30 directions × 4 magnitudes. Component-wise at 5e-2: **1.07e-2 / 1.46e-2**. J[0:3,0:3] = J[9,:] = 0 |
| C21 | My ZOH V_k, my FD C_k, and GTSAM `AdjointMap`, vs FD of my own full re-integration: 8.1e-10 / 2.7e-9 / 2.6e-9 relative at steps 10, 20 and 40 |
| C25 | My own graph with every sigma different from the test's. Ranks 30 (30x45), 39 (39x45), 45 (45x45); a B0 bias prior also gives 45. Cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]. Pose/velocity energy 2.5371. Unwhitened IMU residuals 1.7e-14 / 2.0e-14. Graph error 2.1e-23 |

### 2.3 M2-C29 (brief only): REFUTED

Attack: write out J = (I − P)/(1+s), with P = φ^/2 and s = \|φ\|²/4. Then

  JᵀJ = (I + P)(I − P)/(1+s)² = (I − P²)/(1+s)² = ((1+s)I − φφᵀ/4)/(1+s)².

So the singular values are 1/√(1+s) (twice) and 1/(1+s).

| \|φ\| | max \|JᵀJ − I\| | singular values |
|---|---|---|
| 1e-3 | 5.0e-7 | |
| 1.12e-3 (fixture per-step φ) | 5.0e-7 | |
| 0.1 | 4.98e-3 | 0.99875, 0.99875, 0.99751 |
| 0.4 | **7.54e-2** | 0.98058, 0.98058, 0.96154 (= 1/√1.04, 1/1.04) |
| 2 | 0.75 | 0.7071, 0.7071, 0.5 |
| 5000 random draws, \|φ\| ≲ 5 | **0.997** | |

J_Cay is orthogonal only at φ = 0. It is a contraction, not an isometry, so the Cayley chart is not locally isometric. It is isometric to first order only at the origin.

The rider about the factor fails on its own terms too. The frozen `CayleyFactor.J_R` has singular values 0.49999984 / 0.49895997 (×2) on edge 0, with \|J_Rᵀ J_R − I\| = 0.751. So ξ = J_R δb_g scales bias perturbations by about ½, i.e. by about T = 0.5 s, as expected for a short edge where J_R ≈ −T·I. Do not enter C29 into the yaml. The true statement is the singular-value formula above.

---

## 3. Ruling on F13: **PARTIALLY FIXED**, and new finding **M2-F14 (medium)**

### What is fixed

- `test_c13d` (a) pins `from_edge(edge, BIAS_TRUE).dRbar` against `_indep_cayley_increment`. That helper is retyped and uses `_indep_cay`, the closed polynomial form I + 4/(4+\|φ\|²)(K + K²/2), with no linear solve and no `src` maths. The bound is 1e-12. I measure 2.198e-14 / 2.187e-14 with my own inverse-form Cayley integrator. The Exp integrator sits at 1.872e-8 / 4.049e-8, more than 10⁴ times outside the bound.
- `test_c13d` (b) is a genuine end-to-end band. My \|r19\|_F/√2 at the closed-form keyframes is 2.325744e-8 / 5.517531e-8, against the prediction 2.325744e-8 / 5.517533e-8 (relative −1.9e-7 / −3.4e-7, inside the 1e-4 band). With xi = 0 it does not depend on J_R or ξ, so it tests the stored `dRbar` together with the residual method.
- R01, R02, R04, R05 and R10 are dead as **real file edits** (not monkeypatches). Each dies on c13d, and R01–R10 die only there. S01 (`dRbar · Cay(1e-11 e₃)`) also dies.
- The replay-fidelity guard `test_pass5_replay_is_faithful` does what it says: the monkeypatched replay equals `src` field for field.

### What is not fixed

Every `from_edge` call in `tests/` (8 call sites) passes `T.BIAS_TRUE`, on the default 200-step edges. The *map* bbar ↦ (ΔR̄, Δv̄, Δp̄, bbar) is therefore unpinned, and so is `dRbar` as a function of its input:

| mutant (as a real edit; full suite 112/112 green) | effect at bbar = 0 (edge 0 / 1) |
|---|---|
| **S48** `dRbar` integrated at the hard-coded fixture gyro bias | ΔR̄ wrong by **7.9e-3** |
| **S46** all increments integrated at the hard-coded fixture bias | ΔR̄ 7.9e-3, Δv̄ **4.4e-2 / 4.7e-2**, Δp̄ 1.1e-2 |
| **S47** `from_edge` replaces `bbar` by the fixture literal | the above, plus `bbar` and every J taken at the wrong point |

At bbar = true + [0.02, −0.01, 0.015, 0.1, −0.1, 0.05], the errors are ΔR̄ 1.0e-2, Δv̄ 7.5e-2 and Δp̄ 1.7e-2. These factors leak the fixture's true bias. In a noise-free comparison they look *better* than a correct Rai factor, which is exactly what would mislead an M3/M4 baseline comparison run at an estimated bbar.

**Suggested kill:** run c13d (a), and the c13b field checks, at a second bbar ≠ BIAS_TRUE (for example bbar = 0) against the same retyped references. `_indep_cayley_increment` already takes `bias`.

The same finding covers **S12/S13**. `CayleyFactor.bbar[3:6]` is read by no `src` method; only `xi` uses `bbar[0:3]`, which c13 pins. So zeroing or negating the stored accel nominal survives every *independent* check. It dies only on the copy-of-`src` guard (§4). A consumer that forms δb = b − f.bbar for Eq. (21b)/(21c), the natural way to call `velocity_residual`, would see \|r^v\| at ground truth and true bias go from 2.17e-3 / 2.76e-3 to **8.9e-2 / 8.6e-2**, and \|r^p\| from 5.4e-4 to 2.2e-2.

---

## 4. Mutation sweep: 51 new mutants (S01–S51)

Method:
- each mutant is a file-level string edit, with a unique-anchor assertion, in a fresh scratch copy of the worktree (no `.git`);
- each copy ran `tests/test_baselines.py`, and every survivor was then rerun on the full `tests/`;
- the control copy gives 112/112.

"Killed" means killed by a claim test in `test_baselines.py`. "Guard-only" means killed only by `test_mutation.py::test_pass5_replay_is_faithful`, which compares `src`'s `from_edge` with a retyped *copy of `src`'s own `from_edge`* via `array_equal`. That is a change detector, not an independent reference. It also fires on true no-ops (S03, S11).

**Killed by claim tests (32):**

| id | mutant | killed by |
|---|---|---|
| S01 | `dRbar · Cay(1e-11 e₃)` | c13d |
| S04 | `dvbar × (1+1e-5)` | c10, c10b |
| S05 | `dvbar + 1e-9` | c10 |
| S06 | `dpbar × (1+1e-4)` | c10, c10b |
| S07 | `dpbar + 1e-9` | c10 |
| S08 | `dpbar` from the Exp integrator | c10, c10b |
| S09 | `dvbar` integrated at b_a = 0 | c10, c10b |
| S10 | `Dt × (1+1e-9)` | c10 |
| S14 | `A_vg ↔ A_va` | c10b, c13b |
| S15 | `A_paᵀ` | c10b, c13b |
| S17 | `J_R × (1+2e-8)` | c13b |
| S18 | `A_va × (1+1e-6)` | c13b |
| S19 | `A_va × (1+2e-7)` | c13b |
| S23 | Eq. (4) denominator /4 → /2 | c07 |
| S24 | `cayley_vector` tan → sin | c07, c22, c23, c24 |
| S25 | `isotropic_precision` 3/tr → 1/tr | c12 |
| S27 | noise Q = σ² dt instead of σ²/dt | c06, c06b, c06c |
| S28 | covariance `Ba` position block drops ½ | c06b, c06c |
| S29 | `so3_right_jacobian` sign of the K term | c06b, c06c |
| S30 | `bias_jacobians_fd` J_R in the left chart | c05, c05b, c13b |
| S32 | `dt_total += dt(1+1e-12)` | c10 |
| S33 | initial covariance 1e-30 I | c06c |
| S34 | Forster `predict` drops R_i on Δp | c10, c23, c24 |
| S37 | GTSAM gyro covariance not squared | c06 |
| S38 | Galilean `bias_hat` accel block zeroed | c06b, c17, c19, c20, c23, c24 |
| S40 | `increment_gal3` clock = 0 | c18, c19, c21 |
| S41 | Galilean `dt_total × (1+1e-6)` | c03, c18, c19 |
| S42 | component-wise contrast rotates on the left | c20 |
| S44 | non-combined PIM → `PreintegratedImuMeasurements` (Tangent) | c06b, c19, c20, c23, c24 |
| S49 | integrator gyro bias hard-coded to the fixture | c05, c05b, c10b, c13, c13b, c13c |
| S50 | integrator accel bias hard-coded to the fixture | c13b |
| S51 | `to_gtsam_bias` accel block hard-coded to the fixture | c20 |

**Survived, medium (7):**

| id | mutant | full suite | measured effect |
|---|---|---|---|
| **S46** | `from_edge` increments at the hard-coded fixture bias | survives 112/112 | M2-F14: at bbar = 0, ΔR̄ off by 7.9e-3, Δv̄ by 4.4e-2 |
| **S47** | `from_edge` ignores `bbar` (fixture literal) | survives 112/112 | M2-F14 |
| **S48** | `dRbar` at the hard-coded fixture gyro bias | survives 112/112 | M2-F14: ΔR̄ off by 7.9e-3 at bbar = 0 |
| **S12** | `bbar[3:6]` stored as 0 | guard-only | M2-F14: a δb consumer's r^v goes 2.8e-3 → ~9e-2 |
| **S13** | `bbar[3:6]` negated | guard-only | same as S12 (8.9e-2 / 8.6e-2) |
| **S35** | `setIntegrationCovariance(sigma_int)` not squared | survives 112/112 | M2-F15: Baseline 3 combined covariance position block +12% (1.0% of the whole) at the c25 settings. Only c25 sets sigma_int ≠ 0, and rank is weight-invariant |
| **S36** | bias random-walk covariances gyro ↔ accel | survives 112/112 | M2-F15: bias-block diagonal 5e-9 / 5e-11 becomes 5e-11 / 5e-9, a 100x mis-weighting of both bias random walks |

S35 and S36 are outside the noise-free M2 claims. They are medium because Baseline 3's noise model is what any downstream weighted comparison consumes. **Suggested kill:** assert the 15x15 combined `preintMeasCov` bias blocks equal diag(σ_ba², σ_bg²)·Δt (I measured exactly 5e-9 / 5e-11), plus one sigma_int ≠ 0 position-block check against an independent value.

**Survived, low or equivalent (12):**

| id | mutant | full suite | classification |
|---|---|---|---|
| S02 | `dRbar · Cay(3e-13 e₃)` | guard-only | below the 1e-12 pin; 3e-13 rad against a 2e-8 deficit. Low |
| S03 | `dRbar = ΔR ΔRᵀ ΔR` | guard-only | equivalent (ulp). Shows the guard fires on no-ops |
| S11 | `Dt = edge.duration` | guard-only | equivalent (3.3e-16) |
| S16 | `A_pg × (1+5e-8)` | guard-only | below c13b's 1e-7; FD-noise level |
| S20 | stored `variant = reparam` | guard-only | dead field (as R38) |
| S21 | factor FD step 1e-6 → 1e-3 | guard-only | an *improvement*: A_va error 4.9e-9 → 4.8e-10 |
| S22 | factor preintegration with noise densities on | survives | equivalent for the mean; the factor stores no covariance |
| S26 | `cay` as (I+P)(I−P)⁻¹ | survives | equivalent (the factors commute) |
| S31 | Forster `FD_EPS` 1e-6 → 1e-4 | survives | low; J accuracy unchanged at the 1e-8 level |
| S39 | Galilean `predict` uses `pim.biasHat()` | survives | equivalent |
| S43 | `from_gtsam_bias` no-op arithmetic | survives | equivalent |
| S45 | always `PreintegrationCombinedParams` | survives | equivalent (a subclass; the extra fields are unused) |

### 4.1 Regression replay as real file edits: 20/20 killed

| id | mutant | killed by |
|---|---|---|
| M02 | fold rotation first (exact replay) | c04, c05b, c13b |
| M05 | Γ sign in R̆ | c11, c13c, c13d |
| M08 | W drops t_i | c18 |
| M09 | forward adjoint | c21 |
| M12 | Cay → Exp | c05b, c13b, c13d, c22, c22b, c23, c24 |
| M15 | `xi` drops b̄_g | c13, c13c, c13d |
| M17 | Eq. (19) ξ sign | c09, c11, c13, c13c |
| N04 | Eq. (21b) A_vg sign | c10b |
| P17 | `has_noise` or → and | c06c |
| P29 | truncation sign | c13, c13c |
| P33 | `weighted_square` unsquared | c12 |
| P35 | factor Jacobians from Forster | c13b |
| Q25 | affine method takes b̄_a | c13c, c13d |
| Q45 | method uses −ξ | c13c |
| Q46 | method uses ΔR̄ᵀ | c13c, c13d |
| R01 | `dRbar = pim.dR.T` | c13d |
| R02 | `dRbar` from the Exp integrator | c13d |
| R04 | `dRbar` at b_g = 0 | c13d |
| R05 | `dRbar` from `VARIANT_REPARAM` | c13d |
| R10 | `dRbar · Cay(1e-6 e₁)` | c13d |

A first, mis-transcribed M02 (fold first *and* keep the later fold, so a double rotation) was also killed, by 10 tests.

---

## 5. Tests and assertions that cannot fail, or that compare the code to itself

Items 1–12 of pass 5 §4 still hold:
- the c10 (a) tautology;
- c13's `J_R.tobytes()` and its lower-bound-only gap;
- c05's restatement of `bias_corrected_increment`;
- c08's `1 + x²/4 ≥ 1`;
- c20's `J[9,:] = 0`;
- c25's full-row-rank 30 / 39;
- c06's structural checks;
- c01's shared `T.hat`;
- `test_m04`'s deliberate no-op;
- c13c's reference-vs-reference identity, RNG preconditions and edge-factor self-reads.

New in this pass:

13. **`test_c13d`**: `assert predicted >= 1e-8` is a function of fixture constants only. `src` cannot fail it; only a fixture edit can.
14. **`test_pass5_replay_is_faithful`** compares `src`'s `from_edge` against a *copy* of that same code. It is a change detector, not a reference. It fires on algebraic no-ops (S03, S11) and on improvements (S21). It is the *only* thing that sees `bbar[3:6]` (S12, S13), and it calls `from_edge` at `BIAS_TRUE` only, so it cannot see S46–S48. Its `array_equal` comparison will also break on any legitimate refactor of `from_edge`.
15. **Every `from_edge` field check** (c10, c10b, c13, c13b, c13c, c13d) is evaluated at the single input `bbar = BIAS_TRUE`, where "use the input" and "use the fixture constant" coincide. So no check can tell them apart (M2-F14).
16. **`make_params`' `sigma_int`, `sigma_bg` and `sigma_ba`** are exercised only by c25, whose rank and nullspace assertions are invariant to positive reweighting. So no test can fail on any value of them (M2-F15).

---

## 6. Verdicts per claim

| id | verdict | attack |
|---|---|---|
| C01 | CONFIRMED | Retyped constants; streams `array_equal` |
| C02 | CONFIRMED | quad ≤ 1.8e-16; 5x5 expm vs literals ≤ 3.3e-15 |
| C03 | CONFIRMED | Own chain vs `predict` ≤ 7.7e-15 |
| C04 | CONFIRMED | Own Euler; ratios v 0.5000, p 0.5009→0.5001 |
| C05 | CONFIRMED | Own FD J; ratios 99.98–100.09. M02, S30 killed |
| C06 | CONFIRMED | Own noise FD vs `src` ≤ 9.9e-9 (FD-limited); chart numbers reproduced. S27–S29, S33 killed |
| C07 | CONFIRMED | 2.8e-15; right 5.0e-8, left 0.384. S23 killed |
| C08 | CONFIRMED | 4.4e-15; min 1.00000002 |
| C09 | CONFIRMED | 2.6e-16 on-manifold, ≥ 5.9e-4 off. M17 killed |
| C10 | CONFIRMED | 2.7603e-3 / 2.1709e-3 at ground truth. S04–S10, N04 killed |
| C11 | CONFIRMED | 4.4e-16. M05 killed |
| C12 | CONFIRMED | 1.4e-14 / 219. S25, P33 killed |
| C13 | CONFIRMED | Gap 4.79e-5; J_R Cayley (3e-10 vs 2.3e-7). The claim is about freezing at bbar, and it holds; F14 is about which bbar |
| C14 | CONFIRMED | GTSAM directly |
| C15 | CONFIRMED | All 10 blocks; Expmap = expm(Hat) 1.1e-16 |
| C16 | CONFIRMED | 0.10763 vs `localCoordinates`, 0.0 vs `Logmap` |
| C17 | CONFIRMED | GTSAM directly; S51 killed |
| C18 | CONFIRMED | Own 5x5: W 1.6e-15; F2 18.1485 = \|g\|Δt t_i; M08 killed |
| C19 | CONFIRMED | 2.0e-14; Tangent 2.17e-3; right 1.1e-16 vs left 6.2e-4. S44 killed |
| C20 | CONFIRMED | Right ≤ 4.4e-10; component-wise 1.07e-2 / 1.46e-2. S42 killed |
| C21 | CONFIRMED | 2.6e-9 relative at step 40. M09 killed |
| C22 | CONFIRMED | Edited statement checked metric by metric (§0); M12 killed |
| C23 | CONFIRMED | 1.2867e-3 / 1.2031e-3; ratios in band for B1 and B2 |
| C24 | CONFIRMED | B3 ≤ 2.0e-14; B1 R 1.7e-14; B2 R 2.996e-8 |
| C25 | CONFIRMED | Own graph, every sigma different: 30 / 39 / 45 (+B0 45); cosines identical to 1e-6; pv 2.5371 |
| **C29** | **REFUTED** | Closed-form JᵀJ; 0.997 max, 0.0754 at \|φ\| = 0.4, 5.0e-7 at the fixture φ; factor J_R singular values ≈ ½ |

**Counts: 25 CONFIRMED, 1 REFUTED (C29), 0 UNTESTABLE.**

---

## 7. Recommendations

1. **F14 (medium):** call `from_edge` at a second nominal bias (for example bbar = 0, plus one generic bbar with bbar_a ≠ bbar_g). Pin `dRbar`, `dvbar`, `dpbar`, `bbar` (all six) and `J_R`/`A_*` against the retyped references, which already take `bias`. This kills S46, S47 and S48. Pin `bbar[3:6]` directly, or remove it from the factor if it is meant to be unused, to kill S12 and S13 without relying on the copy guard.
2. **F15 (medium, Baseline 3 noise model):** assert the combined-PIM covariance bias blocks equal diag(σ_ba², σ_bg²)·Δt, and check one sigma_int ≠ 0 case. This kills S35 and S36.
3. Do not enter M2-C29. If a statement is wanted: σ(J_Cay) = {1/√(1+s), 1/√(1+s), 1/(1+s)} with s = \|φ\|²/4.
4. Consider relaxing `test_pass5_replay_is_faithful` from `array_equal` on every field to the fields the replay actually varies, or label it a change detector. As written it fails on no-ops and improvements.
