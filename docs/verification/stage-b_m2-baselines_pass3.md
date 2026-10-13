# Stage-B verification, pass 3: M2 baselines (`stage-b/m2-baselines` @ `e380b18`)

Third independent adversarial pass. It follows remediation commit `e380b18`, which says it
fixes pass-2 findings F07–F10 and kills 38 of the 39 pass-2 mutants (the one left is a
literal no-op). The scope was every claim in `docs/claims/M2.yaml` (C01–C25), plus one
extra claim that exists only in the pass brief (**M2-C26**). Each claim was judged by
computation; none was assumed true.

I did not read `docs/02_*`, any derivation narrative, any `seeds*.yaml`, or anything under
`.cgis_SEEDS_BACKUP*`. All scratch code lives outside the worktree. `git status` was
clean before and after the mutation sweep.

Suite at HEAD on the clean tree: **88 passed** (`tests/`: 29 in `test_baselines.py`, the
rest in `test_mutation.py` / `test_orchestrator.py`).

---

## 0. Summary

| | |
|---|---|
| F07 (covariance) | **FIXED** |
| F08 (order errors invisible to constant inputs) | **FIXED** |
| F09 (Eqs. 21b/21c at db ≠ 0; dp-only fold) | **FIXED** |
| F10 (record hygiene) | **PARTIALLY FIXED**: the M2-C25 *statement* still says "happens to be orthogonal"; only its test_hint and test comment were corrected |
| Verdicts | **25 CONFIRMED (C01–C25), 1 REFUTED (C26), 0 UNTESTABLE** |
| C26 | **REFUTED**. Doubling σ_g changes Σ[3:9,3:9] by **7.1e-3** relative (claimed ≤ 1e-12). ×10 changes it by **0.23** |
| My mutation sweep | **48 new file-level mutants: 37 killed, 11 survived**. Plus 8 file-level replays of pass-2 survivors, all killed |
| Chart explanation (M) | **Correct.** I derived it and confirmed it numerically. GTSAM's `preintMeasCov` matches an independent FD linearisation of the *exact-ZOH* integrator in the right Gal(3) chart to ≤ 3.6e-9. The leftover gap is Euler vs ZOH, with ratios 0.4998–0.5009 |
| test_c06b reference | **Independent** of the code under test (§2.1) |
| test_c05b reference | **Not circular** (§2.2) |

---

## 1. Independent re-implementations (no `src/` or `tests/` import)

`indep.py` uses these pieces:
- `Exp = scipy.linalg.expm∘hat` and `Log = logm`.
- Constants retyped from the M2-C01 text.
- Euler sums for Forster Eq. (35)–(37), plus **Eq. (63) retyped from the paper**:
  A = [[ΔR_{k,k+1}ᵀ,0,0],[−ΔR_ik â dt, I, 0],[−½ΔR_ik â dt², I dt, I]],
  B = [[J_r(ω dt) dt, 0],[0, ΔR_ik dt],[0, ½ΔR_ik dt²]], Q = diag(σ²)/dt.
- Cayley as an explicit `inv(I−φ̂/2)(I+φ̂/2)`.
- Exact ZOH from one 5x5 `expm`.
- My own Gal(3) matrix composition, plus a finite-difference linearisation of the noisy
  integrator with respect to per-step noise. It can run under either step law (Euler or
  ZOH) and in either error chart (Forster additive, or right Gal(3)).

### 1.1 Chart transform: derivation

GTSAM perturbs on the right in Gal(3): U·Exp(ξ) with ξ = (ω, ν, ρ, α). With α = 0, to
first order this gives:
- R → R Exp(ω),
- v → v + R ν,
- p → p + R ρ.

Forster's chart is R Exp(δφ), v + δv, p + δp. So δφ = ω, δv = ΔR ν and δp = ΔR ρ. That
means ξ_Gal = M ξ_F with **M = diag(I, ΔRᵀ, ΔRᵀ)** in (rot, vel, pos) order, and
Σ_Gal = M Σ_F Mᵀ. The direction of the transform matters. Using ΔR instead of ΔRᵀ makes
the error *worse* (0.24 / 0.38 instead of 0.11 / 0.21).

### 1.2 Covariance, numerically (anisotropic σ_g = [1, 1.7, 3]e-4, σ_a = [5, 10, 20]e-3, n = 200)

Values are max relative error per 3x3 block.

| comparison | edge 0 worst block | edge 1 worst block |
|---|---|---|
| my Eq. (63) vs `src` Forster cov | 1.9e-15 | 2.3e-15 |
| my Eq. (63) vs my FD of the **Euler** integrator (Forster chart) | 2.7e-9 | 1.8e-9 |
| GTSAM `preintMeasCov` (reordered) vs my FD of the **ZOH** integrator (right Gal(3) chart) | 1.7e-9 | 3.6e-9 |
| M Σ_F Mᵀ vs my FD of Euler in the right Gal(3) chart | 2.8e-9 | 1.7e-9 |
| raw Σ_F vs GTSAM: rot / R–v / R–p / vv / vp / pp | 0 / 0.112 / 0.113 / 0.130 / 0.129 / 0.129 | 0 / 0.205 / 0.208 / 0.191 / 0.190 / 0.190 |
| M Σ_F Mᵀ vs GTSAM: rot / R–v / R–p / vv / vp / pp | 2.2e-14 / 5.08e-3 / 7.56e-3 / 3.1e-4 / 3.0e-4 / 3.0e-4 | 1.8e-14 / 5.04e-3 / 7.54e-3 / 4.5e-4 / 4.4e-4 / 4.3e-4 |
| remainder ratios per halving, n = 100…1600 (all non-rot blocks) | 0.4998–0.5009 | 0.5000–0.5009 |

Conclusions:
- GTSAM's covariance **is** the exact-ZOH covariance in the right Gal(3) chart.
- Forster's covariance, carried through M, **is** the Euler covariance in that chart.
- The remainder is therefore purely Euler vs ZOH, and it is O(dt). The explanation in
  `e380b18` is right.
- One nuance the C06 hint omits: with anisotropic noise the untransformed vv/vp/pp blocks
  also differ (13%/19%), not only the R–v cross block. The hint does not claim otherwise.

### 1.3 Other reproduced numbers

All computed with independent code.

| claim | quantity | independent |
|---|---|---|
| C02 | §0.4 states, via 5x5 expm | KF2 v, p match the literals to all printed digits |
| C04 | \|dv−dv_ex\| at n = 200; ratios | 2.170879e-3 / 2.760238e-3; v 0.5000, p 0.5009→0.5001 |
| C05 | quadratic ratio | 100.01 / 100.00 |
| C10 | ground-truth \|r^v\|, \|r^p\| | 2.170924e-3, 5.377e-4 / **2.760341e-3**, 6.810e-4 |
| C10b (F09) | Cayley re-integration ratio at db ≠ 0 | 100.01 / 100.00 (v and p) |
| C22 | B2−B1 at KF2, v / p / R | 6.167e-8 / 4.376e-8 / 2.9965e-8; per-edge ‖ω‖³T dt²/12 = 2.3257e-8, 5.5175e-8 |
| C23 | B1/B2 − exact at KF2, v / p | 1.2867e-3 / 1.2031e-3; order ratios B1 v 0.50028…0.50003, **B2 v 0.50026…0.50003, p 0.49998…0.50000** |
| C24 | KF1 rotation, B1 / B2-raw | 9.4e-15 / 2.0e-8; KF2 B2-raw 2.9965e-8 |
| C07 | Cay(2tan(θ/2)a) vs Exp; right / left Jacobian | 2.9e-15; 6.0e-7 / 1.37 |
| C08 / C09 / C11 | identities | 3.6e-15 / 3.1e-16 / 4.4e-16 |
| C03 | `predict` vs component relation | ≤ 9.4e-15 |
| C14 / C15 / C17 | gtsam directly | as claimed (col3 = v, col4 = p; Expmap e_3 → v, e_6 → p, e_9 → t; vector() = [a; g]) |
| C16 | remapped Gal3 log vs `localCoordinates`; vs `NavState.Logmap` | 0.107628; 0.0 |
| C18 | my 5x5 matrix composition | W-form 2.2e-16 at t_i ∈ {0, 3.7}; F2 18.148500 = \|g\|·Δt·t_i |
| C19 | Galilean vs closed form; Tangent dv; right vs left (own expm V_k) | 2.0e-14 / 1.7e-14; 2.17e-3 / 2.76e-3; 3.3e-16 vs 6.2e-4 |
| C20 | right-composition residual; component-wise over 30 directions; Nelder–Mead sup | 8.8e-13…4.4e-10 / 6.3e-13…3.2e-10; 1.956e-5…9.40e-3 / 2.05e-5…1.02e-2; **1.418e-2 / 1.640e-2**; J[0:3,0:3] = J[9,:] = 0 exactly |
| C21 | recursion with my own ZOH V_k and C_k vs FD of re-integration | 4.2e-9 relative |
| C25 | own graph with a *different* noise model | ranks 30 / 39 / 45; cosines [0.978227, 0.978041, 0.839290, 0.815739, 0.299838, 0.299389]; pv energy 2.5371. Identical to the claim, which confirms that the nullspace does not depend on the weighting |
| B1/B2 on the time-varying stream | `src` vs my Euler sums | R ≤ 5.6e-16, v 2e-17, p 1e-18; left-accumulated product 4.6e-5 away |

---

## 2. The three scrutiny points

### 2.1 Is test_c06b's reference independent? **Yes.**

`_covariance_reference` takes these inputs from `src`: `T.edges()` (the stream),
`T.BIAS_TRUE` and `T.hat`. It calls nothing in `src.preintegration_forster`. It builds the
Euler step and the composition law itself, uses scipy `expm`, retypes its own `Log`, and
gets Jacobians by central differences on prefix·step(η)·suffix.

The parts it necessarily shares with the code under test are the *model*, not the code:
- the same Euler discretization (Eq. (63) is defined as that scheme's linearisation),
- the same noise model, Q = diag(σ²)/dt.

My independently written version of the same reference reproduces `src` to ≤ 2.7e-9.

Two minor caveats, neither a defect:
- `hat` is shared.
- The reference is `lru_cache`d on the edge index, so a monkeypatched fixture would see
  a stale reference.

### 2.2 Is test_c05b's "exact step-by-step correction" circular? **No.**

`_gtsam_euler_reference` builds its result from three things:
- GTSAM's `PreintegratedImuMeasurementsG`,
- `T.M1` / `T.M2` (the fixture closed forms, which c02 pins against quadrature),
- `B3.to_gtsam_bias`.

It subtracts Σ R_k(M1 − dt I)a_k and the matching dp term, using GTSAM's own running R_k.
This is an exact algebraic identity: the ZOH sum minus the Euler sum. Nothing from
`ForsterPreintegration` or `CayleyPreintegration` enters it.

For B2, the identity Cay(θu) = Exp(2 arctan(θ/2) u) turns GTSAM's ∏Exp into ∏Cay. That is
an independent fact, and C22b(a) checks it separately.

My own Euler sums agree with `src` on this stream to ≤ 5.6e-16. The test therefore
compares the code to a genuinely different construction.

### 2.3 Chart-transform explanation. **Correct**: see §1.1–1.2.

---

## 3. Rulings on F07–F10

| finding | ruling | evidence |
|---|---|---|
| **F07** covariance | **FIXED** | test_c06b kills every covariance mutant I wrote: Jr→Jl (P08), Jr K² sign (P09), A_pv halved (P12), A_vR using ΔR_{k+1} (P13), Bg = Jrᵀ (P14), Qg/Qa swapped (P15), gyro per-axis order reversed (P16), covariance propagated *after* the mean update (P18), GTSAM cov returned in the wrong block order (P48). It also kills file-level replays of N27 (v-row only), N28 and a **finer** N08 (noise ×1.002). The weakest mutants it kills move a block by ~1e-3 against a 1e-6 bound. One gating survivor remains (P17, §4). It is in the noise switch, not in Eq. (63). |
| **F08** order | **FIXED** | File-level N24 and N11 are killed by c05b. P04 (time-varying gyro made constant) is killed by c05b and c19, so the commuting-input guard itself has teeth. The reference is not circular (§2.2). |
| **F09** db ≠ 0, dp fold | **FIXED** | File-level F05a3 is killed by c04(d) and c05b. N04 (only the A_vg term sign-flipped, finer than pass 2's) is killed by c10b. So are P34 (factor's db blocks swapped), P36 (position residual using A_v*), P31 and P32. Independent ratio 100.0. |
| **F10** hygiene | **PARTIALLY FIXED** | Fixed: the constants are pinned against retyped literals in-process and in the subprocess (P05 is killed by c01; the file-level N29 replay is killed). The C20 hint numbers are reproduced exactly (§1.3). test_c23 now sweeps B2-raw. The N13 contrast swap is killed by c20 at file level. **Not fixed: the M2-C25 *statement* still reads "closes the gap only because no null direction happens to be orthogonal to the bias block"**. The commit says "No claim statement changed". Only the test_hint and the test comment carry the structural argument. The yaml text the claim is judged on still contains the wording pass 2 called "wrong in kind". |

---

## 4. My mutation sweep: 48 new mutants (none in `tests/test_mutation.py`)

Method:
- Each mutant is a file-level edit in a fresh copy of `src/`, `tests/`, `conftest.py`
  and `pyproject.toml` placed in scratch. Each anchor string was asserted unique.
- I ran the full `tests/test_baselines.py` against each copy.
- The control copy passes 29/29. The worktree was never edited.

**Killed (37)**, with the killing tests:

| id | mutant | killed by |
|---|---|---|
| P01 | `so3_log` ×(1+1e-4) | c05b, c22b |
| P02 | M1 K² coefficient ×(1+1e-6) | c02, c19, c24, c25 |
| P03 | `chain()` gravity sign in v | c02, c03, c10, c10b, c16, c18, c24, c25 |
| P04 | time-varying gyro amplitude → 0 | c05b, c19 |
| P05 | accel stream adds B_G | c01, c04, c10, c19, c20, c24, c25 |
| P07 | `hat` handedness flipped | 13 tests |
| P08 | J_r → J_l | c06b |
| P09 | J_r K² sign | c06b |
| P10 | dp mean drops ½ | c04, c05b, c10, c23, c24 |
| P11 | `predict` dp not rotated by R_i | c10, c23, c24 |
| P12 | cov A_pv halved | c06, c06b |
| P13 | cov A_vR uses ΔR_{k+1} | c06b |
| P14 | cov Bg = J_rᵀ | c06b |
| P15 | Qg ↔ Qa | c06, c06b |
| P16 | gyro per-axis density order reversed | c06b |
| P18 | covariance step after the mean update | c06b |
| P19 | J_R FD denominator ε (×2) | c05, c05b |
| P20 | J_vg ↔ J_va | c05, c10b |
| P21 | Eq. (44) db_g ↔ db_a | c05 |
| P24 | Cay(−φ) | 9 tests |
| P25 | `cayley_vector` tan θ | c07, c22, c23, c24 |
| P26 | J_Cay denominator /3 | c07 |
| P27 | Eq. (17) closed form /4.0001 | c08 |
| P28 | Eq. (19) ½ → ¼ | c09, c11, c13 |
| P30 | Eq. (22b) R̆ order (I−Γ/2)ΔR̄ | c11 |
| P31 | Eq. (21b) gravity sign | c10, c10b |
| P32 | Eq. (21c) drops v_i Δt | c10, c10b |
| P34 | factor velocity db blocks swapped | c10b |
| P36 | position residual uses A_vg/A_va | c10b |
| P37 | reparam variant == raw | c22, c23, c24 |
| P38 | H′ position sign | c18 |
| P39 | W drops +Δt in the clock term | c18 |
| P41 | recursion Ad(J + C) | c21 |
| P42 | bias model left-composed | c20 |
| P43 | integration covariance not set (GTSAM default left in) | c06 |
| P47 | `gal3_components` p ↔ v | c18, c20 |
| P48 | `preint_cov` returned in (rot, vel, pos) order | c06, c06b |

**Survived (11):**

| id | mutant | why | severity |
|---|---|---|---|
| **P17** | `has_noise`: `or` → `and`. Covariance is silently **not propagated** when only one density is nonzero | every covariance test sets both σ_g and σ_a > 0. Correct gyro-only trace = 2.9e-7; the mutant returns 0 | **medium**: a real defect class (e.g. a gyro-only ablation gives zero covariance) |
| **P35** | `CayleyFactor.from_edge` takes J_R and A from the **Forster** (Exp) integrator, not Cayley | the difference is 2.3e-7 relative in J_R, A_vg and A_pg, far below any assertion | low–medium: violates Rai Sec. IV-A ("Jacobians of the Cayley retraction") but is numerically O(dt²) |
| P22 | `increment_error` ignores dp | c05's quadratic-ratio test is then blind to dp-correction errors. J_p errors are caught only through c10b (Cayley path) | low–medium |
| P23 | Forster `FD_EPS` 1e-6 → 1e-3 | J_R and A nearly linear in b; no test is sensitive to the FD step | low |
| P46 | Galilean `C_FD_EPS` 1e-6 → 1e-3 | c21 still within 1e-6 relative | low |
| P29 | C13 truncated contrast uses (I − ξ̂) | c13(b) only lower-bounds the contrast's distance. Same class as the pass-2 N13 | low |
| P33 | `weighted_square(r)` returns ‖r‖ (not squared) | c12's isotropic half is invariant under any function of ‖r‖ | low |
| P40 | corrected increment's clock set to 0 | every comparison shares t; J[9,:] = 0 holds trivially | low (equivalent on this fixture) |
| P44 | combined bias random-walk covariances σ_bg ↔ σ_ba | rank and nullspace do not depend on whitening (my own graph with other densities gives identical cosines) | equivalent mutant |
| P45 | `GalileanPreintegration.predict` ignores its `bias` argument | no caller passes `bias` | dead parameter |
| P06 | `REF_R_KF1` corrupted | dead code; no test reads it | dead code |

**File-level replays of pass-2 survivors (not new, run to confirm that the harness's
monkeypatch reaches the real call sites):** N24, F05a3, N04 (single-term variant), N27
(v-row), N11, N13, N28, and N08 at ×1.002. **All 8 killed**, by c05b / c04+c05b / c10b /
c06b / c05b / c20 / c06b / c06b respectively.

---

## 5. Tests and assertions that cannot fail, or that compare the code to itself

1. `test_c10` (a): the self-consistent residual is a tautology (it is labelled as one).
2. `test_c10`, `|r^v_gt| == |dvbar − dv_exact|`: an identity with respect to the Eq. (9)
   sums (the docstring now says so correctly).
3. `test_c13`: `J_R.tobytes()` is unchanged, but no code path writes `J_R`. Also,
   `predicted_rotation == R_i ΔR̄ B2.cay(J_R δ)` re-states the implementation with the same
   `B2.cay`. It is self-comparison, though it does pin the wiring.
4. `test_c13` (b) and `test_c20`'s component-wise ≥ 1e-2: lower bounds on contrast
   models. c13(b) cannot fail on a contrast that gets worse (P29 survives). c20's version
   is now backed by the first-order pin.
5. `test_c05`: `allclose(corrected[0], base.dR @ so3_exp(J_R @ db_g), atol=0, rtol=0)`
   re-states `bias_corrected_increment`'s own formula (self-comparison; it kills only M03).
6. `test_c12` isotropic half: SO(3) orthogonality, blind to P33.
7. `test_c08`: `closed >= 1` is `1 + x²/4 ≥ 1`.
8. `test_c20`: `J[9,:] = 0` cannot fail for any bias-only perturbation (P40).
9. `test_c25`: rank 30 and rank 39 are full row rank (labelled as structural).
10. `test_c06`: the symmetry / PSD / monotone assertions are structural. Only its GTSAM
    magnitude cross-check has teeth (P12, P15, P43, P48).
11. `test_c01`: `R0` vs `expm(T.hat(rot_0))` shares `T.hat` with `R_0 = so3_exp(...)`. A
    handedness change of `hat` passes this line (it dies in c02 instead).
12. `test_mutation.py::test_m04…`: asserts a genuine no-op, by design.
13. Nothing tests the `has_noise` gate with exactly one nonzero density (P17).

---

## 6. Verdicts per claim

| id | verdict | attack |
|---|---|---|
| C01 | CONFIRMED | Constants retyped from the claim text. P05 (stream bias swap) and the file-level N29 replay are both killed by the new literal pins. |
| C02 | CONFIRMED | Independent 5x5 expm reproduces §0.4. P02 (M1 ×(1+1e-6)) is killed. |
| C03 | CONFIRMED | Direct gtsam `predict` vs my chain: ≤ 9.4e-15. P03 is killed. |
| C04 | CONFIRMED | Independent ratios 0.5000 (v) and 0.5009→0.5001 (p). File-level dp-only fold is killed by part (d). |
| C05 | CONFIRMED | Own FD Jacobians on own integrator: ratio 100.01 / 100.00. P19, P20, P21 killed. (P22 shows the metric is dp-blind.) |
| C06 | CONFIRMED | Statement (symmetric / PSD / monotone) holds. The hint's chart numbers are reproduced: raw R–v 0.112 / 0.205, charted 5.08e-3 / 7.56e-3, ratio 0.50. Correctness is now pinned by c06b. |
| C07 | CONFIRMED | 2.9e-15; right 6.0e-7, left 1.37. P25 and P26 killed. |
| C08 | CONFIRMED | 3.6e-15. P27 killed. |
| C09 | CONFIRMED | 3.1e-16 on-manifold. P28 killed. |
| C10 | CONFIRMED | 2.760341e-3 / 2.170924e-3 independently. P31 and P32 killed. |
| C11 | CONFIRMED | 4.4e-16. P30 killed. |
| C12 | CONFIRMED | True by SO(3) orthogonality. The test is blind to P33 (§5). |
| C13 | CONFIRMED | Affine map and retraction re-checked. Truncation vs Cay ≈ ‖ξ‖²/2. (P29 shows the contrast half is only lower-bounded.) |
| C14 | CONFIRMED | gtsam directly. |
| C15 | CONFIRMED | Expmap(1e-6 e_k) block semantics directly. |
| C16 | CONFIRMED | 0.107628; group log 0.0. |
| C17 | CONFIRMED | `ConstantBias` directly. |
| C18 | CONFIRMED | Own matrix composition: W 2.2e-16, F2 18.148500. P38 and P39 killed. |
| C19 | CONFIRMED | 2.0e-14 / 1.7e-14; Tangent 2.17e-3 / 2.76e-3; right 3.3e-16 vs left 6.2e-4 with my own ZOH V_k. P04 killed. |
| C20 | CONFIRMED | Own FD J: right ≤ 4.4e-10; component-wise sup 1.418e-2 / 1.640e-2 ≥ 1e-2 (the 30-direction max on edge 0 is 9.40e-3, as the hint now says). P42 and P47 killed. |
| C21 | CONFIRMED | Own ZOH recursion: 4.2e-9 relative. P41 killed. |
| C22 | CONFIRMED | 6.167e-8 / 4.376e-8 / 2.9965e-8; deficit = ‖ω‖³T dt²/12. P24, P25, P37 killed. |
| C23 | CONFIRMED | Including the B2 order sweep (0.50026…0.50003), which the test now runs. |
| C24 | CONFIRMED | B1 rotation 9.4e-15; B2-raw 2.0e-8 (KF1) and 3.0e-8 (KF2). |
| C25 | CONFIRMED | Own graph with different densities: ranks 30 / 39 / 45 and identical cosines. The wording caveat stands (F10). |
| **C26** | **REFUTED** | See below. |

**M2-C26: REFUTED.** The claim is that scaling σ_g leaves Σ[3:9,3:9] unchanged to 1e-12
relative. Attack: propagate Eq. (63) with σ_a = 1e-2 fixed and σ_g = f·1.7e-4, then
measure max|ΔΣ[3:9,3:9]| / max|Σ[3:9,3:9]|. My independent code and `src` agree to every
digit:

| f | edge 0 | edge 1 |
|---|---|---|
| 0 | 2.369e-3 | 2.213e-3 |
| 0.5 | 1.777e-3 | 1.660e-3 |
| 2 | **7.107e-3** | **6.638e-3** |
| 10 | 0.2345 | 0.2191 |
| 100 (vv block) | ×23.7 | ×22.1 |

The claim is wrong by 9–10 orders of magnitude. Its reasoning ("dv and dp integrate
accelerations only") ignores the rotation-error coupling A[3:6,0:3] = −ΔR_ik â dt and
A[6:9,0:3] = −½ΔR_ik â dt². Through these, gyro noise enters δφ and then δv and δp
(about |a| ≈ 10 m/s² × δφ × T). The effect of gyro noise on the vv/pp blocks therefore
grows as σ_g². At σ_g = 1.7e-4 it is only about 0.5% of those blocks, but it is never
zero. Any covariance mutant that deleted this coupling would contradict Eq. (63), and
c06b would kill it (N27 / R-N27 do die).

**Counts: 25 CONFIRMED, 1 REFUTED (C26), 0 UNTESTABLE.**

---

## 7. Recommendations

1. Add a gyro-only and an accel-only covariance case. This pins the `has_noise` gate (P17).
2. Make `increment_error` and c05 check dp explicitly, or assert each component's
   quadratic ratio separately (P22).
3. Pin `CayleyFactor`'s J_R against the Cayley integrator specifically. The 2.3e-7 gap to
   Forster's J_R is resolvable with a two-sided check (P35).
4. Rewrite the M2-C25 statement to drop "happens to" (F10 remainder).
5. Either delete `REF_R_KF1` and the unused `bias` parameter of
   `GalileanPreintegration.predict`, or test them.
6. Do not enter M2-C26 into the yaml as stated. The true statement is the opposite: gyro
   noise feeds Σ_vv and Σ_pp through the rotation coupling.
