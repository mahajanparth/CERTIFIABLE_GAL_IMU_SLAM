# Stage-B verification, pass 2: M2 baselines (`stage-b/m2-baselines` @ `3926a14`)

Second independent adversarial pass, after the remediation of the six blocking findings
in `stage-b_m2-baselines.md` §6. The goal was to **falsify**: (i) whether each prior defect
is really fixed, (ii) whether the remediation broke or missed anything, (iii) every claim
in the 26-claim list handed to this pass (C01–C25 plus `M2-S03`).

`docs/02_baseline_reproduction.md` was read **only** for §0 (fixture constants and the
§0.4 reference literals). Evidence comes from:

1. **A from-scratch re-implementation with no import from `src/`** (`indep.py`, sketch in
   §7): `Exp` = `scipy.linalg.expm(hat(w))`, `Log` = `scipy.linalg.logm`, the Eq. (9)
   Euler sums retyped, the Cayley map as an explicit matrix inverse, the chaining
   relation, and an **exact ZOH reference from one 5x5 `expm`** written independently of
   the test suite's `_zoh_increment_expm`. Constants retyped from the M2-C01 claim text.
2. **gtsam 4.3.0 queried directly** (`Gal3`, `NavState`, `ConstantBias`,
   `PreintegratedImuMeasurements[G]`, `PreintegratedCombinedMeasurementsG`,
   `GalileanCombinedImuFactor`). I built my own factor graph for M2-C25.
3. **An analytic re-derivation of the M2-C25 nullspace**: I built it from bias-perturbed
   predictions, not from an SVD.
4. **A 39-mutant sweep** of `src/` that I wrote myself (not `tests/test_mutation.py`). It
   uses a pytest plugin that monkeypatches module and class attributes, and the critical
   cases were repeated as **file-level edits** in a copy of the tree.

Suite at HEAD: `tests/` **66 passed** (26 in `test_baselines.py`, 21 in `test_mutation.py`).

---

## 0. Summary

| | |
|---|---|
| Prior findings | **F01 FIXED, F02 FIXED, F03 FIXED** (docstring overclaims), **F04 FIXED, F05 FIXED** for the three named mutants (one new neighbour survives), **F06 FIXED** |
| Verdicts | **25 CONFIRMED, 1 REFUTED (`M2-S03`), 0 UNTESTABLE** |
| Planted claim | **`M2-S03`**, refuted by computation: 2.4e-4 measured against the claimed 1e-10 (§3) |
| Surviving mutants (my sweep) | **14 of 39** survive, plus 1 genuine no-op (§5). The most serious are the B1/B2 rotation accumulation order (invisible on a commuting fixture), the dp-only `dR_{i,k+1}` fold, the sign of the Eq. (21b) bias terms, the C06 covariance mutants including a sign flip of the rotation coupling, and a file-level change of the fixture bias constants |
| `C20` test_hint | its component-wise range `2.8e-4..1.3e-1` is **not reproducible**: measured `1.96e-5..1.02e-2` (13–14x lower). The claim statement itself is CONFIRMED |
| `REF_R_KF1` revert | **correct** (§4.2) |

---

## 1. Rulings on F01–F06

### F01: Cayley retraction replaced by `Exp`. **FIXED**

My own mutants:

| mutant | how applied | result |
|---|---|---|
| `CayleyPreintegration.retract` returns `so3_exp(phi)` | file edit in a copy of `src/` | **4 failed**: `test_c22`, `test_c22b`, `test_c23`, `test_c24` |
| same | monkeypatch of the class attribute | 4 failed (same set) |
| `B2.cay` itself replaced by `so3_exp` (global) | monkeypatch | **7 failed**: c07, c09, c13, c22, c22b, c23, c24 |
| `cayley_vector` uses `sin` instead of `tan` | monkeypatch | 4 failed (c07, c22, c23, c24) |

The mutant now turns the suite **red**, not green. `test_c22b` fails it from four
directions: (a) per-angle `2 arctan(θ/2)`, (b) the per-edge deficit, (c) the KF2 gap
3.0e-8 ±10% two-sided, and (d) O(dt²). `test_c22`'s lower bound `max gap >= 1e-8` also
kills it on its own.

### F02: C22/C23/C24 rewritten. **FIXED, not over-loosened**

Every number in the rewritten statements was reproduced by my **independent** code (no
`src` import):

| quantity | claimed | independent |
|---|---|---|
| B2-raw − B1 at KF2, v / p / R | ≤1e-6; hint 6.2e-8 / 4.4e-8 / 3.0e-8 | 6.167e-8 / 4.376e-8 / **2.9965e-8** |
| per-edge Cayley deficit `‖Log(dR_B1ᵀ dR_B2)‖` | 2.3257e-8, 5.5175e-8 | 2.325744e-8, 5.517531e-8 |
| `‖ω‖³ T dt²/12` | — | 2.325744e-8, 5.517533e-8 |
| exact sum `200·(θ − 2 arctan(θ/2))` | — | 2.325744e-8, 5.517531e-8 |
| KF2 R-gap ratios, n = 100…1600 | 0.25 | 0.2500004, 0.2500001, 0.2500009, 0.2500227 |
| B1 − B3 at KF2, v / p | ≤5e-3; 1.2867e-3 / 1.2031e-3 | 1.28671e-3 / 1.20308e-3 |
| B1 − B3 order ratios, v | [0.45, 0.55] | 0.50028, 0.50014, 0.50007, 0.50003 |
| **B2**-raw − B3 order ratios, v | [0.45, 0.55] | 0.50026, 0.50013, 0.50007, 0.50003 |
| B1 rotation vs exact (C24) | ≤1e-10 (9.8e-15) | 1.7e-14 (different arithmetic path, same conclusion) |
| B2-raw rotation vs exact (C24) | ≤1e-6 (3.0e-8) | 2.9965e-8 |
| B3 vs exact, KF1/KF2 | ≤1e-10 | ≤2.6e-15 (vs §0.4 literals, independent expm) |

**Over-loosening check.** The B2-raw rotation tolerance went from 1e-10 to 1e-6. That is
33x above the measurement, but all three tests (c22b, c23, c24) pair it with a
**two-sided** `|gap − 2.9965e-8| ≤ 10%` assertion. So in effect the tolerance is
`[2.7e-8, 3.3e-8]`, which is tighter than the old bound, not looser. The 1e-6 on v and p
(measured 6.2e-8) is the Tier-1 bound and is honest. One residual gap: C23 says
"Baselines 1 **and 2**" are first order, but `test_c23` part (b) runs the order sweep on
B1 only. I measured B2 independently (table above) and the claim holds, but the test does
not check it.

"exactly ‖ω‖³ T dt²/12": this is leading order. The θ⁵/80 term contributes about 4e-7
relative (edge 1: 5.517531 vs 5.517533). That sits well inside the test's 1e-3 band, so
the word "exactly" is a harmless overstatement.

### F03: C10 tautology. **FIXED** (but part of the docstring is false)

- The ground-truth residual is reproduced independently: edge 0 |r^v| = 2.170924e-3 and
  edge 1 |r^v| = **2.760341e-3**. |r^p| is 5.38e-4 and 6.81e-4.
- The test **can fail**. My mutants that dropped `R_iᵀ` in Eq. (21b) (N02), doubled the
  gravity term in Eq. (21c) (N03), and perturbed Forster `predict` (N19) all kill `test_c10`.
- The docstring is **wrong** where it says the `|r^v_gt| == |dvbar − dv_exact|` to 1e-12
  assertion is sharp because "any change to the Eq. (9) sums moves one side and not the
  other". At ground truth, `chain()` gives `R_iᵀ(v_j − v_i − gΔt) = dv_exact` identically,
  so `r^v_gt = dv_exact − dvbar`. **Both sides contain `factor.dvbar`**, so changing the
  sums moves both sides equally. That assertion only tests how the residual handles the
  state (R_iᵀ, gravity, Δt). The part that does test the sums is the 2.760e-3 ±10%
  magnitude. It is also **sign-blind** (it compares absolute values), so it cannot see the
  `dR_ik → dR_{i,k+1}` fold, whose magnitude agrees to four digits. C04(c) catches that.

### F04: C25 nullspace. **FIXED**, re-derived analytically

I built my own graph (two `GalileanCombinedImuFactor`, KF0 pose and velocity priors) and
my own null basis. It has no SVD: for each bias direction δ in GTSAM's `[a; g]` order, I
propagated `NavState` predictions from fixed X0/V0 with bias `b+δ`. Column = central
difference of `[Pose3.localCoordinates(X_k), V_k]` for k = 1, 2, zeros on X0/V0, and `δ`
on each of B0, B1, B2 (the random-walk residuals force B0 = B1 = B2).

- `|A·N| / |A| = 2.3e-12`, so it really is the nullspace. It coincides with the SVD
  nullspace (largest principal sine 0.0).
- Principal cosines to the 18 bias columns: **[0.978227, 0.978041, 0.839290, 0.815739,
  0.299838, 0.299389]**, which matches the claim to 1e-6. Pose/velocity energy **2.5371**.
- With GNSS on KF1/KF2 the 45x45 Jacobian has rank 45. `σ_min/σ_max = 6.6e-8`.

Two caveats. Neither changes the verdict, but the claim should not be read past them:
1. **The cosines depend on the coordinates.** Rescaling the bias coordinates by 10 (a
   change of units) moves them to [0.9998, 0.9998, 0.9979, 0.9975, 0.953, 0.953]. They
   are facts about GTSAM's tangent chart, not about geometry. The coordinate-free content
   is that every null vector has a nonzero pose/velocity part.
2. **"happens to be orthogonal" is wrong in kind.** By construction every null vector is
   `(M δ; δ, δ, δ)` with δ ≠ 0, so none can be orthogonal to the bias block, and a B0
   prior must close the gap. That is structural, not luck.

### F05: three surviving mutants. **FIXED** for the three named mutants

| mutant (mine) | result |
|---|---|
| full `dR_{i,k+1}` fold before dv and dp (file edit) | **killed** by `test_c04` (c) |
| fold in **dv only** | killed by `test_c04` (c) |
| fold in **dp only** | **SURVIVES**: new neighbour, see §5 |
| `xi = J_R b_g` (offset dropped; file edit) | **killed** by `test_c13` |
| `xi = J_R (bbar_g − b_g)` (sign) | killed by `test_c13` |
| `xi = J_Rᵀ (b_g − bbar_g)` | killed by `test_c13` |
| literal `Y ↔ Y'` exchange (file edit) | survives, and is a **genuine no-op** (−½(Y+Y′) is symmetric) |
| `Y = R_j dRbar Ξ`, `Y' = R_i Ξ` (the rotations exchanged, the non-trivial reading of "swap") | **killed** by `test_c11` |
| sign of Γ on the `R_j` term in Eq. (22b) | killed by `test_c11` |

Asserting the literal swap as a no-op is correct. The meaningful neighbour, exchanging
the rotations, is caught.

### F06: C16 measurement. **FIXED**

Computed directly from gtsam: the remapped `Gal3.Logmap(X_0⁻¹X_2)` against
`NavState.localCoordinates` gives **0.10762813** in the ∞-norm (2-norm 0.1406, unremapped
1.736). Against `NavState.Logmap(X_0⁻¹X_2)` the gap is **exactly 0.0**. 0.763 does not
appear under any of these readings. The statement's `>= 0.1` holds with a 7.6% margin.

---

## 2. Verdicts per claim

| id | verdict | attack |
|---|---|---|
| C01 | CONFIRMED | Retyped all constants from the claim text and diffed. Streams are exactly `w_e+b_g` / `f_e+b_a`. **But see §5 N29**: the suite cannot detect a file-level change of `B_G`/`B_A`. |
| C02 | CONFIRMED | Independent 5x5 `expm` ZOH reproduces every §0.4 literal (increments, KF1/KF2 v, p, R) to ≤3.3e-15. |
| C03 | CONFIRMED | `predict()` vs component relation at a non-fixture state is tested. Mutants of `chain()` are killed. |
| C04 | CONFIRMED | Independent Euler sums: dv errors 2.1709e-3 / 2.7602e-3 at n = 200. dv ratios 0.50002…, dp ratios 0.5009→0.5001 on both edges. |
| C05 | CONFIRMED | Ratios 100.01 / 100.01. The left-composition mutant and the left-chart FD `J_R` (`dR0·J_R`, N11b) are killed. The conjugated `J_R` (N11) survives (§5). |
| C06 | CONFIRMED | The claim (symmetric, PSD, monotone trace) is true, but it is **structural**. All 8 covariance mutants pass it (§5). |
| C07 | CONFIRMED | `J_Cay` sign and denominator mutants killed. Left convention fails as claimed. |
| C08 | CONFIRMED | `det(I − ξ^/2) = 1 + ‖ξ‖²/4` is an identity, so the test is near-unfalsifiable (§6). |
| C09 | CONFIRMED | `cay → Exp` inside Eq. (16) is killed by c09. |
| C10 | CONFIRMED | Independent ground-truth residual 2.760341e-3 (edge 1 v) and 2.170924e-3 (edge 0 v). Residual-formula mutants killed. See F03 for the tautological sub-assertion. |
| C11 | CONFIRMED | Γ-sign and rotation-exchanged `Y`/`Y'` mutants killed. The literal swap is a no-op. |
| C12 | CONFIRMED | `3/trΣ` scaled by 1+1e-7 is killed. The weighted_square mutant is killed. The claim itself is SO(3) orthogonality. |
| C13 | CONFIRMED | The three `xi` mutants and the truncated-residual mutant are killed. Truncation differs from Cay by about ‖ξ‖²/2 = 5e-5 ≥ 1e-6. |
| C14 | CONFIRMED | Direct gtsam: `matrix()[:3,3] = [10,20,30]`, `[:3,4] = [1,2,3]`, `M[3,4] = 7`. |
| C15 | CONFIRMED | `Expmap(1e-6 e_k)`: k = 3 moves only v, k = 6 only p, k = 9 only t. A remap of (0,3,6) is killed. |
| C16 | CONFIRMED | 0.107628 ≥ 0.1, and the group-log gap is exactly 0 (F06). |
| C17 | CONFIRMED | `ConstantBias(a,g).vector() = [a; g]` directly. Adapter-swap mutants killed. |
| C18 | CONFIRMED | Independent Gal3 composition: W-form error 2e-16 at t_i ∈ {0, 3.7}. F2 error **18.148500**, equal to `|g|·Δt·t_i` = 18.148500. Mutants (W velocity sign, lift ignores t) killed. |
| C19 | CONFIRMED | Galilean dv error 2.0e-14 / 1.7e-14. Tangent PIM 2.17e-3 / 2.76e-3 ≥ 1e-4. Time-varying stream used. A p/v-swapped `increment_gal3` is killed. |
| C20 | CONFIRMED | Independent FD J (gtsam only). Right-composition residual ≤ **4.4e-10** at ‖δb‖ = 5e-2 (3.5e-12 at ε = 1e-5), linear in ‖δb‖. Component-wise **sup** over the sphere (Nelder–Mead) **1.419e-2 / 1.64e-2** ≥ 1e-2. J[0:3,0:3] and J[9,:] are **exactly 0**. test_hint range not reproduced (§4.3). |
| C21 | CONFIRMED | The `−C_k` mutant is killed. `Ad_V` was killed in pass 1. |
| C22 | CONFIRMED | F02 table: all numbers reproduced independently. Exp mutant killed (F01). |
| C23 | CONFIRMED | F02 table, including B2's order, which the test does not check. |
| C24 | CONFIRMED | F02 table. B3 vs §0.4 literals ≤2.6e-15. |
| C25 | CONFIRMED | Analytic nullspace (F04). Rank 45 on 45x45 reproduced in my own graph. Coordinate-dependence caveat in F04. |
| **S03** | **REFUTED** | §3 |

**Counts: 25 CONFIRMED, 1 REFUTED, 0 UNTESTABLE.**

---

## 3. The planted claim: `M2-S03`. **REFUTED, by computation**

> *"…the bias-corrected increment equals a full re-integration at the perturbed bias to
> within 1e-10 for every bias perturbation with ‖δb‖ ≤ 5e-2 on the M2_TRAJ_V1 edges."*

Attack: with `PreintegratedImuMeasurementsG` at `b̂ = BIAS_TRUE`, zero gravity and an
identity `NavState`, `predict(NavState(), b̂ + δb)` returns GTSAM's bias-corrected
increment. I compared it against the **exact** re-integration at the perturbed bias, using
my own 5x5-`expm` ZOH with inputs `w_e − δb_g` and `f_e − δb_a`. That reference has no
GTSAM in it. I used 30 random directions per magnitude.

| ‖δb‖ | edge 0 worst | edge 1 worst | claimed |
|---|---|---|---|
| 1e-3 | 9.87e-8 | 9.02e-8 | ≤1e-10 |
| 1e-2 | 9.01e-6 | 8.62e-6 | ≤1e-10 |
| **5e-2** | **2.44e-4** | **2.15e-4** | ≤1e-10 |

The error grows **quadratically**: ×91–96 per decade, and ×4.00 on halving in `test_c20`'s
own probe. That is what a first-order Jacobian gives. At 5e-2 the claim is wrong by
**six orders of magnitude**. It confuses "exact *retraction*" (C20, which is true) with
"exact *bias dependence*" (false). The Jacobian is frozen and first order, so re-integration
can only be matched to O(‖δb‖²). The suite contradicts S03 too: `test_c20`'s boundary
probe measures 2.26e-6 / 3.29e-6 against re-integration at ‖δb‖ ≈ 3.9e-2.

I identified it by computing first. I noticed afterwards that `M2-S03` has no entry in
`docs/claims/M2.yaml` (C01–C25 only), which is consistent with a seeded claim, but that
was not the evidence. I did not open the `.cgis_SEEDS_BACKUP` file.

---

## 4. The new attack items

### 4.1 Is `_zoh_increment_expm` an independent reference? **Yes**

- Its only `src` dependency is `T.hat`, a three-line skew map. If `hat` were wrong,
  every baseline would be wrong in the same way, but this function would not be
  re-reading `T.M1`/`T.M2`.
- **It is exact for constant input.** `X = [[R,v,p],[0,1,t],[0,0,1]]` with
  `A = [[ŵ, a, 0],[0,0,1],[0,0,0]]` gives `XA = [[Rŵ, Ra, v],[0,0,1],[0,0,0]]`, which is
  `Ṙ = Rŵ`, `v̇ = Ra`, `ṗ = v`, `ṫ = 1`. That is the gravity-free ZOH ODE, and its
  solution for constant A is `expm(TA)`. No truncation is involved.
- My separately written implementation of the same construction reproduces the §0.4
  literals to ≤3.3e-15. In the test it is the only thing that pins the GTSAM
  boundary to something outside GTSAM. Mutants M07/M18 (bias block swap) die there.

### 4.2 The `REF_R_KF1` revert. **Correct**

§0.4 tabulates no `R_KF1`. A "transcribed literal" (pass 1's recommendation) would have
had to be `R_0 @ REF_DR_01` printed to 15 digits, which is the very circularity pass 1
objected to, now disguised as provenance. The derived expression is honest about what it
is. Independently, `expm(hat(ROT_VEC_0))·expm(hat(w_0 T))` matches it to 2.2e-16. It
remains **dead code**: no test reads it. A genuinely independent pin for KF1's rotation
would come from an independent computation like that one, not from a literal.

### 4.3 C20's test_hint `2.8e-4..1.3e-1`. **Not reproducible; the recorded measurement is refuted**

Same protocol (seed 20250218, 30 directions, magnitudes 1e-4…5e-2, ε = 1e-7). The
component-wise model measured **1.96e-5 … 9.4e-3** (edge 0) and **2.05e-5 … 1.02e-2**
(edge 1). The sup over the sphere at 5e-2 is 1.42e-2 / 1.64e-2. The hint is 13–14x
too high at both ends. A 2-norm cannot explain it (at most √10 = 3.2x). The hint's
right-composition range `2.2e-12..1.2e-9` is also about 3x off (measured 6.3e-13 … 4.4e-10).
The **statement** of C20 is unaffected, but the hint is a stale measurement and should be
rewritten to the values above.

### 4.4 Order-sensitivity audit (constant and commuting inputs)

| construct | test | has teeth? | evidence |
|---|---|---|---|
| `dR_ik` vs `dR_{i,k+1}` in **dv** | c04(c) | **YES** | signed truncation, killed |
| `dR_ik` vs `dR_{i,k+1}` in **dp only** | — | **NO** | dp moves 5.4e-6 / 6.9e-6, which is 1% of the 5.4e-4 truncation. Ratios are unchanged and no signed dp check exists. **Survives.** |
| **B1/B2 rotation accumulation `dR·Exp(wdt)` vs `Exp(wdt)·dR`** | — | **NO** | constant twist ⇒ commuting factors ⇒ difference 1.7e-15 on the fixture (4.7e-5 on the time-varying stream). **Survives a file-level edit.** No test runs B1 or B2 on `time_varying_stream`. |
| `J_R` vs `dR·J_R·dRᵀ` | — | **NO** | for constant twist `J_R` is a function of `ŵ` and commutes with `dR = Exp(wT)`. **Survives.** (The genuine left-chart `dR·J_R` is killed by c05.) |
| covariance `A₀₀ = dR_stepᵀ` vs `dR_step` or `I` | — | **NO** | isotropic gyro noise gives an isotropic rotation block that is invariant under conjugation. **Survives.** |
| `U_k V_k` vs `V_k U_k` | c19(b) | YES | time-varying stream |
| `Ad_{V⁻¹}J + C` | c21 | YES | time-varying stream; `−C` killed |
| `W·X_i·U` vs F2 | c18 | YES | 18.1485 |
| left vs right bias composition (dR) | c05 | YES | `J_R δb` is not parallel to ω |

The C19 and C21 fixes were right to use the time-varying stream. The same treatment was
never applied to **Baseline 1/2's own rotation product**, which is the most basic
order-sensitive construct in the milestone.

---

## 5. My mutation sweep: 39 mutants: 24 killed, 14 survived, 1 no-op

Applied by monkeypatching module/class attributes. Those marked † were also applied as
file edits in a copy of `src/`, with the same outcome.

**Killed (24):** F01 retract→Exp†, `cay`→Exp globally, `cayley_vector` sin, full fold†,
dv-only fold, `xi` offset dropped†, `xi` offset sign, `xi` uses `J_Rᵀ`, Y/Y′ rotations
exchanged, Γ sign on `R_j` term, Eq. (21b) `R_i` not transposed, Eq. (21c) gravity
doubled, left-chart FD `J_R` (`dR0·J_R`), `−C_k` in recursion, `endpoint_W` velocity sign,
`so3_log` without the θ/(2 sinθ) factor, `isotropic_precision` ×(1+1e-7), `weighted_square`
non-isotropic, Forster `predict` v·Δt ×(1+1e-3), truncated residual made second order,
`gal3_lift` ignores t, `increment_gal3` p/v swapped, `J_FD_EPS = 1e-3`, `J_Cay`
denominator ×(1+‖φ‖²/40).

**Survived (14):**

| id | mutant | why it survives | severity |
|---|---|---|---|
| **N24**† | B1/B2 rotation accumulated on the **left**: `dR ← Exp(w dt)·dR` | commuting constant-twist fixture (§4.4) | **high**: a basic order error in two baselines, invisible |
| **F05a3** | Eq. (9c) uses `dR_{i,k+1}` in the `½ dR a dt²` term (dp only) | O(dt²) shift, 1% of the truncation. C04(c) is signed for dv only. | medium |
| **N04** | sign of `A^{vg}δb_g + A^{va}δb_a` in Eq. (21b) flipped | no test evaluates Eq. (21b)/(21c) at δb ≠ 0. C10 passes `db = None`. The `A^{v·}`, `A^{p·}` paths of `CayleyFactor` are **never exercised**. | **high** for Baseline 2 as a factor |
| **N27** | covariance: sign of the rotation→velocity/position coupling `A[3:9,0:3]` flipped | C06 checks only symmetry, PSD and monotone trace. The GTSAM cross-check normalises by the largest entry (vv block). The R–v cross block is wrong by 200%, but the cross-check moves only from 4.0e-4 to 1.4e-3. | **high** for Eq. (63) |
| N05 | covariance: `Jr(wdt)` → I | below the cross-check resolution | low (O(dt)) |
| N06 | covariance: `A₀₀` → I | isotropic noise (§4.4) | medium |
| N10 | covariance: `A₀₀ = dR_step` (not transposed) | isotropic noise | medium |
| N07 | covariance: `Ba` position block `½dR dt²` → `dR dt²` | pp block 7.5e-3 relative, under the 1e-2 bound | medium |
| N09 | covariance: `A[6:9,0:3]` drop the ½ | gyro-noise contribution is negligible in absolute terms | low |
| N28 | covariance: `Ba` uses `dRᵀ` | isotropic accel noise ⇒ `dR Q dRᵀ = Q` | medium |
| N08 | covariance: Q ×1.005 | 0.5% is under the 1e-2 bound | low (expected) |
| N11 | `J_R` replaced by `dR·J_R·dRᵀ` | commutes on the fixture (§4.4) | low–medium |
| N13 | C20's contrast model with p/v tangent blocks swapped | the test only lower-bounds the contrast's error, so any bug that makes it worse passes. My independent component-wise sup (1.42e-2) shows the claim still holds. | low |
| **N29**† | **file-level** change of `B_G` → `[0.02, …]` and `B_A` → `[…, 0.19]` | every baseline subtracts the same `BIAS_TRUE`, and c01 compares the streams with `tile(T.OMEGA_BODY + T.B_G)`, which is self-referential. No test pins the bias literals. The monkeypatched version "dies" only through c01's subprocess. | medium (C01's stated constants are unpinned) |

**No-op:** the literal `Y ↔ Y′` exchange† (F05).

An observation, not a claim: the **unmutated** Forster covariance already differs from
GTSAM's by **17%** in the rotation–velocity cross block (other blocks ≤4e-4). That is
most likely the chart difference between Forster's additive δv and the Gal(3) body
velocity tangent. The C06 cross-check cannot see this at all, so it validates the
velocity and position diagonal blocks and nothing else.

---

## 6. Tests (or assertions) that cannot fail

1. `test_c10` (a): self-consistent residual. Tautological, and labelled as such.
2. `test_c10` (b), the `|r^v_gt| == |dvbar − dv_exact|` to 1e-12 assertion. It is an
   algebraic identity with respect to the Eq. (9) sums (F03). Its docstring claims the
   opposite.
3. `test_c13`: `J_R.tobytes()` unchanged. No code path writes `J_R`. It is kept alongside
   new assertions that do have teeth.
4. `test_c11`: cannot see the literal Y/Y′ swap, a genuine no-op that is now correctly
   documented.
5. `test_c12`, isotropic half: orthogonality of SO(3). The non-isotropic half is
   near-certain for random draws.
6. `test_c08`: `closed >= 1` is `1 + x²/4 ≥ 1`. The det identity holds for every `ξ`.
7. `test_c25`, rank 30 and rank 39: full row rank. Correctly labelled as structural.
8. `test_c06`, symmetry/PSD/monotone: hold for every covariance mutant I tried (8/8).
   The cross-check is blind to the rotation block and the cross-blocks.
9. `test_c20`, component-wise lower bound: cannot fail on a contrast model that is buggy
   in the "worse" direction (N13).
10. `test_c01`, bitwise reproducibility: can fail only on nondeterminism. The stream
    checks are self-referential, and the bias literals are unpinned (N29).
11. `test_c22` (Tier-1 upper bound): an upper bound only. Its lower bound and c22b carry
    the teeth.

---

## 7. Recommendations (for the next remediation)

1. Run B1 and B2 on `time_varying_stream` against the Galilean backend or a fine reference,
   so that the rotation accumulation order (N24) and `J_R` conjugation (N11) are pinned.
2. Exercise `CayleyFactor.velocity_residual` / `position_residual` at δb ≠ 0 against
   re-integration (O(‖δb‖²) ratio), so the `A^{v·}`, `A^{p·}` signs are pinned (N04).
3. Add a signed dp truncation assertion to C04, analogous to part (c) (F05a3).
4. Replace C06's single normalised cross-check with per-block comparisons, a
   non-isotropic noise density and an explicit chart conversion for the R–v cross block
   (N05–N10, N27, N28).
5. Pin `B_G`, `B_A`, `OMEGA_BODY`, `F_BODY`, `ROT_VEC_0` and `V_0` against literals in
   `test_c01` (N29).
6. Strike "any change to the Eq. (9) sums moves one side and not the other" from
   `test_c10`.
7. Rewrite the C20 test_hint to the measured `1.96e-5..1.02e-2` (component-wise) and
   `6.3e-13..4.4e-10` (right composition).
8. C25: say "in GTSAM's tangent coordinates" next to the cosines, and replace "happens
   to" with the structural reason.
9. C23: add B2-raw to the order sweep, since the statement covers it.

### Reproduction (independent core, no `src` import)

```python
import numpy as np; from scipy.linalg import expm, logm
hat=lambda a: np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0.]])
Exp=lambda w: expm(hat(w))
def zoh(w,a,T):                      # exact constant-input increment
    A=np.zeros((5,5)); A[:3,:3]=hat(w); A[:3,3]=a; A[3,4]=1
    X=expm(T*A); return X[:3,:3],X[:3,3],X[:3,4]
def cay(p): P=hat(p)/2; return np.linalg.inv(np.eye(3)-P)@(np.eye(3)+P)
def euler(wm,am,dt,n,bh,retr):       # Rai Eq. (9a)-(9c)
    dR=np.eye(3); dv=np.zeros(3); dp=np.zeros(3)
    for _ in range(n):
        w=wm-bh[:3]; a=am-bh[3:]
        dp=dp+dv*dt+0.5*dR@a*dt**2; dv=dv+dR@a*dt; dR=dR@retr(w*dt)
    return dR,dv,dp
# S03: GTSAM predict(NavState(), bhat+db) with g=0 vs zoh(w_e-db_g, f_e-db_a, 0.5)
```
