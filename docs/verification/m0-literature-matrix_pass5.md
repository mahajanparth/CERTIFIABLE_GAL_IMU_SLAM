# M0 adversarial verification, pass 5: literature matrix (narrow)

**Verifier:** independent pass 5. **Branch:** `m0-literature-matrix` at `d939780` (F12 fix commit).
**Date:** 2026-10-13.
**Under test:** M0-C01..C15 (`docs/claims/M0.yaml`), M0-C19 (dispatch brief only), and the M0-F12 fix
(the restated `error_handedness` definition in the `src/conventions.py` module docstring).

**Independence.** I did not read `docs/00_literature_matrix.md`, any `seeds*.yaml`, or anything outside
this worktree. Fixtures were decompressed into a fresh scratch directory and read with my own extractors
(LaTeXML row id / tag / alttext scanner; a mystmd `"type":"math"` -> `"enumerator"` scan of the GTSAM page).
All invariance numbers in section 3 come from my own transcriptions of the printed equations, evaluated with
numpy / gtsam 4.3.0, using the NAMED objects (log, cay^-1, Frobenius norm), not the suite's surrogate. The
suite's code was used only as the mutation mechanism (`tests.test_mutation.run_mutant`) and for a seed sweep
of its own `error_invariance`.

**Scope of change since pass 4.** `git diff --stat dd60acf d939780` touches only `src/conventions.py`, and
only its module docstring (18+/9-). No table value, test, fixture or claim changed. C01..C15 are therefore
re-confirmed by a clean suite run plus spot checks, not re-derived from scratch.

---

## 1. Verdict table

| id | verdict | evidence this pass |
|---|---|---|
| M0-C01 | **CONFIRMED** | Own count, rai HTML stripped of script/style: 0 for each of the ten forbidden strings. |
| M0-C02 | **CONFIRMED** | Own extraction `S4.E19` (19): `r^R_ij ≜ (R_j − R_iΔR̄_ij) − ½(R_j + R_iΔR̄_ij)ξ_ij^`; (15) `ξ_ij ≜ J^R_ij δb_g`. Unchanged. |
| M0-C03 | **CONFIRMED** | Unchanged since pass 4 (5.3e-15); suite passes. |
| M0-C04 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C05 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C06 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C07 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C08 | **CONFIRMED** | Own extraction (24): `φ(X_k, ξ_k) ≔ (Υ_k C_k, Ad^∨_{C_k^-1}(b_k − γ_k^∨))` (right action). |
| M0-C09 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C10 | **CONFIRMED** | Own extraction: delama (34) contains `\log(` and `J_L(·)^{-1}`; rai (19), barfoot (25)/(28) none. |
| M0-C11 | **CONFIRMED** | Own extraction delama (34), GTSAM (21) `e_k = ξ̂_k^{-1}ξ_k = (Υ̂_k^{-1}Υ_k, β_k − β̂_k)`; the GTSAM page itself continues "It is left-invariant because (Zξ̂)^{-1}(Zξ) = ξ̂^{-1}ξ". Own invariance numbers in section 3. |
| M0-C12 | **CONFIRMED** | Unchanged; suite passes. |
| M0-C13 | **CONFIRMED** | Runtime: `Gal3(I,[1,2,3],[10,20,30],7).matrix()` col 3 = [10,20,30], col 4 = [1,2,3], (3,4) = 7; `ConstantBias([1,2,3],[40,50,60]).vector() = [1,2,3,40,50,60]`. |
| M0-C14 | **CONFIRMED** | Own runtime, 50 random trajectories (50..200 steps, random bias): max ‖X_j ⊖ W·φ(X_i)·U‖ = **3.2e-15** against `PreintegratedImuMeasurements.predict`. |
| M0-C15 | **CONFIRMED** | Own extraction of delama (18) `Γ_ij = [[I, gΔt, −½gΔt²],[0,1,−Δt],[0,0,1]]`; 2000 draws: max\|Γ − T(−dt)H′\| = **0** exactly; min max-abs\|Γ − H′T\|/(‖g‖dt²) = 0.58 (≥ 1/√3, never zero). |
| **M0-C19** | **REFUTED** | Runtime (section 2). |

**Counts: 15 CONFIRMED, 1 REFUTED (M0-C19), 0 UNTESTABLE.**

---

## 2. REFUTED: M0-C19

> *"In GTSAM 4.3.0, gtsam.NavState composition adds velocities without rotating them: (N1 * N2).velocity()
> == v1 + v2 for any NavState(R1, p1, v1), NavState(R2, p2, v2), consistent with velocity being a world-frame
> quantity (checked at runtime)."*

Runtime, gtsam 4.3.0 (`importlib.metadata.version("gtsam") == "4.3.0"`), 1000 random pairs (R uniform on
SO(3), p, v ~ 10·N(0, I)):

```
max |(N1*N2).velocity() - (v1 + v2)|     = 65.3        (claim)
max |(N1*N2).velocity() - (v1 + R1 v2)|  = 7.1e-15     (SE_2(3) law)
max |(N1*N2).position() - (p1 + R1 p2)|  < 1e-12
N1.compose(N2) identical to N1*N2
N1 = NavState(Rz(90 deg), 0, [1,0,0]), N2 = NavState(I, 0, [1,0,0]):  velocity [1, 1, 0]   (claim: [2, 0, 0])
```

NavState composition is the extended-pose product `(R1R2, p1 + R1p2, v1 + R1v2)`: the second velocity is
rotated by R1. The claim holds only in the special case R1 = I — which is exactly the case the table uses
(`W_ij = NavState(I, ½g dt², g dt)` on the LEFT, M0-C14), and is probably where the over-generalisation came
from. Nothing in the table records the general claim, so no table value is affected.

---

## 3. M0-F12: ruling **FIXED** (with new low residuals, filed as M0-F14)

The three defects pass 4 filed are gone: (i) the carve-out "pre-integrated increments … do not translate"
is removed, and the translated pairs for delama and GTSAM are now named as the group-valued pair
(Υ_k, Υ̂_k); (ii) "bias terms are held fixed" is now paired with an explicit statement that it is the
COMPONENT `log(ΥΥ̂^-1)` that is classified and that the full (34) is right-invariant only when the bias
moves under Eq. (24); (iii) the field is now explicitly about the error component, not the whole minimised
cost, and rai's full (20) is explicitly said to be non-invariant. Rule-of-the-brief sub-questions:

### (a) Internal consistency, and match with `tests/test_conventions.py::error_invariance`

Internally consistent: one rule (translate the named group-valued pair on one side, hold everything else
fixed, use the printed weight; bi-invariance excluded), and the consequences it states follow from it.
Row-by-row against the code:

| row | wording | code computes | match |
|---|---|---|---|
| rai | r^R of (19) inside τ_R‖r^R‖²_F of (20); R_i, R_j translate; ΔR̄_ij, ξ_ij fixed; scalar weight | evaluates (19) from its LaTeX on (R_i, R_j) → (SR_i, SR_j) / (R_iS, R_jS), ΔR̄, ξ fixed, W = c·I_9 | **exact** |
| delama | log(ΥΥ̂^-1)^∨ of (34), general SPD weight | vec(ΥΥ̂^-1 − I)ᵀ W vec(ΥΥ̂^-1 − I), W random SPD 25×25 | verdict-equivalent surrogate, not the named object |
| barfoot | cay^-1(CC̃_mᵀ), cay^-1(TT̃_m^-1) with W_m | vec(E − I) quadratic form; BOTH (25) and (28) executed on random SE(3) 4×4 elements (the rotation-averaging case is not drawn on SO(3)) | verdict-equivalent surrogate |
| gtsam | Υ̂^-1Υ of (21) | vec(Υ̂^-1Υ − I) quadratic form; β − β̂ not evaluated (held fixed ⇒ unchanged) | verdict-equivalent |

So "exactly" holds for rai and at the level of verdicts for the other three, not at the level of the
computed object. Verdict-equivalence is structural: on the invariant side the composed element E is
unchanged exactly, so any function of it is; on the other side E → SES^-1 and a generic SPD form of log(E),
cay^-1(E) or E − I all change (my own computations with the named objects, table below, give the same sides).
Not stated in the docstring; filed in M0-F14(iii).

The suite's own text has NOT been updated to the new definition: `error_invariance`'s docstring
(`tests/test_conventions.py:1545-1561`) still reads "Relative change of each row's **minimised cost** …
THE definition of `error_handedness` … a cost is left-invariant if it is unchanged when **every group-valued
state argument** is left-multiplied", the `_COST_WEIGHT` comment (l. 1533) says "The minimised cost each row's
error enters", and `decide_error_handedness` (l. 1604) says "cost invariance under group translation". The new
module docstring says the field "describes the error COMPONENT, not the whole minimised cost". M0-F14(ii).

### (b) Named components vs printed equations (own extraction)

| row | printed (own extractor) | named in docstring | ok |
|---|---|---|---|
| rai (19) | `r^R_ij ≜ (R_j − R_iΔR̄_ij) − ½(R_j + R_iΔR̄_ij)ξ_ij^` | r^R of Eq. (19) | yes |
| rai (20) | `F^Cay_ij = τ_{R,ij}‖r^R_ij‖²_F + τ_{v,ij}‖r^v_ij‖²₂ + τ_{p,ij}‖r^p_ij‖²₂` | τ_R‖r^R‖²_F of Eq. (20) | yes |
| delama (34) | `ε_k = (log(Υ_kΥ̂_k^-1)^∨, −J_L(log(Υ_kΥ̂_k^-1)^∨)^-1 Ad^∨_{Υ_k}(b_k − b̂_k))` | log(Υ_kΥ̂_k^-1) of (34); bias = "second component" | yes |
| delama (24) | `φ(X_k, ξ_k) ≔ (Υ_kC_k, Ad^∨_{C_k^-1}(b_k − γ_k^∨))` | "delama's own action (Eq. 24)" | yes |
| barfoot (25) | `min Σ_m cay^-1(CC̃_mᵀ)^∨ᵀ W_m cay^-1(CC̃_mᵀ)^∨, C ∈ SO(3)` | cay^-1(C C~_m^T) with W_m | yes |
| barfoot (28) | `min Σ_m cay^-1(TT̃_m^-1)^∨ᵀ W_m cay^-1(TT̃_m^-1)^∨, T ∈ SE(3)` | cay^-1(T T~_m^-1) | yes |
| gtsam (21) | `e_k = ξ̂_k^{-1}ξ_k = (Υ̂_k^{-1}Υ_k, β_k − β̂_k)` | Upsilon_hat^-1 Upsilon of (21) | yes |

Wording nit: the rule calls the second member of the pair "its estimate / prediction"; barfoot's C̃_m, T̃_m are
measurements (the pre-fix text said "estimate / measurement"), and rai's translated pair is (R_i, R_j), two states,
not (state, prediction). Verdict-neutral (rai reading B below). M0-F14(iv).

### (c) Own computation under the new wording

Relative change |c(translated) − c| / c, [min, max] over 1000 draws (S random in the group unless stated),
script `arb5.py` / `dlfull.py`:

| object | left | right | value |
|---|---|---|---|
| rai A: τ_R‖r^R‖²_F, (R_i,R_j) translate, ΔR̄, ξ fixed | [0, 1.2e-15] | [1.4e-4, 44] | **left** |
| rai B: translate R_j and its prediction R_iΔR̄ (ΔR̄ effectively conjugated), ξ fixed | [0, 1.2e-15] | [1.2e-4, 38] | left (same) |
| rai, counterfactual general SPD 9×9 weight (not printed) | [6.1e-4, 11] | [1.0e-3, 110] | none — the scalar τ_R is load-bearing, and is printed |
| delama: log(ΥΥ̂^-1)^∨ (gtsam `Gal3.Logmap`), SPD 10×10 | [2.1e-4, 82] | [0, 2.5e-15] | **right** |
| barfoot (25): cay^-1(CC̃_mᵀ)^∨, C on SO(3), SPD W_m | [6.4e-4, 30] | [0, 4.0e-14] | **right** |
| barfoot (28): cay^-1(TT̃_m^-1)^∨, SE(3), SPD W_m | [1.6e-4, 68] | [0, 9.3e-14] | **right** |
| gtsam (21): (Log(Υ̂^-1Υ), β − β̂), SPD 16×16, β fixed | [0, 1.7e-15] | [1.4e-3, 29] | **left** |

All four row values follow. The docstring's stated consequences:

| consequence in docstring | own check | ruling |
|---|---|---|
| delama full (34), right translation, b fixed: not invariant | [1.7e-5, 17] | true |
| delama full (34), right, b → Ad_{S^-1}b (Eq. 24 with γ = 0): invariant | [0, 4.0e-15] | true (J_L^-1 from gtsam's Logmap Jacobian at Exp(−x), checked vs finite difference, 5.7e-8) |
| rai full (20), v_i, p_i in R^3 fixed: not invariant either side | L [2.0e-6, 3.0], R [6.2e-4, 3.2] | true |
| "(only the yaw subgroup about gravity leaves it invariant)" | yaw S about g, **v, p fixed**: L [1.5e-4, 1.4] — NOT invariant; yaw S with v, p **rotated by S**: L [0, 1.1e-15] | **inaccurate as attached**: the parenthetical qualifies the "v_i, p_i in R^3 [held fixed]" reading, under which yaw does not leave (20) invariant; it needs v, p to rotate with S (the 4-DoF gauge). M0-F14(i). |

Seed sweep of the suite's own `error_invariance` (seeds 0..99): rai L ≤ 1.1e-15 / R ≥ 3.8e-3; delama
R ≤ 6.8e-16 / L ≥ 7.3e-3; barfoot R ≤ 7.9e-16 / L ≥ 6.3e-2; gtsam L ≤ 7.2e-16 / R ≥ 1.4e-2. No borderline
values, no bi-invariance.

### (d) Is any row still decided by an unstated special case?

No decisive one. The two conventions pass 4 called unstated — holding ΔR̄_ij, ξ_ij fixed (which makes rai
one-sided rather than bi-invariant) and rejecting bi-invariance — are now written in the rule, as is the per-row
component selection (the `assert key == "rai"` branch now corresponds to a named rai component). Remaining
unstated choices, all verdict-neutral by computation: the vec(E − I) surrogate for log / cay^-1; barfoot (25)
drawn on SE(3); the `weight_pat.startswith(r"\\tau")` scalar test (pass-4 L-3; the scalar reading IS decisive
for rai, see the counterfactual row, but the rule states it and the source prints τ_{R,ij}).

**Ruling: M0-F12 FIXED.** The definition now matches what decides every row, and every row follows from it
under my own computation. Residual low items are new and are filed as M0-F14, not as a failure of F12.

---

## 4. Mutation sweep (24 mutants; 22 not in `ALL_MUTANTS`)

Mechanism: `tests.test_mutation.run_mutant` (in-process, deep-snapshot restore asserted per mutant). Null run
clean before and after; no path in `UNVERIFIED_COMMENTARY`; `git status` clean afterwards.

| field | mutants | result |
|---|---|---|
| `rai.error_handedness` | "bi-invariant", "left" | killed, killed |
| `delama.error_handedness` | "bi-invariant", "right" | killed, killed |
| `barfoot.error_handedness` | "Right-invariant", "bi-invariant" | killed, killed |
| `gtsam.error_handedness` | "bi-invariant", "left-invariant " (trailing space) | killed, killed |
| `rai.eq.rotation_residual` (19) | "21a" (identical LaTeX restatement; in ALL), "15", "14" | killed ×3 (21a by the claim-text cross-check, the others by the C03 identity test) |
| `rai.eq.inertial_factor` (20) | "19" (in ALL), "22a" | killed ×2 |
| `delama.eq.error_local_coords` (34) | "31", "39", "33" | killed ×3 |
| `delama.exp_log_dependencies.error_local_coords.eq` | "31" | killed |
| `barfoot.eq.rotation_averaging_cost` (25) | "28", "26" | killed ×2 |
| `barfoot.eq.pose_averaging_cost` (28) | "25", "27" | killed ×2 |
| `gtsam.eq.error` (21) | "20", "34", "23" | killed ×3 |

**24/24 killed, 0 survivors.** In addition, with the full (non-stop-at-first) suite, the side-swap mutants are
killed by the decider-bearing row tests, not only by enum checks: rai → right: `test_rai_row_is_sourced`;
delama → none: `test_delama_row_is_sourced` + C11 test; barfoot → left: `test_barfoot_row_is_sourced`;
gtsam → right: C11 test + `test_gtsam_row_is_sourced`.

---

## 5. New findings

| id | severity | finding |
|---|---|---|
| **M0-F14** | low | Residuals of the F12 restatement, none changing a value: (i) the parenthetical "(only the yaw subgroup about gravity leaves it invariant)" is false under the condition it is attached to (v_i, p_i in R^3 held fixed: yaw changes (20) by up to 1.4 relative); it is true only if v, p rotate with S. (ii) The suite's own docstrings still state the superseded definition as "THE definition" — `error_invariance` (`tests/test_conventions.py:1545-1561`, "minimised cost", "every group-valued state argument"), `_COST_WEIGHT` comment (l. 1533), `decide_error_handedness` (l. 1604, "cost invariance"); `tests/test_mutation.py:10-11` repeats it historically. (iii) The code classifies vec(E − I) under a random SPD form (25×25 Gal(3), 16×16 SE(3)), and draws barfoot (25) on SE(3), rather than the named log / cay^-1 component on SO(3); verdict-equivalent (section 3c) but not "exactly what the code computes". (iv) "its estimate / prediction" omits barfoot's measurements C̃_m, T̃_m; for rai the translated pair is (R_i, R_j). Verdict-neutral. |
| — | info | M0-C19's statement is the R1 = I special case generalised; the table's only use of it (W_ij on the left, M0-C14) is in that special case and is correct. |

Open from earlier passes, not re-tested here: M0-F09, M0-F13.

---

Scripts (verifier scratchpad `s5/`): `ext.py`, `grepalt.py`, `gt.py` (extractors), `arb5.py` (section 3c),
`dlfull.py` (delama full (34)), `mut5.py` (section 4). C01, C13, C14, C15, C19 run inline.

Full suite on the clean tree after committing this report: see the last line.
