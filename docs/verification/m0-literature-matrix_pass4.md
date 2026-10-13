# M0 adversarial verification, pass 4: literature matrix

**Verifier:** independent pass 4. **Branch:** `m0-literature-matrix` at `8018c7b` (fix commit
`ceb6e1e`). **Date:** 2026-10-12.
**Under test:** `src/conventions.py` (including the new `error_handedness` definition in its
docstring), `docs/claims/M0.yaml` (M0-C01..C15), M0-C18 (stated in the dispatch brief only),
`tests/test_conventions.py`, `tests/test_mutation.py`, `tests/fixtures/`.

**Independence.** I did not read `docs/00_literature_matrix.md`, any `seeds*.yaml`, or anything
outside this worktree. I decompressed the fixtures into a fresh scratch directory and read them
with my own extractors: a LaTeXML `ltx_eqn_row` scanner (row id, printed tag, alttext), a prose
renderer (each `<math>` replaced by its alttext), and a mystmd `"type":"math"` -> `"enumerator"`
scan of the GTSAM page. All invariance numbers below come from my own transcriptions of the
printed equations, evaluated with numpy / gtsam 4.3.0. I used the suite's code only in two
places, and say so: the mutation harness (`tests.test_mutation.run_mutant`, used as the
mechanism) and a seed sweep of the suite's own `error_invariance`.

---

## 1. Verdict table

| id | verdict | attack (own extractor / own computation) |
|---|---|---|
| M0-C01 | **CONFIRMED** | Counts on the script/style-stripped rai HTML: 0 for each of `Galilean`, `SE_2`, `\mathrm{SE}`, `SE}_{2`, `extended pose`, `equivariant`, `Equivariant`, `Delama`, `Fornasier`, `Weiss`; `Barfoot` = 2; `Mahony` = 1. Table: `group "SO(3)"`, `element_order`/`tangent_order` None, v, p in R^3. |
| M0-C02 | **CONFIRMED** | Row `S4.E19.m1` (19): `r^R_ij ≜ (R_j − R_iΔR̄_ij) − ½(R_j + R_iΔR̄_ij)ξ_ij^`; (15) `ξ_ij ≜ J^R_ij δb_g`; the display after (10c) `δb ≜ b − b̄`. Signs and the perturbation reading hold. |
| M0-C03 | **CONFIRMED** | Own transcription of (22a)–(22c) vs (19), 9000 draws at scales 1, 1e-8, 1e4: max relative difference **5.3e-15**. |
| M0-C04 | **CONFIRMED** | (28) `x ≜ [{vec(R_i), v_i, p_i}_i, {b_m}_m, …, s]`, s last; (31) `s²=1, R_iᵀR_i = s²I_3, R_i^(a)×R_i^(b) = sR_i^(c)`. |
| M0-C05 | **CONFIRMED** | §VII-B prose: "`2×7×12×5=840` SDP solves", "at least `0.88`, and none of the corresponding `504` solves is certified. Adding B2 substantially reduces…", "the median gap is `2.8×10⁻⁵`". (32)–(35) labels as recorded. |
| M0-C06 | **CONFIRMED** | Prose, verbatim: "…contain products such as `R_iᵀv_j` and `R_iᵀp_j`. Rather than lifting these products directly, we use `τ‖r‖²₂ = τ‖R_i r‖²₂, R_i ∈ SO(3)`. **This equality requires an isotropic weight. We therefore approximate the three covariance blocks by scalar precisions** (23)". (23) is `τ⁻¹ = ⅓ tr Σ` for φ, v, p. The new polarity check kills "does not require an isotropic weight" (section 4). |
| M0-C07 | **CONFIRMED** | (3), (4), (5), (6) as recorded; own numeric check of the (5) inverse over 10 000 draws: max ‖X X⁻¹ − I‖∞ = **7.9e-15**. |
| M0-C08 | **CONFIRMED** | (1) `XY=(AB, a+Ad_A[b])`; sentence after (24), verbatim: "Then, φ is a transitive and free right group action of **G** on ℳ." (36)/(37) built from `I_10`/`0_{10×10}` blocks. |
| M0-C09 | **CONFIRMED** | Virtual velocity input `ν = 0_{3×1}` and the virtual time bias in the bias state; Q_d on an untagged "Define:" line of Algorithm 1. |
| M0-C10 | **CONFIRMED** | rai (19) and barfoot (25)/(28) contain no exp/log (only `Cay`/`cay⁻¹`); delama (31), (34) contain `log` and `J_L(·)⁻¹`; GTSAM (6) `Exp`, (42) `logmap`; rai (14) chordal on `Exp(ξ)`, (3) `Cay`. |
| M0-C11 | **CONFIRMED** | delama (34) `log(Υ_kΥ̂_k⁻¹)`, (39) `exp((J_Υ Δb̂)^)Υ̂` (left); GTSAM (21) `ξ̂⁻¹ξ = (Υ̂⁻¹Υ, β − β̂)`, (34) `Υ̂ Exp(Jδβ)` (right). Own invariance numbers in section 3 agree with both values. Caveat (does not refute the claim, which names the group composition): delama's FULL 20-vector (34) is right-invariant only if the bias transforms by the source's own action (24) — see section 3(b). |
| M0-C12 | **CONFIRMED** | Barfoot: "we consider log SVR of at least `5` to be rank 1"; Sec. 5: "Our continuous-time experiments had pose measurements only at the start, middle, and end". rai §VII-A: "number of eigenvalues larger than `10⁻⁶λ_max(Ẑ)`". Own anchor (1, 10^-5.5, 1e-9, 1e-12×7) in a random basis: log-SVR **5.500** (accept), rai rank **2** (reject). |
| M0-C13 | **CONFIRMED** | Runtime on 4.3.0: `Gal3(I,[1,2,3],[10,20,30],7).matrix()` column index 3 = `[10,20,30]`, index 4 = `[1,2,3]`; `Gal3.Hat(e_3..5)` → column 3, `e_6..8` → column 4, `e_9` → (3,4); `ConstantBias([1,2,3],[40,50,60]).vector() = [1,2,3,40,50,60]`; `integrateMeasurement([1,2,3],0,0.1)` gives ΔR = I, Δv = `[.1,.2,.3]`; GTSAM (38) `P_N` rows `(0,1,2,6,7,8,3,4,5)`; delama (18) `−½gΔt²`, GTSAM (15) `+½gΔt²`, GTSAM (3) `p₁+R₁p₂+t₂v₁`. |
| M0-C14 | **CONFIRMED** | Own sweep, 100 trajectories (n ∈ [50,200), Δt_ij = 0.5, random bias, ‖v_i‖ ∈ [0.11, 5]): worst ‖X_j ⊖ W·φ(X_i)·U‖ = **4.4e-16**; ‖X_j ⊖ W·X_i·U‖/(‖v_i‖Δt) ∈ [1 − 9e-16, 1 + 1.3e-15]. |
| M0-C15 | **CONFIRMED** | Own Γ from delama (18), 2000 draws (\|dt\| ≥ 0.05, ‖g‖ ∈ [1,20]): max\|Γ − T(−dt)H′\| = **0** exactly; min ‖Γ − H′T‖/(‖g‖dt²) = **1.0**. |
| **M0-C18** | **REFUTED** | Runtime (section 2). |

**Counts: 15 CONFIRMED, 1 REFUTED (M0-C18), 0 UNTESTABLE.**

---

## 2. REFUTED: M0-C18

> *"In GTSAM 4.3.0, gtsam.NavState.Logmap of NavState(I, p, v) returns [0, v, p]: the NavState
> tangent stores velocity before position, the same slot order as the Gal3 tangent
> (omega, nu, rho, alpha) (checked at runtime)."*

Runtime, gtsam 4.3.0 (`importlib.metadata.version("gtsam") == "4.3.0"`):

```
X = NavState(Rot3(), p=[1,2,3], v=[10,20,30])
NavState.Logmap(X)                    -> [0 0 0  1  2  3  10 20 30]   (claim predicts [0,0,0,10,20,30,1,2,3])
NavState().localCoordinates(X)        -> [0 0 0  1  2  3  10 20 30]
NavState().retract(e_3)               -> position [1,0,0], velocity [0,0,0]
Gal3.Logmap(Gal3(Rot3(), r=[1,2,3], v=[10,20,30], 0)) -> [0 0 0 10 20 30 1 2 3 0]
1000 random (p, v), scales 1e-3..1e3: max |Logmap − [0, p, v]| / max(1, |.|) = 0
```

The NavState tangent is (δθ, δp, δv): position before velocity, the OPPOSITE of the Gal3 tangent
slot order. This is exactly the O2 incompatibility the table records (`navstate_tangent_order
("dtheta","dp","dv")`, GTSAM doc (37) `n=(δθ,δp,δv)`, (38) `P_N`). Mutating the table to the
C18 order (`navstate_tangent_order = ("dtheta","dv","dp")` or the matching `navstate_tangent_slots`)
is killed by the suite at runtime (section 5).

---

## 3. Arbitration: `error_handedness` under the written definition

The definition now in the module docstring: *"the invariance of the minimised error / cost under
group translation … if the cost is unchanged when every group-valued state argument (the state
AND its estimate / measurement of the same state) is left-multiplied … evaluated on the source's
printed cost / residual WITH the weight the source prints … Relative measurements and inputs
(pre-integrated increments, bias terms) are not states and do not translate."*

All numbers are relative changes |c(translated) − c| / c, min / max over random draws, from my
own transcriptions (script `arb.py`, section 8).

### (a) rai: "left-invariant" is correct for ONE summand, not for the minimised cost of Eq. (20)

Source rows: (19) above; **(20)** `F^Cay_ij = τ_{R,ij}‖r^R_ij‖²_F + τ_{v,ij}‖r^v_ij‖²₂ + τ_{p,ij}‖r^p_ij‖²₂`;
(21b) `r^v_ij = R_iᵀ(v_j − v_i − g^W Δt_ij) − Δv̄_ij − A^{vg}δb_g − A^{va}δb_a`; (21c) likewise for p.

| cost evaluated | translation | min | max | verdict |
|---|---|---|---|---|
| τ_R‖r^R‖²_F (19 inside 20), ΔR̄, ξ fixed | R → S R | 0 | 1.0e-15 | invariant |
| same | R → R S | 9.2e-6 | 1.6e+2 | not invariant |
| same, but ΔR̄ → SᵀΔR̄S, ξ → Sᵀξ (increment conjugated) | R → R S | 0 | 2.6e-15 | invariant |
| **full F^Cay of (20)**, v, p fixed (they are R^3, not group-valued) | R → S R | 8.2e-6 | 6.0 | **not invariant** |
| full F^Cay, v → Sv, p → Sp | R → S R | 1.4e-4 | 6.0 | not invariant (g^W is fixed) |
| full F^Cay | R → R S | 7.8e-5 | 5.3 | not invariant |
| full F^Cay, yaw S about g, v, p rotated | R → S R | 0 | 1.2e-15 | invariant (the 4-DoF gauge) |

(2000 draws each.) Rulings on the brief's questions:

- **Is "left-invariant" correct?** Only for the rotation summand. The definition says "the
  minimised error / cost", and the minimised cost of rai is the whole of Eq. (20). With v_i, p_i
  in R^3 held fixed — which is what the definition's "group-valued state argument" clause and the
  table's own comment ("v_i and p_i never leave R^3") require — `r^v` and `r^p` change under R → SR,
  so F^Cay is neither left- nor right-invariant: the literal value is **"none"**. Even rotating
  v and p with S does not rescue it, because g^W is a constant; only the yaw subgroup about
  gravity leaves F^Cay invariant. The docstring escapes this by naming the summand explicitly
  ("rai's tau ||r^R||_F^2 of Eq. (20) left-invariant"), and the test hard-codes the same choice
  (`_COST_WEIGHT["rai"]` matches only the `\tau_{R,ij}‖r^R_ij‖_F^2` term; `_ERROR_EQ["rai"] =
  ("rotation_residual",)`). That is a per-row selection the written rule does not state.
- **Is it right that v_i, p_i and ΔR̄_ij do not translate?** v_i, p_i: yes under the rule as
  written (not group-valued), but then the full cost is not invariant (above). ΔR̄_ij: not
  translating under R → SR is physically right (a world-frame change leaves a body-frame
  increment unchanged). Under R → RS the physically consistent transformation conjugates the
  increment and ξ; then the rotation term is **bi-invariant** (2.6e-15), which
  `decide_error_handedness` rejects by fiat (`assert len(inv) < 2  # bi-invariance is not a value
  of the field`). So rai's "left-invariant" rests on two conventions — summand selection and the
  non-conjugated increment — that no rai text supports; rai's prose never uses "invariant" at all
  (0 occurrences).

### (b) delama, barfoot, gtsam under the same rule

| row, printed object | translation | min | max |
|---|---|---|---|
| barfoot (25) `cay⁻¹(C C̃_mᵀ)ᵀ W_m cay⁻¹(·)`, general SPD W_m | C, C̃ → S· | 3.0e-4 | 31 |
| same | C, C̃ → ·S | 0 | 3.1e-13 |
| gtsam (21) `(Log(Υ̂⁻¹Υ), β − β̂)`, general SPD 16×16 | left | 0 | 1.4e-15 |
| same | right | 9.8e-4 | 18 |
| delama (34), Gal(3) component `log(ΥΥ̂⁻¹)` only, SPD 10×10 | right | 0 | 2.3e-15 |
| same | left | 1.2e-3 | 28 |
| **delama (34), full 20-vector**, b, b̂ fixed (the rule's "bias terms do not translate") | right | 6.1e-4 | 24 |
| delama (34), full, b → Ad_{S⁻¹}b (the source's own action, Eq. 24) | right | 0 | 2.6e-15 |

(barfoot 1000 draws; delama, gtsam 500 draws; my J_L⁻¹ via gtsam's Logmap Jacobian at Exp(−x),
checked against a finite difference.) barfoot "right-invariant" and gtsam "left-invariant" hold
for the full printed object under any reading. delama "right-invariant" holds for the group
composition and for the full ε_k only when the bias moves under delama's action (24); under the
rule as written (bias fixed) the second component `−J_L(·)⁻¹ Ad_{Υ_k}(b_k − b̂_k)` changes, and the
full (34) is "none". The suite evaluates only the composition: `_ERROR_FN` executes `X Xh⁻¹` and
puts `vec(E − I)` in the quadratic form, discarding the bias component of (34). The suite's own
`error_invariance` is seed-robust (300 seeds: invariant side ≤ 1.3e-15, other side ≥ 1.9e-3 for
every row, no borderline values).

### (c) Is the definition applied consistently? **No.**

1. **The carve-out contradicts two rows.** "Pre-integrated increments … are not states and do not
   translate", yet the arguments the decider translates for delama and gtsam ARE pre-integrated
   increments and ARE the states of those sources: delama "Let … represent the state of our
   system. The preintegration matrix `Υ_k = (ΔR_k, Δv_k, Δp_k, …)`" (Eq. 19 `Υ_ij ≔ ∏ exp(…)`);
   GTSAM "The preintegration state consists of the accumulated Galilean motion and the physical
   six-axis IMU bias" (Eq. 1 `ξ_k = (Υ_k, β_k)`, Eq. 14 `Υ̂_{i,k+1} = Υ̂_{ik}V_k`). Read literally,
   nothing in delama (34) or gtsam (21) translates and both rows are trivially bi-invariant.
2. **"Bias terms do not translate"** makes delama's printed (34) not right-invariant (table above).
3. **Object selection is per row.** barfoot and gtsam are evaluated on the full printed
   cost / error; rai on one summand of (20); delama on one component of (34). Each choice is a
   hard-coded table entry in the test (`_ERROR_EQ`, `_ERROR_FORMS`, `_COST_WEIGHT`), and the rai
   branch is guarded by `assert key == "rai"`.
4. One decider function is applied to every row, which the commit claims, and the values it
   produces (rai left, delama right, barfoot right, gtsam left) are the values a consistent rule
   gives if the rule is restated as: *"the invariance of the cost restricted to its group-valued
   error component (rotation for rai, Gal(3) for delama/gtsam, SO(3)/SE(3) for barfoot), with
   every non-group quantity — R^3 states, biases, relative increments — held fixed"*. That
   restatement is what the code implements; the docstring does not say it, and in two places says
   the opposite. Filed as M0-F12.

---

## 4. Rulings on pass-3 findings

| finding | ruling | evidence |
|---|---|---|
| **M0-F10** (isotropy conclusion unread; always-true C06 line; delama.linearized truncation) | **FIXED** | (i) The pass-3 survivor (inverted conclusion) and three variants are in `PASS3_SURVIVORS` and killed. My own variants are killed too: **"…that equality does not require an isotropic weight — so the scalar precision is a tuning choice, not structurally forced"** and the same with the conclusion kept are both killed (`isotropy_reason states no decidable conclusion`), as are `R_i in GL(3)` and wrong product names. (ii) The always-true line `isotropy_required is ("isotropic weight" in group(2))` is gone; `isotropy_required` is now compared with a polarity decided from the source clause and its consequence sentence, and `isotropy_required = False` is killed. Its source half is necessarily a fixture constant (see section 6). (iii) `delama.linearized (35)-(36)` is killed. **(37) does belong**: my prose renderer shows, between "The linearized error dynamics about ε_k = 0 ∈ R^20 is then derived as follows" and the heading "IV-E Bias update and practical implementation", exactly the tags (35) `ε_{k+1} ≈ Â ε + B̂ η`, (36) `Â_{k+1}` and (37) `B̂_{k+1}`; (38) Φ_b sits after IV-E and is correctly rejected. Residual low survivors are filed as M0-F13. |
| **M0-F11** (error_handedness undefined; rai decided by a different rule) | **PARTIALLY FIXED** | Fixed: a definition is written in the table docstring; one function (`error_invariance` → `decide_error_handedness`) decides all four rows; rai "none" and "right-invariant", and every row's alternatives, are killed. Not fixed: the written definition is internally inconsistent with the delama and gtsam decisions, and rai's value holds only for a summand the rule does not select (section 3). The "different rule for rai" problem has moved from the decider into unstated per-row object selection. → M0-F12. |
| M0-F09 (bare parent labels) | open, confirmed | `rai.eq.rotation_residual_restated = "21"` and `barfoot.eq.cayley_inverse = "14"` still survive. Not re-filed. |

**Count check.** The brief and commit refer to "94/94" / "13 new mutants". `tests/test_mutation.py`
has 41 + 20 + 20 + 12 = **93** mutants (`ALL_MUTANTS`); all 93 are killed in the full run
(143 passed). The net increase from pass 3 is 13 (+2 −1 in `PASS2_SURVIVORS`, +12 new); "94" is
a miscount, not a hidden survivor.

---

## 5. Own mutation sweep (58 mutants; none of the (path, value) pairs is in `ALL_MUTANTS`)

Mechanism: the suite's `run_mutant` (in-process, deep-snapshot restore, restore asserted after
every mutant); I asserted programmatically that no pair is in `ALL_MUTANTS` and no path is in
`UNVERIFIED_COMMENTARY`; the null run is clean before and after; `git status` clean afterwards.

**Medium (46 mutants, 45 killed, 1 survived).** Killed, among others:
`rai.eq.inertial_factor ∈ {21a, 23, 12a}`; `rai.eq.isotropic_weighting = 20`; the two
"does not require an isotropic weight" reasons; `R_i in GL(3)`; wrong products in the reason;
`delama.linearized` dropping (35), pulling in (34), and citing (38) for the bias update;
`delama.exact + "error dynamics Eq. (35)"`; `delama.eq.error_local_coords ∈ {32, 29}`;
`delama.action_handedness = left`; `delama.bias_correction_side = right`; delama position-first
`tangent_slots`; `adjoint_time_row (0,0,1,0)`; swapped virtual inputs; `gtsam.eq.error = 22`;
the M0-C18 order in `gtsam.navstate_tangent_order` / `navstate_tangent_slots`; identity
`navstate_row_selection`, `navstate_block_map (0,3,6)`, identity `O2.P_N_rows`;
`gtsam.ctor_order = (R,v,p,t)`; swapped Gal3 tangent letters; `gtsam.bias_correction_side = left`;
`imu_sample_order (omega, acc)`; NavState argument swap in `W_ij`; `phi_dt` with `p − v·dt`;
rotation-first barfoot `tangent_slots`; barfoot threshold 6; `rotation_averaging_cost = 28`;
rai floor 1e-5; `median_gap 2.8e-6`; `certified 0/840`; rai bias order (b_a, b_g); `s` first in
the lifted vector; `s` dropped from the handedness constraint; invented B2 count; `J_L⁻¹`
dropped from `log_G.requires`; `O7.b` coefficient −0.5; `Ad_B[b]` in the semidirect law; a
linearized item moved into `rai.exact`.

| severity | field | mutant | ruling |
|---|---|---|---|
| medium | `delama.linearized[1]` | "bias update Eq. (39), **exact** in Delta b_hat" | **survivor**. (39) prints `≈`; the item contradicts its own field. `check_prose_equation_refs` checks the cited key, not the qualifier. → M0-F13 |

**Low (12 mutants, 8 killed, 4 survived).** Killed: `R_i^T r` in the identity, rai citation
date, "left group action" in the delama quote, "Lasserre" relaxation, O6 "by convention",
`velocity_space = Gal(3)`, `rai.state_dim = 10`, `package_version 4.2.0`.

| severity | field | mutant | ruling |
|---|---|---|---|
| low | `rai.isotropy_reason` | "…products, and **it is false that** that equality holds only for an isotropic weight — so … structurally forced, not a tuning choice" | **survivor**. The conclusion regex is unanchored on the left. → M0-F13 |
| low | `delama.linearized[0]` | "…(35)-(37) about epsilon_k = **1**" | **survivor**. The linearisation point is unread. → M0-F13 |
| low | `gtsam.retraction` | appended "; equivalently Exp(delta) X" | **survivor**. Appended free text (the table docstring already discloses this class). |
| low | `delama.linearized[0]` | "(35), (36), (37)" spelled out | **equivalent** (control; survives as it should). |

Totals: **58 mutants, 53 killed, 4 genuine survivors (1 medium, 3 low), 1 equivalent.**

---

## 6. Tests that cannot fail, and tests leaning on the test author's choices

### Cannot fail under any table mutation (fail only if a fixture or the library changes)

- **C06 source half.** `necessity[clause]` and `consequence` are computed from the fixture alone,
  so `forced` is the constant True; `assert clause in necessity` and the consequence test can only
  fail on a fixture change. This is inherent to "the source decides" and acceptable; the
  table-side assertions (`concl` polarity, `isotropy_required is forced`) can fail and do.
- **C15 reversed-product half** (`reversed_err > bar`), unchanged from pass 3 (deliberate, DERIVED).
- **SENTINEL_EVIDENCE prose patterns** for `delama.eq.continuous_time_ins`, `barfoot.eq.sparsity`,
  `barfoot.eq.licq_failure`: still constants; only the `== COULD_NOT_EXTRACT` half reads the table.
- **O7 block**: the `X_i,X_j\in\mathrm{SE}_2(3)` raw-page check and
  `localCoordinates(...).size == Gal3.Dim() − 1` read no table value.
- **Instrument tests** `test_fixture_provenance`, `test_equation_parser_never_invents_a_label`
  (the latter still does not probe bare parent labels, M0-F09).

### Lean on the test author's choices rather than a source

- **L-1 `error_invariance` object selection** (`_ERROR_EQ`, `_ERROR_FORMS`, `_COST_WEIGHT`): which
  summand / component of the printed cost is evaluated is chosen per row by the test author; the
  rai branch is `assert key == "rai"`. Decisive for rai and delama (section 3).
- **L-2 bi-invariance rejected by fiat** (`assert len(inv) < 2`): with a conjugated increment rai's
  rotation term is bi-invariant; the field has no such value, so the convention decides.
- **L-3 `scalar = weight_pat.startswith(r"\\tau")`**: whether a weight is scalar is keyed on the
  test's own regex string, not on a parse of the source weight.
- **L-4 `_LINEARIZED_PRESENTATION`** has one entry (delama). gtsam's "transition Eqs. (25)-(29)"
  and rai's linearized items are not block-checked (`prose = ""` for gtsam).
- **L-5 C06 vocabulary**: the polarity table knows four phrases and the conclusion regex accepts
  exactly two hard-coded phrasings; a correct reason in other words fails, and a left-prefixed
  negation passes (M0-F13).
- Pass-3 L-2..L-6 (`_BIAS_ARGUMENT` text order, `_overlap` stems, one-sided `EQ_SIGNATURES`, C01
  None-only scans, `assert_claim_statement`) are unchanged.

---

## 7. New findings

| id | severity | finding |
|---|---|---|
| **M0-F12** | low–medium | The written `error_handedness` definition does not match what decides the rows. (i) Its carve-out "pre-integrated increments, bias terms … are not states and do not translate" is contradicted by delama and GTSAM, whose error arguments Υ_k, Υ̂_k are pre-integrated increments and are those sources' states (both quoted in section 3c); (ii) under "bias terms do not translate" delama's full printed (34) is not right-invariant (min 6.1e-4, max 24); (iii) rai's "minimised cost" (20) is not left-invariant with v, p in R^3 fixed (min 8.2e-6, max 6.0), so "left-invariant" holds only for the τ_R‖r^R‖²_F summand the docstring names by hand. The code implements a consistent rule — group-valued error component only, all non-group quantities fixed — but the docstring does not state it. Fix: restate the definition that way (and say that relative increments are held fixed under both translations, which is what makes rai one-sided rather than bi-invariant), or record rai as "none" under the full-cost reading. No claim depends on rai's value. |
| **M0-F13** | low | Four survivors of plausible prose corruptions: `delama.linearized` second item "exact in Delta b_hat" (medium-plausibility; the qualifier is unread), the linearisation point "epsilon_k = 1", a left-prefixed negation inside `rai.isotropy_reason`'s necessity clause, and appended text in `gtsam.retraction`. |

---

## 8. Independent field checks (≥15 required; 90+ done, none through the suite's parsers)

All match the table.

- **rai** (own row scan): `cayley 3`, `cayley_right_jacobian 4`, `qcqp 5`, `sdp 6`,
  `chordal_residual 14`, `xi_definition`/`bias_correction 15` (`R_iᵀR_j = ΔR̄ Cay(ξ) Cay(−δφ)`, correction
  on the right), `cayley_determinant 17`, `denominator_clearing 18`, `rotation_residual 19`,
  `inertial_factor 20` (new; `F^Cay = τ_R‖r^R‖²_F + …`), `21a/21b/21c`, `22a/22b/22c`,
  `isotropic_weighting 23`, `u_definitions 26` (26a/26b), `lifted_vector 28`, `lifting_constraints 30`,
  `homogeniser 31`, `state_set 36`; untagged `b ≜ [b_gᵀ b_aᵀ]ᵀ` → `bias_order (b_g, b_a)`; the
  "prolonged stationary segment, leaving the heading unobservable" cause; ablation figures; rank rule.
- **delama**: `1, 2, 3, 4, 5, 6, 8, 9, 10, 17, 18, 19, 22, 24, 26, 29, 30, 31, 34, 35, 36, 37, 39` as
  recorded; (38) is Φ_b; right action sentence; `inverse_formula`; `semidirect_product_law`.
- **barfoot**: `5, 8, 10 ([ρ;φ]), 11, 14b, 15, 17, 18, 19, 20, 25, 28`; `14a` from the PDF excerpt
  (line 30 `(14a)`, line 33 `,(14b)`); log-SVR ≥ 5 and sparsity prose.
- **gtsam doc** (own enumerator scan, 49 equations): `1` (β = [b_a; b_ω]), `2`, `3`, `4`, `5`, `6`, `9`,
  `10`, `12`, `14`, `15`, `16`, `17`, `20`, `21`, `22`, `24`, `25`, `29`, `34`, `35`, `36`, `37`, `38`,
  `42`, `47` all as recorded.
- **gtsam runtime**: O1, O3, O4 as in section 1; retraction `X.retract(δ)` vs `X·Exp(δ)` 0.0, vs
  `Exp(δ)·X` 0.25; Gal3 `Hat` columns; NavState Logmap / localCoordinates / retract (section 2).

Scripts (verifier scratchpad, `scripts4/`): `ext.py` (row/prose extractor), `gt.py` (GTSAM
enumerator scan), `arb.py` (all invariance numbers of section 3), `mut4.py` (58 mutants with the
not-in-`ALL_MUTANTS` and restore assertions). C03, C07, C12, C14, C15, C18 and the runtime checks
were run inline.

After committing this report I ran the full suite on the clean tree:
`pytest -q` → **143 passed**.
