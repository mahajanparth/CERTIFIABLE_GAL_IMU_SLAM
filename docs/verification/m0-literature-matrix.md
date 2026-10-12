# M0 adversarial verification — literature matrix

**Verifier:** independent pass. **Branch:** `m0-literature-matrix`. **Date:** 2026-10-12.

**Independence constraint honoured.** `docs/00_literature_matrix.md` was *not* read. Every
entry below was checked against primary sources fetched fresh in this session:

| source | fetched | sha/size |
|---|---|---|
| Rai, Xu, Chen, Rosen — arXiv:2609.38048v1 | `https://arxiv.org/html/2609.38048v1` | 462 407 B |
| Delama, Fornasier, Mahony, Weiss — arXiv:2411.05548v8 | `https://arxiv.org/html/2411.05548v8` | 641 374 B |
| Barfoot, Holmes, Dümbgen — arXiv:2308.12418v2 | HTML **and** `arxiv.org/pdf/2308.12418v2` | 650 414 B / PDF |
| GTSAM Galilean IMU doc | `https://borglab.github.io/gtsam/galileanimufactor/` | 1 692 613 B |
| GTSAM runtime | `import gtsam` → `4.3.0` | installed wheel |

Equation text was extracted from the LaTeXML `alttext` attributes (i.e. the authors' own
LaTeX), not from a rendered glyph stream, so every equation number below was read off the
source's own `ltx_tag_equation` next to the source's own LaTeX.

**Fixture integrity.** `tests/fixtures/rai_2609.38048v1.html.gz` decompresses to a file
**byte-identical** to the live fetch (456 935 B both; `fx == live` → `True`). The archived
fixture has not been doctored.

---

## Verdict table

| id | verdict | the attack that was attempted |
|---|---|---|
| M0-C01 | **CONFIRMED** | Re-ran the term scan against the **live** source, not the fixture; compared fixture to live byte-for-byte; checked that "Barfoot" is really ref [6] and "Mahony" really ref [3]. |
| M0-C02 | **CONFIRMED** | Pulled Eq. (19) and Eq. (15) LaTeX verbatim; hunted for a `+1/2`, a missing minus, or `b_g` in place of `δb_g`. Found the paper's own sentence "Using δb_g = b_g − b̄_g". |
| M0-C03 | **CONFIRMED (strengthened)** | Proved it *symbolically* in sympy over free 3×3 matrices — stronger than the claim. Then re-ran numerically with my own quaternion SO(3) sampler (not scipy) at 1e5 draws incl. boundaries. Tried to break the `1e-12` bar: **succeeded by rescaling** (see Finding F-3). |
| M0-C04 | **CONFIRMED** | Read Eq. (28) and Eq. (31) LaTeX directly; checked `s` really is last and that all three constraint families are in (31). |
| M0-C05 | **CONFIRMED** | Read Eqs. (32)–(35) tags **and Table I**, specifically looking for a published per-edge count for B2 that would make `"could not extract"` a shortcut. There is none — Table I has no cardinality column and Eq. (34) is tagged `(B2)` with no count. Re-read §VII-B for all five ablation figures. |
| M0-C06 | **CONFIRMED** | Located the unnumbered display preceding Eq. (23) and the sentence "This equality requires an isotropic weight." Checked Eq. (23) is the scalar-precision definition and not something else. |
| M0-C07 | **CONFIRMED** | Read Eqs. (3)–(6) LaTeX. Checked the Ad last row is `[0,0,0,1]`. **But see Finding F-5**: the `inverse_formula` field is *claimed* and *not tested*. |
| M0-C08 | **CONFIRMED** | Found the sentence verbatim: *"Then, φ is a transitive and free **right** group action of **G** on ℳ."* Checked `R^20` via `J_ξ0 = I_20`, Eqs. (36)–(38) 10×10 blocks. |
| M0-C09 | **CONFIRMED** | Found `b_k = (b_ωk, b_ak, b_νk, b_ρk) ∈ R^10` and *"two additional virtual biases: a velocity bias … and a time bias"*. Confirmed `Q_d` really is an unnumbered `Define:` line of Algorithm 1 (`0_{4×1}` blocks in both halves). |
| M0-C10 | **CONFIRMED** | Checked Eqs. (31) and (34) both contain `log(·)` **and** `J_L(·)^{-1}`; checked rai (3)/(14) remove Exp/Log and barfoot (14)/(15) replace exp; checked gtsam (6) and (42) (a `logmap`). |
| M0-C11 | **CONFIRMED** | Delama (34) uses `Υ_k Υ̂_k^{-1}` (invariant under *right* translation) and (39) applies `exp(·)` on the **left**. GTSAM (21) is `ξ̂^{-1}ξ` with the doc's own words "It is left-invariant"; (34) says "This correction is on the right." Corroborated by the GTSAM doc's own cross-reference to Delama. |
| M0-C12 | **CONFIRMED** | Found the log-SVR definition verbatim in Barfoot §3 prose, and the Fig. 12-vs-Fig. 9 sparsity passage in §5. Confirmed both are unnumbered. Confirmed rai §VII-A uses an *eigenvalue* count at `1e-6·λ_max` — a genuinely different rule. **But see F-6**: `rai.rank1_criterion["statistic"]` is untested. |
| M0-C13 | **CONFIRMED** | Re-ran all seven against installed gtsam 4.3.0 with freshly written code; derived `P_N`'s row selection from the doc's own Eq. (38) *matrix* rather than from the table. Additionally **verified the O7 `reason` numerically** — M0 only substring-matches it (F-4). |
| M0-C14 | **CONFIRMED (qualified)** | Read Eqs. (15)–(17) verbatim. Swept **300 random trajectories**: worst `err_correct = 7.1e-15` (bar 1e-12 — robust). **Broke the second half**: `err_wrong` is exactly `‖v_i‖·Δt_ij`, so it drops below `1e-2` at `Δt_ij ≤ 0.09 s` with the test's own minimum velocity. See F-1. |
| M0-C15 | **CONFIRMED (qualified)** | Re-derived Eq. (18) from source LaTeX with my own 5×5 builder; `Γ − T(−dt)H′ ≡ 0` bit-exactly over 20 000 draws incl. `dt = 1e6` and `dt = 1e-300`, and symbolically. **Broke the `>1e-6` reversed-product bar**: the reversed error is exactly `‖g‖dt²`, below `1e-6` for `dt ≤ 3.2e-4 s`. See F-2. |
| **M0-S01** | **REFUTED** | See below. Contradicted verbatim by the source on both halves. |

**Counts: 15 CONFIRMED (2 qualified), 1 REFUTED, 0 UNTESTABLE.**

---

## REFUTED: M0-S01

> *"Delama et al. define φ as a LEFT group action of G on M, consistent with G being the
> left-trivialized tangent group of Gal(3); the navigation error log(Υ_k Υ̂_k^{-1}) is
> correspondingly left-invariant."*

Both assertions are false. Only the parenthetical about `G` is correct.

**Evidence 1 — the action.** arXiv:2411.05548v8, §IV-C, the sentence immediately following
Eq. (24), quoted verbatim from the source HTML:

> `Then, $\phi$ is a transitive and free right group action of $\mathbf{G}$ on $\mathcal{M}$ .`

The companion input action is likewise stated as right: *"Then, ψ is a right group action of
**G** on 𝕃"* (after Eq. 26). The string `left group action` occurs **zero** times in the
document; `right group action` occurs twice.

**Evidence 2 — the error.** Eq. (34) reads

```
ε_k = ( log(Υ_k Υ̂_k^{-1})^∨ ,  −J_L(log(Υ_k Υ̂_k^{-1})^∨)^{-1} Ad^∨_{Υ_k}(b_k − b̂_k) )
```

`Υ Υ̂^{-1}` is invariant under **right** translation (`(ΥS)(Υ̂S)^{-1} = Υ Υ̂^{-1}`) and is
*not* invariant under left translation. It is therefore right-invariant, which is the
textbook definition. This is independently corroborated by a *second* primary source: the
GTSAM Galilean doc, §1, says in its own words

> "Delama et al.'s related construction instead uses the navigation error `Υ Υ̂^{-1}`, which
> is **right-invariant**, together with a left-applied first-order correction."

**Evidence 3 — the stated justification is a non-sequitur.** `G` *is* the left-trivialized
tangent group `Gal(3) ⋉ gal(3)` (abstract and §II-B confirm this). But left-trivialization of
the tangent bundle says nothing about the handedness of a *state action* on a different
manifold `ℳ = Gal(3) × R^10`. The claim derives "left action" from a property that does not
imply it.

**Evidence 4 — it contradicts the project's own claim set.** M0-S01 is absent from
`docs/claims/M0.yaml` (which contains only `M0-C01`…`M0-C15`) and directly contradicts
`M0-C08` (`action_handedness == "right"`) and `M0-C11` (`error_handedness ==
"right-invariant"`), both of which I verified against the source. The table is right and
M0-S01 is wrong.

---

## Independent re-implementations

### R-1 — Rai Eq. (22b) ≡ Eq. (19), proved symbolically

Transcribed from the source LaTeX of `S4.E22` / `S4.E19` only, with my own skew operator and
no reuse of `src/` or `tests/`:

```
Γ   = (J^R b̄_g)^              [22a]
R̆   = ΔR̄ (I − Γ/2)            [22a]
Y   = R_i ΔR̄ (J^R b_g)^        [22c]
Y′  = R_j (J^R b_g)^            [22c]
22b = R_j(I + Γ/2) − R_i R̆ − ½(Y + Y′)
19  = (R_j − R_i ΔR̄) − ½(R_j + R_i ΔR̄)(J^R(b_g − b̄_g))^
```

`sympy.expand(22b − 19)` is the **zero matrix** entrywise, over free 3×3 symbols for
`R_i, R_j, ΔR̄, J^R` and free 3-vectors `b_g, b̄_g`. No orthogonality, no small-angle
assumption, no chart. The by-hand reason: `ξ^ = B − Γ` by linearity of `(·)^`, so
`22b = (R_j − R_iΔR̄) − ½(R_j + R_iΔR̄)(B − Γ)`. The claim is therefore *stronger* than
stated — exact, not merely "to 1e-12".

Randomized confirmation with an **independent** quaternion SO(3) sampler (not
`scipy.Rotation.random`), 5 × 20 000 = 10⁵ draws:

| regime | max abs diff | max rel diff |
|---|---|---|
| nominal | 5.3e-15 | 1.2e-11 |
| near-zero bias (1e-8) | 4.4e-16 | 1.1e-11 |
| **huge J and bias (1e4)** | **4.8e-07** | 4.2e-11 |
| `b_g ≈ b̄_g` (ξ → 0) | 6.7e-16 | 2.6e-11 |
| `b_g == b̄_g` exactly | 3.2e-15 | — |

### R-2 — Delama Eq. (18) and the O7 reason

Eq. (18) transcribed literally from `S4.EGx9`:
`Γ_ij = [[I₃, gΔt, −½gΔt²], [0, 1, −Δt], [0, 0, 1]]`. Built with my own `mat5`.

* `Γ − T(−dt)·H′` is **exactly** the zero matrix, symbolically and over 20 000 numeric draws
  including `dt ∈ {1e-300, 1e-12, ±1e6}`.
* `Γ − H′·T(−dt)` is symbolically `+g·dt²` in the position column and zero elsewhere, so its
  norm is exactly `‖g‖dt²`.
* **Physical consequence of the O7 `reason`, which M0 never checks numerically:** over
  20 000 random `(R_i, v_i, p_i, ΔR, Δv, Δp, g, dt)`, the Gal(3) route
  `Γ_ij(−½gdt²) · X_i(t=0) · U(dt)` and the SE₂(3)/NavState route
  `p_j = +½gdt² + (p_i + v_i dt) + R_iΔp`, `v_j = gdt + v_i + R_iΔv` agree to **1.1e-13**.
  The `t₂v₁` explanation is correct, and using `+½gdt²` inside the Gal(3) left factor is off
  by exactly `g·dt²` (verified: `[0,0,−4.8069] == g·0.7²`). **The reason string is true.**

### R-3 — gtsam 4.3.0 runtime, freshly written

```
Gal3(I, [7,8,9], [−1,−2,−3], 4.5).matrix(): col3 = [−1,−2,−3] (3rd arg), col4 = [7,8,9] (2nd arg)   → O1 ✓
Gal3.Hat(e_i): i∈{3,4,5} → column 3 ;  i∈{6,7,8} → column 4 ;  i=9 → (3,4)                          → O2 ✓
NavState.retract(e_i):   i∈{3,4,5} moves POSITION ; i∈{6,7,8} moves VELOCITY                        → O2 ✓
P_N from the doc's own Eq. (38) matrix  →  rows (0,1,2,6,7,8,3,4,5)                                 → O2 ✓
ConstantBias([1,2,3],[40,50,60]).vector() == [1,2,3,40,50,60]                                       → O3 ✓
integrateMeasurement([1,2,3], 0, 0.1) → ΔR = I, Δv = [0.1,0.2,0.3]                                  → O4 ✓
PreintegrationParams.MakeSharedU(9.81).n_gravity == [0,0,−9.81]
```

`P_N` second-order agreement swept over 50 random base points × 4 step sizes: at `eps = 1e-6`
the permuted map agrees to `4.9e-12` (bar `1e-9`) while the un-permuted selection is wrong at
`7.3e-6` (bar `1e-7`). At `eps = 1e-3` the second-order bar correctly fails — the test's
choice of `eps = 1e-6` is appropriate, not tuned to hide anything.

---

## Findings

### F-1 (M0-C14) — `err_wrong > 1e-2` is a fixture artifact, not a property

`err_wrong = ‖X_j.localCoordinates(W·X_i·U)‖` equals **exactly `‖v_i‖·Δt_ij`**. Measured:

| Δt_ij | ‖v_i‖ | err_correct | err_wrong | |
|---|---|---|---|---|
| 0.50 | 0.110 | 3.0e-16 | 5.50e-02 | |
| 0.10 | 0.110 | 2.9e-17 | 1.10e-02 | |
| **0.05** | **0.110** | 2.0e-17 | **5.50e-03** | ← claim's `> 1e-2` is **false** |
| **0.02** | **0.110** | 4.8e-17 | **2.20e-03** | ← false |
| **0.01** | **0.110** | 1.6e-17 | **1.10e-03** | ← false |

Both configurations satisfy the *test's own* guards (`‖v_i‖ > 0.1`), so the only thing
keeping the assertion alive is the test additionally asserting `dt_ij ≈ 0.5`. The claim
statement and its `domain:` field pin neither quantity.

**Is the claim still meaningfully testable? Partly yes.** The substantive half — that the
endpoint relation is `W·φ_Δt(X_i)·U` rather than `H X D` — is robust: over **300 random
trajectories** (varying `n`, `h`, acc, gyro, bias, `X_i`) the worst `err_correct` was
**7.1e-15**, three orders inside the `1e-12` bar. The disambiguating half is testable *only
once `Δt_ij` and `‖v_i‖` are pinned*, which the test does and the claim does not.

**The hint's quoted figures are not reproducible and should be deleted.** Running the
*exact* fixture in `tests/test_conventions.py` gives `err_correct = 4.4867e-16` and
`err_wrong = 0.3536` — matching the implementer's `4.5e-16` / `0.354`, **not** the hint's
`9.4e-17` / `0.229`. `0.3536 = ‖[0.5,−0.4,0.3]‖ × 0.5` exactly; the hint's `0.229` implies
`‖v_i‖ ≈ 0.458`, i.e. a different trajectory. Verdict: **CONFIRMED, not UNTESTABLE** — the
margin is real (3 orders on the half that carries the physics), but the hint's numerals are
decoration and the second bar needs its fixture written into the claim.

### F-2 (M0-C15) — `reversed_err > 1e-6` is likewise fixture-bound

`‖Γ − H′T‖ = ‖g‖dt²` exactly. With `g = 9.81`, this falls below `1e-6` for any
`dt ≤ 3.2e-4 s` — i.e. for a 10 kHz IMU sample interval, which is not pathological. The
test survives only because its own loop draws `dt ∼ U(0.05, 2.0)` and `‖g‖ ∈ [1,20]`, giving
a floor of `2.5e-3`. The claim's `domain:` says only "`dt != 0`", which does not support the
`1e-6` bar. The *mathematical* content (`H′T ≠ Γ` for `dt ≠ 0`) is correct, and the forward
half is bit-exact under every stress I applied, so the claim stands.

### F-3 (M0-C03) — the `1e-12` bar is an absolute tolerance on an unbounded quantity

At `J^R` and `b` scaled by `1e4` the max absolute difference is **4.8e-07**, violating the
claim's stated `< 1e-12`, while the max *relative* difference stays at machine level
(4.2e-11). The claim's `domain:` explicitly says "holds for arbitrary 3×3 J^R and arbitrary
b_g", which the absolute bar does not. Since I proved the identity *symbolically*, the claim
is CONFIRMED on substance; the tolerance should be relative, or the domain should say
"O(1)-scaled draws".

### F-4 (M0-C13 / O7) — the `reason` string is substring-matched only

`test_seven_ordering_incompatibilities_against_installed_gtsam` asserts only
`"t2*v1" in reason`, `"eq 3" in reason`, `"not a contradiction" in reason`. Mutating
`INCOMPATIBILITIES["O7"]["a"]["says"]` to `"nonsense but truthy"` leaves the suite green.
**I verified the physical content myself** (R-2 above): it is correct to 1.1e-13 over 20 000
draws. No defect in the table — a gap in the test.

### F-5 (M0-C07) — a claimed fact that no test can falsify

M0-C07 asserts "inverse at eq '5' equal to `(A^T, −A^T a, −A^T(b − c a), −c)`". The test
verifies that formula **by hard-coding it in the test body** and checking `X @ X_inv == I`.
It never reads `CONVENTIONS["delama"]["inverse_formula"]`. Mutation confirms:

```
CONVENTIONS["delama"]["inverse_formula"] = "X^-1 = (A, +A a, +A (b + c a), +c)  [WRONG]"
→ 16 passed
```

The table field happens to be correct (Eq. 5 verified against source), but M0 cannot tell.

### F-6 — mutation testing: 40 of 41 corrupted fields survive the suite

Harness validated with four positive controls (`rai.eq.rotation_residual`,
`delama.eq.inverse`, `gtsam.bias_order`, `O7.a.gravity_position_coefficient` — all killed).
Surviving mutants include fields named in claim statements or in the sources:

`delama.inverse_formula`, `delama.semidirect_inverse_law`, `delama.ctor_order`,
`delama.element_slots`, `delama.eq.{exp,left_jacobian,endpoint}`,
`rai.eq.{cayley_determinant,u_definitions,state_set,lifting_constraints}`,
`rai.rank1_criterion.{kind,statistic,relation}`, `rai.tightness_degrades_with`,
`rai.ablation.instance`, `barfoot.eq.{cayley,se3_element,shor_relaxation}`,
`barfoot.element_order`, `barfoot.rank1_criterion.{kind,relative_floor}`,
`barfoot.tightness_degrades_evidence`, `gtsam.retraction`,
`gtsam.eq.{residual,rotating_frame,adjoint,held_input_composition,bias_lift_E}`,
`{rai,barfoot,gtsam}.state_dim`, `O1.runtime_check`, `O2.P_N_blocks`,
`O3.a.eq`, `O4.b.eq`, `O7.a.says`.

Note in particular that `O3.a.eq` and `O4.b.eq` can be set to any digit string: the
structural guard `test_every_eq_entry_is_a_label_or_the_exact_sentinel` checks the *format*
(`^\d+[a-z]?$`) and nothing else, so it cannot distinguish `"19"` from `"91"`. **I checked
every one of these by hand against the sources (see the entry-by-entry table below); all are
correct.** The exposure is that nothing in the repo would notice if they stopped being.

### F-7 — assertions that cannot fail

1. `tests/test_conventions.py` (M0-C11): `assert "error_local_coords" != "bias_correction"`
   compares two **string literals written in the test source**. It is `assert "a" != "b"`.
   It is labelled an "eq-label collision guard" and guards nothing.
2. (M0-C02) The "numerical anchor" builds `pieces` from the test's own hard-coded
   expressions and sums them with the table's coefficients; the preceding
   `assert rr["terms"] == [(+1,"R_j"), (-1,"R_i dRbar"), (-0.5,…)]` already pins every
   coefficient, so `from_table == literal` is forced. The anchor adds zero falsification
   power.
3. (M0-C06) The anisotropic-weight anchor (`r^T W r ≠ (Rr)^T W (Rr)` for SPD non-scalar `W`)
   is a textbook fact about quadratic forms. It reads *nothing* from the table and cannot
   fail. It does not decide the word "structurally forced"; it only restates it.
4. (M0-C04) The improper-rotation anchor is likewise closed-form: for
   `improper = R·diag(1,1,−1)`, each cyclic violation is **identically 2** for *any* `R`, so
   `assert min(violations) > 1.0` cannot fail. It reads nothing from the table.
5. (M0-C09) `labels = np.array(["b_omega"]*3 + ["b_a"]*3 + ["b_nu"]*3 + ["b_rho"])` is
   constructed **by the test**, then the test asserts the table's slices index into it
   correctly. Given the literal `bias_virtual_slices` assertion three lines above, this is
   circular.
6. Forced-by-construction assertions, each already implied by a literal equality earlier in
   the same test: `ab["decisive_subset"] in rc`; `rai["lifted_vector"][-1] == ("global","s")`
   and the `.count("s") == 1`; `sum(dim for _, dim in group_dim_components) == 20` and
   `len(block_row_dims) == 2`; `d["manifold"] != d["group"]`; `d["citation"] != g["citation"]`;
   `d["error_handedness"] != g["error_handedness"]`; `d["bias_correction_side"] !=
   g["bias_correction_side"]`; `g["bias_dim"] != d["bias_dim"]`; `rc != bc`;
   `sign_gal3 != sign_navstate`; `d["tangent_order"] != g["tangent_order"]`;
   `widths == [3,1] and sum(widths) == 4`; `assert dt != 0.0` inside a loop that draws
   `dt = U(0.05,2.0)·±1`.
7. (M0-C01) The `absent` tuple contains both `"\mathrm{SE}"` and `"\\mathrm{SE}"`; the second
   string **contains** the first, so its count is forced to 0 whenever the first is 0. It
   cannot independently fail.

---

## Entry-by-entry equation-number audit

Every `eq` value in `CONVENTIONS` and `INCOMPATIBILITIES` was opened in the source. **All
correct.** Highlights of what could have gone wrong and did not:

**rai (arXiv:2609.38048v1)** — `cayley 3` ✓, `cayley_right_jacobian 4` ✓ (`J_Cay`), `qcqp 5` ✓,
`sdp 6` ✓, `chordal_residual 14` ✓ (Frobenius residual), `xi_definition 15` ✓, `cayley_determinant 17` ✓,
`denominator_clearing 18` ✓, `rotation_residual 19` ✓, `rotation_residual_restated 21a` ✓,
`velocity_residual 21b` ✓, `position_residual 21c` ✓, `lifted_rotation_residual 22b` ✓
(verified against the surrounding prose that 22a is `Γ/R̆`, 22b the residual, 22c the `Y`s),
`lifting_definitions_Y 22c` ✓, `isotropic_weighting 23` ✓, `u_definitions 26` ✓,
`lifted_vector 28` ✓, `lifting_constraints 30` ✓, `homogeniser 31` ✓, `state_set 36` ✓.
`bias_ordering = could not extract` ✓ — the display `b ≜ [b_g^T b_a^T]^T` is `S3.Ex2a`,
genuinely untagged between (10c) and (11). `rank1_criterion = could not extract` ✓ — §VII-A
prose, verbatim: *"We compute its numerical rank as the number of eigenvalues larger than
10⁻⁶λ_max(Ẑ) and regard the relaxation as numerically tight when rank(Ẑ) = 1."*

**delama (arXiv:2411.05548v8)** — `semidirect_product 1` ✓, `semidirect_inverse 2` ✓,
`element 3` ✓, `tangent 4` ✓, `inverse 5` ✓, `adjoint 6` ✓ (last row `[0,0,0,1]` confirmed),
`exp 8` ✓, `log 9` ✓, `left_jacobian 10` ✓, `endpoint 17` ✓, `gravity_left_factor 18` ✓,
`preintegrated_increment 19` ✓, `mean_propagation 22` ✓, `state_action 24` ✓, `input_action 26` ✓,
`equivariant_error 29` ✓, `local_chart 30` ✓, `log_G 31` ✓, `error_local_coords 34` ✓,
`error_dynamics 35` ✓, `error_dynamics_A 36` ✓, `error_dynamics_B 37` ✓, `bias_correction 39` ✓.
`bias_process_noise = could not extract` ✓ — `Q_d` is on the `Define:` line of Algorithm 1,
unnumbered, and its `0_{4×1}` blocks confirm zero process noise on `(b_ν, b_ρ)`.
`continuous_time_ins = could not extract` ✓ — the source cross-references it as `( IV-A )`,
a section tag, not an equation number.

**barfoot (arXiv:2308.12418v2)** — `so3 5` ✓, `se3_element 8` ✓, `se3_tangent 10` ✓
(`ξ^ = [ρ; φ]^`, translation first), `adjoint 11` ✓, `nonconvex 17` ✓, `homogeneous_qcqp 18` ✓,
`rank1_sdp 19` ✓, `shor_relaxation 20` ✓.

> **`cayley = "14a"` — attempted refutation, failed.** The string `14a` occurs **zero** times
> in the arXiv **HTML**; LaTeXML tags only `(14b)`, and the authors' own cross-reference reads
> "applying (15) and (16)". I was prepared to call this a fabricated label. I then pulled the
> **PDF** (`arxiv.org/pdf/2308.12418v2`) and `pdftotext` shows `(14a)`, `(14b)`, `(15a)`,
> `(15b)`, `(16a)`, `(16b)`. **`"14a"` and `"14b"` are correct.** Minor: `cayley_se3 = "15"`
> is strictly the pair (15a)/(15b); the authors themselves write "(15)", so this is
> defensible, but the forward SE(3) map alone is (15a).

`rank1_criterion = could not extract` ✓ — §3 prose, verbatim: *"we define the logarithmic SVR
(SVR) as the base-10 logarithm of the ratio of the largest to second-largest singular values
of X; we consider a log SVR of at least 5 to represent rank 1."* `sparsity = could not
extract` ✓ — §5 prose, verbatim: *"the log SVR values are quite a bit lower in Figure 12 as
compared to Figure 9. This seems to be mainly due to the fact that we are now using a sparser
set of measurements… pose measurements only at the start, middle, and end."*
`licq_failure = could not extract` ✓ — §6 prose, *"we do not meet the technical condition of
LICQ"*.

**gtsam (galileanimufactor doc + wheel 4.3.0)** — `state 1` ✓ (`β_k = [b_a; b_ω]`), `element 2` ✓,
`product 3` ✓ (*"The term t₂v₁ is the defining Galilean coupling"* — this is the O7 `reason`'s
citation, correct), `inverse 4` ✓, `tangent 5` ✓, `exp 6` ✓, `adjoint 9` ✓, `imu_sample 10` ✓,
`composition 12` ✓, `held_input_composition 14` ✓, `gravity_left_factor 15` ✓,
`preintegrated_increment 16` ✓, `endpoint 17` ✓, `bias_lift_E 20` ✓ (`E ∈ R^{10×6}`, exactly
the `(a,ω) → (ω,ν,ρ,α)` permutation), `error 21` ✓, `error_vector 22` ✓, `C_matrix 24` ✓,
`transition 29` ✓, `bias_correction 34` ✓ (*"This correction is on the right"*),
`bias_jacobian_recursion 35` ✓, `bias_correction_restated 36` ✓, `navstate_tangent 37` ✓,
`navstate_projection 38` ✓, `residual 42` ✓ (a `logmap`), `rotating_frame 47` ✓.
`navstate_block_map_symbol = could not extract` ✓ — a C++ symbol in a compiled wheel.

**INCOMPATIBILITIES** — `O1.a gtsam 2` ✓ / `O1.b could not extract` ✓ (the `Gal3(R,r,v,t)`
signature is an API fact with no numbered equation; confirmed at runtime);
`O2.a gtsam 5` ✓ / `O2.b gtsam 37` ✓ / `P_N_eq 38` ✓; `O3.a gtsam 1` ✓ /
`O3.b rai could not extract` ✓; `O4.a gtsam 10` ✓ / `O4.b rai 8a` ✓ (8a is `ω̃`, 8b is `ã` —
omega first, as claimed); `O5.a barfoot 10` ✓ / `O5.b gtsam 5` ✓; `O6.a barfoot 10` ✓ /
`O6.b delama 4` ✓; `O7.a delama 18` ✓ / `O7.b gtsam 15` ✓.

---

## The two flagged weak spots, adjudicated

**(a) M0-C14's unreproducible hint figures.** Adjudicated **CONFIRMED, not UNTESTABLE** —
see F-1. The implementer's `4.5e-16` / `0.354` is exactly what the committed fixture
produces; the hint's `9.4e-17` / `0.229` corresponds to a different `‖v_i‖`. The first bar
has three orders of margin and survives 300 random trajectories, so it is not fixture-bound.
The second bar *is* fixture-bound but the test pins its fixture. Recommendation: delete the
numerals from the hint and move `Δt_ij ≈ 0.5` and `‖v_i‖ > 0.1` into the claim's `domain:`.

**(b) M0-C05's dict shape.** The statement reads `"maps A1->'32' … with cardinalities
A1 == '6/node'"` — two different values under the same key `A1`. Taken literally the two
halves are inconsistent. The implementation resolves it as
`redundant_constraints["A1"] = {"eq": "32", "cardinality": "6/node", "identity": …}`, which
is the **only** shape that satisfies both halves at once; the test reads `rc["A1"]["eq"]` and
`rc["A1"]["cardinality"]` separately. **Adjudicated: the implementation satisfies the
statement as written.** The statement should be reworded, but the code is not wrong, and the
contents are correct against Table I and Eqs. (32)–(35).

**(c) M0-C13's substring-only `reason`.** See F-4. The string is correct and I verified its
physical consequence numerically to 1.1e-13; M0 simply does not.

---

## Nothing I could not test

Every claim in the list was decidable against a primary source or the installed wheel. No
`UNTESTABLE` verdicts. The only facts I could not reach *directly* are:

* `CONVENTIONS["gtsam"]["navstate_block_map_symbol"]` — `kGal3BlockForNavState` is a C++
  symbol inside a compiled wheel. Recorded as `could not extract`, which is the right call;
  the behaviour it names is tested at runtime and is correct.
* Barfoot's `(14a)` label is invisible in the arXiv HTML; resolved via the PDF.

## Reproduction

```
python  exeq.py srcs/<src>.html srcs/<src>_eqs.json   # per-equation LaTeX + tag extraction
python  v_c03.py      # symbolic + 1e5-draw independent check of M0-C03
python  v_c15.py      # bit-exact M0-C15 + the O7 physical consequence
python  v_gtsam.py    # O1–O4 runtime + 300-trajectory M0-C14 sweep + adversarial sweep
python  v_o2.py       # NavState tangent ordering + P_N second-order sweep
python  mutate.py     # 41-mutant survival run against tests/test_conventions.py
```

Scripts live in the verification scratchpad; the sources were fetched to a separate
directory and only ever read, never executed from.
