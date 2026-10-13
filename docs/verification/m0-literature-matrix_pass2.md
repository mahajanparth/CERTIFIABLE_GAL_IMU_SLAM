# M0 adversarial verification, pass 2: literature matrix

**Verifier:** independent pass 2. **Branch:** `m0-literature-matrix` at `d8257e5`. **Date:** 2026-10-12.
**Under test:** `src/conventions.py` (`CONVENTIONS`, `INCOMPATIBILITIES`), `docs/claims/M0.yaml`
(M0-C01..C15), M0-C16 (stated in the dispatch brief only), `tests/test_conventions.py`,
`tests/test_mutation.py`.

**Independence.** I did not read `docs/00_literature_matrix.md`, any `seeds*.yaml`, or anything
outside this worktree. The evidence is the four archived fixtures in `tests/fixtures/`,
decompressed into a scratch directory and only read, plus the installed `gtsam` 4.3.0 wheel.
Equation tags and LaTeX were pulled out with my own extractors: a `<tr>`-row regex over LaTeXML
`ltx_tag_equation` and `alttext` for the arXiv HTML, and a regex over `"enumerator"` and `"value"`
for the GTSAM mystmd JSON. I did not use the suite's `source_equations`, `canon` or `bmatrices`.
The suite's own parser was imported in exactly one place: to diff its label index against mine
(finding N-1).

| fixture (decompressed) | bytes | sha256 (prefix) |
|---|---|---|
| `rai_2609.38048v1.html` | 462 407 | `2a66413a4ae8` |
| `delama_2411.05548v8.html` | 641 374 | `efc2c26da284` |
| `barfoot_2308.12418v2.html` | 650 414 | `63ec40f48bef` |
| `gtsam_galileanimufactor.html` | 1 692 613 | `eef019ebd6a3` |

No PDF of any source is archived. `references/*.pdf` is gitignored, and no `references/`
directory exists in this worktree.

---

## Verdict table

| id | verdict | attack |
|---|---|---|
| M0-C01 | **CONFIRMED** | Counted the raw substrings in the script/style-stripped rai HTML: every listed term gives 0; `Barfoot` gives 2 (refs [6] and [13]); `Mahony` gives 1, and it sits in ref [3] (Shonan averaging), not in a Delama citation. The table fields `group == "SO(3)"`, `element_order is None` and `tangent_order is None` hold. |
| M0-C02 | **CONFIRMED** | Read Eq. (19) off its own row: `r^R_ij ≜ (R_j − R_iΔR̄_ij) − ½(R_j + R_iΔR̄_ij)ξ_ij^`. Eq. (15) gives `ξ_ij ≜ J^R_ij δb_g`. The prose says "Using δb_g = b_g − b̄_g". I looked for `+½`, a missing minus, and `b_g` in place of `δb_g`, and found none. |
| M0-C03 | **CONFIRMED** | sympy over free 3×3 `R_i, R_j, ΔR̄, J^R` and free 3-vectors `b_g, b̄_g`: `expand(22b − 19)` is the zero matrix. The identity is exact, so it holds at any scale. |
| M0-C04 | **CONFIRMED** | Eq. (28) row text: `{vec(R_i), v_i, p_i}_i, {b_m}_m, {vec(Y_ij), vec(Y'_ij), W^v_ij, W^p_ij}_(i,j), s`, with `s` last. Eq. (31) prints `s²=1`, `R_iᵀR_i = s²I_3` and `R_i^(a)×R_i^(b) = sR_i^(c)`, so all three families are present. |
| M0-C05 | **CONFIRMED** | Eqs. (32)–(35) print `(A1, 6/node)`, `(B1, 3/lifting)`, `(B2)` with no count, and `(C1, 2/edge)`. §VII-B prints `2×7×12×5=840`, "at least 0.88", "none of the corresponding 504 solves is certified", "median gap is 2.8×10⁻⁵" and "Adding B2 substantially reduces". The instance is `K=8, M=2, n=340` (K=16/n=652 is the other instance). |
| M0-C06 | **CONFIRMED** | The unnumbered display `τ‖r‖² = τ‖R_i r‖², R_i ∈ SO(3)` is followed verbatim by "This equality requires an isotropic weight. We therefore approximate … by scalar precisions", and then Eq. (23) `τ⁻¹ = ⅓ tr Σ`. |
| M0-C07 | **CONFIRMED** | Delama Eq. (3) is `[[A,a,b],[0,1,c],[0,0,1]]`, Eq. (4) is `(ω, v, r, α)`, and Eq. (5) is `[[Aᵀ, −Aᵀa, −Aᵀ(b − ca)],[0,1,−c],[0,0,1]]`. The last row of Eq. (6) is `[0_{1×3}, 0_{1×3}, 0_{1×3}, 1]`. |
| M0-C08 | **CONFIRMED** | Eq. (1) is `XY = (AB, a + Ad_A[b])`. The sentence after Eq. (24) is, verbatim, "Then, φ is a transitive and free right group action of **G** on ℳ." The sentence after (26) also says "right". There is no "left group action" in the source. Algorithm 1 has `J_ξ0 = I_20`; Eqs. (36)/(37) use 10×10 blocks (`I_10`, `0_{10×10}`). |
| M0-C09 | **CONFIRMED** | The source prints `b_k = (b_ωk, b_ak, b_νk, b_ρk) ∈ R^10`, and "two additional virtual biases: a velocity bias b_νk ∈ R³ and a time bias b_ρk ∈ R". Algorithm 1 has `Define: … Q_d = diag(σ_ω², σ_a², 0_{4×1}, σ_τω², σ_τa², 0_{4×1})`, untagged, and its second half zeroes the τ-noise on (b_ν, b_ρ). The input is `(ω̃, ã, 0_{3×1}, 1)`. |
| M0-C10 | **CONFIRMED** | Delama (31) and (34) both contain `log(·)` and `J_L(·)^{-1}`; (9) is `log(X)` and (10) is `J_L`. rai (3) is `Cay` and (14) is the Frobenius residual on `Exp(ξ)`. The forward Cayley row of barfoot block 14 has no exp or ln. GTSAM (6) is `Exp` and (42) is `logmap`. Caveat on barfoot's label: N-1 and the 14a section below. |
| M0-C11 | **CONFIRMED** | Delama (34) is `log(Υ_kΥ̂_k^{-1})`, which is right-invariant, and (39) is `exp(·)Υ̂`, a left correction. GTSAM (21) is `ξ̂^{-1}ξ`, which is left-invariant, and (34) is `Υ̂ Exp(Jδβ)`, a right correction. The GTSAM prose says Delama's error "is right-invariant, together with a left-applied first-order correction". |
| M0-C12 | **CONFIRMED** | Barfoot §3 prose, verbatim: "…base-10 logarithm of the ratio of the largest to second-largest singular values of **X**; we consider a log SVR of at least 5 to represent rank 1". §5: "lower in Figure 12 as compared to Figure 9 … sparser set of measurements". rai §VII-A: "number of eigenvalues larger than 10⁻⁶λ_max(Ẑ) … rank(Ẑ)=1". On the spectrum (1, 10^-5.5, 1e-9, …) the log-SVR is 5.5 (accept) while rai's rank is 2 (reject). |
| M0-C13 | **CONFIRMED** | Freshly written runtime checks on 4.3.0. `Gal3(I,[1,2,3],[10,20,30],7)` puts `[10,20,30]` in column 3 and `[1,2,3]` in column 4. `Hat(e_3..5)` lands in column 3, `Hat(e_6..9)` in column 4. `NavState.retract(e_3..5)` moves position and `retract(e_6..8)` moves velocity. Doc Eq. (38) `P_N` gives rows `(0,1,2,6,7,8,3,4,5)`. `ConstantBias([1,2,3],[40,50,60]).vector() == [1,2,3,40,50,60]`. `integrateMeasurement([1,2,3],0,0.1)` gives ΔR = I and Δv = `[.1,.2,.3]`. Delama (18) has `−½gΔt²`; GTSAM (15) has `+½gΔt²`; GTSAM (3) has `p₁+R₁p₂+t₂v₁`. Barfoot (10) is `[ρ; φ]`. |
| M0-C14 | **CONFIRMED** | Over 200 random trajectories (n ∈ [50,200] samples, Δt_ij = 0.5, random bias, X_i, acc, gyro), the worst `‖X_j ⊖ W·φ(X_i)·U‖` is **1.8e-15**. `‖X_j ⊖ W·X_i·U‖/(‖v_i‖Δt)` lies in [1−1e-15, 1+1e-15], so the coasting-free error is exactly ‖v_i‖Δt; the domain now pins this regime. Putting Delama's `−½` into W gives an error of 2.45. |
| M0-C15 | **CONFIRMED** | I transcribed Γ from Delama (18) myself. Over 5000 draws with |dt| ≥ 0.05 and ‖g‖ ≥ 1, `max|Γ − T(−dt)H′|` is exactly 0, and `min ‖Γ − H′T‖/(‖g‖dt²)` is 1.0. |
| **M0-C16** | **REFUTED** | Counterexample in gtsam 4.3.0, below. |

**Counts: 15 CONFIRMED, 1 REFUTED (M0-C16), 0 UNTESTABLE.** One table *field* cannot be decided
from the cached sources, although no claim depends on it: `barfoot.eq.cayley = "14a"` (see below).

---

## REFUTED: M0-C16

> *"In GTSAM 4.3.0 the rotation-free elements Gal3(I, r, v, t) form an abelian subgroup … A*B ==
> B*A to machine precision."*

Run on the installed wheel:

```
A = Gal3(I, r=0, v=[1,0,0], t=0);  B = Gal3(I, r=0, v=0, t=1)
A.compose(B).position() = [1, 0, 0]
B.compose(A).position() = [0, 0, 0]
componentwise sum      -> position [0, 0, 0]
```

Over 20 000 random rotation-free pairs at scales 1e-3..1e3, `max |AB − BA| / max(1, s²) = 9.8` and
`max |AB − (r₁+r₂, v₁+v₂, t₁+t₂)| = 8.7e6`. The cause is the defining Galilean coupling of GTSAM
Eq. (3), `p = p₁ + R₁p₂ + t₂v₁`: with R = I, `AB` has position `r₁ + r₂ + t₂v₁` while `BA` has
`r₁ + r₂ + t₁v₂`. The rotation-free elements are closed under composition, so they do form a
subgroup, but the subgroup is non-abelian. It is a Heisenberg-type group, not R⁷. The claim is
false on both halves: composition is not componentwise addition, and `A*B != B*A`.

---

## Rulings on the earlier findings (pass 1, `m0-literature-matrix.md`)

| finding | ruling | evidence |
|---|---|---|
| F-1 (C14 `>1e-2` fixture-bound) | **FIXED** | The C14 `domain` now pins `dt_ij ~ 0.5 s` and `‖v_i‖ > 0.1`. The test reads both values from the YAML, checks `want_v·want_dt > bar`, and asserts `err_wrong == ‖v_i‖Δt` (rel 1e-6). The unreproducible 9.4e-17 / 0.229 figures are gone. The statement still writes "> 1e-2" without qualification; the domain is what qualifies it. |
| F-2 (C15 `>1e-6` fixture-bound) | **FIXED** | The domain states `|dt| ≥ 0.05, ‖g‖ ≥ 1`. The test reads those floors and asserts `reversed_err == ‖g‖dt²`. |
| F-3 (C03 absolute 1e-12) | **FIXED** | The domain and test_hint make the bar relative to `max(1, max|Eq.19|)`, and the test runs at scales 1, 1e-8 and 1e4. The statement text still says "to 1e-12" and leaves the relative reading to the domain. |
| F-4 (O7 `reason` substring-only) | **PARTIALLY FIXED** | `O7.a.says` is now built into a 5×5 matrix and compared bit-for-bit with Eq. (18), which kills mutants. The `reason` string's `t2*v1` / `eq 3` / "re-contributes … stored … land on" clauses are parsed and evaluated. But "not a contradiction" is still a substring test, and my mutant that rewrote the final sentence to "SE_2(3) has the same coupling." **survived**. |
| F-5 (`inverse_formula` unread) | **FIXED** | The recorded formula is executed and `X·X⁻¹ = I_5` is checked. Two of my own sign mutants (`c` vs `−c`) and their `(b + c a)` mutant are all killed. |
| F-6 (40/41 mutants survive) | **FIXED** for the 41; residual exposure remains | `tests/test_mutation.py` passes, so all 41 are killed, with working null, identity and gross-corruption controls. My own 190-mutant sweep killed 172 of 190. Every survivor is in a field that no claim names (N-2). |
| F-7 (assertions that cannot fail) | **PARTIALLY FIXED** | Items 1, 2, 4, 5, 6 and 7 are gone; I grepped the suite for each one. Item 3 (C06) was replaced by a source-display parse, which is good, but the numeric "RUN" block after it is forced and still cannot fail (T-1). New cannot-fail assertions are listed under N-3. |
| Pass-1 (b) C05 dict shape | **unchanged, still acceptable** | The statement was not reworded; the implementation still satisfies both halves. |
| Pass-1 M0-S01 | **still REFUTED**; table unchanged | `delama.action_handedness == "right"` is unchanged, and the mutant `"left"` is killed by C08. |

### The implementer's "14 earlier failures were all test bugs, none a table error"

**CONFIRMED.** I checked it directly. `src/conventions.py` is byte-identical between `f4b26c3` and
`d8257e5`; its last change is `c820899`. Running the `f4b26c3` test file against today's table
reproduces exactly 14 failures. These are the first failing assertions:

- **10 parser, indexer or regex bugs:**
  - C02: xi array shape.
  - C08: the parser resolved `("delama","semidirect_product")` without `"eq"`.
  - C09: the same bug for `bias_process_noise`.
  - C10, C11, gtsam-row and register-row: the GTSAM index held 0 equations because its AST lives in a `<script>`.
  - C12: compared the string `1e-6` with the float `1e-06`.
  - delama-row: a signature regex missed `\Delta\!\prescript{}{}{\mathbf{R}}`.
  - barfoot-row: a whitespace mismatch in the Eq. (10) signature.
- **2 gaps in the claim YAML, not the table:** C14 and C15 domains were unpinned. These are pass-1 F-1 and F-2, fixed by editing the YAML `domain` and `test_hint` fields only. I diffed the parsed YAML: no `statement` changed.
- **2 invented heuristics:**
  - rai-row: word overlap with `docs/00_literature_matrix.md`, the deriver's own write-up.
  - C13: "`where` must name both sources". `O4.where` is "IMU sample ordering". This is a prose-style requirement, not a sourced fact.

Caveat: only the first failing assertion of each old test was observed. The later assertions in
those tests were masked. They are covered now, because the new suite passes on the same table.

---

## Barfoot `eq.cayley = "14a"` and `eq.cayley_se3 = "15"` (items the implementer flagged)

**The cached HTML does not contain "14a" at all.** The string `14a` occurs 0 times in
`barfoot_2308.12418v2.html`, and so do `15a` and `16a`. Block `S2.E14` has two rows. The first,
`C(φ) = cay(φ^) = (I − ½φ^)⁻¹(I + ½φ^)`, has **no tag**. The second, `φ = cay⁻¹(C)^∨ …`, is tagged
`(14b)`. Blocks 15 and 16 have the same shape. The authors' only cross-reference to these blocks
is "applying (15) and (16)", i.e. the block labels.

- **`cayley = "14a"`: not decidable from the cached sources.** The label is a sound *inference*:
  the untagged first row of a block whose second row is `(14b)` can only be `(14a)`. But the
  archived source never prints it. Pass 1 cites `pdftotext` of the arXiv PDF as showing `(14a)`,
  but that PDF is not archived in the repo, and I did not reconstruct its contents from memory.
  The suite "finds" 14a only because `_latexml_equations` *manufactures* a sub-letter by row
  position (N-1). Its docstring justifies this by the PDF, which is a deriver's note, not a cached
  source. To make the field decidable, either archive the PDF as a fixture, or record the label
  as `"14"` (block) with a note.
- **`cayley_se3 = "15"`: defensible.** "(15)" is how the authors themselves cite the block. The
  forward SE(3) map alone is the untagged first row (15a by the same inference). The mutants
  `"16"` (the adjoint Cayley block) and `"15b"` would be caught; I ran `"16"` and it was killed.

---

## Independent field checks (≥15 required; 40+ done)

All of these were read with my own extractors, or run on gtsam 4.3.0. Every one matches the
table unless marked.

- **rai.** `eq.cayley 3`, `cayley_right_jacobian 4`, `qcqp 5`, `sdp 6`, `chordal_residual 14`,
  `xi_definition 15`, `cayley_determinant 17`, `denominator_clearing 18`, `rotation_residual 19`,
  `21a/21b/21c`, `22a/22b/22c`, `isotropic_weighting 23`, `u_definitions 26`, `lifted_vector 28`,
  `lifting_constraints 30`, `homogeniser 31`, `A1..C1 32..35` with counts, `state_set 36`.
  `bias_ordering` is the untagged display `b ≜ [b_gᵀ b_aᵀ]ᵀ`; `O4.b 8a` is `ω̃`. The five
  ablation figures, the rank rule, and the "prolonged stationary segment, leaving the heading
  unobservable" cause also match.
- **delama.** Eqs. 1, 2, 3, 4, 5, 6 (Ad time row), 8, 9, 10, 17, 18, 19, 22, 24, 26, 29, 30, 31,
  34, 35, 36, 37 and 39. The bias vector, the virtual biases, `Q_d` and the inputs match. The
  manifold `ℳ ≔ Gal(3) × R^10` and the citation DOI and date (`10.1109/LRA.2024.3511424`,
  21 Aug 2025) are both printed in the source.
- **barfoot.** Eqs. 5, 8, 10, 11, 14b, 17, 18, 19 and 20 match, as do the log-SVR, sparsity and
  LICQ prose. `14a` cannot be decided from the cache (above). **Not in the cached source:** the
  citation fragment "IJRR 44(3):366-387, 2025"; `366` and `44(3)` occur 0 times. "31 May 2024"
  is present.
- **gtsam doc.** Eqs. 1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 16, 17, 20, 21, 22, 24, 29, 34, 35,
  36, 37, 38, 42 and 47 match.
- **gtsam runtime.** O1, O2 (`Hat` and `NavState.retract`), O3 and O4 match. The retraction is
  `X·Exp(δ)` (the error vs `Exp(δ)·X` is 0.039). The Eq. (3) product law holds to 0.0.

---

## Own mutation sweep (none of these (path, value) pairs is in `tests/test_mutation.py`)

**Harness.** The live table was mutated in-process, every `test_*` in `tests/test_conventions.py`
was called, and the table was restored from a deep snapshot. Restoration was asserted after every
mutant, and the null run was clean. `src/` was never written to; `git status` was clean afterward.
The values are plausible but wrong: neighbouring equations, the other paper's value, swapped
orderings, sign flips, labels the parser invents, prose eq-refs off by one, and wrong citations.

**Result: 172 of 190 killed, 18 survived.** By source:

| group | mutants | killed |
|---|---|---|
| rai | 51 | 44 |
| delama | 42 | 40 |
| barfoot | 27 | 25 |
| gtsam | 44 | 42 |
| INCOMPATIBILITIES | 26 | 21 |

**Survivors (18)**, all in fields that no claim statement names:

| field | mutant | why it survives |
|---|---|---|
| `rai.error_handedness` | `"left-invariant"` | Only the structural enum check reads it; rai has no group error. |
| `rai.bias_correction_side` | `"right"` | Enum check only. |
| `rai.eq.u_definitions` | `"26a"` | The signature requires only the v-half (`W^v`, `u^v`), so it never checks 26b. |
| `rai.exact` | "Eq. (20)" for the residual | The prose eq-ref check only verifies that the label exists. |
| `rai.linearized` | "Eq. (22)" for weighting | Same. |
| `delama.exact` | "Eq. (17)" for Γ | Same. |
| `barfoot.exact` | "Eqs. (14), (16)" | Same. |
| `gtsam.linearized` | "Eqs. (24)-(29)" | Same. |
| `barfoot.citation` | "IJRR 43(3) … 2024" | Only arXiv ids and DOIs are checked; the journal fragment is not in the cache anyway. |
| `rai.citation` | date "30 Sep 2026" | Dates are not checked. |
| `rai.citation` | author "Carlone" for "Rosen" | Authors are not checked. |
| `delama.citation` | title "… an SE_2(3) Group Approach" | Titles are not checked. |
| `gtsam.endpoint_rejected_form` | "(gravity factor dropped)" | Only a `startswith` on `H_ij X_i D_ij`. |
| `O1.consequence` | "nothing swaps; the constructor and matrix agree" | `consequence` is never read. |
| `O2.consequence` | "a selection only; no permutation is needed" | Same. |
| `O4.consequence` | "gtsam Eq. 24" for E | Same. |
| `O6.consequence` | "ordering may be inferred from a letter" | Same. |
| `O7.reason` | last sentence → "SE_2(3) has the same coupling." | Only the first clauses are parsed (F-4). |

Some notable kills:

- the neighbouring-label pairs `delama.eq.error_dynamics_A ↔ B`, `state_action ↔ input_action`,
  `equivariant_error ↔ local_chart`, `gtsam.eq.error ↔ error_vector` and
  `bias_correction ↔ bias_correction_restated`;
- `barfoot.eq.cayley ∈ {"14", "15a", "24", "14c"}`;
- `gtsam.eq.residual = "42a"`;
- `rai.eq.u_definitions = "25c"`, a label the parser itself invents;
- the swapped Gal3 tangent and the delama/gtsam letter swaps;
- every sign flip in `W_ij`, `phi_dt`, `inverse_formula` and the rai residual.

---

## Tests that cannot fail, and tests leaning on the deriver's notes

### Cannot fail (no table mutation, and no source change short of editing the fixture, can make them fail)

- **T-1, C06.** The "recorded identity is RUN" block. The preceding regex already forces side 1
  to be `tau*||r||^2` and side 2 to be `tau*||R_i r||^2`, so the numeric check is `|r| = |Rr|`
  for rotations, which is a theorem. This is the F-7.3 pattern again, in a different form.
- **T-2, C03.** The 3×1e4-draw loop builds both Eq. (22b) and Eq. (19) from expressions written
  in the test and reads no table value. The identity is exactly true (sympy), so the loop cannot
  fail. Only the three `assert_equation_says` lines at the end read the table. This is fine as
  evidence for a DERIVED claim, but it has zero table-falsification power, and it is the deriver's
  transcription rather than a parse of the source.
- **T-3, C15.** `reversed_err == approx(‖g‖dt²)` and `reversed_err > bar` use Γ from the fixture
  and H′, T from the claim; the table is never read. Only the `O7.a.says` array-equality line can
  fail on a table mutation.
- **T-4, C14 and C15.** `want_v·want_dt > bar_wrong` and `g_floor·dt_floor² > bar` read only the
  YAML.
- **T-5.** Fixture-only checks: C09 `not any("Q_{\scriptscriptstyle{d}}" in numbered eq)`, C12
  `not any("SVR" in numbered eq)`, and the `SENTINEL_EVIDENCE` "no numbered hit" halves. These
  can only fail if the fixture changes, never if the table does.
- **T-6.** `SENTINEL_EVIDENCE[("delama","eq","continuous_time_ins")]` has `numbered=None`, so its
  "not in a numbered equation" half is vacuous.
- **T-7, C14.** `isinstance(phi, gtsam.NavState)` holds by construction, because `phi` is built
  with `gtsam.NavState(*…)`.

### Leaning on the deriver's own notes rather than a source

- **N-1: the label-by-position heuristic in `_latexml_equations`.** It invents labels the cached
  sources never print. Diffing its index against the raw tags gives: barfoot `14a, 15a, 16a,
  30b`; rai `25c`; delama none. In rai it also **mis-assigns rows**:
  - the invented `25c` is really the first line of (25b), `R_i r^p_ij = …`;
  - (25a) loses its first line, `R_i r^v_ij = …`;
  - in block 37 the untagged `Δv̄`/`Δp̄` rows are overwritten by the real (37b)/(37c).

  No current table value depends on the rai damage. `barfoot.eq.cayley = "14a"` passes *only*
  because of this heuristic, and so does M0-C10's barfoot check, because `_RESIDUAL_EQ["barfoot"]`
  points at that label. The heuristic's justification ("present in the PDF") refers to a file
  that is not in the repo.
- **N-4: fields checked only against the claim YAML (the deriver's prose), never against a
  source.** These are `delama.group` (`"Gal(3) |x gal(3)"`), `delama.manifold` (read from the
  C08 `domain`), `rai.group`, and `element_order is None` / `tangent_order is None` for rai. In
  C07, `element_slots` is also matched against the domain's wording, although the group-law check
  next to it is source-independent and sound. The values do match the source (`ℳ ≔ Gal(3)×R^10`,
  `Gal(3)⋉gal(3)`); the suite just does not show it.
- **`_RESIDUAL_EQ`.** Which equation counts as each row's "estimator residual" is the test
  author's choice. For barfoot it is the Cayley-map *definition*, not a cost; that choice decides
  the `exp_log_essential` verdict.
- **`EQ_SIGNATURES`.** These are deriver-written regexes describing what each equation "says".
  They are checked against the source, which is good. Some are one-sided, though; `u_definitions`
  lets `26a` survive.

### Other observations

- **N-5.** The GTSAM doc page installs `gtsam-develop` in its notebook cell and carries no version
  banner. The doc conventions are confirmed against the 4.3.0 runtime only where a runtime check
  exists (O1–O4, the retraction, the C14 predictor, and the C16 product law).
- **N-6.** Pass 1 states the rai fixture is "byte-identical … 456 935 B both", but its own source
  table lists 462 407 B. The archived fixture decompresses to **462 407 B** today. I did not
  refetch, so the byte-identity claim of pass 1 is unconfirmed by this pass. It does not affect any
  verdict, since all checks here are on the archived bytes.

---

## Reproduction

The scripts are kept in the verifier scratchpad and read the fixtures copied there.

| script | what it does |
|---|---|
| `dump.py` | LaTeXML row/tag/alttext extractor, independent of the suite |
| `gt.py` | GTSAM mystmd enumerator/value extractor |
| `rt.py` | O1–O4 runtime checks, retraction, Eq. (3), and the M0-C16 refutation |
| `c14.py` | 200-trajectory M0-C14 sweep and 5000-draw M0-C15 check |
| `c03.py` | sympy proof of M0-C03 and the M0-C12 spectrum anchor |
| `mut.py`, `mut2.py` | 158 + 32 mutants; skips any (path, value) in `tests/test_mutation.py` and asserts table restoration |

The "14 failures" replay is `git archive f4b26c3` into the scratchpad followed by
`pytest tests/test_conventions.py`, which gives 14 failed / 7 passed.

After the report was committed, the full suite was run on the clean tree, so the gitignored
`.orchestrator_evidence.json` holds the clean results.
