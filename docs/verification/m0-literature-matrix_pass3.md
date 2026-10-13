# M0 adversarial verification, pass 3: literature matrix

**Verifier:** independent pass 3. **Branch:** `m0-literature-matrix` at `4ef1ee5`. **Date:** 2026-10-12.
**Under test:** `src/conventions.py`, `docs/claims/M0.yaml` (M0-C01..C15), M0-C17 (stated in the
dispatch brief only), `tests/test_conventions.py`, `tests/test_mutation.py`, and the new fixtures
(`barfoot_2308.12418v2_pdf_p4_cayley.txt`, `crossref_10.1177_02783649241269337.json`,
`provenance.json`).

**Independence.** I did not read `docs/00_literature_matrix.md`, any `seeds*.yaml`, or anything
outside this worktree. I decompressed the fixtures into a scratch directory and read them with my
own extractors:

- a `<tr class="ltx_eqn_row">` regex that prints each row's LaTeXML id, its printed
  `ltx_tag_equation` and its `alttext`;
- a prose renderer that replaces each `<math>` with its alttext;
- an `"enumerator"` → `"value"` scan of the GTSAM mystmd JSON.

I also fetched three things over the network into scratch:

- the arXiv PDF of Barfoot v2. Its sha256 is `7aca0ca9…5a02c` and it is 3 536 041 B, which
  matches the excerpt header.
- the rai HTML;
- the Crossref record.

I used the suite's own parser in one place only: to probe whether it invents labels (section 4).
The installed wheel reports `importlib.metadata.version("gtsam") == "4.3.0"`.

---

## 1. Verdict table

| id | verdict | attack (own extractor / own computation) |
|---|---|---|
| M0-C01 | **CONFIRMED** | Raw counts on the script/style-stripped rai HTML are 0 for each of `Galilean`, `SE_2`, `\mathrm{SE}`, `SE}_{2`, `extended pose`, `equivariant`, `Equivariant`, `Delama`, `Fornasier` and `Weiss`. `Barfoot` = 2 and `Mahony` = 1 (ref. [3], Shonan). The table has `group == "SO(3)"`, `element_order is None` and `tangent_order is None`. |
| M0-C02 | **CONFIRMED** | Row `S4.E19.m1` (19) is `r^R_ij ≜ (R_j − R_iΔR̄_ij) − ½(R_j + R_iΔR̄_ij)ξ_ij^`. Eq. (15) gives `ξ_ij ≜ J^R_ij δb_g`, and the display after (10c) is `δb ≜ b − b̄`. Both signs and the perturbation reading hold. |
| M0-C03 | **CONFIRMED** | Own sympy over free symbolic 3×3 `R_i, R_j, ΔR̄, J^R` and free 3-vectors `b_g, b̄_g`, transcribed from (22a)–(22c) and (19): `expand(22b − 19) == 0`. The identity is exact. |
| M0-C04 | **CONFIRMED** | Rows `S4.E28X` + `S4.E28Xa`: `{vec(R_i), v_i, p_i}_i, {b_m}_m, {vec(Y_ij), vec(Y'_ij), W^v_ij, W^p_ij}_(i,j), s`, with `s` last. Eq. (31) is `s²=1, R_iᵀR_i = s²I_3, R_i^(a)×R_i^(b) = sR_i^(c)`, so all three families are present. |
| M0-C05 | **CONFIRMED** | (32) `(A1, 6/node)`, (33) `(B1, 3/lifting)`, (34) `(B2)` with no count, (35) `(C1, 2/edge)`. §VII-B prints `2×7×12×5=840`, "at least 0.88", "none of the corresponding 504 solves is certified", "median gap is 2.8×10⁻⁵" and "Adding B2 substantially reduces"; the instance is `K=8, M=2, n=340`. |
| M0-C06 | **CONFIRMED** | Row `S4.Ex1.m1` is `τ‖r‖²₂ = τ‖R_i r‖²₂, R_i ∈ SO(3)`. The prose continues "This equality requires an isotropic weight". Eq. (23) is `τ⁻¹ = ⅓ tr Σ` for each of R, v, p. |
| M0-C07 | **CONFIRMED** | (3) `[[A,a,b],[0,1,c],[0,0,1]]`; (4) `(ω, v, r, α)`; (5) `[[Aᵀ, −Aᵀa, −Aᵀ(b − ca)],[0,1,−c],[0,0,1]]`; the last row of (6) is `0_{1×3}&0_{1×3}&0_{1×3}&1`. GTSAM (5) is `(ω, ν, ρ, α)`, so the slots are identical. |
| M0-C08 | **CONFIRMED** | (1) is `XY=(AB, a+Ad_A[b])`. The sentence after (24) is, verbatim, "Then, φ is a transitive and free right group action of **G** on ℳ." Sec. I has "left-trivialized tangent group Gal(3)⋉gal(3)", and the manifold is defined as "Define ℳ ≔ 𝒢al(3)×ℝ¹⁰". (36)/(37) are built from `I_10` and `0_{10×10}` blocks, and Algorithm 1 has `J_ξ0 = I_20`. |
| M0-C09 | **CONFIRMED** | `b_k = (b_ωk, b_ak, b_νk, b_ρk) ∈ ℝ¹⁰`, with "two additional virtual biases: a velocity bias … and a time bias". The `Q_d = diag(σ_ω², σ_a², 0_{4×1}, σ_τω², σ_τa², 0_{4×1})` line sits on an untagged "Define:" line of Algorithm 1. |
| M0-C10 | **CONFIRMED** | No `exp`/`ln`/`Log`/`Exp` appears in rai (19) or in barfoot (25) and (28), which contain only `cay⁻¹`. Delama (31) and (34) contain `log(·)` and `J_L(·)⁻¹`, with (9) = `log(X)` and (10) = `J_L`. GTSAM (6) is `Exp` and (42) is `logmap`. rai (14) is the Frobenius `‖·‖_F²` residual on `Exp(ξ)`, and (3) is `Cay`. |
| M0-C11 | **CONFIRMED** | Delama (34) is `log(Υ_kΥ̂_k⁻¹)` (right-invariant), and (39) is `exp((J_Υ Δb̂)^)Υ̂`, a left correction. GTSAM (21) is `ξ̂⁻¹ξ` (left-invariant), and (34) is `Υ̂ Exp(J_kδβ)`, a right correction. The GTSAM page itself says: "Delama et al.'s … navigation error ΥΥ̂⁻¹, which is right-invariant, together with a left-applied first-order correction." |
| M0-C12 | **CONFIRMED** | Barfoot §3: "…ratio of the largest to second-largest singular values of **X**; we consider a log SVR of at least 5 to represent rank 1". rai §VII-A: "number of eigenvalues larger than 10⁻⁶λ_max(Ẑ)". Own anchor on the spectrum (1, 10^-5.5, 1e-9, 1e-12×7) in a random orthogonal basis: log-SVR is 5.50 (accept) while rai's rank is 2 (reject). |
| M0-C13 | **CONFIRMED** | Fresh runtime checks on 4.3.0. `Gal3(I,[1,2,3],[10,20,30],7).matrix()` gives column index 3 = `[10,20,30]` (the statement's "column 4" is 1-based) and column index 4 = `[1,2,3]`. `Hat(e_3..5)` lands in column 3 and `Hat(e_6..9)` in column 4. `NavState().retract(e_3..5)` moves position and `retract(e_6..8)` moves velocity. Doc (38) `P_N` gives rows `(0,1,2,6,7,8,3,4,5)`. `ConstantBias([1,2,3],[40,50,60]).vector() == [1,2,3,40,50,60]`. `integrateMeasurement([1,2,3],0,0.1)` gives ΔR = I and Δv = `[.1,.2,.3]`. Barfoot (10) is `[ρ;φ]`. Delama (18) has `−½gΔt²`, GTSAM (15) has `+½gΔt²`, and GTSAM (3) has `p₁+R₁p₂+t₂v₁`. |
| M0-C14 | **CONFIRMED** | Own sweep of 200 random trajectories (n ∈ [50,200), Δt_ij = 0.5, random bias, X_i with ‖v_i‖ > 0.1). The worst `‖X_j ⊖ W·φ(X_i)·U‖` is **4.5e-16**. `‖X_j ⊖ W·X_i·U‖/(‖v_i‖Δt)` lies in [1 − 6e-16, 1 + 9e-16]. |
| M0-C15 | **CONFIRMED** | Own transcription of Γ from Delama (18), 5000 draws with \|dt\| ≥ 0.05 and ‖g‖ ≥ 1. `max\|Γ − T(−dt)H′\|` is exactly 0, and `min ‖Γ − H′T‖/(‖g‖dt²)` is 1.0. |
| **M0-C17** | **REFUTED** | Source and runtime (section 2). |

**Counts: 15 CONFIRMED, 1 REFUTED (M0-C17), 0 UNTESTABLE.**

---

## 2. REFUTED: M0-C17

> *"In GTSAM 4.3.0, gtsam.Gal3.inverse() of X = Gal3(R, r, v, t) has position block −Rᵀr,
> exactly as in SE(3): the inverse's position does not depend on v or t."*

**Source.** The cached GTSAM page, Eq. (4), prints
`(R,v,p,t)⁻¹ = (Rᵀ, −Rᵀv, −Rᵀ(p − tv), −t)`, and Delama Eq. (5) agrees: `−Aᵀ(b − ca)`. The
position block of the inverse depends on both `v` and `t`.

**Runtime (gtsam 4.3.0).**

```
Gal3(Rot3(), r=0, v=[1,0,0], t=1).inverse().position()  ->  [1, 0, 0]      (claim predicts [0,0,0])
20 000 random X, scales 1e-3..1e3:
  max |inv.position() - (-R^T r)|      / max(1,s^2) = 9.96
  max |inv.position() - (-R^T(r - t v))| / max(1,s^2) = 1.6e-15
```

The claim is false. It is the SE(3)/SE₂(3) inverse, which drops the Galilean `t·v` coupling. That
coupling is the same one behind O7 and pass-2's M0-C16. Mutating `delama.inverse_formula` to this
form (`-A^T b`) is killed by the suite.

---

## 3. Arbitration of the two changed table values

### (a) `rai.bias_correction_side`: `"none"` → `"right"`. **CORRECT.**

These are the source texts, from my own extractor:

- **Eq. (11a)** (row `S3.E11.1`):
  `ΔR̃_ij(b) ≈ ΔR̃_ij(b̄) Exp(ξ_ij)`, with `ξ_ij ≜ J^R_ij δb_g` (untagged display just before it).
- **Eq. (14)** (row `S3.E14.m1`):
  `‖R_j − R_iΔR̃_ij(b̄) Exp(ξ_ij)‖²_F`. The prose that follows reads: *"The second issue is the
  exponential map used for bias correction. … Since ξ_ij = J^R_ij δb_g depends on the unknown
  gyroscope bias … The next section replaces this exponential-map correction with the Cayley
  map."*
- **Eq. (15)** (row `S4.E15.m1`):
  `R_iᵀR_j = ΔR̄_ij Cay(ξ_ij) Cay(−δφ_ij),  ξ_ij ≜ J^R_ij δb_g`.
- **Eq. (16)**: `R_j = R_i ΔR̄_ij Cay(ξ_ij)`.
- **Sec. VI prose**: *"Changes in the estimated bias are handled by the rotation correction in
  ( 15 ) and the bias Jacobians in ( 21b )–( 21c )."*

The equation numbers exist as cited. In every form, the first-order gyro-bias correction
multiplies the nominal pre-integrated rotation on the **right**. The other rows give the field
this meaning:

- gtsam `"right"`, from `Υ̂ Exp(Jδβ)`, Eq. (34);
- delama `"left"`, from `exp(·) Υ̂`, Eq. (39).

So rai is `"right"` under the same reading. `eq.bias_correction = "15"` is also the label the
authors themselves use for it. The former `"none"` was the error.

### (b) `barfoot.error_handedness`: `"none"` → `"right-invariant"`. **CORRECT.**

These are the source texts:

- **Eq. (25)** (row `S3.E25.m1`):
  `min Σ_m cay⁻¹(C C̃_mᵀ)^∨ᵀ W_m cay⁻¹(C C̃_mᵀ)^∨  w.r.t. C  s.t. C ∈ SO(3)`.
- **Eq. (28)** (row `S3.E28.m1`):
  `min Σ_m cay⁻¹(T T̃_m⁻¹)^∨ᵀ W_m cay⁻¹(T T̃_m⁻¹)^∨  w.r.t. T  s.t. T ∈ SE(3)`.
- **Consistent elsewhere**: (22) uses `ln(C C̃_mᵀ)`, (37) `cay⁻¹(T_k T̃_k⁻¹)`, (38b)
  `cay⁻¹(T_{k+1}T_k⁻¹T̃_{k+1,k}⁻¹)` and (48a) `cay⁻¹(T_k T̃_k⁻¹)`. The noise model (24),
  `C̃_m = cay(φ_m^)C`, is a left perturbation. No `X̃⁻¹X` form appears anywhere in the paper.

Both equation numbers exist as cited. The error is `X X̃⁻¹`. This is the same composition the
table calls "right-invariant" for delama (`Υ Υ̂⁻¹`, Eq. 34), the opposite of gtsam's
"left-invariant" `ξ̂⁻¹ξ` (Eq. 21). The GTSAM page states the same convention: *"left-invariant
error and right perturbation describe the same convention … the navigation error ΥΥ̂⁻¹, which is
right-invariant"*.

The value is also the same under the stricter cost-invariance reading. Under `C → CS`,
`C̃ → C̃S` the cost is invariant for any `W_m`. Under left translation it is invariant only for
isotropic `W_m`. So the value does not depend on which reading is used.

### Field-meaning observation (not one of the two changed values): `rai.error_handedness = "none"`

The meaning of `error_handedness` is not written anywhere in `src/conventions.py`. The suite
(`decide_error_handedness`) applies two different tests:

- For delama, barfoot and gtsam it executes the printed `X X̂⁻¹` / `X̂⁻¹X` composition.
- For rai it requires the evaluated residual matrix of Eq. (19) to change under both left and
  right translation.

Rai's residual is left-*equivariant*. I measured `r(SR_i, SR_j) = S·r(R_i, R_j)` to 4e-16, so the
minimised cost is left-invariant:

- `‖r‖` is 3.0717 under the original states and 3.0717 under left translation;
- under right translation it is 2.8703.

The cost (19) replaces, Eq. (12a), is `Log(ΔR̃ᵀR_iᵀR_j)`, which is the `X̂⁻¹X` (left-invariant)
form. Read as "a group error printed in the residual", `"none"` is right. Read as "the invariance
of the minimised cost" (the reading the implementer used to justify (b)), rai would be
`"left-invariant"`, and the suite currently kills that value. This is **AMBIGUOUS-FIELD-MEANING**
and is filed as M0-F11 below. It does not affect (a) or (b), whose values are the same under
both readings.

---

## 4. Rulings on pass-2 findings

| finding | ruling | evidence |
|---|---|---|
| **M0-F06** (18 surviving corruptions) | **FIXED** for the 18 filed; a small residual is filed as M0-F09/F10 | All 18 pass-2 survivors are in `PASS2_SURVIVORS`. For `rai.bias_correction_side` the pass-2 value `"right"` is now the true value, so the two alternatives `"none"` and `"left"` are mutated instead (19 entries). All are killed in the full run (130/130 green, which includes `test_mutation_is_killed[*]`). The identity, null and gross-corruption controls pass, so the harness is live. My own 136 new mutants (section 6) leave 4 non-equivalent survivors, all low severity, and none of them is one of the 18. |
| **M0-F07** (invented sub-labels; unsourced citation; fixture size) | **FIXED** for the three items filed; a residual is filed as M0-F09 | See the three items below the table. |
| **M0-F08** (tests that cannot fail / lean on the deriver) | **PARTIALLY FIXED** | Most items are fixed; three are not. Both lists follow the table, and section 7 has the full residual list. |

**M0-F07, item by item:**

1. **Sub-labels.** I diffed the suite's `source_equations()` key set against my own count of
   printed `ltx_tag_equation` tags. The difference is empty both ways for rai, delama and barfoot.
   `25c` (rai) and `30b` (barfoot) now raise. The misfiled rai rows are fixed: (25a) is
   `R_i r^v_ij …` and (25b) is `R_i r^p_ij …`, (37b) is `z^p_i = …`, and the `Δv̄`/`Δp̄` rows of
   block 37 are no longer filed under (37b).
2. **(14a).** The excerpt is genuine. I fetched the PDF independently (same sha256 and byte
   count), re-ran the exact recorded command
   `pdftotext -layout -f 4 -l 4 -x 0 -y 0 -W 302 -H 794`, and kept lines 36–89. The result is
   **byte-identical** to the committed excerpt body.

   The PDF prints `(14a)`. `pdftotext -bbox` places it at y = 369–378 pt. The `C(ϕ) = cay(ϕ^)`
   row is at y = 348–358 and the `ϕ = cay⁻¹(C)` row, which carries `,(14b)`, is at y = 383–393.
   The suite resolves `(14a)` by matching the LHS printed in the PDF just before the tag
   (`C(ϕ)`) to the unique untagged row of HTML block 14 with that LHS. It does not count rows.
   That is LHS matching, not row-position inference. The only positional ingredient is text
   order inside the PDF excerpt, and (14a) is unambiguous because (14b) sits on the other row.
3. **Citation.** `IJRR 44(3):366-387, 2025` is sourced twice:
   - the cached Crossref record (volume 44, issue 3, page 366-387, published-print 2025-03; I
     re-fetched it and it is identical);
   - rai's own ref. [6], which prints "Int. J. Robot. Res. 44 (3), pp. 366–387".
4. **Fixture size.** 462 407 B is the byte count and 456 935 the UTF-8 character count. I
   re-measured both, and a live re-fetch of the rai URL today is byte-identical
   (sha256 `2a66413a…c4c66a`).

**M0-F08, fixed:**

- The C06 numeric theorem block is deleted.
- C03 now evaluates (15), (19) and (22a–c) from the source at the table's labels, so a moved
  label fails. I checked this: `lifting_definitions_Gamma = "22"` and `xi_definition = "16"` are
  killed.
- The YAML-vs-YAML threshold checks in C14 and C15 are deleted.
- `isinstance(phi, NavState)` is deleted.
- The "not in a numbered equation" scans are deleted.
- The O7 verdict is now computed by composing both routes in gtsam.
- `delama.group`, `delama.manifold` and `rai.group` are read from source prose. The mutant
  `manifold = "SE_2(3) x R^9"`, an intermediate manifold the source also prints, is killed.
- The C10 barfoot residual is now the minimised costs (25) and (28).

**M0-F08, not fixed:**

- The C15 reversed-product half still reads no table value. The test now says so and keeps it
  deliberately, as evidence for a DERIVED claim.
- Three SENTINEL prose patterns are constants.
- One C06 assertion degenerated to a constant.

**provenance.json.** I re-hashed all six files: bytes, UTF-8 character count and sha256 all match
for every entry.

| file (decompressed) | bytes | utf8 chars | sha256 prefix |
|---|---|---|---|
| rai_2609.38048v1.html | 462 407 | 456 935 | 2a66413a4ae8 |
| delama_2411.05548v8.html | 641 374 | 630 251 | efc2c26da284 |
| barfoot_2308.12418v2.html | 650 414 | 643 088 | 63ec40f48bef |
| gtsam_galileanimufactor.html | 1 692 613 | 1 688 785 | eef019ebd6a3 |
| barfoot_…_pdf_p4_cayley.txt | 3 562 | 3 471 | a1e7a00dbd29 |
| crossref_10.1177_….json | 20 759 | 20 759 | 86c435a73a7d |

### Trying to make the parser emit an unprinted label

- The **index** (`source_equations`) never does. I compared it with the printed tags for all
  three arXiv sources.
- The **resolver** (`equation_latex`) does accept bare parent labels of sub-numbered blocks.
  Probing every `n` / `n[a-g]` that is not a printed tag, it resolves:
  - rai `8, 9, 10, 11, 12, 21, 22, 24, 25, 26, 27, 30, 37`;
  - barfoot `14, 15, 16, 30, 38, 40, 47, 48, 50, 54, 58, 59, 66, 79, 90, 95, 96`, plus `14a`,
    `15a` and `16a`, which come from the PDF and are printed there.

  Several of these bare labels are printed nowhere in the cached sources: not as a tag, not in
  an HTML cross-reference, and for barfoot `(14)` not in the full PDF text either. In rai,
  `(21)`, `(22)`, `(24)`, `(25)`, `(26)` and `(27)` are never cited; the prose cites only
  `21b`, `21c`, `22c`, `26a` and `26b`.

  Two table values use such labels:
  - `rai.eq.u_definitions = "26"`;
  - the prose item `barfoot.exact = "cayley lifting Eqs. (14), (15)"`.

  This is the conventional subequations parent, not an invented sub-letter, so I do not count it
  against F07. It does let two plausible corruptions survive (M0-F09).

---

## 5. Independent field checks (≥15 required; 70+ done, none through the suite's parsers)

All of these match the table.

- **rai**:
  - `eq` labels: `cayley 3`, `cayley_right_jacobian 4`, `qcqp 5`, `sdp 6`,
    `chordal_residual 14`, `xi_definition 15`, `bias_correction 15`, `cayley_determinant 17`,
    `denominator_clearing 18`, `rotation_residual 19`, `21a/21b/21c`, `22a/22b/22c`,
    `isotropic_weighting 23`, `u_definitions 26` (block 26a/26b), `lifted_vector 28` (order),
    `lifting_constraints 30`, `homogeniser 31` (three families), `A1..C1 = 32..35` with
    cardinalities, `state_set 36`.
  - `bias_ordering` is the untagged `b ≜ [b_gᵀ b_aᵀ]ᵀ` and `bias_order (b_g, b_a)`.
  - O4.b `8a` is `ω̃ = ω + b_g + η^g`.
  - The rotation-residual terms and the `rank1_criterion` prose.
  - All five ablation figures, and the "prolonged stationary segment, leaving the heading
    unobservable" cause.
- **delama**:
  - `eq` labels 1, 2, 3, 4, 5, 6 (time row `[0,0,0,1]`), 8, 9, 10, 17, 18 (`−½gΔt²`, `−Δt`), 19,
    22, 24, 26, 29, 30, 31, 34, 35, 36, 37, 39.
  - No tag `(14)` is printed, so `continuous_time_ins = could not extract` is right; the source
    refers to it as "( IV-A )".
  - `inverse_formula`, `semidirect_product_law`, `bias_order`, the virtual-bias kinds, `Q_d` and
    the right action.
- **barfoot**:
  - `eq` labels 5, 8, 10 (`[ρ;φ]`), 11, 14b, 15, 17, 18, 19, 20, 25, 28; `14a` via the PDF.
  - `element_order (C, r)`.
  - The log-SVR, sparsity and LICQ prose.
- **gtsam doc**: Eqs. 1–47, with every recorded label matching (own enumerator scan). Eq. (1)
  gives `β = [b_a; b_ω]`, (4) the inverse, (5) the tangent and (38) `P_N`.
- **gtsam runtime**:
  - O1–O4 match.
  - The retraction is `X·Exp(δ)`: `X.retract(δ)` differs from `X·Exp(δ)` by 0.0 and from
    `Exp(δ)·X` by 0.025.
  - `Gal3.Dim() == 10`.

---

## 6. Own mutation sweep (136 mutants, none of whose (path, value) pairs is in `tests/test_mutation.py`)

**Harness.** The mechanics are the same as the suite's own harness:

- The live table is mutated in-process and every `test_*` in `tests/test_conventions.py` is run;
  any exception counts as a kill.
- The table is restored and compared with a deep snapshot after every mutant.
- The null run is clean.
- I asserted programmatically that 0 of the 136 (path, value) pairs occur in
  `ALL_MUTANTS`, and no `UNVERIFIED_COMMENTARY` path is mutated.
- `src/` was never written to, and `git status` was clean afterwards.

The mutants are plausible, wrong values:

- real neighbouring labels and bare parent labels;
- labels the HTML never prints (`delama.eq.continuous_time_ins = "14"`);
- digit labels where the source prints prose;
- the other paper's value, swapped orderings and sign flips;
- inverted conclusions inside prose fields, and appended contradictions;
- citation date, author-order and page changes.

**Result: 131 of 136 killed, 5 survived (1 equivalent, 4 genuine and low severity).**

| field | mutant | ruling |
|---|---|---|
| `rai.eq.rotation_residual_restated` | `"21"` (bare block; rai never prints "(21)") | **survivor**. The C02 check accepts any block that contains a parseable `r^R` row. → M0-F09 |
| `barfoot.eq.cayley_inverse` | `"14"` (block holding both the forward and inverse maps; barfoot never prints "(14)") | **survivor**. The signature only requires `cay^{-1}` to be present. The mirror mutant `cayley = "14"` is killed. → M0-F09 |
| `rai.isotropy_reason` | "…so the scalar precision is **a tuning choice, not structurally forced**" | **survivor**. The C06 regex reads only the identity clause, and the conclusion C06's statement asserts is unread. → M0-F10 |
| `delama.linearized` | "error dynamics Eqs. (35)-(**36**)" (drops B̂, Eq. 37) | **survivor**. Incomplete rather than false. The endpoint 36 is a recorded key that the item names. → M0-F10 |
| `gtsam.linearized` | "bias correction Eq. (**36**), first order" | **equivalent**. (36) restates (34) and is also first order (`≃`). It is not a defect. |

Notable kills:

- `delama.manifold = "SE_2(3) x R^9"` (the source prints it as an intermediate);
- `barfoot.eq.cayley = "15a"` and `cayley_se3 = "16a"`, both PDF-printed sub-labels;
- `barfoot.eq.pose_averaging_cost ∈ {"37", "46"}`, two other `T T̃⁻¹` costs in the same paper;
- `gtsam.eq.transition = "28"`, the boxed `ε⁺ = Aε + Bη`;
- `rai.eq.bias_correction ∈ {"11", "14"}`, the Exp-form corrections;
- `delama.eq.continuous_time_ins = "14"`;
- all eight digit-for-sentinel mutants;
- `O1.consequence + "; this is harmless"` and `O7.reason + " SE_2(3) has the same coupling."`;
- `delama.inverse_formula` set to the M0-C17 form;
- the author-order swaps in all three arXiv citations, and the barfoot page range `366-386`.

---

## 7. Tests that cannot fail, and tests leaning on the deriver

### Cannot fail under any table mutation (they fail only if a fixture or the library changes)

- **T-1, C15**: `reversed_err > bar`. It reads the claim's `H′` and `T(−dt)` and the source's
  Γ, and no table value. The test says this is deliberate, as evidence for the DERIVED claim.
  Only the `O7.a.says` array-equality line reads the table.
- **T-2, C06**: `assert rai["isotropy_required"] is ("isotropic weight" in disp.group(2))`. The
  `disp` regex already forces `group(2)` to be the literal "This equality requires an isotropic
  weight", so the right-hand side is the constant `True`. The line is `isotropy_required is True`:
  it can fail on a table mutation, but its "source decides" half is vacuous.
- **T-3, SENTINEL_EVIDENCE**: the constant prose patterns for `delama.eq.continuous_time_ins`,
  `barfoot.eq.sparsity` and `barfoot.eq.licq_failure` are not built from any table value. Only
  their `== COULD_NOT_EXTRACT` half reads the table.
- **T-4, O7 block of `test_seven_ordering_incompatibilities_…`**: three lines check the fixture
  and the library, not the table:
  - `X_i,X_j\in\mathrm{SE}_2(3)` in the raw GTSAM page;
  - the NavState composition position law;
  - `localCoordinates(...).size == Gal3.Dim() − 1`.
- **T-5, instrument tests**: `test_fixture_provenance` and `test_equation_parser_never_invents_a_label`
  read no table value, by design. The second one does not cover `equation_latex`'s bare-parent
  resolution (M0-F09), so it passes while the resolver still accepts unprinted labels.

### Leaning on the deriver's or test author's choices rather than a source

- **L-1, `decide_error_handedness`**. It encodes a field meaning that is written nowhere in the
  table: printed group composition for three rows, evaluated residual non-invariance for rai.
  Under the cost-invariance reading, rai would be `"left-invariant"` (M0-F11).
- **L-2, `_ERROR_FORMS` / `_BIAS_ARGUMENT` / `_NOMINAL`**. These are per-source hand-written
  regexes; the barfoot ones transcribe Eqs. (25) and (28) literally. `decide_bias_correction_side`
  decides the side from the text order of two regex hits inside the RHS. That order is correct
  for every source here, but it is a heuristic.
- **L-3, `check_prose_equation_refs`**. It decides which recorded key a prose item "names" by
  4-letter-stem word overlap (`_overlap`), and "first order" by `approximat|first.order` in the
  400 characters before the tag. The F10 survivor `delama.linearized (35)-(36)` passes through
  this.
- **L-4, `EQ_SIGNATURES`**. These are deriver-written regexes; two are one-sided
  (`cayley_inverse`, `rotation_residual_restated`).
- **L-5, C01**. The source scans for `element_order` and `tangent_order` run only
  `if … is None`. A non-None mutant is killed only by the comparison with the claim statement
  (the YAML).
- **L-6, `assert_claim_statement`**. Throughout the suite it compares the table with the
  deriver's claim YAML. This is by design (the claim is the spec), and every such field also has
  a source check, except those noted in L-5.

---

## 8. New findings

| id | severity | finding |
|---|---|---|
| **M0-F09** | low | `equation_latex` resolves bare parent labels of sub-numbered blocks whether or not the source ever prints them. Rai never prints "(21)", "(22)", "(26)" etc.; barfoot never prints "(14)", even in the PDF. As a result `rai.eq.rotation_residual_restated = "21"` and `barfoot.eq.cayley_inverse = "14"` survive. Two table values rely on such labels: `rai.eq.u_definitions = "26"` and `barfoot.exact … Eqs. (14), (15)`. The fix is either to record printed labels (`"26a"`/`"26b"`, `"14a"`, `"14b"`), or to accept a bare parent label only when the source cites it bare (rai: 9, 10, 11, 30, 37; barfoot: 15, 16, …), and in either case to add parent-label probing to `test_equation_parser_never_invents_a_label`. |
| **M0-F10** | low | Two prose clauses are still unread. (i) The "structurally forced, not a tuning choice" conclusion inside `rai.isotropy_reason` can be inverted without failing anything, even though the conclusion is what M0-C06 asserts. (ii) `delama.linearized` can drop Eq. (37) from its range. |
| **M0-F11** | low–medium | `error_handedness` has no written definition. The suite enforces rai = `"none"` with a different test than it uses for the other rows. Under the cost-invariance reading the implementer used to justify the barfoot change, rai's minimised Eq. (19)/(20) cost is left-invariant (and Eq. 12a is `X̂⁻¹X`). The table should either state the field meaning ("handedness of a printed group-valued error; `none` if the residual composes no inverse") or record rai as `"left-invariant"`. Today a future edit could flip either way and only the suite's private definition decides. |

---

## 9. Reproduction

The scripts are in the verifier scratchpad. They read only fixtures copied there, or files
fetched into fresh directories.

| script | what it does |
|---|---|
| `dump.py` / `prose.py` | LaTeXML row/tag/alttext extractor and prose renderer (independent of the suite) |
| `gt.py` | GTSAM mystmd enumerator→value scan |
| `parsercheck.py` | diff of the suite's index against printed tags, and the unprinted-label probe of `equation_latex` |
| `mut3.py`, `mut4.py` | 110 + 26 mutants with the snapshot-restore assertion, and a check that no pair is in `ALL_MUTANTS` |

These checks were run inline: M0-C17 (20 000 draws), M0-C14 (200 trajectories), M0-C15 (5000
draws), M0-C03 (sympy), M0-C12 (spectrum anchor), O1–O4 and the retraction (runtime), the PDF
re-extraction and `-bbox` check, and the rai and Crossref re-fetches.

After committing this report, I ran the full suite on the clean tree:
`pytest -q` → **130 passed**.
