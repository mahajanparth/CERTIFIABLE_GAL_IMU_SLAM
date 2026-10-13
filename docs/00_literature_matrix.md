# M0 — Literature matrix

A cited mathematical comparison of the four source works this project builds on. Produced by the
M0 **deriver**. Nothing here is implemented; the numbered claims in
[`claims/M0.yaml`](claims/M0.yaml) are what the implementer binds tests to.

**Tag legend** (every statement below carries one):

| tag | meaning |
| --- | --- |
| `VERIFIED-PUBLISHED` | read out of the cited source at the cited equation / section, by direct inspection of that source |
| `DERIVED` | algebra or a numerical experiment performed here, on top of published material |
| `ASSUMPTION` | a choice this project makes that the sources do not force |
| `OPEN-HYPOTHESIS` | believed, not established; must not be relied on downstream |

**Hard rule observed throughout:** no equation number is guessed. Where a source states something in
prose, a caption, or an unnumbered display, the equation-number field reads **`could not extract`**
and the prose location is given instead.

---

## 1. Sources, exactly as read

| key | citation | what was read | retrieved |
| --- | --- | --- | --- |
| `rai` | U. Rai, Z. Xu, B.-S. Chen, D. M. Rosen, *A QCQP-Representable IMU Pre-Integration Factor for Certifiable State Estimation*, **arXiv:2609.38048v1 [cs.RO], 29 Sep 2026** (Robust Autonomy Lab, Northeastern University) | full LaTeXML HTML render `arxiv.org/html/2609.38048v1`, equations via MathML `alttext` | 2026-10-12 |
| `delama` | G. Delama, A. Fornasier, R. Mahony, S. Weiss, *Equivariant IMU Preintegration with Biases: a Galilean Group Approach*, **arXiv:2411.05548v8 [cs.RO], 21 Aug 2025**; RA-L, DOI `10.1109/LRA.2024.3511424` | full HTML render `arxiv.org/html/2411.05548v8` | 2026-10-12 |
| `barfoot` | T. D. Barfoot, C. Holmes, F. Dümbgen, *Certifiably Optimal Rotation and Pose Estimation Based on the Cayley Map*, **arXiv:2308.12418v2, 31 May 2024**; IJRR **44**(3):366–387, 2025 | full HTML render `arxiv.org/html/2308.12418v2` | 2026-10-12 |
| `gtsam` | GTSAM Galilean preintegration docs: `borglab.github.io/gtsam/galileanimufactor/` and `.../galileanimufactornees/`, plus the **installed `gtsam 4.3.0`** Python package in `.venv` | both doc pages; API behaviour exercised directly against `gtsam 4.3.0` | 2026-10-12 |

> `ASSUMPTION` — the two GTSAM doc pages carry no version banner; they are the `develop` docs. Every
> ordering and relation attributed to `gtsam` below was **re-verified at runtime against the
> installed `gtsam 4.3.0`**, so the doc/package correspondence is checked rather than assumed. The
> volume/issue/page numbers for Delama (RA-L 10(1):724–731) are **not** present in the arXiv HTML —
> only the DOI is — so they are carried over from the project brief, not from the source.

---

## 2. Master convention-comparison table

> **Superseded rows (orchestrator note, 2026-10-13).** Two rows of the table below are out of date with
> `src/conventions.py`, which is authoritative and source-tested. Under the project definition of
> *error handedness* (invariance of the minimised cost under group translation; see the
> `src/conventions.py` docstring), Rai is **left-invariant**: r(S R_i, S R_j) = S r, so the cost is unchanged
> under left translation (verify-M0-003, impl-M0-005). Rai's bias correction is on the **right**, Eq. (15).
> Barfoot's minimised costs Eqs. (25)/(28) are X X~^-1 errors, so Barfoot is **right-invariant**.


| | `rai` | `delama` | `barfoot` | `gtsam` (and this project) |
| --- | --- | --- | --- | --- |
| **Group / manifold the estimate lives on** | `SO(3)` **only**; `v_i, p_i ∈ R^3` are plain Euclidean variables. State set Eq. (36). No `Gal(3)`, no `SE_2(3)`. | Manifold `M = Gal(3) × R^10` (Sec. IV-C); *symmetry* group `G = Gal(3) ⋉ gal(3)`, **dim 20**. Keyframe states live on `SE_2(3) × R^9` (Sec. IV-A). | `SO(3)` (Eq. 5) and `SE(3)` (Eq. 8); also `Ad(SE(3))` (Eq. 11), unused. No velocity in the state. | `M = Gal(3) × R^6` **direct product**, Eq. (1) of `galileanimufactor`. Keyframe states stay `SE_2(3)` (`NavState`). |
| **Group-element ordering (matrix)** | n/a — no 5×5 element. `ΔR̄_ij ∈ SO(3)`, `Δv̄_ij, Δp̄_ij ∈ R^3` carried separately. | `X = [[A, a, b],[0,1,c],[0,0,1]]`, Eq. (3); in the preintegration instance `(a,b,c) = (Δv, Δp, Δt)`, Eq. (19). **Velocity column before position column.** | `T = [[C, r],[0,1]]`, Eq. (8). 4×4, no velocity, no time. | `X = [[R, v, p],[0,1,t],[0,0,1]]`, Eq. (2). **Velocity column before position column** — identical layout to `delama` Eq. (3). |
| **Constructor / tuple ordering** | n/a | tuple written `(ΔR_k, Δv_k, Δp_k, Δt_k)` — matches the matrix. | n/a — no composite constructor; the element is just `T = [[C, r],[0,1]]`, Eq. (8). | `Gal3(R, r, v, t)` — **position `r` before velocity `v`**, i.e. *opposite* to the matrix. `NavState(R, p, v)` — position before velocity. |
| **Tangent (algebra) ordering** | n/a — the only tangent object is `ξ_ij ∈ R^3` (gyro-bias correction). | `x = (ω, v, r, α) ∈ R^10`, Eq. (4): rot, **velocity-slot**, position-slot, time. | `ξ = [ρ; φ] ∈ R^6`, Eq. (10): **translation first**. | `x = (ω, ν, ρ, α) ∈ R^10`, Eq. (5): rot, **velocity**, position, time — *componentwise identical to* `delama` Eq. (4). But `NavState` tangent is `(δθ, δp, δv) ∈ R^9`, Eq. (37) — **position before velocity**. |
| **Error handedness** | No group error at all. Chordal/Frobenius matrix residual Eq. (19) plus Euclidean residuals Eqs. (21b)(21c). "Left vs right" does not apply. | **Navigation error `Υ_k Υ̂_k^{-1}` ⇒ right-invariant**, Eq. (34). The underlying *state action* `φ` of Eq. (24) is a **right action** (stated in the sentence immediately after Eq. 24: "*φ is a transitive and free right group action of G on M*"). | Not an error-state formulation; cost is built directly on the lifted variables. | **Left-invariant error `e = ξ̂^{-1} ξ`**, Eq. (21); equivalently right perturbation `X ⊕ δ = X Exp(δ)`, §1 table. The doc states the contrast explicitly (§1 prose). |
| **Bias model & dimension** | `b_m = [b_g; b_a] ∈ R^6` per bias **block**, **gyro first** (unnumbered display after Eq. 10c; block form in Eq. 36). `M` blocks over `K` keyframes, random walk Eq. (37d), prior Eq. (37c). | `b_k = (b_ω, b_a, b_ν, b_ρ) ∈ R^10`, **gyro first**: 6 physical + **3 virtual velocity** + **1 virtual time** (Sec. IV-C). | **No bias.** | `β = [b_a; b_ω] ∈ R^6`, **accel first**, Eq. (1) — exactly `gtsam::imuBias::ConstantBias`. "No additional bias coordinates." |
| **Bias correction applied on which side** | Neither — the correction is absorbed algebraically: `Cay(ξ_ij)` right-multiplies `ΔR̄_ij` in Eq. (16), then the denominator is cleared (Eq. 18) and the product is *lifted*, Eq. (22c). | **LEFT**: `Υ̂⁺ ≈ exp((J_Υ Δb̂)^∧) Υ̂`, Eq. (39). | n/a | **RIGHT**: `Υ(β̂+δβ) ≃ Υ̂(β̂) Exp(J δβ)`, Eq. (34)/(36); recursion `J_{k+1} = Ad_{V_k^{-1}} J_k − C_k`, Eq. (35). |
| **Exact vs linearized** | Cayley residual Eq. (19) is **exact and polynomial** in `(R_i, R_j, ξ_ij)`. Bias→measurement maps `A^{vg}, A^{va}, A^{pg}, A^{pa}` are **first-order** Jacobians at `b̄` (Eqs. 21b, 21c). Cost weighting is an **approximation** (Eq. 23). | Mean propagation Eq. (22) is **exact** for held input. Error dynamics Eq. (35) with Eqs. (36)(37) are **linearized about `ε_k = 0`**. Bias update Eq. (39) is **first order**. | Cayley lifting is **exact**; the relaxation Eq. (20) is the only approximation. | Held-input Galilean composition Eq. (12)/(14) and endpoint Eq. (17) are **exact**; linearization Eqs. (25)–(29) and bias correction Eq. (34) are **first order**. Rotating-frame Eq. (47) is **exact for constant rate**. |
| **Does `exp`/`log` appear essentially?** | **No.** `Log` is removed by the chordal residual Eq. (14); `Exp` is removed by the Cayley map Eq. (3). Even the nominal `ΔR̄_ij` is a *product of Cayley maps* (§IV-A replaces `Exp` and `J_r` by `Cay` and `J_Cay` throughout Eqs. 9a–9c), so no trigonometry enters at all. | **Yes.** `log_G` Eq. (31) needs `Gal(3)` `log` Eq. (9) **and** `J_L(·)^{-1}` Eq. (10); the error in coordinates Eq. (34) contains `J_L(log(Υ_k Υ̂_k^{-1})^∨)^{-1}`. `Γ_1, Γ_2` are trigonometric (`κ_1, κ_2, κ_3` in Sec. III). | **No.** Cayley Eqs. (14)/(15) replace the exponential wholesale. | **Yes.** `Exp` Eq. (6) uses `J_L(ω)` and `Γ_2(ω)`; the factor residual Eq. (42) is `NavState::logmap`. |
| **Homogenisation device** | Scalar `s` with `s² = 1`, Eq. (31); gravity/known terms carry `s` (Eqs. 25a, 25b). | n/a | `x^T A_0 x = 1`, Eq. (18). | n/a |
| **Cayley convention** | `Cay(φ) = (I − ½φ^)^{-1}(I + ½φ^)`, `φ = 2 tan(θ/2) a`, Eq. (3). Right Jacobian Eq. (4). | not used | `cay(A) = (I − ½A)^{-1}(I + ½A)`, Eq. (14a); the rotation vector is `2 tan(angle/2) · axis` (Eq. 14b text), the same half-angle convention as `rai` Eq. (3). Also `cay(ξ^)` on `SE(3)`, Eq. (15). | not used |
| **Certifiable?** | **Yes** — QCQP Eq. (5)/SDP Eq. (6), rank-1 certificate. | **No** — `log`/`J_L^{-1}` are not polynomial. | **Yes** — Eqs. (17)–(20). | **No**, and claims none. |

All cells above: `VERIFIED-PUBLISHED` unless the cell itself says otherwise.

---

## 3. Ordering incompatibility register

These are the places where two sources (or two APIs inside one source) disagree about what the
coordinates *mean* while agreeing about their *shape*. Every one of them produces a silently wrong
number rather than an exception.

| # | where | A says | B says | consequence if conflated |
| --- | --- | --- | --- | --- |
| **O1** | `gtsam` matrix vs `gtsam` constructor | `X = [[R, v, p],[0,1,t],[0,0,1]]`, Eq. (2) — **velocity in column 4**, position in column 5 | `Gal3(R, r, v, t)` — **position is the 2nd argument**, velocity the 3rd | `v` and `p` swap. Verified at runtime: `Gal3(R, [1,2,3], [10,20,30], 7).matrix()` puts `[10,20,30]` in column 4. `VERIFIED-PUBLISHED` + runtime-checked |
| **O2** | `gtsam` `Gal3` tangent vs `gtsam` `NavState` tangent | `Gal3`: `(ω, ν, ρ, α)` — **velocity block before position block** (Eq. 5) | `NavState`: `(δθ, δp, δv)` — **position block before velocity block** (Eq. 37) | the 9-of-10 selection is a selection **and a permutation**: `P_N` in Eq. (38) picks `Gal3` rows `[0:3, 6:9, 3:6]`. Dropping the permutation silently swaps the velocity and position covariance blocks. `VERIFIED-PUBLISHED` + runtime-checked (`NavState.retract(e_3)` moves position; `Gal3.Hat(e_3)` fills the velocity column) |
| **O3** | `gtsam` bias vs `rai` / `delama` / this project | `imuBias::ConstantBias(accel, gyro)`, `vector() = [b_a; b_ω]` — **accel first** (Eq. 1; verified: `ConstantBias([1,2,3],[40,50,60]).vector() == [1,2,3,40,50,60]`) | `rai`: `b = [b_g; b_a]` (display after Eq. 10c); `delama`: `b = (b_ω, b_a, b_ν, b_ρ)` (Sec. IV-C); **this project fixes `[b_g; b_a]`** | the two 3-blocks swap. Accelerometer bias gets applied to the gyroscope. **One explicit permutation at the GTSAM boundary, in one place.** `VERIFIED-PUBLISHED` + runtime-checked |
| **O4** | `gtsam` IMU sample order | `u = [a; ω]`, Eq. (10), "same accelerometer-then-gyroscope ordering as the bias"; `integrateMeasurement(measuredAcc, measuredOmega, deltaT)` | `rai` Eqs. (8a)(8b) list `ω̃` then `ã` | same swap as O3, on the *measurements* instead of the biases. The lift matrix `E`, Eq. (20), is exactly this permutation, `R^{10×6}`. `VERIFIED-PUBLISHED` |
| **O5** | `barfoot` `SE(3)` tangent vs everything else | `ξ = [ρ; φ]` — **translation first**, Eq. (10) | `gtsam`/`delama` put rotation first in every tangent vector | any constraint template copied from `barfoot` into a `Gal(3)` lifting lands in the wrong block. `VERIFIED-PUBLISHED` |
| **O6** | naming collision: `r` and `ρ` | `barfoot`: `ρ` = **translation** coordinate (Eq. 10); `gtsam` `Gal3(R, r, …)`: `r` = **position** | `delama` Eq. (4): `r` = the **position-slot** tangent coordinate; `v` = velocity-slot — so `delama`'s `(ω, v, r, α)` **is** `gtsam`'s `(ω, ν, ρ, α)` componentwise | the letters agree with `gtsam` by luck, not by convention. Do not infer ordering from a letter. `VERIFIED-PUBLISHED` |
| **O7** | gravity block sign in the endpoint relation | `delama` Eq. (18): `Γ_ij = [[I, gΔt, **−**½gΔt²],[0,1,−Δt],[0,0,1]]`, applied on the **left** of a `Gal(3)` product | `gtsam` Eq. (15): `W_ij = NavState(I, **+**½gΔt², gΔt)`, applied on the left of an `SE_2(3)` product after the coasting map `φ_Δt` | **not a contradiction.** The sign difference is exactly the Galilean `t₂·v₁` coupling term of Eq. (3): with `t₂ = Δt` and `v₁ = gΔt` the left `Gal(3)` factor re-contributes `+gΔt²`, so the stored entry must be `−½gΔt²` to land on `+½gΔt²`. Using `+½gΔt²` inside a `Gal(3)` left factor is off by exactly `+gΔt²` — **this is pre-flight defect F2.** `DERIVED` (verified numerically, §6) |

---

## 4. Source-by-source detail

### 4.1 `rai` — Cayley on `SO(3)` only, biases lifted

`VERIFIED-PUBLISHED` unless marked.

- **Scope.** Cayley is applied **only to the rotation**: Eq. (3) defines `Cay: R^3 → SO(3)`, Eq. (15)
  writes `R_i^T R_j = ΔR̄_ij Cay(ξ_ij) Cay(−δφ_ij)`, and `v, p` never leave `R^3` (Eqs. 21b, 21c, 36).
- **Zero Galilean content.** The archived source text contains **0** occurrences of `Galilean`,
  `SE_2(3)` (in any LaTeX spelling: `SE_2`, `\mathrm{SE}`, `SE}_{2`), `extended pose`, `equivariant`,
  `Delama`, `Fornasier`, `Weiss`. `Mahony` appears **once**, only as a co-author of reference [3]
  (Shonan rotation averaging). `Barfoot`/`Dümbgen` appear as reference **[6]** — i.e. `rai` cites
  `barfoot`, and cites it twice, both in §I.
- **Rotation residual, Eq. (19):**
  ```
  r^R_ij  =  (R_j − R_i ΔR̄_ij)  −  ½ (R_j + R_i ΔR̄_ij) ξ_ij^
  ```
  with `ξ_ij ≜ J^R_ij δb_g` (Eq. 15), **`δb_g` is the bias *perturbation*** `b_g − b̄_g`, not the bias.
  Eq. (21a) restates it verbatim. The map is polynomial because Eq. (17) gives
  `det(I − ½ξ^) = 1 + ¼‖ξ‖² > 0`, so the denominator of Eq. (16) can be cleared (Eq. 18).
- **The `Exp` → `Cay` swap is forced in two places, not one.** §III-C: (i) the geodesic residual
  Eq. (12a) contains `Log` — replaced by the chordal residual Eq. (14); (ii) the *bias correction*
  `Exp(ξ_ij)` is still trigonometric in the unknown `b_g` — this is what Cayley fixes. Replacing only
  (i) leaves a non-polynomial problem.
- **Lifted decision vector, Eq. (28):**
  ```
  x = [ {vec(R_i), v_i, p_i}_i , {b_m}_m , {vec(Y_ij), vec(Y'_ij), W^v_ij, W^p_ij}_(i,j) , s ]^T
  ```
  with `Y_ij ≜ R_i ΔR̄_ij (J^R_ij b_g)^` and `Y'_ij ≜ R_j (J^R_ij b_g)^` (Eq. 22c) — note these use the
  **full bias `b_g`**, not `δb_g` — and `W^v_ij ≜ R_i u^v_ij`, `W^p_ij ≜ R_i u^p_ij` with
  `u^v_ij ≜ A^{vg}_ij b_g + A^{va}_ij b_a`, `u^p_ij ≜ A^{pg}_ij b_g + A^{pa}_ij b_a` (Eq. 26).
- **Primary constraints, Eqs. (30)–(31):** lifting definitions `(R_iΔR̄_ij)^T Y_ij = (J^R_ij b_g)^ s`,
  `R_j^T Y'_ij = (J^R_ij b_g)^ s`, `R_i^T W^{v,p}_ij = u^{v,p}_ij s` (Eq. 30), plus `s² = 1`,
  `R_i^T R_i = s² I_3`, and the **cross-product handedness** `R_i^{(a)} × R_i^{(b)} = s R_i^{(c)}`
  over cyclic `(a,b,c)` (Eq. 31). This matches pre-flight F3: absolute rotations need the full
  `SO(3)` lifting including handedness.
- **Redundant constraints (Table I):**

  | id | identity | equation | cardinality |
  | --- | --- | --- | --- |
  | A1 | `R_k R_k^T = s² I_3` (row orthonormality) | **Eq. (32)** | 6 / node |
  | B1 | `Y^{(′)}_ij w_ij = 0`, `w_ij ≜ J^R_ij b_{g,m(i,j)}` | **Eq. (33)** | 3 / lifting ⇒ 6 / edge |
  | B2 | `(Y^{(′)}_ij)^T Y^{(′)}_ij = ‖w_ij‖² I_3 − w_ij w_ij^T` | **Eq. (34)** | `could not extract` — Table I and Eq. (34) give no per-edge count |
  | C1 | `‖W^•_ij‖² = ‖u^•_ij‖²`, `• ∈ {v,p}` | **Eq. (35)** | 2 / edge |

- **Ablation (§VII-B).** `K = 8`, `M = 2`, `n = 340`; 2 motion primitives × 7 noise levels × 12 seeds
  × 5 cumulative stages = **840** SDP solves. Through B1 "the relative optimality gap remains at
  least **0.88**, and none of the corresponding **504** solves is certified". **Adding B2
  substantially reduces both the numerical rank and the optimality gap.** With the full set including
  C1, median gap **2.8 × 10⁻⁵** and median numerical rank 1. Same ordering at `K = 16` (`n = 652`).
  **B2 is the decisive subset** — confirmed.
- **Isotropic weighting is structurally forced, not a tuning choice.** §IV-B states the device first —
  `τ‖r‖² = τ‖R_i r‖²` for `R_i ∈ SO(3)` — then: *"This equality requires an isotropic weight. We
  therefore approximate the three covariance blocks by scalar precisions"*, Eq. (23)
  `τ_{R,ij}^{-1} = ⅓ tr Σ^φ_ij` etc. Premultiplying by `R_i` is what removes the `R_i^T v_j`,
  `R_i^T p_j` products; the isotropy is the price. "This approximation drops the cross-correlations
  and directional weighting in the original covariance."
- **Numerical-rank criterion.** `rai` calls the relaxation numerically tight when
  `rank(Ẑ) = 1`, counting eigenvalues above `10⁻⁶ λ_max(Ẑ)` (§VII-A). This is a **different test**
  from `barfoot`'s log-SVR (§4.3) — see §5.
- **Problem sizes (Table II).** `n ∈ {334, 646, 652, 958, 970}`; 35 of 36 real-world windows
  certified rank-1. The one failure is a `K=24, M=3` UrbanNav *medium urban* window with a prolonged
  stationary segment (heading unobservable) returning rank 3. This corroborates pre-flight **F7**
  (problem size is not a risk) and is a published instance of the observability failure mode that
  pre-flight **F9** found for 2-keyframe instances.

### 4.2 `delama` — `Gal(3) ⋉ gal(3)`, equivariant, transcendental

`VERIFIED-PUBLISHED` unless marked.

- **`Gal(3)`, Eq. (3):** `X = [[A, a, b],[0_{1×3}, 1, c],[0_{1×3}, 0, 1]] ∈ R^{5×5}`, `A ∈ SO(3)`,
  `a, b ∈ R^3`, `c ∈ R`.
- **`gal(3)`, Eq. (4):** `x = (ω, v, r, α) ∈ R^10`,
  `x^ = [[ω^, v, r],[0, 0, α],[0, 0, 0]]`. **Ordering `(ω, v, r, α)` confirmed.**
  *Source typo:* Eq. (4) is annotated "`ω^ ∈ SO(3)`"; it must be `so(3)`. Harmless but noted.
- **Inverse, Eq. (5):** `X^{-1} = [[A^T, −A^T a, −A^T(b − c a)],[0,1,−c],[0,0,1]]` — matches pre-flight
  **F4** exactly.
- **Adjoint, Eq. (6):** block form with `a^A`, `(b − ca)^A`, `−cA`, and the `a` entry in the
  time column; the time row is `[0 0 0 1]`, i.e. **the time coordinate is `Ad`-invariant** — matches
  pre-flight **F8**.
- **Symmetry group.** `G ≜ Gal(3) ⋉ gal(3)` (Sec. IV-C), the left-trivialised tangent group.
  Product **Eq. (1):** `XY = (AB, a + Ad_A[b])`; inverse **Eq. (2):** `X^{-1} = (A^{-1}, −Ad_{A^{-1}}[a])`;
  identity `(I, 0)`. Dimension **20** — confirmed twice over: `u_k ∈ L ⊆ R^20` and
  `ε_k ∈ R^20` (Eq. 35 linearises about `ε_k = 0 ∈ R^20`); `Â, B̂` are `20×20` in two `10×10` block
  rows (Eqs. 36, 37).
- **The state action is a RIGHT action.** Eq. (24):
  `φ(X_k, ξ_k) ≜ (Υ_k C_k, Ad^∨_{C_k^{-1}}(b_k − γ_k^∨))`, followed **immediately** by:
  *"Then, `φ` is a transitive and free **right** group action of `G` on `M`."* The input action `ψ`,
  Eq. (26), is likewise stated to be a right action. **Confirmed, with the exact location.**
- **Bias, 10-dimensional (Sec. IV-C):** `b_k = (b_{ωk}, b_{ak}, b_{νk}, b_{ρk}) ∈ R^10`:

  | slice | content | physical? |
  | --- | --- | --- |
  | `[0:3]` | `b_ω` gyroscope bias | yes |
  | `[3:6]` | `b_a` accelerometer bias | yes |
  | `[6:9]` | `b_ν` **virtual velocity bias** | **no** |
  | `[9]` | `b_ρ` **virtual time bias** (scalar) | **no** |

  The matching virtual *inputs* are `ν_k = 0_{3×1}` and `ρ_k = 1`.
- **The four virtual channels carry zero process noise.** Algorithm 1, `Define:` line:
  `Q_d = diag(σ_dω², σ_da², **0_{4×1}**, σ_dτω², σ_dτa², **0_{4×1}**)`. The two `0_{4×1}` blocks are
  exactly `(b_ν, b_ρ)` and their `τ` inputs. Equation number for `Q_d`: **`could not extract`** — it
  is an unnumbered line of Alg. 1. **Confirmed.**
- **The error is transcendental.** Equivariant error **Eq. (29)**
  `e_k ≜ φ(X̂_k^{-1}, ξ_k) = (Υ_k Ĉ_k^{-1}, Ad^∨_{Ĉ_k} b_k + γ̂_k^∨)`; local chart **Eq. (30)**
  `ϑ(e_k) ≜ log_G(φ_ξ̊^{-1}(e_k))`; the tangent-group logarithm **Eq. (31)**
  `log_G(X_k) = (log(C_k)^∨, J_L(log(C_k)^∨)^{-1} γ_k^∨)`; and the expanded error **Eq. (34)**
  ```
  ε_k = ( log(Υ_k Υ̂_k^{-1})^∨ ,  −J_L(log(Υ_k Υ̂_k^{-1})^∨)^{-1} Ad^∨_{Υ_k} (b_k − b̂_k) )
  ```
  `log` is Eq. (9) (contains `Γ_1(·)^{-1}` and `log(A)`); `J_L` is Eq. (10) (contains `Γ_1, Γ_2, Q_1,
  Q_2, U_1`, all trigonometric via `κ_1, κ_2, κ_3`). **Both `log_G` and `J_L^{-1}` are essential —
  confirmed.** This is the precise reason `delama`'s error cannot be a QCQP residual.
- **Navigation error is right-invariant, bias update is left-applied.** Eq. (34) uses
  `Υ_k Υ̂_k^{-1}`, which is invariant under right translation. Eq. (39):
  `Υ̂⁺_{k+1} ≈ exp((J_Υ,k+1 Δb̂)^∧) Υ̂_{k+1}` — **left**. `gtsam` states this contrast independently
  (§1 and §5 prose of `galileanimufactor`), and both statements agree with what Eq. (34)/(39) say.
- **Endpoint relation, Eqs. (17)–(19):** `T_j = Γ_ij T_i Υ_ij`, with
  `Γ_ij = [[I, gΔt_ij, −½gΔt_ij²],[0, 1, −Δt_ij],[0, 0, 1]]` (Eq. 18, exact: a product of
  `exp((G − N)δt)`), and `Υ_ij = [[ΔR_ij, Δv_ij, Δp_ij],[0,1,Δt_ij],[0,0,1]]` (Eq. 19). The `−Δt_ij`
  entry in `Γ_ij` is what returns `T_j` to zero elapsed time. See §6.
- **Exact mean, linearised covariance.** Eq. (22) `Υ_{k+1} = Υ_k exp((w_k^ − b_k^)δt)` is the *exact*
  held-input update; Eqs. (35)–(37) are first order about `ε_k = 0`; Eq. (39) is first order in `Δb̂`.

### 4.3 `barfoot` — Cayley, redundant constraints, tightness diagnostics

`VERIFIED-PUBLISHED` unless marked.

- **Cayley, same half-angle convention as `rai`.** `cay(A) = (I − ½A)^{-1}(I + ½A)`;
  `C(φ⃗) = cay(φ⃗^)` with `φ⃗ = 2 tan(φ/2) a`, `φ` the rotation angle and `a` the unit axis (Eq. 14a,
  angle/axis stated in the text at Eq. 14b), and inverse `cay^{-1}(C)^∨ = (2(C−I)(C+I)^{-1})^∨` Eq. (14b). Extended to `SE(3)` by
  `T(ξ) = cay(ξ^)` (Eq. 15) — i.e. **`barfoot` Cayley-parameterises rotation *and* translation**,
  where `rai` Cayley-parameterises only the rotation. Also noted (Selig 2007):
  `T(ξ) = cay(ξ^)` and `𝒯(ξ) = cay(ξ^⋏)` are **not** related by `Ad`, i.e.
  `𝒯(ξ) ≠ Ad(T(ξ))` — a trap if Cayley is ever pushed through an adjoint.
- **Relaxation pipeline.** Eq. (17) nonconvex → Eq. (18) homogeneous QCQP with `x^T A_0 x = 1` →
  Eq. (19) rank-1 SDP → Eq. (20) Shor's relaxation. Tightness ≡ `rank(X) = 1`.
- **Rank-1 criterion: log-SVR ≥ 5.** *"we define the logarithmic SVR (SVR) as the base-10 logarithm
  of the ratio of the largest to second-largest singular values of `X`; we consider a log SVR of at
  least `5` to represent rank 1."* Equation number: **`could not extract`** — this is prose in the
  rotation-averaging experiment section (the paragraph discussing Fig. 4), restated verbatim in the
  captions of Figs. 2, 4, 5, 9, 12. **Confirmed.**
- **Tightness degrades with measurement sparsity.** *"the log SVR values are quite a bit lower in
  Figure 12 as compared to Figure 9. This seems to be mainly due to the fact that we are now using a
  sparser set of measurements. Our continuous-time experiments had pose measurements only at the
  start, middle, and end, whereas the discrete-time case had pose measurements at every timestep. It
  is known that a sparser measurement graph can impact SDP tightness (Holmes and Barfoot, 2023)."*
  Equation number: **`could not extract`** (§5 discussion prose). **Confirmed.**
- **Redundant constraints break LICQ.** §6: *"in most of the problems of this paper, we require
  redundant constraints to tighten our SDP. This means that we do not meet the technical condition of
  LICQ ... It turns out that this makes it more challenging to calculate an optimality certificate."*
  This applies verbatim to this project (which inherits `rai`'s A1/B1/B2/C1) and is M4's problem, not
  M0's. Equation number: **`could not extract`**.
- **Tangent ordering `ξ = [ρ; φ]`** (Eq. 10), translation first — see **O5**.

### 4.4 `gtsam` — `Gal(3) × R^6`, left-invariant, right-corrected

`VERIFIED-PUBLISHED` (doc) **and runtime-checked against `gtsam 4.3.0`** unless marked.

- **State, Eq. (1):** `M = Gal(3) × R^6`, `ξ_k = (Υ_k, β_k)`, `β_k = [b_a; b_ω]`. The doc is explicit:
  *"The second component is exactly `gtsam::imuBias::ConstantBias`; its vector representation stores
  accelerometer bias before gyroscope bias. There are no additional bias coordinates."*
- **Matrix, Eq. (2):** `X = (R, v, p, t) ↔ [[R, v, p],[0,1,t],[0,0,1]]`.
- **Product, Eq. (3):**
  `(R₁,v₁,p₁,t₁)(R₂,v₂,p₂,t₂) = (R₁R₂, v₁+R₁v₂, p₁+R₁p₂+t₂v₁, t₁+t₂)`.
  **Inverse, Eq. (4):** `(R^T, −R^Tv, −R^T(p−tv), −t)`. Both match pre-flight **F4** and `delama`
  Eq. (5). Runtime-verified against `gtsam.Gal3.compose` / `.inverse()` (§6).
- **Tangent, Eq. (5):** `x = (ω, ν, ρ, α) ∈ R^10`. **Exp, Eq. (6):**
  `Exp(x) = (Exp(ω), J_L(ω)ν, J_L(ω)ρ + α Γ_2(ω)ν, α)` — the `αΓ_2(ω)ν` term is the `½aΔt²`.
  Identical in content to `delama` Eq. (8).
- **Adjoint, Eq. (9):** matches pre-flight **F8** block-for-block (`R`, `v^R`, `(p−tv)^R`, `−tR`,
  `v`, time row `[0 0 0 1]`).
- **Retraction convention.** §1 table: `X ⊕ δ = X Exp(δ)` (right-applied increment),
  `Local(X, Y) = Log(X^{-1} Y)`, error `X = X̂ Exp(ε)`. *"left-invariant error and right perturbation
  describe the same convention."*
- **Endpoint relation, Eq. (17):**
  ```
  X_j = W_ij · φ_{Δt_ij}(X_i) · U_ij(β_i)
  ```
  with (Eq. 15) `W_ij = NavState(I, ½gΔt_ij², gΔt_ij)` and `φ_Δt(R,p,v) = (R, p + vΔt, v)`, and
  (Eq. 16) `U_ij(β_i) = NavState(ΔR_ij, Δp_ij, Δv_ij)`. **This is `W φ(X) U`, not `H X D`** — the
  coasting map `φ_Δt` is a separate left factor. `Gal(3)` is internal to preintegration; the
  optimised keyframe states remain `SE_2(3)`. **Confirmed; verified numerically in §6 to `9.4e-17`,
  and the `H X D`-shaped `W · X_i · U` form shown wrong by `0.229`.**
- **Left-invariant error, Eqs. (21)–(22):** `e_k = ξ̂_k^{-1} ξ_k = (Υ̂_k^{-1}Υ_k, β_k − β̂_k)`,
  `ε_k = (e_k, δβ_k) ∈ R^16`, `e_k = Log(Υ̂_k^{-1}Υ_k)`.
- **Transition, Eq. (29):** `A_LI = [[Ad_{V^{-1}}, −C],[0_{6×10}, I_6]] ∈ R^{16×16}`,
  `B_LI = [−C; 0_{6×6}] ∈ R^{16×6}`, with `C = J_R(x) h E ∈ R^{10×6}` (Eq. 24) and `E` the `(a,ω)` →
  `(ω,ν,ρ,α)` lift of Eq. (20). *"no virtual bias columns are introduced."*
- **Bias correction on the right, Eqs. (34)–(36):** `Υ_k(β̂+δβ) ≃ Υ̂_k(β̂) Exp(J_k δβ)`,
  `J_{k+1} = Ad_{V_k^{-1}} J_k − C_k`, `J_i = 0`. *"The six columns of `J_k` are ordered `(b_a, b_ω)`,
  exactly like `ConstantBias::vector()`. There is no intermediate `10×10` virtual-bias Jacobian and no
  final column permutation."* Restated as Eq. (3) of the NEES page.
- **The `R^10 → R^9` projection, Eq. (38):** `n = P_N e` with
  `P_N = [[I₃,0,0,0],[0,0,I₃,0],[0,I₃,0,0]] ∈ R^{9×10}` — a selection **and** a swap of the velocity
  and position blocks. Applied to *"local errors, covariances, and Jacobians"* (Eqs. 39, 40). **O2.**
- **Residual, Eq. (42):** `r_ij = Local_NavState(X_j, predictedState_j) = X_j.logmap(predictedState_j)`
  — argument order is deliberate.
- **Combined variant.** `PreintegratedCombinedMeasurementsG` / `GalileanCombinedImuFactor` propagate a
  public `15×15` covariance over `(R, p, v, b_a, b_ω)`; *"The `Gal(3)` clock coordinate is
  deterministic and is projected out exactly."*
- **Rotating frame, Eq. (47):** `X_j = P_Ω(W^Ω_ij φ_{Δt}(L_Ω(X_i)) U_ij(β_i))` with the
  transported-velocity maps Eq. (44) and kernels Eq. (45). Exact for constant navigation-frame rate.
  Out of scope for Stage B, recorded so it is not reinvented.
- **Not certifiable, and does not claim to be.** The residual Eq. (42) is a `logmap`.
- `could not extract` — the doc pages give no C++ symbol for the `Gal3`→`NavState` block map. The
  project's `CLAUDE.md` names it `kGal3BlockForNavState = {0, 6, 3}`; the installed package is a
  compiled wheel, so **the symbol name could not be confirmed from source here**. Its *behaviour*
  is confirmed: it is exactly `P_N` of Eq. (38), and `{0, 6, 3}` is consistent with
  `Gal3` rows `[0:3, 6:9, 3:6]`.

---

## 5. Two different tightness verdicts — do not mix them

`VERIFIED-PUBLISHED`. The two certifiable sources use **different numerical rank tests**, and a
single downstream "is it tight?" helper that quotes one while citing the other is a silent error.

| | `rai` §VII-A | `barfoot` §3 prose |
| --- | --- | --- |
| rank test | `#{λ_i(Ẑ) > 10⁻⁶ λ_max(Ẑ)} == 1` | `log₁₀(σ₁/σ₂) ≥ 5` |
| gap | `η = \|x̂^T Q x̂ − p⋆\| / \|x̂^T Q x̂\|`, relative, dual optimum `p⋆` | not the primary diagnostic |
| reported threshold for "tight" | rank 1 **and** negligible `η` | log-SVR ≥ 5 |

They are not equivalent. `rai` cuts at `10⁻⁶` relative, `barfoot` at `10⁻⁵` relative, so the two
verdicts differ on every spectrum whose second-to-first ratio lands in `(10⁻⁶, 10⁻⁵)` — there,
**`barfoot` says rank 1 and `rai` says rank 2.** `barfoot` is also a ratio of the top two *singular*
values, `rai` a count of *eigenvalues* above a floor; for the symmetric PSD matrices at hand these
coincide, but the thresholds do not. `ASSUMPTION` for this project: **report both**, and treat a
window as certified only when `rank(Ẑ) = 1` under `rai`'s eigenvalue rule **and** `log-SVR ≥ 5` —
i.e. take the stricter of the two.

---

## 6. Cross-source corroborations, verified numerically

All checks run under `.venv` (`gtsam 4.3.0`, numpy, scipy). Script kept out of the repo; the claims
in `claims/M0.yaml` carry the recipes so the implementer re-derives rather than copies.

| # | check | result | tag |
| --- | --- | --- | --- |
| **X1** | `gtsam` Eq. (17): `‖Local(X_j, W_ij·φ_Δt(X_i)·U_ij)‖` with `X_j = pim.predict(X_i, β)`, 50 IMU samples, `Δt = 0.5 s` | `9.4e-17` — **exact** | `DERIVED` |
| **X2** | the `H X D`-shaped variant `W_ij · X_i · U_ij` (coasting map dropped) | `0.229` in `NavState` local coordinates — **wrong**, as the doc's `Wφ(X)U` wording implies | `DERIVED` |
| **X3** | `delama` Eq. (18) `Γ_ij` vs pre-flight **F2**'s corrected `T(−Δt)·H'`, `H' = [[I, gΔt, −½gΔt²],[0,1,0],[0,0,1]]`, `T(−Δt) = [[I,0,0],[0,1,−Δt],[0,0,1]]` | `‖Γ_ij − T(−Δt)H'‖ = 0.0` — **bit-identical** | `DERIVED` |
| **X4** | the same with `T(−Δt)` on the **right**: `‖Γ_ij − H'·T(−Δt)‖` | `2.45` — **wrong side matters**, as F2 states | `DERIVED` |
| **X5** | `gtsam.Gal3.compose` against `gtsam` Eq. (3) and `delama` Eq. (5) inverse, random elements | `≤ 1.1e-16` on every block; `‖X X^{-1} − I₅‖ = 2.8e-16` | `DERIVED` |
| **X6** | `rai` Eq. (22b) (lifted form, in `Y, Y', Γ_ij, R̆_ij`) against `rai` Eq. (19) (raw form, in `ξ_ij`), 10⁴ random `(R_i, R_j, ΔR̄, J^R, b_g, b̄_g)` | max abs difference `3.6e-15` — **algebraically identical**, the lifting is exact, not an approximation | `DERIVED` |

**X3/X4 are the important ones.** Pre-flight F2 is the project's seeded defect: the original contract
asserted `X_j = H_ij X_i D_ij` with a positive gravity-position block and no left time factor, and F2
corrected it by direct numerical integration. That correction is now shown to **reproduce a published
equation** — `delama` Eq. (18) — exactly, including the `−½gΔt²` sign and the `−Δt` time entry. The
corrected relation is therefore not merely numerically right; it is the convention of the literature.

> `OPEN-HYPOTHESIS` — the `Gal(3)`-left form (`delama` Eq. 18) and the `SE_2(3)` `Wφ(X)U` form
> (`gtsam` Eq. 17) are expected to be two factorisations of the same map, differing only by where the
> `t₂v₁` coupling is accounted. X1 and X3 each verify one of them against its own source, but **no
> check here composes the two**. M1 should close this.

---

## 7. What this project takes from where

`VERIFIED-PUBLISHED` for the attributions, `ASSUMPTION` for the choices.

| ingredient | taken from | note |
| --- | --- | --- |
| state `Gal(3) × R^6`, direct product | `gtsam` Eq. (1) | prior work |
| held-input Galilean composition | `delama` Eq. (22), via `gtsam` Eq. (12) | prior work |
| physical 6-D bias, no virtual channels | `gtsam` Eq. (1) | prior work; **deliberately not** `delama`'s 10-D |
| endpoint relation | `delama` Eq. (17)/(18) ≡ pre-flight F2 corrected form | `ASSUMPTION`: Stage B uses the `Gal(3)`-left form `X_j = T(−Δt)·H'·X_i·D_ij` |
| Cayley residual, denominator clearing | `rai` Eqs. (3), (17), (18), (19); `barfoot` Eq. (14) | prior work |
| lifting + homogenisation `s² = 1` | `rai` Eqs. (28), (31) | prior work |
| redundant constraints A1/B1/B2/C1 | `rai` Eqs. (32)–(35) | prior work; must be **re-derived** for `Gal(3)` liftings, not copied |
| rank-1 / log-SVR diagnostics | `rai` §VII-A, `barfoot` §3 | prior work; report both (§5) |
| bias ordering `[b_g; b_a]` | — | `ASSUMPTION` — agrees with `rai`/`delama`, **disagrees with `gtsam`**; one boundary permutation (**O3**) |
| **certification of the Galilean factor** | — | the contribution, and nothing more |

---

## 8. Machine-readable conventions table (contract for the implementer)

The M0 claims are assertions about a conventions table that the implementer builds — a plain Python
mapping, suggested location `src/conventions.py`, exposing `CONVENTIONS: dict[str, dict]`. The
deriver does not write it. Required shape:

```python
CONVENTIONS = {
  "<source_key>": {                 # one of: "rai", "delama", "barfoot", "gtsam"
    "citation": str,                # arXiv id + version, or doc URL + package version
    "group": str,                   # e.g. "SO(3)", "Gal(3) x R^6", "Gal(3) |x gal(3)"
    "state_dim": int | None,
    "element_order": tuple[str,...] | None,   # matrix column semantics, e.g. ("R","v","p","t")
    "ctor_order":   tuple[str,...] | None,    # API tuple order, e.g. ("R","p","v","t")
    "tangent_order": tuple[str,...] | None,   # e.g. ("omega","nu","rho","alpha")
    "error_handedness": "left-invariant" | "right-invariant" | "none",
    "action_handedness": "left" | "right" | "none",
    "bias_order": tuple[str,...] | None,      # e.g. ("b_g","b_a") or ("b_a","b_g")
    "bias_dim": int | None,
    "bias_virtual_slices": tuple[slice,...],  # () when there are none
    "bias_zero_process_noise": tuple[str,...],
    "bias_correction_side": "left" | "right" | "none",
    "exp_log_essential": bool,
    "exact": tuple[str,...],        # short labels of what is exact
    "linearized": tuple[str,...],   # short labels of what is first-order
    "eq": dict[str, str],           # fact label -> equation number AS A STRING,
                                    # or the literal "could not extract"
  },
  ...
}
```

Rules the tests enforce:

1. Every value of an `eq` entry is either a digit-bearing equation label (e.g. `"19"`, `"22b"`,
   `"37a"`) or **exactly** the string `"could not extract"`. No other sentinel, no `None`, no `""`.
2. `eq` labels are **per-source**; `rai` Eq. (19) and `delama` Eq. (19) are different equations and
   must never be compared across rows.
3. An incompatibility register `INCOMPATIBILITIES: dict[str, dict]` keyed `"O1".."O7"` records, for
   each row of §3, the two sides and the affected coordinate blocks.

---

## 9. Could not extract

Recorded so that no downstream agent invents a number to fill the hole.

| item | source | status |
| --- | --- | --- |
| equation number for `rai`'s bias ordering `b = [b_g^T b_a^T]^T` | `rai` | `could not extract` — unnumbered display between Eqs. (10c) and (11); block form appears in the Eq. (36) discussion |
| per-edge scalar count for subset **B2** | `rai`, Table I / Eq. (34) | `could not extract` — A1 (6/node), B1 (3/lifting), C1 (2/edge) are stated; B2 is not |
| equation number for `Q_d` (the zero process noise on the 4 virtual channels) | `delama`, Alg. 1 | `could not extract` — unnumbered `Define:` line of Algorithm 1 |
| equation number for the continuous-time biased INS equations | `delama`, §IV-A | `could not extract` — the HTML render cross-references them as "( IV-A )" with no visible number |
| equation number for the log-SVR ≥ 5 rank-1 threshold | `barfoot` | `could not extract` — §3 prose and the captions of Figs. 2, 4, 5, 9, 12 |
| equation number for the measurement-sparsity/tightness statement | `barfoot` | `could not extract` — §5 discussion prose (Fig. 12 vs Fig. 9) |
| equation number for the LICQ failure under redundant constraints | `barfoot` | `could not extract` — §6 prose |
| C++ symbol for the `Gal3`→`NavState` block map | `gtsam` | `could not extract` from the doc pages or the compiled wheel. Behaviour = Eq. (38) `P_N`, confirmed at runtime |
| RA-L volume / issue / pages for `delama` | `delama` | `could not extract` from the arXiv HTML (DOI `10.1109/LRA.2024.3511424` only) |

---

## 10. Claim index

See [`claims/M0.yaml`](claims/M0.yaml) — **15 claims**, `M0-C01` … `M0-C15`.

| id | subject | kind |
| --- | --- | --- |
| M0-C01 | `rai` is `SO(3)`-only and Galilean-free | `VERIFIED-PUBLISHED` |
| M0-C02 | `rai` rotation residual Eq. (19) | `VERIFIED-PUBLISHED` |
| M0-C03 | `rai` Eq. (22b) ≡ Eq. (19) | `DERIVED` |
| M0-C04 | `rai` lifted vector Eq. (28), homogeniser Eq. (31) | `VERIFIED-PUBLISHED` |
| M0-C05 | `rai` A1/B1/B2/C1 → Eqs. (32)–(35) and the ablation outcome | `VERIFIED-PUBLISHED` |
| M0-C06 | `rai` isotropy Eq. (23) is structurally forced | `VERIFIED-PUBLISHED` |
| M0-C07 | `delama` `Gal(3)` Eqs. (3)–(6) and tangent order | `VERIFIED-PUBLISHED` |
| M0-C08 | `delama` symmetry group dim 20, Eq. (1), **right** action | `VERIFIED-PUBLISHED` |
| M0-C09 | `delama` 10-D bias and the 4 zero-noise virtual channels | `VERIFIED-PUBLISHED` |
| M0-C10 | `exp`/`log`-essential column across all four rows | `VERIFIED-PUBLISHED` |
| M0-C11 | error handedness and bias-correction side register | `VERIFIED-PUBLISHED` |
| M0-C12 | `barfoot` log-SVR ≥ 5 and the sparsity caveat | `VERIFIED-PUBLISHED` |
| M0-C13 | the seven ordering incompatibilities, runtime-checked | `VERIFIED-PUBLISHED` |
| M0-C14 | `gtsam` endpoint relation Eq. (17) `W φ(X) U` | `VERIFIED-PUBLISHED` |
| M0-C15 | `delama` Eq. (18) `Γ_ij` = pre-flight F2's `T(−Δt)H'` | `DERIVED` |
