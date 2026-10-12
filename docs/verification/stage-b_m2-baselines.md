# Stage-B verification: M2 baselines (`stage-b/m2-baselines`)

Independent adversarial pass. The goal was to **falsify**, not to confirm.
`docs/02_baseline_reproduction.md` was deliberately **not read** (it is the deriver's
narrative; checking code against it is circular). Evidence comes from four independent
authorities only:

1. **The Rai paper itself** — arXiv:2609.38048v1, fetched and read verbatim, including
   Eq. (1)–(4), (9a)–(9c), (10)–(13), (15)–(23).
2. **A from-scratch re-implementation** (`scratchpad/verify/indep.py`) with **no import
   from `src/`**: `Exp` via `scipy.linalg.expm` of the skew (not Rodrigues), `Log` via
   `scipy.linalg.logm`, the Eq. (9) Euler recursion retyped from the paper's text, the
   Cayley map, and the chaining relation. Fixture constants were **retyped from the
   M2-C01 claim text**, not copied from `src/trajectory.py`.
3. **Two independent exact references** for the increment: `mpmath` at 50 decimal digits,
   and direct adaptive quadrature of `∫Exp(ws)a ds` and `∫(T-s)Exp(ws)a ds`.
4. **The installed gtsam 4.3.0**, queried directly.

Plus an **18-mutant mutation-testing sweep** of `src/` against the full suite, to attack
the tests rather than the source.

Suite state at HEAD: **22 passed, 3 failed** (`test_c22`, `test_c23`, `test_c24`).

---

## 1. Verdict table

| Claim | Verdict | Concrete attack attempted |
|---|---|---|
| M2-C01 | **CONFIRMED** | Retyped all 14 fixture constants from the claim text and diffed against `src`: no mismatches. Verified `BIAS_TRUE` is `[b_g; b_a]` and the streams are exactly `w_e + b_g` / `f_e + b_a`. |
| M2-C02 | **CONFIRMED** | Three independent routes: mpmath 50-digit closed form, scipy `quad` of the defining integrals, and `src`. Pairwise agreement ≤ 2.0e-15. Chained the mpmath increments myself: the 15-digit Sec. 0.4 literals reproduce to ≤ 2.7e-15. *Caveat:* `REF_R_KF1` is defined **circularly** in `src` as `R_0 @ REF_DR_01`; it is not an independent literal (it is also unused by any test). |
| M2-C03 | **CONFIRMED** | My own chaining formula vs `PreintegratedImuMeasurementsG.predict()` from a nonzero state: 9.4e-15 / 1.1e-16 / 8.9e-16. |
| M2-C04 | **CONFIRMED** | My own Euler recursion vs the 50-digit exact increment, n = 100…3200: dv ratios 0.5000 ×5, dp ratios 0.5009→0.5001. Inside [0.45, 0.55]. dR telescopes to ≤1e-13. |
| M2-C05 | **CONFIRMED** | Error ratio is **exactly 100.0** on both edges and is **invariant** across FD steps ε = 1e-4 … 1e-8 (98.8–100.3), so the quoted quadratic law is not FD noise. Mutation M03 (left composition) is killed by this test. |
| M2-C06 | **CONFIRMED** | Swept 12 combinations of (σ_g ∈ {1.7e-4, 1e-2, 1e-1}, σ_a ∈ {1e-2, 1.0}, n ∈ {20, 200}): trace **never** decreased, max&#124;S−Sᵀ&#124; ≤ 2.5e-16, min eig ≥ −8.1e-18. The GTSAM cross-check has teeth: replacing the `P` remap by the identity gives relative 9.2e-1 → fails. |
| M2-C07 | **CONFIRMED** | Verified *within* the stated θ<3 domain (worst 2.1e-15) **and** verified the domain restriction is necessary: at θ = 3.14159 the error is 8.4e-11 > 1e-13. Separately re-derived Eq. (4) numerically — see §2. |
| M2-C08 | **CONFIRMED** | Exact to ≤1e-12 for ‖ξ‖ ≤ 5. Pushed to ‖ξ‖ = 1e7: the *absolute* deviation grows to 1.0e3 while the *relative* deviation stays ≤ 4.2e-11 — i.e. the identity is exact in exact arithmetic and the 1e-12 **absolute** tolerance is domain-bound, as the claim's domain says. |
| M2-C09 | **CONFIRMED** | On-manifold ≤1e-15. Hunted the weakest off-manifold direction over 3000 random perturbations: min 5.87e-4, comfortably ≥ 1e-4 (analytic floor ≈ ε/√3 ≈ 5.8e-4). |
| **M2-C10** | **REFUTED** | See §3.1. At the *actual* ground-truth states the residual is **2.76e-3**, not ≤1e-9. |
| M2-C11 | **CONFIRMED** (identity) | Verified the identity by hand-expansion and over randomised instantiations. **But** the test's advertised discriminating power is false — see §4.3. |
| M2-C12 | **CONFIRMED** (vacuously) | Checked Eq. (23)'s form against the paper's own line (`τ⁻¹ = ⅓ tr Σ`). The "identity" under test is orthogonality of SO(3) and exercises essentially no project code — see §4.7. |
| M2-C13 | **CONFIRMED** (part b only) | Part (b): gap ≈ ½‖ξ‖² = 5e-5 ≥ 1e-6, robust. Part (a) is **structurally unfalsifiable** (§4.5), and the claim's substantive content — the frozen *affine* map ξ = J_R(b_g − b̄_g) — is **untested** (mutant M15 survives, §4.6). |
| M2-C14 | **CONFIRMED** | Measured directly on installed gtsam 4.3.0: `matrix()[:,3] == velocity`, `matrix()[:,4] == position`, `M[3,4] == t`, constructor is `(R, pos, vel, t)`. |
| M2-C15 | **CONFIRMED** | `Hat(e_k)` block occupancy and `Expmap(1e-6 e_k)` measured for k = 0…9. Mutating the remap to (0,3,6) is killed. |
| **M2-C16** | **CONFIRMED as stated; recorded measurement REFUTED** | See §3.4. The right number is **0.107628**, not 0.763. |
| M2-C17 | **CONFIRMED** | Measured on installed gtsam. The adapter mutation (no swap) is killed by four tests. |
| M2-C18 | **CONFIRMED** | Reproduced with **my own 5×5 matrix products**, never calling `Gal3.compose`: W-form error 2.2e-16 at t_i ∈ {0, 3.7}; F2-form 18.14850 at t_i = 3.7 vs \|g\|·Δt·t_i = 18.14850 exactly; swapped H′·T(−Δt) form 15.70. Also attacked the order: `X_i·W·U` is wrong by 8.9e-1. |
| M2-C19 | **CONFIRMED** | Part (b) genuinely has teeth — see §4.1. |
| M2-C20 | **CONFIRMED** | See §3.5. True supremum measured at 1.40e-2 / 1.62e-2; the test's analytic direction **is** the sup, not a relaxation. FD regime analysed. |
| M2-C21 | **CONFIRMED** | Recursion vs FD of full re-integration after 40 steps: 2.71e-10 absolute against \|J\|_max = 0.1 ⇒ **2.7e-9 relative** ≤ 1e-6. Has real teeth: the `Ad_V` mutation produces 9.99e-2. |
| **M2-C22** | **REFUTED** | §2. 1e-10 for R is unreachable by a factor of ~300 and is forbidden by the O(dt²) structure of the raw Cayley retraction. |
| **M2-C23** | **REFUTED** (rotation parenthetical only) | §2. The Tier-2 magnitude and convergence-order content is independently **confirmed**. |
| **M2-C24** | **REFUTED** (Baseline-2 rotation half only) | §2. The Baseline-3 half and the Baseline-1 half are independently **confirmed**. |
| **M2-C25** | **REFUTED** (nullspace sub-assertion) | §3.2 / §3.3. The 6 remaining directions are **not** the bias block; two of them sit at ~72° from it. Two of the three rank numbers are forced by row count. |
| **M2-S02** | **REFUTED — PLANTED** | §3.6. Off by **9 orders of magnitude** (9.99e-2 vs 1e-10). |

Counts: **16 CONFIRMED**, **1 CONFIRMED-with-defective-measurement (C16)**, **6 REFUTED
(C10, C22, C23, C24, C25, S02)**, **0 UNTESTABLE**. Several CONFIRMED verdicts carry a
"the test cannot fail" rider; those are in §4.

---

## 2. Ruling on the C22 / C23 / C24 dispute

**I rule for the implementer on the outcome, but on different and stronger grounds, and
I reject the implementer's premise.** The claims are defective. The code is correct. But
the ambiguity the claims appeal to **does not exist** — the paper settles it.

### 2.1 The paper settles the reading, and it settles it on `raw`

The implementer (and M2-C22's `domain` field) assert "the paper does not state which
reading is intended". That is false. Three pieces of evidence from the paper, in
increasing order of force:

**(a) Sec. IV-A, verbatim.** "the pre-integration derivation in Sec. III-A carries over
after replacing `Exp` and its right Jacobian with `Cay` and the Cayley right Jacobian in
(4)". Eq. (9a) is `ΔR̃_ij ≜ ∏ Exp((ω̃_k − b̄_g)Δt)`. The literal substitution instructed
by Sec. IV-A gives `∏ Cay((ω̃_k − b̄_g)Δt)` — the **raw** reading. No reparameterisation
step appears anywhere in the paper.

**(b) Sec. II-A, verbatim.** "The Cayley map satisfies Cay(φ) ∈ SO(3) for every φ ∈ ℝ³,
and **agrees with the exponential map to second order, Cay(φ) = Exp(φ) + O(‖φ‖³)**."
This sentence is *only non-vacuous under the raw reading*. Under the reparameterised
reading Cay(2tan(θ/2)a) ≡ Exp(θa) **identically** — I measured the gap at 1.1e-16, i.e.
round-off. A paper does not write "agrees to second order" about a pair of maps that are
bit-for-bit equal. I confirmed the O(‖φ‖³) order directly: halving ‖φ‖ divides
‖Cay(φ) − Exp(φ)‖ by 8 (ratio to ‖φ‖³ constant at 0.073 over ‖φ‖ = 0.1 → 0.0125).

**(c) Eq. (4) is decisive.** The paper gives `J_Cay(φ) = (I − ½φ^)/(1 + ¼‖φ‖²)` as *the*
Cayley right Jacobian and explicitly distinguishes it from `Exp`'s. I computed, by
Richardson-extrapolated central differences, which map Eq. (4) is actually the right
Jacobian of:

| ‖φ‖ | ‖J_Cay − J_num(raw `Cay`)‖ | ‖J_Cay − J_num(`Cay∘reparam`)‖ | ‖J_Cay − J_r(`Exp`)‖ |
|---|---|---|---|
| 0.05 | 4.7e-11 | 4.3e-4 | 4.3e-4 |
| 0.20 | 2.9e-11 | 6.5e-3 | 6.5e-3 |
| 0.40 | 2.5e-11 | 3.5e-2 | 3.5e-2 |
| 0.80 | 2.6e-11 | 8.7e-2 | 8.7e-2 |
| 1.50 | 3.7e-11 | 3.6e-1 | 3.6e-1 |

Eq. (4) is the right Jacobian of the **raw** map to FD precision (~3e-11, flat in ‖φ‖),
and is wrong by up to 0.36 for the reparameterised map — for which the correct right
Jacobian is `J_r(Exp)` (verified: 9.2e-12 / 4.2e-11). **If the paper meant the
reparameterised reading, its own Eq. (4) would be wrong.** It does not; it means raw.

**(d) Structural.** Under the reparameterised reading `Cay ≡ Exp`, Baseline 2 would be
numerically indistinguishable from Baseline 1 and would not be a separate baseline at
all. The "two admissible readings" of M2-C22 are not two readings of a retraction; one
of them is just `Exp` with extra arithmetic.

**Conclusion:** the ambiguity is **not real and not unresolved**. There is one admissible
reading. `DEFAULT_VARIANT = VARIANT_RAW` in `src/preintegration_cayley.py` is the correct
spec choice. M2-C22's `domain` field ("the paper does not state which reading is
intended") is itself wrong, and the statement's "for both admissible readings" is a false
dichotomy that should be deleted, not merely re-toleranced.

### 2.2 Is 1e-10 achievable for the raw reading? **No — provably not.**

`Cay(θa)` rotates by `2 arctan(θ/2)`, losing `θ³/12 + O(θ⁵)` per step. The fixture's body
rate is constant per edge, so all per-step rotations share an axis and commute, and the
deficits **add exactly**:

| | ‖ω‖ | θ = ‖ω‖·dt | deficit/step | ×200 | θ³/12 × 200 |
|---|---|---|---|---|---|
| edge 0 | 0.446990 | 1.117475e-3 | 1.162872e-10 | 2.325744e-8 | 2.325744e-8 |
| edge 1 | 0.596154 | 1.490386e-3 | 2.758765e-10 | 5.517531e-8 | 5.517533e-8 |

Total angle lost over the fixture's 400 steps: **7.843e-8 rad**.

My independent integrator (no `src` import) over the whole fixture reproduces the
implementer's number **to every printed digit**:

```
B1(Exp) vs B2-raw     : R 2.9965e-08   v 6.1672e-08   p 4.3763e-08
B1(Exp) vs B2-reparam : R 2.1205e-14   v 1.5365e-13   p 5.9286e-14
```

The deficit is `‖ω‖³·T·dt²/12`, i.e. **O(dt²)** — I measured the ratio at exactly 0.25
per halving over n = 100 → 1600 (1.199e-7, 2.997e-8, 7.491e-9, 1.873e-9, 4.682e-10).
Reaching 1e-10 requires dt ≈ 3.9e-5, i.e. **n ≈ 12800 steps per edge — a 64× refinement
of a fixture whose dt = 2.5e-3 is pinned by M2-C01**. At the stated fixture, 1e-10 for
the raw reading is not merely unmet; it is **mathematically unattainable**.

### 2.3 The claims contradict themselves

Confirmed by reading `docs/claims/M2.yaml` directly (not the implementer's paraphrase):

* M2-C22's `statement` demands 1e-10 in R "for both admissible readings"; its own
  `test_hint` records "raw variant 6.2e-8 (v) / 4.4e-8 (p) / **3.0e-8 (R)**" and then
  says "The 1e-6 tolerance gives ~16x margin over the worst case" — and 1e-6/6.2e-8 =
  16.1. The author sized **one** tolerance (1e-6) against **all six** recorded numbers,
  R included, then wrote 1e-10 for R into the statement.
* M2-C24's `test_hint` records "B1 and B2 rotation: **2.4e-14**". I reproduce 2.4e-14 for
  B2 **only under the reparameterised variant** (2.3759e-14) and 9.8e-15 for B1. The raw
  variant is 3.0e-8. So M2-C24 was measured on a different Baseline 2 than the one
  M2-C22's `domain` fixes as the spec. The implementer's observation here is correct.

### 2.4 Verdicts

* **M2-C22 — REFUTED.** The "1e-10 in R" sub-assertion is unattainable on the stated
  fixture, and "for both admissible readings" rests on an ambiguity the paper does not
  have. The v/p half (≤1e-6: measured 6.17e-8 / 4.38e-8) is **correct**. Corrected
  statement: *B1 and B2 agree at KF2 to 1e-6 in v, p and R under the raw (spec) reading
  of the Cayley substitution*, with 16× margin.
* **M2-C23 — REFUTED**, parenthetical only. "(rotations agree to 1e-10)" fails at 3.0e-8
  for B2-raw. The substantive content is **independently confirmed**: my own integrator
  vs GTSAM at n = 200 gives v 1.2867e-3, p 1.2031e-3 (≤5e-3, ~3.9× margin) and ratios
  0.5000 across n = 100…1600.
* **M2-C24 — REFUTED**, the "Baselines 1 and 2 reproduce the reference ROTATIONS to
  1e-10" half, for B2 only. B3 (1.7e-14 / 3.9e-14 / 1.7e-14) and B1 (9.8e-15) are
  **confirmed**.

**No code bug was found behind any of the three failures.** But see §4.2 — the suite's
response to this situation is dangerous.

---

## 3. REFUTED claims, in detail

### 3.1 M2-C10 — REFUTED

The statement says Eq. (21b)/(21c) "both vanish to 1e-9 **at the fixture ground-truth
states**". They do not:

```
baseline's own propagated states (what the test uses) : worst |r| = 1.6520e-13
EXACT ground-truth states (what the claim SAYS)       : worst |r| = 2.7603e-03
```

That is **six orders of magnitude** above the stated tolerance. The test silently
substitutes `B1.integrate_edge_states(..., cls=B2.CayleyPreintegration)` — the baseline's
own predictions — for the ground truth. At those states the residual is an **algebraic
tautology**: the states are produced by `chain()` as `v_j = v_i + g·Δt + R_i·dv̄`, and
Eq. (21b) computes `R_iᵀ(v_j − v_i − g·Δt) − dv̄ ≡ dv̄ − dv̄ ≡ 0`. Nothing is measured.
The claim's own `test_hint` concedes the substitution ("self-consistency, not exactness")
and thereby contradicts its own statement. The claim should either say *self-consistent*
states, or carry the 2.2e-3 Tier-2 tolerance.

### 3.2 M2-C25 — REFUTED (nullspace sub-assertion)

"adding only the KF0 pose+velocity priors gives 39/45 **with the 6 remaining directions
being exactly the bias block**." I extracted the actual 6-dimensional right null space of
the rank-39 Jacobian and measured where it lives:

```
null dir 0: energy in BIAS cols 0.481  | in POSE/VEL cols 0.519
null dir 1:                     0.727  |                  0.273
null dir 2:                     0.676  |                  0.324
null dir 3:                     0.334  |                  0.666
null dir 4:                     0.613  |                  0.387
null dir 5:                     0.631  |                  0.369
total pose/vel energy in the nullspace: 2.537 / 6   (42%)
principal cosines (nullspace, bias-coordinate subspace):
       [0.978227 0.978041 0.839290 0.815739 0.299838 0.299389]
```

All six would be 1.0 if the claim held. Two of them sit at **arccos(0.30) ≈ 72.5°** from
the bias block. The 6 unobservable directions are **mixed bias/pose/velocity gauge
directions**, not the bias block. The test cannot detect this: it only checks that adding
a 6-dim bias prior raises the rank to 45, which merely requires that no null direction be
*orthogonal* to the bias block — a strictly weaker statement.

The headline — rank 45/45 on the full graph with no bias prior — is **confirmed and
robust**: the full Jacobian is 45×45 with σ_max 2.450e5, σ_min 1.622e-2, cond 1.51e7, and
σ_min/tol = 6.6e6.

### 3.3 M2-C25 — two of the three rank numbers are forced

```
imu-only Jacobian shape (30, 45) rank 30
+KF0 priors     shape (39, 45) rank 39
full            shape (45, 45) rank 45
```

`rank == 30` and `rank == 39` are the **maximum possible** for a 30-row and a 39-row
matrix. Those two assertions can only fail on row degeneracy; they verify nothing about
observability structure. Only `rank == 45` on the square 45×45 system is a genuine
observability statement. Noted as a weakness rather than a separate refutation, since the
stated numbers are correct.

### 3.4 M2-C16 — statement confirmed, recorded measurement refuted

Under the claim's own configuration (KF0 and KF2, Gal(3) lift with **t = 0**, `{0,6,3}`
reorder, **infinity** norm) the measured gap is **0.107628**, which clears the stated
`>= 0.1` with 7.6 % margin. The `test_hint`'s "Measured 0.763" is **not reproducible**.
I enumerated 36 nearby readings — all three KF pairs × four clock conventions ×
{inf-norm, 2-norm, un-remapped, group-log}. The full spectrum runs 0.000 … 2.498 and
0.763 appears nowhere. The two nearest values are 0.743 (KF0→KF1, t_j = 1.0, 2-norm) and
0.738 (KF1→KF2, t = 0, **un**remapped) — neither is the claim's configuration.

**The right number is 0.1076.** The claim's bound survives, with an 8 % margin rather
than the 660 % the recorded number implies — i.e. the bound is near-tight and should be
treated as such.

A bonus result that *sharpens* the claim and that I did not expect: the Gal(3) logarithm
reordered by `{0,6,3}` equals `NavState.Logmap(X_i⁻¹ ∘ X_j)` **exactly** (0.0e+00, all
three KF pairs). NavState's group *is* Gal(3) restricted to t = 0 (i.e. SE₂(3)), so the
charts agree perfectly as *groups*. The 0.1076 gap is specifically against
`localCoordinates` — GTSAM's NavState **retraction**, which is not the group exponential.
C16's prose already says this; the measurement now pins exactly which object disagrees.

### 3.5 M2-C20 — the supremum question, resolved in the test's favour

The brief flagged that 30 random directions undersample a supremum. I measured the true
sup directly:

| | 30 dirs | 300 dirs | 20 000 dirs | test's analytic SVD direction | 12 ± coordinate axes |
|---|---|---|---|---|---|
| edge 0 | 1.351e-2 | 1.341e-2 | **1.403e-2** | **1.4041e-2** | 1.192e-2 |
| edge 1 | 1.205e-2 | 1.396e-2 | **1.619e-2** | **1.6216e-2** | 1.365e-2 |

The test's analytic direction **is** the supremum to three digits. It measures the bound;
it does not relax it. **The claim's `>= 1e-2` is correct as a supremum statement.**

However, the implementer's stated justification for adding it — "a 30-sample max lands
around 0.9e-2, below the stated bound for sampling reasons alone" — is **seed-specific,
not structural**. With the test's own seed (20250218) the bare-30 max is 9.40e-3 on edge 0
but **1.017e-2 on edge 1** (passes); four of six other seeds clear 1e-2 on edge 0; and
even the 12 unit coordinate axes clear it on both edges without any sampling at all. The
augmentation is legitimate but the comment overstates its necessity.

Separately, the `test_hint`'s "component-wise 2.8e-4..1.3e-1" is **uniformly ~10× my
measurements** (2.8e-5 … 1.4e-2) and is not reproducible under this configuration.

**FD regime.** The brief asked whether the quoted errors are FD noise. They are, and the
test says so. The right-composition residual at ‖db‖ = 5e-2 against an FD Jacobian:

| FD ε | 1e-8 | 1e-7 (the setting) | 1e-6 | 1e-5 |
|---|---|---|---|---|
| residual | 3.72e-9 | **3.31e-10** | 3.95e-11 | **2.37e-12** |

ε = 1e-7 is **roundoff-dominated**, not the stable regime; residual ∝ 1/ε confirms
round-off (δ/ε) rather than truncation (ε²) dominates. This does **not** invalidate the
claim: the quantity bounded is `model error + FD noise`, so `≤ 1e-8` remains a valid and
*conservative* upper bound on model error. Tightening ε to 1e-5 shows the model error is
actually **≤ 2.4e-12** — the right-composition model is exact to round-off, a stronger
result than the claim states. `J[0:3,0:3]` and `J[9,:]` come out as **exact** zeros
(0.00e+00) at every ε, not merely ≤1e-10.

### 3.6 M2-S02 — REFUTED. This is the planted claim.

> "replacing `Ad_{V_k^-1}` by `Ad_{V_k}` leaves the accumulated J unchanged to 1e-10 …
> because each per-step increment V_k is near-identity at dt = 2.5e-3."

| k | ‖J‖_max | ‖Ad⁻¹ form − FD‖ | ‖Ad form − FD‖ | ‖Ad⁻¹ − Ad‖ |
|---|---|---|---|---|
| 1 | 5.0e-3 | 6.93e-12 | 1.283e-4 | 1.283e-4 |
| 9 | 2.5e-2 | 5.30e-11 | 5.773e-3 | 5.773e-3 |
| 19 | 5.0e-2 | 8.94e-11 | 2.437e-2 | 2.437e-2 |
| **39** | **1.0e-1** | **2.71e-10** | **9.994e-2** | **9.994e-2** |

**Measured 9.994e-2 against a claimed ≤1e-10 — false by a factor of 10⁹.** The stated
*reason* is also wrong: `Ad_{V⁻¹} − Ad_V ≈ −2·ad_v` is O(dt) **per step**, and the
recursion multiplies the accumulated J by it at every step, so near-identity increments
do not make the error small — they make it accumulate linearly. The forward-adjoint
variant is not merely inaccurate; it diverges from the true Jacobian at the same order of
magnitude as the Jacobian itself.

Two corroborating facts: **M2-S02 does not appear in `docs/claims/M2.yaml` at all**
(0 occurrences), and the mutation implementing it (M09) is killed by `test_c21`.

**Procedural note, as requested.** I found M2-S02 **by computing** (and by noticing its
absence from `M2.yaml`) *before* looking at git history. The claim leaked into
`git log -p` is a **different** one: commit `66eb398` "Move M2 seeded defect out of
version control" removes a `seed:` block holding **M2-S01** —

> "In Baseline 1 the position bias Jacobians are an exact multiple of the velocity ones,
> J^{pa} = 0.5 Δt J^{va} and J^{pg} = 0.5 Δt J^{vg}, agreeing to 1e-6 on both fixture
> edges"

which is **also false** (measured: ‖J^{pa} − ½Δt J^{va}‖ = 3.7e-3 / 4.5e-3 and
‖J^{pg} − ½Δt J^{vg}‖ = 1.04e-1 / 9.8e-2 against a claimed 1e-6). M2-S01 was **not** in
the verifier's claim list, so it was never under test here; I report it only so the leak
is on record. **The planted claim I was given, M2-S02, was caught by computation, not by
history archaeology.**

---

## 4. Attacking the tests: every test that cannot fail

Method: 18 single-point mutations of `src/`, each run against the whole suite.
**Fourteen were killed. Four survived.** Survivors are defects the suite is blind to.

### 4.1 Order-sensitivity audit (as requested)

The fixture's constant-twist increments commute, so any order test fed constant inputs is
dead. I checked each order-sensitive construct by substituting the constant stream:

| Construct | Test | Has teeth? | Evidence |
|---|---|---|---|
| `U_{k+1} = U_k·V_k` vs `V_k·U_k` | C19(b) | **YES** | time-varying stream → left_err 6.17e-4. **Substituting the fixture's constant stream collapses left_err to 5.55e-16 — fully invisible.** The test correctly uses the time-varying stream; its docstring and the claim's `domain` both say why. |
| `Ad_{V⁻¹}·J + C` vs `Ad_V·J + C` | C21 | **YES** | mutation M09 → 9.99e-2, killed. |
| `W·X_i·U` vs `T(−Δt)·H′·…` vs `H′·T(−Δt)·…` | C18 | **YES** | 18.15 and 15.70; I additionally confirmed `X_i·W·U` is wrong by 8.9e-1. |
| `dp` updated before `dv` within a step | C10 | yes (incidentally) | mutation M01 killed — but only by the tautological C10 (§3.1). |
| `dR_ij(b̄)·Exp(J_R δb)` vs left composition | C05 | **YES** | mutation M03 killed. |
| Bias block order at the GTSAM boundary | C17/C19/C20/C25 | **YES** | mutations M07, M18 killed by four tests / one test. |
| **`dR_ik` vs `dR_{i,k+1}` in Eq. (9b)/(9c)** | — | **NO** | **§4.2 — survives the entire suite.** |
| **Cayley retraction vs `Exp`** | — | **NO** | **§4.2 — survives, and makes the suite green.** |
| **Y ↔ Y′ in Eq. (22b)** | — | **NO** | **§4.3.** |

### 4.2 The two most serious blind spots

**(i) `M12` — replacing Baseline 2's Cayley retraction by `Exp` SURVIVES, and turns the
suite from 3-failed/22-passed into 25 PASSED.**

```
UNMUTATED                  : 3 failed, 22 passed
M12 Cayley retract -> Exp  : 25 passed
```

The single property that makes Baseline 2 a distinct baseline — that its per-step
retraction is `Cay`, not `Exp` — is pinned by **no passing test**. The only three tests
that would notice (C22/C23/C24) are exactly the three that already fail, and deleting the
Cayley retraction makes them pass. A maintainer responding to a red suite by "making the
tests pass" would silently delete Baseline 2's reason to exist and be rewarded for it.
This is the most dangerous finding in M2. **Recommendation:** add a test that asserts
B2-raw's rotation differs from B1's by ≈ ‖ω‖³·T·dt²/12 (3.0e-8 ± 10 %) and that the gap
is O(dt²) — turning the current failure into a positive, falsifiable property of the
method.

**(ii) `M02` — folding the step's rotation in *before* using `dR_ik` for `dv`/`dp`
SURVIVES all 25 tests.**

```
M02 fold rotation first : 3 failed, 22 passed   (identical to unmutated)
```

This is a genuine misreading of Rai Eq. (9b)/(9c) (`ΔR̃_ik`, the *partial* increment, not
`ΔR̃_{i,k+1}`), and it is not a no-op: at n = 200, `|dv_correct − dv_wrong| = 4.342e-3`,
**twice the truncation error itself** (2.171e-3). Nothing catches it because the only
sharp assertions on B1's increment are **ratios** — and both variants are O(dt), so both
give 0.5000. C04 compares to the exact increment but asserts only the ratio; C10 is
self-consistent; C22 compares B1 to B2, both of which would be mutated.
**Recommendation:** C04 should assert the *magnitude* `|dv − dv_exact| ≈ 2.17e-3` with a
±10 % band at n = 200, not only its halving ratio.

### 4.3 `M04` — swapping Y and Y′ in Eq. (22b) is structurally undetectable

Eq. (22b) uses `−½(Y + Y′)`. A sum is symmetric, so exchanging Y and Y′ is **literally a
no-op**. Mutation M04 survives. `test_c11`'s docstring asserts the opposite: *"a sign
slip or a swapped Y/Y′ shows up immediately."* **The sign-slip half is true** (M05 is
killed by c11); **the swapped-Y/Y′ half is false and should be struck from the docstring.**

### 4.4 `M15` — Baseline 2's frozen affine bias map is untested

`CayleyFactor.xi` computes `J_R·(b_g − b̄_g)`. Dropping the `b̄_g` offset — so
`ξ = J_R·b_g` — survives the whole suite. At the fixture's `b̄_g` this is a **10×** error
in ξ (correct `[-9.9e-4, 5.2e-4, -2.4e-4]`, mutated `[-5.9e-3, 7.9e-3, -4.6e-3]`; the
dropped offset has norm 9.86e-3). M2-C13's substantive content — "the map
`δb_g → ξ_ij = J^R_ij δb_g` is linear with J^R frozen" — is therefore **not exercised by
any test**: C13(b) constructs ξ directly, C10 passes `db = None`, and C13(a) only reads
`J_R`'s bytes.

### 4.5 M2-C13 part (a) is unfalsifiable by construction

"assert the stored `J^R` is bitwise unchanged after a predict at a perturbed bias."
`grep` over `src/preintegration_cayley.py` finds **no code path anywhere that writes to
`J_R`** after construction — it is a frozen dataclass field read by `xi()`. The assertion
cannot fail for any input. It is a regression guard against a hypothetical future edit,
not a measurement, and should not be counted as evidence for the claim.

### 4.6 M2-C10's test is a tautology

See §3.1. `r^v = R_iᵀ(v_j − v_i − g·Δt) − dv̄` evaluated at states produced by
`v_j = v_i + g·Δt + R_i·dv̄` is identically `dv̄ − dv̄`. Measured 1.65e-13, i.e. pure
round-off. The test can only fail on a *sign* error that is not mirrored in `chain()`.

### 4.7 M2-C12's isotropic half tests linear algebra, not the project

`| ‖r‖² − ‖R r‖² | ≤ 1e-12` is orthogonality of SO(3). The only project code on the path
is `weighted_square(r) = r @ r`. The non-isotropic half (`≥ 1.0` for `W = diag(1,2,5)`
over 2000 draws of `r ~ N(0,I)`) is likewise a near-certain event. The one assertion with
real content is `isotropic_precision(diag(.3,.6,.9)) == 3/1.8`, which *is* killed by the
`τ = trΣ/3` mutation. The claim is true; the test's discriminating power is almost
entirely in that one line.

### 4.8 M2-C25's rank 30 and rank 39

Full row rank — the maximum possible. See §3.3.

### 4.9 `REF_R_KF1` is circular

`src/trajectory.py` defines `REF_R_KF1 = R_0 @ REF_DR_01` rather than transcribing it.
Any test using it would be partly circular. No test currently does, so this is latent
rather than active — but it should be replaced with a transcribed literal before anything
starts asserting against KF1's rotation.

---

## 5. Full mutation-sweep results

| # | Mutation | Outcome |
|---|---|---|
| M01 | Forster: update `dv` before `dp` within a step | killed by c10 |
| **M02** | **Forster: fold the step rotation in before using `dR_ik`** | **SURVIVED** |
| M03 | Forster: LEFT bias composition for `dR` | killed by c05 |
| **M04** | **Eq. (22b): swap Y and Y′** | **SURVIVED** |
| M05 | Eq. (22b): sign flip on Γ in R̆ | killed by c11 |
| M06 | Gal3 remap → (0,3,6) | killed by c15 |
| M07 | bias adapter: no block swap | killed by c17, c19, c20, c25 |
| M08 | `endpoint_W`: drop the t_i term | killed by c18 |
| M09 | recursion: `Ad_V` instead of `Ad_{V⁻¹}` (= M2-S02) | killed by c21 |
| M10 | `J_Cay`: sign flip | killed by c07 |
| M11 | `isotropic_precision`: `trΣ/3` instead of `3/trΣ` | killed by c12 |
| **M12** | **Cayley retraction → `Exp` (B2 becomes B1)** | **SURVIVED — and turns the suite GREEN** |
| M13 | `chain()`: drop the ½ on the gravity position term | killed by c02, c03, c18, c25 |
| M14 | `M2()` closed form: sign flip on the K term | killed by c02, c04, c10, c19, c25 |
| **M15** | **`CayleyFactor.xi`: drop the b̄_g offset** | **SURVIVED** |
| M16 | Forster covariance: drop the `A P Aᵀ` term | killed by c06 |
| M17 | Eq. (19): sign flip on the ξ term | killed by c09, c11, c13 |
| M18 | `corrected_increment_gal3`: no bias block swap | killed by c20 |

Kill rate 14/18 (78 %). The four survivors are §4.2–§4.4.

---

## 6. What I would change

Ordered by severity.

1. **Add a positive test that Baseline 2's retraction is `Cay`** (§4.2 i). Today the
   suite rewards deleting it.
2. **Fix M2-C22 / C23 / C24.** Drop "for both admissible readings" (§2.1 — there is one),
   set the raw-reading R tolerance to 1e-6 alongside v and p, and record 3.0e-8 as a
   *property* of the method rather than an error. M2-C24's rotation line should name the
   reparameterised variant or be restricted to Baseline 1.
3. **Fix M2-C10's statement** to say *self-consistent* states, or carry 2.2e-3 (§3.1).
4. **Fix M2-C25's nullspace sub-assertion** (§3.2) — it is false as written, and add the
   principal-cosine measurement to the test so the real structure is pinned.
5. **Correct M2-C16's recorded measurement** from 0.763 to 0.1076 (§3.4) and note the
   bound is near-tight, not 660 % loose.
6. **Add a magnitude assertion to M2-C04** (§4.2 ii) so the within-step rotation order is
   pinned.
7. **Strike the "swapped Y/Y′ shows up immediately" sentence** from `test_c11` (§4.3).
8. **Exercise `CayleyFactor.xi` at a nonzero `δb_g`** (§4.4).
9. Optionally tighten M2-C20's FD ε to 1e-5 and restate the bound as ≤1e-11 (§3.5) — the
   model is far more exact than the claim says.
10. Replace the circular `REF_R_KF1` with a transcribed literal (§4.9).

## 7. Reproduction

All scripts are under the session scratchpad (not committed):
`indep.py` (from-scratch reference), `a_c02.py`, `a_cayley.py`, `a_jcay.py`,
`a_s02_c16.py`, `a_c16b.py`, `a_c25.py`, `a_c25b_c06.py`, `a_c20.py`, `a_batch.py`,
`a_final.py`, `a_gal3.py`, `mutate.py`, `mut2.py`.
Paper text: `rai.html` / `rai.txt` (arXiv:2609.38048v1).
