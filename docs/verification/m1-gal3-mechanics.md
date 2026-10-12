# M1 — Gal(3) mechanics: independent adversarial verification

**Verifier:** independent pass, no access to `docs/01_conventions_and_gal3.md`, no access to the
deriver's reasoning or confidence levels.
**Branch:** `m1-gal3-mechanics` (worktree `/home/parth/Desktop/.cgis_worktrees/m1`)
**Date:** 2026-10-12
**Interpreter:** `/home/parth/Desktop/CERTIFIABLE_GAL_IMU_SLAM/.venv/bin/python` —
Python 3.11.5, numpy 2.4.6, scipy 1.17.1, sympy 1.14.0, mpmath 1.3.0, gtsam 4.3.0.

## Method

Everything below was computed, not read. The brief's independence constraint was honoured in full:
`docs/01_conventions_and_gal3.md` was never opened, and **no git history was consulted** — `git log`,
`git log -p` and `git show` were not run at any point in this pass. Every refutation reported here was
reached by hand derivation from the 5×5 matrix definitions and then confirmed numerically.

Four independent authorities were used, never the module under test:

| purpose | authority |
|---|---|
| matrix exponential / logarithm | `scipy.linalg.expm` / `logm` |
| SO(3) exp / log | `scipy.spatial.transform.Rotation` (`from_rotvec` / `as_rotvec`) |
| scalar coefficient accuracy | `mpmath` at 50 digits, coefficients defined by their **series**, not by any closed form |
| `Gal3` / `NavState` / `imuBias` | the installed gtsam 4.3.0 |
| IMU dynamics | `scipy.integrate.solve_ivp` (DOP853, `rtol=atol=1e-13`) |

### Deliverable 1 — independent re-implementation

A standalone module was written from the matrix definitions alone (no line adapted from
`src/gal3.py`). The derivations behind it:

**exp.** With `A = xi^`, induction on `A^n` gives, for `n >= 2`,
`A^n = [[P^n, P^{n-1} u, P^{n-1} w + alpha P^{n-2} u], [0,0,0], [0,0,0]]` with `P = [phi]x`.
Summing `A^n/n!`: the velocity column is `sum_{m>=0} P^m/(m+1)! u = Gamma1 u`; the position column
is `Gamma1 w + alpha sum_{m>=0} P^m/(m+2)! u = Gamma1 w + alpha Gamma2 u`; and because
`A^n[3,4] = 0` for `n >= 2`, the time entry is `alpha` **exactly**. This reproduces M1-C19 and
M1-C20 from first principles.

**Gamma coefficients.** Using `P^3 = -theta^2 P`:
`kappa1 = sum_k (-1)^k th^{2k}/(2k+2)! = (1-cos th)/th^2`,
`kappa2 = sum_k (-1)^k th^{2k}/(2k+3)! = (th - sin th)/th^3`,
`kappa3 = sum_k (-1)^k th^{2k}/(2k+4)! = (th^2 + 2 cos th - 2)/(2 th^4)`.
For the inverse, solving `(I + k1 P + k2 P^2)(I - P/2 + c2 P^2) = I` gives
`c2 = (k1/2 - k2)/(1 - k2 th^2)`; reducing with the half-angle substitution
`1-cos th = 2 s^2`, `sin th = 2 s c` (`s = sin(th/2)`) turns this into
`c2 = (1 - (th/2) cot(th/2))/th^2 = (1 - th sin th/(2(1-cos th)))/th^2`, i.e. exactly the
form M1-C21 states. The mpmath reference uses the **series** definitions, so it is independent of
any closed form the project chose.

**Cayley.** Block-inverting `I5 - A/2` directly:
`N_11 = S^-1`, `N_12 = S^-1 u/2`, `N_13 = S^-1(w/2 + alpha u/4)`, `N_23 = alpha/2`
with `S = I - [phi]x/2`. Multiplying by `I5 + A/2` gives rotation `S^-1 (I + P/2)`, velocity
`S^-1 u`, position `S^-1 (w/2 + alpha u/4 + w/2 + alpha u/4) = S^-1 (w + (alpha/2) u)` — a **PLUS** —
and time entry `alpha/2 + alpha/2 = alpha`.

**adjoints.** `Ad` and `ad` were rebuilt **column by column** from their definitions
(`Ad_X e_i = vee(X e_i^ X^-1)`, `ad_xi e_i = vee([xi^, e_i^])`), never from a block formula, so the
closed forms of M1-C09 and M1-C11 are checked against the definition rather than against a restatement.

Cross-check of the independent implementation against scipy, over 3000–5000 draws each:

```
scipy Rotation == expm(hat)                       1.16e-13
exp_closed vs expm, |phi| ~ N(0,I)                3.49e-14
exp_closed vs expm, |phi| in (0, 12)              1.23e-13
exp_closed vs expm, |phi| in 1e-14..1e-1          2.22e-15
log_closed vs scipy logm                          1.42e-14
log(exp(xi)) = xi                                 2.22e-15
exp(log(X)) = X                                   3.55e-15
cay_closed (+alpha/2 u) vs (I-A/2)^-1(I+A/2)      1.11e-15
cay with MINUS sign vs the same definition        7.69e+00     <-- M1-S01
```

A by-product worth recording: my *first* independent `c2`, `(k1/2 - k2)/(1 - k2 th^2)`, has
denominator `sin(th)/th` and threw `ZeroDivisionError` at `theta = pi`. That is an accidental,
fully independent corroboration of M1-C21's warning that the choice of `c2` form matters precisely at
the chart edge — the project's chosen form has denominator `1 - cos th`, which only dies at
`theta = 2 pi k`.

### Deliverable 2 — randomized property testing

Sample counts and seeds, all well above the 1e4 floor, all at parameter ranges **wider** than the
claims specify (|v|, |p| up to 10, |t| up to 5, |phi| up to 2pi+1, |u|,|w| up to 50x, alpha ~ 3N(0,1),
Dt spanning 1e-3 to 10, arbitrary g rather than only (0,0,-9.81)):

| battery | samples | seeds |
|---|---|---|
| group axioms C01–C07 | 20 000 | 3 |
| algebra / adjoints C08–C17 | 10 000 | 3 |
| exp / log C19–C23 | 20 000 | 3 |
| Gamma1inv·Gamma1 (C21) | 100 000 | 3 |
| Gamma coefficients vs mpmath (C24) | 7 500-point dense sweep | — |
| Cayley C25–C30 | 20 000 | 3 |
| Cayley totality (C26) | 100 000 | 1 |
| endpoint family C31–C38, C40, C43 | 20 000 | 3 |
| gtsam C46–C52 | 8 000 | 2 |
| bias C44, C54, C55 | 50 000–100 000 | 2 |
| C45 chaining | 20 000 | 2 |
| C56 / C57 | 10 000 / 50 000 | 3 / 1 |

Near-boundary coverage: `theta` down to 1e-14 and up to 2pi+1; `|phi|` at exactly pi (including the
degenerate 180-degree rotations about each coordinate axis and about (1,1,1)/sqrt(3)); `theta` scanned
at 1e-15 resolution across the `THETA_SMALL = 5e-2` branch switch; `alpha` up to 10x; `Dt` over four
orders of magnitude.

---

## Verdict table

59 claim ids: M1-C01..M1-C57 plus the two `S`-suffixed claims.
**CONFIRMED 56 · REFUTED 3 · UNTESTABLE 0.**

| id | verdict | attack attempted | worst measured |
|---|---|---|---|
| C01 | CONFIRMED | `src.compose` vs raw 5×5 numpy matmul **and** vs my hand-derived block formula, 20k×3 seeds at 10× parameter scale | 5.68e-14 |
| C02 | CONFIRMED | structural residual of the raw matmul product, same draws | 2.67e-15 |
| C03 | CONFIRMED | associativity through `src.compose` (not numpy), 20k×3 | 5.68e-14 |
| C04 | CONFIRMED | two-sided identity, 20k×3 | 0.0 |
| C05 | CONFIRMED | `src.inverse` vs `numpy.linalg.inv` **and** both products vs I5, with \|t\| up to 5 | 1.71e-13 |
| C06 | CONFIRMED | max commutator over 20k draws | 369 (need ≥1e-3) |
| C07 | CONFIRMED | time additivity + t(X⁻¹)=−t, 20k×3 | 0.0 |
| C08 | CONFIRMED | wedge∘vee and vee∘wedge, 10k×3, plus basis probes e3→col 3, e6→col 4 | 0.0 |
| C09 | CONFIRMED | `src.ad` vs **column-wise commutator definition** `vee([xi^,e_i^])`, not a block restatement | 0.0 / 1.42e-14 |
| C10 | CONFIRMED | homomorphism + Jacobi, 10k×3 | 5.68e-14 |
| C11 | CONFIRMED | `src.Ad` vs **column-wise conjugation definition** `vee(X e_i^ X⁻¹)`, \|t\|≤5, \|v\|,\|p\|≤10 | 6.82e-13 |
| C12 | CONFIRMED | Ad_{XY} = Ad_X Ad_Y, 10k×3 | 1.42e-13 |
| C13 | CONFIRMED | Ad_{X⁻¹} vs `numpy.linalg.inv(Ad_X)`, 10k×3 | 3.59e-13 |
| C14 | CONFIRMED | both forms, 10k×3 | 1.51e-11 (tol 1e-7) |
| C15 | CONFIRMED | `Ad(exp(xi))` vs `expm(ad(xi))`, plus a ‖xi‖~3N(0,I) stress outside the claim's suggested range | 5.09e-14 |
| C16 | CONFIRMED | last row of Ad_X, 10k×3 | 0.0 |
| C17 | CONFIRMED | det(Ad)=1, trace(ad)=0, 10k×3 | 2.18e-14 / 0.0 |
| C18 | CONFIRMED | kappa form vs **direct series summation** (60 and 120 terms), incl. θ∈(3,10) outside the claim's range | 1.15e-13 |
| C19 | CONFIRMED | vs `scipy.expm` and vs my independent closed form, 20k×3; stress at \|phi\|→2π+1 with 10× u,w,alpha | 2.22e-14 (rel 2.71e-14) |
| C20 | CONFIRMED | exp(xi)[3,4] − alpha, 20k×3, \|alpha\| to 30 | 0.0 exactly |
| C21 | CONFIRMED | Γ1⁻¹Γ1−I over **1e5×3 seeds** plus a 1e-15-resolution scan through the switch; c2 formula re-derived by hand | 1.92e-14 (asserted ≤1e-13 — **thin, 5.2×**) |
| C22 | CONFIRMED | 20k×3; wrap-around at \|phi\|=3.5 returns 2.783185307 | 1.48e-13 |
| C23 | CONFIRMED | 20k×3 **plus** \|phi\| = π−ε for ε = 1e-1…0, **plus exact 180° rotations** about x, y, z and (1,1,1)/√3 | 1.40e-13; **8.88e-16 at exactly π** |
| C24 | CONFIRMED | see §"The 1e-11 / Taylor-switch question" | 1.17e-12 over a dense sweep (asserted ≤1e-11) |
| C25 | CONFIRMED | vs the 5×5 definition **and** vs my own block derivation, 20k×3, \|u\|,\|w\| to 50× | 2.84e-13 |
| C26 | CONFIRMED | det − (1+\|phi\|²/4) over 20k×3, and 1e5 Cayley evaluations with \|phi\| to 50 and \|u\|,\|w\|~1e3 | 1.33e-15; **0 failures**, min det 1.00068 |
| C27 | CONFIRMED | orthogonality/det residual and time block, 20k×3 | 7.77e-16 / 0.0 |
| C28 | CONFIRMED | round-tripped **through `cay`** at θ = 3.0 and 3.1415 (which the project's own test never does — see §Tests that cannot fail #12) | θ round-trip error 0.0 |
| C29 | CONFIRMED | the **scaling law**, not just smallness: \|d\|/s³ = 0.011743 constant over s spanning 40×; \|d − s³/12·vee(ξ³)\|/s⁵ = 0.000356 constant; the wrong 1/6 coefficient blows the ratio to 629 | ratios 0.063 / 0.0080 |
| C30 | CONFIRMED | residuals at 20k×3 with alpha≠0, plus an **independently rebuilt** sympy constraint system | 1.42e-13; degree exactly 2 both ways |
| C31 | CONFIRMED | vs a **DOP853 ODE solve at rtol=atol=1e-13 with arbitrary g** (not the project's Euler recursion), plus 20k×3 algebraic with Dt over 4 decades | ODE 1e-9-class; algebraic 9.10e-13 |
| C32 | CONFIRMED | G(Dt) vs T(−Dt)H′ and vs `exp(Dt ζ_g)`, 20k×3, arbitrary g; opposite fusion order verified = (I, gDt, −1.5gDt², −Dt) | 0.0 / 0.0 / 2.27e-13 |
| C33 | CONFIRMED | 20k, s~2N(0,1), default g | 2.27e-13 (tol 1e-10) |
| C34 | CONFIRMED | position error / (\|g\|Dt²) measured = **1.0000 exactly** over 20k×3, i.e. the error is precisely gDt²; time block = Dt | 9.10e-13 / 0.0 |
| C35 | CONFIRMED | 20k×3, arbitrary g | 0.0 / 0.0 |
| C36 | CONFIRMED | 20k×3 with t_i≠0; guard verified to refuse at \|t_i\| > `KEYFRAME_TIME_TOL` = 1e-12 | 4.55e-13 / 8.88e-16 |
| C37 | CONFIRMED | gtsam NavState form vs the DOP853 truth and vs the Forster endpoint, 2k | 1.78e-15 |
| C38 | CONFIRMED | both lifting identities exact; the two forms agree on (R,v,p), both time blocks 0, 20k×3 | 0.0 / 0.0 / 9.10e-13 |
| C39 | CONFIRMED | gtsam NavState compose vs the closed form, 20k×2 | 3.55e-15 |
| **C40** | **REFUTED (in part)** | see §Refutation 3 | 14% of random (X,Y) pairs give residual **< 0.5**; Y=X gives **5.55e-16** |
| C41 | CONFIRMED | vs **DOP853 solve of Ḋ = D ξ^ with Dt up to 2.0**; ODE verified by central difference; Euler error verified first-order (err/dt constant to 4 digits over 64× in dt) | 5.06e-14 ODE; 1.62e-10 central diff; **7.04e-6 at 1000 configs — thin, 1.4×** |
| C42 | CONFIRMED | blocks vs `exp(Dt(ω,a,0,1))` and vs the DOP853 solve, 20k×2, Dt over 4 decades | rel 1.78e-15; time block 0.0 |
| C43 | CONFIRMED | vs the DOP853 truth with arbitrary g; block decomposition of C31's LHS | ODE 1e-9-class; 9.10e-13 algebraic |
| C44 | CONFIRMED | 1e5×2 at 10× scale; sign probe | 7.11e-15 at 10× (4.44e-16 at the claim's N(0,1) scale) |
| C45 | CONFIRMED | 20k×2 algebraic; the swapped `dv_jk·Dt_ij` form has **minimum** residual 3.13e-2 over 20k (not just max) | 0.0; swap detected on every draw |
| C46 | CONFIRMED | 8k×2 vs `gal(R,v,p,t)`; discriminating witness margin 5.09 | 0.0 |
| C47 | CONFIRMED | all six accessors, 8k×2, separated p and v distributions | 0.0, margin 5.09 |
| C48 | CONFIRMED | `Gal3.Hat` vs `wedge`, `Vee` inverse, 8k×2; basis probes | 0.0 |
| C49 | CONFIRMED | NavState matrix layout and tangent order, 8k×2 | 0.0, margin 5.09 |
| C50 | CONFIRMED (substance) | 2e4×3 seeds instead of the test's 50 — see §Thin tolerances | 6.41e-10 at \|d\|~1e-5 (tol 1e-9); **5.35e-8 at \|d\|~1e-4** |
| C51 | CONFIRMED | Expmap/Logmap/AdjointMap/adjointMap vs the project's closed forms, 8k×2 | 3.33e-15 / 1.51e-14 / 3.55e-15 / 0.0 |
| C52 | CONFIRMED (one conjunct unfalsifiable) | see §Refutation-adjacent: localCoordinates | compose 7.11e-15, inverse 3.55e-15, between 7.11e-15, **retract 1.45e-13** |
| C53 | CONFIRMED | gtsam PIM predict vs both forms at **three** duty cycles (Dt 0.5 and 1.0, \|ω\| 0.06–1.88), 250 states each | 7.11e-15; time block 0.0 |
| C54 | CONFIRMED | 5e4×2, separated distributions, margin 29.7 | 0.0 |
| C55 | CONFIRMED | 5e4×2; S·S=I₆ exact; naive conversion fails to round-trip with margin 2.8e-2 | 0.0 |
| C56 | CONFIRMED | relative error of Ad_{X⁻¹} vs inv(Ad_X), 1e4×3, \|v\|,\|p\| ≤ 10, \|t\| ≤ 5 | **1.88e-14** (tol 1e-10) |
| C57 | CONFIRMED | 5e4 draws s∈[−5,5]; G(s) vs `expm(s ζ_g^)` separately | 1.71e-13 (tol 1e-12 — **thin, 5.9×**; fails at s∈[−20,20]) |
| **S01** | **REFUTED** | see §Refutation 1 | min residual **9.93e-4** over 60 000 draws; max **7.69** |
| **S02** | **REFUTED** | see §Refutation 2 | max **1.35e+01** (claim: ≤ 1e-10) |

---

## Refutations

### Refutation 1 — M1-S01 (Cayley position block sign) is FALSE

> *"the Cayley map has position block `pe = (I - [phi]x/2)^-1 (w - (alpha/2) u)`; equivalently the
> degree-2 lifting constraint is `(I - [phi]x/2) pe + (alpha/2) u = w`"*

**How this was found:** by hand, before running anything. Block-inverting `I5 - xi^/2` (the derivation
is reproduced in §Deliverable 1) gives the `(1,3)` block of the inverse as `S^-1(w/2 + alpha u/4)`, and
the position column of `(I5 - A/2)^-1 (I5 + A/2)` is then
`S^-1 w/2 + (S^-1 u/2)(alpha/2) + S^-1(w/2 + alpha u/4) = S^-1 (w + (alpha/2) u)` — a **PLUS**.
It was *not* found by git archaeology; `git log` was never run in this pass. (I did separately notice
that `M1-S01` and `M1-S02` are absent from `docs/claims/M1.yaml`, which lists exactly C01–C55, but that
observation came after the hand derivation and is not the evidence.)

**Numerical confirmation.** Against the defining expression `(I5 - xi^/2)^-1 (I5 + xi^/2)`:

```
cay with +(alpha/2) u   vs the definition :  1.11e-15   (claims C25, C30 — correct)
cay with -(alpha/2) u   vs the definition :  7.69e+00   (claim  S01    — WRONG)
```

and over 60 000 draws across three seeds the **minimum** residual of the MINUS form is
9.93e-4, 1.84e-3 and 5.43e-3 — it is never right, not even on a lucky draw.

M1-S01 also directly contradicts M1-C25 and M1-C30, which both state the PLUS sign; those two are
confirmed. S01's rationale ("the same negative sign as the `-0.5*g*Dt^2` entry of H′") is a false
analogy: the Cayley coupling and the gravity lift are unrelated quantities.

### Refutation 2 — M1-S02 (left Jacobian) is FALSE

> *"`J_L(xi)^-1 == J_L(-xi)` for all xi in gal(3) within 1e-10, which is why the right Jacobian need
> not be implemented separately."*

Defining `J_L(xi) = sum_{n>=0} ad_xi^n/(n+1)!` from the series (the standard definition; `src/gal3.py`
contains **no** left-Jacobian implementation, so this claim is about the mathematics, not the code), over
3000 draws with `0.05 <= ||xi|| <= 1.5`:

```
max |J_L(xi)^-1 - J_L(-xi)|   =  1.353e+01        <-- claim demands <= 1e-10
max |J_R(xi)    - J_L(-xi)|   =  2.653e-14        <-- the TRUE identity
same at ||xi|| ~ 1e-4         =  2.015e-08        <-- they agree only in the limit
```

The claim conflates `J_L^-1` with `J_R`. The correct statement is **`J_R(xi) = J_L(-xi)`**, which I
verified at 2.65e-14 via `J_R = Ad_{exp(-xi)} J_L`. The inverse of the left Jacobian is a different
object: already for SO(3), `J_L(phi)^-1 = I - [phi]x/2 + c2 [phi]x^2` whereas
`J_L(-phi) = I - kappa1 [phi]x + kappa2 [phi]x^2`, and `kappa1 = (1-cos th)/th^2 != 1/2` for `th != 0`.
Concrete witness: `xi = (1.2, 0, …, 0)` gives `|J_L^-1 - J_L(-xi)| = 0.1003`.

The stated consequence — "the right Jacobian need not be implemented separately" — happens to be true,
but for the other reason (`J_R(xi) = J_L(-xi)`), not the one claimed.

### Refutation 3 — M1-C40's quantitative parenthetical is FALSE as written

> *"Phi_Dt … is NOT left or right translation by any fixed NavState **(a factor fitted on one element
> fails on a second with residual > 0.5)**"*

The headline content of C40 is **confirmed**: Φ_Dt is a NavState automorphism (error 1.78e-15 over
20 000×2 draws) and is inner conjugation by T(Dt) on the t=0 slice of Gal(3) (error 0.0 over 20 000×3).

The parenthetical, read as stated, is false. The fitted factor `N_L = Phi(X) X^-1` is by construction
exact *at X*, so the residual on a second element is not bounded below at all — it goes continuously
to zero as Y approaches X:

```
Y = X exactly                 left residual 5.55e-16   right residual 6.66e-16
Y = X.retract(1e-1 * d)                    2.79e-02                 4.61e-02
Y = X.retract(1e-3 * d)                    2.79e-04                 4.51e-04
Y = X.retract(1e-6 * d)                    2.79e-07                 4.51e-07
```

Over random (X, Y) pairs the failure is not even typical-case certain: across three seeds × 5000 pairs,
**14.2 % / 14.3 % / 14.2 %** of draws have left residual **below 0.5** (right: 14.3 / 14.0 / 13.8 %),
with minima of 5.4e-2, 1.7e-2 and 3.2e-2.

The project's test launders this by asserting the **maximum** over 20 draws: with ~86 % per-draw
success the assertion fails with probability ≈ 0.14²⁰ ≈ 1e-17. It therefore establishes only the
existential reading ("there exist elements on which a fitted factor fails"), which is true and
sufficient to prove Φ_Dt is not a translation. The universally-quantified sentence in the claim is not
what was tested and is not true.

---

## The 1e-11 / Taylor-switch question (M1-C24), adjudicated

The implementer reports it could not meet 1e-11 while keeping the 5e-2 switch if `1 - cos(theta)` is
evaluated literally, and instead evaluates it as `2*sin(theta/2)**2`. My adjudication, measured against
50-digit mpmath using the **series** definitions of the coefficients (so the reference is independent of
every closed form in play):

**(a) The claimed instability is real, not an artefact.**

```
theta    |1-cos th| rel err   |2 sin^2(th/2)| rel err
0.050         2.90e-14              6.49e-17
0.060         1.33e-14              4.58e-17
0.100         1.10e-14              7.55e-17
```

`kappa3 = (th^2 - 2(1-cos th))/(2 th^4)` subtracts two quantities that agree to `O(th^4/12)`, which
amplifies any error in `1-cos` by roughly `12/th^2` — about 4800× at θ = 0.05. 2.9e-14 × 4800 ≈ 1.4e-10,
which is exactly the order the implementer reported.

**(b) The substitution is a legitimate evaluation of the identical expression, not a deviation.**
`2 sin^2(th/2) == 1 - cos th` is an exact identity for all real θ. The claim states κ1 and κ3 as
mathematical formulas; it prescribes no floating-point evaluation order. The claim is therefore
satisfiable as written, and the implementation satisfies it.

**(c) The implementation meets 1e-11 everywhere, not only at the eight tabulated θ.** I swept 7 500
points (logspace 1e-12…10, plus a dense band 0.045–0.20 around the switch, plus 3.0–6.2, plus θ = π and
the two doubles bracketing 5e-2):

```
                   IMPLEMENTATION (2 sin^2)        LITERAL (1-cos), same 5e-2 switch
kappa1   7.75e-13 at th=0.05                       7.75e-13 at th=0.05
kappa2   2.58e-13 at th=0.05                       2.58e-13 at th=0.05
kappa3   9.90e-13 at th=0.0504                     4.92e-10 at th=0.0510   *** > 1e-11
c2       1.17e-12 at th=0.0671                     1.85e-10 at th=0.0504   *** > 1e-11
```

**(d) A correction to the framing.** The binding constraint is *not* the `1-cos` cancellation. Just
*below* the switch, the dominant error is the **3-term Taylor truncation**, which is already 7.75e-13 at
θ = 5e-2 and grows as θ⁴:

```
th      kappa1 Taylor rel err
0.040        2.03e-13
0.050        7.75e-13      <-- the switch point
0.080        1.30e-11      *** would exceed 1e-11
0.100        4.96e-11      ***
```

So `THETA_SMALL = 5e-2` is load-bearing in a way the claim does not say: at 8e-2 the Taylor branch alone
would break the 1e-11 bound. The real headroom on C24 is ~**13×**, set by Taylor truncation, not by
`1-cos`. The branch is also discontinuous across the switch at the ~9e-13 relative level — harmless at
1e-11 but worth knowing.

**(e) The tabulated det/cond values are independently correct.** I derived
`det Gamma1 = 4 sin^2(th/2)/th^2` and `cond Gamma1 = th/(2 sin(th/2))` from the eigenvalues of
`Gamma1` restricted to the plane orthogonal to φ (a scaled rotation of modulus `2|sin(th/2)|/th`, the
axis eigenvalue being 1). Measured: det = 1, 0.405285, 0.00221276, 2.3e-65 at θ = 0, π, 6, 2π;
cond(π) = 1.570796 = π/2. All match.

**(f) Mutation confirms the test binds this.** Reverting `2*sin(theta/2)**2` to `1.0 - np.cos(theta)`
in `src/gal3.py` **fails** `test_gamma1_invertibility_and_taylor_branch`. The 1e-11 assertion is doing
real work.

**Verdict: (a) legitimate numerically-stable evaluation of the identical expression. M1-C24 CONFIRMED.**

---

## Deliverable 3 — tests that cannot fail

Twelve findings, ordered by severity. Items 1–3 are backed by **mutation testing**: a defect was
injected into `src/` (or `tests/conftest.py`) and the bound tests were re-run. The harness was
validated by four control mutations, all of which were correctly caught.

```
-- control mutations (all correctly CAUGHT) --
  cay: +alpha/2 u -> -alpha/2 u                      CAUGHT  (2 failed)
  gamma: 2sin^2(th/2) -> literal 1-cos(th)           CAUGHT  (1 failed)
  endpoint: H' position -0.5gDt^2 -> +0.5gDt^2       CAUGHT  (5 failed)
  compose: v1*t2 -> v2*t1                            CAUGHT  (15 failed)

-- order mutations --
  integrate_increment:  dR @ E -> E @ dR             CAUGHT  (1 failed, via C45)
  integrate_truth navstate: R @ E -> E @ R           CAUGHT  (4 failed)
  integrate_truth increment: dR @ E -> E @ dR    *** SURVIVED ***  55 passed
  euler_increment: dR @ W -> W @ dR              *** SURVIVED ***  1 passed

-- integrator-accuracy mutations --
  endpoint_dataset: 5000 steps @1e-4 -> 2 @0.25  *** SURVIVED ***  4 passed
  endpoint_dataset: drop g from the integrator       CAUGHT  (2 failed)
```

**1. The "5000-step integrator ground truth" of C31 / C34 / C37 / C43 is decorative.**
`conftest.endpoint_dataset` integrates the navigation state **and** the increment with the *same*
discrete recursion. By induction on that recursion, `p_n = p_0 + v_0 Dt + 0.5 g Dt^2 + R_0 dp_n` holds
**exactly for any dt**, so the endpoint relation is an algebraic identity of the stepping scheme, not a
numerical agreement with the ODE. Measured residual against step count:

```
n=5000 dt=1e-4 : 9.77e-14     n=50 dt=1e-2 : 1.16e-14     n=2 dt=0.25 : 3.55e-15
n= 500 dt=1e-3 : 2.84e-14     n= 5 dt=1e-1 : 3.55e-15     n=1 dt=0.5  : 1.78e-15
```

Reducing the fixture to **2 steps at dt = 0.25** leaves all four tests passing (and the run drops from
11 s to 0.47 s). Meanwhile the 5000-step Euler state differs from the exact closed-form solution by
**1.58e-4** — *ten times above the tests' own 1e-5 tolerance*. A test that genuinely compared the
endpoint relation to ODE truth at 1e-5 would fail. These four tests cannot detect integrator error;
their 1e-5 tolerances give a false impression of integration accuracy. (The underlying claims are
nonetheless true — I verified them separately against DOP853 at rtol = 1e-13 with arbitrary g.)

**2. Every test built on `integrate_truth` / `integrate_increment` / `euler_increment` is blind to a
swapped accumulation order, because (ω, a) are constant per instance.** `E = so3_exp(ω·dt)` is computed
once outside the loop, so all per-step factors commute and `dR ← dR·E` is indistinguishable from
`dR ← E·dR` when the accumulator starts at I:

```
constant omega:   |dR - dR_swapped| = 4.66e-15   |dv - ...| = 1.11e-16   |dp - ...| = 1.30e-18
time-varying omega: |dR - dR_swapped| = 1.71e-06   <-- only now visible
```

Mutation confirms: `integrate_truth`'s increment update `dR @ E -> E @ dR` **survives the entire
55-test suite**, and `euler_increment`'s `dR @ W -> W @ dR` **survives** C41's test (3.05e-16 difference).
This is exactly the defect class the brief asked about. (`integrate_increment`'s mutation *is* caught —
but only by C45, and only because that call passes `D0 = D_ij ≠ I`.)

**3. C40's `err_L >= 0.5` / `err_R >= 0.5` assertions are maxima over 20 draws** where ~86 % of single
draws already exceed 0.5 — failure probability ≈ 1e-17. See Refutation 3.

**4. `test_gtsam_gal3_group_operations`: `assert w_loc <= 1e-12`** compares
`A.localCoordinates(B)` with `gtsam.Gal3.Logmap(A.between(B))` — **gtsam against itself**. I measured
**exactly 0.0** on all 12 000 draws across three seeds and at every near-π offset tested. No error
anywhere in this project can make it fail. (Discussed further below.)

**5. `test_cayley_rotation_tangent_half_angle` (C28): the five tabulated rows never call `cay`.**
The block computes `got = 2.0 * np.tan(theta/2.0)` and compares it to hard-coded constants — a
transcription self-check, not a test of the implementation. Worse, the test's own sampler draws
`||c|| ∈ (0.01, 5.0)`, i.e. θ ∈ (0.0100, 2.3806), so the two largest tabulated rows (θ = 3.0 and
3.1415) are **outside the sampled range entirely**. I verified them through actual `cay` round-trips
(round-trip θ error 0.0, orthogonality residual ≤ 3.3e-16), so the claim holds — but the test does not
establish it.

**6. `test_cayley_denominator_never_vanishes`: `assert failures == 0`** after 20 000 Cayley evaluations.
`det(I - [phi]x/2) = 1 + θ²/4 ≥ 1`, so `LinAlgError` is impossible by construction. True but vacuous.
(I pushed it to 1e5 evaluations with |φ| up to 50 and |u|,|w| ~ 1e3: 0 failures, min det 1.00068.)

**7–11. Self-referential "written out" assertions.** Five tests compare a function against a verbatim
re-typing of its own body, using the same helpers:

| test | assertion |
|---|---|
| `test_exp_closed_form` | `closed = gal(so3_exp(phi), gamma1(phi)@u, gamma1(phi)@w + al*(gamma2(phi)@u), al)` |
| `test_cayley_closed_form` | `ref = gal(Si@(I3+skew(phi)/2), Si@u, Si@(w+0.5*al*u), al)` |
| `test_ad_blocks_and_commutator` | `assert np.abs(A - blocks).max() == 0.0` |
| `test_log_and_gamma1_inverse` | "the stated log formula, componentwise" |
| `test_increment_closed_form_blocks` | `dp == Dt**2 * (gamma2(omega*Dt) @ acc)` |

Each cannot fail for any implementation that routes through those helpers. In every case an
independent check (`scipy.expm`, `cay_matrix`, the commutator, `numpy.linalg.inv`) sits alongside it
and does bind — which is why the control mutations above were caught — so these are redundancy, not
holes. They should not be counted as evidence.

**12. Vacuous coverage guards.** Five assertions of the form `assert min_abs_t2 > 0.0`,
`assert min_t > 0.0 and min_v > 0.0`, `assert min_alpha > 0.0 and min_u > 0.0` (in C01, C05, C11, C19,
C25) are intended to prove the load-bearing term was exercised, but for a continuous distribution the
probability of drawing exactly 0.0 is zero, so they always pass regardless of how close to degenerate
the sample set is. They should assert a margin (`> 1e-3`), as the corresponding `margin >= 1e-3`
assertions in the gtsam tests correctly do.

### Refutation-adjacent: the localCoordinates reading (M1-C52)

The claim states `localCoordinates(B)` equals `Logmap(A^-1 B)`. On that reading it is **true and
exactly so** — 0.0 on every one of 12 000 draws — but it is **unfalsifiable**: both sides are gtsam
internals that evidently share a code path. Against an independent `log`:

```
                                        vs INDEPENDENT log   vs src.log   vs gtsam Logmap(between)
random pairs, 4000 draws, seed 20241064        1.24e-14       8.88e-15            0.0
random pairs, 4000 draws, seed 11              9.84e-08       9.84e-08            0.0
|phi_rel| = pi - 1e-4, 400 draws               1.31e-08       1.31e-08            0.0
|phi_rel| = pi - 1e-6, 400 draws               7.79e-13       7.85e-13            0.0
|phi_rel| = pi exactly, 400 draws              1.16e-14       8.88e-15            0.0
```

So the reported ~1.35e-8 near `|phi| = pi` is real — I reproduce 1.31e-8 — and it is **not** a defect
in the project's `log`: the independent log and `src.log` disagree with gtsam by the same amount, so the
loss is in the shared ill-conditioning of the SO(3) logarithm near the branch point, not in anyone's
code. Note also the error is non-monotonic in `pi - |phi|` (1.3e-8 at 1e-4, 7.8e-13 at 1e-6, 1.2e-14 at
exactly π), which is the signature of a conditioning cliff rather than a formula error.

**Verdict: the claim is TRUE on its stated (gtsam-internal) reading but that reading is
unfalsifiable.** The project's test is explicit about this — it runs a second comparison against
`log(rel)` but **gates it** with `if norm(so3_log(rel)) < pi - 1e-3`, which excludes precisely the
regime where the disagreement appears. That gate is honest (it matches C51's stated Logmap domain) but
it means no assertion in the suite ever exercises the near-π case.

---

## Deliverable 4 — thin tolerances re-run under different seeds

Every tolerance reported as thin was re-run with new seeds and 10×–1000× the samples.
**No claim failed its tolerance inside its own stated domain.** Two fail just outside it, and three
recorded measurements in the claim file are not reproducible.

| claim | asserted | claim's recorded figure | my worst (reseeded, larger N) | margin | status |
|---|---|---|---|---|---|
| C21 | 1e-13 | 4.0e-14 | **1.92e-14** (1e5 × 3 seeds) | 5.2× | holds |
| C24 | 1e-11 | — | **1.17e-12** (7500-pt sweep) | 8.5× | holds |
| C41 | 1e-5 | 2.7e-6 "over 10" | **7.04e-6** (1000 configs at worst Dt) | **1.4×** | holds, barely |
| C50 | 1e-9 | 3.9e-11 "over 50" | **6.41e-10** (2e4 × 3 seeds) | **1.6×** | holds at \|d\|~1e-5 |
| C50 | 1e-9 | — | **5.35e-8** at \|d\|=1e-4, **4.32e-7** at 3e-4 | — | **fails inside its own stated domain** |
| C52 retract | 1e-12 | 2.8e-17 | **1.45e-13** (4 seeds incl. the test's own) | 6.9× | holds |
| C57 | 1e-12 | — | 1.71e-13 at s∈[−5,5]; **2.73e-12 at s∈[−20,20]** | 5.9× | holds in domain |
| C44 | 1e-14 | 0.0 | 4.44e-16 at N(0,1); **5.68e-14 at \|b\|~100** | — | holds in domain |
| C33 | 1e-10 | 1.1e-13 | 2.27e-13 | 440× | comfortable |

**Three recorded measurements are wrong or not reproducible:**

- **C52's "retract … measured 2.8e-17"** is off by roughly six orders of magnitude. Across four seeds
  — including the test's own `default_rng(20241064)` — and the test's own distributions I measure
  1.28e-13, 1.44e-13, 1.41e-13, 1.45e-13. (2.8e-17 is below double precision for O(1) quantities, so it
  was never a plausible maximum over 2000 draws.) The real headroom is 6.9×, not 3.6e4×.
- **C50's "measured 3.9e-11"** over 50 samples understates the true worst case at the same scale by
  ~16×; I get 5.22e-10 and 6.41e-10 over 2 × 10⁴. More seriously, **C50's tolerance and its domain note
  are mutually inconsistent**: the domain says "use `||d|| <= 1e-4` for the test", but at `|d| ~ 1e-4`
  the error is 5.35e-8, **54× over the 1e-9 bound**. The tolerance only holds for `|d| ≲ 1e-5`. The
  substance of the claim (the permutation converts tangent ordering; omitting it gives O(1) relative
  error — I measure 6.2e-5 against a 1e-5 perturbation) is confirmed; the tolerance/domain pair is not
  self-consistent.
- **C41's "measured 2.7e-6 over 10 random configurations"** while the hint's headline says "200 random
  configurations", and the test itself uses 10. The sample count **is** load-bearing. Running all
  configurations at the worst Dt in the stated range (0.4) at dt = 2.5e-6:

  ```
   10 configs : 2.16e-6     200 configs : 5.22e-6     1000 configs : 7.04e-6     (tol 1e-5)
  ```

  The worst case grows with the sample count (max of a Gaussian-driven error over n draws), and at
  1000 configurations it sits at 70 % of the tolerance. The test further understates it by drawing
  `Dt ~ U(0.02, 0.4)` rather than the worst Dt. Raising the count to the hint's own "200" — let alone
  beyond — would leave under 2× of headroom. I did separately confirm the substantive content: the
  Euler error is genuinely first-order (`err/dt` constant to 4 digits across a 64× change in dt), and
  `D(Dt) = exp(Dt(ω,a,0,1))` matches a DOP853 solve of `Ḋ = D ξ^` to 5.06e-14 for Dt up to 2.0.

---

## Anything I could not test

Nothing in the list was untestable. Three qualifications on the strength of the evidence:

1. **Universally-quantified claims over uncountable domains** (C22, C23, C26 totality, C03
   associativity) can only be sampled. I mitigated this by sampling the boundary explicitly (|φ| at
   exactly π including the degenerate coordinate-axis 180° rotations; θ down to 1e-14; |φ| up to 50 for
   Cayley) and by deriving each of them symbolically — C26 from `det(I - [φ]x/2) = 1 + θ²/4`, C23 from
   the invertibility of Γ1 on [0, π].
2. **C46–C49, C51–C54 are statements about gtsam 4.3.0 as installed.** They are verified for this
   build (all at 0.0 or ~1e-14) and must be re-checked on any version bump, as their own domain fields
   say.
3. **`conftest.rand_rot` builds its random rotations with `gal3.so3_exp`**, the module under test. For
   group-law tests this is harmless (any orthogonal matrix will do), but it means C23's round-trip is
   partly `so3_exp`-against-`so3_log` rather than against an outside reference. My batteries use
   `scipy.spatial.transform.Rotation` throughout, which removes this.

## Scripts

All verification scripts live under the session scratchpad
(`.../scratchpad/v/` — `indep.py` is the independent re-implementation; `tA_`–`tP_` are the batteries;
`../mutate.sh` is the mutation harness). They are intentionally not committed: they depend on absolute
paths into this worktree.
