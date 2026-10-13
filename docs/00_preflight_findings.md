# Pre-flight findings

Established **before** any milestone work, by direct numerical check. These are the ground the
Stage B triad builds on. All of F2–F4, F8, F9 are reproducible with:

```bash
python -m experiments.preflight.gal3_checks
```

Every one of these is re-checked independently by the M1 verifier. Nothing here is to be taken on
trust merely because it is written down.

---

## F2 — the project's own math contract has a reproducible error (TWO defects)

`CLAUDE.md` warns that `X_j = H_ij X_i D_ij` has a nonzero time block. The warning is correct but
**incomplete**. Checked against ground-truth IMU integration (5000 steps, `dt=1e-4`, nonzero
`omega`, `a`, `v`):

| claim as written | result |
| --- | --- |
| time block of RHS `= 0` | **FAILS** — equals `Delta t` |
| `p` block `= p_j` | **FAILS** — off by exactly `+g*Dt^2` |
| `v` block `= v_j` | passes |
| `R` block `= R_j` | passes |

**Root cause.** With `H` on the *left*, its velocity block re-enters the position accumulation
through the group product's `v1*t2` term, double-counting gravity. The sign of `H`'s position entry
must flip.

**Corrected relation** (verified exact to `1e-5` on the full 5x5):

```
X_j = T(-Dt) . H' . X_i . D_ij
H'     = [[I, g*Dt, -0.5*g*Dt^2], [0,1,0], [0,0,1]]   # MINUS
T(-Dt) = [[I, 0, 0], [0,1,-Dt], [0,0,1]]              # applied on the LEFT
```

`T(-Dt)` on the left zeroes the time block without perturbing `(R,v,p)`. On the right it would
corrupt `p` by `-v*Dt`. **This corrected relation is the convention for all downstream work.**

> This defect is the canonical **seeded defect** for verifier calibration (see
> `orchestrator/state.yaml`). It is a real error that shipped in a document presented as
> authoritative, which makes it an honest test of whether a verifier checks or defers.

## F3 — the Cayley constraint block is correct

For `e^ = [[phi_x, u, w],[0,0,0],[0,0,0]]`, `Cay(e)` lands in `Gal(3)` with zero time block, `Re` is
exactly `SO(3)` with `det=+1`, and all three equalities hold, each **degree <= 2** when
`(phi, Re, ve, pe)` are independent lifted variables:

```
(I - phi_x/2) Re = I + phi_x/2
(I - phi_x/2) ve = u
(I - phi_x/2) pe = w
```

> **F3 was INCOMPLETE — corrected by M1.** The three equalities above were only ever checked on the
> `alpha = 0` slice. For general `alpha != 0` the position block carries a time coupling:
>
> ```
> (I - phi_x/2) pe = w + (alpha/2) u          # sign is PLUS
> ```
>
> Verified independently to `5.7e-14` over 20000 random `gal(3)` elements with `alpha != 0`; the
> minus form is off by up to `2.5e2`. At `alpha = 0` the term vanishes identically, which is exactly
> why the original check could not see it. The constraint is still degree <= 2, since `alpha` and `u`
> are independent lifted variables. **Use `docs/01_conventions_and_gal3.md` as authoritative.**

**Chart coverage measured:** `|phi| = 2 tan(theta/2)` gives `1.09` at 1 rad, `28.2` at 3 rad,
`4.3e4` at 3.1415 rad. Cayley is numerically unusable near `theta = pi`.

**Consequence for lifting (20000 random samples, `det` range `[1.0, 1.0]`):** the Cayley residual
rotation `Re` needs **no** determinant constraint — Cayley yields a proper rotation automatically.
The **absolute** rotations `R_i, R_j` are *not* Cayley-parameterised and **do** need full `SO(3)`
lifting including cross-product handedness. Conflating the two over-constrains (hurting tightness)
or admits `det = -1` spurious points.

## F4 — group product and inverse are correct

`X1 X2 = (R1R2, v1 + R1 v2, p1 + R1 p2 + v1 t2, t1 + t2)` and
`X^-1 = (R^T, -R^T v, -R^T(p - tv), -t)` both verified numerically.

## F7 — problem size is not a risk

3-KF instance lands around `n ~ 150-190`. Rai et al. solve `n = 334` to `n = 970` on a 2017-era CPU.

## F8 — the Gal(3) adjoint is not a lifting obstruction *(Stage C material, preserved not executed)*

```
phi'   = R phi
u'     = R u + [v]x R phi
w'     = R w - t R u + [p - t v]x R phi + alpha v
alpha' = alpha                                   (time coordinate is Ad-invariant)
```

Raw degree **3**, but a two-stage lifting closes it: `a1 = R phi`, `a2 = [v]x a1`, `a3 = [p]x a1`,
`a4 = R u`, each degree 2.

Also measured: over 20000 random `gal(3)` elements *with nonzero time coordinate*, **Cayley is total
on `gal(3) -> Gal(3)`** (zero failures) and its only denominator is `det(I5 - A/2) = 1 + ||phi||^2/4 > 0`
(max deviation `9.1e-13`) — the exact analogue of Rai Eq. (17).

## F9 — a 2-keyframe instance CANNOT observe biases

Finite-difference Jacobian rank over the 30-DOF 2-keyframe state:

| configuration | residuals | rank | nullity |
| --- | --- | --- | --- |
| IMU edge alone | 15 | 15/30 | **15** |
| + pose anchor on `i` | 21 | 21/30 | 9 |
| + pose & velocity anchor on `i` | 24 | 24/30 | **6** |
| + pose, vel on `i` & position on `j` | 27 | 27/30 | 3 |
| + pose, vel, **bias prior** on `i` | 30 | 30/30 | 0 |

Pose+velocity anchoring leaves **exactly the 6 bias directions unobservable**. A bias prior restores
rank but makes the estimate prior-dominated — a vacuous test of a bias-estimating formulation.

**3 keyframes / 2 edges with distinct motion fixes it:**

| configuration | residuals | rank | nullity |
| --- | --- | --- | --- |
| pose+vel anchor on KF0 only | 39 | 39/45 | 6 |
| **+ GNSS positions on KF1, KF2** | 45 | **45/45** | **0** |

All six biases observable with **no bias prior**. Condition number `1.8e4`.

**So M3/M4 use a 3-KF / 2-edge / GNSS-anchored instance.** The 2-KF case is kept only as a
degenerate smoke test with its rank deficiency documented.


---

## Findings recorded in the session plan, added here so agents can cite them (2026-10-13)

F5, F6, F10 and F11 were established during pre-flight but recorded only in the session plan. They are reproduced verbatim below. F1 (the working directory was not a repository) was resolved by cloning this repository and has no technical content.

## F5. Toolchain gaps — **resolved, both fixes are plain wheel installs.**
`numpy 2.2.3`, `scipy 1.15.2`, `sympy 1.14.0`, `pytest 9.0.2`, `matplotlib`, `pandas` present.
`cvxpy`, `mosek`, `scs`, `clarabel` **all MISSING** — currently no way to solve or certify anything.

`gtsam 4.2` has **zero** symbols matching `Gal`. Cause identified: **the Galilean IMU factor first ships in
GTSAM 4.3.0 (released 2026-09-18)**; 4.2 predates it by ~3 years. Verified by file-existence probe across tags —
`Gal3.{h,cpp}` appears at 4.3a0, `Gal3ImuEKF` at 4.3a1, and `GalileanImuFactor.h` / `GalileanPreintegration.{h,cpp}`
**only at the 4.3.0 final tag**. It is fully Python-wrapped (`navigation.i`, `geometry.i`) and a
`cp311 manylinux2014_x86_64` wheel exists — an exact match for this interpreter, so **no compilation is needed**.

`cvxpy 1.9.3` requires `numpy>=2.0` and `python>=3.11` — this machine satisfies both. Clarabel and SCS arrive
automatically as cvxpy dependencies.

## F6. GTSAM 4.3.0 uses our exact convention, but a **different endpoint relation**.
The GTSAM docs state `X = [[R,v,p],[0,1,t],[0,0,1]]` character-for-character, with the same product and inverse —
so F2/F4 transfer directly. But GTSAM does **not** use `X_j = H_ij X_i D_ij` at all. It uses

```
X_j = W_ij . phi_{Dt}(X_i) . U_ij(beta_i)
W_ij = NavState(I, 0.5 g Dt^2, g Dt)      # gravity, navigation frame, on the LEFT
phi_{Dt}(R,p,v) = (R, p + v Dt, v)        # affine "coasting" map, NOT a group element
U_ij = NavState(dR, dp, dv)               # bias-corrected IMU motion, body frame, on the RIGHT
```

Keyframes stay in `SE_2(3)`/`NavState`; **Gal(3) is internal to preintegration only**, and the time coordinate is
projected out losslessly at the factor boundary by `P_N` (lossless because no noise enters the clock row).

This sidesteps the F2 time-block problem entirely by never forming a Galilean endpoint product. **Design decision
for M1: implement both** — our corrected pure-group relation (F2) *and* GTSAM's `W.phi.U` form — and unit-test
them against each other. They must agree on `(R, v, p)`; if they disagree, one of them is wrong, and that is
exactly the kind of silent convention bug this project cannot afford.

**Three ordering footguns in GTSAM 4.3.0**, all verified in source, all of which will cause silent wrong answers:
1. Matrix column order is `v` before `p`.
2. C++ constructor order is `Gal3(R, r, v, t)` — **position before velocity**, the opposite of the matrix.
3. Tangent order is `(omega, nu, rho, alpha)` = (rot, **vel**, pos, time), while `NavState` tangent is (rot, **pos**, vel).
   GTSAM's own remap table `kGal3BlockForNavState[3] = {0, 6, 3}` is the ground truth here.
Additionally `imuBias::ConstantBias` is ordered **(accel, gyro)**, while this project's contract fixes `[b_g; b_a]`.

## F10. The real tightness risk is **graph sparsity**, not the Cayley singularity.
Barfoot et al. report that their relaxation stays rank-1 up to `sigma ~ 0.5-1.0 rad` for rotation averaging,
but that tightness **degrades with measurement sparsity** — their continuous-time sparse-measurement case had
"quite a bit lower" log-SVR and a failure even at `sigma = 0.01`. **An IMU preintegration chain is precisely a
sparse, chain-structured measurement graph.** This is the regime where their relaxation was weakest.

Corroborated independently by Rai et al.'s own ablation: primary constraints plus their A1 and B1 families give
a relative optimality gap of **at least 0.88 with 0/504 solves certified**. Only adding **B2** — the Gram identity
`(Y')^T Y' = ||w||^2 I - w w^T` on the rotation-bias wedge lifting — collapses the gap to a median `2.8e-5`.
Their single real-world failure (rank 3) was a window with a prolonged stationary segment and unobservable heading.

**Three consequences baked into the plan:**
1. **Redundant constraints are not optional polish; they are the whole ballgame.** Budget M3/M4 accordingly, and
   expect to need Galilean analogues of B2, not just a port of A1/B1.
2. Use **AutoTight** (Dumbgen, Holmes, Agro, Barfoot, T-RO 2024) — sample feasible points, build the monomial
   data matrix, take its **nullspace** — rather than only hand-deriving identities. Rai explicitly follows it.
3. **Excitation and observability are first-class experiment variables.** F9 already showed bias observability is
   fragile; Rai's failure case and Barfoot's sparsity finding say tightness is fragile in the same direction.

## F11. Solver accuracy is a correctness issue, not a performance one.
Barfoot et al. declare rank-1 at **log-SVR >= 5**, i.e. `sigma_1/sigma_2 >= 1e5`, solved with MOSEK at
`cvx_precision high`. A first-order solver converged to `~1e-4` **cannot distinguish a genuine `sigma_2 ~ 1e-6`
from its own residual**. Therefore: **Clarabel (interior-point, `~1e-9` gaps) is the default backend; SCS is for
scaling sanity checks only and its duality gaps and SVRs must never be quoted as certificates.** SDPA gives a
second, independent interior-point implementation for cross-checking marginal certificates.

Per the user's decision, **MOSEK is not used.** Documented consequence: numbers will not be bit-comparable with
Rai's or Barfoot's published MOSEK results. Any comparison to their tables must say so explicitly.

---

## Execution plan — Stage B
