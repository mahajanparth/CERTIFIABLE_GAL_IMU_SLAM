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
