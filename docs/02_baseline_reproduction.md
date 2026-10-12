# M2 — Baseline reproduction specification

Three baselines, one shared synthetic trajectory. In the noise-free, zero-bias-error limit all
three must reproduce the same `(R_j, v_j, p_j)` — but **not to the same tolerance**, because they
do not share a discretization. That asymmetry is the main finding of this milestone and is pinned
by the claims in [`claims/M2.yaml`](claims/M2.yaml).

Everything numeric below was measured against the installed **gtsam 4.3.0** or against a
closed-form reference, not assumed. Where this document contradicts the milestone brief it was
given, the contradiction is flagged explicitly and the measurement is shown.

---

## 0. The shared trajectory  (fixture `M2_TRAJ_V1`)

**Design driver (pre-flight F9).** A 2-keyframe instance leaves exactly the 6 bias directions
unobservable; a bias prior restores rank but makes the test vacuous. So this is a **3-keyframe /
2-edge** instance with **distinct motion on each edge**, anchored by pose+velocity on KF0 and GNSS
position on KF1 and KF2, and **no bias prior**. Rank was re-measured for *this specific*
trajectory, not inherited from F9 — see §5.

**Motion model: piecewise-constant body twist.** On each edge the body angular rate `w_e` and the
body-frame specific force `f_e` are constant. This is chosen deliberately:

- the noise-free IMU stream is then literally constant per edge — `omega_k = w_e + b_g`,
  `a_k = f_e + b_a` — so the fixture is trivially reproducible with no RNG at all;
- the true increment has a **closed form** (§0.3), so "ground truth" is exact, not a finer
  numerical integration;
- both `w_e` and `f_e` are nonzero and differ between edges, so both bias blocks are excited and
  the two edges are not redundant.

### 0.1 Constants

| symbol | value | units |
| --- | --- | --- |
| `g_W` | `[0, 0, -9.81]` | m/s^2 (z-up; this is the gravity **vector**, i.e. GTSAM `n_gravity`) |
| `dt` | `2.5e-3` (400 Hz) | s |
| `T_edge` | `0.5` | s |
| `n_steps` | `200` per edge | — |
| `n_edges` | `2` (KF0 -> KF1 -> KF2) | — |
| `b_g` (nominal, true) | `[0.0100, -0.0150, 0.0080]` | rad/s |
| `b_a` (nominal, true) | `[0.0600, -0.0400, 0.0900]` | m/s^2 |
| seed | **none required** — the fixture is deterministic by construction | — |

The project state convention fixes the bias as `b = [b_g; b_a]`. GTSAM's `imuBias::ConstantBias`
is the other way round (§3.4); the fixture stores `[b_g; b_a]` and converts at the GTSAM boundary.

> There is no RNG in the noise-free fixture. A seed is only needed for the noisy ablation that
> M2 does **not** cover; when one is added it must be a named constant, not a default.

### 0.2 Per-edge inputs and initial state

| edge | `w_e` (rad/s) | `f_e` (m/s^2, body frame) |
| --- | --- | --- |
| 0 (KF0 -> KF1) | `[ 0.35, -0.22,  0.17]` | `[ 0.60, 0.45, 9.95]` |
| 1 (KF1 -> KF2) | `[-0.28,  0.41, -0.33]` | `[-0.75, 0.90, 9.60]` |

Initial state: `R_0 = Exp([0.10, -0.20, 0.30])`, `p_0 = [0,0,0]`, `v_0 = [0.50, -0.30, 0.20]`.

Noise-free measurements on edge `e`, for every `k`: `omega_k = w_e + b_g`, `a_k = f_e + b_a`.

### 0.3 Closed-form ground truth

For constant `w` and constant body-frame specific force `a` over `[0, T]`, with
`theta = ||w||`, `K = w^/theta`:

```
Exp(w s) = I + sin(theta s) K + (1 - cos(theta s)) K^2

M1(T) = int_0^T Exp(w s) ds       = T I + ((1 - cos(theta T))/theta) K + (T - sin(theta T)/theta) K^2
M2(T) = int_0^T M1(s) ds          = (T^2/2) I + ((T - sin(theta T)/theta)/theta) K
                                     + (T^2/2 - (1 - cos(theta T))/theta^2) K^2

dR = Exp(w T)      dv = M1(T) a      dp = M2(T) a
```

Verified against adaptive quadrature: `|M1 - quad| = 7.6e-17`, `|M2 - quad| = 1.9e-16`.

Keyframes are then chained with the exact-gravity endpoint relation of §1.1.

### 0.4 Reference values (exact, 15 significant digits)

```
dR_01 = [[ 9.903776532493884e-01, -9.387914277777526e-02, -1.016799414611560e-01],
         [ 7.470913760838102e-02,  9.811536442685306e-01, -1.782023319050390e-01],
         [ 1.164931272738697e-01,  1.688911865370475e-01,  9.787262734840944e-01]]
dv_01 = [ 2.860848012625113e-02, -2.059158118715534e-01,  4.976091490259237e+00]
dp_01 = [ 2.924832251685720e-02, -1.526611626991823e-02,  1.245394361998341e+00]

dR_12 = [[ 9.656306111940006e-01,  1.483233923728816e-01,  2.134424840562463e-01],
         [-1.768115211449302e-01,  9.767603374607809e-01,  1.211483766045474e-01],
         [-1.905130145567926e-01, -1.547236712287475e-01,  9.694149662791377e-01]]
dv_12 = [ 1.707829111988740e-01,  7.855781189893493e-01,  4.753841859545298e+00]
dp_12 = [-3.379126843155931e-03,  1.697985087871268e-01,  1.194510739753956e+00]

KF1:  v = [-3.092347809743656e-01, -1.121267155202117e+00,  1.401383484057339e-01]
      p = [ 5.715027067778627e-02, -3.148113358346768e-01,  9.348023290152563e-02]
KF2:  v = [-1.520564406279614e+00, -1.903478183400628e+00, -1.641944548085670e-01]
      p = [-4.312534482841444e-01, -1.111986889144151e+00,  7.234717707859262e-02]
      R = [[ 9.677181225278726e-01, -2.412143599148208e-01, -7.305660751781723e-02],
           [ 2.301509945763355e-01,  9.638954740062700e-01, -1.339254825854730e-01],
           [ 1.027236828908148e-01,  1.127880656856546e-01,  9.882948432589516e-01]]
```

---

## 1. Baseline 1 — conventional Forster-style preintegration

Forster, Carlone, Dellaert, Scaramuzza, *On-Manifold Preintegration for Real-Time Visual-Inertial
Odometry*, IEEE T-RO 33(1), 2017.

### 1.1 Endpoint relation (component form)

```
R_j = R_i dR_ij
v_j = v_i + g_W Dt_ij + R_i dv_ij
p_j = p_i + v_i Dt_ij + 0.5 g_W Dt_ij^2 + R_i dp_ij
```

Measured against `gtsam.PreintegratedImuMeasurementsG.predict(...)`: agreement `4.6e-15 / 0 / 0`
(R / v / p). This is the shared contract all three baselines are compared through.

### 1.2 Increment recursion

Rai Eq. (9), which is the standard Forster discretization:

```
dR_ij = prod_{k=i}^{j-1} Exp((omega_k - b_g) dt)
dv_ij = sum_{k=i}^{j-1} dR_ik (a_k - b_a) dt
dp_ij = sum_{k=i}^{j-1} [ dv_ik dt + 0.5 dR_ik (a_k - b_a) dt^2 ]
```

**This is first order in `dt`.** On the fixture (measured, edge 0, `dt = 2.5e-3`):

| quantity | error vs exact |
| --- | --- |
| `dR` | `1.2e-14` — *exact*, because for constant `w` the product telescopes to `Exp(w T)` |
| `dv` | `2.17e-3` |
| `dp` | `5.38e-4` |

and the error halves exactly when `dt` halves (ratio measured `0.5003, 0.5001, 0.5001, 0.5000`).
Do not mistake the exact `dR` for a general property — it is an artifact of the
piecewise-constant-twist design and holds only per edge.

### 1.3 Bias correction (first-order, frozen Jacobians)

```
dR_ij(b) ~= dR_ij(bbar) Exp(J^R_ij  d b_g)
dv_ij(b) ~= dv_ij(bbar) + J^{vg}_ij d b_g + J^{va}_ij d b_a
dp_ij(b) ~= dp_ij(bbar) + J^{pg}_ij d b_g + J^{pa}_ij d b_a        d b = b - bbar
```

Jacobians are accumulated by the usual recursion during preintegration and are **frozen at
`bbar`**. Note the chart asymmetry that is intrinsic to Forster: the rotation correction is a
*right group composition*, the velocity and position corrections are *additive*. Baselines 2 and 3
each resolve this asymmetry differently, and that is precisely what distinguishes them.

GTSAM does not expose `delRdelBiasOmega` / `delVdelBias*` / `delPdelBias*` on either
`PreintegratedImuMeasurements` or `PreintegratedImuMeasurementsG` in the Python wrapper (checked:
the two classes expose an identical member set, and it contains none of them). The implementer
must obtain reference Jacobians by finite differences on re-integration, or compute them in the
baseline's own code.

### 1.4 Covariance propagation

Linear propagation `Sigma_{k+1} = A_k Sigma_k A_k^T + B_k Sigma_eta B_k^T` over the 9-dim
`(dphi, dv, dp)` error state, with `Sigma_eta = diag(sigma_g^2 I, sigma_a^2 I)/dt`. The checkable
properties are symmetry, positive semidefiniteness, and monotone growth of `tr(Sigma)` under
noise-free integration with nonzero noise densities.

GTSAM's `preintMeasCov()` (9x9) is the cross-reference, and **its block order is NavState order
`(rot, pos, vel)`, not `(rot, vel, pos)`** — compare like with like before asserting agreement.
Measured by injecting an anisotropic noise budget (`sigma_a^2 = 1e-2`, `sigma_g^2 = 1e-8`,
`Dt = 1`): the three diagonal blocks come out `3.0e-8`, `3.33e-3`, `1.0e-2`. The middle block
carries `sigma_a^2 Dt^3/3 = 3.33e-3` (position) and the last carries `sigma_a^2 Dt = 1e-2`
(velocity) — ratio measured `0.33333`, i.e. `Dt^2/3`. Position is in the middle. The same matrix
is symmetric to `2.2e-20` with minimum eigenvalue `1.0e-8 > 0`.

---

## 2. Baseline 2 — Rai-style Cayley factor (reimplementation from the paper)

Rai, Xu, Chen, Rosen, *A QCQP-Representable IMU Pre-Integration Factor for Certifiable State
Estimation*, arXiv:2609.38048v1 [cs.RO], submitted 29 Sep 2026 (Robust Autonomy Lab, Northeastern).

> **No public code exists.** Re-confirmed for this milestone: the arXiv landing page carries no
> GitHub / GitLab / project-page link, only arXiv-generic third-party links. Every claim in this
> section is tagged `from-paper reimplementation` and must stay that way. Nothing here has been
> checked against an author-provided implementation, because there is none.

### 2.1 Cayley map, Eq. (3) and Eq. (4)

```
Cay(phi)   = (I - phi^/2)^-1 (I + phi^/2),      phi = 2 tan(theta/2) a         (3)
J_Cay(phi) = (I - phi^/2) / (1 + ||phi||^2/4)                                  (4)
```

Two things measured, both of which the implementer needs:

1. **The parameterization in Eq. (3) matters.** `Cay(2 tan(theta/2) a) = Exp(theta a)` to `1.6e-15`
   over 2000 random axes with `theta < 3`. Feeding a raw tangent `w dt` to `Cay` instead is a
   *different* retraction, agreeing only to `O(||w dt||^3)` per step. On the fixture this is the
   whole difference between the two admissible readings of "replace Exp with Cay": it moves `dR`
   by `3.0e-8` and `v` at KF2 by `6.2e-8`. **The spec fixes the literal reading** —
   `phi_k = (omega_k - bbar_g) dt` substituted directly into `Cay` — because that is what the
   paper's sentence says, and records the gap as a known, bounded ambiguity.
2. **Eq. (4) is the RIGHT Jacobian**: `Cay(phi + d) = Cay(phi) Exp(J_Cay(phi) d) + O(||d||^2)`.
   Measured `5.3e-7` at a finite-difference step of `1e-7` (i.e. FD-limited). The left convention
   is wrong by `2.1e0` — not a near miss.

### 2.2 Invertibility, Eq. (17)

```
det(I - xi^/2) = 1 + ||xi||^2/4 > 0                                            (17)
```

Measured to `7.1e-14` over 5000 random `xi` with `||xi||` up to 5. The denominator of the Cayley
map is therefore never singular and the retraction can be cleared algebraically with no case
analysis. Pre-flight F8 independently measured the exact `gal(3)` analogue
(`det(I5 - A/2) = 1 + ||phi||^2/4`, max deviation `9.1e-13`).

### 2.3 The polynomial rotation residual, Eq. (15)–(19)

```
R_i^T R_j = dRbar_ij Cay(xi_ij) Cay(-dphi_ij),     xi_ij = J^R_ij d b_g        (15)
R_j = R_i dRbar_ij (I - xi^/2)^-1 (I + xi^/2)                                  (16)
R_j (I - xi_ij^/2) = R_i dRbar_ij (I + xi_ij^/2)                               (18)
r^R_ij = (R_j - R_i dRbar_ij) - 0.5 (R_j + R_i dRbar_ij) xi_ij^                (19)
```

Eq. (18) is Eq. (16) with the denominator cleared — legitimate precisely because of Eq. (17) —
and Eq. (19) is Eq. (18) moved to one side. **Eq. (19) is exactly equivalent to Eq. (16); nothing
is truncated.** That is the point: the residual is a *polynomial* of degree <= 2 in
`(R_i, R_j, b_g)`, hence QCQP-representable, while still encoding the Cayley retraction exactly.

Eq. (21a) restates Eq. (19) inside the factor definition.

### 2.4 Velocity and position residuals, Eq. (21b)–(21c)

```
r^v_ij = R_i^T (v_j - v_i - g_W Dt_ij) - dvbar_ij - A^{vg}_ij d b_g - A^{va}_ij d b_a      (21b)
r^p_ij = R_i^T (p_j - p_i - v_i Dt_ij - 0.5 g_W Dt_ij^2) - dpbar_ij
         - A^{pg}_ij d b_g - A^{pa}_ij d b_a                                               (21c)
```

These are **first order in `d b`** and additive — unlike the rotation residual, which is exact.
The asymmetry is real and inherited from Forster; it is not an error in the transcription.

### 2.5 Affine reduction, Eq. (22)

```
Gamma_ij = (J^R_ij bbar_g)^,    Rbreve_ij = dRbar_ij (I - Gamma_ij/2)                      (22a)
r^R_ij   = R_j (I + Gamma_ij/2) - R_i Rbreve_ij - 0.5 (Y_ij + Y'_ij)                       (22b)
Y_ij     = R_i dRbar_ij (J^R_ij b_g)^,    Y'_ij = R_j (J^R_ij b_g)^                        (22c)
```

Paper: *"We introduce `Y_ij` and `Y'_ij` as auxiliary variables, making `r^R_ij` affine."*

Eq. (22b) is an **exact algebraic identity** with Eq. (19), not an approximation — verified to
`4.4e-16` over 2000 random instantiations, substituting `xi = J^R(b_g - bbar_g)`. This is a cheap,
sharp test of a correct transcription and it is claimed as such.

### 2.6 Isotropic precisions, Eq. (23) — structurally forced

```
tau_R^-1 = (1/3) tr Sigma^phi_ij,   tau_v^-1 = (1/3) tr Sigma^v_ij,   tau_p^-1 = (1/3) tr Sigma^p_ij   (23)
```

The paper is explicit that this is **not** a convenience. Eq. (20) weights the residuals as
`tau_R ||r^R||_F^2 + tau_v ||r^v||^2 + tau_p ||r^p||^2`, and the stated reason is:

> *"This equality requires an isotropic weight. We therefore approximate the three covariance
> blocks by scalar precisions."*
> *"Premultiplying the translation residuals by `R_i` removes the rotation-state products."*

The identity being relied on is `||r||^2 = ||R_i r||^2` for `R_i` in `SO(3)`, which lets `R_i^T`
be moved out of Eq. (21b)/(21c) so the residual becomes affine in the state and the cost stays
quadratic. Measured: the identity holds to `7.1e-15` for the isotropic weight, and **fails by
`O(1)` (measured `28.9`) for a non-isotropic `W = diag(1,2,5)`**. So a full-covariance weight
destroys QCQP-representability. The paper also acknowledges the price:

> *"This approximation drops the cross-correlations and directional weighting in the original
> covariance."*

**Correction to the brief:** the brief attributes the forcing to the SDP structure. The paper does
not say that. The forcing is the `SO(3)`-invariance of the 2-norm under the `R_i` premultiplication
that removes rotation-state products. The conclusion ("structurally forced, not a convenience") is
right; the stated mechanism needed fixing.

### 2.7 The heart of the contribution: frozen Jacobians, exact retraction

Two independent choices that are easy to conflate:

| | what is done | consequence |
| --- | --- | --- |
| **Preintegration / bias Jacobians** | computed **once** at the nominal bias `bbar`, **never re-linearized**; `J^R_ij`, `A^{vg}`, `A^{va}`, `A^{pg}`, `A^{pa}` are **frozen** | the residual is *first order* in `d b` |
| **Cayley retraction of the correction** | kept **exactly** — `(I - xi^/2)^-1 (I + xi^/2)` with the denominator **cleared algebraically** via Eq. (17)/(18), **not** truncated | `r^R` is exactly equivalent to Eq. (16), and is polynomial of degree <= 2 |

Paper, on the first: *"Pre-integration is performed at the nominal bias"*, *"Recomputing the
pre-integration for every value of b would be expensive, so the change is approximated to first
order"*, *"All Jacobians are evaluated at the nominal bias."*

So: **the map `d b_g -> xi_ij` is linear and frozen; the map `xi_ij -> rotation` is exact.**
A reimplementation that truncates `Cay(xi) ~= I + xi^` would be wrong in a way that is invisible
at small `d b_g` and quietly destroys the exactness that the whole construction exists to provide.

### 2.8 Could not extract

- No explicit closed form for `J^R_ij`; the paper says only *"These Jacobians are computed
  recursively during pre-integration."* The reimplementation uses the Forster recursion with `Cay`
  and `J_Cay` in place of `Exp` and `J_r`, and this is tagged `ASSUMPTION`.
- No explicit formulas for the Cayley-flavoured `dRbar_ij`, `dvbar_ij`, `dpbar_ij`; the paper
  defers to Eq. (9) with *"the pre-integration derivation in Sec. III-A carries over after
  replacing Exp and its right Jacobian with Cay and the Cayley right Jacobian in (4)."*
- No explicit formulas for `A^{vg}`, `A^{va}`, `A^{pg}`, `A^{pa}` beyond their role in (21b)/(21c).
- Whether `Cay` receives the raw tangent or the Eq. (3) reparameterization — see §2.1.

---

## 3. Baseline 3 — GTSAM 4.3.0 direct-product `Gal(3) x R^6`

This one is real, installed, and runnable. Everything below was measured on
`gtsam 4.3.0` at
`.venv/lib/python3.11/site-packages/gtsam/`. The relevant internals are
`gtsam::GalileanPreintegration` (and `gtsam::TangentPreintegration` for contrast), confirmed
present in the compiled extension.

### 3.1 `Gal3` matrix layout — **confirmed**

```
X = [[R, v, r],
     [0, 1, t],
     [0, 0, 1]]
```

`Gal3(R, [1,2,3], [10,20,30], 7.0).matrix()` gives column 3 `= [10,20,30]` (velocity) and column 4
`= [1,2,3]` (position), `M[3,4] = 7.0`. **Velocity comes before position in the matrix.**

### 3.2 Footgun 1 — constructor order is the *opposite* of the matrix — **confirmed**

```
Gal3(R: Rot3, r: Vector3, v: Vector3, t: double)      # POSITION third-from-last, VELOCITY second
```

So `Gal3(R, r, v, t)` writes `v` into the earlier matrix column and `r` into the later one. The
accessors are unambiguous (`position()`, `velocity()`, `time()`, and `translation()` is an alias
for `position()`) and should be used in preference to positional construction wherever possible.

### 3.3 Footgun 2 — tangent order `(omega, nu, rho, alpha)` — **confirmed**

Measured from `Gal3.Hat(e_k)` and `Gal3.Expmap(1e-6 e_k)` for each basis index:

| tangent index | block | meaning |
| --- | --- | --- |
| `0:3` | rotation block | `omega` (rotation) |
| `3:6` | matrix column 3 | `nu` (**velocity**) |
| `6:9` | matrix column 4 | `rho` (**position**) |
| `9` | `M[3,4]` | `alpha` (time) |

`NavState` tangent is `(rot, pos, vel)`, so the block remap is
`kGal3BlockForNavState = {0, 6, 3}`: NavState block 0 (rot) <- Gal3 offset 0, NavState block 1
(pos) <- Gal3 offset 6, NavState block 2 (vel) <- Gal3 offset 3. The compiled extension does not
export this symbol to Python; the table is reconstructed from the measured block semantics above
and is consistent with it.

> **This is a block-index permutation and nothing more.** It does **not** make the two charts
> agree. Measured on the fixture: `Gal3.Logmap(X_i^-1 X_j)` reordered by `{0,6,3}` differs from
> `NavState(R_i,p_i,v_i).localCoordinates(NavState(R_j,p_j,v_j))` by **`0.763`** — `NavState`'s
> retraction is not the `Gal(3)` exponential. Conflating the two is the natural next mistake after
> getting the remap right, and it is not a small one.

Also measured: `Gal3.Expmap(xi) == expm(Gal3.Hat(xi))` to `5.6e-17`, and `Logmap . Expmap = id` to
`5.6e-17`. The Galilean exponential is the true matrix exponential of the hat.

### 3.4 Footgun 3 — bias ordering — **confirmed**

```
imuBias.ConstantBias(biasAcc: Vector3, biasGyro: Vector3)     # ACCEL FIRST
b.vector()         ->  [b_a; b_g]
pim.biasHatVector() ->  [b_a; b_g]
```

The project fixes `b = [b_g; b_a]`. **Every crossing of the GTSAM boundary must swap.** The bias
Jacobian columns returned by any GTSAM-facing code are likewise in `(accel, gyro)` column order.

### 3.5 Endpoint relation — **confirmed, with a domain restriction the brief omits**

The brief states `X_j = W_ij . phi_Dt(X_i) . U_ij` (not `H X_i D`), and pre-flight F2 gives the
corrected form `X_j = T(-Dt) . H' . X_i . D_ij` with `H' = [[I, g Dt, -0.5 g Dt^2],[0,1,0],[0,0,1]]`.
Both were re-measured. With `U_ij = Gal3(dR, dp, dv, Dt)`:

| form | `t_i = 0` | `t_i = 3.7` |
| --- | --- | --- |
| `T(-Dt) . H' . X_i . U_ij` (F2) | `p` error `8.9e-16` | `p` error **`3.63e+01`** |
| `W(t_i) . X_i . U_ij` | `p` error `8.9e-16` | `p` error `8.9e-16` |

**The F2 relation is only valid when the time coordinate of `X_i` is zero.** At `t_i != 0` it is
wrong by exactly `g Dt t_i` (`9.81 * 1.0 * 3.7 = 36.297`, measured `36.3`), because the group
product's `v1 t2` term couples the gravity boost to the absolute clock. The general form is

```
X_j = W_ij(t_i) . X_i . U_ij ,      W_ij(t_i) = Gal3( I, 0.5 g Dt^2 - g Dt (t_i + Dt), g Dt, 0 )
```

which yields `t_j = t_i + Dt` (the physically correct clock) rather than F2's `t_j = t_i`.

This restriction is benign *for GTSAM* — `NavState` carries no clock, so the lift always has
`t = 0` — but it is a live hazard for M3/M4, where `Gal(3)` states are the decision variables and
there is nothing stopping a nonzero time coordinate. Both forms are claimed, each with its domain.

The plain component relation of §1.1 holds unconditionally and is the safer contract.

Also measured: `H' . T(-Dt) . X_i . U` (same factors, swapped order) is wrong by `9.81`. The order
is load-bearing.

### 3.6 The increment is **exact** zero-order-hold — confirmed, and it is a real difference

Against the closed-form continuous-time integral of the same constant `(w, a)`:

| backend | `dR` | `dv` | `dp` |
| --- | --- | --- | --- |
| `PreintegratedImuMeasurementsG` (Galilean) | `1.0e-14` | `2.0e-14` | `1.8e-15` |
| `PreintegratedImuMeasurements` (Tangent) | `5.6e-17` | **`7.4e-3`** | **`3.7e-3`** |

On the fixture the Galilean increment reproduces the §0.4 reference to `4e-14` on both edges.
**Baseline 3 is not an approximation of the ground truth on this trajectory — it is the ground
truth**, to machine precision. That is what forces the two-tier tolerance of §4.

Accumulation is **right** composition, `U_{k+1} = U_k . V_k`, measured exactly `0.0` on a
time-varying measurement stream; left composition `V_k . U_k` is wrong by `4.3e-4` after 40 steps.
(A constant-input stream cannot distinguish the two — the increments commute — so this must be
tested with time-varying inputs. The fixture's per-edge inputs are constant, so this particular
claim uses its own time-varying stream.)

### 3.7 Bias correction: first-order Jacobian, composed **exactly on the right** — confirmed

Write `Uhat` for the increment at `bbar` and `U(d b)` for `predict()`'s corrected increment. Three
candidate models, tested over 30 random directions at each magnitude:

| `||d b||` | A: `U = Uhat . Exp_Gal3(J d b)` | B: component-wise first order | Tangent: `zeta = zetahat + J_T d b` |
| --- | --- | --- | --- |
| `1e-4` | `2.2e-12` | `2.8e-4` | `8.4e-13` |
| `1e-3` | `2.2e-11` | `3.2e-3` | `8.5e-12` |
| `1e-2` | `2.2e-10` | `2.8e-2` | `8.8e-11` |
| `5e-2` | `1.2e-9`  | **`1.3e-1`** | `3.7e-10` |

Model A's residual scales **linearly** with `||d b||` at a constant relative level of `2.2e-8` —
that is the error in the finite-difference estimate of `J`, not model error. **Model A is exact.**
Component-wise addition (B) is wrong by `0.13` at `||d b|| = 0.05`: this is not a subtle
distinction.

So, precisely:

```
U(d b) = Uhat . Expmap_Gal3( J d b )        J frozen at bbar,  J in R^{10x6}
```

The *Jacobian* is first order. The *retraction* is exact — `Expmap`, composed on the right in
`Gal(3)`, not a truncated series and not an addition in a component chart.

**Refinement of the brief.** The brief says this "differs from `TangentPreintegration`", implying
`TangentPreintegration` is only approximate. Measured: `TangentPreintegration` is **also** exactly
affine — but in *its own* chart, `zeta = (Log dR, dp, dv)`, with `zeta(d b) = zetahat + J_T d b`
exact to FD noise. The real difference is **which chart the frozen Jacobian is exact in**, and it
is a genuine difference: applying the Galilean `J` additively in components (model B) is off by
`0.13`. Both are first-order-in-`d b` approximations of re-integration; against re-integration the
Galilean residual is `O(||d b||^2)` with coefficient `~0.67` (measured `6.7e-9 / 6.7e-7 / 6.7e-5`
at `||d b|| = 1e-4 / 1e-3 / 1e-2`), about 2.8x smaller than Tangent's.

Structural zeros, measured: `J[0:3, 0:3] = 0` (rotation is independent of accelerometer bias, max
`3.4e-12`) and `J[9, :] = 0` exactly (the time coordinate is independent of bias).

### 3.8 Bias Jacobian recursion — confirmed

```
J_{k+1} = Ad_{V_k^-1} J_k + C_k ,       J_i = 0
```

where `V_k` is the single-step `Gal(3)` increment and `C_k` is that step's own right-chart bias
Jacobian, `C_k = d/d(d b) Log( Vhat_k^-1 V_k(d b) )`. Derivation: from `U_{k+1} = U_k V_k`,

```
U_{k+1}(db) = Uhat_k Exp(J_k db) Vhat_k Exp(C_k db)
            = Uhat_{k+1} Exp(Ad_{Vhat_k^-1} J_k db) Exp(C_k db)
            = Uhat_{k+1} Exp( (Ad_{Vhat_k^-1} J_k + C_k) db ) + O(||db||^2)
```

Measured against the true (re-integration) Jacobian on a time-varying stream: `6.9e-9` after 40
steps against `|J| ~ 0.21`, i.e. ~3e-8 relative. `Gal3.AdjointMap()` supplies `Ad`.

**Sign note.** The brief writes `J_{k+1} = Ad_{V_k^-1} J_k - C_k`. That is the same recursion under
the opposite sign convention for `C_k` (the brief's `C_k` is the Jacobian w.r.t. the *measurement*
rather than the *bias*, and `a_k - b_a` flips the sign). The form claimed here is `+C_k` with
`C_k` defined above, and the definition is given explicitly so the sign cannot float.

---

## 4. Cross-baseline agreement — the two-tier tolerance

In the noise-free, zero-bias-error limit the three baselines **do not** all agree to machine
precision, and a single tolerance would be dishonest. Measured at KF2 on the fixture
(`dt = 2.5e-3`), `|.|_inf`:

| pair | `dR` | `dv` | `dp` |
| --- | --- | --- | --- |
| B1 (Forster) vs B2 (Cayley, Eq.(3) reparameterized) | `2.3e-14` | `1.5e-13` | `5.8e-14` |
| B1 vs B2a (Cayley, raw `w dt`) | `3.0e-8` | `6.2e-8` | `4.4e-8` |
| B3 (GTSAM Galilean) vs exact | `1.7e-14` | `3.9e-14` | `1.7e-14` |
| **B1 / B2 vs B3** | `2.1e-14` | **`1.29e-3`** | **`1.20e-3`** |

### Tier 1 — same discretization: `1e-6`

B1 and B2 share Rai Eq. (9) exactly and differ *only* in the rotation retraction. Measured gap
`6.2e-8` (worst case, raw-`w dt` reading); tolerance `1e-6` gives ~16x margin and still catches any
transcription error. Rotations agree to `1e-10`.

### Tier 2 — different discretization: `5e-3`, **plus a convergence-order test**

B1/B2 use the first-order Euler sums; B3 is exact ZOH. The gap is *not* a bug in either — it is
`O(dt)` truncation, and a fixed tolerance alone would not distinguish "correctly first-order" from
"wrong but small". So the claim is stated as a pair:

1. **magnitude**: gap `<= 5e-3` at `dt = 2.5e-3` (measured max `1.29e-3`, ~3.9x margin);
2. **order**: halving `dt` halves the gap — `gap(dt/2) / gap(dt)` in `[0.45, 0.55]`.

Measured ratios at KF2 across `dt = 5e-3 -> 3.125e-4`:

| `dt` | `gap_v` | `gap_p` | ratio_v | ratio_p |
| --- | --- | --- | --- | --- |
| `5.000e-3` | `2.572e-3` | `2.406e-3` | — | — |
| `2.500e-3` | `1.287e-3` | `1.203e-3` | `0.5003` | `0.5000` |
| `1.250e-3` | `6.435e-4` | `6.015e-4` | `0.5001` | `0.5000` |
| `6.250e-4` | `3.218e-4` | `3.008e-4` | `0.5001` | `0.5000` |
| `3.125e-4` | `1.609e-4` | `1.504e-4` | `0.5000` | `0.5000` |

The order test is the one that actually has teeth. The magnitude test alone would pass for a
baseline that is wrong by a constant `1e-3`.

### Tier 0 — against exact ground truth: `1e-10`

B3 must reproduce §0.4 to `1e-10` (measured `4e-14`). B1 and B2 must reproduce the *rotations* to
`1e-10` (measured `2.4e-14`; exact for this piecewise-constant-twist design) but only the Tier-2
tolerance in `v` and `p`.

---

## 5. Observability of the fixture — re-measured, not inherited

F9 established that 3 KF / 2 edges with GNSS anchoring gives full rank. That was a *different*
trajectory. Re-measured on **this** fixture, with per-keyframe bias (state dim
`3 x (6 pose + 3 vel + 6 bias) = 45`), two `GalileanCombinedImuFactor` edges, and **no bias prior**:

| configuration | residuals | rank | nullity |
| --- | --- | --- | --- |
| IMU edges alone | 30 | 30/45 | 15 |
| + pose & velocity prior on KF0 | 39 | 39/45 | 6 |
| **+ GNSS position on KF1 and KF2** | 45 | **45/45** | **0** |

All six bias directions observable with no bias prior, reproducing the F9 pattern on the new
trajectory. Total graph error at ground truth: `7.5e-23`; per-factor unwhitened residual `<= 2.2e-14`.

The condition number is **not** claimed: it is set by the noise model, not the geometry
(measured `7.5e7` with tight covariances and `7.6e6` with realistic MEMS densities). Rank is
invariant to positive reweighting and is what the claim pins.

---

## 6. Summary of corrections to the milestone brief

| brief said | measured |
| --- | --- |
| `Gal3` matrix `[[R,v,p],[0,1,t],[0,0,1]]` | **confirmed** |
| constructor `Gal3(R, r, v, t)`, position before velocity | **confirmed** |
| tangent `(omega, nu, rho, alpha)`; remap `{0,6,3}` | **confirmed** (symbol not exported to Python; reconstructed from measured block semantics) |
| `ConstantBias` is `(accel, gyro)` | **confirmed** |
| increment is exact ZOH | **confirmed** (`2e-14` vs Tangent's `7.4e-3`) |
| bias correction composed exactly on the right | **confirmed** (`Uhat . Exp(J db)`, exact) |
| "...differs from `TangentPreintegration`" | **refined** — Tangent is *also* exactly affine, in its own chart; the difference is *which chart*, and it is still a real `0.13` difference |
| `J_{k+1} = Ad_{V_k^-1} J_k - C_k` | **confirmed up to the sign convention of `C_k`**; claimed as `+C_k` with `C_k` defined explicitly |
| Rai Eq. (18)/(19) is the polynomial residual | **refined** — (18) is the cleared-denominator *equality*, (19) is the residual (restated at 21a) |
| Rai Eq. (23) isotropy is "structurally forced" | **confirmed**, but the mechanism is `SO(3)`-invariance of the 2-norm under the `R_i` premultiplication, **not** the SDP structure; the paper does not claim the latter |
| F2 relation `X_j = T(-Dt) H' X_i D` | **valid only at `t_i = 0`**; wrong by `g Dt t_i` otherwise. General form given in §3.5 |
| Rai et al. have no public code | **re-confirmed** |
