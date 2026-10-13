# M3 — QCQP derivation for the direct-product Galilean IMU factor

**What this is.** A QCQP (`min z^T Q z  s.t.  z^T A_k z = b_k`) whose feasible set is exactly the
lifted image of the 3-keyframe GNSS–IMU smoothing state, and whose cost is a frozen-weight Cayley
surrogate of GTSAM 4.3.0's `GalileanImuFactor` graph. Certification is Rai et al.'s / Barfoot et
al.'s machinery; the Galilean bias handling (right-applied first-order correction in the `Gal(3)`
right chart) is GTSAM's. **The only thing done here is applying the former to the latter.** Nothing
in this document is solved as an SDP; that is M4.

Claims: [`claims/M3.yaml`](claims/M3.yaml). Every number below is printed by a named script in
`experiments/m3_scratch/` (run as `python -m experiments.m3_scratch.<name>`) or quoted with its
source equation number.

Sources (cached HTML under the M0 worktree's `tests/fixtures/`): Rai et al. arXiv:2609.38048v1
("Rai"), Barfoot et al. arXiv:2308.12418v2 ("Barfoot"), the GTSAM Galilean IMU factor guide
("GTSAM doc"). Pre-flight items F2, F3, F8, F9 are in `docs/00_preflight_findings.md`. The brief
also cites pre-flight F6, F10 and F11; **those identifiers do not exist in this worktree's
`00_preflight_findings.md`**. The content they were cited for (GTSAM's `W.phi.U` form and ordering
footguns; tightness risk; solver accuracy) is taken here from GTSAM doc Eq. (15)/(17), M2 §3, and
the CLAUDE.md guardrails.

---

## 0. Conventions and the one boundary

- `X(R,v,p,t) = [[R,v,p],[0,1,t],[0,0,1]]` (velocity column before position column).
- Bias `b = [b_g; b_a]`. `Gal(3)` tangent order `(omega, nu, rho, alpha)` = (rot, **vel**, pos, time).
- GTSAM `ConstantBias(acc, gyro)` and `NavState` (rot, **pos**, vel). **All conversions happen in
  `experiments/m3_scratch/instance.py`**: `_gt_bias` (project -> `ConstantBias`), the bias Jacobian
  `jacobian()` (finite differences in project order through `_gt_bias`), and `taus()` (reads the
  `preintMeasCov()` blocks in NavState order rot `[0:3]`, pos `[3:6]`, vel `[6:9]`). Nothing
  downstream sees a GTSAM-ordered object.
- `sk(a) = [a]_x`, `vec` is column-major (Rai Eq. (27a)/(27b)).

---

## (a) Test instance

Trajectory = M2 fixture `M2_TRAJ_V1` (`docs/02_baseline_reproduction.md` §0, M2 branch):
piecewise-constant body twist, 400 Hz, `T = 0.5 s` per edge, `g_W = [0,0,-9.81]`,
`R_0 = Exp([0.10,-0.20,0.30])`, `v_0 = [0.50,-0.30,0.20]`, `p_0 = 0`, true bias
`b = [0.0100,-0.0150,0.0080, 0.0600,-0.0400,0.0900]` (`[b_g; b_a]`), noise-free measurements.

| item | definition |
| --- | --- |
| keyframes | KF0, KF1, KF2; state `(R_k, v_k, p_k, b_k)`, **per-keyframe bias** |
| IMU edges | `(0,1)`, `(1,2)`; edge `(i,i+1)` uses `b_i` |
| priors | KF0 rotation, velocity, position (`sigma = 1e-3` each, so weight `1e6`) |
| GNSS | position of KF1 and KF2 (`sigma = 0.05 m`, weight `400`) |
| bias random walk | `b_1 - b_0`, `b_2 - b_1`, `Lambda_rw = diag(sigma_bg^-2 Dt^-1 I, sigma_ba^-2 Dt^-1 I)` (Rai, text after Eq. (39)), `sigma_bg = 1.9e-5`, `sigma_ba = 3.0e-3` |
| bias prior | **none** (pre-flight F9) |
| preintegration | `gtsam.PreintegratedImuMeasurementsG` at nominal `bbar` (two settings used: `bbar = b_true` and `bbar = 0`) |

`Dt = 0.5000000000000003` per edge (`instance.py`). Isotropic precisions by Rai Eq. (23) applied to
GTSAM's `preintMeasCov()`: edge 0 `tau_R = 6.92e7, tau_v = 4.80e5, tau_p = 5.89e6`; edge 1
`6.92e7, 4.82e5, 5.90e6` (`instance.py`).

**Observability, re-measured on this cost (not inherited).** The whitened residual Jacobian of the
M3 cost w.r.t. the 45-dim local state (per KF: `dtheta, dv, dp, db`) at ground truth has rank
**45/45** (condition `3.95e7` at `bbar = b_true`, `4.53e7` at `bbar = 0`; `check_cost.py`). The
condition number is set by the weights and is not claimed beyond being finite.

---

## (b) The lifted vector `z`

### b.1 The Galilean residual and why it needs more liftings than Rai's

For edge `(i,j)`, GTSAM's model is (GTSAM doc Eq. (36), (17)):

```
U_ij(b) ~= Ubar_ij Exp( J_ij (b - bbar) )            J_ij in R^{10x6}, frozen at bbar
X_j     =  W_ij phi_Dt(X_i) U_ij(b_i)                 (NavState form, Eq. (17))
```

In 5x5 form, with `X_i, X_j` lifted at time coordinate 0, this is pre-flight F2's corrected
relation `X_j = G_ij X_i U_ij`, `G_ij = T(-Dt) H' = X(I, g Dt, -1/2 g Dt^2, -Dt)` (valid only at
`t_i = 0`, M2 §3.5; **`z` contains no time variable**, so the restriction holds structurally).

**Cayley surrogate (the modelling choice of this work, following Rai Eq. (15)–(16)).** Replace
`Exp` by the `Gal(3)` Cayley map: `U_C(b) = Ubar Cay(x)`, `x = J(b - bbar)`. With
`A = G^-1 X_j` and `B = X_i Ubar`, `X_i^-1 G^-1 X_j = Ubar Cay(x)` is cleared exactly as in
Rai Eq. (18):

```
r_ij := G^-1 X_j (I_5 - x^/2) - X_i Ubar (I_5 + x^/2)            (Galilean analogue of Rai Eq. (19))
```

- `r = 0  <=>  X_j = G X_i Ubar Cay(x)`, because `det(I_5 - x^/2) = 1 + |omega_x|^2/4 > 0`
  (pre-flight F8, the analogue of Rai Eq. (17)). Measured: at `X_j := G X_i Ubar Cay(x)`,
  `max|r| = 1.8e-15` over 2000 random `(X_i, b)` (`misc_checks.py` (2)).
- The bottom two rows of `r` vanish identically: `alpha_x = J[9,:] db = 0` (structural zero,
  `jacobian_structure.py`) and `A`, `B` both have time entry `Dt`. Measured `0.0`.
- **Exact relation to GTSAM's local error.** If `X_i^-1 G^-1 X_j = Ubar Cay(x) Exp(eps^)` then
  `r = X_i Ubar Cay(x) (Exp(eps^) - I)(I - x^/2)` exactly (since `Cay(x)(I - x^/2) = I + x^/2`);
  measured `2.7e-15` (`misc_checks.py` (1)). So `r ~= M eps^ (I - x^/2)` to first order, with
  `M = R_i dRbar` orthogonal on the left and a near-identity distortion `I - x^/2` on the right.

Split `x` exactly as Rai Eq. (22a)–(22c) split `xi`: with `J_R = J[0:3,0:3]`, `J_nu = J[3:6,:]`,
`J_rho = J[6:9,:]` (project order) define

```
w = J_R b_g,  wbar = J_R bbar_g,  Gam = [wbar]_x,  c^v = J_nu b,  cbar^v = J_nu bbar,  c^p = J_rho b,  cbar^p = J_rho bbar
M = R_i dRbar   (Rai's M_ij, Sec. V)
```

and expand the top three rows. The result is **affine in `z`**:

```
r_R = R_j (I + Gam/2) - R_i dRbar (I - Gam/2) - 1/2 (Y + Y')                         (same form as Rai Eq. (22b))
r_v = v_j - v_i - g Dt s - R_i dvbar + 1/2 (R_j + M) cbar^v - 1/2 (V + V')
r_p = p_j - p_i - Dt v_i - 1/2 g Dt^2 s - R_i dpbar + 1/2 (R_j + M) cbar^p - 1/2 (P + P')

Y  = M [w]_x,   Y' = R_j [w]_x,   V = M c^v,   V' = R_j c^v,   P = M c^p,   P' = R_j c^p
```

(`R_i dvbar`, `M cbar`, `R_j cbar` are linear in `vec(R)` by Rai Eq. (27b).)

**Difference from Rai.** Rai Eq. (21b)–(21c) correct `dv`, `dp` *additively* and premultiply by
`R_i`, so Rai needs one vector lifting per translational residual (`W^v = R_i u^v`, Eq. (26a)).
GTSAM's correction is a **right group composition** (M2 §3.7: additive component-wise correction
is off by `0.13` at `|db| = 0.05`), so the velocity/position corrections are rotated by the
*corrected* rotation and appear through `1/2 (R_j + M)`: both an `M`-side and an `R_j`-side
lifting are needed. Note also Rai's `W^v` uses `R_i`, ours uses `M = R_i dRbar`.

**Structure of `J` (measured, `jacobian_structure.py`).** `J[0:3,3:6] = 0`, `J[9,:] = 0`, and
**`J[3:6,3:6] = J[0:3,0:3]`** to FD noise (`<= 2.7e-10` vs `|J_R| ~ 0.4–0.5`, on the fixture and
on five random time-varying streams). This follows the recursion GTSAM doc Eq. (35) given
`J^{omega,a} = 0`, if the `(nu,nu)` and `(omega,omega)` blocks of the `Gal(3)` right Jacobian
coincide; that last step is measured, not proven here.

### b.2 Index map (0-based, half-open), `n = 124`

| block | indices | dim |
| --- | --- | --- |
| `vec(R_k)`, `k = 0,1,2` | `[15k, 15k+9)` | 9 x 3 |
| `v_k` | `[15k+9, 15k+12)` | 3 x 3 |
| `p_k` | `[15k+12, 15k+15)` | 3 x 3 |
| `b_k = [b_g; b_a]` | `[45+6k, 51+6k)` | 6 x 3 |
| edge `e = 0,1`, base `o = 63 + 30e`: `vec(Y)` | `[o, o+9)` | 9 x 2 |
| `vec(Y')` | `[o+9, o+18)` | 9 x 2 |
| `V`, `V'` | `[o+18, o+21)`, `[o+21, o+24)` | 6 x 2 |
| `P`, `P'` | `[o+24, o+27)`, `[o+27, o+30)` | 6 x 2 |
| homogenising `s` (`s^2 = 1`) | `123` | 1 |

Total `45 + 18 + 60 + 1 = 124`. The anisotropic variant (§d.3) appends `rho_v,e` at
`[124+6e, 127+6e)` and `rho_p,e` at `[127+6e, 130+6e)`: `n = 136`.

Absolute rotations are full 3x3 matrices with orthogonality **and** handedness constraints. Cayley
appears only in the bias-correction factor `Cay(x)`, which is never a decision variable — it is
cleared algebraically, so no Cayley coordinate is lifted and no `theta = pi` chart issue arises for
the absolute rotations (pre-flight F3).

**Comparison with Rai's sizes.** Counting Rai Eq. (28) for *this* instance (3 KF, 2 edges, 3 bias
blocks): `45 + 18 + 24*2 + 1 = 112`. Ours is `112 + 12`, the extra 12 being `V', P'` (two 3-vectors
per edge). Rai's published `n` (Sec. VII-B: `n = 340` at `K = 8, M = 2`, `652` at `K = 16, M = 2`;
Table II: `334, 646, 652, 958, 970`) all satisfy `n = 39K + 6M + 16`; a direct count of Eq. (28)
with `K` keyframes and `K-1` edges gives `39K + 6M - 23`, exactly `39` (one keyframe plus one edge)
fewer. The offset **could not be reconciled from the text** (one consistent reading is `K+1`
keyframe states, but Sec. VII-A says each window contains `K` keyframes). Per keyframe-plus-edge we
use `15 + 30 = 45` against Rai's `39`.

---

## (c) Constraints `{A_k, b_k}`

Every constraint is written as `z^T A_k z = b_k` with `A_k` symmetric; `b_k = 1` only for `s^2 = 1`.
Each is therefore homogeneous of degree 2 in `z` (`check_constraints.py`: asymmetry `0`, homogeneity
defect `<= 1.1e-13`). As polynomials in the *original* state many would be degree 3–4 (e.g.
`Y^T Y` is quartic in `(R, b)`); the liftings are what keep them quadratic. `Lifting side` below
means the pair `(M; Y, V, P)` or `(R_j; Y', V', P')`; `E = [[w]_x, c^v, c^p]` (3x5),
`L = [Y, V, P] = M E`.

### c.1 Primary (define the feasible set)

| id | identity | count here | source |
| --- | --- | --- | --- |
| P0 | `s^2 = 1` | 1 | Rai Eq. (31) |
| P1 | `R_k^T R_k = s^2 I` | 6/KF = 18 | Rai Eq. (31) |
| P2 | `R_k^(a) x R_k^(b) = s R_k^(c)`, cyclic | 9/KF = 27 | Rai Eq. (31) |
| P3 | `M^T Y = [w]_x s`, `R_j^T Y' = [w]_x s`, `M^T V = c^v s`, `R_j^T V' = c^v s`, `M^T P = c^p s`, `R_j^T P' = c^p s` | 30/edge = 60 | Rai Eq. (30a)–(30c) for `Y, Y', V, P`; `V', P'` rows DERIVED |
| P4 (aniso only) | `M rho_v = s r_v`, `M rho_p = s r_p` | 6/edge = 12 | DERIVED |

Primary total `106` (rank `106`); anisotropic `118` (rank `118`).

### c.2 Redundant (valid on every feasible point; tighten the relaxation only)

| id | identity | count | source / Rai Table I |
| --- | --- | --- | --- |
| RA1 | `R_k R_k^T = s^2 I` | 18 | Rai Eq. (32), A1 |
| RB1 | `Y w = 0`, `Y' w = 0` | 6/edge = 12 | Rai Eq. (33), B1 |
| RB2 | `Y^T Y = |w|^2 I - w w^T` (and `Y'`) | 12/edge = 24 | Rai Eq. (34), **B2** |
| RC1 | `|V|^2 = |c^v|^2`, `|P|^2 = |c^p|^2` (both sides) | 4/edge = 8 | Rai Eq. (35), C1 (`R_j` side DERIVED) |
| RG | `Y^T V = [w]_x^T c^v`, `Y^T P = [w]_x^T c^p`, `V.P = c^v.c^p` (both sides) | 14/edge = 28 | DERIVED |
| RL | `s Y = M [w]_x`, `s V = M c^v`, `s P = M c^p` (both sides) | 30/edge = 60 | DERIVED (Rai's definitions (22c), (26) as degree-2 equalities) |
| RH | `(M e_a) x L_k = M (e_a x E_k)` for each lifted column `L_k = M E_k` (both sides) | 90/edge = 180 | DERIVED |
| RS | `Y M^T = [M w]_x`, `Y' R_j^T = [R_j w]_x` | 18/edge = 36 | DERIVED |
| RR (aniso only) | `|rho_v|^2 = |r_v|^2`, `|rho_p|^2 = |r_p|^2`, `rho_v.rho_p = r_v.r_p` | 3/edge = 6 | DERIVED |

Redundant total `366`; joint rank of all `472` base constraints `453`; anisotropic `490`, joint
rank `471` (`check_constraints.py`). Known linear dependencies: `tr(R R^T) = tr(R^T R)` gives one
per keyframe between P1 and RA1 (`rank(P1 u RA1) = 11` per KF, `misc_checks.py` (3)); RS adds `20`
of its `36` to the span.

**The Galilean Gram identity.** RB2, RG and RC1 are the blocks of one identity per lifting side,

```
L^T L = E^T E        (15 scalar equations; valid because M^T M = I, resp. R_j^T R_j = I)
```

Rai's B2 is its `[w]_x`-`[w]_x` block and C1 its two vector diagonals. The off-diagonal blocks
(RG) couple the rotation-bias and translation-bias liftings. They are not in Rai's Table I (in Rai's
variables the analogue would be `Y^T W^v = [w]_x^T dRbar^T u^v`, also quadratic). Whether RG matters for
tightness the way B2 does in Rai's Fig. 3 ablation is **OPEN** (M4).

**Removed duplicate (review of the WIP scratch).** The WIP had a "row handedness" family
`R_(a) x R_(b) = s R_(c)`. Column and row cross products are both entries of `cof(R)`, so the 9 row
forms are *identical* to the 9 P2 forms (max difference `0.0`, `check_constraints.py`). It is not
counted.

**Which families carry handedness.** P2, RH and RS use `det = +1` (`M [w]_x M^T = det(M)[M w]_x`);
everything else holds for any orthogonal matrix (§e).

### c.3 AutoTight-style cross-check (described, run in scratch only)

Following Rai's reference to AutoTight (Rai Sec. V) and Barfoot's use of the same tool (Barfoot, text after
Eq. (21): "we have been developing a tool to automatically find such constraints"):

1. Choose a sub-vector `z_S` (here the 67 entries touched by edge 0: `R_0, v_0, p_0, R_1, v_1,
   p_1, b_0`, the 30 edge-0 liftings, `s`).
2. Draw `N` i.i.d. feasible points by sampling the *state* (`R_k ~ Exp(N(0,4I))` so all of
   `SO(3)` including near-`pi` is covered; `v, p ~ N(0,I)`; `b ~ N(0, 0.09 I)`) and computing
   the liftings from their definitions.
3. Evaluate all `m(m+1)/2` monomials `z_a z_b` (`a <= b`); the null space of the `N x 2278`
   sample matrix is the space of *all* quadratic identities on the variety in this basis.
4. Compare its dimension with the span of the hand-derived families restricted to `z_S`, and
   check that every hand-derived form lies inside it (soundness).

Result (`autotight_nullspace.py`, `N = 6000`): sample rank `2015`, null space **263**
(singular-value gap `6.0e-5 -> 1.4e-13`). Hand-derived span by stage: primary `60`; + Rai Table I
analogues `92`; + RG `106`; + RL `136`; + RH `226`; + RS `236`. Every hand-derived form lies in the
null space (outside-component `<= 1.3e-11`). **27 null-space directions remain unidentified**; 18
of them involve only lifted, bias and `s` entries (no rotation entries), and none are in the pure
lifted×lifted / bias×bias class (where the Gram identities account for all 30). They are not
included in the families above. Sampling only `s = +1` is sufficient because every form is even
under `z -> -z`.

---

## (d) The cost `Q` and the isotropy question

### d.1 Cost (isotropic baseline weighting)

```
J(z) = sum_e [ 1/2 tau_R ||r_R||_F^2 + tau_v ||r_v||^2 + tau_p ||r_p||^2 ]          IMU (cleared Galilean-Cayley)
     + sum_{k=1,2} w_g ||p_k - y_k s||^2                                               GNSS
     + 1/2 w_R ||R_0 - Rbar_0 s||_F^2 + w_v ||v_0 - vbar_0 s||^2 + w_p ||p_0 - pbar_0 s||^2   prior
     + sum_{k=0,1} (b_{k+1} - b_k)^T Lambda_rw (b_{k+1} - b_k)                         random walk
     = sum_t (F_t z)^T W_t (F_t z) = z^T Q z,     Q = sum_t F_t^T W_t F_t  (PSD)
```

- The **`1/2` on the rotation terms** matches GTSAM's `tau_R |r_rot|^2`, because
  `||M eps^||_F^2 = 2|eps|^2`. Rai Eq. (20) writes `tau_R ||r^R||_F^2` without the `1/2`; with the
  literal Eq. (20) weight the cost differs from the GTSAM-weighted reference by `79%` near truth,
  with the `1/2` by `6.1e-7` (`misc_checks.py` (4), `eps = 1e-4`). This is a deviation from Rai's
  equation, stated, and immaterial to QCQP-representability.
- **Lifted form, as a claim.** The IMU rotation rows of `F` have coefficient exactly `-1/2 I_9` on
  the `vec(Y)` and `vec(Y')` columns and `0` on every bias column; likewise `-1/2 I_3` on `V, V'`
  (`P, P'`) and `0` on bias for `r_v` (`r_p`) (`check_cost.py`, deviation `0.0`). Bias enters `Q`
  only through the liftings and the random-walk term (`max|Q[b, non-b]| = 0`). The lifting is
  load-bearing: perturbing only the `Y_0` entries of `z` by `1e-6` at truth moves the cost from
  `1.2e-20` to `1.5e-4` (`misc_checks.py` (6)), which an implementation evaluating Rai Eq. (19)
  from `(R, b)` would not do. (M2's verifier note: Rai Eq. (22b) evaluated with definitions is
  numerically equal to Eq. (19), so only a test on `z`'s own lifted entries distinguishes them.)
- **Exact identity with the unlifted 5x5 cost.** For any lifted state, `z^T Q z` equals the cost
  computed independently from 5x5 `Gal(3)` matrices and Eq. (19)-style clearing (`direct_cost`):
  `max rel. diff 1.1e-15` over 500 random states (`check_cost.py`).
- **Agreement with GTSAM** (reference: `GalileanImuFactor.unwhitenedError` weighted by the same
  scalar `tau`s, plus GTSAM `PriorFactorPose3`, `PriorFactorVector`, `GPSFactor`,
  `BetweenFactorConstantBias`; `check_cost.py`):
  - `bbar = b_true`: cost at truth `1.2e-20` (factored evaluation; the dense `z^T Q z` gives
    `-5.6e-8` from cancellation, since `max|Q| = 1.1e10`). Near truth `max|cost/J_gtsam - 1|` over
    50 perturbations: `1.2e-4, 3.7e-6, 2.5e-7` at `eps = 1e-2, 1e-3, 1e-4`.
  - `bbar = 0` (bias correction active, `|b_true - bbar| = 0.117`): at truth the two costs are
    `3.2968e-3` vs `3.3157e-3` (rel. diff `5.7e-3`); the relative difference halves with the bias
    offset (`5.72e-3, 2.94e-3, 1.48e-3, 7.49e-4` for offsets `0.117, 0.0585, 0.0292, 0.0146`). Its
    sources are `Cay` vs `Exp` (`max|Ubar(Cay(x) - Exp(x))|` scales by `8.0` per halving, i.e.
    cubically) and the `(I - x^/2)` metric distortion; both vanish as `bbar -> b`.
  The two costs are therefore **not equal** off truth; they agree to first order in the state error
  at a model-consistent `bbar`. That is the precise sense of "the cost is GTSAM's factor".

### d.2 The isotropy question — answer

**Rai's mechanism does not arise here, but a different one does.** Rai's scalar weight is forced
because `R_i^T` multiplies the translational residual (Eq. (21b)) and is removed via
`tau ||r||^2 = tau ||R_i r||^2` (text before Eq. (23)). The cleared Galilean residual contains no
`R_i^T` at all: it is natively affine in `z` (§b.1). Isotropy is therefore not needed to make the
*residual* affine. It is needed, without further lifting, for the *weight*: GTSAM's covariance
(GTSAM doc Eq. (39)) is for the local error `eps`, and `r ~= M eps^ (I - x^/2)`, so a body-frame
weight `W` corresponds to `r^T M W M^T r`, which depends on the rotation state.

**(i) Obstruction (proved, DERIVED).** Let `rho_v = M^T r_v` and suppose `rho_v^T W rho_v = z^T Q z`
on the feasible set. No entry of `z` other than `v_j` depends on `v_j`, so replacing `v_j` by
`v_j + lam d` changes `z` only in those entries, and the `lam^2` coefficient of `z^T Q z` is
`d^T Q_{v_j v_j} d`, a constant. The `lam^2` coefficient of `rho_v^T W rho_v` is
`d^T M W M^T d`. Equality for all `M in SO(3)` forces `M W M^T` constant, i.e. `W = c I`. The same
argument with `p_j` covers `rho_p`. **This is exactly where anisotropy fails: the `v_j v_j` (and
`p_j p_j`) block of `Q` cannot depend on the rotation.** Numerically (`isotropy_fit.py`): the
`lam^2` coefficient is constant (`4.020000`) for `W = 3I` and ranges over `[1.348, 6.693]` for
`W = diag(1,2,5)` across 200 random `R_0`; a least-squares fit of `rho_v^T W rho_v` by all 2278
degree-2 monomials of the edge-0 sub-vector has relative residual `2.4e-15` (`3I`) versus
`7.9e-2` (`diag(1,2,5)`), `8.7e-4` (`diag(1,1,1.01)`), `1.6e-1` (random SPD).

**(ii) What anisotropy is available with no extra lifting (DERIVED).** Weights applied on the
*right* of the 3x5 top block `rt = [r_R | r_v | r_p]` are rotation-invariant on the left:
`tr(rt G rt^T)` with constant `G >= 0` is quadratic in `z` and `~= tr(Et G Et^T)`,
`Et = [eps_omega^ | eps_nu | eps_rho]`. With `G = blkdiag(D, G2)`:

- rotation: `tr(eps^ D eps^T) = eps^T (tr(D) I - D) eps` — any SPD `W_R` with
  `lambda_max(W_R) <= tr(W_R)/2` (so that `D = tr(W_R)/2 I - W_R >= 0`);
- velocity/position: `G2 (x) I_3` — directionally isotropic within each block, but with a full
  2x2 **v–p correlation**. Rai's Eq. (23) drops this; on the fixture the trace-reduced
  correlation coefficient is `-0.867` (`isotropy_fit.py` (d)).

Against GTSAM weighted by the same structured `W` (full `W_R = Sigma_rot^-1`, `G2 (x) I_3`):
`max|ratio - 1| = 1.4e-4, 3.7e-6, 2.0e-7` at `eps = 1e-2, 1e-3, 1e-4`. The LSQ fit agrees: the
cross term `rho_v . rho_p` is a quadratic form in base `z` (residual `3.1e-15`). This device is a
property of the weighting, not of the Galilean structure; it would apply equally to Rai's residual.

**(iii) Full directional anisotropy with extra liftings (DERIVED).** Lift `rho_v = M^T r_v`,
`rho_p = M^T r_p` (6 per edge, `n = 136`) with the degree-2 primary constraint `M rho = s r` (P4)
and redundant RR; then `[rho_v; rho_p]^T W_vp [rho_v; rho_p]` is quadratic for any 6x6
`W_vp >= 0`. Measured (`isotropy_fit.py` (c)): `z^T Q z` equals the direct anisotropic cost
(`7.1e-16`); with `W_vp` the inverse of GTSAM's (vel,pos) marginal block, agreement with GTSAM
weighted by the same `W_vp` is `1.0e-4, 5.2e-6, 4.2e-7` at `eps = 1e-2, 1e-3, 1e-4`
(`bbar = b_true`), while the same `W_vp` applied to the world-frame `r` stalls at `1.4e-2` — the
frame is right only with `M^T`. Rotation–translation cross-correlations would additionally need
`rho_R = M^T r_R` (9 per edge, `n = 154`); that extension is described, **not built or checked**.

**Answer.** Yes, the direct-product Galilean formulation admits an anisotropic weight with a
quadratic cost — partially for free (right-multiplied: rotation anisotropy up to the
`lambda_max <= tr/2` bound and a v–p correlation), fully at the price of `6` (translational) or
`15` (all blocks) extra lifted coordinates per edge. Directional velocity/position anisotropy is
impossible in the base lifting, for the reason in (i). Confidence: high for representability (proof
plus three independent numerical checks); **unknown for tightness** — the extra liftings enlarge the
SDP and are exactly the kind of variable Rai needed B2/C1-type redundancy for (OPEN-HYPOTHESIS,
M4). None of this is a novelty claim: residual lifting is standard and applies to Rai's factor too.

---

## (e) Lifting equivalence in both directions

**Forward.** Ground truth satisfies every family (`max|g| <= 1.5e-15`, except RR `<= 9.4e-15`; also on 200 random lifted states, `<= 5.7e-12`), for `bbar = b_true` and `bbar = 0`
(`check_constraints.py`). Each family is non-trivial: at `z_truth + 1e-3 N(0, I)` every family is
violated (smallest family-max over 20 draws `3.3e-6`, P0).

**Reverse.** Feasible set of P0–P3 = `{ +- z(state) }`: `s = +-1` by P0; P2 with `s = 1` forces
`R in SO(3) u {0}` (the cross-product relations make the columns mutually orthogonal with
`|a| = |b||c|` cyclically, so the product of norms is 0 or 1, and a zero column forces `R = 0`);
P1 excludes `R = 0`; with `M` orthogonal, P3 determines each lifting uniquely
(`Y = M [w]_x s`, ...). Numerically (`spurious.py`, trust-region least squares onto the primary
constraints from 200 random `z_0`): 200/200 converged, every solution equals `s z(state(z))` to
`3.2e-13`, `det R_k in [1 - 3e-16, 1 + 4e-16]`. `misc_checks.py` (5): P2 alone from 300 random
starts gives 238 `SO(3)` points, 62 zero matrices, 0 others.

**Spurious points and what excludes them** (`spurious.py`):

| spurious point | satisfies | violated (= excluded by) |
| --- | --- | --- |
| `R_0 -> R_0 diag(1,1,-1)` (det `-1`), liftings consistent | P0, P1, P3, RA1, RB1, RB2, RC1, RG, RL | **P2**, RH, RS |
| `R_0 = 0`, `b_0 = 0` | P0, P2, P3, all lifting families | **P1**, RA1 |
| edge-0 `Y, V, P` built with a wrong rotation `Q` | P0–P2, RA1, RB1, RB2, RC1, RG | **P3**, RL, RH, RS |
| edge-0 `Y, V, P` sign-flipped | same as above | **P3**, RL, RH, RS |
| edge-0 liftings from a different bias | P0–P2, RA1 | **P3**, RB1, RB2, RC1, RG, RL, RH, RS |
| global sign `-z` | all | none — **not spurious**: same decoded state, same cost |

Dropping P2 lets the solver reach `det = -1` (range `[-1, 1]`); dropping P3 reaches liftings off
their definitions by up to `4.6e2`; dropping P1 reaches `det ~ 0` (`spurious.py`). **The Gram and
kernel identities (Rai's B1/B2/C1 and RG) do not pin a lifting**: they hold for `L = Q E` with any
orthogonal `Q`; only P3/RL (and the handedness-type RH/RS) tie the lifting to `M`.

---

## (f) What is frozen, and what is therefore certified

1. **Bias Jacobian `J` frozen at `bbar`** (GTSAM doc Eq. (34)–(36); Rai: "All Jacobians are
   evaluated at the nominal bias"). The correction is first order in `db`; against
   re-integration the Galilean model error is `O(|db|^2)` (M2 §3.7).
2. **`Exp` replaced by `Cay`** in the correction: an `O(|J db|^3)` deviation from GTSAM's own
   model (measured cubic, §d.1). This is the surrogate's choice, not GTSAM's.
3. **Weights frozen**: `Sigma` from `preintMeasCov()` at `bbar`, reduced to three scalars per edge
   by Rai Eq. (23) (baseline) or to the structured/lifted forms of §d.2. Cross-correlations dropped
   in the baseline include a v–p correlation of `-0.867` on this fixture.
4. **Metric distortion** of the cleared residual: `r = B Cay(x) (Exp(eps^) - I)(I - x^/2)`; the
   right factor makes the Frobenius weight differ from the local-error weight at `O(|x|)`.
5. **Prior on `R_0`** is chordal (Frobenius), GTSAM's is geodesic; they agree to first order.

A certificate from M4 would therefore certify global optimality **of this frozen-weight Cayley
surrogate QCQP**, not of GTSAM's nonlinear least-squares problem, and not of the physical
estimation problem.

---

## (g) Four things that must not be conflated

| statement | status here |
| --- | --- |
| **QCQP-representable** | **Yes, by construction**: affine residuals in `z`, `Q = sum F^T W F`, every constraint `z^T A_k z = b_k`; feasible set = `{+- z(state)}` (§e). |
| **Exact physical dynamics** | **No.** `Ubar` is exact ZOH at `bbar` (M2 §3.6), but the bias dependence is first order and the retraction is `Cay`, not `Exp`; weights are frozen and isotropised. |
| **Exact (tight) convex relaxation** | **Unknown.** Nothing has been solved. OPEN-HYPOTHESIS for M4. |
| **Certified global optimum** | **Not claimed.** Would need, in M4, a Clarabel (not SCS) solve with rank-1 by Barfoot's criterion (log singular-value ratio `>= 5`, Barfoot Fig. 2 caption) and a small duality gap, and would then apply to the surrogate of §f only. |

---

## Foreseen tightness risks (for M4)

1. **Graph sparsity.** A 2-edge chain with GNSS at two keyframes is sparse; Barfoot report weaker
   tightness with sparser measurement graphs (continuous-time experiment, citing Holmes and
   Barfoot 2023).
2. **More liftings per edge than Rai** (30 vs 24, plus 6 or 15 for anisotropy). Rai's ablation
   (Fig. 3) shows primary + A1 + B1 never certified (gap `>= 0.88` over 504 solves) and B2 decisive;
   whether RG/RH/RS play that role for `V', P'` is untested.
3. **27 AutoTight directions unaccounted for** (§c.3) — any of them could be the decisive one.
4. **Conditioning.** `max|Q| = 1.1e10` with `O(1)` constraints; the dense `z^T Q z` already loses
   `~1e-7` absolute to cancellation at truth. Barfoot needed to rescale `Q` for reliable solves
   (continuous-time experiment). Whether Clarabel resolves `sigma_1/sigma_2 >= 1e5` at this
   conditioning is untested.
5. **Weak bias excitation.** Bias directions are observable (rank 45/45) but the Jacobian
   condition is `~4e7`; near-degenerate directions are where Rai's one failed window (rank 3,
   UrbanNav) came from.
6. **Noise-free data.** At `bbar = b_true` the optimum has zero cost, which is the easiest case;
   tightness must be re-measured with noise.

---

## Reproduce

```bash
PY=/home/parth/Desktop/CERTIFIABLE_GAL_IMU_SLAM/.venv/bin/python
$PY -m experiments.m3_scratch.instance             # instance, taus, J model check
$PY -m experiments.m3_scratch.jacobian_structure   # J zeros and J_va = J_R
$PY -m experiments.m3_scratch.check_constraints    # families: counts, ranks, vanishing, degree
$PY -m experiments.m3_scratch.check_cost           # cost identities, GTSAM agreement, rank 45/45
$PY -m experiments.m3_scratch.isotropy_fit         # isotropy (a)-(d)
$PY -m experiments.m3_scratch.spurious             # spurious points, reverse direction (~5 min)
$PY -m experiments.m3_scratch.autotight_nullspace  # AutoTight-style cross-check
$PY -m experiments.m3_scratch.misc_checks          # per-claim small checks
```
