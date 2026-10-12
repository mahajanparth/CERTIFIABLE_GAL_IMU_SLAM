# M1 — Conventions and Gal(3) mechanics

**Deriver's document.** Everything here is either (i) an algebraic identity derived in full below, or
(ii) a measured property of the installed `gtsam` 4.3.0 build. Nothing is asserted on the authority
of a paper or of `CLAUDE.md`. Every checkable fact has a numbered claim in
[`claims/M1.yaml`](claims/M1.yaml); the claim id is given inline as `[M1-Cnn]`.

The implementer of `src/gal3.py` and `src/imu_dynamics.py` must build from **this document alone**.
It is written so that no formula requires reading the verification scripts to disambiguate.

---

## 0. Notation and the fixed convention

### 0.1 The group

```
X(R, v, p, t) = [[ R      v   p ],
                 [ 0_{1x3} 1   t ],
                 [ 0_{1x3} 0   1 ]]                 in R^{5x5},   R in SO(3)
```

Column 3 (0-indexed) is **velocity**, column 4 is **position**, and the time coordinate sits at
entry `(3, 4)`. Write `X = (R, v, p, t)` for this element. The identity is `I_5 = (I, 0, 0, 0)`.

> **Footgun.** The matrix puts `v` before `p`; `gtsam.Gal3`'s *constructor* puts `p` before `v`.
> See §7.

### 0.2 The algebra

```
xi = (phi, u, w, alpha) in R^10,    phi, u, w in R^3,  alpha in R
xi^ = [[ [phi]x   u   w ],
       [ 0_{1x3}  0  alpha ],
       [ 0_{1x3}  0   0   ]]
```

Block order in the 10-vector is **(rot, vel, pos, time)**: `u` occupies the velocity column, `w` the
position column. `[a]x` is the skew matrix with `[a]x b = a x b`.

### 0.3 Physics

```
omega_m = omega + b_g + n_g         gyroscope
a_m     = a     + b_a + n_a         accelerometer, specific force in BODY frame
Rdot = R [omega]x       vdot = g + R a        pdot = v
```

so the bias-corrected inputs fed to the kinematics are `omega = omega_m - b_g`,
`a = a_m - b_a` (noise-free case) `[M1-C44]`. `g` is the gravity vector in the world frame
(`g = (0, 0, -9.81)` for the ENU convention used throughout; nothing below depends on its value).

### 0.4 Bias ordering

The project orders the bias vector **gyro first**:

```
b = [b_g ; b_a] in R^6
```

GTSAM orders it the other way. See §8.

---

## 1. Group axioms

### 1.1 Product

Multiplying the two 5x5 matrices block-wise:

```
X1 X2 = ( R1 R2,
          v1 + R1 v2,
          p1 + R1 p2 + v1 t2,
          t1 + t2 )
```

`[M1-C01]`. The `v1 t2` term in the position block is the whole reason this group is not
`SE_2(3)`; it is also the term that makes the endpoint relation of §5 subtle.

**Closure** `[M1-C02]`: `R1 R2 in SO(3)`; rows 3 and 4 of the product are `[0 0 0 1 t1+t2]` and
`[0 0 0 0 1]` respectively, so the product has the structural form of §0.1. The time block is
**additive**, `t(X1 X2) = t1 + t2` `[M1-C07]`. Every claim about the endpoint relation in §5 is a
claim about this entry as much as about `(R, v, p)` — the defect in the project's `CLAUDE.md` was
first visible here.

**Associativity** `[M1-C03]` is inherited from matrix multiplication; it is still worth a numerical
claim because an implementation that stores `(R, v, p, t)` as a tuple and hand-codes the product can
break it (e.g. by writing `v2 t1` instead of `v1 t2`).

**Identity** `[M1-C04]`: `I_5 X = X I_5 = X`.

**Non-commutativity** `[M1-C06]`: `X1 X2 != X2 X1` in general; a certifying witness must be
exhibited so that an implementation accidentally returning a commutative (e.g. purely additive)
product is rejected.

### 1.2 Inverse

Solve `X Y = I_5`:

```
X^-1 = ( R^T,  -R^T v,  -R^T (p - t v),  -t )
```

`[M1-C05]`. Check the position block: with `Y = (R^T, -R^T v, q, -t)`, the product's position block is
`p + R q + v(-t) = 0`, so `q = -R^T(p - t v)`. The `t v` term is again the group's signature.
The inverse is **total** — no domain restriction, since `R^T` always exists.

---

## 2. The Lie algebra

### 2.1 wedge / vee

`vee` reads the 10 components back out of the matrix:

```
vee(A) = ( (A32, A13, A21), A[0:3,3], A[0:3,4], A[3,4] )
```

and `vee(wedge(xi)) = xi`, `wedge(vee(A)) = A` for every `A in gal(3)` `[M1-C08]`. Both are exact
(no floating-point tolerance is needed beyond exact equality; claim uses `1e-14`).

### 2.2 The bracket, `ad`

With `xi = (phi, u, w, alpha)`, `eta = (psi, x, y, beta)`:

```
[xi^, eta^] = ( phi x psi,
                phi x x - psi x u,
                phi x y - psi x w + beta u - alpha x,
                0 )^
```

so `ad_xi` as a 10x10 matrix in the block order (rot, vel, pos, time) is

```
          |  [phi]x      0        0      0 |
ad_xi  =  |  [u]x      [phi]x     0      0 |
          |  [w]x      -alpha I [phi]x   u |
          |   0          0        0      0 |
```

`[M1-C09]`. Note the two entries that do not appear in `se_2(3)`: the `-alpha I` block and the
`u` column. The last row is identically zero: **the time coordinate of a bracket is always zero**,
i.e. `alpha` is a central-direction coordinate for the bracket.

`ad` is a Lie-algebra homomorphism, `ad_{[xi,eta]} = [ad_xi, ad_eta]`, and the Jacobi identity holds
`[M1-C10]`. `trace(ad_xi) = 0` for all `xi`, so Gal(3) is **unimodular** `[M1-C17]`.

### 2.3 `Ad`

`Ad_X(xi) := vee(X xi^ X^-1)`. Carrying out the triple product:

```
phi'   = R phi
u'     = R u + [v]x R phi
w'     = R w - t R u + [p - t v]x R phi + alpha v
alpha' = alpha
```

`[M1-C11]`, confirming pre-flight F8. As a 10x10 matrix,

```
          |      R          0     0   0 |
Ad_X   =  |   [v]x R        R     0   0 |
          | [p - t v]x R  -t R    R   v |
          |      0          0     0   1 |
```

Consequences, each separately claimed because each is a different way for an implementation to be
wrong:

- `Ad_{X Y} = Ad_X Ad_Y` `[M1-C12]`;
- `Ad_{X^-1} = (Ad_X)^-1` `[M1-C13]` — note this is *not* obtained by transposing anything;
- **Ad–ad compatibility**: `Ad_X ad_xi Ad_X^-1 = ad_{Ad_X xi}`, equivalently
  `Ad_X( ad_xi(eta) ) = ad_{Ad_X xi}( Ad_X eta )` `[M1-C14]`;
- `Ad_{exp(xi)} = exp(ad_xi)` (matrix exponential on the right) `[M1-C15]`;
- the last row of `Ad_X` is `e_10^T`, i.e. the time coordinate is **Ad-invariant** `[M1-C16]`;
- `det(Ad_X) = 1` for every `X` `[M1-C17]` (block-triangular with diagonal blocks `R, R, R, 1`).

`Ad` is total — defined for every `X in Gal(3)`, no restriction.

---

## 3. Exponential and logarithm

### 3.1 The three coefficient functions

Let `theta = ||phi||`, `Phi = [phi]x`. Define

```
kappa1(theta) = (1 - cos theta) / theta^2
kappa2(theta) = (theta - sin theta) / theta^3
kappa3(theta) = (theta^2 + 2 cos theta - 2) / (2 theta^4)
```

and

```
Gamma1(phi) = I + kappa1 Phi + kappa2 Phi^2          ( = the SO(3) left Jacobian J_l )
Gamma2(phi) = (1/2) I + kappa2 Phi + kappa3 Phi^2
```

`[M1-C18]`. These are the `Gamma_m(phi) = sum_{n>=0} Phi^n / (n+m)!` family; the closed forms follow
from `Phi^3 = -theta^2 Phi`. (This matches the `Gamma1/Gamma2`, `kappa1..3` presentation attributed
to Delama et al.; the **equation numbers could not be extracted** — the paper is not available in
this workspace, so the claim is tagged `DERIVED`, not `VERIFIED-PUBLISHED`.)

### 3.2 `exp`

For `A = xi^`, `A^n = [[Phi^n, Phi^{n-1} u, Phi^{n-1} w + alpha Phi^{n-2} u], [0,0,0], [0,0,0]]`
for `n >= 2`, and row 3 of `A^n` vanishes for `n >= 2`. Summing `sum A^n / n!`:

```
exp(phi, u, w, alpha) = ( Exp_SO3(phi),
                          Gamma1(phi) u,
                          Gamma1(phi) w + alpha Gamma2(phi) u,
                          alpha )
```

`[M1-C19]`. The **time block of `exp(xi)` equals `alpha` exactly** `[M1-C20]` — exactly, not to
first order; this is the identity the endpoint relation of §5 exploits. `exp` is **total** on
`gal(3)`.

### 3.3 `log`

Invert block by block:

```
phi   = Log_SO3(R)
u     = Gamma1(phi)^-1 v
w     = Gamma1(phi)^-1 ( p - t Gamma2(phi) u )
alpha = t
```

`[M1-C21]`, with

```
Gamma1(phi)^-1 = I - (1/2) Phi + c2(theta) Phi^2,
c2(theta) = ( 1 - theta sin theta / (2 (1 - cos theta)) ) / theta^2
```

**Use this form of `c2`, not `1/theta^2 - (1 + cos theta)/(2 theta sin theta)`.** The two are
algebraically identical, but the latter has `sin theta` in a denominator and so is numerically
singular at `theta = pi` — precisely the edge of the chart we care about. The form above is singular
only where `1 - cos theta = 0`, i.e. `theta = 2 pi k`.

### 3.4 Domain

- `Gamma1(phi)` is invertible **iff `theta != 2 pi k`, `k = 1, 2, ...`**. Measured
  `det Gamma1`: `1.0` at `theta = 0`, `0.405` at `theta = pi`, `2.2e-3` at `theta = 6.0`, `0` at
  `theta = 2 pi`. Condition number at `theta = pi` is `pi/2 = 1.5708` `[M1-C24]`.
- `Log_SO3` is single-valued only for `theta < pi`. Hence **`log(exp(xi)) = xi` holds for
  `||phi|| < pi`** `[M1-C22]` and fails (returns the other branch) above it.
- `exp(log(X)) = X` holds for **every** `X in Gal(3)` `[M1-C23]`: `exp` is surjective, because
  every `R` has some `phi` with `theta <= pi` and `Gamma1` is invertible there.
- The installed `gtsam.Gal3.Logmap` returns `||phi|| <= pi` (measured max `3.14083` over 5000
  random elements with `||phi_input||` up to `3 sigma = 9`).

### 3.5 Small-angle branches (implementation requirement)

`kappa1, kappa2, kappa3, c2` all suffer catastrophic cancellation as `theta -> 0`. Measured relative
error of the closed forms against 50-digit arithmetic:

| theta | kappa1 | kappa2 | kappa3 | c2 |
| --- | --- | --- | --- | --- |
| 1e-1 | 1.1e-14 | 1.9e-14 | 1.4e-11 | 1.3e-11 |
| 1e-2 | 2.9e-13 | 2.6e-12 | **2.2e-7** | **3.5e-8** |
| 1e-4 | 5.2e-9 | 3.1e-8 | **1.0e0** | **6.3e0** |

So the implementation **must** switch to the truncated series for small `theta`:

```
kappa1 = 1/2  - theta^2/24  + theta^4/720
kappa2 = 1/6  - theta^2/120 + theta^4/5040
kappa3 = 1/24 - theta^2/720 + theta^4/40320
c2     = 1/12 + theta^2/720 + theta^4/30240
```

whose relative truncation error at `theta = 0.1` is at most `5e-11` and at `theta = 0.03` at most
`4e-14`. **Recommended switch point `theta_small = 5e-2`**, which keeps both branches below `1e-11`
relative error `[M1-C24]`. With that branch in place, `Gamma1(phi)^-1 Gamma1(phi) - I` measures
`4.0e-14` over 20000 samples with `||phi||` uniform in `(1e-9, pi]`; without it, `9.8e-10`.

---

## 4. The Cayley map

### 4.1 Closed form, including the time coupling

`Cay(A) = (I_5 - A/2)^-1 (I_5 + A/2)`. With `S = I - [phi]x/2`:

```
Cay(phi, u, w, alpha) = ( S^-1 (I + [phi]x/2),
                          S^-1 u,
                          S^-1 ( w + (alpha/2) u ),
                          alpha )
```

`[M1-C25]`. Equivalently, as the **degree <= 2** constraint block used for lifting:

```
(I - [phi]x/2) Re = I + [phi]x/2
(I - [phi]x/2) ve = u
(I - [phi]x/2) pe = w + (alpha/2) u
```

`[M1-C30]`. Pre-flight F3 established the first three equalities for the special case `alpha = 0`;
the `+ (alpha/2) u` coupling is **new here** and is the generalisation to residuals that carry a
nonzero time block. **The sign is `+`.** It mirrors the `+ alpha Gamma2 u` of `exp` (§3.2), and the
two agree to second order because `S^-1 = I + Phi/2 + Phi^2/4 + ...` and
`Gamma1 = I + Phi/2 + Phi^2/6 + ...` share their first two terms while `Gamma2 -> 1/2 I`.

Note this does **not** make the position equality degree 3: `alpha` and `u` are independent lifted
variables, so `(alpha/2) u` is degree 2, same as before.

### 4.2 Domain

`I_5 - A/2` is block upper triangular, so

```
det(I_5 - A/2) = det(I - [phi]x/2) = 1 + ||phi||^2 / 4   >   0
```

for every `xi in gal(3)` `[M1-C26]`. Cayley is therefore **total** on `gal(3) -> Gal(3)`: the sole
denominator never vanishes (the exact analogue of Rai et al.'s Eq. (17) for SO(3)). `Re` is in
`SO(3)` with `det = +1` **automatically**, and the time block of `Cay(xi)` is `alpha` exactly
`[M1-C27]`.

### 4.3 Chart coverage

The rotation produced by `Cay([c]x)` has axis `c/||c||` and angle `theta` with

```
||c|| = 2 tan(theta / 2)
```

`[M1-C28]`. Measured: `0.511` at `0.5 rad`, `1.093` at `1 rad`, `3.115` at `2 rad`, `28.20` at
`3 rad`, `4.32e4` at `3.1415 rad`, `1.51e6` at `3.14159 rad`. **Cayley is numerically unusable near
`theta = pi` and cannot represent `theta = pi` at all.** Residual rotations in preintegration are
small, so this is acceptable for the *residual*; the *absolute* rotations `R_i, R_j` must not be
Cayley-parameterised (pre-flight F3).

### 4.4 Relation to `exp`

```
log( Cay(A) ) = A + A^3/12 + O(||A||^5)
```

`[M1-C29]`. Proof: `Cay(A) = I + A + A^2/2 + A^3/4 + O(A^4)`, and feeding
`B = Cay(A) - I` into `log(I+B) = B - B^2/2 + B^3/3 - ...` gives
`A + A^2(1/2 - 1/2) + A^3(1/4 - 1/2 + 1/3) = A + A^3/12`. Because `Cay(-A) = Cay(A)^-1`,
`log Cay` is an odd function of `A` and the next correction is `O(A^5)`, not `O(A^4)`.
`A^3/12 in gal(3)` (its top-left block is `Phi^3/12 = -theta^2 Phi/12`, still skew; its last row is
zero), so the statement is well formed in the algebra.

Measured raw ratios over 300 unit-norm `xi`: `||log Cay(s xi) - s xi||_inf / s^3 <= 0.052` at
`s in {1e-1, 5e-2}`, and after subtracting the `s^3` term,
`|| . ||_inf / s^5 <= 7.1e-3` at `s = 1e-2`.

Hence Cayley and `exp` **agree to second order** and differ at third order with a known leading
coefficient. The verifier should check the *scaling* (`||log Cay(s xi) - s xi|| / s^3 -> ||vee(A^3)||/12`),
not just that the difference is small.

---

## 5. The endpoint relation

This is where the project's own math contract was wrong, so the section is written as a derivation
rather than a statement.

### 5.1 Ground truth

Integrating §0.3 over `[t_i, t_j]`, `Dt = t_j - t_i`, with the gravity-free increment
`D_ij = (dR, dv, dp, Dt)` defined in §6:

```
R_j = R_i dR
v_j = v_i + g Dt + R_i dv
p_j = p_i + v_i Dt + 0.5 g Dt^2 + R_i dp
```

`[M1-C43]` (the Forster component form). Everything below must reproduce exactly this.

### 5.2 Form (a): the corrected pure-group form

Keyframe states are held with **time block zero**: `X_i = (R_i, v_i, p_i, 0)`. Then by §1.1

```
X_i D_ij = ( R_i dR,  v_i + R_i dv,  p_i + R_i dp + v_i Dt,  Dt )
         = ( R_j,     v_j - g Dt,    p_j - 0.5 g Dt^2,       Dt )
```

The residual is a velocity offset `g Dt` and a position offset `0.5 g Dt^2`, plus a stray `Dt` in the
time block. Left-multiplying by `H' = (I, g Dt, -0.5 g Dt^2, 0)`:

```
H' (X_i D_ij) = ( R_j,
                  g Dt + (v_j - g Dt),
                  -0.5 g Dt^2 + (p_j - 0.5 g Dt^2) + (g Dt)(Dt),
                  0 + Dt )
              = ( R_j, v_j, p_j, Dt )
```

The position block works out **because** the group's `v1 t2` term contributes `+g Dt^2`, which is
exactly what turns `-0.5 g Dt^2 - 0.5 g Dt^2` into `0`. That is the origin of the **minus sign** in
`H'`: with `+0.5 g Dt^2` the `v1 t2` term double-counts gravity and the position comes out
`p_j + g Dt^2` `[M1-C34]`.

Finally `T(-Dt) = (I, 0, 0, -Dt)` on the **left** clears the time block, because left-multiplying by
an element with `R = I, v = 0, p = 0` touches nothing but `t`:

```
X_j = T(-Dt) . H' . X_i . D_ij                                            [M1-C31]
```

**Ordering matters and both orderings are claimed.** Applying `T(-Dt)` on the **right** instead
gives `(R_j, v_j, p_j - v_j Dt, 0)` — the time block is cleared but the position is corrupted by
exactly `-v_j Dt`, because on the right `T`'s `t2 = -Dt` multiplies `v1 = v_j` `[M1-C35]`.

### 5.3 Form (a) collapses to a one-parameter subgroup

`T(-Dt)` and `H'` can be fused:

```
G(Dt) := T(-Dt) H' = ( I,  g Dt,  -0.5 g Dt^2,  -Dt )
```

(the opposite order `H' T(-Dt) = (I, g Dt, -1.5 g Dt^2, -Dt)` is **different** — the fusion is not
commutative). Two facts make `G` the natural object:

```
G(s1) G(s2) = G(s1 + s2)                                                   [M1-C33]
G(s)        = exp( s * zeta_g ),   zeta_g = (0, g, 0, -1) in gal(3)        [M1-C32]
```

The second follows from §3.2 at `phi = 0`, where `Gamma1 = I` and `Gamma2 = I/2`:
`exp(0, s g, 0, -s) = (I, s g, 0 + (-s)(1/2)(s g), -s) = (I, s g, -0.5 g s^2, -s)`. So

```
X_j = exp(Dt * zeta_g) . X_i . D_ij
```

is the entire endpoint relation, with a **single fixed generator** `zeta_g = (0, g, 0, -1)`
independent of `Dt`. The `-1` in the time slot is what cancels the `+Dt` the increment accumulates:
gravity injection and time-block bookkeeping are *the same generator*.

### 5.4 Domain restriction: `t_i` must be `0`

Repeating §5.2 with `X_i = (R_i, v_i, p_i, t_i)`, the `v1 t2` term in `H' (X_i D)` becomes
`(g Dt)(t_i + Dt)` instead of `(g Dt)(Dt)`, so

```
position block of T(-Dt) H' X_i D_ij  =  p_j + g Dt t_i                    [M1-C36]
```

and the time block comes out `t_i`, not `0`. **The relation as written is valid only on the slice
`t_i = 0`.** If absolute timestamps must be carried in the time block, the gravity factor has to be
`H'(t_i) = (I, g Dt, -0.5 g Dt^2 - g Dt t_i, 0)` — i.e. it is no longer a fixed one-parameter
subgroup. The project convention is therefore: **keyframe states always carry `t = 0`**; only
increments carry a nonzero time block. `src/gal3.py` should assert this.

### 5.5 Form (b): GTSAM 4.3.0's `NavState` form

GTSAM works with `NavState`, which is `(R, p, v)` with **no time coordinate** and composition

```
N1 N2 = ( R1 R2,  p1 + R1 p2,  v1 + R1 v2 )                                [M1-C39]
```

(verified against the installed build; note there is no `v1 t2` term — `NavState` is `SE_2(3)`, not
Gal(3)). The endpoint relation is

```
X_j = W_ij . Phi_Dt(X_i) . U_ij
W_ij     = NavState(I, 0.5 g Dt^2, g Dt)        # ctor is (R, position, velocity): p=+0.5 g Dt^2
Phi_Dt(R, p, v) = (R, p + v Dt, v)              # affine "coasting"
U_ij     = NavState(dR, dp, dv)
```

`[M1-C37]`. Expanding: `W Phi_Dt(X_i) = (R_i, p_i + v_i Dt + 0.5 g Dt^2, v_i + g Dt)`, then
right-composing `U_ij` adds `R_i dp` to position and `R_i dv` to velocity — exactly §5.1.

Two corrections to the loose description of `Phi_Dt`:

- It is **not** a group element and **not** left or right translation by any fixed `NavState`
  (measured residual `>1.17` and `>0.72` respectively when a candidate factor fitted on one element
  is applied to a second) `[M1-C40]`.
- It **is** a group *automorphism* of `NavState`: `Phi_Dt(N1) Phi_Dt(N2) = Phi_Dt(N1 N2)` exactly.
  The reason is visible only in Gal(3): on the `t = 0` slice,

  ```
  T(-Dt) X T(Dt) = (R, v, p + v Dt, 0)                                     [M1-C40]
  ```

  so **coasting is conjugation by `T(Dt)`**, an inner automorphism of Gal(3) restricted to the slice.

### 5.6 Do (a) and (b) agree?

**Yes — and not merely numerically; they are the same identity.** Lift (b) into Gal(3) by giving
`W_ij` and `U_ij` time block `0`, i.e. `W_g = (I, g Dt, +0.5 g Dt^2, 0)` and `U_g = (dR, dv, dp, 0)`.
Then

```
W_g = T(-Dt) H' T(Dt)          (exact, verified to 0.0 in double precision)
T(Dt) U_g = (dR, dv, dp, Dt) = D_ij
```

so

```
W_g . [T(-Dt) X_i T(Dt)] . U_g = T(-Dt) H' T(Dt) T(-Dt) X_i T(Dt) U_g
                               = T(-Dt) H' X_i D_ij
```

which is form (a) `[M1-C38]`.

> **The apparent sign disagreement is not a disagreement.** GTSAM's `W_ij` has position
> `+0.5 g Dt^2`; the corrected `H'` has `-0.5 g Dt^2`. Both are right. `W_g` and `H'` are
> **conjugate by `T(Dt)`**, and conjugation by `T(Dt)` maps `-0.5 g Dt^2` to
> `-0.5 g Dt^2 + (g Dt)(Dt) = +0.5 g Dt^2`. The sign one writes depends on whether the gravity
> factor stands to the left of a *time-advanced* state (GTSAM: `Phi_Dt(X_i)`, `+`) or to the left of
> the *raw* state (form (a): `X_i`, `-`). Writing `+` in form (a)'s slot is the `CLAUDE.md` bug;
> writing `-` in GTSAM's slot would be the mirror-image bug.

Measured agreement on `(R, v, p)`: `3.6e-15` over 200 random instances; both agree with ground-truth
IMU integration (5000 steps, `dt = 1e-4`) to `6.8e-14`. GTSAM's own
`PreintegratedImuMeasurements.predict` agrees with both to `4.4e-16` `[M1-C53]`.

---

## 6. The preintegration increment

### 6.1 Definition

`D_ij = (dR, dv, dp, Dt)` is the **gravity-free, `R_i`-free** integral of the bias-corrected IMU over
`[t_i, t_j]`, with `D(0) = I_5`:

```
dRdot = dR [omega]x        dvdot = dR a        dpdot = dv        Dtdot = 1
```

Written on the 5x5 matrix this is exactly a **left-invariant ODE on Gal(3)**:

```
Ddot = D . xi^,       xi = (omega, a, 0, 1)
```

(check: `D xi^` has blocks `[dR [omega]x, dR a, dv]` in row 1 and `[0, 0, 1]` in row 2, matching
term by term). Therefore, for `(omega, a)` **constant** over the interval,

```
D(Dt) = exp( Dt * (omega, a, 0, 1) )                                       [M1-C41]
```

and by §3.2,

```
dR = Exp_SO3(omega Dt)
dv = Dt   * Gamma1(omega Dt) a
dp = Dt^2 * Gamma2(omega Dt) a
Dt = Dt
```

`[M1-C42]` — the classical closed-form preintegration for a constant IMU sample, recovered with no
extra work from the Gal(3) exponential. The `w` slot of the generator is `0`: position is driven
purely through the `alpha Gamma2 u` coupling of §3.2.

### 6.2 Chaining is the group product

Concatenating two adjacent increments:

```
D_ij D_jk = ( dR_ij dR_jk,
              dv_ij + dR_ij dv_jk,
              dp_ij + dR_ij dp_jk + dv_ij Dt_jk,
              Dt_ij + Dt_jk )
          = D_ik
```

`[M1-C45]`. This **is** the Forster chaining rule, term for term — including the
`dv_ij Dt_jk` term, which is the group's `v1 t2` term. Note the index pattern carefully: it is the
**first** segment's velocity times the **second** segment's duration. The Gal(3) product is not
merely compatible with preintegration chaining; it *is* preintegration chaining, which is the
structural reason to prefer Gal(3) over `SE_2(3)` here.

### 6.3 Bias

`D_ij` depends on the bias through `omega = omega_m - b_g`, `a = a_m - b_a` `[M1-C44]`. Bias update
Jacobians are M2 material and are not derived here.

---

## 7. Cross-validation against `gtsam.Gal3` (4.3.0, installed build)

All of the following were **measured on the installed module**, not read from documentation.

### 7.1 Matrix layout — `v` before `p`

`gtsam.Gal3.matrix()` returns exactly the §0.1 layout: column 3 is velocity, column 4 is position,
`(3,4)` is time `[M1-C46]`.

### 7.2 Constructor — position before velocity

```python
gtsam.Gal3(R: Rot3, r: Vector3, v: Vector3, t: float)     # r = POSITION, v = VELOCITY
```

The second argument `r` is the **position** and the third `v` is the **velocity** — the opposite
order to the matrix columns `[M1-C47]`. Accessors: `.position()` and `.translation()` both return
`r`; `.velocity()` returns `v`; `.time()` returns `t`; `.rotation()` and `.attitude()` return `R`.
So the round trip is

```python
X = gtsam.Gal3(Rot3(R), p, v, t)         #  <-- p then v
assert X.matrix()[:3,3] == v and X.matrix()[:3,4] == p
```

### 7.3 Tangent ordering — `(omega, nu, rho, alpha)`

`gtsam.Gal3.Hat` is **bit-identical** to the §0.2 wedge: components `0:3` -> `[phi]x`, `3:6` ->
velocity column, `6:9` -> position column, `9` -> `(3,4)`. So GTSAM's Gal(3) tangent order is
**(rot, vel, pos, time)**, matching this project's `(phi, u, w, alpha)` `[M1-C48]`.

`gtsam.NavState`'s tangent order is **(rot, pos, vel)** — the opposite inner pair — and
`NavState.matrix()` puts position in column 3 and velocity in column 4, also the opposite of `Gal3`
`[M1-C49]`. Feeding a Gal(3) tangent's first nine components to `NavState.Expmap` silently swaps
position and velocity (measured `2.8e-5` discrepancy at perturbation scale `1e-5`, i.e. an `O(1)`
relative error) `[M1-C50]`. The permutation that converts is

```
P (9x10):  rows 0:3 = cols 0:3,   rows 3:6 = cols 6:9,   rows 6:9 = cols 3:6
```

### 7.4 The three footguns, side by side

| | rot | slot 2 | slot 3 | slot 4 |
| --- | --- | --- | --- | --- |
| `Gal3` matrix columns | `R` | **`v`** | **`p`** | `t` at `(3,4)` |
| `Gal3(...)` constructor | `R` | **`r` = `p`** | **`v`** | `t` |
| `Gal3` tangent | `omega` | **`nu` = vel** | **`rho` = pos** | `alpha` |
| `NavState` matrix columns | `R` | **`p`** | **`v`** | — |
| `NavState` tangent | `omega` | **pos** | **vel** | — |

### 7.5 Everything else agrees

Measured over 2000 random elements each: `Expmap`/`Logmap` match §3 closed forms to `1.0e-13`;
`AdjointMap` matches §2.3 to `1.8e-15`; `adjointMap` matches §2.2 **exactly** (`0.0`);
`compose`, `__mul__`, `inverse`, `between` match the matrix operations to `6.2e-15`;
`localCoordinates(B) = Logmap(A^-1 B)` exactly and `retract(d) = A Expmap(d)` (left-invariant
retraction) to `2.8e-17` `[M1-C51]`, `[M1-C52]`.

---

## 8. Bias ordering

```
this project:  b = [b_g ; b_a]            gyro first
GTSAM:         gtsam.imuBias.ConstantBias(biasAcc, biasGyro)
               .vector() -> [b_a ; b_g]   accel first
```

`[M1-C54]`. The conversion is the involutive block-swap

```
S = [[0, I3], [I3, 0]],      S S = I6
ConstantBias.vector() = S b          b = S ConstantBias.vector()
```

`[M1-C55]`. Any code that passes a `[b_g; b_a]` vector straight into `ConstantBias(...)` swaps gyro
and accelerometer bias **silently** — the constructor's first argument is the accelerometer bias.
The conversion must live in exactly one place in `src/`.

---

## 9. What is *not* settled here

- Left/right Jacobians of Gal(3) (`J_l`, `J_l^-1` on the full 10-dim algebra) are **not** derived.
  They are not needed for the direct-product Stage-B formulation (which uses `retract = X Exp(d)`
  and never needs `J_l^-1` of the full group), and deriving them is M2/Stage-C work. If a later
  milestone needs them, they must be derived, not assumed to be the block-triangular guess.
- Bias-update Jacobians of `D_ij` (M2).
- Noise propagation / `preintMeasCov` (M2).
- Equation numbers in Delama et al.: **could not extract** (paper not present in the workspace).
  Every formula in §3 is therefore tagged `DERIVED` and verified against `scipy.linalg.expm` and the
  installed `gtsam.Gal3`, not against the paper.
