# M1 twelfth-pass adversarial verification

Branch `m1-gal3-mechanics`, HEAD 3b3ad27, the response to M1-F30..F33. Agent verify-M1-013.

**src/ and docs/claims/ are unchanged.** `git diff 0ebd5aa 3b3ad27 -- src docs/claims` is empty (also
unchanged since 8e1ecae). 3b3ad27 touches only tests/conftest.py, tests/test_gal3.py,
tests/test_imu_dynamics.py and tests/test_mutation.py.

References are independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp` DOP853), mpmath (40 to 60
digits), Python `fractions` (exact rational arithmetic), sympy and gtsam 4.3.0. No reference imports
`src` or `tests` (the backstop literal set was read by parsing conftest.py text, not by import). I
re-implemented these myself (`_p12/ref.py`, `_p12/backstop.py`, `_p12/c2libm.py`):

- the Gal(3) adjoint as `vee(X e_k^ X^-1)` with `np.linalg.inv`, and ad as the commutator;
- exp as `expm(xi^)`, log as scipy Rotation log + 60-term Gamma series + linear solves;
- a 40-digit Rodrigues and 60-digit Gamma coefficients;
- the navigation ODE and the increment ODE (DOP853, rtol = atol = 1e-13), with non-constant
  sinusoidal (omega(t), a(t));
- the free-fall kinematics of the backstops, independently: DOP853, hand-built 5x5 products and gtsam
  `NavState` composition;
- an exact-rational evaluation of c2 on src's double sin inputs, with per-point +-1 / +-2-ulp sin
  perturbations;
- a mutation runner (`_p12/mut.py`). It makes real textual edits on a temp copy of src/ + tests/ +
  conftest.py + pyproject.toml. Each edit must match exactly once (all 147 edit lists passed the
  check), and edits are applied in sequence. Each mutant is judged by the full
  `tests/test_gal3.py tests/test_imu_dynamics.py` run, nothing deselected.

**Run hygiene.** Every pytest / mutant run was under `timeout -k 10 900` (suite: 5400) with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Each mutant took 33 to 92 s. **No mutant
timed out, so no result is KILLED-BY-TIMEOUT.**

Equivalence rules, as stated in the brief:

- rule 1: effect <= 1/10 of the tightest claim tolerance;
- rule 2: within 8 ulp and no worse than clean;
- rule 3: a written proof;
- rule 4 (amended 2026-10-16): excess over clean sup-norm error <= 2 eps max|value|.

The gate requires zero surviving non-equivalent mutants.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C66 (brief only) | **REFUTED.** Over 10^4 phi (\|phi\| in (1e-3, 1)): max \|Gamma1 - Jr\| = **0.911** (min 9.2e-4). Gamma1 is the LEFT Jacobian: \|Gamma1 - Jl\| <= 3.8e-14. Against DOP853 (200 constant (omega, a), both Dt signs): dv = Dt Jr(omega Dt) a is off by **1.44**, while Dt Jl(omega Dt) a = Dt Gamma1 a matches to 1.95e-14. |
| M1-F30 | **FIXED for the named survivors.** H_LOW is killed by `mag_max` (11.41 to 11.50 < 14) in 9 tests. All 6 CLIP + H_LOW compounds are killed. With every random-g guard off, all 6 are killed by the free-fall backstops. |
| M1-F31 | **FIXED for the named survivors.** H_ODD_DS is killed by `parity` (0). PFLIP_UNC + H_ODD_DS is killed by the C34 backstop (test_gal3.py:1495, 72.24), with guards on and off. GNORM_UNC + H_MAG2 + magnitude guard off is killed by the same backstop (18.35). |
| M1-F32 | **FIXED.** The lift time pins are exact: time 5e-13 (S43b), 1e-12 N33/2 (S43d), 1e-13 (S43) and even 1e-300 are all killed at test_gal3.py:1739 (C38) and :1896 (C40), `lift_not_exact` = 500. |
| M1-F33 | **FIXED and robust across libm.** See "c2 pin" below. |
| Gravity-fixture CLASS | **NOT CLOSED.** Two in-domain region defects survive **with every guard ON**: a deadzone on small components (M1-F34) and a magnitude-band defect in the literal-set gap \|g\| in (14.59, 18.70) (M1-F35). A rotation-coupled g defect survives the backstops, which all have omega = 0 (M1-F36, advisory: it needs every guard off). |
| Fresh sweep | **147 mutants** (64 gravity-class, 83 src): **118 killed, 29 survived.** Of the survivors, 6 are equivalent and **23 are NON-EQUIVALENT**. |
| Harness | The clean run kills all 160 harness mutants and passes the unmutated copy |
| Clean suite | `M1_MUTATION_WORKERS=4 pytest -q tests/`: **235 passed** in 154.8 s, exit 0 |
| **Gate** | **FAIL** |

## M1-C66: REFUTED

The claim: Gamma1(phi) = sum_k [phi]x^k/(k+1)! equals the RIGHT Jacobian
Jr(phi) = I - (1 - cos th)/th^2 [phi]x + (th - sin th)/th^3 [phi]x^2, and dv = Dt Jr(omega Dt) a.

The series sum_k K^k/(k+1)! = I + kappa1 K + kappa2 K^2 has a PLUS sign on the K term, so it is the
LEFT Jacobian Jl. The right Jacobian is Jr(phi) = Jl(-phi) = Gamma1(phi)^T. The two differ by
2 kappa1 [phi]x, which is nonzero for every phi != 0.

Measurements (seed 66066):

- 10^4 phi (uniform \|phi\| in (1e-3, 1), random axis):
  - Gamma1 (own 40-term series) vs Jr: max 0.911, min 9.2e-4;
  - Gamma1 vs Jl: 3.8e-14.
- 200 DOP853 solves of Rdot = R[omega]x, vdot = R a, with constant (omega, a), Dt in +-(0.2, 1) and
  \|omega Dt\| in (0.05, 1):
  - dv = Dt Jr a: off by 1.44;
  - dv = Dt Jl a = Dt Gamma1 a: 1.95e-14. This is also C42 as stated (Gamma1).
- Minimal counterexample, phi = e3 (theta = 1): Gamma1[1, 0] = +0.4597 = 1 - cos 1, Jr[1, 0] = -0.4597.

My sweep's K16 (increment_blocks with gamma1(-phi), i.e. Jr) and F24 (gamma1 with -kappa1) are both
killed (C42; C18 and 9 others). **M1-C66 must not enter the yaml.**

## M1-F30..F33 rulings (replays as real exactly-once edits)

| replay | result | failing assertion and value |
|---|---|---|
| H_LOW (\|g\| ~ U(2, 11.5)) | KILLED | conftest.py:375 `mag_max` 11.41..11.50 in 9 tests |
| CLIP_{LIFT, UNC, ELEM, GEN, TRUTH, FORST} + H_LOW | all 6 KILLED | `mag_max`, 9 tests each |
| same + ALL_GUARDS_OFF | all 6 KILLED | backstops: LIFT C31 :1280 9.24, C35 :1559, C36 :1658 11.55; UNC C34 :1495 15.25; ELEM C32 :1375 / C33 :1427 9.24; GEN C32 :1376 9.24; TRUTH C31 :1281 9.24; FORST C37 imu:117 / C43 imu:592 9.24 |
| H_ODD_DS | KILLED | conftest.py:420 `parity` 0, 4 tests |
| PFLIP_UNC + H_ODD_DS (also + ALL_GUARDS_OFF) | KILLED | C34 backstop :1495 72.24 (and parity) |
| GNORM_UNC + H_MAG2 + magnitude guard off | KILLED | C34 backstop :1495 18.35 |
| S43b (lift time 5e-13), S43d (1e-12 N33 / 2) | KILLED | :1739 and :1896 `lift_not_exact` 500 |
| S43 (1e-13), own S43e (1e-300) | KILLED | same |

**Are the backstop expected values independent of src, and correct?** Yes.

- `freefall_cases` and every reference expression appended to C31..C38 and C43 are numpy written in
  the tests: `mat5`, the constant-specific-force kinematics, and hand-built 5x5 products. Only the
  function under test comes from src. The C32 generator check builds the 5x5 by hand and exponentiates
  with scipy.
- I re-derived each expected value independently on all 23 x 3 x 2 instances (`_p12/backstop.py`, own
  random keyframes):

  | expected value | independent check | error |
  |---|---|---|
  | (v_j, p_j) free fall | DOP853 | 2.2e-15 relative |
  | D = (I, a Dt, a Dt^2/2, Dt) | expm(Dt (0, a, 0, 1)^) | 4.4e-16 |
  | C34 ref (R_j, v_j, p_j + g Dt^2, Dt) | own H X_i D | 7.1e-15 |
  | C35 ref p_j - v_j Dt | own H' X_i D T(-Dt) | 5.3e-15 |
  | C36 ref p_j + g Dt t_i, time t_i | own T H' X_i D | 7.1e-15 |
  | C38 T(-Dt) H' T(Dt) = (I, g Dt, +g Dt^2/2, 0) | own | 0.0 |
  | C38 W_g (T X_i T) U_g | own | 3.6e-15 |
  | C33 G(s1) G(s2) X_i D | own | 4.4e-15 |
  | C37 gtsam W Phi U | gtsam | 3.6e-15 |

- `assert_backstop_set_covers` holds on the literal set: octant counts 3/2/3/2/2/4/4/3; \|g\|
  1.985..19.978; axis margins +19.0 / +19.4 / +19.6 and -19.9 / -18.9 / -19.3.

**Does every g-dependent src function reach a backstop?** Yes:

- gravity_lift: C31, C36, C38, C43;
- gravity_lift_uncorrected and endpoint_uncorrected: C34;
- gravity_element: C32, C33;
- gravity_generator: C32;
- endpoint_right_time_shift: C35;
- forster_endpoint: C37, C43;
- integrate_truth: C31.

The g defaults are pinned by `bad_defaults` (C53). **But** every backstop instance has omega = 0 and
dR = I (see M1-F36).

## The gravity-fixture class: NOT closed (M1-F34, M1-F35, M1-F36)

I tried to build defects that are the identity on all 23 literal vectors (and on G) yet wrong on the
claims' stated domain. These are not point-avoiding: each acts on an open, natural region of the
domain that the literal set does not sample.

1. **Deadzone (M1-F34).** The edit is `g = np.where(np.abs(g) < 0.25, 0.0, g)`.
   - The smallest nonzero \|g_k\| in G_BACKSTOP_SET is 0.3, and G's zero components map to zero, so the
     deadzone is the identity on every literal vector. On the stated domain it zeroes any component
     below 0.25: about 9.3 % of uniform-direction, \|g\| ~ U(2, 20) draws. The effect is up to 0.25
     \|Dt\| in v, which is non-equivalent.
   - DZ alone is killed at all 6 sites, by the random-g blocks only: C31 0.0087, C34 0.0052, C32 0.217
     / 0.237, C33 2.13, C37 0.233.
   - The masking fixture FZ (`np.where(|g| < 0.25, copysign(0.25, g), g)` on rand_gravity; RNG stream,
     signs, octants, maxima and magnitudes are essentially unchanged) passes **every guard**. No guard
     constrains small components.
   - **DZ_{LIFT, UNC, ELEM, GEN, TRUTH, FORST} + FZ: all 6 SURVIVE with every guard on.**
2. **Magnitude band (M1-F35).** The edit is `if 14.7 < |g| < 18.6: g = 1.05 g`.
   - The literal magnitudes have a gap (14.591, 18.704); 21.7 % of the stated distribution lies in
     (14.7, 18.6). BAND alone is killed at all 6 sites (C31 0.250, C34 0.150, C32 0.856 / 0.911, C33
     15.1, C37 0.855).
   - The masking fixture FB2 folds (14.7, 18.6) into (18.6, 20). It keeps mag_max >= 18.6 and
     mag_ratio high, and passes every guard.
   - **BAND_{all 6 sites} + FB2: all 6 SURVIVE with every guard on.** My first band fixture (FB, which
     folds the band down to (2, 5.9)) was caught by `mag_ratio` (1.50) on a 4-g dataset; with every
     guard off it survives as well (BAND_LIFT + FB + guards off).
3. **Rotation-coupled gravity (M1-F36, advisory).** Every free-fall instance has omega = 0, so
   D_ij = (I, ...).
   - The edit (in endpoint) is `g = G_DEFAULT + dR(D_ij) (g - G_DEFAULT)`. It is the identity at
     g = G and on every backstop instance (dR = I), and wrong for any g != G with rotation.
   - Alone it is killed (C31 10.16, C36 46.5, C38 14.3, C43 10.16).
   - With rand_gravity returning G, ALL_GUARDS_OFF, and the dataset count/parity guards off, it
     **SURVIVES.** (With only ALL_GUARDS_OFF it is killed by the 4-distinct-g dataset guard,
     conftest.py:403.) No backstop exercises g together with a rotating increment.
   - Advisory only: with the guards on it is killed. The defect is contrived, since it references
     G_DEFAULT.

**Reasoning on closure.** A literal set can only kill a region-restricted defect whose region contains
a literal point.

- At the top of the range the domain is bounded (\|g\| < 20), and F30's remedy covers it.
- At the bottom of each component (\|g_k\| -> 0) and between literal magnitudes, the domain is NOT
  bounded away from unsampled regions. Natural predicates on the domain ("a component is small", "the
  magnitude is in a band") pick out regions with no literal point.
- So the class is closed only for defects whose active region meets the set. Short of an owner-defined
  resolution, no finite set closes it.

**Concrete remedy (both halves):**

- literal vectors with one component exactly 0 and with components 1e-3, 1e-2 and 0.1 on each axis
  and sign (for example (1e-3, 9.0, -4.0));
- a magnitude ladder with no gap wider than ~1 m/s^2 over (2, 20);
- random-g guards requiring min over the block of min_k \|g_k\| <= 0.02 \|g\|, and every one of ~6
  magnitude bins of (2, 20) occupied;
- for M1-F36, one backstop with rotating D. Use D = expm(Dt (omega, a, 0, 1)^) by scipy and
  reference expm(Dt zeta_g^) X_i D, with the 5x5 matrices built in the test, on the literal set.

## c2 pin (M1-F33): robust

The pin evaluates the 60-digit reference on src's own double sin values. I re-ran that measurement in
exact rational arithmetic, and also with every sin and sin(theta/2) input moved independently by a
random -1/0/+1 or -2..+2 ulp, as a non-correctly-rounded libm would (160 trials, 2 seeds).

| | this host | 160 perturbed trials |
|---|---|---|
| clean sup | 9.917e-13 | max **1.0246e-12**, median 9.5e-13 |
| F20 rewrite sup | 1.683e-12 | min **1.2867e-12**, median 1.46e-12 |

The pin is 1.2e-12.

- Clean never exceeds the pin. That is 1.17x headroom over the worst trial.
- F20 is never under the pin. The worst-trial margin is 1.07x.

Faithful libm variation moves which double inputs are hit, but the statistic measures only src's
IEEE +, *, / rounding on those inputs, and its sup is stable at about 1.0e-12. Robust. The 1.07x F20
margin is narrow but held in all 160 trials.

## Fresh mutation sweep (147 mutants)

The sweep has four spec files:

- `spec_g.py` (49): replays, deadzone, band and rotation-coupled hunts;
- `spec_g2.py` (10): band fixture FB2 and guard-off compounds;
- `spec_s.py` (83): 59 gal3 edits (F01..F60) and 23 imu edits (K01..K23);
- `spec_r.py` (5): the structural_residual sign family.

Result: **118 killed, 29 survived.**

### NON-EQUIVALENT SURVIVORS (23)

| id | edit | measured effect | finding |
|---|---|---|---|
| **FZ** (fixture) | rand_gravity components pushed out of (-0.25, 0.25) | passes every guard; masks DZ | M1-F34 |
| **DZ_{LIFT, UNC, ELEM, GEN, TRUTH, FORST} + FZ** (6) | `g = np.where(np.abs(g) < 0.25, 0.0, g)` | alone: C31 0.0087, C34 0.0052, C32 0.217 / 0.237, C37 0.233 | M1-F34 |
| **FB2** (fixture) | \|g\| band (14.7, 18.6) folded into (18.6, 20) | passes every guard; masks BAND | M1-F35 |
| **BAND_{LIFT, UNC, ELEM, GEN, TRUTH, FORST} + FB2** (6) | `if 14.7 < \|g\| < 18.6: g = 1.05 g` | alone: C31 0.250, C34 0.150, C32 0.856 / 0.911, C37 0.855 | M1-F35 |
| **BAND_LIFT + FB + all guards off** | as above | the backstop has no point in the band | M1-F35 |
| **RC_ENDPT + FG + all guards off** (incl. the dataset count/parity guards) | endpoint g -> G + dR (g - G) | alone: C31 10.16 | M1-F36 (advisory: guards on kill it) |
| **FG + all guards off** | rand_gravity = G, guards deleted | masks RC_ENDPT | M1-F36 |
| **F13** | structural_residual `abs(X[3, 3]) - 1.0` | X33 = 0.5: clean 0.5, mutant 0.0 (unflagged) | M1-F37 |
| **SR1** | `X[4, 4] - 1.0` (signed) | X44 = 0.5: 0.5 -> 0.0 | M1-F37 |
| **SR2** | `X[3, 0:3].max()` (signed) | X[3, j] = -0.5 unflagged | M1-F37 |
| **SR3** | `X[4, 0:4].max()` (signed) | X[4, j] = -0.5 unflagged | M1-F37 |
| **SR4** | `(R.T @ R - I).max()` (signed) | R = 0.95 I: residual 0.143 vs clean 0.143 (det) but the orth term is lost (0.0975 -> 0) | M1-F37 |
| **SR5** | `1.0 - det(R)` | R = 1.1 I: residual 0.331 -> 0.21; det > 1 is never seen by the det term | M1-F37 |

On the claim's own domain (products of Gal(3) elements), the structural_residual mutants differ only
at rounding level. They are counted non-equivalent because `structural_residual` is the C02
instrument, and its defect-detection behaviour is asserted by the C02 test itself ("the diagnostic
must actually see each structural defect"). Under these mutants it is blind to every negative
deviation, and the probes are all positive (+0.5, 1.5, x1.1, det -1).

### EQUIVALENT SURVIVORS (6)

Effects were measured with `_p12/eqv.py`, which loads the mutated gal3 in-process and compares it with
clean and with a 40-digit Rodrigues or an exact rational product.

| id | edit | measured effect | rule |
|---|---|---|---|
| F04 | so3_exp Taylor switch at 0.5 THETA_SMALL | [0.025, 0.05), 20000 phi: sup 1.1e-16 vs clean 9.99e-16 (more accurate) | rule 2 |
| F04b | so3_exp Taylor switch at 1e-3 | [1e-3, 0.05): sup 1.1e-16 vs clean 9.99e-16 | rule 2 (as pass-11 J01/J02) |
| F08b | so3_log small branch at `n < 1e-9` | theta in [2e-10, 2e-9), 20000 R: change 0.0; relative error 1.63e-16 = clean | rule 2 |
| F43 | endpoint `compose(left, compose(X_i, D))` | 3000 instances vs exact rational: sup 3.8e-15 vs clean 5.6e-15 (2 eps max\|v\| = 1.28e-14) | rule 2 / rule 4 |
| F57 | so3_log `qw <= 0.0` | differs only at qw = 0 exactly (theta = pi), where both signs are valid logs (C22: two-valued at pi) | rule 3 |
| F58 | Shepperd branch `>=` | only exact ties: 4000 tie rotations, sup 8.9e-16 = clean | rule 2 |

### Killed (118)

- **Gravity-class replays (21):** listed in the F30..F33 table, plus S43, S43e and DZ/BAND alone (12)
  as given above.
- **FB (1):** `mag_ratio` 1.50.
- **RC_ENDPT and RC_ENDPT + FG + ALL_GUARDS_OFF (2):** the latter at conftest.py:403, the
  4-distinct-g dataset guard.
- **gal3:**
  - F01 skew y: 20 tests
  - F03 so3_exp c t4/721: C42 imu:408 2.9e-14
  - F03b: C42 band
  - F08 log small 1e-6: C22
  - F10, F11 compose: 15 and 13 tests
  - F12 inverse: C05
  - F14 / F14b residual short rows: C02 `{'row4[3]': 0.0}`
  - F15, F16 ad: 5 tests
  - F17, F18 Ad: 6 tests
  - F19 Ad_apply: C11
  - F22, F23 Taylor drops: C24
  - F23b kappa2 /120.0001: C24 :694 1.04e-10
  - F23c kappa1 /24.0000001: C24 :721 1.64e-12
  - F24 gamma1 with -kappa1 (Jr): 10 tests
  - F25, F26, F27, F28, F29
  - F30, F31, F32 cayley
  - F33 cay_denominator + : C26 :905 248.1
  - F34, F35, F36, F36b cayley residual and symbolic: C30
  - F37 time_shift: 8 tests
  - F38, F39 \|Dt\|
  - F40 generator w
  - F41 NaN keyframe
  - F42 Dt + t_i: C36
  - F44, F45, F46, F47
  - F48 navstate swap
  - F49, F50
  - F51 lift swap
  - F52 perm
  - F53 G_DEFAULT +1 ulp: imu:925 1.78e-15
  - F54 keyframe tol 2e-12: C36 `[2e-12, -2e-12]`
  - F55 THETA_SMALL 4e-2: imu:500 / :722
  - F55b 6e-2
  - F56 kappa3 with 1 - cos: C24 :659 3.6e-14
  - F59 unpack \|t\|: 22 tests
  - F60 wedge alpha at (3, 3): 9 tests
- **imu:**
  - K01..K04 bias conventions: C44, C54, C55
  - K05 batch c /721: C42 2.9e-14
  - K06, K08 skew batch
  - K07, K12, K15, K19 \|dt\| / \|Dt\|
  - K09 longer streams accepted: imu:856
  - K10 truth update order
  - K13 Euler second-order rotation: C41 :215 1.99
  - K14 semi-implicit Euler: C41 2.57
  - K16 dv with Jr: C42
  - K17 generator w
  - K18 chain order
  - K20 f with R E
  - K21 drop t0: C45
  - K22 batch s drop t6
  - K23 \|b\|

## Property tests and the hunt for spurious behaviour (own re-implementation)

- 10^4 triples (seed 13013, \|t\| ~ 3 N):
  - compose vs matmul 3.6e-15; associativity 1.4e-14;
  - inverse (vs `np.linalg.inv` and both products) 2.8e-14; time entries exact.
- 10^4 X: Ad vs own `vee(X e_k^ X^-1)` 2.4e-15 relative (Ad and Ad_apply).
  - Homomorphism 1.5e-15 relative; Ad_{X^-1} vs inv 1.2e-14; Ad-ad 4.0e-15 relative.
  - det(Ad) 4.7e-15; last row exactly e10; tr(ad) 0.
- 10^4 xi with \|phi\| log-uniform in (1e-6, 2 pi - 0.3): exp vs expm 5.5e-14 relative; time 0.0.
- 10^4 xi with \|phi\| in (0.05, pi - 1e-3): log(exp xi) 3.3e-15. 10^4 X with R = Exp(5 N) plus 8
  rotations at pi - 10^-k: exp(log X) 1.5e-13.
- 10^4 Cayley: closed form vs 5x5 definition 8.5e-16 relative; orthogonality/det/time 1.1e-15.
  - \|c\| = 2 tan(theta/2) and axis: 2.2e-15.
- **Spurious feasible points:** det(I - [phi]x/2) = 1 + \|phi\|^2/4 (C26, 6.6e-16 relative) > 0, so
  the 15 degree-2 Cayley equations have a unique solution. Solving them from scratch on 2000 xi
  reproduces src cay to 1.8e-15.
- Keyframe guard: 0, -0.0, +-1e-12 and 5e-13 accepted; nextafter(1e-12, inf), NaN, +-inf and 1e-6
  refused.

## Per-claim verdicts

Scripts: `_p12/claims_a.py` (C01..C30), `_p12/claims_b.py` (C31..C55) and `_p12/c4950.py`. They use
36 DOP853 instances with non-constant (omega, a), Dt in {+0.45, -0.35, +0.8}, and g = (0, 0, -9.81)
plus 27 random g. The random g have components -15.3..15.9 and \|g\| 2.77..19.5.

| claim | verdict | measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | 10^4: 3.6e-15 |
| C02 | CONFIRMED | 2.8e-15; residual flags reflection 1.29, row3/row4/(3,3)/(4,4) +0.3 (but see M1-F37 for negative deviations) |
| C03 | CONFIRMED | 1.4e-14 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 2.8e-14; time entry exactly 0 |
| C06 | CONFIRMED | max commutator 33.0 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0; wedge = own hat exactly |
| C09 | CONFIRMED | commutator 1.8e-15; vs own ad 0.0 |
| C10 | CONFIRMED | 3.6e-15 / 7.3e-15 |
| C11 | CONFIRMED | 2.4e-15 relative |
| C12 | CONFIRMED | 1.5e-15 relative |
| C13 | CONFIRMED | 1.2e-14 |
| C14 | CONFIRMED | 4.0e-15 relative |
| C15 | CONFIRMED | 2000 xi (\|xi\| <= 3): 6.4e-15 |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.7e-15; trace 0.0 |
| C18 | CONFIRMED | 10^4 vs 60-term series, theta in (1e-3, 2 pi - 0.3): 1.9e-14 |
| C19 | CONFIRMED | 10^4 vs expm: 5.5e-14 relative |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | G1inv G1 - I 8.9e-16 (10^4, theta < pi - 1e-3); c2 at pi - 10^-k vs 50-digit 1.1e-17; log vs own 1.2e-14 |
| C22 | CONFIRMED | 3.3e-15; small angle (1e-10..1e-2) relative 4.7e-16; \|phi\| = 3.5 wraps to 2.7832 = 2 pi - 3.5 |
| C23 | CONFIRMED | 1.5e-13 (incl. pi - 10^-8) |
| C24 | CONFIRMED | det 1.0 / 0.40528 / 2.2128e-3 / 2.3e-65; cond(pi) 1.570796; THETA_SMALL 0.05; literal polynomial 1.10e-16 / 2.18e-16 / 2.20e-16 / 2.17e-16; at 8 points vs 60-digit 6.9e-13 (1e-11); just below switch 7.75e-13 / 2.58e-13 / 1.03e-13 / 1.55e-13; theta < 0 raises |
| C25 | CONFIRMED | 8.5e-16 relative |
| C26 | CONFIRMED | 6.6e-16 relative; matrix = I5 - xi^/2 |
| C27 | CONFIRMED | 1.1e-15; time exact |
| C28 | CONFIRMED | 2.2e-15; 0.51068, 1.09260, 3.11482, 28.20284, 43171.56 |
| C29 | CONFIRMED | \|d\|/s^3 0.035916..0.035921 (s = 0.05..0.005); remainder/s^5 0.002401..0.002402 (constant) |
| C30 | CONFIRMED | 15 expressions, 25 symbols, total degree 2; symbolic 2.2e-16; numeric 2.2e-16 |
| C31 | CONFIRMED | DOP853, random g, both Dt signs: 1.0e-14 relative (src and own product); time 0; discrete recursion identity 4.3e-15 (first-order gap to DOP853 1.1e-4) |
| C32 | CONFIRMED | 10^4: 6.4e-16 relative; generator exact; DOP853 1.0e-14 |
| C33 | CONFIRMED | 10^4: 1.2e-14 relative |
| C34 | CONFIRMED | time Dt, position p_j + g Dt^2, R/v correct: 1.0e-14 |
| C35 | CONFIRMED | 7.9e-15; time 0 |
| C36 | CONFIRMED | 1.0e-14; guard as above |
| C37 | CONFIRMED | gtsam W Phi U vs DOP853: 2.0e-14 |
| C38 | CONFIRMED | W_g = T(-Dt) H' T(Dt), T(Dt) U_g = D, forms agree: 1.0e-14 |
| C39 | CONFIRMED | 10^4 vs gtsam compose: 8.9e-16 |
| C40 | CONFIRMED | automorphism 3.6e-15; lift = conjugation 0.0; Phi(I) = I exactly |
| C41 | CONFIRMED | 10^4 vs expm, Dt in [-5, 20]: 3.1e-14 relative; own DOP853 1.5e-14; Euler err x n/Dt 1.3031 / 1.3031 (first order) |
| C42 | CONFIRMED | 10^4: 3.1e-14 relative |
| C43 | CONFIRMED | 2.0e-14 |
| C44 | CONFIRMED | 10^4 incl. gtsam correctGyroscope / correctAccelerometer: 0.0 |
| C45 | CONFIRMED | 10^4 chain 3.6e-15; split vs whole 1.4e-15 / D0 continuation 0.0; ordered scipy product 6.4e-15 |
| C46 / C47 | CONFIRMED | 10^4: 0.0 |
| C48 | CONFIRMED | 10^4: 0.0 |
| C49 | CONFIRMED | NavState retract(1e-3 e3) -> position (1e-3, 0, 0), velocity 0; Gal3.Expmap(1e-3 e3) -> velocity; NavState matrix col 3 = p |
| C50 | CONFIRMED | own P, at identity (Expmap, d ~ s N(0, I10)): s = 1e-5, 20000 draws: 4.6e-10 (1e-9); s = 1e-4: 4.3e-8; unpermuted O(1) (1.34 relative); src P exact |
| C51 | CONFIRMED | 10^4: 4.5e-15 |
| C52 | CONFIRMED | 10^4 compose/*/inverse/between/retract: 8.9e-15 |
| C53 | CONFIRMED | predict vs endpoint 1.4e-17 with g omitted and passed; G_DEFAULT = n_gravity |
| C54 | CONFIRMED | vector [1..6]; halves |
| C55 | CONFIRMED | 10^4: 0.0; S S = I |
| **C66** | **REFUTED** | max \|Gamma1 - Jr\| 0.911; dv error 1.44 (Gamma1 = Jl to 3.8e-14) |

**Totals: 55 CONFIRMED, 1 REFUTED (C66), 0 UNTESTABLE.**

## New findings

- **M1-F34 (medium, fixture + backstop coverage): a small-component deadzone survives with every
  guard on.**
  - The backstop set's smallest nonzero \|g_k\| is 0.3, and no guard bounds how small a random
    component gets.
  - FZ masks `g = np.where(|g| < 0.25, 0, g)` at all 6 g sites.
  - Kill: add literal vectors with a component in {0, +-1e-3, +-1e-2, +-0.1} on each axis. Add a
    random-block guard that some draw has min_k \|g_k\| <= 0.02 \|g\| (clean 400..2000-draw blocks
    trivially satisfy it).
- **M1-F35 (medium, same class): magnitude-band defects in the literal gap (14.59, 18.70) survive
  with every guard on.**
  - The fixture FB2 (fold the band to the top) passes mag_max and mag_ratio.
  - Kill: a literal magnitude ladder with gaps <= 1 m/s^2 over (2, 20), and a guard that every one of
    ~6 equal bins of (2, 20) holds a draw on the >= 400-draw blocks.
- **M1-F36 (advisory): all free-fall backstops have omega = 0, dR = I.**
  - A g x rotation coupling in endpoint survives the backstops when the random fixtures are
    degenerated to G with every guard off. With guards on it is killed.
  - Kill: one literal backstop with rotating D = scipy expm(Dt (omega, a, 0, 1)^), referenced against
    scipy expm(Dt zeta_g^) X_i D, with the 5x5 matrices built in the test.
- **M1-F37 (low, C02 instrument): structural_residual's sign handling is unpinned.**
  - Six sign-blind edits survive: (3,3), (4,4), row 3, row 4, the orth term and the det term.
  - Every C02 defect probe is a positive deviation.
  - Kill: add negative probes (X[3, 3] = X[4, 4] = 0.5, X[3, j] = X[4, j] = -0.5, R x 0.9, a det > 1
    scaling) and require each to be flagged at >= 0.2.
- **Class ruling.** The literal free-fall backstops close the class only for defects whose active
  region contains a literal point. No finite set closes region-restricted defects on regions that the
  domain does not bound away (small components, magnitude gaps). The owner needs to fix a coverage
  resolution, for example "every literal gap in \|g\| and in min_k \|g_k\| below X", to make "closed"
  decidable.
- Advisory: C50's agreement is tested at the identity (Expmap). At a general base point X with
  v != 0, a nonzero time component d[9] moves Gal3 retract's position at first order: 1.2e-4 at
  s = 1e-5. The permutation is valid there only for d[9] = 0 (5.7e-10). The claim's domain text
  could say so.
- Advisory: on [1e-3, 0.05) the Taylor branch of so3_exp (sup 9.99e-16) is less accurate than the
  closed form (1.1e-16). Not a defect.
- Advisory carried over: A4. C52's domain text still says "in every implementation".

## Clean suite

`M1_MUTATION_WORKERS=4 python -m pytest -q tests/` at 3b3ad27: **235 passed in 154.78 s**, exit 0.
That is 55 claim tests, 161 test_mutation (160 mutants plus the unmutated copy) and 19
test_orchestrator. Each mutant run of the 55 claim tests took 33 to 92 s. The scratch `_p12/` was
deleted before commit.

## Gate

**FAIL.** 23 non-equivalent survivors:

- FZ plus 6 DZ + FZ compounds (M1-F34);
- FB2 plus 6 BAND + FB2 compounds, and BAND_LIFT + FB + guards off (M1-F35);
- RC_ENDPT + FG + guards off, and FG + guards off (M1-F36);
- F13 and SR1..SR5 (M1-F37).

M1-F30, F31, F32 and F33 are fixed for every named survivor. Every claim C01..C55 is confirmed.
M1-C66 is false (Gamma1 is the left Jacobian) and must not enter the yaml.
