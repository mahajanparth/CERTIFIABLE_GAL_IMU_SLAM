# M1 tenth-pass adversarial verification

Branch `m1-gal3-mechanics`. The code verified is HEAD 8e1ecae, the response to M1-F22, M1-F23 and
advisories A4/A5. Agent verify-M1-011.

**src/ is unchanged.** `git diff 3f735b8 8e1ecae -- src` is empty. `git diff b763ab2 8e1ecae --stat`
touches only docs/claims/M1.yaml, tests/conftest.py, tests/test_gal3.py, tests/test_imu_dynamics.py
and tests/test_mutation.py. In M1.yaml only the `conventions.gravity` paragraph changed. No claim
statement, domain or test_hint changed.

References are independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp` DOP853), mpmath (40 to
80 digits), sympy and gtsam 4.3.0. No reference imports `src` or `tests`. I re-implemented these
myself:

- the Gal(3) adjoint, as `vee(X e_k^ X^-1)` with `np.linalg.inv`, and in 40-digit mpmath;
- ad as the commutator `vee(A e_k^ - e_k^ A)`;
- the SO(3) log, in 50-digit mpmath (atan2 of the rounded R's skew part and trace);
- the Gamma coefficients, as 60-digit mpmath series, with c2 in 140 digits;
- a 40-digit mpmath Rodrigues;
- the navigation ODE and the 5x5 increment ODE, DOP853 with rtol = atol = 1e-13 and
  non-constant (omega, a);
- a mutation runner (`_p10/mut.py`). It makes real textual edits, each required to match exactly
  once, applied in sequence, on a temp copy of src/ + tests/ + conftest.py + pyproject.toml. Each
  mutant is judged by the full `tests/test_gal3.py tests/test_imu_dynamics.py` run, with no
  assertion removed and nothing deselected. All 118 edits passed the exactly-once check.

I applied the equivalence rules as stated:

- rule 1: measured effect <= 1/10 of the tightest claim tolerance for that quantity;
- rule 2: the mutant's sup-norm error over the affected domain is within 8 ulp AND no worse than
  clean;
- rule 3: a written proof;
- rule 4 (owner decision 2026-10-15): the mutant's sup-norm error exceeds clean's by at most
  2 eps max|value|, AND the mutant stays >= 100x inside the tightest claim tolerance.

As in pass 9, "claim tolerance" includes the bounds written in a claim's test_hint, including the
pass-8 pins. The pass-9 rule-1 rulings that the owner accepted used the same reading.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C64 (brief only) | **REFUTED.** Over 10^4 random pairs, max \|claimed p - (X1 X2)[0:3,4]\| = 8.73 (min 1.4e-3). gtsam Gal3.compose disagrees with it by up to 8.92. The true term is v1 t2, which matches to 3.6e-15. |
| M1-F22 | **FIXED** for N29/N30, both killed by `worst_default`. Pinning was limited to `G_DEFAULT` and `endpoint`'s default. The 8 other `g=G_DEFAULT` parameter defaults are unpinned (**M1-F25**). |
| M1-F23 | **FIXED** for H_HEMI. H_HEMI and both compounds are killed by `signed` in 9 tests. The guard family is still not exhaustive: a fixed-magnitude fixture (**M1-F26**) and an antipodal sign-flip fixture (**M1-F27**) pass every guard and mask g-dependent src defects. |
| A4 / A5 | A5 resolved: the negative-theta raise is checked, and my F49 is killed. A4 was resolved test-side only: C52's domain text still says "in every implementation" (advisory, unchanged). |
| M1-F24 re-ruling (rule 4) | **B1, B2, B3, B3b, B4: all NON-EQUIVALENT.** Each one fails rule 4(b). B3b and B4 also fail 4(a). Numbers below. |
| Harness | Clean run: 101/101 harness mutants killed, plus the unmutated copy |
| Replays | N29, N30, H_HEMI, YABS_LIFT+H_HEMI, YABS_TRUTH+H_HEMI (and YABS_* alone): **all KILLED** |
| Fresh sweep | 118 mutants (incl. replays and hunts): 74 killed, 44 survived. Of the survivors, 21 are equivalent and **23 are NON-EQUIVALENT** |
| Clean suite | `M1_MUTATION_WORKERS=4 pytest -q tests/`: **176 passed in 130.4 s**, exit 0 |
| **Gate** | **FAIL** |

## M1-C64: REFUTED

Claim: the position block of X1 X2 is p1 + R1 p2 + t1 R1 v2, to 1e-12.

The block product gives row 0:3 of column 4 as R1 p2 + v1 t2 + p1 · 1. The coupling is the
FIRST element's velocity times the SECOND element's time (C01, C45), not t1 R1 v2.

Measurement (own scipy rotations, 5x5 numpy products, seed 7064, 10^4 pairs):

- max error of the claimed form 8.73, min 1.4e-3;
- the true form p1 + R1 p2 + v1 t2: 3.6e-15.

Minimal counterexamples:

- X1 = (I, 0, 0, 1), X2 = (I, e1, 0, 0): the product has p = 0, the claim gives e1.
- X1 = (I, e1, 0, 0), X2 = (I, 0, 0, 1): the product has p = e1, the claim gives 0.

Against gtsam `Gal3.compose`.position(): the claimed form is 8.92 off, while v1 t2 is 1.8e-15 off.
M1-C64 must not enter the yaml.

## M1-F22 / M1-F23 rulings

### F22: FIXED as specified; residual is M1-F25

- conftest's `G` is now the literal (0, 0, -9.81).
- C53 calls `gal3.endpoint` without g and pins `gal3.G_DEFAULT` / `imu.G_DEFAULT` to the literal
  and to gtsam's `n_gravity`.

Replays:

- N30 is killed at `assert worst_default <= 1e-9` (9.81);
- N29 is killed there too (1.675e-3);
- my D_endpoint (`endpoint(..., g=-G_DEFAULT)`) is killed (9.81).

**Residual.** Only `endpoint` exercises its default. Every other function with `g=G_DEFAULT` is
always called with an explicit g in the tests:

- `gravity_lift`, `gravity_lift_uncorrected`, `gravity_element`, `gravity_generator`,
  `endpoint_uncorrected`, `endpoint_right_time_shift` and `forster_endpoint` in gal3;
- `integrate_truth` in imu_dynamics. Its only g-less call is in the stream-rejection test, which
  checks no output.

`def f(..., g=-G_DEFAULT)` survives the full suite for all 8 of them (M1-F25).

### F23: FIXED for the half-space; residuals are M1-F26 and M1-F27

H_HEMI, YABS_LIFT+H_HEMI and YABS_TRUTH+H_HEMI are each killed in all 9 span tests, at
`assert signed >= min_signed`, with values -0.0218, -0.0166 and -0.0473. YABS_LIFT and YABS_TRUTH
are killed alone (C31: 9.9988).

**Hunt.** I looked for gravity distributions that pass off_z, perm_gap, spread, comp, span AND
signed by construction, not by seed luck.

| fixture | passes guards? | masked src defect | result |
|---|---|---|---|
| H_QUAD: g_y := copysign(\|g_y\|, g_x) | no, seed-dependent: span 0.0866 on the Dt = +0.4 dataset | YCOPY (g_y -> copysign(g_y, g_x)) | killed (span) |
| H_MAG: 9.81 * rand_axis, skipping the magnitude draw | no: shifts the RNG stream, signed = -4.19 | GNORM | killed (signed) |
| **H_MAG2**: magnitude draw consumed and discarded, return 9.81 * rand_axis | **yes**: the directions are bit-identical to clean, and every guard depends only on direction, or on components bounded below by 9.81 / 20 of clean | **GNORM** (g -> 9.81 g / \|g\|) in gravity_lift, integrate_truth, forster_endpoint | **all 3 SURVIVE** under H_MAG2; each is killed alone (C31 3.145) |
| **H_ODD**: g := -g where g_x g_y g_z < 0 | **yes**: off_z, perm_gap, spread, span and comp are invariant under per-draw sign flips (\|u.u\|, singular values, \|g - Pg\|, \|g\|), and the four kept octants (+++, +--, -+-, --+) give every component both signs | **PFLIP** (g -> -g if g_x g_y g_z < 0) in gravity_lift, integrate_truth | **both SURVIVE** under H_ODD; each is killed alone (C31 12.45) |

**Structural point.** The guards test marginal properties of the direction distribution. A
per-draw nonlinear map that preserves those marginals masks the defect it inverts. Two such maps
are shown above:

- the magnitude is never guarded at all;
- per-draw sign flips are invisible to every guard except the per-component sign marginals.

GNORM is a realistic defect: code that hard-codes the standard magnitude. Kills are proposed under
M1-F26 and M1-F27.

## M1-F24 re-ruling under rule 4

Clean and mutant sup-norm errors, on the affected domain of each mutant, against 40/50-digit
mpmath references (`_p10/band.py`, `_p10/band_log.py`):

- Rodrigues: 3501 angles in [1e-8, 0.05);
- so3_log: 20000 R with theta log-uniform below the domain edge;
- polynomial: 5001 theta in (1e-12, 0.05).

All five still survive the suite at HEAD.

| id | quantity, max\|value\| | clean sup | mutant sup | excess | 2 eps max\|v\| | (a) | tightest claim tolerance | 100x inside needs | (b) | ruling |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 so3_exp c + t4 t2/200000 | R, 1.0 | 9.99e-16 | 1.221e-15 | 2.22e-16 | 4.44e-16 | pass | C42 1.3e-15 (both Rodrigues on [0.04, 0.052]) | <= 1.3e-17 | **fail** | NON-EQUIVALENT |
| B2 batch c + t2^3/200000 | R, 1.0 | 9.99e-16 | 1.221e-15 | 2.22e-16 | 4.44e-16 | pass | C42 1.3e-15 | <= 1.3e-17 | **fail** | NON-EQUIVALENT |
| B3 so3_log n < 3e-8 (theta < 6e-8) | phi, 5.89e-8 | abs 1.32e-23 / rel 3.77e-16 | abs 1.99e-23 / rel 5.05e-16 | abs 6.6e-24 / rel 1.3e-16 | 2.62e-23 (abs), 4.44e-16 (rel) | pass | C22 small-angle relative 1e-15 | rel <= 1e-17 | **fail** | NON-EQUIVALENT |
| B3b so3_log n < 5e-8 (theta < 1e-7) | phi, 9.79e-8 | abs 2.65e-23 / rel 4.30e-16 | abs 9.26e-23 / rel 1.03e-15 | abs 6.6e-23 / rel 6.0e-16 | 4.35e-23 / 4.44e-16 | **fail** | C22 relative 1e-15 (my set exceeds it: 1.03e-15) | rel <= 1e-17 | **fail** | NON-EQUIVALENT |
| B4 kappa1 t4/720.00002 vs stated polynomial | kappa1, 0.5 | abs 5.46e-17 / rel 1.09e-16 | abs 2.91e-16 / rel 5.83e-16 | abs 2.37e-16 | 2.22e-16 | **fail** | C24 literal polynomial 1e-15 relative | rel <= 1e-17 | **fail** | NON-EQUIVALENT |

Why rule 4 cannot rescue these:

- Condition (b) cannot hold for ANY mutant on these quantities. The clean errors themselves sit
  1.3x (C42), 2.3x (C22) and 4.5x (C24) inside the pass-8 pins, not 100x. A mutant can stay 100x
  inside only if it improves on clean by roughly 100x.
- Under the alternative reading, where only non-pin bounds count:
  - B1/B2 still fail (b) against C42's 1e-14 Rodrigues bound (100x needs 1e-16; measured 1.22e-15);
  - B3/B3b fail against C22's 1e-15 relative bound;
  - B4 fails against C24's 1e-11 only on (a).
- So no reading of rule 4 makes all five equivalent.

The rule as written is a sub-ulp floor only where the claim tolerance is >= 100x above clean. The
F24 bands are exactly where it is not. Either the pins move to clean + 1 to 2 ulp (the kills
proposed in pass 9), or the owner amends rule 4(b).

My F19 (below) is a sixth member of this class.

## Harness audit

- The 10 new harness entries match the pass-9 survivors. The compound entries
  (`(H_HEMI, YABS_LIFT)` etc.) go through the unchanged `_edits` / `_mutate_and_run` path. That path
  counts each `old` in the already-modified text and returns `code = None` unless the count is 1.
- Expected-assertion substrings are the new checks: `worst_default`, `signed` and
  `not not_rejected`.
- The clean run kills all 101 harness mutants and passes `test_unmutated_copy_passes`: 102 of the
  176 collected tests are in test_mutation.py.
- Concurrency logic is unchanged since pass 9: per-mutant directories, loud failure modes.

## Fresh mutation sweep (118 mutants)

### NON-EQUIVALENT SURVIVORS (23)

| id | edit | measured effect | why non-equivalent |
|---|---|---|---|
| **D_gravity_lift** | `def gravity_lift(Dt, g=-G_DEFAULT)` | a caller omitting g gets gravity pointing UP: 2 \|g\| = 19.62 in the H' velocity block | default never exercised (M1-F25) |
| **D_gravity_lift_uncorrected** | same, `gravity_lift_uncorrected` | as above | M1-F25 |
| **D_gravity_element** | same, `gravity_element` | as above | M1-F25 |
| **D_gravity_generator** | `gravity_generator(g=-G_DEFAULT)` | zeta_g has g flipped | M1-F25 |
| **D_endpoint_uncorrected** | same, `endpoint_uncorrected` | as above | M1-F25 |
| **D_endpoint_right_time_shift** | same, `endpoint_right_time_shift` | as above | M1-F25 |
| **D_forster_endpoint** | `forster_endpoint(..., Dt, g=-G_DEFAULT)` | as above | M1-F25 |
| **D_integrate_truth** | `integrate_truth(..., g=-G_DEFAULT, ...)` | the ground-truth integrator's default gravity points up | M1-F25 |
| **H_MAG2** (fixture) | rand_gravity returns 9.81 * rand_axis (RNG stream preserved) | passes every guard | masks GNORM (M1-F26) |
| **GNORM_LIFT + H_MAG2** | gravity_lift g -> 9.81 g/\|g\| | survives; alone killed (C31 3.145) | M1-F26 |
| **GNORM_TRUTH + H_MAG2** | integrate_truth, same | survives; alone killed (C31 3.145) | M1-F26 |
| **GNORM_FORST + H_MAG2** | forster_endpoint, same | survives; alone killed (C31 3.145) | M1-F26 |
| **H_ODD** (fixture) | g := -g where g_x g_y g_z < 0 | passes every guard | masks PFLIP (M1-F27) |
| **PFLIP_LIFT + H_ODD** | gravity_lift g -> -g if g_x g_y g_z < 0 | survives; alone killed (C31 12.45) | M1-F27 |
| **PFLIP_TRUTH + H_ODD** | integrate_truth, same | survives; alone killed (C31 12.45) | M1-F27 |
| **B1, B2, B3, B3b, B4** | pass-9 bands | see the rule-4 table | M1-F24 |
| **F19** | kappa2 Taylor `t4 / 5040.0001` | vs the stated polynomial: rel 3.58e-16 (clean 2.19e-16), abs excess 2.3e-17 | rule 1: change 1.4e-16 > 1e-16 (1/10 of C24's 1e-15 pin); rule 2: worse than clean; rule 4(a) passes (2.3e-17 <= 7.4e-17) but (b) fails (3.58e-16 vs 1e-17). Same class as B4 (M1-F24) |
| **F20** | c2 closed form as `1/t2 - theta s/(2 omc t2)` (algebraically identical; not the forbidden form) | relative change of c2 1.14e-12 (10^4 theta in [0.05, 0.2]); sup error on [0.05, 0.055] 2.43e-12 vs clean 1.68e-12, [0.055, 0.07] 1.83e-12 vs 1.35e-12 | rule 1: 1.14e-12 > 1e-12 (1/10 of C24's 1e-11); rules 2 and 4 fail (worse by 7.5e-13 relative). M1-F28 |
| **F41** | gamma_coefficients `theta <= THETA_SMALL` | at theta = 0.05 exactly the Taylor branch is used: kappa1 relative 7.75e-13 (clean 1.7e-16); Gamma1 matrix change 1.29e-14; Gamma1inv Gamma1 - I = 1.30e-14 (clean 2.2e-16) | C24 states the closed form at theta = 0.05 ("Taylor ... for theta < 5e-2"). Rule 1: 1.29e-14 > 1e-14 (1/10 of C21's 1e-13); rules 2/4 fail. M1-F29 |

### EQUIVALENT SURVIVORS (21)

Effects are measured with `_p10/dump.py`, which compares mutant and clean outputs on 2x10^4
inputs per quantity, and with `_p10/eqv.py` / `band.py` / `i03b.py`, which measure the sup error
against mpmath on each affected domain.

| id | edit | measured effect | rule |
|---|---|---|---|
| D_imu_reexport | imu `G_DEFAULT = gal3.G_DEFAULT * 1.0` | values identical (pinned to the literal in C53); 0.0 | rule 1 |
| F02 | so3_exp Taylor c `- t4 t2/40320` (next term added) | Rodrigues on [0.04, 0.05): sup 1.1e-16 vs clean 9.99e-16 (better); change 8.9e-16 = 4 ulp | rule 2 (no claim states so3_exp's polynomial) |
| F03 | so3_exp closed-form c as `(1 - cos)/th^2` | sup [0.05, 0.052] 1.1e-16 = clean; [0.2, 3.1] 7.8e-16 vs clean 8.9e-16; change 8.9e-16 | rule 1 (<= 1e-15, 1/10 of C42's 1e-14; on [0.05, 0.052] change 1.1e-16 <= 1.3e-16) and rule 2 |
| I03 | batch c as `2 sin^2(th/2)/th^2` | [0.05, 0.052]: change 1.1e-16, sup equal to clean; [0.2, 3.1]: sup 8.9e-16 vs clean 7.8e-16, change 8.9e-16 | rule 1: <= 1e-15 against C42's 1e-14 (147-angle bound); 1.1e-16 <= 1.3e-16 on the pinned band |
| F05 | so3_log `n < 2e-10` | sup on theta < 5e-10 (3000 R): 0.0 = clean | rule 2 |
| F05b | so3_log `n <= 1e-10` | as F05 | rule 2 |
| F06 | so3_log `qw <= 0.0` | differs only at qw = 0 (theta = pi exactly), where it returns -phi; both are logs of R (exp(-phi) = exp(phi) at \|phi\| = pi); C22's domain is \|phi\| < pi and C23 holds for either | rule 3 |
| F07 | Shepperd branch `>=` | only exact ties; 600 tie rotations (axis (1, +-1, 0)/sqrt2 and (1, 0, +-1)/sqrt2, theta in (2.1, pi)): sup 8.9e-16 = clean | rule 2 |
| F11 | inverse `-Rt p + t (Rt v)` | change 7.1e-15 (2.4e-16 of scale) | rule 1 (C05 1e-10) |
| F17 | Ad_apply `R (w - t u)` | change 7.1e-15 (9e-17 of scale) | rule 1 (C11 1e-9) |
| F24 | log `G1i p - t G1i (G2 u)` | change 5.3e-15; log(exp x) sup 5.2e-15 vs 4.2e-15 | rule 1 (C52 localCoordinates 1e-13 -> 1e-14) |
| F34 | endpoint `compose(left, compose(X_i, D))` | change 1.7e-13 abs at entries up to 7e2 (2.4e-16 relative); vs 40-digit truth 2.8e-14 vs clean 5.7e-14 | rules 1 and 2 (associativity, rounding only) |
| F42 | so3_exp `theta <= THETA_SMALL` | only theta = 0.05 exactly: sup 9.99e-16 = clean | rule 2 |
| I04 | batch `th <= THETA_SMALL` | as F42 | rule 2 |
| F43 | so3_log `tr > -0.5` | changes only tr in (-0.5, 0]: so3_log change 1.8e-15, Gal log 8.7e-15 | rule 1 (C52 1e-13 -> 1e-14; that band lies outside C22's 2.5e-15 band, which is tr <= -0.9) |
| F44 | `qw = s / 4.0` | bit-identical (both are exact power-of-2 scalings) | rule 3 |
| F45 | Ad `R skew(R^T v)` for `skew(v) R` | change 1.8e-15 | rule 1 (C11 1e-9) |
| F47 | unskew reads `-A[2, 0]` for `A[0, 2]` | bit-identical on every dump; on gal(3) (C08's domain) A[0, 2] = -A[2, 0] exactly | rules 1 and 3 |
| F50 | endpoint `left = gravity_element(Dt, g)` | bit-identical: compose(T(-Dt), H') adds only exact zeros and I-products | rules 1 and 3 |
| F53 | small-angle return `* (1 + n^2/4)` | n < 1e-10 gives a factor 1 + 2.5e-21, which rounds to 1; 0.0 | rule 1 |
| I27 | euler_increment `n * dt` for `int(n) * dt` | identical for every integral-valued n, the only n any claim uses | rule 3 |

### Killed (74)

Each line gives the mutant, then the failing test and value.

**Replays and hunts:**

- N30: C53 `worst_default` 9.81
- N29: C53 `worst_default` 1.675e-3
- H_HEMI: `signed` -0.0218 (9 tests)
- YABS_LIFT: C31 9.9988
- YABS_TRUTH: C31 9.9988
- YABS_*+H_HEMI: `signed` (9 tests)
- H_QUAD: span 0.0866 (seed-dependent)
- YCOPY_LIFT: C31 9.9988
- YCOPY_TRUTH: C31 9.9988
- YCOPY_*+H_QUAD: span
- H_MAG: `signed` -4.19 (seed-dependent)
- GNORM_LIFT: C31 3.145
- GNORM_TRUTH: C31 3.145
- GNORM_FORST: C31 3.145
- GNORM_*+H_MAG: `signed`
- PFLIP_LIFT: C31 12.45
- PFLIP_TRUTH: C31 12.45
- D_endpoint: C53 9.81

**gal3:**

- F01 so3_exp Taylor without th^6: C42 1.49e-13
- F04 THETA_SMALL 4e-2: C24 `0.04 == 0.05`
- F08 Shepperd branch-2 sign: C22 4.22
- F09 `arcsin(n)`: C22 8.9e-15 (band)
- F09b `arccos(qw)`: C22 2.1e-8
- F10 compose `t1 R1 v2` (the C64 form): C01 9.70
- F12 residual without (4, 4): C02 `{'diag4': 0.0}`
- F13 residual row 4 [0:3]: C02 `{'row4[3]': 0.0}`
- F14 ad +alpha: C09 15.3
- F15 ad (6:9, 9) = w: C09 11.8
- F16 Ad (6:9, 9) = p: C11 29.0
- F18 kappa2 `t4/5040.5`: C24 7.4e-13
- F21 kappa3 with 1 - cos: C24 band
- F22 Gamma1^-1 +P/2: C21 2.00
- F23 exp -alpha G2 u: C15 2.40
- F25 Cayley alpha u: C25 4.22
- F26 cay_matrix swapped: C25 8.70
- F27 constraint residual without alpha u/2: C30 4.20
- F28 symbolic alpha u: C30 2.20
- F29 time_shift -dt: C31 1.0
- F30 uncorrected lift sign: C34 2.45
- F31 gravity_element +Dt: C32 1.99
- F32 generator +1: C32 9.71
- F33 keyframe `< tol`: C36 `[1e-12, -1e-12]`
- F35 right shift +Dt: C35 25.9
- F36 Forster Dt\|Dt\|: C31 (Dt = -0.3) 1.52
- F37 NavState compose without R1: C39 6.69
- F38 phi \|Dt\|: C40 17.5
- F39 lift_navstate swap: C38 4.90
- F40 perm rows: C50
- F46 ad +alpha [u]x: C09 15.8
- F48 so3_exp K K^T: C15 6.74
- F49 raise only below -1e-300: C24 `[-5e-324, -1e-300]`
- F52 residual drops the last entry: C02 `{'diag4': 0.0}`

**imu_dynamics:**

- I01 bias halves swapped: C44 5.41
- I02 to_gtsam swapped: C55 5.04
- I05 step dt\|dt\|: C45 0.325
- I07 truth E R: C31 2.67
- I10 Euler W dR: C41
- I12 dp Dt\|Dt\|: C42 1.41
- I13 generator -1: C41 26.0
- I14 \|Dt\|: C41 2.43
- I15 chain swapped: C45 0.019
- I16 acc length check dropped: C45 rejection
- I17 truth acc_s[0]: C31 2.91
- I18 increment E_s[0]: C45 0.106
- I20 batch s without th^6: C42 1.53e-13
- I21 batch skew sign: C41 1.95
- I22 from_gtsam swapped: C55 5.04
- I24 D0 dv <- D0[2]: C45 0.320
- I25 truth dR E: C31 0.704
- I26 truth R^T a: C31 5.57
- I28 increment acc x (1 + 1e-10): C45 2.7e-11

## Property tests and the hunt for spurious behaviour (own re-implementation)

- 10^4 pairs/triples: compose vs matmul 3.6e-15, associativity 7.1e-15, inverse (vs
  `np.linalg.inv`, \|t\| up to ~9) 1.4e-14, time entry 0.0.
- 10^4 (X, xi): own `vee(X e_k^ X^-1)` Ad vs src Ad and Ad_apply 6.2e-15 relative. Ad
  homomorphism 2.0e-15 relative. Ad_{X^-1} vs inv(Ad_X) 1.3e-14. Ad-ad compatibility 2.2e-14 /
  7.1e-15. det(Ad) 5.3e-15. Last row exactly e10.
- 10^4 xi, \|phi\| log-uniform in (1e-6, 2 pi - 0.3): exp vs expm 4.8e-14 relative; time block 0.0.
- 10^4 xi, \|phi\| in (0.05, pi - 1e-3): log(exp xi) 2.4e-15. 10^4 X, R = Exp(5 N): exp(log X) 3.6e-14.
- 10^4 Cayley evaluations: closed form vs 5x5 definition 5.5e-16 relative; Re orthogonality and
  det 8.9e-16; \|c\| = 2 tan(theta/2) and axis 3.3e-15.
- No spurious feasible Cayley points: S = I - [phi]x/2 has det 1 + \|phi\|^2/4 >= 1, so the residual
  zero set is the Cay point.
- 2x10^4 C50 draws at s = 1e-5: 4.57e-10 (bound 1e-9). At s = 1e-4: 4.26e-8.
- 20000 small-angle R (theta log-uniform in [1e-10, 1e-2]): so3_log absolute 3.5e-18. On the edge
  band tr in (-0.99, -0.9]: 8.9e-16.
- Keyframe guard: 0, -0.0, +-1e-12 and 5e-13 accepted; nextafter(1e-12, inf), NaN, +-inf and 1e-6
  refused.

## Per-claim verdicts

Scripts: `_p10/claims_a.py` (C01..C30) and `_p10/claims_b.py` (C31..C55). They use my own ODE
streams with non-constant (omega, a), random g and Dt in {+0.45, -0.35, +0.8}, and 36 instances.
The g block has both signs on every component (max 6.7 / 10.1 / 18.7, min -12.3 / -12.1 / -9.8).

| claim | verdict | measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | 10^4 pairs: 3.6e-15 (1e-12) |
| C02 | CONFIRMED | 2.4e-15; residual flags reflection, scaling, each of 7 zeros, (3,3), (4,4) at >= 0.3 |
| C03 | CONFIRMED | 7.1e-15 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 1.4e-14; time entry 0.0 |
| C06 | CONFIRMED | max commutator 16.3 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0; wedge(e4)[0:3,3] = e1, wedge(e7)[0:3,4] = e1 |
| C09 | CONFIRMED | vs own commutator ad: 1.8e-15; last row 0, (6:9, 9) = u exactly |
| C10 | CONFIRMED | 3.6e-15 / 1.1e-14 |
| C11 | CONFIRMED | 6.2e-15 relative |
| C12 | CONFIRMED | 2.0e-15 relative |
| C13 | CONFIRMED | 1.3e-14 |
| C14 | CONFIRMED | 2.2e-14 / 7.1e-15 |
| C15 | CONFIRMED | 2000 xi: 2.0e-14 (1e-8) |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 5.3e-15; trace 0.0 |
| C18 | CONFIRMED | vs 60-term series, \|phi\| in (1e-3, 3): 1.3e-14 |
| C19 | CONFIRMED | 10^4 vs expm: 4.8e-14 relative |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | 20000 phi: 1.8e-14 (1e-13); c2 at pi, pi - 10^-k: 1.05e-16 (1e-13); log formula vs own scipy log + series Gamma 1.3e-14 |
| C22 | CONFIRMED | 2.4e-15; \|phi\| = 3.5 wraps to 2.78319; edge band 8.9e-16 (2.5e-15); small angle 3.5e-18 |
| C23 | CONFIRMED | 3.6e-14 (1e-9) |
| C24 | CONFIRMED | det 1.0, 0.40528, 2.2128e-3, 2.5e-20 at 2 pi - 1e-9; cond(pi) 1.570796; coefficients 1.41e-12 at theta = 0.0504 (1e-11; 4502 points); literal polynomial 1.09e-16 / 2.18e-16 / 2.21e-16 / 2.20e-16 (1e-15); THETA_SMALL 0.05 |
| C25 | CONFIRMED | 5.5e-16 relative |
| C26 | CONFIRMED | matrix = I5 - xi^/2 exactly; det 1.3e-15 |
| C27 | CONFIRMED | 8.9e-16; time 0.0 |
| C28 | CONFIRMED | 10^4: 3.3e-15; table 0.51068, 1.09260, 3.11482, 28.2028, 43171.6 |
| C29 | CONFIRMED | \|d\|/s^3 = 0.0067357 constant over s = 0.05..0.005; remainder / s^5 = 5.70e-5 constant |
| C30 | CONFIRMED | 15 expressions, 25 symbols, total degree 2; numeric residual 1.3e-15; symbolic eval 3.3e-16 |
| C31 | CONFIRMED | own DOP853, random g, 36 instances: 3.9e-14 (also own T H' X D product); time 0.0 |
| C32 | CONFIRMED | 10^4: 2.8e-14 (incl. expm(Dt zeta_g)) |
| C33 | CONFIRMED | 5.1e-14 relative |
| C34 | CONFIRMED | time - Dt and R, v 3.9e-14; position - (p_j + g Dt^2) 3.6e-14 |
| C35 | CONFIRMED | position + v_j Dt 3.9e-14; time 0.0 |
| C36 | CONFIRMED | p_j + g Dt t_i 3.9e-14; time - t_i 2.2e-16; guard as above |
| C37 | CONFIRMED | gtsam W Phi U vs Forster form, 3000 random g/Dt: 1.8e-15; vs ODE via C43 3.9e-14 |
| C38 | CONFIRMED | W_g = T(-Dt) H' T(Dt), T(Dt) U_g = D, two forms vs ODE: 3.8e-14 |
| C39 | CONFIRMED | vs gtsam compose: 8.9e-16 |
| C40 | CONFIRMED | 0.0 / automorphism 8.9e-16; Phi(I) = I exactly |
| C41 | CONFIRMED | vs expm, Dt in [-5, 20]: 6.8e-14 relative; own DOP853 6.0e-14; Euler err x n/Dt = 1.6234 at n = 1e3 and 4e3 (first order); signed-dt Euler vs matrix power 3.3e-15 |
| C42 | CONFIRMED | 6.8e-14 relative (1e-12); Rodrigues scalar/batch vs mpmath on [0.04, 0.052]: 9.99e-16 / 9.99e-16 (1.3e-15) |
| C43 | CONFIRMED | 3.9e-14 |
| C44 | CONFIRMED | 10^4: 0.0, incl. gtsam correctGyroscope / correctAccelerometer |
| C45 | CONFIRMED | closed form 0.0; split-segment DOP853 vs chain 1.2e-12 (ODE tolerance); discrete split 6.7e-16; ordered product of own steps 8.9e-16; malformed streams rejected |
| C46 | CONFIRMED | 0.0 |
| C47 | CONFIRMED | 0.0 (position/translation = r, velocity = v, time, rotation/attitude) |
| C48 | CONFIRMED | Hat = wedge and Vee inverse: 0.0 |
| C49 | CONFIRMED | NavState retract(1e-3 e3) moves position (1e-3, 0, 0); Gal3.Expmap moves velocity; NavState.matrix() column 3 = p, column 4 = v |
| C50 | CONFIRMED | 20000 at s = 1e-5: 4.57e-10 (1e-9); unpermuted 7.0e-5 (O(1) relative); s = 1e-4: 4.26e-8; P as stated |
| C51 | CONFIRMED | 10^4: Expmap 2.6e-14, Logmap 2.2e-14 (\|phi\| < pi - 1e-3), AdjointMap 1.8e-15, adjointMap 0.0 |
| C52 | CONFIRMED | compose / * / inverse / between / retract 8.9e-15; localCoordinates on the stated domain 8.9e-15 |
| C53 | CONFIRMED | Dt = 0.5; predict vs endpoint 1.8e-15 with g passed and 1.8e-15 with g omitted; G_DEFAULT = n_gravity = (0, 0, -9.81) |
| C54 | CONFIRMED | vector [1..6]; accelerometer, gyroscope halves |
| C55 | CONFIRMED | 10^4: 0.0; S S = I |
| **C64** | **REFUTED** | max error 8.73 (claimed <= 1e-12); the true term is v1 t2 |

**Totals: 55 CONFIRMED, 1 REFUTED (C64), 0 UNTESTABLE.**

## New findings

- **M1-F25 (medium-low): per-function gravity defaults are unpinned.**
  - F22 pinned `G_DEFAULT` and `endpoint`'s default only.
  - The other 8 parameter defaults can be changed (e.g. to `-G_DEFAULT`) and the suite passes:
    `gravity_lift`, `gravity_lift_uncorrected`, `gravity_element`, `gravity_generator`,
    `endpoint_uncorrected`, `endpoint_right_time_shift`, `forster_endpoint` and
    `imu.integrate_truth`.
  - This is the F22 hazard per function: a Stage-B caller that omits g silently gets the wrong
    gravity.
  - Kill: for each function, assert `inspect.signature(f).parameters['g'].default` equals the
    retyped literal (0, 0, -9.81), or call each once without g and compare with the explicit-g
    call.
- **M1-F26 (low-medium, fixture): gravity magnitude is never guarded.**
  - Under H_MAG2 (\|g\| = 9.81 on every draw), every guard passes, because the directions are
    unchanged. GNORM (g -> 9.81 g/\|g\|, i.e. hard-coding the standard magnitude) then survives the
    suite in gravity_lift, integrate_truth and forster_endpoint. Each is killed alone (C31 3.145).
  - Kill: on every guarded block of m >= 3, require max\|g\| / min\|g\| >= 2 (recorded ranges from
    U(2, 20) are wide).
  - Or, structurally, append a retyped literal g set with several magnitudes (e.g. 3, 9.81, 17) to
    each random-g block.
- **M1-F27 (low, fixture): per-draw sign flips evade every guard except the sign marginals.**
  - Under H_ODD (g -> -g wherever g_x g_y g_z < 0), every guard passes by construction: off_z,
    perm_gap, spread, span and comp are invariant under g -> -g, and the four kept octants give
    each component both signs.
  - PFLIP (the matching sign-dependent defect) then survives in gravity_lift and integrate_truth.
    Each is killed alone (C31 12.45).
  - Kill, structural: run every random-g check on the pair {g, -g}, which closes all per-draw sign
    maps at once. Or require all 8 sign octants on the 400..2000-draw blocks, plus a retyped
    literal octant set in the 4-g datasets.
  - Note: guards on marginal statistics cannot exclude every nonlinear defect. A retyped literal g
    set, which a fixture edit cannot reach without a visible textual change to a literal, is the
    durable remedy.
- **M1-F28 (low): c2's closed form is pinned only to 1e-11.**
  - F20 (`1/t2 - theta s/(2 omc t2)`, algebraically the stated form) changes c2 by 1.14e-12
    relative. Its sup error on [0.05, 0.055] is 2.43e-12, against clean 1.68e-12.
  - Kill: a dense grid on [0.05, 0.055] (4000 points) with c2 relative error <= 2.0e-12 against
    60-digit mpmath. That gives 1.19x headroom over clean, so it is sound only with a fixed grid.
- **M1-F29 (low): the branch at theta = THETA_SMALL exactly is untested.**
  - F41 (`theta <= THETA_SMALL` in gamma_coefficients) uses the Taylor branch at 0.05, against
    C24's "for theta < 5e-2". kappa1 is then 7.75e-13 off (clean 1.7e-16), and
    Gamma1inv Gamma1 - I is 1.30e-14 at \|phi\| = 0.05.
  - Kill: at theta = 0.05 exactly, kappa1 relative error <= 1e-14 against mpmath; or
    \|Gamma1inv Gamma1 - I\| <= 1e-15 at \|phi\| = 0.05.
  - (The same edit in so3_exp / the batch, F42 / I04, is equivalent: 9.99e-16 either way.)
- **M1-F24 (open, re-ruled): B1, B2, B3, B3b, B4 and now F19 are NON-EQUIVALENT under rule 4.**
  - Rule 4(b) cannot be met for any mutant on a quantity whose pin sits within 100x of clean.
  - Either tighten the pins to clean + 1 to 2 ulp (pass-9 kills: C42 Rodrigues <= 1.05e-15;
    C22 small-angle relative <= 4e-16 on the test set, re-measured, or an exact-formula check
    below theta 1.2e-7; C24 polynomial <= 2.5e-16 for kappa1 and kappa2), or the owner amends 4(b).

Advisory (not a gate item): A4. C52's domain text still says the near-pi disagreement occurs "in
every implementation". The test comment now says it is gtsam's; the claim text should say so too.

## Clean suite

`M1_MUTATION_WORKERS=4 python -m pytest -q tests/` at 8e1ecae: **176 passed in 130.44 s**, exit 0.
That is 46 test_gal3, 9 test_imu_dynamics, 102 test_mutation (101 mutants plus the unmutated copy)
and 19 test_orchestrator. Each mutant run of test_gal3 + test_imu_dynamics on the clean tree:
55 passed. The scratch `_p10/` was deleted before commit.

## Gate

**FAIL.** 23 non-equivalent survivors:

- 8 per-function gravity defaults (M1-F25);
- H_MAG2 plus 3 GNORM compounds (M1-F26);
- H_ODD plus 2 PFLIP compounds (M1-F27);
- F20 (M1-F28);
- F41 (M1-F29);
- B1, B2, B3, B3b, B4 and F19 (M1-F24 under rule 4).

M1-F22 and M1-F23 are fixed for every named pass-9 survivor. Every claim C01..C55 is confirmed.
M1-C64 is false and must not enter the yaml.
