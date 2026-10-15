# M1 ninth-pass adversarial verification

Branch `m1-gal3-mechanics`. The code verified is HEAD 3f735b8, the response to M1-F18..F21 and
advisories A1/A2. Agent verify-M1-010.

**src/ is unchanged.** `git diff 94ad0ea 3f735b8 --stat` touches only docs/claims/M1.yaml,
tests/conftest.py, tests/test_gal3.py, tests/test_imu_dynamics.py and tests/test_mutation.py.
`-- src` is empty.

References are independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp` DOP853), mpmath (40 to
130 digits, `logm`), sympy and gtsam 4.3.0. No reference imports `src` or `tests`. One exception is
labelled where it occurs: the instrumented copy of `tests/conftest.py` used to read the guard's
own input data. It is not used as a reference. I re-implemented these myself:

- the Gal(3) adjoint, as `vee(X e_k^ X^-1)` with `np.linalg.inv`;
- the SO(3) log (atan2 angle, eigen-axis near pi), and a 50-digit mpmath log of the rounded R;
- the Gamma coefficients, as 50-digit mpmath series (c2 in 130 digits, see below);
- a 40-digit mpmath Rodrigues;
- the navigation ODE and the 5x5 increment ODE (DOP853, rtol = atol = 1e-13);
- a mutation runner (`_p9/mut.py`). It makes real textual edits, each required to match exactly
  once, on a temp copy of src/ + tests/, and judges each by the full
  `tests/test_gal3.py tests/test_imu_dynamics.py` run with no assertion removed.

Equivalence rules are applied as stated:

- rule 1: measured effect <= 1/10 of the tightest claim or test-pin tolerance on the affected
  quantity;
- rule 2: the mutant's sup-norm error over the affected domain is within 8 ulp AND no worse than
  clean;
- rule 3: a written proof.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C63 (brief only) | **REFUTED**: max \|Ad_X^T Ad_X - I\| = 372.9 over 10^4 SE(3)-embedded X (min 5.6e-4); 1.0 exactly at X = (I, 0, e1, 0) |
| M1-F18 | **FIXED** for S01/S02/S03 (and R07/S04/S05). The polynomials are retyped from C24's statement in mpmath, not taken from src. The 1e-15 pin leaves a sub-pin band (B4, M1-F24) |
| M1-F19 | **FIXED** for S06/S07. The 1.3e-15 pin is **sound, not fragile**, but it does not close the band (B1/B2, M1-F24) |
| M1-F20 | **FIXED** for S08. The 1e-15 bound still admits a branch at n < 3e-8 or n < 5e-8 (B3/B3b, M1-F24) |
| M1-F21 | **FIXED** for planar distributions (F1W, and my own tilted plane): sound against every *linear* g defect (proof below). A half-space distribution (g_y >= 0) passes every guard and masks sign-dependent g defects (M1-F23) |
| A1, A2 | **Resolved.** Signed-dt Euler check kills S30 and my M13. The stream-length check kills M06 and my M15 |
| Harness | 91/91 mutants killed plus the unmutated copy, inside the clean run. Compound edits are applied each-exactly-once, sequentially. Concurrency cannot mask a failure (see below) |
| Replays | S01, S02, S03, S06, S07, S08, F1W, Y_LIFT/ELEM/TRUTH/FORST, alone, +F1W and +tilted plane: **all 22 KILLED** |
| Fresh sweep | 77 mutants: 59 killed, 18 survived. Of the survivors, 8 are equivalent (measured or proved) and **10 are NON-EQUIVALENT** |
| Clean suite | `pytest tests/` with M1_MUTATION_WORKERS=4: **166 passed in 122.8 s**, exit 0 |
| **Gate** | **FAIL** |

## M1-C63: REFUTED

Claim: for X = (R, 0, p, 0), Ad_X is orthogonal, Ad_X^T Ad_X = I to 1e-12.

Ad_X = [[R,0,0,0],[0,R,0,0],[[p]x R,0,R,0],[0,0,0,1]]. Its (phi, phi) block of Ad^T Ad is
I + R^T [p]x^T [p]x R, and that is I only when p = 0. A skew off-diagonal block does not make a
block-triangular matrix orthogonal.

Measurement: own Ad from `vee(X e_k^ X^-1)`, which agrees with src `Ad` to 2.2e-16.

- 10^4 draws, R = Exp(N(0, I)), p = N(0, I) * U(0, 5): max \|Ad^T Ad - I\| = 372.9, min 5.6e-4.
- \|Ad xi\| / \|xi\| ranges over [0.545, 14.2]: Ad does not preserve the norm.
- At X = (I, 0, e1, 0): \|Ad^T Ad - I\|max = 1.0, and \|Ad (phi = e2)\| = sqrt(2).
- Control, p = 0 (pure rotation): 2.2e-16.

The claim holds only for p = 0.

## M1-F18..F21 rulings

### F18: C24's literal polynomial (FIXED)

- The test's `stated_poly` is 1/2 - T2/24 + T4/720, 1/6 - T2/120 + T4/5040,
  1/24 - T2/720 + T4/40320, 1/12 + T2/720 + T4/30240. That matches C24's statement character for
  character. It is evaluated on `mp.mpf(th)` at 50 digits and reads nothing from src.
- My own check over 6000 theta in (1e-12, 0.05): clean relative error 1.09e-16 / 2.20e-16 / 2.21e-16
  / 2.20e-16, the same figures as the hint.
- Replay: S01 is killed on `poly_k1` (1.69e-13), S02 on `poly_k3` (9.22e-14) and S03 on `poly_c2`
  (8.20e-14).
- **Residual band:** B4, `t4/720.0 -> t4/720.00002`, moves kappa1 by 5.6e-16 relative. It
  measures 5.8e-16 against the stated polynomial (clean 1.1e-16) and passes the 1e-15 pin. See
  M1-F24.

### F19: Rodrigues band pin 1.3e-15 (FIXED; sound, not fragile)

S06 and S07 are both killed at 1.665e-15 (`worst_sw_scalar` and `worst_sw_batch`).

On fragility, the clean maximum is not rounding noise:

- It is the Taylor truncation of (1 - cos)/th^2, th^8/8! = 9.69e-16 at th = 0.05, which rounds
  to 9.99e-16.
- It is computed with IEEE +, *, / only (no libm), so it is platform-independent. It is the same
  9.992e-16 on 5 independent seeds of 600 angles.
- The libm-dependent closed-form side, [0.05, 0.052], measures 1.1e-16. A 1 to 2 ulp libm
  difference cannot reach 1.3e-15.

The 1.30x headroom is therefore deterministic margin, and the pin is sound. It does not close
the band, though: B1/B2 (`+ th^6/200000`) survive (M1-F24).

### F20: so3_log small-angle relative 1e-15 (FIXED for S08)

- S08 (n < 1e-7) is killed at 3.31e-15 (`worst_small_rel`).
- B3c (n < 1e-8): change 0.0 on 20000 R. Equivalent, as the hint says.
- **B3 (n < 3e-8) and B3b (n < 5e-8) survive** (M1-F24).

### F21: gravity-direction guard (FIXED for planar g; half-space gap is M1-F23)

Replays:

- F1W (x-z plane) is killed by `comp >= min_component` (0.0) in all 10 random-g tests.
- My own tilted plane (normal (0.3, 0.5, 0.81), every component nonzero) is killed by
  `span >= min_span` (6.7e-17..1.6e-16) in 9 tests, the tests with >= 3-draw guards.
- Y_LIFT, Y_ELEM, Y_TRUTH and Y_FORST are killed alone. Measured: C31 worst_ode 9.999,
  C32 29.8, C33 649.7, C35 16.27, C31 identity 9.999, C34 2.000, C37 9.999, C43 33.8.
- Each Y_* is also killed under F1W (comp guard) and under the tilted plane (span guard).

**Proof that the span guard is sufficient for every linear-in-g src defect.** Every use of g in
src is linear: `g * Dt`, `-0.5 * g * Dt * Dt`, `g + R a` and `0.5 * g * Dt * Dt`. A defect that
replaces g by M g, with M != I, is invisible on a block {g_k} only if (M - I) g_k = 0 for all k.
That forces M = I whenever the g_k span R^3. Every guarded block of m >= 3 passes all of its g
to the function under test, so it detects any such M. The guard holds on every recorded block:
the 4-g datasets and the 400..2000-draw blocks. C45's single draw is guarded by comp only, but
integrate_truth, its only g consumer, is also covered by the spanning 4-g datasets.
min_span = 0.1 is enough: a linear defect of size \|M - I\| moves some g_k by
>= 0.1 sqrt(m) \|M - I\| \|g_k\|, far above the 1e-9..1e-12 tolerances.

**The hunt.** The guard is not sufficient for *nonlinear* (sign- or axis-dependent) defects:

- H_AXIS (every g snapped to a signed coordinate axis) is killed: a 4-g dataset misses an axis,
  so comp = 0.0.
- **H_HEMI (g_y := \|g_y\|) passes every guard.** Under it, YABS_LIFT and YABS_TRUTH
  (g_y -> \|g_y\| in gravity_lift / integrate_truth) **survive the full suite**. Each is killed
  alone (C31 worst_ode 9.999, identity 9.999). See M1-F23.

### A1 / A2

- Euler with dt < 0 (`worst_pow_neg`, own check 1.49e-13 vs `matrix_power(I + dt A, n)`) kills
  harness S30. My M13 (`int(n) * abs(dt)`) is killed at `abs(eT - (-0.4))`.
- The stream-length check kills harness M06. My M15 (omega check dropped) is killed at
  `not not_rejected`.

## Harness audit

- `_edits` returns each (file, old, new) of a compound mutant. `_mutate_and_run` counts `old` in
  the text as already modified by the earlier parts, and returns `code = None` unless the count is
  1. `test_mutation_is_killed` then fails on `assert code is not None`. I probed this directly:
  - a missing second part gives "occurs 0 times";
  - an ambiguous second part gives "occurs 42 times";
  - a second part whose target the first part consumed gives "occurs 0 times".

  Every one of the 91 harness edits matches exactly once, sequentially, in the clean tree.
- Concurrency: each mutant gets its own directory. The 91 sanitised names are distinct, and a
  collision would raise in `copytree`, which is loud. Results are keyed by the mutant id the worker
  returns. Failure modes are loud too:
  - an OOM kill gives `code = -9`, which fails `code == 1`;
  - a timeout re-raises from `f.result()`, which errors the whole module;
  - a collection error gives `code = 2`, which fails.

  The baseline runs inside the same pool. **Concurrency cannot mask a failing mutant.**

## Replays (real exactly-once edits; full test_gal3 + test_imu_dynamics)

| id | edit | result | killed by (statement: value) |
|---|---|---|---|
| S01 | kappa1 `t4/719.993` | KILLED | C24 `poly_k1 <= 1e-15`: 1.69e-13 |
| S02 | kappa3 `t4/40319` | KILLED | C24 `poly_k3`: 9.22e-14 |
| S03 | c2 `t4/30239` | KILLED | C24 `poly_c2`: 8.20e-14 |
| S06 | so3_exp c `+ t4*t2/60000` | KILLED | C42 `worst_sw_scalar <= 1.3e-15`: 1.665e-15 |
| S07 | batch c `+ t2^3/60000` | KILLED | C42 `worst_sw_batch`: 1.665e-15 |
| S08 | so3_log `n < 1e-7` | KILLED | C22 `worst_small_rel <= 1e-15`: 3.31e-15 |
| F1W | rand_gravity in the x-z plane | KILLED | `comp >= min_component`: 0.0 (10 tests) |
| TILT (own) | rand_gravity in a tilted plane | KILLED | `span >= min_span`: 6.7e-17 (9 tests) |
| Y_LIFT | gravity_lift g_y sign | KILLED | C31, C32, C35, C36, C38, C43 (6 tests), e.g. worst_ode 9.999 |
| Y_ELEM | gravity_element g_y sign | KILLED | C32 29.8, C33 649.7 |
| Y_TRUTH | integrate_truth g_y sign | KILLED | C31, C34, C37, C43, C45 (5 tests) |
| Y_FORST | forster_endpoint g_y sign | KILLED | C31, C37 (33.8), C43 |
| Y_* + F1W (4) | | KILLED | comp guard, 10 tests each |
| Y_* + TILT (4) | | KILLED | span guard, 9-10 tests each |

## Fresh mutation sweep (77 mutants)

### NON-EQUIVALENT SURVIVORS (10)

| id | edit | measured effect | why non-equivalent |
|---|---|---|---|
| **N30** | `G_DEFAULT = (0, 0, -9.81) -> (0, 0, +9.81)` | the default gravity points UP; every src call that omits g is wrong by 2\|g\| | tests/conftest.py takes `G = gal3.G_DEFAULT`, and C53 passes gtsam's n_gravity explicitly, so nothing pins the value the conventions state (M1-F22) |
| **N29** | `G_DEFAULT -> (0, 0, -9.80665)` | default g changes by 3.4e-3 | as N30 |
| **H_HEMI** (fixture) | rand_gravity with g_y := \|g_y\| | passes off_z, perm_gap, spread, comp and span | masks YABS_LIFT and YABS_TRUTH (M1-F23) |
| **YABS_LIFT + H_HEMI** | gravity_lift g_y -> \|g_y\| | survives the full suite; alone it is killed (C31 9.999) | |
| **YABS_TRUTH + H_HEMI** | integrate_truth g_y -> \|g_y\| | survives; alone it is killed (C31 identity 9.999) | |
| **B1** | so3_exp Taylor c `+ t4*t2/200000` | matrix change 2.22e-16; sup on [0.04, 0.05) 1.221e-15 vs clean 9.99e-16 | 2.2e-16 > 1.3e-16 (1/10 of 1.3e-15); 1 ulp but worse than clean (M1-F24) |
| **B2** | batch c `+ t2^3/200000` | same as B1 on the batch | as B1 |
| **B3** | so3_log `n < 3e-8` | relative change 2.92e-16; sup relative on the affected domain (theta < 6.6e-8, 20000 R) 4.79e-16 vs clean 3.59e-16 | > 1e-16 (1/10 of 1e-15); worse than clean |
| **B3b** | so3_log `n < 5e-8` | relative change 9.08e-16; sup 9.69e-16 vs clean 3.50e-16 (2.8x) | as B3 |
| **B4** | kappa1 `t4/720.00002` | relative change 5.6e-16; vs the stated polynomial 5.84e-16 vs clean 1.10e-16 | > 1e-16; worse than clean against C24's literal statement |

N29 and N30 are one finding. The B items are rounding-band leaks of 1 to 5 ulp.

### EQUIVALENT SURVIVORS (8)

| id | edit | measured effect | rule |
|---|---|---|---|
| N04 | so3_log `tr > 0.0 -> tr > 0.5` | changes only theta in (1.82, 2.09): so3_log 1.11e-15, gal log 1.39e-15. Small-angle and (-0.99, -0.9] band change 0.0 | rule 1 (C52 1e-13 -> 1e-14) |
| N05 | `arctan2(n, qw) -> arctan(n / qw)` | change: small-angle relative 4.1e-16, mid 8.9e-16, band 0.0. Mutant sup errors equal clean on all three domains (5.56e-16 rel, 8.9e-16, 8.9e-16) | rule 2 (<= 4 ulp, no worse) |
| N32 | so3_log `elif R11 >= R22` | 0.0 on random R; on 3000 exact ties (axis (0, 1, +-1)/sqrt2, theta in (2.1, pi)) change 8.9e-16, sup 8.9e-16 = clean | rule 2 |
| N34 | Ad `skew(p) R - t skew(v) R` | 8.3e-16 relative | rule 1 (C11 1e-10) |
| M05 | integrate_truth: v updated first, then `p + v dt - half f` | algebraically identical; position change 2.9e-15 relative | rules 1 and 3 |
| B5 | kappa1 in Horner form | vs the stated polynomial 5.6e-17 (clean 1.1e-16); change 1.1e-16 (0.5 ulp) | rule 2 |
| B3c | so3_log `n < 1e-8` | 0.0 on 20000 R (n^2/3 < 3.4e-17) | rule 1 |
| N13 | gamma_coefficients accepts theta in [-1, 0) | every caller (gamma1, gamma2, gamma1_inv) passes a norm >= 0. No claim states the raise | rule 3 (unreachable on every claim domain; advisory A5) |

### Killed (59)

Each line gives the mutant, then the failing test and value.

**Fixture hunt (gal3 / imu):**

- H_AXIS: comp guard 0.0
- SNAP_LIFT: C31 3.87
- SNAP_TRUTH: C31 3.87
- YABS_LIFT: C31 9.999
- YABS_TRUTH: C31 9.999
- SNAP_* + H_AXIS: comp guard

**gal3:**

- N01 skew sign: C08 6.82
- N02 s `t4/24`: C23 1.0e-8
- N03 s th^6 sign: C42 2.97e-13
- N06 Shepperd s without the factor 2: C22 2.73
- N07 residual without det: C02 `{'det': 0.0}`
- N08 residual row 4 [0:3]: C02 `{'row4[3]': 0.0}`
- N09 inverse +t: C05 51.7
- N10 compose t1 - t2: C01 8.02
- N11 Ad +tR: C11 21.7
- N12 Ad_apply alpha p: C11 29.0
- N14 kappa3 denominator: C15 1.17
- N15 c2 with 1 - cos: C24 7.1e-13 band
- N16 Gamma1 -k2: C15 4.51
- N17 exp G1 u: C15 1.10
- N18 log G2 v: C21 9.05
- N19 cay S plus: C25 4.17
- N20 denominator 0.25: C26 62.0
- N21 Cayley residual sign: C30 2.88
- N22 uncorrected lift sign: C34 2.45
- N23 KEYFRAME_TIME_TOL 1e-11: C36 accepted 2e-12
- N24 endpoint Dt from (3, 3): C31 4.91
- N25 Forster v Dt^2: C31 1.57
- N26 NavState compose without R: C39 6.69
- N27 unpack t from (3, 3): C01 8.26
- N28 unskew order: C08 5.56
- N31 THETA_SMALL 4.5e-2: C24 `0.045 == 0.05`
- N33 Shepperd branch-1 sign: C22 2.71
- N35 cay alpha u: C25 4.22
- N36 generator w = -g/2: C32 4.88
- N37 navstate_phi v * (1 + 1e-7): C40 4.0e-7
- N38 keyframe check `abs > tol`: C36 `[nan]`
- N39 `check_time=False` default: C36 DID NOT RAISE
- N40 ad -[u]x: C09 17.0

**imu_dynamics:**

- M01 accelerometer + b_a: C44 7.52
- M02 batch K12 sign: endpoint convergence 3.39
- M03 step without 1/2: C45 0.116
- M04 truth half dt\|dt\|: C31 (Dt = -0.3) 5.1e-4
- M06 truth v-first, unadjusted: C31 7.4e-4
- M07 increment \|dt\|: C45 1.2
- M08 increment dp 2 half: C45 6.8e-6
- M09 generator w = a/2: C41 6.11
- M10 preintegrate \|Dt\|: C41 2.43
- M11 truth E(\|dt\|): convergence 1.49
- M12 Euler dv first: C41 (b') 2.57
- M13 Euler T \|dt\|: C41 signed-dt 0.4
- M14 dp via Gamma1/2: C42 0.062
- M15 omega length check dropped: C45 rejection
- M16 batch s th^6 sign: C42 3.07e-13
- M17 bias_from order: C55 5.04
- M18 truth f without R: C31 4.78
- M19 D0 t0 dropped: C45 0.04
- M20 chain swapped: C45 0.019

## Property tests and the hunt for spurious behaviour (own re-implementation)

- 10^4 SE(3)-embedded X: own Ad vs src Ad 2.2e-16; orthogonality fails (C63).
- 10^4 xi, theta in (1e-9, pi - 1e-3) and log-uniform (1e-9, 0.2): log(exp(xi)) 2.2e-14 relative.
- 5000 X including pi - 10^-k: exp(log X) 6.4e-15.
- 2000 X: src log vs own formula (own SO(3) log, mpmath Gamma, linear solves) 1.8e-14.
- 3000 near-pi X (pi - 10^-u, u in (2, 5)): src exp(log X) - X = 2.7e-15, while gtsam's Logmap
  gives 9.9e-8 (advisory A4).
- 9000 thetas incl. a 4001-point grid on [0.05, 0.07]: Gamma coefficients vs 50-digit series
  1.58e-12 at theta = 0.0519 (c2; bound 1e-11).
  - Note: my first reference evaluated c2's closed form at 50 digits, and that lost all digits at
    theta = 1e-12 (an apparent 1.5e-2 error). That was a reference bug, fixed by 130 digits. src
    is correct there.
- 10^4 random Cayley vectors \|c\| in (0.01, 5): angle/axis 8.0e-15.
- 5000 xi: Cay closed form vs the 5x5 definition 1.5e-15, det(I - xi^/2) vs 1 + \|phi\|^2/4
  1.4e-15, Re orthogonality 1.8e-15. There are no spurious feasible Cayley points: S = I - [phi]x/2
  has sigma_min = 1, so the residual block's zero set is the Cay point.
- 10^4 bias draws vs gtsam correctGyroscope/correctAccelerometer and the swap round trip: 0.0.
- 20000 C50 draws at s = 1e-5: 5.6e-10 (bound 1e-9); unpermuted 24.9 relative.
- Keyframe guard: 0, -0.0, +-1e-12 and 5e-13 accepted; nextafter(1e-12, inf), NaN, +-inf and 1e-6
  refused.

## Per-claim verdicts

Scripts: `_p9/claims_a.py` (C01..C30) and `_p9/claims_b.py` (C31..C55). The ODE streams, random
g and Dt in {+0.45, -0.35, +0.8} are my own.

| claim | verdict | measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | 3000 pairs: 7.1e-15 |
| C02 | CONFIRMED | 2.2e-15; residual flags reflection, scaling, 7 zeros, (3,3), (4,4) at >= 0.3 |
| C03 | CONFIRMED | 8.2e-16 relative |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 3.6e-14 (incl. vs np.linalg.inv); time entry 0.0 |
| C06 | CONFIRMED | max commutator 56.1 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0 |
| C09 | CONFIRMED | vs own ad and the commutator: 3.6e-15 |
| C10 | CONFIRMED | 1.4e-14 |
| C11 | CONFIRMED | Ad and Ad_apply vs own vee(X e^ X^-1): 4.4e-15 relative |
| C12 | CONFIRMED | 1.4e-15 relative |
| C13 | CONFIRMED | 3.0e-15 relative |
| C14 | CONFIRMED | 5.9e-15 relative |
| C15 | CONFIRMED | 4.1e-14 relative (1e-8) |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 5.3e-15; trace 0.0 |
| C18 | CONFIRMED | vs 60-term series, 400 theta in (1e-6, 3): 1.7e-14 |
| C19 | CONFIRMED | 4000 xi vs expm: 2.6e-14 |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | c2 at pi, pi - 10^-k: 1.05e-16 (1e-13); log formula 1.8e-14 |
| C22 | CONFIRMED | 2.2e-14 (1e-9); \|phi\| = 3.5 wraps to 2.78319 |
| C23 | CONFIRMED | 6.4e-15 |
| C24 | CONFIRMED | det 1.0, 0.40528, 2.2128e-3, 2.5e-20 at 2 pi - 1e-9; cond(pi) 1.570796; coefficients 1.58e-12 (1e-11); below the switch 7.75e-13 / 2.58e-13 / 1.03e-13 / 1.55e-13; literal polynomial 1.09e-16 / 2.20e-16 / 2.21e-16 / 2.20e-16; THETA_SMALL 0.05 |
| C25 | CONFIRMED | 1.5e-15 relative |
| C26 | CONFIRMED | 1.4e-15; matrix = I5 - xi^/2 |
| C27 | CONFIRMED | 1.8e-15; time 0.0 |
| C28 | CONFIRMED | 10^4: 8.0e-15; table 0.51068, 1.09260, 3.11482, 28.2028, 43171.6, all within the 4-s.f. rounding |
| C29 | CONFIRMED | s^-3 ratio 0.011716 (constant); s^-5 remainder 1.863e-4 (constant) |
| C30 | CONFIRMED | 15 expressions = own sympy exactly; 25 symbols; degree 2; residual 2.2e-16 |
| C31 | CONFIRMED | own DOP853, random g, 36 instances: 2.8e-14; time 0.0 |
| C32 | CONFIRMED | 1.4e-14 (incl. expm(Dt zeta_g)) |
| C33 | CONFIRMED | 6.4e-15 relative |
| C34 | CONFIRMED | time - Dt 0.0; position - (p_j + g Dt^2) 1.4e-14; R, v 2.8e-14 |
| C35 | CONFIRMED | position + v_j Dt 3.1e-14; time 0.0 |
| C36 | CONFIRMED | 2.8e-14; guard as above |
| C37 | CONFIRMED | gtsam W Phi U vs own ODE: 2.8e-14 |
| C38 | CONFIRMED | lift identities 4.4e-16; forms 1.8e-15 |
| C39 | CONFIRMED | vs gtsam compose: 8.9e-16 |
| C40 | CONFIRMED | 2.7e-15; Phi(I) = I exactly |
| C41 | CONFIRMED | vs expm, Dt in [-5, 20]: 1.3e-14 relative; own DOP853 6.2e-14; Euler err/dt 0.9551 (n = 1e3), 0.9552 (n = 4e3): first order; D(0) = I5; signed-dt Euler vs matrix power 1.5e-13 |
| C42 | CONFIRMED | 3000 vs expm: 1.6e-13 (1e-12); Rodrigues scalar/batch vs mpmath on [0.04, 0.052]: 9.99e-16 / 9.99e-16 (1.3e-15) |
| C43 | CONFIRMED | 2.8e-14 |
| C44 | CONFIRMED | 0.0, incl. gtsam correct* |
| C45 | CONFIRMED | three own DOP853 solves 7.3e-15; algebraic 0.0; step element 1.1e-15; ordered product 1.0e-15; malformed streams rejected |
| C46 | CONFIRMED | 0.0 |
| C47 | CONFIRMED | 0.0 |
| C48 | CONFIRMED | 0.0 |
| C49 | CONFIRMED | NavState retract(1e-3 e3) moves position (1e-3, 0, 0); Gal3.Expmap moves velocity |
| C50 | CONFIRMED | 5.6e-10 (1e-9); unpermuted O(1); P as stated |
| C51 | CONFIRMED | Exp 1.8e-14, Log 3.8e-14 (\|phi\| < pi - 1e-3, 10^4), Ad 2.2e-16, ad 0.0 |
| C52 | CONFIRMED | compose 1.8e-15, inverse 4.4e-15, between 5.3e-15; localCoordinates on the stated domain 8.0e-15 (edge 6.2e-15; 10^4 pairs); retract (0.1 N) 4.4e-16 |
| C53 | CONFIRMED | 1.8e-15; time 0.0 |
| C54 | CONFIRMED | vector [1..6]; accelerometer, gyroscope halves |
| C55 | CONFIRMED | 0.0; S S = I; S = block swap |
| **C63** | **REFUTED** | max \|Ad^T Ad - I\| = 372.9 (claimed <= 1e-12); holds only for p = 0 |

**Totals: 55 CONFIRMED, 1 REFUTED (C63), 0 UNTESTABLE.**

## New findings

- **M1-F22 (medium-low): `G_DEFAULT` is unpinned.**
  - tests/conftest.py imports `G = gal3.G_DEFAULT`, so every "g = G_DEFAULT" test follows a
    mutation of it. C53 passes gtsam's `n_gravity` explicitly.
  - N30 (default gravity UP, (0, 0, +9.81)) and N29 (-9.80665) survive the full suite. The value
    is a stated convention (M1.yaml conventions; the imu_dynamics module docstring). Every src
    function defaults `g=G_DEFAULT`, so a Stage-B caller that omits g silently gets the wrong
    gravity.
  - Kill: define the test gravity as an independent literal `np.array([0.0, 0.0, -9.81])` and
    assert `gal3.G_DEFAULT` equals it, or gtsam `MakeSharedU(9.81).n_gravity`. Also call
    `gal3.endpoint(X_i, D)` without g in C53 and compare it with gtsam's predict.
- **M1-F23 (low, fixture): the gravity guard admits a HALF-SPACE distribution.**
  - g_y := \|g_y\| keeps comp, span and every older guard satisfied. Under it, a sign-dependent g
    defect (g_y -> \|g_y\|) in gravity_lift or integrate_truth survives the suite. Both are killed
    under the clean fixture.
  - Linear defects are provably caught (proof above), so this concerns nonlinear defects only.
  - Kill: for blocks of m >= 3, require each component to take BOTH signs with magnitude >= 1
    (max_k g_k >= 1 and min_k g_k <= -1). I instrumented a copy of the guard on the clean suite.
    Every recorded block, including both 4-g datasets, has both signs at >= 2.10 in every
    component. The single-draw C45 block is exempt.
- **M1-F24 (low): the three pass-8 pins leave rounding-band survivors** (B1, B2, B3, B3b, B4).
  - Each pin leaves headroom above clean: 1.30x for C42 Rodrigues, 2.3x for the C22 small-angle
    relative bound and 4.5x for the C24 polynomial pin.
  - At these magnitudes, 1/10 of the pin (rule 1) is 1.3e-16 or 1e-16, which is at or below 1 ulp
    of the quantities. A perturbation that fits in the headroom therefore fails rule 1 and, being
    worse than clean, fails rule 2.
  - Measured: B1/B2 (Rodrigues c + th^6/200000) 1.22e-15 vs 9.99e-16; B3/B3b (log branch
    n < 3e-8 / 5e-8) 4.8e-16 / 9.7e-16 vs 3.6e-16 relative; B4 (kappa1 t4/720.00002)
    5.8e-16 vs 1.1e-16 against the stated polynomial.
  - The literal-statement pins close a band only when the pin sits within about 1 ulp of clean.
  - Concrete kills:
    - C42 Rodrigues <= 1.05e-15 (clean 9.99e-16, deterministic as shown);
    - C22 small-angle relative <= 4e-16 on the test's set (clean 4.30e-16 measured there per the
      hint; re-measure), or a dedicated set on theta < 1.2e-7 asserting the result equals
      2 atan2(n, qw)/n qv to <= 2 ulp;
    - C24 polynomial pin <= 2.5e-16.
  - Process note for the orchestrator: with the rules as written, any pin with more than about
    1 ulp of headroom at the 1e-15 scale admits a non-equivalent survivor. Either pins must sit at
    clean + 1 ulp, or the rules need a sub-ulp floor. I applied the rules as written.

Advisories (not gate items):

- A4: C52's domain text says the ~1.3e-8 near-pi disagreement occurs "in every implementation".
  It does not: src's `log` round-trips to 2.7e-15 at pi - 1e-2..pi - 1e-5, while gtsam's Logmap
  is 9.9e-8 off there. The disagreement is gtsam's. Reword it.
- A5: `gamma_coefficients`' negative-theta ValueError is untested (N13). No claim states it.
  Either drop it from the contract or add a one-line raise check.

## Clean suite

`M1_MUTATION_WORKERS=4 pytest -q tests/` at 3f735b8: **166 passed in 122.8 s**, exit 0. It
includes the 91 harness mutants and `test_unmutated_copy_passes`. `tests/test_gal3.py` with
`tests/test_imu_dynamics.py` alone: 55 passed in 75.9 s. The scratch `_p9/` was deleted before
commit.

## Gate

**FAIL.** 10 non-equivalent survivors:

- N30, N29: G_DEFAULT unpinned (M1-F22);
- H_HEMI fixture, plus YABS_LIFT and YABS_TRUTH under it (M1-F23);
- B1, B2, B3, B3b, B4: rounding bands under the pass-8 pins, 1 to 5 ulp (M1-F24).

All named pass-8 survivors (S01, S02, S03, S06, S07, S08, F1W) and the Y_* compounds are killed
by independent checks. Every claim C01..C55 is confirmed. M1-C63 is false and must not enter the
yaml.
