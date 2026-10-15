# M1 eighth-pass adversarial verification

Branch `m1-gal3-mechanics`. The code verified is HEAD b91c8ef (src/ unchanged since 1cd8562; tests/
and docs/claims/M1.yaml changed in 6c7ad9e..b91c8ef, the response to M1-F17). Agent verify-M1-009b,
resuming the interrupted verify-M1-009. Its scratch was used as a starting point. Everything this
report relies on was re-run or re-checked.

References are independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp` DOP853), mpmath (50 to
130 digits), sympy and gtsam 4.3.0. No reference imports `src` or `tests`. Re-implemented by me:

- Gal(3) `exp`, `log` (own SO(3) log: atan2 angle, eigen-axis near pi), `Ad`, `compose` and
  `inverse`;
- an own 5x5 navigation/increment ODE (DOP853);
- mpmath Rodrigues, an mpmath SO(3) log, and mpmath Gamma coefficients (series below 1e-3).

Equivalence rules are applied as stated:

- rule 1: measured effect <= 1/10 of the tightest claim tolerance for the affected quantity. Where
  a claim's test_hint now carries a tighter pin, the pin is the tolerance;
- rule 2: the mutant's sup-norm error over the affected domain is within 8 ulp AND no worse than
  clean;
- rule 3: a written proof.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C62 (brief only) | **REFUTED**: 2.584e-13 relative at theta = 0.05, against the claimed < 1e-16 |
| M1-F17 | **FIXED**: all six named survivors (R01, R32, R03, R05, F1, F2) and their compounds are killed by independent checks. The six data-implied assertions, the trunc_at_switch lower bound and the Q12 comment are resolved. Two remedies are incomplete (see M1-F18 and M1-F21) |
| Fresh sweep | 101 src mutants (56 from verify-M1-009's list, re-judged by the full suite, plus 45 new) and 17 fixture or compound mutants. Results: **6 src survivors and 1 fixture survivor (+4 compounds) are NON-EQUIVALENT**; 31 sweep survivors plus R07 are proven or measured equivalent; the rest are killed |
| Harness replay | tests/test_mutation.py: **70/70 killed** (+ the unmutated-copy check), inside the clean full run |
| Clean suite | `pytest` (all of tests/): **145 passed in 97 s**, exit 0 |
| **Gate** | **FAIL**: 7 non-equivalent survivors (S01, S02, S03, S06, S07, S08 on src; F1W on the fixture) |

## The interrupted pass: C21 and C28 "BAD" were script bugs

`_p8/claims_a.out` flagged two claims.

- **C21 BAD (5.4e-15)**: the script's verdict also required `c2 rel <= 1e-13` over a point set that
  included theta = 0.0499 and 0.051. Those points are outside the claim's stated points
  (pi, and pi - 10^-k for k = 3..9). At 0.0499 the Taylor branch is legitimately at 1.53e-13
  (C24 pins it at <= 3e-13). Re-run on the claim's points: 2.3e-17..1.05e-16. **Not a
  refutation.**
- **C28 BAD (5.3e-15)**: the table check compared `f"{2 tan(theta/2):.5g}"` with values tabulated
  to 4 significant figures. 0.510684 rounds to 0.51068 at 5 s.f. but to the tabulated 0.5107 at
  4 s.f. Re-run at the tabulated precision, through `cay`: every theta is recovered within the
  4-s.f. tolerance. **Not a refutation.**

## M1-C62: REFUTED

Claim: for 0 < theta <= 0.05, 1/6 - theta^2/120 + theta^4/5040 approximates
kappa2 = (theta - sin theta)/theta^3 to relative error below 1e-16.

Measurement: 10005 thetas on [1e-4, 0.05], plus 1e-6, 1e-3, 1e-2, nextafter(0.05, 0) and 0.05. The
polynomial is evaluated in 50-digit arithmetic, so there is no rounding.

- max relative error **2.5838e-13 at theta = 0.05**. This equals the first omitted term,
  (theta^6/9!)/(1/6) = 2.583e-13.
- The error is >= 1e-16 for every theta >= 0.0135.
- src kappa2 at nextafter(0.05, 0): 2.584e-13 relative.

The claim is false by more than three orders of magnitude at the switch. It is also inconsistent
with C24's own pin (kappa2 <= 5e-13, measured 2.58e-13). The Taylor branch is NOT exact to machine
precision at 0.05: kappa1 7.75e-13, kappa2 2.58e-13, kappa3 1.03e-13, c2 1.55e-13.

## M1-F17 ruling: replay of the pass-7 survivors

Each survivor was applied as a real, exactly-once textual edit (count == 1 enforced) on a temporary
copy of src/ and tests/. The copy was judged by the full tests/test_gal3.py + tests/test_imu_dynamics.py
run, with no assertion removed.

| survivor | edit | now | killed by (failing statement, value) | independent? |
|---|---|---|---|---|
| R01 | `so3_exp` Taylor c `+ t4*t2/10000` | **KILLED** | C42 `worst_sw_scalar <= 2e-15`: 4.88e-15; also C45 `worst_prod_switch` | yes (mpmath Rodrigues) |
| R32 | batch c `+ t2^3/10000` | **KILLED** | C42 `worst_sw_batch <= 2e-15`: 4.88e-15; also C45 `worst_prod_switch` | yes |
| R03 | `so3_log` small branch `n < 1e-5` | **KILLED** | C22 `worst_small_rel <= 1e-14`: 3.23e-11 | yes (50-digit mpmath log) |
| R05 | c2 Taylor `t4/30240 -> t4/30200` | **KILLED** | C24 `c2_below <= 3e-13`: 3.13e-12 | yes (series reference) |
| F1 | `rand_gravity` fixed to -z | **KILLED** | input guard `off_z >= min_angle` in 10 tests: 0.0 | guard (by design) |
| F2 | Dt = -0.3 dataset run forward | **KILLED** | `assert_rand_g_datasets` sign check in 4 tests: Dt = +0.3 | guard |
| F1 + Q50/Q51/Q52/Q53 | | KILLED (all 4) | same guard, 10 tests each | |
| F2 + R30 | | KILLED | same sign check | |

The previous pass's degenerate fixtures were re-judged from verify-M1-009's log:

- **F1X** (g in the x-z plane by masking y, then renormalised): killed by C45's `off_z` guard
  (0.1225).
- **F2X** (Dt = -3e-4): killed by `order_gap >= 0.1` and `|g| Dt^2 >= 1e-3`.
- **Y_LIFT, Y_ELEM, Y_TRUTH, Y_FORST** (src defects visible only through g_y) are killed alone, and
  with F1X.

The other F17 items:

- The six data-implied assertions are deleted or rewritten as data-only INPUT GUARDs: C22
  `|back - xi|`, C25 `worst_minus`, C32 `|other - G_Dt|`, C34 `least_wrong`, C40 `max_gap` and
  C42 `|dp|`.
- The C24 `5e-13 <= trunc_at_switch` lower bound is gone.
- The Q12 comment now applies the C52 1e-13 threshold.

**Ruling: M1-F17 FIXED.** Two remedies are incomplete; each is a new finding below.

- **R05 remedy.** It pinned each coefficient's accuracy, not the coefficients themselves. Three
  1e-5-relative perturbations of the th^4 coefficients survive (S01, S02, S03, M1-F18). R07 itself
  (kappa1 gains its true th^6 term) now survives and is EQUIVALENT by rule 2: sup error on the
  Taylor domain 1.1e-16 vs clean 7.75e-13. This is the outcome pass 7 asked for when it called
  the lower bound over-specified.
- **F1 remedy.** The guard admits a planar, off-z random-g distribution, and g_y-only src defects
  survive under it (F1W, M1-F21).

## Mutation sweep

Method:

- Real textual edits with an exactly-once match, applied to a `tempfile` copy of src/ and tests/.
- Judged by the full `pytest` run of tests/test_gal3.py and tests/test_imu_dynamics.py. (I ran
  test_orchestrator.py separately: it does not import src. test_mutation.py replays its own
  mutants.)
- verify-M1-009's 80 runs used a stricter "stage-2" mode: the src-vs-src assertions were replaced by
  `pass`, so a kill there implies a kill in the full suite. Its 30 stage-2 survivors were re-run on
  the unmodified suite. Only S59 is killed there: integrate_truth's increment updated after dR,
  caught by the C31/C34/C37 identity checks at 8.0e-4. That kill is transitive, because the nav
  recursion is pinned to scipy and the endpoint relation is pinned to DOP853.
- For every survivor I measured clean and mutant outputs on fixed inputs (`_p8/probe.py`) against
  own mpmath references:
  - Gamma coefficients on 3208 thetas, dense on [0.04, 0.07];
  - Rodrigues on 1408 rotation vectors, dense on [0.025, 0.052];
  - so3_log on 3000 small angles (1e-10..1e-2) and 6000 generic or near-pi rotations;
  - 3000 exact diagonal ties for the Shepperd branches.

### NON-EQUIVALENT SURVIVORS (7)

| id | edit | measured effect | why non-equivalent |
|---|---|---|---|
| **S01** (low) | gamma Taylor `kappa1 ... + t4/720.0 -> t4/719.993` | kappa1 relative change 1.69e-13; sup error on (0, 0.05) 9.44e-13 vs clean 7.75e-13 | 1.69e-13 > 1e-13 (1/10 of C24's kappa1 pin, 1e-12); worse than clean. Passes the pin with 6% headroom |
| **S02** (low) | `kappa3 ... t4/40320.0 -> t4/40319.0` | kappa3 relative change 9.2e-14; sup error 1.955e-13 vs clean 1.033e-13 (1.9x) | 9.2e-14 > 2e-14 (1/10 of the 2e-13 pin); worse than clean. Passes the pin with 2% headroom |
| **S03** (low) | `c2 ... t4/30240.0 -> t4/30239.0` | c2 relative change 8.2e-14; sup error 7.3e-14 vs clean 1.55e-13 (more accurate, but ~500 ulp away from the stated polynomial) | 8.2e-14 > 3e-14 (1/10 of 3e-13); not within 8 ulp; contradicts C24's stated coefficient 1/30240 |
| **S06** (low) | `so3_exp` Taylor c `+ t4*t2/60000.0` | matrix change 6.66e-16; sup error on [0.04, 0.05) 1.665e-15 vs clean 9.99e-16 | 6.66e-16 > 2e-16 (1/10 of C42's 2e-15 pin); 7.5 ulp but WORSE than clean (rule 2 fails) |
| **S07** (low) | batch c `+ t2^3/60000.0` | same numbers as S06 on the batch | as S06 |
| **S08** (low) | `so3_log` small branch `n < 1e-10 -> n < 1e-7` | relative error 3.39e-15 vs clean 4.37e-16 (7.8x), theta <= 2e-7; absolute 5.6e-22 | 3.4e-15 > 1e-15 (1/10 of C22's relative 1e-14 pin); ~15 ulp, worse than clean |
| **F1W** (fixture, low-medium) | `rand_gravity` direction = (sin b, 0, -cos b), b uniform in (0.6, pi - 0.6): planar, off z | passes every guard: off_z >= 0.6, perm_gap >= 1, spread >= 0.5 | with it, **Y_LIFT, Y_ELEM, Y_TRUTH, Y_FORST** (g -> g * (1, -1, 1) in gravity_lift, gravity_element, integrate_truth, forster_endpoint) **all survive the full suite**; each is killed alone (worst_ode 10.0, worst_def_rg 29.8, e1 10.0, worst_alg 33.8) |

### EQUIVALENT SURVIVORS (31 from the sweep + R07, each with its measurement)

Rule 1 (effect <= 1/10 of the tightest tolerance):

| id | edit | measured effect | tolerance / 10 |
|---|---|---|---|
| S23 | log: `G1i @ p - t*(G1i @ (G2 @ u))` | log change 3.6e-15 | C52 localCoordinates 1e-13 |
| S26 | so3_exp s `/5040 -> /5041` | 3.5e-17 | C42 2e-16 |
| S45 | inverse distributed | 3.6e-15 (time unchanged, exact) | C05/C52 1e-11 |
| S48 | gamma switch `<=` (changes theta == 0.05 only) | at 0.05: k1 7.75e-13, k2 2.58e-13, k3 1.03e-13, c2 1.55e-13 relative (clean 2.2e-16, 3.6e-14, 7.1e-13, 8.2e-13) | C24 table bound at 5e-2: 1e-12 |
| S50 | right-time-shift: compose re-associated | 3.6e-15 | C35 1e-11 |
| S51 | cay: `inv(S) @ rhs` | 7.1e-15 abs; on C26's domain (\|phi\| <= 50, \|u\|,\|w\| ~ 1e3) 2.4e-15 relative | C26 1e-13 relative |
| S56 | batch c as 2 sin^2 | Rodrigues 1.1e-16; large-step integration 2.4e-16 relative | C42 2e-16; C45 1e-13 |
| S75 | Ad_apply `R @ (w - t u)` | 3.6e-15 | C11 1e-10 |
| N13 | kappa2 closed `(theta - s)/(t2*t2)*theta` | 3.3e-16 relative | C24 5e-14 |
| N31 | so3_log trace branch for tr > -0.5 (pass-7 Q12) | phi 1.3e-15, log 8.0e-15 (theta in (2.09, 2.42), outside the tr <= -0.9 band) | C52 1e-13 |
| N33 | so3_exp closed c as (1 - cos)/th^2 | 1.1e-16, sup error unchanged | C42 2e-16 |
| M19 | integrate_truth increment: dv first, then `dp + dv dt - half df` (algebraically identical) | 5e-17..9e-17 relative | C31/C45 1e-13 |

Rule 2 (mutant sup error within 8 ulp and no worse than clean):

| id | edit | mutant / clean sup error |
|---|---|---|
| R07 | kappa1 gains -th^6/40320 | 1.1e-16 / 7.75e-13 (relative, Taylor domain) |
| S04 | kappa2 gains -th^6/362880 | 3.3e-16 / 2.58e-13 |
| S05 | c2 gains +th^6/1209600 | 3.3e-16 / 1.55e-13 |
| S09 | so3_exp c gains -th^6/40320 | 3.9e-16 / 9.99e-16 |
| S24 / S25 | scalar / batch Rodrigues switch at 0.04 | 3.9e-16 / 9.99e-16 |
| S54 | Shepperd `R11 >= R22` (3000 exact ties, theta in (2.1, pi)) | change 1.1e-15 (2.5 ulp); sup 1.33e-15 / 1.33e-15 |
| N35 / M15 | scalar / batch Rodrigues `<=` at the switch | change 8.9e-16 (4 ulp) at theta = 0.05 only; sup unchanged |

R07, S04, S05 and S09 are strictly more accurate. They do contradict C24's literal "3-term"
statement (advisory A3).

Rule 3 (proof; the measured change is 0 in every case):

- **S11** (`n == 0`): for 0 < n < 1e-10, 2 atan2(n, qw)/n and 2/qw differ by a relative n^2/3 <
  3.4e-21, below 1 ulp.
- **S15** (`endpoint` left = `gravity_element`): compose(T(-Dt), H') adds exact zeros and
  multiplies by I3. It is bitwise (I, g Dt, -0.5 g Dt^2, -Dt).
- **S17** (`lift_navstate` time from N[3,4]): a NavState's (3,4) entry is 0 by construction.
- **S18** (`R R^T` in the residual): for square R, R^T R = I iff R R^T = I. Valid-element residual
  change 2.2e-16. Every C02 defect is still flagged >= 0.2, and on 2000 random non-orthogonal
  blocks the minimum flag is 0.059 vs 0.061.
- **S22** (`0.5 theta s / omc`): the two forms differ only by exact power-of-two scalings.
- **S30** (`euler_increment` dp uses \|dt\|): bitwise identical for dt > 0. That is the only
  domain any claim gives the Euler scheme (C41(b), (b'), (c): Dt = 0.4, dt > 0). For dt < 0 it
  changes the output by 3.86 (advisory A1).
- **S43** (`unskew` reads -A[2,0]): identical on skew blocks, i.e. on gal(3), the C08 domain.
- **S74** (`P.T @ P.T`): P^T = -P exactly, so the product is bitwise P P.
- **N32** (`qw <= 0`): qw == 0 only at theta = pi, where phi and -phi are both logs of R. C22
  excludes pi; C23 holds for either.
- **N36** (small branch returns 2 qv): qw = cos(theta/2) rounds to 1.0 for theta < 2e-10 (the
  claim's own documented pass-3 V02).
- **M06** (`_inputs` drops the acc length check): identical on every valid input. Only error
  handling for a malformed stream changes (advisory A2).

### Killed (fresh, 36 of 45; plus verify-M1-009's 28 of 56 re-judged)

New mutants and their kills (statement, value):

- N03 inverse `p + t v`: C05 51.7
- N04 residual without X[3,3]: C02 `diag3` 0.0 flagged
- N06 alpha at (4,3): C08 3.41
- N07 ad -alpha sign: C09 15.3
- N08 ad column 9 = w: C09 11.8
- N09 Ad column 9 = p: C11 29.0
- N10 Ad `p + t v`: C11 50.1
- N11 Ad_apply `alpha p`: C11 29.0
- N15 Gamma1^-1 `+P/2`: C21 2.00
- N16 log uses G2 v: C21 9.05
- N18 cay alpha w: C25 6.04
- N19 cay_matrix inverted: C25 8.70
- N20 residual alpha w: C30 5.70
- N21 symbolic E2 uses w: C30 5.20
- N22 time_shift(-dt): C31 ODE 1.0
- N23 generator time +1: C32 9.71
- N24 keyframe `<`: C36 edge +-1e-12 refused
- N25 check_time off by default: C36 DID NOT RAISE
- N26 right shift `+Dt`: C35 25.9
- N27 Forster 0.5 g Dt: C31 / C37 / C43 1.23
- N28 NavState compose: C39 5.06
- N29 lift swap: C38 / C40 4.90
- N30 identity permutation: C50
- N37 so3_exp s `/121`: C41 / C42 / C52 1.8e-11
- M01 accel corrected by b_g: C44 5.41
- M02 bias_from swapped: C55 5.04
- M03 BIAS_SWAP = I6: C55 5.04
- M04 batch K21 sign: C31 ODE 6.40
- M05 step without 1/2: C45 vs scipy 0.116
- M07 D0 time dropped: C45 0.04
- M09 truth Dt abs: C31 sign guard
- M11 generator time -1: C41 26.0
- M12 chain swapped: C45 0.019
- M16 skew_batch sign: C41 1.95
- M17 truth drops R a: C31 identity 1.5e-4 (transitive, as pass-7 R43)
- M20 Euler semi-implicit: C41 (b') 2.57

The 28 killed mutants from verify-M1-009's list: S10, S12-S14, S16, S19-S21, S27-S29, S34, S38-S41,
S47, S49, S53, S55, S58, S59, S61, S62, S64, S73, S79, S80. Failing values are in that log (e.g.
S53 Shepperd `or`: C22 band 2.16e-14; S55 THETA_SMALL 4.9e-2: input guards).

## Harness replay

tests/test_mutation.py ran in the clean full suite: 70 harness mutants plus
`test_unmutated_copy_passes`, all passed. That means each mutant made every named test fail on its
`expect` statement. This includes the six F17 entries added in b91c8ef:

- `so3_exp_c_th6_over_10000`
- `so3_exp_batch_c_th6_over_10000`
- `so3_log_small_branch_1e-5`
- `c2_taylor_th4_over_30200`
- `fixture_rand_gravity_minus_z`
- `fixture_neg_dt_dataset_forward`

The four src entries were independently replayed above (R01, R32, R03, R05) and fail on mpmath or
series references.

## Property tests and the hunt for spurious behaviour (own re-implementation)

Each line gives the samples, the quantity, and the result:

- 12000: src exp vs own exp (own Gamma, 10-term mpmath series below 0.2), theta log-uniform in
  (1e-9, 0.2) and uniform to pi - 1e-3: 2.3e-14 relative.
- 12000: src log of own exp: 3.5e-15.
- 12000: own log vs src log on random X: 2.1e-15.
- 12000: src Ad vs own Ad: 0.0.
- 12000: compose vs 5x5 matmul: 8.1e-16.
- 12000: inverse vs `np.linalg.inv`: 2.3e-15.
- 20000: Gamma1^-1(phi) times the own 40-term series Gamma1, theta log-uniform in (1e-9, pi):
  6.7e-16 (C21, bound 1e-13).
- 10000: log vs the C21 formula with own linear solves: 6.4e-15.
- 10000 + 10000: Cay rotation, \|c\| in (0.01, 5): 7.1e-15. Log-uniform (5, 5e4): angle 9.5e-13,
  axis 6.8e-13 (C28, 1e-8 / 1e-9).
- 10000: so3_log at theta = pi - 10^-u, u in (6, 12): exp(log R) - R and \|phi\| - theta 1.3e-15.
  No branch flips.
- 10000: bias: `correct_measurements` vs gtsam `correctGyroscope`/`correctAccelerometer` through
  `bias_to_gtsam`, and the round trip: 0.0.
- 10000: **structural_residual on single-entry defects** of size 1e-6..1e-1 in every structural
  entry: flagged >= 0.675 x the defect size. No defect reads as Gal(3).
- 10000: **Cayley constraint block, spurious feasible points**: for random (Re, ve, pe) = Cay blocks
  + d, \|residual\| / \|d\| >= 1.0025. Since sigma_min(I - [phi]x/2) = 1, the zero set is exactly
  the Cay point: there is no spurious feasible point, and Re needs no det constraint.
- Keyframe guard: 0, -0.0, +-1e-12 and 5e-13 accepted; nextafter(1e-12, inf), NaN and +-inf
  refused.

## Per-claim verdicts

Claims C01..C30 come from `_p8/claims_a.py` (re-run and audited). C21, C28 and C62 were re-done in
`_p8/fix.py`. C31..C55 come from `_p8/claims_b.py`, with own DOP853 streams, own random g, and
Dt = +0.45, -0.35, +0.8.

| claim | verdict | measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | 3000 triples, t ~ 2N: 3.6e-15 (1e-12) |
| C02 | CONFIRMED | 1.8e-15 (1e-12); src residual flags reflection, scaling, all 7 zeros, (3,3), (4,4) at >= 0.158 |
| C03 | CONFIRMED | 7.1e-15 (1e-10) |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 1.2e-14 (1e-10); time entry 0.0 |
| C06 | CONFIRMED | max commutator 29.0 (>= 1e-3) |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0 |
| C09 | CONFIRMED | 1.8e-15; last row 0; (6:9, 9) = u |
| C10 | CONFIRMED | 7.1e-15 |
| C11 | CONFIRMED | Ad and Ad_apply vs vee(X e_k^ X^-1): 1.7e-15 relative |
| C12 | CONFIRMED | 1.1e-15 relative |
| C13 | CONFIRMED | 1.7e-15 relative |
| C14 | CONFIRMED | 2.8e-15 relative |
| C15 | CONFIRMED | 1.1e-14 (1e-8) |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.2e-15; trace 0.0 |
| C18 | CONFIRMED | vs 60-term series, theta log-uniform in (1e-6, 3): 1.75e-14 (1e-12) |
| C19 | CONFIRMED | 4000 xi, theta in (1e-8, 0.05) and (0.05, 2.9), vs expm: 3.5e-14 (1e-9) |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | c2 at pi and pi - 10^-k: <= 1.05e-16 (1e-13), finite at pi; Gamma1^-1 Gamma1 - I: 6.7e-16 (20000; 1e-13); log formula 6.4e-15 |
| C22 | CONFIRMED | 2.5e-14 (1e-9); \|phi\| = 3.5 wraps to 2.78319 |
| C23 | CONFIRMED | 5000 X including pi - 10^-k: 2.7e-14 (1e-9) |
| C24 | CONFIRMED | det 1.0000, 0.4053, 2.2128e-3, ~0; cond(pi) 1.57080; coefficients vs 130-digit: 1.42e-12 at theta = 0.0654 (1e-11); just below the switch k1 7.75e-13, k2 2.58e-13, k3 1.03e-13, c2 1.55e-13 (pins 1e-12, 5e-13, 2e-13, 3e-13) |
| C25 | CONFIRMED | 1.3e-15 (1e-9) |
| C26 | CONFIRMED | det 1.3e-15; matrix vs hand I5 - xi^/2: 0.0 |
| C27 | CONFIRMED | 6.7e-16; time 0.0 |
| C28 | CONFIRMED | 7.1e-15 (1e-8, 10000 draws); large \|c\| angle 9.5e-13 (1e-9); table 0.5107, 1.0926, 3.1148, 28.203, 4.3172e4 recovered through cay within the 4-s.f. tolerance |
| C29 | CONFIRMED | s^3 ratio 0.063 (<= 1); s^5 ratio 8.3e-3 (<= 1) |
| C30 | CONFIRMED | the 15 expressions equal an own sympy derivation exactly; 25 plain real Symbols in order; degree 2; residuals 1.3e-15 |
| C31 | CONFIRMED | own DOP853, random g, Dt = +0.45/-0.35/+0.8: 3.1e-14 (1e-9); time 0.0 |
| C32 | CONFIRMED | formulas 1.4e-14; zeta exact; exp vs expm 1.1e-14; opposite order differs as stated |
| C33 | CONFIRMED | 8.5e-14 (1e-10) |
| C34 | CONFIRMED | position - (p_j + g Dt^2) 1.6e-14; time - Dt 0.0; R, v 3.1e-14 |
| C35 | CONFIRMED | 7.1e-15; time 0.0 |
| C36 | CONFIRMED | 1.4e-14 / time 8.9e-16; guard as above |
| C37 | CONFIRMED | gtsam W Phi U vs own ODE: 3.1e-14 |
| C38 | CONFIRMED | 3.6e-15; forms vs Forster 7.1e-15; time 0.0 |
| C39 | CONFIRMED | 8.9e-16 |
| C40 | CONFIRMED | definition 0.0, automorphism 2.7e-15, conjugation 0.0, lift 0.0, Phi(I) = I |
| C41 | CONFIRMED | vs expm 1.9e-14 relative; own DOP853 6.7e-14; Euler err/dt 1.72246 (n = 1e3), 1.72222 (n = 4e3); D(0) = I5 |
| C42 | CONFIRMED | blocks 5.1e-15 (1e-12); scalar and batch Rodrigues vs mpmath 9.99e-16 / 9.99e-16, also on [0.04, 0.052] |
| C43 | CONFIRMED | 3.1e-14 |
| C44 | CONFIRMED | 4.4e-16; sign correct; matches gtsam correct* (0.0) |
| C45 | CONFIRMED | vs three own DOP853 solves 2.8e-14; algebraic 8.9e-16; step element 6.7e-16 |
| C46 | CONFIRMED | 0.0 |
| C47 | CONFIRMED | 0.0 |
| C48 | CONFIRMED | 0.0; Expmap(1e-4 e3) moves velocity |
| C49 | CONFIRMED | 0.0; NavState retract moves position |
| C50 | CONFIRMED | 20000 draws: 6.5e-10 (1e-9); unpermuted 6.8e-5; NAVSTATE_PERM as stated |
| C51 | CONFIRMED | Exp 3.8e-14, Log 1.2e-14, Ad 1.8e-15, ad 0.0 |
| C52 | CONFIRMED | compose 1.8e-15, inverse 8.0e-15, between 8.9e-15, localCoordinates 1.5e-14, retract 1.8e-15 |
| C53 | CONFIRMED | 3.6e-15; time 0.0 |
| C54 | CONFIRMED | 0.0 |
| C55 | CONFIRMED | 0.0; BIAS_SWAP is the block swap |
| **C62** | **REFUTED** | 2.584e-13 relative at theta = 0.05 (claimed < 1e-16); >= 1e-16 for all theta >= 0.0135 |

**Totals: 55 CONFIRMED, 1 REFUTED (C62), 0 UNTESTABLE.**

## New findings

- **M1-F18 (low): the Taylor coefficients of `gamma_coefficients` are pinned by accuracy only, not
  to their stated values.**
  - Survivors: S01 (kappa1 t4/719.993), S02 (kappa3 t4/40319), S03 (c2 t4/30239).
  - Kill: on a grid of (0, 0.05), e.g. 2000 points dense near 0.05, assert each returned coefficient
    equals C24's 3-term polynomial evaluated in 50-digit arithmetic to <= 1e-15 relative (clean:
    rounding, <= ~2 ulp). S01 1.7e-13, S02 9.2e-14 and S03 8.2e-14 fail.
  - This also kills R07/S04/S05, which are more accurate but contradict the literal statement.
    Owner's choice: if only accuracy is wanted, tighten the pins to ~1.1x clean instead
    (kappa1 <= 8.5e-13 kills S01; kappa3 <= 1.2e-13 kills S02). S03 then needs the polynomial
    pin.
- **M1-F19 (low): the Rodrigues Taylor band pin (2e-15) admits a th^6/60000 perturbation of
  (1 - cos)/th^2 in both the scalar and batch forms.**
  - Survivors: S06, S07. Mutant 1.665e-15 vs clean 9.99e-16 on [0.04, 0.05).
  - Kill: assert <= 1.3e-15 on C42's 460 angles (seed 20241012+4201).
- **M1-F20 (low): the so3_log small-angle branch can move to n < 1e-7.**
  - Survivor: S08. Relative error 3.4e-15 vs clean 4.4e-16, under C22's 1e-14.
  - Kill: lower the C22 relative small-angle bound to 1e-15 on the same 20000 R. A branch at
    n < 1e-8 (n^2/3 ~ 3e-17) is then below rounding and equivalent.
- **M1-F21 (low-medium, fixture): `assert_gravity_directions_diverse` admits a PLANAR random-g
  distribution.**
  - Survivor: F1W, g in the x-z plane and >= 0.6 rad off z, passes off_z, perm_gap and spread.
    Under it, Y_LIFT, Y_ELEM, Y_TRUTH and Y_FORST (a g_y sign error in gravity_lift,
    gravity_element, integrate_truth or forster_endpoint) survive the whole suite. The G_DEFAULT
    blocks have g_y = 0 too.
  - Kill: require the block's unit directions to span R^3, e.g. smallest singular value of the
    stacked unit vectors / sqrt(m) >= 0.2, or each component |g_k| >= 1 in some draw. For the
    4-g datasets apply it to the 4 g's. F1W fails this (rank 2), and real draws pass it.

Advisories (no claim affected; not gate items):

- A1: `euler_increment` is never run with dt < 0. S30 (dp uses |dt|) changes it by 3.86 there.
  Either restrict its docstring to dt > 0 or add a signed-dt Euler check.
- A2: the `_inputs` per-step acc-length check is untested (M06). An over-long acc stream is
  silently truncated.
- A3: C24 states a literal 3-term branch, yet strictly more accurate 4-term variants survive (R07,
  S04, S05; rule-2 equivalent). Decide which is the claim and pin it (see F18).

## Clean suite

`pytest -q` over all of tests/ at HEAD (b91c8ef plus the pass-8 stub): **145 passed in 97.1 s**,
exit 0. git status afterwards: only `_p8/` (scratch, deleted before commit) and this report.

## Gate

**FAIL.** 7 non-equivalent survivors remain:

- S01, S02, S03: Taylor coefficients (M1-F18);
- S06, S07: Rodrigues Taylor band (M1-F19);
- S08: so3_log small branch (M1-F20);
- F1W: planar-g fixture, which reopens 4 g_y src defects (M1-F21).

All are low severity: rounding-band defects 1.2x to 8x above clean, under pins with 1.06x to 3x
headroom, plus one fixture-coverage gap. Every claim C01..C55 is confirmed. M1-C62 is false and
must not enter the yaml.
