# M1 seventh-pass adversarial verification

Branch `m1-gal3-mechanics`. The code verified is HEAD 1cd8562 (src/ and tests/ are unchanged
since then). Every reference is independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp`
DOP853/Radau, `logm`), mpmath (40 to 130 digits) and gtsam 4.3.0. No reference imports `src` or
`tests`. The code under test is imported only to be evaluated. Seeds: 70707, 80808, 91919, 12121
and 5. Equivalence rules are applied as the owner stated them:

- rule 1: threshold = 1/10 of the claim tolerance;
- rule 2: rounding floor of 8 ulp, with "no worse than clean" read as the sup-norm error over the
  affected domain. I also report pointwise comparisons;
- rule 3: a written proof.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C61 (brief only) | **REFUTED**: max error 16.0 against the claimed 1e-12 |
| M1-F15 | **FIXED** |
| M1-F16 | **PARTIALLY FIXED**: every named mutant is killed by an independent check, but the new coverage is unguarded (fixture survivors F1, F2), and 6 more assertions of the "cannot fail independently" class remain |
| OPEN items Q03, Q40, N15, Q09, Q12 | **all EQUIVALENT** (numbers below) |
| New mutants | **46** (R01..R43 on src, F1..F3 on fixtures), plus 5 compound mutants: 38 killed, 2 equivalent, **6 SURVIVORS** |
| Harness replay | **23/23 killed**. Every named test failed on its `expect` statement, checked by my own parser. Each mutant has at least one independent-reference kill |
| **Gate** | **FAIL**: 6 surviving non-equivalent mutants (4 low on src, 2 on fixtures; the fixture ones reopen 4 pass-6 medium mutants) |

## M1-C61: REFUTED

H'(Dt1) H'(Dt2) = (I, g(Dt1+Dt2), -0.5 g(Dt1^2 + Dt2^2), 0). Both factors have time 0, so the
v1 t2 coupling vanishes. H'(Dt1 + Dt2) has position -0.5 g (Dt1 + Dt2)^2, so the two differ by
exactly g Dt1 Dt2.

- On 2000 draws (random g, |g| in (2, 20), Dt1 and Dt2 in (-1, 1)): max error **16.002**, equal
  to the predicted max |g|_inf |Dt1 Dt2| (16.002).
- For g = (0, 0, -9.81) and Dt1 = Dt2 = 0.5 the src error is **2.4525**.

The time block being zero is the reason the lifts do NOT form a one-parameter subgroup: position
must then be additive, but it is quadratic in Dt. G(Dt) = T(-Dt) H'(Dt) is the one-parameter
subgroup (C33, confirmed). Gravity over consecutive edges cannot be lifted once over the summed
duration with H'.

## The five OPEN items (mpmath 40-digit Rodrigues; so3_log reference = the generating phi with R = round(Exp_mp(phi)))

"change" means max |mutant - clean|. "err" means max error against the reference.

| item | change | err clean | err mutant | sup-norm clean vs mutant | threshold (rule 1) | ruling |
|---|---|---|---|---|---|---|
| Q03 (`so3_exp` `<=`; 3000 phi with \|phi\| == 0.05 exactly) | **9.99e-16** | 1.1e-16 at 0.05; 9.99e-16 at nextafter(0.05, 0); sup on [0.045, 0.055] 9.99e-16 | 9.99e-16 at 0.05 | equal (9.99e-16 / 9.99e-16) | 1e-15 (C42, 1e-14) | EQUIVALENT by rule 1 (9.99e-16 <= 1e-15) and by rule 2 |
| Q40 (batch `<=`; same points) | 1.11e-15 | 1.1e-16 at 0.05; 9.99e-16 sup | 9.99e-16 | equal | 1e-15 | rule 1 misses by 1 ulp of the change. Rule 2: 1.11e-15 <= 8 ulp(1) = 1.78e-15, and the sup error is unchanged. **EQUIVALENT** |
| N15 (batch switch 1e-3; 8000 angles in [1e-3, 0.05), 2000 of them in [0.045, 0.05)) | 1.11e-15 | 9.99e-16 | **1.1e-16** | the mutant is better; pointwise worse by > 1 ulp on 0/8000 | 1e-15 | rule 2. **EQUIVALENT** |
| Q09 (`>=` ties; 3000 exact ties R00 == R11 > R22 or R00 == R22 > R11, tr <= 0) | 1.11e-15 | 6.66e-16 | 6.66e-16 | equal (60/3000 pointwise worse by <= 1 ulp, the rest equal or better) | 2.5e-16 if a tie falls in the band test's tr range | rule 2 (\|phi\| in [2.09, pi], 8 ulp >= 1.8e-15). **EQUIVALENT** |
| Q12 (trace branch tr > -0.5; 5000 R with tr in (-0.5, 0]) | 1.11e-15 | 6.66e-16 | 6.66e-16 | equal | **1e-13**: theta is in (2.09, 2.42) there, outside the 2.5e-15 band (tr <= -0.9), so the tightest bound is C52 localCoordinates, 1e-12 | **EQUIVALENT** by rule 1. The test comment that applies the band threshold to Q12 is mistaken |

All five are at rounding level and none is worse than the clean code in sup norm. The "OPEN"
markers in tests/test_gal3.py (C22) and tests/test_imu_dynamics.py (C42) can be closed with these
numbers.

## M1-F15: FIXED

- `gal3.so3_exp` carries -th^6/5040. Against mpmath on 346 angles (0, 1e-10..0.05, both sides of
  and exactly on 0.05, up to 12 rad): 9.99e-16, and the batch also 9.99e-16.
- The harness regression `so3_exp_taylor_no_th6` and Q05 (`so3_exp_c_th4_over_700`) were replayed.
  Both are killed by the INDEPENDENT pin `worst_so3_scipy <= 1e-14` (not only by C45's src-vs-src
  product check).
- C45 product identity near the switch (own streams: per-step angles uniform in [0.04, 0.0499],
  60 and 1000 steps, dt = +-0.01, plus dt = +-0.2 large steps): integrate_increment vs ordered
  imu_step_element product **1.69e-13** (<= 1e-12). imu_step_element vs scipy expm/Rotation:
  2.2e-16.
- The pin is loose relative to the 1/10 rule; see survivors R01 and R32. That is a new item, not
  a regression of F15.

## M1-F16: PARTIALLY FIXED

**Fixed.**

- Q49..Q54, Q28..Q31, Q35, Q36 and Q25 were replayed as real edits. All are killed, and each has
  at least one kill against an independent reference: DOP853 truth (C31, C34, C43), hand formulas
  (C32 `worst_def_rg` / `worst_gen_rg`, C35, C36, C38 `_forster_by_hand`), scipy expm (C42, C45),
  gtsam accessors (C40), and a scipy-stepped recursion (C45 `worst_large_nav`, for Q52).
- The four deleted cannot-fail checks are gone, and C42's dp check now uses a test-local series.

**New datasets audited** (tests/conftest.py `endpoint_datasets_rand_g`, rebuilt and compared
with my own solver; data only):

- Own solver: a 5x5 matrix navigation ODE with Radau (rtol 1e-12), and the 5x5 increment ODE
  Ddot = D xi^ with DOP853.
- Dt = +0.4: nav 4.3e-13, increment 3.1e-13.
- Dt = -0.3: nav 1.8e-13, increment 1.6e-13. The backward solve round-trips to I5 within 5.3e-14.
- 4 distinct g per dataset, of genuinely random direction, |g| = 19.8, 7.8, 16.0, 9.6 and 17.0,
  5.4, 18.5, 6.4.
- Recursion first order: error 4.7e-4 / 4.2e-4, ratio 2.0002 / 1.9999. Disc Dt = +0.4 / -0.3.
  Non-commutation gap 0.37 / 0.22.
- `tests/ode_truth.py` imports nothing from src, and the fixture mutant F3 (truth ignores g) is
  killed (C31 `worst_ode` 4.9).

The datasets are independent and correct.

**Not fixed.**

1. The new coverage has no input guard.
   - **F1**: `rand_gravity` returning |g|(0, 0, -1) passes all 55 tests. The compounds F1 + Q50,
     F1 + Q51, F1 + Q52 and F1 + Q53 (the pass-6 harness g-permutation mutants) **all survive**
     (55 passed each).
   - **F2**: making the second dataset forward (dt = +1e-4, still labelled Dt = -0.3) passes all
     55 tests, and F2 + R30 (`gravity_lift_uncorrected` with Dt |Dt|) survives. The datasets are
     C34's only Dt < 0 coverage.
2. Six more assertions of the class F16 named (see "Tests that cannot fail").

## Mutation sweep: 46 new mutants + 5 compounds

Method:

- Each mutant is a textual edit (exact match, count = 1) applied to a fresh `git archive HEAD`
  copy.
- `pytest -x` is run on tests/test_gal3.py and tests/test_imu_dynamics.py.
- For every kill I read the failing statement.
- When that statement compared src with src, I ran a **stage 2**. The 20 such assertions were
  replaced by `pass` (test_gal3.py lines 673, 1027, 1033, 1034, 1058, 1059, 1161-1163, 1193,
  1224-1226, 1308, 1309; test_imu_dynamics.py lines 60, 260, 464, 515, 600). The neutralised
  clean copy passes (55 passed), and the mutant was re-run on it.
- A kill counts only from an independent reference. It also counts transitively, if the other
  side of a src-vs-src check is independently pinned to a tolerance far below the defect (stated
  per mutant).

### SURVIVORS (6, non-equivalent)

| id | edit | measured effect (vs mpmath) | why non-equivalent | what passes it |
|---|---|---|---|---|
| **R01** (low) | `so3_exp` Taylor c gains `+ t4*t2/10000` (wrong th^6 term) | change 3.89e-15; err 4.88e-15 vs clean 9.99e-16 on [0.025, 0.05) | 3.9e-15 > 1e-15 (C42), above 8 ulp(1) and 4.9x worse | killed only by C45 `worst_prod_switch`, src batch vs src scalar. Stage 2: 55 passed. The defect is below both sides' independent 1e-14 pins, so the kill is not attributable, and the same check kills R32 by symmetry |
| **R32** (low) | batch c gains `+ t2^3/10000` | change 3.89e-15; err 4.88e-15 vs 9.99e-16 (on [0.025, 0.03) only 1.1e-16, under the 2.5e-15 pin) | as R01 | as R01; stage 2: 55 passed |
| **R03** (low) | `so3_log` small branch `n < 1e-5` | change 6.65e-16 absolute; err 6.65e-16 vs clean 3.5e-18 (190x worse), theta <= 2.2e-5 | 6.65e-16 > 1e-16 (C22 small-angle bound 1e-15); the quantity is ~2e-5, so 8 ulp ~ 3e-20 | full suite passes (55) |
| **R05** (low) | gamma Taylor c2 `t4/30240 -> t4/30200` | c2 relative change 3.28e-12; err 3.13e-12 vs clean 1.55e-13 (20x) on [1e-3, 0.05); Gamma1^-1 matrix change 6.7e-16 | 3.3e-12 > 1e-12 (C24, 1e-11) | full suite passes. C24's `trunc_at_switch` pins kappa1 only |
| **F1** (fixture) | `rand_gravity` direction fixed to -z | coverage only | the compounds with harness Q50..Q53 survive | 55 passed; F1+Q50/51/52/53 55 passed each |
| **F2** (fixture) | second endpoint dataset forward (dt = +1e-4) | coverage only | the compound with R30 survives | 55 passed; F2+R30 55 passed |

Kills that would close them:

- R01/R32: pin both Rodrigues against mpmath on [0.04, 0.05) at <= 2e-15 (clean 1.0e-15). Rounding
  mutants then fall under rule 2.
- R03: make the C22 small-angle check relative, e.g. |so3_log(R) - phi| / |phi| <= 1e-14 (the
  mutant's relative error is n^2/3 = 3.3e-11).
- R05: pin all four coefficients just below the switch against the series, each within ~2x of
  clean (kappa1 7.75e-13, c2 1.55e-13).
- F1: assert direction diversity of the random g, e.g. that some draw has |g - g[[1,0,2]]| >= 1
  in every random-g block.
- F2: assert `d["Dt"] < 0` for the second dataset, or give C34 an algebraic random-Dt block.

### Proven / measured EQUIVALENT (2)

- **R08** (`so3_exp` Taylor switch at 0.03): on [0.025, 0.05) the change is 9.99e-16 <= 1e-15
  (rule 1). The mutant is MORE accurate: 1.1e-16 vs 9.99e-16, never worse pointwise by more than
  1 ulp (0/4001). This is the scalar twin of N15.
- **R31** (`endpoint_right_time_shift` uses `gravity_lift(abs(Dt))`): rule 3. The position of
  H' X_i D T(-Dt) is p_H + p_i + R_i p_D + (v_H + v_i) Dt - (v_H + v_i + R_i v_D) Dt. Here v_H
  cancels and p_H = -0.5 g Dt^2 = -0.5 g |Dt|^2 is unchanged. Time is unchanged too. C35 claims
  only the position and time blocks. Measured position change 1.1e-16, time 0.
  Advisory: the mutant moves the VELOCITY block of the wrong-side form by 2 |g Dt| (2.66). No
  claim states that block.

### Killed (38)

**Direct kill by an independent reference (26):**

| mutant | kill (claim and check) | failing value |
|---|---|---|
| R02 | C42 so3_exp vs scipy | 3.9e-14 |
| R04 | C22 roundtrip | 2.29 |
| R06 | C24 `trunc_at_switch` vs series, upper bound | 3.19e-12 |
| R07 | the same pin's lower bound (the claimed Taylor truncation is gone) | 5.3e-17 |
| R09 | C11 | 50.1 |
| R10 | C11 | 11.4 |
| R11 | C09 | 11.8 |
| R12 | C08 | 6.8 |
| R13 | C21 | 9.05 |
| R14 | C15 | 1.08 |
| R15 | C15 | 1.24 |
| R16 | C25 hand ref | 1.35 |
| R18 | C30 | 5.14 |
| R20 | C31 ODE | 1.0 |
| R21 | C31 ODE | 2.45 |
| R23 | C34 ODE | 2.45 |
| R25 | C39 gtsam | 3.80 |
| R27 | C50 hand permutation | |
| R28 | C36 guard at nextafter(1e-12) | |
| R30 | C34 ODE, Dt = -0.3 | 1.52 |
| R36 | C41 signed Euler coefficient | 1.28 |
| R37 | C42 expm | 5.46 |
| R38 | C45 expm, dt < 0 | 1.39 |
| R39 | C55 | 5.04 |
| R40 | C55 gtsam vector | 5.04 |
| R42 | C41(c) vs DOP853 | 2.62 |

**Stage 2, independent kill (9):**

| mutant | kill (claim and check) | failing value |
|---|---|---|
| R17 | C26 `worst_rel` vs hand definition | 4.92 |
| R19 | C32 `worst_def_rg` | 2.00 |
| R22 | C35 `worst_p_rg` | 17.7 |
| R24 | C37 gtsam form b vs forster | 8.31 |
| R26 | C38. Proof: `worst_w_rg` compares W_g with the hand (I, g Dt, +0.5 g Dt^2, 0), and the swap puts 0.5 g Dt^2 in the velocity column | 7.35 at `worst_form` |
| R29 | C36 time vs t_i | 6.87 |
| R33 | C45 `worst_large_nav` vs scipy-stepped recursion | 6.60 |
| R34 | C45 `worst_large` vs scipy product | 4.80 |
| R41 | C45 vs three DOP853 solves | 1.9e-2 |

**Transitive (2):**

- **R35** (`D0` time dropped): its time block is off by 0.04 in the C45 identity check against
  src `chain`. `chain` is independently pinned to hand formulas at <= 1e-14 (4000 triples).
- **R43** (integrate_truth's increment updates dv before dp): the identity check is off by 3.1e-4.
  The nav recursion of the same function is pinned to a scipy-stepped recursion at 1e-12, and
  `endpoint` is pinned to DOP853 and hand formulas. Note that no direct independent check of
  integrate_truth's INCREMENT outputs exists below first order: `assert_discrete_converges` alone
  passes R43.

**Fixture (1):** F3 (`ode_truth` ignores g), killed by C31 `worst_ode` (4.9).

## Harness replay (23 earlier entries, real edits, only the named nodes, own `--tb=short` parser)

Killed 23/23, with every named node failing and every `expect` needle in the failing statement:

- all 17 pass-6 entries;
- `bias_to_gtsam_naive`, `chain_swapped_duration`, `fixture_2_steps_dt_0.25`,
  `so3_exp_batch_c_th4_over_700`, `c2_forbidden_form`, `so3_log_trace_branch_below_0`.

Two needles sit on src-vs-src statements:

- `chain_swapped_duration` (C45 line 515): its stage-2 run is killed independently, at C45
  `worst_ode` 5.8e-3;
- `integrate_truth_g_permuted` (C31/C34/C37/C43 identity checks): it is also killed by C45
  `worst_large_nav`, which is independent.

## Per-claim verdicts (independent recomputation)

| claim | verdict | attack / measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | compose vs 5x5 matmul, 2000 pairs, t ~ 2N: 3.6e-15 |
| C02 | CONFIRMED | own residual 2.0e-15. src residual flags each of the 7 zeros, a reflection, a non-orthogonal block and (3,3)/(4,4) at >= 0.3 |
| C03 | CONFIRMED | 3.6e-15 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | vs np.linalg.inv and both products: 1.4e-14; time entry 0.0 |
| C06 | CONFIRMED | worst commutator 17.0 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | vs own hat, both directions: 0.0 |
| C09 | CONFIRMED | ad vs commutator columns: 0.0 |
| C10 | CONFIRMED | 3.6e-15 / 7.1e-15 |
| C11 | CONFIRMED | Ad and Ad_apply vs vee(X e_k^ X^-1), relative 1.0e-15 |
| C12 | CONFIRMED | relative 1.3e-15 |
| C13 | CONFIRMED | relative 2.2e-15 |
| C14 | CONFIRMED | relative 1.1e-15 |
| C15 | CONFIRMED | theta in (1e-3, 2.9): 9.5e-15 |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.4e-15; trace 0.0 |
| C18 | CONFIRMED | vs 60-term series, theta log-uniform in (1e-3, 3): 1.8e-14 (1e-12) |
| C19 | CONFIRMED | 4000 xi, theta in (1e-8, 0.05) and (0.05, 2.9), alpha ~ 1.5N, vs expm: 4.0e-14 (1e-9) |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | c2 vs 60 digits at pi and pi - 10^-k, k = 3..9: 1.4e-16 relative; Gamma1inv times own-series Gamma1, theta in (1e-9, pi): 8.9e-16; log(expm(xi)): 4.2e-15 |
| C22 | CONFIRMED | 2.4e-14; \|phi\| = 3.5 wraps to 2.7832 (2 pi - 3.5) |
| C23 | CONFIRMED | 5000 X including pi - 10^-k (k = 3..12), via expm and src exp: 4.6e-14 |
| C24 | CONFIRMED | det 1.0, 0.4053, 2.213e-3, 2e-65; cond(pi) = 1.5707963. Coefficients vs 130-digit closed forms on 7603 theta (1e-12..10, dense on [0.045, 0.07]): 1.68e-12 at theta = 0.0503 (1e-11) |
| C25 | CONFIRMED | vs own 5x5 solve: 1.3e-15 |
| C26 | CONFIRMED | det 8.9e-16; matrix vs hand I5 - xi^/2: 0.0 |
| C27 | CONFIRMED | 6.7e-16; time 0.0 |
| C28 | CONFIRMED | \| \|c\| - 2 tan(theta/2) \| 6.2e-15. Table: 0.51068, 1.09260, 3.11482, 28.2028, 43171.6 (all round to the stated values) |
| C29 | CONFIRMED | s^3 ratio 0.064 (<= 1); s^5 ratio 8.7e-3 (<= 1) |
| C30 | CONFIRMED | all 15 expressions identical (exact sympy) to an own derivation; 25 symbols are plain real Symbols in order; max degree 2; numeric residuals at Cay 8.9e-16 |
| C31 | CONFIRMED | own stream family (polynomial + cosine rates), own DOP853 nav and 5x5 increment solves, random g, Dt = +0.45 / -0.35 / +0.8: 7.3e-14; time 0; identity on integrate_truth 2.8e-14 |
| C32 | CONFIRMED | 2000 draws, random g, \|Dt\| in (0.05, 1.5) both signs: formulas 3.6e-15, zeta exact, exp vs expm 7.1e-15, opposite order 1.4e-14 |
| C33 | CONFIRMED | product 1.1e-13; formula 0.0 |
| C34 | CONFIRMED | time - Dt 0.0; position - (p_j + g Dt^2) 2.3e-14; R, v 7.3e-14 |
| C35 | CONFIRMED | 7.1e-15; time 0.0 |
| C36 | CONFIRMED | 1.4e-14 / 8.9e-16 / R, v 0.0. Guard: 0, +-5e-13, +-1e-12 accepted; +-nextafter(1e-12), 2e-12, 1e-6, NaN, +-inf refused |
| C37 | CONFIRMED | gtsam W Phi U vs own ODE: 7.3e-14 |
| C38 | CONFIRMED | W_g 7.1e-15, U_g 0.0, both forms vs hand Forster 7.1e-15, time 0.0 |
| C39 | CONFIRMED | gtsam compose vs closed form and vs src navstate_compose: 8.9e-16 |
| C40 | CONFIRMED | Dt in {0.5} and both-sign ranges: definition 0.0; identity factor misses by \|v Dt\| (8.9e-16); automorphism via gtsam 3.6e-15; conjugation and lift 0.0; Phi(I) = I |
| C41 | CONFIRMED | vs expm, Dt in {+-0.02, +-0.4, +-2, +-5, -12, 20}: relative 1.3e-14; own DOP853 4.5e-14; Euler err/dt 1.52685 at n = 1e3 and 1.52684 at 4e3 (first order); dD/ds = D A 1.3e-10 relative; D(0) = I5 |
| C42 | CONFIRMED | blocks vs expm, \|Dt\| in (0.02, 1.3) both signs: 1.8e-14; scalar and batch Rodrigues vs mpmath (346 angles): 9.99e-16 / 9.99e-16 |
| C43 | CONFIRMED | forster_endpoint vs own ODE: 7.3e-14 |
| C44 | CONFIRMED | 4.4e-16; b -> -b gives omega_m + b_g |
| C45 | CONFIRMED | chain vs three separate own DOP853 5x5 solves: 2.7e-14; algebraic 0.0; imu_step_element vs scipy, dt in (-0.2, 0.2): 2.2e-16; near-switch and large-step product identity 1.7e-13 |
| C46-C49 | CONFIRMED | 0.0; Expmap(1e-4 e3) moves velocity; NavState retract moves position |
| C50 | CONFIRMED | 20000 draws: 5.66e-10 (1e-9); unpermuted 5.9e-5; NAVSTATE_PERM equals the stated matrix |
| C51 | CONFIRMED | Expmap 1.9e-14, Logmap 4.4e-15, AdjointMap 1.8e-15, adjointMap 0.0 |
| C52 | CONFIRMED | compose/inverse 7.1e-15, between 1.1e-14, localCoordinates vs scipy logm 8.1e-13 (1e-12; the logm reference itself is the limit), retract vs A expm(d^) 8.9e-16 |
| C53 | CONFIRMED | PIM predict vs src endpoint: 8.9e-16 |
| C54 | CONFIRMED | 0.0 |
| C55 | CONFIRMED | 0.0 (BIAS_SWAP equals the block swap, involutive, round trip) |
| **C61** | **REFUTED** | 16.0 (2.4525 at g = (0, 0, -9.81), Dt1 = Dt2 = 0.5); the claim needs <= 1e-12 |

**Totals: 55 CONFIRMED, 1 REFUTED (C61), 0 UNTESTABLE.**

## Tests that cannot fail (or cannot fail independently)

Each of these is implied by an equality checked just before it plus a property of the random data
(the pattern of the deleted C45 `worst_wrong`):

1. C25 `assert worst_minus >= 1e-3` (test_gal3.py:692). Once cay matches the closed form to
   1e-9, worst_minus = max |S^-1 alpha u|, which is data.
2. C34 `assert least_wrong >= 1e-3` (test_gal3.py:1157). Implied by `worst_p <= 1e-9` and
   |g| Dt^2 >= 0.18.
3. C32 `assert |other - G_Dt| >= 1e-6` (test_gal3.py:1057). Implied by the `worst_other` and
   `worst_fuse` / `worst_def` equalities: the difference is g Dt^2.
4. C22 `assert |back - xi| >= 1e-3` (test_gal3.py:489). Implied by the wrapped-branch equality:
   the difference is 2 pi.
5. C40 `assert max_gap >= 0.5` (test_gal3.py:1421). Implied by the per-draw gap equality and the
   data.
6. C42 `assert |dp| >= 1e-3` (test_imu_dynamics.py:312). Implied by the 1e-15 series check.

Over-specified rather than cannot-fail: C24 `5e-13 <= trunc_at_switch` FAILS for a strictly more
accurate kappa1 (adding its -th^6/40320 term). It pins the claimed truncation, not accuracy.

The src-vs-src checks are not independent references. They can fail, but they cannot by
themselves establish a kill. The full list is the 20 lines neutralised in stage 2: the identity
checks on the discrete recursion (C31, C34, C37, C43), C31 `worst_alg`, C32 `worst_fuse` and
`worst_exp`, C35/C36/C38 first blocks, C25 cay vs cay_matrix, C42 increment_blocks vs
preintegrate_constant, C45 line 515 and `worst_prod_switch`.

The gtsam-only assertions (C46-C49, C50 `worst_un`, C54) cannot fail through src. That is
appropriate for VERIFIED-PUBLISHED claims. Labelled INPUT GUARDs (n_neg, smallest_pert,
order_gap, the angle guards) cannot fail through src by design.

## Clean suite

The full suite (`pytest tests/`, including the 64-entry harness and test_orchestrator) on the
clean tree at HEAD gives **139 passed in 168 s**, exit 0. That run also refreshed
.orchestrator_evidence.json. Afterwards git status was clean apart from this report.

## Gate

**FAIL.** Six non-equivalent survivors remain:

- R01, R32, R03 and R05: tolerance-band defects 3x to 200x above the clean error, but under
  bounds that are about 10x or more loose;
- F1 and F2: unguarded random-g / negative-Dt fixtures. Their compounds reopen pass-6 Q50..Q53
  and kill-path R30.

M1-C61 is refuted and should not enter the yaml as stated.
