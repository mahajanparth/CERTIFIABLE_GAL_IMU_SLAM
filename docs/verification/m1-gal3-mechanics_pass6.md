# M1 sixth-pass adversarial verification (Gal(3) mechanics and IMU dynamics)

Branch `m1-gal3-mechanics`, verified at `9cda195`. gtsam 4.3.0, numpy, scipy, sympy, mpmath.

- I did not read `docs/01_conventions_and_gal3.md`, any derivation narrative, any seeds file or
  anything under `.cgis_SEEDS_BACKUP*`. Nothing in `src/`, `tests/` or `docs/claims/` was edited.
- All mutants ran on scratch copies (session scratchpad). `git status` was clean afterwards.
- Independent references (scripts in the scratchpad, run with `python -I`) import nothing from
  `tests/`; from `src/` they import only the function under test. Authorities: scipy `expm`,
  `Rotation`, `solve_ivp` (DOP853, rtol = atol = 1e-13), mpmath (40-80 digits), gtsam, and an
  independent sympy derivation.

## Summary

| item | result |
|---|---|
| Claims M1-C01..C55 (yaml) | **55 CONFIRMED**, 0 REFUTED, 0 UNTESTABLE (C45 carries a falsified test-hint sub-assertion, F15) |
| M1-C60 (brief only) | **REFUTED**: relative error 3.10e-12 at theta = 0.05, 3.1e4 x the claimed 1e-16 |
| M1-F13 | **FIXED**: P03, P08, P10, P11, P13, P15, P16, P19 and P22 are all killed on the assertion written for them; the hints are current |
| M1-F14 | **FIXED** for every named item. The V02, V18, V39 and N15 equivalence proofs hold over the whole claimed domain. The fix itself opened a new inconsistency (F15, below) |
| Harness replay (all 45 `MUTATIONS`, as real file edits, both claim files run in full) | **45/45 killed**, each on every named test and on the assertion its needle names |
| Own sweep (54 new mutants, Q01..Q54) | 25 killed, 15 proven equivalent, **14 SURVIVE** |
| **Gate (zero non-equivalent survivors)** | **FAILS**: 14 survivors (12 medium, 2 low) |
| Full suite, clean tree | 120 passed at the start (197 s) and at the end, on the committed report (405 s, under load) |

## M1-C60: REFUTED

The claim says 1 - theta^2/6 + theta^4/120 matches sin(theta)/theta to a relative error below
1e-16 for 0 < theta <= 0.05.

The first omitted term of the series is theta^6/5040. At theta = 0.05 that is 3.1e-12 relative.

| theta | relative error (mpmath, 50 digits) |
|---|---|
| 0.05 | **3.10e-12** |
| 0.0275 | 8.6e-14 |
| 0.01 | 2.0e-16 |
| 0.005 | 3.1e-18 |

The bound fails for every theta above (5040e-16)^(1/6) = **8.9e-3**. The worst error is the same
evaluated in double precision (3.10e-12) or exactly, so this is truncation, not rounding.

The batched exponential `_so3_exp_batch` needs the theta^6 term, and since 59330b2 it has it.
The scalar `gal3.so3_exp` still uses exactly the polynomial of C60 below 0.05. Its matrix error
against a 40-digit Rodrigues is **1.33e-13**, against 1.1e-15 for the batch. See F15.

## M1-F13 ruling: FIXED

All nine pass-5 survivors are now harness entries. I replayed them as real file edits, ran both
claim files in full, and parsed the failing statement myself: I took the source line directly
above the first `E ` line, independently of `_failing_statement`. Each fails on the assertion
written for it:

- P03, P10, P13 and P16 fail on `sym_mismatch`;
- P08 and P11 fail on `bad_domains`;
- P15 fails on `not_expr`;
- P19 and P22 fail on `worst_long`.

The C30 and C41 test_hints now describe these pins, and C41's domain states the tested Dt range.
I found no stale hint among C21, C22, C30, C36, C41 and C42.

`assert syms == expected_syms` and `len(set(syms)) == 25` add nothing, though. They are implied
by the name-list equality and the per-element `type(x) is sp.Symbol and x == y` check (see
"Tests that cannot fail").

## M1-F14 ruling: FIXED, with every equivalence proof checked

| item | status | evidence |
|---|---|---|
| V06 (structural_residual without det) | killed | replay |
| V33 (batch Taylor without th^4) | killed | replay (`so3_exp_batch_taylor_no_th4`) |
| V44 (Shepperd trace branch for tr > -0.99) | killed, robustly | see below |
| N06 (Re column-major) | killed | replay |
| N14 (tolerance 1.9e-12) | killed | replay; nextafter pin |
| N19 (batch c th^4/700) | killed | replay. **Its scalar twin in `gal3.so3_exp` survives (Q05)** |
| NaN guard | fixed | independent check: NaN, +inf and -inf are refused; +-1e-12, 0 and -0.0 are accepted; +-nextafter(1e-12) are refused |
| C24 recorded figures | correct | independent 80-digit reference on 30,000+ points: worst 1.654e-12 at theta = 0.05227 (c2), matching the recorded 1.65e-12 |
| C41 recorded figures | consistent | the 2.970 is labelled seed-specific |

**Equivalence proofs:**

- **V02 (`2 qv / qw` -> `2 qv`): valid.** For orthogonal R with tr <= 0, qw^2 <= 1/4, so
  |qv| >= 0.866. The n < 1e-10 return is therefore reached only from the trace branch, where qw is
  1 to within O(delta + 1e-20). The two returns differ by a relative O(eps) on a vector of norm
  < 2e-10, far below every log-claim tolerance (>= 1e-12).
- **V18 (`solve(Gamma1, v)`): valid.** I confirmed cond(Gamma1) <= pi/2 on [0, pi]: the maximum
  is 1.5707963 at theta = pi, over 2001 points. The measured relative difference over 20,000 log
  inputs is **1.8e-14**. The proof's "~4e-16 |v|" understates this, because gamma1_inv carries
  its own c2 rounding, but the conclusion holds with more than 50x margin to 1e-12.
- **V39 (`dR (I + W dt)`): valid.** It is distributivity: the same map in exact arithmetic, with
  rounding differences of O(n eps).
- **N15 (batch switch at 1e-3): valid.** The more aggressive Q41 (switch at 0.03 AND no th^6)
  stays within **4.2e-15** of scipy on a dense 13,000-angle sweep. On every interval, both
  branches evaluate the same s and c to within the claim's 1e-14.

**C22's 2.5e-15 band bound.** I ran the band check (20,000 R with tr in (-0.99, -0.9]) on 100
seeds (0..99) for both the clean code and the V44 mutant:

| | max | median | min |
|---|---|---|---|
| clean | **1.33e-15** | 8.9e-16 | 8.9e-16 |
| V44 | 7.1e-15 | 5.3e-15 | **4.0e-15** |

The clean code never exceeds the bound (1.9x margin), and V44 fails it on **all 100** seeds (1.6x
margin). This is robust across seeds. It is a 6-ulp bound, though, and depends on the rounding of
scipy `Rotation`, so it should be re-measured on any scipy or BLAS change.

The bound pins only thresholds at or below -0.99:

- `tr > -0.5` (Q12) survives, and it is equivalent (6.7e-16 on the tr in (-0.5, 0] band);
- `tr > -1.0` (Q13) is killed (2e-4 near pi).

## New finding M1-F15: scalar and batched Rodrigues now disagree (regression from 59330b2)

59330b2 gave `_so3_exp_batch` the theta^6 term of sin(theta)/theta but left `gal3.so3_exp`
without it. `imu_step_element` builds its rotation with `gal3.so3_exp`, while `integrate_increment`
uses the batch. Before the fix both used the same polynomial, so "integrate_increment equals the
ordered product of imu_step_element" (the C45 hint, bound 1e-12) held to rounding. It no longer
holds once the per-step angle approaches the switch:

| per-step angle, n | clean, at 9cda195 | before the fix (no th^6 in the batch) |
|---|---|---|
| 0.049, n = 60 | **1.23e-12 - 1.46e-12** | 3.3e-16 |
| 0.049, n = 1000 | **5.7e-12 - 5.9e-12** | 6.7e-16 - 1.7e-13 |
| 0.03, n = 1000 | 2.6e-13 | 1.7e-13 |

The error is in `imu_step_element`: its product against a scipy expm/Rotation product is 5.6e-12.
The test checks this only at dt = 5e-5, where the per-step angle is below 1e-3. C45's domain is
"Total", and the `imu_step_element` docstring says "exactly".

**Fix:** add `- t4 * t2 / 5040.0` to `gal3.so3_exp`. As Q46 shows, that edit passes the whole
suite, and it closes Q05's hiding place. Then pin `gal3.so3_exp` to <= 1e-14 against scipy, as
the batch already is.

## Own mutation sweep: 54 new mutants (Q01..Q54)

Each mutant is one textual edit on a fresh scratch copy, followed by a full run of
`tests/test_gal3.py` and `tests/test_imu_dynamics.py`.

### SURVIVORS (14, non-equivalent)

**Fixed-gravity family (medium, 6).** Every test uses g = (0, 0, -9.81). The conventions say "all
claims hold for arbitrary g", and C31 says "Holds for arbitrary g". Ignoring the g argument, or
permuting its x/y components, is therefore invisible. The clean src does hold for arbitrary g:
with random g ~ 5 N(0, I3), the endpoint vs DOP853 is 5.4e-14, and C32-C36 are 0 to 5.7e-14.

| id | edit | claims |
|---|---|---|
| Q49 | `gravity_lift` uses `G_DEFAULT` instead of `g` | C31, C32, C38, C43 |
| Q50 | `gravity_element` uses g[[1,0,2]] | C32, C33 |
| Q51 | `forster_endpoint` uses g[[1,0,2]] | C43 |
| Q52 | `integrate_truth` uses g[[1,0,2]] | C31, C34, C37, C43 (identity checks) |
| Q53 | `gravity_generator` uses g[[1,0,2]] | C32 |
| Q54 | `gravity_lift_uncorrected` uses `G_DEFAULT` | C34 |

Kill: draw g ~ 5 N(0, I3) in at least one instance of each of these tests.

**Negative-duration family (medium, 6).** Every endpoint, gravity, navstate and increment-block
test samples Dt > 0. Only C41 (a') probes Dt < 0, and only through `preintegrate_constant`. The
claims do not restrict the sign of Dt:

- C32: "Total in Dt";
- C33: "for all real s1, s2";
- C41: "Dt is any real number";
- C31, C40, C42, C43 and C45 state no Dt sign restriction.

The clean src is correct for Dt < 0: on Dt in (-2, 3), C32 fusion is 0.0 and C35/C36 are <= 1.4e-14.

| id | edit | claims |
|---|---|---|
| Q28 | `endpoint`: Dt = abs(D[3,4]) | C31, C36, C38 |
| Q29 | `gravity_lift`: velocity g abs(Dt) | C31, C32 (domain "Total in Dt") |
| Q30 | `forster_endpoint`: 0.5 g Dt abs(Dt) | C43 |
| Q31 | `navstate_phi`: p + v abs(Dt) | C40 |
| Q35 | `increment_blocks`: Dt abs(Dt) Gamma2 a | C42 |
| Q36 | `imu_step_element`: 0.5 a dt abs(dt) | C45 |

Kill: add Dt < 0 draws, or state Dt >= 0 in each domain where that is intended. C41 already
states the opposite.

**Low (2):**

- **Q05**: `gal3.so3_exp` c-coefficient th^4/720 -> th^4/700. This is N19's scalar twin. Matrix
  error 6.17e-13 against a 40-digit reference (clean 1.33e-13). Nothing pins `gal3.so3_exp` below
  1e-12: C42's dR check compares `so3_exp` with `so3_exp` (through `exp`), and `worst_step` allows
  1e-12. The project ruled the batch version a killable defect (N19), so the scalar one must be
  too. Kill: the F15 pin.
- **Q25**: `structural_residual` reads `X[3, 0:2]` instead of `X[3, 0:3]`, so it is blind to
  P[3,2]. C02's defect list probes only X[3,1] (and X[4,3]). This is the same class as V06: C02's
  P[3,2] = 0 check becomes vacuous. Kill: probe each of the 7 structural zeros, not one per row.

### Proven equivalent (15)

| id | edit | proof |
|---|---|---|
| Q03 | `so3_exp`: `<` -> `<=` THETA_SMALL | Changes only theta = 0.05 exactly, where the Taylor value differs from the closed form by the same 1.5e-13 the clean code already returns at nextafter(0.05, 0). No claim tolerance (>= 1e-12) separates them |
| Q04 | `so3_exp` c via literal 1 - cos | Absolute error eps/theta^2 in c, multiplied by \|KK\| <= theta^2, so O(eps) in the matrix. Measured 1.40e-13 vs 1.33e-13 clean on [1e-4, 0.3] |
| Q07 | `so3_log` small branch for n < 1e-4 | 2 atan2(n, qw)/n vs 2/qw differs by (2/3) n^3/qw^3 <= 6.7e-13 < 1e-12, the tightest log tolerance (C52 localCoordinates). Measured 5.6e-13 |
| Q08 | `qw <= 0` | Differs only at qw = 0 (theta = pi), where +-phi are both valid logs. C22 excludes pi, and C23 holds for either |
| Q09 | first diagonal branch with `>=` | Changes only ties, where both branches use the same s = 2 sqrt(1 + 2 R_ii - tr) |
| Q12 | trace branch for tr > -0.5 | Exact algebra there. s >= sqrt(2), so rounding is not amplified: 6.7e-16 measured |
| Q17 | kappa2 = (1 - s/theta)/theta^2 | Relative rounding <= 12 eps/theta^2 <= 5.3e-13 for theta >= 0.05, under C24's 1e-11. Measured 2.75e-13 |
| Q18 | remove the theta < 0 ValueError | Every caller passes a norm. C24's statement covers theta in [0, 2 pi) |
| Q23 | unskew reads -A[1,2] | Identical on skew-symmetric input, and C08's domain is gal(3) |
| Q26 | cay_matrix as (I + A/2)(I - A/2)^-1 | Polynomials in A commute, so this is an algebraic identity. Rounding is of the same order (C25/C26 pass at 1e-12 relative) |
| Q38 | `D0` used without a copy | The loop only rebinds (`dp = dp + ...`); it never mutates in place |
| Q39 | per-step length check removed | No claim concerns malformed streams. Every claim input has length n |
| Q40 | batch `<` -> `<=` | Changes only theta = 0.05, where the th^6 Taylor branch is accurate to 5e-18 (that point is in the C42 sweep) |
| Q41 | batch switch at 0.03 and no th^6 | Truncation <= 0.03^7/5040 = 4.3e-15. Above the switch the c rounding is O(eps). Dense sweep 4.2e-15 < 1e-14 |
| Q46 | `so3_exp` gains th^6 | An improvement: changes values by <= 1.5e-13, under every so3_exp-facing tolerance. It removes F15 |

### Killed (25)

Q01 (`so3_exp` without th^4/120: C23, C41, C45, C52), Q02, Q06 (log small branch for n < 1e-2:
C23, C28, C29), Q10 (smaller diagonal branch: C22, C52), Q11 (arcsin: 3.9e-8 near pi; C22, C52),
Q13, Q14, Q15, Q16, Q19, Q20, Q21, Q22, Q24 (no (4,4) term: C02), Q27, Q32, Q33, Q34, Q37, Q42,
Q43, Q44, Q45, Q47, Q48.

## Harness replay

I replayed **all 45** `MUTATIONS` entries. Each was applied textually to a fresh scratch copy, and
both claim files were run in full.

**Result: 45/45 killed.** Every named test failed. Every `expect` needle also appears in the source
statement directly above the first `E ` line, by my own parser, written independently of the
harness's `_failing_statement`.

The replay covers:

- the 4 fixture/order mutants and the 7 positive controls;
- navstate_phi x2 and cay_denominator;
- the Euler/increment mutants;
- the Taylor/THETA_SMALL/c2 mutants;
- the F11/F12 mutants;
- the 18 entries new in 59330b2: P03, P08, P10, P11, P13, P15, P16, P19, P22, N06, N14, 9.9e-13,
  the NaN guard, no_th6, N18, N19, V44 and the foreign-alpha check.

## Per-claim verdicts (independent recomputation, seeds 66666 / 77777)

| claim | verdict | attack / independent measurement |
|---|---|---|
| C01 | CONFIRMED | compose vs 5x5 matmul, 3000 pairs, t2 ~ 3 N: 3.6e-15 |
| C02 | CONFIRMED | residual 2.2e-15; reflection flagged 2.0 (but see Q25) |
| C03 | CONFIRMED | 5.3e-15 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | vs np.linalg.inv, relative 9.8e-15; time entry 0 |
| C06 | CONFIRMED | worst commutator 21.3 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | vs hand-written hat: 0.0 |
| C09 | CONFIRMED | ad vs commutator columns: 0.0 |
| C10 | CONFIRMED | 1.8e-15 / 7.1e-15 |
| C11 | CONFIRMED | Ad and Ad_apply vs vee(X e_i^ X^-1), relative 1.7e-15 |
| C12 | CONFIRMED | relative 1.8e-15 |
| C13 | CONFIRMED | relative 1.7e-15 |
| C14 | CONFIRMED | relative 4.4e-15 / 2.6e-15 |
| C15 | CONFIRMED | theta in [0, 2.9], including the Taylor branch: 5.1e-14 |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.4e-15; trace 0.0 |
| C18 | CONFIRMED | vs the 41-term series, theta in 1e-9..3: 1.2e-14 |
| C19 | CONFIRMED | 5000 draws including theta < 1e-3, < 0.05 and up to 2 pi - 0.3, vs expm: relative 1.5e-13 |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | c2 near pi vs 60 digits: 1.05e-16; G1inv G1 - I, theta in 1e-9..pi: 1.6e-14; expm(log X) = X: 1.1e-14 |
| C22 | CONFIRMED | theta in (1e-8, pi - 1e-3): 5.8e-15; band robustness above |
| C23 | CONFIRMED | 5000 X including pi - 10^-k (k = 3..12), via scipy expm: 6.2e-15 |
| C24 | CONFIRMED | det/cond 1.0, 0.40528, 2.2128e-3, 0.0, 1.5708; coefficients vs 80 digits: 1.65e-12 |
| C25 | CONFIRMED | relative 6.7e-16, including \|u\|, \|w\| ~ 1e3 |
| C26 | CONFIRMED | 4.2e-16 |
| C27 | CONFIRMED | 6.7e-16 |
| C28 | CONFIRMED | 2 tan(theta/2) = 0.51068, 1.09260, 3.11482, 28.2028, 43171.6 (all round to the table); round trip through cay to 4 sf |
| C29 | CONFIRMED | 0.060 (<= 1) and 7.7e-3 (<= 1) |
| C30 | CONFIRMED | 15/15 expressions identical (exact sympy) to an independent derivation; syms identical; numeric 8.9e-16 |
| C31 | CONFIRMED | endpoint vs own DOP853 with RANDOM g and time-varying inputs: 5.4e-14 |
| C32 | CONFIRMED | random g, Dt in (-2, 3): fusion 0.0, exp 1.4e-14 |
| C33 | CONFIRMED | 5.7e-14 |
| C34 | CONFIRMED | time Dt exactly; position p_j + g Dt^2 to 2.5e-14 |
| C35 | CONFIRMED | 1.1e-14 |
| C36 | CONFIRMED | 1.4e-14 / 8.9e-16; guard as above |
| C37 | CONFIRMED | gtsam NavState chain vs DOP853: 5.4e-14 |
| C38 | CONFIRMED | 0.0 / 0.0 |
| C39 | CONFIRMED | 8.9e-16 |
| C40 | CONFIRMED | Dt = 0.7 (the test uses 0.5): automorphism, conjugation and lift 1.8e-15 |
| C41 | CONFIRMED | vs expm, Dt in {0.02 .. 30, -0.3, -4, -25}: relative 1.2e-13; vs own DOP853 of Ddot = D xi^: 2.4e-13 |
| C42 | CONFIRMED | blocks vs expm 1.2e-13; batch vs scipy on 8000 angles: 1.1e-15 |
| C43 | CONFIRMED | 5.4e-14 |
| C44 | CONFIRMED | 4.4e-16 |
| C45 | CONFIRMED (statement) | algebraic 8.9e-16. Hint sub-assertion falsified outside the tested dt (F15: 1.46e-12 > 1e-12) |
| C46-C49 | CONFIRMED | 0.0; retract/Expmap basis witnesses as stated |
| C50 | CONFIRMED | 20,000 draws: 4.7e-10 (<= 1e-9); unpermuted 1.8e-5 |
| C51 | CONFIRMED | Expmap 8.9e-16, Logmap 3.6e-15, Adjoint 2.2e-16, adjoint 0.0 |
| C52 | CONFIRMED | compose/between/retract, relative 1.1e-15 |
| C53 | CONFIRMED | PIM predict vs src endpoint: 1.8e-15 |
| C54, C55 | CONFIRMED | 0.0 |
| **C60** | **REFUTED** | 3.10e-12 relative at theta = 0.05 (needs < 1e-16) |

**Totals: 55 CONFIRMED, 1 REFUTED (C60), 0 UNTESTABLE.**

## Tests that cannot fail (or cannot fail independently)

1. **C30**: `assert syms == expected_syms` and `assert len(set(syms)) == 25` are implied by the
   name-list equality plus the per-element `type(x) is sp.Symbol and x == y` check.
2. **C26**: `assert d > 0.0` in the first loop. `|d - (1 + |phi|^2/4)| <= 1e-10` implies it, since
   1 + |phi|^2/4 >= 1, so it can only fire when the det bound would fail too.
3. **C55**: `worst_naive >= 1e-3` exercises gtsam's own constructor only. No change to src can
   affect it (an input property).
4. **C45**: `worst_wrong >= 1e-3`. Once `worst_alg <= 1e-14` passes, this is a property of the
   random data (dv_ij Dt_jk != dv_jk Dt_ij), not of src.
5. **C42**: `dp - Dt^2 gamma2(omega Dt) @ acc` re-evaluates the exact src expression, so it is
   blind to any gamma2 defect (Q35 passes it). The main C42 block also compares `so3_exp` with
   itself (through `exp`) for dR. That is why Q05 survives.
6. **C46-C49, C54** (gtsam-only assertions) cannot fail through src. That is appropriate for
   VERIFIED-PUBLISHED claims, but they test gtsam, not the project.

## Recommendations (to clear the gate)

1. Arbitrary g: give every gravity-facing test (C31-C38, C43, and the integrate_truth identity
   checks) at least one random g. Kills Q49-Q54.
2. Negative Dt: add Dt < 0 draws to C31, C32, C40, C42, C43 and C45 (`imu_step_element`), or
   state Dt >= 0 in those domains. Kills Q28-Q31, Q35 and Q36.
3. F15: add the th^6 term to `gal3.so3_exp` and pin it to <= 1e-14 against scipy. Kills Q05 and
   restores the C45 product identity at large per-step angles.
4. C02: probe every structural zero (X[3,0..2], X[4,0..3]). Kills Q25.
5. Drop or annotate M1-C60.
