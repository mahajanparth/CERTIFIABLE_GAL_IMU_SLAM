# M1 eleventh-pass adversarial verification

Branch `m1-gal3-mechanics`. The code verified is HEAD 0ebd5aa, the response to M1-F24 (B3b, B4, F19
under amended rule 4) and M1-F25..F29. Agent verify-M1-012.

**src/ and docs/claims/ are unchanged.** `git diff 28bcc9a 0ebd5aa -- src docs/claims` is empty
(also unchanged since 8e1ecae). The commit touches only tests/conftest.py, tests/test_gal3.py,
tests/test_imu_dynamics.py and tests/test_mutation.py.

References are independent: numpy, scipy (`expm`, `Rotation`, `solve_ivp` DOP853), mpmath (40 to
60 digits), sympy and gtsam 4.3.0. No reference imports `src` or `tests`. I re-implemented these
myself (`_p11/ref.py`):

- Gal(3) adjoint as `vee(X e_k^ X^-1)` with `np.linalg.inv`, and ad as the commutator;
- exp as `expm(xi^)`, log as scipy Rotation log + 60-term Gamma series + linear solves;
- a 40-digit Rodrigues, a 50-digit SO(3) log, and 60-digit Gamma coefficients;
- the navigation ODE and the 5x5 increment ODE (DOP853, rtol = atol = 1e-13, non-constant
  omega(t), a(t));
- the trace branch of so3_log, bit-identical to src (0.0 difference), to measure the B3/B3b family;
- a mutation runner (`_p11/mut.py`). It makes real textual edits on a temp copy of src/ + tests/ +
  conftest.py + pyproject.toml. Each edit must match exactly once, and edits are applied in sequence.
  Each mutant is judged by the full `tests/test_gal3.py tests/test_imu_dynamics.py` run, with nothing
  deselected. All 140 edits passed the exactly-once check.

**Run hygiene.** Midway through the sweep, the coordinator reported mutant S08_log_b1_sum as
running for about 25 h. The cause was OpenBLAS thread oversubscription on a loaded host, not the
mutant. After that I added two settings to every pytest and mutant run:

- a hard `timeout -k 10 600`;
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

Re-run alone with both, S08 completes: **KILLED** in 37 s (5 tests, C22 2.26). It completed in
144 s even before the thread cap. No mutant hit the timeout, and no result is KILLED-BY-TIMEOUT.

Equivalence rules, as stated in the brief:

- rule 1: effect <= 1/10 of the tightest claim tolerance;
- rule 2: within 8 ulp and no worse than clean;
- rule 3: a written proof;
- rule 4 (amended 2026-10-16): the sup-norm error over the affected domain exceeds clean's by at
  most 2 eps max|value|, with NO 100x clause.

The gate requires zero surviving non-equivalent mutants.

## Summary

| item | ruling |
|---|---|
| Claims M1-C01..C55 | **55 CONFIRMED** |
| M1-C65 (brief only) | **REFUTED.** Over 10^4 random X (\|t\| <= 5): max \|claimed - inv(X)[0:3,4]\| = **104.1** (min 1.6e-3). The true block -R^T (p - t v) matches to 5.0e-14. gtsam `Gal3.inverse().position()` is 104.1 from the claim and 7.1e-15 from the true form. |
| M1-F24 | **FIXED.** B3b, B4 and F19 are killed on their new pins. No gap: every surviving member of the B3/B4/F19 families is equivalent under amended rule 4. F19 is in fact rule-4 equivalent, so its kill is over-tight but harmless. Numbers below. |
| M1-F25 | **FIXED.** All 8 `g=-G_DEFAULT` defaults are killed at `bad_defaults` (7.2594, or 19.62 for gravity_generator). |
| M1-F26 / M1-F27 | **FIXED for the named survivors.** H_MAG2 is killed by mag_ratio (1.0) and H_ODD by octants (0); all 5 compounds are killed. With the magnitude guard switched off, the literal backstop alone kills GNORM in LIFT, TRUTH and FORST. **Residuals:** a magnitude-RANGE fixture (**M1-F30**) and a dataset-local odd-octant fixture (**M1-F31**) pass every guard and the literal backstops. |
| M1-F28 | **FIXED** (F20 killed: 2.43e-12 > 2.0e-12). The pin is deterministic on this host but libm-fragile (**M1-F33**, advisory). |
| M1-F29 | **FIXED** (F41 killed: kappa1 at 0.05 is 7.75e-13 > 1e-14; clean 1.69e-16) |
| Kill character | Unchanged for older kills: see "Harness audit and kill character" |
| Harness | Clean run: 129/129 harness mutants killed, plus the unmutated copy |
| Fresh sweep | **140 mutants**: 101 killed, 39 survived. Of the survivors, 27 are equivalent and **12 are NON-EQUIVALENT** |
| Clean suite | `M1_MUTATION_WORKERS=4 pytest -q tests/`: **204 passed** (521.6 s; run before the thread cap), exit 0 |
| **Gate** | **FAIL** |

## M1-C65: REFUTED

Claim: the inverse's position block is -R^T (p + t v), to 1e-12.

From X X^-1 = I, row 0:3 of column 4 gives R p' + v t' + p = 0 with t' = -t. So
p' = -R^T (p - t v): the coasting term is SUBTRACTED. This is C05, implemented in src and in
CLAUDE.md.

Measurement (seed 65065, 10^4 X: scipy rotations of 2 N(0, I), v ~ 3 N, p ~ 5 N, t ~ U(-5, 5)):

- claimed form against `np.linalg.inv(X)`: max 104.1, min 1.57e-3;
- -R^T (p - t v): 4.97e-14;
- against gtsam `Gal3(R, p, v, t).inverse().position()`: claimed 104.1, true 7.1e-15;
- src `inverse` is exact (0.0) on the counterexample.

Minimal counterexample: X = (I, e1, 0, 1). The inverse has p = +e1; the claim gives -e1.
M1-C65 must not enter the yaml.

## M1-F24 under amended rule 4

Pins at HEAD:

- C22: edge band theta in [5e-8, 1.2e-7], 5000 R, sup relative error <= 6.5e-16 and
  |mean signed relative| <= 1.5e-16;
- C24 Taylor: mean signed relative error <= 1e-16 on [0.045, 0.05), kappa1 sup <= 2.0e-16, kappa2
  sup <= 3.0e-16.

**Replays** (real edits, full suite):

| id | edit | result | failing assertion and value |
|---|---|---|---|
| B3 | so3_log `n < 3e-8` | survives | owner-ruled equivalent |
| B3b | so3_log `n < 5e-8` | **KILLED** | test_gal3.py:580 `edge_bias` 3.61e-16 (> 1.5e-16) |
| B4 | kappa1 `t4 / 720.00002` | **KILLED** | test_gal3.py:769 `poly_bias` 3.96e-16 (> 1e-16) |
| F19 | kappa2 `t4 / 5040.0001` | **KILLED** | test_gal3.py:769 `poly_bias` 1.72e-16 |
| B1, B2 | Rodrigues c + th^6/200000 | survive | owner-ruled equivalent |

**Gap hunt: is there an unkilled mutant between "rule-4 equivalent" and "killed"?** I measured the
whole family on each pin. The rule-4 excess is the sup absolute error over the affected domain
minus clean's, against the stated polynomial or a 50-digit log. Domain samples are 40k theta for
the polynomials and 10^4 R for so3_log.

| family | member | test statistics (grid sup / mean) | rule-4 excess vs 2 eps max\|v\| | eqv (rule 4) | suite |
|---|---|---|---|---|---|
| kappa1 t4/d | 720.000002 | 1.55e-16 / 4.0e-17 | 2.3e-17 vs 2.22e-16 | yes | survives |
| | 720.000005 | 2.17e-16 / 9.99e-17 | 5.5e-17 | yes | killed |
| | 720.00001 (S22) | 3.41e-16 / 1.97e-16 | 1.17e-16 | yes | killed |
| | 720.00002 (B4) | 5.86e-16 / 3.96e-16 | **2.35e-16** | **no** | killed |
| kappa2 t4/d | 5040.00002 | 2.42e-16 / 7.8e-17 | 4.2e-18 vs 7.40e-17 | yes | survives |
| | 5040.00005 (S23) | 2.78e-16 / 1.16e-16 | 1.1e-17 | yes | killed |
| | 5040.0001 (F19) | 3.51e-16 / 1.72e-16 | 2.25e-17 | **yes** | killed |
| | 5040.0002 | 4.94e-16 / 2.97e-16 | 4.8e-17 | yes | killed |
| so3_log n < T | 3e-8 (B3) | survives | 6.6e-24 vs 2.66e-23 | yes | survives |
| | 3.5e-8 | survives | 1.99e-23 vs 3.08e-23 | yes | survives |
| | 4e-8 | killed | 2.65e-23 vs 3.54e-23 | yes | killed |
| | 5e-8 (B3b) | killed | **6.62e-23 vs 4.41e-23** | **no** | killed |
| kappa3 t4/40320.0004 (S24) | | survives | 1.2e-18 vs 1.85e-17 | yes | survives |
| c2 t4/30240.0003 (S25) | | survives | 1.5e-18 vs 3.70e-17 | yes | survives |

So no rule-4 non-equivalent member of these families survives. The pins are slightly tighter than
rule 4 requires, which is safe for the gate:

- F19, S22 and S23 are killed although they are rule-4 equivalent;
- the kappa2 sup pin has only 1.38x clean headroom.

**Determinism and fragility.**

- C24 Taylor pins (B4/F19): the polynomial is pure Python float arithmetic (+, *, /) on a fixed
  linspace grid. It is deterministic under IEEE-754 on any platform. Robust.
- C22 edge band (B3b): fixed seed, but scipy `from_rotvec` and `arctan2` are libm-dependent. I
  re-measured on 6 independent seeds of 5000 R:
  - clean sup <= 4.55e-16 (pin 6.5e-16, worst 1.43x headroom) and mean <= 3.4e-17;
  - B3b sup >= 9.36e-16 and mean >= 3.67e-16 (killed on every seed);
  - B3 sup <= 4.82e-16 and mean <= 6.7e-17 (passes on every seed).

  An unbiased 1-ulp random arctan2 perturbation keeps clean at 4.42e-16 / 2.3e-17. A
  SYSTEMATICALLY +1-ulp-biased arctan2 would give mean 1.69e-16 > 1.5e-16 (a false failure).
  No real libm is known to me to be biased that way. Advisory only.
- C24 c2 dense pin (F28): see M1-F33. It passes here only because np.sin is correctly rounded at all
  4000 grid points on this host.

## M1-F25..F29 rulings (replays as real exactly-once edits)

- **F25:** D_gravity_lift, D_gravity_lift_uncorrected, D_gravity_element, D_endpoint_uncorrected,
  D_endpoint_right_time_shift, D_forster_endpoint and D_integrate_truth are all killed at
  test_imu_dynamics.py:902 `bad_defaults` (7.2594). D_gravity_generator is killed there too (19.62).
- **F26:**
  - H_MAG2 alone and GNORM_{LIFT,TRUTH,FORST} + H_MAG2 are killed by `mag_ratio` (1.0) in 9 tests.
  - GNORM_* alone: C31 3.145 (as in pass 10).
  - With the magnitude guard off (H_MAG2 + guard off), the literal backstop alone kills GNORM_LIFT
    (C31 `worst_ff_lift`), GNORM_TRUTH (C31 `worst_ff_truth`) and GNORM_FORST (C43 `worst_ff`).
  - **GNORM_UNC + H_MAG2 + guard off SURVIVES:** the uncorrected form is not on the backstop
    (M1-F31).
- **F27:**
  - H_ODD alone: killed at `octants` = 0 in 6 tests.
  - PFLIP_LIFT + H_ODD: C31 `worst_ff_lift` 7.848, plus `octants`.
  - PFLIP_TRUTH + H_ODD: C31 `worst_ff_truth` 7.848, plus `octants`.
- **F28:** F20 is killed at test_gal3.py:802, `c2_dense` 2.43e-12.
- **F29:** F41 is killed at test_gal3.py:801, `k1_at_switch` 7.75e-13.

**Hunt for another degenerate gravity distribution (brief item).** I need a fixture that passes
the component, span, signed, magnitude-ratio and octant guards AND the literal backstops, plus a src
defect that it masks. I found two.

1. **H_LOW** (fixture): `|g| ~ U(2, 11.5)` in place of U(2, 20). Directions and RNG stream are
   unchanged, the magnitude ratio is still about 5, and every guard passes.
   - The defect it masks is **CLIP**: `g = np.clip(g, -11.5, 11.5)`, a saturating gravity.
   - G_LITERAL_SET's largest components are 9.0 / 8.2 / 11.0 (|g| <= 14.6), so CLIP is invisible
     on it.
   - CLIP alone is killed at all 6 sites: LIFT C31 1.63, TRUTH C31 1.63, FORST C31 1.63,
     UNC C34 0.976, ELEM C32 5.74, GEN C32 8.01.
   - **CLIP_{LIFT, UNC, ELEM, GEN, TRUTH, FORST} + H_LOW all SURVIVE**, and so does H_LOW alone.
   - The guards bound the magnitude RATIO but not the RANGE. A fixture that drops the top of the
     claimed range (2, 20) hides every defect that acts only on large |g_k|.
2. **H_ODD_DS** (fixture, local to `endpoint_datasets_rand_g`): the odd-octant flip of pass 10,
   applied only to the two 4-g datasets. Their guard results:

   | dataset | flips | signed | span | mag_ratio |
   |---|---|---|---|---|
   | 3101 | 3 | 4.32 | 0.122 | 2.54 |
   | 3102 | 1 | 1.33 | 0.305 | 3.43 |

   The octant guard does not apply at m = 4, and nothing global changes.
   - The defect it masks is **PFLIP_UNC** (g -> -g if g_x g_y g_z < 0 in
     gravity_lift_uncorrected). PFLIP_UNC alone is killed (C34 7.47).
   - **PFLIP_UNC + H_ODD_DS SURVIVES.** C34 (uncorrected form) is tested only on the 4-g datasets,
     and the literal free-fall backstop covers only gravity_lift (via endpoint), integrate_truth and
     forster_endpoint.
   - The same fixture does NOT mask PFLIP in ELEM (C32 31.3), GEN (C32 36.3), FORST (C43 algebraic
     block 36.6), LIFT or TRUTH (killed by the literal backstop or by m >= 8 blocks).

## Harness audit and kill character

- The 28 new harness entries go through the unchanged `_edits` / `_mutate_and_run` path, which
  enforces exactly-once matching. Expected-assertion substrings name the new checks:
  `worst_ff_lift`, `worst_ff_truth`, `worst_ff`, `mag_ratio`, `octants`, `bad_defaults`,
  `edge_bias`, `poly_bias`, `k1_at_switch` and `c2_dense`.
- The clean run kills all 129 and passes the unmutated copy.
- **Older kills cannot silently change character.**
  - New blocks are APPENDED at the end of C22 (`test_log_exp_roundtrip`, after `worst_small_rel`),
    C31, C43 and C53. Every earlier assertion runs first, so a mutant killed before dies on the same
    line.
  - In C24 the new pins are inserted, but the only assertion after them is the negative-theta raise
    check (F49's), which no positive-theta mutant reaches.
  - My replays confirm the old values exactly: GNORM C31 3.1454 (test_gal3.py:1191 / 1198), PFLIP
    C31 12.4538, YABS (harness) unchanged, F36-type S34 at C31 Dt = -0.3. The harness's expect maps
    pass for all 129.

## Fresh mutation sweep (140 mutants)

The sweep has three parts:

- 126 in `_p11/spec.py`: replays, hunts, 50 gal3 edits (S01..S50), 17 imu edits (J01..J17), and
  gravity-site defects;
- 7 in `_p11/spec2.py`: guard-off backstop replays and lift-time variants;
- 7 so3_log thresholds (C22 only, used for the gap table).

Result: **101 killed, 39 survived.**

### NON-EQUIVALENT SURVIVORS (12)

| id | edit | measured effect | why non-equivalent |
|---|---|---|---|
| **H_LOW** (fixture) | `|g| ~ U(2, 11.5)` | passes every guard and backstop | masks CLIP (M1-F30) |
| **CLIP_LIFT + H_LOW** | gravity_lift `np.clip(g, -11.5, 11.5)` | alone: C31 1.63 | M1-F30 |
| **CLIP_TRUTH + H_LOW** | integrate_truth, same | alone: C31 1.63 | M1-F30 |
| **CLIP_FORST + H_LOW** | forster_endpoint, same | alone: C31 1.63 | M1-F30 |
| **CLIP_UNC + H_LOW** | gravity_lift_uncorrected, same | alone: C34 0.976 | M1-F30 |
| **CLIP_ELEM + H_LOW** | gravity_element, same | alone: C32 5.74 | M1-F30 |
| **CLIP_GEN + H_LOW** | gravity_generator, same | alone: C32 8.01 | M1-F30 |
| **H_ODD_DS** (fixture) | odd-octant flip on the two 4-g datasets only | passes every guard | masks PFLIP_UNC (M1-F31) |
| **PFLIP_UNC + H_ODD_DS** | gravity_lift_uncorrected g -> -g if g_x g_y g_z < 0 | alone: C34 7.47 | M1-F31 |
| **GNORM_UNC + H_MAG2 + mag guard off** | g -> 9.81 g/\|g\| in gravity_lift_uncorrected | the guard-off compound that pass-10's harness uses for LIFT/TRUTH/FORST survives for UNC | no behavioural backstop on the uncorrected form (M1-F31) |
| **S43b** | lift_navstate time `5e-13` (not 0.0) | the lifted element is off the t = 0 slice by 5e-13 | C38 says the lift has "time block 0" and W_g = T(-Dt) H' T(Dt) "EXACTLY"; C40 says "exactly". Tests bound it only at 1e-12, so rule 1 needs <= 1e-13 and 5e-13 exceeds it. M1-F32 |
| **S43d** | lift_navstate time `1e-12 * N[3,3] * 0.5` (= 5e-13) | as S43b | M1-F32 |

S43 itself (time 1e-13) also survives. It sits exactly at the rule-1 bound (1e-13 <= 1e-12/10), so
it is counted equivalent. S43c (9e-13) is killed by C38.

### EQUIVALENT SURVIVORS (27)

Effects are measured with `_p11/eqv*.py`, which loads the mutated src in-process and compares it
with clean and with the mpmath references on the affected domain.

| id | edit | measured effect | rule |
|---|---|---|---|
| B1, B2, B3 | pass-9 bands | owner-ruled equivalent under amended rule 4; re-measured B3 excess 6.6e-24 <= 2.66e-23 | rule 4 |
| LOGT 1e-8, 2e-8, 3e-8 (= B3), 3.5e-8 | so3_log `n < T` (C22-only runs) | excess <= 1.99e-23 <= 3.08e-23 (monotone in T) | rule 4 |
| (analytic only, not run as mutants) kappa1 720.000002, kappa2 5040.00002 | stated-polynomial denominators | excess 2.3e-17 / 4.2e-18 (see the gap table) | rule 4 |
| S24 | kappa3 `t4/40320.0004` | excess 1.2e-18 <= 1.85e-17 | rule 4 |
| S25 | c2 `t4/30240.0003` | excess 1.5e-18 <= 3.70e-17 | rule 4 |
| GD_negzero | `G_DEFAULT = [-0.0, -0.0, -9.81]` | -0.0 == 0.0 in every comparison and product; bad_defaults diff 0.0 | rules 1 and 3 |
| S02 | unskew reads `-A[0, 1]` for `A[1, 0]` | identical on skew input, which is the only input (as pass-10 F47) | rule 3 |
| S04 | so3_log trace branch `tr > 0.1` | only tr in (0, 0.1] changes, by 1.33e-15. Outside C22's 2.5e-15 band (tr <= -0.9), so the binding tolerance is C52's 1e-13. Error vs phi equal to clean (8.9e-16) | rule 1 (1.3e-15 <= 1e-14) and rule 2 |
| S06 | `arctan(n / qw)` for `arctan2(n, qw)` | (qw >= 0 after the hemisphere flip.) Abs sup equal to clean on every band (8.9e-16 / 6.7e-16; 0.0 on the C22 band). Small-angle relative 4.17e-16 vs clean 3.81e-16 (excess 3.6e-17 <= 4.4e-16). Passes the F24 edge pins | rules 2 and 4 |
| S07 | small-angle return `2 qv` (drops `/ qw`) | qw = 1 - O(1e-21) for n < 1e-10: 0.0 change. Pass-3 V02 proof in C22 | rule 3 |
| S10 | `tr >= 0.0` | only the tie tr = 0: 0.0 on 4000 R near tr = 0 | rule 2 |
| S11 | `R[1,1] >= R[2,2]` | only exact ties: 600 tie rotations, sup 6.7e-16 = clean | rule 2 |
| S36 | kappa2 `(1 - s/theta)/t2` | relative change 1.33e-13 on [0.05, 6]; sup 2.75e-13 vs clean 1.64e-13. The binding tolerance in the closed-form region is C24's 1e-11 | rule 1 (pass-6 Q17, same form) |
| S38 | so3_exp `sin(theta) * (1/theta)` | change 2.2e-16; sup 8.9e-16 = clean | rule 2 |
| S39 | compose `+ 0.0 * t1` | bit-identical for finite t1 | rule 3 |
| S43 | lift time 1e-13 | 1e-13 = 1/10 of the 1e-12 bound | rule 1 (at the bound) |
| S48 | so3_exp closed c `(1 - cos)/th^2` | [0.05, 0.2]: sup 1.1e-16 = clean; [0.2, 3.1]: 6.7e-16 vs clean 8.9e-16 | rule 2 (pass-10 F03 class) |
| J01 / J02 | batch Taylor switch `th < 1e-3` / `1e-2` | the closed form is MORE accurate below 0.05: sup on [1e-3, 0.05) 1.1e-16 vs clean 9.99e-16 | rule 2 |
| J05 | integrate_increment D0 without `copy=True` | no in-place update anywhere (`dR = dR @ E`, `dp = dp + ...` rebind) | rule 3 |
| J13 | batch c as `2 sin^2(th/2)/t2` | change <= 4.4e-16; [0.04, 0.052] sup 9.99e-16 = clean | rule 1 (C42 1e-14) and rule 2 |
| J14 | step `so3_exp(-omega dt).T` | bit-identical: skew(-phi) = -skew(phi) exactly, s and c identical, K K computed symmetric bitwise | rule 3 |
| J15 | truth `v += f dt; p += v dt - f dt^2/2` | algebraically identical; change 9.9e-14 at scale 92 (1.1e-15 relative), 4 streams incl. Dt < 0 | rule 1 (C31 identity 1e-12 -> 1e-13) |

### Killed (101)

Each line gives the mutant, then the failing test and value. Replays and pass-10 classes are listed
above.

- Gravity sites:
  - CLIP_* alone: C31 1.63, C34 0.976, C32 5.74 / 8.01
  - PFLIP_{UNC, ELEM, GEN, FORST} alone and with H_ODD_DS: C34 7.47, C32 31.3 / 36.3, C43 36.6
  - GNORM_{UNC, ELEM, GEN} alone
  - EP_noG: C31 10.15
  - EPU_noG: C34 6.09
  - EPR_noG: C35 9.61
- gal3:
  - S01 skew sign: C09 20.7
  - S03 so3_exp s t4/121: C42 band 1.8e-11
  - S05 Shepperd `or`: C22 band 2.16e-14
  - S08 / S09 log branch signs: C22 2.26 / 6.63
  - S12 R2 R1: C01 1.98
  - S13 inverse +t: C05 51.7
  - S14, S15, S16 residual: C02 `{'det': 0.0}`, `{'diag3': 0.0}`, `{'row3[2]': 0.0}`
  - S17 ad alpha sign: C09 15.3
  - S18 Ad t sign: C11 21.7
  - S19 Ad_apply alpha p: C11 29.0
  - S20 kappa3 /40321: C24 9.2e-14
  - S21 c2 /30241: C24 8.2e-14
  - S22 kappa1 720.00001: poly_bias 1.97e-16
  - S23 kappa2 5040.00005: poly_bias 1.16e-16
  - S26 exp G1 u: C15 1.10
  - S27 log w: C21 9.05
  - S28 cay rhs: C25 0.677
  - S29 uncorrected sign: C34 2.45
  - S30 generator +1: C32 9.71
  - S31 keyframe tol 1e-11: C36
  - S32 endpoint \|Dt\|: C31 10.1
  - S33 check_time default False: C36 DID NOT RAISE
  - S34 Forster v\|Dt\|: C31 3.43
  - S35 kappa3 with cos: C24 3.65e-11
  - S37 kappa1 1 - cos: C24 k1_at_switch 2.9e-14
  - S40 / S41 wedge/vee alpha: C08 6.83
  - S42 perm without rot: C50
  - S44 keyframe `<`: C36 `[1e-12, -1e-12]`
  - S45 gamma1_inv -c2: C21 1.21
  - S46 gamma2 k2 for k3: C15 1.53
  - S47 cay alpha/2: C25 2.17
  - S49 inverse p + t v (the C65 form): C05 51.7
  - S50 right shift on the left: C35 12.9
- imu_dynamics:
  - J03 `and` in stream check: C45 rejection
  - J04 drop t0: C45 0.04
  - J06 Euler with half term: C41 1.28
  - J07 Dt\|Dt\|: C42 1.41
  - J08 truth half dt\|dt\|: C31 6.3e-4 (Dt = -0.3)
  - J09 b_g sign: C44 7.06
  - J10 swap = I: C55 5.04
  - J11 \|Dt\| returned: input guard (Dt sign)
  - J12 batch s t4/121: C42 2.1e-11
  - J16 preintegrate \|Dt\|: C41 2.43
  - J17 gyro x (1 + 1e-7): C55 3.3e-7
- Thresholds: LOGT 4e-8, 4.5e-8, 5e-8; S43c lift time 9e-13.

## Property tests and the hunt for spurious behaviour (own re-implementation)

- 10^4 pairs/triples (seed 11011, \|t\| ~ 3 N):
  - compose vs matmul 3.6e-15; associativity 7.1e-15;
  - inverse (vs `np.linalg.inv` and both products) 2.2e-14; time entries 0.0.
- 10^4 X: src Ad vs own `vee(X e_k^ X^-1)` 1.1e-15 relative (Ad and Ad_apply).
  - Homomorphism 1.7e-15 relative; Ad_{X^-1} vs inv 2.3e-14.
  - Ad-ad 7.5e-14 / 1.4e-14 (\|xi\| ~ 0.5 N(0, I10)).
  - det(Ad) 4.0e-15; last row exactly e10.
- 10^4 xi with \|phi\| log-uniform in (1e-6, 2 pi - 0.3): exp vs expm 5.1e-14 relative; time block 0.0.
- 10^4 xi with \|phi\| in (0.05, pi - 1e-3): log(exp xi) 2.2e-15. 10^4 X with R = Exp(5 N):
  exp(log X) 3.5e-14.
- 10^4 Cayley: closed form vs 5x5 definition 7.4e-16 relative; Re orthogonality/det/time 9.1e-16.
  - \|c\| = 2 tan(theta/2) and axis: 1.3e-15.
- **Spurious feasible points:** S = I - [phi]x/2 has det 1 + \|phi\|^2/4 >= 1 (C26, measured
  6.2e-16 relative). So the solution of the 15 degree-2 residual equations is unique. On 2000 xi,
  solving the residual system by hand reproduces src cay to 8.9e-16.
- 20000 small R (theta log-uniform in [1e-10, 1e-2]): absolute 1.7e-18. 4000 of them against the
  50-digit log: relative 4.4e-16. Edge band tr in (-0.99, -0.9]: 8.9e-16.
- Keyframe guard: 0, -0.0, +-1e-12 and 5e-13 accepted; nextafter(1e-12, inf), NaN, +-inf and 1e-6
  refused.

## Per-claim verdicts

Scripts: `_p11/claims_a.py` (C01..C30), `_p11/claims_b.py` (C31..C55) and `_p11/c24.py`. They use
my own DOP853 streams with non-constant (omega, a), Dt in {+0.45, -0.35, +0.8} and 36 instances,
with g = (0, 0, -9.81) plus 11 random g:

- components min -3.9 / -4.5 / -10.0, max 10.6 / 16.0 / 10.7;
- \|g\| from 2.70 to 16.3.

| claim | verdict | measurement (bound) |
|---|---|---|
| C01 | CONFIRMED | 10^4 pairs: 3.6e-15 (1e-12) |
| C02 | CONFIRMED | 2.0e-15; residual flags reflection, each of 7 zeros, (3,3), (4,4) and scaling at >= 0.3 |
| C03 | CONFIRMED | 7.1e-15 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 2.2e-14 (\|t\| up to ~10); time entry 0.0 |
| C06 | CONFIRMED | max commutator 24.1 |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0, and wedge = own hat exactly |
| C09 | CONFIRMED | commutator identity 1.8e-15; vs own ad 0.0; last row 0, (6:9, 9) = u exactly |
| C10 | CONFIRMED | 3.6e-15 / 1.3e-14 |
| C11 | CONFIRMED | 1.1e-15 relative |
| C12 | CONFIRMED | 1.7e-15 relative |
| C13 | CONFIRMED | 2.3e-14 |
| C14 | CONFIRMED | 7.5e-14 / 1.4e-14 (1e-7) |
| C15 | CONFIRMED | 2000 xi: 1.7e-14 (1e-8) |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.0e-15; trace 0.0 |
| C18 | CONFIRMED | 10^4 vs 60-term series, \|phi\| in (1e-3, 3): 1.5e-14 |
| C19 | CONFIRMED | 10^4 vs expm: 5.1e-14 relative |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | 20000 phi: 1.6e-14 (1e-13); c2 at pi and pi - 10^-k: 1.05e-16 (1e-13); log vs own scipy log + series 1.1e-14 |
| C22 | CONFIRMED | 2.2e-15; \|phi\| = 3.5 wraps to 2.783185 = 2 pi - 3.5; band 8.9e-16 (2.5e-15); small angle 1.7e-18 abs, 4.4e-16 relative (1e-15) |
| C23 | CONFIRMED | 3.5e-14 (1e-9) |
| C24 | CONFIRMED | det 1.0, 0.40528, 2.2128e-3, 2.3e-65 at 2 pi; cond(pi) 1.570796; coefficients at the 8 points 8.24e-13 (1e-11); literal polynomial (4001 theta in (1e-7, 0.05)) 1.10e-16 / 2.18e-16 / 2.20e-16 / 2.17e-16 (1e-15); at nextafter(0.05, 0) 7.75e-13 / 2.58e-13 / 1.03e-13 / 1.55e-13 (pins 1e-12 / 5e-13 / 2e-13 / 3e-13); THETA_SMALL 0.05 |
| C25 | CONFIRMED | 7.4e-16 relative |
| C26 | CONFIRMED | det formula 6.2e-16 relative; matrix = I5 - xi^/2 exactly |
| C27 | CONFIRMED | 9.1e-16; time exact |
| C28 | CONFIRMED | 10^4: 1.3e-15; table 0.51068, 1.09260, 3.11482, 28.20284, 43171.56 |
| C29 | CONFIRMED | \|d\|/s^3 = 0.9185, 0.9198, 0.9200, 0.9200 for s = 0.05..0.005; (d - s^3 term)/s^5 = 0.6214..0.6226 (constant) |
| C30 | CONFIRMED | 15 expressions, 25 symbols, total degree 2; numeric residual 8.9e-16; symbolic evaluation 2.2e-16 |
| C31 | CONFIRMED | own DOP853, random g, Dt of both signs: 2.6e-14 (also own T H' X D product); time 0.0 |
| C32 | CONFIRMED | 10^4 (incl. expm(Dt zeta_g) and generator literal): 1.4e-14 |
| C33 | CONFIRMED | 1.1e-14 relative |
| C34 | CONFIRMED | time - Dt, R, v and position - (p_j + g Dt^2): 2.6e-14 |
| C35 | CONFIRMED | position + v_j Dt, time 0: 2.6e-14 |
| C36 | CONFIRMED | p_j + g Dt t_i and time t_i (t_i ~ N): 2.6e-14; guard as above |
| C37 | CONFIRMED | gtsam W . Phi(X_i) . U vs own ODE: 2.6e-14 |
| C38 | CONFIRMED | 10^4: W_g = T(-Dt) H' T(Dt), T(Dt) U_g = D, forms agree: 2.5e-14 |
| C39 | CONFIRMED | vs gtsam compose: 8.9e-16 |
| C40 | CONFIRMED | automorphism 3.6e-15; lift = conjugation 0.0; Phi(I) = I exactly (Dt in (-3, 3)) |
| C41 | CONFIRMED | vs expm, Dt in [-5, 20]: 4.8e-14 relative; own DOP853 4.9e-14; Euler err x n/Dt = 2.212 / 2.210 at n = 1e3 / 4e3 (first order) |
| C42 | CONFIRMED | 6.9e-14 relative (1e-12); Rodrigues scalar/batch vs 40-digit on [0.04, 0.052] (1500 angles): 9.99e-16 / 9.99e-16 (1.3e-15) |
| C43 | CONFIRMED | 2.6e-14 |
| C44 | CONFIRMED | 10^4: 8.9e-16, incl. gtsam correctGyroscope / correctAccelerometer |
| C45 | CONFIRMED | 10^4 chain = product 8.9e-16; split stream chain vs whole 6.7e-16; ordered product of own scipy steps 1.6e-15 |
| C46 / C47 / C48 | CONFIRMED | 10^4: matrix layout, accessors, constructor order, Hat = wedge, Vee: 0.0 |
| C49 | CONFIRMED | NavState retract(1e-3 e3) moves position (1e-3, 0, 0), velocity 0; Gal3.Expmap(1e-3 e3) moves velocity; NavState.matrix() column 3 = p |
| C50 | CONFIRMED | 20000 at s = 1e-5: 6.7e-10 (1e-9); unpermuted relative 2.0 (O(1)); s = 1e-4: 4.9e-8 (bound does not hold there, as the domain states); P as stated, exact |
| C51 | CONFIRMED | 10^4: Expmap / Logmap / AdjointMap / adjointMap: 3.3e-14 |
| C52 | CONFIRMED | compose / * / inverse / between / retract / localCoordinates (on the stated domain): 1.4e-14 |
| C53 | CONFIRMED | Dt = 0.5; predict vs endpoint 8.9e-16 with g omitted and with g passed; W Phi U position 0.0; G_DEFAULT = n_gravity = (0, 0, -9.81) |
| C54 | CONFIRMED | vector [1..6]; accelerometer, gyroscope halves |
| C55 | CONFIRMED | 10^4: 0.0; S S = I exactly |
| **C65** | **REFUTED** | max error 104.1 (claimed <= 1e-12); the true term is -R^T (p - t v) |

**Totals: 55 CONFIRMED, 1 REFUTED (C65), 0 UNTESTABLE.**

## New findings

- **M1-F30 (low-medium, fixture): the magnitude RANGE of random g is unguarded, and the literal set
  stops at |g_k| = 11.**
  - H_LOW (|g| ~ U(2, 11.5)) passes off_z, perm_gap, spread, comp, span, signed, mag_ratio and
    octants, as well as both literal backstops.
  - Under it, a saturating gravity `np.clip(g, -11.5, 11.5)` survives at all 6 g-consuming sites.
    Each site is killed alone (C31 1.63, C34 0.976, C32 5.74 / 8.01).
  - The claims say "arbitrary g" with tests on |g| in (2, 20). G_LITERAL_SET reaches only
    9.0 / 8.2 / 11.0 per component.
  - Kill: on each m >= 3 block, require max |g| >= 15 (recorded 4-g datasets: 19.8 and 18.5; the
    400..2000-draw blocks reach ~20). Also append retyped literals with one large component per axis
    and sign, e.g. (19.0, 0.5, -0.8), (-0.6, -19.0, 1.0), (0.7, -0.9, 19.5) and their negatives.
  - Note: no finite set excludes a threshold defect beyond its largest magnitude. The remedy is to
    cover the stated range (2, 20) at its top.
- **M1-F31 (low): the uncorrected forms have no behavioural backstop and no octant coverage.**
  - gravity_lift_uncorrected and endpoint_uncorrected (C34) are exercised only on the two 4-g
    datasets, which are exempt from the octant guard (m = 4), and on G.
  - A dataset-local odd-octant flip (H_ODD_DS, guards: signed 4.32 / 1.33, span 0.122 / 0.305,
    mag_ratio 2.54 / 3.43) masks PFLIP_UNC (alone killed, C34 7.47).
  - With the magnitude guard off, GNORM_UNC + H_MAG2 also survives, where the LIFT / TRUTH / FORST
    equivalents are caught by the literal backstop.
  - Kill: extend the literal free-fall backstop to C34. For each g in G_LITERAL_SET and Dt = +0.4 /
    -0.3, the uncorrected endpoint of a zero increment must equal (R_i, v_i + g Dt,
    p_i + v_i Dt + 3 g Dt^2 / 2, Dt) exactly.
  - Ideally, add a literal check for gravity_element / gravity_generator too (G(Dt) vs
    expm(Dt zeta_g) on G_LITERAL_SET). Or apply the octant guard to the union of the 8 dataset g.
- **M1-F32 (low): lift_navstate's time block is pinned only to 1e-12.**
  - Returning time 5e-13 (S43b, S43d) survives.
  - C38 states the lift has time block 0 and W_g = T(-Dt) H' T(Dt) EXACTLY; C40 says "exactly".
  - A lifted keyframe with t = 5e-13 also passes `_check_keyframe_time` (tol 1e-12), and the
    endpoint then carries a g Dt t_i position error.
  - Kill: assert `gal3.lift_navstate(N)[3, 4] == 0.0` exactly (and `[3, 3] == 1.0`) on the C38/C40
    draws.
- **M1-F33 (advisory, platform robustness): the F28 c2 pin depends on correctly rounded sin.**
  - On this host (numpy 2.4.6, X86_V3, no AVX512), np.sin is correctly rounded at all 4000 grid
    points, and clean c2 reaches 1.68e-12 against the 2.0e-12 pin.
  - Under any faithful (<= 1 ulp) sin, the same clean code gives 3.05e-12 to 3.72e-12. That is 5
    random trials plus the adversarial 1-ulp case, and it would FAIL the pin.
  - A libm-independent form distinguishes F20 deterministically: evaluate the 60-digit reference of
    the stated formula ON the double values sin(theta), sin(theta/2) that src uses. Then clean is
    9.92e-13 and F20 1.68e-12, unchanged under +-1-ulp sin inputs. A bound of 1.3e-12 keeps 1.3x
    headroom.
  - Not a gate item.
- Advisory: the B3b mean-signed pin (1.5e-16) has about 1 ulp of arctan2 headroom against a
  systematically biased libm (1.69e-16 with a +1-ulp bias). It is robust to reseeding (6 seeds:
  clean <= 3.4e-17, B3b >= 3.67e-16) and to unbiased 1-ulp noise.

Advisory carried over (not a gate item): A4. C52's domain text still says "in every implementation".

## Clean suite

`M1_MUTATION_WORKERS=4 python -m pytest -q tests/` at 0ebd5aa: **204 passed in 521.56 s**, exit 0.
That is 46 + 9 claim tests, 130 test_mutation (129 mutants plus the unmutated copy) and 19
test_orchestrator. The run was before the thread cap; with `OPENBLAS_NUM_THREADS=1` each mutant run
of the 55 claim tests takes 35 to 60 s. Each mutant run on the clean tree: 55 passed. The scratch
`_p11/` was deleted before commit.

## Gate

**FAIL.** 12 non-equivalent survivors:

- H_LOW plus 6 CLIP compounds (M1-F30);
- H_ODD_DS, PFLIP_UNC + H_ODD_DS and GNORM_UNC + H_MAG2 + magnitude guard off (M1-F31);
- S43b and S43d (M1-F32).

M1-F24, F25, F26, F27, F28 and F29 are fixed for every named survivor. Every claim C01..C55 is
confirmed. M1-C65 is false and must not enter the yaml.
