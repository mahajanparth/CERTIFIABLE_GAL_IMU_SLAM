# M1 third-pass adversarial verification (Gal(3) mechanics and IMU dynamics)

Branch `m1-gal3-mechanics`, verified at `8bab3df` (on top of the remediation commit `fa616ce`).
gtsam 4.3.0, numpy, scipy, sympy, mpmath. The verifier did not read `docs/01_conventions_and_gal3.md`,
any derivation narrative, or any seeds file. Nothing in `src/`, `tests/` or `docs/claims/` was edited.

## Summary

| item | result |
|---|---|
| Claims M1-C01..C55 (yaml) | **55 CONFIRMED**, 0 REFUTED, 0 UNTESTABLE |
| M1-C57 (brief only, not in yaml) | **REFUTED**: the alpha*v term is missing; max error 8.28 over 1e4 draws, bound 1e-12 |
| F07 surviving mutants | **FIXED** |
| F08 C41 rule at its own boundary | **FIXED** (two cosmetic nits) |
| F09 tests that cannot fail | **PARTIALLY FIXED**: every named item is resolved; one new tautology (C41) and one content-blind check (C30) remain |
| F10 stale recorded figures | **FIXED**: every figure re-measured at its stated seed reproduces |
| Own mutation sweep (50 mutants, none in `tests/test_mutation.py`) | 40 killed, 10 survived: **4 medium**, 3 low, 3 equivalent |
| Full suite, clean tree | 95 passed (62 s), before and after this pass |

## Method

`ref.py` is an independent reference. It uses numpy, scipy and mpmath only and imports nothing from
`src/` or `tests/`:

- group operations as 5x5 `@` and `np.linalg.inv`;
- exp as `scipy.linalg.expm`, and Cayley as the 5x5 solve;
- Ad as columns of vee(X e_i^ X^-1), and ad as columns of the commutator;
- log via scipy's rotvec plus a numerical solve of the (u, w) block, which is linear for fixed
  (phi, alpha);
- the increment ODE Ddot = D xi(t)^ and the navigation ODE written out from scratch and solved with
  DOP853 at rtol = atol = 1e-13. They use a different input family from `tests/ode_truth.py`
  (polynomial-times-trig omega(t) and a(t)), random gravity |g| ~ 5 N(0, I), and Dt in (0.1, 1.5).

Rotations come from scipy's `Rotation.from_rotvec`. `src` appears only as the code under test.

The property batteries use 1e4 draws wherever they are cheap. They include samples near the chart
edges:

- theta < 0.05 down to 1e-9;
- theta in (2.9, 2 pi - 0.3) for exp;
- R at pi - {0, 1e-14, 1e-10, 1e-7, 1e-4} for exp(log X);
- Cayley with |phi| ~ 50 and |u|, |w| ~ 1e3.

## Per-claim verdicts

Each figure is the maximum absolute error over the draws stated, against the independent reference;
the claim's bound is in brackets.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | 1e4 pairs, \|t\| up to ~10: compose vs 5x5 @ 3.6e-15 [1e-12] |
| C02 | CONFIRMED | 1e4: structural residual 2.3e-15 [1e-12] |
| C03 | CONFIRMED | 1e4 triples through `compose`: 7.5e-15 [1e-10] |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 1e4 with t ~ 3 N: 1.4e-14 vs I5 and vs `np.linalg.inv`; (X X^-1)[3,4] = 0.0 |
| C06 | CONFIRMED | worst commutator 35.8 (1e4 pairs); 16.1465 at the test seed |
| C07 | CONFIRMED | 0.0 exactly |
| C08 | CONFIRMED | 0.0 |
| C09 | CONFIRMED | `ad` vs commutator-built 10x10: 0.0 |
| C10 | CONFIRMED | homomorphism 1.8e-15, Jacobi 8.9e-15 |
| C11 | CONFIRMED | `Ad_apply` vs vee(X xi^ X^-1): 2.1e-14; `Ad` vs column-built Ad: 7.1e-15 |
| C12 | CONFIRMED | 1.1e-14 |
| C13 | CONFIRMED | 1.8e-14 |
| C14 | CONFIRMED | 3.1e-14 |
| C15 | CONFIRMED | 2000 draws, theta < 2.9 (incl. < 0.05): 7.9e-13 [1e-8] |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det 4.9e-15, trace 0.0 |
| C18 | CONFIRMED | 3000 phi, \|phi\| in (1e-3, 3): vs a 60-term series, 1.1e-14 [1e-12] |
| C19 | CONFIRMED | 3000 draws, theta in (1e-9, 2 pi - 0.3), alpha ~ 2 N: rel. 1.2e-13 vs expm [1e-9]; 3.77e-15 at the test seed |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | Gamma1^-1 Gamma1 - I: 1.8e-14 over 2e4 (1.82e-14 at the test seed); log vs independent log 1.2e-14; c2 rel. error at pi and pi - 10^-k (k = 3..9) <= 1.05e-16 vs mpmath |
| C22 | CONFIRMED | 1.4e-13; \|phi\| = 3.5 returns 2.78319 = 2 pi - 3.5 |
| C23 | CONFIRMED | 1e4: 1.4e-13; at R = pi - {0, 1e-14, ..., 1e-4}, 2000 draws: 2.2e-15; 1.485e-13 at the test seed |
| C24 | CONFIRMED | det Gamma1 = 1.0, 0.405285, 2.2128e-3, 2.3e-65; cond(pi) = 1.570796; coefficients vs an 80-digit SERIES reference: 1.106e-12 on the test's 863 points, 1.35e-12 on my 3301 points (both sides of 5e-2, out to 6.25) [1e-11] |
| C25 | CONFIRMED | 1e4, alpha ~ 3 N: 1.8e-15 |
| C26 | CONFIRMED | det 1.4e-14; matrix I5 - xi^/2 exact (0.0); 1e4 extreme draws (\|phi\| ~ 50, \|u\|,\|w\| ~ 1e3): rel. 6.7e-15 |
| C27 | CONFIRMED | 8.9e-16; time 0.0 |
| C28 | CONFIRMED | 1e4: 7.1e-15; tabulated values reproduce: 0.51068, 1.09260, 3.11482, 28.2028, 43171.6 |
| C29 | CONFIRMED | 1000 unit xi: s^3-scaled 0.059 [1.0]; s^5 remainder 8.2e-3 [1.0] (logm reference) |
| C30 | CONFIRMED | numeric residuals 1.8e-15 over 1e4; the 15 sympy expressions, lambdified and evaluated at Cay(xi), give 1.8e-15; max total degree 2. (The TEST does not check the symbolic content; see "Tests that cannot fail" and mutants V22/V23.) |
| C31 | CONFIRMED | own ODE, 60 instances, random g: 3.8e-14, time block 0.0 |
| C32 | CONFIRMED | 2000 Dt in (-2, 2): 7.1e-15; opposite order = (I, gDt, -1.5 g Dt^2, -Dt) to 7.1e-15 |
| C33 | CONFIRMED | 5.7e-14 |
| C34 | CONFIRMED | time block Dt (0.0 error); position offset g Dt^2 to 2.6e-14; R, v correct to 3.8e-14 |
| C35 | CONFIRMED | 5.5e-14 / 0.0 |
| C36 | CONFIRMED | t_i ~ 2 N: 3.8e-14 / 4.4e-16; the guard raises at t_i = 1e-9 (but see mutant V26) |
| C37 | CONFIRMED | gtsam NavState form vs own ODE: 3.8e-14 |
| C38 | CONFIRMED | 1.8e-15 |
| C39 | CONFIRMED | 1e4 gtsam pairs: 8.9e-16 |
| C40 | CONFIRMED | 1e4, Dt in (-2, 2): `navstate_phi` = definition (0.0); automorphism 2.2e-15; conjugation = lift(phi) 0.0; phi(I) = I |
| C41 | CONFIRMED | exp(Dt xi) vs own ODE at Dt up to 3.0: 1.8e-13; Euler coefficient: see F08 |
| C42 | CONFIRMED | 5000 draws, Dt in (1e-4, 2), \|omega\| ~ 2 N: 1.1e-13 vs expm [1e-12] |
| C43 | CONFIRMED | 3.8e-14 vs own ODE |
| C44 | CONFIRMED | 1e4: 4.4e-16 |
| C45 | CONFIRMED | three separate own-ODE solves, 30 streams: 1.6e-14; 1e4 algebraic: 8.9e-16 |
| C46 | CONFIRMED | 1e4: 0.0 |
| C47 | CONFIRMED | 0.0 |
| C48 | CONFIRMED | Hat/Vee 0.0; basis checks hold |
| C49 | CONFIRMED | retract(1e-4 e3) moves position; matrix col 3 = p |
| C50 | CONFIRMED | 2e4 at s = 1e-5: 5.6e-10 [1e-9]; unpermuted 6.5e-5; s = 1e-4: 5.2e-8; P exact |
| C51 | CONFIRMED | 1e4: Expmap 1.4e-13, Logmap 1.2e-13, AdjointMap 1.4e-14, adjointMap 0.0 |
| C52 | CONFIRMED | compose 3.6e-15, inverse 2.1e-14, between 2.1e-14, localCoordinates vs independent log 6.8e-14, retract vs X expm(d) 3.6e-15 |
| C53 | CONFIRMED | 100 states: form (a) 1.8e-15, form (b) 4.4e-16 |
| C54 | CONFIRMED | 0.0 |
| C55 | CONFIRMED | 0.0 |
| **C57** | **REFUTED** | see below |

### M1-C57 is false

The claim states that the position component of Ad_X(xi) is w' = R w - t R u + [p - t v]x R phi,
and that alpha does not enter it.

Block multiplication gives the top-right block of X xi^ X^-1 as

    R w + alpha v - (R phi)x (p - t v) - t R u
      = R w - t R u + [p - t v]x R phi + alpha v.

So alpha enters through alpha v. Measured over 1e4 random (X, xi) against vee(X xi^ X^-1):

- the stated formula misses by up to **8.28** (bound: 1e-12);
- raising alpha by 1 moves w' by exactly v (residual 4.7e-15).

The claim holds only when alpha v = 0. `src.gal3.Ad_apply` and M1-C11 already include `+ alpha v`.
Mutant V11, which drops that term from `src`, is killed by C11's test.

## F07–F10 rulings

**F07 — FIXED.** The harness `tests/test_mutation.py` passes on the clean tree: 20/20 killed, and each
of the five survivors fails on its named assertion (`worst_shift`, `gap`, `worst_mat`, `worst_w0`,
`worst_step`, `worst_coef`).

The new references are independent of `src`:

- `navstate_phi` is checked against the definition evaluated on gtsam NavState accessors, with
  products by gtsam `compose`.
- `imu_step_element` is checked against `scipy.linalg.expm` and `scipy Rotation`.
- `integrate_increment` is checked against `tests/ode_truth.py`, which imports numpy and scipy only.
  The check is exact at omega = 0 and first order on the time-varying stream.

Variants of the same defects that the harness does not contain were also killed (see the sweep):

- V30 (phi with 2 Dt) and V31 (body-frame R v Dt), on `navstate_phi`;
- V35 (step element with time 0), on `imu_step_element`;
- V36, V37, V38 (zero half, D0 time dropped, D0 velocity dropped), on `integrate_increment`;
- V40 (extra 1/2 a dt^2 in Euler dp), on `euler_increment`.

**F08 — FIXED.** The coefficient, derived: explicit Euler on Ddot = D A gives D_n = (I5 + dt A)^n.
With n log(I + dt A) = T A - (T dt/2) A^2 + O(dt^2), and A^2 commuting with A, this gives
D_n - exp(TA) = -(T dt/2) A^2 exp(TA) + O(dt^2). The worst-case err/dt is therefore
max_k (T/2) max|A_k^2 exp(T A_k)|.

At the test seed (20241012+41, after replaying part (a)'s draws), the measurements are:

- predicted coefficient: 2.96982 (configuration 406); median 0.5631, mean 0.6153, as recorded;
- measured err/dt through an independent `matrix_power`: 2.97055, 2.97000, 2.96987 and 2.96983 at
  n = 1e3, 4e3, 1.6e4 and 1.6e5;
- at dt = 2.5e-6: worst 7.4246e-6, so the headroom is 1e-5/7.4246e-6 = 1.347x, as recorded;
- at dt = 4e-6: 1.1879e-5; at dt = 5e-6: 1.4849e-5 with 4 configurations over 1e-5, as recorded.

Two cosmetic nits:

1. The admissible dt is 1e-5/2.9698 = 3.367e-6. At the rounded "3.37e-6" itself the worst error is
   1.0008e-5, one configuration over.
2. The headroom holds only for this seed. Over 300 other seeds of the same distribution, the worst
   coefficient over 1000 configurations ranges from 1.72 to 4.09 (median 2.29). One seed in 300 (0.3%)
   would break dt <= 2.5e-6. The hint says "on these configurations", so this is not a defect.

**F09 — PARTIALLY FIXED.** All items named in pass 2 are gone or rebound:

- C40 now goes through `src.navstate_phi`;
- C26 now checks the matrix itself;
- `trunc_at_8e2`, `forbidden_worst`, `rel_k3/rel_c2`, the C28 transcription line, C37's
  `worst_minus` and C06's `witness` are deleted, and the strawman is removed from `src`.

Remaining issues:

- **C41's last block cannot fail with respect to `src`.** The central-difference check
  d/ds expm(sA) = expm(sA) A holds for every matrix. On 50 arbitrary random 5x5 A it gives 9.5e-11,
  within the 1e-8 bound. It is not labelled as a guard.
- **C30's symbolic check is content-blind.** It asserts only `len == 15` and `max degree == 2`.
  Mutants V22 (E3 without the (alpha/2) u term) and V23 (S = I + [phi]x/2) survive the whole suite.
  Their symbolic residuals evaluated at Cay(xi) are 7.95 and 13.5 respectively.
- The re-typed implementation bodies listed in pass 2, item 4, are still present: C09 `blocks`,
  C19 `closed`, C21 componentwise log, C25 `ref`, C42 `dp`. Each still sits beside a binding check.

**F10 — FIXED.** Every figure was re-measured at its stated seed with an independent harness:

| figure | re-measured | recorded |
|---|---|---|
| C05 | 1.066e-14 | 1.07e-14 |
| C06 | 16.1465 | 16.15 |
| C19 | 3.77e-15 | 3.8e-15 |
| C21 | 1.82e-14 | 1.8e-14 |
| C23 | 1.485e-13 | 1.48e-13 |
| C24 dense sweep | 1.106e-12 | 1.106e-12 |
| C41 | 2.970 / 7.425e-6 / 1.485e-5 / 1.19e-5 | same |
| C50 | 5.6e-10 at 2e4 draws on my seed | 4.6e-10 at the test seed; consistent |

C41's "constant to 4 digits" holds only loosely: 2.9705 at n = 1e3 against 2.9698 at n = 1.6e5.

## Own mutation sweep (50 mutants, none in `tests/test_mutation.py`)

Each mutant was a single textual edit to a scratch copy of `src/`, `tests/`, `conftest.py` and
`pyproject.toml`. `tests/test_gal3.py` and `tests/test_imu_dynamics.py` were run in full against each
one. The clean copy passes.

**Survivors: medium.** These would corrupt downstream QCQP/SDP or estimation work.

| id | edit | why it survives / impact |
|---|---|---|
| V22 | `cayley_constraints_symbolic`: E3 = S pe - w (drops (alpha/2) u) | C30 checks only count and degree. The symbolic constraint feeding the QCQP is wrong by 7.95 at Cay(xi). |
| V23 | `cayley_constraints_symbolic`: S = I + [phi]x/2 | Same gap. Residual 13.5. |
| V34 | `_so3_exp_batch` closed-form c = (1 - cos th)/th instead of /th^2 | No test feeds `integrate_*` a step with \|omega\| dt >= THETA_SMALL = 0.05. All use dt <= 1e-4. One-step rotation error is 2.5e-3 at dt = 1e-2, 4.2e-2 at 5e-2 and 0.42 at 0.2. |
| V26 | `KEYFRAME_TIME_TOL` 1e-12 -> 1e-3 | The guard is tested only at t_i = 0.25. Keyframes with \|t_i\| < 1e-3 would be accepted silently, with position error g Dt t_i. |

**Survivors: low.**

| id | edit | why it survives / impact |
|---|---|---|
| V06 | `structural_residual` without the det term | Diagnostic only. C02's det = +1 check becomes vacuous. |
| V33 | `_so3_exp_batch` Taylor without the th^4/120 term | 2.4e-9 at dt = 1e-2; invisible at the test's dt. |
| V44 | `so3_log` Shepperd branch at tr > -0.99 instead of 0 | Numerically near-equivalent. |

**Survivors: equivalent** (controls):

- V02: `so3_log` small branch 2 qv instead of 2 qv/qw, a relative change of about 1e-20;
- V18: u = solve(Gamma1, v);
- V39: dR (I + W dt) instead of dR + dR W dt.

**Killed (40).** The killing tests are listed in parentheses. Medium: V03 hemisphere tolerance (C22,
C51, C52); V04/V05 inverse (C05, C07, C11, C13, C52); V07/V08 ad (C09, C10, C14, C15, C51);
V09/V10/V46 Ad (C11..C15, C51); V11 Ad_apply without alpha v and V12 velocity sign (C11 only);
V15/V17 Gamma1^-1 and log w (C21, C22, C23, C29, C51, C52); V16 Gamma2 I/2 -> I and V45 exp G2^T
(8 tests each); V19/V20 Cayley (C25..C30 / C25, C26); V21 numeric constraint residual sign (C30);
V24/V25 gravity element/generator (C32, C33); V27 right shift moved to the left (C35); V28 Forster
sign (C31, C37, C43); V29 navstate_compose (C39); V30/V31 navstate_phi (C40); V32 BIAS_SWAP = I (C55);
V35..V38 step element / integrate_increment (C45); V40 Euler dp (C41); V41 increment_blocks G1^T
(C42); V42 preintegrate via Cayley (C41, C42); V43 increment using the navigation R (C31, C34, C37,
C43); V47 time_shift sign (8 tests); V48 wedge alpha slot (8 tests); V49 kappa2 denominator (9 tests).
Low: V01 so3_exp Taylor th^4 (C41, C45, C52); V13 c2 Taylor th^4 (C21, C24); V14 kappa3 Taylor (C18,
C24, C41, C52); V50 THETA_SMALL = 1e-2 (C24).

Two mutants are each killed by a single test: V11 (Ad_apply without alpha v) only by C11, and V21
(numeric Cayley residual sign) only by C30.

## Tests that cannot fail (with respect to `src/`)

1. **C41, the central-difference ODE check.** It holds for any matrix (see F09). Not labelled.
2. **C30, symbolic part.** It is not literally unfailable, but it is blind to the content of all
   15 expressions (V22 and V23 survive).
3. **Input guards, correctly labelled.** C41 `order_gap`, the endpoint fixture's
   `assert_stream_non_commuting`, C40 `max_gap >= 0.5` (determined by the draws once the preceding
   `gap` assertion passes), and C23 `worst_theta`.
4. **gtsam-only assertions, by design** (facts about gtsam, not about `src`):
   - C37's accuracy assertion (test-local `_navstate_form_b` against `ode_truth`);
   - C53 form (b);
   - C46 and C47 margins, C48 basis and Expmap checks, all of C49 and C54;
   - C50 `worst_un`;
   - C55 `worst_naive`.
5. **Re-typed bodies** (pass 2, item 4), still present: C09 `blocks`, C19 `closed`, C21 componentwise
   log, C25 `ref`, C42 `dp`.

## Observation (no claim affected)

At the C23 seed, `src.exp(src.log(X))` reproduces X to 1.48e-13. The same log fed to `scipy expm`
gives 6.0e-15. So `src.exp` is about 25x less accurate than expm at large angles, though still four
orders of magnitude inside the 1e-9 bound.

## Scripts

All scripts are in the session scratchpad (`.../scratchpad/v3/`) and are not committed:

- `ref.py`: the independent reference;
- `part1.py` (C01–C30, C57) and `part2.py` (C31–C55): the batteries;
- `hints.py` and `hints2.py`: reproduction of the recorded figures and the C41 coefficient and seed
  spread;
- `mutants.py` and `run_mut.sh`: the 50-mutant sweep;
- `surv.py`: impact of the survivors.
