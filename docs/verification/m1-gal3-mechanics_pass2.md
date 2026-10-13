# M1 — Gal(3) mechanics: second-pass adversarial verification

Branch `m1-gal3-mechanics`, at 67fa62b. gtsam 4.3.0, numpy/scipy/sympy/mpmath from the project venv.
In scope: claims M1-C01..C55 in `docs/claims/M1.yaml`, the extra claim M1-C56 (given in the brief, not
in the yaml), and whether the remediation commits 40e866f..67fa62b fixed findings M1-F01..F06.

Every claim was judged by running a computation, not by reading the code or the yaml for consistency.
I did not read the derivation narrative (`docs/01_conventions_and_gal3.md`) or any seeds file.

## Summary

| | |
|---|---|
| Claims CONFIRMED | **55** (C01–C55) |
| Claims REFUTED | **1** (C56) |
| Claims UNTESTABLE | 0 |
| F01 / F02 / F03 / F04 / F05 / F06 | FIXED / FIXED / FIXED / FIXED (named items; leftovers listed below) / **PARTIALLY FIXED** / FIXED |
| Mutation sweep | 53 mutants of `src/` + 4 mutants of the test side: **52 killed, 5 survived** |
| New findings | P2-F01 (C56 is false), P2-F02 (C41's tolerance rule fails on its own data), P2-F03 (survivors and tests that cannot fail) |

## Method

1. **Independent reference** (`ref.py`, scratchpad; imports nothing from `src/` or `tests/`), written
   from the equations only:
   - product = 5×5 matmul, inverse = `np.linalg.inv`, exp = `scipy.linalg.expm(hat ξ)`, and
     Cayley = `solve(I−A/2, I+A/2)`;
   - ad built column by column as `vee([ξ^, e_i^])`, Ad as `vee(X e_i^ X⁻¹)`, and Γ_m as a 60-term
     series;
   - IMU ground truth: my own `solve_ivp` right-hand sides for the navigation ODE and for the full
     5×5 increment ODE `Ḋ = D ξ(t)^`, ξ = (ω, a, 0, 1). The input family differs from
     `tests/ode_truth.py`: polynomial drift plus sinusoids, so ω changes direction. DOP853 at
     rtol = atol = 1e-13 agrees with Radau (1e-12) to 4.7e-14.
2. **Randomized properties**, 1e4 samples unless noted. The θ draws were mixed across bands:
   - U(0, 2π−0.3);
   - log-uniform 1e-12..1e-1;
   - within ±0.1 % of the Taylor switch (THETA_SMALL);
   - π − 10^U(−12,−2);
   - exactly π, and coordinate-axis 180° rotations.

   Group elements used |t| up to ~9, |p| ~ 5·N(0,1), and Haar-random R.
3. **Mutation sweep** on throw-away copies of the repo in the scratchpad. The full suite (minus
   `test_mutation.py` and `test_orchestrator.py`) was run per mutant. The unmutated copy passes 55/55.
   The worktree was never edited (`git status` clean before and after).
4. **Reproduction of recorded figures**: a scratch copy of the tests was instrumented to print every
   asserted quantity at the tests' own seeds. The printed values were compared with the "measured"
   figures in the yaml.

## Additional claim M1-C56 — REFUTED

> "For ξ = (0, u, w, α), exp(ξ) has position block w + α·u exactly … max abs error ≤ 1e-12."

With φ = 0, A = ξ^ satisfies A³ = 0, so exp(A) = I + A + A²/2. The term A² has (0:3, 4) block α·u,
so the **position block is w + (α/2)·u, not w + α·u**. sympy confirms the closed form:
`(I + A + A²/2)[0:3,4] = (α u_i/2 + w_i)`.

Numerically, over 1e4 random (u, w ~ N(0, I₃), α ~ N(0, 1)):

```
max|expm(xi^)[0:3,4] - (w + alpha u)|     = 4.17         (claimed <= 1e-12)
max|expm(xi^)[0:3,4] - (w + alpha u / 2)| = 1.8e-15 (scipy expm), 0.0 (src.exp), 0.0 (gtsam Gal3.Expmap)
error / |alpha u / 2|                     in [0.999999999998, 1.000000000000]
```

The error is exactly the missing half, on every draw. The "coasting p + v t" intuition belongs to the
group product (the v₁·t₂ term of C01), not to the exponential. Along a one-parameter subgroup the
velocity itself grows from 0, so the transport term is the average velocity, i.e. half. This agrees
with C19 (the α Γ₂(φ) u term, Γ₂(0) = I/2) and with C32 (exp(Dt·(0, g, 0, −1)) has position
−½ g Dt²). C56 contradicts both of those confirmed claims.

## F01–F06 rulings

| finding | ruling | computation |
|---|---|---|
| **F01** decorative 5000-step ground truth | **FIXED** | C31/C34/C37/C43 now compare against a DOP853 truth that imports nothing from `src`. Recorded 1.5e-14; reproduced 1.51e-14. With **my own** ODE (arbitrary random g, Dt ∈ (0.1, 1.5), a different input family), 150 instances: endpoint vs truth 1.3e-13 on all 25 entries, with time block 0.0. The degraded fixture (2 steps, dt = 0.25) is now **killed** by all four tests (my run T2). The discrete recursion is kept only for labelled identity checks, plus a first-order convergence check (e1 = 8.6e-4 ≤ 2e-3, halving ratio inside [1.8, 2.2]). |
| **F02** constant / commuting inputs | **FIXED** | All accumulation-order mutants are now killed with time-varying ω(t): `integrate_truth` increment `dR@E→E@dR` (M39), nav `R@E→E@R` (M40), `integrate_increment` (M41), `euler_increment` `dR@W→W@dR` (M42), and reversed sample order (M44). Zeroing ω's oscillation in `ode_truth` is caught by C41's order-gap guard (T3). Residual gap: the shared endpoint fixture has no guard of its own that its stream is non-commuting. |
| **F03** false C40 lower bound | **FIXED** | The new statement is true. Φ_Dt(I) = I exactly; Φ(X) − X = v·Dt in position (gap − max\|v\|Dt ≤ 2.2e-16). Automorphism error 1.8e-15 over 2e4 pairs. Φ_{−Dt}∘Φ_Dt = id (1.1e-16). In Gal(3), T(−Dt) X T(Dt) = (R, v, p + vDt, 0) with error 0.0 over 2e4. The disclaimed bound really is false: a fitted left factor misses by < 0.5 on 13.9 % of 2e4 random pairs (the claim says 14 %). |
| **F04** assertions that cannot fail | **FIXED** for the named items | C52 localCoordinates is now compared with `src.log`: 7.99e-15 at the test seed. Against an independent check it is 1.95e-14 over 2e4 draws, and gtsam and `src.log` both reproduce `rel` to ≤ 1.4e-14 through expm. (scipy `logm` itself drifts to 2.4e-12 near π, so it is not usable as the reference there.) C28's tabulated rows now go through `cay`. C26 now compares with the 5×5 definition. The `min > 0` guards became `assert_term_exercised`; the test seed gives frac = 0.9985 ≥ 0.99, so the guard can fail. Leftover tautologies are listed below. |
| **F05** unreproducible recorded figures | **PARTIALLY FIXED** | The three figures F05 named now reproduce exactly at the test seeds: C52 retract 1.283e-13 (recorded 1.3e-13), C50 4.636e-10 (4.6e-10) and 3.80e-8 (3.8e-8), C41 Euler 7.425e-6 (7.4e-6). Still wrong: **C41's "err/dt = 0.2494, constant across n"**. The worst-case err/dt is 2.97 at the test seed and 2.095 over my 1000 configurations (constant to 4 digits across n = 1e3..1.6e5, so the first-order part is right). Neither the worst case, the median (0.56) nor any natural statistic gives 0.2494. Following from that, the hint's own rule "dt ≤ 5e-6 with tolerance 1e-5" **fails at dt = 5e-6** on the test's own 1000 configurations: 1.485e-5, with 4 configurations above 1e-5 (P2-F02). Older figures that the remediation did not touch also fail to reproduce at the current test seeds (see table below). |
| **F06** C24 switch point | **FIXED** | The constraint is now stated and asserted, and is correct. κ1 Taylor truncation: 7.752e-13 at 5e-2, 1.034e-11 at 7.7e-2, 1.301e-11 at 8e-2 (crosses 1e-11 at ≈ 0.0766). The M29 mutant (THETA_SMALL → 8e-2) is killed by C24, C21 and C52. Literal 1 − cos above the switch gives κ3 6.1e-10 and c2 2.0e-10 on [0.05, 0.06] (claim: 4.9e-10, 1.9e-10 "near 5e-2"), so the 2 sin² evaluation is required. |

Recorded measurements that still do not reproduce at the current test seeds (none comes near its
tolerance; only cases with ≥ 1.5× discrepancy, or where the recorded value is the smaller):

| claim | recorded | reproduced |
|---|---|---|
| C06 witness | 5.0 | 16.15 |
| C19 vs expm | 3.9e-14 | 3.8e-15 |
| C23 | 7.5e-14 | **1.49e-13** (recorded value understates) |
| C05 | 4.4e-15 | 1.07e-14 |
| C41 central difference | 2.1e-10 | 8.9e-11 |
| C51 Expmap | 3.9e-14 | 1.06e-15 |
| C51 Logmap | 1.0e-13 | 5.3e-15 |
| C53 | 4.4e-16 | 1.8e-15 |
| C24 "≤ 1.1e-12 on the 863-point sweep" | 1.1e-12 | 1.106e-12 on that sweep; **1.36e-12** on my denser 6003-point sweep, still 7× under the 1e-11 bound |

## Per-claim verdicts

My worst over 1e4 samples unless noted; the claim's tolerance in brackets.

| claim | verdict | attack and number |
|---|---|---|
| C01 | CONFIRMED | `compose` vs matmul, \|t\| ~ 3N: 7.1e-15 [1e-12]; M01/M02 killed |
| C02 | CONFIRMED | structural residual 2.4e-15 [1e-12] |
| C03 | CONFIRMED | associativity through `compose` 1.4e-14 [1e-10] |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | vs `np.linalg.inv` 2.8e-14; X X⁻¹ − I 2.3e-14; time entry 0.0 |
| C06 | CONFIRMED | witness 6.47 |
| C07 | CONFIRMED | t(X₁X₂) = t₁ + t₂ bit-exact (tested as equality); t(X⁻¹) = −t exact |
| C08 | CONFIRMED | 0.0; u → column 3, w → column 4 (M07 killed) |
| C09 | CONFIRMED | `ad` vs commutator-built ad_num: 0.0 |
| C10 | CONFIRMED | homomorphism 3.6e-15, Jacobi 7.6e-15 |
| C11 | CONFIRMED | `Ad` vs vee(X e_i^ X⁻¹) 2.1e-14; `Ad_apply` 5.7e-14, \|t\| up to ~9 |
| C12 | CONFIRMED | 3.6e-14 |
| C13 | CONFIRMED | 7.6e-14 |
| C14 | CONFIRMED | 1.1e-13 [1e-7] |
| C15 | CONFIRMED | test seed 1.8e-14 [1e-8] (M08–M11 killed through this test) |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det(Ad) − 1: 5.3e-15; trace(ad) 0.0 |
| C18 | CONFIRMED | Γ1, Γ2 vs 60-term series over all θ bands including θ = π and up to 2π − 0.3: 1.9e-14 [1e-12] |
| C19 | CONFIRMED | `exp` vs expm **including θ < 0.05 and θ up to 2π − 0.3** (beyond the test's 0.05..2.9): 1.5e-13 [1e-9] |
| C20 | CONFIRMED | 0.0 with \|α\| ~ 3N |
| C21 | CONFIRMED | Γ1⁻¹Γ1 − I over θ ≤ π: 1.9e-14 [1e-13]; `log` vs scipy logm (θ < π − 1e-2): 6.1e-14; c2 finite at π |
| C22 | CONFIRMED | log(exp ξ) at \|φ\| = π − {1e-3 … 1e-10}: ≤ 2.8e-15; general 2.0e-13; \|φ\| = 3.5 → 2π − 3.5 with axis flipped (4.4e-16) |
| C23 | CONFIRMED | exp(log X) − X: Haar 1.1e-14; π − 1e-9 1.3e-14; **π exactly** 1.6e-14; axis-aligned π 8.9e-15; 1e-13 3.1e-15 |
| C24 | CONFIRMED | det Γ1 = 1, 0.405285, 0.00221276, ~0 (2.3e-65) at 0, π, 6, 2π; cond(π) = π/2; min det on [0, 2π) > 0. Coefficient relative error vs a 60-digit series reference over 6003 points: ≤ 1.36e-12 [1e-11]. No branch at 1e-4: κ3 1.00, c2 6.29 (matches). Switch numbers: see F06 |
| C25 | CONFIRMED | vs 5×5 definition 3.6e-15; sympy: blocks are exactly S⁻¹(w + α/2·u), S⁻¹u, α |
| C26 | CONFIRMED | sympy: det(I − ξ^/2) − (1 + \|φ\|²/4) ≡ 0; numerical 5.3e-15 |
| C27 | CONFIRMED | orthogonality/det residual 5.2e-14 at \|φ\| up to 1e3; time block exact |
| C28 | CONFIRMED | all 5 rows to 4 s.f. through the definitional Cayley (angle exact, axis ≤ 7e-14); \|θ − 2 atan(\|c\|/2)\| ≤ 2.4e-13 for \|c\| up to 1e4 |
| C29 | CONFIRMED | with independent logm: max/s³ = 0.065 [1.0]; s⁵ residual ratio 8.5e-3 at both s = 1e-2 and 5e-2 (true s⁵ scaling); wrong-sign cubic gives 256 |
| C30 | CONFIRMED | residuals at definitional Cayley 5.0e-14 (\|φ\| up to 10, large u, w); all 15 sympy degrees = 2 |
| C31 | CONFIRMED | own ODE, random g: 1.3e-13 on all 25 entries; time 0.0 [1e-9] |
| C32 | CONFIRMED | exp(Dt ζ_g) X_i D vs truth 1.3e-13; H′T(−Dt) has −1.5 g Dt² (test 3.6e-15) |
| C33 | CONFIRMED | test seed 5.7e-14 [1e-10] |
| C34 | CONFIRMED | time − Dt = 0.0; position − (p_j + g Dt²) 7.5e-14; R, v 1.3e-13; minimum true miss 0.15 |
| C35 | CONFIRMED | position − (p_j − v_j Dt) 1.8e-13 vs ODE truth; time 0.0 |
| C36 | CONFIRMED | own 5×5 build, t_i ~ 2N: position − (p_j + g Dt t_i) 7.5e-14; time − t_i 4.4e-16; R, v unchanged; `ValueError` enforced (M33 killed) |
| C37 | CONFIRMED | gtsam NavState form vs own ODE truth 1.3e-13 |
| C38 | CONFIRMED | test seed: W_g 8.9e-16, T(Dt)U_g 0.0, forms 3.6e-15 (M26 killed) |
| C39 | CONFIRMED | test seed 8.9e-16; M34 killed |
| C40 | CONFIRMED | see F03 |
| C41 | CONFIRMED (statement) | exp(Dt ξ) vs my own 5×5 ODE, constant inputs, Dt ≤ 2: 1.7e-13; Euler first order (err/dt constant to 4 digits). **The hint's tolerance rule is wrong** (P2-F02) |
| C42 | CONFIRMED | blocks vs expm, \|ω Dt\| up to ~15: 2.9e-13 |
| C43 | CONFIRMED | Forster vs own ODE truth 1.3e-13 |
| C44 | CONFIRMED | 8.9e-16; M35/M36 killed |
| C45 | CONFIRMED | three separate own-ODE solves at a random split: 9.9e-14; the dv_jk·Dt_ij form misses by ≥ 1.7e-3; algebraic 8.9e-16 |
| C46 | CONFIRMED | 0.0 (1e4) |
| C47 | CONFIRMED | 0.0; test-side ctor p/v swap (T1) killed by C51 |
| C48 | CONFIRMED | Hat/Vee 0.0 |
| C49 | CONFIRMED | NavState matrix [R\|p\|v] 0.0; retract(1e-4 e3) moves position, Gal3.Expmap moves velocity |
| C50 | CONFIRMED | 5.7e-10 at s = 1e-5 (1e4 draws) [1e-9]; 4.0e-8 at s = 1e-4 (bound correctly disclaimed); P = (0:3, 6:9, 3:6) (M25 killed) |
| C51 | CONFIRMED | Expmap vs expm 5.7e-14; adjointMap vs ad_num 0.0; AdjointMap vs Ad_num 1.4e-14; Logmap 5.8e-14 |
| C52 | CONFIRMED | compose 3.6e-15, inverse 1.8e-14, between 2.1e-14, retract vs X expm(d) 3.6e-15, Exp(d)X misses by ≥ 0.086; localCoordinates see F04 |
| C53 | CONFIRMED | also with **time-varying** inputs (beyond the claim's constant ones): predict vs form (a) 1.8e-15 |
| C54 | CONFIRMED | 0.0 |
| C55 | CONFIRMED | 0.0; M37/M38 killed |
| **C56** | **REFUTED** | position block is w + α u/2; error exactly \|α u\|/2, max 4.17 over 1e4 [claimed ≤ 1e-12] |

## P2-F02 — C41's tolerance rule fails at its own limit

The C41 hint says: "use n with dt ≤ 5e-6 and tolerance 1e-5". Replaying part (b) of the test exactly
(seed 20241053, the same 1000 configurations at Dt = 0.4):

```
dt = 2.5e-6 : worst 7.425e-6   err/dt 2.9698   configs > 1e-5: 0
dt = 5.0e-6 : worst 1.485e-5   err/dt 2.9698   configs > 1e-5: 4
```

The rule passes only because the test picks dt = 2.5e-6. The claim's statement is not affected. The
prescription needs either dt ≤ 3.3e-6, or the tolerance written as about 3.5·dt. Also, the
"err/dt = 0.2494" figure matches no statistic I could compute (see F05).

## Mutation sweep

There are 53 mutants of `src/` (M01–M53), plus T1–T4 on the test side and the fixture. Every listed
test failed for the mutant named.

**Killed: 52 of 57.**

| mutant | killed by (count; main tests) |
|---|---|
| M01 compose v1·t2 → v2·t1 | 15: C01, C03, C05, C12, C31, C34, C35, C36, C38, C40, C43, C45, C52, C53, C32 |
| M02 compose drop v1·t2 | 15 (as M01, minus C03 — the product stays associative — plus C33) |
| M03 compose v1+R1v2 → v1+v2 | 10: C01, C03, C05, C12, C31, C34, C43, C45, C52, C53 |
| M04 / M05 inverse (p − tv → p + tv, p) | C05, C11, C13, C52 |
| M06 `gal()` p/v columns swapped | 28 tests, including C46 |
| M07 wedge u/w swapped | C08, C09, C11, C19, C25, C26, C29, C48 |
| M08 / M09 ad (−αI sign, u → w column) | C09, C10, C14, C15, C51 |
| M10 / M11 Ad (skew(p − tv), −tR signs) | C11, C12, C13, C15, C51 |
| M12 Ad_apply −tRu sign | C11 only |
| M13 κ3 drop ½ | C15, C18, C19, C24, C41, C51, C52 |
| M14 Taylor κ1 drop θ⁴ term | C18, C21, C24, C41, C52 |
| M15 / M16 exp α Γ2 u → α Γ1 u / dropped | C15, C19, C22, C23, C32, C41, C42, C51, C52 |
| M17 so3_exp sin sign | C15, C19, C22, C23, C41, C51, C52 |
| M18 log drop −tΓ2u | C21, C22, C23, C29, C51, C52 |
| M19 / M20 Cayley ±α/2, dropped ½ | C25, C26, C29, C30 |
| M21 H′ −½ → +½ | C31, C32, C38, C43, C53 |
| M22 endpoint X_i/D order swap | C31, C35, C36, C38, C43, C53 |
| M23 T(−Dt)·H′ order swap | C31, C35, C38, C43, C53 |
| M24 so3_log hemisphere test off | C22, C28, C51, C52 |
| M25 NAVSTATE_PERM without the swap (GTSAM footgun) | C50 only |
| M26 lift_navstate p/v swap | C38 only |
| M28 Forster drop v_i Dt | C31, C37, C43 |
| M29 THETA_SMALL → 8e-2 | C21, C24, C52 |
| M30 c2 → the 1/θ² − (1 + cos)/(2θ sin) form | C21 only |
| M31 so3_exp Taylor branch drop θ⁴ term | C41, C52 |
| M33 keyframe-time check off | C36 |
| M34 navstate_compose extra coupling | C39 |
| M35 / M36 bias halves swapped / sign flipped | C44 |
| M37 / M38 bias_to_gtsam / bias_from_gtsam naive order (GTSAM footgun) | C55 |
| M39 / M40 / M43 / M50 integrate_truth (increment order, nav order, dropped ½, gravity sign) | C31, C34, C37, C43 |
| M41 / M44 / M48 integrate_increment order, reversed samples, chain order | C45 |
| M42 euler dR@W → W@dR | C41 |
| M46 increment_blocks dp Γ2 → ½Γ1 | C42 |
| M47 increment_generator: a moved into the w slot | C41, C42 |
| M52 / M53 Γ2 κ2 sign, Γ1⁻¹ −Φ/2 sign | 6–7 each |
| T1 test-side Gal3(R, v, p, t) constructor swap (GTSAM footgun) | C51 |
| T2 fixture 2 steps at dt = 0.25 (F01) | C31, C34, C37, C43 |
| T3 ode_truth ω made constant (F02) | C41 |

**Survived: 5.**

- **M27 / T4** — `navstate_phi`: changing p + vDt to p − vDt, or to no shift at all, survives. **No
  test calls `src.gal3.navstate_phi`.** C40 uses a test-local `phi()` built on gtsam.
- **M32** — `cay_denominator` changed from I − ξ^/2 to I + ξ^/2 survives. Both have determinant
  1 + \|φ\|²/4, and C26 only ever takes `det()` of this function.
- **M45** — `integrate_increment` updating dp with the *new* dv survives. The per-step element becomes
  (E, a dt, 1.5 a dt², dt), which is still a Gal(3) element, so C45's identity check still holds. No
  test compares `integrate_increment` against the ODE.
- **M49** — `imu_step_element` dropping the ½ survives. The function is dead code as far as the tests
  are concerned.
- **M51** — `euler_increment` changed to semi-implicit (dp uses the updated dv) survives. It is still
  first order, so C41's first-order and ratio checks pass. C41's "every block uses the pre-update
  value" is not pinned.

## Tests that cannot fail (with respect to `src/`)

1. **C40, parts (i) and (ii)** never touch `src`. `N_L.compose(X) == phi(X)` holds by construction,
   since N_L = φ(X)X⁻¹. `gap == max|v|·Dt` holds by the test's own definition of φ. Only part (iii)
   exercises `src`.
2. **C26**: `det(cay_denominator)` and `min_det ≥ 1` do not change under the sign of ξ^/2 (M32).
3. **Pure-math or test-internal checks** that no change to `src` can affect: C24 `trunc_at_8e2 > 1e-11`
   (local `taylor_k1`); C21 `forbidden_worst ≥ 1e-10` (local `forbidden`); C28's transcription line
   `2 tan(θ/2) ≈ tabulated`; C37 `worst_minus ≥ 1e-3` (gtsam only); C41 `order_gap` (ode_truth only);
   C24 `rel_k3/rel_c2`, which only exercise `gamma_coefficients_closed_form`, a strawman kept for the
   test. These are valid statements about the mathematics, but they are not evidence about the
   implementation.
4. **Re-typed implementation bodies.** These fail only if that very function body changes, never
   because the formula is wrong; each sits next to an independent check that does bind:
   - C09 `A == blocks`;
   - C19 `closed`;
   - C21 "componentwise log";
   - C25 `ref`;
   - C42 `dp == Dt²Γ2 a`.
5. **C06** `witness is not None`: always true once any draw has a nonzero commutator.

## Scripts

All scripts are in the session scratchpad (`.../scratchpad/v2/`) and are not committed:

- `ref.py` — the independent reference;
- `prop_group.py`, `prop_exp.py`, `c24.py`, `cay.py`, `imu_end.py`, `imu2.py`, `c40.py`, `c52.py`,
  `c56.py` — the batteries;
- `mutants.py`, `mut2.py` — the mutation harness and the test-side mutants;
- `meas/` — the instrumented test copy used for figure reproduction.

The full suite on the clean tree passes: 84 passed.
