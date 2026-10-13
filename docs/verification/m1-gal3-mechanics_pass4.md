# M1 fourth-pass adversarial verification (Gal(3) mechanics and IMU dynamics)

Branch `m1-gal3-mechanics`, verified at `51f4ca9` (the M1-F11 fix commits `4218798..51f4ca9`).
gtsam 4.3.0, numpy, scipy, sympy, mpmath. I did not read `docs/01_conventions_and_gal3.md`, any
derivation narrative or any seeds file. Nothing in `src/`, `tests/` or `docs/claims/` was edited. All
mutants ran on a scratch copy outside the worktree, and `git status` was clean afterwards.

## Summary

| item | result |
|---|---|
| Claims M1-C01..C55 (yaml) | **55 CONFIRMED**, 0 REFUTED, 0 UNTESTABLE |
| M1-C58 (brief only, not in yaml) | **REFUTED**: alpha does not enter det(I5 - xi^/2). Max error of the claimed formula 16.6 over 1e4 draws, bound 1e-12 |
| M1-F11 | **PARTIALLY FIXED**. Every named item is killed or rebound: V22, V23, V26, V34, the C41 tautology, V06, V33. A new hole remains in the C30 symbolic check: symbol identity (N01, below) |
| Replay of the existing harness (26 mutants, applied as real file edits, full claim files run) | **26/26 killed**, each on all of its named tests |
| Own mutation sweep (36 new mutants) | 31 killed, 5 survived: **1 medium** (N01), 2 low, 2 equivalent |
| Full suite, clean tree | 101 passed (65 s) before this pass and again after it |

## Method

`ref.py` (scratch) is an independent reference. It uses numpy, scipy, sympy and mpmath, and imports
nothing from `src/` or `tests/`:

- the group as 5x5 `@` and `np.linalg.inv`;
- exp as `scipy.linalg.expm`, and Cayley as the 5x5 solve;
- Ad and ad as columns of vee(X e_i^ X^-1) and of the commutator;
- log from scipy's rotvec plus a linear solve for (u, w);
- Gamma_m as a 60-term series, and kappa/c2 at 60-digit mpmath;
- the increment ODE and the navigation ODE written from scratch and solved with DOP853 at
  rtol = atol = 1e-13. Inputs are polynomial-plus-trig omega(t) and a(t), with random g ~ 5 N(0, I)
  and Dt in (0.1, 1.5).

`src` appears only as the code under test. gtsam claims were checked directly against gtsam.

## M1-C30: the symbolic Cayley constraints, derived independently

With A = xi^ = [[P, u, w], [0, 0, alpha], [0, 0, 0]] and P = [phi]x:

- I5 - A/2 = [[S, -u/2, -w/2], [0, 1, -alpha/2], [0, 0, 1]], with S = I - P/2;
- I5 + A/2 = [[I + P/2, u/2, w/2], [0, 1, alpha/2], [0, 0, 1]].

Write X = Cay(xi) = [[Re, ve, pe], [0, 1, te], [0, 0, 1]] and match the blocks of
(I5 - A/2) X = I5 + A/2:

- rotation: S Re = I + P/2;
- time row: te - alpha/2 = alpha/2, so te = alpha;
- velocity column: S ve - u/2 = u/2, so S ve = u;
- position column: S pe - (u/2) te - w/2 = w/2, so S pe = w + (alpha/2) u.

The `+ (alpha/2) u` coupling comes from the -u/2 entry multiplying te = alpha.

Exact sympy checks on `src.gal3.cayley_constraints_symbolic()`:

- My three blocks minus src's 15 expressions: **15 of 15 identically zero**, expanded exactly.
- Substituting the fully symbolic 5x5 Cay(xi): all 15 simplify to 0. The symbolic time block is
  alpha, and rows 3 and 4 are exact.
- No spurious solutions. The 15 equations are linear in (Re, ve, pe), with
  det d(exprs)/d(Re, ve, pe) = (|phi|^2 + 4)^5 / 1024 > 0. For every real (phi, u, w, alpha) the
  only solution is Cay(xi), so the variety is exactly the graph of Cay. Re in SO(3) is implied
  (M1-C27). The time output is not a lifted variable: te = alpha is substituted.
- Total degree is exactly 2 in every expression, and the clean src's expressions contain no
  symbol outside `syms`.

### Attacks on the new C30 test

I tried to build a wrong set of degree-2 constraints that the test still accepts. Part (ii) of the
test compares the 15 lambdified values with a hand residual at 300 generic random points, to 1e-12
relative. That pins every expression as a polynomial, so the following are all rejected:

- equivalent forms that vanish at Cayley points: N02 (Re S - (I + P/2)) and N07 (2 x E2);
- sign errors that cancel at alpha = 0: N04, N09;
- wrong operands: N03 (S^T), N08 (uses ve in place of u);
- transposed symbol naming: N05.

**Hole found: symbol identity (N01, medium, SURVIVES).** The test maps values to symbols BY NAME,
both through lambdify and through its `val[...]` dictionary. If the `alpha` returned in `syms` is a
different sympy object with the same name, every check passes. In the mutant it is `Symbol("alpha")`
without `real=True`, while the expressions use the real alpha. As polynomials in the declared lifted
variables, however, the constraints are then WRONG:

- `sp.Poly(E3[2], *syms)` has domain `QQ[alpha]`. alpha is treated as a coefficient, but the
  reported degree is still 2, so the degree assertion passes.
- `sp.hessian(E3[2], syms)` has **zero** alpha-u entries. The quadratic form a QCQP builder would
  extract silently loses the (alpha/2) u coupling.
- `subs(syms -> values)` leaves `-0.5*alpha`, so only a later `float()` fails loudly.

The same hole applies to any variable, and the degree check cannot see a hidden product either. The
fix is one line: `assert set().union(*(e.free_symbols for e in exprs)) <= set(syms)`, and ideally
also `len(set(syms)) == 25`.

**Low (N06, SURVIVES).** Returning `syms` with Re in column-major order passes, because the test
feeds values by name. A consumer that indexes `syms` positionally would mis-assign the Re entries.
The claim does not fix the order, so this is low. Pinning
`[str(s) for s in syms]` to the documented order would close it.

## M1-F11 ruling: PARTIALLY FIXED

| item | ruling | evidence |
|---|---|---|
| V22 (E3 without (alpha/2) u) | FIXED | replayed as a real edit: C30 fails on `worst_sym_cay` |
| V23 (S = I + P/2) | FIXED | replayed: C30 fails |
| C30 content in general | PARTIALLY FIXED | N02..N09 killed; N01 (symbol identity) survives, see above |
| V26 (KEYFRAME_TIME_TOL 1e-3) | FIXED | replayed: C36 fails on `near_bad`; N11 (one-sided guard), N12 (wrong entry), N13 (check off by default) all killed |
| V34 (`_so3_exp_batch` c over th) | FIXED | replayed: C42 and C45 fail; N16 (sin/t^2), N17 (K transposed), N18 (no th^4 in c) killed |
| V33 (Taylor without th^4/120) | FIXED | replayed: C42 fails |
| C41 expm tautology | FIXED | the CD check now differentiates src `preintegrate_constant`; see below |
| V06 (structural_residual without det) | FIXED | replayed: C02 fails; N23 (no row 4), N24 (no (3,3)) killed |

Notes on the fixes:

- **C36 guard.** It is pinned to an interval: any tolerance in [5e-13, 2e-12) passes. N14
  (1.9e-12) survives, which is harmless (equivalent). The guard admits NaN, since abs(nan) > tol is
  False. The NaN then propagates into the endpoint position and time, so the failure is loud. This
  is an observation, not a defect.
- **C41.** The central-difference block alone now fails at 1.0e-2 (bound 1e-8) when src's
  generator is scaled by 1.01. It does not check D(0) = I, so a constant left factor C exp(sA)
  passes it (N20: 7.5e-11). Part (a) of the same test still kills N20 (1.13 > 1e-11), and C42 does
  too.

### The C42 1e-12 bound decision

On 2,906 angles (0, logspace 1e-12..0.05, a dense grid on [0.03, 0.05), both sides of 0.05,
0.05..3.14, pi, 4, 7.5, 12, 30), the batched Rodrigues against scipy and against 40-digit mpmath
gives:

- worst error **1.496e-13** at theta = 0.04999 (recorded 1.53e-13);
- the cause is the th^4-truncated sin(th)/th: th^6/5040 times |K| ~ th, which is 1.5e-13;
- the scalar `so3_exp` has the same 1.50e-13.

The bound has 6.7x headroom and kills every meaningful defect: V33 (2.6e-9), N18 (2.1e-11),
V34 (O(1)).

It is loose in one direction. N19 changes the th^4 coefficient of c from 1/720 to 1/700, a 3%
coefficient error. It gives 6.1e-13 and **survives** (low). The bound also hides that the Taylor
branch is the error source in the first place. N15 moves the batch switch to 1e-3 and gets
**8.9e-16** everywhere (equivalent, survives), because the (1 - cos)/th^2 cancellation is masked
by |K K| ~ th^2.

Verdict: the 1e-12 decision is acceptable for the claim as worded. Two better options exist: a
5-term Taylor branch (error ~1e-16) with a 1e-14 bound, or a switch near 1e-3 for the batch only.

## Per-claim verdicts

Each figure is the maximum absolute error against the independent reference, over my own draws
(seed 777 / 4242 / 31337, not the test seeds). The claim's bound is in brackets.

| claim | verdict | attack and measurement |
|---|---|---|
| C01 | CONFIRMED | 1e4 pairs, t ~ 3 N: compose vs 5x5 @ 3.6e-15 [1e-12] |
| C02 | CONFIRMED | 1e4: 2.2e-15 [1e-12]; residual sees det = -1 alone (2.0), row 3 (0.3), row 4 (0.3), diag (0.3) |
| C03 | CONFIRMED | 1e4 triples through `compose`: 7.1e-15 |
| C04 | CONFIRMED | 0.0 |
| C05 | CONFIRMED | 1e4, t ~ 3 N: 1.8e-14 vs I5 and vs `inv`; (X X^-1)[3,4] = 0.0 |
| C06 | CONFIRMED | max commutator 17.4 over 200 pairs [>= 1e-3] |
| C07 | CONFIRMED | 0.0 |
| C08 | CONFIRMED | 0.0; wedge = hand hat |
| C09 | CONFIRMED | `ad` vs commutator-built: 0.0 |
| C10 | CONFIRMED | homomorphism 3.6e-15, Jacobi 5.3e-15 |
| C11 | CONFIRMED | `Ad`, `Ad_apply` vs vee(X e^ X^-1): 1.1e-14 |
| C12 | CONFIRMED | 7.1e-15 |
| C13 | CONFIRMED | 1.1e-14 |
| C14 | CONFIRMED | 2.3e-14 (both forms) |
| C15 | CONFIRMED | 2000 draws, theta in (0, 3): 2.0e-13 [1e-8] |
| C16 | CONFIRMED | 0.0 |
| C17 | CONFIRMED | det - 1: 4.2e-15; trace 0.0 |
| C18 | CONFIRMED | vs 60-term series, theta in (1e-3, 3): 1.5e-14 [1e-12] |
| C19 | CONFIRMED | 1e4, theta log-uniform 1e-9..2 pi - 0.3, alpha ~ 2 N: 1.5e-13 [1e-9] |
| C20 | CONFIRMED | 0.0 |
| C21 | CONFIRMED | Gamma1^-1 Gamma1 - I over 2e4, theta 1e-9..pi: 1.9e-14 [1e-13]; c2 at pi and pi - 10^-k: rel 1.4e-16 vs mpmath |
| C22 | CONFIRMED | 3000 draws: 3.6e-15; vs independent Log 6.2e-15; at abs(phi) = 3.5 the result is 2.7832 (wrapped) |
| C23 | CONFIRMED | 5000, R = Exp(5 N): 1.2e-13 [1e-9]; R at pi - {0, 1e-14, 1e-10, 1e-7, 1e-4}: 7.8e-16 |
| C24 | CONFIRMED | det 1.0 / 0.40528 / 2.2128e-3, det at 2 pi 3e-30, cond(pi) = 1.570796; 7000-point sweep vs 60-digit series: worst rel 1.60e-12 at 0.0516 [1e-11]. The recorded 1.41e-12 is sweep-dependent; mine is 1.60e-12 |
| C25 | CONFIRMED | 1e4, alpha ~ 2 N: 1.8e-15 |
| C26 | CONFIRMED | matrix exact (0.0); det 1.3e-15; abs(phi) ~ 50, abs(u), abs(w) ~ 1e3: rel 3.6e-15 |
| C27 | CONFIRMED | 7.8e-16; time block exact |
| C28 | CONFIRMED | 2000 draws: 6.2e-15; 5 tabulated values reproduce to 4-5 s.f.; large-c angle readback consistent |
| C29 | CONFIRMED | s^3 ratio 0.060 (src) / 0.053 (ref) [1.0]; s^5 ratio 7.9e-3 / 6.0e-3 [1.0] |
| C30 | CONFIRMED | exact symbolic identity with an independent derivation (above); numeric residuals 0 at Cay; unique-solution determinant (abs(phi)^2 + 4)^5 / 1024 |
| C31 | CONFIRMED | 25 independent DOP853 streams, random g: 7.5e-14 [1e-9] |
| C32 | CONFIRMED | G = T(-Dt) H': 8.9e-16; = expm(Dt zeta_g): 1.8e-15; H' T(-Dt) = (I, g Dt, -1.5 g Dt^2, -Dt): 3.6e-15 |
| C33 | CONFIRMED | 5.7e-14 |
| C34 | CONFIRMED | time Dt exact; position p_j + g Dt^2: 2.5e-14; R, v correct |
| C35 | CONFIRMED | p_j - v_j Dt: 7.6e-14; time 0 |
| C36 | CONFIRMED | t_i ~ 2 N: p_j + g Dt t_i 2.4e-14, time t_i 2.2e-16; the guard refuses 1.0000001e-12, 2e-12, -1e-11, 1e-6 and 0.25; it accepts 0, -0.0, +-5e-13 and 1e-12 (and NaN, see above) |
| C37 | CONFIRMED | gtsam NavState form vs DOP853: 7.5e-14 |
| C38 | CONFIRMED | W_g and T(Dt) U_g: 1.8e-15; form (a) vs form (b): 8.9e-16 |
| C39 | CONFIRMED | gtsam compose vs closed form: 8.9e-16; src mirror 8.9e-16 |
| C40 | CONFIRMED | Phi(I) = I exactly; definition 0.0; automorphism 1.3e-15; conjugation 0.0 |
| C41 | CONFIRMED | closed form vs DOP853 at Dt in {0.02..2}: 1.4e-13 [1e-11]; Euler first order (err/dt 1.9938 / 1.9937 / 1.9937 at n = 1e3 / 4e3 / 1.6e4 on my 200 configurations); signed coefficient mismatch 2.1e-4 at dt = 1e-4 [1e-2]. The 2.970 coefficient is seed-specific (pass 3 reproduced it at the test seed) |
| C42 | CONFIRMED | blocks vs expm, Dt up to 2: 1.0e-13 [1e-12]; batched Rodrigues 1.50e-13 [1e-12] |
| C43 | CONFIRMED | 7.5e-14 |
| C44 | CONFIRMED | 4.4e-16; sign check 8.9e-16 |
| C45 | CONFIRMED | three separate DOP853 solves: 1.7e-13 [1e-10]; algebraic 3.6e-15 |
| C46 | CONFIRMED | 0.0 |
| C47 | CONFIRMED | 0.0 with p ~ 3 N + 10, v ~ N |
| C48 | CONFIRMED | 0.0; basis and Expmap checks hold |
| C49 | CONFIRMED | position in column 3; retract(e3) moves position, velocity 0 |
| C50 | CONFIRMED | 2e4 draws: permuted 6.4e-10 [1e-9]; unpermuted 7.1e-5; s = 1e-4: 4.6e-8 (fails, as the domain states) |
| C51 | CONFIRMED | Expmap 1.1e-13, Logmap 1.6e-14, AdjointMap 3.6e-15, adjointMap 0.0 |
| C52 | CONFIRMED | compose/inverse 1.8e-14, between 1.8e-14, localCoordinates vs independent Log 5.3e-14, retract = X expm(d^) 1.8e-15 |
| C53 | CONFIRMED | PIM predict vs T(-Dt) H' X_i D_ij: 8.9e-16; time 0 |
| C54 | CONFIRMED | 0.0 |
| C55 | CONFIRMED | 0.0 |
| **C58** | **REFUTED** | det(I5 - xi^/2) = det(S) * 1 * 1 = 1 + abs(phi)^2/4, because the lower 2x2 block is unit upper-triangular whatever alpha is. Over 1e4 random xi with alpha ~ 2 N, max abs(det - (1 + abs(phi)^2/4 + alpha^2/4)) = **16.57** [claimed 1e-12]; the SO(3)-only formula holds to 2.7e-15. Witness: phi = 0, alpha = 2 gives det = 1.0, claimed 2.0. The conditioning conclusion is also backwards: at phi = e1, cond(I5 - xi^/2) is 1.118 at alpha = 0 and 4.000 at alpha = 3, so alpha makes it WORSE. Consistent with C26 ("only phi enters") |

## Mutation results

### Replay of `tests/test_mutation.py` (26 mutants, real file edits)

Every mutant in the harness was applied textually to a scratch copy. Both claim files were run in
full against each one. All **26/26 are killed**, and each fails every test it names: the four
fixture/order mutants, the cay/gamma/H'/compose/navstate_phi/denominator positive controls, the five
F07/F09 mutants, the three Taylor/c2 mutants, and the six F11 mutants.

### Own sweep: 36 new mutants (none in the harness)

**Medium** (would corrupt downstream QCQP/SDP or estimation work): 23 mutants, **22 killed,
1 survived**.

| id | edit | result |
|---|---|---|
| N01 | `syms` carries a same-named non-real `alpha` | **SURVIVES**: Hessian loses the alpha-u coupling (see C30) |
| N03 | E2 uses S^T | killed (C30) |
| N04 | E3 `- alpha/2 u` | killed (C30) |
| N05 | Re symbols named transposed | killed (C30) |
| N08 | E3 uses ve for u | killed (C30) |
| N09 | numeric residual E3 alpha sign | killed (C30) |
| N10 | closed-form cay velocity += alpha w / 2 | killed (C25, C26, C29, C30) |
| N11 | guard one-sided (t_i > tol) | killed (C36) |
| N12 | guard reads X[4,3] | killed (C36) |
| N13 | `endpoint(check_time=False)` default | killed (C36) |
| N16 | batch s = sin/th^2 | killed (C42, C45) |
| N17 | batch K transposed | killed (6 tests) |
| N20 | `preintegrate_constant` with a constant left rotation | killed (C41 (a), C42); the C41 CD block alone passes |
| N21 | `increment_blocks` dp = 0.5 Dt^2 Gamma1 a | killed (C42) |
| N22 | generator time slot -1 | killed (C41, C42) |
| N28 | Ad[6:9, 9] = -v | killed (4) |
| N29 | ad +alpha | killed (5) |
| N30 | log w sign | killed (6) |
| N32 | inverse p sign | killed (4) |
| N33 | gravity generator time +1 | killed (C32) |
| N34 | `lift_navstate` p/v swap | killed (C38, C40) |
| N35 | bias halves swapped | killed (C44) |
| N36 | `Ad_apply` drops -t R u | killed (C11 only) |

**Low / equivalent:** 13 mutants, **9 killed, 4 survived**.

| id | edit | result |
|---|---|---|
| N02 | E1 = Re S - (I + P/2) (valid, equivalent at Cayley points) | killed: the test is form-specific |
| N06 | `syms` Re column-major | **SURVIVES** (low): name-based test, positional consumers broken |
| N07 | E2 scaled by 2 (valid) | killed (form-specific) |
| N14 | tol 1.9e-12 | **SURVIVES** (equivalent) |
| N15 | batch switch at 1e-3 | **SURVIVES** (equivalent; more accurate, 8.9e-16) |
| N18 | batch c without th^4 | killed (C42, 2.1e-11) |
| N19 | batch c th^4/700 | **SURVIVES** (low; 6.1e-13 under the 1e-12 bound) |
| N23 | `structural_residual` no row 4 | killed (C02) |
| N24 | `structural_residual` no (3,3) | killed (C02) |
| N25 | kappa3 Taylor t^4 sign | killed (C24) |
| N26 | c2 closed form with 1 - cos | killed (C24) |
| N27 | THETA_SMALL 3e-2 | killed, but only by the literal pin `THETA_SMALL == 0.05` (numerically it would pass) |
| N31 | so3_log small branch at n < 1e-3 | killed (C29) |

Single-test kills: N36 by C11 only; N33 by C32 only; N35 by C44 only.

## Tests that cannot fail, or are blind (with respect to `src/`)

1. **C30 degree assertion and name-keyed content check.** They are blind to symbol identity and
   positional order (N01, N06). The degree is computed in whatever objects `syms` holds, so a
   foreign symbol is silently demoted to a coefficient.
2. **C41 CD block.** It now binds src (fails at 1e-2 under a 1% generator scaling) but cannot see a
   constant factor, because D(0) = I is not asserted there. Covered elsewhere by part (a).
3. **C24 `THETA_SMALL == 0.05`.** A literal pin. It kills a numerically harmless change (N27) and
   is not evidence of accuracy.
4. **Input guards, correctly labelled and not evidence about src.** C30 `smallest_pert >= 1e-2`;
   the C45 large-step `min |omega| dt >= THETA_SMALL`; C41 `order_gap`; C40 `max_gap`; C23
   `worst_theta`.
5. **gtsam-only assertions, by design.** These are as in pass 3: C37 accuracy, C53 form (b), C46-C49
   and C54 margins, C50 `worst_un`, C55 `worst_naive`.
6. **Re-typed bodies (pass 2 item 4).** Still present beside binding checks, for example C42
   `dp - Dt^2 gamma2 a` using src `gamma2`. The C45 large-step `worst_large_nav` re-states the
   first-order recursion. It is intentional (it exercises the batch Rodrigues), but it is not an
   accuracy check.

## Recommendations

1. C30: assert `free_symbols subset of syms`, `len(set(syms)) == 25`, and the documented name order
   of `syms` (kills N01 and N06).
2. C42: tighten the batched-Rodrigues accuracy, with either a longer Taylor or a lower switch, and
   bound it at <= 1e-14 (kills N19).
3. C41: add `preintegrate_constant(omega, a, 0) == I5` to the CD block.
4. Drop or annotate M1-C58. It is false, and its conditioning conclusion is reversed.
