# M1 fifth-pass adversarial verification (Gal(3) mechanics and IMU dynamics)

Branch `m1-gal3-mechanics`, verified at `1bca507`, which is the orchestrator's tests-only fix for
M1-F12 (the remainder of pass 4's F11). gtsam 4.3.0, numpy, scipy, sympy.

- I did not read `docs/01_conventions_and_gal3.md`, any derivation narrative or any seeds file.
- Nothing in `src/`, `tests/` or `docs/claims/` was edited.
- `src/` is unchanged since pass 4: `git diff 51f4ca9 1bca507 -- src` is empty.
- All mutants ran on scratch copies outside the worktree, and `git status` was clean afterwards.

## Summary

| item | result |
|---|---|
| Claims M1-C01..C55 (yaml) | **55 CONFIRMED**, 0 REFUTED, 0 UNTESTABLE |
| M1-C59 (brief only, not in yaml) | **REFUTED**: Cay(xi)^-1 = Cay(-xi) holds exactly on gal(3) with alpha != 0. Max abs difference 3.6e-15 over 1e4 draws; the claim needs at least 1e-3 |
| M1-F12 | **FIXED** for every named item: N01, N06 and N20. Two new residual classes are recorded below, neither a regression: symbol assumptions (P10, medium-narrow) and coefficient domain (P08/P11, low) |
| Harness replay (all 27 `tests/test_mutation.py` mutants, as real file edits, both claim files run in full) | **27/27 killed**, each on every test it names |
| Own sweep (27 new mutants, P01..P27) | 20 killed, 7 survived. Medium: 14 mutants, 12 killed, 2 survived (P10, P19). Low/equivalent: 13 mutants, 8 killed, 5 survived |
| Full suite, clean tree | 102 passed (73 s) at the start of the pass, and again at the end (see the last section) |

## M1-C59: REFUTED

**Claim.** On gal(3), Cay(xi)^-1 and Cay(-xi) differ by at least 1e-3 (max abs) whenever
alpha != 0, because the time block couples the velocity column into the position column.

**Why it is false.** Write A = xi^. The two factors I - A/2 and I + A/2 are polynomials in A, so they
commute. Then

  Cay(-xi) = (I + A/2)^-1 (I - A/2) = [(I - A/2)^-1 (I + A/2)]^-1 = Cay(xi)^-1.

This holds for any square matrix A where both factors are invertible. On gal(3), det(I +- A/2) =
1 + |phi|^2/4 > 0. Neither the time block nor the alpha-u coupling can break the identity.

**Measurements.**

- src `cay`, 1e4 random xi with alpha = 2 N + +-0.5 (smallest abs(alpha) 6.7e-6): max abs
  difference **3.6e-15**. A second sweep over scales 0.1..10 gives max relative difference 2.8e-15.
- An independent hand 5x5 solve gives 1.9e-15.
- Exact sympy, with all 10 coordinates symbolic: C(xi) C(-xi) - I5 simplifies to the zero matrix.
- Witness: phi = 0, u = e1, w = 0, alpha = 2. Cay(xi) has velocity e1, position e1 and time 2, and
  Cay(xi)^-1 - Cay(-xi) = 0 exactly.

## M1-F12 ruling: FIXED

`1bca507` adds three pins to C30:

- the names and order of `syms`;
- `len(set(syms)) == 25`;
- every expression's free symbols lie in `syms`.

It adds one pin to C41: `preintegrate_constant(., ., 0) == I5`. It also adds N01 to the harness.

### C30 pins: what they kill and what gets through

Can a symbol have the same name AND the same assumptions but be a different object? Not as a
`Symbol`. sympy caches symbols structurally, so `Symbol('alpha', real=True)` is the same object
each time, with equal hash. `Symbol('alpha', real=True, finite=True)` also compares equal to it,
because the assumptions are canonicalised.

Different objects that keep the string `alpha` do exist:

- a `Symbol` subclass;
- a `_sympystr` override that prints `alpha` for a symbol whose real name differs.

A `Dummy` does not keep it: it prints as `_alpha`.

| id | edit (in `cayley_constraints_symbolic`) | result |
|---|---|---|
| P01 | `syms` carries `Dummy('alpha', real=True)` | killed (name pin: `_alpha`) |
| P02 | Dummy alpha used in both the exprs and `syms` | killed (name pin) |
| P04 | `syms` carries a same-named Symbol subclass | killed (`stray`) |
| P14 | `syms` carries `Symbol('alpha', extended_real=True)` | killed (`stray`) |
| P05 | Re as `Matrix(MatrixSymbol('Re', 3, 3))` | killed (name pin: `Re[0, 0]`) |
| P06 | `syms` with pe before ve | killed (name pin) |
| P07 | extra unused symbol `beta` appended | killed (name pin; `len`) |
| P09 | `implemented_function('h', x -> x)(alpha)` in E3 | killed (`sp.Poly` raises PolynomialError) |
| P12 | coefficient 0.5 + 1e-9 | killed (content, 1e-12 relative) |
| P15 | expressions returned as `sp.Poly` (equivalent) | killed incidentally (lambdify cannot print Poly) |
| P16 | printer-spoofed symbol (prints `alpha`, real name `beta`) | killed incidentally (lambdify NameError), not by a pin |
| P03 | Symbol **subclass** `alpha`, used consistently in both | **SURVIVES** (low): the Hessian is correct, but `Symbol('alpha', real=True)` is not in `free_symbols`, so a consumer that rebuilds symbols by name misses it |
| P13 | all 25 symbols without `real=True`, consistently | **SURVIVES** (low): coefficients and solve are correct; the same name-rebuild mismatch applies |
| P10 | alpha declared `positive=True`, consistently | **SURVIVES (medium-narrow)**: see below |
| P08 | coefficient is a 0-arg `implemented_function` h() = 0.5 | **SURVIVES** (low): Hessian entry `-h()`, Poly domain `QQ[h()]`; `float()` and `np.array(..., float)` still evaluate it |
| P11 | `0.5 * alpha` (Float) | **SURVIVES** (equivalent): domain RR instead of QQ |

**N01 and N06 are closed.** A foreign alpha cannot enter through `syms` alone. A same-named
symbol cannot hide in the expressions. Positional order is pinned.

**P10 is a new residual (medium-narrow).** The pins check names, not assumptions. With alpha
declared positive:

- the content check, which works on numbers, passes;
- `sp.solve(E3[0](at a true Cayley point with alpha = -1.3), alpha)` returns **[]**, where the clean
  src returns [-1.3].

alpha is a time increment, and the claim's domain is "Total", so alpha can be negative. A sympy
consumer that solves or simplifies would silently lose those solutions. A Hessian-only QCQP builder
is unaffected.

Suggested one-line pins:

- `syms == [sp.Symbol(n, real=True) for n in expected_names]`. This kills P03, P10, P13 and P16
  at once.
- `sp.Poly(e, *syms).domain in (sp.ZZ, sp.QQ)`. This kills P08 and P11.

### C41 initial-condition pin

- **N20 is killed by the block itself.** A constant left factor (`C @ exp`) now fails
  `worst_init`.
- **The block alone is still blind to a factor that is discontinuous at 0.** P17 returns
  `C @ exp(sA)` for s != 0 and I at s = 0. Replaying the CD block plus the init check on its own
  gives worst_ode 7.0e-11 (bound 1e-8) and init 0.0.
- **The test as a whole kills it.** Part (a) fails with 4.0e-3 (bound 1e-11), and C42 also fails.
  P18 (right factor) behaves the same way.
- **Smooth variants die in the CD block.** ODE uniqueness forbids any smooth factor: P20
  (conjugation, D(0) = I) and P21 (O(Dt^2) generator drift, 1e-9) are both killed.
- **Ruling: FIXED.**

## Mutation results

### Replay of the harness (27 mutants, real file edits)

Every `MUTATIONS` entry in `tests/test_mutation.py` was applied textually to a fresh scratch copy.
`tests/test_gal3.py` and `tests/test_imu_dynamics.py` were then run in full. The unmutated copy:
55 passed.

**27/27 killed, and every named test failed.** That covers:

- the 4 fixture/order mutants and the 7 positive controls;
- navstate_phi x2, cay_denominator, and the 3 Euler/increment mutants (M45, M49, M51);
- the 2 commuting-stream mutants and the 3 Taylor/THETA_SMALL/c2 mutants;
- the F11/F12 mutants: E3_no_alpha_u, S_plus, alpha_fresh_symbol (new in `1bca507`, killed by C30),
  keyframe_tol, the 2 batch Rodrigues mutants, and structural_residual_no_det.

### Own sweep: 27 new mutants

**Medium (14): 12 killed, 2 survived**

| id | edit | result |
|---|---|---|
| P01, P04, P14 | foreign alpha in `syms` | killed (C30) |
| P05, P06 | Re as MatrixSymbol; ve/pe order swapped | killed (C30) |
| P09 | non-polynomial f(alpha) in E3 | killed (C30) |
| P10 | alpha `positive=True` | **SURVIVES**: sympy solve loses alpha < 0 |
| P17, P18 | constant left/right factor except at Dt = 0 | killed (C41 (a), C42) |
| P19 | constant left factor only for Dt > 2 | **SURVIVES**: error 5.0e-3 at Dt = 2.5 and 1.0e-2 at Dt = 5 |
| P20 | conjugated exp, D(0) = I | killed (C41, C42) |
| P21 | generator drift Dt^2 * 1e-9 | killed (C41, C42) |
| P24 | cay position + 1e-9 u | killed (C25, C26, C30) |

P19 sits inside C41's stated domain, which has no upper bound on Dt. Every test samples
Dt <= 2.0: C41 (a) uses {0.02, 0.1, 0.4, 1, 2}, C42 uses (0.02, 0.5), and the remaining tests use
smaller values. Suggest adding Dt in {5, 20} to C41 (a), or stating Dt <= 2 in the domain.

**Low / equivalent (13): 8 killed, 5 survived**

| id | edit | result |
|---|---|---|
| P02 | consistent Dummy alpha | killed (name pin) |
| P03 | consistent Symbol subclass | **SURVIVES** (low) |
| P07 | extra unused symbol | killed |
| P08 | 0-arg implemented-function coefficient | **SURVIVES** (low) |
| P11 | Float 0.5 | **SURVIVES** (equivalent) |
| P12 | 0.5 + 1e-9 | killed |
| P13 | all symbols non-real | **SURVIVES** (low) |
| P15 | exprs as Poly | killed (incidental) |
| P16 | printer-spoofed name | killed (incidental) |
| P22 | `exp(abs(Dt) xi)` | **SURVIVES** (low): negative Dt is outside the claim |
| P23 | gravity_element time * (1 + 1e-10) | killed (C32, C33) |
| P25 | `cay_matrix` returns (I + A/2)^-1 (I - A/2), i.e. Cay(-xi) | killed (C25, C26). Note that it equals Cay(xi)^-1 exactly, consistent with C59 being false |
| P26 | inverse with p + t v | killed (C05, C11, C13, C52) |
| P27 | cay time abs(alpha) | killed (C25..C29) |

## Per-claim verdicts

`src` is unchanged since pass 4, and the clean suite passes. I re-ran spot checks against an
independent numpy/scipy/gtsam reference that imports nothing from `src` except the code under
test. Seed 55555; 2000 draws for the group and algebra checks, 500 for increments and gtsam, and 10
DOP853 navigation solves with time-varying inputs and random g.

| claim | verdict | spot check (max error) |
|---|---|---|
| C01 | CONFIRMED | compose vs 5x5 product 7.1e-15 |
| C05 | CONFIRMED | inverse vs `np.linalg.inv` 1.8e-14 |
| C09 | CONFIRMED | ad vs commutator columns 0.0 |
| C11 | CONFIRMED | Ad vs vee(X e^ X^-1), relative 7.6e-16 |
| C12 | CONFIRMED | 3.1e-15 relative |
| C15 | CONFIRMED | 1.3e-14 relative |
| C17 | CONFIRMED | det Ad - 1: 4.7e-15 |
| C19 | CONFIRMED | exp vs expm 4.2e-15 |
| C22 | CONFIRMED | 1.7e-15 |
| C23 | CONFIRMED | 1.4e-15 relative |
| C25 | CONFIRMED | 8.9e-16 |
| C26 | CONFIRMED | 8.9e-16 |
| C27 | CONFIRMED | 7.8e-16 |
| C30 | CONFIRMED | 15/15 expressions identical (exact sympy) to an independent derivation; Hessian alpha-u entry -1/2; degree 2; domain QQ |
| C31 | CONFIRMED | endpoint vs DOP853 1.1e-14 |
| C32 | CONFIRMED | 8.9e-16 |
| C41 | CONFIRMED | preintegrate_constant vs expm, Dt up to 2: 3.4e-14 (but see P19 for Dt > 2) |
| C42 | CONFIRMED | 3.4e-14 |
| C45 | CONFIRMED | 1.8e-15 |
| C51 | CONFIRMED | Expmap 8.9e-16, Logmap 1.6e-15, AdjointMap 1.8e-15 |

**C02-C04, C06-C08, C10, C13, C14, C16, C18, C20, C21, C24, C28, C29, C33-C40, C43, C44, C46-C50,
C52-C55: CONFIRMED.** Their passing tests are unchanged and src is unchanged, so pass 4's
independent measurements stand.

**M1-C59: REFUTED** (above).

**Documentation staleness (observation).** The yaml `test_hint` for C30 does not mention the new
name, order and free-symbol pins, and the one for C41 does not mention the D(0) = I check.

## Recommendations

1. C30: compare `syms` to `[sp.Symbol(n, real=True) for n in expected_names]` (kills P03, P10,
   P13, P16), and require `Poly(e, *syms).domain` in {ZZ, QQ} (kills P08, P11).
2. C41: extend part (a) to Dt in {5, 20}, or bound Dt in the claim's domain (P19).
3. Drop or annotate M1-C59: Cay(-xi) = Cay(xi)^-1 holds identically on gal(3).

## Final suite run (clean tree)

See the commit that closes this pass.
