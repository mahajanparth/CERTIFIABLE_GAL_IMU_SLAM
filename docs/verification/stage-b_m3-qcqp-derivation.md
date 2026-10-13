# Stage B verification: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier report, base commit d2371af.

Scope: src/qcqp/* (fixture, gtsam_boundary, instance, lifting, forms, cost, constraints), tests/test_qcqp.py,
tests/test_mutation_m3.py, docs/claims/M3.yaml (M3-C01..C35, C39), and the brief-only claim M3-C40.
I did not read docs/03_qcqp_derivation.md, experiments/m3_scratch/, or any seeds file.
Python: CERTIFIABLE_GAL_IMU_SLAM/.venv (gtsam 4.3.0). All verifier scripts live outside the worktree
(session scratchpad): `ind.py` (the independent re-implementation), `cmp1-3.py` (comparison against src),
`oracle.py` (the per-mutant observable-output oracle), `hunt2.py` (spurious-point hunt), `claims1-5.py`
(per-claim computations), `mutants.py`/`harness.py` (mutation sweep).

## Verdict summary

* **Claims: 36 CONFIRMED, 1 REFUTED (M3-C40), 0 UNTESTABLE.**
* **Spurious feasible points of P0-P3: none.** The 301 random and penalised least-squares solves that converged
  all decode to valid states, the algebraic constructions show the same, and there is a short proof (Section 3).
* **Cost: z^T Q z agrees with my independent Q to 1.1e-16 relative, and with a GTSAM factor graph
  (2*graph.error) to first order** (1.4e-4, 2.9e-6, 2.5e-7 at eps = 1e-2, 1e-3, 1e-4).
* **Mutation sweep: 68 new mutants. 33 killed, 18 equivalent (each with a proof below), 17 survive and are
  NON-EQUIVALENT (8 medium, 9 low).**
* **GATE (zero surviving non-equivalent mutants): FAILED.** M3 does not pass as it stands. The survivors sit
  in parts of src/qcqp that tests/test_qcqp.py never checks: the fixture literals, the decoder's v/p
  scaling, `project_so3`, `whitened_residual` weights, the PRIMARY/P1 labelling, and some small helpers. The core
  objects M4 needs (z, the families, Q) matched my independent implementation exactly, and every mutant that
  corrupted them was killed.

## 1. Independent re-implementation (task 1)

I wrote `ind.py` from the claim statements and GTSAM directly. It does not import src or tests. It contains:

* PIMs built with the stated params (MakeSharedU(9.81), sigma_g 1.7e-4, sigma_a 2e-3, int 1e-10). Ubar comes
  from `predict(NavState(), bbar)` left-multiplied by a numerically inverted G. J is a central FD (h = 1e-4) of
  the scipy `logm` chart of Ubar^-1 U(b), so it uses neither gtsam.Gal3.Logmap nor the src/test chart. The taus
  are 3/tr of the NavState blocks. Truth comes from chaining `predict` at b_true.
* Lifting: the C01 index map, with V' = R_j J_nu b and the other liftings as stated. Aniso rho = M^T r, with r
  taken from my own 5x5 cleared residual.
* Constraint families (P0-P3, RA1, RB1, RB2, RC1, RG, RL, RH, RS, P4, RR) written as numpy polynomial functions
  of z, not as matrices.
* The residual forms r_R, r_v, r_p, written as functions of z. I derived them by hand from
  A(I - x^/2) - B(I + x^/2), with G^-1 = (I, -g Dt, -1/2 g Dt^2, Dt), and checked them against the 5x5 residual
  (max 3.6e-15 over 200 random lifted states).
* Q: the stated cost (iso, aniso, synthetic-W aniso, partial) evaluated as a function of z and then polarised
  exactly into a matrix.

How src compares (both bbar = b_true and bbar = 0):

| object | max difference |
|---|---|
| dR, dv, dp | 0 (bitwise) |
| J (10x6) | 3.3e-12 to 4.6e-12 (two different FD charts; \|J\| ~ 1.2) |
| tau_R/v/p, W_vp, rw | 0 relative |
| truth (closed form vs GTSAM chain) | <= 1.5e-14 |
| z_of_state base / aniso, 30 states | 3.0e-12 (from the J difference) |
| every family, constraint by constraint (up to sign) as polynomials, at 8 generic random z | 0 unmatched in all 14 families, counts identical |
| Q iso | 1.1e-16 relative to max\|Q\| = 1.108e10 |

In the oracle (`oracle.py`, unmutated tree), Q for iso, aniso, synthetic-W and partial, `factored_cost`,
`whitened_residual`, the decoder at scales {1, -1, -2.5, 0.3}, `project_so3`, the residual forms, every
family, and the helpers all match: ORACLE-CLEAN.

## 2. Spurious-feasible-point hunt (task 2)

(a) Ground truth, at both bbar values, satisfies every family. The largest truth residuals are P1/RA1
2.2e-16 and P4 4.9e-16. Over 200 random lifted states the largest are RR 1.5e-11 and P4 1.3e-12, inside
C12's 1e-10; all the rest are <= 1.6e-13.

(b) Search for primary-feasible points that do not decode to a valid state, at bbar = 0 and at bbar = b_true:

* Plain trust-region least squares on P0-P3 with an analytic Jacobian, from 75 random starts per instance
  (scales 0.3, 1, 3): 75/75 converged to a residual below 1e-10. All decode to s*z(state(z)), with the worst
  value of max(\|z - s z(state)\|, \|det R - 1\|) equal to 2.2e-10.
* Penalised solves that push away from valid states, 12 per penalty: det R_0/s -> -1, det R_2/s -> -1,
  Y_0 -> random, V'_1 -> random, R_0 -> 0, s = -1 with R_0 held, and Y inconsistent with b. Each solve was then
  polished on the primary constraints alone. 151 (zero) and 150 (true) solutions converged in total. **0 were
  bad.** The worst consistency error was 1.6e-9 / 2.4e-9, the same level as the solver tolerance.
* Algebraic constructions. Each one violates a primary family except -z:

| point | primary max | violated |
|---|---|---|
| det R_0 = -1, liftings consistent | 1.95 | P2, RH, RS |
| -R_0 only | 1.95 | P2, P3, RL |
| s = -1 only | 1.98 | P2, P3, RL |
| z negated except s | 1.98 | P2, P3, RL |
| -R_1 and its adjacent liftings negated | 1.84 | P2, RH, RS |
| swap R_1, R_2 blocks | 4.9e-3 | P3, RL, RH, RS |
| swap edge-0 / edge-1 liftings | 2.2e-3 | P3 + all lifting families |
| -Y_0 | 1.5e-2 | P3, RG, RL, RH, RS |
| swap V, V' | 2.3e-4 | P3, RG, RL, RH |
| R_0 rotated by pi, liftings stale | 7.2e-2 | P3, RL, RH, RS |
| R_0 = 0, b_0 = 0 | 1.0 | P1, RA1 (C21) |
| -z (global) | 2.2e-16 | none (decodes to the same state, C23) |

**Proof that none exist.** P0 gives s = ±1. P1 gives R^T R = s^2 I = I. For orthogonal R,
cof(R) = det(R) R, so P2 gives det(R) R = s R, i.e. det(R) = s. Write R = s R' with R' in SO(3): then
cof(R) = cof(R') = R' = s R, which is consistent. P3 gives M^T Y = s[w(b)]_x, so Y = M[w]_x / s. With
M = s M' and b = s b', this is Y = s M'[w(b')]_x, which is s times the lifting of the decoded state. V, V', P, P'
follow the same way. v, p and b are free. So P0-P3 is exactly {±z(state)}. Primary feasible set is
CONFIRMED, with no spurious points.

## 3. Cost vs GTSAM (task 3)

Reference: my own graph. It uses GalileanImuFactor `cloneWithNewNoiseModel(Diagonal.Precisions([tau_R x3,
tau_p x3, tau_v x3]))`. I confirmed the NavState (rot, pos, vel) order by perturbing v_1 alone, which moves only
entries 6:9. The rest of the graph is PriorFactorPose3 (sigma 1e-3), PriorFactorVector v0, GPSFactor KF1/KF2
(0.05), and BetweenFactorConstantBias with sigmas in (accel, gyro) order sqrt(0.5)*(3e-3, 1.9e-5). The cost is
2*graph.error.

* bbar = b_true, at truth: src factored 1.9e-19, mine 5.8e-24, GTSAM 1.7e-26. **The dense z^T Q z gives 1.39e-8**,
  from cancellation against max\|Q\| 1.1e10.
* 30 perturbations each: max \|src/GTSAM - 1\| = 1.37e-4 (eps 1e-2), 2.90e-6 (1e-3), 2.54e-7 (1e-4).
  The C26 bound 0.05*eps holds.
* bbar = 0: truth cost 3.2968e-3 (QCQP) vs 3.3157e-3 (GTSAM), a 0.57% relative gap that does not shrink with
  eps (1.2e-3 to 1.5e-3 at the perturbations). This is the Cayley / first-order surrogate (C27/C28). It is
  outside C26's domain, but M4 should know about it.
* Partial variant (C33): 1.32e-4 / 2.83e-6 / 2.34e-7. Aniso synthetic-W (C34): body frame 1.9e-3 / 1.7e-4 /
  3.1e-5; world frame 7.0e-2 / 5.7e-2 / 6.6e-2.

## 4. Isotropy claims (task 4)

* C31 obstruction: **the proof is valid.** v_j (and p_j) appears in z only in its own slot; no lifting depends
  on v or p. So along v_j -> v_j + lam d, any quadratic form z^T Q z has lam^2 coefficient e_d^T Q e_d,
  whatever the base point. For rho_v = M^T r_v the coefficient is d^T M W M^T d. That is independent of
  M in SO(3) for every d only when W = cI, since the orbit of d^T M W M^T d spans [lam_min, lam_max] |d|^2.
  Measured: 3I is constant at 4.020000 (ptp 5e-13). diag(1,2,5) ranges over [1.347, 6.666] for v and
  [1.348, 6.681] for p, against the theoretical [1.34, 6.70]. The argument applies to any Q, so "only if
  W = cI" holds on the whole feasible set, not just for this Q.
* C32 monomial fit, my sampler with seeds 4 and 99 (the claim uses 2): 3\|rho_v\|^2 gives 2.7e-14 / 2.6e-14,
  rho_v.rho_p 2.7e-14 / 2.6e-14, diag(1,2,5) 8.1e-2 / 8.0e-2, random SPD 1.4e-1, diag(1,1,1.01) 8.9e-4,
  rho_p with diag(1,2,5) 0.22.
* Aniso cost: **exactly 12 extra liftings (rho_v, rho_p, 3 + 3 per edge), n = 136.** P4 (M rho = s r) adds 12
  rows. They fix rho uniquely, because M is invertible on the P1 variety, so the primary feasible set is still
  exactly the lifted states. My polarised aniso cost matches src Q (oracle). Minimality is not claimed and I
  did not test it.
* C33: D eigenvalues are 3.46e7 (>= 0), and lambda_max(W_R) = 6.92e7 = tr(W_R)/2 to 1e-7 relative. W_R is
  nearly isotropic, so D is nearly 0.5 tr(W_R) I. The correlation is -0.8668.

## 5. Per-claim verdicts (task 7)

| claim | verdict | attack / evidence |
|---|---|---|
| C01 | CONFIRMED | independent index map and encoder, z vs src 3e-12 (J-chart limited), s at 123, n 124/136 |
| C02 | CONFIRMED | 112, 12; Rai Eq. (28) text has Y, Y', W^v, W^p per edge (24). V' is needed: R_j J_nu b contains R_j J_R b_a, which no other lifting holds (Y' carries only b_g) |
| C03 | CONFIRMED | all 7 triples: n = 39K+6M+16, direct count minus n = -39; the triples are in the cached Rai HTML (Sec. VII-B, Table II) |
| C04 | CONFIRMED | fixture J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| <= 3.8e-12; 6 new time-varying streams (own logm chart): 0, 0, 3.5e-11; Log-chart linearity 3.7e-12 |
| C05 | CONFIRMED | 1000 constructions max 2.2e-15; bottom rows exactly 0; arbitrary X_j min max\|r\| 0.94 |
| C06 | CONFIRMED | 2.9e-15 |
| C07 | CONFIRMED | hand derivation from 5x5 (Section 1); src forms = my forms (oracle); 3.6e-15 vs 5x5 |
| C08 | CONFIRMED | src forms equal mine, which carry -1/2 on Y, Y', V, V', P, P' and 0 on bias; perturbing Y_0 by 1e-6 gives cost tau_R/8 \|d\|^2 ~ 7.8e-5 >= 1e-6 |
| C09 | CONFIRMED | truth <= 2.2e-16, random <= 1.6e-13 |
| C10 | CONFIRMED | <= 1.7e-15 random |
| C11 | CONFIRMED | <= 4.1e-14 random |
| C12 | CONFIRMED | P4 1.3e-12, RR 1.5e-11 random (b-norm 0.2 sampler); <= 1e-10 |
| C13 | CONFIRMED | min family max 4.2e-6 (P0, base), 1.5e-5 (RB1, aniso) |
| C14 | CONFIRMED | ranks from MY evaluation matrices (700 generic z): sizes as stated, each family full rank, joint 453 (gap at 453: 1.8e-1 -> 8.7e-16), aniso 471, primary 106/118 |
| C15 | CONFIRMED | proof: the column-c cofactor equals R^(a) x R^(b) and the row-c cofactor equals R_(a) x R_(b); both sets are {cof(R)_ij - s R_ij} |
| C16 | CONFIRMED | rank 11 per KF; trace identity residual 2.7e-15 |
| C17 | CONFIRMED | holds trivially for any z^T A z representation; exactly one b = 1 |
| C18 | CONFIRMED | proof in Section 2, plus 301 converged solves, 0 spurious |
| C19 | CONFIRMED | proof: R invertible gives det(R) R^-T = R, so R R^T = det(R) I and det^2 = det^3, so det = 1. Rank 2 is impossible (adj has rank 1). Rank <= 1 gives cof = 0, so R = 0 |
| C20 | CONFIRMED | P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied set <= 2.2e-16 |
| C21 | CONFIRMED | P1 = RA1 = 1.0, rest <= 1.4e-17 |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: >= 1.3e-2; Gram families <= 2e-20 |
| C23 | CONFIRMED | exact; holds for any quadratic form (see Section 7) |
| C24 | CONFIRMED | src Q = my polarised cost (1.1e-16 rel), which uses residuals checked against 5x5 |
| C25 | CONFIRMED | min/max eig -2e-16; Q[b, non-b] exactly 0; bias block = rw exactly |
| C26 | CONFIRMED | Section 3 |
| C27 | CONFIRMED | devs rot/vel/pos 2.01e-2 / -1.22e-2 / -2.10e-3; halving ratios 0.250/0.249/0.249, 0.501/0.500/0.500, 0.504/0.502/0.501 |
| C28 | CONFIRMED | 2.088e-7 ... 4.064e-10, factors 8.02, 8.01, 8.00 (own expm and Cayley) |
| C29 | CONFIRMED | half 1.8e-7, Rai literal 0.78 |
| C30 | CONFIRMED | own whitening: rank 45, cond 3.9e7 / 4.5e7 |
| C31 | CONFIRMED | proof valid (Section 4) |
| C32 | CONFIRMED | seeds 4, 99 (Section 4) |
| C33 | CONFIRMED | Section 3/4 |
| C34 | CONFIRMED | body <= 0.31 eps; world >= 5.7e-2 at every eps |
| C35 | CONFIRMED | seeds 4 and 99: rank 2015, null 263 (gap 1.9e-7 -> 4e-15); my restricted forms span 60/92/106/136/226/236 |
| C39 | CONFIRMED | my PIMs: 6.9204e7, 4.8049e5, 5.8906e6 / 6.9204e7, 4.8167e5, 5.8972e6; rw 5.5402e9, 2.2222e5; Dt 0.5000000000000003 |
| **C40** | **REFUTED** | For orthogonal R, cof(R) = det(R) R, so with s = +1 a reflection gives cof(R) = -R and P2's residual is 2\|R_ij\|, not 0. R = Exp([0.3,-0.7,1.2]) diag(1,1,-1): max\|cof(R) - R\| = 1.85, \|cof(R) + R\| = 2.2e-16. Over 1000 random det = -1 orthogonal R, the minimum of max\|P2\| is 1.39. P2 alone excludes reflections at s = +1 (this is C19 and C20). The redundant rows are not needed for that. |

## 6. Mutation sweep (task 5)

68 new mutants, none of them in tests/test_mutation_m3.py. Each one is a single exactly-once substitution,
applied to a scratch copy of src + tests/test_qcqp.py. The judge is the full tests/test_qcqp.py, slow tests
included, run with `-x`. Every mutant was also run through `oracle.py`, which compares every observable output of
src/qcqp with the independent implementation. "Equivalent" requires ORACLE-CLEAN plus a written proof. The
worktree stayed clean throughout (git status empty apart from this report).

### Killed (33)
fx-m2-sign, fx-sigma-ba-rw (C39), gb-left-chart, gb-integrate-arg-swap, gb-G-time-sign, in-S2-no-third,
in-D-full-trace (C33), in-gnss-kf2-only (C24), in-prior-from-kf1 (C08), in-rw-order-swap, in-Wvp-order,
lf-decode-b-abs (C01), lf-rho-uses-Rj, lf-Yp-uses-M, lf-matform-rowmajor, fm-cross-swap (C09), fm-eps-sign,
co-vecrows-rowmajor (C08), co-gam-uses-ba (C07), co-assemble-nosym (C25 exact symmetry), co-partial-G2-kron-order,
co-p0-prior-half, co-v0-prior-half, co-gnss-sigma-not-squared, co-R0prior-rowmajor, co-rp-dt-vj, co-cp-jside-only,
co-aniso-rot-weight (C24 aniso), cn-RC1-P-uses-cv, cn-RR-cross-sign, cn-P4-uses-Rj, cn-RS-jside-uses-M-w,
cn-RH-drop-VP-cols (C14).

### SURVIVORS, non-equivalent (17: 8 medium, 9 low). Every one is a gate failure.

| id | mutation | oracle difference | class | why it matters |
|---|---|---|---|---|
| fx-sigma-prior | SIGMA_PRIOR 1e-3 -> 3e-3 | prior weights, Q | **medium** | the test file imports the sigma from src, so its reference moves with the mutant; no fingerprint pins it. M4 would certify a different cost from the one the domain states |
| fx-sigma-gnss | SIGMA_GNSS 0.05 -> 0.5 | GNSS weights, Q | **medium** | same mechanism |
| fx-bg-true | B_G[2] 0.008 -> 0.009 | streams, dR (5e-4), J, cov, truth | **medium** | the instance's IMU data change. Reference and src move together; C39's 4-significant-figure taus do not see it |
| fx-v0 | V_0[2] 0.20 -> 0.25 | truth, GNSS, prior | low | instance no longer M2_TRAJ_V1; nothing pins the trajectory literals |
| fx-rotvec0 | ROT_VEC_0[2] 0.30 -> 0.35 | truth, GNSS, prior | low | same |
| lf-decode-v-mul | state_of_z: v / s -> v * s | decode at scale -2.5, 0.3 | **medium** | correct only for s = ±1. M4 rounding from an SDP eigenvector of arbitrary scale returns wrong velocities. The C01 round trip checks only R and b; C23 uses s = -1 |
| lf-decode-p-mul | same for p | decode | **medium** | same, for positions |
| lf-project-nodet | project_so3 without the det fix | returns reflections | **medium** | `project=True` is never tested. M4 rounding of a non-tight relaxation can return det = -1 |
| co-whiten-W-not-sqrt | whitened_residual uses W instead of W^1/2 | norm^2 != cost | **medium** | C30's rank does not change under per-term positive rescaling, so the only test using this function cannot see the defect. A local refinement or certificate step in M4 using it would minimise the wrong cost |
| co-whiten-nonsym-root | U Lambda^1/2 r (U^T dropped) | norm^2 != cost | low (medium if M4 uses it) | same blindness |
| cn-P1-RA1-swap | P1 built from R R^T, RA1 from R^T R | P1/RA1 contents exchanged | **medium** for ablations | the union and the primary feasible set are unchanged, so every test passes, but the primary-only SDP relaxation changes and so do the Rai-style "primary vs +redundant" ablation numbers |
| cn-PRIMARY-incl-RA1 | PRIMARY gains RA1 | constants | low | tests always list family names literally, so CN.PRIMARY / REDUNDANT are untested. An M4 that uses them would mislabel an ablation |
| lf-s-param-ignored | z_of_state ignores `s` | z_of_state(s = -2) | low | the parameter is never exercised |
| lf-bias-indices | bias_indices drops the last block | helper | low | unused and untested |
| cn-family-max-signed | family_max drops abs() | helper | low | untested helper; a feasibility report built on it would understate violations |
| gb-from-bias-swap | from_gtsam_bias returns (accel, gyro) | converter | low | the documented boundary converter (ordering footgun 1) is untested, and src currently never calls it |
| co-whiten-noclip | sqrt(w) without clip | none on this instance | low | equivalent for the instance's PD weights, but `W_vp` is caller-supplied: a PSD-singular weight gives eigh values of order -1e-10 and the output is NaN. No claim domain restricts W_vp to PD |

### Equivalent (18), with proofs
* **fm-sym-identity** (forms.sym returns Q unsymmetrised). Every quadratic tensor reaches the families only
  through `_mk`, which applies 0.5(Q + Q^T). sym is linear and commutes with `hat`, `cvec`, sums and differences,
  so the final symmetric matrix is mathematically identical. The cost uses only linear forms. Oracle clean, and the
  exact-equality C15/C17 tests pass.
* **fm-matmul-nm-swap**, **fm-dot-transposed**. These produce X^T where X was produced, followed by sym:
  0.5(X^T + X) = 0.5(X + X^T) bitwise, since IEEE addition is commutative.
* **cn-mk-nosym**. Every Q passed to `_mk` is already exactly symmetric (differences of `sym` outputs mirror
  entries bitwise; P0 and the s^2 I terms are diagonal), and 0.5(q + q) = q exactly.
* **cn-RG-VP-swap**, **cn-RB2-outer-swap**. dot and outer are symmetrised: V.P = P.V and
  sym(w_a w_b^T) = sym(w_b w_a^T) bitwise.
* **cn-P3-transposed** (Y^T A - W^T s). The entry (a,b) equals the original entry (b,a), so it is the same
  set of 9 forms per block.
* **cn-P3-minus-WT** (A^T Y + W^T s) and **cn-RG-W-not-T** (Y^T V + W c^v). `hat` builds W = -eps.w, so W^T = -W
  exactly (negation is exact). The forms are identical.
* **cn-vech-tril** (A.T[triu]). A is exactly symmetric (asserted by C17), so A.T[triu] = A[triu].
* **cn-residual-plus-b** (-\|b\|). Every b is in {0, 1} by construction.
* **co-gnss-unsorted**. The term set is identical and only the order of a commutative sum changes (ulp-level in
  Q). Terms are looked up by name.
* **fx-so3exp-branch** (series threshold 1e-8 -> 1e-12). For the inputs used (\|ROT_VEC_0\| = 0.37,
  \|w T\| >= 0.2) the branch is never taken. For 1e-12 <= theta < 1e-8 the Rodrigues coefficient
  (1 - cos)/theta^2 loses accuracy, but its weight theta^2 bounds the absolute error by 0.5 theta^2 < 5e-17.
* **gb-assert-removed**. The assertion only fires for non-z-up gravity, and every claim's domain fixes
  MakeSharedU(9.81) with G_W = (0, 0, -9.81).
* **gb-fd-forward**, **gb-fd-step** (h = 0.1). predict applies Ubar Exp(J delta) (GTSAM doc Eq. (36)), so
  Log(Ubar^-1 U(b)) = J(b - bbar) exactly for \|J delta\| < pi. Any FD step and stencil then returns J up to
  round-off. I measured linearity residual 3.7e-12 at \|delta\| ~ 0.1, and oracle J agrees within 2e-11.
* **gb-cov-transpose**, **in-S2-offdiag-swap**. preintMeasCov is symmetric in exact arithmetic (it is
  propagated as A S A^T + B Q B^T). The measured floating-point asymmetry is <= 2.2e-23 absolute on entries
  of about 1e-8 (about 1e-15 relative), so taus, G2 and W_vp move by about 1e-15 relative, below every
  tolerance (oracle 1e-12).

## 7. Tests that cannot fail or compare code to itself (task 6)

1. **test_c23_global_sign_flip.** `c.residual(-z) == c.residual(z)` and `(-z)Q(-z) - zQz == 0` hold bitwise
   for any matrix, because IEEE negation is exact. The decoder part passes for any decoder that is
   sign-homogeneous, including the surviving `v * s` mutant. In effect it can fail only if the decoder does not
   scale at all.
2. **test_c17 homogeneity half.** `lam^2 z^T A z` vs `(lam z)^T A (lam z)` is an identity for every A. Only
   the symmetry and single-b assertions can fail.
3. **test_c31.** The iso/aniso coefficient checks use only test-side reference code. The `qz` part asserts
   that the lam^2 coefficient of z^T Q z is R_0-independent and equals d^T Q[24:27, 24:27] d. Both hold
   identically for ANY Q, by the very argument of the claim, and the second compares src Q with itself. It
   cannot detect any defect in src.
4. **test_c32, test_c05, test_c06** never touch src. They are legitimate mathematical claim tests but
   contribute nothing to mutation coverage. test_c28 touches src only through a J equality already checked in
   the instance test.
5. **test_c30** (rank 45) is invariant to any positive per-term rescaling of the whitening, so it cannot see
   the defects of `whitened_residual` (survivors co-whiten-*).
6. **Shared-literal reference.** tests/test_qcqp.py imports B_G, B_A, OMEGA_BODY, F_BODY, ROT_VEC_0, P_0, V_0,
   SIGMA_PRIOR, SIGMA_GNSS, SIGMA_BG_RW and SIGMA_BA_RW from src. Its "independent" reference therefore moves
   with any change to those literals; 5 fixture survivors show this. C39 pins only the IMU noise model and the
   rw diagonal.
7. **test_c01 decoder round trip** checks only R and b. v and p decoding is never verified at \|s\| != 1.
8. `project=True`, `z_of_state(s=...)`, CN.PRIMARY/REDUNDANT, `family_residuals`/`family_max`,
   `bias_indices` and `from_gtsam_bias` have no test.

## 8. Findings that could undermine M4's SDP certification

* **Q scaling.** max\|Q\| = 1.108e10 (bias random walk, gyro 5.54e9), while GNSS weights are 400. The dense
  z^T Q z at truth evaluates to 1.4e-8 against a true 2e-19. Duality-gap or certificate thresholds computed from
  dense Q will be dominated by cancellation. Rescale Q, or evaluate primal cost in factored form.
* **Redundant equalities.** The full set has 472 constraints of rank 453 (490 of rank 471 aniso), so there are
  19 exact linear dependencies (3 are the C16 traces). Interior-point solvers (CLARABEL/SCS) can misbehave on
  rank-deficient equality systems. Prune to a basis or check solver status carefully.
* **Surrogate gap at bbar != b_true.** At bbar = 0 the QCQP cost at truth is 0.57% off GTSAM's (Cayley /
  first-order correction, C27/C28). QCQP optima should not be compared with GTSAM optima at tighter than about
  1e-2 relative unless bbar is close to the true bias.
* **Rounding path untested.** The decoder's v/p scaling and `project_so3`'s det fix are untested
  (medium survivors). If M4 extracts the state from a rank-1 SDP solution through `state_of_z`, that path is
  unverified.
* **Ablation labels untested.** P1-vs-RA1 content and the PRIMARY/REDUNDANT constants are not tested, so
  primary-only vs +redundant tightness results could be silently mislabelled.
* **No spurious points.** The primary QCQP feasible set is exactly the lifted states up to the global sign.
  Tightness is the open question for M4, not correctness of the variety.
* M3-C40, if it ever enters the yaml, is false and must not be used to justify extra redundant rows.

## 9. Full suite

Baseline on the clean tree (d2371af): `pytest tests/` gives 108 passed in 3582 s. The post-commit run is
recorded below.
