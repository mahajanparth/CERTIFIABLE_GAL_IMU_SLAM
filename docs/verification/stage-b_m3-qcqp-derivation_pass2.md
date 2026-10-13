# Stage B verification, pass 2: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 2. Base commit 2b625b0. src/qcqp and docs/claims/M3.yaml are byte-identical
to pass 1 (d2371af). Only tests/test_qcqp.py (474ef76) and tests/test_mutation_m3.py (2b625b0) changed.

Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py, docs/claims/M3.yaml (M3-C01..C35, C39), and
the brief-only claim M3-C41. I did not read docs/03_qcqp_derivation.md, experiments/m3_scratch/, or any seeds file.
All verifier scripts are outside the worktree, in the session scratchpad `p2/`:

* `fresh_ref.py` is a new independent reference. It imports neither src, tests, nor pass-1 `ind.py`. It contains
  its own PIMs; J from a 4-point stencil on the scipy logm chart; the truth from a Van Loan block-expm closed
  form; z by plain concatenation in claim order; P1/RA1/P2/RS as sympy polynomials converted to matrices; its own
  GTSAM factor graph (Diagonal.Precisions); and its own decoder with SVD projection.
* `fresh_cmp.py` compares that reference with src. `c41.py` tests the brief claim. `fdstep.py` measures one
  equivalence.
* `dump.py` / `diff.py` record about 190 observable output groups of src/qcqp (every public function, constant
  and dataclass field, on fixed inputs including singular / reflected / negative-s decoder inputs and a singular PSD
  weight) and diff a mutant against the clean tree. This gives the measured effect behind each equivalence ruling.
* `harness2.py` / `mutants2.py` / `second.py` run the mutation sweep. Each mutant is a real file edit on a scratch
  copy of src + tests/test_qcqp.py. The judge is the full tests/test_qcqp.py (slow tests included, `-x`). The
  pass-1 oracle (`v/oracle.py`) and `diff.py` also run on every mutant.

Throughout, the worktree had no changes apart from this file.

## Verdict summary

* **F03 (pass 1): FIXED.** All 17 pass-1 survivors, replayed as real file edits, are killed by tests/test_qcqp.py.
  For the 5 fixture-literal mutants the first killer is a pin against retyped domain literals, so I deselected it.
  A second, reference-based test still kills each one. The vacuous tests named in pass-1 Sec. 7 items 1, 2, 5, 6,
  7 and 8 were rebound. Two residual cannot-fail assertions remain (Sec. 5).
* **Claims: 36 CONFIRMED (C01-C35, C39), M3-C41 REFUTED, 0 UNTESTABLE.**
* **Fresh sweep: 99 new mutants over all 7 modules plus 23 replayed earlier mutants (122 real-file-edit runs).**
  New: 83 killed, 16 survive. Replay: 17 killed, 6 survive. Of all 22 survivors, **10 are equivalent** (measured
  or proved) and **12 are NON-EQUIVALENT (2 medium, 10 low)**.
* **GATE (zero surviving non-equivalent mutants, medium or low): FAILED.** The 12 survivors sit in parts of
  src/qcqp that tests/test_qcqp.py still does not check: the decoder's projection when s < 0 or a block is
  singular (both on the M4 rounding path), `truth_state` at bbar != b_true, `Layout.bg`/`Layout.ba`,
  `Instance.n_kf`, `GAL_ROWS`, `family_residuals`, and the forms helpers (`sym`, `outer`, `cmat_left`, `quad_T`).
  Every mutant that changed z, Q, the families, the instance data or the GTSAM boundary was killed.

## 1. Finding M3-F03 ruling: FIXED

Replay of the 17 pass-1 survivors (same substitutions as pass 1, applied as real file edits, full tests/test_qcqp.py):

| mutant | first killing test | measured failure |
|---|---|---|
| fx-sigma-prior | instance test (LIT pin); with it deselected, test_c24 | 2.34e6 abs on cost 2.0e9 (tol 1e-12 rel) |
| fx-sigma-gnss | instance test; deselected: test_c24 | 1577 abs on 2.0e9 |
| fx-bg-true | instance test; deselected: test_c04 (GTSAM FD J) | 4.2e-4 vs 1e-8 |
| fx-v0 | instance test (M2_REF trajectory); deselected: test_c08 | truth cost 2501 vs 1e-15 |
| fx-rotvec0 | instance test; deselected: test_c08 | 2490 vs 1e-15 |
| lf-decode-v-mul / -p-mul | test_c01 (scale -2.5) | 5.06 / 6.34 vs 4.5e-16 |
| lf-project-nodet | test_decoder_projection_is_nearest_rotation | det = -1 |
| co-whiten-W-not-sqrt / -nonsym-root / -noclip | test_whitened_residual_is_symmetric_root_whitening | 1.3e9 / 1.6e4 vs 1.7e-5; NaN |
| cn-P1-RA1-swap | test_P1_is_column_and_RA1_is_row_orthonormality | 1.0 vs 0 |
| cn-PRIMARY-incl-RA1 | test_c14 (labels per C09/C10/C11) | tuple mismatch |
| lf-s-param-ignored | test_c01 (z(s=-2) == -2 z) | array mismatch |
| lf-bias-indices | test_c01 | arange mismatch |
| cn-family-max-signed | test_c13 (own abs max) | 4.3e-4 vs 1.9e-12 |
| gb-from-bias-swap | test_bias_boundary_converters | order mismatch |

Each kill comes from a check against an independent reference: retyped domain literals or M2 reference values,
GTSAM, a test-side 5x5 cost, or the test's own SVD / eigh evaluation. None is a snapshot of src output. The same
mutants also appear in tests/test_mutation_m3.py with the correct killers.

## 2. Independent re-checks (no src imports in the reference)

| check | result |
|---|---|
| instance vs fresh PIMs | dR, dv, tau bitwise equal; J 5.4e-13 / 7.3e-13 (FD chart); truth (Van Loan closed form) 3.3e-15 |
| **index map**: src z_of_state vs concatenation in claim order, 30 random states, base + aniso, both bbar | 9.6e-14 / 1.3e-13 relative (J-limited) |
| decoder src vs fresh at s in {1, -1, -3.7, 0.05}, all blocks | <= 1e-12 (all 60 states x 4 scales) |
| **family P2** (sympy cross products to matrices, 27 forms) | 0 unmatched, exact (bitwise) |
| **families P1 and RA1** (sympy, 18 + 18) | 0 unmatched, exact; P1 is R^T R and RA1 is R R^T |
| **family RS** (sympy with instance dR, J; 36 forms) | worst 1.1e-13 on entries of size 0.5 (J chart) |
| **cost vs own GTSAM graph**, bbar = b_true, 30 perturbed states | factored/GTSAM - 1: 8.5e-5, 3.0e-6, 2.4e-7 at eps 1e-2, 1e-3, 1e-4 (C26 bound 0.05 eps holds) |
| same, dense z^T Q z | identical ratios; at truth the dense value is 2.3e-8 against factored 4.3e-21 and GTSAM 6.2e-22 (cancellation) |
| bbar = 0 | truth 3.2968e-3 vs GTSAM 3.3157e-3 (0.57%); ratios 1.1e-3 to 1.5e-3, not shrinking (outside C26's domain) |
| **M4 rounding path**: X = z z^T + noise N N^T, top eigenvector x sqrt(lambda), sign +/-, state_of_z(project=True) | src = fresh decoder to 7.8e-16; det = 1; error vs truth 1.6e-15 / 9.2e-6 / 9.7e-3 at noise 0 / 1e-6 / 1e-3 |
| projection of reflection, 0, rank-1, -I blocks at s = +/-1 (unmutated src) | det 1 in all cases, equal to fresh |

## 3. Claims

src and claims are unchanged since pass 1, so I re-ran the pass-1 per-claim scripts (`v/claims1,2,3,5.py`; C32
with the claim's own seed 2) and added the fresh checks above. Every number reproduced:

| claims | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | Sec. 2 index map, 124 / 136, s at 123 |
| C02, C03 | CONFIRMED | 112 / 12; all 7 Rai triples give n - direct = 39 |
| C04 | CONFIRMED | J[0:3,3:6] = 0, J[9] = 0 exactly; \|J_nu,a - J_R\| <= 3.8e-12 (fixture), 3.5e-11 (time-varying) |
| C05, C06 | CONFIRMED | 2.2e-15 / 2.9e-15; arbitrary X_j min max\|r\| 0.94 |
| C07, C08, C24, C25 | CONFIRMED | src forms and Q = independent (pass-1 oracle clean); C08 perturbation kills; Q PSD |
| C09-C12 | CONFIRMED | truth <= 4.9e-16; random P1 1.6e-13, P4 1.3e-12, RR 1.5e-11 (tolerances 1e-12 / 1e-10) |
| C13 | CONFIRMED | min family max 4.2e-6 (P0) / 1.5e-5 (RB1) > 1e-7 |
| C14, C16 | CONFIRMED | sizes as stated; joint rank 453 (gap 0.18 -> 8.7e-16), aniso 471; P1 u RA1 rank 11 per KF, trace identity 2.7e-15 |
| C15, C17, C19, C23 | CONFIRMED | proofs in pass 1 Sec. 5; tests pass |
| C18 | CONFIRMED | pass-1 proof and 301 solves; C18 test passes |
| C20, C21, C22 | CONFIRMED | P2 2.0, RH 8.5e-2, RS 1.5e-2; P1 = RA1 = 1.0; rotated/negated violated >= 3.9e-3, satisfied <= 2.2e-16 |
| C26 | CONFIRMED | Sec. 2 (own graph) |
| C27, C28, C29 | CONFIRMED | dev 2.01e-2 / -1.22e-2 / -2.10e-3, halving 0.250/0.249/0.249, 0.501/0.500/0.500, 0.504/0.502/0.501; factors 8.02/8.01/8.00; 1.8e-7 vs 0.78 |
| C30 | CONFIRMED | rank 45 (cond 3.9e7 / 4.5e7) |
| C31, C33, C34 | CONFIRMED | 4.020000 const (ptp 5e-13), diag(1,2,5) range [1.347, 6.666]; C33 1.3e-4 / 2.8e-6 / 2.3e-7; C34 body 1.9e-3 / 1.7e-4 / 3.1e-5, world 7.0e-2 / 5.7e-2 / 6.6e-2 |
| C32, C35 | CONFIRMED | seed 2 rerun: see Sec. 3a; pass 1 seeds 4/99 2.7e-14 vs 8.1e-2; null 263, span 236 |
| C39 | CONFIRMED | taus bitwise equal to fresh PIMs; Dt 0.5000000000000003 |
| **C41** | **REFUTED** | below |

**M3-C41 (brief only): REFUTED.** I built P1 and RA1 as sympy polynomials; they match src bitwise (Sec. 2). Per
keyframe, the rank of the 6 P1 vech vectors is 6, and P1 + RA1 together have rank **11**: adding RA1 raises the
rank by **5, not 0**. Over all keyframes the stacked rank goes from 18 to 33. The relative distances of the 6
RA1 forms from span(P1) are 0.71, 1, 1, 0.71, 1, 0.71, and the singular values of the stacked 12 rows fall from
0.866 straight to 0. A Shor witness shows the difference: X = (1/3) sum_i z_i z_i^T with R_i = sqrt(3) e_1 e_i^T
and s = 1. X is PSD (min eig -1.1e-16) and satisfies P0, and every P1 trace is exactly 0. The RA1 traces are
(2, 0, 0, -1, 0, -1). So RA1 cuts this point away from the P0+P1 relaxation. P1 and RA1 are equivalent on the
variety (real square R) but not in the Shor relaxation. The only shared dependency is C16's trace identity.

### 3a. C32 at the claim's own seed
(filled in below when the run finishes)

## 4. Mutation sweep

### 4.1 New mutants (99; ids, modules and substitutions in `p2/mutants2.py`, none in pass 1 or tests/test_mutation_m3.py)

By module: fixture 15, gtsam_boundary 13, instance 12, lifting 16, forms 9, cost 15, constraints 19.

**Killed (83).** Each first killer is listed with its measured failure. Each killer is a check against GTSAM,
the test-side 5x5 reference, retyped literals or an own evaluation.
* instance test, against the independent GTSAM reference: fx-om1-literal (4.9e-3), fx-fbody0-literal, fx-ba-true,
  fx-dt, fx-steps (Dt 2.5e-3), fx-tedge, fx-gravity, fx-p0, fx-sigma-bg-rw, fx-sigma-g (tau_R), fx-m1-sign,
  fx-truth-dv-unrotated, fx-so3exp-sin, fx-hat-nonskew, fx-streams-no-gyro-bias, gb-gal3-pv-swap,
  gb-nav-to-gal3-swap, gb-increment-compose-order, gb-increment-no-inverse, gb-accel-cov-uses-gyro,
  gb-intcov-not-squared, gb-G-vel-sign, gb-fd-one-sided-half, gb-cov-diag-only, gb-fd-sign, gb-params-fixed-g,
  gb-integrate-half-dt, in-tau-v-from-pos, in-G2-not-inverted, in-WR-not-inverted, in-Svp-offdiag-transposed
  (6.99e4 vs 2.4e-2), in-Dt-nominal (3.3e-16 vs exact C39 Dt), in-gnss-velocity, in-zero-key-half-bias.
* test_c01 (claim index map vs lift_ref): in-JR-from-accel-block (J_nu,a used for J_R, difference 3.8e-12 > 1e-14),
  in-Jnu-rows, lf-Yp-offset, lf-rho_p-offset, lf-s-form-sign, lf-lift-cp-uses-Jnu, lf-rho_p-uses-rv,
  lf-z-s-entry-scaled, lf-decode-R-rowmajor, lf-vec-form-sign, fm-cmat-right-transposed, co-rv-dv-uses-M,
  co-rp-dp-uses-dv, co-rv-vi-sign, co-rv-cbar-full.
* test_decoder_projection: lf-project-transposed, lf-project-middle-axis.
* test_c22: lf-copy-self. test_c33: in-D-sign, co-partial-D-left (15.2 vs 1.3e-3).
* test_c05_shipped: co-rR-M-gamma-sign, co-rR-Y-twice. test_c08: co-prior-R0-sign, co-rw-sum, co-factored-no-W.
  test_c23 (factored vs J_direct): co-rw-range, co-gnss-prior-sigma.
* test_c09: fm-hat-sign, fm-cvec-transposed, fm-linT-identity, fm-eps-anticyclic, cn-P0-b-sign, cn-CYCLIC-anti,
  cn-P3-V-uses-cp, cn-iside-V-P-swap, cn-edge-w-uses-ba. test_c10: cn-RB2-ww-sign, cn-RB1-uses-cv, cn-RC1-V-uses-cp.
  test_c11: cn-RL-V-sign, cn-RG-YtP-uses-cv, cn-RS-hat-sign. test_c12: cn-P4-rp-uses-rv, cn-RR-p-uses-rv.
  test_c14: cn-RH-drop-axis2, cn-vech-strict-upper. test_P1_RA1: cn-UPPER-strict. test_c17: cn-residual-negated.
  test_c18: cn-flat-names-ignored.

### 4.2 Replay of earlier mutants (23, real file edits)
The 17 F03 survivors (Sec. 1, all killed) plus 6 pass-1 equivalents: fx-so3exp-branch, gb-fd-step,
in-S2-offdiag-swap, fm-sym-identity, cn-mk-nosym, cn-vech-tril. All 6 survive. Rulings below. **fm-sym-identity
is re-ruled NON-EQUIVALENT.** Pass 1's proof covered only the downstream families; the forms functions themselves
return different tensors.

### 4.3 COMPLETE survivor list (22) with rulings

Equivalence threshold: 1/10 of the tightest claim tolerance for the quantity, never below the 8-ulp rounding
floor. At or below that floor a mutant is equivalent only if it is no worse than the clean code.

| # | mutant | measured effect (diff.py vs clean) | ruling | class |
|---|---|---|---|---|
| 1 | **lf-project-before-divide** (`project_so3(x * s)`) | decoded R wrong by up to 1.97 (rel 2.0) for s = -1 and s = -2.5 with project=True; identical for s > 0 | **NON-EQUIVALENT** | **medium**: M4 rounds an eigenvector whose sign is arbitrary. With s < 0 and project=True it returns the wrong rotation. The projection test uses s = 1 only |
| 2 | **lf-truth-state-bbar** (`truth_state` uses bbar for b) | truth_state("zero").b = 0 instead of b_true (abs 0.09) | **NON-EQUIVALENT** | **medium**: M4 truth comparisons / truth cost at bbar = 0 would use bias 0. The test checks only v, p of truth_state |
| 3 | lf-project-sign-detA (`sign(det A)`) | zero block -> diag(1,1,0); rank-1 block -> non-rotation (abs 1.0); identical for nonsingular | NON-EQUIVALENT | low (M4 rounding of a degenerate R block, e.g. the C21 point) |
| 4 | cn-family-residuals-abs | family_residuals returns \|g\| (rel 2.0) | NON-EQUIVALENT | low: untested helper |
| 5 | lf-bg-slice | Layout.bg(k) = [48+6k, 51+6k) | NON-EQUIVALENT | low: public, unused, untested |
| 6 | lf-ba-slice | Layout.ba(k) = [45+6k, 48+6k) | NON-EQUIVALENT | low: same |
| 7 | in-n_kf-edges | Instance.n_kf = 2 | NON-EQUIVALENT | low: public, unused, untested |
| 8 | gb-GAL_ROWS-pos-vel | GAL_ROWS vel/pos exchanged | NON-EQUIVALENT | low: the documented Gal3 row map (ordering footgun 3) is unused and untested |
| 9 | fm-outer-nosym | forms.outer unsymmetrised (abs 2.1); every family bitwise identical (cn.* IDENTICAL) | NON-EQUIVALENT | low: breaks the module's documented symmetric contract; no direct forms test |
| 10 | fm-sym-identity (replay) | sym/outer/matmul/matvec/dot/cross unsymmetrised (abs up to 4.3); families identical | NON-EQUIVALENT (re-ruled) | low: same |
| 11 | fm-cmat-left-transposed | cmat_left returns C^T A (abs 7.3) | NON-EQUIVALENT | low: unused helper |
| 12 | fm-quadT-identity | quad_T returns the input (shape (3,2,..) vs (2,3,..)) | NON-EQUIVALENT | low: unused helper |
| 13 | fm-matvec-nm-swap | bitwise IDENTICAL on every output | EQUIVALENT | proof: sym(X^T) = 0.5(X^T + X) = sym(X) bitwise (IEEE + is commutative) |
| 14 | in-bbar-not-copied | bitwise IDENTICAL | EQUIVALENT | proof: only aliasing changes; src never writes into bbar in place (no in-place op on it anywhere), so every output is equal |
| 15 | co-aniso-Wvp-T | whitened residual / Q aniso rel <= 8.7e-17 | EQUIVALENT | W_vp is symmetric to rounding; the effect is below the 8-ulp floor (1.8e-15) and both W and W^T are inv(Svp) to the same accuracy |
| 16 | co-assemble-WT | Q partial rel 3.4e-19 (vs max\|Q\|) | EQUIVALENT | below rounding floor; same accuracy |
| 17 | co-whiten-clip-floor (clip at 1e-12) | instance weights: IDENTICAL; caller-supplied singular W: entry diff 2.3e-6, rel 9.8e-12 of the vector, \|wr\|^2 changes by ~5e-12 on ~1e8 (rel 5e-20) | EQUIVALENT | outside every claim domain (all instance weights >= 400) and 1/10 of C24's 1e-12 cost tolerance is not reached |
| 18 | fx-so3exp-branch (replay) | only so3_exp(\|w\| = 1e-9 probe): 5.9e-19 abs on a unit-scale matrix; every instance output identical (the branch is never reached: \|ROT_VEC_0\| = 0.37, \|wT\| >= 0.2) | EQUIVALENT | below the floor (2.2e-15) on the claim domain |
| 19 | gb-fd-step h = 0.1 (replay) | J changes 2.2e-13 to 6.1e-13. Against an exact least-squares fit of the linear map (residual 2e-15): **clean** error 2.3e-13 to 6.1e-13, **mutant** 6.7e-15 to 9.6e-15 | EQUIVALENT | the mutant is about 40x more accurate (no worse than clean). Pass-1 proof (predict is exactly linear) confirmed |
| 20 | in-S2-offdiag-swap (replay) | bitwise IDENTICAL | EQUIVALENT | preintMeasCov blocks are bitwise symmetric here |
| 21 | cn-mk-nosym (replay) | bitwise IDENTICAL | EQUIVALENT | pass-1 proof |
| 22 | cn-vech-tril (replay) | identical on every constraint matrix; differs only on my non-symmetric probe | EQUIVALENT | vech is documented for symmetric matrices; every A_k is exactly symmetric (C17) |

Totals: survivors 22. Equivalent 10 (#13-#22). **Non-equivalent 12: 2 medium (#1, #2), 10 low (#3-#12).**

## 5. Tests that cannot fail or compare code with itself (after 474ef76)

1. **test_c31, `np.ptp(qm) <= 1e-6 mean`.** For any Q, the lam^2 coefficient of z^T Q z along v_1 + lam d is
   e_d^T Q e_d, since v_1 enters z only in its own slot. So it is R_0-independent for every src Q and cannot
   fail. The new qm-vs-reference half can fail (it is a C24 check at 600 states). The iso/aniso halves are test-side
   mathematics and see no src.
2. **test_c23 at -z.** Constraint residuals at -z and factored cost at -z are bitwise equal to those at z. So these
   asserts fail exactly when the C09-C12 / C24 asserts at z fail: redundant, not independent. The decoder part is
   real (scale -1).
3. test_c28, test_c03, and the original test_c05 / test_c06 still touch no src. They are legitimate claim tests
   with no mutation coverage. Shipped twins now exist for C05 and C06.
4. Untested surface (all of it shows up as survivors): `state_of_z(project=True)` at s <= 0 and on singular blocks,
   `truth_state(...).b`, `Layout.bg/ba`, `Instance.n_kf`, `GAL_ROWS`, `family_residuals`, and
   `forms.sym/outer/cmat_left/quad_T` directly.

The pass-1 items rebound by 474ef76 (C17 identity, C23 bitwise identities, C31 Q-vs-Q, shared literals, decoder
v/p at \|s\| != 1, whitening, labels, helpers) now fail on the corresponding mutants (Sec. 1).

## 6. Risks for M4

* **Rounding path:** `state_of_z(..., project=True)` is correct as shipped (Sec. 2, including s = -1 and singular
  blocks), but its behaviour at s < 0 and for degenerate blocks is unguarded by tests (#1, #3). The decoder has
  no guard for s ~ 0: it divides by s unconditionally, and with a noisy SDP the s-entry can be small. It ignores
  the liftings and does not check consistency, so M4 should report the lifting / constraint residual of the
  rounded point itself.
* **truth_state at bbar = 0** is correct now but not pinned for b (#2).
* **Dense Q cancellation:** z^T Q z at truth is 2.3e-8 against a true 4e-21 (max\|Q\| 1.1e10). Use the factored
  cost for primal values and gaps, or rescale Q.
* **Surrogate gap at bbar = 0:** 0.57% vs GTSAM at truth, not shrinking with eps.
* **Redundant constraints are linearly dependent:** 472 rows of rank 453 (aniso 490 of rank 471).
* **C41 is false:** RA1 is not redundant in the Shor relaxation (rank +5 per KF; explicit PSD witness violating
  RA1 by 2 with P0, P1 exact). Primary-only and +RA1 relaxations can differ. Do not drop RA1 on C41's grounds.
* **J accuracy:** the shipped h = 1e-3 central difference has an error of 2-6e-13. h = 0.1 gives 7e-15, since the
  map is exactly linear. This is not a defect under any claim tolerance, but it sets the floor on z (V, P to
  about 1e-13).

## 7. Full suite
(filled in below)
