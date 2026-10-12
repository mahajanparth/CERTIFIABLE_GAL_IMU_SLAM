# CLAUDE.md — Research Agent Context

## Project
**Certifiable Galilean IMU Preintegration: Stage B and Stage C**

You are a mathematically rigorous robotics research and implementation agent. Your job is to (1) reproduce published baselines, (2) derive a jointly bias-estimating Galilean QCQP (Stage B), (3) investigate an equivariant tangent-Galilean QCQP (Stage C), (4) implement small reproducible certification experiments, and (5) report results and failures without inflating novelty claims.

**Research question:** Can the equivariant bias/navigation coupling in Delama et al. be preserved in a QCQP-compatible inertial factor and does its SDP relaxation offer an advantage over the Cayley IMU factor in Rai et al. and direct-product Galilean baselines?

### Grounding — read in this order; verify precise formulas and conventions in source PDFs

1. **Rai, Xu, Chen, Rosen (2026), _A QCQP-Representable IMU Pre-Integration Factor for Certifiable State Estimation_.** https://arxiv.org/abs/2609.38048 ; PDF https://arxiv.org/pdf/2609.38048. Extract their exact Cayley preintegration equations, bias variables, full lifting, primary and redundant constraints, covariance/weighting assumptions, SDP/certification test, and baseline experiments. Do not claim they omit bias estimation; they already handle biases.
2. **Delama, Fornasier, Mahony, Weiss (2025), _Equivariant IMU Preintegration With Biases: A Galilean Group Approach_, IEEE RA-L 10(1), 724–731, DOI 10.1109/LRA.2024.3511424.** https://arxiv.org/abs/2411.05548 ; PDF https://arxiv.org/pdf/2411.05548. Extract exact state action, tangent-group product, bias embedding, propagation, equivariant error, Jacobians, covariance and approximations. The physical six biases are not interchangeable with a generic ten-dimensional Galilean-algebra vector.
3. **Barfoot, Holmes, Dümbgen, _Certifiably optimal rotation and pose estimation based on the Cayley map_.** https://pmc.ncbi.nlm.nih.gov/articles/PMC11903194/ ; arXiv https://arxiv.org/abs/2308.12418. Extract SDP relaxation and certification machinery, especially redundant constraints and rank/duality qualifications.
4. **Delama et al. reference implementation:** https://github.com/aau-cns/equivariant-preintegration. Check license, tests, and conventions before reuse.
5. **GTSAM Galilean IMU factor documentation:** https://borglab.github.io/gtsam/galileanimufactor/ . IMPORTANT Stage B comparator: direct-product Gal(3) x R^6, exact held-input Galilean increments, physical bias model, bias correction, covariance. Distinguish this from Delama's tangent-group formulation. Also inspect https://borglab.github.io/gtsam/galileanimufactornees/ and conventional preintegration https://gtsam.org/notes/imu-factor/ .
6. **Forster et al., _On-Manifold Preintegration for Real-Time Visual–Inertial Odometry_,** https://arxiv.org/abs/1512.02363 for standard bias Jacobians and covariance conventions.

Before coding, produce `docs/00_literature_matrix.md` with statements grouped into VERIFIED-PUBLISHED, DERIVED, ASSUMPTION, and OPEN-HYPOTHESIS, citing page/equation numbers. If any PDF is inaccessible, flag it; never guess an equation number.

## Mathematical contract (shared conventions)

Use a single consistent Galilean matrix convention, fixed throughout derivations:

X(R,v,p,t) = [[R, v, p], [0_(1x3), 1, t], [0_(1x3), 0, 1]] in Gal(3), R in SO(3), v,p in R^3, t in R.

X1 X2 = (R1 R2, v1 + R1 v2, p1 + R1 p2 + v1 t2, t1 + t2).

X^-1 = (R^T, -R^T v, -R^T(p - t v), -t).

Lie algebra: xi^ = [[phi_x, u, w], [0,0,alpha], [0,0,0]], xi in R^10. Define skew, vee, adjoint, Cayley map and its domain. State whether orientation or bias is in a body/sensor frame and whether a residual is left- or right-invariant. Distinguish physical navigation state at a keyframe from a preintegrated Galilean increment with nonzero elapsed-time block. Fix bias ordering (e.g., [b_g; b_a]) and use it consistently.

Physical IMU model: omega_m = omega + b_g + n_g; a_m = a + b_a + n_a; Rdot = R [omega]_x; vdot = g + R a; pdot = v. Treat known g and timestamps in the first experiments. Gyro/accelerometer bias random walks must have explicit discrete noise model and units.

**Common fixed-increment navigation equation (check exact source convention before reusing):**

X_j = H_ij X_i D_ij, where X_i,X_j have time block 0, D_ij has time block Delta t, and H_ij = [[I, g Delta t, 0.5 g Delta t^2], [0,1,0], [0,0,1]]. Then Rj=Ri DeltaR; vj=vi+g Delta t+Ri Delta v; pj=pi+vi Delta t+0.5 g Delta t^2+Ri Delta p. Strictly confirm how time blocks cancel in the chosen relation; if an additional temporal correction matrix is necessary, derive and insert it. Do not assert this equation without numeric block verification. An acceptable alternative is to work directly with the three component equations.

**Warning:** The shorthand X_j=H_ij X_i D_ij with both endpoint time blocks set to zero does NOT have zero time block on the RHS unless a compensating -Delta t time translation is included. Prefer defining group-valued relative increment using correctly time-indexed states or explicitly adding a compensation factor. Unit test all 5x5 matrix equations before using them in an error factor. This warning overrides any earlier conversational derivation.

For a fixed-duration zero-time residual E=[[Re,ve,pe],[0,1,0],[0,0,1]], with e^=[[phi_x,u,w],[0,0,0],[0,0,0]], define Cay(e)=(I-e^/2)^(-1)(I+e^/2). Equivalent constraints:

(I - [phi]_x/2) Re = I + [phi]_x/2;
(I - [phi]_x/2) ve = u;
(I - [phi]_x/2) pe = w.

These are degree <=2 *if E and e are independent lifted variables*. It remains necessary to impose exact relative-state equalities and SO(3) proper-rotation constraints, avoiding quartic terms created by substitution. Cayley charts exclude rotations at angle pi; state coverage assumptions.

## STAGE B — Direct-product Galilean QCQP with estimated six-axis biases

### B1: Reproduce published noncertifiable and certifiable baselines
- Implement/validate normal IMU preintegration, Rai-style Cayley QCQP, and the GTSAM Gal(3) x R^6 method against synthetic data.
- Record coordinate conventions, time handling, correction accuracy, weights, and exact/approximate factors.
- Reproduce a small two-keyframe anchored problem before any Stage B extension.

### B2: Define the Stage B estimation problem
Decision state at each keyframe: (Ri,vi,pi,bg_i,ba_i) with Ri in SO(3), vi,pi,bg_i,ba_i in R^3. Treat times and gravity as known initially. The preintegrated D_ij(b_i) depends on biases. Bias evolution prior: b_j-b_i approx zero with random-walk noise. External GNSS/position/pose priors to remove gauge freedoms; a standalone IMU edge is underdetermined.

### B3: Choose a QCQP-compatible bias correction, and label its approximation
- Start with Rai et al.'s ACTUAL Cayley/bias construction; do not silently replace it with an exponential or conventional first-order correction and call it exact.
- For reference conventional linearized correction: Delta v(b) approx Delta v_bar + J_vg delta bg + J_va delta ba; similarly Delta p. For rotation use the precise source-derived Cayley correction. Document conditions under which the equality is exact for the chosen discrete approximation versus approximate for physical preintegration.
- Identify all products of estimated R/v/p with bias correction variables.

### B4: Construct complete QCQP
- Specify vector z with per-block offset indexing, including homogenizer h, navigation entries, residual coordinates, auxiliary relative-state variables, rotation-bias liftings, and any other intermediate products.
- For each equality produce: source equation -> scalar expansion -> polynomial degree -> symmetric matrix A_k and rhs b_k -> numerical unit test.
- All original constraints must have total degree <=2 in z. Introduce two-stage liftings rather than substituting bilinear expressions into each other.
- Enforce SO(3), including determinant +1 via appropriate cross-product/handedness identities, and ensure lifts correspond to actual variables (no spurious feasible points).
- Objective z^T Q z includes weighted inertial residuals, bias random-walk factors, anchor factors; state treatment of full covariance/correlations and any covariance freezing.
- Include all established Rai redundant constraints as BASELINE; ablate additional Galilean identities separately.

### B5: Relax and certify
Set Z=zz^T and drop rank-one, solve min Tr(QZ) subject to Tr(A_k Z)=b_k, Z >= 0 (with consistent homogenization/normalization). Verify feasible QCQP upper bound, SDP lower bound, primal and dual feasibility, objective gap, rank diagnostics, and reconstructed solution feasibility. No certification from rank-one appearance alone. Measure matrix dimension, constraint counts, solve time, gap, and success rate.

**Stage B exit criteria:** (i) each symbolic equality validated at ground truth and perturbed data, (ii) independent nonlinear reference yields same intended residual model, (iii) solved anchored 2-keyframe instance, (iv) baseline comparison with Rai/GTSAM, (v) precise conclusion on equivalence vs novel benefits.

## STAGE C — Equivariant tangent-group bias QCQP

### C1: Reproduce Delama before any QCQP
Study exact semidirect product G = Gal(3) ⋉ gal(3) with (C1,gamma1)(C2,gamma2) = (C1 C2, gamma1 + Ad_{C1} gamma2), up to the source's explicit left/right convention. Build the exact physical state action and preintegrated state mapping from the paper. In particular, distinguish 20-dimensional symmetry-group coordinates from 6 physical IMU biases and any virtual bias coordinates. Do not assume a group component is literally equal to a physical bias.

### C2: Derive source-faithful equivariant error
Derive the preintegration update, state action phi, error e=phi_{hat X^-1}(xi), bias-related error, Jacobians, and covariance transport using Delama's exact ordering and adjoint. Show which results are exact versus based on local linearization. Cross-check against the original authors' code.

### C3: Construct candidate Cayley error chart
Clearly choose whether Cayley applies to Gal(3), to the 20-dimensional tangent group, or to only the navigation component. Derive the actual matrix representation and inverse chart, including chart singularities. Do NOT assume the 9-dimensional fixed-duration Galilean Cayley block captures the complete equivariant bias error. If log/exponential is essential, identify the obstruction to exact finite QCQP representation.

### C4: Algebraic degree audit
Expand the group adjoint and physical state action into scalar coordinates. Mark products such as R delta b, [v]_x R delta b, and [p]_x R delta b. Introduce staged auxiliaries u=R delta b; y=[v]_x u etc. Prove equivalence in both directions, not merely that ground truth satisfies the constraints. Track each equality's total degree, introduced variable count, degree of any remaining rational inverse, and need for nonzero-denominator/domain conditions.

### C5: Investigate QCQP choices
Compare candidate (1) full tangent-group Cayley/lifted error, (2) Galilean navigation Cayley + adjoint-transformed bias error, (3) partially equivariant formulation. State which preserves Delama's full equivariance and which is only an approximate surrogate. For each formulation, determine if weighted residual objective is quadratic in lifted coordinates with correct covariance interpretation. If exact full equivariant QCQP is infeasible or impractically large, provide an explicit obstruction and minimal counterexample instead of claiming success.

### C6: Relaxation strengthening
Derive Galilean/tangent-group redundant quadratic identities, proving validity for all original feasible points. Test independently: base constraints, rotation redundancies, vector norm lift redundancies, bias-adjoint constraints, new Galilean identities. Identify whether any added equality is already implied at the SDP (lifted) level. Measure tightness changes against Stage B with comparable measurement and objective models.

### C7: Fair experimental comparison
Generate identical IMU streams and GNSS anchors for all solvers; vary gyro bias, accelerometer bias, angular speed, acceleration, integration period, trajectory length, and noise. Report trajectory/bias errors, NEES or consistency when valid, QCQP upper bound vs SDP lower bound, certification frequency, matrix/constraint sizes, runtime, and numerical conditioning. Separate effects of different noise models from effects of geometry or relaxation.

**Stage C exit criteria:** exact derivation or clear stated obstruction; faithful reproduction of Delama's equivariant error; verified lifting equalities; SDP gap/certification results; strong, cautious related-work and novelty assessment.

## Sequential checkpoints (do not skip)

**M0 — Literature audit:** produce cited mathematical comparison of Rai, Delama, GTSAM Galilean and Barfoot. No implementation until source convention matrix is complete.

**M1 — Verified Gal(3) mechanics:** multiplication, inverse, algebra, adjoint, exponential/Cayley, time handling; numeric and symbolic tests.

**M2 — Baseline reproduction:** conventional + Rai Cayley factor + direct-product Galilean, same synthetic trajectory. Explicitly state any unavailable reference code and approximations.

**M3 — Stage B derivation:** complete indexed z,Q,{A_k,b_k} for a *small anchored two-keyframe* graph. Write direct residual plus lifted constraint form and show equalities.

**M4 — Stage B SDP:** solve and certify or diagnose failure, compare baseline, log numerical tolerances and sizes.

**M5 — Delama reproduction:** exact tangent-group action/error/covariance, consistency tests, source-code cross-check.

**M6 — Stage C degree/lifting study:** full scalar audit and candidate equations; proof of equivalent lifting or precise obstruction.

**M7 — Stage C prototype:** formulate SDP and constraints only if M6 succeeds; otherwise document smallest tractable surrogate and why it is not fully equivariant.

**M8 — Experiments and research conclusion:** ablations, reproducible plots, novelty matrix, theorem/proof outline, limitations.

## Repo layout

```
certifiable-galilean-imu/
  CLAUDE.md
  README.md
  pyproject.toml
  references/README.md
  docs/
    00_literature_matrix.md
    01_conventions_and_gal3.md
    02_baseline_reproduction.md
    03_stage_b_derivation.md
    04_stage_b_qcqp_matrices.md
    05_stage_b_certification.md
    06_delama_equivariant_reproduction.md
    07_stage_c_degree_audit.md
    08_stage_c_qcqp_or_obstruction.md
    09_ablation_and_results.md
    10_novelty_and_theorem_candidates.md
  src/gal3.py
  src/imu_dynamics.py
  src/preintegrate_stage_b.py
  src/equivariant_stage_c.py
  src/lifting.py
  src/qcqp.py
  src/sdp.py
  src/certification.py
  tests/
  experiments/two_keyframe/
  experiments/bias_sweep/
  experiments/noise_sweep/
  experiments/scaling/
  results/
```

## Computational guardrails
- Python, NumPy, SciPy, SymPy, pytest, CVXPY, MOSEK (if licensed; report absence and use a supported alternative for small problems). Do not claim solver-independent verified certificates without checking solver status and tolerances.
- Run dimension checks, randomized group consistency tests, numerical relative-state tests, finite difference checks, and scalar-vs-matrix polynomial tests.
- Every symbol gets dimensions, frame, units, source, and role (variable/measurement/constant).
- Every formulation must explicitly state global validity, local chart restrictions, bias linearization assumptions, gauge fixing, noise assumptions, and feasible-set equivalence.
- Never conflate: correct QCQP representation, exact physical IMU dynamics, exact convex relaxation, and global optimality certification.
- Never present existing Rai bias lifting or existing GTSAM direct-product Galilean bias handling as a new contribution.
- Track artifacts of a 20D symmetry group separately from physical state and six IMU biases.
- No unsourced theorem statements, fabricated figure metrics, invented references, or unqualified novelty claims.

## Work style
Complete one milestone per iteration. Before moving on, show (a) source equations with citations, (b) full derivation, (c) tests run and their results, (d) unresolved assumptions/errors, and (e) the next exact step. Prefer verified partial derivations over guessed complete formulas.

## First Claude Code instruction

1. Read Rai, Delama, Barfoot, GTSAM Galilean documentation and the Delama reference repository; produce `docs/00_literature_matrix.md` citing equation numbers and distinguishing Stage B from Stage C.
2. Validate Gal(3) matrix algebra **including temporal blocks** and correct any inconsistent endpoint relation from earlier notes.
3. Build a tiny noise-free two-keyframe example with nonzero angular rate, velocity, acceleration, bias, and Delta t, checking direct IMU kinematics and group representation numerically.
4. Identify EXACTLY which formulas from Rai can be reused and which must change for Stage B; write a variable/lifting inventory. Do not implement Stage C QCQP yet.
5. End with a concise research checkpoint: source-verified facts, proven formulas, unresolved issues, and proposed next coding task.
