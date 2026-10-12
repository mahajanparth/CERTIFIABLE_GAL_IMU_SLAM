# CLAUDE.md — Research Agent Context

## Project
**Certifiable Galilean IMU Preintegration — Stage B**

A certifiable (QCQP + SDP) formulation of the **direct-product Galilean IMU preintegration factor**.

> Derived from the original research contract at `CLAUDE_CODE_GALILEAN_STAGE_B_C.md`.
> **One error in that document has been corrected here — see "Corrections" below.** Where this file
> and the original disagree, **this file wins**.

## Scope

**Stage B only.** Stage C (equivariant tangent-group formulation) is deferred; see
`docs/06_stage_c_hypothesis.md` on branch `stage-c/m6-degree-audit` for the preserved analysis.

## What the contribution is, stated precisely

| Work | Group structure | Bias model | Certifiable? |
| --- | --- | --- | --- |
| Rai et al. (arXiv 2609.38048) | **SO(3) only**, `v, p` plain `R^3` | joint, piecewise-constant blocks | **yes** |
| Delama et al. (RA-L 2025) | `Gal(3) ⋉ gal(3)` equivariant | 6 physical + 4 virtual | no — needs `log`, `J_L^-1` |
| GTSAM 4.3.0 | `Gal(3) × R^6` direct product | `ConstantBias` | no |
| **This work** | `Gal(3) × R^6` direct product | same as GTSAM | **target: yes** |

Rai et al. never mention Galilean groups or `SE_2(3)` and do not cite Delama. GTSAM's Galilean factor
is not certifiable. **The contribution is the combination — certification applied to the Galilean
factor — and nothing more.** The Galilean bias handling is prior work (GTSAM's / Delama's); the
certification machinery is prior work (Rai's / Barfoot's). Never present either as new.

## Corrections to the original contract

### The endpoint relation was wrong twice over

The original warns that `X_j = H_ij X_i D_ij` has a nonzero time block. True, but **incomplete** —
the position block is ALSO wrong, by exactly `+g*Dt^2`. Both verified numerically against
ground-truth integration (see `docs/00_preflight_findings.md`, F2).

**Use this, not the original:**

```
X_j = T(-Dt) . H' . X_i . D_ij

H'     = [[I, g*Dt, -0.5*g*Dt^2], [0,1,0], [0,0,1]]    # MINUS on the position block
T(-Dt) = [[I, 0, 0], [0,1,-Dt], [0,0,1]]               # applied on the LEFT
```

`T(-Dt)` on the left zeroes the time block without perturbing `(R,v,p)`; on the right it would
corrupt `p` by `-v*Dt`. **Unit-test every 5x5 matrix equation before using it in an error factor.**

### The 2-keyframe test instance cannot observe biases

The original specifies an "anchored two-keyframe" graph. Rank testing shows pose+velocity anchoring
leaves **exactly the 6 bias directions unobservable** (F9). Use **3 keyframes / 2 edges with GNSS
position anchors**, which reaches full rank 45/45 with no bias prior.

## Mathematical contract

```
X(R,v,p,t) = [[R, v, p], [0_{1x3}, 1, t], [0_{1x3}, 0, 1]]     5x5, R in SO(3)
X1 X2 = (R1R2, v1 + R1 v2, p1 + R1 p2 + v1 t2, t1 + t2)
X^-1  = (R^T, -R^T v, -R^T(p - t v), -t)
xi^   = [[phi_x, u, w],[0,0,alpha],[0,0,0]],  ordering (phi, u, w, alpha)

Ad_X(xi):  phi' = R phi
           u'   = R u + [v]x R phi
           w'   = R w - t R u + [p - t v]x R phi + alpha v
           alpha' = alpha

IMU: omega_m = omega + b_g + n_g;  a_m = a + b_a + n_a
     Rdot = R[omega]x;  vdot = g + R a;  pdot = v
Bias ordering: [b_g; b_a] throughout.
```

**Cayley**, for a fixed-duration zero-time residual: `Cay(e) = (I - e^/2)^-1 (I + e^/2)` gives

```
(I - phi_x/2) Re = I + phi_x/2
(I - phi_x/2) ve = u
(I - phi_x/2) pe = w
```

each **degree <= 2** when `(phi, Re, ve, pe)` are independent lifted variables. `Re` is automatically
a proper rotation, so it needs **no** determinant constraint. The **absolute** rotations `R_i, R_j`
are NOT Cayley-parameterised and **do** need full `SO(3)` lifting with cross-product handedness.
Cayley excludes `theta = pi`: `|phi| = 2tan(theta/2)` reaches `4.3e4` at `3.1415 rad`.

## Ordering footguns (these cause silent wrong answers)

GTSAM 4.3.0, all confirmed in source:
1. Matrix column order has **`v` before `p`**.
2. The constructor is `Gal3(R, r, v, t)` — **position before velocity**, opposite to the matrix.
3. Tangent order is `(omega, nu, rho, alpha)` = (rot, **vel**, pos, time), while `NavState` tangent
   is (rot, **pos**, vel). GTSAM's own remap table is `kGal3BlockForNavState = {0, 6, 3}`.
4. `imuBias::ConstantBias` is **(accel, gyro)**; this project fixes `[b_g; b_a]`. Convert at the
   boundary, explicitly, in one place.

Delama's 10-dim bias vector is **not** 6 biases plus padding: coordinates 6-9 are a virtual velocity
bias and a virtual time bias with **zero** process noise. Never silently reinterpret them.

## How work is organised

Every milestone runs a **derive / implement / verify triad** with deliberately separated context:

- **Deriver** — produces numbered, machine-checkable claims in `docs/claims/<MID>.yaml`. Writes
  nothing under `src/`.
- **Implementer** — receives the claim list **only**, never the derivation narrative. Binds each
  claim to a test with `@pytest.mark.claim("M1-C03")`.
- **Verifier** — fresh context, narrative absent, mission is **falsification not confirmation**.
  Must independently re-implement at least one core routine, run >= 1e4-sample property tests, and
  hunt for **spurious feasible points** (not merely check that ground truth satisfies constraints).

Each verifier's claim list is salted with **one deliberately false claim**. If the verifier confirms
it, its entire verdict set is discarded and re-run. `orchestrator/dispatch.py` generates the briefs,
which is what makes the separation structural rather than a matter of remembering.

## Progress

```bash
orchestrator status      # computed from evidence, never self-reported
orchestrator gate M3     # exit 0 only if genuinely passable
orchestrator next        # what to dispatch, and why
```

A claim is `DONE` only when **all five** signals are green: bound to a test, that test passes when
pytest is actually run, an independent verifier returned `CONFIRMED`, that verifier caught its seeded
defect, and the branch is merged.

## Guardrails

- Never conflate these four: *QCQP-representable*, *exact physical dynamics*, *exact convex
  relaxation*, *certified global optimum*.
- **Covariance freezing is a certificate caveat, not a detail.** Rai freeze theirs (preintegration
  and bias Jacobians computed once at nominal `bbar`, covariance reduced to three scalars via
  `tau^-1 = (1/3)tr(Sigma)`). Where we do the same, we certify the *frozen-weight* problem.
- **Certification is never claimed from rank-1 appearance alone**, and never from an SCS solve —
  Barfoot's rank-1 test needs `sigma_1/sigma_2 >= 1e5`, which a `~1e-4` first-order solver cannot
  resolve. Clarabel is the default backend. No MOSEK, so results are not bit-comparable with the
  published tables; say so wherever they are compared.
- **The real tightness risk is graph sparsity, not the Cayley singularity.** An IMU chain is sparse,
  which is where Barfoot's relaxation was weakest. Redundant constraints are the ballgame.
- No unsourced theorems, no fabricated metrics, no invented references, no unqualified novelty claims.
- Never guess an equation number — write `could not extract`.

## Commits

**No `Co-Authored-By` trailer. No generated-with footer.** Author is `parthmahajan`.
