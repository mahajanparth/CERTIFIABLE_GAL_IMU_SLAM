# CERTIFIABLE_GAL_IMU_SLAM

A certifiable (QCQP + SDP) formulation of the **direct-product Galilean IMU preintegration factor**.

## What this is

Three lines of prior work meet here, and the combination is unoccupied:

| Work | Group structure | Bias model | Certifiable? |
| --- | --- | --- | --- |
| Rai, Xu, Chen, Rosen (arXiv 2609.38048) | **SO(3) only**, `v, p` plain `R^3` | joint, piecewise-constant blocks, Cayley bias correction | **yes** |
| Delama, Fornasier, Mahony, Weiss (RA-L 2025) | `Gal(3) ⋉ gal(3)`, equivariant | 6 physical + 4 virtual in a 10-vector | no — needs `log`, `J_L^-1` |
| GTSAM 4.3.0 (Dellaert & Delama) | `Gal(3) × R^6` direct product | `ConstantBias`, left-invariant error | no |
| **This work** | `Gal(3) × R^6` direct product | same as GTSAM | **target: yes** |

Rai et al. never mention Galilean groups or `SE_2(3)` and do not cite Delama et al. GTSAM's Galilean
factor is not certifiable and makes no such claim. **The contribution here is the combination —
certification applied to the Galilean factor — and nothing more.** The Galilean bias handling is prior
work (GTSAM's / Delama's); the certification machinery is prior work (Rai's / Barfoot's).

## Scope

**Stage B only.** Stage C (the equivariant tangent-group formulation) is deferred; its branches are
seeded with the pre-flight analysis and left dormant.

## Progress

Run `orchestrator status`, or read [PROGRESS.md](PROGRESS.md) — regenerated at every milestone
transition. Progress is **computed from evidence** (test bindings, pytest outcomes, verifier verdicts),
never self-reported.

## Layout

```
orchestrator/   progress ledger, gates, agent dispatch  (built first — it measures everything else)
src/            Gal(3) mechanics, IMU dynamics, lifting, QCQP, SDP, certification
docs/           literature matrix, conventions, derivations, verification reports
tests/          one test per numbered claim, bound via @pytest.mark.claim("M3-C07")
experiments/    three_keyframe, ablation, sparsity_sweep
```

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
orchestrator status
pytest tests/ -v
```
