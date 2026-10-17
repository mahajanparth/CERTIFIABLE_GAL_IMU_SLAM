# RESUME — where this was left (2026-10-16)

Read this first when restarting. The ledger (`orchestrator/state.yaml`) is the source of truth; this file is a
pointer to it.

```bash
cd /home/parth/Desktop/CERTIFIABLE_GAL_IMU_SLAM && source .venv/bin/activate
PYTHONPATH=. python -m orchestrator.cli status      # computed progress (orchestrator is not on PATH)
PYTHONPATH=. python -m orchestrator.cli agents      # run history; "running" rows are in flight
```

## One-line state

M0 gate PASSED. M1, M2, M3: every real claim CONFIRMED by independent verifiers and every recent planted claim
caught; each is blocked only by mutation-test survivors under the zero-survivor gate. M4/M5 not started.

| M | Worktree | Last verifier | In flight |
|---|---|---|---|
| M1 | `.cgis_worktrees/m1` | pass 9 (b763ab2): 55/55, F18-F21 fixed, F22-F24 opened | verify-M1-011 (pass 10): F22/F23 remediated in 8e1ecae; F24 to be re-ruled under rule 4 |
| M2 | `.cgis_worktrees/m2` | pass 12 (d6f52f8): 25/25, F29 fixed, F30 opened | verify-M2-013 (pass 13): F30 remediated in 803f6e9 |
| M3 | `.cgis_worktrees/m3` | pass 5 (e6823a1): 36/36, F06 fixed, F07 opened | verify-M3-006 (pass 6): F07 remediated in 75c7908 |

All three verifiers were interrupted once by the API session limit (2026-10-15) and resumed with their scratch
intact (`m1/_p10/`, `m2/results/p13/`, `m3/_v6scratch/`). If they are gone, re-dispatch with the same brief and seed.

## Loop per milestone

remediate -> full suite + mutation harness -> independent verifier with a FRESH planted claim -> record -> gate.

## Gate equivalence rules (owner decisions)

Zero surviving non-equivalent mutants. Equivalent only by: (1) effect <= 1/10 of the tightest claim tolerance;
(2) rounding floor, within 8 ulp and no worse than clean; (3) a written proof; (4) **sub-ulp floor (2026-10-15)**:
at most 2 ulp of the quantity's magnitude worse than clean AND >= 100x inside the tightest claim tolerance.

## Seeds — the one file not in git

`orchestrator/seeds.local.yaml` (gitignored on purpose). Backup: `/home/parth/Desktop/.cgis_SEEDS_BACKUP_DO_NOT_COMMIT.yaml`.
Live seeds: M1-S11 (brief M1-C64), M2-S14 (M2-C36), M3-S06 (M3-C45). Write a fresh one, numerically confirmed false,
for every re-verification. Never put a seed in a tracked file or a dashboard row.

## Ledger editing gotcha

`ledger.save()` strips the hand-kept seed history blocks for M0-M2. Edit via load/save to a temp file and splice back
only the touched milestone section, keeping its original `seed:` block.

## Published pages

- Agent board (dashboard, fed from its database): https://claude.ai/artifact/KJFrVUJ8BoEdQF55854rV5 —
  regenerate rows with `PYTHONPATH=. python -m orchestrator.export_dashboard <dir>` and batch-write the changed ones.
- M3 derivation: https://claude.ai/artifact/D3Nu6aisyzUUFBdW3tZj2k (rebuilt from e6823a1).

## Open items needing an owner or deriver

- M1-C52 `domain` text says the near-pi disagreement occurs "in every implementation" — false (only gtsam's Gal3
  Logmap, ~1e-7; src round-trips to 2.2e-15). Claim field: deriver edit.
- M3 derivation doc cleanup: cites non-existent pre-flight F6/F10/F11; unexplained 39-variable gap vs Rai's
  n = 39K + 6M + 16; stale measurements vs pass 5; one garbled sentence in Sec. c.3.
- M4 risks (from M3 verifiers): dense z^T Q z cancellation (max|Q| ~1.1e10 — use the factored cost or rescale);
  redundant families rank-deficient (472/453); handedness rows load-bearing (never prune P2/RH/RS); report each
  rounded point's own constraint residual; no MOSEK, so results are not bit-comparable with published tables.

## Commits

No `Co-Authored-By` trailer, no generated-with footer. Author `parthmahajan`.
