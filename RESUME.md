# RESUME — where this was left (2026-10-12, session limit)

Read this first when restarting. Everything is committed and pushed **except one file**, called out below.

## One-line state

**95 claims derived · 88 confirmed · 7 refuted · 0 milestone gates passed.** All three active milestones
were independently verified and **blocked on 17 findings**. Remediation was running on all three when the
session limit hit; it is **partial and must be re-dispatched**.

```bash
cd /home/parth/Desktop/CERTIFIABLE_GAL_IMU_SLAM && source .venv/bin/activate
PYTHONPATH=. python -m orchestrator.cli status      # computed progress
PYTHONPATH=. python -m orchestrator.cli gate M0     # blocking reasons, exits 1
```

## ⚠️ The one file not in git — do not lose it

`orchestrator/seeds.local.yaml` holds the **seeded defects** (deliberately false claims used to test whether
verifiers are adversarial). It is gitignored **on purpose** — a committed seed is readable by the verifier it
calibrates, which has already happened once (see below).

- Lives at: `orchestrator/seeds.local.yaml`
- Backup at: `/home/parth/Desktop/.cgis_SEEDS_BACKUP_DO_NOT_COMMIT.yaml`

A fresh `git clone` will **not** have it. If you clone elsewhere, copy the backup into `orchestrator/`.

## Exact state per milestone

| M | Worktree | Suite now | Verifier verdict | Seed | Remediation |
|---|---|---|---|---|---|
| **M0** | `.cgis_worktrees/m0` | **14 failed / 26 passed** | 15 CONFIRMED | caught | **MID-REWRITE — broken** |
| **M1** | `.cgis_worktrees/m1` | **74 passed** | 54 CONFIRMED / 1 REFUTED | both caught | **not started** |
| **M2** | `.cgis_worktrees/m2` | **65 passed / 1 failed** | 19 CONFIRMED / 5 REFUTED | caught | **nearly done** |
| M3–M5 | — | — | — | — | not started (need M1 + M2 gated first) |

Worktrees are at `/home/parth/Desktop/.cgis_worktrees/{m0,m1,m2,m6}`. Every branch is pushed.

### M0 — mid-rewrite, currently broken
The remediation agent was rewriting `tests/test_conventions.py` so tests **read the table** instead of
hard-coding values (finding M0-F01: 40 of 41 corrupted table fields survived the old suite). It stopped
partway. The 14 failures are a half-finished rewrite, **not** a regression in the facts — `src/conventions.py`
was verified correct against the primary sources entry by entry. Commit is labelled `WIP`.

**To resume:** re-dispatch an M0 remediation agent with the brief in this file's "Re-dispatch" section. It
should start from the current WIP commit and finish the rewrite, then build `tests/test_mutation.py`.

### M1 — clean, remediation not started
The agent died before editing anything. Suite is green at the last verified commit plus recorded verdicts.
Six findings to fix, worst first:
- **M1-F01** the "5000-step ground truth" is decorative — the endpoint relation is an exact algebraic identity
  of the discrete recursion for *any* `dt`, so a 2-step fixture passes; the Euler state is actually `1.58e-4`
  from exact, 10x above the tests' own `1e-5` bar. Validate against `solve_ivp(DOP853, rtol=atol=1e-13)`.
- **M1-F02** constant `(omega, a)` hides swapped accumulation order; `dR @ E -> E @ dR` survives all 55 tests.

### M2 — one mutant left
65/66 passing. The single failure is `test_mutation.py::test_mutant_is_killed[M07]` — a mutation that still
survives `test_c20_galilean_bias_correction_is_right_composed_in_gal3`. Everything else in the remediation
landed. **The `src/` edits in the WIP commit are not independently re-verified** — re-run the M2 verifier
after M07 is killed.

## Re-dispatch

Every finding, with full measurements, is recorded in two places:
- `orchestrator/state.yaml` → `blocking_findings` per milestone (the gate reads these)
- `docs/verification/<branch>.md` on each branch — the verifier's full report

The original remediation briefs are reproducible from those. The pattern for each milestone is:
**remediate → re-run the suite → re-run an independent verifier with a FRESH seed → gate.**

**A fresh seed is required for every re-verification.** The current seeds have now been seen by their verifiers.

## Things learned the hard way — keep these

1. **Seeds must never touch a tracked file.** Two derivers wrote seeds into `docs/claims/`; the strip-commit
   left the originals in `git log -p`. The M2 implementer found its seed that way and disclosed it. Both were
   voided and replaced. Derivers must report seeds in their hand-back message only.
2. **Green tests are not evidence.** In all three milestones every claim was bound and passing, and
   independent verification still found something disqualifying. Mutation testing is what exposed it — make it
   a standing verifier deliverable.
3. **Constant / commuting inputs hide order bugs.** Found independently in M1 and M2. Any test of
   composition or accumulation order needs a time-varying stream.
4. **The gate now blocks on unresolved verifier findings**, not just on claim status. A claim bound to a test
   that cannot fail only *looks* verified.

## Corrections made to the original research contract

Kept verbatim at `docs/original_research_contract.md`; `CLAUDE.md` is the corrected, authoritative version.

- **F2** the endpoint relation `X_j = H X_i D` is wrong twice (time block and a `+g*Dt^2` position error).
  Corrected form reproduces Delama Eq. (18) **bit-identically**.
- **F3** was incomplete: Cayley carries a `+(alpha/2)u` time coupling never tested at `alpha=0`.
- **F9** wording was wrong: a 2-keyframe graph has nullity 6, but the null space is **not** the bias block.
- **3 keyframes / 2 edges / GNSS anchors** is the minimal bias-observable instance, not 2 keyframes.

## Commits

No `Co-Authored-By` trailer, no generated-with footer — your standing instruction. Author `parthmahajan`.
