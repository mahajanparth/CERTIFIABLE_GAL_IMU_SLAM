"""Generate agent briefs from the ledger.

This is the mechanism that enforces context separation, and it exists precisely so the
separation does not depend on anyone remembering it. The implementer's brief is rendered
from claim statements with the derivation narrative STRUCTURALLY ABSENT -- there is no
code path here that can emit it. The verifier's brief is rendered the same way, with the
seeded defect shuffled in among the real claims so it is indistinguishable from them.
"""
from __future__ import annotations

import random

from .ledger import milestone

_SEPARATION_NOTE = (
    "You have deliberately NOT been given the derivation narrative, the deriver's "
    "reasoning, or any statement of how confident anyone is in these claims. That "
    "omission is the point: it is what makes your assessment independent. Do not ask "
    "for it, and do not treat its absence as missing context."
)


def deriver_brief(data, mid: str, sources: str) -> str:
    ms = milestone(data, mid)
    return f"""# Derivation task — {ms['id']}: {ms['title']}

Branch: `{ms['branch']}`

## Sources
{sources}

## What to produce
A derivation document at `docs/` containing **numbered claims**. Each claim must be:
- stated as a **machine-checkable predicate** -- something a test can decide true or false,
  not a prose assertion like "the formulation is sound";
- given an id of the form `{ms['id']}-C01`, `{ms['id']}-C02`, ...;
- accompanied by a **suggested test**: the inputs, the expected result, the tolerance.

## Constraints
- **Do not write anything under `src/`.** A separate implementer builds the code from your
  claims alone. If you implement it yourself, the two will agree by construction and the
  cross-check is worthless.
- Cite equation numbers from the sources. If you cannot read an equation clearly, write
  "could not extract" -- never guess a number.
- Distinguish what you READ from what you DERIVED. Tag every claim.
- State domain restrictions explicitly: chart singularities, linearization validity,
  gauge freedoms, what must be nonzero.
"""


def implementer_brief(data, mid: str, notes: str = "") -> str:
    """Claims only. The derivation narrative is not reachable from here by construction."""
    ms = milestone(data, mid)
    claims = ms.get("claims") or []
    body = "\n".join(f"- **{c['id']}** — {c['statement']}" for c in claims) or "  (none yet)"
    return f"""# Implementation task — {ms['id']}: {ms['title']}

Branch: `{ms['branch']}`

## Claims to implement and bind
{body}

## What to produce
- Code under `src/`.
- Tests under `tests/`, **one test per claim**, each bound with the marker:
  ```python
  @pytest.mark.claim("{ms['id']}-C01")
  def test_whatever():
      ...
  ```
  The orchestrator discovers claims through that marker and through nothing else. An
  unbound claim reads as `UNTESTED` and blocks the milestone gate, so a claim you skip
  will be visible rather than silently lost.

## Constraints
- **Do not edit a claim to fit your code.** If a claim looks wrong, implement it as
  written, let the test fail, and say so in your report. A failing test that surfaces a
  bad claim is a success here.
- Randomized property tests should use >= 1e4 samples and cover the stated domain
  boundaries.

{_SEPARATION_NOTE}

{notes}
"""


def verifier_brief(data, mid: str, code_paths: str, source_equations: str, seed_claim: dict | None = None,
                   rng: random.Random | None = None) -> str:
    """Claims + code + sources, narrative absent, seeded defect shuffled in among the rest."""
    ms = milestone(data, mid)
    claims = [{"id": c["id"], "statement": c["statement"]} for c in (ms.get("claims") or [])]
    if seed_claim:
        claims = claims + [seed_claim]
        (rng or random.Random()).shuffle(claims)
    body = "\n".join(f"- **{c['id']}** — {c['statement']}" for c in claims) or "  (none)"
    return f"""# Independent verification task — {ms['id']}: {ms['title']}

**Your job is to falsify these claims, not to confirm them.** A verification pass that
confirms everything is the expected outcome only when the work is flawless, which is rare.
Treat each claim as a hypothesis you are trying to break.

## Claims under test
{body}

## Source equations
{source_equations}

## Code under test
{code_paths}

## Required deliverables
1. An **independent re-implementation** of at least one core routine, written from the
   source equations alone -- not adapted from the code under test.
2. **Randomized property tests**, >= 1e4 samples, including near-boundary cases.
3. A **both-directions** equivalence check where a lifting is involved. Confirming that
   ground truth satisfies the constraints is the easy half. The hard half is hunting for
   **spurious feasible points**: assignments that satisfy every constraint without
   corresponding to a valid state. Look for them actively.
4. A verdict per claim id: `CONFIRMED` / `REFUTED` / `UNTESTABLE`. For each one, name the
   **concrete attack you attempted**. "Looks correct" is not a verdict.

Write your report to `docs/verification/{ms['branch'].replace('/', '_')}.md`.

{_SEPARATION_NOTE}

One further note, stated plainly: at least some claim lists contain a deliberately false
claim used to calibrate whether verification is being done adversarially. Assume this list
may be one of them.
"""
