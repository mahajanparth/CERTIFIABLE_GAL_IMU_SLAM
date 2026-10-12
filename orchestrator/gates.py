"""Milestone gate predicates. A gate is a hard precondition for merging, not a suggestion."""
from __future__ import annotations

from . import evidence as ev
from .ledger import milestone


def evaluate(ms: dict, evidence: dict[str, dict]) -> tuple[bool, list[str]]:
    """Return (passes, blocking_reasons). Empty reasons == genuinely passable."""
    reasons: list[str] = []
    claims = ms.get("claims") or []

    if not claims:
        reasons.append("no claims defined -- a milestone with nothing to check cannot pass")

    for c in claims:
        st = ev.claim_status(c, evidence)
        if st == ev.DONE:
            continue
        detail = {
            ev.UNTESTED: "no test carries @pytest.mark.claim for it",
            ev.FAILING: "its test fails",
            ev.UNVERIFIED: "no independent CONFIRMED verdict",
            ev.REFUTED: "a verifier refuted it",
            ev.UNTESTABLE: "a verifier could not test it",
        }[st]
        reasons.append(f"{c['id']} is {st}: {detail}")

    # A verifier may report findings that invalidate the evidence itself -- most importantly
    # tests that CANNOT FAIL. A claim bound to a vacuous test is not verified, it only looks
    # verified, so passing the gate on it would certify nothing. These block until remediated.
    for f in ms.get("blocking_findings") or []:
        if not f.get("resolved"):
            reasons.append(f"unresolved verifier finding {f.get('id')}: {f.get('summary')}")

    seed = ms.get("seed") or {}
    if not seed.get("id"):
        reasons.append("no seeded defect planted -- verifier was never calibrated")
    elif seed.get("caught") is not True:
        state = "missed" if seed.get("caught") is False else "not yet evaluated"
        reasons.append(f"seeded defect {seed['id']} {state} -- verdicts cannot be trusted")

    return (not reasons), reasons


def evaluate_id(data: dict, mid: str, evidence: dict[str, dict]) -> tuple[bool, list[str]]:
    return evaluate(milestone(data, mid), evidence)
