"""orchestrator — progress, gates and agent dispatch.

Exit codes are meaningful: `orchestrator gate M3` exits 0 only if M3 is genuinely
passable, so it can be used directly in a pre-merge check.
"""
from __future__ import annotations

import argparse
import sys

from . import dispatch, evidence as ev, gates, ledger, render


def _load(fresh: bool):
    data = ledger.load()
    return data, ev.collect(fresh=fresh)


def cmd_status(a):
    data, evidence = _load(a.fresh)
    print(render.status(data, evidence, verbose=not a.brief))
    return 0


def cmd_claims(a):
    data, evidence = _load(a.fresh)
    print(render.claims_table(ledger.milestone(data, a.milestone), evidence))
    return 0


def cmd_gate(a):
    data, evidence = _load(fresh=True)  # a gate always re-measures; stale evidence is worthless
    ms = ledger.milestone(data, a.milestone)
    ok, reasons = gates.evaluate(ms, evidence)
    if ok:
        print(f"{ms['id']} gate PASS — every claim bound, passing and independently confirmed, "
              f"seeded defect caught.")
        return 0
    print(f"{ms['id']} gate BLOCK — {len(reasons)} blocking issue(s):")
    for r in reasons:
        print(f"  - {r}")
    return 1


def cmd_verify(a):
    data, _ = _load(False)
    code, out = ev.run_pytest()
    evidence = ev.collect(fresh=False)
    print(out.strip()[-2000:])
    print()
    print(render.claims_table(ledger.milestone(data, a.milestone), evidence))
    return 0 if code == 0 else 1


def cmd_next(a):
    data, evidence = _load(a.fresh)
    for ms in data["milestones"]:
        ok, reasons = gates.evaluate(ms, evidence)
        if ok and ms["status"] == "done":
            continue
        claims = ms.get("claims") or []
        seed = ms.get("seed") or {}
        if not claims:
            role = "deriver"
            why = "no claims defined yet"
        elif any(ev.claim_status(c, evidence) in (ev.UNTESTED, ev.FAILING) for c in claims):
            role = "implementer"
            n = sum(1 for c in claims if ev.claim_status(c, evidence) in (ev.UNTESTED, ev.FAILING))
            why = f"{n} claim(s) untested or failing"
        elif seed.get("caught") is False:
            role = "verifier"
            why = "previous verifier missed the seeded defect; verdicts were discarded"
        elif any(ev.claim_status(c, evidence) == ev.UNVERIFIED for c in claims):
            role = "verifier"
            n = sum(1 for c in claims if ev.claim_status(c, evidence) == ev.UNVERIFIED)
            why = f"{n} claim(s) pass but are unverified"
        else:
            role = "review"
            why = "; ".join(reasons[:2]) if reasons else "ready to merge"
        print(f"next: dispatch a {role.upper()} for {ms['id']} ({ms['title']})")
        print(f"  branch : {ms['branch']}")
        print(f"  because: {why}")
        return 0
    print("next: nothing — every milestone gate passes.")
    return 0


def cmd_agents(a):
    data, _ = _load(False)
    rows = [(ms, r) for ms in data["milestones"] for r in (ms.get("agent_runs") or [])]
    if not rows:
        print("no agent runs recorded yet")
        return 0
    print(f"{'M':<4} {'ROLE':<12} {'AGENT':<20} {'OUTCOME':<11} {'STARTED':<22} FINISHED")
    for ms, r in rows:
        print(f"{ms['id']:<4} {r['role']:<12} {str(r['agent_id'])[:18]:<20} "
              f"{r['outcome']:<11} {str(r['started'])[:20]:<22} {r['finished'] or '-'}")
    return 0


def cmd_report(a):
    data, evidence = _load(fresh=True)
    path = render.write_progress_md(data, evidence)
    print(f"wrote {path}")
    print()
    print(render.status(data, evidence, verbose=False))
    return 0


def cmd_brief(a):
    data, _ = _load(False)
    if a.role == "implementer":
        print(dispatch.implementer_brief(data, a.milestone))
    elif a.role == "verifier":
        ms = ledger.milestone(data, a.milestone)
        seed = ms.get("seed") or {}
        sc = {"id": seed["id"], "statement": seed["statement"]} if seed.get("id") else None
        print(dispatch.verifier_brief(data, a.milestone, "<code paths>", "<source equations>", sc))
    else:
        print(dispatch.deriver_brief(data, a.milestone, "<sources>"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orchestrator", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fresh", action="store_true", help="re-run pytest before reporting")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="progress dashboard").add_argument(
        "--brief", action="store_true", help="omit per-claim blocking detail")
    for name, fn, helptext in [
        ("claims", cmd_claims, "per-claim table for one milestone"),
        ("gate", cmd_gate, "evaluate a milestone gate; exit 0 only if passable"),
        ("verify", cmd_verify, "re-run the suite and re-measure one milestone"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("milestone")
    sub.add_parser("next", help="what to dispatch next, and why")
    sub.add_parser("agents", help="agent run history")
    sub.add_parser("report", help="regenerate PROGRESS.md")
    bp = sub.add_parser("brief", help="render an agent brief from the ledger")
    bp.add_argument("role", choices=["deriver", "implementer", "verifier"])
    bp.add_argument("milestone")

    a = p.parse_args(argv)
    fns = {"status": cmd_status, "claims": cmd_claims, "gate": cmd_gate, "verify": cmd_verify,
           "next": cmd_next, "agents": cmd_agents, "report": cmd_report, "brief": cmd_brief}
    try:
        return fns[a.cmd](a)
    except ledger.LedgerError as e:
        print(f"ledger error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
