"""Export the ledger as dashboard rows for the published progress artifact.

Writes one JSON file per document under an output directory, plus `writes.json`, a batch
the ArtifactData tool can apply. Counts are computed the same way `orchestrator status`
computes them; nothing here is self-reported.

Seed statements are never exported -- only id, status and how it was caught -- because a
seed's wording is the calibration secret.
"""
from __future__ import annotations

import json
import pathlib
import sys

import yaml

from . import ledger

WORKTREES = pathlib.Path("/home/parth/Desktop/.cgis_worktrees")
TREE = {"M0": "m0", "M1": "m1", "M2": "m2"}


def _claims(ms):
    tree = TREE.get(ms["id"])
    f = WORKTREES / tree / "docs" / "claims" / f"{ms['id']}.yaml" if tree else None
    if f and f.exists():
        return (yaml.safe_load(f.read_text()) or {}).get("claims") or []
    return ms.get("claims") or []


def build(data):
    docs = []
    seeds = (yaml.safe_load(ledger.SEEDS_FILE.read_text()) or {}).get("seeds", {}) if ledger.SEEDS_FILE.exists() else {}
    totals = {"claims": 0, "confirmed": 0, "refuted": 0, "findings_open": 0, "gates_passed": 0}
    for order, ms in enumerate(data["milestones"]):
        claims = _claims(ms)
        conf = sum(1 for c in claims if c.get("verdict") == "CONFIRMED")
        ref = sum(1 for c in claims if c.get("verdict") == "REFUTED")
        finds = ms.get("blocking_findings") or []
        open_f = [f for f in finds if not f.get("resolved")]
        ms_seeds = [s for k, s in seeds.items() if k.split("_")[0] == ms["id"]]
        gate = "pass" if (claims and conf == len(claims) and not open_f and ms_seeds
                         and all(s.get("caught") is True for s in ms_seeds if s.get("caught") != "VOID")) else "block"
        if ms["status"] == "not_started":
            gate = "none"
        totals["claims"] += len(claims); totals["confirmed"] += conf; totals["refuted"] += ref
        totals["findings_open"] += len(open_f); totals["gates_passed"] += gate == "pass"
        docs.append(("milestones", ms["id"], {
            "order": order, "title": ms["title"], "branch": ms["branch"], "status": ms["status"],
            "claims": len(claims), "confirmed": conf, "refuted": ref,
            "findings_open": len(open_f), "findings_total": len(finds), "gate": gate}))
        for i, r in enumerate(ms.get("agent_runs") or []):
            docs.append(("runs", r["agent_id"], {
                "milestone": ms["id"], "seq": order * 100 + i, "role": r["role"],
                "started": r["started"], "finished": r.get("finished"),
                "outcome": r["outcome"], "note": r.get("note") or ""}))
        for f in finds:
            s = f["summary"]
            docs.append(("findings", f["id"], {
                "milestone": ms["id"], "resolved": bool(f.get("resolved")),
                "critical": s.startswith("CRITICAL"), "summary": s.removeprefix("CRITICAL: ")}))
    for key, s in seeds.items():
        status = {True: "caught", False: "missed", "VOID": "void"}.get(s.get("caught"), "pending")
        docs.append(("seeds", s["id"], {
            "milestone": key.split("_")[0], "status": status,
            "how": (s.get("how") or s.get("void_reason") or "")[:400]}))
    docs.append(("meta", "summary", {**totals, "updated": ledger.now()}))
    return docs


def main(out=sys.argv[1] if len(sys.argv) > 1 else "/tmp/dashboard_export"):
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    docs = build(ledger.load())
    writes = []
    for coll, did, body in docs:
        p = out / coll / f"{did}.json"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body, indent=1))
        writes.append({"op": "set", "collection": coll, "doc_id": did, "data": body})
    (out / "writes.json").write_text(json.dumps(writes))
    print(f"{len(docs)} documents -> {out}")


if __name__ == "__main__":
    main()
