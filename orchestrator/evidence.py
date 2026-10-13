"""Collect objective evidence: does a claim have a test, and does that test pass?

Nothing here asks an agent anything. It runs pytest and reads what came back.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

from .ledger import ROOT

EVIDENCE_FILE = ROOT / ".orchestrator_evidence.json"

# claim lifecycle, worst-first -- every state but DONE blocks a milestone gate
UNTESTED = "UNTESTED"       # no test carries @pytest.mark.claim for this id
FAILING = "FAILING"         # a test exists and it fails
UNVERIFIED = "UNVERIFIED"   # test passes, but no independent CONFIRMED verdict
REFUTED = "REFUTED"         # a verifier actively broke it
UNTESTABLE = "UNTESTABLE"   # a verifier could not construct a test
DONE = "DONE"

ORDER = [REFUTED, FAILING, UNTESTED, UNTESTABLE, UNVERIFIED, DONE]


def run_pytest(quiet: bool = True, extra: list[str] | None = None,
               cwd: pathlib.Path | None = None) -> tuple[int, str]:
    """Actually run the suite. Returns (exit_code, tail_of_output)."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header"]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def collect(fresh: bool = False, tree: pathlib.Path | None = None) -> dict[str, dict]:
    """Return {claim_id: {outcome, nodeid}}. With fresh=True, re-runs pytest first.

    ``tree`` is a milestone's worktree: the suite is run THERE and its evidence file read,
    because a branch's tests and claims are not on main until the gate lets them merge.
    """
    evidence_file = (tree / EVIDENCE_FILE.name) if tree else EVIDENCE_FILE
    if fresh or not evidence_file.exists():
        run_pytest(cwd=tree)
    if not evidence_file.exists():
        return {}
    try:
        return json.loads(evidence_file.read_text())
    except json.JSONDecodeError:
        return {}


def claim_status(claim: dict, evidence: dict[str, dict]) -> str:
    """The five-signal rule. A claim is DONE only when every signal is green."""
    verdict = claim.get("verdict")
    if verdict == REFUTED:
        return REFUTED
    ev = evidence.get(claim["id"])
    if ev is None:
        return UNTESTED
    if ev.get("outcome") != "passed":
        return FAILING
    if verdict == UNTESTABLE:
        return UNTESTABLE
    if verdict != "CONFIRMED":
        return UNVERIFIED
    return DONE


def milestone_progress(ms: dict, evidence: dict[str, dict]) -> tuple[int, int]:
    claims = ms.get("claims") or []
    done = sum(1 for c in claims if claim_status(c, evidence) == DONE)
    return done, len(claims)


def overall_progress(data: dict, evidence: dict[str, dict]) -> float:
    """Mean of per-milestone completion. A milestone with no claims counts as 0."""
    ms_list = data.get("milestones") or []
    if not ms_list:
        return 0.0
    total = 0.0
    for ms in ms_list:
        done, n = milestone_progress(ms, evidence)
        total += (done / n) if n else 0.0
    return total / len(ms_list)


def git_merged(branch: str, into: str = "main") -> bool | None:
    if branch == into:
        return True
    try:
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", branch, into],
            cwd=ROOT, capture_output=True, text=True,
        )
        return r.returncode == 0
    except OSError:
        return None
