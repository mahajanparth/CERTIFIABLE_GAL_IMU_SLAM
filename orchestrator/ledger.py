"""Read/write and validate state.yaml -- the single source of truth."""
from __future__ import annotations

import copy
import datetime as _dt
import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = pathlib.Path(__file__).resolve().parent / "state.yaml"

VERDICTS = {"CONFIRMED", "REFUTED", "UNTESTABLE", None}
STATUSES = {"not_started", "deriving", "implementing", "verifying", "blocked", "done"}
ROLES = {"deriver", "implementer", "verifier"}


class LedgerError(RuntimeError):
    pass


CLAIMS_DIR = ROOT / "docs" / "claims"


def load(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Load state.yaml, then overlay per-milestone claim files.

    Claims live in docs/claims/<MID>.yaml rather than in state.yaml so that several
    milestones can be derived concurrently on separate branches without three agents
    fighting over one file. state.yaml keeps only milestone metadata, seeds and run history.
    """
    path = path or STATE
    if not path.exists():
        raise LedgerError(f"no ledger at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    _overlay_claims(data)
    validate(data)
    return data


def _overlay_claims(data: dict[str, Any]) -> None:
    if not CLAIMS_DIR.exists():
        return
    for ms in data.get("milestones") or []:
        f = CLAIMS_DIR / f"{ms['id']}.yaml"
        if not f.exists():
            continue
        doc = yaml.safe_load(f.read_text()) or {}
        claims = doc.get("claims") or []
        for c in claims:
            c.setdefault("verdict", None)
            c.setdefault("attack", None)
        if claims:
            ms["claims"] = claims
        # NOTE: a `seed:` block in a claims file is deliberately IGNORED here. Seeds are
        # read only from the gitignored SEEDS_FILE, so that a committed claims file never
        # tells a verifier which claim was planted. See _overlay_seeds.
    _overlay_seeds(data)


SEEDS_FILE = pathlib.Path(__file__).resolve().parent / "seeds.local.yaml"


def _overlay_seeds(data: dict[str, Any]) -> None:
    """Load seeded defects from the gitignored local file.

    Seeds are never committed. A verifier that goes looking through the repository will
    not find which claim was planted. This is not airtight -- an agent with filesystem
    access could read seeds.local.yaml -- but the failure mode being guarded against is an
    agreeable verifier that rubber-stamps claims, and such a verifier does not go hunting.
    """
    if not SEEDS_FILE.exists():
        return
    _apply_seeds(data, yaml.safe_load(SEEDS_FILE.read_text()) or {})


def _apply_seeds(data: dict[str, Any], doc: dict[str, Any]) -> None:
    """A milestone's live seed is the LAST entry keyed `<MID>` or `<MID>_<suffix>`.

    A voided or already-caught seed is replaced by appending a new entry (e.g. M2_reverify),
    so the gate always judges the current verifier by the seed it was actually given.
    """
    seeds = doc.get("seeds") or {}
    for ms in data.get("milestones") or []:
        mid = ms["id"]
        live = [v for k, v in seeds.items() if k == mid or k.startswith(mid + "_")]
        if live:
            ms["seed"] = {**(ms.get("seed") or {}), **live[-1]}


SEED_PUBLIC_KEYS = ("id", "caught", "published")


def save(data: dict[str, Any], path: pathlib.Path | None = None) -> None:
    """Write state.yaml without anything load() overlaid onto it.

    load() merges in claims (from docs/claims/) and seeds (from the gitignored seeds file).
    Writing those back would put a live seed's statement into a tracked file, so claims are
    dropped and each seed is reduced to its public fields.
    """
    validate(data)
    path = path or STATE
    out = copy.deepcopy(data)
    for ms in out.get("milestones") or []:
        ms["claims"] = []
        seed = ms.get("seed") or {}
        ms["seed"] = {k: seed.get(k) for k in SEED_PUBLIC_KEYS}
    path.write_text(yaml.safe_dump(out, sort_keys=False, width=100))


def validate(data: dict[str, Any]) -> None:
    if "milestones" not in data:
        raise LedgerError("ledger has no 'milestones' key")
    seen_ms: set[str] = set()
    seen_claims: set[str] = set()
    for ms in data["milestones"]:
        mid = ms.get("id")
        if not mid:
            raise LedgerError("milestone with no id")
        if mid in seen_ms:
            raise LedgerError(f"duplicate milestone id {mid}")
        seen_ms.add(mid)
        if ms.get("status") not in STATUSES:
            raise LedgerError(f"{mid}: bad status {ms.get('status')!r}")
        for c in ms.get("claims") or []:
            cid = c.get("id")
            if not cid:
                raise LedgerError(f"{mid}: claim with no id")
            if cid in seen_claims:
                raise LedgerError(f"duplicate claim id {cid}")
            if not cid.startswith(mid + "-"):
                raise LedgerError(f"claim {cid} does not belong to milestone {mid}")
            seen_claims.add(cid)
            if c.get("verdict") not in VERDICTS:
                raise LedgerError(f"{cid}: bad verdict {c.get('verdict')!r}")
        for f in ms.get("blocking_findings") or []:
            if not f.get("id") or not f.get("summary"):
                raise LedgerError(f"{mid}: blocking_finding needs id and summary")
        for run in ms.get("agent_runs") or []:
            if run.get("role") not in ROLES:
                raise LedgerError(f"{mid}: bad agent role {run.get('role')!r}")


def milestone(data: dict[str, Any], mid: str) -> dict[str, Any]:
    for ms in data["milestones"]:
        if ms["id"].upper() == mid.upper():
            return ms
    raise LedgerError(f"no milestone {mid!r}; have {[m['id'] for m in data['milestones']]}")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def add_claim(data, mid: str, cid: str, statement: str) -> None:
    ms = milestone(data, mid)
    ms.setdefault("claims", [])
    if any(c["id"] == cid for c in ms["claims"]):
        raise LedgerError(f"claim {cid} already exists")
    ms["claims"].append({"id": cid, "statement": statement, "verdict": None, "attack": None})


def set_verdict(data, cid: str, verdict: str, attack: str | None = None) -> None:
    if verdict not in VERDICTS or verdict is None:
        raise LedgerError(f"bad verdict {verdict!r}")
    for ms in data["milestones"]:
        for c in ms.get("claims") or []:
            if c["id"].upper() == cid.upper():
                c["verdict"] = verdict
                c["attack"] = attack
                return
    raise LedgerError(f"no claim {cid!r}")


def record_run(data, mid: str, role: str, agent_id: str, outcome: str = "running") -> None:
    ms = milestone(data, mid)
    ms.setdefault("agent_runs", []).append(
        {"role": role, "agent_id": agent_id, "started": now(), "finished": None, "outcome": outcome}
    )


def finish_run(data, mid: str, agent_id: str, outcome: str) -> None:
    ms = milestone(data, mid)
    for run in ms.get("agent_runs") or []:
        if run["agent_id"] == agent_id and run["finished"] is None:
            run["finished"] = now()
            run["outcome"] = outcome
            return
    raise LedgerError(f"no open run {agent_id!r} in {mid}")


def discard_verdicts(data, mid: str) -> int:
    """Void every verdict in a milestone -- used when a verifier misses its seeded defect."""
    ms = milestone(data, mid)
    n = 0
    for c in ms.get("claims") or []:
        if c.get("verdict") is not None:
            c["verdict"] = None
            c["attack"] = None
            n += 1
    for run in ms.get("agent_runs") or []:
        if run["role"] == "verifier" and run["outcome"] in ("completed", "running"):
            run["outcome"] = "discarded"
    seed = ms.get("seed") or {}
    seed["caught"] = None
    ms["seed"] = seed
    ms["status"] = "implementing"
    return n
