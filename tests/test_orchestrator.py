"""Tests for the progress tracker itself.

A tracker that cannot detect missing work is worse than no tracker, because it manufactures
confidence. The decisive test here is `test_gate_blocks_on_seeded_tracker_defect`, which
feeds the orchestrator a milestone containing work that was deliberately left undone and
asserts it reports BLOCK rather than PASS.
"""
import copy
import random

import pytest

from orchestrator import dispatch, evidence as ev, gates, ledger


def _ms(**over):
    base = {
        "id": "M9", "title": "Fixture", "branch": "fix/branch", "status": "verifying",
        "claims": [
            {"id": "M9-C01", "statement": "one", "verdict": "CONFIRMED", "attack": "tried X"},
            {"id": "M9-C02", "statement": "two", "verdict": "CONFIRMED", "attack": "tried Y"},
        ],
        "seed": {"id": "M9-S01", "statement": "false thing", "caught": True, "published": False},
        "agent_runs": [],
    }
    base.update(over)
    return base


def _data(ms):
    return {"project": "T", "stage": "B", "milestones": [ms]}


PASSING = {"M9-C01": {"outcome": "passed", "nodeid": "tests/t.py::a"},
           "M9-C02": {"outcome": "passed", "nodeid": "tests/t.py::b"}}


# --- the five-signal rule -----------------------------------------------------------

@pytest.mark.claim("ORCH-C01")
def test_claim_done_only_when_every_signal_green():
    c = {"id": "M9-C01", "statement": "x", "verdict": "CONFIRMED"}
    assert ev.claim_status(c, PASSING) == ev.DONE


@pytest.mark.claim("ORCH-C02")
def test_unbound_claim_reads_untested():
    c = {"id": "M9-C99", "statement": "x", "verdict": "CONFIRMED"}
    assert ev.claim_status(c, PASSING) == ev.UNTESTED


@pytest.mark.claim("ORCH-C03")
def test_failing_test_reads_failing_even_when_confirmed():
    """A verifier saying CONFIRMED must not override a test that actually fails."""
    c = {"id": "M9-C01", "statement": "x", "verdict": "CONFIRMED"}
    assert ev.claim_status(c, {"M9-C01": {"outcome": "failed", "nodeid": "t"}}) == ev.FAILING


@pytest.mark.claim("ORCH-C04")
def test_passing_but_unverified_is_not_done():
    """The whole point: green tests alone never close a claim."""
    c = {"id": "M9-C01", "statement": "x", "verdict": None}
    assert ev.claim_status(c, PASSING) == ev.UNVERIFIED


@pytest.mark.claim("ORCH-C05")
def test_refuted_beats_a_passing_test():
    c = {"id": "M9-C01", "statement": "x", "verdict": "REFUTED"}
    assert ev.claim_status(c, PASSING) == ev.REFUTED


# --- gates --------------------------------------------------------------------------

@pytest.mark.claim("ORCH-C06")
def test_clean_milestone_passes():
    ok, reasons = gates.evaluate(_ms(), PASSING)
    assert ok, reasons


@pytest.mark.claim("ORCH-C07")
def test_gate_blocks_on_seeded_tracker_defect():
    """THE self-test: a milestone with work deliberately left undone must report BLOCK.

    Two defects are planted at once -- a claim with no test binding, and a claim whose test
    fails -- alongside two that are genuinely finished. If the orchestrator reports PASS
    here, every progress number it produces downstream is worthless.
    """
    ms = _ms(claims=[
        {"id": "M9-C01", "statement": "finished", "verdict": "CONFIRMED"},
        {"id": "M9-C02", "statement": "finished", "verdict": "CONFIRMED"},
        {"id": "M9-C03", "statement": "NEVER BOUND TO A TEST", "verdict": "CONFIRMED"},
        {"id": "M9-C04", "statement": "test fails", "verdict": "CONFIRMED"},
    ])
    evidence = dict(PASSING, **{"M9-C04": {"outcome": "failed", "nodeid": "tests/t.py::d"}})
    ok, reasons = gates.evaluate(ms, evidence)
    assert not ok, "orchestrator reported PASS on a milestone with undone work"
    joined = " ".join(reasons)
    assert "M9-C03" in joined and ev.UNTESTED in joined
    assert "M9-C04" in joined and ev.FAILING in joined
    assert "M9-C01" not in joined and "M9-C02" not in joined


@pytest.mark.claim("ORCH-C08")
def test_gate_blocks_when_seeded_defect_missed():
    ms = _ms(seed={"id": "M9-S01", "statement": "f", "caught": False, "published": False})
    ok, reasons = gates.evaluate(ms, PASSING)
    assert not ok
    assert any("missed" in r for r in reasons)


@pytest.mark.claim("ORCH-C09")
def test_gate_blocks_when_no_seed_planted():
    ms = _ms(seed={"id": None, "statement": None, "caught": None, "published": False})
    ok, reasons = gates.evaluate(ms, PASSING)
    assert not ok
    assert any("no seeded defect" in r for r in reasons)


@pytest.mark.claim("ORCH-C10")
def test_gate_blocks_on_empty_milestone():
    """A milestone with nothing to check must not pass by vacuous truth."""
    ok, reasons = gates.evaluate(_ms(claims=[]), {})
    assert not ok
    assert any("no claims" in r for r in reasons)


# --- ledger integrity ---------------------------------------------------------------

@pytest.mark.claim("ORCH-C11")
@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["milestones"].append(copy.deepcopy(d["milestones"][0])), "duplicate milestone"),
    (lambda d: d["milestones"][0].update(status="finished-ish"), "bad status"),
    (lambda d: d["milestones"][0]["claims"].append({"id": "M8-C01", "statement": "x"}), "does not belong"),
    (lambda d: d["milestones"][0]["claims"][0].update(verdict="LOOKS_FINE"), "bad verdict"),
])
def test_validate_rejects_malformed_ledger(mutate, msg):
    d = _data(_ms())
    mutate(d)
    with pytest.raises(ledger.LedgerError, match=msg):
        ledger.validate(d)


@pytest.mark.claim("ORCH-C12")
def test_real_state_yaml_is_valid():
    ledger.validate(ledger.load())


@pytest.mark.claim("ORCH-C13")
def test_discard_voids_every_verdict_and_reopens():
    d = _data(_ms(agent_runs=[{"role": "verifier", "agent_id": "a1",
                               "started": "t", "finished": "t", "outcome": "completed"}]))
    n = ledger.discard_verdicts(d, "M9")
    ms = d["milestones"][0]
    assert n == 2
    assert all(c["verdict"] is None for c in ms["claims"])
    assert ms["seed"]["caught"] is None
    assert ms["agent_runs"][0]["outcome"] == "discarded"
    assert not gates.evaluate(ms, PASSING)[0]


# --- context separation -------------------------------------------------------------

@pytest.mark.claim("ORCH-C14")
def test_implementer_brief_cannot_leak_derivation_narrative():
    """Separation must be structural, not a matter of remembering to omit things."""
    ms = _ms()
    ms["claims"][0]["derivation_note"] = "SENTINEL_NARRATIVE_LEAK"
    ms["claims"][0]["confidence"] = "SENTINEL_CONFIDENCE_LEAK"
    brief = dispatch.implementer_brief(_data(ms), "M9")
    assert "SENTINEL_NARRATIVE_LEAK" not in brief
    assert "SENTINEL_CONFIDENCE_LEAK" not in brief
    assert "M9-C01" in brief and "one" in brief


@pytest.mark.claim("ORCH-C15")
def test_verifier_brief_hides_the_seed_among_real_claims():
    seed = {"id": "M9-S01", "statement": "deliberately false"}
    brief = dispatch.verifier_brief(_data(_ms()), "M9", "src/", "eq 1",
                                    seed, rng=random.Random(0))
    assert "M9-S01" in brief and "M9-C01" in brief
    # the seed must not be labelled as such anywhere in the brief
    seed_line = [l for l in brief.splitlines() if "M9-S01" in l][0]
    for tell in ("seed", "planted", "false claim", "calibrat"):
        assert tell not in seed_line.lower()


@pytest.mark.claim("ORCH-C16")
def test_overall_progress_counts_empty_milestones_as_zero():
    d = {"project": "T", "stage": "B", "milestones": [_ms(), _ms(id="M8", claims=[])]}
    d["milestones"][1]["claims"] = []
    assert ev.overall_progress(d, PASSING) == pytest.approx(0.5)


@pytest.mark.claim("ORCH-C17")
def test_gate_blocks_on_unresolved_verifier_finding():
    """A claim bound to a test that cannot fail is not verified, it only looks verified.

    The M0 verifier found 40 of 41 corrupted table fields survived the suite -- every claim
    was CONFIRMED and every test passed, yet the tests did not test the table. Passing a gate
    in that state would certify nothing, so verifier findings block until remediated.
    """
    ms = _ms(blocking_findings=[
        {"id": "M0-F01", "summary": "40/41 mutations survive; tests do not read the table",
         "resolved": False}])
    ok, reasons = gates.evaluate(ms, PASSING)
    assert not ok, "gate passed despite an unresolved finding that the tests cannot fail"
    assert any("M0-F01" in r for r in reasons)


@pytest.mark.claim("ORCH-C18")
def test_resolved_finding_stops_blocking():
    ms = _ms(blocking_findings=[{"id": "M0-F01", "summary": "fixed", "resolved": True}])
    assert gates.evaluate(ms, PASSING)[0]


@pytest.mark.claim("ORCH-C19")
def test_save_never_writes_seed_statements_or_claims(tmp_path):
    data = {"milestones": [{"id": "M9", "title": "t", "branch": "b", "status": "verifying",
                            "claims": [{"id": "M9-C01", "statement": "x", "verdict": None, "attack": None}],
                            "seed": {"id": "M9-S01", "statement": "PLANTED", "why_false": "W",
                                     "caught": None, "published": False},
                            "agent_runs": []}]}
    out = tmp_path / "state.yaml"
    ledger.save(data, out)
    text = out.read_text()
    assert "PLANTED" not in text and "why_false" not in text and "M9-C01" not in text
    assert "M9-S01" in text


@pytest.mark.claim("ORCH-C20")
def test_latest_replacement_seed_is_the_live_one():
    data = {"milestones": [{"id": "M2", "seed": {}}, {"id": "M20", "seed": {}}]}
    ledger._apply_seeds(data, {"seeds": {
        "M2": {"id": "M2-S01", "caught": "VOID"},
        "M2_replacement": {"id": "M2-S02", "caught": True},
        "M20": {"id": "M20-S01", "caught": None},
        "M2_reverify": {"id": "M2-S03", "caught": None}}})
    assert data["milestones"][0]["seed"]["id"] == "M2-S03"
    assert data["milestones"][1]["seed"]["id"] == "M20-S01"
