"""Collects @pytest.mark.claim bindings and writes them where the orchestrator can read them.

Dependency-free on purpose: the orchestrator must be able to measure progress without
pulling in a reporting plugin that could itself be missing or misconfigured.
"""
import json
import pathlib

_EVIDENCE = pathlib.Path(__file__).parent / ".orchestrator_evidence.json"
_results: dict[str, dict] = {}


def pytest_configure(config):
    _results.clear()


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    for marker_claims in report.user_properties:
        pass
    # claim ids are attached in the makereport hook below
    claims = getattr(report, "claim_ids", None)
    if not claims:
        return
    for cid in claims:
        prev = _results.get(cid)
        # a claim bound to several tests is DONE only if every one of them passes
        if prev is None or report.outcome != "passed":
            _results[cid] = {"outcome": report.outcome, "nodeid": report.nodeid}


def pytest_runtest_makereport(item, call):
    import _pytest.runner

    report = _pytest.runner.TestReport.from_item_and_call(item, call)
    ids = [m.args[0] for m in item.iter_markers(name="claim") if m.args]
    report.claim_ids = ids
    return report


def pytest_sessionfinish(session, exitstatus):
    _EVIDENCE.write_text(json.dumps(_results, indent=2, sort_keys=True))
