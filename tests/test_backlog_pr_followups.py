"""Current PR traceability must preserve history and fail closed on bad mappings."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.governance.validate_backlog_integrity import BacklogValidator

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs/planning/arx_canonical_backlog.json"
SCHEMA = ROOT / "docs/planning/arx_canonical_backlog.schema.json"


def validate(data):
    return BacklogValidator(
        data, schema_data=json.loads(SCHEMA.read_text()),
        prs_inventory={"pull_requests": [{"number": n} for n in (310, 274, 146)]},
    ).validate_all()


def test_current_prs_are_tracked_without_mutating_historical_tasks():
    data = json.loads(BACKLOG.read_text())
    tasks = copy.deepcopy(data["tasks"])
    verdict, findings = validate(data)
    assert verdict == "PASS", findings
    assert data["tasks"] == tasks
    assert len(tasks) == 43
    assert len(data["obligations"]) == 71
    assert not data["sprint5_execution_authorized"]


@pytest.mark.parametrize("changes", [
    {"canonical_task_id": "ARX-S4-MISSING"},
    {"canonical_task_id": {"malformed": True}},
    {"owner": " "},
    {"owner": "OWNER_UNASSIGNED"},
    {"acceptance_criteria": []},
    {"evidence_refs": [" "]},
    {"disposition": "HELD", "holds": []},
    {"authorization_boundary": "AUTHORIZED_FOR_PRODUCTION"},
    {"register_id": "REG-PR-999"},
])
def test_invalid_followup_does_not_suppress_untracked_warning(changes):
    data = json.loads(BACKLOG.read_text())
    data["pull_request_followups"][0].update(changes)
    verdict, findings = validate(data)
    assert verdict == "FAIL"
    assert any(f.rule_id == "BI-007" and f.severity == "ERROR" for f in findings)
    assert any(f.rule_id == "BI-007" and "#310 is not linked" in f.message for f in findings)


def test_duplicate_followup_is_an_error():
    data = json.loads(BACKLOG.read_text())
    data["pull_request_followups"].append(copy.deepcopy(data["pull_request_followups"][0]))
    verdict, findings = validate(data)
    assert verdict == "FAIL"
    assert any("Duplicate PR follow-up" in f.message for f in findings)


@pytest.mark.parametrize("flags", [["--stage", "strict"], ["--strict"], ["--stage", "report-only"]])
def test_cli_warning_never_claims_merge_eligibility(tmp_path, flags):
    inventory = tmp_path / "prs.json"
    inventory.write_text(json.dumps({"pull_requests": [{"number": 999999}]}))
    result = subprocess.run([
        sys.executable, str(ROOT / "tools/governance/validate_backlog_integrity.py"),
        "--backlog", str(BACKLOG), "--schema", str(SCHEMA),
        "--baseline", str(BACKLOG), "--prs-inventory", str(inventory), *flags,
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == (0 if "report-only" in flags else 1)
    assert "[NOT CLEARED]" in result.stdout
    assert "PR MERGE ELIGIBLE" not in result.stdout
    assert "Sprint 5 Execution Permission): BLOCKED" in result.stdout
