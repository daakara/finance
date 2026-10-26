import copy
import json
import os
import pytest
from tools.governance.validate_backlog_integrity import BacklogValidator

# Minimal valid canonical backlog fixture
MINIMAL_VALID_BACKLOG = {
    "schema_version": "1.0.0",
    "backlog_name": "Test Backlog",
    "last_reconciled": "2026-10-26T12:00:00Z",
    "active_sprint": "SPRINT_4",
    "governance_framework": "ARX Closure Model v2",
    "immutable_holds": {},
    "tasks": [
        {
            "task_id": "ARX-S4-GOV-001",
            "title": "Cross-Sprint Backlog Integrity",
            "origin_sprint": "SPRINT_4",
            "assigned_sprint": "SPRINT_4",
            "priority": "P1",
            "workstream": "GOVERNANCE",
            "owner": "daakara",
            "status": "IN_PROGRESS",
            "disposition": "SPRINT4_EXECUTION",
            "dimensions": {
                "requirements": "DONE",
                "implementation": "IN_PROGRESS",
                "integration": "NOT_STARTED",
                "verification": "PENDING",
                "repository_publication": "LOCAL_ONLY",
                "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE",
                "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "CONDITIONALLY_PREAPPROVED"
            },
            "acceptance_criteria": ["Validate backlog"],
            "dependencies": [],
            "evidence_refs": ["docs/planning/arx_canonical_backlog.json"],
            "github_issues": [],
            "github_prs": [101],
            "authorization_boundary": "BOUNDED_ENGINEERING",
            "operational_state": "ACTIVE",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": None
        }
    ]
}


class TestBacklogIntegrityValidator:
    """15 required automated acceptance tests for GOV-001 Backlog Integrity Validator."""

    # Scenario 1: Missing owner on active executable task
    def test_scenario_01_missing_owner(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["owner"] = "OWNER_UNASSIGNED"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-002" and "no accountable owner" in f.message for f in findings)

    # Scenario 2: Missing or invalid sprint assignment
    def test_scenario_02_missing_sprint(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["assigned_sprint"] = "SPRINT_INVALID"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-003" for f in findings)

    # Scenario 3: Duplicate task ID
    def test_scenario_03_duplicate_task_id(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        dup = copy.deepcopy(data["tasks"][0])
        data["tasks"].append(dup)
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-001" and "Duplicate" in f.message for f in findings)

    # Scenario 4: Orphaned active PR reference
    def test_scenario_04_orphaned_active_pr(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        prs_inventory = {"open_prs": [101, 999]}  # PR 999 is unlinked
        validator = BacklogValidator(data, prs_inventory=prs_inventory)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS_WITH_WARNINGS"
        assert any(f.rule_id == "BI-007" and "PR #999" in f.message for f in findings)

    # Scenario 5: Closed sprint containing an undispositioned commitment
    def test_scenario_05_closed_sprint_undispositioned(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S3-TEST-001",
            "title": "Undispositioned Sprint 3 task",
            "origin_sprint": "SPRINT_3",
            "assigned_sprint": "SPRINT_3",
            "priority": "P2",
            "workstream": "GOVERNANCE",
            "owner": "daakara",
            "status": "IN_PROGRESS",  # Illegal in closed sprint!
            "disposition": "UNVERIFIED",
            "dimensions": {
                "requirements": "DONE", "implementation": "PARTIAL", "integration": "NOT_STARTED",
                "verification": "PENDING", "repository_publication": "NOT_APPLICABLE", "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "HOLD"
            },
            "acceptance_criteria": [],
            "dependencies": [],
            "evidence_refs": [],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "INTERNAL",
            "operational_state": "NONE",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-010" for f in findings)

    # Scenario 6: Carried-forward task with no destination or missing reason
    def test_scenario_06_carryforward_no_destination(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S2-TEST-002",
            "title": "Carried forward without reason",
            "origin_sprint": "SPRINT_2",
            "assigned_sprint": "SPRINT_UNASSIGNED",
            "priority": "P1",
            "workstream": "RADAR_VCP",
            "owner": "daakara",
            "status": "READY",
            "disposition": "SPRINT4_DISCOVERY",
            "dimensions": {
                "requirements": "DONE", "implementation": "NOT_STARTED", "integration": "NOT_STARTED",
                "verification": "PENDING", "repository_publication": "NOT_APPLICABLE", "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "AUTHORIZED"
            },
            "acceptance_criteria": [],
            "dependencies": [],
            "evidence_refs": [],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "DISCOVERY",
            "operational_state": "NONE",
            "carryforward_reason": None,  # Missing!
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-005" for f in findings)

    # Scenario 7: Missing completion evidence on CLOSED_VERIFIED task
    def test_scenario_07_missing_completion_evidence(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["disposition"] = "CLOSED_VERIFIED"
        data["tasks"][0]["status"] = "COMPLETED"
        data["tasks"][0]["evidence_refs"] = []  # Missing evidence!
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-006" for f in findings)

    # Scenario 8: Cyclic supersession chain
    def test_scenario_08_cyclic_supersession(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S2-SUP-001",
            "title": "Superseded task A",
            "origin_sprint": "SPRINT_2",
            "assigned_sprint": "SPRINT_2",
            "priority": "P2",
            "workstream": "GOVERNANCE",
            "owner": "daakara",
            "status": "SUPERSEDED",
            "disposition": "SUPERSEDED",
            "dimensions": {
                "requirements": "DONE", "implementation": "NOT_APPLICABLE", "integration": "NOT_APPLICABLE",
                "verification": "NOT_APPLICABLE", "repository_publication": "NOT_APPLICABLE", "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "NOT_APPLICABLE"
            },
            "acceptance_criteria": [],
            "dependencies": [],
            "evidence_refs": ["ref"],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "SUPERSEDED",
            "operational_state": "SUPERSEDED",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": "ARX-S2-SUP-002"
        })
        data["tasks"].append({
            "task_id": "ARX-S2-SUP-002",
            "title": "Superseded task B",
            "origin_sprint": "SPRINT_2",
            "assigned_sprint": "SPRINT_2",
            "priority": "P2",
            "workstream": "GOVERNANCE",
            "owner": "daakara",
            "status": "SUPERSEDED",
            "disposition": "SUPERSEDED",
            "dimensions": {
                "requirements": "DONE", "implementation": "NOT_APPLICABLE", "integration": "NOT_APPLICABLE",
                "verification": "NOT_APPLICABLE", "repository_publication": "NOT_APPLICABLE", "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "NOT_APPLICABLE"
            },
            "acceptance_criteria": [],
            "dependencies": [],
            "evidence_refs": ["ref"],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "SUPERSEDED",
            "operational_state": "SUPERSEDED",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": "ARX-S2-SUP-001"  # Cycle!
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-008" and "Cyclic" in f.message for f in findings)

    # Scenario 9: False operational-complete claim (EXECUTED without production evidence)
    def test_scenario_09_false_operational_complete(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["dimensions"]["operational_execution"] = "EXECUTED"
        data["tasks"][0]["dimensions"]["production_evidence"] = "INSUFFICIENT"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-011" and "operational_execution" in f.message for f in findings)

    # Scenario 10: Restricted task incorrectly marked authorized
    def test_scenario_10_restricted_task_incorrectly_authorized(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["authorization_boundary"] = "ACTIVATION_HOLD"
        data["tasks"][0]["dimensions"]["owner_authorization"] = "AUTHORIZED"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "ACTIVATION_HOLD" in f.message for f in findings)

    # Scenario 11: Historical task silently removed
    def test_scenario_11_historical_task_silently_removed(self):
        baseline = {
            "tasks": [
                {"task_id": "ARX-S1-HIST-001", "status": "COMPLETED"}
            ]
        }
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)  # Does not include ARX-S1-HIST-001!
        validator = BacklogValidator(data, baseline_data=baseline)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-012" and "ARX-S1-HIST-001" in f.message for f in findings)

    # Scenario 12: Missing GitHub evidence source (CLI exit code test)
    def test_scenario_12_missing_source_file(self):
        import subprocess, sys
        cmd = [sys.executable, "tools/governance/validate_backlog_integrity.py", "--backlog", "nonexistent_file.json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2
        assert "SOURCE_UNAVAILABLE" in res.stderr

    # Scenario 13: Legitimate deferred owner-gated task passes
    def test_scenario_13_deferred_owner_gated_task_passes(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S3-GATE-001",
            "title": "Corporate Events Telemetry Live Activation",
            "origin_sprint": "SPRINT_3",
            "assigned_sprint": "HOLDING_QUEUE_OWNER_GATED",
            "priority": "P1",
            "workstream": "CORPORATE_EVENTS",
            "owner": "daakara",
            "status": "BLOCKED",
            "disposition": "SPRINT4_OWNER_GATED",
            "dimensions": {
                "requirements": "DONE", "implementation": "DONE", "integration": "DONE",
                "verification": "PASS", "repository_publication": "PUBLISHED_MERGED", "deployment": "DEPLOYED_DORMANT",
                "operational_execution": "NOT_PERFORMED", "production_evidence": "INSUFFICIENT",
                "owner_authorization": "HOLD"
            },
            "acceptance_criteria": ["Hold operational activation until Owner grants"],
            "dependencies": [],
            "evidence_refs": ["docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md"],
            "github_issues": [60],
            "github_prs": [53, 79],
            "authorization_boundary": "OWNER_GATE_REQUIRED",
            "operational_state": "DORMANT_HELD",
            "carryforward_reason": "Code merged; live telemetry held pending activation grant",
            "reopening_trigger": "Product Owner activation authorization",
            "historical_aliases": ["INITIATIVE-S3-CORP-EVENTS-RC5"],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"

    # Scenario 14: Legitimate completed task with accepted evidence passes
    def test_scenario_14_completed_task_with_evidence_passes(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S1-001",
            "title": "VCP Inactive Read-Safety Contract Enforcement",
            "origin_sprint": "SPRINT_1",
            "assigned_sprint": "SPRINT_1",
            "priority": "P0",
            "workstream": "RADAR_VCP",
            "owner": "ARX_CORE_TEAM",
            "status": "COMPLETED",
            "disposition": "CLOSED_VERIFIED",
            "dimensions": {
                "requirements": "DONE", "implementation": "DONE", "integration": "DONE",
                "verification": "PASS", "repository_publication": "PUBLISHED_MERGED", "deployment": "DEPLOYED_ACTIVE",
                "operational_execution": "EXECUTED", "production_evidence": "VERIFIED",
                "owner_authorization": "AUTHORIZED"
            },
            "acceptance_criteria": ["Eliminate scan-on-read execution"],
            "dependencies": [],
            "evidence_refs": ["https://github.com/daakara/finance/pull/4", "docs/sprints/SPRINT_1_CLOSEOUT.md"],
            "github_issues": [],
            "github_prs": [4],
            "authorization_boundary": "MERGED_MAINLINE",
            "operational_state": "ACTIVE_SERVING",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": ["BACKLOG-VCP-READ-SAFETY-001"],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"

    # Scenario 15: Normal canonical Sprint 4 intake backlog passes all rules
    def test_scenario_15_canonical_backlog_full_pass(self):
        backlog_path = "docs/planning/arx_canonical_backlog.json"
        with open(backlog_path, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
        validator = BacklogValidator(backlog_data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"
        assert len([f for f in findings if f.severity == "ERROR"]) == 0
