"""
ARX Terminal — Cross-Sprint Backlog Integrity Validator Test Suite (GOV-001)
Validates Rules BI-001 through BI-012, GitHub record reconciliation invariants,
sprint closure integrity, and adversarial edge cases.
"""

import copy
import json
import os
import subprocess
import sys
import pytest

from tools.governance.validate_backlog_integrity import (
    BacklogValidator,
    Finding,
    classify_github_issue_record,
    reconcile_github_inventory,
)

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
            "description": "Cross-sprint backlog integrity framework implementation.",
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
    """Automated acceptance and regression tests for GOV-001."""

    # 1. Duplicate task IDs
    def test_01_duplicate_task_ids(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        dup = copy.deepcopy(data["tasks"][0])
        data["tasks"].append(dup)
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-001" and "Duplicate" in f.message for f in findings)

    # 2. Missing owners
    def test_02_missing_owners(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["owner"] = "OWNER_UNASSIGNED"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-002" and "no accountable owner" in f.message for f in findings)

    # 3. Missing sprint assignments
    def test_03_missing_sprint_assignments(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["assigned_sprint"] = "SPRINT_INVALID"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-003" for f in findings)

    # 4. Missing closure dispositions
    def test_04_missing_closure_dispositions(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["disposition"] = "INVALID_DISPOSITION"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-004" for f in findings)

    # 5. Invalid carry-forward destinations
    def test_05_invalid_carryforward_destinations(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S2-TEST-002",
            "title": "Carried forward without reason",
            "description": "Test task",
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
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-005" for f in findings)

    # 6. Missing evidence
    def test_06_missing_evidence(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["disposition"] = "CLOSED_VERIFIED"
        data["tasks"][0]["status"] = "COMPLETED"
        data["tasks"][0]["evidence_refs"] = []
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-006" for f in findings)

    # 7. Orphaned PR references
    def test_07_orphaned_pr_references(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        prs_inventory = {"open_prs": [101, 999]}  # PR 999 is unlinked
        validator = BacklogValidator(data, prs_inventory=prs_inventory)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS_WITH_WARNINGS"
        assert any(f.rule_id == "BI-007" and "PR #999" in f.message for f in findings)

    # 8. GitHub issue/PR classification
    def test_08_github_issue_pr_classification(self):
        issue_record = {"number": 60, "title": "Corporate Events RC5 Fixes"}
        pr_record = {"number": 73, "title": "Minervini VCP", "pull_request": {"url": "https://..."}}

        assert classify_github_issue_record(issue_record) == "ISSUE"
        assert classify_github_issue_record(pr_record) == "PULL_REQUEST"

    # 9. GitHub pagination
    def test_09_github_pagination(self):
        page1 = [{"number": i, "title": f"Issue {i}"} for i in range(1, 11)]
        page2 = [{"number": i, "title": f"Issue {i}"} for i in range(11, 21)]
        page2.append({"number": 10, "title": "Duplicate Issue 10"})  # Overlapping duplicate

        reconciled = reconcile_github_inventory(page1 + page2, [])
        assert reconciled["genuine_issues_count"] == 20
        assert reconciled["total_records_count"] == 20

    # 10. Partial API failures
    def test_10_partial_api_failures(self):
        issues = [{"number": 60, "title": "Genuine Issue"}]
        pulls = []  # /pulls returned empty or failed
        reconciled = reconcile_github_inventory(issues, pulls)
        assert reconciled["genuine_issues_count"] == 1
        assert reconciled["pull_requests_count"] == 0
        # Does not wipe out known genuine issues

    # 11. Supersession cycles
    def test_11_supersession_cycles(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S2-SUP-001",
            "title": "Superseded task A",
            "description": "Task A",
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
            "description": "Task B",
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
            "superseded_by": "ARX-S2-SUP-001"
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-008" and "Cyclic" in f.message for f in findings)

    # 12. Deleted historical tasks
    def test_12_deleted_historical_tasks(self):
        baseline = {
            "tasks": [
                {"task_id": "ARX-S1-HIST-001", "status": "COMPLETED"}
            ]
        }
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        validator = BacklogValidator(data, baseline_data=baseline)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-012" and "ARX-S1-HIST-001" in f.message for f in findings)

    # 13. Invalid operational-complete states
    def test_13_invalid_operational_complete_states(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["dimensions"]["operational_execution"] = "EXECUTED"
        data["tasks"][0]["dimensions"]["production_evidence"] = "INSUFFICIENT"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-011" and "operational_execution" in f.message for f in findings)

    # 14. Owner-gated tasks incorrectly marked authorized
    def test_14_owner_gated_tasks_incorrectly_authorized(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"][0]["authorization_boundary"] = "ACTIVATION_HOLD"
        data["tasks"][0]["dimensions"]["owner_authorization"] = "AUTHORIZED"
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "ACTIVATION_HOLD" in f.message for f in findings)

    # 15. Legitimately deferred tasks
    def test_15_legitimately_deferred_tasks(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S3-GATE-001",
            "title": "Corporate Events Telemetry Live Activation",
            "description": "Held live telemetry task",
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

    # 16. Legitimately completed and verified tasks
    def test_16_legitimately_completed_and_verified_tasks(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S1-001",
            "title": "VCP Inactive Read-Safety Contract Enforcement",
            "description": "VCP safety contract",
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

    # 17. Sprint closure with controlled carry-forward
    def test_17_sprint_closure_with_controlled_carryforward(self):
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        # SPRINT_3 closed task properly carried forward to SPRINT_4
        data["tasks"].append({
            "task_id": "ARX-S3-CF-001",
            "title": "Sprint 3 Carried Forward Task",
            "description": "Controlled carry-forward item",
            "origin_sprint": "SPRINT_3",
            "assigned_sprint": "SPRINT_4",
            "priority": "P1",
            "workstream": "ARCHITECTURE",
            "owner": "daakara",
            "status": "READY",
            "disposition": "SPRINT4_EXECUTION",
            "dimensions": {
                "requirements": "DONE", "implementation": "IN_PROGRESS", "integration": "NOT_STARTED",
                "verification": "PENDING", "repository_publication": "LOCAL_ONLY", "deployment": "NOT_APPLICABLE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "AUTHORIZED"
            },
            "acceptance_criteria": ["Execute in Sprint 4"],
            "dependencies": [],
            "evidence_refs": ["ARX_SPRINT3_FINAL_CLOSURE_REGISTER.md"],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "BOUNDED_ENGINEERING",
            "operational_state": "QUEUED",
            "carryforward_reason": "Sprint 3 closed with controlled carryforward to Sprint 4",
            "reopening_trigger": "Sprint 4 Primary Workstream Launch",
            "historical_aliases": [],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"

    # 18. Malformed or corrupted register files
    def test_18_malformed_or_corrupted_register_files(self):
        # 1. Nonexistent file
        cmd = [sys.executable, "tools/governance/validate_backlog_integrity.py", "--backlog", "nonexistent_file.json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2
        assert "SOURCE_UNAVAILABLE" in res.stderr

        # 2. Non-list tasks
        corrupted = {"schema_version": "1.0.0", "tasks": "invalid_string_not_list"}
        validator = BacklogValidator(corrupted)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-001" for f in findings)

    # 19. Regression test reproducing the earlier incorrect 0 issues / 11 PRs conclusion
    def test_19_regression_zero_issues_misclassification(self):
        """
        Reproduces the defect where all records returned by GitHub were treated as PRs,
        resulting in '0 issues / 11 PRs'.
        Verifies that the invariant classification properly identifies the 13 genuine issues
        (#58 through #70) and 11 PRs.
        """
        raw_github_issues_endpoint_sample = [
            {"number": 58, "title": "test"},
            {"number": 59, "title": "test"},
            {"number": 60, "title": "Corporate Events RC5 Fixes"},
            {"number": 61, "title": "Radar Error vs Zero Classification"},
            {"number": 62, "title": "Candidate Truth Contract"},
            {"number": 63, "title": "Decision-First UX Alignment"},
            {"number": 64, "title": "VCP Boundary Containment"},
            {"number": 65, "title": "Research Supply Chain Sigstore"},
            {"number": 66, "title": "Human Pilot P00"},
            {"number": 67, "title": "Pre-Flight & Smart Money Phase 1"},
            {"number": 68, "title": "Air Lease Corp (ALC) Valuation Reproduction"},
            {"number": 69, "title": "DECK Undervaluation & Parity Audit"},
            {"number": 70, "title": "Minervini VCP Pattern Analysis"},
            # PRs embedded in /issues with pull_request key
            {"number": 38, "title": "Mobile Nav Dock", "pull_request": {"url": "..."}},
            {"number": 40, "title": "Candle Chart", "pull_request": {"url": "..."}},
            {"number": 42, "title": "S2 Accounting", "pull_request": {"url": "..."}},
            {"number": 44, "title": "S208 Accounting", "pull_request": {"url": "..."}},
            {"number": 47, "title": "S2 RC1", "pull_request": {"url": "..."}},
            {"number": 72, "title": "Pre-Flight Copy", "pull_request": {"url": "..."}},
            {"number": 73, "title": "VCP Activation", "pull_request": {"url": "..."}},
            {"number": 75, "title": "Supply Chain", "pull_request": {"url": "..."}},
            {"number": 78, "title": "HVA01 Wave 4B", "pull_request": {"url": "..."}},
            {"number": 83, "title": "HVA01 Wave 4C", "pull_request": {"url": "..."}},
            {"number": 98, "title": "Synthesis E V3", "pull_request": {"url": "..."}},
        ]

        # Naïve buggy logic: treat everything as PR if repo issues tab is disabled or unchecked
        def naive_buggy_classifier(records):
            prs = [r for r in records if "pull_request" in r]
            issues = []  # Bug: assumed 0 genuine issues
            return len(issues), len(prs)

        buggy_issues_cnt, buggy_prs_cnt = naive_buggy_classifier(raw_github_issues_endpoint_sample)
        assert buggy_issues_cnt == 0  # Demonstrates reproducing the bug!

        # Correct invariant logic
        reconciled = reconcile_github_inventory(raw_github_issues_endpoint_sample, [])
        assert reconciled["genuine_issues_count"] == 13
        assert reconciled["pull_requests_count"] == 11
        assert reconciled["total_records_count"] == 24
        assert set(reconciled["genuine_issues"].keys()) == set(range(58, 71))

    # 20. Canonical backlog passes all rules
    def test_20_canonical_backlog_full_pass(self):
        backlog_path = "docs/planning/arx_canonical_backlog.json"
        with open(backlog_path, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
        validator = BacklogValidator(backlog_data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"
        assert len([f for f in findings if f.severity == "ERROR"]) == 0
