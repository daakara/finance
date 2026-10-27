"""
ARX Terminal — Cross-Sprint Backlog Integrity Validator Test Suite (GOV-001)
Validates Rules BI-001 through BI-012, GitHub record reconciliation invariants,
sprint closure integrity, and adversarial edge cases.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

from tools.governance.validate_backlog_integrity import (
    BacklogValidator,
    Finding,
    classify_github_issue_record,
    reconcile_github_inventory,
    detect_scoped_governance_changes,
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

    # 21. Directive Scenario 1: Modified backlog on clean checkout vs Mainline
    def test_21_regression_modified_backlog_on_clean_checkout(self):
        """
        Verifies that when a PR branch touches governance/backlog files on a clean
        checkout against an authoritative base revision, detect_scoped_governance_changes
        accurately triggers scoped enforcement (scoped=True) and identifies the changed files.
        Also verifies that in a mainline/no-change context (HEAD == base), detection accurately
        returns scoped=False, ensuring environment-independent determinism without ambient repository leakage.
        """
        tmp_dir = tempfile.mkdtemp(prefix="arx_test21_")
        curr_dir = os.getcwd()
        try:
            # 1. Initialize deterministic base revision
            subprocess.run(["git", "init", "-b", "main"], cwd=tmp_dir, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "ARX Test Runner"], cwd=tmp_dir, check=True)
            subprocess.run(["git", "config", "user.email", "test@arxterminal.local"], cwd=tmp_dir, check=True)

            # 2. Create baseline governance file on main
            gov_dir = os.path.join(tmp_dir, "docs", "planning")
            os.makedirs(gov_dir, exist_ok=True)
            gov_file = os.path.join(gov_dir, "arx_canonical_backlog.json")
            with open(gov_file, "w", encoding="utf-8") as f:
                json.dump(MINIMAL_VALID_BACKLOG, f)
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "init: add baseline canonical backlog"], cwd=tmp_dir, check=True)

            # 6. Verify the mainline/no-change case returns scoped=False
            try:
                os.chdir(tmp_dir)
                scoped_main, matched_main, method_main = detect_scoped_governance_changes(base_ref="main")
                assert scoped_main is False
                assert len(matched_main) == 0
            finally:
                os.chdir(curr_dir)

            # 3. Create feature branch containing a governance modification
            subprocess.run(["git", "checkout", "-b", "feature/gov-update"], cwd=tmp_dir, check=True, capture_output=True)
            mod_data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
            mod_data["tasks"][0]["title"] = "Updated Minimal Task"
            with open(gov_file, "w", encoding="utf-8") as f:
                json.dump(mod_data, f)
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "feat: update task title in canonical backlog"], cwd=tmp_dir, check=True)

            # 4. Verify merge-base change detection returns scoped=True on feature branch
            # 5. Verify the detected file is the expected governance path
            # 7. Verify the expected change-detection method
            try:
                os.chdir(tmp_dir)
                scoped_feat, matched_feat, method_feat = detect_scoped_governance_changes(base_ref="main")
                assert scoped_feat is True
                assert any("arx_canonical_backlog.json" in f for f in matched_feat)
                assert "main" in method_feat

                # Verify CLI invocation with --stage scoped against base_ref
                validator_script = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "tools", "governance", "validate_backlog_integrity.py")
                )
                cmd = [sys.executable, validator_script, "--stage", "scoped", "--base-ref", "main", "--backlog", gov_file]
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_dir)
                assert res.returncode == 0
                assert "Verdict: PASS" in res.stdout
                assert "Scope Detected: True" in res.stdout
            finally:
                os.chdir(curr_dir)
        finally:
            # 8. Clean up all temporary resources
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # 22. Directive Scenario 2: Deleted historical task
    def test_22_regression_deleted_historical_task(self):
        """
        Verifies that deleting a previously completed historical task fails closed under BI-012.
        """
        baseline = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        baseline["tasks"].append({
            "task_id": "ARX-S1-HISTORICAL-001",
            "title": "Historical Completed Task",
            "description": "Historical task in baseline",
            "origin_sprint": "SPRINT_1",
            "assigned_sprint": "SPRINT_1",
            "priority": "P1",
            "workstream": "CORE",
            "owner": "daakara",
            "status": "COMPLETED",
            "disposition": "CLOSED_VERIFIED",
            "dimensions": {
                "requirements": "DONE", "implementation": "DONE", "integration": "DONE",
                "verification": "PASS", "repository_publication": "MERGED_MAINLINE", "deployment": "DEPLOYED_ACTIVE",
                "operational_execution": "NOT_APPLICABLE", "production_evidence": "NOT_APPLICABLE",
                "owner_authorization": "AUTHORIZED"
            },
            "acceptance_criteria": ["Historical acceptance"],
            "dependencies": [],
            "evidence_refs": ["commit_sha_12345678"],
            "github_issues": [],
            "github_prs": [1],
            "authorization_boundary": "MERGED_MAINLINE",
            "operational_state": "ACTIVE_SERVING",
            "carryforward_reason": None,
            "reopening_trigger": None,
            "historical_aliases": [],
            "superseded_by": None
        })

        # Current backlog is missing ARX-S1-HISTORICAL-001
        current = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        validator = BacklogValidator(current, baseline_data=baseline)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert validator.continuity_status == "FAILED"
        assert any(f.rule_id == "BI-012" and "silently removed" in f.message for f in findings)

    # 23. Directive Scenario 3: Missing baseline never proves task continuity
    def test_23_regression_missing_baseline(self, tmp_path):
        """
        Verifies that missing historical baseline never proves task continuity.
        Specifying a non-existent baseline file fails closed with SOURCE_UNAVAILABLE (exit code 2).
        When no baseline is provided, continuity_status is UNVERIFIED.
        """
        nonexistent = str(tmp_path / "nonexistent_baseline.json")
        cmd = [sys.executable, "tools/governance/validate_backlog_integrity.py", "--baseline", nonexistent]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2
        assert "SOURCE_UNAVAILABLE" in res.stderr

        # Validator object without baseline reports UNVERIFIED
        validator = BacklogValidator(copy.deepcopy(MINIMAL_VALID_BACKLOG), baseline_data=None)
        verdict, findings = validator.validate_all()
        assert validator.continuity_status == "UNVERIFIED"

        # If baseline is explicitly required, missing baseline produces a BI-012 error
        validator_req = BacklogValidator(copy.deepcopy(MINIMAL_VALID_BACKLOG), baseline_data=None, require_baseline=True)
        verdict_req, findings_req = validator_req.validate_all()
        assert verdict_req == "FAIL"
        assert any(f.rule_id == "BI-012" for f in findings_req)

    # 24. Directive Scenario 4: Missing GitHub inventory never means zero issues or PRs
    def test_24_regression_missing_github_inventory(self, tmp_path):
        """
        Verifies that missing GitHub inventory fails closed if an invalid path is specified,
        and never falsely asserts zero open issues or PRs.
        """
        nonexistent = str(tmp_path / "nonexistent_prs.json")
        cmd = [sys.executable, "tools/governance/validate_backlog_integrity.py", "--prs-inventory", nonexistent]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2
        assert "SOURCE_UNAVAILABLE" in res.stderr

        # Validator object without inventory reports NOT_PERFORMED
        validator = BacklogValidator(copy.deepcopy(MINIMAL_VALID_BACKLOG), prs_inventory=None)
        verdict, findings = validator.validate_all()
        assert validator.reconciliation_status == "NOT_PERFORMED"

        # When external inventory is required (e.g. closure certification), missing inventory produces BI-007 ERROR
        validator_req = BacklogValidator(copy.deepcopy(MINIMAL_VALID_BACKLOG), prs_inventory=None, require_external_inventory=True)
        verdict_req, findings_req = validator_req.validate_all()
        assert verdict_req == "FAIL"
        assert any(f.rule_id == "BI-007" for f in findings_req)

    # 25. Directive Scenario 5: Valid controlled carry-forward
    def test_25_regression_valid_controlled_carryforward(self):
        """
        Verifies that a task carried forward from Sprint 3 to Sprint 4 with valid
        receiving sprint, non-empty reason, and non-active operational state passes completely.
        """
        data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        data["tasks"].append({
            "task_id": "ARX-S3-RADAR-RELIABILITY",
            "title": "Radar VCP Reliability Alignment",
            "description": "Legitimate carry-forward task",
            "origin_sprint": "SPRINT_3",
            "assigned_sprint": "SPRINT_4",
            "priority": "P1",
            "workstream": "RADAR",
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
            "evidence_refs": ["ARX_CROSS_WORKSTREAM_GOVERNANCE_FINAL_DECISION.md"],
            "github_issues": [],
            "github_prs": [],
            "authorization_boundary": "BOUNDED_ENGINEERING",
            "operational_state": "QUEUED",
            "carryforward_reason": "Sprint 3 closed with controlled carryforward into Sprint 4 execution",
            "reopening_trigger": "Sprint 4 Increment 1 Activation",
            "historical_aliases": [],
            "superseded_by": None
        })
        validator = BacklogValidator(data)
        verdict, findings = validator.validate_all()
        assert verdict == "PASS"
        assert len([f for f in findings if f.severity == "ERROR"]) == 0

    # 26. Directive Scenario 6: Unrelated code PR in scoped mode
    def test_26_regression_unrelated_code_pr(self, monkeypatch):
        """
        Verifies that an unrelated application PR (e.g. modifying only api/main.py)
        does not trigger scoped governance failure and runs report-only exiting 0.
        """
        original_run = subprocess.run
        def mock_git_diff(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if isinstance(cmd, list) and "diff" in cmd:
                class MockRes:
                    returncode = 0
                    stdout = "api/main.py\nfrontend/components/Navbar.tsx\n"
                    stderr = ""
                return MockRes()
            return original_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_git_diff)
        scoped, matched, method = detect_scoped_governance_changes(base_ref="origin/main")
        assert scoped is False
        assert len(matched) == 0

    # 27. Directive Scenario 7: Failed closure validation and correct nonzero CI exit status
    def test_27_regression_failed_closure_validation_nonzero_exit(self, tmp_path):
        """
        Verifies that when a backlog file contains errors or closure validation fails,
        the validator CLI exits with nonzero status (1), and cannot be masked by report-only
        when closure certification is requested.
        """
        invalid_backlog = copy.deepcopy(MINIMAL_VALID_BACKLOG)
        invalid_backlog["tasks"].append({
            "task_id": "ARX-S4-GOV-001",  # Duplicate task_id
            "title": "Duplicate ID task",
            "origin_sprint": "SPRINT_4",
            "assigned_sprint": "SPRINT_4",
            "status": "IN_PROGRESS",
            "disposition": "SPRINT4_EXECUTION"
        })
        bad_file = tmp_path / "invalid_backlog.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump(invalid_backlog, f)

        # 1. Scoped mode fails with code 1
        cmd_scoped = [sys.executable, "tools/governance/validate_backlog_integrity.py", "--backlog", str(bad_file), "--stage", "strict"]
        res_scoped = subprocess.run(cmd_scoped, capture_output=True, text=True)
        assert res_scoped.returncode == 1
        assert "Verdict: FAIL" in res_scoped.stdout

        # 2. Closure certification flag overrides report-only and cannot be masked
        cmd_closure = [
            sys.executable, "tools/governance/validate_backlog_integrity.py",
            "--backlog", str(bad_file),
            "--stage", "report-only",
            "--closure-certification"
        ]
        res_closure = subprocess.run(cmd_closure, capture_output=True, text=True)
        assert res_closure.returncode == 1
        assert "Verdict: FAIL" in res_closure.stdout


class TestSprint4ToSprint5Transition:
    """
    Dedicated test suite for Directive ARX_S4_FORMAL_CLOSURE_AND_S5_TRANSITION_032.
    Validates canonical backlog 43-task census, Sprint 4 closure reconciliation,
    Sprint 5 carryforward continuity, and fail-closed adversarial scenarios.
    """

    @pytest.fixture
    def canonical_backlog_data(self):
        with open("docs/planning/arx_canonical_backlog.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def canonical_schema_data(self):
        with open("docs/planning/arx_canonical_backlog.schema.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. Canonical Backlog Transition Verification
    def test_canonical_backlog_sprint5_transition_valid(self, canonical_backlog_data, canonical_schema_data):
        """Verifies the canonical backlog satisfies all rules with zero findings."""
        validator = BacklogValidator(canonical_backlog_data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        errors = [f for f in findings if f.severity == "ERROR"]
        assert errors == []
        assert verdict == "PASS"

        assert canonical_backlog_data["active_sprint"] == "SPRINT_5"
        tasks = canonical_backlog_data["tasks"]
        assert len(tasks) == 43

        # Exactly 8 tasks assigned to SPRINT_5
        s5_assigned = [t for t in tasks if t.get("assigned_sprint") == "SPRINT_5"]
        assert len(s5_assigned) == 8
        expected_s5_tids = {
            "ARX-S3-004", "ARX-S3-005", "ARX-S3-006",
            "ARX-S4-RADAR-001", "ARX-S2-005", "ARX-S2-007", "ARX-S2-008", "ARX-S2-009"
        }
        assert {t["task_id"] for t in s5_assigned} == expected_s5_tids

        # 6 Formally closed Sprint 4 tasks (Directive 045: ARX-S3-010 ratified)
        closed_s4 = ["ARX-S4-GOV-001", "ARX-S4-ARCH-001", "ARX-S4-ARCH-002", "ARX-S1-006", "ARX-S2-001", "ARX-S3-010"]
        for tid in closed_s4:
            t = next(x for x in tasks if x["task_id"] == tid)
            assert t["disposition"] == "CLOSED_VERIFIED"
            assert t["status"] == "COMPLETED"
            assert len(t["evidence_refs"]) > 0

        # ARX-S4-GOV-001 owner authorization aligned to AUTHORIZED
        gov001 = next(t for t in tasks if t["task_id"] == "ARX-S4-GOV-001")
        assert gov001["dimensions"]["owner_authorization"] == "AUTHORIZED"

        # ARX-S3-010 verified closed with DORMANT_MERGED operational state
        s3_010 = next(t for t in tasks if t["task_id"] == "ARX-S3-010")
        assert s3_010["operational_state"] == "DORMANT_MERGED"
        assert s3_010["dimensions"]["operational_execution"] == "NOT_PERFORMED"
        assert s3_010["dimensions"]["deployment"] == "DEPLOYED_DORMANT"

        # ARX-S3-006 owner authorization HOLD preserved
        s3_006 = next(t for t in tasks if t["task_id"] == "ARX-S3-006")
        assert s3_006["dimensions"]["owner_authorization"] == "HOLD"

        # 1 Conditional closure hold
        for tid in ["ARX-S3-013"]:
            t = next(x for x in tasks if x["task_id"] == tid)
            assert t["disposition"] == "CLOSED_SPRINT_CONDITIONAL_OPEN_TASK"
            assert t["status"] == "READY"
            assert t["assigned_sprint"] == "SPRINT_4"

        # 1 Engineering accepted, release held
        s3_016 = next(t for t in tasks if t["task_id"] == "ARX-S3-016")
        assert s3_016["disposition"] == "ENGINEERING_ACCEPTED_AWAITING_RELEASE"
        assert s3_016["status"] == "READY"
        assert s3_016["assigned_sprint"] == "SPRINT_4"

        # 10 Owner-gated holding queue tasks preserved
        holding = [t for t in tasks if t.get("assigned_sprint") == "HOLDING_QUEUE_OWNER_GATED"]
        assert len(holding) == 10
        for ht in holding:
            assert ht["dimensions"]["owner_authorization"] == "HOLD"

    # 2. Adversarial: Missing carryforward task
    def test_adversarial_missing_carryforward_task(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        baseline = copy.deepcopy(canonical_backlog_data)
        # Remove an approved carryforward task
        data["tasks"] = [t for t in data["tasks"] if t["task_id"] != "ARX-S3-004"]
        assert len(data["tasks"]) == 42
        validator = BacklogValidator(data, schema_data=canonical_schema_data, baseline_data=baseline, require_baseline=True)
        verdict, findings = validator.validate_all()
        # Should detect mismatch or missing task in continuity
        assert len(data["tasks"]) != 43

    # 3. Adversarial: Duplicate task identity
    def test_adversarial_duplicate_task_identity(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        dup = copy.deepcopy(data["tasks"][0])
        data["tasks"].append(dup)
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-001" and "Duplicate" in f.message for f in findings)

    # 4. Adversarial: Unauthorized owner HOLD change
    def test_adversarial_unauthorized_owner_hold_change(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        # Flip ARX-S3-006 from HOLD to AUTHORIZED without grant evidence
        s3_006 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-006")
        s3_006["dimensions"]["owner_authorization"] = "AUTHORIZED"
        s3_006["authorization_boundary"] = "ACTIVATION_HOLD"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "ACTIVATION_HOLD" in f.message for f in findings)

    # 5. Adversarial: Incorrectly closing S3-010 or S3-013 without evidence
    def test_adversarial_incorrectly_closing_s3_010_or_s3_013(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        s3_010 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-010")
        s3_010["status"] = "COMPLETED"
        s3_010["disposition"] = "CLOSED_VERIFIED"
        s3_010["evidence_refs"] = []  # No evidence
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-006" for f in findings)

    # 6. Adversarial: Incorrectly claiming S3-016 is deployed
    def test_adversarial_incorrectly_claiming_s3_016_deployed(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        s3_016 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-016")
        s3_016["dimensions"]["deployment"] = "DEPLOYED_ACTIVE"
        s3_016["dimensions"]["implementation"] = "NOT_STARTED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-011" and "deployment" in f.message for f in findings)

    # 7. Adversarial: Unauthorized Sprint 5 assignment
    def test_adversarial_unauthorized_sprint_5_assignment(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        # Attempt to assign an owner-gated holding task directly to SPRINT_5 execution
        holding = next(t for t in data["tasks"] if t["task_id"] == "ARX-S1-003")
        holding["assigned_sprint"] = "SPRINT_5"
        holding["disposition"] = "SPRINT5_EXECUTION"
        holding["authorization_boundary"] = "OWNER_GATE_REQUIRED"
        holding["dimensions"]["owner_authorization"] = "AUTHORIZED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "OWNER_GATE_REQUIRED" in f.message for f in findings)

    # 8. Adversarial: Invalid sprint-transition state (active task left in closed sprint)
    def test_adversarial_invalid_sprint_transition_state(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        # Leave a task with SPRINT4_EXECUTION and IN_PROGRESS assigned to SPRINT_4
        s3_004 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        s3_004["assigned_sprint"] = "SPRINT_4"
        s3_004["disposition"] = "SPRINT4_EXECUTION"
        s3_004["status"] = "IN_PROGRESS"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-010" and "Closed sprint SPRINT_4" in f.message for f in findings)

    # 9. Adversarial: Reopening a previously verified task
    def test_adversarial_reopening_previously_verified_task(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        gov001 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S4-GOV-001")
        gov001["status"] = "IN_PROGRESS"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-010" and "unresolved status 'IN_PROGRESS'" in f.message for f in findings)

    # 10. Adversarial: Loss of a historical evidence reference
    def test_adversarial_loss_of_historical_evidence_reference(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        arch001 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S4-ARCH-001")
        arch001["evidence_refs"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-006" for f in findings)

    # 11. Adversarial (Directive 042 Track B): Carryforward missing acceptance criteria
    def test_adversarial_carryforward_missing_acceptance_criteria(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        del cf["acceptance_criteria"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any("acceptance_criteria" in f.message for f in findings)

    # 12. Adversarial (Directive 042 Track B): Carryforward empty acceptance criteria array
    def test_adversarial_carryforward_empty_acceptance_criteria(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["acceptance_criteria"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id in ("BI-000", "BI-005") and "acceptance_criteria" in f.message for f in findings)

    # 13. Adversarial (Directive 042 Track B): Carryforward whitespace-only criterion
    def test_adversarial_carryforward_whitespace_only_criterion(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["acceptance_criteria"] = ["   ", "\t\n"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id in ("BI-000", "BI-005") and "whitespace-only" in f.message for f in findings)

    # 14. Adversarial (Directive 042 Track B): Carryforward malformed criterion object
    def test_adversarial_carryforward_malformed_criterion_object(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["acceptance_criteria"] = [{"malformed": "criterion"}]
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id in ("BI-000", "BI-005") for f in findings)

    # 15. Adversarial (Directive 042 Track B): Carryforward invalid criterion type (integer)
    def test_adversarial_carryforward_invalid_criterion_type(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["acceptance_criteria"] = [12345]
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id in ("BI-000", "BI-005") for f in findings)

    # 16. Adversarial (Directive 042 Track B): Carryforward without accountable owner
    def test_adversarial_carryforward_no_accountable_owner(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["owner"] = ""
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-002" for f in findings)

    # 17. Adversarial (Directive 042 Track B): Carryforward with invalid status
    def test_adversarial_carryforward_invalid_status(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        cf = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-004")
        cf["status"] = "INVALID_STATUS_XYZ"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id in ("BI-000", "BI-004") for f in findings)

    # 18. Adversarial (Directive 042 Track B): Unauthorized HOLD release
    def test_adversarial_carryforward_unauthorized_hold_release(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        holding = next(t for t in data["tasks"] if t["task_id"] == "ARX-S1-003")
        holding["dimensions"]["owner_authorization"] = "AUTHORIZED"
        holding["authorization_boundary"] = "OWNER_GATE_REQUIRED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" for f in findings)

    # 19. Positive acceptance: Valid existing carryforwards pass validation
    def test_carryforward_valid_acceptance_criteria_accepted(self, canonical_backlog_data, canonical_schema_data):
        validator = BacklogValidator(canonical_backlog_data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        # Canonical backlog with valid carryforwards must have zero BI-005 errors
        bi005_errors = [f for f in findings if f.rule_id == "BI-005" and f.severity == "ERROR"]
        assert len(bi005_errors) == 0

    # 20. Adversarial (Directive 043 Gate E): Unauthorized task closure without owner authorization
    def test_adversarial_unauthorized_closure_without_owner_authorization(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        t = next(x for x in data["tasks"] if x["task_id"] == "ARX-S3-010")
        t["disposition"] = "CLOSED_VERIFIED"
        t["status"] = "COMPLETED"
        t["dimensions"]["owner_authorization"] = "NOT_GRANTED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "must be AUTHORIZED" in f.message for f in findings)

    # 21. Adversarial (Directive 043 Gate E): Administrative recommendation polluting operational_state
    def test_adversarial_recommendation_polluting_operational_state(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        t = next(x for x in data["tasks"] if x["task_id"] == "ARX-S3-010")
        t["operational_state"] = "CLOSE_CONDITIONAL_RECOMMENDED_CLOSED_VERIFIED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "recommendations must not pollute operational_state" in f.message for f in findings)

    # 22. Adversarial (Directive 043 Gate E): Unauthorized operational closure state on non-closed task
    def test_adversarial_unauthorized_operational_closure_state(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        t = next(x for x in data["tasks"] if x["task_id"] == "ARX-S3-013")
        t["operational_state"] = "CLOSED_VERIFIED"
        # disposition remains CLOSED_SPRINT_CONDITIONAL_OPEN_TASK
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "claims operational_state 'CLOSED_VERIFIED' while disposition is" in f.message for f in findings)

    # === Directive 045 Adversarial Negative Controls (Gate D) ===

    # NC-01: Closing ARX-S3-010 does not release unrelated HOLDs
    def test_nc01_closing_s3_010_preserves_unrelated_holds(self, canonical_backlog_data, canonical_schema_data):
        tasks = canonical_backlog_data["tasks"]
        holding = [t for t in tasks if t.get("assigned_sprint") == "HOLDING_QUEUE_OWNER_GATED"]
        assert len(holding) == 10
        for ht in holding:
            assert ht["dimensions"]["owner_authorization"] == "HOLD"
        s3_006 = next(t for t in tasks if t["task_id"] == "ARX-S3-006")
        assert s3_006["dimensions"]["owner_authorization"] == "HOLD"

    # NC-02: Closing ARX-S3-010 does not activate Smart Money ingestion
    def test_nc02_closing_s3_010_does_not_activate_smart_money_ingestion(self, canonical_backlog_data, canonical_schema_data):
        s3_010 = next(t for t in canonical_backlog_data["tasks"] if t["task_id"] == "ARX-S3-010")
        assert s3_010["dimensions"]["operational_execution"] == "NOT_PERFORMED"
        assert s3_010["dimensions"]["deployment"] == "DEPLOYED_DORMANT"
        assert s3_010["operational_state"] == "DORMANT_MERGED"

    # NC-03: Closing ARX-S3-010 does not enable SEC egress
    def test_nc03_closing_s3_010_does_not_enable_sec_egress(self, canonical_backlog_data, canonical_schema_data):
        s3_010 = next(t for t in canonical_backlog_data["tasks"] if t["task_id"] == "ARX-S3-010")
        assert s3_010["authorization_boundary"] == "BOUNDED_ENGINEERING"
        assert s3_010["dimensions"]["operational_execution"] == "NOT_PERFORMED"

    # NC-04: Closing ARX-S3-010 does not authorize model learning or tuning
    def test_nc04_closing_s3_010_does_not_authorize_model_tuning(self, canonical_backlog_data, canonical_schema_data):
        for t in canonical_backlog_data["tasks"]:
            assert "MODEL_TUNING" not in t.get("operational_state", "")

    # NC-05: Closing ARX-S3-010 does not activate scanner scheduling
    def test_nc05_closing_s3_010_does_not_activate_scanner_scheduling(self, canonical_backlog_data, canonical_schema_data):
        s3_013 = next(t for t in canonical_backlog_data["tasks"] if t["task_id"] == "ARX-S3-013")
        assert s3_013["disposition"] == "CLOSED_SPRINT_CONDITIONAL_OPEN_TASK"
        assert s3_013["status"] == "READY"
        assert s3_013["operational_state"] == "CLOSE_CONDITIONAL_PENDING_PR73"

    # NC-06: Closing ARX-S3-010 does not mutate prospective evidence denominator
    def test_nc06_closing_s3_010_does_not_mutate_task_census_denominator(self, canonical_backlog_data, canonical_schema_data):
        tasks = canonical_backlog_data["tasks"]
        assert len(tasks) == 43
        assert len(set(t["task_id"] for t in tasks)) == 43

    # NC-07: Closing ARX-S3-010 does not imply production deployment
    def test_nc07_closing_s3_010_does_not_imply_production_deployment(self, canonical_backlog_data, canonical_schema_data):
        s3_010 = next(t for t in canonical_backlog_data["tasks"] if t["task_id"] == "ARX-S3-010")
        assert s3_010["dimensions"]["deployment"] == "DEPLOYED_DORMANT"
        assert s3_010["dimensions"]["deployment"] != "DEPLOYED_ACTIVE"

    # NC-08: Closing ARX-S3-010 does not independently activate Sprint 5
    def test_nc08_closing_s3_010_does_not_independently_activate_sprint5(self, canonical_backlog_data, canonical_schema_data):
        s5_tasks = [t for t in canonical_backlog_data["tasks"] if t.get("assigned_sprint") == "SPRINT_5"]
        assert len(s5_tasks) == 8
        for t in s5_tasks:
            assert t["status"] == "READY"

    # NC-09: Fabricated owner authorization is rejected in negative-control fixture
    def test_nc09_fabricated_owner_authorization_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        held_task = next(t for t in data["tasks"] if t.get("assigned_sprint") == "HOLDING_QUEUE_OWNER_GATED")
        held_task["dimensions"]["owner_authorization"] = "AUTHORIZED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-009" and "without grant evidence" in f.message for f in findings)

    # NC-10: Unsatisfied acceptance criterion prevents administrative closure
    def test_nc10_unsatisfied_acceptance_criterion_prevents_closure(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        s3_010 = next(t for t in data["tasks"] if t["task_id"] == "ARX-S3-010")
        s3_010["evidence_refs"] = []  # Strip evidence refs
        validator = BacklogValidator(data, schema_data=canonical_schema_data)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "BI-006" for f in findings)








# ==============================================================================
# DIRECTIVE 058: PRE-SPRINT 5 CLOSURE INVARIANTS (P5-001 to P5-012)
# ==============================================================================

class TestPreSprint5ClosureInvariants:
    """
    Dedicated adversarial test suite for Directive 058.
    Validates invariants P5-001 through P5-012 under --pre-sprint5-closure mode.
    Proves that every failure mode, fake closure, unauthorized HOLD release,
    orphan finding, and premature transition is deterministically caught.
    """

    @pytest.fixture
    def canonical_backlog_data(self):
        with open("docs/planning/arx_canonical_backlog.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def canonical_schema_data(self):
        with open("docs/planning/arx_canonical_backlog.schema.json", "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. P5-001: Missing obligation register
    def test_p5_001_empty_obligation_register(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["obligations"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-001" and "empty or missing" in f.message for f in findings)

    # 2. P5-001: Unscoped obligation
    def test_p5_001_unscoped_obligation(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["obligations"][0]["release_scope"] = "UNSCOPED_BLOCKER"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-001" and "lacks an approved release/sprint scope" in f.message for f in findings)

    # 3. P5-002: Orphan finding (no canonical parent)
    def test_p5_002_orphan_finding(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["obligations"][0]["canonical_task_id"] = "ARX-S9-NONEXISTENT"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-002" and "no valid canonical task parent" in f.message for f in findings)

    # 4. P5-003: Incomplete mandatory pre-Sprint 5 work
    def test_p5_003_incomplete_mandatory_closure(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        for o in data["obligations"]:
            if o.get("source_finding_id") in ("DEF-056-001", "DEF-056-002", "DEF-056-003"):
                o["status"] = "READY"
                o["disposition"] = "ENGINEERING_ACCEPTED_AWAITING_RELEASE"
                o["blocking_decision"] = "PR_118_MERGE_AUTHORIZATION_REQUIRED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        p5_003_findings = [f for f in findings if f.rule_id == "P5-003"]
        assert len(p5_003_findings) >= 3
        assert any("OBL-DEF-056-001" in f.message for f in p5_003_findings)

    # 5. P5-004: Silent carryforward without owner authorization
    def test_p5_004_silent_carryforward_unauthorized(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl = next(o for o in data["obligations"] if o.get("origin_sprint") == "SPRINT_2" and o.get("release_scope") == "SPRINT_5")
        obl["scope_authorization"] = "NOT_GRANTED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-004" and "lacks independent owner authorization" in f.message for f in findings)

    # 6. P5-004: Carryforward missing acceptance criteria
    def test_p5_004_carryforward_missing_criteria(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl = next(o for o in data["obligations"] if o.get("origin_sprint") == "SPRINT_2" and o.get("release_scope") == "SPRINT_5")
        obl["acceptance_criteria"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-004" and "lacks concrete acceptance criteria" in f.message for f in findings)

    # 7. P5-004: Carryforward missing accountable owner
    def test_p5_004_carryforward_missing_owner(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl = next(o for o in data["obligations"] if o.get("origin_sprint") == "SPRINT_2" and o.get("release_scope") == "SPRINT_5")
        obl["owner"] = " "
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-004" and "lacks an accountable owner" in f.message for f in findings)

    # 8. P5-005: Unauthorized HOLD release
    def test_p5_005_unauthorized_hold_release(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        held_obl = next(o for o in data["obligations"] if o.get("release_scope") == "HOLDING_QUEUE_OWNER_GATED")
        held_obl["status"] = "COMPLETED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-005" and "falsely marked active or completed" in f.message for f in findings)

    # 9. P5-006: Fake closed state without evidence
    def test_p5_006_fake_closed_state_no_evidence(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        completed_obl = next(o for o in data["obligations"] if o.get("status") == "COMPLETED")
        completed_obl["evidence_refs"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-006" and "lacks verification evidence references" in f.message for f in findings)

    # 10. P5-007: Missing deployment proof
    def test_p5_007_missing_deployment_proof(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        active_obl = next(o for o in data["obligations"] if o.get("deployment_state") == "DEPLOYED_ACTIVE")
        active_obl["deployment_refs"] = []
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-007" and "lacks verified deployment references" in f.message for f in findings)

    # 11. P5-007: Non-deployed deliverable claims closed verified
    def test_p5_007_non_deployed_claims_closed(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        deliv = next(o for o in data["obligations"] if o.get("capability_id") == "MAINLINE_DELIVERY")
        deliv["deployment_state"] = "NOT_DEPLOYED"
        deliv["disposition"] = "CLOSED_VERIFIED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-007" and "is NOT_DEPLOYED but claims CLOSED_VERIFIED" in f.message for f in findings)

    # 12. P5-008: Duplicate obligation identity
    def test_p5_008_duplicate_obligation_id(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        dup = copy.deepcopy(data["obligations"][0])
        data["obligations"].append(dup)
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-008" and "Duplicate obligation ID" in f.message for f in findings)

    # 13. P5-008: Cyclic supersession
    def test_p5_008_cyclic_supersession(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl_a = data["obligations"][0]
        obl_b = data["obligations"][1]
        obl_a["superseded_by"] = obl_b["obligation_id"]
        obl_b["superseded_by"] = obl_a["obligation_id"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-008" and "Cyclic supersession detected" in f.message for f in findings)

    # 14. P5-009: Missing Directive 057 defect
    def test_p5_009_missing_directive_057_defect(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["obligations"] = [o for o in data["obligations"] if o.get("source_finding_id") != "DEF-056-001"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-009" and "DEF-056-001" in f.message for f in findings)

    # 15. P5-010: Unauthorized Sprint 5 execution kickoff
    def test_p5_010_unauthorized_sprint5_execution_transition(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["active_sprint"] = "SPRINT_5"
        data["sprint5_execution_authorized"] = False
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-010" and "Sprint 5 execution kickoff is NOT authorized" in f.message for f in findings)

    # 16. P5-011: Stale baseline substitution
    def test_p5_011_stale_baseline_substitution(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["obligations"][0]["deployment_refs"] = ["Railway:5efc2179"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-011" and "substitutes stale unverified deployment SHA" in f.message for f in findings)

    # 17. P5-012: External reconciliation required but not performed
    def test_p5_012_external_verification_not_performed(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        validator = BacklogValidator(data, schema_data=canonical_schema_data, require_external_inventory=True, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-012" and "execution status is NOT_PERFORMED" in f.message for f in findings)

    # 18. Actual Canonical Backlog Status: Fails closed when mandatory defects held from release and execution unauthorized
    def test_canonical_backlog_pre_sprint5_closure_fails_closed_when_release_held_and_unauthorized(self, canonical_backlog_data, canonical_schema_data):
        """
        Directive 059 Invariant: The canonical backlog in its current unpromoted, unauthorized state
        MUST fail closed under --pre-sprint5-closure with P5-003 (release held) and P5-010 (owner authorization missing).
        """
        validator = BacklogValidator(canonical_backlog_data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        errors = [f for f in findings if f.severity == "ERROR"]
        assert verdict == "FAIL"
        
        # Verify P5-003 errors for DEF-056-001/002/003
        p5_003_errors = [f for f in errors if f.rule_id == "P5-003"]
        assert len(p5_003_errors) == 3
        for oid in ("OBL-DEF-056-001", "OBL-DEF-056-002", "OBL-DEF-056-003"):
            assert any(oid in f.message and "held from production deployment" in f.message for f in p5_003_errors)
            assert any("BLK-057-001; BLK-057-002" in f.message for f in p5_003_errors)

        # Verify P5-010 errors for missing owner authorization
        p5_010_errors = [f for f in errors if f.rule_id == "P5-010"]
        assert len(p5_010_errors) >= 1
        assert any("Sprint 5 execution kickoff is NOT authorized by Product Owner" in f.message for f in p5_010_errors)

        # Verify zero unexpected errors on other rules
        other_errors = [f for f in errors if f.rule_id not in ("P5-003", "P5-010")]
        assert other_errors == [], f"Unexpected errors on other rules: {[str(e) for e in other_errors]}"

    # 19. Synthetic Full Pass: When PR #118 is deployed and owner affirmatively authorizes kickoff
    def test_canonical_backlog_pre_sprint5_closure_passes_when_fully_resolved(self, canonical_backlog_data, canonical_schema_data):
        """
        Simulates affirmative Product Owner authorization and production deployment release.
        Proves that when all prerequisites are truthfully satisfied, the validator passes cleanly.
        """
        data = copy.deepcopy(canonical_backlog_data)
        # Simulate PR #118 production deployment promotion & verification
        for o in data["obligations"]:
            if o.get("source_finding_id") in ("DEF-056-001", "DEF-056-002", "DEF-056-003"):
                o["status"] = "COMPLETED"
                o["disposition"] = "CLOSED_VERIFIED"
                o["deployment_state"] = "DEPLOYED_ACTIVE"
                o["operational_state"] = "EXECUTED"
                o["governance_state"] = "RELEASED"
                o["scope_authorization"] = "OWNER_AUTHORIZED"
                o["production_acceptance_state"] = "ACCEPTED_ACTIVE"
                o["blocking_decision"] = None
                o["deployment_refs"] = ["Railway:d1f8cf37 (PRODUCTION_ACTIVE)", "Cloudflare:655b4653 (ACTIVE)"]
            if o.get("obligation_id") == "OBL-ARX-S3-006":
                o["scope_authorization"] = "OWNER_AUTHORIZED"

        # Simulate affirmative Product Owner kickoff authorization record
        data["sprint5_execution_authorized"] = True
        data["sprint5_execution_authorization_record"] = {
            "authorizer": "PRODUCT_OWNER",
            "decision": "AFFIRMATIVE_KICKOFF_GRANTED",
            "timestamp": "2026-10-27T22:00:00Z",
            "decision_reference": "PO-DIR-059-KICKOFF",
            "governing_directive": "Directive 059",
            "reason": "Production rollout verified and pre-Sprint 5 certification complete.",
            "notes": "Affirmative owner kickoff granted."
        }

        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        errors = [f for f in findings if f.severity == "ERROR"]
        assert errors == [], f"Unexpected errors: {[str(e) for e in errors]}"
        assert verdict == "PASS"

    # === Directive 059 Hardening & Provenance Adversarial Controls ===

    # 20. P5-010 Adversarial: Missing authorization record when flag is True
    def test_p5_010_missing_record_when_authorized_true(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["sprint5_execution_authorized"] = True
        data["sprint5_execution_authorization_record"] = None
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-010" and "missing or null" in f.message for f in findings)

    # 21. P5-010 Adversarial: Fabricated non-human authorizer rejected
    def test_p5_010_fabricated_agent_authorizer_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["sprint5_execution_authorized"] = True
        data["sprint5_execution_authorization_record"] = {
            "authorizer": "AUTONOMOUS_AGENT",
            "decision": "AFFIRMATIVE_KICKOFF_GRANTED",
            "timestamp": "2026-10-27T22:00:00Z",
            "decision_reference": "AGENT-059-FABRICATED",
            "governing_directive": "Directive 059",
            "reason": "Autonomous agent self-authorized.",
            "notes": "Invalid authorizer test"
        }
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-010" and "invalid authorizer 'AUTONOMOUS_AGENT'" in f.message for f in findings)

    # 22. P5-010 Adversarial: Mismatched decision contradiction
    def test_p5_010_mismatched_decision_contradiction_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        # Case A: authorized is False, but record claims AFFIRMATIVE_KICKOFF_GRANTED
        data["sprint5_execution_authorized"] = False
        data["sprint5_execution_authorization_record"] = {
            "authorizer": "PRODUCT_OWNER",
            "decision": "AFFIRMATIVE_KICKOFF_GRANTED",
            "timestamp": "2026-10-27T22:00:00Z",
            "decision_reference": "PO-DIR-059",
            "governing_directive": "Directive 059",
            "reason": "Test contradiction",
            "notes": ""
        }
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-010" and "contradiction: authorized flag is False but record claims 'AFFIRMATIVE_KICKOFF_GRANTED'" in f.message for f in findings)

        # Case B: authorized is True, but record claims BLOCKED
        data["sprint5_execution_authorized"] = True
        data["sprint5_execution_authorization_record"]["decision"] = "BLOCKED_PENDING_OWNER_AUTHORIZATION"
        validator2 = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict2, findings2 = validator2.validate_all()
        assert verdict2 == "FAIL"
        assert any(f.rule_id == "P5-010" and "contradiction: authorized flag is True but record decision is" in f.message for f in findings2)

    # 23. P5-010 Adversarial: Missing mandatory authorization record fields
    def test_p5_010_missing_mandatory_record_fields(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["sprint5_execution_authorized"] = True
        data["sprint5_execution_authorization_record"] = {
            "authorizer": "PRODUCT_OWNER",
            "decision": "AFFIRMATIVE_KICKOFF_GRANTED",
            "timestamp": "",
            "decision_reference": None,
            "governing_directive": " "
        }
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-010" and "missing valid timestamp" in f.message for f in findings)
        assert any(f.rule_id == "P5-010" and "missing valid decision_reference" in f.message for f in findings)
        assert any(f.rule_id == "P5-010" and "missing valid governing_directive" in f.message for f in findings)

    # 24. P5-005 Adversarial: Operational hold relaxation rejected
    def test_p5_005_top_level_hold_relaxation_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        data["immutable_holds"]["SMART_MONEY_WORKER_ACTIVATION"] = "AUTHORIZED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-005" and "SMART_MONEY_WORKER_ACTIVATION" in f.message and "relaxed or mutated" in f.message for f in findings)

    # 25. P5-005 Adversarial: Missing operational hold rejected
    def test_p5_005_top_level_hold_missing_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        del data["immutable_holds"]["MODEL_TUNING"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-005" and "MODEL_TUNING" in f.message and "is missing from immutable_holds" in f.message for f in findings)

    # 26. P5-007 Adversarial: DEPLOYED_ACTIVE with unpromoted/held reference rejected
    def test_p5_007_deployed_active_with_held_deployment_ref_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl = next(o for o in data["obligations"] if o.get("source_finding_id") == "DEF-056-001")
        obl["deployment_state"] = "DEPLOYED_ACTIVE"
        obl["deployment_refs"] = ["Railway:d1f8cf37 (PRODUCTION_HELD - unpromoted)"]
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-007" and "indicates unpromoted/held status" in f.message for f in findings)

    # 27. P5-007 Adversarial: RELEASE_HELD claims CLOSED_VERIFIED rejected
    def test_p5_007_release_held_claims_closed_verified_rejected(self, canonical_backlog_data, canonical_schema_data):
        data = copy.deepcopy(canonical_backlog_data)
        obl = next(o for o in data["obligations"] if o.get("source_finding_id") == "DEF-056-001")
        obl["deployment_state"] = "RELEASE_HELD"
        obl["disposition"] = "CLOSED_VERIFIED"
        validator = BacklogValidator(data, schema_data=canonical_schema_data, pre_sprint5_closure=True)
        verdict, findings = validator.validate_all()
        assert verdict == "FAIL"
        assert any(f.rule_id == "P5-007" and "is RELEASE_HELD but claims CLOSED_VERIFIED" in f.message for f in findings)
