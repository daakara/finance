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
        Verifies that detect_scoped_governance_changes operates with full determinism in an
        isolated Git repository, satisfying all Gate B requirements:
        1. Clean checkout (HEAD == base): returns scoped=False, matched=[], and valid method string.
        2. Non-governance changes (negative case): frontend code PR returns scoped=False, matched=[].
        3. Modified governance files: returns scoped=True, identifies modified file, reports merge-base method.
        4. Added governance files: returns scoped=True, identifies added governance file.
        5. Return value contract: (bool, list, str).
        6. Working tree detection: staged and unstaged governance changes are accurately detected.
        7. Clean tear-down of temporary resources.
        """
        tmp_dir = tempfile.mkdtemp(prefix="arx_test21_")
        curr_dir = os.getcwd()
        try:
            # 1. Initialize deterministic base repository on main
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

            # 3. Clean checkout verification (HEAD == base): scoped=False
            try:
                os.chdir(tmp_dir)
                scoped_main, matched_main, method_main = detect_scoped_governance_changes(base_ref="main")
                assert scoped_main is False
                assert isinstance(scoped_main, bool)
                assert isinstance(matched_main, list)
                assert len(matched_main) == 0
                assert isinstance(method_main, str)
            finally:
                os.chdir(curr_dir)

            # 4. Negative case: non-governance / frontend-only changes return scoped=False
            subprocess.run(["git", "checkout", "-b", "feature/frontend-only"], cwd=tmp_dir, check=True, capture_output=True)
            fe_dir = os.path.join(tmp_dir, "frontend", "components")
            os.makedirs(fe_dir, exist_ok=True)
            fe_file = os.path.join(fe_dir, "Navbar.tsx")
            with open(fe_file, "w", encoding="utf-8") as f:
                f.write("// Unrelated frontend component\nexport const Navbar = () => null;\n")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "feat(frontend): add Navbar component"], cwd=tmp_dir, check=True)

            try:
                os.chdir(tmp_dir)
                scoped_fe, matched_fe, method_fe = detect_scoped_governance_changes(base_ref="main")
                assert scoped_fe is False
                assert len(matched_fe) == 0
                assert isinstance(method_fe, str)
            finally:
                os.chdir(curr_dir)

            # 5. Positive case 1: Modified governance file returns scoped=True
            subprocess.run(["git", "checkout", "-b", "feature/gov-modify", "main"], cwd=tmp_dir, check=True, capture_output=True)
            mod_data = copy.deepcopy(MINIMAL_VALID_BACKLOG)
            mod_data["tasks"][0]["title"] = "Updated Minimal Task"
            with open(gov_file, "w", encoding="utf-8") as f:
                json.dump(mod_data, f)
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "feat: update task title in canonical backlog"], cwd=tmp_dir, check=True)

            try:
                os.chdir(tmp_dir)
                scoped_mod, matched_mod, method_mod = detect_scoped_governance_changes(base_ref="main")
                assert scoped_mod is True
                assert any("arx_canonical_backlog.json" in f for f in matched_mod)
                assert "main" in method_mod

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

            # 6. Positive case 2: Added governance file returns scoped=True
            subprocess.run(["git", "checkout", "-b", "feature/gov-add", "main"], cwd=tmp_dir, check=True, capture_output=True)
            new_gov_file = os.path.join(tmp_dir, "docs", "planning", "ARX_SPRINT_4_REGISTRATION.md")
            with open(new_gov_file, "w", encoding="utf-8") as f:
                f.write("# Sprint 4 Registration Record\n")
            subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
            subprocess.run(["git", "commit", "-m", "docs: register sprint 4 planning record"], cwd=tmp_dir, check=True)

            try:
                os.chdir(tmp_dir)
                scoped_add, matched_add, method_add = detect_scoped_governance_changes(base_ref="main")
                assert scoped_add is True
                assert any("ARX_SPRINT_4_REGISTRATION.md" in f for f in matched_add)
                assert "main" in method_add
            finally:
                os.chdir(curr_dir)

            # 7. Positive case 3: Unstaged / working tree governance changes
            subprocess.run(["git", "checkout", "main"], cwd=tmp_dir, check=True, capture_output=True)
            with open(gov_file, "a", encoding="utf-8") as f:
                f.write("\n")  # Unstaged modification
            try:
                os.chdir(tmp_dir)
                scoped_wt, matched_wt, method_wt = detect_scoped_governance_changes(base_ref="main")
                assert scoped_wt is True
                assert any("arx_canonical_backlog.json" in f for f in matched_wt)
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

