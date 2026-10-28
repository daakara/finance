"""
ARX TERMINAL - DIRECTIVE 062 ADVERSARIAL NEGATIVE CONTROLS SUITE
Targeted tests verifying that the governance validator strictly rejects:
1. A ninth falsely authorized carryforward.
2. A substituted historical task title or ID where external baseline validation is required.
3. A superseded PR treated as active execution scope.
4. A PR merge reference treated as owner authorization.
5. An ARX-S3-006 HOLD bypass.
6. An unsupported claim that external reconciliation has been performed.
7. A premature active_sprint = SPRINT_5 transition.
"""

import copy
import json
import os
import pytest

from tools.governance.validate_backlog_integrity import (
    BacklogValidator,
    Finding,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
WORKTREE_DIR = os.path.dirname(TEST_DIR)
CANONICAL_BACKLOG_PATH = os.path.join(WORKTREE_DIR, "docs", "planning", "arx_canonical_backlog.json")
CANONICAL_SCHEMA_PATH = os.path.join(WORKTREE_DIR, "docs", "planning", "arx_canonical_backlog.schema.json")


def load_canonical_backlog():
    with open(CANONICAL_BACKLOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_canonical_schema():
    with open(CANONICAL_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_validator(data, baseline=None, pre_sprint5_closure=False, pre_sprint5_readiness=False, require_external=False, reconciliation_status="NOT_PERFORMED"):
    schema = load_canonical_schema()
    if baseline is None:
        baseline = copy.deepcopy(data)
    val = BacklogValidator(
        backlog_data=data,
        schema_data=schema,
        baseline_data=baseline,
        pre_sprint5_closure=pre_sprint5_closure,
        pre_sprint5_readiness=pre_sprint5_readiness,
        require_external_inventory=require_external,
    )
    val.reconciliation_status = reconciliation_status
    val.validate_all()
    return val


class TestDirective062AdversarialControls:
    """Directive 062 Section 8 Negative Controls."""

    def test_reject_ninth_falsely_authorized_carryforward(self):
        """1. A ninth falsely authorized carryforward must be rejected."""
        data = load_canonical_backlog()
        # Attempt to falsely authorize a 9th carryforward from Sprint 4 without valid owner authorization
        for obl in data["obligations"]:
            if obl.get("obligation_id") == "OBL-PR-109":
                obl["origin_sprint"] = "SPRINT_4"
                obl["release_scope"] = "SPRINT_5"
                obl["scope_authorization"] = "UNAUTHORIZED_EXPANSION"
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-004" and "OBL-PR-109" in f.message]
        assert len(errs) > 0, "Falsely authorizing a 9th carryforward without OWNER_AUTHORIZED must fail P5-004"

    def test_reject_substituted_historical_task_title_or_id(self):
        """2. A substituted historical task title or ID where external baseline validation is required must be rejected."""
        data = load_canonical_backlog()
        baseline = copy.deepcopy(data)
        # Substitute an authoritative task ID ARX-S2-007 with a fabricated ID ARX-S4-DATA-002
        for t in data["tasks"]:
            if t.get("task_id") == "ARX-S2-007":
                t["task_id"] = "ARX-S4-DATA-002"
                t["title"] = "TimescaleDB Continuous Aggregates"
        val = run_validator(data, baseline=baseline, pre_sprint5_readiness=True)
        # Baseline check BI-012 or parent check P5-002 must trigger
        errs = [f for f in val.findings if f.severity == "ERROR" and (f.rule_id in ("BI-012", "P5-002"))]
        assert len(errs) > 0, "Substituting authoritative task ID must fail historical continuity (BI-012 or P5-002)"

    def test_reject_superseded_pr_treated_as_active_execution_scope(self):
        """3. A superseded PR treated as active execution scope must be rejected."""
        data = load_canonical_backlog()
        # Take superseded obligation OBL-PR-114 and assign it to active Sprint 5 execution
        for obl in data["obligations"]:
            if obl.get("obligation_id") == "OBL-PR-114":
                obl["origin_sprint"] = "SPRINT_4"
                obl["release_scope"] = "SPRINT_5"
                obl["scope_authorization"] = "UNAUTHORIZED_EXECUTION"
                obl["status"] = "READY"
                obl["disposition"] = "SPRINT5_EXECUTION"
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-004" and "OBL-PR-114" in f.message]
        assert len(errs) > 0, "Treating superseded PR as active execution scope without owner approval must fail P5-004"

    def test_reject_pr_merge_reference_treated_as_owner_authorization(self):
        """4. A PR merge reference treated as owner authorization must be rejected."""
        data = load_canonical_backlog()
        # For ARX-S3-006 (held task), adding a merge commit reference (8b8eb58) while attempting
        # to claim CLOSED_VERIFIED on a RELEASE_HELD or NOT_DEPLOYED item must be rejected
        for obl in data["obligations"]:
            if obl.get("canonical_task_id") == "ARX-S3-006":
                obl["commit_refs"] = ["8b8eb58296adbb6374be254cd9a00c36c59786c6"]
                obl["deployment_state"] = "NOT_DEPLOYED"
                obl["disposition"] = "CLOSED_VERIFIED"
                obl["status"] = "COMPLETED"
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-007"]
        assert len(errs) > 0, "Claiming CLOSED_VERIFIED on NOT_DEPLOYED task using PR merge commit must fail P5-007"

    def test_reject_arx_s3_006_hold_bypass(self):
        """5. An ARX-S3-006 HOLD bypass must be rejected."""
        data = load_canonical_backlog()
        # Attempt to mark the held task ARX-S3-006 as completed/closed without owner authorization
        for t in data["tasks"]:
            if t.get("task_id") == "ARX-S3-006":
                t["status"] = "COMPLETED"
                t["disposition"] = "CLOSED_VERIFIED"
                t["dimensions"]["owner_authorization"] = "HOLD"
        val = run_validator(data)
        # Rule BI-009 strictly enforces that tasks cannot be marked COMPLETED/CLOSED_VERIFIED while owner_authorization is HOLD
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "BI-009" and "ARX-S3-006" in f.message]
        assert len(errs) > 0, "Marking ARX-S3-006 COMPLETED while on HOLD must fail BI-009"

    def test_reject_unsupported_claim_external_reconciliation_performed(self):
        """6. An unsupported claim that external reconciliation has been performed must be rejected."""
        data = load_canonical_backlog()
        # Validator running with require_external_inventory=True when reconciliation_status is NOT_PERFORMED
        val = run_validator(data, pre_sprint5_readiness=True, require_external=True, reconciliation_status="NOT_PERFORMED")
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-012"]
        assert len(errs) > 0, "Claiming pre-sprint5 closure without verified external reconciliation must fail P5-012"

    def test_reject_premature_active_sprint_sprint5_transition(self):
        """7. A premature active_sprint = SPRINT_5 transition must be rejected."""
        data = load_canonical_backlog()
        data["active_sprint"] = "SPRINT_5"
        data["sprint5_execution_authorized"] = False
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-010"]
        assert len(errs) > 0, "Setting active_sprint = SPRINT_5 without affirmative owner authorization must fail P5-010"
