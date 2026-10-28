"""
ARX TERMINAL - DIRECTIVE 061 ADVERSARIAL NEGATIVE CONTROLS SUITE
Validates pre-merge certification invariants, owner HOLD authority,
historical obligation preservation, and external closure gates.
"""

import copy
import json
import os
import pytest

from tools.governance.validate_backlog_integrity import (
    BacklogValidator,
    REQUIRED_IMMUTABLE_HOLDS,
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


def run_validator(data, pre_sprint5_closure=False, pre_sprint5_readiness=False, require_external=False):
    schema = load_canonical_schema()
    baseline = copy.deepcopy(data)
    val = BacklogValidator(
        backlog_data=data,
        schema_data=schema,
        baseline_data=baseline,
        pre_sprint5_closure=pre_sprint5_closure,
        pre_sprint5_readiness=pre_sprint5_readiness,
        require_external_inventory=require_external,
    )
    val.validate_all()
    return val


class TestDirective061AdversarialControls:
    """Adversarial negative controls mandated by Directive 061 Sections 5 and 7."""

    def test_adv_01_missing_original_sprint4_task_rejected(self):
        """Negative Control 1: Deleting an original Sprint 4 task must fail closure/readiness."""
        data = load_canonical_backlog()
        # Delete original task ARX-S4-ARCH-001
        data["tasks"] = [t for t in data["tasks"] if t.get("task_id") != "ARX-S4-ARCH-001"]
        val = run_validator(data, pre_sprint5_readiness=True)
        # Obligation for that task now has no parent in tasks
        errs = [f for f in val.findings if f.severity == "ERROR" and (f.rule_id == "P5-002" or "ARX-S4-ARCH-001" in f.message)]
        assert len(errs) > 0, "Deleting original Sprint 4 task must fail parent traceability (P5-002)"

    def test_adv_02_substituted_carryforward_lacks_owner_authorization(self):
        """Negative Control 2: Substituted carryforward lacking owner authorization must fail P5-004."""
        data = load_canonical_backlog()
        fake_obl = {
            "obligation_id": "OBL-FAKED-CARRYFORWARD-001",
            "canonical_task_id": "ARX-S4-RADAR-001",
            "origin_sprint": "SPRINT_4",
            "release_scope": "ALREADY_APPROVED_SPRINT5_DELIVERY",
            "scope_authorization": "UNAPPROVED_SUBSTITUTION",
            "acceptance_criteria": ["Some fake criteria"],
            "owner": "engineer-x",
            "status": "READY",
            "disposition": "SPRINT5_EXECUTION",
            "deployment_state": "NOT_DEPLOYED",
            "governance_state": "UNAUTHORIZED"
        }
        data["obligations"].append(fake_obl)
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-004" and "OBL-FAKED-CARRYFORWARD-001" in f.message]
        assert len(errs) > 0, "Substituted carryforward without OWNER_AUTHORIZED must fail P5-004"

    def test_adv_03_incorrectly_authorized_owner_hold_rejected(self):
        """Negative Control 3: Releasing an owner HOLD without valid owner authorization must fail P5-005."""
        data = load_canonical_backlog()
        for obl in data["obligations"]:
            if obl.get("canonical_task_id") == "ARX-S3-007":
                obl["status"] = "COMPLETED"
                obl["deployment_state"] = "DEPLOYED_ACTIVE"
                obl["deployment_refs"] = ["https://github.com/daakara/finance/commit/fake1234"]
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-005"]
        assert len(errs) > 0, "Falsely marking governance-held obligation active/completed must fail P5-005"

    def test_adv_04_tampered_immutable_operational_hold_rejected(self):
        """Negative Control 4: Mutating any of the 11 immutable operational holds must fail P5-005."""
        data = load_canonical_backlog()
        data["immutable_holds"]["SMART_MONEY_SEC_EGRESS"] = "ENABLED"
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-005" and "SMART_MONEY_SEC_EGRESS" in f.message]
        assert len(errs) > 0, "Mutating mandatory operational hold must fail P5-005"

    def test_adv_05_deleted_mandatory_pre_sprint5_obligation_rejected(self):
        """Negative Control 5: Deleting DEF-056-001 must fail cross-audit completeness (P5-009)."""
        data = load_canonical_backlog()
        data["obligations"] = [o for o in data["obligations"] if o.get("source_finding_id") != "DEF-056-001"]
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-009" and "DEF-056-001" in f.message]
        assert len(errs) > 0, "Deleting required cross-audit obligation DEF-056-001 must fail P5-009"

    def test_adv_06_fabricated_production_evidence_rejected(self):
        """Negative Control 6: Claiming DEPLOYED_ACTIVE with unpromoted/held deployment ref must fail P5-007."""
        data = load_canonical_backlog()
        for obl in data["obligations"]:
            if obl.get("obligation_id") == "OBL-DEF-056-001":
                obl["deployment_state"] = "DEPLOYED_ACTIVE"
                obl["deployment_refs"] = ["Railway deployment d1f8cf37 (RELEASE_HELD)"]
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-007" and "RELEASE_HELD" in f.message]
        assert len(errs) > 0, "Claiming DEPLOYED_ACTIVE with HELD deployment ref must fail P5-007"

    def test_adv_07_stale_baseline_sha_substitution_rejected(self):
        """Negative Control 7: Substituting stale unverified deployment SHA '5efc2179' must fail P5-011."""
        data = load_canonical_backlog()
        data["obligations"][0]["deployment_refs"] = ["commit 5efc2179 unverified"]
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-011"]
        assert len(errs) > 0, "Substituting stale unverified SHA 5efc2179 must fail P5-011"

    def test_adv_08_sprint5_active_without_owner_kickoff_rejected(self):
        """Negative Control 8: Setting active_sprint to SPRINT_5 without kickoff authorization must fail P5-010."""
        data = load_canonical_backlog()
        data["active_sprint"] = "SPRINT_5"
        data["sprint5_execution_authorized"] = False
        val = run_validator(data, pre_sprint5_readiness=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and f.rule_id == "P5-010"]
        assert len(errs) > 0, "Setting active_sprint: SPRINT_5 without affirmative authorization must fail P5-010"

    def test_adv_09_scoped_passes_while_closure_gates_fail_closed(self):
        """Negative Control 9: Scoped validator passes Gate 1, but Gate 2A and Gate 2B fail closed."""
        data = load_canonical_backlog()
        # 1. Scoped gate passes (PR merge eligible)
        val_scoped = run_validator(data)
        errs_scoped = [f for f in val_scoped.findings if f.severity == "ERROR"]
        assert len(errs_scoped) == 0, f"Scoped governance check must pass with 0 errors, got: {errs_scoped}"

        # 2. Gate 2A (Readiness) fails closed because 3 obligations are RELEASE_HELD
        val_readiness = run_validator(data, pre_sprint5_readiness=True)
        errs_readiness = [f for f in val_readiness.findings if f.severity == "ERROR" and f.rule_id == "P5-003"]
        assert len(errs_readiness) == 3, f"Gate 2A must report exactly 3 release-held blockers, got {len(errs_readiness)}"

        # 3. Gate 2B (Execution) fails closed because readiness is blocked and PO kickoff is missing
        val_closure = run_validator(data, pre_sprint5_closure=True)
        errs_closure = [f for f in val_closure.findings if f.severity == "ERROR"]
        assert len(errs_closure) == 4, f"Gate 2B must report exactly 4 blockers (3 release + 1 PO auth), got {len(errs_closure)}"

    def test_adv_10_missing_external_reconciliation_fails_closure_certification(self):
        """Negative Control 10: External reconciliation requirement must fail closed if inventory missing."""
        data = load_canonical_backlog()
        val = run_validator(data, pre_sprint5_closure=True, require_external=True)
        errs = [f for f in val.findings if f.severity == "ERROR" and (f.rule_id == "BI-007" or f.rule_id == "P5-012")]
        assert len(errs) > 0, "External reconciliation requirement must fail closed when inventory missing"
