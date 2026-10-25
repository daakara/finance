"""
Tests for ARX Terminal Smart Money
GOVERNANCE DECISION ENGINE POLICY V1
Mechanically validates policy encoding, 20 canonical rules, 19 approvals,
and deterministic replay fixtures.
"""

import os
import json
import hashlib
import subprocess
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLICY_DIR = os.path.join(PROJECT_ROOT, "docs", "domain", "smart_money", "external_authority")
POLICY_MD = os.path.join(POLICY_DIR, "GOVERNANCE_POLICY_V1.md")
POLICY_JSON = os.path.join(POLICY_DIR, "GOVERNANCE_POLICY_V1.json")
FIXTURES_JSON = os.path.join(POLICY_DIR, "GOVERNANCE_REPLAY_FIXTURES.json")


class TestGovernanceDecisionEnginePolicy:

    def test_01_policy_files_exist(self):
        assert os.path.isfile(POLICY_MD), f"Missing {POLICY_MD}"
        assert os.path.isfile(POLICY_JSON), f"Missing {POLICY_JSON}"
        assert os.path.isfile(FIXTURES_JSON), f"Missing {FIXTURES_JSON}"
        assert os.path.getsize(POLICY_MD) > 2000
        assert os.path.getsize(POLICY_JSON) > 2000

    def test_02_all_twenty_rules_present(self):
        with open(POLICY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = data.get("rules", {})
        assert len(rules) == 20, f"Expected exactly 20 rules, found {len(rules)}"

        for i in range(1, 21):
            rule_id = f"R-GOV-{i:03d}"
            assert rule_id in rules, f"Rule {rule_id} missing from policy JSON"

    def test_03_nineteen_approvals_registered_and_approved(self):
        with open(POLICY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        approvals = data.get("required_approvals_register", {})
        assert len(approvals) == 19, f"Expected 19 approvals, found {len(approvals)}"

        for k, v in approvals.items():
            assert v == "APPROVED", f"Approval {k} is not APPROVED: {v}"

    def test_04_non_waivable_defect_classes(self):
        with open(POLICY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        r3 = data["rules"]["R-GOV-003"]
        non_waivable = set(r3["non_waivable_defect_classes"])

        mandatory_classes = {
            "AUTHORITY_INTEGRITY_CONTRADICTION",
            "SECOND_GENESIS_VULNERABILITY",
            "CONSUMER_SELF_AUTHORIZATION_PATH",
            "EVIDENCE_MUTABILITY_VIOLATING_CUSTODY",
            "STATE_ROLLBACK_VULNERABILITY",
            "QUALIFICATION_TO_NATURAL_CONTAMINATION",
            "UNVERIFIABLE_AUTHORITY_BOUNDARY",
            "UNSUPPORTED_AUTHORITY_PROVENANCE",
        }
        assert mandatory_classes.issubset(non_waivable), f"Missing non-waivable classes: {mandatory_classes - non_waivable}"

    def test_05_unknown_propagation_matrix_complete(self):
        with open(POLICY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        r5 = data["rules"]["R-GOV-005"]["unknown_gate_matrix"]
        required_gates = [
            "ARCHITECTURE_CLOSURE",
            "GLOBAL_PROVISIONING_READINESS",
            "THEOREM_A_PROVISIONING",
            "THEOREM_B_PROVISIONING",
            "THEOREM_C_PROVISIONING",
            "THEOREM_D_PROVISIONING",
            "THEOREM_A_QUALIFICATION",
            "THEOREM_B_QUALIFICATION",
            "THEOREM_C_QUALIFICATION",
            "THEOREM_D_QUALIFICATION",
            "THEOREM_E_EXECUTION",
            "PRODUCTION_ARMING",
            "NATURAL_OBSERVER_ARMING",
        ]
        for g in required_gates:
            assert g in r5, f"Gate {g} missing from UNKNOWN propagation matrix"

    def test_06_lifecycle_transitions_enforce_no_direct_open_to_verified(self):
        with open(POLICY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        r10 = data["rules"]["R-GOV-010"]
        allowed = r10["allowed_lifecycle_transitions"]
        assert "VERIFIED_RESOLVED" not in allowed["OPEN"], "Direct OPEN -> VERIFIED_RESOLVED transition must be forbidden"
        assert "FIXED_PENDING_VERIFICATION" in allowed["IN_REMEDIATION"]
        assert "VERIFIED_RESOLVED" in allowed["FIXED_PENDING_VERIFICATION"]

    def test_07_replay_fixtures_pass(self):
        with open(FIXTURES_JSON, "r", encoding="utf-8") as f:
            fixtures_data = json.load(f)

        fixtures = fixtures_data["fixtures"]
        assert len(fixtures) >= 10, "Expected at least 10 replay fixtures"

        # Verify JCS-SHA256 canonicalization fixture
        jcs_fixture = [fix for fix in fixtures if fix["rule_id"] == "R-GOV-018"][0]
        raw_obj = jcs_fixture["input"]["snapshot"]
        canonical_str = json.dumps(raw_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert canonical_str == jcs_fixture["expected_canonical_json"]
        assert hashlib.sha256(canonical_str.encode("utf-8")).hexdigest() == jcs_fixture["expected_sha256"]

    def test_08_candidate004_immutability(self):
        freeze_path = os.path.join(PROJECT_ROOT, "docs", "domain", "vcp", "sprint_3", "candidates", "CANDIDATE_GENERATION_004_FREEZE.json")
        with open(freeze_path, "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90"

        diff = subprocess.check_output(["git", "diff", "--stat", freeze_path], cwd=PROJECT_ROOT, text=True).strip()
        assert diff == ""
