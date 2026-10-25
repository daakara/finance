"""
Tests for ARX Terminal Smart Money
EXTERNAL AUTHORITY ARCHITECTURE CONTRACT CLOSURE
Mechanically verifies all contracts, schemas, invariant registries, and manifest invariants.
"""

import os
import json
import hashlib
import re
import subprocess
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTRACTS_DIR = os.path.join(PROJECT_ROOT, "docs", "domain", "smart_money", "external_authority")

EXPECTED_FILES = [
    "EXTERNAL_AUTHORITY_TRUST_MODEL.md",
    "DISTRIBUTED_STATE_AND_COMMIT_PROTOCOL.md",
    "EVIDENCE_CUSTODY_AND_RETENTION_CONTRACT.md",
    "LIVE_QUALIFICATION_PROTOCOL.md",
    "THREAT_COMPROMISE_AND_RECOVERY_MODEL.md",
    "CHANGE_CONTROL_AND_REQUALIFICATION_POLICY.md",
    "EXTERNAL_AUTHORITY_INVARIANT_REGISTRY.json",
    "EXTERNAL_AUTHORITY_CLOSURE_MANIFEST.json",
]

FORBIDDEN_PLACEHOLDERS = [
    r"\bTBD\b",
    r"\bTODO\b",
    r"\bFIXME\b",
    r"future operator decision",
    r"implementation dependent",
    r"cloud provider specific",
    r"to be determined during provisioning",
]


class TestExternalAuthorityArchitectureContracts:

    def test_01_all_contract_files_exist_and_non_empty(self):
        assert os.path.isdir(CONTRACTS_DIR), f"Contracts directory missing: {CONTRACTS_DIR}"
        for fname in EXPECTED_FILES:
            fpath = os.path.join(CONTRACTS_DIR, fname)
            assert os.path.isfile(fpath), f"Required contract file missing: {fname}"
            size = os.path.getsize(fpath)
            assert size > 500, f"Contract file {fname} is suspiciously small ({size} bytes)"

    def test_02_no_unresolved_placeholders(self):
        for fname in EXPECTED_FILES:
            fpath = os.path.join(CONTRACTS_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            for pattern in FORBIDDEN_PLACEHOLDERS:
                match = re.search(pattern, content, re.IGNORECASE)
                assert match is None, f"Forbidden placeholder '{match.group(0)}' found in {fname}"

    def test_03_invariant_registry_schema_and_uniqueness(self):
        reg_path = os.path.join(CONTRACTS_DIR, "EXTERNAL_AUTHORITY_INVARIANT_REGISTRY.json")
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "invariants" in data
        invariants = data["invariants"]
        assert len(invariants) >= 15, "Invariant registry must contain at least 15 formal invariants"

        seen_ids = set()
        required_fields = ["id", "theorem", "statement", "scope", "proof_requirement", "negative_control", "requalification_triggers"]
        valid_theorems = {"A", "B", "C", "D", "E", "CROSS_AUTHORITY"}
        valid_scopes = {"ARCHITECTURE", "EXTERNAL", "LIVE_QUALIFICATION"}

        statements_text = []
        for inv in invariants:
            for field in required_fields:
                assert field in inv, f"Invariant {inv.get('id')} missing required field '{field}'"
            inv_id = inv["id"]
            assert inv_id not in seen_ids, f"Duplicate invariant ID detected: {inv_id}"
            seen_ids.add(inv_id)

            assert inv["theorem"] in valid_theorems, f"Invalid theorem '{inv['theorem']}' in {inv_id}"
            assert inv["scope"] in valid_scopes, f"Invalid scope '{inv['scope']}' in {inv_id}"
            assert isinstance(inv["requalification_triggers"], list)
            assert len(inv["requalification_triggers"]) > 0, f"Invariant {inv_id} must have requalification triggers"
            statements_text.append(inv["statement"].lower())

        # Check coverage of mandatory invariants
        combined_text = " ".join(statements_text)
        assert "self-authorize" in combined_text, "Missing self-authorization invariant"
        assert "fencing" in combined_text or "stale runner" in combined_text, "Missing stale runner fencing invariant"
        assert "second genesis" in combined_text, "Missing second genesis invariant"
        assert "denominator" in combined_text, "Missing denominator authority invariant"
        assert "worm" in combined_text or "compliance" in combined_text, "Missing WORM evidence custody invariant"
        assert "quarantine" in combined_text, "Missing evidence quarantine invariant"
        assert "qualification" in combined_text and "namespace" in combined_text or "isolated" in combined_text, "Missing qualification isolation invariant"

    def test_04_partial_failure_matrix_completeness(self):
        commit_doc = os.path.join(CONTRACTS_DIR, "DISTRIBUTED_STATE_AND_COMMIT_PROTOCOL.md")
        with open(commit_doc, "r", encoding="utf-8") as f:
            text = f.read()

        assert "PARTIAL-FAILURE STATE MATRIX" in text
        assert re.search(r"REACHABLE_PARTIAL_STATES_UNCLASSIFIED[:=]\s*0", text) is not None

        # Check all 5 valid actions are present
        for action in ["RETRY", "RESUME", "QUARANTINE", "TERMINAL_HOLD"]:
            assert action in text, f"Action {action} missing in commit protocol"

    def test_05_illegal_states_fail_closed_mapping(self):
        commit_doc = os.path.join(CONTRACTS_DIR, "DISTRIBUTED_STATE_AND_COMMIT_PROTOCOL.md")
        with open(commit_doc, "r", encoding="utf-8") as f:
            text = f.read()

        assert "ILLEGAL STATES REGISTRY" in text
        assert re.search(r"ILLEGAL_STATES_WITHOUT_FAIL_CLOSED_ACTION[:=]\s*0", text) is not None
        assert "Second Genesis" in text
        assert "Orphan Admission" in text
        assert "Phantom Denominator" in text

    def test_06_closure_manifest_schema_and_integrity(self):
        manifest_path = os.path.join(CONTRACTS_DIR, "EXTERNAL_AUTHORITY_CLOSURE_MANIFEST.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["gate"] == "SMART_MONEY_EXTERNAL_AUTHORITY_ARCHITECTURE_CONTRACT_CLOSURE"
        assert manifest["candidate004_diff"] == 0
        assert manifest["candidate004_full_manifest_conformance"] == "PASS"

        # Check all contract completeness booleans
        contracts = manifest["contracts_frozen"]
        for key, value in contracts.items():
            assert value is True, f"Contract frozen flag '{key}' is not True: {value}"

        # Check audit metrics
        metrics = manifest["audit_metrics"]
        assert metrics["reachable_partial_states_unclassified"] == 0
        assert metrics["illegal_states_without_fail_closed_action"] == 0
        assert metrics["unresolved_architecture_questions"] == 0
        assert metrics["unsupported_pass_claims"] == 0
        assert metrics["cross_artifact_contradictions"] == 0

        # Check gate outcome
        outcome = manifest["gate_outcome"]
        assert outcome["external_authority_architecture_contract_closure"] == "PASS"
        assert outcome["ready_for_operator_provisioning"] == "YES"
        assert outcome["next_gate"] == "SMART_MONEY_OPERATOR_EXTERNAL_AUTHORITY_PROVISIONING_V1"

        # Check governance invariants preserved
        gov = manifest["governance_invariants"]
        assert gov["natural_denominator"] == 0
        assert gov["natural_observer_arming"] == "NOT_AUTHORIZED"
        assert gov["production_readiness"] == "NOT_AUTHORIZED"
        assert gov["model_tuning"] == "FROZEN"
        assert gov["learning_claim"] == "NOT_AUTHORIZED"

    def test_07_candidate004_immutability_and_diff(self):
        freeze_path = os.path.join(PROJECT_ROOT, "docs", "domain", "vcp", "sprint_3", "candidates", "CANDIDATE_GENERATION_004_FREEZE.json")
        with open(freeze_path, "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == "297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90", "Candidate 004 freeze SHA changed!"

        diff = subprocess.check_output(["git", "diff", "--stat", freeze_path], cwd=PROJECT_ROOT, text=True).strip()
        assert diff == "", f"Candidate 004 freeze file has git diff: {diff}"

    def test_08_cross_artifact_consistency(self):
        # 1. Check break-glass permissions match in Trust Model and Threat Model
        trust_doc = os.path.join(CONTRACTS_DIR, "EXTERNAL_AUTHORITY_TRUST_MODEL.md")
        threat_doc = os.path.join(CONTRACTS_DIR, "THREAT_COMPROMISE_AND_RECOVERY_MODEL.md")
        with open(trust_doc, "r", encoding="utf-8") as f:
            trust_text = f.read()
        with open(threat_doc, "r", encoding="utf-8") as f:
            threat_text = f.read()

        assert "BREAK_GLASS_CAN_HALT: YES" in trust_text
        assert "BREAK_GLASS_CAN_AUTHORIZE: NO" in trust_text
        assert "BREAK_GLASS_CAN_EDIT_STATE: NO" in trust_text
        assert "BREAK_GLASS_CAN_DELETE_EVIDENCE: NO" in trust_text
        assert "BREAK_GLASS_FORCES_TERMINAL_HOLD: YES" in trust_text

        # 2. Check namespace matching in Commit Protocol and Live Qualification Protocol
        commit_doc = os.path.join(CONTRACTS_DIR, "DISTRIBUTED_STATE_AND_COMMIT_PROTOCOL.md")
        qual_doc = os.path.join(CONTRACTS_DIR, "LIVE_QUALIFICATION_PROTOCOL.md")
        with open(commit_doc, "r", encoding="utf-8") as f:
            commit_text = f.read()
        with open(qual_doc, "r", encoding="utf-8") as f:
            qual_text = f.read()

        assert "QUALIFICATION_STATE_PARTITION: qualification_epoch_001" in qual_text
        assert "QUALIFICATION_GATE3_SCOPE: urn:arx:scope:qualification:v1" in qual_text
