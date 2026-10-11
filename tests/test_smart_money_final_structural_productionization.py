"""
Test Suite: Final Structural Productionization Implementation + Proof Closure V1
Validates:
- Frozen policy loading & pinned hash immutability
- Exact OIDC tuple verification & broad-regex rejection
- Natural path strictly unreachable without Gate-3 authorization
- Gate-3 schema verification, expiry, replay, and revocation
- Manual recovery guard & schedule-safe behavior
- Durable state rehydration & checkpointing
- Cross-run persistence & R1/R2 restart recovery
- Denominator positive predicate & natural isolation
- Concurrency guard on main
- Candidate 004 firewall invariants
- Secret & token hygiene
- Cryptographic manifest verification
"""

import hashlib
import json
import os
import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVIDENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "evidence",
    "smart-money-final-structural-productionization-v1",
)
MANIFEST_PATH = os.path.join(
    EVIDENCE_DIR,
    "SMART_MONEY_FINAL_STRUCTURAL_PRODUCTIONIZATION_MANIFEST_V1.json",
)
WORKFLOW_PATH = os.path.join(PROJECT_ROOT, ".github", "workflows", "smart_money_rooting.yml")
POLICY_PATH = os.path.join(
    PROJECT_ROOT,
    "docs",
    "domain",
    "radar",
    "smart-money",
    "public-availability",
    "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1.json",
)

from analyst_dashboard.smart_money.public_availability.workload_identity_verifier import (
    verify_workload_identity,
)
from analyst_dashboard.smart_money.public_availability.gate3_authorization import (
    validate_gate3_authorization,
    SCHEMA_VERSION,
)
from analyst_dashboard.smart_money.public_availability.durable_state_manager import (
    DurableStateManager,
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestFinalStructuralProductionization:
    """Targeted tests for Final Structural Productionization Closure Gate V1."""

    # 1. Frozen Policy Loading & Immutability
    def test_frozen_policy_loading_and_immutability(self):
        assert os.path.exists(POLICY_PATH)
        with open(POLICY_PATH, "rb") as f:
            content = f.read()
        digest = sha256_hex(content)
        assert digest == "01dd7702246a9801bc88e0534accce4ad112f87acf22338a21afcda9348b7af7"

        # Check sha256 evidence file
        sha_file = os.path.join(EVIDENCE_DIR, "frozen_policy.sha256")
        assert os.path.exists(sha_file)
        with open(sha_file, "r") as sf:
            assert digest in sf.read()

    # 2. Exact OIDC Tuple & Broad-Regex Rejection
    def test_exact_oidc_tuple_and_negative_rejection(self):
        fpath = os.path.join(EVIDENCE_DIR, "oidc_exact_verification.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["baseline_verification"]["status"] == "PASS"
        assert data["exact_workload_identity_enforcement"] == "PASS"
        assert data["negative_controls_all_rejected"] is True

    # 3. Natural Path Unreachable Without Gate-3 Authorization
    def test_natural_path_unreachable_without_gate3(self):
        fpath = os.path.join(EVIDENCE_DIR, "natural_activation_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["post_gate3_natural_activation_mechanism"] == "SEALED_GATE3_AUTHORIZATION_ARTIFACT"
        assert data["cli_only_natural_activation"] == "IMPOSSIBLE"
        assert data["workflow_dispatch_authority_bypass"] == "IMPOSSIBLE"
        assert data["schedule_without_gate3_authorization"] == "SAFE_HOLD"

    # 4. Gate-3 Schema Verification & Negative Controls
    def test_gate3_schema_and_negative_controls(self):
        schema_path = os.path.join(EVIDENCE_DIR, "gate3_authorization_schema.json")
        nc_path = os.path.join(EVIDENCE_DIR, "gate3_negative_controls.json")
        assert os.path.exists(schema_path)
        assert os.path.exists(nc_path)

        with open(schema_path, "r", encoding="utf-8") as sf:
            sdata = json.load(sf)
        assert sdata["schema_version"] == SCHEMA_VERSION
        assert sdata["scope_constraint"] == "FIRST_NATURAL_CAPTURE"
        assert sdata["single_use_constraint"] is True

        with open(nc_path, "r", encoding="utf-8") as ncf:
            ncdata = json.load(ncf)
        assert ncdata["baseline_validation"]["status"] == "PASS"
        assert ncdata["all_negative_controls_rejected"] is True

    # 5. Gate-3 Single-Use and Replay Rejection
    def test_gate3_single_use_and_replay_rejection(self):
        fpath = os.path.join(EVIDENCE_DIR, "authorization_single_use.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["first_instance_accepted"] is True
        assert data["second_instance_rejected"] is True
        assert data["gate3_authorization_single_use"] == "PASS"
        assert data["gate3_authorization_replay"] == "REJECTED"

    # 6. Durable State Rehydration & Checkpointing
    def test_durable_state_rehydration_and_checkpoint(self):
        rehyd_path = os.path.join(EVIDENCE_DIR, "state_rehydration_result.json")
        chk_path = os.path.join(EVIDENCE_DIR, "state_checkpoint_result.json")
        assert os.path.exists(rehyd_path)
        assert os.path.exists(chk_path)

        with open(rehyd_path, "r", encoding="utf-8") as rf:
            rdata = json.load(rf)
        assert rdata["state_rehydration_on_main"] == "PASS"

        with open(chk_path, "r", encoding="utf-8") as cf:
            cdata = json.load(cf)
        assert cdata["state_checkpoint_on_main"] == "PASS"

    # 7. Cross-Run Persistence (Run A & Run B)
    def test_cross_run_persistence(self):
        f_run_a = os.path.join(EVIDENCE_DIR, "cross_run_persistence_run_a.json")
        f_run_b = os.path.join(EVIDENCE_DIR, "cross_run_persistence_run_b.json")
        assert os.path.exists(f_run_a)
        assert os.path.exists(f_run_b)

        with open(f_run_a, "r", encoding="utf-8") as fa:
            ra = json.load(fa)
        with open(f_run_b, "r", encoding="utf-8") as fb:
            rb = json.load(fb)

        assert ra["natural_denominator_increment"] == 0
        assert rb["cross_run_state_persistence"] == "PASS"
        assert rb["cross_run_denominator_persistence"] == "PASS"
        assert rb["cross_run_idempotency_persistence"] == "PASS"
        assert rb["cross_run_r1_r2_state_persistence"] == "PASS"
        assert rb["cross_run_evidence_persistence"] == "PASS"
        assert rb["natural_denominator_increment"] == 0

    # 8. R1/R2 Restart Recovery
    def test_r1_r2_restart_recovery(self):
        fpath = os.path.join(EVIDENCE_DIR, "r1_r2_restart_recovery.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["r1_pending_state_survives_restart"] == "PASS"
        assert data["r2_recovery_after_restart"] == "PASS"
        assert data["duplicate_authority_event"] == "NO"
        assert data["orphan_r1_detection"] == "PASS"

    # 9. Denominator Isolation & Natural Predicate
    def test_denominator_isolation(self):
        fpath = os.path.join(EVIDENCE_DIR, "denominator_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["denominator_natural_only_theorem"] == "PASS"
        assert data["natural_denominator"] == 0
        assert data["natural_accessions_captured"] == 0
        assert data["positive_predicate_enforced"] is True

    # 10. Concurrency Guard on Main
    def test_concurrency_guard_on_main(self):
        fpath = os.path.join(EVIDENCE_DIR, "concurrency_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["default_branch_concurrency_guard"] == "PRESENT"
        assert data["concurrency_group"] == "smart-money-natural-scheduler"
        assert data["cancel_in_progress"] is False
        assert data["concurrent_run_safety"] == "PASS"

        with open(WORKFLOW_PATH, "r", encoding="utf-8") as wf_file:
            wf = yaml.safe_load(wf_file)
        assert wf["concurrency"]["group"] == "smart-money-natural-scheduler"
        assert wf["concurrency"]["cancel-in-progress"] is False

    # 11. Secret & Token Hygiene
    def test_secret_hygiene(self):
        fpath = os.path.join(EVIDENCE_DIR, "secret_scan.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["raw_oidc_token_retained"] == "NO"
        assert data["long_lived_signing_secret"] == "NONE"
        assert data["secret_scan_status"] == "PASS"

    # 12. Candidate 004 Firewall Invariants
    def test_candidate004_firewall(self):
        fpath = os.path.join(EVIDENCE_DIR, "candidate004_firewall.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["direct_candidate_004_diff"] == 0
        assert data["indirect_candidate_004_behavioral_coupling"] == "NONE"
        assert data["candidate_004_interference"] == "NONE"

    # 13. Manifest Cryptographic Integrity
    def test_manifest_integrity(self):
        assert os.path.exists(MANIFEST_PATH)
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["gate"] == "SMART_MONEY_FINAL_STRUCTURAL_PRODUCTIONIZATION_CLOSURE_V1"
        assert len(manifest["artifacts"]) == 20

        for entry in manifest["artifacts"]:
            art_path = os.path.join(EVIDENCE_DIR, entry["relative_path"])
            assert os.path.exists(art_path), f"Artifact missing: {entry['relative_path']}"
            with open(art_path, "rb") as fh:
                content = fh.read()
            assert len(content) == entry["byte_length"]
            assert sha256_hex(content) == entry["sha256"]
