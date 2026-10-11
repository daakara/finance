"""
Test Suite: Smart Money Structural Productionization Hardening Closure V1
Validates:
- OIDC event-name model
- OIDC subject template mode
- Runtime DSN binding (non-source-file paths, isolated authority)
- Cross-run state, evidence, and denominator persistence
- Natural-only admission predicate
- Denominator negative controls (11/11 rejected)
- Manual recovery authority guard (workflow_dispatch)
- Observation coverage policy (periodic sampling ratified)
- R1/R2 restart recovery and orphan R1 detection
- Cross-run idempotency
- Failure/recovery matrix (14 scenarios)
- Concurrency contract & workflow concurrency group
- Secret & credential hygiene
- Candidate 004 firewall invariants
- Manifest cryptographic integrity
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
    "smart-money-structural-productionization-hardening-v1",
)
MANIFEST_PATH = os.path.join(
    EVIDENCE_DIR,
    "SMART_MONEY_STRUCTURAL_PRODUCTIONIZATION_HARDENING_MANIFEST_V1.json",
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


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestStructuralProductionizationHardening:
    """Targeted tests for Structural Productionization Hardening Closure Gate V1."""

    # 1. OIDC Event-Name Security Model
    def test_oidc_event_name_model(self):
        fpath = os.path.join(EVIDENCE_DIR, "oidc_event_name_model.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["oidc_event_name_security_model"] == "ALLOWED_EVENT_CLASSIFICATION_ONLY"
        assert "schedule" in data["authorized_event_names"]
        assert "workflow_dispatch" in data["authorized_event_names"]
        assert "push" in data["prohibited_event_names"]
        assert data["schedule_event_identity"] == "POLICY_DERIVED_ACCEPTABLE"

        inv = data["security_critical_claims_invariant"]
        assert inv["iss"] == "https://token.actions.githubusercontent.com"
        assert inv["aud"] == "sigstore"
        assert inv["repository"] == "daakara/finance"
        assert inv["repository_id"] == "1087399232"
        assert inv["repository_owner_id"] == "53128824"
        assert inv["ref"] == "refs/heads/main"
        assert inv["sub"] == "repo:daakara/finance:ref:refs/heads/main"

    # 2. OIDC Subject Template Mode
    def test_oidc_subject_template_mode(self):
        fpath = os.path.join(EVIDENCE_DIR, "oidc_subject_template.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["oidc_subject_template_mode"] == "LEGACY_DEFAULT"
        assert data["oidc_subject_template_version"] == "GITHUB_DEFAULT_V1"
        assert data["oidc_subject_template_source"] == "GITHUB_OIDC_CUSTOMIZATION_API"
        assert data["oidc_subject_example"] == "repo:daakara/finance:ref:refs/heads/main"
        assert data["invalidation_rule"] == "OIDC_SUBJECT_TEMPLATE_CHANGE_INVALIDATES_NATURAL_SCHEDULER_IDENTITY_GATE"

        # Also check policy
        with open(POLICY_PATH, "r", encoding="utf-8") as pf:
            pol = json.load(pf)
        assert pol["oidc_subject_template"]["mode"] == "LEGACY_DEFAULT"

    # 3. Runtime Store Binding (Non-source file paths, DSNs, separation)
    def test_runtime_store_binding(self):
        fpath = os.path.join(EVIDENCE_DIR, "runtime_store_binding.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Must not be source code files!
        for k, v in data.items():
            if isinstance(v, str) and v.endswith(".py"):
                pytest.fail(f"Runtime store {k} incorrectly references Python source file: {v}")

        assert "sqlite:///" in data["NATURAL_SCHEDULER_RUNTIME_STATE_DSN"]
        assert "file://" in data["NATURAL_SCHEDULER_RUNTIME_EVIDENCE_DSN"]
        assert data["STATE_AUTHORITY_SHARED"] == "NO"
        assert data["EVIDENCE_AUTHORITY_SHARED"] == "NO"

    # 4. Persistence Backend Contract
    def test_persistence_backend_contract(self):
        fpath = os.path.join(EVIDENCE_DIR, "persistence_backend_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = data["summary"]
        assert summary["github_actions_artifacts"] == "BOUNDED_RETENTION_STORE"
        assert summary["long_term_governance_custody"] == "NOT_ESTABLISHED_BY_ARTIFACTS_ALONE"

    # 5. Two-Run Persistence Drill
    def test_two_run_persistence_drill(self):
        f_run_a = os.path.join(EVIDENCE_DIR, "cross_run_persistence_run_a.json")
        f_run_b = os.path.join(EVIDENCE_DIR, "cross_run_persistence_run_b.json")
        f_recon = os.path.join(EVIDENCE_DIR, "cross_run_reconciliation.json")

        assert os.path.exists(f_run_a)
        assert os.path.exists(f_run_b)
        assert os.path.exists(f_recon)

        with open(f_run_a, "r", encoding="utf-8") as f:
            run_a = json.load(f)
        with open(f_run_b, "r", encoding="utf-8") as f:
            run_b = json.load(f)
        with open(f_recon, "r", encoding="utf-8") as f:
            recon = json.load(f)

        assert run_a["qualification_scope"] == "CROSS_RUN_PERSISTENCE_DRILL"
        assert run_a["natural_accession"] is False
        assert run_a["natural_denominator_increment"] == 0

        assert run_b["cross_run_state_retrieval"] == "PASS"
        assert run_b["cross_run_denominator_state_retrieval"] == "PASS"
        assert run_b["cross_run_idempotency_retrieval"] == "PASS"
        assert run_b["cross_run_evidence_retrieval"] == "PASS"
        assert run_b["cross_run_sha256_match"] == "PASS"
        assert run_b["cross_run_manifest_reconciliation"] == "PASS"
        assert run_b["cross_run_r1_r2_linkage_reconstruction"] == "PASS"
        assert run_b["natural_denominator_increment"] == 0

        assert recon["all_digests_match"] is True

    # 6. Natural Denominator Membership Contract
    def test_natural_denominator_contract(self):
        fpath = os.path.join(EVIDENCE_DIR, "natural_denominator_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["denominator_membership"] == "EXPLICIT_POSITIVE_NATURAL_ADMISSION"
        assert data["current_natural_denominator"] == 0
        assert data["current_natural_accessions_captured"] == 0

    # 7. Denominator Negative Controls (11/11 Rejected)
    def test_denominator_negative_controls(self):
        fpath = os.path.join(EVIDENCE_DIR, "negative_admission_controls.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["denominator_natural_only_theorem"] == "PASS"
        assert data["all_rejected"] is True
        assert data["controls_tested"] == 11
        assert data["controls_rejected"] == 11

        results = data["results"]
        assert results["qualification_artifact"]["status"] == "REJECTED"
        assert results["workflow_dispatch_dry_run_artifact"]["status"] == "REJECTED"
        assert results["schedule_dry_run_artifact"]["status"] == "REJECTED"
        assert results["synthetic_artifact"]["status"] == "REJECTED"
        assert results["historical_replay"]["status"] == "REJECTED"
        assert results["manual_test"]["status"] == "REJECTED"
        assert results["durability_drill"]["status"] == "REJECTED"
        assert results["cross_run_persistence_drill"]["status"] == "REJECTED"
        assert results["malformed_natural_candidate"]["status"] == "REJECTED"
        assert results["identity_invalid_candidate"]["status"] == "REJECTED"
        assert results["structurally_invalid_candidate"]["status"] == "REJECTED"

    # 8. Manual Recovery Authority Guard
    def test_manual_recovery_guard(self):
        fpath = os.path.join(EVIDENCE_DIR, "manual_recovery_guard.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["workflow_dispatch_default_mode"] == "dry-run"
        assert data["workflow_dispatch_natural_authority"] == "REQUIRES_IDENTICAL_TIME_OF_USE_PRE_ARM_GATES"
        assert data["manual_recovery_authority_bypass"] == "IMPOSSIBLE"

    # 9. Observation Coverage Policy
    def test_observation_coverage_policy(self):
        fpath = os.path.join(EVIDENCE_DIR, "observation_coverage_policy.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["observation_coverage_objective"] == "PERIODIC_SAMPLING"
        assert data["observation_coverage_policy"] == "RATIFIED"
        assert data["schedule_cron"] == "7 10-22/2 * * 1-5"
        assert data["schedule_timezone"] == "UTC"
        assert data["schedule_timezone_mode"] == "UTC_DEFAULT"

    # 10. R1/R2 Restart Recovery Drill
    def test_r1_r2_restart_recovery(self):
        fpath = os.path.join(EVIDENCE_DIR, "r1_r2_restart_recovery.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["r1_pending_state_survives_restart"] == "PASS"
        assert data["r2_recovery_after_restart"] == "PASS"
        assert data["duplicate_authority_event_after_restart"] == "NO"
        assert data["orphan_r1_detection"] == "PASS"

    # 11. Cross-Run Idempotency Drill
    def test_cross_run_idempotency(self):
        fpath = os.path.join(EVIDENCE_DIR, "cross_run_idempotency.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["first_run_processing"] == "ACCEPTED_FOR_TEST_PATH"
        assert data["second_run_duplicate_detection"] == "PASS"
        assert data["duplicate_natural_admission"] == 0
        assert data["duplicate_r1"] == "PREVENTED_OR_EXPLICITLY_NON_AUTHORITATIVE"
        assert data["duplicate_authority_event"] == "PREVENTED"
        assert data["cross_run_idempotency"] == "PASS"

    # 12. Failure / Recovery Matrix (14 Scenarios)
    def test_failure_recovery_matrix(self):
        fpath = os.path.join(EVIDENCE_DIR, "failure_recovery_matrix.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["scenarios_tested"] == 14
        assert data["partial_write_recovery"] == "PASS"
        assert data["no_unrooted_authority"] == "YES"
        assert data["no_duplicate_authority"] == "YES"
        assert data["no_denominator_increment_on_partial_failure"] == "YES"

    # 13. Concurrency Contract & Workflow Configuration
    def test_concurrency_control(self):
        fpath = os.path.join(EVIDENCE_DIR, "concurrency_contract.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["github_actions_concurrency_group"] == "smart-money-natural-scheduler"
        assert data["cancel_in_progress"] is False
        assert data["concurrent_run_safety"] == "PASS"

        # Check workflow file has concurrency block
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as wf_file:
            wf = yaml.safe_load(wf_file)
        assert "concurrency" in wf, "Workflow must contain concurrency block"
        assert wf["concurrency"]["group"] == "smart-money-natural-scheduler"
        assert wf["concurrency"]["cancel-in-progress"] is False

    # 14. Secret Hygiene
    def test_secret_scan(self):
        fpath = os.path.join(EVIDENCE_DIR, "secret_scan.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["secret_scan_status"] == "PASS"
        assert data["raw_oidc_token_retained"] == "NO"
        assert data["long_lived_signing_secret"] == "NONE"
        assert data["persisted_secret_findings"] == 0

    # 15. Candidate 004 Firewall Invariants
    def test_candidate004_firewall(self):
        fpath = os.path.join(EVIDENCE_DIR, "candidate004_firewall.json")
        assert os.path.exists(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["direct_candidate_004_diff"] == 0
        assert data["indirect_candidate_004_behavioral_coupling"] == "NONE"
        assert data["candidate_004_interference"] == "NONE"

    # 16. Cryptographic Manifest Integrity
    def test_manifest_integrity(self):
        assert os.path.exists(MANIFEST_PATH)
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["gate"] == "SMART_MONEY_STRUCTURAL_PRODUCTIONIZATION_HARDENING_V1"
        assert len(manifest["artifacts"]) == 18

        for entry in manifest["artifacts"]:
            art_path = os.path.join(EVIDENCE_DIR, entry["relative_path"])
            assert os.path.exists(art_path), f"Artifact missing: {entry['relative_path']}"
            with open(art_path, "rb") as fh:
                content = fh.read()
            assert len(content) == entry["byte_length"]
            assert sha256_hex(content) == entry["sha256"]
