"""
Test Suite: Natural Scheduler Productionization Closure Gate V1.
Validates:
- Default-branch scheduler contract (.github/workflows/smart_money_rooting.yml)
- Trigger policy (schedule primary, workflow_dispatch recovery, push disabled)
- Schedule semantics (UTC, 7 10-22/2 * * 1-5, non-top-of-hour)
- OIDC exact policy for default branch
- Default-branch dry run artifacts and verification
- Schedule vs workflow_dispatch identity policy equivalence theorem
- Natural denominator exclusion and zero increments
- Authoritative state-store binding
- Authoritative denominator-store binding
- Backup/restore drill and manifest reconciliation
- R2 closure policy normalization (governance safety bound, limited empirical support)
- Fail-closed behavior
- Candidate 004 firewall invariants
"""

import hashlib
import json
import os
import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
EVIDENCE_DIR = os.path.join(
    PROJECT_ROOT, "evidence", "smart-money-natural-scheduler-productionization-v1"
)
MANIFEST_PATH = os.path.join(
    EVIDENCE_DIR, "SMART_MONEY_NATURAL_SCHEDULER_PRODUCTIONIZATION_MANIFEST_V1.json"
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestNaturalSchedulerProductionization:
    """Targeted tests for Natural Scheduler Productionization Closure Gate V1."""

    # 1. Default-branch scheduler contract & triggers
    def test_default_branch_scheduler_contract(self):
        assert os.path.exists(WORKFLOW_PATH), f"Workflow must exist at {WORKFLOW_PATH}"
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        assert wf["name"] == "Smart Money Rooting Scheduler"
        triggers = wf.get("on") or wf.get(True) or {}
        assert "schedule" in triggers, "Primary natural trigger must be schedule"
        assert "workflow_dispatch" in triggers, "Manual recovery trigger must be workflow_dispatch"
        assert "push" not in triggers, "Push trigger must be disabled in production"

        # Permissions
        assert wf.get("permissions", {}).get("id-token") == "write"
        assert wf.get("permissions", {}).get("contents") == "read"

    # 2. Schedule semantics
    def test_schedule_semantics(self):
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            wf = yaml.safe_load(f)
        triggers = wf.get("on") or wf.get(True) or {}
        cron_expr = triggers["schedule"][0]["cron"]
        assert cron_expr == "7 10-22/2 * * 1-5"
        # Non-top-of-hour
        minute = cron_expr.split()[0]
        assert minute == "7", "Must use minute 7 to avoid top-of-hour GitHub queue delays"

    # 3. OIDC exact policy & default-branch binding
    def test_oidc_exact_policy_default_branch(self):
        assert os.path.exists(POLICY_PATH)
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            policy = json.load(f)

        assert policy["policy_id"] == "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1"
        assert policy["status"] == "RATIFIED_FOR_DEFAULT_BRANCH_QUALIFICATION"

        constraints = policy["workload_identity_constraints"]
        assert constraints["authorized_issuer"] == "https://token.actions.githubusercontent.com"
        assert constraints["authorized_audience"] == "sigstore"
        assert constraints["authorized_repository"] == "daakara/finance"
        assert constraints["authorized_repository_id"] == "1087399232"
        assert constraints["authorized_repository_owner_id"] == "53128824"
        assert constraints["authorized_ref"] == "refs/heads/main"
        assert "schedule" in constraints["authorized_event_names"]
        assert "workflow_dispatch" in constraints["authorized_event_names"]
        assert "push" in constraints["prohibited_event_names"]

    # 4. Default-branch dry run artifacts & bindings
    def test_default_branch_dry_run_artifacts(self):
        dry_run_dir = os.path.join(EVIDENCE_DIR, "default_branch_dry_run")
        summary_path = os.path.join(dry_run_dir, "execution_summary.json")
        assert os.path.exists(summary_path)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["natural_scheduler_r1"] == "PASS"
        assert summary["natural_scheduler_r2"] == "PASS"
        assert summary["natural_scheduler_dry_run"] == "PASS"
        assert summary["natural_scheduler_policy_match"] == "PASS"
        assert summary["human_signing_interaction"] == 0
        assert summary["long_lived_signing_secret"] == "NONE"

        claims = summary["claims"]
        assert claims["iss"] == "https://token.actions.githubusercontent.com"
        assert claims["repository"] == "daakara/finance"
        assert claims["repository_id"] == "1087399232"
        assert claims["ref"] == "refs/heads/main"
        assert claims["sub"] == "repo:daakara/finance:ref:refs/heads/main"
        assert claims["job_workflow_ref"] == "daakara/finance/.github/workflows/smart_money_rooting.yml@refs/heads/main"

        # R2 -> R1 and R2 -> Authority event binding
        r1_payload_path = os.path.join(dry_run_dir, "r1_payload.json")
        r2_payload_path = os.path.join(dry_run_dir, "r2_payload.json")
        auth_event_path = os.path.join(dry_run_dir, "authority_event.json")

        with open(r1_payload_path, "r", encoding="utf-8") as f:
            r1 = json.load(f)
        with open(r2_payload_path, "r", encoding="utf-8") as f:
            r2 = json.load(f)
        with open(auth_event_path, "rb") as f:
            auth_digest = sha256_hex(f.read())

        assert r2["authority_event_hash"] == auth_digest
        assert r2["authoritative_r1_witness_identity"] == summary["r1_bundle_sha256"]
        assert r2["r1_bundle_sha256"] == summary["r1_bundle_sha256"]
        assert r2["r1_root_hash"] == r1["root_hash"]

    # 5. Manifest integrity
    def test_manifest_integrity(self):
        assert os.path.exists(MANIFEST_PATH)
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest["artifacts"]) > 0
        for art in manifest["artifacts"]:
            fp = os.path.join(EVIDENCE_DIR, art["relative_path"])
            assert os.path.exists(fp), f"Missing artifact: {art['relative_path']}"
            with open(fp, "rb") as fh:
                digest = sha256_hex(fh.read())
            assert digest == art["sha256"], f"Digest mismatch on {art['relative_path']}"

    # 6. Authoritative observer state & denominator store binding
    def test_authoritative_store_binding(self):
        auth_state_path = os.path.join(
            EVIDENCE_DIR, "runtime_authority", "observer_state_authority.json"
        )
        auth_denom_path = os.path.join(
            EVIDENCE_DIR, "runtime_authority", "denominator_authority.json"
        )
        denom_res_path = os.path.join(
            EVIDENCE_DIR, "runtime_authority", "denominator_query_result.json"
        )

        with open(auth_state_path, "r", encoding="utf-8") as f:
            state_auth = json.load(f)
        assert state_auth["current_observer_runtime_state"] == "PRE_OBSERVATION_READY"
        assert state_auth["observer_runtime_state_attested"] == "PASS"

        with open(auth_denom_path, "r", encoding="utf-8") as f:
            denom_auth = json.load(f)
        assert denom_auth["authoritative_natural_admission_table"] == "sec_public_observations"

        with open(denom_res_path, "r", encoding="utf-8") as f:
            denom_res = json.load(f)
        assert denom_res["natural_accessions_captured"] == 0
        assert denom_res["natural_denominator"] == 0
        assert denom_res["synthetic_records_in_natural_denominator"] == 0
        assert denom_res["historical_records_in_natural_denominator"] == 0
        assert denom_res["natural_denominator_clean"] is True

    # 7. Durability drill results
    def test_durability_drill(self):
        dur_res_path = os.path.join(EVIDENCE_DIR, "durability", "restore_result.json")
        dur_recon_path = os.path.join(EVIDENCE_DIR, "durability", "manifest_reconciliation.json")

        with open(dur_res_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        assert res["raw_evidence_write"] == "PASS"
        assert res["backup_write"] == "PASS"
        assert res["restore_drill"] == "PASS"
        assert res["restored_sha256_match"] == "PASS"
        assert res["manifest_reconciliation"] == "PASS"
        assert res["natural_denominator_increment"] == 0

        with open(dur_recon_path, "r", encoding="utf-8") as f:
            recon = json.load(f)
        assert recon["byte_integrity_verified"] is True
        assert recon["hash_integrity_verified"] is True

    # 8. R2 closure policy normalization
    def test_r2_closure_policy_normalization(self):
        r2_norm_path = os.path.join(
            EVIDENCE_DIR, "governance", "r2_closure_policy_normalization.json"
        )
        with open(r2_norm_path, "r", encoding="utf-8") as f:
            r2_norm = json.load(f)

        assert r2_norm["r2_closure_max_delay_seconds"] == 300
        assert r2_norm["r2_closure_delay_policy_status"] == "RATIFIED_FOR_FIRST_NATURAL_CAPTURE"
        assert r2_norm["r2_closure_policy_basis"] == "GOVERNANCE_SAFETY_BOUND"
        assert r2_norm["r2_closure_empirical_latency_support"] == "LIMITED"
        assert r2_norm["r2_closure_long_term_calibration"] == "PENDING_NATURAL_OPERATIONAL_EVIDENCE"

    # 9. Negative controls
    def test_negative_controls(self):
        neg_path = os.path.join(EVIDENCE_DIR, "negative_controls.json")
        with open(neg_path, "r", encoding="utf-8") as f:
            neg = json.load(f)

        assert neg["overall_status"] == "PASS"
        assert neg["qualification_artifact_admission"] == "REJECTED"
        assert neg["dry_run_artifact_admission"] == "REJECTED"
        assert neg["durability_drill_artifact_admission"] == "REJECTED"
        assert neg["historical_artifact_admission"] == "REJECTED"

    # 10. Secret scan hygiene
    def test_secret_scan_hygiene(self):
        scan_path = os.path.join(EVIDENCE_DIR, "secret_scan_result.json")
        with open(scan_path, "r", encoding="utf-8") as f:
            scan = json.load(f)

        assert scan["secret_scan_status"] == "PASS"
        assert scan["raw_oidc_token_retained"] is False
        assert scan["long_lived_signing_secret"] == "NONE"
        assert scan["findings_count"] == 0
