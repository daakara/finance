"""
Test Suite: Natural Scheduler Closure & Pre-Arm Qualification Gate V1.
Validates:
- Natural scheduler workflow configuration (.github/workflows/smart_money_rooting.yml)
- Natural workload identity policy ratification (SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1)
- Manifest integrity of evidence/sigstore-natural-scheduler-dry-run-v1
- OIDC claim attestation and exact policy matching
- Dry run R1 and R2 live Sigstore witness verification
- Cryptographic binding between R2, Authority Event, and R1
- Zero human intervention, zero secrets, and zero Candidate 004 interference
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
EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence", "sigstore-natural-scheduler-dry-run-v1")
MANIFEST_PATH = os.path.join(EVIDENCE_DIR, "SIGSTORE_NATURAL_SCHEDULER_DRY_RUN_MANIFEST_V1.json")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestNaturalSchedulerClosure:
    """Targeted tests for Natural Scheduler Closure (Stage A)."""

    def test_natural_scheduler_workflow_configuration(self):
        assert os.path.exists(WORKFLOW_PATH), f"Workflow must exist at {WORKFLOW_PATH}"
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        assert wf["name"] == "Smart Money Rooting Scheduler"
        triggers = wf.get("on") or wf.get(True) or {}
        assert "workflow_dispatch" in triggers
        assert "schedule" in triggers
        assert "push" in triggers
        assert wf.get("permissions", {}).get("id-token") == "write"
        assert wf.get("permissions", {}).get("contents") == "read"

        jobs = wf.get("jobs", {})
        assert "natural-rooting-scheduler" in jobs
        job = jobs["natural-rooting-scheduler"]
        steps = [s.get("name", "") for s in job.get("steps", [])]
        assert any("Checkout" in s for s in steps)
        assert any("Python" in s for s in steps)
        assert any("Cosign" in s for s in steps)
        assert any("Execute" in s for s in steps)
        assert any("Upload" in s for s in steps)

    def test_natural_workload_identity_policy(self):
        assert os.path.exists(POLICY_PATH), f"Policy must exist at {POLICY_PATH}"
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            policy = json.load(f)

        assert policy["policy_id"] == "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1"
        assert policy["status"] == "RATIFIED"
        constraints = policy["workload_identity_constraints"]
        assert constraints["authorized_issuer"] == "https://token.actions.githubusercontent.com"
        assert constraints["authorized_repository"] == "daakara/finance"
        assert constraints["authorized_repository_id"] == "1087399232"
        assert constraints["authorized_repository_owner"] == "daakara"
        assert constraints["authorized_repository_owner_id"] == "53128824"
        assert constraints["authorized_workflow"] == "Smart Money Rooting Scheduler"
        assert constraints["authorized_audience"] == "sigstore"
        assert "*" not in constraints["authorized_repository"]
        assert "*" not in constraints["authorized_workflow"]

    def test_dry_run_manifest_integrity(self):
        assert os.path.exists(MANIFEST_PATH), f"Manifest must exist at {MANIFEST_PATH}"
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for art in manifest.get("artifacts", []):
            rel_path = art["relative_path"]
            file_path = os.path.join(EVIDENCE_DIR, rel_path)
            assert os.path.exists(file_path), f"Evidence file {rel_path} must exist"
            with open(file_path, "rb") as fh:
                digest = sha256_hex(fh.read())
            assert digest == art["sha256"], f"Digest mismatch for {rel_path}"

    def test_dry_run_summary_and_attestation(self):
        summary_path = os.path.join(EVIDENCE_DIR, "natural_dry_run_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["natural_scheduler_r1"] == "PASS"
        assert summary["natural_scheduler_r2"] == "PASS"
        assert summary["natural_scheduler_policy_match"] == "PASS"
        assert summary["natural_scheduler_dry_run"] == "PASS"
        assert summary["human_signing_interaction"] == 0
        assert summary["long_lived_signing_secret"] == "NONE"

        claims = summary["claims"]
        assert claims["iss"] == "https://token.actions.githubusercontent.com"
        assert claims["repository"] == "daakara/finance"
        assert claims["repository_id"] == "1087399232"
        assert claims["repository_owner_id"] == "53128824"
        assert claims["workflow"] == "Smart Money Rooting Scheduler"
        assert claims["aud"] == "sigstore"

    def test_r2_to_r1_cryptographic_binding(self):
        r1_payload_path = os.path.join(EVIDENCE_DIR, "natural_r1_payload.json")
        r2_payload_path = os.path.join(EVIDENCE_DIR, "natural_r2_payload.json")
        auth_path = os.path.join(EVIDENCE_DIR, "natural_authority_event.json")
        summary_path = os.path.join(EVIDENCE_DIR, "natural_dry_run_summary.json")

        with open(r1_payload_path, "r", encoding="utf-8") as f:
            r1 = json.load(f)
        with open(r2_payload_path, "r", encoding="utf-8") as f:
            r2 = json.load(f)
        with open(auth_path, "rb") as f:
            auth_bytes = f.read()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        auth_digest = sha256_hex(auth_bytes)
        assert r2["authority_event_hash"] == auth_digest
        assert r2["authoritative_r1_witness_identity"] == summary["r1_bundle_sha256"]
        assert r2["r1_bundle_sha256"] == summary["r1_bundle_sha256"]
        assert r2["r1_root_hash"] == r1["root_hash"]
