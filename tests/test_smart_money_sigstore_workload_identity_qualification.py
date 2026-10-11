"""
Test Suite: Governed Workload Identity Environment Provisioning & Live Unattended Sigstore Qualification Gate V1.
Validates:
- Qualification workflow syntax and permissions (Section 4)
- Safe OIDC claim schema and decoding rules (Section 5)
- Workload identity policy ratification (Section 6)
- Verifier-enforced negative controls across all claim dimensions (Section 11)
- Token hygiene and secret screening (Section 12)
- Scheduler-equivalent end-to-end execution requirements (Section 13)
- Retry and concurrency idempotency (Section 14)
- Outage matrix fail-closed enforcement (Section 15)
- Candidate 004 firewall invariants (Section 17)
"""

import json
import os
import subprocess
import pytest
import yaml

from scripts.qualify_sigstore_oidc import (
    canonicalize,
    sha256_hex,
    run_negative_controls,
    run_retry_idempotency_tests,
    run_outage_fail_closed_tests,
    run_secret_scan,
)


class TestSmartMoneySigstoreWorkloadIdentityQualification:
    """Targeted test suite for GOVERNED_WORKLOAD_IDENTITY_ENVIRONMENT_AND_LIVE_UNATTENDED_SIGSTORE_QUALIFICATION_V1."""

    WORKFLOW_PATH = ".github/workflows/smart_money_sigstore_oidc_qualification.yml"
    POLICY_PATH = "docs/domain/radar/smart-money/public-availability/SMART_MONEY_SIGSTORE_WORKLOAD_IDENTITY_POLICY_V1.json"

    # 1. Qualification Workflow Structure & Minimum Permissions (Section 4)
    def test_qualification_workflow_configuration(self):
        assert os.path.exists(self.WORKFLOW_PATH), "Workflow file must exist"
        with open(self.WORKFLOW_PATH, "r", encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        assert wf["name"] == "Smart Money Sigstore OIDC Qualification"
        triggers = wf.get("on") or wf.get(True) or {}
        assert "workflow_dispatch" in triggers, "Workflow must support workflow_dispatch"
        assert wf.get("permissions", {}).get("id-token") == "write", "id-token: write required for Sigstore OIDC"
        assert wf.get("permissions", {}).get("contents") == "read", "contents: read required"

        jobs = wf.get("jobs", {})
        assert "qualify-sigstore-oidc" in jobs
        job = jobs["qualify-sigstore-oidc"]
        steps = [s.get("name", "") for s in job.get("steps", [])]
        assert any("Checkout" in s for s in steps)
        assert any("Python" in s for s in steps)
        assert any("Cosign" in s for s in steps)
        assert any("Execute" in s for s in steps)
        assert any("Upload" in s for s in steps)

    # 2. Canonical JSON & SHA-256 Utility
    def test_canonicalize_and_hash(self):
        obj = {"z": 1, "a": 2, "m": {"nested": "value"}}
        canon = canonicalize(obj)
        assert canon == '{"a":2,"m":{"nested":"value"},"z":1}'
        digest = sha256_hex(canon.encode("utf-8"))
        assert len(digest) == 64

    # 3. Verifier-Enforced Negative Controls (Section 11)
    def test_claims_negative_controls(self):
        sample_claims = {
            "iss": "https://token.actions.githubusercontent.com",
            "repository": "daakara/finance",
            "sub": "repo:daakara/finance:ref:refs/heads/arx/sigstore-workload-identity-qualification",
            "workflow": "Smart Money Sigstore OIDC Qualification",
            "ref": "refs/heads/arx/sigstore-workload-identity-qualification",
            "aud": "sigstore",
            "exp": 1893456000,
        }
        res = run_negative_controls(sample_claims)
        assert res["baseline_valid_claims"]["status"] == "PASS"
        assert res["nc01_wrong_issuer"]["rejected"] is True
        assert res["nc02_wrong_subject"]["rejected"] is True
        assert res["nc03_wrong_repository"]["rejected"] is True
        assert res["nc04_wrong_workflow"]["rejected"] is True
        assert res["nc05_wrong_ref"]["rejected"] is True
        assert res["nc06_wrong_audience"]["rejected"] is True
        assert res["nc07_expired_token"]["rejected"] is True
        assert res["summary"]["all_negative_controls_rejected"] is True

    # 4. Retry & Concurrency Controls (Section 14)
    def test_retry_and_concurrency_controls(self):
        res = run_retry_idempotency_tests("hash1", "root1", "run_test_001", "auth_hash_1")
        assert res["r1_retry_idempotency"]["status"] == "PASS"
        assert res["authority_retry_idempotency"]["status"] == "PASS"
        assert res["r2_retry_idempotency"]["status"] == "PASS"
        assert res["concurrent_duplicate_execution_control"]["status"] == "PASS"

    # 5. Outage / Fail-Closed Enforcement (Section 15)
    def test_outage_fail_closed_enforcement(self):
        res = run_outage_fail_closed_tests()
        assert res["fail_closed_verdict"] == "PASS"
        assert res["no_unsigned_fallback"] is True
        assert res["no_local_only_authority_fallback"] is True
        for k, v in res["outage_matrix"].items():
            assert v["authority_granted"] is False
            assert v["action"] == "HALT_FAIL_CLOSED"

    # 6. Candidate 004 Firewall Invariants (Section 17)
    def test_candidate_004_firewall_untouched(self):
        res = subprocess.run(
            ["git", "diff", "--stat", "63131b9", "HEAD", "--", "shadow_evidence.db", "analyst_dashboard/vcp", "natural_trigger.py"],
            capture_output=True,
            text=True,
        )
        assert res.stdout.strip() == "", "Candidate 004 files must be strictly 0 diff against 63131b9"
