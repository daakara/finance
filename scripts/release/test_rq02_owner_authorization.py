"""ARX Release Qualification - RQ-02 Solo-Owner PR Authorization Adversarial Test Suite.

Comprehensive negative and positive controls for verifiable solo-owner PR admission.
Verifies immutable SHA binding, source scope containment, forbidden path protection,
owner impersonation defense, and fail-closed security invariants.

Run: python -m unittest scripts/release/test_rq02_owner_authorization.py
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rq02_owner_authorization import (
    load_authorization_registry,
    verify_candidate_authorization,
    AUTHORIZED_OWNER,
)

FIXTURE_HEAD_SHA = "5655e4d4d01c3186e37cd1bd5f53cd90e0e3006e"
FIXTURE_BASE_SHA = "e4030398b3c9fb3b9e226244a6ca6124af1f85ab"
TAMPERED_HEAD_SHA = "1111111111111111111111111111111111111111"

VALID_REGISTRY = {
    "policy_name": "ARX_SOLO_OWNER_PR_AUTHORIZATION_POLICY",
    "policy_version": "1.0.0",
    "repository": "daakara/finance",
    "owner": AUTHORIZED_OWNER,
    "historical_approved_prs": [147, 196, 201, 205],
    "authorizations": {
        "207": {
            "pr_number": 207,
            "candidate_head_sha": FIXTURE_HEAD_SHA,
            "base_sha": FIXTURE_BASE_SHA,
            "target_branch": "main",
            "authorized_by": AUTHORIZED_OWNER,
            "directive": "ARX-ARCH-001",
            "allowed_paths": [
                "analyst_dashboard/analyzers/self_healing_engine.py",
                "tests/test_self_healing_audit_claims.py",
            ],
            "authorized_at_utc": "2026-11-06T09:46:00Z",
        }
    },
}


class SoloOwnerAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry_file = os.path.join(self.temp_dir.name, "pr_authorizations.json")
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(VALID_REGISTRY, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Positive: Valid solo-owner authorized PR passes
    def test_positive_owner_authorized_pr(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", "tests/test_self_healing_audit_claims.py"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertTrue(ok, f"Expected pass, got: {reason}")
        self.assertEqual(reason, "OWNER_AUTHORIZATION_VERIFIED")
        self.assertEqual(details["mode"], "EXPLICIT_OWNER_AUTHORIZATION")

    # 2. Positive: Historical approved PR passes
    def test_positive_historical_pr(self):
        diff = [("M", "analyst_dashboard/something.py")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=147,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertTrue(ok, f"Expected pass, got: {reason}")
        self.assertEqual(reason, "HISTORICAL_PR_AUTHORIZED")
        self.assertEqual(details["mode"], "HISTORICAL_APPROVAL")

    # 3. Negative: Unapproved PR number strictly fails closed
    def test_negative_unapproved_pr_number(self):
        diff = [("M", "analyst_dashboard/analyzers/self_healing_engine.py")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAPPROVED_PR_NUMBER", reason)

    # 4. Negative: Immutable SHA binding - modified candidate SHA strictly fails closed
    def test_negative_candidate_sha_mismatch(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", "tests/test_self_healing_audit_claims.py"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=TAMPERED_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("CANDIDATE_SHA_MISMATCH", reason)

    # 5. Negative: Scope expansion - extra file in diff strictly fails closed
    def test_negative_scope_expansion_extra_file(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", "tests/test_self_healing_audit_claims.py"),
            ("M", "api/routes/analytics.py"),  # Unapproved extra file!
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAUTHORIZED_DIFF_PATH", reason)

    # 6. Negative: Forbidden path - secret modification strictly fails closed
    def test_negative_forbidden_secret_modification(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", ".secrets/compromised.json"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("FORBIDDEN_PROTECTED_PATH_MODIFICATION", reason)

    # 7. Negative: Forbidden path - release manifest modification fails closed
    def test_negative_forbidden_manifest_modification(self):
        diff = [
            ("M", "scripts/release/release_manifest.json"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("FORBIDDEN_PROTECTED_PATH_MODIFICATION", reason)

    # 8. Negative: Forbidden path - production deployment config fails closed
    def test_negative_forbidden_production_config(self):
        diff = [
            ("M", "railway.json"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("FORBIDDEN_PROTECTED_PATH_MODIFICATION", reason)

    # 9. Negative: Unauthorized signer fails closed
    def test_negative_unauthorized_signer(self):
        tampered_reg = dict(VALID_REGISTRY)
        tampered_reg["authorizations"] = {
            "207": dict(VALID_REGISTRY["authorizations"]["207"], authorized_by="malicious_actor")
        }
        tampered_path = os.path.join(self.temp_dir.name, "tampered_reg.json")
        with open(tampered_path, "w", encoding="utf-8") as f:
            json.dump(tampered_reg, f)

        diff = [("M", "analyst_dashboard/analyzers/self_healing_engine.py")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=tampered_path,
        )
        self.assertFalse(ok)
        self.assertIn("UNAUTHORIZED_SIGNER", reason)

    # 10. Negative: Owner impersonation in hosted GitHub Actions fails closed
    def test_negative_owner_impersonation(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", "tests/test_self_healing_audit_claims.py"),
        ]
        spoofed_identity = {
            "event_name": "pull_request",
            "pr_author": "unauthorized_contributor",
            "author_association": "NONE",
        }
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            identity=spoofed_identity,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("FORBIDDEN_OWNER_IMPERSONATION", reason)

    # 11. Negative: Path traversal attempts fail closed
    def test_negative_path_traversal(self):
        diff = [("M", "../etc/passwd")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("PATH_TRAVERSAL_DETECTED", reason)

    # 12. Negative: Destructive deletion of application code fails closed
    def test_negative_unauthorized_deletion(self):
        diff = [("D", "analyst_dashboard/analyzers/self_healing_engine.py")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAUTHORIZED_FILE_OPERATION", reason)

    # 13. Negative: Missing registry fails closed
    def test_negative_missing_registry_fails_closed(self):
        diff = [("M", "analyst_dashboard/analyzers/self_healing_engine.py")]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=os.path.join(self.temp_dir.name, "nonexistent.json"),
        )
        self.assertFalse(ok)
        self.assertIn("AUTHORIZATION_REGISTRY_MISSING", reason)

    # 14. Negative: Target branch mismatch fails closed
    def test_negative_target_branch_mismatch(self):
        diff = [
            ("M", "analyst_dashboard/analyzers/self_healing_engine.py"),
            ("A", "tests/test_self_healing_audit_claims.py"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=207,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="release/production",  # Mismatch! Authorized for main
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("TARGET_BRANCH_MISMATCH", reason)

    # 15. Positive: Pure solo-owner governance PR targeting main passes without pre-registered number
    def test_positive_pure_governance_pr(self):
        diff = [
            ("M", "scripts/release/rq02_diff_integrity.py"),
            ("A", "scripts/release/pr_authorizations.json"),
            ("M", "docs/architecture.md"),
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,  # Any new governance PR number
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertTrue(ok, f"Expected pass, got: {reason}")
        self.assertEqual(reason, "GOVERNANCE_PR_AUTHORIZED")
        self.assertEqual(details["mode"], "SOLO_OWNER_GOVERNANCE_ADMISSION")

    # 16. Negative: Governance PR slipping in application code strictly rejected
    def test_negative_governance_pr_with_app_leak(self):
        diff = [
            ("M", "scripts/release/rq02_diff_integrity.py"),
            ("M", "api/routes/analytics.py"),  # Application code leak!
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAPPROVED_PR_NUMBER", reason)

    # 17. Negative: Governance PR slipping in secrets strictly rejected
    def test_negative_governance_pr_with_secret_leak(self):
        diff = [
            ("M", "scripts/release/rq02_diff_integrity.py"),
            ("A", ".secrets/leak.key"),  # Secret leak!
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAPPROVED_PR_NUMBER", reason)

    # 18. Negative: Governance PR slipping in release manifest strictly rejected
    def test_negative_governance_pr_with_manifest_leak(self):
        diff = [
            ("M", "scripts/release/rq02_diff_integrity.py"),
            ("M", "scripts/release/release_manifest.json"),  # Manifest tampering!
        ]
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("UNAPPROVED_PR_NUMBER", reason)

    # 19. Negative: Non-owner governance PR strictly rejected
    def test_negative_governance_pr_non_owner(self):
        diff = [
            ("M", "scripts/release/rq02_diff_integrity.py"),
        ]
        spoofed_identity = {
            "event_name": "pull_request",
            "pr_author": "unauthorized_user",
            "author_association": "COLLABORATOR",
        }
        ok, reason, details = verify_candidate_authorization(
            pr_number=999,
            head_sha=FIXTURE_HEAD_SHA,
            base_sha=FIXTURE_BASE_SHA,
            clean_target="main",
            diff_entries=diff,
            identity=spoofed_identity,
            registry_path=self.registry_file,
        )
        self.assertFalse(ok)
        self.assertIn("FORBIDDEN_OWNER_IMPERSONATION", reason)


if __name__ == "__main__":
    unittest.main()
