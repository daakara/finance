"""Adversarial tests for base-anchored PR differential qualification."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parent / "pr_diff_qualification.py"
spec = importlib.util.spec_from_file_location("pr_diff_qualification", FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BASE = "a" * 40
HEAD = "b" * 40
PR = 139
PATH = "frontend/app/compare/page.tsx"
CONTENT = b"approved comparison content\n"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


class DifferentialContractTests(unittest.TestCase):
    def setUp(self):
        self.approval = {
            "schema_version": 1,
            "pr_number": PR,
            "files": {PATH: {"operation": "M", "sha256": DIGEST}},
        }
        self.diff = "M\t" + PATH
        self.actual = CONTENT
        self.event = None

    def mock_git(self, *args, binary=False):
        if args[0] in ("cat-file", "merge-base"):
            return b"" if binary else ""
        if args[0] == "show":
            if args[1] == f"{BASE}:scripts/release/pr_approvals/{PR}.json":
                return json.dumps(self.approval)
            if args[1] == f"{HEAD}:{PATH}":
                return self.actual
            raise mod.QualificationFailure("NOT_FOUND")
        if args[0] == "diff":
            return self.diff
        raise AssertionError(args)

    def evaluate(self):
        with patch.object(mod, "git", side_effect=self.mock_git):
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=True):
                return mod.qualify(BASE, HEAD, PR)

    def test_approved_exact_inventory_passes(self):
        receipt = self.evaluate()
        self.assertEqual(receipt["verdict"], "PASS")
        self.assertEqual(receipt["files"][PATH]["sha256"], DIGEST)
        self.assertIn(BASE, receipt["approval_source"])

    def test_no_trusted_approval_fails(self):
        self.approval = {}
        original = self.mock_git

        def no_approval(*args, **kwargs):
            if args[0] == "show" and args[1].endswith(f"/{PR}.json"):
                raise mod.QualificationFailure("MISSING")
            return original(*args, **kwargs)

        with patch.object(mod, "git", side_effect=no_approval), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(mod.QualificationFailure, "NO_TRUSTED_BASE_APPROVAL"):
                mod.qualify(BASE, HEAD, PR)

    def test_changed_content_hash_rejected(self):
        self.actual = b"unauthorized edit"
        with self.assertRaisesRegex(mod.QualificationFailure, "CONTENT_HASH_MISMATCH"):
            self.evaluate()

    def test_unapproved_file_rejected(self):
        self.diff += "\nA\tfrontend/extra.ts"
        with self.assertRaisesRegex(mod.QualificationFailure, "UNAPPROVED_OR_MISSING_FILES"):
            self.evaluate()

    def test_missing_approved_file_rejected(self):
        self.diff = ""
        with self.assertRaisesRegex(mod.QualificationFailure, "UNAPPROVED_OR_MISSING_FILES"):
            self.evaluate()

    def test_operation_mismatch_rejected(self):
        self.approval["files"][PATH]["operation"] = "A"
        with self.assertRaisesRegex(mod.QualificationFailure, "OPERATION_MISMATCH"):
            self.evaluate()

    def test_qualification_framework_tampering_rejected(self):
        self.diff = "M\tscripts/release/rq02_diff_integrity.py"
        self.approval["files"] = {
            "scripts/release/rq02_diff_integrity.py": {
                "operation": "M", "sha256": DIGEST,
            }
        }
        with self.assertRaisesRegex(mod.QualificationFailure, "PROTECTED_QUALIFICATION_FILE_CHANGED"):
            self.evaluate()

    def test_duplicate_approval_key_fails(self):
        original = self.mock_git

        def duplicate(*args, **kwargs):
            if args[0] == "show" and args[1].endswith(f"/{PR}.json"):
                return '{"schema_version":1,"schema_version":1,"pr_number":139,"files":{}}'
            return original(*args, **kwargs)

        with patch.object(mod, "git", side_effect=duplicate), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(mod.QualificationFailure, "DUPLICATE_APPROVAL_KEY"):
                mod.qualify(BASE, HEAD, PR)

    def test_wrong_pr_rejected(self):
        self.approval["pr_number"] = 140
        with self.assertRaisesRegex(mod.QualificationFailure, "APPROVAL_PR_OR_VERSION_MISMATCH"):
            self.evaluate()

    def test_untrusted_ci_event_rejected(self):
        with patch.object(mod, "git", side_effect=self.mock_git):
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True):
                with self.assertRaisesRegex(mod.QualificationFailure, "UNTRUSTED_CI_EVENT_CONTEXT"):
                    mod.qualify(BASE, HEAD, PR)

    def test_wrong_ci_head_rejected(self):
        repo = {"full_name": "daakara/finance"}
        event = {
            "number": PR,
            "repository": repo,
            "pull_request": {
                "number": PR,
                "head": {"sha": "c" * 40, "repo": repo},
                "base": {"sha": BASE, "repo": repo},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(event, stream)
            stream.flush()
            with patch.object(mod, "git", side_effect=self.mock_git):
                with patch.dict(os.environ, {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_REPOSITORY": "daakara/finance",
                    "GITHUB_EVENT_PATH": stream.name,
                }, clear=True):
                    with self.assertRaisesRegex(mod.QualificationFailure, "CI_EVENT_IDENTITY_MISMATCH"):
                        mod.qualify(BASE, HEAD, PR)

    def test_forked_ci_repository_rejected(self):
        repo = {"full_name": "daakara/finance"}
        event = {
            "number": PR,
            "repository": repo,
            "pull_request": {
                "number": PR,
                "head": {"sha": HEAD, "repo": {"full_name": "attacker/finance"}},
                "base": {"sha": BASE, "repo": repo},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as stream:
            json.dump(event, stream)
            stream.flush()
            with patch.object(mod, "git", side_effect=self.mock_git):
                with patch.dict(os.environ, {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_EVENT_NAME": "pull_request",
                    "GITHUB_REPOSITORY": "daakara/finance",
                    "GITHUB_EVENT_PATH": stream.name,
                }, clear=True):
                    with self.assertRaisesRegex(mod.QualificationFailure, "CI_REPOSITORY_IDENTITY_MISMATCH"):
                        mod.qualify(BASE, HEAD, PR)

    def test_invalid_head_sha_fails(self):
        with self.assertRaisesRegex(mod.QualificationFailure, "INVALID_HEAD"):
            mod.qualify(BASE, "abc", PR)


if __name__ == "__main__":
    unittest.main()
