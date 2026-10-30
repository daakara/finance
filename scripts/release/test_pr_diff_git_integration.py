"""Real Git-repository adversarial tests for PR-differential qualification.

These tests do not write to the user's repository. Each case makes a temporary
Git repository with independently committed trusted-base approval and PR head.
Run: python -m unittest scripts/release/test_pr_diff_git_integration.py -v
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).with_name("pr_diff_qualification.py")
spec = importlib.util.spec_from_file_location("arx_pr_diff_qualification", MODULE)
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)

PR = 139
PATH = "frontend/app/compare/page.tsx"
APPROVAL = f"scripts/release/pr_approvals/{PR}.json"
CONTENT = "verified-fundamentals-only\n"


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr}")
    return result.stdout.strip()


class RealGitQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        git(self.root, "init")
        git(self.root, "config", "user.email", "arx-test@example.invalid")
        git(self.root, "config", "user.name", "ARX Test")
        self.actual = CONTENT
        self.approved = CONTENT
        self.path = PATH
        self.approval_paths = {PATH: {
            "operation": "M", "sha256": hashlib.sha256(CONTENT.encode()).hexdigest(),
        }}
        self.base = None
        self.head = None

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, message):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-m", message)
        return git(self.root, "rev-parse", "HEAD")

    def prepare(self, approval=True):
        self.write(self.path, "base-market-observation\n")
        if approval:
            self.write(APPROVAL, json.dumps({
                "schema_version": 1,
                "pr_number": PR,
                "files": self.approval_paths,
            }, sort_keys=True))
        self.base = self.commit("trusted base and qualification policy")
        self.write(self.path, self.actual)
        self.head = self.commit("candidate change")
        return self.base, self.head

    def qualify(self, base=None, head=None, pr=PR):
        with patch.object(qualifier, "git", wraps=qualifier.git):
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}, clear=True):
                with patch("os.getcwd", return_value=str(self.root)):
                    original = qualifier.git

                    def in_repo(*args, binary=False):
                        result = subprocess.run(
                            ["git", "-C", str(self.root), *args],
                            capture_output=True, check=False,
                        )
                        if result.returncode:
                            raise qualifier.QualificationFailure(
                                "GIT_FAILURE: " + result.stderr.decode(errors="replace")
                            )
                        return result.stdout if binary else result.stdout.decode().strip()

                    with patch.object(qualifier, "git", side_effect=in_repo):
                        return qualifier.qualify(
                            base or self.base, head or self.head, pr
                        )

    def test_exact_approved_content_passes(self):
        self.prepare()
        outcome = self.qualify()
        self.assertEqual(outcome["verdict"], "PASS")
        self.assertEqual(outcome["approval_source"], self.base + ":" + APPROVAL)

    def test_no_base_approval_fails_closed(self):
        self.prepare(approval=False)
        with self.assertRaisesRegex(qualifier.QualificationFailure, "NO_TRUSTED_BASE_APPROVAL"):
            self.qualify()

    def test_unapproved_changed_content_fails(self):
        self.actual = "forged-fundamental-metric\n"
        self.prepare()
        with self.assertRaisesRegex(qualifier.QualificationFailure, "CONTENT_HASH_MISMATCH"):
            self.qualify()

    def test_unapproved_new_file_fails(self):
        self.prepare()
        self.write("frontend/app/unsafe.ts", "export const active = true;\n")
        self.head = self.commit("unapproved added file")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "UNAPPROVED_OR_MISSING_FILES"):
            self.qualify()

    def test_wrong_base_fails(self):
        self.prepare()
        wrong = self.head
        with self.assertRaises(qualifier.QualificationFailure):
            self.qualify(base=wrong)

    def test_rewritten_or_unrelated_head_fails_ancestry(self):
        self.prepare()
        git(self.root, "checkout", "--orphan", "unrelated")
        git(self.root, "rm", "-r", "--cached", ".")
        self.write(self.path, "unrelated candidate\n")
        self.head = self.commit("unrelated root")
        with self.assertRaises(qualifier.QualificationFailure):
            self.qualify()

    def test_candidate_cannot_self_authorize(self):
        self.prepare(approval=False)
        self.write(APPROVAL, json.dumps({
            "schema_version": 1, "pr_number": PR,
            "files": self.approval_paths,
        }))
        self.head = self.commit("candidate writes its own approval")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "NO_TRUSTED_BASE_APPROVAL"):
            self.qualify()

    def test_unapproved_deletion_fails(self):
        self.prepare()
        (self.root / self.path).unlink()
        self.head = self.commit("remove previously approved modified file")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "OPERATION_MISMATCH"):
            self.qualify()

    def test_rename_cannot_reuse_old_path_approval(self):
        self.prepare()
        new_path = "frontend/app/renamed-compare.tsx"
        (self.root / new_path).write_text(self.actual, encoding="utf-8")
        (self.root / self.path).unlink()
        self.head = self.commit("rename previously approved file")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "UNAPPROVED_OR_MISSING_FILES"):
            self.qualify()

    def test_candidate_changed_approval_is_rejected(self):
        self.prepare()
        self.write(APPROVAL, json.dumps({
            "schema_version": 1,
            "pr_number": PR,
            "files": {
                PATH: {
                    "operation": "M",
                    "sha256": "0" * 64,
                }
            },
        }))
        self.head = self.commit("candidate tampers approval registry")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "PROTECTED_QUALIFICATION_FILE_CHANGED"):
            self.qualify()

    def test_protected_qualification_framework_edit_fails(self):
        protected = "scripts/release/rq02_diff_integrity.py"
        self.approval_paths[protected] = {
            "operation": "A",
            "sha256": hashlib.sha256(b"tampered\n").hexdigest(),
        }
        self.prepare()
        self.write(protected, "tampered\n")
        self.head = self.commit("attempt to edit framework")
        with self.assertRaisesRegex(qualifier.QualificationFailure, "PROTECTED_QUALIFICATION_FILE_CHANGED"):
            self.qualify()


if __name__ == "__main__":
    unittest.main()
