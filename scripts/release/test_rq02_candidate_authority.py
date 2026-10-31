"""Regression controls for candidate-bound framework hashes and PR identity."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq02_diff_integrity as rq

HEAD = "a" * 40
BASE = "b" * 40
EXPECTED = "c" * 64
OTHER = "d" * 64


class QualificationAuthorityTests(unittest.TestCase):
    def run_gate(self, mode, candidate_hash=EXPECTED, disk_hash=EXPECTED,
                 pr_number=147, valid_identity=True):
        identity = dict(source_sha=HEAD, base_sha=BASE, event_name="pull_request",
                        target_branch="main", pr_number=pr_number)
        receipts = []
        with tempfile.TemporaryDirectory() as temp:
            framework = Path(temp) / "framework.py"
            framework.write_text("fixture")
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({
                "base_anchor_commit": BASE, "base_commit": BASE,
                "release_candidate_commit": HEAD, "candidate_commit": HEAD,
                "qualification_framework_files": {str(framework): {"sha256": EXPECTED}},
                "files": {}
            }))
            argv = ["rq02", "--mode", mode, "--base", BASE, "--head", HEAD]
            if mode != "pr":
                argv += ["--manifest", str(manifest)]
            def blob(ref, path):
                # Historical objects match even when candidate is tampered.
                return candidate_hash if ref == HEAD else EXPECTED
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.dict(os.environ, {
                    "ARX_TARGET_BRANCH": "main",
                    "ARX_RQ02_SKIP_ADVERSARIAL_RECURSION": "1"
                }, clear=True))
                for name, value in {
                    "resolve_identity_contract": (valid_identity, identity, "invalid"),
                    "git_commit_exists": True,
                    "is_git_ancestor": True,
                    "run_git_merge_base": BASE,
                    "run_git_diff_status": [("M", "api/example.py")] if mode == "pr" else [],
                    "compute_file_sha256": disk_hash,
                    "save_stage_receipt": "unused",
                }.items():
                    stack.enter_context(patch.object(rq, name, return_value=value))
                stack.enter_context(patch.object(rq, "get_git_commit",
                                                side_effect=lambda ref="HEAD": BASE if ref == BASE else HEAD))
                stack.enter_context(patch.object(rq, "compute_git_blob_sha256", side_effect=blob))
                stack.enter_context(patch.object(rq, "create_stage_receipt",
                                                side_effect=lambda *a, **kw: kw))
                stack.enter_context(patch.object(rq, "save_stage_receipt",
                                                side_effect=lambda receipt: receipts.append(receipt)))
                stack.enter_context(patch.object(rq.subprocess, "check_output", return_value=""))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaises(SystemExit) as stopped:
                    rq.main()
            return stopped.exception.code, receipts[-1]

    def test_candidate_and_disk_match_pass(self):
        for mode in ("cumulative", "administrative"):
            with self.subTest(mode=mode):
                code, receipt = self.run_gate(mode)
                self.assertEqual(code, 0, receipt)

    def test_historical_match_cannot_mask_candidate_mismatch(self):
        for mode in ("cumulative", "administrative"):
            with self.subTest(mode=mode):
                code, receipt = self.run_gate(mode, candidate_hash=OTHER)
                self.assertEqual(code, 1)
                self.assertTrue(any("UNAUTHORIZED_QUALIFICATION_FRAMEWORK" in e
                                    for e in receipt["errors"]))

    def test_dirty_disk_cannot_be_masked_by_candidate_match(self):
        for mode in ("cumulative", "administrative"):
            with self.subTest(mode=mode):
                code, receipt = self.run_gate(mode, disk_hash=OTHER)
                self.assertEqual(code, 1)
                self.assertTrue(any("UNAUTHORIZED_QUALIFICATION_FRAMEWORK" in e
                                    for e in receipt["errors"]))

    def test_missing_candidate_blob_fails(self):
        for mode in ("cumulative", "administrative"):
            with self.subTest(mode=mode):
                code, receipt = self.run_gate(mode, candidate_hash=None)
                self.assertEqual(code, 1)

    def test_missing_or_unapproved_identity_fails(self):
        for number in (None, "", 0, 9999):
            with self.subTest(number=number):
                code, receipt = self.run_gate("pr", pr_number=number)
                self.assertEqual(code, 1)
                self.assertTrue(any("UNAPPROVED_PR_NUMBER" in e for e in receipt["errors"]))

    def test_invalid_identity_cannot_authorize_approved_number(self):
        code, receipt = self.run_gate("pr", valid_identity=False)
        self.assertEqual(code, 1)
        self.assertTrue(any("UNAPPROVED_PR_NUMBER" in e for e in receipt["errors"]))

    def test_valid_approved_identity_passes(self):
        code, receipt = self.run_gate("pr")
        self.assertEqual(code, 0, receipt)


if __name__ == "__main__":
    unittest.main()
