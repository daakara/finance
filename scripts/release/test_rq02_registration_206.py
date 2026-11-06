"""Adversarial controls for the single-use PR #206 registration bootstrap.

Run: python -m unittest scripts/release/test_rq02_registration_206.py
"""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rq02_registration_206 import (
    BASE_ANCHOR, SOURCE, SELF, TEST, verify_candidate_contents,
    verify_registration_only,
)

ORIGINAL = "intro\nAPPROVED_PRS = {196, 201}\n        # Candidate PR Authorization check\n        try:\n            pr_val = int(pr_number)\n            is_approved_pr = pr_val in APPROVED_PRS\n        except Exception:\n            is_approved_pr = False\n"

from rq02_registration_206 import EXPECTED_BOOTSTRAP_BLOCK
BOOTSTRAP = EXPECTED_BOOTSTRAP_BLOCK
CANDIDATE = ORIGINAL.replace("201}", "201, 205}").replace(
    "        # Candidate PR Authorization check\n",
    BOOTSTRAP + "        # Candidate PR Authorization check\n",
).replace(
    "is_approved_pr = pr_val in APPROVED_PRS",
    "is_approved_pr = (pr_val in APPROVED_PRS) or registration_bootstrap",
)

class RegistrationBootstrapTests(unittest.TestCase):
    def test_positive_single_pr_registration(self):
        ok, reason = verify_candidate_contents(ORIGINAL, CANDIDATE)
        self.assertTrue(ok, reason)

    def test_unrelated_code_edit_rejected(self):
        ok, _ = verify_candidate_contents(ORIGINAL, CANDIDATE.replace("is_approved_pr = False", "is_approved_pr = True"))
        self.assertFalse(ok)

    def test_extra_pr_number_rejected(self):
        ok, _ = verify_candidate_contents(ORIGINAL, CANDIDATE.replace("201, 205}", "201, 205, 999}"))
        self.assertFalse(ok)

    def test_wrong_governance_pr_rejected(self):
        ok, _ = verify_candidate_contents(ORIGINAL, CANDIDATE.replace('== "206"', '== "207"'))
        self.assertFalse(ok)

    def test_changed_application_file_rejected(self):
        with patch("rq02_registration_206._run_git") as git:
            git.side_effect = [
                f"M\\t{SOURCE}\\nA\\t{SELF}\\nA\\t{TEST}\\nM\\tapi/main.py\\n".replace("\\t", "\t").replace("\\n","\n").encode(),
            ]
            ok, reason = verify_registration_only(base_sha=BASE_ANCHOR, head_sha="a"*40, repo_root="/tmp")
        self.assertFalse(ok)
        self.assertEqual(reason, "UNEXPECTED_REGISTRATION_SCOPE")

    def test_wrong_base_rejected(self):
        ok, reason = verify_registration_only(base_sha="0"*40, head_sha="a"*40, repo_root="/tmp")
        self.assertFalse(ok)
        self.assertEqual(reason, "UNEXPECTED_REGISTRATION_BASE_ANCHOR")

if __name__ == "__main__":
    unittest.main()
