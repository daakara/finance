import unittest
import json
import hashlib
from pathlib import Path
import sys

# Ensure scripts directory is importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "scripts" / "research"))

from verify_discovery_pilot import verify_pilot, FROZEN_BASELINE_LEDGER_SHA256, AUTHORIZED_PUBLISHED_SYMBOLS


class TestEvidenceLedDiscoveryPilot(unittest.TestCase):
    def test_discovery_pilot_verification(self):
        res = verify_pilot(repo_root)
        self.assertTrue(res["passed"], f"Verification failed: {res['errors']}")
        self.assertEqual(res["records_verified"], 3)

    def test_public_publication_ineligibility(self):
        ledger_path = repo_root / "docs" / "research" / "evidence_led_discovery" / "discovery_opportunity_ledger.json"
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for symbol, rec in data["deep_investigations"].items():
            dim = rec["dimensional_classification"]
            self.assertEqual(
                dim["public_publication_eligibility"],
                "INTERNAL_PILOT_RECORD_NOT_ELIGIBLE",
                f"Record {symbol} must have public_publication_eligibility == INTERNAL_PILOT_RECORD_NOT_ELIGIBLE"
            )

    def test_published_watch_manifest_integrity(self):
        manifest_path = repo_root / "docs" / "research" / "opportunity_discovery" / "published_watch_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["published_symbols"], AUTHORIZED_PUBLISHED_SYMBOLS)
        self.assertEqual(manifest["published_count"], 6)

    def test_odp05_baseline_ledger_immutability(self):
        odp05_path = repo_root / "docs" / "research" / "opportunity_discovery" / "odp05_discovery_ledger.json"
        with open(odp05_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual_sha, FROZEN_BASELINE_LEDGER_SHA256)


if __name__ == "__main__":
    unittest.main()
