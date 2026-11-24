"""
tests/test_offline_fundamentals_reader.py

Offline regression tests for Sourced Fundamentals offline reader integration:
- Verified hash, issuer CIK, entity name, and accession checks.
- Resolution of acceptanceDateTime from retained submissions JSON.
- Separation of research cutoff (2026-10-16) from currentness retrieval date (2026-11-22).
- Preservation of null values (UNAVAILABLE) without synthetic zero substitution.
- Tamper detection and fail-closed security for corrupted hashes or CIKs.
- Anti-masking protection preventing runtime factor estimates from masking verified filings.
- Production scoring and publication held under HOLD_SEC_001.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from analyst_dashboard.research.sourced_fundamentals.offline_reader import (
    OfflineFundamentalsReader,
    load_offline_fundamentals,
    protect_against_runtime_masking,
    ManifestIntegrityError,
    IdentityIntegrityError,
    compute_file_sha256,
)
from analyst_dashboard.research.sourced_fundamentals.contracts import MetricStatus


class OfflineFundamentalsReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = OfflineFundamentalsReader()

    def test_offline_reader_success_nvda(self):
        """NVDA offline fundamentals load with verified hashes, dates, and accessions."""
        result = self.reader.load_symbol("NVDA")
        self.assertEqual(result.symbol, "NVDA")
        self.assertEqual(result.cik, "0001045810")
        self.assertEqual(result.company_name, "NVIDIA CORP")

        # Temporal separation: cutoff vs retrieval currentness
        self.assertEqual(result.research_cutoff_date, "2026-10-16")
        self.assertEqual(result.retrieval_date, "2026-11-22")
        self.assertEqual(result.fundamentals.latest_filing_date, "2026-08-26")
        self.assertEqual(result.fundamentals.latest_reported_period, "2026-07-26")

        # Production scoring held
        self.assertTrue(result.scoring_held)
        self.assertEqual(result.holding_code, "HOLD_SEC_001")
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.provenance["isVerified"])

        # Cryptographic hash verification against custody manifest
        manifest_files = result.manifest["files"]
        for fn, expected_sha in result.verified_hashes.items():
            self.assertEqual(expected_sha.lower(), manifest_files[fn]["sha256"].lower())

        # Fact accessions and acceptanceDateTime resolution
        rev = result.fundamentals.facts.get("revenue_ttm")
        self.assertIsNotNone(rev)
        self.assertEqual(rev.accession_number, "0001045810-26-000075")
        self.assertEqual(rev.acceptance_time, "2026-08-26T20:36:00.000Z")
        self.assertEqual(rev.filing_date, "2026-08-26")

        # Explicit unresolved lineage verification
        self.assertIn("unresolvedLineage", result.provenance)
        lineage = result.unresolved_lineage
        self.assertEqual(lineage["lineageStatus"], "PARTIAL_HISTORICAL_COVERAGE")
        self.assertEqual(lineage["unresolvedAccessionsCount"], 45)
        self.assertEqual(len(lineage["unresolvedAccessions"]), 45)

    def test_null_preservation_no_synthetic_zero(self):
        """Metrics with no reported XBRL facts remain None with status UNAVAILABLE."""
        result = self.reader.load_symbol("NVDA")
        facts = result.fundamentals.facts

        # short_term_investments
        sti = facts.get("short_term_investments")
        self.assertIsNotNone(sti)
        self.assertIsNone(sti.value)
        self.assertEqual(sti.status, MetricStatus.UNAVAILABLE)

        # total_cash_and_investments
        tcai = facts.get("total_cash_and_investments")
        self.assertIsNotNone(tcai)
        self.assertIsNone(tcai.value)
        self.assertEqual(tcai.status, MetricStatus.UNAVAILABLE)

        # common_shares_outstanding
        cso = facts.get("common_shares_outstanding")
        self.assertIsNotNone(cso)
        self.assertIsNone(cso.value)
        self.assertEqual(cso.status, MetricStatus.UNAVAILABLE)

    def test_unmatched_accession_remains_unresolved(self):
        """Accessions not present in the recent submissions index keep acceptance_time = None."""
        norm = self.reader.normalizer
        norm._accession_map = {"0001045810-26-000075": "2026-08-26T20:36:00.000Z"}
        # Matched accession
        self.assertEqual(norm._resolve_acceptance_time("0001045810-26-000075"), "2026-08-26T20:36:00.000Z")
        # Unmatched older accession
        self.assertIsNone(norm._resolve_acceptance_time("0001045810-19-000023"))
        # Empty or None accession
        self.assertIsNone(norm._resolve_acceptance_time(None))
        self.assertIsNone(norm._resolve_acceptance_time(""))

    def test_uncataloged_symbol_fails_closed(self):
        """Uncataloged symbol fails closed with IdentityIntegrityError."""
        with self.assertRaises(IdentityIntegrityError):
            self.reader.load_symbol("UNKNOWN_TICKER_999")

    def test_tampered_manifest_hash_fails_closed(self):
        """Tampered file hash in manifest fails closed with ManifestIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cik = "0001045810"
            cik_dir = tmp_path / f"CIK{cik}"
            cik_dir.mkdir(parents=True)

            sub_file = cik_dir / f"CIK{cik}_submissions.json"
            sub_file.write_text(json.dumps({"cik": cik, "entityName": "NVIDIA CORP", "filings": {"recent": {}}}))
            cf_file = cik_dir / f"CIK{cik}_companyfacts.json"
            cf_file.write_text(json.dumps({"cik": int(cik), "entityName": "NVIDIA CORP", "facts": {"us-gaap": {}}}))

            # Corrupted hash in manifest
            manifest = {
                "cik": cik,
                "symbol": "NVDA",
                "files": {
                    f"CIK{cik}_submissions.json": {
                        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                        "size_bytes": sub_file.stat().st_size,
                    },
                    f"CIK{cik}_companyfacts.json": {
                        "sha256": compute_file_sha256(cf_file),
                        "size_bytes": cf_file.stat().st_size,
                    },
                },
            }
            (cik_dir / "manifest.json").write_text(json.dumps(manifest))

            temp_reader = OfflineFundamentalsReader(raw_base_dir=tmp_path)
            with self.assertRaises(ManifestIntegrityError) as ctx:
                temp_reader.load_symbol("NVDA")
            self.assertIn("hash mismatch", str(ctx.exception).lower())

    def test_tampered_payload_cik_fails_closed(self):
        """Inner payload CIK mismatch fails closed with IdentityIntegrityError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cik = "0001045810"
            cik_dir = tmp_path / f"CIK{cik}"
            cik_dir.mkdir(parents=True)

            # Submissions file has wrong CIK (spoofed Apple CIK)
            sub_file = cik_dir / f"CIK{cik}_submissions.json"
            sub_file.write_text(json.dumps({"cik": "0000320193", "entityName": "Apple Inc.", "filings": {"recent": {}}}))
            cf_file = cik_dir / f"CIK{cik}_companyfacts.json"
            cf_file.write_text(json.dumps({"cik": int(cik), "entityName": "NVIDIA CORP", "facts": {"us-gaap": {}}}))

            manifest = {
                "cik": cik,
                "symbol": "NVDA",
                "files": {
                    f"CIK{cik}_submissions.json": {
                        "sha256": compute_file_sha256(sub_file),
                        "size_bytes": sub_file.stat().st_size,
                    },
                    f"CIK{cik}_companyfacts.json": {
                        "sha256": compute_file_sha256(cf_file),
                        "size_bytes": cf_file.stat().st_size,
                    },
                },
            }
            (cik_dir / "manifest.json").write_text(json.dumps(manifest))

            temp_reader = OfflineFundamentalsReader(raw_base_dir=tmp_path)
            with self.assertRaises(IdentityIntegrityError) as ctx:
                temp_reader.load_symbol("NVDA")
            self.assertIn("does not match authoritative CIK", str(ctx.exception))

    def test_anti_masking_guard(self):
        """Runtime estimates cannot mask verified filing-backed facts; scoring remains held."""
        filing_result = load_offline_fundamentals("NVDA")
        runtime_factor = {
            "symbol": "NVDA",
            "source": "LIVE_RUNTIME",
            "qualityScore": 99.0,
            "piotroski_f": 9,
            "roic": 45.2,
        }

        guard_result = protect_against_runtime_masking(filing_result, runtime_factor)
        self.assertTrue(guard_result["hasFilingEvidence"])
        self.assertTrue(guard_result["runtimeMaskingPrevented"])
        self.assertEqual(guard_result["filingSource"], "sec_edgar_offline")
        self.assertEqual(guard_result["runtimeSource"], "LIVE_RUNTIME")
        self.assertTrue(guard_result["scoringHeld"])
        self.assertEqual(guard_result["holdingCode"], "HOLD_SEC_001")
        self.assertEqual(guard_result["status"], "unavailable")
        self.assertIn("ungrounded runtime estimates cannot mask filing facts", guard_result["disqualificationReason"])


if __name__ == "__main__":
    unittest.main()
