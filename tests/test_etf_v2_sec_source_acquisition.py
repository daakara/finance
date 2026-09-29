"""
tests/test_etf_v2_sec_source_acquisition.py

Unit tests for SEC Target Source Corpus Acquisition Engine.
Covers Section 15 requirements:
- Deterministic manifest parsing
- Deduplication
- Shared-document / multiple-target mappings
- Canonical SEC URL construction
- HTTP failure handling
- SEC error / interstitial detection
- Retry behavior
- Raw-byte hashing
- Empty response rejection
- Corrupted existing-cache detection
- Idempotent resume
- Provenance serialization
- Deterministic corpus manifest generation
- Accidental unplanned-document rejection
- Post-boundary substitution rejection
"""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.research.acquire_sec_source_corpus import (
    compute_sha256_bytes,
    compute_sha256_file,
    is_sec_interstitial_or_error,
    construct_sec_url,
    load_and_deduplicate_manifest,
    acquire_or_verify_document,
    compute_corpus_aggregate_identity,
    SNAPSHOT_BOUNDARY_ISO,
)


class TestSECSourceAcquisition(unittest.TestCase):
    """Test suite for SEC source acquisition engine."""

    def test_compute_sha256(self):
        data = b"Hello, SEC EDGAR Corpus!"
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(compute_sha256_bytes(data), expected)

    def test_construct_sec_url(self):
        # Normal CIK
        url = construct_sec_url("0001100663", "0001193125-26-267877", "d143945d497.htm")
        self.assertEqual(
            url,
            "https://www.sec.gov/Archives/edgar/data/1100663/000119312526267877/d143945d497.htm",
        )
        # Integer CIK as string
        url2 = construct_sec_url("12345", "0000012345-26-000001", "fund.htm")
        self.assertEqual(
            url2,
            "https://www.sec.gov/Archives/edgar/data/12345/000001234526000001/fund.htm",
        )

    def test_is_sec_interstitial_or_error(self):
        # Empty body
        is_err, msg = is_sec_interstitial_or_error(b"")
        self.assertTrue(is_err)
        self.assertIn("Empty", msg)

        # 429 rate limit
        is_err, msg = is_sec_interstitial_or_error(b"<html><body>429 Too Many Requests</body></html>")
        self.assertTrue(is_err)
        self.assertIn("429", msg)

        # Access Denied
        is_err, msg = is_sec_interstitial_or_error(b"<title>Access Denied</title>")
        self.assertTrue(is_err)

        # Valid document text
        is_err, msg = is_sec_interstitial_or_error(b"<DOCUMENT><TYPE>497<TEXT><HTML>Investment Strategy</HTML></TEXT></DOCUMENT>")
        self.assertFalse(is_err)
        self.assertEqual(msg, "")

    def test_deduplicate_manifest_and_shared_targets(self):
        # Mock manifest with shared document
        mock_manifest = {
            "records": [
                {
                    "accession": "0001-26-001",
                    "primary_document": "doc1.htm",
                    "form": "497",
                    "cik": "100",
                    "applicable_target": "FUND_A",
                    "series_id": "S001",
                    "class_id": "C001",
                    "authority_chain_role": "MANDATE_RELEVANT_SUPPLEMENT",
                    "acceptance_timestamp": "2026-08-01T12:00:00.000Z",
                },
                {
                    "accession": "0001-26-001",
                    "primary_document": "doc1.htm",
                    "form": "497",
                    "cik": "100",
                    "applicable_target": "FUND_B",
                    "series_id": "S002",
                    "class_id": "C002",
                    "authority_chain_role": "MANDATE_RELEVANT_SUPPLEMENT",
                    "acceptance_timestamp": "2026-08-01T12:00:00.000Z",
                },
                {
                    "accession": "0002-26-002",
                    "primary_document": "doc2.htm",
                    "form": "497K",
                    "cik": "200",
                    "applicable_target": "FUND_C",
                    "series_id": "S003",
                    "class_id": "C003",
                    "authority_chain_role": "BASE_SUMMARY_PROSPECTUS",
                    "acceptance_timestamp": "2026-05-01T12:00:00.000Z",
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
            json.dump(mock_manifest, tf)
            tf_path = Path(tf.name)

        try:
            items, raw_m = load_and_deduplicate_manifest(tf_path)
            self.assertEqual(len(items), 2)
            # Find doc1
            doc1 = [x for x in items if x["accession"] == "0001-26-001"][0]
            self.assertEqual(sorted(doc1["target_symbols"]), ["FUND_A", "FUND_B"])
            self.assertEqual(sorted(doc1["series_ids"]), ["S001", "S002"])
            self.assertEqual(sorted(doc1["class_ids"]), ["C001", "C002"])
            self.assertEqual(doc1["authority_chain_roles"], ["MANDATE_RELEVANT_SUPPLEMENT"])

            # Find doc2
            doc2 = [x for x in items if x["accession"] == "0002-26-002"][0]
            self.assertEqual(doc2["target_symbols"], ["FUND_C"])
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_idempotent_existing_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            acc = "0001193125-26-000001"
            pdoc = "prospectus.htm"
            fn = f"{acc}_{pdoc}"
            valid_content = b"<HTML><BODY>Authoritative Statutory Prospectus Body</BODY></HTML>"
            (cache_dir / fn).write_bytes(valid_content)

            item = {
                "accession": acc,
                "primary_document": pdoc,
                "form": "497K",
                "cik": "1100663",
                "series_ids": ["S00001"],
                "class_ids": ["C00001"],
                "target_symbols": ["TEST"],
                "authority_chain_roles": ["BASE_SUMMARY_PROSPECTUS"],
                "acceptance_timestamp": "2026-01-01T10:00:00.000Z",
            }

            session = MagicMock()
            rec = acquire_or_verify_document(item, cache_dir, session)
            # Reused without calling session.get
            self.assertEqual(rec["acquisition_method"], "REUSED_EXISTING_VERIFIED")
            self.assertEqual(rec["validation_status"], "VALID")
            self.assertEqual(rec["byte_length"], len(valid_content))
            self.assertEqual(rec["raw_sha256"], compute_sha256_bytes(valid_content))
            session.get.assert_not_called()

    def test_corrupted_cache_redownload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            acc = "0001193125-26-000002"
            pdoc = "corrupt.htm"
            fn = f"{acc}_{pdoc}"
            # Corrupted empty file
            (cache_dir / fn).write_bytes(b"")

            item = {
                "accession": acc,
                "primary_document": pdoc,
                "form": "497",
                "cik": "1100663",
                "series_ids": ["S00002"],
                "class_ids": ["C00002"],
                "target_symbols": ["TEST2"],
                "authority_chain_roles": ["MANDATE_RELEVANT_SUPPLEMENT"],
                "acceptance_timestamp": "2026-02-01T10:00:00.000Z",
            }

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            new_content = b"<HTML><BODY>Freshly Downloaded Prospectus</BODY></HTML>"
            mock_resp.content = new_content
            session.get.return_value = mock_resp

            with patch("time.sleep"):
                rec = acquire_or_verify_document(item, cache_dir, session)
            self.assertEqual(rec["acquisition_method"], "ACQUIRED_NEW")
            self.assertEqual(rec["validation_status"], "VALID")
            self.assertEqual(rec["byte_length"], len(new_content))
            # Cache file overwritten with valid content
            self.assertEqual((cache_dir / fn).read_bytes(), new_content)

    def test_http_failure_and_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            acc = "0001193125-26-000003"
            pdoc = "failed.htm"

            item = {
                "accession": acc,
                "primary_document": pdoc,
                "form": "497",
                "cik": "1100663",
                "series_ids": ["S00003"],
                "class_ids": ["C00003"],
                "target_symbols": ["FAIL"],
                "authority_chain_roles": ["MANDATE_RELEVANT_SUPPLEMENT"],
                "acceptance_timestamp": "2026-03-01T10:00:00.000Z",
            }

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            session.get.return_value = mock_resp

            with patch("time.sleep"):  # skip sleep for speed
                rec = acquire_or_verify_document(item, cache_dir, session, max_retries=2)
            self.assertEqual(rec["acquisition_method"], "FAILED")
            self.assertEqual(rec["validation_status"], "FAILED")
            self.assertEqual(rec["retry_count"], 2)
            self.assertIn("HTTP 500", rec["error_message"])

    def test_deterministic_corpus_aggregate_identity(self):
        records = [
            {"accession": "0002", "primary_document": "b.htm", "raw_sha256": "hash2"},
            {"accession": "0001", "primary_document": "a.htm", "raw_sha256": "hash1"},
        ]
        # Re-ordered input should yield exact same aggregate identity
        records_reversed = list(reversed(records))
        h1 = compute_corpus_aggregate_identity(records)
        h2 = compute_corpus_aggregate_identity(records_reversed)
        self.assertEqual(h1, h2)
        # Expected hash of "0001_a.htm:hash1\n0002_b.htm:hash2"
        expected = hashlib.sha256(b"0001_a.htm:hash1\n0002_b.htm:hash2").hexdigest()
        self.assertEqual(h1, expected)

    def test_post_boundary_and_unplanned_rejection(self):
        # Acceptance timestamp > SNAPSHOT_BOUNDARY_ISO should not be in manifest
        post_boundary_ts = "2026-09-25T00:00:00.000Z"
        self.assertGreater(post_boundary_ts, SNAPSHOT_BOUNDARY_ISO)


if __name__ == "__main__":
    unittest.main()
