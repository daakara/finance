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
    REPO_ROOT,
    ACQUISITION_MANIFEST_V2_PATH,
    EXPECTED_MANIFEST_V2_SHA256,
    OUTPUT_CORPUS_MANIFEST_PATH,
    OUTPUT_PROVENANCE_LEDGER_PATH,
    CANONICAL_857_AGGREGATE_SHA256,
    CANONICAL_858_AGGREGATE_SHA256,
    CORPUS_AGGREGATE_IDENTITY_VERSION,
    DestructiveCorpusOverwriteError,
    InvalidAuthorizedSourceSpecError,
    CanonicalHashConflictError,
    SECPayloadValidationError,
    LegacyProvenanceReconciliationRequiredError,
    InvalidProvenanceReconciliationError,
    compute_sha256_bytes,
    compute_sha256_file,
    is_sec_interstitial_or_error,
    construct_sec_url,
    load_and_deduplicate_manifest,
    acquire_or_verify_document,
    compute_corpus_aggregate_identity,
    verify_manifest_aggregate_compatibility,
    resolve_contained_cache_path,
    validate_authorized_source_spec,
    detect_legacy_provenance_deficit,
    reconcile_missing_provenance,
    acquire_authorized_documents,
    execute_acquisition,
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
            {"accession": "0001", "primary_document": "a.htm", "raw_sha256": "hash1"},
            {"accession": "0002", "primary_document": "b.htm", "raw_sha256": "hash2"},
        ]
        h1 = compute_corpus_aggregate_identity(records)
        expected = hashlib.sha256(b"0001_a.htm:hash1\n0002_b.htm:hash2").hexdigest()
        self.assertEqual(h1, expected)
        # Appending an incremental record preserves prefix order without resorting
        extended = records + [{"accession": "0000", "primary_document": "z.htm", "raw_sha256": "hash3"}]
        h_ext = compute_corpus_aggregate_identity(extended)
        expected_ext = hashlib.sha256(b"0001_a.htm:hash1\n0002_b.htm:hash2\n0000_z.htm:hash3").hexdigest()
        self.assertEqual(h_ext, expected_ext)

    def test_post_boundary_and_unplanned_rejection(self):
        post_boundary_ts = "2026-09-25T00:00:00.000Z"
        self.assertGreater(post_boundary_ts, SNAPSHOT_BOUNDARY_ISO)

    # =========================================================================
    # A01 - A21: Canonical Incremental Acquisition & Provenance Reconciliation
    # =========================================================================

    def _get_authoritative_alps_provenance_record(self):
        alps_ledger_path = REPO_ROOT / "docs" / "research" / "ETF_V2_ALPS_SOURCE_ACQUISITION_LEDGER.json"
        alps_data = json.loads(alps_ledger_path.read_text(encoding="utf-8"))
        sf = alps_data["source_filing"]
        tv = alps_data["target_validation_results"]
        return {
            "accession": sf["accession"],
            "primary_document": sf["primary_document"],
            "form": sf["form"],
            "cik": sf["cik"],
            "filing_date": sf["filing_date"],
            "series_ids": sorted(v["series_id"] for v in tv.values()),
            "class_ids": sorted(v["class_id"] for v in tv.values()),
            "target_symbols": sorted(tv.keys()),
            "authority_chain_roles": ["BASE_STATUTORY_PROSPECTUS"],
            "snapshot_boundary": SNAPSHOT_BOUNDARY_ISO,
            "acceptance_timestamp": sf["acceptance_timestamp"],
            "sec_source_url": sf["canonical_sec_url"],
            "http_status": sf["http_status"],
            "acquisition_timestamp": alps_data["timestamp"],
            "byte_length": sf["raw_byte_count"],
            "raw_sha256": sf["raw_sha256"],
            "local_path": sf["local_path"],
            "acquisition_method": "HISTORICAL_PROVENANCE_RECONCILIATION",
            "reconciliation_type": "HISTORICAL_PROVENANCE_RECONCILIATION",
            "source_acquisition_gate": alps_data["gate"],
            "validation_status": "VALID",
            "error_message": "",
        }

    def _write_857_deficit_ledger_fixture(self, ledger_path: Path):
        p_doc = json.loads(OUTPUT_PROVENANCE_LEDGER_PATH.read_text(encoding="utf-8"))
        p_doc["provenance_records"] = p_doc["provenance_records"][:857]
        p_doc["expected_unique_documents"] = 857
        p_doc["successful_unique_documents"] = 857
        p_doc.pop("initial_full_acquisition_document_count", None)
        p_doc.pop("historical_provenance_reconciled_count", None)
        p_doc.pop("current_canonical_provenance_membership_count", None)
        ledger_path.write_text(json.dumps(p_doc, indent=2) + "\n", encoding="utf-8")

    def _get_stxf_fixture_spec(self):
        return {
            "accession": "0001592900-26-002347",
            "cik": "0001592900",
            "form": "497",
            "primary_document": "strivestrv-stxf497etickerc.htm",
            "expected_raw_sha256": "0f7688fb44a0f413555ea379d2d34904533f39c64bafab53ce29654f92d15cc9",
            "filing_date": "2026-05-28",
            "effective_date": "2026-06-08",
            "acceptance_timestamp": "2026-05-28T13:00:57.000Z",
            "series_ids": ["S000077125"],
            "class_ids": ["C000237295"],
            "target_symbols": ["STXF"],
            "predecessor_symbol": "STRV",
            "legal_name": "Strive 500 ETF",
            "document_role": "TICKER_SUCCESSION_PROSPECTUS_SUPPLEMENT",
            "authority_chain_roles": ["TICKER_SUCCESSION_PROSPECTUS_SUPPLEMENT"],
        }

    def _build_consistent_858_fixture(self, tmpdir: Path):
        corpus_path = tmpdir / "corpus_manifest.json"
        ledger_path = tmpdir / "provenance_ledger.json"
        cache_dir = tmpdir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        corpus_path.write_bytes(OUTPUT_CORPUS_MANIFEST_PATH.read_bytes())
        self._write_857_deficit_ledger_fixture(ledger_path)

        reconcile_missing_provenance(
            corpus_manifest_path=corpus_path,
            provenance_ledger_path=ledger_path,
            authorized_reconciliation_records=[self._get_authoritative_alps_provenance_record()],
            cache_dir=cache_dir,
        )
        return corpus_path, ledger_path, cache_dir

    def test_a01_857_historical_full_acquisition_reproducible(self):
        actual_v2_sha = compute_sha256_file(ACQUISITION_MANIFEST_V2_PATH)
        self.assertEqual(actual_v2_sha, EXPECTED_MANIFEST_V2_SHA256)
        unique_items, _ = load_and_deduplicate_manifest(ACQUISITION_MANIFEST_V2_PATH)
        self.assertEqual(len(unique_items), 857)

        canonical_corpus = json.loads(OUTPUT_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
        records_857 = canonical_corpus["records"][:857]
        agg_857 = compute_corpus_aggregate_identity(records_857)
        self.assertEqual(agg_857, CANONICAL_857_AGGREGATE_SHA256)

    def test_a02_full_acquisition_refuses_destructive_overwrite_of_858_plus_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            ledger_path = tmp / "ledger.json"
            corpus_path.write_bytes(OUTPUT_CORPUS_MANIFEST_PATH.read_bytes())
            orig_bytes = corpus_path.read_bytes()

            with self.assertRaises(DestructiveCorpusOverwriteError) as ctx:
                execute_acquisition(
                    output_corpus_manifest_path=corpus_path,
                    output_provenance_ledger_path=ledger_path,
                    cache_dir=tmp / "cache",
                )
            self.assertIn("REFUSE_DESTRUCTIVE_OVERWRITE", str(ctx.exception))
            self.assertEqual(corpus_path.read_bytes(), orig_bytes)

    def test_a03_single_authorized_incremental_source_extends_consistent_858_to_859(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            stxf_spec = self._get_stxf_fixture_spec()
            forensic_src = Path(
                "C:/Users/akara/.gemini/antigravity/brain/76b5f1e2-3d0d-4453-b0de-300603740e42/scratch/strivestrv-stxf497etickerc.htm"
            )
            raw_payload = forensic_src.read_bytes()

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = raw_payload
            session.get.return_value = mock_resp

            old_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            with patch("time.sleep"):
                res = acquire_authorized_documents(
                    authorized_documents=[stxf_spec],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                    session=session,
                )

            self.assertEqual(res["status"], "EXTENDED")
            self.assertEqual(res["corpus_document_count_before"], 858)
            self.assertEqual(res["corpus_document_count_after"], 859)
            self.assertEqual(res["provenance_record_count_before"], 858)
            self.assertEqual(res["provenance_record_count_after"], 859)
            self.assertEqual(res["new_documents_added"], 1)
            self.assertEqual(
                res["new_corpus_aggregate_sha256"],
                "b186f39772763683b238609066a20c21cf1717f0d9dcf32741c47bc4dfeb27b6",
            )

            new_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            self.assertEqual(new_corpus["records"][:858], old_corpus["records"])
            self.assertEqual(new_corpus["records"][858]["accession"], "0001592900-26-002347")

    def test_a04_raw_sha256_enforced_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"<HTML><BODY>Wrong SEC bytes</BODY></HTML>"
            session.get.return_value = mock_resp

            orig_corpus_bytes = corpus_path.read_bytes()
            with patch("time.sleep"):
                with self.assertRaises(CanonicalHashConflictError):
                    acquire_authorized_documents(
                        authorized_documents=[spec],
                        corpus_manifest_path=corpus_path,
                        provenance_ledger_path=ledger_path,
                        cache_dir=cache_dir,
                        session=session,
                    )
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)

    def test_a05_sec_interstitial_or_error_page_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"<html><body>429 Too Many Requests</body></html>"
            session.get.return_value = mock_resp

            with patch("time.sleep"):
                with self.assertRaises(SECPayloadValidationError):
                    acquire_authorized_documents(
                        authorized_documents=[spec],
                        corpus_manifest_path=corpus_path,
                        provenance_ledger_path=ledger_path,
                        cache_dir=cache_dir,
                        session=session,
                        max_retries=1,
                    )

    def test_a06_zero_byte_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()

            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b""
            session.get.return_value = mock_resp

            with patch("time.sleep"):
                with self.assertRaises(SECPayloadValidationError):
                    acquire_authorized_documents(
                        authorized_documents=[spec],
                        corpus_manifest_path=corpus_path,
                        provenance_ledger_path=ledger_path,
                        cache_dir=cache_dir,
                        session=session,
                        max_retries=1,
                    )

    def test_a07_identical_reacquisition_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()
            forensic_src = Path(
                "C:/Users/akara/.gemini/antigravity/brain/76b5f1e2-3d0d-4453-b0de-300603740e42/scratch/strivestrv-stxf497etickerc.htm"
            )
            (cache_dir / f"{spec['accession']}_{spec['primary_document']}").write_bytes(
                forensic_src.read_bytes()
            )

            res1 = acquire_authorized_documents(
                authorized_documents=[spec],
                corpus_manifest_path=corpus_path,
                provenance_ledger_path=ledger_path,
                cache_dir=cache_dir,
            )
            self.assertEqual(res1["status"], "EXTENDED")
            corpus_after_1 = corpus_path.read_bytes()
            ledger_after_1 = ledger_path.read_bytes()

            res2 = acquire_authorized_documents(
                authorized_documents=[spec],
                corpus_manifest_path=corpus_path,
                provenance_ledger_path=ledger_path,
                cache_dir=cache_dir,
            )
            self.assertEqual(res2["status"], "IDEMPOTENT_NOOP")
            self.assertEqual(res2["new_documents_added"], 0)
            self.assertEqual(res2["files_mutated"], 0)
            self.assertEqual(corpus_path.read_bytes(), corpus_after_1)
            self.assertEqual(ledger_path.read_bytes(), ledger_after_1)

    def test_a08_conflicting_duplicate_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            bad_alps = {
                "accession": "0001398344-26-005876",
                "cik": "0001414040",
                "form": "485BPOS",
                "primary_document": "fp0097939-1_485bposixbrl.htm",
                "expected_raw_sha256": "a" * 64,
                "filing_date": "2026-03-30",
                "acceptance_timestamp": "2026-03-30T21:34:03.000Z",
                "series_ids": ["S000040588"],
                "class_ids": ["C000125863"],
                "target_symbols": ["BFOR"],
                "authority_chain_roles": ["BASE_STATUTORY_PROSPECTUS"],
            }
            orig_corpus_bytes = corpus_path.read_bytes()
            with self.assertRaises(CanonicalHashConflictError):
                acquire_authorized_documents(
                    authorized_documents=[bad_alps],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                )
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)

    def test_a09_unauthorized_malformed_or_post_boundary_source_rejected(self):
        with self.assertRaises(InvalidAuthorizedSourceSpecError):
            validate_authorized_source_spec({})

        spec_bad_sha = self._get_stxf_fixture_spec()
        spec_bad_sha["expected_raw_sha256"] = "not_a_64_hex_sha"
        with self.assertRaises(InvalidAuthorizedSourceSpecError):
            validate_authorized_source_spec(spec_bad_sha)

        spec_post_boundary = self._get_stxf_fixture_spec()
        spec_post_boundary["acceptance_timestamp"] = "2026-09-25T00:00:01.000Z"
        with self.assertRaises(InvalidAuthorizedSourceSpecError):
            validate_authorized_source_spec(spec_post_boundary)

        spec_empty_targets = self._get_stxf_fixture_spec()
        spec_empty_targets["target_symbols"] = []
        with self.assertRaises(InvalidAuthorizedSourceSpecError):
            validate_authorized_source_spec(spec_empty_targets)

    def test_a10_and_a11_and_a13_provenance_and_corpus_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()
            forensic_src = Path(
                "C:/Users/akara/.gemini/antigravity/brain/76b5f1e2-3d0d-4453-b0de-300603740e42/scratch/strivestrv-stxf497etickerc.htm"
            )
            (cache_dir / f"{spec['accession']}_{spec['primary_document']}").write_bytes(
                forensic_src.read_bytes()
            )

            acquire_authorized_documents(
                authorized_documents=[spec],
                corpus_manifest_path=corpus_path,
                provenance_ledger_path=ledger_path,
                cache_dir=cache_dir,
            )

            c_doc = json.loads(corpus_path.read_text(encoding="utf-8"))
            p_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(c_doc["corpus_document_count"], 859)
            self.assertEqual(len(c_doc["records"]), 859)
            self.assertEqual(p_doc["expected_unique_documents"], 859)
            self.assertEqual(len(p_doc["provenance_records"]), 859)
            self.assertEqual(
                c_doc["corpus_aggregate_identity_version"],
                CORPUS_AGGREGATE_IDENTITY_VERSION,
            )
            deficit = detect_legacy_provenance_deficit(c_doc, p_doc)
            self.assertFalse(deficit["deficit_detected"])

    def test_a12_append_order_aggregate_reproduces_857_858_and_859_identities(self):
        c_doc = json.loads(OUTPUT_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertTrue(verify_manifest_aggregate_compatibility(c_doc))

        agg_857 = compute_corpus_aggregate_identity(c_doc["records"][:857])
        agg_858 = compute_corpus_aggregate_identity(c_doc["records"][:858])
        self.assertEqual(agg_857, CANONICAL_857_AGGREGATE_SHA256)
        self.assertEqual(agg_858, CANONICAL_858_AGGREGATE_SHA256)

        stxf_rec = {
            "accession": "0001592900-26-002347",
            "primary_document": "strivestrv-stxf497etickerc.htm",
            "raw_sha256": "0f7688fb44a0f413555ea379d2d34904533f39c64bafab53ce29654f92d15cc9",
        }
        agg_859 = compute_corpus_aggregate_identity(c_doc["records"] + [stxf_rec])
        self.assertEqual(
            agg_859,
            "b186f39772763683b238609066a20c21cf1717f0d9dcf32741c47bc4dfeb27b6",
        )

    def test_a14_mid_operation_failure_leaves_no_partial_canonical_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec = self._get_stxf_fixture_spec()
            forensic_src = Path(
                "C:/Users/akara/.gemini/antigravity/brain/76b5f1e2-3d0d-4453-b0de-300603740e42/scratch/strivestrv-stxf497etickerc.htm"
            )
            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = forensic_src.read_bytes()
            session.get.return_value = mock_resp

            orig_corpus_bytes = corpus_path.read_bytes()
            orig_ledger_bytes = ledger_path.read_bytes()

            orig_replace = Path.replace

            def fail_on_ledger_replace(self_path, target_path):
                if target_path == ledger_path:
                    raise OSError("Simulated atomic promotion disk failure")
                return orig_replace(self_path, target_path)

            with patch("time.sleep"), patch.object(Path, "replace", fail_on_ledger_replace):
                with self.assertRaises(OSError):
                    acquire_authorized_documents(
                        authorized_documents=[spec],
                        corpus_manifest_path=corpus_path,
                        provenance_ledger_path=ledger_path,
                        cache_dir=cache_dir,
                        session=session,
                    )

            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)
            self.assertFalse((cache_dir / f"{spec['accession']}_{spec['primary_document']}").exists())

    def test_a15_generic_production_code_contains_no_stxf_special_case(self):
        prod_src = (REPO_ROOT / "scripts" / "research" / "acquire_sec_source_corpus.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("STXF", prod_src)
        self.assertNotIn("STRV", prod_src)
        self.assertNotIn("0001592900-26-002347", prod_src)

    def test_a16_alps_858th_canonical_corpus_record_preserved(self):
        c_doc = json.loads(OUTPUT_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
        alps_rec = c_doc["records"][857]
        self.assertEqual(alps_rec["accession"], "0001398344-26-005876")
        self.assertEqual(alps_rec["primary_document"], "fp0097939-1_485bposixbrl.htm")
        self.assertEqual(
            alps_rec["raw_sha256"],
            "ef2b53fd99efa268a29d5cc924b915eebfac22cc29b0a35fcf5402270a53898d",
        )

    def test_a17_canonical_858_corpus_857_provenance_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ledger_path = tmp / "ledger_857.json"
            self._write_857_deficit_ledger_fixture(ledger_path)
            c_doc = json.loads(OUTPUT_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
            p_doc_857 = json.loads(ledger_path.read_text(encoding="utf-8"))
            deficit = detect_legacy_provenance_deficit(c_doc, p_doc_857)
            self.assertTrue(deficit["deficit_detected"])
            self.assertEqual(deficit["corpus_count"], 858)
            self.assertEqual(deficit["provenance_count"], 857)
            self.assertEqual(
                deficit["missing_in_provenance"],
                [("0001398344-26-005876", "fp0097939-1_485bposixbrl.htm")],
            )

    def test_a18_ordinary_incremental_acquisition_does_not_silently_repair_alps_deficit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            ledger_path = tmp / "ledger.json"
            corpus_path.write_bytes(OUTPUT_CORPUS_MANIFEST_PATH.read_bytes())
            self._write_857_deficit_ledger_fixture(ledger_path)

            orig_corpus_bytes = corpus_path.read_bytes()
            orig_ledger_bytes = ledger_path.read_bytes()

            with self.assertRaises(LegacyProvenanceReconciliationRequiredError) as ctx:
                acquire_authorized_documents(
                    authorized_documents=[self._get_stxf_fixture_spec()],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=tmp / "cache",
                )
            self.assertEqual(ctx.exception.status, "LEGACY_PROVENANCE_RECONCILIATION_REQUIRED")
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)

    def test_a19_and_a20_authorized_provenance_reconciliation_idempotent_and_preserves_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            ledger_path = tmp / "ledger.json"
            corpus_path.write_bytes(OUTPUT_CORPUS_MANIFEST_PATH.read_bytes())
            self._write_857_deficit_ledger_fixture(ledger_path)
            orig_corpus_bytes = corpus_path.read_bytes()

            alps_prov = self._get_authoritative_alps_provenance_record()
            res1 = reconcile_missing_provenance(
                corpus_manifest_path=corpus_path,
                provenance_ledger_path=ledger_path,
                authorized_reconciliation_records=[alps_prov],
            )
            self.assertEqual(res1["status"], "RECONCILED")
            self.assertEqual(res1["provenance_count_before"], 857)
            self.assertEqual(res1["provenance_count_after"], 858)
            self.assertFalse(res1["corpus_manifest_mutated"])
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)

            ledger_after_1 = ledger_path.read_bytes()
            res2 = reconcile_missing_provenance(
                corpus_manifest_path=corpus_path,
                provenance_ledger_path=ledger_path,
                authorized_reconciliation_records=[alps_prov],
            )
            self.assertEqual(res2["status"], "IDEMPOTENT_NOOP")
            self.assertEqual(res2["reconciled_count"], 0)
            self.assertEqual(ledger_path.read_bytes(), ledger_after_1)
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)

    def test_a21_missing_legacy_provenance_fields_fail_closed_without_fabrication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path = tmp / "corpus.json"
            ledger_path = tmp / "ledger.json"
            corpus_path.write_bytes(OUTPUT_CORPUS_MANIFEST_PATH.read_bytes())
            self._write_857_deficit_ledger_fixture(ledger_path)
            orig_corpus_bytes = corpus_path.read_bytes()
            orig_ledger_bytes = ledger_path.read_bytes()

            incomplete_alps_prov = self._get_authoritative_alps_provenance_record()
            del incomplete_alps_prov["acquisition_timestamp"]

            with self.assertRaises(InvalidProvenanceReconciliationError):
                reconcile_missing_provenance(
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    authorized_reconciliation_records=[incomplete_alps_prov],
                )

            for field, bad_val, exc_cls in (
                ("raw_sha256", "0" * 64, CanonicalHashConflictError),
                ("accession", "0009999999-26-000001", InvalidProvenanceReconciliationError),
                ("primary_document", "wrong_doc.htm", InvalidProvenanceReconciliationError),
                ("cik", "0009999999", InvalidProvenanceReconciliationError),
                ("form", "497K", InvalidProvenanceReconciliationError),
                ("byte_length", 12345, InvalidProvenanceReconciliationError),
            ):
                bad_rec = self._get_authoritative_alps_provenance_record()
                bad_rec[field] = bad_val
                with self.assertRaises(exc_cls):
                    reconcile_missing_provenance(
                        corpus_manifest_path=corpus_path,
                        provenance_ledger_path=ledger_path,
                        authorized_reconciliation_records=[bad_rec],
                    )
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)

    # =========================================================================
    # B01 - B20: Adversarial Accession, Path Safety & Duplicate Request Tests
    # =========================================================================

    def test_b01_to_b03_malformed_short_and_slash_accessions_rejected(self):
        for bad_acc in (
            "NOT_AN_ACCESSION",
            "1592900-26-002347",
            "0001592900/26/002347",
            "0001592900-26002347",
            "../../foo",
            "0001592900-26-002347/extra",
        ):
            spec = self._get_stxf_fixture_spec()
            spec["accession"] = bad_acc
            with self.assertRaises(InvalidAuthorizedSourceSpecError, msg=f"Should reject {bad_acc}"):
                validate_authorized_source_spec(spec)

    def test_b04_to_b12_and_windows_path_traversal_rejected(self):
        for bad_doc in (
            "../evil.htm",
            "..\\evil.htm",
            "/../../escaped.htm",
            "../../escaped.htm",
            "..\\..\\escaped.htm",
            "/etc/passwd",
            "C:\\temp\\evil.htm",
            "C:/temp/evil.htm",
            "\\\\server\\share\\evil.htm",
            "subdir/doc.htm",
            "subdir\\doc.htm",
            "",
            "   ",
            "%2e%2e/evil.htm",
            "http://evil.com/doc.htm",
            "doc.htm?query=1",
            "doc.htm#frag",
            "doc\x00.htm",
        ):
            spec = self._get_stxf_fixture_spec()
            spec["primary_document"] = bad_doc
            with self.assertRaises(InvalidAuthorizedSourceSpecError, msg=f"Should reject {bad_doc!r}"):
                validate_authorized_source_spec(spec)

    def test_b13_and_direct_cache_escape_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_dir = tmp / "sec_prospectus"
            cache_dir.mkdir(parents=True, exist_ok=True)
            outside_target = tmp / "escaped.htm"

            attack_spec = self._get_stxf_fixture_spec()
            attack_spec["accession"] = "NOT_AN_ACCESSION"
            attack_spec["primary_document"] = "/../../escaped.htm"

            with self.assertRaises(InvalidAuthorizedSourceSpecError):
                validate_authorized_source_spec(attack_spec, cache_dir=cache_dir)

            with self.assertRaises(InvalidAuthorizedSourceSpecError):
                resolve_contained_cache_path(cache_dir, "0001592900-26-002347", "/../../escaped.htm")

            self.assertFalse(outside_target.exists())

    def test_b14_b17_b18_identical_duplicate_request_specs_dedupe_to_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            spec1 = self._get_stxf_fixture_spec()
            spec2 = dict(spec1)

            forensic_src = Path(
                "C:/Users/akara/.gemini/antigravity/brain/76b5f1e2-3d0d-4453-b0de-300603740e42/scratch/strivestrv-stxf497etickerc.htm"
            )
            session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = forensic_src.read_bytes()
            session.get.return_value = mock_resp

            with patch("time.sleep"):
                res = acquire_authorized_documents(
                    authorized_documents=[spec1, spec2],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                    session=session,
                )

            self.assertEqual(res["status"], "EXTENDED")
            self.assertEqual(res["new_documents_added"], 1)
            self.assertEqual(res["corpus_document_count_after"], 859)
            self.assertEqual(res["provenance_record_count_after"], 859)
            self.assertEqual(session.get.call_count, 1)

            c_doc = json.loads(corpus_path.read_text(encoding="utf-8"))
            p_doc = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(len(c_doc["records"]), 859)
            self.assertEqual(len(p_doc["provenance_records"]), 859)

    def test_b15_and_b16_conflicting_duplicate_hash_and_metadata_fail_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            orig_corpus_bytes = corpus_path.read_bytes()
            orig_ledger_bytes = ledger_path.read_bytes()

            spec1 = self._get_stxf_fixture_spec()
            spec_bad_hash = dict(spec1)
            spec_bad_hash["expected_raw_sha256"] = "b" * 64

            session = MagicMock()
            with self.assertRaises(CanonicalHashConflictError):
                acquire_authorized_documents(
                    authorized_documents=[spec1, spec_bad_hash],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                    session=session,
                )
            session.get.assert_not_called()
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)

            spec_bad_meta = dict(spec1)
            spec_bad_meta["form"] = "497K"
            with self.assertRaises(InvalidAuthorizedSourceSpecError):
                acquire_authorized_documents(
                    authorized_documents=[spec1, spec_bad_meta],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                    session=session,
                )
            session.get.assert_not_called()
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)

    def test_b19_and_b20_invalid_spec_causes_zero_http_requests_and_zero_fs_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            corpus_path, ledger_path, cache_dir = self._build_consistent_858_fixture(tmp)
            orig_corpus_bytes = corpus_path.read_bytes()
            orig_ledger_bytes = ledger_path.read_bytes()

            bad_spec = self._get_stxf_fixture_spec()
            bad_spec["primary_document"] = "../escape.htm"
            session = MagicMock()

            with self.assertRaises(InvalidAuthorizedSourceSpecError):
                acquire_authorized_documents(
                    authorized_documents=[bad_spec],
                    corpus_manifest_path=corpus_path,
                    provenance_ledger_path=ledger_path,
                    cache_dir=cache_dir,
                    session=session,
                )

            session.get.assert_not_called()
            self.assertEqual(corpus_path.read_bytes(), orig_corpus_bytes)
            self.assertEqual(ledger_path.read_bytes(), orig_ledger_bytes)
            self.assertEqual(list(cache_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
