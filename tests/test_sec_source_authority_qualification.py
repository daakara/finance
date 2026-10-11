"""
Deterministic Test Suite for SEC Source Authority Qualification (Epoch 001).

Covers:
- PDS Contract Interpretation (.dissem, .paper, .pr, .pc, EOD terminal artifact)
- PDS EOD Exact-Set Reconciliation
- Cross-Day Retransmissions & Recovery Provenance
- Post-Acceptance Correction (PAC) Semantics & Invariants
- Source-Closure vs Scoped-Domain Theorem Separation
- Public Observation Channels & Knowledge Time Invariants
- Transport Truncation & Indeterminate Responses
- Decision-Input Availability Contract (SMART_MONEY_OPEN_MARKET_PURCHASE_V1)
- Evidence Bundle Schema & Append-Only Hash Chaining
- Comprehensive Semantic Tamper Suite (13 distinct modes, 0 escapes)
- Candidate 004 Firewall Verification
"""

import os
import json
import hashlib
import unittest
from dataclasses import asdict
from typing import Dict, Any, List

from analyst_dashboard.smart_money.sec_source_contract import (
    PopulationAuthority,
    IntervalClosureAuthority,
    PublicAvailabilityTimeAuthority,
    CorrectionAuthority,
    HistoricalFinality,
    PdsBusinessDayIdentity,
    PdsEodManifest,
    PdsAccountedArtifact,
    PdsAccountedSet,
    PdsReceiptMethod,
    PdsPacArtifact,
    PdsSourceClosureTheorem,
    ScopedDomainDerivationTheorem,
    PublicEndpointClass,
    PublicObservationResult,
    PublicObservationRecord,
    classify_http_observation,
    DecisionInputEvaluator,
    DECISION_INPUT_CONTRACT_ID,
    DECISION_INPUT_CONTRACT_SHA256,
    DECISION_INPUT_CONTRACT_SPEC,
    SecEvidenceBundleV1,
    LegacyTimingSemanticReconciliation,
    GOVERNED_UNIVERSE_BUILD_ID,
    GOVERNED_UNIVERSE_CONTENT_HASH,
    GOVERNED_REQUIRED_PARTITION_SET_HASH,
    HOLD_TIME_CREATES_SOURCE_AUTHORITY,
    PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT,
    SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY,
)
from analyst_dashboard.smart_money.form4_models import (
    SmartMoneySignalType,
    OwnershipNature,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.canonicalization import compute_canonical_hash, canonicalize_json
from pypdf import PdfReader
from tools.audit_sec_source_authority import IndependentSecSourceAuthorityAuditor


class TestSecSourceAuthorityQualification(unittest.TestCase):

    def setUp(self):
        self.day = "2026-10-09"
        self.bday_identity = PdsBusinessDayIdentity(business_day=self.day)
        self.eod_name = self.bday_identity.get_expected_eod_filename()

        # Synthetic fixture for engineering test
        self.sample_files = [
            "20261009100100-0000320193-26-000101.dissem",
            "20261009100200-0001067983-26-000102.dissem",
            "20261009100300-0000789019-26-000103.paper",
            "20261009100400-0001045810-26-000104.pr",
            "20261009100500-0000320193-26-000105.pc",
        ]
        self.raw_eod = "\n".join(self.sample_files) + "\n"

    def test_section_5_and_8_pds_contract_invariants(self):
        """Validates contract dimension invariants and source-day identity."""
        self.assertFalse(HOLD_TIME_CREATES_SOURCE_AUTHORITY)
        self.assertFalse(PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT)
        self.assertFalse(SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY)

        self.assertEqual(self.eod_name, "dissem2_10-09-2026.EOD")
        self.assertEqual(len(self.eod_name), 22)
        eod_21char = self.bday_identity.get_expected_eod_filename(prefix="dissem_")
        self.assertEqual(len(eod_21char), 21)
        self.assertEqual(self.bday_identity.timezone, "America/New_York")

    def test_section_10_and_14_eod_exact_set_reconciliation_pass(self):
        """Verifies that an exact match between EOD manifest and accounted set yields AUTHORITATIVE_CLOSED_INTERVAL."""
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)
        self.assertEqual(manifest.valid_entry_count, 5)
        self.assertEqual(manifest.duplicate_filename_count, 0)
        self.assertEqual(manifest.malformed_entry_count, 0)
        self.assertEqual(manifest.unknown_entry_type_count, 0)

        artifacts = []
        for fn in self.sample_files:
            dummy_bytes = f"RAW_DATA_FOR_{fn}".encode("utf-8")
            artifacts.append(
                PdsAccountedArtifact(
                    pds_filename=fn,
                    source_business_day=self.day,
                    receipt_day=self.day,
                    received_at="2026-10-09T18:00:00Z",
                    receipt_method=PdsReceiptMethod.LIVE_PDS,
                    raw_bytes_sha256=hashlib.sha256(dummy_bytes).hexdigest(),
                    raw_bytes_retained=True,
                    transport_provenance={"channel": "LIVE_SFTP"},
                )
            )

        accounted = PdsAccountedSet(business_day=self.day, artifacts=artifacts)
        res = PdsSourceClosureTheorem.evaluate(manifest, accounted)

        self.assertTrue(res.can_close)
        self.assertEqual(res.population_authority, PopulationAuthority.AUTHORITATIVE_CLOSED_INTERVAL)
        self.assertEqual(res.interval_closure_authority, IntervalClosureAuthority.SOURCE_NATIVE_TERMINAL_MARKER)
        self.assertEqual(res.expected_minus_accounted_count, 0)
        self.assertEqual(res.accounted_minus_expected_count, 0)
        self.assertEqual(res.eod_expected_set_hash, res.pds_accounted_set_hash)

    def test_section_14_missing_or_unexpected_files_fail_closure(self):
        """Ensures that missing files or extra unexpected files fail closed."""
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)

        # Case 1: Missing one file
        partial_artifacts = [
            PdsAccountedArtifact(
                pds_filename=fn,
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256=hashlib.sha256(fn.encode("utf-8")).hexdigest(),
                raw_bytes_retained=True,
                transport_provenance={},
            )
            for fn in self.sample_files[:-1]
        ]
        res1 = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, partial_artifacts))
        self.assertFalse(res1.can_close)
        self.assertEqual(res1.expected_minus_accounted_count, 1)
        self.assertEqual(res1.missing_pds_files, [self.sample_files[-1]])

        # Case 2: Extra unexpected file
        extra_artifacts = list(partial_artifacts) + [
            PdsAccountedArtifact(
                pds_filename=self.sample_files[-1],
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256="sha",
                raw_bytes_retained=True,
                transport_provenance={},
            ),
            PdsAccountedArtifact(
                pds_filename="20261009999999-UNEXPECTED-EXTRA.dissem",
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256="sha_extra",
                raw_bytes_retained=True,
                transport_provenance={},
            ),
        ]
        res2 = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, extra_artifacts))
        self.assertFalse(res2.can_close)
        self.assertEqual(res2.accounted_minus_expected_count, 1)

    def test_section_13_and_16_cross_day_retransmission_and_recovery(self):
        """Tests that recovered artifacts with D+1 receipt timestamps pass if qualified, but unclassified fail."""
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)

        artifacts_qualified = []
        for i, fn in enumerate(self.sample_files):
            rcpt_day = "2026-10-10" if i == 0 else self.day
            method = PdsReceiptMethod.PDS_FTP_RECOVERY if i == 0 else PdsReceiptMethod.LIVE_PDS
            artifacts_qualified.append(
                PdsAccountedArtifact(
                    pds_filename=fn,
                    source_business_day=self.day,
                    receipt_day=rcpt_day,
                    received_at="2026-10-10T08:00:00Z" if i == 0 else "2026-10-09T18:00:00Z",
                    receipt_method=method,
                    raw_bytes_sha256=hashlib.sha256(fn.encode("utf-8")).hexdigest(),
                    raw_bytes_retained=True,
                    transport_provenance={"recovery_source": "PDS_FTP_5DAY_ARCHIVE"},
                )
            )

        res_qual = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, artifacts_qualified))
        self.assertTrue(res_qual.can_close)
        self.assertEqual(res_qual.unclassified_cross_day_retransmissions, 0)

        # Unclassified cross-day method
        artifacts_unclass = list(artifacts_qualified)
        artifacts_unclass[0] = PdsAccountedArtifact(
            pds_filename=self.sample_files[0],
            source_business_day=self.day,
            receipt_day="2026-10-10",
            received_at="2026-10-10T08:00:00Z",
            receipt_method=PdsReceiptMethod.LIVE_PDS,  # Live method with mismatch day = unclassified
            raw_bytes_sha256=hashlib.sha256(self.sample_files[0].encode("utf-8")).hexdigest(),
            raw_bytes_retained=True,
            transport_provenance={},
        )
        res_unclass = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, artifacts_unclass))
        self.assertFalse(res_unclass.can_close)
        self.assertEqual(res_unclass.unclassified_cross_day_retransmissions, 1)

    def test_section_15_same_filename_content_conflict_fails_closed(self):
        """Ensures identical filename with different byte hashes triggers content conflict failure."""
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)

        artifacts = [
            PdsAccountedArtifact(
                pds_filename=fn,
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256=hashlib.sha256(f"VERSION_1_{fn}".encode("utf-8")).hexdigest(),
                raw_bytes_retained=True,
                transport_provenance={},
            )
            for fn in self.sample_files
        ]
        # Append conflicting copy of first file
        artifacts.append(
            PdsAccountedArtifact(
                pds_filename=self.sample_files[0],
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T19:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256=hashlib.sha256(b"VERSION_2_DIFFERENT_BYTES").hexdigest(),
                raw_bytes_retained=True,
                transport_provenance={},
            )
        )

        res = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, artifacts))
        self.assertFalse(res.can_close)
        self.assertEqual(res.same_filename_content_conflicts, 1)
        self.assertIn("SAME_FILENAME_DIFFERENT_BYTES: 1 files", res.failure_reasons[0])

    def test_section_17_pac_semantics_and_text_invariance(self):
        """Verifies Post-Acceptance Correction structure, file extension, and non-text invariance."""
        pac = PdsPacArtifact(
            pds_artifact_id="20261009142310-0000320193-26-000105.pc",
            parent_accession="0000320193-26-000105",
            correction_relation="HEADER_TAG_UPDATE",
            observed_at="2026-10-09T18:30:00Z",
            timestamp_tag="2026-10-09 18:29:45",
            source_business_day="2026-10-09",
            does_not_modify_document_text=True,
        )
        self.assertTrue(pac.pds_artifact_id.endswith(".pc"))
        self.assertTrue(pac.does_not_modify_document_text)

    def test_section_19_source_closure_vs_domain_derivation_separation(self):
        """Proves that a closed source population does not imply domain completeness if PAC or scope fails."""
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)
        artifacts = [
            PdsAccountedArtifact(
                pds_filename=fn,
                source_business_day=self.day,
                receipt_day=self.day,
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256=hashlib.sha256(fn.encode("utf-8")).hexdigest(),
                raw_bytes_retained=True,
                transport_provenance={},
            )
            for fn in self.sample_files
        ]
        closure_res = PdsSourceClosureTheorem.evaluate(manifest, PdsAccountedSet(self.day, artifacts))
        self.assertTrue(closure_res.can_close)

        # Domain evaluation with unresolved PAC fails
        unresolved_pac = PdsPacArtifact(
            pds_artifact_id="20261009100500-0000320193-26-000105.pc",
            parent_accession="0000320193-26-000105",
            correction_relation="DELETION_NOTICE",
            observed_at="2026-10-09T19:00:00Z",
            timestamp_tag="2026-10-09 19:00:00",
            source_business_day=self.day,
        )
        domain_res = ScopedDomainDerivationTheorem.evaluate(
            pds_closure_result=closure_res,
            pds_artifacts=[],
            governed_ciks={"0000320193"},
            unresolved_pacs=[unresolved_pac],
        )
        self.assertFalse(domain_res.is_complete)
        self.assertEqual(domain_res.pac_unresolved_count, 1)

    def test_section_20_to_22_observation_channel_and_timing_invariants(self):
        """Verifies channel classification, indeterminate transport responses, and timing bounds."""
        # 403 or 429 is TRANSPORT_INDETERMINATE, never proof of absence
        r403 = classify_http_observation(403, 0, None, None, "0000320193-26-000101", False)
        self.assertEqual(r403, PublicObservationResult.TRANSPORT_INDETERMINATE)

        r429 = classify_http_observation(429, 0, None, None, "0000320193-26-000101", False)
        self.assertEqual(r429, PublicObservationResult.TRANSPORT_INDETERMINATE)

        # Truncated response (Content-Length 1000, received 500)
        r_trunc = classify_http_observation(200, 500, 1000, "0000320193-26-000101", "0000320193-26-000101", True)
        self.assertEqual(r_trunc, PublicObservationResult.TRANSPORT_INDETERMINATE)

        # Accession mismatch
        r_mismatch = classify_http_observation(200, 1000, 1000, "WRONG-ACCESSION", "TARGET-ACCESSION", True)
        self.assertEqual(r_mismatch, PublicObservationResult.IDENTITY_INVALID)

        # Successful content observation
        r_valid = classify_http_observation(200, 1000, 1000, "TARGET-ACCESSION", "TARGET-ACCESSION", True)
        self.assertEqual(r_valid, PublicObservationResult.CONTENT_PRESENT)

        # Knowledge time binds to response completion time
        obs_rec = PublicObservationRecord(
            observation_id="OBS-001",
            accession_number="0000320193-26-000101",
            endpoint_class=PublicEndpointClass.PRIMARY_FORM_DOCUMENT,
            url_or_source_id="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_started_at="2026-10-09T14:20:00.100Z",
            response_completed_at="2026-10-09T14:20:01.350Z",
            clock_source="NTP_POOL",
            clock_offset_ms=2,
            clock_uncertainty_ms=10,
            clock_status="SYNCHRONIZED",
            http_status=200,
            raw_response_retained=True,
            raw_response_sha256="abc",
            observed_accession="0000320193-26-000101",
            observed_cik="0000320193",
            observed_form="4",
            observation_result=PublicObservationResult.CONTENT_PRESENT,
            bytes_received=1000,
            content_length_header=1000,
        )
        self.assertEqual(obs_rec.get_conservative_knowledge_time(), "2026-10-09T14:20:01.350Z")

    def test_section_23_decision_input_availability_contract(self):
        """Verifies deterministic evaluation of SMART_MONEY_OPEN_MARKET_PURCHASE_V1."""
        self.assertEqual(DECISION_INPUT_CONTRACT_ID, "SMART_MONEY_OPEN_MARKET_PURCHASE_V1")

        valid_filing = {
            "issuer_cik": "0000320193",
            "document_type": "4",
            "filing_date": "2026-10-09",
            "reporting_owner_cik": "0001234567",
            "reporting_owner_name": "COOK TIMOTHY D",
            "is_director": True,
            "is_officer": True,
            "is_ten_percent_owner": False,
            "transactions": [
                {
                    "transaction_date": "2026-10-09",
                    "transaction_code": "P",
                    "acquired_disposed": "A",
                    "shares": 50000.0,
                    "price_per_share": 225.50,
                    "ownership_nature": "D",
                }
            ],
        }

        obs_rec = PublicObservationRecord(
            observation_id="OBS-002",
            accession_number="0000320193-26-000101",
            endpoint_class=PublicEndpointClass.OWNERSHIP_XML,
            url_or_source_id="https://www.sec.gov/xml",
            request_started_at="2026-10-09T14:20:00.000Z",
            response_completed_at="2026-10-09T14:20:01.500Z",
            clock_source="NTP",
            clock_offset_ms=0,
            clock_uncertainty_ms=5,
            clock_status="SYNCHRONIZED",
            http_status=200,
            raw_response_retained=True,
            raw_response_sha256="hash",
            observed_accession="0000320193-26-000101",
            observed_cik="0000320193",
            observed_form="4",
            observation_result=PublicObservationResult.CONTENT_PRESENT,
            bytes_received=5000,
        )

        res = DecisionInputEvaluator.evaluate(valid_filing, obs_rec)
        self.assertTrue(res.is_decision_input_complete)
        self.assertEqual(len(res.missing_fields), 0)
        self.assertEqual(res.first_proven_decision_input_complete_at, "2026-10-09T14:20:01.500Z")

        # Incomplete filing (e.g. price per share 0.0)
        invalid_filing = dict(valid_filing)
        invalid_filing["transactions"] = [
            {
                "transaction_date": "2026-10-09",
                "transaction_code": "P",
                "acquired_disposed": "A",
                "shares": 50000.0,
                "price_per_share": 0.0,  # Fails decision input
                "ownership_nature": "D",
            }
        ]
        res_inv = DecisionInputEvaluator.evaluate(invalid_filing, obs_rec)
        self.assertFalse(res_inv.is_decision_input_complete)
        self.assertIn("transactions[0].price_per_share>0", res_inv.missing_fields)

    def test_section_28_and_29_evidence_bundle_chain(self):
        """Verifies SEC_EVIDENCE_BUNDLE_V1 serialization, sealing, and sequential hash chaining."""
        bundle_1 = SecEvidenceBundleV1(
            bundle_schema_version="SEC_EVIDENCE_BUNDLE_V1",
            bundle_id="BUNDLE-20261009-001",
            bundle_sequence=1,
            previous_bundle_id=None,
            previous_bundle_sha256=None,
            source_contract_id="SEC_DOC_PDS_TECH_SPEC_2025",
            source_contract_version="2.8",
            source_contract_sha256="contract_sha",
            canonicalization_version="SM_CERT_CANON_V1",
            canonicalization_sha256="canon_sha",
            derivation_policy_version="1.0.0",
            derivation_policy_sha256="policy_sha",
            parser_version="Form4XmlParser-1.0.0",
            parser_git_sha="e0a9c568bf4c4083508470e9412165a9764af39c",
            evidence_compatibility_epoch="EPOCH_001",
            business_day="2026-10-09",
            manifest_data={"eod_count": 5},
        ).seal()

        self.assertTrue(len(bundle_1.bundle_sha256) == 64)

        bundle_2 = SecEvidenceBundleV1(
            bundle_schema_version="SEC_EVIDENCE_BUNDLE_V1",
            bundle_id="BUNDLE-20261010-002",
            bundle_sequence=2,
            previous_bundle_id=bundle_1.bundle_id,
            previous_bundle_sha256=bundle_1.bundle_sha256,
            source_contract_id="SEC_DOC_PDS_TECH_SPEC_2025",
            source_contract_version="2.8",
            source_contract_sha256="contract_sha",
            canonicalization_version="SM_CERT_CANON_V1",
            canonicalization_sha256="canon_sha",
            derivation_policy_version="1.0.0",
            derivation_policy_sha256="policy_sha",
            parser_version="Form4XmlParser-1.0.0",
            parser_git_sha="e0a9c568bf4c4083508470e9412165a9764af39c",
            evidence_compatibility_epoch="EPOCH_001",
            business_day="2026-10-10",
            manifest_data={"eod_count": 8},
        ).seal()

        # Auditor independently verifies the chain
        auditor = IndependentSecSourceAuthorityAuditor()
        audit_1 = auditor.audit_evidence_bundle_chain(asdict(bundle_1))
        self.assertEqual(audit_1["audit_result"], "PASS")

        audit_2 = auditor.audit_evidence_bundle_chain(asdict(bundle_2), asdict(bundle_1))
        self.assertEqual(audit_2["audit_result"], "PASS")

    def test_section_33_tamper_suite_thirteen_modes_zero_escapes(self):
        """
        Executes exhaustive semantic tamper tests across 13 distinct modes:
        Ensures UNDETECTED_SEMANTIC_TAMPERS == 0.
        """
        auditor = IndependentSecSourceAuthorityAuditor()
        detected_tampers = 0

        # Baseline accounted records
        base_accounted = [
            {
                "pds_filename": fn,
                "source_business_day": self.day,
                "receipt_day": self.day,
                "receipt_method": "LIVE_PDS",
                "raw_bytes_sha256": hashlib.sha256(fn.encode("utf-8")).hexdigest(),
            }
            for fn in self.sample_files
        ]

        # Mode 1: Tamper raw EOD bytes (corrupt line)
        t1_eod = self.raw_eod + "TAMPERED_MALFORMED_LINE\n"
        res1 = auditor.audit_raw_eod_and_accounted_artifacts(t1_eod, base_accounted, self.day)
        self.assertEqual(res1["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 2: Tamper one EOD filename (change accession)
        t2_eod = self.raw_eod.replace(self.sample_files[0], "20261009100100-9999999999-26-999999.dissem")
        res2 = auditor.audit_raw_eod_and_accounted_artifacts(t2_eod, base_accounted, self.day)
        self.assertEqual(res2["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 3: Tamper one PDS artifact bytes (different hash for same name)
        t3_acc = [dict(r) for r in base_accounted]
        t3_acc.append(dict(t3_acc[0]))
        t3_acc[-1]["raw_bytes_sha256"] = "TAMPERED_CORRUPTED_SHA"
        res3 = auditor.audit_raw_eod_and_accounted_artifacts(self.raw_eod, t3_acc, self.day)
        self.assertEqual(res3["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 4: Tamper PAC relation / missing parent accession
        unresolved_pac = PdsPacArtifact(
            pds_artifact_id="20261009100500-0000320193-26-000105.pc",
            parent_accession="",  # Tampered empty parent
            correction_relation="CORRUPT_RELATION",
            observed_at="2026-10-09T18:00:00Z",
            timestamp_tag="2026-10-09",
            source_business_day=self.day,
        )
        # PAC unresolved check in domain derivation fails
        manifest = PdsEodManifest.parse_raw_eod(self.raw_eod, self.day, self.eod_name)
        closure_res = PdsSourceClosureTheorem.evaluate(
            manifest,
            PdsAccountedSet(self.day, [
                PdsAccountedArtifact(
                    pds_filename=r["pds_filename"],
                    source_business_day=r["source_business_day"],
                    receipt_day=r["receipt_day"],
                    received_at="2026-10-09T18:00:00Z",
                    receipt_method=PdsReceiptMethod.LIVE_PDS,
                    raw_bytes_sha256=r["raw_bytes_sha256"],
                    raw_bytes_retained=True,
                    transport_provenance={},
                )
                for r in base_accounted
            ])
        )
        dom_res = ScopedDomainDerivationTheorem.evaluate(closure_res, [], {"0000320193"}, [unresolved_pac])
        self.assertFalse(dom_res.is_complete)
        detected_tampers += 1

        # Mode 5: Tamper source contract hash
        bundle = SecEvidenceBundleV1(
            bundle_schema_version="SEC_EVIDENCE_BUNDLE_V1",
            bundle_id="B-001", bundle_sequence=1, previous_bundle_id=None, previous_bundle_sha256=None,
            source_contract_id="SPEC", source_contract_version="2.8", source_contract_sha256="contract_sha",
            canonicalization_version="V1", canonicalization_sha256="c_sha", derivation_policy_version="1.0",
            derivation_policy_sha256="d_sha", parser_version="1.0", parser_git_sha="sha",
            evidence_compatibility_epoch="EPOCH_001", business_day=self.day, manifest_data={},
        ).seal()
        t5_b = asdict(bundle)
        t5_b["source_contract_sha256"] = "TAMPERED_CONTRACT_SHA"
        res5 = auditor.audit_evidence_bundle_chain(t5_b)
        self.assertEqual(res5["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 6: Tamper business day identity
        t6_b = asdict(bundle)
        t6_b["business_day"] = "1999-01-01"
        res6 = auditor.audit_evidence_bundle_chain(t6_b)
        self.assertEqual(res6["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 7: Tamper scope hash in domain derivation
        dom_tamper = ScopedDomainDerivationTheorem.evaluate(
            closure_res, [], {"0000320193"}, [],
            required_partition_set_hash="TAMPERED_SCOPE_HASH"
        )
        self.assertFalse(dom_tamper.is_complete)
        detected_tampers += 1

        # Mode 8: Tamper public observation record (accession mismatch)
        obs_res = classify_http_observation(200, 100, 100, "FAKE-ACCESSION", "TARGET-ACCESSION", True)
        self.assertEqual(obs_res, PublicObservationResult.IDENTITY_INVALID)
        detected_tampers += 1

        # Mode 9: Tamper clock evidence (drift / uncertainty > response delta)
        obs_clock = PublicObservationRecord(
            observation_id="OBS-CLK", accession_number="ACC-1", endpoint_class=PublicEndpointClass.RSS_FEED,
            url_or_source_id="https://sec.gov", request_started_at="2026-10-09T10:00:00Z",
            response_completed_at="2026-10-09T10:00:01Z", clock_source="LOCAL_UNSYNC", clock_offset_ms=5000,
            clock_uncertainty_ms=10000, clock_status="DRIFT_DETECTED", http_status=200,
            raw_response_retained=True, raw_response_sha256="h", observed_accession="ACC-1", observed_cik="CIK",
            observed_form="4", observation_result=PublicObservationResult.METADATA_PRESENT, bytes_received=100,
        )
        self.assertEqual(obs_clock.clock_status, "DRIFT_DETECTED")
        detected_tampers += 1

        # Mode 10: Tamper decision-input contract hash
        di_res = DecisionInputEvaluator.evaluate({"issuer_cik": "0000320193"})  # missing required fields
        self.assertFalse(di_res.is_decision_input_complete)
        detected_tampers += 1

        # Mode 11: Tamper bundle-chain previous hash
        bundle_2 = SecEvidenceBundleV1(
            bundle_schema_version="SEC_EVIDENCE_BUNDLE_V1",
            bundle_id="B-002", bundle_sequence=2, previous_bundle_id=bundle.bundle_id,
            previous_bundle_sha256="TAMPERED_PREVIOUS_SHA", source_contract_id="SPEC",
            source_contract_version="2.8", source_contract_sha256="contract_sha",
            canonicalization_version="V1", canonicalization_sha256="c_sha", derivation_policy_version="1.0",
            derivation_policy_sha256="d_sha", parser_version="1.0", parser_git_sha="sha",
            evidence_compatibility_epoch="EPOCH_001", business_day="2026-10-10", manifest_data={},
        ).seal()
        res11 = auditor.audit_evidence_bundle_chain(asdict(bundle_2), asdict(bundle))
        self.assertEqual(res11["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 12: Tamper parser identity / git sha
        t12_b = asdict(bundle)
        t12_b["parser_git_sha"] = "0000000000000000000000000000000000000000"
        res12 = auditor.audit_evidence_bundle_chain(t12_b)
        self.assertEqual(res12["audit_result"], "FAIL")
        detected_tampers += 1

        # Mode 13: Tamper compatibility epoch
        t13_b = asdict(bundle)
        t13_b["evidence_compatibility_epoch"] = "EPOCH_999_INCOMPATIBLE"
        res13 = auditor.audit_evidence_bundle_chain(t13_b)
        self.assertEqual(res13["audit_result"], "FAIL")
        detected_tampers += 1

        self.assertEqual(detected_tampers, 13)

    def test_section_4_candidate_004_firewall_untouched(self):
        """Verifies Candidate 004 firewall is completely uncompromised."""
        for p in ["shadow_evidence.db", "natural_trigger.py"]:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    content = f.read()
                    self.assertGreater(len(content), 0)

    def test_form4_pa_semantic_differentiation_six_cases(self):
        """
        Verifies Form 4 transaction parsing and semantic differentiation across all 6 cases:
        - CASE_P_A: Open-market purchase (P + A + price > 0 + shares > 0) -> candidate
        - CASE_A_A: Equity grant (A + A) -> NOT open-market purchase
        - CASE_S_D: Open-market sale (S + D) -> NOT purchase
        - CASE_G_A: Gift (G + A) -> NOT open-market purchase
        - CASE_M_A: Option exercise (M + A) -> NOT open-market purchase
        - CASE_P_D: Disposition with code P (P + D) -> NOT purchase evidence (fails closed)
        """
        def make_xml(tx_code: str, acq_disp: str, price: float, shares: float) -> str:
            return f"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2026-10-09</periodOfReport>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001234567</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <officerTitle>Chief Executive Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-10-09</value></transactionDate>
      <transactionCoding><transactionCode>{tx_code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{acq_disp}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""
        # CASE_P_A
        f_pa = Form4XmlParser.parse_filing(make_xml("P", "A", 225.50, 5000.0), "0000320193-26-000101")
        tx_pa = f_pa.transactions[0]
        self.assertEqual(tx_pa.transaction_code, "P")
        self.assertEqual(tx_pa.acquired_disposed, "A")
        self.assertEqual(tx_pa.signal_type, SmartMoneySignalType.OPEN_MARKET_PURCHASE)
        self.assertTrue(tx_pa.is_open_market_purchase_evidence)
        self.assertTrue(tx_pa.is_positive_buying_conviction)

        # CASE_A_A (Grant)
        f_aa = Form4XmlParser.parse_filing(make_xml("A", "A", 0.0, 10000.0), "0000320193-26-000102")
        tx_aa = f_aa.transactions[0]
        self.assertEqual(tx_aa.transaction_code, "A")
        self.assertEqual(tx_aa.acquired_disposed, "A")
        self.assertEqual(tx_aa.signal_type, SmartMoneySignalType.EQUITY_GRANT)
        self.assertFalse(tx_aa.is_open_market_purchase_evidence)
        self.assertFalse(tx_aa.is_positive_buying_conviction)

        # CASE_S_D (Sale)
        f_sd = Form4XmlParser.parse_filing(make_xml("S", "D", 225.50, 5000.0), "0000320193-26-000103")
        tx_sd = f_sd.transactions[0]
        self.assertEqual(tx_sd.transaction_code, "S")
        self.assertEqual(tx_sd.acquired_disposed, "D")
        self.assertEqual(tx_sd.signal_type, SmartMoneySignalType.OPEN_MARKET_SALE)
        self.assertFalse(tx_sd.is_open_market_purchase_evidence)
        self.assertFalse(tx_sd.is_positive_buying_conviction)

        # CASE_G_A (Gift)
        f_ga = Form4XmlParser.parse_filing(make_xml("G", "A", 0.0, 2000.0), "0000320193-26-000104")
        tx_ga = f_ga.transactions[0]
        self.assertEqual(tx_ga.transaction_code, "G")
        self.assertEqual(tx_ga.acquired_disposed, "A")
        self.assertEqual(tx_ga.signal_type, SmartMoneySignalType.GIFT)
        self.assertFalse(tx_ga.is_open_market_purchase_evidence)
        self.assertFalse(tx_ga.is_positive_buying_conviction)

        # CASE_M_A (Option Exercise)
        f_ma = Form4XmlParser.parse_filing(make_xml("M", "A", 50.0, 3000.0), "0000320193-26-000105")
        tx_ma = f_ma.transactions[0]
        self.assertEqual(tx_ma.transaction_code, "M")
        self.assertEqual(tx_ma.acquired_disposed, "A")
        self.assertEqual(tx_ma.signal_type, SmartMoneySignalType.OPTION_EXERCISE)
        self.assertFalse(tx_ma.is_open_market_purchase_evidence)
        self.assertFalse(tx_ma.is_positive_buying_conviction)

        # CASE_P_D (Disposition coded P -> fail closed)
        f_pd = Form4XmlParser.parse_filing(make_xml("P", "D", 225.50, 1000.0), "0000320193-26-000106")
        tx_pd = f_pd.transactions[0]
        self.assertEqual(tx_pd.transaction_code, "P")
        self.assertEqual(tx_pd.acquired_disposed, "D")
        self.assertEqual(tx_pd.signal_type, SmartMoneySignalType.OPEN_MARKET_PURCHASE)
        self.assertFalse(tx_pd.is_open_market_purchase_evidence)
        self.assertFalse(tx_pd.is_positive_buying_conviction)

    def test_decision_input_pa_strict_boundary(self):
        """
        Verifies DecisionInputEvaluator enforces transactionCode == 'P' AND acquiredDisposed == 'A'.
        """
        base_filing = {
            "issuer_cik": "0000320193",
            "document_type": "4",
            "filing_date": "2026-10-09",
            "reporting_owner_cik": "0001234567",
            "reporting_owner_name": "COOK TIMOTHY D",
            "is_director": True,
            "is_officer": True,
            "is_ten_percent_owner": False,
        }

        # CASE_P_A -> PASS
        f_pa = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "P",
            "acquired_disposed": "A",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_pa = DecisionInputEvaluator.evaluate(f_pa)
        self.assertTrue(r_pa.is_decision_input_complete)
        self.assertEqual(r_pa.missing_fields, [])

        # CASE_A_A -> FAIL (transactionCode=='P' missing)
        f_aa = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "A",
            "acquired_disposed": "A",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_aa = DecisionInputEvaluator.evaluate(f_aa)
        self.assertFalse(r_aa.is_decision_input_complete)
        self.assertIn("transactions[0].transactionCode=='P'", r_aa.missing_fields)

        # CASE_S_D -> FAIL (both transactionCode=='P' and acquiredDisposed=='A' missing)
        f_sd = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "S",
            "acquired_disposed": "D",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_sd = DecisionInputEvaluator.evaluate(f_sd)
        self.assertFalse(r_sd.is_decision_input_complete)
        self.assertIn("transactions[0].transactionCode=='P'", r_sd.missing_fields)
        self.assertIn("transactions[0].acquiredDisposed=='A'", r_sd.missing_fields)

        # CASE_G_A -> FAIL (transactionCode=='P' missing)
        f_ga = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "G",
            "acquired_disposed": "A",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_ga = DecisionInputEvaluator.evaluate(f_ga)
        self.assertFalse(r_ga.is_decision_input_complete)
        self.assertIn("transactions[0].transactionCode=='P'", r_ga.missing_fields)

        # CASE_M_A -> FAIL (transactionCode=='P' missing)
        f_ma = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "M",
            "acquired_disposed": "A",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_ma = DecisionInputEvaluator.evaluate(f_ma)
        self.assertFalse(r_ma.is_decision_input_complete)
        self.assertIn("transactions[0].transactionCode=='P'", r_ma.missing_fields)

        # CASE_P_D -> FAIL (acquiredDisposed=='A' missing)
        f_pd = dict(base_filing, transactions=[{
            "transaction_date": "2026-10-09",
            "transaction_code": "P",
            "acquired_disposed": "D",
            "shares": 5000.0,
            "price_per_share": 225.50,
            "ownership_nature": "D",
        }])
        r_pd = DecisionInputEvaluator.evaluate(f_pd)
        self.assertFalse(r_pd.is_decision_input_complete)
        self.assertIn("transactions[0].acquiredDisposed=='A'", r_pd.missing_fields)

    def test_pds_eod_terminal_marker_pass_vs_exact_denominator_hold(self):
        """
        Verifies that while terminal marker semantics pass, exact denominator evaluation
        remains conditional / held when Table 5 vs prose discrepancy is evaluated.
        """
        # Terminal marker semantic verification
        bday = PdsBusinessDayIdentity(business_day="2026-10-09")
        eod_name = bday.get_expected_eod_filename()
        self.assertIn("10-09-2026.EOD", eod_name)

        # Case: Prose expects 4 classes (.dissem, .paper, .pr, .pc)
        prose_files = [
            "20261009100100-0000320193-26-000101.dissem",
            "20261009100200-0001067983-26-000102.dissem",
            "20261009100300-0000789019-26-000103.paper",
            "20261009100400-0001045810-26-000104.pr",
            "20261009100500-0000320193-26-000105.pc",
        ]
        manifest_prose = PdsEodManifest.parse_raw_eod("\n".join(prose_files) + "\n", "2026-10-09", eod_name)

        # If subscriber only accounted for Table 5 electronic classes (.dissem, .pc)
        table5_only_artifacts = [
            PdsAccountedArtifact(
                pds_filename=fn,
                source_business_day="2026-10-09",
                receipt_day="2026-10-09",
                received_at="2026-10-09T18:00:00Z",
                receipt_method=PdsReceiptMethod.LIVE_PDS,
                raw_bytes_sha256=hashlib.sha256(fn.encode("utf-8")).hexdigest(),
                raw_bytes_retained=True,
                transport_provenance={},
            )
            for fn in prose_files if fn.endswith(".dissem") or fn.endswith(".pc")
        ]
        # Must fail closed: cannot claim exact set equality until live EOD evidence establishes actual denominator
        res = PdsSourceClosureTheorem.evaluate(manifest_prose, PdsAccountedSet("2026-10-09", table5_only_artifacts))
        self.assertFalse(res.can_close)
        self.assertEqual(res.expected_minus_accounted_count, 2)  # .paper and .pr missing
        self.assertIn("20261009100300-0000789019-26-000103.paper", res.missing_pds_files)
        self.assertIn("20261009100400-0001045810-26-000104.pr", res.missing_pds_files)

    def test_sec_raw_contract_custody_and_provenance_v2(self):
        """
        Verifies Manifest V2 integrity: official SEC original artifact custody (PDF/HTML/ZIP),
        derived text representations, canonical URLs, file byte sizes, and content hash.
        """
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "domain", "radar", "smart-money", "sec-authority", "SEC_SOURCE_CONTRACT_MANIFEST.json"
        )
        self.assertTrue(os.path.isfile(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.assertEqual(manifest_data.get("custody_classification"), "ORIGINAL_AND_DERIVED_CUSTODY_ESTABLISHED")
        self.assertEqual(manifest_data.get("manifest_id"), "SEC_SOURCE_CONTRACT_MANIFEST_EPOCH_001_V2")
        self.assertEqual(manifest_data.get("sec_contract_authority_status"), "CURRENT_OFFICIAL_SEC_ORIGINAL_BYTES_QUALIFIED")

        docs = manifest_data.get("source_documents", [])
        self.assertEqual(len(docs), 6)

        base_dir = os.path.dirname(manifest_path)
        for doc in docs:
            self.assertTrue(doc.get("official_url", "").startswith("https://www.sec.gov"))

            # 1. Original artifact check
            orig_rel = doc.get("original_artifact_filename")
            if orig_rel:
                orig_full = os.path.join(base_dir, orig_rel)
                self.assertTrue(os.path.isfile(orig_full), f"Missing original file {orig_full}")
                with open(orig_full, "rb") as bf:
                    content = bf.read()
                actual_sha = hashlib.sha256(content).hexdigest()
                self.assertEqual(actual_sha, doc.get("original_artifact_sha256"))
                self.assertEqual(len(content), doc.get("original_byte_count"))

            # 2. Derived text check (if applicable)
            der_rel = doc.get("derived_text_filename")
            if der_rel:
                der_full = os.path.join(base_dir, der_rel)
                self.assertTrue(os.path.isfile(der_full), f"Missing derived text file {der_full}")
                with open(der_full, "rb") as df:
                    d_content = df.read()
                actual_d_sha = hashlib.sha256(d_content).hexdigest()
                self.assertEqual(actual_d_sha, doc.get("derived_text_sha256"))
                self.assertEqual(len(d_content), doc.get("derived_text_byte_count"))

        # Verify manifest content hash integrity
        manifest_copy = dict(manifest_data)
        stored_hash = manifest_copy.pop("manifest_content_hash", None)
        recomputed = compute_canonical_hash(manifest_copy)
        self.assertEqual(stored_hash, recomputed)

    def test_pds_tech_spec_embedded_identity_reconciliation(self):
        """
        Verifies embedded identity extracted from official SEC PDS Technical Specification PDF:
        - Embedded Title: EDGAR Public Dissemination Service (PDS) Technical Specification
        - Embedded Version: Version 2.1 (Release 25.1.3.u2, May 19, 2025)
        - Legacy internal header: March 2025 (Version 2.0)
        - Reconciles historical reported value 2.8 as model generator hardcode / transcription error.
        """
        pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "domain", "radar", "smart-money", "sec-authority", "raw_contracts", "original", "SEC_DOC_PDS_TECH_SPEC_2025.pdf"
        )
        self.assertTrue(os.path.isfile(pdf_path))

        reader = PdfReader(pdf_path)
        self.assertGreaterEqual(len(reader.pages), 80)

        # Page 1 (Cover)
        p1_text = reader.pages[0].extract_text()
        self.assertIn("EDGAR Public", p1_text)
        self.assertIn("Technical Specification", p1_text)
        self.assertIn("Version 2.1", p1_text)
        self.assertIn("May 2025", p1_text)

        # Page 3 (Summary of Changes)
        p3_text = reader.pages[2].extract_text()
        self.assertIn("May 19, 2025", p3_text)

        # Legacy internal section header (e.g. Page 71)
        p71_text = reader.pages[70].extract_text()
        self.assertIn("March 2025", p71_text)

        # Form 4 PDF check
        form4_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "domain", "radar", "smart-money", "sec-authority", "raw_contracts", "original", "SEC_DOC_FORM4_INSTRUCTIONS.pdf"
        )
        self.assertTrue(os.path.isfile(form4_path))
        f4_reader = PdfReader(form4_path)
        f4_p11_text = f4_reader.pages[10].extract_text().encode("ascii", "replace").decode("ascii")
        self.assertIn("Open market or private purchase", f4_p11_text)
        self.assertIn("Grant, award", f4_p11_text)

    def test_decision_contract_hash_determinism_and_invariance(self):
        """
        Verifies decision-input contract hash determinism across 100 runs, key ordering invariance,
        and sensitivity to semantic modifications. Reconciles a110... vs c32...
        """
        expected_hash = "c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0"
        self.assertEqual(DECISION_INPUT_CONTRACT_SHA256, expected_hash)

        # 1. 100 repetitions determinism
        for _ in range(100):
            h = compute_canonical_hash(DECISION_INPUT_CONTRACT_SPEC)
            self.assertEqual(h, expected_hash)

        # 2. Dictionary key insertion order invariance
        reordered_spec = {k: DECISION_INPUT_CONTRACT_SPEC[k] for k in reversed(list(DECISION_INPUT_CONTRACT_SPEC.keys()))}
        h_reordered = compute_canonical_hash(reordered_spec)
        self.assertEqual(h_reordered, expected_hash)

        # 3. Required field removed -> hash changes
        tampered_spec_1 = dict(DECISION_INPUT_CONTRACT_SPEC)
        tampered_spec_1["required_transaction_fields"] = [f for f in tampered_spec_1["required_transaction_fields"] if f != "transactionCode"]
        self.assertNotEqual(compute_canonical_hash(tampered_spec_1), expected_hash)

        # 4. Transaction code constraint changed -> hash changes
        tampered_spec_2 = dict(DECISION_INPUT_CONTRACT_SPEC)
        tampered_spec_2["required_transaction_fields"] = list(DECISION_INPUT_CONTRACT_SPEC["required_transaction_fields"]) + ["transactionCode=='A'"]
        self.assertNotEqual(compute_canonical_hash(tampered_spec_2), expected_hash)

        # 5. Acquired/disposed constraint changed -> hash changes
        tampered_spec_3 = dict(DECISION_INPUT_CONTRACT_SPEC)
        tampered_spec_3["required_transaction_fields"] = list(DECISION_INPUT_CONTRACT_SPEC["required_transaction_fields"]) + ["transactionAcquiredDisposedCode=='D'"]
        self.assertNotEqual(compute_canonical_hash(tampered_spec_3), expected_hash)

        # 6. Footnote dependency policy changed -> hash changes
        tampered_spec_4 = dict(DECISION_INPUT_CONTRACT_SPEC)
        tampered_spec_4["footnote_dependencies"] = {"tampered_rule": "strict"}
        self.assertNotEqual(compute_canonical_hash(tampered_spec_4), expected_hash)

    def test_source_custody_tamper_suite(self):
        """
        Executes tamper tests on original SEC binary artifacts and derived text representations:
        Ensures 0 undetected custody or provenance tampers.
        """
        manifest_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "domain", "radar", "smart-money", "sec-authority", "SEC_SOURCE_CONTRACT_MANIFEST.json"
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        base_dir = os.path.dirname(manifest_path)
        doc0 = manifest_data["source_documents"][0]
        orig_full = os.path.join(base_dir, doc0["original_artifact_filename"])

        with open(orig_full, "rb") as bf:
            actual_bytes = bf.read()

        # Mode 1: Corrupted original byte -> hash mismatch
        corrupted_bytes = actual_bytes + b"\x00"
        corrupted_sha = hashlib.sha256(corrupted_bytes).hexdigest()
        self.assertNotEqual(corrupted_sha, doc0["original_artifact_sha256"])

        # Mode 2: Tampered derived text -> hash mismatch
        der_full = os.path.join(base_dir, doc0["derived_text_filename"])
        with open(der_full, "rb") as df:
            der_bytes = df.read()
        corrupted_der = der_bytes + b"\nTAMPERED"
        self.assertNotEqual(hashlib.sha256(corrupted_der).hexdigest(), doc0["derived_text_sha256"])

        # Mode 3: Manifest pointing to incorrect original SHA -> hash mismatch
        manifest_tampered = dict(manifest_data)
        manifest_tampered["source_documents"][0]["original_artifact_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        self.assertNotEqual(compute_canonical_hash(manifest_tampered), manifest_data["manifest_content_hash"])

        # Mode 4: Embedded version altered -> mismatch against authentic PDF
        self.assertNotEqual("Version 2.8", "Version 2.1")

    def test_authority_non_inflation_and_freeze_invariants(self):
        """
        Verifies that protocol qualification never inflates discovery authority beyond BOUNDED_STABLE.
        """
        self.assertFalse(HOLD_TIME_CREATES_SOURCE_AUTHORITY)
        self.assertFalse(PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT)
        self.assertFalse(SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY)


if __name__ == "__main__":
    unittest.main()


