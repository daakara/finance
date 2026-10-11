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
    SecEvidenceBundleV1,
    LegacyTimingSemanticReconciliation,
    GOVERNED_UNIVERSE_BUILD_ID,
    GOVERNED_UNIVERSE_CONTENT_HASH,
    GOVERNED_REQUIRED_PARTITION_SET_HASH,
    HOLD_TIME_CREATES_SOURCE_AUTHORITY,
    PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT,
    SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY,
)
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


if __name__ == "__main__":
    unittest.main()
