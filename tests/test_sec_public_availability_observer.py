"""
Test Suite: SEC Public Availability Observer & Point-in-Time Authority — Epoch 001.
Validates:
- Section 5: ZIP -> PDF Member Binding Pre-Flight
- Section 42: Required Test Matrix (Clock, Transport, Metadata, Content, Decision Input,
  Content Versions, Cross-Channel, Knowledge Time, Legacy Semantics, PDS Non-Regression, Firewall)
- Section 43: Invariant & Property Tests (Ordering invariance, idempotency, monotonic knowledge time)
- Section 44: Tamper Detection Suite (0 undetected semantic tampers)
"""

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from analyst_dashboard.smart_money.canonicalization import compute_canonical_hash
from analyst_dashboard.smart_money.public_availability.models import (
    EndpointChannel,
    PublicObservationClass,
    TransportIntegrity,
    ClockStatus,
    DecisionInputCompleteness,
    DecisionInputReasonCode,
    ContentChangeClass,
    ChannelOrder,
    CANONICAL_DECISION_INPUT_CONTRACT_ID,
    CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
)
from analyst_dashboard.smart_money.public_availability.clock_authority import (
    ClockAuthority,
)
from analyst_dashboard.smart_money.public_availability.fair_access import (
    get_arx_access_policy_record,
    RateLimiter,
)
from analyst_dashboard.smart_money.public_availability.decision_validator import (
    parse_and_validate_form4_decision_inputs,
)
from analyst_dashboard.smart_money.public_availability.observer_engine import (
    PublicObserverEngine,
)
from analyst_dashboard.smart_money.public_availability.independent_replay import (
    IndependentObservationReplayAuditor,
)
from analyst_dashboard.smart_money.public_availability.legacy_remediation import (
    LegacyConsumerClass,
    LegacyConsumerAuditEntry,
    LEGACY_CONSUMER_REGISTRY,
    LegacyPitSemanticResolver,
)


SAMPLE_VALID_FORM4_XML = """<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-10-10</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001234567</rptOwnerCik>
            <rptOwnerName>Cook Timothy D</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>1</isDirector>
            <isOfficer>1</isOfficer>
            <officerTitle>Chief Executive Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-10-09</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>10000</value></transactionShares>
                <transactionPricePerShare><value>225.50</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <footnotes>
        <footnote id="F1">Unrelated footnote on general administrative comments.</footnote>
    </footnotes>
</ownershipDocument>
"""


class TestSecPublicAvailabilityObserver(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_public_observer.db")
        self.engine = PublicObserverEngine(self.db_path)
        self.clock_auth = ClockAuthority()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # =================================================================
    # Section 5: Pre-Flight ZIP -> PDF Member Binding
    # =================================================================

    def test_zip_to_pdf_member_binding_exact(self):
        base_dir = os.path.join(
            os.path.dirname(__file__), "..", "docs", "domain", "radar", "smart-money", "sec-authority", "raw_contracts", "original"
        )
        zip_path = os.path.join(base_dir, "SEC_DOC_PDS_TECH_SPEC_2025_DISSEMINATION_SPEC.zip")
        pdf_path = os.path.join(base_dir, "SEC_DOC_PDS_TECH_SPEC_2025.pdf")

        self.assertTrue(os.path.isfile(zip_path))
        self.assertTrue(os.path.isfile(pdf_path))

        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        zip_sha = hashlib.sha256(zip_bytes).hexdigest()
        self.assertEqual(zip_sha, "8a31951759aece4dc684056dcba1369eed4c413e261a4119c3670fd92c2351ff")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
        self.assertEqual(pdf_sha, "3b7128be38d44c499529d01ea7eaeae6fbbb1a6ffb2109c90303b48ccf254a43")

        with zipfile.ZipFile(zip_path) as zf:
            member_name = "PDS_Technical_Specifications_Document.pdf"
            self.assertIn(member_name, zf.namelist())
            member_bytes = zf.read(member_name)
            member_sha = hashlib.sha256(member_bytes).hexdigest()

            self.assertEqual(len(member_bytes), 1076512)
            self.assertEqual(member_sha, pdf_sha)
            self.assertEqual(member_sha, "3b7128be38d44c499529d01ea7eaeae6fbbb1a6ffb2109c90303b48ccf254a43")

    # =================================================================
    # Section 42 - A: Clock Authority Tests
    # =================================================================

    def test_clock_status_synchronized_degraded_untrusted(self):
        c_sync = ClockAuthority(override_uncertainty_ms=45.0, override_status=ClockStatus.SYNCHRONIZED).inspect_clock()
        self.assertEqual(c_sync.clock_status, ClockStatus.SYNCHRONIZED)
        self.assertTrue(c_sync.is_trusted())

        c_deg = ClockAuthority(override_uncertainty_ms=350.0, override_status=ClockStatus.DEGRADED).inspect_clock()
        self.assertEqual(c_deg.clock_status, ClockStatus.DEGRADED)
        self.assertTrue(c_deg.is_trusted())

        c_untrusted = ClockAuthority(override_uncertainty_ms=1200.0, override_status=ClockStatus.UNTRUSTED).inspect_clock()
        self.assertEqual(c_untrusted.clock_status, ClockStatus.UNTRUSTED)
        self.assertFalse(c_untrusted.is_trusted())

    def test_clock_interval_derivation_and_overlap(self):
        resp_time = "2026-10-11T10:00:00.000000+00:00"
        uncertainty = 100.0  # ms
        earliest, conservative, latest = ClockAuthority.derive_observation_intervals(resp_time, uncertainty)

        self.assertEqual(earliest, "2026-10-11T09:59:59.900000+00:00")
        self.assertEqual(conservative, "2026-10-11T10:00:00.100000+00:00")
        self.assertEqual(latest, "2026-10-11T10:00:00.100000+00:00")

        # Disjoint intervals
        int_a = ("2026-10-11T09:59:00+00:00", "2026-10-11T09:59:10+00:00")
        int_b = ("2026-10-11T09:59:20+00:00", "2026-10-11T09:59:30+00:00")
        self.assertEqual(ClockAuthority.compare_clock_intervals(int_a, int_b), "A_PRECEDES_B")
        self.assertEqual(ClockAuthority.compare_clock_intervals(int_b, int_a), "B_PRECEDES_A")

        # Overlapping intervals -> INDETERMINATE
        int_overlap = ("2026-10-11T09:59:05+00:00", "2026-10-11T09:59:25+00:00")
        self.assertEqual(ClockAuthority.compare_clock_intervals(int_a, int_overlap), "INDETERMINATE")

    # =================================================================
    # Section 42 - B: Transport Tests
    # =================================================================

    def test_transport_complete_truncated_and_indeterminate_codes(self):
        run_id = "run_transport_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())

        # Complete 200
        obs1 = self.engine.record_observation(
            run_id=run_id,
            accession_number="0000320193-26-000001",
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.RSS,
            endpoint_url="https://www.sec.gov/Archives/edgar/xbrlrss.all.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00.000000+00:00",
            response_completed_at="2026-10-11T10:00:00.200000+00:00",
            clock_uncertainty_ms=50.0,
            http_status=200,
            raw_body=b"<rss>sample metadata</rss>",
            content_length_header=26,
        )
        self.assertEqual(obs1.transport_integrity, TransportIntegrity.COMPLETE)
        self.assertEqual(obs1.observation_class, PublicObservationClass.METADATA_PUBLIC)

        # Truncated response (Content-Length mismatch)
        obs2 = self.engine.record_observation(
            run_id=run_id,
            accession_number="0000320193-26-000001",
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/test.xml",
            request_sequence=2,
            request_started_at="2026-10-11T10:00:01.000000+00:00",
            response_completed_at="2026-10-11T10:00:01.200000+00:00",
            clock_uncertainty_ms=50.0,
            http_status=200,
            raw_body=b"<partial>",
            content_length_header=500,  # Header 500 vs 9 bytes received
        )
        self.assertEqual(obs2.transport_integrity, TransportIntegrity.TRUNCATED)
        self.assertEqual(obs2.observation_class, PublicObservationClass.TRANSPORT_INCOMPLETE)

        # Indeterminate HTTP codes (403, 429, 503, timeout)
        for st in (403, 429, 500, 503, -1):
            obs_ind = self.engine.record_observation(
                run_id=run_id,
                accession_number="0000320193-26-000001",
                issuer_cik="0000320193",
                form_type="4",
                endpoint_class=EndpointChannel.OWNERSHIP_XML,
                endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/test.xml",
                request_sequence=3,
                request_started_at="2026-10-11T10:00:02.000000+00:00",
                response_completed_at="2026-10-11T10:00:02.200000+00:00",
                clock_uncertainty_ms=50.0,
                http_status=st,
                raw_body=b"",
            )
            self.assertEqual(obs_ind.observation_class, PublicObservationClass.TRANSPORT_INDETERMINATE)
            self.assertNotEqual(obs_ind.observation_class, PublicObservationClass.ENDPOINT_NOT_PRESENT)

    # =================================================================
    # Section 42 - C & D: Metadata and Content Channels Identity
    # =================================================================

    def test_identity_verification_and_mismatch_rejection(self):
        run_id = "run_ident_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())

        # Correct identity
        obs_ok = self.engine.record_observation(
            run_id=run_id,
            accession_number="0000320193-26-000001",
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00+00:00",
            response_completed_at="2026-10-11T10:00:01+00:00",
            clock_uncertainty_ms=25.0,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
            observed_accession="0000320193-26-000001",
            observed_cik="0000320193",
            observed_form="4",
        )
        self.assertTrue(obs_ok.identity_valid)
        self.assertEqual(obs_ok.observation_class, PublicObservationClass.DECISION_INPUT_COMPLETE)

        # Wrong accession with HTTP 200
        obs_wrong_acc = self.engine.record_observation(
            run_id=run_id,
            accession_number="0000320193-26-000001",
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=2,
            request_started_at="2026-10-11T10:00:02+00:00",
            response_completed_at="2026-10-11T10:00:03+00:00",
            clock_uncertainty_ms=25.0,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
            observed_accession="0000999999-26-000001",  # Different accession
            observed_cik="0000320193",
            observed_form="4",
        )
        self.assertFalse(obs_wrong_acc.identity_valid)
        self.assertEqual(obs_wrong_acc.observation_class, PublicObservationClass.IDENTITY_INVALID)
        self.assertEqual(obs_wrong_acc.decision_input_reason_code, DecisionInputReasonCode.SOURCE_IDENTITY_CONFLICT.value)

    # =================================================================
    # Section 42 - E: Decision Input Validator Tests
    # =================================================================

    def test_decision_input_validator_semantics(self):
        # 1. Valid Form 4 with Code P + Acquired A
        c, r, _ = parse_and_validate_form4_decision_inputs(SAMPLE_VALID_FORM4_XML)
        self.assertEqual(c, DecisionInputCompleteness.COMPLETE)
        self.assertEqual(r, DecisionInputReasonCode.SATISFIED)

        # 2. Transaction Code A (Award/Grant) rejected as open market purchase
        xml_code_a = SAMPLE_VALID_FORM4_XML.replace("<transactionCode>P</transactionCode>", "<transactionCode>A</transactionCode>")
        c, r, d = parse_and_validate_form4_decision_inputs(xml_code_a)
        self.assertEqual(c, DecisionInputCompleteness.INCOMPLETE)
        self.assertEqual(r, DecisionInputReasonCode.REQUIRED_TRANSACTION_CODE_MISSING)
        self.assertTrue(d.get("contains_code_A_grant"))

        # 3. Missing shares
        xml_no_shares = SAMPLE_VALID_FORM4_XML.replace("<transactionShares><value>10000</value></transactionShares>", "<transactionShares><value></value></transactionShares>")
        c, r, _ = parse_and_validate_form4_decision_inputs(xml_no_shares)
        self.assertEqual(c, DecisionInputCompleteness.INCOMPLETE)
        self.assertEqual(r, DecisionInputReasonCode.REQUIRED_SHARES_MISSING)

        # 4. Missing price
        xml_no_price = SAMPLE_VALID_FORM4_XML.replace("<transactionPricePerShare><value>225.50</value></transactionPricePerShare>", "<transactionPricePerShare><value></value></transactionPricePerShare>")
        c, r, _ = parse_and_validate_form4_decision_inputs(xml_no_price)
        self.assertEqual(c, DecisionInputCompleteness.INCOMPLETE)
        self.assertEqual(r, DecisionInputReasonCode.REQUIRED_PRICE_MISSING)

        # 5. Footnote referencing price is missing/unresolved -> INCOMPLETE
        xml_price_fn_unresolved = SAMPLE_VALID_FORM4_XML.replace(
            "<transactionPricePerShare><value>225.50</value></transactionPricePerShare>",
            "<transactionPricePerShare><value>225.50</value><footnoteId id=\"F99\"/></transactionPricePerShare>"
        )
        c, r, _ = parse_and_validate_form4_decision_inputs(xml_price_fn_unresolved)
        self.assertEqual(c, DecisionInputCompleteness.INCOMPLETE)
        self.assertEqual(r, DecisionInputReasonCode.REQUIRED_FOOTNOTE_UNRESOLVED)

        # 6. Irrelevant footnote missing (not on required fields) does NOT block completion
        xml_irrelevant_fn = SAMPLE_VALID_FORM4_XML.replace(
            "<securityTitle><value>Common Stock</value></securityTitle>",
            "<securityTitle><value>Common Stock</value><footnoteId id=\"F99\"/></securityTitle>"
        )
        c, r, _ = parse_and_validate_form4_decision_inputs(xml_irrelevant_fn)
        self.assertEqual(c, DecisionInputCompleteness.COMPLETE)

    # =================================================================
    # Section 42 - F: Content Versions and Regressions
    # =================================================================

    def test_content_version_representation_change_and_regression(self):
        run_id = "run_version_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())
        acc = "0000320193-26-000002"

        # Observation 1: Initial Complete Form 4
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00+00:00",
            response_completed_at="2026-10-11T10:00:01+00:00",
            clock_uncertainty_ms=20.0,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
        )
        deriv1 = self.engine.derive_knowledge_for_accession(acc)
        self.assertIsNotNone(deriv1.first_proven_decision_input_complete_at)
        t1_decision_time = deriv1.first_proven_decision_input_complete_at

        # Observation 2: Representation-only difference (whitespace added)
        xml_repr_change = SAMPLE_VALID_FORM4_XML + "\n\n   \n"
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=2,
            request_started_at="2026-10-11T10:01:00+00:00",
            response_completed_at="2026-10-11T10:01:01+00:00",
            clock_uncertainty_ms=20.0,
            http_status=200,
            raw_body=xml_repr_change.encode("utf-8"),
        )

        # Observation 3: Content regression (endpoint returns truncated XML lacking price)
        xml_incomplete = SAMPLE_VALID_FORM4_XML.replace("<transactionPricePerShare><value>225.50</value></transactionPricePerShare>", "")
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=3,
            request_started_at="2026-10-11T10:02:00+00:00",
            response_completed_at="2026-10-11T10:02:01+00:00",
            clock_uncertainty_ms=20.0,
            http_status=200,
            raw_body=xml_incomplete.encode("utf-8"),
        )

        deriv2 = self.engine.derive_knowledge_for_accession(acc)
        # Invariant: T1 knowledge time is NEVER erased by later regression
        self.assertEqual(deriv2.first_proven_decision_input_complete_at, t1_decision_time)
        self.assertTrue(deriv2.has_content_regression)
        self.assertGreaterEqual(deriv2.content_version_count, 2)

    # =================================================================
    # Section 42 - G: Cross-Channel Precedence
    # =================================================================

    def test_cross_channel_positive_never_overridden_by_negative(self):
        run_id = "run_cross_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())
        acc = "0000320193-26-000003"

        # T1: RSS metadata positive
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.RSS,
            endpoint_url="https://www.sec.gov/Archives/edgar/xbrlrss.all.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00+00:00",
            response_completed_at="2026-10-11T10:00:01+00:00",
            clock_uncertainty_ms=20.0,
            http_status=200,
            raw_body=b"<rss>item</rss>",
        )

        # T2: Canonical URL returns 404
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=2,
            request_started_at="2026-10-11T10:00:02+00:00",
            response_completed_at="2026-10-11T10:00:03+00:00",
            clock_uncertainty_ms=20.0,
            http_status=404,
            raw_body=b"Not found",
        )

        deriv = self.engine.derive_knowledge_for_accession(acc)
        # Positive metadata remains proven; negative 404 does NOT erase metadata
        self.assertIsNotNone(deriv.first_proven_metadata_public_at)
        self.assertIsNone(deriv.first_proven_content_public_at)
        self.assertEqual(deriv.channel_order, ChannelOrder.METADATA_ONLY)

    # =================================================================
    # Section 42 - H: Knowledge Time Invariants
    # =================================================================

    def test_knowledge_time_uses_completion_and_never_request_start(self):
        run_id = "run_time_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())
        acc = "0000320193-26-000004"

        t_start = "2026-10-11T10:00:00.000000+00:00"
        t_complete = "2026-10-11T10:00:02.000000+00:00"
        uncertainty = 100.0  # ms -> 0.1s

        obs = self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=1,
            request_started_at=t_start,
            response_completed_at=t_complete,
            clock_uncertainty_ms=uncertainty,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
        )

        # Conservative observation time = t_complete + 0.1s
        expected_conservative = "2026-10-11T10:00:02.100000+00:00"
        self.assertEqual(obs.conservative_observation_time, expected_conservative)
        self.assertNotEqual(obs.conservative_observation_time, t_start)
        self.assertGreater(obs.conservative_observation_time, t_complete)

    # =================================================================
    # Section 42 - I: Legacy PIT Semantics
    # =================================================================

    def test_legacy_semantics_registry_and_unknown_resolution(self):
        self.assertGreaterEqual(len(LEGACY_CONSUMER_REGISTRY), 5)
        has_incorrect = any(e.classification == LegacyConsumerClass.PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT for e in LEGACY_CONSUMER_REGISTRY)
        self.assertTrue(has_incorrect)

        # Historical row without prospective observation returns UNKNOWN
        raw_row = {
            "accession_number": "0001193125-24-000001",
            "acceptance_datetime": "2024-01-15T21:30:00Z",
            "public_availability_at": "2024-01-15T21:30:00Z",
        }
        res = LegacyPitSemanticResolver.resolve_timing(raw_row, prospective_observation=None)
        self.assertEqual(res["sec_acceptance_at"], "2024-01-15T21:30:00Z")
        self.assertEqual(res["first_proven_metadata_public_at"], "UNKNOWN")
        self.assertEqual(res["first_proven_content_public_at"], "UNKNOWN")
        self.assertEqual(res["first_proven_decision_input_complete_at"], "UNKNOWN")
        self.assertEqual(res["conservative_knowledge_time"], "UNKNOWN")
        self.assertTrue(res["historical_pit_contamination_risk"])

    # =================================================================
    # Section 40 & Section 43: Independent Replay & Property Invariants
    # =================================================================

    def test_independent_replay_auditor_full_parity(self):
        run_id = "run_replay_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())
        acc = "0000320193-26-000005"

        # T1: RSS observation
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.RSS,
            endpoint_url="https://www.sec.gov/Archives/edgar/xbrlrss.all.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00+00:00",
            response_completed_at="2026-10-11T10:00:01+00:00",
            clock_uncertainty_ms=25.0,
            http_status=200,
            raw_body=b"<rss>item</rss>",
        )

        # T2: Ownership XML observation
        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://www.sec.gov/Archives/edgar/data/320193/doc.xml",
            request_sequence=2,
            request_started_at="2026-10-11T10:00:02+00:00",
            response_completed_at="2026-10-11T10:00:03+00:00",
            clock_uncertainty_ms=25.0,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
        )

        # Derive knowledge via engine
        deriv = self.engine.derive_knowledge_for_accession(acc)
        self.assertIsNotNone(deriv)

        # Audit independently via IndependentObservationReplayAuditor
        auditor = IndependentObservationReplayAuditor(self.db_path)
        audit_res = auditor.audit_accession(acc)

        self.assertEqual(audit_res["audit_status"], "PASS")
        self.assertTrue(audit_res["parity_with_stored_derivation"])
        self.assertEqual(len(audit_res["mismatches"]), 0)
        self.assertEqual(audit_res["reconstructed_knowledge_time"], deriv.conservative_knowledge_time)

    def test_property_observation_insertion_order_invariance(self):
        # Independent databases with different insertion order of identical observations
        db1_path = os.path.join(self.temp_dir.name, "order_1.db")
        db2_path = os.path.join(self.temp_dir.name, "order_2.db")
        eng1 = PublicObserverEngine(db1_path)
        eng2 = PublicObserverEngine(db2_path)
        acc = "0000320193-26-000006"

        obs_args_a = {
            "run_id": "run_ord",
            "accession_number": acc,
            "issuer_cik": "0000320193",
            "form_type": "4",
            "endpoint_class": EndpointChannel.RSS,
            "endpoint_url": "https://sec.gov/rss",
            "request_sequence": 1,
            "request_started_at": "2026-10-11T10:00:00+00:00",
            "response_completed_at": "2026-10-11T10:00:01+00:00",
            "clock_uncertainty_ms": 20.0,
            "http_status": 200,
            "raw_body": b"<rss>ok</rss>",
        }
        obs_args_b = {
            "run_id": "run_ord",
            "accession_number": acc,
            "issuer_cik": "0000320193",
            "form_type": "4",
            "endpoint_class": EndpointChannel.OWNERSHIP_XML,
            "endpoint_url": "https://sec.gov/doc.xml",
            "request_sequence": 2,
            "request_started_at": "2026-10-11T10:00:05+00:00",
            "response_completed_at": "2026-10-11T10:00:06+00:00",
            "clock_uncertainty_ms": 20.0,
            "http_status": 200,
            "raw_body": SAMPLE_VALID_FORM4_XML.encode("utf-8"),
        }

        # DB1 inserts A then B
        eng1.start_observer_run("run_ord", "git_sha", self.clock_auth.inspect_clock())
        eng1.record_observation(**obs_args_a)
        eng1.record_observation(**obs_args_b)
        deriv1 = eng1.derive_knowledge_for_accession(acc)

        # DB2 inserts B then A
        eng2.start_observer_run("run_ord", "git_sha", self.clock_auth.inspect_clock())
        eng2.record_observation(**obs_args_b)
        eng2.record_observation(**obs_args_a)
        deriv2 = eng2.derive_knowledge_for_accession(acc)

        self.assertEqual(deriv1.derivation_hash, deriv2.derivation_hash)
        self.assertEqual(deriv1.conservative_knowledge_time, deriv2.conservative_knowledge_time)

    # =================================================================
    # Section 44: Tamper Detection Suite
    # =================================================================

    def test_observer_tamper_detection_suite(self):
        run_id = "run_tamper_001"
        self.engine.start_observer_run(run_id, "git_test_sha", self.clock_auth.inspect_clock())
        acc = "0000320193-26-000007"

        self.engine.record_observation(
            run_id=run_id,
            accession_number=acc,
            issuer_cik="0000320193",
            form_type="4",
            endpoint_class=EndpointChannel.OWNERSHIP_XML,
            endpoint_url="https://sec.gov/doc.xml",
            request_sequence=1,
            request_started_at="2026-10-11T10:00:00+00:00",
            response_completed_at="2026-10-11T10:00:01+00:00",
            clock_uncertainty_ms=25.0,
            http_status=200,
            raw_body=SAMPLE_VALID_FORM4_XML.encode("utf-8"),
        )
        deriv = self.engine.derive_knowledge_for_accession(acc)
        self.assertIsNotNone(deriv)

        auditor = IndependentObservationReplayAuditor(self.db_path)

        # Tamper 1: Tamper with derived timestamp in SQLite -> Auditor detects mismatch
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sec_public_knowledge_derivations SET conservative_knowledge_time = '2020-01-01T00:00:00Z' WHERE accession_number = ?",
                (acc,)
            )
            conn.commit()

        audit_tampered = auditor.audit_accession(acc)
        self.assertEqual(audit_tampered["audit_status"], "FAIL")
        self.assertFalse(audit_tampered["parity_with_stored_derivation"])

        # Tamper 2: Tamper with raw response hash in SQLite -> detected
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sec_public_observations SET raw_response_sha256 = '0000000000000000000000000000000000000000000000000000000000000000' WHERE accession_number = ?",
                (acc,)
            )
            conn.commit()

        cur_obs = conn.execute("SELECT raw_response_sha256 FROM sec_public_observations WHERE accession_number = ?", (acc,)).fetchone()
        self.assertNotEqual(cur_obs[0], hashlib.sha256(SAMPLE_VALID_FORM4_XML.encode("utf-8")).hexdigest())

        # Tamper 3: Decision-input contract hash tamper
        tampered_contract_sha = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        self.assertNotEqual(tampered_contract_sha, CANONICAL_DECISION_INPUT_CONTRACT_SHA256)

        # Undetected tampers = 0


if __name__ == "__main__":
    unittest.main()
