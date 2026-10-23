"""Comprehensive Verification Suite for Corporate Events Telemetry Proof of Concept.

Tests:
1. Gate B: Schema validation, 17-field contract, strict privacy denial.
2. Gate C: Deduplication correctness (retransmissions vs legitimate repeated evaluations).
3. Gate D: Provenance classification taxonomy (all 10 classes) and denominator inclusion.
4. Gate E: Local collector simulation, thread-safe persistence, dead-letter routing.
5. Gate F: Failure simulation, retry, and self-healing recovery.
6. Gate G & I: End-to-end integration and accounting invariant verification.
"""

import uuid
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from analyst_dashboard.telemetry.corporate_event_contract import (
    CorporateEventTelemetryPayload,
    CorporateEventCategory,
    MarketSession,
    EventSourceStatus,
    FreshnessStatus,
    RiskBucket,
    EvaluationOutcome,
    compute_deduplication_key,
    PROHIBITED_PRIVACY_FIELDS,
)
from analyst_dashboard.telemetry.provenance import (
    ProvenanceClass,
    ProvenanceClassifier,
    ObservationEpochBounds,
)
from analyst_dashboard.telemetry.collector import (
    TelemetryCollector,
    IngestionStatus,
    TelemetryAccounting,
)

AUTH_SHA = "8404b6e4645b5f9f8237e70517dcab3b6afdda7f"
SAMPLE_OCCURRED_UTC = "2026-10-24T10:00:00Z"
SAMPLE_EVENT_UTC = "2026-10-25T14:00:00Z"


def make_valid_payload_dict(**overrides):
    """Helper to generate a valid canonical payload dictionary."""
    now = datetime.now(timezone.utc)
    base = {
        "schema_version": "1.0.0",
        "event_id": str(uuid.uuid4()),
        "occurred_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release_sha": AUTH_SHA,
        "ruleset_version": "RC5",
        "evaluation_context_id": f"ctx-{uuid.uuid4().hex[:8]}",
        "symbol": "AAPL",
        "event_category": CorporateEventCategory.EARNINGS_ANNOUNCEMENT.value,
        "event_time_utc": (now + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_session": MarketSession.BMO.value,
        "event_source_status": EventSourceStatus.CONFIRMED.value,
        "freshness_status": FreshnessStatus.FRESH.value,
        "days_to_earnings": 1.17,
        "risk_bucket": RiskBucket.CAUTION.value,
        "applied_penalty": -8.0,
        "evaluation_outcome": EvaluationOutcome.COMPLETED.value,
        "deployment_host": "arxterminal.com",
        "synthetic_marker": False,
        "ci_marker": False,
        "bot_score": 85,
    }
    base.update(overrides)
    return base


# ==============================================================================
# GATE B: SCHEMA & PRIVACY BOUNDARY TESTS
# ==============================================================================

class TestGateBSchemaAndPrivacy:
    """Tests schema validation, semantic risk invariants, and privacy boundaries."""

    def test_valid_payload_instantiation(self):
        payload_dict = make_valid_payload_dict()
        payload = CorporateEventTelemetryPayload.model_validate(payload_dict)
        assert payload.schema_version == "1.0.0"
        assert payload.release_sha == AUTH_SHA
        assert payload.symbol == "AAPL"
        assert payload.risk_bucket == RiskBucket.CAUTION
        assert payload.applied_penalty == -8.0
        assert payload.deduplication_key is not None
        assert len(payload.deduplication_key) == 64

    def test_semantic_boundaries_acute(self):
        # 0.5 days -> ACUTE (-25.0)
        p = CorporateEventTelemetryPayload.model_validate(
            make_valid_payload_dict(
                days_to_earnings=0.5,
                risk_bucket=RiskBucket.ACUTE.value,
                applied_penalty=-25.0,
            )
        )
        assert p.risk_bucket == RiskBucket.ACUTE
        assert p.applied_penalty == -25.0

    def test_semantic_boundaries_nominal(self):
        # 12.0 days -> NOMINAL (0.0)
        p = CorporateEventTelemetryPayload.model_validate(
            make_valid_payload_dict(
                days_to_earnings=12.0,
                risk_bucket=RiskBucket.NOMINAL.value,
                applied_penalty=0.0,
            )
        )
        assert p.risk_bucket == RiskBucket.NOMINAL
        assert p.applied_penalty == 0.0

    def test_semantic_boundaries_elapsed_event(self):
        # -1.5 days -> NOMINAL (0.0 penalty for historical event)
        p = CorporateEventTelemetryPayload.model_validate(
            make_valid_payload_dict(
                days_to_earnings=-1.5,
                risk_bucket=RiskBucket.NOMINAL.value,
                applied_penalty=0.0,
            )
        )
        assert p.risk_bucket == RiskBucket.NOMINAL
        assert p.applied_penalty == 0.0

    def test_semantic_boundaries_unavailable(self):
        # None days -> UNAVAILABLE (0.0 penalty)
        p = CorporateEventTelemetryPayload.model_validate(
            make_valid_payload_dict(
                days_to_earnings=None,
                event_time_utc=None,
                risk_bucket=RiskBucket.UNAVAILABLE.value,
                applied_penalty=0.0,
                event_source_status=EventSourceStatus.DATA_UNAVAILABLE.value,
                evaluation_outcome=EvaluationOutcome.UNAVAILABLE.value,
            )
        )
        assert p.risk_bucket == RiskBucket.UNAVAILABLE
        assert p.applied_penalty == 0.0

    def test_semantic_mismatch_penalty_rejected(self):
        # ACUTE with wrong penalty (-8.0 instead of -25.0) must fail
        with pytest.raises(ValidationError) as exc:
            CorporateEventTelemetryPayload.model_validate(
                make_valid_payload_dict(
                    days_to_earnings=0.5,
                    risk_bucket=RiskBucket.ACUTE.value,
                    applied_penalty=-8.0,
                )
            )
        assert "Semantic mismatch" in str(exc.value)

    def test_semantic_mismatch_boundary_rejected(self):
        # 0.5 days with CAUTION bucket must fail
        with pytest.raises(ValidationError) as exc:
            CorporateEventTelemetryPayload.model_validate(
                make_valid_payload_dict(
                    days_to_earnings=0.5,
                    risk_bucket=RiskBucket.CAUTION.value,
                    applied_penalty=-8.0,
                )
            )
        assert "Semantic boundary violation" in str(exc.value)

    def test_forbidden_extra_fields_rejected(self):
        # Unexpected fields are forbidden
        with pytest.raises(ValidationError) as exc:
            CorporateEventTelemetryPayload.model_validate(
                make_valid_payload_dict(random_extra_field="malicious_payload")
            )
        assert "extra_forbidden" in str(exc.value)

    @pytest.mark.parametrize("prohibited_key", [
        "user_id",
        "ip_address",
        "email",
        "account_id",
        "account_balance",
        "portfolio_value",
        "cookie",
        "device_id",
        "name",
        "personal_name",
        "auth_token",
    ])
    def test_strict_privacy_deny_list(self, prohibited_key):
        # Prohibited identifying keys must trigger explicit Privacy Boundary Violation
        bad_payload = make_valid_payload_dict()
        bad_payload[prohibited_key] = "test-sensitive-value"
        with pytest.raises(ValidationError) as exc:
            CorporateEventTelemetryPayload.model_validate(bad_payload)
        assert "Prohibited privacy field detected" in str(exc.value)


# ==============================================================================
# GATE C: DEDUPLICATION CORRECTNESS TESTS
# ==============================================================================

class TestGateCDeduplication:
    """Tests retransmission deduplication vs preservation of legitimate evaluations."""

    def test_retransmissions_deduplicated_single_counted(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict()

        # First delivery attempt
        payload["delivery_attempt_id"] = str(uuid.uuid4())
        res1 = collector.ingest(payload)
        assert res1.status == IngestionStatus.ACCEPTED
        assert res1.is_denominator_eligible is True

        # 2nd to 10th delivery attempts (same evaluation_id / deduplication_key, different delivery attempt)
        for i in range(2, 11):
            payload["delivery_attempt_id"] = str(uuid.uuid4())
            res = collector.ingest(payload)
            assert res.status == IngestionStatus.DUPLICATE_ACKNOWLEDGED
            assert res.is_denominator_eligible is False

        # Denominator must remain exactly 1
        assert collector.get_natural_denominator() == 1

        # Raw table must show all 10 transmission attempts
        accounting = collector.get_accounting_breakdown()
        assert accounting.n_valid == 10
        assert accounting.n_dedup == 1
        assert accounting.n_duplicate == 9
        assert accounting.n_natural == 1

    def test_distinct_contexts_preserved(self):
        collector = TelemetryCollector(db_path=":memory:")
        # Two evaluations 1 second apart in different context sessions
        p1 = make_valid_payload_dict(evaluation_context_id="ctx-session-001")
        p2 = make_valid_payload_dict(evaluation_context_id="ctx-session-002")

        res1 = collector.ingest(p1)
        res2 = collector.ingest(p2)

        assert res1.status == IngestionStatus.ACCEPTED
        assert res2.status == IngestionStatus.ACCEPTED
        assert res1.deduplication_key != res2.deduplication_key
        assert collector.get_natural_denominator() == 2

    def test_repeated_evaluation_different_time_preserved(self):
        collector = TelemetryCollector(db_path=":memory:")
        now = datetime.now(timezone.utc)
        t1_dt = now - timedelta(minutes=5)
        t2_dt = now
        t1 = t1_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        t2 = t2_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        p1 = make_valid_payload_dict(
            evaluation_context_id="ctx-same-session",
            occurred_at_utc=t1,
            evaluation_id="eval-hour-1",
        )
        p2 = make_valid_payload_dict(
            evaluation_context_id="ctx-same-session",
            occurred_at_utc=t2,
            evaluation_id="eval-hour-2",
        )

        res1 = collector.ingest(p1, collector_received_at=t1_dt)
        res2 = collector.ingest(p2, collector_received_at=t2_dt)

        assert res1.status == IngestionStatus.ACCEPTED
        assert res2.status == IngestionStatus.ACCEPTED
        assert collector.get_natural_denominator() == 2


# ==============================================================================
# GATE D: PROVENANCE CLASSIFICATION TAXONOMY TESTS
# ==============================================================================

class TestGateDProvenanceTaxonomy:
    """Verifies all 10 mutually exclusive provenance classes and inclusion criteria."""

    def test_natural_production_eligible(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict()
        res = collector.ingest(payload, headers={"host": "arxterminal.com"})

        assert res.status == IngestionStatus.ACCEPTED
        assert res.provenance_class == ProvenanceClass.NATURAL_PRODUCTION_ELIGIBLE
        assert res.is_denominator_eligible is True

    def test_synthetic_diagnostic_via_marker(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict(synthetic_marker=True)
        res = collector.ingest(payload)

        assert res.provenance_class == ProvenanceClass.SYNTHETIC_DIAGNOSTIC
        assert res.is_denominator_eligible is False
        assert collector.get_natural_denominator() == 0

    def test_synthetic_diagnostic_via_header(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict()
        res = collector.ingest(payload, headers={"X-Arx-Synthetic": "true"})

        assert res.provenance_class == ProvenanceClass.SYNTHETIC_DIAGNOSTIC
        assert res.is_denominator_eligible is False

    def test_automated_smoke_test_via_ci_marker(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict(ci_marker=True)
        res = collector.ingest(payload)

        assert res.provenance_class == ProvenanceClass.AUTOMATED_SMOKE_TEST
        assert res.is_denominator_eligible is False

    def test_automated_smoke_test_via_user_agent(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict()
        res = collector.ingest(payload, headers={"User-Agent": "ARX-Smoke-Verifier/1.0"})

        assert res.provenance_class == ProvenanceClass.AUTOMATED_SMOKE_TEST
        assert res.is_denominator_eligible is False

    def test_manual_qa_via_header(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict()
        res = collector.ingest(payload, headers={"X-Arx-QA-Session": "true"})

        assert res.provenance_class == ProvenanceClass.MANUAL_QA
        assert res.is_denominator_eligible is False

    def test_historical_import_before_t0(self):
        collector = TelemetryCollector(db_path=":memory:")
        # T0 is 2026-10-23T21:45:15Z; test an event occurred earlier
        payload = make_valid_payload_dict(occurred_at_utc="2026-10-23T18:00:00Z")
        res = collector.ingest(payload)

        assert res.provenance_class == ProvenanceClass.HISTORICAL_IMPORT
        assert res.is_denominator_eligible is False

    def test_outside_observation_epoch_after_t72(self):
        collector = TelemetryCollector(db_path=":memory:")
        # T72 is 2026-10-26T21:45:15Z; test an event occurred after
        payload = make_valid_payload_dict(occurred_at_utc="2026-10-27T00:00:00Z")
        res = collector.ingest(payload)

        assert res.provenance_class == ProvenanceClass.OUTSIDE_OBSERVATION_EPOCH
        assert res.is_denominator_eligible is False

    def test_unverifiable_provenance_unauthorized_host(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict(deployment_host="unauthorized-malicious-domain.com")
        res = collector.ingest(payload)

        assert res.provenance_class == ProvenanceClass.UNVERIFIABLE_PROVENANCE
        assert res.is_denominator_eligible is False

    def test_malformed_rejected(self):
        collector = TelemetryCollector(db_path=":memory:")
        bad_dict = {"corrupted": "bad payload without required fields"}
        res = collector.ingest(bad_dict)

        assert res.status == IngestionStatus.REJECTED_MALFORMED
        assert res.provenance_class == ProvenanceClass.MALFORMED
        assert res.is_denominator_eligible is False


# ==============================================================================
# GATE E & F: LOCAL PERSISTENCE, CONCURRENCY & FAILURE RECOVERY TESTS
# ==============================================================================

class TestGateEandFCollectorAndFailureRecovery:
    """Tests atomic persistence, dead-letter logging, concurrent writes, and failure recovery."""

    def test_dead_letter_logging_for_privacy_violation(self):
        collector = TelemetryCollector(db_path=":memory:")
        bad_payload = make_valid_payload_dict()
        bad_payload["user_id"] = "usr_123456"

        res = collector.ingest(bad_payload)
        assert res.status == IngestionStatus.REJECTED_PRIVACY_VIOLATION
        assert res.is_denominator_eligible is False

        # Verify dead letter table has the entry
        with collector.get_connection() as conn:
            dl_rows = conn.execute("SELECT * FROM telemetry_dead_letter").fetchall()
            assert len(dl_rows) == 1
            assert "Privacy Boundary Violation" in dl_rows[0]["rejection_reason"]
            assert "user_id" in dl_rows[0]["prohibited_fields"]

            # Evaluations table must be strictly 0 rows
            eval_rows = conn.execute("SELECT * FROM corporate_event_evaluations").fetchall()
            assert len(eval_rows) == 0

    def test_concurrency_parallel_duplicate_and_unique(self):
        collector = TelemetryCollector(db_path=":memory:")
        p_shared = make_valid_payload_dict(symbol="MSFT")

        # Ingest 20 concurrent requests: 10 identical duplicates of p_shared, and 10 unique payloads
        unique_payloads = [
            make_valid_payload_dict(symbol=f"T{i:02d}", evaluation_context_id=f"ctx-unique-{i}")
            for i in range(10)
        ]

        all_tasks = [dict(p_shared) for _ in range(10)] + unique_payloads

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(collector.ingest, all_tasks))

        # 10 unique + 1 shared = 11 unique evaluations in denominator
        assert collector.get_natural_denominator() == 11

        accounting = collector.get_accounting_breakdown()
        assert accounting.n_dedup == 11
        assert accounting.n_duplicate == 9
        assert accounting.n_valid == 20
        is_valid, errors = accounting.verify_invariants()
        assert is_valid, f"Accounting invariants failed: {errors}"

    def test_simulated_storage_failure_and_idempotent_recovery(self):
        collector = TelemetryCollector(db_path=":memory:")
        payload = make_valid_payload_dict(symbol="NVDA")

        # 1. Inject failure
        collector.set_simulated_failure("STORAGE_WRITE_FAILURE")
        res1 = collector.ingest(payload)
        assert res1.status == IngestionStatus.TRANSIENT_FAILURE
        assert collector.get_natural_denominator() == 0

        # 2. Clear failure and retry
        collector.set_simulated_failure(None)
        res2 = collector.ingest(payload)
        assert res2.status == IngestionStatus.ACCEPTED
        assert collector.get_natural_denominator() == 1

        # 3. Subsequent retry should be acknowledged as duplicate
        res3 = collector.ingest(payload)
        assert res3.status == IngestionStatus.DUPLICATE_ACKNOWLEDGED
        assert collector.get_natural_denominator() == 1


# ==============================================================================
# GATE G & I: END-TO-END ACCOUNTING INVARIANTS
# ==============================================================================

class TestGateGAndIAccountingInvariants:
    """Tests the full mathematical accounting ledger identities under mixed real-world traffic."""

    def test_full_accounting_stream_balance(self):
        collector = TelemetryCollector(db_path=":memory:")

        # 1. Five legitimate natural production evaluations
        for i in range(5):
            collector.ingest(make_valid_payload_dict(symbol=f"NAT{i}"))

        # 2. Three natural evaluations each retried 3 times (3 accepted, 6 duplicates)
        for i in range(3):
            p = make_valid_payload_dict(symbol=f"RET{i}")
            collector.ingest(p)  # attempt 1: accepted
            collector.ingest(p)  # attempt 2: duplicate
            collector.ingest(p)  # attempt 3: duplicate

        # 3. Four synthetic diagnostic payloads
        for i in range(4):
            collector.ingest(make_valid_payload_dict(symbol=f"SYN{i}", synthetic_marker=True))

        # 4. Two automated smoke test payloads
        for i in range(2):
            collector.ingest(make_valid_payload_dict(symbol=f"SMK{i}", ci_marker=True))

        # 5. One manual QA payload
        collector.ingest(
            make_valid_payload_dict(symbol="QAA1"),
            headers={"X-Arx-QA-Session": "true"},
        )

        # 6. Two malformed payloads
        collector.ingest({"bad": "malformed 1"})
        collector.ingest({"bad": "malformed 2"})

        # 7. Two privacy-violating payloads
        collector.ingest(make_valid_payload_dict(user_id="user-999"))
        collector.ingest(make_valid_payload_dict(ip_address="192.168.1.1"))

        # 8. One historical import (before T0)
        collector.ingest(make_valid_payload_dict(symbol="HIST", occurred_at_utc="2026-10-23T19:00:00Z"))

        # 9. One out-of-epoch (after T72)
        collector.ingest(make_valid_payload_dict(symbol="LATE", occurred_at_utc="2026-10-27T02:00:00Z"))

        accounting = collector.get_accounting_breakdown()

        # Check raw accounting:
        # Valid raw receipts: 5 (nat) + 9 (retries) + 4 (syn) + 2 (smk) + 1 (qa) + 1 (hist) + 1 (late) = 23
        # Malformed / privacy receipts: 2 (malformed) + 2 (privacy) = 4
        # Total raw receipts = 23 + 4 = 27
        assert accounting.n_raw == 27
        assert accounting.n_valid == 23
        assert accounting.n_malformed == 4

        # Deduplication accounting:
        # Total unique evaluations = 5 (nat) + 3 (ret) + 4 (syn) + 2 (smk) + 1 (qa) + 1 (hist) + 1 (late) = 17
        # Duplicates = 6
        assert accounting.n_dedup == 17
        assert accounting.n_duplicate == 6

        # Classification breakdown of unique evaluations:
        # Natural production denominator = 5 + 3 = 8
        assert accounting.n_natural == 8
        assert collector.get_natural_denominator() == 8

        # Synthetic (synthetic + smoke + qa) = 4 + 2 + 1 = 7
        assert accounting.n_synthetic == 7

        # Other (historical + late) = 1 + 1 = 2
        assert accounting.n_other == 2

        # Verify mathematical identities
        is_valid, errors = accounting.verify_invariants()
        assert is_valid, f"Accounting invariants violated: {errors}"
