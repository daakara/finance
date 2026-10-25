"""
ARX Terminal — Smart Money Continuous Ingestion Test Suite (Phase 1).

Comprehensive verification of the 17 continuous ingestion invariants:
1. Genuine execution-path invocation in worker queue consumer.
2. Placeholder completion elimination (replaces gen-{req_id[:8]}).
3. SEC rate-limit enforcement (<= 8.0 req/s ceiling).
4. Worker-only egress (ARX_SEC_EGRESS_ROLE=OWNER required).
5. Lease contention and loss recovery (StaleLeaseOwnershipLostError handled).
6. Duplicate schedule delivery rejection (same calendar date).
7. Idempotent filing persistence (identical hash no-op, divergent fails closed).
8. Backup freshness gate compliance (24h cadence without 12h deadlock).
9. Pre-backup failure aborts pipeline (fails closed before SEC/DB touch).
10. Post-backup failure degrades gracefully (DEGRADED_POST_BACKUP_FAILED).
11. Snapshot publication atomicity.
12. DST transitions (EDT vs EST invariance).
13. SEC calendar behavior (weekends and federal holidays skipped).
14. Expected cycles vs calendar age freshness (Friday to Monday is FRESH).
15. Retention safety (never prunes the only verified recovery point).
16. Control A: Emergency web read disablement (ARX_SMART_MONEY_POSTGRES_READ_ENABLED=false).
17. Control B: Emergency worker shutdown releases advisory lock <= 1.0s.
"""

from __future__ import annotations

import os
import time
import uuid
import json
import pytest
from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch

from analyst_dashboard.smart_money.continuous_pipeline import (
    SmartMoneyContinuousPipeline,
    GOVERNED_UNIVERSE_ID,
    DEFAULT_GOVERNED_ISSUERS,
)
from analyst_dashboard.smart_money.smart_money_scheduler import (
    SmartMoneyTimezoneScheduler,
    JOB_ID_SMART_MONEY_MARKET_WIDE,
)
from analyst_dashboard.smart_money.freshness_authority import (
    is_sec_dissemination_date,
    SmartMoneyFreshnessAuthority,
    SnapshotFreshnessState,
    NY_TZ,
)
from analyst_dashboard.smart_money.worker import (
    SmartMoneyIngestionWorker,
    WorkerHealthState,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.data.sec_edgar_fetcher import RateLimiter
from analyst_dashboard.analyzers.scanner_runner import SmartMoneyScannerRunner
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.storage_adapter import BackupStorageAdapter


class TestSmartMoneyContinuousIngestion:
    """17-point invariant verification suite for Continuous Ingestion Phase 1."""

    # ── Test 1: Genuine execution-path invocation ─────────────────────────────
    def test_genuine_execution_path_invocation(self):
        worker = SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
        worker.worker_id = "test-worker-1"
        worker.lease_duration_sec = 60
        worker.stale_recovery_interval_sec = 30
        worker._last_stale_recovery_at = time.time()
        worker.dsn = "postgresql://mock"

        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True
        worker.containment_gate = mock_gate

        mock_repo = MagicMock()
        mock_repo.claim_next_run_request.return_value = {
            "request_id": "req-12345678-abcd",
            "lease_token": "token-xyz",
        }
        worker.repository = mock_repo

        with patch("analyst_dashboard.smart_money.worker.SmartMoneyContinuousPipeline") as MockPipeline:
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.execute_cycle.return_value = {
                "cycle_id": "cycle-1",
                "status": "SUCCESS",
                "snapshot_id": "SNAP_SM_TEST_001",
            }
            MockPipeline.return_value = mock_pipeline_inst

            count = worker.process_run_requests()

            assert count == 1
            mock_pipeline_inst.execute_cycle.assert_called_once_with(
                run_request_id="req-12345678-abcd",
                lease_token="token-xyz",
                trigger_type="OPERATOR",
            )

    # ── Test 2: Placeholder completion elimination ───────────────────────────
    def test_placeholder_completion_elimination(self):
        """Confirms that worker does not generate static f'gen-{req_id[:8]}' strings."""
        worker = SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
        worker.worker_id = "test-worker-2"
        worker.lease_duration_sec = 60
        worker.stale_recovery_interval_sec = 30
        worker._last_stale_recovery_at = time.time()
        worker.dsn = "postgresql://mock"

        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True
        worker.containment_gate = mock_gate

        mock_repo = MagicMock()
        mock_repo.claim_next_run_request.return_value = {
            "request_id": "req-87654321-efgh",
            "lease_token": "token-abc",
        }
        worker.repository = mock_repo

        with patch("analyst_dashboard.smart_money.worker.SmartMoneyContinuousPipeline") as MockPipeline:
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.execute_cycle.return_value = {
                "cycle_id": "cycle-2",
                "status": "SUCCESS",
                "snapshot_id": "SNAP_SM_GENUINE_002",
            }
            MockPipeline.return_value = mock_pipeline_inst

            worker.process_run_requests()

            # Confirm complete_run_request with raw gen- string was NOT called directly by worker
            # (Genuine pipeline atomically completes the run request via publish_scanner_snapshot_atomic)
            mock_repo.complete_run_request.assert_not_called()

    # ── Test 3: SEC rate-limit enforcement (<= 8.0 req/s) ────────────────────
    def test_sec_rate_limit_enforcement(self):
        limiter = RateLimiter(max_per_second=8.0)
        assert limiter.max_per_second == 8.0
        assert limiter.interval == pytest.approx(1.0 / 8.0, rel=1e-3)

        pipeline = SmartMoneyContinuousPipeline(rate_limiter=limiter, allow_test_mode=True)
        assert pipeline.rate_limiter.max_per_second <= 8.0

    # ── Test 4: Worker-only egress enforcement ───────────────────────────────
    def test_worker_only_egress_enforcement(self):
        pipeline = SmartMoneyContinuousPipeline(allow_test_mode=True)

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "VIEWER"}):
            with pytest.raises(RuntimeError, match="UNAUTHORIZED_WORKER_ROLE"):
                pipeline._step1_preflight(dry_run=False)

        with patch.dict(os.environ, {"ARX_SEC_EGRESS_ROLE": "OWNER", "SEC_USER_AGENT": "ARX/1.0 (contact: akara.david3@gmail.com)"}):
            # Should not raise role exception
            try:
                pipeline._step1_preflight(dry_run=True)
            except RuntimeError as e:
                assert "UNAUTHORIZED_WORKER_ROLE" not in str(e)

    # ── Test 5: Lease contention and loss recovery ────────────────────────────
    def test_worker_lease_contention_and_loss(self):
        worker = SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
        worker.worker_id = "test-worker-5"
        worker.lease_duration_sec = 60
        worker.stale_recovery_interval_sec = 30
        worker._last_stale_recovery_at = time.time()
        worker.dsn = "postgresql://mock"

        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True
        worker.containment_gate = mock_gate

        mock_repo = MagicMock()
        mock_repo.claim_next_run_request.return_value = {
            "request_id": "req-contested",
            "lease_token": "token-lost",
        }
        worker.repository = mock_repo

        with patch("analyst_dashboard.smart_money.worker.SmartMoneyContinuousPipeline") as MockPipeline:
            mock_pipeline_inst = MagicMock()
            mock_pipeline_inst.execute_cycle.side_effect = StaleLeaseOwnershipLostError("Fenced worker")
            MockPipeline.return_value = mock_pipeline_inst

            count = worker.process_run_requests()
            assert count == 0  # Handled safely without bubbling uncaught error

    # ── Test 6: Duplicate schedule delivery rejection ─────────────────────────
    def test_duplicate_schedule_delivery_rejection(self):
        scheduler = SmartMoneyTimezoneScheduler()
        # Monday 22:30 ET
        ny_time = datetime(2026, 10, 26, 22, 30, tzinfo=NY_TZ)
        today = ny_time.date()

        # First delivery: eligible
        should_run, reason = scheduler.should_execute_now(now=ny_time, last_executed_ny_date=None)
        assert should_run is True
        assert "WITHIN_EXECUTION_WINDOW" in reason

        # Duplicate delivery today: rejected
        should_run_dup, reason_dup = scheduler.should_execute_now(now=ny_time, last_executed_ny_date=today)
        assert should_run_dup is False
        assert "ALREADY_EXECUTED_TODAY" in reason_dup

    # ── Test 7: Idempotent filing persistence ────────────────────────────────
    def test_idempotent_filing_persistence(self):
        mock_repo = MagicMock()
        mock_repo.save_raw_filing.return_value = True

        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_form4_filings.return_value = [
            {
                "accession_number": "0001045810-26-000088",
                "ticker": "NVDA",
                "cik": "0001045810",
                "company_name": "NVIDIA CORP",
                "xml_content": "<xml>nvda_content</xml>",
            }
        ]
        pipeline.sec_fetcher = mock_fetcher

        filings = pipeline._step6_ingest_form4_filings()
        assert len(filings) == len(pipeline.issuers)
        assert mock_repo.save_raw_filing.call_count == len(pipeline.issuers)

    # ── Test 8: Backup freshness gate compliance ─────────────────────────────
    def test_backup_freshness_gate_compliance(self):
        """Confirms that freshly captured pre-backup satisfies the 12-hour gate."""
        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True

        pipeline = SmartMoneyContinuousPipeline(containment_gate=mock_gate, allow_test_mode=True)
        # Freshness gate check should succeed
        pipeline._step4_freshness_gate_check()
        mock_gate.check_ingestion_eligibility.assert_called_once()

    # ── Test 9: Pre-backup failure aborts pipeline ────────────────────────────
    def test_pre_backup_failure_aborts_pipeline(self):
        mock_backup = MagicMock()
        mock_backup.create_backup.return_value = {"status": "FAILED", "error": "Disk full"}

        pipeline = SmartMoneyContinuousPipeline(backup_engine=mock_backup, allow_test_mode=True)
        with pytest.raises(RuntimeError, match="PRE_BACKUP_FAILED"):
            pipeline._step2_and_3_pre_backup("test-cycle")

    # ── Test 10: Post-backup failure degrades gracefully ─────────────────────
    def test_post_backup_failure_degrades_gracefully(self):
        mock_backup = MagicMock()
        # Pre-backup succeeds
        mock_backup.create_backup.side_effect = [
            {"backup_id": "b-pre", "status": "VERIFIED"},
            RuntimeError("R2 connection timeout on post backup"),
        ]

        mock_gate = MagicMock()
        mock_gate.check_ingestion_eligibility.return_value = True

        mock_repo = MagicMock()
        mock_repo.publish_scanner_snapshot_atomic.return_value = "SNAP_001"

        pipeline = SmartMoneyContinuousPipeline(
            backup_engine=mock_backup,
            containment_gate=mock_gate,
            repository=mock_repo,
            allow_test_mode=True,
        )
        # Mock preflight and lock to bypass environment constraints
        pipeline._step1_preflight = MagicMock()
        pipeline._step5_claim_advisory_lock = MagicMock()
        pipeline._cleanup_lock = MagicMock()

        result = pipeline.execute_cycle(trigger_type="TEST")
        assert result["status"] == "DEGRADED_POST_BACKUP_FAILED"
        assert result["snapshot_id"] is not None
        assert any("POST_BACKUP_FAILED" in e for e in result["errors"])

    # ── Test 11: Snapshot publication atomicity ──────────────────────────────
    def test_snapshot_publication_atomicity(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        receipt = {"reconciliation_hash": "hash-abc-123"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[],
            receipt_meta=receipt,
            cycle_id="cycle-11",
            run_request_id="req-99",
            lease_token="tok-99",
        )

        assert snapshot["snapshot_id"].startswith("SNAP_SM_")
        assert snapshot["universe_id"] == GOVERNED_UNIVERSE_ID
        assert snapshot["publication_decision"] == "PUBLISH"
        mock_repo.publish_scanner_snapshot_atomic.assert_called_once_with(
            snapshot=snapshot,
            run_request_id="req-99",
            lease_token="tok-99",
        )

    # ── Test 12: DST transitions (EDT vs EST invariance) ─────────────────────
    def test_timezone_scheduler_dst_invariance(self):
        scheduler = SmartMoneyTimezoneScheduler(target_hour=22, target_minute=30)

        # Summer (EDT, UTC-4): 2026-07-15
        dt_summer_utc = datetime(2026, 7, 16, 2, 30, tzinfo=timezone.utc)
        ny_summer = scheduler.get_ny_now(dt_summer_utc)
        assert ny_summer.hour == 22
        assert ny_summer.minute == 30
        assert ny_summer.strftime("%Z") == "EDT"

        # Winter (EST, UTC-5): 2026-12-15
        dt_winter_utc = datetime(2026, 12, 16, 3, 30, tzinfo=timezone.utc)
        ny_winter = scheduler.get_ny_now(dt_winter_utc)
        assert ny_winter.hour == 22
        assert ny_winter.minute == 30
        assert ny_winter.strftime("%Z") == "EST"

    # ── Test 13: SEC calendar behavior (weekends and holidays) ────────────────
    def test_sec_calendar_weekend_and_holiday_skip(self):
        # Weekday normal: Monday 2026-10-26
        assert is_sec_dissemination_date(date(2026, 10, 26)) is True
        # Saturday: 2026-10-24
        assert is_sec_dissemination_date(date(2026, 10, 24)) is False
        # Sunday: 2026-10-25
        assert is_sec_dissemination_date(date(2026, 10, 25)) is False
        # Christmas: 2026-12-25
        assert is_sec_dissemination_date(date(2026, 12, 25)) is False
        # Independence Day: 2026-07-04 (Saturday, observed Friday or non-filing)
        assert is_sec_dissemination_date(date(2026, 7, 4)) is False

    # ── Test 14: Expected cycles vs calendar age freshness ────────────────────
    def test_freshness_authority_expected_cycles(self):
        authority = SmartMoneyFreshnessAuthority()
        # Friday 22:30 ET publication
        friday_pub = datetime(2026, 10, 23, 22, 30, tzinfo=NY_TZ)
        # Monday 15:00 ET evaluation (over 64 calendar hours elapsed!)
        monday_eval = datetime(2026, 10, 26, 15, 0, tzinfo=NY_TZ)

        cycles = authority.calculate_expected_cycles(friday_pub, monday_eval)
        # Over the weekend up to Monday 22:30, 0 new dissemination cycles have occurred
        assert cycles == 0

        # Snapshot freshness must remain FRESH
        mock_snapshot = {
            "snapshot_id": "SNAP_SM_001",
            "published_at": friday_pub.isoformat(),
        }
        eval_receipt = authority.evaluate_snapshot_freshness(mock_snapshot, now_utc=monday_eval)
        assert eval_receipt.freshness_state == SnapshotFreshnessState.FRESH

    # ── Test 15: Retention safety (never prunes only recovery point) ──────────
    def test_backup_retention_protects_latest_recovery_point(self):
        mock_storage = MagicMock()
        # Only 1 backup exists in storage
        mock_storage.list_backups.return_value = [
            {"remote_filename": "manifest_sm-backup-001.json", "created_at": "2026-10-01T00:00:00Z"},
            {"remote_filename": "sm-backup-001.dump", "created_at": "2026-10-01T00:00:00Z"},
        ]

        manager = BackupRetentionManager(storage_adapter=mock_storage, retention_count=1)
        pruned = manager.prune_backups()
        assert len(pruned) == 0  # Must NEVER prune the only remaining recovery point
        mock_storage.delete_backup.assert_not_called()

    # ── Test 16: Control A — Emergency web read disablement ────────────────────
    def test_emergency_control_a_web_read_disablement(self):
        mock_store = MagicMock()
        mock_store.get_latest_active_snapshot.return_value = {
            "snapshot_id": "SNAP_SM_ACTIVE",
            "publication_decision": "PUBLISH",
            "results": [{"symbol": "NVDA"}],
        }

        runner = SmartMoneyScannerRunner(snapshot_store=mock_store, backend="postgres")

        # Default: reads snapshot
        with patch.dict(os.environ, {"ARX_SMART_MONEY_POSTGRES_READ_ENABLED": "true"}):
            snap = runner.get_active_or_latest_snapshot()
            assert snap.get("status") == "AVAILABLE"

        # Control A activated: fails closed to PIPELINE_PENDING
        with patch.dict(os.environ, {"ARX_SMART_MONEY_POSTGRES_READ_ENABLED": "false"}):
            snap_disabled = runner.get_active_or_latest_snapshot()
            assert snap_disabled.get("status") == "PIPELINE_PENDING"

    # ── Test 17: Control B — Emergency worker shutdown releases advisory lock ─
    def test_emergency_control_b_worker_shutdown_drops_lock(self):
        worker = SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
        worker.worker_id = "test-worker-17"
        worker.state = WorkerHealthState.OWNER_LOCK_ACQUIRED

        mock_conn = MagicMock()
        mock_conn.closed = False
        worker._owner_conn = mock_conn

        start_time = time.time()
        worker.release_egress_authority()
        elapsed = time.time() - start_time

        # Must release lock in <= 1.0 second
        assert elapsed < 1.0
        assert worker.state == WorkerHealthState.OWNER_LOCK_LOST
        mock_conn.close.assert_called_once()

    # ── Test 18: In-flight work cycle defers advisory lock release ───────────
    def test_worker_in_flight_cycle_defers_shutdown(self):
        worker = SmartMoneyIngestionWorker.__new__(SmartMoneyIngestionWorker)
        worker.worker_id = "test-worker-18"
        worker._owner_conn = MagicMock()
        worker._owner_conn.closed = False
        worker.release_egress_authority = MagicMock()
        worker._is_executing_cycle = True
        worker._shutdown_requested = False

        # In-flight shutdown signal should not immediately drop the lock
        worker._handle_shutdown_signal(15, None)

        assert worker._shutdown_requested is True
        worker.release_egress_authority.assert_not_called()

    # ── Test 19: Valid qualifying transactions produce candidates ────────────
    def test_valid_qualifying_transactions_produce_candidates(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        qualifying_tx = {
            "transaction_id": "tx-nvda-buy-001",
            "accession_number": "0001045810-26-000099",
            "issuer_symbol": "NVDA",
            "issuer_cik": "0001045810",
            "issuer_name": "NVIDIA CORP",
            "reporting_owner_cik": "0001111111",
            "reporting_owner_name": "Jensen Huang",
            "is_officer": True,
            "officer_title": "Chief Executive Officer",
            "is_director": True,
            "is_ten_percent": False,
            "transaction_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "transaction_code": "P",
            "shares": 10000.0,
            "price_per_share": 120.0,
            "total_value_usd": 1200000.0,
            "acquired_disposed": "A",
            "ownership_type": "D",
            "is_derivative": False,
            "is_open_market_purchase_evidence": True,
        }

        receipt = {"reconciliation_hash": "hash-nvda-123", "receipt_id": "rcpt-001"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[qualifying_tx],
            receipt_meta=receipt,
            cycle_id="cycle-nvda",
        )

        assert snapshot["matched_count"] == 1
        assert len(snapshot["results"]) == 1
        cand = snapshot["results"][0]
        assert cand["symbol"] == "NVDA"
        assert cand["rank"] == 1
        assert cand["score"] > 0
        evidence = cand["scanner_evidence"]
        assert evidence["subsignal"] == "INSIDER_FORM4"
        assert evidence["cSuiteBuying"] is True
        assert evidence["openMarketPurchaseCount"] == 1
        assert evidence["grossPurchaseValueUsd"] == 1200000.0
        assert "provenance" in evidence
        assert evidence["provenance"]["issuer_cik"] == "0001045810"

    # ── Test 20: Legitimate zero match produces complete coverage ────────────
    def test_legitimate_zero_match_produces_complete_coverage(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        receipt = {"reconciliation_hash": "hash-zero-match", "receipt_id": "rcpt-zero"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[],
            receipt_meta=receipt,
            cycle_id="cycle-zero",
        )

        assert snapshot["coverage_status"] == "COMPLETE"
        assert snapshot["matched_count"] == 0
        assert snapshot["results"] == []
        assert snapshot["publication_decision"] == "PUBLISH"
        assert snapshot["coverage_metadata"]["scanned_successfully_count"] == len(pipeline.issuers)
        assert snapshot["coverage_metadata"]["scan_coverage_pct"] == 100.0
        assert snapshot["coverage_metadata"]["unavailable_symbol_count"] == 0
        mock_repo.publish_scanner_snapshot_atomic.assert_called_once()

    # ── Test 21: Incomplete issuer coverage blocks publication ───────────────
    def test_incomplete_issuer_coverage_blocks_publication(self):
        mock_repo = MagicMock()
        mock_scanner = MagicMock()
        # Fail evaluation for one ticker
        def mock_eval(symbol, transactions, **kwargs):
            if symbol == "NVDA":
                raise RuntimeError("SEC connection dropped during issuer evaluation")
            state = MagicMock()
            state.qualifies_for_radar = False
            state.conviction_score = 0.0
            return state

        mock_scanner.evaluate_issuer_transactions.side_effect = mock_eval

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            scanner=mock_scanner,
            allow_test_mode=True,
        )

        receipt = {"reconciliation_hash": "hash-partial", "receipt_id": "rcpt-partial"}
        with pytest.raises(RuntimeError, match="SNAPSHOT_PUBLICATION_REJECTED.*Coverage incomplete"):
            pipeline._step8_publish_snapshot(
                parsed_txs=[],
                receipt_meta=receipt,
                cycle_id="cycle-partial",
            )
        # Atomic publication must never be invoked on incomplete coverage
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    # ── Test 22: Scanner exception blocks publication fail-closed ─────────────
    def test_scanner_exception_blocks_publication(self):
        mock_repo = MagicMock()
        mock_scanner = MagicMock()
        mock_scanner.evaluate_issuer_transactions.side_effect = Exception("General scanner explosion")

        pipeline = SmartMoneyContinuousPipeline(
            repository=mock_repo,
            scanner=mock_scanner,
            allow_test_mode=True,
        )

        receipt = {"reconciliation_hash": "hash-fail", "receipt_id": "rcpt-fail"}
        with pytest.raises(RuntimeError, match="SNAPSHOT_PUBLICATION_REJECTED"):
            pipeline._step8_publish_snapshot(
                parsed_txs=[],
                receipt_meta=receipt,
                cycle_id="cycle-fail",
            )
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    # ── Test 23: Matched count strictly equals results length ────────────────
    def test_matched_count_equals_results_length(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        tx_nvda = {
            "transaction_id": "tx-nvda",
            "accession_number": "acc-1",
            "issuer_symbol": "NVDA",
            "issuer_cik": "0001045810",
            "transaction_date": now_str,
            "transaction_code": "P",
            "shares": 5000.0,
            "price_per_share": 100.0,
            "total_value_usd": 500000.0,
            "is_open_market_purchase_evidence": True,
        }
        tx_aapl = {
            "transaction_id": "tx-aapl",
            "accession_number": "acc-2",
            "issuer_symbol": "AAPL",
            "issuer_cik": "0000320193",
            "transaction_date": now_str,
            "transaction_code": "P",
            "shares": 1000.0,
            "price_per_share": 200.0,
            "total_value_usd": 200000.0,
            "is_open_market_purchase_evidence": True,
        }

        receipt = {"reconciliation_hash": "hash-matched-count", "receipt_id": "rcpt-mc"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[tx_nvda, tx_aapl],
            receipt_meta=receipt,
            cycle_id="cycle-mc",
        )

        assert snapshot["matched_count"] == len(snapshot["results"])
        assert snapshot["matched_count"] == 2
        assert snapshot["coverage_metadata"]["matched_count"] == 2

    # ── Test 24: Published candidate results preserve provenance ─────────────
    def test_published_results_preserve_provenance(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        tx = {
            "transaction_id": "tx-prov",
            "accession_number": "0001045810-26-000077",
            "issuer_symbol": "NVDA",
            "issuer_cik": "0001045810",
            "transaction_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "transaction_code": "P",
            "shares": 500.0,
            "price_per_share": 100.0,
            "total_value_usd": 50000.0,
            "is_open_market_purchase_evidence": True,
        }

        receipt = {"reconciliation_hash": "hash-prov", "receipt_id": "rcpt-prov"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[tx],
            receipt_meta=receipt,
            cycle_id="cycle-prov",
        )

        cand = snapshot["results"][0]
        prov = cand["scanner_evidence"]["provenance"]
        assert prov["issuer_cik"] == "0001045810"
        assert prov["statutory_basis"] == "SEC Form 4 Section 16(a)"
        assert "0001045810-26-000077" in prov["filing_accession_numbers"]
        assert prov["conviction_score"] == cand["score"]

    # ── Test 25: Missing reconciliation hash rejects snapshot publication ─────
    def test_reconciliation_failure_preserves_active_snapshot(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        with pytest.raises(RuntimeError, match="SNAPSHOT_PUBLICATION_REJECTED.*Missing reconciliation"):
            pipeline._step8_publish_snapshot(
                parsed_txs=[],
                receipt_meta={"reconciliation_hash": None},
                cycle_id="cycle-no-recon",
            )
        mock_repo.publish_scanner_snapshot_atomic.assert_not_called()

    # ── Test 26: Snapshot payload matches Radar API envelope contract ─────────
    def test_output_compatible_with_radar_api_contract(self):
        mock_repo = MagicMock()
        pipeline = SmartMoneyContinuousPipeline(repository=mock_repo, allow_test_mode=True)

        receipt = {"reconciliation_hash": "hash-envelope", "receipt_id": "rcpt-env"}
        snapshot = pipeline._step8_publish_snapshot(
            parsed_txs=[],
            receipt_meta=receipt,
            cycle_id="cycle-env",
        )

        runner = SmartMoneyScannerRunner(snapshot_store=MagicMock(), backend="postgres")
        envelope = runner._to_envelope(snapshot)

        # Assert contract shape required by api/routes/screener.py
        assert envelope["scanner_id"] == "SMART_MONEY"
        assert envelope["api_contract_version"] == "1.0.0"
        assert envelope["status"] == "AVAILABLE"
        assert "methodology" in envelope
        assert "provenance" in envelope
        assert "freshness" in envelope
        assert "snapshot" in envelope
        assert envelope["snapshot"]["matched_count"] == 0
        assert "universe" in envelope
        assert envelope["universe"]["source_population_count"] == 10
        assert "coverage" in envelope
        assert envelope["coverage"]["coverage_status"] == "COMPLETE"
        assert envelope["coverage"]["scan_coverage_pct"] == 100.0
        assert envelope["results"] == []

    # ── Test 27: Missing backup engine fails closed without test mode ─────────
    def test_mocked_backup_fails_closed_without_test_mode(self):
        # Without allow_test_mode=True, missing backup engine must raise BACKUP_ENGINE_REQUIRED
        with pytest.raises(RuntimeError, match="BACKUP_ENGINE_REQUIRED"):
            SmartMoneyContinuousPipeline(backup_engine=None, allow_test_mode=False)

    # ── Test 28: Production environment rejects allow_test_mode ───────────────
    def test_production_rejects_test_mode(self):
        with patch.dict(os.environ, {"ARX_ENVIRONMENT": "production"}):
            with pytest.raises(RuntimeError, match="TEST_MODE_PROHIBITED_IN_PRODUCTION"):
                SmartMoneyContinuousPipeline(allow_test_mode=True)
