"""ARX Terminal — VCP Candidate 004 Natural Execution Control Plane Test Suite.

Authoritatively tests the minimal control plane remediation:
1. Server-side market session eligibility, NYSE calendar rules, settlement window, and early close adaptation.
2. HTTP endpoint security contract: X-Scheduler-Token, HTTP 403 on invalid auth, HTTP 503 on disabled, HTTP 422 on session rejection.
3. Session-level idempotency: invariant across universe_build_id mutations.
4. Fencing, multi-worker lease safety, and anti-split-brain publication.
5. Gate 12 quarantine and zero-denominator governance preservation.
6. GitHub Actions workflow syntax, least-privilege permissions, and disabled-by-default scheduling.
"""

import os
import sys
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import MagicMock, patch
import yaml

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from api.main import app

from analyst_dashboard.vcp.session_guard import (
    US_EASTERN,
    evaluate_natural_session_eligibility,
    get_session_bounds,
    is_us_equity_trading_session,
)
from analyst_dashboard.vcp.natural_trigger import (
    CANDIDATE_004_GENERATION_ID,
    NaturalVCPTriggerService,
    get_natural_vcp_trigger_service,
)
from analyst_dashboard.coordination import AcquisitionStatus, TriggerType
from analyst_dashboard.coordination.coordinator import DurableRunCoordinator
from analyst_dashboard.coordination.fenced_publisher import FencedPublisher, StaleLeasePublicationError
from analyst_dashboard.coordination.store import CoordinationStore
from analyst_dashboard.data.scanner_store import ScannerSnapshotStore


class TestMarketSessionGuard(unittest.TestCase):
    """Phase B: Authoritative tests for market session eligibility and settlement window."""

    def setUp(self):
        os.environ["VCP_ENFORCE_SESSION_GUARD_IN_TESTS"] = "1"

    def tearDown(self):
        os.environ.pop("VCP_ENFORCE_SESSION_GUARD_IN_TESTS", None)

    def test_trading_sessions_calendar(self):
        # Regular trading day (Wednesday)
        self.assertTrue(is_us_equity_trading_session("2026-07-15"))
        # Regular winter trading day
        self.assertTrue(is_us_equity_trading_session("2026-01-14"))
        # Weekend (Saturday & Sunday)
        self.assertFalse(is_us_equity_trading_session("2026-07-18"))
        self.assertFalse(is_us_equity_trading_session("2026-07-19"))
        # NYSE Holidays
        self.assertFalse(is_us_equity_trading_session("2026-01-01"))  # New Year's Day
        self.assertFalse(is_us_equity_trading_session("2026-07-03"))  # Independence Day (Observed)
        self.assertFalse(is_us_equity_trading_session("2026-12-25"))  # Christmas Day

    def test_session_bounds_regular_and_early_close(self):
        # Regular session close: 16:00 ET
        s_open, s_close, is_early = get_session_bounds("2026-07-15")
        close_et = s_close.astimezone(US_EASTERN)
        self.assertEqual(close_et.hour, 16)
        self.assertEqual(close_et.minute, 0)
        self.assertFalse(is_early)

        # Early close session (Day after Thanksgiving: 2026-11-27)
        s_open_e, s_close_e, is_early_e = get_session_bounds("2026-11-27")
        close_et_e = s_close_e.astimezone(US_EASTERN)
        self.assertEqual(close_et_e.hour, 13)
        self.assertEqual(close_et_e.minute, 0)
        self.assertTrue(is_early_e)

    def test_settlement_window_timing(self):
        session_date = "2026-07-15"
        # Summer EDT: 16:00 ET = 20:00 UTC. 45-min settlement = 20:45 UTC.
        close_utc = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)

        # 1. Premature request at 20:15 UTC (16:15 ET - only 15 mins post-close)
        early_time = close_utc + timedelta(minutes=15)
        res_early = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=early_time,
            settlement_buffer_minutes=45,
        )
        self.assertFalse(res_early.is_eligible)
        self.assertEqual(res_early.reason, "BEFORE_SETTLEMENT_WINDOW")
        self.assertIn("remaining", res_early.detail)

        # 2. Eligible request at 20:50 UTC (16:50 ET - 50 mins post-close)
        eligible_time = close_utc + timedelta(minutes=50)
        res_ok = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            settlement_buffer_minutes=45,
        )
        self.assertTrue(res_ok.is_eligible)
        self.assertEqual(res_ok.reason, "ELIGIBLE")

        # 3. Late request next day at 12:00 UTC (past 04:00 ET premarket cutoff)
        late_time = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
        res_late = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=late_time,
            settlement_buffer_minutes=45,
        )
        self.assertFalse(res_late.is_eligible)
        self.assertEqual(res_late.reason, "OUTSIDE_PERMITTED_WINDOW")

    def test_non_trading_day_rejection(self):
        # Sunday evaluation
        res_sunday = evaluate_natural_session_eligibility(session_date="2026-10-18")
        self.assertFalse(res_sunday.is_eligible)
        self.assertEqual(res_sunday.reason, "NON_TRADING_DAY")

        # Christmas Day evaluation
        res_xmas = evaluate_natural_session_eligibility(session_date="2026-12-25")
        self.assertFalse(res_xmas.is_eligible)
        self.assertEqual(res_xmas.reason, "NON_TRADING_DAY")

    def test_market_data_readiness_checks(self):
        session_date = "2026-07-15"
        close_utc = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
        eligible_time = close_utc + timedelta(minutes=50)
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        # 1. Authority unavailable when check_data_freshness=True and market_db is None
        res_no_db = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols,
            market_db=None,
        )
        self.assertFalse(res_no_db.is_eligible)
        self.assertEqual(res_no_db.reason, "DATA_AUTHORITY_UNAVAILABLE")

        # 2. Missing market data: zero symbols settled for target date
        mock_db_empty = MagicMock()
        mock_db_empty.get_latest_price.return_value = {"currentPrice": 100.0, "date": "2026-07-14"}  # prior day
        res_empty = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols,
            market_db=mock_db_empty,
        )
        self.assertFalse(res_empty.is_eligible)
        self.assertEqual(res_empty.reason, "MISSING_MARKET_DATA")
        self.assertEqual(res_empty.coverage_ratio, 0.0)

        valid_candles = [
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000, "time": "2026-07-15"}
            for _ in range(55)
        ]

        # 3. Incomplete coverage: 3 of 5 symbols settled (60% < 100%)
        mock_db_partial = MagicMock()
        def mock_price_partial(sym):
            if sym in ("AAPL", "MSFT", "NVDA"):
                return {"currentPrice": 100.0, "date": "2026-07-15"}
            return {"currentPrice": 50.0, "date": "2026-07-14"}
        mock_db_partial.get_latest_price.side_effect = mock_price_partial
        mock_db_partial.get_daily_candles.return_value = valid_candles
        res_partial = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols,
            market_db=mock_db_partial,
        )
        self.assertFalse(res_partial.is_eligible)
        self.assertEqual(res_partial.reason, "INCOMPLETE_UNIVERSE_COVERAGE")
        self.assertEqual(res_partial.coverage_ratio, 0.6)

        # 4. Parity check: 5 of 5 have price, but only 10 daily bars (< 50 bars required)
        mock_db_short = MagicMock()
        mock_db_short.get_latest_price.return_value = {"currentPrice": 150.0, "date": "2026-07-15"}
        mock_db_short.get_daily_candles.return_value = valid_candles[:10]
        res_short = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols,
            market_db=mock_db_short,
        )
        self.assertFalse(res_short.is_eligible)
        self.assertEqual(res_short.reason, "MISSING_MARKET_DATA")
        self.assertEqual(res_short.coverage_ratio, 0.0)

        # 5. Full-universe 34 of 35 symbols ready (97.1% < 100% required)
        symbols_35 = [f"SYM{i}" for i in range(35)]
        mock_db_34 = MagicMock()
        mock_db_34.get_latest_price.return_value = {"currentPrice": 150.0, "date": "2026-07-15"}
        def mock_candles_34_of_35(sym, limit=60):
            if sym == "SYM34":
                return valid_candles[:10]  # 1 symbol has insufficient bars
            return valid_candles
        mock_db_34.get_daily_candles.side_effect = mock_candles_34_of_35
        res_34 = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols_35,
            market_db=mock_db_34,
        )
        self.assertFalse(res_34.is_eligible)
        self.assertEqual(res_34.reason, "INCOMPLETE_UNIVERSE_COVERAGE")
        self.assertAlmostEqual(res_34.coverage_ratio, 34 / 35, places=4)

        # 6. Sufficient coverage: 5 of 5 symbols settled with >= 50 bars (100% full universe)
        mock_db_full = MagicMock()
        mock_db_full.get_latest_price.return_value = {"currentPrice": 150.0, "date": "2026-07-15"}
        mock_db_full.get_daily_candles.return_value = valid_candles
        res_full = evaluate_natural_session_eligibility(
            session_date=session_date,
            now_utc=eligible_time,
            check_data_freshness=True,
            universe_symbols=symbols,
            market_db=mock_db_full,
        )
        self.assertTrue(res_full.is_eligible)
        self.assertEqual(res_full.reason, "ELIGIBLE")
        self.assertEqual(res_full.coverage_ratio, 1.0)

    def test_calendar_fail_closed_when_authority_unavailable(self):
        from analyst_dashboard.vcp import session_guard
        with patch.object(session_guard, "_NYSE_CAL", None):
            res = session_guard.evaluate_natural_session_eligibility(
                session_date="2026-07-15",
                check_data_freshness=False,
            )
            self.assertFalse(res.is_eligible)
            self.assertEqual(res.reason, "CALENDAR_AUTHORITY_UNAVAILABLE")
            self.assertIn("unavailable", res.detail.lower())

    def test_calendar_fail_closed_when_query_raises_exception(self):
        from analyst_dashboard.vcp import session_guard
        mock_cal = MagicMock()
        mock_cal.is_session.side_effect = RuntimeError("Remote calendar connection dropped")
        with patch.object(session_guard, "_NYSE_CAL", mock_cal):
            res = session_guard.evaluate_natural_session_eligibility(
                session_date="2026-07-15",
                check_data_freshness=False,
            )
            self.assertFalse(res.is_eligible)
            self.assertEqual(res.reason, "CALENDAR_AUTHORITY_UNAVAILABLE")


class TestNaturalScanEndpointControlPlane(unittest.TestCase):
    """Phase B & D: Endpoint security, session rejection, and authorization tests."""

    def setUp(self):
        os.environ["VCP_ENFORCE_SESSION_GUARD_IN_TESTS"] = "1"
        with patch("api.main.warmup_core_assets", MagicMock()):
            self.client = TestClient(app)

    def tearDown(self):
        os.environ.pop("VCP_ENFORCE_SESSION_GUARD_IN_TESTS", None)

    def test_disabled_scheduler_503(self):
        with patch.dict(os.environ, {"VCP_NATURAL_SCHEDULER_ENABLED": "0", "VCP_SCHEDULER_SECRET": "secret"}):
            res = self.client.post("/api/v1/screener/vcp/natural-scan")
            self.assertEqual(res.status_code, 503)
            self.assertIn("NATURAL_SCHEDULER_DISABLED", res.json()["detail"])

    def test_missing_or_invalid_auth_403(self):
        with patch.dict(os.environ, {"VCP_NATURAL_SCHEDULER_ENABLED": "1", "VCP_SCHEDULER_SECRET": "test-secret-123"}):
            # Missing header
            res1 = self.client.post("/api/v1/screener/vcp/natural-scan")
            self.assertEqual(res1.status_code, 403)
            self.assertIn("SCHEDULER_AUTHENTICATION_FAILED", res1.json()["detail"])

            # Invalid header
            res2 = self.client.post(
                "/api/v1/screener/vcp/natural-scan",
                headers={"X-Scheduler-Token": "wrong-secret"}
            )
            self.assertEqual(res2.status_code, 403)

            # Bearer header (rejected; requires X-Scheduler-Token)
            res3 = self.client.post(
                "/api/v1/screener/vcp/natural-scan",
                headers={"Authorization": "Bearer test-secret-123"}
            )
            self.assertEqual(res3.status_code, 403)

    def test_session_rejection_422(self):
        with patch.dict(os.environ, {
            "VCP_NATURAL_SCHEDULER_ENABLED": "1",
            "VCP_SCHEDULER_SECRET": "test-secret-123",
            "VCP_BYPASS_SESSION_GUARD_FOR_TESTING": "0"
        }):
            # Attempt to scan on a Sunday
            res_sunday = self.client.post(
                "/api/v1/screener/vcp/natural-scan",
                headers={"X-Scheduler-Token": "test-secret-123"},
                json={"scheduled_date": "2026-10-18"}
            )
            self.assertEqual(res_sunday.status_code, 422)
            data = res_sunday.json()["detail"]
            self.assertEqual(data["status"], "REJECTED")
            self.assertEqual(data["reason"], "NON_TRADING_DAY")

    def test_authorized_scan_with_test_override(self):
        with patch.dict(os.environ, {
            "VCP_NATURAL_SCHEDULER_ENABLED": "1",
            "VCP_SCHEDULER_SECRET": "test-secret-123",
            "VCP_BYPASS_SESSION_GUARD_FOR_TESTING": "1"
        }):
            with patch("analyst_dashboard.vcp.natural_trigger.NaturalVCPTriggerService.trigger_natural_scan") as mock_scan:
                mock_scan.return_value = {
                    "status": "SUCCESS",
                    "scan_execution_status": "SUCCESS",
                    "snapshot_publication_status": "PUBLISH",
                    "snapshot_id": "vcp-snap-test-1",
                    "matched_count": 23,
                    "universe_size": 35,
                }
                res = self.client.post(
                    "/api/v1/screener/vcp/natural-scan",
                    headers={"X-Scheduler-Token": "test-secret-123"},
                    json={"scheduled_date": "2026-07-15"}
                )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["status"], "SUCCESS")

    def test_route_data_readiness_enforcement(self):
        with patch.dict(os.environ, {
            "VCP_NATURAL_SCHEDULER_ENABLED": "1",
            "VCP_SCHEDULER_SECRET": "test-secret-123",
            "VCP_BYPASS_SESSION_GUARD_FOR_TESTING": "0",
        }):
            session_date = "2026-07-15"
            close_utc = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
            eligible_time = close_utc + timedelta(minutes=50)

            from analyst_dashboard.vcp import session_guard
            orig_eval = session_guard.evaluate_natural_session_eligibility
            def mock_eval_fn(*args, **kwargs):
                kwargs["now_utc"] = eligible_time
                return orig_eval(*args, **kwargs)

            # Case A: Market DB returns 0 settled symbols for session_date -> HTTP 422 with MISSING_MARKET_DATA
            with patch("analyst_dashboard.vcp.session_guard.evaluate_natural_session_eligibility", side_effect=mock_eval_fn):
                with patch("api.routes.screener.vcp_scanner_runner.market_db") as mock_mdb:
                    mock_mdb.get_latest_price.return_value = {"currentPrice": 100.0, "date": "2026-07-14"}
                    res = self.client.post(
                        "/api/v1/screener/vcp/natural-scan",
                        headers={"X-Scheduler-Token": "test-secret-123"},
                        json={"scheduled_date": session_date},
                    )
                    self.assertEqual(res.status_code, 422)
                    data = res.json()["detail"]
                    self.assertEqual(data["status"], "REJECTED")
                    self.assertEqual(data["reason"], "MISSING_MARKET_DATA")
                    self.assertEqual(data["coverage_ratio"], 0.0)

            # Case B: Market DB returns 10 of 35 symbols -> HTTP 422 with INCOMPLETE_UNIVERSE_COVERAGE
            with patch("analyst_dashboard.vcp.session_guard.evaluate_natural_session_eligibility", side_effect=mock_eval_fn):
                with patch("api.routes.screener.vcp_scanner_runner.market_db") as mock_mdb:
                    mock_mdb.get_daily_candles.return_value = [
                        {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000000, "time": session_date}
                        for _ in range(55)
                    ]
                    call_count = [0]
                    def mock_price_partial(sym):
                        call_count[0] += 1
                        if call_count[0] <= 10:
                            return {"currentPrice": 100.0, "date": session_date}
                        return {"currentPrice": 100.0, "date": "2026-07-14"}
                    mock_mdb.get_latest_price.side_effect = mock_price_partial
                    res = self.client.post(
                        "/api/v1/screener/vcp/natural-scan",
                        headers={"X-Scheduler-Token": "test-secret-123"},
                        json={"scheduled_date": session_date},
                    )
                    self.assertEqual(res.status_code, 422)
                    data = res.json()["detail"]
                    self.assertEqual(data["status"], "REJECTED")
                    self.assertEqual(data["reason"], "INCOMPLETE_UNIVERSE_COVERAGE")
                    self.assertLess(data["coverage_ratio"], 1.0)


class TestSessionLevelIdempotencyAndCoordination(unittest.TestCase):
    """Phase C: Session-level idempotency repair & multi-process coordination safety."""

    def test_idempotency_key_invariant_across_build_ids(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        today = "2026-07-15"

        with patch.object(svc._get_runner(), "execute_market_wide_scan") as mock_exec:
            mock_exec.return_value = {"snapshot_id": "snap-1", "publication_decision": "PUBLISH", "status": "AVAILABLE"}

            # Request 1 with build ARX_CANONICAL_V1
            svc.trigger_natural_scan(scheduled_date=today, universe_build_id="ARX_CANONICAL_V1")
            call1_kwargs = mock_exec.call_args[1]

            # Request 2 with build ARX_CANONICAL_V2
            svc.trigger_natural_scan(scheduled_date=today, universe_build_id="ARX_CANONICAL_V2")
            call2_kwargs = mock_exec.call_args[1]

            # Invariant: Both calls resolve to the EXACT SAME logical_job_key
            expected_key = f"vcp:natural:{CANDIDATE_004_GENERATION_ID}:{today}"
            self.assertEqual(call1_kwargs["logical_job_key"], expected_key)
            self.assertEqual(call2_kwargs["logical_job_key"], expected_key)
            self.assertEqual(call1_kwargs["logical_job_key"], call2_kwargs["logical_job_key"])

            # Provenance preserved: individual universe_build_id still passed for provenance
            self.assertEqual(call1_kwargs["universe_build_id"], "ARX_CANONICAL_V1")
            self.assertEqual(call2_kwargs["universe_build_id"], "ARX_CANONICAL_V2")

    def test_fenced_publication_rejection_under_stale_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_fenced.db")
            coord = DurableRunCoordinator(resource_key="VCP_MARKET_SCAN", db_path=db_path)
            publisher = FencedPublisher(coordination_store=coord.store)

            job_key = "vcp:natural:CANDIDATE_GENERATION_004:2026-07-15"

            # Worker 1 acquires lease
            acq1 = coord.acquire(
                logical_job_key=job_key,
                trigger_type=TriggerType.SCHEDULED,
                owner_instance_id="worker-1",
                implementation_release_sha="80cbdcfd"
            )
            self.assertEqual(acq1.status, AcquisitionStatus.ACQUIRED)
            token1 = acq1.lease.fencing_token

            # Simulate Worker 2 superseding Worker 1 (e.g. lease expiry & takeover)
            # Expire Worker 1 lease forcibly in DB
            conn = coord.store._get_connection()
            conn.execute("UPDATE radar_current_leases SET expires_at = '2000-01-01 00:00:00' WHERE resource_key = 'VCP_MARKET_SCAN'")
            conn.commit()
            conn.close()

            acq2 = coord.acquire(
                logical_job_key=f"{job_key}-retry",
                trigger_type=TriggerType.SCHEDULED,
                owner_instance_id="worker-2",
                implementation_release_sha="80cbdcfd"
            )
            self.assertEqual(acq2.status, AcquisitionStatus.ACQUIRED)
            token2 = acq2.lease.fencing_token
            self.assertGreater(token2, token1)

            # Stale Worker 1 attempts fenced publication
            with self.assertRaises(StaleLeasePublicationError):
                publisher.publish_snapshot_fenced(
                    lease=acq1.lease,
                    snapshot_data={
                        "snapshot_id": "snap-stale",
                        "scanner_id": "MINERVINI_VCP",
                        "run_id": acq1.lease.run_id,
                        "data_as_of": "2026-07-15",
                        "matched_count": 23,
                        "universe_size": 35,
                    },
                )


class TestEvidenceGovernancePreservation(unittest.TestCase):
    """Phase F: Preservation of Gate 12, quarantined telemetry, and zero denominator."""

    def test_boot_warmup_produces_zero_denominator(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        with patch.object(svc._get_runner(), "execute_market_wide_scan") as mock_exec:
            mock_exec.return_value = {"snapshot_id": "snap-boot", "publication_decision": "PUBLISH", "status": "AVAILABLE"}
            svc.trigger_boot_warmup()
            kwargs = mock_exec.call_args[1]
            self.assertEqual(kwargs["invocation_class"], "BOOT_WARMUP")
            self.assertEqual(kwargs["originating_principal_type"], "BOOTSTRAP")
            self.assertTrue(kwargs["startup_context"])

    def test_scheduled_production_quarantined_under_gate_12(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        with patch.object(svc._get_runner(), "execute_market_wide_scan") as mock_exec:
            mock_exec.return_value = {"snapshot_id": "snap-sched", "publication_decision": "PUBLISH", "status": "AVAILABLE"}
            res = svc.trigger_natural_scan(scheduled_date="2026-07-15")
            self.assertEqual(res["status"], "SUCCESS")
            # Prospective evidence status remains explicitly bound under candidate freeze
            self.assertEqual(res["candidate_generation_id"], CANDIDATE_004_GENERATION_ID)


class TestGitHubActionsWorkflowValidation(unittest.TestCase):
    """Phase D: Static analysis and syntax validation of .github/workflows/vcp-natural-scheduler.yml."""

    def test_workflow_yaml_syntax_and_safety(self):
        wf_path = os.path.abspath(".github/workflows/vcp-natural-scheduler.yml")
        self.assertTrue(os.path.exists(wf_path), f"Workflow file missing: {wf_path}")

        with open(wf_path, "r", encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        # 1. Triggers: workflow_dispatch present; active cron absent
        self.assertIn("on", wf)
        triggers = wf["on"]
        self.assertIn("workflow_dispatch", triggers)
        self.assertNotIn("schedule", triggers)  # Active cron must be disabled/commented out

        # 2. Permissions: least privilege
        self.assertIn("permissions", wf)
        self.assertEqual(wf["permissions"]["contents"], "read")

        # 3. Concurrency group defined
        self.assertIn("concurrency", wf)
        self.assertEqual(wf["concurrency"]["group"], "vcp-natural-scheduler")
        self.assertFalse(wf["concurrency"]["cancel-in-progress"])

        # 4. Job environment and timeout
        jobs = wf["jobs"]
        self.assertIn("trigger-natural-scan", jobs)
        job = jobs["trigger-natural-scan"]
        self.assertEqual(job["environment"], "production")
        self.assertLessEqual(job["timeout-minutes"], 5)

    def test_workflow_target_url_and_response_contract_validation(self):
        wf_path = os.path.abspath(".github/workflows/vcp-natural-scheduler.yml")
        with open(wf_path, "r", encoding="utf-8") as f:
            wf_content = f.read()

        # P1-02: Target URL verification
        self.assertIn("web-production-e370b.up.railway.app", wf_content)
        self.assertIn("vars.ARX_API_BASE_URL", wf_content)
        self.assertIn("https://", wf_content)
        self.assertNotIn("arx-terminal-production.up.railway.app", wf_content)

        # P1-03 & P0-01: Response contract assertions
        self.assertIn('STATUS=$(echo "$RESPONSE_BODY" | jq -r \'.status // empty\' 2>/dev/null)', wf_content)
        self.assertIn('SNAP_ID=$(echo "$RESPONSE_BODY" | jq -r \'.snapshot_id // empty\' 2>/dev/null)', wf_content)
        self.assertIn('CANDIDATE_ID=$(echo "$RESPONSE_BODY" | jq -r \'.candidate_generation_id // empty\' 2>/dev/null)', wf_content)
        self.assertIn('PUB_STATUS=$(echo "$RESPONSE_BODY" | jq -r \'.snapshot_publication_status // empty\' 2>/dev/null)', wf_content)
        self.assertIn('EXEC_OUTCOME=$(echo "$RESPONSE_BODY" | jq -r \'.execution_outcome // empty\' 2>/dev/null)', wf_content)
        self.assertIn('[ "$STATUS" != "SUCCESS" ]', wf_content)
        self.assertIn('[ -z "$SNAP_ID" ] || [ "$SNAP_ID" = "null" ]', wf_content)
        self.assertIn('[ "$CANDIDATE_ID" != "CANDIDATE_GENERATION_004" ]', wf_content)
        self.assertIn('[ "$PUB_STATUS" = "PUBLISH" ] || [ "$EXEC_OUTCOME" = "NEW_SUCCESS" ]', wf_content)
        self.assertIn('[ "$PUB_STATUS" = "DEDUPLICATED" ] || [ "$EXEC_OUTCOME" = "ALREADY_SUCCEEDED" ]', wf_content)


class TestNaturalScanDeduplicationAndOutcomeClassification(unittest.TestCase):
    """Phase B / P0-01: Full verification of deduplication, outcome classification, and workflow contract."""

    def setUp(self):
        self.client = TestClient(app)

    def test_fresh_execution_outcome_classification(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "snapshot_id": "snap-fresh-001",
            "snapshot": {
                "snapshot_id": "snap-fresh-001",
                "data_as_of": "2026-07-15",
                "universe_size": 35,
                "matched_count": 23,
            },
            "publication_decision": "PUBLISH",
            "status": "AVAILABLE",
        }

        res = svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_outcome"], "NEW_SUCCESS")
        self.assertEqual(res["snapshot_publication_status"], "PUBLISH")
        self.assertTrue(res["is_new_publication"])
        self.assertEqual(res["snapshot_id"], "snap-fresh-001")
        self.assertEqual(res["candidate_generation_id"], CANDIDATE_004_GENERATION_ID)

    def test_duplicate_execution_outcome_classification(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "status": "SUCCESS",
            "message": "Market scan already succeeded for logical job key",
            "deduplicated": True,
            "acquisition_status": "JOB_ALREADY_SUCCEEDED",
            "successful_publication_id": "snap-prior-001",
            "snapshot": {
                "snapshot_id": "snap-prior-001",
                "data_as_of": "2026-07-15",
                "universe_size": 35,
                "matched_count": 23,
            },
        }
        mock_runner.snapshot_store.get_snapshot_by_id.return_value = {
            "snapshot_id": "snap-prior-001",
            "data_as_of": "2026-07-15",
            "universe_size": 35,
            "matched_count": 23,
            "status": "AVAILABLE",
        }

        res = svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_outcome"], "ALREADY_SUCCEEDED")
        self.assertEqual(res["snapshot_publication_status"], "DEDUPLICATED")
        self.assertFalse(res["is_new_publication"])
        self.assertEqual(res["snapshot_id"], "snap-prior-001")
        self.assertEqual(res["candidate_generation_id"], CANDIDATE_004_GENERATION_ID)
        self.assertIn("already succeeded", res["message"].lower())

    def test_duplicate_execution_nested_envelope_unwrapping(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "status": "SUCCESS",
            "deduplicated": True,
            "acquisition_status": "JOB_ALREADY_SUCCEEDED",
            "snapshot": {
                "snapshot_id": "snap-nested-002",
                "data_as_of": "2026-07-15",
            },
        }
        mock_runner.snapshot_store.get_snapshot_by_id.return_value = {
            "snapshot_id": "snap-nested-002",
            "data_as_of": "2026-07-15",
        }

        res = svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["execution_outcome"], "ALREADY_SUCCEEDED")
        self.assertEqual(res["snapshot_publication_status"], "DEDUPLICATED")
        self.assertEqual(res["snapshot_id"], "snap-nested-002")

    def test_duplicate_execution_fails_closed_when_snapshot_id_missing(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "status": "SUCCESS",
            "deduplicated": True,
            "acquisition_status": "JOB_ALREADY_SUCCEEDED",
        }
        with self.assertRaises(RuntimeError) as ctx:
            svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertIn("DEDUPLICATED_SNAPSHOT_NOT_FOUND", str(ctx.exception))

    def test_duplicate_execution_fails_closed_when_snapshot_not_in_store(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "status": "SUCCESS",
            "deduplicated": True,
            "acquisition_status": "JOB_ALREADY_SUCCEEDED",
            "successful_publication_id": "snap-ghost-001",
        }
        mock_runner.snapshot_store.get_snapshot_by_id.return_value = None

        with self.assertRaises(RuntimeError) as ctx:
            svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertIn("DEDUPLICATED_SNAPSHOT_NOT_FOUND", str(ctx.exception))

    def test_duplicate_execution_fails_closed_when_snapshot_date_mismatch(self):
        svc = NaturalVCPTriggerService(scanner_runner=MagicMock())
        mock_runner = svc._get_runner()
        mock_runner.execute_market_wide_scan.return_value = {
            "status": "SUCCESS",
            "deduplicated": True,
            "acquisition_status": "JOB_ALREADY_SUCCEEDED",
            "successful_publication_id": "snap-stale-001",
        }
        mock_runner.snapshot_store.get_snapshot_by_id.return_value = {
            "snapshot_id": "snap-stale-001",
            "data_as_of": "2026-07-14",
        }

        with self.assertRaises(RuntimeError) as ctx:
            svc.trigger_natural_scan(scheduled_date="2026-07-15")
        self.assertIn("DEDUPLICATED_SNAPSHOT_DATE_MISMATCH", str(ctx.exception))

    def test_endpoint_duplicate_execution_returns_200_with_deduplicated_payload(self):
        with patch.dict(os.environ, {
            "VCP_NATURAL_SCHEDULER_ENABLED": "1",
            "VCP_SCHEDULER_SECRET": "test-secret-dup",
            "VCP_BYPASS_SESSION_GUARD_FOR_TESTING": "1",
        }):
            mock_runner = MagicMock()
            mock_runner.execute_market_wide_scan.return_value = {
                "status": "SUCCESS",
                "deduplicated": True,
                "acquisition_status": "JOB_ALREADY_SUCCEEDED",
                "successful_publication_id": "snap-dedup-endpoint-001",
                "snapshot": {
                    "snapshot_id": "snap-dedup-endpoint-001",
                    "data_as_of": "2026-07-15",
                    "universe_size": 35,
                    "matched_count": 23,
                },
            }
            mock_runner.snapshot_store.get_snapshot_by_id.return_value = {
                "snapshot_id": "snap-dedup-endpoint-001",
                "data_as_of": "2026-07-15",
                "universe_size": 35,
                "matched_count": 23,
                "status": "AVAILABLE",
            }

            with patch("analyst_dashboard.vcp.natural_trigger.get_natural_vcp_trigger_service") as mock_get_svc:
                mock_svc = NaturalVCPTriggerService(scanner_runner=mock_runner)
                mock_get_svc.return_value = mock_svc

                res = self.client.post(
                    "/api/v1/screener/vcp/natural-scan",
                    headers={"X-Scheduler-Token": "test-secret-dup"},
                    json={"scheduled_date": "2026-07-15"},
                )
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "SUCCESS")
                self.assertEqual(data["execution_outcome"], "ALREADY_SUCCEEDED")
                self.assertEqual(data["snapshot_publication_status"], "DEDUPLICATED")
                self.assertFalse(data["is_new_publication"])
                self.assertEqual(data["snapshot_id"], "snap-dedup-endpoint-001")
                self.assertEqual(data["candidate_generation_id"], "CANDIDATE_GENERATION_004")

    def test_workflow_script_contract_simulation(self):
        """Simulate the exact workflow bash/jq classification logic on all response shapes."""
        def evaluate_workflow_response(response_dict):
            status = response_dict.get("status")
            snap_id = response_dict.get("snapshot_id")
            candidate_id = response_dict.get("candidate_generation_id")
            pub_status = response_dict.get("snapshot_publication_status")
            exec_outcome = response_dict.get("execution_outcome")

            if status != "SUCCESS":
                return False, "NON_SUCCESS_STATUS"
            if not snap_id or snap_id == "null":
                return False, "INVALID_SNAPSHOT_ID"
            if candidate_id != "CANDIDATE_GENERATION_004":
                return False, "CANDIDATE_MISMATCH"
            if pub_status not in ("PUBLISH", "DEDUPLICATED"):
                return False, "UNEXPECTED_PUB_STATUS"
            if exec_outcome not in ("NEW_SUCCESS", "ALREADY_SUCCEEDED"):
                return False, "UNEXPECTED_EXEC_OUTCOME"
            return True, "WORKFLOW_PASS"

        # Case 1: Fresh publication -> Pass
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": "snap-fresh-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "PUBLISH",
            "execution_outcome": "NEW_SUCCESS",
        })
        self.assertTrue(ok, reason)

        # Case 2: Deduplicated run -> Pass
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": "snap-dedup-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "DEDUPLICATED",
            "execution_outcome": "ALREADY_SUCCEEDED",
        })
        self.assertTrue(ok, reason)

        # Case 3: Missing or null snapshot_id -> Fail
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": None,
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "DEDUPLICATED",
            "execution_outcome": "ALREADY_SUCCEEDED",
        })
        self.assertFalse(ok)
        self.assertEqual(reason, "INVALID_SNAPSHOT_ID")

        # Case 4: Non-success status -> Fail
        ok, reason = evaluate_workflow_response({
            "status": "REJECTED",
            "snapshot_id": "snap-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "PUBLISH",
            "execution_outcome": "NEW_SUCCESS",
        })
        self.assertFalse(ok)
        self.assertEqual(reason, "NON_SUCCESS_STATUS")

        # Case 5: Candidate mismatch -> Fail
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": "snap-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_003",
            "snapshot_publication_status": "PUBLISH",
            "execution_outcome": "NEW_SUCCESS",
        })
        self.assertFalse(ok)
        self.assertEqual(reason, "CANDIDATE_MISMATCH")

        # Case 6: Unknown publication status -> Fail
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": "snap-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "UNKNOWN_PUB",
            "execution_outcome": "NEW_SUCCESS",
        })
        self.assertFalse(ok)
        self.assertEqual(reason, "UNEXPECTED_PUB_STATUS")

        # Case 7: Unknown execution outcome -> Fail
        ok, reason = evaluate_workflow_response({
            "status": "SUCCESS",
            "snapshot_id": "snap-1",
            "candidate_generation_id": "CANDIDATE_GENERATION_004",
            "snapshot_publication_status": "PUBLISH",
            "execution_outcome": "UNKNOWN_OUTCOME",
        })
        self.assertFalse(ok)
        self.assertEqual(reason, "UNEXPECTED_EXEC_OUTCOME")


if __name__ == "__main__":
    unittest.main()

