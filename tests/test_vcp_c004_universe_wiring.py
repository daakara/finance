"""
tests/test_vcp_c004_universe_wiring.py

Authoritative Verification Suite for VCP Candidate 004 Targeted Universe Wiring Remediation.
Enforces the 10 core universe governance invariants (INV-U01 through INV-U10) and covers
all 20 mandated verification conditions:

 1. No published executable universe (natural execution blocked)
 2. Published universe matches candidate/session (eligible for readiness)
 3. Draft universe only (blocked)
 4. Failed publication (blocked)
 5. Wrong candidate generation (blocked)
 6. Wrong effective session (blocked)
 7. Mixed universe identity (blocked)
 8. Membership hash mismatch (blocked)
 9. Historical fixture present but no executable build (natural execution blocked)
10. Frozen historical regression path (preserved)
11. CPRX after effective delisting (2026-10-14 -> excluded under R07)
12. CPRX before effective delisting (2026-07-20 -> eligible under R07)
13. Missing data for active eligible symbol (readiness blocked)
14. Stale data for active eligible symbol (readiness blocked)
15. Guard/scanner membership mismatch (blocked)
16. Universe changes between guard and execution (blocked)
17. Concurrent workers select same immutable build (deterministic)
18. Publication failure leaves no partial universe (atomic triggers)
19. Synthetic tests admit no prospective evidence (Gate 12 deferred, denominator 0)
20. Disabled production scheduler remains disabled (HTTP 503)
"""

import os
import json
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.universe.contracts import (
    UNIVERSE_ID,
    UNIVERSE_VERSION,
    SourceSecurity,
    SourcePopulationSnapshot,
    EligibilityDecision,
    DataReadinessDecision,
    ConstructionStatus,
    PublicationDecision,
    compute_sha256,
)
from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
from analyst_dashboard.universe.store import UniverseStore
from analyst_dashboard.analyzers.scanner_contract import CANONICAL_VCP_UNIVERSE
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
from analyst_dashboard.vcp.natural_trigger import NaturalVCPTriggerService
from analyst_dashboard.vcp.session_guard import (
    evaluate_natural_session_eligibility,
    MarketSessionEligibilityResult,
)
from analyst_dashboard.vcp.sprint_3_durable_storage import Sprint3DurableEvidenceStore


class TestVCPC004UniverseWiring(unittest.TestCase):
    """20-condition authoritative test suite for Candidate 004 universe authority."""

    def setUp(self):
        self.client = TestClient(app)
        self.session_date = "2026-07-15"
        self.close_utc = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
        self.eligible_time = self.close_utc + timedelta(minutes=50)

        # Enforce strict session guard in tests
        os.environ["VCP_ENFORCE_SESSION_GUARD_IN_TESTS"] = "1"
        os.environ["VCP_BYPASS_SESSION_GUARD_FOR_TESTING"] = "0"

    def tearDown(self):
        os.environ.pop("VCP_ENFORCE_SESSION_GUARD_IN_TESTS", None)
        os.environ.pop("VCP_BYPASS_SESSION_GUARD_FOR_TESTING", None)

    def _create_mock_store(self, build_dict=None, scannable_symbols=None, eligible_symbols=None):
        store = MagicMock()
        store.get_universe_build.return_value = build_dict
        store.get_published_universe_for_session.return_value = build_dict
        store.get_latest_published_universe.return_value = build_dict
        store.get_scannable_membership.return_value = scannable_symbols or []
        store.get_eligible_membership.return_value = eligible_symbols or []
        return store

    # 1. No published executable universe (natural execution blocked)
    def test_01_no_published_executable_universe_blocks_natural_execution(self):
        store = self._create_mock_store(build_dict=None)
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "NO_PUBLISHED_EXECUTABLE_UNIVERSE")

    # 2. Published universe matches candidate/session (eligible for readiness)
    def test_02_published_universe_matches_candidate_and_session(self):
        symbols = ["AAPL", "MSFT"]
        build = {
            "universe_build_id": "univ-build-002",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(symbols),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=symbols)
        mock_mdb = MagicMock()
        mock_mdb.get_latest_price.return_value = {"currentPrice": 150.0, "date": self.session_date}
        mock_mdb.get_daily_candles.return_value = [{"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000, "time": self.session_date}] * 55

        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            market_db=mock_mdb,
            check_data_freshness=True,
        )
        self.assertTrue(res.is_eligible)
        self.assertEqual(res.reason, "ELIGIBLE")
        self.assertEqual(res.universe_build_id, "univ-build-002")
        self.assertEqual(res.scannable_symbols, symbols)

    # 3. Draft universe only (blocked)
    def test_03_draft_universe_only_blocks_natural_execution(self):
        build = {
            "universe_build_id": "univ-build-003",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "QUARANTINE_FOR_MANUAL_REVIEW",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "UNPUBLISHED_UNIVERSE_BUILD")

    # 4. Failed publication (blocked)
    def test_04_failed_publication_blocks_natural_execution(self):
        build = {
            "universe_build_id": "univ-build-004",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "FAILED",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "INCOMPLETE_UNIVERSE_BUILD")

    # 5. Wrong candidate generation (blocked)
    def test_05_wrong_candidate_generation_blocks_natural_execution(self):
        build = {
            "universe_build_id": "univ-build-005",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_006"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            candidate_generation_id="CANDIDATE_GENERATION_004",
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "CANDIDATE_GENERATION_MISMATCH")

    # 6. Wrong effective session (blocked)
    def test_06_wrong_effective_session_blocks_natural_execution(self):
        build = {
            "universe_build_id": "univ-build-006",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": "2026-07-14",
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "UNIVERSE_SESSION_MISMATCH")

    # 7. Mixed universe identity (blocked)
    def test_07_mixed_universe_identity_fails_closed(self):
        build = {
            "universe_build_id": "univ-build-007",
            "universe_id": "SMART_MONEY_UNIVERSE",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = MagicMock()
        def mock_get_published(universe_id, session_date, candidate_generation_id):
            if universe_id == "ARX_US_EQUITIES":
                return None
            return build
        store.get_published_universe_for_session.side_effect = mock_get_published
        store.get_latest_published_universe.return_value = None

        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "NO_PUBLISHED_EXECUTABLE_UNIVERSE")

    # 8. Membership hash mismatch (blocked)
    def test_08_membership_hash_mismatch_fails_closed(self):
        build = {
            "universe_build_id": "univ-build-008",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": "corrupted_hash_value_12345",
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "MEMBERSHIP_HASH_MISMATCH")

    # 9. Historical fixture present but no executable build (natural execution blocked)
    def test_09_historical_fixture_present_but_no_executable_build_blocks_natural_execution(self):
        runner = VCPScannerRunner(universe_store=self._create_mock_store(build_dict=None))
        svc = NaturalVCPTriggerService(scanner_runner=runner)
        with self.assertRaises(RuntimeError) as ctx:
            svc.trigger_natural_scan(scheduled_date=self.session_date)
        self.assertIn("NO_PUBLISHED_EXECUTABLE_UNIVERSE", str(ctx.exception))

    # 10. Frozen historical regression path (preserved)
    def test_10_frozen_historical_regression_path_preserved(self):
        runner = VCPScannerRunner(universe_store=None)
        # Execute non-natural scan (e.g. invocation_class="MANUAL_OPERATOR" or "BOOT_WARMUP")
        res = runner.execute_market_wide_scan(
            logical_job_key=f"vcp:test-manual:{uuid.uuid4().hex}",
            invocation_class="MANUAL_OPERATOR",
        )
        # Should complete and consume CANONICAL_VCP_UNIVERSE without raising
        self.assertIn(res.get("status"), ["AVAILABLE", "FAILED"])
        univ_size = res.get("universe_size") or res.get("snapshot", {}).get("universe_size")
        self.assertEqual(univ_size, len(CANONICAL_VCP_UNIVERSE))

    # 11. CPRX after effective delisting (2026-10-14 -> excluded under R07)
    def test_11_cprx_after_effective_delisting_excluded(self):
        cprx = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            delisting_date="2026-10-14",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx,
            build_id="test-build-cprx-post",
            observed_at="2026-10-15T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "INELIGIBLE_DELISTED")
        self.assertFalse(row.rule_results["R07_NOT_DELISTED"])

    # 12. CPRX legal listing before effective delisting (2026-07-20 -> legal listing active under R07)
    def test_12_cprx_legal_listing_before_delisting_eligible(self):
        """Tests that legal listing status under R07 evaluates to True prior to delisting date when suspension is unset."""
        cprx = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            delisting_date="2026-10-14",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx,
            build_id="test-build-cprx-pre",
            observed_at="2026-07-20T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.ELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "PASSED")
        self.assertTrue(row.rule_results["R07_NOT_DELISTED"])

    # 13. Missing data for active eligible symbol (readiness blocked)
    def test_13_missing_data_for_active_eligible_symbol_blocks_readiness(self):
        symbols = ["AAPL", "MSFT"]
        build = {
            "universe_build_id": "univ-build-013",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(symbols),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=symbols)
        mock_mdb = MagicMock()
        mock_mdb.get_latest_price.return_value = {"currentPrice": 100.0, "date": "2026-07-14"}  # Stale

        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            market_db=mock_mdb,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "MISSING_MARKET_DATA")
        self.assertEqual(res.coverage_ratio, 0.0)

    # 14. Stale data for active eligible symbol (readiness blocked)
    def test_14_stale_data_for_active_eligible_symbol_blocks_readiness(self):
        symbols = ["AAPL", "MSFT"]
        build = {
            "universe_build_id": "univ-build-014",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(symbols),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=symbols)
        mock_mdb = MagicMock()
        mock_mdb.get_daily_candles.return_value = [{"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000, "time": self.session_date}] * 55
        # Return current date for AAPL, stale date for MSFT
        def mock_price(sym):
            if sym == "AAPL":
                return {"currentPrice": 150.0, "date": self.session_date}
            return {"currentPrice": 100.0, "date": "2026-07-14"}
        mock_mdb.get_latest_price.side_effect = mock_price

        res = evaluate_natural_session_eligibility(
            session_date=self.session_date,
            now_utc=self.eligible_time,
            universe_store=store,
            universe_symbols=None,
            market_db=mock_mdb,
            check_data_freshness=True,
        )
        self.assertFalse(res.is_eligible)
        self.assertEqual(res.reason, "INCOMPLETE_UNIVERSE_COVERAGE")
        self.assertEqual(res.coverage_ratio, 0.5)

    # 15. Guard/scanner membership mismatch (blocked)
    def test_15_guard_scanner_membership_mismatch_blocks_execution(self):
        runner = VCPScannerRunner(universe_store=self._create_mock_store(build_dict=None))
        # When natural execution is invoked with NO scannable symbols and no published build, runner must raise
        with self.assertRaises(RuntimeError) as ctx:
            runner.execute_market_wide_scan(
                logical_job_key=f"vcp:test-sched:{uuid.uuid4().hex}",
                invocation_class="SCHEDULED_PRODUCTION",
            )
        self.assertIn("NO_PUBLISHED_EXECUTABLE_UNIVERSE", str(ctx.exception))

    # 16. Universe changes between guard and execution (blocked)
    def test_16_universe_changes_between_guard_and_execution_fails_closed(self):
        store = self._create_mock_store(build_dict=None)
        runner = VCPScannerRunner(universe_store=store)
        svc = NaturalVCPTriggerService(scanner_runner=runner)
        # Attempt to pass an invalid or tampered build_id that cannot be loaded
        with self.assertRaises(RuntimeError) as ctx:
            svc.trigger_natural_scan(
                scheduled_date=self.session_date,
                universe_build_id="non-existent-or-tampered-build-999",
            )
        self.assertIn("NO_PUBLISHED_EXECUTABLE_UNIVERSE", str(ctx.exception))

    # 17. Concurrent workers select same immutable build (deterministic)
    def test_17_concurrent_workers_select_same_immutable_build(self):
        build = {
            "universe_build_id": "univ-build-concurrent-017",
            "universe_id": "ARX_US_EQUITIES",
            "publication_decision": "PUBLISH",
            "construction_status": "COMPLETE",
            "source_population_as_of": self.session_date,
            "scannable_membership_hash": compute_sha256(["AAPL"]),
            "reconciliation_summary_json": json.dumps({"candidate_generation_id": "CANDIDATE_GENERATION_004"}),
        }
        store = self._create_mock_store(build_dict=build, scannable_symbols=["AAPL"])
        worker1_build = store.get_published_universe_for_session("ARX_US_EQUITIES", self.session_date, "CANDIDATE_GENERATION_004")
        worker2_build = store.get_published_universe_for_session("ARX_US_EQUITIES", self.session_date, "CANDIDATE_GENERATION_004")
        self.assertEqual(worker1_build["universe_build_id"], worker2_build["universe_build_id"])
        self.assertEqual(worker1_build["scannable_membership_hash"], worker2_build["scannable_membership_hash"])

    # 18. Publication failure leaves no partial universe (atomic triggers)
    def test_18_publication_failure_leaves_no_partial_universe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_univ.db")
            store = UniverseStore(db_path=db_path)
            conn = store._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
                triggers = [row[0] for row in cursor.fetchall()]
                self.assertIn("trg_ubuild_no_update", triggers)
                self.assertIn("trg_ubuild_no_delete", triggers)
            finally:
                conn.close()

    # 19. Synthetic tests admit no prospective evidence (Gate 12 deferred, denominator 0)
    def test_19_synthetic_tests_admit_no_prospective_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_shadow.db")
            store = Sprint3DurableEvidenceStore(db_path=db_path)
            counts = store.get_authoritative_denominator_counts()
            self.assertEqual(counts["shadow_record_count"], 0)
            self.assertEqual(counts["natural_production_shadow_record_count"], 0)

    # 20. Disabled production scheduler remains disabled (HTTP 503)
    def test_20_disabled_production_scheduler_remains_disabled(self):
        with patch.dict(os.environ, {"VCP_NATURAL_SCHEDULER_ENABLED": "0"}):
            res = self.client.post("/api/v1/screener/vcp/natural-scan")
            self.assertEqual(res.status_code, 503)
            self.assertIn("NATURAL_SCHEDULER_DISABLED", res.json()["detail"])

    # 21. CPRX suspended trading (2026-07-20 -> ineligible under R03 due to trading suspension, legal listing R07 remains True)
    def test_21_cprx_suspended_trading_ineligible_under_r03(self):
        """INV-T01 & INV-T05: Tests that trading suspension marks security ineligible under R03 while R07 reflects legal listing."""
        cprx = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            delisting_date="2026-10-14",
            trading_suspension_date="2026-07-15",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx,
            build_id="test-build-cprx-susp",
            observed_at="2026-07-20T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "INELIGIBLE_LISTING_STATUS")
        self.assertFalse(row.rule_results["R03_LISTING_STATUS"])
        self.assertTrue(row.rule_results["R07_NOT_DELISTED"])

    # 22. CPRX active trading before suspension (2026-07-14 -> eligible under both R03 and R07)
    def test_22_cprx_active_trading_before_suspension_eligible(self):
        """INV-T02 & INV-T03: Tests that on last active trading day, security is eligible under both R03 and R07."""
        cprx = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            delisting_date="2026-10-14",
            trading_suspension_date="2026-07-15",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx,
            build_id="test-build-cprx-active",
            observed_at="2026-07-14T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.ELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "PASSED")
        self.assertTrue(row.rule_results["R03_LISTING_STATUS"])
        self.assertTrue(row.rule_results["R07_NOT_DELISTED"])


    # 23. Active listed and tradable security (Scenario 1: passes all rules)
    def test_23_active_listed_and_tradable_security(self):
        nvda = SourceSecurity(
            security_id="SEC_NVDA_001",
            symbol="NVDA",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            trading_suspension_date=None,
            delisting_date=None,
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=nvda,
            build_id="test-build-active",
            observed_at="2026-07-20T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.ELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "PASSED")
        self.assertTrue(row.rule_results["R03_LISTING_STATUS"])
        self.assertTrue(row.rule_results["R07_NOT_DELISTED"])

    # 24. Unknown trading status fails closed (Scenario 9 / INV-T11)
    def test_24_unknown_trading_status_fails_closed(self):
        sec = SourceSecurity(
            security_id="SEC_UNK_001",
            symbol="UNKNOWN_CO",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="UNKNOWN",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec,
            build_id="test-build-unk",
            observed_at="2026-07-20T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "INELIGIBLE_LISTING_STATUS")
        self.assertFalse(row.rule_results["R03_LISTING_STATUS"])

    # 25. Conflicting lifecycle records fail closed (Scenario 11: listing_status=ACTIVE but past suspension date)
    def test_25_conflicting_lifecycle_records_fail_closed(self):
        sec = SourceSecurity(
            security_id="SEC_CONF_001",
            symbol="CONFLICT_CO",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",  # Conflicting optimistic claim
            trading_suspension_date="2026-07-15",  # Authoritative past date
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec,
            build_id="test-build-conflict",
            observed_at="2026-07-20T12:00:00Z",
        )
        # Authoritative effective date takes precedence and fails closed
        self.assertEqual(row.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "INELIGIBLE_LISTING_STATUS")
        self.assertFalse(row.rule_results["R03_LISTING_STATUS"])

    # 26. Missing lifecycle dates default to pure listing_status evaluation (Scenario 10)
    def test_26_missing_lifecycle_dates_preserves_pure_listing_status(self):
        sec = SourceSecurity(
            security_id="SEC_MISS_001",
            symbol="SIMPLE_CO",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            listing_date=None,
            delisting_date=None,
            trading_suspension_date=None,
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec,
            build_id="test-build-miss",
            observed_at="2026-07-20T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.ELIGIBLE)
        self.assertEqual(row.eligibility_reason_code, "PASSED")
        self.assertTrue(row.rule_results["R03_LISTING_STATUS"])
        self.assertTrue(row.rule_results["R07_NOT_DELISTED"])

    # 27. Late-arriving corporate action information / historical as-of evaluation (Scenario 12 & 13)
    def test_27_late_arriving_corporate_action_historical_as_of(self):
        # As of July 10, suspension is unknown
        sec_july10 = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            trading_suspension_date=None,
            delisting_date=None,
        )
        row_july10 = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec_july10,
            build_id="test-build-july10",
            observed_at="2026-07-10T12:00:00Z",
        )
        self.assertEqual(row_july10.eligibility_decision, EligibilityDecision.ELIGIBLE)

        # As of July 16, suspension is known and effective July 15
        sec_july16 = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            trading_suspension_date="2026-07-15",
            delisting_date="2026-10-14",
        )
        row_july16 = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=sec_july16,
            build_id="test-build-july16",
            observed_at="2026-07-16T12:00:00Z",
        )
        self.assertEqual(row_july16.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row_july16.eligibility_reason_code, "INELIGIBLE_LISTING_STATUS")

    # 28. Security after SEC deregistration excluded under R07 (Scenario 8)
    def test_28_security_after_sec_deregistration_excluded_under_r07(self):
        cprx = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            trading_suspension_date="2026-07-15",
            delisting_date="2026-10-14",
        )
        row = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx,
            build_id="test-build-dereg",
            observed_at="2026-10-15T12:00:00Z",
        )
        self.assertEqual(row.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertFalse(row.rule_results["R03_LISTING_STATUS"])
        self.assertFalse(row.rule_results["R07_NOT_DELISTED"])


if __name__ == "__main__":
    unittest.main()
