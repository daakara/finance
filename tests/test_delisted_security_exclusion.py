"""
tests/test_delisted_security_exclusion.py

Authoritative acceptance test suite for Delisted-Security Exclusion Directive (Addendum to RADAR-ACT-006R).
Validates end-to-end exclusion of confirmed delisted securities across:
1. Authoritative exchange and corporate action records (CPRX primary regression fixture).
2. Effective-date-aware lifecycle classification (ACTIVE, SUSPENDED, DELISTED, ACQUIRED, UNRESOLVED, REINSTATED).
3. Canonical security-identity level exclusions in Universe Builder (Rule R03 & R07).
4. Cross-engine enforcement across VCP screening, Value/GARP, Smart Money, Radar rankings, and trade plans.
5. Historical record preservation and immutable prior snapshot safety.
6. Stale cached price and curated catalog immunity (stale data cannot reintroduce delisted assets).
7. Explicit exclusion reasons and lifecycle timestamps in API contracts.
8. Backend and frontend population count consistency.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi import Response
from fastapi.testclient import TestClient

from analyst_dashboard.universe.lifecycle import (
    SecurityLifecycleStatus,
    CorporateActionType,
    SecurityLifecycleRecord,
    SecurityLifecycleRegistry,
    get_security_lifecycle_registry,
)
from analyst_dashboard.universe.contracts import (
    SourceSecurity,
    EligibilityDecision,
    compute_sha256,
)
from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.scanner_runner import VCPScannerRunner
from api.routes.screener import (
    run_screener_get,
    derive_canonical_radar_categories,
    CONFIRMED_CORPORATE_ACTIONS,
    MASTER_CATALOG_CANDIDATES,
)
from api.routes.analytics import _build_tactical_setup, get_asset_analytics
from api.main import app


class TestDelistedSecurityExclusion(unittest.TestCase):
    """Authoritative test suite for Delisted Security Exclusion Directive."""

    def setUp(self):
        self.registry = get_security_lifecycle_registry()
        self.client = TestClient(app)

    def test_01_authoritative_corporate_action_records(self):
        """
        Requirement 1: Verify listing status against authoritative exchange and corporate action records.
        CPRX (Catalyst Pharmaceuticals, CIK 0001369568) must cite Form 25-NSE and Form 15-12B.
        """
        record = self.registry.get_latest_record_by_symbol("CPRX")
        self.assertIsNotNone(record, "CPRX must be registered in authoritative corporate action registry")
        self.assertEqual(record.symbol, "CPRX")
        self.assertEqual(record.canonical_id, "SEC_CPRX_001")
        self.assertEqual(record.lifecycle_status, SecurityLifecycleStatus.ACQUIRED)
        self.assertEqual(record.action_type, CorporateActionType.CASH_MERGER_ACQUISITION)
        self.assertEqual(record.effective_date, "2026-07-21")
        self.assertEqual(record.trading_suspension_date, "2026-07-15")
        self.assertIn("Angelini Pharma", record.acquirer)
        self.assertIn("$31.50", record.consideration)

        # Primary SEC filing citations
        forms = [f["form"] for f in record.filings]
        self.assertIn("25-NSE", forms, "Must cite Form 25-NSE (Notification of Removal from Listing on Nasdaq)")
        self.assertIn("15-12B", forms, "Must cite Form 15-12B (Termination of Registration)")

    def test_02_effective_date_aware_lifecycle_classification(self):
        """
        Requirement 2: Implement effective-date-aware lifecycle classification.
        - Prior to suspension (2026-07-14): ACTIVE, trading active, screening eligible.
        - Suspension window (2026-07-15 to 2026-07-20): SUSPENDED (temporary halt != permanent delisting).
        - On/after delisting (2026-07-21): DELISTED / ACQUIRED, ineligible.
        """
        # A. Prior to suspension (Active trading point-in-time)
        res_active = self.registry.evaluate_lifecycle("CPRX", as_of_date="2026-07-14")
        self.assertEqual(res_active.lifecycle_status, SecurityLifecycleStatus.ACTIVE)
        self.assertTrue(res_active.is_active_for_trading)
        self.assertTrue(res_active.is_screening_eligible)
        self.assertFalse(res_active.is_delisted)
        self.assertFalse(res_active.is_suspended)

        # B. During trading suspension (Halted pending corporate action)
        res_susp = self.registry.evaluate_lifecycle("CPRX", as_of_date="2026-07-18")
        self.assertEqual(res_susp.lifecycle_status, SecurityLifecycleStatus.SUSPENDED)
        self.assertFalse(res_susp.is_active_for_trading)
        self.assertFalse(res_susp.is_screening_eligible)
        self.assertTrue(res_susp.is_suspended)
        self.assertFalse(res_susp.is_delisted, "Temporary halt/suspension must NOT be classified as permanent delisting")
        self.assertIn("trading suspended", res_susp.exclusion_reason.lower())

        # C. On/after confirmed delisting
        res_delist = self.registry.evaluate_lifecycle("CPRX", as_of_date="2026-07-21")
        self.assertEqual(res_delist.lifecycle_status, SecurityLifecycleStatus.ACQUIRED)
        self.assertFalse(res_delist.is_active_for_trading)
        self.assertFalse(res_delist.is_screening_eligible)
        self.assertTrue(res_delist.is_delisted)
        self.assertFalse(res_delist.is_suspended)
        self.assertIn("delisted/acquired", res_delist.exclusion_reason.lower())

        # D. Current date (defaults to current delisted state)
        res_current = self.registry.evaluate_lifecycle("CPRX")
        self.assertTrue(res_current.is_delisted)
        self.assertFalse(res_current.is_screening_eligible)

    def test_03_reinstatement_and_canonical_id_isolation(self):
        """
        Requirement 2 & Acceptance Test: Ticker reuse and reinstatements respect canonical ID.
        """
        custom_reg = SecurityLifecycleRegistry([])

        # Register delisted company under Old CIK
        custom_reg.register(SecurityLifecycleRecord(
            canonical_id="SEC_XYZ_OLD",
            symbol="XYZ",
            lifecycle_status=SecurityLifecycleStatus.DELISTED,
            action_type=CorporateActionType.REGULATORY_DELISTING,
            effective_date="2026-01-01",
            notes="Old entity delisted.",
        ))

        # Check Old Entity evaluates as delisted
        old_eval = custom_reg.evaluate_lifecycle("XYZ", canonical_id="SEC_XYZ_OLD")
        self.assertTrue(old_eval.is_delisted)

        # Reinstated entity on 2026-09-01
        custom_reg.register(SecurityLifecycleRecord(
            canonical_id="SEC_XYZ_REINSTATED",
            symbol="XYZ",
            lifecycle_status=SecurityLifecycleStatus.REINSTATED,
            action_type=CorporateActionType.REINSTATEMENT,
            effective_date="2026-01-01",
            reinstatement_date="2026-09-01",
            notes="Reinstated after compliance cure.",
        ))

        # Before reinstatement date: Delisted
        pre_eval = custom_reg.evaluate_lifecycle("XYZ", canonical_id="SEC_XYZ_REINSTATED", as_of_date="2026-08-15")
        self.assertTrue(pre_eval.is_delisted)

        # On or after reinstatement date: Admitted
        post_eval = custom_reg.evaluate_lifecycle("XYZ", canonical_id="SEC_XYZ_REINSTATED", as_of_date="2026-09-02")
        self.assertEqual(post_eval.lifecycle_status, SecurityLifecycleStatus.REINSTATED)
        self.assertTrue(post_eval.is_active_for_trading)
        self.assertTrue(post_eval.is_screening_eligible)

    def test_04_universe_builder_canonical_identity_exclusion(self):
        """
        Requirement 3 & 6: Exclusions at canonical security-identity level in Universe Builder.
        Stale cached prices or curated catalog entries cannot override confirmed delisting.
        """
        # Even if an incoming feed marks CPRX as 'ACTIVE' with no delisting date attached:
        cprx_stale_feed = SourceSecurity(
            security_id="SEC_CPRX_001",
            symbol="CPRX",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",  # Stale claim
            delisting_date=None,  # Missing from feed
            trading_suspension_date=None,  # Missing from feed
        )

        # Evaluated as of today (2026-11-02): Registry MUST override stale feed
        row_current = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx_stale_feed,
            build_id="test-build-delisted-001",
            observed_at="2026-11-02T12:00:00Z",
        )
        self.assertEqual(row_current.eligibility_decision, EligibilityDecision.INELIGIBLE)
        self.assertEqual(row_current.eligibility_reason_code, "INELIGIBLE_DELISTED")
        self.assertFalse(row_current.rule_results["R07_NOT_DELISTED"])
        self.assertFalse(row_current.rule_results["R03_LISTING_STATUS"])

        # Evaluated historically before delisting (2026-07-14): Admitted
        row_hist = DeterministicUniverseBuilder.evaluate_security_eligibility(
            security=cprx_stale_feed,
            build_id="test-build-delisted-hist",
            observed_at="2026-07-14T12:00:00Z",
        )
        self.assertEqual(row_hist.eligibility_decision, EligibilityDecision.ELIGIBLE)
        self.assertEqual(row_hist.eligibility_reason_code, "PASSED")
        self.assertTrue(row_hist.rule_results["R07_NOT_DELISTED"])
        self.assertTrue(row_hist.rule_results["R03_LISTING_STATUS"])

    def test_05_vcp_screening_exclusion(self):
        """
        Requirement 4: Enforce eligibility across VCP screening.
        Delisted securities cannot be scanned or admitted into VCP snapshots.
        """
        runner = VCPScannerRunner()
        mock_mdb = MagicMock()
        mock_mdb.get_latest_price.return_value = {"currentPrice": 31.49, "date": "2026-07-14"}
        mock_mdb.get_daily_candles.return_value = [{"open": 31.0, "high": 31.5, "low": 30.8, "close": 31.49, "volume": 100000, "time": "2026-07-14"}] * 60
        runner.market_db = mock_mdb

        # Execute scan with CPRX in candidate universe override (unique operator_request_id per run)
        import uuid
        res = runner.execute_market_wide_scan(
            universe_override=["CPRX"],
            operator_request_id=f"test-vcp-delisted-cprx-{uuid.uuid4().hex[:8]}",
            scheduled_for="2026-11-02",
        )
        # CPRX must be classified under DELISTED_SECURITY in unavailable reasons
        cov = res.get("coverage", res.get("coverage_metadata", {}))
        unavail = cov.get("unavailable_reasons", res.get("unavailable_reasons", {}))
        self.assertIn("DELISTED_SECURITY", unavail)
        self.assertEqual(unavail["DELISTED_SECURITY"], 1)
        scanned_count = cov.get("scanned_successfully_count", res.get("scanned_successfully_count", 0))
        self.assertEqual(scanned_count, 0)
        candidates = res.get("results", res.get("qualified_candidates", []))
        self.assertEqual(len(candidates), 0)

    def test_06_value_garp_discovery_exclusion(self):
        """
        Requirement 4 & 5: Value / GARP discovery and catalog preservation.
        CPRX remains in KNOWN_GEMS_DATA for auditability, but evaluate_candidates suppresses active scoring.
        derive_canonical_radar_categories strictly excludes CPRX from VALUE_GARP, VCP, and SMART_MONEY.
        """
        screener = HiddenGemsScreener()
        self.assertIn("CPRX", screener.KNOWN_GEMS_DATA, "CPRX must be preserved in historical catalog for auditability")

        # Category classification must strictly return empty list for delisted security
        categories = derive_canonical_radar_categories(
            expert_model="Greenblatt Magic Formula",
            peg_ratio=0.78,
            roic_pct=34.2,
            gross_margin_pct=78.9,
            symbol="CPRX",
            vcp_matched_symbols={"CPRX"},  # Even if spoofed
            smart_money_matched_symbols={"CPRX"},  # Even if spoofed
        )
        self.assertEqual(categories, [], "Delisted CPRX must NEVER be assigned active discovery categories")

        # Filtered radar query for value_garp strictly excludes CPRX
        resp = Response()
        data_vg = run_screener_get(resp, filter_type="value_garp")
        symbols_vg = [c["symbol"] for c in data_vg.get("candidates", [])]
        self.assertNotIn("CPRX", symbols_vg, "CPRX must never appear in value_garp filter")

    def test_07_smart_money_discovery_exclusion(self):
        """
        Requirement 4: Delisted securities strictly excluded from Smart Money candidate discovery.
        """
        resp = Response()
        data_sm = run_screener_get(resp, filter_type="smart_money")
        symbols_sm = [c["symbol"] for c in data_sm.get("candidates", [])]
        self.assertNotIn("CPRX", symbols_sm, "CPRX must never appear in smart_money filter")

    def test_08_radar_rankings_and_trade_plan_actionability(self):
        """
        Requirement 4 & 7: Trade-plan actionability suppressed for delisted securities.
        All trade levels (optimalEntry, stopLoss, takeProfit) must be explicit None / null (NEVER 0.0).
        """
        setup = _build_tactical_setup("CPRX", "LONG_TERM", [], is_stale=False)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["executionStatus"], "DELISTED_SECURITY")
        self.assertEqual(setup["decisionState"], "INELIGIBLE_INACTIVE")
        self.assertFalse(setup["isActionable"])
        self.assertTrue(setup["isSuppressed"])
        self.assertIsNone(setup["entryPivot"])
        self.assertIsNone(setup["stopLoss"])
        self.assertIsNone(setup["target1"])
        self.assertIsNone(setup["target2"])
        self.assertEqual(setup["confluenceScore"], 0.0)

        # Single-asset custom tickers query on screener route
        resp = Response()
        data = run_screener_get(resp, custom_tickers="CPRX")
        cprx = data["candidates"][0]
        self.assertEqual(cprx["executionStatus"], "DELISTED_SECURITY")
        self.assertEqual(cprx["statusLabel"], "🛑 Delisted / Inactive")
        self.assertEqual(cprx["statusColor"], "rose")
        self.assertFalse(cprx["screeningEligible"])
        self.assertEqual(cprx["allowedActions"], [], "Delisted securities must have zero allowed actions")
        self.assertIsNone(cprx["stopLoss"])
        self.assertIsNone(cprx["optimalEntryMin"])
        self.assertIsNone(cprx["optimalEntryMax"])
        self.assertIsNone(cprx["takeProfit1"])
        self.assertIsNone(cprx["takeProfit2"])

    def test_09_api_contract_exclusion_reasons_and_timestamps(self):
        """
        Requirement 7: Return explicit exclusion reasons and lifecycle timestamps through API contracts.
        """
        # 1. Screener route returns corporateAction envelope with effective date and filings
        resp = Response()
        data = run_screener_get(resp, custom_tickers="CPRX")
        cprx = data["candidates"][0]
        self.assertIn("corporateAction", cprx)
        corp = cprx["corporateAction"]
        self.assertEqual(corp["status"], "DELISTED")
        self.assertEqual(corp["effective_date"], "2026-07-21")
        self.assertEqual(corp["trading_suspension_date"], "2026-07-15")
        self.assertIn("Angelini Pharma", corp["acquirer"])
        self.assertEqual(len(corp["filings"]), 2)

        # 2. Analytics asset route returns explicit 404 with corporate action and effective date detail
        res = self.client.get("/api/v1/analytics/CPRX")
        self.assertEqual(res.status_code, 404)
        detail = res.json()["detail"]
        self.assertIn("delisted", detail.lower())
        self.assertIn("Angelini Pharma", detail)

    def test_10_backend_frontend_count_consistency(self):
        """
        Requirement 5 (Frontend): Backend and frontend population count consistency.
        referenceUniverseCount = activeScreeningCount + delistedRecordCount + unverifiedRecordCount.
        """
        resp = Response()
        data = run_screener_get(resp, filter_type="all")
        pop = data.get("population", {})

        ref_count = pop.get("referenceUniverseCount", 0)
        active_count = pop.get("activeScreeningCount", 0)
        delisted_count = pop.get("delistedRecordCount", 0)
        unverified_count = pop.get("unverifiedRecordCount", 0)

        self.assertGreater(ref_count, 0)
        self.assertGreaterEqual(delisted_count, 1, "Delisted CPRX must be accounted for")
        self.assertEqual(ref_count, active_count + delisted_count + unverified_count)

        # Delisted securities must NOT inflate current opportunities count
        opp_count = pop.get("currentOpportunityCount", 0)
        candidates = data.get("candidates", [])
        eligible_cands = [c for c in candidates if c.get("screeningEligible") is True]
        self.assertEqual(opp_count, len(eligible_cands))
        self.assertFalse(any(c["symbol"] == "CPRX" for c in eligible_cands), "CPRX must never be counted as eligible opportunity")


if __name__ == "__main__":
    unittest.main()
