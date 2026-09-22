"""Regression and invariant test suite for F_12: Point-in-Time Fundamental Evidence.

Enforces:
1. Canonical PIT Invariant: availableFrom <= evaluation_timestamp
2. Acceptance timestamp temporal gap (period_end < filing_date <= acceptance_datetime)
3. Current-only records strictly barred from historical evaluation
4. Versioned factor snapshot progression without future leakage
5. Fail-closed decision resolution in DecisionHierarchyEngine when PIT fundamentals are absent
6. Static reference catalog isolation in HiddenGemsScreener
7. SEC EDGAR fetcher provenance extraction
8. Database schema migration idempotency
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from analyst_dashboard.data.market_db import MarketDatabase
from analyst_dashboard.governance.decision_context import (
    FundamentalEvidenceContract,
    PointInTimeStatus,
    EvidenceQualityState,
    is_pit_eligible,
    weakest_pit_status,
)
from analyst_dashboard.analyzers.decision_hierarchy import (
    DecisionHierarchyEngine,
    DecisionState,
)
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher


class TestPointInTimeFundamentals(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_market.db")
        self.db = MarketDatabase(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_pit_temporal_gap_acceptance_date(self):
        """Fundamental evidence becomes eligible strictly at available_from (acceptance_datetime),
        never at period_end or filing_date alone."""
        # Q4 2025: Period ended 2025-12-31, filed & accepted 2026-02-18T16:05:22Z
        snapshot = {
            "symbol": "AAPL",
            "as_of_date": "2025-12-31",
            "period_end": "2025-12-31",
            "filing_date": "2026-02-18",
            "acceptance_datetime": "2026-02-18T16:05:22Z",
            "available_from": "2026-02-18T16:05:22Z",
            "point_in_time_status": PointInTimeStatus.POINT_IN_TIME,
            "source": "sec_edgar",
            "currentPrice": 220.0,
            "priceChangePct24h": 1.5,
            "growthScore": 85,
            "qualityScore": 92,
            "valuationScore": 70,
            "momentumScore": 80,
            "tailRiskScore": 88,
            "compositeFactorScore": 85,
            "piotroskiFScore": 8,
            "verdict": "High Quality Compounder",
        }
        self.db.save_factor_snapshot("AAPL", snapshot)

        # 1. At quarter end (2025-12-31): Not yet filed or public -> None
        res_quarter_end = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2025-12-31T23:59:59Z")
        self.assertIsNone(res_quarter_end)

        # 2. Mid January (2026-01-15): Still unfiled -> None
        res_mid_jan = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-01-15T00:00:00Z")
        self.assertIsNone(res_mid_jan)

        # 3. Exact morning before acceptance (2026-02-18T10:00:00Z) -> None
        res_morning = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-02-18T10:00:00Z")
        self.assertIsNone(res_morning)

        # 4. One second before SEC acceptance (2026-02-18T16:05:21Z) -> None
        res_one_sec_prior = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-02-18T16:05:21Z")
        self.assertIsNone(res_one_sec_prior)

        # 5. Exact second of SEC acceptance (2026-02-18T16:05:22Z) -> Eligible
        res_exact = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-02-18T16:05:22Z")
        self.assertIsNotNone(res_exact)
        self.assertEqual(res_exact["growth_score"], 85)
        self.assertEqual(res_exact["available_from"], "2026-02-18T16:05:22Z")
        self.assertEqual(res_exact["point_in_time_status"], "POINT_IN_TIME")

        # 6. Future evaluation date (2026-03-01T00:00:00Z) -> Eligible
        res_future = self.db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-03-01T00:00:00Z")
        self.assertIsNotNone(res_future)
        self.assertEqual(res_future["composite_score"], 85)

    def test_current_only_records_blocked_from_historical_evaluation(self):
        """Records marked CURRENT_ONLY (e.g. unversioned scrape or static fallback)
        can be served live but must NEVER leak into historical evaluation."""
        current_only_snapshot = {
            "symbol": "MSFT",
            "as_of_date": "2026-02-20",
            "point_in_time_status": PointInTimeStatus.CURRENT_ONLY,
            "source": "yahoo_unversioned",
            "currentPrice": 420.0,
            "growthScore": 88,
            "qualityScore": 95,
            "valuationScore": 65,
            "momentumScore": 82,
            "tailRiskScore": 90,
            "compositeFactorScore": 86,
            "piotroskiFScore": 8,
            "verdict": "Fortress Balance Sheet",
        }
        self.db.save_factor_snapshot("MSFT", current_only_snapshot)

        # Live query (evaluation_timestamp=None) -> returns latest snapshot
        live_res = self.db.get_factor_snapshot("MSFT", evaluation_timestamp=None)
        self.assertIsNotNone(live_res)
        self.assertEqual(live_res["point_in_time_status"], "CURRENT_ONLY")
        self.assertEqual(live_res["growth_score"], 88)

        # Historical query at any timestamp -> strictly None
        hist_res_1 = self.db.get_factor_snapshot("MSFT", evaluation_timestamp="2026-02-20T12:00:00Z")
        self.assertIsNone(hist_res_1)

        hist_res_2 = self.db.get_factor_snapshot("MSFT", evaluation_timestamp="2026-03-01T00:00:00Z")
        self.assertIsNone(hist_res_2)

    def test_snapshot_versioning_progression(self):
        """Successive fundamental snapshots (Q3, then Q4) are appended and correctly resolved
        chronologically according to evaluation_timestamp."""
        # Q3 2025: Filed Nov 5, 2025
        q3 = {
            "symbol": "NVDA",
            "as_of_date": "2025-10-31",
            "period_end": "2025-10-31",
            "filing_date": "2025-11-05",
            "acceptance_datetime": "2025-11-05T21:00:00Z",
            "available_from": "2025-11-05T21:00:00Z",
            "point_in_time_status": PointInTimeStatus.POINT_IN_TIME,
            "source": "sec_edgar",
            "currentPrice": 140.0,
            "growthScore": 80,
            "qualityScore": 90,
            "compositeFactorScore": 82,
        }
        self.db.save_factor_snapshot("NVDA", q3)

        # Q4 2025: Filed Feb 25, 2026
        q4 = {
            "symbol": "NVDA",
            "as_of_date": "2026-01-31",
            "period_end": "2026-01-31",
            "filing_date": "2026-02-25",
            "acceptance_datetime": "2026-02-25T21:00:00Z",
            "available_from": "2026-02-25T21:00:00Z",
            "point_in_time_status": PointInTimeStatus.POINT_IN_TIME,
            "source": "sec_edgar",
            "currentPrice": 160.0,
            "growthScore": 96,
            "qualityScore": 95,
            "compositeFactorScore": 93,
        }
        self.db.save_factor_snapshot("NVDA", q4)

        # Verify both snapshots are stored
        snapshots = self.db.get_factor_history("NVDA")
        self.assertEqual(len(snapshots), 2)

        # Before Q3 is filed -> None
        self.assertIsNone(self.db.get_factor_snapshot("NVDA", evaluation_timestamp="2025-11-01T00:00:00Z"))

        # Between Q3 and Q4 -> Returns Q3 (growth=80)
        res_mid = self.db.get_factor_snapshot("NVDA", evaluation_timestamp="2025-12-15T00:00:00Z")
        self.assertIsNotNone(res_mid)
        self.assertEqual(res_mid["growth_score"], 80)
        self.assertEqual(res_mid["composite_score"], 82)

        # One second before Q4 is filed -> Still returns Q3
        res_pre_q4 = self.db.get_factor_snapshot("NVDA", evaluation_timestamp="2026-02-25T20:59:59Z")
        self.assertIsNotNone(res_pre_q4)
        self.assertEqual(res_pre_q4["growth_score"], 80)

        # At Q4 acceptance second -> Returns Q4 (growth=96)
        res_post_q4 = self.db.get_factor_snapshot("NVDA", evaluation_timestamp="2026-02-25T21:00:00Z")
        self.assertIsNotNone(res_post_q4)
        self.assertEqual(res_post_q4["growth_score"], 96)
        self.assertEqual(res_post_q4["composite_score"], 93)

        # Future date -> Returns Q4
        res_future = self.db.get_factor_snapshot("NVDA", evaluation_timestamp="2026-06-01T00:00:00Z")
        self.assertIsNotNone(res_future)
        self.assertEqual(res_future["growth_score"], 96)

    def test_no_future_leakage_invariant(self):
        """Ensures that for any evaluation timestamp T, no record with available_from > T is returned."""
        q1_2026 = {
            "symbol": "GOOGL",
            "available_from": "2026-04-25T20:30:00Z",
            "point_in_time_status": PointInTimeStatus.POINT_IN_TIME,
            "growthScore": 90,
            "compositeFactorScore": 88,
        }
        self.db.save_factor_snapshot("GOOGL", q1_2026)

        # Prior evaluation timestamp
        eval_ts = "2026-04-20T00:00:00Z"
        res = self.db.get_factor_snapshot("GOOGL", evaluation_timestamp=eval_ts)
        self.assertIsNone(res)

        # Direct invariant helper check
        self.assertFalse(is_pit_eligible("2026-04-25T20:30:00Z", eval_ts))
        self.assertTrue(is_pit_eligible("2026-04-25T20:30:00Z", "2026-04-25T20:30:00Z"))
        self.assertTrue(is_pit_eligible("2026-04-25T20:30:00Z", "2026-04-26T00:00:00Z"))
        self.assertFalse(is_pit_eligible(None, "2026-04-26T00:00:00Z"))
        self.assertFalse(is_pit_eligible("2026-04-25T20:30:00Z", None))

    def test_decision_hierarchy_fails_closed_when_pit_fundamentals_absent(self):
        """When historical PIT fundamentals are absent (has_fundamentals=False),
        DecisionHierarchyEngine must evaluate to EVIDENCE_INCOMPLETE and block actionability."""
        decision = DecisionHierarchyEngine.resolve_decision_state(
            symbol="AMD",
            current_price=175.0,
            candle_count=100,
            freshness_status="LIVE",
            has_fundamentals=False,  # Unverified or absent PIT fundamentals at T
            confluence_score=85.0,
            stage_phase=2,
            is_in_buy_zone=True,
            risk_reward_ratio=3.0,
        )
        self.assertEqual(decision["state"], DecisionState.EVIDENCE_INCOMPLETE.value)
        self.assertFalse(decision["isActionable"])
        self.assertFalse(decision["canSizeTrade"])
        self.assertIn("sec", decision["disqualificationReason"].lower())

    def test_hidden_gems_screener_pit_isolation(self):
        """HiddenGemsScreener static curated catalog is CURRENT_ONLY and barred from historical evaluation."""
        screener = HiddenGemsScreener()

        # Class invariants
        self.assertTrue(HiddenGemsScreener.STATIC_REFERENCE_DATA)
        self.assertFalse(HiddenGemsScreener.CAN_ENTER_HISTORICAL_MODEL_EVIDENCE)
        self.assertEqual(HiddenGemsScreener.POINT_IN_TIME_STATUS, "CURRENT_ONLY")

        # Live evaluation (evaluation_timestamp=None) -> returns curated metrics annotated with CURRENT_ONLY
        live_res = screener.evaluate_candidates(["NVDA"], evaluation_timestamp=None)
        self.assertEqual(len(live_res), 1)
        self.assertEqual(live_res[0]["ticker"], "NVDA")
        self.assertGreater(live_res[0]["composite_score"], 80)
        self.assertEqual(live_res[0]["point_in_time_status"], "CURRENT_ONLY")
        self.assertFalse(live_res[0]["historical_eligible"])

        # Historical evaluation (evaluation_timestamp="2025-06-01T00:00:00Z") -> fails closed
        hist_res = screener.evaluate_candidates(["NVDA"], evaluation_timestamp="2025-06-01T00:00:00Z")
        self.assertEqual(len(hist_res), 1)
        self.assertEqual(hist_res[0]["ticker"], "NVDA")
        self.assertEqual(hist_res[0]["composite_score"], 0.0)
        self.assertIn("Unverified", hist_res[0]["factor_verdict"])
        self.assertEqual(hist_res[0]["point_in_time_status"], "CURRENT_ONLY")
        self.assertFalse(hist_res[0]["historical_eligible"])

        # Uncataloged ticker fails closed regardless
        uncataloged = screener.evaluate_candidates(["UNKNOWN_XYZ"], evaluation_timestamp="2025-06-01T00:00:00Z")
        self.assertEqual(uncataloged[0]["composite_score"], 0.0)
        self.assertEqual(uncataloged[0]["point_in_time_status"], "UNKNOWN")
        self.assertFalse(uncataloged[0]["historical_eligible"])

    def test_weakest_pit_status_provenance_inheritance(self):
        """Derived factor composite inherits the weakest provenance of its inputs."""
        self.assertEqual(
            weakest_pit_status([PointInTimeStatus.POINT_IN_TIME, PointInTimeStatus.POINT_IN_TIME]),
            PointInTimeStatus.POINT_IN_TIME,
        )
        self.assertEqual(
            weakest_pit_status([PointInTimeStatus.POINT_IN_TIME, PointInTimeStatus.CURRENT_ONLY]),
            PointInTimeStatus.CURRENT_ONLY,
        )
        self.assertEqual(
            weakest_pit_status([PointInTimeStatus.POINT_IN_TIME, PointInTimeStatus.UNKNOWN]),
            PointInTimeStatus.UNKNOWN,
        )
        self.assertEqual(
            weakest_pit_status([PointInTimeStatus.CURRENT_ONLY, PointInTimeStatus.UNKNOWN]),
            PointInTimeStatus.UNKNOWN,
        )
        self.assertEqual(weakest_pit_status([]), PointInTimeStatus.UNKNOWN)

    def test_sec_edgar_fetcher_extracts_acceptance_and_report_date(self):
        """SecEdgarFetcher extracts acceptanceDateTime, reportDate, and forms available_from."""
        fetcher = SecEdgarFetcher()
        mock_submissions = {
            "filings": {
                "recent": {
                    "form": ["10-Q", "8-K"],
                    "filingDate": ["2026-02-18", "2026-01-20"],
                    "reportDate": ["2025-12-31", "2026-01-20"],
                    "acceptanceDateTime": ["2026-02-18T16:05:22.000Z", "2026-01-20T17:15:00.000Z"],
                    "accessionNumber": ["0000320193-26-000010", "0000320193-26-000005"],
                    "primaryDocDescription": ["Quarterly Report", "Current Report"],
                }
            }
        }
        with patch.object(fetcher, "get_company_submissions", return_value=mock_submissions), \
             patch.object(fetcher, "resolve_cik", return_value="0000320193"):
            filings = fetcher.get_recent_filings("AAPL", form_types=["10-Q"])
            self.assertEqual(len(filings), 1)
            f = filings[0]
            self.assertEqual(f["form"], "10-Q")
            self.assertEqual(f["filing_date"], "2026-02-18")
            self.assertEqual(f["report_date"], "2025-12-31")
            self.assertEqual(f["acceptance_datetime"], "2026-02-18T16:05:22.000Z")
            self.assertEqual(f["available_from"], "2026-02-18T16:05:22.000Z")

    def test_schema_migration_idempotency_and_preservation(self):
        """Legacy asset_factor_snapshots table without available_from is migrated to versioned table
        preserving legacy records marked as CURRENT_ONLY."""
        legacy_db_file = os.path.join(self.temp_dir.name, "legacy_market.db")
        # Create legacy table manually
        conn = sqlite3.connect(legacy_db_file)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE asset_factor_snapshots (
                symbol TEXT PRIMARY KEY,
                current_price REAL,
                price_change_24h REAL,
                growth_score INTEGER,
                quality_score INTEGER,
                valuation_score INTEGER,
                momentum_score INTEGER,
                tail_risk_score INTEGER,
                composite_score INTEGER,
                piotroski_f INTEGER,
                verdict TEXT
            )
        """)
        c.execute("""
            INSERT INTO asset_factor_snapshots (symbol, current_price, growth_score, composite_score, verdict)
            VALUES ('LEGACY_SYM', 99.5, 75, 80, 'Legacy Candidate')
        """)
        conn.commit()
        conn.close()

        # Instantiate MarketDatabase to trigger migration
        migrated_db = MarketDatabase(db_path=legacy_db_file)

        # Verify legacy row preserved and marked CURRENT_ONLY
        conn = sqlite3.connect(legacy_db_file)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM asset_factor_snapshots WHERE symbol = 'LEGACY_SYM'")
        row = dict(c.fetchone())
        conn.close()

        self.assertEqual(row["symbol"], "LEGACY_SYM")
        self.assertEqual(row["current_price"], 99.5)
        self.assertEqual(row["growth_score"], 75)
        self.assertEqual(row["point_in_time_status"], "CURRENT_ONLY")

        # Second instantiation should be idempotent and not fail
        second_db = MarketDatabase(db_path=legacy_db_file)
        self.assertIsNotNone(second_db)


if __name__ == "__main__":
    unittest.main()
