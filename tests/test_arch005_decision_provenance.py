"""ARCH-005: Adversarial regression tests for decision provenance and point-in-time evidence in MarketDatabase."""
import os
import tempfile
import pytest

from analyst_dashboard.data.market_db import MarketDatabase

pytestmark = pytest.mark.tier2b


def test_market_db_pit_rejects_lookahead_via_timezone_offsets():
    """Lexical comparison says 16 < 18, but 16:00-04:00 is 20:00Z (future at 18:00Z).
    MarketDatabase must fail closed and never leak future fundamental data."""
    with tempfile.TemporaryDirectory() as td:
        db = MarketDatabase(db_path=os.path.join(td, "test.db"))
        db.save_factor_snapshot("AAPL", {
            "symbol": "AAPL",
            "available_from": "2026-04-25T16:00:00-04:00",  # 20:00:00 UTC
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 99,
            "compositeFactorScore": 99,
        })
        res = db.get_factor_snapshot("AAPL", evaluation_timestamp="2026-04-25T18:00:00Z")
        assert res is None, f"Leaked future filing: {res}"


def test_market_db_pit_admits_past_snapshot_with_positive_offset():
    """Lexical comparison says 21 > 20, but 21:00+02:00 is 19:00Z (already public at 20:00Z).
    MarketDatabase must admit point-in-time evidence based on true instant."""
    with tempfile.TemporaryDirectory() as td:
        db = MarketDatabase(db_path=os.path.join(td, "test.db"))
        db.save_factor_snapshot("NVDA", {
            "symbol": "NVDA",
            "available_from": "2026-04-25T21:00:00+02:00",  # 19:00:00 UTC
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 88,
            "compositeFactorScore": 88,
        })
        res = db.get_factor_snapshot("NVDA", evaluation_timestamp="2026-04-25T20:00:00Z")
        assert res is not None, "Failed to admit point-in-time snapshot with positive timezone offset"
        assert res["growth_score"] == 88


def test_market_db_pit_rejects_timezone_naive_available_from():
    """Timestamps without explicit timezone offset or Z must not be admitted for point-in-time."""
    with tempfile.TemporaryDirectory() as td:
        db = MarketDatabase(db_path=os.path.join(td, "test.db"))
        db.save_factor_snapshot("MSFT", {
            "symbol": "MSFT",
            "available_from": "2026-04-25 16:00:00",
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 75,
        })
        res = db.get_factor_snapshot("MSFT", evaluation_timestamp="2026-04-25T18:00:00Z")
        assert res is None, f"Timezone-naive available_from admitted: {res}"


def test_market_db_pit_selects_latest_eligible_instant_across_differing_offsets():
    """Multiple snapshots with differing timezone offsets must resolve to the latest eligible UTC instant."""
    with tempfile.TemporaryDirectory() as td:
        db = MarketDatabase(db_path=os.path.join(td, "test.db"))
        # Snapshot A: April 25, 18:00 UTC
        db.save_factor_snapshot("GOOGL", {
            "symbol": "GOOGL",
            "available_from": "2026-04-25T18:00:00Z",
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 80,
            "compositeFactorScore": 80,
        })
        # Snapshot B: April 25, 15:30-04:00 = 19:30 UTC
        db.save_factor_snapshot("GOOGL", {
            "symbol": "GOOGL",
            "available_from": "2026-04-25T15:30:00-04:00",
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 90,
            "compositeFactorScore": 90,
        })
        # Snapshot C: April 25, 17:00-04:00 = 21:00 UTC
        db.save_factor_snapshot("GOOGL", {
            "symbol": "GOOGL",
            "available_from": "2026-04-25T17:00:00-04:00",
            "point_in_time_status": "POINT_IN_TIME",
            "growthScore": 95,
            "compositeFactorScore": 95,
        })

        # At 19:00:00Z -> only Snapshot A (18:00Z) is eligible
        res_19 = db.get_factor_snapshot("GOOGL", evaluation_timestamp="2026-04-25T19:00:00Z")
        assert res_19 is not None
        assert res_19["growth_score"] == 80

        # At 20:00:00Z -> Snapshot A (18:00Z) and B (19:30Z) eligible; B is latest
        res_20 = db.get_factor_snapshot("GOOGL", evaluation_timestamp="2026-04-25T20:00:00Z")
        assert res_20 is not None
        assert res_20["growth_score"] == 90

        # At 22:00:00Z -> Snapshot A, B, and C (21:00Z) eligible; C is latest
        res_22 = db.get_factor_snapshot("GOOGL", evaluation_timestamp="2026-04-25T22:00:00Z")
        assert res_22 is not None
        assert res_22["growth_score"] == 95
