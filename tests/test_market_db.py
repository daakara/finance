"""Unit tests for SQLite persistent MarketDatabaseEngine."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.data.market_db import MarketDatabaseEngine


def test_market_database_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        db = MarketDatabaseEngine(db_path=temp_db)

        # 1. Test Saving & Retrieving Candles
        raw_candles = [
            {"time": f"2026-01-0{i+1}", "open": 100.0 + i, "high": 102.0 + i, "low": 99.0 + i, "close": 101.0 + i, "volume": 1000000 + i * 1000}
            for i in range(9)
        ]
        raw_candles.append({"time": "2026-01-10", "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0, "volume": 1010000})

        db.save_daily_candles("LNTH", raw_candles)
        candles = db.get_daily_candles("LNTH", limit=5)
        assert len(candles) == 5
        assert candles[-1]["close"] == 110.0

        # 2. Test Latest Price
        price_info = db.get_latest_price("LNTH")
        assert price_info is not None
        assert price_info["symbol"] == "LNTH"
        assert price_info["currentPrice"] == 110.0

        # 3. Test Factor Snapshot Persistence
        snapshot = {
            "currentPrice": 110.0,
            "priceChangePct24h": 2.5,
            "growthScore": 92,
            "qualityScore": 95,
            "valuationScore": 80,
            "momentumScore": 88,
            "tailRiskScore": 84,
            "compositeFactorScore": 88,
            "piotroskiFScore": 9,
            "verdict": "Greenblatt Magic Formula",
        }
        db.save_factor_snapshot("LNTH", snapshot)
        loaded_factors = db.get_factor_snapshot("LNTH")
        assert loaded_factors is not None
        assert loaded_factors["growth_score"] == 92
        assert loaded_factors["piotroski_f"] == 9

        # 4. Test Catalyst Persistence
        catalyst = {
            "company_name": "Lantheus Holdings Inc.",
            "sector": "Healthcare / Diagnostics",
            "primary_drug_trial": "PYLARIFY Imaging Volume",
            "trial_phase": "Commercial",
            "trial_readout_timeline": "Q3 2026",
            "efficacy_summary": "PSMA PET imaging monopoly",
            "competitive_edge": "High margin diagnostic consumables",
        }
        db.save_catalyst("LNTH", catalyst)
        loaded_cat = db.get_catalyst("LNTH")
        assert loaded_cat is not None
        assert loaded_cat["primary_drug_trial"] == "PYLARIFY Imaging Volume"

    finally:
        del db
        import gc
        gc.collect()
        try:
            if os.path.exists(temp_db):
                os.remove(temp_db)
        except Exception:
            pass


if __name__ == "__main__":
    test_market_database_lifecycle()
    print("[PASS] ALL MARKET DB TESTS PASSED SUCCESSFULLY")
