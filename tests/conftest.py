"""ARX Terminal — 4-Tier Test Gate Configuration.

Marker Definitions:
  tier1: Core Invariant Gate (always runs, <60s)
  tier2a: Presentation & UX Gate (path-filtered)
  tier2b: State & Solver Gate (path-filtered)
  tier2c: Provenance & Data Gate (path-filtered)
  tier3: Pre-Flight Release Gate (merge to main only)
"""
import json
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def seed_test_market_database():
    """Ensure MarketDatabaseEngine contains baseline factor snapshots and daily candles.

    Fresh CI runners (e.g. GitHub Actions) have empty SQLite stores and lack external
    market API access. This seeds the local database from a deterministic static fixture
    if the snapshot table is empty.
    """
    try:
        from analyst_dashboard.data.market_db import MarketDatabaseEngine

        db = MarketDatabaseEngine()
        with db._get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM asset_factor_snapshots"
            ).fetchone()[0]
            if count == 0:
                fixture_path = os.path.join(
                    os.path.dirname(__file__), "fixtures", "test_market_cache.json"
                )
                if os.path.exists(fixture_path):
                    with open(fixture_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for sym, snap in data.get("snapshots", {}).items():
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO asset_factor_snapshots (
                                symbol, current_price, price_change_24h, growth_score,
                                quality_score, valuation_score, momentum_score,
                                tail_risk_score, composite_score, piotroski_f, verdict
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                sym,
                                snap.get("current_price", 100.0),
                                snap.get("price_change_24h", 0.0),
                                snap.get("growth_score", 80),
                                snap.get("quality_score", 80),
                                snap.get("valuation_score", 80),
                                snap.get("momentum_score", 80),
                                snap.get("tail_risk_score", 80),
                                snap.get("composite_score", 80),
                                snap.get("piotroski_f", 7),
                                snap.get("verdict", "Strong Buy"),
                            ),
                        )
                    for sym, candles in data.get("candles", {}).items():
                        for c in candles:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO asset_ohlcv_daily (
                                    symbol, trade_date, open, high, low, close, volume
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    sym,
                                    c["trade_date"],
                                    c["open"],
                                    c["high"],
                                    c["low"],
                                    c["close"],
                                    c["volume"],
                                ),
                            )
                    conn.commit()
    except Exception as e:
        print(f"[conftest] Note: Market database seeding skipped: {e}")
