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

# Ensure non-empty operator token is available for authorized test suites
os.environ.setdefault("ARX_OPERATOR_TOKEN", "test-operator-auth-token-xyz")


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


@pytest.fixture
def published_screener_universe(tmp_path):
    """Scoped fixture providing an authorized published universe build for screener tests.

    Constructed strictly via the real supported lifecycle:
    SourcePopulationSnapshot -> DeterministicUniverseBuilder -> UniverseStore.save_universe_build.
    Isolates UniverseStore, ScannerSnapshotStore, and MarketDatabaseEngine in a temporary directory and
    restores original stores and caches on teardown.
    """
    import datetime
    from analyst_dashboard.universe.contracts import (
        UNIVERSE_ID,
        PublicationDecision,
        ConstructionStatus,
    )
    from analyst_dashboard.universe.fixtures import create_canonical_fixture_snapshot
    from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
    from analyst_dashboard.universe.store import UniverseStore
    from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
    from analyst_dashboard.data.market_db import MarketDatabaseEngine
    import api.routes.screener as screener_module
    from api.routes.screener import vcp_scanner_runner

    univ_db_path = str(tmp_path / "test_universe.db")
    test_univ_store = UniverseStore(db_path=univ_db_path)

    snap_db_path = str(tmp_path / "test_scanner_snapshots.db")
    test_snap_store = ScannerSnapshotStore(db_path=snap_db_path)

    market_db_path = str(tmp_path / "test_market.db")
    test_market_db = MarketDatabaseEngine(db_path=market_db_path)

    snapshot = create_canonical_fixture_snapshot(snapshot_id="test-fixture-snap-001")
    builder = DeterministicUniverseBuilder()
    attestation, elig_rows, ready_rows = builder.build_universe(
        snapshot=snapshot, build_id="test-fixture-ubuild-001"
    )

    attestation.reconciliation_summary["candidate_generation_id"] = "CANDIDATE_GENERATION_004"
    attestation.publication_decision = PublicationDecision.PUBLISH
    attestation.construction_status = ConstructionStatus.COMPLETE

    test_univ_store.save_source_snapshot(snapshot)
    test_univ_store.save_universe_build(attestation, elig_rows, ready_rows)

    # Seed isolated MarketDatabaseEngine with >= 200 sessions (252 bars) for all fixture equities
    # Satisfies Weinstein/Minervini 200-SMA minimum-history invariant deterministically.
    today = datetime.date.today()
    base_date = today - datetime.timedelta(days=260)

    # 9 Stage 2 advancing equities (close > 50 SMA > 150 SMA > 200 SMA, rising 200 SMA)
    s2_syms = ["NVDA", "PLTR", "AAPL", "MSFT", "AMD", "CRWD", "ACLS", "LNTH", "TMDX"]
    # 1 Stage 3 distribution top equity (prior advance + topping 200 SMA rollover)
    s3_syms = ["DUOL"]
    # 1 Stage 4 declining equity (close < 200 SMA, declining 200 SMA)
    s4_syms = ["LLY"]
    # 1 Stage 1 basing equity (flat 200 SMA, base consolidation)
    s1_syms = ["VRTX"]

    all_syms = s2_syms + s3_syms + s4_syms + s1_syms
    candles = []

    for sym in s2_syms:
        for i in range(252):
            d = base_date + datetime.timedelta(days=i)
            p = 100.0 + (i * 0.4)
            candles.append((sym, d.isoformat(), p - 0.5, p + 1.0, p - 1.0, p, 2000000))

    for sym in s3_syms:
        closes = [60.0] * 30 + [130.0] * 22 + [100.0] * 178 + [101.0] * 22
        for i, c in enumerate(closes):
            d = base_date + datetime.timedelta(days=i)
            candles.append((sym, d.isoformat(), c, c + 1.0, c - 1.0, c, 1000000))

    for sym in s4_syms:
        for i in range(252):
            d = base_date + datetime.timedelta(days=i)
            p = 200.0 - (i * 0.4)
            candles.append((sym, d.isoformat(), p + 0.5, p + 1.0, p - 1.0, p, 2000000))

    for sym in s1_syms:
        for i in range(252):
            d = base_date + datetime.timedelta(days=i)
            p = 100.0
            candles.append((sym, d.isoformat(), p, p + 0.5, p - 0.5, p, 1000000))

    with test_market_db._get_connection() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, candles)
        for sym in all_syms:
            conn.execute("""
                INSERT OR REPLACE INTO asset_factor_snapshots (
                    symbol, current_price, price_change_24h, growth_score,
                    quality_score, valuation_score, momentum_score,
                    tail_risk_score, composite_score, piotroski_f, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (sym, 100.0, 0.5, 80, 80, 80, 80, 80, 80, 7, "Strong Buy"))
        conn.commit()

    orig_univ_store = vcp_scanner_runner.universe_store
    orig_snap_store = vcp_scanner_runner.snapshot_store
    orig_cached = vcp_scanner_runner._cached_active_snapshot
    orig_market_db = screener_module.market_db
    orig_runner_market_db = vcp_scanner_runner.market_db

    vcp_scanner_runner.universe_store = test_univ_store
    vcp_scanner_runner.snapshot_store = test_snap_store
    vcp_scanner_runner._cached_active_snapshot = None
    screener_module.market_db = test_market_db
    vcp_scanner_runner.market_db = test_market_db
    screener_module._STAGE_CLASSIFICATION_CACHE.clear()

    try:
        yield test_univ_store
    finally:
        vcp_scanner_runner.universe_store = orig_univ_store
        vcp_scanner_runner.snapshot_store = orig_snap_store
        vcp_scanner_runner._cached_active_snapshot = orig_cached
        screener_module.market_db = orig_market_db
        vcp_scanner_runner.market_db = orig_runner_market_db
        screener_module._STAGE_CLASSIFICATION_CACHE.clear()
