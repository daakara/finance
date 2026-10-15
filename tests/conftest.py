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


@pytest.fixture
def published_screener_universe(tmp_path):
    """Scoped fixture providing an authorized published universe build for screener tests.

    Constructed strictly via the real supported lifecycle:
    SourcePopulationSnapshot -> DeterministicUniverseBuilder -> UniverseStore.save_universe_build.
    Isolates both UniverseStore and ScannerSnapshotStore in a temporary directory and
    restores the original store and cache on teardown.
    """
    from analyst_dashboard.universe.contracts import (
        UNIVERSE_ID,
        PublicationDecision,
        ConstructionStatus,
    )
    from analyst_dashboard.universe.fixtures import create_canonical_fixture_snapshot
    from analyst_dashboard.universe.builder import DeterministicUniverseBuilder
    from analyst_dashboard.universe.store import UniverseStore
    from analyst_dashboard.data.scanner_store import ScannerSnapshotStore
    from api.routes.screener import vcp_scanner_runner

    univ_db_path = str(tmp_path / "test_universe.db")
    test_univ_store = UniverseStore(db_path=univ_db_path)

    snap_db_path = str(tmp_path / "test_scanner_snapshots.db")
    test_snap_store = ScannerSnapshotStore(db_path=snap_db_path)

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

    orig_univ_store = vcp_scanner_runner.universe_store
    orig_snap_store = vcp_scanner_runner.snapshot_store
    orig_cached = vcp_scanner_runner._cached_active_snapshot

    vcp_scanner_runner.universe_store = test_univ_store
    vcp_scanner_runner.snapshot_store = test_snap_store
    vcp_scanner_runner._cached_active_snapshot = None

    try:
        yield test_univ_store
    finally:
        vcp_scanner_runner.universe_store = orig_univ_store
        vcp_scanner_runner.snapshot_store = orig_snap_store
        vcp_scanner_runner._cached_active_snapshot = orig_cached

