"""
Comprehensive Universe Authority, Determinism, Accounting Closure,
and Property Test Suite for Smart Money Wave 2A.
"""

import os
import random
import pytest

from analyst_dashboard.smart_money.universe_models import (
    SmartMoneySecurity,
    SmartMoneyEligibilityDecision,
    CikMappingStatus,
    SmartMoneyUniverseBuildManifest,
)
from analyst_dashboard.smart_money.universe_builder import (
    SmartMoneyUniverseBuilder,
    MAJOR_US_EXCHANGES,
)
from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner

# Deterministic Benchmark Sample representing diverse instruments
CANONICAL_TEST_POPULATION = [
    # Eligible Common Stocks with verified CIK
    {"ticker": "NVDA", "exchange": "NASDAQ", "name": "NVIDIA CORP", "cik": 1045810, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "AAPL", "exchange": "NASDAQ", "name": "APPLE INC", "cik": 320193, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "MSFT", "exchange": "NASDAQ", "name": "MICROSOFT CORP", "cik": 789019, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "PLTR", "exchange": "NASDAQ", "name": "PALANTIR TECHNOLOGIES INC", "cik": 1321655, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "VRT", "exchange": "NYSE", "name": "VERTIV HOLDINGS CO", "cik": 1674101, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "LLY", "exchange": "NYSE", "name": "ELI LILLY & CO", "cik": 59478, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "CRWD", "exchange": "NASDAQ", "name": "CROWDSTRIKE HOLDINGS INC", "cik": 1535527, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "AMD", "exchange": "NASDAQ", "name": "ADVANCED MICRO DEVICES INC", "cik": 2488, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},

    # Ineligible: OTC Exchange (Rule SM-E01)
    {"ticker": "TCEHY", "exchange": "OTC", "name": "TENCENT HOLDINGS LTD", "cik": 1293452, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"ticker": "BYDDF", "exchange": "OTC", "name": "BYD COMPANY LTD", "cik": 1423456, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},

    # Ineligible: ETF (Rule SM-E03)
    {"ticker": "SPY", "exchange": "ARCA", "name": "SPDR S&P 500 ETF TRUST", "cik": 884394, "security_type": "ETF", "asset_class": "ETF", "listing_status": "ACTIVE"},
    {"ticker": "QQQ", "exchange": "NASDAQ", "name": "INVESCO QQQ TRUST", "cik": 1067839, "security_type": "ETF", "asset_class": "ETF", "listing_status": "ACTIVE"},

    # Ineligible: Foreign Private Issuer / ADR (Rule SM-E03)
    {"ticker": "TSM", "exchange": "NYSE", "name": "TAIWAN SEMICONDUCTOR MFG CO", "cik": 1046179, "security_type": "ADR", "listing_status": "ACTIVE"},
    {"ticker": "NVO", "exchange": "NYSE", "name": "NOVO NORDISK A S", "cik": 353278, "security_type": "ADR", "listing_status": "ACTIVE"},

    # Ineligible: Inactive / Delisted (Rule SM-E04)
    {"ticker": "OLDCO", "exchange": "NASDAQ", "name": "OLD INACTIVE CORP", "cik": 999991, "security_type": "COMMON_STOCK", "listing_status": "INACTIVE"},
    {"ticker": "DELIST", "exchange": "NYSE", "name": "DELISTED CORP", "cik": 999992, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "delisting_date": "2026-01-01"},

    # Unresolved: Missing CIK (Rule SM-E07)
    {"ticker": "NOCIK", "exchange": "NASDAQ", "name": "NO CIK CORP", "cik": None, "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
]


def test_universe_builder_accounting_closure(tmp_path):
    """Verify that source_count == eligible + ineligible + unresolved and scannable closure holds."""
    db_path = str(tmp_path / "test_universe.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    manifest, scannable = builder.build_universe(
        raw_records=CANONICAL_TEST_POPULATION,
        source_authority="TEST_SEC_DIRECTORY",
        as_of="2026-10-10T12:00:00Z",
    )

    # Accounting closures
    assert manifest.source_population_count == len(CANONICAL_TEST_POPULATION)
    assert manifest.source_population_count == (
        manifest.eligible_count + manifest.ineligible_count + manifest.unresolved_count
    )
    assert manifest.eligible_count == 8  # NVDA, AAPL, MSFT, PLTR, VRT, LLY, CRWD, AMD
    assert manifest.cik_mapped_count == 8
    assert manifest.scannable_count == 8
    assert manifest.ineligible_count == 8  # 2 OTC, 2 ETF, 2 ADR, 2 Delisted/Inactive
    assert manifest.unresolved_count == 1  # 1 NOCIK
    assert len(scannable) == 8


def test_universe_build_id_and_hash_parity_across_runs(tmp_path):
    """Verify that building twice from the same source produces 100% identical ID, hash, and order."""
    db_path = str(tmp_path / "test_universe.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    manifest_1, scannable_1 = builder.build_universe(
        raw_records=CANONICAL_TEST_POPULATION,
        source_snapshot_id="SNAP_FIXED_001",
        as_of="2026-10-10T12:00:00Z",
    )

    manifest_2, scannable_2 = builder.build_universe(
        raw_records=CANONICAL_TEST_POPULATION,
        source_snapshot_id="SNAP_FIXED_001",
        as_of="2026-10-10T12:00:00Z",
    )

    # Invariants: Parity checks
    assert manifest_1.universe_build_id == manifest_2.universe_build_id
    assert manifest_1.member_identity_hash == manifest_2.member_identity_hash
    assert manifest_1.source_snapshot_hash == manifest_2.source_snapshot_hash
    assert [s.symbol for s in scannable_1] == [s.symbol for s in scannable_2]


def test_ambiguous_cik_collision_fails_closed(tmp_path):
    """Verify that a ticker colliding with multiple candidate CIKs fails closed and is quarantined."""
    db_path = str(tmp_path / "test_collision.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    colliding_records = [
        {"ticker": "AMBIG", "exchange": "NASDAQ", "name": "AMBIG CORP A", "cik": 111111, "security_type": "COMMON_STOCK"},
        {"ticker": "AMBIG", "exchange": "NASDAQ", "name": "AMBIG CORP B", "cik": 222222, "security_type": "COMMON_STOCK"},
        {"ticker": "AAPL", "exchange": "NASDAQ", "name": "APPLE INC", "cik": 320193, "security_type": "COMMON_STOCK"},
    ]

    manifest, scannable = builder.build_universe(colliding_records, as_of="2026-10-10T12:00:00Z")

    # AMBIG must NOT enter scannable population
    scannable_symbols = [s.symbol for s in scannable]
    assert "AMBIG" not in scannable_symbols
    assert "AAPL" in scannable_symbols
    assert manifest.scannable_count == 1
    assert manifest.unresolved_count >= 1  # Quarantined


def test_missing_cik_fails_closed(tmp_path):
    """Verify that an instrument with missing CIK never enters scannable population."""
    db_path = str(tmp_path / "test_missing.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    records = [
        {"ticker": "GHOST", "exchange": "NASDAQ", "name": "GHOST INC", "cik": None, "security_type": "COMMON_STOCK"},
    ]
    manifest, scannable = builder.build_universe(records, as_of="2026-10-10T12:00:00Z")

    assert manifest.scannable_count == 0
    assert len(scannable) == 0
    assert manifest.unresolved_count == 1


def test_scanner_market_wide_requires_universe_build_id(tmp_path):
    """Verify that SmartMoneyForm4Scanner requires universe_build_id in MARKET_WIDE mode."""
    db_path = str(tmp_path / "test_u.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)
    manifest, _ = builder.build_universe(CANONICAL_TEST_POPULATION, as_of="2026-10-10T12:00:00Z")

    scanner = SmartMoneyForm4Scanner(universe_builder=builder)

    # 1. MARKET_WIDE without universe_build_id must raise ValueError
    with pytest.raises(ValueError, match="SNAPSHOT_WITHOUT_UNIVERSE_BUILD_IN_MARKET_WIDE_MODE"):
        scanner.scan_issuers(is_bounded_sample=False, universe_build_id=None)

    # 2. MARKET_WIDE with valid universe_build_id succeeds and binds manifest metadata
    snap = scanner.scan_issuers(is_bounded_sample=False, universe_build_id=manifest.universe_build_id)
    assert snap.status == "PIPELINE_PENDING"  # Governed fail-closed until production authorization
    assert snap.universe_build_id == manifest.universe_build_id
    assert snap.universe_content_hash == manifest.member_identity_hash
    assert snap.scannable_count == manifest.scannable_count
    assert snap.matched_count is None  # Invariant: No false-zero


def test_property_500_permutations_member_hash_invariance(tmp_path):
    """
    Property Test: 500 randomized permutations of source records must yield
    100% identical member_identity_hash and member sort order every single time.
    """
    db_path = str(tmp_path / "test_perm.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    # Baseline build
    manifest_base, scannable_base = builder.build_universe(
        CANONICAL_TEST_POPULATION,
        source_snapshot_id="SNAP_PERM_BASE",
        as_of="2026-10-10T12:00:00Z",
    )
    expected_hash = manifest_base.member_identity_hash
    expected_order = [s.symbol for s in scannable_base]

    # Run 500 randomized permutations
    for i in range(500):
        permuted_records = list(CANONICAL_TEST_POPULATION)
        random.shuffle(permuted_records)

        manifest_perm, scannable_perm = builder.build_universe(
            permuted_records,
            source_snapshot_id="SNAP_PERM_BASE",
            as_of="2026-10-10T12:00:00Z",
        )

        assert manifest_perm.member_identity_hash == expected_hash, (
            f"Hash drift detected on permutation {i}!"
        )
        assert [s.symbol for s in scannable_perm] == expected_order, (
            f"Ordering drift detected on permutation {i}!"
        )


def test_bounded_universe_build_parity_against_35_benchmark_tickers(tmp_path):
    """
    Verify bounded universe build parity against the 35 historical benchmark tickers.
    Proves that:
    1. Historical 35 tickers are treated strictly as a bounded test subset, never confused with market-wide authority.
    2. ADRs within the 35 (e.g. NVO, ASML, ONON) are excluded under Rule SM-E03 (Exchange Act Rule 3a12-3).
    3. Accounting closure holds exactly across all 35 instruments.
    """
    from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE
    db_path = str(tmp_path / "test_35.db")
    builder = SmartMoneyUniverseBuilder(db_path=db_path)

    raw_35_records = []
    adrs = {"NVO", "ASML", "ONON"}
    for idx, sym in enumerate(HISTORICAL_VCP_35_REGRESSION_FIXTURE):
        sec_type = "ADR" if sym in adrs else "COMMON_STOCK"
        raw_35_records.append({
            "ticker": sym,
            "exchange": "NASDAQ" if sym in {"AAPL", "NVDA", "MSFT", "ASML", "CRWD", "DUOL"} else "NYSE",
            "name": f"{sym} CORP",
            "cik": 100000 + idx,
            "security_type": sec_type,
            "listing_status": "ACTIVE",
        })

    manifest, scannable = builder.build_universe(
        raw_records=raw_35_records,
        source_authority="HISTORICAL_35_FIXTURE",
        source_snapshot_id="SNAP_HISTORICAL_35",
        as_of="2026-10-10T12:00:00Z",
    )

    assert manifest.source_population_count == 35
    assert manifest.source_population_count == (
        manifest.eligible_count + manifest.ineligible_count + manifest.unresolved_count
    )
    # The 3 ADRs must be ineligible under Rule SM-E03
    assert manifest.ineligible_count == 3
    assert manifest.eligible_count == 32
    assert manifest.scannable_count == 32
    assert len(scannable) == 32

    scannable_syms = {s.symbol for s in scannable}
    for adr_sym in adrs:
        assert adr_sym not in scannable_syms
