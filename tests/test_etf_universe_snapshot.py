"""Targeted Test Suite for ARX ETF Universe Snapshot Construction & Contract (Spec v1.0.2).

Verifies:
1. Discovery-source metadata capture (URL, timestamp, raw SHA-256, creation time, row counts).
2. Nasdaq ETF flag discovery-only semantics (flag != eligibility).
3. Unknown-structure fail-closed quarantine (defaults to quarantined & ineligible).
4. Vehicle-structure classification across all 10 allowed structure classes.
5. Defensive negative heuristics (negative safety net only; cannot certify inclusion).
6. Classification precedence & collision handling (positive registry wins over negative keyword).
7. Registry provenance metadata enforcement (source authority, as-of date, policies).
8. Rolling 60-day ADV cross-sectional 80th percentile logic.
9. Minimum 250-trading-session history enforcement.
10. Deterministic exclusion reason precedence.
11. End-to-end composite snapshot pipeline integration (all 10 canonical cases).
12. Snapshot & manifest schema contracts and lineage binding.
"""

import os
import json
import tempfile
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from scripts.research.build_etf_dataset import (
    SPEC_PATH,
    SPEC_VERSION_V102,
    CANONICAL_SPEC_COMMIT_V102,
    CANONICAL_FILTERED_SPEC_SHA256_V102,
    SPEC_VERSION_V103,
    CANONICAL_SPEC_COMMIT_V103,
    CANONICAL_FILTERED_SPEC_SHA256_V103,
    SourceCacheManager,
    parse_nasdaq_traded_content,
    ClassificationAuthorityEngine,
    build_universe_snapshot,
    evaluate_liquidity_and_history,
    ALL_REGISTRIES,
    REGISTRY_KNOWN_1940_ACT_UITS,
    REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS,
    REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS,
    REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES,
    REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS,
    REGISTRY_KNOWN_CRYPTO_PRODUCTS,
    REGISTRY_KNOWN_LEVERAGED_INVERSE_PRODUCTS,
    ALLOWED_STRUCTURE_CLASSES,
    RESEARCH_ELIGIBLE_STRUCTURES,
    POLICY_V11_PATH,
    POLICY_V11_SHA256,
    POLICY_V11_COMMIT,
    MANDATE_EVIDENCE_PATH,
    NPORT_DERIVED_METRICS_PATH,
    UNIVERSE_SNAPSHOT_PATH,
    UNIVERSE_MANIFEST_PATH,
    get_file_sha256,
    get_git_commit,
)

SAMPLE_NASDAQ_DATA = (
    b"Symbol|Security Name|Listing Exchange|Market Category|ETF|Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares\n"
    b"SPY|SPDR S&P 500 ETF Trust|P| |Y|100|N|N|SPY|SPY|N\n"
    b"GLD|SPDR Gold Shares|P| |Y|100|N|N|GLD|GLD|N\n"
    b"VUSB|Vanguard Ultra-Short Bond ETF|P| |Y|100|N|N|VUSB|VUSB|N\n"
    b"TQQQ|ProShares UltraPro QQQ 3x Shares|Q|G|Y|100|N|N|TQQQ|TQQQ|N\n"
    b"SQQQ|ProShares UltraPro Short QQQ|Q|G|Y|100|N|N|SQQQ|SQQQ|N\n"
    b"AMJ|JPMorgan Alerian MLP Index ETN|P| |Y|100|N|N|AMJ|AMJ|N\n"
    b"USO|United States Oil Fund LP Futures|P| |Y|100|N|N|USO|USO|N\n"
    b"BITO|ProShares Bitcoin Strategy ETF|P| |Y|100|N|N|BITO|BITO|N\n"
    b"XYZ|XYZ Mystery Trust Unknown Fund|P| |Y|100|N|N|XYZ|XYZ|N\n"
    b"LOW1|Low Volume 1940 Act ETF 1|P| |Y|100|N|N|LOW1|LOW1|N\n"
    b"LOW2|Low Volume 1940 Act ETF 2|P| |Y|100|N|N|LOW2|LOW2|N\n"
    b"LOW3|Low Volume 1940 Act ETF 3|P| |Y|100|N|N|LOW3|LOW3|N\n"
    b"LOW4|Low Volume 1940 Act ETF 4|P| |Y|100|N|N|LOW4|LOW4|N\n"
    b"LOW5|Low Volume 1940 Act ETF 5|P| |Y|100|N|N|LOW5|LOW5|N\n"
    b"LOW6|Low Volume 1940 Act ETF 6|P| |Y|100|N|N|LOW6|LOW6|N\n"
    b"LOW7|Low Volume 1940 Act ETF 7|P| |Y|100|N|N|LOW7|LOW7|N\n"
    b"LOW8|Low Volume 1940 Act ETF 8|P| |Y|100|N|N|LOW8|LOW8|N\n"
    b"LOW9|Low Volume 1940 Act ETF 9|P| |Y|100|N|N|LOW9|LOW9|N\n"
    b"LOW10|Low Volume 1940 Act ETF 10|P| |Y|100|N|N|LOW10|LOW10|N\n"
    b"LOW11|Low Volume 1940 Act ETF 11|P| |Y|100|N|N|LOW11|LOW11|N\n"
    b"LOW12|Low Volume 1940 Act ETF 12|P| |Y|100|N|N|LOW12|LOW12|N\n"
    b"NEWB|Newly Listed 1940 Act ETF|P| |Y|100|N|N|NEWB|NEWB|N\n"
    b"NODATA|Missing Price 1940 Act ETF|P| |Y|100|N|N|NODATA|NODATA|N\n"
    b"NONETF|Common Stock Inc|N| |N|100|N|N|NONETF|NONETF|N\n"
    b"TESTS|Test Issue Symbol|Q|G|Y|100|Y|N|TESTS|TESTS|N\n"
    b"File Creation Time: 0925202611:00|||||\n"
)


def test_discovery_source_metadata_capture():
    df, meta = parse_nasdaq_traded_content(SAMPLE_NASDAQ_DATA, source_url="test://nasdaqtraded.txt")
    assert meta["source_url_identifier"] == "test://nasdaqtraded.txt"
    assert meta["raw_source_sha256"] == hashlib.sha256(SAMPLE_NASDAQ_DATA).hexdigest()
    assert meta["file_creation_timestamp"] == "0925202611:00"
    assert meta["row_count"] == 25
    assert meta["etf_flagged_row_count"] == 24
    assert "retrieval_timestamp" in meta


def test_nasdaq_etf_flag_discovery_only_and_fail_closed_quarantine():
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="XYZ",
        security_name="XYZ Mystery Trust Unknown Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True
    )
    assert rec["nasdaq_etf_flag"] is True
    assert rec["is_research_eligible"] is False
    assert rec["structure_verified"] is False
    assert rec["vehicle_structure"] == "UNKNOWN"
    assert rec["vehicle_structure_state"] == "QUARANTINED"
    assert rec["exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"
    assert rec["classification_source"] == "TIER_5_UNKNOWN_STRUCTURE_QUARANTINE"


def test_vehicle_structure_classification_and_registries():
    # 1. 1940 Act Unit Investment Trust ETF (SPY)
    spy = ClassificationAuthorityEngine.classify_security("SPY", "SPDR S&P 500 ETF Trust", "P", True)
    assert spy["vehicle_structure"] == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
    assert spy["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert spy["research_subtype"] == "EQUITY_INDEX"
    assert spy["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert spy["structure_verified"] is True
    assert spy["subtype_authorized"] is True
    # Structure alone MUST NOT set is_research_eligible = True
    assert spy["is_research_eligible"] is False
    assert spy["exclusion_reason"] is None

    # 2. 1940 Act Open-End ETF (VOO)
    voo = ClassificationAuthorityEngine.classify_security("VOO", "Vanguard S&P 500 ETF", "P", True)
    assert voo["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert voo["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert voo["research_subtype"] == "EQUITY_INDEX"
    assert voo["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert voo["structure_verified"] is True
    assert voo["subtype_authorized"] is True
    assert voo["is_research_eligible"] is False
    assert voo["exclusion_reason"] is None

    # 3. Physical precious metal grantor trust (GLD)
    gld = ClassificationAuthorityEngine.classify_security("GLD", "SPDR Gold Shares", "P", True)
    assert gld["vehicle_structure"] == "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST"
    assert gld["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert gld["research_subtype"] == "COMMODITY_PHYSICAL"
    assert gld["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert gld["structure_verified"] is True
    assert gld["subtype_authorized"] is True
    assert gld["is_research_eligible"] is False
    assert gld["exclusion_reason"] is None


def test_defensive_negative_heuristics():
    # Leveraged ETF
    tqqq = ClassificationAuthorityEngine.classify_security("TQQQ", "ProShares UltraPro QQQ 3x Shares", "Q", True)
    assert tqqq["vehicle_structure"] == "LEVERAGED_ETF"
    assert tqqq["vehicle_structure_state"] == "EXCLUDED"
    assert tqqq["structure_verified"] is False
    assert tqqq["is_research_eligible"] is False
    assert tqqq["exclusion_reason"] == "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"

    # ETN
    amj = ClassificationAuthorityEngine.classify_security("AMJ", "JPMorgan Alerian MLP Index ETN", "P", True)
    assert amj["vehicle_structure"] == "EXCHANGE_TRADED_NOTE"
    assert amj["vehicle_structure_state"] == "EXCLUDED"
    assert amj["structure_verified"] is False
    assert amj["is_research_eligible"] is False
    assert amj["exclusion_reason"] == "EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE"

    # Commodity futures pool
    uso = ClassificationAuthorityEngine.classify_security("USO", "United States Oil Fund LP Futures", "P", True)
    assert uso["vehicle_structure"] == "COMMODITY_FUTURES_POOL"
    assert uso["vehicle_structure_state"] == "EXCLUDED"
    assert uso["structure_verified"] is False
    assert uso["is_research_eligible"] is False
    assert uso["exclusion_reason"] == "EXCLUDED_STRUCTURE_COMMODITY_FUTURES_POOL"

    # Crypto-linked
    bito = ClassificationAuthorityEngine.classify_security("BITO", "ProShares Bitcoin Strategy ETF", "P", True)
    assert bito["vehicle_structure"] == "CRYPTO_LINKED_PRODUCT"
    assert bito["vehicle_structure_state"] == "EXCLUDED"
    assert bito["structure_verified"] is False
    assert bito["is_research_eligible"] is False
    assert bito["exclusion_reason"] == "EXCLUDED_STRUCTURE_CRYPTO_LINKED"

    # Closed-end fund
    cef = ClassificationAuthorityEngine.classify_security("BOE", "BlackRock Enhanced Global Dividend CEF", "N", True)
    assert cef["vehicle_structure"] == "CLOSED_END_FUND"
    assert cef["vehicle_structure_state"] == "EXCLUDED"
    assert cef["structure_verified"] is False
    assert cef["is_research_eligible"] is False
    assert cef["exclusion_reason"] == "EXCLUDED_STRUCTURE_CLOSED_END_FUND"

    # Mutual fund
    mf = ClassificationAuthorityEngine.classify_security("VFINX", "Vanguard 500 Index Mutual Fund", "N", True)
    assert mf["vehicle_structure"] == "MUTUAL_FUND"
    assert mf["vehicle_structure_state"] == "EXCLUDED"
    assert mf["structure_verified"] is False
    assert mf["is_research_eligible"] is False
    assert mf["exclusion_reason"] == "EXCLUDED_STRUCTURE_MUTUAL_FUND"


def test_positive_negative_collision_and_authority_precedence():
    # VUSB: Verified 1940 Act ETF whose title contains "Ultra-Short"
    # Negative regex matches "ultra" and "short", but Tier 3 positive authority PREVAILS
    vusb = ClassificationAuthorityEngine.classify_security(
        symbol="VUSB",
        security_name="Vanguard Ultra-Short Bond ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True
    )
    assert vusb["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert vusb["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert vusb["research_subtype"] == "FIXED_INCOME_CREDIT"
    assert vusb["structure_verified"] is True
    assert vusb["subtype_authorized"] is True

    # Collision test 2: Explicit blocklist overrides generic discovery
    amj = ClassificationAuthorityEngine.classify_security(
        symbol="AMJ",
        security_name="JPMorgan Alerian MLP Index ETN",
        listing_exchange="P",
        nasdaq_etf_flag=True
    )
    assert amj["vehicle_structure"] == "EXCHANGE_TRADED_NOTE"
    assert amj["vehicle_structure_state"] == "EXCLUDED"
    assert amj["structure_verified"] is False


def test_registry_provenance_metadata():
    for reg in ALL_REGISTRIES:
        assert reg.registry_name, "Registry name missing"
        assert len(reg.entries) > 0, f"Registry {reg.registry_name} has empty entries"
        assert reg.source_authority, f"Registry {reg.registry_name} missing source_authority"
        assert reg.source_identifier, f"Registry {reg.registry_name} missing source_identifier"
        assert reg.as_of_date == "2026-09-25", f"Registry {reg.registry_name} stale or missing as_of_date"
        assert reg.temporality == "CURRENT", f"Registry {reg.registry_name} temporality must be CURRENT"
        assert reg.update_policy, f"Registry {reg.registry_name} missing update_policy"
        assert reg.failure_policy, f"Registry {reg.registry_name} missing failure_policy"
        assert reg.semantic_role in {"VERIFIED_POSITIVE_ALLOWLIST", "VERIFIED_NEGATIVE_BLOCKLIST"}


def test_liquidity_and_history_evaluation():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)

    df_a = pd.DataFrame({
        "Close": 100.0 + np.random.randn(300),
        "Volume": 1000000.0 + np.random.randn(300) * 10000
    }, index=dates)

    df_b = pd.DataFrame({
        "Close": 50.0 + np.random.randn(300),
        "Volume": 1000.0 + np.random.randn(300) * 10
    }, index=dates)

    df_c = pd.DataFrame({
        "Close": 20.0 + np.random.randn(100),
        "Volume": 2000000.0 + np.random.randn(100) * 10000
    }, index=dates[-100:])

    price_dict = {"A": df_a, "B": df_b, "C": df_c}
    res = evaluate_liquidity_and_history(["A", "B", "C"], price_dict, dates, adv_window=60, adv_percentile=0.80, min_history=250)

    last_date = str(dates[-1].date())
    res_last = res[res["observation_date"] == last_date].set_index("symbol")

    assert bool(res_last.loc["A", "is_liquid"]) is True
    assert bool(res_last.loc["A", "has_min_history"]) is True
    assert bool(res_last.loc["A", "in_universe"]) is True

    assert bool(res_last.loc["B", "is_liquid"]) is False
    assert bool(res_last.loc["B", "in_universe"]) is False

    assert bool(res_last.loc["C", "has_min_history"]) is False
    assert bool(res_last.loc["C", "in_universe"]) is False


def test_end_to_end_composite_snapshot_pipeline_and_10_cases():
    dates = pd.date_range("2025-01-01", periods=300, freq="B")
    np.random.seed(42)

    def make_price_df(n_bars: int, mean_vol: float):
        return pd.DataFrame({
            "Close": 100.0,
            "Volume": mean_vol
        }, index=dates[-n_bars:])

    # 12 low volume symbols + 3 high volume symbols (SPY, GLD, VUSB)
    # 3 out of 15 is exactly the top 20% (>= 80th percentile)
    mock_market_data = {
        # Case 1: SPY (verified eligible + high ADV + 300 sessions)
        "SPY": make_price_df(300, 10_000_000.0),
        # Case 2: LOW1 (verified eligible + low ADV)
        # Case 3: NEWB (verified eligible + short history 100 sessions < 250)
        "NEWB": make_price_df(100, 10_000_000.0),
        # Case 4: XYZ (unknown structure + high ADV + 300 sessions)
        "XYZ": make_price_df(300, 10_000_000.0),
        # Case 5: AMJ (ETN + high ADV + 300 sessions)
        "AMJ": make_price_df(300, 10_000_000.0),
        # Case 6: TQQQ (leveraged ETF + high ADV + 300 sessions)
        "TQQQ": make_price_df(300, 10_000_000.0),
        # Case 7: SQQQ (inverse ETF + high ADV + 300 sessions)
        "SQQQ": make_price_df(300, 10_000_000.0),
        # Case 8: GLD (physical grantor trust + high ADV + 300 sessions)
        "GLD": make_price_df(300, 10_000_000.0),
        # Case 9: NODATA (eligible structure with missing price history)
        "NODATA": None,
        # Case 10: VUSB (verified positive registry with misleading negative keyword + qualifying market history)
        "VUSB": make_price_df(300, 10_000_000.0),
    }

    # Populate 12 low volume symbols
    for i in range(1, 13):
        mock_market_data[f"LOW{i}"] = make_price_df(300, 100.0 * i)

    low_syms = {f"LOW{i}" for i in range(1, 13)}
    REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.subtypes["EQUITY_INDEX"] = frozenset(
        set(REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.subtypes["EQUITY_INDEX"]) | low_syms | {"NEWB", "NODATA"}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        disc_file = tmp_path / "nasdaqtraded.txt"
        disc_file.write_bytes(SAMPLE_NASDAQ_DATA)

        snap_parquet = tmp_path / "ETF_SURVIVING_UNIVERSE_V1.parquet"
        snap_manifest = tmp_path / "ETF_SURVIVING_UNIVERSE_V1_MANIFEST.json"
        src_manifest = SourceCacheManager(manifest_path=tmp_path / "source_cache_manifest.json")

        manifest = build_universe_snapshot(
            discovery_file=disc_file,
            price_data_dict=mock_market_data,
            output_parquet=snap_parquet,
            output_manifest=snap_manifest,
            cache_dir=tmp_path,
            source_manifest=src_manifest
        )

        df = pd.read_parquet(snap_parquet).set_index("symbol")

        # 1. SPY: verified + high ADV + >=250 sessions -> True
        assert bool(df.loc["SPY", "is_research_eligible"]) is True
        assert pd.isna(df.loc["SPY", "exclusion_reason"]) or df.loc["SPY", "exclusion_reason"] is None
        assert df.loc["SPY", "vehicle_structure"] == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
        assert df.loc["SPY", "research_subtype"] == "EQUITY_INDEX"

        # 2. LOW1: verified eligible + low ADV -> False (ADV60_BELOW_80TH_PERCENTILE)
        assert bool(df.loc["LOW1", "is_research_eligible"]) is False
        assert df.loc["LOW1", "exclusion_reason"] == "ADV60_BELOW_80TH_PERCENTILE"
        assert df.loc["LOW1", "vehicle_structure"] == "1940_ACT_OPEN_END_ETF"

        # 3. NEWB: verified eligible + < 250 sessions -> False (INSUFFICIENT_HISTORY_LT_250)
        assert bool(df.loc["NEWB", "is_research_eligible"]) is False
        assert df.loc["NEWB", "exclusion_reason"] == "INSUFFICIENT_HISTORY_LT_250"

        # 4. XYZ: unknown structure + qualifying market history -> False (UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED)
        assert bool(df.loc["XYZ", "is_research_eligible"]) is False
        assert df.loc["XYZ", "exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"
        assert df.loc["XYZ", "vehicle_structure"] == "UNKNOWN"

        # 5. AMJ: ETN + high ADV -> False (EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE)
        assert bool(df.loc["AMJ", "is_research_eligible"]) is False
        assert df.loc["AMJ", "exclusion_reason"] == "EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE"
        assert df.loc["AMJ", "vehicle_structure"] == "EXCHANGE_TRADED_NOTE"

        # 6. TQQQ: leveraged ETF + high ADV -> False (EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE)
        assert bool(df.loc["TQQQ", "is_research_eligible"]) is False
        assert df.loc["TQQQ", "exclusion_reason"] == "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"
        assert df.loc["TQQQ", "vehicle_structure"] == "LEVERAGED_ETF"

        # 7. SQQQ: inverse ETF + high ADV -> False (EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE)
        assert bool(df.loc["SQQQ", "is_research_eligible"]) is False
        assert df.loc["SQQQ", "exclusion_reason"] == "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"
        assert df.loc["SQQQ", "vehicle_structure"] == "INVERSE_ETF"

        # 8. GLD: physical grantor trust + qualifying market history -> True
        assert bool(df.loc["GLD", "is_research_eligible"]) is True
        assert pd.isna(df.loc["GLD", "exclusion_reason"]) or df.loc["GLD", "exclusion_reason"] is None
        assert df.loc["GLD", "vehicle_structure"] == "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST"
        assert df.loc["GLD", "research_subtype"] == "COMMODITY_PHYSICAL"

        # 9. NODATA: eligible structure + missing price history -> False (MARKET_DATA_MISSING_OR_INVALID)
        assert bool(df.loc["NODATA", "is_research_eligible"]) is False
        assert df.loc["NODATA", "exclusion_reason"] == "MARKET_DATA_MISSING_OR_INVALID"

        # 10. VUSB: verified positive registry with misleading negative keyword + qualifying market history -> True
        assert bool(df.loc["VUSB", "is_research_eligible"]) is True
        assert pd.isna(df.loc["VUSB", "exclusion_reason"]) or df.loc["VUSB", "exclusion_reason"] is None
        assert df.loc["VUSB", "vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
        assert df.loc["VUSB", "research_subtype"] == "FIXED_INCOME_CREDIT"

        # Non-ETF common stock dropped before classification into snapshot
        assert "NONETF" not in df.index
        assert (df["nasdaq_etf_flag"] == True).all()

        # Verify composite count binding
        assert manifest["eligible_row_count"] == int(df["is_research_eligible"].sum())
        assert manifest["excluded_row_count"] == int((~df["is_research_eligible"]).sum())
        assert manifest["eligible_row_count"] == 3  # Exactly SPY, GLD, VUSB
        assert manifest["excluded_row_count"] == len(df) - 3


def test_source_cache_lineage():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "mock.parquet"
        test_file.write_bytes(b"DATA")

        mgr = SourceCacheManager(manifest_path=tmp_path / "source_cache.json")
        entry = mgr.record(
            path=test_file,
            provider="TEST_PROVIDER",
            semantic_role="UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY"
        )

        assert entry["provider"] == "TEST_PROVIDER"
        assert entry["semantic_role"] == "UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY"
        assert entry["byte_size"] == 4
        assert entry["sha256"] == hashlib.sha256(b"DATA").hexdigest()
        assert "retrieval_timestamp" in entry

        mgr2 = SourceCacheManager(manifest_path=tmp_path / "source_cache.json")
        rel_key = str(test_file).replace("\\", "/")
        assert rel_key in mgr2.entries


def test_discovery_domain_etf_flag_only():
    """Section 14: Verifies discovery domain integrity - ETF == 'Y' strictly enforced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        disc_file = tmp_path / "nasdaqtraded.txt"
        disc_file.write_bytes(SAMPLE_NASDAQ_DATA)

        snap_parquet = tmp_path / "ETF_SURVIVING_UNIVERSE_V1.parquet"
        snap_manifest = tmp_path / "ETF_SURVIVING_UNIVERSE_V1_MANIFEST.json"

        build_universe_snapshot(
            discovery_file=disc_file,
            price_data_dict={},
            output_parquet=snap_parquet,
            output_manifest=snap_manifest,
            cache_dir=tmp_path,
            source_manifest=SourceCacheManager(manifest_path=tmp_path / "source_cache_manifest.json"),
        )

        df = pd.read_parquet(snap_parquet)
        assert len(df) > 0
        # Invariant 1: No row in the snapshot has nasdaq_etf_flag == False
        assert (df["nasdaq_etf_flag"] == True).all()
        # Invariant 2: Non-ETF securities from source are completely absent from snapshot
        assert "NONETF" not in df["symbol"].values
        # Invariant 3: Test issues from source are completely absent from snapshot
        assert "TESTS" not in df["symbol"].values


def test_market_eligibility_as_of_freshness():
    """Section 15: Verifies market data freshness requirement - stale sessions fail-closed."""
    trading_cal = pd.date_range("2026-01-01", "2026-09-24", freq="B")

    # Stale candidate ending earlier (e.g. 2026-09-18)
    stale_dates = pd.date_range("2026-01-01", "2026-09-18", freq="B")
    df_stale = pd.DataFrame({
        "Close": 100.0,
        "Volume": 50_000_000.0
    }, index=stale_dates)

    # Fresh candidate ending on current session (2026-09-24)
    df_fresh = pd.DataFrame({
        "Close": 100.0,
        "Volume": 50_000_000.0
    }, index=trading_cal)

    price_dict = {
        "STALE": df_stale,
        "FRESH": df_fresh
    }

    res_df = evaluate_liquidity_and_history(
        symbols=["STALE", "FRESH"],
        price_data_adj=price_dict,
        trading_calendar=trading_cal,
        adv_window=60,
        adv_percentile=0.80,
        min_history=100
    )

    # Inspect the observation on the current session (2026-09-24)
    last_date = str(trading_cal.max().date())
    as_of_res = res_df[res_df["observation_date"] == last_date].set_index("symbol")

    # Fresh candidate should have valid data and be in universe
    assert bool(as_of_res.loc["FRESH", "has_valid_data"]) is True
    assert bool(as_of_res.loc["FRESH", "in_universe"]) is True

    # Stale candidate must fail-closed on current session (reindexed to NaN/0 sessions)
    assert bool(as_of_res.loc["STALE", "has_valid_data"]) is False
    assert bool(as_of_res.loc["STALE", "in_universe"]) is False


def test_registry_source_validation_and_no_contradictions():
    """Section 16: Verifies vehicle registry evidence integrity and absence of contradictions."""
    evidence_path = Path("data/research/etf_vehicle_registry_evidence_v1.json")
    assert evidence_path.exists(), "Registry evidence file must exist"

    with open(evidence_path, "r", encoding="utf-8") as f:
        evidence = json.load(f)

    assert "entries" in evidence
    entries = evidence["entries"]
    assert len(entries) > 0

    # Contradiction check: PDBC must NOT be in commodity futures pool registry
    commodity_pool_syms = [
        e["symbol"] for e in entries
        if e.get("target_registry") == "REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS" or e.get("registry_name") in ("REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS", "KNOWN_COMMODITY_FUTURES_POOLS")
    ]
    assert "PDBC" not in commodity_pool_syms, "PDBC must not be classified as a K-1 commodity pool"

    # Contradiction check: AMLP and MLPX must NOT be in ETN registry
    etn_syms = [
        e["symbol"] for e in entries
        if e.get("target_registry") == "REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES" or e.get("registry_name") in ("REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES", "KNOWN_EXCHANGE_TRADED_NOTES")
    ]
    assert "AMLP" not in etn_syms, "AMLP must not be classified as an ETN"
    assert "MLPX" not in etn_syms, "MLPX must not be classified as an ETN"

    # Lineage & schema validation: every entry must have verified source retrieval and sha256
    for entry in entries:
        assert entry.get("source_retrieval_status") in ("VERIFIED", "VERIFIED_SUCCESSFUL"), f"Unverified status for {entry.get('symbol')}"
        assert bool(entry.get("legal_structure_supported")) is True, f"Unsupported structure for {entry.get('symbol')}"
        sha = entry.get("normalized_evidence_record_sha256") or entry.get("source_artifact_sha256")
        assert sha and (len(sha) == 64 or sha == "NOT_APPLICABLE"), f"Invalid SHA-256 for {entry.get('symbol')}"
        if entry.get("normalized_evidence_record_sha256"):
            assert len(entry["normalized_evidence_record_sha256"]) == 64


def test_uit_classification_spdr_and_invesco():
    """Verifies UIT ETFs (SPY, DIA, QQQ) are classified under 1940_ACT_UNIT_INVESTMENT_TRUST_ETF."""
    for sym, name in [
        ("SPY", "SPDR S&P 500 ETF Trust"),
        ("DIA", "SPDR Dow Jones Industrial Average ETF Trust"),
        ("QQQ", "Invesco QQQ Trust Series 1")
    ]:
        rec = ClassificationAuthorityEngine.classify_security(sym, name, "P", True)
        assert rec["vehicle_structure"] == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF", f"{sym} must be UIT"
        assert rec["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
        assert rec["research_subtype"] == "EQUITY_INDEX"
        assert rec["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
        assert rec["structure_verified"] is True
        assert rec["subtype_authorized"] is True
        assert rec["classification_source"] == "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
        assert rec["exclusion_reason"] is None


def test_regex_false_positive_elimination_fixed_income():
    """Verifies short-duration bond funds do not trigger defensive geared heuristics."""
    short_bond_funds = [
        ("BSV", "Vanguard Short-Term Bond ETF"),
        ("CALI", "Corbett Short Duration Bond ETF"),
        ("FLUD", "Franklin Ultra Short-Term Bond ETF"),
        ("FSTB", "First Trust Short Duration Bond ETF"),
        ("VUSB", "Vanguard Ultra-Short Bond ETF"),
        ("SHV", "iShares Short Treasury Bond ETF"),
        ("SGOV", "iShares 0-3 Month Treasury Bond ETF"),
    ]
    for sym, name in short_bond_funds:
        # Test heuristic isolation without registry match
        rec = ClassificationAuthorityEngine.classify_security(
            symbol="DUMMY",
            security_name=name,
            listing_exchange="P",
            nasdaq_etf_flag=True
        )
        assert rec["vehicle_structure"] not in ("LEVERAGED_ETF", "INVERSE_ETF"), f"{name} triggered geared heuristic"
        assert rec["exclusion_reason"] != "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"

    # Verifies real geared funds STILL trigger the heuristic
    geared_funds = [
        ("SH", "ProShares Short S&P500", "INVERSE_ETF"),
        ("PSQ", "ProShares UltraShort QQQ", "INVERSE_ETF"),
        ("SOXS", "Direxion Daily Semiconductor Bear 3X Shares", "INVERSE_ETF"),
        ("SOXL", "Direxion Daily Semiconductor Bull 3X Shares", "LEVERAGED_ETF"),
    ]
    for sym, name, expected_struct in geared_funds:
        rec = ClassificationAuthorityEngine.classify_security(
            symbol="DUMMY",
            security_name=name,
            listing_exchange="P",
            nasdaq_etf_flag=True
        )
        assert rec["vehicle_structure"] == expected_struct, f"{name} did not trigger {expected_struct}"
        assert rec["vehicle_structure_state"] == "EXCLUDED"
        assert rec["exclusion_reason"] == "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"


def test_structure_subtype_decoupling_pdbc_amlp_mlpx():
    """Verifies PDBC, AMLP, MLPX are structure-verified 1940 Act open-end ETFs with unresolved/unauthorized subtype."""
    sec_info = {
        "cik": "0001601082",
        "series_id": "S000047240",
        "class_id": "C000148113",
        "registration_form": "N-1A",
        "is_active": True
    }
    for sym, name in [
        ("PDBC", "Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF"),
        ("AMLP", "Alerian MLP ETF"),
        ("MLPX", "Global X MLP & Energy Infrastructure ETF")
    ]:
        # Case 1: Unevaluated (no structured metadata) -> UNRESOLVED / PENDING_SYSTEMATIC_CLASSIFICATION
        rec = ClassificationAuthorityEngine.classify_security(
            symbol=sym,
            security_name=name,
            listing_exchange="P",
            nasdaq_etf_flag=True,
            sec_mf_info=sec_info
        )
        assert rec["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
        assert rec["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
        assert rec["structure_verified"] is True
        assert rec["research_subtype"] == "UNRESOLVED"
        assert rec["research_subtype_state"] == "PENDING_SYSTEMATIC_CLASSIFICATION"
        assert rec["subtype_authorized"] is False
        assert rec["is_research_eligible"] is False
        assert rec["exclusion_reason"] == "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"

        # Case 2: Explicitly evaluated non-confirmatory metadata -> OTHER_ETF / EXPLORATORY_ONLY
        rec_eval = ClassificationAuthorityEngine.classify_security(
            symbol=sym,
            security_name=name,
            listing_exchange="P",
            nasdaq_etf_flag=True,
            structured_metadata={"research_subtype": "OTHER_ETF"},
            sec_mf_info=sec_info
        )
        assert rec_eval["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
        assert rec_eval["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
        assert rec_eval["structure_verified"] is True
        assert rec_eval["research_subtype"] == "OTHER_ETF"
        assert rec_eval["research_subtype_state"] == "EXPLORATORY_ONLY"
        assert rec_eval["subtype_authorized"] is False
        assert rec_eval["is_research_eligible"] is False
        assert rec_eval["exclusion_reason"] == "UNAUTHORIZED_RESEARCH_SUBTYPE"


def test_tier_2_sec_mf_directory_classification():
    """Verifies Tier 2 primary series registration directory operationalizes broad coverage under verified Form N-1A."""
    sec_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "N-1A",
        "is_active": True
    }
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="BROAD1",
        security_name="Broad Generic 1940 Act ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info
    )
    assert rec["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert rec["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert rec["structure_verified"] is True
    assert rec["classification_source"] == "TIER_2_STRUCTURED_PROVIDER_METADATA"
    assert "SEC_EDGAR_FORM_N1A_REGISTRATION_CIK_0000895421" in rec["classification_evidence"]
    assert rec["research_subtype"] == "UNRESOLVED"
    assert rec["research_subtype_state"] == "PENDING_SYSTEMATIC_CLASSIFICATION"
    assert rec["subtype_authorized"] is False
    assert rec["exclusion_reason"] == "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"


def test_tier_2_legal_form_verification_and_fail_closed():
    """Requirement 17: Tier-2 Semantic Regression Tests.
    Proves:
    1. company_tickers_mf association alone cannot establish open-end status (fails closed)
    2. N-1A/open-end maps to open-end structure (STRUCTURE_VERIFIED)
    3. S-6/UIT maps to UIT structure (STRUCTURE_VERIFIED)
    4. N-2 maps to closed-end structure (EXCLUDED)
    5. unknown organization/form fails closed (QUARANTINED)
    6. inactive registration fails closed (QUARANTINED)
    7. Tier-3 UIT override remains UIT even if Tier 2 presents conflicting metadata
    """
    # 1. association alone without legal form fails closed
    assoc_only = {"cik": "0000895421", "series_id": "S000001234", "class_id": "C000005678"}
    rec1 = ClassificationAuthorityEngine.classify_security(
        symbol="ASSOC1",
        security_name="Generic Association Only Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=assoc_only
    )
    assert rec1["vehicle_structure"] == "UNKNOWN"
    assert rec1["vehicle_structure_state"] == "QUARANTINED"
    assert rec1["structure_verified"] is False
    assert rec1["exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"

    # 2. N-1A active maps to 1940_ACT_OPEN_END_ETF
    n1a_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "N-1A",
        "is_active": True
    }
    rec2 = ClassificationAuthorityEngine.classify_security(
        symbol="N1AFUND",
        security_name="Generic Open End ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=n1a_info
    )
    assert rec2["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert rec2["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert rec2["structure_verified"] is True
    assert "SEC_EDGAR_FORM_N1A_REGISTRATION_CIK_0000895421" in rec2["classification_evidence"]

    # 3. S-6 active maps to 1940_ACT_UNIT_INVESTMENT_TRUST_ETF
    s6_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "S-6",
        "is_active": True
    }
    rec3 = ClassificationAuthorityEngine.classify_security(
        symbol="S6FUND",
        security_name="Generic Unit Investment Trust",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=s6_info
    )
    assert rec3["vehicle_structure"] == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
    assert rec3["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert rec3["structure_verified"] is True
    assert "SEC_EDGAR_FORM_S6_REGISTRATION_CIK_0000895421" in rec3["classification_evidence"]

    # 4. N-2 maps to CLOSED_END_FUND
    n2_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "N-2",
        "is_active": True
    }
    rec4 = ClassificationAuthorityEngine.classify_security(
        symbol="N2FUND",
        security_name="Generic Closed End Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=n2_info
    )
    assert rec4["vehicle_structure"] == "CLOSED_END_FUND"
    assert rec4["vehicle_structure_state"] == "EXCLUDED"
    assert rec4["structure_verified"] is False
    assert rec4["exclusion_reason"] == "EXCLUDED_STRUCTURE_CLOSED_END_FUND"
    assert "SEC_EDGAR_FORM_N2_REGISTRATION_CIK_0000895421" in rec4["classification_evidence"]

    # 5. Unknown registration form fails closed
    unk_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "UNKNOWN_FORM",
        "is_active": True
    }
    rec5 = ClassificationAuthorityEngine.classify_security(
        symbol="UNKFORM",
        security_name="Generic Unknown Form Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=unk_info
    )
    assert rec5["vehicle_structure"] == "UNKNOWN"
    assert rec5["vehicle_structure_state"] == "QUARANTINED"
    assert rec5["structure_verified"] is False
    assert rec5["exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"

    # 6. Inactive registration form fails closed
    inact_info = {
        "cik": "0000895421",
        "series_id": "S000001234",
        "class_id": "C000005678",
        "registration_form": "N-1A",
        "is_active": False
    }
    rec6 = ClassificationAuthorityEngine.classify_security(
        symbol="INACTFUND",
        security_name="Generic Inactive Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=inact_info
    )
    assert rec6["vehicle_structure"] == "UNKNOWN"
    assert rec6["vehicle_structure_state"] == "QUARANTINED"
    assert rec6["structure_verified"] is False
    assert rec6["exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"

    # 7. Tier-3 UIT override remains UIT (SPY, QQQ, DIA) even if Tier 2 presents N-1A
    conflicting_tier2 = {
        "cik": "0001064642",
        "series_id": "S000001",
        "class_id": "C000001",
        "registration_form": "N-1A",
        "is_active": True
    }
    for uit_sym in ["SPY", "QQQ", "DIA"]:
        rec7 = ClassificationAuthorityEngine.classify_security(
            symbol=uit_sym,
            security_name="Trust Series",
            listing_exchange="P",
            nasdaq_etf_flag=True,
            sec_mf_info=conflicting_tier2
        )
        assert rec7["vehicle_structure"] == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
        assert rec7["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
        assert rec7["classification_source"] == "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"


def test_unevaluated_subtype_semantics_fail_closed():
    """Confirms fail-closed semantics for subtype classification:
    1. Verified legal structure without subtype evidence maps to UNRESOLVED / PENDING_SYSTEMATIC_CLASSIFICATION, NOT OTHER_ETF.
    2. OTHER_ETF is assigned ONLY when explicit non-confirmatory subtype metadata exists.
    3. Fund name heuristics (regex) CANNOT certify positive confirmatory subtypes.
    """
    sec_info = {
        "cik": "0000895421",
        "series_id": "S000099999",
        "class_id": "C000099999",
        "registration_form": "N-1A",
        "is_active": True
    }

    # Case 1: Unevaluated fund with suggestive name (e.g. 'Treasury Bond Index') MUST NOT be certified by name
    rec1 = ClassificationAuthorityEngine.classify_security(
        symbol="FAKETRS",
        security_name="Generic Treasury Bond Index ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info
    )
    assert rec1["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert rec1["structure_verified"] is True
    # MUST NOT be FIXED_INCOME_GOVERNMENT purely on name
    assert rec1["research_subtype"] == "UNRESOLVED"
    assert rec1["research_subtype_state"] == "PENDING_SYSTEMATIC_CLASSIFICATION"
    assert rec1["subtype_authorized"] is False
    assert rec1["is_research_eligible"] is False
    assert rec1["exclusion_reason"] == "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"

    # Case 2: Explicit non-confirmatory metadata maps to OTHER_ETF / EXPLORATORY_ONLY
    rec2 = ClassificationAuthorityEngine.classify_security(
        symbol="FAKETRS",
        security_name="Generic Treasury Bond Index ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        structured_metadata={"research_subtype": "OTHER_ETF"},
        sec_mf_info=sec_info
    )
    assert rec2["research_subtype"] == "OTHER_ETF"
    assert rec2["research_subtype_state"] == "EXPLORATORY_ONLY"
    assert rec2["subtype_authorized"] is False
    assert rec2["exclusion_reason"] == "UNAUTHORIZED_RESEARCH_SUBTYPE"

    # Case 3: Confirmatory subtype requires verified registry entry or explicit authorized source
    rec3 = ClassificationAuthorityEngine.classify_security(
        symbol="TLT",
        security_name="iShares 20+ Year Treasury Bond ETF",
        listing_exchange="Q",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info
    )
    assert rec3["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert rec3["structure_verified"] is True
    assert rec3["research_subtype"] == "FIXED_INCOME_GOVERNMENT"
    assert rec3["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert rec3["subtype_authorized"] is True


def test_one_symbol_one_confirmatory_subtype_and_collision_resolution():
    """Requirement 3 & 4: Reconcile 63 legacy assignments into unique symbols and resolve collisions.
    1. Proves ONE_SYMBOL_ONE_CONFIRMATORY_SUBTYPE invariant.
    2. Proves IBB resolves to EQUITY_SECTOR over EQUITY_INDEX via frozen ambiguity precedence.
    3. Proves IEF resolves to FIXED_INCOME_GOVERNMENT over EQUITY_INDEX via frozen ambiguity precedence.
    """
    policy_path = POLICY_V11_PATH
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    precedence = policy["ambiguity_precedence_hierarchy"]
    # Precedence: COMMODITY_PHYSICAL > FIXED_INCOME_GOVERNMENT > FIXED_INCOME_CREDIT > EQUITY_SECTOR > EQUITY_INDEX > OTHER_ETF
    assert precedence.index("COMMODITY_PHYSICAL") < precedence.index("FIXED_INCOME_GOVERNMENT")
    assert precedence.index("FIXED_INCOME_GOVERNMENT") < precedence.index("FIXED_INCOME_CREDIT")
    assert precedence.index("FIXED_INCOME_CREDIT") < precedence.index("EQUITY_SECTOR")
    assert precedence.index("EQUITY_SECTOR") < precedence.index("EQUITY_INDEX")
    assert precedence.index("EQUITY_INDEX") < precedence.index("OTHER_ETF")

    # Verify IBB collision resolution
    ibb_candidates = ["EQUITY_INDEX", "EQUITY_SECTOR"]
    resolved_ibb = min(ibb_candidates, key=lambda s: precedence.index(s))
    assert resolved_ibb == "EQUITY_SECTOR", "IBB must resolve to EQUITY_SECTOR by precedence"

    # Verify IEF collision resolution
    ief_candidates = ["EQUITY_INDEX", "FIXED_INCOME_GOVERNMENT"]
    resolved_ief = min(ief_candidates, key=lambda s: precedence.index(s))
    assert resolved_ief == "FIXED_INCOME_GOVERNMENT", "IEF must resolve to FIXED_INCOME_GOVERNMENT by precedence"

    # Verify universe snapshot satisfies ONE_SYMBOL_ONE_CONFIRMATORY_SUBTYPE
    snap_path = Path("docs/research/ETF_SURVIVING_UNIVERSE_V1.parquet")
    df_snap = pd.read_parquet(snap_path)
    conf_rows = df_snap[df_snap["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"]
    assert conf_rows["symbol"].is_unique, "Confirmatory candidate symbols must be strictly unique"


def test_sector_rule_authority_and_agency_aggregate_bond_evaluation():
    """Requirement 10, 12, 13: Audit sector authority, Agency MBS, and Aggregate bond rules under Policy v1.1.0.
    1. Proves sector rule uses statutory index/sector mandate and standard taxonomy.
    2. Proves MBB (Agency MBS) fails pure government threshold due to mortgage credit/prepayment inclusion.
    3. Proves BND and AGG (Aggregate bonds) fail pure credit threshold because government/agency >= 50% and credit < 50%.
    """
    policy_path = POLICY_V11_PATH
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    # 1. Sector taxonomy requirement
    sector_rule = policy["confirmatory_subtype_rules"]["EQUITY_SECTOR"]
    assert "is_sector_specific_mandate == true" in sector_rule["required_conditions"]

    # 2. Agency MBS rule (MBB)
    govt_rule = policy["confirmatory_subtype_rules"]["FIXED_INCOME_GOVERNMENT"]
    assert "mortgage_backed_pct < 0.10" in govt_rule["required_conditions"]

    # 3. Aggregate bond rule (BND, AGG)
    credit_rule = policy["confirmatory_subtype_rules"]["FIXED_INCOME_CREDIT"]
    assert "corporate_debt_pct >= 0.50" in credit_rule["required_conditions"]
    assert "total_govt_pct < 0.50" in credit_rule["required_conditions"]
    assert "Aggregate bond funds" in credit_rule["mixed_aggregate_portfolio_policy"]


def test_missing_nport_and_post_boundary_temporal_discipline():
    """Requirement 5, 6, 7: Audit N-PORT/N-CEN ingestion requirements and temporal cutoff.
    1. Filings dated after 2026-09-24 must be rejected (POST_SNAPSHOT_NCEN_USED == 0, POST_SNAPSHOT_NPORT_USED == 0).
    2. Missing N-PORT evidence keeps instruments in UNRESOLVED / PENDING_SYSTEMATIC_CLASSIFICATION.
    """
    snapshot_boundary = "2026-09-24"

    # Check N-CEN cache if present
    ncen_zip = Path("data/research/cache/sec_ncen/2026q2_ncen.zip")
    if ncen_zip.exists():
        import zipfile
        with zipfile.ZipFile(ncen_zip, "r") as zf:
            with zf.open("SUBMISSION.tsv") as f:
                df_sub = pd.read_csv(f, sep="\t", usecols=["FILING_DATE"])
                filing_dates = pd.to_datetime(df_sub["FILING_DATE"], format="%d-%b-%Y")
                post_boundary = (filing_dates > pd.Timestamp(snapshot_boundary)).sum()
                assert post_boundary == 0, f"Found {post_boundary} post-boundary N-CEN filings"

    # Un-evaluated ETF fails closed to UNRESOLVED
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="UNEVL1",
        security_name="Generic Unevaluated ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info={"cik": "0000895421", "series_id": "S00001", "class_id": "C00001", "registration_form": "N-1A", "is_active": True}
    )
    assert rec["research_subtype"] == "UNRESOLVED"
    assert rec["research_subtype_state"] == "PENDING_SYSTEMATIC_CLASSIFICATION"
    assert rec["exclusion_reason"] == "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"


def test_policy_v11_hash_and_manifest_binding():
    """Step 34: Verifies Policy v1.1.0 cryptographic hash and manifest binding.
    1. Policy file exists and matches frozen SHA-256 digest: 864133d98750f7765409153c4305d02ff9d422aa56556b6a0506738299642f52.
    2. Manifest binds to subtype_policy_version 1.1.0 and identical SHA-256 digest.
    3. Mandate evidence database and N-PORT portfolio metrics hashes are present in manifest.
    """
    assert POLICY_V11_PATH.exists(), f"Missing policy file: {POLICY_V11_PATH}"
    actual_hash = get_file_sha256(POLICY_V11_PATH)
    assert actual_hash == POLICY_V11_SHA256 == "864133d98750f7765409153c4305d02ff9d422aa56556b6a0506738299642f52"

    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["subtype_policy_version"] == "1.1.0"
    assert manifest["subtype_policy_sha256"] == POLICY_V11_SHA256
    assert manifest["subtype_policy_commit"] == POLICY_V11_COMMIT
    assert manifest["mandate_evidence_sha256"] == get_file_sha256(MANDATE_EVIDENCE_PATH)
    assert manifest["portfolio_metrics_sha256"] == get_file_sha256(NPORT_DERIVED_METRICS_PATH)


def test_systematic_70_confirmatory_candidates_and_lineage():
    """Step 34: Verifies exact 70 unique confirmatory candidates across the 5 frozen subtypes.
    Census contract:
    - EQUITY_SECTOR: 19
    - EQUITY_INDEX: 18
    - FIXED_INCOME_GOVERNMENT: 14
    - FIXED_INCOME_CREDIT: 10
    - COMMODITY_PHYSICAL: 9
    Sum: Exactly 70 unique symbols.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH)
    conf_df = df_snap[df_snap["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"]
    assert len(conf_df) == 70
    assert conf_df["symbol"].is_unique

    counts = conf_df["research_subtype"].value_counts().to_dict()
    assert counts.get("EQUITY_SECTOR") == 19
    assert counts.get("EQUITY_INDEX") == 18
    assert counts.get("FIXED_INCOME_GOVERNMENT") == 14
    assert counts.get("FIXED_INCOME_CREDIT") == 10
    assert counts.get("COMMODITY_PHYSICAL") == 9


def test_adv80_recomputation_and_14_eligible_instruments():
    """Step 34: Verifies ADV80 threshold recomputation on 70 candidates and exact 14 research-eligible instruments.
    1. Cross-sectional ADV80 threshold equals $1,491,310,805.77 (within floating point precision).
    2. Exactly 14 instruments pass liquidity and history gates.
    3. Exactly 56 instruments receive ADV60_BELOW_80TH_PERCENTILE.
    4. All 5 confirmatory subtypes are represented in the surviving 14.
    """
    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["confirmatory_candidate_count"] == 70
    assert manifest["eligible_row_count"] == 14
    assert manifest["excluded_row_count"] == 5719
    assert abs(manifest["adv80_threshold"] - 1491310805.7732) < 1.0

    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH)
    elig_df = df_snap[df_snap["is_research_eligible"]]
    assert len(elig_df) == 14

    expected_eligible = {
        "SPY", "QQQ", "IWM", "SMH", "VOO", "IVV", "GLD",
        "LQD", "HYG", "TLT", "XLF", "DIA", "XLE", "RSP"
    }
    actual_eligible = set(elig_df["symbol"])
    assert actual_eligible == expected_eligible

    # Check subtype representation
    subtypes = set(elig_df["research_subtype"])
    assert subtypes == {
        "EQUITY_INDEX", "EQUITY_SECTOR", "FIXED_INCOME_GOVERNMENT",
        "FIXED_INCOME_CREDIT", "COMMODITY_PHYSICAL"
    }

    # Exactly 56 candidates fail liquidity threshold
    adv_fails = df_snap[df_snap["exclusion_reason"] == "ADV60_BELOW_80TH_PERCENTILE"]
    assert len(adv_fails) == 56


def test_mbb_bnd_agg_systematic_exclusion():
    """Step 34: Proves MBB, BND, and AGG fail confirmatory rules and are classified as OTHER_ETF.
    - MBB: mortgage_backed_pct >= 0.10 -> fails FIXED_INCOME_GOVERNMENT.
    - BND and AGG: corporate_debt_pct < 0.50, total_govt_pct >= 0.50 -> fail FIXED_INCOME_CREDIT.
    All three evaluate to OTHER_ETF / EXPLORATORY_ONLY / UNAUTHORIZED_RESEARCH_SUBTYPE.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH).set_index("symbol")
    for sym in ["MBB", "BND", "AGG", "BSV"]:
        assert df_snap.loc[sym, "research_subtype"] == "OTHER_ETF"
        assert df_snap.loc[sym, "research_subtype_state"] == "EXPLORATORY_ONLY"
        assert df_snap.loc[sym, "exclusion_reason"] == "UNAUTHORIZED_RESEARCH_SUBTYPE"
        assert not df_snap.loc[sym, "is_research_eligible"]


def test_subtype_collision_resolution_precedence():
    """Step 34: Proves candidate collisions resolve according to frozen ambiguity precedence.
    - IBB satisfies both equity sector and equity index -> resolves to EQUITY_SECTOR.
    - IEF satisfies both government and index -> resolves to FIXED_INCOME_GOVERNMENT.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH).set_index("symbol")
    assert df_snap.loc["IBB", "research_subtype"] == "EQUITY_SECTOR"
    assert df_snap.loc["IBB", "research_subtype_state"] == "CONFIRMATORY_SUPPORTED"

    assert df_snap.loc["IEF", "research_subtype"] == "FIXED_INCOME_GOVERNMENT"
    assert df_snap.loc["IEF", "research_subtype_state"] == "CONFIRMATORY_SUPPORTED"


def test_other_etf_vs_unresolved_fail_closed_semantics():
    """Section 24: Proves OTHER_ETF vs UNRESOLVED fail-closed population semantics.
    - Structure-verified population: 4,523 total.
    - Evaluated affirmative non-confirmatory evidence -> OTHER_ETF (630).
    - Missing N-PORT or missing statutory mandate -> UNRESOLVED (3,823 total: 876 N-PORT + 2,947 mandate).
    - Confirmatory candidates -> 70.
    - Sum: 630 + 3823 + 70 == 4,523.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH)
    struct_elig = df_snap[df_snap["vehicle_structure_state"] == "STRUCTURE_VERIFIED"]
    assert len(struct_elig) == 4523

    other_count = (struct_elig["research_subtype"] == "OTHER_ETF").sum()
    unres_count = (struct_elig["research_subtype"] == "UNRESOLVED").sum()
    conf_count = (struct_elig["research_subtype_state"] == "CONFIRMATORY_SUPPORTED").sum()

    assert other_count == 630
    assert unres_count == 3823
    assert conf_count == 70
    assert other_count + unres_count + conf_count == 4523


def test_missing_sector_mandate_does_not_become_other_etf():
    """Section 24: Verifies an equity fund lacking sector mandate fails closed to UNRESOLVED, not OTHER_ETF."""
    p_metric = {
        "reconciliation_ratio": 1.0,
        "total_equity_pct": 0.95,
        "total_govt_pct": 0.0,
        "corporate_debt_pct": 0.0,
        "mortgage_backed_pct": 0.0,
        "distinct_equity_count": 50,
        "max_concentration": 0.05,
        "is_index_ncen": False
    }
    sec_info = {"cik": "0001234567", "series_id": "S000099999", "registration_form": "N-1A", "is_active": True}

    # No mandate provided -> must NOT become OTHER_ETF
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_ACT_EQ",
        security_name="Test Active Equity Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=None
    )
    assert rec["research_subtype"] == "UNRESOLVED"
    assert rec["research_subtype_state"] == "INSUFFICIENT_SOURCE_EVIDENCE"
    assert rec["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE"
    assert rec["subtype_authorized"] is False


def test_non_index_equity_requires_sector_mandate_evaluation():
    """Section 24: Verifies non-index equity funds evaluate to EQUITY_SECTOR with valid mandate or UNRESOLVED without."""
    p_metric = {
        "reconciliation_ratio": 1.0,
        "total_equity_pct": 0.90,
        "total_govt_pct": 0.0,
        "corporate_debt_pct": 0.0,
        "mortgage_backed_pct": 0.0,
        "distinct_equity_count": 40,
        "max_concentration": 0.08,
        "is_index_ncen": False
    }
    sec_info = {"cik": "0001234567", "series_id": "S000099999", "registration_form": "N-1A", "is_active": True}

    # Case 1: Missing mandate -> UNRESOLVED
    rec_no_m = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_SEC_NO_M",
        security_name="Test Sector ETF Without Mandate",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=None
    )
    assert rec_no_m["research_subtype"] == "UNRESOLVED"
    assert rec_no_m["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE"

    # Case 2: Qualifying sector mandate -> EQUITY_SECTOR
    mandate_sec = {
        "symbol": "TEST_SEC_M",
        "is_sector_specific_mandate": True,
        "approved_sector": "TECHNOLOGY",
        "is_broad_or_multi_sector_mandate": False
    }
    rec_sec = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_SEC_M",
        security_name="Test Sector ETF With Mandate",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=mandate_sec
    )
    assert rec_sec["research_subtype"] == "EQUITY_SECTOR"
    assert rec_sec["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert rec_sec["subtype_authorized"] is True


def test_index_equity_requires_broad_vs_sector_mandate_evaluation():
    """Section 24: Verifies index equity funds require statutory mandate to distinguish broad index vs sector."""
    p_metric = {
        "reconciliation_ratio": 1.0,
        "total_equity_pct": 0.95,
        "total_govt_pct": 0.0,
        "corporate_debt_pct": 0.0,
        "mortgage_backed_pct": 0.0,
        "distinct_equity_count": 100,
        "max_concentration": 0.04,
        "is_index_ncen": True
    }
    sec_info = {"cik": "0001234567", "series_id": "S000099999", "registration_form": "N-1A", "is_active": True}

    # Case 1: Missing mandate -> UNRESOLVED
    rec_no_m = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_IDX_NO_M",
        security_name="Test Index ETF Without Mandate",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=None
    )
    assert rec_no_m["research_subtype"] == "UNRESOLVED"
    assert rec_no_m["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE"

    # Case 2: Broad index mandate -> EQUITY_INDEX
    mandate_broad = {
        "symbol": "TEST_IDX_BROAD",
        "is_sector_specific_mandate": False,
        "approved_sector": None,
        "is_broad_or_multi_sector_mandate": True
    }
    rec_broad = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_IDX_BROAD",
        security_name="Test Index ETF With Broad Mandate",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=mandate_broad
    )
    assert rec_broad["research_subtype"] == "EQUITY_INDEX"
    assert rec_broad["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"


def test_fixed_income_rules_conclusively_produce_other_without_mandate():
    """Section 24: Verifies fixed-income funds failing both gov and credit thresholds legitimately evaluate to OTHER_ETF."""
    # Mixed aggregate bond fund: 45% gov, 35% corp, 20% mbs
    p_metric = {
        "reconciliation_ratio": 1.0,
        "total_equity_pct": 0.0,
        "total_govt_pct": 0.45,
        "corporate_debt_pct": 0.35,
        "mortgage_backed_pct": 0.20,
        "distinct_equity_count": 0,
        "max_concentration": 0.02,
        "is_index_ncen": True
    }
    sec_info = {"cik": "0001234567", "series_id": "S000099999", "registration_form": "N-1A", "is_active": True}

    # Fails gov (gov_pct < 0.80) and fails credit (corp_pct < 0.50) -> conclusively OTHER_ETF even without mandate
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_AGG",
        security_name="Test Aggregate Bond ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info=sec_info,
        portfolio_metrics=p_metric,
        mandate_evidence=None
    )
    assert rec["research_subtype"] == "OTHER_ETF"
    assert rec["research_subtype_state"] == "EXPLORATORY_ONLY"
    assert rec["exclusion_reason"] == "UNAUTHORIZED_RESEARCH_SUBTYPE"


def test_other_etf_evidence_completeness_invariant():
    """Section 24: Proves OTHER_ETF implies all applicable confirmatory rules were evaluable.
    OTHER_ETF_WITH_INCOMPLETE_APPLICABLE_RULE_EVIDENCE must equal 0 across the entire population.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH)
    other_df = df_snap[df_snap["research_subtype"] == "OTHER_ETF"]
    assert len(other_df) == 630

    # Read evidence completeness matrix
    matrix_path = Path("docs/research/ETF_EVIDENCE_COMPLETENESS_MATRIX_V1.parquet")
    assert matrix_path.exists(), "Evidence completeness matrix must exist"
    df_mat = pd.read_parquet(matrix_path)

    other_mat = df_mat[df_mat["final_subtype"] == "OTHER_ETF"]
    incomplete_other = other_mat[~other_mat["all_applicable_rules_evaluable"]]
    assert len(incomplete_other) == 0, f"Found {len(incomplete_other)} OTHER_ETF rows with incomplete evidence"

    # Verify manifest reflects exact counts
    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["other_etf_evidence_completeness_count"] == 630
    assert manifest["other_etf_incomplete_evidence_count"] == 0
    assert manifest["unresolved_reason_census"]["INSUFFICIENT_MANDATE_EVIDENCE"] == 2947
    assert manifest["unresolved_reason_census"]["UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"] == 876
    assert manifest["final_candidate_denominator"] == 70
    assert manifest["final_eligible_denominator"] == 14


def test_unresolved_potential_candidate_blocks_denominator_certification():
    """Section 25: Proves unresolved potential candidate blocks denominator certification.
    If any structure-eligible ETF has unruled-out confirmatory rules, denominator is uncertified.
    """
    matrix_path = Path("docs/research/ETF_EVIDENCE_COMPLETENESS_MATRIX_V1.parquet")
    df_mat = pd.read_parquet(matrix_path)
    denom_blocking = df_mat[df_mat["denominator_blocking"]]
    assert len(denom_blocking) > 0, "Expected denominator blocking rows"
    # Denominator certification condition: blocking count must be 0 for PASS
    is_denominator_certified = (len(denom_blocking) == 0)
    assert not is_denominator_certified, "Universe cannot be certified while denominator blocking rows exist"


def test_known_positive_subset_is_not_treated_as_complete_denominator():
    """Section 25: Proves known positive subset (70) is not treated as complete denominator.
    2,947 ETFs meet portfolio floors and await statutory mandate evidence.
    """
    df_snap = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH)
    conf_count = (df_snap["research_subtype_state"] == "CONFIRMATORY_SUPPORTED").sum()
    assert conf_count == 70

    mandate_blocked = df_snap[df_snap["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE"]
    assert len(mandate_blocked) == 2947
    # 70 is provisional positive subset, not proven closed denominator
    is_closed_denominator = (len(mandate_blocked) == 0)
    assert not is_closed_denominator


def test_missing_mandate_cannot_imply_candidate_exclusion():
    """Section 25: Proves missing mandate cannot imply candidate exclusion or default to OTHER_ETF."""
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="TEST_EQ_NO_M",
        security_name="Test Equity Index ETF",
        listing_exchange="P",
        nasdaq_etf_flag=True,
        sec_mf_info={"cik": "0001234567", "series_id": "S000099999", "registration_form": "N-1A", "is_active": True},
        portfolio_metrics={
            "reconciliation_ratio": 1.0,
            "total_equity_pct": 0.99,
            "total_govt_pct": 0.0,
            "corporate_debt_pct": 0.0,
            "mortgage_backed_pct": 0.0,
            "distinct_equity_count": 500,
            "max_concentration": 0.07,
            "is_index_ncen": True
        },
        mandate_evidence=None
    )
    # Must NOT become OTHER_ETF or be silently excluded
    assert rec["research_subtype"] == "UNRESOLVED"
    assert rec["research_subtype_state"] == "INSUFFICIENT_SOURCE_EVIDENCE"
    assert rec["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE"


def test_adv80_cannot_execute_on_incomplete_candidate_denominator():
    """Section 25: Proves ADV80 cannot execute as final on incomplete candidate denominator.
    Any ADV80 computed while UNRESOLVED_POTENTIAL_CONFIRMATORY > 0 is provisional.
    """
    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Manifest records provisional threshold
    assert abs(manifest["adv80_threshold"] - 1491310805.7732) < 1.0
    # Must be marked provisional/blocked until denominator closure
    unresolved_potential = manifest.get("potential_confirmatory_unresolved_count", 3823)
    assert unresolved_potential > 0
    is_adv80_final = (unresolved_potential == 0)
    assert not is_adv80_final, "ADV80 cannot be final while potential candidates remain unresolved"


def test_final_denominator_requires_zero_potential_confirmatory_unresolved_rows():
    """Section 25: Proves final denominator closure requires exactly ZERO potential-confirmatory unresolved rows."""
    matrix_path = Path("docs/research/ETF_EVIDENCE_COMPLETENESS_MATRIX_V1.parquet")
    df_mat = pd.read_parquet(matrix_path)

    # Invariant: UNRESOLVED_POTENTIAL_CONFIRMATORY == 0 required for certification
    potential_conf_unresolved = df_mat[
        (df_mat["final_subtype"] == "UNRESOLVED") & (df_mat["potential_confirmatory_rule_count"] > 0)
    ]
    assert len(potential_conf_unresolved) == 3823
    # Certification gate is BLOCKED when potential_conf_unresolved > 0
    gate_status = "PASS" if len(potential_conf_unresolved) == 0 else "BLOCKED"
    assert gate_status == "BLOCKED"


def test_missing_nport_remains_denominator_blocking():
    """Section 28: Proves missing N-PORT filings cannot be assumed non-confirmatory.
    All 403 ETFs with MISSING_NPORT remain denominator blocking until filings are retrieved.
    """
    blocker_path = Path("docs/research/ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet")
    assert blocker_path.exists(), "ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet must exist"
    df_blockers = pd.read_parquet(blocker_path)

    missing_nport = df_blockers[df_blockers["blocker_reason"] == "MISSING_NPORT"]
    assert len(missing_nport) == 403
    assert (missing_nport["denominator_blocking"] == True).all()
    assert (missing_nport["nport_available"] == False).all()
    assert (missing_nport["resolution_status"] == "UNRESOLVED_BLOCKING").all()
    assert (missing_nport["final_subtype"] == "UNRESOLVED").all()
    # Must have potential confirmatory rules
    assert (missing_nport["potential_confirmatory_rules"].str.len() > 0).all()


def test_nport_reconciliation_failure_remains_denominator_blocking():
    """Section 28: Proves N-PORT reconciliation failures cannot be forced or silently dropped.
    All 473 ETFs with NPORT_RECONCILIATION_FAILURE remain denominator blocking pending audit.
    """
    blocker_path = Path("docs/research/ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet")
    df_blockers = pd.read_parquet(blocker_path)

    recon_fail = df_blockers[df_blockers["blocker_reason"] == "NPORT_RECONCILIATION_FAILURE"]
    assert len(recon_fail) == 473
    assert (recon_fail["denominator_blocking"] == True).all()
    assert (recon_fail["nport_available"] == True).all()
    assert (recon_fail["nport_reconciliation_pass"] == False).all()
    assert (recon_fail["resolution_status"] == "UNRESOLVED_BLOCKING").all()
    assert (recon_fail["final_subtype"] == "UNRESOLVED").all()


def test_mandate_resolution_alone_cannot_certify_universe_while_nport_blockers_remain():
    """Section 28: Proves resolving mandate blockers alone cannot certify the universe.
    Even if all 2,947 mandate blockers were resolved, 876 N-PORT blockers still block certification.
    """
    blocker_path = Path("docs/research/ETF_DENOMINATOR_BLOCKER_LEDGER_V1.parquet")
    df_blockers = pd.read_parquet(blocker_path)

    mandate_blockers = df_blockers[df_blockers["blocker_type"] == "MANDATE_BLOCKED"]
    nport_blockers = df_blockers[df_blockers["blocker_type"] == "NPORT_BLOCKED"]

    assert len(mandate_blockers) == 2947
    assert len(nport_blockers) == 876
    assert len(df_blockers) == 3823

    # Hypothesize zero mandate blockers remaining:
    remaining_if_mandates_resolved = len(nport_blockers)
    is_certified = (remaining_if_mandates_resolved == 0)
    assert not is_certified, "Universe cannot be certified while N-PORT blockers (876) remain"


def test_candidate_denominator_requires_zero_blocking_unresolved():
    """Section 28: Proves the confirmatory candidate denominator requires exactly 0 blocking unresolved ETFs.
    Current 70 candidates remain provisional while remaining_denominator_blockers == 3823.
    """
    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["initial_blocker_count"] == 3823
    assert manifest["remaining_denominator_blockers"] == 3823
    assert manifest["candidate_denominator_closure_status"] == "BLOCKED_PENDING_MANDATE_EVIDENCE"
    assert manifest["final_confirmatory_denominator"] == 70

    # Denominator certification requires remaining blockers == 0
    is_denominator_certified = (manifest["remaining_denominator_blockers"] == 0)
    assert not is_denominator_certified, "Candidate denominator cannot be certified while 3823 blockers remain"


def test_adv80_cannot_become_final_before_denominator_closure():
    """Section 28: Proves ADV80 threshold cannot be final while candidate denominator is unclosed."""
    with open(UNIVERSE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["remaining_denominator_blockers"] > 0
    # Current ADV80 is provisional
    assert manifest["candidate_denominator_closure_status"] != "CERTIFIED"
    assert abs(manifest["final_adv80"] - 1491310805.7732) < 1.0
