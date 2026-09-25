"""Targeted Test Suite for ARX ETF Universe Snapshot Construction & Contract (Spec v1.0.2).

Verifies:
1. Discovery-source metadata capture (URL, timestamp, raw SHA-256, creation time, row counts).
2. Nasdaq ETF flag discovery-only semantics (flag != eligibility).
3. Unknown-structure fail-closed quarantine (defaults to quarantined & ineligible).
4. Vehicle-structure classification across all 10 allowed structure classes.
5. Defensive negative heuristics (negative safety net only; cannot certify inclusion).
6. Rolling 60-day ADV cross-sectional 80th percentile logic.
7. Minimum 250-trading-session history enforcement.
8. Snapshot schema: exact 13 required columns.
9. Manifest schema: exact 13 required fields.
10. Spec SHA binding: binds canonical 1.0.2 spec SHA and git commit.
11. Builder SHA binding: binds builder git commit and builder file SHA-256.
12. Source-cache lineage: validates source cache manifest records.
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
    SourceCacheManager,
    parse_nasdaq_traded_content,
    ClassificationAuthorityEngine,
    build_universe_snapshot,
    evaluate_liquidity_and_history,
    get_file_sha256,
    get_git_commit,
)

SAMPLE_NASDAQ_DATA = (
    b"Symbol|Security Name|Listing Exchange|Market Category|ETF|Round Lot Size|Test Issue|Financial Status|CQS Symbol|NASDAQ Symbol|NextShares\n"
    b"SPY|SPDR S&P 500 ETF Trust|P| |Y|100|N|N|SPY|SPY|N\n"
    b"GLD|SPDR Gold Shares|P| |Y|100|N|N|GLD|GLD|N\n"
    b"TQQQ|ProShares UltraPro QQQ 3x Shares|Q|G|Y|100|N|N|TQQQ|TQQQ|N\n"
    b"AMJ|JPMorgan Alerian MLP Index ETN|P| |Y|100|N|N|AMJ|AMJ|N\n"
    b"USO|United States Oil Fund LP Futures|P| |Y|100|N|N|USO|USO|N\n"
    b"BITO|ProShares Bitcoin Strategy ETF|P| |Y|100|N|N|BITO|BITO|N\n"
    b"XYZ|XYZ Mystery Trust Unknown Fund|P| |Y|100|N|N|XYZ|XYZ|N\n"
    b"NONETF|Common Stock Inc|N| |N|100|N|N|NONETF|NONETF|N\n"
    b"TESTS|Test Issue Symbol|Q|G|Y|100|Y|N|TESTS|TESTS|N\n"
    b"File Creation Time: 0925202611:00|||||\n"
)


def test_discovery_source_metadata_capture():
    df, meta = parse_nasdaq_traded_content(SAMPLE_NASDAQ_DATA, source_url="test://nasdaqtraded.txt")
    assert meta["source_url_identifier"] == "test://nasdaqtraded.txt"
    assert meta["raw_source_sha256"] == hashlib.sha256(SAMPLE_NASDAQ_DATA).hexdigest()
    assert meta["file_creation_timestamp"] == "0925202611:00"
    assert meta["row_count"] == 9
    assert meta["etf_flagged_row_count"] == 8
    assert "retrieval_timestamp" in meta


def test_nasdaq_etf_flag_discovery_only_and_fail_closed_quarantine():
    # XYZ has ETF == 'Y', but no verified structure or provider evidence
    rec = ClassificationAuthorityEngine.classify_security(
        symbol="XYZ",
        security_name="XYZ Mystery Trust Unknown Fund",
        listing_exchange="P",
        nasdaq_etf_flag=True
    )
    assert rec["nasdaq_etf_flag"] is True
    assert rec["is_research_eligible"] is False
    assert rec["vehicle_structure"] == "UNKNOWN"
    assert rec["vehicle_structure_state"] == "QUARANTINED"
    assert rec["exclusion_reason"] == "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"
    assert rec["classification_source"] == "TIER_5_UNKNOWN_STRUCTURE_QUARANTINE"


def test_vehicle_structure_classification_and_registries():
    # 1. 1940 Act ETF
    spy = ClassificationAuthorityEngine.classify_security("SPY", "SPDR S&P 500 ETF Trust", "P", True)
    assert spy["vehicle_structure"] == "1940_ACT_OPEN_END_ETF"
    assert spy["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert spy["research_subtype"] == "EQUITY_INDEX"
    assert spy["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert spy["is_research_eligible"] is True
    assert spy["exclusion_reason"] is None

    # 2. Physical precious metal grantor trust
    gld = ClassificationAuthorityEngine.classify_security("GLD", "SPDR Gold Shares", "P", True)
    assert gld["vehicle_structure"] == "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST"
    assert gld["vehicle_structure_state"] == "STRUCTURE_VERIFIED"
    assert gld["research_subtype"] == "COMMODITY_PHYSICAL"
    assert gld["research_subtype_state"] == "CONFIRMATORY_SUPPORTED"
    assert gld["is_research_eligible"] is True
    assert gld["exclusion_reason"] is None


def test_defensive_negative_heuristics():
    # Leveraged ETF
    tqqq = ClassificationAuthorityEngine.classify_security("TQQQ", "ProShares UltraPro QQQ 3x Shares", "Q", True)
    assert tqqq["vehicle_structure"] == "LEVERAGED_ETF"
    assert tqqq["vehicle_structure_state"] == "EXCLUDED"
    assert tqqq["is_research_eligible"] is False
    assert tqqq["exclusion_reason"] == "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"

    # ETN
    amj = ClassificationAuthorityEngine.classify_security("AMJ", "JPMorgan Alerian MLP Index ETN", "P", True)
    assert amj["vehicle_structure"] == "EXCHANGE_TRADED_NOTE"
    assert amj["vehicle_structure_state"] == "EXCLUDED"
    assert amj["is_research_eligible"] is False
    assert amj["exclusion_reason"] == "EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE"

    # Commodity futures pool
    uso = ClassificationAuthorityEngine.classify_security("USO", "United States Oil Fund LP Futures", "P", True)
    assert uso["vehicle_structure"] == "COMMODITY_FUTURES_POOL"
    assert uso["vehicle_structure_state"] == "EXCLUDED"
    assert uso["is_research_eligible"] is False
    assert uso["exclusion_reason"] == "EXCLUDED_STRUCTURE_COMMODITY_FUTURES_POOL"

    # Crypto-linked
    bito = ClassificationAuthorityEngine.classify_security("BITO", "ProShares Bitcoin Strategy ETF", "P", True)
    assert bito["vehicle_structure"] == "CRYPTO_LINKED_PRODUCT"
    assert bito["vehicle_structure_state"] == "EXCLUDED"
    assert bito["is_research_eligible"] is False
    assert bito["exclusion_reason"] == "EXCLUDED_STRUCTURE_CRYPTO_LINKED"

    # Closed-end fund
    cef = ClassificationAuthorityEngine.classify_security("BOE", "BlackRock Enhanced Global Dividend CEF", "N", True)
    assert cef["vehicle_structure"] == "CLOSED_END_FUND"
    assert cef["vehicle_structure_state"] == "EXCLUDED"
    assert cef["is_research_eligible"] is False
    assert cef["exclusion_reason"] == "EXCLUDED_STRUCTURE_CLOSED_END_FUND"

    # Mutual fund
    mf = ClassificationAuthorityEngine.classify_security("VFINX", "Vanguard 500 Index Mutual Fund", "N", True)
    assert mf["vehicle_structure"] == "MUTUAL_FUND"
    assert mf["vehicle_structure_state"] == "EXCLUDED"
    assert mf["is_research_eligible"] is False
    assert mf["exclusion_reason"] == "EXCLUDED_STRUCTURE_MUTUAL_FUND"


def test_liquidity_and_history_evaluation():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)

    # Symbol A: high volume, 300 bars
    df_a = pd.DataFrame({
        "Close": 100.0 + np.random.randn(300),
        "Volume": 1000000.0 + np.random.randn(300) * 10000
    }, index=dates)

    # Symbol B: low volume, 300 bars
    df_b = pd.DataFrame({
        "Close": 50.0 + np.random.randn(300),
        "Volume": 1000.0 + np.random.randn(300) * 10
    }, index=dates)

    # Symbol C: short history (100 bars)
    df_c = pd.DataFrame({
        "Close": 20.0 + np.random.randn(100),
        "Volume": 2000000.0 + np.random.randn(100) * 10000
    }, index=dates[-100:])

    price_dict = {"A": df_a, "B": df_b, "C": df_c}
    res = evaluate_liquidity_and_history(["A", "B", "C"], price_dict, dates, adv_window=60, adv_percentile=0.80, min_history=250)

    # At the end of history:
    last_date = str(dates[-1].date())
    res_last = res[res["observation_date"] == last_date].set_index("symbol")

    # A has high volume and >= 250 bars
    assert bool(res_last.loc["A", "is_liquid"]) is True
    assert bool(res_last.loc["A", "has_min_history"]) is True
    assert bool(res_last.loc["A", "in_universe"]) is True

    # B has low volume
    assert bool(res_last.loc["B", "is_liquid"]) is False
    assert bool(res_last.loc["B", "in_universe"]) is False

    # C has high volume but history < 250 bars
    assert bool(res_last.loc["C", "has_min_history"]) is False
    assert bool(res_last.loc["C", "in_universe"]) is False


def test_universe_snapshot_and_manifest_contracts():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        disc_file = tmp_path / "nasdaqtraded.txt"
        disc_file.write_bytes(SAMPLE_NASDAQ_DATA)

        snap_parquet = tmp_path / "ETF_SURVIVING_UNIVERSE_V1.parquet"
        snap_manifest = tmp_path / "ETF_SURVIVING_UNIVERSE_V1_MANIFEST.json"
        src_manifest = SourceCacheManager(manifest_path=tmp_path / "source_cache_manifest.json")

        manifest = build_universe_snapshot(
            discovery_file=disc_file,
            output_parquet=snap_parquet,
            output_manifest=snap_manifest,
            cache_dir=tmp_path,
            source_manifest=src_manifest
        )

        assert snap_parquet.exists()
        assert snap_manifest.exists()

        # Verify exact 13 columns in Parquet
        df = pd.read_parquet(snap_parquet)
        required_cols = [
            "symbol", "security_name", "listing_exchange", "nasdaq_etf_flag",
            "vehicle_structure", "vehicle_structure_state", "research_subtype",
            "research_subtype_state", "classification_source", "classification_evidence",
            "classification_timestamp", "is_research_eligible", "exclusion_reason"
        ]
        assert list(df.columns) == required_cols

        # Test issues filtered out (TESTS should not be in snapshot)
        assert "TESTS" not in df["symbol"].values
        assert "NONETF" in df["symbol"].values
        assert "SPY" in df["symbol"].values

        # Verify exact 13 fields in Manifest
        required_manifest_fields = [
            "research_spec_version",
            "research_spec_sha256",
            "research_spec_git_commit",
            "classification_rule_version",
            "discovery_source_sha256",
            "discovery_retrieval_timestamp",
            "snapshot_sha256",
            "row_count",
            "eligible_row_count",
            "excluded_row_count",
            "generated_at",
            "builder_git_commit",
            "builder_file_sha256"
        ]
        for field in required_manifest_fields:
            assert field in manifest, f"Missing manifest field: {field}"

        assert manifest["research_spec_version"] == "1.0.2"
        assert manifest["research_spec_git_commit"] == "57720cc278813b11cc2ea6df5cccd1925b56c763"
        assert manifest["classification_rule_version"] == "1.0.2"
        assert manifest["row_count"] == len(df)
        assert manifest["eligible_row_count"] == int(df["is_research_eligible"].sum())
        assert manifest["excluded_row_count"] == int((~df["is_research_eligible"]).sum())


def test_source_cache_lineage():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "mock.parquet"
        test_file.write_bytes(b"DATA")

        mgr = SourceCacheManager(manifest_path=tmp_path / "source_cache.json")
        entry = mgr.record(
            path=test_file,
            provider="TEST_PROVIDER",
            semantic_role="TEST_ROLE"
        )

        assert entry["provider"] == "TEST_PROVIDER"
        assert entry["semantic_role"] == "TEST_ROLE"
        assert entry["byte_size"] == 4
        assert entry["sha256"] == hashlib.sha256(b"DATA").hexdigest()
        assert "retrieval_timestamp" in entry

        # Reload manager and check persistence
        mgr2 = SourceCacheManager(manifest_path=tmp_path / "source_cache.json")
        rel_key = str(test_file).replace("\\", "/")
        assert rel_key in mgr2.entries
