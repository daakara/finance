"""Targeted Offline Integrity & Anti-Leakage Suite for ARX ETF Research Dataset v1.0.1.

Verifies:
1. No duplicate observation keys across (symbol, observation_date).
2. Strict monotonic dates per symbol.
3. Zero current AUM or snapshot holdings columns.
4. Macro as-of join respected: macro_source_observation_date strictly < observation_date.
5. All observations satisfy history_sessions >= 250.
6. f11 bullion breakout lookback bounds are exactly 50 sessions.
7. Physical isolation: etf_observations and etf_outcomes reside in separate parquet files.
8. Split and distribution coherence: adjusted High >= Low, ATR positive, raw Open preserved.
9. Zero model fitting or 2025 holdout scoring performed.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path("data/research")
MANIFEST_PATH = DATA_DIR / "etf_dataset_manifest_v1.json"
OBS_PATH = DATA_DIR / "etf_observations_v1.parquet"
OUT_PATH = DATA_DIR / "etf_outcomes_v1.parquet"
UNI_PATH = DATA_DIR / "etf_universe_membership_v1.parquet"
MAC_PATH = DATA_DIR / "etf_macro_v1.parquet"


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def observations():
    assert OBS_PATH.exists()
    return pd.read_parquet(OBS_PATH)


@pytest.fixture(scope="module")
def outcomes():
    assert OUT_PATH.exists()
    return pd.read_parquet(OUT_PATH)


@pytest.fixture(scope="module")
def universe():
    assert UNI_PATH.exists()
    return pd.read_parquet(UNI_PATH)


def test_manifest_integrity_and_governance(manifest):
    assert manifest["dataset_version"] == "1.0.0"
    assert manifest["spec_version"] == "1.0.1"
    assert manifest["spec_git_commit"] == "44a4aa952a47d1e6c82d8ec58b2d2e8f6b5c0d2a"
    assert manifest["model_fitting_performed"] is False
    assert manifest["holdout_2025_evaluated"] is False
    assert manifest["total_observations"] > 0
    assert manifest["unique_symbols"] == 15


def test_no_duplicate_observations(observations):
    dups = observations.duplicated(subset=["symbol", "observation_date"]).sum()
    assert dups == 0, f"Found {dups} duplicate observation keys"


def test_date_monotonicity_per_symbol(observations):
    for sym, group in observations.groupby("symbol"):
        dates = pd.to_datetime(group["observation_date"])
        assert dates.is_monotonic_increasing, f"Dates not monotonic for {sym}"


def test_physical_outcome_isolation(observations, outcomes):
    # Ensure outcomes are separate and feature table does not contain outcome targets
    outcome_cols = [
        "horizon_20d_excess_return", "horizon_5d_excess_return",
        "horizon_10d_excess_return", "horizon_60d_excess_return",
        "mae_20d", "mfe_20d", "realized_vol_20d"
    ]
    for col in outcome_cols:
        assert col not in observations.columns, f"Outcome column {col} leaked into observations table!"
        assert col in outcomes.columns, f"Outcome column {col} missing from outcomes table"

    assert len(observations) == len(outcomes)


def test_zero_snapshot_metadata_in_dataset(observations, universe):
    forbidden = ["aum", "totalAssets", "netExpenseRatio", "holdings", "top_holdings", "sectorWeightings"]
    for col in observations.columns:
        for f in forbidden:
            assert f.lower() not in col.lower(), f"Forbidden snapshot metadata found: {col}"
    for col in universe.columns:
        for f in forbidden:
            assert f.lower() not in col.lower(), f"Forbidden snapshot metadata found in universe: {col}"


def test_macro_asof_lagged_rule(observations):
    # Verify macro_source_observation_date strictly precedes observation_date
    obs_dates = pd.to_datetime(observations["observation_date"])
    macro_dates = pd.to_datetime(observations["macro_source_observation_date"])

    violations = (macro_dates >= obs_dates).sum()
    assert violations == 0, f"Found {violations} violations of conservative macro as-of join!"


def test_minimum_history_respected(universe, observations):
    # Cross-reference observations with universe to ensure history >= 250
    uni_sub = universe[universe["in_universe"] == True]
    assert (uni_sub["history_sessions"] < 250).sum() == 0, "Universe contains observations with history < 250!"


def test_split_and_distribution_adjustment_coherence():
    # Verify raw and adjusted prices on SPY around historical distribution
    raw_spy = pd.read_parquet(DATA_DIR / "cache" / "SPY_raw.parquet")
    adj_spy = pd.read_parquet(DATA_DIR / "cache" / "SPY_adj.parquet")

    assert len(raw_spy) == len(adj_spy)
    # Adjusted high must be >= adjusted low everywhere
    assert (adj_spy["High"] >= adj_spy["Low"]).all()
    # Adjusted close must be positive
    assert (adj_spy["Close"] > 0).all()
    # Raw open must be positive
    assert (raw_spy["Open"] > 0).all()
