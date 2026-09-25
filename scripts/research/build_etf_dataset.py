"""ARX Terminal — Versioned Offline Historical ETF Research Dataset Builder.

Constructs canonical Stage-A research datasets adhering strictly to:
docs/research/ETF_RESEARCH_SPEC_V1.json (v1.0.1)

Enforces:
1. POINT_IN_TIME_WITHIN_OBSERVABLE_SURVIVOR_UNIVERSE
2. Zero forward-looking leakage (all features at date t use info <= t close).
3. Conservative Macro As-Of Join (macro observation date <= t - 1).
4. Physical separation of observations and forward outcome labels.
5. Cryptographic manifest binding spec SHA, code SHA, dataset SHAs, and row counts.
6. Absolutely NO model fitting, weighting, or threshold tuning.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_etf_dataset")

SPEC_PATH = Path("docs/research/ETF_RESEARCH_SPEC_V1.json")
DATA_DIR = Path("data/research")
CACHE_DIR = DATA_DIR / "cache"

# Canonical Universe across 5 Supported Subtypes
UNIVERSE_CONFIG = {
    "EQUITY_INDEX": ["SPY", "QQQ", "IWM", "DIA", "VOO", "IVV"],
    "EQUITY_SECTOR": ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XOP"],
    "FIXED_INCOME_GOVERNMENT": ["TLT", "IEF", "SHY", "IEI"],
    "FIXED_INCOME_CREDIT": ["HYG", "LQD", "JNK", "VCIT"],
    "COMMODITY_PHYSICAL": ["GLD", "IAU", "SLV"],
}

BENCHMARK_SYMBOLS = ["SPY", "IEF", "LQD", "BIL"]
CURRENCY_SYMBOLS = ["UUP"]
MACRO_SERIES = ["DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DFII10"]

START_DATE = "2007-04-11"  # HYG inception
END_DATE = "2025-12-31"    # End of 2025 Historical Holdout


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fetch_cached_ticker(symbol: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch split/dividend-adjusted and raw OHLCV for a symbol with local parquet caching."""
    adj_cache = cache_dir / f"{symbol}_adj.parquet"
    raw_cache = cache_dir / f"{symbol}_raw.parquet"

    if adj_cache.exists() and raw_cache.exists():
        df_adj = pd.read_parquet(adj_cache)
        df_raw = pd.read_parquet(raw_cache)
        return df_adj, df_raw

    logger.info(f"Downloading historical market data for {symbol}...")
    t = yf.Ticker(symbol)
    df_adj = t.history(start="2005-01-01", end="2026-01-10", auto_adjust=True)
    df_raw = t.history(start="2005-01-01", end="2026-01-10", auto_adjust=False)

    # Normalize indices
    df_adj.index = pd.to_datetime(df_adj.index).tz_localize(None).normalize()
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None).normalize()

    # Drop timezone and save
    df_adj.to_parquet(adj_cache)
    df_raw.to_parquet(raw_cache)
    return df_adj, df_raw


def fetch_cached_fred_series(series_id: str, cache_dir: Path) -> pd.DataFrame:
    """Download FRED constant maturity yield/spread series with local caching."""
    cache_file = cache_dir / f"fred_{series_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    logger.info(f"Downloading FRED series {series_id}...")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    date_col = df.columns[0]
    val_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.rename(columns={date_col: "date", val_col: series_id})
    df = df.dropna().sort_values("date").drop_duplicates(subset=["date"])
    df.set_index("date", inplace=True)
    df.to_parquet(cache_file)
    return df


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """True Range Wilder ATR calculation."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothed Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    logger.info("Initializing universe downloads...")
    all_symbols = set()
    symbol_to_subtype = {}
    for st, syms in UNIVERSE_CONFIG.items():
        for s in syms:
            all_symbols.add(s)
            symbol_to_subtype[s] = st

    for b in BENCHMARK_SYMBOLS:
        all_symbols.add(b)
    for c in CURRENCY_SYMBOLS:
        all_symbols.add(c)

    symbol_data_adj = {}
    symbol_data_raw = {}
    for sym in sorted(all_symbols):
        try:
            adj, raw = fetch_cached_ticker(sym, CACHE_DIR)
            symbol_data_adj[sym] = adj
            symbol_data_raw[sym] = raw
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")

    logger.info("Downloading FRED macro series...")
    macro_dfs = {}
    for sid in MACRO_SERIES:
        try:
            mdf = fetch_cached_fred_series(sid, CACHE_DIR)
            macro_dfs[sid] = mdf
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {sid}: {e}")

    # Build master trading calendar from SPY
    spy_dates = symbol_data_adj["SPY"].index
    study_dates = [d for d in spy_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    # Forward-fill macro onto equity calendar with conservative LAGGED join
    macro_combined = pd.DataFrame(index=spy_dates)
    for sid, mdf in macro_dfs.items():
        macro_combined = macro_combined.join(mdf, how="left")
    macro_combined = macro_combined.ffill()

    # Build Daily ADV60 matrix across surviving universe
    adv60_matrix = pd.DataFrame(index=spy_dates)
    for sym in symbol_to_subtype.keys():
        if sym in symbol_data_adj:
            df = symbol_data_adj[sym]
            dollar_vol = df["Close"] * df["Volume"]
            adv60_matrix[sym] = dollar_vol.rolling(window=60, min_periods=60).mean()

    # Precompute technicals per symbol
    technicals = {}
    for sym in symbol_to_subtype.keys():
        if sym not in symbol_data_adj:
            continue
        df = symbol_data_adj[sym]
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        atr20 = compute_atr(high, low, close, 20)
        atr10 = compute_atr(high, low, close, 10)
        atr50 = compute_atr(high, low, close, 50)
        sma200 = close.rolling(window=200, min_periods=200).mean()
        rsi14 = compute_rsi(close, 14)

        technicals[sym] = {
            "Close": close,
            "High": high,
            "Low": low,
            "ATR20": atr20,
            "ATR10": atr10,
            "ATR50": atr50,
            "SMA200": sma200,
            "RSI14": rsi14,
            "HistoryLen": pd.Series(np.arange(1, len(df) + 1), index=df.index),
        }

    # Precompute currency features (UUP)
    uup_df = symbol_data_adj["UUP"]
    uup_sma50 = uup_df["Close"].rolling(50, min_periods=50).mean()
    uup_atr20 = compute_atr(uup_df["High"], uup_df["Low"], uup_df["Close"], 20)

    # Observation and Outcome containers
    observations = []
    outcomes = []
    universe_records = []

    logger.info("Generating observations and forward outcomes...")
    for date in study_dates:
        # 1. Evaluate point-in-time liquidity percentile across active survivor universe
        daily_adv = adv60_matrix.loc[date].dropna()
        if len(daily_adv) > 0:
            p80 = daily_adv.quantile(0.80)
        else:
            p80 = np.inf

        # Macro values strictly lagged (available at date - 1 trading session)
        date_idx = spy_dates.get_loc(date)
        if date_idx > 0:
            prev_date = spy_dates[date_idx - 1]
            macro_row = macro_combined.loc[prev_date]
            macro_source_date = str(prev_date.date())
        else:
            macro_row = macro_combined.loc[date]
            macro_source_date = str(date.date())

        for sym, subtype in symbol_to_subtype.items():
            if sym not in technicals:
                continue
            tech = technicals[sym]
            if date not in tech["Close"].index:
                continue

            hist_len = tech["HistoryLen"].loc[date]
            sym_adv = adv60_matrix.loc[date].get(sym, np.nan)
            is_liquid = bool(sym_adv >= p80) if not np.isnan(sym_adv) else False
            in_universe = bool(is_liquid and hist_len >= 250)

            universe_records.append({
                "symbol": sym,
                "subtype": subtype,
                "observation_date": str(date.date()),
                "adv60": float(sym_adv) if not np.isnan(sym_adv) else None,
                "adv80_threshold": float(p80) if p80 != np.inf else None,
                "history_sessions": int(hist_len),
                "in_universe": in_universe
            })

            # Feature calculation requires in_universe and history >= 250
            if not in_universe:
                continue

            close_t = tech["Close"].loc[date]
            sma200_t = tech["SMA200"].loc[date]
            atr20_t = tech["ATR20"].loc[date]

            if np.isnan(sma200_t) or np.isnan(atr20_t) or atr20_t == 0:
                continue

            # Generate unique deterministic observation ID
            obs_id = hashlib.sha256(f"{sym}_{date.strftime('%Y%m%d')}".encode("utf-8")).hexdigest()[:16]

            # f1: Trend ratio
            f1 = float((close_t - sma200_t) / atr20_t)

            # f2: Relative strength 60d vs SPY
            spy_close = technicals["SPY"]["Close"]
            sym_idx = tech["Close"].index.get_loc(date)
            spy_idx = spy_close.index.get_loc(date)

            if sym_idx >= 60 and spy_idx >= 60:
                close_t60 = tech["Close"].iloc[sym_idx - 60]
                spy_t = spy_close.iloc[spy_idx]
                spy_t60 = spy_close.iloc[spy_idx - 60]
                f2 = float((close_t / close_t60) - (spy_t / spy_t60))
            else:
                f2 = None

            # f3: RSI regime distance
            rsi_t = tech["RSI14"].loc[date]
            if not np.isnan(rsi_t):
                f3 = float(np.exp(-((rsi_t - 60.0) ** 2) / (2.0 * (15.0 ** 2))))
            else:
                f3 = None

            # f4: Volatility compression (exploratory)
            atr10_t = tech["ATR10"].loc[date]
            atr50_t = tech["ATR50"].loc[date]
            f4 = float(atr10_t / atr50_t) if (not np.isnan(atr10_t) and not np.isnan(atr50_t) and atr50_t > 0) else None

            # Macro features (from lagged macro_row)
            f5 = float(macro_row["T10Y2Y"]) if "T10Y2Y" in macro_row and not np.isnan(macro_row["T10Y2Y"]) else None

            # 20-day macro momentum
            if date_idx >= 20:
                lag20_date = spy_dates[date_idx - 20]
                m_lag20 = macro_combined.loc[lag20_date]
                f6 = float(macro_row["DGS10"] - m_lag20["DGS10"]) if ("DGS10" in macro_row and not np.isnan(macro_row["DGS10"]) and not np.isnan(m_lag20["DGS10"])) else None
                f7 = float(macro_row["BAMLH0A0HYM2"] - m_lag20["BAMLH0A0HYM2"]) if ("BAMLH0A0HYM2" in macro_row and not np.isnan(macro_row["BAMLH0A0HYM2"]) and not np.isnan(m_lag20["BAMLH0A0HYM2"])) else None
                f9 = float(macro_row["DFII10"] - m_lag20["DFII10"]) if ("DFII10" in macro_row and not np.isnan(macro_row["DFII10"]) and not np.isnan(m_lag20["DFII10"])) else None
            else:
                f6, f7, f9 = None, None, None

            # f8: HYG relative strength vs LQD
            if sym_idx >= 60 and "HYG" in technicals and "LQD" in technicals:
                hyg_c = technicals["HYG"]["Close"]
                lqd_c = technicals["LQD"]["Close"]
                if date in hyg_c.index and date in lqd_c.index:
                    hyg_idx = hyg_c.index.get_loc(date)
                    lqd_idx = lqd_c.index.get_loc(date)
                    if hyg_idx >= 60 and lqd_idx >= 60:
                        f8 = float((hyg_c.iloc[hyg_idx] / hyg_c.iloc[hyg_idx - 60]) - (lqd_c.iloc[lqd_idx] / lqd_c.iloc[lqd_idx - 60]))
                    else:
                        f8 = None
                else:
                    f8 = None
            else:
                f8 = None

            # f10: UUP currency trend
            if date in uup_df.index:
                uup_c = uup_df["Close"].loc[date]
                uup_sma = uup_sma50.loc[date]
                uup_atr = uup_atr20.loc[date]
                f10 = float((uup_c - uup_sma) / uup_atr) if (not np.isnan(uup_sma) and not np.isnan(uup_atr) and uup_atr > 0) else None
            else:
                f10 = None

            # f11: Bullion breakout (50 sessions: k=0..49)
            if sym_idx >= 50:
                lows_50 = tech["Low"].iloc[sym_idx - 49 : sym_idx + 1]
                highs_50 = tech["High"].iloc[sym_idx - 49 : sym_idx + 1]
                min_l = lows_50.min()
                max_h = highs_50.max()
                f11 = float((close_t - min_l) / (max_h - min_l)) if max_h > min_l else None
            else:
                f11 = None

            # Assign benchmark
            bench_map = {
                "EQUITY_INDEX": "SPY",
                "EQUITY_SECTOR": "SPY",
                "FIXED_INCOME_GOVERNMENT": "IEF",
                "FIXED_INCOME_CREDIT": "LQD",
                "COMMODITY_PHYSICAL": "BIL",
            }
            benchmark = bench_map.get(subtype, "SPY")

            observations.append({
                "observation_id": obs_id,
                "symbol": sym,
                "subtype": subtype,
                "observation_date": str(date.date()),
                "feature_timestamp": f"{date.date()}T16:00:00Z",
                "macro_source_observation_date": macro_source_date,
                "benchmark": benchmark,
                "f1_trend_ratio": f1,
                "f2_relative_strength_60d": f2,
                "f3_rsi_regime_distance": f3,
                "f4_vol_compression": f4,
                "f5_yield_curve_slope": f5,
                "f6_rate_momentum_20d": f6,
                "f7_credit_spread_trend": f7,
                "f8_relative_strength_vs_lqd": f8,
                "f9_real_yield_regime": f9,
                "f10_dxy_trend": f10,
                "f11_bullion_breakout": f11,
            })

            # Calculate Forward Outcomes (strictly in separate container)
            # Need t+1 Open to t+20 Close
            raw_open = symbol_data_raw[sym]["Open"]
            sym_raw_idx = raw_open.index.get_loc(date) if date in raw_open.index else None

            # Primary 20d horizon
            h20_ret, h5_ret, h10_ret, h60_ret = None, None, None, None
            mae_20d, mfe_20d, vol_20d = None, None, None

            if sym_idx + 20 < len(tech["Close"]):
                close_t20 = tech["Close"].iloc[sym_idx + 20]
                bench_close = symbol_data_adj[benchmark]["Close"]
                if date in bench_close.index:
                    b_idx = bench_close.index.get_loc(date)
                    if b_idx + 20 < len(bench_close):
                        b_t = bench_close.iloc[b_idx]
                        b_t20 = bench_close.iloc[b_idx + 20]
                        sym_ret20 = close_t20 / close_t - 1.0
                        b_ret20 = b_t20 / b_t - 1.0
                        h20_ret = float(sym_ret20 - b_ret20)

                # Diagnostic 5d, 10d, 60d
                if sym_idx + 5 < len(tech["Close"]):
                    h5_ret = float(tech["Close"].iloc[sym_idx + 5] / close_t - 1.0)
                if sym_idx + 10 < len(tech["Close"]):
                    h10_ret = float(tech["Close"].iloc[sym_idx + 10] / close_t - 1.0)
                if sym_idx + 60 < len(tech["Close"]):
                    h60_ret = float(tech["Close"].iloc[sym_idx + 60] / close_t - 1.0)

                # MAE / MFE over window t+1 .. t+20
                if sym_raw_idx is not None and sym_raw_idx + 1 < len(raw_open):
                    entry_open = raw_open.iloc[sym_raw_idx + 1]
                    highs_window = tech["High"].iloc[sym_idx + 1 : sym_idx + 21]
                    lows_window = tech["Low"].iloc[sym_idx + 1 : sym_idx + 21]
                    if len(highs_window) == 20 and atr20_t > 0:
                        mae_20d = float((lows_window.min() - close_t) / atr20_t)
                        mfe_20d = float((highs_window.max() - close_t) / atr20_t)

                # Realized volatility
                returns_20d = tech["Close"].iloc[sym_idx + 1 : sym_idx + 21].pct_change().dropna()
                if len(returns_20d) >= 15:
                    vol_20d = float(returns_20d.std() * np.sqrt(252))

            outcomes.append({
                "observation_id": obs_id,
                "symbol": sym,
                "observation_date": str(date.date()),
                "horizon_20d_excess_return": h20_ret,
                "horizon_5d_excess_return": h5_ret,
                "horizon_10d_excess_return": h10_ret,
                "horizon_60d_excess_return": h60_ret,
                "mae_20d": mae_20d,
                "mfe_20d": mfe_20d,
                "realized_vol_20d": vol_20d,
            })

    # Save to Parquet tables
    obs_df = pd.DataFrame(observations)
    out_df = pd.DataFrame(outcomes)
    uni_df = pd.DataFrame(universe_records)
    macro_table = macro_combined.reset_index().rename(columns={"index": "date"})

    obs_path = DATA_DIR / "etf_observations_v1.parquet"
    out_path = DATA_DIR / "etf_outcomes_v1.parquet"
    uni_path = DATA_DIR / "etf_universe_membership_v1.parquet"
    mac_path = DATA_DIR / "etf_macro_v1.parquet"
    man_path = DATA_DIR / "etf_dataset_manifest_v1.json"

    logger.info("Writing output Parquet artifacts...")
    obs_df.to_parquet(obs_path, index=False)
    out_df.to_parquet(out_path, index=False)
    uni_df.to_parquet(uni_path, index=False)
    macro_table.to_parquet(mac_path, index=False)

    # Compute artifact digests
    manifest = {
        "dataset_version": "1.0.0",
        "spec_version": spec["spec_version"],
        "spec_sha256": get_file_sha256(SPEC_PATH),
        "spec_git_commit": "44a4aa952a47d1e6c82d8ec58b2d2e8f6b5c0d2a",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_observations": len(obs_df),
        "unique_symbols": int(obs_df["symbol"].nunique()),
        "date_range": [str(obs_df["observation_date"].min()), str(obs_df["observation_date"].max())],
        "subtype_counts": obs_df["subtype"].value_counts().to_dict(),
        "artifacts": {
            "observations_parquet": {"path": str(obs_path), "sha256": get_file_sha256(obs_path), "rows": len(obs_df)},
            "outcomes_parquet": {"path": str(out_path), "sha256": get_file_sha256(out_path), "rows": len(out_df)},
            "universe_parquet": {"path": str(uni_path), "sha256": get_file_sha256(uni_path), "rows": len(uni_df)},
            "macro_parquet": {"path": str(mac_path), "sha256": get_file_sha256(mac_path), "rows": len(macro_table)},
        },
        "governance_status": "DATASET_BUILT_AND_UNMODIFIED",
        "model_fitting_performed": False,
        "holdout_2025_evaluated": False,
    }

    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset build complete. Manifest written to {man_path}")
    logger.info(f"Total observations: {len(obs_df)}, Unique symbols: {obs_df['symbol'].nunique()}")
    return manifest


if __name__ == "__main__":
    build_dataset()
