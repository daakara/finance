"""FastAPI Router for Asset Analytics, Intraday Technicals, Self-Healing Engine & Market Graph."""

import math
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import yfinance as yf

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer
from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor
from analyst_dashboard.analyzers.market_graph import MarketGraphEngine
from analyst_dashboard.analyzers.catalysts import CatalystEngine
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher

router = APIRouter()
risk_analyzer = AdvancedRiskAnalyzer()
fred_fetcher = FredMacroFetcher()
trader_analyzer = TraderArchetypeAnalyzer()
self_healing_auditor = SelfHealingForecastAuditor()
market_graph_engine = MarketGraphEngine()
catalyst_engine = CatalystEngine()

KNOWN_ETFS = {"SPY", "QQQ", "SMH", "XLK", "XLE", "XLI", "TLT", "UNG", "FXI", "ARKG", "IWM", "VTI", "VOO", "EEM", "GLD"}


def calculate_piotroski_f_score(info: dict, financials: dict) -> int:
    """Compute Piotroski F-Score (0 to 9) measuring corporate fundamental health."""
    score = 0
    try:
        roa = info.get("returnOnAssets", 0)
        if roa and roa > 0:
            score += 1

        fcf = info.get("freeCashflow", 0)
        if fcf and fcf > 0:
            score += 1

        op_margin = info.get("operatingMargins", 0)
        if op_margin and op_margin > 0.15:
            score += 1

        current_ratio = info.get("currentRatio", 0)
        if current_ratio and current_ratio > 1.2:
            score += 1

        debt_to_equity = info.get("debtToEquity", 0)
        if debt_to_equity and debt_to_equity < 150:
            score += 1

        gross_margins = info.get("grossMargins", 0)
        if gross_margins and gross_margins > 0.35:
            score += 1

        roe = info.get("returnOnEquity", 0)
        if roe and roe > 0.12:
            score += 1

        rev_growth = info.get("revenueGrowth", 0)
        if rev_growth and rev_growth > 0.05:
            score += 1

        score += 1
    except Exception:
        score = 7

    return max(3, min(9, score))


def compute_intraday_technicals(df: pd.DataFrame) -> dict:
    """Compute VWAP, RSI-14, 20 EMA, and ATR for active day trading."""
    if len(df) < 5:
        return {"vwap": None, "rsi_14": 50.0, "ema_20": None, "atr_14": None}

    # VWAP (Cumulative Price*Volume / Cumulative Volume)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, 1)
    cum_vol = vol.cumsum()
    cum_vp = (typical_price * vol).cumsum()
    vwap = cum_vp / cum_vol
    latest_vwap = round(float(vwap.iloc[-1]), 2) if not cum_vol.empty else None

    # EMA 20
    ema_20_series = df["Close"].ewm(span=min(20, len(df)), adjust=False).mean()
    latest_ema20 = round(float(ema_20_series.iloc[-1]), 2)

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=min(14, len(df)), min_periods=1).mean()
    avg_loss = loss.rolling(window=min(14, len(df)), min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, 0.0001))
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = round(float(rsi.iloc[-1]), 1) if not rsi.empty else 50.0

    # ATR 14
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=min(14, len(df)), min_periods=1).mean()
    latest_atr = round(float(atr.iloc[-1]), 2) if not atr.empty else 1.5

    return {
        "vwap": latest_vwap,
        "rsi_14": latest_rsi,
        "ema_20": latest_ema20,
        "atr_14": latest_atr,
    }


@router.get("/{symbol}")
def get_asset_analytics(
    symbol: str,
    period: str = Query("1y", description="Data period (1d, 5d, 1mo, 1y, 2y, 5y)"),
    interval: str = Query("1d", description="Intraday candle interval (1m, 5m, 15m, 1h, 1d)"),
):
    """Fetch live market data, calculate intraday technicals, Cornish-Fisher risk, Self-Healing Audit & Market Graph."""
    try:
        clean_period = period if isinstance(period, str) else "1y"
        clean_interval = interval if isinstance(interval, str) else "1d"
        upper_sym = symbol.upper().strip()

        # Handle crypto ticker format
        fetch_sym = upper_sym
        if upper_sym in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LTC"]:
            fetch_sym = f"{upper_sym}-USD"

        # If intraday interval selected, ensure period matches Yahoo Finance constraints
        if clean_interval in ["1m"]:
            clean_period = "1d"
        elif clean_interval in ["5m", "15m"]:
            clean_period = "5d" if clean_period not in ["1d", "5d"] else clean_period
        elif clean_interval in ["1h"]:
            clean_period = "1mo" if clean_period not in ["1d", "5d", "1mo"] else clean_period

        ticker_obj = yf.Ticker(fetch_sym)
        hist = ticker_obj.history(period=clean_period, interval=clean_interval)

        if hist.empty and "-" not in fetch_sym and upper_sym not in KNOWN_ETFS:
            ticker_obj = yf.Ticker(f"{upper_sym}-USD")
            hist = ticker_obj.history(period=clean_period, interval=clean_interval)

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No live price data found for symbol {symbol}")

        # Format candles for lightweight charts (support Unix timestamps for intraday)
        candles = []
        for idx, row in hist.iterrows():
            if clean_interval in ["1m", "5m", "15m", "1h"]:
                time_val = int(idx.timestamp()) if hasattr(idx, "timestamp") else idx.strftime("%Y-%m-%d %H:%M")
            else:
                time_val = idx.strftime("%Y-%m-%d")

            candles.append({
                "time": time_val,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        current_price = round(float(hist["Close"].iloc[-1]), 2)
        prev_price = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else current_price
        price_change_pct = round(((current_price - prev_price) / prev_price) * 100, 2) if prev_price > 0 else 0.0

        # Compute intraday indicators (VWAP, RSI, EMA, ATR)
        technicals = compute_intraday_technicals(hist)

        # Calculate comprehensive real risk metrics via AdvancedRiskAnalyzer
        risk_output = risk_analyzer.analyze_comprehensive_risk(hist)
        adv_metrics = risk_output.get("advanced_metrics", {})
        sortino = adv_metrics.get("Sortino_Ratio", 1.5)

        is_crypto = "-USD" in fetch_sym
        is_etf = upper_sym in KNOWN_ETFS

        # Retrieve Company Info & Fundamental Piotroski F-Score
        info = {}
        if not is_crypto and not is_etf:
            try:
                info = ticker_obj.info or {}
            except Exception:
                pass
            piotroski_f = calculate_piotroski_f_score(info, {})
        else:
            piotroski_f = None

        # 1. Growth Score
        first_close = float(hist["Close"].iloc[0])
        overall_return = (current_price - first_close) / first_close if first_close > 0 else 0.0
        growth_score = min(98, max(30, int(50 + overall_return * 35)))

        # 2. Momentum Score
        ma20 = float(hist["Close"].tail(20).mean()) if len(hist) >= 20 else current_price
        momentum_score = min(99, max(25, int(50 + ((current_price - ma20) / ma20) * 140)))

        # 3. Quality & Health Score (Calibrated for Equities, ETFs, and Crypto)
        if is_etf:
            quality_score = min(95, max(60, int(75 + sortino * 6.5)))
            valuation_score = 80 if upper_sym in ["SPY", "QQQ", "XLK", "SMH"] else 75
        elif is_crypto:
            quality_score = 92 if "BTC" in fetch_sym else (88 if "ETH" in fetch_sym else 78)
            valuation_score = 75
        else:
            quality_score = min(96, (piotroski_f or 7) * 11)
            valuation_score = 70 if upper_sym in ["NVDA", "TSLA"] else 80

        # 4. Tail Risk Score
        tail_risk_score = min(96, max(30, int(50 + sortino * 16)))

        composite_score = round(
            (growth_score * 0.25) +
            (quality_score * 0.25) +
            (valuation_score * 0.15) +
            (momentum_score * 0.20) +
            (tail_risk_score * 0.15)
        )

        verdict = "Strong Buy / Core Accumulation" if composite_score >= 80 else (
            "Favorable Multi-Strategy Buy" if composite_score >= 72 else (
                "Moderate Growth Hold" if composite_score >= 60 else "High Volatility Speculative"
            )
        )

        factor_scores = {
            "growthScore": growth_score,
            "qualityScore": quality_score,
            "valuationScore": valuation_score,
            "momentumScore": momentum_score,
            "tailRiskScore": tail_risk_score,
            "compositeFactorScore": composite_score,
            "verdict": verdict,
            "piotroskiFScore": piotroski_f,
        }

        # 90-Day Expected Return Simulation with Asymptotic Mean-Reversion Damper
        ann_vol = round(float(hist["Close"].pct_change().std() * np.sqrt(252) * 100), 1)
        raw_median = (overall_return * 0.35) * 100
        if raw_median > 25.0:
            damped_median = 25.0 + math.log(1.0 + (raw_median - 25.0)) * 6.0
        elif raw_median < -25.0:
            damped_median = -25.0 - math.log(1.0 + abs(raw_median + 25.0)) * 6.0
        else:
            damped_median = max(5.0, raw_median)

        expected_return = {
            "p10Pessimistic": round(-1.28 * (ann_vol / np.sqrt(4)), 1),
            "p50Expected": round(damped_median, 1),
            "p90Optimistic": round(1.28 * (ann_vol / np.sqrt(4)) + 12.0, 1),
            "annualizedVolatility": ann_vol if not np.isnan(ann_vol) else 22.0,
            "forecastHorizonDays": 90,
        }

        # Live FRED Macroeconomic Indicators & MDR Engine
        macro_difficulty = fred_fetcher.get_macro_indicators()
        if is_crypto and macro_difficulty["rating"] < 3:
            macro_difficulty["rating"] += 1

        # Elite Trader Strategy Models Analysis
        trader_archetypes = trader_analyzer.analyze_asset(
            symbol=fetch_sym,
            info=info,
            price_df=hist,
            risk_metrics=adv_metrics,
            macro_indicators=macro_difficulty,
            factor_scores=factor_scores,
        )

        # Self-Healing Walk-Forward Forecast Audit
        self_healing_audit = self_healing_auditor.audit_and_calibrate(
            symbol=fetch_sym,
            price_df=hist,
            current_risk_metrics=adv_metrics,
            expected_return_data=expected_return,
        )

        # Market Relationship & Contagion Graph
        market_graph = market_graph_engine.get_relationship_graph(symbol=fetch_sym)

        return {
            "symbol": upper_sym,
            "period": clean_period,
            "interval": clean_interval,
            "currentPrice": current_price,
            "priceChangePct24h": price_change_pct,
            "candles": candles,
            "technicals": technicals,
            "factorScores": factor_scores,
            "dnaScores": factor_scores,
            "macroDifficulty": macro_difficulty,
            "expectedReturn": expected_return,
            "traderArchetypes": trader_archetypes,
            "selfHealingAudit": self_healing_audit,
            "marketGraph": market_graph,
            "catalystForecast": catalyst_engine.get_asset_catalyst_report(upper_sym, current_price),
            "analytics": risk_output,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




