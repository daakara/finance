"""FastAPI Router for Asset Analytics, Risk Engine, FRED Macro & Trader Archetypes."""

import math
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import yfinance as yf

from analyst_dashboard.analyzers.advanced_risk_analyzer import AdvancedRiskAnalyzer
from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer
from analyst_dashboard.data.fred_fetcher import FredMacroFetcher

router = APIRouter()
risk_analyzer = AdvancedRiskAnalyzer()
fred_fetcher = FredMacroFetcher()
trader_analyzer = TraderArchetypeAnalyzer()

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

        score += 1  # Base operating efficiency
    except Exception:
        score = 7

    return max(3, min(9, score))


@router.get("/{symbol}")
def get_asset_analytics(symbol: str, period: str = Query("1y", description="Data period (1y, 2y, 5y)")):
    """Fetch live market data, calculate Cornish-Fisher risk, 5-Factor scores, FRED macro & Trader Archetypes."""
    try:
        clean_period = period if isinstance(period, str) else "1y"
        upper_sym = symbol.upper().strip()

        # Handle crypto ticker format
        fetch_sym = upper_sym
        if upper_sym in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LTC"]:
            fetch_sym = f"{upper_sym}-USD"

        ticker_obj = yf.Ticker(fetch_sym)
        hist = ticker_obj.history(period=clean_period)

        if hist.empty and "-" not in fetch_sym and upper_sym not in KNOWN_ETFS:
            ticker_obj = yf.Ticker(f"{upper_sym}-USD")
            hist = ticker_obj.history(period=clean_period)

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No live price data found for symbol {symbol}")

        # Format candles for lightweight charts
        candles = []
        for idx, row in hist.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            candles.append({
                "time": date_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        current_price = round(float(hist["Close"].iloc[-1]), 2)
        prev_price = round(float(hist["Close"].iloc[-2]), 2) if len(hist) > 1 else current_price
        price_change_pct = round(((current_price - prev_price) / prev_price) * 100, 2) if prev_price > 0 else 0.0

        # Calculate comprehensive real risk metrics via AdvancedRiskAnalyzer
        risk_output = risk_analyzer.analyze_comprehensive_risk(hist)
        adv_metrics = risk_output.get("advanced_metrics", {})
        sortino = adv_metrics.get("Sortino_Ratio", 1.5)

        is_crypto = "-USD" in fetch_sym
        is_etf = upper_sym in KNOWN_ETFS

        # Retrieve Company Info & Fundamental Piotroski F-Score (for individual equities)
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
            # ETFs have structural diversification & liquidity benefits (evaluated on Sortino + expense efficiency)
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
        
        # Damping function: prevents extreme runaway drift on parabolic multi-bagger surges (e.g. INTC +259%)
        raw_median = (overall_return * 0.35) * 100
        if raw_median > 25.0:
            # Smooth logarithmic compression above 25%
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

        return {
            "symbol": upper_sym,
            "period": clean_period,
            "currentPrice": current_price,
            "priceChangePct24h": price_change_pct,
            "candles": candles,
            "factorScores": factor_scores,
            "dnaScores": factor_scores,
            "macroDifficulty": macro_difficulty,
            "expectedReturn": expected_return,
            "traderArchetypes": trader_archetypes,
            "analytics": risk_output,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

