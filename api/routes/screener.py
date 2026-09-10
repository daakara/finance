"""FastAPI Router for Hidden Gems Screener with Peter Lynch, Joel Greenblatt & Disruptive Innovation Models."""

from fastapi import APIRouter, Response
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.market_db import MarketDatabaseEngine

router = APIRouter()
screener = HiddenGemsScreener()
optimal_engine = OptimalExecutionEngine()
confluence_engine = ConfluenceEngine()
smart_money_engine = SmartMoneyEngine()
market_db = MarketDatabaseEngine()

# Authentic Multi-Sector Dual-Horizon Universes (60 Total Quality Assets)
DAY_TRADER_CANDIDATES = [
    # AI & Megacap Momentum
    "NVDA", "TSLA", "PLTR", "ARM", "SMCI", "AMD", "META", "AAPL", "MSFT", "AMZN",
    # Cloud & Cybersecurity
    "CRWD", "PANW", "NET", "DDOG", "MDB",
    # Crypto & FinTech Beta
    "COIN", "MARA", "MSTR", "HOOD",
    # High-Beta Volatility & Squeeze Runners
    "DUOL", "CELH", "IONQ", "RKLB", "APP",
]

LONG_TERM_CANDIDATES = [
    # MedTech & Biotech Monopolies
    "LNTH", "CPRX", "MEDP", "TMDX", "ISRG", "VRTX", "LLY", "NVO", "DXCM", "PODD",
    # High-Moat Semiconductors & SiC Ion Implantation
    "ACLS", "POWI", "ON", "MPWR", "KLAC", "LRCX", "ASML", "AVGO",
    # Peter Lynch GARP & Organic Consumer Compounders
    "ELF", "DECK", "LULU", "ONON", "MNST", "ULTA",
    # Clean Tech, Power Infrastructure & Industrials
    "VRT", "ETN", "PWR", "GEV", "FIX", "EME",
    # Disruptive Cloud, EdTech & EDA Infrastructure
    "DUOL", "ANET", "NOW", "SNPS", "CDNS",
]

DEFAULT_CANDIDATES = LONG_TERM_CANDIDATES


class ScreenerRequest(BaseModel):
    tickers: Optional[List[str]] = None
    user_role: Optional[str] = "LONG_TERM"


@router.post("/run")
def run_screener(request: ScreenerRequest = None, response: Response = None):
    """Run the Hidden Gems Discovery Screener against Peter Lynch GARP and Greenblatt Magic Formula criteria."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
    role = request.user_role if request and request.user_role else "LONG_TERM"
    default_pool = DAY_TRADER_CANDIDATES if role == "DAY_TRADER" else LONG_TERM_CANDIDATES
    tickers = (request.tickers if request and request.tickers else default_pool)
    results = screener.evaluate_candidates(tickers)

    return {
        "total_candidates": len(tickers),
        "gems_found": len(results),
        "results": results,
    }


run_screener_post = run_screener


@router.get("/run")
def run_screener_get(
    response: Response = None,
    filter_type: str = "all",
    user_role: str = "LONG_TERM",
    custom_tickers: Optional[str] = None,
):
    """GET endpoint supporting live screener execution, archetype filtering, and on-demand custom ticker scanning."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"

    is_day_trader = (user_role == "DAY_TRADER")

    # On-demand custom watchlist input support
    if custom_tickers and custom_tickers.strip():
        parsed = [t.strip().upper() for t in custom_tickers.replace(",", " ").split() if t.strip()]
        active_universe = parsed if parsed else (DAY_TRADER_CANDIDATES if is_day_trader else LONG_TERM_CANDIDATES)
    else:
        active_universe = DAY_TRADER_CANDIDATES if is_day_trader else LONG_TERM_CANDIDATES

    results = screener.evaluate_candidates(active_universe)

    # Map candidate fields with live optimal execution levels
    mapped_candidates = []
    for r in results:
        raw_sym = r.get("ticker") or r.get("symbol")
        if not raw_sym or not str(raw_sym).strip():
            continue
        sym = str(raw_sym).strip().upper()
        roic_val = r.get("roic_pct")
        margin_val = r.get("gross_margin_pct")

        # Retrieve current price and candles from market database
        latest_info = market_db.get_latest_price(sym)
        current_price = latest_info.get("currentPrice") if (latest_info and latest_info.get("currentPrice") and latest_info["currentPrice"] > 0) else None

        if current_price is None or current_price <= 0:
            current_price = None
            execution = {
                "current_price": None,
                "optimal_entry_min": None,
                "optimal_entry_max": None,
                "stop_loss": None,
                "stop_loss_pct": None,
                "take_profit_1": None,
                "take_profit_1_pct": None,
                "take_profit_2": None,
                "take_profit_2_pct": None,
                "risk_reward_ratio": None,
                "execution_status": "UNVERIFIED_ASSET",
                "setup_pattern": "Unverified Asset Setup",
                "entry_thesis": "Pricing and exchange tape unavailable. Trade levels suppressed.",
                "invalidation_condition": "Awaiting market data.",
                "stage_phase": "Unverified Asset",
                "vcp_contraction_status": "Unverified",
                "breakout_pivot": None,
                "atr_14": None,
                "liquidity_defense": None,
            }
        else:
            db_candles = market_db.get_daily_candles(sym, limit=60)
            if db_candles:
                hist_df = pd.DataFrame([{
                    "Open": c["open"], "High": c["high"], "Low": c["low"], "Close": c["close"], "Volume": c["volume"]
                } for c in db_candles], index=pd.to_datetime([c["time"] for c in db_candles]))
            else:
                hist_df = pd.DataFrame()

            execution = optimal_engine.calculate_trade_levels(hist_df, current_price, user_role=user_role)

        entry_min = execution["optimal_entry_min"]
        entry_max = execution["optimal_entry_max"]
        stop_loss = execution["stop_loss"]
        tp1 = execution["take_profit_1"]
        tp2 = execution["take_profit_2"]
        rr_ratio = execution["risk_reward_ratio"]
        setup_pat = execution["setup_pattern"]
        entry_th = execution["entry_thesis"]
        atr_14 = execution.get("atr_14")

        # Pure Mathematical Execution State Determination
        if execution.get("execution_status") == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
            execution_status = "UNVERIFIED_ASSET"
            status_label = "⚠️ Unverified Asset"
            status_color = "slate"
        elif execution.get("execution_status") == "INSUFFICIENT_HISTORY" or stop_loss is None:
            execution_status = "INSUFFICIENT_HISTORY"
            status_label = "⏳ Insufficient History"
            status_color = "cyan"
        elif execution.get("stage_phase") == "Stage 4 Markdown (Awaiting New Base)":
            execution_status = "WAITING_PULLBACK"
            status_label = "⏳ Awaiting Base Formation"
            status_color = "cyan"
        elif stop_loss is not None and current_price is not None and current_price < stop_loss:
            execution_status = "STOPPED_OUT"
            status_label = "🛑 Below Stop Loss"
            status_color = "rose"
        elif execution.get("execution_status") == "APPROACHING_TARGET" or (tp1 is not None and current_price is not None and current_price >= tp1 * 0.96):
            execution_status = "APPROACHING_TARGET"
            status_label = "🚀 Session ORB Breakout" if is_day_trader else "🚀 Near TP Target"
            status_color = "amber"
        elif entry_min is not None and entry_max is not None and current_price is not None and (entry_min <= current_price <= entry_max * 1.008 or abs(current_price - entry_max) / max(0.01, current_price) <= 0.015):
            execution_status = "IN_BUY_ZONE"
            status_label = "🎯 Active VWAP Bounce" if is_day_trader else "🎯 Active Buy Zone"
            status_color = "emerald"
        else:
            execution_status = "WAITING_PULLBACK"
            status_label = "⏳ Pullback Pending"
            status_color = "cyan"

        # Liquidity Guard (Shadow Observation Mode): Informational execution metadata only; does not mutate frozen decision state
        liq_def = execution.get("liquidity_defense")

        # Deterministic Smart Money & Catalyst Attributes
        if execution_status == "UNVERIFIED_ASSET" or current_price is None or current_price <= 0:
            confluence_res = {
                "confluenceScore": 0.0,
                "confluenceRating": "Unverified Asset / No Market Data",
                "badgeColor": "slate",
                "reasons": ["Pricing, tape, and regulatory filings unavailable for unverified security."],
                "warnings": ["Do not trade unverified assets. Trade levels suppressed."],
            }
            rvol_val = "N/A"
            short_float_val = "N/A"
        elif execution_status == "INSUFFICIENT_HISTORY":
            confluence_res = {
                "confluenceScore": 0.0,
                "confluenceRating": "Insufficient History (< 50 Bars)",
                "badgeColor": "cyan",
                "reasons": ["Asset has fewer than 50 historical trading sessions on exchange record."],
                "warnings": ["Trade levels and risk geometry suppressed until seasoning threshold met."],
            }
            rvol_val = "N/A"
            short_float_val = "N/A"
        else:
            sec_trades = smart_money_engine.get_sec_insider_trades(sym)
            cong_trades = smart_money_engine.get_congressional_trades(sym)
            has_insider = len(sec_trades) > 0
            has_congress = len(cong_trades) > 0
            insider_val = 0.0
            if has_insider:
                for t in sec_trades:
                    val_raw = t.get("total_value") or t.get("transaction_value") or 0.0
                    if isinstance(val_raw, str):
                        cleaned = val_raw.replace("$", "").replace(",", "").strip()
                        try:
                            insider_val += float(cleaned)
                        except ValueError:
                            pass
                    elif isinstance(val_raw, (int, float)):
                        insider_val += float(val_raw)
            days_to_earn = r.get("days_to_earnings")

            smart_data = {
                "has_insider_buy": has_insider,
                "insider_value_usd": insider_val if has_insider else 0.0,
                "insider_name": sec_trades[0].get("reporting_owner", "") if has_insider else "",
                "has_congress_buy": has_congress,
                "has_options_flow": False,
            } if (has_insider or has_congress) else None

            # Compute multi-factor confluence conviction score
            confluence_res = confluence_engine.calculate_confluence(
                symbol=sym,
                technical_data={
                    "executionStatus": execution_status,
                    "riskRewardRatio": rr_ratio,
                    "setup_pattern": setup_pat,
                    "stage_phase": "Stage 2 Breakout" if execution_status == "IN_BUY_ZONE" else "Institutional Accumulation",
                    "rsi_14": execution.get("rsi_14"),
                    "stop_loss": stop_loss,
                    "current_price": current_price,
                },
                smart_money_data=smart_data,
                fundamental_data={
                    "qualityScore": float(r.get("quality_score")) if r.get("quality_score") is not None else None,
                    "growthScore": float(r.get("growth_score")) if r.get("growth_score") is not None else None,
                    "valuationScore": float(r.get("valuation_score")) if r.get("valuation_score") is not None else None,
                    "piotroski_f": int(r.get("piotroski_f")) if r.get("piotroski_f") is not None else None,
                    "roic": roic_val,
                    "peg": float(r.get("peg_ratio")) if r.get("peg_ratio") is not None else None,
                } if any(r.get(k) is not None for k in ["quality_score", "growth_score", "valuation_score", "piotroski_f", "roic_pct", "peg_ratio"]) else None,
                catalyst_data={
                    "days_to_earnings": days_to_earn,
                } if days_to_earn is not None else None,
                macro_data=None,  # No fabricated yield curve 0.25 / credit spread 3.5
            )

            if not hist_df.empty and len(hist_df) >= 5 and "Volume" in hist_df.columns:
                recent_vol = float(hist_df["Volume"].iloc[-1])
                avg_vol = float(hist_df["Volume"].tail(20).mean())
                if avg_vol > 0:
                    rvol_val = f"{round(recent_vol / avg_vol, 1)}x"
                else:
                    rvol_val = "1.0x"
            else:
                rvol_val = "1.0x"

            short_float_num = r.get("short_float_pct")
            if short_float_num is not None:
                short_float_val = f"{short_float_num}%"
            else:
                short_float_val = "N/A"

        mapped_candidates.append({
            "symbol": sym,
            "companyName": r.get("company_name", sym),
            "currentPrice": current_price,
            "gemScore": int(r.get("composite_score")) if r.get("composite_score") is not None else (0 if execution_status == "UNVERIFIED_ASSET" else None),
            "expertArchetype": r.get("expert_model") or ("Unverified Asset" if execution_status == "UNVERIFIED_ASSET" else ("High-Beta Momentum Leader" if is_day_trader else "Peter Lynch GARP Compounder")),
            "roic": f"{roic_val}%" if roic_val is not None else "N/A",
            "pegRatio": str(r.get("peg_ratio")) if r.get("peg_ratio") is not None else ("0.0" if execution_status == "UNVERIFIED_ASSET" else "N/A"),
            "grossMargin": f"{margin_val}%" if margin_val is not None else "N/A",
            "atr14": f"${atr_14:.2f}" if (atr_14 is not None and execution_status not in ["UNVERIFIED_ASSET", "INSUFFICIENT_HISTORY"]) else "N/A",
            "rvol": rvol_val,
            "shortFloat": short_float_val,
            "dayTraderSetup": "Pricing and intraday VWAP tape unavailable." if execution_status == "UNVERIFIED_ASSET" else "Intraday momentum trend continuation above 5m VWAP anchor with defined ATR risk.",
            "thesis": r.get("investment_thesis", "Disclosures and market data unavailable for unverified security." if execution_status == "UNVERIFIED_ASSET" else ("High relative volume momentum with clear intraday VWAP risk definition." if is_day_trader else "High return on capital with strong free cash flows.")),
            "catalyst": r.get("primary_catalyst", "Awaiting verified disclosures." if execution_status == "UNVERIFIED_ASSET" else ("Intraday institutional flow breakout." if is_day_trader else "Product cycle expansion and margin gains.")),
            "riskLevel": "Unverified Risk" if execution_status == "UNVERIFIED_ASSET" else ("High Volatility (Intraday)" if is_day_trader else r.get("risk_rating", "Low-to-Medium Risk")),
            # Execution Scanner Levels
            "executionStatus": execution_status,
            "statusLabel": status_label,
            "statusColor": status_color,
            "optimalEntryMin": entry_min,
            "optimalEntryMax": entry_max,
            "stopLoss": stop_loss,
            "stopLossPct": round(((stop_loss - current_price) / current_price) * 100, 1) if (stop_loss is not None and current_price > 0) else None,
            "takeProfit1": tp1,
            "takeProfit1Pct": round(((tp1 - current_price) / current_price) * 100, 1) if (tp1 is not None and current_price > 0) else None,
            "takeProfit2": tp2,
            "takeProfit2Pct": round(((tp2 - current_price) / current_price) * 100, 1) if (tp2 is not None and current_price > 0) else None,
            "riskRewardRatio": rr_ratio,
            "setupPattern": setup_pat,
            "entryThesis": entry_th,
            # Confluence Conviction Score & Position Sizing
            "confluenceScore": confluence_res["confluenceScore"],
            "confluenceRating": confluence_res["confluenceRating"],
            "confluenceBadgeColor": confluence_res["badgeColor"],
            "confluenceReasons": confluence_res["reasons"],
            "confluenceWarnings": confluence_res["warnings"],
            "liquidityDefense": liq_def,
        })

    # Apply Selected Filter dynamically based on numerical thresholds
    if filter_type in ["in_buy_zone", "vwap_pullback"]:
        filtered = [c for c in mapped_candidates if c["executionStatus"] == "IN_BUY_ZONE"]
    elif filter_type in ["approaching_target", "orb_breakout"]:
        filtered = [c for c in mapped_candidates if c["executionStatus"] == "APPROACHING_TARGET"]
    elif filter_type == "high_rr":
        filtered = [
            c for c in mapped_candidates
            if c.get("riskRewardRatio") is not None and c["riskRewardRatio"] >= 2.0 and (c["executionStatus"] == "IN_BUY_ZONE" or (c.get("optimalEntryMax") is not None and c["currentPrice"] <= c["optimalEntryMax"] * 1.02))
        ]
    elif filter_type == "high_confluence":
        filtered = [c for c in mapped_candidates if c["confluenceScore"] >= 80.0 and c["executionStatus"] != "UNVERIFIED_ASSET"]
    elif filter_type == "high_rvol":
        filtered = [c for c in mapped_candidates if c["rvol"] != "N/A" and float(c["rvol"].replace("x", "")) >= 2.5]
    elif filter_type == "squeeze":
        filtered = [c for c in mapped_candidates if c["shortFloat"] != "N/A" and float(c["shortFloat"].replace("%", "")) >= 6.0]
    elif filter_type == "lynch":
        filtered = [c for c in mapped_candidates if c["executionStatus"] != "UNVERIFIED_ASSET" and ((0 < float(c["pegRatio"]) <= 1.05) or "Lynch" in c["expertArchetype"] or "GARP" in c["expertArchetype"])]
    elif filter_type == "greenblatt":
        filtered = [c for c in mapped_candidates if c["executionStatus"] != "UNVERIFIED_ASSET" and (float(c["roic"].replace("%", "")) >= 28.0 or "Greenblatt" in c["expertArchetype"] or "Magic" in c["expertArchetype"])]
    elif filter_type == "rule_breakers":
        filtered = [c for c in mapped_candidates if c["executionStatus"] != "UNVERIFIED_ASSET" and (float(c["grossMargin"].replace("%", "")) >= 65.0 or "Rule Breakers" in c["expertArchetype"] or "Disruptive" in c["expertArchetype"])]
    else:
        filtered = mapped_candidates

    return {
        "totalCandidates": len(active_universe),
        "gemsFound": len(filtered),
        "activeFilter": filter_type,
        "userRole": user_role,
        "candidates": filtered,
        "results": results,
    }


@router.get("/position-size")
def calculate_trade_position_size(
    account_equity: float = 25000.0,
    risk_pct: float = 1.0,
    entry_price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit_1: float = 105.0,
):
    """Interactive Position Sizer calculating exact share quantities, risk limit, and Kelly allocation."""
    return ConfluenceEngine.calculate_position_size(
        account_equity=account_equity,
        risk_pct=risk_pct,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
    )


@router.get("/phase26/validation-report")
def get_phase26_validation_report():
    """Returns Phase 26 prospective liquidity validation report covering Experiment 26-A and 26-B."""
    from analyst_dashboard.governance.liquidity_validation import Phase26ValidationEngine
    return Phase26ValidationEngine.generate_phase26_validation_report()
