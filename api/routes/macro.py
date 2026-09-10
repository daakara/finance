"""FastAPI Router for Macro Ribbon and Global Market Telemetry.

Provides genuine, un-fabricated market observations with explicit per-field availability,
observation timestamps, exchange session calculation via exchange_calendars (XNYS),
and failure isolation.
"""

import logging
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
from fastapi import APIRouter, Response
import yfinance as yf
import exchange_calendars as xcals

logger = logging.getLogger("api.macro")
router = APIRouter()

_nyse_calendar = None


def _get_nyse_calendar():
    global _nyse_calendar
    if _nyse_calendar is None:
        _nyse_calendar = xcals.get_calendar("XNYS")
    return _nyse_calendar


def compute_vix_tier(vix_level: Optional[float]) -> str:
    """Compute VIX volatility tier with correct ordering and explicit boundary handling."""
    if vix_level is None:
        return "UNAVAILABLE"
    if vix_level >= 30.0:
        return "CRITICAL"
    if vix_level >= 20.0:
        return "ELEVATED"
    return "NORMAL"


def get_nyse_session_status(dt: Optional[datetime] = None) -> Dict[str, Any]:
    """Derive NYSE market session from authoritative exchange calendar (XNYS).
    
    Handles:
    - Authoritative holidays via exchange_calendars (XNYS)
    - Early closes (e.g. 13:00 close day after Thanksgiving, Christmas Eve)
    - Weekends
    - Pre-market trading (04:00 - open)
    - Regular trading hours (open - close)
    - Post-market trading (close - 20:00)
    - Overnight closed (20:00 - 04:00)
    """
    try:
        if dt is None:
            now_et = datetime.now(ZoneInfo("America/New_York"))
        elif dt.tzinfo is None:
            now_et = dt.replace(tzinfo=ZoneInfo("America/New_York"))
        else:
            now_et = dt.astimezone(ZoneInfo("America/New_York"))

        date_str = now_et.date().isoformat()
        cal = _get_nyse_calendar()

        is_session = cal.is_session(date_str)
        if not is_session:
            weekday = now_et.weekday()
            reason = "Weekend" if weekday >= 5 else "NYSE Market Holiday / Non-Trading Day"
            return {
                "status": "CLOSED",
                "reason": reason,
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": False,
            }

        # Valid trading session: extract schedule open and close
        sched = cal.schedule.loc[date_str]
        mkt_open = sched["open"].tz_convert("America/New_York")
        mkt_close = sched["close"].tz_convert("America/New_York")

        open_time = mkt_open.time()
        close_time = mkt_close.time()
        curr_time = now_et.time()

        is_early_close = close_time < time(16, 0)
        early_note = f" (Early Close: {close_time.strftime('%H:%M')} ET)" if is_early_close else ""

        if curr_time < time(4, 0):
            return {
                "status": "CLOSED",
                "reason": "Overnight Closed (20:00 - 04:00 ET)",
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": True,
                "marketOpen": mkt_open.strftime("%H:%M ET"),
                "marketClose": mkt_close.strftime("%H:%M ET"),
            }
        elif time(4, 0) <= curr_time < open_time:
            return {
                "status": "PRE_MARKET",
                "reason": f"Pre-Market Trading (04:00 - {open_time.strftime('%H:%M')} ET)",
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": True,
                "marketOpen": mkt_open.strftime("%H:%M ET"),
                "marketClose": mkt_close.strftime("%H:%M ET"),
            }
        elif open_time <= curr_time < close_time:
            return {
                "status": "OPEN",
                "reason": f"Regular Trading Hours ({open_time.strftime('%H:%M')} - {close_time.strftime('%H:%M')} ET){early_note}",
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": True,
                "marketOpen": mkt_open.strftime("%H:%M ET"),
                "marketClose": mkt_close.strftime("%H:%M ET"),
            }
        elif close_time <= curr_time < time(20, 0):
            return {
                "status": "POST_MARKET",
                "reason": f"Post-Market Trading ({close_time.strftime('%H:%M')} - 20:00 ET)",
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": True,
                "marketOpen": mkt_open.strftime("%H:%M ET"),
                "marketClose": mkt_close.strftime("%H:%M ET"),
            }
        else:
            return {
                "status": "CLOSED",
                "reason": "Overnight Closed (20:00 - 04:00 ET)",
                "timezone": "America/New_York",
                "sessionTime": now_et.strftime("%H:%M:%S ET"),
                "isSession": True,
                "marketOpen": mkt_open.strftime("%H:%M ET"),
                "marketClose": mkt_close.strftime("%H:%M ET"),
            }
    except Exception as e:
        logger.error(f"Error calculating NYSE session status: {e}")
        return {
            "status": "UNAVAILABLE",
            "reason": f"Session calendar computation error: {str(e)}",
            "timezone": "America/New_York",
            "sessionTime": None,
            "isSession": None,
        }


def _fetch_isolated_bar(symbol: str, is_yield: bool = False) -> Dict[str, Any]:
    """Fetch daily bar for a single ticker with failure isolation and zero fabricated fallbacks."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", interval="1d")
        if hist.empty or len(hist) < 2:
            return {
                "symbol": symbol,
                "available": False,
                "status": "UNAVAILABLE",
                "price": None,
                "change": None,
                "changePct": None,
                "provider": "yfinance",
                "observationTime": None,
                "error": f"Insufficient historical bars returned for {symbol}",
            }

        p = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        chg = p - prev
        chg_pct = (chg / prev) * 100 if prev != 0 else 0.0

        # Authentic observation timestamp from DataFrame index
        obs_time = hist.index[-1].isoformat()

        if is_yield:
            # Yield in percentage points (e.g. 4.21%), daily change in basis points
            chg_bp = chg * 100
            return {
                "symbol": symbol,
                "available": True,
                "status": "AVAILABLE",
                "price": round(p, 2),
                "value": round(p, 2),
                "yield": round(p, 2),
                "change": round(chg, 3),
                "changePct": round(chg_pct, 2),
                "changeBps": round(chg_bp, 1),
                "dailyChangeBp": round(chg_bp, 1),
                "provider": "yfinance",
                "observationTime": obs_time,
                "error": None,
            }

        return {
            "symbol": symbol,
            "available": True,
            "status": "AVAILABLE",
            "price": round(p, 2),
            "level": round(p, 2),
            "value": round(p, 2),
            "change": round(chg, 2),
            "changePct": round(chg_pct, 2),
            "provider": "yfinance",
            "observationTime": obs_time,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Provider retrieval failure for {symbol}: {e}")
        return {
            "symbol": symbol,
            "available": False,
            "status": "UNAVAILABLE",
            "price": None,
            "change": None,
            "changePct": None,
            "provider": "yfinance",
            "observationTime": None,
            "error": str(e),
        }


@router.get("/ribbon", tags=["Macro Telemetry"])
def get_macro_ribbon(response: Response = None):
    """Fetch authentic macro ribbon telemetry.
    
    Guarantees:
    1. Zero fabricated fallback numbers.
    2. Per-field availability and failure isolation.
    3. Separation of generatedAt response time and observationTime.
    4. Authoritative NYSE market session derivation via exchange_calendars.
    5. Correct VIX tier threshold ordering (CRITICAL >= 30, ELEVATED >= 20).
    """
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=60, stale-while-revalidate=120"

    generated_at = datetime.now(timezone.utc).isoformat()
    session_info = get_nyse_session_status()

    # 1. Fetch isolated fields
    spy_data = _fetch_isolated_bar("SPY")
    qqq_data = _fetch_isolated_bar("QQQ")
    vix_raw = _fetch_isolated_bar("^VIX")
    tnx_raw = _fetch_isolated_bar("^TNX", is_yield=True)

    # 2. Compute VIX tier
    vix_data = {
        **vix_raw,
        "tier": compute_vix_tier(vix_raw.get("price")),
    }

    treasury_data = {
        **tnx_raw,
        "value": tnx_raw.get("price"),
        "dailyChangeBp": tnx_raw.get("changeBps"),
    }

    # 3. Derive Macro Regime (Requires genuine SPY history; never fabricated default)
    regime = "UNAVAILABLE"
    regime_summary = "Macro indicators unavailable; regime determination suspended."
    
    if spy_data.get("available", False):
        try:
            spy_1y = yf.Ticker("SPY").history(period="1y", interval="1d")
            if not spy_1y.empty and len(spy_1y) >= 20:
                returns = spy_1y["Close"].pct_change().dropna()
                vol_annual = float(returns.std() * (252 ** 0.5) * 100)
                return_annual = float(returns.mean() * 252 * 100)

                vix_val = vix_data.get("price")
                if vol_annual >= 22.0 or (vix_val is not None and vix_val >= 25.0):
                    regime = "DEFENSIVE"
                    regime_summary = "Elevated tail risk and high volatility require defensive positioning and stop-discipline."
                elif vol_annual < 18.0 and return_annual > 0:
                    regime = "RISK_ON"
                    regime_summary = "Confirmed uptrend with controlled volatility and positive equity momentum."
                else:
                    regime = "NEUTRAL"
                    regime_summary = "Mixed macroeconomic indicators; selective stock-picking recommended."
            else:
                regime_summary = "Insufficient historical bars to compute statistical volatility."
        except Exception as e:
            logger.warning(f"Failed to calculate statistical regime from SPY: {e}")
            regime_summary = f"Regime calculation error: {str(e)}"

    # 4. Determine overall data source indicator
    fields = [spy_data, qqq_data, vix_data, treasury_data]
    available_count = sum(1 for f in fields if f.get("available"))
    
    if available_count == len(fields):
        data_source = "DAILY_CLOSE"
    elif available_count > 0:
        data_source = "PARTIAL_AVAILABLE"
    else:
        data_source = "UNAVAILABLE"

    is_open = session_info.get("status") == "OPEN"

    return {
        "generatedAt": generated_at,
        "observationTime": spy_data.get("observationTime"),
        "marketSession": session_info.get("status", "UNAVAILABLE"),
        "sessionDetail": session_info,
        "dataSource": data_source,
        "settlementPinned": not is_open,
        "isSettlementPinned": not is_open,
        "regime": regime,
        "regimeSummary": regime_summary,
        "spx": spy_data,
        "spy": spy_data,
        "qqq": qqq_data,
        "vix": vix_data,
        "vixTier": vix_data.get("tier"),
        "treasury10Y": treasury_data,
        "tenYearYield": treasury_data,
        # Legacy/UI compatibility aliases populated strictly from authentic fields
        "updatedAt": generated_at,
    }
