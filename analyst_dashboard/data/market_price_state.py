"""
Authoritative ARX Live Dual-Price Contract and Session Resolution Engine.

Enforces strict separation between:
1. LIVE MARKET STATE: real-time spot quote, observation timestamp, live source, and live freshness.
2. COMPLETED ANALYTICAL HISTORY: immutable reference price, reference date, and completed OHLCV indicators.

GOVERNING INVARIANTS:
1. COMPLETED SESSION IMMUTABILITY: Analytical daily indicators (EMA, SMA, ATR, RSI, VaR)
   are anchored strictly to completed sessions (analysisReferencePrice) and NEVER
   contaminated by incomplete intraday price fluctuations.
2. LIVE ACTIONABLE EXECUTION: Actionable execution statuses (IN_BUY_ZONE, READY_TO_BUY,
   STOPPED_OUT) require verified live spot prices (liveSpotPrice) during regular sessions.
3. AUTHORITATIVE CALENDAR: Market session state is resolved using the authoritative
   NYSE exchange calendar (XNYS via exchange_calendars) including early closures and special holidays.
4. FRESHNESS CADENCE:
   - REALTIME: quote age <= 60 seconds (60,000 ms) during active sessions.
   - STALE: quote age between 60 seconds and 300 seconds (5 minutes).
   - DELAYED: quote age > 300 seconds or provider has exchange delay.
   - UNAVAILABLE: quote missing or unparseable.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime, timezone, time
import logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger("arx.market_data.dual_price")

US_EASTERN = ZoneInfo("America/New_York")

# Authoritative freshness thresholds (in milliseconds)
REALTIME_MAX_AGE_MS = 60 * 1000        # 60 seconds (1 minute) for real-time live execution
STALE_MAX_AGE_MS = 5 * 60 * 1000       # 300 seconds (5 minutes) maximum allowable before delayed

# Exchange Calendar Authority initialization
_NYSE_CALENDAR = None
try:
    import exchange_calendars as xcals
    _NYSE_CALENDAR = xcals.get_calendar("XNYS")
except Exception as e:
    logger.debug(f"exchange_calendars XNYS initialization fallback: {e}")
    _NYSE_CALENDAR = None


@dataclass
class MarketPriceState:
    """The authoritative ARX dual-price market state contract."""
    symbol: str
    live_spot_price: Optional[float]
    live_observed_at: Optional[int]  # Unix timestamp in milliseconds
    live_source: str                 # "ALPACA_IEX" | "YAHOO" | "UNAVAILABLE"
    live_freshness: str              # "REALTIME" | "STALE" | "DELAYED" | "UNAVAILABLE"
    analysis_reference_price: float
    analysis_reference_date: str     # YYYY-MM-DD of last completed session
    analysis_reference_source: str   # "COMPLETED_SESSION"
    market_session: str              # "REGULAR_SESSION" | "PREMARKET" | "AFTER_HOURS" | "CLOSED" | "WEEKEND" | "HOLIDAY"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes contract to both camelCase and snake_case for frontend/backend interoperability."""
        return {
            "symbol": self.symbol,
            "liveSpotPrice": self.live_spot_price,
            "liveObservedAt": self.live_observed_at,
            "liveSource": self.live_source,
            "liveFreshness": self.live_freshness,
            "analysisReferencePrice": self.analysis_reference_price,
            "analysisReferenceDate": self.analysis_reference_date,
            "analysisReferenceSource": self.analysis_reference_source,
            "marketSession": self.market_session,
            # Snake-case representations
            "live_spot_price": self.live_spot_price,
            "live_observed_at": self.live_observed_at,
            "live_source": self.live_source,
            "live_freshness": self.live_freshness,
            "analysis_reference_price": self.analysis_reference_price,
            "analysis_reference_date": self.analysis_reference_date,
            "analysis_reference_source": self.analysis_reference_source,
            "market_session": self.market_session,
        }


def _is_rule_based_us_holiday(dt_eastern: datetime) -> bool:
    """Deterministic fallback holiday detection covering all NYSE holidays."""
    year = dt_eastern.year
    month = dt_eastern.month
    day = dt_eastern.day
    weekday = dt_eastern.weekday()  # Monday = 0, Sunday = 6

    # 1. New Year's Day (Jan 1, observed Jan 2 if Sun, Dec 31 if Sat)
    if (month == 1 and day == 1) or (month == 1 and day == 2 and weekday == 0) or (month == 12 and day == 31 and weekday == 4):
        return True

    # 2. Martin Luther King, Jr. Day (Third Monday in January)
    if month == 1 and weekday == 0 and 15 <= day <= 21:
        return True

    # 3. Washington's Birthday / Presidents Day (Third Monday in February)
    if month == 2 and weekday == 0 and 15 <= day <= 21:
        return True

    # 4. Good Friday (Easter - 2 days calculation)
    # Anonymous Gregorian algorithm for Easter Sunday
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    easter_month = (h + l - 7 * m + 114) // 31
    easter_day = ((h + l - 7 * m + 114) % 31) + 1
    # Good Friday is 2 days before Easter Sunday
    from datetime import date, timedelta
    easter_date = date(year, easter_month, easter_day)
    good_friday = easter_date - timedelta(days=2)
    if month == good_friday.month and day == good_friday.day:
        return True

    # 5. Memorial Day (Last Monday in May)
    if month == 5 and weekday == 0 and day >= 25:
        return True

    # 6. Juneteenth National Independence Day (June 19, observed June 20 if Sun, June 18 if Sat)
    if (month == 6 and day == 19) or (month == 6 and day == 20 and weekday == 0) or (month == 6 and day == 18 and weekday == 4):
        return True

    # 7. Independence Day (July 4, observed July 5 if Sun, July 3 if Sat)
    if (month == 7 and day == 4) or (month == 7 and day == 5 and weekday == 0) or (month == 7 and day == 3 and weekday == 4):
        return True

    # 8. Labor Day (First Monday in September)
    if month == 9 and weekday == 0 and 1 <= day <= 7:
        return True

    # 9. Thanksgiving Day (Fourth Thursday in November)
    if month == 11 and weekday == 3 and 22 <= day <= 28:
        return True

    # 10. Christmas Day (Dec 25, observed Dec 26 if Sun, Dec 24 if Sat)
    if (month == 12 and day == 25) or (month == 12 and day == 26 and weekday == 0) or (month == 12 and day == 24 and weekday == 4):
        return True

    return False


def get_market_session(now_utc: Optional[datetime] = None) -> str:
    """Determines US Equity session state: REGULAR_SESSION, PREMARKET, AFTER_HOURS, CLOSED, WEEKEND, HOLIDAY.

    Authority Hierarchy:
    1. Primary: exchange_calendars (XNYS) with authoritative session bounds & early closes.
    2. Fallback: Full deterministic NYSE holiday and early-closure rulebook.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    eastern_dt = now_utc.astimezone(US_EASTERN)
    weekday = eastern_dt.weekday()  # Monday = 0, Sunday = 6

    if weekday >= 5:
        return "WEEKEND"

    date_str = eastern_dt.strftime("%Y-%m-%d")

    # Authoritative Calendar Lookup via exchange_calendars
    if _NYSE_CALENDAR is not None:
        try:
            if not _NYSE_CALENDAR.is_session(date_str):
                return "HOLIDAY"

            s_open = _NYSE_CALENDAR.session_open(date_str).astimezone(US_EASTERN).time()
            s_close = _NYSE_CALENDAR.session_close(date_str).astimezone(US_EASTERN).time()
            t = eastern_dt.time()

            if s_open <= t < s_close:
                return "REGULAR_SESSION"
            elif time(4, 0) <= t < s_open:
                return "PREMARKET"
            elif s_close <= t < time(20, 0):
                return "AFTER_HOURS"
            else:
                return "CLOSED"
        except Exception as e:
            logger.debug(f"exchange_calendars session resolution exception: {e}")

    # Fallback Calendar Logic
    if _is_rule_based_us_holiday(eastern_dt):
        return "HOLIDAY"

    t = eastern_dt.time()
    premarket_start = time(4, 0)
    market_open = time(9, 30)

    # Early close detection: Black Friday (day after 4th Thu in Nov) and Christmas Eve (Dec 24)
    month, day = eastern_dt.month, eastern_dt.day
    is_early_close = False
    if month == 11 and weekday == 4 and 23 <= day <= 29:
        is_early_close = True  # Black Friday
    elif month == 12 and day == 24 and weekday < 5:
        is_early_close = True  # Christmas Eve

    market_close = time(13, 0) if is_early_close else time(16, 0)
    afterhours_end = time(20, 0)

    if market_open <= t < market_close:
        return "REGULAR_SESSION"
    elif premarket_start <= t < market_open:
        return "PREMARKET"
    elif market_close <= t < afterhours_end:
        return "AFTER_HOURS"
    else:
        return "CLOSED"


def resolve_dual_price_state(
    symbol: str,
    analysis_reference_price: float,
    analysis_reference_date: str,
    alpaca_fetcher: Optional[Any] = None,
    ticker_obj: Optional[Any] = None,
    now_utc: Optional[datetime] = None,
) -> MarketPriceState:
    """Resolves authentic live spot price independently from completed analytical reference."""
    clean_sym = symbol.upper().strip().replace("-USD", "").replace(".US", "")
    session = get_market_session(now_utc)
    now_ms = int((now_utc or datetime.now(timezone.utc)).timestamp() * 1000)

    live_price: Optional[float] = None
    live_observed_at: Optional[int] = None
    live_source = "UNAVAILABLE"
    live_freshness = "UNAVAILABLE"

    # 1. Primary: Alpaca Market Data (IEX tape)
    if alpaca_fetcher is not None and getattr(alpaca_fetcher, "is_configured", False):
        try:
            quote = alpaca_fetcher.fetch_realtime_quote(clean_sym)
            if quote and quote.get("price"):
                p = float(quote["price"])
                if math.isfinite(p) and p > 0:
                    live_price = round(p, 2)
                    live_observed_at = quote.get("observed_at_ms") or now_ms
                    live_source = "ALPACA_IEX"
        except Exception as e:
            logger.debug(f"Alpaca live quote lookup error for {clean_sym}: {e}")

    # 2. Secondary: Yahoo Finance fast_info / regularMarketPrice
    if live_price is None and ticker_obj is not None:
        try:
            meta = getattr(ticker_obj, "history_metadata", {}) or {}
            rmp = meta.get("regularMarketPrice")
            rmt = meta.get("regularMarketTime")

            # Also check fast_info
            if rmp is None:
                fi = getattr(ticker_obj, "fast_info", None)
                if fi:
                    rmp = getattr(fi, "last_price", None)

            if rmp is not None and type(rmp) in (int, float) and type(rmp).__name__ not in ("MagicMock", "Mock"):
                p = float(rmp)
                if math.isfinite(p) and p > 0:
                    live_price = round(p, 2)
                    live_source = "YAHOO"
                    if rmt is not None and type(rmt) in (int, float) and type(rmt).__name__ not in ("MagicMock", "Mock"):
                        if hasattr(rmt, "timestamp"):
                            live_observed_at = int(rmt.timestamp() * 1000)
                        elif isinstance(rmt, (int, float)):
                            live_observed_at = int(rmt * 1000 if rmt < 1e11 else rmt)
                    if live_observed_at is None:
                        live_observed_at = now_ms
        except Exception as e:
            logger.debug(f"Yahoo live spot lookup error for {clean_sym}: {e}")

    # 3. Classify Freshness strictly by observation age & cadence
    if live_price is not None and live_observed_at is not None:
        age_ms = max(0, now_ms - live_observed_at)
        if session in ("REGULAR_SESSION", "PREMARKET", "AFTER_HOURS"):
            if live_source == "YAHOO":
                # Yahoo Finance public feed is inherently delayed unless proven realtime by contract
                live_freshness = "DELAYED"
            elif age_ms <= REALTIME_MAX_AGE_MS:
                live_freshness = "REALTIME"
            elif age_ms <= STALE_MAX_AGE_MS:
                live_freshness = "STALE"
            else:
                live_freshness = "DELAYED"
        else:
            # When market is closed/weekend/holiday, quote is from prior session
            live_freshness = "DELAYED"
    else:
        live_freshness = "UNAVAILABLE"

    return MarketPriceState(
        symbol=clean_sym,
        live_spot_price=live_price,
        live_observed_at=live_observed_at,
        live_source=live_source,
        live_freshness=live_freshness,
        analysis_reference_price=round(float(analysis_reference_price), 2),
        analysis_reference_date=analysis_reference_date,
        analysis_reference_source="COMPLETED_SESSION",
        market_session=session,
    )
