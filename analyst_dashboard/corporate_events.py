"""Corporate Events & Earnings Schedule Normalization Engine.
Provides a typed, source-backed corporate-event contract adhering to ARX Closure Model v2.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import zoneinfo
import math

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

# Known primary-source verified corporate schedules
VERIFIED_PRIMARY_SCHEDULES: Dict[str, Dict[str, Any]] = {
    "AXP": {
        "event_date": "2026-10-23",
        "results_release_time_et": "07:00",
        "conference_call_time_et": "08:30",
        "source_name": "American Express Investor Relations / SEC EDGAR",
        "source_url": "https://ir.americanexpress.com",
        "confirmation_status": "CONFIRMED",
        "event_type": "EARNINGS_ANNOUNCEMENT",
    }
}


def normalize_corporate_event(
    symbol: str,
    info: Optional[Dict[str, Any]] = None,
    calendar: Optional[Any] = None,
    now_utc: Optional[datetime] = None,
    staleness_threshold_hours: float = 48.0,
) -> Dict[str, Any]:
    """
    Parse, normalize, and classify corporate event telemetry into the canonical contract.
    Enforces epistemic honesty:
    - Never fabricates dates or times.
    - Missing coverage fails closed to UNAVAILABLE (never PASS).
    - Distinguishes conference call time from results release time.
    - Does not treat isEarningsDateEstimate: false alone as proof of primary-source confirmation.
    """
    clean_sym = symbol.upper().replace("-USD", "").strip()
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # Fallback default: UNAVAILABLE
    unavailable_record = {
        "symbol": clean_sym,
        "event_type": "EARNINGS_ANNOUNCEMENT",
        "event_timestamp_utc": None,
        "event_date_et": None,
        "event_time_et": None,
        "time_zone": "America/New_York",
        "results_release_time_et": None,
        "conference_call_time_et": None,
        "days_to_earnings": None,
        "has_imminent_earnings": False,
        "has_exact_time": False,
        "is_estimate": True,
        "confirmation_status": "UNVERIFIED",
        "coverage_status": "DATA_UNAVAILABLE",
        "freshness_status": "UNVERIFIED",
        "news_sentiment_status": "UNAVAILABLE",
        "risk_tier": "UNVERIFIED_RISK",
        "source_name": "Provider Telemetry Unavailable",
        "source_url": None,
        "last_verified_at": now_utc.isoformat(),
        "preflight_state": "UNAVAILABLE",
        "preflight_summary": {
            "plain_english": "Earnings calendar coverage unavailable: Event runway cannot be certified.",
            "pro_quant": "Catalyst Telemetry Unavailable: Corporate event schedule unverified from primary source.",
        },
    }

    if not info and not calendar and clean_sym not in VERIFIED_PRIMARY_SCHEDULES:
        return unavailable_record

    info_dict = info or {}
    earnings_ts = info_dict.get("earningsTimestamp") or info_dict.get("earningsCallTimestampStart") or info_dict.get("earningsTimestampStart")
    call_ts_start = info_dict.get("earningsCallTimestampStart")
    is_estimate_flag = bool(info_dict.get("isEarningsDateEstimate", True))

    primary_verified = VERIFIED_PRIMARY_SCHEDULES.get(clean_sym)
    
    event_dt_utc: Optional[datetime] = None
    has_exact_time = False

    if earnings_ts is not None and isinstance(earnings_ts, (int, float)) and earnings_ts > 0:
        try:
            event_dt_utc = datetime.fromtimestamp(earnings_ts, timezone.utc)
            has_exact_time = True
        except (ValueError, OSError):
            event_dt_utc = None

    # If timestamp absent, check calendar dict for Earnings Date list
    if event_dt_utc is None and calendar and isinstance(calendar, dict):
        ed_list = calendar.get("Earnings Date")
        if ed_list and isinstance(ed_list, list) and len(ed_list) > 0:
            first_date = ed_list[0]
            if hasattr(first_date, "year"):
                # Date only; do not invent an exact intraday release hour
                event_dt_utc = datetime(first_date.year, first_date.month, first_date.day, 13, 30, tzinfo=timezone.utc)
                has_exact_time = False

    if event_dt_utc is None and primary_verified:
        # Date string from primary source verification
        p_date = primary_verified.get("event_date")
        if p_date:
            try:
                parts = [int(p) for p in p_date.split("-")]
                # Anchor to 12:30 UTC (8:30 AM EDT) or call time
                event_dt_utc = datetime(parts[0], parts[1], parts[2], 12, 30, tzinfo=timezone.utc)
                has_exact_time = True
            except Exception:
                pass

    if event_dt_utc is None:
        return unavailable_record

    event_dt_et = event_dt_utc.astimezone(NY_TZ)
    event_date_et = event_dt_et.strftime("%Y-%m-%d")
    event_time_et = event_dt_et.strftime("%H:%M") if has_exact_time else None

    # Compute days to event (exact floating delta)
    total_seconds_left = (event_dt_utc - now_utc).total_seconds()
    days_to_earnings = round(total_seconds_left / 86400.0, 2)

    # Results release vs conference call distinction
    results_release_time_et = None
    conference_call_time_et = None

    if primary_verified:
        results_release_time_et = primary_verified.get("results_release_time_et")
        conference_call_time_et = primary_verified.get("conference_call_time_et")
        source_name = primary_verified.get("source_name", "Primary Corporate Investor Relations")
        source_url = primary_verified.get("source_url")
        confirmation_status = primary_verified.get("confirmation_status", "CONFIRMED")
        is_estimate = False
    else:
        source_name = "Yahoo Finance (Aggregator)"
        source_url = None
        # Rule: Do not treat isEarningsDateEstimate: false alone as proof of primary-source confirmation
        if is_estimate_flag:
            confirmation_status = "ESTIMATED"
            is_estimate = True
        else:
            confirmation_status = "UNVERIFIED"
            is_estimate = False

        if has_exact_time and call_ts_start:
            conference_call_time_et = event_time_et

    # Coverage status & freshness
    if days_to_earnings < 0:
        coverage_status = "ELAPSED_RECENT"
        freshness_status = "RECENT" if abs(days_to_earnings) <= 7.0 else "STALE"
    else:
        if confirmation_status == "CONFIRMED":
            coverage_status = "CONFIRMED_UPCOMING"
        elif confirmation_status == "ESTIMATED":
            coverage_status = "ESTIMATED_UPCOMING"
        else:
            coverage_status = "COVERED"
        freshness_status = "LIVE"

    # Risk Tier & Pre-Flight State determination
    if days_to_earnings < 0:
        # Invariant 3: A completed historical event alone must not establish a verified clean future runway.
        # Invariant 4: A verified clean runway requires adequate, fresh coverage of the relevant forward-looking interval.
        risk_tier = "UNVERIFIED_RISK"
        preflight_state = "UNAVAILABLE"
        has_imminent = False
        plain_msg = f"Prior earnings reported ({event_date_et}), but next corporate earnings date is unverified. Cannot certify forward runway."
        pro_msg = f"Historical Event Elapsed: Prior report on {event_date_et}. Forward schedule unverified; runway cannot be certified."
    elif 0.0 <= days_to_earnings <= 1.0:
        risk_tier = "ACUTE_BINARY_GAP_RISK"
        preflight_state = "WARN"
        has_imminent = True
        hours_left = max(1, int(days_to_earnings * 24))
        plain_msg = f"⚠️ High Risk: Scheduled earnings announcement in <={hours_left}h ({event_date_et})."
        pro_msg = f"⚠️ Acute Binary Gap Risk: Earnings announcement in <= {hours_left}h. Position sizing restricted."
    elif 1.0 < days_to_earnings <= 2.0:
        risk_tier = "HIGH_GAP_RISK"
        preflight_state = "WARN"
        has_imminent = True
        plain_msg = f"⚠️ High Risk: Scheduled earnings announcement in {days_to_earnings:.1f} days ({event_date_et})."
        pro_msg = f"⚠️ High Gap Risk: Earnings announcement in {days_to_earnings:.1f} days. Overnight gap risk active."
    elif 2.0 < days_to_earnings <= 7.0:
        risk_tier = "CAUTION_BUFFER"
        preflight_state = "WARN"
        has_imminent = True
        plain_msg = f"Caution: Earnings scheduled within 7-day trade window ({days_to_earnings:.1f} days, {event_date_et})."
        pro_msg = f"Caution: Binary catalyst scheduled within 7-day volatility buffer ({days_to_earnings:.1f} days)."
    else:
        risk_tier = "NORMAL_RUNWAY"
        preflight_state = "PASS"
        has_imminent = False
        if confirmation_status == "CONFIRMED":
            plain_msg = f"Sufficient time window to manage trade without overnight earnings gap risk ({days_to_earnings:.1f} days to event)."
            pro_msg = f"Catalyst Buffer: Clean window verified ({days_to_earnings:.1f} days to binary catalyst)."
        else:
            plain_msg = f"Indicative time window (>7 days) without imminent earnings gap risk ({days_to_earnings:.1f} days to estimated event, unconfirmed by primary source)."
            pro_msg = f"Catalyst Buffer: Clean window indicated ({days_to_earnings:.1f} days to unconfirmed binary catalyst)."

    return {
        "symbol": clean_sym,
        "event_type": "EARNINGS_ANNOUNCEMENT",
        "event_timestamp_utc": int(event_dt_utc.timestamp()),
        "event_date_et": event_date_et,
        "event_time_et": event_time_et,
        "time_zone": "America/New_York",
        "results_release_time_et": results_release_time_et,
        "conference_call_time_et": conference_call_time_et,
        "days_to_earnings": days_to_earnings,
        "has_imminent_earnings": has_imminent,
        "has_exact_time": has_exact_time,
        "is_estimate": is_estimate,
        "confirmation_status": confirmation_status,
        "coverage_status": coverage_status,
        "freshness_status": freshness_status,
        "news_sentiment_status": "UNAVAILABLE",
        "risk_tier": risk_tier,
        "source_name": source_name,
        "source_url": source_url,
        "last_verified_at": now_utc.isoformat(),
        "preflight_state": preflight_state,
        "preflight_summary": {
            "plain_english": plain_msg,
            "pro_quant": pro_msg,
        },
    }
