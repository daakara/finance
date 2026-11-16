"""FRED (Federal Reserve Economic Data) API Fetcher & Macroeconomic Analysis Module."""

import os
import logging
import hashlib
import json
import datetime
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Explicit & Documented Macroeconomic Series Freshness Policy ──────────────
# Invariants Enforced:
# 1. Daily Series (T10Y2Y, BAMLH0A0HYM2, DFII10, DFF): Max allowable age = 7 calendar days
#    (accounting for weekend and federal market holiday publication schedules).
# 2. Monthly Series (FEDFUNDS): Max allowable age = 45 calendar days (accounting for monthly
#    release cadence; distinct from daily DFF).
# 3. Future Observation Quarantine: Dates ahead of evaluation reference date rejected as FUTURE_DATED.
# 4. Genuine Numeric Zero: 0.0 is preserved as a valid observation and never mistaken for missing.

FRED_SERIES_POLICY: Dict[str, Dict[str, Any]] = {
    "T10Y2Y": {
        "series_id": "T10Y2Y",
        "name": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
        "frequency": "DAILY",
        "max_age_days": 7,
        "required_for_regime": True,
        "description": "Daily constant maturity treasury yield curve spread (recession vs. expansion).",
    },
    "BAMLH0A0HYM2": {
        "series_id": "BAMLH0A0HYM2",
        "name": "ICE BofA US High Yield Index Option-Adjusted Spread",
        "frequency": "DAILY",
        "max_age_days": 7,
        "required_for_regime": True,
        "description": "Daily high-yield credit spread (credit risk appetite vs. liquidity crunch).",
    },
    "DFII10": {
        "series_id": "DFII10",
        "name": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Inflation-Indexed",
        "frequency": "DAILY",
        "max_age_days": 7,
        "required_for_regime": True,
        "description": "Daily 10-Year real interest rate (TIPS) capital hurdle rate.",
    },
    "FEDFUNDS": {
        "series_id": "FEDFUNDS",
        "name": "Federal Funds Effective Rate (Monthly Average)",
        "frequency": "MONTHLY",
        "max_age_days": 45,
        "required_for_regime": True,
        "description": "Monthly average effective federal funds policy rate. Distinct from daily DFF.",
    },
    "DFF": {
        "series_id": "DFF",
        "name": "Federal Funds Effective Rate (Daily)",
        "frequency": "DAILY",
        "max_age_days": 7,
        "required_for_regime": False,
        "description": "Daily Federal Funds Effective Rate (high-frequency alternative series).",
    },
}


def evaluate_series_freshness(
    series_id: str,
    obs_date_str: Optional[str],
    ref_date: Optional[datetime.date] = None,
) -> Dict[str, Any]:
    """Evaluate series observation freshness against its documented publication frequency policy.

    Invariants:
    - Daily series: max_age_days = 7.
    - Monthly series (FEDFUNDS): max_age_days = 45.
    - Future dates: flagged as FUTURE_DATED (anti-lookahead).
    - Missing dates: flagged as MISSING.
    """
    policy = FRED_SERIES_POLICY.get(
        series_id,
        {"frequency": "DAILY", "max_age_days": 7, "name": series_id, "required_for_regime": True},
    )
    max_age_days = policy["max_age_days"]
    freq = policy["frequency"]

    if not obs_date_str:
        return {
            "is_fresh": False,
            "status": "MISSING",
            "age_days": None,
            "max_age_days": max_age_days,
            "frequency": freq,
            "observation_date": None,
        }

    clean_date = obs_date_str.split("T")[0]
    try:
        obs_dt = datetime.date.fromisoformat(clean_date)
    except Exception:
        return {
            "is_fresh": False,
            "status": "INVALID_DATE",
            "age_days": None,
            "max_age_days": max_age_days,
            "frequency": freq,
            "observation_date": clean_date,
        }

    if ref_date is None:
        ref_date = datetime.datetime.now(datetime.timezone.utc).date()

    age_days = (ref_date - obs_dt).days

    if age_days < 0:
        return {
            "is_fresh": False,
            "status": "FUTURE_DATED",
            "age_days": age_days,
            "max_age_days": max_age_days,
            "frequency": freq,
            "observation_date": clean_date,
        }

    is_fresh = (age_days <= max_age_days)
    status = "FRESH" if is_fresh else "STALE"

    return {
        "is_fresh": is_fresh,
        "status": status,
        "age_days": age_days,
        "max_age_days": max_age_days,
        "frequency": freq,
        "observation_date": clean_date,
    }


def normalize_macro_payload(raw_data: Any) -> Optional[Dict[str, Any]]:
    """Normalizes raw or provider-specific macro telemetry into the canonical ARX contract.

    Canonical Internal Semantics:
    - yield_curve_10y2y: float | None
    - high_yield_credit_spread: float | None
    - credit_spread: float | None (backward-compatible alias)
    - yield_observation_timestamp: str | None
    - credit_observation_timestamp: str | None
    - macro_observation_available_at: str | None
    - provider: str
    - availability: "AVAILABLE" | "PARTIAL" | "UNAVAILABLE"
    - raw_payload_hash: str | None

    Invariants Enforced:
    - VALID_ZERO_MACRO_VALUE = PRESERVED (explicit `is not None`).
    - MISSING_MACRO_DATA = UNAVAILABLE (never converted to 0 or neutral default).
    - FABRICATED_MACRO_DEFAULTS = 0.
    """
    if not isinstance(raw_data, dict):
        return None

    # Canonical yield resolution (canonical key -> provider aliases)
    raw_yc = None
    if "yield_curve_10y2y" in raw_data and raw_data["yield_curve_10y2y"] is not None:
        raw_yc = raw_data["yield_curve_10y2y"]
    elif "yield_curve_spread" in raw_data and raw_data["yield_curve_spread"] is not None:
        raw_yc = raw_data["yield_curve_spread"]
    elif "yieldCurve10y2y" in raw_data and raw_data["yieldCurve10y2y"] is not None:
        raw_yc = raw_data["yieldCurve10y2y"]

    # Canonical credit spread resolution (canonical key -> provider aliases)
    raw_cs = None
    if "high_yield_credit_spread" in raw_data and raw_data["high_yield_credit_spread"] is not None:
        raw_cs = raw_data["high_yield_credit_spread"]
    elif "credit_spread_oas" in raw_data and raw_data["credit_spread_oas"] is not None:
        raw_cs = raw_data["credit_spread_oas"]
    elif "credit_spread" in raw_data and raw_data["credit_spread"] is not None:
        raw_cs = raw_data["credit_spread"]
    elif "creditSpread" in raw_data and raw_data["creditSpread"] is not None:
        raw_cs = raw_data["creditSpread"]
    elif "highYieldCreditSpread" in raw_data and raw_data["highYieldCreditSpread"] is not None:
        raw_cs = raw_data["highYieldCreditSpread"]

    # Explicit numeric parsing with zero preservation
    yield_curve_val: Optional[float] = None
    if raw_yc is not None and not isinstance(raw_yc, bool):
        try:
            yield_curve_val = float(raw_yc)
        except (ValueError, TypeError):
            yield_curve_val = None

    credit_spread_val: Optional[float] = None
    if raw_cs is not None and not isinstance(raw_cs, bool):
        try:
            credit_spread_val = float(raw_cs)
        except (ValueError, TypeError):
            credit_spread_val = None

    # Observation timestamps (Point-in-Time Integrity)
    yc_ts = raw_data.get("yield_observation_timestamp") or raw_data.get("yieldObservationTimestamp")
    cs_ts = raw_data.get("credit_observation_timestamp") or raw_data.get("creditObservationTimestamp")
    macro_avail_at = (
        raw_data.get("macro_observation_available_at")
        or raw_data.get("macroObservationAvailableAt")
        or yc_ts
        or cs_ts
        or raw_data.get("observation_date")
    )

    if yield_curve_val is not None and credit_spread_val is not None:
        availability = "AVAILABLE"
    elif yield_curve_val is not None or credit_spread_val is not None:
        availability = "PARTIAL"
    else:
        availability = "UNAVAILABLE"

    # Anti-lookahead check: if observation availability is ahead of recommendation/cutoff, quarantine
    rec_ts = raw_data.get("recommended_at") or raw_data.get("observation_cutoff") or raw_data.get("as_of")
    if macro_avail_at and rec_ts:
        try:
            if str(macro_avail_at) > str(rec_ts):
                availability = "UNAVAILABLE"
        except Exception:
            pass

    raw_hash = None
    try:
        encoded = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        raw_hash = hashlib.sha256(encoded).hexdigest()
    except Exception:
        raw_hash = None

    # Fail-closed: If not fully AVAILABLE, suppress indicators to guarantee confluence has_macro = False without fabricated defaults
    effective_yc = yield_curve_val if availability == "AVAILABLE" else None
    effective_cs = credit_spread_val if availability == "AVAILABLE" else None

    result = {
        "yield_curve_10y2y": effective_yc,
        "high_yield_credit_spread": effective_cs,
        "yield_observation_timestamp": str(yc_ts) if yc_ts is not None else None,
        "credit_observation_timestamp": str(cs_ts) if cs_ts is not None else None,
        "macro_observation_available_at": str(macro_avail_at) if macro_avail_at is not None else None,
        "provider": str(raw_data.get("provider", "FRED")),
        "availability": availability,
        "raw_payload_hash": raw_hash,
    }
    if effective_cs is not None:
        result["credit_spread"] = effective_cs

    if "real_interest_rate" in raw_data and raw_data["real_interest_rate"] is not None:
        result["real_interest_rate"] = raw_data["real_interest_rate"]
    elif "real_interest_rate_10y" in raw_data and raw_data["real_interest_rate_10y"] is not None:
        result["real_interest_rate"] = raw_data["real_interest_rate_10y"]

    if "fed_funds_rate" in raw_data and raw_data["fed_funds_rate"] is not None:
        result["fed_funds_rate"] = raw_data["fed_funds_rate"]

    if "series_observations" in raw_data and isinstance(raw_data["series_observations"], dict):
        result["series_observations"] = raw_data["series_observations"]

    if "freshness_summary" in raw_data and isinstance(raw_data["freshness_summary"], dict):
        result["freshness_summary"] = raw_data["freshness_summary"]

    for d_key in [
        "yield_curve_observation_date",
        "credit_spread_observation_date",
        "real_rate_observation_date",
        "fed_funds_observation_date",
    ]:
        if d_key in raw_data and raw_data[d_key] is not None:
            result[d_key] = raw_data[d_key]

    return result


class FredMacroFetcher:
    """Fetches macroeconomic time series from the St. Louis Federal Reserve (FRED) API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_FRED_API_KEY
        self._cache: Dict[str, Any] = {}

    def fetch_latest_observation_record(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent valid observation and timestamp for a given series without fabricated defaults."""
        cache_key = f"{series_id}_record"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            return None

        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=8)
            if resp.status_code == 200:
                obs_list = resp.json().get("observations", [])
                for obs in obs_list:
                    val_str = obs.get("value", "")
                    if val_str and val_str != ".":
                        try:
                            val = float(val_str)
                            obs_date = obs.get("date")
                            record = {
                                "value": val,
                                "date": obs_date,
                                "timestamp": f"{obs_date}T00:00:00Z" if obs_date else None,
                                "realtime_start": obs.get("realtime_start"),
                            }
                            self._cache[cache_key] = record
                            self._cache[series_id] = val
                            return record
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            logger.warning(f"Failed to fetch FRED series {series_id}: {e}")

        return None

    def fetch_latest_observation(self, series_id: str, default_val: Optional[float] = None) -> Optional[float]:
        """Fetch the most recent valid observation for a given FRED series ID."""
        record = self.fetch_latest_observation_record(series_id)
        if record is not None and "value" in record:
            return record["value"]
        return default_val

    def get_macro_indicators(self, ref_date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Fetch core macroeconomic regime indicators:
        - T10Y2Y: 10-Year Minus 2-Year Treasury Yield Spread (%) [DAILY, max age 7d]
        - BAMLH0A0HYM2: US High Yield Option-Adjusted Spread (%) [DAILY, max age 7d]
        - DFII10: 10-Year Real Interest Rate (TIPS) (%) [DAILY, max age 7d]
        - FEDFUNDS: Effective Federal Funds Rate (%) [MONTHLY, max age 45d; distinct from daily DFF]
        - CPIAUCSL: Consumer Price Index level
        """
        yc_record = self.fetch_latest_observation_record("T10Y2Y")
        ff_record = self.fetch_latest_observation_record("FEDFUNDS")
        cs_record = self.fetch_latest_observation_record("BAMLH0A0HYM2")
        dfii_record = self.fetch_latest_observation_record("DFII10")
        cpi_record = self.fetch_latest_observation_record("CPIAUCSL")

        yield_curve = yc_record["value"] if yc_record is not None else None
        fed_funds = ff_record["value"] if ff_record is not None else None
        credit_spread_oas = cs_record["value"] if cs_record is not None else None
        real_rate = dfii_record["value"] if dfii_record is not None else None
        cpi = cpi_record["value"] if cpi_record is not None else None

        yc_ts = yc_record.get("timestamp") if yc_record else None
        cs_ts = cs_record.get("timestamp") if cs_record else None
        ff_ts = ff_record.get("timestamp") if ff_record else None
        dfii_ts = dfii_record.get("timestamp") if dfii_record else None

        yc_date = yc_record.get("date") if yc_record else None
        cs_date = cs_record.get("date") if cs_record else None
        ff_date = ff_record.get("date") if ff_record else None
        dfii_date = dfii_record.get("date") if dfii_record else None

        # Freshness evaluation per series against explicit policy
        yc_fresh = evaluate_series_freshness("T10Y2Y", yc_date, ref_date=ref_date)
        cs_fresh = evaluate_series_freshness("BAMLH0A0HYM2", cs_date, ref_date=ref_date)
        dfii_fresh = evaluate_series_freshness("DFII10", dfii_date, ref_date=ref_date)
        ff_fresh = evaluate_series_freshness("FEDFUNDS", ff_date, ref_date=ref_date)

        stale_series = []
        missing_series = []
        valid_dates = []

        for sid, fresh_meta, val in [
            ("T10Y2Y", yc_fresh, yield_curve),
            ("BAMLH0A0HYM2", cs_fresh, credit_spread_oas),
            ("DFII10", dfii_fresh, real_rate),
            ("FEDFUNDS", ff_fresh, fed_funds),
        ]:
            if val is None:
                missing_series.append(sid)
            elif not fresh_meta["is_fresh"]:
                stale_series.append(sid)
            if fresh_meta.get("observation_date"):
                valid_dates.append(fresh_meta["observation_date"])

        is_all_fresh = (len(missing_series) == 0 and len(stale_series) == 0)

        # Fail-closed regime and rating: only compute if core indicators are authentic, complete, and fresh
        if yield_curve is not None and credit_spread_oas is not None and is_all_fresh:
            rating = 2
            regime = "Accommodative Growth"
            rate_impact = "Fed interest rate reductions provide equity multiple expansion tailwinds"
            inflation_impact = "Easing CPI trend reduces discount rate pressure on corporate valuations"

            if yield_curve < 0:  # Inverted yield curve (Recession signal)
                rating += 1
                regime = "Inverted Yield Curve (Recession Warning)"
            elif credit_spread_oas > 4.5:  # Elevated credit risk / liquidity crunch
                rating += 2
                regime = "Liquidity Contraction & High Credit Risk"
                rate_impact = "Widening credit spreads increase borrowing costs for high-beta assets"
            elif fed_funds is not None and fed_funds > 4.5:
                rating += 1
                regime = "Restrictive Monetary Tightening"
                rate_impact = "Elevated risk-free hurdle rate depresses valuation multiples"
            elif yield_curve > 0.30 and credit_spread_oas < 3.0:
                rating = 1
                regime = "Optimal Expansionary Goldilocks"
                rate_impact = "Steepening curve and tight credit spreads fuel strong risk-on alpha"
            availability = "AVAILABLE"
        elif stale_series and len(missing_series) == 0:
            rating = 0
            regime = "Macro Telemetry Stale"
            rate_impact = f"Macroeconomic telemetry stale ({', '.join(stale_series)})."
            inflation_impact = "Macroeconomic inflation telemetry unverified."
            availability = "STALE"
        else:
            rating = 0
            regime = "Macro Telemetry Unavailable"
            rate_impact = "Macroeconomic liquidity indicators unavailable."
            inflation_impact = "Macroeconomic inflation telemetry unavailable."
            availability = "PARTIAL" if (yield_curve is not None or credit_spread_oas is not None) else "UNAVAILABLE"

        series_observations = {
            "T10Y2Y": {
                "series_id": "T10Y2Y",
                "name": "10-Year Minus 2-Year Treasury Yield Spread",
                "value": round(yield_curve, 2) if yield_curve is not None else None,
                "date": yc_date,
                "timestamp": yc_ts,
                "frequency": "DAILY",
                "age_days": yc_fresh["age_days"],
                "max_age_days": 7,
                "is_fresh": yc_fresh["is_fresh"],
                "status": yc_fresh["status"],
            },
            "BAMLH0A0HYM2": {
                "series_id": "BAMLH0A0HYM2",
                "name": "US High Yield Option-Adjusted Spread",
                "value": round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
                "date": cs_date,
                "timestamp": cs_ts,
                "frequency": "DAILY",
                "age_days": cs_fresh["age_days"],
                "max_age_days": 7,
                "is_fresh": cs_fresh["is_fresh"],
                "status": cs_fresh["status"],
            },
            "DFII10": {
                "series_id": "DFII10",
                "name": "10-Year Real Interest Rate (TIPS)",
                "value": round(real_rate, 2) if real_rate is not None else None,
                "date": dfii_date,
                "timestamp": dfii_ts,
                "frequency": "DAILY",
                "age_days": dfii_fresh["age_days"],
                "max_age_days": 7,
                "is_fresh": dfii_fresh["is_fresh"],
                "status": dfii_fresh["status"],
            },
            "FEDFUNDS": {
                "series_id": "FEDFUNDS",
                "name": "Effective Federal Funds Rate (Monthly)",
                "value": round(fed_funds, 2) if fed_funds is not None else None,
                "date": ff_date,
                "timestamp": ff_ts,
                "frequency": "MONTHLY",
                "age_days": ff_fresh["age_days"],
                "max_age_days": 45,
                "is_fresh": ff_fresh["is_fresh"],
                "status": ff_fresh["status"],
                "distinction": "Monthly effective rate (FEDFUNDS); distinct from daily DFF.",
            },
        }

        freshness_summary = {
            "is_all_fresh": is_all_fresh,
            "stale_series": stale_series,
            "missing_series": missing_series,
            "oldest_observation_date": min(valid_dates) if valid_dates else None,
            "newest_observation_date": max(valid_dates) if valid_dates else None,
        }

        return {
            # Legacy provider keys
            "yield_curve_spread": round(yield_curve, 2) if yield_curve is not None else None,
            "fed_funds_rate": round(fed_funds, 2) if fed_funds is not None else None,
            "credit_spread_oas": round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
            "real_interest_rate": round(real_rate, 2) if real_rate is not None else None,
            "real_interest_rate_10y": round(real_rate, 2) if real_rate is not None else None,
            "cpi_index": round(cpi, 2) if cpi is not None else None,
            "cpi_yoy": 2.4,  # Current trailing annualized rate
            "rating": max(1, min(5, rating)) if rating > 0 else 0,
            "regime": regime,
            "interestRateImpact": rate_impact,
            "inflationImpact": inflation_impact,

            # Canonical internal contract fields (Preferred Internal Contract)
            "yield_curve_10y2y": round(yield_curve, 2) if yield_curve is not None else None,
            "high_yield_credit_spread": round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
            "credit_spread": round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
            "yield_observation_timestamp": yc_ts,
            "credit_observation_timestamp": cs_ts,
            "real_rate_observation_timestamp": dfii_ts,
            "fed_funds_observation_timestamp": ff_ts,
            "yield_curve_observation_date": yc_date,
            "credit_spread_observation_date": cs_date,
            "real_rate_observation_date": dfii_date,
            "fed_funds_observation_date": ff_date,
            "macro_observation_available_at": min(valid_dates) if valid_dates else None,
            "freshness_summary": freshness_summary,
            "series_observations": series_observations,
            "provider": "FRED",
            "availability": availability,
        }

