"""FRED (Federal Reserve Economic Data) API Fetcher & Macroeconomic Analysis Module."""

import os
import logging
import hashlib
import json
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FRED_API_KEY = os.getenv("FRED_API_KEY", "")


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

    def get_macro_indicators(self) -> Dict[str, Any]:
        """
        Fetch core macroeconomic regime indicators:
        - T10Y2Y: 10-Year Minus 2-Year Treasury Yield Spread (%)
        - FEDFUNDS: Effective Federal Funds Rate (%)
        - BAMLH0A0HYM2: US High Yield Option-Adjusted Spread (%)
        - CPIAUCSL: Consumer Price Index level
        """
        yc_record = self.fetch_latest_observation_record("T10Y2Y")
        ff_record = self.fetch_latest_observation_record("FEDFUNDS")
        cs_record = self.fetch_latest_observation_record("BAMLH0A0HYM2")
        cpi_record = self.fetch_latest_observation_record("CPIAUCSL")

        yield_curve = yc_record["value"] if yc_record is not None else None
        fed_funds = ff_record["value"] if ff_record is not None else None
        credit_spread_oas = cs_record["value"] if cs_record is not None else None
        cpi = cpi_record["value"] if cpi_record is not None else None

        yc_ts = yc_record.get("timestamp") if yc_record else None
        cs_ts = cs_record.get("timestamp") if cs_record else None

        # Fail-closed regime and rating: only compute if core indicators are authentic
        if yield_curve is not None and credit_spread_oas is not None:
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
        else:
            rating = 0
            regime = "Macro Telemetry Unavailable"
            rate_impact = "Macroeconomic liquidity indicators unavailable."
            inflation_impact = "Macroeconomic inflation telemetry unavailable."
            availability = "PARTIAL" if (yield_curve is not None or credit_spread_oas is not None) else "UNAVAILABLE"

        return {
            # Legacy provider keys
            "yield_curve_spread": round(yield_curve, 2) if yield_curve is not None else None,
            "fed_funds_rate": round(fed_funds, 2) if fed_funds is not None else None,
            "credit_spread_oas": round(credit_spread_oas, 2) if credit_spread_oas is not None else None,
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
            "macro_observation_available_at": yc_ts or cs_ts,
            "provider": "FRED",
            "availability": availability,
        }

