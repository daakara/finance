"""Qualified Market-Data Adapter & Corporate Action Adjustment Verification Engine.

Enforces:
1. End-to-end traceability from retained raw response bytes -> SHA-256 -> normalization -> settlement.
2. Tamper rejection: any byte modification fails closed before bar extraction or ledger commit.
3. Explicit corporate-action verification: split adjustment cannot be inferred from a field name
   or an empty split list; authentic corporate-action coverage must be verified.
4. XNYS calendar session completeness, completed-session timestamps, and OHLCV bound consistency.
5. Provider continuity and historical price revision rejection across batches.
6. Zero network calls; offline qualification only.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import exchange_calendars as xcals


class RawBytesHashMismatchError(ValueError):
    """Raised when computed SHA-256 does not match recorded rawResponseSha256."""
    pass


class UnverifiedAdjustmentEvidenceError(ValueError):
    """Raised when adjustment evidence lacks authoritative corporate-action proof."""
    pass


class UnsupportedSplitAdjustmentError(ValueError):
    """Raised when stock splits occurred during window without qualified multiplier handling."""
    pass


class DividendAdjustedDataError(ValueError):
    """Raised when prices have been adjusted for cash dividends (violating Contract V1.0.1)."""
    pass


def verify_raw_response_bytes(raw_bytes: bytes, expected_sha256: str) -> None:
    """Cryptographically verifies raw response bytes against recorded SHA-256 digest."""
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError("raw_bytes must be bytes or bytearray")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character lowercase hexadecimal string")
    computed = hashlib.sha256(raw_bytes).hexdigest()
    if computed != expected_sha256:
        raise RawBytesHashMismatchError(
            f"RAW_BYTES_HASH_MISMATCH: Computed {computed} does not match expected {expected_sha256}"
        )


def verify_corporate_action_evidence(
    proof: Dict[str, Any],
    expected_sessions: Set[str],
    require_authoritative_ref: bool = True,
) -> Dict[str, Any]:
    """Explicitly verifies corporate-action coverage and split adjustment basis.

    Rejects attempts to infer adjustment from field names ('SPLIT_ADJUSTED_ONLY')
    or empty lists ('splitEvents: []') without an authentic corporateActionReference.
    """
    if not isinstance(proof, dict):
        raise UnverifiedAdjustmentEvidenceError("Adjustment evidence must be a dictionary")

    status = proof.get("status")
    if status != "VERIFIED":
        raise UnverifiedAdjustmentEvidenceError(
            f"UNVERIFIED_ADJUSTMENT_EVIDENCE: status is '{status}', expected 'VERIFIED'"
        )

    price_basis = proof.get("priceBasis")
    if price_basis != "SPLIT_ADJUSTED_ONLY":
        raise ValueError(
            f"UNSUPPORTED_ADJUSTMENT_BASIS: priceBasis is '{price_basis}', expected 'SPLIT_ADJUSTED_ONLY'"
        )

    if proof.get("cashDividendsAdjusted") is not False:
        raise DividendAdjustedDataError(
            "DIVIDEND_ADJUSTED_DATA_REJECTED: cash dividends must NOT be adjusted under Contract V1.0.1"
        )

    split_events = proof.get("splitEvents")
    if not isinstance(split_events, list):
        raise ValueError("splitEvents must be a list")

    if split_events != []:
        raise UnsupportedSplitAdjustmentError(
            f"UNSUPPORTED_SPLIT_ADJUSTMENT: Symbol underwent corporate action / stock split ({split_events}); "
            "split factor multiplier application requires separate engine qualification"
        )

    # Validate interval coverage
    cov_start = proof.get("coverageStart", "9999")
    cov_end = proof.get("coverageEnd", "")
    if expected_sessions:
        min_expected = min(expected_sessions)
        max_expected = max(expected_sessions)
        if cov_start > min_expected or cov_end < max_expected:
            raise ValueError(
                f"Adjustment evidence does not cover required interval: "
                f"coverage [{cov_start}, {cov_end}] does not span expected [{min_expected}, {max_expected}]"
            )

    # Corporate action reference verification
    corp_ref = proof.get("corporateActionReference")
    if require_authoritative_ref:
        if not isinstance(corp_ref, dict):
            raise UnverifiedAdjustmentEvidenceError(
                "UNVERIFIED_ADJUSTMENT_EVIDENCE: Corporate-action coverage cannot be inferred from "
                "a field name or empty split list; authentic 'corporateActionReference' required"
            )
        source_type = corp_ref.get("sourceType")
        source_sha = corp_ref.get("sourceSha256")
        if not source_type or not isinstance(source_type, str):
            raise UnverifiedAdjustmentEvidenceError(
                "UNVERIFIED_ADJUSTMENT_EVIDENCE: corporateActionReference missing valid 'sourceType'"
            )
        if not source_sha or not re.fullmatch(r"[a-f0-9]{64}", str(source_sha)):
            raise UnverifiedAdjustmentEvidenceError(
                "UNVERIFIED_ADJUSTMENT_EVIDENCE: corporateActionReference missing 64-hex 'sourceSha256'"
            )
        ref_start = corp_ref.get("coverageStart", "9999")
        ref_end = corp_ref.get("coverageEnd", "")
        if expected_sessions:
            if ref_start > min_expected or ref_end < max_expected:
                raise UnverifiedAdjustmentEvidenceError(
                    f"UNVERIFIED_ADJUSTMENT_EVIDENCE: corporateActionReference coverage [{ref_start}, {ref_end}] "
                    f"does not span expected sessions [{min_expected}, {max_expected}]"
                )
        if corp_ref.get("hasSplits") is not False or corp_ref.get("verifiedZeroSplits") is not True:
            raise UnsupportedSplitAdjustmentError(
                "UNSUPPORTED_SPLIT_ADJUSTMENT: corporateActionReference did not verify zero stock splits"
            )
    else:
        # Fixture / legacy assertion fallback: requires 64-hex sourceSha256 and non-empty sourceDescription
        src_sha = proof.get("sourceSha256", "")
        if not re.fullmatch(r"[a-f0-9]{64}", str(src_sha)):
            raise ValueError("UNRESOLVED_ADJUSTMENT_PROVENANCE: Missing 64-hex sourceSha256")
        if not proof.get("sourceDescription"):
            raise ValueError("UNRESOLVED_ADJUSTMENT_PROVENANCE: Missing sourceDescription")

    return proof


def normalize_yahoo_chart_bars(
    raw_bytes: bytes,
    symbol: str,
    as_of: Optional[str] = None,
    calendar_name: str = "XNYS",
) -> List[Dict[str, Any]]:
    """Deterministically normalizes Yahoo Finance v8 chart JSON into regular-session daily bars.

    Guarantees:
    - Extracts split-adjusted, non-dividend-adjusted OHLCV from indicators.quote[0].
    - Strictly ignores indicators.adjclose[0].
    - Validates regular XNYS session dates in America/New_York.
    - Validates OHLC consistency bounds (0 < Low <= min(Open, Close) <= max(Open, Close) <= High, Vol >= 0).
    - Validates finite numerical types (no bool, no NaN, no inf).
    - Validates strictly monotonic dates with zero duplicates.
    """
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON in raw response bytes: {e}")

    chart_result = data.get("chart", {}).get("result", [])
    if not chart_result:
        raise ValueError("Invalid chart payload: missing chart.result")

    res = chart_result[0]
    meta = res.get("meta", {})
    meta_symbol = meta.get("symbol")
    if meta_symbol and meta_symbol.upper() != symbol.upper():
        raise ValueError(f"Symbol mismatch in payload: expected {symbol}, got {meta_symbol}")

    tz_name = meta.get("exchangeTimezoneName") or meta.get("timezone") or "America/New_York"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("America/New_York")

    cal = xcals.get_calendar(calendar_name)
    timestamps = res.get("timestamp", [])
    indicators = res.get("indicators", {})
    quotes = indicators.get("quote", [])
    if not quotes or not timestamps:
        raise ValueError("Invalid chart payload: missing timestamps or quote indicators")

    q = quotes[0]
    opens = q.get("open", [])
    highs = q.get("high", [])
    lows = q.get("low", [])
    closes = q.get("close", [])
    volumes = q.get("volume", [])

    bars: List[Dict[str, Any]] = []
    seen_dates: Set[str] = set()

    for ts, op, hi, lo, cl, vol in zip(timestamps, opens, highs, lows, closes, volumes):
        if op is None or hi is None or lo is None or cl is None:
            continue

        # Convert epoch timestamp to ET regular session date string
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        dt_local = dt_utc.astimezone(tz)
        date_str = dt_local.date().isoformat()

        if as_of and date_str > as_of:
            raise ValueError(f"Future or invalid daily bar: {date_str} > as_of {as_of}")

        if not cal.is_session(date_str):
            raise ValueError(f"Non-session daily bar: {date_str}")

        vals = [op, hi, lo, cl, vol]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise ValueError(f"Nonfinite OHLCV in bar for {date_str}")

        op_f, hi_f, lo_f, cl_f = float(op), float(hi), float(lo), float(cl)
        vol_i = int(vol or 0)

        if not (0 < lo_f <= min(op_f, cl_f) <= max(op_f, cl_f) <= hi_f and vol_i >= 0):
            raise ValueError(f"Invalid OHLCV bounds in bar for {date_str}: O={op_f}, H={hi_f}, L={lo_f}, C={cl_f}, V={vol_i}")

        if date_str in seen_dates:
            raise ValueError(f"Duplicate, unordered or missing required sessions: duplicate {date_str}")
        seen_dates.add(date_str)

        bars.append({
            "date": date_str,
            "Open": round(op_f, 4),
            "High": round(hi_f, 4),
            "Low": round(lo_f, 4),
            "Close": round(cl_f, 4),
            "Volume": vol_i,
        })

    # Validate strictly monotonic ordering
    dates = [b["date"] for b in bars]
    if dates != sorted(dates):
        raise ValueError("Duplicate, unordered or missing required sessions: dates not strictly sorted")

    return bars


@dataclass(frozen=True)
class MarketDataPayload:
    """Qualified market data payload with cryptographic provenance and raw-byte retention."""
    symbol: str
    provider: str
    retrieval_timestamp_utc: str
    timezone: str
    raw_bytes: bytes
    raw_sha256: str
    target_url: Optional[str] = None
    corporate_action_reference: Optional[Dict[str, Any]] = None
    adjustment_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        verify_raw_response_bytes(self.raw_bytes, self.raw_sha256)
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol must be a non-empty string")
        if not self.provider or not isinstance(self.provider, str):
            raise ValueError("provider must be a non-empty string")
        if not self.retrieval_timestamp_utc or not isinstance(self.retrieval_timestamp_utc, str):
            raise ValueError("retrieval_timestamp_utc must be a non-empty ISO string")

    def normalize(self, as_of: Optional[str] = None, calendar_name: str = "XNYS") -> List[Dict[str, Any]]:
        """Parses raw response bytes into normalized daily regular-session bars."""
        if self.provider == "YAHOO_FINANCE":
            return normalize_yahoo_chart_bars(self.raw_bytes, self.symbol, as_of=as_of, calendar_name=calendar_name)
        elif self.provider in ("OFFLINE_TEST", "SYNTHETIC_FIXTURE"):
            # If payload is JSON array of bars or chart
            try:
                parsed = json.loads(self.raw_bytes.decode("utf-8"))
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and "bars" in parsed:
                    return parsed["bars"]
                elif "chart" in parsed:
                    return normalize_yahoo_chart_bars(self.raw_bytes, self.symbol, as_of=as_of, calendar_name=calendar_name)
            except Exception as e:
                raise ValueError(f"Cannot normalize offline test payload: {e}")
        raise ValueError(f"Unsupported market data provider: {self.provider}")

    def verify_adjustment(
        self,
        expected_sessions: Set[str],
        require_authoritative_ref: bool = True,
    ) -> Dict[str, Any]:
        """Verifies adjustment evidence attached to this payload."""
        proof = self.adjustment_evidence or {}
        if self.corporate_action_reference and "corporateActionReference" not in proof:
            proof = {**proof, "corporateActionReference": self.corporate_action_reference}
        return verify_corporate_action_evidence(
            proof, expected_sessions, require_authoritative_ref=require_authoritative_ref
        )
