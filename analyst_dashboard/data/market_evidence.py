"""Canonical Market Provenance and Request Evidence Types for ARX Terminal.

F_13A Metadata-Only Provenance Foundation.
Separates immutable persisted historical provenance from request-time serving metadata.
Zero execution mutations, zero decision vetoes, zero row filtering.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


# ── Factual Enums / Standard Vocabulary ──────────────────────────────────────

class Provider:
    YFINANCE = "YFINANCE"
    FRED = "FRED"
    EODHD = "EODHD"
    SEC_EDGAR = "SEC_EDGAR"
    UNKNOWN = "UNKNOWN"


class IngestionSource:
    DIRECT_PROVIDER = "DIRECT_PROVIDER"
    FALLBACK_PROVIDER = "FALLBACK_PROVIDER"
    UNKNOWN = "UNKNOWN"


class ServingSource:
    DIRECT_PROVIDER = "DIRECT_PROVIDER"
    LOCAL_CACHE = "LOCAL_CACHE"
    FALLBACK_PROVIDER = "FALLBACK_PROVIDER"
    FALLBACK_CACHE = "FALLBACK_CACHE"


class CacheOrigin:
    NONE = "NONE"
    SQLITE_MARKET_STORE = "SQLITE_MARKET_STORE"
    DISKCACHE_PIPELINE = "DISKCACHE_PIPELINE"
    IN_MEMORY_CACHE = "IN_MEMORY_CACHE"
    UNKNOWN = "UNKNOWN"


class ObservationPrecision:
    TIMESTAMP = "TIMESTAMP"
    DATE = "DATE"
    UNKNOWN = "UNKNOWN"


class ObservationSource:
    PROVIDER_BAR_TIMESTAMP = "PROVIDER_BAR_TIMESTAMP"
    DERIVED_FROM_TRADE_DATE = "DERIVED_FROM_TRADE_DATE"
    EXCHANGE_TIMESTAMP = "EXCHANGE_TIMESTAMP"
    UNKNOWN = "UNKNOWN"


class AdjustmentState:
    SPLIT_AND_DIVIDEND_ADJUSTED = "SPLIT_AND_DIVIDEND_ADJUSTED"
    UNADJUSTED = "UNADJUSTED"
    UNKNOWN = "UNKNOWN"


class StructuralQuality:
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    CORRUPT = "CORRUPT"
    UNKNOWN = "UNKNOWN"


# ── Structural Quality Assessment (Descriptive Only) ──────────────────────────

def assess_structural_quality(candles: Any) -> str:
    """Assess structural validity and geometry of market candle data.
    
    IMPORTANT INVARIANT:
    This assessment is purely descriptive metadata. It must NEVER:
    1. Filter or drop rows
    2. Trigger synthetic fallbacks
    3. Modify numeric computations
    4. Veto trading decisions
    """
    if candles is None:
        return StructuralQuality.UNKNOWN

    if isinstance(candles, pd.DataFrame):
        if candles.empty:
            return StructuralQuality.UNKNOWN
        cols = {str(c).lower(): c for c in candles.columns}
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(cols.keys())):
            return StructuralQuality.PARTIAL
        o = candles[cols["open"]]
        h = candles[cols["high"]]
        l = candles[cols["low"]]
        c = candles[cols["close"]]
        if (
            o.isna().any() or h.isna().any() or l.isna().any() or c.isna().any()
            or np.isinf(o).any() or np.isinf(h).any() or np.isinf(l).any() or np.isinf(c).any()
        ):
            return StructuralQuality.CORRUPT
        if (
            (h < l).any() or (h < o).any() or (h < c).any()
            or (l > o).any() or (l > c).any() or (l < 0).any()
        ):
            return StructuralQuality.CORRUPT
        return StructuralQuality.COMPLETE

    if isinstance(candles, list):
        if not candles:
            return StructuralQuality.UNKNOWN
        for bar in candles:
            if not isinstance(bar, dict):
                return StructuralQuality.CORRUPT
            vals = {}
            for k in ["open", "high", "low", "close", "Open", "High", "Low", "Close"]:
                if k in bar:
                    vals[k.lower()] = bar[k]
            if len(vals) < 4:
                return StructuralQuality.PARTIAL
            try:
                o_val = float(vals["open"])
                h_val = float(vals["high"])
                l_val = float(vals["low"])
                c_val = float(vals["close"])
            except (ValueError, TypeError):
                return StructuralQuality.CORRUPT
            if (
                math.isnan(o_val) or math.isnan(h_val) or math.isnan(l_val) or math.isnan(c_val)
                or math.isinf(o_val) or math.isinf(h_val) or math.isinf(l_val) or math.isinf(c_val)
            ):
                return StructuralQuality.CORRUPT
            if h_val < l_val or h_val < o_val or h_val < c_val or l_val > o_val or l_val > c_val or l_val < 0:
                return StructuralQuality.CORRUPT
        return StructuralQuality.COMPLETE

    return StructuralQuality.UNKNOWN


# ── Persisted Provenance Dataclass ───────────────────────────────────────────

@dataclass(frozen=True)
class MarketProvenance:
    """Immutable factual historical provenance persisted in the sidecar table.
    
    Represents facts known at ingestion time. Does NOT contain request-time
    serving metadata like served_at or request cache hits.
    """
    provider: str = Provider.UNKNOWN
    ingestion_source: str = IngestionSource.UNKNOWN
    observed_at: Optional[str] = None
    observed_date: Optional[str] = None
    observation_precision: str = ObservationPrecision.UNKNOWN
    observation_source: str = ObservationSource.UNKNOWN
    ingested_at: Optional[str] = None
    adjustment_state: str = AdjustmentState.UNKNOWN
    structural_quality: str = StructuralQuality.UNKNOWN
    fallback_status: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "ingestion_source": self.ingestion_source,
            "observed_at": self.observed_at,
            "observed_date": self.observed_date,
            "observation_precision": self.observation_precision,
            "observation_source": self.observation_source,
            "ingested_at": self.ingested_at,
            "adjustment_state": self.adjustment_state,
            "structural_quality": self.structural_quality,
            "fallback_status": self.fallback_status,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MarketProvenance":
        if not data:
            return cls()
        return cls(
            provider=str(data.get("provider", Provider.UNKNOWN)),
            ingestion_source=str(data.get("ingestion_source", IngestionSource.UNKNOWN)),
            observed_at=data.get("observed_at"),
            observed_date=data.get("observed_date"),
            observation_precision=str(data.get("observation_precision", ObservationPrecision.UNKNOWN)),
            observation_source=str(data.get("observation_source", ObservationSource.UNKNOWN)),
            ingested_at=data.get("ingested_at"),
            adjustment_state=str(data.get("adjustment_state", AdjustmentState.UNKNOWN)),
            structural_quality=str(data.get("structural_quality", StructuralQuality.UNKNOWN)),
            fallback_status=bool(data.get("fallback_status", False)),
        )


# ── Request Evidence Envelope Dataclass ──────────────────────────────────────

@dataclass(frozen=True)
class MarketEvidence:
    """Ephemeral request-time evidence envelope attached to API responses.
    
    Combines immutable underlying MarketProvenance with transient request context
    (serving_source, cache_origin, served_at, candle_count).
    """
    provenance: MarketProvenance
    serving_source: str = ServingSource.LOCAL_CACHE
    cache_origin: str = CacheOrigin.NONE
    served_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    candle_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "serving_source": self.serving_source,
            "cache_origin": self.cache_origin,
            "served_at": self.served_at,
            "candle_count": self.candle_count,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MarketEvidence":
        if not data:
            return cls(provenance=MarketProvenance())
        raw_prov = data.get("provenance", {})
        prov = raw_prov if isinstance(raw_prov, MarketProvenance) else MarketProvenance.from_dict(raw_prov)
        return cls(
            provenance=prov,
            serving_source=str(data.get("serving_source", ServingSource.LOCAL_CACHE)),
            cache_origin=str(data.get("cache_origin", CacheOrigin.NONE)),
            served_at=str(data.get("served_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))),
            candle_count=int(data.get("candle_count", 0)),
        )


# ── Canonical Construction Helpers ───────────────────────────────────────────

def create_legacy_provenance(
    trade_date: Optional[str] = None,
    structural_quality: str = StructuralQuality.COMPLETE,
) -> MarketProvenance:
    """Factual representation for historical rows without sidecar provenance."""
    return MarketProvenance(
        provider=Provider.UNKNOWN,
        ingestion_source=IngestionSource.UNKNOWN,
        observed_at=None,
        observed_date=trade_date,
        observation_precision=ObservationPrecision.DATE if trade_date else ObservationPrecision.UNKNOWN,
        observation_source=ObservationSource.DERIVED_FROM_TRADE_DATE if trade_date else ObservationSource.UNKNOWN,
        ingested_at=None,
        adjustment_state=AdjustmentState.UNKNOWN,
        structural_quality=structural_quality,
        fallback_status=False,
    )


def create_direct_evidence(
    provider: str,
    candles: Any,
    observed_at: Optional[str] = None,
    observed_date: Optional[str] = None,
    observation_precision: str = ObservationPrecision.DATE,
    observation_source: str = ObservationSource.DERIVED_FROM_TRADE_DATE,
    adjustment_state: str = AdjustmentState.SPLIT_AND_DIVIDEND_ADJUSTED,
    fallback_status: bool = False,
    candle_count: Optional[int] = None,
) -> MarketEvidence:
    """Helper to build MarketEvidence for live direct provider fetches."""
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    quality = assess_structural_quality(candles)
    count = candle_count if candle_count is not None else (len(candles) if candles is not None else 0)

    # Derive observed_date from DataFrame or list if not provided
    if observed_date is None:
        if isinstance(candles, pd.DataFrame) and not candles.empty:
            last_idx = candles.index[-1]
            observed_date = last_idx.strftime("%Y-%m-%d") if hasattr(last_idx, "strftime") else str(last_idx).split("T")[0]
        elif isinstance(candles, list) and candles:
            last_item = candles[-1]
            if isinstance(last_item, dict):
                t = last_item.get("time") or last_item.get("trade_date") or last_item.get("date")
                if t:
                    observed_date = str(t).split("T")[0]

    prov = MarketProvenance(
        provider=provider,
        ingestion_source=IngestionSource.FALLBACK_PROVIDER if fallback_status else IngestionSource.DIRECT_PROVIDER,
        observed_at=observed_at,
        observed_date=observed_date,
        observation_precision=observation_precision,
        observation_source=observation_source,
        ingested_at=now_utc,
        adjustment_state=adjustment_state,
        structural_quality=quality,
        fallback_status=fallback_status,
    )

    return MarketEvidence(
        provenance=prov,
        serving_source=ServingSource.FALLBACK_PROVIDER if fallback_status else ServingSource.DIRECT_PROVIDER,
        cache_origin=CacheOrigin.NONE,
        served_at=now_utc,
        candle_count=count,
    )


def create_cached_evidence(
    provenance: MarketProvenance,
    cache_origin: str = CacheOrigin.SQLITE_MARKET_STORE,
    candle_count: int = 0,
) -> MarketEvidence:
    """Helper to construct MarketEvidence when reading from a local cache."""
    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    serving = ServingSource.FALLBACK_CACHE if provenance.fallback_status else ServingSource.LOCAL_CACHE
    return MarketEvidence(
        provenance=provenance,
        serving_source=serving,
        cache_origin=cache_origin,
        served_at=now_utc,
        candle_count=candle_count,
    )
