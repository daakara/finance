"""
analyst_dashboard/security_master/openfigi_adapter.py

OpenFIGI V3 Mapping Adapter for ARX Terminal Security Master.
Authoritative source for structural security subtype, composite FIGI, share class FIGI.

Invariants Enforced:
- OFIGI-SEC-01: Normalized subtype mappings preserve fail-closed boundary.
- OFIGI-SEC-02: Unknown provider vocabulary resolves strictly to UNKNOWN / UNVERIFIED.
- OFIGI-SEC-03: Unknown subtype is never inferred as Common Stock.
- OFIGI-SEC-04: Provider rate-limit quota is globally coordinated with ETF v2.
- Secret safety: API key is never logged or exposed.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import requests

from .models import SecurityType, ClassificationStatus, AssetClass
from scripts.research.etf_v2.openfigi_rate_limiter import GlobalSQLiteRateLimiter

logger = logging.getLogger("arx.security_master.openfigi")


class OpenFIGISubtypeEvidence:
    """Structured evidence payload returned by OpenFIGI v3 mapping."""

    def __init__(
        self,
        symbol: str,
        success: bool,
        security_type: SecurityType = SecurityType.UNKNOWN,
        classification_status: ClassificationStatus = ClassificationStatus.UNVERIFIED,
        raw_security_type: Optional[str] = None,
        raw_security_type2: Optional[str] = None,
        market_sector: Optional[str] = None,
        figi: Optional[str] = None,
        composite_figi: Optional[str] = None,
        share_class_figi: Optional[str] = None,
        exch_code: Optional[str] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        observed_at: Optional[str] = None,
    ):
        self.symbol = symbol
        self.success = success
        self.security_type = security_type
        self.classification_status = classification_status
        self.raw_security_type = raw_security_type
        self.raw_security_type2 = raw_security_type2
        self.market_sector = market_sector
        self.figi = figi
        self.composite_figi = composite_figi
        self.share_class_figi = share_class_figi
        self.exch_code = exch_code
        self.raw_payload = raw_payload or {}
        self.error_message = error_message
        self.observed_at = observed_at or datetime.now(timezone.utc).isoformat()

    def to_provenance(self) -> Dict[str, Any]:
        """Sanitized provenance dictionary safe for audit storage (secrets redacted)."""
        return {
            "provider": "OPENFIGI_V3_MAPPING",
            "success": self.success,
            "security_type": self.security_type.value,
            "classification_status": self.classification_status.value,
            "raw_security_type": self.raw_security_type,
            "raw_security_type2": self.raw_security_type2,
            "market_sector": self.market_sector,
            "figi": self.figi,
            "composite_figi": self.composite_figi,
            "share_class_figi": self.share_class_figi,
            "exch_code": self.exch_code,
            "raw_payload": self.raw_payload,
            "error_message": self.error_message,
            "observed_at": self.observed_at,
        }


# Frozen normalized mapping table for OpenFIGI security types
MAPPING_TABLE = {
    "common stock": SecurityType.COMMON_STOCK,
    "adr": SecurityType.ADR,
    "depositary receipt": SecurityType.ADR,
    "reit": SecurityType.REIT,
    "etp": SecurityType.ETF,
    "mutual fund": SecurityType.ETF,
    "preferred stock": SecurityType.PREFERRED,
    "preference shares": SecurityType.PREFERRED,
    "warrant": SecurityType.WARRANT,
    "equity wrt": SecurityType.WARRANT,
    "unit": SecurityType.UNIT,
    "right": SecurityType.RIGHT,
    "crypto": SecurityType.CRYPTO,
}


class OpenFIGISubtypeAdapter:
    """
    Adapter for querying OpenFIGI v3 (/v3/mapping).
    Authoritative for structural security subtype and FIGI identifiers.
    Enforces shared global provider rate limit quota.
    """

    DEFAULT_BASE_URL = "https://api.openfigi.com/v3/mapping"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limiter: Optional[GlobalSQLiteRateLimiter] = None,
        timeout: float = 6.0,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = (api_key or os.getenv("OPENFIGI_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("OPENFIGI_API_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-OPENFIGI-KEY"] = self.api_key
        return headers

    def normalize_security_type(
        self,
        raw_type: Optional[str],
        raw_type2: Optional[str]
    ) -> Tuple[SecurityType, ClassificationStatus]:
        """
        Normalizes OpenFIGI raw security types into canonical SecurityType.
        Never infers unknown types as Common Stock.
        """
        t1 = (raw_type or "").strip().lower()
        t2 = (raw_type2 or "").strip().lower()

        # Check primary type first
        if t1 in MAPPING_TABLE:
            return MAPPING_TABLE[t1], ClassificationStatus.VERIFIED

        # Check secondary type
        if t2 in MAPPING_TABLE:
            return MAPPING_TABLE[t2], ClassificationStatus.VERIFIED

        # Partial checks for compound names
        for k, v in MAPPING_TABLE.items():
            if k in t1 or k in t2:
                return v, ClassificationStatus.VERIFIED

        # Fail closed on unrecognized vocabulary
        return SecurityType.UNKNOWN, ClassificationStatus.UNVERIFIED

    def fetch_subtype_evidence(self, symbol: str) -> OpenFIGISubtypeEvidence:
        """
        Queries OpenFIGI v3 mapping for ticker symbol.
        Respects provider-global rolling rate limit.
        """
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return OpenFIGISubtypeEvidence(
                symbol="",
                success=False,
                error_message="EMPTY_SYMBOL",
                classification_status=ClassificationStatus.UNVERIFIED,
            )

        # Coordinate global rate limit reservation before dispatching network request
        if self.rate_limiter is not None:
            try:
                self.rate_limiter.acquire(timeout=self.timeout)
            except TimeoutError:
                logger.warning(f"OpenFIGI rate limit reservation timed out for {clean_symbol}")
                return OpenFIGISubtypeEvidence(
                    symbol=clean_symbol,
                    success=False,
                    error_message="RATE_LIMIT_TIMEOUT",
                    classification_status=ClassificationStatus.UNVERIFIED,
                )

        payload = [{"idType": "TICKER", "idValue": clean_symbol, "exchCode": "US"}]

        try:
            resp = self.session.post(
                self.base_url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                results = resp.json()
                if not results or not isinstance(results, list):
                    return OpenFIGISubtypeEvidence(
                        symbol=clean_symbol,
                        success=False,
                        error_message="EMPTY_OR_MALFORMED_RESPONSE",
                        classification_status=ClassificationStatus.UNVERIFIED,
                    )

                first_entry = results[0]
                if "error" in first_entry:
                    return OpenFIGISubtypeEvidence(
                        symbol=clean_symbol,
                        success=False,
                        error_message=f"OPENFIGI_ERROR_{first_entry['error']}",
                        classification_status=ClassificationStatus.UNVERIFIED,
                    )

                if "warning" in first_entry:
                    # e.g. "No identifier found."
                    return OpenFIGISubtypeEvidence(
                        symbol=clean_symbol,
                        success=False,
                        error_message=first_entry["warning"],
                        classification_status=ClassificationStatus.UNVERIFIED,
                    )

                data_list = first_entry.get("data", [])
                if not data_list:
                    return OpenFIGISubtypeEvidence(
                        symbol=clean_symbol,
                        success=False,
                        error_message="NO_IDENTIFIER_FOUND",
                        classification_status=ClassificationStatus.UNVERIFIED,
                    )

                match = data_list[0]
                raw_type = match.get("securityType")
                raw_type2 = match.get("securityType2")
                sector = match.get("marketSector")
                figi = match.get("figi")
                composite_figi = match.get("compositeFIGI")
                share_class_figi = match.get("shareClassFIGI")
                exch_code = match.get("exchCode")

                sec_type, status = self.normalize_security_type(raw_type, raw_type2)

                safe_raw = {
                    "figi": figi,
                    "name": match.get("name"),
                    "ticker": match.get("ticker"),
                    "exchCode": exch_code,
                    "securityType": raw_type,
                    "securityType2": raw_type2,
                    "marketSector": sector,
                    "shareClassFIGI": share_class_figi,
                }

                return OpenFIGISubtypeEvidence(
                    symbol=clean_symbol,
                    success=True,
                    security_type=sec_type,
                    classification_status=status,
                    raw_security_type=raw_type,
                    raw_security_type2=raw_type2,
                    market_sector=sector,
                    figi=figi,
                    composite_figi=composite_figi,
                    share_class_figi=share_class_figi,
                    exch_code=exch_code,
                    raw_payload=safe_raw,
                )
            elif resp.status_code == 429:
                logger.warning(f"OpenFIGI HTTP 429 Too Many Requests for {clean_symbol}")
                return OpenFIGISubtypeEvidence(
                    symbol=clean_symbol,
                    success=False,
                    error_message="HTTP_429_RATE_LIMITED",
                    classification_status=ClassificationStatus.UNVERIFIED,
                )
            else:
                return OpenFIGISubtypeEvidence(
                    symbol=clean_symbol,
                    success=False,
                    error_message=f"HTTP_{resp.status_code}",
                    classification_status=ClassificationStatus.UNVERIFIED,
                )
        except requests.exceptions.Timeout:
            logger.warning(f"OpenFIGI request timed out for {clean_symbol}")
            return OpenFIGISubtypeEvidence(
                symbol=clean_symbol,
                success=False,
                error_message="TIMEOUT",
                classification_status=ClassificationStatus.UNVERIFIED,
            )
        except Exception as e:
            logger.error(f"OpenFIGI mapping request failed for {clean_symbol}: {type(e).__name__}")
            return OpenFIGISubtypeEvidence(
                symbol=clean_symbol,
                success=False,
                error_message=f"EXCEPTION_{type(e).__name__}",
                classification_status=ClassificationStatus.UNVERIFIED,
            )
