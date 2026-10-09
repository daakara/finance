"""
analyst_dashboard/security_master/alpaca_adapter.py

Alpaca Reference Asset Directory Adapter for ARX Terminal Security Master.
Authoritative source for asset identity, primary exchange, listing/activity status, and broad class.

Invariants Enforced:
- ALPACA_BROAD_CLASS_IS_SUBTYPE_AUTHORITY = NO
- Provider failure returns unresolved evidence, never a permissive default.
- Secret safety: API keys and auth headers are never logged or exposed.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import requests

from .models import AssetClass, ListingStatus
from .source_governance_models import (
    RawSourceRecord,
    RawSourceSnapshot,
    SnapshotStatus,
)

logger = logging.getLogger("arx.security_master.alpaca")


class AlpacaAssetEvidence:
    """Structured evidence payload returned by Alpaca reference directory probe."""

    def __init__(
        self,
        symbol: str,
        success: bool,
        provider_asset_id: Optional[str] = None,
        broad_asset_class: AssetClass = AssetClass.UNKNOWN,
        primary_exchange: str = "UNKNOWN",
        listing_status: ListingStatus = ListingStatus.UNVERIFIED,
        tradability_metadata: Optional[Dict[str, Any]] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        observed_at: Optional[str] = None,
    ):
        self.symbol = symbol
        self.success = success
        self.provider_asset_id = provider_asset_id
        self.broad_asset_class = broad_asset_class
        self.primary_exchange = primary_exchange
        self.listing_status = listing_status
        self.tradability_metadata = tradability_metadata or {}
        self.raw_payload = raw_payload or {}
        self.error_message = error_message
        self.observed_at = observed_at or datetime.now(timezone.utc).isoformat()

    def to_provenance(self) -> Dict[str, Any]:
        """Sanitized provenance dictionary safe for audit storage (secrets redacted)."""
        return {
            "provider": "ALPACA_ASSET_DIRECTORY",
            "success": self.success,
            "provider_asset_id": self.provider_asset_id,
            "broad_asset_class": self.broad_asset_class.value,
            "primary_exchange": self.primary_exchange,
            "listing_status": self.listing_status.value,
            "tradability_metadata": self.tradability_metadata,
            "raw_payload": self.raw_payload,
            "error_message": self.error_message,
            "observed_at": self.observed_at,
        }


class AlpacaIdentityAdapter:
    """
    Adapter for querying Alpaca Paper/Live Asset Directory (/v2/assets/{symbol}).
    Authoritative solely for identity, exchange, active listing status, and broad asset class.
    """

    DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2/assets"

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        api_secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 4.0,
        session: Optional[requests.Session] = None,
    ):
        self.api_key_id = (os.getenv("ALPACA_API_KEY_ID", "") if api_key_id is None else api_key_id).strip()
        self.api_secret_key = (os.getenv("ALPACA_API_SECRET_KEY", "") if api_secret_key is None else api_secret_key).strip()
        self.base_url = (base_url or os.getenv("ALPACA_API_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key_id and self.api_secret_key)

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
            "Accept": "application/json",
        }

    def fetch_asset_evidence(self, symbol: str) -> AlpacaAssetEvidence:
        """
        Fetches asset reference metadata for symbol.
        Fails closed on missing credentials, HTTP errors, timeouts, or 404s.
        """
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return AlpacaAssetEvidence(
                symbol="",
                success=False,
                error_message="EMPTY_SYMBOL",
                listing_status=ListingStatus.UNVERIFIED,
            )

        if not self.is_configured:
            logger.debug("Alpaca credentials missing in environment; returning unverified evidence.")
            return AlpacaAssetEvidence(
                symbol=clean_symbol,
                success=False,
                error_message="ALPACA_CREDENTIALS_UNCONFIGURED",
                listing_status=ListingStatus.UNVERIFIED,
            )

        url = f"{self.base_url}/{clean_symbol}"
        try:
            resp = self.session.get(url, headers=self._headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                raw_class = data.get("class", "").lower()
                raw_status = data.get("status", "").lower()

                # Normalize broad class
                if raw_class == "us_equity":
                    broad_class = AssetClass.EQUITY
                elif raw_class == "crypto":
                    broad_class = AssetClass.CRYPTO
                else:
                    broad_class = AssetClass.OTHER

                # Normalize listing status
                if raw_status == "active":
                    listing_status = ListingStatus.ACTIVE
                elif raw_status == "inactive":
                    listing_status = ListingStatus.INACTIVE
                else:
                    listing_status = ListingStatus.UNKNOWN

                tradability = {
                    "tradable": bool(data.get("tradable", False)),
                    "shortable": bool(data.get("shortable", False)),
                    "fractionable": bool(data.get("fractionable", False)),
                    "marginable": bool(data.get("marginable", False)),
                }

                # Sanitize raw payload (ensure no secrets)
                safe_raw = {
                    "id": data.get("id"),
                    "class": data.get("class"),
                    "exchange": data.get("exchange"),
                    "symbol": data.get("symbol"),
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "tradable": data.get("tradable"),
                }

                return AlpacaAssetEvidence(
                    symbol=data.get("symbol", clean_symbol),
                    success=True,
                    provider_asset_id=data.get("id"),
                    broad_asset_class=broad_class,
                    primary_exchange=data.get("exchange", "UNKNOWN"),
                    listing_status=listing_status,
                    tradability_metadata=tradability,
                    raw_payload=safe_raw,
                )
            elif resp.status_code == 404:
                return AlpacaAssetEvidence(
                    symbol=clean_symbol,
                    success=False,
                    error_message=f"ASSET_NOT_FOUND_404",
                    listing_status=ListingStatus.UNKNOWN,
                )
            else:
                return AlpacaAssetEvidence(
                    symbol=clean_symbol,
                    success=False,
                    error_message=f"HTTP_{resp.status_code}",
                    listing_status=ListingStatus.UNVERIFIED,
                )
        except requests.exceptions.Timeout:
            logger.warning(f"Alpaca asset directory request timed out for {clean_symbol}")
            return AlpacaAssetEvidence(
                symbol=clean_symbol,
                success=False,
                error_message="TIMEOUT",
                listing_status=ListingStatus.UNVERIFIED,
            )
        except Exception as e:
            logger.error(f"Alpaca asset directory request failed for {clean_symbol}: {type(e).__name__}")
            return AlpacaAssetEvidence(
                symbol=clean_symbol,
                success=False,
                error_message=f"EXCEPTION_{type(e).__name__}",
                listing_status=ListingStatus.UNVERIFIED,
            )

    def enumerate_active_us_equities(self) -> List[Dict[str, Any]]:
        """
        Enumerates all active US equities from Alpaca asset directory.
        Fails closed on missing credentials, network errors, timeouts, or non-200.
        """
        if not self.is_configured:
            logger.warning("Alpaca credentials missing in environment; cannot enumerate assets.")
            return []

        url = self.base_url
        params = {"status": "active", "asset_class": "us_equity"}
        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=max(self.timeout, 15.0))
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            logger.error(f"Alpaca asset directory enumeration failed with HTTP {resp.status_code}")
            return []
        except Exception as e:
            logger.error(f"Alpaca asset directory enumeration exception: {type(e).__name__}")
            return []

    def build_raw_source_snapshot(
        self,
        as_of: str = "2026-10-09T00:00:00Z",
        snapshot_id: Optional[str] = None,
        implementation_sha: str = "c868115c7b31b8de6daf4daca47ede049b5bb23b",
    ) -> RawSourceSnapshot:
        """
        Fetches active US equity population and packages into immutable RawSourceSnapshot.
        """
        raw_items = self.enumerate_active_us_equities()
        now_iso = datetime.now(timezone.utc).isoformat()
        snap_id = snapshot_id or f"SNAP_ALPACA_{as_of[:10].replace('-', '')}_{len(raw_items)}"

        records: List[RawSourceRecord] = []
        for item in raw_items:
            rec_id = str(item.get("id", ""))
            sym = str(item.get("symbol", ""))
            r = RawSourceRecord(
                source_id="ALPACA_ASSET_DIRECTORY",
                source_snapshot_id=snap_id,
                provider_record_id=rec_id,
                provider_symbol=sym,
                raw_payload=item,
                observed_at=now_iso,
                effective_as_of=as_of,
            )
            records.append(r)

        snapshot = RawSourceSnapshot(
            source_snapshot_id=snap_id,
            source_id="ALPACA_ASSET_DIRECTORY",
            source_authority_version="1.0.0",
            retrieved_at=now_iso,
            effective_as_of=as_of,
            population_temporal_scope="CURRENT_PROVIDER_POPULATION",
            records=records,
            raw_record_count=len(records),
            snapshot_status=SnapshotStatus.VALID if records else SnapshotStatus.QUARANTINED,
            implementation_sha=implementation_sha,
        )
        pop_hash = snapshot.compute_population_hash()
        object.__setattr__(snapshot, "source_population_hash", pop_hash)
        return snapshot
