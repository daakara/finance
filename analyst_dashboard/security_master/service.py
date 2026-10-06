"""
analyst_dashboard/security_master/service.py

High-level orchestrating service for ARX Terminal Security Master.
Coordinates caching, provider resolution, normalization, and execution eligibility.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import CanonicalInstrument
from .alpaca_adapter import AlpacaIdentityAdapter
from .openfigi_adapter import OpenFIGISubtypeAdapter
from .normalization import SecurityMasterNormalizationEngine
from .persistence import SecurityMasterRepository
from scripts.research.etf_v2.openfigi_rate_limiter import GlobalSQLiteRateLimiter

logger = logging.getLogger("arx.security_master.service")

_GLOBAL_SERVICE: Optional[SecurityMasterService] = None


class SecurityMasterService:
    """
    Central server-owned security master service.
    Resolves instruments authoritatively, preserves fail-closed invariants, and coordinates quotas.
    """

    def __init__(
        self,
        alpaca_adapter: Optional[AlpacaIdentityAdapter] = None,
        openfigi_adapter: Optional[OpenFIGISubtypeAdapter] = None,
        normalization_engine: Optional[SecurityMasterNormalizationEngine] = None,
        repository: Optional[SecurityMasterRepository] = None,
    ):
        self.alpaca_adapter = alpaca_adapter or AlpacaIdentityAdapter()
        rate_limiter = GlobalSQLiteRateLimiter()
        self.openfigi_adapter = openfigi_adapter or OpenFIGISubtypeAdapter(rate_limiter=rate_limiter)
        self.normalization_engine = normalization_engine or SecurityMasterNormalizationEngine()
        self.repository = repository or SecurityMasterRepository()

    def get_or_resolve_instrument(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> CanonicalInstrument:
        """
        Retrieves canonical instrument from cache/persistence, or authoritatively resolves
        it from provider adapters and normalizes according to field-level precedence.
        """
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return self.normalization_engine.normalize("", None, None)

        if not force_refresh:
            cached = self.repository.get(clean_symbol)
            if cached is not None:
                return cached

        # Resolve live from providers
        alpaca_evidence = self.alpaca_adapter.fetch_asset_evidence(clean_symbol)
        openfigi_evidence = self.openfigi_adapter.fetch_subtype_evidence(clean_symbol)

        instrument = self.normalization_engine.normalize(
            symbol=clean_symbol,
            alpaca_evidence=alpaca_evidence,
            openfigi_evidence=openfigi_evidence,
        )

        # Persist resolved record
        try:
            self.repository.save(instrument)
        except Exception as e:
            logger.error(f"Failed to persist Security Master record for {clean_symbol}: {e}")

        return instrument


def get_security_master_service() -> SecurityMasterService:
    """Returns singleton instance of SecurityMasterService."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        _GLOBAL_SERVICE = SecurityMasterService()
    return _GLOBAL_SERVICE


def set_security_master_service(service: Optional[SecurityMasterService]) -> None:
    """Sets or overrides global SecurityMasterService (used in tests)."""
    global _GLOBAL_SERVICE
    _GLOBAL_SERVICE = service
