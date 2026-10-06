"""
api/routes/security_master.py

Canonical Server Security Master API Routes.
Exposes authoritative instrument classification and execution eligibility to internal and client consumers.
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Response
from pydantic import BaseModel, Field

from analyst_dashboard.security_master import (
    CanonicalInstrument,
    SecurityMasterService,
    get_security_master_service,
)

logger = logging.getLogger("api.routes.security_master")
router = APIRouter()


class BatchInstrumentsRequest(BaseModel):
    symbols: List[str] = Field(..., max_length=100, description="List of symbols to resolve")
    force_refresh: bool = Field(default=False, description="Whether to bypass cache and refresh from providers")


class BatchInstrumentsResponse(BaseModel):
    instruments: Dict[str, CanonicalInstrument]
    count: int


@router.get(
    "/instruments/{symbol}",
    response_model=CanonicalInstrument,
    summary="Get canonical instrument classification and execution eligibility",
)
def get_canonical_instrument(
    symbol: str,
    force_refresh: bool = Query(default=False, description="Bypass cache and force provider refresh"),
    response: Response = None,
) -> CanonicalInstrument:
    """
    Returns canonical server-owned classification and execution eligibility for a symbol.
    Decouples identity (Alpaca), security subtype (OpenFIGI), and execution eligibility (policy).
    """
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        raise HTTPException(status_code=400, detail="Symbol cannot be empty.")

    service = get_security_master_service()
    instrument = service.get_or_resolve_instrument(clean_sym, force_refresh=force_refresh)

    if response is not None:
        # Client cache-control: 5 minutes private cache for verified instruments
        status_val = (
            instrument.classification_status.value
            if hasattr(instrument.classification_status, "value")
            else str(instrument.classification_status)
        )
        if status_val == "VERIFIED":
            response.headers["Cache-Control"] = "private, max-age=300"
        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    return instrument


@router.post(
    "/instruments/batch",
    response_model=BatchInstrumentsResponse,
    summary="Batch resolve canonical instruments",
)
def batch_resolve_instruments(
    request: BatchInstrumentsRequest = Body(...),
) -> BatchInstrumentsResponse:
    """
    Resolves multiple instruments in a single call.
    """
    service = get_security_master_service()
    result = {}
    for sym in request.symbols:
        clean = sym.strip().upper()
        if clean:
            result[clean] = service.get_or_resolve_instrument(clean, force_refresh=request.force_refresh)

    return BatchInstrumentsResponse(instruments=result, count=len(result))


@router.get("/status", summary="Security Master subsystem status")
def security_master_status():
    """
    Returns diagnostic health and provider configuration state.
    """
    service = get_security_master_service()
    return {
        "status": "online",
        "subsystem": "ARX_SERVER_SECURITY_MASTER",
        "authorities": {
            "identity_and_listing": "ALPACA_ASSET_DIRECTORY",
            "security_subtype": "OPENFIGI_V3_MAPPING",
            "execution_eligibility": "CANONICAL_PRECONDITION_RULES_ENGINE",
        },
        "alpaca_configured": getattr(service.alpaca_adapter, "is_configured", False),
        "openfigi_configured": getattr(service.openfigi_adapter, "is_configured", False),
        "db_path": str(service.repository.db_path),
        "lru_capacity": service.repository.lru_cache.capacity,
    }
