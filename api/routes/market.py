"""
api/routes/market.py

Dedicated Market and Instrument Classification API Router for ARX Terminal.
Exposes canonical server-owned Security Master classification and execution eligibility.

Invariants Enforced:
- Response conforms strictly to CanonicalInstrument schema.
- Zero secrets, raw API keys, or authorization headers are exposed.
- Fail-closed semantics for invalid or unverified instruments.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from analyst_dashboard.security_master import (
    CanonicalInstrument,
    get_security_master_service,
)

logger = logging.getLogger("api.routes.market")
router = APIRouter()

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")


class InstrumentResponse(BaseModel):
    symbol: str
    provider_symbol: str
    asset_class: str
    security_type: str
    primary_exchange: str
    listing_status: str
    classification_status: str
    execution_eligibility: str
    analytics_capability: str
    classification_authority: str
    classification_timestamp: str
    stable_identifiers: Dict[str, Optional[str]] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


@router.get("/instruments/{symbol}", response_model=InstrumentResponse)
def get_instrument(
    symbol: str,
    response: Response = None,
):
    """
    Retrieves the authoritative canonical instrument classification and execution eligibility.
    """
    clean_sym = symbol.strip().upper()
    if not SYMBOL_REGEX.match(clean_sym):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format '{symbol}'. Must be 1-16 alphanumeric characters."
        )

    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300"

    try:
        service = get_security_master_service()
        instrument = service.get_or_resolve_instrument(clean_sym)
        return InstrumentResponse(
            symbol=instrument.symbol,
            provider_symbol=instrument.provider_symbol,
            asset_class=instrument.asset_class.value if hasattr(instrument.asset_class, "value") else str(instrument.asset_class),
            security_type=instrument.security_type.value if hasattr(instrument.security_type, "value") else str(instrument.security_type),
            primary_exchange=instrument.primary_exchange,
            listing_status=instrument.listing_status.value if hasattr(instrument.listing_status, "value") else str(instrument.listing_status),
            classification_status=instrument.classification_status.value if hasattr(instrument.classification_status, "value") else str(instrument.classification_status),
            execution_eligibility=instrument.execution_eligibility.value if hasattr(instrument.execution_eligibility, "value") else str(instrument.execution_eligibility),
            analytics_capability=instrument.analytics_capability.value if hasattr(instrument.analytics_capability, "value") else str(instrument.analytics_capability),
            classification_authority=instrument.classification_authority,
            classification_timestamp=instrument.classification_timestamp,
            stable_identifiers=instrument.stable_identifiers,
        )
    except Exception as e:
        logger.error(f"Failed to resolve instrument for {clean_sym}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while resolving instrument classification."
        )
