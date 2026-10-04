"""FastAPI Router for User Portfolio Holdings with Persistent SQLite Storage.

Migrated to PortfolioApplicationService boundary in Phase 1F-B.
"""

import re
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Response, Header, Depends
from pydantic import BaseModel, Field

from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
from api.context.request_context import RequestContext
from api.context.resolver import resolve_request_context
from api.services.portfolio_service import PortfolioApplicationService
from api.services.authorizer import DefaultWorkspaceAuthorizer
from api.services.entitlement_resolver import DefaultEntitlementResolver

logger = logging.getLogger("api.routes.portfolio")

router = APIRouter()
history_db = HistoryDatabaseEngine()

portfolio_service = PortfolioApplicationService(
    authorizer=DefaultWorkspaceAuthorizer(),
    entitlement_resolver=DefaultEntitlementResolver(),
    db_engine=history_db,
)

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")


class HoldingItem(BaseModel):
    symbol: str = Field(..., description="Ticker symbol e.g. NVDA, AAPL")
    name: Optional[str] = Field(None, description="Asset display name")
    shares: float = Field(..., gt=0, description="Positive holding shares count (supports fractional quantities)")
    entryPrice: float = Field(..., gt=0, description="Positive entry price per share in USD")
    currentPrice: Optional[float] = Field(None, description="Last known current price")
    targetPrice: Optional[float] = Field(None, gt=0, description="Optional target price")
    stopLossPrice: Optional[float] = Field(None, gt=0, description="Optional stop loss price")
    addedAt: Optional[str] = Field(None, description="Date added in YYYY-MM-DD")
    assetType: Optional[str] = Field("Stock", description="Asset class e.g. Stock, ETF, Crypto")


class BulkMigrateRequest(BaseModel):
    holdings: List[HoldingItem]


PRIVATE_CACHE_HEADERS = {
    "Cache-Control": "private, no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}


def _set_private_cache_headers(response: Optional[Response]) -> None:
    """Enforce private non-shared cache policy (INV-SAAS-02)."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = PRIVATE_CACHE_HEADERS["Cache-Control"]
        response.headers["Pragma"] = PRIVATE_CACHE_HEADERS["Pragma"]


@router.get("", response_model=List[dict])
def get_portfolio(
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Retrieve all saved holdings for the authenticated or session user."""
    _set_private_cache_headers(response)
    try:
        return portfolio_service.get_portfolio(context)
    except PermissionError as pe:
        logger.warning(f"Permission denied retrieving portfolio for context {context}: {pe}")
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error retrieving portfolio for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure retrieving portfolio holdings.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.post("", status_code=201)
def add_holding(
    holding: HoldingItem,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Add or update a portfolio holding with fractional precision support."""
    _set_private_cache_headers(response)
    upper_sym = holding.symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format '{holding.symbol}'. Must be 1-16 alphanumeric characters.",
            headers=PRIVATE_CACHE_HEADERS,
        )

    holding_dict = {
        "symbol": upper_sym,
        "name": holding.name or upper_sym,
        "shares": holding.shares,
        "entryPrice": holding.entryPrice,
        "currentPrice": holding.currentPrice,
        "targetPrice": holding.targetPrice,
        "stopLossPrice": holding.stopLossPrice,
        "addedAt": holding.addedAt,
        "assetType": holding.assetType or "Stock",
    }

    try:
        portfolio_service.save_holding(context, holding_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error saving holding {upper_sym} for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to save portfolio holding to persistent storage.",
            headers=PRIVATE_CACHE_HEADERS,
        )

    return {"status": "saved", "symbol": upper_sym, "shares": holding.shares}


@router.put("/{symbol}")
def update_holding(
    symbol: str,
    holding: HoldingItem,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Update an existing portfolio holding."""
    _set_private_cache_headers(response)
    upper_sym = symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol format.", headers=PRIVATE_CACHE_HEADERS)

    holding_dict = {
        "symbol": upper_sym,
        "name": holding.name or upper_sym,
        "shares": holding.shares,
        "entryPrice": holding.entryPrice,
        "currentPrice": holding.currentPrice,
        "targetPrice": holding.targetPrice,
        "stopLossPrice": holding.stopLossPrice,
        "addedAt": holding.addedAt,
        "assetType": holding.assetType or "Stock",
    }

    try:
        portfolio_service.save_holding(context, holding_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error updating holding {upper_sym} for context {context}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update holding in storage.", headers=PRIVATE_CACHE_HEADERS)

    return {"status": "updated", "symbol": upper_sym}


@router.delete("/{symbol}")
def delete_holding(
    symbol: str,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Delete a holding from the portfolio."""
    _set_private_cache_headers(response)
    upper_sym = symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol format.", headers=PRIVATE_CACHE_HEADERS)

    try:
        portfolio_service.remove_holding(context, upper_sym)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error deleting holding {upper_sym} for context {context}: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove holding.", headers=PRIVATE_CACHE_HEADERS)

    return {"status": "deleted", "symbol": upper_sym}


@router.post("/migrate")
def migrate_holdings(
    body: BulkMigrateRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Migrate client-side localStorage holdings into backend persistent database without overwriting existing entries."""
    _set_private_cache_headers(response)
    items = [
        {
            "symbol": h.symbol.upper().strip(),
            "name": h.name or h.symbol.upper().strip(),
            "shares": h.shares,
            "entryPrice": h.entryPrice,
            "currentPrice": h.currentPrice,
            "targetPrice": h.targetPrice,
            "stopLossPrice": h.stopLossPrice,
            "addedAt": h.addedAt,
            "assetType": h.assetType or "Stock",
        }
        for h in body.holdings
        if SYMBOL_REGEX.match(h.symbol.upper().strip()) and h.shares > 0 and h.entryPrice > 0
    ]
    total_submitted = len(body.holdings)
    try:
        saved_count = portfolio_service.bulk_migrate_holdings(context, items)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error during portfolio migration for context {context}: {e}")
        raise HTTPException(status_code=500, detail="Failed to migrate holdings due to database error.", headers=PRIVATE_CACHE_HEADERS)

    if total_submitted > 0 and saved_count == 0:
        logger.error(f"Migration failed completely for context {context}: 0 of {total_submitted} persisted.")
        raise HTTPException(
            status_code=500,
            detail=f"Migration failed: 0 of {total_submitted} holdings could be persisted.",
            headers=PRIVATE_CACHE_HEADERS,
        )

    status = "migrated" if saved_count == total_submitted else ("partial" if saved_count > 0 else "no_op")
    return {
        "status": status,
        "migratedCount": saved_count,
        "totalSubmitted": total_submitted,
        "failedCount": total_submitted - saved_count,
    }
