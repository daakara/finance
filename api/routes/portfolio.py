"""FastAPI Router for User Portfolio Holdings with Persistent SQLite Storage."""

import re
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Response, Header
from pydantic import BaseModel, Field
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine

logger = logging.getLogger("api.routes.portfolio")

router = APIRouter()
history_db = HistoryDatabaseEngine()

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


def _resolve_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Derive user or session identifier from header or default."""
    if x_user_id and x_user_id.strip():
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", x_user_id.strip())
        if cleaned:
            return cleaned[:64]
    return "default_user"


@router.get("", response_model=List[dict])
def get_portfolio(
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Retrieve all saved holdings for the authenticated or session user."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)
    return history_db.get_user_portfolio(user_id)


@router.post("", status_code=201)
def add_holding(
    holding: HoldingItem,
    x_user_id: Optional[str] = Header(None),
):
    """Add or update a portfolio holding with fractional precision support."""
    upper_sym = holding.symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format '{holding.symbol}'. Must be 1-16 alphanumeric characters.",
        )

    user_id = _resolve_user_id(x_user_id)
    success = history_db.save_user_holding(
        user_id=user_id,
        holding={
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
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save portfolio holding to persistent storage.")

    return {"status": "saved", "symbol": upper_sym, "shares": holding.shares}


@router.put("/{symbol}")
def update_holding(
    symbol: str,
    holding: HoldingItem,
    x_user_id: Optional[str] = Header(None),
):
    """Update an existing portfolio holding."""
    upper_sym = symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol format.")

    user_id = _resolve_user_id(x_user_id)
    success = history_db.save_user_holding(
        user_id=user_id,
        holding={
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
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update holding in storage.")

    return {"status": "updated", "symbol": upper_sym}


@router.delete("/{symbol}")
def delete_holding(
    symbol: str,
    x_user_id: Optional[str] = Header(None),
):
    """Delete a holding from the portfolio."""
    upper_sym = symbol.upper().strip()
    if not SYMBOL_REGEX.match(upper_sym):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol format.")

    user_id = _resolve_user_id(x_user_id)
    success = history_db.delete_user_holding(user_id, upper_sym)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to remove holding.")

    return {"status": "deleted", "symbol": upper_sym}


@router.post("/migrate")
def migrate_holdings(
    body: BulkMigrateRequest,
    x_user_id: Optional[str] = Header(None),
):
    """Migrate client-side localStorage holdings into backend persistent database without overwriting existing entries."""
    user_id = _resolve_user_id(x_user_id)
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
    saved_count = history_db.bulk_save_holdings(user_id, items)
    return {"status": "migrated", "migratedCount": saved_count, "totalSubmitted": len(body.holdings)}
