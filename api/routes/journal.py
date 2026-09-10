"""FastAPI Router for Journal Trade Executions & Authoritative Behavioral Risk Telemetry."""

import re
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response, Header
from pydantic import BaseModel, Field
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine

logger = logging.getLogger("api.routes.journal")

router = APIRouter()
history_db = HistoryDatabaseEngine()

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")


class JournalTradeItem(BaseModel):
    symbol: str = Field(..., description="Ticker symbol e.g. NVDA, AAPL")
    setupName: Optional[str] = Field("Stage 2 Breakout", description="Setup name or pattern")
    entryPrice: float = Field(..., gt=0, description="Entry execution price per share")
    exitPrice: Optional[float] = Field(None, gt=0, description="Exit execution price per share")
    shares: float = Field(..., gt=0, description="Number of shares executed")
    rAchieved: Optional[float] = Field(0.0, description="Realized R-multiple profit or loss")
    followedRules: Optional[bool] = Field(True, description="Whether execution strictly followed trade plan")
    confidence: Optional[float] = Field(70.0, ge=0.0, le=100.0, description="Pre-trade subjective confidence percentage")
    pnl: Optional[float] = Field(0.0, description="Realized net profit or loss in USD")
    status: Optional[str] = Field("CLOSED", description="Trade status e.g. OPEN, CLOSED")
    entryDate: Optional[str] = Field(None, description="Trade execution date YYYY-MM-DD")


def _resolve_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Derive user or session identifier from header or default."""
    if x_user_id and x_user_id.strip():
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", x_user_id.strip())
        if cleaned:
            return cleaned[:64]
    return "default_user"


@router.get("/telemetry", response_model=Dict[str, Any])
def get_risk_telemetry(
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Retrieve authoritative behavioral risk telemetry derived directly from persistent trades and portfolio."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)
    try:
        return history_db.get_risk_telemetry(user_id)
    except Exception as e:
        logger.error(f"Database error calculating risk telemetry for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure deriving behavioral risk telemetry.",
        )


@router.get("/trades", response_model=List[Dict[str, Any]])
def get_trades(
    response: Response = None,
    limit: int = Query(50, ge=1, le=500),
    x_user_id: Optional[str] = Header(None),
):
    """Retrieve chronological trade log for the user."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "private, no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)
    try:
        return history_db.get_journal_trades(user_id, limit=limit)
    except Exception as e:
        logger.error(f"Database error retrieving journal trades for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure retrieving trade logs.",
        )


@router.post("/trades", response_model=Dict[str, Any])
def log_trade(
    trade: JournalTradeItem,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Record an executed trade plan in the persistent journal."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)

    sym = trade.symbol.strip().upper()
    if not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    try:
        trade_dict = trade.model_dump() if hasattr(trade, "model_dump") else trade.dict()
        trade_dict["symbol"] = sym
        return history_db.save_journal_trade(user_id, trade_dict)
    except Exception as e:
        logger.error(f"Database error logging trade for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure saving trade log.",
        )
