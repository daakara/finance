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
    setupName: Optional[str] = Field(None, description="Setup name or pattern")
    entryPrice: float = Field(..., gt=0, description="Entry execution price per share")
    exitPrice: Optional[float] = Field(None, gt=0, description="Exit execution price per share")
    shares: float = Field(..., gt=0, description="Number of shares executed")
    rAchieved: Optional[float] = Field(None, description="Realized R-multiple profit or loss")
    followedRules: Optional[bool] = Field(None, description="Whether execution strictly followed trade plan")
    confidence: Optional[float] = Field(None, ge=0.0, le=100.0, description="Pre-trade subjective confidence percentage")
    pnl: Optional[float] = Field(None, description="Realized net profit or loss in USD")
    status: str = Field(..., description="Explicit trade lifecycle status e.g. OPEN, CLOSED")
    entryDate: Optional[str] = Field(None, description="Trade execution date YYYY-MM-DD")
    exitDate: Optional[str] = Field(None, description="Trade closing date YYYY-MM-DD")
    remainingShares: Optional[float] = Field(None, ge=0, description="Remaining open shares count")
    parentTradeId: Optional[int] = Field(None, description="Parent trade ID for multi-leg exits")
    executionRole: Optional[str] = Field(None, description="Role: ENTRY, PARTIAL_EXIT, or FULL_EXIT")
    idempotencyKey: Optional[str] = Field(None, description="Unique client idempotency token")
    notes: Optional[str] = Field(None, description="Trade execution notes")
    target1: Optional[float] = Field(None, gt=0, description="Profit target 1")
    stopLoss: Optional[float] = Field(None, gt=0, description="Stop loss price")


class RecordFillRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol e.g. NVDA, AAPL")
    setupName: Optional[str] = Field(None, description="Setup name or pattern")
    entryPrice: float = Field(..., gt=0, description="Actual broker execution fill price per share")
    shares: float = Field(..., gt=0, description="Number of shares filled")
    stopLoss: Optional[float] = Field(None, gt=0, description="Protective stop loss price")
    target1: Optional[float] = Field(None, gt=0, description="Initial profit target price")
    confidence: Optional[float] = Field(None, ge=0.0, le=100.0, description="Subjective setup confidence percentage")
    entryDate: Optional[str] = Field(None, description="Execution date YYYY-MM-DD")
    idempotencyKey: Optional[str] = Field(None, description="Unique client idempotency token")
    notes: Optional[str] = Field(None, description="Trade execution notes")


class RecordExitRequest(BaseModel):
    tradeId: Optional[int] = Field(None, description="Journal trade ID to exit")
    symbol: Optional[str] = Field(None, description="Ticker symbol to exit if tradeId not supplied")
    exitPrice: float = Field(..., gt=0, description="Exit execution price per share")
    shares: Optional[float] = Field(None, gt=0, description="Number of shares exited (partial or all)")
    exitDate: Optional[str] = Field(None, description="Exit execution date YYYY-MM-DD")
    followedRules: Optional[bool] = Field(None, description="Whether exit strictly adhered to trading rules")
    idempotencyKey: Optional[str] = Field(None, description="Unique client idempotency token")
    notes: Optional[str] = Field(None, description="Exit notes or post-trade reflection")


class RecordCloseRequest(BaseModel):
    tradeId: Optional[int] = Field(None, description="Journal trade ID to close completely")
    symbol: Optional[str] = Field(None, description="Ticker symbol to close if tradeId not supplied")
    exitPrice: float = Field(..., gt=0, description="Exit execution price per share")
    exitDate: Optional[str] = Field(None, description="Exit execution date YYYY-MM-DD")
    followedRules: Optional[bool] = Field(None, description="Whether trade strictly adhered to trading rules")
    idempotencyKey: Optional[str] = Field(None, description="Unique client idempotency token")
    notes: Optional[str] = Field(None, description="Closing notes or post-trade reflection")


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

    status = trade.status.strip().upper()
    if status not in ("OPEN", "CLOSED"):
        raise HTTPException(status_code=400, detail="Invalid trade lifecycle status. Must be 'OPEN' or 'CLOSED'.")

    try:
        trade_dict = trade.model_dump() if hasattr(trade, "model_dump") else trade.dict()
        trade_dict["symbol"] = sym
        trade_dict["status"] = status
        return history_db.save_journal_trade(user_id, trade_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Database error logging trade for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure saving trade log.",
        )


@router.post("/fill", response_model=Dict[str, Any])
def record_fill(
    fill: RecordFillRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Record an actual broker execution fill in the persistent journal and sync active portfolio holding."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)

    sym = fill.symbol.strip().upper()
    if not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    try:
        fill_dict = fill.model_dump() if hasattr(fill, "model_dump") else fill.dict()
        fill_dict["symbol"] = sym
        return history_db.record_trade_fill(user_id, fill_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Database error recording fill for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure recording trade fill.",
        )


@router.post("/exit", response_model=Dict[str, Any])
def record_exit(
    exit_req: RecordExitRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Record a partial scale-out or complete exit on an active open position."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)

    if not exit_req.tradeId and not exit_req.symbol:
        raise HTTPException(status_code=400, detail="Either tradeId or symbol must be specified.")

    sym = exit_req.symbol.strip().upper() if exit_req.symbol else None
    if sym and not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    try:
        exit_dict = exit_req.model_dump() if hasattr(exit_req, "model_dump") else exit_req.dict()
        if sym:
            exit_dict["symbol"] = sym
        return history_db.record_trade_exit(user_id, exit_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Database error recording exit for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure recording trade exit.",
        )


@router.post("/close", response_model=Dict[str, Any])
def record_close(
    close_req: RecordCloseRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
):
    """Convenience endpoint to close 100% of an active open holding."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    user_id = _resolve_user_id(x_user_id)

    if not close_req.tradeId and not close_req.symbol:
        raise HTTPException(status_code=400, detail="Either tradeId or symbol must be specified.")

    sym = close_req.symbol.strip().upper() if close_req.symbol else None
    if sym and not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.")

    try:
        close_dict = close_req.model_dump() if hasattr(close_req, "model_dump") else close_req.dict()
        if sym:
            close_dict["symbol"] = sym
        # shares=None in record_trade_exit defaults to all parent remaining shares
        close_dict["shares"] = None
        return history_db.record_trade_exit(user_id, close_dict)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Database error closing position for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure closing position.",
        )
