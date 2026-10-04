"""FastAPI Router for Journal Trade Executions & Authoritative Behavioral Risk Telemetry.

Migrated to JournalApplicationService boundary in Phase 1F-B.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response, Header, Depends
from pydantic import BaseModel, Field

from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
from api.context.request_context import RequestContext
from api.context.resolver import resolve_request_context
from api.services.journal_service import JournalApplicationService
from api.services.authorizer import DefaultWorkspaceAuthorizer
from api.services.entitlement_resolver import DefaultEntitlementResolver

logger = logging.getLogger("api.routes.journal")

router = APIRouter()
history_db = HistoryDatabaseEngine()

journal_service = JournalApplicationService(
    authorizer=DefaultWorkspaceAuthorizer(),
    entitlement_resolver=DefaultEntitlementResolver(),
    db_engine=history_db,
)

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


PRIVATE_CACHE_HEADERS = {
    "Cache-Control": "private, no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}


def _set_private_cache_headers(response: Optional[Response]) -> None:
    """Enforce private non-shared cache policy (INV-SAAS-02)."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = PRIVATE_CACHE_HEADERS["Cache-Control"]
        response.headers["Pragma"] = PRIVATE_CACHE_HEADERS["Pragma"]


@router.get("/telemetry", response_model=Dict[str, Any])
def get_risk_telemetry(
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Retrieve authoritative behavioral risk telemetry derived directly from persistent trades and portfolio."""
    _set_private_cache_headers(response)
    try:
        return journal_service.get_telemetry(context)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error calculating risk telemetry for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure deriving behavioral risk telemetry.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.get("/trades", response_model=List[Dict[str, Any]])
def get_trades(
    response: Response = None,
    limit: int = Query(50, ge=1, le=500),
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Retrieve chronological trade log for the user."""
    _set_private_cache_headers(response)
    try:
        return journal_service.get_trades(context, limit=limit)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error retrieving journal trades for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure retrieving trade logs.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.post("/trades", response_model=Dict[str, Any])
def log_trade(
    trade: JournalTradeItem,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Record an executed trade plan in the persistent journal."""
    _set_private_cache_headers(response)

    sym = trade.symbol.strip().upper()
    if not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.", headers=PRIVATE_CACHE_HEADERS)

    status_str = trade.status.strip().upper()
    if status_str not in ("OPEN", "CLOSED"):
        raise HTTPException(status_code=400, detail="Invalid trade lifecycle status. Must be 'OPEN' or 'CLOSED'.", headers=PRIVATE_CACHE_HEADERS)

    trade_dict = trade.model_dump() if hasattr(trade, "model_dump") else trade.dict()
    trade_dict["symbol"] = sym
    trade_dict["status"] = status_str

    try:
        return journal_service.record_trade(context, trade_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error logging trade for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure saving trade log.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.post("/fill", response_model=Dict[str, Any])
def record_fill(
    fill: RecordFillRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Record an actual broker execution fill in the persistent journal and sync active portfolio holding."""
    _set_private_cache_headers(response)

    sym = fill.symbol.strip().upper()
    if not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.", headers=PRIVATE_CACHE_HEADERS)

    fill_dict = fill.model_dump() if hasattr(fill, "model_dump") else fill.dict()
    fill_dict["symbol"] = sym

    try:
        return journal_service.record_fill(context, fill_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error recording fill for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure recording trade fill.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.post("/exit", response_model=Dict[str, Any])
def record_exit(
    exit_req: RecordExitRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Record a partial scale-out or complete exit on an active open position."""
    _set_private_cache_headers(response)

    if not exit_req.tradeId and not exit_req.symbol:
        raise HTTPException(status_code=400, detail="Either tradeId or symbol must be specified.", headers=PRIVATE_CACHE_HEADERS)

    sym = exit_req.symbol.strip().upper() if exit_req.symbol else None
    if sym and not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.", headers=PRIVATE_CACHE_HEADERS)

    exit_dict = exit_req.model_dump() if hasattr(exit_req, "model_dump") else exit_req.dict()
    if sym:
        exit_dict["symbol"] = sym

    try:
        return journal_service.record_exit(context, exit_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error recording exit for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure recording trade exit.",
            headers=PRIVATE_CACHE_HEADERS,
        )


@router.post("/close", response_model=Dict[str, Any])
def record_close(
    close_req: RecordCloseRequest,
    response: Response = None,
    x_user_id: Optional[str] = Header(None),
    context: RequestContext = Depends(resolve_request_context),
):
    """Convenience endpoint to close 100% of an active open holding."""
    _set_private_cache_headers(response)

    if not close_req.tradeId and not close_req.symbol:
        raise HTTPException(status_code=400, detail="Either tradeId or symbol must be specified.", headers=PRIVATE_CACHE_HEADERS)

    sym = close_req.symbol.strip().upper() if close_req.symbol else None
    if sym and not SYMBOL_REGEX.match(sym):
        raise HTTPException(status_code=400, detail="Invalid symbol format.", headers=PRIVATE_CACHE_HEADERS)

    close_dict = close_req.model_dump() if hasattr(close_req, "model_dump") else close_req.dict()
    if sym:
        close_dict["symbol"] = sym

    try:
        return journal_service.record_close(context, close_dict)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe), headers=PRIVATE_CACHE_HEADERS)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve), headers=PRIVATE_CACHE_HEADERS)
    except Exception as e:
        logger.error(f"Database error closing position for context {context}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Persistent database failure closing position.",
            headers=PRIVATE_CACHE_HEADERS,
        )
