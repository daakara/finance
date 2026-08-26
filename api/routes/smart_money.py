"""FastAPI Router for Smart Money, Congressional Disclosures & Options Flow."""

from fastapi import APIRouter
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine

router = APIRouter()
smart_money_engine = SmartMoneyEngine()

@router.get("/overview")
def get_smart_money_overview():
    """Get market-wide congressional disclosures and unusual options flow overview."""
    return smart_money_engine.get_smart_money_overview()

@router.get("/congress")
def get_congress_trades(symbol: str = None):
    """Get Capitol Hill stock disclosures, optionally filtered by symbol."""
    return {"trades": smart_money_engine.get_congressional_trades(symbol)}

@router.get("/options-flow")
def get_options_flow(symbol: str = None):
    """Get institutional options sweeps and dark pool blocks, optionally filtered by symbol."""
    return {"flow": smart_money_engine.get_options_flow(symbol)}
