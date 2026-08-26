"""FastAPI Router for Smart Money, Congressional Disclosures, SEC Filings & FINRA Dark Pool."""

from fastapi import APIRouter
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher
from analyst_dashboard.data.capitol_trades_fetcher import CapitolTradesFetcher

router = APIRouter()
smart_money_engine = SmartMoneyEngine()
sec_fetcher = SecEdgarFetcher()
finra_fetcher = FinraTransparencyFetcher()
capitol_fetcher = CapitolTradesFetcher()

@router.get("/overview")
def get_smart_money_overview():
    """Get market-wide congressional disclosures and unusual options flow overview."""
    overview = smart_money_engine.get_smart_money_overview()
    overview["regulatory_sources"] = {
        "sec_edgar": "Official SEC Form 4 & 10-K Public API",
        "finra_ats": "FINRA ATS Dark Pool Transparency Aggregation",
        "capitol_trades": "US House & Senate STOCK Act Financial Disclosures",
    }
    return overview

@router.get("/congress")
def get_congress_trades(symbol: str = None):
    """Get Capitol Hill stock disclosures, optionally filtered by symbol."""
    return {
        "trades": smart_money_engine.get_congressional_trades(symbol),
        "source_meta": capitol_fetcher.get_filing_source_info(),
    }

@router.get("/options-flow")
def get_options_flow(symbol: str = None):
    """Get institutional options sweeps and dark pool blocks, optionally filtered by symbol."""
    return {"flow": smart_money_engine.get_options_flow(symbol)}

@router.get("/sec-filings/{symbol}")
def get_sec_filings(symbol: str):
    """Fetch official SEC EDGAR 10-K, 10-Q, 8-K, and Form 4 insider filings."""
    return {
        "symbol": symbol.upper(),
        "filings": sec_fetcher.get_recent_filings(symbol),
    }

@router.get("/finra-darkpool/{symbol}")
def get_finra_darkpool(symbol: str):
    """Fetch official FINRA Alternative Trading System (ATS) dark pool shares & short volumes."""
    return {
        "symbol": symbol.upper(),
        "metrics": finra_fetcher.get_ats_metrics(symbol),
    }
