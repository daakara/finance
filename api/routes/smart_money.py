import re
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher
from analyst_dashboard.data.capitol_trades_fetcher import CapitolTradesFetcher

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-]{1,12}$")

router = APIRouter()
smart_money_engine = SmartMoneyEngine()
sec_fetcher = SecEdgarFetcher()
finra_fetcher = FinraTransparencyFetcher()
capitol_fetcher = CapitolTradesFetcher()


def _validate_symbol(sym: Optional[str]) -> Optional[str]:
    if not sym:
        return None
    upper = sym.upper().strip()
    if not SYMBOL_REGEX.match(upper):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ticker symbol format '{sym}'. Tickers must be 1-12 alphanumeric characters.",
        )
    return upper


@router.get("/overview")
def get_smart_money_overview(response: Optional[Response] = None):
    """Get market-wide congressional disclosures and unusual options flow overview."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=180, stale-while-revalidate=600"
    overview = smart_money_engine.get_smart_money_overview()
    overview["regulatory_sources"] = {
        "sec_edgar": "Official SEC Form 4 & 10-K Public API",
        "finra_ats": "FINRA ATS Dark Pool Transparency Aggregation",
        "capitol_trades": "US House & Senate STOCK Act Financial Disclosures",
    }
    return overview


@router.get("/congress")
def get_congress_trades(symbol: Optional[str] = None):
    """Get Capitol Hill stock disclosures, optionally filtered by symbol."""
    valid_sym = _validate_symbol(symbol)
    return {
        "trades": smart_money_engine.get_congressional_trades(valid_sym),
        "source_meta": capitol_fetcher.get_filing_source_info(),
    }


@router.get("/options-flow")
def get_options_flow(symbol: Optional[str] = None):
    """Get institutional options sweeps and dark pool blocks, optionally filtered by symbol."""
    valid_sym = _validate_symbol(symbol)
    return {"flow": smart_money_engine.get_options_flow(valid_sym)}


@router.get("/sec-filings/{symbol}")
def get_sec_filings(symbol: str):
    """Fetch official SEC EDGAR 10-K, 10-Q, 8-K, and Form 4 insider filings."""
    valid_sym = _validate_symbol(symbol)
    return {
        "symbol": valid_sym,
        "filings": sec_fetcher.get_recent_filings(valid_sym),
    }


@router.get("/finra-darkpool/{symbol}")
def get_finra_darkpool(symbol: str):
    """Fetch official FINRA Alternative Trading System (ATS) dark pool shares & short volumes."""
    valid_sym = _validate_symbol(symbol)
    return {
        "symbol": valid_sym,
        "metrics": finra_fetcher.get_ats_metrics(valid_sym),
    }
