import re
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from analyst_dashboard.analyzers.smart_money import SmartMoneyEngine
from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
from analyst_dashboard.data.finra_fetcher import FinraTransparencyFetcher
from analyst_dashboard.data.capitol_trades_fetcher import CapitolTradesFetcher

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")

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
def get_smart_money_overview(response: Response = None):
    """Get market-wide congressional disclosures and unusual options flow overview."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=300, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=300, stale-while-revalidate=86400, stale-if-error=86400"
    overview = smart_money_engine.get_smart_money_overview()
    overview["regulatory_sources"] = {
        "sec_edgar": "Official SEC Form 4 & 10-K Public API",
        "finra_ats": "FINRA ATS Dark Pool Transparency Aggregation",
        "capitol_trades": "US House & Senate STOCK Act Financial Disclosures",
    }
    return overview


@router.get("/congress")
def get_congress_trades(symbol: Optional[str] = None, response: Response = None):
    """Get Capitol Hill stock disclosures with explicit curated provenance."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=300, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=300, stale-while-revalidate=86400, stale-if-error=86400"
    valid_sym = _validate_symbol(symbol)
    trades = smart_money_engine.get_congressional_trades(valid_sym)
    return {
        "symbol": valid_sym,
        "status": "CURATED",
        "dataset_date": "2026-08-28",
        "trades": trades,
        "source_meta": capitol_fetcher.get_filing_source_info(),
        "disclosure": "Curated historical STOCK Act research dataset (August 2026). Not a real-time live trading feed.",
    }


@router.get("/options-flow")
def get_options_flow(symbol: Optional[str] = None, response: Response = None):
    """Get institutional options sweeps with verified provider provenance."""
    import os
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=120, stale-while-revalidate=86400, stale-if-error=86400"
    valid_sym = _validate_symbol(symbol)
    has_live_provider = bool(os.getenv("POLYGON_API_KEY") or os.getenv("OPRA_API_KEY"))
    flow = smart_money_engine.get_options_flow(valid_sym, include_curated=True if not valid_sym else False)
    return {
        "symbol": valid_sym,
        "available": has_live_provider,
        "flow": flow,
        "message": "Live OPRA options feed active" if has_live_provider else "Live options tape unavailable without Polygon.io / Trade Alert API Key",
    }


@router.get("/sec-filings/{symbol}")
def get_sec_filings(symbol: str, response: Response = None):
    """Fetch official SEC EDGAR 10-K, 10-Q, 8-K, and Form 4 insider filings with coverage disclosure."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=120, s-maxage=600, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=600, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=600, stale-while-revalidate=86400, stale-if-error=86400"
    valid_sym = _validate_symbol(symbol)
    cik = sec_fetcher.resolve_cik(valid_sym)
    filings = sec_fetcher.get_recent_filings(valid_sym) if cik else []
    return {
        "symbol": valid_sym,
        "available": cik is not None,
        "cik": cik,
        "coverage": "Supported SEC Issuer" if cik else "Unsupported Issuer / Foreign FPI (Exempt from Form 4)",
        "filings": filings,
        "message": "Official SEC EDGAR filings retrieved" if cik else f"SEC EDGAR Form 4 filings unavailable for {valid_sym}",
    }


@router.get("/finra-darkpool/{symbol}")
def get_finra_darkpool(symbol: str, response: Response = None):
    """Fetch official FINRA Alternative Trading System (ATS) dark pool shares & short volumes."""
    if response is not None and hasattr(response, "headers"):
        response.headers["Cache-Control"] = "public, max-age=120, s-maxage=600, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["CDN-Cache-Control"] = "max-age=600, stale-while-revalidate=86400, stale-if-error=86400"
        response.headers["Cloudflare-CDN-Cache-Control"] = "max-age=600, stale-while-revalidate=86400, stale-if-error=86400"
    valid_sym = _validate_symbol(symbol)
    metrics = finra_fetcher.get_ats_metrics(valid_sym)
    return {
        "symbol": valid_sym,
        "available": metrics is not None,
        "metrics": metrics,
        "message": None if metrics is not None else "FINRA ATS off-exchange transparency data is unavailable for this ticker.",
    }
