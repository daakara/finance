"""FastAPI Router for ETF Analytics, Risk Profile, and Dynamic Sector Decomposition (Phase P2).

Guarantees:
1. Canonical Delegation: Route handler contains ZERO quantitative business logic.
   All mathematical computations delegate strictly to ETFAnalyzer in analysis/etf.py.
2. Symbol Validation: Rejects invalid formats (400) and non-ETF instruments (400).
3. Deterministic Error States: Returns structured fail-closed responses.
4. Temporal Metadata: All payloads include ISO-8601 UTC timestamps and observation periods.
5. Cache Control: Public caching with stale-while-revalidate for edge performance.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from analysis.etf import ETFAnalyzer
from api.routes.analytics import KNOWN_ETFS
from data.fetchers import stock_fetcher

logger = logging.getLogger("api.routes.etf")
router = APIRouter()

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-_]{1,16}$")
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}

# Known corporate equities that must be rejected as non-ETF
KNOWN_STOCKS = {
    "AAPL", "NVDA", "MSFT", "TSLA", "PLTR", "NVO", "LLY", "CPRX", "POWI", "LNTH",
    "KO", "SBUX", "AMZN", "GOOGL", "AMD", "ARM", "SMCI", "COIN", "VRT", "ISRG",
    "KLAC", "CIEN", "ACLS", "TMDX", "MEDP", "ELF", "DUOL", "JPM", "V", "MA",
    "DIS", "COST", "WMT", "CRWD", "PANW", "MSTR", "MARA", "IONQ", "RKLB", "VRTX"
}


# =====================================================================
# Pydantic Result Models (P2I21, P2I22, P2I23)
# =====================================================================

class QualityState(BaseModel):
    state: str = Field(..., description="Data quality state: ESTABLISHED, PARTIAL, INSUFFICIENT_HISTORY, NOT_AVAILABLE")
    warnings: List[str] = Field(default_factory=list, description="Quality warnings or sample size caveats")


class DrawdownMetrics(BaseModel):
    maximum: Optional[float] = Field(None, description="Max drawdown as fraction, e.g. 0.339")
    maximum_pct: Optional[float] = Field(None, description="Max drawdown percentage, e.g. 33.9")
    current: Optional[float] = Field(None, description="Current drawdown from peak as fraction")
    current_pct: Optional[float] = Field(None, description="Current drawdown percentage")
    peak_date: Optional[str] = Field(None, description="Date of the peak preceding max drawdown (YYYY-MM-DD)")
    trough_date: Optional[str] = Field(None, description="Date of maximum drawdown trough (YYYY-MM-DD)")
    recovery_date: Optional[str] = Field(None, description="Date of full recovery to prior peak (YYYY-MM-DD)")
    recovery_days: Optional[int] = Field(None, description="Trading days from trough to recovery")
    recovery_state: str = Field("INSUFFICIENT_DATA", description="Recovery state: RECOVERED, UNRECOVERED, INSUFFICIENT_DATA")


class RiskAdjustedReturns(BaseModel):
    sharpe: Optional[float] = Field(None, description="Annualized Sharpe ratio (Rf=2.0% annual)")
    sortino: Optional[float] = Field(None, description="Annualized Sortino ratio (downside semi-deviation)")
    calmar: Optional[float] = Field(None, description="Calmar ratio: annualized return / max drawdown")


class VarDetail(BaseModel):
    confidence: float = Field(..., description="Confidence level, e.g. 0.95 or 0.99")
    method: str = Field("MODIFIED_CORNISH_FISHER", description="VaR estimation methodology")
    daily_var: float = Field(..., description="Daily VaR loss as return fraction")
    daily_var_pct: float = Field(..., description="Daily VaR loss as percentage")
    unit: str = Field("RETURN_FRACTION", description="Unit of measurement")
    sign_convention: str = Field("POSITIVE_LOSS", description="Positive number indicates expected loss magnitude")
    z_gaussian: float = Field(..., description="Standard normal quantile")
    z_cornish_fisher: float = Field(..., description="Cornish-Fisher adjusted quantile")
    skewness: float = Field(..., description="Sample skewness")
    excess_kurtosis: float = Field(..., description="Sample excess kurtosis")
    observations: int = Field(..., description="Observation count")


class ValueAtRiskMetrics(BaseModel):
    var_95: Optional[VarDetail] = Field(None, description="95% Modified Cornish-Fisher daily VaR")
    var_99: Optional[VarDetail] = Field(None, description="99% Modified Cornish-Fisher daily VaR")
    horizon: str = Field("1d", description="Risk horizon (1 trading day)")
    method: str = Field("MODIFIED_CORNISH_FISHER", description="Methodology")
    unit: str = Field("RETURN_FRACTION", description="Unit of measure")


class VolatilityMetrics(BaseModel):
    realized_annualized_pct: Optional[float] = Field(None, description="Realized annualized volatility percentage")
    regime: str = Field("UNKNOWN", description="Volatility regime: LOW (<12%), MODERATE (12-22%), HIGH (>22%)")
    history_200d: List[float] = Field(default_factory=list, description="Trailing 200-day rolling vol points for sparkline")


class SectorAllocationItem(BaseModel):
    sector: str = Field(..., description="Standardized institutional sector name")
    weightPct: float = Field(..., ge=0.0, description="Weight percentage in portfolio")
    raw_sector_key: Optional[str] = Field(None, description="Raw source sector identifier")


class EtfRiskProfileResponse(BaseModel):
    symbol: str = Field(..., description="Canonical ticker symbol")
    as_of: str = Field(..., description="UTC ISO-8601 calculation timestamp")
    history_start: Optional[str] = Field(None, description="Start date of price history analyzed")
    history_end: Optional[str] = Field(None, description="End date of price history analyzed")
    period: str = Field("1y", description="Analysis lookback period")
    observation_count: int = Field(0, description="Number of trading session observations")
    quality: QualityState = Field(..., description="Data quality assessment")
    drawdown: DrawdownMetrics = Field(..., description="Drawdown history and recovery metrics")
    risk_adjusted_returns: RiskAdjustedReturns = Field(..., description="Risk-adjusted return ratios")
    value_at_risk: ValueAtRiskMetrics = Field(..., description="Value at Risk metrics")
    volatility: VolatilityMetrics = Field(..., description="Volatility analysis and regime")

    # Flat convenience access fields (matching Section 14)
    max_drawdown_pct: Optional[float] = None
    max_drawdown_date: Optional[str] = None
    recovery_days: Optional[int] = None
    current_drawdown_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    var_95_daily_pct: Optional[float] = None
    var_99_daily_pct: Optional[float] = None
    annualized_volatility_pct: Optional[float] = None
    volatility_regime: str = "UNKNOWN"
    vol_history_200d: List[float] = Field(default_factory=list)
    sectors: List[SectorAllocationItem] = Field(default_factory=list)
    source_provenance: str = "ARX_ETF_ANALYZER_V1"


class EtfSectorDecompositionResponse(BaseModel):
    symbol: str
    as_of: str
    source: str
    conservation_policy: str
    total_weight_pct: float
    sectors: List[SectorAllocationItem]
    is_dynamic: bool


# =====================================================================
# Symbol Validation & Discrimination Helpers
# =====================================================================

def is_etf_instrument(clean_sym: str) -> bool:
    """
    Authoritative backend ETF discrimination.
    Returns True if symbol is deterministically an ETF; False if corporate equity or unknown.
    """
    if clean_sym in KNOWN_STOCKS:
        return False
    if clean_sym in KNOWN_ETFS:
        return True

    # Check yfinance ticker info/funds_data
    try:
        import yfinance as yf
        ticker = yf.Ticker(clean_sym)
        # If funds_data exists with holdings or sector weightings, it is an ETF/fund
        fd = getattr(ticker, "funds_data", None)
        if fd and getattr(fd, "sector_weightings", None):
            return True
        info = getattr(ticker, "fast_info", None) or getattr(ticker, "info", None) or {}
        quote_type = (info.get("quoteType") or "").upper()
        if quote_type in ("ETF", "MUTUALFUND"):
            return True
        if quote_type == "EQUITY":
            return False
    except Exception as e:
        logger.debug(f"Instrument discrimination check error for {clean_sym}: {e}")

    # If not recognized as ETF, fail-closed as False
    return False


# =====================================================================
# API Routes
# =====================================================================

@router.get("/profile/{symbol}", response_model=EtfRiskProfileResponse, tags=["ETF Analytics & Risk Profile"])
def get_etf_risk_profile(
    symbol: str,
    period: str = Query("1y", description="Analysis lookback period (e.g. 1mo, 3mo, 6mo, 1y, 2y, 5y, max)"),
    response: Response = None,
):
    """
    Retrieve comprehensive institutional risk profile and fund-native metrics for an ETF.

    Delegates all calculations to ETFAnalyzer in analysis/etf.py.
    Contains ZERO quantitative business formulas.
    """
    clean_sym = symbol.strip().upper().replace("-USD", "")

    if not SYMBOL_REGEX.match(clean_sym):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid symbol format: '{symbol}'"
        )

    if period not in VALID_PERIODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period: '{period}'. Supported periods: {sorted(list(VALID_PERIODS))}"
        )

    # Enforce non-ETF rejection (Criterion P2I27)
    if not is_etf_instrument(clean_sym):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Symbol '{clean_sym}' is not an ETF. Risk profile analytics are only available for ETF instruments."
        )

    try:
        data = ETFAnalyzer.get_etf_risk_profile(clean_sym, period)
        if response is not None:
            response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=3600, stale-while-revalidate=86400"
        return data
    except Exception as e:
        logger.error(f"Error calculating ETF risk profile for {clean_sym}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred generating ETF risk profile for {clean_sym}."
        )


@router.get("/sectors/{symbol}", response_model=EtfSectorDecompositionResponse, tags=["ETF Analytics & Risk Profile"])
def get_etf_sectors(
    symbol: str,
    response: Response = None,
):
    """
    Retrieve dynamic sector weight decomposition for an ETF.

    Enforces conservation policy: PARTIAL_WITH_DISCLOSED_UNCLASSIFIED_WEIGHT.
    Delegates strictly to ETFAnalyzer.
    """
    clean_sym = symbol.strip().upper().replace("-USD", "")

    if not SYMBOL_REGEX.match(clean_sym):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid symbol format: '{symbol}'"
        )

    if not is_etf_instrument(clean_sym):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Symbol '{clean_sym}' is not an ETF. Dynamic sector allocations are only available for ETF instruments."
        )

    try:
        data = ETFAnalyzer.get_etf_sector_allocations(clean_sym)
        if response is not None:
            response.headers["Cache-Control"] = "public, max-age=86400, s-maxage=86400, stale-while-revalidate=604800"
        return data
    except Exception as e:
        logger.error(f"Error fetching dynamic sector allocations for {clean_sym}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred retrieving sector allocations for {clean_sym}."
        )
