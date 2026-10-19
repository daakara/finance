"""
api/routes/fundamentals.py

Canonical Server Sourced Fundamentals & Financial Evidence API Routes.
Exposes read-only SEC EDGAR-sourced company fundamentals, derived calculations,
and evidence-aware research assessments.
Strictly zero side-effects: GET requests do not harvest SEC filings, write to disk,
or mutate database state.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from analyst_dashboard.research.sourced_fundamentals.service import (
    get_sourced_fundamentals_service,
    SourcedFundamentalsService,
)

logger = logging.getLogger("api.routes.fundamentals")
router = APIRouter()


class CompanyCatalogItem(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full corporate legal name")
    cik: Optional[str] = Field(None, description="SEC Central Index Key")
    primary_exchange: Optional[str] = Field(None, description="Primary listing exchange")
    sector_industry: Optional[str] = Field(None, description="Sector and industry classification")
    latest_reported_period: Optional[str] = Field(None, description="Latest fiscal period end (YYYY-MM-DD)")
    latest_filing_date: Optional[str] = Field(None, description="Latest Form 10-K / 10-Q filing date (YYYY-MM-DD)")
    disposition: Optional[str] = Field(None, description="Research candidate disposition")
    valuation_status: str = Field(..., description="Status of valuation model ('AVAILABLE', 'SUPPRESSED_UNAVAILABLE')")
    total_financial_debt_evidenced: bool = Field(..., description="Whether total financial debt is fully evidenced in primary XBRL")


class FundamentalsCatalogResponse(BaseModel):
    metadata: Dict[str, Any] = Field(..., description="Snapshot ledger metadata and cutoff attestation")
    companies: List[CompanyCatalogItem] = Field(..., description="List of cataloged companies")
    count: int = Field(..., description="Total count of available companies")


@router.get(
    "/",
    response_model=FundamentalsCatalogResponse,
    summary="List available sourced company fundamentals catalog",
)
def list_fundamentals_catalog(response: Response = None) -> FundamentalsCatalogResponse:
    """
    Returns the catalog of companies with available sourced fundamental evidence.
    Pure read-only endpoint with zero mutation side effects.
    """
    service = get_sourced_fundamentals_service()
    metadata = service.get_snapshot_metadata()
    companies = service.list_available_companies()

    if response is not None:
        # Cacheable for 5 minutes since snapshot is versioned/immutable in-process
        response.headers["Cache-Control"] = "public, max-age=300"

    return FundamentalsCatalogResponse(
        metadata=metadata,
        companies=companies,
        count=len(companies),
    )


@router.get(
    "/{symbol}",
    summary="Get detailed company sourced financial evidence snapshot",
)
def get_company_fundamentals(
    symbol: str,
    response: Response = None,
) -> Dict[str, Any]:
    """
    Retrieves full structured financial evidence snapshot for a company by symbol.
    Fails closed with 404 Not Found if the company is not cataloged in the research snapshot.
    Preserves null values without synthetic zero substitution.
    """
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        raise HTTPException(status_code=400, detail="Symbol parameter cannot be empty.")

    service = get_sourced_fundamentals_service()
    snapshot = service.get_company_snapshot(clean_sym)

    if snapshot is None:
        available = [c["symbol"] for c in service.list_available_companies()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Sourced financial evidence not found for symbol '{clean_sym}'. "
                f"Currently available cataloged symbols: {', '.join(available)}."
            ),
        )

    if response is not None:
        response.headers["Cache-Control"] = "public, max-age=300"

    return snapshot


@router.get(
    "/system/status",
    summary="Get sourced fundamentals service status and data source provenance",
)
def get_fundamentals_service_status() -> Dict[str, Any]:
    """
    Returns diagnostic health and metadata regarding the sourced fundamentals data layer.
    """
    service = get_sourced_fundamentals_service()
    meta = service.get_snapshot_metadata()
    catalog = service.list_available_companies()
    return {
        "status": "online",
        "subsystem": "ARX_SOURCED_FUNDAMENTALS_EVIDENCE_ENGINE",
        "read_only": True,
        "companies_cataloged_count": len(catalog),
        "cataloged_symbols": [c["symbol"] for c in catalog],
        "snapshot_metadata": meta,
    }
