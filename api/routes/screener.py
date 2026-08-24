"""FastAPI Router for Hidden Gems Multi-Factor Screening."""

from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
from pydantic import BaseModel

from analyst_dashboard.analyzers.gem_screener import HiddenGemScreener, GemCriteria

router = APIRouter()


class ScreenRequest(BaseModel):
    tickers: List[str]
    min_market_cap: Optional[float] = 100e6
    max_market_cap: Optional[float] = 5e9
    min_revenue_growth: Optional[float] = 0.30
    min_gross_margin: Optional[float] = 0.30


@router.post("/run")
def run_screening(payload: ScreenRequest = Body(...)):
    """Run multi-factor Hidden Gem screening across candidate tickers."""
    try:
        criteria = GemCriteria(
            min_market_cap=payload.min_market_cap,
            max_market_cap=payload.max_market_cap,
            min_revenue_growth=payload.min_revenue_growth,
            min_gross_margin=payload.min_gross_margin,
        )
        screener = HiddenGemScreener(criteria=criteria)
        results = screener.screen_universe(payload.tickers)

        formatted_results = []
        for gem in results:
            formatted_results.append({
                "ticker": gem.ticker,
                "composite_score": gem.composite_score,
                "risk_rating": gem.risk_rating,
                "investment_thesis": gem.investment_thesis,
                "primary_catalyst": gem.primary_catalyst,
            })

        return {
            "total_candidates": len(payload.tickers),
            "gems_found": len(formatted_results),
            "results": formatted_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

