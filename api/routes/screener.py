"""FastAPI Router for Hidden Gems Screener with Peter Lynch, Joel Greenblatt & Disruptive Innovation Models."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from analyst_dashboard.analyzers.gem_screener import HiddenGemsScreener

router = APIRouter()
screener = HiddenGemsScreener()

DEFAULT_CANDIDATES = [
    "PLTR", "CRWD", "ENPH", "NVDA", "SMH", "BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT"
]


class ScreenerRequest(BaseModel):
    tickers: Optional[List[str]] = None


@router.post("/run")
def run_screener(request: ScreenerRequest = None):
    """Run the Hidden Gems Discovery Screener against Peter Lynch GARP and Greenblatt Magic Formula criteria."""
    tickers = (request.tickers if request and request.tickers else DEFAULT_CANDIDATES)
    results = screener.evaluate_candidates(tickers)

    return {
        "total_candidates": len(tickers),
        "gems_found": len(results),
        "results": results,
    }

