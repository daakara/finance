"""SEC EDGAR Regulatory Filing & Corporate Financial Disclosure Fetcher (Free Public API)."""

import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# SEC EDGAR requires a User-Agent in format: User-Agent: Sample Company Name AdminContact@<sample company domain>.com
SEC_USER_AGENT = "FinanceTerminalResearchApp admin@financeterminal.internal"

class SecEdgarFetcher:
    """Fetches official 10-K, 10-Q, 8-K, Form 4 insider transactions, and 13F institutional holdings."""

    BASE_URL = "https://data.sec.gov"
    CIK_MAP = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "NVDA": "0001045810",
        "NVO": "0000353278",
        "PLTR": "0001321655",
        "TSLA": "0001318605",
        "LLY": "0000059478",
        "CRWD": "0001535527",
        "AMD": "0000002488",
        "AVGO": "0001730168",
        "VRT": "0001674101",
        "TSM": "0001046179",
        "GE": "0000040545",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SEC_USER_AGENT})

    def get_company_submissions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time company submissions including Form 4 (insiders) and 8-K (material events)."""
        upper = symbol.upper().strip()
        cik = self.CIK_MAP.get(upper)
        if not cik:
            return None

        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        try:
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"SEC EDGAR submissions fetch failed for {symbol}: {e}")
        return None

    def get_recent_filings(self, symbol: str, form_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get formatted list of recent regulatory filings."""
        data = self.get_company_submissions(symbol)
        if not data or "filings" not in data or "recent" not in data["filings"]:
            return []

        recent = data["filings"]["recent"]
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])

        allowed_forms = set(form_types) if form_types else {"10-K", "10-Q", "8-K", "4", "13F-HR"}
        results = []

        for i in range(min(len(forms), 25)):
            form = forms[i]
            if form in allowed_forms:
                clean_acc = accessions[i].replace("-", "")
                cik_int = int(self.CIK_MAP.get(symbol.upper(), "0"))
                results.append({
                    "form": form,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                    "description": descriptions[i] if i < len(descriptions) else form,
                    "accession_number": accessions[i],
                    "sec_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{accessions[i]}.txt",
                })

        return results
