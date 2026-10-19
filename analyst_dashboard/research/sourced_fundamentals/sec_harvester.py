"""
analyst_dashboard/research/sourced_fundamentals/sec_harvester.py

Authoritative SEC EDGAR Harvester & Market Data Acquirer.
Adheres strictly to official SEC fair access guidance:
- Declares genuine operator identity: "ARXTerminal Research Akara.david26@gmail.com"
- Enforces strict conservative rate ceiling (<= 4 req/sec) with backoff.
- Implements deterministic local caching in data/research/sourced_fundamentals/raw/
  to guarantee offline replayability and avoid redundant network load.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "ARXTerminal Research Akara.david26@gmail.com"
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data", "research", "sourced_fundamentals", "raw"
)


class SecEdgarHarvester:
    """SEC EDGAR client for submissions and XBRL company facts."""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, min_request_interval_sec: float = 0.25):
        self.cache_dir = cache_dir
        self.min_request_interval_sec = min_request_interval_sec
        self.last_request_time = 0.0
        self.user_agent = SEC_USER_AGENT
        os.makedirs(self.cache_dir, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)
        self.last_request_time = time.time()

    def _fetch_url(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
        }
        for attempt in range(max_retries):
            self._rate_limit()
            req = urllib.request.Request(url, headers=headers)
            try:
                import gzip
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()
                    # Decompress if gzipped
                    if resp.info().get("Content-Encoding") == "gzip" or (len(raw) > 2 and raw[:2] == b'\x1f\x8b'):
                        raw = gzip.decompress(raw)
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in [429, 503]:
                    backoff = (attempt + 1) * 2.0
                    logger.warning(f"SEC HTTP {e.code} for {url}. Backing off {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"SEC HTTP Error {e.code} for {url}: {e.reason}")
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Network error fetching {url}: {e}")
                    raise
                time.sleep(1.0)
        raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")

    def get_company_facts(self, cik: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch official structured XBRL company facts:
        https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json
        """
        cik_clean = cik.zfill(10)
        cache_file = os.path.join(self.cache_dir, f"CIK{cik_clean}_companyfacts.json")

        if not force_refresh and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_clean}.json"
        data = self._fetch_url(url)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    def get_company_submissions(self, cik: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch entity submissions metadata:
        https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json
        """
        cik_clean = cik.zfill(10)
        cache_file = os.path.join(self.cache_dir, f"CIK{cik_clean}_submissions.json")

        if not force_refresh and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
        data = self._fetch_url(url)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data


def fetch_market_price_yfinance(symbol: str) -> Dict[str, Any]:
    """
    Fetch authoritative price observation with dated timestamp.
    Separates market price observation from reported financial statement facts.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        fast = ticker.fast_info
        
        price = (
            fast.get("last_price")
            or fast.get("previous_close")
            or info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        shares = (
            fast.get("shares")
            or info.get("sharesOutstanding")
        )
        
        hist = ticker.history(period="5d")
        last_date = str(hist.index[-1].date()) if not hist.empty else datetime.utcnow().strftime("%Y-%m-%d")
        
        return {
            "symbol": symbol,
            "market_price": round(float(price), 2) if price else None,
            "price_date": last_date,
            "shares_outstanding_market": int(shares) if shares else None,
            "provider": "yfinance",
            "observation_time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.warning(f"Failed to fetch market price for {symbol}: {e}")
        return {
            "symbol": symbol,
            "market_price": None,
            "price_date": None,
            "shares_outstanding_market": None,
            "provider": "yfinance",
            "observation_time": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }
