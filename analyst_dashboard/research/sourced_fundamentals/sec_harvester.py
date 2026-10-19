"""
analyst_dashboard/research/sourced_fundamentals/sec_harvester.py

Authoritative SEC EDGAR Harvester & Market Data Acquirer.
Adheres strictly to official SEC fair access guidance:
- Declares genuine operator identity: "ARXTerminal Research Akara.david26@gmail.com"
- Enforces strict conservative rate ceiling (<= 4 req/sec) with backoff.
- Implements deterministic local caching in data/research/sourced_fundamentals/raw/
  to guarantee offline replayability and avoid redundant network load.
- Tracks provenance via raw/manifest.json containing URLs, timestamps, SHA-256 hashes,
  and research cutoff dates.
"""

import os
import json
import time
import hashlib
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "ARXTerminal Research Akara.david26@gmail.com"
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "data", "research", "sourced_fundamentals", "raw"
)
RESEARCH_CUTOFF_DATE = "2026-10-16"


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SecEdgarHarvester:
    """SEC EDGAR client for submissions and XBRL company facts."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        min_request_interval_sec: float = 0.25,
        offline_mode: bool = False,
    ):
        self.cache_dir = cache_dir
        self.min_request_interval_sec = min_request_interval_sec
        self.offline_mode = offline_mode
        self.last_request_time = 0.0
        self.user_agent = SEC_USER_AGENT
        os.makedirs(self.cache_dir, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)
        self.last_request_time = time.time()

    def _fetch_url(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        if self.offline_mode:
            raise RuntimeError(
                f"Cannot make network request to {url} while offline_mode=True. "
                "All required primary sources must be present in local cache."
            )
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

        if self.offline_mode:
            raise FileNotFoundError(
                f"Offline replay mode enabled, but facts file {cache_file} was not found."
            )

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

        if self.offline_mode:
            raise FileNotFoundError(
                f"Offline replay mode enabled, but submissions file {cache_file} was not found."
            )

        url = f"https://data.sec.gov/submissions/CIK{cik_clean}.json"
        data = self._fetch_url(url)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data

    def build_manifest(self, ciks: List[str]) -> Dict[str, Any]:
        """
        Builds manifest of all raw SEC files and market data in the cache dir
        with source URLs, file sizes, SHA-256 hashes, and research cutoff date.
        """
        manifest: Dict[str, Any] = {
            "title": "ARX Terminal Sourced Fundamentals Raw Data Manifest",
            "research_cutoff_date": RESEARCH_CUTOFF_DATE,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "operator_identity": self.user_agent,
            "offline_replayable": True,
            "files": {}
        }

        for cik in ciks:
            cik_clean = cik.zfill(10)
            # 1. companyfacts
            cf_name = f"CIK{cik_clean}_companyfacts.json"
            cf_path = os.path.join(self.cache_dir, cf_name)
            if os.path.exists(cf_path):
                manifest["files"][cf_name] = {
                    "doc_type": "companyfacts",
                    "cik": cik_clean,
                    "source_url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_clean}.json",
                    "sha256": compute_file_sha256(cf_path),
                    "size_bytes": os.path.getsize(cf_path),
                    "retrieval_timestamp": datetime.fromtimestamp(os.path.getmtime(cf_path)).isoformat() + "Z",
                }

            # 2. submissions
            sub_name = f"CIK{cik_clean}_submissions.json"
            sub_path = os.path.join(self.cache_dir, sub_name)
            if os.path.exists(sub_path):
                manifest["files"][sub_name] = {
                    "doc_type": "submissions",
                    "cik": cik_clean,
                    "source_url": f"https://data.sec.gov/submissions/CIK{cik_clean}.json",
                    "sha256": compute_file_sha256(sub_path),
                    "size_bytes": os.path.getsize(sub_path),
                    "retrieval_timestamp": datetime.fromtimestamp(os.path.getmtime(sub_path)).isoformat() + "Z",
                }

        # 3. Market prices
        mp_name = "market_prices.json"
        mp_path = os.path.join(self.cache_dir, mp_name)
        if os.path.exists(mp_path):
            manifest["files"][mp_name] = {
                "doc_type": "market_prices",
                "source": "Yahoo Finance (retained authoritative closing observations)",
                "observation_date": RESEARCH_CUTOFF_DATE,
                "sha256": compute_file_sha256(mp_path),
                "size_bytes": os.path.getsize(mp_path),
                "retrieval_timestamp": datetime.fromtimestamp(os.path.getmtime(mp_path)).isoformat() + "Z",
            }

        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


def fetch_market_price_yfinance(
    symbol: str,
    offline_mode: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Fetch authoritative price observation with dated timestamp.
    In offline_mode (or if market_prices.json exists), loads from retained file.
    Explicitly couples market price observation with its exact trading date.
    """
    mp_file = os.path.join(cache_dir, "market_prices.json")
    if os.path.exists(mp_file):
        try:
            with open(mp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if symbol in data:
                return data[symbol]
        except Exception as e:
            logger.warning(f"Failed to read market_prices.json for {symbol}: {e}")

    if offline_mode:
        return {
            "symbol": symbol,
            "market_price": None,
            "price_date": None,
            "shares_outstanding_market": None,
            "provider": "offline_cache",
            "observation_time": datetime.utcnow().isoformat() + "Z",
            "error": f"Symbol {symbol} not found in market_prices.json and offline_mode=True",
        }

    # Online fallback if explicitly enabled
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty:
            last_date = str(hist.index[-1].date())
            price = round(float(hist["Close"].iloc[-1]), 2)
        else:
            fast = ticker.fast_info
            price = round(float(fast.get("last_price") or fast.get("previous_close")), 2)
            last_date = RESEARCH_CUTOFF_DATE

        shares = ticker.fast_info.get("shares") or (ticker.info or {}).get("sharesOutstanding")

        return {
            "symbol": symbol,
            "market_price": price,
            "price_date": last_date,
            "shares_outstanding_market": int(shares) if shares else None,
            "provider": "yfinance",
            "observation_time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.warning(f"Failed to fetch market price online for {symbol}: {e}")
        return {
            "symbol": symbol,
            "market_price": None,
            "price_date": None,
            "shares_outstanding_market": None,
            "provider": "yfinance",
            "observation_time": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }
