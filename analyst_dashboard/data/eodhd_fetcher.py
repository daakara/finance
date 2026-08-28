"""High-Performance Market Data Fetcher & Failover Pipeline (EODHD & Financial APIs)."""

import os
import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_EODHD_KEY = os.getenv("EODHD_API_KEY", "")

class EODHDMarketFetcher:
    """Institutional real-time quote, intraday tape, and EOD historical candlestick fetcher."""

    BASE_URL = "https://eodhd.com/api"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_EODHD_KEY

    def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch live ticker snapshot including bid/ask, VWAP proxy, and 24h change."""
        if not self.api_key:
            return None

        upper = symbol.upper().strip()
        ticker_code = f"{upper}.US" if not upper.endswith(".US") and "-" not in upper else upper
        if "-" in upper:  # Crypto (e.g. BTC-USD -> BTC-USD.CC)
            ticker_code = f"{upper}.CC"

        url = f"{self.BASE_URL}/real-time/{ticker_code}"
        params = {"api_token": self.api_key, "fmt": "json"}

        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and "close" in data and data["close"] > 0:
                    return data
        except Exception as e:
            logger.warning(f"EODHD realtime fetch failed for {symbol}: {e}")

        return None

    def fetch_historical_candles(self, symbol: str, period_days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch EOD historical OHLCV candles as a clean Pandas DataFrame."""
        if not self.api_key:
            return None

        upper = symbol.upper().strip()
        ticker_code = f"{upper}.US" if not upper.endswith(".US") and "-" not in upper else upper
        if "-" in upper:
            ticker_code = f"{upper}.CC"

        url = f"{self.BASE_URL}/eod/{ticker_code}"
        params = {
            "api_token": self.api_key,
            "fmt": "json",
            "period": "d",
            "order": "a",
        }

        try:
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                rows = resp.json()
                if isinstance(rows, list) and len(rows) > 0:
                    df = pd.DataFrame(rows)
                    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["Date"])
                    df = df.set_index("Date")
                    df = df.rename(columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    })
                    return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            logger.warning(f"EODHD historical fetch failed for {symbol}: {e}")

        return None
