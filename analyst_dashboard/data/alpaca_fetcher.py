"""Alpaca Market Data API Fetcher (Equities & Free IEX Real-Time Tape).

Provides high-performance live quote snapshots, trades, and historical OHLCV bars
for ARX Terminal (Radar, Analysis, Trade Plan). Uses the 100% free IEX feed by default.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default credentials from environment or config
DEFAULT_ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", os.getenv("APCA_API_KEY_ID", os.getenv("ALPACA_API_KEY", "")))
DEFAULT_ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", "")))
DEFAULT_ALPACA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")


class AlpacaMarketFetcher:
    """High-performance fetcher for Alpaca Market Data v2 API with free IEX feed."""

    BASE_URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        api_secret_key: Optional[str] = None,
        feed: Optional[str] = None,
        timeout: int = 8,
    ):
        self.api_key_id = (api_key_id if api_key_id is not None else DEFAULT_ALPACA_KEY).strip()
        self.api_secret_key = (api_secret_key if api_secret_key is not None else DEFAULT_ALPACA_SECRET).strip()
        self.feed = (feed if feed is not None else DEFAULT_ALPACA_FEED).strip().lower()
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def is_configured(self) -> bool:
        """Returns True if API credentials are present."""
        return bool(self.api_key_id and self.api_secret_key)

    def _headers(self) -> Dict[str, str]:
        """Headers required by Alpaca Market Data v2."""
        return {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
            "Accept": "application/json",
        }

    def fetch_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch latest real-time quote and last trade for a single symbol (IEX feed).
        
        Returns standardized dict with price, bid, ask, sizes, and timestamp.
        """
        if not self.is_configured:
            return None

        clean_symbol = symbol.upper().strip()
        # Alpaca stocks endpoint does not use exchange suffixes like .US
        clean_symbol = clean_symbol.replace(".US", "")

        url = f"{self.BASE_URL}/{clean_symbol}/quotes/latest"
        params = {"feed": self.feed}

        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                quote_data = data.get("quote", {})
                bid = float(quote_data.get("bp", 0.0))
                ask = float(quote_data.get("ap", 0.0))
                bid_size = int(quote_data.get("bs", 0))
                ask_size = int(quote_data.get("as", 0))
                timestamp = quote_data.get("t", "")

                # Mid-price or best available quote
                price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (bid or ask)

                # Attempt to get latest trade for exact last executed price
                trade_price = None
                try:
                    trade_url = f"{self.BASE_URL}/{clean_symbol}/trades/latest"
                    t_resp = self.session.get(trade_url, headers=self._headers(), params=params, timeout=self.timeout)
                    if t_resp.status_code == 200:
                        t_data = t_resp.json().get("trade", {})
                        if t_data.get("p"):
                            trade_price = float(t_data["p"])
                except Exception as te:
                    logger.debug(f"Failed to fetch trade price for {clean_symbol}: {te}")

                effective_price = trade_price if (trade_price and trade_price > 0) else price

                return {
                    "symbol": clean_symbol,
                    "price": round(effective_price, 4) if effective_price else 0.0,
                    "bid": bid,
                    "ask": ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "timestamp": timestamp,
                    "feed": self.feed,
                }
            elif resp.status_code in (401, 403):
                logger.warning("Alpaca API authentication failed. Verify API Key and Secret.")
            else:
                logger.warning(f"Alpaca quote fetch failed for {clean_symbol} (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            logger.warning(f"Alpaca quote fetch error for {clean_symbol}: {e}")

        return None

    def fetch_snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch batch ticker snapshots for multiple symbols in a single round-trip.
        
        Essential for the ARX Radar screener universe.
        """
        if not self.is_configured or not symbols:
            return {}

        clean_symbols = [s.upper().strip().replace(".US", "") for s in symbols if s]
        url = f"{self.BASE_URL}/snapshots"
        params = {
            "symbols": ",".join(clean_symbols),
            "feed": self.feed,
        }

        results = {}
        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                for sym, snap in data.items():
                    latest_quote = snap.get("latestQuote", {})
                    latest_trade = snap.get("latestTrade", {})
                    daily_bar = snap.get("dailyBar", {})
                    prev_daily = snap.get("prevDailyBar", {})

                    price = latest_trade.get("p") or daily_bar.get("c") or latest_quote.get("ap") or 0.0
                    prev_close = prev_daily.get("c") or 0.0
                    change_pct = 0.0
                    if price and prev_close and prev_close > 0:
                        change_pct = round(((price - prev_close) / prev_close) * 100.0, 2)

                    results[sym] = {
                        "symbol": sym,
                        "price": float(price),
                        "change_pct": change_pct,
                        "volume": int(daily_bar.get("v", 0)),
                        "high": float(daily_bar.get("h", 0.0)),
                        "low": float(daily_bar.get("l", 0.0)),
                        "open": float(daily_bar.get("o", 0.0)),
                        "prev_close": float(prev_close),
                        "timestamp": latest_trade.get("t") or daily_bar.get("t") or "",
                        "feed": self.feed,
                    }
            else:
                logger.warning(f"Alpaca snapshots fetch failed (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            logger.warning(f"Alpaca snapshots fetch error: {e}")

        return results

    def fetch_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 365,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV bars as a pandas DataFrame.
        
        Columns returned: ['Open', 'High', 'Low', 'Close', 'Volume', 'TradeCount', 'VWAP']
        Indexed by DatetimeIndex (Date).
        """
        if not self.is_configured:
            return None

        clean_symbol = symbol.upper().strip().replace(".US", "")
        url = f"{self.BASE_URL}/{clean_symbol}/bars"

        # Default start to limit days ago if not specified
        if not start:
            start_dt = datetime.utcnow() - timedelta(days=limit * 2)
            start = start_dt.strftime("%Y-%m-%d")

        params = {
            "timeframe": timeframe,
            "feed": self.feed,
            "limit": min(limit, 1000),
            "start": start,
            "sort": "asc",
        }
        if end:
            params["end"] = end

        try:
            resp = self.session.get(url, headers=self._headers(), params=params, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                bars = data.get("bars", [])
                if isinstance(bars, list) and len(bars) > 0:
                    df = pd.DataFrame(bars)
                    df["Date"] = pd.to_datetime(df["t"], errors="coerce")
                    df = df.dropna(subset=["Date"])
                    df = df.set_index("Date")
                    df = df.rename(
                        columns={
                            "o": "Open",
                            "h": "High",
                            "l": "Low",
                            "c": "Close",
                            "v": "Volume",
                            "n": "TradeCount",
                            "vw": "VWAP",
                        }
                    )
                    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "TradeCount", "VWAP"] if c in df.columns]
                    return df[cols].tail(limit)
            else:
                logger.warning(f"Alpaca historical bars failed for {clean_symbol} (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            logger.warning(f"Alpaca historical bars error for {clean_symbol}: {e}")

        return None
