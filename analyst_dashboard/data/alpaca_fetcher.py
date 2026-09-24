"""
Alpaca Market Data v2 Provider Integration (IEX Real-Time Feed).

Provides ultra-low-latency real-time quotes, latest trades, and batch snapshots
for US equities via the Alpaca Market Data v2 API using the free IEX feed.

Governing Principles:
1. STRICT LIVE SEPARATION: Live quote and trade data is used SOLELY for real-time
   market state, execution readiness, and position sizing. It NEVER mutates or
   contaminates completed historical daily OHLCV series.
2. FAIL-CLOSED ARCHITECTURE: Missing credentials, network timeouts, or rate limits
   fail closed and return None, allowing graceful fallback to secondary providers.
3. AUTHENTIC TIMESTAMPS: Quote timestamps derive strictly from the provider's
   exchange observation timestamp, never from local receipt time.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import requests

from config import Config

logger = logging.getLogger("arx.market_data.alpaca")


class AlpacaMarketFetcher:
    """Ultra-low-latency market data fetcher using Alpaca Market Data v2 (IEX tape)."""

    BASE_URL = "https://data.alpaca.markets/v2/stocks"

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        api_secret_key: Optional[str] = None,
        feed: Optional[str] = None,
        timeout: float = 3.0,
    ):
        self.api_key_id = (api_key_id or getattr(Config, "ALPACA_API_KEY_ID", "") or os.getenv("ALPACA_API_KEY_ID", "")).strip()
        self.api_secret_key = (api_secret_key or getattr(Config, "ALPACA_API_SECRET_KEY", "") or os.getenv("ALPACA_API_SECRET_KEY", "")).strip()
        self.feed = (feed or getattr(Config, "ALPACA_DATA_FEED", "iex") or os.getenv("ALPACA_DATA_FEED", "iex")).strip().lower()
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

        clean_symbol = symbol.upper().strip().replace(".US", "").replace("-USD", "")

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
                raw_ts = quote_data.get("t", "")

                # Mid-price or best available quote
                price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (bid or ask)

                # Attempt to get latest trade for exact last executed price
                trade_price = None
                trade_ts = None
                try:
                    trade_url = f"{self.BASE_URL}/{clean_symbol}/trades/latest"
                    t_resp = self.session.get(trade_url, headers=self._headers(), params=params, timeout=self.timeout)
                    if t_resp.status_code == 200:
                        t_data = t_resp.json().get("trade", {})
                        if t_data.get("p"):
                            trade_price = float(t_data["p"])
                            trade_ts = t_data.get("t")
                except Exception as te:
                    logger.debug(f"Failed to fetch trade price for {clean_symbol}: {te}")

                effective_price = trade_price if (trade_price and trade_price > 0) else price
                effective_ts = trade_ts or raw_ts

                # Parse timestamp to epoch ms
                observed_at_ms = None
                if effective_ts:
                    try:
                        dt = datetime.fromisoformat(effective_ts.replace("Z", "+00:00"))
                        observed_at_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        pass

                return {
                    "symbol": clean_symbol,
                    "price": round(effective_price, 4) if effective_price else None,
                    "bid": bid if bid > 0 else None,
                    "ask": ask if ask > 0 else None,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "timestamp": effective_ts,
                    "observed_at_ms": observed_at_ms,
                    "source": "ALPACA_IEX",
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
        """Fetch batch ticker snapshots for multiple symbols in a single round-trip."""
        if not self.is_configured or not symbols:
            return {}

        clean_symbols = [s.upper().strip().replace(".US", "").replace("-USD", "") for s in symbols if s]
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
                        "price": float(price) if price else None,
                        "change_pct": change_pct,
                        "volume": int(daily_bar.get("v", 0)),
                        "high": float(daily_bar.get("h", 0.0)),
                        "low": float(daily_bar.get("l", 0.0)),
                        "open": float(daily_bar.get("o", 0.0)),
                        "prev_close": float(prev_close),
                        "timestamp": latest_trade.get("t") or daily_bar.get("t") or "",
                        "source": "ALPACA_IEX",
                        "feed": self.feed,
                    }
            else:
                logger.warning(f"Alpaca snapshots fetch failed (HTTP {resp.status_code}): {resp.text}")
        except Exception as e:
            logger.warning(f"Alpaca snapshots fetch error: {e}")

        return results
