"""Smart Money, Congressional Trades & Institutional Options Flow Engine."""

from typing import List, Dict, Any
from datetime import datetime, timedelta

CONGRESSIONAL_TRADES: List[Dict[str, Any]] = [
    {
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-08-14",
        "transaction_date": "2026-07-28",
        "strike_price": "$180 Calls",
        "days_to_filing": 17,
        "performance_since_pct": 14.8,
        "sentiment": "Strong Bullish",
    },
    {
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "NVO",
        "asset_name": "Novo Nordisk A/S",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-08-18",
        "transaction_date": "2026-08-02",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 6.4,
        "sentiment": "Strong Bullish",
    },
    {
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "LLY",
        "asset_name": "Eli Lilly & Co.",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-20",
        "transaction_date": "2026-08-05",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 5.2,
        "sentiment": "Bullish",
    },
    {
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "MSFT",
        "asset_name": "Microsoft Corp.",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$500,000 - $1,000,000",
        "filing_date": "2026-08-15",
        "transaction_date": "2026-07-30",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 3.8,
        "sentiment": "Bullish",
    },
    {
        "politician": "Dan Crenshaw (R-TX)",
        "chamber": "House",
        "ticker": "PLTR",
        "asset_name": "Palantir Technologies",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$50,000 - $100,000",
        "filing_date": "2026-08-22",
        "transaction_date": "2026-08-10",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 12,
        "performance_since_pct": 12.1,
        "sentiment": "Strong Bullish",
    },
    {
        "politician": "Josh Gottheimer (D-NJ)",
        "chamber": "House",
        "ticker": "AAPL",
        "asset_name": "Apple Inc.",
        "transaction_type": "Sale (Partial)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-10",
        "transaction_date": "2026-07-25",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": -1.2,
        "sentiment": "Neutral / Profit Take",
    },
]

UNUSUAL_OPTIONS_FLOW: List[Dict[str, Any]] = [
    {
        "time": "10:42:15",
        "ticker": "NVDA",
        "type": "CALL SWEEP",
        "strike": "$220.00",
        "expiration": "2026-09-18",
        "spot_price": 213.05,
        "premium": "$3,450,000",
        "volume_oi_ratio": 4.85,
        "implied_volatility": "44.2%",
        "order_type": "Ask (Aggressive Buying)",
        "sentiment": "Strong Bullish",
    },
    {
        "time": "10:38:40",
        "ticker": "NVO",
        "type": "CALL BLOCK",
        "strike": "$145.00",
        "expiration": "2026-10-16",
        "spot_price": 138.50,
        "premium": "$1,820,000",
        "volume_oi_ratio": 3.42,
        "implied_volatility": "32.8%",
        "order_type": "Above Ask (High Urgency)",
        "sentiment": "Strong Bullish",
    },
    {
        "time": "10:31:05",
        "ticker": "TSLA",
        "type": "PUT SWEEP",
        "strike": "$330.00",
        "expiration": "2026-08-28",
        "spot_price": 350.25,
        "premium": "$2,150,000",
        "volume_oi_ratio": 5.12,
        "implied_volatility": "58.4%",
        "order_type": "Bid (Aggressive Hedging)",
        "sentiment": "Bearish / Tail Hedge",
    },
    {
        "time": "10:24:50",
        "ticker": "SPY",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 765.91,
        "premium": "$48,200,000",
        "volume_oi_ratio": 2.10,
        "implied_volatility": "14.5%",
        "order_type": "Cross Trade",
        "sentiment": "Institutional Inflow",
    },
    {
        "time": "10:15:22",
        "ticker": "LLY",
        "type": "CALL SWEEP",
        "strike": "$950.00",
        "expiration": "2026-11-20",
        "spot_price": 920.40,
        "premium": "$2,640,000",
        "volume_oi_ratio": 3.90,
        "implied_volatility": "28.5%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Strong Bullish",
    },
]

class SmartMoneyEngine:
    """Quantitative engine for tracking Capitol Hill disclosures & institutional options flow."""

    @staticmethod
    def get_congressional_trades(symbol: str = None) -> List[Dict[str, Any]]:
        if symbol:
            sym_clean = symbol.upper().strip()
            return [t for t in CONGRESSIONAL_TRADES if t["ticker"] == sym_clean]
        return CONGRESSIONAL_TRADES

    @staticmethod
    def get_options_flow(symbol: str = None) -> List[Dict[str, Any]]:
        if symbol:
            sym_clean = symbol.upper().strip()
            return [f for f in UNUSUAL_OPTIONS_FLOW if f["ticker"] == sym_clean]
        return UNUSUAL_OPTIONS_FLOW

    @staticmethod
    def get_smart_money_overview() -> Dict[str, Any]:
        return {
            "total_congress_filings_30d": len(CONGRESSIONAL_TRADES),
            "net_political_sentiment": "Bullish (83.3% Purchases)",
            "top_congress_bought_sector": "Semiconductors & GLP-1 Healthcare",
            "unusual_flow_volume_today": "$58.2M",
            "call_to_put_dollar_ratio": 2.85,
            "congress_trades": CONGRESSIONAL_TRADES,
            "options_flow": UNUSUAL_OPTIONS_FLOW,
        }
