"""ARX Golden Regression Universe Fixture Definitions (Phase 20E).

Contains frozen canonical baseline data for 6 archetype assets:
  1. NVDA: Strong Institutional Compounder.
  2. FIX: Weak / Stage 4 Distribution Archetype.
  3. CPRX: Catalyst-Driven Bio-Pharma Runway.
  4. SPY: Broad-Market Macro Benchmark ETF.
  5. GLX: Insufficient-History Unseasoned Asset (< 50 sessions).
  6. TEST_AAA: Completely Unknown / Unverified Asset.
"""

from typing import Dict, Any

GOLDEN_ASSETS: Dict[str, Dict[str, Any]] = {
    "NVDA": {
        "archetype": "STRONG_COMPOUNDER",
        "expected_state": ["ACTIONABLE_SETUP", "VALID_SETUP"],
        "min_candle_count": 50,
        "requires_fundamentals": True,
        "min_confluence_score": 75.0,
        "is_actionable_expected": True,
        "can_size_trade_expected": True,
    },
    "FIX": {
        "archetype": "WEAK_DISTRIBUTION",
        "expected_state": ["VALID_SETUP"],
        "min_candle_count": 50,
        "requires_fundamentals": True,
        "expected_stage": 4,
        "max_confluence_score": 55.0,
        "is_actionable_expected": False,
        "can_size_trade_expected": False,
    },
    "CPRX": {
        "archetype": "CATALYST_DRIVEN",
        "expected_state": ["VALID_SETUP", "ACTIONABLE_SETUP"],
        "min_candle_count": 50,
        "has_catalyst": True,
    },
    "SPY": {
        "archetype": "BROAD_MARKET_ETF",
        "expected_state": ["VALID_SETUP", "ACTIONABLE_SETUP"],
        "min_candle_count": 50,
        "is_etf": True,
    },
    "GLX": {
        "archetype": "INSUFFICIENT_HISTORY",
        "expected_state": ["INSUFFICIENT_DATA"],
        "max_candle_count": 49,
        "is_actionable_expected": False,
        "can_size_trade_expected": False,
        "trade_levels_suppressed": True,
    },
    "TEST_AAA": {
        "archetype": "UNKNOWN_UNVERIFIED",
        "expected_state": ["UNVERIFIED"],
        "candle_count": 0,
        "is_actionable_expected": False,
        "can_size_trade_expected": False,
        "expected_http_status": 404,
    },
}
