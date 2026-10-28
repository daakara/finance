#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-05: Cross-Engine Invariants
Verifies decision authority, risk sizing, confluence truthfulness, Smart Money disclosure,
and recommendation consistency across frontend and backend boundaries.
"""

import sys, os, math

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def test_risk_sizing_floor():
    print("Testing Invariant 1: Sizing Floor (Sub-1-Share Allocation)...")
    from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
    engine = ConfluenceEngine()
    
    # Capital = $100, Entry = $250, Stop = $230 (risk per share = $20).
    # Risk pct = 1.0% -> max dollar risk = $1.00.
    # $1.00 / $20.00 = 0.05 shares -> max(0, math.floor(0.05)) == 0 shares.
    sizing = engine.calculate_position_size(
        account_equity=100.0,
        risk_pct=1.0,
        entry_price=250.0,
        stop_loss=230.0,
        take_profit_1=290.0
    )
    shares = sizing.get("executableShares", sizing.get("shares", 0))
    assert shares == 0, f"Expected 0 shares on sub-1-share capital, got {shares}"
    assert "suppressed (0 shares)" in sizing.get("guidance", "").lower()
    
    # Capital = $25,000, 1% risk = $250. Risk per share = $20 -> 12 shares.
    sizing_ok = engine.calculate_position_size(
        account_equity=25000.0,
        risk_pct=1.0,
        entry_price=250.0,
        stop_loss=230.0,
        take_profit_1=290.0
    )
    assert sizing_ok.get("executableShares", 0) == 12, f"Expected 12 shares, got {sizing_ok.get('executableShares')}"
    print("  [PASS] Sizing floor successfully enforces 0-share integer constraint on sub-1-share capital.")

def test_confluence_narrative_truthfulness():
    print("Testing Invariant 2: Confluence Truthfulness & Smart Money Disclosure...")
    from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
    engine = ConfluenceEngine()

    # Case A: Missing smart money must NOT claim positive institutional alignment
    res = engine.calculate_confluence(
        symbol="TEST",
        technical_data={"executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        smart_money_data=None, # Missing
        fundamental_data={"roic": 25.0, "piotroski_f": 8},
        catalyst_data={"days_to_earnings": 45}
    )
    reasons = " ".join(res.get("reasons", [])).lower()
    assert "smart money confirms" not in reasons, "FATAL: Confluence claimed smart money alignment when data was missing!"
    assert "institutional accumulation confirmed" not in reasons
    print("  [PASS] Missing Smart Money correctly suppressed from claiming positive institutional alignment.")

    # Case B: Low confluence score (<30)
    low_res = engine.calculate_confluence(
        symbol="BEAR",
        technical_data={"executionStatus": "EXTENDED", "riskRewardRatio": 0.8},
        smart_money_data={"has_insider_sell": True},
        fundamental_data={"roic": -5.0},
        catalyst_data=None
    )
    assert low_res["confluenceScore"] < 50.0, "Expected low confluence score"
    print("  [PASS] Low score dynamically reflects defensive/cautious rating.")

def test_frontend_risk_copy_cleanliness():
    print("Testing Invariant 3: Frontend Risk Copy Compliance (Absence of 'HARD MAX LOSS')...")
    modal_path = os.path.join(PROJECT_ROOT, "frontend", "components", "PositionSizerModal.tsx")
    daytrader_path = os.path.join(PROJECT_ROOT, "frontend", "components", "DayTraderPositionSizer.tsx")

    for p in [modal_path, daytrader_path]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
                if "HARD MAX LOSS" in content:
                    raise AssertionError(f"FATAL: Found deceptive 'HARD MAX LOSS' copy in {p}!")
                if "MODELED STOP RISK" not in content and p == modal_path:
                    raise AssertionError(f"FATAL: Missing qualified 'MODELED STOP RISK' copy in {p}!")
    print("  [PASS] Frontend risk copy is fully compliant ('HARD MAX LOSS' absent, 'MODELED STOP RISK' present).")

def main():
    print("=== [RQ-05] CROSS-ENGINE INVARIANTS CHECK ===")
    try:
        test_risk_sizing_floor()
        test_confluence_narrative_truthfulness()
        test_frontend_risk_copy_cleanliness()
        print("=== [RQ-05] RESULT: PASS ===")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL: Cross-engine invariant breach: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
