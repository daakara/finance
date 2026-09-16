"""Authoritative Live Runtime Smoke Test for ARX Recommendation Consistency & Decision Contract."""

import sys
from fastapi.testclient import TestClient
from api.main import app
from analyst_dashboard.analyzers.optimal_execution import ACTIONABLE_EXECUTION_STATUSES

client = TestClient(app)

def run_smoke():
    print("==================================================================")
    print("RUNNING LIVE RECOMMENDATION & DECISION CONTRACT RUNTIME SMOKE")
    print("==================================================================")

    # 1. Horizon & User Role Query Propagation Test
    print("\n[SMOKE 1] Horizon & User Role Query Propagation")
    res_swing = client.get("/api/v1/analytics/NVDA?user_role=SWING_TRADER")
    assert res_swing.status_code == 200, f"Analytics request failed: {res_swing.status_code}"
    data_swing = res_swing.json()
    assert data_swing.get("userRole") == "SWING_TRADER", f"Expected SWING_TRADER, got {data_swing.get('userRole')}"
    opt_swing = data_swing.get("optimalExecution") or {}
    print(f"  [OK] NVDA SWING_TRADER: userRole={data_swing.get('userRole')}, exec_status={opt_swing.get('execution_status')}, is_actionable={opt_swing.get('is_actionable')}")

    res_day = client.get("/api/v1/analytics/NVDA?user_role=DAY_TRADER")
    assert res_day.status_code == 200, f"Analytics request failed: {res_day.status_code}"
    data_day = res_day.json()
    assert data_day.get("userRole") == "DAY_TRADER", f"Expected DAY_TRADER, got {data_day.get('userRole')}"
    opt_day = data_day.get("optimalExecution") or {}
    print(f"  [OK] NVDA DAY_TRADER: userRole={data_day.get('userRole')}, exec_status={opt_day.get('execution_status')}, is_actionable={opt_day.get('is_actionable')}")

    # Verify tighter stop loss for DAY_TRADER vs SWING_TRADER if levels exist
    if opt_swing.get("stop_loss_pct") and opt_day.get("stop_loss_pct"):
        print(f"  [OK] Stop Loss Pct: Swing={opt_swing.get('stop_loss_pct')}% vs Day={opt_day.get('stop_loss_pct')}%")
        assert abs(opt_day.get("stop_loss_pct")) <= abs(opt_swing.get("stop_loss_pct")), "Day trader stop loss must be tighter than swing trader"

    # 2. Tactical Setups Actionability Parity Test
    print("\n[SMOKE 2] Tactical Setups Actionability & Status Taxonomy Parity")
    res_setups = client.get("/api/v1/analytics/setups?user_role=SWING_TRADER")
    assert res_setups.status_code == 200, f"Tactical setups request failed: {res_setups.status_code}"
    setups_payload = res_setups.json()
    setups = setups_payload.get("setups", [])
    print(f"  [OK] Retrieved {len(setups)} setups from /api/v1/analytics/setups")
    
    assert len(setups) > 0, "Setups list must not be empty"

    for s in setups:
        sym = s.get("symbol") or s.get("ticker")
        status = s.get("executionStatus")
        is_act = s.get("isActionable")
        entry = s.get("entryPivot")
        stop = s.get("stopLoss")
        is_supp = s.get("isSuppressed")

        dec_state = s.get("decisionState")
        reason = s.get("reasonSuppressed")

        if is_act:
            assert status in ACTIONABLE_EXECUTION_STATUSES, f"Actionable setup {sym} has non-actionable status: {status}"
            assert entry is not None and entry > 0, f"Actionable setup {sym} missing valid entryPivot: {entry}"
            assert stop is not None and stop > 0, f"Actionable setup {sym} missing valid stopLoss: {stop}"
            assert is_supp is False, f"Actionable setup {sym} cannot be marked suppressed"
            assert dec_state == "ACTIONABLE_SETUP", f"Actionable setup {sym} must have ACTIONABLE_SETUP decision state, got: {dec_state}"
        else:
            assert is_supp is True, f"Non-actionable setup {sym} must be marked suppressed"
            assert dec_state != "ACTIONABLE_SETUP", f"Suppressed setup {sym} cannot have ACTIONABLE_SETUP state, got: {dec_state}"
            assert reason is not None and len(str(reason)) > 0, f"Suppressed setup {sym} must state reason for suppression"

    actionable_count = sum(1 for s in setups if s.get("isActionable"))
    suppressed_count = sum(1 for s in setups if not s.get("isActionable"))
    print(f"  [OK] Verified {len(setups)} setups: {actionable_count} actionable, {suppressed_count} suppressed. Zero taxonomy leaks.")

    # 3. Suppressed Non-Actionable Ticker Test (e.g. WAITING_PULLBACK)
    print("\n[SMOKE 3] Single-Ticker Setup Endpoint Consistency")
    res_nvda_setup = client.get("/api/v1/analytics/setups/NVDA?user_role=SWING_TRADER")
    assert res_nvda_setup.status_code == 200
    nvda_setup = res_nvda_setup.json()
    assert nvda_setup.get("userRole") == "SWING_TRADER"
    assert nvda_setup.get("executionStatus") in ("IN_BUY_ZONE", "READY_TO_BUY", "WAITING_PULLBACK", "APPROACHING_TARGET")
    if nvda_setup.get("isActionable"):
        assert nvda_setup.get("executionStatus") in ACTIONABLE_EXECUTION_STATUSES
    else:
        assert nvda_setup.get("isSuppressed") is True
    print(f"  [OK] /analytics/setups/NVDA: status={nvda_setup.get('executionStatus')}, isActionable={nvda_setup.get('isActionable')}, isSuppressed={nvda_setup.get('isSuppressed')}")

    # 4. Unknown / Delisted Asset Returns 404 without Hallucination
    print("\n[SMOKE 4] Unknown Asset Rejection (Zero Fabricated Levels)")
    res_unknown = client.get("/api/v1/analytics/FAKEUNVERIFIED99?user_role=SWING_TRADER")
    assert res_unknown.status_code == 404, f"Expected 404 for unknown asset, got {res_unknown.status_code}"
    print(f"  [OK] Unknown asset correctly rejected with 404 (no fabricated levels generated)")

    res_unknown_setup = client.get("/api/v1/analytics/setups/FAKEUNVERIFIED99?user_role=SWING_TRADER")
    assert res_unknown_setup.status_code in (404, 422), f"Expected 404 or 422 for unknown setup, got {res_unknown_setup.status_code}"
    print(f"  [OK] Unknown asset setup correctly rejected with {res_unknown_setup.status_code}")

    # 5. Direct Parity Test: Analysis (/analytics/{sym}) vs Trade Plan (/analytics/setups/{sym})
    print("\n[SMOKE 5] Analysis vs Trade Plan Recommendation Parity")
    test_symbols = ["NVDA", "AAPL"]
    for sym in test_symbols:
        res_analysis = client.get(f"/api/v1/analytics/{sym}?period=1y&interval=1d&user_role=SWING_TRADER")
        res_setup = client.get(f"/api/v1/analytics/setups/{sym}?user_role=SWING_TRADER")
        if res_analysis.status_code == 200 and res_setup.status_code == 200:
            a_data = res_analysis.json()
            s_data = res_setup.json()
            trace = a_data.get("decisionTrace", {})
            trace_state = trace.get("decisionState")
            setup_state = s_data.get("decisionState")
            trace_act = trace.get("isActionable")
            setup_act = s_data.get("isActionable")
            
            assert trace_state == setup_state, (
                f"{sym} state divergence: Analysis={trace_state} vs TradePlan={setup_state}"
            )
            assert trace_act == setup_act, (
                f"{sym} actionability divergence: Analysis={trace_act} vs TradePlan={setup_act}"
            )
            print(f"  [OK] {sym} Parity Confirmed: decisionState={trace_state}, isActionable={trace_act}")

    print("\n==================================================================")
    print("ALL LIVE RUNTIME SMOKE TESTS PASSED!")
    print("==================================================================")

if __name__ == "__main__":
    run_smoke()
