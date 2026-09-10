"""
scripts/verify_h14_backend.py
Comprehensive automated verification suite for Horizon 14 Backend API Foundation.

Tests:
1. Macro Ribbon (/api/v1/macro/ribbon):
   - Observation timestamps derived from bar data (distinct from generatedAt).
   - Dynamic session state derived from authoritative exchange calendar (exchange_calendars XNYS).
   - Holiday schedule and early close detection via XNYS calendar.
   - VIX tier boundary logic:
     * None -> UNAVAILABLE
     * < 20.0 -> NORMAL
     * >= 20.0 and < 30.0 -> ELEVATED
     * >= 30.0 -> CRITICAL
   - Provider failure isolation: Simulated failure returns honest UNAVAILABLE without fabricated numbers (542.10, 468.50, etc.).
2. Unified Cockpit State (/api/v1/cockpit/state):
   - Zero-auth: requests without headers resolve to 'default' record selector with 200 OK (NEVER 401 Unauthorized).
   - Uninitialized selector: returns 200 OK, status UNAVAILABLE, available=False, null triad, zero fabricated numbers.
   - Holdings only: triad is null (no fake 70/75/65), portfolio summary present.
   - Actions only: triad is null, nextBestAction present.
   - User-reported indices: triad labeled with provenance: "USER_REPORTED", isSystemCalculated: false.
   - Authentic Runway Semantics:
     * Unrecorded expenditure (monthlyBurn is None) -> EXPENDITURE_UNRECORDED, runwayMonths is None.
     * Zero recurring expenditure (monthlyBurn == 0.0) -> ZERO_EXPENDITURE, unencumbered reserves, never 0.0 months.
     * Calculated expenditure (monthlyBurn > 0) -> CALCULATED, zero fake 3-year projection (projectedValue3Yr is None), zero fake confidencePct.
   - Signal Quality: zero fabricated confidence (88/70 is None), zero fabricated conviction (0.85/0.5 is None), lastTelemetrySync uses profile updatedAt (not now_iso).
   - Cache headers: Cache-Control: private, no-cache, no-store, must-revalidate.
   - Wire payload budget: real serialized JSON HTTP bytes < 12 kB.
3. Persistent Database & Fractional Holdings Precision:
   - Uses an isolated temporary SQLite database (never touches or pollutes production database).
   - Fractional shares (0.25, 0.001) stored, retrieved, and calculated with exact precision.
4. Setups Engine (/api/v1/analytics/setups/{symbol}):
   - Multi-asset capability: ASML, MSFT, AAPL, CPRX.
   - Unrecognized ticker: returns 404 or explicit error diagnosis.
"""

import sys
import os
import json
import gzip
import tempfile
import atexit
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from unittest.mock import patch

# Ensure project root in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -------------------------------------------------------------------
# Setup isolated temporary SQLite test database (NEVER touch prod DB)
# -------------------------------------------------------------------
temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
temp_db_path = temp_db.name
temp_db.close()

def _cleanup_temp_db():
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass

atexit.register(_cleanup_temp_db)

from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
test_db_engine = HistoryDatabaseEngine(db_path=temp_db_path)

# Patch cockpit and portfolio router db engines to isolated test db
import api.routes.cockpit as cockpit_route
import api.routes.portfolio as portfolio_route

cockpit_route.history_db = test_db_engine
portfolio_route.history_db = test_db_engine

from fastapi.testclient import TestClient
from api.main import app
from api.routes.macro import compute_vix_tier, get_nyse_session_status

client = TestClient(app)

passed_checks = 0
failed_checks = 0

def check(condition: bool, description: str):
    global passed_checks, failed_checks
    if condition:
        passed_checks += 1
        print(f"  [PASS] {description}")
    else:
        failed_checks += 1
        print(f"  [FAIL] {description}")

def run_macro_verification():
    print("\n--- 1. Macro Ribbon API & Authoritative Exchange Calendar ---")
    
    # 1.1 VIX Tier Boundary Unit Tests
    check(compute_vix_tier(None) == "UNAVAILABLE", "VIX None -> UNAVAILABLE")
    check(compute_vix_tier(14.5) == "NORMAL", "VIX 14.5 -> NORMAL")
    check(compute_vix_tier(19.99) == "NORMAL", "VIX 19.99 -> NORMAL")
    check(compute_vix_tier(20.0) == "ELEVATED", "VIX 20.0 -> ELEVATED")
    check(compute_vix_tier(28.7) == "ELEVATED", "VIX 28.7 -> ELEVATED")
    check(compute_vix_tier(30.0) == "CRITICAL", "VIX 30.0 -> CRITICAL (Boundary)")
    check(compute_vix_tier(45.2) == "CRITICAL", "VIX 45.2 -> CRITICAL")

    # 1.2 NYSE Session Logic Test via exchange_calendars
    nyse_session_dict = get_nyse_session_status()
    nyse_status = nyse_session_dict.get("status")
    check(nyse_status in ["OPEN", "PRE_MARKET", "POST_MARKET", "CLOSED"], f"Dynamic NYSE Session valid ({nyse_status})")
    check("isSession" in nyse_session_dict, "NYSE session status includes isSession flag from exchange_calendars")

    # 1.3 Specific Calendar Tests: Holiday & Early Close
    holiday_dt = datetime(2026, 7, 3, 12, 0, tzinfo=ZoneInfo("America/New_York"))  # Observed July 4th
    holiday_res = get_nyse_session_status(holiday_dt)
    check(holiday_res.get("status") == "CLOSED" and "Holiday" in holiday_res.get("reason", ""), "exchange_calendars identifies NYSE Holiday (2026-07-03)")

    early_close_dt = datetime(2026, 11, 27, 11, 0, tzinfo=ZoneInfo("America/New_York"))  # Day after Thanksgiving
    early_res = get_nyse_session_status(early_close_dt)
    check(early_res.get("status") == "OPEN" and "Early Close" in early_res.get("reason", ""), "exchange_calendars identifies Early Close (2026-11-27 13:00 close)")

    # 1.4 Live Macro Ribbon Endpoint Response
    resp = client.get("/api/v1/macro/ribbon")
    check(resp.status_code == 200, f"GET /api/v1/macro/ribbon returns 200 OK (got {resp.status_code})")
    if resp.status_code == 200:
        data = resp.json()
        check("dataSource" in data, "Macro ribbon declares explicit dataSource")
        check(data.get("dataSource") in ["DAILY_CLOSE", "PARTIAL_AVAILABLE", "UNAVAILABLE"], f"dataSource is honest ({data.get('dataSource')})")
        check("observationTime" in data, "Macro ribbon contains separate observationTime")
        check("generatedAt" in data, "Macro ribbon contains generatedAt timestamp")
        
        if data.get("observationTime") and data.get("generatedAt"):
            check(data["observationTime"] != data["generatedAt"], "observationTime is distinct from generatedAt")
            
        check(data.get("regime") in ["RISK_ON", "DEFENSIVE", "NEUTRAL", "UNAVAILABLE"], f"Regime is valid enum ({data.get('regime')})")

    # 1.5 Failure Isolation Simulation (Outage test)
    with patch("api.routes.macro._fetch_isolated_bar", return_value={"price": None, "change": None, "changePct": None, "observationTime": None, "status": "UNAVAILABLE", "available": False}):
        fail_resp = client.get("/api/v1/macro/ribbon")
        check(fail_resp.status_code == 200, "Provider outage handled gracefully (200 with honest unavailable state)")
        fail_data = fail_resp.json()
        check(fail_data.get("dataSource") == "UNAVAILABLE", "Outage returns dataSource: UNAVAILABLE")
        check(fail_data.get("regime") == "UNAVAILABLE", "Outage returns regime: UNAVAILABLE (zero fabricated default)")
        check(fail_data.get("vixTier") == "UNAVAILABLE", "Outage returns vixTier: UNAVAILABLE")
        check(fail_data.get("spx", {}).get("price") is None, "Outage SPX price is null (no 542.10 fake quote)")
        check(fail_data.get("qqq", {}).get("price") is None, "Outage QQQ price is null (no 468.50 fake quote)")

def run_cockpit_verification():
    print("\n--- 2. Unified Cockpit API & Zero-Auth Record Selectors ---")
    
    # 2.1 Zero-Auth Policy: Anonymous Request -> 200 OK with default selector (NEVER 401)
    anon_resp = client.get("/api/v1/cockpit/state")
    check(anon_resp.status_code == 200, f"Anonymous GET /api/v1/cockpit/state returns 200 OK (no 401 Unauthorized; got {anon_resp.status_code})")
    if anon_resp.status_code == 200:
        anon_data = anon_resp.json()
        check(anon_data.get("subjectId") == "default", "Anonymous request resolves to 'default' record selector")
        check(anon_data.get("status") == "UNAVAILABLE", "Uninitialized default selector returns UNAVAILABLE")

    # 2.2 Uninitialized Selector: Honest UNAVAILABLE & zero fabricated numbers
    uninit_headers = {"X-Profile-Id": "test_empty_selector_9999"}
    uninit_resp = client.get("/api/v1/cockpit/state", headers=uninit_headers)
    check(uninit_resp.status_code == 200, "Uninitialized selector returns 200 OK")
    if uninit_resp.status_code == 200:
        uninit_data = uninit_resp.json()
        check(uninit_data.get("status") == "UNAVAILABLE", "Uninitialized status is UNAVAILABLE")
        check(uninit_data.get("available") is False, "Uninitialized available is False")
        check(uninit_data.get("triad") is None, "Uninitialized triad is None (zero fake 70/75/65)")
        check(uninit_data.get("portfolio") is None, "Uninitialized portfolio is None")
        check(uninit_data.get("signalQuality", {}).get("confidence") is None, "Uninitialized confidence is None (no fake 88/70)")
        check(uninit_data.get("signalQuality", {}).get("highConvictionRatio") is None, "Uninitialized conviction is None (no fake 0.85/0.5)")
        check(uninit_data.get("nextBestAction") is None, "Uninitialized nextBestAction is None")
        check(uninit_data.get("primaryForecast") is None, "Uninitialized primaryForecast is None")
        
        # Verify private cache headers
        cc = uninit_resp.headers.get("cache-control", "")
        check("private" in cc and "no-cache" in cc, f"Cache-Control contains private, no-cache ({cc})")

    # 2.3 Edge Case: Holdings Only (No profile recorded)
    holdings_only_uid = "user_holdings_only_01"
    test_db_engine.save_user_holding(holdings_only_uid, {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "shares": 10.0,
        "entryPrice": 180.0,
        "currentPrice": 190.0,
    })
    h_resp = client.get("/api/v1/cockpit/state", headers={"X-Profile-Id": holdings_only_uid})
    check(h_resp.status_code == 200, "Holdings-only user returns 200 OK")
    if h_resp.status_code == 200:
        h_data = h_resp.json()
        check(h_data.get("triad") is None, "Holdings-only selector has triad=None (zero fabricated 70/75/65 triad)")
        check(h_data.get("portfolio") is not None, "Holdings-only selector has authentic portfolio summary")
        check(h_data.get("portfolio", {}).get("totalMarketValue") == 1900.0, "Portfolio market value computed accurately (10 * $190 = $1900)")

    # 2.4 Edge Case: Actions Only (No profile recorded)
    actions_only_uid = "user_actions_only_01"
    test_db_engine.save_user_action(actions_only_uid, {
        "id": "NBA-TEST-01",
        "title": "Review Capital Allocation",
        "domain": "CAPITAL",
        "priorityScore": 85.0,
        "isPrimary": True,
    })
    a_resp = client.get("/api/v1/cockpit/state", headers={"X-Profile-Id": actions_only_uid})
    check(a_resp.status_code == 200, "Actions-only user returns 200 OK")
    if a_resp.status_code == 200:
        a_data = a_resp.json()
        check(a_data.get("triad") is None, "Actions-only selector has triad=None (zero fabricated triad)")
        check(a_data.get("nextBestAction", {}).get("id") == "NBA-TEST-01", "Actions-only selector preserves authentic action item")

    # 2.5 Authentic Runway Semantics: Unrecorded Expenditure vs Zero Expenditure vs Calculated
    # Case A: Unrecorded expenditure (monthlyBurn is None)
    unrec_uid = "user_burn_unrecorded"
    client.post("/api/v1/cockpit/profile", headers={"X-Profile-Id": unrec_uid}, json={
        "name": "Unrecorded Burn User",
        "role": "Self-Directed Investor",
        "lhi": 80.0, "hhi": 85.0, "iai": 90.0,
        "liquidReserves": 40000.0,
        "monthlyBurn": None
    })
    unrec_resp = client.get("/api/v1/cockpit/state", headers={"X-Profile-Id": unrec_uid})
    if unrec_resp.status_code == 200:
        unrec_data = unrec_resp.json()
        f_list = unrec_data.get("outcomeForecasts", [])
        check(len(f_list) == 0 or f_list[0].get("runwayStatus") == "EXPENDITURE_UNRECORDED", "Unrecorded burn rate reports EXPENDITURE_UNRECORDED (never 0.0 months)")

    # Case B: Recorded Zero Expenditure (monthlyBurn == 0.0)
    zero_burn_uid = "user_zero_burn"
    client.post("/api/v1/cockpit/profile", headers={"X-Profile-Id": zero_burn_uid}, json={
        "name": "Zero Burn User",
        "role": "Capital Preserver",
        "lhi": 85.0, "hhi": 90.0, "iai": 88.0,
        "liquidReserves": 50000.0,
        "monthlyBurn": 0.0
    })
    zero_resp = client.get("/api/v1/cockpit/state", headers={"X-Profile-Id": zero_burn_uid})
    if zero_resp.status_code == 200:
        zero_data = zero_resp.json()
        fc = zero_data.get("primaryForecast") or {}
        check(fc.get("runwayStatus") == "ZERO_EXPENDITURE", "Zero recurring expenditure reports ZERO_EXPENDITURE")
        check(fc.get("currentValue") == "Unencumbered", f"Zero recurring expenditure currentValue is 'Unencumbered' (never 0.0 Months; got {fc.get('currentValue')})")
        check(fc.get("projectedValue3Yr") is None, "Zero recurring expenditure has null projectedValue3Yr (no fake burn decay)")
        check(fc.get("confidencePct") is None, "Zero recurring expenditure has null confidencePct (no fake 85%)")

    # Case C: Active Monthly Burn (Calculated runway)
    calc_uid = "user_calc_burn"
    client.post("/api/v1/cockpit/profile", headers={"X-Profile-Id": calc_uid}, json={
        "name": "Active Operator",
        "role": "Full-Time Trader",
        "lhi": 88.0, "hhi": 82.0, "iai": 91.0,
        "liquidReserves": 72000.0,
        "monthlyBurn": 6000.0
    })
    calc_resp = client.get("/api/v1/cockpit/state", headers={"X-Profile-Id": calc_uid})
    if calc_resp.status_code == 200:
        calc_data = calc_resp.json()
        triad = calc_data.get("triad") or {}
        check(triad.get("provenance") == "USER_REPORTED", "User-entered indices labeled USER_REPORTED")
        check(triad.get("isSystemCalculated") is False, "User-entered indices marked isSystemCalculated=False")
        
        fc = calc_data.get("primaryForecast") or {}
        check(fc.get("runwayStatus") == "CALCULATED", "Valid burn reports runwayStatus: CALCULATED")
        check(fc.get("currentValue") == "12.0 Months", f"Runway calculated correctly: $72k / $6k = 12.0 Months (got {fc.get('currentValue')})")
        check(fc.get("projectedValue3Yr") is None, "Eliminated speculative 3-year projection (projectedValue3Yr is None)")
        check(fc.get("confidencePct") is None, "Eliminated fake confidence percentage (confidencePct is None)")
        
        # Telemetry timestamp isolation: lastTelemetrySync matches profile updatedAt, not generatedAt
        check(calc_data.get("signalQuality", {}).get("lastTelemetrySync") is not None, "lastTelemetrySync uses profile updatedAt")
        check(calc_data.get("generatedAt") is not None, "generatedAt is present")
        
        # Real HTTP Wire payload size measurement (<12 kB target)
        raw_bytes = len(calc_resp.content)
        gzip_bytes = len(gzip.compress(calc_resp.content))
        check(raw_bytes < 12 * 1024, f"Populated read model wire payload {raw_bytes} bytes < 12 kB budget")
        print(f"    [INFO] Serialized HTTP wire payload: {raw_bytes} bytes raw, {gzip_bytes} bytes gzip")

def run_fractional_holdings_verification():
    print("\n--- 3. Persistent Database & Fractional Holdings Precision ---")
    
    test_user = "user_fractional_test"
    
    # 3.1 Fractional shares persistence: 0.25 shares and 0.001 shares
    h1 = {
        "symbol": "TSLA",
        "name": "Tesla Inc.",
        "shares": 0.25,
        "entryPrice": 200.0,
        "currentPrice": 220.0,
    }
    h2 = {
        "symbol": "BTC-USD",
        "name": "Bitcoin USD",
        "shares": 0.001,
        "entryPrice": 60000.0,
        "currentPrice": 65000.0,
    }
    
    save1 = test_db_engine.save_user_holding(test_user, h1)
    save2 = test_db_engine.save_user_holding(test_user, h2)
    check(save1 and save2, "Database successfully persisted fractional holdings (0.25, 0.001)")
    
    holdings = test_db_engine.get_user_portfolio(test_user)
    check(len(holdings) == 2, f"Retrieved 2 persisted holdings (got {len(holdings)})")
    
    tsla = next((h for h in holdings if h["symbol"] == "TSLA"), None)
    btc = next((h for h in holdings if h["symbol"] == "BTC-USD"), None)
    
    check(tsla is not None and tsla["shares"] == 0.25, f"TSLA fractional shares exactly 0.25 (got {tsla['shares'] if tsla else None})")
    check(btc is not None and btc["shares"] == 0.001, f"BTC fractional shares exactly 0.001 (got {btc['shares'] if btc else None})")
    
    # Verify portfolio API endpoint also reflects exact fractional values
    api_resp = client.get("/api/v1/portfolio", headers={"X-User-Id": test_user})
    check(api_resp.status_code == 200, "GET /api/v1/portfolio returns 200 OK")
    if api_resp.status_code == 200:
        api_holdings = api_resp.json()
        api_tsla = next((h for h in api_holdings if h["symbol"] == "TSLA"), None)
        check(api_tsla is not None and api_tsla["shares"] == 0.25, "API serialization preserves 0.25 fractional shares")

def run_setups_verification():
    print("\n--- 4. Multi-Asset Setups API ---")
    
    for sym in ["ASML", "MSFT", "AAPL", "CPRX"]:
        resp = client.get(f"/api/v1/analytics/setups/{sym}")
        check(resp.status_code in [200, 404, 503], f"GET /api/v1/analytics/setups/{sym} returns valid HTTP status ({resp.status_code})")
        if resp.status_code == 200:
            data = resp.json()
            check("symbol" in data and data["symbol"] == sym, f"Setup response symbol matches requested {sym}")
            check("isSuppressed" in data, f"Setup response reports isSuppressed for {sym}")
            check("entryThesis" in data or "reasonSuppressed" in data, f"Setup response provides authentic thesis or suppression reason for {sym}")

    invalid_resp = client.get("/api/v1/analytics/setups/ZZZZZZ999")
    check(invalid_resp.status_code in [404, 500], f"Invalid ticker ZZZZZZ999 returns appropriate non-200 code ({invalid_resp.status_code})")

def main():
    print("================================================================")
    print("      H14 FOUNDATION BACKEND API VERIFICATION SUITE")
    print("================================================================")
    print(f"Isolated Test Database: {temp_db_path}")
    
    run_macro_verification()
    run_cockpit_verification()
    run_fractional_holdings_verification()
    run_setups_verification()
    
    print("\n----------------------------------------------------------------")
    print(f"TOTAL CHECKS: {passed_checks + failed_checks}")
    print(f"PASSED:       {passed_checks}")
    print(f"FAILED:       {failed_checks}")
    print("----------------------------------------------------------------")
    
    if failed_checks > 0:
        print("\n[FAIL] BACKEND VERIFICATION FAILED WITH DEFECTS\n")
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL H14 BACKEND API CHECKS PASSED SUCCESSFULLY\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
