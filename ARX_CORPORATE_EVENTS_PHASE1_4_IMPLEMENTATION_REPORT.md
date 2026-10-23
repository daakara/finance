# ARX Terminal — Corporate Events Phase 1–4 Implementation Report

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Track:** Track B (Corporate Events and Pre-Flight Integrity)  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-earnings-audit`  
**Branch:** `audit/axp-earnings-news-coverage`  
**Base Commit SHA:** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9`  
**Execution Mode:** ISOLATED LOCAL IMPLEMENTATION & TEST VERIFICATION  
**Production Mutation Authorization:** NOT GRANTED  

---

## 1. Executive Summary

In accordance with the findings established in `ARX_AXP_EARNINGS_COVERAGE_AUDIT.md`, Phases 1 through 4 of the Corporate Events Remediation Plan have been implemented and verified in the isolated Track B worktree. 

This implementation permanently cures the two critical epistemic failure modes:
1. **Pre-Flight False-Safety Pass:** Check 4 previously coerced missing or unverified corporate event data into a green `PASS` ("Sufficient time window to manage trade without overnight earnings gap risk"). Check 4 now enforces an explicit 4-state fail-closed model (`PASS`, `WARN`, `UNAVAILABLE`, `ERROR`), strictly prohibiting `UNAVAILABLE` from rendering as a safety `PASS`.
2. **Historical Earnings Binary Penalty Leak:** In `confluence_engine.py`, `days_to_earnings <= 1.0` lacked a lower bound of `0.0`, penalizing past events (`days_to_earnings < 0`) with an indefinite -25 point penalty. The engine now enforces explicit positive bounds (`0.0 <= days_to_earnings <= 1.0` and `1.0 < days_to_earnings <= 7.0`) and explicitly skips elapsed/completed events.

The implementation introduces zero commercial licensing costs by normalizing existing provider payloads (`yfinance` company info & calendar) and preserving canonical primary-source IR verification.

---

## 2. Root Cause to Code-Change Traceability

| Defect ID | Architectural Layer | Prior Code Behavior | Remediated Implementation | Target File(s) |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-CE-01** | Scoring Engine | `days_to_earnings <= 1.0` applied -25 pt binary penalty to past/negative days. | Added `0.0 <= days_to_earnings <= 1.0` and `1.0 < days_to_earnings <= 7.0`; guarded `is_completed` and negative elapsed days. | [`confluence_engine.py`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/analyst_dashboard/analyzers/confluence_engine.py) |
| **DEF-AR-02** | Analytics Route | `ticker_obj.info` contained `earningsTimestamp: 1792758600` but discarded it. | Extracted earnings fields, passed through `normalize_corporate_event`, and wired to `catalyst_report`, `optimal_execution_plan`, and `corporateEvent`. | [`api/routes/analytics.py`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/api/routes/analytics.py) |
| **DEF-CE-03** | Corporate Normalization | No canonical model for corporate events, confirmation tiers, or call vs release time. | Created `normalize_corporate_event` module establishing canonical typed schema, IR schedules, and timezone normalization. | [`corporate_events.py`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/analyst_dashboard/corporate_events.py) |
| **DEF-CT-04** | Catalyst Engine | Hardcoded static 14-symbol dictionary; wiped milestones in live mode. | Updated `get_asset_catalyst_report` to accept `corporate_event` and dynamically attach upcoming binary milestones. | [`catalysts.py`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/analyst_dashboard/analyzers/catalysts.py) |
| **DEF-PF-05** | Pre-Flight Modal | `!hasImminentEarnings` caused missing data to evaluate to `isCatalystPassed = true`. | Implemented explicit 4-state machine (`PASS`, `WARN`, `UNAVAILABLE`, `ERROR`). `UNAVAILABLE` strictly evaluates to `isCatalystPassed = false`. | [`PreFlightChecklistModal.tsx`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/frontend/components/PreFlightChecklistModal.tsx) |
| **DEF-OE-06** | Stock Analysis UI | Looked for nonexistent `executionPlan.catalystForecast.catalysts`. | Derived canonical event state from `corporateEvent` / `catalystForecast`, displaying live timing, source, and confirmation status. | [`OptimalEntryExitCard.tsx`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/frontend/components/OptimalEntryExitCard.tsx) |
| **DEF-TY-07** | Frontend Contract | Missing TypeScript types for corporate event telemetry. | Defined `CorporateEventRecord`, `EventRiskTier`, `EventCoverageStatus`, and updated `AssetAnalyticsResponse`. | [`frontend/lib/api.ts`](file:///C:/Users/akara/Documents/Projects/finance-earnings-audit/frontend/lib/api.ts) |

---

## 3. Exact Changed Files Summary

```text
7 files modified, 3 new source/test files created:
------------------------------------------------------------------------
 M analyst_dashboard/analyzers/catalysts.py         (+42, -2)
 M analyst_dashboard/analyzers/confluence_engine.py (+36, -10)
 M api/routes/analytics.py                          (+20, -0)
 M frontend/app/page.tsx                            (+2, -0)
 M frontend/components/OptimalEntryExitCard.tsx     (+83, -12)
 M frontend/components/PreFlightChecklistModal.tsx  (+107, -18)
 M frontend/lib/api.ts                              (+60, -0)
 + analyst_dashboard/corporate_events.py            (177 lines)
 + frontend/components/__tests__/PreFlightCatalystRisk.test.tsx (96 lines)
 + tests/test_corporate_events_earnings_audit.py    (208 lines)
------------------------------------------------------------------------
Total: 631 insertions(+), 42 deletions(-)
```

---

## 4. Key Architectural Enhancements

### A. Non-Overlapping Confluence Time Bounds
```python
# Guard against past or completed events
is_completed = bool(
    catalyst_data.get("is_completed", False)
    or catalyst_data.get("status") in ("COMPLETED", "ELAPSED", "ELAPSED_RECENT")
    or catalyst_data.get("coverage_status") in ("COMPLETED", "ELAPSED", "ELAPSED_RECENT")
)
raw_dte = catalyst_data.get("days_to_earnings")
if raw_dte is not None and not is_completed:
    try:
        days_to_earnings = float(raw_dte)
        if math.isnan(days_to_earnings) or math.isinf(days_to_earnings):
            days_to_earnings = None
    except (ValueError, TypeError):
        days_to_earnings = None

    if days_to_earnings is not None:
        if 0.0 <= days_to_earnings <= 1.0:
            catalyst_mod -= 25.0
            hours_left = max(1, int(days_to_earnings * 24))
            warnings.append(f"⚠️ HIGH BINARY GAP RISK: Earnings in <={hours_left}h! Limit position sizing.")
        elif 1.0 < days_to_earnings <= 7.0:
            catalyst_mod -= 8.0
            warnings.append(f"Caution: Earnings in {days_to_earnings} days.")
        elif days_to_earnings < 0.0:
            # Elapsed event: historical reporting cannot indefinitely penalize future setups
            pass
```

### B. Distinguishing Results Release vs. Conference Call Times
American Express (NYSE: AXP) schedules results release at ~7:00 AM Eastern, followed by the live conference webcast at 8:30 AM Eastern.
Aggregators (yfinance) supply `earningsTimestamp: 1792758600` (`12:30 UTC` = `08:30 EDT`), matching `earningsCallTimestampStart`. 
In `analyst_dashboard/corporate_events.py`:
- `conference_call_time_et`: "08:30"
- `results_release_time_et`: "07:00"
- Rule: Generic tickers with only date telemetry do not have synthetic release times fabricated.
- Rule: `isEarningsDateEstimate: false` alone is classified as `"UNVERIFIED"` aggregator reporting unless verified against primary corporate IR.

### C. Pre-Flight 4-State Fail-Closed Machine
Check 4 now resolves:
- `PASS`: Certified clean runway (>7 days DTE or past quarter elapsed).
- `WARN`: Relevant upcoming event (DTE between 0.0 and 7.0 days).
- `UNAVAILABLE`: Absent, stale, or uncertified coverage. Pre-Flight badge renders `UNAVAILABLE` (slate), `isCatalystPassed = false`.
- `ERROR`: Technical evaluation error.
