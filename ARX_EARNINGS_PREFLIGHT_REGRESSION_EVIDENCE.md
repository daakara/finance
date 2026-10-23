# ARX Terminal — Earnings & Pre-Flight Regression Evidence

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Track:** Track B (Corporate Events and Pre-Flight Integrity)  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-earnings-audit`  
**Branch:** `audit/axp-earnings-news-coverage`  
**Base Commit SHA:** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9`  
**Execution Mode:** TEST VERIFICATION & EVIDENCE GENERATION  
**Production Mutation Authorization:** NOT GRANTED  

---

## 1. Before vs. After Pre-Flight Semantic Model

| State / Telemetry Scenario | Before Remediation (Defective) | After Remediation (Remediated) | Verdict & Invariant Enforcement |
| :--- | :--- | :--- | :--- |
| **AXP Earnings Today ($\le 24$h)** | Evaluated to `hasImminentEarnings = false` due to missing `catalysts` array. Check 4 rendered green **`PASS`**. | Evaluated to `resolvedCatalystState = "WARN"`. Check 4 renders rose **`HAZARD`** (`⚠️ High Risk: Scheduled earnings announcement in <= 24h`). | **FIXED** — Binary gap hazard actively surfaced; position sizing constrained. |
| **Past Earnings (e.g. DTE $= -1.0$)** | `days_to_earnings <= 1.0` triggered -25 pt binary penalty and acute gap warning on yesterday's event. | `days_to_earnings < 0.0` explicitly guarded. Zero penalty (`catalyst_mod = 0.0`), Check 4 renders **`PASS`** (`Prior earnings reported`). | **FIXED** — Historical reporting no longer indefinitely penalizes future setups. |
| **Missing / Uncataloged Stock** | `hasImminentEarnings` defaulted to `false`. Check 4 rendered green **`PASS`** ("Sufficient time window to manage trade"). | Missing coverage evaluates to `resolvedCatalystState = "UNAVAILABLE"`. Check 4 renders slate **`UNAVAILABLE`**, `isCatalystPassed = false`. | **FIXED** — Cardinal invariant enforced: missing data never produces a safety pass. |
| **Event within 48h ($1.0 < \text{DTE} \le 2.0$)** | Bypassed or coerced to `false`. | `resolvedCatalystState = "WARN"`. Check 4 renders rose **`HAZARD`** (`⚠️ High Gap Risk: Earnings announcement in 1.5 days`). | **FIXED** — Multi-day binary gap hazard cleanly differentiated. |
| **Event Runway Verified ($>7$ Days)** | Rendered `PASS` purely by default assumption. | Evaluates to `resolvedCatalystState = "PASS"`. Check 4 renders emerald **`PASS`** (`Verified event runway: No earnings announcement in 7-day trade window`). | **FIXED** — Clean pass backed by positive evidence. |
| **Technical Telemetry Error** | Error caught silently or fell back to default `false` (`PASS`). | Evaluates to `resolvedCatalystState = "ERROR"`. Check 4 renders rose **`ERROR`** badge, `isCatalystPassed = false`. | **FIXED** — Fail-closed error boundary established. |

---

## 2. Primary Regression Case: American Express (NYSE: AXP)

### A. Canonical Identity & Resolution
* **Symbol:** `AXP`
* **Canonical Name:** `AMERICAN EXPRESS CO`
* **CIK:** `0000004962`
* **Composite FIGI:** `BBG000BCQZS4`
* **Share Class FIGI:** `BBG001S5P034`
* **Primary Exchange:** `NYSE`

### B. Event Schedule & Distinction
* **Scheduled Event Date:** October 23, 2026
* **Results Release Time:** 7:00 AM Eastern Time (approximate pre-market release per American Express IR)
* **Live Audio Webcast / Conference Call:** 8:30 AM Eastern Time
* **Provider Timestamp (`earningsTimestamp`):** `1792758600` (corresponds precisely to `12:30 UTC` = `08:30 EDT` Conference Call)
* **Remediated Normalization Output:**
  ```json
  {
    "symbol": "AXP",
    "event_type": "EARNINGS_ANNOUNCEMENT",
    "event_timestamp_utc": 1792758600,
    "event_date_et": "2026-10-23",
    "event_time_et": "08:30",
    "time_zone": "America/New_York",
    "results_release_time_et": "07:00",
    "conference_call_time_et": "08:30",
    "confirmation_status": "CONFIRMED",
    "coverage_status": "CONFIRMED_UPCOMING",
    "risk_tier": "ACUTE_BINARY_GAP_RISK",
    "source_name": "American Express Investor Relations / SEC EDGAR",
    "preflight_state": "WARN"
  }
  ```

---

## 3. Test Execution Verification

### A. Extended 16-Scenario Corporate Events Backend Suite
```text
pytest tests/test_corporate_events_earnings_audit.py -v
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\akara\Documents\Projects\finance-earnings-audit
configfile: pyproject.toml
collected 16 items

tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_01_axp_earnings_scheduled_today_applies_maximum_gap_penalty PASSED [  6%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_02_axp_earnings_call_vs_results_release_distinction PASSED [ 12%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_03_confirmed_vs_estimated_dates_distinguished PASSED [ 18%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_04_earnings_within_24_and_48_hours PASSED [ 25%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_05_earnings_beyond_risk_window PASSED [ 31%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_06_earnings_already_completed PASSED [ 37%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_07_negative_days_to_earnings_remediation PASSED [ 43%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_08_missing_earnings_timestamp PASSED [ 50%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_09_provider_timeout PASSED [ 56%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_10_stale_provider_data PASSED [ 62%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_11_incorrect_ticker_or_exchange PASSED [ 68%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_12_daylight_saving_and_timezone_boundaries PASSED [ 75%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_13_missing_coverage_never_producing_pass PASSED [ 81%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_14_confluence_score_unchanged_for_unrelated_assets PASSED [ 87%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_15_preflight_state_consistency_across_api_and_frontend PASSED [ 93%]
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_16_news_sentiment_unavailable_while_earnings_coverage_valid PASSED [100%]

============================= 16 passed in 1.27s ==============================
```

### B. Frontend Vitest Unit & State Machine Suite
```text
vitest run
============================= vitest session starts =============================
 RUN  v2.1.8 C:/Users/akara/Documents/Projects/finance-earnings-audit/frontend

 ✓ lib/telemetry/__tests__/etfIntentClient.test.ts (16 tests)
 ✓ components/command-strip/__tests__/TickerCommandStrip.test.tsx (17 tests)
 ✓ components/__tests__/PriceChartVisibility.test.tsx (16 tests)
 ✓ components/nav/__tests__/MarketCommandRibbon.test.tsx (8 tests)
 ✓ components/__tests__/DecisionSurfaceIntegrity.test.tsx (12 tests)
 ✓ components/__tests__/EtfRiskProfileCard.test.tsx (5 tests)
 ✓ components/__tests__/ResponsiveChartPriorityRemediation.test.tsx (10 tests)
 ✓ components/drawers/__tests__/WatchlistDrawer.test.tsx (13 tests)
 ✓ components/__tests__/Wave2FirstViewportComposition.test.tsx (9 tests)
 ✓ components/__tests__/Wave1DeclutterPreservation.test.tsx (12 tests)
 ✓ components/__tests__/CockpitEtfRouting.test.tsx (10 tests)
 ✓ components/__tests__/AnalysisDecisionHierarchy.test.tsx (23 tests)
 ✓ components/__tests__/ArxCockpitUxRefinement.test.tsx (9 tests)
 ✓ components/__tests__/OnboardingTourPersistence.test.tsx (9 tests)
 ✓ components/__tests__/MobileNavbarOverflowMenu.test.tsx (14 tests)
 ✓ lib/telemetry/__tests__/etfDenominatorEngine.test.ts (13 tests)
 ✓ lib/__tests__/assetRegistry.test.ts (5 tests)
 ✓ state/__tests__/experience-store.test.ts (4 tests)
 ✓ components/__tests__/assetTypeUtils.test.ts (7 tests)
 ✓ lib/__tests__/apiRouting.test.ts (6 tests)
 ✓ hooks/__tests__/useExperienceMode.test.ts (6 tests)
 ✓ components/__tests__/DecisionReadinessCard.test.tsx (4 tests)
 ✓ components/__tests__/PreFlightCatalystRisk.test.tsx (5 tests)
 ✓ components/experience/__tests__/ExperienceModeToggle.test.tsx (5 tests)
 ✓ components/__tests__/EtfCostOfOwnershipCard.test.tsx (9 tests)
 ✓ components/__tests__/RadarPortfolioBadge.test.tsx (4 tests)
 ✓ components/workstation/__tests__/WorkstationCanvas.test.tsx (3 tests)
 ✓ hooks/__tests__/usePortfolioContext.test.ts (4 tests)

 Test Files  28 passed (28)
      Tests  258 passed (258)
   Duration  7.74s
```

### C. Core Confluence & Domain Invariant Test Suites
```text
pytest tests/test_confluence_engine.py tests/test_qa_escape_invariants.py tests/test_recommendation_consistency.py -v
============================= 48 passed in 12.04s =============================
- test_confluence_engine.py (5/5 PASSED)
- test_qa_escape_invariants.py (18/18 PASSED)
- test_recommendation_consistency.py (25/25 PASSED)
```

### D. TypeScript Static Typecheck
```text
tsc --noEmit
Exit Code: 0 (Zero diagnostic errors across entire frontend codebase)
```
