# ARX TERMINAL — RC5 SEMANTIC REGRESSION EVIDENCE

**Date:** 2026-10-23T11:51:00Z (Local: 2026-10-23T13:51:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_MUTATION_AUTHORIZED`  
**Target Modules:** `analyst_dashboard/analyzers/confluence_engine.py`, `analyst_dashboard/corporate_events.py`  

---

## 1. Verified Invariant Coverage

| Semantic Invariant / Feature | Test Identifier / Suite | Verified Behavior | Exit Code |
|---|---|---|---|
| **Timestamp Normalization** | `test_02_axp_earnings_call_vs_results_release_distinction` | Distinguishes earnings results release from conference call time | 0 |
| **Confirmed vs Estimated Dates** | `test_03_confirmed_vs_estimated_dates_distinguished` | Confirmed announcements receive priority over algorithmic estimates | 0 |
| **Event Proximity Boundaries** | `test_boundary_exactly_zero_days`, `test_boundary_exactly_one_day` | $\le 1$ day applies acute hazard penalty ($-25.0$ pts) | 0 |
| **Cautionary Runway Boundary** | `test_boundary_exactly_seven_days`, `test_05_earnings_beyond_risk_window` | $2 \le \text{days} \le 7$ applies cautionary penalty ($-8.0$ pts); $>7$ days zero penalty | 0 |
| **Elapsed & Completed Events** | `test_boundary_stale_event`, `test_06_earnings_already_completed` | Completed / elapsed earnings events incur zero permanent confluence penalty | 0 |
| **Negative Guidance Integration** | `test_01_axp_earnings_scheduled_today_applies_maximum_gap_penalty` | Negative guidance suppresses execution readiness and position sizing | 0 |
| **Score Clamping Invariant** | `test_contract_score_bounding_and_schema` | Confluence score strictly clamped to $[0.0, 100.0]$ under all penalties | 0 |
| **Zero-Drift on Missing Events** | `test_14_confluence_score_unchanged_for_unrelated_assets` | Uncataloged tickers and missing events produce exactly $0.0$ modifier (bit-for-bit identical to baseline) | 0 |
| **Portfolio Independence** | `test_radarPortfolioContext.test.ts` | Corporate event processing has zero dependence on or mutation of portfolio holdings | 0 |
| **Scanner Independence** | `test_radarTaxonomy.test.ts` | Scanners operate independently; `PIPELINE_PENDING` preserved without zero-result collapse | 0 |

---

## 2. Test Execution Log Summary

### A. Corporate Events Boundaries (`tests/test_corporate_events_boundaries.py`)
- **Command:** `python -m pytest tests/test_corporate_events_boundaries.py -v`
- **Results:** 10 passed, 0 failed in 0.54s (Exit Code: 0).

### B. Corporate Events & Earnings Audit (`tests/test_corporate_events_earnings_audit.py`)
- **Command:** `python -m pytest tests/test_corporate_events_earnings_audit.py -v`
- **Results:** 16 passed, 0 failed in 1.22s (Exit Code: 0).

### C. Frontend Architectural Tests (`frontend`)
- **Command:** `npm.cmd run test:arch`
- **Results:** 29 suites passed, 0 failed (Exit Code: 0).
