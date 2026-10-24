# ARX TERMINAL — SPRINT 3 / CORPORATE EVENTS RC5 SEMANTIC COMPATIBILITY AUDIT

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Reference:** `RC5_GATE_B_ISOLATED_INTEGRATION_AUTHORIZED`  
**Integration Commit SHA:** `c060700b2b3eedde2ec82800462b6c97a67c2939`  
**Timestamp:** 2026-10-23T10:25:50+02:00  

---

## 1. Domain Semantic Audit Matrix

The combined codebase was audited against the semantic specifications defined in Section 5 of the Gate B mandate.

| Category | Requirement | Audited Code / File | Status | Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Earnings Proximity** | Precise day delta calculation: `(event_dt - as_of).total_seconds() / 86400.0` | `analyst_dashboard/corporate_events.py` | **PASS** | Unit test 01: AXP same-day earnings correctly returns 0.0-1.0 days. |
| **Timing Distinction** | Release time (e.g. BMO/AMC) vs Conference Call time (e.g. 08:30 ET) explicitly differentiated | `corporate_events.py` | **PASS** | Unit test 02: Results release timestamp distinct from presentation call. |
| **Timezone Alignment** | New York local market time (`America/New_York`) normalization | `corporate_events.py` (`NY_TZ`) | **PASS** | Unit test 12: DST boundaries and UTC conversion verified. |
| **Missing Data Handling** | Absent earnings coverage yields `UNAVAILABLE` without false confidence | `PreFlightChecklistModal.tsx` & `corporate_events.py` | **PASS** | Unit test 08, 13: Absent data produces `status: UNAVAILABLE`, never `PASS`. |
| **Pre-Flight Hazard** | Imminent earnings inside 7-day window classified as `HAZARD` | `PreFlightChecklistModal.tsx` | **PASS** | Frontend Vitest test 01 & backend test 01, 04: Renders `HAZARD` indicator. |
| **Elapsed Event Remediation** | Past earnings events (`days < 0`) apply 0.0 penalty (no historical leakage) | `analyst_dashboard/analyzers/confluence_engine.py` | **PASS** | Unit test 06, 07: Negative days to earnings yield 0 penalty points. |
| **Confluence Score Bounds** | Penalties clamped [0, 100]; zero-day (-25 pts), 7-day (-15 pts) | `confluence_engine.py` | **PASS** | Unit test 01, 04, 05: Proper score deduction and bounds clamping verified. |
| **API Response Contract** | `corporateEvent` optional payload preserves backward compatibility | `api/routes/analytics.py` & `frontend/lib/api.ts` | **PASS** | Staging smoke & API contract tests pass; existing clients parse cleanly. |
| **Presentation Truthfulness** | Distinguishes Confirmed vs Estimated vs Elapsed without retail alarmism | `OptimalEntryExitCard.tsx` & `PreFlightChecklistModal.tsx` | **PASS** | Frontend Vitest: Institutional wording used; zero retail scare strings. |

---

## 2. In-Depth Boundary Analysis

### 2.1 Confluence Engine Clamping & Leakage Guard
In `analyst_dashboard/analyzers/confluence_engine.py`:
```python
# RC5 Earnings Proximity Penalty Boundary
if days_to_earnings is not None:
    if 0.0 <= days_to_earnings <= 1.0:
        penalty = 25.0
    elif 1.0 < days_to_earnings <= 7.0:
        penalty = 15.0
    else:
        penalty = 0.0  # past events (< 0) or far future (> 7) have zero penalty
```
- **Boundary 0.0 Days:** Handled in top bracket (-25 points).
- **Boundary 1.0 Day:** Handled in top bracket (-25 points).
- **Boundary 7.0 Days:** Handled in secondary bracket (-15 points).
- **Negative Days (< 0.0):** Evaluates strictly to `penalty = 0.0`. Eliminates the historical regression where past events permanently depressed scores.
- **Score Composition:** Clamped with `max(0.0, min(100.0, raw_score))`.

### 2.2 Pre-Flight Checklist Modal Semantics
In `frontend/components/PreFlightChecklistModal.tsx`:
- **Imminent Event (<= 7 days):**
  - Status: `HAZARD`
  - Color / Badge: Amber warning
  - Explanatory copy: High binary event risk; volatility expansion expected.
- **Missing Event Data:**
  - Status: `UNAVAILABLE`
  - Color / Badge: Neutral gray
  - Explanatory copy: No verified schedule found; exercise caution.
  - Invariant: NEVER renders `PASS` or green assurance.
- **Clean Runway (> 7 days):**
  - Status: `PASS`
  - Explanatory copy: Next scheduled results outside immediate tactical window.

### 2.3 Provider Attribution & Pipeline Honesty
- Confirmed schedules are attributed to primary filings/company IR.
- No claim is made in code, UI, or docs that live SEC EDGAR 8-K streaming or Polygon news feeds are currently active. They remain documented as future roadmap additions in `ARX_POLYGON_NEWS_PIPELINE_INVENTORY.md`.
