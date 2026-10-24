# ARX Terminal — American Express (NYSE: AXP) Earnings & Event Coverage Audit

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Track:** Track B (Isolated Research & Audit)  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-earnings-audit`  
**Branch:** `audit/axp-earnings-news-coverage`  
**Base Commit SHA:** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9` (Sprint 2 RC3 Baseline)  
**Execution Mode:** READ-ONLY INVESTIGATION + ISOLATED TESTING  
**Production Mutation Authorization:** NOT GRANTED  
**Final Disposition:** `AXP_EARNINGS_GAP_ROOT_CAUSE_CONFIRMED`  

---

## 1. Executive Summary & Core Finding

On **October 23, 2026**, American Express Company (NYSE: AXP) is scheduled to release its Q3 2026 financial results at **7:00 AM Eastern**, followed by an investor conference call at **8:30 AM Eastern**. 

Despite this major imminent binary catalyst, inspecting ARX Terminal's Stock Analysis and Pre-Flight Checklist reveals:
1. **Pre-Flight Check 4 Erroneously Passes:** Check 4 (*"News Event Safety / Catalyst Hazard Buffer"*) displays a bright green **`PASS`** with the reassuring description: *"Sufficient time window to manage trade without overnight earnings gap risk."*
2. **Stock Analysis Catalyst Card Omits the Event:** The Catalyst Card displays a generic placeholder for AXP (*"Clinical Trials, Pipeline Catalysts & 5-Year Earnings Model"*) with an uncataloged sector label, an empty milestone timeline, and zero mention of today's Q3 earnings report.
3. **Trading Confluence Score Ignores Binary Risk:** The Confluence Engine applies a **`0.0`** catalyst risk deduction because `days_to_earnings` is `None`, omitting the mandatory -25 point penalty for imminent binary earnings gap risk.

### Root Cause Summary:
* **Root Cause 1 (Zero News / Calendar Ingestion):** ARX Terminal possesses **zero code calling Polygon's `/v2/reference/news` endpoint**, zero news polling daemons, zero news cache, and zero corporate events ingestion pipelines.
* **Root Cause 2 (Polygon API Capability Boundary):** Standard Polygon.io REST API plans **do not provide an upcoming earnings calendar endpoint**. Polygon provides historical statements (`/stocks/financials/v1/*`), dividends, and splits, but no forward earnings calendar feed.
* **Root Cause 3 (Static Knowledge Base Silo):** `CatalystEngine` (`analyst_dashboard/analyzers/catalysts.py`) relies on a hardcoded static dictionary of 14 curated tickers (which excludes AXP). For all uncataloged tickers, it generates empty milestones and never calculates `days_to_earnings`. In live mode, it even purges milestones for curated assets.
* **Root Cause 4 (Data Discard in Analytics Route):** `api/routes/analytics.py` already instantiates `ticker_obj = yf.Ticker("AXP")` and retrieves `ticker_obj.info`, which **already contains `earningsTimestamp: 1792758600` (2026-10-23 08:30:00 EDT) and `isEarningsDateEstimate: False`**. However, the route never parses or forwards these fields to `catalyst_report`.
* **Root Cause 5 (Frontend Contract Disconnect):** `OptimalEntryExitCard.tsx` looks for `(executionPlan as any)?.catalystForecast?.catalysts` (which does not exist in the backend schema). This causes `derivedHasImminentEarnings` to unconditionally evaluate to `false`.
* **Root Cause 6 (Dangerous Invariant Violation in Pre-Flight):** In `PreFlightChecklistModal.tsx`, line 163 defines `const isCatalystPassed = isPriceValid && !hasImminentEarnings;`. Because `hasImminentEarnings` is `false`, **missing data is equated with a clean catalyst runway**. Check 4 falsely passes instead of reporting `UNAVAILABLE` or `UNVERIFIED`.
* **Root Cause 7 (Confluence Lower-Bound Defect):** `confluence_engine.py` line 228 checks `if days_to_earnings <= 1.0:`, lacking a `0.0 <= days_to_earnings` lower bound. Any past/elapsed earnings (negative days) erroneously triggers a permanent -25 point binary gap penalty.

---

## 2. Reference Event & Primary-Source Verification

The reference event details were verified against primary sources:

| Event Attribute | Reference Specification | Authoritative Primary Source Evidence | Verification Status |
| :--- | :--- | :--- | :--- |
| **Company Name** | American Express Company | SEC EDGAR Form 10-K / CIK `0000004962` | **VERIFIED** |
| **Symbol / Ticker** | `AXP` | NYSE Listed Directory (`other_listed_directory.txt`: `AXP\|American Express Company Common Stock\|N\|AXP`) | **VERIFIED** |
| **Exchange** | NYSE (New York Stock Exchange) | SEC Company Tickers Exchange Feed: `[4962, 'AMERICAN EXPRESS CO', 'AXP', 'NYSE']` | **VERIFIED** |
| **Event Date** | October 23, 2026 | American Express Investor Relations (`ir.americanexpress.com`) | **VERIFIED** |
| **Scheduled Results** | ~7:00 AM Eastern (11:00 UTC) | Official IR Earnings Release Schedule: Results and presentation posted to IR site at 7:00 AM ET | **VERIFIED** |
| **Conference Call** | 8:30 AM Eastern (12:30 UTC) | Official IR Webcast Schedule: Live audio webcast at 8:30 AM ET | **VERIFIED** |
| **Confirmation Status** | Confirmed (Not Estimated) | `isEarningsDateEstimate: False`, `earningsTimestamp: 1792758600` | **VERIFIED** |

---

## 3. Canonical Security Master Resolution for AXP

Running `sm.get_or_resolve_instrument('AXP')` against ARX Terminal's canonical Security Master confirms:

```text
symbol:                     'AXP'
provider_symbol:            'AXP'
asset_class:                'EQUITY'
security_type:              'COMMON_STOCK'
primary_exchange:           'NYSE' (via SEC / Nasdaq Trader Directory)
classification_status:      'VERIFIED' (via OpenFIGI mapping)
execution_eligibility:      'FAIL_CLOSED' (until live data feed is bound)
stable_identifier (FIGI):   'BBG000BCQZS4'
composite_figi:             'BBG000BCQZS4'
share_class_figi:           'BBG001S5P034'
issuer_name:                'AMERICAN EXPRESS CO'
```

**Security Master Finding:** Instrument identity resolution for AXP is 100% sound. The failure to display the earnings event is not caused by symbol mismatch, composite FIGI failure, or exchange misattribution.

---

## 4. Empirical Polygon News API Audit for AXP

To determine whether Polygon's API returns relevant articles, a bounded, authenticated, read-only probe was executed against `https://api.polygon.io/v2/reference/news?ticker=AXP&limit=10`:

```text
HTTP Status: 200 OK
Total Results Returned: 10
Articles Returned:
  [1] 2026-10-21T22:50:04Z | Zacks Investment Research
      Title: Triumph Financial (TFIN) Q3 Earnings Miss Estimates
      AXP Sentiment: positive (Expected to report strong Q3 2026 results with earnings growth of +8% YoY...)
  [2] 2026-10-21T13:00:12Z | Zacks Investment Research
      Title: Houlihan Lokey (HLI) Expected to Beat Earnings Estimates...
      AXP Sentiment: neutral (Negative Earnings ESP of -0.17% combined with Zacks Rank #3...)
  [8] 2026-10-16T14:50:00Z | Zacks Investment Research
      Title: Will Rising Expenses Affect American Express' Q3 Earnings?
  [9] 2026-10-16T13:00:05Z | Zacks Investment Research
      Title: American Express (AXP) Earnings Expected to Grow: Should You Buy?
```

### Taxonomy Classification:
* **Condition 1:** Provider has relevant news? **YES.** (Zacks and Motley Fool articles discuss AXP Q3 earnings schedule and expectations).
* **Condition 2:** Provider has no relevant news? **NO.**
* **Condition 3:** Provider access restricted? **NO.** (HTTP 200 OK with valid 32-character key).
* **Condition 4:** ARX does not request the endpoint? **YES. ARX has zero code querying `/v2/reference/news`.**
* **Condition 5:** ARX requests but discards? **NO.**
* **Condition 6:** ARX stores but does not render? **NO.**
* **Condition 7:** ARX maps to wrong instrument? **NO.**
* **Condition 8:** Data is stale or cached incorrectly? **PARTIALLY.** (Underlying `info` object contains the date, but it is discarded before caching in `market_db.save_catalyst`).

---

## 5. End-to-End Trace of the Pre-Flight Failure

```
[api/routes/analytics.py]
  │
  ├── 1. Calls ticker_obj.info
  │      └── Contains earningsTimestamp: 1792758600 (2026-10-23 08:30:00 EDT)
  │      └── Contains isEarningsDateEstimate: False
  │      └── (DEFECT: Analytics route DISCARDS these fields!)
  │
  ├── 2. Calls catalyst_engine.get_asset_catalyst_report("AXP")
  │      └── AXP is not in static dict ASSET_CATALYST_KNOWLEDGE
  │      └── Returns fallback dict: upcoming_milestones = [], days_to_earnings = None
  │
  ├── 3. Calls confluence_engine.calculate_confluence(..., catalyst_data)
  │      └── raw_dte = catalyst_data.get("days_to_earnings") is None
  │      └── catalyst_mod = 0.0 (Bypasses -25 point penalty!)
  │
  └── 4. Sends HTTP payload to frontend:
         "catalystForecast": {
             "symbol": "AXP",
             "primary_drug_trial": "Net Interest Income & Credit Quality (Awaiting Disclosures)",
             "trial_phase": "Data Unavailable",
             "upcoming_milestones": [],
             "multi_year_forecast": []
         }

[frontend/components/OptimalEntryExitCard.tsx]
  │
  └── Line 276: const catalysts = (executionPlan as any)?.catalystForecast?.catalysts;
      └── catalysts is undefined (key does not exist!)
      └── derivedHasImminentEarnings = false

[frontend/components/PreFlightChecklistModal.tsx]
  │
  └── Line 163: const isCatalystPassed = isPriceValid && !hasImminentEarnings;
      └── hasImminentEarnings is false
      └── isCatalystPassed = true!
      └── Renders: "✅ 4. News Event Safety: PASS (Sufficient time window to manage trade)"
```

---

## 6. Pre-Flight and Decision Quality Hazard

This failure pattern creates a severe **false-safety trap** for traders:
1. A retail or institutional trader reviewing AXP in ARX Terminal on October 23, 2026 sees an active trade plan (optimal entry, stop loss, profit target).
2. The trader clicks **"Pre-Flight Checklist"** to verify event risk before entering the trade.
3. Check 4 explicitly tells them: **`PASS: Clean window (>7 days to binary catalyst)`** and **`No surprise earnings report tomorrow`**.
4. In reality, American Express releases results at 7:00 AM and hosts its earnings call at 8:30 AM, exposing the trader's position to immediate, unmodeled overnight gap risk that can blow through standard stop-loss orders.

---

## 7. Audit Conclusion & Next Action

The root cause of the missing AXP earnings visibility is **100% confirmed across all backend, engine, and frontend layers**. 

Remediation requires implementing the source-backed corporate-event contract defined in [ARX_CORPORATE_EVENTS_MINIMUM_REMEDIATION_PLAN.md](file:///c:/Users/akara/Documents/Projects/finance-earnings-audit/ARX_CORPORATE_EVENTS_MINIMUM_REMEDIATION_PLAN.md) and wiring the already available `yfinance` earnings timestamps into `api/routes/analytics.py` and `PreFlightChecklistModal.tsx`.
