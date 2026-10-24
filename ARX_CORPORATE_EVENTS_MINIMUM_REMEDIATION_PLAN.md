# ARX Terminal — Corporate Events & Earnings Minimum Remediation Plan

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Track:** Track B (Isolated Research & Audit)  
**Operating Worktree:** `C:\Users\akara\Documents\Projects\finance-earnings-audit`  
**Branch:** `audit/axp-earnings-news-coverage`  
**Base Commit SHA:** `385d9c6faa9986aa3460c6f2943bdd1667fc80d9` (Sprint 2 RC3 Baseline)  
**Execution Mode:** SPECIFICATION & TEST-VERIFIED ARCHITECTURAL DESIGN  
**Production Mutation Authorization:** NOT GRANTED  

---

## 1. Executive Remediation Strategy

To resolve the AXP earnings visibility defect and permanently eliminate the false-safety pass in Pre-Flight, ARX Terminal must implement a **Source-Backed Corporate Events Contract**.

### Zero-Cost Architecture Principle:
We do **not** need to license a new paid third-party earnings calendar API. 
The backend route `api/routes/analytics.py` already calls `ticker_obj = yf.Ticker(fetch_sym)` and retrieves `ticker_obj.info` and `ticker_obj.calendar`. These data structures **already provide verified earnings timestamps (`earningsTimestamp`), call times, and confirmation status (`isEarningsDateEstimate: False`)**. 

The remediation consists of **extracting, normalizing, passing, and truthfully rendering** this existing data through the backend and frontend pipelines.

---

## 2. Canonical Source-Backed Corporate Event Contract

```typescript
export type CorporateEventType = 
  | "EARNINGS_ANNOUNCEMENT"
  | "CONFERENCE_CALL"
  | "DIVIDEND_EX_DATE"
  | "STOCK_SPLIT"
  | "SHAREHOLDER_MEETING";

export type EventCoverageStatus = 
  | "CONFIRMED_UPCOMING"     // Official company announcement verified
  | "ESTIMATED_UPCOMING"     // Historical cadence estimate (unconfirmed)
  | "ELAPSED_RECENT"         // Reported within last 48 hours
  | "DATA_UNAVAILABLE"       // Provider offline or uncataloged
  | "UNVERIFIED";            // Missing authoritative confirmation

export type EventRiskTier = 
  | "ACUTE_BINARY_GAP_RISK"  // <= 24 hours to announcement (-25 pt penalty)
  | "HIGH_GAP_RISK"          // 24h - 48h to announcement (-15 pt penalty)
  | "CAUTION_BUFFER"         // 2 - 7 days to announcement (-8 pt penalty)
  | "NORMAL_RUNWAY"          // > 7 days (Clean window, 0 pt modifier)
  | "UNVERIFIED_RISK";       // Event schedule unknown (Fail-closed)

export interface CorporateEventRecord {
  symbol: string;
  provider_symbol: string;
  event_type: CorporateEventType;
  event_timestamp_utc: number;          // Epoch seconds (e.g. 1792758600)
  event_date_et: string;                // YYYY-MM-DD in America/New_York
  scheduled_call_time_et?: string;      // e.g. "08:30"
  time_zone: string;                    // "America/New_York"
  is_estimate: boolean;                 // False for confirmed IR dates
  days_to_event: number;                // Floating point days from now
  coverage_status: EventCoverageStatus;
  risk_tier: EventRiskTier;
  source_name: string;                  // e.g. "American Express Investor Relations / SEC"
  source_url?: string;
  last_verified_at: string;             // ISO 8601 UTC timestamp
  freshness_status: "LIVE" | "RECENT" | "STALE" | "UNVERIFIED";
  preflight_summary: {
    plain_english: string;
    pro_quant: string;
  };
}
```

---

## 3. Strict Semantics for Unavailable vs. Clean Data

> [!CAUTION]
> **Cardinal Pre-Flight Invariant:**
> **Never equate missing earnings information with no upcoming earnings.**
> 
> Under no circumstances may an asset with unverified earnings dates receive a green `PASS` ("Sufficient time window to manage trade"). 
> 
> When earnings data is absent or unverified, Pre-Flight Check 4 must render **`UNAVAILABLE`** or **`UNVERIFIED`** with an amber/slate badge, matching the defensive fail-closed architecture of Check 3 (Capital Risk) and Check 5 (Macro Regime).

### Pre-Flight Check 4 Decision Table:

| `days_to_earnings` | `coverage_status` | `is_estimate` | Pre-Flight Check 4 Badge | Check 4 Status Label | Confluence Score Penalty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\le 1.0$ day | `CONFIRMED_UPCOMING` | `false` | **`HAZARD`** (Rose) | `⚠️ Acute Binary Gap Risk: Earnings in <= 24h` | `-25.0` pts |
| $1.0 < \text{DTE} \le 2.0$ | `CONFIRMED_UPCOMING` | `false` | **`HAZARD`** (Rose) | `⚠️ High Gap Risk: Earnings in <= 48h` | `-15.0` pts |
| $2.0 < \text{DTE} < 7.0$ | `CONFIRMED_UPCOMING` | `false` / `true` | **`CAUTION`** (Amber) | `⚠️ Caution: Earnings within 7-day buffer` | `-8.0` pts |
| $\ge 7.0$ days | `CONFIRMED_UPCOMING` | `false` | **`PASS`** (Emerald) | `✅ Catalyst Runway: Clean window (>7 days)` | `0.0` pts |
| $\ge 7.0$ days | `ESTIMATED_UPCOMING` | `true` | **`PASS (EST)`** (Emerald) | `✅ Estimated Runway: >7 days (Unconfirmed Date)` | `0.0` pts |
| `None` / `null` | `DATA_UNAVAILABLE` | N/A | **`UNAVAILABLE`** (Slate) | `Corporate Event Data Unavailable: Cannot certify runway` | `0.0` pts (Preserves defensive 50 cap) |
| $< 0.0$ (Elapsed) | `ELAPSED_RECENT` | N/A | **`PASS`** (Emerald) | `✅ Catalyst Passed: Prior quarter reported` | `0.0` pts |

---

## 4. File-Level Remediation Scope

### 1. `analyst_dashboard/analyzers/confluence_engine.py` (Defect Fix)
* **Location:** Lines 224–235.
* **Bug:** `if days_to_earnings <= 1.0:` lacks a lower bound, penalizing past earnings forever.
* **Fix:**
  ```python
  if raw_dte is not None:
      days_to_earnings = float(raw_dte)
      if 0.0 <= days_to_earnings <= 1.0:
          catalyst_mod -= 25.0
          hours_left = max(1, int(days_to_earnings * 24))
          warnings.append(f"⚠️ HIGH BINARY GAP RISK: Earnings in <={hours_left}h! Limit position sizing.")
      elif 1.0 < days_to_earnings <= 2.0:
          catalyst_mod -= 15.0
          warnings.append(f"⚠️ High Gap Risk: Earnings in {days_to_earnings:.1f} days.")
      elif 2.0 < days_to_earnings < 7.0:
          catalyst_mod -= 8.0
          warnings.append(f"Caution: Earnings in {days_to_earnings:.1f} days.")
  ```

### 2. `api/routes/analytics.py` (Data Extraction & Normalization)
* **Location:** Lines 1000–1020 & 1114–1130.
* **Fix:** Extract earnings timestamps from `info` or `ticker_obj.calendar`:
  ```python
  # Extract earnings schedule from yfinance ticker object
  earnings_ts = info.get("earningsTimestamp") or info.get("earningsCallTimestampStart")
  is_estimate = bool(info.get("isEarningsDateEstimate", True))
  days_to_earnings = None
  event_date_et = None
  call_time_et = None

  if earnings_ts and earnings_ts > 0:
      event_dt_utc = datetime.fromtimestamp(earnings_ts, timezone.utc)
      event_dt_et = event_dt_utc.astimezone(zoneinfo.ZoneInfo("America/New_York"))
      now_utc = datetime.now(timezone.utc)
      days_to_earnings = round((event_dt_utc - now_utc).total_seconds() / 86400.0, 2)
      event_date_et = event_dt_et.strftime("%Y-%m-%d")
      call_time_et = event_dt_et.strftime("%H:%M")

  catalyst_report["days_to_earnings"] = days_to_earnings
  catalyst_report["event_date_et"] = event_date_et
  catalyst_report["scheduled_call_time_et"] = call_time_et
  catalyst_report["is_estimate"] = is_estimate
  catalyst_report["has_imminent_earnings"] = (days_to_earnings is not None and 0.0 <= days_to_earnings <= 7.0)
  ```

### 3. `frontend/components/OptimalEntryExitCard.tsx` (Contract Parity)
* **Location:** Lines 275–284.
* **Bug:** Reads non-existent `.catalysts` array.
* **Fix:**
  ```typescript
  const derivedHasImminentEarnings = Boolean(
    executionPlan?.catalystForecast?.has_imminent_earnings ??
    (executionPlan?.catalystForecast?.days_to_earnings != null &&
     executionPlan.catalystForecast.days_to_earnings >= 0 &&
     executionPlan.catalystForecast.days_to_earnings <= 7.0)
  );
  ```

### 4. `frontend/components/PreFlightChecklistModal.tsx` (Pre-Flight Integrity)
* **Location:** Lines 163, 406–427.
* **Bug:** Equates `!hasImminentEarnings` with `isCatalystPassed = true`.
* **Fix:** Introduce explicit `catalystEvidenceState`:
  ```typescript
  const hasEarningsData = catalystForecast?.days_to_earnings != null;
  const isCatalystPassed = isPriceValid && hasEarningsData && !hasImminentEarnings;
  const isCatalystHazard = isPriceValid && hasEarningsData && hasImminentEarnings;
  const isCatalystUnavailable = !hasEarningsData;
  ```
  Render `PASS` if verified clean, `HAZARD` if imminent, and `UNAVAILABLE` if data is missing.

---

## 5. Licensing and Rate-Limit Analysis

1. **Massive / Polygon News (`/v2/reference/news`):**
   - Free/Starter Tier allows 5 calls/min.
   - Enterprise/Paid Tiers allow unlimited calls.
   - Polling news for all universe assets in real-time is cost-ineffective and rate-limit hazardous.
   - Polygon news should only be queried on-demand when a user views Stock Analysis for a specific symbol, cached in SQLite with a 15-minute TTL.
2. **Calendar Ingestion (`yfinance` / SEC EDGAR):**
   - `yfinance` queries are already integrated into the analytics pipeline and cached via `INFO_CACHE` (TTL: 1 hour).
   - Zero additional API licensing cost or commercial subscription required.

---

## 6. Recommended Phased Implementation Sequence

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Engine Defect Correction                            │
│  - Fix confluence_engine.py lower-bound logic                │
│  - Verify with test_corporate_events_earnings_audit.py (PASS)│
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Analytics Route Data Enrichment                     │
│  - Extract earningsTimestamp from ticker_obj.info            │
│  - Compute days_to_earnings & format America/New_York call   │
│  - Populate catalyst_report and persist in SQLite            │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Frontend Contract Alignment                         │
│  - Update OptimalEntryExitCard.tsx to read DTE               │
│  - Update PreFlightChecklistModal.tsx for UNAVAILABLE state  │
│  - Update CatalystForecastCard.tsx to display earnings date  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase 4: Nonproduction Vitest & E2E Validation               │
│  - Verify AXP renders HAZARD in Pre-Flight                   │
│  - Verify KO / SBUX render clean PASS                        │
│  - Verify uncataloged asset renders UNAVAILABLE              │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Outstanding Evidence Gaps & Risks

1. **After-Hours Announcement Nuance:** Companies reporting after market close (e.g. 4:05 PM ET) have gap risk for the *next morning's open*. The `days_to_earnings` formula must account for market trading hours rather than pure calendar days.
2. **SEC Form 8-K Item 2.02 Ingestion:** To attain 100% institutional-grade primary source provenance, ARX should eventually ingest SEC EDGAR Form 8-K (Item 2.02: Results of Operations and Financial Condition) directly via SEC RSS feeds.
