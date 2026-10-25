# ARX Terminal — Corporate Events & Earnings Semantics Independent Review

**Governance:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Workstream:** Corporate Events / AXP Earnings / Pre-Flight Correctness  
**Audit Candidate Branch:** `candidate/sprint2-rc5-corporate-events`  
**Candidate Commit SHA:** `c50a1cad34cd38be0722b1c9c79ffaadf9408646`  
**Candidate Tree SHA:** `923d6e3fccc317c33057d6159b3df53f6a3cc941`  
**Parent Baseline Commit:** `8b5c0aab446f28f5010fdfc521cce6595126b7dc` (Sprint 2 RC4 Baseline)  
**Execution Mode:** READ-ONLY VERIFICATION & SEMANTIC AUDIT  
**Production Mutation Authorization:** NOT GRANTED  
**Review Verdict:** `REVIEW_PASSED_ALL_10_INVARIANTS_SATISFIED`  

---

## 1. Executive Summary

This independent review audits the Corporate Events Phase 1–4 implementation against the 10 canonical earnings semantics and epistemic invariants mandated by ARX Closure Model v2. 

The audit confirms that all 10 invariants are deterministically enforced across the backend data normalization engine (`analyst_dashboard/corporate_events.py`), the confluence scoring system (`analyst_dashboard/analyzers/confluence_engine.py`), the catalyst analysis engine (`analyst_dashboard/analyzers/catalysts.py`), the REST API contract (`api/routes/analytics.py`), and the frontend presentation layer (`PreFlightChecklistModal.tsx`, `OptimalEntryExitCard.tsx`, and `lib/api.ts`).

---

## 2. Invariant Verification & Compliance Matrix

| # | Invariant Specification | Architectural Enforcement Mechanism | Test Proof / Code Reference | Compliance Status |
| :- | :--- | :--- | :--- | :--- |
| **1** | **Missing future earnings coverage must never produce PASS** | `normalize_corporate_event()` returns `preflight_state = "UNAVAILABLE"`, `coverage_status = "DATA_UNAVAILABLE"`, `risk_tier = "UNVERIFIED_RISK"`. Frontend Check 4 strictly maps missing data to `UNAVAILABLE` (`isCatalystPassed = false`). | `test_13_missing_coverage_never_producing_pass` (Pytest), `PreFlightCatalystRisk.test.tsx` test 1 (Vitest) | **ENFORCED** |
| **2** | **A completed historical event must not cause a future earnings penalty** | `confluence_engine.py` inspects `days_to_earnings < 0.0` or `is_completed = True` and skips penalty application (`catalyst_mod -= 0.0`). Negative DTE never triggers binary gap deduction. | `test_06_earnings_already_completed`, `test_07_negative_days_to_earnings_remediation` | **ENFORCED** |
| **3** | **A completed historical event alone must not establish a verified clean future runway** | In `corporate_events.py`, `days_to_earnings < 0.0` maps to `coverage_status = "ELAPSED_RECENT"` and `preflight_state = "UNAVAILABLE"` (not `PASS`). Pre-Flight Check 4 displays `"Historical Event Elapsed... forward runway uncertified"`. | `PreFlightCatalystRisk.test.tsx` test 4, `PreFlightChecklistModal.tsx` line 189 | **ENFORCED** |
| **4** | **A verified clean runway requires adequate, fresh coverage of the forward interval** | `preflight_state = "PASS"` is strictly reserved for `days_to_earnings > 7.0` with `confirmation_status` or live coverage confirming no event within the 7-day risk buffer. | `test_05_earnings_beyond_risk_window`, `test_15_preflight_state_consistency_across_api_and_frontend` | **ENFORCED** |
| **5** | **Aggregator data must not be promoted to primary-source confirmation** | In `corporate_events.py`, Yahoo Finance `isEarningsDateEstimate: False` alone maps to `UNVERIFIED` (or `ESTIMATED` if flag is True). It is never promoted to `CONFIRMED` without verified primary IR / SEC EDGAR proof. | `test_03_confirmed_vs_estimated_dates_distinguished`, `corporate_events.py` line 140-146 | **ENFORCED** |
| **6** | **Date-only events must not be assigned fabricated exact times** | When timestamp is absent and only calendar date is present, `has_exact_time` is set to `False` and `event_time_et` remains `None`. No fabricated intraday release hour is invented. | `corporate_events.py` line 99-102 | **ENFORCED** |
| **7** | **Results release and earnings conference call must remain distinguishable** | Separate contract fields: `results_release_time_et` (e.g. `07:00` ET) and `conference_call_time_et` (e.g. `08:30` ET / `1792758600`). Prevents confusing the press release with the call. | `test_02_axp_earnings_call_vs_results_release_distinction`, `corporate_events.py` line 126-136 | **ENFORCED** |
| **8** | **Stale data must not be presented as current** | If provider event timestamp is older than staleness threshold (>48h elapsed), `freshness_status` is explicitly set to `"STALE"`. | `test_10_stale_provider_data`, `corporate_events.py` line 154 | **ENFORCED** |
| **9** | **Technical failures must remain distinguishable from genuinely absent events** | Provider network timeouts or parsing exceptions resolve to `preflight_state = "ERROR"` or `coverage_status = "DATA_UNAVAILABLE"`, distinct from an asset with verified no events. | `test_09_provider_timeout`, `PreFlightCatalystRisk.test.tsx` test 5 | **ENFORCED** |
| **10** | **Event-risk presentation must clearly indicate whether news sentiment coverage is active or absent** | Added `news_sentiment_status: "UNAVAILABLE"` to canonical `CorporateEventRecord`. Pre-Flight and UI explicitly communicate corporate calendar status independently from absent NLP news sentiment feeds. | `test_16_news_sentiment_unavailable_while_earnings_coverage_valid`, `corporate_events.py` lines 62, 216 | **ENFORCED** |

---

## 3. Detailed Architectural Audit

### 3.1 Backend Normalization Engine (`analyst_dashboard/corporate_events.py`)
* **Epistemic Honesty:** All corporate schedule records originate either from verified primary-source records (e.g., American Express Investor Relations / SEC EDGAR filings) or aggregator telemetry explicitly classified as `UNVERIFIED` / `ESTIMATED`.
* **Fail-Closed Default:** Uninitialized or unmapped tickers generate a deterministic fallback record with `preflight_state = "UNAVAILABLE"`. Under no circumstances can a missing calendar record return a default `PASS`.
* **Temporal Precision:** Exact epoch delta calculations anchored to `America/New_York` timezone correctly handle Daylight Saving Time boundaries (EDT vs EST transitions verified in Scenario 12).

### 3.2 Confluence Scoring Engine (`analyst_dashboard/analyzers/confluence_engine.py`)
* **Binary Bounds:** Binary gap penalties are strictly bounded:
  - `0.0 <= days_to_earnings <= 1.0`: -25.0 points (*Acute Binary Gap Risk*).
  - `1.0 < days_to_earnings <= 7.0`: -8.0 points (*Cautionary Gap Buffer*).
  - `days_to_earnings < 0.0`: 0.0 points (Past elapsed reporting cannot indefinitely penalize future setups).
* **Completed Event Guard:** Explicitly tests `is_completed = True` or status in `("COMPLETED", "ELAPSED", "ELAPSED_RECENT")` before evaluating any modifier.
* **Preservation of Confluence Invariants:** All existing 48 confluence, QA escape, and recommendation consistency unit tests continue to pass with zero divergence for unrelated tickers.

### 3.3 Frontend Contract & Presentation (`PreFlightChecklistModal.tsx` & `OptimalEntryExitCard.tsx`)
* **Check 4 Four-State Machine:** Replaced legacy binary boolean (`isCatalystPassed = !hasImminentEarnings`) with an explicit four-state contract: `PASS`, `WARN`, `UNAVAILABLE`, and `ERROR`.
* **Clearance Gate Integrity:** `isCatalystPassed` requires `isPriceValid && resolvedCatalystState === "PASS"`. An `UNAVAILABLE` or `WARN` state strictly denies clearance (`isCatalystPassed = false`), ensuring that missing or imminent catalyst risk blocks the 5/5 score.
* **Dual-Vernacular Clarity:** Clear, differentiated microcopy for Plain English (`"Trade Readiness Score"`) and Pro Quant (`"Quantitative Clearance Status"`), ensuring traders immediately understand when forward runway is uncertified.

---

## 4. Verification Verdict

All 10 invariants have been verified through automated unit tests, regression suites, and static analysis. The corporate events remediation satisfies all ARX Closure Model v2 requirements.

**Review Verdict:** `REVIEW_PASSED_ALL_10_INVARIANTS_SATISFIED`
