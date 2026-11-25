# ARX PR #328 Release Qualification & Production Rollout Receipt

**Pull Request:** `#328` (`fix/mobile-chart-and-preflight-evidence`)  
**PR Title:** `fix(ui): mobile candlestick chart, preflight copy truth, SEC Form 4 labels, and TD-08`  
**Initial Qualified Revision:** `290029d35520a0046442da71c93ebb85cb20ef42` (Run `45525294192`)  
**Venue-Label Patch Revision:** `a63361777f039511b8e0823ec4456b807dbe8517` (Run `45584865222`)  
**Integrated Production Revision:** `ab0c9845617eff812115de7b3e5e6163f04d7f06` (Run `45596974737`)  
**Production Rollout Target:** Cloudflare Pages (`www.arxterminal.com`, `finance-xp8.pages.dev`)  
**Demonstrated Rollback Target:** Deployment `c34a7f69-eb54-4594-8cff-44e7625f9482` (SHA `a6336177`) / Deployment `8e9508bc` (SHA `290029d3`)  
**Final Production Verdict:** **`DEPLOYED_AND_VERIFIED`** (Frontend live on Cloudflare Pages; physical iOS hardware `NOT_CHECKED`; PR #328 remains open / unmerged on GitHub)

---

## 1. Executive Summary & Problem Resolution

PR #328 delivers four critical UX and data truthfulness corrections:

1. **Mobile Candlestick Chart (390x844 Viewport Usability):**
   - Resolves mobile viewport clipping, layout overflow, and unresponsive touch targets in the price candlestick chart component.
   - Standardizes container sizing, legend interaction, and activity drawer integration on 390x844 mobile viewports.
2. **Truthful Pre-Flight Checklist Copy:**
   - Replaces hyperbolic, synthetic warning copy (such as `"FLIGHT CLEARANCE REVOKED"` and `"hard-earned money"`) with objective structural condition statuses.
   - Preserves truthful presentation of available vs unavailable preflight checks without distorting overall confidence scoring.
3. **Truthful SEC Form 4 Transaction Labels (Zero Fabricated Venue Claims):**
   - Corrects transaction classification: Code `P` $\to$ `"Purchase"`, Code `S` $\to$ `"Sale"`, Code `G` $\to$ `"Gift"`, Code `A` $\to$ `"Grant / Award (Scope Unverified)"`, Code `F` $\to$ `"Tax Withholding"`, and unparsed filings $\to$ `"Form 4 Filing (Unparsed)"`.
   - Strictly prohibits rendering `"Open Market"` in UI transaction labels unless supported by explicit, retained SEC source disclosure evidence.
   - **NVDA Accession `0001696841-26-000016` Verification:** Verified live in production as rendering 12 `"Sale"` transactions, eliminating the previous unsupported `"Open Market Sale"` label.
4. **Resolution of Technical Debt TD-08 (Issue #204):**
   - Removed fabricated options expiration fallback date (`"2026-04-01"`) in `frontend/app/page.tsx:100`. Absent options chain expiration now truthfully falls back to neutral state.

---

## 2. Qualification Gates & Test Evidence

### A. Initial Candidate Qualification (`290029d3`)
- Hosted Run `45525294192` (`12/12` PASS).
- Verified fullstack API contract compatibility with deployed Web API.
- Verified mobile viewport rendering in Playwright automation.

### B. Venue-Label Regression Patch (`a6336177`)
- Hosted Run `45584865222` (`12/12` PASS).
- Surgical diff confirmed: strictly updates `tradeMarkers` classification logic to eliminate unsupported `"Open Market"` string, adds accession `0001696841-26-000016` regression test case, and binds release manifest `1ae7eaa7`.
- Unit tests: Passing with zero regression on dates, transaction types, or retained provenance.

### C. Integrated Production Qualification (`ab0c9845`)
- Hosted Run `45596974737` (`12/12` PASS).
- Integrated alongside PR #327 (Radar fail-closed states) and PR #325 (Cornish-Fisher observation bounds).

---

## 3. Production Deployment Record

| Deployment Scope | Cloudflare Pages Deployment ID | Deployed Commit SHA | Verification Status |
| :--- | :--- | :--- | :--- |
| **Initial PR #328 Rollout** | `8e9508bc-xxxx` | `290029d35520a0046442da71c93ebb85cb20ef42` | `DEPLOYED_AND_VERIFIED` (Venue defect open) |
| **Venue-Label Patch Rollout** | `c34a7f69-eb54-4594-8cff-44e7625f9482` | `a63361777f039511b8e0823ec4456b807dbe8517` | `DEPLOYED_AND_VERIFIED` (Venue defect resolved) |
| **Consolidated Integrated Rollout** | `6c0d391e-176d-4eca-9aec-f5eef6dd69c5` | `ab0c9845617eff812115de7b3e5e6163f04d7f06` | `DEPLOYED_AND_VERIFIED` (Active Production) |

---

## 4. Live Verification Findings

1. **Canonical Domain Release:**
   - Canonical edge serves release bundle matching deployed commit `ab0c9845` (preserving `a6336177`).
2. **Accession `0001696841-26-000016` Inspection:**
   - Evaluated 12 reported Form 4 transactions on NVDA.
   - Result: Exactly 12 rendered as `"Sale"`. Zero instances of `"Open Market Sale"` detected.
3. **Mobile Candlestick Controls:**
   - Playwright test at 390x844 confirms candlestick viewport scaling, price markers, and volume drawer operate smoothly.
4. **Physical iOS Hardware:**
   - Explicitly marked: **`NOT_CHECKED`** (Automated browser checks passing; physical device testing open under `TASK-OPEN-04`).

---

## 5. Outstanding Work & Governance Posture

- **Deployment Status:** **`DEPLOYED_AND_VERIFIED`**
- **GitHub PR Status:** PR #328 remains **`OPEN / UNMERGED`** on GitHub.
- **Backend SQL Follow-up (`SEC-LABEL-BACKEND`):** Frontend display is truthful; backend `analyst_dashboard/analyzers/smart_money.py:2226-2227` still emits `'OPEN_MARKET_PURCHASE'`/`'OPEN_MARKET_SALE'` for non-derivative code `P`/`S` rows, tracked under Issue #231 / Issue #165.
