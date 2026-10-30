# ARX TERMINAL — DIRECTIVE 058B
## Technical Handoff Specification for ChatGPT UX Workstream

**Directive ID:** `ARX_058B_CHATGPT_UX_HANDOFF`  
**Repository:** `daakara/finance`  
**Author:** Antigravity (Non-UX Engineering Workstream)  
**Recipient:** ChatGPT (UX Redesign & Frontend Presentation Workstream)  
**Governance:** Directive 058B (§1, §3, §8) — Strict UX Exclusion & Clean Boundary  
**Mainline Anchor:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`  

---

### Executive Summary

Under **Directive 058B**, Antigravity has completed the backend and API contract remediation for the outstanding Sprint 1–4 obligations. In accordance with the Product Owner mandate:
- Antigravity owns and certifies backend, data, and contract correctness.
- ChatGPT exclusively owns UX redesign, presentation hierarchy, layout consolidation, and frontend component implementations.
- Antigravity has **not** modified any UX-owned frontend components (`frontend/app/radar/page.tsx`, `*TerminalView.tsx`, `AlertTriggerModal.tsx`, etc.).

This document provides the exact technical contract specification for the frontend dependencies that require presentation in the UX layer.

---

### Handoff 1: `ARX-S3-004-FE` (Radar Error-versus-Zero Semantics & UI Retry)

#### 1. Finding ID & Scope
- **Finding ID:** `OBL-ARX-S3-004-FE` (Frontend presentation counterpart to `OBL-ARX-S3-004-BE`)
- **Core Requirement:** Ensure the Radar UI clearly distinguishes:
  - Genuinely empty search results (`datasetState: "GENUINELY_EMPTY"`, `status: "SUCCESS"`).
  - Backend/provider error states (`datasetState: "SCAN_FAILED"`, `status: "ERROR"`).
  - Partial data retrieval failures (`status: "PARTIAL"`).
  - Provide an accessible UI Retry control to reload without full page refresh.

#### 2. Backend Contract Changes (`/api/screener/run` GET)

```json
// Previous Response (Ambiguous on Error/Empty):
{
  "totalCandidates": 35,
  "gemsFound": 0,
  "candidates": [],
  "capabilities": { ... },
  "population": { ... }
}

// New Authoritative Contract (Self-Describing Status & Dataset State):
{
  "status": "SUCCESS" | "PARTIAL" | "ERROR",
  "scanState": "COMPLETED" | "PARTIAL" | "FAILED",
  "datasetState": "CANDIDATES_AVAILABLE" | "GENUINELY_EMPTY" | "SCAN_FAILED",
  "error": null | "Error description string",
  "partialErrors": null | [
    {
      "ticker": "NVDA",
      "error": "Provider timeout"
    }
  ],
  "totalCandidates": 35,
  "gemsFound": 0,
  "candidates": [],
  "capabilities": { ... },
  "population": {
    "referenceUniverseCount": 35,
    "activeScreeningCount": 34,
    "currentOpportunityCount": 0,
    "delistedRecordCount": 1,
    "unverifiedRecordCount": 0
  }
}
```

#### 3. Frontend Presentation Requirements (Owned by ChatGPT)
- **Error State (`status: "ERROR"` or `datasetState: "SCAN_FAILED"`):**
  - Suppress candidate cards and tables completely.
  - Display standard ARX error alert banner with message: `Unable to load live screening data: {error}. Missing API connectivity is not represented as zero qualifying candidates.`
  - Render an explicit **"Retry Scan"** button with `role="button"`, touch target $\ge 44\text{px}$, which re-invokes `fetchScreenerGems()`.
- **Genuinely Empty State (`status: "SUCCESS"` and `datasetState: "GENUINELY_EMPTY"`):**
  - Render the standard empty state card: `No qualifying setups found matching active filter criteria. Market conditions or technical filters currently exclude all reference candidates.`
  - Do NOT display an error banner or retry button.
- **Partial State (`status: "PARTIAL"`):**
  - Render qualified candidates normally.
  - Display an informational warning pill indicating: `Partial data feed: {len(partialErrors)} assets awaiting provider tape.`

---

### Handoff 2: `ARX-S3-005-FE` (Radar Candidate Truth & Date Provenance)

#### 1. Finding ID & Scope
- **Finding ID:** `OBL-ARX-S3-005-FE` (Frontend presentation counterpart to `OBL-ARX-S3-005-BE`)
- **Core Requirement:** Eliminate false affordances, display authentic price date provenance, and prevent execution on delisted/unverified assets.

#### 2. Backend Contract Changes (Candidate Object in `/api/screener/run`)
- `screeningEligible`: Guaranteed boolean. Strictly `True` ONLY when `executionStatus === "IN_BUY_ZONE"`. Strictly `False` for `APPROACHING_TARGET`, `WAITING_PULLBACK`, `DELISTED_SECURITY`, `UNVERIFIED_ASSET`.
- `priceBasis`: `"DAILY_CLOSE"` when verified, or `null`.
- `priceDate`: ISO-8601 date string `"YYYY-MM-DD"` of authoritative close, or `null`.
- `corporateAction`: Present for delisted assets (e.g. `CPRX`), containing `status: "DELISTED"`, `action_type`, `effective_date`, `notes`.
- `allowedActions`:
  - Normal asset: `["RESEARCH_PROFILE", "ADD_WATCHLIST", "SET_ALERT"]`.
  - Unverified asset: `["RESEARCH_PROFILE"]`.
  - Delisted asset: `[]` (empty list; trade sizing, alerts, and watchlist additions are forbidden).

#### 3. Frontend Presentation Requirements (Owned by ChatGPT)
- **Delisted Securities (`executionStatus === "DELISTED_SECURITY"`):**
  - Render status badge: `🛑 Delisted / Inactive`.
  - Suppress trade action buttons, buy zone indicators, and alert triggers.
  - Display filing note: `Ceased trading pursuant to cash merger. Historical record retained for audit integrity.`
- **Price Provenance:**
  - Display observation date badge formatted via `formatPriceObservationDate(candidate.priceDate, candidate.priceBasis)`.
  - For unverified prices, suppress daily close claim and render `⚠️ Unverified Price`.

---

### Handoff 3: `ARX-S2-009-FE` (Radar Coverage Disclosure & Analysis Handoff)

#### 1. Finding ID & Scope
- **Finding ID:** `OBL-ARX-S2-009-FE` (Frontend presentation counterpart to `OBL-ARX-S2-009-BE`)
- **Core Requirement:** Present truthful 5-tier lifecycle status and provide seamless handoff into Single-Asset Analysis Hub (`/stock/[ticker]`).

#### 2. Backend Contract Changes
- `capabilities.VCP.lifecycleState`: Exposes authoritative lifecycle state:
  - `"COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED"`
  - `"SCANNER_OPERATIONAL"`
  - `"STALE_OR_ROLLED_BACK_PUBLICATION"`
  - `"NO_COMMITTED_PUBLICATION"`
- `candidate.decisionContextId`: Unique decision trace anchor for single-asset evaluation handoff (`"dec-radar-{sym}-{timestamp}"`).

#### 3. Frontend Presentation Requirements (Owned by ChatGPT)
- **Coverage Disclosure Card:**
  - When lifecycle state is `COMMITTED_PUBLICATION_SCANNER_NOT_ENROLLED`, render:
    `Universe publication active with {populationDenominator} scannable symbols; batch scanner pipeline is pending operational activation.`
- **Analysis Handoff Button:**
  - On candidate card or row, provide primary CTA: `Open in Analysis Hub →` linking to `/stock/{candidate.ticker}?contextId={candidate.decisionContextId}`.

---

### Verification and Test Expectations

When ChatGPT completes the UI presentation work, the following automated frontend test suites will validate end-to-end integration:
1. `frontend/tests/radarErrorAndZeroSemantics.test.ts` (16 test points)
2. `frontend/tests/radarCandidateTruthRemediation.test.ts` (12 test points)
3. `frontend/tests/radarPriceProvenance.test.ts` (5 test points)
4. `frontend/tests/radarCoverageDisclosure.test.ts` (8 test points)

Antigravity stands ready to provide test runner support, mock fixtures, and backend qualification validation whenever ChatGPT presents the candidate UI changes.
