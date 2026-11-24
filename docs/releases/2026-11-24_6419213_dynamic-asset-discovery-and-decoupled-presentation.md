# ARX Terminal — Production Release Notes

## Dynamic Company-Name Asset Discovery, Provider Failure Handling & Decoupled Limited-History Presentation

### Release Identity

```ini
RELEASE_DATE =
  2026-11-24
RELEASE_CANDIDATE_SHA =
  6419213f56695cbc0a5fa6dc08b10e0496f37afa
QUALIFICATION_RECEIPT_SHA =
  2f4294db6f97c9f1d30603fc310530a01a5c866d
RELEASE_MANIFEST_SHA =
  71c0cb980223ecf184e130dc68cce29168ebe87b
BRANCH =
  feat/dynamic-asset-discovery
GOVERNING_PR =
  #323
QUALIFICATION_POLICY_VERSION =
  2.2.0-D074A4
RELEASE_QUALIFICATION_SUITE =
  RQ-01 through RQ-13 (13/13 CONTROLS PASS, 100.33s runtime)
CLOUDFLARE_PAGES_DEPLOYMENT_ID =
  f96c89d2-08e8-410e-a23c-e3c684f9f76c
LIVE_PRODUCTION_URL =
  https://www.arxterminal.com/
PRODUCTION_VERIFICATION =
  PASS (Playwright Live Production Audit, 3/3 checks PASS)
```

---

### Release Purpose & Classification

```ini
RELEASE_PURPOSE =
  Eliminate raw query submission on homepage discovery, resolve company names to exchange tickers against provider directory, suppress arbitrary custom-ticker launches when provider matches exist, isolate provider search errors, establish epistemic tri-state asset handling (404 unknown vs 422 limited-history vs 200 seasoned), decouple Security Master listing identity from SEC regulatory filing coverage in limited-history presentation, and clarify the 15-session baseline quantitative modeling gate.

RELEASE_CLASSIFICATION =
  FRONTEND_DISCOVERY_ENHANCEMENT
  EDGE_PROXY_ADDITION
  EPISTEMIC_TRUTH_GOVERNANCE
  UX_STATE_REMEDIATION

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  YES / INTENTIONAL:
  1. Searching by company name (e.g. "Anthropic") resolves dynamically to equity ticker "ANT".
  2. Raw query string cannot be submitted while discovery is in-flight.
  3. Arbitrary custom symbol launch button is suppressed when valid provider matches exist.
  4. Unknown/unindexed symbols render explicit `asset-not-found-state` with actionable guidance; stale tape diagnostics and synthetic 0/100 scoring are suppressed.
  5. Securities with limited history render `insufficient-history-state` disclosing exact observation dates, trading session count, and decoupled capability status:
     - Listing Identity: Verified (confirmed via Security Master)
     - SEC Filing Coverage: Not verified (strictly defaults to unverified)
  6. Clarifies that 15 daily sessions is the baseline modeling gate before quantitative evaluation commences, while individual indicators enforce their own history requirements.
  7. Missing indicator values are rendered as UNAVAILABLE or N/A, never as numerical zero.

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE: Underlying quantitative engines, factor formulas, and Stage 2/VCP parameters remain unchanged.
```

---

### Key Architectural Invariants Enforced

1. **`ARX_INV_001` (Zero-Invariance):** Missing disclosures, insufficient price bars, and unassessed indicators are never defaulted to `0.0`, `0`, or `0/100`. Missing metrics display as `UNAVAILABLE` or `N/A`.
2. **`ARX_INV_003` (Fail-Closed Safety):** Missing data does not confer analytical or trading eligibility. Limited-history assets cannot be routed to execution surfaces.
3. **`ARX_INV_007` (Provider Fault Isolation):** External directory proxy outages return `503 SERVICE_UNAVAILABLE`; provider errors are never collapsed into an empty result array.
4. **Epistemic Tri-State Architecture:**
   - State 1: `asset-not-found-state` (HTTP 404).
   - State 2: `insufficient-history-state` (HTTP 422).
   - State 3: Quantitative Tape Observable (HTTP 200).
5. **Decoupled Verification Invariant:** Active exchange listing in the Security Master verifies listing identity, but **never** implies filing completeness, insider ownership verification, or scoring eligibility.
6. **The Cardinal Separation Principle:**
   $$\boxed{\text{Listing Identity} \ne \text{SEC Filing Completeness} \ne \text{Scoring Eligibility} \ne \text{Execution Clearance}}$$
   Sufficient session counts never automatically establish analytical, scoring, or execution readiness:
   - *15 sessions:* Enforced runtime gate for daily analytics (`validate_and_prune_daily_history` in `api/routes/analytics.py:1121`). Minimum input count for daily decision analytics.
   - *20 sessions:* Smoothing heuristic for 20 EMA / 20 ADV (`api/routes/analytics.py:143`, `optimal_execution.py:194`). Unenforced runtime gate (`pandas.ewm` calculates from $N \ge 1$).
   - *30 sessions:* Statistical confidence standard (`ARX_INV_013` sample size $N \ge 30$). Runtime gate in `advanced_risk_analyzer.py` is unresolved; calculation proceeds on available returns.
   - *50 sessions:* Enforced runtime gate for 50 SMA (`api/routes/analytics.py:156`) and swing trade level generation (`optimal_execution.py:137`). `ConfluenceEngine` does not enforce a 50-session cutoff.
   - *200+ sessions:* Enforced mathematical requirement for 200 SMA (200 bars), 22-session slope (222 bars), and 52-week metrics (up to 252 bars) in Minervini Stage 2 evaluation (`classifier.py:82-90`, `predicate_registry.py:132,163`).

---

### Release Qualification Summary (RQ-01..RQ-13)

| Check | Control Name | Verdict | Scope & Assertion |
| :--- | :--- | :---: | :--- |
| **RQ-01** | Candidate Identity | **PASS** | Validated commit `6419213f56695cbc0a5fa6dc08b10e0496f37afa` ancestry. |
| **RQ-02** | Release Diff Integrity | **PASS** | 51/51 files strictly contained in `release_manifest.json`. |
| **RQ-03** | Backend Regression | **PASS** | Complete backend test suite passed with 0 failures. |
| **RQ-04** | Frontend Regression & Typecheck | **PASS** | Verified 6 targeted qualification controls: TypeScript typecheck (0 errors), 17/17 governor sizing tests, weekly spotlight selection, CDNS price trend parity, SEC disclosure view, and radar ranking handoff. |
| **RQ-05** | Cross-Engine Invariants | **PASS** | 28 domain invariants verified across all engines. |
| **RQ-06** | Governance Invariants | **PASS** | Backlog integrity, JSON schema, and Sprint 5 lock verified. |
| **RQ-07** | Deployment Safety | **PASS** | Environment configs, CSP headers, and secret isolation verified. |
| **RQ-08** | Artifact Provenance | **PASS** | Pinned cryptographic digests verified for all runtime assets. |
| **RQ-09** | Compatibility Suite | **PASS** | Worker runtime contracts and Redis/Postgres compatibility verified. |
| **RQ-10** | Configuration Integrity | **PASS** | Railway and Cloudflare Pages build configurations verified. |
| **RQ-11** | Rollback Readiness | **PASS** | Zero destructive migrations; forward and rollback paths verified. |
| **RQ-13** | Financial Integrity | **PASS** | Zero synthetic market data in production ledgers; sizing contracts verified. |
| **RQ-12** | Qualification Summary | **PASS** | Master receipt emitted with PASS verdict (Duration: 100.33s). |

---

### Frontend Test Suite Distinction & Unit Run Disposition

- **Targeted RQ-04 Qualification Scope:** Executed `scripts/release/rq04_frontend_regression.py` covering targeted regression gates (TypeScript type-check, 17 governor sizing tests, spotlight selection, CDNS trend parity, SEC disclosure view, and radar ranking handoff). All 6 checks passed cleanly (`PASS`).
- **Full Unit Suite Scope (`npm test` / `vitest run`):** Executed all 63 test files across the frontend application (684 tests):
  - 61 test files passed (682 tests passed).
  - 2 test files reported 1 failure each (exiting with code 1):
    1. `components/__tests__/DecisionSurfaceIntegrity.test.tsx` (Test: `ensures all three terminal views render identical actionability badge and zero duplicate label`): Assertion failed at line 402 (`expect(screen.queryByText(/^WAIT FOR TRIGGER$/i)).toBeNull()`). Root cause: `GuidedTerminalView` renders a `<button ...>Wait for Trigger</button>`. **Disposition: PRE-EXISTING** (verified identical in baseline commit `2eccc64a2cb3c649ad4d584a942285a6406d1ce0` at `components/terminal/GuidedTerminalView.tsx:301`).
    2. `components/__tests__/Wave1DeclutterPreservation.test.tsx` (Test: `page.tsx relocates PageIntro, IntentHero, and WeeklyConfluenceSpotlight below detailed tabs`): Assertion failed at line 179 (`expect(pageIntroIdx).toBeGreaterThan(detailedTabsIdx)`). Root cause: In `page.tsx`, `<PageIntro` (line 455) appears prior to `detailed-domain-content` (line 780). **Disposition: PRE-EXISTING** (verified identical in baseline commit `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`).
- **PR #323 Candidate Behavioral Tests:** 10/10 tests passed in `lib/__tests__/limitedHistoryPresentation.test.ts` and 5/5 tests passed in `lib/__tests__/analyticsUnavailableHandling.test.ts`.

---

### Live Production Verification Evidence

Live Playwright verification was conducted against `https://www.arxterminal.com/` (Deployment `f96c89d2-08e8-410e-a23c-e3c684f9f76c`):

1. **ANT Decoupled Presentation Check:**
   - Navigated to `https://www.arxterminal.com/?symbol=ANT`.
   - Verified `insufficient-history-state` is present.
   - Verified decoupled capability chips: `Listing Identity: Verified` and `SEC Filing Coverage: Not verified`.
   - Verified clarified modeling gate: *"15 completed daily sessions is the baseline gate before quantitative evaluation commences"*.
   - **Result: PASS**.
2. **Unknown Asset Differentiation Check:**
   - Navigated to `https://www.arxterminal.com/?symbol=ANTHROPIC`.
   - Verified `asset-not-found-state` is present.
   - Verified clean empty slate with message: *"Asset unavailable or not found. Check ticker spelling or verify active US equity listing."*
   - Verified stale-tape diagnostics and synthetic scores are absent.
   - **Result: PASS**.
3. **NVDA Baseline Continuity Check:**
   - Navigated to `https://www.arxterminal.com/?symbol=NVDA`.
   - Verified standard seasoned workstation loads correctly with price charts, VCP setups, and Confluence score.
   - **Result: PASS**.

All operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`) remain strictly active.
