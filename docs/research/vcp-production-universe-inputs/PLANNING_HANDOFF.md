# ARX Terminal — Planning Handoff: Production-Universe Input Qualification & Field Gaps
**Backlog Reference:** `BACKLOG-VCP-PROD-UNIVERSE-003` (Research Investigation)  
**Execution Worktree:** `C:\Users\akara\Documents\Projects\finance-vcp-universe-inputs`  
**Branch:** `research/vcp-production-universe-inputs`  
**Date:** 2026-10-16  
**Status:** PROPOSED PLANNING UPDATE — AWAITING PRODUCT OWNER REVIEW

---

## 1. Backlog Item Status & Separate Governance Tracking

```text
BACKLOG_ID                      = BACKLOG-VCP-PROD-UNIVERSE-003 (Proposed Item)
TITLE                           = Production-Universe Implementation Package & Field Gaps
WORKSTREAM                      = Worktree C
RESEARCH_PACKAGE_STATUS         = COMPLETE_WITH_DOCUMENTED_GAPS
FIELD_GAP_INVESTIGATION_STATUS  = FIELD_REQUIREMENTS_ESTABLISHED
IMPLEMENTATION_PACKAGE_STATUS   = COMPLETE_FOR_REVIEW
SOURCE_QUALIFICATION_STATUS     = UNQUALIFIED_FOR_PRODUCTION
PUBLICATION_READINESS           = BLOCKED (Worktree A unmerged, preflight unexecuted, source gaps)
CONTRACT_REFERENCE_SHA          = 6c9b5b00de26545239616ac46f78b35a343ee3dd (Tree: 3ef8ead46c34754724cfdc5e906af66271fa2ed2)
PRODUCTION_REFERENCE_SHA        = 2eb032b2aec18ac4fea0f654eee8fb6c028466fe (PR #5; zero diff on universe contracts)
PRODUCTION_MARKET_COVERAGE      = NOT_INSPECTED
```

---

## 2. Completed Acceptance Criteria

1. **Worktree Isolation:** Independent worktree `finance-vcp-universe-inputs` created from base `787c1977b0071f2b31c4ad3d36a7f872147ad59e` with zero modification to Worktrees A or B.
2. **Contract Extraction & Evidence Requirements:** Complete 12-field evidence requirement specifications authored in `FIELD_EVIDENCE_REQUIREMENTS.md` documenting consumer code paths, rule definitions, missing-data behavior, and the silent default-filling code path in `contracts.py#L122-L125`.
3. **Three Source Families Qualified:** Evaluated SEC EDGAR, Nasdaq Trader / CTA, and OpenFIGI across documented attributes, observed bounded probes, inferences, and unknowns in `GAP_RESOLUTION_OPTIONS.md`.
4. **Targeted Public Probes:** Executed and preserved three public probes in scratch:
   - `PROBE_SEC_SUBMISSIONS_MULTICLASS` (Alphabet Inc. CIK 0001652044 with 4 tickers).
   - `PROBE_OPENFIGI_DISAMBIGUATION` (Disambiguation of GOOG/GOOGL, AAPL, SPY, and REIT O).
   - `PROBE_NASDAQ_TRADE_HALTS` (Real-time regulatory trade halts feed).
5. **Illustrative Research Model:** Standalone script executed in pure Python demonstrating discrete synthetic rule tests, raw sample overlap audit (noting truncated sample slice), and simulated readiness/session-guard arithmetic without DB access or scanner dispatch.
6. **Publication Runbook Proposal:** 7-step publication sequence formulated, separating publication from natural execution authorization and reconciling $E = DR + DU$ with the 100% runtime execution floor.

---

## 3. Concrete Gaps & Open Technical Questions

1. **Common Stock Verification (Rule R02):** Checking `ETF=N` and `Test Issue=N` with name regex is an unvalidated candidate classifier, not proof of common stock. OpenFIGI `securityType2` provides explicit classification.
2. **Security ID Collisions (CIK):** CIK identifies an issuer, not a security line. Multiple share classes (e.g. `GOOG` and `GOOGL`) share a CIK, causing primary-key collisions unless disambiguated via Share Class FIGI or an authorized compound key.
3. **Missing Source Fields (Rules R01, R05, R06):** Raw directory feeds do not provide `asset_class`, `currency`, or contract `primary_listing`.
4. **Point-in-Time Delisting Data (Rule R07):** Daily directory feeds drop delisted symbols overnight without recording historical `delisting_date`. Historical replay requires an independent SEC Form 25 ingestion pipeline.
5. **Intraday Trading Halts (Rule R03):** Daily directories reflect morning status (~03:00 ET) only; the Nasdaq Trade Halts RSS feed provides affirmative intraday halt data.
6. **Production Market Data Coverage:** Coverage in `/root/.finance_market_store.db` is uninspected per authorization constraints. The preflight query must be executed prior to publication.

---

## 4. Policy Decision Proposals Submitted for Product Owner Determination

Four formal policy proposals are detailed in [`POLICY_DECISION_PROPOSALS.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/POLICY_DECISION_PROPOSALS.md):
1. **Decision 1: Primary Listing Semantics (Rule R06):** **STATUS: UNRESOLVED.** Allowed-exchange membership alone does not resolve secondary or cross-listing status. Requires binding PO determination among venue proxy, strict MIC, or waiver.
2. **Decision 2: Quotation Currency (Rule R05):** Adopt Option 2A (Codified Venue Rulebook Convention for US major exchanges within defined scope and exceptions). Positive pricing is strictly rejected as currency evidence.
3. **Decision 3: Security Identity Granularity (Rule R00):** **Listing-Level Granularity Required.** Authorize `CIK_{cik}_{symbol}` as temporary single-session bridging key; transition to OMG **Composite/Listing FIGI** in Phase 2.
4. **Decision 4: Instrument Classification & Eligibility (Rule R02):** Adopt Option 4B (OpenFIGI `securityType2` for taxonomy). Separate external taxonomy from PO determination on REIT/BDC eligibility.

---

## 5. Engineering Implementation Package Deliverables

The implementation package prepares all requirements, architecture, and planning necessary to execute an authorized production build upon PO approval:

1. [`POLICY_DECISION_PROPOSALS.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/POLICY_DECISION_PROPOSALS.md): 4 policy proposals with explicit tradeoffs and recommendations.
2. [`INPUT_VALIDATION_SPEC.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/INPUT_VALIDATION_SPEC.md): Fail-closed input validation boundary, F01–F12 field specifications, collision assertions, temporal evidence requirements, and default-filling mitigation.
3. [`PUBLICATION_TOOLING_SPEC.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/PUBLICATION_TOOLING_SPEC.md): Architecture for future 6-phase atomic CLI publication tool adhering to verified `UniverseStore` capabilities, dry-run mode, forward-only revocation, and read-only handoff.
4. [`IMPLEMENTATION_PLAN.md`](file:///C:/Users/akara/Documents/Projects/finance-vcp-universe-inputs/docs/research/vcp-production-universe-inputs/IMPLEMENTATION_PLAN.md): Gap-to-implementation mapping, 6-task roadmap, 12 expanded acceptance scenarios, and partitioned effort estimate (14–20 hands-on engineering hours decoupled from PO calendar and vendor onboarding).

---

## 6. Worktree Reconciliation & Handoff to Worktree A

- Worktree C deliverables are purely non-mutating research and specification documents stored in `docs/research/vcp-production-universe-inputs/`.
- Zero application code, shared configuration, manifests, or database schemas were modified.
- Finding on silent default-filling in `contracts.py#L122-L125` is documented for Worktree A's release assessment. Worktree A is not blocked by Worktree C's research status.
- Worktree A candidate wiring has been merged to `origin/main` via PR #5 (`2eb032b2aec18ac4fea0f654eee8fb6c028466fe`). Actual path parity verification confirmed `analyst_dashboard/vcp/session_guard.py` exists on disk and has zero diff against production.
- Publication remains inert: publication does NOT authorize execution, activate schedulers, or dispatch scans.

