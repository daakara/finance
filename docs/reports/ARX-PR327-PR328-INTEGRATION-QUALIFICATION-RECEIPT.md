# ARX PR #327 + PR #328 Integration & Production Rollout Qualification Receipt

**Target Integration Commit SHA:** `ab0c9845617eff812115de7b3e5e6163f04d7f06`  
**Base Commit SHA:** `25f0bf10c78cc8766a61629e318910ef469c230c`  
**Integration Branch:** `fix/arx-s4-radar-001-operational-truth`  
**Hosted Qualification Run:** `45596974737` (`12/12` `RELEASE_QUALIFIED`)  
**Production Rollout Date:** 2026-11-25  
**Final Verdict:** **`DEPLOYED_AND_VERIFIED`** (Live on Cloudflare Pages and Railway Web API; GitHub PRs remain OPEN; operational holds strictly active)

---

## 1. Executive Summary & Integrated Scope

Commit `ab0c9845617eff812115de7b3e5e6163f04d7f06` unifies three critical operational and UI workstreams into a single release-qualified production deployment:

1. **PR #327 (Market Opportunity Radar Operational Truth & Fail-Closed Semantics):**
   - Implements fail-closed states (`ERROR`, `PENDING`, `STALE`) across Radar capability surfaces with `matchedCount: null` (prohibiting synthetic zero-match claims when data is missing or pipelines are pending).
   - Truthful mixed-date tape disclosures: Explicitly reports `marketDataOldestAsOf..marketDataAsOf` (`2026-11-06..2026-11-24`, 196/500 tickers evaluated).
   - Historical VCP snapshot notice: Discloses historical snapshot `vcp-snap-1794130818-3c0eb357` (`UB-ARX-EXPANSION-500-20261108`, `2026-11-06`, 0/500 matches) as a frozen historical artifact rather than a fresh market-wide zero result.
   - Preserves frozen Smart Money pilot badge (`SNAP_SM_1792939989_48f30db6`) under `HOLD_SEC_001`.
   - Links canonical tasks `ARX-S4-RADAR-001`, `ARX-S3-004`, `ARX-S3-005`, and `ARX-S2-009`.

2. **PR #328 (Mobile Candlestick Chart, Truthful Preflight Copy, Form 4 Labels & TD-08):**
   - Mobile chart responsiveness: Restores responsive container layout, touch-friendly legend, and drawer controls in a 390x844 viewport.
   - Pre-Flight checklist copy truth: Removes hyperbolic synthetic phrases (`"FLIGHT CLEARANCE REVOKED"`, `"hard-earned money"`), replacing them with truthful structural status copy.
   - Form 4 disclosure corrections: Emits precise semantic labels (`Sale`, `Gift`, `Grant / Award (Scope Unverified)`, `Tax Withholding`, `Form 4 Filing (Unparsed)`); prohibits unsupported `"Open Market"` labels. Accession `0001696841-26-000016` confirmed live with 12 `"Sale"` transactions.
   - `TD-08` resolution: Removes hardcoded `"2026-04-01"` options expiration fallback in `frontend/app/page.tsx`.

3. **PR #325 (Cornish–Fisher Minimum-Observation Runtime Enforcement):**
   - Enforces $N \ge 30$ closed daily return observations for Cornish–Fisher VaR calculation.
   - Returns explicit `UNAVAILABLE` when observations are fewer than 30, eliminating synthetic fallback assumptions.
   - Preserves multi-exchange calendar gating with `exchange_calendars-4.13.3`.

---

## 2. Hosted Qualification Matrix (Run `45596974737`)

Hosted CI qualification Run `45596974737` passed all twelve release quality gates:

| Gate | Description | Target / Requirement | Result |
| :--- | :--- | :--- | :---: |
| **RQ-01** | Repository Integrity | Clean working tree, manifest valid, zero untracked runtime artifacts | **`PASS`** |
| **RQ-02** | Security & Secret Isolation | Zero plaintext secrets, zero client-exposed credentials, safe environment parsing | **`PASS`** |
| **RQ-03** | Lint & Type Integrity | Zero TypeScript errors (`tsc --noEmit`), Python AST validation | **`PASS`** |
| **RQ-04** | Syntax & Structure | Clean JSON/YAML manifests, syntax verification | **`PASS`** |
| **RQ-05** | Unit Test Suite | Core engine, financial calculation, and analytical test suites pass | **`PASS`** |
| **RQ-06** | Database Schema Integrity | PostgreSQL 16 migration validity, immutable triggers, constraint parity | **`PASS`** |
| **RQ-07** | Build & Asset Compilation | Production Next.js build succeeds with zero fatal warnings | **`PASS`** |
| **RQ-08** | API Contract Compliance | Governance `/api/v1/governance/epoch-2/status` schema preserved | **`PASS`** |
| **RQ-09** | Observability & Telemetry | Sentry DSN configuration, subsystem tagging, trace context intact | **`PASS`** |
| **RQ-10** | Rollback Traceability | Validated rollback targets for Railway API and Cloudflare Pages | **`PASS`** |
| **RQ-11** | CSP & Header Security | Cloudflare Pages `_headers` Content-Security-Policy integrity | **`PASS`** |
| **RQ-12** | Release Manifest Binding | Release manifest SHA bound to candidate `ab0c9845` | **`PASS`** |

*Note on Frontend Unit Baseline:* Full Vitest suite reports 61/63 files passed (682 passed, 2 failed in `DecisionSurfaceIntegrity.test.tsx` and `Wave1DeclutterPreservation.test.tsx`), pre-existing since baseline `2eccc64a` and tracked under `TASK-OPEN-04`.

---

## 3. Production Deployment Identifiers & Rollback Anchors

Sequential deployment was executed using fail-closed ordering (Railway backend deployed and verified first, followed by Cloudflare Pages frontend):

| Subsystem | Active Production Deployment | Active Release SHA | Demonstrated Rollback Deployment | Rollback Release SHA |
| :--- | :--- | :--- | :--- | :--- |
| **Railway Backend (`web`)** | `ea3ccfba-658b-4993-8d70-931ca63a131a` | `ab0c9845617eff812115de7b3e5e6163f04d7f06` | `2891062b-e7bf-44cb-a2e5-57baf7e87daa` | `25f0bf10c78cc8766a61629e318910ef469c230c` |
| **Cloudflare Pages (`frontend`)** | `6c0d391e-176d-4eca-9aec-f5eef6dd69c5` | `ab0c9845617eff812115de7b3e5e6163f04d7f06` | `c34a7f69-eb54-4594-8cff-44e7625f9482` | `a63361777f039511b8e0823ec4456b807dbe8517` |

---

## 4. Live Production Verification Evidence

1. **Governance & Backend Status Endpoint:**
   - URL: `https://www.arxterminal.com/api/v1/governance/epoch-2/status`
   - Verified Response:
     - `runningReleaseSha`: `"ab0c9845617eff812115de7b3e5e6163f04d7f06"`
     - `prospectiveCaptureAuthorized`: `false`
     - `activeHold`: `"HOLD_SEC_001"`
2. **Canonical Pages Release Marker:**
   - Client bundle serves release marker bound to `ab0c9845617eff812115de7b3e5e6163f04d7f06`.
3. **Form 4 Venue Label Truthfulness:**
   - Target Accession: `0001696841-26-000016` (NVDA).
   - Displayed Rows: Exactly 12 transactions rendered with `"Sale"` (previously mislabeled as `"Open Market Sale"`).
   - Retained XML evidence matches source disclosure with zero fabricated venue claims.
4. **Mobile Candlestick Chart (390x844 Viewport):**
   - Playwright automated mobile inspection confirms candlestick chart renders with visible price scales, usable legend, and functional drawer.
5. **Physical iOS Hardware Inspection:**
   - **Status:** **`NOT_CHECKED`** (Playwright automated viewport check passed; physical Apple hardware audit remains open under `TASK-OPEN-04`).

---

## 5. Operational Governance & Holding Posture

- **Deployment Status:** **`DEPLOYED_AND_VERIFIED`**
- **GitHub PR Status:** PR #327, PR #328, and PR #325 remain **`OPEN / UNMERGED`** pending formal GitHub PR review and administrative trunk merge.
- **Operational Holds:**
  - `HOLD_SCAN_001`: Autonomous Radar scheduling and live 500-universe snapshot publication remain **HELD** (`VCP_NATURAL_SCHEDULER_ENABLED=0`).
  - `HOLD_SEC_001`: Live SEC ingestion remains **HELD**; worker operates in disabled standby.
  - `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, and `PHASE_A3_UX_HOLD` remain in full effect.
