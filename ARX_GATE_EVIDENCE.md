# ARX TERMINAL — CANONICAL GATE EVIDENCE: ALCON V5 PUBLICATION & MAINLINE PRODUCTION RELEASE

**Document Identifier:** `ARX_GATE_EVIDENCE`  
**Execution Classification:** `ARX_ALCON_V5_CONTROLLED_PRODUCTION_RELEASE`  
**Governance Framework:** ARX Closure Model v2 + Governance Efficiency v1.1  
**Research Owner Authority:** David Akara (Sole merge, release, risk-policy, and human-research approval authority)  
**Authorization Directives:**
- `ARX_ALCON_SCENARIO_DERIVED_NET_DEBT_CORRECTION_AND_CANDIDATE_REQUALIFICATION_AUTHORIZED`
- `ARX_ALCON_V5_CONDITIONAL_MERGE_FRONTEND_DEPLOYMENT_AND_VERIFICATION_AUTHORIZED`
- Directive: `PATH 1 — PRESERVE PRODUCTION UX AND REGULARIZE ITS MAINLINE LINEAGE`
**Classification Tier:** `TIER_4_PRODUCTION_DECISION_SAFETY`  
**Owner Decision Authorization:** `APPROVED`  
**Deployed Frontend Release SHA:** `64b5f892bf8b20f55b810599393bbbeb4d6aba91`  
**Base Lineage Integration SHAs:**
- PR #86 Merge: `ae28f73335d94c3d41072def6bdaaa6fdc6d91c6` (Production UX regularized to main)
- PR #85 Merge: `136ecb301bca1ba4acfa6b0540802ba7ef86e6e2` (Alcon V5 Scenario-Derived Model Certified)
- PR #87 Merge: `64b5f892bf8b20f55b810599393bbbeb4d6aba91` (Design Parity Gate v1.1 & Pre-Flight Canonical Controls)
**Active Production Cloudflare Deployment:** `1ce68307-6cec-40ad-95a1-44b062d4a4da`  
**Production Canonical URLs:** `https://www.arxterminal.com`, `https://1ce68307.finance-xp8.pages.dev`  
**Release Artifact Tree SHA-256:** `90cabe89e4c289e43358e6ca72bab3fbcb2f1f0f8a0ed7773c2a1aab9cffe404` (424 static files)  
**Cloudflare Auto-Deploy Status:** `DISABLED` (`production_deployments_enabled: false`)  
**Rollback Targets:** `2370f8fe-ba17-4d3b-90e8-585fcac534a5`, `2a7beafd-c12f-4d1e-a731-f2b44231e843`  
**Preserved Railway Backend Deployment:** `059f9d79-6848-4095-aa61-5936ac3ee650` (Commit `a864562`, HTTP 200 OK `{"status":"online"}`)  
**Consolidated Machine Verdict:** **`ARX_ALCON_V5_PRODUCTION_RELEASE_VERIFIED`**  
**Timestamp:** `2026-10-25T08:50:00Z`  

---

## 1. Executive Summary & Release Verification Outcome

Pursuant to the Research Owner's authorizations `ARX_ALCON_SCENARIO_DERIVED_NET_DEBT_CORRECTION_AND_CANDIDATE_REQUALIFICATION_AUTHORIZED` and `ARX_ALCON_V5_CONDITIONAL_MERGE_FRONTEND_DEPLOYMENT_AND_VERIFICATION_AUTHORIZED` following integration directive `PATH 1 — PRESERVE PRODUCTION UX AND REGULARIZE ITS MAINLINE LINEAGE`, the complete Alcon V5 publication model and mainline UX lineage have been formally integrated, built in an isolated production worktree, deployed to Cloudflare Pages production, and verified via Playwright live browser acceptance on `https://www.arxterminal.com`.

### 1.1 Lineage & Pull Request Reconciliation
1. **UX Lineage Regularization (PR #86):** Production commits `9b00142` and `8a15f1a` were reconciled into mainline `main` via PR #86 (merge commit `ae28f73335d94c3d41072def6bdaaa6fdc6d91c6`), eliminating mainline divergence without rolling production back.
2. **Alcon V5 Model Publication (PR #85):** Alcon PR #85 was synchronized with updated `main`, verified across all 64 Python valuation/integrity tests and 36 frontend architectural suites, transitioned from Draft to Ready for Review, and squash-merged into `main` via commit `136ecb301bca1ba4acfa6b0540802ba7ef86e6e2`.
3. **Design Parity Gate v1.1 Integration (PR #87):** Mainline commit `64b5f892bf8b20f55b810599393bbbeb4d6aba91` incorporated canonical pre-flight single-source safety controls and Design Parity Gate v1.1, cleanly preserving Alcon V5 model integrity and immutable financial ledgers.

---

## 2. Alcon V5 Valuation & Modeling Reconciliation

### 2.1 Security Identity & Filer Provenance
- **Reconciled CIK:** `0001167379` (Alcon Inc., Swiss Registered Shares, ISIN `CH0432492467`, primary NYSE trading line `ALC`).
- **Primary Source Attestation:** Reconciled from audited SEC EDGAR 20-F annual report (accn `0001167379-26-000014`) and 6-K filings.
- **Reconciliation Audit:** Replaces internal or secondary drafting references across canonical ledgers, exporter (`scripts/research/export_alc_dossier.py`), opportunity dossier snapshots, catalog (`lib/opportunityDossiers.ts`), and rendered view.

### 2.2 Terminal Net-Debt Policy (Owner Decision Option B)
In accordance with Owner Decision Option B, terminal net debt in Alcon Model v5.0 is derived strictly from cash-flow dynamics:
$$\text{terminal\_net\_debt} = \text{ending\_gross\_debt} - \text{ending\_cash}$$
Historical Model v4.0 is preserved as an immutable baseline (`ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`, SHA-256: `d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4`), where fixed -$2,176.2M unamortized net debt was maintained as a modeling assumption.

#### Reference Scenarios Audit Ledger:
| Scenario | Ending Gross Debt | Ending Cash | Debt − Cash (V5 Net Debt) | Historical V4 Net Debt | Net Debt Delta | V5 Target Price | V4 Target Price | V5 Annual IRR | V4 Annual IRR | IRR Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conservative** | $0.0M | $2,045.79M | **-$2,045.79M** | -$2,176.20M | +$130.41M | **$76.36** | $76.64 | **3.11%** | 3.18% | -7 bps |
| **Base Multiple** | $0.0M | $3,687.55M | **-$3,687.55M** | -$2,176.20M | -$1,511.35M | **$93.20** | $90.02 | **7.23%** | 6.50% | +73 bps |
| **Base DCF (Primary)**| $0.0M | $3,687.55M | **-$3,687.55M** | -$2,176.20M | -$1,511.35M | **$95.00** | $91.82 | **7.64%** | 6.92% | +72 bps |
| **Optimistic** | $0.0M | $5,290.81M | **-$5,290.81M** | -$2,176.20M | -$3,114.61M | **$129.97** | $123.42 | **14.50%** | 13.34% | +116 bps |
| **Conservative Dynamic**| $1,000.0M | $3,645.79M | **-$2,645.79M** | -$2,645.79M | $0.0M | **$76.70** | $76.70 | **3.20%** | 3.20% | 0 bps |
| **Optimistic Dynamic** | $0.0M | $5,290.81M | **-$5,290.81M** | -$5,290.81M | $0.0M | **$129.06** | $129.06 | **14.34%** | 14.34% | 0 bps |

### 2.3 Hurdle-Specific Accumulation Ceilings
- **Annual IRR Basis:** 8.00% Hurdle: **$66.31** | 8.50% Hurdle: **$64.82** | 10.00% Hurdle: **$60.56**
- **Calendar XIRR Basis:** 8.00% Hurdle: **$66.36** | 8.50% Hurdle: **$64.86** | 10.00% Hurdle: **$60.61**

### 2.4 Immutable Ledger Cryptographic Attestations
- **Certified V5 Scenario Ledger:** `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json`  
  SHA-256: `c4db315f037416a10c01ebf0eb1ccfda52a5ed66fa9396e6f2b8aa9d38256c38`
- **Preserved Historical V4 Baseline Ledger:** `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`  
  SHA-256: `d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4`
- **Opportunity Dossier Export Payload:** `frontend/lib/alconDraftDossier.json`  
  SHA-256: `b10b39a88b3096fab87bf170175ad85217064f9f59d0ff52016dbfe8accc1f6f`

---

## 3. Production Release & Live Acceptance Audit Matrix

### 3.1 Pre-Flight Gate Qualification
| Gate ID | Domain | Assertion / Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **REL-01** | Ledger Integrity | V5 & V4 ledgers match cryptographic SHA-256 bitwise | Verified 100% | **PASS** |
| **REL-02** | Export Consistency | `python scripts/research/export_alc_dossier.py --check` | SHA `b10b39a8...` match | **PASS** |
| **REL-03** | Snapshot Gate | `npm run check:dossier` validates snapshot & manifests | Exit code 0 | **PASS** |
| **REL-04** | Architectural Tests | 36 test suites in `npm run test:arch` | 36/36 Suites Passed (100%) | **PASS** |
| **REL-05** | Static Compilation | `next build` static export across 149 routes | 149/149 SSG Routes (0 errors) | **PASS** |
| **REL-06** | Content Digest | Deterministic SHA-256 digest over 424 files in `out/` | `90cabe89e4c289e43358e6ca72bab3fbcb2f1f0f8a0ed7773c2a1aab9cffe404` | **PASS** |
| **REL-07** | Pages Deployment | Static deployment to Cloudflare Pages production | Deployment `1ce68307` | **PASS** |
| **REL-08** | Auto-Deploy Invariant | Cloudflare auto-deployment remains disabled | `production_deployments_enabled: false` | **PASS** |
| **REL-09** | Backend Invariant | Railway backend `059f9d79` preserved untouched | HTTP 200 OK `{"status":"online"}` | **PASS** |

### 3.2 Live Browser Acceptance Audit (`https://www.arxterminal.com`)
Audit runner: `verify_expanded_alcon.py` & `verify_production_live.py` (Playwright Chromium, Desktop 1440x900 & Mobile 390x844).

| Route / Surface | Tested Capability | Verified Evidence | Status |
| :--- | :--- | :--- | :---: |
| **`/` (Root Desktop)** | Four-Pillar Confluence | Technical, Fundamental, Institutional Flow, Macro Environment (all `25% Weight`) | **PASS** |
| **`/` (Root Desktop)** | Pre-Flight Readiness Bar | Mounted directly beneath candlestick chart in DOM hierarchy | **PASS** |
| **`/` (Root Mobile)** | Responsive Confluence Layout | Clean mobile wrapping at 390x844, zero layout shifts | **PASS** |
| **`/research?symbol=ALC`** | Model Version Badge | `Model v5.0 Certified (Scenario-Derived Net Debt)` | **PASS** |
| **`/research?symbol=ALC`** | SEC Filer Identity | `CIK 0001167379 • Swiss Registered Shares (ISIN CH0432492467)` | **PASS** |
| **`/research?symbol=ALC`** | Valuation Targets | $76.36 (Cons), $93.20 (Base Mult), $95.00 (Base DCF), $129.97 (Opt) | **PASS** |
| **`/research?symbol=ALC`** | Sensitivity Variants | $76.70 (Cons Dynamic), $129.06 (Opt Dynamic) | **PASS** |
| **`/research?symbol=ALC`** | Accumulation Ceilings | $66.31 (8.0%), $64.82 (8.5%), $60.56 (10.0%) | **PASS** |
| **`/research?symbol=ALC`** | Net Debt Bridge Accordion | -$2,045.8M (Cons), -$3,687.6M (Base), -$5,290.8M (Opt) | **PASS** |
| **`/research?symbol=ALC`** | Historical V4 Baseline Table | Complete comparison table with Net Debt Δ, Price Δ, IRR Δ (bps) | **PASS** |
| **`/research?symbol=ALC`** | Audited Recurring Sales | 88.37% non-equipment sales certified ($9,119M / $10,319M) | **PASS** |
| **`/research?symbol=ALC`** | Statutory Buyback Accordion | Approved 2026 Program: $1,500M CHF (AGM Authorized May 5, 2026) | **PASS** |
| **`/research?symbol=ALC`** | Dated Cash Flows Accordion | 7 discrete dated events across 1,826 days | **PASS** |
| **`/stock/alc`** | Primary Stock Route | ALC quote, overview, and navigation link to research dossier | **PASS** |
| **`/research?symbol=FIX`** | Quality Watch Regression | Existing Quality Watch portfolio asset loads cleanly without regression | **PASS** |
| **All Routes** | Console & Hydration Integrity | **0 console errors, 0 hydration mismatches** | **PASS** |

---

## 4. Preserved Production Governance Invariants

```ini
MODEL_TUNING = FROZEN
LIVE_ORDER_EXECUTION = PROHIBITED
UNRELATED_BACKEND_DEPLOYMENT = PROHIBITED
HUMAN_RESEARCH = NOT_AUTHORIZED
HVA_ALGORITHM_ACCEPTANCE = NOT_AUTHORIZED
RESEARCH_EVIDENCE_MANIPULATION = PROHIBITED
UNQUALIFIED_PRODUCTION_CHANGE = PROHIBITED

ALCON_V5_NET_DEBT_POLICY = SCENARIO_DERIVED_FROM_CASH_FLOWS
ALCON_V4_HISTORICAL_BASELINE = IMMUTABLE_FROZEN
SEC_FILER_CIK_ALC = 0001167379
CLOUDFLARE_PAGES_AUTO_DEPLOY = DISABLED
PRODUCTION_DEPLOYMENT_METHOD = WRANGLER_STATIC_OUTPUT_EXACT_SHA
RAILWAY_BACKEND_DEPLOYMENT = 059f9d79-6848-4095-aa61-5936ac3ee650 (PRESERVED)
```

---

## 5. Smart Money Final Provisioning Safety Gate Evidence (2026-10-25)

**Mainline SHA**: `54b9b7e846e879097c87ead45da19d9d87fcefa8`  
**Isolated Remediation Branch**: `fix/smart-money-provisioning-safety-gates`  
**Verdict**: **`SMART_MONEY_FINAL_PROVISIONING_SAFETY_PASS`**

### Gate 1 — Atomic Duplicate Prevention Under Advisory Lock
1. **Transaction Ordering Remediation**:
   - Previously, pre-ingestion backup & R2 upload ran at Steps 2 & 3, prior to claiming the advisory lock at Step 5.
   - Remediated ordering:
     - **Step 1**: Preflight check (`_step1_preflight`).
     - **Step 2**: Session advisory lock claim (`_step2_claim_advisory_lock`) acquiring exclusive `LOCK_SEC_EGRESS_OWNER` (`401007311509093371`).
     - **Step 3**: Atomic durable date eligibility check (`_step3_check_durable_date_eligibility_under_lock`) under the held lock.
     - **Step 4**: Pre-ingestion backup & R2 upload (`_step4_pre_backup`).
     - **Step 5**: Backup freshness gate verification (`_step5_freshness_gate_check`).
     - **Step 6**: SEC Form 4 ingestion (`_step6_ingest_form4_filings`).
2. **Atomic Invariant Enforced**:
   - If another worker holds the lock, incoming trigger fails closed immediately with `RuntimeError: CONCURRENT_INGESTION_BLOCKED`.
   - If execution already occurred for today's New York calendar date (`data_as_of` in latest active snapshot matches today), pipeline exits with `SKIPPED_ALREADY_EXECUTED_TODAY` without creating pre-backups or contacting SEC EDGAR.
   - Lock is released cleanly in `finally:` block across all normal and error exit paths.

### Gate 2 — R2 Backup Digest Integrity
1. **Elimination of ETag = SHA-256 Assumption**:
   - In AWS S3 / Cloudflare R2, ETag is an MD5 hash (for single uploads) or `<MD5>-<part_count>` (for multipart uploads), never SHA-256.
   - `S3CompatibleStorageAdapter.upload_file()` now executes:
     1. Remote content length verification (`uploaded_size == source_size`).
     2. Remote user-metadata `sha256` header validation against local source SHA-256.
     3. Independent streaming content readback (`s3.get_object()['Body']`) calculating true SHA-256 byte digest.
     4. Detection of multipart uploads (`is_multipart = True`) without confusing ETag with digest.
2. **Fail-Closed Gate**:
   - If metadata or streaming digest mismatches, adapter raises `IOError(R2_UPLOAD_CONTENT_CORRUPTION)` or `IOError(R2_UPLOAD_METADATA_CORRUPTION)`.
   - Pre-backup aborts at Step 4; Step 6 (SEC Form 4 ingestion) is never reached.

### Gate 3 — Non-Destructive Emergency Stop Hierarchy
1. **Disproof of Destructive Runbook Documentation**:
   - Schema audit of `scanner_active_snapshots` proves columns are `(scan_type, active_snapshot_id, published_at, published_by)`.
   - There is NO `is_active` or `scanner_id` column. Executing `UPDATE scanner_active_snapshots SET is_active = FALSE` throws `psycopg2.errors.UndefinedColumn`.
   - Executing `DELETE FROM scanner_active_snapshots` causes an immediate public outage on `/radar/` and `/api/radar/smart-money`.
2. **Authoritative Non-Destructive Stop Hierarchy**:
   - **Level 1 (Worker Kill Switch)**: Set `ARX_SEC_EGRESS_ROLE=DISABLED`. Halts any cycle in $\le 1.0\text{ s}$ at Step 1 (`_step1_preflight`) with `UNAUTHORIZED_WORKER_ROLE`.
   - **Level 2 (Process Scaling)**: Scale Railway worker replica count to 0 or pause worker service.
   - **Level 3 (Database Containment)**: `REVOKE INSERT, UPDATE ON form4_raw_filings, form4_parsed_transactions, scanner_snapshots, scanner_active_snapshots FROM smart_money_ingestion;`.
3. **Availability Invariant**:
   - Across all 3 levels, the active pointer in `scanner_active_snapshots` remains unchanged, preserving serving snapshot `SNAP_SM_1792939989_48f30db6` with 100% read-only Radar uptime via `smart_money_web`.

### Test Suite Evidence
- **Dedicated Provisioning Test Suite**: `tests/test_smart_money_provisioning_safety_gates.py` (12/12 passed in 2.57s).
- **Consolidated Smart Money Test Suite**: 55/55 passed in 13.31s (`test_smart_money_provisioning_safety_gates.py`, `test_smart_money_continuous_ingestion.py`, `test_smart_money_radar_contract.py`, `test_radar_scanner_pipeline.py`).
- **Linter & Syntax Verification**: flake8 passed (0 errors with `--count --select=E9,F63,F7,F82`).