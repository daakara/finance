# ARX Terminal — Directive 058
## Baseline Integrity and Source Inventory Report (Gate G0)

**Document ID:** `ARX_058_BASELINE_AND_SOURCE_INVENTORY`  
**Directive:** `ARX_PRE_SPRINT5_OBLIGATION_RECOVERY_058`  
**Repository:** `daakara/finance`  
**Authoritative Mainline Base:** `6d4020ca3fef3ae07ac01204b03e0629b6a7cc79` (origin/main)  
**Execution Worktree:** `C:\Users\akara\Documents\Projects\finance-d058-obligation-recovery`  
**Execution Branch:** `gov/d058-pre-sprint5-obligation-recovery`  
**Gate G0 Verdict:** **PASS (Baseline Frozen & Verified)**  
**Date:** 2026-10-27  

---

## 1. Executive Summary & Baseline Freeze

Under Directive 058, ARX Terminal mandates the exhaustive discovery, tracking, and closure reconciliation of all outstanding Sprint 1–4 obligations before any Sprint 5 engineering kickoff. 

In accordance with Phase 0 (§3), the actual baseline was frozen and verified prior to any repository mutation:
1. **Authoritative Mainline SHA:** Remote `origin/main` was fetched and confirmed at commit `6d4020ca3fef3ae07ac01204b03e0629b6a7cc79` (Merge PR #117: Directive 053 — Smart Money truthfulness, strategy disclosures, and exchange metadata).
2. **Worktree Isolation:** An isolated git worktree was established at `C:\Users\akara\Documents\Projects\finance-d058-obligation-recovery` tracking `origin/main`. The uncommitted working tree in the primary repository (`C:\Users\akara\Documents\Projects\finance`) containing concurrent Directive 057 work was preserved without disturbance.
3. **Validator Baseline:** `tools/governance/validate_backlog_integrity.py` executed against `origin/main`, verifying rules BI-000 through BI-012 with **0 errors and 0 warnings**.
4. **Validator Test Suite Baseline:** All **59 of 59** unit and adversarial tests in `tests/test_backlog_integrity_validator.py` passed cleanly in 9.48 seconds.
5. **Production Provenance Reconciliation:** Live production environments (Cloudflare Pages edge and Railway containerized API) were probed and reconciled against build metadata and logs.

---

## 2. Production Deployment & Runtime Provenance

Authoritative live probing of the production infrastructure confirms current runtime status:

| Service / Layer | Platform / Host | Active Deployment ID | Source Build Commit | HTTP Status | Effective Runtime State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Edge Frontend** | Cloudflare Pages (`finance-xp8.pages.dev` / `www.arxterminal.com`) | `655b4653-d3e3-4205-84a0-f069d91e8fc7` | `92b2fb0d` / `6d4020c` | **200 OK** | 145/145 static routes compiled; CSP active; cross-mode layouts live |
| **API Backend** | Railway (`web-production-e370b.up.railway.app`) | `d1f8cf37-7592-4ad5-b986-4bbc9b4ed042` | `856170c` (PR #114) | **200 OK** | `/health` online; `/api/v1/analytics/AAPL` online; 0 backend code lag to `6d4020c` |

### Resolution of Prior Historical Discrepancies
- **Railway Source SHA:** Prior audit artifacts referenced `5efc2179` due to a historical SQLite table record (`activationRecord.release_sha`). As proven in Directive 055 §1.1, Railway container logs demonstrate that the active container was built on October 27, 2026 at 12:48:41 UTC from commit `856170c` (PR #114). The delta between `856170c` and `6d4020c` consists of commits `92b2fb0` (PR #115), `f62668c` (PR #116), and `6d4020c` (PR #117), which are 100% frontend UI and documentation updates. There is **zero backend code lag** between deployed production and `origin/main`.
- **PR #98 Status:** PR #98 (`feat/synthesis-e-v3-ux-parity-clean`) was previously release-held, but was formally merged into mainline at commit `8b8eb58296adbb6374be254cd9a00c36c59786c6` and deployed via PR #107 and subsequent frontend releases.
- **Directive 057 Status:** Branch `origin/fix/confluence-sizing-risk-remediation-057` exists at commit `1b3badac140821eced37dceed3963033fe7b7bbf` and has open PR #118. It has **not** been merged to `origin/main` and is **not** deployed to production.

---

## 3. GitHub PR & Issue Inventory

A live query of the GitHub repository `daakara/finance` identified **12 open pull requests**:

| PR # | State | Draft | Head Branch | Title | Relevant Scope / Target |
|:---:|:---:|:---:|:---|:---|:---|
| **#118** | OPEN | No | `fix/confluence-sizing-risk-remediation-057` | `fix(risk): controlled confluence disclosure, position sizing floor and modeled-risk qualification (D057)` | Remediates `DEF-056-001`, `002`, `003` (Candidate commit `1b3bada`) |
| **#109** | OPEN | Yes | `feat/sprint-control-system-v1` | `feat(governance): implement Sprint Control System (SCS-v1) and action tracker` | Non-activating governance tracker framework |
| **#103** | OPEN | No | `fix/gov001-ci-isolation` | `fix(governance): isolate test 21 from ambient git context in backlog integrity suite (GOV-001)` | Governance CI test isolation |
| **#83** | OPEN | Yes | `research/arx-hva01-v2-1-wave4c1-dual-tsa-local` | `feat(hva01): Wave 4C-1 dual-TSA timestamp verification [DRAFT]` | Associated with `ARX-S3-015` (Governance Held) |
| **#78** | OPEN | Yes | `research/arx-hva01-v2-1-wave4b-synthetic-witness` | `fix(hva01): Wave 4B-S restricted canonicalization and schema remediation [DRAFT]` | Associated with `ARX-S3-014` (Governance Held) |
| **#75** | OPEN | Yes | `research/arx-ux-instr-001-durability` | `research(arx): INSTR-001 evidence custody and synthetic observer validation` | Associated with `ARX-S3-008` (Governance Held) |
| **#73** | OPEN | Yes | `feat/radar-minervini-vcp-activation` | `Activate canonical Minervini VCP universe screening` | Associated with `ARX-S3-007` / `ARX-S4-RADAR-001` (Holding Queue) |
| **#47** | OPEN | Yes | `release/arx-sprint2-rc1-preparation` | `ARX Sprint 2 RC1 — Consolidated Integration & Nonproduction CI` | Superseded by RC-A foundation (PR #107) |
| **#44** | OPEN | Yes | `feat/sprint-2-s208-full-population-instrumentation` | `feat(universe): S2-08 full-population assessment outcome accounting and telemetry envelope` | Associated with `ARX-S2-008` (Sprint 5 candidate) |
| **#42** | OPEN | Yes | `fix/sprint2-publication-accounting-closure-semantics` | `feat(sprint2): governed publication, scanner membership and truthful Radar disclosure` | Associated with `ARX-S2-006` / `007` / `009` |
| **#40** | OPEN | Yes | `fix/production-candle-chart-visibility` | `fix(chart): restore candlestick chart visibility, ensure preview proxy parity, and guard legacy Render warmup` | Superseded by Lightweight Charts (PR #115) |
| **#38** | OPEN | Yes | `feat/mobile-research-discoverability` | `feat(mobile): direct Research access in mobile navigation dock` | Superseded by Synthesis E V3 (PR #98) |

---

## 4. Authoritative Canonical Backlog Baseline Census

Inspection of `docs/planning/arx_canonical_backlog.json` on `origin/main` establishes:
- **Total Canonical Tasks:** **43**
- **Schema Version:** `1.0.0`
- **Active Sprint:** `SPRINT_5` (Administrative planning designation; execution kickoff is unratified and blocked)
- **Holding Queue Tasks (Owner-Gated):** **10**
- **Distribution across Sprint Assignments:**
  - `SPRINT_1`: 4 tasks (`ARX-S1-001`, `ARX-S1-002`, `ARX-S1-004`, `ARX-S1-005`)
  - `SPRINT_2`: 5 tasks (`ARX-S2-003`, `ARX-S2-004`, `ARX-S2-014`, `ARX-S2-015`, `ARX-S2-016`)
  - `SPRINT_3`: 8 tasks (`ARX-S2-011`, `ARX-S2-012`, `ARX-S2-013`, `ARX-S3-001`, `ARX-S3-002`, `ARX-S3-011`, `ARX-S3-012`, `ARX-S3-017`)
  - `SPRINT_4`: 8 tasks (`ARX-S4-GOV-001`, `ARX-S4-ARCH-001`, `ARX-S4-ARCH-002`, `ARX-S1-006`, `ARX-S2-001`, `ARX-S3-010`, `ARX-S3-013`, `ARX-S3-016`)
  - `HOLDING_QUEUE_OWNER_GATED`: 10 tasks (`ARX-S1-003`, `ARX-S2-002`, `ARX-S2-006`, `ARX-S2-010`, `ARX-S3-003`, `ARX-S3-007`, `ARX-S3-008`, `ARX-S3-009`, `ARX-S3-014`, `ARX-S3-015`)
  - `SPRINT_5`: 8 tasks (`ARX-S4-RADAR-001`, `ARX-S2-005`, `ARX-S2-007`, `ARX-S2-008`, `ARX-S2-009`, `ARX-S3-004`, `ARX-S3-005`, `ARX-S3-006`)

### Reconciliation with Directive 054 V3 (52 Audited Units)
The 52 audited release units in Directive 054 V3 comprise:
- **43 Canonical Backlog Tasks**
- **9 Post-Sprint 4 Closure Mainline PRs:**
  1. `PR #107`: ARX Sprint 4 RC-A Core Foundation & Governance (`de5d246`)
  2. `PR #110`: Scanner Status & Coverage Metadata Contract (`4e0979e`)
  3. `PR #111`: Formal Sprint 4 Administrative Closure (`4a44325`)
  4. `PR #112`: Invalidate Stale Radar Asset Selection on Search Intent Change (`7f27ae1`)
  5. `PR #113`: Pre-Flight Macro VIX Authority & Catalyst Parity (Directive 048, `2e3fa18`)
  6. `PR #114`: Cross-Mode Candlestick Integrity Initial (`856170c`, superseded by PR #115)
  7. `PR #115`: Lightweight Charts Multi-Timeframe Candlestick Engine (Directive 049B, `92b2fb0`)
  8. `PR #116`: Cross-Mode Decision Parity & Layout Harmonization (Directive 051, `f62668c`)
  9. `PR #117`: Smart Money Truthfulness, Disclosures & Exchange Metadata (Directive 053, `6d4020c`)

Total Units = $43 + 9 = 52$. Zero untracked historical units.

---

## 5. Governance Containment & Hold Baseline

All platform governance fences and containment flags are verified intact on `origin/main`:
- `VCP_NATURAL_SCHEDULER_ENABLED=0` (Background scanner execution strictly disabled)
- `SMART_MONEY_WORKER_ACTIVATION=NOT_AUTHORIZED` (Live Smart Money daily ingestion disabled)
- `OPTIONS_INGESTION_ENGINE=OFFLINE` (CBOE options flow ingestion held)
- `SEC_FORM4_WORKER=HOLD` (SEC Form 4 scraper held dormant)
- `HVA_HUMAN_EVALUATION=LOCKED` (Human protocol participation locked)
- `EXTERNAL_CUSTODY_SIGSTORE=HOLD` (Sigstore verification unratified)

---

## 6. Baseline Verification Checkpoint (Gate G0)

- [x] Remote `origin/main` fetched and SHA confirmed: `6d4020ca3fef3ae07ac01204b03e0629b6a7cc79`
- [x] Working tree clean and isolated from active feature branches
- [x] All 12 open GitHub PRs identified and classified
- [x] Canonical backlog (43 tasks) verified by validator (BI-000 to BI-012 PASS)
- [x] All 59 validator tests passing
- [x] Cloudflare and Railway deployment identities verified
- [x] Directive 054 V3 census (52 units) and Directive 055-057 findings reconciled
- [x] Gate G0 status: **PASS**
