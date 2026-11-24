# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-24`  
**Effective Date:** 2026-11-24  
**Authoritative Production Baseline:** Cloudflare Pages `f96c89d2-08e8-410e-a23c-e3c684f9f76c` | Candidate `6419213f56695cbc0a5fa6dc08b10e0496f37afa`  
**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_17.md`

---

## 1. Executive Summary & Purpose

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository as of November 24, 2026. It incorporates the completion and qualification of:
1. **Dynamic Asset Discovery & Provider Fault Isolation (PR #323):** Company-name autocomplete/resolution across all discovery surfaces, query submission gating, and custom ticker launch suppression.
2. **Epistemic Tri-State Asset Architecture & Decoupled Limited-History Presentation (Candidate `6419213f`):** Separation of unknown (404), limited-history (422), and seasoned (200) assets, decoupling Security Master listing identity from SEC filing coverage, and codifying the 15-session baseline modeling gate.
3. **Decoupled SEC Form 4 Presentation (PR #319):** Separates stored filing disclosures from held scoring eligibility.
4. **Sourced Fundamentals & Offline Reader Integration (PR #318):** Anti-masking reader with cryptographic hash/issuer/accession verification.

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Verified Production Runtime Baseline

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **Cloudflare Pages Frontend** | Deployment `f96c89d2-08e8-410e-a23c-e3c684f9f76c` (Candidate `6419213f56695cbc0a5fa6dc08b10e0496f37afa`) | Cloudflare Pages (`arxterminal.com`) | Live production deployment. Dynamic company search, submission gating, 404 unavailable disclosures, decoupled limited-history presentation (`Listing Identity: Verified` / `SEC Filing Coverage: Not verified`), and NVDA baseline verified via live Playwright audit (Receipt: `ARX-ANT-ANALYSIS-EXPERIENCE-QUALIFICATION-RECEIPT.md`). |
| **Release Candidate Manifest** | Candidate `6419213f56695cbc0a5fa6dc08b10e0496f37afa` (`scripts/release/release_manifest.json` @ `71c0cb98`) | GitHub `feat/dynamic-asset-discovery` | 13/13 Release Qualification controls (RQ-01..13) 100% PASS in 100.33s. Qualification receipt committed (`2f4294db`). |
| **Railway Backend** | Commit `308f5392d7518a95bf4170ec5c1beaf64d2cba0b` | Railway `analyst-dashboard` & `smart-money-worker` | Backend release serving PR #305 (`web` deployment `bb115266`, `smart-money-worker` deployment `94eb27f0`). Worker in strict `STANDBY` mode (`ARX_SEC_EGRESS_ROLE=DISABLED`), 0 locks, 0 active queue rows. |
| **Railway Database** | PostgreSQL 16.x (port 5432) | Railway `postgres` | Holds 16 of 16 NVDA Form 4 accessions in the frozen August 18 – November 16, 2026 baseline (100% coverage) with Parker amendment linked. |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 3. Pull Requests & Engineering Workstream Status

### 3.1 Recently Completed & Qualified Pull Requests

| PR # | Branch / Title | Scope & Artifacts | Qualification Verdict & Status |
| :--- | :--- | :--- | :--- |
| **#323** | `feat/dynamic-asset-discovery`<br>`feat(search): dynamic company-name asset discovery and decoupled limited-history presentation` | Dynamic symbol and company-name resolution via edge proxy `/api/assets/search`, query submission gating while discovery is in-flight, suppression of custom symbol launch buttons when provider matches exist, 404 `asset-not-found-state`, 422 `insufficient-history-state`, decoupled capability chips (`Listing Identity: Verified` / `SEC Filing Coverage: Not verified`), clarified 15-session baseline modeling gate. Candidate `6419213f`. | **QUALIFIED & DEPLOYED:**<br>13/13 RQ controls PASS (100.33s). Deployed to Cloudflare Pages `f96c89d2`. 3/3 live audit checks PASS. Receipt: `ARX-ANT-ANALYSIS-EXPERIENCE-QUALIFICATION-RECEIPT.md`. |
| **#319** | `fix/sec-insider-flow-presentation`<br>`fix(confluence): decouple SEC filing availability from held scoring` | Decoupled stored Form 4 disclosure metadata and transaction counts from held scoring; historical coverage and uncertified indicators separated from issuer support. | **QUALIFIED:**<br>13/13 RQ controls PASS (Receipt `13dbd552`). Candidate `d3c611c4`. |
| **#318** | `fix/fundamental-evidence-diagnostics`<br>`feat(fundamentals): integrate offline reader with verified hash/issuer/accession checks` | Offline fundamentals reader (`offline_reader.py`) with cryptographic hash verification, anti-masking protection, and NVDA acquisition tooling. | **QUALIFIED:**<br>13/13 RQ controls PASS (Receipt `e375ad69`). Candidate `0d0c9044`. |
| **#315** | `fix/backup-retention-bounded-verification-001`<br>`fix(ops): bounded backup retention qualification` | Bounded backup retention qualification with fail-closed R2 integrity. Candidate `1cd1a0cf`. | **QUALIFIED:**<br>13/13 RQ controls PASS. |
| **#313 / #314**| `feat/sec-disclosure-evidence-view`<br>`feat(sec): SEC Form 4 disclosure evidence view` | Consolidated disclosure view for Form 4 filings with immutable audit event trail. Candidate `b3f3a930`. | **QUALIFIED:**<br>13/13 RQ controls PASS. |
| **#312** | `release/production`<br>`fix(sec): dispatch durable ingestion-only trial slots` | Supervised slot execution and verification for trial. Commit `868d7afc`. | **QUALIFIED & MERGED.** |
| **#309** | `feat/sec-aapl-nvda-pair-wrapper`<br>`gov(release): record release qualification receipts for PR 309` | Sequential pair wrapper qualification for AAPL/NVDA ingestion trial. Candidate `807e6d57`. | **QUALIFIED:**<br>13/13 RQ controls PASS. |
| **#305** | `fix/sec-ingestion-historical-coverage-amendments`<br>`fix(sec): issuer-bound historical coverage and amendment-effect classification` | Atomic Step 7 commit and read-only PIT historical snapshot extraction under repeatable-read isolation. Commit `f1f8b674`. | **MERGED & DEPLOYED.** |
| **#304** | `fix(sec): qualify atomic ingestion and historical point-in-time snapshots` | Railway backend deployment `c6e3c03b`. | **MERGED & DEPLOYED.** |
| **#303** | `fix(catalysts): disentangle NVDA results release from conference call` | Cloudflare Pages deployment `933295cc`. | **MERGED & DEPLOYED.** |
| **#302** | `fix(preflight): use the canonical second profit target` | Target 2 alignment in `PreFlightChecklistInputs`. Commit `7a0f7676`. | **MERGED & DEPLOYED.** |

---

## 4. Operational Holds Summary

All six operational holds remain in active, fail-closed enforcement:

1. **`HOLD_SEC_001` (SEC Form 4 Continuous Ingestion):** Worker in `STANDBY` (`ARX_SEC_EGRESS_ROLE=DISABLED`), 0 locks, 0 active queue rows. Continuous background polling is strictly held.
2. **`HOLD_SCAN_001` (Market Opportunity Radar):** Schedulers disabled; serves static/published single-universe snapshots (250 evaluated, 100 Stage 2).
3. **`HOLD_EXEC_001` (Live Execution & Capital Commitment):** Zero live broker API credentials; all orders route to simulated execution with fail-closed sizing (`INV-28`).
4. **`PHASE_3_HOLD` (Autonomous Trading & Sprint 5 Activation):** Locked (`sprint5_execution_authorized = false`).
5. **`ARX_CONFLUENCE_INTEGRITY_HOLD` (Automated Scoring Promotion):** Held pending complete multi-pillar decision trace review.
6. **`PHASE_A3_UX_HOLD` (Progressive Disclosure Redesign PR #274):** Held pending authentic 5-participant real-human formative evaluation.
