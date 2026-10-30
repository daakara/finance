# ARX TERMINAL — MASTER PROJECT STATUS
## Integrated System State Across Research, Engineering, UX, Governance & Production
**Status As Of:** 29 October 2026 (22:30 UTC+1)  
**Mainline Anchor SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Mainline Tree SHA:** `dbbb15705335f9d0bd95473f520b241ccd31551e`  
**Governing Directive:** `MASTER DIRECTIVE ARX-DOC-2026-10-29`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Status Dashboard

| Dimension | Authoritative Status | Key Identity / Version | Operational Meaning & Boundary |
| :--- | :--- | :--- | :--- |
| **Mainline Git State** | `MERGED_INTEGRATED` | `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` | Contains PR #129 (WP-02 / RQ-05) verified equity provenance and sizing isolation. |
| **Production Backend (Railway)** | `DEPLOYED_STABLE` | Deployment `2fb9e82f-4b62-471e-abe2-63961a6c9c61`<br>Commit `de5d2463cc4c6503bd700bcaf2b1bc36fd5e03d9` | Active production backend. Clean health check, manual trigger required for updates. PR #129 not yet deployed to prod. |
| **Production Frontend (Cloudflare)** | `DEPLOYED_STABLE` | Deployment `f74a0db2` | `production_deployments_enabled = false`. Preview branch builds isolated. No automatic merge deployments. |
| **Market-Wide Research (MW-004)** | `DIRECTION_RATIFIED`<br>`LOCAL_CANDIDATE_FROZEN` | Candidate `f154032152710a51b209d80a4a21dfb655d312e4`<br>Tree `30345ac093a913e6a50c8d0a018d77d52083c010`<br>PR #128 Ref `3fff68108593f304275cdf18b696c8697f5b8257` | Model E+ canonical security-centric core with derived F1 opportunity projections ratified. Research quarantine under `/poc-market-wide`. Zero live execution authority. |
| **Synthesis E V3 UX** | `DESIGN_APPROVED`<br>`DRAFT_PR_OPEN` | PR #131 (`c6ca21f0c133d57eec5ac97d160455a59730712e`)<br>Paper Light Merged (`3e0f8dc`, PR #123) | Complete 3-mode UX (Guided, Standard, Quant). Hosted CI passing. Pending formal owner visual sign-off and staging deployment. |
| **Client API Standardization** | `DRAFT_PR_OPEN` | PR #130 (`d47e858d131917c401b2bca1280daa422fcacb1a`) | Dynamic API routing (`getApiBaseUrl`), edge `/api/backend/*` gateway proxy. |
| **Radar Reliability** | `DRAFT_PR_OPEN` | PR #127 (`fd9a13868a027e8e699ef3cad9ecffcac305ecd3`) | Radar state model, error/empty truth states, provenance telemetry. |
| **Governance Baseline** | `SPRINT_4_ADMIN_CLOSED`<br>`SPRINT_5_UNAUTHORIZED` | Backlog schema v2.1 | Sprint 4 closed administratively. Sprint 5 is `NOT_AUTHORIZED` (`sprint5_execution_authorized = false`). All 14 platform holds active. |

---

## 2. Multi-Horizon Environmental Separation

ARX Terminal enforces strict boundaries between its environments. Under no circumstances may code progress between these horizons without explicit cryptographic receipts and Product Owner authorization:

```
[ Research Worktree / POC ]  -->  [ Mainline Git PRs ]  -->  [ Release Qualification ]  -->  [ Production Deployment ]
  • Model E+ Candidate             • PR #129 (Merged)         • release-qualification.yml      • Railway Backend (de5d246)
  • /poc-market-wide               • PR #131 (Draft)          • Ruleset 25187391               • Cloudflare Pages (f74a0db)
  • Non-executable                 • PR #130 (Draft)          • Ruleset 26024894               • Auto-deploy: DISABLED
  • Historical reconstruction      • PR #127 (Draft)          • WP-01 (Pending Pub)            • Production Gate: FAIL-CLOSED
```

---

## 3. Pull Request Status & Integration Landscape

As verified via the authenticated GitHub App installation on 29 October 2026:

### Merged Production & Mainline Lineage
- **PR #129 (`e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`):** Merged at `2026-10-29T21:07:55Z`. Solves RQ-05 / WP-02 by introducing verified equity provenance (`PortfolioSummary.provenance_status`), quarantining untrusted localStorage mutations, and isolating risk governor sizing.
- **PR #126 (`29ead6e4df6cbf4e1e611feb74d61f0370b80e53`):** Administrative backlog reconciliation and historical test fixture isolation.
- **PR #125 (`6b7a563`):** Validator hardening against cross-baseline authority mutation (ADV-04) and task deletion (ADV-05).
- **PR #123 (`3e0f8dc`):** Paper Light surfaces and unified theme persistence.
- **PR #121 (`2c900cd`):** Release qualification framework and hosted PR identity contract.

### Open Active Draft PRs
- **PR #131 (`c6ca21f0c133d57eec5ac97d160455a59730712e`):** Synthesis E V3 UX completion (`codex/ux-v3-final-completion-20261029`). Draft open. Hosted CI passed.
- **PR #130 (`d47e858d131917c401b2bca1280daa422fcacb1a`):** Client API standardization (`remediation/client-api-standardization`). Draft open.
- **PR #128 (`3fff68108593f304275cdf18b696c8697f5b8257`):** Market-Wide Top 35 POC (`poc/arx-market-wide-top35`). Draft open. Maintained as an unmerged reference branch.
- **PR #127 (`fd9a13868a027e8e699ef3cad9ecffcac305ecd3`):** Radar reliability and provenance (`fix/d058b-radar-reliability-and-provenance`). Draft open.
- **PR #124 (`f7da12755a3fcc02b5af68898b4027398e93851c`):** Intermediate UX V3 draft (superseded by PR #131).

### Combined Integration Candidate (D4)
- **Candidate Commit:** `d76a9b900c51199200062be47525b18de47895d5`
- **Candidate Tree:** `8b91aff95c05bfcf8c6075331b360453908c3a24`
- **Scope:** Unifies PR #127, PR #130, and PR #131 on top of base commit `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`. Locally verified with zero build regressions; pending rebase onto merged PR #129 (`e10de3b`).

---

## 4. Governance & Operational Hold Register

All 14 platform security and execution holds remain strictly active:

| Hold ID | Title | Scope | Governing Invariant | Enforcement Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `HOLD_EXEC_001` | Live Financial Execution Hold | Trading / Broker Egress | INV-01, INV-02 | Hard rejection in broker adapter; simulated order state only. |
| `HOLD_SM1_001` | Smart Money Production Ingestion Hold | SEC Form 4 Parser | INV-05, INV-18 | Production background scheduler disabled; local replay only. |
| `HOLD_TUNE_001` | Model Tuning & Prospective Learning Hold | Neural/Quant Models | INV-14 | Static model parameters; prospective weight modification blocked. |
| `HOLD_SEC_001` | Live SEC EDGAR Egress Hold | Corporate Events | INV-22 | Network firewall; test fixtures only. |
| `HOLD_SCAN_001` | Full Market-Wide Scanner Daemon Hold | Scheduler / Cron | INV-09 | Scan daemons disabled in production Railway config. |
| `SPRINT5_HOLD` | Sprint 5 Execution Authorization Hold | Backlog Governance | SPRINT_5_AUTH = FALSE | Fail-closed validator rejects Sprint 5 task advancement. |

---

## 5. Architectural Verification & Summary of Evidence

1. **Portfolio Provenance (RQ-05):** Verified holdings are cryptographically and structurally separated from unverified paper holdings. Position sizing rejects unverified equity states (`NOT_LOADED`, `unverified_paper`, `unconfigured`) and enforces a zero-size allocation.
2. **Market-Wide Research Truth (MW-004):** The historical 65.0 score fallback bug has been documented and falsified. Population arithmetic is reconciled: 13,308 SEC directory records $\to$ 5,792 common equities $\to$ 127 stored scannable equities (2.19% stored coverage) $\to$ 41 Stage 1 rejections, 86 Stage 2 evaluations, 46 Cohort A qualified, 40 Cohort B developing, 35 displayed.
3. **CI/CD Security (RQ-12):** GitHub App operates under least privilege. Mainline protected by Ruleset `25187391`; production release branch protected by Ruleset `26024894`. Workflows are immutable to non-owner actors.
