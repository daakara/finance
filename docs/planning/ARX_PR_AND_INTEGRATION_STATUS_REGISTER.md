# ARX TERMINAL — PULL REQUEST & INTEGRATION STATUS REGISTER
## Comprehensive Accounting of Merged, Open, Draft & Candidate Branches
**Review Date:** 29 October 2026 (22:30 UTC+1)  
**Mainline Anchor SHA:** `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Authentication & Verification:** GitHub App JWT via `scripts/scratch_query_prs.py`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary of Pull Request Landscape

As of 29 October 2026, the ARX Terminal GitHub repository (`daakara/finance`) contains 16 open pull requests and 113 merged pull requests. This register provides the authoritative status, base, head SHA, purpose, and integration dependencies for all active and historically significant PRs.

```
[ Mainline origin/main (e10de3b) ]
  ├── [MERGED] PR #129 (fix(portfolio): verified equity provenance & sizing isolation)
  ├── [MERGED] PR #126 (chore(governance): administrative backlog reconciliation)
  ├── [MERGED] PR #125 (fix(governance): validator hardening against ADV-04/ADV-05)
  ├── [MERGED] PR #123 (fix(ux): Paper Light surfaces & unified theme persistence)
  └── [MERGED] PR #121 (feat(ci): release qualification framework & hosted PR identity)

[ Open Draft PR Queue — Pending Owner Review & Release Staging ]
  ├── PR #131 (codex/ux-v3-final-completion-20261029) ──> Synthesis E V3 UX Completion (Hosted CI PASS)
  ├── PR #130 (remediation/client-api-standardization) ──> Client API Standardization & Edge Routing
  ├── PR #128 (poc/arx-market-wide-top35) ───────────────> Market-Wide Discovery POC (Reference Open)
  ├── PR #127 (fix/d058b-radar-reliability) ─────────────> Radar Operational Truth & State Model
  └── PR #124 (ux-v3 intermediate) ─────────────────────> Superseded by PR #131
```

---

## 2. Exhaustive Active PR Register (Open & In-Flight PRs)

| PR # | State | Title | Head Branch | Head Commit SHA | Base Lineage | Hosted CI Status | Integration Purpose & Next Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#131** | `DRAFT OPEN` | `feat(ux): Synthesis E V3 terminal completion and responsive modes` | `codex/ux-v3-final-completion-20261029` | `c6ca21f0c133d57eec5ac97d160455a59730712e` | `29ead6e4df6c` | `PASSED` (`40419208311`) | Implements Guided/Standard/Quant modes, Paper Light theme, responsive layouts, accessibility. Ready for owner visual sign-off. |
| **#130** | `DRAFT OPEN` | `fix(api): client API standardization and edge gateway proxy routing` | `remediation/client-api-standardization` | `d47e858d131917c401b2bca1280daa422fcacb1a` | `29ead6e4df6c` | `PASSED` | Introduces `getApiBaseUrl()` dynamic client resolution, `/api/backend/*` Cloudflare edge proxy, and error envelopes. |
| **#128** | `DRAFT OPEN` | `feat(poc): market-wide top 35 opportunity discovery engine` | `poc/arx-market-wide-top35` | `3fff68108593f304275cdf18b696c8697f5b8257` | `29ead6e4df6c` | `PASSED` | Retained as open, unmerged research reference branch. Final candidate `f154032` preserved in research worktree. |
| **#127** | `DRAFT OPEN` | `fix(radar): radar operational truth, state modeling and provenance telemetry` | `fix/d058b-radar-reliability-and-provenance` | `fd9a13868a027e8e699ef3cad9ecffcac305ecd3` | `29ead6e4df6c` | `PASSED` | Resolves Radar pending/empty states, prevents false 100% complete coverage on error, and adds telemetry. |
| **#124** | `DRAFT OPEN` | `feat(ux): synthesis e v3 preliminary architecture` | `feature/synthesis-e-v3-prelim` | `f7da12755a3fcc02b5af68898b4027398e93851c` | `29ead6e4df6c` | `PASSED` | Superseded by PR #131. Retained as open reference pending formal owner closure. |
| **#109** | `DRAFT OPEN` | `chore(deps): bump dependencies` | `dependabot/npm_and_yarn/all` | `0853fe6c41b8` | `8f141c1a6bda` | `PENDING` | Automated dependabot dependency bump. Gated under dependency audit. |
| **#103** | `DRAFT OPEN` | `fix(engine): historical scanner telemetry` | `fix/scanner-telemetry-s4` | `482fa8d39c91` | `de5d2463cc4c` | `PASSED` | Historical patch superseded by PR #110 and PR #127. |
| **#83** | `DRAFT OPEN` | `feat(vcp): holdout epoch 002 pattern fixtures` | `feat/vcp-epoch-002` | `e2a47291a182` | `29ead6e4df6c` | `PASSED` | Custodial test fixtures for Minervini VCP validation. Held under `HOLD_CUSTODY`. |
| **#78** | `DRAFT OPEN` | `feat(smart-money): SEC Form 4 historical parser`| `feat/smart-money-sec` | `5c89104b2c17` | `29ead6e4df6c` | `PASSED` | SEC Form 4 parser implementation. Held under `HOLD_SM1_001`. |
| **#75** | `DRAFT OPEN` | `feat(events): corporate earnings calendar prototype` | `prototype/corporate-events`| `93b1d092c481` | `29ead6e4df6c` | `PASSED` | Corporate events prototype. Held under `HOLD_SEC_001`. |
| **#73** | `DRAFT OPEN` | `fix(radar): initial radar state remediation` | `remediation/radar-state-initial` | `a104f9812b48`| `29ead6e4df6c` | `PASSED` | Early radar patch superseded by PR #127. |
| **#47** | `DRAFT OPEN` | `feat(analytics): alternative macro indicators` | `research/macro-indicators` | `6b901fa49c12` | `83da285269ef` | `PASSED` | Experimental macro indicators. Quarantined in research. |
| **#44** | `DRAFT OPEN` | `fix(ux): mobile drawer swipe gestures` | `fix/mobile-drawer-swipe` | `2c9041a78b54` | `83da285269ef` | `PASSED` | Incorporated into PR #131. |
| **#42** | `DRAFT OPEN` | `feat(quant): Cornish-Fisher modified VaR` | `feat/quant-cf-var` | `51082c9f8a34` | `83da285269ef` | `PASSED` | Merged into `engines/risk_engine.py` via Sprint 3; branch open reference. |
| **#40** | `DRAFT OPEN` | `fix(chart): dual-horizon candlestick rendering` | `fix/chart-candlestick-render` | `7a83d91c2b54` | `83da285269ef` | `PASSED` | Merged into mainline via direct delivery; branch open reference. |
| **#38** | `DRAFT OPEN` | `chore(docs): sprint 2 package b closeout notes` | `docs/sprint2-package-b` | `33b91a82c401` | `83da285269ef` | `PASSED` | Historical documentation draft. |

---

## 3. Recently Merged Mainline PRs (Sprint 4 Milestones)

- **PR #129 (`e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`):** Merged 2026-10-29T21:07:55Z. Resolves RQ-05 / WP-02. Introduces `PortfolioSummary` provenance contract, quarantines untrusted localStorage equity injections, and enforces fail-closed zero-size position allocation when account provenance is unverified.
- **PR #126 (`29ead6e4df6cbf4e1e611feb74d61f0370b80e53`):** Reconciles administrative backlog and isolates historical test fixtures.
- **PR #125 (`6b7a563`):** Hardens validator against cross-baseline authority mutation (ADV-04) and task deletion (ADV-05).
- **PR #123 (`3e0f8dc`):** Introduces high-contrast Paper Light surface theming and unified theme persistence.
- **PR #121 (`2c900cd`):** Formalizes release qualification framework and hosted PR identity contract (Directive 074-A4.1).

---

## 4. Multi-PR Combined Candidate (D4 Integration)

- **Candidate Commit:** `d76a9b900c51199200062be47525b18de47895d5`
- **Candidate Tree SHA:** `8b91aff95c05bfcf8c6075331b360453908c3a24`
- **Included PR Scopes:**
  - PR #127 (`fd9a138`) — Radar reliability and provenance.
  - PR #130 (`d47e858`) — Client API standardization.
  - PR #131 (`c6ca21f`) — Synthesis E V3 UX completion.
- **Base Anchor:** `29ead6e4df6cbf4e1e611feb74d61f0370b80e53`.
- **Integration Status:** Locally verified with 100% build pass; pending rebase onto merged PR #129 (`e10de3b`).

---

## 5. Work Package 01 (WP-01) Local Candidate

- **Candidate Commit:** `df38cf52b41789c4644421da596e0f9d7f16ec7c`
- **Scope:** Formalization of `.github/workflows/release-qualification.yml` and hosted PR identity contract.
- **Authority Status:** `OWNER_PUBLICATION_PENDING`. Requires manual Product Owner push due to GitHub App `workflows: read-only` security boundary.
