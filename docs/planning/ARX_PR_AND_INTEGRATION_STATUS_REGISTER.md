# ARX TERMINAL — PULL REQUEST & INTEGRATION STATUS REGISTER
## Comprehensive Accounting of Merged, Open, Draft & Candidate Branches
**Review Date:** 29 October 2026 (22:30 UTC+1)  
**Mainline Anchor SHA:** `09c2112e5e9f415b3dec5bea9cb2c85311fb22dd`  
**Authentication & Verification:** GitHub App JWT via `scripts/scratch_query_prs.py`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary of Pull Request Landscape

As of 29 October 2026, the ARX Terminal GitHub repository (`daakara/finance`) contains 16 open pull requests and 113 merged pull requests. This register provides the authoritative status, base, head SHA, purpose, and integration dependencies for all active and historically significant PRs.

```
[ Mainline origin/main (09c2112) ]
  ├── [MERGED] PR #134 (chore(release): cumulative release manifest reconciliation & RQ-02 hardening)
  ├── [MERGED] PR #133 (fix(radar): fail-closed contract normalization & decision continuity)
  ├── [MERGED] PR #127 (fix(radar): error-versus-zero semantics & candidate truth contracts)
  ├── [MERGED] PR #131 (feat(ux): Synthesis E V3 completion - RC2 review containment)
  ├── [MERGED] PR #130 (fix(api): client API standardization and edge gateway proxy routing)
  ├── [MERGED] PR #129 (fix(portfolio): verified equity provenance & sizing isolation)
  ├── [MERGED] PR #126 (chore(governance): administrative backlog reconciliation)
  ├── [MERGED] PR #125 (fix(governance): validator hardening against ADV-04/ADV-05)
  ├── [MERGED] PR #123 (fix(ux): Paper Light surfaces & unified theme persistence)
  └── [MERGED] PR #121 (feat(ci): release qualification framework & hosted PR identity)

[ Open Draft PR Queue — Pending Owner Review & Release Staging ]
  ├── PR #135 (chore/sprint4-closure-remediation) ───────> Sprint 4 Final Administrative Closure Remediation
  ├── PR #128 (poc/arx-market-wide-top35) ───────────────> Market-Wide Discovery POC (Reference Open)
  └── PR #124 (ux-v3 intermediate) ─────────────────────> Superseded by PR #131
```

---

## 2. Exhaustive Active PR Register (Open & In-Flight PRs)

| PR # | State | Title | Head Branch | Head Commit SHA | Base Lineage | Hosted CI Status | Integration Purpose & Next Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#142** | `DRAFT OPEN` | `docs(governance): RQ-02 independent review and manifest issuance enforcement proposal` | `gov/rq02-independent-review-enforcement-proposal-20261030` | `e3937c963768d76d1d512df17e3bdea332ed9d38` | `main` | `PASSED` | Explicitly deferred work / non-blocking governance improvement. Requires external reviewer/FIDO2 infrastructure (deferred risk accepted under interim solo-maintainer governance). |
| **#141** | `DRAFT OPEN` | `gov(rq02): prototype base-anchored PR differential qualification and negative controls` | `gov/pr-scoped-rq02-qualification-20261030` | `3b786655dd0c6a33de4e1e5d7b0d5e8de1c53391` | `main` | `PASSED` | Superseded proposal. Exploratory differential qualification controls reconciled into mainline and Candidate 8dbf142 via PR #144. |
| **#131** | `MERGED` | `feat(ux): Synthesis E V3 completion — RC2 review containment` | `codex/ux-v3-final-completion-20261029` | `882d42495c5eb4b20d9e0cb75edf0a38f0695b1a` | `main` | `PASSED` | Merged into main 2026-10-30T03:33:43Z. Implements Guided/Standard/Quant modes, Paper Light theme, responsive layouts, accessibility. |
| **#130** | `MERGED` | `feat(api): client API standardization and edge gateway proxy routing` | `remediation/client-api-standardization` | `d47e858d131917c401b2bca1280daa422fcacb1a` | `main` | `PASSED` | Merged into main 2026-10-30T00:54:19Z. Introduces `getApiBaseUrl()` dynamic client resolution, `/api/backend/*` Cloudflare edge proxy, and error envelopes. |
| **#128** | `DRAFT OPEN` | `feat(poc): market-wide top 35 opportunity discovery engine` | `poc/arx-market-wide-top35` | `3fff68108593f304275cdf18b696c8697f5b8257` | `29ead6e4df6c` | `PASSED` | Retained as open, unmerged research reference branch. Final candidate `f154032` preserved in research worktree. |
| **#127** | `MERGED` | `fix(radar): Directive 058B.1 Batch 2 — fail-closed error-versus-zero semantics, candidate truth contracts, and census reconciliation` | `fix/d058b-radar-reliability-and-provenance` | `1e1a1ace457a612e95f020148ab80eb5d80aa2f5` | `main (882d42495c5e)` | `PASSED` | Merged into main 2026-10-30T03:54:52Z (merge commit `38d3028608b8eaceed70f7437f1324774b9d60fa`). Resolves Radar pending/empty states and candidate truth contracts. |
| **#135** | `DRAFT OPEN` | `chore(governance): Sprint 4 final administrative closure remediation` | `chore/sprint4-closure-remediation` | `6a6fcdd0aaeb6a19d0dfdb2c4b1e5ee2b4167fe6` | `main (09c2112e5e9f)` | `PASSED` | Bounded administrative closure remediation for Sprint 4. |
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

- **PR #144 (`69422421a9c6af919c28ff940b121930799f58f3`):** Merged 2026-10-31T09:12:12Z. Reconciles release candidate manifest and RQ-02 qualification framework for Candidate 8dbf142.
- **PR #143 (`9f9cdb9eb7d724c8ea22573826e76cabd1d2944f`):** Merged into `release/production` 2026-10-31T09:13:48Z. Production release promotion of Candidate 8dbf142 (Cloudflare 810b1a73, Railway 6950deb1, 0 DDL migrations).
- **PR #140 (`8dbf142ac7400e0ccc2fcf9860217f0a2f1f683d`):** Merged 2026-10-30T17:15:00Z. Radar / Smart Money remediation (cross-engine score parity, truthful disclosures).
- **PR #139 (`379768fc009d0a790586e30b8c6a2e414c278c2e`):** Merged 2026-10-30T17:10:00Z. Approved UX redesign (execution ladder, Paper Light, responsive layouts).
- **PR #134 (`09c2112e5e9f415b3dec5bea9cb2c85311fb22dd`):** Merged 2026-10-30T11:26:11Z. Reconciles cumulative release manifest, enforces fail-closed RQ-02 diff integrity, and integrates permanent hosted D12C attack regression suite.
- **PR #133 (`f1ed7171e9b7031c9233f1290655ec0d7fa095af`):** Merged 2026-10-30T07:11:00Z. Finalizes Synthesis E V3 radar contracts and responsive mobile navigation drawer polish.
- **PR #127 (`38d3028608b8eaceed70f7437f1324774b9d60fa`):** Merged 2026-10-30T03:54:52Z. Directive 058B.1 Batch 2 fail-closed error-versus-zero semantics, candidate truth contracts, and census reconciliation.
- **PR #131 (`882d42495c5eb4b20d9e0cb75edf0a38f0695b1a`):** Merged 2026-10-30T03:33:43Z. Synthesis E V3 terminal completion, guided/standard/quant modes, and RC2 review containment.
- **PR #130 (`b413b03d3572bb7a57aaa71f7b4bcc30d13f1573`):** Merged 2026-10-29T22:48:01Z. Client API standardization, dynamic base URL resolution, and Cloudflare Pages edge proxy routing.
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
