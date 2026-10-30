# ARX TERMINAL — DOCUMENTATION CONFLICT & CORRECTION LOG
## Substantive Inconsistency Identification, Resolution & Authoritative Baseline Log
**Review Date:** 29 October 2026  
**Governing Directive:** `MASTER DIRECTIVE ARX-DOC-2026-10-29 (Section 11)`  
**Document Authority:** `CANONICAL_ACTIVE / FAIL_CLOSED`  
**Review Status:** `READY_FOR_OWNER_REVIEW`  

---

## 1. Executive Summary & Resolution Methodology

During the repository-wide documentation audit, multiple substantive inconsistencies, outdated assumptions, and conflicting status claims were surfaced across legacy reports, sprint summaries, and architecture drafts. This log catalogs every substantive conflict identified, details the conflicting sources, and establishes the binding authoritative resolution grounded in cryptographic and git evidence.

```
┌────────────────────────────────────────────────────────────────────────┐
│               DOCUMENTATION CONFLICT RESOLUTION PROTOCOL               │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Git & Runtime Evidence Trumps Narrative Assertions                  │
│    (Live git rev-parse and deployment IDs override unverified claims)  │
│ 2. Fail-Closed Governance Prevents Premature Authorization Claims       │
│    (Features remain experimental/held unless signed receipts exist)    │
│ 3. Explicit Distinction of Horizons                                    │
│    (Research POC != Mainline Git != Production Cloudflare/Railway)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Master Conflict & Resolution Register

| Conflict ID | Domain / Area | Conflicting Sources | Nature of Contradiction | Authoritative Resolution | Governing Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`CONF-001`** | Mainline SHA Baseline | Legacy Sprint 4 notes vs. GitHub Mainline | Some sprint summaries referenced `29ead6e4df6c` or `83da285` as current mainline HEAD. | Mainline HEAD is strictly verified as `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6` (incorporating PR #129). | `git rev-parse origin/main`; commit timestamp `2026-10-29T21:07:54Z`. |
| **`CONF-002`** | PR #129 Status | Sizing design drafts vs. GitHub API | Design notes described RQ-05 / WP-02 as an open proposal or unmerged draft. | PR #129 is formally `MERGED` into mainline (`e10de3b`). Verified equity provenance and sizing isolation are canonical. | GitHub PR #129 merge receipt; 18 passing adversarial test suites. |
| **`CONF-003`** | Sprint 5 Authorization | Prospective roadmap memos vs. Canonical Backlog | Certain roadmap slides referenced Sprint 5 as "active" or "in execution". | Sprint 5 is strictly `NOT_AUTHORIZED`. Sprint 4 is closed administratively; `sprint5_execution_authorized = false`. | `docs/planning/arx_canonical_backlog.json`; backlog integrity validator. |
| **`CONF-004`** | Market-Wide Architecture | Early MW-004 drafts vs. MW-004H Ratification | Early research memos debated Model E+ vs. F1 or suggested building a new persistent opportunity engine. | Model E+ canonical security-centric core with derived F1 opportunity projections is formally ratified (`FUNCTIONALLY_EQUIVALENT_WITHIN_TESTED_SCOPE`). | Product Owner Ratification Directive ARX-MW-004H; Candidate `f154032`. |
| **`CONF-005`** | Scoring Defect Status | Early MW-004 empirical tables vs. MW-004B Audit | Historical memos reported asset confluence scores clustering around 65.0 as valid market findings. | Confirmed as a silent deserialization fallback defect (`confluenceScore` vs `confluence_score`). Historical 65.0 findings are formally **falsified and superseded**. | Directive MW-004B forensic audit; `ScoreProvenance` enum implementation. |
| **`CONF-006`** | Market Coverage Arithmetic | External marketing claims vs. Database Reality | Claims that ARX Terminal screens "the entire US market of 13,000+ stocks" in real-time. | Out of 13,308 directory records, 5,792 are eligible common stocks, and the bounded research clean store contains **127 scannable equities (2.19% coverage)**. | Database audit of `asset_ohlcv_daily` (40,310 rows across 127 tickers). |
| **`CONF-007`** | PR #128 vs. Candidate Commit | Research PR comments vs. Git Tree | Some documents described PR #128 as containing the final, verified market-wide implementation. | PR #128 (`3fff681`) is strictly an open unmerged reference. The final candidate with 7 corrective fixes is commit `f154032` (Tree `30345ac`). | Git log and tree inspection in `finance-arx-market-wide-top35`. |
| **`CONF-008`** | Historical Replay Capability | Research strategy notes vs. Data Provenance | Strategy notes claimed exact point-in-time order replay of past SEC disclosures. | The store does not preserve microsecond SEC filing acceptance timestamps. Replay is strictly classified as `HISTORICAL_RECONSTRUCTION`, not `EXACT_ORIGINAL_REPLAY`. | Directive MW-004F findings; `FN-005` validation exception enforcement. |
| **`CONF-009`** | Synthesis E V3 Status | Frontend release notes vs. Production Site | Release memos described Synthesis E V3 as "launched" or "deployed". | Synthesis E V3 is `DESIGN_APPROVED` and implemented in PR #131 (Draft). Production Cloudflare frontend (`f74a0db2`) remains on pre-V3 build. | Live Cloudflare Pages readback; `production_deployments_enabled = false`. |
| **`CONF-010`** | Scanner Coverage Contract | Historical scanner docs vs. PR #110 Fix | Older API docs implied failed scans still report 100% complete universe coverage. | PR #110 and PR #127 established that a failed scan (`status = ERROR`) must strictly emit `coverage_status = ERROR` and `scan_coverage_pct = 0.0`. | PR #110 merge receipt; `resolve_coverage_envelope()` regression tests. |
| **`CONF-011`** | Workflow Filename | Various CI audit scripts vs. Repository File | References to `.github/workflows/release_qualification.yml` (underscore). | The canonical filename is `.github/workflows/release-qualification.yml` (hyphen). | File system inspection of `.github/workflows/`. |
| **`CONF-012`** | Smart Money Production State | Analytical feature descriptions vs. Governance | Product overviews described insider buying signals as active in production. | Smart Money is strictly held under `HOLD_SM1_001`. Production ingestion worker is disabled. | `api/routes/smart_money.py`; Railway background process inspection. |
| **`CONF-013`** | Position Sizer Fallback | Legacy UI docs vs. PR #129 Invariants | Legacy docs stated position sizer defaults to browser localStorage when offline. | PR #129 completely neutralized localStorage fallbacks. Sizing engine enforces fail-closed zero-size allocation if equity is unverified. | `governorSizingEngine.ts`; 18 adversarial unit tests. |

---

## 3. Residual Ambiguities Requiring Owner Adjudication

1. **Dormant Draft PR Cleanup:** Multiple draft PRs (#124, #103, #73, #47, #44, #40, #38) have been functionally superseded by newer unified PRs. Formal closure requires owner decision.
2. **WP-01 Workflow Push:** Local candidate `df38cf5` cannot be published via agent bot due to least-privilege boundaries and requires manual owner push.
