# 📈 ARX Terminal | Institutional Financial Intelligence & Research Platform

> **Current operational reconciliation (2026-11-20, production SHA [`868d7afcd528006c2323e6b79b5a17f2f20f677c`](https://github.com/daakara/finance/commit/868d7afcd528006c2323e6b79b5a17f2f20f677c)):** See [Active work register (2026-11-20)](docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_20.md), [Integrated status addendum (2026-11-20)](docs/ARX_STATUS_ADDENDUM_2026_11_20.md), and [Operational holds register](docs/governance/ARX_OPERATIONAL_HOLDS_REGISTER.md). Release qualification and surgical post-launch releases through PR #312 (Catalyst card ticker sync, sector labels, FRED macro contract, SEC Form 4 presentation truth, v2.5.3 engine manifest, Radar evidence adapter, Pre-Flight freshness safeguards, Confluence drawer presentation, quote provenance & sizing authorization, canonical second profit target, earnings call time disentanglement, SEC atomic PIT snapshots, amendment classification, bounded one-shot NVDA executor, allowlisted AAPL canary, offline sequential pair qualification, fail-closed sequential canary wrapper with deterministic content preservation, crash-safe two-issuer ingestion-only trial qualification under PR #311, and durable trial slot dispatch under PR #312) are merged and deployed to production. Supervised one-shot canaries and candidate ingestions under Issue #165 are complete; production database holds 16 of 16 accessions in the frozen NVDA backfill baseline plus 6 AAPL Form 4 filings (including canary `0001140361-26-044773` and accession `0001140361-26-044952`) across 140 parsed transactions, with active snapshot `SNAP_SM_1792939989_48f30db6` intact. Authoritative superseding provenance correction v2 (`6c7bf075-0d62-4a95-9727-044515c3e376`) is verified in production. Separately, the two-issuer recurring ingestion trial `ARX-TRIAL-2026-NVDA-AAPL-001` is implemented, qualified, and Day 1 is armed (`ARX_SEC_EGRESS_ROLE=OWNER`, single-reading grant `dc58b294b7a744da9730c5a2b97725b1` expiring 2026-11-21T08:03:31Z, $\le 23\text{h}55\text{m}$, holding advisory lock `401007311509093371`) awaiting natural scheduled execution in Slot 1 (09:15 EST / 14:15 UTC); zero recurring trial cycles have completed. Operational status: `ARMED`. Documentation PR #310 merge is deferred to protect running worker continuity and prevent release-bound grant invalidation during the active trial window. All six operational holds (`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`) remain strictly active (with `HOLD_SEC_001` scoped for bounded 2-issuer ingestion-only trial, with snapshot publication and market-wide scanning held).


A professional-grade quantitative intelligence platform featuring real-time market surveillance, Mark Minervini Volatility Contraction Pattern (VCP) setup detection, dual-horizon candlestick geometry, multi-pillar Confluence scoring, and Cornish-Fisher Modified Value-at-Risk modeling.

**Live Deployment**: [https://www.arxterminal.com/](https://www.arxterminal.com/)  
**Mainline Anchor SHA**: `e10de3b2714e7c7cba49c51e0e7a4eb13a7033f6`  
**Governance Baseline**: Sprint 4 Administratively Closed | Sprint 5 `NOT_AUTHORIZED` | `HOLD_EXEC_001` Active

![Next.js](https://img.shields.io/badge/next.js-14.2-black.svg)
![React](https://img.shields.io/badge/react-18-cyan.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.110+-green.svg)
![Cloudflare Pages](https://img.shields.io/badge/deployed-cloudflare%20pages-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Current direction and status (2026-11-20)

For the current authoritative operational status across all five ARX engineering workstreams, see:

- [Canonical sprint roadmap (2026-11-20)](docs/planning/ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md) — six-sprint backlog prioritization (Sprint 0–5), multi-perspective Project Brain architecture, capacity allocation, and active execution packets.
- [Integrated status addendum — 2026-11-20](docs/ARX_STATUS_ADDENDUM_2026_11_20.md) — comprehensive reconciliation across Smart Money, Radar, Confluence, Pre-Flight Decision Authority, Frontend UX, and Infrastructure.
- [Active work register — 2026-11-20](docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_20.md) — complete inventory, deduplication, and status across all open issues, candidate PRs, and merged production releases.
- [Operational holds register](docs/governance/ARX_OPERATIONAL_HOLDS_REGISTER.md) — authoritative matrix of the 6 active fail-closed platform holds, enforcement, and exit criteria.
- [Radar investment research direction](docs/architecture/ARX_RADAR_INVESTMENT_RESEARCH_DIRECTION_2026_11_07.md) — product scope, existing rule owners, Stage 2/VCP separation, data quality, and wave sequencing.

The October 29 master status, November 7, November 14, and November 17 addenda, and historical release notes remain preserved as immutable historical records. All operational holds remain strictly in force.

---

## 🏛️ Authoritative Documentation & System Portal

The ARX Terminal documentation estate is consolidated into a canonical hierarchy. Refer to the master registers below for authoritative specifications, research decisions, and governance ledgers:

| Dimension | Canonical Documentation Portal | Primary Focus & Governance Scope |
| :--- | :--- | :--- |
| **Canonical Sprint Roadmap (2026-11-20)** | [`ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md`](docs/planning/ARX_CANONICAL_SPRINT_ROADMAP_2026_11_20.md) | Six-sprint execution plan (Sprint 0–5), capacity model, dependency graph, and Sprint 1 active packet. |
| **Operational Holds Register** | [`ARX_OPERATIONAL_HOLDS_REGISTER.md`](docs/governance/ARX_OPERATIONAL_HOLDS_REGISTER.md) | Authoritative matrix of 6 active fail-closed platform holds, live enforcement, and exit criteria. |
| **Active Work Register (2026-11-20)** | [`ARX_ACTIVE_WORK_REGISTER_2026_11_20.md`](docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_20.md) | Authoritative inventory, deduplication, and status across all open issues, open PRs, and merged releases. |
| **Integrated Status Addendum (2026-11-20)** | [`ARX_STATUS_ADDENDUM_2026_11_20.md`](docs/ARX_STATUS_ADDENDUM_2026_11_20.md) | Single source of operational truth across Smart Money, Radar, Confluence, Pre-Flight, UX, and DR. |
| **Integrated Project Status (2026-10-29)** | [`ARX_MASTER_PROJECT_STATUS_2026_10_29.md`](docs/ARX_MASTER_PROJECT_STATUS_2026_10_29.md) | Historical master status across research, UX, engines, engineering, and hosting status. |
| **Documentation Master Inventory** | [`ARX_DOCUMENTATION_MASTER_INVENTORY_2026_10_29.md`](docs/planning/ARX_DOCUMENTATION_MASTER_INVENTORY_2026_10_29.md) | Comprehensive audit and 8-tier authority classification of all 1,478 repository documents. |
| **Engine Architecture & Contracts** | [`ARX_ARCHITECTURE_AND_ENGINE_CONTRACT_RECONCILIATION.md`](docs/architecture/ARX_ARCHITECTURE_AND_ENGINE_CONTRACT_RECONCILIATION.md) | Full 12-engine contract reconciliation, input/output data boundaries, fallbacks, and sizing. |
| **Market-Wide Discovery Research** | [`ARX_MARKET_WIDE_RESEARCH_DECISION_AND_EVIDENCE_REGISTER.md`](docs/research/ARX_MARKET_WIDE_RESEARCH_DECISION_AND_EVIDENCE_REGISTER.md) | MW-004 research reconciliation, Model E+ ratification, candidate freeze, and scoring repair. |
| **Synthesis E V3 UX Architecture** | [`ARX_UX_V3_APPROVAL_IMPLEMENTATION_AND_RELEASE_STATUS.md`](docs/ux/ARX_UX_V3_APPROVAL_IMPLEMENTATION_AND_RELEASE_STATUS.md) | Three-mode terminal (Guided/Standard/Quant), Paper Light theme, accessibility, and PR #131. |
| **Sprint 1–4 Obligations & Holds** | [`ARX_SPRINT1_4_OBLIGATION_AND_HOLD_REGISTER.md`](docs/planning/ARX_SPRINT1_4_OBLIGATION_AND_HOLD_REGISTER.md) | Deduplicated historical findings, 14 platform security holds, and administrative closure. |
| **CI/CD & Deployment Authority** | [`ARX_CI_RELEASE_AND_DEPLOYMENT_AUTHORITY_MATRIX.md`](docs/releases/ARX_CI_RELEASE_AND_DEPLOYMENT_AUTHORITY_MATRIX.md) | Release Qualification Suite (RQ-01..12), GitHub rulesets `25187391` & `26024894`, hosting boundaries. |
| **PR & Integration Register** | [`ARX_PR_AND_INTEGRATION_STATUS_REGISTER.md`](docs/planning/ARX_PR_AND_INTEGRATION_STATUS_REGISTER.md) | Exhaustive register of all open and merged PRs, commit SHAs, and combined candidate D4. |
| **Conflict & Correction Log** | [`ARX_DOCUMENTATION_CONFLICT_AND_CORRECTION_LOG.md`](docs/planning/ARX_DOCUMENTATION_CONFLICT_AND_CORRECTION_LOG.md) | Authoritative resolutions for historical contradictions, stale SHAs, and superseded findings. |
| **Evidence Traceability Matrix** | [`ARX_DOCUMENTATION_EVIDENCE_TRACEABILITY_MATRIX.csv`](docs/planning/ARX_DOCUMENTATION_EVIDENCE_TRACEABILITY_MATRIX.csv) | Machine-readable claim-to-source mapping covering core architectural facts. |
| **Documentation QA Report** | [`ARX_DOCUMENTATION_QA_REPORT.md`](docs/planning/ARX_DOCUMENTATION_QA_REPORT.md) | 18-point documentation verification suite results (100% PASS, zero broken links). |
| **Owner Review & Publication Handoff** | [`ARX_DOCUMENTATION_OWNER_REVIEW_AND_PUBLICATION_HANDOFF.md`](docs/planning/ARX_DOCUMENTATION_OWNER_REVIEW_AND_PUBLICATION_HANDOFF.md) | Final executive handoff package for Product/Research Owner formal review. |

---

## 🎯 Product & Capabilities

### Target Audience & Core Purpose
ARX Terminal is an institutional financial intelligence and discretionary trading workstation engineered for quantitative portfolio managers, research analysts, and risk officers. It synthesizes disparate technical, fundamental, setup, and macroeconomic data into a coherent, evidence-grounded research cockpit.

### Core Operational Capabilities
1. **Interactive Dual-Horizon Candlestick Engine:**
   - *Intraday Scalp View:* 1m, 5m, 15m, 1h timeframes overlaid with Volume-Weighted Average Price (VWAP).
   - *Macro Horizon View:* 1M, 6M, 1Y, 3Y, 5Y timeframes overlaid with 20 Exponential Moving Average (20 EMA).
2. **Minervini Volatility Contraction Pattern (VCP) Detection:**
   - Algorithmic stage detection (Stage 2 uptrend prerequisite, progressive contraction count, volume dry-up).
3. **Multi-Pillar Confluence Scoring:**
   - Synthesizes Technicals ($35\%$), Fundamentals ($30\%$), Setups/VCP ($20\%$), and Macro ($15\%$) into a calibrated 0–100 score.
4. **Governed Position Sizing & Portfolio Provenance (PR #129):**
   - Fail-closed sizing calculator enforcing verified account equity. LocalStorage overrides strictly quarantined.
5. **Synthesis E V3 Three-Mode Terminal (PR #131):**
   - Guided Mode (Executive Summary), Standard Mode (Active Swing Trading), and Quant Mode (Factor Sensitivities).

### Experimental & Operationally Restricted Capabilities
- **Market Opportunity Radar:** Single published universe (250 evaluated, 100 Stage 2), candidate narrative deduplication (PR #285), and read-only Smart Money evidence adapter (PR #294) live on Cloudflare Pages (`page-7644a93d0a184821.js`). Natural background scanning, SEC discovery, and cron schedulers remain strictly held under `HOLD_SCAN_001` (discovery held; static universe served pending revalidation of historical readiness blockers: 439/500 stale symbols, 67 unprovenanced symbols, and published snapshot release lineage divergence).
- **Smart Money (SEC Form 4):** Supervised single-cycle one-shot canaries for NVDA (PR #306) and AAPL (PR #307) under Issue #165 are complete; sequential pair execution qualified in fail-closed wrapper v1.1.0 (PR #309). Supervised candidate accession `0001140361-26-044952` was qualified and ingested. Production database holds 16 NVDA accessions (100% trailing 90-day coverage) with Parker amendment linked, plus 6 AAPL Form 4 accessions (including canary `0001140361-26-044773` and accession `0001140361-26-044952`), totaling 140 parsed transactions. Active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) is preserved untouched. Authoritative superseding provenance correction v2 (`6c7bf075-0d62-4a95-9727-044515c3e376`) attributes canary execution to `16549be1`. Separately, the two-issuer recurring trial was qualified for crash safety (PR #311) and durable slot dispatch (PR #312); Day 1 of trial `ARX-TRIAL-2026-NVDA-AAPL-001` is armed under single-reading 23h55m grant `dc58b294b7a744da9730c5a2b97725b1`; worker is online in `OWNER` role holding exclusive session advisory lock `401007311509093371`, polling silently in `HEALTHY` state. Operational status: `ARMED` awaiting natural scheduled execution in Slot 1 (`2026-11-20:SLOT_1:0915`); zero recurring cycles completed. Continuous background market-wide ingestion and snapshot publication remain strictly held under `HOLD_SEC_001`. PR #310 merge is deferred to protect worker continuity and grant binding.
- **Multi-Pillar Confluence Scoring:** Provenance parity (PR #277), latency SLA (PR #275), ratified engine manifest v2.5.3 (PR #291), unverified/held distinction (PR #295), FRED macro contract (PR #296), and drawer presentation truth ([PR #299](https://github.com/daakara/finance/pull/299)) deployed. Automated live signal promotion is held under `ARX_CONFLUENCE_INTEGRITY_HOLD`.
- **Pre-Flight Decision Authority:** Epistemic separation (simulated/historical/live), decision freshness safeguards (PR #287), analysis cache authority (PR #288), and live quote provenance / canonical sizing authorization ([PR #300](https://github.com/daakara/finance/pull/300)) deployed or under qualification.
- **Frontend UX Progressive Disclosure:** Progressive disclosure redesign (PR #274) is held under `PHASE_A3_UX_HOLD` awaiting a genuine 5-participant formative human usability study. Zero fabricated observations permitted.
- **Corporate Events:** Held under `HOLD_SEC_001`. Live SEC EDGAR network egress is blocked in production; mock fixtures and primary-source confirmed schedules only.
- **Financial Execution:** Held under `HOLD_EXEC_001`. Zero live trade execution code exists in the production runtime; all orders are simulated.

---

## 🏗️ System Architecture & Technology Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT / EDGE PRESENTATION                      │
│  • Next.js 14.2 (App Router) + React 18 + Tailwind CSS                 │
│  • Lightweight Charts Canvas (Dual-Horizon Rendering)                  │
│  • Cloudflare Pages Hosting (Manual Rollouts, Auto-Deploy DISABLED)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ /api/backend/* (Proxy Gateway)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         BACKEND APPLICATION API                        │
│  • FastAPI 0.110+ (Python 3.12) Async Application Engine               │
│  • 12 Specialized Computational Engines (Technical, VCP, Risk, etc.)   │
│  • Railway Production Container (Commit 6ea2328, Auto-Deploy DISABLED) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Strict Invariant Barriers
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE & LINEAGE                          │
│  • PostgreSQL 16.x Production Worker Store (Railway tranquil-radiance) │
│  • SQLite Production Scanner & State Store (.finance_scanner_store.db) │
│  • Cloudflare R2 Automated WAL Archiving & Backup Verification         │
│  • Frozen Engine Manifest V2.5.1 + Point-in-Time Research Datasets     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Governance & Fail-Closed Safety Invariants

ARX Terminal operates under 28 strict architectural and mathematical invariants (`docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md`) and six active operational holds (`docs/governance/ARX_OPERATIONAL_HOLDS_REGISTER.md`). Key fail-closed boundaries include:

- **INV-01 / INV-02 (`HOLD_EXEC_001`):** Absolute prohibition on live capital commitment. All order routing paths are simulated.
- **INV-05 / INV-18 (`HOLD_SEC_001`):** Production background continuous ingestion of SEC Form 4 records is permanently halted. Worker operates in fail-closed `STANDBY`.
- **INV-09 (`HOLD_SCAN_001`):** Radar integrity protection. Natural background scanning and cron schedulers remain disabled.
- **INV-11 / INV-12 (`ARX_CONFLUENCE_INTEGRITY_HOLD`):** Confluence scoring live automation held pending complete multi-pillar provenance parity and decision trace integrity.
- **INV-14 (`PHASE_3_HOLD`):** Static model weights. Online learning and prospective parameter mutation are blocked. Sprint 5 locked (`sprint5_execution_authorized = false`).
- **INV-22 (`HOLD_SEC_001`):** Corporate events live SEC EDGAR egress is blocked; static fixture replay only.
- **INV-28 (Equity Provenance):** Position sizing requires `telemetry_verified` broker equity. Unverified equity forces 0 shares.
- **UX Governance (`PHASE_A3_UX_HOLD`):** PR #274 release admission held pending genuine 5-participant real-human formative evaluation. Zero fabricated data permitted.

---

## 💻 Local Development Setup

### Prerequisites
- Node.js $\ge 18.17.0$
- Python $\ge 3.11$ (Python 3.12 recommended)
- Git

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/daakara/finance.git
cd finance

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend test suite
pytest
```

### 2. Frontend Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Run frontend unit tests and type checks
npm run type-check
npm test

# Launch development server
npm run dev
```

The frontend will start locally at `http://localhost:3000` and proxy API calls to the backend running at `http://localhost:8000`.

---

## 🤝 Branching, PR & Release Conventions

- **Mainline Branch (`main`):** Protected by GitHub Ruleset `25187391`. All PRs require clean passing checks across frontend lint/test, backend pytest, and backlog integrity validation.
- **Release Branch (`release/production`):** Protected by GitHub Ruleset `26024894`. Requires complete pass of `.github/workflows/release-qualification.yml` and formal Product Owner authorization.
- **Deployment Decoupling:** Merging code into `main` **does not** deploy to Cloudflare Pages or Railway. Deployments require explicit manual triggers and preflight verification.

---

## 🚨 Disclaimer

*ARX Terminal is an institutional quantitative research and financial modeling environment. All signals, confluence scores, setup alerts, and valuations are generated for analytical evaluation only and do not constitute financial advice, investment recommendations, or an offer to buy or sell securities. Live trade execution is permanently disabled under platform security holds.*
