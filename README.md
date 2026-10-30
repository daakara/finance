# 📈 ARX Terminal | Institutional Financial Intelligence & Research Platform

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

## 🏛️ Authoritative Documentation & System Portal

The ARX Terminal documentation estate is consolidated into a canonical hierarchy. Refer to the master registers below for authoritative specifications, research decisions, and governance ledgers:

| Dimension | Canonical Documentation Portal | Primary Focus & Governance Scope |
| :--- | :--- | :--- |
| **Integrated Project Status** | [`ARX_MASTER_PROJECT_STATUS_2026_10_29.md`](docs/ARX_MASTER_PROJECT_STATUS_2026_10_29.md) | Single source of truth across research, UX, engines, engineering, and hosting status. |
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
- **Market-Wide Discovery (Top 35):** Quarantined under `/poc-market-wide` (Model E+ candidate `f154032`). Research evaluation only.
- **Smart Money (SEC Form 4):** Strictly held under `HOLD_SM1_001`. Background production ingestion is disabled; local replay only.
- **Corporate Events:** Held under `HOLD_SEC_001`. Live SEC EDGAR network egress is blocked in production; mock fixtures only.
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
│  • Railway Production Container (Commit de5d246, Auto-Deploy DISABLED) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Strict Invariant Barriers
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE & LINEAGE                          │
│  • SQLite Production Scanner & State Store (.finance_scanner_store.db) │
│  • Frozen Engine Manifest V2.5.1 + Point-in-Time Research Datasets     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Governance & Fail-Closed Safety Invariants

ARX Terminal operates under 28 strict architectural and mathematical invariants (`docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md`). Key fail-closed boundaries include:

- **INV-01 / INV-02 (`HOLD_EXEC_001`):** Absolute prohibition on live capital commitment. All order routing paths are simulated.
- **INV-05 / INV-18 (`HOLD_SM1_001`):** Production background ingestion of SEC Form 4 records is permanently halted.
- **INV-09 (Radar Integrity):** A failed scanner run (`status = ERROR`) must never report 100% complete market coverage.
- **INV-14 (`HOLD_TUNE_001`):** Static model weights. Online learning and prospective parameter mutation are blocked.
- **INV-22 (`HOLD_SEC_001`):** Corporate events live SEC EDGAR egress is blocked; static fixture replay only.
- **INV-28 (Equity Provenance):** Position sizing requires `telemetry_verified` broker equity. Unverified equity forces 0 shares.
- **Sprint 5 Invariant:** `sprint5_execution_authorized = false`. Sprint 4 remains closed administratively.

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
