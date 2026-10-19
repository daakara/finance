# SOURCED FUNDAMENTALS IMPLEMENTATION & HANDOFF RECORD

**Workstream Status**: `SOURCED_FUNDAMENTALS_CORRECTIONS_AND_OFFLINE_REPLAY_LOCAL_VERIFIED`  
**Date**: October 19, 2026  
**Repository**: `daakara/finance`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-sourced-fundamentals-pilot`  
**Branch**: `feat/sourced-fundamentals-pilot`  
**Base Commit**: `3b953f834a49d88496dfa9dae56584e17e60bae4` (main after PR #30)  
**Operator Identity**: `ARXTerminal Research Akara.david26@gmail.com`  

---

## 1. Summary of Delivered Architecture

This workstream delivered a reusable, rigorous research framework for investigating undervalued growth businesses without compromising production Radar stability or violating epistemic integrity.

All code lives in an isolated module under `analyst_dashboard/research/sourced_fundamentals/` and adheres strictly to:
1. **Field-Level Provenance**: Every financial number links back to its US-GAAP concept, SEC accession number, filing form (10-K, 10-Q), period dates, and status (`REPORTED`, `DERIVED`, `ANALYST_ADJUSTED`, or `UNAVAILABLE`).
2. **Strict Fiscal Boundary & Duration Validation**: TTM facts are assembled via the authoritative SEC identity $TTM = FY + Current\_YTD - Prior\_Comparable\_YTD$, with start and end dates validated against the company's fiscal boundaries. Standalone 3-month quarters sharing an interim end date are rejected in favor of true YTD observations.
3. **Lease-as-Operating-Expense Valuation Framework**: Operating lease obligations are evaluated under a consistent convention where leases are expensed in operating earnings, preventing double-counting of lease liabilities in the net debt bridge.
4. **Corporate Action Normalization**: Multi-year share dilution comparisons explicitly account for forward splits (`DECK` 6:1 split on Sep 13, 2024; `CELH` 3:1 split on Nov 15, 2023; `KLAC` 10:1 split in 2026). Split multipliers are applied only to pre-split filings, eliminating phantom dilution spikes while avoiding double adjustment on restated observations.
5. **Reverse-DCF Target Bracketing & Roundtrip Verification**: Reverse-DCF solves for 5-year Revenue CAGR ($g$) with target bracketing verification. Target prices outside reasonable bounds return explicit `None` instead of boundary artifacts. Solved CAGRs are validated via forward-model roundtrip check ($|DCF - target| < $0.05).
6. **Programmatic Report Parity**: All markdown tables and analytical reports are generated directly from `sourced_fundamentals_ledger.json`, eliminating human transcription errors and report/model divergence.
7. **Complete Offline Replay**: Primary SEC JSONs and dated market prices are retained in `data/research/sourced_fundamentals/raw/` with a cryptographic `manifest.json` ledger. The entire research package executes in `--offline` mode with zero network requests.

---

## 2. Component Inventory

| File Path | Description | Verified Status |
| :--- | :--- | :---: |
| `analyst_dashboard/research/sourced_fundamentals/contracts.py` | Immutable data contracts: `FinancialFact`, `CompanyFundamentals`, `DerivedMetrics`, `ResearchAssessmentRecord`, `ValuationScenario`, `DeepValuationAssessment`. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/sec_harvester.py` | Rate-limited SEC EDGAR API harvester and offline replay loader with SHA-256 manifest generation. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/normalizer.py` | Duration-validated US-GAAP concept mapper, fiscal boundary matcher, and instant balance-sheet fact extractor. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/calculations.py` | Derived metrics engine: growth, margins, non-GAAP CFO - Capex, ROIC, multiples, and arithmetic verification. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/qualitative_assessor.py` | Dynamic qualitative assessment engine assigning dispositions (`RESEARCH_FURTHER`, `WATCH`) and thesis invalidation conditions. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/valuation.py` | FCFF multi-scenario DCF engine, numerical Reverse-DCF solver with bracketing, and 2D sensitivity matrix generator. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/cli.py` | CLI orchestrator supporting `--offline` and `--refresh` modes, split-normalized dilution, and evidence ledger export. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/report_generator.py` | Programmatic Markdown deliverable generator reading directly from `sourced_fundamentals_ledger.json`. | Verified |
| `analyst_dashboard/research/sourced_fundamentals/__init__.py` | Package entrypoint exposing clean public interfaces. | Verified |
| `data/research/sourced_fundamentals/raw/manifest.json` | Provenance manifest tracking raw SEC JSONs, URLs, timestamps, and SHA-256 hashes. | Verified |
| `data/research/sourced_fundamentals/sourced_fundamentals_ledger.json` | Master JSON evidence ledger containing complete facts, derivations, assessments, and deep valuations for 8 companies. | Verified |
| `tests/test_sourced_fundamentals_pilot.py` | Comprehensive 17-scenario failure-reproducing test suite verifying duration selection, split adjustments, reverse DCF, and offline replay. | Verified |

---

## 3. Offline Replay Instructions

To reproduce the exact normalized financial facts, derived metrics, qualitative assessments, deep valuations, and evidence ledger locally without making any network requests:

```bash
# From workspace root:
python -m analyst_dashboard.research.sourced_fundamentals.cli --offline
```

To run the automated failure-reproducing test suite:
```bash
python -m pytest tests/test_sourced_fundamentals_pilot.py -v
```

---

## 4. Confirmed Limitations & Disclosed Boundaries

1. **Research Leads Are Not Certified Intrinsic Values**: Multi-scenario valuations (`Conservative`, `Base`, `Optimistic`) and Reverse-DCF growth rates represent conditional economic scenarios under stated assumptions, not certified price targets or investment recommendations.
2. **Scanner & Radar Independence**: This workstream operates in an isolated research tree. No production Radar routes, scanner configurations, or schedules (`VCP_NATURAL_SCHEDULER_ENABLED="0"`) were altered.
3. **KLAC Operating Expenses**: KLAC does not report an explicit `OperatingIncomeLoss` XBRL tag; its operating income is derived via `Revenues - CostOfRevenue - R&D - SG&A` from its 10-K report.
4. **Retail 52/53-Week Fiscal Calendars**: ULTA and LULU report on 52/53-week fiscal years ending in late January / early February. Fiscal year boundary matching allows ±4 days tolerance to accommodate these retail calendars.
