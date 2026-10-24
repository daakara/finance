# ARX Terminal — Workstream E
# Radar US-Listed Coverage & Requirements Reconciliation: Coverage Acceptance Contract

**Document ID**: `DOCS-RESEARCH-RADAR-COVERAGE-ACCEPTANCE-CONTRACT-001`  
**Classification**: Normative Specification & Acceptance Contract  
**Workstream**: E (Radar US-Listed Coverage & Requirements Reconciliation)  
**Baseline Anchor**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d` (`origin/main` post-PR #8)  
**Repository**: `daakara/finance`  
**Branch**: `research/radar-us-listed-coverage`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-radar-coverage-reconciliation`  
**Date**: 2026-10-16T18:09:00+02:00  

---

## 1. Confirmed Product Owner Requirement & Acceptance Criterion

The Product Owner has authoritatively confirmed the product mandate for ARX Radar:

1. **Target Population**: **ALL US-LISTED STOCKS**, including foreign companies listed on US exchanges.
2. **Core Purpose**: **Market-wide discovery** rather than dependence on a manually selected watchlist or curated regression fixture.
3. **Core Product Owner Acceptance Criterion**:
   > **“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”**
4. **Boundary Invariants**:
   - **Population inclusion does NOT equate to strategy eligibility.** A security may belong to the US-listed population while being legitimately excluded from the Minervini VCP or Value GARP strategy by deterministic rule.
   - A pilot, comprehensive directory, or successful parser alone does **NOT** satisfy this acceptance criterion.
   - Unresolved venue and security-type definitions remain explicit policy decisions; foreign US-listed issuers must not be silently excluded, nor OTC silently added.

---

## 2. Denominator Architecture, Pipeline Stages, and Terminal Outcome Accounting

To prevent confusing intermediate pipeline stages with terminal outcomes, this contract establishes:
1. **Pipeline Stage Progression**: Describing how securities transition through ingestion, classification, and readiness.
2. **Disjoint Terminal Outcomes**: Ensuring every distinct security in the defined population receives exactly one reconciled terminal status.
3. **Distinct Securities vs. Execution Attempts**: Accounting for retries and worker cycles without distorting security-level denominators.

### A. Pipeline Stage Progression (Intermediate Denominators)
- **Source Population ($N_{\text{source}}$)**: Raw records ingested from authoritative feeds (SEC EDGAR company tickers and Nasdaq Trader directories).
- **Classified Population ($N_{\text{classified}}$)**: Records with verified identifier, recognized venue, and unambiguous security type ($N_{\text{source}} = N_{\text{classified}} + N_{\text{unclassified}}$).
- **Strategy-Eligible Population ($N_{\text{eligible}}$)**: Securities satisfying canonical strategy rules R01–R07 ($N_{\text{classified}} = N_{\text{eligible}} + N_{\text{strategy\_excluded}}$).
- **Data-Ready Population ($N_{\text{ready}}$)**: Eligible securities with active daily market tape and $\ge 50$ historical trading sessions ($N_{\text{eligible}} = N_{\text{ready}} + N_{\text{data\_unavailable}} + N_{\text{data\_unresolved}}$).

### B. Disjoint Terminal Outcomes (The Reconciliation Equation)
A monotonic denominator hierarchy alone does **not** account for exclusions, unsupported assessments, processing failures, or retries. Every distinct security in the agreed population must map to exactly one mutually exclusive terminal outcome:

$$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded\_by\_rule}} + N_{\text{unsupported\_assessment}} + N_{\text{insufficient\_data}} + N_{\text{processing\_failure}}$$

| Terminal Outcome Code | Description | Strategy Status | Data Status |
| :--- | :--- | :--- | :--- |
| **`ASSESSED_NO_SETUP`** | Evaluated through full quantitative strategy engine; no qualifying base or contraction pattern found. | Supported / Evaluated | Ready |
| **`ASSESSED_QUALIFYING_OR_DEVELOPING`** | Evaluated and matched; constructive Stage 2 base, contraction geometry, or buy zone breakout. | Supported / Evaluated | Ready |
| **`EXCLUDED_BY_STRATEGY_RULE`** | Excluded by explicit strategy rule (e.g. ETF, preferred share, warrant, delisted). | Excluded by Rule | Any |
| **`UNSUPPORTED_ASSESSMENT`** | Admitted population member having no active quantitative model (e.g. BDCs, REITs, royalty trusts). | Unsupported Model | Any |
| **`INSUFFICIENT_UNAVAILABLE_DATA`** | Eligible for strategy, but lacks required 50-session history or fresh daily market tape. | Supported | Unavailable / Pending |
| **`PROCESSING_FAILURE`** | Eligible and data-ready, but worker encountered an unhandled exception, compute timeout, or database lock. | Supported | Ready (Execution failed) |

### C. Distinct Securities vs. Execution Attempts (Retry Accounting)
- Retries and worker restarts mean total execution attempts ($N_{\text{attempts}}$) can exceed distinct securities evaluated ($N_{\text{securities}}$).
- The universe attestation engine tracks execution attempts separately from security-level outcomes. A retry that succeeds updates the terminal outcome from `PROCESSING_FAILURE` to `ASSESSED_NO_SETUP` or `ASSESSED_QUALIFYING_OR_DEVELOPING`; an exhausted retry logs the terminal state as `PROCESSING_FAILURE`.

---

## 3. Policy Reconciliations & Decisions Requiring Product Owner Input

The investigation identified three critical policy questions where existing code contracts clash with the PO mandate or where requirements are underspecified.

### Decision 1: Foreign Companies Listed on US Exchanges (ADR Policy vs Rule R02)

- **The Clash**:
  - PO Requirement: Target population includes foreign companies listed on US exchanges.
  - Existing Rule R02 (`CANONICAL_ELIGIBILITY_RULES["R02_SECURITY_TYPE"]`):
    `expected: "COMMON_STOCK"`, description: *"Security subtype must be Common Stock (no ETFs, ADRs, REITs, Warrants in stock VCP)"*.
  - Many high-profile growth stocks trade on US exchanges as American Depositary Receipts (ADRs) or foreign ordinary shares (e.g. `TSM`, `ASML`, `BABA`, `NVO`, `SE`, `MELI`).
- **Population vs Strategy Distinction**:
  - Comprehensive accounting of foreign US-listed stocks in the *population* does **not** automatically alter R01–R07 or approve ADR strategy eligibility.
- **Options for Product Owner**:
  - **Option 1A (Recommended — Full Market Intent)**: Revise rule R02 to `R02_SECURITY_TYPE in ['COMMON_STOCK', 'ADR', 'FOREIGN_ORDINARY']` for US major exchange listings (Level II and Level III exchange-traded ADRs on NYSE/NASDAQ). Retain explicit exclusion of ETFs, ETNs, CEFs, Warrants, and Units.
  - **Option 1B (Strict Common Stock Only)**: Preserve R02 as `COMMON_STOCK` only. Account for ADRs in the population as `EXCLUDED_BY_STRATEGY_RULE` or `UNSUPPORTED_ASSESSMENT`, deferring them to a distinct future "International ADR VCP" scanner.
  - **Option 1C (Dual Track)**: Permit Level II/III ADRs under VCP only if their primary US daily volume exceeds a higher threshold (e.g. $\$10\text{M}$ average daily volume) to eliminate illiquid sponsored ADRs.

### Decision 2: Exchange Venue Scope & OTC Handling (Rule R04)

- **The Issue**:
  - Rule R04 currently restricts venues to `['NASDAQ', 'NYSE', 'ARCA', 'BATS']`.
  - The OTC Markets (OTCQX, OTCQB, Pink Sheets) contain ~12,000+ unlisted securities, many of which are illiquid shell companies, foreign unlisted ADRs, or penny stocks with wide bid-ask spreads.
- **Options for Product Owner**:
  - **Option 2A (Recommended — Institutional Standard)**: Formally reaffirm that "ALL US-LISTED STOCKS" means **exchange-listed stocks** on SEC-registered national exchanges (NYSE, NASDAQ, NYSE American/ARCA, Cboe/BATS). Explicitly exclude OTC Markets from primary Radar discovery.
  - **Option 2B (Include Premier OTC)**: Include OTCQX (the highest tier of OTC with audited financials) while strictly excluding OTCQB and Pink Sheets.
  - **Option 2C (All OTC)**: Ingest all ~12,000 OTC securities (requires specialized penny-stock liquidity guards and massive pipeline scaling).

### Decision 3: Security Subtypes: REITs, BDCs, and Preferred Shares

- **The Issue**:
  - Real Estate Investment Trusts (REITs) and Business Development Companies (BDCs) trade as common equities on NYSE/NASDAQ, but their valuation, cash flow geometry, and price movements differ substantially from operating corporations (e.g. FFO vs EPS, debt covenant distributions, fixed yield behavior).
- **Options for Product Owner**:
  - **Option 3A (Recommended)**: Account for REITs, BDCs, Royalty Trusts, and Preferred Shares in the population, but assign them terminal outcome `UNSUPPORTED_ASSESSMENT` or `EXCLUDED_BY_STRATEGY_RULE` for VCP growth screening.
  - **Option 3B**: Include REITs and BDCs in VCP scanning, treating them identically to common operating equities.

### Decision 4: Radar UI Coverage Presentation & Denominator Disclosure

- **The Issue**:
  - Radar currently displays only `allAssets.length` (the matched setups). Users cannot tell whether 20 matches represents 20 out of 35 stocks (a curated list) or 20 out of 5,500 stocks (the full market).
- **Options for Product Owner**:
  - **Option 4A (Recommended — Institutional Transparency)**: Add an authoritative **Coverage Disclosure Bar** at the top of Radar:
    `"US Equities Scanned: 5,420 | Qualifying Setups: 18 | Developing: 34 | Data Pending: 142 | Excluded: 2,110"` with an explanatory popover detailing the 6 accounting outcomes.
  - **Option 4B (Minimal Counter)**: Display only `Scanned: 5,420 US Equities` next to the category filter tabs.
  - **Option 4C (Status Quo)**: Retain match count only; expose detailed coverage only via a developer/admin route.

### Decision 5: Scanner Freshness SLA & Weekend Ingestion Policy

- **The Issue**:
  - US equity regular trading hours end at 16:00 ET (Eastern Time). EOD consolidated tapes, volume normalization, and corporate action adjustments are typically finalized by providers between 18:00 ET and 21:00 ET.
  - **Timezone Awareness**: Eastern Time shifts between EST (UTC-5) and EDT (UTC-4). A fixed UTC schedule (e.g. 22:00 UTC) corresponds to 18:00 EDT in summer, but 17:00 EST in winter.
  - The freshness policy must be exchange-calendar and timezone aware rather than anchored to a naive 24-hour UTC countdown that flags Monday pre-market snapshots as stale.
- **Options for Product Owner**:
  - **Option 5A (Recommended — Proposed)**: Propose daily EOD batch execution anchored to **Eastern Time (e.g. 18:30 ET / post-settlement consolidation)** on active US exchange trading days. Propose weekend and exchange-holiday grace periods where snapshots remain valid until the next trading day's execution window.
  - **Option 5B (Dual Run)**: Propose a dual-run schedule: Pre-market refresh (08:30 ET) + Post-close batch (18:30 ET).
  - *Note: This task proposes policy options only; it does not approve or activate a schedule or freshness policy.*

---

## 4. Acceptance Criteria for Full Market-Wide Delivery

To consider the "All US-Listed Stocks" capability delivered and production-ready, the implementation must satisfy all of the following verifiable criteria:

1. **AC-COV-001 (Source Completeness)**: The source population ingestion pipeline successfully parses active US-listed securities from official primary feeds (SEC EDGAR company tickers and Nasdaq Trader directories) without manual curation.
2. **AC-COV-002 (Terminal Outcome Reconciliation)**: Every security in the defined US-listed population is accounted for in an immutable attestation, reconciling exactly: $N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$. Zero silent drops.
3. **AC-COV-003 (Deterministic Single Outcome)**: Every single security receives exactly one terminal outcome in the execution log. Retries are tracked separately as attempts.
4. **AC-COV-004 (ADR Policy Reconciliation)**: Foreign operating companies listed on NYSE/NASDAQ are handled according to explicit Product Owner policy determinations on R02, with zero ambiguous classifications.
5. **AC-COV-005 (Scanner Scaling & Capacity Verification)**: The market-wide VCP scanner execution architecture is selected based on measured benchmarking and capacity evidence across the defined population, persisting an `ImmutableScannerSnapshot` with full coverage metadata.
6. **AC-COV-006 (Read Safety Invariant)**: Public GET requests to `/screener/run`, `/radar`, and `/vcp/snapshot` remain strictly read-only: zero scans triggered, zero leases acquired, zero database writes.
7. **AC-COV-007 (Frontend Denominator Truthfulness)**: Radar UI explicitly displays both the matched setup count AND the total scanned population denominator, eliminating any perception that a curated fixture represents market-wide coverage.
8. **AC-COV-008 (Analysis Handoff Parity)**: Any candidate surfaced in Radar transitions seamlessly to `/analysis?ticker={ticker}`, preserving the selected evaluation or clearly identifying if a newer assessment exists.

