# ARX Terminal Product Requirements Document (PRD) — Addendum 003
# Radar Market-Wide US Equity Coverage & Accounting Contracts

**Product**: ARX Terminal  
**Subsystem**: Radar (`/radar`) & Market-Wide Scanners (`MINERVINI_VCP`, `VALUE_GARP`, `SMART_MONEY`)  
**Document ID**: `PRD-ADDENDUM-003-RADAR-MARKET-WIDE-COVERAGE`  
**Classification**: Product Requirements Document (PRD) Addendum  
**Governing Authority**: ARX Terminal Institutional Decision Workstation Architecture  
**Parent PRDs**:
- `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md` (Core Workstation PRD)
- `docs/prd/PRD_ADDENDUM_001_DECISION_INTEGRITY.md` (Decision Integrity & Zero Semantics)
- `docs/prd/PRD_ADDENDUM_002_DECISION_READINESS.md` (Decision Readiness & Shared Presentation)  
**Status**: `PROPOSED_FOR_PRODUCT_OWNER_REVIEW`  
**Owner**: `OWNER_UNASSIGNED` (Pending Product Owner allocation)  
**Date**: 2026-10-16T18:48:00+02:00  

---

## 1. Justification & Target Gap Analysis

Existing PRD documentation governs single-asset conviction formation (`PRD-VNEXT`), zero-state truthfulness (`PRD-ADDENDUM-001`), and cross-surface presentation formatting (`PRD-ADDENDUM-002`). However, an audit of the repository identified **three critical product authority gaps**:

1. **Underspecified Discovery Denominator**: Existing PRDs establish the 6-stage lifecycle for an individual asset, but provide no specification for how Radar discovers setups across the broader market. The platform currently lacks a product requirement defining the scanned population denominator, resulting in regression fixtures (35 tickers) being confused with market-wide discovery.
2. **Missing Conceptual Accounting Outcomes**: The platform requires a formal accounting standard ensuring every stock in the US-listed universe receives an unambiguous, non-overlapping disposition (assessed, excluded, unsupported, data pending, failed).
3. **Foreign US-Listed Stock Policy Disconnect**: The Product Owner's mandate to include foreign companies listed on US exchanges requires explicit reconciliation with strategy rule R02 in `CANONICAL_ELIGIBILITY_RULES`. Population inclusion must be decoupled from strategy eligibility.

### Core Product Owner Acceptance Criterion
> **“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”**
- *Boundary Invariant*: A pilot, comprehensive directory, or successful parser alone does **NOT** satisfy this acceptance criterion.

---

## 2. Normative Requirements

### 2.1. Target Market Population & Strategy Boundary (REQ-RADAR-COV-01)
- **Statement**: ARX Radar discovery scanners shall define their target population as **ALL US-LISTED STOCKS** traded on major national securities exchanges (NYSE, NASDAQ, NYSE American/ARCA, Cboe/BATS).
- **Inclusion of Foreign Operating Companies**: Foreign operating corporations listed on US exchanges (via Level II/III exchange-traded ADRs, New York Registry shares, or Canadian ordinary common shares) shall be included in the candidate population accounting.
- **Population Inclusion $\neq$ Strategy Eligibility**: Inclusion in the population does **not** automatically modify strategy eligibility rules R01–R07 or approve ADR/REIT/BDC/preferred-share strategy eligibility. Strategy qualification remains governed by canonical rule models.
- **Exclusion of OTC / Unlisted Equities**: Securities traded exclusively on OTC Markets (OTCQX, OTCQB, Pink Sheets) are formally excluded from primary Radar discovery unless explicitly authorized by Product Owner policy.

### 2.2. Denominator Architecture & Disjoint Terminal Accounting (REQ-RADAR-COV-02)
- **Statement**: Radar data contracts and API responses shall structure candidate populations into distinct pipeline stages and enforce arithmetic reconciliation across disjoint terminal outcomes.
- **Pipeline Stages (Intermediate)**:
  1. **Source Population ($N_{\text{source}}$)**: Raw records ingested from authoritative feeds (SEC EDGAR and Nasdaq directories).
  2. **Classified Population ($N_{\text{classified}}$)**: Records with verified identifier, recognized venue, and unambiguous security subtype.
  3. **Strategy-Eligible Population ($N_{\text{eligible}}$)**: Securities satisfying canonical strategy rules R01–R07.
  4. **Data-Ready Population ($N_{\text{ready}}$)**: Eligible securities with $\ge 50$ daily trading bars in the market database and active daily market tape.
- **Disjoint Terminal Outcome Reconciliation**: A monotonic denominator hierarchy alone does not account for exclusions, failures, unsupported assessments, or retries. Every distinct security in the defined population must receive exactly one terminal outcome:
  $$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded\_by\_rule}} + N_{\text{unsupported\_assessment}} + N_{\text{insufficient\_data}} + N_{\text{processing\_failure}}$$
- **Distinct Securities vs. Execution Attempts**: Worker retries and restarts can cause total execution attempts ($N_{\text{attempts}}$) to exceed distinct securities ($N_{\text{securities}}$). The universe attestation engine must track execution attempts separately from security-level terminal states.

### 2.3. Six Conceptual Accounting Outcomes (REQ-RADAR-COV-03)
- **Statement**: Every security in the target population must receive exactly one of six terminal outcomes:
  1. `ASSESSED_NO_SETUP`: Evaluated through all quantitative criteria; no qualifying pattern identified.
  2. `ASSESSED_QUALIFYING_OR_DEVELOPING`: Evaluated and matched; active breakout or developing base.
  3. `EXCLUDED_BY_STRATEGY_RULE`: Recognized as US-listed, but excluded by rule R01–R07 (e.g. ETF, preferred, warrant, non-USD).
  4. `UNSUPPORTED_ASSESSMENT`: Admitted asset class with no active quantitative model (e.g. BDCs, REITs).
  5. `INSUFFICIENT_UNAVAILABLE_DATA`: $< 50$ sessions on exchange record or missing market tape.
  6. `PROCESSING_FAILURE`: Worker encountered an unhandled exception or timeout during scan execution.

### 2.4. Radar UI Coverage Transparency (REQ-RADAR-COV-04)
- **Statement**: The Radar user interface shall prominently display the total scanned denominator alongside matched setup counts.
- **Disclosure Display**:
  - The page header must show: `"US Equities Scanned: {N_ready} | Setups: {N_matched} | Data Pending: {N_insufficient_data}"`.
  - Zero results must render truthfully as `0 Setups` when scanning is active, rather than falling back to `Pending`.
  - The UI shall prevent users from inferring that a fixture or partial scan represents total market coverage.

### 2.5. Analysis Workstation Handoff Parity (REQ-RADAR-COV-05)
- **Statement**: When a candidate is selected in Radar, clicking the asset shall open `/analysis?ticker={ticker}` preserving the selected evaluation or clearly identifying if a newer assessment or market data update exists.

---

## 3. Product Owner Decision Items for Formal Sign-Off

Before this Addendum transitions from `PROPOSED` to `APPROVED_FOR_IMPLEMENTATION`, the Product Owner must record formal determinations on the following five policy choices:

1. **Rule R02 Revision (ADR Policy)**:
   - [ ] **Option 1A (Recommended)**: Revise R02 to permit Level II/III exchange-listed ADRs of operating companies in stock VCP.
   - [ ] **Option 1B**: Keep R02 strictly limited to domestic common stock; account for ADRs in population as `EXCLUDED_BY_STRATEGY_RULE` or `UNSUPPORTED_ASSESSMENT`.
2. **Venue Allowlist Scope**:
   - [ ] **Option 2A (Recommended)**: Confine scope to national exchanges (NYSE, NASDAQ, AMEX/ARCA, BATS); exclude OTC.
   - [ ] **Option 2B**: Include premier OTCQX equities; exclude OTCQB/Pink.
3. **Specialized Corporate Structures**:
   - [ ] **Option 3A (Recommended)**: Exclude REITs, BDCs, and Preferred Shares from VCP growth screening (`UNSUPPORTED_ASSESSMENT`).
   - [ ] **Option 3B**: Include REITs and BDCs in VCP scanning without structural differentiation.
4. **Radar Coverage Disclosure UI**:
   - [ ] **Option 4A (Recommended)**: Render full coverage summary bar (Scanned, Matched, Pending, Excluded) on Radar header.
   - [ ] **Option 4B**: Minimal display (Scanned denominator counter only).
5. **Scanner Freshness & Cadence**:
   - [ ] **Option 5A (Recommended — Proposed)**: Propose daily EOD batch execution anchored to Eastern Time post-settlement (e.g. 18:30 ET) on exchange trading days, with exchange-calendar-aware weekend grace periods.
   - [ ] **Option 5B**: Dual-scan cadence (pre-market and post-close).
   - *Note: Proposes policy options only; does not approve or activate a schedule.*

---

## 4. Document History & Approval

| Revision | Date | Author / Workstream | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| `1.0.0-PROPOSAL` | 2026-10-16 | Workstream E Core Team | Initial proposal defining market-wide US equity coverage, denominator hierarchy, and accounting outcomes. | `PROPOSED_FOR_PRODUCT_OWNER_REVIEW` |
