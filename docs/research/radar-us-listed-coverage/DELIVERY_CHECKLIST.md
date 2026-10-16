# ARX Terminal — Workstream E
# Radar US-Listed Coverage & Requirements Reconciliation: Delivery Checklist & Implementation Sequence

**Document ID**: `DOCS-RESEARCH-RADAR-DELIVERY-CHECKLIST-001`  
**Classification**: Engineering Delivery Plan & Implementation Sequence  
**Workstream**: E (Radar US-Listed Coverage & Requirements Reconciliation)  
**Baseline Anchor**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d` (`origin/main` post-PR #8)  
**Repository**: `daakara/finance`  
**Branch**: `research/radar-us-listed-coverage`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-radar-coverage-reconciliation`  
**Date**: 2026-10-16T18:46:00+02:00  

---

## 1. Operating Capability Inventory & Reusable Assets

Before scheduling new implementation packages for Sprint 2, every existing component in the repository bearing on universe construction, scanner execution, and publication was audited.

| Component / Subsystem | Current State / Location | Operational Classification | Evidence & Reusability |
| :--- | :--- | :--- | :--- |
| **Input Validation Core** | `analyst_dashboard/universe/validator.py` | `IMPLEMENTED_AND_TESTED` | Deployed via PR #6 (`tests/test_universe_validator.py`). Reusable without modification. Fail-closed atomic batch validation; never fabricates defaults. |
| **Offline Source Parsers & Bridge** | `analyst_dashboard/universe/adapters/` | `IMPLEMENTED_AND_TESTED` | Deployed via PR #8 (`tests/test_universe_offline_adapters.py`). Pure in-memory parsers for SEC EDGAR, Nasdaq Trader, OpenFIGI; bridge to validator. |
| **Universe Builder & Accounting Engine** | `analyst_dashboard/universe/builder.py` | `IMPLEMENTED_AND_TESTED` | Verified in `tests/test_radar_universe_market_wide.py`. Enforces arithmetic closure, R01–R07 evaluation, and deterministic decision hashing. |
| **Universe Store & Attestation DB** | `analyst_dashboard/universe/store.py` | `IMPLEMENTED_AND_TESTED` | SQLite/Postgres schema for `universe_builds`, `eligibility_ledger`, and `data_readiness_ledger`. |
| **Inactive Read-Safety Contract** | `analyst_dashboard/analyzers/scanner_runner.py` | `IMPLEMENTED_AND_TESTED` | Deployed via PR #4 (`tests/test_vcp_inactive_read_safety.py`). Read paths are strictly side-effect free; zero scans on read. |
| **Distributed Run Coordination** | `analyst_dashboard/analyzers/run_coordinator.py` | `IMPLEMENTED_AND_TESTED` | Verified in `tests/test_radar_distributed_coordination.py`. Fencing tokens, lease management, collision prevention. |
| **Natural Scheduler Trigger Service** | `analyst_dashboard/vcp/natural_trigger.py` | `IMPLEMENTED_AND_TESTED` | Verified in `tests/test_vcp_natural_control_plane.py`. Exchange-session eligibility and settlement window checks. |
| **Listing Acquisition Pipeline** | *Not implemented in repository* | `MISSING` | Requires out-of-band scheduled ingestion to download and archive raw SEC EDGAR and Nasdaq Trader directory feeds. |
| **Execution Scaling & Capacity** | `scanner_runner.py` (sequential loop) | `PARTIAL` | Works correctly on fixture pools; architecture for full-population assessment must follow empirical benchmarking evidence. |
| **Market Database Historical Breadth** | `MarketDatabaseEngine` (`market_db.py`) | `IMPLEMENTED_EVIDENCE_INCOMPLETE` | Stores daily candles for actively tracked assets; market-wide coverage requires pre-populating daily price history across the eligible population. |
| **Radar UI Coverage Presentation** | `frontend/app/radar/page.tsx` | `PARTIAL` | Displays matched asset count (`allAssets.length`); completely lacks population denominator and accounting metrics. |
| **Smart Money Universe Screener** | `smart_money_scanner_runner.py` | `PARTIAL` | Fails closed to `PIPELINE_PENDING` (by design per PRD Addendum 001). Single-ticker disclosures active; screener worker pending. |

---

## 2. Agreed Sprint 2 Delivery Direction: Ordered Packages (S2-01 to S2-10)

Sprint 1 is closed to new scope. All implementation belongs to Sprint 2 or later. The agreed delivery sequence consists of ten ordered packages:

```
[S2-01: Population & Strategy Boundaries]
                     |
                     v
[S2-02: Comprehensive Listing Acquisition & Source Evidence]
                     |
                     v
[S2-03: Security Identity & Required Evidence]
                     |
                     v
[S2-04: Connected Adapter / Validator / Builder Integration]
                     |
                     v
[S2-05: Market-Data Coverage & Capacity Verification]
                     |
                     v
[S2-06: Governed Build / Publication Tooling]
                     |
                     v
[S2-07: Scanner & Radar Membership Integration]
                     |
                     v
[S2-08: Full-Population Assessment & Outcome Accounting]
                     |
         +-----------+-----------+
         |                       |
         v                       v
[S2-09: Radar Coverage   [S2-10: Operational Acceptance
  Disclosure & Handoff]    & Controlled Activation]
```

### Package Definitions and Acceptance Contracts

#### `S2-01` — Population and Strategy Boundaries
- **Objective**: Establish explicit population scope and separate assessment/eligibility treatment.
- **Acceptance Contract**:
  - Explicitly define the US-listed population boundary.
  - Distinguish population inclusion from strategy eligibility (rules R01–R07).
  - Ensure zero silent omission of unresolved stock types (ADRs, REITs, BDCs, preferreds).
- **Dependencies**: None. Pure domain policy and contract typing. Can start independently.

#### `S2-02` — Comprehensive Listing Acquisition and Source Evidence
- **Objective**: Automated, repeatable listing acquisition from primary sources.
- **Acceptance Contract**:
  - Out-of-band acquisition worker for SEC EDGAR (`company_tickers_exchange.json`) and Nasdaq Trader (`nasdaqlisted.txt`, `otherlisted.txt`).
  - Preserves immutable raw source bytes and SHA-256 content hashes.
  - Incorporates feed completeness checks and explicit fail-closed behavior for missing or partial feeds.
- **Dependencies**: `S2-01`.

#### `S2-03` — Security Identity and Required Evidence
- **Objective**: Resolve security-level identity, classification, and listing status evidence.
- **Acceptance Contract**:
  - Security identity scheme defined with share-class granularity (preventing dual-class ticker collapse).
  - Currency, venue listing, and active status evidence requirements enforced.
  - Records with uncertified or missing evidence remain explicitly marked `UNRESOLVED` rather than dropped or defaulted.
- **Dependencies**: `S2-01`, `S2-02`.

#### `S2-04` — Connected Adapter / Validator / Builder Integration
- **Objective**: Connect acquired listing feeds through existing adapters, validator core, and universe builder.
- **Acceptance Contract**:
  - Real acquired observations flow through `AdapterValidatorBridge` and `UniverseInputValidator` without fabricated defaults.
  - Feeds into `DeterministicUniverseBuilder` to produce a verifiable attestation.
  - Reconciles mathematical accounting: $N_{\text{source}} = N_{\text{eligible}} + N_{\text{ineligible}} + N_{\text{unresolved}}$.
- **Dependencies**: `S2-02`, `S2-03`.

#### `S2-05` — Market-Data Coverage and Capacity Verification
- **Objective**: Measure historical market-data coverage and execution capacity over the defined population.
- **Acceptance Contract**:
  - Quantify the proportion of strategy-eligible securities having $\ge 50$ trading sessions and active daily market tape in `market_db`.
  - Missing data is captured explicitly with failure reasons (IPO seasoning, trading halt, data provider gap).
  - Benchmark scanner execution capacity across the eligible population; scaling architecture follows empirical capacity evidence rather than premature assumptions.
- **Dependencies**: `S2-04`.

#### `S2-06` — Governed Build / Publication Tooling
- **Objective**: Reviewed tooling and repeatable publication safeguards.
- **Acceptance Contract**:
  - Tooling for universe construction and publication (`scripts/publish_universe.py`) reviewed and hardened.
  - Enforces repeatable execution, safe failure handling, and strict authorization boundaries.
  - **Boundary Invariant**: Publishing a universe build stores the authorized asset; it does **not** activate scanning or trigger execution.
- **Dependencies**: `S2-04`, `S2-05`.

#### `S2-07` — Scanner and Radar Membership Integration
- **Objective**: Connect scanner execution and Radar candidate routes to published universe membership.
- **Acceptance Contract**:
  - Published universe membership drives scanner assessment.
  - Fixed candidate lists (`HISTORICAL_VCP_35`) do not restrict discovery or discard valid out-of-list matches.
  - Public route `/screener/run` returns qualified setups from the entire published universe.
- **Dependencies**: `S2-06`.

#### `S2-08` — Full-Population Assessment and Outcome Accounting
- **Objective**: Complete market assessment ensuring every stock in the defined population receives a reconciled terminal outcome.
- **Acceptance Contract**:
  - Reconciles the terminal outcome equation across all distinct securities:
    $$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$
  - Retries, worker failures, unsupported assessments, and unavailable data are recorded distinctly and never conflated with "no qualifying setup".
  - Execution attempts ($N_{\text{attempts}}$) are tracked separately from distinct securities ($N_{\text{securities}}$).
- **Dependencies**: `S2-05`, `S2-07`.

#### `S2-09` — Radar Coverage Disclosure and Analysis Handoff
- **Objective**: Truthful UI disclosure of scanned denominators and seamless Analysis handoff.
- **Acceptance Contract**:
  - Radar UI displays total scanned denominator, matched setups, data pending counts, and coverage limitations.
  - Truthful zero semantics: `0 Setups` rendered when scanning is active, rather than collapsing into `Pending`.
  - Handoff from Radar to `/analysis?ticker={ticker}` preserves the selected evaluation or clearly identifies if a newer assessment exists.
- **Dependencies**: `S2-07`, `S2-08`. (Can be prototyped in parallel once API contracts are pinned).

#### `S2-10` — Operational Acceptance and Controlled Activation
- **Objective**: Production readiness verification, cadence, and guarded activation.
- **Acceptance Contract**:
  - Execution cadence, timezone awareness (Eastern Time exchange calendar), retry limits, and monitoring alerts verified.
  - Failure modes tested: partial results cannot appear complete; stale data triggers truthful advisory banners.
  - Activation requires explicit, separate operational authorization.
- **Dependencies**: `S2-08`, `S2-09`.

*Note: S2 labels are planning references, not proof of authoritative backlog registration.*

---

## 3. Prioritization, Capacity, and Phased Scheduling

### Core Delivery Path vs Acceptance Controls
- **Core Full-Market Delivery Path**: `S2-01` through `S2-08`. This is the sequential critical path required to discover, validate, ingest, assess, and account for the full US-listed stock population.
- **Release Acceptance Controls**: `S2-09` (Coverage Disclosure) and `S2-10` (Operational Controls) are mandatory release acceptance criteria. While their frontend/operational components can be developed concurrently, full release acceptance requires all 10 packages.

### Phased Capacity Allocation: Sprint 2 vs Sprint 3 Split
Without empirical execution benchmarks on full-market data ingestion (which will be established in `S2-05`), **promising completion of all 10 packages within a single sprint would be ungrounded**. A responsible, capacity-conscious split is established:

```
+-----------------------------------------------------------------------------------+
| SPRINT 2 FOCUS: Foundation, Ingestion, Validation & Capacity (S2-01 to S2-06)      |
| Deliverables:                                                                     |
| - S2-01: Population & strategy boundary specifications                            |
| - S2-02: Automated primary source ingestion worker (SEC + Nasdaq)                 |
| - S2-03: Security identity & evidence models                                      |
| - S2-04: Connected adapter/validator/builder integration                          |
| - S2-05: Market-data coverage audit & scanner capacity benchmarking               |
| - S2-06: Governed build/publication tooling                                       |
| Milestone Outcome: Authorized, attested, published full-population universe build  |
| (Labeled: PARTIAL MILESTONE - scanning not yet activated)                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| SPRINT 3 FOCUS: Scanner Integration, Accounting & Controlled Release (S2-07 to 10)|
| Deliverables:                                                                     |
| - S2-07: Scanner & Radar membership integration (route un-truncation)             |
| - S2-08: Full-population assessment & 6-outcome accounting reconciliation         |
| - S2-09: Radar coverage disclosure & Analysis parity handoff                      |
| - S2-10: Operational acceptance & guarded natural scheduler activation             |
| Milestone Outcome: Full-market discovery operational on Radar                      |
+-----------------------------------------------------------------------------------+
```

### Critical Path & Safe Parallelism
- **Strict Critical Path**: `S2-01` $\to$ `S2-02` $\to$ `S2-03` $\to$ `S2-04` $\to$ `S2-05` $\to$ `S2-06` $\to$ `S2-07` $\to$ `S2-08` $\to$ `S2-10`.
- **Safe Parallel Work**:
  - `S2-09` (Frontend Coverage Disclosure UI) can be built against fixture envelopes once `S2-01` pins the accounting contracts.
  - `S2-06` (Publication Tooling) can be hardened in parallel with `S2-05` capacity benchmarking.
- **Scope Starting Independently**: `S2-01` (Boundary definitions and PRD Addendum 003 reconciliation) requires zero network or backend code and can proceed immediately.

### Out of Scope for Minimum Full-Market Release
To ensure delivery focus and prevent scope creep, the following items are **explicitly excluded** from the minimum full-market release:
1. Ownership-aware enhancements or portfolio tracking filters.
2. Advanced multi-factor ranking, personalization, or client-side portfolio fit.
3. Intraday price change alerts or breakout notification webhooks.
4. Data.gov macroeconomic connector implementation (remains strategic research under `BACKLOG-DATA-GOV-DISCOVERY-001`).
5. Broad redesign of the `/analysis` workstation layout.
6. Changes to core quantitative scoring formulas, weights, or Minervini volatility contraction criteria.

---

## 4. Diagnostic Subset (Pilot Boundary)

To safely verify integration mechanics before running full-market assessment:
- **Nature of the Subset**: A representative diagnostic subset with an explicit selection rationale (e.g. cross-venue, cross-liquidity, domestic operating companies and foreign US-listed issuers).
- **Deduplicated Count**: Exact deduplicated size to be established during `S2-04` execution based on verified feed contents.
- **Acceptance Invariant**:
  > [!IMPORTANT]
  > A diagnostic subset is solely an engineering verification vehicle for pipeline mechanics. **Executing a diagnostic subset does NOT satisfy the Product Owner acceptance criterion of full-market coverage.**

---

## 5. Workstream Isolation & Tracking

```
WORKSTREAM_A = UNIVERSE_WIRING_RELEASED_EVIDENCE_CLOSED
WORKSTREAM_B = DATA_GOV_DISCOVERY_COMPLETE_WITH_LIMITATIONS
WORKSTREAM_C = SLICE_1_PR_6_DEPLOYED | SLICE_2_PR_8_MERGED
WORKSTREAM_D = PR_7_RELEASED_TO_PRODUCTION
WORKSTREAM_E = SPRINT_1_RECONCILIATION_COMPLETE | SPRINT_2_HANDOFF_PREPARED
```

- Application code changes: `NO`
- Remote push / PR: `NO`
- Production deploy / database access: `NO`
