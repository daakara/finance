# ARX Terminal — Sprint 2 Kickoff Record
## Foundation Scope & Package 1: Listing Acquisition and Source-Population Accounting

**Document ID**: `SPRINT-2-KICKOFF-RECORD`  
**Classification**: Authoritative Sprint 2 Kickoff & Package 1 Execution Plan  
**Governing Authorization**: Product Owner Sprint 2 Kickoff Directive (Package 1 Approval)  
**Baseline Commit**: `caf7c79c5163539a35b9d5779e81568934c54464` (PR #9 Closeout on `origin/main`)  
**Baseline Tree**: `f18f8fe7624c329c3bc0e1710dbb08edbcf56b67`  
**Implementation Branch**: `feat/sprint-2-listing-acquisition`  
**Implementation Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition`  
**Date**: 2026-10-16  
**Sprint Status**: `IN_PROGRESS`  
**Package 1 Status**: `VERIFIED_LOCAL`  

---

## 1. Approved Objective & Scope Fence

### 1.1 Product Owner Approved Foundation Scope
The Product Owner has approved Sprint 2's foundation direction focused strictly on establishing and sizing the US-listed population:
1. Comprehensive accounting of US-listed stocks, including foreign issuers.
2. Listing acquisition and preserved source evidence.
3. Security identity and required-field evidence handling.
4. Connected adapter $\to$ validator $\to$ builder integration in isolation.
5. Measured market-data coverage and processing capacity.

### 1.2 Package 1 Bounded Objective
Package 1 executes **Listing Acquisition & Source-Population Accounting**:
- Bounded, explicitly invoked network acquisition of public SEC EDGAR and Nasdaq Trader directory feeds.
- Preservation of raw response bytes and attributable metadata in isolated local storage.
- Replay through existing pure in-memory offline parsers (`analyst_dashboard/universe/adapters/`).
- Machine-readable and concise human-readable source accounting reports.
- Comprehensive unit tests with mocked transport and one bounded live verification.

### 1.3 Core Product Acceptance Criterion
> **“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”**

*Explicit Invariant*: A successful download, directory join, parser, or diagnostic subset alone does **NOT** satisfy this criterion.

---

## 2. Unapproved Draft Elements & Capacity Clarifications

The draft charter proposals generated prior to this kickoff contained elements that were **NOT** approved by the Product Owner and must **NOT** be imported as policy:

| Draft Proposal (Superseded) | Authority Status | Binding Position for Sprint 2 |
|:---|:---|:---|
| **3-Week Sprint Duration** | `UNAPPROVED_DRAFT` | **UNCONFIRMED**. Sprint duration is uncommitted pending empirical data sizing. |
| **Fixed Staffing Allocation** | `UNAPPROVED_DRAFT` | **UNCONFIRMED**. Package 1 owner is current execution session; other owners unassigned. |
| **CIK+Symbol as Production Identity** | `UNAPPROVED_DRAFT` | **REJECTED AS PRODUCTION STANDARD**. Identity granularity standard requires unique listing-level identity; temporary bridging key cannot replace permanent security master. |
| **95% FIGI Coverage Gate** | `UNAPPROVED_DRAFT` | **UNAPPROVED**. FIGI mapping is diagnostic in Sprint 2; network acquisition of OpenFIGI is out of scope for Package 1. |
| **18:30 ET Schedule / 72h Freshness** | `UNAPPROVED_DRAFT` | **UNAPPROVED**. Operational cadence remains an unactivated specification proposal; no scheduler activation is authorized. |
| **ADR / REIT / BDC Strategy Eligibility** | `UNAPPROVED_DRAFT` | **UNAPPROVED**. Existing eligibility rules (R01–R07) and frozen engine manifest v2.5.1 remain strictly unchanged. |
| **Production Publication as Committed Milestone**| `UNAPPROVED_DRAFT` | **CONDITIONAL / EXCLUDED**. Publication tooling remains conditional and outside Package 1. |

---

## 3. Backlog Crosswalk & Contribution Mapping

The integrated backlog ([`docs/planning/ENGINEERING_BACKLOG.md`](file:///docs/planning/ENGINEERING_BACKLOG.md)) registers `BACKLOG-RADAR-US-LISTED-COVERAGE-001` with an ordered 10-package sequence (`S2-01` to `S2-10`). The working scope for this kickoff aligns as follows:

| Integrated Backlog ID | Package Title | Sprint 2 Role | Package 1 Contribution |
|:---|:---|:---|:---|
| **`S2-01`** | Population & Strategy Boundaries | Foundation | **Contributes**: Codifies source feeds, venue mapping, and explicit non-inference boundaries. *Does not complete policy approval.* |
| **`S2-02`** | Comprehensive Listing Acquisition | Foundation | **Contributes**: Implements automated, bounded fetch and archival for SEC + Nasdaq Trader. *Does not establish complete market coverage.* |
| **`S2-03`** | Security Identity & Required Evidence | Foundation | Downstream (Package 2). Out of scope for Package 1. |
| **`S2-04`** | Connected Integration Core | Foundation | Downstream (Package 3). Out of scope for Package 1. |
| **`S2-05`** | Market-Data Coverage & Capacity | Foundation | Downstream (Package 4). Out of scope for Package 1. |
| **`S2-06`** | Governed Build / Publication Tooling | Conditional | Conditional. Out of scope for Package 1. |
| **`S2-07`** | Scanner & Radar Membership Integration | Sprint 3 | Deferred to Sprint 3. Out of scope. |
| **`S2-08`** | Full-Population Assessment & Accounting | Sprint 3 | Deferred to Sprint 3. Out of scope. |
| **`S2-09`** | Radar Coverage Disclosure & Parity | Sprint 3 | Deferred to Sprint 3. Out of scope. |
| **`S2-10`** | Operational Acceptance & Activation | Sprint 3 | Deferred to Sprint 3. Out of scope. |

---

## 4. Operational & Governance Boundary Invariants

The following boundaries are strictly enforced across this package:

```ini
PACKAGE_1_OWNER = ARX_EXECUTION_SESSION
OTHER_PACKAGE_OWNERS = UNASSIGNED
SPRINT_DURATION = UNCONFIRMED
ENGINEERING_CAPACITY = UNCONFIRMED
LIVE_SCANNER_EXECUTION = STRICTLY_PROHIBITED
HISTORICAL_DATA_BULK_BACKFILL = NOT_AUTHORIZED
SCHEDULER_ACTIVATION = STRICTLY_PROHIBITED
LIVE_RADAR_ROUTE_MUTATION = STRICTLY_PROHIBITED
CANDIDATE_004_CAPTURE_ENABLEMENT = NOT_AUTHORIZED
PRODUCTION_UNIVERSE_PUBLICATION = NOT_AUTHORIZED
PRODUCTION_INGESTION_INTEGRATION = NOT_IMPLEMENTED
GATE12_ADVANCEMENT = STRICTLY_PROHIBITED
FROZEN_QUANT_ENGINES = UNTOUCHED (Manifest v2.5.1 / Strategy v2.5.0)
READ_SAFETY_CONTRACT = ENFORCED (Zero scans, leases, or writes on GET requests)
```

---

## 5. Package 1 Acceptance Criteria & Verification Status

- [x] **1. Explicit Invocability**: Network acquisition layer is explicitly invoked via function call or diagnostic script (`scripts/research/acquire_listing_directories.py`). Zero background workers, crons, API route hooks, or startup listeners.
- [x] **2. Raw Evidence Preservation**: Raw response bytes and sidecar metadata JSON (SHA-256, URL, timestamps, headers) are preserved in isolated local storage (`data/raw_feeds/{YYYY-MM-DD}/`).
- [x] **3. Safe Storage Invariant**: Failed, timed-out, or malformed acquisitions can **never** overwrite or corrupt an accepted historical artifact; failures route to isolated quarantine logs.
- [x] **4. Offline Parser Replay**: Preserved bytes replay directly through existing pure in-memory parsers (`parse_sec_company_tickers_exchange`, `parse_nasdaq_trader_directory`) without synthetic defaults or manufactured IDs.
- [x] **5. Visible Errors & Empirical Diagnostics**: Structural errors and missing attributes remain explicitly counted and reported in diagnostics (10,444 SEC data rows: 10,276 usable, 168 rows with missing exchange attributes, 0 structurally rejected syntax errors; 471 cross-feed venue conflicts analyzed: 275 AMEX, 144 ARCA, 44 BATS, 8 TXSE; 0 symbol collisions).
- [x] **6. Accounting Truthfulness**: Output report explicitly affirms `PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED`. Total candidate count (16,448) represents normalized-symbol candidate keys across feeds, NOT verified unique operating companies.
- [x] **7. Test Verification**: Full unit test suite with mocked transport (22 tests) covering all failure modes, pre-request redirect downgrade rejection, contact User-Agent requirement, and path traversal sanitization passes under `tier2c`. Existing adapter and validator suites pass with zero regressions (124 passed: 22 acquisition + 102 adapter/validator).
- [x] **8. Bounded Live Verification**: Exactly one bounded live request per approved public source verifies live payload structure, logged outside git. Extracted Nasdaq Trader footer timestamp (`1016202614:01`) recorded without assuming a fixed 03:00 ET schedule. Elapsed session duration (~20 min) noted as operator session time, not benchmark capacity.

