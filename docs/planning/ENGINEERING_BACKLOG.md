# ARX Terminal — Engineering Backlog & Follow-Up Activities

**Document ID**: `BACKLOG-ARX-SPRINT-1-CLOSEOUT-RECONCILIATION`  
**Classification**: Authoritative Engineering Backlog & Follow-Up Register  
**Governing Authorization**: ARX Terminal — Sprint 1 Final Consolidation & Closeout (Product Owner Directive)  
**Baseline Commit**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d` (PR #8 Merged Production Baseline)  
**Baseline Tree**: `dd4b1dfe4c0644cdc3c7aa0edfa87621b01b88e9`  
**Date**: 2026-10-16T19:30:00+02:00  
**External Tracking State**: `REGISTRATION_PENDING_EXTERNAL_UPDATE`  

---

## 1. Backlog Prioritization Schema

| Priority Tier | Classification | Operational & Architectural Impact | Target Sequencing |
| :--- | :--- | :--- | :--- |
| **P0 (Release Blocker / Safety)** | Immediate production safety, defect containment, non-breaking read invariants. | Eliminates latent scan-on-read regressions, lease acquisition races, and false-positive UI state representation. | Next immediate bounded release (PR candidate) |
| **P1 (High Impact Capability)** | Controlled feature reintroductions, schema alignment, data preparation. | Reinstates verified architectural capabilities without coupling or regression hazards. | Post-safety stabilization |
| **P2 (Strategic Discovery / Evaluation)** | Research, dataset discovery, provenance audits, external source qualification. | Establishes feasibility, licensing, temporal validity, and empirical utility before architecture commit. | Next available sprint with capacity |
| **P3 (Maintenance & Optimization)** | Performance tuning, documentation hygiene, non-critical telemetry refinements. | Continuous engineering hygiene without direct customer-facing impact. | Ongoing |

---

## 2. Activity Registers

### `BACKLOG-VCP-READ-SAFETY-001`: Read-Safety Release
- **Category**: Production Defect Remediation / Read-Safety Contract
- **Priority**: **P0 (Immediate Delivery)**
- **Priority Rationale**: Proactive elimination of latent scan-on-read execution, lease acquisition races, and potential unperformed scan states on cache miss or release mismatch. (Note: Proactive bounded defect remediation; does not imply an active production incident).
- **Status**: `RELEASED_TO_PRODUCTION` (Deployed via PR #4, commit `787c1977b0071f2b31c4ad3d36a7f872147ad59e`)
- **Release Target**: Release Candidate PR #4 (Completed)
- **Assigned Sprint**: `SPRINT_UNASSIGNED` (Tracked via Release Target, not sprint allocation)
- **Owner**: ARX Terminal Core Team
- **Scope**:
  - Implement and enforce the bounded inactive read-safety contract across `analyst_dashboard/analyzers/scanner_runner.py`.
  - Remove all scan-on-read execution paths: public snapshot reads (`get_active_or_latest_snapshot()`) must NEVER trigger `execute_market_wide_scan()`, acquire execution leases, write to SQLite/Postgres storage, or publish snapshots.
  - Introduce explicit, structured status envelopes for all lifecycle states:
    - Empty cache / missing snapshot -> `ScannerStatus.PIPELINE_PENDING` with reason `NO_ACTIVE_SNAPSHOT`.
    - Engine release mismatch -> `ScannerStatus.PIPELINE_PENDING` with reason `RELEASE_MISMATCH`.
    - Storage exception / disk fault -> `ScannerStatus.ERROR` with reason `STORAGE_ERROR`.
    - Expired data past freshness cutoff -> `ScannerStatus.STALE` with truthful `data_as_of` timestamp.
  - Pin root-level metadata contract (`snapshot_id`, `matched_count`, `data_as_of`, `generated_at`).
  - Add durable Tier 2 regression suite (`tests/test_vcp_inactive_read_safety.py`) covering all 6 core scenarios.
- **Dependencies**:
  - `origin/main` commit `fcb43f7aae1797bde762b91b5c309817ab25f459` (PR #3 merged baseline).
- **Acceptance Criteria**:
  1. `VCPScannerRunner.get_active_or_latest_snapshot()` never calls `execute_market_wide_scan()`.
  2. Zero execution leases requested or held during public GET requests.
  3. Zero database writes, snapshot creations, or side effects on reads.
  4. Missing snapshots return HTTP 200 with envelope `status: "PIPELINE_PENDING"`, `matched_count: null`.
  5. All 6 scenarios in `tests/test_vcp_inactive_read_safety.py` pass.
  6. Full Tier 1, Tier 2, and Tier 3 suites pass with zero regressions against PR #3 baseline.
- **Release Record**:
  - Verified in production deployment on Railway (`6bdc3de9-39d6-4e4b-abb5-09e53dfb7445`) and Cloudflare Pages (`0b0fa048-51ab-4098-8c0d-4794ead8c854`).

---

### `BACKLOG-VCP-UNIVERSE-WIRING-002`: Candidate 004 Universe-Wiring Reintroduction (Track A)
- **Category**: Architecture / Domain Feature Reintroduction
- **Priority**: **P1 (Follow-Up Delivery)**
- **Status**: `RELEASED_TO_PRODUCTION` (Production Verified; Release Evidence Closed)
- **Release Target**: Candidate 004 Universe Wiring Release (PR #5)
- **Assigned Sprint**: Sprint 1
- **Owner**: ARX Terminal Core Team
- **Delivery Details**:
  - **Pull Request**: [daakara/finance#5](https://github.com/daakara/finance/pull/5)
  - **Merged Release Commit**: `2eb032b2aec18ac4fea0f654eee8fb6c028466fe`
  - **Merged Tree**: `483bee72e2f8df63adecf15891b6e0091d52a738`
- **Scope & Fulfillment**:
  - Reintroduced Candidate 004 universe, session, and publication wiring decoupled from public read endpoints.
  - Preserved relational natural denominator and epoch membership verification (`SPRINT3_CANDIDATE004_EPOCH_001`).
  - Authoritative 20-condition test suite (`tests/test_vcp_c004_universe_wiring.py`) passed cleanly.
  - Zero changes to frozen engines (`optimal_execution.py`, `confluence_engine.py`, `decision_hierarchy.py`) or historical manifests.
- **Operational Status**:
  - Candidate 004 universe wiring remotely verified on `main`. Autonomous scheduler arming remains `NOT_AUTHORIZED` pending separate authorization.

---

### `BACKLOG-VCP-PROD-UNIVERSE-003`: Production Universe Preparation & Publication (Track C)
- **Category**: Operations / Data Asset Governance
- **Priority**: **P1 (Operational Delivery / Core Infrastructure)**
- **Status**: `PARTIALLY_DELIVERED_OPERATIONALLY_INCOMPLETE`
- **Assigned Sprint**: Sprint 1 (Research & Slices 1–2 Delivered; Operational Slices Deferred to Sprint 2+)
- **Owner**: ARX Terminal Core Team / Data Asset Governance
- **Operational Incompleteness Attestations (Mandatory Invariants)**:
  ```ini
  LIVE_INGESTION = NOT_IMPLEMENTED
  SOURCE_QUALIFICATION = NOT_ESTABLISHED
  UNIVERSE_BUILD/PUBLICATION = NOT_PERFORMED
  FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED
  ```
- **Delivered Deliverables**:
  1. **Research & Specification Package**:
     - 12 comprehensive architecture and governance documents authored in `docs/research/vcp-production-universe-inputs/` (pinned commit `6d1432585dac2dcfaceb2794d73dc51d1fc7ac1b`, tree `ad7b13fb814ce4575ea6027b68de6bef11b2b331`).
     - Documents: `CONTRACT_TO_SOURCE_MATRIX.md`, `EVIDENCE_INDEX.md`, `FIELD_EVIDENCE_REQUIREMENTS.md`, `GAP_RESOLUTION_OPTIONS.md`, `IMPLEMENTATION_PLAN.md`, `INPUT_QUALIFICATION_REPORT.md`, `INPUT_SPECIFICATION.md`, `INPUT_VALIDATION_SPEC.md`, `PLANNING_HANDOFF.md`, `POLICY_DECISION_PROPOSALS.md`, `PUBLICATION_HANDOFF.md`, `PUBLICATION_TOOLING_SPEC.md`.
  2. **Slice 1: Policy-Independent Input Validation Core**:
     - **Pull Request**: [daakara/finance#6](https://github.com/daakara/finance/pull/6)
     - **Merged Release Commit**: `5f26f3fcf8a4fa3140401bb16a687ec86c53de0d`
     - **Merged Tree**: `04000d9516f1d2a1252caed51bc6b082d01484d1`
     - **Implementation Note**: [`docs/research/vcp-production-universe-inputs/SLICE_1_IMPLEMENTATION_NOTE.md`](file:///docs/research/vcp-production-universe-inputs/SLICE_1_IMPLEMENTATION_NOTE.md)
     - **Status**: Released; standalone validator remains disconnected from live ingestion.
  3. **Slice 2: Offline Adapters & Validator Bridge**:
     - **Pull Request**: [daakara/finance#8](https://github.com/daakara/finance/pull/8)
     - **Merged Release Commit**: `31af2393d81504d24fffff4b10bcc9e0539e5a6d`
     - **Merged Tree**: `dd4b1dfe4c0644cdc3c7aa0edfa87621b01b88e9`
     - **Implementation Note**: [`docs/research/vcp-production-universe-inputs/SLICE_2_OFFLINE_ADAPTER_NOTE.md`](file:///docs/research/vcp-production-universe-inputs/SLICE_2_OFFLINE_ADAPTER_NOTE.md)
     - **Status**: Production verified; offline in-memory parsing only; no live source acquisition.
- **Remaining Operational Work (Deferred to Sprint 2+)**:
  - Implementation of real live network acquisition clients for SEC EDGAR, Nasdaq Trader, and OpenFIGI (`LIVE_INGESTION = NOT_IMPLEMENTED`).
  - Formal source qualification protocol and certification against authoritative primary agency feeds (`SOURCE_QUALIFICATION = NOT_ESTABLISHED`).
  - Governed execution of `scripts/publish_universe.py` with cryptographic signing and storage audit receipts (`UNIVERSE_BUILD/PUBLICATION = NOT_PERFORMED`).
  - Full US-listed assessment & denominator accounting across active Radar universe (`FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED`).
  - Execution boundaries: Strictly `NOT_AUTHORIZED` in Sprint 1; carryover to Sprint 2+.

---

### `BACKLOG-DATA-GOV-DISCOVERY-001`: Data.gov Discovery and Primary-Source Qualification (Track B)
- **Category**: Research / Data Provenance Discovery
- **Priority**: **P2 (Strategic Evaluation)**
- **Status**: `DISCOVERY_COMPLETE_WITH_LIMITATIONS` (Closed with Documented Limitations)
- **Assigned Sprint**: Sprint 1 (Completed & Closed)
- **Branch**: `research/data-gov-discovery`
- **Corrected Commit**: `0d7cd818da352c0c81c0126f06f4f362d613aec1`
- **Tree**: `0355aaf972eb8048ecc557df26635d1d315b02ff`
- **Owner**: ARX Terminal Core Team
- **Deliverables**:
  - [`docs/research/data-gov-discovery/DISCOVERY_REPORT.md`](file:///docs/research/data-gov-discovery/DISCOVERY_REPORT.md)
  - [`docs/research/data-gov-discovery/SOURCE_QUALIFICATION_MATRIX.md`](file:///docs/research/data-gov-discovery/SOURCE_QUALIFICATION_MATRIX.md)
  - [`docs/research/data-gov-discovery/EVIDENCE_INDEX.md`](file:///docs/research/data-gov-discovery/EVIDENCE_INDEX.md)
  - [`docs/research/data-gov-discovery/PLANNING_HANDOFF.md`](file:///docs/research/data-gov-discovery/PLANNING_HANDOFF.md)
- **Evaluated Source Decisions & Reopening Conditions**:
  1. **CFTC Commitments of Traders (COT)**: `DEFER`. Qualified as descriptive macro sentiment and speculative positioning context for equity index futures (`E-MINI S&P 500`). Deferred pending architectural design of point-in-time dual timestamps (`observed_as_of` Tuesday vs `available_at` Friday 15:30 ET) and instrument mapping dictionaries. Reopening condition: Prioritization of macro sentiment layer.
  2. **U.S. Treasury Fiscal Data (Operating Cash Balance / TGA)**: `DEFER`. Qualified as descriptive sovereign liquidity metric (daily cash balance and net flow). Deferred pending product prioritization. Reopening condition: User demand for sovereign liquidity context in Macro Ribbon.
  3. **BLS CPI API**: `REJECT_FOR_CURRENT_USE_CASE`. Redundant with existing FRED integration (`fred_fetcher.py`). Reopening condition: Validated need for granular non-headline CPI sub-indices not indexed by FRED.
  4. **BEA Industry Economic Accounts**: `REJECT_FOR_CURRENT_USE_CASE`. Multi-month publication latency (60–90 days) and extensive retroactive revisions make it unsuitable for active swing trading horizons. Reopening condition: Structural multi-year macroeconomic sector weighting module.
  5. **OFAC SDN Sanctions List**: `REJECT_FOR_CURRENT_USE_CASE`. Out of current product scope; no requirement for entity sanctions screening on US-listed exchange equities. Reopening condition: Compliance mandate for OTC or international listings.
- **Architectural Qualification**:
  - Data.gov is strictly a discovery catalog layer, not a runtime evidence authority.
  - Data.gov research is explicitly **NOT a prerequisite** for US-listed stock scanning.
- **Proposed Follow-Up Engineering Items (Sprint 2+ Candidates; Implementation NOT Authorized)**:
  - `BACKLOG-MACRO-COT-FEASIBILITY-001` (CFTC COT Anti-Lookahead Ingestion & Instrument Mapping Design; P3, estimated 2–3 days).
  - `BACKLOG-TREASURY-TGA-LIQUIDITY-001` (Treasury TGA Sovereign Liquidity Display Feasibility; P4, estimated 1 day).

---

### `BACKLOG-ARX-ANALYSIS-001` / `UX-ANALYSIS-001`: Canonical Decision-State Consistency & Truthful Corridor Location (Track D)
- **Category**: Defect Remediation / State Invariant / UX Architecture
- **Priority**: **P1 (High Impact Capability)**
- **Status**: `DELIVERED_AND_RELEASED` (Production Verified; Release Record Corrected & Closed)
- **Release Target**: PR #7
- **Assigned Sprint**: Sprint 1 (Slice 1 Delivered; Remaining Proposals Deferred to Sprint 2+)
- **Owner**: ARX Terminal Core Team
- **Delivery Details**:
  - **Pull Request**: [daakara/finance#7](https://github.com/daakara/finance/pull/7)
  - **Merged Release Commit**: `11d6813500270788085989bde228190b5b6d3629`
  - **Merged Tree**: `0bfb2f004fc36a4cf594cbd69fcdbff4f2385994`
  - **Local Evidence Cleanup Addendum Commit**: `d20cc9a3471a166f60a248701fd639bdda6bc017`
  - **Main CI Run**: [Run #38919069761](https://github.com/daakara/finance/actions/runs/38919069761) (`SUCCESS`)
  - **Railway Deployment**: `ea33ea7e-6c96-49a4-b704-364814acf500` (`SUCCESS`)
  - **Cloudflare Pages Production URL**: `https://finance-xp8.pages.dev`
- **Delivered Scope (Slice 1)**:
  - Reconciled spatial copy contradictions: eliminated false claims that price extended above ceiling is "inside" corridor.
  - Suppressed actionable scale-in instructions whenever canonical action or technical gates are blocking.
  - Introduced 3-tier readiness presentation: `EXECUTION_READY`, `AWAITING_CANONICAL_CLEARANCE`, `SETUP_IN_PROGRESS`.
  - Reconciled workflow step title to actual 145 SSG routes in CI.
  - Preserved existing 2% implementation tolerance with explicit contract qualification.
  - Verified across 18-case automated regression suite (`tests/corridorDecisionConsistency.test.ts`) and 10 controlled browser fixture captures (`scripts/verify-slice1-states.mjs`).
- **Research & Governance Artifacts**:
  - [`docs/research/analysis-ux-audit/AUDIT_REPORT.md`](file:///docs/research/analysis-ux-audit/AUDIT_REPORT.md)
  - [`docs/research/analysis-ux-audit/BACKLOG_HANDOFF.md`](file:///docs/research/analysis-ux-audit/BACKLOG_HANDOFF.md)
  - [`docs/research/analysis-ux-audit/EVIDENCE_ADDENDUM.md`](file:///docs/research/analysis-ux-audit/EVIDENCE_ADDENDUM.md)
  - [`docs/research/analysis-ux-audit/EVIDENCE_INDEX.md`](file:///docs/research/analysis-ux-audit/EVIDENCE_INDEX.md)
  - [`docs/research/analysis-ux-audit/SLICE_1_DECISION_CONSISTENCY.md`](file:///docs/research/analysis-ux-audit/SLICE_1_DECISION_CONSISTENCY.md) (with Section 8 Evidence-Cleanup Addendum & Sprint 1 Closeout).
- **Follow-Up Proposed UX Items (Deferred to Sprint 2+; Implementation NOT Authorized in Sprint 1)**:
  - `UX-ANALYSIS-002`: Calculation & Score Provenance Disclosures (explicit reference denominators for stops, targets, and blended 50/50 R:R).
  - `UX-ANALYSIS-003`: Data-Context and Freshness Clarity (standardized data freshness, exchange timestamps, live vs delayed vs demo badges).
  - `UX-ANALYSIS-004`: Content Consolidation & Institutional Wording (eliminate redundant setup text and replace informal slang).
  - `UX-ANALYSIS-005`: Ownership-Aware Analysis Flow (cost-basis capture, basis-relative portfolio metrics).
  - `UX-RADAR-001`: Actionability and Confluence Filters on Radar.
  - `UX-ANALYSIS-006`: Supporting-Evidence Redesign (4-pillar scorecard with explicit 25% weights and deconstructed scores).
  - `UX-MOBILE-001`: Mobile Readability & Accessibility Remediation (sticky bottom bar collisions, 44px touch targets, ARIA attributes).
  - Separate Quantitative Review of 2% Tolerance: Formal review of whether the existing frontend implementation tolerance should be ratified or replaced with dynamic ATR-based boundaries in backend engines.

---

### `BACKLOG-RADAR-US-LISTED-COVERAGE-001`: Radar US-Listed Coverage & Requirements Reconciliation (Track E)
- **Category**: Product Requirements & Discovery Architecture
- **Priority**: **P1 (High Impact Capability / Architectural Alignment)**
- **Priority Rationale**: Reconciles the Product Owner's confirmed mandate of market-wide discovery across ALL US-listed stocks (~6,000–9,000 securities, including foreign companies listed on US exchanges) with existing code execution, which currently evaluates a 35-symbol historical regression fixture. Resolves policy boundaries (Rule R02 Common Stock vs ADRs), defines mathematical accounting closure across 6 conceptual outcomes, and establishes the 10-package delivery sequence (S2-01 to S2-10) for full-market Radar screening.
- **Status**: `RECONCILIATION_COMPLETE_PENDING_PO_POLICY_DECISIONS`
- **Release Target**: Post-PR #8 Universe Delivery Phase (Sprint 2 Foundation / Sprint 3 Activation)
- **Assigned Sprint**: `SPRINT_UNASSIGNED` (Tracked via capability delivery sequence, not arbitrary calendar sprint)
- **Owner**: `OWNER_UNASSIGNED` (Pending Product Owner resource allocation)
- **Core Product Owner Acceptance Criterion**:
  > **“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”**
  - Explicit Note: A pilot, comprehensive directory, or successful parser alone does NOT satisfy this criterion.
- **Ordered Implementation Sequence (S2-01 to S2-10)**:
  - **S2-01 (Population and Strategy Boundaries)**: Explicit population scope and separate assessment/eligibility treatment. No silent omission of unresolved stock types.
  - **S2-02 (Listing Acquisition & Source Evidence)**: Supported acquisition of SEC EDGAR and Nasdaq directories, preserved source versions, completeness checks, and explicit missing/partial-source behavior.
  - **S2-03 (Security Identity & Evidence)**: Identity granularity (dual-class handling), classification, currency, venue listing/status evidence; unresolved records remain explicitly visible.
  - **S2-04 (Connected Adapter/Validator/Builder Integration)**: Real acquired observations flow through existing components without fabricated defaults; eligibility and readiness accounting reconciles.
  - **S2-05 (Market-Data Coverage & Capacity Verification)**: Measured history/quote coverage and execution capacity over defined population. Missing data is explicit; scaling architecture follows evidence.
  - **S2-06 (Governed Build/Publication Tooling)**: Reviewed tooling, repeatability, safe failure behavior, and authorization boundaries. Publication does NOT activate scanning.
  - **S2-07 (Scanner & Radar Membership Integration)**: Published membership drives assessment; fixed candidate lists do not restrict discovery or discard valid out-of-list results.
  - **S2-08 (Full-Population Assessment & Outcome Accounting)**: Every stock in the defined population has a reconciled terminal outcome:
    $$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$
    Retries, failures, unsupported assessments, and unavailable data are distinguished from "no qualifying setup". Execution attempts ($N_{\text{attempts}}$) are tracked separately from distinct securities ($N_{\text{securities}}$).
  - **S2-09 (Radar Coverage Disclosure & Analysis Handoff)**: Coverage, timestamps, and limitations are visible on UI. Analysis preserves selected evaluation or clearly identifies a newer assessment.
  - **S2-10 (Operational Acceptance & Controlled Activation)**: Cadence (Eastern Time exchange calendar aware), freshness, retry/failure handling, and monitoring verified before separately authorized activation. Partial results cannot appear complete.
- **Prioritization & Phased Delivery**:
  - **Sprint 2 Focus (Proposed)**: Foundation, Ingestion, Validation & Capacity (S2-01 to S2-06) $\to$ produces an authorized published build (labeled partial milestone; scanning inactive).
  - **Sprint 3 Focus (Proposed)**: Scanner Integration, Full-Population Accounting & Controlled Activation (S2-07 to S2-10).
  - Sprint 2/3 allocation is strictly **PROPOSED**, not scheduled or capacity-approved. Publication and activation require separate later authorization.
- **Retained Architecture Qualifications**:
  - Population inclusion is separate from strategy eligibility.
  - Venue/type boundaries and policy details remain explicit decisions.
  - 6,000–9,000 is an estimate, not an established denominator.
  - Historical 20-match snapshot population attribution is unestablished.
  - Fixed/fallback code paths and route filtering are separate findings.
  - A diagnostic subset is not full-market acceptance.
  - Scaling choices require measurement.
  - Cadence must be exchange-calendar/timezone aware.
  - Terminal outcomes need disjoint definitions and precedence.
  - Retries are not additional distinct securities.
- **Research Deliverables**:
  - [`docs/research/radar-us-listed-coverage/AUTHORITY_AND_CURRENT_STATE.md`](file:///docs/research/radar-us-listed-coverage/AUTHORITY_AND_CURRENT_STATE.md)
  - [`docs/research/radar-us-listed-coverage/BACKLOG_HANDOFF.md`](file:///docs/research/radar-us-listed-coverage/BACKLOG_HANDOFF.md)
  - [`docs/research/radar-us-listed-coverage/COVERAGE_ACCEPTANCE_CONTRACT.md`](file:///docs/research/radar-us-listed-coverage/COVERAGE_ACCEPTANCE_CONTRACT.md)
  - [`docs/research/radar-us-listed-coverage/DELIVERY_CHECKLIST.md`](file:///docs/research/radar-us-listed-coverage/DELIVERY_CHECKLIST.md)
  - [`docs/research/radar-us-listed-coverage/PRD_ADDENDUM_PROPOSAL.md`](file:///docs/research/radar-us-listed-coverage/PRD_ADDENDUM_PROPOSAL.md)
- **Dependencies & Inter-Backlog Cross-Links**:
  - `BACKLOG-VCP-READ-SAFETY-001` (PR #4): Read-safety contract strictly preserved.
  - `BACKLOG-VCP-UNIVERSE-WIRING-002` (PR #5): Candidate 004 universe wiring and session guard structures.
  - `BACKLOG-VCP-PROD-UNIVERSE-003` (PR #6, PR #8): Production universe publication tooling; builds upon S2-02 to S2-06.
  - `BACKLOG-DATA-GOV-DISCOVERY-001`: Primary agency provenance foundation.
- **Next Step**:
  - Obtain Product Owner determinations on the 5 policy questions in `COVERAGE_ACCEPTANCE_CONTRACT.md` to unblock S2-01 execution.

---

### `BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-043`: Evidence-Led Opportunity Discovery Pilot (Issue #43)
- **Category**: Research / Opportunity Discovery & Hypothesis Tracking
- **Priority**: **P2 (Strategic Discovery / Evaluation)**
- **Priority Rationale**: Tests whether evidence-led discovery beyond the initial 10-candidate shortlist improves ARX's research prioritization enough to justify ongoing preparation and maintenance effort.
- **Governing Issue**: [daakara/finance#43](https://github.com/daakara/finance/issues/43)
- **Status**: `COMPLETED_AND_ADOPTED` (Adopted into Mainline via PR #45; Issue #43 Closed as Completed)
- **Assigned Sprint**: Research Sprint 2 (Completed & Closed)
- **Owner**: ARX Terminal Core Team
- **Completed Prerequisites**:
  - PR #41 merged to `main` at commit `b9e9fe3a801f42e8a93d9f090ec67ec76a515bf3` (Tree `53ed31419e172ed4b309bbba6bb4e4e3eda90c9b`).
  - Active published Quality Watch collection: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK`.
- **Scope & Methodological Core**:
  - Combines multiple candidate-generation routes (company financial change, company operating change, industry constraints, price-vs-business divergence, customer/supplier tracing) beyond the initial 10-candidate shortlist (`KLAC, LULU, EME, VRT, ANET, CDNS, PWR, SNPS, NOW, ELF`).
  - Implements compact reusable opportunity record answering:
    1. Observable development (structural, cyclical, acquisition, accounting, temporary).
    2. Economic significance (materiality to consolidated cash flows).
    3. Value capture (capex, competition, debt service, SBC/dilution, explicit limitations).
    4. Price and expectations (reconciled to primary filings and conservative valuation bounds).
    5. Downside and survival (debt covenants, financial resilience under slowdown).
    6. Assessment changes (observable disconfirmation criteria).
    7. Research priority relative to examined universe.
  - Strict separation of dimensions: research maturity, opportunity mechanism, expectation-gap status, financial resilience, and public publication eligibility.
  - Concludes with an evidence-based evaluation report delivering a qualified process decision: `CONTINUE`.
- **Standardized Decision-Focused Research Process**:
  - Authoritative process standard codified in [`docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md`](file:///docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md) (`ARX-DECISION-RESEARCH-PROCESS-V1`).
  - Enforces universal 6-step decision workflow across discovery, quality-watch preparation, and dossier maintenance:
    - Step A: Define the Decision (Investigate Further, Prioritize Queue, Prepare Quality-WATCH, Maintain/Revise, Deeper Analysis).
    - Step B: Identify Decisive Uncertainties (few questions capable of changing outcome; no universal risk checklists).
    - Step C: Connect Outcomes to Consequences (cash generation, debt service, dilution, downside resilience).
    - Step D: Investigate Proportionately (primary SEC evidence, facts vs management statements vs calculations, counter-evidence).
    - Step E: Make the Assessment (supported conclusions, unresolved items, explicit next action).
    - Step F: Set Review Triggers and Stop (observable falsification events, clear stopping rule).
  - Business-specific illustrative questions documented for `DECK`, `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`.
  - Canonical 9-field reusable research record template established.
  - Prospectively integrated into `QUALITY_WATCH_SELECTION_STANDARD.md` and `PILOT_MAINTENANCE_POLICY.md` (completed releases remain closed; no retrospective records fabricated).
- **Delivered Milestones**:
  - **Merged Pull Request**: [daakara/finance#45](https://github.com/daakara/finance/pull/45) (Merge Commit: `9d74ea903098e6c3628f8fab433de425a82b024c`, Candidate Commit: `12cd34671cd04acaea457173b8a0a1db50943dec`, Tree: `98178b045f9a7e992b804e9870cd404283872921`).
  - **Adopted Standard**: `ARX-DECISION-RESEARCH-PROCESS-V1` codified in [`docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md`](file:///docs/research/DECISION_FOCUSED_RESEARCH_PROCESS.md).
  - **Discovery Pilot Evaluation**: Qualified `CONTINUE` decision rendered in [`docs/research/evidence_led_discovery/PILOT_EVALUATION_REPORT.md`](file:///docs/research/evidence_led_discovery/PILOT_EVALUATION_REPORT.md).
  - **Closed Governing Issue**: [daakara/finance#43](https://github.com/daakara/finance/issues/43) closed as completed.
- **Operational Boundaries & Invariants**:
  - Internal research pilot only: no public publication, no manifest additions, no production deployment.
  - Zero scanner activation, universe expansion, or live market-data ingestion changes.
  - Baseline ledger (`odp05_discovery_ledger.json`) SHA-256 (`a65be3b...`) remains strictly immutable.
  - Keep internal pilot records completely outside frontend import paths and bundles.
  - Do not promise identification of future multibaggers or claim proven investment alpha.

---

### `BACKLOG-RESEARCH-EVIDENCE-DISCOVERY-046`: Evidence-Led Opportunity Discovery Cycle 2
- **Category**: Research / Opportunity Discovery & Hypothesis Tracking
- **Priority**: **P2 (Strategic Discovery / Evaluation)**
- **Priority Rationale**: Tests whether a reproducible search process can surface research-worthy opportunities beyond ARX's existing manually selected cohort, establishing whether ongoing discovery justifies effort vs relying on manual queue additions.
- **Status**: `IN_PROGRESS` (Active Branch: `feat/research-evidence-led-discovery-cycle-2`)
- **Assigned Sprint**: Research Sprint 3 (Discovery Capability Track)
- **Owner**: ARX Terminal Core Team
- **Completed Prerequisites**:
  - Process Standard `ARX-DECISION-RESEARCH-PROCESS-V1` adopted into `main` via PR #45 (Commit `9d74ea903098e6c3628f8fab433de425a82b024c`).
  - Active published Quality Watch collection: `FIX`, `DUOL`, `ULTA`, `MEDP`, `ISRG`, `DECK` (6 symbols intact).
- **Scope & Methodological Core**:
  - Tests 2 reproducible discovery routes across an examined universe of 14 operating companies:
    - Route A: Company-Led Operating / Cash-Conversion Change & Transition to Self-Funding (`TMDX`, `PODD`, `NVR`, `APP`, `LNTH`, `TDG`, `MEDP`).
    - Route B: Industry-Led Structural Change & Value-Chain Constraint Tracing (`ETN`, `CEG`, `POWI`, `HUBB`, `GEV`, `PWR`, `VRT`).
  - Bounded active-work budget with honest stage effort logging.
  - 6-step decision workflow applied proportionately to triage candidates and investigate deep records (`TMDX`, `ETN`).
  - Evaluates search reproducibility, coverage achieved, candidate novelty, questions surfaced, priority defensibility, upkeep effort, and publication readiness.
- **Operational Boundaries & Invariants**:
  - Internal research experiment only: no public publication, no manifest additions, no production deployment.
  - Zero scanner activation, universe expansion, or live market-data ingestion changes.
  - Baseline ledger (`odp05_discovery_ledger.json`) SHA-256 (`a65be3b...`) remains strictly immutable.
  - Internal records remain completely outside frontend import paths and bundles.
  - Discloses explicit limitations; no claims of predictive alpha or guaranteed returns.

---

## 3. Governance & System Traceability

- **Registration State**: Reconciled and integrated on PR branch (`docs/sprint-1-closeout`), pending PR merge and Product Owner review.
- **External Synchronization**: If enterprise planning (Jira, Linear, GitHub Projects) is authoritative, these records are marked `REGISTRATION_PENDING_EXTERNAL_UPDATE` pending external synchronization. Local files are not represented as autonomous mutations of external tracking systems.
- **Precedence Hierarchy**: All registered items are subordinate to `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md` and `FROZEN_ENGINE_MANIFEST_V2_5_1.json`.
