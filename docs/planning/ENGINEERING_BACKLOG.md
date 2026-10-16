# ARX Terminal — Engineering Backlog & Follow-Up Activities

**Document ID**: `BACKLOG-ARX-POST-PR3-CONTINUATION`  
**Classification**: Authoritative Engineering Backlog & Follow-Up Register  
**Governing Authorization**: ARX Terminal — Post-PR3 Delivery Continuation & Planning Updates (Product Owner Decisions)  
**Baseline Commit**: `fcb43f7aae1797bde762b91b5c309817ab25f459` (PR #3 Runtime Authority Correction Baseline)  
**Date**: 2026-10-16T08:42:00+02:00  
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
- **Status**: `LOCAL_VERIFICATION_COMPLETE` (Ready for remote preflight)
- **Release Target**: Next Release Candidate (Post-PR #3)
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
- **Next Step**:
  - Push branch to remote and observe CI and non-production preview.

---

### `BACKLOG-VCP-UNIVERSE-WIRING-002`: Candidate 004 Universe-Wiring Reintroduction
- **Category**: Architecture / Domain Feature Reintroduction
- **Priority**: **P1 (Follow-Up Delivery)**
- **Status**: `HELD_PENDING_DEPENDENCY_REVIEW`
- **Release Target**: Post-Read-Safety Release
- **Assigned Sprint**: `SPRINT_UNASSIGNED`
- **Owner**: `OWNER_UNASSIGNED`
- **Scope**:
  - Re-evaluate and reintroduce the Candidate 004 universe, session, and publication wiring that was rolled back in Candidate 004 recovery (PR #2).
  - Explicit Note: The rollback of Candidate 004 was an operational containment revert driven by migration timing, lease races, and scan-on-read coupling; it does NOT constitute cancellation of the feature requirements.
  - Perform dependency analysis between universe lifecycle, session management, and runtime scanner execution.
  - Ensure universe-wiring contracts remain decoupled from public GET request lifecycles.
  - Preserve all valid Candidate 004 domain models, historical migration units, and relational denominator integrity invariants.
- **Dependencies**:
  - Successful deployment, verification, and stabilization of `BACKLOG-VCP-READ-SAFETY-001`.
  - Comprehensive review of session/universe lifecycle dependencies.
- **Acceptance Criteria**:
  1. Complete decoupling between scanner runtime execution and API read pathways.
  2. Preserved relational natural denominator and epoch membership verification.
  3. Migration scripts and DDL verified against isolated database environments.
  4. Dedicated candidate freeze manifest and release package generated independently.
- **Next Step**:
  - Architecture and dependency review of Candidate 004 universe/session lifecycle.

---

### `BACKLOG-VCP-PROD-UNIVERSE-003`: Production Universe Preparation & Publication
- **Category**: Operations / Data Asset Governance
- **Priority**: **P1 (Operational Delivery)**
- **Status**: `HELD_PENDING_UNIVERSE_POLICY_AUTHORIZATION`
- **Release Target**: Controlled Data Deployment Cycle
- **Assigned Sprint**: `SPRINT_UNASSIGNED`
- **Owner**: `OWNER_UNASSIGNED`
- **Scope**:
  - Operational preparation and publication of the production universe asset for VCP and Radar scanners.
  - Strict decoupling from application code deployment: universe publication is a governed data-asset workflow, not a CI/CD deployment side effect.
  - Review, test, and harden `scripts/publish_universe.py` under authorized execution boundaries.
  - Proposed cryptographic signing, publication policy verification, and durable storage audit receipts for all published universe artifacts (note: signing is proposed for policy evaluation; no signing implementation introduced under this task).
  - Note: Strictly NO production universe generation, publication, or execution under the current task.
- **Dependencies**:
  - `BACKLOG-VCP-UNIVERSE-WIRING-002` dependency architecture completion.
  - Authorized publication policy and verified primary market universe inputs.
- **Acceptance Criteria**:
  1. Authoritative primary data sources verified and audited.
  2. Universe publication policy validated against domain governance rules.
  3. Durable storage schema validated for published universe snapshots.
  4. Script execution verified in dry-run/preview mode with zero unsolicited write access.
- **Next Step**:
  - Review publication policy and storage authorization requirements.

---

### `BACKLOG-DATA-GOV-DISCOVERY-001`: Data.gov Discovery and Primary-Source Qualification
- **Category**: Research / Data Provenance Discovery
- **Priority**: **P2 (Strategic Evaluation)**
- **Status**: `PO_APPROVED_FOR_NEXT_AVAILABLE_SPRINT`
- **Assigned Sprint**: `PO_APPROVED_FOR_NEXT_AVAILABLE_SPRINT`
- **Sprint Scheduling Status**:
  - **Approval**: Product Owner approved for earliest sprint with room for 1-day timebox.
  - **Sprint ID / Dates**: Missing authoritative sprint calendar and capacity records in repository. Sprint schedule is held in external tracking system. Exact sprint ID and calendar dates pending external sprint planning synchronization.
- **Owner**: `OWNER_UNASSIGNED` (No supported assignment exists in repository records; not invented)
- **Estimate / Timebox**: 1 working day (8 hours), reassessed if scope requires more.
- **Scope**:
  - **Discovery Only**: Strictly prohibited from implementing connector code, API adapters, database tables, or network ingestion clients under this activity.
  - Identify and catalog relevant ARX Terminal use cases and candidate datasets hosted or indexed on Data.gov (e.g., macroeconomic indicators, industry production indices, trade statistics, debt/liquidity monitors).
  - Source Qualification Protocol:
    - Assess source provenance, agency ownership, and upstream update cadence.
    - Treat Data.gov strictly as a discovery and metadata catalog layer; qualify the actual primary agency sources (e.g., BEA, BLS, US Treasury, Federal Reserve, Census Bureau) for prospective runtime consumption.
    - Evaluate temporal/vintage availability: determine whether point-in-time / historical revision vintages are preserved or whether retroactive lookahead contamination exists.
    - Audit licensing, commercial terms, redistribution rights, rate limits, API stability, and uptime SLA.
    - Quantify incremental predictive or risk-management value relative to existing market data feeds.
  - Deliverable: Comprehensive, evidence-backed Adopt / Defer / Reject decision dossier.
- **Dependencies**:
  - **Zero technical dependencies established for discovery**: strictly NOT dependent on Candidate 004 or universe wiring.
- **Acceptance Criteria**:
  1. Identify relevant ARX use cases and candidate datasets cataloged on Data.gov.
  2. Qualify primary-agency sources discovered through Data.gov.
  3. Assess provenance, coverage, freshness, historical vintages, access/licensing, maintenance, and incremental product value.
  4. Produce an ADOPT / DEFER / REJECT recommendation with supporting evidence.
  5. Identify any justified follow-up implementation separately without premature architecture commit.
- **Next Step**:
  - Align sprint calendar and capacity in external planning system; schedule 1-day discovery timebox.

---

## 3. Governance & System Traceability

- **Registration State**: Staged locally in repository planning documentation under Product Owner authorization.
- **External Synchronization**: If enterprise planning (Jira, Linear, GitHub Projects) is authoritative, these records are marked `REGISTRATION_PENDING_EXTERNAL_UPDATE` pending external synchronization. Local files are not represented as autonomous mutations of external tracking systems.
- **Precedence Hierarchy**: All registered items are subordinate to `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md` and `FROZEN_ENGINE_MANIFEST_V2_5_1.json`.
