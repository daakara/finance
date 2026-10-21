# ARX TERMINAL — SPRINT 2 / TRACK C
# IMPLEMENTATION BACKLOG, ROLLBACK RUNBOOK & ENGINEERING PLAN

**Document Identifier:** `C_IMPLEMENTATION_BACKLOG.md`  
**Governing Authority:** ARX Closure Model v2 (Track C Ownership Boundary)  
**Track Responsibility:** Engineering Plan, Working-Day Estimates & Release Runbook  
**Target Milestone:** Sprint 2 / Package C Architecture & Readiness Review  
**Status:** COMPLETE / CANONICAL PLAN  

---

## 1. Executive Summary & Minimal Publication Candidate

This engineering plan outlines the sequential, phase-by-phase implementation required to bring the **Track C Expanded Universe Publication Architecture** from its current specification-only design state to production readiness.

To avoid operational risk, we define a **Minimal Publication Candidate (MPC)**:
Rather than attempting to publish all 133 securities simultaneously, Track C will first execute an isolated, controlled publication of an **8-Security Initial Cohort**:
- **3 Historical Pilots:** `AAPL`, `GOOG`, `GOOGL` (100% seasoned, known baseline, zero drift risk).
- **5 High-Conviction Sourced Equities:** `FIX`, `DUOL`, `ULTA`, `KLAC`, `DECK` (already verified under PR #31/#32 with primary-source SEC XBRL fundamentals, verified CIK bindings, and active US listings).

This MPC allows complete validation of Track C's state machine, Merkle attestation, and consumer firewalls before scaling to the full 133-security universe.

---

## 2. Six-Phase Implementation Sequence

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRACK C IMPLEMENTATION PHASES                          │
├─────────┬────────────────────────────────────────────────────────┬──────────────┤
│ Phase 1 │ Contract, Dataclass & State Machine Scaffolding        │ 1.5 Days     │
├─────────┼────────────────────────────────────────────────────────┼──────────────┤
│ Phase 2 │ Track B Certification Ingestion & Verification Harness │ 1.5 Days     │
├─────────┼────────────────────────────────────────────────────────┼──────────────┤
│ Phase 3 │ Isolated Namespace Dry-Run Build & Diff Tooling        │ 1.5 Days     │
├─────────┼────────────────────────────────────────────────────────┼──────────────┤
│ Phase 4 │ Consumer Gating & Anti-Actionability UI Badging        │ 1.5 Days     │
├─────────┼────────────────────────────────────────────────────────┼──────────────┤
│ Phase 5 │ Fast-Pointer Rollback Harness & Fault Injection Tests  │ 1.0 Days     │
├─────────┼────────────────────────────────────────────────────────┼──────────────┤
│ Phase 6 │ Formal Production Publication Gate & Operator Sign-off │ 1.5 Days     │
└─────────┴────────────────────────────────────────────────────────┴──────────────┘
Total Estimated Working Days: 8.5 Days (Nominal)
```

### Phase 1: Contract, Dataclass & State Machine Scaffolding
- **Objective:** Establish the core types, dataclasses, and state validation logic in pure Python without external dependencies.
- **Tasks:**
  1. Add `TrackBDataQualityCertification` and sub-models to `analyst_dashboard/universe/contracts.py`.
  2. Implement `PublicationStateMachine` enforcing the 7 lifecycle states and transition predicates.
  3. Author unit tests `tests/test_universe_publication_state_machine.py` validating that State $N+1$ cannot be instantiated without State $N$ evidence.
- **Estimated Effort:** 1.5 Working Days.

### Phase 2: Track B Certification Ingestion & Verification Harness
- **Objective:** Create the intake adapter that validates Track B certificates, cross-checks `identity.db`, and commits readiness ledger rows.
- **Tasks:**
  1. Implement `TrackBCertificationVerifier` checking Merkle digests, session counts ($\ge 50$), and calendar continuity.
  2. Build fail-closed triage: certificates with missing bars or unadjusted splits are logged to `universe_readiness_ledger` as `DATA_UNAVAILABLE`.
  3. Author integration tests `tests/test_track_b_certificate_ingestion.py` using mock valid and tampered certificates.
- **Estimated Effort:** 1.5 Working Days.

### Phase 3: Isolated Namespace Dry-Run Build & Diff Tooling
- **Objective:** Enhance `DeterministicUniverseBuilder` to support targeted namespaces and automated membership diff previews.
- **Tasks:**
  1. Update `DeterministicUniverseBuilder.build_universe()` to target `ARX_US_EQUITIES_EXPANSION_150` without mutating `ARX_US_EQUITIES`.
  2. Implement `compute_membership_diff(build_current, build_prior)` returning `(ADDED, REMOVED, UNCHANGED)` lists.
  3. Author CLI script `scripts/universe/build_universe_dry_run.py` emitting formatted diff summaries and accounting receipts.
- **Estimated Effort:** 1.5 Working Days.

### Phase 4: Downstream Consumer Gating & Anti-Actionability UI Badging
- **Objective:** Enforce anti-actionability firewalls across Radar, VCP scanner, and stock detail views.
- **Tasks:**
  1. Verify `VCPScannerRunner` rejects non-strategy-admitted symbols and prunes candidates with volume $< \$5\text{M}$/day.
  2. Update `frontend/lib/masterCatalog.ts` and `generateStaticParams()` to pre-render stock detail pages for the MPC cohort.
  3. Ensure `SourcedFundamentalsCard` and `PriceChart` display explicit status banners when data is uncataloged or dormant.
  4. Author frontend tests verifying zero execution ladders or trade pivots appear for uncompressed candidates.
- **Estimated Effort:** 1.5 Working Days.

### Phase 5: Fast-Pointer Rollback Harness & Fault Injection Tests
- **Objective:** Implement and verify instantaneous rollback capabilities under simulated failure conditions.
- **Tasks:**
  1. Implement `UniverseStore.rollback_to_build(universe_id, target_build_id)` using append-only pointer re-affirmation.
  2. Test fault injection: simulate corrupted market data or unauthorized symbol injection; assert rollback restores prior state in $< 1.0\text{ second}$.
  3. Author adversarial tests `tests/test_universe_publication_rollback_adversarial.py`.
- **Estimated Effort:** 1.0 Working Days.

### Phase 6: Formal Production Publication Gate & Operator Sign-off
- **Objective:** Execute live dry-run in production environment (read-only), review audit receipts with Product Owner, and obtain Gate 12 clearance.
- **Tasks:**
  1. Perform non-destructive production dry-run on Railway container.
  2. Generate `C_PRODUCTION_DRY_RUN_RECEIPT.json` and verify bit-for-bit parity with local builds.
  3. Present formal authorization dossier to authorized operator.
- **Estimated Effort:** 1.5 Working Days.

---

## 3. Upstream Dependencies on Package B

Track C execution remains strictly blocked until Track B resolves its known entitlement and ingestion deficiencies:

| Dependency Milestone | Owner | Current Status | Acceptance Criteria for Track C |
| :--- | :---: | :---: | :--- |
| **B-DEP-01: API Credentials** | Track B | DEFICIENT | Active Polygon key & Alpaca SIP subscription verified |
| **B-DEP-02: Historical SIP Ingestion**| Track B | BLOCKED | 50 consecutive trading sessions ingested in production `market.db` |
| **B-DEP-03: Quality Certification** | Track B | PROPOSED | Signed `TrackBDataQualityCertification` bundle generated |
| **B-DEP-04: Split Reconciliation** | Track B | PENDING | Zero unhandled corporate actions across target population |

---

## 4. Rollback Runbook: Standard Operating Procedure (SOP)

If a post-publication regression, unseasoned asset leakage, or downstream consumer crash occurs following a universe publication:

### Step 1: Triage & Identification
1. Execute diagnostic readback:
   ```bash
   python3 scripts/universe/inspect_active_universe.py --universe-id ARX_US_EQUITIES
   ```
2. Identify the active `universe_build_id` and the defective symbol(s).

### Step 2: Rollback Execution (Fast-Pointer Reversion)
1. Execute atomic rollback script targeting the last known good build ID:
   ```bash
   python3 scripts/universe/rollback_universe.py \
     --universe-id ARX_US_EQUITIES \
     --target-build-id ubuild-1792400000 \
     --reason "REVERSION_UNSEASONED_LEAKAGE" \
     --authorized-by "OPERATOR_DAVID_AKARA"
   ```
2. The script commits a new pointer record referencing the prior build's Merkle root in $< 1.0\text{s}$.
3. Historical rows remain 100% intact; zero database deletion is performed.

### Step 3: Cache Invalidation & Consumer Verification
1. Restart/warm up background scanner workers to drop in-memory cached builds:
   ```bash
   railway service restart --service web -e production
   ```
2. Trigger Cloudflare CDN cache invalidation for `/api/v1/screener/*` routes.
3. Query `GET /api/v1/screener/vcp/snapshot` and verify `universe_size` matches the rolled-back baseline.

---

## 5. Independent Production Authorization Requirements

Before any universe publication command may be executed against `/root/analyst_dashboard/data/` on the production Railway container, the following five criteria must be satisfied and recorded:

1. **Formal Gate 12 Clearance:** Track G12 must issue an affirmative release decision certifying that downstream execution systems are protected against expanded asset order routing.
2. **Package B Certification Record:** Verified SHA-256 certificate digest from Track B confirming zero data gaps across the target cohort.
3. **Scheduled Maintenance Window:** Approved execution window with designated Approver, Executor, Incident Response Owner, and Rollback Owner.
4. **Pre-Apply Target State Digest:** Verified SQLite checksum of the target database prior to transaction execution.
5. **Operator Command Attestation:** Single-maintainer operational risk explicitly reaffirmed by user.
