# ARX TERMINAL — PROSPECTIVE FULL DECISION CAPTURE DESIGN V1.0.1
## NARROW SEMANTIC CORRECTION & EVIDENCE ARCHITECTURE
**Status:** DESIGN ONLY — FROZEN
**Contract Version:** `1.0.1`
**Governing Authority:** `ARX_GOVERNANCE_GATE`
**Supersedes:** `1.0.0` (commit `b9b0245`)
**Created At UTC:** `2026-09-27T21:20:00Z`
**Target Schema:** [`docs/governance/ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json`](file:///c:/Users/akara/Documents/Projects/finance/docs/governance/ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json) (v1.0.1)

---

## 1. GOVERNANCE BOUNDARIES

```ini
MODEL_TUNING = PROHIBITED
FEATURE_CHANGES = PROHIBITED
THRESHOLD_CHANGES = PROHIBITED
FILTER_CHANGES = PROHIBITED
PRODUCTION_DECISION_LOGIC_CHANGES = PROHIBITED
HISTORICAL_OUTCOME_REINTERPRETATION = PROHIBITED
LEARNING_CLAIM = NOT_AUTHORIZED
PROSPECTIVE_CAPTURE_DESIGN = AUTHORIZED
OBSERVATION_ACTIVE = NO
PROSPECTIVE_DENOMINATOR = 0
```

---

## 2. SPLIT EMPIRICAL DENOMINATORS

A critical flaw in naive telemetry design is conflating universe coverage with model predictive quality. ARX v1.0.1 establishes four distinct, mathematically non-interchangeable denominator families:

```ini
COVERAGE_DENOMINATOR_IS_MODEL_DENOMINATOR = NO
```

### A. Scope / Coverage Denominator
Evaluates whether production successfully processed all mandated assets in the investment universe:
- `ALL_IN_SCOPE_OPPORTUNITIES`: Total assets defined by the active universe mandate.
- `EXPECTED_PRODUCTION_EVALUATIONS`: Assets scheduled for evaluation in a cycle.
- `OBSERVED_PRODUCTION_EVALUATIONS`: Assets that completed pipeline execution.
- `MISSING_EXPECTED_EVALUATIONS`: Expected assets dropped due to pipeline crashes/timeouts.
- `INTENTIONALLY_OUT_OF_SCOPE`: Assets filtered out by mandate definition (e.g. penny stocks).
- `SCOPE_UNKNOWN`: Assets with unresolvable or corrupted metadata.

$$\text{Coverage Rate} = \frac{N(\text{OBSERVED\_PRODUCTION\_EVALUATIONS})}{N(\text{EXPECTED\_PRODUCTION\_EVALUATIONS})}$$

### B. Model Decision-Quality Denominator
Restricted strictly to assets that were authentically evaluated by the live model:
```ini
MODEL_DECISION_DENOMINATOR =
  IN_SCOPE
  + OBSERVED_PRODUCTION_EVALUATION
  + NATURAL_PRODUCTION
```
Missing or partially crashed evaluations are **strictly prohibited** from entering model predictive scoring.

### C. Recommendation Denominator
Separately partitions the evaluated cohort into decision states:
- `ACTIONABLE_RECOMMENDATIONS`: Assets reaching `confluenceScore >= 75.0` with actionable trade plans.
- `NON_RECOMMENDATIONS`: Assets evaluated and routed to `WATCH`, `WAIT`, or `REJECT`.

### D. Infrastructure Defect Denominator
Monitors operational health independent of predictive accuracy:
```ini
INFRASTRUCTURE_DEFECT_DENOMINATOR =
  EXPECTED_EVALUATION
  + INFRASTRUCTURE_FAILURE
```

---

## 3. EXPECTED-EVALUATION AUTHORITY

An un-evaluated asset cannot be classified as a coverage failure without an authoritative, point-in-time expectation record. ARX establishes the **Expected-Evaluation Contract**:
- Before an evaluation pipeline executes, the dispatching subsystem registers an `EvaluationCycle` and emits deterministic `ExpectedEvaluationRecord` entries for every candidate asset.
- **Governed Expectation Triggers:**
  1. `SCHEDULED_UNIVERSE_SCAN`: Daily automated batch scan of the eligible universe.
  2. `RADAR_CYCLE`: Periodic intraday ribbon screening cycle.
  3. `DISCOVERY_CYCLE`: Thematic gem/archetype discovery run.
  4. `EXPLICIT_USER_REQUEST`: On-demand ticker analysis submitted by a human user.
- If a worker crashes before evaluating ticker $X$, ticker $X$ retains its `ExpectedEvaluationRecord` with `actualEvaluationStatus = NO_OBSERVED_PRODUCTION_EVALUATION`, enabling exact reconciliation:
$$\text{Expected} = \text{Actual} + \text{Missing} + \text{Failed}$$

---

## 4. EVALUATION CYCLE IDENTITY

Every evaluation belongs to an immutable `evaluation_cycle_id`:
```text
CYC_{cycleType}_{timestampUtc}_{hash8}
Example: CYC_SCHEDULED_UNIVERSE_SCAN_20260928T133000Z_4f8a12bc
```
The evaluation cycle serves as the unifying relational key linking:
- Universe snapshot
- Expected evaluation roster
- Actual decision events
- Infrastructure failure events
- Aggregate cycle reconciliation statistics

For single-symbol user requests (`EXPLICIT_USER_REQUEST`), an ephemeral cycle is generated, and `universeSnapshotId` is legitimately `null`.

---

## 5. DETERMINISTIC DECISION ID ARCHITECTURE

To guarantee idempotency and eliminate randomness claims, `decision_id` is constructed deterministically:
```text
decision_id = DEC_{symbol}_{sha256(instrumentId + evaluationCycleId + evaluationTimestampUtc + engineSha)[:12]}
Example: DEC_AAPL_c4ca4238a0b9
```
```ini
DECISION_ID = DETERMINISTIC_CRYPTOGRAPHIC_HASH
DECISION_ID_DETERMINISTIC = YES
```
Re-running an identical evaluation on identical inputs reproduces the identical `decision_id`, preventing duplicate phantom entries on retry.

---

## 6. DISCRETE RULE EVALUATION STATES

Rule outcomes are not simple booleans. Downstream rules that never executed because an upstream rule short-circuited the pipeline must NEVER be recorded as PASS or FAIL. ARX freezes six explicit states:
- `PASS`: Rule executed; input satisfied threshold.
- `FAIL`: Rule executed; input violated threshold.
- `NOT_EVALUATED_AFTER_BINDING_FAILURE`: Upstream binding failure aborted evaluation; rule was never evaluated.
- `NOT_APPLICABLE`: Rule does not apply to this asset class or regime (e.g. debt-to-equity on commercial banks).
- `UNAVAILABLE`: Input data missing; rule could not execute.
- `ERROR`: Rule execution threw an unhandled exception.

---

## 7. FIRST-BINDING RULE & NO SHADOW EXECUTION

To preserve observational invariance:
```ini
TELEMETRY_CAUSES_SHADOW_RULE_EXECUTION = NO
```
The capture system MUST NOT execute downstream model rules solely to collect telemetry if the production engine naturally short-circuited.
- **First-Binding Rule:** The earliest chronologically executed rule whose failure determines the non-actionable state is recorded as `FIRST_BINDING_RULE` with `isBinding = true`.
- **Downstream Tracking:** Subsequent unexecuted rules are faithfully recorded as `NOT_EVALUATED_AFTER_BINDING_FAILURE`.
- **All Executed Failed Rules:** The record stores `allExecutedFailedRuleIds` representing rules that *genuinely executed and failed*, eliminating false claims of global rule knowledge.

---

## 8. INFRASTRUCTURE PRECEDENCE & EXECUTION ORDER

Infrastructure events are not categorized by synthetic priority over model rules. Instead, ARX separates:
- `ruleCategory` (`MANDATORY_PRODUCT_CONSTRAINT`, `AUDITED_MODEL_RULE`, `INFRASTRUCTURE_CONSTRAINT`)
- `actualExecutionOrder` (1, 2, 3...)
Infrastructure checks (e.g. quote staleness) may execute at ingress, mid-pipeline (e.g. fundamentals fetch), or egress. The runtime records the exact temporal sequence.

---

## 9. PARTIAL EVALUATION LIFECYCLE

If an evaluation begins but fails mid-pipeline, ARX assigns an explicit `EvaluationStatus`:
- `COMPLETED`: Full evaluation finished normally.
- `PARTIAL_INFRASTRUCTURE_FAILURE`: Pipeline halted due to external provider drop mid-analysis.
- `FAILED_BEFORE_MODEL_EVALUATION`: Ingress failure (e.g. symbol resolution).
- `FAILED_DURING_MODEL_EVALUATION`: Unhandled engine exception during feature computation.

A partially evaluated asset is quarantined:
```ini
PARTIAL_PIPELINE_FAILURE = NOT_A_MODEL_REJECTION
```

---

## 10. DECISION VS. INFRASTRUCTURE BOUNDARIES

```ini
INFRASTRUCTURE_FAILURE = NOT_A_MODEL_DECISION
PARTIAL_PIPELINE_FAILURE = NOT_A_MODEL_REJECTION
MISSING_REQUIRED_DATA = MODEL_REJECTION_ONLY_IF_EXISTING_FROZEN_MODEL_LOGIC_EXPLICITLY_DEFINES_IT_AS_SUCH
```
If frozen model logic explicitly dictates that missing 3-year revenue history disqualifies a growth stock, that is an audited model rejection. If an API times out fetching price, that is an infrastructure failure.

---

## 11. PERFORMANCE TARGETS & MEASUREMENT PROTOCOL

No empirical latency benchmarks were executed during this design phase. Previous estimates are formally reclassified:
```ini
CAPTURE_LATENCY_REQUIREMENT = < 5.0 ms (p95), < 10.0 ms (p99)
CAPTURE_LATENCY_MEASURED = NOT_ASSESSED
```
Actual latency benchmarks ($p50, p95, p99, \max$) must be measured and certified during the implementation gate.

---

## 12. DUAL-MODE WRITE FAILURE SEMANTICS

```ini
CAPTURE_WRITE_FAILURE_EFFECT_ON_USER_DECISION = FAIL_OPEN
CAPTURE_WRITE_FAILURE_EFFECT_ON_EMPIRICAL_RECORD = FAIL_CLOSED
```
1. **User Decision Availability (Fail-Open):** If telemetry storage encounters a lock or disk error, the analytical response is returned to the user with a `TELEMETRY_DEGRADED` header.
2. **Empirical Certification Validity (Fail-Closed):** The unwritten evaluation is recorded in the operational log as `UNCERTIFIED_CAPTURE_FAILURE`. It is **strictly excluded** from the empirical research denominator.

---

## 13. CANONICAL STORE VS. RECOVERY STREAM

To prevent competing sources of truth:
```ini
CANONICAL_DECISION_STORE = SQLITE_GOVERNANCE_DB
RECOVERY_STREAM = APPEND_ONLY_JSONL
STORE_DISAGREEMENT_POLICY = SQLITE_CANONICAL_WITH_JSONL_RECOVERY_REPLAY
```
- **Primary Source of Authority:** SQLite (`/root/analyst_dashboard/data/governance.db`) with WAL mode and immutability triggers.
- **Recovery & Replication:** Append-only JSONL write-ahead log. If the SQLite file is lost, the canonical DB is reconstructed deterministically by replaying JSONL records.

---

## 14. IMMUTABILITY & EXPLICIT CORRECTION EVENTS

Original decision events in SQLite strictly prohibit `UPDATE` and `DELETE` via database triggers. If an erroneous record must be corrected:
- The original record remains 100% intact.
- An append-only `DecisionCorrectionEvent` is inserted into `prospective_decision_corrections`:
  - `correctionEventId`: `COR_{originalDecisionId}_{timestamp}_{uuid6}`
  - `originalDecisionId`: Foreign key to original decision
  - `correctedFieldName`: Target field
  - `originalValue` / `correctedValue`
  - `correctionReason`: Audited justification
  - `authorizedBy`: Governance authority
  - `timestampUtc`: Timestamp of amendment

---

## 15. GENERALIZED UNIVERSE LINKAGE

The schema supports both universe-driven batch runs and single-symbol ad-hoc evaluations via `evaluationCycleId`:
- Scheduled Universe Scans link a non-null `universeSnapshotId`.
- Ad-hoc user queries set `universeSnapshotId = null` with `cycleType = EXPLICIT_USER_REQUEST`, maintaining relational integrity without artificial dummy snapshots.

---

## 16. ADVERSARIAL TEST MATRIX (10 SCENARIOS)

| # | Adversarial Scenario | Expected Persisted State | Denominator Membership | Model-Quality Eligibility | Coverage Status |
|---|---|---|---|---|---|
| **S-01** | Asset expected but never evaluated (worker crash) | `ExpectedEvaluationRecord` with status `NO_OBSERVED_PRODUCTION_EVALUATION` | Scope / Coverage Denominator | **Ineligible** (Excluded) | **Coverage Defect** |
| **S-02** | Asset evaluation fails before first model rule (HTTP 500) | `DecisionEvent` with status `FAILED_BEFORE_MODEL_EVALUATION`, `infrastructureFailure` populated | Infrastructure Defect Denominator | **Ineligible** (Excluded) | **Infrastructure Failure** |
| **S-03** | Asset fails Rule 2; downstream rules 3–10 do not run | Rule 2 `FAIL` (`isBinding=true`); Rules 3–10 `NOT_EVALUATED_AFTER_BINDING_FAILURE` | Model Decision Denominator | **Eligible** (Non-Recommendation) | **Evaluated** |
| **S-04** | Duplicate retry of same evaluation cycle | Idempotent duplicate rejected by unique `decisionId` index; 0 duplicate rows | Original Denominator Count Preserved | **Eligible** (Once) | **Evaluated** |
| **S-05** | Telemetry database locked / unavailable | Fail-open to user terminal; log warning emitted; record marked uncertified | Excluded from Empirical Denominator | **Ineligible** (Uncertified) | **Capture Defect** |
| **S-06** | SQLite write succeeds; JSONL replication fails | SQLite commit holds; background worker retries JSONL append; alert emitted | Model Decision Denominator | **Eligible** | **Evaluated** |
| **S-07** | JSONL write succeeds; SQLite transaction fails | SQLite transaction rolled back; JSONL entry marked uncommitted; alert emitted | Excluded until replayed | **Ineligible** | **Capture Defect** |
| **S-08** | Single-symbol query with no universe snapshot | `universeSnapshotId = null`, `cycleType = EXPLICIT_USER_REQUEST` | Ad-Hoc Request Pool | **Eligible** (Informational) | **N/A (Ad-Hoc)** |
| **S-09** | Same asset evaluated in two distinct cycles on same day | Distinct `decisionId` and `evaluationCycleId`; linked to same ongoing `episodeId` | Single Episode in Opportunity Denominator | **Eligible** | **Evaluated** |
| **S-10** | Replay runner attempts to write to production store | Blocked by `ContextVar` firewall (`RuntimeError: TestPollutionViolation`) | Quarantined in Replay Store | **Ineligible** (Replay) | **N/A (Replay)** |

---

## 17. REVISED DATABASE DDL SPECIFICATION

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1. Evaluation Cycles (Root Entity for Denominator Accounting)
CREATE TABLE IF NOT EXISTS prospective_evaluation_cycles (
    evaluation_cycle_id TEXT PRIMARY KEY,
    cycle_type TEXT NOT NULL,
    cycle_start_timestamp_utc TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    universe_snapshot_id TEXT,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    expected_evaluations_count INTEGER NOT NULL DEFAULT 0,
    actual_evaluations_count INTEGER NOT NULL DEFAULT 0,
    missing_evaluations_count INTEGER NOT NULL DEFAULT 0,
    infrastructure_failures_count INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

-- 2. Expected Evaluation Manifest
CREATE TABLE IF NOT EXISTS prospective_expected_evaluations (
    expected_evaluation_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    expectation_established_at_utc TEXT NOT NULL,
    actual_evaluation_status TEXT NOT NULL,
    decision_id TEXT,
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 3. Decision Events
CREATE TABLE IF NOT EXISTS prospective_decision_events (
    decision_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    evaluation_timestamp_utc TEXT NOT NULL,
    market_session TEXT NOT NULL,
    evidence_origin TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    evaluation_status TEXT NOT NULL,
    empirical_certification_state TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    decision_engine_sha TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    universe_snapshot_id TEXT,
    data_provider TEXT NOT NULL,
    provider_source_timestamp_utc TEXT NOT NULL,
    ingestion_timestamp_utc TEXT NOT NULL,
    freshness_status TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    decision_state TEXT NOT NULL,
    actionability_state TEXT NOT NULL,
    confluence_score REAL,
    cross_sectional_rank INTEGER,
    cross_sectional_population_size INTEGER,
    first_binding_rule_id TEXT,
    first_binding_rule_category TEXT,
    rejection_reason_json TEXT NOT NULL,
    trade_plan_json TEXT,
    infrastructure_failure_json TEXT,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 4. Rule Evaluations (Explicit States)
CREATE TABLE IF NOT EXISTS prospective_rule_evaluations (
    rule_eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    rule_category TEXT NOT NULL,
    actual_execution_order INTEGER NOT NULL,
    evaluation_state TEXT NOT NULL,
    input_values_json TEXT NOT NULL,
    threshold_value TEXT,
    is_binding INTEGER NOT NULL,
    failure_message TEXT,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id)
);

-- 5. Decision Corrections (Append-Only)
CREATE TABLE IF NOT EXISTS prospective_decision_corrections (
    correction_event_id TEXT PRIMARY KEY,
    original_decision_id TEXT NOT NULL,
    corrected_field_name TEXT NOT NULL,
    original_value_json TEXT,
    corrected_value_json TEXT,
    correction_reason TEXT NOT NULL,
    authorized_by TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    FOREIGN KEY (original_decision_id) REFERENCES prospective_decision_events(decision_id)
);

-- Immutability Triggers
CREATE TRIGGER IF NOT EXISTS trg_prevent_update_prospective_decision_events
BEFORE UPDATE ON prospective_decision_events
BEGIN
    SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: Updates to prospective_decision_events are strictly prohibited.');
END;

CREATE TRIGGER IF NOT EXISTS trg_prevent_delete_prospective_decision_events
BEFORE DELETE ON prospective_decision_events
BEGIN
    SELECT RAISE(ABORT, 'IMMUTABILITY_VIOLATION: Deletions from prospective_decision_events are strictly prohibited.');
END;
```

---

## 18. GATE SUMMARY & DECLARATION

```ini
GATE = PASS
DESIGN_VERSION = 1.0.1
DESIGN_STATUS = COMPLETE
PRODUCTION_CODE_CHANGED = NO
COVERAGE_DENOMINATOR = FROZEN
MODEL_DECISION_DENOMINATOR = FROZEN
EXPECTED_EVALUATION_CONTRACT = FROZEN
EVALUATION_CYCLE_CONTRACT = FROZEN
DECISION_ID_SEMANTICS = DETERMINISTIC
RULE_EVALUATION_STATES = FROZEN
FIRST_BINDING_RULE = FROZEN
SHADOW_RULE_EXECUTION = PROHIBITED
PARTIAL_EVALUATION_CONTRACT = FROZEN
INFRASTRUCTURE_MODEL_SEPARATION = FROZEN
CANONICAL_DECISION_STORE = SQLITE_GOVERNANCE_DB
CAPTURE_LATENCY_MEASURED = NOT_ASSESSED
PROSPECTIVE_DENOMINATOR = 0
OBSERVATION_ACTIVE = NO
TUNING_AUTHORIZED = NO
MODEL_LEARNING_CLAIM = NOT_AUTHORIZED
NEXT_ACTION = PROSPECTIVE_FULL_DECISION_CAPTURE_IMPLEMENTATION_GATE
```
