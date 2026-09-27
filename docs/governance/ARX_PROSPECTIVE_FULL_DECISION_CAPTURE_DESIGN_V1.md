# ARX TERMINAL — PROSPECTIVE FULL DECISION CAPTURE DESIGN V1.0.2
## FINAL IMPLEMENTATION-READINESS CORRECTION
**Status:** DESIGN ONLY — FROZEN
**Contract Version:** `1.0.2`
**Governing Authority:** `ARX_GOVERNANCE_GATE`
**Supersedes:** `1.0.1` (commit `59af490`)
**Created At UTC:** `2026-09-27T21:50:00Z`
**Target Schema:** [`docs/governance/ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json`](file:///c:/Users/akara/Documents/Projects/finance/docs/governance/ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json) (v1.0.2)

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

## 2. EVALUATION-CYCLE IDENTITY ACROSS RETRIES

An evaluation cycle represents **one logical production decision cycle/request**. Transient infrastructure retries must not generate synthetic phantom cycles:

```ini
RETRY_CREATES_NEW_EVALUATION_CYCLE = NO
RETRY_REUSES_ORIGINAL_CYCLE_ID = YES
```

- When a scheduled scan or user request begins, an authoritative `evaluation_cycle_id` is assigned:
  ```text
  CYC_{cycleType}_{cycleStartedAtUtc}_{hash8}
  Example: CYC_SCHEDULED_UNIVERSE_SCAN_20260928T133000Z_4f8a12bc
  ```
- If network timeouts or worker restarts force the pipeline to retry individual assets or the batch 30 seconds later, the retry MUST reuse the original `evaluation_cycle_id`, `cycleStartedAtUtc`, `cycleType`, and universe boundary authority.
- Genuinely new scheduled daily scans, radar passes, or independent user requests receive new, distinct cycle IDs.

---

## 3. DECISION ID IDEMPOTENCY & ATTEMPT IDENTITY

### 3.1 Deterministic Decision Identity
To ensure idempotency across worker crashes and retries, `decision_id` MUST NOT contain mutable execution timestamps or random UUIDs. It is constructed strictly from canonical immutable parameters:

```text
decision_id = DEC_{sha256(instrumentId + evaluationCycleId + engineSha + decisionSchemaVersion)[:16]}
Example: DEC_7f8a12b4c9e0d1a3
```

```ini
SAME_LOGICAL_EVALUATION_RETRY = SAME_DECISION_ID
SAME_ASSET_DIFFERENT_CYCLE = DIFFERENT_DECISION_ID
DECISION_ID_DETERMINISTIC = YES
```

Dynamic timestamps are persisted as evidence fields, not identity inputs:
- `evaluationStartedAtUtc`: Timestamp of initial evaluation start.
- `evaluationCompletedAtUtc`: Timestamp of pipeline completion (or null if crashed).
- `captureWrittenAtUtc`: Timestamp when record was persisted to storage.

### 3.2 Evaluation Attempt Identity
When execution retries occur, each retry attempt is recorded under a distinct forensic attempt ID while preserving the single logical `decision_id`:

```text
evaluation_attempt_id = ATT_{decisionId}_{attemptNumber}_{hash8}
Example: ATT_DEC_7f8a12b4c9e0d1a3_2_9a8b7c6d
```

$$\text{1 Logical Evaluation} \rightarrow \text{1 Decision ID} \rightarrow N \text{ Execution Attempts}$$
Retries update forensic diagnostic tables but **never increment the empirical decision denominator**.

---

## 4. EXPECTED-EVALUATION RECONCILIATION

The reconciliation of expected evaluations is strictly mutually exclusive. Every expected evaluation settles into exactly one terminal state:

- `COMPLETED_EVALUATION`: Asset fully processed by model pipeline; decision state assigned.
- `FAILED_BEFORE_MODEL_EVALUATION`: Infrastructure fault before model evaluation (e.g. quote fetch HTTP 500, symbol resolution error).
- `FAILED_DURING_MODEL_EVALUATION`: Pipeline crashed mid-analysis (e.g. feature engine exception).
- `MISSING_EXPECTED_EVALUATION`: Expected asset never picked up by any worker before cycle timeout.

### The Mutually Exclusive Reconciliation Equation
$$\text{EXPECTED\_N} = \text{COMPLETED\_N} + \text{FAILED\_BEFORE\_N} + \text{FAILED\_DURING\_N} + \text{MISSING\_N}$$

$$\sum \text{Terminal States} \equiv \text{EXPECTED\_N}$$

---

## 5. OBSERVED ATTEMPTS VS. COMPLETED MODEL DECISIONS

An observed production evaluation attempt is not inherently an eligible model decision:

```ini
OBSERVED_ATTEMPT_IS_COMPLETED_MODEL_DECISION = NOT_NECESSARILY
```

- `evaluation_observed = YES / NO`: Indicates whether telemetry intercepted a live runtime attempt.
- `evaluation_completion_state`: Identifies whether the attempt completed successfully or failed.
- Only records with `evaluationCompletionState == COMPLETED_EVALUATION` and `empiricalCertificationState == CERTIFIED_NATURAL_PRODUCTION` are admitted to the **Model Decision-Quality Denominator**.

---

## 6. PRODUCTION CYCLE COHORTS

All decision records retain their `cycleType` to prevent invalid cross-cohort pooling:
1. `SCHEDULED_UNIVERSE_SCAN`: Comprehensive end-of-day or pre-market scan of the full mandated universe.
2. `RADAR_CYCLE`: Intraday high-velocity screening pass.
3. `DISCOVERY_CYCLE`: Thematic gem/archetype discovery run.
4. `EXPLICIT_USER_REQUEST`: Ad-hoc, on-demand ticker analysis submitted by a human user.

---

## 7. EXPLICIT USER REQUEST SELECTION BIAS

Ad-hoc user requests are natural production events, but represent a **user-selected sample** subject to strong survivorship and popularity bias (e.g., users frequently query high-flying meme stocks).

ARX strictly separates operational origin from sampling mode:
```ini
EVIDENCE_ORIGIN = NATURAL_PRODUCTION
POPULATION_SAMPLING_MODE = SYSTEMATIC / USER_SELECTED
```
- For `EXPLICIT_USER_REQUEST`: `POPULATION_SAMPLING_MODE = USER_SELECTED`.
- For `SCHEDULED_UNIVERSE_SCAN`: `POPULATION_SAMPLING_MODE = SYSTEMATIC`.

---

## 8. MODEL-QUALITY VS. OPPORTUNITY-CAPTURE ELIGIBILITY

To ensure user-selected queries do not corrupt universe-level statistics, every `DecisionEvent` stores four distinct analytical eligibility flags:

| Analytical Dimension | `SCHEDULED_UNIVERSE_SCAN` | `EXPLICIT_USER_REQUEST` | Governance Rationale |
|---|---|---|---|
| `eligibleForDecisionQuality` | **YES** (if completed) | **YES** (if completed) | The model's setup validity and execution levels can be evaluated on any complete input. |
| `eligibleForCoverageAnalysis` | **YES** | **NO** | User requests do not have a defined universe denominator. |
| `eligibleForRankingAnalysis` | **YES** | **NO** | Ad-hoc single tickers cannot be cross-sectionally ranked against an arbitrary peer universe. |
| `eligibleForOpportunityCapture` | **YES** | **NO** | User-selected assets cannot be used to evaluate market-wide false negatives or opportunity capture. |

```ini
EXPLICIT_USER_REQUEST_DECISION_QUALITY = ELIGIBLE_IF_COMPLETE
EXPLICIT_USER_REQUEST_COVERAGE_ANALYSIS = NOT_ELIGIBLE
EXPLICIT_USER_REQUEST_RANKING_ANALYSIS = NOT_ELIGIBLE_UNLESS_PART_OF_FROZEN_PEER_UNIVERSE
EXPLICIT_USER_REQUEST_OPPORTUNITY_CAPTURE = NOT_ELIGIBLE
```

---

## 9. SYSTEMATIC POPULATION COHORT AUTHORITY

The primary prospective empirical opportunity-capture population is frozen strictly as:

```ini
PRIMARY_OPPORTUNITY_CAPTURE_CYCLE_TYPES = {
  SCHEDULED_UNIVERSE_SCAN
}
```

Only `SCHEDULED_UNIVERSE_SCAN` satisfies all five mandatory scientific conditions:
1. Complete, known in-scope universe denominator.
2. Frozen pre-run expected-evaluation authority.
3. Unbiased systematic evaluation trigger.
4. Natural production runtime execution.
5. Deterministic cycle reconciliation.

---

## 10. EPISODE COUNTING ACROSS CYCLES

When an asset surfaces as a setup in consecutive daily cycles:
- Each cycle generates a distinct `decision_id` (e.g. Monday evaluation, Tuesday evaluation).
- If setup criteria remain continuously valid without entry, both decisions link to the same ongoing `episode_id`.

```ini
DAILY_REEVALUATION_EQUALS_NEW_OPPORTUNITY = NO
```

### Denominator Separation Invariant
All empirical reports must report the three distinct metrics separately:
$$\text{Evaluation Count } (N_{\text{eval}}) \ge \text{Decision Count } (N_{\text{dec}}) \ge \text{Episode Count } (N_{\text{ep}})$$
Substituting Decision Count for Opportunity/Episode Count is strictly prohibited.

---

## 11. RETRY ADVERSARIAL ACCEPTANCE TESTS

### Test A: Same Cycle, Same Asset, Retry 30s Later
- Scenario: Worker times out on NVDA during `CYC_SCHEDULED_UNIVERSE_SCAN_..._1a2b`. Retry worker picks up NVDA 30 seconds later under the same cycle.
- Persisted State:
  - `decision_id`: `DEC_7f8a...` (Identical).
  - Attempt 1: `ATT_DEC_7f8a..._1` (`attemptStatus = TIMEOUT`).
  - Attempt 2: `ATT_DEC_7f8a..._2` (`attemptStatus = SUCCESS`).
  - `attemptCount`: 2.
- Invariant: `EMPIRICAL_DECISION_COUNT = 1`.

### Test B: Same Asset, Next Scheduled Cycle
- Scenario: NVDA evaluated on Monday at close (`CYC_..._MONDAY`) and Tuesday at close (`CYC_..._TUESDAY`).
- Persisted State:
  - Monday: `decision_id = DEC_MONDAY_...`, `episode_id = EP_NVDA_01`.
  - Tuesday: `decision_id = DEC_TUESDAY_...`, `episode_id = EP_NVDA_01`.
- Invariant: 2 distinct decisions; 1 continuous opportunity episode.

### Test C: Process Crash Between Expectation and Decision
- Scenario: Dispatcher records expected evaluation for AMD. Worker crashes immediately before starting evaluation. Cycle reaches timeout.
- Persisted State:
  - `ExpectedEvaluationRecord`: Present for AMD.
  - `DecisionEvent`: 0 records.
  - Reconciliation: Reconciled as `terminalReconciliationState = MISSING_EXPECTED_EVALUATION`.
- Invariant: Exactly 1 expected evaluation; 0 phantom decisions; 1 coverage defect.

### Test D: Explicit User Request Followed by Scheduled Scan
- Scenario: User queries TSLA at 14:00. Scheduled universe scan runs at 16:00 including TSLA.
- Persisted State:
  - Request 1: `cycleType = EXPLICIT_USER_REQUEST`, `populationSamplingMode = USER_SELECTED`, `eligibleForOpportunityCapture = false`.
  - Request 2: `cycleType = SCHEDULED_UNIVERSE_SCAN`, `populationSamplingMode = SYSTEMATIC`, `eligibleForOpportunityCapture = true`.
- Invariant: Separate decision IDs, separate cycles, segregated population eligibility.

---

## 12. DATABASE SCHEMA DDL SPECIFICATION (V1.0.2)

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1. Evaluation Cycles (Root Entity for Denominator Accounting)
CREATE TABLE IF NOT EXISTS prospective_evaluation_cycles (
    evaluation_cycle_id TEXT PRIMARY KEY,
    cycle_type TEXT NOT NULL,
    cycle_started_at_utc TEXT NOT NULL,
    cycle_completed_at_utc TEXT,
    universe_version TEXT NOT NULL,
    universe_snapshot_id TEXT,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    expected_evaluations_count INTEGER NOT NULL DEFAULT 0,
    completed_evaluations_count INTEGER NOT NULL DEFAULT 0,
    failed_before_model_evaluations_count INTEGER NOT NULL DEFAULT 0,
    failed_during_model_evaluations_count INTEGER NOT NULL DEFAULT 0,
    missing_evaluations_count INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

-- 2. Expected Evaluation Manifest (Pre-Run Roster)
CREATE TABLE IF NOT EXISTS prospective_expected_evaluations (
    expected_evaluation_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    expectation_established_at_utc TEXT NOT NULL,
    terminal_reconciliation_state TEXT NOT NULL,
    decision_id TEXT,
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 3. Decision Events (Immutable Canonical Entity)
CREATE TABLE IF NOT EXISTS prospective_decision_events (
    decision_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    current_attempt_id TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    cycle_type TEXT NOT NULL,
    population_sampling_mode TEXT NOT NULL,
    evidence_origin TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    evaluation_completion_state TEXT NOT NULL,
    empirical_certification_state TEXT NOT NULL,
    eligible_for_decision_quality INTEGER NOT NULL,
    eligible_for_coverage_analysis INTEGER NOT NULL,
    eligible_for_ranking_analysis INTEGER NOT NULL,
    eligible_for_opportunity_capture INTEGER NOT NULL,
    evaluation_started_at_utc TEXT NOT NULL,
    evaluation_completed_at_utc TEXT,
    capture_written_at_utc TEXT NOT NULL,
    market_session TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    decision_engine_sha TEXT NOT NULL,
    decision_schema_version TEXT NOT NULL,
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

-- 4. Evaluation Attempts (Forensic Execution Audit Log)
CREATE TABLE IF NOT EXISTS prospective_evaluation_attempts (
    evaluation_attempt_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    evaluation_cycle_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    attempt_started_at_utc TEXT NOT NULL,
    attempt_completed_at_utc TEXT,
    attempt_status TEXT NOT NULL,
    failure_details TEXT,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id),
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 5. Immutability Enforcers
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

## 13. GATE DECLARATION & ARTIFACT LINEAGE

This gate establishes **Design V1.0.2** as the final implementation-ready specification:

```ini
GATE = PASS
DESIGN_VERSION = 1.0.2
DESIGN_STATUS = IMPLEMENTATION_READY
RETRY_REUSES_CYCLE_ID = YES
SAME_LOGICAL_RETRY_SAME_DECISION_ID = YES
ATTEMPT_ID_CONTRACT = FROZEN
EXPECTED_EVALUATION_RECONCILIATION = MUTUALLY_EXCLUSIVE
CYCLE_TYPE_COHORTS = FROZEN
USER_SELECTED_SAMPLE_SEPARATION = FROZEN
PRIMARY_OPPORTUNITY_CAPTURE_CYCLE_TYPES = {SCHEDULED_UNIVERSE_SCAN}
EPISODE_VS_DECISION_DENOMINATORS = FROZEN
PRODUCTION_CODE_CHANGED = NO
CAPTURE_LATENCY_MEASURED = NOT_ASSESSED
PROSPECTIVE_DENOMINATOR = 0
OBSERVATION_ACTIVE = NO
TUNING_AUTHORIZED = NO
MODEL_LEARNING_CLAIM = NOT_AUTHORIZED
NEXT_ACTION = PROSPECTIVE_FULL_DECISION_CAPTURE_IMPLEMENTATION_GATE
```
