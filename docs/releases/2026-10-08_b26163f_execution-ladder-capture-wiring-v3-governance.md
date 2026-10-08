# ARX Terminal — Production Release Notes

## Execution Ladder Capture Wiring Remediation & Epoch 4 V3 Observation Governance

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
FUNCTIONAL_RELEASE_SHA =
  b26163f275b54052a6e8757c46748fcb8119f69c
PREVIOUS_RUNTIME_SHA =
  d23bbf34655aee48ba8b9d0ccf6f9e60fd39d354
PREVIOUS_FUNCTIONAL_RELEASE_SHA =
  defdf8900ab64d14705b57ecdd0c8ed902526195
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway deployment
DEPLOYMENT_ID =
  2e44b5e4-781b-47ed-b585-70c615e81ce8
DEPLOYMENT_STATUS =
  SUCCESS
PRODUCTION_VERIFICATION =
  PASS
```

---

### Release Purpose & Classification

```ini
RELEASE_PURPOSE =
  restore natural Execution Ladder capture entrypoint wiring

GOVERNANCE_PURPOSE =
  activate Epoch 4 observation manifest V3 lineage

RELEASE_CLASSIFICATION =
  OBSERVATION_CAPTURE_WIRING_REMEDIATION
  GOVERNANCE_MANIFEST_SUCCESSION
  NONCONTAMINATING_OBSERVABILITY_RESTORED

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  NO (Capture observation pathway restored; zero business/API schema changes)

EXECUTION_LADDER_OBSERVATION_PATH_CHANGE =
  YES / INTENTIONAL

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

MODEL_PARAMETER_CHANGE =
  NONE

MODEL_TUNING =
  NO

EXECUTION_LADDER_AUTHORITY_CHANGE =
  NONE

DATABASE_SCHEMA_CHANGE =
  NONE

EMPIRICAL_QUALITY =
  INSUFFICIENT_EVIDENCE
```

---

### Root Cause Analysis & Technical Remediation

#### 1. Entrypoint Wiring Defects
During prospective epoch 1 observation audits, natural production requests for `/api/v1/analytics/{symbol}` generated valid execution ladder structures but failed to invoke capture hooks due to two entrypoint-level defects in `api/routes/analytics.py`:
* **Defect 1 (Gating on Actionability)**: Prospective capture for the Execution Ladder was nested inside `if is_actionable:`, causing legitimate non-actionable ladder plans (`WAIT`, `MONITOR`, `AVOID`) to be bypassed rather than recorded as prospective observations.
* **Defect 2 (Epoch ID Omission & Route Fallthrough)**: The invocation of `record_natural_recommendation()` failed to provide `epoch_id=EXECUTION_LADDER_EPOCH_ID` (`EXECUTION_LADDER_PROSPECTIVE_EPOCH_001`), causing capture routing to fall through to legacy Epoch 4 handling instead of `record_execution_ladder_plan()`.

#### 2. Remediation Implementation
* **Unnested Execution Ladder Capture**: Execution Ladder capture is now unnested and evaluated for all recommendations where `execution_ladder` is present, regardless of actionability.
* **Explicit Epoch Routing**: Capture calls explicitly pass `epoch_id="EXECUTION_LADDER_PROSPECTIVE_EPOCH_001"` directly to `record_execution_ladder_plan()`, while preserving legacy Epoch 4 capture for actionable signals.
* **Deterministic Dual-Contract Parity**: Added unit and contract tests in `tests/test_execution_ladder_passive_capture.py` covering both actionable and non-actionable Execution Ladder generation paths.

---

### Observation Manifest Lineage & Succession

* **Historical V2 Manifest (`EPOCH_4_MANIFEST.json`)**:
  * Status: `VERIFIED`
  * Version: `v2.0.0`
  * Manifest Hash: `2e550089a1f4ff56ff84322ab3aba22428d5a7079cdae5a20c456e66ea248e66`
  * Boundary: Immutable historical record frozen at `5efc217950a6009270344122bfc07fca854ff788`.
* **Successor V3 Manifest (`EPOCH_4_MANIFEST_V3.json`)**:
  * Status: `VERIFIED`
  * Version: `v3.0.0`
  * Manifest Hash: `7fc5ece99d67510807593d1c7f4d1efe7a7295d41934385a3b958765b5535dd8`
  * Boundary: Active production runtime boundary reflecting legitimate evolution across 10 governed observation runtime files.
  * Activation: First deployed runtime containing V3 and passing manifest verification (`b26163f275b54052a6e8757c46748fcb8119f69c`).
* **Legacy Epoch 4 Preservation**:
  * `EPOCH_4_POLICY_STATUS`: `ACTIVE / UNSUPERSEDED`
  * `EPOCH_4_RUNTIME_OBSERVATION_AUTHORIZATION`: `PRE_ACTIVATION / CURRENTLY_FALSE`
  * `EPOCH_4_OBSERVATION_SEMANTICS_CHANGED`: `NO`
  * `LEGACY_ROUTE_PRESENT`: `YES`

---

### Prospective Evidence & Denominator Snapshots

Strict non-contaminating release protocol was observed during this gate:
* `RELEASE_VERIFICATION_ANALYTICS_REQUESTS = 0`: Zero analytics or recommendation endpoints were called as smoke tests.
* `PRE_DEPLOY_PRODUCTION_PROSPECTIVE_DENOMINATOR = 0` (Observed at `2026-10-08T20:00:26.892104Z`)
* `POST_DEPLOY_RAW_RECORD_COUNT = 0` (Observed at `2026-10-08T20:04:35.125327Z`)
* `FIRST_NATURAL_PRODUCTION_CAPTURE = NOT_OBSERVED`
* `POTENTIAL_NEW_PRODUCTION_CAPTURE = NO`

#### Passive Production Log Observation
* `NATURAL_ANALYTICS_REQUEST_OBSERVED = NO`
* `CAPTURE_HOOK_REACHED = NO`
* `CAPTURE_RESULT = N/A`
* `NATURAL_CAPTURE_PATH_RUNTIME_EXECUTION = NOT_OBSERVED`

> [!NOTE]
> Deployed code identity and manifest boundaries are fully verified in the production container. Natural capture path execution will only occur upon subsequent natural user or automated operational requests.

---

### Verification Summary

```ini
CRITICAL_LOCAL_TESTS =
  120 / 120 PASS (9.11s)

PRODUCTION_V3_STATUS =
  VERIFIED (10/10 files match SHA-256)

PRODUCTION_V2_HISTORICAL_STATUS =
  VERIFIED

V3_ACTIVATED =
  YES

V3_ACTIVATION_RUNTIME_SHA =
  b26163f275b54052a6e8757c46748fcb8119f69c

BACKEND_HEALTH =
  PASS (HTTP 200, status: online)

FRONTEND_HEALTH =
  PASS (HTTP 200)

DATABASE_REACHABLE_READ_ONLY =
  YES

RELEASE_VERIFICATION_ANALYTICS_REQUESTS =
  0
```

---

### Limitations & Next Authorized Event

* **Replay & Backfill Exclusion**: This release restores forward entrypoint wiring for naturally occurring traffic. Replay exclusion and backfill exclusion must be verified independently when evaluating any newly persisted prospective rows.
* **Empirical Quality**: The prospective denominator remains at 0. Evaluation of execution ladder performance awaits certified natural observations.
* **Next Authorized Event**: Natural production observation monitoring and first natural production capture certification upon arrival of genuine traffic.
