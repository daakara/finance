# ARX Terminal — Production Release Notes

## Execution Ladder Prospective Passive Capture

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  5dcfeb41d75bb3ae02f25cb3599ab86c0cb03950
PREVIOUS_PRODUCTION_SHA =
  3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway deployment
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION =
  VERIFIED
```

---

### Release Summary

This release introduces prospective passive capture for the Execution Ladder. The system can now persist qualifying Execution Ladder plans prospectively for future evidence settlement while preserving the existing quantitative recommendation engine intact.

```ini
QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE
```

This release consists entirely of governance, observability, and data-capture infrastructure, not a model upgrade or algorithmic modification.

---

### What Changed

#### Added
* **Prospective Execution Ladder Plan Capture**: Integrated passive capture hook to record qualifying Execution Ladder recommendation plans during active production runs.
* **Deterministic Plan Identity**: Content-addressed SHA-256 plan identifier (`PLAN_<hex24>`) derived from canonical plan levels, symbol, user role, trading date, release SHA, and execution authority SHA, excluding volatile request timestamps.
* **Persistent Prospective Storage**: Implemented `execution_ladder_prospective_plans` table in `governance.db` (SQLite WAL mode on Railway persistent volume) enforcing strict table schema and triggers preventing UPDATE and DELETE mutations.
* **Two-Tier Provenance Fields**: Explicit capture of runtime deployment identity (`release_sha`) and immutable quantitative engine specification (`execution_ladder_authority_sha`).
* **Same-Day State-Evolution Preservation**: Intra-day plan changes resulting from genuine status progression or level adjustments receive distinct deterministic plan identities.
* **Retry & Request Deduplication**: Repeated identical client queries on the same trading day map to the identical plan ID, preventing redundant writes.
* **Passive Failure Isolation**: Capture failures, database contention, or metadata serialization errors are trapped and swallowed, ensuring live recommendations are returned without latency or user-facing interruption.

#### Governance & Observability
The capture layer records canonical Execution Ladder outputs without becoming an alternative recommendation authority.
* Duplicate requests do not inflate the prospective denominator.
* Legitimate substantive plan evolution remains separately observable.
* Deployment, startup, and health check requests do not create prospective evidence.
* Capture failure does not alter or degrade recommendation results.
* No future empirical benefits or predictive qualities are described or assumed to be validated.

---

### User Impact

```ini
USER_VISIBLE_CHANGE =
  NONE_EXPECTED
```

The release represents internal governance and prospective-observation infrastructure. Existing recommendations, entries, stops, targets, rankings, and scoring across both Long-Term and Day-Trader horizons are completely unchanged.

---

### Quantitative Impact

```ini
QUANT_ENGINE_CHANGED =
  NO
RECOMMENDATION_LOGIC_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
ENTRY_LOGIC_CHANGED =
  NO
STOP_LOGIC_CHANGED =
  NO
TARGET_LOGIC_CHANGED =
  NO
MODEL_PARAMETERS_CHANGED =
  NO
MODEL_TUNING =
  NO
QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE
```

---

### Data & Evidence Impact

```ini
NEW_PROSPECTIVE_STORAGE =
  execution_ladder_prospective_plans
PROSPECTIVE_CAPTURE_CHANGED =
  YES
PRE_DEPLOY_PROSPECTIVE_DENOMINATOR =
  0
POST_DEPLOY_PROSPECTIVE_DENOMINATOR =
  0
DENOMINATOR_DELTA =
  0
HISTORICAL_BACKFILL =
  0
SYNTHETIC_EVIDENCE_ADDED =
  0
```

Deployment of the capture infrastructure does not itself constitute a prospective observation. The system remains awaiting the first qualifying natural production capture.

---

### QA & Verification

```ini
TARGETED_TEST_RESULT =
  147 PASS / 1 FAIL
RELEASE_GATE =
  PASS_WITH_BASELINE_EXCEPTION
PRE_EXISTING_BASELINE_FAILURES =
  1
CANDIDATE_INTRODUCED_FAILURES =
  0
PRODUCTION_SMOKE =
  PASS
PRODUCTION_VERIFICATION =
  VERIFIED
PRODUCTION_DENOMINATOR_DELTA =
  0
```

Verification evidence:
* Executed 10 targeted test modules covering model governance, passive capture certification, optimal execution, and write firewall boundaries. 147 passed; 1 test failed due to a known pre-existing baseline defect.
* Live production backend health checks (`GET /health`) returned `200 OK` (`{"status":"online"}`).
* Representative live NAUT analytics requests (`GET /api/v1/analytics/NAUT?user_role=LONG_TERM` and `GET /api/v1/analytics/NAUT?user_role=DAY_TRADER`) returned identical numerical levels and status semantics (`WAITING_PULLBACK` and `IN_BUY_ZONE`, respectively).
* Production database query confirmed `0` records in `execution_ladder_prospective_plans` post-deployment.

---

### Known Issues

```ini
KNOWN_ISSUE =
  Frozen engine manifest certification reports CORRUPTED.
INTRODUCED_BY_THIS_RELEASE =
  NO
RELEASE_BLOCKING_FOR_THIS_CHANGE =
  NO — independently reproduced on the previous production baseline
```

`tests/test_arx_step2_passive_capture_certification.py::test_stage2_production_deployment_identity` fails because `ExperimentLedger.verify_frozen_engine_manifest()` returns `CORRUPTED` instead of `VERIFIED` against `DECISION_ENGINE_SHA = "7ad44595826c147cc77f93cd676af520764c7442"`. This issue predated the release branch and was not introduced or affected by the passive-capture extension. It remains tracked as an independent governance defect and is not marked resolved.

---

### Rollback

```ini
ROLLBACK_TARGET_SHA =
  3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb
```

```ini
CODE_ROLLBACK
  !=
EVIDENCE_REWRITE
```

The database schema is fully backward compatible; older code safely ignores `execution_ladder_prospective_plans`. Any legitimate prospective records captured in production remain immutable historical evidence protected by database triggers even if application code is subsequently rolled back.

---

### Governance State After Release

```ini
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION_STATUS =
  VERIFIED
PROSPECTIVE_DENOMINATOR =
  0
EMPIRICAL_QUALITY =
  INSUFFICIENT_EVIDENCE
MODEL_TUNING =
  FROZEN
LEARNING_CLAIM =
  NOT_AUTHORIZED
NEXT_AUTHORIZED_EVENT =
  AWAITING_FIRST_NATURAL_PRODUCTION_CAPTURE
```

Technical deployment does not demonstrate predictive improvement or learning.

---

### Release Classification

* `GOVERNANCE`
* `OBSERVABILITY`
* `DATA_INFRASTRUCTURE`
* `QA_HARDENING`

*(Not classified as QUANT_CHANGE, MODEL_UPDATE, or EMPIRICAL_VALIDATION).*

---

### Final Release Ledger

```ini
RELEASE_SHA =
  5dcfeb41d75bb3ae02f25cb3599ab86c0cb03950
PREVIOUS_PRODUCTION_SHA =
  3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb
REMOTE_PUSH =
  COMPLETE
DEPLOYMENT =
  COMPLETE
PRODUCTION_VERIFIED =
  YES
RELEASE_NOTES_CREATED =
  YES
QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE
PROSPECTIVE_DENOMINATOR =
  0
KNOWN_BASELINE_FAILURES =
  1
NEW_PRODUCTION_DEFECTS =
  0 KNOWN FROM RELEASE VERIFICATION
EMPIRICAL_QUALITY =
  INSUFFICIENT_EVIDENCE
MODEL_TUNING =
  FROZEN
LEARNING_CLAIM =
  NOT_AUTHORIZED
NEXT_AUTHORIZED_EVENT =
  AWAITING_FIRST_NATURAL_PRODUCTION_CAPTURE
```
