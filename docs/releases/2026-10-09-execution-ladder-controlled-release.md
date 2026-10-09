# ARX Terminal — Production Release Notes
## Execution Ladder Controlled Remediation Release

### Release Identity
```ini
RELEASE_DATE =
  2026-10-09
CERTIFIED_CANDIDATE_SHA =
  74baf306cfe2b2b53da8269990e6f7363c2fe42d
ORIGIN_MAIN_ANCHOR_SHA =
  5a90b918b0975151b74e936b3fbfa536b575edd7
RELEASE_CLASSIFICATION =
  GOVERNANCE_DATA_INTEGRITY_REMEDIATION
SCOPE =
  EXECUTION_LADDER_PROSPECTIVE_CAPTURE_NORMALIZATION
STATUS =
  PREPARED_LOCAL_HOLD
PUSH_STATUS =
  NOT_AUTHORIZED
DEPLOY_STATUS =
  NOT_AUTHORIZED
```

---

### 1. Purpose and Scope
Remediate the epoch-millisecond timestamp slicing defect in `compute_execution_ladder_plan_id()`, enforce complete 11-field canonical identity equivalence in `find_equivalent_execution_ladder_plan()`, and guarantee concurrency-safe atomic SQLite admission in `insert_execution_ladder_plan_atomic()`.
- Scope is strictly non-quantitative governance and data-capture remediation.
- No trading algorithms, scoring rules, or entry/stop/target calculations are modified.

---

### 2. Timestamp Normalization Correction
Replaced fragile 10-character string slicing (`source_ts[:10]`) with fail-closed deterministic parser `extract_canonical_trading_date(ts_val)`.
- Rejects non-finite numbers, booleans, empty values, and timezone-naive timestamps.
- Correctly parses ISO-8601 strings, timezone-aware datetimes, and numeric epoch milliseconds/seconds into canonical UTC calendar dates (`YYYY-MM-DD`).

---

### 3. Complete 11-Field Identity Isolation
`find_equivalent_execution_ladder_plan()` now enforces exact equivalence across the full 11-field canonical identity tuple:
1. `epoch_id`
2. `symbol`
3. `user_role`
4. `canonical_utc_trading_date`
5. `release_sha`
6. `execution_ladder_authority_sha`
7. `planned_entry`
8. `structural_invalidation`
9. `take_profit_1`
10. `take_profit_2`
11. `execution_status`

Cross-release merging is strictly prohibited: differing `release_sha` values always create distinct canonical plans.

---

### 4. SQLite Safeguards & Concurrency Protection
- `insert_execution_ladder_plan_atomic()` executes under `BEGIN IMMEDIATE` transaction control to prevent TOCTOU race conditions.
- Protected by bounded exponential retry backoff (`@retry_sqlite(max_retries=5, base_delay=0.05)`).
- Historical plan rows are protected by immutable triggers (`trg_prevent_update_execution_ladder_plans`, `trg_prevent_delete_execution_ladder_plans`).

---

### 5. Denominator Adjudication
- **Historical Cohort (Release `01683a3`)**: 4 raw stored rows deduplicate substantively to **3 unique canonical plans** (`HISTORICAL_DENOMINATOR = 3`).
- **Cumulative Cohort (Across Releases `01683a3` & `5a90b91`)**: 5 persisted rows across releases evaluate to **4 unique canonical plans** (`CUMULATIVE_DENOMINATOR = 4`).

---

### 6. Certification Results
- Total Tests Executed: **374**
- Passed: **372**
- Documented Pre-Existing Failures: **2** (EXC-001, EXC-002)
- Errors / Regressions: **0**

---

### 7. Documented Release Exceptions
- **EXC-001**: Pre-existing frozen engine manifest hash mismatch (`CORRUPTED != VERIFIED`). Fails because `verify_frozen_engine_manifest()` evaluates hashes pinned prior to Sprint 2A/2B. Zero candidate diff.
- **EXC-002**: Pre-existing `is_actionable` KeyError in unit test fixture. Unit test calls raw static engine method which normalizes price bounds but does not emit `is_actionable`. Core safety invariant passes unconditionally. Zero candidate diff.
- Risk Acceptance: `PENDING_PRODUCT_OWNER_APPROVAL`.

---

### 8. Residual Risks and Ownership
- Residual risk is low and contained to pre-existing non-safety-critical test fixture discrepancies.
- Ownership: Governance Auditor & Production Release Manager.

---

### 9. Rollback and Recovery Strategy
If unexpected behavior occurs post-deployment:
1. Re-deploy previous production release SHA `d97801e783620294454d1989164c907534ed4358` (Deployment ID: `0685134c-134c-4cce-8e0a-850299e18c34`).
2. SQLite persistent volume mount retains all historical and admitted rows cleanly without schema mutation.
3. No historical rows require deletion or alteration during rollback.

---

### 10. Passive Production Verification Requirements
Post-deployment verification requires:
1. Verifying GET `/api/execution-ladder` responds with valid status codes.
2. Observing prospective capture stream logs for proper UTC date extraction.
3. Confirming zero duplicate plan ID generation for repeated same-day requests.

---

### 11. VCP Epoch 002 Isolation
Legacy VCP Epoch 002 signals remain paused and strictly segregated from Execution Ladder Epoch 001 prospective capture. Zero cross-stream pollution.

---

### 12. Evidence Distinction
- **Historical Evidence**: Immutable production records preserved in production SQLite ledger.
- **Local Certification**: 372 passing tests executed under strict isolation.
- **Future Deployment Verification**: To be performed post-deployment upon explicit Product Owner authorization.

---

### 13. Production Verification Disclaimer
This document represents pre-release preparation only. It makes NO claim of production deployment or production verification prior to actual deployment.
