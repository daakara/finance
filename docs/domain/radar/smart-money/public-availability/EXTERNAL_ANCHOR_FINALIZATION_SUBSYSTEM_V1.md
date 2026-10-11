# ARX Terminal — Smart Money: External Anchor Finalization Subsystem (Epoch 001)

**Gate Identifier:** `EXTERNAL_ANCHOR_FINALIZATION_SUBSYSTEM_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `3714b3119162d54b0260e656089695069457c726`  
**Functional Commit:** `46b5a3cef11e2cb84f9acf1b0f0e7f073e3d4672`  
**Authority:** Independent Anchor Finalization Architect & Governance State-Machine Auditor  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `APPEND_ONLY` + `POLICY_VERSIONED` + `IDEMPOTENT` + `CRASH_CONSISTENT` + `AUTHORITY_ELIGIBILITY_SEPARATE_FROM_EFFECTIVENESS` + `NO_RETROACTIVE_AUTHORIZATION` + `NO_QUORUM_DOWNGRADE` + `NO_PRODUCTION`

---

## 1. Executive Summary & Epistemic Scope

This gate implements, audits, and verifies the control plane governing post-capture evidence anchoring and authority effectiveness transitions.

1. **Ratified Policy Epoch (`SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1`):**  
   - External anchor is mandatory for authority effectiveness (`EXTERNAL_ANCHOR_REQUIRED = YES`, `AUTHORITY_EFFECTIVENESS_REQUIRES_ANCHOR = YES`).
   - Authority eligibility is strictly separated from authority effectiveness (`AUTHORITY_ELIGIBILITY_CAN_EXIST_WITHOUT_ANCHOR = YES`).
   - Quorum downgrades during provider outages and human overrides are strictly prohibited (`HUMAN_OVERRIDE_CAN_GRANT_AUTHORITY = NO`, `PROVIDER_OUTAGE_CAN_LOWER_QUORUM = NO`).
   - Exact semantic bytes and SHA-256 for all 8 governed policy artifacts are retained in `POLICY_ARTIFACT_REGISTRY`.
2. **Deterministic Dual-Anchor Quorum & Independence Proof:**  
   - Configured providers: `AWS_S3_OBJECT_LOCK` (Provider A) and `GCP_CLOUD_STORAGE_LOCK` (Provider B).
   - Quorum policy: `ALL_REQUIRED` (2 of 2).
   - Independence verification confirms separate credential domains, separate key domains, separate admin domains, and separate write authorities (`DUAL_ANCHOR_INDEPENDENCE = PASS`).
3. **Run Identity & Ledger Sealing:**  
   - `RUN_ID_REUSE_ALLOWED = NO`. Run transition ledger must be sealed prior to payload construction (`RUN_LEDGER_SEALED = YES`, `POST_SEAL_RUN_LEDGER_MUTATION = PROHIBITED`).
   - Idempotency key derived deterministically: $\text{SHA256}(\text{run\_id} + ":" + \text{run\_final\_hash} + ":" + \text{policy\_set\_hash})$.
   - Invariant enforced: `SAME_IDEMPOTENCY_KEY_DIFFERENT_PAYLOAD = CRITICAL_INTEGRITY_FAILURE`.
4. **Finalization State Machine ($A_0 \dots A_{11}$):**  
   - Implements discrete states from `A0_ANCHOR_NOT_REQUIRED` to `A11_TERMINAL_HOLD`.
   - Distinct timestamps: $T_1$ (observed), $T_2$ (eligible), $T_3$ (effective). Enforces $T_1 \le T_2 \le T_3$; zero backdating permitted.
5. **Separate Append-Only Finalization & Attempt Ledgers:**  
   - Implemented in SQLite with ACID serializable transactions and cryptographic hash chaining (`PREVIOUS_FINALIZATION_EVENT_HASH -> FINALIZATION_EVENT_HASH`).
   - Sequence authority: `SERIALIZABLE_COORDINATOR`.
6. **Ambiguous Commit Result Policy (`QUERY_EXISTING_ANCHOR_BEFORE_RETRY`):**  
   - Any indeterminate submission triggers an anchor query using the idempotency key and run ID before retry. Matching anchor confirms; conflicting anchor forces terminal hold.
7. **Downstream-Use Firewall:**  
   - Hard fail-closed consumer gate blocks all production decisions, empirical denominators, model tuning, learning claims, and backfills while `AUTHORITY_EFFECTIVE_STATUS != EFFECTIVE`.
   - Zero unauthorized downstream acceptances (`UNAUTHORIZED_DOWNSTREAM_ACCEPTANCES = 0`).
8. **Late Anchor Recovery & No Retroactive Authorization:**  
   - Expired pending anchors ($>300\text{s}$) transition to `A7_ANCHOR_PENDING_EXPIRED`.
   - Late recovery transitions $A_7 \to A_9 \to A_{10}$. Authority effectiveness is timestamped at recovery time, never retroactively applied to past decisions.
9. **Independent Reviewer Reconstruction:**  
   - 12-point forensic verification achieves `PASS` across all parity, attempt continuity, receipt validity, quorum, and hash-chain checks with `UNEXPLAINED_FINALIZATION_EVENTS = 0`.
10. **Candidate 004 Firewall:**  
    - `PROTECTED_FILES_CHANGED = 0`, `SHADOW_EVIDENCE_DB_WRITES = 0`, `CANDIDATE_004_INTERFERENCE = NONE`.

---

## 2. Finalization State Machine Architecture ($A_0 \dots A_{11}$)

| State ID | State Name | Description | Preceding States |
|---|---|---|---|
| $A_0$ | `ANCHOR_NOT_REQUIRED` | Initial state when run does not require anchoring | None |
| $A_1$ | `ANCHOR_REQUIRED` | Capture complete, eligibility established, anchor required | $A_0$ |
| $A_2$ | `ANCHOR_PAYLOAD_SEALED` | Canonical payload constructed and hashed | $A_1$ |
| $A_3$ | `ANCHOR_SUBMISSION_PENDING` | Active HTTP request transmission to provider | $A_2, A_4, A_5, A_6$ |
| $A_4$ | `ANCHOR_RETRY_WAIT` | Retryable transport error encountered, waiting for backoff | $A_3, A_5$ |
| $A_5$ | `ANCHOR_COMMIT_INDETERMINATE` | Request timed out or transport indeterminate, awaiting lookup | $A_3$ |
| $A_6$ | `ANCHOR_CONFIRMED` | Cryptographic provider receipt verified | $A_3, A_5$ |
| $A_7$ | `ANCHOR_PENDING_EXPIRED` | Max pending duration ($>300\text{s}$) exceeded | $A_2, A_3, A_4, A_5$ |
| $A_8$ | `ANCHOR_CONFLICT` | Conflicting anchor payload, hash mismatch, or receipt invalid | $A_3, A_5$ |
| $A_9$ | `GOVERNANCE_RECOVERY_REVIEW` | Manual or recovery gate review for late anchor | $A_7$ |
| $A_{10}$ | `AUTHORITY_EFFECTIVE` | Quorum satisfied (from $A_6$) or recovery authorized (from $A_9$) | $A_6, A_9$ |
| $A_{11}$ | `TERMINAL_HOLD` | Irreconcilable conflict or recovery rejection | $A_8, A_9$ |

---

## 3. Quorum & Dual-Anchor Independence Specification

```ini
ANCHOR_PROVIDER_COUNT = 2
ANCHOR_REQUIRED_COUNT = 2
ANCHOR_QUORUM_POLICY = ALL_REQUIRED

[PROVIDER_A]
ANCHOR_A_PROVIDER = AWS_S3_OBJECT_LOCK
ANCHOR_A_CREDENTIAL_DOMAIN = AWS_IAM_ROLE_ARX_ANCHOR_A
ANCHOR_A_KEY_DOMAIN = AWS_KMS_KEY_ARX_ANCHOR_A
ANCHOR_A_ADMIN_DOMAIN = AWS_ACCOUNT_SECURITY_ORG_A
ANCHOR_A_WRITE_AUTHORITY = PUT_OBJECT_LEGAL_HOLD_WORM

[PROVIDER_B]
ANCHOR_B_PROVIDER = GCP_CLOUD_STORAGE_LOCK
ANCHOR_B_CREDENTIAL_DOMAIN = GCP_SERVICE_ACCOUNT_ARX_ANCHOR_B
ANCHOR_B_KEY_DOMAIN = GCP_KMS_KEY_ARX_ANCHOR_B
ANCHOR_B_ADMIN_DOMAIN = GCP_PROJECT_SECURITY_ORG_B
ANCHOR_B_WRITE_AUTHORITY = INSERT_OBJECT_RETENTION_LOCK

DUAL_ANCHOR_INDEPENDENCE = PASS
```

---

## 4. Retained Policy Artifacts (Section 24)

| Policy ID | Semantic Digest (SHA-256) | Retained Status |
|---|---|---|
| `anchor_policy` | `f5926ecdb5536443c5b5502c4bce81452a382e7e43d4107b3ecf6d92ec3a4176` | `YES` |
| `state_machine_policy` | `5c18195aeb214fb0ff1a99a7732a1cf05c65f9a6572eb0499ea8d4bc481ca03d` | `YES` |
| `clock_policy` | `56f5c88bf1e35fbb2414e0f0689b94098933b93409a80e1cbb5b00c6d7adce36` | `YES` |
| `observer_policy` | `37cb992b4507ea1259e894df62a74c76b1f23ee796247d2f97cba2a7c4ebf9be` | `YES` |
| `authority_admission_policy` | `b99d6fb35efb988f5416b7ebbb186cf921d7b38cbef1a44e5ce6c321038380e2` | `YES` |
| `transport_classification_policy` | `2467d1fc0d170f2f3eef4d4f26019a716c5476a086e0fc70dc3a1055728a4a49` | `YES` |
| `identity_policy` | `c432742918e9dbe267156ceeb03f393f9ef21bf8ae032338ff9eb7baea0634ee` | `YES` |
| `decision_input_contract` | `83906a5e171dae2985863d030cfdbe8d2035efaaeb00b1784ca8e18dbcf25608` | `YES` |

---

## 5. Verification Results & Test Suite Coverage

### 5.1 Unit & Integration Test Suite (`tests/test_smart_money_anchor_finalization.py`)
- **Total Test Cases:** 25
- **Passed:** 25 | **Failed:** 0 | **Skipped:** 0
- **Duration:** 2.69s

### 5.2 Full Repository Smart Money Regression Suite
- **Executed Files (10 test files):**
  - `tests/test_sec_public_availability_observer.py`: 13 passed
  - `tests/test_sec_source_authority_qualification.py`: 20 passed
  - `tests/test_smart_money_certification_hardening.py`: 25 passed
  - `tests/test_smart_money_incremental_wave2c.py`: 12 passed
  - `tests/test_smart_money_form4_pipeline.py`: 8 passed
  - `tests/test_smart_money_remediation_r1.py`: 9 passed
  - `tests/test_smart_money_closure_reconciliation.py`: 11 passed
  - `tests/test_smart_money_backfill.py`: 12 passed
  - `tests/test_smart_money_universe_authority.py`: 7 passed
  - `tests/test_smart_money_anchor_finalization.py`: 25 passed
- **Aggregate Total:** **142 passed test cases**, 0 failures, 0 errors, 0 regressions.

---

## 6. Outcome Verdict & Next Action

- **Verdict:** `EXTERNAL_ANCHOR_FINALIZATION_SUBSYSTEM_STATUS = VERIFIED`
- **Production Status:** `PRODUCTION_READINESS = NOT_AUTHORIZED`
- **Next Action:** `USE_THIS_SUBSYSTEM_DURING_NEXT_VALID_NATURAL_EDGAR_CAPTURE`
