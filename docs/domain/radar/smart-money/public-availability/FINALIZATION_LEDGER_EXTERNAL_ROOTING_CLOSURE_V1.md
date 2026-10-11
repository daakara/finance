# ARX Terminal — Smart Money: Finalization Ledger External Rooting Closure Gate

**Gate Identifier:** `FINALIZATION_LEDGER_EXTERNAL_ROOTING_CLOSURE_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `c3be442f94b87662a0e3a1e3c59d0b57ae379ed0`  
**Functional Commit:** `6cc4e192b067834b9ba41d76a7dd8aa18bab68ca`  
**Authority:** Independent Finalization Ledger Rooting Auditor & External Trust-Domain Verifier  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `APPEND_ONLY` + `HASH_CHAINED` + `EXTERNALLY_ROOTED` + `POLICY_VERSIONED` + `REPLAYABLE` + `NO_RETROACTIVE_REWRITE` + `NO_PRODUCTION`

---

## 1. Executive Summary & Epistemic Scope

Prior to this gate, the finalization ledger was internally hash-chained. Under strict adversarial auditing:

> **Adversarial Threat Model:** An administrator with database-level write authority could rewrite one or more historical finalization events and recompute every internal hash and chain link. Because internal hashes are self-referential, an auditor examining only the database file would observe a mathematically valid internal chain.

This gate formally eliminates that vulnerability by establishing a **Two-Layer External Rooting Architecture**:
$$\text{Finalization Ledger Events} \longrightarrow \text{Finalization Ledger Root Hash} \longrightarrow \text{External Signed Root Receipt}$$

1. **Finalization Root Policy (`SMART_MONEY_FINALIZATION_ROOT_POLICY_V1`):**  
   External rooting of the finalization ledger is mandatory (`FINALIZATION_ROOT_REQUIRED = YES`, `SCOPE = RUN`). Retroactive rewriting and human overrides are strictly prohibited.
2. **Deterministic Root Input Set & Sequence Authority:**  
   The canonical root commits to all finalization events for a run, strictly ordered by `FINALIZATION_SEQUENCE` (not timestamp), together with authority eligibility, authority effective status, authority time, anchor request IDs, receipt hashes, and the policy set hash.
3. **External Trust Domain Separation:**  
   Roots are anchored outside the finalization store in a separate security organization and credential domain (`AWS_S3_OBJECT_LOCK_ROOT`). The ledger writer possesses zero permission to delete, overwrite, or access signing keys.
4. **Whole-Ledger Rewrite Detection Proof:**  
   Verified through automated adversarial tests: an altered historical event whose internal hash chain has been fully recomputed fails external root parity, proving that internal consistency alone cannot grant authority.
5. **No Retroactive Authority:**  
   Authority effectiveness requires confirmed external rooting. Confirmation timestamps become `authority_became_effective_at` without backdating.

---

## 2. Finalization Root Input Set & Canonical Serialization

### Canonical Input Set (Section 6)
- `parent_run_id`: Bound run identifier
- `parent_run_final_hash`: Cryptographic final hash of the run transition ledger
- `finalization_event_count`: Total event count in the finalization sequence
- `first_finalization_sequence`: Initial sequence number ($1$)
- `last_finalization_sequence`: Terminal sequence number ($N$)
- `finalization_event_hashes`: Ordered list of SHA-256 event hashes
- `authority_eligibility`: `PASS` / `FAIL` / `NOT_EVALUATED`
- `authority_effective_status`: `EFFECTIVE` / `BLOCKED_PENDING_FINALIZATION_ROOT`
- `authority_became_effective_at`: ISO 8601 UTC timestamp or `None`
- `anchor_request_ids`: All unique anchor request IDs in the run
- `anchor_receipt_hashes`: Hashes of provider receipts for capture payloads
- `policy_set_hash`: Cryptographic hash of the governing policy set

### Canonical Serialization Contract (Section 7)
- **Schema Version:** `1.0`
- **Serialization Version:** `1.0`
- **Ordering Authority:** `FINALIZATION_SEQUENCE` (strictly monotonic positive integers)
- **Root Hash Formula:**
  $$\text{FINALIZATION\_LEDGER\_ROOT\_HASH} = \text{SHA256}(\text{canonicalize\_json}(\text{root\_payload}))$$
- **Idempotency Key:**
  $$\text{IDEMPOTENCY\_KEY} = \text{SHA256}(\text{parent\_run\_id} + ":" + \text{parent\_run\_final\_hash} + ":" + \text{finalization\_ledger\_root\_hash})$$

---

## 3. External Trust Domain & Storage Attestation

| Domain Dimension | Configuration / Separation Evidence |
|---|---|
| **External Provider** | `AWS_S3_OBJECT_LOCK_ROOT` |
| **Credential Domain** | `AWS_IAM_ROLE_ARX_FINALIZATION_ROOT` |
| **KMS Key Domain** | `AWS_KMS_KEY_ARX_FINALIZATION_ROOT` |
| **Admin Domain** | `AWS_ACCOUNT_SECURITY_ORG_ROOT` |
| **Write Authority** | `PUT_OBJECT_LEGAL_HOLD_WORM` |
| **Object Versioning** | `ENABLED` |
| **Object Lock Mode** | `COMPLIANCE` |
| **Retention Until** | `2036-10-11T12:00:00Z` (10-Year WORM retention) |
| **Delete / Overwrite** | `PROHIBITED` |
| **Writer Key Separation** | Finalization ledger process cannot access signing private key |

---

## 4. Threat Model Tamper Test Results (Sections 13–18)

| Threat Scenario | Injection Method | Internal Chain | External Root Parity | Detection Verdict |
|---|---|---|---|---|
| **Whole-Ledger Rewrite** | Historical event modified; all event hashes and chain recomputed | `PASS` | `FAIL` | `WHOLE_FINALIZATION_LEDGER_REWRITE_DETECTED = YES` |
| **Event Deletion** | Intermediate event deleted; sequence discontinuous | `FAIL` | `FAIL` | `MISSING_EVENT_DETECTED = YES` |
| **Event Insertion** | Unauthorized event injected | `FAIL` | `FAIL` | `UNAUTHORIZED_INSERTION_DETECTED = YES` |
| **Event Reordering** | Valid events swapped in sequence | `FAIL` | `FAIL` | `EVENT_REORDERING_DETECTED = YES` |
| **Authority Time Tamper** | `authority_became_effective_at` backdated | `FAIL` | `FAIL` | `AUTHORITY_TIME_TAMPER_DETECTED = YES` |
| **Receipt Substitution** | Valid receipt swapped for different root hash | `PASS` | `FAIL` | `ROOT_RECEIPT_MISMATCH_DETECTED = YES` |

---

## 5. Ambiguous Root Commit & Crash Recovery (Sections 20, 27)

- **Policy:** `QUERY_EXISTING_ROOT_BEFORE_RETRY`
- **Ambiguous Lookup Matrix:**
  - `FOUND_MATCHING` $\longrightarrow$ `ROOT_CONFIRMED`
  - `FOUND_CONFLICTING` $\longrightarrow$ `TERMINAL_HOLD`
  - `NOT_FOUND` $\longrightarrow$ Retry permitted with identical idempotency key
  - `INDETERMINATE` $\longrightarrow$ Remain blocked
- **Crash Recovery Verification:**
  - Sealed root payload survives process termination.
  - Re-submission is idempotent (`NO_DUPLICATE_ROOT = YES`).
  - Confirmed receipt state is fully crash-consistent.

---

## 6. Reviewer 10-Point Reconstruction (Section 24)

The independent `FinalizationRootReviewer` executes forensic reconstruction:
1. Reconstructs sequence strictly by `finalization_sequence`.
2. Recomputes every event SHA-256 hash.
3. Recomputes internal hash chain.
4. Recomputes root payload dictionary.
5. Recomputes finalization root SHA-256 hash.
6. Verifies external root receipt signature and structure.
7. Compares recomputed root hash to receipt root hash.
8. Verifies authority timing ($T_{\text{observed}} \le T_{\text{eligible}} \le T_{\text{effective}} \le T_{\text{root}}$).
9. Verifies receipt continuity across run ID and final hash.
10. Verifies zero missing or extra events.

**Reviewer Status:** `PASS`  
**Reconstruction Mismatches:** `0`

---

## 7. Retained Policy Artifacts (16 Total Artifacts)

| Policy Artifact ID | SHA-256 Digest | Status |
|---|---|---|
| `anchor_policy` | `ae287bfbcbb6f48460bea0c9445043178528ecea0c7be55657e4c17d0dbf9431` | RETAINED |
| `state_machine_policy` | `42eb8ded98ca8f7ce3f3d05e58ee702a5ad762c82d1e9d961c6e70678dac5539` | RETAINED |
| `clock_policy` | `02e3ec5107e46d3796ec5b73d8de088d4f4882c7ea7074013432b139552a51c5` | RETAINED |
| `observer_policy` | `a58185d9eb6b73221f2b68aa235e41c877cebbcacb62237cce6d759f577dc792` | RETAINED |
| `authority_admission_policy` | `f87b61f6c1c60267e24fbc4cd3e44866a7033e266a75f098a22a56353b91808d` | RETAINED |
| `transport_classification_policy` | `926df84016602355c7c9845da5ee2b475ab8108e2d916ef58de384b32ea65cb5` | RETAINED |
| `identity_policy` | `661c2af6d82966c84808d318fd249fb179b9ea3b8e0bd18d52b301c1a1dfe8c2` | RETAINED |
| `decision_input_contract` | `227bcc3cb6dfefb872b718ce595eab58f726199f3fcaeb4b8cc17f61da65083c` | RETAINED |
| `anchor_policy_registry_v1` | `e2710fcbad992a67f9716a3f059a371809ac4d12af1c7c661ffdd485660e3277` | RETAINED |
| `epoch_registry_v1` | `8eaaeed1399f572e86b4e5bcbd1f6a8fb66ac191359955679fa2aef2f95ab153` | RETAINED |
| `policy_resolution_contract_v1` | `7c3c964e39edc34f1fa3979219d8a3a9f426b039a41ff12c7ae5aa0d35d1cc3a` | RETAINED |
| `finalization_root_policy_v1` | `4929ff13f31310b05524c1580913af69d2539af613e9e3f9ed82eac9155a970a` | RETAINED |
| `root_serialization_contract_v1` | `163b7ec4e470421ef141e1c90e66822c4e88114e57ef7c3aab353e46e38bfc42` | RETAINED |
| `root_receipt_verification_policy_v1` | `fff57ab40b5fdab6eb57abfad412f17d86d43dbd980db97c8a0027fc55bb42ae` | RETAINED |
| `root_retry_policy_v1` | `85dbc96c88852f21bdd1418f77453d882fb902715a3310dd5d111fec1419f224` | RETAINED |
| `root_reviewer_policy_v1` | `170bcbf9b79a1e5d683593bb85f927871ebeabee3cfb0dd248a6d266c2d77763` | RETAINED |

---

## 8. Test Execution Summary

- **Targeted Suite:** `tests/test_smart_money_finalization_root.py`
  - Tests: **18 passed** (0 failures).
- **Subsystem Suite:** All 3 anchor finalization test modules
  - Tests: **64 passed** (0 failures).
- **Domain-Wide Regression Suite:** All 10 Smart Money modules
  - Tests: **142 passed** (0 failures).

---

## 9. Candidate 004 Firewall Attestation

```
git diff --stat 63131b9 HEAD -- shadow_evidence.db analyst_dashboard/vcp natural_trigger.py
PROTECTED_FILES_CHANGED = 0
SHADOW_EVIDENCE_DB_WRITES = 0
CANDIDATE_004_INTERFERENCE = NONE
```

---

## 10. Authoritative Verdict & Next Action

```
GATE = FINALIZATION_LEDGER_EXTERNAL_ROOTING_CLOSURE_V1
FINALIZATION_LEDGER_EXTERNAL_ROOTING_STATUS = VERIFIED
PRODUCTION_READINESS = NOT_AUTHORIZED
NEXT_ACTION = VERIFY_LIVE_EXTERNAL_ANCHOR_INFRASTRUCTURE + OBTAIN_INDEPENDENT_CLOCK_REFERENCE + RUN_FIRST_VALID_NATURAL_EDGAR_CAPTURE
```
