# ARX Terminal — Smart Money: Anchor Policy Resolution Closure Gate

**Gate Identifier:** `ANCHOR_POLICY_RESOLUTION_CLOSURE_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/public-availability-observer-epoch-001`  
**Entry HEAD:** `91fa1745f65300022afdeb5519e0c45f6b83ac3e`  
**Functional Commit:** `cdc2ee8f2c60d6d387924fb673157c6ab7947e3e`  
**Authority:** Independent Policy Resolution Auditor & Governance Registry Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `DETERMINISTIC` + `POLICY_VERSIONED` + `REPLAYABLE` + `NO_RETROACTIVE_APPLICATION` + `NO_MID_RUN_REBIND` + `NO_PRODUCTION`

---

## 1. Executive Summary & Epistemic Scope

This gate formally resolves the open governance issue identified following the External Anchor Finalization Subsystem:

> **Core Resolution:** Replaced ambiguous field `ANCHOR_POLICY_EFFECTIVE_FROM_RUN_EPOCH` with deterministic `ANCHOR_POLICY_MIN_RUN_EPOCH = "EPOCH_001"` and `ANCHOR_POLICY_MIN_RUN_EPOCH_ORDINAL = 1`, backed by an authoritative `EpochRegistry` and `PolicyRegistry`.
> Policy applicability is governed by the strict rule `ANCHOR_POLICY_APPLICABILITY_RULE = "ALL_PREDICATES_REQUIRED"`.

### Key Epistemic Principles Implemented

1. **Epoch Ordinal Monotonicity (Section 5 & 6):**  
   Epoch authority sequence cannot be safely inferred from lexicographical strings. `EpochRegistry` provides an append-only mapping from symbolic epoch names to monotonic positive integers (`EPOCH_001 -> 1`, `EPOCH_002 -> 2`, etc.). Runs referencing unknown epochs are fail-closed rejected with `BLOCKED_UNKNOWN_EPOCH`.

2. **Run-Start Time Interval Authority (Section 8 & 9):**  
   Because point timestamps ignore physical clock uncertainty, run genesis is evaluated over an authoritative interval $[T_{\text{start}}, T_{\text{end}}]$ derived from runtime clock evidence. Boundary relations against policy `valid_from` and `valid_until` are mathematically classified as `BEFORE`, `AFTER_OR_EQUAL`, or `STRADDLES_BOUNDARY`.

3. **Fail-Closed Boundary Straddle Handling (Section 10):**  
   If a run-start interval straddles the activation boundary of a policy, applicability is indeterminate (`POLICY_APPLICABILITY_INDETERMINATE`). Authority effectiveness is strictly blocked without exception.

4. **Exactly-One Policy Selection (Section 11):**  
   Every run must resolve to exactly one policy (`APPLICABLE_POLICY_COUNT = 1`). Multiple candidate policies trigger `CONFLICTING_APPLICABLE_POLICIES` and immediate fail-closed rejection.

5. **No Mid-Run Rebinding & Replay Parity (Sections 15, 17, 18):**  
   Policy binding occurs once and immutably at run genesis (`POLICY_BINDING_IMMUTABLE = True`, `MID_RUN_POLICY_REBINDING = PROHIBITED`). Independent historical replay reproduces the exact binding from raw genesis metadata and historical registry bytes without referencing stored binding decisions.

---

## 2. Policy & Epoch Registries

### Authoritative Epoch Registry (`EPOCH_REGISTRY_V1`)
- **Version:** `1.0`
- **Canonical SHA-256:** `509d2052e6a0b40f41af730683b341319b3bca84815b3d241c09c3ba6e944eed`
- **Mapping:**
  - `EPOCH_001`: `1`
  - `EPOCH_002`: `2`
  - `EPOCH_003`: `3`
  - `EPOCH_004`: `4`

### Authoritative Policy Registry (`ANCHOR_POLICY_REGISTRY_V1`)
- **Version:** `1.0`
- **Canonical SHA-256:** `2df525bcb91939fc2df58b2fec9e3ad7cfcd03f8e889cfae5ed11f4b49b2851e`
- **Canonical Entry:**
  - `policy_id`: `SMART_MONEY_EXTERNAL_ANCHOR_POLICY_V1`
  - `policy_version`: `1.0`
  - `policy_sha256`: `f5926ecdb5536443c5b5502c4bce81452a382e7e43d4107b3ecf6d92ec3a4176`
  - `valid_from`: `2026-10-11T12:00:00Z`
  - `valid_until`: `OPEN`
  - `min_run_epoch_ordinal`: `1`
  - `revocation_status`: `ACTIVE`

---

## 3. Applicability Theorem & Boundary Semantics

Under `ALL_PREDICATES_REQUIRED`, a policy $P$ is applicable to run $R$ if and only if all three predicates hold:

$$\text{Applicable}(P, R) \iff (O(R.\text{epoch}) \ge P.\text{min\_ordinal}) \land (R.T_{\text{start}} \ge P.\text{valid\_from}) \land (R.T_{\text{end}} < P.\text{valid\_until}) \land (P.\text{revocation} = \text{ACTIVE})$$

### Boundary Relations
- **Before:** $R.T_{\text{end}} < P.\text{valid\_from} \implies \text{BEFORE} \implies \text{NO\_APPLICABLE\_POLICY}$
- **Straddles:** $R.T_{\text{start}} < P.\text{valid\_from} \le R.T_{\text{end}} \implies \text{STRADDLES\_BOUNDARY} \implies \text{INDETERMINATE\_BOUNDARY}$
- **After or Equal:** $R.T_{\text{start}} \ge P.\text{valid\_from} \implies \text{AFTER\_OR\_EQUAL} \implies \text{APPLICABLE}$

---

## 4. Retained Policy Artifacts (11 Artifacts)

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

---

## 5. Verification & Test Suite Parity

The targeted test suite `tests/test_smart_money_anchor_policy_resolution.py` executes 21 exhaustive tests:
- **15 Canonical Test Cases:** Passed (100%).
- **Clock Boundary Edge-Case Precision:** Passed (100%).
- **Registry Conflict & Overlap Prevention:** Passed (100%).
- **Emergency Revocation Semantics:** Passed (100%).
- **No Retroactive Mid-Run Rebinding:** Passed (100%).
- **Downstream Firewall Fail-Closed Blocking:** Passed (100%).
- **Historical Replay Parity:** Passed (100%).

### Full Domain Regression
- **Test Modules Executed:** 9 Smart Money test suites.
- **Total Tests Passed:** 124 of 124.
- **Failures:** 0.

---

## 6. Candidate 004 Firewall Attestation

- `PROTECTED_FILES_CHANGED = 0`
- `SHADOW_EVIDENCE_DB_WRITES = 0`
- `CANDIDATE_004_INTERFERENCE = NONE`

---

## 7. Operational Verdict

```
GATE = ANCHOR_POLICY_RESOLUTION_CLOSURE_V1
ANCHOR_POLICY_RESOLUTION_STATUS = VERIFIED
PRODUCTION_READINESS = NOT_AUTHORIZED
NEXT_ACTION = COMPLETE_FINALIZATION_LEDGER_EXTERNAL_ROOTING_DECISION, THEN_RUN_FIRST_VALID_NATURAL_EDGAR_CAPTURE
```
