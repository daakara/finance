# ARX Terminal — Controlled Synthetic Intake Pipeline Verification (Gate E)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate E — Controlled Synthetic Intake Pipeline  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Status:** `VERIFIED_SYNTHETIC_PIPELINE`  

---

## 1. 10-Step Atomic Intake Pipeline Sequence

The evidence intake pipeline is implemented in `analyst_dashboard/research/custody/intake.py` via the `IntakePipeline` class. It enforces strict boundary validation and privacy-first ingestion before any payload reaches long-term storage:

```
[Raw Submission Bytes Ingested]
               |
[Step 1: Boundary & Size Validation] -> Max 2MB limit; reject empty payloads
               |
[Step 2: Pre-Parse Wire Hashing]     -> Deterministic SHA-256 digest before JSON decode
               |
[Step 3: Safe JSON Decoding]         -> UTF-8 parse; suppress raw bytes in exceptions
               |
[Step 4: Draft-07 Schema Audit]      -> Validate against OBSERVER_RATING_SCHEMA.json
               |
[Step 5: Pseudonym & Mode Check]     -> Synthetic mode strictly rejects OBS-HUMAN-*
               |
[Step 6: Trial & Taxonomy Audit]     -> 30 trials (ALPHA, BETA, GAMMA); check CM/SM codes
               |
[Step 7: Replay & Duplicate Check]   -> Reject duplicate raw hash or duplicate active rater
               |
[Step 8: Authenticated Encryption]   -> AES-256-GCM encryption with unique 96-bit nonce
               |
[Step 9: Vault Storage]              -> Atomic staging write to off-Git vault sealed/ partition
               |
[Step 10: Ledger Inscription]        -> Append receipt; assign PENDING or PAIR_SEALED state
```

---

## 2. Ingestion Rules & Boundary Defenses

1. **Size Boundary:** Submissions exceeding 2,097,152 bytes (2MB) are rejected immediately (`IntakeValidationError`).
2. **Pre-Parse Digest:** SHA-256 is computed directly on wire bytes. This prevents any ambiguity arising from JSON key reordering or whitespace canonicalization.
3. **Draft-07 Schema Conformity:** Uses `jsonschema.Draft7Validator` against `tests/fixtures/qualification/observer_package/OBSERVER_RATING_SCHEMA.json`. Errors report exact schema path without dumping user notes.
4. **Synthetic Mode Enforcement:**  
   - Operating under `is_synthetic_mode = True`.
   - Observer IDs matching `^OBS-HUMAN-` are rejected unless `human_authorization_token == "ARX_HUMAN_INTAKE_AUTHORIZED"` is provided.
   - Synthetic pseudonyms (`OBS-SYNTHETIC-*`, `OBS-TEST-*`) or `synthetic_provenance: true` are required.
5. **30-Trial Exhaustive Coverage:** Submissions must contain all 30 approved trial IDs across `ALPHA-01..10`, `BETA-01..10`, and `GAMMA-01..10`. Duplicate trials or missing trials cause immediate fail-closed rejection.
6. **Taxonomy Code Whitelist:** Assigned codes are strictly validated against canonical `CM-01..CM-09` and supplemental `SM-01..SM-06`. Unrecognized codes fail closed.
7. **Replay Rejection:** If `raw_submission_sha256` matches an existing record in the ledger, the submission is rejected as a replay attack.

---

## 3. Automated Test Evidence

The intake pipeline was verified across 8 automated adversarial test cases in `tests/test_evidence_custody_security.py`:

| Test ID | Scenario Description | Tested Condition | Result |
| :--- | :--- | :--- | :---: |
| `test_04` | Unauthorized Human Token | `OBS-HUMAN-01` rejected under synthetic mode | **PASSED** |
| `test_05` | Malformed JSON Payload | Syntax error handled without leaking raw bytes | **PASSED** |
| `test_06` | Invalid Schema Structure | Missing required `ratings` field rejected | **PASSED** |
| `test_07` | Duplicate Observer Submission | Same observer submitting twice for same batch | **PASSED** |
| `test_08` | Submission Replay Attack | Exact same byte hash submitted across batches | **PASSED** |
| `test_12` | Missing Batch Identity | Empty `registered_batch_id` rejected | **PASSED** |
| `test_13` | Missing Package Hash | Empty `expected_package_sha` rejected | **PASSED** |
| `test_17` | Dual Valid Intake Execution | Two raters ingested; state moves to `PAIR_SEALED` | **PASSED** |

### Execution Command:
```powershell
python -m pytest tests/test_evidence_custody_security.py -k "unauthorized or malformed or schema or duplicate or replay or package or dual" -v
```
**Exit Code:** `0` (8 passed)

---

## 4. Current Operational Boundary

- **Active Intake Status:** `SYNTHETIC_TEST_MODE_ONLY`
- **Real Human Submissions Processed:** `0`
- **Real Human Observers Assigned:** `0`
- **Pilot P00 Activation:** `NOT_AUTHORIZED`
