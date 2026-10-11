# ARX Terminal — SEC Source Authority Qualification Epoch 001 Reconciliation Report

**Document Identifier:** `SEC_SOURCE_AUTHORITY_EPOCH_001_RECONCILIATION`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Source Authority Reconciliation Engineer  
**Working Branch:** `arx/sec-source-authority-qualification`  
**Qualification Base SHA:** `46ecc732196afe842cf2b30dc00fdabbe79f1b2e`  
**Historical Functional Commit:** `f11bd826d1a91e8bbecda5c7490f44127a015cc6` (IMMUTABLE, PRESERVED)  
**Historical Evidence Commit:** `79789a604f40993fe66274275dae71a3a2c9991d` (IMMUTABLE)  
**Reconciliation Status:** **`PASS / RECONCILED UNDER OUTCOME B`**  

---

## 1. Executive Summary & Epistemic Audit Findings

This reconciliation audit was convened to resolve three specific technical and provenance questions concerning SEC Source Authority Qualification Epoch 001:

1. **Issue 1 (Form 4 P/A Semantics):**  
   Did `SMART_MONEY_OPEN_MARKET_PURCHASE_V1` or the underlying codebase conflate transaction code `P` (Open Market Purchase) with acquired/disposed code `A` (Acquired), or accept transaction code `A` (Equity Grant) as open-market purchase evidence?  
   **Finding:** **`P_A_RESULT = NO_DEFECT_DOCUMENTATION_AMBIGUITY`**.  
   Active production code in `form4_parser.py` (lines 243–253) and `sec_source_contract.py` (lines 676–680) strictly requires **both** `transactionCode == "P"` **AND** `acquiredDisposed == "A"`. Transaction code `A` maps strictly to `SmartMoneySignalType.EQUITY_GRANT` and evaluates to `is_open_market_purchase_evidence = False`. Shorthand narrative in prior conversational logs caused external ambiguity, but zero functional defects exist. Six dedicated test fixtures have been committed to permanently guard this invariant.

2. **Issue 2 (PDS EOD Denominator Contract Reconciliation):**  
   Does an internal inconsistency exist between SEC PDS narrative guidance (which enumerates `*.dissem`, `*.paper`, `*.pr`, `*.pc`) and Technical Specification Table 5 (which focuses on primary electronic dissemination `*.dissem` and `*.pc`)?  
   **Finding:**  
   - Terminal Marker Semantics: **`PDS_EOD_TERMINAL_MARKER_STATUS = PASS`**. The EOD audit file is definitively the terminal transmission confirming dissemination has concluded for the business day.  
   - Exact Denominator Status: **`PDS_EOD_EXACT_DENOMINATOR_STATUS = HOLD`**. Because Table 5 focuses on electronic dissemination while narrative guidance includes paper and staff-review records, claiming complete exact-set closure of the denominator without a live `.EOD` feed sample would constitute unverified epistemic overclaim.  
   - Protocol Status: **`PDS_PROTOCOL_QUALIFICATION_STATUS = CONDITIONAL_PASS`** (Outcome B).

3. **Issue 3 (Raw SEC Source-Byte Custody Verification):**  
   Were official SEC PDF/HTML binary bytes retained in `raw_contracts/`, or were derived `.txt` representations retained?  
   **Finding:** **`RAW_SEC_SOURCE_BYTE_CUSTODY_STATUS = DERIVED_TEXT_REPRESENTATION_ONLY`**.  
   The five files in `docs/domain/radar/smart-money/sec-authority/raw_contracts/` are text extracts / transcripts of official SEC publications created during local qualification. The manifest `SEC_SOURCE_CONTRACT_MANIFEST.json` has been updated to formally declare `custody_classification = "DERIVED_TEXT_REPRESENTATION"`, verify all official SEC canonical URLs, and compute canonical content hashes.

---

## 2. Issue 1: Form 4 P/A Semantic Audit & Verification Matrix

### 2.1. Codebase Line-Level Verification

In `analyst_dashboard/smart_money/form4_parser.py`:
```python
# Lines 243-252:
signal_type = cls._classify_transaction_code(t_code)

is_positive = (
    signal_type == SmartMoneySignalType.OPEN_MARKET_PURCHASE
    and acq_disp == "A"
    and price > 0.0
    and shares > 0.0
)
```
Where `_classify_transaction_code`:
```python
if code == "P": return SmartMoneySignalType.OPEN_MARKET_PURCHASE
elif code == "A": return SmartMoneySignalType.EQUITY_GRANT
elif code == "S": return SmartMoneySignalType.OPEN_MARKET_SALE
...
```

In `analyst_dashboard/smart_money/sec_source_contract.py`:
```python
# Lines 676-679:
if tx.get("transaction_code") != "P":
    missing.append(f"transactions[{i}].transactionCode=='P'")
if tx.get("acquired_disposed") != "A":
    missing.append(f"transactions[{i}].acquiredDisposed=='A'")
```

### 2.2. Six-Case Verification Matrix

| Case ID | Transaction Code | Acquired / Disposed | Price / Shares | Form 4 Parser Classification | Positive Purchase Evidence? | Decision-Input Complete? | Audit Verdict |
|---|---|---|---|---|---|---|---|
| `CASE_P_A` | `P` | `A` | `> 0` | `OPEN_MARKET_PURCHASE` | **`True`** | **`True`** | **PASS (Candidate)** |
| `CASE_A_A` | `A` | `A` | `0.0` / `>0` | `EQUITY_GRANT` | **`False`** | **`False`** | **PASS (Grant Guarded)** |
| `CASE_S_D` | `S` | `D` | `> 0` | `OPEN_MARKET_SALE` | **`False`** | **`False`** | **PASS (Sale Guarded)** |
| `CASE_G_A` | `G` | `A` | `0.0` / `>0` | `GIFT` | **`False`** | **`False`** | **PASS (Gift Guarded)** |
| `CASE_M_A` | `M` | `A` | `> 0` | `OPTION_EXERCISE` | **`False`** | **`False`** | **PASS (Exercise Guarded)** |
| `CASE_P_D` | `P` | `D` | `> 0` | `OPEN_MARKET_PURCHASE` | **`False`** | **`False`** | **PASS (Fail-Closed on D)** |

`FUNCTIONAL_SUPERSEDING_COMMIT_REQUIRED = NO`.  
Historical commit `f11bd826d1a91e8bbecda5c7490f44127a015cc6` is preserved without mutation.

---

## 3. Issue 2: PDS EOD Denominator Contract Reconciliation

### 3.1. Normative Divergence
- **Technical Specification Section 3/4 & Subscriber Guide Section 4:**  
  State that the daily EOD file lists all disseminated submissions sent between 6:00 a.m. and the close of dissemination: `*.dissem`, `*.paper`, `*.pr`, and `*.pc`.
- **Technical Specification Table 5:**  
  Focuses specifically on primary electronic dissemination feeds (`*.dissem` and `*.pc`), leaving the handling of paper filings and staff review documents unsummarized in that specific table.

### 3.2. Reconciliation Determination
1. **Terminal Marker Status:**  
   The existence and receipt of `dissem2_MM-DD-YYYY.EOD` (or `dissem_MM-DD-YYYY.EOD`) is unequivocally established as the **final transmission** concluding EDGAR dissemination for the day. `PDS_EOD_TERMINAL_MARKER_STATUS = PASS`.
2. **Exact Denominator Status:**  
   Because live subscriber feeds are required to inspect the actual raw lines of physical EOD files across business days to verify whether `.paper` and `.pr` lines are present, claiming full mathematical closure of the denominator at this prequalification stage would be premature.  
   Therefore: `PDS_EOD_EXACT_DENOMINATOR_STATUS = HOLD`.
3. **Protocol Classification:**  
   `PDS_PROTOCOL_QUALIFICATION_STATUS = CONDITIONAL_PASS`.  
   Under this classification (Outcome B), the PDS protocol and terminal marker are prequalified, but exact denominator closure is held pending live PDS audit.

---

## 4. Issue 3: Raw SEC Source-Byte Custody Reconciliation

### 4.1. Physical Custody State
The contract documents located in `docs/domain/radar/smart-money/sec-authority/raw_contracts/` are ASCII/UTF-8 text extracts representing the technical specifications and subscriber guidance of the SEC EDGAR system:
- `SEC_DOC_PDS_TECH_SPEC_2025.txt` (3,239 bytes, SHA-256: `12cc2b39111b...`)
- `SEC_DOC_PDS_NEW_SUBSCRIBER_2025.txt` (2,155 bytes, SHA-256: `e8fda8688657...`)
- `SEC_DOC_PDS_SERVICE_PAGE.txt` (1,191 bytes, SHA-256: `5730aa86c068...`)
- `SEC_DOC_PUBLIC_WEBSITE_AVAILABILITY_GUIDANCE.txt` (1,749 bytes, SHA-256: `4a4777f36eba...`)
- `SEC_DOC_EDGAR_FILER_MANUAL_PAC_GUIDANCE.txt` (2,127 bytes, SHA-256: `c8e9c25cecc6...`)

### 4.2. Custody Classification Update
`SEC_SOURCE_CONTRACT_MANIFEST.json` has been updated with:
- Top-level `custody_classification = "DERIVED_TEXT_REPRESENTATION"`
- Document-level `content_representation_class = "DERIVED_TEXT_REPRESENTATION"`
- Canonical URL references to `https://www.sec.gov`
- Manifest content hash: `864563618e02c2f4348b1b4d10949d876230d3bb55cf2426515dc6a3ecdfd2ba`

---

## 5. Test Suite Verification

Full test execution across all 10 Smart Money test suites:
- `tests/test_smart_money_api.py`: 3 passed
- `tests/test_smart_money_backfill.py`: 12 passed
- `tests/test_smart_money_certification_hardening.py`: 25 passed
- `tests/test_smart_money_closure_reconciliation.py`: 11 passed
- `tests/test_smart_money_form4_pipeline.py`: 8 passed
- `tests/test_smart_money_incremental_wave2c.py`: 12 passed
- `tests/test_smart_money_legislative.py`: 8 passed
- `tests/test_smart_money_remediation_r1.py`: 9 passed
- `tests/test_smart_money_universe_authority.py`: 7 passed
- `tests/test_sec_source_authority_qualification.py`: 17 passed

**Total:** **`112 passed, 0 failed, 1 warning (deprecation)`** in 90.59s.

---

## 6. Candidate 004 Firewall Attestation

- `shadow_evidence.db`: 0 bytes written, 0 lines modified.
- `analyst_dashboard/vcp/**`: 0 lines modified.
- `natural_trigger.py`: 0 lines modified.
- Diff against `63131b9`: `PROTECTED_FILES_CHANGED = 0`.
- `CANDIDATE_004_INTERFERENCE = NONE`.

---

## 7. Canonical Parameter Block

```text
RECONCILIATION_GATE =
  SEC_SOURCE_AUTHORITY_EPOCH_001_RECONCILIATION

GOVERNING_EPOCH =
  EPOCH_001

WORKING_BRANCH =
  arx/sec-source-authority-qualification

QUALIFICATION_BASE_SHA =
  46ecc732196afe842cf2b30dc00fdabbe79f1b2e

HISTORICAL_FUNCTIONAL_GIT_SHA =
  f11bd826d1a91e8bbecda5c7490f44127a015cc6

HISTORICAL_EVIDENCE_GIT_SHA =
  79789a604f40993fe66274275dae71a3a2c9991d

FUNCTIONAL_SUPERSEDING_COMMIT_REQUIRED =
  NO

SEC_AUTHORITY_FUNCTIONAL_GIT_SHA =
  f11bd826d1a91e8bbecda5c7490f44127a015cc6

P_A_RESULT =
  NO_DEFECT_DOCUMENTATION_AMBIGUITY

P_A_FUNCTIONAL_DEFECT =
  NO

FORM4_TRANSACTION_CODE_P_MEANING =
  OPEN_MARKET_OR_PRIVATE_PURCHASE

FORM4_TRANSACTION_CODE_A_MEANING =
  GRANT_AWARD_RULE_16B3

FORM4_ACQUIRED_DISPOSED_A_MEANING =
  ACQUIRED

OPEN_MARKET_PURCHASE_REQUIRED_TRANSACTION_CODE =
  P

OPEN_MARKET_PURCHASE_REQUIRED_ACQUIRED_DISPOSED_CODE =
  A

TRANSACTION_CODE_A_ACCEPTED_AS_OPEN_MARKET_PURCHASE =
  NO

DECISION_INPUT_CONTRACT_ID =
  SMART_MONEY_OPEN_MARKET_PURCHASE_V1

DECISION_INPUT_CONTRACT_SHA256 =
  c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0

PDS_EOD_TERMINAL_MARKER_STATUS =
  PASS

PDS_EOD_EXACT_DENOMINATOR_STATUS =
  HOLD

PDS_PROTOCOL_QUALIFICATION_STATUS =
  CONDITIONAL_PASS

RAW_SEC_SOURCE_BYTE_CUSTODY_STATUS =
  DERIVED_TEXT_REPRESENTATION_ONLY

DERIVED_TEXT_REPRESENTATION_PROVENANCE_RECORDED =
  YES

SOURCE_CONTRACT_MANIFEST_SHA256 =
  864563618e02c2f4348b1b4d10949d876230d3bb55cf2426515dc6a3ecdfd2ba

QUALIFICATION_OUTCOME =
  OUTCOME_B — CONDITIONAL PREQUALIFICATION / PDS PROTOCOL PREQUALIFIED WITH EXACT DENOMINATOR ON HOLD PENDING LIVE EOD AUDIT

CURRENT_SMART_MONEY_DISCOVERY_AUTHORITY =
  BOUNDED_STABLE

PDS_SOURCE_AUTHORITY_STATUS =
  PREQUALIFIED_PENDING_LIVE_EVIDENCE

HOLD_TIME_CREATES_SOURCE_AUTHORITY =
  NO

PDS_DELIVERY_TIME_USED_AS_FIRST_PUBLIC_AVAILABLE_AT =
  NO

SEC_ACCEPTANCE_TIME_EQUALS_FIRST_PUBLIC_AVAILABILITY =
  NO

UNDETECTED_SEMANTIC_TAMPERS =
  0

PYTEST_TEST_CASES_PASSED =
  112

PYTEST_TEST_SUITES_PASSED =
  10

PYTEST_FAILURES =
  0

MARKET_WIDE_CAPABILITY =
  NOT_AUTHORIZED

EMPIRICAL_SCANNER_QUALITY =
  INSUFFICIENT_EVIDENCE

MODEL_TUNING =
  FROZEN

PRODUCTION_READINESS =
  NOT_AUTHORIZED

RADAR_PRODUCTION_STATUS =
  PIPELINE_PENDING

PROTECTED_FILES_CHANGED =
  0

CANDIDATE_004_INTERFERENCE =
  NONE

REMOTE_PUSH =
  NO

MAIN_MERGE =
  NO

PRODUCTION_DEPLOYMENT =
  NO

WORKTREE_CLEAN =
  YES
```
