# ARX Terminal — SEC Contract Evidence Identity & Custody Closure Report

**Document Identifier:** `SEC_CONTRACT_EVIDENCE_IDENTITY_CUSTODY_CLOSURE`  
**Governing Epoch:** `EPOCH_001`  
**Authority:** Independent SEC Evidence Identity Auditor  
**Working Branch:** `arx/sec-source-authority-qualification`  
**Entry Head:** `56b7a895c1d2cbe4751ab0f317ae3a1eacd79970`  
**Historical Functional Commit:** `f11bd826d1a91e8bbecda5c7490f44127a015cc6` (IMMUTABLE, PRESERVED)  
**Historical Evidence Commit:** `79789a604f40993fe66274275dae71a3a2c9991d` (IMMUTABLE, PRESERVED)  
**Historical Reconciliation Commit:** `56b7a895c1d2cbe4751ab0f317ae3a1eacd79970` (IMMUTABLE, PRESERVED)  
**Status:** **`OUTCOME_A — IDENTITY & CUSTODY CLOSED`**  

---

## 1. Executive Summary & Epistemic Audit Findings

This closure gate resolves all three evidence-identity, custody, and contract-hash questions governing SEC Source Authority Qualification Epoch 001:

1. **Issue A (PDS Technical Specification Version Identity):**  
   - Historical reported value `2.8` did NOT originate from any official SEC publication.  
   - Detailed forensic tracing across Git and agent transcripts proved `2.8` was introduced in local scratch script `generate_sec_source_contract_manifest.py` as a model generator hardcode / transcription error during synthetic text compilation.  
   - The authentic official SEC PDS Technical Specification (`PDS_Technical_Specifications_Document.pdf`, retained directly from `https://www.sec.gov/files/edgar/pds_dissemination_spec.zip`) embeds:  
     - **Title:** `EDGAR Public Dissemination Service (PDS) Technical Specification`  
     - **Embedded Version:** `Version 2.1` (Page 1)  
     - **Embedded Date:** `May 2025` (Page 1)  
     - **Release Status:** `FINAL (EDGAR Release 25.1.3.u2, May 19, 2025)` (Page 3)  
     - **Legacy Internal Section Header:** `March 2025 (Version 2.0)` (e.g. Page 71)  
     - **SEC Landing Page Update Date:** `05/19/2025`  
   - **Verdict:** `PDS_TECH_SPEC_VERSION_RECONCILIATION = PREVIOUS_VALUE_INCORRECT`. The historical reported value `2.8` is superseded by official `Version 2.1 (May 2025)`. Functional impact is strictly `NONE`.

2. **Issue B (Original SEC Source-Byte Custody & Derived-Text Provenance):**  
   - Official SEC source artifacts have been directly retrieved from SEC-hosted endpoints under SEC Fair Access policy compliance and permanently retained in `raw_contracts/original/`.  
   - Exact physical response bytes, HTTP status codes, headers (Content-Type, Content-Length, ETag, Last-Modified), and SHA-256 hashes are recorded in `SEC_SOURCE_CONTRACT_MANIFEST.json` (Manifest V2).  
   - Existing text files under `raw_contracts/` are classified as `DERIVED_TEXT_REPRESENTATION`. Their extraction provenance is classified as `MANUAL_OR_MODEL_DERIVED` / `NOT_FULLY_REPRODUCIBLE`, but a comprehensive semantic parity audit against authentic SEC PDF/HTML bytes proved **`SOURCE_EXTRACTION_SEMANTIC_DEFECT = NO`** across all six material dimensions.

3. **Issue C (Decision-Input Contract Hash Reconstruction):**  
   - Recomputation of the canonical hash of `DECISION_INPUT_CONTRACT_SPEC` (`SMART_MONEY_OPEN_MARKET_PURCHASE_V1`) directly from historical Git trees at `f11bd82`, `79789a6`, `56b7a89`, and `HEAD` yields **identically** `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0`.  
   - The value `a11093159930f7d54d2417726ff2fb979bfd0a5146c6504bc2d2d9990562e84c` never appeared in any committed repository file; it was an isolated chat transcription / reporting error in the assistant's prose response at step 30618.  
   - **Verdict:** `DECISION_INPUT_CONTRACT_HASH_RESULT = SAME_SEMANTICS_PREVIOUS_REPORTING_ERROR`. Contract semantics, implementation code, and hash determinism are 100% stable and unchanged.

---

## 2. Issue A: Forensic Reconstruction of "Version 2.8"

### 2.1. Trace of Origin
- In the original prompt and earlier engineering phases of Epoch 001, local text files were created via scratch script `generate_sec_source_contract_manifest.py` (step 30504).
- The generator hardcoded: `Document Version: 2.8 (May 19, 2025 Revision)` without referencing a live document download.
- Classification: **`PDS_VERSION_2_8_ORIGIN_CLASS = GENERATOR_HARDCODE / MODEL_INFERENCE`**.

### 2.2. Authentic Embedded Identity
Inspection of the official SEC PDF bytes (`PDS_Technical_Specifications_Document.pdf`) reveals:
- **Cover Page (Page 1):**  
  `EDGAR Public Dissemination Service (PDS) Technical Specification`  
  `Version 2.1  May 2025`
- **Summary of Changes (Page 3):**  
  `On May 19, 2025, EDGAR Release 25.1.3.u2 will introduce the following update...`
- **Internal Legacy Running Headers (Page 71):**  
  `EDGAR PDS Technical Specification  71  March 2025`  
  (Explains why historical operators observed "March 2025 / Version 2.0" in internal tables).
- **SEC Landing Page (`dissemin.htm`):**  
  `The EDGAR PDS Technical Specification Document (updated 05/19/2025)...`

Reconciliation determination:
```text
PDS_TECH_SPEC_EMBEDDED_VERSION = Version 2.1
PDS_TECH_SPEC_EMBEDDED_DATE = May 2025
PDS_TECH_SPEC_EMBEDDED_STATUS = FINAL (Release 25.1.3.u2, May 19, 2025)
PDS_TECH_SPEC_SEC_PAGE_UPDATED_AT = 05/19/2025
PDS_TECH_SPEC_VERSION_RECONCILIATION = PREVIOUS_VALUE_INCORRECT
HISTORICAL_REPORTED_VALUE = 2.8
RECONCILED_VALUE = Version 2.1 (May 2025)
HISTORICAL_VALUE_STATUS = SUPERSEDED_BY_EVIDENCE_IDENTITY_RECONCILIATION
FUNCTIONAL_IMPACT = NONE
```

---

## 3. Issue B: Original SEC Source-Byte Custody & Semantic Parity

### 3.1. Retained Original SEC Binary & HTML Artifacts

All files retrieved directly from official SEC endpoints under User-Agent `ArxTerminal/1.0 (akara@finance.app)` and retained in `docs/domain/radar/smart-money/sec-authority/raw_contracts/original/`:

| Source Document ID | Official SEC URL | Local Filename | Content Type | Bytes Received | SHA-256 |
|---|---|---|---|---|---|
| `SEC_DOC_PDS_TECH_SPEC_2025_ZIP` | `https://www.sec.gov/files/edgar/pds_dissemination_spec.zip` | `SEC_DOC_PDS_TECH_SPEC_2025_DISSEMINATION_SPEC.zip` | `application/zip` | 891,844 | `8a31951759aece4dc684056dcba1369eed4c413e261a4119c3670fd92c2351ff` |
| `SEC_DOC_PDS_TECH_SPEC_2025` | `.../pds_dissemination_spec.zip#PDS_Technical_Specifications_Document.pdf` | `SEC_DOC_PDS_TECH_SPEC_2025.pdf` | `application/pdf` | 1,076,512 | `3b7128be38d44c499529d01ea7eaeae6fbbb1a6ffb2109c90303b48ccf254a43` |
| `SEC_DOC_PDS_NEW_SUBSCRIBER_2025` | `https://www.sec.gov/files/edgar-pds-new-subscriber-document.pdf` | `SEC_DOC_PDS_NEW_SUBSCRIBER_2025.pdf` | `application/pdf` | 291,829 | `e1de5daecd6ddbf1d727b0f28fcbe6d91f1d435ea18727862d42376b12bb352b` |
| `SEC_DOC_PDS_SERVICE_PAGE` | `https://www.sec.gov/info/edgar/ednews/dissemin.htm` | `SEC_DOC_PDS_SERVICE_PAGE.html` | `text/html; charset=utf-8` | 65,051 | `f7d27ddaa34d12dd2c688cea949f3a9858c78c063c176052523648b343b93f9a` |
| `SEC_DOC_PUBLIC_WEBSITE_AVAILABILITY_GUIDANCE` | `https://www.sec.gov/os/accessing-edgar-data` | `SEC_DOC_PUBLIC_WEBSITE_AVAILABILITY_GUIDANCE.html` | `text/html; charset=utf-8` | 80,342 | `393dccc830a92f314e5d4aded255c43350dc4282fd73ad45ed5874c553e0ce05` |
| `SEC_DOC_FORM4_INSTRUCTIONS` | `https://www.sec.gov/files/form4.pdf` | `SEC_DOC_FORM4_INSTRUCTIONS.pdf` | `application/pdf` | 419,278 | `275cfc9188bc1c7b7a5b8c45abb03c799b64f9e1fe74c92a28e97598778abc2d` |
| `SEC_DOC_EDGAR_FILER_MANUAL_PAC_GUIDANCE` | `https://www.sec.gov/edgar/filer-information/current-edgar-filer-manual` | `SEC_DOC_EDGAR_FILER_MANUAL_PAC_GUIDANCE.html` | `text/html; charset=utf-8` | 98,632 | `6e433a2ed2e51fcfc152d3b561c705a621e91a2f4cfa2ac3e525b602b350793c` |

### 3.2. Semantic Parity Audit (Original Bytes vs Extracted Texts)

| Semantic Claim Dimension | Authoritative Source Clause (Original SEC Bytes) | Parity Verdict |
|---|---|---|
| **EOD Terminal Marker** | Tech Spec Page 5, Section 2: "5. Daily Audit file, named dissem2_MM-DD-YYYY.EOD... final transmission concluding dissemination" | **`PASS`** |
| **EOD Operating Hours** | Tech Spec Page 5 & 14: 6:00 a.m. to 10:00 p.m. Eastern Time | **`PASS`** |
| **PDS Recovery Mechanisms** | New Subscriber Doc Section 2.3: Retransmissions, staging redirection, PDS FTP rolling 5-day recovery | **`PASS`** |
| **PAC Immutability** | Tech Spec Page 5/9: PAC (`.pc`) priority 3, does not alter underlying filed document text | **`PASS`** |
| **Fair Access Public Web Availability** | SEC Accessing EDGAR Data guidance: Public website publication concurrent/prior to PDS; PDS receipt time does not establish priority | **`PASS`** |
| **Form 4 P/A Semantics** | Form 4 General Instruction 8 (Page 11): `P` = Open market purchase, `A` = Rule 16b-3 Grant/Award; Column 4 = (A) or (D) | **`PASS`** |

`SOURCE_EXTRACTION_SEMANTIC_DEFECT = NO`.

---

## 4. Preservation of the PDS EOD Denominator HOLD

The official SEC Technical Specification PDF directly contains the normative divergence:
- **Page 5, Section 2:** Explicitly lists four dissemination file classes (`.dissem`, `.paper`, `.pr`, `.pc`) and one daily audit file (`dissem2_MM-DD-YYYY.EOD`).
- **Page 9, Table 5:** Explicitly specifies: *"Submission name (*.dissem or *.pc); repeated for each disseminated submission for that day"*.

This discrepancy is a genuine property of official SEC documentation, not an extraction error.
Therefore:
```text
PDS_EOD_TERMINAL_MARKER_STATUS = PASS
PDS_EOD_EXACT_COMPLETENESS_SEMANTICS = HOLD
PDS_PROTOCOL_QUALIFICATION_STATUS = CONDITIONAL_PASS
HOLD_REASON = CURRENT_SEC_SOURCE_DOCUMENT_SCOPE_DISCREPANCY
EOD_DENOMINATOR_SOURCE = RAW_EOD_MANIFEST_LINES
EOD_DENOMINATOR_EXTENSION_WHITELIST = NONE
EOD_DENOMINATOR_IMPLEMENTATION_DEFECT = NO
```

---

## 5. Issue C: Decision Contract Hash Reconstruction

### 5.1. Byte-Level Recomputation Matrix

`DECISION_INPUT_CONTRACT_SPEC` was recomputed across all target commits using `canonicalize_json` (`SM_CERT_CANON_V1`):

| Target Commit | Git Object Path | Canonical Payload Length | Canonical Payload SHA-256 | Semantic Match? |
|---|---|---|---|---|
| `f11bd82` | `analyst_dashboard/smart_money/sec_source_contract.py` | 811 bytes | `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0` | Identical |
| `79789a6` | `analyst_dashboard/smart_money/sec_source_contract.py` | 811 bytes | `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0` | Identical |
| `56b7a89` | `analyst_dashboard/smart_money/sec_source_contract.py` | 811 bytes | `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0` | Identical |
| `HEAD` | `analyst_dashboard/smart_money/sec_source_contract.py` | 811 bytes | `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0` | Identical |

### 5.2. Explanation of Divergence
- The hash in the committed JSON evidence (`SEC_SOURCE_AUTHORITY_QUALIFICATION_EPOCH_001_EVIDENCE.json`), the Markdown report (`SEC_SOURCE_AUTHORITY_QUALIFICATION_EPOCH_001_REPORT.md`), and the Python code was ALWAYS `c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0`.
- The string `a11093159930f7d54d2417726ff2fb979bfd0a5146c6504bc2d2d9990562e84c` existed solely in the conversational assistant message at step 30618 due to a prose transcription error.
- **Classification:** `DECISION_INPUT_CONTRACT_HASH_RESULT = SAME_SEMANTICS_PREVIOUS_REPORTING_ERROR`.
- `FUNCTIONAL_SUPERSEDING_COMMIT_REQUIRED = NO`.

---

## 6. Test Suite & Independent Reconstruction

### 6.1. Full Pytest Regression
- **Test files executed:** 10
- **Total test cases passed:** **115** (including 20 in `test_sec_source_authority_qualification.py`)
- **Total test cases failed:** **0**
- Determinism proved over 100 iterations, key-reordering invariance, field removal sensitivity, constraint mutation sensitivity, and custody tamper rejection.

### 6.2. Independent Reconstruction
- `INDEPENDENT_SOURCE_IDENTITY_RECONSTRUCTION = PASS`
- `INDEPENDENT_SOURCE_CUSTODY_RECONSTRUCTION = PASS`
- `INDEPENDENT_DECISION_CONTRACT_HASH_RECONSTRUCTION = PASS`

---

## 7. Candidate 004 Firewall Attestation
- `shadow_evidence.db`: 0 bytes written, 0 lines modified.
- `analyst_dashboard/vcp/**`: 0 lines modified.
- `natural_trigger.py`: 0 lines modified.
- Diff against `63131b9`: `PROTECTED_FILES_CHANGED = 0`.
- `CANDIDATE_004_INTERFERENCE = NONE`.

---

## 8. Canonical Parameter Block

```text
GATE =
  SEC_CONTRACT_EVIDENCE_IDENTITY_AND_CUSTODY_CLOSURE

ENTRY_BRANCH =
  arx/sec-source-authority-qualification

ENTRY_HEAD =
  56b7a895c1d2cbe4751ab0f317ae3a1eacd79970

HISTORICAL_FUNCTIONAL_GIT_SHA =
  f11bd826d1a91e8bbecda5c7490f44127a015cc6

HISTORICAL_EVIDENCE_GIT_SHA =
  79789a604f40993fe66274275dae71a3a2c9991d

HISTORICAL_RECONCILIATION_GIT_SHA =
  56b7a895c1d2cbe4751ab0f317ae3a1eacd79970

PDS_VERSION_2_8_ORIGIN_CLASS =
  GENERATOR_HARDCODE / MODEL_INFERENCE

PDS_VERSION_2_8_ORIGIN =
  LOCAL_SCRATCH_MANIFEST_GENERATOR_SCRIPTHASH_PLACEHOLDER

PDS_TECH_SPEC_EMBEDDED_VERSION =
  Version 2.1 (prior internal header Version 2.0)

PDS_TECH_SPEC_EMBEDDED_DATE =
  May 2025 (prior internal header March 2025)

PDS_TECH_SPEC_EMBEDDED_STATUS =
  FINAL (Release 25.1.3.u2, May 19, 2025)

PDS_TECH_SPEC_SEC_PAGE_UPDATED_AT =
  05/19/2025

PDS_TECH_SPEC_VERSION_RECONCILIATION =
  PREVIOUS_VALUE_INCORRECT

ORIGINAL_SEC_TECH_SPEC_BYTES_RETAINED =
  YES

ORIGINAL_SEC_TECH_SPEC_SHA256 =
  3b7128be38d44c499529d01ea7eaeae6fbbb1a6ffb2109c90303b48ccf254a43

TECH_SPEC_DERIVED_TEXT_SHA256 =
  12cc2b39111b6c42e1b23f9eea5273bed046cbbf8ebf75a2289f1527720d57eb

ORIGINAL_SEC_SUBSCRIBER_GUIDE_BYTES_RETAINED =
  YES

ORIGINAL_SEC_SUBSCRIBER_GUIDE_SHA256 =
  e1de5daecd6ddbf1d727b0f28fcbe6d91f1d435ea18727862d42376b12bb352b

SUBSCRIBER_GUIDE_DERIVED_TEXT_SHA256 =
  e8fda8688657a420209622c088c1f1739b24eca5fb71948e9bb5e91251962d96

ORIGINAL_FORM4_INSTRUCTIONS_BYTES_RETAINED =
  YES

ORIGINAL_FORM4_INSTRUCTIONS_SHA256 =
  275cfc9188bc1c7b7a5b8c45abb03c799b64f9e1fe74c92a28e97598778abc2d

SOURCE_CONTRACT_EVIDENCE_CUSTODY =
  PASS

FULL_SOURCE_PROVENANCE_REPLAY =
  PASS

SOURCE_EXTRACTION_SEMANTIC_DEFECT =
  NO

EOD_TERMINAL_CLAIM_SOURCE_PARITY =
  PASS

EOD_SCOPE_CLAIM_SOURCE_PARITY =
  PASS

PDS_RECOVERY_CLAIM_SOURCE_PARITY =
  PASS

PAC_CLAIM_SOURCE_PARITY =
  PASS

PUBLIC_AVAILABILITY_CLAIM_SOURCE_PARITY =
  PASS

FORM4_P_A_CLAIM_SOURCE_PARITY =
  PASS

PDS_EOD_TERMINAL_MARKER_STATUS =
  PASS

PDS_EOD_EXACT_COMPLETENESS_SEMANTICS =
  HOLD

PDS_PROTOCOL_QUALIFICATION_STATUS =
  CONDITIONAL_PASS

EOD_DENOMINATOR_SOURCE =
  RAW_EOD_MANIFEST_LINES

EOD_DENOMINATOR_EXTENSION_WHITELIST =
  NONE

EOD_DENOMINATOR_IMPLEMENTATION_DEFECT =
  NO

DECISION_INPUT_CONTRACT_ID =
  SMART_MONEY_OPEN_MARKET_PURCHASE_V1

HISTORICAL_REPORTED_CONTRACT_HASH =
  a11093159930f7d54d2417726ff2fb979bfd0a5146c6504bc2d2d9990562e84c

RECONCILIATION_REPORTED_CONTRACT_HASH =
  c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0

HISTORICAL_RECOMPUTED_CONTRACT_HASH =
  c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0

CURRENT_RECOMPUTED_CONTRACT_HASH =
  c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0

DECISION_INPUT_CONTRACT_HASH_RESULT =
  SAME_SEMANTICS_PREVIOUS_REPORTING_ERROR

DECISION_CONTRACT_HASH_SCHEME_VERSION =
  SM_CERT_CANON_V1

DECISION_CONTRACT_HASH_DETERMINISTIC =
  YES

DECISION_CONTRACT_HASH_ENVIRONMENT_INDEPENDENT =
  YES

DECISION_INPUT_CONTRACT_SEMANTIC_CHANGE =
  NO

P_A_FUNCTIONAL_DEFECT =
  NO

TRANSACTION_CODE_A_ACCEPTED_AS_OPEN_MARKET_PURCHASE =
  NO

SYNTHETIC_ENGINEERING_REPLAY =
  PASS

SYNTHETIC_INDEPENDENT_RECONSTRUCTION =
  PASS

LIVE_PDS_BUNDLE_REPLAY =
  NOT_EXECUTED

LIVE_PDS_INDEPENDENT_RECONSTRUCTION =
  NOT_EXECUTED

CURRENT_SMART_MONEY_DISCOVERY_AUTHORITY =
  BOUNDED_STABLE

REAL_PDS_ACCESS =
  NOT_AVAILABLE

REAL_PDS_SOURCE_AUTHORITY =
  NOT_ESTABLISHED

EXACT_FIRST_PUBLIC_AVAILABLE_AT =
  NOT_ESTABLISHED

SMART_MONEY_MARKET_WIDE_AUTHORITY =
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

FUNCTIONAL_SUPERSEDING_COMMIT_REQUIRED =
  NO

SEC_IDENTITY_CLOSURE_FUNCTIONAL_SHA =
  f11bd826d1a91e8bbecda5c7490f44127a015cc6

SEC_IDENTITY_CLOSURE_EVIDENCE_SHA =
  [CAPTURED_AT_EVIDENCE_FREEZE]

REMOTE_PUSH =
  NO

MAIN_MERGE =
  NO

PRODUCTION_DEPLOYMENT =
  NO

WORKTREE_CLEAN =
  YES
```
