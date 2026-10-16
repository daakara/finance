# VCP Production Universe Inputs — Slice 1 Implementation Note
**Component:** `analyst_dashboard/universe/validator.py`  
**Test Suite:** `tests/test_universe_validator.py`  
**Baseline Anchor:** `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (`origin/main`)  
**Branch:** `fix/universe-input-validation-core`  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-universe-validation-core`  

---

### 1. Module Interface & Trust Boundary

The `UniverseInputValidator` module provides an in-memory, fail-closed input validation layer designed to prevent missing, malformed, conflicting, or unsupported source evidence from silently becoming certified universe inputs.

#### Public Interface
1. **`UniverseInputValidator.validate_record(raw_record, requirements, temporal_context=None, evidence=None) -> RecordValidationResult`**
   - Validates a single incoming dictionary against explicit, caller-supplied `ValidationRequirements`.
   - Never infers defaults; never populates missing fields with "USD", "EQUITY", "COMMON_STOCK", "ACTIVE", or `primary_listing=True`.
   - Distinguishes missing keys (`MISSING_REQUIRED_FIELD`), prohibited nulls / empty strings (`NULL_NOT_PERMITTED`), boolean/string confusion (`BOOLEAN_STRING_AMBIGUITY`), malformed Gregorian dates (`MALFORMED_VALUE`), uncertified tokens (`UNKNOWN_EVIDENCE`), and temporal lookahead (`LOOKAHEAD_EVIDENCE_UNAVAILABLE`).
   - If validation fails, `downstream_admission_permitted` is `False` and `validated_values` is `None`. Unvalidated data is never exposed.

2. **`UniverseInputValidator.validate_batch(raw_records, requirements, temporal_context=None, evidence_map=None) -> BatchValidationResult`**
   - Validates a collection of records against caller requirements and batch-level uniqueness constraints.
   - Enforces the **atomic batch contract**: if any record fails or if any batch collision occurs, downstream admission is blocked for the entire batch.
   - Detects duplicate records (`DUPLICATE_RECORD`) and attribute collisions (`RECORD_COLLISION`).
   - Treats supplied identifiers as opaque. Multiple securities sharing an issuer identifier (e.g. CIK) are not inferred as duplicates when security keys differ.

3. **`to_source_security(record_result: RecordValidationResult) -> SourceSecurity`**
   - Guarded builder boundary conversion helper for an admitted single record.
   - Strictly requires `record_result.status == ValidationStatus.VALIDATION_PASSED` and `downstream_admission_permitted == True`.
   - Enforces conversion completeness: all 12 fields consumed by `SourceSecurity` must have been explicitly validated.
   - Completely prevents `SourceSecurity` from silently populating missing fields with its default parameters.
   - Zero side-effects: does not execute the universe builder, query databases, or write to storage.

4. **`BatchValidationResult.to_source_securities() -> List[SourceSecurity]`**
   - Batch-approved conversion helper.
   - Converts all records in a batch if and only if `is_batch_valid is True`.
   - Raises `ValueError` if called on any rejected batch.

#### Trust Boundary:
- `VALIDATION_PASSED` asserts **only** that the record structurally satisfies caller-supplied validation requirements.
- External evidence references (`EvidenceReference`) remain external claims; input declarations do not constitute independent verification.
- Strategy eligibility (rules R01–R07), publication readiness, and execution authorization remain strictly unevaluated.

---

### 2. Validation versus Strategy Eligibility

| Dimension | Input Validation Core (`validator.py`) | Strategy Eligibility Builder (`builder.py`) |
|---|---|---|
| **Purpose** | Evidence integrity, schema conformance, type safety, anti-lookahead. | Investment strategy qualification under canonical rules R01–R07. |
| **Authority** | Policy-independent; checks explicit syntax and constraints. | Governed by `CANONICAL_ELIGIBILITY_RULES` and freeze authority V2.5.1. |
| **Decision Range** | `VALIDATION_PASSED`, `VALIDATION_FAILED`, `VALIDATION_UNRESOLVED`. | `ELIGIBLE`, `INELIGIBLE`, `UNRESOLVED`. |
| **Output Target** | Validated structured records (or admission blocked). | `EligibilityLedgerRow` with rule evaluation hash and decision hash. |

Passing input validation is a necessary prerequisite for builder consideration, but does **not** mark a security as `ELIGIBLE`.

---

### 3. Review Outcomes & Public Admission Boundaries

#### Review A — Conversion Completeness & Temporal Handling
- **Mandatory Fields:** All nine non-temporal fields (`security_id`, `symbol`, `exchange`, `security_type`, `listing_status`, `currency`, `country`, `primary_listing`, `asset_class`) must be explicitly present and non-null in `validated_values`.
- **Temporal Event Fields:** `listing_date`, `delisting_date`, and `trading_suspension_date` must be explicitly evaluated during validation:
  1. *Omitted / Missing Evidence:* If an event field is omitted from the raw record, `to_source_security()` rejects conversion fail-closed with `ValueError`. Missing evidence cannot silently disappear into `SourceSecurity` defaults.
  2. *Absence of an Event:* If the caller explicitly provides `None` (e.g. `delisting_date: None`), this is validated as a permitted null representing "no event" and preserved as `None` in `SourceSecurity`.
  3. *Affirmative Event Date:* If a valid date string is provided (e.g. `delisting_date: "2026-10-14"`), it is validated for Gregorian date syntax and temporal ordering (`delisting_date > listing_date`), and preserved in `SourceSecurity`.
  4. *Malformed / Unknown Dates:* Malformed dates trigger `MALFORMED_VALUE`; explicit tokens ("UNKNOWN") trigger `UNKNOWN_EVIDENCE`. Both block conversion.

#### Review B — Record versus Batch Admission Semantics
- **Atomic Batch Contract:** Batch admission is strictly atomic. Partial-batch acceptance is not permitted.
- **Clean Record in Rejected Batch:** A record that is structurally valid in isolation within a batch that fails (due to another record's missing field, an identity collision, or an unpermitted duplicate) has its downstream admission blocked (`downstream_admission_permitted = False`, `validated_values = None`, issue code `BATCH_ADMISSION_BLOCKED`).
- **Batch Safe Conversion:** Calling `to_source_security()` on any record from a failed batch raises `ValueError`. Batch conversion via `batch_result.to_source_securities()` or `batch_result.get_validated_records()` raises `ValueError` unless `batch_result.is_batch_valid is True`.
- **Identity & Cardinality Behavior:**
  - *Identical Duplicates:* Flagged as `DUPLICATE_RECORD` and rejected fail-closed unless `allow_identical_duplicates=True`.
  - *Conflicting Records:* Records sharing an identity key with conflicting attributes trigger `RECORD_COLLISION` and fail the batch.
  - *Issuer CIK Cardinality:* Multiple securities sharing an issuer CIK (e.g. `GOOG` and `GOOGL` sharing `0001652044`) are admitted normally when security identities differ.

#### Review C — Evidence Requirements
- **Configurable Evidence Demand:** `require_evidence_reference` explicitly dictates whether provenance evidence is required. When `False`, evidence remains optional. When `True`, missing evidence triggers `MISSING_EVIDENCE_REFERENCE` and blocks admission.
- **Trust Boundary Preservation:** `EvidenceReference.is_external_attestation` defaults to `True`. Supplied evidence is never labeled as independently verified by the Terminal.
- **Conversion Requirement Boundary:** Conversion via `to_source_security()` cannot bypass the requirements under which validation ran.

---

### 4. CI Marker Selection & Test Pyramid Rationale

- **Assigned Marker:** `pytestmark = pytest.mark.tier2c` (`Provenance and Data Gate`).
- **Reconciliation Rationale:** The initial implementation assigned both `tier1` and `tier2c` to `test_universe_validator.py`. In `ci.yml`, Step 32 runs `pytest -m tier1` and Step 38 runs `pytest -m "tier2a or tier2b or tier2c"`. Marking both caused 100% duplicate execution (136 total test runs for 68 unique tests).
- **Alignment:** Every other data and provenance test suite in the repository (`test_adversarial_phase17_provenance.py`, `test_data_fetchers.py`, `test_regulatory_data_feeds.py`) is assigned `tier2c`. Aligning to `tier2c` ensures the suite executes exactly once per PR in the Tier 2 gate, completing within ~1.6s.

---

### 5. Active Production Integration Status & Specification Corrections

- **Status:** **`NOT_IMPLEMENTED` (Disconnected Slice)**
- `validator.py` is an isolated module. It is **not** wired into production ingestion (`scripts/publish_universe.py`), API screener routes, or background schedulers.
- **Specification Corrections:**
  1. *No "Zero-Risk" Claims:* Slice 1 is bounded and isolated, but not characterized by unsupportable zero-risk claims.
  2. *Identifier Semantics:* Listing FIGIs and Composite FIGIs are not interchangeable. Neither is selected as the production key in this slice; supplied identifiers remain opaque.
  3. *Strategy Eligibility Thresholds:* No $10M ADV or volatility contraction floors are introduced into the validation core.
  4. *Freshness Cadence:* No 72-hour freshness limit or fixed calendar rule is hardcoded. Temporal lag tolerance is configurable via `TemporalEvaluationContext.max_directory_lag_seconds`.
  5. *Unapproved Tooling Proposals:* References to "hash-chain audit logging" in research notes remain unapproved proposals and are outside this validation slice.
  6. *Feed Parsing Decoupling:* Feed parsing (e.g. SEC EDGAR, Nasdaq Trader directories) can be investigated and implemented separately from policy-dependent eligibility mapping under subsequent authorization.

---

### 6. Unresolved Policy Dependencies & Next Integration Slices

Implementation of downstream publication and feed ingestion remains blocked pending Product Owner determination on:
- **Decision D-1:** Primary listing verification semantics (venue proxy vs composite MIC vs rule demotion).
- **Decision D-2:** Currency determination evidence (codified rulebook convention vs security master).
- **Decision D-3:** Production security identity key strategy (surrogate key vs Listing FIGI).
- **Decision D-4:** REIT and BDC strategy eligibility (strict operating stock exclusion vs liquidity-qualified admission).
- **Decision D-5:** Authoritative directory freshness lag threshold ($\Delta t_{\text{max}}$).

**Next Slice (Slice 2):** Dedicated feed parsers (`sec_edgar`, `nasdaq_trader`, `openfigi`) consuming raw vendor/regulatory feeds and producing typed input dictionaries for `UniverseInputValidator`.
