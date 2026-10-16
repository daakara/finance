# VCP Production Universe Inputs — Slice 1 Implementation Note
**Component:** `analyst_dashboard/universe/validator.py`  
**Test Suite:** `tests/test_universe_validator.py`  
**Baseline Anchor:** `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (`origin/main`)  
**Branch:** `fix/universe-input-validation-core`  
**Worktree:** `C:\Users\akara\Documents\Projects\finance-universe-validation-core`  

---

### 1. Module Interface & Trust Boundary

The `UniverseInputValidator` module provides an in-memory, fail-closed input validation layer designed to prevent missing, malformed, conflicting, or unsupported source evidence from entering the universe pipeline.

#### Public Interface:
1. **`UniverseInputValidator.validate_record(raw_record, requirements, temporal_context=None, evidence=None) -> RecordValidationResult`**
   - Validates a single incoming dictionary against explicit `ValidationRequirements`.
   - Never infers defaults; never populates missing fields with "USD", "EQUITY", "COMMON_STOCK", "ACTIVE", or `primary_listing=True`.
   - Distinguishes missing keys (`MISSING_REQUIRED_FIELD`), prohibited nulls / empty strings (`NULL_NOT_PERMITTED`), boolean/string confusion (`BOOLEAN_STRING_AMBIGUITY`), malformed Gregorian dates (`MALFORMED_VALUE`), uncertified tokens (`UNKNOWN_EVIDENCE`), and temporal lookahead (`LOOKAHEAD_EVIDENCE_UNAVAILABLE`).
   - If validation fails, `downstream_admission_permitted` is `False` and `validated_values` is `None`. Unvalidated data is never exposed.

2. **`UniverseInputValidator.validate_batch(raw_records, requirements, temporal_context=None, evidence_map=None) -> BatchValidationResult`**
   - Validates a collection of records.
   - Enforces batch-level uniqueness constraints under caller-supplied `uniqueness_keys`.
   - Detects duplicate records (`DUPLICATE_RECORD`) and attribute collisions (`RECORD_COLLISION`).
   - Rejects colliding records fail-closed; does not silently select first or last write.
   - Treats supplied identifiers as opaque. Multiple securities sharing an issuer identifier (e.g. CIK) are not assumed to be duplicates when security keys differ.

3. **`to_source_security(record_result: RecordValidationResult) -> SourceSecurity`**
   - Guarded builder boundary conversion helper.
   - Strictly requires `record_result.status == ValidationStatus.VALIDATION_PASSED` and `downstream_admission_permitted == True`.
   - Requires explicit, validated values for all nine mandatory `SourceSecurity` fields (`security_id`, `symbol`, `exchange`, `security_type`, `listing_status`, `currency`, `country`, `primary_listing`, `asset_class`).
   - Completely prevents `SourceSecurity` from silently populating missing fields with its default parameters.
   - Does not invoke the universe builder, execute database queries, or write to storage.

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

### 3. Active Production Integration Status

- **Status:** **`NOT_IMPLEMENTED` (Disconnected Slice)**
- `validator.py` is an isolated module.
- It is **not** wired into production ingestion scripts (`scripts/publish_universe.py`), API screener routes (`api/routes/screener.py`), or background schedulers.
- No existing production contracts, schemas, or running services are modified by this slice.

---

### 4. Specification Clarifications & Corrections

This implementation note records the following mandatory corrections to earlier research drafts:
1. **No "Zero-Risk" Claims:** All software modifications carry operational, interface, and regression considerations. Slice 1 is bounded and isolated, but not characterized by unsupportable zero-risk claims.
2. **Identifier Semantics:** Listing FIGIs and Composite FIGIs are not interchangeable. Share Class FIGI is global; Composite FIGI represents a national composite; Listing FIGI represents venue-specific trading lines. Neither is selected as the production key in this slice; supplied identifiers remain opaque.
3. **Strategy Eligibility Thresholds:** No $10M ADV or volatility contraction floors are introduced into the validation core. Strategy thresholds belong exclusively to engine and strategy policy.
4. **Freshness Cadence:** No 72-hour freshness limit or fixed calendar rule is hardcoded. Temporal lag tolerance is configurable via `TemporalEvaluationContext.max_directory_lag_seconds`.
5. **Halts, Suspensions, and Delistings:** Regulatory trading halts and delisting events must follow canonical contracts and verified point-in-time rules. Form 25 interpretation is deferred to feed adapters.
6. **Publication Status Semantics:** `PUBLISHED_INACTIVE` is a descriptive handoff state unless supported as a persisted database enum value in `contracts.py` and SQLite triggers.
7. **Tooling Guarantees:** Hash-chain audit ledgers, atomic rollback triggers, and forward-only quarantine builds are capabilities of `UniverseStore` and publication tooling, outside the scope of this in-memory validation slice.

---

### 5. Unresolved Policy Dependencies & Next Integration Slices

Implementation of downstream publication and feed ingestion remains blocked pending Product Owner determination on:
- **Decision D-1:** Primary listing verification semantics (venue proxy vs composite MIC vs rule demotion).
- **Decision D-2:** Currency determination evidence (codified rulebook convention vs security master).
- **Decision D-3:** Production security identity key strategy (surrogate key vs Listing FIGI).
- **Decision D-4:** REIT and BDC strategy eligibility (strict operating stock exclusion vs liquidity-qualified admission).
- **Decision D-5:** Authoritative directory freshness lag threshold ($\Delta t_{\text{max}}$).

**Next Slice (Slice 2):** Authoritative Feed Adapters (`sec_edgar`, `nasdaq_trader`, `openfigi`) consuming raw vendor/regulatory feeds and producing validated dictionaries for `UniverseInputValidator`.
