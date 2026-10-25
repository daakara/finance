# ARX Terminal — Sprint 2 Historical SIP & Identity Recovery Code Release
## Primary Author Review & Audit Handoff

> [!NOTE]
> **Status:** Author Release Candidate Handoff (Distinct from Independent Audit)  
> **Target Base:** `origin/main` (`adc9cb3500988da621600a0c6558854b476faf28`)  
> **Release Branch:** `fix/sprint2-historical-sip-release`  
> **Scope:** Main-relative bounded historical SIP ingestion adapter, atomic replacement writer safeguards, verified identity execution recovery diagnosis, applied-receipt semantic retention, deterministic test fixtures, and remote CI matrix integration.  
> **Independent Reviewer Verdict:** Pending independent review against pushed release PR.

---

### 1. Executive Summary & Review Readiness

This release candidate consolidates the independently closed findings from the historical SIP ingestion and identity-recovery audits onto current mainline `origin/main` (`adc9cb3500988da621600a0c6558854b476faf28`).

#### Key Corrections Included
1. **Historical SIP Ingestion & Validation (`scripts/ingest_alpaca_historical.py`):**
   - Implements bounded daily OHLCV bar acquisition over the official Alpaca Market Data v2 endpoint (`feed="sip"`, `adjustment="split"`).
   - Enforces strict calendar session verification via `exchange_calendars` (NYSE calendar), rejecting weekends, market holidays, and dates past `MAX_COMPLETED_SESSION = "2026-10-19"`.
   - Cycle detection and bounded pagination budget safety (max 25 pages).
   - Defect A closure: incremental refresh strictly requires that the stored prefix `[start_date, latest_stored]` contains zero missing sessions (`prefix_missing == set()`). A missing session (e.g. `2026-02-06`) blocks refresh and forces backfill.
   - Rejects replacement if unprovenanced records exist outside the reviewed scope.
   - Post-retrieval validation confirms full store window coverage (`INCOMPLETE_FINAL_STORE_WINDOW`).
   - L03 closing-price equivalence is conditionally framed (`l03_equivalence_established = False`).
2. **Atomic Store Scope Safeguards (`analyst_dashboard/data/market_db.py`):**
   - Supports non-mutating `auto_init=False` engine construction.
   - `save_daily_candles_with_evidence` enforces mandatory non-null `scope_start_date` and `scope_end_date` whenever `replace_existing_scope=True`.
   - Validates ISO date formats and interval ordering (`start <= end`).
   - Completely removes the whole-symbol deletion fallback (`DELETE ... WHERE symbol = ?`).
3. **Identity Execution Recovery & Applied-Receipt Integrity (`scripts/apply_registration_plan_wrapper.py`):**
   - `diagnose_execution_state`: Read-only triage diagnosis requiring reviewed plan evidence, matching audit ledger digests, and verifying full association attributes (`operating_mic`, `effective_from`, `effective_to`).
   - `create_verified_sqlite_backup`: Atomic pre-apply verified SQLite backup with target state digest agreement.
   - Defect B closure: `archive_and_verify_audit_bundle` deserializes applied receipts into `IdentityAdoptionPlan`, validates recomputed content digests, compares approved transformation digests, performs element-by-element attribute matching for all 130 entities and 130 associations, checks coherent authorization flags, and verifies target database audit ledger linkage. Fails closed with `RuntimeError` and preserves staging on any defect.
4. **Clean-Checkout Deterministic Fixtures (`data/`, `evidence/coverage_expansion/`):**
   - All necessary research fixtures (`bounded_150_dev_market_store.db`, `raw_feeds/2026-10-16`, `reference_polygon/tickers`, and the 14 operational package specifications) are tracked for clean-checkout test reproducibility without live provider calls or credentials.
5. **Remote CI Coverage (`.github/workflows/ci.yml`):**
   - Adds an explicit workflow step to execute all 8 historical SIP and identity expansion test modules on both Python 3.11 and 3.12.
   - Adds bytecode compilation verification for the new scripts.

**Recommendation:** The code candidate is **READY FOR GUARDED MERGE REVIEW**.

---

### 2. Main-Relative Changes & Dependency Chain

| File Path | Nature | Main-Relative Role / Justification |
|---|---|---|
| `analyst_dashboard/data/alpaca_fetcher.py` | Modified | Adds `fetch_historical_bars` method for single-symbol daily OHLCV retrieval with tape feed, adjustment, and pagination support. |
| `analyst_dashboard/data/market_db.py` | Modified | Adds `auto_init=False` constructor parameter; enforces strict date bounds on `replace_existing_scope`; removes unbounded whole-symbol deletion fallback. |
| `analyst_dashboard/data/market_evidence.py` | Modified | Adds `Provider.ALPACA` and `AdjustmentState.SPLIT_ADJUSTED_ONLY` enum values. |
| `analyst_dashboard/universe/builder.py` | Modified | Adds `evaluate_candidate_eligibility` alias method. |
| `scripts/ingest_alpaca_historical.py` | Added | Bounded Alpaca historical SIP ingestion adapter and CLI (Defect A prefix coverage, calendar validation, read-only inspection, safe writer). |
| `scripts/apply_registration_plan_wrapper.py` | Modified | Production-ready registration wrapper: verified pre-apply backup, recovery triage diagnosis, and semantic applied-receipt audit retention (Defect B). |
| `scripts/universe/build_candidate_manifest_150.py` | Added | Deterministic candidate manifest builder and input resolution script. |
| `evidence/coverage_expansion/*.json` | Added | 14 retained operational package specifications, candidate manifests, and target state exports. |
| `data/research/market_data/bounded_150_dev_market_store.db` | Added | Retained bounded market store fixture for deterministic offline testing. |
| `data/raw_feeds/2026-10-16/*` | Added | Retained directory feed snapshots for offline candidate manifest regeneration testing. |
| `data/research/reference_polygon/tickers/*.json` | Added | Retained reference ticker JSON fixtures for candidate proposal locator testing. |
| `tests/test_alpaca_historical_sip_ingestion.py` | Added | 53 unit, integration, and adversarial tests covering SIP-01..SIP-06, R1..R5, Defect A, and Writer Safeguards. |
| `tests/test_registration_execution_wrapper.py` | Modified | Expanded to 39 tests covering execution wrapper, pre-apply backup, recovery triage, and Defect B applied-receipt integrity. |
| `tests/test_bounded_150_candidate_expansion.py` | Added | 71 governance and invariant tests for the 150-candidate bounded expansion. |
| `.github/workflows/ci.yml` | Modified | Adds explicit execution step for the 8 expansion and market data test modules across Python 3.11 and 3.12. |
| `docs/reviews/sprint2-historical-sip-release/audit.md` | Added | This primary review document. |

---

### 3. Verification Evidence

#### Local Verification Suite Summary (Windows, Python 3.11)

```powershell
Set-Location C:\Users\akara\Documents\Projects\finance-sprint2-historical-sip-release

# 1. Six Expansion Modules
python -m pytest `
  tests/test_bounded_150_candidate_expansion.py `
  tests/test_identity_adoption_dry_run.py `
  tests/test_identity_persistence.py `
  tests/test_three_candidate_registration_plan.py `
  tests/test_registration_execution_wrapper.py `
  tests/test_alpaca_historical_sip_ingestion.py -q
# Result: 216 passed in 10.20s

# 2. Market Database Modules
python -m pytest `
  tests/test_market_db.py `
  tests/test_market_evidence_f13a.py -q
# Result: 14 passed in 1.13s

# Total: 230 passed, 0 failed, 0 skipped, 0 errors
```

#### Detailed Test Breakdown
- `tests/test_bounded_150_candidate_expansion.py`: **71 passed** (real vs synthetic isolation, share-class isolation, Rule R01/R02 exclusions, market data readiness accounting, proposal evidence locators)
- `tests/test_identity_adoption_dry_run.py`: **22 passed** (dry-run engine invariants, collision detection, stable ID generation)
- `tests/test_identity_persistence.py`: **18 passed** (atomic persistence, transaction rollback, idempotent replay)
- `tests/test_three_candidate_registration_plan.py`: **13 passed** (pilot candidate registration plan verification)
- `tests/test_registration_execution_wrapper.py`: **39 passed** (clean execution, pre-apply backup, duplicate rejection, triage diagnosis, Defect B applied receipt verification, staging preservation)
- `tests/test_alpaca_historical_sip_ingestion.py`: **53 passed** (SIP-01..05 invariants, feed/adjustment validation, pagination cycle detection, Defect A prefix gap rejection, complete prefix refresh, writer safeguard bounds)
- `tests/test_market_db.py`: **1 passed** (persistence retry decorator and error propagation)
- `tests/test_market_evidence_f13a.py`: **13 passed** (evidence provenance schema, observation precision, immutability)

---

### 4. Operational Separation & Safety Boundaries

The release candidate strictly adheres to operational containment:
- **No Production Database Mutation:** Zero migrations or writes to production databases.
- **No Identity Plan Application:** `package_a_approval_manifest_template.json` remains unsigned. No identity application script is invoked against production targets.
- **No Live Provider Calls:** All tests and verifications execute against hermetic local SQLite fixtures and mock fetchers.
- **No Universe Publication:** Package C publication specifications remain unapproved. Default radar scanner population is unmodified.
- **No Automatic Activation:** Ingestion, identity adoption, and scanner pipelines remain dormant until explicit operational invocation.

---

### 5. Remaining Blockers & Next Actions

1. **Remote CI Quality Gate:** Confirm that the GitHub Actions CI pipeline on the pushed release branch finishes with green status across both Python 3.11 and 3.12.
2. **Guarded Merge Review:** Submit PR for independent code review and approval.
3. **Operational Phase 3 (Post-Merge):** Formal operator signature on `package_a_approval_manifest.json` before any container execution.
