# Package 1 Acquisition Boundary Review & Checks A–E Audit
**Sprint 2 / Foundation Milestone: Listing Acquisition & Source-Population Accounting**

---

### 1. Context, Objectives & Scope

- **Workstream**: Track E / Sprint 2 Package 1 (`S2-02`) — Listing Acquisition & Source-Population Accounting
- **Repository**: `daakara/finance`
- **Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint-2-listing-acquisition`
- **Branch**: `feat/sprint-2-listing-acquisition`
- **Base Commit**: `caf7c79c5163539a35b9d5779e81568934c54464` (`origin/main`, PR #9 squash merge)
- **Base Tree**: `f18f8fe7624c329c3bc0e1710dbb08edbcf56b67`
- **Status**: `BOUNDARY_VERIFIED_LOCAL`

This document records the formal boundary audit and resolution of Checks A through E for Package 1, reconciles the commit identity history between `bf0ca7a` and `dfb7b76`, and verifies that all non-inference, read-safety, and isolation boundaries are maintained prior to remote push and pull request creation.

---

### 2. Reconciliation of Commit Identity: `bf0ca7a` vs `dfb7b76`

#### 2.1 Commit Provenance & Reflog Evidence
Inspection of the local worktree reflog confirms the exact relationship between the two commit IDs:

```text
bf0ca7a HEAD@{0}: commit (amend): feat(universe): implement bounded listing directory acquisition and accounting (Package 1)
dfb7b76 HEAD@{1}: commit: feat(universe): implement bounded listing directory acquisition and accounting (Package 1)
caf7c79 HEAD@{2}: reset: moving to HEAD
```

- **Commit `dfb7b76`** (`dfb7b76ab034de3f842ace6fc517d36c539b2429`, Tree: `94829d76efb37f1c189f15976a7ddcd428b226a4`):  
  Initial local implementation commit of Package 1. Included the core bounded fetcher (`fetcher.py`), atomic storage engine (`storage.py`), typed models (`models.py`), acquisition runner (`runner.py`), diagnostic CLI (`acquire_listing_directories.py`), kickoff documentation (`SPRINT_2_KICKOFF.md`), delivery checklist update, and 13 unit tests.
- **Commit `bf0ca7a`** (`bf0ca7a6b32832f2bf30e56770b2a8ace839283d`, Tree: `3594b606ef0a165c29f512acba68e85d27fe5dce`):  
  Amended local commit created via `git commit --amend` prior to any remote push. Incorporated:
  1. `analyst_dashboard/universe/acquisition/report.py` for machine-readable JSON and human-readable Markdown accounting report generation.
  2. Diagnostic CLI flags `--save-report` and `--offline-dir`.
  3. Added Unit Test 14 (`test_dual_report_generation`).
  4. Updated `PACKAGE_1_ACQUISITION_IMPLEMENTATION_NOTE.md` and `SPRINT_2_KICKOFF.md` reflecting 14 tests passing under `tier2c` (116 total provenance tests).

#### 2.2 Conclusive Reconciliation Determination
`bf0ca7a` was an in-place local amendment of `dfb7b76` executed in the private worktree before remote publishing. It represents the complete, canonical Package 1 implementation baseline on branch `feat/sprint-2-listing-acquisition`. All subsequent boundary verification and follow-up commits build directly on top of `bf0ca7a`.

---

### 3. Formal Checks A through E Audit

| Check | Boundary Invariant | Verification Status | Operational Evidence & Guard |
|:---|:---|:---:|:---|
| **Check A** | **Surrogate Key Non-Synthesis & User-Agent Boundaries** | **PASSED** | Zero production primary keys, synthetic IDs, or `CIK_{cik}_{symbol}` surrogate keys are manufactured. Parsed rows preserve native source fields (`cik`, `ticker`, `Symbol`, `Exchange`, `CQS Symbol`). SEC EDGAR enforces fail-closed User-Agent contact validation per SEC Fair Access Policy. *Clarification*: Email-format checking validates header syntax only; it does NOT verify contact ownership, mailbox deliverability, or authorized domain control. Package 2 (`S2-03`) remains the dedicated boundary for identity resolution. |
| **Check B** | **OpenFIGI Decoupling & Conflict Taxonomy** | **PASSED** | Listing directory acquisition feeds are strictly decoupled from OpenFIGI network endpoints (`CANONICAL_SOURCES` contains zero OpenFIGI entries). Cross-source matching preserves OpenFIGI multi-match results as ambiguous (`is_ambiguous = True`) with zero automatic selection. 471 cross-feed venue conflicts are analyzed and documented as exchange taxonomy umbrella groupings (e.g. NYSE American categorized under parent NYSE in SEC EDGAR) rather than entity collisions. *Equivalence Protection*: Disagreements are NEVER cleared through unsupported equivalence mappings (particularly TXSE vs FINRA ADF / 'F'); original raw venue values are strictly preserved. SEC parser distinguishes missing exchange attributes (168 rows) from syntax corruption (0 syntax errors). |
| **Check C** | **Source Completeness Accounting & Coverage Gates** | **PASSED** | System outputs explicitly state: `PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED` and `FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED`. Directory presence is explicitly acknowledged as not proving active trading status, primary listing verification, or total market census. Accounting reports surface all 5 core coverage limitations and 4 unresolved policy decisions. |
| **Check D** | **Read-Safety & Transport Invariants** | **PASSED** | Pure on-demand, explicit invocation. Zero background daemons, zero cron jobs, zero Celery/worker tasks, zero API route decorators/listeners, and zero database writes. Pre-request redirect scheme inspection halts with `TLS_ERROR` *before* issuing any HTTP request to an insecure target. *Retry-After Deferral*: `Retry-After` handling NEVER retries earlier than requested; if the requested wait exceeds the execution wait budget (`config.max_retry_wait_seconds`), the fetcher stops and defers the source immediately without early retry. |
| **Check E** | **Offline/Isolated Execution & Overwrite Protection** | **PASSED** | Offline replay via `offline_bytes` operates 100% network free. Filesystem persistence uses temporary file staging and atomic replace (`os.replace`) for individual files. Safe overwrite protection guarantees that failed or timed-out attempts route to isolated `quarantine/` logs and NEVER overwrite accepted historical artifacts. Strict path traversal sanitization prevents malicious or relative source IDs from escaping `base_dir`. |

---

### 4. Technical Details of Checks A–E Implementation & Tests

#### 4.1 Check A: Identity Non-Synthesis & SEC Fair Access Enforcement
- **Non-Synthesis Invariant**: In `analyst_dashboard/universe/acquisition/models.py` and `analyst_dashboard/universe/adapters/models.py`, `ParsedSourceRow` records contain only raw values extracted directly from source payloads (`raw_values: dict[str, Any]`). No `security_id` or synthetic key is assigned.
- **SEC Fair Access Invariant**: `ListingSourceFetcher.fetch()` inspects configured headers for SEC EDGAR sources. If the User-Agent does not declare an administrative contact email (`@`), the fetch fails closed immediately with `AcquisitionOutcome.FAILED`, preventing automated unauthenticated crawling.
- **Contact Ownership Clarification**: Email-format checking validates header syntax only per SEC Fair Access Policy; it does NOT verify contact ownership, mailbox deliverability, or authorized domain control.
- **Automated Regression**: Verified by `test_check_a_surrogate_key_non_synthesis`, `test_sec_edgar_requires_explicit_contact_user_agent`, and `test_user_agent_email_format_validation_does_not_verify_contact_ownership`.

#### 4.2 Check B: OpenFIGI Decoupling & Conflict Taxonomy Analysis
- **Decoupling Invariant**: `CANONICAL_SOURCES` in `config.py` contains only:
  - `sec_company_tickers_exchange`
  - `nasdaq_listed_directory`
  - `other_listed_directory`
  OpenFIGI mapping is not invoked during listing directory acquisition.
- **Ambiguity Invariant**: When multi-feed matching is executed via `match_sources_with_accounting()`, `ParsedOpenFIGIResult` with status `MULTI_MATCH` leaves `selected_figi = None`.
- **Venue Conflict Taxonomy (471 Observations)**:
  - 275 NYSE vs AMEX: NYSE American securities categorized under parent "NYSE" umbrella in SEC EDGAR.
  - 144 NYSE vs ARCA: NYSE Arca securities categorized under parent "NYSE" umbrella in SEC EDGAR.
  - 44 CBOE vs BATS: Cboe BZX exchange securities ("CBOE" in SEC EDGAR vs "Z"/BATS in Nasdaq Trader).
  - 8 TXSE vs F: Texas Stock Exchange announcements vs FINRA ADF.
  These 471 observations reflect venue hierarchy taxonomy differences across publishers, not ticker collisions or corrupt records.
- **Unsupported Equivalence Rejection**: Disagreements are NEVER cleared through unsupported equivalence mappings. Original raw venue values are preserved exactly (`sec_exchange='TXSE'`, `venue_exchange='F'`), and the conflict remains explicitly cataloged.
- **Attribute vs Syntax Distinction**: The SEC EDGAR parser distinguishes rows that are structurally valid but missing the exchange attribute (168 rows, where exchange is null/empty) from structural syntax corruption (0 rows).
- **Automated Regression**: Verified by `test_check_b_openfigi_isolation_and_multimatch_ambiguity_protection`, `test_sec_diagnostic_row_categorization`, and `test_venue_conflict_preserves_raw_values_without_unsupported_txse_equivalence`.

#### 4.3 Check C: Completeness Truthfulness & Coverage Gates
- **Accounting Invariant**: Every `SourceAccountingReport` generated by `ListingAcquisitionRunner` or `save_accounting_reports()` explicitly affirms:
  ```json
  "production_population_completeness": "NOT_ESTABLISHED",
  "full_us_listed_assessment": "NOT_ESTABLISHED"
  ```
- **Standard Disclosures**: Reports enumerate standard coverage limitations:
  1. Directory presence does not establish global instrument primary listing or trading suspension status.
  2. SEC `company_tickers_exchange.json` is issuer-level (CIK-based); multiple share classes share one CIK.
  3. OTC and unlisted securities are excluded from exchange directory feeds.
  4. Foreign issuers listed on US exchanges (ADRs, ordinary shares) have distinct regulatory reporting requirements.
  5. Market coverage denominator across full US-listed population is NOT_ESTABLISHED by directory join alone.
- **Automated Regression**: Verified by `test_check_c_source_completeness_accounting_truthfulness`.

#### 4.4 Check D: Read-Safety & Pre-Request Redirect Inspection
- **Read-Safety Invariant**: All acquisition modules are synchronous and callable on-demand. Zero background workers, crons, or web listeners exist. Zero imports of application database models or stores occur.
- **Pre-Request TLS Inspection**: `ListingSourceFetcher.fetch()` inspects redirect location headers iteratively before issuing the redirected request. If a redirect targets an insecure `http://` URL while `enforce_https = True`, the fetcher terminates immediately with `AcquisitionOutcome.TLS_ERROR` *without* issuing any request to the insecure HTTP endpoint.
- **Retry-After Wait Budget Deferral**: `fetcher.py` parses integer seconds or RFC HTTP date from `Retry-After`. It NEVER retries earlier than requested. If the requested delay exceeds the execution wait budget (`config.max_retry_wait_seconds`), the source is stopped and deferred immediately with `HTTP_ERROR` without early retries.
- **Automated Regression**: Verified by `test_check_d_read_safety_and_non_triggering_guarantees`, `test_redirect_insecure_downgrade_detected`, `test_retry_after_never_retries_earlier_than_requested_and_defers_when_budget_exceeded`, and `test_architectural_boundary_invariants`.

#### 4.5 Check E: Offline Replay, Atomic Storage & Path Sanitization
- **Offline Replay**: `ListingAcquisitionRunner.run()` accepts `offline_bytes: Mapping[str, bytes]`. When provided, network calls are completely bypassed, allowing 100% offline unit and integration testing.
- **Atomic Storage**: `ArtifactStorage.save_artifact()` writes raw payload bytes to a `.tmp` file and commits via atomic `replace()`. Accepted artifacts and metadata sidecars are updated together.
- **Quarantine Invariant**: Any failed, timed-out, or malformed attempt writes failure metadata exclusively to `quarantine/failed_{timestamp}_{source_id}.meta.json` and leaves any previously accepted artifact untouched.
- **Path Sanitization**: `_determine_filename()` sanitizes `source_id` using `Path(source_id).name` and alphanumeric/underscore/hyphen filtering. It resolves paths against `base_dir` and fails closed if any attempt tries to escape `base_dir`.
- **Automated Regression**: Verified by `test_check_e_offline_replay_and_quarantine_integrity`, `test_safe_artifact_write_invariant_never_overwrites_accepted`, and `test_artifact_storage_path_traversal_sanitization`.

---

### 5. Test Suite Verification Summary

- **Acquisition & Boundary Suite (`tests/test_universe_listing_acquisition.py`)**: 25 tests passing.
- **Offline Adapters Suite (`tests/test_universe_offline_adapters.py`)**: 34 tests passing.
- **Universe Validator Suite (`tests/test_universe_validator.py`)**: 68 tests passing.
- **Full Provenance & Data Gate (`tier2c`)**: **127 passed in 1.99s (0 failures, 0 regressions)**.

---

### 6. Operational Status & Boundaries

```ini
SPRINT_2_STATUS = IN_PROGRESS
PACKAGE_1_STATUS = BOUNDARY_VERIFIED_LOCAL
SOURCE_ACQUISITION_COVERAGE = BOUNDED_EXCHANGE_FEEDS_PARSED
CHECKS_A_THROUGH_E = ALL_PASSED
PRODUCTION_POPULATION_COMPLETENESS = NOT_ESTABLISHED
FULL_US_LISTED_ASSESSMENT = NOT_ESTABLISHED

PACKAGE_2_IMPLEMENTATION = NOT_STARTED
SURROGATE_IDENTITY_ADOPTION = PROHIBITED
REMOTE_PUSH = AUTHORIZED_PENDING_COMMIT
PR_CREATION = AUTHORIZED_PENDING_PUSH
REMOTE_CI = PENDING_PR
PRODUCTION_ACCESS = NO
PRODUCTION_INGESTION_INTEGRATION = NOT_IMPLEMENTED
UNIVERSE_BUILD = NO
UNIVERSE_PUBLICATION = NO
SCAN_EXECUTION = NO
SCHEDULER_ACTIVATION = NO
PROSPECTIVE_CAPTURE_ENABLEMENT = NO
MODEL_TUNING = FROZEN
EMPIRICAL_VALIDATION = NOT_ESTABLISHED
GATE12_ADVANCEMENT = NO
```
