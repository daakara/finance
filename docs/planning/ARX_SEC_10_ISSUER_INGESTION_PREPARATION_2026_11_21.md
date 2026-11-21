# SEC ingestion expansion — preparation and bounded proposal

Owner: David Akara (`@daakara`). Existing trackers: Issue #165 / Sprint P3 #231.
Candidate branch: `codex/sec-expansion-preparation` (Draft PR #317).
Reference source: production release `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
Disposition: **OFFLINE IMPLEMENTATION AND QUALIFICATION COMPLETE; LIVE ACTIVATION BLOCKED ON OPERATOR GO**.
This candidate branch executes offline qualification only; it creates no production grant, schedule, queue request, configuration change, SEC contact, or production write.

## Evidence obtained offline

170 tests passed on PostgreSQL 16.2:
- 5 archived-replay preparation checks (`tests/test_sec_archived_replay_preparation.py`)
- 9 bounded resumable ingestion qualification checks (`tests/test_sec_bounded_resumable_ingestion.py`)
- 84 two-issuer trial suite tests (`tests/test_smart_money_two_issuer_trial.py`)
- 72 SEC evidence-integrity checks (`tests/test_sec_ingestion_evidence_integrity.py`)

All Python socket connections and DNS requests were restricted to loopback by `sec_offline_test_support`; no SEC endpoints were contacted. PostgreSQL used disposable local databases over TCP, with application schema, transactions, triggers, and repository code.

The authentic archived AAPL fixture `tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml` has SHA-256 `eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`. It parsed into S, S, and G transactions, persisted three transactions plus raw bytes and availability evidence, replayed without duplicate rows, retained PARTIAL coverage, and rolled back all filing writes when availability persistence failed. No scanner snapshot was created.

Final offline qualification result: **170 passed, zero skipped, zero failures**.

## Resolved Implementation Blockers (Implemented on `codex/sec-expansion-preparation`)

| Source | Pre-Fix Defect / Blocker | Resolution Implemented | Qualification Status |
| --- | --- | --- | --- |
| `continuous_authorization.py` | `require_continuous_authorization` only accepted `DAILY_10_ISSUER_FORM4` or `RECURRING_2_ISSUER_FORM4_TRIAL`. | Added `SCOPE_RECURRING_10_INGESTION = "RECURRING_10_ISSUER_FORM4_INGESTION_ONLY"`. Enforces mandatory `ARX_SMART_MONEY_INGESTION_ONLY=true` and exact 10 tickers/CIKs in `_TEN_ISSUERS_CIKS`. | Verified by `test_scope_recurring_10_ingestion_authorization_enforcement`. |
| `continuous_pipeline.py` | `ingestion_only=True` capped total cycle transmissions to 8 regardless of cohort size (`max_total_ceiling = 8 if (self.ingestion_only or len(self.issuers) <= 2)`). | Decoupled cohort ceiling: 8 requests strictly reserved for `is_trial_2_issuer`; expanded cohort uses `min(len(self.issuers) * 4, 40)`. | Verified by `test_current_ten_issuer_ingestion_only_budget_covers_catalogs` (asserts 40 requests). |
| `continuous_pipeline.py` | In-memory batch accumulation across all issuers: failure on issuer 2 lost issuer 1's fetched/parsed data and repeated egress on retry. | Implemented `_execute_bounded_cohort_ingestion`: single-issuer atomic persistence unit (raw filing, availability, parsed transactions, amendments, unit receipt). Earlier progress survives later failures. | Verified by `test_late_issuer_failure_preserves_earlier_progress` & `test_crash_between_commit_and_queue_completion_survives`. |
| `postgres_repository.py` | Resumption after restart or retry lacked unit-level receipt checking, causing duplicate requests. | Implemented `get_completed_unit_receipt` querying `sm_reconciliation_receipts`. Resumes completed units with zero repeated wire egress. | Verified by `test_restart_and_resumption_skips_duplicate_requests`. |
| Zero Schema Migrations | Adding new tables or altering schema carries production DDL risk. | Confirmed zero schema migrations required: `sm_reconciliation_receipts.failure_reasons` is JSONB, natively housing unit status, per-ticker outcomes, and accession inventory. | Schema unchanged (`sm_schema_migrations` intact). |
| `continuous_pipeline.py` | Snapshot publication was possible under misconfigured 10-issuer runs. | Hardened `_step8_publish_snapshot` to fail closed with `SNAPSHOT_PUBLICATION_PROHIBITED_UNDER_INGESTION_ONLY` if called under ingestion-only profiles. | Verified by `test_ingestion_only_forbids_snapshot_publication_and_retention`. |
| `continuous_pipeline.py` | Retention pruning was only deferred for `is_trial_2_issuer`. | Updated `_step10_prune_retention` to defer retention when `self.ingestion_only` is true, preventing destructive R2 calls on ingestion paths. | Verified by `test_ingestion_only_forbids_snapshot_publication_and_retention`. |
| `continuous_pipeline.py` | Queue completion reported `COMPLETED` before cleanup duration was evaluated against the cycle deadline. | Moved queue status evaluation to after cleanup: sets `DEGRADED_DEADLINE_EXCEEDED` if entire worker execution overshoots deadline. | Verified in `execute_cycle` implementation. |

## Authoritative Live Acceptance Packet (For Future Operator GO)

This envelope governs any authorized future live trial of the expanded 10-issuer cohort:

1. **Exact Governed Cohort Allowlist**:
   - NVDA (`0001045810`), AAPL (`0000320193`), MSFT (`0000789019`), AMZN (`0001018724`), GOOGL (`0001652044`), META (`0001326801`), TSLA (`0001318605`), BRK-B (`0001067983`), JPM (`0000019617`), V (`0001403161`).
   - Any ticker or CIK deviation fails closed immediately with `CONTINUOUS_AUTH_ISSUER_MISMATCH`.
2. **Ceilings & Budgets**:
   - `max_total_requests`: 40 per cycle.
   - `max_per_issuer_requests`: 4 per cycle.
   - Strict rate limiter: <= 5.0 requests/second.
   - Max cycle duration: <= 60 seconds for the **entire** worker execution through cleanup.
   - Observation schedule: One natural business-day cycle per day; review after two completed cycles. Zero automatic weekend runs, catch-ups, or replays.
3. **Fail-Closed Stop Conditions**:
   - Immediate abort and transition to `DISABLED/STANDBY` upon:
     - Any signature, grant ID, universe ID, or release SHA mismatch.
     - Loss or contention of PostgreSQL owner lock `401007311509093371`.
     - Cycle execution duration exceeding monotonic deadline (> 60s).
     - Missing qualifying database backup within the <= 12-hour age ceiling.
     - Any attempt to execute snapshot publication or R2 retention pruning.
4. **Rollback Procedure**:
   - Immediately revert `ARX_SEC_EGRESS_ROLE=DISABLED`.
   - Ensure `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`.
   - Verify unheld owner advisory lock via authenticated direct SQL.
   - Verify zero active or retry-eligible queue rows in `sm_run_requests`.
5. **Preservation of Baseline & Historical Grants**:
   - Frozen NVDA baseline and active snapshot `SNAP_SM_1792939989_48f30db6` remain untouchable.
   - `ARX-TRIAL-2026-NVDA-AAPL-001` quotas (Day 1: 32 charged, 64 remaining of 96) and expiry `2026-11-25T02:35:00Z` remain protected.
   - All historical reconciliation receipts and audit rows preserved without mutation or deletion.

## Explicit Status of Live Verification Gates

The following operational gates cannot be verified by offline test harnesses and are recorded as:

- **Production Advisory Lock (401007311509093371)**: `NOT_CHECKED` (offline qualification uses local disposable database; production SQL is prohibited).
- **Production Queue States (`sm_run_requests`)**: `NOT_CHECKED` (production SQL is prohibited).
- **Production Backup Age & Isolated Restore Verification**: `NOT_CHECKED` (live backup storage is not accessed).
- **Production Release SHA Alignment in Railway**: `NOT_CHECKED` (candidate PR #317 is unmerged and undeployed).
- **Production Natural-Cycle Ingestion Latency**: `NOT_CHECKED` (local test execution does not represent live SEC network latency).

The [corrected Day 2 readiness assessment](../reports/ARX_ISSUE165_DAY2_READINESS_ASSESSMENT_CORRECTED_2026_11_21.md) separates dated Saturday evidence, Monday prerequisite checks and the first post-fix natural-cycle acceptance. It also records conservative charged quotas and the existing read-only monitor boundaries. It creates no execution authority.

## Reproduction

Install repository test dependencies plus `pytest`, `pgserver`, and `psycopg2-binary`. From repository root:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=tools/qualification:. ARX_REQUIRE_LOCAL_POSTGRES_TESTS=true \
  python -m pytest -p sec_offline_test_support \
  tests/test_sec_archived_replay_preparation.py \
  tests/test_sec_bounded_resumable_ingestion.py \
  tests/test_smart_money_two_issuer_trial.py \
  tests/test_sec_ingestion_evidence_integrity.py -q --tb=short
```
