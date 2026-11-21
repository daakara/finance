# SEC ingestion expansion — candidate and acceptance boundaries

Owner: David Akara (`@daakara`). Trackers: Issue #165 / Sprint P3 #231.
Candidate: existing draft PR #317, `codex/sec-expansion-preparation`.
Reference production source: `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
Disposition: **OFFLINE REMEDIATION; LIVE ACTIVATION AND RELEASE QUALIFICATION PENDING**.

No production grant, schedule, queue request, SEC contact, SQL write, configuration, deployment, storage deletion or hold was changed by this work. Candidate implementation is not permission to operate.

## Independent findings and correction

The earlier 170 passing offline tests did not establish that the new default unit path preserved the original commit fences, exact resume identity, partial queue semantics or durable expanded quotas. Independent probes against `c4b53a0c` demonstrated a commit after deadline expiry, reuse of a receipt from a different trial, and a full-cycle SUCCESS result despite failed issuer delivery.

The candidate now:

- Reuses `_parse_and_reconcile_transaction` for issuer units instead of duplicating parser/persistence/amendment code. A shared pre-commit callback checks the monotonic deadline, cancellation/revocation, live grant, exact owner-lock session and active queue lease immediately before commit. No-work and cohort receipts use the same fence. Remote commit latency is not claimed to be interruptible.
- Binds resumed units conjunctively to the durable execution ID, trial/slot, issuer/CIK, cohort/profile/phase/release binding hash, release SHA and fixed watermark. Resumption errors fail closed. Sealed unit metadata is checked against raw XML hashes, availability records and parsed transaction counts. Unresolved amendment gaps and accession counts survive resume. Old unbound receipts are not promoted into verified resume evidence.
- Preserves PARTIAL worker results. Partial delivery leaves queue status FAILED with `PARTIAL_INGESTION`; deadline overruns produce `DEGRADED_DEADLINE_EXCEEDED` in the worker result and FAILED queue status with the deadline failure code. Terminal transition failures propagate. An owned terminal row is reconciled to FAILED if terminal database I/O itself crosses the deadline.
- Preserves the original publication path for non-ingestion-only operation. Ingestion-only publication and remote retention remain prohibited.

## Expanded durable admission contract

The expanded scope is `RECURRING_10_ISSUER_FORM4_INGESTION_ONLY`. It requires mandatory ingestion-only behavior and exact ticker/CIK bindings. The separately approved phase must supply `ARX_INGESTION_PHASE_ID` (8–128 alphanumeric/underscore/hyphen characters, beginning with an alphanumeric). This phase ID persists across grant renewal; it is not the grant ID and must not be changed to evade limits.

Exact cohort:

| Ticker | CIK |
| --- | --- |
| NVDA | 0001045810 |
| AAPL | 0000320193 |
| MSFT | 0000789019 |
| AMZN | 0001018724 |
| GOOGL | 0001652044 |
| META | 0001326801 |
| TSLA | 0001318605 |
| BRK-B | 0001067983 |
| JPM | 0000019617 |
| V | 0001403161 |

The expanded profile reserves 40 requests for each durable queue execution before egress, at most one new execution per New York business day and two executions / 80 charged requests per phase. Configuration can lower the per-cycle transport cap. Each issuer remains capped at four requests; guarded transport remains <=5 requests/second; whole-cycle deadline remains <=60 seconds.

Reservations and conservative send intents use the existing append-only audit table in the separate `EXPANDED_INGESTION_BUDGET` namespace. No DDL is introduced. Transaction-scoped locks serialize phase admission. A send intent is committed immediately before a guarded wire attempt, including attempts that fail; it is **not proof of a successful or actually transmitted HTTP request**. A crash between intent and transmission conservatively consumes that budget. Restart restores total/per-issuer intent counts and the original watermark rather than refunding them. Changing the grant ID does not reset the phase.

Resume is limited to the same durable queue request and New York business day. A resumed request cannot silently become another day's budget. Completed units are not fetched again in that execution. A later day's separately charged execution still checks catalogs, while existing accessions prevent duplicate document persistence/download. A partial terminal failure requires an explicitly governed recovery decision; this candidate does not automatically replay terminal jobs.

The candidate now has an explicitly gated daily delivery path through the existing queue. It requires two approved post-trial NY dates, a 09:15–09:20 NY window and ARX_INGESTION_SCHEDULE_ENABLED=true in addition to live authorization/ownership. It is disabled by default and not enabled in production. The [rollout/teardown packet](../releases/ARX_SEC_INGESTION_10_ROLLOUT_PACKET_317.md) specifies the bounded proposal and remaining live gates. These expanded controls do not enlarge, rename, refund or reset `ARX-TRIAL-2026-NVDA-AAPL-001`. Its AAPL/NVDA scope, 8/32/96 limits and <=12-cycle ceiling remain separate. Day 1's reported 32 charged / 64 remaining is historical and requires fresh SQL reconciliation before dispatch.

## Offline evidence and limitations

Previous resume-remediation run: **222 passed, zero failures, zero skips (78.53s)** on disposable PostgreSQL 16.2. Suites: archived replay 5, bounded ingestion 9, new resume safety 22, original two-issuer trial 84, evidence integrity 72, and continuous ingestion 30. Existing warnings concern test tooling and deprecated UTC datetime calls, not failed gates. A preceding rerun hit local tmpfs capacity during initdb (192 passed / 30 setup errors); the final run placed disposable PostgreSQL data on a workspace volume with sufficient space, without relaxing tests or application guards. This is offline regression evidence, not hosted release qualification or live acceptance.

The existing authentic archived AAPL fixture is accession `0001140361-26-044773`, SHA-256 `eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`. It supplies three transactions (S, S, G). Other adversarial filing inputs are explicitly synthetic. Tests use real disposable PostgreSQL 16.2 transactions, triggers, receipts, leases and queue recovery; Python external connections/DNS are denied by the offline plugin. Network delivery and remote backups are simulated; local test elapsed time is not production latency evidence.

Delivery/qualification extension: **250 passed, zero failures, zero skips (71.98s)**, including 20 explicit-date/window/deduplication/delivery-fence tests and eight negative/positive release-gate controls. The mandatory RQ-03 SEC subprocess separately requires all 242 offline SEC checks to pass with no skips. Hosted evidence for the final registered candidate remains pending until recorded.

The new safety suite covers deadline/owner/grant/revocation failure, expiry during receipt persistence, exact resume binding, lookup errors, unresolved gaps, partial terminal outcome, durable phase/issuer quotas, failed wire attempts and terminal DB overrun. Its process test actually terminates a worker process after AAPL commits and before queue completion. A second fresh worker claims the lease-recovered request, with a different grant ID, skips AAPL fetch, preserves the reservation/send counts and completes the queue. This supersedes the earlier test that merely raised and caught an exception after helper return.

Remaining limitations: issuer-level atomicity deliberately retains unresolved amendment dependencies. It does not claim filing-level durable downloads within a failed issuer fetch, market-wide capacity, adaptive scheduling, complete historical coverage, production role/config restore parity or live source freshness. Historical coverage remains PARTIAL. Broader engineering should address a demonstrated need rather than silently weaken these limits.

Reproduce with repository dependencies plus pytest, pgserver and psycopg2-binary:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=tests:tools/qualification:. ARX_REQUIRE_LOCAL_POSTGRES_TESTS=true \
  python -m pytest -p sec_offline_test_support \
  tests/test_sec_archived_replay_preparation.py \
  tests/test_sec_bounded_resumable_ingestion.py \
  tests/test_sec_resume_safety_regressions.py \
  tests/test_sec_expanded_delivery.py \
  tests/test_smart_money_two_issuer_trial.py \
  tests/test_sec_ingestion_evidence_integrity.py \
  tests/test_smart_money_continuous_ingestion.py -q --tb=short
```

## Live gates — NOT_CHECKED by offline qualification

Production advisory lock `401007311509093371`, active/retryable queue, current authorization, deployed release parity, qualifying backup/isolated restore freshness, observed transmission rate and post-fix natural-cycle latency are **NOT_CHECKED** here. Authenticated SQL is required for SQL-only gates. The [corrected Day 2 assessment](../reports/ARX_ISSUE165_DAY2_READINESS_ASSESSMENT_CORRECTED_2026_11_21.md) remains the dated preflight guidance; Saturday evidence is not Monday clearance.

Before deployment or live expansion, independently qualify the cumulative PR (including hosted execution of the new offline suites), migrations (none currently), configuration delta, exact scope/phase/schedule, stop conditions and source-bound operator GO. Preserve active snapshot `SNAP_SM_1792939989_48f30db6`, the frozen NVDA baseline, all historical receipts and every operational hold. Ingestion does not activate scoring, Radar, snapshot publication or trading.

Existing-trial grants retain their trial ID and quotas, use actual issuance and the verified release, remain <=24 hours, and expire no later than `2026-11-25T02:35:00Z`. Do not rely on unenforced `ARX_TRIAL_SHUTDOWN_AT`. On a mandatory gate violation, stop further work and use the approved operator teardown: DISABLED/STANDBY, authorization false, no active/retryable queue and owner lock unheld, independently verified. This candidate's failed result is not proof that production configuration was automatically disabled. No automatic rollout or hold change is authorized by these tests.
