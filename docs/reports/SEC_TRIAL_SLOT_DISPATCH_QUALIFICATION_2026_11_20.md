# Issue #165: recurring trial slot dispatch

Base: `b937706e1ff2391cbcb61b359ffd981d98827c0a` (PR #311).

## Demonstrated gap and resulting behavior

The deployed worker consumed queued operator requests but did not enqueue due
trial slots. An eligibility predicate alone was not a recurring dispatcher.
The new worker poll enqueues an eligible NVDA/AAPL slot before consuming work.
It requires an unexpired release-bound authorization, verified exclusive owner
lock, explicit trial dates, and readable durable progress. No egress or execution
budget is consumed by dispatch itself. The existing pipeline still reserves eight
requests before execution, performs ingestion only, and rechecks eligibility.

Queue identity hashes the stable trial ID and NY slot ID. Enqueue is serialized
and deduplicates all statuses, including failed/completed jobs, without DDL.
Failed slots are not retried or reset. A delayed delivery cannot run in a later
slot; slot identity is rechecked under the pipeline owner lock before reservation.
The independent daily profile does not dispatch trial work.

## Qualification boundary

New tests cover due/ineligible dispatch, expired grants, missing owner proof,
missing dates, emergency stop, unavailable progress, restart-safe terminal
queue deduplication, delayed jobs, and slot rollover immediately before execution.
A disposable PostgreSQL test exercises a real worker poll through queue creation,
lease claiming, budget reservation, production lock SQL, PARTIAL reconciliation,
and completion. External backup/SEC operations are synthetic in that test.
No production trial delivery is claimed from offline qualification.

## Required production acceptance before activation

1. Inspect the actual deployed revision and retain DISABLED/STANDBY until ready.
2. Confirm zero queue rows in PENDING, CLAIMED, RUNNING, RETRY_ELIGIBLE and zero
   held SEC owner locks (classid 93366790, objid 1926593531, objsubid 1).
3. Bind a verified restore receipt to the exact fresh backup ID and artifact hash.
   Upload success alone is insufficient. If a matching independent restore receipt
   exists while the manifest says PENDING_VERIFICATION, reconcile that evidence
   instead of blindly repeating a restore or claiming either result.
4. Rehearse the append-only historical receipt correction offline, preserving the
   original inventory, catalog/XML source hashes, gaps, and timestamps. Attribute
   historical execution to 16549be1, not the new release. Verify idempotency and
   canonical hash/readback before any production insertion.
5. Set ARX_TRIAL_ID plus explicit ARX_TRIAL_START_DATE and ARX_TRIAL_END_DATE in
   America/New_York. The Nov 20/23/24 cohort requires start 2026-11-20 and end
   2026-11-24; do not enable default dates silently. If slots were missed, replan
   explicitly; no force/catch-up or resetting trial counters.
6. For that cohort the final 20:30 EST slot starts Nov 25 at 01:30Z and its one-hour
   admission window ends 02:30Z. Allow the bounded 60-second run and cleanup
   before teardown, for example 02:35Z (03:35 Europe/Berlin), without extending
   admission beyond the final NY date. A 00:00Z cutoff is premature.
7. Renew release-bound authorization for at most 24 hours per grant, retaining
   trial identity and quotas. A three-day grant is invalid. Ingestion only remains
   mandatory; snapshot publication, Radar scanning, and trading stay held.

## Stop semantics and diagnostics

Changing Railway variables is a platform configuration/deployment action. It
cannot mutate os.environ in an already-running Python process. Do not describe
it as a measured five/ten-second in-process pause. Distinguish grant expiry,
cooperative cancellation, and actual container termination. Use the documented
platform restart/redeploy procedure with disabled role/authorization persisted
before restarting, and measure lock release and post-stop network behavior.
No production stop-latency guarantee is established by these offline tests.

Run diagnostics inside the authenticated worker environment, importing
get_database_url from analyst_dashboard.smart_money.storage.config and the
repository from analyst_dashboard.smart_money.storage.postgres_repository.
Retain timestamped SQL evidence and compare baseline content, not just counts.

Deployment qualification, production acceptance, and operational activation are
separate verdicts. Any missing required evidence is NOT_CHECKED and activation
remains NO-GO. This change does not grant a live execution window.
