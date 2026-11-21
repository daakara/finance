# PR #317 — bounded SEC ingestion rollout and teardown packet

**Status: PREPARED, NOT APPLIED.** Owner: David Akara (`@daakara`). Existing trackers: #165 / Sprint P3 #231. Candidate: draft PR #317; source SHA and hosted qualification run must be read from its final verified head. Production reference is `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`. No merge, deployment, grant, schedule enablement, live queue delivery, SEC call, production SQL write or hold change is authorized by this document.

## 1. Separate operations

Monday Nov 23's existing 08:30 Europe/Berlin read-only preflight and original AAPL/NVDA trial remain unchanged. This expanded phase begins only after that trial's independently verified teardown after `2026-11-25T02:35:00Z`. Do not deploy this candidate during an armed release-bound trial. Code qualification does not itself grant a production rollout or an ingestion exception under `HOLD_SEC_001`.

Preserve original trial ID `ARX-TRIAL-2026-NVDA-AAPL-001`, its historical reservations, charged quotas, attempts and Day 1 latency failures. Preserve frozen NVDA evidence and active snapshot `SNAP_SM_1792939989_48f30db6`. Scoring, publication, Radar, trading, UX and research holds remain independent.

## 2. Concrete proposed phase — requires fresh operator ratification

| Parameter | Proposal / binding |
| --- | --- |
| Railway project | `a339a92a-7236-4ac0-9a3a-d7ad440f9690` |
| Production environment | `11cad5e1-358e-430e-84fc-189953cff48a` |
| Worker service | `380acddd-8f0b-49cf-99db-59ecbf1a7aae` |
| Profile | `RECURRING_10_ISSUER_FORM4_INGESTION_ONLY` |
| Durable phase ID | Proposed `ARX-SEC-INGESTION-10-20261125-001`; retain across both grants and all retries. No phase-ID rotation to refund quotas. |
| Explicit NY dates | Proposed `2026-11-25,2026-11-27`; validated by the code's dissemination calendar. Reconfirm the authoritative SEC calendar and actual readiness before approval. Dates are examples for ratification, not applied production values. |
| Each delivery window | 09:15:00 inclusive to 09:20:00 exclusive America/New_York; no catch-up, weekend or unlisted-date execution. |
| Corresponding UTC / Berlin | Nov 25 and Nov 27: 14:15–14:20 UTC / 15:15–15:20 Europe/Berlin. |
| Final in-flight fence | Last approved date 09:21 NY, i.e. proposed Nov 27 14:21 UTC / 15:21 Berlin. Every expanded grant must expire by this derived fence and remain <=24 hours. |
| Allowlist | NVDA/0001045810, AAPL/0000320193, MSFT/0000789019, AMZN/0001018724, GOOGL/0001652044, META/0001326801, TSLA/0001318605, BRK-B/0001067983, JPM/0000019617, V/0001403161. No scoring-universe publication is implied. |
| Durable ceilings | 40 conservatively reserved requests/execution; one new execution/NY day; two executions / 80 charged requests/phase. Failed or ambiguous send intents consume budget. |
| Transport ceilings | <=40 requests/cycle, <=4/issuer, <=5 requests/sec, <=60 seconds through cleanup/terminal completion. Configuration may lower caps, not raise them. |
| Persistence and recovery | Atomic verified issuer units; exact source-bound resume within the same queue execution and NY day. Stale lease recovery may reclaim that request inside its original window; terminal jobs are not automatically replayed. |

The built-in daily delivery uses the existing worker and `sm_run_requests`; it creates no additional service, timer or external scheduler. A deterministic phase/date/release/date-set subject hash identifies one `EXPANDED_DAILY_FORM4` row, including terminal rows. Consumption and under-lock pipeline admission recheck the window. Up to three claims permit abandoned-lease recovery; they do not create new phase reservations or refund send intents.

## 3. Exact configuration contract — no activation block

A future deployment first remains **DISABLED / authorization false / schedule disabled**. Verify the deployed merge SHA, declared release and container SHA agree; the qualification source commit is not automatically the eventual squash-merge SHA. Verify source-tree lineage and deployment artifact binding, not just a health HTTP 200.

Only after the explicit bounded GO, the operator may reconcile these named fields in one reviewed operation:

- `ARX_RELEASE_SHA` and `ARX_CONTINUOUS_AUTH_RELEASE_SHA`: exact verified deployed release SHA.
- `ARX_CONTINUOUS_AUTH_SCOPE`: expanded ingestion-only profile above.
- `ARX_CONTINUOUS_AUTH_UNIVERSE_ID`: `UB-US-EQUITIES-20261025-001`.
- `ARX_INGESTION_PHASE_ID`: ratified stable phase ID above.
- `ARX_INGESTION_DATES`: exactly the two ratified ISO NY dates, comma separated and chronological.
- `ARX_SMART_MONEY_INGESTION_ONLY`: true, non-overridable for this profile.
- `ARX_INGESTION_SCHEDULE_ENABLED`: absent/false until GO; true only for the approved windows.
- `ARX_CONTINUOUS_AUTH_ID`: freshly issued grant ID for each authorized observation day; changing this never resets the durable phase.
- `ARX_CONTINUOUS_AUTH_ISSUED_AT`: actual UTC arming time; never backdated or replaced with a scheduled timestamp.
- `ARX_CONTINUOUS_AUTH_EXPIRES_AT`: shortest reviewed interval covering that day's window plus its <=60-second execution, bounded by <=24 hours and the derived final phase fence.
- `ARX_CONTINUOUS_INGESTION_AUTHORIZED` and `ARX_SEC_EGRESS_ROLE`: remain false/DISABLED until that bounded GO. No copy-ready OWNER activation command is issued here.

For the proposed dates, separate day-specific authorizations avoid maintaining OWNER throughout the intervening holiday. Return to deauthorized standby after each observation. Re-enable only following the second day's independent preflight. Keep the same phase, dates, scope and verified release; a release/scope/date change causes a binding mismatch and requires review, not a quota-reset workaround.

## 4. Preconditions and acceptance

Before each observation:

1. Verify final hosted qualification passed for the exact current PR candidate, including mandatory offline SEC suites with no skips. Confirm cumulative manifest digests, workflow/framework integrity, negative controls and rollback target. PR #310 is not a prerequisite to this work.
2. Verify original trial shutdown through authoritative configuration, runtime and authenticated SQL: role DISABLED/STANDBY, authorization revoked, owner lock `401007311509093371` unheld, no active/retryable queue. A log or HTTP 200 does not prove SQL state.
3. Verify deployment/declared/grant SHA parity, exact cohort/profile/phase/dates, no unexpected active job, preserved baseline/snapshot and immutable budget history. Check outstanding retries using deployed eligibility criteria, not only future retry timestamps.
4. Verify a matching successful isolated restore receipt for a qualifying backup whose source age is <=12 hours at dispatch. Upload success or a Saturday restore is insufficient; verify pre-ingestion containment as well.
5. Review actual issue/expiry times and the ratified calendar/windows. Preflight alone yields READY_FOR_BOUNDED_VALIDATION, not proven live latency or full Smart Money activation.

After each natural delivery, correlate subject key, request ID, attempts/lease, phase reservation, conservative send intents, observed actual requests including failures, issuer receipts, aggregate receipt, queue terminal state, cleanup and worker return. Distinguish no-work, ingested evidence, unresolved amendments, partial delivery, exhausted budget and missing delivery. Observe full-cycle duration and actual rate with sufficient timestamps; configured limits are not measurements. No new snapshot or scope expansion may occur.

If a mandatory gate is unavailable, label it NOT_CHECKED and block arming. A quiet no-work cycle does not prove new-filing delivery. No automatic third observation, larger cohort, longer window or publication follows success.

## 5. Teardown and rollback — operator actions, not performed here

Exact teardown prompt:

> For Railway project a339a92a-7236-4ac0-9a3a-d7ad440f9690, environment 11cad5e1-358e-430e-84fc-189953cff48a, worker 380acddd-8f0b-49cf-99db-59ecbf1a7aae, end only the approved expanded ingestion phase. Set ARX_INGESTION_SCHEDULE_ENABLED=false, ARX_CONTINUOUS_INGESTION_AUTHORIZED=false and ARX_SEC_EGRESS_ROLE=DISABLED using the authorized operational procedure. Preserve historical grants, phase IDs, counters, reservations, filings and receipts; do not issue a replacement grant or replay jobs. Observe cancellation/exit and stable STANDBY. Verify authoritative config and direct authenticated SQL show no owner lock 401007311509093371 and no active or retry-eligible queue. If a live claim remains, use the approved fenced drain/cancellation procedure; do not delete evidence or force-unlock another session. Return BLOCKED if any proof is missing. Preserve every other hold and the frozen snapshot; perform no SEC calls or publication. Report exact UTC timestamps and deployment identity.

The grant fence denies further authorized egress, but expiry alone does not prove DISABLED configuration, queue cleanup or lock release. Perform and verify teardown after each observation and by the final fence.

Software rollback target is the last independently verified safe production release (reference `2eccc64a...`, recheck before action). Disable/deauthorize first; only a separately authorized deployment may restore that software. Reconcile runtime release variables without rebinding expired grants. Retain append-only data and receipts. A production database restore is a separate quiesced recovery operation, not a routine software rollback or permission to delete committed units.

## 6. Current evidence limit

This packet was prepared offline. No fresh production SQL, Railway configuration, R2 recoverability or live SEC measurement is claimed. The original Monday monitors were not changed or duplicated. Hosted run IDs, final source SHA and any outstanding qualification findings belong in the PR's current acceptance receipt; a draft packet is not a release or activation receipt.
