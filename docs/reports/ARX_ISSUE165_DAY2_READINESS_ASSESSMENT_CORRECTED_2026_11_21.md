# Issue #165 — corrected Day 2 readiness assessment

**Trial:** `ARX-TRIAL-2026-NVDA-AAPL-001`  
**Disposition:** `SATURDAY_BASELINE_RECORDED — MONDAY_PREFLIGHT_PENDING`  
**Source assessment timestamp:** `2026-11-21T15:26:00Z` / 16:26 Europe/Berlin  
**Correction date:** 2026-11-21  
**Scope:** Correct the operator-supplied assessment; no fresh production audit, activation or SQL verification is claimed.

This supersedes the supplied receipt's assertions that all prerequisites passed, Day 2 duration compliance is established, or a Monday grant is ready to apply. The original Mac artifact was not accessible from this workspace; this is a corrected counterpart, not a claim that the original file was edited.

## 1. Evidence and gate disposition

| Gate | Evidence in the supplied Saturday assessment | Correct disposition |
| --- | --- | --- |
| Deployed release parity | Worker `51e051dc`, web `359667dc`, backup `ff215425` reported on `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`; health HTTP 200. | **REPORTED PASS AT SOURCE TIMESTAMP.** Current runtime and Monday parity are NOT_CHECKED by this correction. HTTP 200 is not database evidence. |
| Authorization, lock and queue | Role DISABLED; authorization false; historical Day 1 grant expired; authenticated SQL reportedly found owner lock unheld and no active/retryable queue. | **REPORTED CONTAINMENT PASS AT SOURCE TIMESTAMP.** Monday needs fresh authoritative authorization and SQL checks; logs cannot substitute for SQL. |
| Day 1 accounting and latency | Four slots, eight actual transmissions, no new eligible filings observed, four PARTIAL receipts; three complete worker cycles exceeded 60 seconds. | **HISTORICAL LATENCY FAILURES PRESERVED.** No-work is distinct from ingestion of new filings and does not certify complete historical coverage. |
| Deployed retention remediation | PR #315 defers Step 10 retention during the two-issuer trial. Source inspection confirms the early return. | **IMPLEMENTED REMEDIATION; POST-FIX LIVE LATENCY NOT_CHECKED.** A fast cleanup test does not prove the full worker cycle completes within 60 seconds. |
| Recoverable backup | Reported restore of the 06:03 UTC Saturday backup; a newer 12:04 UTC upload lacks matching verified restore evidence in this receipt. | **REPORTED PASS FOR SATURDAY ONLY.** Neither backup establishes Monday readiness without qualifying fresh backup/restore evidence. |
| Scope, baseline, snapshot and holds | AAPL/NVDA ingestion-only, PARTIAL coverage, frozen historical NVDA baseline and active snapshot `SNAP_SM_1792939989_48f30db6` reportedly preserved. | **REPORTED PASS AT SOURCE TIMESTAMP.** Recheck Monday; ingestion readiness is separate from scoring/publication/trading readiness. |

No production SQL, current authorization readback, R2 download/restore or SEC requests were performed for this correction. Production-only gates are NOT_CHECKED here; dated operator evidence is retained with its original attribution.

## 2. Accounting and timing correction

Keep these Day 1 quantities separate:

- Executed/reserved slots: 4.
- Requests reserved: 8 per slot, 32 total.
- Actual transmissions reported: 2 per slot, 8 total (4 NVDA, 4 AAPL).
- Conservative charged trial budget: 32, **not 8**. Source `PostgresSmartMoneyRepository.get_trial_progress` takes the maximum reserved/transmitted charge per slot; reservation and completion do not double-charge that slot.
- Conditional remaining capacity, if fresh SQL confirms no additional attempts: 8 of 12 cycles and 64 of 96 charged requests. The arithmetic difference of 88 actual-wire requests is not usable durable admission headroom.

Preserve limits: <=4 cycles/day, <=12/trial, <=8 transmissions/cycle, <=4/issuer/cycle, <=32 charged requests/day, <=96 charged/trial, <=5 requests/second and <=60 seconds for the complete worker execution. Renewals and restarts must not reset accounting.

Earlier corrected operator ledger reported approximately 59.9s, 70.1s, 76.1s and 77.0s for the four complete worker cycles. The supplied assessment uses different precise values. This correction does not invent a new precision or recompute elapsed time without the original timestamp records. Use the original claim/start-to-worker-return ledger when reconciling exact values. Slots 2–4 remain historical failures regardless of rounding.

For the first post-fix natural cycle, correlate durable reservation, request ID, attempt, last egress, commit, receipt, queue terminal transition, cleanup and worker return. Queue COMPLETED at roughly 3–4 seconds is not the full-cycle measurement. Configuration of a 5 req/s limiter alone does not prove observed peak rate; mark the rate measurement NOT_CHECKED without sufficient transmission timestamps.

## 3. Backup freshness at dispatch

The reported verified source is `sm-backup-20261121T060348Z-cdf2fdf2`, SHA-256 `a5fec36b5a6f137f2fd39b1a3f734a53979d0c6d3cf5cd9a1288f10d09db640c`, 86,561 bytes. Its reported isolated PostgreSQL 16 restore at 08:43 UTC checked 12 tables, four triggers and rejection probes. That restore does not make the source backup younger or establish complete production role/configuration recovery.

Calculate source age from backup creation at approximately 06:03:48 UTC, not from restore time:

| Evaluation | Age / result |
| --- | --- |
| Saturday 15:26 UTC source assessment | Approximately 9h22m; within the 12-hour ceiling at that time. |
| Saturday freshness boundary | Approximately 18:03:48 UTC. |
| Monday 07:30 UTC preflight | Approximately 49h26m; stale. |
| Monday 14:15 UTC Slot 1 | Approximately 56h11m; stale. |
| A Monday 00:00 UTC backup at Slot 1 | 14h15m; too old even if restored successfully. |
| A Monday 06:00 UTC backup at Slot 1 | 8h15m; could qualify with matching successful restore evidence. |
| A Monday 12:00 UTC backup at Slot 1 | 2h15m; could qualify with matching successful restore evidence. |

Use actual creation timestamps and approved freshness evidence for each dispatch. A particular cron hour is not itself proof of a successful recoverable backup. A qualifying 06:00 backup also ages beyond 12 hours before the later Monday slots; reassess availability of newer qualifying evidence rather than treating Slot 1 clearance as whole-day clearance. Retain pre-ingestion backup and containment checks; do not weaken them to meet the cycle deadline.

## 4. Authorization proposal — illustrative only

No activation block or copy-ready OWNER values are issued in this correction. The draft grant ID from the supplied assessment remains an unissued proposal, not a durable authorization record.

An illustrative interval of Monday `2026-11-23T07:30:00Z` through Tuesday `2026-11-24T03:30:00Z` is 20 hours and covers Monday's four scheduled slots. Actual issuance must use the true arming timestamp; recalculate duration and coverage if that timestamp changes. Do not reuse a stale SHA or silently backdate issuance.

Every proposed renewal must retain the same trial ID, exact AAPL/NVDA ticker/CIK bindings, ingestion-only flag and durable quotas; bind to the freshly verified qualified release; and satisfy the deployed <=24-hour ceiling. A separate operator decision is required to apply it. The authorization is a time-and-release-bound configuration interlock; no cryptographic-signature guarantee is established here.

Any final grant must expire no later than `2026-11-25T02:35:00Z` / 03:35 Berlin. `ARX_TRIAL_SHUTDOWN_AT` is not an enforced fence. Expiry alone does not prove role DISABLED, explicit deauthorization, lock release or queue cleanup; verify those separately at final teardown.

## 5. Existing schedule and non-circular Monday decision

The existing **Check SEC grant expiry** automation was read directly during this correction. It is enabled at 08:30 Europe/Berlin daily for four occurrences beginning Nov 21, including **Monday Nov 23 at 08:30 Berlin / 07:30 UTC**. Its prompt is read-only and prohibits dispatch, SEC calls, grant renewal, configuration changes, restart/deployment, production SQL writes and hold changes. It checks available backup evidence and emits a focused operator prompt if evidence is missing; it does not automatically perform a restore or arm the worker. No schedule or prompt was changed and no duplicate task was created. A scheduled occurrence is not a guarantee of completion or delivery.

The existing **Review Smart Money trial** automation is also enabled, including its final Nov 25 03:45 Berlin review after the 03:35 Berlin hard boundary. It must report BLOCKED and an operator teardown prompt if shutdown cannot be independently verified. It does not perform shutdown.

Monday slots, all belonging to the Nov 23 New York trial business day:

| Slot | UTC | Europe/Berlin |
| --- | --- | --- |
| SLOT_1:0915 | Nov 23 14:15 | Nov 23 15:15 |
| SLOT_2:1300 | Nov 23 18:00 | Nov 23 19:00 |
| SLOT_3:1630 | Nov 23 21:30 | Nov 23 22:30 |
| SLOT_4:2030 | Nov 24 01:30 | Nov 24 02:30 |

Decision sequence:

1. Freshly verify prerequisite gates: deployed release, containment, authoritative authorization, authenticated SQL lock/queue/attempts/quotas, exact scope, baseline/snapshot and qualifying recoverable backup at dispatch. Missing mandatory evidence means ARMING BLOCKED; use NOT_CHECKED rather than inferring success.
2. If prerequisite evidence and operator authorization support the existing trial, classify **READY_FOR_BOUNDED_VALIDATION**, not “all execution acceptance gates passed.” Do not require an already-proven post-fix natural-cycle result as a prerequisite to observing the first such cycle.
3. Observe only the existing approved natural slot, without manual replay or extra weekend/catch-up work. Measure the complete worker cycle and reconcile all evidence. Successful no-work still does not exercise new-filing delivery/parse/persistence.
4. On a bound violation or incomplete accounting, retain failures and escalate through the approved containment/teardown procedure. The read-only monitor must not mutate production itself.
5. Reassess subsequent slots, backup freshness and Tuesday authorization independently. No automatic scope expansion, rollover or grant renewal follows a passing first slot.

All Radar, publication, scoring, trading, research and UX holds remain independent. This correction provides no operational activation authority and does not change production.
