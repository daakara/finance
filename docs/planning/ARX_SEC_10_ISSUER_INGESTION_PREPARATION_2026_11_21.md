# SEC ingestion expansion — preparation and bounded proposal

Owner: David Akara (`@daakara`). Existing trackers: Issue #165 / Sprint P3 #231.
Reference source: production release `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
Disposition: **PREPARATION COMPLETE; LIVE EXPANSION BLOCKED ON CODE QUALIFICATION AND OPERATOR GO**.
This proposal creates no grant, schedule, queue request, configuration or production write.

## Evidence obtained offline

161 tests passed on PostgreSQL 16.2: the two-issuer trial suite, SEC evidence-integrity suite and five archived-replay preparation checks. All Python socket connections and DNS requests were restricted to loopback by `sec_offline_test_support`; no SEC endpoints were contacted. PostgreSQL used disposable local databases over TCP, with application schema, transactions, triggers and repository code.

The existing archived AAPL fixture `tests/fixtures/sec_edgar/AAPL_0001140361-26-044773.xml` has SHA-256 `eadd8fccc8161b484c2727d79e75a1d37b5303a625b9a22b13c36dd8f34e1ce9`. It parsed into S, S and G transactions, persisted three transactions plus raw bytes and availability evidence, replayed without duplicate rows, retained PARTIAL coverage, and rolled back all filing writes when availability persistence failed. No scanner snapshot was created. Existing synthetic tests additionally exercised amendment lineage, malformed input, rollback, quotas and preservation of a seeded snapshot.

This is one archived real issuer document plus synthetic adversarial evidence, **not** ten issuers of independently verified archived evidence, a live filing delivery, a production performance benchmark, or fresh SQL verification of production state.

Initial test infrastructure attempted Unix sockets, denied by this environment (54 setup errors); a loopback deadline test also failed while proxy environment variables were inherited. The final run used real PostgreSQL TCP, removed proxy variables only from the test process, and enabled the external-egress guard. No application behavior was changed to obtain the passing result. Final result: **161 passed, zero skipped, zero failures**.

## Confirmed implementation blockers

| Source | Finding | Required before expansion |
| --- | --- | --- |
| `continuous_authorization.py:require_continuous_authorization` | The recurring trial profile accepts exactly NVDA/AAPL. | A separately qualified expanded ingestion-only authorization profile; never enlarge the current trial grant. |
| `continuous_pipeline.py:SmartMoneyContinuousPipeline.__init__` | `ingestion_only=True` sets the aggregate cap to 8 even with ten issuers. Ten catalog requests already exceed it. | Explicit profile-bound budget allowing the complete cohort; no removal of global guards. A regression test reproduces this incompatibility. |
| `continuous_pipeline.py:_step10_prune_retention` | The unconditional retention bypass applies only to `is_trial_2_issuer`. | Keep remote retention/deletion outside the expanded ingestion-only cycle path and qualify its full duration. |
| Existing `DAILY_10_ISSUER_FORM4` profile | Publication is possible unless ingestion-only is explicitly enabled. | Make no-scoring/no-publication behavior non-overridable for the proposed profile and test attempts to disable it. |
| Trial scheduler and admission | Current durable trial reservations are tied to the two-issuer trial. | Qualify durable expanded-profile admission/accounting, duplicate delivery, restart and renewal behavior. Existing trial rows and budgets must remain untouched. |

## Proposed first expansion envelope — not deployed

Use the existing code's ten-issuer roster as a compatibility starting point, not as an investment ranking or representative market sample:

| Issuer | CIK |
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

- Proposed code profile: `RECURRING_10_ISSUER_FORM4_INGESTION_ONLY` (does not exist in deployed authorization code).
- Initial observation: one natural business-day cycle per day; review after two cycles before permitting additional frequency or scope. No automatic dates, weekend cycles, catch-up or replays.
- Proposed ceilings: 40 transmitted requests/cycle, 4/issuer/cycle, 40/day, 80 across the two-cycle observation, 5 requests/second and 60 seconds for the **entire** worker cycle through cleanup. These are maxima, not targets or demonstrated capacity.
- Count catalog pages, XML, fallback and failed transmissions. Reserve before egress; charge conservatively without resetting counters on restart or grant replacement. Never truncate an inventory and label omitted filings complete; exhausted work must remain explicitly deferred/unreconciled with a qualified resume policy.
- Exact allowlist and CIK binding; one OWNER worker; ingestion-only and no retention deletion. Historical coverage stays PARTIAL. Existing frozen NVDA evidence and active snapshot `SNAP_SM_1792939989_48f30db6` remain protected.
- Every authorization uses actual issuance time, a verified deployed release, and the deployed maximum of 24 hours. No grant is drafted with future/past fabricated issue times.
- Preserve the identity and durable quotas of `ARX-TRIAL-2026-NVDA-AAPL-001`. This proposal is a separately authorized phase, not a renewal or budget reset of that trial. Earliest execution is after verified trial teardown, unless the operator explicitly approves a superseding trial protocol. Any overlap must still honor `2026-11-25T02:35:00Z`; the unimplemented `ARX_TRIAL_SHUTDOWN_AT` setting is not a shutdown control.

## Acceptance gates and operator sequence

The [corrected Day 2 readiness assessment](../reports/ARX_ISSUE165_DAY2_READINESS_ASSESSMENT_CORRECTED_2026_11_21.md) separates dated Saturday evidence, Monday prerequisite checks and the first post-fix natural-cycle acceptance. It also records conservative charged quotas and the existing read-only monitor boundaries. It creates no execution authority.

1. Complete the existing Monday Nov 23 preflight at 08:30 Europe/Berlin. First natural slot is 15:15 Berlin / 14:15 UTC. Fresh deployment/release, authorization, authenticated SQL lock/queue/quota, baseline and snapshot evidence are required. Verify a qualifying backup and matching successful isolated restore; calculate its <=12-hour age at dispatch from backup creation time, not restore time. SQL-only gates are NOT_CHECKED if authenticated SQL is unavailable.
2. If authorized under the existing trial, observe the corrected natural slot. Correlate reservation, attempt, every transmitted request including failures, immutable receipt, queue terminal state and worker return. Queue COMPLETED alone does not establish <=60-second full-cycle acceptance. Preserve Day 1's three historical latency failures.
3. Implement and qualify only the code gaps above in the existing Issue #165 workstream. Prove immutable exact scope, mandatory ingestion-only behavior, denied publication, bounded cleanup, failure accounting and durable expanded quotas. This preparation PR changes tests/support/documentation only.
4. Before any expansion, obtain a fresh source-bound GO for the concrete profile, cohort, limits, schedule and stop procedure. A passing grant check or green offline suite does not activate Smart Money scoring, publication, Radar or trading.
5. Stop on a bound violation, evidence conflict, owner-lock loss, unexpected issuer, publication attempt, missing recoverable backup or incomplete accounting. Preserve receipts and return to DISABLED/STANDBY using the approved operator procedure. Require direct SQL proof of no owner lock and no active/retry-eligible queue after teardown; do not infer these from standby logs.

Operator prompt for the next engineering step:

> In Issue #165 / Sprint P3 #231, use this preparation receipt and the existing code. Prepare an isolated implementation of the proposed ten-issuer ingestion-only profile, fixing the reproduced eight-request incompatibility, trial-only retention bypass and missing expanded durable-admission contract. Keep existing two-issuer trial IDs, counters, acceptance history and all holds unchanged. Require the exact cohort, fail-closed quotas, mandatory no-publication behavior, full-cycle deadline evidence, restart/renewal accounting and offline failure tests above. Do not create/apply grants, dispatch/enqueue, contact SEC, write production SQL, change configuration, deploy, prune backups or modify holds. Return a tested candidate with remaining live gates explicitly NOT_CHECKED and a concrete bounded GO packet; do not request a generic full-platform audit.

## Reproduction

Install the repository test dependencies plus `pytest`, `pgserver` and `psycopg2-binary` in an isolated environment. From repository root:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  PYTHONPATH=tools/qualification:. ARX_REQUIRE_LOCAL_POSTGRES_TESTS=true \
  python -m pytest -p sec_offline_test_support \
  tests/test_sec_archived_replay_preparation.py \
  tests/test_smart_money_two_issuer_trial.py \
  tests/test_sec_ingestion_evidence_integrity.py -q --tb=short
```

This evidence does not verify current production locks, queue, grants, backup freshness, source lag, live rate or post-fix natural-cycle latency. Those remain **NOT_CHECKED by this offline preparation**.
