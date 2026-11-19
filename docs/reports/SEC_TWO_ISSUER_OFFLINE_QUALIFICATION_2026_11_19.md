# AAPL / NVDA sequential canary qualification

Directive: ARX-SEC-TWO-ISSUER-QUALIFICATION-001
Base: 06f1ffdf82ef0dffec057b6acc7159017db57bb3 (PR #307)
Status: Offline qualification specification. Exact-head hosted checks and the linked pull request record execution results.

## Scope

This change adds offline integration coverage to the existing canary suite. It does not add or enable a production multi-issuer scheduler, change the allowlist, lift operational holds, or modify production ingestion code.

The qualification recipe is exactly two sequential existing one-shot executions: AAPL followed by NVDA. Each has a separate atomic transaction and explicit test-only process authorization. A shared SecRateLimiter is used across the pair. A failure stops the batch without retries; only the existing NO_UNSEEN_ELIGIBLE_FILINGS result is normalized to NO_ELIGIBLE_WORK before continuing.

## Budgets and boundaries

| Boundary | Limit |
| --- | --- |
| Issuers | AAPL / 0000320193, then NVDA / 0001045810 |
| Per-issuer requests | 4, enforced by the existing constructor and tracked_get |
| Aggregate requests | 8 maximum (two invocations only) |
| Shared request rate | 5 requests/second |
| Per-issuer duration | At most 30 seconds |
| Aggregate test-harness duration | At most 60 seconds |
| Unseen documents | At most 3 per issuer |
| Publication | No snapshot publication |

These limits are for the two-run offline qualification recipe. The older ten-issuer pilot is not being promoted or reused as an alternative persistence implementation. A manual sequence of unrelated CLI invocations must not be represented as sharing an in-process rate limiter.

## New PostgreSQL scenarios

1. Seed the verified AAPL filing through the existing atomic executor, preserve a nonempty active snapshot, then run AAPL duplicate and NVDA empty catalog. Compare full row contents across raw, parsed, availability, amendments, audits, receipts, snapshots and queue; expect no changes and one catalog request per issuer.
2. In an isolated empty database, ingest the real pinned AAPL fixture and synthetic NVDA XML sequentially. Verify separate CIK/ticker row groups, 3 AAPL plus 1 NVDA transaction, two issuer-bound PARTIAL receipts, four requests total, and unchanged active snapshot.
3. Supply NVDA XML under the AAPL catalog. Reject issuer mismatch, persist nothing, release ownership, revoke process authorization, and do not invoke the next issuer.
4. Inject failure in receipt persistence after AAPL transaction writes. Verify all writes roll back, all prior snapshot contents remain identical, the next issuer is not invoked, and a new session can acquire/release the owner lock.

The existing suite continues to cover request/timeout ceilings, invalid scope and CIKs, source integrity, compressed responses, trickling deadlines, CLI parity, queue contention and historical replay.

## Verification

- Local Python AST syntax check: PASS for four added tests.
- Local PostgreSQL execution: NOT RUN (pytest, pgserver, psycopg2 and requests are absent from this runtime).
- Required hosted command: ARX_REQUIRE_LOCAL_POSTGRES_TESTS=true python -m pytest tests/test_smart_money_nvda_canary_runner.py -v -rs
- Expected suite size: 117 tests (113 existing + 4 sequential scenarios).
- SEC evidence suite and RQ-01 through RQ-13 remain required.
- Qualification is not PASS until hosted required PostgreSQL tests complete without skips and exact-head CI succeeds.

## Operational boundary

No live SEC requests or production database writes are part of this qualification. Production AAPL success remains evidence for one issuer; NVDA no-work results do not certify historical zero activity. Recurring ingestion, scanner execution, trading and snapshot publication remain held.

This test harness is not a deployable batch executor. Any later live two-issuer execution must use an explicitly reviewed bounded plan and preserve aggregate budgets and pacing. No third issuer is admitted here.
