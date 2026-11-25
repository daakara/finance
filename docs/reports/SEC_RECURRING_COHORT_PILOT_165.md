# Issue #165 recurring SEC Form 4 cohort pilot

**Decision pending owner authorization.** Prepare and qualify the worker candidate offline. The old NVDA/AAPL trial grant and quota ledger are exhausted. A new grant must name this pilot scope, candidate release and exact start and expiry. The worker stays disabled during preparation.

## Proposed production window

The proposed date is **Monday, 30 November 2026**. If release qualification or authorization is still pending, select a future SEC dissemination day and record fresh UTC bounds before creating the grant.

| Event | Europe/Berlin | America/New_York | UTC |
| --- | --- | --- | --- |
| Grant starts | 15:00 | 09:00 | 14:00 |
| Slot 1 | 15:15–15:25 | 09:15–09:25 | 14:15–14:25 |
| Slot 2 | 19:00–19:10 | 13:00–13:10 | 18:00–18:10 |
| Slot 3 | 22:30–22:40 | 16:30–16:40 | 21:30–21:40 |
| Grant expires | 22:41 | 16:41 | 21:41 |

- Use one dedicated worker and the existing PostgreSQL SEC owner lock. The existing worker dispatches three durable slots and shares one guarded SEC session across the issuers in each slot.
- Scope is `RECURRING_10_ISSUER_FORM4_PILOT`, trial is `ARX-TRIAL-10-ISSUER-001`, and universe is `UB-US-EQUITIES-20261025-001`. The worker verifies the exact ticker and CIK pairs for NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B, JPM and V.
- Each slot has one attempt and reserves 40 requests up front. Cap actual transmissions at 4 per issuer per slot, 40 per slot and 120 for the entire pilot, including redirects, fallbacks and failed sends. The shared limiter caps egress at 5 requests per second. A quiet slot normally uses 10 transmissions. Never refund a reservation.
- A started slot has a 60-second processing deadline. Advance neither the grant nor the trial identifier after a reservation. A missed or failed slot is recorded for review. A later authorized slot re-reads the bounded interval and can pick up outstanding accessions.
- The grant permits Form 4 and 4/A ingestion from its issuance time into the isolated evidence database. The pilot excludes historical page fetching, scoring, public availability of pilot records, snapshot publication, scanner activation and trading. An amendment whose original is absent from the pilot store is recorded as unresolved.

## Activation and stop

1. Provision a private PostgreSQL 16 database named `finance_smart_money_cohort_001` with the existing Smart Money schema. Give the worker an ingestion-principal DSN through `SMART_MONEY_COHORT_PILOT_DATABASE_URL` and keep the API pointed at the canonical database. Keep `SMART_MONEY_DATABASE_URL` on the worker pointed at that canonical database for the shared advisory lock. Configure verified external backup storage under the exact off-host prefix `cohort-pilot/ARX-TRIAL-10-ISSUER-001`. The pilot's pre- and post-slot backups use the isolated database and backup prefix. The worker requires both database identities and the off-host prefix before executing a slot.
2. Qualify the exact candidate. Rehearse the worker against disposable PostgreSQL and fixture SEC responses. Verify a fresh off-host backup, receipt integrity and restore applicability on that release. Confirm the dedicated worker deployment, API non-owner role, inactive scheduler, and existing snapshot and holds before considering an OWNER grant.
3. Record a separate owner decision on [Issue #165](https://github.com/daakara/finance/issues/165) with the exact deployment, release SHA, pilot scope, UTC window, caps and stop owner. Use the approved grant ID, universe, SHA, issued time and expiry in the existing `ARX_CONTINUOUS_AUTH_*` settings. Require `ARX_CONTINUOUS_INGESTION_AUTHORIZED=true` and `ARX_SEC_EGRESS_ROLE=OWNER` only on that worker for the authorized run. Keep `ARX_SMART_MONEY_INGESTION_ONLY=true`. The code enforces ingestion-only regardless of whether that setting is supplied and rejects attempts to disable it.
4. Observe each slot. Stop on lock loss, backup failure, unexplained ledger or queue discrepancy, SEC 429, invalid catalog or unexpected active snapshot change. For an operator stop, activate the existing emergency stop, revoke continuous authorization and restore the disabled worker role. Verify that the owner lock is released and no transmissions occur after the stop.
5. At expiry, the grant blocks new sends and the worker exits its owner loop. Restore the disabled configuration to avoid restarting with an expired OWNER grant. Reconcile the queue, including any unclaimed stale slot left by a process crash, and verify the snapshot, holds, and lack of post-expiry requests.

## Reconcile the pilot

- Use the three `TRIAL_CYCLE_EVENT` reservations and corresponding outcomes. Compare reserved and actual transmissions, per-issuer transmissions, slot duration, deadline, and pre/post backup receipts. Investigate any reservation without an outcome before increasing frequency.
- Follow each completed queue row's `result_generation_id` to `sm_reconciliation_receipts.receipt_id`. Its `failure_reasons.cohort_evidence` contains the issuer receipts. Match each `artifact_id` to its append-only `SEC_COHORT_CATALOG_OBSERVED` event. Base64-decode the retained response body and verify its SHA-256. Use `inventory_from_catalog` in `cohort_catalog.py` with the receipt's CIK, start and cutoff to reproduce the accession inventory. Recompute the issuer receipt hash over its contents excluding `sha256`, and recompute the reconciliation hash over the complete `cohort_evidence` JSON using `evidence_sha256`.
- For each accession, reconcile its disposition and hash against raw filings, public availability, parsed evidence and amendment records. A quiet result requires a retained catalog that covers the start boundary with no potentially overlapping unfetched page. Deferred documents and unresolved amendments leave explicit gaps and hold the issuer checkpoint.
- Accept the pilot when all three slots reconcile to one attempt each, every successful slot has ten reproducible issuer receipts, all counters and latency are within the declared limits, backups verify and the worker ceases SEC traffic by expiry. Record observed latency and request use for each slot. Any failed or incomplete condition stays open on Issue #165. Receipt-level accounting covers the observed catalog interval and the general reconciliation receipt stays `PARTIAL`; historical certification follows its existing process.
