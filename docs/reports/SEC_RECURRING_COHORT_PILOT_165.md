# Issue #165 Recurring SEC Form 4 Cohort Pilot Runbook

**Governing Authority:** Owner Authorized under decree recorded in Step 11823 / Register Commit `27de416b`.  
**Trial Identifier:** `ARX-TRIAL-10-ISSUER-001`  
**Authorized Grant ID:** `c5986d3f8980fed2ef2f153306536b21`  
**Release Candidate SHA:** `1063a10fed1ed312feadeb671980a642e757e601`  
**Tracking Task / Container:** `SEC-COV-FOLLOWUP` / GitHub Issue #165  
**Designated Operator & Supervision:** David Akara (`daakara`)  
**Automated Schedule Sidecar:** `arx-issue-165-cohort-pilot` (`CRON_TZ=Europe/Berlin 0 15 30 11 *`)  

---

## 1. Production Execution Window (Monday, 30 November 2026)

The pilot is strictly bounded to Monday, 30 November 2026. Prior to window open (15:00 CET), the worker remains in fail-closed `DISABLED` standby.

| Milestone / Window | Europe/Berlin (CET) | America/New_York (EST) | UTC | Requirement / Action |
| :--- | :--- | :--- | :--- | :--- |
| **Preflight & Backup Pass** | 14:00–15:00 | 08:00–09:00 | 13:00–14:00 | Fresh off-host backup; verify SHA, DBs, unheld lock, snapshot |
| **Grant Validity Window Opens** | 15:00 | 09:00 | 14:00 | Arm worker if and only if all preflight gates PASS |
| **Slot 1 Window** | 15:15–15:25 | 09:15–09:25 | 14:15–14:25 | Single attempt; 40-req upfront reserve; cap 4/issuer, 40 total |
| **Slot 2 Window** | 19:00–19:10 | 13:00–13:10 | 18:00–18:10 | Single attempt; 40-req upfront reserve; cap 4/issuer, 40 total |
| **Slot 3 Window** | 22:30–22:40 | 16:30–16:40 | 21:30–21:40 | Single attempt; 40-req upfront reserve; cap 4/issuer, 40 total |
| **Grant Expiry & Forced Disable**| 22:41 | 16:41 | 21:41 | Restore DISABLED; verify lock release & zero post-cutoff sends |

---

## 2. Release-Bound Grant Configuration (Candidate `1063a10f`)

Verified against `analyst_dashboard/smart_money/continuous_authorization.py` on candidate `1063a10fed1ed312feadeb671980a642e757e601`:

```bash
# Core Authority & Role (Standby: DISABLED / false; Armed: OWNER / true)
ARX_CONTINUOUS_INGESTION_AUTHORIZED=true
ARX_SEC_EGRESS_ROLE=OWNER
ARX_SMART_MONEY_INGESTION_ONLY=true

# Release-Bound Grant Parameters (uses ARX_CONTINUOUS_AUTH_ID)
ARX_CONTINUOUS_AUTH_ID=c5986d3f8980fed2ef2f153306536b21
ARX_CONTINUOUS_AUTH_SCOPE=RECURRING_10_ISSUER_FORM4_PILOT
ARX_CONTINUOUS_AUTH_UNIVERSE_ID=UB-US-EQUITIES-20261025-001
ARX_CONTINUOUS_AUTH_RELEASE_SHA=1063a10fed1ed312feadeb671980a642e757e601
ARX_CONTINUOUS_AUTH_ISSUED_AT=2026-11-30T14:00:00Z
ARX_CONTINUOUS_AUTH_EXPIRES_AT=2026-11-30T21:41:00Z

# Database Routing & Isolation
SMART_MONEY_DATABASE_URL=postgresql://...  # Canonical database (PostgreSQL advisory lock)
SMART_MONEY_COHORT_PILOT_DATABASE_URL=postgresql://smart_money_ingestion:...@postgres.railway.internal:5432/finance_smart_money_cohort_001
```

- **Universe & Issuers (Exact 10)**: NVDA (`0001045810`), AAPL (`0000320193`), MSFT (`0000789019`), AMZN (`0001018724`), GOOGL (`0001652044`), META (`0001326801`), TSLA (`0001318605`), BRK-B (`0001067983`), JPM (`0000019617`), V (`0001403161`).
- **Scope & Isolation**: Form 4 and 4/A ingestion only into `finance_smart_money_cohort_001`. Public API, scanner activation, and scoring remain strictly decoupled and frozen.

---

## 3. Arming Handoff & Pre-Flight Protocol (Before 15:00 CET)

### Arming Ownership
- **Operator Ownership**: Explicit manual operator supervision by David Akara (`daakara`), responsible for verifying preflight, monitoring each slot, and executing emergency revocation or final disablement.
- **Automated Sidecar Handoff**: `arx-issue-165-cohort-pilot` (task container `SEC-COV-FOLLOWUP`) scheduled at `CRON_TZ=Europe/Berlin 0 15 30 11 *` to initiate the session.

### Preflight Gates (Must complete before 15:00 CET / 14:00 UTC)
1. **Fresh Off-Host Backup**:
   - Run `pg_dump` on `finance_smart_money_cohort_001`.
   - Store under verified off-host prefix `cohort-pilot/ARX-TRIAL-10-ISSUER-001/sm-backup-20261130T<HHMMSS>Z-<hash>`.
   - Record file size, SHA-256 hash, and verify readability in external storage (R2).
2. **Worker Candidate SHA**:
   - Confirm Railway `smart-money-worker` reports `ARX_RELEASE_SHA=1063a10fed1ed312feadeb671980a642e757e601`.
3. **Database Identities**:
   - Recheck dual DSN reachability: canonical database for lock, isolated database for evidence writes.
4. **Canonical Owner Lock**:
   - Verify PostgreSQL advisory lock is unheld while worker is in standby.
5. **Frozen Snapshot**:
   - Confirm active canonical snapshot remains frozen at `SNAP_SM_1792939989_48f30db6`.
6. **Operator Coverage**:
   - Confirm operator David Akara (`daakara`) is present and actively monitoring through 22:41 CET.

> [!CAUTION]
> **Fail-Closed Gate Rule**: If ANY check fails or supervision is unavailable, leave the worker in `DISABLED` standby (`ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`) and report `HOLD`.

---

## 4. Slot Execution & Transmission Accounting Rules

1. **Attempts & Missed Windows**:
   - Exactly **one attempt** allowed per window:
     - Slot 1: 15:15–15:25 CET
     - Slot 2: 19:00–19:10 CET
     - Slot 3: 22:30–22:40 CET
   - If a window closes without an attempt, record the miss and skip replay.
2. **Upfront Request Reservations & Caps**:
   - Reserve 40 requests up front per attempted slot via `TRIAL_CYCLE_EVENT`.
   - Reconcile reservations against actual transmissions (including redirects, fallbacks, and failures).
   - Enforce:
     - 4 transmissions per issuer per slot
     - 40 transmissions per slot
     - 120 total transmissions across all three slots
     - Shared rate limiter $\le 5\text{ req/s}$
   - Never refund an upfront reservation.
3. **Ingestion & Evidence Custody**:
   - Evidence writes route strictly to isolated database `finance_smart_money_cohort_001`.
   - Form 4 and 4/A only; deferred documents hold issuer checkpoints; unindexed amendments recorded as unresolved.

---

## 5. Anomaly Stops, Forced Expiry & Disablement

### Immediate Revocation Triggers
Actively revoke authorization and restore `DISABLED` immediately if:
- SEC returns HTTP `429 Too Many Requests`
- PostgreSQL advisory owner lock is lost
- Pre- or post-slot off-host backup fails
- Request accounting diverges between reservations and audit events
- Unexpected publication or snapshot mutation occurs

### Forced Disablement at Expiry (22:41 CET / 21:41 UTC)
At 22:41 CET, whether through natural expiration or early anomaly stop:
1. Explicitly restore Railway environment variables:
   - `ARX_SEC_EGRESS_ROLE=DISABLED`
   - `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`
2. Verify worker returns to disabled standby loop.
3. Verify canonical advisory owner lock is completely released.
4. Verify zero transmissions occurred after 22:41:00 CET (21:41:00 UTC).
5. Verify pilot queue has zero active or retryable work.

---

## 6. Reconciliation & Closeout Receipt

1. Reconcile `sm_audit_events` reservations, actual transmissions, and per-issuer transmission counts across all three slots.
2. Trace each completed queue row (`result_generation_id` $\to$ `sm_reconciliation_receipts.receipt_id`) and verify `cohort_evidence` hashes.
3. Confirm all retained catalogs match `SEC_COHORT_CATALOG_OBSERVED` SHA-256 hashes.
4. Record observed slot durations, transmission counts, latencies, and backup receipts in `ARX_ACTIVE_WORK_REGISTER_2026_11_24.md`.
5. Post final formal `ACCEPTED` or `HOLD` closeout receipt on GitHub Issue #165. Any rescheduling or extension requires a newly authorized bounded grant.
