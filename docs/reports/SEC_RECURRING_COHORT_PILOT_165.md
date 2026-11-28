# Issue #165 Recurring SEC Form 4 Cohort Pilot Runbook

**Governing Authority:** Owner Authorized decree on 2026-11-28 (moving trial from Tuesday, 1 December 2026 to Monday, 30 November 2026; supersedes Tuesday reservation; Tuesday is strictly NOT authorized).  
**Trial Identifier:** `ARX-TRIAL-10-ISSUER-001`  
**Authorized Grant ID:** `c5986d3f8980fed2ef2f153306536b21` (`ARX_CONTINUOUS_AUTH_ID`)  
**Release Candidate SHA:** `1063a10fed1ed312feadeb671980a642e757e601`  
**Tracking Task / Container:** `SEC-COV-FOLLOWUP` / GitHub Issue #165  
**Sole Manual Arming Operator:** David Akara (`daakara`)  
**Arming Path:** Single manual operator arming path (`SEC-COV-FOLLOWUP` / Issue #165); recurring cron triggers strictly prohibited and removed.  

---

## 1. Production Execution Window (Monday, 30 November 2026)

The pilot is strictly bounded to Monday, 30 November 2026. Prior to window open (15:00 CET), the worker remains in fail-closed `DISABLED` standby. This decree authorizes **one supervised natural slot only** (15:15–15:25 CET). Tuesday, 1 December 2026 is superseded and is **not authorized**.

| Milestone / Window | Europe/Berlin (CET) | America/New_York (EST) | UTC | Requirement / Action |
| :--- | :--- | :--- | :--- | :--- |
| **Preflight & Backup Pass** | 14:00–15:00 | 08:00–09:00 | 13:00–14:00 | Fresh scoped off-host backup with verified disposable restore $\le 12\text{ h}$; verify SHA, DBs, clean queue/locks, snapshot |
| **Grant Validity Window Opens** | 15:00 | 09:00 | 14:00 | Arm worker if and only if all 5 preflight gates PASS & operator present |
| **Natural Slot Window** | 15:15–15:25 | 09:15–09:25 | 14:15–14:25 | Single attempt; 40-req upfront reserve; cap 4/issuer, 40 total across pilot |
| **Grant Expiry & Forced Disable**| 15:35 | 09:35 | 14:35 | Restore DISABLED; verify lock release, zero post-cutoff sends, clean queue |

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
ARX_CONTINUOUS_AUTH_EXPIRES_AT=2026-11-30T14:35:00Z

# Database Routing & Isolation
SMART_MONEY_DATABASE_URL=postgresql://...  # Canonical database (PostgreSQL advisory lock)
SMART_MONEY_COHORT_PILOT_DATABASE_URL=postgresql://smart_money_ingestion:...@postgres.railway.internal:5432/finance_smart_money_cohort_001
```

- **Universe & Issuers (Exact 10)**: NVDA (`0001045810`), AAPL (`0000320193`), MSFT (`0000789019`), AMZN (`0001018724`), GOOGL (`0001652044`), META (`0001326801`), TSLA (`0001318605`), BRK-B (`0001067983`), JPM (`0000019617`), V (`0001403161`).
- **Scope & Isolation**: Form 4 and 4/A ingestion only into `finance_smart_money_cohort_001`. Public API, scanner activation, and scoring remain strictly decoupled and frozen.

---

## 3. Arming Handoff & Pre-Flight Protocol (Before 15:00 CET)

### Single Arming Path & Manual Operator Ownership
- **Sole Arming Operator:** David Akara (`daakara`) is the sole manual arming operator under tracking task ID `SEC-COV-FOLLOWUP` (GitHub Issue #165).
- **Zero Recurring Trigger Guarantee:** Exactly ONE arming path exists. All recurring cron triggers (including any annual sidecar definitions) are deleted and prohibited. There is no automated or recurring schedule trigger.
- **Preflight Timing:** Complete and record the fresh off-host pilot backup and all preflight checks before 15:00 CET on Monday, 30 November 2026.
- **Arming Verification:** At arming time (at or after 15:00 CET / 14:00 UTC), verify the preflight receipt and active operator presence. If either the preflight receipt or operator presence is missing, leave the worker disabled (`ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`) and report `NO-GO`.

### Mandatory Preflight Gates (Must complete before 15:00 CET / 14:00 UTC)
1. **Fresh Off-Host Backup with Verified Disposable Restore ($\le 12\text{ h}$ old)**:
   - Run `pg_dump` on `finance_smart_money_cohort_001`.
   - Store under verified off-host prefix `cohort-pilot/ARX-TRIAL-10-ISSUER-001/sm-backup-20261130T<HHMMSS>Z-<hash>`.
   - Record file size, SHA-256 hash, and execute disposable restore verification into ephemeral PG16 target (`VERIFIED_RECOVERABLE`, no older than 12 hours).
2. **Worker Candidate SHA**:
   - Confirm Railway `smart-money-worker` (`380acddd`) reports `ARX_RELEASE_SHA=1063a10fed1ed312feadeb671980a642e757e601`.
3. **Database Identities & Isolation**:
   - Recheck dual DSN reachability: canonical database for lock, isolated database `finance_smart_money_cohort_001` for evidence writes (least-privilege `smart_money_ingestion`).
4. **Clean Queue & Canonical Advisory Locks**:
   - Verify PostgreSQL advisory lock `401007311509093371` is unheld in `pg_locks` and worker queue has 0 active/retryable rows.
5. **Frozen Snapshot & Operational Holds**:
   - Confirm active canonical snapshot remains frozen at `SNAP_SM_1792939989_48f30db6` and all 6 operational holds are active.
6. **Operator Coverage**:
   - Confirm sole operator David Akara (`daakara`) is present and actively monitoring through slot execution and expiry (15:35 CET).

> [!CAUTION]
> **Fail-Closed Gate Rule**: If ANY check fails, the preflight receipt is missing, or operator presence cannot be verified, remain disarmed (`ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`) and report **`NO-GO`**. No manual dispatch, no backfill, no extra slots, and no automatic Tuesday retry.

---

## 4. Slot Execution & Transmission Accounting Rules

1. **Attempts & Single Natural Window**:
   - Exactly **one attempt** allowed during the single natural slot:
     - **Natural Slot:** 15:15–15:25 CET (14:15–14:25 UTC)
   - If the window closes without an attempt, record the miss and skip replay. No extra slots or Tuesday attempts authorized.
2. **Upfront Request Reservations & Caps**:
   - Reserve 40 requests up front for the single attempted slot via `TRIAL_CYCLE_EVENT`.
   - Reconcile reservations against actual transmissions (including redirects, fallbacks, and failures).
   - Enforce:
     - 4 transmissions per issuer per slot
     - 40 transmissions per slot
     - 40 total transmissions max across the pilot
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

### Forced Disablement at Expiry (15:35 CET / 14:35 UTC)
At 15:35 CET (or immediately upon slot completion at 15:25 CET / early anomaly stop):
1. Explicitly restore Railway environment variables:
   - `ARX_SEC_EGRESS_ROLE=DISABLED`
   - `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`
2. Verify worker returns to disabled standby loop.
3. Verify canonical advisory owner lock `401007311509093371` is completely released.
4. Verify zero transmissions occurred after cutoff.
5. Verify pilot queue has zero active or retryable work.

---

## 6. Reconciliation & Closeout Receipt

1. Reconcile `sm_audit_events` reservations, actual transmissions, and per-issuer transmission counts across the single slot.
2. Trace each completed queue row (`result_generation_id` $\to$ `sm_reconciliation_receipts.receipt_id`) and verify `cohort_evidence` hashes.
3. Confirm all retained catalogs match `SEC_COHORT_CATALOG_OBSERVED` SHA-256 hashes.
4. Record observed slot duration, transmission counts, latencies, and backup receipts in `ARX_ACTIVE_WORK_REGISTER_2026_11_24.md`.
5. Post final formal `ACCEPTED` or `HOLD` closeout receipt on GitHub Issue #165. Any rescheduling or extension requires a newly authorized bounded grant.

