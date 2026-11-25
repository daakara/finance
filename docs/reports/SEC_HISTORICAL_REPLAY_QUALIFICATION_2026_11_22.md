# ARX SEC Ingestion PR #317 — Authoritative Release Preflight & Governance Evidence Receipt

**Issue Reference:** Issue #165 — Bounded Resumable Ingestion & Exact Phase Delivery  
**Candidate Commit SHA:** `f25c10849773c2c234de6f4399a1410b1e18d72a`  
**Passing Hosted Qualification:** GitHub Actions Run ID [`45011651092`](https://github.com/daakara/finance/actions/runs/45011651092) (All 13 gates PASS)  
**Production Release Baseline:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`  
**Rollout Packet Reference:** [ARX_SEC_INGESTION_10_ROLLOUT_PACKET_317.md](file:///Users/iDavid/Projects/finance-sec-expansion/docs/releases/ARX_SEC_INGESTION_10_ROLLOUT_PACKET_317.md)  
**Preflight Timestamp:** `2026-11-22T13:47:40Z` (Local: `2026-11-22T14:47:40+01:00`)  

---

## 1. Exhaustive Deployed Queue Eligibility Audit (`sm_run_requests`)

*Direct authenticated read-only SQL executed against production PostgreSQL 16.15 database `finance_smart_money`:*

```sql
SELECT status, count(*) FROM sm_run_requests GROUP BY status;
-- Total Rows: 6 (4 COMPLETED, 2 FAILED)
```

### Complete Category Breakdown Across All Deployed Eligibility Invariants

| Category / Rule | SQL Condition Evaluated | Count | State Assessment |
| :--- | :--- | :---: | :--- |
| **Pending Jobs** | `status = 'PENDING'` | **0** | Clean |
| **Active Leased Running Jobs** | `status = 'RUNNING' AND lease_expires_at > NOW()` | **0** | Clean |
| **Stale / Expired Running Jobs** | `status = 'RUNNING' AND lease_expires_at <= NOW()` | **0** | Clean |
| **Running Jobs with Null Lease Expiry** | `status = 'RUNNING' AND lease_expires_at IS NULL` | **0** | Clean |
| **Due Retries** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at <= NOW()` | **0** | Clean |
| **Future Retries** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at > NOW()` | **0** | Clean |
| **Null Retry Timestamps** | `status = 'RETRY_ELIGIBLE' AND next_attempt_at IS NULL` | **0** | Clean |
| **Active Lease Tokens Held** | `lease_token IS NOT NULL AND status NOT IN ('COMPLETED', 'FAILED')` | **0** | Clean |
| **Non-Terminal Failures** | `status = 'FAILED' AND attempt_count < max_attempts` | **0** | Clean |
| **Terminal Failures** | `status = 'FAILED' AND attempt_count >= max_attempts` | **2** | Historical terminal records (`7873e2a9`, `5de6e1b3`) |
| **Terminal Completed Jobs** | `status = 'COMPLETED'` | **4** | Completed trial slot jobs (`abae9bd6`, `fc90cef2`, `ed340c5c`, `4b386b60`) |
| **Unhandled / Unknown Statuses** | `status NOT IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE', 'FAILED', 'COMPLETED')` | **0** | Clean |
| **Active Owner Advisory Locks** | `pg_locks.locktype = 'advisory' AND ((classid::bigint << 32) \| objid) = 401007311509093371` | **0** | **UNHELD (0 total advisory locks)** |

---

## 2. Cryptographic Validation of Baseline & Snapshot Contents

*Evaluated directly from database storage without inferring immutability from timestamps or row counts:*

### 2.1 Active Snapshot `SNAP_SM_1792939989_48f30db6`
- **Scanner ID / Universe:** `SMART_MONEY` / `SMART_MONEY_US_EQUITIES` (Size: 10, Matched: 10)
- **Publication Decision:** `PUBLISH` (Generated: `2026-10-25T14:53:08.136422Z`, Published: `2026-10-25T14:53:08.944349Z`)
- **Implementation Release SHA:** `b9588f95188ab7e204474773f67dacbc9f6afb1f`
- **Semantic Fingerprint:** `8082f0531f9c3b1cf679c029804ed7f13c2f68954d150c1c828ee12d7bb05b20`
- **Results Payload Digest:** SHA-256 `c3dc95f4b3240d9be872b1b32857c2ff789b7d2c571547807eb654cb59b952fd` (2,091 bytes)
- **Provenance Payload Digest:** SHA-256 `0952cf7f615adc6fb503297d150cb96523e669624eba0f50239b81921c995889` (227 bytes)
- **Freshness Payload Digest:** SHA-256 `cad649069fd68809e5cdd31676573694eb02abfa2cfe7577bd4ea4442a8d8d50` (271 bytes)
- **Subsequent Published Snapshots:** **0** (Active snapshot pointer remains strictly frozen)

### 2.2 Frozen NVDA Baseline (16 Form 4 Filings)
Every individual raw XML filing payload in `form4_raw_filings` was recomputed directly from `xml_content` and matched against its database `content_hash`:

| Accession Number | Filing Date | Accepted Datetime | Recorded SHA-256 Digest | XML Recomputed Match |
| :--- | :---: | :---: | :---: | :---: |
| `0001347842-26-000015` | `2026-08-24` | `2026-08-24T21:32:04+00:00` | `55fb0a5b2692762de621421e0726267b8eb3f9c2e8b66bf17874fdb8c5a741d1` | **MATCH (True)** |
| `0001199039-26-000012` | `2026-09-02` | `2026-09-02T22:10:34+00:00` | `d22cd5d85d32dbf39d62d7d6e599089411c4c11191642ab3618ab8c2dbcb7d33` | **MATCH (True)** |
| `0001696841-26-000010` | `2026-09-02` | `2026-09-02T22:07:18+00:00` | `86679077ccf5f0c83f960b34407f5af9d2c8df44f1ff885ffe9368527b4264d7` | **MATCH (True)** |
| `0001197647-26-000009` | `2026-09-04` | `2026-09-04T21:00:35+00:00` | `2d3af938f865688553a60efb056ef6c5c607712c17112cad294006dd5a4c112f` | **MATCH (True)** |
| `0001199039-26-000014` | `2026-09-08` | `2026-09-08T21:09:19+00:00` | `e902c5ae660d258a2606ad30dc7b756adc0f76271d83e319cc0ddabc50a7e5e6` | **MATCH (True)** |
| `0002152188-26-000005` | `2026-09-11` | `2026-09-11T21:04:47+00:00` | `21657169ed85656c099bdd3126f26de59d91b02c9f198d290a966af0ba29a7bc` | **MATCH (True)** |
| `0001197649-26-000012` | `2026-09-18` | `2026-09-18T21:10:59+00:00` | `822eaaac11fc1cd84bead616a51b338fc8b5511b763a5ad0ac1e694272cecfab` | **MATCH (True)** |
| `0001243821-26-000007` | `2026-09-18` | `2026-09-18T21:15:34+00:00` | `3b434b21cb3b6e0d166dfd360fc1fabb67a4b804797be0b6042f08a515d40c23` | **MATCH (True)** |
| `0001283854-26-000010` | `2026-09-18` | `2026-09-18T21:14:45+00:00` | `9030fa885d369e1b4d9d5f1247dd18205e9f233667a710a68f905c623fecfde7` | **MATCH (True)** |
| `0001588670-26-000014` | `2026-09-18` | `2026-09-18T21:12:42+00:00` | `1f019bc438b269561e5c054aec78c42956daa558e4d6bc89a0900c9f532c3afd` | **MATCH (True)** |
| `0001696841-26-000012` | `2026-09-18` | `2026-09-18T21:13:29+00:00` | `7f70894a4cf331995cf3a1fa40388d6db78b7c19836a834fae28b909d1ceb80c` | **MATCH (True)** |
| `0001199039-26-000016` | `2026-09-22` | `2026-09-22T20:56:23+00:00` | `872cdb869ad15f7a8a6d3396e722d23ba32e66af46cc4c495795765615594548` | **MATCH (True)** |
| `0001696841-26-000014` | `2026-09-23` | `2026-09-23T20:56:28+00:00` | `87b3a8f9f46990cc520bcd8af0970fbfb1aac270147ef430a23e24ef926be228` | **MATCH (True)** |
| `0001197647-26-000011` | `2026-10-09` | `2026-10-09T21:04:09+00:00` | `708b054a77a77701bd7bb39802e4ac07cdf15876c461825ecb14cd4695374985` | **MATCH (True)** |
| `0002152188-26-000007` | `2026-10-13` | `2026-10-13T21:09:29+00:00` | `2ac0f16ab6661c9ccd85c35f476657b9dbdc4b4109b15607709522e45de92f56` | **MATCH (True)** |
| `0001197647-26-000013` | `2026-11-06` | `2026-11-06T21:12:02+00:00` | `1488ef1a14107224ff366fa9164f2ec6c9e2b709c236206b8e8da43e3d95295d` | **MATCH (True)** |

- **Integrity Score:** **16/16 verified exactly.**
- **Aggregate NVDA Baseline Digest (Ordered SHA-256 Concat):**  
  `c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2`
- **Baseline Trusted Reference Status Reconciled:** Because no static offline reference file of these digests was previously checked into git, this result is formally characterized as **current database internal integrity verification** (each row's payload verified against its internal database column hash), rather than confirmation against an external pre-recorded trusted reference.

---

## 3. Automated Restore Verifier Execution (Disposable PostgreSQL)

*Executed via `BackupRestoreVerifier` using read-only Cloudflare R2 storage credentials against an ephemeral PostgreSQL 16.2 local socket target spawned via `pgserver` (Target: `/private/var/folders/.../arx_disposable_pg_target_...`). Zero production restore or R2 writes.*

### 3.1 Retrieved Backup Identity & Manifest Metadata
- **Backup ID:** `sm-backup-20261122T120306Z-a1179f0e`
- **Creation Timestamps:** Started `2026-11-22T12:03:06.951650Z` | Completed `2026-11-22T12:03:09.304078Z`
- **Artifact Size & Cryptographic Hash:** 87,270 bytes | SHA-256 `cb63fe7a39f0fde755699d8a28200ca382eb4bbdfd0e2a914449467c3c15ed5f`
- **Source Release SHA:** `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`
- **Remote Custody:** Cloudflare R2 (`s3://arx-smart-money-backups/production`)

### 3.2 Formal Isolated Restore Receipt
```json
{
  "verification_scope": "ISOLATED_RESTORE_STRUCTURE_AND_APPLICATION_READ",
  "backup_id": "sm-backup-20261122T120306Z-a1179f0e",
  "status": "VERIFIED_RECOVERABLE",
  "tables_count": 12,
  "present_tables": [
    "form4_parsed_transactions",
    "form4_raw_filings",
    "governed_issuers",
    "scanner_active_snapshots",
    "scanner_snapshots",
    "sm_amendments",
    "sm_audit_events",
    "sm_reconciliation_receipts",
    "sm_run_requests",
    "sm_schema_migrations",
    "sm_source_availability_ledger",
    "sm_source_gaps"
  ],
  "row_counts": {
    "scanner_active_snapshots": 1,
    "scanner_snapshots": 1,
    "form4_raw_filings": 54,
    "sm_run_requests": 6,
    "form4_parsed_transactions": 140,
    "governed_issuers": 13,
    "sm_amendments": 1,
    "sm_audit_events": 82,
    "sm_reconciliation_receipts": 8,
    "sm_schema_migrations": 4,
    "sm_source_availability_ledger": 45,
    "sm_source_gaps": 0
  },
  "triggers_count": 4,
  "active_snapshot_id": "SNAP_SM_1792939989_48f30db6",
  "globals_restored": false,
  "application_read_successful": true,
  "sha256_overhead_sec": 0.0005,
  "verification_duration_sec": 0.904,
  "verified_at": "2026-11-22T13:47:40Z",
  "source_data_parity_verified": false,
  "roles_and_grants_verified": false,
  "trigger_behavior_verified": false
}
```

### 3.3 Explicitly Bound Scope Limits (Marked NOT_CHECKED)
Per the authoritative verification contract:
1. `source_data_parity_verified: false` (**NOT_CHECKED**): Row counts are verified; row-by-row digest parity between production and target was not established.
2. `roles_and_grants_verified: false` (**NOT_CHECKED**): Routine logical backups omit cluster globals (`include_globals=false`); role/grant recovery was not tested.
3. `trigger_behavior_verified: false` (**NOT_CHECKED**): Trigger presence, definition, and attachment to `abort_immutable_record` were verified; active destructive mutation simulation was not performed.

---

## 4. Strict Separation of Three Operational Tracks

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRACK 1: STANDBY SOFTWARE RELEASE (PR #317 Candidate f25c1084 / 27dd7086)              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Preconditions:                                                                       │
│   1. Hosted qualification 100% passing (Run 45011651092).                              │
│   2. Owner approval to transition PR #317 from draft and merge to release/production. │
│   3. Railway worker deployed strictly in DISABLED / STANDBY configuration:             │
│      - ARX_SEC_EGRESS_ROLE = DISABLED                                                  │
│      - ARX_CONTINUOUS_INGESTION_AUTHORIZED = false                                     │
│      - ARX_INGESTION_SCHEDULE_ENABLED = false / unset                                  │
│      - ARX_SMART_MONEY_INGESTION_ONLY = true                                           │
│      - ARX_CONTINUOUS_AUTH_ID = expired / unset                                         │
│ • Scheduling Invariant: DO NOT DEPLOY DURING AN ARMED RELEASE-BOUND TRIAL.             │
│   Deployment must occur before arming or after trial completion.                       │
│ • Repository Guard: Keep PR #317 unmerged where merging could trigger automatic        │
│   deployment.                                                                          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRACK 2: MONDAY/TUESDAY ORIGINAL-TRIAL AUTHORIZATION (ARX-TRIAL-2026-NVDA-AAPL-001)    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Scope: Strictly 2 issuers — AAPL (0000320193) and NVDA (0001045810).                 │
│ • Immutable Release Binding: Bound strictly to release SHA 2eccc64a2cb3c649ad4d584a942.│
│ • Preflight Window: Monday Nov 23 at 08:30 Europe/Berlin (07:30 UTC).                  │
│ • Monday Preflight Task: Use Monday’s existing preflight task to verify actual         │
│   authorization, queue/lock/quota state, baseline and a qualifying fresh backup with   │
│   successful isolated restore evidence. Reuse scheduled backups; do not create an      │
│   additional backup job or renew authorization automatically.                          │
│ • Backup Freshness Rule: MANDATORY backup source age <= 12 hours at dispatch.          │
│   Sunday verification does NOT establish Monday readiness. A fresh preflight backup    │
│   and restore verification pass must be conducted Monday morning prior to arming.      │
│ • Shutdown Boundary: Authorized execution must end no later than 2026-11-25T02:35:00Z, │
│   with shutdown independently verified after that boundary.                            │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRACK 3: LATER EXPANDED-PHASE ACTIVATION (10-Issuer Profile)                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Prerequisite: Independent verified teardown of Track 2 after 2026-11-25T02:35:00Z.   │
│ • Operator Ratification: Formal confirmation of rollout packet parameters:             │
│   - Phase ID: ARX-SEC-INGESTION-10-20261125-001                                        │
│   - Ratified NY Dates: 2026-11-25, 2026-11-27 (Windows: 09:15–09:20 America/New_York)  │
│   - Allowlist: 10 governed CIKs (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B,     │
│     JPM, V)                                                                            │
│   - Ceilings: <=40 req/cycle, <=4/issuer, <=5 req/s, <=60s execution                   │
│ • Arming: Fresh daily grant ID (<=24h), fresh backup (<=12h), and schedule enabled     │
│   strictly for approved windows. Immediate return to deauthorized standby thereafter.  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Executive Release Decision

### Verdict: **`READY_FOR_STANDBY_RELEASE_REVIEW`**
*(Operational Code & Runtime Preflight Complete; Production Deployment Remains Gated / Blocked)*

### Exact Remaining Conditions for Release:
1. **Standby Release Review:** Owner review of PR #317 to approve transition from draft to ready-for-merge on `release/production`. Keep PR #317 unmerged where merging could trigger automatic deployment.
2. **Trial Preservation:** Preserve production release `2eccc64a` for the remaining original trial. Do not deploy candidate code while Monday/Tuesday’s release-bound trial (`ARX-TRIAL-2026-NVDA-AAPL-001` on `2eccc64a`) is armed or executing.
3. **Monday Preflight & Fresh Backup:** Use Monday’s existing preflight task to verify actual authorization, queue/lock/quota state, baseline and a qualifying fresh backup with successful isolated restore evidence. Backup source age must be $\le 12$ hours at dispatch time on Monday morning. Reuse scheduled backups; do not create an additional backup job or renew authorization automatically.
4. **Shutdown Boundary Verification:** Authorized execution must end no later than `2026-11-25T02:35:00Z`, with shutdown independently verified after that boundary.
5. **Expanded Phase Gating:** Expanded 10-issuer phase activation remains gated pending original-trial teardown after `2026-11-25T02:35:00Z` and explicit operator ratification of rollout parameters.

---

## 6. ARX Issue #165 — Offline NVDA Form 4 Evidence Reconciliation & Window Audit

### 6.1 Baseline Inventory & Cryptographic Parity (16 Filings)
Recomputed SHA-256 digests directly from `xml_content.encode("utf-8")` across all 16 frozen filings in `form4_raw_filings` (exported to `data/research/nvda_form4_frozen_baseline_export.json`):

| # | Accession Number | Filing Date | Accepted Datetime (UTC) | Reporting Owner (CIK) | XML Size (bytes) | SHA-256 Digest | Match |
| - | :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| 1 | `0001347842-26-000015` | `2026-08-24` | `2026-08-24T21:32:04Z` | Puri Ajay K (`0001347842`) | 2,044 | `55fb0a5b2692762de621421e0726267b8eb3f9c2e8b66bf17874fdb8c5a741d1` | **MATCH** |
| 2 | `0001199039-26-000012` | `2026-09-02` | `2026-09-02T22:10:34Z` | Stevens Mark A (`0001199039`) | 16,761 | `d22cd5d85d32dbf39d62d7d6e599089411c4c11191642ab3618ab8c2dbcb7d33` | **MATCH** |
| 3 | `0001696841-26-000010` | `2026-09-02` | `2026-09-02T22:07:18Z` | Teter Timothy S. (`0001696841`) | 8,630 | `86679077ccf5f0c83f960b34407f5af9d2c8df44f1ff885ffe9368527b4264d7` | **MATCH** |
| 4 | `0001197647-26-000009` | `2026-09-04` | `2026-09-04T21:00:35Z` | Coxe Tench (`0001197647`) | 5,282 | `2d3af938f865688553a60efb056ef6c5c607712c17112cad294006dd5a4c112f` | **MATCH** |
| 5 | `0001199039-26-000014` | `2026-09-08` | `2026-09-08T21:09:19Z` | Stevens Mark A (`0001199039`) | 16,764 | `e902c5ae660d258a2606ad30dc7b756adc0f76271d83e319cc0ddabc50a7e5e6` | **MATCH** |
| 6 | `0002152188-26-000005` | `2026-09-11` | `2026-09-11T21:04:47Z` | Parker Nicholas P. (`0002152188`) | 3,952 | `21657169ed85656c099bdd3126f26de59d91b02c9f198d290a966af0ba29a7bc` | **MATCH** |
| 7 | `0001197649-26-000012` | `2026-09-18` | `2026-09-18T21:10:59Z` | Huang Jen Hsun (`0001197649`) | 12,761 | `822eaaac11fc1cd84bead616a51b338fc8b5511b763a5ad0ac1e694272cecfab` | **MATCH** |
| 8 | `0001243821-26-000007` | `2026-09-18` | `2026-09-18T21:15:34Z` | Gawel Scott (`0001243821`) | 3,659 | `3b434b21cb3b6e0d166dfd360fc1fabb67a4b804797be0b6042f08a515d40c23` | **MATCH** |
| 9 | `0001283854-26-000010` | `2026-09-18` | `2026-09-18T21:14:45Z` | Shoquist Debora (`0001283854`) | 4,796 | `9030fa885d369e1b4d9d5f1247dd18205e9f233667a710a68f905c623fecfde7` | **MATCH** |
| 10 | `0001588670-26-000014` | `2026-09-18` | `2026-09-18T21:12:42Z` | Kress Colette (`0001588670`) | 14,906 | `1f019bc438b269561e5c054aec78c42956daa558e4d6bc89a0900c9f532c3afd` | **MATCH** |
| 11 | `0001696841-26-000012` | `2026-09-18` | `2026-09-18T21:13:29Z` | Teter Timothy S. (`0001696841`) | 4,586 | `7f70894a4cf331995cf3a1fa40388d6db78b7c19836a834fae28b909d1ceb80c` | **MATCH** |
| 12 | `0001199039-26-000016` | `2026-09-22` | `2026-09-22T20:56:23Z` | Stevens Mark A (`0001199039`) | 7,281 | `872cdb869ad15f7a8a6d3396e722d23ba32e66af46cc4c495795765615594548` | **MATCH** |
| 13 | `0001696841-26-000014` | `2026-09-23` | `2026-09-23T20:56:28Z` | Teter Timothy S. (`0001696841`) | 8,873 | `87b3a8f9f46990cc520bcd8af0970fbfb1aac270147ef430a23e24ef926be228` | **MATCH** |
| 14 | `0001197647-26-000011` | `2026-10-09` | `2026-10-09T21:04:09Z` | Coxe Tench (`0001197647`) | 5,282 | `708b054a77a77701bd7bb39802e4ac07cdf15876c461825ecb14cd4695374985` | **MATCH** |
| 15 | `0002152188-26-000007` | `2026-10-13` | `2026-10-13T21:09:29Z` | Parker Nicholas P. (`0002152188`) | 4,189 | `2ac0f16ab6661c9ccd85c35f476657b9dbdc4b4109b15607709522e45de92f56` | **MATCH** |
| 16 | `0001197647-26-000013` | `2026-11-06` | `2026-11-06T21:12:02Z` | Coxe Tench (`0001197647`) | 5,282 | `1488ef1a14107224ff366fa9164f2ec6c9e2b709c236206b8e8da43e3d95295d` | **MATCH** |

- **Integrity Result:** 16/16 verified exactly against internal database content hashes and raw XML digests.
- **Ordered Concatenated Aggregate SHA-256:** `c6c246f7a18315399160e1c6e8db540fcce9152672ac75c54ac481e9781faee2`.
- **Issuer Identity:** 100% NVDA / CIK `0001045810` / `NVIDIA CORP`. Zero cross-issuer pollution.
- **Duplicate Records:** **0 duplicates** (16 unique accessions).
- **Missing Records:** **0 missing** (16 of 16 accessions resolved with full XML payloads).

---

### 6.2 Derivation of the 36 Parsed Transactions & Code Distribution

Each XML document was parsed with `Form4XmlParser.parse_filing`, deriving transaction lines into non-derivative and derivative tables:

| Accession Number | Document Type | Filing Date | Non-Derivative Txs | Derivative Txs | Disclosed Codes | Total Rows |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `0001347842-26-000015` | `4` | `2026-08-24` | 0 | 0 | None (retirement exit disclosure) | **0** |
| `0001199039-26-000012` | `4` | `2026-09-02` | 7 | 0 | 7× Code S | **7** |
| `0001696841-26-000010` | `4` | `2026-09-02` | 3 | 0 | 3× Code S | **3** |
| `0001197647-26-000009` | `4` | `2026-09-04` | 1 | 0 | 1× Code G | **1** |
| `0001199039-26-000014` | `4` | `2026-09-08` | 7 | 0 | 7× Code S | **7** |
| `0002152188-26-000005` | `4` | `2026-09-11` | 1 | 0 | 1× Code A (172,507 shares) | **1** |
| `0001197649-26-000012` | `4` | `2026-09-18` | 2 | 0 | 1× Code F, 1× Code G | **2** |
| `0001243821-26-000007` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001283854-26-000010` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001588670-26-000014` | `4` | `2026-09-18` | 4 | 0 | 1× Code F, 3× Code S | **4** |
| `0001696841-26-000012` | `4` | `2026-09-18` | 1 | 0 | 1× Code F | **1** |
| `0001199039-26-000016` | `4` | `2026-09-22` | 2 | 0 | 2× Code S | **2** |
| `0001696841-26-000014` | `4` | `2026-09-23` | 3 | 0 | 3× Code S | **3** |
| `0001197647-26-000011` | `4` | `2026-10-09` | 1 | 0 | 1× Code G | **1** |
| `0002152188-26-000007` | `4/A` | `2026-10-13` | 1 | 0 | 1× Code A (160,418 shares) | **1** |
| `0001197647-26-000013` | `4` | `2026-11-06` | 1 | 0 | 1× Code G | **1** |
| **Total Baseline** | — | — | **36** | **0** | **25 S, 5 F, 4 G, 2 A** | **36** |

#### A. Root Cause of Zero-Transaction Parse (`0001347842-26-000015`)
- Accession `0001347842-26-000015` was filed by Ajay K. Puri on `2026-08-24`.
- XML content has completely empty `<nonDerivativeTable>` and `<derivativeTable>` elements, and zero `<nonDerivativeHolding>` / `<derivativeHolding>` entries.
- Explicit `<remarks>` element states:  
  `"The Reporting Person retired from his role of Executive Vice President, Worldwide Field Operations effective August 24, 2026 and is no longer subject to Section 16."`
- The filing is an administrative retirement notification (exit notice from Section 16 insider reporting obligations) rather than a transaction report.
- The parser disposition is correctly set to `FETCHED_NO_RELEVANT_TRANSACTIONS`, confirming that the zero-transaction parse is structurally authentic and verified under Section 16 rules.

#### B. Disclosed Transaction Code Inventory
- **Code S (Sale):** 25 transactions ($70\%$ of total) across Mark Stevens, Timothy Teter, and Colette Kress.
- **Code F (Tax Withholding):** 5 transactions ($14\%$) across Jen-Hsun Huang, Scott Gawel, Debora Shoquist, Colette Kress, and Timothy Teter.
- **Code G (Bona Fide Gift):** 4 transactions ($11\%$) across Tench Coxe and Jen-Hsun Huang.
- **Code A (Equity Grant / RSU Award):** 2 transactions ($5\%$) across Nicholas Parker (base grant + amendment).
- **Code P (Purchase):** **0 transactions** ($0\%$). No code-P transactions were found in the retained baseline.

---

### 6.3 Form 4/A Amendment Validation & Deduplication Analysis

#### A. Amendment Document Evidence & Scope Classification
- **Amendment Filing:** `0002152188-26-000007` (Form 4/A, filed `2026-10-13`, accepted `2026-10-13T21:09:29Z`).
- **Base Filing:** `0002152188-26-000005` (Form 4, filed `2026-09-11`, accepted `2026-09-11T21:04:47Z`).
- **Reporting Owner:** Nicholas P. Parker (CIK `0002152188`).
- **Date of Original Submission:** `2026-09-11`.
- **Period of Report:** `2026-09-09`.
- **Disclosed Remarks in XML:**  
  `"This amendment is being filed solely to correct an administrative error in the number of acquired shares on the Reporting Person's Form 4 filed on September 11, 2026."`
- **Classifier Verdict:** Formally evaluated by `classify_amendment_effect`:
  - **Status:** `PARTIAL_CORRECTION`
  - **Reason:** `"Explicit partial correction phrase verified in source narrative: 'solely to correct'."`
  - **1-to-1 Mapping:** Base transaction `0002152188-26-000005:0002152188:NON_DERIVATIVE:0` (172,507 shares @ $0.00, Code A) $\rightarrow$ Amendment transaction `0002152188-26-000007:0002152188:NON_DERIVATIVE:0` (160,418 shares @ $0.00, Code A).

#### B. Point-in-Time Supersession vs Static Aggregation (Verified in Disposable PostgreSQL)
- **Static Raw Database Storage:** Exactly 36 transaction rows are persisted in `form4_parsed_transactions` (complete audit trail preserved; no raw rows are deleted upon amendment).
- **Point-in-Time Supersession Query Behavior (`get_bulk_transactions_as_of`):**
  - **Before Amendment Boundary (`2026-10-13T21:09:28Z`):** 14 filings are available (Filings 1–14). Base transaction `0002152188-26-000005:0002152188:NON_DERIVATIVE:0` is effective with 172,507 shares. Amendment is not yet available. Returns **34 effective transactions**.
  - **At/After Amendment Boundary (`2026-10-13T21:09:29Z`):** Amendment `0002152188-26-000007` becomes available and supersedes the base transaction via verified mapping in `sm_audit_events`. Base transaction is excluded; amendment line is included with 160,418 shares. Later filing 16 is not yet available. Returns **34 effective transactions** ($34 - 1 + 1 = 34$).
  - **After All Baseline Filings Available (`2026-11-20T00:00:00Z`):** Filing 16 (`0001197647-26-000013`, accepted `2026-11-06T21:12:02Z`, 1 transaction) is now available. Base line remains superseded. Returns **35 effective transactions**.
  - **Raw Stored Rows:** Database maintains strictly **36 raw parsed rows** across all temporal query boundaries.

---

### 6.4 Submissions Index Comparison (August 26 – November 23, 2026)

Reconciled against retained external catalog `CIK0001045810_submissions.json` (SHA-256 `8dd44fbc9bc7fd47f0b16155e66cd22445345d0f9706f316e9e55d7ce4f48045`), retrieved `2026-11-22T08:53:12Z`:

#### A. Window Comparison Table

| Accession Number | Document Form | Filing Date | Acceptance Datetime | In Window (Aug 26 – Nov 23)? | In Frozen Baseline? | Disposition |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `0001197647-26-000013` | `4` | `2026-11-06` | `2026-11-06T21:12:02Z` | **YES** | **YES** | In-window baseline filing |
| `0002152188-26-000007` | `4/A` | `2026-10-13` | `2026-10-13T21:09:29Z` | **YES** | **YES** | In-window amendment |
| `0001197647-26-000011` | `4` | `2026-10-09` | `2026-10-09T21:04:09Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000014` | `4` | `2026-09-23` | `2026-09-23T20:56:28Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000016` | `4` | `2026-09-22` | `2026-09-22T20:56:23Z` | **YES** | **YES** | In-window baseline filing |
| `0001243821-26-000007` | `4` | `2026-09-18` | `2026-09-18T21:15:34Z` | **YES** | **YES** | In-window baseline filing |
| `0001283854-26-000010` | `4` | `2026-09-18` | `2026-09-18T21:14:45Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:13:29Z` | **YES** | **YES** | In-window baseline filing |
| `0001588670-26-000014` | `4` | `2026-09-18` | `2026-09-18T21:12:42Z` | **YES** | **YES** | In-window baseline filing |
| `0001197649-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:10:59Z` | **YES** | **YES** | In-window baseline filing |
| `0002152188-26-000005` | `4` | `2026-09-11` | `2026-09-11T21:04:47Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000014` | `4` | `2026-09-08` | `2026-09-08T21:09:19Z` | **YES** | **YES** | In-window baseline filing |
| `0001197647-26-000009` | `4` | `2026-09-04` | `2026-09-04T21:00:35Z` | **YES** | **YES** | In-window baseline filing |
| `0001199039-26-000012` | `4` | `2026-09-02` | `2026-09-02T22:10:34Z` | **YES** | **YES** | In-window baseline filing |
| `0001696841-26-000010` | `4` | `2026-09-02` | `2026-09-02T22:07:18Z` | **YES** | **YES** | In-window baseline filing |
| `0001347842-26-000015` | `4` | `2026-08-24` | `2026-08-24T21:32:04Z` | **NO** (Pre-window) | **YES** | Retained older baseline member |
| `0001310264-26-000008` | `4` | `2026-08-12` | `2026-08-12T21:13:10Z` | **NO** (Pre-window) | **NO** | Outside baseline window |
| `0001197647-26-000007` | `4` | `2026-08-07` | `2026-08-07T20:47:24Z` | **NO** (Pre-window) | **NO** | Outside baseline window |

#### B. Submissions Index Window Accounting
- **Unique In-Window Accessions in Submissions Index (`2026-08-26` to `2026-11-23`):** Exactly **15 unique accessions** (14 Form 4 + 1 Form 4/A).
- **Baseline Agreement:** All 15 in-window accessions exist in the retained baseline (100% agreement).
- **August 24 Out-of-Window Accession (`0001347842-26-000015`):** Filed two days prior to the August 26 window start. Retained in the older 16-filing baseline from the initial August 18–November 16 setup.
- **August 12 & August 7 Accessions:** Filed prior to August 26; properly excluded from the 16-filing baseline.

---

### 6.5 Epistemic Labeling & Critical Scope Limits

1. **Source Freshness Boundary:**  
   The retained submissions catalog was retrieved on `2026-11-22T08:53:12Z`. **A November 22 retrieval cannot certify completeness through November 23.** Any Form 4 filed during Monday November 23 trading hours (or Sunday evening post-retrieval) cannot be proven absent from historical evidence alone.
2. **Filing-Date vs Transaction-Date Coverage:**  
   Filing-date completeness does not equal transaction-date completeness:
   - Section 16(a) allows a 2-business-day reporting delay, meaning transactions on Friday November 20 or Monday November 23 may not be filed until Wednesday November 25.
   - Conversely, filings in this window include transactions that occurred in earlier calendar periods.
3. **Anti-Inference Rule Enforced:**  
   Coverage status remains **`PARTIAL`** and **`NOT_CERTIFIED`**. Absence of index entries or matching counts does not authorize removing operational holds or publishing snapshots.
4. **Preserved Operational Holds:**  
   `HOLD_SEC_001` remains **`ACTIVE`**. Egress remains `DISABLED`.

---

### 6.6 Summary of Reconciliation Findings & Smallest Remaining Action

| Dimension | Previous Status / Finding | Reconciled Finding | Authoritative Evidence |
| :--- | :--- | :--- | :--- |
| **Baseline Inventory** | 16 raw filings reported | **16 raw filings verified (16/16 SHA-256 match)** | Recomputed digests match `form4_raw_filings` |
| **Issuer Identity** | NVDA reported | **100% NVDA (`0001045810`), 0 duplicates** | All 16 XML roots verified |
| **Parsed Transactions** | 36 reported | **36 parsed rows (all 36 non-derivative, 0 derivative)** | Full ElementTree parse of all 16 filings |
| **Zero-Transaction Filing** | 1 unexplained zero parse | **`0001347842-26-000015` verified retirement exit notice** | Section 16 retirement text in `<remarks>` |
| **Amendment Effect** | Unclassified / ambiguous | **`PARTIAL_CORRECTION` via classifier** | 1-to-1 share correction (172,507 $\rightarrow$ 160,418) |
| **Active Transactions** | 36 reported | **34 $\rightarrow$ 34 $\rightarrow$ 35 effective transactions** | Real PostgreSQL `get_bulk_transactions_as_of` |
| **In-Window Filings** | Ambiguous 15 vs 16 | **Exactly 15 in-window filings (`2026-08-26` to `2026-11-23`)** | Aug 24 filing is pre-window baseline member |
| **Coverage Certification** | Unproved | **`PARTIAL` / `NOT_CERTIFIED`** | Nov 22 snapshot cannot certify Nov 23 |

#### Review and Candidate Disposition:
Review candidate PR #317 against offline qualification evidence. Candidate code remains strictly offline, non-deployable, with `HOLD_SEC_001=ACTIVE` remaining active. Zero production database writes, zero live SEC HTTP wire requests, zero schedule activations, zero grant renewals, and zero hold mutations.

---

## 7. ARX Issue #165 — Finalized Bounded NVDA Historical SEC Form 4 Coverage Qualification Assessment

**Assessment Finalization Timestamp:** `2026-11-25T08:00:00Z` (Local: `2026-11-25T09:00:00+01:00`)  
**Database As-Of Timestamp:** `2026-11-25T06:15:00Z`  
**Issuer Under Audit:** NVIDIA CORP (`NVDA` / CIK `0001045810` / SIC `3674`)  
**Parallel Production Trial:** `ARX-TRIAL-2026-NVDA-AAPL-001` (Trial completed; deauthorized; fail-closed standby verified)  
**Associated Machine-Readable Artifact:** [nvda_historical_coverage_reconciliation_165.json](file:///Users/iDavid/Projects/finance-clean/data/research/nvda_historical_coverage_reconciliation_165.json)

---

### 7.0 Corrections & Superseded Claims Ledger

To ensure absolute audit clarity and preserve evidence provenance, four reporting dimensions have been formally reconciled against authoritative read-only SQL, Railway container telemetry, and retained repository fixtures:

1. **Gate 1 Closeout Terminology Reconciliation:**
   - **Trial Rate Ceiling:** Recorded strictly as **<= 5 requests/sec** (enforced by application software `RateLimiter(max_per_second=5.0)`).
   - **Historical Rate Telemetry Status:** Peak <= 1 req/sec is formally withdrawn because container logs record slot-level batch execution bounds rather than microsecond-level HTTP transmission emissions. Per-request transmission rate telemetry is marked **`NOT_CHECKED`**. Offline limiter verification confirms that the application software RateLimiter(max_per_second=5.0) enforces <= 5.0 req/s pacing in code, but historical wire-rate compliance was unobserved in production container telemetry.
   - **Duration Terminology Correction:** "SQL Lease Duration" is renamed to **"Run-request duration"** (`completed_at - claimed_at` in `sm_run_requests`), reflecting the total task execution duration recorded by PostgreSQL timestamps, not a database transaction lease or row lock holding period.
   - **Decoupled Duration Metric Sources:**
     - *Run-request duration (`sm_run_requests`):* Total task execution duration recorded by PostgreSQL (`completed_at - claimed_at`), ranging from 3.021s to 4.641s (mean: 3.816s across all 12 slots), distinct from a database transaction lease.
     - *Container log cycle duration (Railway stdout):* Measured from worker process log start to completion: **6.342s (2026-11-23T14:36:29.286Z → 2026-11-23T14:36:35.628Z)**.
     - *Audit event cycle duration (`sm_audit_events`):* Measured from worker dispatch initiation to final audit emission commit (e.g. Day 2 Slot 1: 6.471s).
   - **Historical Day 1 Latency Breaches Retained:** Slots 2, 3, and 4 on November 20 (70.107s, 76.095s, 77.033s against the <= 60.0s ceiling) are permanently retained. Root cause: synchronous Step 10 backup retention sweeps checking 55+ legacy manifests sequentially over WAN (remediated in PR #315; Days 2 & 3 all passed sub-10s).
2. **Catalog Completeness Technical Standard Reconciled:**
   - *Superseded Claim:* Preliminary draft stated that the retained submissions catalog could not prove the absence of Form 4 filings through November 19 because the latest Form 4 in the catalog was dated November 06.
   - *Authoritative Finding:* Under SEC EDGAR filing architecture and the governing validation contract (`validate_submissions_catalog_evidence` §8), an authoritative, validated submissions catalog is technically capable of supporting the absence of Form 4 filings within any interval it contiguously spans (establishing technical evidence, not a legal opinion). For the closed window `[2026-08-26, 2026-11-19T21:45:59Z]`, the catalog contiguously spans the period, and all 15 in-window Form 4/A accessions are fully accounted for with zero gaps. The latest Form 4 date alone neither proves nor disproves completeness. However, for the full assessment interval `[2026-08-26T00:00:00Z, 2026-11-25T00:00:00Z)`, the catalog fails the governing validation requirement because it was retrieved on `2026-11-22T08:53:12Z` (latest filing: `2026-11-19T21:45:59Z`), which pre-dates the window end and pre-dates forward filing `0001696841-26-000016` (accepted `2026-11-23T21:54:28Z`). Thus, `PARTIAL_UNCERTIFIED` is retained without forcing premature certification or unjustified rejection.
3. **AAPL Stored Count Discrepancy Reconciled (Elimination of Subtraction Artifact):**
   - *Superseded Claim:* Earlier report version stated 38 AAPL raw filings and 104 AAPL transactions by subtracting NVDA from total database rows (55 - 17 = 38; 152 - 48 = 104).
   - *Authoritative Finding:* Direct authenticated read-only SQL grouping by `(ticker, cik)` on live database `finance_smart_money` at `2026-11-24T10:28:40Z` confirms AAPL has **exactly 6 raw filings** and **18 parsed transactions** (100% agreement with earlier baseline audits). The remaining 32 raw filings and 86 parsed transactions belong to 11 other historical governed issuers (GOOGL: 4/29, TSLA: 4/14, META: 4/12, MSFT: 4/7, V: 3/9, BRK-B: 3/5, AMZN: 4/4, JPM: 3/3, ACLS: 1/1, LLY: 1/1, VRTX: 1/1). Total database state is exactly 55 raw filings and 152 parsed transactions (17 + 6 + 32 = 55; 48 + 18 + 86 = 152).
4. **Transaction Terminology & Execution Venue Distinction Standardized:**
   - *Superseded Claim:* Transaction Code S was labeled broadly as "open-market sale" based on trading plan and pricing footnote text.
   - *Authoritative Finding:* Transaction Code S is strictly labeled as **"Sale"** per official SEC Form 4 specifications. Rule 10b5-1 trading plan adoption and weighted-average pricing disclosures are tracked as separate transaction attributes. Full ElementTree XML search across all 17 NVDA filings confirms the phrase "open market" appears **0 times** in `<footnotes>` and `<remarks>`. The designation "open-market sale" is therefore suppressed, as execution venue is unspecified in filing narrative.

---

### 7.1 Gate 2: Assessment Interval & Remaining Coverage Gap

The historical coverage qualification is governed strictly by the codebase authority in `analyst_dashboard.smart_money.storage.postgres_repository:validate_submissions_catalog_evidence` and `analyst_dashboard.smart_money.reconciliation_engine:FilingReconciliationEngine`:

1. **Explicit Assessment Interval:**
   $$\boxed{[\text{2026-08-26T00:00:00Z},\ \text{2026-11-25T00:00:00Z})}$$
2. **Current Teardown Verification:**
   - Authorization remains **`false`** (`ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`).
   - Role remains **`DISABLED`** (`ARX_SEC_EGRESS_ROLE=DISABLED`).
   - Owner advisory locks in PostgreSQL: **0 rows** (owner lock `401007311509093371` absent).
   - Queue state in `sm_run_requests`: **0 active or retryable rows** (clean queue).
3. **Known Retained Catalog Audit:**
   - Location checked: `tests/fixtures/sec_edgar/CIK0001045810_submissions.json`.
   - Size & SHA-256: 160,255 bytes | SHA-256 `8dd44fbc9bc7fd47f0b16155e66cd22445345d0f9706f316e9e55d7ce4f48045`.
   - Retrieval timestamp: `2026-11-22T08:53:12.691561+00:00`.
   - Issuer identity: CIK `0001045810`, ticker `NVDA`, company name `NVIDIA CORP`.
   - Schema & pagination: Validated. `filings.files` points to historical archive prior to 2020; all 2026 filings are 100% self-contained in `filings.recent`.
   - **Identified Coverage Gap:** The retained catalog terminates at `2026-11-19T21:45:59Z`. For the interval `[2026-11-19T21:45:59Z, 2026-11-25T00:00:00Z)`, no retained submissions catalog exists in repository or database storage.
   - **Backup Search Policy:** Repeated searching of production backups is bypassed; read-only inspection has already conclusively documented that natural trial runs do not retain submissions catalog response bytes.

---

### 7.2 Audit of Reusable Trial Evidence & Submissions Retention Gap

A read-only architectural and database audit confirms:

1. **Natural Trial Execution Gap:**
   - The trial worker queries `https://data.sec.gov/submissions/CIK{cik}.json` in volatile container memory during scheduled cycles.
   - It enumerates new accessions and persists primary Form 4 XML into `form4_raw_filings` with its SHA-256 hash.
   - **Parent Submissions Catalog Response Bytes Are NOT Retained:** The worker does not write catalog response bytes, retrieval timestamps, source URLs, or catalog SHA-256 to PostgreSQL or R2.
2. **Single Retained Catalog Limitation:**
   - The only retained catalog fixture in the repository is `tests/fixtures/sec_edgar/CIK0001045810_submissions.json` (ending November 19).
3. **Single Newly Ingested Filing Non-Proof Rule:**
   - Trial Slot 4 ingested forward accession `0001696841-26-000016` (Timothy S. Teter, accepted `2026-11-23T21:54:28Z`).
   - Ingesting this single filing proves that Teter filed on November 23, but **does not prove interval completeness** between November 06 and November 25.

---

### 7.3 Gate 4: Offline Accession Reconciliation & Transaction Accounting

All Forms 4 and 4-A accessions within the assessment interval `[2026-08-26T00:00:00Z, 2026-11-25T00:00:00Z)` are reconciled below, separating the immutable 16-filing baseline from forward trial additions:

| # | Accession Number | Form | Filing Date | Acceptance Datetime (UTC) | Reporting Owner (CIK) | SHA-256 Match | Stored Raw Txs | Corrected Terminology & Attributes | Accounting Disposition |
| - | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | `0001347842-26-000015` | `4` | `2026-08-24` | `2026-08-24T21:32:04Z` | Ajay K. Puri (`0001347842`) | **MATCH** | 0 | None (Holdings only) | Pre-window baseline member; verified Section 16 retirement exit notice |
| 2 | `0001199039-26-000012` | `4` | `2026-09-02` | `2026-09-02T22:10:34Z` | Mark A. Stevens (`0001199039`) | **MATCH** | 7 | 7x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | In-window baseline filing; 7 sales accounted & parsed |
| 3 | `0001696841-26-000010` | `4` | `2026-09-02` | `2026-09-02T22:07:18Z` | Timothy S. Teter (`0001696841`) | **MATCH** | 3 | 3x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | In-window baseline filing; 3 sales accounted & parsed |
| 4 | `0001197647-26-000009` | `4` | `2026-09-04` | `2026-09-04T21:00:35Z` | Tench Coxe (`0001197647`) | **MATCH** | 1 | 1x Bona Fide Gift (Code G) [10b5-1: Yes, WtdAvg: No, Venue: Unspecified] | In-window baseline filing; 1 gift accounted & parsed |
| 5 | `0001199039-26-000014` | `4` | `2026-09-08` | `2026-09-08T21:09:19Z` | Mark A. Stevens (`0001199039`) | **MATCH** | 7 | 7x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | In-window baseline filing; 7 sales accounted & parsed |
| 6 | `0002152188-26-000005` | `4` | `2026-09-11` | `2026-09-11T21:04:47Z` | Nicholas P. Parker (`0002152188`) | **MATCH** | 1 | 1x Equity Grant / Award (Code A) [172,507 sh; 10b5-1: No] | In-window baseline base filing; superseded by Form 4/A |
| 7 | `0001197649-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:10:59Z` | Jen-Hsun Huang (`0001197649`) | **MATCH** | 2 | 1x Tax Withholding (Code F), 1x Gift (Code G) [10b5-1: No] | In-window baseline filing; 2 transactions accounted & parsed |
| 8 | `0001243821-26-000007` | `4` | `2026-09-18` | `2026-09-18T21:15:34Z` | Scott Gawel (`0001243821`) | **MATCH** | 1 | 1x Tax Withholding (Code F) [10b5-1: No] | In-window baseline filing; 1 transaction accounted & parsed |
| 9 | `0001283854-26-000010` | `4` | `2026-09-18` | `2026-09-18T21:14:45Z` | Debora Shoquist (`0001283854`) | **MATCH** | 1 | 1x Tax Withholding (Code F) [10b5-1: No] | In-window baseline filing; 1 transaction accounted & parsed |
| 10 | `0001588670-26-000014` | `4` | `2026-09-18` | `2026-09-18T21:12:42Z` | Colette Kress (`0001588670`) | **MATCH** | 4 | 1x Tax Withholding (Code F), 3x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes] | In-window baseline filing; 4 transactions accounted & parsed |
| 11 | `0001696841-26-000012` | `4` | `2026-09-18` | `2026-09-18T21:13:29Z` | Timothy S. Teter (`0001696841`) | **MATCH** | 1 | 1x Tax Withholding (Code F) [10b5-1: No] | In-window baseline filing; 1 transaction accounted & parsed |
| 12 | `0001199039-26-000016` | `4` | `2026-09-22` | `2026-09-22T20:56:23Z` | Mark A. Stevens (`0001199039`) | **MATCH** | 2 | 2x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | In-window baseline filing; 2 sales accounted & parsed |
| 13 | `0001696841-26-000014` | `4` | `2026-09-23` | `2026-09-23T20:56:28Z` | Timothy S. Teter (`0001696841`) | **MATCH** | 3 | 3x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | In-window baseline filing; 3 sales accounted & parsed |
| 14 | `0001197647-26-000011` | `4` | `2026-10-09` | `2026-10-09T21:04:09Z` | Tench Coxe (`0001197647`) | **MATCH** | 1 | 1x Bona Fide Gift (Code G) [10b5-1: Yes, WtdAvg: No, Venue: Unspecified] | In-window baseline filing; 1 gift accounted & parsed |
| 15 | `0002152188-26-000007` | `4/A` | `2026-10-13` | `2026-10-13T21:09:29Z` | Nicholas P. Parker (`0002152188`) | **MATCH** | 1 | 1x Equity Grant / Award (Code A) [160,418 sh; 10b5-1: No] | In-window amendment; `SCOPE_PARTIAL_CORRECTION` replaces base |
| 16 | `0001197647-26-000013` | `4` | `2026-11-06` | `2026-11-06T21:12:02Z` | Tench Coxe (`0001197647`) | **MATCH** | 1 | 1x Bona Fide Gift (Code G) [10b5-1: Yes, WtdAvg: No, Venue: Unspecified] | In-window baseline filing; 1 gift accounted & parsed |
| 17 | `0001696841-26-000016` | `4` | `2026-11-23` | `2026-11-23T21:54:28Z` | Timothy S. Teter (`0001696841`) | **MATCH** | 12 | 12x Sale (Code S) [10b5-1: Yes, WtdAvg: Yes, Venue: Unspecified] | Forward trial accretion; 12 sales from Slot 4 accounted & parsed |

#### Forward Trial Addition Details:
- **Accession Number:** `0001696841-26-000016`
- **Filing Date:** `2026-11-23`
- **Acceptance Datetime (UTC):** `2026-11-23T21:54:28Z`
- **Period of Report (Reporting Date):** `2026-11-19`
- **Reporting Owner:** Timothy S. Teter (CIK `0001696841`)
- **Relationship:** Officer (Executive Vice President and General Counsel)
- **Transaction Codes:** 12x Sale (Code S) under Rule 10b5-1 trading plan, weighted average price disclosures; execution venue unspecified in narrative.
- **Disposition:** Discovered and ingested during trial Slot 4 (`52e3acb1-e682-4302-84b4-2f40f030d78f`), persisted with SHA-256 `c7ee40386d6a142d3c644acc348ce994aaf1f57a1ce8207fa40e0b9148f73077`, yielding 12 raw and effective transactions.

---

### 7.4 Quantitative Parity: Raw vs Effective Amended Transactions & Multi-Issuer Reconciliation

1. **Frozen Baseline Set (16 NVDA Filings):**
   - **Stored Raw Transactions:** Exactly **36 rows** in `form4_parsed_transactions` (25 S, 5 F, 4 G, 2 A).
   - **Zero-Transaction Filing:** `0001347842-26-000015` (Ajay K. Puri, filed `2026-08-24`) contains 0 transactions (Section 16 retirement exit notice).
   - **Amendment Effect:** `0002152188-26-000007` (Form 4/A) supersedes base `0002152188-26-000005` (Code A grant shares corrected from 172,507 to 160,418).
   - **Net Effective Transactions:** Exactly **35 effective transactions** after amendment supersession.
2. **Forward Addition Set (1 NVDA Filing):**
   - **Stored Raw Transactions:** Exactly **12 rows** (all 12 Code S).
   - **Net Effective Transactions:** Exactly **12 effective transactions**.
3. **Combined Sets Reconciled (17 NVDA Filings):**
   - **Total Raw Filings:** **17 rows** (16 baseline + 1 forward).
   - **Total Stored Raw Transactions:** **48 rows** (36 baseline + 12 forward).
   - **Total Effective Transactions:** **47 transactions** (35 baseline effective + 12 forward effective).
   - **Cumulative Transaction Codes:** 37 Sale (Code S), 5 Tax Withholding (Code F), 4 Bona Fide Gift (Code G), 2 Equity Grant (Code A raw, 1 effective after Form 4/A supersession), 0 Purchase (Code P).
4. **Authoritative Anti-Inference Rules Enforced:**
   - **Matching Counts Non-Inference:** Completeness is strictly NOT inferred from matching baseline or forward counts.
   - **Latest Filing Date Non-Inference:** Completeness is strictly NOT inferred from the presence of a filing on November 23.
   - **`sm_source_gaps` Non-Inference:** Completeness is strictly NOT inferred from the absence of rows in `sm_source_gaps`.
5. **Multi-Issuer Database Inventory (`finance_smart_money` at `2026-11-24T10:28:40Z`):**
   All 55 raw filings and 152 parsed transactions across all 13 governed issuers are fully accounted for, eliminating the earlier AAPL subtraction artifact (55 total - 17 NVDA = 38; 152 total - 48 NVDA = 104). AAPL has exactly 6 filings and 18 transactions; the remaining 32 filings and 86 transactions belong to the other 11 historical governed issuers (GOOGL: 4/29, TSLA: 4/14, META: 4/12, MSFT: 4/7, V: 3/9, BRK-B: 3/5, AMZN: 4/4, JPM: 3/3, ACLS: 1/1, LLY: 1/1, VRTX: 1/1).

---

### 7.5 UI / Radar Guardrails & Scoring Hold Verification

Direct read-only SQL inspection confirms that offline reconciliation does not compromise live system state:
1. **Active Snapshot:** `scanner_active_snapshots` points strictly to `SNAP_SM_1792939989_48f30db6` (`PARTIAL` coverage, published `2026-10-25T14:53:08Z`).
2. **Subsequent Published Snapshots:** **0 newer snapshots**. Pointer remains frozen.
3. **Reconciliation Receipts:** 11 `PARTIAL` receipts, 1 historical `COMPLETE` receipt (`1c224d5b`).
4. **Radar Adapter Fail-Closed Invariant:** `analyst_dashboard.smart_money.radar_adapter` suppresses factor scoring for governed issuers lacking a current `COMPLETE` coverage receipt.
5. **Operational Holds Maintained:** `HOLD_SEC_001` (Active), `HOLD_SCAN_001` (Active), `HOLD_EXEC_001` (Active), `PHASE_3_HOLD` (Active), and `ARX_CONFLUENCE_INTEGRITY_HOLD` (Active).

---

### 7.6 Gate 3: Conditional Acquisition Proposal

Because no retained submissions catalog covers the remaining interval `[2026-11-19T21:45:59Z, 2026-11-25T00:00:00Z)`, a single diagnostic acquisition is prepared—but **NOT executed**:

- **Target Resource:** `GET https://data.sec.gov/submissions/CIK0001045810.json`
- **Execution Constraints:**
  1. **Maximum HTTP Requests:** Exactly 1 request. Zero automatic retries; zero automatic redirects.
  2. **Durable Authorization Control:** Single-request budget survives process restarts and concurrent invocations via atomic disk claim (`os.O_CREAT | os.O_EXCL`). Reusing authorization ID (`AUTH-NVDA-DIAG-20261125-001`) fails closed before socket creation or rate limiter wait, surviving timeouts and crashes.
  3. **SEC User-Agent Header:** Formatted strictly per SEC EDGAR requirements: `ARXTerminal Research Akara.david26@gmail.com`.
  4. **Rate Limiter Pacing:** Governed by existing `RateLimiter` controls at <= 5.0 req/sec.
  5. **Authorization & Budget:** Governed by a separate diagnostic authorization and egress budget. Never reuses the exhausted/expired trial allowance.
  6. **Verifiable Storage Retention:** Retain exact raw response bytes, request URL, UTC acquisition timestamp, HTTP status, byte count, and response SHA-256 in verifiable storage before any downstream processing.
  7. **Downstream Isolation:** Zero Form 4/4-A XML downloads, zero filing parsing, and zero database ingestion.
  8. **Governance Hold Invariance & Intended Boundaries:** Coverage flag remains `PARTIAL_UNCERTIFIED`; zero factor scoring activation, zero snapshot publication, zero live trade execution. Production snapshot and holds are explicitly labeled as intended governance boundaries preserved by contract; offline diagnostic scripts and fixture tests do not independently establish live production state.

---

### 7.7 Deliverable & Operational Verdict

$$\boxed{\mathbf{VERDICT:\ ACQUISITION\_AUTHORIZATION\_REQUIRED}}$$

#### Exact Execution Prompt:
```bash
python3 scripts/acquire_nvda_submissions_diagnostic.py --auth-id "AUTH-NVDA-DIAG-20261125-001" --user-agent "ARXTerminal Research Akara.david26@gmail.com" --output-dir "data/sec_coverage_evidence/CIK0001045810"
```

#### Exact Gaps Preventing a Definitive Verdict:
1. **Submissions Catalog Coverage Gap:** Retained catalog `CIK0001045810_submissions.json` terminates on `2026-11-19T21:45:59Z`, leaving `[2026-11-19T21:45:59Z, 2026-11-25T00:00:00Z)` without contiguous index evidence.
2. **Trial Submissions Non-Retention:** The completed trial worker did not persist submissions catalog response bytes during Slot 4 execution.

#### Distinction Between Assessment Completion & Operational Certification:
- **Assessment Completion:** The mathematical and cryptographic reconciliation of all known retained filings and parsed transactions across baseline and forward accretion sets is 100% complete.
- **Operational Certification:** Requires unbroken, verified submissions catalog coverage spanning the entire assessment window to certify that zero unindexed Form 4/4-A filings occurred. Without contiguous catalog evidence covering through `2026-11-25T00:00:00Z`, operational certification cannot be granted.

#### Preserved Production State (Intended Governance Boundaries):
- **Verification Status:** `INTENDED_BOUNDARY_UNVERIFIED_IN_DIAGNOSTIC` (Offline/fixture tests do not establish live production state).
- **Active Snapshot:** `SNAP_SM_1792939989_48f30db6` (unmodified intended pointer).
- **Analytical Baseline:** 16 filings / 36 raw transactions / 35 effective transactions (strictly immutable).
- **Production Coverage Status:** **`PARTIAL_UNCERTIFIED`** (preserved).
- **Operational Holds:** `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD` (all strictly active intended boundaries).
