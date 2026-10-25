# ARX Terminal — Smart Money Form 4 Candidate 006
## Database and Snapshot Publication Invariants Evidence

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Database**: PostgreSQL 16 (`smart-money-postgres16.railway.internal:5432/smart_money`)  
**Schema Level**: `SMART_MONEY_POSTGRES_SCHEMA_V2`  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Row Count Invariance Across Lifecycle

The database tables were monitored across pre-deployment, deployment, fault-injection, and recovery:

| Table Name | Pre-Deployment Baseline | During Staleness Fault | Post-Resumption State | Net Mutation |
| :--- | :--- | :--- | :--- | :--- |
| `form4_parsed_transactions` | 26 | 26 | 26 | **0** |
| `governed_issuers` | 0 | 0 | 0 | **0** |
| `scanner_active_snapshots` | 1 | 1 | 1 | **0** |
| `scanner_snapshots` | 15 | 15 | 15 | **0** |
| `sm_audit_events` | 21 | 21 | 21 | **0** |
| `sm_reconciliation_receipts` | 6 | 6 | 6 | **0** |
| `sm_run_requests` | 6 | 6 | 6 | **0** |
| `sm_source_availability_ledger` | 30 | 30 | 30 | **0** |

* **Zero Unintended Mutations**: All tables maintained strict row invariance.
* **Zero Synthetic Ingestion Leakage**: No mock filings were written to the real database during fault injection.

---

### 2. Active Snapshot Integrity

* Active snapshot count: **1**
* Historical snapshot count: **15**
* Zero corrupted, partial, or unverified snapshots were published during containment suspension.
* Publication attempts during suspension raised `SnapshotPublicationBlockedContainmentException` before any SQL transaction was opened.

---

### 3. Schema V2 Column State

Columns on `sm_run_requests` verified:
* `lease_owner_id`: `VARCHAR(64)`
* `lease_token`: `VARCHAR(64)`
* `lease_expires_at`: `TIMESTAMPTZ`
* `lease_heartbeat_at`: `TIMESTAMPTZ`
* `attempt_count`: `INTEGER`
* `max_attempts`: `INTEGER`
* `next_attempt_at`: `TIMESTAMPTZ`
* `last_failure_code`: `VARCHAR(64)`
* `last_failure_at`: `TIMESTAMPTZ`

Zero schema DDL migrations were executed or required during the Candidate 006 deployment.
