# Containment Claim Disproof & Evidence Classification Matrix

## 1. Adversarial Evidence Disproof & Separation of Mechanisms

An adversarial review of the Railway cloud-runtime containment fault-injection test (`run_cloud_rehearsal_containment_test.py`) was conducted to separate genuine operational observations from in-process test mechanics.

### Critical Disproof Boundaries
1. **Pre-Mutation Gate Interception vs Rejected SQL Transaction**:
   The fault injection verified that `enforce_ingestion_gate()` and `enforce_snapshot_publication_gate()` raised `BackupAgeContainmentViolationException` and `SnapshotPublicationBlockedContainmentException`. This intercepted execution **before** any SQL statement or transaction was issued to PostgreSQL. No database engine write error or transaction abort was exercised because the application gate successfully prevented database contact.
2. **Policy Threshold Manipulation vs Dynamic Ingestion Recovery**:
   The transition from `INGESTION_SUSPENDED` back to `HEALTHY` in step T4 was achieved by restoring the evaluation policy threshold to 12 hours against the existing fresh backup. It did **not** test dynamic resumption triggered by an active background backup pass generating a new recovery point while locked in staleness.
3. **Worker Daemon State Independence**:
   While the in-container gate test observed `INGESTION_SUSPENDED`, the running worker daemon's state in `pg_stat_activity` remained `OWNER_LOCK_ACQUIRED`. The worker daemon itself was in internal work cycle sleep (`Work cycle suspended: Backup age containment gate closed`), not a full container lifecycle termination.

---

## 2. Evidence Classification Matrix

| Claim | Observed Mechanism | Evidence Type | Verdict |
| :--- | :--- | :--- | :--- |
| **Freshness Evaluation** | `BackupFreshnessAuthority.evaluate_freshness()` performed live HTTP/S3 metadata read from Cloudflare R2 bucket `arx-smart-money-backups` and verified SHA256 checksum. | Live Cloud (Railway -> R2) | **PASS** |
| **Ingestion Gate Staleness Detection** | Lowering threshold below recovery point age immediately shifted `BackupAgeContainmentGate` state to `INGESTION_SUSPENDED` (`eligible=False`). | Live Container Runtime (In-Process) | **PASS** |
| **Protected Ingestion Blocked** | `enforce_ingestion_gate()` raised `BackupAgeContainmentViolationException`, halting execution before SQL repository mutation. | In-Process Gate Logic | **PASS** |
| **Snapshot Publication Blocked** | `enforce_snapshot_publication_gate()` raised `SnapshotPublicationBlockedContainmentException`, preventing snapshot promotion. | In-Process Gate Logic | **PASS** |
| **Database Immutability** | PostgreSQL table row counts (`form4_parsed_transactions`, `scanner_snapshots`, `scanner_active_snapshots`, `sm_audit_events`) remained 100% invariant during fault injection. | Live Cloud PostgreSQL Query | **PASS** |
| **Operational Worker Recovery** | Restoring 12h policy returned gate to `HEALTHY` and re-enabled work cycles. Recovery via dynamic scheduled backup generation during active staleness was not tested. | Simulated Policy Restoration | **CONDITIONAL** |
| **Worker Ownership Concurrency** | Dedicated advisory locks `SEC_EGRESS_OWNER` (`401007311509093371`) and `BACKUP_EXECUTION` (`872055449083353674`) verified mutually exclusive and independent. | Live Cloud PostgreSQL Query | **PASS** |
| **SEC EDGAR Polling Suspension** | Worker logs verified `Work cycle suspended: Backup age containment gate closed`, preventing SEC HTTP egress during containment. | Live Container Runtime Log | **PASS** |

---

## 3. Disproof Conclusions

- **Gate Enforcement**: Robust, fail-closed, and verified in the real Railway container runtime environment against live R2 objects.
- **Limitation**: Operational recovery from an actual multi-day staleness breach via automated daemon backup generation remains an unobserved condition requiring prolonged soak verification.
