# ARX Terminal — Smart Money Form 4
# Architecture & Backup Freshness Authority Specification

**Status**: IMPLEMENTED & LOCALLY VERIFIED  
**Module**: `analyst_dashboard.smart_money.storage.backup.freshness_authority`  
**Execution Mode**: `NONPRODUCTION_BACKUP_DURABILITY_REMEDIATION_AND_REQUALIFICATION`  
**Security Boundary**: Fail-Closed, Zero Production Mutation  

---

## 1. Problem Statement & Historical Weakness

Prior to this implementation, the ARX Smart Money pipeline evaluated PostgreSQL backup freshness through indirect proxies:
1. Scheduler thread loop execution timestamps;
2. Local database audit table entries (`smart_money_backup_audit_log`);
3. Rehearsal deployment status without continuous storage validation.

These proxies created critical durability blindspots:
- If a Cloudflare R2 upload silently failed, or if credentials expired, the scheduler might log a failure, but the ingestion worker continued ingesting live SEC Form 4 filings into PostgreSQL without an off-volume recovery point.
- If a primary PostgreSQL node suffered catastrophic volume corruption, unbacked-up filings would be permanently lost, violating Recovery Point Objective (RPO) bounds.
- Local modification times or scheduler timestamps could drift, be spoofed, or desynchronize from actual remote off-volume object storage state.

---

## 2. The Canonical Freshness Authority

The newly implemented `BackupFreshnessAuthority` establishes a cryptographically grounded, single-source authority for PostgreSQL backup freshness.

### Authoritative Architecture

```
+-----------------------------------------------------------------------+
|                       BackupFreshnessAuthority                        |
|                                                                       |
|  1. Query Storage Adapter (R2 / S3 / Local Off-Volume)                |
|     - List remote manifests: manifest_*.json                          |
|     - Sort deterministically by ISO-8601 completed_at timestamp       |
|                                                                       |
|  2. Cryptographic & Contractual Validation                           |
|     - Parse JSON manifest schema                                      |
|     - Validate dump artifact existence via HEAD request               |
|     - Corroborate uncompressed bytes & sha256 checksum               |
|     - Verify database major version (PostgreSQL 16)                   |
|     - Verify upload_verified == True                                  |
|                                                                       |
|  3. Clock Skew & Temporal Integrity                                  |
|     - Reject future-dated recovery points (> 300s skew)               |
|     - Calculate exact age: now() - completed_at                       |
|     - Compare against policy threshold (default: 43,200s / 12h)       |
|                                                                       |
|  4. Emit Immutable FreshnessEvaluationReceipt                         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                       BackupAgeContainmentGate                        |
|  - HEALTHY / FRESHNESS_WARNING / INGESTION_SUSPENDED                  |
|  - Fail-closed gating on Ingestion, DB Mutation, Snapshot Publish    |
+-----------------------------------------------------------------------+
```

### Prohibited Freshness Proxies
The following proxies are **strictly rejected** as evidence of freshness:
- Scheduler loop iteration timestamp alone;
- Host filesystem modification time (`os.path.getmtime`);
- Database audit records without corresponding verified storage manifests;
- Presumed rehearsal health without live storage queries.

---

## 3. Recovery Point Contract & Validation Rules

A recovery point is classified as `FRESH` if and only if all of the following rules pass:

| Rule | Requirement | Failure Code |
| :--- | :--- | :--- |
| **Manifest Discovery** | At least one valid manifest exists in remote storage | `NO_BACKUP_MANIFESTS_FOUND` |
| **Schema Integrity** | Manifest contains `completed_at`, `database_artifact`, `artifact_sha256`, `upload_verified` | `MALFORMED_MANIFEST_SCHEMA_KEYS` |
| **Upload Status** | `upload_verified` field equals `True` | `UNVERIFIED_BACKUP_UPLOAD_FLAG` |
| **Artifact Existence** | Storage adapter verifies remote dump artifact exists | `REMOTE_DUMP_ARTIFACT_NOT_FOUND` |
| **Size Integrity** | Dump artifact byte size matches manifest size (within 1024b) | `ARTIFACT_SIZE_MISMATCH` |
| **Temporal Integrity** | `completed_at` <= current time + `clock_skew_tolerance` (300s) | `FUTURE_DATED_RECOVERY_POINT_EXCEEDS_CLOCK_SKEW` |
| **Threshold Containment** | `backup_age_seconds` <= `max_verified_age_seconds` (43,200s) | `RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD` |
| **Database Version** | Database version matches PostgreSQL 16 contract | `UNSUPPORTED_POSTGRES_VERSION` |

If any single rule fails, the authority emits `is_fresh: False`, logs an audit warning, and the containment gate transitions to `INGESTION_SUSPENDED`.

---

## 4. Freshness Evaluation Receipt Schema

Every evaluation produces a strongly typed `FreshnessEvaluationReceipt`:

```python
@dataclass(frozen=True)
class FreshnessEvaluationReceipt:
    recovery_point_id: Optional[str]
    verified_at: str                        # ISO-8601 UTC timestamp of check
    backup_completed_at: Optional[str]      # ISO-8601 UTC timestamp of backup
    backup_age_seconds: Optional[float]     # Exact elapsed seconds
    freshness_threshold_seconds: float     # Configured threshold (e.g. 43200.0)
    verification_source: str               # "REMOTE_STORAGE_MANIFEST" | "STORAGE_ADAPTER_ERROR"
    verification_status: str               # "FRESH" | "STALE" | "MISSING" | "UNVERIFIABLE"
    reason_code: str                       # Detailed diagnostic code
    is_fresh: bool                         # Boolean eligibility flag
    manifest_metadata: Optional[Dict[str, Any]] = None
```

---

## 5. Clock Skew and Frozen Clock Testing

To guarantee determinism in production and testing:
1. `BackupFreshnessAuthority` supports dependency injection via `clock_fn: Optional[Callable[[], datetime]]`.
2. In production, `clock_fn` defaults to `datetime.now(timezone.utc)`.
3. In adversarial testing, frozen clocks simulate threshold crossing, future manifests, and leap seconds without flaky `time.sleep()` calls.
4. Future manifests exceeding 300 seconds (5 minutes) clock skew are rejected fail-closed to protect against NTP synchronization failures.
