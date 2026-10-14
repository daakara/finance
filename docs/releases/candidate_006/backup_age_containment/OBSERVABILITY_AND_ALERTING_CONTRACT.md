# ARX Terminal — Smart Money Form 4
# Observability & Alerting Contract

**Module**: `analyst_dashboard.smart_money.storage.backup.containment_gate`  
**Endpoint**: `/api/smart-money/health`, Worker Envelope  
**Security Invariant**: Zero Tokens, Secrets, or Credentials in Observability Streams  

---

## 1. Structured Log Formats

The containment gate emits structured, machine-parsable logs at standardized severities with correlation IDs for cross-service tracing:

### Ingestion Suspended (CRITICAL)
Emitted whenever backup freshness is breached:
```
CRITICAL smart_money.backup.containment_gate: INGESTION_SUSPENDED [cid=9509dc93-aa86-45d2-b073-bfd6ee60bf40]: Containment gate closed. Status=STALE, Reason=RECOVERY_POINT_AGE_EXCEEDS_MAX_THRESHOLD, Age=43500.0s, Threshold=43200.0s.
```

### Freshness Warning (WARNING)
Emitted when backup age approaches the threshold (80% / 9.6h):
```
WARNING smart_money.backup.containment_gate: BACKUP_FRESHNESS_WARNING: Backup age 37800.0s approaches threshold 43200.0s.
```

### Ingestion Resumed (INFO)
Emitted upon successful verification of a new backup:
```
INFO smart_money.backup.containment_gate: INGESTION_RESUMED: Verified recovery point b-r2-20261014-112500-success (age=300.0s). Ingestion eligibility restored.
```

---

## 2. Health Endpoint Observability Envelope

The worker health envelope (`SmartMoneyIngestionWorker.get_health_envelope()`) and web status endpoint include a dedicated `containment` section:

```json
{
  "status": "healthy",
  "worker": {
    "state": "ONLINE_ACTIVE",
    "uptime_seconds": 86400,
    "current_run_request_id": "req-9821"
  },
  "containment": {
    "containment_enabled": true,
    "containment_effective": true,
    "current_state": "HEALTHY",
    "is_ingestion_eligible": true,
    "last_evaluation": {
      "recovery_point_id": "b-r2-20261014-112500-success",
      "verified_at": "2026-10-14T11:30:00Z",
      "backup_completed_at": "2026-10-14T11:25:00Z",
      "backup_age_seconds": 300.0,
      "freshness_threshold_seconds": 43200.0,
      "verification_source": "REMOTE_STORAGE_MANIFEST",
      "verification_status": "FRESH",
      "reason_code": "RECOVERY_POINT_WITHIN_THRESHOLD",
      "is_fresh": true
    },
    "last_suspend_at": "2026-10-14T11:05:00Z",
    "last_resume_at": "2026-10-14T11:30:00Z",
    "suspension_count": 1,
    "resumption_count": 1,
    "last_error_code": null
  }
}
```

---

## 3. Production Alerting Thresholds

| Severity | Alert Name | Condition | PagerDuty / Notification Route | Action Plan |
| :--- | :--- | :--- | :--- | :--- |
| **P1 — CRITICAL** | `ArxSmartMoneyIngestionSuspended` | `current_state == 'INGESTION_SUSPENDED'` for > 5 min | Primary On-Call Engineer | Check R2 credentials, inspect PostgreSQL backup logs, trigger manual backup |
| **P2 — WARNING** | `ArxSmartMoneyBackupAgeApproachingLimit` | `current_state == 'FRESHNESS_WARNING'` for > 30 min | Slack `#ops-alerts` | Verify backup scheduler container health, check Railway CPU/memory limits |
| **P3 — NOTICE** | `ArxSmartMoneyBackupVerificationFailed` | `verification_status == 'CHECK_FAILED'` | Slack `#eng-telemetry` | Inspect Cloudflare R2 API quota, verify egress network connectivity |

---

## 4. Privacy & Credential Scrubbing Guarantees

In strict adherence to security rules:
- No database passwords, DSNs, or hostnames are logged;
- No Cloudflare R2 Access Keys, Secret Keys, or Account IDs appear in receipts;
- All errors are mapped to standardized enumeration codes (`NO_BACKUP_MANIFESTS_FOUND`, `REMOTE_DUMP_ARTIFACT_NOT_FOUND`, `ARTIFACT_SIZE_MISMATCH`, etc.).
