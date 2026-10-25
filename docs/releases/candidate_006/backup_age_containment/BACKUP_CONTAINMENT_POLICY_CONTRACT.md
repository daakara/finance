# ARX Terminal — Smart Money Form 4
# Backup Containment Policy Contract (Policy A)

**Policy Identifier**: `ARX_POLICY_SMART_MONEY_BACKUP_CONTAINMENT_V1`  
**Proposed Containment Threshold**: 12 Hours (43,200 Seconds)  
**Warning Threshold Ratio**: 80% of Max Age (9.6 Hours / 34,560 Seconds)  
**Current Status**: IMPLEMENTED & LOCALLY VERIFIED (PENDING OWNER PRODUCTION RATIFICATION)  

---

## 1. Governance Status Differentiation

To ensure strict compliance with release governance principles, the operational status of Policy A is distinctly categorized:

| Dimension | Status | Evidence / Location |
| :--- | :--- | :--- |
| **Code Implementation** | **IMPLEMENTED** | `analyst_dashboard/smart_money/storage/backup/freshness_authority.py`, `containment_gate.py` |
| **Local Adversarial Testing** | **VERIFIED (30/30 PASS)** | `tests/test_smart_money_backup_containment_adversarial.py` |
| **Fault-Injection Exercise** | **VERIFIED (11/11 STEPS PASS)** | `scratch/run_fault_injection_lifecycle.py` |
| **Non-Production Configuration** | **CONFIGURED & ACTIVE** | Active in unit/adversarial test harness |
| **Rehearsal Environment** | **PENDING EXPLICIT DEPLOYMENT** | Rehearsal services remain unmodified |
| **Production Authorization** | **PENDING OWNER RATIFICATION** | Hard fail-closed preflight gate enforced |

---

## 2. Configuration Parameters & Environment Contract

The policy contract is configured via standard environment variables:

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ARX_BACKUP_CONTAINMENT_ENABLED` | bool | `False` | Global kill-switch for containment gate |
| `ARX_BACKUP_MAX_VERIFIED_AGE_SEC` | float | `43200.0` | Maximum allowable backup age (12 hours) |
| `ARX_BACKUP_FRESHNESS_CHECK_TIMEOUT_SEC`| float | `10.0` | Timeout for querying storage adapter |
| `ARX_BACKUP_FRESHNESS_RECHECK_INTERVAL_SEC`| float | `60.0` | Freshness evaluation cache TTL |
| `ARX_BACKUP_FRESHNESS_CLOCK_SKEW_TOLERANCE_SEC`| float| `300.0` | Allowable future clock drift (5 minutes) |
| `ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED` | bool | `False` | Mandatory production release gate flag |

---

## 3. Production Authorization Gate (Fail-Closed)

To prevent accidental, unratified activation in production environments, `SmartMoneyIngestionWorker.validate_preflight()` enforces a strict invariant:

```python
# worker.py: validate_preflight()
if os.environ.get("ARX_ENVIRONMENT") == "production":
    if self.containment_policy.enabled:
        if not self.containment_policy.production_authorized:
            raise RuntimeError(
                "PREFLIGHT_FAIL_CLOSED: Backup containment is enabled in production "
                "but ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED is not True. "
                "Owner ratification of Policy A is mandatory before production activation."
            )
```

### Invariant Rules:
1. If running in `production` with `ARX_BACKUP_CONTAINMENT_ENABLED=true`, worker boot **aborts immediately** unless `ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=true`.
2. This eliminates the risk of an engineer enabling the feature in Railway without owner approval.
3. In non-production environments (`development`, `rehearsal`, `test`), the policy can be evaluated and exercised without production credentials or tokens.

---

## 4. Failure Modes and Fail-Closed Defaults

| Failure Condition | System Action | Resulting State |
| :--- | :--- | :--- |
| Cloudflare R2 API timeout | Catch exception, treat as `UNVERIFIABLE` | `INGESTION_SUSPENDED` |
| Cloudflare R2 403 Forbidden | Mark `CHECK_FAILED`, log alert | `INGESTION_SUSPENDED` |
| Manifest JSON corrupted | Reject schema, mark `UNVERIFIABLE` | `INGESTION_SUSPENDED` |
| Dump artifact deleted from R2 | HEAD check fails, mark `MISSING` | `INGESTION_SUSPENDED` |
| Dump checksum altered | Corroboration fails, mark `UNVERIFIABLE` | `INGESTION_SUSPENDED` |
| Worker rebooted while backup stale | Preflight passes but cycle halts | `BLOCKED_DATA_INTEGRITY` |
| Concurrent workers contend | Mutual exclusion locks hold | Single worker suspended |
