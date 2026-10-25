# ARX TERMINAL — SMART MONEY CANDIDATE 006
# REGRESSION & COMPATIBILITY REPORT

**Execution Mode**: `NONPRODUCTION_INTEGRITY_CORRECTION_AND_EVIDENCE_REQUALIFICATION`  
**Target Candidate**: Candidate 006 Backup-Age Containment  
**Baseline SHA**: `9fdcc69fc2a291b8552cd9df8d56e3a2343cda04`  
**Rollback Reference SHA**: `76a8691339c30edc35e4e056d5b295509a81fe3a`

---

## 1. Full Regression Test Battery Results

| Test Suite | Tests Collected | Passed | Failed | Errors | Duration | Evidence Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `test_smart_money_backup_containment_adversarial.py` | 37 | 37 | 0 | 0 | 4.17s | Simulated / Contract |
| `test_smart_money_backup_checksum_integrity_adversarial.py` | 12 | 12 | 0 | 0 | 2.15s | Simulated / Storage Mock |
| `test_smart_money_backup_timestamp_authority_adversarial.py` | 12 | 12 | 0 | 0 | 1.95s | Simulated / Timestamp |
| `test_smart_money_backup_containment_fault_injection.py` | 1 | 1 | 0 | 0 | 2.17s | `LOCAL_SIMULATED_FAULT_INJECTION` |
| `test_smart_money_backup_containment_postgres_integration.py` | 1 | 1 | 0 | 0 | 9.92s | `REAL_DATABASE_INTEGRATION_VERIFIED` |
| `test_smart_money_candidate_006_readiness_adversarial.py` | 7 | 7 | 0 | 0 | 2.80s | Isolated Real Postgres |
| `test_smart_money_backup_security_adversarial.py` | 17 | 17 | 0 | 0 | 0.85s | Security & Least Privilege |
| `test_smart_money_backup_automation.py` | 5 | 5 | 0 | 0 | 18.20s | Real pg_dump & Restore |
| `test_smart_money_r2_adapter_compatibility.py` | 9 | 9 | 0 | 0 | 0.35s | Storage Adapter API |
| `test_smart_money_sole_owner_fido2_governance.py` | 7 | 7 | 0 | 0 | 0.25s | Governance FIDO2 Policy |
| `test_smart_money_worker_lease_recovery.py` | 10 | 10 | 0 | 0 | 1.10s | Worker Lease State |
| `test_smart_money_postgres_unit.py` | 12 | 12 | 0 | 0 | 0.40s | PostgreSQL Config & Types |
| **TOTALS** | **130** | **130** | **0** | **0** | **44.31s** | **100% PASS RATE** |

---

## 2. Backward & Forward Compatibility Evaluation

### Database Schema Compatibility
- **Active Schema**: `SMART_MONEY_POSTGRES_SCHEMA_V2`
- **DDL Migrations Introduced**: **ZERO**.
- Tables `form4_raw_filings`, `form4_parsed_transactions`, `scanner_snapshots`, `sm_audit_events` remain completely unchanged.

### API Contract Compatibility
- `GET /api/smart-money/health`: Observability envelope schema is 100% backward compatible.
- Containment states (`HEALTHY`, `FRESHNESS_WARNING`, `INGESTION_SUSPENDED`, `BACKUP_RECOVERY_PENDING`, `RECOVERY_POINT_VERIFIED`, `RESUMPTION_ELIGIBLE`) remain stable.

### Manifest Contract Compatibility
- Manifests written by `backup_engine.py` already populate `completed_at`, `started_at`, `artifact_sha256`, `artifact_size_bytes`, and `upload_verified`.
- Manifest parsing adheres strictly to the existing schema while rejecting incomplete or malformed instances fail-closed.

### Historical Backup Compatibility
- Backups uploaded prior to remote metadata tagging will return `UNVERIFIABLE` under the default strict contract (`require_remote_checksum_metadata=True`).
- A non-mutating audit flag (`require_remote_checksum_metadata=False`) is available for historical inspection without modifying objects.

### Rollback Safe Target
- Commit **`76a8691339c30edc35e4e056d5b295509a81fe3a`** (or `9cad087868386d40dc24970be80776abb3f8756a`) remains the verified rollback target. Zero schema migration or data transform is required to revert.
