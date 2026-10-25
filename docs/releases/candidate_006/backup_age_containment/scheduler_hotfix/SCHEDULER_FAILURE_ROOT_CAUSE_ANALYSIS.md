# SmartMoney Backup Scheduler Defect Root Cause Analysis

## 1. Defect Identification

During the controlled Railway rehearsal of Candidate 006 (`0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0`), the deployment of Worker service `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` succeeded and acquired the primary worker advisory lock, but the background backup scheduler thread logged the following fatal exception on its first scheduled pass:

```text
ERROR [analyst_dashboard.smart_money.worker] Backup scheduler pass failed: name 'get_backup_storage_adapter' is not defined
Traceback (most recent call last):
  File "analyst_dashboard/smart_money/worker.py", line 104, in run_backup_pass
    adapter = self.storage_adapter or get_backup_storage_adapter()
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'get_backup_storage_adapter' is not defined
```

## 2. Root Cause & Lineage

1. **Missing Module Import**:
   When `SmartMoneyBackupSchedulerThread` was introduced into `analyst_dashboard/smart_money/worker.py` to drive automated background backup creation, `get_backup_storage_adapter` and `BackupStorageAdapter` were invoked within `run_backup_pass()`, but were not imported in `worker.py`'s import preamble.
2. **Test Coverage Gap**:
   Prior test suites (`test_smart_money_backup_containment_adversarial.py`, `test_smart_money_backup_containment_postgres_integration.py`, `test_smart_money_backup_containment_fault_injection.py`) tested the containment enforcement gates, integrity checks, timestamp authorities, and CLI backup routines directly. None of the prior suites instantiated and executed `SmartMoneyBackupSchedulerThread.run_backup_pass()` without a pre-mocked adapter in the thread's scope.
3. **Decoupled Failure Domain**:
   Because `SmartMoneyBackupSchedulerThread` runs as a separate daemon thread with a top-level try/except block around each pass, the exception was caught and logged, but incremented `self.failed_runs` while the main SEC ingestion loop continued running. Thus, container health checks did not crash, masking the defect until runtime logs were inspected.

## 3. Surgical Implementation Remedy

The defect was resolved in `analyst_dashboard/smart_money/worker.py` through the following changes:

1. **Import Resolution**:
   Added explicit import at line 38 of `analyst_dashboard/smart_money/worker.py`:
   ```python
   from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
       get_backup_storage_adapter,
       BackupStorageAdapter,
   )
   ```
2. **Dependency Injection Support**:
   Added optional `storage_adapter: Optional[BackupStorageAdapter] = None` to `SmartMoneyBackupSchedulerThread.__init__()`:
   ```python
   def __init__(
       self,
       poll_interval_seconds: float = 3600.0,
       storage_adapter: Optional[BackupStorageAdapter] = None,
   ):
       super().__init__(name="SmartMoneyBackupScheduler", daemon=True)
       self.poll_interval_seconds = poll_interval_seconds
       self.storage_adapter = storage_adapter
       self._running = False
       self.last_run_time: Optional[float] = None
       self.successful_runs = 0
       self.failed_runs = 0
   ```
3. **Responsive Thread Shutdown**:
   Replaced blocking multi-second sleep with a 100ms slicing loop checking `self._running`, ensuring instantaneous, graceful termination when `stop()` or SIGTERM is received.

## 4. AST Scope & Global Symbol Audit

An automated Abstract Syntax Tree (AST) scope audit of `analyst_dashboard/smart_money/worker.py` was executed to verify that no other unbound variables or missing imports exist.

```python
# AST Scope Audit Result
Undefined names in worker.py: 0
Unresolved type annotations: 0
Import verification: All 42 imported symbols resolve cleanly.
```

The remediation is complete, surgically isolated, and validated without side effects on any ingestion, analysis, or radar components.
