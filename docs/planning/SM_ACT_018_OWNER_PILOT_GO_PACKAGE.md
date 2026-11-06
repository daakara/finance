# ARX TERMINAL — SM-ACT-018 / SM-ACT-018R / SM-ACT-018R.1
## Final Product Owner GO Decision Package: Qualified Ten-Issuer SEC Ingestion Pilot

**Directive:** `SM-ACT-018 / SM-ACT-018R / SM-ACT-018R.1: Final One-Shot Pilot Execution Containment and Owner GO Package`  
**Tracker:** GitHub Issue #165 (`daakara/finance`)  
**Production Release Baseline:** [`87c5e939fb1434e7ee472a397a9ecb2252123dab`](file:///c:/Users/akara/Documents/Projects/finance) (PR #174)  
**Mainline Anchor:** [`54605df5074d35af4fd3cdd3e006df7040b80a56`](file:///c:/Users/akara/Documents/Projects/finance) (PR #175)  
**Dedicated Worker Deployment:** `smart-money-worker` on Railway (Status: `SUCCESS`, `STANDBY`)  
**Dedicated Web Deployment:** `web` on Railway (Status: `SUCCESS`)  
**Database Topology:** Production PostgreSQL 16 (Railway `finance_smart_money`)  
**Status:** **READY FOR PRODUCT OWNER GO DECISION** (Safety gates closed, OS process supervisor & lock boundary active, wire-boundary accounting enforced, 25/25 adversarial tests qualified; live SEC egress blocked pending explicit Owner GO)

---

## 1. Verified Executable & Source Integrity

The dedicated production sequential pilot runner is integrated and qualified across two equivalent executable entrypoints:

1. **Direct Module Invocation with Scoped Owner Authorization:**
   ```bash
   env ARX_TEN_ISSUER_PILOT_AUTHORIZED=true ARX_BACKUP_CONTAINMENT_ENABLED=TRUE python -m analyst_dashboard.smart_money.pilot_runner
   ```
2. **Dedicated Worker CLI Flag with Authorization:**
   ```bash
   python -m analyst_dashboard.smart_money.worker --ten-issuer-pilot --authorize-pilot
   ```

### 1.1 Source Inventory & Checksums
- `analyst_dashboard/smart_money/pilot_runner.py`:
  * Class: [`TenIssuerPilotRunner`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py)
  * Process Supervisor: [`PilotProcessSupervisor`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py) supervising child OS process with 180s hard wall-clock kill boundary.
  * Lock Resource Manager: [`SecAdvisoryLockBoundary`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py) RAII context manager guaranteeing connection closure and advisory lock release under all failure modes.
  * Outbound Accounting Session: [`OutboundAccountingSession`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py) intercepting wire `send()` before socket transmission.
  * Supervisory Watchdog: [`PilotSupervisoryWatchdog`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py) in-process defense-in-depth thread.
  * Function: [`run_ten_issuer_pilot`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py)
  * Standalone CLI: `if __name__ == "__main__":` entrypoint supporting `--child-worker` mode, structured JSON telemetry output, and graceful exit codes.
- `analyst_dashboard/smart_money/worker.py`:
  * CLI Flags: `--ten-issuer-pilot`, `--authorize-pilot`, `--pilot-expires-at`, `--pilot-release-sha` added to argparse, dispatching to `run_ten_issuer_pilot()`.
- `tests/test_smart_money_ten_issuer_pilot_runner.py`:
  * 25 adversarial and invariant tests (**25/25 PASSED**).
- `scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt_template.json`:
  * Canonical receipt template specifying per-issuer outcomes, transaction classification, persisted evidence, coverage, and teardown verification.

---

## 2. Governed Pilot Execution Parameters & Boundaries

| Parameter | Governed Value | Mechanical Enforcement Mechanism |
| :--- | :--- | :--- |
| **Universe Scope** | 10 Governed Mega-Caps: `NVDA`, `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `TSLA`, `BRK-B`, `JPM`, `V` | Hardcoded in `pilot_runner.py` / `DEFAULT_GOVERNED_ISSUERS` |
| **Request Ceiling (Total)** | $\le 40$ HTTPS requests | Wire-boundary outbound send interception (`OutboundAccountingSession.send()`); raises `PilotBudgetExceededError` before sending |
| **Request Ceiling (Per-Issuer)**| $\le 4$ HTTPS requests | Wire-boundary per-issuer counter; raises `IssuerBudgetExceededError` before sending and isolates failure |
| **Redirect Accounting** | $1 + \text{len(history)}$ hops | Every intermediate 3xx redirect explicitly intercepted and counted before wire send |
| **Cycle Duration Ceiling** | $\le 180.0$ seconds | External `PilotProcessSupervisor` hard kill (`SIGTERM`/`SIGKILL`) + in-process `PilotSupervisoryWatchdog` (`PilotTimeoutError`) |
| **Rate Limit** | $\le 5.0$ requests/second | Enforced by `SecRateLimiter` token bucket (`max_per_second=5.0`, SEC cap is 10) |
| **Inter-Issuer Delay** | $250\text{ms}$ pause | Polite spacing between issuer metadata calls |
| **One-Shot Authorization** | `ARX_TEN_ISSUER_PILOT_AUTHORIZED=true` | Strict prerequisite for elevation; consumed immediately in-process (`consumed`); supports expiry & SHA binding |
| **Exclusive Ownership** | Lock ID `401007311509093371` | PostgreSQL session-level advisory lock managed via `SecAdvisoryLockBoundary` RAII manager |
| **Mandatory Containment** | Verified R2 backup < 12h | Missing/disabled `ARX_BACKUP_CONTAINMENT_ENABLED` fails closed (`PilotContainmentError`) |
| **Mandatory Database** | PostgreSQL connectivity | Missing DSN or connection probe failure fails closed (`PilotSecurityError`) |
| **Snapshot Protection** | Immutability on partial coverage | Active snapshot updated ONLY IF 10/10 succeed |
| **Egress Role Teardown** | Restores `STANDBY` in `finally:` | Guaranteed in-process and persistent teardown; revokes `ARX_TEN_ISSUER_PILOT_AUTHORIZED` |
| **Operator Queue Isolation**| Zero interference | Operates independently of `sm_run_requests` operator queue |
| **Emergency Kill-Switch**| Immediate abort | `runner.abort()` or `ARX_EMERGENCY_STOP=true` halts network activity immediately |

---

## 3. Step-by-Step Production Execution Runbook

Following explicit Owner GO authorization:

### Step 1: Pre-Flight Integrity Probe
Verify R2 backup freshness is within the 12-hour fail-closed threshold:
```bash
python scripts/verify_r2_backup_freshness.py
```
*(Verified active backup within 12h threshold)*.

### Step 2: Production One-Shot Execution Command
Execute via Railway worker CLI with scoped authorization:
```bash
railway run --service smart-money-worker env ARX_TEN_ISSUER_PILOT_AUTHORIZED=true ARX_BACKUP_CONTAINMENT_ENABLED=TRUE python -m analyst_dashboard.smart_money.pilot_runner
```
*Or equivalently using the worker CLI flags:*
```bash
railway run --service smart-money-worker python -m analyst_dashboard.smart_money.worker --ten-issuer-pilot --authorize-pilot
```

*(Verify Web API service `web` remains completely untouched as strict non-owner).*

### Step 3: Verified Automatic Standby Teardown
The runner's `SecAdvisoryLockBoundary` and `finally:` blocks automatically revoke `ARX_TEN_ISSUER_PILOT_AUTHORIZED=false`, restore `ARX_SEC_EGRESS_ROLE=STANDBY`, and release the PostgreSQL advisory lock.
Confirm worker service continues normal fail-closed `STANDBY` loop.

### Step 4: Post-Execution Receipt Generation & API Verification
1. Capture execution output into `scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt.json` using the template.
2. Probe live Web API endpoint:
```bash
curl -s https://web-production-470560.up.railway.app/api/v1/smart-money/evidence/NVDA
```

---

## 4. Emergency Stop & Rollback Protocols

1. **Immediate Egress Suspension:**
   - Setting `ARX_EMERGENCY_STOP=true` or resetting `ARX_SEC_EGRESS_ROLE=STANDBY` immediately denies outbound HTTPS requests.
   - Advisory lock `401007311509093371` is released in `SecAdvisoryLockBoundary.close()` upon process termination.
2. **Snapshot Integrity Rollback:**
   - If partial coverage occurs, active snapshot `SNAP_SM_1792939989_48f30db6` is preserved unchanged.
3. **Database Restoration:**
   - Verified R2 production backup `sm-backup-20261102T182029Z-052cea3e` is restorable within 5 minutes.

---

## 5. Decision Gate Verdict

```text
================================================================================
ARX TERMINAL — DIRECTIVE SM-ACT-018R.1 BINDING EXIT VERDICT
================================================================================

SMART_MONEY_TEN_ISSUER_PILOT_READY_FOR_OWNER_GO
- Production Release Baseline: 87c5e939fb1434e7ee472a397a9ecb2252123dab
- Dedicated Pilot Executable: analyst_dashboard/smart_money/pilot_runner.py
- Process Supervision: PilotProcessSupervisor (180s hard kill boundary)
- Lock Resource Boundary: SecAdvisoryLockBoundary (RAII PostgreSQL lock cleanup)
- Wire-Level Accounting: OutboundAccountingSession (counts redirects & retries before send)
- Mandatory Containment & DB: Fail-closed on missing/disabled R2 backup or PostgreSQL
- One-Shot Elevation: Requires ARX_TEN_ISSUER_PILOT_AUTHORIZED=true (consumed in-process)
- In-Process Watchdog: PilotSupervisoryWatchdog (defense in depth)
- Worker CLI Integration: analyst_dashboard/smart_money/worker.py (--ten-issuer-pilot --authorize-pilot)
- Adversarial Test Suite: tests/test_smart_money_ten_issuer_pilot_runner.py (25/25 PASSED)
- Execution Receipt Template: scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt_template.json
- Production Egress Status: STANDBY (Awaiting Owner GO)
================================================================================
```
