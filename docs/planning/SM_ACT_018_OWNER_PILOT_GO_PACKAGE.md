# ARX TERMINAL — SM-ACT-018
## Final Product Owner GO Decision Package: Qualified Ten-Issuer SEC Ingestion Pilot

**Directive:** `SM-ACT-018: Ten-Issuer Pilot Runner Integration and Final Owner GO Qualification`  
**Tracker:** GitHub Issue #165 (`daakara/finance`)  
**Production Release Baseline:** [`9850dae57ab48b010b15930c61a96aec0d7875ad`](file:///c:/Users/akara/Documents/Projects/finance) (PR #170)  
**Mainline Anchor:** [`ae314184b3e5833df152b4a5d6f5060c576ac0c9`](file:///c:/Users/akara/Documents/Projects/finance) (PR #169)  
**Dedicated Worker Deployment:** `9fa15ef3-e33c-46e5-92a7-1046df1406c8` (`smart-money-worker` on Railway, Status: `SUCCESS`, `STANDBY`)  
**Dedicated Web Deployment:** `bb5bf9fb-9746-4e9d-a69c-959369b21906` (`web` on Railway, Status: `SUCCESS`)  
**Database Topology:** Production PostgreSQL 16 (Railway `finance_smart_money`)  
**Status:** **READY FOR PRODUCT OWNER GO DECISION** (Integration and non-live qualification 100% complete; live SEC egress blocked pending explicit Owner GO)

---

## 1. Verified Executable & Source Integrity

The dedicated production sequential pilot runner is integrated and qualified across two equivalent executable entrypoints:

1. **Direct Module Invocation:**
   ```bash
   python -m analyst_dashboard.smart_money.pilot_runner
   ```
2. **Dedicated Worker CLI Flag:**
   ```bash
   python -m analyst_dashboard.smart_money.worker --ten-issuer-pilot
   ```

### 1.1 Source Inventory & Checksums
- `analyst_dashboard/smart_money/pilot_runner.py`:
  * Class: [`TenIssuerPilotRunner`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py)
  * Function: [`run_ten_issuer_pilot`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/pilot_runner.py)
  * Standalone CLI: `if __name__ == "__main__":` entrypoint with logging, JSON telemetry formatting, and process exit codes.
- `analyst_dashboard/smart_money/worker.py`:
  * CLI Flag: `--ten-issuer-pilot` added to argparse, dispatching to `run_ten_issuer_pilot()`.
- `tests/test_smart_money_ten_issuer_pilot_runner.py`:
  * 13 adversarial and invariant tests (**13/13 PASSED**).
- `scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt_template.json`:
  * Canonical receipt template specifying per-issuer outcomes, transaction classification, persisted evidence, coverage, and teardown verification.

---

## 2. Governed Pilot Execution Parameters & Boundaries

| Parameter | Governed Value | Mechanical Enforcement Mechanism |
| :--- | :--- | :--- |
| **Universe Scope** | 10 Governed Mega-Caps: `NVDA`, `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`, `TSLA`, `BRK-B`, `JPM`, `V` | Hardcoded in `pilot_runner.py` / `DEFAULT_GOVERNED_ISSUERS` |
| **Request Ceiling (Total)** | $\le 40$ HTTPS requests | Tracked HTTP boundary counter; raises `PilotBudgetExceededError` if exceeded |
| **Request Ceiling (Per-Issuer)**| $\le 4$ HTTPS requests | Tracked per-issuer counter; raises `IssuerBudgetExceededError` and isolates failure |
| **Redirect Accounting** | $1 + \text{len(history)}$ hops | Every intermediate 3xx redirect explicitly incremented against issuer and total budgets |
| **Cycle Duration Ceiling** | $\le 180.0$ seconds | Wall-clock deadline check before every HTTP call (`PilotTimeoutError`) |
| **Rate Limit** | $\le 5.0$ requests/second | Enforced by `SecRateLimiter` token bucket (`max_per_second=5.0`, SEC cap is 10) |
| **Inter-Issuer Delay** | $250\text{ms}$ pause | Polite spacing between issuer metadata calls |
| **Exclusive Ownership** | Lock ID `401007311509093371` | PostgreSQL session-level advisory lock (`LOCK_SEC_EGRESS_OWNER`) |
| **Snapshot Protection** | Immutability on partial coverage | Active snapshot updated ONLY IF 10/10 succeed |
| **Egress Role Teardown** | Restores `STANDBY` in `finally:` | Guaranteed in-process and persistent teardown |
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
*(Verified active backup `sm-backup-20261102T182029Z-052cea3e` with SHA-256 digest `8b721e36...`)*.

### Step 2: Temporary Dedicated Worker Authorization
```bash
railway variables set ARX_SEC_EGRESS_ROLE=OWNER ARX_ENABLE_SEC_INGESTION=true --service smart-money-worker
```
*(Verify Web API service `web` remains completely untouched as strict non-owner).*

### Step 3: Execute Bounded Ten-Issuer Pilot Pass
Execute via Railway worker CLI:
```bash
railway run --service smart-money-worker python -m analyst_dashboard.smart_money.pilot_runner
```
*Or equivalently:*
```bash
railway run --service smart-money-worker python -m analyst_dashboard.smart_money.worker --ten-issuer-pilot
```

### Step 4: Immediate Safe Teardown
```bash
railway variables set ARX_SEC_EGRESS_ROLE=STANDBY ARX_ENABLE_SEC_INGESTION=false --service smart-money-worker
```
Verify worker startup log confirms return to fail-closed `STANDBY` loop.

### Step 5: Post-Execution Receipt Generation & API Verification
1. Capture execution output into `scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt.json` using the template.
2. Probe live Web API endpoint:
```bash
curl -s https://web-production-470560.up.railway.app/api/v1/smart-money/evidence/NVDA
```

---

## 4. Emergency Stop & Rollback Protocols

1. **Immediate Egress Suspension:**
   - Setting `ARX_EMERGENCY_STOP=true` or resetting `ARX_SEC_EGRESS_ROLE=STANDBY` immediately denies outbound HTTPS requests.
   - Advisory lock `401007311509093371` is released in `finally:` upon process termination.
2. **Snapshot Integrity Rollback:**
   - If partial coverage occurs, active snapshot `SNAP_SM_1792939989_48f30db6` is preserved unchanged.
3. **Database Restoration:**
   - Verified R2 production backup `sm-backup-20261102T182029Z-052cea3e` is restorable within 5 minutes.

---

## 5. Decision Gate Verdict

```text
================================================================================
ARX TERMINAL — DIRECTIVE SM-ACT-018 BINDING EXIT VERDICT
================================================================================

SMART_MONEY_TEN_ISSUER_PILOT_READY_FOR_OWNER_GO
- Production Release Baseline: 9850dae57ab48b010b15930c61a96aec0d7875ad
- Dedicated Pilot Executable: analyst_dashboard/smart_money/pilot_runner.py
- Worker CLI Integration: analyst_dashboard/smart_money/worker.py (--ten-issuer-pilot)
- Adversarial Test Suite: tests/test_smart_money_ten_issuer_pilot_runner.py (13/13 PASSED)
- Execution Receipt Template: scripts/release/receipts/sm_act_018_ten_issuer_pilot_receipt_template.json
- Production Egress Status: STANDBY (Awaiting Owner GO)
================================================================================
```
