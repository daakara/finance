# ARX TERMINAL — SPRINT 2 PACKAGE B
# PRODUCTION CODE RELEASE RUNBOOK & ROLLBACK PLAN

## 1. Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Gate**: `G-S2-ACT`.
- **VCP Holdout Authority**: `G-VCP-012 = ACTIVATION_DEFERRED` (Independently governed; production historical ingestion does NOT require Gate 12).
- **Subject**: Release procedure for Package B canonical code release.
- **Current Execution State**: Code release preparation only; deployment execution is `NOT_AUTHORIZED`.
- **Ingestion Execution State**: `NOT_AUTHORIZED` (Blocked by `AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL`).

---

## 2. Release Pathway Specification

### 2.1 Clean Source Branch & Commit Plan
- **Branch**: `fix/sprint2-package-b-canonical-ingestion-release`
- **Base Commit**: `origin/main` (`f6c5755126059b2f227e28ce90ddaad43997884c`)
- **Staged File Inventory**:
  1. `scripts/ingest_alpaca_historical.py` (Modified, +107 / -24)
  2. `analyst_dashboard/data/guarded_recovery.py` (New, 377 lines)
  3. `scripts/guarded_market_recovery.py` (New, 73 lines)
  4. `tests/test_pilot_identity_and_guarded_recovery.py` (New test, 470 lines)
  5. `tests/test_package_b_deployment_rehearsal.py` (New test, 385 lines)
  6. `docs/reviews/sprint2-package-b/` (Release documentation & V3 manifest)
- **Commit Message Convention**:
  `fix(package-b): canonical historical SIP ingestion adapter and guarded recovery controls (#34)`

### 2.2 CI Gates & Protected Branch Controls
- **Required Pull Request**: Open PR from `fix/sprint2-package-b-canonical-ingestion-release` into `main`.
- **Required CI Pass**:
  - `pytest tests/test_package_b_deployment_rehearsal.py tests/test_pilot_identity_and_guarded_recovery.py tests/test_alpaca_historical_sip_ingestion.py` (All green).
- **Required Approver**: Repository administrator and Sprint 2 Lead under authority `G-S2-ACT`.
- **Merge Strategy**: Squash-and-merge or merge commit per repository policy.

### 2.3 Railway Deployment Configuration & Trigger Audit
- **Platform**: Railway
- **Project**: `tranquil-radiance` (`a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
- **Environment**: `production` (`11cad5e1-358e-430e-84fc-189953cff48a`)
- **Service**: `web` (`56e38249-b78c-4a60-80d7-303ce7451bd8`)
- **Deployment Trigger Analysis**:
  - **Feature Branch Push**: Pushing `fix/sprint2-package-b-canonical-ingestion-release` does **NOT** trigger production deployment.
  - **PR Creation**: Opening a pull request does **NOT** trigger production deployment (PR preview environments are not enabled on `web`).
  - **Merge to Main**: Merging the PR into `main` **WILL** trigger an automated container build and deployment on Railway.

---

## 3. Pre-Deployment Zero-Ingestion Baseline Attestation

Prior to any code deployment, the exact logical contents of the production SQLite database (`/root/.finance_market_store.db`) were captured via read-only inspection:

- **Deployment ID**: `4aa235d8-b571-4f67-bb41-2cdfa103fa00`
- **Total `asset_ohlcv_daily` Rows**: `36,673`
- **Total `asset_ohlcv_provenance` Rows**: `287`
- **Database File Location**: `/root/.finance_market_store.db` (WAL mode, `web-volume`)

### Four-Symbol Pre-Deployment Logical Baseline

| Symbol | Candle Rows | Provenance Rows | Earliest Date | Latest Date | Candle Logical SHA-256 | Provenance Logical SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | 289 | 0 | `2025-08-27` | `2026-10-20` | `BD96272C784A6CD1E85016A9878BC4C91B46CA68BA164A87E97CE6BB5F62124E` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |
| **GOOG** | 287 | 287 | `2025-08-27` | `2026-10-16` | `C149DE4587DC9C1CB8021C09C3AA61D03F17F9A83C972E34D8ABE454583EEBB0` | `27A9A825EF160C94519D4BFA5B83C1F176D077285033A6B1D8B6336F4A2DE4B2` |
| **GOOGL** | 288 | 0 | `2025-08-27` | `2026-10-19` | `5CFA99858CFBCE97339B51CB448176459668543E4C2CB26A3C91F5FFD667D496` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |
| **ABBV** | 0 | 0 | None | None | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |

*Note: Empty set digest `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` corresponds to SHA-256 of `[]`.*

### Ingestion Holding Record: AAPL Tail Out-of-Scope
```
AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION
```
- Production baseline contains trade date `2026-10-20` for AAPL (unprovenanced).
- Canonical V3 manifest window is `2025-08-27` to `2026-10-19`.
- Current adapter fail-closed check (`outside_unprovenanced > 0`) detects this session and rejects ingestion with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` with zero database writes.
- Manifest V3 is NOT automatically expanded; the production row is NOT deleted or quarantined; V3 digest (`0F54C702...`) is preserved.
- Blocks production historical ingestion, NOT code release.

---

## 4. Post-Deployment Verification Protocol

Immediately upon container deployment completion, the operator executes:

### Step 1: Health Check Probe
- Issue HTTP GET:
  `curl -s https://web-production-470560.up.railway.app/api/v1/governance/status`
- Expected: HTTP 200 OK with `isProductionRuntime: true`, persistent storage valid on `/root`.

### Step 2: Container Script Parity Check
Execute via Railway CLI read-only command:
```bash
railway ssh --service web "python3 -c \"import hashlib; from pathlib import Path; print('ADAPTER_SHA256:', hashlib.sha256(Path('/app/scripts/ingest_alpaca_historical.py').read_bytes()).hexdigest().upper()); print('RECOVERY_SHA256:', hashlib.sha256(Path('/app/analyst_dashboard/data/guarded_recovery.py').read_bytes()).hexdigest().upper())\""
```
**Expected Output**:
- `ADAPTER_SHA256`: `7DA7CFC6BE8717C86EC15568F6D16E7173A44A7F33084653E889800F80C273C7`
- `RECOVERY_SHA256`: `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D`

### Step 3: Logical Zero-Ingestion Verification
Compare the post-deployment logical digests against the baseline in Section 3:
```bash
railway ssh --service web "python3 -c \"import sqlite3, hashlib, json; conn = sqlite3.connect('file:/root/.finance_market_store.db?mode=ro', uri=True); cur = conn.cursor(); rows = cur.execute('SELECT trade_date, open, high, low, close, volume FROM asset_ohlcv_daily WHERE symbol = \\\"AAPL\\\" ORDER BY trade_date;').fetchall(); digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest().upper(); print('AAPL_DIGEST_MATCH:', digest == 'BD96272C784A6CD1E85016A9878BC4C91B46CA68BA164A87E97CE6BB5F62124E'); conn.close()\""
```
**Expected Output**:
- `AAPL_DIGEST_MATCH: True`
- All symbol candle and provenance logical digests must match Section 3 exactly, proving zero logical data mutation.

---

## 5. Granular Code-Only Rollback Protocol

If application startup fails or parity checks fail, rollback must be executed per the exact merge method used:

### Case A: Squash Merge Rollback
If the PR was merged via **Squash and Merge**:
```bash
git checkout main
git pull origin main
git revert <squash_merge_commit_sha>
# Commit message: Revert "fix(package-b): canonical historical SIP ingestion adapter..."
git push origin main
```
*Effect*: Reverts the single squashed commit. Railway triggers a clean automated redeployment of the reverted code.

### Case B: Ordinary Commit Rollback
If merged via rebase / fast-forward single commit:
```bash
git checkout main
git pull origin main
git revert <commit_sha>
git push origin main
```
*Effect*: Creates a clean revert commit.

### Case C: True Merge Commit Rollback
If merged via standard merge commit with multiple parents:
```bash
git checkout main
git pull origin main
git revert -m 1 <merge_commit_sha>
# Flag -m 1 specifies parent 1 (mainline branch) as the baseline to retain
git push origin main
```
*Effect*: Reverts the merge commit changes relative to the main branch parent.

### Operational Safety Rule for Rollback
> **MANDATORY**: Any git push to `main` to execute a rollback requires a separate explicit operator authorization token (`G-S2-ACT-ROLLBACK-001`). Automated rollback scripts without human operator authorization are strictly forbidden.

### Railway Container Fallback
If immediate git revert cannot be pushed, the operator can instantly roll back from the Railway UI:
- Navigate to Railway -> `tranquil-radiance` -> `production` -> `web` -> Deployments.
- Click deployment `4aa235d8-b571-4f67-bb41-2cdfa103fa00` -> Click "Redeploy".
- Rollback takes < 30 seconds with zero data loss (persistent database volume `/root` is untouched).
