# Railway Rehearsal Runtime Identity Verification

## 1. Environment & Service Registry

- **Railway Project**: `tranquil-radiance` (`a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
- **Authorized Rehearsal Environment**: `smart-money-rehearsal` (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)
- **Worker Service**: `smart-money-worker-rehearsal` (`929874a7-bcd9-41b3-9bc7-d5cd7720f7bf`)
- **Web Service**: `smart-money-web-rehearsal` (`6b53ad3f-0d59-4a7f-83dc-8731efe48cf5`)

---

## 2. Deployment SHA & Runtime Parity Matrix

| Service | Active Deployment ID | Created At (UTC) | Status | Runtime Git Commit SHA | Code Parity vs `1cc9fa8` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Worker** | `e116ba9c-2fc0-47bb-8efa-fe3cac69f99f` | `2026-10-14T10:17:30.302Z` | `SUCCESS` | `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf` | **100% Identical** |
| **Web** | `0c35ee64-2ea8-468a-b77b-223907bd9f36` | `2026-10-14T10:17:30.301Z` | `SUCCESS` | `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf` | **100% Identical** |

### Code Parity Reconciliation
The active deployment was triggered automatically by Railway's GitHub integration when documentation commit `1693e2c0f39cb1276e5d44b22673b0db89c5f8bf` was pushed to remote branch `arx/smart-money-candidate-006-backup-age-containment`.
- `git diff 1cc9fa8 1693e2c -- analyst_dashboard/ tests/` confirmed exactly **0 code changes**.
- The application codebase executing in container `e116ba9c-...` is byte-for-byte identical to Candidate 006 hotfix commit `1cc9fa886d28908270df4794e64536d4cb1948e2`.

---

## 3. Worker Process & Advisory Lock Health

- **Process Liveness**: Worker PID 2772 active and stable.
- **Worker State**: `OWNER_LOCK_ACQUIRED`.
- **Primary Advisory Lock**:
  `SEC_EGRESS_OWNER` (`401007311509093371`) held exclusively.
- **Lock Contention Handover**:
  During container startup at `10:17:56 UTC`, the new container cleanly waited 4 retry cycles (8.0s) for the outgoing worker (`89d26e9b-...`) to release its session lock, successfully acquired the lock at `10:18:04 UTC`, and avoided split-brain worker execution.
- **Crash / Restart Analysis**:
  Zero container crashes or unhandled exceptions observed.

---

## 4. Web Service Health

- **Local Probe**: `curl http://127.0.0.1:8000/health` inside container returned:
  ```json
  {"status": "online"}
  ```
- **Active Snapshot Publication State**:
  `scanner_active_snapshots` holds `snap-final-717f17e4`, verified intact.
