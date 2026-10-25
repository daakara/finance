# ARX Terminal — Smart Money Form 4 Candidate 006
## Deployment SHA Parity & Build Evidence

**Execution Mode**: `AUTHORIZED_EXISTING_REHEARSAL_DEPLOYMENT`  
**Target Release SHA**: `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0`  
**Classification**: `OBSERVED_IN_RAILWAY`

---

### 1. Build and Deployment Verification Matrix

| Service Name | Service ID | Deployment ID | Created At (UTC) | Reported Commit SHA | Final Status | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`smart-money-worker-rehearsal`** | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `381c2cf6-79ef-46f7-ab22-47c6039b7471` | `2026-10-14T09:48:20.038Z` | `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | **`SUCCESS`** | `OBSERVED_IN_RAILWAY` |
| **`smart-money-web-rehearsal`** | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `84c77394-2e1f-48e6-9c09-f3ead0213cd7` | `2026-10-14T09:48:08.733Z` | `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` | **`SUCCESS`** | `OBSERVED_IN_RAILWAY` |

---

### 2. Live Runtime Container Evidence

The runtime commit hashes were independently inspected directly inside the running Linux containers via `railway ssh`:

#### 2.1 Web Container Runtime Check
* **Command Executed**:
  ```bash
  railway ssh -s smart-money-web-rehearsal python -c "import os; print('WEB_RUNTIME_SHA:', os.environ.get('RAILWAY_GIT_COMMIT_SHA'), 'DEPLOYMENT_ID:', os.environ.get('RAILWAY_DEPLOYMENT_ID'))"
  ```
* **Observed Output**:
  ```text
  WEB_RUNTIME_SHA: 0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0 DEPLOYMENT_ID: 84c77394-2e1f-48e6-9c09-f3ead0213cd7
  ```
* **Parity Status**: **100% Match** to Authorized Candidate SHA.

#### 2.2 Worker Container Runtime Check
* **Command Executed**:
  ```bash
  railway ssh -s smart-money-worker-rehearsal python -c "import os; print('WORKER_RUNTIME_SHA:', os.environ.get('RAILWAY_GIT_COMMIT_SHA'), 'DEPLOYMENT_ID:', os.environ.get('RAILWAY_DEPLOYMENT_ID'))"
  ```
* **Observed Output**:
  ```text
  WORKER_RUNTIME_SHA: 0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0 DEPLOYMENT_ID: 381c2cf6-79ef-46f7-ab22-47c6039b7471
  ```
* **Parity Status**: **100% Match** to Authorized Candidate SHA.

---

### 3. Build Immutability Invariants

1. **Zero Branch Advance**: Railway pulled commit `0dd2d8136e157672cdc5f9f645c4bb49f86a6bd0` explicitly; no subsequent commits existed on the branch.
2. **Zero Silently Rebuilt Images**: Both services built from source Dockerfile pinning Python 3.11 with dependencies installed cleanly.
3. **Parity Verdict**: **`DEPLOYMENT_SHA_PARITY_VERIFIED`**.
