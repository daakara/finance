# ARX TERMINAL — SMART MONEY CANDIDATE 006
## RELEASE IDENTITY & APPLICATION CODE PARITY AUDIT
**Document ID:** `ARX-DOC-SM-C006-PREFLIGHT-FINAL-PARITY-001`  
**Security Classification:** `CONFIDENTIAL — ARX INTERNAL / OWNER REVIEW`  
**Verification Timestamp:** `2026-10-14T09:34:00Z`  
**Governing Principle:** Immutable Release Identity & Byte-for-Byte Application Parity

---

### 1. Authoritative Commit & Tree Lineage Matrix

| Entity | Role / Description | Commit SHA | Tree SHA | Branch / Location | Active Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Base Candidate 006** | Initial Candidate 006 Release | `da7f6ff52b8d79658e041f935424c0fa57f42984` | `9d7b4a2...` | `arx/smart-money-candidate-006` | Preserved History |
| **Governance Amendment** | Single-owner governance amendment | `c8cc1cda0be5a2a62816d3024b6499515e97101a` | `d65e57a5865d30479d1058c7dab50d3608e26e7b` | `arx/smart-money-candidate-006-governance-amendment` | Preserved History |
| **Security Remediation** | Role de-escalation, fail-closed S3 | `9cad087868386d40dc24970be80776abb3f8756a` | `82f9903306ed3a0027549d55454d563143bbeb7b` | `arx/smart-money-candidate-006-r2-security-remediation` | Preserved History |
| **Deployed Candidate Code** | **Candidate 006 Runtime Baseline** | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `80309593afb6b1b722bed87c111bf2b58056c688` | Active in Railway Rehearsal (`7f7e4317`) | **DEPLOYED REHEARSAL** |
| **Reconciliation Baseline** | Documentation reconciliation | `38baa50f4a2da5e54c7dca6823901b0f562725ea` | `f5ddb9a...` | `origin/arx/smart-money-candidate-006-r2-security-remediation` | Pushed Commit |
| **Production Baseline** | **Untouched Active Production Web** | `bab9776b8013646b634871c49b52e61e4234d36b` | `13d14564baf94eceab215478280241668a9b1d71` | Production Env (`11cad5e1`) | **PRODUCTION ACTIVE** |

---

### 2. Application Code Parity Audit (`analyst_dashboard` & `api`)

To verify whether Candidate 006 application code has remained pristine and unmutated during documentation and test stabilization passes, a diff audit was performed against the authoritative candidate commit `76a8691339c30edc35e4e056d5b295509a81fe3a`:

```powershell
git diff 76a8691339c30edc35e4e056d5b295509a81fe3a -- analyst_dashboard api
```

#### Observed Output
```
(empty - 0 bytes returned)
```

#### Finding
- **Byte-for-Byte Parity:** **100.00%**.
- **Application Code Mutations:** **ZERO (0)** lines changed in `analyst_dashboard/` or `api/`.
- **Candidate Release Identity Preservation:** Confirmed. The exact binary and logic deployed to Railway rehearsal environment (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`) remains the exact candidate under evaluation for production authorization.

---

### 3. Isolated Test & Documentation Diff Audit

Changes between Candidate Code `76a8691` and HEAD are strictly limited to non-application boundaries:
1. `tests/test_smart_money_backup_security_adversarial.py`:
   - Injected deterministic clock fixture (`deterministic_clock`) to eliminate time-dependent fragility.
   - Added 7 explicit independent test cases for governance validation.
2. `docs/releases/candidate_006/production_preflight/`:
   - Reconciliation dossier (8 files).
   - Traceable errata updates in preflight docs.
   - Final closure dossier (6 files).

---

### 4. Railway Rehearsal Deployed State Verification

| Component | Railway Service ID | Expected Commit | Verified Runtime Commit | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Rehearsal Worker** | `c21d8ff8-e160-449e-b7ca-4b07fb880cb1` | `76a8691` | `76a8691339c30edc35e4e056d5b295509a81fe3a` | **HEALTHY / RUNNING** |
| **Rehearsal Web** | `4b63891d-0453-4dc7-a3a8-44445695029e` | `76a8691` | `76a8691339c30edc35e4e056d5b295509a81fe3a` | **HEALTHY / RUNNING** |
| **Rehearsal Database** | `e1d4b655-2260-4c57-ab13-fdf4be4dfae0` | PostgreSQL 16 | PostgreSQL 16.15 on Linux | **HEALTHY / UNPRIVILEGED** |

---

### 5. Final Release Identity Verification Verdict

- **Code Identity Verified:** `76a8691339c30edc35e4e056d5b295509a81fe3a`.
- **Tree Identity Verified:** `80309593afb6b1b722bed87c111bf2b58056c688`.
- **Verdict:** **IDENTITY_AND_PARITY_VERIFIED_PASS**.
