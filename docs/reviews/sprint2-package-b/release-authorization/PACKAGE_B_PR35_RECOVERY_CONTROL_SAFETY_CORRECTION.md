# ARX TERMINAL — SPRINT 2 PACKAGE B
# PR #35 RECOVERY-CONTROL SAFETY CORRECTION & MERGE SAFETY GATE

## 1. Governing Authority & Executive Summary

- **Governing Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Pull Request**: [daakara/finance#35](https://github.com/daakara/finance/pull/35)
- **Base Branch**: `main` (`f6c5755126059b2f227e28ce90ddaad43997884c`)
- **Remote PR HEAD**: `695cd986f10b46d315ada8afa13058e81f4b4652`
- **GitHub CI Status**: [Run #39566286992](https://github.com/daakara/finance/actions/runs/39566286992) — **SUCCESS (100% Green)**
- **Operating Gate**: `G-S2-ACT`
- **VCP Holdout Gate**: `G-VCP-012 = ACTIVATION_DEFERRED`
- **Operational Disposition**:
  $$\mathbf{PACKAGE\_B\_PR35\_RECOVERY\_SAFETY\_CORRECTED\_LOCAL\_HELD}$$

This engineering pass corrects the architectural discrepancy between documented guarded-recovery guarantees and implementation in PR #35 without altering the historical SIP ingestion adapter or invalidating previous CI passes.

---

## 2. Recovery Authorization Contract: Audit Reference vs. Credential

### 2.1 Reality of the Authorization Tokens
The tokens `incident_id`, `operator_authorization`, and `maintenance_mode_token` are **operational audit references** (structured correlation identifiers), **NOT cryptographic bearer credentials or secrets**.

| Parameter | Type | Purpose | Conferred Authority |
| :--- | :--- | :--- | :--- |
| `incident_id` | Audit Reference | Correlates recovery execution with a tracked incident (e.g. `INC-G-S2-PILOT-001`). | None (Identifier only) |
| `operator_authorization` | Governance Token | Correlates recovery execution with a formal operational release gate (e.g. `G-S2-ACT`). | None (Correlation only) |
| `maintenance_mode_token` | Isolation Token | Proves the operator declared maintenance isolation for this specific incident (e.g. `MAINT-INC-...`). | None (Correlation only) |

### 2.2 Where Authority Actually Resides
Authority to replace the production SQLite database is NOT conferred by possession of strings. Real recovery safety and authorization are enforced by:
1. **Zero Active Connection Handle Fence**: Proving zero open file descriptors across all system processes (`/proc/*/fd` on Linux/POSIX) and zero active database readers/writers before unlinking.
2. **Exclusive Quiescence Lock Probe**: Verifying that a write transaction (`BEGIN EXCLUSIVE`) succeeds before attempting replacement.
3. **Pre-flight Cryptographic Verification**: Verifying the backup file exists, passes `PRAGMA integrity_check`, and matches the expected SHA-256 digest BEFORE any modification or unlinking of the target store.
4. **Human Operational Isolation**: Physical application service isolation ensuring no incoming web requests or background workers can access the database volume.

### 2.3 Elimination of Hardcoded CLI Defaults
- **Removed**: Default value `"MAINT-G-S2-ACT-PILOT-001"` in `scripts/guarded_market_recovery.py` and `analyst_dashboard/data/guarded_recovery.py`.
- **Enforced**: `--maintenance-token` is now an explicit required command-line argument (`required=True`).
- **Validation**: Generic placeholders (`"MAINT-"`, `"MAINT-DEFAULT"`, `"MAINT-TEST"`, `"TODO"`) and tokens matching `incident_id` are strictly rejected with `ValueError`.

---

## 3. Receipt Integrity: Content-Hashed Audit Records vs. PKI

### 3.1 Audit Findings on Receipt Generation
- Inspection of `analyst_dashboard/data/guarded_recovery.py` confirms that `GuardedRecoveryReceipt` is serialized as plain JSON.
- No public/private key infrastructure (PKI), asymmetric keys (Ed25519/RSA), or HMAC secrets exist in the repository for signing receipts.
- Describing these receipts as "cryptographically signed" or "digitally signed" was inaccurate documentation.

### 3.2 Corrected Architecture: Deterministic Content Hashing
- `GuardedRecoveryReceipt` now includes a `receipt_sha256` field computed via `compute_receipt_sha256()`.
- The hash is a deterministic SHA-256 digest of the canonical JSON representation of the complete execution record (excluding `receipt_sha256` itself).
- Provides **tamper-evident audit integrity**: Any modification to receipt timestamps, quarantined file records, pre/post metadata, or status will invalidate the content hash.
- Documentation accurately states that receipts are **content-hashed deterministic audit records**, not digital signatures.

---

## 4. Recovery Sequence Ordering & Failure Isolation

### 4.1 Previous Flaw: Late Backup Verification
In the initial implementation, Step 6 (Quarantine affected database) unlinked and renamed `/root/.finance_market_store.db` *before* Step 7 (Verify backup integrity). If the backup was corrupt or had an invalid hash, the production database was already unlinked and quarantined, leaving the target store missing.

### 4.2 Corrected 10-Step Execution Sequence
The recovery sequence in `analyst_dashboard/data/guarded_recovery.py` has been reordered and reinforced:

```
[1. Verify Operator Authorization] (Incident & gate format validation)
             │
[2. Verify Maintenance Mode Token] (Explicit, non-generic, distinct from incident ID)
             │
[3. CRITICAL PRE-FLIGHT: Verify Backup Integrity] (SHA-256 check & PRAGMA integrity_check)
             │  ↳ If backup invalid: ABORT IMMEDIATELY. Target DB is 100% untouched.
             │
[4. Verify Reader/Writer Handle Fence] (Proves zero open file descriptors)
             │
[5. Verify Quiescence Write Lock Probe] (Exclusive lock transaction)
             │
[6. Non-Destructive Evidence Quarantine] (.db, .db-wal, .db-shm moved to quarantine dir)
             │
[7. Staged Candidate Restoration] (Copy backup to .restore_candidate and verify integrity)
             │
[8. Atomic Replacement] (Replace candidate to target path)
             │  ↳ If swap fails: AUTOMATIC ROLLBACK restores quarantined target DB files!
             │
[9. Readback & Schema Verification] (Query row counts, tables, and provenance)
             │
[10. Complete Receipt Generation] (Deterministic SHA-256 content hashing)
```

### 4.3 Automated Post-Quarantine Rollback Protection
If any failure occurs after quarantine (during candidate staging or atomic swap), `_rollback_quarantine_to_target()` automatically copies the quarantined `.db`, `.db-wal`, and `.db-shm` files back to their original target paths, ensuring zero data loss while preserving the quarantine directory for forensics.

---

## 5. Verification & Test Suite Results

### 5.1 Test Suite Summary
5 new targeted unit and adversarial tests were added to `tests/test_pilot_identity_and_guarded_recovery.py`:
1. `test_recovery_refused_with_malformed_or_placeholder_tokens`: Proves refusal on empty, placeholder, or generic tokens.
2. `test_preflight_backup_validation_failure_preserves_target_database_unquarantined`: Proves target DB is untouched when backup hash fails.
3. `test_recovery_failure_post_quarantine_rolls_back_original_database_files`: Proves automated rollback restores target DB if swap fails.
4. `test_guarded_recovery_receipt_deterministic_content_hash`: Proves deterministic SHA-256 content hashing and tamper sensitivity.
5. `test_cli_requires_explicit_maintenance_token`: Proves CLI rejects execution without explicit `--maintenance-token`.

### 5.2 Execution Results (Release Worktree)
```bash
pytest tests/test_package_b_deployment_rehearsal.py tests/test_pilot_identity_and_guarded_recovery.py tests/test_alpaca_historical_sip_ingestion.py
```
- **Passed**: `99 / 99` tests (100% green in 9.96s).
- **Baseline Invariant**: All 94 existing deployment rehearsal and SIP ingestion tests continue to pass.
- **Cross-Boundary Regressions**: 159 CI tests verified green in GitHub Actions Run #39566286992. Total verified test corpus: 258 tests.

---

## 6. Cryptographic Artifact Digest Ledger

| Component | Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- |
| **Ingestion Adapter** | `scripts/ingest_alpaca_historical.py` | `7DA7CFC6BE8717C86EC15568F6D16E7173A44A7F33084653E889800F80C273C7` | **100% UNCHANGED (INVARIANT)** |
| **Pilot Manifest V3** | `docs/.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | **100% UNCHANGED (INVARIANT)** |
| **Pilot Manifest V2** | `docs/.../PILOT_PROPOSED_MANIFEST.json` | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` | **100% UNCHANGED (INVARIANT)** |
| **Guarded Recovery Engine** | `analyst_dashboard/data/guarded_recovery.py` | `217377EC4BFF02C86B7B7122D39FC77A4112820B9F4BCA2E8C077888EA0DA684` | Corrected (466 lines) |
| **Recovery CLI Script** | `scripts/guarded_market_recovery.py` | `7B0F79B6F57BB94E5D5CA77B99D5B7237CD54B1E5FDF5A317429701F81917D82` | Corrected (73 lines) |
| **Acceptance Test Suite** | `tests/test_pilot_identity_and_guarded_recovery.py` | `E02FF5235D186E3C3318B411F2FD1FF2E8470D9819D3618F3C723379419949D8` | Expanded (642 lines, 20 tests) |

---

## 7. Strict Gating Boundaries & Production Holds

- **Remote Branch Push**: **`HELD`** (Zero push commands executed; remote branch remains at commit `695cd986...`).
- **PR #35 Amendment**: **`HELD`** (Awaiting explicit operator authorization).
- **PR Merge into Main**: **`NOT_AUTHORIZED`**.
- **Railway Container Deployment**: **`NOT_AUTHORIZED`** (Active deployment `4aa235d8-b571-4f67-bb41-2cdfa103fa00` unchanged).
- **Database Mutation**: **`0 (ZERO)`** writes executed to production or local stores.
- **Pilot Ingestion Execution**: **`NOT_AUTHORIZED`**.
- **AAPL Tail Ingestion Hold**: **`BLOCKING_FOR_INGESTION`** (`AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL` preserved).
- **VCP Gate 12**: **`ACTIVATION_DEFERRED`**.

---

## 8. Explicit Recommendation for Operator Action

Two distinct pathways are available to the operator:

### Option A (Recommended): Update PR #35 with Corrective Commit
- Authorize pushing the corrective commit to `fix/sprint2-package-b-canonical-ingestion-release`.
- Updates PR #35 with explicit token requirements, pre-flight backup validation, automated quarantine rollback, and 5 new adversarial tests.
- Re-triggers GitHub CI to confirm 100% green status on the final tightened implementation.

### Option B: Merge PR #35 As-Is, Land Recovery Corrections in Follow-Up
- Since PR #35 at `695cd986...` already passed 100% of CI checks, the operator may choose to merge PR #35 as-is.
- Land the recovery-control hardening in a focused subsequent PR (`fix/guarded-recovery-hardening`).
- Note: Standalone recovery CLI `guarded_market_recovery.py` is not automatically executed on deployment, making either option safe. Option A is recommended for complete codebase cleanliness.
