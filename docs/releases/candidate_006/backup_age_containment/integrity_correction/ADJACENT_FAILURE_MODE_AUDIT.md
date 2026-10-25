# ARX TERMINAL — SMART MONEY CANDIDATE 006
# ADJACENT FAILURE-MODE SECURITY & ARCHITECTURAL AUDIT

**Execution Mode**: `NONPRODUCTION_INTEGRITY_CORRECTION_AND_EVIDENCE_REQUALIFICATION`  
**Target Candidate**: Candidate 006 Backup-Age Containment  
**Audit Scope**: Adjacent edge cases, race conditions, and boundary failure modes introduced by or related to the integrity corrections.

---

## 1. Multi-Manifest Recovery-Point Selection

### Risk
If the freshness authority only inspects `manifest_items[0]` (the newest file by timestamp), a single failed backup attempt (which generates a manifest with `upload_verified=False` or corrupt JSON) would permanently trap the worker in `UNVERIFIABLE` / `INGESTION_SUSPENDED`, even if a valid backup taken 1 hour ago is still within the 12-hour freshness window.

### Remediation Implemented
[`freshness_authority.py`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/smart_money/storage/backup/freshness_authority.py) now implements a multi-candidate descending iteration loop:
1. Iterates through all candidate manifests sorted newest to oldest;
2. Skips corrupted or failed uploads while recording rejection telemetry;
3. Evaluates the first valid recovery point that satisfies all cryptographic and schema contracts;
4. If that verified recovery point is within the 12-hour age window, evaluates to `FRESH`; if older, evaluates to `STALE`;
5. Only returns `UNVERIFIABLE` if all candidate manifests in storage fail verification.

---

## 2. Manifest and Object Identity Substitution

### Risk
An attacker or corrupted pipeline could craft a manifest containing directory traversal sequences (e.g., `dump_filename: "../../../etc/passwd"`) or substitute an arbitrary S3 key, tricking the freshness authority into corroborating an unrelated object.

### Remediation Implemented
In `freshness_authority.py`:
- `dump_filename` is strictly checked against directory traversal: `os.path.basename(dump_filename) == dump_filename`, containing zero `/`, `\`, or null bytes;
- Filename must match the canonical pattern (`smart_money_*.dump`);
- Filename must contain the manifest's `backup_id`. Violations fail closed with `INVALID_DUMP_FILENAME_TRAVERSAL_OR_SUBSTITUTION`.

---

## 3. Backup Metadata Trust Boundary

### Risk
Treating S3 / Cloudflare R2 user metadata (`x-amz-meta-sha256`) as cryptographic tamper-proofing when the manifest and the object metadata are written using the exact same cloud credentials.

### Architectural Classification
- **Writer Corroboration**: Matching metadata verifies that the object uploaded by the backup engine matches the manifest specification and was not truncated or modified post-upload.
- **Not Independent Tamper-Proofing**: Since both the object and the manifest share the R2 API key boundary, matching metadata is **corroboration**, not out-of-band proof. Out-of-band verification is provided by the database audit ledger (`sm_audit_events`) and Level 7 tested restore.

---

## 4. Freshness Cache Overhang

### Risk
If a freshness receipt has a 60-second cache TTL, and was evaluated when the backup was at 43,180 seconds old (20s before the 43,200s limit), a naive cache check would return `is_fresh=True` for the remaining 40 seconds even though the actual backup has become stale.

### Remediation Implemented
In `freshness_authority.evaluate_freshness`:
```python
elapsed = time.monotonic() - self._last_evaluated_monotonic
if elapsed < self.policy.recheck_interval_seconds:
    if self._last_receipt.is_fresh and self._last_receipt.backup_age_seconds is not None:
        projected_age = self._last_receipt.backup_age_seconds + elapsed
        if projected_age <= self.policy.max_verified_age_seconds:
            return self._last_receipt
        # If projected_age > threshold, the cache expires immediately!
```
Furthermore, `enforce_snapshot_publication_gate()` always forces an uncached fresh evaluation (`force_refresh=True`).

---

## 5. Fail-Closed Configuration Parsing

### Risk
If environment variables such as `ARX_BACKUP_MAX_VERIFIED_AGE_SEC` contain invalid, negative, zero, or non-finite values (e.g. `"-100"`, `"0"`, `"NaN"`, `"inf"`), a weak parser might fall back to a default, silently changing safety posture.

### Remediation Implemented
Implemented `_parse_safety_float()`:
- Non-numeric strings raise `ValueError`;
- `NaN`, `inf`, `-inf` raise `ValueError`;
- Non-positive values for `max_age` and `timeout` raise `ValueError`;
- Negative values for `interval` and `skew` raise `ValueError`.
Verified across 7 dedicated adversarial vectors in `test_smart_money_backup_containment_adversarial.py`.

---

## 6. Worker Ownership & Handover Concurrency

### Risk
Does the 10-second handover retry loop allow two containers to own SEC egress simultaneously during Railway rolling restart?

### Verification
In PostgreSQL:
- `pg_try_advisory_lock(401007311509093371)` is atomic and strictly mutually exclusive in PostgreSQL shared memory;
- It returns `true` to exactly one session; any concurrent caller receives `false`;
- The retry loop in `acquire_egress_authority` retries 5 times at 2.0s intervals. If the old container has not terminated after 10s, the new container fails closed (`sys.exit(1)`).
- At no point during the 10s window can two workers simultaneously acquire the lock.
