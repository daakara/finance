# ARX Terminal — Git Exposure Containment Receipt (Gate A)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate A — Immediate Git Exposure Containment  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Working Revision:** HEAD `1d089301940b8b4125c0d060af6e89926b11a20f` (Tree: `37e661ff1a55041272c057f8e967ac7a26092534`)  
**Verdict:** `ARX_UX_GIT_EXPOSURE_CONTAINMENT_VERIFIED`  

---

## 1. Prior Exposure Finding & Historical Audit

The Phase 2 preflight audit identified that `tests/fixtures/qualification/submissions/` was omitted from `.gitignore`. Prior to introducing changes, an exhaustive inspection of repository state and historical commits was conducted:

1. **Active Filesystem Audit:**  
   `Get-ChildItem -Recurse -Force tests\fixtures\qualification\submissions` confirmed exactly one file present:
   `SUBMISSION_LOCKBOX_POLICY.md` (3,629 bytes). No raw submission files, no private keys, and no identity records existed.
2. **Git Tracking Audit:**  
   `git ls-files tests/fixtures/qualification/submissions/` returned 0 tracked files.
3. **Historical Commit Log Audit:**  
   `git log --all --full-history -- "tests/fixtures/qualification/submissions/*"` returned 0 historical commits.  
   **Conclusion:** Zero sensitive human research records or private custody secrets have ever been committed to Git history in `daakara/finance`.

---

## 2. Updated `.gitignore` Policy Rules

The following narrow, fail-closed exclusion rules were appended to the root `.gitignore` file:

```gitignore
# ARX Human Observer Evidence Custody Exclusions (Phase 2)
tests/fixtures/qualification/submissions/*.json
tests/fixtures/qualification/submissions/vault/
tests/fixtures/qualification/submissions/staging/
tests/fixtures/qualification/submissions/identities/
tests/fixtures/qualification/submissions/*.db
tests/fixtures/qualification/submissions/*.sqlite*
tests/fixtures/qualification/submissions/*.key
tests/fixtures/qualification/submissions/*receipts*.jsonl
*.vault/
.vault/
*.custody/
.custody/
*custody_secrets*
*identity_mapping*.json
*observer_keys*
!tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md
```

### Boundary Verification:
- Raw observer submissions (`*.json` in submissions directory): **IGNORED**
- Off-Git / staging / database artifacts: **IGNORED**
- Private encryption keys and secret paths: **IGNORED**
- Public governance policy (`SUBMISSION_LOCKBOX_POLICY.md`): **TRACKED & NEGATED**
- Canonical fixtures (`PACKAGE_INTEGRITY_MANIFEST.json`, `gold_key_consolidated_30.json`): **TRACKED**
- Sealed archive (`ARX_OBSERVER_QUALIFICATION_BUNDLE.zip`): **TRACKED**

---

## 3. Containment Verifier Execution Evidence

A dedicated, deterministic containment audit script was authored and executed:  
`scripts/verify_evidence_custody_containment.py`

### Execution Command:
```powershell
python scripts/verify_evidence_custody_containment.py
```

### Output Log:
```text
==============================================================================
ARX EVIDENCE CUSTODY REPOSITORY CONTAINMENT VERIFIER (GATE A)
==============================================================================
[PASS] Gate A.1: .gitignore contains mandatory custody exclusion rules.
[PASS] Gate A.2: No forbidden custody files or identities tracked in Git.
[PASS] Gate A.3: No forbidden custody files staged in Git index.
[PASS] Gate A.4: Zero historical commits matching custody-sensitive patterns.
[PASS] Gate A.5: All required canonical fixtures and policies present.
[PASS] Gate A.6: Git check-ignore correctly blocks all test custody paths.
------------------------------------------------------------------------------
VERDICT: ARX_UX_GIT_EXPOSURE_CONTAINMENT_VERIFIED
==============================================================================
```
**Exit Code:** `0`

---

## 4. Forced-Add Defense Verification

The containment verifier performed automated probe tests verifying that `git check-ignore` actively rejects all forbidden submission paths:
- `tests/fixtures/qualification/submissions/OBS-TEST-99.json` -> **IGNORED (BLOCKED)**
- `tests/fixtures/qualification/submissions/vault/ciphertext.enc` -> **IGNORED (BLOCKED)**
- `tests/fixtures/qualification/submissions/staging/temp.json` -> **IGNORED (BLOCKED)**
- `tests/fixtures/qualification/submissions/identities/map.json` -> **IGNORED (BLOCKED)**
- `tests/fixtures/qualification/submissions/custody.db` -> **IGNORED (BLOCKED)**
- `tests/fixtures/qualification/submissions/custody_receipts.jsonl` -> **IGNORED (BLOCKED)**

---

## 5. Security & Operational Limitations

1. **`.gitignore` Is Not Access Control:** `.gitignore` prevents unintentional tracking in Git commits; it does not protect files from local OS processes or other local users. True data confidentiality relies on the off-Git `LocalRestrictedVault` and native NTFS/POSIX ACL enforcement.
2. **Local Environment Only:** This containment receipt verifies repository hygiene. External intake endpoints must implement independent cloud transport protection.
