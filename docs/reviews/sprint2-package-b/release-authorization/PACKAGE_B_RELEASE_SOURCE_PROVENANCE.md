# ARX TERMINAL — SPRINT 2 PACKAGE B
# RELEASE CANDIDATE SOURCE PROVENANCE REPORT

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Gate**: `G-S2-ACT`.
- **Deferred Authority**: `G-VCP-012` (separately governed).
- **Target Release Branch**: `fix/sprint2-package-b-canonical-ingestion-release`.
- **Target Release Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release`.
- **Current Production Status**: Code deployment `NOT_AUTHORIZED`; Production ingestion `NOT_AUTHORIZED`.

---

## 2. Reconstructed Source Authority & Baseline

Inspection of the repository network and Railway live production establishes:

| Metric | Upstream Production Main | Serving Railway Container | Dedicated Release Worktree |
| :--- | :--- | :--- | :--- |
| **Git SHA** | `f6c5755126059b2f227e28ce90ddaad43997884c` | `f6c5755126059b2f227e28ce90ddaad43997884c` | `f6c5755126059b2f227e28ce90ddaad43997884c` |
| **Commit Message** | `fix(data): enforce bounded SIP ingestion and verified identity audit retention (#33)` | Built from PR #33 merge commit | Branch `fix/sprint2-package-b-canonical-ingestion-release` |
| **Tracking Upstream** | `origin/main` | Railway `web` service (`tranquil-radiance`) | `origin/main` |
| **Adapter SHA-256** | `47568eea7da38edf594d908b931bc6d2fae419d7f659c573748c6219b6e1c472` | `47568eea7da38edf594d908b931bc6d2fae419d7f659c573748c6219b6e1c472` | `CCF4A96197B0AF689B119C99C8E6CE7598D74DC619835945D423657F9A1D137D` |
| **Recovery Engine** | Absent | Absent | `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D` |
| **Recovery CLI** | Absent | Absent | `73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF` |

---

## 3. Workstream Forensics & Isolation

Two prior engineering workstreams contributed to the candidate:

1. **Workstream A** (`C:\Users\akara\Documents\Projects\finance-sprint2-historical-sip-release`):
   - Branch: `fix/sprint2-historical-sip-release` (`9b18815`).
   - Contributed: `candidate_bindings` parser support, explicit `action == "REPLACE"`, and `incompatible_outside` provider check.
2. **Workstream B** (`C:\Users\akara\Documents\Projects\finance`):
   - Branch: `arx/smart-money-candidate-006-backup-age-containment` (`1d08930`).
   - Contributed: strict canonical ID format validation (`ARX_SEC_<12_HEX>`, `ASSOC_...`), noncanonical alias rejection (`sec_aapl_...`), static `is_prov_compatible`, guarded recovery engine with `/proc/*/fd` handle fencing, and `MAINT-...` token.
   - **Protection Confirmation**: All 218 staged files relating to Smart Money Candidate 006 (SEC Form 4 parser, PostgreSQL repository, etc.) remain strictly isolated in Workstream B. Zero Smart Money files were transferred.

---

## 4. Release Candidate File-by-File Inventory

The clean release candidate in `fix/sprint2-package-b-canonical-ingestion-release` consists strictly of:

| File Path | Change Type | Lines | Bytes | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/ingest_alpaca_historical.py` | MODIFIED | 1,567 | 73,674 | `CCF4A96197B0AF689B119C99C8E6CE7598D74DC619835945D423657F9A1D137D` |
| `analyst_dashboard/data/guarded_recovery.py` | NEW | 377 | 16,446 | `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D` |
| `scripts/guarded_market_recovery.py` | NEW | 73 | 2,867 | `73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF` |
| `tests/test_pilot_identity_and_guarded_recovery.py` | NEW TEST | 470 | 21,910 | `A3A2BA15DFFA7C8BEB804D771AA7249CC592C965604097ADCF4898E0A01DF203` |
| `tests/test_package_b_deployment_rehearsal.py` | NEW TEST | 385 | 16,846 | `97DEFEBDCC9D29A3A28F5A67BEE2983B4BE18A50EE51E7734C737E9E8CE7130A` |
| `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | NEW DOC | 226 | 9,972 | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` |

**Zero Unrelated Drift**: No files outside the above inventory exist in the release worktree dirty state.
