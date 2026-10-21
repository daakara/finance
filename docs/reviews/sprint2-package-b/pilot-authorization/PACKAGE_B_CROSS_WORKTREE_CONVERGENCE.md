# ARX TERMINAL — SPRINT 2 PACKAGE B
# CROSS-WORKTREE RELEASE CONVERGENCE REPORT

## 1. Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Sprint 2 Operational Gate**: `G-S2-ACT`.
- **Deferred VCP Holdout Authority**: `G-VCP-012` (separately governed).
- **Execution Disposition**: `PACKAGE_B_PILOT_PREFLIGHT_CONDITIONAL`.
- **Production Status**: Code deployment `NOT_AUTHORIZED`; Ingestion execution `NOT_AUTHORIZED`.

---

## 2. Workstream Identification & Forensic Comparison

Two local engineering workstreams exist for Package B:

| Attribute | Workstream A (`finance-sprint2-historical-sip-release`) | Workstream B (`finance`) |
| :--- | :--- | :--- |
| **Repository Path** | `C:\Users\akara\Documents\Projects\finance-sprint2-historical-sip-release` | `C:\Users\akara\Documents\Projects\finance` |
| **Branch** | `fix/sprint2-historical-sip-release` | `arx/smart-money-candidate-006-backup-age-containment` |
| **HEAD SHA** | `9b18815ff5f6e1995ecd2d47bb5ced8e0c430c57` | `1d089301940b8b4125c0d060af6e89926b11a20f` |
| **Upstream Tracking** | `origin/fix/sprint2-historical-sip-release` | `origin/arx/smart-money-candidate-006-backup-age-containment` |
| **Staged / Dirty Status** | `scripts/ingest_alpaca_historical.py` modified | Multiple files modified/untracked across universe, data, scanner |
| **Ingestion Adapter SHA-256** | `843BC8C54F3B9EB41184962AA58E009F45753D4413E4F793BCFBFE808B10047D` | `CCF4A96197B0AF689B119C99C8E6CE7598D74DC619835945D423657F9A1D137D` (Reconciled) |
| **Guarded Recovery Engine** | `ABSENT` | `analyst_dashboard/data/guarded_recovery.py` (`8BE7D6A6...`) + CLI (`73E94035...`) |
| **Primary Proposed Manifest** | `evidence/coverage_expansion/package_b_four_security_pilot_manifest.json` | `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json` |
| **Manifest Digest** | `65F5D9AB185056FD8C092AD1601A6DB195494063B5CCFCE562A11AAD8AA063CD` | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` |
| **Manifest Window** | `2025-08-27` to `2026-10-19` (288 sessions) | `2025-09-15` to `2026-10-19` (276 sessions - defective boundary) |

Neither worktree was reset, cleaned, merged, or overwritten.

---

## 3. Authoritative Railway Production Deployment State

Direct read-only inspection performed on live Railway production container:

- **Project**: `tranquil-radiance` (`a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
- **Environment**: `production` (`11cad5e1-358e-430e-84fc-189953cff48a`)
- **Service**: `web` (`56e38249-b78c-4a60-80d7-303ce7451bd8`)
- **Active Serving Deployment ID**: `4aa235d8-b571-4f67-bb41-2cdfa103fa00`
- **Deployment Timestamp**: `2026-10-20 15:00:29 +02:00`
- **Persistent Volume Mount**: `/root` on `web-volume` (`/root/.finance_market_store.db`, 5,427,200 bytes, journal_mode WAL)
- **Container Application Root**: `/app`
- **Container Ingestion Adapter**: `/app/scripts/ingest_alpaca_historical.py`
  - Byte length: `68,267` bytes
  - SHA-256: `47568eea7da38edf594d908b931bc6d2fae419d7f659c573748c6219b6e1c472`
  - Status: `DEPLOYED_BUT_UNPATCHED` (lacks canonical identity format validation, noncanonical alias rejection, and explicit `REPLACE` support)
- **Container Recovery Engine**:
  - `/app/analyst_dashboard/data/guarded_recovery.py`: `ABSENT` (`NOT_DEPLOYED`)
  - `/app/scripts/guarded_market_recovery.py`: `ABSENT` (`NOT_DEPLOYED`)

### Reconciliation of Discrepant Deployment Reports

- **Report 1** ("Adapter exists"): Inspected `/app/scripts/ingest_alpaca_historical.py` on the live container filesystem; observed unpatched adapter (`47568eea...`).
- **Report 2** ("Adapter absent"): Checked commit `b816f773cfbc6481d28413e2758a362ac153e995` referenced in target identity export metadata, where the adapter was not yet committed.
- **Authoritative Reality**: The adapter exists on production container deployment `4aa235d8...`, but is unpatched. Guarded recovery tooling is completely absent on the container.

---

## 4. Ingestion Adapter Divergence & Reconciled Synthesis

| Capability / Check | Workstream A | Workstream B (Initial) | Reconciled Candidate |
| :--- | :--- | :--- | :--- |
| **`pilot_candidates` JSON Parsing** | No | Yes | **Yes** |
| **`candidate_bindings` JSON Parsing** | Yes | No | **Yes** |
| **Canonical Identity Format Enforcement (`ARX_SEC_...`, `ASSOC_...`)** | No | Yes | **Yes** |
| **Noncanonical Alias Rejection (`sec_aapl_...`)** | No | Yes | **Yes** |
| **`is_prov_compatible` Static Helper** | Inline | Static method | **Static method with provider & split check** |
| **Outside-Scope Provider Check (`incompatible_outside`)** | Yes | Missing provider check | **Yes (checks split + provider)** |
| **Explicit `REPLACE` Action Support** | Yes | Collapsed to `BACKFILL` | **Yes (explicit `REPLACE`)** |
| **Guarded Recovery Integration (`MAINT-...`, `/proc/*/fd` handle fence)** | No | Yes | **Yes** |

**Conclusion**: Reconciled implementation in `finance` synthesizes all required safety controls into a single, cohesive release candidate.
