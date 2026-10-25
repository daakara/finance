# ARX TERMINAL — SPRINT 2 PACKAGE B
# CLEAN RELEASE CANDIDATE DIFF REPORT

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Base Revision**: `origin/main` (`f6c5755126059b2f227e28ce90ddaad43997884c`).
- **Target Revision**: `fix/sprint2-package-b-canonical-ingestion-release`.
- **Disposition**: Clean code-only delta ready for approval.

---

## 2. Overall Release Delta Summary

```text
 scripts/ingest_alpaca_historical.py | 131 +++++++++++++++++++++++++++++-------
 1 file changed, 107 insertions(+), 24 deletions(-)
```

### New Untracked Production Modules
1. `analyst_dashboard/data/guarded_recovery.py` (+377 lines, 16,446 bytes)
2. `scripts/guarded_market_recovery.py` (+73 lines, 2,867 bytes)

### New Untracked Acceptance & Rehearsal Test Suites
1. `tests/test_pilot_identity_and_guarded_recovery.py` (+470 lines, 21,910 bytes)
2. `tests/test_package_b_deployment_rehearsal.py` (+385 lines, 16,846 bytes)

### New Canonical Manifest & Review Artifacts
1. `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`
2. `docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_*`

---

## 3. Surgical Code Diffs in `scripts/ingest_alpaca_historical.py`

### 3.1 Dual Manifest Ingestion & Canonical Identity Format Validation
- **Location**: `load_manifest` lines 360–415.
- **Delta**:
  - Adds support for `ingestion_scope.candidate_bindings` from Package B manifests.
  - Adds support for `pilot_candidates` array.
  - Adds strict regex validation for durable security IDs (`^ARX_SEC_[0-9A-F]{12}$`) and listing associations (`^ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z]{4}_[A-Z0-9]+_\d{4}-\d{2}-\d{2}$`).
  - Rejects any symbol starting with `SEC_` or failing `^[A-Za-z]{1,5}$` with `NONCANONICAL_IDENTITY_ALIAS_REJECTED`.

### 3.2 Provider Lineage Purity (`is_prov_compatible`)
- **Location**: lines 430–465.
- **Delta**:
  - Extracted to static method `is_prov_compatible(existing_provenance, existing_candles) -> Tuple[bool, Optional[str]]`.
  - Rejects non-Alpaca stored lineage (`MASSIVE_POLYGON`, etc.) with `INCOMPATIBLE_PROVIDER`.
  - Enforces `SPLIT_ADJUSTED_ONLY` basis with `INCOMPATIBLE_ADJUSTMENT`.
  - Validates 1:1 candle-to-provenance correspondence.

### 3.3 Explicit `REPLACE` Action & Incremental Refresh Suppression
- **Location**: lines 1125–1165.
- **Delta**:
  - Manifest directive `action="REPLACE"` unconditionally sets `action = "REPLACE"` and `sym_start = start_date`, suppressing incremental refresh.
  - Passes `replace_existing_scope=(action in ("BACKFILL", "REPLACE"))` to atomic SQLite write engine.

### 3.4 Outside-Scope Incompatible Provider Scan
- **Location**: lines 1175–1190.
- **Delta**:
  - Checks existing records before `sym_start` and after `end_date`.
  - Fails closed with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` if unprovenanced, foreign provider, or non-split records exist outside reviewed scope.

---

## 4. Blast Radius Containment Verification

- **Zero API/Endpoint Changes**: No routes in `api/` or controllers modified.
- **Zero Schema Migrations**: No SQLite or PostgreSQL DDL executed or altered.
- **Zero Scanner/VCP Modifications**: `analyst_dashboard/vcp/` and `scanner_runner.py` untouched.
- **Zero Smart Money Leakage**: Form 4 parsers, PostgreSQL migrations, and accounting tables are 100% excluded.
