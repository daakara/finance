# ARX Forward Paper-Trading Pilot — Storage Directory (`data/forward_pilot/`)

**Status:** INITIALIZED (ISOLATED STORAGE)  
**Governing Roadmap:** [`docs/planning/ARX_PAPER_TRADING_ROADMAP.md`](../../docs/planning/ARX_PAPER_TRADING_ROADMAP.md)  
**Accountable Owner:** `daakara`  

---

## 1. Directory Structure & Layout

This directory is the dedicated, durable storage location for the ARX Forward Paper-Trading Pilot. All forward artifacts are isolated here to ensure zero contamination of historical baseline records or production databases.

```
data/forward_pilot/
├── README.md                  # This specification and isolation contract
├── capture_stream.jsonl       # Immutable append-only candidate evaluation decisions
├── settlement_ledger.json     # State checkpoint for forward setups & outcome events (flock protected)
└── raw_responses/             # Preserved raw daily price responses & SHA-256 integrity metadata
```

---

## 2. Invariants & Isolation Rules

1. **Cohort ID Namespace Isolation:**
   - All forward pilot records must use the prefix `COHORT_FORWARD_PILOT_*` (e.g. `COHORT_FORWARD_PILOT_002`).
   - Forward routines are strictly barred from reading from, appending to, or mutating historical cohort namespaces (`COHORT_HISTORICAL_PILOT_001`, `ARX_PROSPECTIVE_VALIDATION_EPOCH_*`, `V2_4_0_*`).
2. **Write-Path Confinement & Reusable Validation:**
   - All file writes targeting this directory must be validated by `analyst_dashboard.governance.storage.validate_forward_write_path(target_path, cohort_id)`.
   - Writes outside the configured forward root, path traversals (`..`), symlink escapes, or writes intersecting protected historical paths (`analyst_dashboard/data`, `docs/governance`, `tests/fixtures`) raise fail-closed exceptions before modifying disk.
   - Atomic writes must use `analyst_dashboard.governance.storage.safe_write_forward_pilot_file` to guarantee zero files created upon validation failure.
3. **Filesystem Durability vs. Local Workstation Storage:**
   - **Local Workstation / Test Runtime:** When running locally, `data/forward_pilot` resides inside the local repository checkout. This is local developer storage only and does NOT guarantee physical volume durability.
   - **Production Persistent Mount Requirement:** In deployed container environments (e.g. Railway), true filesystem durability requires configuring `ARX_FORWARD_PILOT_DATA_DIR` on a dedicated persistent volume mount (e.g. `/root/analyst_dashboard/data/forward_pilot` on mount `/root`). Durability is attested via filesystem device ID checks (`st_dev != root_dev`) in `storage.attest_persistent_storage()`. Simply naming a container path `data/forward_pilot` on overlayfs is ephemeral and strictly fails closed.
4. **Append-Only Decision Log (`capture_stream.jsonl`):**
   - Candidate decisions are written append-only before subsequent price bars exist.
   - Every evaluated symbol across the 20-session × 10-symbol universe (200 records) is recorded with UTC timestamp, spot price, corridor, stop floor, and confluence score.
5. **Atomic Settlement Ledger (`settlement_ledger.json`):**
   - Checkpointed under POSIX advisory locking (`fcntl.flock`).
   - Outcome transitions (`ENTRY_TRIGGERED`, `STOP_LOSS`, `TAKE_PROFIT_1`, `HORIZON_EXIT`, `ENTRY_WINDOW_EXPIRED`) are appended with execution timestamps, fill prices, and split-adjustment provenance.
6. **Zero Production Mutation:**
   - No records in this directory interact with production PostgreSQL, production historical databases, or `.finance_market_store.db`.
7. **Zero Broker API Connectivity:**
   - No broker order routing or execution adapters are wired. All entries and exits are simulated internally under Contract V1.0.1 rules.
