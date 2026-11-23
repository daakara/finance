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
└── COHORT_FORWARD_PILOT_002/   # Independent namespace per cohort
    ├── cohort.json            # Immutable start session, universe/contract digest and engine SHA
    ├── inputs/                # Content-addressed original evaluation batches
    ├── capture_stream.jsonl   # Immutable candidate records, ten per committed cycle
    ├── settlement_ledger.json # Future PT-03 checkpoint (not implemented/activated)
    └── raw_responses/         # Future retained market-data custody
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

## 3. PR-PT-02 manual retained-input capture

`scripts/capture_forward_candidates.py` is a local, offline-input capture adapter.
It does **not** fetch data, run Radar/Confluence, schedule work, enter simulated
trades, or settle outcomes. Production-runtime execution is rejected. A future
qualified evaluator adapter is required for natural forward evidence.

Input JSON schema (all ten symbols are expected; missing rows are recorded):

```json
{
  "schemaVersion": "1.0",
  "samplingMode": "SYSTEMATIC",
  "sessionDate": "2026-11-23",
  "engineSha": "<actual 40-character lowercase engine commit SHA>",
  "evaluations": {
    "AAPL": {
      "decisionState": "VALID_SETUP",
      "sourceTimestampUtc": "2026-11-23T21:00:00Z",
      "observedAtUtc": "2026-11-23T21:10:00Z",
      "plan": {
        "spot": 100, "corridorMin": 98, "corridorMax": 102,
        "stopLoss": 95, "takeProfit1": 110, "confluenceScore": 70
      }
    }
  }
}
```

The prices above illustrate the schema only; they are not an authentic signal.
Rows may instead carry `error` for a failed upstream evaluation. Each missing,
failed, stale/future or invalid result remains in the ten-symbol denominator,
with no fabricated plan. All imports are labelled `OFFLINE_INPUT` and
`eligibleForOpportunityCapture=false`; metadata alone does not certify origin.

For a new cohort, from repository root (local implementation use only):

```bash
python scripts/capture_forward_candidates.py \
  --input /absolute/path/to/retained-batch.json \
  --forward-root /absolute/path/to/existing-isolated-forward-root \
  --cohort-id COHORT_FORWARD_PILOT_002 --start-session YYYY-MM-DD --initialize
```

Omit `--initialize` on continuation. Missing/corrupt streams or manifests stop
continuation rather than reset history. The input session must be the actual
current New York session; the CLI has no backdating/clock override. Capture is
allowed from exchange close +15 minutes until next session open minus 12 hours.
XNYS holidays, DST and early closes are respected (16:15 ET on regular days).

The frozen manifest is `docs/governance/ARX_FORWARD_PILOT_UNIVERSE_V1.json`.
The first valid capture binds the cohort's engine, contract, universe and start
session. Exactly ten records are committed atomically under `flock`; identical
retries are no-ops, conflicting retries fail, and later cycles preserve existing
record bytes. Twenty consecutive exchange-session opportunities permit up to
200 records. Missed sessions are not backfilled or counted as completed.
The 20-entry cap belongs to future settlement; this CLI creates **zero entries**.

Verification (blocks socket/database connections and bypasses unrelated DB seeds):

```bash
PYTHONPATH=tools/qualification:. python -m pytest --noconftest \
  -p paper_trading_offline_support tests/test_forward_candidate_capture.py \
  tests/test_paper_trading_baseline_determinism.py tests/test_paper_trading_outcome_evaluator.py -q
```

Git does not persist operational captures (only this README is allowlisted).
Atomic replacement/fsync protects local writes; it does not prove a durable
production mount or authorize activation. No hold or runtime setting is changed.
