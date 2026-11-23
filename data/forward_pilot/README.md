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
This capture CLI creates **zero entries**. The separate offline settlement CLI
enforces the 20-slot cap described below.

Verification (blocks socket/database connections and bypasses unrelated DB seeds):

```bash
PYTHONPATH=tools/qualification:. python -m pytest --noconftest \
  -p paper_trading_offline_support tests/test_forward_candidate_capture.py \
  tests/test_forward_settlement.py tests/test_paper_trading_baseline_determinism.py \
  tests/test_paper_trading_outcome_evaluator.py -q
```

Git does not persist operational captures (only this README is allowlisted).
Atomic replacement/fsync protects local writes; it does not prove a durable
production mount or authorize activation. No hold or runtime setting is changed.

## PT-03: manual offline settlement

```bash
python scripts/settle_forward_cohort.py \
  --input /absolute/path/to/retained-price-batch.json \
  --forward-root /absolute/path/to/existing-isolated-forward-root \
  --cohort-id COHORT_FORWARD_PILOT_002 --initialize
```

Omit `--initialize` for continuation. This command neither fetches prices nor
evaluates the live model. It rejects production execution and leaves capture
records unchanged. All results retain `OFFLINE_INPUT` and opportunity-ineligible
labels. They are simulated paths, not broker fills or a certified forward cohort.

The price batch has `schemaVersion: "1.0"`, `cohortId`, `asOfSession` (a completed
XNYS session), and `series` keyed by symbol. Each series contains `provider`,
`bars` (ordered objects with `date`, `Open`, `High`, `Low`, `Close`, `Volume`) and
`adjustmentEvidence`. Required evidence fields are:

```json
{
  "status": "VERIFIED",
  "priceBasis": "SPLIT_ADJUSTED_ONLY",
  "cashDividendsAdjusted": false,
  "splitEvents": [],
  "sourceSha256": "<64 lowercase hexadecimal characters>",
  "sourceDescription": "<retained corporate-action metadata reference>",
  "coverageStart": "YYYY-MM-DD",
  "coverageEnd": "YYYY-MM-DD"
}
```

These are retained **input assertions**, not independently authenticated evidence.
The runner requires them but cannot establish that an upstream source or its
corporate-action inventory is complete. Never manufacture these fields to pass
validation. A real source adapter and its byte-to-bar/adjustment reconciliation
must be qualified before natural forward use. This version supports no-split
intervals only; nonempty split events or unverified basis block the batch.

Supply the complete retained price prefix at every run, including every XNYS
session from the next session after each eligible signal through `asOfSession`,
bounded to its maximum 24-session evaluation horizon. Gaps, duplicates, future
bars, nonfinite OHLCV, changed providers and historical revisions block advancement.
The frozen evaluator applies the 5-session entry window, 20-session holding
window, gap/limit rules, 25 bps friction and unresolved daily-bar collisions.

Capacity is allocated by entry date, then signal date, frozen universe order and
decision ID, independent of returns. A daily collision with uncertain entry
reserves one slot conservatively without claiming a confirmed fill; thus confirmed
entries plus uncertain reservations never exceed 20. Later triggered candidates
are `SKIPPED_CAP_REACHED`; all captured decisions stay in the denominator.
Multiple setups for one symbol remain independent simulated recommendations;
this is not a portfolio, position-sizing or cash-account simulator.

`settlement_inputs/{sha256}.json` retains original batch bytes.
`settlement_ledger.json` contains current results, price history, immutable prior
event entries, input receipts and a state digest. A shared capture lock serializes
capture/settlement; atomic replacement commits each complete state. Five identical
replays leave ledger bytes unchanged. Missing/corrupt state or receipts block
continuation. An interrupted initial commit can leave a raw input without a
ledger; reconcile that state explicitly rather than reinitializing it.

No scheduler, production writes, broker adapter, hold changes or activation are
included. Settlement can continue beyond capture Session 20 so the last setup
receives its full entry and holding windows (up to pilot Session 44).

---

## 4. Operational Modes, Authenticity Audit, and Evidence Inventory

To prevent epistemic contamination and ensure strict governance, the ARX paper-trading infrastructure enforces strict separation between integrity, source authenticity, candidate origin, and activation states.

### 1. Four-Layer Evidence Inventory
1. **Historical Retained Prices (Custody Verified):** Authentic raw response bytes from Yahoo Finance v8 chart API exist for `VOO`, `QQQM` (both spanning 2026-09-25 to 2026-10-28), and `JEPI` (2026-09-25 to 2026-10-22). Each retained file is content-addressed under its SHA-256 digest with UTC retrieval timestamp and timezone metadata.
2. **Current-Universe Overlap (2 of 10 Symbols in Custody):** The forward pilot universe manifest (`ARX_FORWARD_PILOT_UNIVERSE_V1.json`) defines 10 symbols: `VOO`, `QQQM`, `IWM`, `XLE`, `XLK`, `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`. Exactly **two symbols** (`VOO` and `QQQM`) overlap with retained historical price custody. `JEPI` is outside the forward universe; the other 8 symbols have zero authentic historical price custody.
3. **Corporate-Action Evidence (Zero External Filings Acquired):** Exactly **zero** authentic external corporate-action filings have been acquired. All existing references are input assertions or synthetic fixtures. No authentic zero-splits certification exists. Matching bytes, hashes, and schema metadata establish integrity and schema consistency; source authenticity requires external acquisition provenance.
4. **Forward Readiness (Simulation Ready, Live Activation Disabled):** Internal simulation, candidate capture (`capture_stream.jsonl`), and outcome settlement (`settlement_ledger.json`) mechanics are 100% qualified and idempotent offline. Live forward pilot activation remains strictly disabled (`HOLD_SCAN_001=ACTIVE`, `HOLD_EXEC_001=ACTIVE`, `activationState: DISABLED`). All imported decisions remain `OFFLINE_INPUT` and opportunity-ineligible (`eligibleForOpportunityCapture: False`).

### 2. Audit of Authenticity Labels: Integrity vs. Source Authenticity
- **Integrity and Schema Consistency (`INTEGRITY_VERIFIED_SCHEMA_CONSISTENT`):** Matching SHA-256 digests, valid JSON structure, matching symbols, and agreeing date intervals prove that bytes have not been corrupted and fields conform to schema. They do **not** prove that the data originated from an authoritative external market data provider.
- **Source Authenticity (`AUTHENTIC_CORPORATE_ACTION_VERIFIED`):** Source authenticity requires verified **external acquisition provenance** (`acquisitionProvenance` containing authoritative external provider such as `SEC_EDGAR_SUBMISSIONS`, canonical target URL, ISO 8601 UTC timestamp, and HTTP status 200). Fabricated or fixture JSON claiming source authenticity or zero-splits certification without external acquisition provenance fails closed with `UNSUPPORTED_AUTHENTICITY_CLAIM`.
- **Authority Labeling in Offline Execution:** Because zero authentic corporate-action filings have been acquired, all offline and fixture settlement runs record `adjustmentEvidenceAuthority: "RETAINED_INPUT_ASSERTION_NOT_INDEPENDENTLY_AUTHENTICATED"`. Attempting to record `AUTHENTIC_CORPORATE_ACTION_VERIFIED` without verified external acquisition provenance fails closed with `UNSUPPORTED_AUTHENTICITY_CERTIFICATION`.
- **Candidate Decoupling Invariant:** Authentic price evidence can **never** upgrade imported candidate decisions (`evidenceOrigin: OFFLINE_INPUT`) to authentic forward observations. Attempts to pass `evidenceOrigin: AUTHENTIC_FORWARD` against imported decisions fail closed with `CANNOT_UPGRADE_OFFLINE_DECISIONS`.

### 3. Smallest Useful Validation Specification
Rather than executing an unnecessary ten-symbol historical price download, the smallest useful validation reuses retained `VOO` and `QQQM` price tape (0 price download requests) and acquires only the genuinely missing corporate-action filings from SEC EDGAR:
- **Request Accounting:** Exactly 2 external HTTP GET requests total (0 price requests, 2 corporate-action requests, 0 metadata requests).
- **Target 1 (VOO):** Vanguard Index Funds (CIK `0000036405`), submissions URL `https://data.sec.gov/submissions/CIK0000036405.json`. Covered interval: `2026-09-20` to `2026-11-05`.
- **Target 2 (QQQM):** Invesco Exchange-Traded Fund Trust II (CIK `0001378872`), submissions URL `https://data.sec.gov/submissions/CIK0001378872.json`. Covered interval: `2026-09-20` to `2026-11-05`.
- **Completeness Evidence:** `filings.recent` form history confirms zero stock split corporate actions across the 24-session evaluation window.

### 4. Pilot Activation (STRICTLY DISABLED)
- **Status:** **`DISABLED`** (`HOLD_SCAN_001=ACTIVE`, `HOLD_EXEC_001=ACTIVE`).
- **Enforcement:**
  - Zero background daemon / cron executions (`VCP_NATURAL_SCHEDULER_ENABLED=0`).
  - Zero live broker order placement (Alpaca Paper API integration is deferred to Phase 2).
  - Zero production database mutations (writes to `/root/analyst_dashboard/data` or `governance.db` are blocked).
  - Zero model retraining or heuristic adjustments.
- **Activation Blockers:** Activation requires formal unfreeze authorization by Accountable Owner (`daakara`), completion of natural recommendation candidate capture qualification, and full release qualification (RQ-01 through RQ-13).
