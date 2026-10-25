# ARX TERMINAL — SPRINT 2 PACKAGE B
# CODE-ONLY STARTUP SAFETY & RUNTIME NON-INTERFERENCE REPORT

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Gate**: `G-S2-ACT`.
- **Subject**: Verification that code release deployment cannot automatically initiate ingestion or mutate state.
- **Production Status**: Deployment `NOT_AUTHORIZED`; Ingestion `NOT_AUTHORIZED`.

---

## 2. Deployment Architecture & Entrypoint Forensics

Inspection of Railway configuration and container startup hooks confirms:

- **Railway Config (`railway.json`)**:
  - `startCommand`: `sh -c 'uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2'`
- **Container Definition (`Dockerfile`)**:
  - `CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]`
- **Process Entrypoint (`Procfile`)**:
  - `web: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2`

---

## 3. Runtime Non-Interference Proofs

Detailed audit of `api/main.py` and application lifespans establishes:

1. **Zero Adapter Imports**:
   - Neither `scripts/ingest_alpaca_historical.py` nor `scripts/guarded_market_recovery.py` are imported by `api.main` or any route module.
2. **Zero Ingestion Invocation**:
   - `AlpacaHistoricalSipIngestionAdapter.execute_bounded_ingestion` is never referenced or called in application lifespan or background tasks.
3. **`--apply` Flag Safety**:
   - The `--apply` flag exists exclusively as a CLI argument parsed via `argparse` in `scripts/ingest_alpaca_historical.py`. It cannot be triggered by web worker initialization.
4. **Zero Backfill Schedulers**:
   - No cron, celery, or scheduled background tasks exist for historical market data backfill.
5. **Zero Write Exposure to `asset_ohlcv_daily` / `asset_ohlcv_provenance`**:
   - Background warmup in `_warmup_worker()` touches only volatile cache for pre-warmed watchlist symbols (LNTH, CIEN, etc.) via YFinance if absent; it never writes to `asset_ohlcv_provenance` or interacts with Alpaca SIP tables.
6. **Zero Universe Publication**:
   - Universe publication remains strictly guarded by Package C release gates and Gate 12 authority.
7. **Zero Scanner Activation**:
   - Background scanners run in read-only pre-warming mode or remain dormant under feature flags.
8. **Zero Unattended Recovery**:
   - `guarded_market_recovery.py` is a standalone CLI script requiring an explicit `--maint-token` and interactive operator execution. It is never triggered on startup or failure.

---

## 4. Invariant Assertion

> **Production deployment of the Package B code release is strictly decoupled from ingestion execution.**  
> Container deployment simply stages the verified adapter and recovery tooling into `/app/scripts/`. No historical bars will be fetched, no database connections will be opened by the adapter, and no records will be written until an authorized operator explicitly executes the CLI command under separate Stage 2 authorization.
