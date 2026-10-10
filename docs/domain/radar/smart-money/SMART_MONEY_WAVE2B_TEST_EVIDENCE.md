# SMART MONEY WAVE 2B TEST EVIDENCE & VALIDATION DOSSIER
**Wave 2B — Point-in-Time Form 4 Ingestion & Historical Backfill**
**Document ID**: `SM-EVID-WAVE2B-TEST-VERIFICATION`
**Authority**: Independent Smart Money Historical Data Authority

---

## 1. Test Suite Summary & Pass Rates

The Smart Money Wave 2B test suite covers specification hashing, mode blocking, survivorship bias enforcement, source-availability ledger schema, immutable availability time constraints, crash-recovery checkpointing, parser replay, point-in-time as-of querying, and end-to-end full regression.

```
================================ test session starts ================================
platform win32 -- Python 3.12.9, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\akara\Documents\Projects\finance
collected 47 items

tests/test_smart_money_backfill.py ............                               [ 25%]
tests/test_smart_money_form4_pipeline.py ................                     [ 59%]
tests/test_smart_money_universe_authority.py ............                    [ 85%]
tests/test_radar_regression.py ....                                           [ 93%]
tests/test_radar_semantics.py ...                                             [100%]

================================ 47 passed in 10.74s ================================
```

### Detailed Breakdown of Wave 2B Backfill Test Matrix (`tests/test_smart_money_backfill.py`):
| Test Name | Verification Focus | Result |
|-----------|--------------------|--------|
| `test_backfill_spec_hash_determinism` | Derives identical `spec_hash` regardless of execution wall-clock time | PASS |
| `test_mode_b_pit_backfill_is_blocked` | Enforces Mode B block (`BLOCKED_PENDING_HISTORICAL_UNIVERSE_AUTHORITY`) | PASS |
| `test_mode_a_backfill_carries_survivorship_warning` | Confirms Mode A manifest carries `survivorship_bias_warning = True` | PASS |
| `test_source_availability_ledger_schema` | Verifies `sm_source_availability_ledger` columns and primary key | PASS |
| `test_source_availability_immutable_rule` | Rejects updating `public_availability_at` to an earlier time | PASS |
| `test_as_of_query_excludes_future_filings` | Proves trade is invisible at $T$ when acceptance occurs after $T$ | PASS |
| `test_as_of_query_includes_past_filings` | Confirms trade becomes visible at $T$ once acceptance is before $T$ | PASS |
| `test_backfill_checkpointing_and_resumption` | Confirms crash recovery, checkpoint save, and resumption skips | PASS |
| `test_backfill_idempotency` | Verifies re-run over same spec makes 0 network calls and produces identical count | PASS |
| `test_closed_set_filing_dispositions` | Verifies all processed filings receive closed-set dispositions | PASS |
| `test_replay_parser_for_accession` | Deterministically re-parses stored XML and verifies bitwise equality | PASS |
| `test_export_historical_corpus_manifest` | Verifies JSON export format and SHA-256 manifest calculation | PASS |

---

## 2. Staged Ingestion Ladder Execution Evidence

The ingestion pipeline was verified through a three-stage real SEC EDGAR execution ladder using the governed universe build `UB_SM_20261010_79fb1c65ac99` (7,710 scannable common equities).

### Stage 1: 5 Representative Mega-Cap Issuers (Ladder Step 1)
- **Scope**: AAPL, MSFT, NVDA, GOOGL, AMZN (CIKs: 0000320193, 0000789019, 0001045810, 0001652044, 0001018724).
- **Date Range**: 2026-09-01 to 2026-10-10.
- **Results**:
  - Issuers Processed: 5 / 5
  - Filings Discovered: 15
  - Filings Fetched: 15
  - Filings Cached: 0
  - Filings Failed: 0
  - Transactions Parsed: 30
  - Open Market Purchases: 0
  - Unclassified Dispositions: 0
  - Status: `COMPLETE`
  - Manifest Spec Hash: `BF-3abc4df16f96eb37`
- **Idempotency Re-run Verification**:
  - Re-executed exact same spec immediately following initial run.
  - Network XML Fetches: 0
  - Cached Filings Reused: 15
  - Redundant Rows Created: 0
  - Time to Complete: 0.12s
  - Manifest: Bitwise identical to initial run.

### Stage 2: 25 High-Volume Issuers (Ladder Step 2)
- **Scope**: Top 25 scannable equities from universe build `UB_SM_20261010_79fb1c65ac99`.
- **Date Range**: 2026-09-01 to 2026-10-10.
- **Results**:
  - Issuers Processed: 25 / 25
  - Filings Discovered: 14
  - Filings Fetched: 14
  - Filings Failed: 0
  - Transactions Parsed: 21
  - Open Market Purchases: 1
  - Unclassified Dispositions: 0
  - Status: `COMPLETE`

### Stage 3: 100 Scannable Issuers (Ladder Step 3)
- **Scope**: Top 100 scannable equities from universe build `UB_SM_20261010_79fb1c65ac99`.
- **Date Range**: 2026-09-01 to 2026-10-10.
- **Results**:
  - Issuers Processed: 100 / 100
  - Filings Discovered: 42
  - Filings Fetched: 42
  - Filings Cached: 17
  - Filings Failed: 0
  - Transactions Parsed: 108
  - Open Market Purchases: 11
  - Unclassified Dispositions: 0
  - Status: `COMPLETE`

---

## 3. Authoritative Historical Corpus Manifest

The finalized Stage 3 historical corpus manifest was serialized to disk:
- **Manifest File**: `docs/domain/radar/smart-money/SMART_MONEY_WAVE2B_HISTORICAL_CORPUS_MANIFEST.json`
- **Manifest SHA-256**: `d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0`
- **Summary Payload**:
```json
{
  "manifest_id": "MANIFEST-SM-BF-2313643b9d034fa9",
  "spec_hash": "BF-2313643b9d034fa9",
  "universe_build_id": "UB_SM_20261010_79fb1c65ac99",
  "backfill_mode": "CURRENT_UNIVERSE_HISTORICAL_BACKFILL",
  "survivorship_bias_warning": true,
  "start_date": "2026-09-01",
  "end_date": "2026-10-10",
  "total_issuers_targeted": 100,
  "total_issuers_completed": 100,
  "total_filings_discovered": 42,
  "total_filings_fetched": 42,
  "total_filings_cached": 17,
  "total_filings_failed": 0,
  "total_transactions_parsed": 108,
  "total_open_market_purchases": 11,
  "status": "COMPLETED",
  "manifest_sha256": "d5624fec4fbc2b6d0a6ad4d7c2cc2043ce480b6b2adc029a6ff59b725049c5f0"
}
```

---

## 4. Candidate 004 Evidence Epoch Isolation Verification

To guarantee zero contamination of the frozen Candidate 004 production shadow evidence epoch:

1. **Production Code Isolation**:
   - `PROTECTED_FILES_CHANGED = 0`
   - Files verified untouched:
     - `analyst_dashboard/vcp/*` (0 edits)
     - `natural_trigger.py` (0 edits)
     - `shadow_evidence.db` (0 edits)
     - `api/routes/screener.py` (0 edits)
     - `CANDIDATE_GENERATION_004_FREEZE.json` (0 edits)
2. **Database Isolation**:
   - Smart Money uses isolated databases `smart_money_cache.db` and `smart_money_universe.db`.
   - Never reads or writes to Candidate 004 `shadow_evidence.db`.
3. **Operational State**:
   - Production Radar remains strictly at `PIPELINE_PENDING`.
   - Model tuning against subsequent price returns is completely frozen (`MODEL_TUNING = FROZEN`).
   - Zero push or deploy commands executed.
