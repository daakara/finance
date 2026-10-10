# SMART MONEY WAVE 1 — TEST EVIDENCE & INTEGRATION AUDIT

## 1. Executive Summary

This document captures the unit test execution and real SEC EDGAR bounded integration audit evidence for Smart Money Wave 1.

---

## 2. Test Matrix Execution

```text
TEST_SUITE = tests/test_smart_money_form4_pipeline.py
RUNNER = pytest 8.4.2 (Python 3.11.0)
COLLECTED_TESTS = 8
PASSED = 8
FAILED = 0
EXECUTION_TIME = 1.09s
```

### Test Case Breakdown

| Test Function | Target Invariant | Result |
|:---|:---|:---|
| `test_form4_parser_open_market_purchase` | Parses Form 4 XML, extracts code `P`, acquired `A`, shares, positive price ($225.50), CEO title, direct ownership. | **PASS** |
| `test_false_positive_controls_grant_and_exercise` | Rejects equity grants (`A`), option exercises (`M`), and tax withholdings (`F`) from positive buying conviction. | **PASS** |
| `test_false_positive_controls_zero_price_purchase` | Fails closed on code `P` with `$0.00` price. | **PASS** |
| `test_deduplication_record_key` | Generates deterministic composite primary key `{accession}:{reporting_owner_cik}:NON_DERIVATIVE:{seq}` and content SHA256. | **PASS** |
| `test_ingestion_and_sqlite_persistence` | Verifies SQLite transaction persistence and idempotent insertion (zero duplicate rows on replay). | **PASS** |
| `test_scanner_clustering_and_conviction_scoring` | Verifies multi-insider cluster buy detection (2 distinct buyers in window), C-Suite participation bonus, and arithmetic score >= 90.0. | **PASS** |
| `test_readiness_contract_false_zero_prevention` | Enforces that market-wide production scanner fails closed to `PIPELINE_PENDING` with `matched_count: None` (no false-zero). | **PASS** |
| `test_bounded_sample_snapshot_with_zero_matches` | Verifies that bounded sample with 0 matches emits `NO_MATCHES` with governed `matched_count: 0`. | **PASS** |

---

## 3. Bounded Local Integration Run Audit (SEC EDGAR Live Source)

A bounded sample of 5 major US equity issuers (`AAPL`, `NVDA`, `PLTR`, `MSFT`, `VRT`) was audited against live SEC EDGAR data:

```text
SAMPLE_UNIVERSE_COUNT = 5
FILINGS_DISCOVERED = 10
TRANSACTIONS_PARSED = 20
OPEN_MARKET_PURCHASES = 0
EXCLUDED_NON_DIRECTIONAL = 20
ERRORS = 0
DUPLICATES = 0
REQUEST_COUNT = 10 (Initial Run) / 0 (Cached Re-run)
CACHE_HITS = 0 (Initial Run) / 10 (Cached Re-run)
SCANNER_STATUS = NO_MATCHES
MATCHED_COUNT = 0
```

### Empirical Findings
1. **False-Positive Filtration**:
   - Recent filings included Tim Cook (AAPL) 374,541 share option exercise (`M`), tax withholding (`F`), 10b5-1 sales (`S`), and gift (`G`); Tench Coxe (NVDA) 500,000 share gift (`G`); David Glazer (PLTR) 92,908 share grant (`A`); Brad Smith (MSFT) 29,077 share grant (`A`).
   - The parser correctly excluded all 20 transactions from open-market purchase conviction, preventing massive false-positive buying signals.
2. **Readiness & State Truthfulness**:
   - The scanner correctly output `status: NO_MATCHES` with `matched_count: 0` for the bounded sample, proving that zero matches is truthfully reported only when an actual complete scan is performed.
3. **Caching & Rate Safety**:
   - On replay, `CACHE_HITS = 10` and `REQUEST_COUNT = 0`, verifying local SQLite caching and zero redundant SEC egress.
