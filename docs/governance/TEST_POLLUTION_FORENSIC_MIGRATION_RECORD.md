# Forensic Migration Record: Automated Test Pollution Remediation & Preservation

## 1. Executive Summary

This record documents the remediation and forensic preservation of 21 test-generated signals that were inadvertently written to `analyst_dashboard/data/paper_trading_ledger.json` during automated test runs following the initial implementation of the Step 2 passive capture hook.

In strict adherence to the project's evidence-preservation invariant:
```text
bad / invalid / synthetic evidence
  → preserve
  → classify
  → exclude
```
the 21 entries have been removed from the authoritative production ledger to protect primary denominator purity, while their complete payloads, timestamps, inputs, and hashes have been permanently preserved in an immutable forensic archive.

---

## 2. Invariant & Governance Ledger

```ini
TEST_POLLUTION_RECORDS_REMOVED_FROM_PRIMARY_LEDGER =
  21
TEST_POLLUTION_FORENSIC_ARCHIVE =
  VERIFIED (analyst_dashboard/data/test_pollution_forensic_archive.json)
TEST_POLLUTION_RECORD_IDS =
  PRESERVED (21 distinct IDs)
REMOVAL_REASON =
  AUTOMATED_TEST_LEDGER_CONTAMINATION
PRIMARY_PROSPECTIVE_DENOMINATOR_IMPACT =
  NONE
ARCHIVE_RECORDS_SHA256 =
  001dcdc29392c28788e7e54b192a214b463330e3e0255d4c229232950a96425f
FIREWALL_GUARD =
  ACTIVE (PassiveCaptureHook test runner bypass)
```

---

## 3. Incident Timeline & Root Cause Analysis

1. **Originating Commit (`9bc1854`)**:
   `PassiveCaptureHook.record_natural_recommendation` was wired into `api/routes/analytics.py` using `ledger_path: Optional[str] = None`, defaulting to `DEFAULT_LEDGER_PATH` (`analyst_dashboard/data/paper_trading_ledger.json`).
   
2. **Pollution Vector**:
   Subsequent test runs executing API routes or integration tests invoked the analytics pipeline without isolating the ledger path. As a result, 21 mock and synthetic assets (including test fixtures `SYM_TEST_UNKNOWN` and `CNT_SCORES`, as well as standard tickers like `NVDA`, `SPY`, `PLTR`) were appended to `paper_trading_ledger.json`.

3. **Remediation & Firewall (`1725fd8`)**:
   A fail-closed ledger firewall was introduced into `PassiveCaptureHook.record_natural_recommendation`:
   ```python
   if ledger_path is None and ("PYTEST_CURRENT_TEST" in os.environ or os.getenv("ARX_TEST_MODE") == "1"):
       logger.debug("[PASSIVE_CAPTURE] Bypassing capture to production ledger during automated test execution.")
       return None
   ```
   All test fixtures were explicitly redirected to isolated temporary ledger paths.

4. **Preservation Action**:
   The 21 contaminated records were extracted from Git history (`9bc1854`), validated, and serialized to [`analyst_dashboard/data/test_pollution_forensic_archive.json`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/data/test_pollution_forensic_archive.json).

---

## 4. Removed Record Manifest (21 Records)

| # | Signal ID | Symbol | Signal Date | Confluence Score | Entry Price | Origin / Vector |
|---|---|---|---|---|---|---|
| 1 | `NVDA_2026-09-19` | NVDA | 2026-09-19 | 82.5 | 119.50 | Test runner integration |
| 2 | `SPY_2026-09-19` | SPY | 2026-09-19 | 65.0 | 560.20 | Test runner integration |
| 3 | `SYM_TEST_UNKNOWN_2026-09-19` | SYM_TEST_UNKNOWN | 2026-09-19 | 50.0 | 100.00 | Synthetic unit test fixture |
| 4 | `PLTR_2026-09-19` | PLTR | 2026-09-19 | 74.0 | 32.40 | Test runner integration |
| 5 | `CNT_SCORES_2026-09-19` | CNT_SCORES | 2026-09-19 | 60.0 | 50.00 | Synthetic unit test fixture |
| 6 | `NVO_2026-09-19` | NVO | 2026-09-19 | 71.0 | 134.80 | Test runner integration |
| 7 | `CPRX_2026-09-19` | CPRX | 2026-09-19 | 78.5 | 18.20 | Test runner integration |
| 8 | `ACLS_2026-09-19` | ACLS | 2026-09-19 | 76.0 | 95.10 | Test runner integration |
| 9 | `TMDX_2026-09-19` | TMDX | 2026-09-19 | 80.0 | 155.00 | Test runner integration |
| 10 | `LNTH_2026-09-19` | LNTH | 2026-09-19 | 73.0 | 112.40 | Test runner integration |
| 11 | `MEDP_2026-09-19` | MEDP | 2026-09-19 | 75.5 | 340.00 | Test runner integration |
| 12 | `POWI_2026-09-19` | POWI | 2026-09-19 | 68.0 | 62.10 | Test runner integration |
| 13 | `ELF_2026-09-19` | ELF | 2026-09-19 | 79.0 | 165.50 | Test runner integration |
| 14 | `DUOL_2026-09-19` | DUOL | 2026-09-19 | 77.0 | 210.00 | Test runner integration |
| 15 | `VRT_2026-09-19` | VRT | 2026-09-19 | 81.0 | 88.30 | Test runner integration |
| 16 | `CRWD_2026-09-19` | CRWD | 2026-09-19 | 72.5 | 280.00 | Test runner integration |
| 17 | `TSM_2026-09-19` | TSM | 2026-09-19 | 84.0 | 175.20 | Test runner integration |
| 18 | `INTC_2026-09-19` | INTC | 2026-09-19 | 55.0 | 21.30 | Test runner integration |
| 19 | `ETH-USD_2026-09-19` | ETH-USD | 2026-09-19 | 62.0 | 2450.00 | Test runner integration |
| 20 | `LLY_2026-09-19` | LLY | 2026-09-19 | 80.5 | 920.00 | Test runner integration |
| 21 | `MSFT_2026-09-19` | MSFT | 2026-09-19 | 76.0 | 435.00 | Test runner integration |

---

## 5. Auditability & Forensic Verification

The exact content of all 21 records can be cryptographically verified at any time using:
```bash
python -c "
import json, hashlib
archive = json.load(open('analyst_dashboard/data/test_pollution_forensic_archive.json'))
records = archive['records']
assert len(records) == 21
assert hashlib.sha256(json.dumps(records, sort_keys=True).encode('utf-8')).hexdigest() == archive['recordsSha256']
print('Archive integrity VERIFIED:', archive['recordsSha256'])
"
```
