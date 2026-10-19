# ARX Terminal Sourced Fundamentals — Offline Replay Instructions

This package contains the complete, self-contained Sourced Fundamentals & Undervalued Growth Research package.
It includes all corrected modules, test suites, generated markdown reports, the finalized evidence ledger, and all retained raw primary SEC EDGAR XBRL filings and dated market prices needed to reproduce results offline with zero network requests.

## 1. System Requirements & Dependencies
- Python 3.10+ (tested under Python 3.11)
- Standard library dependencies (`json`, `urllib`, `hashlib`, `dataclasses`, `datetime`, `argparse`, `logging`)
- Development/Testing dependencies:
  - `pytest` (optional, for running test suite)

No external database or API credentials are required.

## 2. Offline Replay Command
To execute the pipeline and regenerate the normalized financial facts, derived metrics, qualitative assessments, deep multi-scenario valuations, and evidence ledger completely offline:

```bash
python -m analyst_dashboard.research.sourced_fundamentals.cli --offline
```

This command:
1. Loads raw primary SEC XBRL facts and submissions from `data/research/sourced_fundamentals/raw/`.
2. Loads authoritative closing market prices from `data/research/sourced_fundamentals/raw/market_prices.json`.
3. Validates fiscal year boundaries, normalizes durations, and computes non-overlapping TTM metrics.
4. Normalizes historical share dilution for stock splits (DECK 6:1, CELH 3:1, KLAC 10:1).
5. Applies the Lease-as-Operating-Expense valuation framework.
6. Solves for 5-year Revenue CAGRs with bracketing and roundtrip forward-model verification.
7. Exports the finalized `sourced_fundamentals_ledger.json`.
8. Automatically triggers `report_generator.py` to regenerate all Markdown deliverables with 100% numerical parity.

## 3. Running Automated Tests
To execute the 23-scenario failure-reproducing test suite:

```bash
python -m pytest tests/test_sourced_fundamentals_pilot.py -v
```

All 23 tests verify duration extraction, split normalization, bracketing, report parity, aggregate completeness, share duration discrimination, evidence-responsive valuation suppression, conflicting observations handling, cutoff enforcement, and offline execution with network egress disabled.
