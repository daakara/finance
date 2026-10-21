# ARX TERMINAL — SPRINT 2 TRACK C
# CERTIFICATION EVIDENCE BOUNDARY & PROVENANCE INTEGRITY

## 1. Architectural Principle: Manifest Intent vs. Data Certification

A fundamental architectural error in prior integration iterations was treating manifest validation (`MANIFEST_VALIDATED`) as an automatic grant of market-data qualification (`DATA_CERTIFIED`).

Under the ARX Closure Model v2, these are strictly separated states:

```
[Manifest On Disk / Proposed]
             │
             ▼
    (PackageBV3Consumer.load_manifest)
             │
             ▼
   MANIFEST_VALIDATED (Authorization State: PROPOSED_NOT_AUTHORIZED)
             │
             │   [NO DATA INGESTED YET]
             │   [NO DATA QUALIFIED]
             ▼
    (Package B Ingestion Execution — SEPARATELY GOVERNED)
             │
             ▼
        DATA_INGESTED (Bars written to asset_ohlcv_daily)
             │
             ▼
    (Package B Quality Audit / Provenance Sidecar Readback)
             │
             ▼
        DATA_CERTIFIED (Uniform SIP, Split-Adjusted, Clean, 288 bars)
             │
             ▼
    PUBLICATION_ELIGIBLE (State 4)
             │
             ▼
    PUBLISHED_UNIVERSE (State 5)
```

### Manifest Alone Yields Zero Qualified Candidates
When `PackageBV3Consumer.build_candidates_from_manifest(manifest)` is invoked without an explicit market data fixture:
- `candle_count = 0`
- `has_live_price = False`
- `provenance_feed = "UNVERIFIED"`
- `adjustment_state = "UNVERIFIED"`
- `structural_quality = "UNVERIFIED"`
- `is_qualified_candidate = False`

When evaluated by `IsolatedPublicationEngine`:
- Every candidate receives `data_readiness_decision = "DATA_UNAVAILABLE"`.
- Every candidate remains trapped at State 2 (`PROVIDER_RESOLVABLE`).
- Publication dry-run produces `published_symbols = []`, `data_ready_count = 0`, and `scannable_count = 0`.

---

## 2. Anti-Inference Firewall & Synthetic Test Fixture Enforcement

To prevent synthetic test data from leaking into or being mistaken for production certification, Package C enforces a strict typing constraint on all test fixtures:

```python
if market_data_fixture is not None:
    if market_data_fixture.get("_fixture_nature") != "SYNTHETIC_TEST_FIXTURE":
        raise PolicyViolationError(
            "Market data fixture must be explicitly declared with _fixture_nature='SYNTHETIC_TEST_FIXTURE'. "
            "Manifest intent alone cannot confer data certification."
        )
```

### Purpose of the Declaration
1. **Prevents Accidental Production Bypass:** If an integration script accidentally passes a partial dictionary or uncertified live readback, it fails closed immediately.
2. **Maintains Audit Provenance:** Every test suite and rehearsal must declare that it operates on synthetic fixtures.
3. **Verified in TEST-08:** `test_08_synthetic_test_fixture_requires_explicit_label` verifies that unlabeled fixtures fail closed with `PolicyViolationError`.

---

## 3. AAPL October 20 Production Tail Forensic Analysis

### 3.1. The Production Store Contamination Risk
The production SQLite market store (`finance_market_store.db`) currently contains 289 daily bars for `AAPL` ending on `2026-10-20`. These bars were populated by earlier unverified ingestions and lack valid provenance sidecars (`provenance_count = 0`).

The canonical Package B V3 historical ingestion window is:
- **Requested Start Date:** `2025-08-27`
- **Requested End Date:** `2026-10-19`
- **Total Completed NYSE Sessions in Window:** `288 sessions`

### 3.2. The Tail Contamination Failure Mode
If Package B applies Option B Full-History replacement for the window `2025-08-27` to `2026-10-19`:
1. The 288 sessions within the window are overwritten with certified Alpaca SIP data.
2. However, session `2026-10-20` (the 289th session) remains in the store outside the certified window.
3. This creates an **Unprovenanced Tail Remnant**:
   - Total bars in store: `289`
   - Provenance records: `288`
   - Provenance deficit: `1 bar` (`2026-10-20` has unprovenanced or legacy lineage).

### 3.3. Track C Fail-Closed Consumer Verification
Track C implemented and verified `test_10_aapl_october_20_unprovenanced_tail_fails_closed`:
- When AAPL data contains the uncertified 289th session (`2026-10-20`), or exhibits `structural_quality = "DEFECTIVE_OUTSIDE_WINDOW"` / `is_qualified_candidate = False`, Track C's evaluation immediately fails closed:
  - `data_readiness_decision = "DATA_UNAVAILABLE"`
  - `data_readiness_reason = "UNQUALIFIED_PENDING_PACKAGE_B_DATA"`
  - State remains `PROVIDER_RESOLVABLE` (State 2).
  - AAPL is strictly excluded from `published_symbols`.

### 3.4. Package B Production Ingestion Prerequisite
This forensic analysis confirms why **Package B production ingestion remains NOT_AUTHORIZED**. Package B must either:
1. Truncate/delete the unprovenanced `2026-10-20` row prior to replacement, or
2. Formally expand the canonical manifest window to encompass `2026-10-20` if `2026-10-20` market close data is certified.

Until Package B resolves this tail boundary in its release procedure, Track C guarantees that any unprovenanced tail remnant fails closed at the consumer publication boundary.
