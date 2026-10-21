# ARX TERMINAL — SPRINT 2 TRACK C
# SCANNABILITY ACCOUNTING SEMANTICS & IDENTITY ISOLATION

## 1. Accounting Clarification: Scannability vs. Scanner Enrollment

In prior Track C artifacts, the term `scannable_count` created confusion regarding whether published securities were automatically enrolled into active radar scanners. Under ARX Closure Model v2, this semantic boundary is now formally codified in `AccountingClosureSummary`.

### 1.1. Precise Definitions

| Accounting Field | State Boundary | Definition & Invariant | Sprint 2 Value |
| :--- | :--- | :--- | :--- |
| `source_count` | Pre-State 1 | Total unique candidates evaluated from input source. | 4 (pilot cohort) |
| `eligible_count` | State 4 | Candidates passing R01-R07 eligibility rules. | 4 |
| `data_ready_count` | State 3 | Candidates with certified market data meeting lookback floors. | 4 (post-replacement) |
| `scannable_count` | State 4 / 5 | **POTENTIAL Candidate Scannability:** Candidates that have completed data qualification and eligibility (`scannable_count == data_ready_count`). Denotes theoretical readiness for strategy evaluation. | 4 |
| `published_count` | State 5 | Candidates published to the active in-memory universe pointer. | 4 |
| `strategy_admitted_count` | State 6 | Candidates admitted to a specific trading strategy (e.g. MINERVINI_VCP under `G-VCP-012` or SMART_MONEY). Strictly governed. | **0** |
| `scanner_enrolled_count` | State 7 | Candidates actively dispatched to a live scanning schedule. Strictly requires `G-S2-ACT` operational authority. | **0** |

### 1.2. Consumer Protection Firewall
Track C enforces `IsolatedPublicationEngine.assert_no_scanner_enrollment(published_symbols, active_scanner_symbols)`:
- If any published symbol appears in `active_scanner_symbols`, an immediate `PolicyViolationError` ("Actionability leakage detected") is raised.
- Verified in `test_12_published_only_zero_scanner_enrollment`:
  - `active_scanner_symbols = set()` -> **PASS**
  - `active_scanner_symbols = {"AAPL"}` -> **RAISES PolicyViolationError**

---

## 2. GOOG / GOOGL Multi-Share-Class Identity Isolation

### 2.1. The Capital Structure Context
Alphabet Inc. maintains a dual-class share structure listed on NASDAQ (MIC: `XNAS`):
- **GOOG**: Class C Capital Stock (no voting rights, distinct CUSIP/ISIN).
- **GOOGL**: Class A Common Stock (one vote per share, distinct CUSIP/ISIN).

### 2.2. Prior Methodological Flaw
Earlier Track C prototype code attempted to verify share-class separation by comparing numerical price series (`goog_prices != googl_prices`). This was a false premise:
- Market prices between share classes of the same issuer frequently converge, trade in parity, or cross momentarily.
- Numerical price equality does not indicate identity corruption.
- Testing price inequality resulted in fragile, non-authoritative heuristics.

### 2.3. Authoritative Identity-Based Verification
Under ARX Closure Model v2, share-class integrity is determined strictly by authoritative master records and listing associations:

```python
@classmethod
def verify_multi_share_class_isolation(
    cls,
    goog_cand: IsolatedUniverseCandidate,
    googl_cand: IsolatedUniverseCandidate,
    goog_prices: Optional[List[float]] = None,
    googl_prices: Optional[List[float]] = None,
    test_fixture_alias_collision_check: bool = False,
) -> None:
    # 1. Distinct internal security IDs
    if goog_cand.internal_security_id == googl_cand.internal_security_id:
        raise PolicyViolationError(...)
    # 2. Distinct listing association IDs
    if goog_cand.association_id and googl_cand.association_id:
        if goog_cand.association_id == googl_cand.association_id:
            raise PolicyViolationError(...)
    # 3. Distinct symbols
    if goog_cand.symbol == googl_cand.symbol:
        raise PolicyViolationError(...)
    # 4. Distinct provider locator symbols
    if goog_cand.alpaca_symbol and googl_cand.alpaca_symbol and goog_cand.alpaca_symbol == googl_cand.alpaca_symbol:
        raise PolicyViolationError(...)
```

### 2.4. Verification Evidence in Test Suite
In `test_11_goog_googl_share_class_separation_and_pricing`:
1. Authoritative candidate bindings verified:
   - `GOOG` -> `internal_security_id="ARX_SEC_A11A24D442E2"`, `association_id="ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"`, symbol `"GOOG"`
   - `GOOGL` -> `internal_security_id="ARX_SEC_B7EF9ED7A35A"`, `association_id="ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"`, symbol `"GOOGL"`
2. Identical price series (`[180.25, 180.50, 181.00]` for both) passed to `verify_multi_share_class_isolation` passes without error.
3. Artificial collision of `internal_security_id` immediately raises `PolicyViolationError("Share class collision")`.
