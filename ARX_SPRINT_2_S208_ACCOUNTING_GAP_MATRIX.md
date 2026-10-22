# ARX Terminal — Sprint 2 S2-08 Accounting Gap Matrix

**Governing Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Task:** S2-08 — Full-Population Assessment & Outcome Accounting  
**Status:** **AUDITED / RECONCILED**  
**Date:** October 22, 2026  
**Implementation Branch:** `feat/sprint-2-s208-full-population-instrumentation`  
**Delivered Commit:** `488d9de493048c7058c5b18c8fdc5d386209230f`  

---

## 1. Denominator Transition & Accounting Gap Matrix

This matrix maps each of the **11 distinct accounting quantities** in the full-population assessment lifecycle to its authoritative producer, field name, persistence mode, verification state, and remediation requirement.

| # | Accounting Quantity | Canonical Producer | Data Contract & Field | Consumer | Persistence | Verification State | Remediation Requirement |
| :-: | :--- | :--- | :--- | :--- | :-: | :-: | :--- |
| **1** | **Source Observations ($N_{\text{obs}}$)** | `CoverageDiagnostics` / Raw feed adapters | `DiagnosticInputRecord.record_locator` | Universe Ingestion | File / Disk | `IMPLEMENTED_VERIFIED` | None. Multi-feed locators and file hashes preserved. |
| **2** | **Distinct Lookup Keys ($N_{\text{keys}}$)** | `load_acquisition_artifacts_to_records()` | `CoverageAccountingSummary.distinct_lookup_keys` | Security Master | Persistent | `IMPLEMENTED_VERIFIED` | None. Deduplicates multi-venue listings by uppercase symbol. |
| **3** | **Validated Identities ($N_{\text{identities}}$)** | `sec_edgar.py` / `openfigi.py` | `CanonicalSecurity.security_id` (`ARX_SEC_*`) | Identity Store | Database | `IMPLEMENTED_VERIFIED` | None. Mapped to SEC CIK / OpenFIGI; fails closed if ambiguous. |
| **4** | **Policy-Eligible ($N_{\text{eligible}}$)** | `UniverseBuilder.evaluate_eligibility()` | `EligibilityLedgerRow.decision` (`ELIGIBLE`) | Universe Builder | Database | `IMPLEMENTED_VERIFIED` | None. Rules R01–R07 enforce Common Stock on US major exchanges. |
| **5** | **Committed Published ($N_{\text{published}}$)** | `IsolatedPublicationStateMachine` | `IsolatedUniverseCandidate.active_build_cardinality` | Publication Pointer | Database | `IMPLEMENTED_VERIFIED` | S2-06 remediated. Cardinality reflects committed pointer only. |
| **6** | **Scanner-Enrolled ($N_{\text{enrolled}}$)** | `ScannerPublicationMembershipConsumer` | `membershipStatus == 'AVAILABLE'`, `is_scannable` | Scanner Dispatch | In-Memory / DB | `IMPLEMENTED_VERIFIED` | S2-07 implemented. Five-tier lifecycle isolates unadmitted scanners. |
| **7** | **Dispatch-Attempted ($N_{\text{dispatch}}$)** | `VCPScannerRunner.execute_market_wide_scan()` | `run_id`, `execution_attempt_id` | Optimal Execution | Transient / Log | `IMPLEMENTED_VERIFIED` | S2-08 added explicit dispatch count tracking in summary. |
| **8** | **Assessment-Completed ($N_{\text{completed}}$)** | `VCPScannerRunner` / `FullPopulationAssessmentEngine` | `scanned_successfully_count`, `n_completed` | Scanner Snapshot | Persistent | `IMPLEMENTED_VERIFIED` | S2-08 closed: $N_{\text{completed}} = N_{\text{matches}} + N_{\text{nonmatches}}$. |
| **9** | **Assessment-Failed / Excluded** | `VCPScannerRunner` / `FullPopulationAssessmentEngine` | `unavailable_reasons`, `n_failed_or_excluded` | Scanner Snapshot | Persistent | `IMPLEMENTED_VERIFIED` | S2-08 closed: aggregates failed, data-insufficient, & excluded. |
| **10** | **Authoritative Matches ($N_{\text{matches}}$)** | `VCPScannerRunner` / Confluence Engine | `matched_count`, `authoritative_matches` | Snapshot Envelope | Persistent | `IMPLEMENTED_VERIFIED` | Strict integer type (`int >= 0`), rejects boolean/negative values. |
| **11** | **Frontend-Displayed ($N_{\text{displayed}}$)** | `frontend/app/radar/page.tsx` | `filteredAssets.length` | Operator UI | Ephemeral Client | `IMPLEMENTED_VERIFIED` | Strictly decoupled from `matchedCount` in UI display banner. |

---

## 2. Terminal Outcome Taxonomy & Failure Precedence Matrix

Every security dispatched to the assessment pipeline must resolve to exactly **one** terminal outcome. Precedence is applied fail-closed: earlier errors dominate downstream calculations.

| Outcome Code | Canonical Precedence Tier | Existing Authority Reference | Category in Formula | Verification State | Failure Precedence Rule |
| :--- | :---: | :--- | :--- | :-: | :--- |
| `EXECUTION_TIMEOUT` | **1** | `DurableRunCoordinator` | $N_{\text{failed}}$ | `IMPLEMENTED_VERIFIED` | Lease expiry aborts run; prevents partial completion assertion. |
| `RUNTIME_ERROR` | **2** | `VCPScannerRunner` | $N_{\text{failed}}$ | `IMPLEMENTED_VERIFIED` | Unhandled calculation or schema fault captured in outcome ledger. |
| `AMBIGUOUS_IDENTITY` | **3** | `CoverageDiagnostics` | $N_{\text{unsupported}}$ | `IMPLEMENTED_VERIFIED` | Missing or conflicted CIK/FIGI fails closed to unsupported. |
| `POLICY_INELIGIBLE` | **4** | `UniverseBuilder` | $N_{\text{excluded}}$ | `IMPLEMENTED_VERIFIED` | Excluded by Rules R01–R07 prior to market data query. |
| `UNENROLLED` | **5** | `MembershipConsumer` | $N_{\text{excluded}}$ | `IMPLEMENTED_VERIFIED` | Published security excluded from scanner strategy scope. |
| `DISPATCH_DEFERRED` | **6** | `VCPScannerRunner` | $N_{\text{excluded}}$ | `IMPLEMENTED_VERIFIED` | Deferred by concurrency throttling or worker capacity limits. |
| `MISSING_MARKET_DATA` | **7** | `CoverageDiagnostics` | $N_{\text{insufficient\_data}}$ | `IMPLEMENTED_VERIFIED` | No price or daily bar records exist in market store. |
| `INSUFFICIENT_HISTORY` | **8** | `CoverageDiagnostics` | $N_{\text{insufficient\_data}}$ | `IMPLEMENTED_VERIFIED` | Less than 50 trading days of historical daily candles. |
| `INVALID_OHLCV` | **9** | `CoverageDiagnostics` | $N_{\text{insufficient\_data}}$ | `IMPLEMENTED_VERIFIED` | Non-positive prices, negative volume, or High < Low contradiction. |
| `STALE_QUOTE` | **10** | `CoverageDiagnostics` | $N_{\text{insufficient\_data}}$ | `IMPLEMENTED_VERIFIED` | Quote timestamp does not match target market session date. |
| `COMPLETED_MATCH` | **11** | `OptimalExecutionEngine` | $N_{\text{qualifying\_or\_developing}}$ | `IMPLEMENTED_VERIFIED` | VCP Stage 2 Advancing + Confluence threshold satisfied. |
| `COMPLETED_NONMATCH` | **12** | `OptimalExecutionEngine` | $N_{\text{no\_setup}}$ | `IMPLEMENTED_VERIFIED` | Valid technical assessment completed; pattern criteria not met. |

---

## 3. Operational Telemetry Gap & Readiness Matrix

This matrix evaluates existing operational measurement capabilities against S2-08 production requirements:

| Operational Metric | Required Unit | Producer Location | Current Telemetry State | Offline Feasibility | Production Remediation Requirement |
| :--- | :--- | :--- | :-: | :-: | :--- |
| **Assessment Duration** | `milliseconds` | `CapacityBenchmarker` | `IMPLEMENTED_VERIFIED` | Yes | Recorded per scan run in `AssessmentTelemetryEnvelope`. |
| **Throughput** | `securities/second` | `CapacityBenchmarker` | `IMPLEMENTED_VERIFIED` | Yes | Calculated as $N_{\text{dispatch}} / \text{elapsed\_seconds}$. |
| **Queue / Execution Wait** | `milliseconds` | Distributed Coordinator | `PARTIAL` | `NOT_MEASURED` | In-memory coordinator measures lease acquisition; network queue wait requires live scheduler. |
| **Data-Access Latency** | `milliseconds` | `ReadOnlyMarketStore` | `IMPLEMENTED_VERIFIED` | Yes | Measured via `measure_store_queries_60()` in benchmarker. |
| **CPU Utilization** | `percentage` | Host OS / psutil | `ABSENT` | `NOT_MEASURED` | Reported honestly as `NOT_MEASURED` offline; live background agent hook required in S2-10. |
| **Memory RSS** | `bytes` | Host OS / psutil | `ABSENT` | `NOT_MEASURED` | Reported honestly as `NOT_MEASURED` offline; live worker hook required in S2-10. |
| **Worker Concurrency** | `worker_count` | `DurableRunCoordinator` | `PARTIAL` | `NOT_MEASURED` | Single-worker in offline tests; distributed pool requires State 7 live activation. |
| **Timeout Rate** | `integer_count` | `DurableRunCoordinator` | `IMPLEMENTED_VERIFIED` | Yes | Monitored via lease renewal timeouts. |
| **Retry Rate** | `integer_count` | `DurableRunCoordinator` | `IMPLEMENTED_VERIFIED` | Yes | Monitored via coordinator retry policy counters. |
| **Cache Hits / Misses** | `integer_count` | Market Data Cache | `ABSENT` | `NOT_MEASURED` | Reported honestly as `NOT_MEASURED` offline; requires live Redis/in-memory cache layer. |
| **Completion Fraction** | `ratio_0_to_1` | `FullPopulationAssessmentEngine` | `IMPLEMENTED_VERIFIED` | Yes | Evaluated as $N_{\text{completed}} / N_{\text{dispatch}}$ (0.0 to 1.0). |

---

## 4. Arithmetic Closure Proof

The canonical accounting formula:
$$N_{\text{population}} = N_{\text{no\_setup}} + N_{\text{qualifying\_or\_developing}} + N_{\text{excluded}} + N_{\text{unsupported}} + N_{\text{insufficient\_data}} + N_{\text{failed}}$$

Is verified by `FullPopulationAssessmentEngine` with strict invariant checks:
1. **$N_{\text{population}}$:** Exact cardinality of defined population input set.
2. **$N_{\text{no\_setup}}$:** Sum of all `COMPLETED_NONMATCH` outcomes.
3. **$N_{\text{qualifying\_or\_developing}}$:** Sum of all `COMPLETED_MATCH` outcomes.
4. **$N_{\text{excluded}}$:** Sum of `POLICY_INELIGIBLE` + `UNENROLLED` + `DISPATCH_DEFERRED`.
5. **$N_{\text{unsupported}}$:** Sum of `AMBIGUOUS_IDENTITY`.
6. **$N_{\text{insufficient\_data}}$:** Sum of `MISSING_MARKET_DATA` + `INSUFFICIENT_HISTORY` + `INVALID_OHLCV` + `STALE_QUOTE`.
7. **$N_{\text{failed}}$:** Sum of `RUNTIME_ERROR` + `EXECUTION_TIMEOUT`.

**Residual Closure Delta:**
$$\Delta = \left(\sum \text{categories}\right) - N_{\text{population}} \equiv 0$$
If $\Delta \neq 0$, or if any security appears multiple times or is missing, closure fails closed with `closure_verified = False`.

---

## 5. Audit Conclusion

All offline accounting gaps for S2-08 are resolved. Operational telemetry reporting follows the fail-closed `"NOT_MEASURED"` principle for unavailable infrastructure metrics. No further code modifications are required for S2-08 offline readiness.
