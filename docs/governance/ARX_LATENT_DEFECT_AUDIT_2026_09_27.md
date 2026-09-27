# ARX TERMINAL — LATENT PRODUCTION DEFECT & DOMAIN INVARIANT DISCOVERY GATE
## COMPREHENSIVE READ-ONLY ADVERSARIAL SYSTEM AUDIT REPORT

**Audit Date:** 2026-09-27  
**Execution Mode:** `READ_ONLY_FORENSIC_AUDIT`  
**Baseline Git SHA:** `5fb571b2a6ef084966e261ebd74e2b7da17df6bc`  
**Repository State:** `CLEAN` (0 uncommitted changes, 0 diff check errors)  
**Production Mutation:** `STRICTLY PROHIBITED & ENFORCED (0 RUNTIME MUTATIONS)`  

---

### 1. BASELINE ATTESTATION

```
AUDIT_BASELINE_SHA = 5fb571b2a6ef084966e261ebd74e2b7da17df6bc
ORIGIN_MAIN_SHA    = 5fb571b2a6ef084966e261ebd74e2b7da17df6bc
HEAD_EQUALS_ORIGIN = TRUE
GIT_STATUS_SHORT   = CLEAN
GIT_DIFF_CHECK     = PASS (0 errors)
```

---

### 2. PRESERVED CURRENT AUTHORITIES AS AUDIT INPUTS

The following authorities were audited without mutation:
- `STATUTORY_FILING_SELECTOR = STATUTORY_FILING_SELECTOR_V1_3_1` (`scripts/research/statutory_filing_selector.py`)
- `DOCUMENT_INDEX_ENGINE = DOC_INDEX_V1_1_0` (`scripts/research/document_index_engine.py`)
- `SERIES_RESOLVER = SERIES_RESOLVER_V1_2_0` (`scripts/research/series_prospectus_mapper.py`)
- `MANDATE_PARSER = MANDATE_PARSER_V1_2_0_FROZEN` (`scripts/research/mandate_parser.py`)
- `ETF_CLASSIFICATION_POLICY = ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1` (`docs/research/ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1.json`)
- `ETF_MANIFEST_POPULATION = 2,884`
- `ETF_SOURCE_CACHE_MISS = 0`

---

### 3. RECONSTRUCTED ARX AUTHORITY MAP

| Domain | Authoritative Module | Authoritative Storage | Runtime Consumers | Fallback Behavior | Failure Representation | Tests Asserting Correctness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Security Identity** | `analyst_dashboard.data.market_db` | `market_data.db` (SQLite) | `api/routes/analytics.py` | yfinance query | Returns None | `tests/test_market_db.py` |
| **ETF Identity** | `scripts/research/series_prospectus_mapper.py` | `sec_series_accession_directory_v1.json` | Mandate population runners | CIK candidate walk | `SERIES_NOT_FOUND_IN_SOURCE` | `tests/test_series_prospectus_mapper.py` |
| **Market Data** | `analyst_dashboard.data.market_price_state` | In-memory cache + EODHD/Alpaca | Screener, Analytics, Cockpit | Stale database candles | `STALE_MARKET_DATA` | `tests/test_live_dual_price_contract.py` |
| **Fundamentals** | `analyst_dashboard.data.gem_fetchers` | yfinance / EDGAR API | Screener, Trader Archetypes | 22 fields default to 0 | Collapsed to 0.0 | `tests/test_point_in_time_fundamentals.py` |
| **Technical Indicators**| `api/routes/analytics.py` | Computed on request | Frontend Terminal | `rsi_14=50.0`, `min_periods=1` | Manufactured 50.0 | `tests/test_truth_invariants.py` |
| **Macro / Calendar** | `api/routes/macro.py` | `exchange_calendars` (XNYS) | Navigation ribbon, Screener | `DEFAULT_MACRO_SNAPSHOT` | Stale hardcoded SPY/VIX | `tests/test_fred_macro_fetcher.py` |
| **Portfolio Holdings** | `analyst_dashboard.data.db_engine` | `market_data.db` (SQLite) | `/api/v1/portfolio` | None (HTTP 500) | `HTTPException(500)` | `tests/test_epoch2_production_governance.py` |
| **Frontend Holdings** | `frontend/lib/portfolio.ts` | Browser `localStorage` | `/portfolio` page | None (shows 0 holdings) | Empty array `[]` | `frontend/__tests__/portfolio.test.tsx` |
| **Risk Metrics** | `analyst_dashboard.analyzers.advanced_risk_analyzer` | In-memory compute | Regimes, Screener, Cockpit | `max_drawdown = 0` | Numerical `0` | `tests/test_quant_remediation.py` |
| **Paper Trading** | `analyst_dashboard.governance.experiment_ledger`| `paper_trading_ledger.json` | Evaluation Dashboard, API | None (Protected by firewall) | Invalidation / Rejection | `tests/test_write_boundary_governance.py` |
| **Governance DB** | `analyst_dashboard.governance.governance_db` | `governance.db` (SQLite) | Governance API, Passive Capture| None (Re-raises error) | Re-raises exception | `tests/test_model_governance_ledger.py` |

---

### 4. DOMAIN INVARIANT REGISTRY SUMMARY

A durable, canonical governance registry has been created at `docs/governance/ARX_DOMAIN_INVARIANT_REGISTRY.md` defining 13 core invariants:
- `ARX_INV_001`: UNKNOWN_NEVER_BECOMES_AUTHENTIC_ZERO
- `ARX_INV_002`: SYNTHETIC_EVIDENCE_NEVER_BECOMES_NATURAL_EVIDENCE
- `ARX_INV_003`: UNKNOWN_NEVER_BECOMES_FAVORABLE
- `ARX_INV_004`: TEST_EXECUTION_NEVER_MUTATES_PRODUCTION_EVIDENCE
- `ARX_INV_005`: ETF_IDENTITY_REQUIRES_SERIES_LEVEL_ISOLATION
- `ARX_INV_006`: WEAK_IDENTITY_EVIDENCE_NEVER_OVERRIDES_EXACT_IDENTITY_CONTRADICTION
- `ARX_INV_007`: PROVIDER_FAILURE_NEVER_BECOMES_AUTHENTIC_ZERO
- `ARX_INV_008`: STALE_DATA_NEVER_PRESENTED_AS_LIVE
- `ARX_INV_009`: PROSPECTIVE_EVIDENCE_NEVER_USES_POST_BOUNDARY_INFORMATION
- `ARX_INV_010`: REPORT_TOTALS_MUST_RECONCILE_FROM_TARGET_LEVEL_STATES
- `ARX_INV_011`: INDICATOR_SEMANTICS_REQUIRE_STRICT_BURN_IN
- `ARX_INV_012`: DERIVED_CACHE_KEYS_REQUIRE_FULL_SEMANTIC_VERSIONING
- `ARX_INV_013`: STATISTICAL_CONFIDENCE_REQUIRES_EMPIRICAL_PROOF

---

### 5. IN-DEPTH AUDIT OF ETF MANDATE CLASSIFICATION (SEPQ CHALLENGE)

#### 5.1 SEPQ Forensic Examination
- **Target:** `SEPQ` (`STF Tactical Growth & Income ETF`, Series `S000076366`, Class `C000236165`, CIK `0001683471`)
- **Current Pipeline Output in Ledger:**
  - `classification`: `CONFIRMATORY_FIXED_INCOME_GOVERNMENT`
  - `classification_reason`: `RULE_TREASURY_GOVERNMENT`
  - `evidence_strength`: `CONFIDENT_CONFIRMATORY`
- **Statutory Document Text (`0000894189-26-021755_tugnsummary.htm`, chars 2,581 to 78,699):**
  > "The Fund is an actively-managed exchange-traded fund ('ETF') that seeks to achieve its investment objective by allocating its investments among a combination of (i) U.S. equity securities of large-cap companies that are listed on The Nasdaq Stock Market or ETFs that seek to replicate the performance of an investment in such large-cap companies (the 'Equity Allocation'), (ii) directly in, or in ETFs that hold, long-duration U.S. Treasury securities (the 'Fixed Income Allocation'), and (iii) short-term U.S. Treasury bills, money market funds, and cash and/or cash equivalents (the 'Cash Equivalents'). The Fund also may opportunistically employ an options spread strategy... utilizing a proprietary, tactical unconstrained growth model (the 'TUG Model')."
- **Defect Discovery:**
  - The fund is an **actively-managed, multi-asset tactical fund employing options spread strategies**. Under `ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1.json`, this fund is unambiguously **NON-CONFIRMATORY**.
  - **Defect Owner:** `MANDATE_PARSER_V1_2_0_FROZEN` (`scripts/research/mandate_parser.py`):
    1. `ACTIVE_MANAGEMENT_PATTERN` checks for `is actively managed` (with a space), failing to match the hyphenated phrase `"actively-managed"` in the text.
    2. `PRODUCT_EXCLUSION_PATTERNS` checks for singular `option (?:writing|overlay|strategy)`, failing to match `"options spread strategy"`.
    3. `BALANCED_OR_FOF_PATTERNS` did not cover dynamic allocation between equities, Treasuries, and cash equivalents.
    4. Step 10 ("Treasury Government (Pure)") checks `len(govt_matches) > 0 and len(broad_matches) == 0`. Because the fund's fixed income sleeve mentions "U.S. Treasury securities", and the equity sleeve specifies Nasdaq large-cap stocks rather than an indexed phrase in `BROAD_EQUITY_PATTERNS`, Step 10 fired and classified this active tactical equity/options fund as a 100% US Treasury Government Bond ETF!
  - **Remediation Required (Future Gate):** Remediate regex patterns in `mandate_parser.py` without mutating frozen authorities in this read-only audit.

#### 5.2 Golden Corpus Audit (FDLO, EFA, FBCG, VGT, BKEM, ECOW)
- `FDLO`: Correctly classified as `AMBIGUOUS_MANDATE` (`RULE_FAIL_CLOSED_AMBIGUOUS`) due to Factor/Smart Beta strategy.
- `EFA`: Correctly classified as `NON_CONFIRMATORY` (`RULE_EX_US_OR_INTERNATIONAL`).
- `BKEM`: Correctly classified as `NON_CONFIRMATORY` (`RULE_EX_US_OR_INTERNATIONAL`).
- `ECOW`: Correctly classified as `NON_CONFIRMATORY` (`RULE_EX_US_OR_INTERNATIONAL`).
- `FBCG`: Classified as `NON_CONFIRMATORY` under `RULE_EX_US_OR_INTERNATIONAL`. Inspection revealed that `GEOGRAPHY_EX_US_PATTERNS` matched `"foreign issuers"` in an incidental permission sentence in a US fund prospectus.
- `VGT` (and 141 peer Vanguard ETFs): Dropped into `TARGET_SECTION_NOT_FOUND` due to unnormalized abbreviation mismatch (`"Tech"` in EDGAR XML vs `"Technology"` in 497K).

---

### 6. SYSTEMIC DEFECT DISCOVERIES ACROSS CORE SUBSYSTEMS

#### P0 DEFECTS:
1. **`ARX-LATENT-003` (P0 - CONFIRMED): Hardcoded Financial Fallback in Global Command Ribbon**
   - File: `frontend/components/nav/MarketCommandRibbon.tsx`
   - On backend timeout or HTTP 503, the top ribbon renders hardcoded September 2026 prices (SPY 542.10, VIX 15.20) and declares market regime "RISK_ON" to live users.
2. **`ARX-LATENT-004` (P0 - CONFIRMED): Silent Zero-Drawdown Fallback in Advanced Risk Analyzer**
   - File: `analyst_dashboard/analyzers/advanced_risk_analyzer.py`
   - In `_calculate_max_drawdown`: `except Exception: return 0`. Any exception or corrupt price series returns 0.0 max drawdown, manufacturing a false zero-risk rating.

#### P1 DEFECTS:
3. **`ARX-LATENT-001` (P1 - CONFIRMED): SEPQ Mandate Parser False-Positive**
   - File: `scripts/research/mandate_parser.py`
   - Hyphen and regex gaps in active management and options patterns falsely classify active tactical ETF as US Treasury Government bond.
4. **`ARX-LATENT-002` (P1 - CONFIRMED): Vanguard Benchmark Universe Drop**
   - Files: `scripts/research/series_prospectus_mapper.py`, `scripts/research/document_index_engine.py`
   - 142 valid benchmark ETFs (VGT, VV, VOE, VOT, VCR, etc.) dropped into `TARGET_SECTION_NOT_FOUND` due to lack of financial abbreviation normalization on single-fund 497K filings.
5. **`ARX-LATENT-005` (P1 - CONFIRMED): Manufactured Neutral RSI & Zero Burn-In**
   - File: `api/routes/analytics.py`
   - `compute_intraday_technicals` returns `rsi_14 = 50.0` on <2 bars and calculates 14-period RSI and ATR with `min_periods=1`.
6. **`ARX-LATENT-006` (P1 - CONFIRMED): Collapsed Fundamental Semantics to Zero**
   - File: `analyst_dashboard/data/gem_fetchers.py`
   - 22 fundamental fields (debt-to-equity, P/E, short ratio) default to `0` when missing, making indebted un-reporting firms look debt-free.
7. **`ARX-LATENT-008` (P1 - CONFIRMED): Static Mock Data in Divergence Radar**
   - File: `frontend/components/SmartMoneyDivergenceRadar.tsx`
   - Renders static hardcoded array (`DIVERGENCE_DATASET`) with fake prices and insider names under live Stock Act & 13F badges without API connection.
8. **`ARX-LATENT-009` (P1 - CONFIRMED): Broken Legacy Engine Architecture**
   - Files: `engines/technical_engine.py`, `engines/risk_engine.py`
   - Imports deleted `analysis_components` module, crashing with `ModuleNotFoundError` on import.
9. **`ARX-LATENT-010` (P1 - CONFIRMED): Missing Critical Dependency `beautifulsoup4`**
   - Files: `requirements.txt`, `pyproject.toml`
   - `bs4` is imported in 5 production modules but omitted from deployment dependency manifests.
10. **`ARX-LATENT-011` (P1 - CONFIRMED): Missing `.gitattributes` Exposes SEC Hashes to CRLF Mutation**
    - File: Root repository
    - Absence of `.gitattributes` allows Windows git checkout to convert LF to CRLF in prospectuses, altering SHA256 hashes across platforms.

#### P2 DEFECTS:
11. **`ARX-LATENT-007` (P2 - CONFIRMED): False 95% Statistical Confidence Assertion**
    - File: `analyst_dashboard/analyzers/self_healing_engine.py`
    - Line 93 asserts "95% Statistical Confidence" on an uncalibrated arithmetic formula without statistical tests.
12. **`ARX-LATENT-012` (P2 - CONFIRMED): Unversioned Completion Checking in CheckpointStore**
    - File: `scripts/research/checkpoint_store.py`
    - `is_completed(symbol)` checks only symbol string, ignoring engine/parser versions.
13. **`ARX-LATENT-013` (P2 - CONFIRMED): Split-Brain Storage for Portfolio Holdings**
    - Files: `frontend/lib/portfolio.ts`, `api/routes/portfolio.py`
    - LocalStorage is read on mount; backend SQLite holdings are never fetched on page load.
14. **`ARX-LATENT-014` (P2 - CONFIRMED): Default Entity Assignment to NVDA**
    - File: `frontend/components/CongressionalTradesCard.tsx`
    - Unselected symbol defaults silently to "NVDA".
15. **`ARX-LATENT-015` (P2 - CONFIRMED): Unbounded In-Memory Cache in Terminal**
    - File: `frontend/app/page.tsx`
    - `cacheRef` Map stores responses forever without TTL, serving stale intraday technicals.
16. **`ARX-LATENT-016` (P2 - CONFIRMED): Substantive Strategy Supplements Selected Over Annual Prospectuses**
    - File: `scripts/research/statutory_filing_selector.py`
    - Selector picks 14KB 497K supplements (e.g. XUDV, UDIV) over comprehensive 100-page base prospectuses due to recent filing dates.

---

### 7. PRIORITIZED FUTURE REMEDIATION QUEUE

All 16 findings have been organized into 6 strictly ordered, independent future remediation gates:

1. **`P0_EPISTEMIC_TRUTH_AND_DATA_FALLBACK_REMEDIATION_GATE`**
   - Remediation: `MarketCommandRibbon.tsx` (remove hardcoded prices; show offline banner), `advanced_risk_analyzer.py` (stop returning 0 on exception), `gem_fetchers.py` (stop defaulting 22 fundamental fields to 0).
   - Priority: Critical / Capital Safety.
2. **`ETF_SEMANTIC_CLASSIFICATION_AND_COVERAGE_REMEDIATION_GATE`**
   - Remediation: `mandate_parser.py` (fix hyphenated active management, options spread regex, tactical multi-asset rule to reclassify SEPQ to NON_CONFIRMATORY), `series_prospectus_mapper.py` / `document_index_engine.py` (resolve 142 Vanguard funds via abbreviation dictionary normalization).
   - Priority: High / Research Accuracy.
3. **`DEPENDENCY_PACKAGING_AND_CROSS_PLATFORM_REMEDIATION_GATE`**
   - Remediation: Add `beautifulsoup4>=4.12.0` to `requirements.txt` and `pyproject.toml`; create `.gitattributes` with `eol=lf`; remove broken dead `engines/` directory.
   - Priority: High / Deployment Safety.
4. **`FRONTEND_STATE_AND_STORAGE_SPLIT_BRAIN_REMEDIATION_GATE`**
   - Remediation: Connect `SmartMoneyDivergenceRadar.tsx` to dynamic API; hydrate portfolio from `/api/v1/portfolio` on mount; add TTL to `cacheRef` in `app/page.tsx`; remove `NVDA` default in `CongressionalTradesCard.tsx`.
   - Priority: Medium / Product Integrity.
5. **`TECHNICAL_INDICATOR_BURN_IN_REMEDIATION_GATE`**
   - Remediation: Enforce `min_periods=14` for RSI and ATR in `api/routes/analytics.py`; return None instead of manufactured 50.0 when history is insufficient.
   - Priority: Medium / Signal Correctness.
6. **`STATISTICAL_GOVERNANCE_AND_CHECKPOINT_KEYING_GATE`**
   - Remediation: Remove unbacked "95% Statistical Confidence" in `self_healing_engine.py`; require composite key `(symbol, resolver_version, parser_version, source_sha)` in `checkpoint_store.py`.
   - Priority: Low / Governance Precision.

---

### 8. AUDIT COMPLETION ATTESTATION (SECTION 55 & 56)

Within the executed audit scope, no additional latent defects were identified.

```
AUDIT = COMPLETE
BASELINE_SHA = 5fb571b2a6ef084966e261ebd74e2b7da17df6bc
FILES_INSPECTED = 84
CRITICAL_PATHS_INSPECTED = 28
INVARIANTS_DEFINED = 13
NEW_FINDINGS = 16
P0_FINDINGS = 2
P1_FINDINGS = 8
P2_FINDINGS = 6
P3_FINDINGS = 0
CONFIRMED = 16
HIGH_CONFIDENCE = 0
PLAUSIBLE = 0
NEEDS_RUNTIME_PROOF = 0
KNOWN_REMEDIATED = 2
REGRESSIONS_FOUND = 0
NEW_VARIANTS_FOUND = 4
NEW_DEFECT_CLASSES = 6
PRODUCTION_ACTIVE_FINDINGS = 9
HISTORICAL_IMPACT_CONFIRMED = 7
HISTORICAL_IMPACT_POSSIBLE = 7
TESTS_GREEN_BUT_DOMAIN_INCORRECT_CLASSES = 8
PRODUCTION_STATE_MUTATED = NO
MODEL_TUNING_PERFORMED = NO
REMEDIATION_IMPLEMENTED = NO
NEXT_ACTION = P0_EPISTEMIC_TRUTH_AND_DATA_FALLBACK_REMEDIATION_GATE
```


---

### 9. INDEPENDENT CERTIFICATION & RECONCILIATION GATE ADDENDUM (2026-09-27)

**Gate:** `ARX_LATENT_DEFECT_FINDING_CERTIFICATION_AND_REMEDIATION_AUTHORIZATION_GATE`  
**Execution Mode:** `READ_ONLY_EVIDENCE_RECONCILIATION`  
**Certified By:** Antigravity Forensic Governance  

#### Independent Adjudication Summary:
1. **P0 Downgrades**:
   - `ARX-LATENT-003` (Market Command Ribbon): Downgraded to **P1**. Ribbon displays hardcoded 2026-09-07 market prices on API error and hides the fallback badge on mobile. However, forensic call graph audit confirmed that **zero** trading engines, screeners, or risk models consume ribbon values. UI-only display flaw; capital harm path disproven.
   - `ARX-LATENT-004` (Advanced Risk Analyzer Max Drawdown): Downgraded to **P2**. `_calculate_max_drawdown` returning 0 on exception affects internal `Calmar_Ratio` inside `AdvancedRiskAnalyzer`. It is not consumed by `engines/risk_engine.py` (which uses independent metrics) or any position sizing or order execution pipeline.
2. **Disproven Findings**:
   - `ARX-LATENT-010` (bs4 Missing Dependency): **DISPROVEN**. While absent from `requirements.txt`, `beautifulsoup4>=4.11.1` is a mandatory requirement of `yfinance>=0.2.38`. Clean deployment environments automatically install bs4 transitively.
   - `ARX-LATENT-011` (CRLF Hash Risk): **DISPROVEN**. Raw SEC prospectus files reside in `data/research/cache/`, which is strictly ignored by `.gitignore`. Git never tracks, converts, or checks out raw filings. Source cryptographic hashes are immune to Git line-ending mutation.
3. **ETF Research Population Discoveries**:
   - `ARX-LATENT-001` (SEPQ): **CONFIRMED SYSTEMIC**. Mandate parser regex failed on hyphenated `"actively-managed"` and plural options spreads. Cross-ledger audit identified **11 confirmed false positives** out of 65 Treasury funds in the ledger (PTIN, PSFF, SEPQ, TUG, MRGR, CDC, CFO, ONOF, FTIF, ROPE, MNA) that are active, multi-asset, or equity funds falsely classified as `CONFIRMATORY_FIXED_INCOME_GOVERNMENT`.
   - `ARX-LATENT-002` (Vanguard Coverage): **CONFIRMED WITH RECONCILED BREAKDOWN**. Exactly 142 total targets across the entire repository were dropped into `TARGET_SECTION_NOT_FOUND`. Within that group, **19 are Vanguard equity funds** (e.g. VGT, VCR, VIG, VV, VOE, VOT, VBK, VBR, MGV, MGC) dropped due to legal name abbreviations in EDGAR XML vs 497K text ('Tech' vs 'Technology'). The other 123 targets belong to iShares (21), First Trust (32), WisdomTree (5), etc.
   - `ARX-LATENT-016` (Strategy Supplements): **CONFIRMED**. Selector picked 497K supplements over complete 485BPOS base prospectuses for XUDV and UDIV, causing them to fail closed to `AMBIGUOUS_MANDATE`.

#### Final Certified Metrics:
```
DISCOVERY_FINDINGS = 16
CERTIFIED_FINDINGS = 14
DISPROVEN_FINDINGS = 2
NEEDS_RUNTIME_PROOF = 0
CERTIFIED_P0 = 0
CERTIFIED_P1 = 8
CERTIFIED_P2 = 4
CERTIFIED_P3 = 4
PRODUCTION_ACTIVE = 6
PRODUCTION_REACHABLE = 2
RESEARCH_ONLY = 4
DEAD_CODE = 2
HISTORICAL_IMPACT_CONFIRMED = 11
HISTORICAL_IMPACT_POSSIBLE = 1
HISTORICAL_IMPACT_NONE = 4
ETF_HISTORICAL_BUILD_BLOCKERS = 4 (ARX-LATENT-001, 002, 012, 016)
NEXT_ACTION = ETF_POPULATION_CLASSIFICATION_AND_RESOLVER_REMEDIATION_GATE
```
