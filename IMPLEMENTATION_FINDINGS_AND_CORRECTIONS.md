# ARX TERMINAL — IMPLEMENTATION FINDINGS AND CORRECTIONS
**Bounded Fundamental Evidence & Screener Qualification Remediation**  
*Branch: `fix/undervalued-growth-pilot` | Worktree: `finance-undervalued-growth-pilot`*  
*Base Commit: `2ac200b393dafaaf23fff90f8b5e6824e2c3ef96` | Audit Date: October 19, 2026*

---

## 1. Executive Summary of Confirmed Findings

Following the forensic audit of ARX Terminal's current fundamental coverage, four specific systemic defects were identified, isolated, and remediated in bounded local implementation:

### 1.1 Finding 4.1: Static Fundamental Data & Synthetic Zeroing
- **Defect**: Assets missing from the hardcoded `KNOWN_GEMS_DATA` dictionary in `analyst_dashboard/analyzers/gem_screener.py` had their fundamental valuation metrics (`peg_ratio`, `roic_pct`, `gross_margin_pct`) synthesized as `0.0`.
- **Epistemic Harm**: In financial analysis, `0.0` represents an extreme measured value (e.g., zero return on invested capital or infinite earnings multiple), not missing data. Downstream filters treating `peg_ratio <= 1.05` either misclassified missing data or failed open.
- **Remediation**: Replaced synthetic `0.0` with explicit `None` across `peg_ratio`, `roic_pct`, and `gross_margin_pct`. Updated composite score comparator to safely handle `None` entries without type errors.

### 1.2 Finding 4.2: Whitelist Bypass of Quantitative Qualification
- **Defect**: In `api/routes/screener.py`, `derive_canonical_radar_categories` contained an uninspected bypass:
  ```python
  if expert_model in VALUE_GARP_MODELS:
      categories.append("VALUE_GARP")
  ```
  This allowed any stock assigned a static archetype label (such as `"Peter Lynch GARP Compounder"`) to be stamped with the `VALUE_GARP` discovery tag regardless of actual valuation metrics (e.g. `PWR` with PEG 1.30, `ETN` with PEG 1.25).
- **Epistemic Harm**: Equated descriptive archetype labels with quantitative undervaluation.
- **Remediation**: Eliminated the archetype whitelist bypass. `VALUE_GARP` membership now strictly requires quantitative verification:
  - Standard GARP: `0.0 < peg_ratio <= 1.05` AND `roic_pct >= 15.0%`.
  - Elite Capital Efficiency: `roic_pct >= 28.0%` AND `0.0 < peg_ratio <= 1.15`.
  - Missing fundamental data explicitly yields unassigned/pending status.

### 1.3 Finding 4.3: Delisted / Inactive Security Status Handling (`CPRX`)
- **Defect**: Catalyst Pharmaceuticals, Inc. (`CPRX`) was acquired in an all-cash merger by Angelini Pharma S.p.A. at $31.50 per share (Form 25-NSE filed July 10, 2026; Form 15-12B filed July 20, 2026). It remains in `market_db` with a frozen closing price of $31.49.
- **Epistemic Harm**: ARX continued to display CPRX with active technical patterns (`VCP`, `STAGE_2`), active discovery categories (`VALUE_GARP`), and actionable execution levels (`BUY_ZONE`, `STOP_LOSS`, `PROFIT_TARGET`).
- **Remediation**:
  - Implemented `CONFIRMED_CORPORATE_ACTIONS` registry in `api/routes/screener.py`.
  - Preserved historical database records and last traded price ($31.49).
  - Assigned explicit non-actionable execution status: `executionStatus = "DELISTED_SECURITY"`, `statusLabel = "🛑 Delisted / Inactive"`, `isInvestable = False`, `screeningEligible = False`.
  - Stripped CPRX of all active discovery categories (`categories = []`) and suppressed entry/stop/target execution prices to `0.0` with explanatory metadata.
  - Added dedicated frontend badge styling (`bg-rose-950/80 text-rose-300 border-rose-800/50`) and status parsing in `frontend/app/radar/page.tsx`.

### 1.4 Finding 4.4: Confluence Score vs. Valuation / Margin of Safety Sorting
- **Defect**: The Radar UI sorted `VALUE_GARP` candidates by `confluenceScore`. Confluence score is a composite technical-flow metric (25% technical structure, 25% fundamental solvency, 25% smart money flow, 25% macro trend) heavily influenced by RSI, moving average proximity, and institutional order blocks.
- **Epistemic Harm**: Users naturally interpreted the top-ranked stock in the "Value / GARP" tab as the "most undervalued" stock with the largest margin of safety, when in reality it was simply the asset with the strongest technical confluence near its execution pivot.
- **Remediation**:
  - Added an explicit `VALUE_GARP` truthfulness disclosure banner in `frontend/app/radar/page.tsx`:
    > *"Curated Reference Catalog & Technical Confluence: Assets in this view originate from a curated reference baseline with verified fundamental metrics (PEG ≤ 1.05, ROIC ≥ 15%). Rankings reflect technical confluence and execution setup proximity, not intrinsic valuation discount or margin of safety."*
  - Relabeled the sort button to `"Score (Confluence)"` with descriptive tooltips.
  - Formally updated `RADAR_CAPABILITY_CONTRACT["VALUE_GARP"]` in `api/routes/screener.py` with `catalogType: "CURATED_REFERENCE_CATALOG"` and `sortingSemantics: "CONFLUENCE_TECHNICAL_FLOW"`.
  - Exposed structured `fundamentalEvidence` metadata blocks on all screener assets.

---

## 2. Exact Files & Components Modified

| File Path | Component / Function | Modification Description |
| :--- | :--- | :--- |
| `analyst_dashboard/analyzers/gem_screener.py` | `evaluate_candidates` | Replaced synthetic `0.0` fallbacks with `None` for uncataloged assets. Updated composite score sort key to safely handle `None` values. |
| `api/routes/screener.py` | `CONFIRMED_CORPORATE_ACTIONS` | Added authoritative corporate actions registry documenting CPRX cash merger ($31.50/share). |
| `api/routes/screener.py` | `derive_canonical_radar_categories` | Removed whitelist bypass. Added delisting filter. Enforced strict quantitative verification (`0.0 < peg <= 1.05` and `roic >= 15.0%`). |
| `api/routes/screener.py` | `get_radar_screener` | Enforced `DELISTED_SECURITY` execution status, suppressed actionable levels, stripped categories, and populated `fundamentalEvidence` metadata block. |
| `api/routes/screener.py` | `RADAR_CAPABILITY_CONTRACT` | Updated `VALUE_GARP` capability contract with explicit truthfulness disclosures (`CURATED_REFERENCE_CATALOG`, `CONFLUENCE_TECHNICAL_FLOW`). |
| `frontend/app/radar/page.tsx` | `parseCanonicalExecutionStatus`, `formatRadarExecutionStatus` | Added `DELISTED_SECURITY` parsing, label mapping, and badge rendering. Added `VALUE_GARP` disclosure banner and relabeled sort button. |
| `tests/test_screener_execution.py` | `test_non_actionable_levels_are_suppressed` | Added `DELISTED_SECURITY` to non-actionable suppression assertions. |
| `tests/test_undervalued_growth_corrections.py` | *New test suite* | 6 new rigorous regression tests covering archetype label truthfulness, delisting exclusion, `None` metric serialization, and contract disclosures. |

---

## 3. Preservation of Protected Architectural Boundaries

All changes were strictly constrained outside protected authorities:
- **`optimal_execution.py`**: **Unchanged**. Frozen hash preserved. No modifications to execution engine or pattern recognition formulas.
- **`confluence_engine.py`**: **Unchanged**. Frozen hash preserved. Confluence scoring weights and calculation logic remained untouched.
- **`decision_hierarchy.py`**: **Unchanged**. Frozen hash preserved. Conflict resolution hierarchy remained intact.
- **`scanner_contract.py`**: **Unchanged**. Canonical market-wide scanner store contract remained untouched.
- **Release Manifests & Lockfiles**: **Unchanged**. No package additions, no lockfile mutations.
- **Production Database**: **Untouched**. Zero mutations or migrations applied to production databases.

---

## 4. Verification Evidence & Test Execution

### 4.1 Backend Python Regression Suite
Executed:
```bash
pytest tests/test_undervalued_growth_corrections.py tests/test_screener_execution.py tests/test_radar_taxonomy_remediation.py tests/test_financial_epistemic_invariants.py -v
```
**Result**: **37 passed in 25.86s** (100% success rate).
- `tests/test_undervalued_growth_corrections.py` (6 passed):
  - `test_archetype_label_alone_does_not_certify_value_garp` PASSED
  - `test_delisted_security_is_excluded_from_active_discovery_and_non_actionable` PASSED
  - `test_missing_fundamental_metrics_are_none_not_zero` PASSED
  - `test_fundamental_evidence_provenance_is_exposed` PASSED
  - `test_radar_capability_contract_discloses_value_garp_limitations` PASSED
  - `test_lynch_filter_requires_quantitative_qualification` PASSED
- `tests/test_screener_execution.py` (10 passed)
- `tests/test_radar_taxonomy_remediation.py` (11 passed)
- `tests/test_financial_epistemic_invariants.py` (10 passed)

### 4.2 Frontend Architecture & Metric Suite
Executed:
```bash
npx tsx tests/radarTaxonomy.test.ts
npx tsx tests/radarMetricCleanup.test.ts
```
**Result**: **10 passed in 1.48s** (100% success rate).

---

## 5. Known Remaining Limitations & Next Steps

1. **Curated Reference Scope**:
   - The fundamental data in `KNOWN_GEMS_DATA` remains a static, curated reference catalog (covering ~16 assets). While missing assets now safely receive `None` and are excluded from unverified claims, full market-wide dynamic fundamental analysis requires an integrated fundamental financial statement provider pipeline.
2. **Dynamic Corporate Action Ingestion**:
   - CPRX was remediated via the authoritative `CONFIRMED_CORPORATE_ACTIONS` registry. A dynamic corporate actions provider (tracking SEC Form 25-NSE and 15-12B filings) is recommended for automated delisting propagation across the entire universe.
3. **Independent 150-Candidate Workstream**:
   - This bounded pilot operates strictly within the current coverage baseline and does not depend upon or modify the 150-candidate expansion workstream.

---
*Report certified by ARX Quant Semantics and Governance Engine.*
