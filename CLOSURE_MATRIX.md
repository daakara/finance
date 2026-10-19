# SOURCED FUNDAMENTALS PILOT: TARGETED SEMANTIC CORRECTIONS CLOSURE MATRIX

**Workstream Status**: `SOURCED_FUNDAMENTALS_TARGETED_SEMANTIC_CORRECTIONS_REVIEW_READY`  
**Date**: October 19, 2026  
**Repository**: `daakara/finance`  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-sourced-fundamentals-pilot`  
**Branch**: `feat/sourced-fundamentals-pilot`  
**Base Commit**: `3b953f834a49d88496dfa9dae56584e17e60bae4`  
**Target HEAD Commit**: `9ea6315f68cdf44ab8d8601fd53e36eab709cc14`  

---

## Executive Summary

All four independently confirmed defects in the sourced-fundamentals research pilot have been resolved and verified with dedicated negative and edge-case unit tests. All 30 tests in `tests/test_sourced_fundamentals_pilot.py` pass cleanly in offline mode with zero network egress.

---

## Defect Resolution & Test Verification Matrix

| Issue ID | Defect Description | Root Cause | Implemented Resolution | Verifying Test(s) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Issue 1** | Historical dilution varies by source observation order (DUOL original order: 2.10% 1Y, 6.74% 3Y vs reversed order: 28.71% 1Y, 34.62% 3Y). | `_extract_diluted_shares` and `compute_split_normalized_dilution` lacked duration basis filtering; in DUOL/FIX 10-Qs, 6-month YTD (~180d) observations share end dates with 3-month quarterly (~90d) observations. Reversing array caused 6M YTD observations to overwrite quarterly observations. | Enforced strict duration basis matching (`75 <= duration <= 105` days for quarterly, `345 <= duration <= 385` days for annual). Implemented deterministic deduplication and canonical sorting by end date, filing date, and accession number. Shuffling or reversing inputs yields 100% identical dilution. | `test_24_historical_dilution_order_invariance_and_basis_matching`<br>`test_2_permutation_invariance_source_observations`<br>`test_12_split_normalized_share_dilution_deck_and_celh`<br>`test_19_share_period_duration_discrimination_and_denominator_semantics`<br>`test_23_split_normalization_without_magnitude_guessing` | **VERIFIED PASS** |
| **Issue 2** | Quarterly weighted-average diluted shares divided into TTM Net Income under label `diluted_eps_ttm` and report label `Diluted Shares (TTM)`. | Under US GAAP ASC 260, diluted weighted-average shares cannot be aggregated or averaged across interim quarters because treasury stock method option/RSU dilution is dependent on quarterly average market prices. Relabeling quarterly shares as TTM shares violated ASC 260 denominator semantics. | Suppressed conventional `diluted_eps_ttm` (`None`) for interim reporting periods. Added explicit research estimate `ttm_net_income_per_latest_quarter_diluted_share` with status `RESEARCH_ESTIMATE`, explicit numerator period (`2025-07-01 to 2026-06-30`), and explicit denominator period (`2026-04-01 to 2026-06-30 (quarterly weighted-average shares)`). Relabeled report headers to `Diluted Shares (Latest Period)` and `TTM Net Inc / Diluted Share (Est)`. | `test_25_ttm_eps_interim_suppression_and_research_estimate_semantics`<br>`test_16_report_ledger_numeric_parity` | **VERIFIED PASS** |
| **Issue 3** | Assessment narratives contain unconditional praise phrases ("fortress balance sheet", "disciplined", "exceptional liquidity", "immense liquidity cushion", "robust") even when balance sheet facts are missing or adverse financials are supplied. | Hardcoded promotional narrative templates in `qualitative_assessor.py` asserted liquidity cushions and disciplined dilution unconditionally without checking whether balance sheet facts were present or whether growth/margins were negative. | Completely refactored `qualitative_assessor.py` to be strictly evidence-conditional: (1) Fails closed with `INSUFFICIENT_EVIDENCE` when balance sheet facts and operating results are unevidenced. (2) Truthfully reports revenue contractions, operating losses, cash burn, net debt, and share dilution. (3) Completely purged all 8 prohibited praise phrases from codebase. | `test_26_evidence_conditional_narratives_missing_balance_sheet_case_a`<br>`test_27_evidence_conditional_narratives_adverse_financials_case_b`<br>`test_28_evidence_conditional_narratives_acceptance_matrix_c_to_f`<br>`test_29_actual_eight_company_narrative_soundness`<br>`test_20_evidence_sufficiency_responsive_assessments` | **VERIFIED PASS** |
| **Issue 4** | Cumulative patch exported as UTF-16 LE with BOM instead of clean UTF-8. | PowerShell default redirection operator (`>`) exported diff stream as UTF-16 LE with BOM `\xff\xfe`, causing `git apply` failures on Unix/Linux tooling. | Exported cumulative diff directly via binary byte stream in Python from `git diff 3b953f8 HEAD`. Enforced byte-level validations: no `\xff\xfe` (UTF-16 LE), no `\xfe\xff` (UTF-16 BE), no `\xef\xbb\xbf` (UTF-8 BOM), and zero NUL bytes (`\x00`). Verified with isolated worktree `git apply --check`. | `test_30_valid_utf8_cumulative_patch_verification` | **VERIFIED PASS** |

---

## Test Execution Summary

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-8.4.2, pluggy-1.6.0
rootdir: C:\Users\akara\Documents\Projects\finance-sourced-fundamentals-pilot
collected 30 items

tests/test_sourced_fundamentals_pilot.py::test_1_ttm_duration_selection_counterexample_fix PASSED [  3%]
tests/test_sourced_fundamentals_pilot.py::test_2_permutation_invariance_source_observations PASSED [  6%]
tests/test_sourced_fundamentals_pilot.py::test_3_annual_quarterly_fiscal_year_end_discrimination PASSED [ 10%]
tests/test_sourced_fundamentals_pilot.py::test_4_comparable_prior_year_ytd_selection PASSED [ 13%]
tests/test_sourced_fundamentals_pilot.py::test_5_duplicate_comparative_observations_deduplication PASSED [ 16%]
tests/test_sourced_fundamentals_pilot.py::test_6_non_calendar_retail_fiscal_year PASSED [ 20%]
tests/test_sourced_fundamentals_pilot.py::test_7_missing_or_ambiguous_ttm_components PASSED [ 23%]
tests/test_sourced_fundamentals_pilot.py::test_8_source_cutoff_and_amendment_handling PASSED [ 26%]
tests/test_sourced_fundamentals_pilot.py::test_9_missing_debt_not_becoming_reported_zero PASSED [ 30%]
tests/test_sourced_fundamentals_pilot.py::test_10_reported_zero_remaining_zero PASSED [ 33%]
tests/test_sourced_fundamentals_pilot.py::test_11_missing_valuation_inputs_do_not_produce_populated_valuations PASSED [ 36%]
tests/test_sourced_fundamentals_pilot.py::test_12_split_normalized_share_dilution_deck_and_celh PASSED [ 40%]
tests/test_sourced_fundamentals_pilot.py::test_13_non_positive_invested_capital_handling PASSED [ 43%]
tests/test_sourced_fundamentals_pilot.py::test_14_forward_and_reverse_dcf_reconciliation PASSED [ 46%]
tests/test_sourced_fundamentals_pilot.py::test_15_unbracketed_reverse_dcf_targets PASSED [ 50%]
tests/test_sourced_fundamentals_pilot.py::test_16_report_ledger_numeric_parity PASSED [ 53%]
tests/test_sourced_fundamentals_pilot.py::test_17_offline_replay_zero_network_access PASSED [ 56%]
tests/test_sourced_fundamentals_pilot.py::test_18_aggregate_missing_data_propagation_and_completeness PASSED [ 60%]
tests/test_sourced_fundamentals_pilot.py::test_19_share_period_duration_discrimination_and_denominator_semantics PASSED [ 63%]
tests/test_sourced_fundamentals_pilot.py::test_20_evidence_sufficiency_responsive_assessments PASSED [ 66%]
tests/test_sourced_fundamentals_pilot.py::test_21_conflicting_observations_and_restatement_precedence PASSED [ 70%]
tests/test_sourced_fundamentals_pilot.py::test_22_declared_research_cutoff_enforced_end_to_end PASSED [ 73%]
tests/test_sourced_fundamentals_pilot.py::test_23_split_normalization_without_magnitude_guessing PASSED [ 76%]
tests/test_sourced_fundamentals_pilot.py::test_24_historical_dilution_order_invariance_and_basis_matching PASSED [ 80%]
tests/test_sourced_fundamentals_pilot.py::test_25_ttm_eps_interim_suppression_and_research_estimate_semantics PASSED [ 83%]
tests/test_sourced_fundamentals_pilot.py::test_26_evidence_conditional_narratives_missing_balance_sheet_case_a PASSED [ 86%]
tests/test_sourced_fundamentals_pilot.py::test_27_evidence_conditional_narratives_adverse_financials_case_b PASSED [ 90%]
tests/test_sourced_fundamentals_pilot.py::test_28_evidence_conditional_narratives_acceptance_matrix_c_to_f PASSED [ 93%]
tests/test_sourced_fundamentals_pilot.py::test_29_actual_eight_company_narrative_soundness PASSED [ 96%]
tests/test_sourced_fundamentals_pilot.py::test_30_valid_utf8_cumulative_patch_verification PASSED [100%]

============================= 30 passed in 8.21s ==============================
```
