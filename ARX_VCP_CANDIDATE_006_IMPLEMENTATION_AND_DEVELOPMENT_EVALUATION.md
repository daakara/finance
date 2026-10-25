# ARX Terminal — Candidate 006 Implementation & Development Evaluation

**Document Identifier**: ARX_VCP_CANDIDATE_006_IMPLEMENTATION_AND_DEVELOPMENT_EVALUATION
**Workstream Identifier**: ARX_VCP_RESEARCH_WAVE_FRAGMENTATION_01 (Nonproduction Research Experiment)
**Execution Timestamp**: 2026-10-24T01:35:00+02:00
**Reporting Reconciliation Date**: 2026-10-24T01:45:00+02:00
**Governing Framework**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
**Final Approval Authority**: David Akara
**Final Disposition**: ARX_VCP_CANDIDATE_006_DEVELOPMENT_EXPERIMENT_COMPLETE

> **Reporting Reconciliation Note (2026-10-24T01:45:00+02:00)**:
> This report has been reconciled under authorization `ARX_VCP_CANDIDATE_006_LOCAL_RESEARCH_BASELINE_COMMIT_AUTHORIZED` to:
> 1. Formulate control classification findings precisely separating true negative classifications from insufficient-data abstentions across the four control subsets.
> 2. Accurately describe the LLY qualification tradeoff relative to the approved Candidate 005 final-trough contract rather than informal phrasing.
> 3. Document the full harness execution history including the two initial invocation exceptions prior to the successful full run.
> 4. Enforce strict claim limits: No implementation defect found within the verified scope; observed development recognition does not constitute independently validated sensitivity or predictive performance.
>
> **Packaging & Lineage Addendum (2026-10-24T08:45:00+02:00)**:
> Reconciled under authorization `ARX_VCP_CANDIDATE_006_RESEARCH_TRANSPLANT_CI_PACKAGING_AND_LOCAL_COMMITS_AUTHORIZED`:
> - Reconciled FTNT cutoffs (`2026-10-02`, `2026-10-05`) and DHLGY cutoff (`2026-09-01`) to match authoritative manifest records.
> - Distinguished POS-04 final-wave volume ratio (`0.6448`) from POS-03 (`0.6273`).
> - Formally qualified control abstentions: 6 missing input data (< 200 bars), 1 SPY unavailable base-anchor geometry & volume.
> - Formally qualified harness execution: zero completed case records; first-case assessment had already been computed before serialization failed.
> - Updated operational risk phrasing: Candidate 004 remains default, while shared-code regression risk remains subject to verification.

---

## 1. Decision & Executive Summary

Under authorization ARX_VCP_CANDIDATE_006_NONPRODUCTION_IMPLEMENTATION_AND_DEVELOPMENT_EVALUATION_AUTHORIZED, Candidate 006 was implemented in an isolated local worktree (eat/vcp-candidate-006-merging) branched from Candidate 005 baseline commit 13d4e0862561ac0ebb486d0ca59bbd5defb14075.

### Material Results Summary

`
+---------------------------------------------------------------------------------------------------+
| CANDIDATE 006 DEVELOPMENT EVALUATION SCORECARD                                                   |
+-----------------------------------+--------------------+--------------------+---------------------+
| Metric Dimension                  | Candidate 004      | Candidate 005      | Candidate 006       |
+-----------------------------------+--------------------+--------------------+---------------------+
| Synthetic Conformance             | 19 / 19 passed     | 11 / 11 passed     | 15 / 15 passed      |
| Dual / Tri Reproducibility        | Verified           | Verified           | 6 / 6 passed        |
| Positive Cases Qualified          | 0 / 15 (0.0%)      | 1 / 15 (6.7%)      | 3 / 15 (20.0%)      |
| Positive Cases Tickers            | 0 / 9 issuers      | 1 / 9 issuers      | 2 / 9 issuers       |
| Control False Positives           | 0 / 45 (0.0%)      | 0 / 45 (0.0%)      | 0 / 45 (0.0%)       |
| Control Negative Classifications  | 38 / 45 (84.4%)    | 38 / 45 (84.4%)    | 38 / 45 (84.4%)     |
| Control Data Abstentions          | 7 / 45 (15.6%)     | 7 / 45 (15.6%)     | 7 / 45 (15.6%)      |
| Wave Fragmentation Resolution     | Baseline           | Fractured (10-22w) | Resolved (2-8w)     |
+-----------------------------------+--------------------+--------------------+---------------------+
`

### Key Substantive Findings

1. **Wave Fragmentation Successfully Resolved**:
   - FTNT (POS-03, POS-04): Consolidated from 11 micro-waves under Candidate 005 into **4 macro waves** (.60\% \to 13.85\% \to 12.70\% \to 6.51\%$). Progressive tightening, volume dry-up (.6273 \le 0.70$), and pivot zone confirmed. **Both cases QUALIFIED**.
   - DHLGY (POS-10): Consolidated from 5 waves under Candidate 005 (which failed tightening) into **3 clean macro waves** (.06\% \to 6.73\% \to 5.95\%$). Volume dry-up (.4670 \le 0.70$) confirmed. **QUALIFIED**.
   - NVDA (POS-01, POS-02): Consolidated from 10 micro-waves under Candidate 005 into **4 macro waves** (.68\% \to 15.20\% \to 9.41\% \to 8.27\%$). Because Wave 1 to Wave 2 is only a .06\%$ relative reduction, it correctly fails strict \%$ tightening (NON_TIGHTENING_SEQUENCE) while independently failing volume (.9163 > 0.70$).
   - VRT (POS-07, POS-08): Consolidated from 22 micro-waves under Candidate 005 down to **8 waves**, cleanly failing count bounds ( > 6$) and detecting true wide volatility expansion (VOLATILITY_EXPANSION_DETECTED).

2. **Control Specificity & Accounting**:
   - Zero positive outputs across 45 control inputs, comprising 38 negative classifications and seven insufficient-data abstentions.
   - Evaluated control subsets reported separately:
     * Stage 3/4: 10 inputs, nine negatives, one abstention.
     * Non-VCP / failed base: 20 inputs, 19 negatives, one abstention.
     * Developing VCP: 10 negatives.
     * Insufficient evidence: five abstentions.
   - Negative classifications (38) and data abstentions (7) match Candidate 004 and Candidate 005 bit-exact. Abstentions do not establish specificity, but reflect data-availability constraints faithfully preserved from the baseline.

3. **Causal Integrity & LLY Tradeoff**:
   - Manifest Case: HVA00-POS-11 / LLY / 2026-01-07T21:00:00Z.
   - Candidate 005 accepted HVA00-POS-11 / LLY under its approved final-trough confirmation rule, which recognized Wave 2 based on a 1-day recovery meeting the threshold.
   - Candidate 006 applies a different experimental causal confirmation contract requiring both $\ge 5\%$ recovery from trough AND at least 3 elapsed eligible trading sessions ( - t_{\\text{trough}} \ge 3$) to prevent premature wave closure during micro-consolidation.
   - At the evaluation cutoff date of 2026-01-07T21:00:00Z, Wave 2 had elapsed only 1 trading session post-trough ( - t_{\\text{trough}} = 1 < 3$). It remained unconfirmed at cutoff and was classified as FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF (with only 1 confirmed wave, triggering INSUFFICIENT_WAVES).
   - LLY\'s loss of qualification is an explicit research tradeoff of the causal 3-session confirmation contract.
   - The new rule\'s conservatism does not independently prove that Candidate 005\'s approved qualification was incorrect; rather, Candidate 006 enforces a more stringent causal confirmation threshold before declaring a contraction complete.

4. **Scope Claim Limits**:
   - **No implementation defect found within the verified scope.**
   - Technical conformance (15/15 synthetic tests passing), development safety (0/45 false positives), and recognition findings (3/15 qualified) are distinct evaluation dimensions.
   - Qualification of 3/15 positive-label development cases across two distinct issuers demonstrates successful macro wave consolidation on exposed development data, but does **not** constitute independently validated sensitivity or predictive performance on unseen market regimes.

---

## 2. Corrected Specification Identity

- **Specification Document**: [ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION.md](file:///C:/Users/akara/.gemini/antigravity/brain/48da04d0-f5bf-4544-8766-b2bf86390360/ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION.md)
- **Revision**: 2.3.0 (Contract Freeze with Binding Corrections)
- **SHA-256**: e389f6d0af8d5612b50606a954cec31905609e90e4d9c3f265ae3320137bbbd
- **Preserved Rev 2.2.0 Spec**: ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION_REV_2_2_0.md (SHA-256: d129436b0d9184da6743270b0d7b04dbd1af290c59628b6beaaae7a6be6c85b8)
- **Binding Contract Corrections Implemented**:
  1. *Strict Tightening Preserved*: {\\text{next}} \le 0.95 \times D_{\\text{previous}}$ via erify_progressive_tightening(depths, relative_reduction=0.05, eps_float=1e-7). Proposed $+0.01$ expansion allowance strictly rejected.
  2. *Initial Depth Floor Preserved*:  \ge 0.0800$ enforced. General 2% minimum does not replace  \ge 0.08$.
  3. *Faithful Expansion Extraction*: Segmentation records expanding sequences without dropping waves; predicate layer evaluates and fails under tightening.
  4. *Elapsed Time Accurate*: $\ge 3$ elapsed eligible sessions post-trough ( - t_{\\text{trough}} \ge 3$). No intermediate 2% support floor.
  5. *Numerical Guard*: ^{-7}$ ratio guard implemented as an explicit narrow acceptance band.
  6. *Same-Bar Non-Ordering Policy*: Bar establishing or updating candidate peak cannot also confirm downswing rollover; rollover evaluated strictly on subsequent bars ( > t_{\\text{peak}}$).

---

## 3. Implementation & Test Manifest

### Worktree Baseline
- **Location**: C:/Users/akara/Documents/Projects/finance-vcp-candidate-006-merging
- **Branch**: eat/vcp-candidate-006-merging
- **Base Commit**: 13d4e0862561ac0ebb486d0ca59bbd5defb14075 (Tree: 40cb84d2952eedef7715defa62559bb20a5d94de)

### File Inventory & Checksums

| File Path | Role | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| nalyst_dashboard/vcp/classifier.py | C006 causal segmentation engine | 40,581 | 98d2e11b3b39ffe3f65f972e3abd20ed4d194befe43b5c7b1a3d2ecac2379026 |
| nalyst_dashboard/vcp/numeric_contract.py | Precision & progressive tightening contract | 7,722 | 74aaa38e268b7fa8736eb1ddb80e702f3658478eaf2d98003ef3b1a1ad0b6d55 |
| nalyst_dashboard/vcp/predicate_registry.py | Predicate registry & domain assessment | 38,428 | 22756a2f2dbee5919e88dc0f6dea5ea51c555950392ac55d4c6ac52fbabb8e2e |
| nalyst_dashboard/analyzers/scanner_contract.py | C006 algorithm & ruleset version constants | 9,263 | 6b35f1feebcffa87e8ce5a742a5d151f7107de573f5b203406f22c4f69bf911c |
| nalyst_dashboard/analyzers/scanner_publication_integrity.py | Ruleset specification dictionary & hash | 12,698 | 131c1eff7f514028790e2aefa01acccf397b36889e5c46c43aa921fae259d58e |
| 	ests/test_vcp_candidate_006_conformance.py | 15 synthetic conformance tests | 26,051 | 736229f062aac754af77d66c9a6861264efec509668fd5ddd2b4be3a59d3c76 |
| 	ests/test_vcp_candidate_005_conformance.py | 11 preserved C005 conformance tests | 20,148 | 687c8a80e0ffada4bb0bb9ef02c81c39e0ee5e7ee89bf7619d84b9058847c3fb |
| 	ests/test_vcp_candidate_reproducibility.py | 6 reproducibility & isolation tests | 14,567 | 74b1d5672dacfb7eff4a50134a4cc138bc7ea85acda17b8c78d0d2a3ad20252a |
| scratch/evaluate_candidate_006_hva00.py | Reproducible evaluation harness | 9,611 | 3daa19a8db15bc21b10261d788d8dbc9809fafb1998cfbedc18276e1a85203a8 |

---

## 4. Synthetic Traceability Matrix

All 15 synthetic test scenarios in 	ests/test_vcp_candidate_006_conformance.py passed bit-exact:

| Test ID | Scenario Description | Core Assertion Verified | Result |
| :---: | :--- | :--- | :---: |
| SYN-01 | Intervening low lower than endpoints | Captured trough at .00$ ({183}$) rather than endpoints .00$ or .50$. | **PASS** |
| SYN-02 | New peak & same-bar non-ordering | Peak updated to .00$ at {180}$; rollover prevented on same bar. | **PASS** |
| SYN-03 | Equal-high and equal-low ties | Retained earliest chronological bar index for both peak and trough. | **PASS** |
| SYN-04 | Outside bar precedence | High takes precedence; rollover deferred to subsequent session. | **PASS** |
| SYN-05 | 3 elapsed sessions without persistence | Confirmed wave at {178}$ when session 3 close exceeded 5%, despite intermediate closes < 5%. | **PASS** |
| SYN-06 | Lower low resets trough | Undercut before confirmation reset running trough and elapsed-session counter. | **PASS** |
| SYN-07 | Undercut after confirmation | Wave 1 preserved immutable; Wave 2 recorded at depth 21.05%; rejected by tightening rule. | **PASS** |
| SYN-08 | Strict tightening boundary | .20 \to 0.1900$ PASS, .20 \to 0.1901$ FAIL, $+0.005$ expansion FAIL. | **PASS** |
| SYN-09 | Initial depth floor preservation |  = 0.06 < 0.08$ rejected with INSUFFICIENT_BASE_ANCHOR_CANDIDATES. | **PASS** |
| SYN-10 | Numerical guards (^{-7}$) | Guard band boundaries evaluated accurately for both recovery and rollover. | **PASS** |
| SYN-11 | Peak bar low exclusion | Peak bar low (.00$) excluded from trough search; trough initialized post-peak. | **PASS** |
| SYN-12 | Cutoff during unconfirmed downswing | Unfinished downswing failed with FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF. | **PASS** |
| SYN-13 | Fixed-start prefix consistency | Confirmed Wave 1 records identical between prefix $ and prefix +10$. | **PASS** |
| SYN-14 | Price-scale invariance | 10x scaled price series produced identical wave counts, indices, and percentage depths. | **PASS** |
| SYN-15 | Invalid inputs & history | Handled $< 40$ bars and nonpositive prices safely without exception. | **PASS** |

---

## 5. Candidate 004 / 005 Preservation Evidence

Execution of 	ests/test_vcp_candidate_reproducibility.py confirmed:
1. 	est_01: Default instantiation of VCPClassifier() selects Candidate 004 (ARX_VCP_CLASSIFIER_V1_CANDIDATE_004).
2. 	est_02: Default erify_progressive_tightening enforces monotonic tightening ({\\text{curr}} < D_{\\text{prev}}$) for Candidate 004.
3. 	est_03: Candidate 004 outputs match frozen cp_hva00_frozen_algorithm_results.json 100% bit-exact across development cases.
4. 	est_04: Dual execution of Candidate 004 and Candidate 005 exhibits zero cross-contamination.
5. 	est_05: Candidate 005 outputs match frozen cp_hva02_candidate_005_results.json 100% bit-exact across development cases.
6. 	est_06: Candidate execution order invariance confirmed (C004 $\to$ C005 $\to$ C006 vs C006 $\to$ C005 $\to$ C004 produce bit-exact identical observations and assessments).

---

## 6. Pre-Evaluation Freeze & Execution Trace

### Pre-Evaluation Corpus & Database Freeze
- **Corpus Manifest**: data/research/vcp_hva00_case_manifest.json (SHA-256: 407fb5f7dab217501cda0b7c936f6ef894b9b4ce773153da35b142e1278b9bb7)
- **Market Database**: data/research/market_data/bounded_150_dev_market_store.db (SHA-256: b31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615)
- **C004 Baseline Reference**: data/research/vcp_hva00_frozen_algorithm_results.json (SHA-256: efe13e8a4672422b0e58e179c97cbdcd7b4c33937d036767eb5a011c12cc51b4)
- **C005 Baseline Reference**: data/research/vcp_hva02_candidate_005_results.json (SHA-256: e34863b566df0087822eff71309381c50d6f538636f52120a02b3018935d62fc)

### Evaluation Harness Invocations & Trace Accounting

| Invoc. ID | Timestamp | Harness Target | Cases Processed | Results Saved | Failure Location & Cause | Action Taken |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Attempt 1** | 2026-10-24T01:30:51+02:00 | evaluate_candidate_006_hva00.py | 0 / 60 | None | Line 125: AttributeError: 'VCPDomainAssessment' object has no attribute 'decision_reason_codes' | Fixed attribute name to
eason_codes. |
| **Attempt 2** | 2026-10-24T01:31:34+02:00 | evaluate_candidate_006_hva00.py | 0 / 60 | None | Line 125: AttributeError: 'VCPDomainAssessment' object has no attribute 'reason_codes' | Fixed attribute name to classification_reason_codes per predicate_registry.py. |
| **Attempt 3** | 2026-10-24T01:31:59+02:00 | evaluate_candidate_006_hva00.py | 60 / 60 | Complete | None (Execution finished cleanly in 2.87s) | Results written to cp_hva02_candidate_006_results.json. |

**Forensic Evaluation Notes**:
1. *Zero Case Leakage / Pre-Extraction Computation*: In Attempts 1 and 2, zero completed case records were created. However, the classifier assessment on Case 1 (`HVA00-POS-01 / NVDA`) had already been computed by `classify_case()`, `extract_observations()`, and `evaluate_predicates()` before execution crashed on line 125 during output property extraction. Zero cases were buffered, displayed, or written to disk.
2. *Strict Code Isolation*: Only harness extraction boilerplate changed between attempts. The core Candidate 006 algorithm in classifier.py, the ruleset constants in scanner_contract.py, and all test fixtures remained 100% frozen.
3. *Single Completed Run*: Attempt 3 represents the sole completed evaluation run across the full 60-case corpus.

---

## 7. Development Evaluation Outcomes (All 60 Cases)

### Positive-Label Cases (Stratum 1: 15 cases, 9 distinct issuers)

| Case ID | Symbol | Cutoff Date | C004 | C005 | C006 Status | C006 Waves | Contraction Depths (%) | C006 Primary Rejection Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| HVA00-POS-01 | NVDA | 2026-01-29 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 4 | [15.68, 15.20, 9.41, 8.27] | NON_TIGHTENING_SEQUENCE, VOLUME_HEAVY_ON_PULLBACK |
| HVA00-POS-02 | NVDA | 2026-01-30 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 4 | [15.68, 15.20, 9.41, 8.27] | FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF, VOLUME_HEAVY |
| HVA00-POS-03 | FTNT | 2026-10-02 | Non-Qual | Non-Qual | **QUALIFIED** | 4 | [15.60, 13.85, 12.70, 6.51] | *None (All 13 Predicates Passed)* |
| HVA00-POS-04 | FTNT | 2026-10-05 | Non-Qual | Non-Qual | **QUALIFIED** | 4 | [15.60, 13.85, 12.70, 6.51] | *None (All 13 Predicates Passed)* |
| HVA00-POS-05 | PLTR | 2026-01-14 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 2 | [12.03, 11.64] | NON_TIGHTENING_SEQUENCE, PRICE_EXTENDED_PAST_PIVOT |
| HVA00-POS-06 | PLTR | 2026-01-15 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 2 | [12.03, 11.64] | NON_TIGHTENING_SEQUENCE, VOLUME_HEAVY_ON_PULLBACK |
| HVA00-POS-07 | VRT | 2026-01-22 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 8 | [26.26, 7.21, 22.01, 4.10, ...] | CONTRACTION_COUNT_OUT_OF_BOUNDS, VOLATILITY_EXPANSION |
| HVA00-POS-08 | VRT | 2026-01-23 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 8 | [26.26, 7.21, 22.01, 4.10, ...] | CONTRACTION_COUNT_OUT_OF_BOUNDS, VOLATILITY_EXPANSION |
| HVA00-POS-09 | DHLGY| 2026-01-08 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 2 | [9.15, 6.73] | FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF, VOLUME_HEAVY |
| HVA00-POS-10 | DHLGY| 2026-09-01 | Non-Qual | Non-Qual | **QUALIFIED** | 3 | [10.06, 6.73, 5.95] | *None (All 13 Predicates Passed)* |
| HVA00-POS-11 | LLY | 2026-01-07 | Non-Qual | Qual | **NON-QUALIFIED** | 1 | [11.97] | FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF, INSUFFICIENT_WAVES|
| HVA00-POS-12 | LLY | 2026-01-08 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 1 | [11.97] | FINAL_CONTRACTION_UNFINISHED_AT_CUTOFF, INSUFFICIENT_WAVES|
| HVA00-POS-13 | RTX | 2026-01-15 | Non-Qual | Non-Qual | **ABSTENTION** | 0 | [] | INSUFFICIENT_BASE_ANCHOR_CANDIDATES, VOLUME_DATA_UNAVAILABLE|
| HVA00-POS-14 | MEDP | 2026-01-16 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 1 | [16.88] | 150_SMA_BELOW_200_SMA (Stage 2 Failed), VOLUME_HEAVY |
| HVA00-POS-15 | ROST | 2026-01-22 | Non-Qual | Non-Qual | **NON-QUALIFIED** | 2 | [6.49, 15.19] | VOLATILITY_EXPANSION_DETECTED (6.49% -> 15.19%), VOLUME_HEAVY|

---

### Control Strata Breakdown (45 control inputs)

Zero positive outputs across 45 control inputs, comprising 38 negative classifications and seven insufficient-data abstentions:

| Control Subset | Input Count | C004 Qual | C005 Qual | C006 Qual | C006 Non-Qual (True Neg) | C006 Abstentions | Safety / Specificity Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Developing VCP (Stratum 2)** | 10 | 0 | 0 | **0** | 10 | 0 | 10 negative classifications (100% negative) |
| **Non-VCP / Failed Base (Strata 3 & 4)** | 20 | 0 | 0 | **0** | 19 | 1 | 19 negative classifications, 1 abstention |
| **Stage 3 & 4 Rejection (Strata 5 & 6A)** | 10 | 0 | 0 | **0** | 9 | 1 | 9 negative classifications, 1 abstention |
| **Insufficient Evidence (Stratum 6B)** | 5 | 0 | 0 | **0** | 0 | 5 | 5 data abstentions (preserved from baseline) |
| **Total Control Inputs** | **45** | **0** | **0** | **0** | **38** | **7** | **Zero False Positives Across All 45 Inputs** |

*Note on Control Accounting*: The 7 abstentions are formally partitioned into two distinct failure mechanisms:
1. *Missing Raw Input Data (6 cases)*: `HVA00-INS-01` through `05` and `HVA00-REJ-04` (< 200 trading sessions available, triggering `INSUFFICIENT_HISTORY_SESSIONS` and `VOLUME_DATA_UNAVAILABLE`).
2. *Unavailable Derived Geometry / Base Anchor (1 case)*: `HVA00-S2N-09 / SPY` has 284 trading sessions and passes Stage 2 and prior uptrend, but abstains due to `INSUFFICIENT_BASE_ANCHOR_CANDIDATES` (no valid 15-to-90 day consolidation base anchor exists in a smooth monotonic uptrend) combined with ETF volume omissions (`VOLUME_DATA_UNAVAILABLE`).
Abstentions do not establish classifier specificity, but confirm that Candidate 006 preserves data integrity gates without converting data-starved or unanchored inputs into false signals.

---

## 8. Preserved Reproduction Evidence

Machine-readable results and reproduction artifacts are preserved in the persistent evidence directory:
- **Evidence Directory**: C:/Users/akara/.gemini/antigravity/brain/48da04d0-f5bf-4544-8766-b2bf86390360/evidence/vcp_candidate_006/
- **Results File**: cp_hva02_candidate_006_results.json (374,810 bytes, SHA-256: 9945108c2e0647654384be79265e6df07787f2ecb8185fdb5b49cda4e360a54a)
- **Evaluation Harness**: evaluate_candidate_006_hva00.py (9,611 bytes, SHA-256: 3daa19a8db15bc21b10261d788d8dbc9809fafb1998cfbedc18276e1a85203a8)

---

## 9. Remaining Limitations & Recommended Next Decision

### Remaining Limitations
1. **Volume Pullback Sensitivity**: Several positive setups (NVDA, PLTR) exhibited clean macro contraction geometry ($ and $ waves) but were rejected by heavy pullback volume ($> 0.70$). This reflects true domain conservatism rather than wave-segmentation failure.
2. **Cutoff Proximity on Tight Handles**: On LLY, Wave 2 formed immediately prior to the cutoff date and had only elapsed 1 session post-trough. The causal 3-session requirement properly avoided false premature qualification, but naturally requires subsequent trading sessions to confirm.

### Recommended Next Owner Decision
**Recommend Candidate 006 for Research Preservation and Bounded Holdout Pre-Commitment Review**:
- Candidate 006 demonstrates superior contraction segmentation over Candidates 004 and 005, successfully solving wave fragmentation on FTNT and DHLGY without producing a single false positive across 45 controls.
- Production activation remains on **HOLD** (PR #73 held). Candidate 004 remains default, while shared-code regression risk remains subject to verification. A future main merge may trigger normal deployment automation and requires separate release authorization.
- Recommend David Akara authorize creating a scoped local research commit in worktree inance-vcp-candidate-006-merging to preserve this experimental milestone.
