# ARX Terminal — Phase 2 Evidence Custody Regression Evidence (Gate K)

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Gate:** Gate K — Regression and Package Preservation  
**Execution Timestamp:** `2026-10-23T16:42:00+02:00`  
**Repository:** `daakara/finance`  
**Environment:** Python 3.11.0, pytest-8.4.2, pluggy-1.6.0 on Windows 11 (win32)  
**Total Tests Executed:** `51 passed in 5.39s`  
**Verdict:** `ARX_UX_CUSTODY_REGRESSION_VERIFIED`  

---

## 1. Regression Suite Execution Summary

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-8.4.2, pluggy-1.6.0 -- C:\Program Files\Python311\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\akara\Documents\Projects\finance
configfile: pyproject.toml
plugins: anyio-4.14.2, asyncio-0.25.3, cov-7.1.0
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None
collecting ... collected 51 items

tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_01_missing_vault_configuration PASSED [  1%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_02_vault_path_inside_git_repository PASSED [  3%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_03_unsafe_vault_permissions PASSED [  5%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_04_unauthorized_observer_token PASSED [  7%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_05_malformed_json PASSED [  9%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_06_invalid_schema PASSED [ 11%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_07_duplicate_submission PASSED [ 13%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_08_submission_replay PASSED [ 15%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_09_file_replacement_attempt PASSED [ 17%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_10_modified_ciphertext PASSED [ 19%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_11_incorrect_decryption_key PASSED [ 21%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_12_missing_package_identity PASSED [ 23%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_13_incorrect_package_hash PASSED [ 25%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_14_gold_key_access_attempt PASSED [ 27%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_15_observer_cross_access_attempt PASSED [ 29%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_16_premature_scoring_with_one_submission PASSED [ 31%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_17_two_independently_sealed_valid_synthetic_submissions PASSED [ 33%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_18_withdrawal_invalidation PASSED [ 35%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_19_deletion_workflow PASSED [ 37%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_20_no_sensitive_logs PASSED [ 39%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_21_git_forced_add_containment PASSED [ 41%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_22_retained_canonical_qualification_fixtures PASSED [ 43%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_23_incomplete_backup_deletion_documentation PASSED [ 45%]
tests/test_evidence_custody_security.py::TestEvidenceCustodySecurity::test_24_crash_interruption_during_atomic_intake PASSED [ 47%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_01_frozen_canonical_cm_definitions PASSED [ 49%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_02_ratified_supplemental_sm_definitions PASSED [ 50%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_03_no_drifted_taxonomy_terms PASSED [ 52%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_04_ratified_concept_identities PASSED [ 54%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_05_trial_ids_align_with_gold_key PASSED [ 56%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_06_synthetic_provenance_explicitly_disclosed PASSED [ 58%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_07_zip_bundle_has_no_answers_or_rationales PASSED [ 60%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_08_manifest_and_zip_hash_match PASSED [ 62%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_09_production_containment_preserved PASSED [ 64%]
tests/test_observer_distribution_integrity.py::TestObserverDistributionIntegrity::test_10_no_fabricated_human_results PASSED [ 66%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_01_perfect_agreement PASSED [ 68%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_02_systematic_disagreement PASSED [ 70%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_03_rare_positive_categories PASSED [ 72%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_04_all_negative_category PASSED [ 74%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_05_undefined_kappa_handling PASSED [ 76%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_06_multilabel_overlap PASSED [ 78%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_07_missing_responses PASSED [ 80%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_08_duplicate_trial_ids PASSED [ 82%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_09_missing_gold_key_ids PASSED [ 84%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_10_incorrect_taxonomy_codes PASSED [ 86%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_11_false_positives PASSED [ 88%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_12_false_negatives PASSED [ 90%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_13_prevalence_imbalance PASSED [ 92%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_14_observer_vs_gold_measurement_a PASSED [ 94%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_15_observer_pairwise_measurement_b PASSED [ 96%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_16_supplemental_gamma_corpus_execution PASSED [ 98%]
tests/test_scoring_utility_regression.py::TestScoringUtilityRegression::test_17_consolidated_30_trial_complete_coverage PASSED [100%]

============================= 51 passed in 5.39s ==============================
```

---

## 2. Canonical Package Hash Verification

```powershell
python -c "
import hashlib
from pathlib import Path
p = Path('tests/fixtures/qualification/ARX_OBSERVER_QUALIFICATION_BUNDLE.zip')
digest = hashlib.sha256(p.read_bytes()).hexdigest()
size = p.stat().st_size
print(f'Archive Size: {size} bytes')
print(f'Archive SHA-256: {digest}')
assert digest == '9a4a53e642d7a8ea55a1dbceab9024408d3d58913f8a019dac52609044a8d6c5'
assert size == 17810
print('PASS: Archive is 100% byte-identical and untampered.')
"
```

### Output:
```text
Archive Size: 17810 bytes
Archive SHA-256: 9a4a53e642d7a8ea55a1dbceab9024408d3d58913f8a019dac52609044a8d6c5
PASS: Archive is 100% byte-identical and untampered.
```
**Exit Code:** `0`

---

## 3. Git Repository Containment Verification

```powershell
python scripts/verify_evidence_custody_containment.py
```

### Output:
```text
==============================================================================
ARX EVIDENCE CUSTODY REPOSITORY CONTAINMENT VERIFIER (GATE A)
==============================================================================
[PASS] Gate A.1: .gitignore contains mandatory custody exclusion rules.
[PASS] Gate A.2: No forbidden custody files or identities tracked in Git.
[PASS] Gate A.3: No forbidden custody files staged in Git index.
[PASS] Gate A.4: Zero historical commits matching custody-sensitive patterns.
[PASS] Gate A.5: All required canonical fixtures and policies present.
[PASS] Gate A.6: Git check-ignore correctly blocks all test custody paths.
------------------------------------------------------------------------------
VERDICT: ARX_UX_GIT_EXPOSURE_CONTAINMENT_VERIFIED
==============================================================================
```
**Exit Code:** `0`
