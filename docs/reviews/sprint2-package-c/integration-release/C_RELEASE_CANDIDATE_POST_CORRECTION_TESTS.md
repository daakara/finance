# ARX TERMINAL — SPRINT 2 TRACK C
# RELEASE-CANDIDATE POST-CORRECTION TEST EXECUTION REPORT

## 1. Test Execution Summary

- **Timestamp of Execution:** 2026-10-21T10:08:41Z
- **Environment:** Windows PowerShell, Python 3.11.0, pytest 9.1.1
- **Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-package-c`
- **Branch:** `feat/sprint-2-package-c`
- **Total Tests Collected:** 62
- **Total Tests Passed:** 62 (100%)
- **Total Tests Failed:** 0
- **Total Tests Skipped:** 0

---

## 2. Test Suite Breakdown

| Test File | Test Count | Scope & Coverage | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_package_b_v3_canonical_consumer.py` | 14 | Canonical V3 manifest loading, predecessor fail-closed rejection, manifest vs certification boundary, synthetic fixture labeling, pre/post replacement, AAPL tail boundary, GOOG/GOOGL share-class separation, scanner enrollment protection, lookback thresholds, accounting closure. | **14 / 14 PASSED** |
| `tests/test_package_bc_disposable_integration_rehearsal.py` | 11 | End-to-end integration rehearsal with disposable in-memory SQLite repos: Package A identity -> B2 projection -> certified data -> publication dry-run & rollback, V3 canonical rehearsal. | **11 / 11 PASSED** |
| `tests/test_publication_state_machine_and_foundation.py` | 22 | Strict 7-state monotonic lifecycle, evidence validation, R01-R07 eligibility predicates, rollbacks, immutable build records, order-invariant SHA-256 digest. | **22 / 22 PASSED** |
| `tests/test_strategy_governance_adversarial.py` | 9 | Adversarial isolation between `G-VCP-012` (VCP holdout epoch 002) and `G-S2-ACT` (Sprint 2 operational dispatch); non-VCP strategy admission autonomy. | **9 / 9 PASSED** |
| `tests/test_lookback_threshold_boundaries.py` | 6 | Mathematical lookback threshold boundaries (50 builder floor, 60 momentum window, 260 annual lookback). | **6 / 6 PASSED** |
| **Total** | **62** | **Full Track C Consumer & Publication Verification** | **62 / 62 PASSED** |

---

## 3. Phase 8 Fourteen-Point Coverage Matrix

All 14 points mandated by the independent review have been implemented and verified in `tests/test_package_b_v3_canonical_consumer.py`:

| Point # | Requirement Description | Test Implementation | Result |
| :--- | :--- | :--- | :--- |
| **Point 1** | Canonical `0F54C702` digest accepted | `test_01_canonical_v3_manifest_digest_accepted` | **PASS** |
| **Point 2** | Superseded `65F5D9AB` digest rejected | `test_02_superseded_predecessor_workstream_a_digest_rejected` | **PASS** |
| **Point 3** | Superseded `8923DAD` digest rejected | `test_03_superseded_predecessor_workstream_b_digest_rejected` | **PASS** |
| **Point 4** | Tampered manifest (altered bytes/digest) rejected | `test_04_tampered_manifest_fails_closed` | **PASS** |
| **Point 5** | Canonical identity associations and bindings validated | `test_05_identity_association_validation` | **PASS** |
| **Point 6** | Manifest alone does NOT imply ingestion (`candle_count=0`) | `test_06_manifest_alone_does_not_imply_ingestion` | **PASS** |
| **Point 7** | Manifest alone does NOT imply data certification (fails State 2->3) | `test_07_manifest_alone_does_not_imply_certification` | **PASS** |
| **Point 8** | Synthetic test fixture requires `_fixture_nature='SYNTHETIC_TEST_FIXTURE'` | `test_08_synthetic_test_fixture_requires_explicit_label` | **PASS** |
| **Point 9** | Uncertified production-like data fails closed (`DATA_UNAVAILABLE`) | `test_09_uncertified_production_like_data_fails_closed` | **PASS** |
| **Point 10** | AAPL October 20 unprovenanced tail fails closed and cannot qualify | `test_10_aapl_october_20_unprovenanced_tail_fails_closed` | **PASS** |
| **Point 11** | GOOG/GOOGL share-class separation via identity; price equality permitted | `test_11_goog_googl_share_class_separation_and_pricing` | **PASS** |
| **Point 12** | Published-only candidates have zero scanner enrollment (`assert_no_scanner_enrollment`) | `test_12_published_only_zero_scanner_enrollment` | **PASS** |
| **Point 13** | Lookback thresholds remain 50/60/260 | `test_13_lookback_threshold_boundaries` | **PASS** |
| **Point 14** | Deterministic digest and exact arithmetic accounting closure | `test_14_deterministic_digest_and_accounting_closure` | **PASS** |

---

## 4. Execution Terminal Output

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\akara\Documents\Projects\finance-sprint-2-package-c
configfile: pyproject.toml
plugins: anyio-4.14.2, asyncio-1.4.0, cov-7.1.0, mock-3.15.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 62 items

tests\test_publication_state_machine_and_foundation.py ................. [ 27%]
.....                                                                    [ 35%]
tests\test_strategy_governance_adversarial.py .........                  [ 50%]
tests\test_lookback_threshold_boundaries.py ......                       [ 59%]
tests\test_package_bc_disposable_integration_rehearsal.py ...........    [ 77%]
tests\test_package_b_v3_canonical_consumer.py ..............             [100%]

============================= 62 passed in 1.27s ==============================
```
