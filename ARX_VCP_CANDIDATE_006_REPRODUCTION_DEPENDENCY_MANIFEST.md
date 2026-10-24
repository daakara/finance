# ARX Terminal — Candidate 006 Reproduction Dependency Manifest

**Document Identifier**: `ARX_VCP_CANDIDATE_006_REPRODUCTION_DEPENDENCY_MANIFEST`
**Execution Timestamp**: `2026-10-24T01:45:00+02:00`
**Packaging Addendum Timestamp**: `2026-10-24T08:45:00+02:00`
**Governing Framework**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
**Final Approval Authority**: David Akara
**Workstream**: `ARX_VCP_RESEARCH_WAVE_FRAGMENTATION_01`
**Target Branch**: `feat/vcp-candidate-006-research`
**Target Worktree**: `C:\Users\akara\Documents\Projects\finance-vcp-candidate-006-research`
**Base Commit**: `8404b6e4645b5f9f8237e70517dcab3b6afdda7f` (`origin/main`)
**Status**: `REPRODUCTION_DEPENDENCIES_COMMITTED_AND_PACKAGED`

---

## 1. Executive Summary

This manifest catalogues every asset, artifact, test oracle, and data dependency required to reproduce the synthetic conformance tests, cross-candidate preservation tests, and historical development evaluations for **ARX VCP Classifier Candidate 006** (`ARX_VCP_CLASSIFIER_V1_CANDIDATE_006`, Ruleset `1.2.0`).

### Reproduction Taxonomy & Boundary Declarations
Reproduction is formally partitioned into two distinct operational tiers:

1. **Self-Contained Test & Conformance Baseline (Clean-Checkout CI)**:
   - `tests/test_vcp_candidate_006_conformance.py` (15 tests: SYN-01 through SYN-15)
   - `tests/test_vcp_candidate_005_conformance.py` (11 tests: ACC-01 through ACC-11)
   - `tests/test_vcp_candidate_reproducibility.py` (8 tests: test_00 through test_07)
   - The reproduction tests execute against the committed market database `data/research/market_data/bounded_150_dev_market_store.db` (committed via PR #33 commit `f6c5755`) and the committed fixture package in `tests/fixtures/vcp/`.
   - **All 34 tests execute and pass in a clean git checkout with zero external downloads, brain directories, or skips.**

2. **Full Historical Development Replay Package (60 HVA-00 Cases)**:
   - Replays all 60 development cases from scratch using `scripts/research/vcp/replay_candidate_006.py`.
   - Reads from `tests/fixtures/vcp/vcp_hva00_case_manifest.json` and `data/research/market_data/bounded_150_dev_market_store.db`.
   - Evaluates all cases and verifies:
     * 3/15 positive-label cases qualified (`FTNT` POS-03, `FTNT` POS-04, `DHLGY` POS-10).
     * 0/45 control false positives (38 negative classifications + 7 data abstentions).
     * 8 total abstentions (7 control + 1 positive case `RTX`).

---

## 2. Comprehensive Dependency Inventory

### Tier 1: Committed Repository Assets (Source Control Baseline)

These assets are versioned directly in Git on branch `feat/vcp-candidate-006-research` and are immediately available in any clone.

| Asset / Relative Path | Git Status | Size (Bytes) | SHA-256 Hash | Role / Test Requirement |
| :--- | :---: | :---: | :--- | :--- |
| `analyst_dashboard/vcp/classifier.py` | Committed | 40,581 | `98d2e11b3b39ffe3f65f972e3abd20ed4d194befe43b5c7b1a3d2ecac2379026` | C006 causal wave segmentation engine; maintains C004 default. |
| `analyst_dashboard/vcp/numeric_contract.py` | Committed | 7,722 | `74aaa38e268b7fa8736eb1ddb80e702f3658478eaf2d98003ef3b1a1ad0b6d55` | Precision contracts & `verify_progressive_tightening` logic. |
| `analyst_dashboard/vcp/predicate_registry.py` | Committed | 38,428 | `22756a2f2dbee5919e88dc0f6dea5ea51c555950392ac55d4c6ac52fbabb8e2e` | Reason codes including `INITIAL_CONTRACTION_TOO_SHALLOW`. |
| `analyst_dashboard/analyzers/scanner_contract.py` | Committed | 9,263 | `6b35f1feebcffa87e8ce5a742a5d151f7107de573f5b203406f22c4f69bf911c` | `VCP_CANDIDATE_006_RULESET_VERSION` / `ALGORITHM_VERSION`. |
| `analyst_dashboard/analyzers/scanner_publication_integrity.py` | Committed | 12,698 | `131c1eff7f514028790e2aefa01acccf397b36889e5c46c43aa921fae259d58e` | Canonical ruleset specification v1.2.0 & integrity registration. |
| `tests/test_vcp_candidate_006_conformance.py` | Committed | 26,051 | `a736229f062aac754af77d66c9a6861264efec509668fd5ddd2b4be3a59d3c76` | 15 synthetic conformance tests (SYN-01 through SYN-15). |
| `tests/test_vcp_candidate_005_conformance.py` | Committed | 20,148 | `687c8a80e0ffada4bb0bb9ef02c81c39e0ee5e7ee89bf7619d84b9058847c3fb` | 11 preserved C005 conformance tests (ACC-01 through ACC-11). |
| `tests/test_vcp_candidate_reproducibility.py` | Committed | Packaged | (Updated) | 8 cross-candidate reproducibility, fixture integrity, and isolation tests. |
| `data/research/market_data/bounded_150_dev_market_store.db` | Committed | 4,714,496 | `ab31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615` | Historical OHLCV market database (tracked via commit `f6c5755` on `origin/main`). |
| `tests/fixtures/vcp/vcp_hva00_case_manifest.json` | Committed | 65,822 | `407fb5f7dab217501cda0b7c936f6ef894b9b4ce773153da35b142e1278b9bb7` | Sealed 60-case corpus manifest for clean-checkout CI replay. |
| `tests/fixtures/vcp/vcp_hva00_frozen_algorithm_results.json` | Committed | 310,897 | `efe13e8a4672422b0e58e179c97cbdcd7b4c33937d036767eb5a011c12cc51b4` | Candidate 004 frozen reference results for CI regression. |
| `tests/fixtures/vcp/vcp_hva02_candidate_005_results.json` | Committed | 318,426 | `e34863b566df0087822eff71309381c50d6f538636f52120a02b3018935d62fc` | Candidate 005 frozen reference results for CI regression. |
| `tests/fixtures/vcp/vcp_hva02_candidate_006_results.json` | Committed | 374,810 | `9945108c2e0647654384be79265e6df07787f2ecb8185fdb5b49cda4e360a54a` | Candidate 006 frozen reference results for CI regression. |
| `tests/fixtures/vcp/FIXTURE_MANIFEST.json` | Committed | Packaged | (Manifest) | Cryptographic hash inventory for all packaged fixtures. |
| `scripts/research/vcp/replay_candidate_006.py` | Committed | Packaged | (Script) | Portable historical development replay runner. |
| `scripts/research/vcp/README.md` | Committed | Packaged | (Doc) | Concise reproduction instructions with exact commands. |
| `ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION.md` | Committed | 41,306 | `ae389f6d0af8d5612b50606a954cec31905609e90e4d9c3f265ae3320137bbbd` | Candidate 006 frozen research specification (Revision 2.3.0). |
| `ARX_VCP_CANDIDATE_006_IMPLEMENTATION_AND_DEVELOPMENT_EVALUATION.md` | Committed | Reconciled | (Updated) | Reconciled evaluation report on 60 exposed development cases. |
| `ARX_VCP_CANDIDATE_006_REPRODUCTION_DEPENDENCY_MANIFEST.md` | Committed | Reconciled | (This Doc) | This dependency mapping and reproduction protocol document. |

---

## 3. Environment & Runtime Specifications

- **Python Version**: Python 3.11.x (tested on 3.11.0/3.11.9, Windows x86-64).
- **Test Framework**: `pytest >= 7.0`.
- **Required Standard Library Modules**: `hashlib`, `json`, `sqlite3`, `dataclasses`, `typing`, `datetime`, `math`, `pathlib`.
- **Zero Third-Party Model / ML Dependencies**: Candidate 006 is 100% deterministic rule-based code. No torch, tensorflow, or network APIs required.
- **Path / Configuration Variables**:
  - `PYTHONPATH=.` when executing pytest or scripts from repository root.
  - Zero environment secrets, API tokens, or external network connectivity required.

---

## 4. Exact Reproduction Procedures

### Scope 1: Run Conformance & Reproducibility Suite (34 tests)
```bash
python -m pytest tests/test_vcp_candidate_006_conformance.py tests/test_vcp_candidate_005_conformance.py tests/test_vcp_candidate_reproducibility.py -v
```
*Expected*: 34 passed in ~5.5 seconds (zero skipped tests in clean-checkout mode).

### Scope 2: Replay Historical Development Evaluation (60 Cases)
```bash
python scripts/research/vcp/replay_candidate_006.py
```
*Expected*: Replays all 60 cases against `bounded_150_dev_market_store.db`, outputs summary confirming:
- 3/15 positive qualifications across 2 distinct issuers:
  * `FTNT` (`HVA00-POS-03` cutoff `2026-10-02`, volume ratio `0.6273`)
  * `FTNT` (`HVA00-POS-04` cutoff `2026-10-05`, volume ratio `0.6448`)
  * `DHLGY` (`HVA00-POS-10` cutoff `2026-09-01`, volume ratio `0.4670`)
- 0/45 control false positives (38 negative classifications + 7 abstentions).
- 8 total abstentions (7 control abstentions + 1 positive case `RTX`).
