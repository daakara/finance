# ARX Terminal — Candidate 006 Reproduction Dependency Manifest

**Document Identifier**: ARX_VCP_CANDIDATE_006_REPRODUCTION_DEPENDENCY_MANIFEST  
**Execution Timestamp**: 2026-10-24T01:45:00+02:00  
**Governing Framework**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Final Approval Authority**: David Akara  
**Workstream**: ARX_VCP_RESEARCH_WAVE_FRAGMENTATION_01  
**Target Branch**: eat/vcp-candidate-006-merging  
**Target Worktree**: C:\Users\akara\Documents\Projects\finance-vcp-candidate-006-merging  
**Base Commit**: 13d4e0862561ac0ebb486d0ca59bbd5defb14075  
**Base Tree**: 40cb84d2952eedef7715defa62559bb20a5d94de  
**Status**: REPRODUCTION_DEPENDENCIES_IDENTIFIED_AND_MAPPED

---

## 1. Executive Summary

This manifest catalogues every asset, artifact, test oracle, and data dependency required to reproduce the synthetic conformance tests, cross-candidate preservation tests, and historical development evaluations for **ARX VCP Classifier Candidate 006** (ARX_VCP_CLASSIFIER_V1_CANDIDATE_006, Ruleset 1.2.0).

### Reproduction Taxonomy & Boundary Declarations
Reproduction is formally partitioned into three distinct operational tiers to prevent ambiguity regarding what can be verified solely within Git versus what requires external research assets:

1. **Self-Contained Synthetic Test Tier**:
   - 	ests/test_vcp_candidate_006_conformance.py (15 tests: SYN-01 through SYN-15)
   - 	ests/test_vcp_candidate_005_conformance.py (11 tests: ACC-01 through ACC-11)
   - These tests construct OHLCV series dynamically in memory and verify exact causal segmentation, strict tightening thresholds, tie-breaking, depth calculations, and error-handling in complete isolation.
   - **Zero external data, environment variables, or databases are required.**

2. **Cross-Candidate Preservation & Dual-Execution Tier**:
   - 	ests/test_vcp_candidate_reproducibility.py (6 tests: test_01 through test_06)
   - Verifies Candidate 004 default selection, Candidate 004/005 isolation, and replay against frozen results when present.
   - Tests 	est_01, 	est_02, and the tri-candidate execution order invariance in 	est_06 execute in memory without external files.
   - Tests 	est_03, 	est_04, and 	est_05 replay against the committed SQLite database data/research/market_data/bounded_150_dev_market_store.db and compare outputs against frozen results files. If the untracked JSON results files are absent, tests gracefully invoke pytest.skip() without failing.

3. **Full Historical Development Replay Tier (60 HVA-00 Cases)**:
   - Full re-execution of the 60 development cases requires the external/untracked research corpus manifest, market database, and evaluation harness.
   - **Crucial Boundary Disclosure**: A completely self-contained Git-only clone cannot execute the full 60-case historical evaluation without restoring the Tier 2 JSON assets and Tier 3 evaluation harness preserved in the persistent evidence directory.

---

## 2. Comprehensive Dependency Inventory

### Tier 1: Committed Repository Assets (Source Control Baseline)

These assets are versioned directly in Git on branch eat/vcp-candidate-006-merging and are immediately available in any clone.

| Asset / Relative Path | Git Status | Size (Bytes) | SHA-256 Hash | Role / Test Requirement |
| :--- | :---: | :---: | :--- | :--- |
| nalyst_dashboard/vcp/classifier.py | Committed | 40,581 | 98d2e11b3b39ffe3f65f972e3abd20ed4d194befe43b5c7b1a3d2ecac2379026 | C006 causal wave segmentation engine; maintains C004 default. |
| nalyst_dashboard/vcp/numeric_contract.py | Committed | 7,722 | 74aaa38e268b7fa8736eb1ddb80e702f3658478eaf2d98003ef3b1a1ad0b6d55 | Precision contracts & erify_progressive_tightening logic. |
| nalyst_dashboard/vcp/predicate_registry.py | Committed | 38,428 | 22756a2f2dbee5919e88dc0f6dea5ea51c555950392ac55d4c6ac52fbabb8e2e | Reason codes including INITIAL_CONTRACTION_TOO_SHALLOW. |
| nalyst_dashboard/analyzers/scanner_contract.py | Committed | 9,263 | 6b35f1feebcffa87e8ce5a742a5d151f7107de573f5b203406f22c4f69bf911c | VCP_CANDIDATE_006_RULESET_VERSION / ALGORITHM_VERSION. |
| nalyst_dashboard/analyzers/scanner_publication_integrity.py | Committed | 12,698 | 131c1eff7f514028790e2aefa01acccf397b36889e5c46c43aa921fae259d58e | Canonical ruleset specification v1.2.0 & integrity registration. |
| 	ests/test_vcp_candidate_006_conformance.py | Committed | 26,051 | 736229f062aac754af77d66c9a6861264efec509668fd5ddd2b4be3a59d3c76 | 15 synthetic conformance tests (SYN-01 through SYN-15). |
| 	ests/test_vcp_candidate_005_conformance.py | Committed | 20,148 | 687c8a80e0ffada4bb0bb9ef02c81c39e0ee5e7ee89bf7619d84b9058847c3fb | 11 preserved C005 conformance tests (ACC-01 through ACC-11). |
| 	ests/test_vcp_candidate_reproducibility.py | Committed | 14,567 | 74b1d5672dacfb7eff4a50134a4cc138bc7ea85acda17b8c78d0d2a3ad20252a | 6 cross-candidate reproducibility and isolation tests. |
| data/research/market_data/bounded_150_dev_market_store.db | Committed | 4,714,496 | b31fdffa0b8b20af714bf12becf55f9ea81d17018fdda81c77e3c7bb1ade615 | Historical OHLCV market database (tracked via commit 6c5755). |
| ARX_VCP_WAVE_FRAGMENTATION_RESEARCH_SPECIFICATION.md | Committed | 41,306 | e389f6d0af8d5612b50606a954cec31905609e90e4d9c3f265ae3320137bbbd | Candidate 006 frozen research specification (Revision 2.3.0). |
| ARX_VCP_CANDIDATE_006_IMPLEMENTATION_AND_DEVELOPMENT_EVALUATION.md | Committed | 21,550 | 4082ac74bd2bf1e8257795baaec96bc9c94860bb107cee6ee0ba467c387a175b | Reconciled evaluation report on 60 exposed development cases. |
| ARX_VCP_CANDIDATE_006_REPRODUCTION_DEPENDENCY_MANIFEST.md | Committed | (Current) | (Self) | This dependency mapping and reproduction protocol document. |

---

### Tier 2: Excluded Research Data Assets (Ignored by Git, Preserved in Evidence Store)

These files are located on disk under data/research/ but are excluded from Git commits by .gitignore:42:data/*. Byte-identical copies are preserved in the persistent brain evidence store.

| Asset / File Path | Git Status | Size (Bytes) | SHA-256 Hash | Persistent Evidence Store Location |
| :--- | :---: | :---: | :--- | :--- |
| data/research/vcp_hva00_case_manifest.json | Ignored (data/*) | 65,822 | 407fb5f7dab217501cda0b7c936f6ef894b9b4ce773153da35b142e1278b9bb7 | .../evidence/vcp_candidate_005/vcp_hva00_case_manifest.json |
| data/research/vcp_hva00_frozen_algorithm_results.json | Ignored (data/*) | 310,897 | efe13e8a4672422b0e58e179c97cbdcd7b4c33937d036767eb5a011c12cc51b4 | .../evidence/vcp_candidate_005/vcp_hva00_frozen_algorithm_results.json |
| data/research/vcp_hva02_candidate_005_results.json | Ignored (data/*) | 318,426 | e34863b566df0087822eff71309381c50d6f538636f52120a02b3018935d62fc | .../evidence/vcp_candidate_005/vcp_hva02_candidate_005_results.json |
| data/research/vcp_hva02_candidate_006_results.json | Ignored (data/*) | 374,810 | 9945108c2e0647654384be79265e6df07787f2ecb8185fdb5b49cda4e360a54a | .../evidence/vcp_candidate_006/vcp_hva02_candidate_006_results.json |

---

### Tier 3: Evaluation Harness (Transient Scratch Script, Preserved in Evidence Store)

The evaluation runner is located in scratch/evaluate_candidate_006_hva00.py (ignored by .git/info/exclude:8:scratch/). A byte-identical copy is preserved in the persistent brain evidence store.

| Asset / File Path | Git Status | Size (Bytes) | SHA-256 Hash | Persistent Evidence Store Location |
| :--- | :---: | :---: | :--- | :--- |
| scratch/evaluate_candidate_006_hva00.py | Ignored (scratch/) | 9,611 | 3daa19a8db15bc21b10261d788d8dbc9809fafb1998cfbedc18276e1a85203a8 | .../evidence/vcp_candidate_006/evaluate_candidate_006_hva00.py |

---

## 3. Environment & Runtime Specifications

- **Python Version**: Python 3.11.x (tested on 3.11.9, Windows x86-64).
- **Test Framework**: pytest >= 7.0 (with pytest-asyncio if full suite invoked).
- **Required Standard Library Modules**: hashlib, json, sqlite3, dataclasses, 	yping, datetime, math.
- **Zero Third-Party Model / ML Dependencies**: Candidate 006 is 100% deterministic rule-based code. No torch, tensorflow, or network APIs required.
- **Path / Configuration Variables**:
  - PYTHONPATH=. when executing pytest or scripts from repository root.
  - No environment secrets, API tokens, or external network connectivity required.

---

## 4. Exact Reproduction Procedures

### Scope 1: Run Candidate 006 Synthetic Conformance Tests (15 tests)
`powershell
python -m pytest tests/test_vcp_candidate_006_conformance.py -v
`
*Expected*: 15 passed in ~0.60 seconds.

### Scope 2: Run Full Unit & Reproducibility Suite (32 tests)
`powershell
python -m pytest tests/test_vcp_candidate_006_conformance.py tests/test_vcp_candidate_005_conformance.py tests/test_vcp_candidate_reproducibility.py -v
`
*Expected*: 32 passed in ~1.30 seconds.

### Scope 3: Replay Historical Development Evaluation (60 Cases)
`powershell
python scratch/evaluate_candidate_006_hva00.py
`
*Expected*: Replays all 60 cases against ounded_150_dev_market_store.db, writes data/research/vcp_hva02_candidate_006_results.json, and outputs summary confirming 3/15 positive qualifications (FTNT POS-03, FTNT POS-04, DHLGY POS-10) and 0/45 control false positives (38 negative classifications, 7 abstentions).
