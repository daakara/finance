# ARX Terminal — VCP Candidate 006 Research Replay Guide

This directory contains reproduction tooling for **Candidate 006 Causal Wave Segmentation** (`ARX_VCP_CLASSIFIER_V1_CANDIDATE_006`, Ruleset `1.2.0`).

## 1. Quick Verification Commands

### A. Run Conformance & Reproducibility Test Suites (Pytest)
```bash
# Run 15 synthetic conformance tests (pure in-memory, < 2s)
python -m pytest tests/test_vcp_candidate_006_conformance.py -v

# Run 11 preserved Candidate 005 conformance tests (< 1s)
python -m pytest tests/test_vcp_candidate_005_conformance.py -v

# Run cross-candidate preservation and reproducibility tests (< 2s)
python -m pytest tests/test_vcp_candidate_reproducibility.py -v
```

### B. Run Historical Development Corpus Replay (60 Cases)
```bash
python scripts/research/vcp/replay_candidate_006.py
```
*Expected Output*:
- Evaluates 60 sealed development cases against `data/research/market_data/bounded_150_dev_market_store.db`.
- Confirms 3 / 15 positive qualifications across 2 issuers (`FTNT` POS-03, `FTNT` POS-04, `DHLGY` POS-10).
- Confirms 0 / 45 control false positives (38 negative classifications + 7 data abstentions).
- Confirms 8 total abstentions (7 control abstentions + 1 positive case `RTX` base-anchor/volume omission).

## 2. Packaged Fixtures & Data Dependencies

All reproduction fixtures are packaged in `tests/fixtures/vcp/`:
- `vcp_hva00_case_manifest.json`: 60-case corpus definitions and cutoffs.
- `vcp_hva00_frozen_algorithm_results.json`: Candidate 004 frozen outputs.
- `vcp_hva02_candidate_005_results.json`: Candidate 005 frozen outputs.
- `vcp_hva02_candidate_006_results.json`: Candidate 006 frozen outputs.
- `FIXTURE_MANIFEST.json`: Cryptographic SHA-256 hash inventory for all fixtures.

The historical market database `data/research/market_data/bounded_150_dev_market_store.db` is committed in the repository (tracked via commit `f6c5755` on `origin/main`).
