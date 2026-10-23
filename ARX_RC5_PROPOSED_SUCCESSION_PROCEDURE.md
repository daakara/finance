# ARX TERMINAL — RC5 PROPOSED MANIFEST SUCCESSION PROCEDURE

**Date:** 2026-10-23T11:25:00Z (Local: 2026-10-23T13:25:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Status:** PROPOSAL ONLY — STRICTLY NON-MUTATING UNTIL SEPARATE AUTHORIZATION  
**Required Authorization Token for Execution:** `ARX_RC5_MANIFEST_SUCCESSION_MUTATION_AUTHORIZED`  

---

## 1. Principles of Valid Succession

1. **No Silent Hash Bumping:** Modifying a hash without accompanying semantic invariance proof and historical preservation is strictly forbidden.
2. **Chain-of-Custody Preservation:** Predecessor manifests (`FROZEN_ENGINE_MANIFEST_V2_5_1.json`, etc.) must remain immutable and preserved on disk.
3. **Atomic Transition:** The creation of the successor manifest and the update of corresponding test assertions must occur in a single atomic commit.
4. **Human Ratification Required:** Only the repository owner (`daakara`) can ratify the successor manifest into production authority.

---

## 2. Step-by-Step Execution Sequence (When Authorized)

```
[Separate PO Authorization]
          │
          ▼
1. Pre-flight Verification (Verify worktree clean & tree hash matches e6900b...)
          │
          ▼
2. Create Successor Manifest FROZEN_ENGINE_MANIFEST_V2_5_2.json
   - Optimal Execution SHA-256: 84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576
   - Decision Hierarchy SHA-256: 823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3
   - Confluence Engine SHA-256: 0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a
   - Supersedes: FROZEN_ENGINE_MANIFEST_V2_5_1.json
          │
          ▼
3. Create Successor Observation Manifest EPOCH_4_MANIFEST_V6.json
   - api/routes/analytics.py SHA-256: 361ae1daa6ca5379222b65f00cc31c4bad0ec5589d53dac3d7c2ad9e39aea278
   - Preserves 9 other executable governance file SHAs unchanged
          │
          ▼
4. Update Active Test Compliance Assertions
   - tests/test_f13a_epoch_compatibility.py:
       * Add V2_5_1 historical immutability check
       * Update active production check to V2_5_2 / 0f77d3b0...
   - tests/test_measurement_adversarial_suite.py:
       * Update test_production_engine_freeze_manifest_compliance
          │
          ▼
5. Full Local Verification & Linter Pass
   - python -m pytest -m tier1 -v
   - python -m pytest tests/test_corporate_events_boundaries.py tests/test_corporate_events_earnings_audit.py
   - ruff check . && bandit -r analyst_dashboard api
          │
          ▼
6. Human Ratification & Git Commit
   - Document in ARX_RC5_MANIFEST_SUCCESSION_RATIFICATION_RECORD.md
   - Author commit as arx-terminal-agent[bot]
          │
          ▼
7. Scoped Push to reconcile/sprint3-rc5-corporate-events
          │
          ▼
8. CI Verification on GitHub (Run against PR #53)
```

---

## 3. Exact Proposed Manifest Content (`FROZEN_ENGINE_MANIFEST_V2_5_2.json`)

```json
{
  "manifestVersion": "2.5.2",
  "frozenTimestamp": "2026-10-23T12:00:00Z",
  "provenanceCommit": "7980d3a6e5dde0fe172d215c729ec213b9a164b1",
  "frozenStrategyVersion": "2.5.2",
  "releaseAuthorityBinding": "SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION",
  "ratifiedAuthorityRecord": "ARX_RC5_MANIFEST_SUCCESSION_RATIFICATION_RECORD.md",
  "supersedesManifest": "FROZEN_ENGINE_MANIFEST_V2_5_1.json",
  "authorityScope": "CURRENT_PRODUCTION_AUTHORITY",
  "engines": {
    "optimal_execution": {
      "filePath": "analyst_dashboard/analyzers/optimal_execution.py",
      "sha256": "84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576",
      "ratificationRecord": "Release Addendum 002 (Binding PO Rejection of EXC-002)",
      "ratificationCommit": "cd0922471767775636957df74406a2d5efb8f519"
    },
    "confluence_engine": {
      "filePath": "analyst_dashboard/analyzers/confluence_engine.py",
      "sha256": "0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a",
      "ratificationRecord": "Corporate Events RC5 Runway Risk Integration",
      "ratificationCommit": "7980d3a6e5dde0fe172d215c729ec213b9a164b1"
    },
    "decision_hierarchy": {
      "filePath": "analyst_dashboard/analyzers/decision_hierarchy.py",
      "sha256": "823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3",
      "ratificationRecord": "Synthesis E Wave 3 Decision Integrity (PRD Addendum 001)",
      "ratificationCommit": "9d5fc2bc9b5029f02177dbe2ab50e026fbfb5f69"
    }
  },
  "invariants": {
    "zero_production_mutation": true,
    "enforced_by": "tests/test_measurement_adversarial_suite.py::test_production_engine_freeze_manifest_compliance"
  }
}
```

---

## 4. Rollback and Fail-Closed Containment

If the ratification or verification fails at any stage:
- The worktree is restored immediately using `git reset --hard 7980d3a6e5dde0fe172d215c729ec213b9a164b1`.
- Any uncommitted files are removed.
- Production mainline (`main`) remains 100% untouched and insulated.
