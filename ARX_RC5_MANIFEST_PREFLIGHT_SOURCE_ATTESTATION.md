# ARX TERMINAL — RC5 FROZEN ENGINE MANIFEST PREFLIGHT SOURCE ATTESTATION

**Date:** 2026-10-23T11:20:00Z (Local: 2026-10-23T13:20:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_READ_ONLY_PREFLIGHT_AUTHORIZED`  
**Repository:** `daakara/finance`  
**Execution Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint3-reconcile`  
**Execution Mode:** Read-Only Source Attestation and Cryptographic Audit  

---

## 1. Executive Attestation

This document establishes the verified cryptographic source baseline for the Corporate Events RC5 candidate currently under Draft Pull Request #53. All observations are derived from primary immutable Git and filesystem evidence within an isolated, clean worktree. Zero modifications to manifests, source files, or remote repository states have occurred.

---

## 2. Canonical Git Identities

| Identity Role | Git Object SHA | Tree SHA | Verified Reference |
|---|---|---|---|
| **Sprint 2 RC4 Production Baseline (`main`)** | `a62a91defd8a8f5f479017e50f633981d35a1672` | `d5ef77221356773256dc80524b34b94f3665ed98` | Origin `main` / Production Release |
| **RC5 Integration Merge Commit** | `c060700b2b3eedde2ec82800462b6c97a67c2939` | `c512187a9959b8a649e2aa5b0fa2841769643aaf` | Gate B Candidate Merge |
| **RC5 Stage 1 Test Addition Commit** | `a495d6b3c9da918df3300a1a16603eb99c12f917` | `f315dbb621e204c3561a34b28d7a1e05d048e91e` | Corporate Events Invariant Tests |
| **RC5 Stage 1 PR HEAD Commit** | `7980d3a6e5dde0fe172d215c729ec213b9a164b1` | `e6900bff5c624a16fc7571a2ef3bf381333f3ac9` | GitHub PR #53 HEAD |

### Remote PR Verification
- **Pull Request:** #53 (`https://github.com/daakara/finance/pull/53`)
- **PR State:** `open` (Draft: `true`)
- **Author:** `arx-terminal-agent[bot]`
- **Head Branch:** `reconcile/sprint3-rc5-corporate-events` (`7980d3a6e5dde0fe172d215c729ec213b9a164b1`)
- **Base Branch:** `main` (`a62a91defd8a8f5f479017e50f633981d35a1672`)
- **Mergeable State:** `blocked` (Ruleset `25187391` requires 1 review + 4 CI checks)

---

## 3. Dedicated Worktree Verification

- **Path:** `C:\Users\akara\Documents\Projects\finance-sprint3-reconcile`
- **Branch:** `reconcile/sprint3-rc5-corporate-events`
- **HEAD Commit:** `7980d3a6e5dde0fe172d215c729ec213b9a164b1`
- **Worktree Cleanliness:** Verified via `git status` (`nothing to commit, working tree clean`).

---

## 4. Cryptographic SHA-256 Attestation of Protected Assets

All file hashes recalculated using SHA-256 across normalized LF content:

| Protected File Path | Baseline (`main` / Manifest) SHA-256 | PR #53 Candidate SHA-256 | Invariant Status |
|---|---|---|---|
| `analyst_dashboard/analyzers/confluence_engine.py` | `147c010eb6068c3470fa3269f08610c0f436ea117b99b3b5217e7020b911e2b1` | `0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a` | **MODIFIED** (RC5 Runway Logic) |
| `analyst_dashboard/analyzers/optimal_execution.py` | `84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576` | `84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576` | **BITWISE IDENTICAL** (Compliant) |
| `analyst_dashboard/analyzers/decision_hierarchy.py` | `823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3` | `823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3` | **BITWISE IDENTICAL** (Compliant) |
| `analyst_dashboard/corporate_events.py` | *(New in RC5)* | `469f5f99c74da8cf426436520a7d4116437a7534665daf3af29e5085bf632585` | **NEW FILE** (RC5 Core Engine) |
| `analyst_dashboard/analyzers/catalysts.py` | `37803e4811a216a6729a99ea298ec436a0666a7b7d7ca9e2b1b36e890bb86f99` | `86e1240db20c2061721fd29e773f5c643d3a01e3f07dfaaf395db91ef9a1062c` | **MODIFIED** (RC5 Integration) |
| `api/routes/analytics.py` | `e1642878cf672eaeb49e29a9c8bb26c317ad5655b3eaef92b588ee279d48b111` | `361ae1daa6ca5379222b65f00cc31c4bad0ec5589d53dac3d7c2ad9e39aea278` | **MODIFIED** (API Contract Addition) |
| `FROZEN_ENGINE_MANIFEST_V2_5_1.json` | `bb51df94f6d328a721089cea7d6004b4dcac1c2b1eeea55d763b5f8f6a58c5f0` | `bb51df94f6d328a721089cea7d6004b4dcac1c2b1eeea55d763b5f8f6a58c5f0` | **UNMODIFIED** (Pending Succession) |
| `EPOCH_4_MANIFEST_V5.json` | `ca7b860da5aa72225d5ee2d7212d4a7d4b22dfb4de299d8dfc3f08d451fd40c4` | `ca7b860da5aa72225d5ee2d7212d4a7d4b22dfb4de299d8dfc3f08d451fd40c4` | **UNMODIFIED** (Pending Succession) |

---

## 5. Summary Conclusion

The candidate under PR #53 is strictly identified as `7980d3a6e5dde0fe172d215c729ec213b9a164b1`. Exactly one file governed under `FROZEN_ENGINE_MANIFEST_V2_5_1.json` (`confluence_engine.py`) has been modified, directly accounting for the cryptographic certification failure in CI.
