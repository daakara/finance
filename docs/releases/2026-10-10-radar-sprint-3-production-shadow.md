# ARX Terminal — Production Shadow Release Note
## Radar Sprint 3 Production Shadow: Candidate Freeze & Non-Actioning Shadow Deployment

**CANDIDATE_GENERATION_ID** = CANDIDATE_GENERATION_001  
**CANDIDATE_FUNCTIONAL_SHA** = bf0a574de569c2aefc219d5e9b1f891d9b6219d5  
**CANDIDATE_SEMANTIC_CLOSURE_HASH** = 53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6  
**GOVERNANCE_SHA256** = ce9ca0a1ca32390ee0f3d4818a76bb3739a1a336b1d0c4fdbb8dc0d1f128d207  
**RELEASE_DATE** = 2026-10-10  
**PRODUCTION_BASELINE_GIT_HEAD** = a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f  
**CANDIDATE_FREEZE_STATUS** = FROZEN_PRE_DEPLOY  
**EXTERNAL_VALIDATION_STATUS** = DEFERRED  
**GATE_12_STATUS** = DEFERRED / NOT_SATISFIED  
**PRODUCTION_VERIFICATION** = VERIFIED  
**DEPLOYED_GIT_HEAD** = 793df4844cfe0879ad8e6143a256dd214cf0b908  
**SHADOW_ACTIVATED_AT** = 2026-10-10T06:17:05Z  
**FIRST_NATURAL_SHADOW_CAPTURE_STATUS** = AWAITING_NATURAL_PRODUCTION_CAPTURE  
**SPRINT_3_PRODUCTION_SHADOW_STATUS** = ACTIVE / PRODUCTION_VERIFIED  

---

### 1. Release Scope & Objectives

This release activates Sprint 3 Production Shadow engineering for the ARX Terminal Radar subsystem under explicit Product Owner external-validation deferral.
The release encompasses:
1. **Candidate Semantic Closure**: Explicit enumeration and classification of all 17 decision-affecting inputs into `CandidateSemanticClosure` with immutable hash `53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6`.
2. **Candidate Generation Freeze**: Formal freeze of `CANDIDATE_GENERATION_001` bound to functional commit `bf0a574de569c2aefc219d5e9b1f891d9b6219d5`.
3. **Production Shadow Routing Wiring**: Integration of `Sprint3ShadowGovernanceSuite` into `VCPScannerRunner.execute_market_wide_scan()` recording prospective observations atomically with zero manifest violation.
4. **Contamination Controls & Holdout Exclusion**: Atomic registration of observed tickers/setups into `ProductionExposureLedger` and immediate exclusion from holdout pools via `HoldoutExclusionRegistry`.
5. **Denominator Classification & Isolation**: Strict segregation ensuring synthetic, replay, test, and admin-forced observations never enter the natural production denominator.
6. **Strict Non-Actioning Guarantees**: Absolute fail-closed isolation preventing shadow outcomes from placing orders, executing trades, mutating portfolios, or overriding primary decisions.

---

### 2. Governance, Deferral & Anti-Contamination Anchors

- **External Validation Deferral**: External validation remains explicitly `DEFERRED`. No claim of generalizability or gold/silver external domain authority is authorized (`NOT_ESTABLISHED`).
- **Gate 12 Status**: Remains `DEFERRED / NOT_SATISFIED`. No inference of gate satisfaction is permitted.
- **Model Tuning Policy**: `FROZEN`. Model weights, hyperparameters, and screening thresholds are immutable. `ONLINE_LEARNING = DISABLED`; `LEARNING_CLAIM = NOT_AUTHORIZED`.
- **Holdout Protection**: Any candidate evaluated in production shadow is irreversibly recorded in `HoldoutExclusionRegistry` to prevent future validation leakage.
- **Outcome Access Boundary**: Developers have zero unrestricted access to live outcome streams for parameter adjustments.

---

### 3. Non-Actioning Architectural Guarantees

The deployed runtime enforces:
- `SHADOW_USER_ORDER_EXECUTION` = `DISABLED`
- `SHADOW_PORTFOLIO_MUTATION` = `DISABLED`
- `SHADOW_BROKER_EXECUTION_HOOK` = `DISABLED`
- `SHADOW_PRIMARY_USER_DECISION_OVERRIDE` = `DISABLED`
- `SHADOW_ACTIONING_CALL_PATHS` = `0`

Shadow observations are passively logged to structured JSONL prospective streams with cryptographic provenance, isolated from transactional systems.

---

### 4. Verification & Testing Evidence

- **Focused Shadow Governance Tests**: 30 collected / 30 passed / 0 failed in 1.49s.
  - Tests A–T: Exposure accounting, holdout exclusion, fail-closed generation validation, denominator isolation, immutability.
  - Tests U–Z: Mutation sensitivity tests (VCP predicate, thresholds, universe, data interpretation, non-semantic stability, holdout leak prevention).
  - Tests AA–DD: Atomic observation recording, unregistered exposure rejection, denominator metrics isolation, and scanner runner wiring.
- **Release Union Regression Suite**: 550 collected / 550 passed / 0 failed in 30.03s.
- **Critical AST & Syntax Check**: `python -m flake8 . --count --select=E9,F63,F7,F82` passed with 0 errors.
- **Local CI Replication**:
  - `CI_NEW_FAILURE_COUNT` = `0`
  - `CI_PRE_EXISTING_FAILURE_COUNT` = `10` (flake8 on research scripts) + `3` (legacy tier1 tests untouched since pre-baseline).
  - `CI_RELEASE_EXCEPTION_STATUS` = `UNCHANGED_PRE_EXISTING_FAILURE`.

---

### 5. Deployment Tracking & Runtime Status

- **Pre-Push Baseline**: `a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f`
- **Candidate Functional Commit**: `bf0a574de569c2aefc219d5e9b1f891d9b6219d5`
- **Target Backend**: Railway (`web` service, production environment)
- **Target Frontend**: Cloudflare Pages (`finance-xp8.pages.dev`, `arxterminal.com`)
- **Deployed Git Head**: `793df4844cfe0879ad8e6143a256dd214cf0b908`
- **Functional Candidate Inclusion**: `git merge-base --is-ancestor bf0a574 793df48` = `YES`
- **Railway Deployment**: ID `1fdf85e8-c214-4b71-9710-c4fa136ccaef` (`SUCCESS` at `2026-10-10 08:16:27 +02:00`)
- **Runtime Attestation**: Live process attestation `backend_release_sha="793df4844cfe0879ad8e6143a256dd214cf0b908"`
- **Cloudflare Edge Deployment**: Verified (`200 OK` at `https://finance-xp8.pages.dev` and `https://www.arxterminal.com`)
- **GitHub Actions CI**: Run ID `38030363811` on `793df48` (evaluated)
- **Runtime Health**: `/health` (`200 OK`), `/api/v1/screener/vcp/snapshot` (`200 OK`)
- **Non-Actioning Gate**: `PASS` (zero order execution, zero portfolio mutation, zero broker hooks)
- **Shadow Capture Activation**: `ACTIVE` as of `2026-10-10T06:17:05Z`
- **First Natural Capture Status**: `AWAITING_NATURAL_PRODUCTION_CAPTURE` (Outcome B, non-manufactured)
- **Production Status**: `ACTIVE / PRODUCTION_VERIFIED`
