# ARX Terminal — Production Release Notes

## Production QA Escape Hardening & Release Quality Governance

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  5a084a9a9b9d01fdd090c1866dfbc4ffea1e3889
PREVIOUS_RUNTIME_SHA =
  6f0559d93a681c3bb7c3a89883e0a820934a4a7a
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway & Cloudflare deployment
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION =
  VERIFIED
```

---

### Release Classification & Behavior Invariants

```ini
RELEASE_CLASSIFICATION =
  QA_HARDENING
  GOVERNANCE
  OBSERVABILITY

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

PRODUCTION_CODE_BEHAVIOR_CHANGE =
  NONE
```

This release introduces comprehensive QA escape analysis, property-based regression suites, semantic invariant checks, pre-promotion smoke gating, and deterministic post-deploy release verification. Zero modifications were made to production quantitative recommendation models, portfolio algorithms, execution ladder thresholds, or production application runtime logic.

---

### What Changed

#### Added
* **ARX QA Escape Registry (`docs/governance/ARX_QA_ESCAPE_REGISTRY.md`)**: Codified 10 historical production escapes (`QA-ESC-001` through `QA-ESC-010`) tracking four independent lifecycle states (`DEFECT_STATUS`, `REGRESSION_COVERAGE_STATUS`, `PREVENTION_CONTROL_STATUS`, `PRODUCTION_VERIFICATION_STATUS`) with authoritative fix provenance.
* **Pre-Promotion Smoke Gate (`scripts/qa/production_candidate_smoke_gate.py`)**: Standalone, automated pre-flight audit testing 5 representative canonical instruments (`AAPL`, `SPY`, `TSM`, `AMT`, `UNKNOWN_TICKER`) across technical indicators, data schemas, semantic invariants, and execution ladder ordering prior to promotion.
* **Post-Deploy Production Verification Gate (`scripts/qa/post_deploy_production_verification.py`)**: Automated verification gate inspecting backend health, macro ribbon authority, security master parity, frontend bundle delivery, and deterministic functional release resolution via Git ancestry and release note metadata.
* **Backend Property-Based Regression Suite (`tests/test_qa_escape_invariants.py`)**: 8 permanent backend regression tests locking down macro regime TTL, un-entered prospective ladder directionality, smart money state triads, corporate 10-K exemptions for ETFs, and fail-closed security master routing.
* **Deployment Identity & Ancestry Test Suite (`tests/test_post_deploy_verification.py`)**: 8 permanent regression tests validating documentation-only redeployment semantics, missing release note failure detection, and ancestry validation.
* **Frontend Semantic Invariants Suite (`frontend/tests/qaEscapeSemanticInvariants.test.ts`)**: 8 permanent architecture tests integrated into `npm run test:arch` preventing regression of tour skip persistence, WebKit touch safe-areas, non-collapsing empty states, institutional copy standards, and MiniSparkline truthfulness (Rule D04).
* **Comprehensive Audit Dossier (`ARX_PRODUCTION_QA_ESCAPE_ANALYSIS_REPORT.md`)**: Full forensic analysis document detailing detection failure modes, systemic remediation patterns, and integration gates.

---

### Release Quality Architecture

Every future ARX Terminal candidate release must pass through the 5-stage release quality architecture:

```text
STATIC_CHECKS
  ↓
INVARIANT_SUITES
  ↓
PRE_PROMOTION_SMOKE
  ↓
POST_DEPLOY_AUDIT
  ↓
RELEASE_NOTES_CLOSURE
```

1. **STATIC_CHECKS**: Type-check (`tsc --noEmit`), ESLint, Python syntax verification.
2. **INVARIANT_SUITES**: Frontend architecture invariants (`test:arch`) and backend property invariants (`tests/test_qa_escape_invariants.py`).
3. **PRE_PROMOTION_SMOKE**: Live testing of 5 canonical instrument types using `scripts/qa/production_candidate_smoke_gate.py`.
4. **POST_DEPLOY_AUDIT**: Runtime verification inspecting Railway backend, Cloudflare frontend, macro ribbon, and security master using `scripts/qa/post_deploy_production_verification.py`.
5. **RELEASE_NOTES_CLOSURE**: Immutable committed release notes under `docs/releases/` verifying functional release ancestry.

---

### Test Coverage Accounting & Explicit Verification Limits

```ini
REGRESSION_TESTS_ADDED =
  24
SEMANTIC_INVARIANTS_ADDED =
  8
POST_DEPLOY_TESTS_ADDED =
  8
DOM_COMPONENT_AND_ARCH_TESTS_ADDED =
  8
RELEASE_SMOKE_CHECKS_ADDED =
  9

TRUE_BROWSER_E2E_TESTS_ADDED =
  0
WEBKIT_E2E_TESTS_ADDED =
  0
PHYSICAL_IOS_TESTS_EXECUTED =
  0
```

> [!NOTE]
> Physical iOS touch interactions and real-browser WebKit rendering were certified during earlier hotfix gates (`a10476a`, `2924391`). This QA hardening workstream codified permanent static DOM and CSS architectural invariants in JSDOM rather than adding headless browser dependencies.

---

### Passive Capture & Prospective Denominator

```ini
PASSIVE_CAPTURE_RESULT =
  42/42 PASS
PROSPECTIVE_DENOMINATOR =
  0
DENOMINATOR_DELTA =
  0
```

Zero verification requests, smoke runs, or test fixtures have written to prospective tables or logs. The prospective denominator remains exactly zero.
