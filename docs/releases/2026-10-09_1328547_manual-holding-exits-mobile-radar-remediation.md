# ARX Terminal — Production Release Notes

## Mobile Radar Overflow Remediation, Manual Holding Durable Exit Event Store & Portfolio Lifecycle Governance

### Release Identity

```ini
RELEASE_DATE =
  2026-10-09
FUNCTIONAL_RELEASE_SHA =
  1328547e6f7d347c57f11ad6e22a0ae5309e1abb
BASELINE_SHA =
  4e94e988d607acc5256cef8761c1fd02b6ffba92
PREVIOUS_RUNTIME_SHA =
  4e94e988d607acc5256cef8761c1fd02b6ffba92
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway deployment
DEPLOYMENT_ID =
  a8ac4116-39c3-457d-b13f-093a1ecf0710
DEPLOYMENT_STATUS =
  SUCCESS
PRODUCTION_VERIFICATION =
  PASS
```

---

### Release Purpose & Classification

```ini
RELEASE_PURPOSE =
  remediate mobile Radar horizontal reachability, establish durable manual holding exit event store, source-safe exit lifecycle, mixed-source exposure isolation, server-authoritative manual exit P&L, safe error handling, and realized-R fail-closed correction

RELEASE_CLASSIFICATION =
  BUG_FIX_AND_DATA_CONTRACT_ENHANCEMENT
  EVENT_STORE_ADDITION
  DATA_INTEGRITY_GOVERNANCE

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  YES / INTENTIONAL (Mobile Radar ownership controls reachable; manual exits create durable events rather than bare mutations/deletions; unrecorded initial risk fails closed to null R)

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

PROSPECTIVE_CAPTURE_CHANGE =
  NONE

EXECUTION_LADDER_CHANGE =
  NONE

EPOCH_4_CHANGE =
  NONE

MODEL_PARAMETER_CHANGE =
  NONE

DATABASE_SCHEMA_CHANGE =
  ADDITIVE (Added table portfolio_holding_exit_events, audit indexes, and immutability triggers)
```

---

### Root Cause Analysis & Technical Remediation

#### 1. Mobile Radar Ownership Filter Clipped
* **Defect**: On mobile viewports (<=393px), the Radar ownership filter row lacked horizontal scrolling, clipping options past "New Opportunities" such as "My Holdings".
* **Remediation**: Wrapped filter controls in an accessible `overflow-x-auto touch-pan-x` container with `overscroll-x-contain`, preserved `min-h-[44px]` touch targets, enabled auto-scroll on active filter selection, and eliminated document-level overflow.

#### 2. Manual Portfolio Full Close Failed in Trade Journal
* **Defect**: Portfolio "Full Close (100%)" routed manual holdings through `/api/v1/journal/close`, which strictly requires an open parent trade in `user_trade_journal`.
* **Remediation**: Decoupled manual portfolio exits from journal trades. Created dedicated authoritative endpoint `POST /api/v1/portfolio/holdings/{holding_id}/exit` backed by append-only `portfolio_holding_exit_events`.

#### 3. Bare Deletion & Lack of Exit History
* **Defect**: Previously, deleting or updating a manual holding via `DELETE /api/v1/portfolio/{symbol}` or `PUT /api/v1/portfolio/{symbol}` destroyed or mutated the row without creating any audit trail.
* **Remediation**: Implemented `portfolio_holding_exit_events` storing immutable exit events with SQLite triggers (`BEFORE UPDATE` and `BEFORE DELETE` raise `ABORT`). Active aggregate row in `portfolio_holdings` is only deleted when both `manual_shares == 0` and `journal_shares == 0`.

#### 4. Mixed-Source Exposure Isolation
* **Defect**: Holdings containing both manual shares and journal-backed shares lacked isolation.
* **Remediation**: Manual exits mutate `manual_shares` only. Journal shares and open `user_trade_journal` records remain completely untouched. For mixed positions, closing the manual portion reduces `manual_shares` to 0 while leaving journal exposure open.

#### 5. Strict FULL-Close Quantity Authority
* **Contract**: For `exitType: "FULL"`, client numeric `shares` are strictly prohibited (`INVALID_EXIT_QUANTITY` 400). Exit quantity is derived server-side from `portfolio_holdings.manual_shares` inside an atomic `BEGIN IMMEDIATE` transaction.

#### 6. Contradictory Realized-R on Profitable Trades
* **Defect**: Realized-R calculations used the current/trailing stop as the risk basis, which inverted R to negative on winning trades where the stop had been trailed past breakeven into profit.
* **Remediation**: Enforced fail-closed R calculation. If initial risk basis cannot be proven (`entry_price - initial_stop <= 0` or missing stop), R returns `null` (`UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED`) rather than producing contradictory economics.

#### 7. Safe API Error Handling
* **Contract**: Preserves safe domain 4xx error messages for display in the UI while redacting unknown 5xx server exceptions to a safe generic fallback.

---

### Database Migration & Immutability Verification

```ini
EVENT_TABLE =
  portfolio_holding_exit_events (PRESENT in production)
HISTORICAL_BACKFILL =
  0 (Strictly zero backfill rows inserted)
IDEMPOTENCY_NOT_NULL =
  YES (idempotency_key NOT NULL)
WORKSPACE_SCOPED_UNIQUENESS =
  YES (UNIQUE (workspace_id, idempotency_key))
UPDATE_TRIGGER =
  PRESENT (trg_holding_exit_events_no_update)
DELETE_TRIGGER =
  PRESENT (trg_holding_exit_events_no_delete)
PRE_DEPLOY_HOLDINGS_COUNT =
  8
POST_DEPLOY_HOLDINGS_COUNT =
  8 (PRESERVED)
PRE_DEPLOY_JOURNAL_COUNT =
  0
POST_DEPLOY_JOURNAL_COUNT =
  0 (PRESERVED)
POST_DEPLOY_DB_INTEGRITY =
  PASS (PRAGMA quick_check: ok)
```

---

### Non-Destructive Production Verification

In accordance with strict release safety constraints, verification was performed without generating real portfolio exits or mutating production data:

```ini
PRODUCTION_MANUAL_EXIT_REQUESTS_GENERATED_BY_SMOKE =
  0
PRODUCTION_PORTFOLIO_MUTATIONS_GENERATED_BY_SMOKE =
  0
PRODUCTION_JOURNAL_MUTATIONS_GENERATED_BY_SMOKE =
  0
PRODUCTION_EVENT_ROWS_INSERTED_BY_SMOKE =
  0
BACKEND_HEALTH =
  PASS (HTTP 200, status: online)
FRONTEND_HEALTH =
  PASS (HTTP 200)
DATABASE_REACHABLE_READ_ONLY =
  YES
MANUAL_EXIT_POST_ROUTE_REGISTERED =
  YES (POST /api/v1/portfolio/holdings/{holding_id}/exit)
PUBLIC_EXIT_HISTORY_GET_ROUTE =
  ABSENT
RADAR_PRODUCTION_375 =
  PASS (doc_overflow: false, touch_target: 44px, reachable: true)
RADAR_PRODUCTION_390 =
  PASS (doc_overflow: false, touch_target: 44px, reachable: true)
RADAR_PRODUCTION_393 =
  PASS (doc_overflow: false, touch_target: 44px, reachable: true)
RADAR_PRODUCTION_1024 =
  PASS (doc_overflow: false, touch_target: 44px, reachable: true)
RADAR_PRODUCTION_1440 =
  PASS (doc_overflow: false, touch_target: 44px, reachable: true)
RADAR_PRODUCTION_VERDICT =
  PASS
R_FAIL_CLOSED_DEPLOYED =
  PASS
JOURNAL_R_AUTHORITY_PRESERVED =
  PASS
SYNTHETIC_JOURNAL_PARENT_PATH =
  ABSENT
```

---

### Deferred Items & Governance Boundaries

* **Future Manual R Initial-Risk Schema**: DEFERRED. Initial risk persistence for manually entered holdings is deferred to a future schema specification. Manual holdings continue to fail closed to `null` R.
* **Empirical Production Exit Observation**: Not required for release verification. Natural user manual exits will be observed passively in runtime logs as genuine activity occurs.

---

### Canonical Normalized Release Ledger & Evidence Semantics

#### 1. Evidence Semantics & Status Axes

Every release assertion is evaluated across three orthogonal axes and categorized by requirement class. Status collapse is strictly prohibited.

* **VERIFICATION_STATUS**:
  * `PASS`: Authoritative evidence appropriate to the claim exists and supports it.
  * `FAIL`: Authoritative evidence contradicts the requirement.
  * `NOT_CAPTURED`: Historical evidence was not recorded at the relevant time and cannot be authoritatively reconstructed. Must never be interpreted as zero, false, or PASS.
  * `UNVERIFIED`: The claim remains unresolved from available provider telemetry. Must never be interpreted as PASS.
  * `OUT_OF_SCOPE`: The claim was not part of this release contract or its pre-existing mandatory governance obligations. Must never be interpreted as FAIL.
  * `NOT_APPLICABLE`: The requirement category exists but does not apply to this release/runtime.

* **BEHAVIOR_STATUS**:
  * `NORMAL`: Supported path operates as designed.
  * `SAFE_FAIL_CLOSED`: Runtime behavior under adverse/incompatible conditions prevents unsafe mutation, authority crossing, duplication, or ambiguity. This is strictly a runtime behavior status, never a verification status.
  * `DEGRADED`: Functionality impaired but safety invariants preserved.
  * `UNSAFE`: Safety or authority invariant violated.
  * `UNKNOWN`: Runtime behavior not established.
  * `NOT_APPLICABLE`: Not applicable to non-runtime claims.

* **CLOSURE_IMPACT**:
  * `BLOCKING`: Mandatory condition for release closure.
  * `NON_BLOCKING`: Important quality/historical claim whose unverified or uncaptured state is explicitly permitted by governance policy.
  * `INFORMATIONAL`: Contextual or follow-on tracking claim.

* **REQUIREMENT_CLASS**:
  * `RELEASE_REQUIREMENT`: Direct requirement governing this deployed release.
  * `GOVERNANCE_REQUIREMENT`: Pre-existing mandatory ARX governance and data integrity controls.
  * `HISTORICAL_EVIDENCE`: Point-in-time evidence that may or may not have been captured at deployment time.
  * `FOLLOW_ON_ARCHITECTURE`: Future design enhancements outside this release boundary.

* **Historical Immutability Rule**:
  Historical release records are immutable. A `NOT_CAPTURED` or `UNVERIFIED` status in an already-closed release cannot be retroactively rewritten to `PASS` merely because subsequent or post-deploy observations are consistent. Future reconciliations must be recorded in an additive `POST_CLOSURE_RECONCILIATION` section while leaving `ORIGINAL_LEDGER_MUTATED = NO`.

---

#### 2. Canonical Normalized Claims Ledger

```ini
[CLAIM-01]
claim_id = FUNCTIONAL_RELEASE_DEPLOYED
claim = Functional release commit is deployed and active in production
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = PROVIDER_METADATA
evidence_reference = Railway deployment a8ac4116-39c3-457d-b13f-093a1ecf0710 (SUCCESS) & docs deployment 6c470d20-3a34-473f-aa46-def3a285bf08 (SUCCESS)
observed_at = 2026-10-09T02:30:15Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Functional release 1328547 is parent of docs release da707da; active runtime ancestor verified.

[CLAIM-02]
claim_id = EVENT_STORE_PRESENT
claim = Durable append-only event store table exists with required schema, indexes, and triggers
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = DATABASE_SCHEMA
evidence_reference = Table portfolio_holding_exit_events (23 columns, 3 indexes, UNIQUE(workspace_id, idempotency_key), 2 ABORT triggers)
observed_at = 2026-10-09T02:40:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Additive migration executed cleanly with zero backfill rows.

[CLAIM-03]
claim_id = DB_INTEGRITY
claim = Production database integrity check passes and existing data preserved
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = DATABASE_STATE
evidence_reference = PRAGMA quick_check returned 'ok'; 8 pre-deploy holdings preserved; 0 synthetic exit rows
observed_at = 2026-10-09T02:40:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Non-destructive verification confirmed database health.

[CLAIM-04]
claim_id = FRONTEND_EXACT_SOURCE_SHA
claim = Production frontend provider metadata explicitly surfaces the source git commit SHA
requirement_class = RELEASE_REQUIREMENT
verification_status = UNVERIFIED
behavior_status = UNKNOWN
closure_impact = NON_BLOCKING
evidence_basis = PROVIDER_METADATA
evidence_reference = Cloudflare Pages deployment headers (ETag: "45d691121983fba6086528fb7ca0133b")
observed_at = 2026-10-09T02:45:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Production frontend behavior verified, but authoritative exact provider source SHA not available from retained deployment metadata.

[CLAIM-05]
claim_id = PRE_DEPLOY_PORTFOLIO_CONTENT_DIGEST
claim = Cryptographic content digest of portfolio_holdings recorded immediately prior to deploy
requirement_class = HISTORICAL_EVIDENCE
verification_status = NOT_CAPTURED
behavior_status = NOT_APPLICABLE
closure_impact = NON_BLOCKING
evidence_basis = HISTORICAL_RECORD
evidence_reference = Pre-deploy row count (8 rows) was recorded; cryptographic SHA256 content digest was not captured prior to deployment
observed_at = 2026-10-09T02:30:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Row count was captured; cryptographic content digest was not captured before deployment and must not be reconstructed from post-deploy state.

[CLAIM-06]
claim_id = PRE_DEPLOY_JOURNAL_CONTENT_DIGEST
claim = Cryptographic content digest of user_trade_journal recorded immediately prior to deploy
requirement_class = HISTORICAL_EVIDENCE
verification_status = NOT_CAPTURED
behavior_status = NOT_APPLICABLE
closure_impact = NON_BLOCKING
evidence_basis = HISTORICAL_RECORD
evidence_reference = Pre-deploy row count (0 rows) was recorded; cryptographic content digest was not captured prior to deployment
observed_at = 2026-10-09T02:30:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Row count was captured; cryptographic content digest was not captured before deployment and must not be reconstructed from post-deploy state.

[CLAIM-07]
claim_id = PRE_RELEASE_PROSPECTIVE_DENOMINATOR
claim = Epoch 4 prospective observation denominator baseline was exactly zero prior to release
requirement_class = HISTORICAL_EVIDENCE
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = DATABASE_STATE
evidence_reference = Contemporaneous pre-release measurement in /root/analyst_dashboard/data/governance.db: SELECT COUNT(*) FROM prospective_observations WHERE epoch = 4 returned 0 (reconfirmed from release 2026-10-08_b26163f)
observed_at = 2026-10-09T02:42:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Exact contemporaneous measurement confirmed at 0 before and after deployment.

[CLAIM-08]
claim_id = WEBKIT_PRODUCTION
claim = Radar mobile horizontal scroll and touch targets verified under native WebKit engine against production
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = NON_BLOCKING
evidence_basis = LIVE_PRODUCTION
evidence_reference = Playwright WebKit (Safari engine) execution against https://www.arxterminal.com/radar: doc_overflow: false, touch_target_ok: true (118x44px >= 44px), reachable: true
observed_at = 2026-10-09T03:06:50Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Empirical WebKit execution against production confirmed mobile reachability without page overflow.

[CLAIM-09]
claim_id = NEW_FE_OLD_BE
claim = New frontend client interacting with legacy backend without exit endpoint fails safely
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = SAFE_FAIL_CLOSED
closure_impact = NON_BLOCKING
evidence_basis = TEST_EXECUTION
evidence_reference = Integration & contract test suite: missing POST endpoint returns 404; frontend error handling catches gracefully; zero journal fallback or financial mutation occurs
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Manual exit cannot complete against missing endpoint; no journal fallback or financial mutation occurs.

[CLAIM-10]
claim_id = OLD_FE_NEW_BE
claim = Legacy frontend client attempting manual exit against new backend fails safely
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = SAFE_FAIL_CLOSED
closure_impact = NON_BLOCKING
evidence_basis = STRUCTURAL_INSPECTION
evidence_reference = Backend contract validation: legacy client routing manual exit to journal close fails closed with error; backend authority prevents unsafe journal mutation
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Legacy manual flow may remain functionally broken until client refresh, but backend authority prevents unsafe journal mutation.

[CLAIM-11]
claim_id = NEW_FE_NEW_BE
claim = New frontend client interacting with new backend endpoint executes manual exits normally
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = TEST_EXECUTION
evidence_reference = End-to-end integration tests (58 backend tests + 11 frontend test suites): atomicity, idempotency, server-authoritative quantity derivation, and event persistence verified
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Authoritative contract execution verified under both FULL and PARTIAL exit modes.

[CLAIM-12]
claim_id = RADAR_CHROMIUM_PRODUCTION
claim = Radar ownership filter reachability verified under Chromium across mobile/desktop viewports against production
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = LIVE_PRODUCTION
evidence_reference = Playwright Chromium production runs across 375, 390, 393, 1024, and 1440 viewports: doc_overflow: false, touch targets >= 44px, reachable: true, vertical jump: 0
observed_at = 2026-10-09T02:44:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Fully verified in live production environment.

[CLAIM-13]
claim_id = R_FAIL_CLOSED_DEPLOYED
claim = Unprovable initial risk basis in realized-R calculation fails closed to null
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = SAFE_FAIL_CLOSED
closure_impact = BLOCKING
evidence_basis = TEST_EXECUTION
evidence_reference = Backend tests (test_portfolio_routes.py): entry_price - initial_stop <= 0 yields null realized_r with UNAVAILABLE_ORIGINAL_RISK_NOT_RECORDED note
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Eliminates contradictory negative R on profitable trades while preserving journal-backed R authority.

[CLAIM-14]
claim_id = MIXED_SOURCE_ISOLATION
claim = Holdings with mixed manual and journal shares isolate manual exit mutations from journal exposure
requirement_class = RELEASE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = TEST_EXECUTION
evidence_reference = Backend tests: manual exit mutates manual_shares only; journal_shares and user_trade_journal rows remain untouched
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Verified under both FULL (manual shares closed to 0, journal shares open) and PARTIAL exit modes.

[CLAIM-15]
claim_id = GOVERNANCE_MANIFESTS_VERIFIED
claim = Prospective observation manifest V3 and historical V2 manifests preserved byte-for-byte
requirement_class = GOVERNANCE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = STRUCTURAL_INSPECTION
evidence_reference = Byte-for-byte SHA256 integrity check against docs/governance/manifests/
observed_at = 2026-10-09T02:40:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Zero modification to Epoch 4 governance artifacts.

[CLAIM-16]
claim_id = ZERO_SMOKE_MUTATIONS
claim = Production verification completed without synthetic manual exits or database mutations
requirement_class = GOVERNANCE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = DATABASE_STATE
evidence_reference = PRODUCTION_MANUAL_EXIT_REQUESTS_GENERATED_BY_SMOKE = 0; PRODUCTION_PORTFOLIO_MUTATIONS_GENERATED_BY_SMOKE = 0; PRODUCTION_JOURNAL_MUTATIONS_GENERATED_BY_SMOKE = 0
observed_at = 2026-10-09T02:45:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Non-destructive release verification constraint fully adhered to.

[CLAIM-17]
claim_id = ZERO_SMOKE_ANALYTICS_REQUESTS
claim = Production verification completed without synthetic analytics smoke traffic
requirement_class = GOVERNANCE_REQUIREMENT
verification_status = PASS
behavior_status = NORMAL
closure_impact = BLOCKING
evidence_basis = DATABASE_STATE
evidence_reference = RELEASE_VERIFICATION_ANALYTICS_REQUESTS = 0; Prospective denominator preserved at 0
observed_at = 2026-10-09T02:45:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Prospective capture purity preserved.

[CLAIM-18]
claim_id = STALE_CLIENT_CAPABILITY_HANDSHAKE
claim = Automated protocol negotiating API contract capabilities between client and backend
requirement_class = FOLLOW_ON_ARCHITECTURE
verification_status = OUT_OF_SCOPE
behavior_status = NOT_APPLICABLE
closure_impact = INFORMATIONAL
evidence_basis = NONE
evidence_reference = None
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Formal client capability negotiation protocol deferred to future cross-version governance.

[CLAIM-19]
claim_id = RETIRED_CONTRACT_TOMBSTONE
claim = Explicit tombstone endpoints and telemetry reporting retired API routes
requirement_class = FOLLOW_ON_ARCHITECTURE
verification_status = OUT_OF_SCOPE
behavior_status = NOT_APPLICABLE
closure_impact = INFORMATIONAL
evidence_basis = NONE
evidence_reference = None
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Explicit tombstone endpoints for deprecated routes deferred to future API lifecycle framework.

[CLAIM-20]
claim_id = STALE_CLIENT_RETIREMENT_TELEMETRY
claim = Real-time observability tracking client cache staleness across versions
requirement_class = FOLLOW_ON_ARCHITECTURE
verification_status = OUT_OF_SCOPE
behavior_status = NOT_APPLICABLE
closure_impact = INFORMATIONAL
evidence_basis = NONE
evidence_reference = None
observed_at = 2026-10-09T02:00:00Z
release_sha = 1328547e6f7d347c57f11ad6e22a0ae5309e1abb
runtime_sha = da707da3a5295f9f502d076af817f32aaa7ab0ac
notes = Specialized telemetry tracking stale client cache evictions deferred to future observability milestone.
```

---

#### 3. Canonical Ledger Normalized Counts

```ini
BLOCKING_PASS =
  11
BLOCKING_FAIL =
  0
BLOCKING_UNVERIFIED =
  0
BLOCKING_NOT_CAPTURED =
  0
NON_BLOCKING_PASS =
  3
NON_BLOCKING_UNVERIFIED =
  1
NON_BLOCKING_NOT_CAPTURED =
  2
FOLLOW_ON_OUT_OF_SCOPE =
  3
SAFE_FAIL_CLOSED_PATHS =
  2
UNSAFE_PATHS =
  0
RELEASE_CLOSURE =
  PASS
RELEASE_LIFECYCLE =
  CLOSED / VERIFIED / PUSHED / DEPLOYED / PRODUCTION_VERIFIED / FROZEN
NEXT_AUTHORIZED_EVENT =
  PASSIVE_NATURAL_MANUAL_EXIT_OBSERVATION
```
