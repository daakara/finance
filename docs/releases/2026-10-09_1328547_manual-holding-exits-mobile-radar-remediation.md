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
