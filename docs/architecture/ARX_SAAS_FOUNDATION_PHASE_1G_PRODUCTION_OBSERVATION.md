# ARX TERMINAL — SAAS FOUNDATION PHASE 1G PRODUCTION OBSERVATION REPORT

## 0. Frozen Predecessor State

This observation epoch proceeds strictly from the qualified production release:

```ini
PREDECESSOR_GATE =
  PASS_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_QUALIFIED

PHASE_1G_PRODUCTION =
  VERIFIED_WITH_HISTORICAL_EVIDENCE_LIMITATIONS

INTEGRATION_RUNTIME_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

CURRENT_HEAD_SHA =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

ORIGIN_MAIN_SHA =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

REMOTE_MAIN_SHA =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

CURRENT_MAIN_RELATION_TO_RUNTIME =
  DOCUMENTATION_ONLY_SUCCESSOR

PRODUCTION_RELEASE_IDENTITY =
  EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR

DATABASE_VOLUME_PERSISTENT =
  YES

PRODUCTION_DATABASE_PATH =
  /root/.finance_platform_history.db

PRE_DEPLOYMENT_BACKUP =
  NOT_ESTABLISHED

HISTORICAL_ZERO_ROW_LOSS =
  NOT_FULLY_ADJUDICABLE

HISTORICAL_EVIDENCE_LIMITATION_ACCEPTED =
  YES

CONFIRMED_DATA_LOSS =
  NO

CONFIRMED_PRODUCTION_DEFECT =
  NO

INV_SAAS_01 =
  PRESERVED

INV_SAAS_02 =
  PRESERVED

INV_SAAS_03 =
  ENFORCED

INV_SAAS_04 =
  PRESERVED

INV_SAAS_05 =
  PRESERVED

INV_SAAS_06 =
  ENFORCED

INV_SAAS_07 =
  ENFORCED

AUTHENTICATION_IMPLEMENTED =
  NO

SUBSCRIPTIONS_IMPLEMENTED =
  NO

BILLING_IMPLEMENTED =
  NO

CONTRACT_PHASE =
  NOT_AUTHORIZED
```

---

## 1. Observation Objective & Scope

This gate formally maintains the **passive natural production observation epoch** for the deployed Phase 1G expand-only workspace persistence foundation.

### Monitored Invariants & Behaviors
- Actor-bound workspace resolution correctness (`INV-SAAS-01`, `INV-SAAS-06`).
- Cross-workspace authorization isolation (`INV-SAAS-03`).
- Private persistence reliability on Railway persistent storage (`web-volume`).
- Prohibition of private persistence under shared `ws_default` (`INV-SAAS-07`).
- Dual-read / dual-write transitional parity (`INV-SAAS-04`).
- Private cache non-shared header enforcement (`INV-SAAS-02`).
- Public route context-free independence (`INV-SAAS-05`).
- Database durability and absence of SQLite operational errors.

### Strict Scope Exclusions
The observation epoch strictly prohibits:
- Authentication, signup, or login rollout.
- Subscriptions, billing, Stripe, or pricing logic.
- Workspace / team management UI.
- Contract-phase schema migration (`workspace_id NOT NULL`, removal of legacy `user_id`).
- Phase 1H execution.

---

## 2. Observation Epoch Configuration & Release Identity

```ini
OBSERVATION_EPOCH =
  ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_EPOCH_1

OBSERVATION_MODE =
  PASSIVE_NATURAL_PRODUCTION_ONLY

EPOCH_START_UTC =
  2026-10-04T20:25:00Z

CURRENT_HEAD =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

ORIGIN_MAIN =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

REMOTE_MAIN =
  8a75a0718ce3fc423fa22b927bfe4ded0159f308

FRONTEND_PRODUCTION_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

BACKEND_PRODUCTION_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

PHASE_1G_RUNTIME_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

RELEASE_DRIFT =
  NO
```

*Verification Rationale*: Zero diff exists across `analyst_dashboard/data/db_engine.py`, `api/capabilities/`, `api/context/`, `api/routes/cockpit.py`, `api/routes/journal.py`, `api/routes/portfolio.py`, `api/services/`, `database/migrations/002*`, `database/models.py`, `database/workspace*`, and `frontend/lib/saas/` between release commit `756674fb9d9e9dd4bd9709c6d00408f34dc02bcd` and `HEAD`. Successors are strictly documentation updates.

---

## 3. Re-Attestation of Frozen Phase 1G Authorities

The established production contract is re-attested and preserved:

```ini
WORKSPACE_AUTHORITY =
  SERVER_CONFIRMED

MEMBERSHIP_AUTHORITY =
  SERVER_CONFIRMED

WORKSPACE_ISOLATION =
  ENFORCED_SERVER_SIDE

PRIVATE_CONTEXT_CACHE =
  PRIVATE_NO_STORE

PUBLIC_CONTEXT_CACHE =
  PUBLIC_ONLY

DEFAULT_WORKSPACE_PRIVATE_MUTATION =
  FORBIDDEN

CROSS_WORKSPACE_UNAUTHORIZED_ACCESS =
  FORBIDDEN
```

All 92 SaaS foundation and invariant tests pass cleanly in local verification suite (`92 passed in 7.84s`).

---

## 4. Denominator Definition & Observability Audit

### 4.1 Frozen Denominator Definition
- **Denominator Primary Source**: `RAILWAY_CONTAINER_ACCESS_LOGS_ARX_API`
- **Denominator Event / Record**: `http_request_completed` (routes: `/api/v1/portfolio`, `/api/v1/cockpit/state`, `/api/v1/cockpit/profile`, `/api/v1/cockpit/actions`, `/api/v1/journal/trades`, `/api/v1/journal/telemetry`)
- **Denominator Identity Rule**: `AUTHORITATIVE_PRIVATE_CONTEXT_REQUEST_RESOLVER`
- **Denominator Time Boundary**: `>= 2026-10-04T20:25:00Z`
- **Denominator Deduplication Rule**: `EXCLUDE_SAME_SESSION_REHYDRATIONS_WITHIN_WINDOW`
- **Retry Double-Counting**: `NO`
- **Rehydration Double-Counting**: `NO`
- **Polling Double-Counting**: `NO`

### 4.2 Observability Determination
```ini
DENOMINATOR_OBSERVABILITY =
  VERIFIED
```

*Verification Rationale*:
Railway access logs capture every incoming HTTP request completion on private routes in structured JSON format (`event_name="http_request_completed"`, `route`, `method`, `status_code`, `request_id`, `correlation_id`, `duration_ms`, `backend_release_sha`). The logging infrastructure reliably differentiates natural browser sessions (preceded by browser CORS `OPTIONS` preflight, accompanied by UI resource queries) from direct operator curl diagnostic probes.

---

## 5. Traffic Accounting & Deduplication

All traffic recorded since epoch start (`2026-10-04T20:25:00Z`) is accounted and classified:

### 5.1 Reconciled Natural Traffic ($N=3$)
Observed natural user session on `2026-10-05` between `19:43:07Z` and `19:43:17Z`:
1. `2026-10-05T19:43:07.630558Z`: `GET /api/v1/portfolio` (200 OK, 3.37ms, `request_id="cc6ee945-4d1e-442c-9bcb-c98cb009279b"`, browser origin with `OPTIONS` preflight) → **NATURAL** ($N=1$).
2. `2026-10-05T19:43:14.734431Z`: `GET /api/v1/journal/telemetry` (200 OK, 26.27ms, `request_id="45402..."`) → **NATURAL** ($N=1$).
3. `2026-10-05T19:43:14.817575Z`: `GET /api/v1/cockpit/state` (200 OK, 5.67ms, `request_id="35078..."`, preceded by `OPTIONS /api/v1/cockpit/state`) → **NATURAL** ($N=1$).
4. `2026-10-05T19:43:17.763240Z`: `GET /api/v1/portfolio` (200 OK, 2.15ms, `request_id="dd5daa19..."`) → Deduplicated session re-fetch/rehydration ($N=0$ additional).

### 5.2 Excluded Traffic
- `EXCLUDED_OPERATOR_DIAGNOSTICS` = `1` (`2026-10-05T19:30:54.402019Z`, operator curl probe during release gate)
- `EXCLUDED_SYNTHETIC_REQUESTS` = `0`
- `EXCLUDED_TEST_TRAFFIC` = `0`
- `EXCLUDED_REPLAYS` = `0`
- `UNCLASSIFIED_REQUESTS` = `0`

```ini
NATURAL_PRIVATE_CONTEXT_REQUESTS =
  3

NATURAL_PRIVATE_WORKSPACE_LOADS =
  1

NATURAL_MEMBERSHIP_CONTEXT_LOADS =
  1

NATURAL_PORTFOLIO_OPERATIONS =
  1

NATURAL_JOURNAL_OPERATIONS =
  1

NATURAL_COCKPIT_OPERATIONS =
  1

NATURAL_PUBLIC_ROUTE_REQUESTS =
  1

SYNTHETIC_DENOMINATOR_INFLATION =
  NO

FORCED_TRAFFIC_USED =
  NO

TEST_TRAFFIC_COUNTED =
  NO

OPERATOR_TRAFFIC_COUNTED =
  NO
```

---

## 6. Checkpoint Progression & Eligibility Evaluation

### 6.1 Checkpoint 1 (Baseline)
- Timestamp: `2026-10-04T20:34:55Z`
- Elapsed: `0.165 hours` (< 24.0h) → NOT SATISFIED
- Natural Private Requests: `0` (< 10) → NOT SATISFIED
- Gate: `HOLD_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_OBSERVATION`

### 6.2 Checkpoint 2 (Eligibility-First)
- **Checkpoint Timestamp (UTC)**: `2026-10-05T20:30:35Z`
- **Checkpoint Timestamp (Local)**: `2026-10-05T22:30:35+02:00`
- **Elapsed Observation Time**: `24.09 hours` ($\ge 24.0$ hours threshold) → **SATISFIED**
- **Temporal Eligibility**: **SATISFIED**
- **Denominator Observability**: **VERIFIED**
- **Natural Private Context Requests**: `3` ($< 10$ required) → **NOT SATISFIED**
- **Cohort Eligibility**: **NOT_SATISFIED**
- **Overall Checkpoint Eligibility**: **NOT_SATISFIED**

---

## 7. Eligibility-First Stop Rule Application

Pursuant to Section 12:
- Because `CHECKPOINT_ELIGIBILITY = NOT_SATISFIED`, full observation matrix adjudication is **STRICTLY NOT AUTHORIZED**.
- Workspace isolation, cache isolation, membership correctness, and cross-user behavior are **NOT** claimed empirically PASS based on predecessor tests alone.
- Criteria dependent on full empirical traffic remain classified as `NOT_OBSERVED` or `INSUFFICIENT_EVIDENCE`.

| ID | Description | Checkpoint 2 Status | Rationale |
|---|---|:---:|---|
| **SAAS-1G-OBS01** | Release identity stable | **PASS** | Deployed release matches runtime SHA `756674f` (doc-only successors verified) |
| **SAAS-1G-OBS02** | Elapsed time >= 24h | **PASS** | Elapsed observation time is 24.09 hours ($\ge 24.0$h threshold met) |
| **SAAS-1G-OBS03** | Natural private denominator >= 10 | **INSUFFICIENT_EVIDENCE** | Denominator count is 3 ($< 10$ required) |
| **SAAS-1G-OBS04** | Workspace resolution correctness | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS05** | `ws_default` private persistence safety | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS06** | Cross-workspace isolation | **NOT_OBSERVED** | Multi-tenant cohort unauthorized pending cohort threshold |
| **SAAS-1G-OBS07** | Private persistence reliability | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS08** | Database durability | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS09** | Migration state remains expand-only | **PASS** | Expand-only invariant verified, zero contract DDL |
| **SAAS-1G-OBS10** | Dual-read / dual-write transitional parity | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS11** | Private cache isolation | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS12** | Public route context independence | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS13** | Runtime reliability | **NOT_OBSERVED** | Container error review: PASS |
| **SAAS-1G-OBS14** | Concurrent-track integrity | **NOT_OBSERVED** | Full empirical matrix unauthorized pending cohort threshold |
| **SAAS-1G-OBS15** | No synthetic denominator inflation | **PASS** | Zero synthetic or operator requests admitted into denominator |

---

## 8. Cryptographic Manifest Attestation

- **Manifest Path**: `docs/architecture/ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_OBSERVATION_MANIFEST.json`
- **Algorithm**: `SHA-256`
- **Manifest SHA-256**: `b55005123f1ee866625141dafd83d2befae898eb5c26ad83b08070ff1cbc935e`

---

## 9. Checkpoint 2 Formal Gate Verdict

Pursuant to Section 21 (`HOLD — Natural Cohort Below Threshold`):

```ini
GATE =
  HOLD_ARX_SAAS_PHASE_1G_PRODUCTION_OBSERVATION

CHECKPOINT_TYPE =
  ELIGIBILITY_FIRST

CHECKPOINT_NUMBER =
  2

CHECKPOINT_TIMESTAMP_UTC =
  2026-10-05T20:30:35Z

CHECKPOINT_TIMESTAMP_LOCAL =
  2026-10-05T22:30:35+02:00

PRIMARY_VERDICT =
  INSUFFICIENT_NATURAL_EVIDENCE

CHECKPOINT_STATE =
  HOLD

CHECKPOINT_ELIGIBILITY =
  NOT_SATISFIED

TEMPORAL_ELIGIBILITY =
  SATISFIED

COHORT_ELIGIBILITY =
  NOT_SATISFIED

DENOMINATOR_OBSERVABILITY =
  VERIFIED

ELAPSED_OBSERVATION_TIME_HOURS =
  24.09

NATURAL_PRIVATE_CONTEXT_REQUESTS =
  3

SYNTHETIC_DENOMINATOR_INFLATION =
  NO

CONFIRMED_PRIVACY_DEFECT =
  NO

CONFIRMED_RELEASE_DEFECT =
  NO

NEXT_ACTION =
  CONTINUE_PASSIVE_NATURAL_OBSERVATION

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 10. Mandatory Stop Conditions Enforced

In strict compliance with Section 28:
- Zero private-context traffic was manufactured.
- Zero synthetic workspace sessions were created.
- Zero manual refreshes were triggered to inflate $N$.
- Zero operator diagnostics ($N=1$) were admitted into the natural denominator.
- Phase 1G production code was not modified.
- No telemetry was added automatically.
- Automatic successor execution is strictly **NOT_AUTHORIZED**. All operations halt.
