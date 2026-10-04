# ARX TERMINAL — SAAS FOUNDATION PHASE 1G PRODUCTION RELEASE EVIDENCE RECONCILIATION REPORT

## 0. Executive Reconciliation Summary

This report establishes the reconciled, evidence-grounded production status for the Phase 1G expand-only workspace persistence foundation, specifically resolving four critical claims:
1. **Production SQLite Persistence**: The primary backend configured in the production frontend bundle (`https://www.arxterminal.com`) is Railway (`https://web-production-e370b.up.railway.app`), where persistent ext4 volume `web-volume` is mounted at `/root` (`DATA_DIR=/root`, `PRODUCTION_DATABASE_PATH=/root/.finance_platform_history.db`). Physical persistence is actively verified via storage device ID distinctness (`dev st_dev` distinct from container root `/`). Render operates as a stateless secondary mirror.
2. **Pre-Deployment Backup & Recovery**: Prior to migration execution, no physical database snapshot file was independently exported or hashed. Reclassified honestly from `PASS` to `PRE_DEPLOYMENT_BACKUP = NOT_ESTABLISHED`, requiring explicit governance adjudication.
3. **Legacy Data Preservation**: Because no independent pre-migration table row counts were recorded in a pre-deployment artifact, zero row loss cannot be mathematically verified. Reclassified honestly to `DATA_PRESERVATION = NOT_FULLY_ADJUDICABLE` with `CONFIRMED_DATA_LOSS = NO`.
4. **Production Release Identity**: The diff between integration SHA `756674fb9d9e9dd4bd9709c6d00408f34dc02bcd` and current main `edfa436a03e112a6f88933c3ac51a04bc3f9b6c2` consists strictly of two documentation files. Classified as `EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR`.

---

## 1. Predecessor Integration State

The Phase 1G production release proceeds strictly from the verified integration predecessor:

```ini
PREDECESSOR_GATE =
  PASS_ARX_SAAS_FOUNDATION_PHASE_1G_INTEGRATION

PHASE_1G_INTEGRATION =
  VERIFIED_AND_FROZEN

SOURCE_RELEASE_SHA =
  98ee62f15fcf4f79c7823ae798e53536dd8ad350

MAIN_INTEGRATION_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

REMOTE_MAIN =
  VERIFIED_IN_SYNC

MIGRATION_STRATEGY =
  EXPAND_CONTRACT

CURRENT_MIGRATION_PHASE =
  EXPAND

CONTRACT_PHASE_AUTHORIZED =
  NO
```

---

## 2. Production Release Identity Classification

The relationship between the live running backend, the integrated release SHA, and the documentation commits on `main` is reconciled as follows:

```ini
INTEGRATION_RUNTIME_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

CURRENT_MAIN_SHA =
  edfa436a03e112a6f88933c3ac51a04bc3f9b6c2

CURRENT_MAIN_RELATION_TO_RUNTIME =
  DOCUMENTATION_ONLY_SUCCESSOR

PRODUCTION_RELEASE_IDENTITY =
  EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR
```

### Git Diff Attestation (`756674f..edfa436`)
```text
A docs/architecture/ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_MANIFEST.json
A docs/architecture/ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_REPORT.md
2 files changed, 775 insertions(+)
```
Zero code files, zero schema migrations, and zero configuration files were modified between `756674f` and `edfa436`. The runtime behavior deployed on Railway and Render is 100% equivalent to integration commit `756674f`.

---

## 3. Deployment Topology & Storage Durability

Investigation of the live Cloudflare Pages client bundle (`https://www.arxterminal.com`) established the authoritative backend routing:

```text
Client Target: https://web-production-e370b.up.railway.app
```

| Component | Platform / Host | Production URL / Location | Storage Durability Class |
|---|---|---|---|
| **Frontend Shell** | Cloudflare Pages | `https://www.arxterminal.com` (`finance-xp8.pages.dev`) | Edge SSG / Static Assets |
| **Primary Backend API** | Railway (`tranquil-radiance`) | `https://web-production-e370b.up.railway.app` | **PERSISTENT_DISK** (`web-volume` on `/root`) |
| **Secondary Backend API**| Render (`finance-backend-api`) | `https://finance-backend-api-qis0.onrender.com` | **EPHEMERAL** (Stateless Mirror) |
| **Primary Database Path**| Railway Container Mount | `/root/.finance_platform_history.db` | Persistent Volume Subpath |
| **Database Engine** | SQLite 3 | WAL mode (`PRAGMA journal_mode = WAL`) | ACID Compliant |

### Storage Durability Attributes
```ini
PRODUCTION_DATABASE_PATH =
  /root/.finance_platform_history.db

DATA_DIR =
  /root

DATABASE_ENGINE =
  SQLite

DATABASE_JOURNAL_MODE =
  WAL

DATABASE_STORAGE_CLASS =
  PERSISTENT_DISK

DATABASE_VOLUME_PERSISTENT =
  YES

VOLUME_MOUNT_PATH =
  /root

DATABASE_PATH_WITHIN_PERSISTENT_VOLUME =
  YES
```
Physical persistence is verified on Railway via `analyst_dashboard/governance/storage.py`, which validates that the persistent volume mount device ID (`st_dev`) is distinct from the root container overlayfs (`/`).

---

## 4. Backup & Recovery Boundary Reconciliation

Re-attestation of pre-deployment recovery assets establishes that no physical database snapshot file was independently exported or hashed prior to migration execution:

```ini
BACKUP_AVAILABLE =
  NO

BACKUP_LOCATION =
  NOT_ESTABLISHED

BACKUP_CREATED_AT =
  NOT_ESTABLISHED

BACKUP_FILE_SIZE =
  NOT_ESTABLISHED

BACKUP_SHA256 =
  NOT_ESTABLISHED

BACKUP_SOURCE_DATABASE =
  NOT_ESTABLISHED

BACKUP_EXTERNAL_TO_EPHEMERAL_RUNTIME =
  NO

PRE_DEPLOYMENT_BACKUP =
  NOT_ESTABLISHED

CURRENT_RECOVERY_BACKUP =
  NOT_ESTABLISHED

RECOVERY_READINESS =
  GOVERNANCE_ADJUDICATION_REQUIRED
```

Per Section 5 of the governing protocol, historical reality is not rewritten. Rather than creating a post-hoc snapshot and retroactively labeling it "pre-deployment", `PRE_DEPLOYMENT_BACKUP` is recorded honestly as `NOT_ESTABLISHED`.

---

## 5. Migration Execution & Schema Verification

The Phase 1G migration executes additively and idempotently:

```ini
MIGRATION_STRATEGY =
  EXPAND_CONTRACT

CURRENT_MIGRATION_PHASE =
  EXPAND

CONTRACT_PHASE_APPLIED =
  NO

WORKSPACE_ID_NOT_NULL =
  NO

LEGACY_USER_ID_REMOVED =
  NO

LEGACY_USER_ID_RENAMED =
  NO

DESTRUCTIVE_SCHEMA_CHANGE =
  NO

UNAUTHORIZED_SCHEMA_CHANGE =
  NO
```

### Table & Column Status
- `workspaces`: Present (`workspace_id PRIMARY KEY`, `name`, `created_at`, `updated_at`).
- `workspace_memberships`: Present (`id PRIMARY KEY`, `workspace_id`, `user_id`, `role`, `UNIQUE(workspace_id, user_id)`).
- `portfolio_holdings.workspace_id`: Present, NULLABLE.
- `user_trade_journal.workspace_id`: Present, NULLABLE.
- `user_cockpit_actions.workspace_id`: Present, NULLABLE.
- `user_profiles`: Preserved as `ACTOR_PROFILE` (`workspace_id` strictly absent).

---

## 6. Legacy Data Preservation Re-Adjudication

In accordance with Section 7 and Section 9:
- Historical pre-migration row counts were not independently captured in a frozen pre-deployment artifact.
- Current live API responses confirm schema integrity and absence of runtime errors.
- However, absence of observed corruption cannot be treated as mathematical proof of zero row loss.

```ini
PRE_MIGRATION_PORTFOLIO_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_JOURNAL_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_COCKPIT_ACTION_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_USER_PROFILE_ROWS =
  NOT_ESTABLISHED

DATA_PRESERVATION =
  NOT_FULLY_ADJUDICABLE

CONFIRMED_DATA_LOSS =
  NO

HISTORICAL_ZERO_ROW_LOSS =
  NOT_FULLY_ADJUDICABLE
```

---

## 7. Backfill & Membership Safety

Querying aggregate contracts and live API responses confirms strict adherence to backfill invariants:

```ini
ROWS_WITH_NULL_EMPTY_LEGACY_USER_AND_WS_DEFAULT =
  0

WS_DEFAULT_PRIVATE_ROWS =
  0

GUESSED_WORKSPACE_ASSIGNMENTS =
  0

WS_DEFAULT_MEMBERSHIPS =
  0

NULL_USER_MEMBERSHIPS =
  0

EMPTY_USER_MEMBERSHIPS =
  0
```

- Records without valid `user_id` remain `workspace_id = NULL`.
- Zero private data is owned by or backfilled into `ws_default`.

---

## 8. Live Production Behavioral Probes

Probes against both the primary Railway production backend and the secondary Render mirror confirm live enforcement of Phase 1G boundaries:

### A. Non-Persistent `ws_default` Enforcement (`INV_SAAS_07`)
```http
POST /api/v1/portfolio HTTP/1.1
Host: web-production-e370b.up.railway.app
X-User-Id: usr_test
X-Workspace-Id: ws_default
Content-Type: application/json

{"symbol":"AAPL","shares":10,"entryPrice":150.0,"name":"Apple Inc"}
```
**Railway Response**:
```http
HTTP/1.1 403 Forbidden
Content-Type: application/json
Cache-Control: private, no-cache, no-store, must-revalidate
Pragma: no-cache

{"detail":"Private persistence operations require an authenticated or actor-bound workspace. Shared 'ws_default' cannot own persistent data (INV-SAAS-07)."}
```

### B. Cross-Workspace Isolation (`INV_SAAS_03`)
```http
GET /api/v1/portfolio HTTP/1.1
Host: web-production-e370b.up.railway.app
X-User-Id: usr_test
X-Workspace-Id: ws_unauthorized_999
```
**Railway Response**:
```http
HTTP/1.1 403 Forbidden
Content-Type: application/json
Cache-Control: private, no-cache, no-store, must-revalidate
Pragma: no-cache

{"detail":"Actor 'usr_test' is not authorized to access workspace 'ws_unauthorized_999'."}
```

### C. Private Cache Boundary (`INV_SAAS_02`)
- `GET /api/v1/portfolio`: `HTTP 200 []` with `Cache-Control: private, no-cache, no-store, must-revalidate` and `Pragma: no-cache`.
- `GET /api/v1/journal/trades?user_id=usr_demo`: `HTTP 200 []` with `Cache-Control: private, no-cache, no-store, must-revalidate` and `Pragma: no-cache`.
- `GET /api/v1/cockpit/state?user_id=usr_demo`: `HTTP 200` with strict private cache and `Vary` headers.

### D. Public Route Context-Free Independence (`INV_SAAS_05`)
- `GET /api/v1/regimes/current`: `HTTP 200` with `Cache-Control: public, max-age=60, s-maxage=300, stale-while-revalidate=86400`. Zero request context or workspace dependency.
- `POST /api/v1/screener/run`: `HTTP 200` with `Cache-Control: public, max-age=30, s-maxage=120, stale-while-revalidate=86400, stale-if-error=86400`.
- `GET /api/v1/etf/profile/SPY`: `HTTP 200` with `Cache-Control: public, max-age=3600, s-maxage=3600, stale-while-revalidate=86400`.

---

## 9. Invariants Ledger (INV-SAAS-01 through INV-SAAS-07)

| Invariant | Title | Requirement | Production Status |
|---|---|---|---|
| **INV-SAAS-01** | Quant Purity | Zero tenancy/commercial concerns in quant engine | **PRESERVED** |
| **INV-SAAS-02** | Private Cache Boundary | Private non-shared cache headers on user/private data | **PRESERVED** |
| **INV-SAAS-03** | Workspace Isolation | Workspace-bounded queries and authorization checks | **ENFORCED** |
| **INV-SAAS-04** | Dual-Read / Dual-Write | Transitional compatibility across expand phase | **PRESERVED** |
| **INV-SAAS-05** | Public Independence | Public routes operate context-free without workspace lookups | **PRESERVED** |
| **INV-SAAS-06** | Deterministic Compatibility | SHA-256 derived compatibility workspaces for legacy users | **ENFORCED** |
| **INV-SAAS-07** | `ws_default` Non-Persistence | `ws_default` prohibited from owning private persistent data | **ENFORCED** |

---

## 10. Commercial Boundary & Concurrent Tracks

- `AUTHENTICATION_IMPLEMENTED = NO`
- `SUBSCRIPTIONS_IMPLEMENTED = NO`
- `BILLING_IMPLEMENTED = NO`
- `STRIPE_IMPLEMENTED = NO`
- `PRICING_IMPLEMENTED = NO`
- Zero regressions across ARX Radar, Tactical Setups, ETF V2, and OpenFIGI.

---

## 11. Reconciled Acceptance Matrix Evaluation

| ID | Description | Reconciled Result | Evidence / Adjudication |
|---|---|---|---|
| **SAAS-1G-PROD01** | Integration predecessor PASS | **PASS** | `PASS_ARX_SAAS_FOUNDATION_PHASE_1G_INTEGRATION` verified |
| **SAAS-1G-PROD02** | Main release SHA re-attested | **PASS** | `756674f` integrated; main documentation successor at `edfa436` |
| **SAAS-1G-PROD03** | Production topology identified | **PASS** | Cloudflare Pages -> Railway primary backend (`web-volume`) |
| **SAAS-1G-PROD04** | Persistent production DB established | **PASS** | Persistent ext4 disk at `/root` verified via device ID attestation |
| **SAAS-1G-PROD05** | Pre-deployment backup verified | **NOT_ESTABLISHED** | No physical pre-deployment backup file was captured prior to migration |
| **SAAS-1G-PROD06** | Migration expand-only | **PASS** | Additive schema expansion only |
| **SAAS-1G-PROD07** | Contract migration absent | **PASS** | `workspace_id` remains nullable; `user_id` retained |
| **SAAS-1G-PROD08** | Workspace tables created | **PASS** | `workspaces` and `workspace_memberships` initialized |
| **SAAS-1G-PROD09** | Only authorized `workspace_id` columns added | **PASS** | `portfolio_holdings`, `user_trade_journal`, `user_cockpit_actions` only |
| **SAAS-1G-PROD10** | Legacy `user_id` retained | **PASS** | All legacy columns, constraints, and indexes preserved |
| **SAAS-1G-PROD11** | Legacy data preserved | **NOT_FULLY_ADJUDICABLE** | Historical pre-migration row counts were not independently captured |
| **SAAS-1G-PROD12** | Deterministic backfill verified | **PASS** | `derive_compatibility_workspace_id` verified |
| **SAAS-1G-PROD13** | `ws_default` private persistence prohibited | **PASS** | Live probe returns HTTP 403 with INV-SAAS-07 detail |
| **SAAS-1G-PROD14** | Actor-bound private persistence preserved | **PASS** | Private application services require bound workspace |
| **SAAS-1G-PROD15** | INV-SAAS-01 preserved | **PASS** | Quant engine remains 100% pure |
| **SAAS-1G-PROD16** | INV-SAAS-02 preserved | **PASS** | Strict private cache headers verified live on all private endpoints |
| **SAAS-1G-PROD17** | INV-SAAS-03 enforced | **PASS** | Unauthorized workspace access returns HTTP 403 live |
| **SAAS-1G-PROD18** | INV-SAAS-04 preserved | **PASS** | Dual-read and dual-write transitional parity verified |
| **SAAS-1G-PROD19** | INV-SAAS-05 preserved | **PASS** | Public routes operate context-free with public cache headers |
| **SAAS-1G-PROD20** | INV-SAAS-06 enforced | **PASS** | Deterministic SHA-256 compatibility workspace verified |
| **SAAS-1G-PROD21** | INV-SAAS-07 enforced | **PASS** | `ws_default` prohibited from owning private persistent data |
| **SAAS-1G-PROD22** | Private cache boundary preserved | **PASS** | `private, no-cache, no-store, must-revalidate` on all private responses |
| **SAAS-1G-PROD23** | Public routes context-free | **PASS** | `GET /api/v1/regimes/current` operates without workspace context |
| **SAAS-1G-PROD24** | Authentication absent | **PASS** | Zero auth / login / signup code added |
| **SAAS-1G-PROD25** | Subscriptions absent | **PASS** | Zero subscription management code added |
| **SAAS-1G-PROD26** | Billing absent | **PASS** | Zero Stripe or billing logic added |
| **SAAS-1G-PROD27** | Production release identity established | **PASS** | `EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR` |
| **SAAS-1G-PROD28** | Frontend/backend available | **PASS** | HTTP 200 verified on Cloudflare Pages and Railway API |
| **SAAS-1G-PROD29** | No confirmed Radar regression | **PASS** | Radar routes and APIs operating normally |
| **SAAS-1G-PROD30** | No confirmed Tactical regression | **PASS** | Setups routes and screener APIs operating normally |
| **SAAS-1G-PROD31** | No confirmed ETF/OpenFIGI regression | **PASS** | ETF profile and OpenFIGI contracts operating normally |
| **SAAS-1G-PROD32** | No confirmed migration defect | **PASS** | Additive schema and backfill applied cleanly |
| **SAAS-1G-PROD33** | No confirmed data-loss defect | **NOT_FULLY_ADJUDICABLE** | Absence of observed corruption is not proof of zero historical data loss |
| **SAAS-1G-PROD34** | Production manifest generated and hashed | **PASS** | Manifest SHA-256: `983b5624d835c18c1906562a608c0c9cd372920c1498c48ab9ed20b28dd34fdb` |
| **SAAS-1G-PROD35** | No unsupported production-evidence claims | **PASS** | All claims reconciled without fabricated historical artifacts |

---

## 12. Cryptographic Attestation

- **Manifest Path**: `docs/architecture/ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_MANIFEST.json`
- **Algorithm**: `SHA-256`
- **Manifest SHA-256**: `983b5624d835c18c1906562a608c0c9cd372920c1498c48ab9ed20b28dd34fdb`

---

## 13. Reconciled Gate Verdict

Per Section 17 of the governing reconciliation protocol, because runtime safety, storage persistence, and schema integrity are proven live, but historical pre-deployment backup and table counts were never independently recorded, the evidence is honestly classified without retroactive fabrication:

```ini
GATE =
  HOLD_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_RECONCILIATION

RUNTIME_PRODUCTION_HEALTH =
  VERIFIED

CURRENT_DATABASE_INTEGRITY =
  VERIFIED

PRE_DEPLOYMENT_BACKUP =
  NOT_ESTABLISHED

HISTORICAL_ZERO_ROW_LOSS =
  NOT_FULLY_ADJUDICABLE

CONFIRMED_DATA_LOSS =
  NO

CONFIRMED_PRODUCTION_DEFECT =
  NO

NEXT_ACTION =
  GOVERNANCE_ADJUDICATION_OF_MISSING_HISTORICAL_EVIDENCE

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```
