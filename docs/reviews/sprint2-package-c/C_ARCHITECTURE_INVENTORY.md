# ARX TERMINAL — SPRINT 2 / TRACK C
# CURRENT-STATE ARCHITECTURE INVENTORY: UNIVERSE MODELS & READ/WRITE PATHS

**Document Identifier:** `C_ARCHITECTURE_INVENTORY.md`  
**Governing Authority:** ARX Closure Model v2 (Track C Ownership Boundary)  
**Track Responsibility:** Universe Publication Architecture & Contract Readiness  
**Target Milestone:** Sprint 2 / Package C Architecture & Readiness Review  
**Repository Coordinates:** `daakara/finance` (`main` @ `f6c5755126059b2f227e28ce90ddaad43997884c`)  
**Status:** COMPLETE / CANONICAL AUDIT  

---

## 1. Executive Summary & Audit Scope

This document establishes the comprehensive, authoritative technical inventory of all existing universe models, stores, data paths, admission predicates, scanner interfaces, scheduling hooks, consumer APIs, and legacy fallbacks within the ARX Terminal platform.

As mandated by Track C governance:
- **Baseline Authority:** Package A reference identity registration is closed and qualified (`PACKAGE_A_PRODUCTION_EXECUTION_CLOSED_QUALIFIED`), establishing an authoritative reference registry of **133 internal securities and 133 listing associations** in `/root/analyst_dashboard/data/identity.db`.
- **Trading Universe Reality:** The active published trading universe in production remains at its **existing three-pilot baseline** (`AAPL`, `GOOG`, `GOOGL`) under namespace `ARX_US_EQUITIES` (or `ARX_US_EQUITIES_PILOT_3C`).
- **External Dependencies:** Package B market data ingestion has **NOT** been certified (`account_entitlement_status: VERIFIED_DEFICIENT_CREDENTIALS`).
- **Safety Boundaries:** Scanners and schedulers remain strictly dormant; Gate 12 remains locked; zero production database mutations or universe table writes are authorized.

---

## 2. Table Schemas Across Production & Offline Stores

The platform maintains four distinct functional data stores, each governed by specific persistence mechanisms and immutability controls:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ARX DATA STORES TOPOLOGY                           │
├─────────────────────┬──────────────────────┬────────────────────────────────────┤
│ Store Name          │ Physical Target      │ Authority & Lifecycle              │
├─────────────────────┼──────────────────────┼────────────────────────────────────┤
│ 1. Identity Store   │ identity.db          │ Reference Entity Master & Listings │
│ 2. Universe Store   │ .finance_universe... │ Evaluated Universe & Ledgers       │
│ 3. Scanner Store    │ .finance_scanner...  │ Fenced Scans & Snapshot Store      │
│ 4. Market Store     │ market.db / Postgres │ OHLCV Bars & Live Quotes           │
└─────────────────────┴──────────────────────┴────────────────────────────────────┘
```

### 2.1 Identity Store (`identity.db`)
- **Source Module:** `analyst_dashboard/universe/identity_persistence.py`
- **Physical Path (Production):** `/root/analyst_dashboard/data/identity.db` (Persistent Volume Mount `/root`)
- **Physical Path (Offline/Test):** Configurable via `SqliteIdentityRepository(db_path=...)` (defaults to in-memory or local test fixtures).
- **Core Tables & Schemas:**

```sql
-- Table 1: Internal Securities Master
CREATE TABLE internal_securities (
    internal_security_id TEXT PRIMARY KEY,          -- Opaque durable ID: ARX_SEC_<12_HEX>
    issuer_cik TEXT,                                -- SEC Central Index Key (10 digits)
    primary_symbol TEXT NOT NULL,                   -- Normalized canonical ticker symbol
    share_class TEXT,                               -- Distinguishes share classes (e.g., CLASS_A vs CLASS_C)
    created_at_utc TEXT NOT NULL,                   -- ISO 8601 UTC creation timestamp
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'RETIRED', 'SUSPENDED')),
    continuity_basis TEXT NOT NULL DEFAULT 'INTERNAL_LEDGER_REGISTERED',
    certifies_continuity INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);
CREATE INDEX idx_sec_cik ON internal_securities(issuer_cik);
CREATE INDEX idx_sec_primary_symbol ON internal_securities(primary_symbol);

-- Table 2: Listing Associations
CREATE TABLE listing_associations (
    association_id TEXT PRIMARY KEY,                -- Opaque durable ID: ASSOC_<12_HEX>
    internal_security_id TEXT NOT NULL REFERENCES internal_securities(internal_security_id) ON DELETE RESTRICT,
    symbol TEXT NOT NULL,                           -- Trading locator ticker symbol
    operating_mic TEXT NOT NULL,                    -- ISO 10383 Operating MIC (e.g., XNAS, XNYS)
    effective_from TEXT NOT NULL,                   -- Effective start date (ISO date)
    effective_to TEXT,                              -- Effective end date (NULL = currently active)
    source_reference TEXT NOT NULL DEFAULT '',     -- Attributable source citation
    is_disputed INTEGER NOT NULL DEFAULT 0,
    dispute_reason TEXT,
    created_at_utc TEXT NOT NULL,
    CHECK (effective_to IS NULL OR effective_to >= effective_from)
);
CREATE INDEX idx_assoc_sec_id ON listing_associations(internal_security_id);
CREATE INDEX idx_assoc_sym_mic ON listing_associations(symbol, operating_mic);
CREATE UNIQUE INDEX idx_active_assoc_unique ON listing_associations(symbol, operating_mic) WHERE effective_to IS NULL;

-- Table 3: Identity Evidence References
CREATE TABLE identity_evidence_references (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    internal_security_id TEXT REFERENCES internal_securities(internal_security_id) ON DELETE CASCADE,
    association_id TEXT REFERENCES listing_associations(association_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    asserted_value TEXT,
    evidence_nature TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    source_record_locator TEXT NOT NULL,
    source_content_hash TEXT,
    observation_timestamp_utc TEXT,
    transformation_rule TEXT,
    mapping_version TEXT DEFAULT '1.0.0',
    does_certify_claim INTEGER DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

-- Table 4: Applied Plan Audit Ledger
CREATE TABLE applied_plan_audit_ledger (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,                          -- e.g., PLAN_ADOPT_9A2E4E04E878
    plan_content_digest TEXT NOT NULL,             -- Cryptographic digest of reviewed plan
    target_state_digest_before TEXT NOT NULL,      -- SQLite state hash prior to transaction
    target_state_digest_after TEXT NOT NULL,       -- SQLite state hash following commit
    applied_at_utc TEXT NOT NULL,
    authorization_scope TEXT NOT NULL,
    authorized_by TEXT,
    entities_applied INTEGER NOT NULL,
    associations_applied INTEGER NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE INDEX idx_audit_plan_id ON applied_plan_audit_ledger(plan_id);
```

### 2.2 Universe Store (`.finance_universe_store.db`)
- **Source Module:** `analyst_dashboard/universe/store.py`
- **Physical Path:** `os.path.join(DATA_DIR, ".finance_universe_store.db")` where `DATA_DIR` resolves to `os.getenv("FINANCE_DATA_DIR", os.getenv("DATA_DIR", expanduser("~")))`.
- **Core Tables & Immutability Triggers:**

```sql
-- Table 1: Source Population Snapshots
CREATE TABLE universe_source_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    source_authority TEXT NOT NULL,
    as_of TEXT NOT NULL,
    count INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    securities_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Table 2: Universe Builds & Attestations
CREATE TABLE universe_builds (
    universe_build_id TEXT PRIMARY KEY,
    universe_id TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    source_population_authority TEXT NOT NULL,
    source_population_snapshot_id TEXT NOT NULL,
    source_population_as_of TEXT NOT NULL,
    source_population_count INTEGER NOT NULL,
    source_population_hash TEXT NOT NULL,
    eligibility_rule_version TEXT NOT NULL,
    universe_definition_hash TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    normalization_hash TEXT NOT NULL,
    eligibility_input_snapshot_id TEXT NOT NULL,
    eligibility_input_hash TEXT NOT NULL,
    eligible_count INTEGER NOT NULL,
    ineligible_count INTEGER NOT NULL,
    eligibility_unresolved_count INTEGER NOT NULL,
    data_ready_count INTEGER NOT NULL,
    data_unavailable_count INTEGER NOT NULL,
    data_unresolved_count INTEGER NOT NULL,
    scannable_count INTEGER NOT NULL,
    eligible_membership_hash TEXT NOT NULL,
    scannable_membership_hash TEXT NOT NULL,
    per_security_decision_hash TEXT NOT NULL,
    readiness_decision_hash TEXT NOT NULL,
    construction_status TEXT NOT NULL,
    publication_decision TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    implementation_release_sha TEXT NOT NULL,
    reconciliation_summary_json TEXT
);
CREATE INDEX idx_ubuild_pub ON universe_builds(universe_id, publication_decision);

-- Table 3: Per-Security Eligibility Ledger
CREATE TABLE universe_eligibility_ledger (
    universe_build_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    source_record_hash TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    normalized_security_hash TEXT NOT NULL,
    normalization_status TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    eligibility_rule_version TEXT NOT NULL,
    universe_definition_hash TEXT NOT NULL,
    eligibility_decision TEXT NOT NULL,
    eligibility_reason_code TEXT NOT NULL,
    eligibility_input_hash TEXT NOT NULL,
    rule_evaluation_hash TEXT NOT NULL,
    decision_hash TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    implementation_release_sha TEXT NOT NULL,
    rule_results_json TEXT NOT NULL,
    PRIMARY KEY (universe_build_id, symbol)
);
CREATE INDEX idx_el_sym ON universe_eligibility_ledger(symbol);

-- Table 4: Per-Security Data Readiness Ledger
CREATE TABLE universe_readiness_ledger (
    universe_build_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    required_input_role TEXT NOT NULL,
    data_authority TEXT NOT NULL,
    data_as_of TEXT,
    freshness_status TEXT NOT NULL,
    completeness_status TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    readiness_result TEXT NOT NULL,
    readiness_reason_code TEXT NOT NULL,
    readiness_hash TEXT NOT NULL,
    candle_count INTEGER NOT NULL,
    has_live_price INTEGER NOT NULL,
    PRIMARY KEY (universe_build_id, symbol)
);

-- Immutability Triggers (Schema-Enforced Fail-Closed Controls)
CREATE TRIGGER trg_ubuild_no_update BEFORE UPDATE ON universe_builds
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be updated.');
END;

CREATE TRIGGER trg_ubuild_no_delete BEFORE DELETE ON universe_builds
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_builds records are immutable and cannot be deleted.');
END;

CREATE TRIGGER trg_eligibility_no_update BEFORE UPDATE ON universe_eligibility_ledger
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be updated.');
END;

CREATE TRIGGER trg_eligibility_no_delete BEFORE DELETE ON universe_eligibility_ledger
BEGIN
    SELECT RAISE(FAIL, 'CANONICAL_INVARIANT_VIOLATION: universe_eligibility_ledger records are immutable and cannot be deleted.');
END;
```

### 2.3 Scanner Store (`.finance_scanner_store.db`)
- **Source Module:** `analyst_dashboard/data/scanner_store.py` & `analyst_dashboard/coordination/store.py`
- **Physical Path:** `os.path.join(DATA_DIR, ".finance_scanner_store.db")`
- **Core Tables:**
  - `scanner_snapshots`: Immutable snapshot store containing version tuples, semantic fingerprints, results JSON, provenance, freshness, and publication decisions.
  - Coordination & Fencing Tables: `radar_jobs`, `radar_runs`, `radar_current_leases`, `radar_coordination_epochs`, `radar_fencing_sequence`, `radar_lease_events`.

---

## 3. Snapshot Generation Architecture

Snapshot generation is executed by `DeterministicUniverseBuilder` (`analyst_dashboard/universe/builder.py`):

```
                   Source Population (Directory Feeds / Master)
                                      │
                                      ▼
                       SourcePopulationSnapshot (Frozen)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   Eligibility Evaluation                          Data Readiness Evaluation
  (Rules R01-R07 Deterministic)                   (Market DB Live Price & 50 Bars)
              │                                               │
              ▼                                               ▼
   EligibilityLedgerRow                            DataReadinessLedgerRow
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                         UniverseBuildAttestation
                       (Merkle Hashes & Accounting)
                                      │
                                      ▼
                         UniverseStore (Atomic WAL)
```

### 3.1 Mathematical Accounting Closure Formulas
The builder enforces strict arithmetic equality before any build can be marked `COMPLETE` or `PUBLISH`:
$$\text{source\_population\_count} = \text{eligible\_count} + \text{ineligible\_count} + \text{eligibility\_unresolved\_count}$$
$$\text{eligible\_count} = \text{data\_ready\_count} + \text{data\_unavailable\_count} + \text{data\_unresolved\_count}$$
$$\text{scannable\_count} = \text{data\_ready\_count}$$

If any arithmetic identity fails, construction status evaluates to `FAILED` and publication decision to `QUARANTINE`.

---

## 4. Rule R06 Admission Predicates & Eligibility Ruleset

The canonical eligibility ruleset is defined in `analyst_dashboard/universe/contracts.py` lines 60-96 under version `eligibility-rule-v1.0`:

| Rule ID | Field Inspected | Operator | Expected Value | Failure Reason Code | Description |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **R01_ASSET_CLASS** | `asset_class` | `==` | `"EQUITY"` | `INELIGIBLE_ASSET_CLASS` | Excludes ETFs, commodities, bonds, and crypto pairs |
| **R02_SECURITY_TYPE** | `security_type` | `==` | `"COMMON_STOCK"` | `INELIGIBLE_SECURITY_TYPE` | Excludes ADRs, REITs, Warrants, Preferreds, and Units |
| **R03_LISTING_STATUS**| `listing_status`| `==` | `"ACTIVE"` | `INELIGIBLE_LISTING_STATUS` | Excludes suspended, halted, or defunct listings |
| **R04_PRIMARY_EXCHANGE**| `exchange` | `IN` | `["NASDAQ", "NYSE", "ARCA", "BATS"]` | `INELIGIBLE_EXCHANGE` | Restricts to recognized major US exchanges |
| **R05_CURRENCY** | `currency` | `==` | `"USD"` | `INELIGIBLE_CURRENCY` | Requires trading currency to be USD |
| **R06_PRIMARY_LISTING**| `primary_listing` | `==` | `True` | `INELIGIBLE_SECONDARY_LISTING` | Enforces primary operating listing; rejects cross/secondary listings |
| **R07_NOT_DELISTED** | `delisting_date` | `IS_NONE` | `None` / `""` | `INELIGIBLE_DELISTED` | Excludes delisted securities with termination dates |

### Rule R06 Specifics
In Track C, **Rule R06 eligibility remains unchanged**. In the identity layer (`analyst_dashboard/universe/admission_mapping.py`), primary listing status is independently established from primary exchange directory filings (SEC EDGAR, Nasdaq Listed Directory, Other Listed Directory). Where listing evidence is contradictory or absent, primary listing evaluates to `None` / `UNKNOWN`, preventing speculative admission.

---

## 5. Scanner Inputs & Scheduling Dependencies

### 5.1 VCP Scanner Runner (`analyst_dashboard/analyzers/scanner_runner.py`)
- **Execution Flow (`execute_market_wide_scan`):**
  1. Resolves scannable membership from `UniverseStore.get_scannable_membership(target_build_id)` where `readiness_result == 'DATA_READY'`.
  2. If `UniverseStore` has no published builds, falls back to `CANONICAL_VCP_UNIVERSE = HISTORICAL_VCP_35_REGRESSION_FIXTURE`.
  3. Executes multi-process or sequential evaluation against `MarketDatabaseEngine`.
  4. Requires $\ge 50$ daily bars and live current price $> 0$.
  5. Applies Minervini VCP 3-stage contraction detection and Stage 2 advancing filter.
  6. Calculates multi-factor confluence conviction score (0-100).
  7. Publishes immutable snapshot via `FencedPublisher` and `DurableRunCoordinator`.

### 5.2 Schedulers & Natural Triggers
- **VCP Natural Trigger (`analyst_dashboard/vcp/natural_trigger.py`):**
  - Controlled by environment flag `VCP_NATURAL_SCHEDULER_ENABLED` (default: `"0"`, disabled).
  - Protected by dedicated secret `VCP_SCHEDULER_SECRET`.
  - When dormant, no automatic cron or background scanning occurs.
- **Smart Money Scheduler (`analyst_dashboard/smart_money/smart_money_scheduler.py`):**
  - Background worker thread with lease recovery; currently dormant in production.

---

## 6. Consumer APIs & Caching Behavior

| Endpoint | Method | Source Path | Upstream Engine | Auth Required | Cache Semantics |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `/api/v1/screener/vcp/snapshot` | GET | `api/routes/screener.py` | `VCPScannerRunner` | No | In-memory `_cached_active_snapshot` + Edge CDN |
| `/api/v1/screener/smart-money/snapshot` | GET | `api/routes/screener.py` | `SmartMoneyScannerRunner`| No | Fast-path indexed snapshot ID |
| `/api/v1/screener/vcp/scan` | POST | `api/routes/screener.py` | `VCPScannerRunner` | `X-Operator-Token` | None (Mutates store) |
| `/api/v1/screener/vcp/natural-scan` | POST | `api/routes/screener.py` | `VCPScannerRunner` | `VCP_SCHEDULER_SECRET`| None (Scheduled scan) |
| `/api/v1/screener/vcp/coordination` | GET | `api/routes/screener.py` | `DurableRunCoordinator`| No | Dynamic query from SQLite |
| `/api/v1/radar` | GET | Static Next.js export | Frontend static data | No | Cloudflare Edge Cache / Static HTML |

---

## 7. Legacy Snapshot Fallbacks & Anti-Regression Controls

In PR #29 (`c5c151631e41494195b38eb0de38427421c3c9ba`, commit `fc3463e`), the platform instituted affirmative classification trust:
- Legacy snapshots stored under previous versions or missing affirmative setup/stage evidence fail closed:
  - `status_at_publication`: `"PIPELINE_PENDING"`
  - Reason code: `"UNSUPPORTED_LEGACY_MATCHES"`
- `VCPScannerRunner.get_active_or_latest_snapshot()` verifies that `persisted.get("implementation_release_sha") == CURRENT_IMPLEMENTATION_RELEASE_SHA`.
- If a hash mismatch or legacy schema is detected, the runner refuses to serve stale matches and executes a clean, affirmative market-wide scan or returns pending state.

---

## 8. Exact Source Paths & Existing Test Inventory

### 8.1 Authoritative Source Paths
- Universe Contracts & Types: `analyst_dashboard/universe/contracts.py`
- Universe Persistent Store: `analyst_dashboard/universe/store.py`
- Universe Builder Engine: `analyst_dashboard/universe/builder.py`
- Universe Historical Fixtures: `analyst_dashboard/universe/fixtures.py`
- Identity Persistence: `analyst_dashboard/universe/identity_persistence.py`
- Admission Mapping Engine: `analyst_dashboard/universe/admission_mapping.py`
- Identity Adoption Models: `analyst_dashboard/universe/identity_adoption.py`
- Scanner Store: `analyst_dashboard/data/scanner_store.py`
- Scanner Runner: `analyst_dashboard/analyzers/scanner_runner.py`
- Scanner Contracts: `analyst_dashboard/analyzers/scanner_contract.py`
- Publication Integrity: `analyst_dashboard/analyzers/scanner_publication_integrity.py`
- Screener Router: `api/routes/screener.py`

### 8.2 Existing Regression Test Suite
- `tests/test_golden_universe.py`: Tests deterministic universe builds, accounting closures, and invariant triggers.
- `tests/test_radar_universe_market_wide.py`: Tests market-wide scanner execution across scannable universe.
- `tests/test_scanner_truthfulness_remediation.py`: 400 lines validating affirmative classification trust and legacy fail-closed behaviors.
- `tests/test_bounded_150_candidate_expansion.py`: 1,808 lines validating the 150-candidate pool, 28 exclusions, and 133 qualifying launch universe.
- `tests/test_registration_execution_wrapper.py`: 884 lines validating atomic plan application, target state digests, and pre-apply backups.
- `tests/test_alpaca_historical_sip_ingestion.py`: 1,280 lines validating historical SIP session calendars, interior gap detection, and bar validation.
- `tests/test_package_a_staging_and_recovery_adversarial.py`: Adversarial tests for host-to-container staging, plan tampering rejection, and rollback durability.
