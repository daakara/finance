# ARX Terminal — Smart Money Form 4 Market-Wide Pipeline Specification (V1)

**Release Candidate Lineage**: Candidate 003 (`arx/smart-money-form4-market-wide-rc-v3`)  
**Candidate 002 Baseline**: `714d47271c6dae9f92b52ce11570de2f13d8fd42` (immutable audit evidence)  
**Parent / Baseline**: `0b644698f46d14dd454425cb842b08e59b8dafcb`  
**Frozen RC Ancestor**: `9d0bbcf21673fd6a3428a6674713dd244ec948bc` (mechanically merged)  
**Governance State**: Fail-Closed Pre-Release (`PRODUCTION_DEPLOYMENT_AUTHORIZED: false`)

---

## 1. Executive Summary & Purpose

The ARX Terminal Smart Money Form 4 market-wide pipeline reconciles low-latency SEC EDGAR filing discovery with authoritative end-of-day master index reconciliation. It produces point-in-time-safe, completeness-certified insider accumulation snapshots for the ARX Terminal Market Opportunity Radar.

The pipeline transitions from historical issuer-partitioned discovery to an accession-driven global market architecture governed by six fail-closed release gates.

---

## 2. End-to-End Architectural Topology

```
SEC EDGAR PUBLIC DISCLOSURE REPOSITORIES
    │
    ├── Low-Latency Fast Path: SEC EDGAR Submissions Atom Feed (`cgi-bin/browse-edgar?action=getcurrent`)
    │   └── Coverage Status: INTRADAY_FAST_PATH / PARTIAL
    │
    └── Completeness Reconciliation: SEC Daily Master Indexes (`edgar/daily-index/{year}/{qtr}/master.{date}.idx`)
        └── Coverage Status: RECONCILED / COMPLETE (upon zero unresolved gaps)
    │
    ▼
ACCESSION-LEVEL IDEMPOTENT INGESTION (`ingest_accession`)
    │   • Canonical Identity: SEC Accession Number (e.g. 0001045810-26-000088)
    │   • Content Integrity: SHA-256 fingerprint comparison (conflicts raise SOURCE_INTEGRITY_CONFLICT)
    │   • Strict XML Target Lineage: Form 4/A targets matched explicitly or isolated with AMENDMENT_TARGET_AMBIGUOUS
    │
    ▼
PERSISTENCE LAYER (Canonical Connection Factory: `connect_smart_money_db`)
    │   • SQLite PRAGMAs: journal_mode = WAL, busy_timeout = 5000ms, synchronous = NORMAL
    ├── `form4_raw_filings`: Unmodified XML payload, SHA-256, HTTP/SEC metadata
    ├── `form4_parsed_transactions`: Normalized non-derivative transactions
    ├── `form4_public_availability_ledger`: Point-in-time certified availability timestamps
    ├── `sm_source_gaps`: Unresolved discovery, fetch, or parse discrepancies
    └── `sm_reconciliation_reports`: Cryptographic audit receipts certifying completeness
    │
    ▼
COMPLETENESS RECONCILIATION ENGINE (`reconcile_interval` / `reconcile_daily_index`)
    │   • Availability-driven daily master index check (returns RECONCILIATION_PENDING if not yet published)
    │   • Mathematical set difference: E_index \ L_local = missing accessions
    │   • Evaluation: CoverageStatus: COMPLETE vs PARTIAL
    │   • Watermark Advance: Certified watermark W advances ONLY upon receipt with coverage_status = COMPLETE
    │
    ▼
CERTIFIED WATERMARK W
    │
    ▼
POINT-IN-TIME TRANSACTION EVALUATION (`scan_issuers_as_of(W)`)
    │   • Strict predicate: public_availability_at <= W
    │   • Zero backward temporal leakage of late filings or subsequent amendments
    │   • Receipt Verification: Validates receipt in `sm_reconciliation_reports`
    │
    ▼
FROZEN QUALIFICATION SCORING (`FORM4_RADAR_QUALIFICATION_V1`)
    │   • Predicate: Open-market purchase (code 'P') exists AND gross_purchase_value_usd >= $25,000.00
    │   • Arithmetic Conviction Model (100-Point Max, 40/30/20/10 distribution):
    │       - Purchase Value Factor: max 40 pts
    │       - Clustering Factor: max 30 pts
    │       - Role Factor (C-suite / Director): max 20 pts
    │       - Recency Factor (executed <= 7d/14d): max 10 pts
    │   • Decoupled Semantics: signalStatus = "QUALIFIED_INSIDER_ACCUMULATION", subsignal = "INSIDER_FORM4"
    │     (Execution Ladder status IN_BUY_ZONE strictly eliminated)
    │
    ▼
IMMUTABLE SNAPSHOT PUBLICATION (`ScannerSnapshotStore`)
    │   • Snapshot Table: `scanner_snapshots` protected by BEFORE UPDATE / BEFORE DELETE SQLite triggers
    │   • Content Hashing: Canonical JSON representation SHA-256 hashed into `semantic_fingerprint`
    │   • Atomic Active Pointer: Atomic UPDATE to `scanner_active_snapshots`
    │   • Publication Guard: Uncertified / partial snapshots quarantined from active pointer
    │
    ▼
RADAR READ-ONLY SERVING LAYER (`/api/screener`, `/api/screener/smart-money/snapshot`)
    • Zero live computation during request handling
    • Read-only query of active published snapshot
    • Explicit taxonomy separation: INSIDER_FORM4 distinguished from Institutional 13F and Dark Pool
```

---

## 3. The Six Frozen Engineering Gates

### Gate B1: SEC Global Filing Enumeration & Authoritative Reconciliation
- **Low-Latency Feed**: Polling `cgi-bin/browse-edgar?action=getcurrent` every interval. Extracts accession, CIK, filing date, and form type.
- **Reconciliation Feed**: Daily SEC `master.idx` processed through availability-driven verification. If index not yet published, status is `RECONCILIATION_PENDING`.
- **Form Type Scope**: Form 4 and Form 4/A. (Form 3 and Form 5 are preserved in raw store if observed, but excluded from Form 4 accumulation scoring).

### Gate B2: Point-in-Time Generation Safety
- No filing or transaction is eligible for generation $G$ unless `public_availability_at <= G.certified_as_of`.
- Trade date alone is strictly rejected as a temporal boundary.
- Late filers (filings accepted after $W$) are temporally isolated: they cannot retroactively mutate generations with watermark prior to their acceptance datetime.
- Idempotency: Duplicate accessions with identical content hashes return `KNOWN_IDEMPOTENT_NOOP`. Duplicate accessions with differing content hashes trigger `SOURCE_INTEGRITY_CONFLICT`.

### Gate B3: Prospective Radar Qualification Predicate & Conviction Model
- **Frozen Rule**: `FORM4_RADAR_QUALIFICATION_V1`
  - Condition: At least one transaction with `transaction_code == 'P'` AND `gross_purchase_value_usd >= 25000.00`.
- **Arithmetic Conviction Scoring (100-Point Max)**:
  - **1. Purchase Value Factor (max 40 pts)**:
    - $\ge \$1,000,000$: 40.0 pts
    - $\ge \$250,000$: 30.0 pts
    - $\ge \$50,000$: 20.0 pts
    - $<\$50,000$: 10.0 pts
  - **2. Clustering Factor (max 30 pts)**:
    - $\ge 2$ distinct insider buyers within lookback: 30.0 pts
    - Single insider buyer: 15.0 pts
  - **3. Role Factor (max 20 pts)**:
    - C-suite executive (CEO, CFO, President, Chairman): 20.0 pts
    - Officer or Director: 10.0 pts
  - **4. Recency Factor (max 10 pts)**:
    - Transaction date $\le 7$ days ago: 10.0 pts
    - Transaction date $\le 14$ days ago: 5.0 pts
- `scoring_model_changed`: `false` (Preserves exact pre-existing arithmetic implementation in `form4_scanner.py`).

### Gate B4: Semantic Taxonomy & Execution Ladder Decoupling
- **Taxonomy Isolation**: Form 4 insider transactions are strictly labeled `INSIDER_FORM4`. They are never aggregated or conflated with Institutional 13F, Dark Pool, Congressional, or Options Flow evidence.
- **Execution Ladder Decoupling**: Form 4 scanner candidate results emit `signalStatus = "QUALIFIED_INSIDER_ACCUMULATION"`. The label `IN_BUY_ZONE` is completely purged from Smart Money outputs; buy-zone classification is reserved exclusively for the technical Execution Ladder engine.

### Gate B5: Immutable Snapshot Publishing & Storage
- Snapshots are committed to `scanner_snapshots` with SQLite database triggers prohibiting `UPDATE` or `DELETE`.
- Content verification is audited using `semantic_fingerprint`.
- Active generation pointer updated atomically in `scanner_active_snapshots`.
- Quarantined snapshots (`PUBLICATION_DECISION != PUBLISH` or uncertified `coverage_status == 'PARTIAL'`) never advance the active pointer.

### Gate B6: Operational Topology, Concurrency & Rate Governance
- Global request budget: Strictly enforced at $\le 5$ requests per second across all processes using shared `DEFAULT_SEC_RATE_LIMITER`.
- User-Agent compliance: Enforces declared format `<Entity/AppName> <AdminContact@domain.com>`.
- Concurrency & Coordination: `DurableRunCoordinator` with distributed SQLite lease prevents overlapping scans (`RESOURCE_KEY_SMART_MONEY_PIPELINE`).
- SQLite Configuration: All Smart Money database access routed through `connect_smart_money_db(db_path)` applying `WAL`, `busy_timeout = 5000ms`, `synchronous = NORMAL`.

### Candidate 003 Operational Remediations:
1. **Single SEC Egress Owner (`ARX_SEC_EGRESS_ROLE`)**:
   - `ARX_SEC_EGRESS_ROLE=DISABLED` by default on all web API replicas.
   - Web API endpoints (`/sec-filings/{symbol}`, submissions) are strictly cache-only. Cache miss returns `status = CONFIGURATION_BLOCKED`, `items = null` with explicit error detail, performing ZERO outbound HTTP calls.
   - Only dedicated worker processes with `ARX_SEC_EGRESS_ROLE=OWNER` may initiate outbound SEC requests.
2. **Point-in-Time Effective Form 4 Evidence & Amendment Resolution**:
   - Resolved amendment chains supersede prior accessions point-in-time: at watermark $W$, accession $A$ is excluded if an amending filing $B$ was publicly available at or before $W$.
   - Ambiguous amendments (`AMENDMENT_TARGET_AMBIGUOUS`) are excluded from effective transaction sets, excluded from gross purchase value, and disqualify the issuer from Radar qualification under `AMENDMENT_UNRESOLVED`.
3. **Material Gap Certification Gating**:
   - `sm_reconciliation_reports` strictly prohibits `coverage_status = COMPLETE` if any material gaps (`MISSING_ACCESSION`, `FETCH_FAILURE`, `PARSE_FAILURE`), source integrity conflicts (`SOURCE_INTEGRITY_CONFLICT`), open gaps in `sm_source_gaps`, or ambiguous amendments exist.
4. **Exact Certification Receipt Binding**:
   - Radar scanning as of watermark $W$ verifies that a reconciliation report explicitly binds $W$ (or $W$'s day interval) with `coverage_status = COMPLETE`, `open_gap_count = 0`, and `missing_count = 0`. Future or unrelated receipts cannot certify prior un-reconciled intervals.
5. **Operator Token Constant-Time Verification**:
   - Removed hardcoded fallback `"arx-operator-secret-v1"`.
   - `verify_operator_token` requires non-empty `ARX_OPERATOR_TOKEN` and executes constant-time `hmac.compare_digest` verification.
6. **Unified SQLite Connection Policy**:
   - All Smart Money modules use `connect_smart_money_db(db_path)` enforcing WAL mode, 5000ms busy timeout, and NORMAL synchronous persistence.

---

## 4. Verification Test Matrix Summary

| Test ID | Gate / Phase | Coverage Scope | Status |
|---|---|---|---|
| `test_gate_b1_global_discovery_enumeration` | Gate B1 | SEC Atom Feed parsing, CIK normalization, accession extraction | PASSED |
| `test_gate_b1_cursor_continuation` | Gate B1 | Pagination and cursor management across batch runs | PASSED |
| `test_gate_b1_reconciliation_parity_and_gap_detection` | Gate B1 / D | Daily index reconciliation, gap recording into `sm_source_gaps` | PASSED |
| `test_gate_b2_pit_temporal_isolation` | Gate B2 / F | Strict boundary isolation at certified watermark $W$ | PASSED |
| `test_gate_b2_late_arrival_temporal_isolation` | Gate B2 | Late filings filed in Oct for June trades excluded from Aug $W$ | PASSED |
| `test_gate_b2_ingestion_idempotency_and_conflict` | Gate B2 / E | Accession idempotency (NOOP) and SHA-256 conflict detection | PASSED |
| `test_gate_b3_frozen_qualification_predicate_boundaries` | Gate B3 / H | Strict threshold boundaries ($20k excluded, $25k included) | PASSED |
| `test_gate_b3_conviction_scoring_components` | Gate B3 | 100-point conviction score component breakdown verification | PASSED |
| `test_gate_b5_snapshot_store_immutability_and_triggers` | Gate B5 / J | SQLite triggers abort `UPDATE` and `DELETE` on snapshot store | PASSED |
| `test_gate_b5_active_pointer_and_quarantine` | Gate B5 | Atomic pointer update and quarantine exclusion | PASSED |
| `test_radar_taxonomy_and_category_classification` | Gate B4 / K | Category classification (`SMART_MONEY_ONLY`, `CONFLUENT`) | PASSED |
| `test_gate_b6_rate_limiter_pacing` | Gate B6 | Enforces 5 req/s ceiling (interval = 0.20s) | PASSED |
| `test_gate_b6_scheduler_overlap_rejection` | Gate B6 / M | Distributed lease coordination prevents overlapping executions | PASSED |

---

## 5. Deployment & Release Boundaries

Under fail-closed governance:
- `PRODUCTION_DEPLOYMENT_AUTHORIZED`: `false`
- `PRODUCTION_MUTATION_PERFORMED`: `false`
- `PRODUCTION_SCHEDULER_ARMED`: `false`
- `PRODUCTION_RADAR_ACTIVATED`: `false`
- `HARDWARE_FIDO2_QUORUM_SATISFIED`: `false`
- `VCP_GOVERNANCE_CHANGED`: `false`
- `EXECUTION_LADDER_WORKSTREAM`: `PARKED`

---

## 6. Ratified PostgreSQL Production Architecture & Topology V1

### Core Topology
```text
                    SEC EDGAR
                        │
                        ▼
              SMART MONEY INGESTION
             ARX_SEC_EGRESS_ROLE=OWNER
                  single active owner
                        │
                        ▼
                  POSTGRESQL
              authoritative shared state
                        ▲
                        │
            ┌───────────┴───────────┐
            │                       │
         WEB API                OTHER READERS
   ARX_SEC_EGRESS_ROLE=DISABLED
      least-privilege DB access
```

### Architectural Tenets
1. **Authoritative Shared State**:
   - PostgreSQL (v16+) replaces container-ephemeral SQLite for production deployments.
   - Schema version: `SMART_MONEY_POSTGRES_SCHEMA_V1`.
2. **Database Role & Privilege Separation**:
   - `smart_money_web`: Least-privilege SELECT on public snapshots/transactions; INSERT into `sm_run_requests`. Strictly denied write access to raw filings, receipts, and snapshots.
   - `smart_money_ingestion`: Operational write credentials for ingestion, reconciliation, and snapshotting. Denied DDL alterations.
   - `smart_money_migration`: Dedicated migration identity for schema migrations.
3. **Dedicated Ingestion Worker & Single SEC Egress Owner**:
   - Single-replica background worker (`python -m analyst_dashboard.smart_money.worker`).
   - Requires `ARX_SEC_EGRESS_ROLE=OWNER`.
   - Acquires PostgreSQL session-level advisory lock `LOCK_SEC_EGRESS_OWNER` (`0x5345435F00000001`). If lock acquisition fails, worker fails closed immediately.
4. **Append-Only Reconciliation Receipts**:
   - Destructive `INSERT OR REPLACE` table `sm_reconciliation_reports` is superseded by immutable, versioned `sm_reconciliation_receipts`.
   - Receipts are protected by database triggers rejecting `UPDATE` and `DELETE`. Corrections append new receipt records with `supersedes_receipt_id`.
5. **Immutable Scanner Snapshots & Atomic Pointer Publication**:
   - `scanner_snapshots` is append-only with database triggers rejecting `UPDATE` and `DELETE`.
   - `matched_count` is nullable: set to `null` for quarantined/partial scans, and non-negative integer for complete scans.
   - `scanner_active_snapshots` is updated in the same serializable publication transaction.
6. **Operator Run Request Queue**:
   - Web API submits run requests to `sm_run_requests` (`status = 'PENDING'`).
   - Dedicated ingestion worker claims requests via `SELECT ... FOR UPDATE SKIP LOCKED` and publishes results.
7. **Production Storage Durability & Backup Contract**:
   - PostgreSQL production authorization requires explicit verification of persistent volumes, scheduled automated backups, and Point-In-Time Recovery (PITR).
   - A verified restore drill must precede production activation.

> **GOVERNANCE ASSERTION**: `POSTGRES IMPLEMENTED ≠ PRODUCTION DEPLOYED`. Implementation and local verification of PostgreSQL topology do not constitute production authorization.

