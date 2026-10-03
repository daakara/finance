-- scripts/research/etf_v2/openfigi_operational_schema.sql
-- Dedicated operational schema for OpenFIGI symbology corroboration evidence.
-- Physically separated from canonical ETF V2 database.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Authoritative, append-only observation history
CREATE TABLE IF NOT EXISTS openfigi_observations (
    observation_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    canonical_internal_id TEXT NOT NULL,
    isin TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    source_population_version TEXT NOT NULL,
    source_snapshot_sha256 TEXT NOT NULL,
    contract_version TEXT NOT NULL DEFAULT '1.0.0',
    request_position INTEGER NOT NULL,
    request_filters_json TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    retry_count INTEGER NOT NULL,
    http_status INTEGER,
    outcome_class TEXT NOT NULL,
    normalized_result_json TEXT NOT NULL,
    provider_response_evidence_json TEXT NOT NULL,
    provider_response_digest TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_openfigi_obs_isin ON openfigi_observations(isin);
CREATE INDEX IF NOT EXISTS idx_openfigi_obs_canonical_id ON openfigi_observations(canonical_internal_id);
CREATE INDEX IF NOT EXISTS idx_openfigi_obs_idempotency ON openfigi_observations(idempotency_key);

-- Derived operational active projection (fast O(1) deduplicated operational queries)
CREATE TABLE IF NOT EXISTS openfigi_active_mappings (
    isin TEXT PRIMARY KEY,
    canonical_internal_id TEXT NOT NULL,
    figi TEXT,
    composite_figi TEXT,
    share_class_figi TEXT,
    ticker TEXT,
    exch_code TEXT,
    security_type TEXT,
    market_sector TEXT,
    name TEXT,
    outcome_class TEXT NOT NULL,
    last_observation_id TEXT NOT NULL REFERENCES openfigi_observations(observation_id),
    source_population_version TEXT NOT NULL,
    source_snapshot_sha256 TEXT NOT NULL,
    contract_version TEXT NOT NULL DEFAULT '1.0.0',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_openfigi_active_figi ON openfigi_active_mappings(figi);
CREATE INDEX IF NOT EXISTS idx_openfigi_active_canonical_id ON openfigi_active_mappings(canonical_internal_id);

-- Global rate limit reservations (atomic multi-process coordination)
CREATE TABLE IF NOT EXISTS openfigi_rate_limit_reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reservation_id TEXT NOT NULL UNIQUE,
    process_id INTEGER NOT NULL,
    reserved_at REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_reserved_at ON openfigi_rate_limit_reservations(reserved_at);
