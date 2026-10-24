-- ============================================================================
-- ARX CORPORATE EVENTS RC5 TELEMETRY PERSISTENCE SCHEMA (INCREMENT 2B)
-- Migration: 005_arx_rc5_corporate_events_telemetry.sql
-- Dialect: ANSI SQL / SQLite 3 (WAL) / PostgreSQL 15+ compatible
-- Description: Additive, non-blocking telemetry tables enforcing dual uniqueness:
--              1. UNIQUE (release_sha, evaluation_id) for logical evaluation at-most-once counting.
--              2. UNIQUE (delivery_attempt_id) for transport delivery packet deduplication.
-- ============================================================================

-- Table 1: Observation Epochs Registry
-- Tracks authorized observation windows for natural production denominators.
CREATE TABLE IF NOT EXISTS arx_rc5_observation_epochs (
    epoch_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    authorized_release_sha VARCHAR(40) NOT NULL,
    start_timestamp_utc VARCHAR(30) NOT NULL,
    end_timestamp_utc VARCHAR(30) NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0, -- 1=Active, 0=Inactive / Closed
    created_at_utc VARCHAR(30) NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_rc5_epoch_active 
    ON arx_rc5_observation_epochs (is_active, authorized_release_sha);

-- Table 2: Authoritative Deduplicated Evaluations
-- Contains exactly one record per logical evaluation. Enforces at-most-once counting.
CREATE TABLE IF NOT EXISTS arx_rc5_telemetry_evaluations (
    evaluation_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    release_sha VARCHAR(40) NOT NULL,
    evaluation_id VARCHAR(64) NOT NULL,
    canonical_digest VARCHAR(64) NOT NULL,
    schema_version VARCHAR(32) NOT NULL,
    event_id VARCHAR(36) NOT NULL,
    occurred_at_utc VARCHAR(30) NOT NULL,
    ruleset_version VARCHAR(64) NOT NULL,
    evaluation_context_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    event_category VARCHAR(32) NOT NULL,
    event_time_utc VARCHAR(30),
    market_session VARCHAR(16) NOT NULL,
    event_source_status VARCHAR(16) NOT NULL,
    freshness_status VARCHAR(16) NOT NULL,
    days_to_earnings REAL,
    risk_bucket VARCHAR(16) NOT NULL,
    applied_penalty REAL NOT NULL,
    evaluation_outcome VARCHAR(16) NOT NULL,
    first_received_at_utc VARCHAR(30) NOT NULL,
    retransmission_count INTEGER NOT NULL DEFAULT 0,
    is_natural_candidate INTEGER NOT NULL DEFAULT 0, -- 1=Eligible, 0=Synthetic/Excluded
    epoch_id VARCHAR(64),
    CONSTRAINT uq_rc5_eval_logical UNIQUE (release_sha, evaluation_id),
    CONSTRAINT fk_rc5_eval_epoch FOREIGN KEY (epoch_id) REFERENCES arx_rc5_observation_epochs(epoch_id)
);

CREATE INDEX IF NOT EXISTS idx_rc5_eval_digest 
    ON arx_rc5_telemetry_evaluations (canonical_digest);

CREATE INDEX IF NOT EXISTS idx_rc5_eval_symbol 
    ON arx_rc5_telemetry_evaluations (symbol, occurred_at_utc);

CREATE INDEX IF NOT EXISTS idx_rc5_eval_natural_counting 
    ON arx_rc5_telemetry_evaluations (is_natural_candidate, release_sha, occurred_at_utc);

-- Table 3: Raw Transmission Delivery Ledger
-- Records every network attempt forwarded from edge gateway, including retransmissions.
CREATE TABLE IF NOT EXISTS arx_rc5_telemetry_raw_events (
    raw_event_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_attempt_id VARCHAR(64) NOT NULL,
    deduplication_key VARCHAR(64) NOT NULL,
    evaluation_id VARCHAR(64) NOT NULL,
    release_sha VARCHAR(40) NOT NULL,
    received_at_utc VARCHAR(30) NOT NULL,
    edge_timestamp_utc VARCHAR(30),
    key_id VARCHAR(32),
    client_ip_hash VARCHAR(64),
    user_agent_hash VARCHAR(64),
    raw_payload_json TEXT NOT NULL,
    ingestion_verdict VARCHAR(32) NOT NULL, -- PERSISTED, CONFIRMED_DUPLICATE, REJECTED
    CONSTRAINT uq_rc5_raw_delivery UNIQUE (delivery_attempt_id)
);

CREATE INDEX IF NOT EXISTS idx_rc5_raw_dedup 
    ON arx_rc5_telemetry_raw_events (deduplication_key);

CREATE INDEX IF NOT EXISTS idx_rc5_raw_eval_ref 
    ON arx_rc5_telemetry_raw_events (release_sha, evaluation_id);

-- Table 4: Telemetry Quarantine & Security Anomaly Table
-- Holds malformed, signature-failed, replay-expired, or conflicting evaluation attempts.
CREATE TABLE IF NOT EXISTS arx_rc5_telemetry_quarantine (
    quarantine_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    quarantine_id VARCHAR(64) NOT NULL,
    quarantined_at_utc VARCHAR(30) NOT NULL,
    reason_code VARCHAR(32) NOT NULL, -- CONFLICTING_IDENTITY, REPLAY_EXPIRED, BAD_HMAC, etc.
    details TEXT,
    delivery_attempt_id VARCHAR(64),
    evaluation_id VARCHAR(64),
    release_sha VARCHAR(40),
    raw_payload_snippet TEXT,
    CONSTRAINT uq_rc5_quarantine_id UNIQUE (quarantine_id)
);

CREATE INDEX IF NOT EXISTS idx_rc5_quarantine_reason 
    ON arx_rc5_telemetry_quarantine (reason_code, quarantined_at_utc);
