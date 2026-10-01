-- Migration 001: ARX ETF Cockpit P1 Telemetry & Audit Tables
-- Target Engines: PostgreSQL (Railway Production) & SQLite (Local / In-Memory Test Fixtures)
-- Invariants:
-- 1. Append-only audit ledger (arx_p1_telemetry_audit_records).
-- 2. Deduplication key uniqueness constraint enforces exactly one canonical observation per user attempt.
-- 3. Referential integrity and strict domain states (VALID, QUARANTINED, EXCLUDED, INVALID).
-- 4. Epoch registry stores authorized release sets and start/end boundaries.

-- 1. Observation Epochs Registry
CREATE TABLE IF NOT EXISTS arx_p1_observation_epochs (
    epoch_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    start_timestamp TIMESTAMP NOT NULL,
    end_timestamp TIMESTAMP,
    authorized_releases TEXT NOT NULL, -- JSON array of authorized Git commit SHAs
    activated_at TIMESTAMP NOT NULL,
    activated_by VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Raw Events Append-Only Intake Log
CREATE TABLE IF NOT EXISTS arx_p1_telemetry_raw_events (
    event_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    attempt_id VARCHAR(64) NOT NULL,
    deduplication_key VARCHAR(64) NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_payload TEXT NOT NULL, -- Full JSON payload string
    client_ip_hash VARCHAR(64),
    user_agent_raw TEXT
);

-- 3. Canonical 16-Field Observation Audit Ledger
CREATE TABLE IF NOT EXISTS arx_p1_telemetry_audit_records (
    observation_unit_id VARCHAR(64) PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    normalized_symbol VARCHAR(20) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    deployment_identity VARCHAR(100) NOT NULL,
    release_sha VARCHAR(64) NOT NULL,
    traffic_class VARCHAR(50) NOT NULL,
    classification_state VARCHAR(20) NOT NULL,
    classification_reason TEXT NOT NULL,
    exclusion_code VARCHAR(50) NOT NULL,
    quarantine_reason VARCHAR(50) NOT NULL,
    deduplication_key VARCHAR(64) UNIQUE NOT NULL,
    replay_identity VARCHAR(64) NOT NULL,
    source_component VARCHAR(100) NOT NULL,
    epoch_id VARCHAR(64),
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Malformed / Invalid Payloads Isolation Quarantine Table
CREATE TABLE IF NOT EXISTS arx_p1_telemetry_invalid_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_unparsed_payload TEXT NOT NULL,
    rejection_reason VARCHAR(255) NOT NULL,
    client_ip_hash VARCHAR(64)
);

-- Indices for rapid audit verification and derived denominator calculation
CREATE INDEX IF NOT EXISTS idx_audit_state_epoch ON arx_p1_telemetry_audit_records (classification_state, epoch_id);
CREATE INDEX IF NOT EXISTS idx_audit_symbol ON arx_p1_telemetry_audit_records (normalized_symbol);
CREATE INDEX IF NOT EXISTS idx_audit_dedup ON arx_p1_telemetry_audit_records (deduplication_key);
CREATE INDEX IF NOT EXISTS idx_raw_dedup ON arx_p1_telemetry_raw_events (deduplication_key);
