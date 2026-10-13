"""
ARX Terminal — PostgreSQL Schema Specification & DDL Definition.

Defines SMART_MONEY_POSTGRES_SCHEMA_V1 tables, indexes, constraints,
and immutable triggers enforcing append-only semantics.
"""

import logging
from psycopg2.extensions import connection as PgConnection
from analyst_dashboard.smart_money.storage.config import SMART_MONEY_POSTGRES_SCHEMA_VERSION

logger = logging.getLogger(__name__)

SCHEMA_DDL = """
-- Schema migration metadata
CREATE TABLE IF NOT EXISTS sm_schema_migrations (
    version_id VARCHAR(64) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    release_sha VARCHAR(64) NOT NULL
);

-- Universal append-only rejection function
CREATE OR REPLACE FUNCTION abort_immutable_record() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'IMMUTABILITY_VIOLATION: Table % is append-only. UPDATE and DELETE are strictly prohibited.', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

-- 1. Raw Filings Store
CREATE TABLE IF NOT EXISTS form4_raw_filings (
    accession_number VARCHAR(32) PRIMARY KEY,
    cik VARCHAR(16) NOT NULL,
    ticker VARCHAR(16) NOT NULL,
    company_name TEXT NOT NULL,
    filing_date DATE NOT NULL,
    accepted_datetime TIMESTAMPTZ NOT NULL,
    xml_content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    source_url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_filings_cik ON form4_raw_filings(cik);
CREATE INDEX IF NOT EXISTS idx_raw_filings_filing_date ON form4_raw_filings(filing_date);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_raw_filings_immutable') THEN
        CREATE TRIGGER trg_raw_filings_immutable
        BEFORE UPDATE OR DELETE ON form4_raw_filings
        FOR EACH ROW EXECUTE FUNCTION abort_immutable_record();
    END IF;
END $$;

-- 2. Parsed Transactions Store
CREATE TABLE IF NOT EXISTS form4_parsed_transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    accession_number VARCHAR(32) NOT NULL REFERENCES form4_raw_filings(accession_number) ON DELETE RESTRICT,
    transaction_date DATE NOT NULL,
    transaction_code VARCHAR(4) NOT NULL,
    shares NUMERIC(18, 4) NOT NULL,
    price_per_share NUMERIC(18, 4) NOT NULL,
    total_value_usd NUMERIC(18, 4) NOT NULL,
    reporting_owner_name TEXT NOT NULL,
    reporting_owner_role TEXT NOT NULL,
    is_director BOOLEAN NOT NULL DEFAULT FALSE,
    is_officer BOOLEAN NOT NULL DEFAULT FALSE,
    is_ten_percent BOOLEAN NOT NULL DEFAULT FALSE,
    officer_title TEXT,
    ownership_type VARCHAR(1) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_parsed_tx_accession ON form4_parsed_transactions(accession_number);
CREATE INDEX IF NOT EXISTS idx_parsed_tx_date ON form4_parsed_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_parsed_tx_code ON form4_parsed_transactions(transaction_code);

-- 3. Point-in-Time Availability Ledger
CREATE TABLE IF NOT EXISTS sm_source_availability_ledger (
    accession_number VARCHAR(32) PRIMARY KEY REFERENCES form4_raw_filings(accession_number) ON DELETE RESTRICT,
    public_availability_at TIMESTAMPTZ NOT NULL,
    source VARCHAR(32) NOT NULL,
    source_reference TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_avail_public_at ON sm_source_availability_ledger(public_availability_at);

-- 4. Amendment Relationship Ledger
CREATE TABLE IF NOT EXISTS sm_amendments (
    amendment_accession VARCHAR(32) PRIMARY KEY,
    base_accession VARCHAR(32),
    amendment_number INTEGER NOT NULL DEFAULT 1,
    resolution_status VARCHAR(32) NOT NULL,
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_amendments_base ON sm_amendments(base_accession);

-- 5. Source Gap & Discrepancy Ledger
CREATE TABLE IF NOT EXISTS sm_source_gaps (
    gap_id VARCHAR(64) PRIMARY KEY,
    accession_number VARCHAR(32),
    issuer_cik VARCHAR(16),
    universe_build_id VARCHAR(64),
    gap_type VARCHAR(64) NOT NULL,
    description TEXT,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(32) NOT NULL DEFAULT 'OPEN'
);

CREATE INDEX IF NOT EXISTS idx_gaps_status ON sm_source_gaps(status);

-- 6. Append-Only Reconciliation Receipts
CREATE TABLE IF NOT EXISTS sm_reconciliation_receipts (
    receipt_id UUID PRIMARY KEY,
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    certified_watermark TIMESTAMPTZ NOT NULL,
    receipt_version INTEGER NOT NULL DEFAULT 1,
    supersedes_receipt_id UUID REFERENCES sm_reconciliation_receipts(receipt_id),
    expected_count INTEGER NOT NULL,
    accounted_count INTEGER NOT NULL,
    missing_count INTEGER NOT NULL,
    material_gap_count INTEGER NOT NULL,
    coverage_status VARCHAR(32) NOT NULL,
    failure_reasons JSONB NOT NULL DEFAULT '[]',
    reconciliation_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    release_sha VARCHAR(64) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_receipts_interval ON sm_reconciliation_receipts(interval_start, interval_end, receipt_version DESC);
CREATE INDEX IF NOT EXISTS idx_receipts_watermark ON sm_reconciliation_receipts(certified_watermark);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_receipts_append_only') THEN
        CREATE TRIGGER trg_receipts_append_only
        BEFORE UPDATE OR DELETE ON sm_reconciliation_receipts
        FOR EACH ROW EXECUTE FUNCTION abort_immutable_record();
    END IF;
END $$;

-- 7. Immutable Scanner Snapshots Store
CREATE TABLE IF NOT EXISTS scanner_snapshots (
    snapshot_id VARCHAR(64) PRIMARY KEY,
    scanner_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    api_contract_version VARCHAR(32) NOT NULL,
    ruleset_version VARCHAR(32) NOT NULL,
    evidence_schema_version VARCHAR(32) NOT NULL,
    score_model_version VARCHAR(32) NOT NULL,
    data_provenance_version VARCHAR(32) NOT NULL,
    universe_version VARCHAR(32) NOT NULL,
    freshness_policy_version VARCHAR(32) NOT NULL,
    implementation_release_sha VARCHAR(64) NOT NULL,
    semantic_fingerprint VARCHAR(64) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    data_as_of TIMESTAMPTZ NOT NULL,
    status_at_publication VARCHAR(32) NOT NULL,
    universe_id VARCHAR(64) NOT NULL,
    universe_size INTEGER NOT NULL,
    matched_count INTEGER,  -- NULLABLE: None for Quarantined/Partial, non-negative integer for Complete
    results_json JSONB NOT NULL,
    provenance_json JSONB NOT NULL,
    freshness_json JSONB NOT NULL,
    publication_decision VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_scanner_created ON scanner_snapshots(scanner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_watermark ON scanner_snapshots(data_as_of);
CREATE INDEX IF NOT EXISTS idx_snapshots_fingerprint ON scanner_snapshots(semantic_fingerprint);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_snapshots_append_only') THEN
        CREATE TRIGGER trg_snapshots_append_only
        BEFORE UPDATE OR DELETE ON scanner_snapshots
        FOR EACH ROW EXECUTE FUNCTION abort_immutable_record();
    END IF;
END $$;

-- 8. Scanner Active Snapshots Pointer
CREATE TABLE IF NOT EXISTS scanner_active_snapshots (
    scan_type VARCHAR(64) PRIMARY KEY,
    active_snapshot_id VARCHAR(64) NOT NULL REFERENCES scanner_snapshots(snapshot_id),
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_by VARCHAR(64) NOT NULL
);

-- 9. Governed Issuers (Universe Authority)
CREATE TABLE IF NOT EXISTS governed_issuers (
    cik VARCHAR(16) PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL,
    company_name TEXT NOT NULL,
    sic VARCHAR(8),
    fiscal_year_end VARCHAR(8),
    scannable BOOLEAN NOT NULL DEFAULT TRUE,
    universe_build_id VARCHAR(64) NOT NULL,
    manifest_hash VARCHAR(64) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_governed_ticker ON governed_issuers(ticker);

-- 10. Append-Only Audit Event Ledger
CREATE TABLE IF NOT EXISTS sm_audit_events (
    event_id UUID PRIMARY KEY,
    event_type VARCHAR(64) NOT NULL,
    subject_id VARCHAR(64),
    payload JSONB NOT NULL,
    event_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_type_created ON sm_audit_events(event_type, created_at DESC);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_audit_append_only') THEN
        CREATE TRIGGER trg_audit_append_only
        BEFORE UPDATE OR DELETE ON sm_audit_events
        FOR EACH ROW EXECUTE FUNCTION abort_immutable_record();
    END IF;
END $$;

-- 11. Run Requests (Queue for Web -> Worker Execution)
CREATE TABLE IF NOT EXISTS sm_run_requests (
    request_id UUID PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_type VARCHAR(64) NOT NULL,
    requested_by_subject_hash VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
    claimed_at TIMESTAMPTZ,
    claimed_by VARCHAR(64),
    completed_at TIMESTAMPTZ,
    result_generation_id VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_run_requests_queue ON sm_run_requests(status, requested_at);
"""


def initialize_postgres_schema(conn: PgConnection, release_sha: str = "b72ea23937cf6ed7b8c4b60096630767a7c4592b") -> None:
    """
    Applies the SMART_MONEY_POSTGRES_SCHEMA_V1 DDL script to the database.
    Records migration execution in sm_schema_migrations.
    """
    with conn.cursor() as cur:
        cur.execute(SCHEMA_DDL)
        cur.execute(
            """
            INSERT INTO sm_schema_migrations (version_id, release_sha)
            VALUES (%s, %s)
            ON CONFLICT (version_id) DO UPDATE SET release_sha = EXCLUDED.release_sha;
            """,
            (SMART_MONEY_POSTGRES_SCHEMA_VERSION, release_sha),
        )
    conn.commit()
    logger.info(f"Initialized schema {SMART_MONEY_POSTGRES_SCHEMA_VERSION} successfully.")
