-- ARX Terminal — PostgreSQL Role & Least-Privilege Permission Definitions
-- Defines database principals: smart_money_web, smart_money_ingestion, smart_money_migration

-- 1. Create Roles (if not existing)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'smart_money_web') THEN
        CREATE ROLE smart_money_web WITH LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'smart_money_ingestion') THEN
        CREATE ROLE smart_money_ingestion WITH LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'smart_money_migration') THEN
        CREATE ROLE smart_money_migration WITH LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'smart_money_backup') THEN
        CREATE ROLE smart_money_backup WITH LOGIN;
    END IF;
END $$;

-- 2. Schema Usage Grants
GRANT USAGE ON SCHEMA public TO smart_money_web, smart_money_ingestion, smart_money_migration, smart_money_backup;

-- 3. WEB PRINCIPAL: Least-Privilege Read-Only & Narrow Run-Request Submission
-- May read snapshots, transactions, universe, availability, and receipts
GRANT SELECT ON
    scanner_snapshots,
    scanner_active_snapshots,
    form4_parsed_transactions,
    governed_issuers,
    sm_source_availability_ledger,
    sm_reconciliation_receipts,
    sm_audit_events
TO smart_money_web;

-- Narrow write capability: web may submit run requests only
GRANT INSERT, SELECT ON sm_run_requests TO smart_money_web;

-- Strictly REVOKE all modifications to authoritative evidence
REVOKE INSERT, UPDATE, DELETE ON form4_raw_filings FROM smart_money_web;
REVOKE INSERT, UPDATE, DELETE ON form4_parsed_transactions FROM smart_money_web;
REVOKE INSERT, UPDATE, DELETE ON sm_reconciliation_receipts FROM smart_money_web;
REVOKE INSERT, UPDATE, DELETE ON scanner_snapshots FROM smart_money_web;
REVOKE UPDATE, DELETE ON scanner_active_snapshots FROM smart_money_web;
REVOKE INSERT, UPDATE, DELETE ON sm_amendments FROM smart_money_web;
REVOKE INSERT, UPDATE, DELETE ON sm_source_gaps FROM smart_money_web;

-- 4. INGESTION PRINCIPAL: Operational Pipeline Writes
GRANT SELECT, INSERT, UPDATE, DELETE ON
    form4_parsed_transactions,
    sm_source_availability_ledger,
    sm_amendments,
    sm_source_gaps,
    governed_issuers,
    scanner_active_snapshots,
    sm_run_requests
TO smart_money_ingestion;

-- Append-only tables (no UPDATE / DELETE)
GRANT SELECT, INSERT ON
    form4_raw_filings,
    sm_reconciliation_receipts,
    scanner_snapshots,
    sm_audit_events
TO smart_money_ingestion;

-- 5. MIGRATION PRINCIPAL: Schema Administration
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO smart_money_migration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO smart_money_migration;

-- 6. BACKUP PRINCIPAL: Read-Only Backup Capture (Least-Privilege)
-- Backup principal has SELECT access only, strictly barred from mutation or DDL
GRANT SELECT ON ALL TABLES IN SCHEMA public TO smart_money_backup;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO smart_money_backup;

