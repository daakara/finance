-- scripts/research/etf_v2/canonical_population_schema.sql
-- Pipeline V2 Canonical Population Authority Schema (v1)
-- Enforces: Relational integrity, foreign keys, ISIN uniqueness, immutability triggers, and audit logging.

PRAGMA foreign_keys = ON;

-- 1. Schema Version Management
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY NOT NULL,
    migration_name TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL
);

-- 2. Canonical Parent Entity (Umbrella / ICAV / SICAV / Trust)
CREATE TABLE IF NOT EXISTS canonical_parent_entity (
    canonical_parent_id TEXT PRIMARY KEY NOT NULL,
    legal_umbrella_name TEXT NOT NULL,
    domicile_iso2 TEXT NOT NULL,
    regulatory_jurisdiction TEXT NOT NULL,
    regulator TEXT NOT NULL,
    legal_entity_structure TEXT NOT NULL,
    national_regulator_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CONSTRAINT uq_parent_umbrella_domicile UNIQUE (legal_umbrella_name, domicile_iso2)
);
CREATE INDEX IF NOT EXISTS idx_parent_jurisdiction ON canonical_parent_entity(regulatory_jurisdiction, domicile_iso2);

-- 3. Canonical Sub-Fund (Statutory Compartment)
CREATE TABLE IF NOT EXISTS canonical_subfund (
    canonical_subfund_id TEXT PRIMARY KEY NOT NULL,
    canonical_parent_id TEXT NOT NULL REFERENCES canonical_parent_entity(canonical_parent_id) ON DELETE RESTRICT,
    legal_subfund_name TEXT NOT NULL,
    subfund_currency TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CONSTRAINT uq_subfund_parent_name UNIQUE (canonical_parent_id, legal_subfund_name)
);
CREATE INDEX IF NOT EXISTS idx_subfund_parent ON canonical_subfund(canonical_parent_id);

-- 4. Canonical Share Class (Tranche / ISIN)
CREATE TABLE IF NOT EXISTS canonical_share_class (
    canonical_share_class_id TEXT PRIMARY KEY NOT NULL,
    isin TEXT UNIQUE NOT NULL,
    canonical_subfund_id TEXT NOT NULL REFERENCES canonical_subfund(canonical_subfund_id) ON DELETE RESTRICT,
    canonical_parent_id TEXT NOT NULL REFERENCES canonical_parent_entity(canonical_parent_id) ON DELETE RESTRICT,
    jurisdiction TEXT NOT NULL,
    regulatory_framework TEXT NOT NULL,
    legal_subfund_name TEXT NOT NULL,
    legal_share_class_name TEXT NOT NULL,
    canonical_status TEXT NOT NULL,
    currentness_state TEXT NOT NULL,
    authority_tier TEXT NOT NULL,
    admission_state TEXT NOT NULL,
    admitted_at TEXT NOT NULL,
    admission_gate_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CONSTRAINT uq_share_class_subfund_name UNIQUE (canonical_subfund_id, legal_share_class_name),
    CONSTRAINT chk_isin_length CHECK (length(isin) = 12)
);
CREATE INDEX IF NOT EXISTS idx_share_class_isin ON canonical_share_class(isin);
CREATE INDEX IF NOT EXISTS idx_share_class_subfund ON canonical_share_class(canonical_subfund_id);
CREATE INDEX IF NOT EXISTS idx_share_class_parent ON canonical_share_class(canonical_parent_id);
CREATE INDEX IF NOT EXISTS idx_share_class_status ON canonical_share_class(canonical_status, admission_state);

-- 5. Canonical Provenance Records (Evidence Linkage)
CREATE TABLE IF NOT EXISTS canonical_provenance_records (
    provenance_record_id TEXT PRIMARY KEY NOT NULL,
    canonical_share_class_id TEXT NOT NULL REFERENCES canonical_share_class(canonical_share_class_id) ON DELETE RESTRICT,
    evidence_object_id TEXT NOT NULL,
    source_candidate_id TEXT NOT NULL,
    source_document_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    authority_tier TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provenance_share_class ON canonical_provenance_records(canonical_share_class_id);
CREATE INDEX IF NOT EXISTS idx_provenance_evidence_obj ON canonical_provenance_records(evidence_object_id);

-- 6. Canonical Hold Records (Pre-Admission Quarantine)
CREATE TABLE IF NOT EXISTS canonical_hold_records (
    hold_record_id TEXT PRIMARY KEY NOT NULL,
    candidate_identifier TEXT NOT NULL,
    hold_category TEXT NOT NULL,
    hold_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_candidate_package TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    hold_governance_gate TEXT NOT NULL,
    reopen_condition TEXT NOT NULL,
    status TEXT NOT NULL,
    resolved_at TEXT,
    resolution_gate_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_hold_candidate ON canonical_hold_records(candidate_identifier);
CREATE INDEX IF NOT EXISTS idx_hold_category_status ON canonical_hold_records(hold_category, status);

-- 7. Canonical Change Audit Log (Append-Only)
CREATE TABLE IF NOT EXISTS canonical_audit_log (
    audit_event_id TEXT PRIMARY KEY NOT NULL,
    admission_batch_id TEXT NOT NULL,
    canonical_share_class_id TEXT NOT NULL,
    isin TEXT NOT NULL,
    change_type TEXT NOT NULL,
    before_state_json TEXT,
    after_state_json TEXT NOT NULL,
    authority_basis TEXT NOT NULL,
    governance_gate_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_share_class ON canonical_audit_log(canonical_share_class_id);
CREATE INDEX IF NOT EXISTS idx_audit_batch ON canonical_audit_log(admission_batch_id);

-- 8. Hard Delete Prohibition Trigger
CREATE TRIGGER IF NOT EXISTS trg_no_hard_delete_share_class
BEFORE DELETE ON canonical_share_class
BEGIN
    SELECT RAISE(ABORT, 'HARD_DELETE_PROHIBITED: Canonical share class records cannot be deleted. Transition status instead.');
END;

-- 9. In-Place ISIN Mutation Prohibition Trigger
CREATE TRIGGER IF NOT EXISTS trg_no_update_isin_share_class
BEFORE UPDATE OF isin ON canonical_share_class
BEGIN
    SELECT RAISE(ABORT, 'ISIN_MUTATION_PROHIBITED: Canonical ISIN cannot be updated in place. Supersede identity instead.');
END;

-- 10. Canonical ID Mutation Prohibition Trigger
CREATE TRIGGER IF NOT EXISTS trg_no_update_id_share_class
BEFORE UPDATE OF canonical_share_class_id ON canonical_share_class
BEGIN
    SELECT RAISE(ABORT, 'ID_MUTATION_PROHIBITED: Canonical share class ID cannot be updated in place.');
END;
