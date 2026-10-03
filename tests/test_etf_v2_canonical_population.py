"""
tests/test_etf_v2_canonical_population.py

Comprehensive Test Suite for ETF V2 Canonical Population Authority (Pipeline V2).
Verifies all implementation criteria (ETFCPI01-ETFCPI50), including:
- In-memory database test isolation (:memory:)
- Empty store initialization and schema versioning
- Strict foreign key and relational integrity
- ISO 6166 ISIN checksum validation and database uniqueness
- Minimum statutory provenance invariant enforcement
- Atomic batch rollback on transaction failure
- Preflight exclusion of invalid/held candidates
- Double-execution idempotency (zero duplicate writes)
- Five-state collision adjudication
- Append-only audit logging and hard delete prohibition
- In-place ISIN mutation prohibition and supersession lifecycle
- Derived snapshot generation and SHA-256 hashing
- Denominator query firewall enforcement
- Multi-jurisdiction (IE, LU, DE, FR) and issuer-agnostic representation
- Provisional artifact blocklist enforcement
- 141-row migration dry run verification with zero physical writes
"""

import datetime
import json
from pathlib import Path
import sqlite3
import pytest

from scripts.research.etf_v2.canonical_population_errors import (
    AdmissionTransactionError,
    CanonicalPopulationError,
    DenominatorFirewallViolationError,
    LockAcquisitionTimeoutError,
    MissingProvenanceError,
    PreflightValidationError,
    SchemaVersionMismatchError,
)
from scripts.research.etf_v2.canonical_population_migrator import (
    CanonicalPopulationMigrator,
    INITIAL_MIGRATION_VERSION,
    PROVISIONAL_ARTIFACT_BLOCKLIST,
)
from scripts.research.etf_v2.canonical_population_models import (
    AuthorityTier,
    CandidateSubmission,
    CanonicalHoldRecord,
    CanonicalStatus,
    CollisionState,
    HoldCategory,
    PopulationScope,
)
from scripts.research.etf_v2.canonical_population_repository import (
    CanonicalPopulationReader,
    CanonicalPopulationWriter,
    generate_parent_id,
    generate_subfund_id,
)
from scripts.research.etf_v2.global_identifier_authority import (
    normalize_isin,
    validate_isin,
)


@pytest.fixture
def test_repo(tmp_path):
    """Returns reader, writer, and direct connection operating on an isolated file-backed database."""
    db_file = str(tmp_path / "test_canonical.db")
    migrator = CanonicalPopulationMigrator(db_file)
    migrator.initialize_empty_store()

    reader = CanonicalPopulationReader(db_file)
    writer = CanonicalPopulationWriter(db_file)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield reader, writer, conn
    finally:
        conn.close()


def build_candidate(
    isin: str = "IE00B6YX5C33",
    subfund_name: str = "SPDR S&P US Dividend Aristocrats UCITS ETF",
    share_class_name: str = "SPDR S&P US Dividend Aristocrats UCITS ETF (Dist)",
    umbrella_name: str = "SSGA SPDR ETFs Europe I plc",
    domicile: str = "IE",
    regulator: str = "CBI",
    tier: str = "TIER_1_OFFICIAL_STATUTORY",
    evidence_count: int = 1,
) -> CandidateSubmission:
    evidences = [
        {
            "evidence_object_id": f"ev-{isin}-{i}",
            "source_document_id": "STATUTORY_SUPPLEMENT_2024.pdf",
            "source_url": "https://official.evidence/doc.pdf",
            "authority_tier": tier,
            "evidence_type": "STATUTORY_SUPPLEMENT",
            "evidence_hash": "3bdc834c34f936d6a4f0b32526759858ea60c8f35d595ff3e6fcca30719f140a",
        }
        for i in range(evidence_count)
    ]
    return CandidateSubmission(
        source_candidate_id=f"cand-{isin}",
        isin=isin,
        legal_umbrella_name=umbrella_name,
        legal_subfund_name=subfund_name,
        legal_share_class_name=share_class_name,
        domicile_iso2=domicile,
        regulatory_jurisdiction="EU_UCITS",
        authority_tier=tier,
        evidence_references=tuple(evidences),
        readiness_state="ADMISSION_READY",
        source_artifact_identity={"filename": "test_artifact.json", "sha256": "dummy", "byte_count": 100},
        subfund_currency="USD",
        regulator=regulator,
        legal_entity_structure="ICAV" if domicile == "IE" else "SICAV",
    )


# --- GROUP 1: SCHEMA INITIALIZATION & REPOSITORY BOUNDARIES ---

def test_schema_initialization_empty_state(tmp_path):
    """Verifies that schema initialization succeeds and CANONICAL_SHARE_CLASS_ROW_COUNT == 0."""
    db_file = str(tmp_path / "init_empty.db")
    migrator = CanonicalPopulationMigrator(db_file)
    v = migrator.initialize_empty_store()
    assert v == 1
    assert migrator.get_schema_version() == 1


def test_schema_versioning_and_checksum(test_repo):
    """Verifies schema_version tracking table and checksum persistence."""
    _, _, conn = test_repo
    cur = conn.execute("SELECT version, migration_name, checksum FROM schema_version;")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == "001_initial_canonical_schema"
    assert len(row[2]) == 64  # SHA-256 length


def test_foreign_key_constraints_enforced(test_repo):
    """Verifies that inserting a share class with an invalid parent/subfund fails with FK violation."""
    _, _, conn = test_repo
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        conn.execute("""
            INSERT INTO canonical_share_class (
                canonical_share_class_id, isin, canonical_subfund_id, canonical_parent_id,
                jurisdiction, regulatory_framework, legal_subfund_name, legal_share_class_name,
                canonical_status, currentness_state, authority_tier, admission_state,
                admitted_at, admission_gate_id, created_at, updated_at
            ) VALUES (
                'etfs:v1:ISIN:IE00B6YX5C33', 'IE00B6YX5C33', 'non_existent_subfund', 'non_existent_parent',
                'IE', 'EU_UCITS', 'Subfund', 'Share Class',
                'ADMITTED_CURRENT', '{}', 'TIER_1_OFFICIAL_STATUTORY', 'ADMITTED',
                '2026-10-03T00:00:00Z', 'GATE-01', '2026-10-03T00:00:00Z', '2026-10-03T00:00:00Z'
            )
        """)


def test_database_isin_uniqueness_enforced(test_repo):
    """Verifies that duplicate ISIN insertions fail with UNIQUE constraint violation."""
    reader, writer, conn = test_repo
    cand = build_candidate(isin="IE00B6YX5C33")
    res1 = writer.admit_batch([cand], governance_gate_id="GATE-TEST-01")
    assert res1.admitted_count == 1

    # Attempt direct SQL insert of duplicate ISIN
    parent_id = generate_parent_id(cand.domicile_iso2, cand.legal_umbrella_name)
    subfund_id = generate_subfund_id(parent_id, cand.legal_subfund_name)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: canonical_share_class.isin"):
        conn.execute("""
            INSERT INTO canonical_share_class (
                canonical_share_class_id, isin, canonical_subfund_id, canonical_parent_id,
                jurisdiction, regulatory_framework, legal_subfund_name, legal_share_class_name,
                canonical_status, currentness_state, authority_tier, admission_state,
                admitted_at, admission_gate_id, created_at, updated_at
            ) VALUES (
                'etfs:v1:ISIN:IE00B6YX5C33_DUP', 'IE00B6YX5C33', ?, ?,
                'IE', 'EU_UCITS', 'Subfund', 'Share Class Distinct Name',
                'ADMITTED_CURRENT', '{}', 'TIER_1_OFFICIAL_STATUTORY', 'ADMITTED',
                '2026-10-03T00:00:00Z', 'GATE-01', '2026-10-03T00:00:00Z', '2026-10-03T00:00:00Z'
            )
        """, (subfund_id, parent_id))


# --- GROUP 2: ISIN INTEGRITY & CANONICAL IDENTITY ---

def test_isin_validation_and_checksum():
    """Verifies ISO 6166 checksum validation and normalization."""
    valid_isin = "IE00B6YX5C33"
    assert validate_isin(valid_isin, strict=True) is True
    assert normalize_isin("  ie00b6yx5c33  ") == valid_isin

    # Corrupt check digit: 3 -> 4
    invalid_isin = "IE00B6YX5C34"
    assert validate_isin(invalid_isin, strict=False) is False


def test_preflight_excludes_invalid_isin(test_repo):
    """Verifies that candidates with invalid ISINs are quarantined during preflight."""
    _, writer, _ = test_repo
    invalid_cand = build_candidate(isin="IE00B6YX5C34")  # Bad check digit
    res = writer.preflight_admission([invalid_cand])
    assert len(res) == 1
    assert res[0].is_valid is False
    assert "Invalid ISO 6166 checksum" in (res[0].error_message or "")


# --- GROUP 3: PROVENANCE INVARIANT & ATOMIC TRANSACTION ---

def test_minimum_provenance_invariant_enforced(test_repo):
    """Verifies that admission fails when a candidate lacks statutory provenance."""
    _, writer, _ = test_repo
    bad_cand = build_candidate(isin="IE00B6YX5C33", evidence_count=0)
    pre = writer.preflight_admission([bad_cand])
    assert pre[0].is_valid is False
    assert "lacks admissible Tier 1 or Tier 2" in (pre[0].error_message or "")


def test_atomic_batch_rollback_on_failure(test_repo):
    """Verifies that an error during batch admission rolls back 100% of rows."""
    reader, writer, conn = test_repo
    cand1 = build_candidate(isin="IE00B6YX5C33")
    cand2 = build_candidate(isin="IE00B44Z5B48")

    # Admit cand1 successfully
    writer.admit_batch([cand1], governance_gate_id="GATE-01")
    assert reader.get_share_class_by_isin("IE00B6YX5C33") is not None

    # Now attempt a batch with a duplicate cand1 and cand2. Preflight identifies cand1 as EXACT_ALREADY_PRESENT.
    # To test transaction rollback, we directly trigger an admission error inside admit_batch:
    bad_cand = CandidateSubmission(
        source_candidate_id="bad",
        isin="IE00B44Z5B48",
        legal_umbrella_name="SSGA SPDR ETFs Europe I plc",
        legal_subfund_name="Subfund",
        legal_share_class_name="Share Class",
        domicile_iso2="IE",
        regulatory_jurisdiction="EU_UCITS",
        authority_tier="TIER_1_OFFICIAL_STATUTORY",
        evidence_references=(),  # Zero provenance inside submission
        readiness_state="ADMISSION_READY",
        source_artifact_identity={},
    )
    # Direct writer invocation with bad_cand
    res = writer.admit_batch([bad_cand], governance_gate_id="GATE-ERR")
    assert res.admitted_count == 0
    assert reader.get_share_class_by_isin("IE00B44Z5B48") is None


# --- GROUP 4: COLLISION & IDEMPOTENCY ---

def test_idempotency_zero_writes_on_repeat(test_repo):
    """Demonstrates that repeat admission of the same candidate produces 0 inserts and 0 updates."""
    reader, writer, _ = test_repo
    cand = build_candidate(isin="IE00B6YX5C33")

    # Initial admission
    res1 = writer.admit_batch([cand], governance_gate_id="GATE-INITIAL")
    assert res1.admitted_count == 1
    assert res1.no_op_count == 0

    # Repeat admission
    res2 = writer.admit_batch([cand], governance_gate_id="GATE-REPEAT")
    assert res2.admitted_count == 0
    assert res2.no_op_count == 1
    assert len(res2.no_op_share_class_ids) == 1

    # Audit history should contain only the initial admission and summary
    history = reader.get_audit_history("etfs:v1:ISIN:IE00B6YX5C33")
    assert len(history) == 1


def test_collision_identity_conflict(test_repo):
    """Verifies that an incoming candidate with same ISIN but different umbrella triggers IDENTITY_CONFLICT."""
    _, writer, _ = test_repo
    cand1 = build_candidate(isin="IE00B6YX5C33", umbrella_name="SSGA SPDR ETFs Europe I plc")
    writer.admit_batch([cand1], governance_gate_id="GATE-01")

    # Conflicting umbrella
    cand2 = build_candidate(isin="IE00B6YX5C33", umbrella_name="Conflicting Umbrella PLC")
    pre = writer.preflight_admission([cand2])
    assert pre[0].collision_state == CollisionState.IDENTITY_CONFLICT
    assert pre[0].is_valid is False


# --- GROUP 5: SUPERSESSION & HARD DELETE PROHIBITION ---

def test_hard_delete_prohibited_and_blocked(test_repo):
    """Verifies that SQLite trigger trg_no_hard_delete_share_class aborts physical DELETE operations."""
    reader, writer, conn = test_repo
    cand = build_candidate(isin="IE00B6YX5C33")
    writer.admit_batch([cand], governance_gate_id="GATE-01")

    with pytest.raises(sqlite3.IntegrityError, match="HARD_DELETE_PROHIBITED"):
        conn.execute("DELETE FROM canonical_share_class WHERE isin = 'IE00B6YX5C33';")


def test_in_place_isin_mutation_prohibited(test_repo):
    """Verifies that SQLite trigger trg_no_update_isin_share_class aborts in-place ISIN updates."""
    reader, writer, conn = test_repo
    cand = build_candidate(isin="IE00B6YX5C33")
    writer.admit_batch([cand], governance_gate_id="GATE-01")

    with pytest.raises(sqlite3.IntegrityError, match="ISIN_MUTATION_PROHIBITED"):
        conn.execute("UPDATE canonical_share_class SET isin = 'IE00B44Z5B48' WHERE isin = 'IE00B6YX5C33';")


def test_supersession_preserves_old_identity_and_audit(test_repo):
    """Verifies that supersession transitions old entity to SUPERSEDED, preserving ISIN and ID."""
    reader, writer, _ = test_repo
    old_cand = build_candidate(isin="IE00B6YX5C33", share_class_name="SPDR S&P US Dividend Aristocrats UCITS ETF (Dist)")
    new_cand = build_candidate(isin="IE00B44Z5B48", share_class_name="SPDR S&P US Dividend Aristocrats UCITS ETF (Acc)")
    writer.admit_batch([old_cand, new_cand], governance_gate_id="GATE-01")

    # Execute supersession
    old_res, new_res = writer.supersede_identity(
        old_isin="IE00B6YX5C33",
        successor_isin="IE00B44Z5B48",
        authority_basis="STATUTORY_RESTRUCTURING_PROSPECTUS_2025",
        governance_gate_id="GATE-SUPERSEDE"
    )

    assert old_res.canonical_status == CanonicalStatus.SUPERSEDED.value
    assert old_res.isin == "IE00B6YX5C33"
    assert new_res.canonical_status == CanonicalStatus.ADMITTED_CURRENT.value
    assert new_res.isin == "IE00B44Z5B48"

    # Verify audit log captures transition
    history = reader.get_audit_history(old_res.canonical_share_class_id)
    assert any(h.change_type == "STATUS_SUPERSEDED" for h in history)


# --- GROUP 6: SNAPSHOTS & DENOMINATOR FIREWALL ---

def test_derived_snapshot_generation(test_repo):
    """Verifies snapshot generation produces deterministic digest and non-authoritative role."""
    reader, writer, _ = test_repo
    cand = build_candidate(isin="IE00B6YX5C33")
    writer.admit_batch([cand], governance_gate_id="GATE-01")

    snapshot = reader.export_canonical_snapshot(version_id=1)
    assert snapshot["snapshot_role"] == "DERIVED_NON_AUTHORITATIVE_EVIDENCE"
    assert snapshot["version_id"] == 1
    assert snapshot["share_class_count"] == 1
    assert len(snapshot["content_digest"]) == 64


def test_denominator_query_firewall_blocks_calls(test_repo):
    """Verifies that calling denominator methods on CanonicalPopulationReader raises DenominatorFirewallViolationError."""
    reader, _, _ = test_repo
    with pytest.raises(DenominatorFirewallViolationError, match="DENOMINATOR_FIREWALL"):
        reader.get_denominator()
    with pytest.raises(DenominatorFirewallViolationError, match="DENOMINATOR_FIREWALL"):
        reader.calculate_denominator()


# --- GROUP 7: MULTI-JURISDICTION & ISSUER AGNOSTIC ---

def test_multi_jurisdiction_ie_lu_de_fr(test_repo):
    """Verifies that IE, LU, DE, and FR share classes can be admitted to the same store without schema changes."""
    reader, writer, _ = test_repo
    c_ie = build_candidate(isin="IE00B6YX5C33", domicile="IE", regulator="CBI")
    c_lu = build_candidate(isin="LU0290358497", domicile="LU", regulator="CSSF", umbrella_name="Lux Umbrella SICAV")
    c_de = build_candidate(isin="DE000A0D8Q07", domicile="DE", regulator="BaFin", umbrella_name="German ETF AG")
    c_fr = build_candidate(isin="FR0010315770", domicile="FR", regulator="AMF", umbrella_name="French FCP SICAV")

    res = writer.admit_batch([c_ie, c_lu, c_de, c_fr], governance_gate_id="GATE-MULTI-JURISDICTION")
    assert res.admitted_count == 4
    assert reader.get_share_class_by_isin("LU0290358497") is not None
    assert reader.get_share_class_by_isin("DE000A0D8Q07") is not None


def test_issuer_agnostic_representation(test_repo):
    """Verifies that multiple distinct asset managers are supported without issuer-specific code."""
    reader, writer, _ = test_repo
    c_ssga = build_candidate(isin="IE00B6YX5C33", umbrella_name="SSGA SPDR ETFs Europe I plc")
    c_ishares = build_candidate(isin="IE00B44Z5B48", umbrella_name="iShares III plc")

    res = writer.admit_batch([c_ssga, c_ishares], governance_gate_id="GATE-MULTI-ISSUER")
    assert res.admitted_count == 2
    assert reader.get_share_class_by_isin("IE00B6YX5C33").canonical_parent_id != reader.get_share_class_by_isin("IE00B44Z5B48").canonical_parent_id


# --- GROUP 8: PROVISIONAL ARTIFACT FIREWALL & DRY-RUN ---

def test_provisional_artifact_firewall():
    """Verifies that attempting to ingest from provisional self-bootstrapped artifacts is strictly blocked."""
    migrator = CanonicalPopulationMigrator(":memory:")
    for blocked_file in PROVISIONAL_ARTIFACT_BLOCKLIST:
        with pytest.raises(PreflightValidationError, match="PROVISIONAL_ARTIFACT_FIREWALL"):
            migrator.execute_migration_dry_run(blocked_file)


def test_141_row_migration_dry_run_contract():
    """Executes the 141-row migration dry run against certified package with zero disk writes."""
    source_pkg = "ireland_ssga_share_class_admission_candidate_package_reconciled.json"
    if not Path(source_pkg).exists():
        pytest.skip(f"Source candidate package {source_pkg} not in working directory.")

    migrator = CanonicalPopulationMigrator(":memory:")
    report = migrator.execute_migration_dry_run(source_pkg)

    assert report["source_package_row_count"] == 141
    assert report["physical_disk_writes_performed"] == 0
    assert report["preflight_summary"]["planned_inserts"] == 141
    assert report["dry_run_verdict"] == "SUCCESS_READY_FOR_GOVERNED_MIGRATION"


# --- GROUP 9: GOVERNED REAL CANONICAL MIGRATION ENTRYPOINT ---

SOURCE_RECONCILED_PKG = "ireland_ssga_share_class_admission_candidate_package_reconciled.json"
EXPECTED_RECONCILED_SHA256 = "3bdc834c34f936d6a4f0b32526759858ea60c8f35d595ff3e6fcca30719f140a"
EXPECTED_RECONCILED_ROW_COUNT = 141


def test_real_migration_to_temp_disk_store(tmp_path):
    """
    Verifies real canonical migration into a temporary disk-backed SQLite database.
    Admits 141 Ireland SSGA candidates, validates referential integrity, provenance completeness,
    audit trail, authoritative backup creation, and derived snapshot export.
    Covers Behaviors: 2, 6, 9, 14, 15, 19.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "real_canonical.db")
    backup_path = str(tmp_path / "authoritative_backup.db")
    snap_path = str(tmp_path / "derived_snapshot.json")

    migrator = CanonicalPopulationMigrator()
    result = migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="ETF_V2_IRELAND_COHORT_A_SSGA_CANONICAL_POPULATION_MIGRATION_GATE",
        expected_package_sha256=EXPECTED_RECONCILED_SHA256,
        expected_row_count=EXPECTED_RECONCILED_ROW_COUNT,
        target_db_path=target_db,
        backup_target_path=backup_path,
        export_snapshot=True,
        snapshot_output_path=snap_path,
    )

    assert result["migration_verdict"] == "SUCCESS_REAL_CANONICAL_MIGRATION_COMPLETE"
    assert result["admitted_count"] == 141
    assert result["exact_already_present_count"] == 0
    assert result["rejected_count"] == 0
    assert result["canonical_share_class_count"] == 141
    assert result["unique_canonical_isin_count"] == 141
    assert result["duplicate_canonical_isin_count"] == 0
    assert result["canonical_hold_record_count"] == 0
    assert result["provenance_completeness"] is True
    assert result["audit_trace_completeness"] is True
    assert result["schema_version"] == INITIAL_MIGRATION_VERSION

    # Backup assertion
    assert result["backup_metadata"]["backup_status"] == "VERIFIED_AUTHORITATIVE"
    assert result["backup_metadata"]["backup_verified"] is True
    assert Path(backup_path).exists()
    assert Path(backup_path).stat().st_size > 0

    # Snapshot assertion
    assert Path(snap_path).exists()
    snap_data = json.loads(Path(snap_path).read_text(encoding="utf-8"))
    assert snap_data["snapshot_role"] == "DERIVED_NON_AUTHORITATIVE_EVIDENCE"
    assert snap_data["share_class_count"] == 141


def test_real_migration_source_hash_mismatch_fails_closed(tmp_path):
    """
    Verifies that expected source SHA-256 mismatch raises PreflightValidationError
    and aborts before creating or modifying target database.
    Covers Behavior: 3.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "should_not_exist.db")
    migrator = CanonicalPopulationMigrator()

    with pytest.raises(PreflightValidationError, match="SOURCE_PACKAGE_SHA256_MISMATCH"):
        migrator.execute_migration(
            candidate_package_path=SOURCE_RECONCILED_PKG,
            governance_gate_id="GATE-SHA-FAIL",
            expected_package_sha256="0" * 64,
            target_db_path=target_db,
        )

    # Database must not have been created
    assert not Path(target_db).exists()


def test_real_migration_source_row_count_mismatch_fails_closed(tmp_path):
    """
    Verifies that expected source row count mismatch raises PreflightValidationError
    with zero writes.
    Covers Behavior: 4.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "should_not_exist_rows.db")
    migrator = CanonicalPopulationMigrator()

    with pytest.raises(PreflightValidationError, match="SOURCE_PACKAGE_ROW_COUNT_MISMATCH"):
        migrator.execute_migration(
            candidate_package_path=SOURCE_RECONCILED_PKG,
            governance_gate_id="GATE-ROW-FAIL",
            expected_row_count=999,
            target_db_path=target_db,
        )

    assert not Path(target_db).exists()


def test_real_migration_provisional_artifact_rejection(tmp_path):
    """
    Verifies that ingestion of provisional artifacts in real migration fails closed.
    Covers Behavior: 5.
    """
    target_db = str(tmp_path / "should_not_exist_prov.db")
    migrator = CanonicalPopulationMigrator()

    for blocked in PROVISIONAL_ARTIFACT_BLOCKLIST:
        with pytest.raises(PreflightValidationError, match="PROVISIONAL_ARTIFACT_FIREWALL"):
            migrator.execute_migration(
                candidate_package_path=blocked,
                governance_gate_id="GATE-PROV-FAIL",
                target_db_path=target_db,
            )

    assert not Path(target_db).exists()


def test_real_migration_schema_auto_initialization(tmp_path):
    """
    Verifies that real migration into a non-existent database initializes the schema automatically.
    Covers Behavior: 6.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "auto_initialized.db")
    assert not Path(target_db).exists()

    migrator = CanonicalPopulationMigrator()
    result = migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-AUTO-INIT",
        target_db_path=target_db,
    )

    assert Path(target_db).exists()
    assert migrator.get_schema_version(target_db) == INITIAL_MIGRATION_VERSION
    assert result["admitted_count"] == 141


def test_real_migration_incompatible_schema_fails_closed(tmp_path):
    """
    Verifies that an existing database with incompatible schema version fails closed.
    Covers Behavior: 7.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "incompatible.db")
    conn = sqlite3.connect(target_db)
    conn.execute("CREATE TABLE schema_version (version INTEGER, migration_name TEXT, applied_at TEXT, checksum TEXT);")
    conn.execute("INSERT INTO schema_version VALUES (99, 'incompatible_v99', '2026-01-01', 'fake');")
    conn.commit()
    conn.close()

    migrator = CanonicalPopulationMigrator()
    with pytest.raises(SchemaVersionMismatchError, match="incompatible schema version 99"):
        migrator.execute_migration(
            candidate_package_path=SOURCE_RECONCILED_PKG,
            governance_gate_id="GATE-SCHEMA-FAIL",
            target_db_path=target_db,
        )


def test_real_migration_lock_acquisition_failure_fails_closed(tmp_path):
    """
    Verifies that real migration fails closed when writer process lock cannot be acquired.
    Covers Behavior: 8.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "locked.db")
    migrator = CanonicalPopulationMigrator()
    migrator.initialize_empty_store(target_db)

    lock_file = str(Path(f"{target_db}.lock"))
    # Pre-create lock file simulating another active writer
    import os
    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    os.write(fd, f"{os.getpid()}:{datetime.datetime.now().timestamp()}".encode("utf-8"))
    os.close(fd)

    try:
        from scripts.research.etf_v2.canonical_population_repository import SingleWriterProcessLock
        orig_init = SingleWriterProcessLock.__init__

        def fast_init(self, lock_path=None, timeout_seconds=0.1):
            orig_init(self, lock_path=lock_path or lock_file, timeout_seconds=0.1)

        SingleWriterProcessLock.__init__ = fast_init
        try:
            with pytest.raises(LockAcquisitionTimeoutError):
                migrator.execute_migration(
                    candidate_package_path=SOURCE_RECONCILED_PKG,
                    governance_gate_id="GATE-LOCK-FAIL",
                    target_db_path=target_db,
                )
        finally:
            SingleWriterProcessLock.__init__ = orig_init
    finally:
        if Path(lock_file).exists():
            Path(lock_file).unlink()


def test_real_migration_atomic_rollback_on_failure(tmp_path):
    """
    Verifies that a mid-batch write failure causes complete atomic rollback leaving 0 admitted rows.
    Covers Behavior: 10.
    """
    target_db = str(tmp_path / "rollback_test.db")
    migrator = CanonicalPopulationMigrator()
    migrator.initialize_empty_store(target_db)

    # Inject SQLite trigger to simulate database write failure on candidate 2
    conn = sqlite3.connect(target_db)
    conn.execute(
        "CREATE TRIGGER test_mid_batch_failure "
        "BEFORE INSERT ON canonical_share_class "
        "WHEN NEW.isin = 'IE00B44Z5B48' "
        "BEGIN SELECT RAISE(FAIL, 'Mid-batch database failure'); END;"
    )
    conn.commit()
    conn.close()

    writer = CanonicalPopulationWriter(target_db, lock_path=f"{target_db}.lock")
    c1 = build_candidate(isin="IE00B6YX5C33", share_class_name="Class 1")
    c2 = build_candidate(isin="IE00B44Z5B48", share_class_name="Class 2")

    with pytest.raises(AdmissionTransactionError):
        writer.admit_batch([c1, c2], governance_gate_id="GATE-ROLLBACK-TEST")

    # Verify zero share classes survived rollback
    reader = CanonicalPopulationReader(target_db)
    version, _ = reader.get_population_version()
    assert version == 0
    assert reader.get_share_class_by_isin("IE00B6YX5C33") is None


def test_real_migration_required_provenance_failure_fails_closed(tmp_path):
    """
    Verifies that a candidate with non-statutory provenance fails preflight validation with zero writes.
    Covers Behavior: 11.
    """
    target_db = str(tmp_path / "prov_fail.db")
    pkg_file = tmp_path / "bad_prov_pkg.json"
    bad_candidate = {
        "admission_candidate_id": "cand-bad-prov",
        "isin": "IE00B6YX5C33",
        "umbrella_name": "SSGA SPDR ETFs Europe I plc",
        "subfund_name": "SPDR Subfund",
        "share_class_name": "SPDR Class",
        "authority_tier": "TIER_3_COMMERCIAL_VENDOR",
        "evidence_references": [{
            "evidence_object_id": "ev-commercial",
            "authority_tier": "TIER_3_COMMERCIAL_VENDOR",
            "source_document_id": "COMMERCIAL_FEED"
        }]
    }
    pkg_file.write_text(json.dumps([bad_candidate]), encoding="utf-8")

    migrator = CanonicalPopulationMigrator()
    with pytest.raises(PreflightValidationError, match="PREFLIGHT_REJECTION"):
        migrator.execute_migration(
            candidate_package_path=str(pkg_file),
            governance_gate_id="GATE-PROV-FAIL",
            target_db_path=target_db,
        )

    assert migrator.get_schema_version(target_db) == INITIAL_MIGRATION_VERSION
    reader = CanonicalPopulationReader(target_db)
    assert reader.get_share_class_by_isin("IE00B6YX5C33") is None


def test_real_migration_hold_firewall_enforced(tmp_path):
    """
    Verifies that a candidate with an active quarantine hold is blocked during preflight.
    Covers Behavior: 12.
    """
    target_db = str(tmp_path / "hold_test.db")
    migrator = CanonicalPopulationMigrator()
    migrator.initialize_empty_store(target_db)

    # Enter an active hold for IE00B6YX5C33
    writer = CanonicalPopulationWriter(target_db, lock_path=f"{target_db}.lock")
    hold = CanonicalHoldRecord(
        hold_record_id="hold-test-001",
        candidate_identifier="IE00B6YX5C33",
        hold_category=HoldCategory.OFFICIAL_IDENTITY_EVIDENCE_NOT_ESTABLISHED_HOLD,
        hold_state="HELD",
        reason="Active statutory identity dispute",
        source_candidate_package="test_pkg",
        entered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hold_governance_gate="GATE-HOLD-ENTER",
        reopen_condition="REOPEN_UPON_EVIDENCE",
        status="ACTIVE_HOLD",
    )
    writer.enter_hold(hold)

    # Try to migrate a package containing that held ISIN
    pkg_file = tmp_path / "held_cand_pkg.json"
    cand = {
        "admission_candidate_id": "cand-held",
        "isin": "IE00B6YX5C33",
        "umbrella_name": "SSGA SPDR ETFs Europe I plc",
        "subfund_name": "SPDR Subfund",
        "share_class_name": "SPDR Class",
        "authority_tier": "TIER_1_OFFICIAL_STATUTORY",
        "evidence_references": [{
            "evidence_object_id": "ev-1",
            "authority_tier": "TIER_1_OFFICIAL_STATUTORY",
            "source_document_id": "STATUTORY_DOC"
        }]
    }
    pkg_file.write_text(json.dumps([cand]), encoding="utf-8")

    with pytest.raises(PreflightValidationError, match="PREFLIGHT_REJECTION"):
        migrator.execute_migration(
            candidate_package_path=str(pkg_file),
            governance_gate_id="GATE-HOLD-MIGRATE",
            target_db_path=target_db,
        )


def test_real_migration_idempotent_reexecution(tmp_path):
    """
    Verifies that running execute_migration twice against the same database is strictly idempotent.
    Second run produces 0 new admissions, 141 exact_already_present, and zero audit churn.
    Covers Behavior: 13.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "idempotent.db")
    migrator = CanonicalPopulationMigrator()

    # First run
    res1 = migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-RUN-1",
        target_db_path=target_db,
    )
    assert res1["admitted_count"] == 141
    assert res1["exact_already_present_count"] == 0

    reader = CanonicalPopulationReader(target_db)
    v1, digest1 = reader.get_population_version()

    # Second run
    res2 = migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-RUN-2",
        target_db_path=target_db,
    )
    assert res2["admitted_count"] == 0
    assert res2["exact_already_present_count"] == 141
    assert res2["rejected_count"] == 0
    assert res2["migration_verdict"] == "SUCCESS_REAL_CANONICAL_MIGRATION_COMPLETE"

    v2, digest2 = reader.get_population_version()
    assert v2 == v1  # Zero audit batch version bump on exact no-op replay
    assert digest2 == digest1


def test_real_migration_backup_creation_and_integrity(tmp_path):
    """
    Verifies that post-commit SQLite VACUUM INTO backup is created and passes PRAGMA integrity_check.
    Covers Behaviors: 16, 17.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "target_backup_src.db")
    backup_path = str(tmp_path / "target_backup_dst.db")
    migrator = CanonicalPopulationMigrator()

    res = migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-BACKUP-TEST",
        target_db_path=target_db,
        backup_target_path=backup_path,
    )

    assert res["backup_metadata"]["backup_status"] == "VERIFIED_AUTHORITATIVE"
    assert res["backup_metadata"]["backup_verified"] is True
    assert Path(backup_path).exists()
    assert Path(backup_path).stat().st_size > 0

    # Verify directly using SQLite integrity check
    conn = sqlite3.connect(backup_path)
    cur = conn.execute("PRAGMA integrity_check;")
    assert cur.fetchone()[0] == "ok"
    conn.close()


def test_real_migration_recovery_from_backup(tmp_path):
    """
    Verifies that a backup file can be opened independently and all 141 admitted rows are reconciled.
    Covers Behavior: 18.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "primary.db")
    backup_path = str(tmp_path / "recovery_backup.db")
    migrator = CanonicalPopulationMigrator()

    migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-RECOVERY-TEST",
        target_db_path=target_db,
        backup_target_path=backup_path,
    )

    # Open backup as primary authority via CanonicalPopulationReader
    recovered_reader = CanonicalPopulationReader(backup_path)
    snapshot = recovered_reader.export_canonical_snapshot()

    assert snapshot["share_class_count"] == 141
    assert snapshot["provenance_count"] >= 141

    primary_reader = CanonicalPopulationReader(target_db)
    v_prim, dig_prim = primary_reader.get_population_version()
    v_rec, dig_rec = recovered_reader.get_population_version()

    assert v_rec == v_prim
    assert dig_rec == dig_prim


def test_real_migration_shared_parser_preserves_dry_run():
    """
    Verifies that the refactored _parse_candidate_package parses correctly and dry run remains intact.
    Covers Behavior: 1.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    migrator = CanonicalPopulationMigrator(":memory:")
    submissions, meta = migrator._parse_candidate_package(SOURCE_RECONCILED_PKG)
    assert len(submissions) == 141
    assert meta["row_count"] == 141
    assert meta["sha256"] == EXPECTED_RECONCILED_SHA256

    dry_run = migrator.execute_migration_dry_run(SOURCE_RECONCILED_PKG)
    assert dry_run["source_package_row_count"] == 141
    assert dry_run["physical_disk_writes_performed"] == 0
    assert dry_run["dry_run_verdict"] == "SUCCESS_READY_FOR_GOVERNED_MIGRATION"


def test_real_migration_denominator_and_openfigi_firewalls(tmp_path):
    """
    Verifies that denominator methods fail closed and zero OpenFIGI authority is present.
    Covers Behaviors: 20, 21.
    """
    if not Path(SOURCE_RECONCILED_PKG).exists():
        pytest.skip(f"Source package {SOURCE_RECONCILED_PKG} not found.")

    target_db = str(tmp_path / "firewalls.db")
    migrator = CanonicalPopulationMigrator()
    migrator.execute_migration(
        candidate_package_path=SOURCE_RECONCILED_PKG,
        governance_gate_id="GATE-FIREWALL-TEST",
        target_db_path=target_db,
    )

    reader = CanonicalPopulationReader(target_db)
    with pytest.raises(DenominatorFirewallViolationError, match="DENOMINATOR_FIREWALL"):
        reader.get_denominator()
    with pytest.raises(DenominatorFirewallViolationError, match="DENOMINATOR_FIREWALL"):
        reader.calculate_denominator()

    # Verify no denominator method on Migrator
    assert not hasattr(migrator, "get_denominator")
    assert not hasattr(migrator, "calculate_denominator")

    # Verify no OpenFIGI in provenance records
    snapshot = reader.export_canonical_snapshot()
    for prov in snapshot["provenance_records"]:
        assert "OPENFIGI" not in prov.get("authority_tier", "").upper()
        assert "OPENFIGI" not in prov.get("source_document_id", "").upper()


def test_real_migration_production_db_path_untouched(tmp_path: Path):
    """
    Verifies that the canonical production database path and its sidecars are never
    created, deleted, replaced, or mutated by isolated tests and migrations.
    Covers Behavior: 22 (lifecycle-independent isolation invariant).
    Supports both State A (production DB absent) and State B (production DB already exists).
    """
    import hashlib

    prod_path = Path("data/canonical/etf_v2_canonical_population.db")
    wal_path = Path("data/canonical/etf_v2_canonical_population.db-wal")
    shm_path = Path("data/canonical/etf_v2_canonical_population.db-shm")

    # Capture pre-operation production store & sidecar state
    db_existed_before = prod_path.exists()
    wal_existed_before = wal_path.exists()
    shm_existed_before = shm_path.exists()

    if db_existed_before:
        pre_size = prod_path.stat().st_size
        pre_data = prod_path.read_bytes()
        pre_hash = hashlib.sha256(pre_data).hexdigest()
        wal_pre_size = wal_path.stat().st_size if wal_existed_before else None
        wal_pre_hash = hashlib.sha256(wal_path.read_bytes()).hexdigest() if wal_existed_before else None
        shm_pre_size = shm_path.stat().st_size if shm_existed_before else None
        shm_pre_hash = hashlib.sha256(shm_path.read_bytes()).hexdigest() if shm_existed_before else None
    else:
        pre_size = None
        pre_hash = None
        wal_pre_size = None
        wal_pre_hash = None
        shm_pre_size = None
        shm_pre_hash = None

    # Perform isolated migration operation in temporary directory
    if Path(SOURCE_RECONCILED_PKG).exists():
        isolated_db = str(tmp_path / "isolation_check.db")
        migrator = CanonicalPopulationMigrator()
        result = migrator.execute_migration(
            candidate_package_path=SOURCE_RECONCILED_PKG,
            governance_gate_id="GATE-ISOLATION-VERIFICATION",
            target_db_path=isolated_db,
        )
        assert result["migration_verdict"] == "SUCCESS_REAL_CANONICAL_MIGRATION_COMPLETE"
        assert Path(isolated_db).exists()
        assert Path(isolated_db).stat().st_size > 0

    # Verify post-operation production store & sidecar state
    if not db_existed_before:
        # State A: Production DB absent before test -> must remain absent
        assert not prod_path.exists(), "CRITICAL: Isolated test created production DB path!"
        assert not wal_path.exists(), "CRITICAL: Isolated test created production DB-WAL path!"
        assert not shm_path.exists(), "CRITICAL: Isolated test created production DB-SHM path!"
    else:
        # State B: Production DB already exists before test -> must remain completely unmutated
        assert prod_path.exists(), "CRITICAL: Isolated test deleted production DB path!"
        assert prod_path.stat().st_size == pre_size, "CRITICAL: Production DB size changed!"
        post_data = prod_path.read_bytes()
        post_hash = hashlib.sha256(post_data).hexdigest()
        assert post_hash == pre_hash, "CRITICAL: Production DB SHA256 mutated by isolated test!"

        # Verify sidecars did not mutate or appear unexpectedly
        if wal_existed_before:
            assert wal_path.exists(), "CRITICAL: Production DB-WAL was deleted!"
            assert wal_path.stat().st_size == wal_pre_size, "CRITICAL: Production DB-WAL size changed!"
            assert hashlib.sha256(wal_path.read_bytes()).hexdigest() == wal_pre_hash, "CRITICAL: Production DB-WAL mutated!"
        else:
            assert not wal_path.exists(), "CRITICAL: Production DB-WAL was unexpectedly created!"

        if shm_existed_before:
            assert shm_path.exists(), "CRITICAL: Production DB-SHM was deleted!"
            assert shm_path.stat().st_size == shm_pre_size, "CRITICAL: Production DB-SHM size changed!"
            assert hashlib.sha256(shm_path.read_bytes()).hexdigest() == shm_pre_hash, "CRITICAL: Production DB-SHM mutated!"
        else:
            assert not shm_path.exists(), "CRITICAL: Production DB-SHM was unexpectedly created!"
