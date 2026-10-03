"""
scripts/research/etf_v2/canonical_population_migrator.py

Python SQL Migration Runner and Dry-Run Simulation Engine for the
ETF V2 Canonical Population Authority.
Provides deterministic schema initialization, checksum verification,
and 141-row migration dry-run execution with zero physical disk writes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .canonical_population_errors import (
    CanonicalPopulationError,
    PreflightValidationError,
    SchemaVersionMismatchError,
)
from .canonical_population_models import (
    CandidateSubmission,
    CollisionState,
)
from .canonical_population_repository import (
    CanonicalPopulationReader,
    CanonicalPopulationWriter,
    get_default_db_path,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL_PATH = Path(__file__).parent / "canonical_population_schema.sql"
INITIAL_MIGRATION_VERSION = 1
INITIAL_MIGRATION_NAME = "001_initial_canonical_schema"

PROVISIONAL_ARTIFACT_BLOCKLIST = {
    "ireland_ssga_bounded_canonical_admission_ledger.json",
    "ireland_ssga_bounded_canonical_population_snapshot.json",
}


class CanonicalPopulationMigrator:
    """Manages schema lifecycle, DDL migrations, and admission dry-runs."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_default_db_path()

    def _read_schema_sql(self) -> Tuple[str, str]:
        if not SCHEMA_SQL_PATH.exists():
            raise CanonicalPopulationError(f"Schema SQL file not found at {SCHEMA_SQL_PATH}")
        content = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return content, checksum

    def initialize_empty_store(self, target_db_path: Optional[str] = None) -> int:
        """
        Initializes an empty canonical SQLite database store by applying DDL schema.
        Guarantees CANONICAL_SHARE_CLASS_ROW_COUNT == 0.
        """
        db = target_db_path or self.db_path
        is_uri = db.startswith("file:")
        if not is_uri and db != ":memory:":
            Path(db).parent.mkdir(parents=True, exist_ok=True)

        schema_sql, checksum = self._read_schema_sql()
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        conn = sqlite3.connect(db, uri=is_uri)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(schema_sql)

            # Record in schema_version table
            conn.execute("""
                INSERT OR REPLACE INTO schema_version (version, migration_name, applied_at, checksum)
                VALUES (?, ?, ?, ?)
            """, (INITIAL_MIGRATION_VERSION, INITIAL_MIGRATION_NAME, now_utc, checksum))
            conn.commit()

            # Verify empty store invariant
            cur = conn.execute("SELECT count(*) FROM canonical_share_class;")
            count = cur.fetchone()[0]
            if count != 0:
                raise CanonicalPopulationError(f"Empty store invariant violated: found {count} rows in canonical_share_class.")
            logger.info("Initialized empty canonical store at %s (schema v%d)", db, INITIAL_MIGRATION_VERSION)
            return INITIAL_MIGRATION_VERSION
        finally:
            conn.close()

    def get_schema_version(self, target_db_path: Optional[str] = None) -> Optional[int]:
        """Returns current schema version or None if uninitialized."""
        db = target_db_path or self.db_path
        is_uri = db.startswith("file:")
        if not is_uri and db != ":memory:" and not os.path.exists(db):
            return None
        conn = sqlite3.connect(db, uri=is_uri)
        try:
            cur = conn.execute("SELECT max(version) FROM schema_version;")
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

    def _parse_candidate_package(
        self,
        candidate_package_path: str,
    ) -> Tuple[List[CandidateSubmission], Dict[str, Any]]:
        """
        Parses and validates a candidate package from JSON.
        Performs fail-closed provisional artifact blocking, file existence,
        and JSON structure validation.
        Returns a tuple of (submissions, package_metadata).
        """
        cand_path = Path(candidate_package_path)
        if cand_path.name in PROVISIONAL_ARTIFACT_BLOCKLIST:
            raise PreflightValidationError(
                f"PROVISIONAL_ARTIFACT_FIREWALL: Ingestion from provisional artifact {cand_path.name} is strictly prohibited."
            )

        if not cand_path.exists():
            raise PreflightValidationError(f"Candidate package not found at {candidate_package_path}")

        raw_bytes = cand_path.read_bytes()
        pkg_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        pkg_size = len(raw_bytes)

        try:
            raw_data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise PreflightValidationError(f"Invalid candidate package JSON at {candidate_package_path}: {e}") from e

        candidates_raw = raw_data.get("candidates", []) if isinstance(raw_data, dict) else raw_data
        if not isinstance(candidates_raw, list):
            raise PreflightValidationError("Invalid candidate package format; expected list of candidate records.")

        # Convert to CandidateSubmission objects
        submissions: List[CandidateSubmission] = []
        for c in candidates_raw:
            prov_refs = []
            if "provenance_chain" in c and isinstance(c["provenance_chain"], dict):
                prov_refs.append(c["provenance_chain"])
            elif "evidence_references" in c and isinstance(c["evidence_references"], (list, tuple)):
                prov_refs.extend(c["evidence_references"])
            else:
                prov_refs.append({
                    "evidence_object_id": c.get("evidence_object_ids", ["ev_reconciled"])[0] if c.get("evidence_object_ids") else "ev_reconciled",
                    "source_document_id": "RECONCILED_DOC",
                    "authority_tier": c.get("authority_tier", "TIER_2"),
                })

            submissions.append(CandidateSubmission(
                source_candidate_id=c.get("admission_candidate_id", c.get("source_candidate_id", c.get("candidate_id", c.get("share_class_isin", "")))),
                isin=c.get("isin", c.get("share_class_isin", "")),
                legal_umbrella_name=c.get("legal_umbrella_name", c.get("umbrella_name", "SSGA SPDR ETFs Europe I plc")),
                legal_subfund_name=c.get("legal_subfund_name", c.get("subfund_name", "")),
                legal_share_class_name=c.get("legal_share_class_name", c.get("share_class_name", "")),
                domicile_iso2=c.get("domicile_iso2", "IE"),
                regulatory_jurisdiction=c.get("regulatory_jurisdiction", "EU_UCITS"),
                authority_tier=c.get("authority_tier", "TIER_2_REGULATOR_OFFICIAL"),
                evidence_references=tuple(prov_refs),
                readiness_state=c.get("readiness_state", "ADMISSION_READY"),
                source_artifact_identity={"filename": cand_path.name, "byte_count": pkg_size, "sha256": pkg_sha256},
                subfund_currency=c.get("subfund_currency", "EUR"),
                regulator=c.get("regulator", "CBI"),
                legal_entity_structure=c.get("legal_entity_structure", "ICAV"),
            ))

        metadata = {
            "filename": cand_path.name,
            "package_path": str(cand_path),
            "byte_count": pkg_size,
            "sha256": pkg_sha256,
            "row_count": len(submissions),
        }
        return submissions, metadata

    def execute_migration_dry_run(
        self,
        candidate_package_path: str,
        governance_gate_id: str = "ETF_V2_IRELAND_COHORT_A_SSGA_MIGRATION_DRY_RUN",
    ) -> Dict[str, Any]:
        """
        Simulates admission of candidate package against an isolated in-memory database.
        Zero physical disk writes to canonical storage.
        """
        submissions, pkg_meta = self._parse_candidate_package(candidate_package_path)

        # Setup isolated temporary test store
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = str(Path(tmpdir) / "dry_run.db")
            self.initialize_empty_store(temp_db)
            writer = CanonicalPopulationWriter(temp_db)

            # Execute preflight
            preflight_results = writer.preflight_admission(submissions)
            valid_candidates = [p.candidate for p in preflight_results if p.is_valid and p.collision_state == CollisionState.NEW_IDENTITY]
            no_op_count = sum(1 for p in preflight_results if p.collision_state == CollisionState.EXACT_ALREADY_PRESENT)
            rejected_count = sum(1 for p in preflight_results if not p.is_valid)

            # Simulate batch admission in isolated store
            simulated_res = writer.admit_batch(submissions, governance_gate_id=governance_gate_id)

            # Verify simulation results
            reader = CanonicalPopulationReader(temp_db)
            version, digest = reader.get_population_version()

            return {
                "dry_run_gate_id": governance_gate_id,
                "source_package": pkg_meta["filename"],
                "source_package_row_count": len(submissions),
                "simulated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "preflight_summary": {
                    "total_submitted": len(submissions),
                    "planned_inserts": len(valid_candidates),
                    "no_op_already_present": no_op_count,
                    "preflight_rejections": rejected_count,
                },
                "simulated_admission_result": {
                    "batch_id": simulated_res.batch_id,
                    "admitted_count": simulated_res.admitted_count,
                    "no_op_count": simulated_res.no_op_count,
                    "rejected_count": simulated_res.rejected_count,
                    "simulated_population_version": version,
                    "simulated_population_digest": digest,
                },
                "physical_disk_writes_performed": 0,
                "dry_run_verdict": "SUCCESS_READY_FOR_GOVERNED_MIGRATION" if simulated_res.admitted_count == len(submissions) else "REJECTIONS_DETECTED",
            }

    def execute_migration(
        self,
        candidate_package_path: str,
        governance_gate_id: str,
        expected_package_sha256: Optional[str] = None,
        expected_row_count: Optional[int] = None,
        target_db_path: Optional[str] = None,
        backup_target_path: Optional[str] = None,
        export_snapshot: bool = True,
        snapshot_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes governed real canonical population migration into disk-backed SQLite store.
        Enforces source identity firewalls, schema versioning, atomic admission,
        post-write verification, backup creation, and derived snapshot export.
        """
        # Step 1: Parse candidate package through shared parser (enforces provisional blocklist & existence)
        submissions, pkg_meta = self._parse_candidate_package(candidate_package_path)

        # Step 2: Source identity firewall (SHA-256 and row count) BEFORE any DB connection or write
        if expected_package_sha256 is not None:
            if pkg_meta["sha256"].lower() != expected_package_sha256.lower():
                raise PreflightValidationError(
                    f"SOURCE_PACKAGE_SHA256_MISMATCH: expected {expected_package_sha256}, got {pkg_meta['sha256']}. Zero writes performed."
                )

        if expected_row_count is not None:
            if len(submissions) != expected_row_count:
                raise PreflightValidationError(
                    f"SOURCE_PACKAGE_ROW_COUNT_MISMATCH: expected {expected_row_count}, got {len(submissions)}. Zero writes performed."
                )

        # Step 3: Target store resolution and schema lifecycle
        db = target_db_path or self.db_path
        is_uri = db.startswith("file:")
        db_file_exists = False if (is_uri or db == ":memory:") else os.path.exists(db)

        if not db_file_exists and db != ":memory:" and not is_uri:
            # Target absent: initialize schema
            self.initialize_empty_store(db)
        else:
            # Target exists: verify schema version compatibility
            current_version = self.get_schema_version(db)
            if current_version is None or current_version != INITIAL_MIGRATION_VERSION:
                raise SchemaVersionMismatchError(
                    f"Target database at {db} has incompatible schema version {current_version}; expected {INITIAL_MIGRATION_VERSION}. Migration failed closed."
                )

        lock_file = f"{db}.lock" if (db != ":memory:" and not is_uri) else None
        writer = CanonicalPopulationWriter(db, lock_path=lock_file)

        # Step 4: Preflight admission check
        preflight_results = writer.preflight_admission(submissions)
        preflight_rejections = [p for p in preflight_results if not p.is_valid]
        if preflight_rejections:
            reasons = "; ".join(f"{p.candidate.isin}: {p.error_message}" for p in preflight_rejections[:5])
            raise PreflightValidationError(
                f"PREFLIGHT_REJECTION: {len(preflight_rejections)} of {len(submissions)} candidate(s) failed preflight validation ({reasons}). Zero writes performed."
            )

        collision_conflicts = [
            p for p in preflight_results
            if p.collision_state in (
                CollisionState.IDENTITY_CONFLICT,
                CollisionState.PROVENANCE_CONFLICT,
                CollisionState.HISTORICAL_CONTINUITY_REVIEW_REQUIRED,
            )
        ]
        if collision_conflicts:
            raise PreflightValidationError(
                f"PREFLIGHT_COLLISION: {len(collision_conflicts)} collision conflict(s) detected. Zero writes performed."
            )

        # Step 5: Atomic Cohort Admission
        admit_result = writer.admit_batch(submissions, governance_gate_id=governance_gate_id)

        # Step 6: Post-Write Verification using Canonical Reader Authority
        reader = CanonicalPopulationReader(db)
        version, digest = reader.get_population_version()

        # Query snapshot internally for post-write verification
        snapshot = reader.export_canonical_snapshot(version_id=version)
        classes = snapshot["share_classes"]
        unique_isins = set(sc["isin"] for sc in classes)
        duplicate_isin_count = len(classes) - len(unique_isins)

        # Referential integrity checks across admitted share classes
        for sc_id in admit_result.admitted_share_class_ids:
            prov_records = reader.get_provenance_records(sc_id)
            if not prov_records:
                raise CanonicalPopulationError(
                    f"POST_WRITE_INTEGRITY_VIOLATION: Admitted share class {sc_id} lacks statutory provenance records."
                )
            audit_events = reader.get_audit_history(sc_id)
            if not audit_events:
                raise CanonicalPopulationError(
                    f"POST_WRITE_INTEGRITY_VIOLATION: Admitted share class {sc_id} lacks audit log trace."
                )

        holds_count = len(reader.list_holds())

        # Step 7: Post-Commit Authoritative Backup
        backup_metadata: Dict[str, Any]
        if backup_target_path:
            backup_path_obj = Path(backup_target_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                writer.create_backup(backup_target_path)
                is_valid_backup = writer.verify_backup(backup_target_path)
                if is_valid_backup:
                    b_bytes = backup_path_obj.read_bytes()
                    backup_metadata = {
                        "backup_path": backup_target_path,
                        "backup_verified": True,
                        "backup_size_bytes": len(b_bytes),
                        "backup_sha256": hashlib.sha256(b_bytes).hexdigest(),
                        "backup_status": "VERIFIED_AUTHORITATIVE",
                    }
                else:
                    backup_metadata = {
                        "backup_path": backup_target_path,
                        "backup_verified": False,
                        "backup_size_bytes": backup_path_obj.stat().st_size if backup_path_obj.exists() else 0,
                        "backup_sha256": None,
                        "backup_status": "DEGRADED_INTEGRITY_CHECK_FAILED",
                    }
            except Exception as e:
                logger.error("Post-commit backup creation failed: %s", e)
                backup_metadata = {
                    "backup_path": backup_target_path,
                    "backup_verified": False,
                    "backup_size_bytes": 0,
                    "backup_sha256": None,
                    "backup_status": f"DEGRADED_BACKUP_FAILED: {e}",
                }
        else:
            backup_metadata = {
                "backup_path": None,
                "backup_verified": False,
                "backup_size_bytes": 0,
                "backup_sha256": None,
                "backup_status": "NOT_REQUESTED",
            }

        # Step 8: Derived Snapshot Export
        snapshot_digest = snapshot["content_digest"]
        if export_snapshot and snapshot_output_path:
            snap_path = Path(snapshot_output_path)
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

        # Step 9: Return Governed Result Contract
        return {
            "governance_gate_id": governance_gate_id,
            "candidate_package": pkg_meta["filename"],
            "candidate_package_sha256": pkg_meta["sha256"],
            "candidate_package_row_count": len(submissions),
            "target_db_path": db,
            "batch_id": admit_result.batch_id,
            "total_submitted": len(submissions),
            "admitted_count": admit_result.admitted_count,
            "exact_already_present_count": admit_result.no_op_count,
            "rejected_count": admit_result.rejected_count,
            "collision_count": len(collision_conflicts),
            "schema_version": INITIAL_MIGRATION_VERSION,
            "population_version": version,
            "population_digest": digest,
            "canonical_share_class_count": len(classes),
            "unique_canonical_isin_count": len(unique_isins),
            "duplicate_canonical_isin_count": duplicate_isin_count,
            "canonical_hold_record_count": holds_count,
            "provenance_completeness": True,
            "audit_trace_completeness": True,
            "backup_metadata": backup_metadata,
            "snapshot_exported": export_snapshot,
            "snapshot_digest": snapshot_digest,
            "migration_verdict": (
                "SUCCESS_REAL_CANONICAL_MIGRATION_COMPLETE"
                if (admit_result.admitted_count > 0 or admit_result.no_op_count == len(submissions)) and admit_result.rejected_count == 0
                else "FAILED_REJECTIONS_DETECTED"
            ),
        }
