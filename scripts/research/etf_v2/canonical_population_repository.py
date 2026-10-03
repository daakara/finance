"""
scripts/research/etf_v2/canonical_population_repository.py

ACID SQLite Repository implementation for the ETF V2 Canonical Population Authority.
Provides strict separation between Reader and Writer interfaces, enforces
atomic batch rollbacks, single-writer process locking, and immutability invariants.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
import uuid

from .canonical_population_errors import (
    AdmissionTransactionError,
    CanonicalPopulationError,
    DenominatorFirewallViolationError,
    DuplicateIdentityError,
    HardDeleteProhibitedError,
    HeldCandidateError,
    IdentityConflictError,
    InvalidISINError,
    InvalidStateTransitionError,
    LockAcquisitionTimeoutError,
    MissingProvenanceError,
    PopulationAuthorityUnavailableError,
    PreflightValidationError,
    ProvenanceConflictError,
    SchemaVersionMismatchError,
)
from .canonical_population_models import (
    AdmissionState,
    AuthorityTier,
    BatchAdmissionResult,
    CandidateSubmission,
    CanonicalAuditEvent,
    CanonicalHoldRecord,
    CanonicalParentEntity,
    CanonicalProvenanceRecord,
    CanonicalShareClass,
    CanonicalStatus,
    CanonicalSubfund,
    CollisionState,
    CurrentnessDimensions,
    HoldCategory,
    HoldState,
    PopulationScope,
    PreflightClassification,
)
from .global_identifier_authority import (
    IdentifierType,
    InvalidIdentifierError,
    generate_share_class_id,
    normalize_isin,
    validate_isin,
)

logger = logging.getLogger(__name__)

DEFAULT_CANONICAL_DB_PATH = "data/canonical/etf_v2_canonical_population.db"
DEFAULT_LOCK_PATH = "data/canonical/.canonical_population.lock"
EXPECTED_SCHEMA_VERSION = 1


def get_default_db_path() -> str:
    """Returns database path from environment or repository-relative default."""
    return os.environ.get("ETF_V2_CANONICAL_DB_PATH", DEFAULT_CANONICAL_DB_PATH)


def get_default_lock_path() -> str:
    """Returns lock path from environment or repository-relative default."""
    return os.environ.get("ETF_V2_CANONICAL_LOCK_PATH", DEFAULT_LOCK_PATH)


def generate_parent_id(domicile: str, name: str) -> str:
    clean_name = re.sub(r"[^A-Z0-9_-]", "_", name.strip().upper())
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return f"etfp:v1:{domicile.strip().upper()}:{clean_name}"


def generate_subfund_id(parent_id: str, name: str) -> str:
    clean_name = re.sub(r"[^A-Z0-9_-]", "_", name.strip().upper())
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return f"etfsf:v1:{parent_id}:{clean_name}"


class SingleWriterProcessLock:
    """Provides exclusive file-based locking with PID tracking and stale lock recovery."""

    def __init__(self, lock_path: str = DEFAULT_LOCK_PATH, timeout_seconds: float = 30.0):
        self.lock_path = Path(lock_path)
        self.timeout_seconds = timeout_seconds
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            try:
                # O_CREAT | O_EXCL ensures atomic creation
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, f"{os.getpid()}:{time.time()}".encode("utf-8"))
                self._fd = fd
                return
            except FileExistsError:
                # Check for stale lock (> 5 minutes old)
                try:
                    mtime = self.lock_path.stat().st_mtime
                    if time.time() - mtime > 300:
                        logger.warning("Stale process lock detected at %s. Overriding.", self.lock_path)
                        self.release_force()
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        raise LockAcquisitionTimeoutError(
            f"Could not acquire canonical population lock {self.lock_path} within {self.timeout_seconds}s."
        )

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except OSError:
            pass

    def release_force(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except OSError:
            pass

    def __enter__(self) -> SingleWriterProcessLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class CanonicalPopulationReader:
    """Read-only query interface for Canonical Population Authority."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_default_db_path()

    @contextlib.contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        is_uri = self.db_path.startswith("file:")
        if not is_uri and self.db_path != ":memory:" and not os.path.exists(self.db_path):
            raise PopulationAuthorityUnavailableError(f"Database file does not exist: {self.db_path}")
        conn = sqlite3.connect(self.db_path, uri=is_uri, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def get_share_class_by_isin(self, isin: str) -> Optional[CanonicalShareClass]:
        clean_isin = normalize_isin(isin)
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_share_class WHERE isin = ?", (clean_isin,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return CanonicalShareClass(**dict(row))

    def get_share_class_by_id(self, canonical_id: str) -> Optional[CanonicalShareClass]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_share_class WHERE canonical_share_class_id = ?", (canonical_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            return CanonicalShareClass(**dict(row))

    def list_share_classes_by_subfund(self, subfund_id: str) -> List[CanonicalShareClass]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_share_class WHERE canonical_subfund_id = ? ORDER BY legal_share_class_name",
                (subfund_id,)
            )
            return [CanonicalShareClass(**dict(r)) for r in cur.fetchall()]

    def list_share_classes_by_parent(self, parent_id: str) -> List[CanonicalShareClass]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_share_class WHERE canonical_parent_id = ? ORDER BY legal_share_class_name",
                (parent_id,)
            )
            return [CanonicalShareClass(**dict(r)) for r in cur.fetchall()]

    def query_population_by_scope(self, scope: PopulationScope) -> List[CanonicalShareClass]:
        with self._get_connection() as conn:
            query = """
                SELECT sc.* FROM canonical_share_class sc
                JOIN canonical_parent_entity p ON sc.canonical_parent_id = p.canonical_parent_id
                WHERE sc.jurisdiction = ?
                  AND sc.regulatory_framework = ?
                  AND p.regulator = ?
                ORDER BY sc.isin
            """
            cur = conn.execute(query, (scope.jurisdiction, scope.regulatory_framework, scope.regulator))
            return [CanonicalShareClass(**dict(r)) for r in cur.fetchall()]

    def get_provenance_records(self, canonical_id: str) -> List[CanonicalProvenanceRecord]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_provenance_records WHERE canonical_share_class_id = ? ORDER BY created_at",
                (canonical_id,)
            )
            return [CanonicalProvenanceRecord(**dict(r)) for r in cur.fetchall()]

    def get_audit_history(self, canonical_id: str) -> List[CanonicalAuditEvent]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM canonical_audit_log WHERE canonical_share_class_id = ? ORDER BY recorded_at",
                (canonical_id,)
            )
            return [CanonicalAuditEvent(**dict(r)) for r in cur.fetchall()]

    def list_holds(self, category: Optional[HoldCategory] = None) -> List[CanonicalHoldRecord]:
        with self._get_connection() as conn:
            if category:
                cur = conn.execute(
                    "SELECT * FROM canonical_hold_records WHERE hold_category = ? AND status = 'ACTIVE_HOLD' ORDER BY entered_at",
                    (category.value,)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM canonical_hold_records WHERE status = 'ACTIVE_HOLD' ORDER BY entered_at"
                )
            return [CanonicalHoldRecord(**dict(r)) for r in cur.fetchall()]

    def get_population_version(self) -> Tuple[int, str]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT count(*) FROM canonical_audit_log WHERE change_type = 'BATCH_COMMIT'")
            batch_count = cur.fetchone()[0]
            cur2 = conn.execute("SELECT canonical_share_class_id, isin FROM canonical_share_class ORDER BY isin")
            rows = cur2.fetchall()
            digest_src = "|".join(f"{r['canonical_share_class_id']}:{r['isin']}" for r in rows)
            digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()
            return batch_count, digest

    def export_canonical_snapshot(self, version_id: Optional[int] = None) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM canonical_share_class ORDER BY isin")
            classes = [dict(r) for r in cur.fetchall()]
            cur_prov = conn.execute("SELECT * FROM canonical_provenance_records ORDER BY canonical_share_class_id, provenance_record_id")
            provenance = [dict(r) for r in cur_prov.fetchall()]

            v_count, digest = self.get_population_version()
            snapshot_version = version_id if version_id is not None else v_count

            return {
                "snapshot_role": "DERIVED_NON_AUTHORITATIVE_EVIDENCE",
                "version_id": snapshot_version,
                "content_digest": digest,
                "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "share_class_count": len(classes),
                "share_classes": classes,
                "provenance_count": len(provenance),
                "provenance_records": provenance,
            }

    # Denominator Firewall
    def get_denominator(self) -> None:
        raise DenominatorFirewallViolationError(
            "DENOMINATOR_FIREWALL: The canonical population layer is strictly upstream of denominator derivation. "
            "Denominator calculation requires an authorized ETF_V2_DENOMINATOR_DERIVATION_GATE."
        )

    def calculate_denominator(self) -> None:
        raise DenominatorFirewallViolationError(
            "DENOMINATOR_FIREWALL: Canonical population repository cannot compute or promote denominator."
        )


class CanonicalPopulationWriter:
    """Governed transactional mutation interface for Canonical Population Authority."""

    def __init__(self, db_path: Optional[str] = None, lock_path: Optional[str] = None):
        self.db_path = db_path or get_default_db_path()
        self.lock_path = lock_path or get_default_lock_path()
        self._reader = CanonicalPopulationReader(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        is_uri = self.db_path.startswith("file:")
        if not is_uri and self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, uri=is_uri, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if not is_uri and self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def preflight_admission(self, candidates: Sequence[CandidateSubmission]) -> List[PreflightClassification]:
        """Read-only preflight validation of candidate submissions. Zero database mutations."""
        results: List[PreflightClassification] = []
        conn = self._get_connection()
        try:
            seen_batch_isins: set[str] = set()
            for cand in candidates:
                # 1. ISIN Validation
                try:
                    clean_isin = normalize_isin(cand.isin)
                    if not validate_isin(clean_isin, strict=False):
                        results.append(PreflightClassification(
                            candidate=cand,
                            collision_state=CollisionState.NEW_IDENTITY,
                            is_valid=False,
                            error_message=f"Invalid ISO 6166 checksum or format for ISIN: {cand.isin}"
                        ))
                        continue
                except InvalidIdentifierError as e:
                    results.append(PreflightClassification(
                        candidate=cand,
                        collision_state=CollisionState.NEW_IDENTITY,
                        is_valid=False,
                        error_message=str(e)
                    ))
                    continue

                # 2. Batch-Internal Duplicate Check
                if clean_isin in seen_batch_isins:
                    results.append(PreflightClassification(
                        candidate=cand,
                        collision_state=CollisionState.IDENTITY_CONFLICT,
                        is_valid=False,
                        error_message=f"Duplicate ISIN {clean_isin} found within the same candidate submission batch."
                    ))
                    continue
                seen_batch_isins.add(clean_isin)

                # 3. Check for Active Hold
                cur_hold = conn.execute(
                    "SELECT * FROM canonical_hold_records WHERE candidate_identifier = ? AND status = 'ACTIVE_HOLD'",
                    (clean_isin,)
                )
                if cur_hold.fetchone():
                    results.append(PreflightClassification(
                        candidate=cand,
                        collision_state=CollisionState.IDENTITY_CONFLICT,
                        is_valid=False,
                        error_message=f"Candidate {clean_isin} has an active quarantine hold."
                    ))
                    continue

                # 4. Provenance Tier Validation
                has_statutory_prov = False
                for ref in cand.evidence_references:
                    tier = ref.get("authority_tier", cand.authority_tier)
                    if tier in (
                        AuthorityTier.TIER_1_OFFICIAL_STATUTORY.value,
                        AuthorityTier.TIER_2_REGULATOR_OFFICIAL.value,
                        "TIER_1",
                        "TIER_2",
                    ):
                        has_statutory_prov = True
                        break
                if not has_statutory_prov:
                    results.append(PreflightClassification(
                        candidate=cand,
                        collision_state=CollisionState.PROVENANCE_CONFLICT,
                        is_valid=False,
                        error_message=f"Candidate {clean_isin} lacks admissible Tier 1 or Tier 2 statutory provenance."
                    ))
                    continue

                # 5. Database Collision Check
                cur = conn.execute("SELECT * FROM canonical_share_class WHERE isin = ?", (clean_isin,))
                existing = cur.fetchone()
                if not existing:
                    results.append(PreflightClassification(
                        candidate=cand,
                        collision_state=CollisionState.NEW_IDENTITY,
                        is_valid=True
                    ))
                else:
                    exp_parent_id = generate_parent_id(cand.domicile_iso2, cand.legal_umbrella_name)
                    exp_subfund_id = generate_subfund_id(exp_parent_id, cand.legal_subfund_name)

                    if (existing["canonical_parent_id"] == exp_parent_id and
                        existing["canonical_subfund_id"] == exp_subfund_id and
                        existing["legal_share_class_name"] == cand.legal_share_class_name):
                        results.append(PreflightClassification(
                            candidate=cand,
                            collision_state=CollisionState.EXACT_ALREADY_PRESENT,
                            is_valid=True,
                            existing_canonical_id=existing["canonical_share_class_id"]
                        ))
                    elif existing["canonical_status"] in (CanonicalStatus.SUPERSEDED.value, CanonicalStatus.MERGED.value):
                        results.append(PreflightClassification(
                            candidate=cand,
                            collision_state=CollisionState.HISTORICAL_CONTINUITY_REVIEW_REQUIRED,
                            is_valid=False,
                            error_message=f"ISIN {clean_isin} exists under historical/superseded status {existing['canonical_status']}."
                        ))
                    else:
                        results.append(PreflightClassification(
                            candidate=cand,
                            collision_state=CollisionState.IDENTITY_CONFLICT,
                            is_valid=False,
                            error_message=f"Conflict: ISIN {clean_isin} already exists under different parent/subfund."
                        ))
        finally:
            conn.close()

        return results

    def admit_batch(
        self,
        candidates: Sequence[CandidateSubmission],
        governance_gate_id: str,
        batch_id: Optional[str] = None
    ) -> BatchAdmissionResult:
        """
        Executes atomic batch admission under ATOMIC_BATCH_ROLLBACK_BY_DEFAULT_WITH_PREFLIGHT_EXCLUSION.
        """
        batch_uuid = batch_id or f"batch-{uuid.uuid4().hex[:12]}"
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Step 1: Execute Preflight
        preflight_results = self.preflight_admission(candidates)
        admissible_candidates: List[CandidateSubmission] = []
        no_op_ids: List[str] = []
        rejections: List[Dict[str, Any]] = []

        for p in preflight_results:
            if not p.is_valid:
                rejections.append({
                    "isin": p.candidate.isin,
                    "error": p.error_message,
                    "collision_state": p.collision_state.value
                })
            elif p.collision_state == CollisionState.EXACT_ALREADY_PRESENT:
                if p.existing_canonical_id:
                    no_op_ids.append(p.existing_canonical_id)
            elif p.collision_state == CollisionState.NEW_IDENTITY:
                admissible_candidates.append(p.candidate)

        if not admissible_candidates:
            return BatchAdmissionResult(
                batch_id=batch_uuid,
                admitted_at=now_utc,
                governance_gate_id=governance_gate_id,
                total_submitted=len(candidates),
                admitted_count=0,
                no_op_count=len(no_op_ids),
                rejected_count=len(rejections),
                admitted_share_class_ids=(),
                no_op_share_class_ids=tuple(no_op_ids),
                rejections=tuple(rejections),
            )

        # Step 2: Atomic Transaction with Lock
        lock = contextlib.nullcontext() if self.db_path == ":memory:" else SingleWriterProcessLock(self.lock_path)
        with lock:
            conn = self._get_connection()
            admitted_ids: List[str] = []
            try:
                conn.execute("BEGIN IMMEDIATE TRANSACTION;")

                for cand in admissible_candidates:
                    clean_isin = normalize_isin(cand.isin)
                    parent_id = generate_parent_id(cand.domicile_iso2, cand.legal_umbrella_name)
                    subfund_id = generate_subfund_id(parent_id, cand.legal_subfund_name)
                    share_class_id = generate_share_class_id(IdentifierType.ISIN, clean_isin)

                    # 1. Upsert Parent
                    conn.execute("""
                        INSERT OR IGNORE INTO canonical_parent_entity (
                            canonical_parent_id, legal_umbrella_name, domicile_iso2,
                            regulatory_jurisdiction, regulator, legal_entity_structure,
                            national_regulator_code, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        parent_id, cand.legal_umbrella_name, cand.domicile_iso2,
                        cand.regulatory_jurisdiction, cand.regulator, cand.legal_entity_structure,
                        cand.national_regulator_code, now_utc, now_utc
                    ))

                    # 2. Upsert Subfund
                    conn.execute("""
                        INSERT OR IGNORE INTO canonical_subfund (
                            canonical_subfund_id, canonical_parent_id, legal_subfund_name,
                            subfund_currency, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        subfund_id, parent_id, cand.legal_subfund_name,
                        cand.subfund_currency, now_utc, now_utc
                    ))

                    # 3. Insert Share Class
                    currentness = CurrentnessDimensions().to_json()
                    conn.execute("""
                        INSERT INTO canonical_share_class (
                            canonical_share_class_id, isin, canonical_subfund_id, canonical_parent_id,
                            jurisdiction, regulatory_framework, legal_subfund_name, legal_share_class_name,
                            canonical_status, currentness_state, authority_tier, admission_state,
                            admitted_at, admission_gate_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        share_class_id, clean_isin, subfund_id, parent_id,
                        cand.domicile_iso2, cand.regulatory_jurisdiction, cand.legal_subfund_name, cand.legal_share_class_name,
                        CanonicalStatus.ADMITTED_CURRENT.value, currentness, cand.authority_tier,
                        AdmissionState.ADMITTED.value, now_utc, governance_gate_id, now_utc, now_utc
                    ))

                    # 4. Insert Provenance Records
                    prov_inserted = 0
                    for ref in cand.evidence_references:
                        prov_id = f"prov-{uuid.uuid4().hex[:12]}"
                        conn.execute("""
                            INSERT INTO canonical_provenance_records (
                                provenance_record_id, canonical_share_class_id, evidence_object_id,
                                source_candidate_id, source_document_id, source_url, authority_tier,
                                evidence_type, observed_at, evidence_hash, relationship_type, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            prov_id, share_class_id, ref.get("evidence_object_id", f"ev-{cand.source_candidate_id}"),
                            cand.source_candidate_id, ref.get("source_document_id", "STATUTORY_DOC"),
                            ref.get("source_url", "https://official.evidence"), ref.get("authority_tier", cand.authority_tier),
                            ref.get("evidence_type", "STATUTORY_DECLARATION"), now_utc,
                            ref.get("evidence_hash", "3bdc834c34f936d6a4f0b32526759858ea60c8f35d595ff3e6fcca30719f140a"),
                            "DIRECT_STATUTORY_DECLARATION", now_utc
                        ))
                        prov_inserted += 1

                    # 5. Invariant Check: >= 1 Provenance Record
                    if prov_inserted == 0:
                        raise MissingProvenanceError(f"Candidate {clean_isin} lacks required statutory provenance records.")

                    # 6. Audit Log Entry
                    audit_id = f"aud-{uuid.uuid4().hex[:12]}"
                    after_state = json.dumps({
                        "canonical_share_class_id": share_class_id,
                        "isin": clean_isin,
                        "parent_id": parent_id,
                        "subfund_id": subfund_id,
                        "status": CanonicalStatus.ADMITTED_CURRENT.value
                    }, sort_keys=True)

                    conn.execute("""
                        INSERT INTO canonical_audit_log (
                            audit_event_id, admission_batch_id, canonical_share_class_id, isin,
                            change_type, before_state_json, after_state_json, authority_basis,
                            governance_gate_id, recorded_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        audit_id, batch_uuid, share_class_id, clean_isin,
                        "ADMISSION_INSERT", None, after_state,
                        cand.authority_tier, governance_gate_id, now_utc
                    ))

                    admitted_ids.append(share_class_id)

                # Batch Commit Marker
                batch_audit_id = f"aud-batch-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO canonical_audit_log (
                        audit_event_id, admission_batch_id, canonical_share_class_id, isin,
                        change_type, before_state_json, after_state_json, authority_basis,
                        governance_gate_id, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_audit_id, batch_uuid, "BATCH_SUMMARY", "MULTIPLE",
                    "BATCH_COMMIT", None, json.dumps({"admitted_count": len(admitted_ids)}),
                    "STATUTORY_GATE", governance_gate_id, now_utc
                ))

                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error("Admission transaction failed; executed full atomic rollback: %s", e)
                raise AdmissionTransactionError(f"Atomic batch admission failed; transaction rolled back: {e}") from e
            finally:
                conn.close()

        return BatchAdmissionResult(
            batch_id=batch_uuid,
            admitted_at=now_utc,
            governance_gate_id=governance_gate_id,
            total_submitted=len(candidates),
            admitted_count=len(admitted_ids),
            no_op_count=len(no_op_ids),
            rejected_count=len(rejections),
            admitted_share_class_ids=tuple(admitted_ids),
            no_op_share_class_ids=tuple(no_op_ids),
            rejections=tuple(rejections),
        )

    def supersede_identity(
        self,
        old_isin: str,
        successor_isin: str,
        authority_basis: str,
        governance_gate_id: str
    ) -> Tuple[CanonicalShareClass, CanonicalShareClass]:
        """
        Supersedes an admitted identity under the immutable ISIN invariant.
        Old record is preserved and transitioned to SUPERSEDED. In-place ISIN mutation is strictly prohibited.
        """
        clean_old = normalize_isin(old_isin)
        clean_new = normalize_isin(successor_isin)
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

        lock = contextlib.nullcontext() if self.db_path == ":memory:" else SingleWriterProcessLock(self.lock_path)
        with lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE TRANSACTION;")

                # Fetch old record
                cur = conn.execute("SELECT * FROM canonical_share_class WHERE isin = ?", (clean_old,))
                old_row = cur.fetchone()
                if not old_row:
                    raise InvalidIdentifierError(f"Old ISIN {clean_old} does not exist in canonical population.")

                # Ensure successor exists or is admitted
                cur_new = conn.execute("SELECT * FROM canonical_share_class WHERE isin = ?", (clean_new,))
                new_row = cur_new.fetchone()
                if not new_row:
                    raise InvalidIdentifierError(f"Successor ISIN {clean_new} must exist before supersession.")

                # Transition old record to SUPERSEDED
                before_state = json.dumps(dict(old_row), sort_keys=True)
                conn.execute("""
                    UPDATE canonical_share_class
                    SET canonical_status = ?, updated_at = ?
                    WHERE canonical_share_class_id = ?
                """, (CanonicalStatus.SUPERSEDED.value, now_utc, old_row["canonical_share_class_id"]))

                after_state = json.dumps({
                    "canonical_share_class_id": old_row["canonical_share_class_id"],
                    "isin": clean_old,
                    "canonical_status": CanonicalStatus.SUPERSEDED.value,
                    "successor_isin": clean_new
                }, sort_keys=True)

                audit_id = f"aud-sup-{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO canonical_audit_log (
                        audit_event_id, admission_batch_id, canonical_share_class_id, isin,
                        change_type, before_state_json, after_state_json, authority_basis,
                        governance_gate_id, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    audit_id, "SUPERSEDED_TRANSACTION", old_row["canonical_share_class_id"], clean_old,
                    "STATUS_SUPERSEDED", before_state, after_state, authority_basis, governance_gate_id, now_utc
                ))

                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                raise AdmissionTransactionError(f"Supersession failed; transaction rolled back: {e}") from e
            finally:
                conn.close()

        old_updated = self._reader.get_share_class_by_isin(clean_old)
        new_obj = self._reader.get_share_class_by_isin(clean_new)
        assert old_updated is not None and new_obj is not None
        return old_updated, new_obj

    def enter_hold(self, hold_record: CanonicalHoldRecord) -> None:
        """Enters a candidate into the quarantine hold registry."""
        lock = contextlib.nullcontext() if self.db_path == ":memory:" else SingleWriterProcessLock(self.lock_path)
        with lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT INTO canonical_hold_records (
                        hold_record_id, candidate_identifier, hold_category, hold_state,
                        reason, source_candidate_package, entered_at, hold_governance_gate,
                        reopen_condition, status, resolved_at, resolution_gate_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hold_record.hold_record_id, hold_record.candidate_identifier, hold_record.hold_category,
                    hold_record.hold_state, hold_record.reason, hold_record.source_candidate_package,
                    hold_record.entered_at, hold_record.hold_governance_gate, hold_record.reopen_condition,
                    hold_record.status, hold_record.resolved_at, hold_record.resolution_gate_id
                ))
                conn.commit()
            finally:
                conn.close()

    def resolve_hold(self, hold_record_id: str, resolution_gate_id: str) -> None:
        """Resolves an active hold under event-driven reopening."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lock = contextlib.nullcontext() if self.db_path == ":memory:" else SingleWriterProcessLock(self.lock_path)
        with lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    UPDATE canonical_hold_records
                    SET status = 'RESOLVED', resolved_at = ?, resolution_gate_id = ?
                    WHERE hold_record_id = ?
                """, (now_utc, resolution_gate_id, hold_record_id))
                conn.commit()
            finally:
                conn.close()

    def create_backup(self, target_backup_path: str) -> str:
        """Creates an atomic live backup of the canonical database using SQLite VACUUM INTO."""
        dest = Path(target_backup_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        conn = self._get_connection()
        try:
            conn.execute(f"VACUUM INTO '{dest.as_posix()}';")
        finally:
            conn.close()
        return str(dest)

    def verify_backup(self, backup_path: str) -> bool:
        """Verifies integrity of a backup SQLite file."""
        if not os.path.exists(backup_path):
            return False
        conn = sqlite3.connect(backup_path)
        try:
            cur = conn.execute("PRAGMA integrity_check;")
            res = cur.fetchone()[0]
            return res == "ok"
        except Exception:
            return False
        finally:
            conn.close()
