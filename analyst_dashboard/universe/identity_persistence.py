"""
analyst_dashboard/universe/identity_persistence.py

Sprint 2: Persistent Identity Repository & Controlled Local Plan Application Service.

Architectural Principle Formally Approved by Product Owner:
"An evidence-resolved internal security identity and supported listing association
may be registered independently of strategy eligibility and R06 admission status.
Such registration grants no conversion, admission, publication or scanner authority."

Operational Invariants Enforced:
1. Pure Local/Test Scope:
   - Zero production database writes, zero automatic migrations, zero default production paths.
   - Verified exclusively against temporary local stores (file-based or in-memory SQLite).
2. Separation of Concerns:
   - Storing an identity does NOT grant strategy eligibility, SourceSecurity conversion,
     or universe publication.
   - Missing price history is NOT missing identity evidence; registration does NOT depend
     on market-data availability unless an existing contract mandates it.
   - Rule R06 primary-listing status remains independently adjudicated; None remains UNKNOWN.
3. Durable Entity & Listing Integrity:
   - Opaque durable security IDs (ARX_SEC_<12_HEX>) are preserved without hashing CIK/MIC/date.
   - Share-class separation: GOOG (Class C) and GOOGL (Class A) under Alphabet (CIK 0001652044)
     maintain distinct, non-overlapping durable identities.
   - Active listing locators (symbol, operating_mic) cannot overlap across distinct securities.
   - Contradictory effective date ranges (effective_to < effective_from) fail closed immediately.
4. Cryptographic Plan & Transactional Boundaries:
   - Plan content integrity (plan_content_digest) and target state freshness (target_state_digest)
     are verified inside the transaction lock before application.
   - Atomic whole-plan rule: mixed blocked/ready batches fail closed unless allow_partial=True.
   - 100% rollback on any mid-application failure using native SQLite transactions.
   - Idempotent re-application: applying an already-applied plan produces verified no-ops.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

from .contracts import compute_sha256
from .admission_mapping import (
    AttributableFieldEvidence,
    EffectiveDatedListingAssociation,
    EvidenceNature,
    InternalSecurityRecord,
    ListingLocator,
    OfflineAdmissionMappingEngine,
)
from .identity_adoption import (
    AdoptionPlanSummary,
    IdentityAdoptionAuthorization,
    IdentityAdoptionPlan,
    IdentityAdoptionStatus,
    PlanApplicationError,
    PlanIntegrityError,
    ProposedAssociationAdoption,
    ProposedEntityAdoption,
    StalePlanError,
)

logger = logging.getLogger("arx.universe.identity_persistence")

# ─── 0. Formally Approved Architectural Principle ──────────────────────────────

APPROVED_ARCHITECTURAL_PRINCIPLE = (
    "An evidence-resolved internal security identity and supported listing association "
    "may be registered independently of strategy eligibility and R06 admission status. "
    "Such registration grants no conversion, admission, publication or scanner authority."
)

SCHEMA_VERSION = "1.0.0"


def canonical_hash(data: Any) -> str:
    """Computes deterministic SHA-256 digest of arbitrary structured payload."""
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ─── 1. Application Outcome & Status Models ───────────────────────────────────

class IdentityApplicationStatus(str, Enum):
    APPLIED = "APPLIED"
    NO_OP_ALREADY_APPLIED = "NO_OP_ALREADY_APPLIED"
    REJECTED_UNAUTHORIZED = "REJECTED_UNAUTHORIZED"
    REJECTED_STALE_STATE = "REJECTED_STALE_STATE"
    REJECTED_INTEGRITY_TAMPERED = "REJECTED_INTEGRITY_TAMPERED"
    REJECTED_BLOCKED_ITEMS = "REJECTED_BLOCKED_ITEMS"
    REJECTED_CONFLICT = "REJECTED_CONFLICT"
    REJECTED_ERROR = "REJECTED_ERROR"


@dataclass(frozen=True)
class IdentityApplicationOutcome:
    """
    Structured outcome of an identity plan application execution.
    Reports exact applied, no-op, and blocked counts, and audit fingerprints.
    """
    status: IdentityApplicationStatus
    plan_id: str
    target_state_digest_before: str
    target_state_digest_after: str
    applied_entities_count: int
    noop_entities_count: int
    blocked_entities_count: int
    applied_associations_count: int
    noop_associations_count: int
    blocked_associations_count: int
    audit_entry_recorded: bool
    details: List[str] = field(default_factory=list)
    applied_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# ─── 2. Persistent SQLite Identity Repository ──────────────────────────────────

class SqliteIdentityRepository:
    """
    Persistent SQLite repository for InternalSecurityRecord and EffectiveDatedListingAssociation.

    Guarantees:
    - Pure SQLite standard library implementation with zero external ORM dependencies.
    - Foreign key constraints enabled (`PRAGMA foreign_keys = ON;`).
    - Partial unique indexes enforcing that no more than one association is open-ended
      (effective_to IS NULL) per (symbol, operating_mic).
    - Interval overlap validation rejecting overlapping dates across distinct security IDs.
    - Full evidence reference tracking via dedicated child table.
    - Deterministic snapshot export compatible with OfflineAdmissionMappingEngine.
    """

    DDL_SCHEMA = """
    -- Table 1: Master Security Registry
    CREATE TABLE IF NOT EXISTS internal_securities (
        internal_security_id TEXT PRIMARY KEY,
        issuer_cik TEXT,
        primary_symbol TEXT NOT NULL,
        share_class TEXT,
        created_at_utc TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'RETIRED', 'SUSPENDED')),
        continuity_basis TEXT NOT NULL DEFAULT 'INTERNAL_LEDGER_REGISTERED',
        certifies_continuity INTEGER NOT NULL DEFAULT 0,
        notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_sec_cik ON internal_securities(issuer_cik);
    CREATE INDEX IF NOT EXISTS idx_sec_primary_symbol ON internal_securities(primary_symbol);

    -- Table 2: Listing Associations
    CREATE TABLE IF NOT EXISTS listing_associations (
        association_id TEXT PRIMARY KEY,
        internal_security_id TEXT NOT NULL REFERENCES internal_securities(internal_security_id) ON DELETE RESTRICT,
        symbol TEXT NOT NULL,
        operating_mic TEXT NOT NULL,
        effective_from TEXT NOT NULL,
        effective_to TEXT,
        source_reference TEXT NOT NULL DEFAULT '',
        is_disputed INTEGER NOT NULL DEFAULT 0,
        dispute_reason TEXT,
        created_at_utc TEXT NOT NULL,
        CHECK (effective_to IS NULL OR effective_to >= effective_from)
    );

    CREATE INDEX IF NOT EXISTS idx_assoc_sec_id ON listing_associations(internal_security_id);
    CREATE INDEX IF NOT EXISTS idx_assoc_sym_mic ON listing_associations(symbol, operating_mic);

    -- Enforce that at most ONE association is active (open-ended) per (symbol, operating_mic)
    CREATE UNIQUE INDEX IF NOT EXISTS idx_active_assoc_unique 
    ON listing_associations(symbol, operating_mic) 
    WHERE effective_to IS NULL;

    -- Table 3: Identity Evidence References
    CREATE TABLE IF NOT EXISTS identity_evidence_references (
        evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        internal_security_id TEXT REFERENCES internal_securities(internal_security_id) ON DELETE CASCADE,
        association_id TEXT REFERENCES listing_associations(association_id) ON DELETE CASCADE,
        field_name TEXT NOT NULL,
        asserted_value TEXT,
        evidence_nature TEXT NOT NULL,
        source_identifier TEXT NOT NULL,
        source_record_locator TEXT NOT NULL,
        source_content_hash TEXT,
        observation_timestamp_utc TEXT,
        transformation_rule TEXT,
        mapping_version TEXT DEFAULT '1.0.0',
        does_certify_claim INTEGER DEFAULT 0,
        created_at_utc TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_ev_sec_id ON identity_evidence_references(internal_security_id);
    CREATE INDEX IF NOT EXISTS idx_ev_assoc_id ON identity_evidence_references(association_id);

    -- Table 4: Applied Plan Audit Ledger
    CREATE TABLE IF NOT EXISTS applied_plan_audit_ledger (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id TEXT NOT NULL,
        plan_content_digest TEXT NOT NULL,
        target_state_digest_before TEXT NOT NULL,
        target_state_digest_after TEXT NOT NULL,
        applied_at_utc TEXT NOT NULL,
        authorization_scope TEXT NOT NULL,
        authorized_by TEXT,
        entities_applied INTEGER NOT NULL,
        associations_applied INTEGER NOT NULL,
        summary_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audit_plan_id ON applied_plan_audit_ledger(plan_id);
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        read_only: bool = False,
        auto_init: bool = True,
    ):
        self.db_path = str(db_path)
        self.is_memory = self.db_path == ":memory:" or self.db_path.startswith("file::memory:")
        self.read_only = read_only
        self.auto_init = auto_init

        self._persistent_conn: Optional[sqlite3.Connection] = None
        if self.is_memory:
            # Memory DB requires persistent connection across object lifetime
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys = ON;")
            if self.auto_init and not self.read_only:
                self._init_schema(self._persistent_conn)
        else:
            p = Path(self.db_path).resolve()
            if self.read_only:
                if not p.exists():
                    raise FileNotFoundError(f"Database file not found for read-only repository: {self.db_path}")
                self.db_path = str(p)
            else:
                self.db_path = str(p)
                if self.auto_init:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with self._get_connection() as conn:
                        conn.execute("PRAGMA foreign_keys = ON;")
                        conn.execute("PRAGMA journal_mode = WAL;")
                        self._init_schema(conn)

    def initialize_schema(self) -> None:
        """Explicitly initializes database directory, connection, and tables."""
        if self.read_only:
            raise PermissionError("Cannot initialize schema on a read-only repository.")
        if not self.is_memory:
            p = Path(self.db_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(p)
        with self._get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            if not self.is_memory:
                conn.execute("PRAGMA journal_mode = WAL;")
            self._init_schema(conn)

    def _get_connection(self) -> sqlite3.Connection:
        if self.is_memory and self._persistent_conn is not None:
            return self._persistent_conn
        if self.read_only:
            # Enforce SQLite driver-level read-only URI mode (?mode=ro)
            uri = Path(self.db_path).resolve().as_uri()
            conn = sqlite3.connect(f"{uri}?mode=ro", uri=True, check_same_thread=False)
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(self.DDL_SCHEMA)
        conn.commit()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager providing real ACID transaction semantics via SQLite BEGIN IMMEDIATE.
        Guarantees that write locks are acquired immediately, preventing race conditions.
        Automatically commits on normal exit; executes ROLLBACK on any raised exception.
        """
        if self.read_only:
            raise PermissionError("Cannot initiate write transaction on a read-only repository.")
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if not self.is_memory:
                conn.close()

    def close(self) -> None:
        """Closes any persistent memory connection."""
        if self.is_memory and self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def __enter__(self) -> SqliteIdentityRepository:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ─── Query Methods ─────────────────────────────────────────────────────────

    def get_security(
        self, internal_security_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[InternalSecurityRecord]:
        """Look up internal security record by exact opaque ID."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM internal_securities WHERE internal_security_id = ?",
                (internal_security_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return InternalSecurityRecord(
                internal_security_id=row["internal_security_id"],
                issuer_cik=row["issuer_cik"],
                primary_symbol=row["primary_symbol"],
                share_class=row["share_class"],
                created_at_utc=row["created_at_utc"],
                status=row["status"],
                continuity_basis=row["continuity_basis"],
                certifies_continuity=bool(row["certifies_continuity"]),
            )
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_securities_by_cik(
        self, issuer_cik: str, conn: Optional[sqlite3.Connection] = None
    ) -> List[InternalSecurityRecord]:
        """Look up all security records for an issuer CIK."""
        clean_cik = str(issuer_cik).zfill(10)
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM internal_securities WHERE issuer_cik = ? ORDER BY primary_symbol",
                (clean_cik,),
            )
            return [
                InternalSecurityRecord(
                    internal_security_id=row["internal_security_id"],
                    issuer_cik=row["issuer_cik"],
                    primary_symbol=row["primary_symbol"],
                    share_class=row["share_class"],
                    created_at_utc=row["created_at_utc"],
                    status=row["status"],
                    continuity_basis=row["continuity_basis"],
                    certifies_continuity=bool(row["certifies_continuity"]),
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_securities_by_symbol(
        self, symbol: str, conn: Optional[sqlite3.Connection] = None
    ) -> List[InternalSecurityRecord]:
        """Look up all security records with matching primary symbol."""
        clean_sym = symbol.strip().upper()
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM internal_securities WHERE primary_symbol = ? ORDER BY internal_security_id",
                (clean_sym,),
            )
            return [
                InternalSecurityRecord(
                    internal_security_id=row["internal_security_id"],
                    issuer_cik=row["issuer_cik"],
                    primary_symbol=row["primary_symbol"],
                    share_class=row["share_class"],
                    created_at_utc=row["created_at_utc"],
                    status=row["status"],
                    continuity_basis=row["continuity_basis"],
                    certifies_continuity=bool(row["certifies_continuity"]),
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def list_all_securities(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> List[InternalSecurityRecord]:
        """List all securities in the repository ordered by primary symbol."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute("SELECT * FROM internal_securities ORDER BY primary_symbol, internal_security_id")
            return [
                InternalSecurityRecord(
                    internal_security_id=row["internal_security_id"],
                    issuer_cik=row["issuer_cik"],
                    primary_symbol=row["primary_symbol"],
                    share_class=row["share_class"],
                    created_at_utc=row["created_at_utc"],
                    status=row["status"],
                    continuity_basis=row["continuity_basis"],
                    certifies_continuity=bool(row["certifies_continuity"]),
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_association(
        self, association_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[EffectiveDatedListingAssociation]:
        """Look up listing association by exact ID."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute("SELECT * FROM listing_associations WHERE association_id = ?", (association_id,))
            row = cur.fetchone()
            if not row:
                return None
            return EffectiveDatedListingAssociation(
                association_id=row["association_id"],
                internal_security_id=row["internal_security_id"],
                symbol=row["symbol"],
                operating_mic=row["operating_mic"],
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                source_reference=row["source_reference"],
                is_disputed=bool(row["is_disputed"]),
                dispute_reason=row["dispute_reason"],
            )
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_associations_for_security(
        self, internal_security_id: str, conn: Optional[sqlite3.Connection] = None
    ) -> List[EffectiveDatedListingAssociation]:
        """Look up all listing associations linked to a security ID."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM listing_associations WHERE internal_security_id = ? ORDER BY effective_from",
                (internal_security_id,),
            )
            return [
                EffectiveDatedListingAssociation(
                    association_id=row["association_id"],
                    internal_security_id=row["internal_security_id"],
                    symbol=row["symbol"],
                    operating_mic=row["operating_mic"],
                    effective_from=row["effective_from"],
                    effective_to=row["effective_to"],
                    source_reference=row["source_reference"],
                    is_disputed=bool(row["is_disputed"]),
                    dispute_reason=row["dispute_reason"],
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_active_association(
        self,
        symbol: str,
        operating_mic: str,
        as_of_date: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[EffectiveDatedListingAssociation]:
        """
        Look up active listing association for (symbol, operating_mic) on as_of_date.
        If as_of_date is None, returns open-ended active association (effective_to IS NULL).
        """
        clean_sym = symbol.strip().upper()
        clean_mic = operating_mic.strip().upper()
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            if as_of_date is None:
                cur.execute(
                    "SELECT * FROM listing_associations WHERE symbol = ? AND operating_mic = ? AND effective_to IS NULL",
                    (clean_sym, clean_mic),
                )
            else:
                cur.execute(
                    "SELECT * FROM listing_associations WHERE symbol = ? AND operating_mic = ? "
                    "AND effective_from <= ? AND (effective_to IS NULL OR effective_to >= ?)",
                    (clean_sym, clean_mic, as_of_date, as_of_date),
                )
            row = cur.fetchone()
            if not row:
                return None
            return EffectiveDatedListingAssociation(
                association_id=row["association_id"],
                internal_security_id=row["internal_security_id"],
                symbol=row["symbol"],
                operating_mic=row["operating_mic"],
                effective_from=row["effective_from"],
                effective_to=row["effective_to"],
                source_reference=row["source_reference"],
                is_disputed=bool(row["is_disputed"]),
                dispute_reason=row["dispute_reason"],
            )
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def list_all_associations(
        self, conn: Optional[sqlite3.Connection] = None
    ) -> List[EffectiveDatedListingAssociation]:
        """List all associations in repository."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute("SELECT * FROM listing_associations ORDER BY symbol, effective_from")
            return [
                EffectiveDatedListingAssociation(
                    association_id=row["association_id"],
                    internal_security_id=row["internal_security_id"],
                    symbol=row["symbol"],
                    operating_mic=row["operating_mic"],
                    effective_from=row["effective_from"],
                    effective_to=row["effective_to"],
                    source_reference=row["source_reference"],
                    is_disputed=bool(row["is_disputed"]),
                    dispute_reason=row["dispute_reason"],
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def get_evidence_references(
        self,
        internal_security_id: Optional[str] = None,
        association_id: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[AttributableFieldEvidence]:
        """Retrieve attributable evidence records linked to a security or association."""
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            if internal_security_id and association_id:
                cur.execute(
                    "SELECT * FROM identity_evidence_references WHERE internal_security_id = ? AND association_id = ?",
                    (internal_security_id, association_id),
                )
            elif internal_security_id:
                cur.execute(
                    "SELECT * FROM identity_evidence_references WHERE internal_security_id = ?",
                    (internal_security_id,),
                )
            elif association_id:
                cur.execute(
                    "SELECT * FROM identity_evidence_references WHERE association_id = ?",
                    (association_id,),
                )
            else:
                cur.execute("SELECT * FROM identity_evidence_references")

            return [
                AttributableFieldEvidence(
                    field_name=row["field_name"],
                    asserted_value=row["asserted_value"],
                    evidence_nature=EvidenceNature(row["evidence_nature"]),
                    source_identifier=row["source_identifier"],
                    source_record_locator=row["source_record_locator"],
                    source_content_hash=row["source_content_hash"],
                    observation_timestamp_utc=row["observation_timestamp_utc"],
                    transformation_rule=row["transformation_rule"],
                    mapping_version=row["mapping_version"],
                    does_certify_claim=bool(row["does_certify_claim"]),
                )
                for row in cur.fetchall()
            ]
        finally:
            if conn is None and not self.is_memory:
                c.close()

    # ─── Write & Validation Methods ────────────────────────────────────────────

    def check_association_overlap(
        self,
        symbol: str,
        operating_mic: str,
        effective_from: str,
        effective_to: Optional[str],
        internal_security_id: str,
        association_id: str,
        conn: sqlite3.Connection,
    ) -> None:
        """
        Validates that the date range [effective_from, effective_to] does NOT overlap
        with any existing association for (symbol, operating_mic) mapped to a DIFFERENT security ID.
        Fails closed with ValueError if an overlapping conflict is detected.
        """
        clean_sym = symbol.strip().upper()
        clean_mic = operating_mic.strip().upper()
        r1_end = effective_to or "9999-12-31"

        cur = conn.cursor()
        cur.execute(
            "SELECT association_id, internal_security_id, effective_from, effective_to "
            "FROM listing_associations WHERE symbol = ? AND operating_mic = ?",
            (clean_sym, clean_mic),
        )
        for row in cur.fetchall():
            ex_assoc_id = row["association_id"]
            ex_sec_id = row["internal_security_id"]
            ex_from = row["effective_from"]
            ex_to = row["effective_to"] or "9999-12-31"

            # Skip exact same association record
            if ex_assoc_id == association_id:
                continue

            # Interval overlap condition: max(A, C) <= min(B, D)
            start_overlap = max(effective_from, ex_from)
            end_overlap = min(r1_end, ex_to)

            if start_overlap <= end_overlap:
                if ex_sec_id != internal_security_id:
                    raise ValueError(
                        f"Overlapping association conflict: Listing '{clean_sym}' on '{clean_mic}' "
                        f"during [{effective_from}, {effective_to or 'ACTIVE'}] overlaps with existing "
                        f"association '{ex_assoc_id}' for distinct security '{ex_sec_id}' "
                        f"during [{ex_from}, {row['effective_to'] or 'ACTIVE'}]."
                    )

    def save_security(
        self,
        sec: InternalSecurityRecord,
        evidence_refs: Optional[List[AttributableFieldEvidence]] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        Saves or verifies an InternalSecurityRecord.
        Returns True if a new row was inserted; False if already exists with matching attributes (no-op).
        Raises ValueError if internal_security_id already exists with conflicting attributes.
        """
        if self.read_only:
            raise PermissionError("Cannot save security in a read-only repository.")
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM internal_securities WHERE internal_security_id = ?",
                (sec.internal_security_id,),
            )
            existing = cur.fetchone()
            if existing:
                # Concordance check: verify existing attributes match
                clean_cik = str(sec.issuer_cik).zfill(10) if sec.issuer_cik else None
                ex_cik = existing["issuer_cik"]
                if existing["primary_symbol"] != sec.primary_symbol or ex_cik != clean_cik:
                    raise ValueError(
                        f"Security ID conflict: Existing security '{sec.internal_security_id}' has "
                        f"symbol '{existing['primary_symbol']}' / CIK '{ex_cik}', conflicting with "
                        f"attempted insert of symbol '{sec.primary_symbol}' / CIK '{clean_cik}'."
                    )
                return False  # Idempotent no-op

            clean_cik = str(sec.issuer_cik).zfill(10) if sec.issuer_cik else None
            cur.execute(
                "INSERT INTO internal_securities ("
                "internal_security_id, issuer_cik, primary_symbol, share_class, created_at_utc, "
                "status, continuity_basis, certifies_continuity"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sec.internal_security_id,
                    clean_cik,
                    sec.primary_symbol.strip().upper(),
                    sec.share_class,
                    sec.created_at_utc,
                    sec.status,
                    sec.continuity_basis,
                    int(sec.certifies_continuity),
                ),
            )

            # Persist evidence references if supplied
            if evidence_refs:
                now_utc = datetime.now(timezone.utc).isoformat()
                for ref in evidence_refs:
                    cur.execute(
                        "INSERT INTO identity_evidence_references ("
                        "internal_security_id, field_name, asserted_value, evidence_nature, "
                        "source_identifier, source_record_locator, source_content_hash, "
                        "observation_timestamp_utc, transformation_rule, mapping_version, "
                        "does_certify_claim, created_at_utc"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            sec.internal_security_id,
                            ref.field_name,
                            str(ref.asserted_value),
                            ref.evidence_nature.value,
                            ref.source_identifier,
                            ref.source_record_locator,
                            ref.source_content_hash,
                            ref.observation_timestamp_utc,
                            ref.transformation_rule,
                            ref.mapping_version,
                            int(ref.does_certify_claim),
                            now_utc,
                        ),
                    )

            if conn is None:
                c.commit()
            return True
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def save_association(
        self,
        assoc: EffectiveDatedListingAssociation,
        evidence_refs: Optional[List[AttributableFieldEvidence]] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        Saves or verifies an EffectiveDatedListingAssociation.
        Validates date ordering and verifies interval overlaps across distinct securities.
        Returns True if a new row was inserted; False if already exists with matching attributes (no-op).
        """
        if self.read_only:
            raise PermissionError("Cannot save association in a read-only repository.")
        if assoc.effective_to and assoc.effective_to < assoc.effective_from:
            raise ValueError(
                f"Contradictory date range: effective_to ({assoc.effective_to}) "
                f"cannot be earlier than effective_from ({assoc.effective_from})."
            )

        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute(
                "SELECT * FROM listing_associations WHERE association_id = ?",
                (assoc.association_id,),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["internal_security_id"] != assoc.internal_security_id
                    or existing["symbol"] != assoc.symbol.strip().upper()
                    or existing["operating_mic"] != assoc.operating_mic.strip().upper()
                ):
                    raise ValueError(
                        f"Association ID conflict: Existing association '{assoc.association_id}' "
                        f"has security '{existing['internal_security_id']}', conflicting with "
                        f"attempted insert of security '{assoc.internal_security_id}'."
                    )
                return False  # Idempotent no-op

            # Validate date intervals against existing associations for this symbol+mic
            self.check_association_overlap(
                symbol=assoc.symbol,
                operating_mic=assoc.operating_mic,
                effective_from=assoc.effective_from,
                effective_to=assoc.effective_to,
                internal_security_id=assoc.internal_security_id,
                association_id=assoc.association_id,
                conn=c,
            )

            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "INSERT INTO listing_associations ("
                "association_id, internal_security_id, symbol, operating_mic, effective_from, "
                "effective_to, source_reference, is_disputed, dispute_reason, created_at_utc"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    assoc.association_id,
                    assoc.internal_security_id,
                    assoc.symbol.strip().upper(),
                    assoc.operating_mic.strip().upper(),
                    assoc.effective_from,
                    assoc.effective_to,
                    assoc.source_reference,
                    int(assoc.is_disputed),
                    assoc.dispute_reason,
                    now_utc,
                ),
            )

            if evidence_refs:
                for ref in evidence_refs:
                    cur.execute(
                        "INSERT INTO identity_evidence_references ("
                        "association_id, field_name, asserted_value, evidence_nature, "
                        "source_identifier, source_record_locator, source_content_hash, "
                        "observation_timestamp_utc, transformation_rule, mapping_version, "
                        "does_certify_claim, created_at_utc"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            assoc.association_id,
                            ref.field_name,
                            str(ref.asserted_value),
                            ref.evidence_nature.value,
                            ref.source_identifier,
                            ref.source_record_locator,
                            ref.source_content_hash,
                            ref.observation_timestamp_utc,
                            ref.transformation_rule,
                            ref.mapping_version,
                            int(ref.does_certify_claim),
                            now_utc,
                        ),
                    )

            if conn is None:
                c.commit()
            return True
        finally:
            if conn is None and not self.is_memory:
                c.close()

    # ─── Snapshot & Integration Export ─────────────────────────────────────────

    def to_snapshot_dict(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Serializes repository state into a dictionary structure exactly compatible with
        OfflineAdmissionMappingEngine.to_dict() and IdentityAdoptionDryRunEngine.compute_target_state_digest.
        """
        c = conn or self._get_connection()
        try:
            cur = c.cursor()
            cur.execute("SELECT * FROM internal_securities ORDER BY internal_security_id")
            sec_dict: Dict[str, Any] = {}
            cik_map: Dict[str, List[str]] = {}
            for row in cur.fetchall():
                sec_dict[row["internal_security_id"]] = {
                    "internal_security_id": row["internal_security_id"],
                    "issuer_cik": row["issuer_cik"],
                    "primary_symbol": row["primary_symbol"],
                    "share_class": row["share_class"],
                    "created_at_utc": row["created_at_utc"],
                    "status": row["status"],
                    "continuity_basis": row["continuity_basis"],
                    "certifies_continuity": bool(row["certifies_continuity"]),
                }
                if row["issuer_cik"]:
                    cik_map.setdefault(row["issuer_cik"], []).append(row["internal_security_id"])

            cur.execute("SELECT * FROM listing_associations ORDER BY association_id")
            assoc_dict: Dict[str, Any] = {}
            for row in cur.fetchall():
                assoc_dict[row["association_id"]] = {
                    "association_id": row["association_id"],
                    "internal_security_id": row["internal_security_id"],
                    "symbol": row["symbol"],
                    "operating_mic": row["operating_mic"],
                    "effective_from": row["effective_from"],
                    "effective_to": row["effective_to"],
                    "source_reference": row["source_reference"],
                    "is_disputed": bool(row["is_disputed"]),
                    "dispute_reason": row["dispute_reason"],
                }

            sorted_cik_map = {k: sorted(v) for k, v in sorted(cik_map.items())}
            return {
                "security_registry": sec_dict,
                "association_ledger": assoc_dict,
                "cik_to_securities": sorted_cik_map,
            }
        finally:
            if conn is None and not self.is_memory:
                c.close()

    def compute_target_state_digest(self, conn: Optional[sqlite3.Connection] = None) -> str:
        """Computes deterministic SHA-256 fingerprint of current repository state."""
        return canonical_hash(self.to_snapshot_dict(conn=conn))

    def load_into_admission_engine(
        self,
        engine: Optional[OfflineAdmissionMappingEngine] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> OfflineAdmissionMappingEngine:
        """Loads repository records into an in-memory OfflineAdmissionMappingEngine instance."""
        target_engine = engine or OfflineAdmissionMappingEngine()
        snapshot = self.to_snapshot_dict(conn=conn)
        for k, v in snapshot["security_registry"].items():
            target_engine._security_registry[k] = InternalSecurityRecord(**v)
        for k, v in snapshot["association_ledger"].items():
            assoc = EffectiveDatedListingAssociation(**v)
            target_engine._association_ledger[k] = assoc
            key = (assoc.symbol, assoc.operating_mic)
            target_engine._active_listing_index.setdefault(key, []).append(assoc)
        for k, v in snapshot["cik_to_securities"].items():
            target_engine._cik_to_securities[k] = set(v)
        return target_engine

    def import_from_admission_engine(
        self,
        engine: OfflineAdmissionMappingEngine,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[int, int]:
        """Imports all records from an OfflineAdmissionMappingEngine into this repository."""
        c = conn or self._get_connection()
        inserted_secs = 0
        inserted_assocs = 0
        try:
            snapshot = engine.to_dict()
            for v in snapshot.get("security_registry", {}).values():
                sec = InternalSecurityRecord(**v)
                if self.save_security(sec, conn=c):
                    inserted_secs += 1
            for v in snapshot.get("association_ledger", {}).values():
                assoc = EffectiveDatedListingAssociation(**v)
                if self.save_association(assoc, conn=c):
                    inserted_assocs += 1
            if conn is None:
                c.commit()
            return inserted_secs, inserted_assocs
        finally:
            if conn is None and not self.is_memory:
                c.close()


# ─── 3. Controlled Identity Plan Application Service ──────────────────────────

class IdentityPlanApplicationService:
    """
    Narrowly scoped application service that atomically applies a reviewed IdentityAdoptionPlan
    to a persistent SqliteIdentityRepository.

    Strict Architectural Boundaries:
    - Rejects blocked or unauthorized plans; fails closed on any unreviewed proposal.
    - Requires explicit authorization context (`auth.is_authorized is True`).
    - Validates plan content integrity (`plan_content_digest`) and target state applicability
      (`target_state_digest`) inside the SQLite transaction lock to eliminate TOCTOU race windows.
    - Enforces atomic whole-plan execution: mixed blocked/ready batches are rejected unless allow_partial=True.
    - Guaranteed 100% rollback on any mid-application failure using SQLite native transactions.
    - Completely idempotent on repeated application.
    """

    @classmethod
    def apply_plan(
        cls,
        plan: IdentityAdoptionPlan,
        repository: SqliteIdentityRepository,
        authorization: Optional[IdentityAdoptionAuthorization] = None,
        allow_partial: bool = False,
        simulate_mid_failure_on_symbol: Optional[str] = None,
    ) -> IdentityApplicationOutcome:
        """
        Executes controlled application of an adoption plan against the target repository.
        """
        auth = authorization or plan.authorization

        # 1. Authorization Gate
        if not auth.is_authorized:
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_UNAUTHORIZED,
                plan_id=plan.plan_id,
                target_state_digest_before=repository.compute_target_state_digest(),
                target_state_digest_after=repository.compute_target_state_digest(),
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=["Plan application rejected: authorization.is_authorized is False."],
            )

        # 2. Plan Content Integrity Gate
        if not plan.is_plan_content_valid():
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED,
                plan_id=plan.plan_id,
                target_state_digest_before=repository.compute_target_state_digest(),
                target_state_digest_after=repository.compute_target_state_digest(),
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=["Plan content tampering detected: recomputed plan digest does not match declared digest."],
            )

        # 3. Target State Freshness Check (Pre-lock verification)
        digest_before = repository.compute_target_state_digest()
        if digest_before != plan.target_state_digest:
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_STALE_STATE,
                plan_id=plan.plan_id,
                target_state_digest_before=digest_before,
                target_state_digest_after=digest_before,
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=[
                    f"Stale plan rejected: Plan target state digest '{plan.target_state_digest[:12]}...' "
                    f"does not match current repository digest '{digest_before[:12]}...'."
                ],
            )

        # 4. Atomic Whole-Batch Rule
        has_blocked_items = (
            plan.summary.entities_blocked > 0
            or plan.summary.associations_blocked > 0
            or plan.summary.conflicts_count > 0
        )
        if has_blocked_items and not allow_partial:
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_BLOCKED_ITEMS,
                plan_id=plan.plan_id,
                target_state_digest_before=digest_before,
                target_state_digest_after=digest_before,
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=[
                    f"Atomic whole-batch rule rejected plan: Contains {plan.summary.entities_blocked} blocked "
                    f"entities, {plan.summary.associations_blocked} blocked associations, and {plan.summary.conflicts_count} "
                    f"conflicts while allow_partial is False."
                ],
            )

        # 5. Atomic Execution Inside Transaction Lock
        applied_ent_count = 0
        noop_ent_count = 0
        applied_assoc_count = 0
        noop_assoc_count = 0
        details: List[str] = []

        try:
            with repository.transaction() as conn:
                # Re-verify target state digest inside transaction lock to eliminate TOCTOU race window
                lock_digest = repository.compute_target_state_digest(conn=conn)
                if lock_digest != plan.target_state_digest:
                    raise StalePlanError(
                        f"Target repository state mutated during transaction acquisition: '{lock_digest[:12]}...'."
                    )

                now_utc = datetime.now(timezone.utc).isoformat()

                # Process Entity Creations
                for ent in plan.entities:
                    if ent.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION and ent.is_creation:
                        if simulate_mid_failure_on_symbol and ent.primary_symbol == simulate_mid_failure_on_symbol:
                            raise RuntimeError(
                                f"Simulated mid-application transaction failure on entity '{ent.primary_symbol}'."
                            )

                        sec_rec = InternalSecurityRecord(
                            internal_security_id=ent.proposed_durable_security_id,
                            issuer_cik=ent.issuer_cik,
                            primary_symbol=ent.primary_symbol,
                            share_class=ent.share_class,
                            created_at_utc=now_utc,
                            status="ACTIVE",
                            continuity_basis="ADOPTED_UNDER_REVIEWED_PLAN",
                            certifies_continuity=False,
                        )
                        was_inserted = repository.save_security(sec_rec, conn=conn)
                        if was_inserted:
                            applied_ent_count += 1
                            details.append(f"Created entity '{sec_rec.internal_security_id}' ({sec_rec.primary_symbol}).")
                        else:
                            noop_ent_count += 1
                            details.append(f"Entity '{sec_rec.internal_security_id}' already exists (no-op).")

                    elif ent.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE:
                        noop_ent_count += 1
                        details.append(f"Entity '{ent.primary_symbol}' already matches durable '{ent.matched_existing_security_id}'.")

                # Process Association Creations
                for assoc in plan.associations:
                    if assoc.status == IdentityAdoptionStatus.APPROVED_FOR_ADOPTION and assoc.is_creation:
                        if simulate_mid_failure_on_symbol and assoc.symbol == simulate_mid_failure_on_symbol:
                            raise RuntimeError(
                                f"Simulated mid-application transaction failure on association '{assoc.symbol}'."
                            )

                        assoc_id = assoc.proposed_durable_association_id or (
                            f"ASSOC_{assoc.target_durable_security_id}_{assoc.operating_mic}_{assoc.symbol}_{assoc.effective_from}"
                        )
                        assoc_rec = EffectiveDatedListingAssociation(
                            association_id=assoc_id,
                            internal_security_id=assoc.target_durable_security_id,
                            symbol=assoc.symbol,
                            operating_mic=assoc.operating_mic or "UNRESOLVED",
                            effective_from=assoc.effective_from,
                            effective_to=assoc.effective_to,
                            source_reference=f"ADOPTED_PLAN_{plan.plan_id}",
                        )
                        was_inserted = repository.save_association(assoc_rec, conn=conn)
                        if was_inserted:
                            applied_assoc_count += 1
                            details.append(f"Created association '{assoc_rec.association_id}' ({assoc_rec.symbol}).")
                        else:
                            noop_assoc_count += 1
                            details.append(f"Association '{assoc_rec.association_id}' already exists (no-op).")

                    elif assoc.status == IdentityAdoptionStatus.MATCHES_EXISTING_DURABLE:
                        noop_assoc_count += 1
                        details.append(f"Association '{assoc.symbol}' already matches durable '{assoc.matched_existing_association_id}'.")

                # Compute new digest inside transaction before audit insert
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO applied_plan_audit_ledger ("
                    "plan_id, plan_content_digest, target_state_digest_before, target_state_digest_after, "
                    "applied_at_utc, authorization_scope, authorized_by, entities_applied, associations_applied, summary_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        plan.plan_id,
                        plan.plan_content_digest,
                        digest_before,
                        # Will compute state digest post entity/assoc insert
                        repository.compute_target_state_digest(conn=conn),
                        now_utc,
                        auth.scope,
                        auth.authorized_by,
                        applied_ent_count,
                        applied_assoc_count,
                        json.dumps(plan.summary.to_dict()),
                    ),
                )

            # Transaction committed successfully
            digest_after = repository.compute_target_state_digest()
            final_status = (
                IdentityApplicationStatus.NO_OP_ALREADY_APPLIED
                if applied_ent_count == 0 and applied_assoc_count == 0
                else IdentityApplicationStatus.APPLIED
            )

            return IdentityApplicationOutcome(
                status=final_status,
                plan_id=plan.plan_id,
                target_state_digest_before=digest_before,
                target_state_digest_after=digest_after,
                applied_entities_count=applied_ent_count,
                noop_entities_count=noop_ent_count,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=applied_assoc_count,
                noop_associations_count=noop_assoc_count,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=True,
                details=details,
                applied_at_utc=now_utc,
            )

        except (ValueError, StalePlanError) as e:
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_CONFLICT,
                plan_id=plan.plan_id,
                target_state_digest_before=digest_before,
                target_state_digest_after=repository.compute_target_state_digest(),
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=[f"Transaction rolled back due to conflict: {e}"],
            )
        except Exception as e:
            return IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_ERROR,
                plan_id=plan.plan_id,
                target_state_digest_before=digest_before,
                target_state_digest_after=repository.compute_target_state_digest(),
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=[f"Transaction rolled back due to error: {e}"],
            )


# ─── 4. Local Operator CLI Entry Point ────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """
    Local operator command line interface for identity plan application.
    Requires an explicit local target --db-path with NO production default.
    """
    parser = argparse.ArgumentParser(
        description="ARX Terminal: Bounded Local Identity Plan Application Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strict Governance Requirements:
- Target --db-path must be explicitly specified (no production default).
- Operations execute within an atomic SQLite transaction.
- Production identity adoption remains unauthorized.
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: init / setup
    init_parser = subparsers.add_parser("init", aliases=["setup"], help="Initialize empty repository schema on target database path.")
    init_parser.add_argument("--db-path", required=True, help="Explicit target SQLite database path (REQUIRED).")

    # Subcommand: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect local repository entities and associations.")
    inspect_parser.add_argument("--db-path", required=True, help="Explicit target SQLite database path (REQUIRED).")

    # Subcommand: apply
    apply_parser = subparsers.add_parser("apply", help="Apply a reviewed adoption plan to local repository.")
    apply_parser.add_argument("--db-path", required=True, help="Explicit target SQLite database path (REQUIRED, no default).")
    apply_parser.add_argument("--plan-file", required=True, help="Path to reviewed IdentityAdoptionPlan JSON file.")
    apply_parser.add_argument("--allow-partial", action="store_true", help="Allow partial application of non-blocked items.")
    apply_parser.add_argument("--authorize", action="store_true", help="Explicit operator authorization flag.")
    apply_parser.add_argument("--authorized-by", default=None, help="Operator identifier authorizing execution.")
    apply_parser.add_argument("--scope", default="LOCAL_TEMPORARY", help="Execution scope (default: LOCAL_TEMPORARY).")

    args = parser.parse_args(argv)

    if args.command in ("init", "setup"):
        p = Path(args.db_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        repo = SqliteIdentityRepository(str(p), auto_init=False)
        repo.initialize_schema()
        print(f"Repository schema successfully initialized at: {str(p)}")
        return 0

    elif args.command == "inspect":
        if not os.path.exists(args.db_path):
            print(f"Error: Target database '{args.db_path}' does not exist.", file=sys.stderr)
            return 1
        with SqliteIdentityRepository(args.db_path, read_only=True) as repo:
            secs = repo.list_all_securities()
            assocs = repo.list_all_associations()
            digest = repo.compute_target_state_digest()
            print(f"Repository: {args.db_path}")
            print(f"Target State Digest: {digest}")
            print(f"Total Securities: {len(secs)}")
            for s in secs:
                print(f"  Sec ID: {s.internal_security_id} | Sym: {s.primary_symbol} | CIK: {s.issuer_cik} | Class: {s.share_class}")
            print(f"Total Associations: {len(assocs)}")
            for a in assocs:
                print(f"  Assoc ID: {a.association_id} -> {a.internal_security_id} | {a.symbol} on {a.operating_mic} [{a.effective_from} to {a.effective_to or 'ACTIVE'}]")
            return 0

    elif args.command == "apply":
        if not os.path.exists(args.plan_file):
            print(f"Error: Plan file '{args.plan_file}' does not exist.", file=sys.stderr)
            return 1

        try:
            with open(args.plan_file, "r", encoding="utf-8") as f:
                plan = IdentityAdoptionPlan.from_json(f.read())
        except Exception as e:
            print(f"Error: Failed to parse plan file: {e}", file=sys.stderr)
            return 1

        # Authorization Gate prior to touching target database
        if not args.authorize:
            outcome = IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_UNAUTHORIZED,
                plan_id=plan.plan_id,
                target_state_digest_before="",
                target_state_digest_after="",
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=["Plan application rejected: --authorize flag was not supplied."],
            )
            print(outcome.to_json(indent=2))
            return 1

        # Content integrity verification prior to touching target database
        if not plan.is_plan_content_valid():
            outcome = IdentityApplicationOutcome(
                status=IdentityApplicationStatus.REJECTED_INTEGRITY_TAMPERED,
                plan_id=plan.plan_id,
                target_state_digest_before="",
                target_state_digest_after="",
                applied_entities_count=0,
                noop_entities_count=0,
                blocked_entities_count=plan.summary.entities_blocked,
                applied_associations_count=0,
                noop_associations_count=0,
                blocked_associations_count=plan.summary.associations_blocked,
                audit_entry_recorded=False,
                details=["Plan content tampering detected: recomputed plan digest does not match declared digest."],
            )
            print(outcome.to_json(indent=2))
            return 1

        # Check target database exists - do NOT auto-create on apply
        if not os.path.exists(args.db_path):
            print(
                f"Error: Target database '{args.db_path}' does not exist. "
                "Repository must be explicitly initialized with 'init' before applying plans.",
                file=sys.stderr,
            )
            return 1

        auth = IdentityAdoptionAuthorization(
            is_authorized=True,
            authorized_by=args.authorized_by or "LOCAL_OPERATOR",
            scope=args.scope,
            authorization_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

        with SqliteIdentityRepository(args.db_path, auto_init=False) as repo:
            outcome = IdentityPlanApplicationService.apply_plan(
                plan=plan,
                repository=repo,
                authorization=auth,
                allow_partial=args.allow_partial,
            )
            print(outcome.to_json(indent=2))
            return 0 if outcome.status in (IdentityApplicationStatus.APPLIED, IdentityApplicationStatus.NO_OP_ALREADY_APPLIED) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
