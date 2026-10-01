"""FastAPI Router for Canonical Telemetry Persistence & Audit Read Path (P1 Milestone).

Guarantees:
1. Sole Persistence Authority: Atomically persists raw events and canonical 16-field audit records.
2. Race-Safe Deduplication: Unique constraint on deduplication_key prevents duplicate counting.
3. Zero Parallel Classification Logic: Re-classification is strictly prohibited; accepts canonical
   classification records computed by the ratified etfDenominatorEngine.
4. Derived Denominator Authority: Prospective denominator is derived dynamically:
   COUNT(DISTINCT observation_unit_id) WHERE classification_state = 'VALID' AND active epoch.
5. Inactive Epoch Invariant: When no active epoch exists, denominator is strictly 0.
"""

import os
import json
import hmac
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Header, status
from pydantic import BaseModel, Field

logger = logging.getLogger("api.telemetry")
router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~"))
DEFAULT_DB_PATH = os.path.join(DATA_DIR, ".finance_platform_history.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_telemetry_tables(db_path: str = DEFAULT_DB_PATH):
    """Initialize P1 telemetry tables if not already present."""
    migration_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "database", "migrations", "001_arx_p1_telemetry_tables.sql"
    )
    if os.path.exists(migration_path):
        with open(migration_path, "r", encoding="utf-8") as f:
            ddl = f.read()
        with get_db_connection(db_path) as conn:
            conn.executescript(ddl)
            conn.commit()


# Models for Ingestion & Query
class RawEventPayload(BaseModel):
    schema_version: str = "1.0.0"
    event_id: str
    session_id: str
    observation_unit_id: str
    deduplication_key: str
    timestamp: str
    normalized_symbol: str
    intent_type: str = "ETF_SYMBOL_SELECT"
    route: str = "/"
    source_component: str = "handleSelectSymbol"
    environment: Optional[str] = "production"
    deployment_identity: Optional[str] = "production-cloudflare-pages"
    release_sha: Optional[str] = None
    synthetic_marker: Optional[bool] = False
    ci_run_marker: Optional[bool] = False
    manual_qa_marker: Optional[bool] = False
    user_agent_raw: Optional[str] = ""


class AuditRecordPayload(BaseModel):
    observation_unit_id: str
    event_id: str
    session_id: str
    timestamp: str
    normalized_symbol: str
    environment: str
    deployment_identity: str
    release_sha: str
    traffic_class: str
    classification_state: str  # VALID, QUARANTINED, EXCLUDED, INVALID
    classification_reason: str
    exclusion_code: str
    quarantine_reason: str
    deduplication_key: str
    replay_identity: str
    source_component: str
    epoch_id: Optional[str] = None


class TelemetryPersistRequest(BaseModel):
    raw_event: RawEventPayload
    audit_record: AuditRecordPayload


def get_telemetry_internal_secret() -> Optional[str]:
    """Retrieve internal telemetry secret from environment.
    Fails closed: Returns None if variable is missing or empty.
    Zero hardcoded fallback credentials.
    """
    secret = os.getenv("TELEMETRY_INTERNAL_SECRET")
    if secret is None or not secret.strip():
        return None
    return secret.strip()


def verify_internal_telemetry_auth(
    x_internal_secret: Optional[str] = Header(None, alias="X-Internal-Secret")
) -> None:
    """Authenticate internal telemetry caller via X-Internal-Secret.

    Invariants:
    1. Server secret missing/empty -> HTTP 500 (Fail closed, server misconfiguration).
    2. Missing/empty header -> HTTP 401 Unauthorized.
    3. Wrong header -> HTTP 403 Forbidden.
    4. Constant-time comparison (hmac.compare_digest) to prevent timing attacks.
    5. Zero credential logging.
    """
    server_secret = get_telemetry_internal_secret()
    if server_secret is None:
        logger.error("Telemetry persistence rejected: server TELEMETRY_INTERNAL_SECRET is unconfigured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telemetry persistence service misconfigured: internal secret not configured.",
        )

    if not x_internal_secret or not x_internal_secret.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: missing internal telemetry secret header.",
        )

    if not hmac.compare_digest(x_internal_secret.strip().encode("utf-8"), server_secret.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: invalid internal telemetry secret.",
        )


@router.post("/persist", status_code=status.HTTP_200_OK)
def persist_telemetry_record(
    req: TelemetryPersistRequest,
    db_path: Optional[str] = None,
    x_internal_secret: Optional[str] = Header(None, alias="X-Internal-Secret"),
):
    """Persist raw event and canonical audit record in a single atomic transaction.

    Enforces:
    1. Internal authentication via X-Internal-Secret (fail-closed).
    2. Defense-in-depth on VALID records: Requires active epoch and exact authorized release match.
    3. Atomic database write via SQLite/Postgres transaction.
    """
    # 1. Enforce internal authentication fail-closed
    verify_internal_telemetry_auth(x_internal_secret)

    path = db_path or DEFAULT_DB_PATH
    init_telemetry_tables(path)

    raw = req.raw_event
    audit = req.audit_record

    # Verify ID consistency
    if raw.observation_unit_id != audit.observation_unit_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mismatched observation_unit_id between raw event and audit record.",
        )
    if raw.deduplication_key != audit.deduplication_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mismatched deduplication_key between raw event and audit record.",
        )

    conn = get_db_connection(path)
    try:
        with conn:
            # 2. Defense-in-depth on VALID records: enforce active epoch & authorized release bounds
            effective_epoch_id = audit.epoch_id
            if audit.classification_state == "VALID":
                epoch_row = conn.execute(
                    "SELECT epoch_id, name, start_timestamp, authorized_releases FROM arx_p1_observation_epochs WHERE is_active = 1 LIMIT 1"
                ).fetchone()

                if not epoch_row:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cannot persist VALID audit record: no active observation epoch exists.",
                    )

                active_epoch_id = epoch_row["epoch_id"]
                try:
                    authorized_list = json.loads(epoch_row["authorized_releases"])
                    if not isinstance(authorized_list, list):
                        authorized_list = []
                except Exception:
                    authorized_list = []

                target_sha = (audit.release_sha or "").lower().strip()
                authorized_set = {str(r).lower().strip() for r in authorized_list}

                if not target_sha or target_sha not in authorized_set:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot persist VALID audit record: release_sha '{audit.release_sha}' is not authorized in active epoch '{active_epoch_id}'.",
                    )

                if audit.epoch_id and audit.epoch_id != active_epoch_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cannot persist VALID audit record: provided epoch_id '{audit.epoch_id}' does not match active epoch '{active_epoch_id}'.",
                    )

                effective_epoch_id = audit.epoch_id or active_epoch_id

            # 3. Insert raw event intake record
            conn.execute(
                """
                INSERT OR IGNORE INTO arx_p1_telemetry_raw_events (
                    event_id, session_id, attempt_id, deduplication_key, event_timestamp, raw_payload, user_agent_raw
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw.event_id,
                    raw.session_id,
                    raw.observation_unit_id,
                    raw.deduplication_key,
                    raw.timestamp,
                    json.dumps(raw.model_dump()),
                    raw.user_agent_raw,
                ),
            )

            # 4. Insert canonical audit record with unique constraint protection
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO arx_p1_telemetry_audit_records (
                    observation_unit_id, event_id, session_id, timestamp, normalized_symbol,
                    environment, deployment_identity, release_sha, traffic_class, classification_state,
                    classification_reason, exclusion_code, quarantine_reason, deduplication_key,
                    replay_identity, source_component, epoch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (deduplication_key) DO NOTHING
                """,
                (
                    audit.observation_unit_id,
                    audit.event_id,
                    audit.session_id,
                    audit.timestamp,
                    audit.normalized_symbol.upper(),
                    audit.environment,
                    audit.deployment_identity,
                    audit.release_sha,
                    audit.traffic_class,
                    audit.classification_state,
                    audit.classification_reason,
                    audit.exclusion_code,
                    audit.quarantine_reason,
                    audit.deduplication_key,
                    audit.replay_identity,
                    audit.source_component,
                    effective_epoch_id,
                ),
            )

            if cursor.rowcount == 0:
                # Deduplication key already existed; idempotent duplicate delivery
                return {
                    "status": "DUPLICATE",
                    "observation_unit_id": audit.observation_unit_id,
                    "deduplication_key": audit.deduplication_key,
                    "classification_state": "EXCLUDED",
                    "exclusion_code": "DUPLICATE",
                }

        return {
            "status": "PERSISTED",
            "observation_unit_id": audit.observation_unit_id,
            "deduplication_key": audit.deduplication_key,
            "classification_state": audit.classification_state,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to atomically persist telemetry record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Telemetry persistence error: {str(e)}",
        )
    finally:
        conn.close()


@router.get("/denominator")
def get_prospective_denominator(db_path: Optional[str] = None):
    """Derived canonical denominator authority:
    COUNT(DISTINCT observation_unit_id) WHERE classification_state = 'VALID' AND active epoch.
    """
    path = db_path or DEFAULT_DB_PATH
    init_telemetry_tables(path)

    conn = get_db_connection(path)
    try:
        # Check active epoch
        epoch_row = conn.execute(
            "SELECT epoch_id, name, start_timestamp FROM arx_p1_observation_epochs WHERE is_active = 1 LIMIT 1"
        ).fetchone()

        if not epoch_row:
            return {
                "prospective_denominator": 0,
                "active_epoch": None,
                "epoch_status": "NOT_ESTABLISHED",
                "authority": "DERIVED_FROM_CANONICAL_AUDIT_LEDGER",
            }

        active_epoch_id = epoch_row["epoch_id"]
        start_ts = epoch_row["start_timestamp"]

        count_row = conn.execute(
            """
            SELECT COUNT(DISTINCT observation_unit_id) as valid_count
            FROM arx_p1_telemetry_audit_records
            WHERE classification_state = 'VALID'
              AND timestamp >= ?
              AND (epoch_id = ? OR epoch_id IS NULL)
            """,
            (start_ts, active_epoch_id),
        ).fetchone()

        valid_count = count_row["valid_count"] if count_row else 0
        return {
            "prospective_denominator": valid_count,
            "active_epoch": active_epoch_id,
            "epoch_status": "ACTIVE",
            "authority": "DERIVED_FROM_CANONICAL_AUDIT_LEDGER",
        }
    finally:
        conn.close()


@router.get("/audit")
def get_audit_records(
    state: Optional[str] = Query(None, description="Filter by classification state (VALID, QUARANTINED, EXCLUDED, INVALID)"),
    symbol: Optional[str] = Query(None, description="Filter by normalized ticker"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db_path: Optional[str] = None,
):
    """Read-only audit path for governance verification."""
    path = db_path or DEFAULT_DB_PATH
    init_telemetry_tables(path)

    conn = get_db_connection(path)
    try:
        query = "SELECT * FROM arx_p1_telemetry_audit_records WHERE 1=1"
        params: List[Any] = []

        if state:
            query += " AND classification_state = ?"
            params.append(state.upper())
        if symbol:
            query += " AND normalized_symbol = ?"
            params.append(symbol.upper())
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return {
            "count": len(rows),
            "records": [dict(r) for r in rows],
        }
    finally:
        conn.close()


@router.get("/epochs")
def get_observation_epochs(db_path: Optional[str] = None):
    """Retrieve configured observation epochs."""
    path = db_path or DEFAULT_DB_PATH
    init_telemetry_tables(path)

    conn = get_db_connection(path)
    try:
        rows = conn.execute("SELECT * FROM arx_p1_observation_epochs ORDER BY created_at DESC").fetchall()
        return {"epochs": [dict(r) for r in rows]}
    finally:
        conn.close()
