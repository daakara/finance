"""FastAPI Router for Corporate Events RC5 Durable Telemetry Persistence (Increment 2B).

Guarantees:
1. Sole Persistence Authority: Atomically commits deduplicated evaluation records and raw delivery attempts.
2. Dual Uniqueness Enforcement:
   - UNIQUE (release_sha, evaluation_id) for logical at-most-once evaluation counting.
   - UNIQUE (delivery_attempt_id) for transport delivery packet deduplication.
3. Verification-Before-Confirmation: On unique constraint collision, explicitly reads back committed record
   and verifies matching canonical_digest before acknowledging CONFIRMED_DUPLICATE.
4. Fail-Closed Feature Flag: Ingestion rejected with status DISABLED when ARX_PR53_TELEMETRY_ENABLED is false.
5. Strict Natural Denominator Isolation: Synthetics, test replays, and unverified records receive is_natural_candidate = 0.
   REAL_NATURAL_PRODUCTION_DENOMINATOR remains NOT_OBSERVABLE.
"""

from __future__ import annotations

import os
import json
import hmac
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Header, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from analyst_dashboard.telemetry.corporate_event_contract import (
    CorporateEventTelemetryPayload,
    compute_canonical_digest,
    derive_authoritative_evaluation_id,
)
from analyst_dashboard.telemetry.migration_rc5 import (
    apply_rc5_telemetry_migration,
    verify_rc5_telemetry_schema_compatibility,
    get_default_migration_path,
)

logger = logging.getLogger("api.telemetry.rc5")
router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~"))
DEFAULT_DB_PATH = os.path.join(DATA_DIR, ".finance_platform_history.db")
SYNTHETIC_TEST_SECRET = "arx_synthetic_test_internal_secret"

MAX_EVENT_AGE_MS = 24 * 60 * 60 * 1000  # 24-hour replay window
MAX_FUTURE_SKEW_MS = 60 * 1000          # 60-second future clock skew


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 30000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.OperationalError:
        pass
    return conn


def is_telemetry_enabled() -> bool:
    """Checks whether Corporate Events telemetry persistence is enabled via environment."""
    val = os.getenv("ARX_PR53_TELEMETRY_ENABLED", "").lower().strip()
    return val in ("true", "1", "yes")


def get_expected_internal_secret() -> Optional[str]:
    """Retrieves server internal secret from environment, with synthetic test fallback support."""
    secret = os.getenv("TELEMETRY_INTERNAL_SECRET")
    if secret and secret.strip():
        return secret.strip()
    if os.getenv("ARX_TEST_ENVIRONMENT") == "1":
        return SYNTHETIC_TEST_SECRET
    return None


def verify_internal_secret_auth(x_internal_secret: Optional[str] = Header(None, alias="X-Internal-Secret")) -> None:
    expected = get_expected_internal_secret()
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telemetry persistence service misconfigured: internal secret not set.",
        )
    if not x_internal_secret or not x_internal_secret.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-Internal-Secret header.",
        )
    if not hmac.compare_digest(x_internal_secret.strip().encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: invalid internal telemetry secret.",
        )


class EnvelopePayloadWrapper(BaseModel):
    raw_event: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    delivery_attempt_id: Optional[str] = None
    client_timestamp: Optional[str] = None
    synthetic_marker: Optional[bool] = None


@router.post("/persist", status_code=status.HTTP_200_OK)
@router.post("/events", status_code=status.HTTP_200_OK)
def persist_corporate_event(
    body: Dict[str, Any],
    db_path: Optional[str] = None,
    x_internal_secret: Optional[str] = Header(None, alias="X-Internal-Secret"),
):
    """Atomically ingests, verifies, and persists a Corporate Events telemetry record.
    
    Adheres strictly to the 11-state acknowledgment model and fail-closed operational invariants.
    """
    path = db_path or os.getenv("ARX_TELEMETRY_DB_PATH") or DEFAULT_DB_PATH

    # 1. Feature Flag Precedence (Fail Closed)
    if not is_telemetry_enabled():
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "contract_version": "1.1.0",
                "status": "DISABLED",
                "accepted": False,
                "forwarded": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "message": "Corporate Events telemetry persistence is disabled by policy",
            },
        )

    # 2. Internal Secret Verification
    verify_internal_secret_auth(x_internal_secret)

    # 3. Unpack event payload
    raw_event = body.get("raw_event") or body.get("payload") or body
    delivery_attempt_id = (
        body.get("delivery_attempt_id")
        or raw_event.get("delivery_attempt_id")
        or f"deliv-{datetime.now(timezone.utc).timestamp()}"
    )

    # 4. Replay Window & Clock Skew Bounds
    occurred_at_str = raw_event.get("occurred_at_utc")
    if not occurred_at_str or "T" not in str(occurred_at_str):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "contract_version": "1.1.0",
                "status": "REJECTED",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "error_code": "INVALID_TIMESTAMP",
                "message": "occurred_at_utc must be a valid ISO 8601 UTC timestamp",
            },
        )

    try:
        # Handle ISO timestamp parsing
        clean_ts = str(occurred_at_str).replace("Z", "+00:00")
        occurred_dt = datetime.fromisoformat(clean_ts)
        if occurred_dt.tzinfo is None:
            occurred_dt = occurred_dt.replace(tzinfo=timezone.utc)
        occurred_at_ms = int(occurred_dt.timestamp() * 1000)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "contract_version": "1.1.0",
                "status": "REJECTED",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "error_code": "INVALID_TIMESTAMP",
                "message": "occurred_at_utc failed datetime parsing",
            },
        )

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if now_ms - occurred_at_ms > MAX_EVENT_AGE_MS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "contract_version": "1.1.0",
                "status": "REJECTED",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "error_code": "EXPIRED_REPLAY_WINDOW",
                "message": "Event age exceeds 24-hour replay window horizon",
            },
        )
    if occurred_at_ms - now_ms > MAX_FUTURE_SKEW_MS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "contract_version": "1.1.0",
                "status": "REJECTED",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "error_code": "FUTURE_TIMESTAMP_REJECTED",
                "message": "Event timestamp exceeds 60-second future clock skew bound",
            },
        )

    # 5. Schema Validation & 12-Field Semantic Digest Calculation
    try:
        validated_model = CorporateEventTelemetryPayload.model_validate(raw_event)
    except Exception as val_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "contract_version": "1.1.0",
                "status": "REJECTED",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": False,
                "error_code": "SCHEMA_VALIDATION_ERROR",
                "message": f"Schema validation error: {val_err}",
            },
        )

    canonical_digest = compute_canonical_digest(raw_event)
    release_sha = str(validated_model.release_sha).lower().strip()

    # Determine or verify evaluation_id
    evaluation_id = validated_model.evaluation_id
    if not evaluation_id:
        evaluation_id = derive_authoritative_evaluation_id(
            release_sha=release_sha,
            symbol=validated_model.symbol,
            ruleset_version=validated_model.ruleset_version,
            event_time_utc=validated_model.event_time_utc,
            market_session=str(validated_model.market_session.value if hasattr(validated_model.market_session, "value") else validated_model.market_session),
            evaluation_timestamp=validated_model.occurred_at_utc,
        )

    # 6. Database Connection & Schema Compatibility Verification
    conn = get_db_connection(path)
    try:
        compatible, compat_errors = verify_rc5_telemetry_schema_compatibility(conn)
        if not compatible:
            logger.error(f"Schema compatibility check failed: {compat_errors}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "contract_version": "1.1.0",
                    "status": "TEMPORARILY_UNAVAILABLE",
                    "accepted": False,
                    "persisted": "NOT_PERSISTED",
                    "retryable": True,
                    "error_code": "SCHEMA_INCOMPATIBLE",
                    "message": f"Database schema incompatibility detected: {'; '.join(compat_errors)}",
                },
            )

        now_utc_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 7. Atomic Transaction: Check for duplicates or conflict
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT canonical_digest, retransmission_count FROM arx_rc5_telemetry_evaluations WHERE release_sha = ? AND evaluation_id = ?",
                (release_sha, evaluation_id),
            )
            existing_row = cursor.fetchone()

            if existing_row:
                stored_digest = existing_row["canonical_digest"]
                if stored_digest == canonical_digest:
                    # Confirmed Duplicate: identical identity and identical semantic content
                    new_count = existing_row["retransmission_count"] + 1
                    conn.execute(
                        "UPDATE arx_rc5_telemetry_evaluations SET retransmission_count = ? WHERE release_sha = ? AND evaluation_id = ?",
                        (new_count, release_sha, evaluation_id),
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO arx_rc5_telemetry_raw_events (
                            delivery_attempt_id, deduplication_key, evaluation_id, release_sha,
                            received_at_utc, edge_timestamp_utc, raw_payload_json, ingestion_verdict
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            delivery_attempt_id,
                            validated_model.deduplication_key or f"dedup-{delivery_attempt_id}",
                            evaluation_id,
                            release_sha,
                            now_utc_str,
                            raw_event.get("client_timestamp"),
                            json.dumps(raw_event),
                            "CONFIRMED_DUPLICATE",
                        ),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "contract_version": "1.1.0",
                            "status": "CONFIRMED_DUPLICATE",
                            "accepted": True,
                            "persisted": "CONFIRMED",
                            "retryable": False,
                            "evaluation_id": evaluation_id,
                            "delivery_attempt_id": delivery_attempt_id,
                            "canonical_digest": canonical_digest,
                            "retransmission_count": new_count,
                            "message": "Authoritative duplicate acknowledged; persistence verified from prior submission",
                        },
                    )
                else:
                    # Conflicting Identity: same ID claimed by divergent semantic payload!
                    conn.execute(
                        """
                        INSERT INTO arx_rc5_telemetry_quarantine (
                            quarantine_id, quarantined_at_utc, reason_code, details, delivery_attempt_id, evaluation_id, release_sha, raw_payload_snippet
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"quar-{datetime.now(timezone.utc).timestamp()}",
                            now_utc_str,
                            "CONFLICTING_IDENTITY",
                            f"Digest mismatch: incoming '{canonical_digest}' != stored '{stored_digest}'",
                            delivery_attempt_id,
                            evaluation_id,
                            release_sha,
                            json.dumps(raw_event)[:500],
                        ),
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO arx_rc5_telemetry_raw_events (
                            delivery_attempt_id, deduplication_key, evaluation_id, release_sha,
                            received_at_utc, edge_timestamp_utc, raw_payload_json, ingestion_verdict
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            delivery_attempt_id,
                            validated_model.deduplication_key or f"dedup-{delivery_attempt_id}",
                            evaluation_id,
                            release_sha,
                            now_utc_str,
                            raw_event.get("client_timestamp"),
                            json.dumps(raw_event),
                            "REJECTED",
                        ),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_409_CONFLICT,
                        content={
                            "contract_version": "1.1.0",
                            "status": "CONFLICTING_IDENTITY",
                            "accepted": False,
                            "persisted": "NOT_PERSISTED",
                            "retryable": False,
                            "evaluation_id": evaluation_id,
                            "delivery_attempt_id": delivery_attempt_id,
                            "error_code": "CONFLICTING_IDENTITY",
                            "message": "Evaluation ID reused with conflicting immutable attributes",
                        },
                    )

            # 8. Check observation epoch and derive natural candidate eligibility
            is_natural_candidate = 0
            epoch_id: Optional[str] = None

            is_synthetic = (
                bool(validated_model.synthetic_marker)
                or bool(validated_model.ci_marker)
                or bool(raw_event.get("synthetic_marker"))
                or bool(raw_event.get("ci_marker"))
                or bool(raw_event.get("manual_qa_marker"))
            )

            bot_score = raw_event.get("bot_score")
            bot_ok = bot_score is None or (isinstance(bot_score, (int, float)) and bot_score >= 30)

            cursor.execute(
                "SELECT epoch_id, authorized_release_sha, start_timestamp_utc, end_timestamp_utc FROM arx_rc5_observation_epochs WHERE is_active = 1 LIMIT 1"
            )
            active_epoch = cursor.fetchone()

            if active_epoch and not is_synthetic and bot_ok:
                auth_sha = active_epoch["authorized_release_sha"].lower().strip()
                start_ts = active_epoch["start_timestamp_utc"]
                end_ts = active_epoch["end_timestamp_utc"]
                if release_sha == auth_sha and start_ts <= validated_model.occurred_at_utc <= end_ts:
                    is_natural_candidate = 1
                    epoch_id = active_epoch["epoch_id"]

            # 9. Insert brand-new authoritative evaluation row
            conn.execute(
                """
                INSERT INTO arx_rc5_telemetry_evaluations (
                    release_sha, evaluation_id, canonical_digest, schema_version, event_id, occurred_at_utc,
                    ruleset_version, evaluation_context_id, symbol, event_category, event_time_utc,
                    market_session, event_source_status, freshness_status, days_to_earnings, risk_bucket,
                    applied_penalty, evaluation_outcome, first_received_at_utc, retransmission_count,
                    is_natural_candidate, epoch_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    release_sha,
                    evaluation_id,
                    canonical_digest,
                    validated_model.schema_version,
                    validated_model.event_id,
                    validated_model.occurred_at_utc,
                    validated_model.ruleset_version,
                    validated_model.evaluation_context_id,
                    validated_model.symbol,
                    str(validated_model.event_category.value if hasattr(validated_model.event_category, "value") else validated_model.event_category),
                    validated_model.event_time_utc,
                    str(validated_model.market_session.value if hasattr(validated_model.market_session, "value") else validated_model.market_session),
                    str(validated_model.event_source_status.value if hasattr(validated_model.event_source_status, "value") else validated_model.event_source_status),
                    str(validated_model.freshness_status.value if hasattr(validated_model.freshness_status, "value") else validated_model.freshness_status),
                    validated_model.days_to_earnings,
                    str(validated_model.risk_bucket.value if hasattr(validated_model.risk_bucket, "value") else validated_model.risk_bucket),
                    validated_model.applied_penalty,
                    str(validated_model.evaluation_outcome.value if hasattr(validated_model.evaluation_outcome, "value") else validated_model.evaluation_outcome),
                    now_utc_str,
                    0,
                    is_natural_candidate,
                    epoch_id,
                ),
            )

            # 10. Record raw delivery attempt
            conn.execute(
                """
                INSERT OR IGNORE INTO arx_rc5_telemetry_raw_events (
                    delivery_attempt_id, deduplication_key, evaluation_id, release_sha,
                    received_at_utc, edge_timestamp_utc, raw_payload_json, ingestion_verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_attempt_id,
                    validated_model.deduplication_key or f"dedup-{delivery_attempt_id}",
                    evaluation_id,
                    release_sha,
                    now_utc_str,
                    raw_event.get("client_timestamp"),
                    json.dumps(raw_event),
                    "PERSISTED",
                ),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "contract_version": "1.1.0",
                "status": "PERSISTED",
                "accepted": True,
                "persisted": "CONFIRMED",
                "retryable": False,
                "evaluation_id": evaluation_id,
                "delivery_attempt_id": delivery_attempt_id,
                "canonical_digest": canonical_digest,
                "is_natural_candidate": is_natural_candidate,
                "message": "Persisted upstream successfully",
            },
        )
    except sqlite3.IntegrityError:
        # Concurrent collision resolution:
        # A concurrent worker committed the (release_sha, evaluation_id) row between our SELECT and INSERT.
        # Re-query under a new transaction to verify whether it is a confirmed duplicate or conflicting mutation.
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT canonical_digest, retransmission_count FROM arx_rc5_telemetry_evaluations WHERE release_sha = ? AND evaluation_id = ?",
                (release_sha, evaluation_id),
            )
            collided_row = cursor.fetchone()
            if collided_row:
                stored_digest = collided_row["canonical_digest"]
                if stored_digest == canonical_digest:
                    new_count = collided_row["retransmission_count"] + 1
                    conn.execute(
                        "UPDATE arx_rc5_telemetry_evaluations SET retransmission_count = ? WHERE release_sha = ? AND evaluation_id = ?",
                        (new_count, release_sha, evaluation_id),
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO arx_rc5_telemetry_raw_events (
                            delivery_attempt_id, deduplication_key, evaluation_id, release_sha,
                            received_at_utc, edge_timestamp_utc, raw_payload_json, ingestion_verdict
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            delivery_attempt_id,
                            validated_model.deduplication_key or f"dedup-{delivery_attempt_id}",
                            evaluation_id,
                            release_sha,
                            now_utc_str,
                            raw_event.get("client_timestamp"),
                            json.dumps(raw_event),
                            "CONFIRMED_DUPLICATE",
                        ),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "contract_version": "1.1.0",
                            "status": "CONFIRMED_DUPLICATE",
                            "accepted": True,
                            "persisted": "CONFIRMED",
                            "retryable": False,
                            "evaluation_id": evaluation_id,
                            "delivery_attempt_id": delivery_attempt_id,
                            "canonical_digest": canonical_digest,
                            "retransmission_count": new_count,
                            "message": "Authoritative duplicate acknowledged; persistence verified from prior submission",
                        },
                    )
                else:
                    # Conflicting identity detected under concurrent collision
                    conn.execute(
                        """
                        INSERT INTO arx_rc5_telemetry_quarantine (
                            quarantine_id, quarantined_at_utc, reason_code, details, delivery_attempt_id, evaluation_id, release_sha, raw_payload_snippet
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"quar-{datetime.now(timezone.utc).timestamp()}",
                            now_utc_str,
                            "CONFLICTING_IDENTITY",
                            f"Digest mismatch on concurrent collision: incoming '{canonical_digest}' != stored '{stored_digest}'",
                            delivery_attempt_id,
                            evaluation_id,
                            release_sha,
                            json.dumps(raw_event)[:500],
                        ),
                    )
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO arx_rc5_telemetry_raw_events (
                            delivery_attempt_id, deduplication_key, evaluation_id, release_sha,
                            received_at_utc, edge_timestamp_utc, raw_payload_json, ingestion_verdict
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            delivery_attempt_id,
                            validated_model.deduplication_key or f"dedup-{delivery_attempt_id}",
                            evaluation_id,
                            release_sha,
                            now_utc_str,
                            raw_event.get("client_timestamp"),
                            json.dumps(raw_event),
                            "REJECTED",
                        ),
                    )
                    return JSONResponse(
                        status_code=status.HTTP_409_CONFLICT,
                        content={
                            "contract_version": "1.1.0",
                            "status": "CONFLICTING_IDENTITY",
                            "accepted": False,
                            "persisted": "NOT_PERSISTED",
                            "retryable": False,
                            "evaluation_id": evaluation_id,
                            "delivery_attempt_id": delivery_attempt_id,
                            "error_code": "CONFLICTING_IDENTITY",
                            "message": "Evaluation ID reused with conflicting immutable attributes",
                        },
                    )
            raise
    except sqlite3.OperationalError as op_err:
        if "locked" in str(op_err).lower() or "busy" in str(op_err).lower():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "contract_version": "1.1.0",
                    "status": "TEMPORARILY_UNAVAILABLE",
                    "accepted": False,
                    "persisted": "NOT_PERSISTED",
                    "retryable": True,
                    "error_code": "STORAGE_ENGINE_BUSY",
                    "message": "Storage engine lock contention: database is busy or locked",
                },
            )
        logger.error(f"Operational error during persistence: {op_err}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "contract_version": "1.1.0",
                "status": "TEMPORARILY_UNAVAILABLE",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": True,
                "error_code": "DATABASE_OPERATIONAL_ERROR",
                "message": "Internal storage operational error",
            },
        )
    except Exception as e:
        logger.error(f"Unexpected persistence error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "contract_version": "1.1.0",
                "status": "TEMPORARILY_UNAVAILABLE",
                "accepted": False,
                "persisted": "NOT_PERSISTED",
                "retryable": True,
                "error_code": "STORAGE_EXCEPTION",
                "message": "Unexpected storage exception encountered",
            },
        )
    finally:
        conn.close()


@router.get("/denominator", status_code=status.HTTP_200_OK)
def get_natural_denominator(db_path: Optional[str] = None):
    """Returns authoritative eligible natural evaluation denominator count.
    
    Guarantees:
    - Disabled by policy: returns 0.
    - No active epoch: returns 0.
    - Synthetics / replays: excluded.
    - In accordance with PR #53 release baseline, REAL_NATURAL_PRODUCTION_DENOMINATOR = NOT_OBSERVABLE.
    """
    path = db_path or os.getenv("ARX_TELEMETRY_DB_PATH") or DEFAULT_DB_PATH

    if not is_telemetry_enabled():
        return {
            "status": "DISABLED",
            "natural_eligible_denominator": 0,
            "real_natural_production_denominator": "NOT_OBSERVABLE",
            "message": "Corporate Events telemetry persistence is disabled by policy",
        }

    conn = get_db_connection(path)
    try:
        compatible, _ = verify_rc5_telemetry_schema_compatibility(conn)
        if not compatible:
            return {
                "status": "INCOMPATIBLE_SCHEMA",
                "natural_eligible_denominator": 0,
                "real_natural_production_denominator": "NOT_OBSERVABLE",
            }

        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM arx_rc5_telemetry_evaluations WHERE is_natural_candidate = 1"
        )
        count = cursor.fetchone()[0]

        return {
            "status": "ACTIVE",
            "natural_eligible_denominator": count,
            "real_natural_production_denominator": "NOT_OBSERVABLE",
            "observation_status": "PR53_RELEASE_CLOSED_PRODUCTION_VERIFIED",
        }
    finally:
        conn.close()
