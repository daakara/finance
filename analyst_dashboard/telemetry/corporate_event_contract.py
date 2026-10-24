"""Corporate Events Telemetry Contract (v1.0.0).

Strict Pydantic schema enforcing the 17 non-identifying telemetry fields,
strict field-level type validation, privacy deny-list enforcement,
and corporate events semantic risk invariants.
"""

from __future__ import annotations

import re
import uuid
import hashlib
import hmac
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Set
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Non-negotiable privacy deny-list: Zero tolerance for identifying information
PROHIBITED_PRIVACY_FIELDS: Set[str] = {
    "user_id",
    "userid",
    "user_token",
    "ip",
    "ip_address",
    "client_ip",
    "remote_addr",
    "email",
    "account_id",
    "accountid",
    "account_number",
    "account_balance",
    "balance",
    "portfolio_value",
    "trade_size",
    "dollar_amount",
    "cookie",
    "session_cookie",
    "device_id",
    "device_fingerprint",
    "canvas_fingerprint",
    "name",
    "personal_name",
    "full_name",
    "first_name",
    "last_name",
    "phone",
    "address",
    "ssn",
    "auth_token",
    "bearer_token",
    "password",
}

SHA_REGEX = re.compile(r"^[0-9a-f]{40}$")
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{1,5}$")


class CorporateEventCategory(str, Enum):
    EARNINGS_ANNOUNCEMENT = "EARNINGS_ANNOUNCEMENT"
    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    ACQUISITION = "ACQUISITION"
    NONE = "NONE"
    UNAVAILABLE = "UNAVAILABLE"


class MarketSession(str, Enum):
    BMO = "BMO"
    AMC = "AMC"
    DURING = "DURING"
    UNKNOWN = "UNKNOWN"


class EventSourceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNVERIFIED = "UNVERIFIED"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class FreshnessStatus(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNVERIFIED = "UNVERIFIED"


class RiskBucket(str, Enum):
    ACUTE = "ACUTE"
    CAUTION = "CAUTION"
    NOMINAL = "NOMINAL"
    UNAVAILABLE = "UNAVAILABLE"


class EvaluationOutcome(str, Enum):
    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


def compute_deduplication_key(
    evaluation_context_id: str,
    symbol: str,
    event_time_utc: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    occurred_at_utc: Optional[str] = None,
    salt: str = "arx_telemetry_corporate_events_salt_v1",
) -> str:
    """Deterministically compute deduplication key for an evaluation instance.

    Guarantees:
    - Retransmissions of the same evaluation produce the identical key.
    - Legitimate repeated evaluations in different contexts or different runs produce distinct keys.
    """
    key_material = [
        f"ctx:{evaluation_context_id}",
        f"sym:{symbol.upper()}",
        f"event:{event_time_utc or 'none'}",
    ]
    if evaluation_id:
        key_material.append(f"eval_id:{evaluation_id}")
    elif occurred_at_utc:
        key_material.append(f"time:{occurred_at_utc}")

    raw_message = "|".join(key_material).encode("utf-8")
    return hmac.new(salt.encode("utf-8"), raw_message, hashlib.sha256).hexdigest()


class CorporateEventTelemetryPayload(BaseModel):
    """The 17-field canonical telemetry event payload for Corporate Events RC5."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # 1. Schema version
    schema_version: str = Field(
        default="1.0.0", description="Semver version string of telemetry contract"
    )

    # 2. Event ID (unique delivery / message identifier)
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Random UUIDv4 message identifier",
    )

    # 3. Occurred timestamp
    occurred_at_utc: str = Field(
        ..., description="ISO 8601 UTC timestamp of client evaluation"
    )

    # 4. Release SHA
    release_sha: str = Field(
        ..., description="40-hex Git commit SHA of deployed build"
    )

    # 5. Ruleset version
    ruleset_version: str = Field(
        ..., description="Ruleset version identifier ('RC5' or 'RC4')"
    )

    # 6. Evaluation context ID
    evaluation_context_id: str = Field(
        ..., description="Ephemeral anonymous evaluation context identifier"
    )

    # 7. Asset symbol
    symbol: str = Field(..., description="1-5 char uppercase equity ticker")

    # 8. Event category
    event_category: CorporateEventCategory = Field(
        ..., description="Catalyst classification category"
    )

    # 9. Catalyst time
    event_time_utc: Optional[str] = Field(
        default=None, description="ISO 8601 UTC timestamp of event, or None"
    )

    # 10. Market session
    market_session: MarketSession = Field(
        default=MarketSession.UNKNOWN,
        description="Market session (BMO, AMC, DURING, UNKNOWN)",
    )

    # 11. Event source status
    event_source_status: EventSourceStatus = Field(
        ..., description="Confirmation state of primary event feed"
    )

    # 12. Freshness status
    freshness_status: FreshnessStatus = Field(
        ..., description="Freshness status of underlying event feed"
    )

    # 13. Days to event delta t
    days_to_earnings: Optional[float] = Field(
        default=None, description="Signed floating point days to event"
    )

    # 14. Categorical risk bucket
    risk_bucket: RiskBucket = Field(
        ..., description="Categorical risk bucket for UI and confluence"
    )

    # 15. Applied penalty
    applied_penalty: float = Field(
        ..., description="Confluence score deduction (-25.0, -8.0, 0.0)"
    )

    # 16. Evaluation outcome
    evaluation_outcome: EvaluationOutcome = Field(
        ..., description="Execution status of evaluation"
    )

    # 17. Provenance classification (optional in client envelope, resolved by collector)
    provenance_class: Optional[str] = Field(
        default=None, description="Determined provenance classification"
    )

    # Optional delivery & envelope metadata (proportional, non-identifying)
    evaluation_id: Optional[str] = Field(
        default=None,
        description="Evaluation instance ID to distinguish separate runs",
    )
    delivery_attempt_id: Optional[str] = Field(
        default=None,
        description="Delivery attempt ID for network retry tracking",
    )
    deduplication_key: Optional[str] = Field(
        default=None,
        description="Cryptographic deduplication hash",
    )
    collector_received_at_utc: Optional[str] = Field(
        default=None,
        description="Edge ingress timestamp set by collector",
    )
    deployment_host: Optional[str] = Field(
        default=None,
        description="Origin host of evaluation (e.g. arxterminal.com)",
    )
    synthetic_marker: bool = Field(
        default=False,
        description="Explicit synthetic diagnostic marker",
    )
    ci_marker: bool = Field(
        default=False,
        description="Automated smoke / CI test marker",
    )
    bot_score: Optional[int] = Field(
        default=None,
        description="Edge bot management verification score (0-100)",
    )

    @model_validator(mode="before")
    @classmethod
    def check_prohibited_privacy_fields(cls, data: Any) -> Any:
        """Strictly reject any payload containing prohibited PII or identifying fields."""
        if isinstance(data, dict):
            for key in data.keys():
                normalized_key = key.lower().replace("-", "_")
                if normalized_key in PROHIBITED_PRIVACY_FIELDS:
                    raise ValueError(
                        f"Prohibited privacy field detected: '{key}'. "
                        f"Zero-tolerance privacy boundary violated. Payload strictly rejected."
                    )
        return data

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v not in ("1.0.0", "1.1.0"):
            raise ValueError(f"Unsupported schema version: '{v}'. Expected '1.0.0' or '1.1.0'")
        return v

    @field_validator("release_sha")
    @classmethod
    def validate_release_sha(cls, v: str) -> str:
        if not SHA_REGEX.match(v.lower()):
            raise ValueError(
                f"Invalid release_sha: '{v}'. Must be exactly 40 lowercase hexadecimal characters."
            )
        return v.lower()

    @field_validator("ruleset_version")
    @classmethod
    def validate_ruleset_version(cls, v: str) -> str:
        if v not in ("RC5", "RC4"):
            raise ValueError(f"Invalid ruleset_version: '{v}'. Must be 'RC5' or 'RC4'")
        return v

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if not SYMBOL_REGEX.match(v_upper):
            raise ValueError(
                f"Invalid symbol: '{v}'. Must be 1 to 5 uppercase alphanumeric characters."
            )
        return v_upper

    @field_validator("applied_penalty")
    @classmethod
    def validate_applied_penalty(cls, v: float) -> float:
        if v not in (-25.0, -8.0, 0.0):
            raise ValueError(
                f"Invalid applied_penalty: {v}. Must be exactly -25.0, -8.0, or 0.0"
            )
        return float(v)

    @field_validator("days_to_earnings")
    @classmethod
    def round_days_to_earnings(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return round(float(v), 2)
        return None

    @model_validator(mode="after")
    def validate_semantic_invariants_and_keys(self) -> CorporateEventTelemetryPayload:
        """Enforce semantic invariants between days_to_earnings, risk_bucket, and applied_penalty."""
        # 1. Penalty must match Risk Bucket exactly
        if self.risk_bucket == RiskBucket.ACUTE and self.applied_penalty != -25.0:
            raise ValueError(
                f"Semantic mismatch: risk_bucket ACUTE requires applied_penalty -25.0, got {self.applied_penalty}"
            )
        if self.risk_bucket == RiskBucket.CAUTION and self.applied_penalty != -8.0:
            raise ValueError(
                f"Semantic mismatch: risk_bucket CAUTION requires applied_penalty -8.0, got {self.applied_penalty}"
            )
        if self.risk_bucket in (RiskBucket.NOMINAL, RiskBucket.UNAVAILABLE) and self.applied_penalty != 0.0:
            raise ValueError(
                f"Semantic mismatch: risk_bucket {self.risk_bucket.value} requires applied_penalty 0.0, got {self.applied_penalty}"
            )

        # 2. Risk Bucket must match days_to_earnings boundaries
        if self.days_to_earnings is not None:
            dt = self.days_to_earnings
            if 0.0 <= dt <= 1.0:
                if self.risk_bucket != RiskBucket.ACUTE:
                    raise ValueError(
                        f"Semantic boundary violation: 0.0 <= days_to_earnings ({dt}) <= 1.0 must be ACUTE, got {self.risk_bucket.value}"
                    )
            elif 1.0 < dt <= 7.0:
                if self.risk_bucket != RiskBucket.CAUTION:
                    raise ValueError(
                        f"Semantic boundary violation: 1.0 < days_to_earnings ({dt}) <= 7.0 must be CAUTION, got {self.risk_bucket.value}"
                    )
            elif dt > 7.0:
                if self.risk_bucket != RiskBucket.NOMINAL:
                    raise ValueError(
                        f"Semantic boundary violation: days_to_earnings ({dt}) > 7.0 must be NOMINAL, got {self.risk_bucket.value}"
                    )
            elif dt < 0.0:
                if self.risk_bucket != RiskBucket.NOMINAL:
                    raise ValueError(
                        f"Semantic boundary violation: elapsed event ({dt} < 0.0) must be NOMINAL, got {self.risk_bucket.value}"
                    )
        else:
            if self.risk_bucket not in (RiskBucket.UNAVAILABLE, RiskBucket.NOMINAL):
                raise ValueError(
                    f"Missing earnings date requires risk_bucket UNAVAILABLE or NOMINAL, got {self.risk_bucket.value}"
                )

        # 3. Ensure deduplication key is populated
        if not self.deduplication_key:
            self.deduplication_key = compute_deduplication_key(
                evaluation_context_id=self.evaluation_context_id,
                symbol=self.symbol,
                event_time_utc=self.event_time_utc,
                evaluation_id=self.evaluation_id,
                occurred_at_utc=self.occurred_at_utc,
            )

        return self


def compute_canonical_digest(payload: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hex digest across the 12 semantic decision fields.
    
    Guarantees:
    - Byte-level parity with TypeScript computeCanonicalDigest.
    - Strictly excludes transport identifiers (event_id, occurred_at_utc, evaluation_context_id, evaluation_id).
    - Immutable decision output identity.
    """
    sym = str(payload.get("symbol", "")).upper()
    sha = str(payload.get("release_sha", "")).lower()
    ruleset = str(payload.get("ruleset_version", ""))
    
    cat = payload.get("event_category", "")
    if hasattr(cat, "value"):
        cat = cat.value
    cat = str(cat)

    event_time = payload.get("event_time_utc")
    event_time_str = str(event_time) if event_time else "null"

    session = payload.get("market_session", "")
    if hasattr(session, "value"):
        session = session.value
    session = str(session)

    source_status = payload.get("event_source_status", "")
    if hasattr(source_status, "value"):
        source_status = source_status.value
    source_status = str(source_status)

    freshness = payload.get("freshness_status", "")
    if hasattr(freshness, "value"):
        freshness = freshness.value
    freshness = str(freshness)

    dte = payload.get("days_to_earnings")
    dte_str = f"{float(dte):.2f}" if dte is not None else "null"

    risk = payload.get("risk_bucket", "")
    if hasattr(risk, "value"):
        risk = risk.value
    risk = str(risk)

    penalty = payload.get("applied_penalty", 0.0)
    penalty_str = f"{float(penalty):.1f}"

    outcome = payload.get("evaluation_outcome", "")
    if hasattr(outcome, "value"):
        outcome = outcome.value
    outcome = str(outcome)

    parts = [
        sym,
        sha,
        ruleset,
        cat,
        event_time_str,
        session,
        source_status,
        freshness,
        dte_str,
        risk,
        penalty_str,
        outcome,
    ]
    canonical_string = "|".join(parts)
    return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()


def derive_authoritative_evaluation_id(
    release_sha: str,
    symbol: str,
    ruleset_version: str,
    event_time_utc: Optional[str],
    market_session: str,
    evaluation_timestamp: str,
) -> str:
    """Deterministically derives the authoritative evaluation instance identifier.
    
    Guarantees:
    - Identical inputs and evaluation execution timestamp produce the identical evaluation_id.
    - Two browser tabs viewing the same execution result produce the identical evaluation_id.
    - Two genuine executions (with distinct execution timestamps or inputs) produce distinct evaluation_ids.
    - Rehydration of an existing execution retains its frozen evaluation_id.
    - Byte-level parity with TypeScript deriveAuthoritativeEvaluationId.
    """
    event_time_str = event_time_utc if event_time_utc else "null"
    s = f"eval|{release_sha.lower()}|{symbol.upper()}|{ruleset_version}|{event_time_str}|{market_session}|{evaluation_timestamp}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

