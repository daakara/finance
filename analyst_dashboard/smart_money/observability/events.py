"""
ARX Terminal — Smart Money Operational Observability & Event Taxonomy.

Defines:
1. 19-Event Taxonomy across worker, ingestion, publication, backup, and containment lifecycles;
2. Standard Structured JSON Event Envelope;
3. Fail-closed secret/credential sanitizer scrubbing tokens, DSNs, and auth headers.
"""

import os
import re
import uuid
import time
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class SmartMoneyEventType(Enum):
    WORKER_STARTUP = "WORKER_STARTUP"
    WORKER_SHUTDOWN = "WORKER_SHUTDOWN"
    UNAUTHORIZED_EXECUTION = "UNAUTHORIZED_EXECUTION"
    BACKUP_FAILURE = "BACKUP_FAILURE"
    BACKUP_FRESHNESS_VIOLATION = "BACKUP_FRESHNESS_VIOLATION"
    SEC_RATE_LIMIT_429 = "SEC_RATE_LIMIT_429"
    SEC_AUTH_FAILURE = "SEC_AUTH_FAILURE"
    INGESTION_FAILURE = "INGESTION_FAILURE"
    PARSER_FAILURE = "PARSER_FAILURE"
    DATABASE_CONNECTION_LOSS = "DATABASE_CONNECTION_LOSS"
    ADVISORY_LOCK_CONTENTION = "ADVISORY_LOCK_CONTENTION"
    LEASE_EXPIRY_FENCED = "LEASE_EXPIRY_FENCED"
    SNAPSHOT_PUBLICATION_FAILURE = "SNAPSHOT_PUBLICATION_FAILURE"
    POST_BACKUP_FAILURE = "POST_BACKUP_FAILURE"
    MISSED_SCHEDULED_CYCLE = "MISSED_SCHEDULED_CYCLE"
    REPEATED_ZERO_MATCH = "REPEATED_ZERO_MATCH"
    DATA_FRESHNESS_BREACH = "DATA_FRESHNESS_BREACH"
    RECOVERY_ACTION_TRIGGERED = "RECOVERY_ACTION_TRIGGERED"
    EMERGENCY_STOP_TRIGGERED = "EMERGENCY_STOP_TRIGGERED"


class EventSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PipelineStage(Enum):
    PREFLIGHT = "PREFLIGHT"
    INGESTION = "INGESTION"
    PARSING = "PARSING"
    RECONCILIATION = "RECONCILIATION"
    SNAPSHOT_PUBLICATION = "SNAPSHOT_PUBLICATION"
    BACKUP = "BACKUP"
    CONTAINMENT = "CONTAINMENT"
    RECOVERY = "RECOVERY"
    SHUTDOWN = "SHUTDOWN"


SECRET_PATTERNS = [
    re.compile(r"postgres(?:ql)?://([^:]+):([^@]+)@", re.IGNORECASE),
    re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"token=([A-Za-z0-9_\-\.]+)", re.IGNORECASE),
    re.compile(r"password=([^\s&]+)", re.IGNORECASE),
    re.compile(r"secret(?:_key)?=([^\s&]+)", re.IGNORECASE),
    re.compile(r"ARX_[A-Z0-9_]*TOKEN[=:\s]+([^\s,]+)", re.IGNORECASE),
]


def sanitize_secrets(value: Any) -> Any:
    """Recursively redacts database passwords, tokens, API keys, and auth headers."""
    if isinstance(value, str):
        sanitized = value
        for pattern in SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
    elif isinstance(value, dict):
        sanitized_dict = {}
        for k, v in value.items():
            lower_k = str(k).lower()
            if any(term in lower_k for term in ["password", "token", "secret", "authorization", "api_key", "credential"]):
                sanitized_dict[k] = "[REDACTED]"
            else:
                sanitized_dict[k] = sanitize_secrets(v)
        return sanitized_dict
    elif isinstance(value, list):
        return [sanitize_secrets(item) for item in value]
    return value


@dataclass
class SmartMoneyEvent:
    event_type: SmartMoneyEventType
    severity: EventSeverity
    message: str
    pipeline_stage: PipelineStage
    action_required: bool = False
    run_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    environment: str = field(default_factory=lambda: os.environ.get("ARX_ENVIRONMENT", "production"))
    release_sha: str = field(default_factory=lambda: os.environ.get("ARX_RELEASE_SHA", "86037f4ba54fb6b66f94056cb07299482960a636"))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes event to clean JSON dictionary with secrets redacted."""
        raw_payload = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "run_id": self.run_id,
            "pipeline_stage": self.pipeline_stage.value,
            "occurred_at_utc": self.occurred_at_utc,
            "environment": self.environment,
            "release_sha": self.release_sha,
            "correlation_id": self.correlation_id,
            "action_required": self.action_required,
            "metadata": self.metadata,
        }
        return sanitize_secrets(raw_payload)
