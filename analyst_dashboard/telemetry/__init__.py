"""ARX Terminal - Corporate Events Telemetry Proof of Concept Module.

Provides versioned schema validation, strict privacy boundary enforcement,
10-class provenance classification, race-safe deduplication, and local collector persistence.
"""

from .corporate_event_contract import (
    CorporateEventTelemetryPayload,
    CorporateEventCategory,
    MarketSession,
    EventSourceStatus,
    FreshnessStatus,
    RiskBucket,
    EvaluationOutcome,
    compute_deduplication_key,
    PROHIBITED_PRIVACY_FIELDS,
)
from .provenance import (
    ProvenanceClass,
    ProvenanceClassifier,
    ClassificationResult,
    ObservationEpochBounds,
)
from .collector import (
    TelemetryCollector,
    IngestionResult,
    IngestionStatus,
    TelemetryAccounting,
)

__all__ = [
    "CorporateEventTelemetryPayload",
    "CorporateEventCategory",
    "MarketSession",
    "EventSourceStatus",
    "FreshnessStatus",
    "RiskBucket",
    "EvaluationOutcome",
    "compute_deduplication_key",
    "PROHIBITED_PRIVACY_FIELDS",
    "ProvenanceClass",
    "ProvenanceClassifier",
    "ClassificationResult",
    "ObservationEpochBounds",
    "TelemetryCollector",
    "IngestionResult",
    "IngestionStatus",
    "TelemetryAccounting",
]
