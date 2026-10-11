"""Smart Money Form 4 Market-Wide Scanning Package."""

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
    SmartMoneyIssuerState,
    SmartMoneyScanSnapshot,
    SmartMoneyReadiness,
    SmartMoneySignalType,
    OwnershipNature,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
try:
    from analyst_dashboard.smart_money.form4_ingestion import Form4IngestionManager
except ImportError:
    Form4IngestionManager = None  # type: ignore[assignment, misc]
from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner

from analyst_dashboard.smart_money.universe_models import (
    CikMappingStatus,
    SmartMoneyEligibilityDecision,
    SmartMoneyCikMapping,
    SmartMoneySecurity,
    SmartMoneyUniverseBuildManifest,
)
from analyst_dashboard.smart_money.universe_builder import SmartMoneyUniverseBuilder
from analyst_dashboard.smart_money.backfill_models import (
    BackfillMode,
    BackfillStatus,
    FilingDisposition,
    AmendmentRelationship,
    SmartMoneyBackfillSpec,
    SmartMoneyBackfillManifest,
    SourceAvailabilityRecord,
)
from analyst_dashboard.smart_money.backfill_engine import SmartMoneyBackfillEngine
from analyst_dashboard.smart_money.incremental_models import (
    IncrementalRunStatus,
    GapType,
    GapStatus,
    LateArrivalEventType,
    FreshnessStatus,
    ReadinessStatus,
    IncrementalCycle,
    SourceGap,
    PartitionWatermark,
    WatermarkReceipt,
    LateArrivalEvent,
    FreshnessEvaluation,
)
from analyst_dashboard.smart_money.incremental_engine import SmartMoneyIncrementalEngine

__all__ = [
    "SmartMoneyTransaction",
    "SmartMoneyFiling",
    "SmartMoneyIssuerState",
    "SmartMoneyScanSnapshot",
    "SmartMoneyReadiness",
    "SmartMoneySignalType",
    "OwnershipNature",
    "Form4XmlParser",
    "Form4IngestionManager",
    "SmartMoneyForm4Scanner",
    "CikMappingStatus",
    "SmartMoneyEligibilityDecision",
    "SmartMoneyCikMapping",
    "SmartMoneySecurity",
    "SmartMoneyUniverseBuilder",
    "BackfillMode",
    "BackfillStatus",
    "FilingDisposition",
    "AmendmentRelationship",
    "SmartMoneyBackfillSpec",
    "SmartMoneyBackfillManifest",
    "SourceAvailabilityRecord",
    "SmartMoneyBackfillEngine",
    "IncrementalRunStatus",
    "GapType",
    "GapStatus",
    "LateArrivalEventType",
    "FreshnessStatus",
    "ReadinessStatus",
    "IncrementalCycle",
    "SourceGap",
    "PartitionWatermark",
    "WatermarkReceipt",
    "LateArrivalEvent",
    "FreshnessEvaluation",
    "SmartMoneyIncrementalEngine",
]

