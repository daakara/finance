"""
analyst_dashboard/security_master/__init__.py

ARX Terminal Canonical Server Security Master.
Authoritative source of instrument identity, security subtype, and execution eligibility.
"""

from .models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
    ExecutionEligibility,
    AnalyticsCapability,
    CanonicalInstrument,
)
from .alpaca_adapter import AlpacaIdentityAdapter, AlpacaAssetEvidence
from .openfigi_adapter import OpenFIGISubtypeAdapter, OpenFIGISubtypeEvidence
from .normalization import SecurityMasterNormalizationEngine
from .eligibility import evaluate_execution_eligibility
from .persistence import SecurityMasterRepository
from .config import resolve_security_master_db_path, get_security_master_ttl
from .service import SecurityMasterService, get_security_master_service, set_security_master_service

from .source_governance_models import (
    canonical_hash,
    canonical_json_dumps,
    RawSourceRecord,
    RawSourceSnapshot,
    CanonicalIssuer,
    CanonicalSecurity,
    CanonicalListing,
    CanonicalFieldDecision,
    SourceConflictRecord,
    MembershipEvent,
    CanonicalGeneration,
    SnapshotStatus,
    ConflictSeverity,
    ConflictResolution,
    MembershipState,
    ListingState,
    MembershipTransitionType,
    SurvivorshipStatus,
    HistoricalMembershipAuthority,
    QuarantineScope,
    PromotionStatus,
    ReasonCode,
    PointInTimeStatus,
    HistoricalUniverseQueryResult,
    HistoricalMembershipUnavailableError,
)
from .source_governance_policy import (
    FieldAuthorityPolicyRegistry,
    SourceConflictClassifier,
    AuthorityGraphValidationError,
    normalize_symbol_string,
    normalize_exchange_mic,
    CANONICAL_MIC_TABLE,
    CANONICAL_SECURITY_TYPES,
)
from .source_resolver import (
    SourceReconciliationEngine,
    GenerationLifecycleManager,
    StaleCanonicalPromotionError,
    ReconciliationIntegrityError,
)

__all__ = [
    "AssetClass",
    "SecurityType",
    "ListingStatus",
    "ClassificationStatus",
    "ExecutionEligibility",
    "AnalyticsCapability",
    "CanonicalInstrument",
    "AlpacaIdentityAdapter",
    "AlpacaAssetEvidence",
    "OpenFIGISubtypeAdapter",
    "OpenFIGISubtypeEvidence",
    "SecurityMasterNormalizationEngine",
    "evaluate_execution_eligibility",
    "SecurityMasterRepository",
    "resolve_security_master_db_path",
    "get_security_master_ttl",
    "SecurityMasterService",
    "get_security_master_service",
    "set_security_master_service",
    "canonical_hash",
    "canonical_json_dumps",
    "RawSourceRecord",
    "RawSourceSnapshot",
    "CanonicalIssuer",
    "CanonicalSecurity",
    "CanonicalListing",
    "CanonicalFieldDecision",
    "SourceConflictRecord",
    "MembershipEvent",
    "CanonicalGeneration",
    "SnapshotStatus",
    "ConflictSeverity",
    "ConflictResolution",
    "MembershipState",
    "ListingState",
    "MembershipTransitionType",
    "SurvivorshipStatus",
    "HistoricalMembershipAuthority",
    "PointInTimeStatus",
    "HistoricalUniverseQueryResult",
    "HistoricalMembershipUnavailableError",
    "QuarantineScope",
    "PromotionStatus",
    "ReasonCode",
    "FieldAuthorityPolicyRegistry",
    "SourceConflictClassifier",
    "AuthorityGraphValidationError",
    "normalize_symbol_string",
    "normalize_exchange_mic",
    "CANONICAL_MIC_TABLE",
    "CANONICAL_SECURITY_TYPES",
    "SourceReconciliationEngine",
    "GenerationLifecycleManager",
    "StaleCanonicalPromotionError",
    "ReconciliationIntegrityError",
]

