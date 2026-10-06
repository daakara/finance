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
]
