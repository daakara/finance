"""
analyst_dashboard/universe module

Exports canonical universe contracts, builders, and stores.
"""

from .contracts import (
    UNIVERSE_ID,
    UNIVERSE_VERSION,
    ELIGIBILITY_RULE_VERSION,
    NORMALIZATION_VERSION,
    RADAR_SCOPE_LABEL,
    EligibilityDecision,
    DataReadinessDecision,
    ConstructionStatus,
    PublicationDecision,
    MembershipTransition,
    SourceSecurity,
    SourcePopulationSnapshot,
    EligibilityLedgerRow,
    DataReadinessLedgerRow,
    UniverseBuildAttestation,
)
from .builder import DeterministicUniverseBuilder
from .store import UniverseStore
from .fixtures import (
    HISTORICAL_VCP_35_REGRESSION_FIXTURE,
    create_canonical_fixture_snapshot,
)

__all__ = [
    "UNIVERSE_ID",
    "UNIVERSE_VERSION",
    "ELIGIBILITY_RULE_VERSION",
    "NORMALIZATION_VERSION",
    "RADAR_SCOPE_LABEL",
    "EligibilityDecision",
    "DataReadinessDecision",
    "ConstructionStatus",
    "PublicationDecision",
    "MembershipTransition",
    "SourceSecurity",
    "SourcePopulationSnapshot",
    "EligibilityLedgerRow",
    "DataReadinessLedgerRow",
    "UniverseBuildAttestation",
    "DeterministicUniverseBuilder",
    "UniverseStore",
    "HISTORICAL_VCP_35_REGRESSION_FIXTURE",
    "create_canonical_fixture_snapshot",
]
