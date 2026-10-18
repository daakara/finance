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
from .connected_integration import (
    ConnectedIntegrationResult,
    IntegrationConfig,
    format_markdown_report,
    run_connected_integration,
)

from .coverage_diagnostics import (
    DiagnosticInputRecord,
    ObservationDiagnosticResult,
    SymbolDiagnosticResult,
    CoverageAccountingSummary,
    PrimaryDiagnosticOutcome,
    BuildReadinessOutcome,
    SessionReadinessOutcome,
    ReadOnlyMarketStore,
    CoverageDiagnosticEngine,
    CapacityBenchmarker,
    CapacityBenchmarkReport,
    load_acquisition_artifacts_to_records,
)

from .identity_adoption import (
    IdentityAdoptionStatus,
    IdentityAdoptionAuthorization,
    ProposedEntityAdoption,
    ProposedAssociationAdoption,
    AdoptionPlanSummary,
    IdentityAdoptionPlan,
    IdentityAdoptionDryRunEngine,
    StalePlanError,
    PlanIntegrityError,
    PlanApplicationError,
    apply_adoption_plan_test_only,
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
    "ConnectedIntegrationResult",
    "IntegrationConfig",
    "format_markdown_report",
    "run_connected_integration",
    "DiagnosticInputRecord",
    "ObservationDiagnosticResult",
    "SymbolDiagnosticResult",
    "CoverageAccountingSummary",
    "PrimaryDiagnosticOutcome",
    "BuildReadinessOutcome",
    "SessionReadinessOutcome",
    "ReadOnlyMarketStore",
    "CoverageDiagnosticEngine",
    "CapacityBenchmarker",
    "CapacityBenchmarkReport",
    "load_acquisition_artifacts_to_records",
    "IdentityAdoptionStatus",
    "IdentityAdoptionAuthorization",
    "ProposedEntityAdoption",
    "ProposedAssociationAdoption",
    "AdoptionPlanSummary",
    "IdentityAdoptionPlan",
    "IdentityAdoptionDryRunEngine",
    "StalePlanError",
    "PlanIntegrityError",
    "PlanApplicationError",
    "apply_adoption_plan_test_only",
]


