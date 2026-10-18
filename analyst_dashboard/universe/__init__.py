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
from .offline_build_coverage import (
    OfflineCandidateRecord,
    OfflineBuildCoverageConfig,
    OfflineBuildCoverageResult,
    Stage1IdentityAssociationResult,
    Stage2ValidationAndConversionResult,
    Stage3BuilderEvaluationResult,
    Stage4CoverageDiagnosticsResult,
    run_offline_build_and_coverage,
    build_scenario_a_candidates,
    build_scenario_b_candidates,
    build_scenario_b2_conversion_failure_candidates,
    build_scenario_d_identity_conflict_candidates,
    build_scenario_c_candidates,
    format_offline_build_coverage_markdown,
    save_offline_build_coverage_artifacts,
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

from .identity_proposal import (
    build_seven_candidate_proposal,
    build_three_candidate_proposals,
)

from .identity_persistence import (
    APPROVED_ARCHITECTURAL_PRINCIPLE,
    IdentityApplicationStatus,
    IdentityApplicationOutcome,
    SqliteIdentityRepository,
    IdentityPlanApplicationService,
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
    "OfflineCandidateRecord",
    "OfflineBuildCoverageConfig",
    "OfflineBuildCoverageResult",
    "Stage1IdentityAssociationResult",
    "Stage2ValidationAndConversionResult",
    "Stage3BuilderEvaluationResult",
    "Stage4CoverageDiagnosticsResult",
    "run_offline_build_and_coverage",
    "build_scenario_a_candidates",
    "build_scenario_b_candidates",
    "build_scenario_b2_conversion_failure_candidates",
    "build_scenario_d_identity_conflict_candidates",
    "build_scenario_c_candidates",
    "format_offline_build_coverage_markdown",
    "save_offline_build_coverage_artifacts",
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
    "APPROVED_ARCHITECTURAL_PRINCIPLE",
    "IdentityApplicationStatus",
    "IdentityApplicationOutcome",
    "SqliteIdentityRepository",
    "IdentityPlanApplicationService",
]



