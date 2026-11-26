"""
analyst_dashboard/research/sourced_fundamentals

Sourced Fundamentals & Undervalued Growth Research package.
Enforces field-level SEC provenance, non-overlapping TTM construction,
transparent cash flow derivation, and multi-scenario DCF / Reverse-DCF modeling.
"""

from .contracts import (
    FinancialFact,
    CompanyFundamentals,
    DerivedMetrics,
    ResearchAssessmentRecord,
    ValuationScenario,
    DeepValuationAssessment,
    MetricStatus,
    ResearchDisposition,
    LeaseAccountingConvention,
    AccountingFramework,
    LeaseIntegrityError,
    LeaseDoubleCountingError,
    LeaseConventionMismatchError,
    LeaseFrameworkIncompatibilityError,
    LeaseAggregationConflictError,
    LeaseIncompatibleComponentsError,
    LeaseComponentFact,
    LeaseLiabilityMetadata,
)
from .sec_harvester import SecEdgarHarvester, fetch_market_price_yfinance
from .normalizer import SecFactNormalizer
from .calculations import compute_derived_metrics
from .qualitative_assessor import assess_company
from .valuation import (
    run_fcff_model,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
)
from .offline_reader import (
    OfflineFundamentalsReader,
    OfflineFundamentalsResult,
    load_offline_fundamentals,
    protect_against_runtime_masking,
    ManifestIntegrityError,
    IdentityIntegrityError,
    EvidenceProtectionError,
)

__all__ = [
    "FinancialFact",
    "CompanyFundamentals",
    "DerivedMetrics",
    "ResearchAssessmentRecord",
    "ValuationScenario",
    "DeepValuationAssessment",
    "MetricStatus",
    "ResearchDisposition",
    "LeaseAccountingConvention",
    "AccountingFramework",
    "LeaseIntegrityError",
    "LeaseDoubleCountingError",
    "LeaseConventionMismatchError",
    "LeaseFrameworkIncompatibilityError",
    "LeaseAggregationConflictError",
    "LeaseIncompatibleComponentsError",
    "LeaseComponentFact",
    "LeaseLiabilityMetadata",
    "SecEdgarHarvester",
    "fetch_market_price_yfinance",
    "SecFactNormalizer",
    "compute_derived_metrics",
    "assess_company",
    "run_fcff_model",
    "solve_reverse_dcf",
    "generate_sensitivity_matrix",
    "OfflineFundamentalsReader",
    "OfflineFundamentalsResult",
    "load_offline_fundamentals",
    "protect_against_runtime_masking",
    "ManifestIntegrityError",
    "IdentityIntegrityError",
    "EvidenceProtectionError",
]
