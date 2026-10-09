"""ARX VCP Domain Authority & Conformance Package.

Sprint 2B Domain-Authority Resolution.
"""

from analyst_dashboard.vcp.case_compiler import (
    TemporalReadEvent,
    VCPTemporalCaseCompiler,
    VCPTemporalCasePackage,
)
from analyst_dashboard.vcp.classifier import VCPClassifier
from analyst_dashboard.vcp.conformance_oracle import (
    ADJUDICATORS,
    Adjudicator,
    IsolationLevel,
    OracleTier,
    VCPConformanceCorpus,
    VCPCorpusCase,
    generate_clean_history,
)
from analyst_dashboard.vcp.domain_glossary import (
    GLOSSARY_TERMS,
    DomainTermDefinition,
    TermStatus,
    VCPDomainGlossary,
)
from analyst_dashboard.vcp.domain_source_registry import (
    VCP_DOMAIN_SOURCES,
    AuthorityClass,
    ContentUsageRight,
    DomainClaimRecord,
    DomainSource,
    VCPDomainSourceRegistry,
)
from analyst_dashboard.vcp.label_authorization import (
    LABEL_AUTHORIZATION_CATALOG,
    LabelAuthorizationRecord,
    VCPLabelAuthorizationMatrix,
)
from analyst_dashboard.vcp.mutation_harness import (
    MutationTestResult,
    VCPMutationHarness,
)
from analyst_dashboard.vcp.numeric_contract import (
    NUMERIC_SPECS,
    NumericSpecification,
    VCPNumericContract,
)
from analyst_dashboard.vcp.predicate_registry import (
    PREDICATE_DEFINITIONS,
    ConformanceRole,
    PredicateDefinition,
    PredicateResult,
    PredicateStatus,
    VCPDomainAssessment,
    VCPObservation,
    VCPPredicateRegistry,
)
from analyst_dashboard.vcp.temporal_contract import (
    DailyOHLCVBar,
    VCPTemporalContract,
)

__all__ = [
    "AuthorityClass",
    "ContentUsageRight",
    "DomainClaimRecord",
    "DomainSource",
    "VCP_DOMAIN_SOURCES",
    "VCPDomainSourceRegistry",
    "TermStatus",
    "DomainTermDefinition",
    "GLOSSARY_TERMS",
    "VCPDomainGlossary",
    "PredicateStatus",
    "ConformanceRole",
    "PredicateResult",
    "VCPObservation",
    "VCPDomainAssessment",
    "PredicateDefinition",
    "PREDICATE_DEFINITIONS",
    "VCPPredicateRegistry",
    "NumericSpecification",
    "NUMERIC_SPECS",
    "VCPNumericContract",
    "DailyOHLCVBar",
    "VCPTemporalContract",
    "TemporalReadEvent",
    "VCPTemporalCasePackage",
    "VCPTemporalCaseCompiler",
    "OracleTier",
    "IsolationLevel",
    "Adjudicator",
    "ADJUDICATORS",
    "VCPCorpusCase",
    "generate_clean_history",
    "VCPConformanceCorpus",
    "VCPClassifier",
    "LabelAuthorizationRecord",
    "LABEL_AUTHORIZATION_CATALOG",
    "VCPLabelAuthorizationMatrix",
    "MutationTestResult",
    "VCPMutationHarness",
]
