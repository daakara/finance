"""
scripts/research/etf_v2/models.py

Typed domain models for ARX Terminal ETF Research Pipeline V2.
Enforces strict contracts between architectural layers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class EntityIdentity:
    """Authoritative entity identity where CIK/Series/Class dominate heuristics."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    legal_name: str
    historical_aliases: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        return bool(self.symbol and self.cik and self.series_id and self.class_id)


@dataclass(frozen=True)
class FilingMetadata:
    """Pre-boundary SEC regulatory filing metadata."""
    accession: str
    form: str
    filing_date: str
    acceptance_timestamp: str
    document_filename: str
    file_path: Optional[Path] = None
    raw_sha256: str = ""
    report_period: Optional[str] = None
    effective_date: Optional[str] = None


@dataclass(frozen=True)
class ProspectusAuthority:
    """Selected authoritative statutory prospectus for a target."""
    filing: FilingMetadata
    document_role: str  # SUMMARY_PROSPECTUS, BASE_STATUTORY_PROSPECTUS, PROSPECTUS_SUPPLEMENT
    selection_rank: int
    qualification_evidence: str
    raw_source_sha256: str


@dataclass(frozen=True)
class StructuralSection:
    """DOM/structural heading or item boundary in statutory filing."""
    heading: str
    heading_role: str  # PRINCIPAL_STRATEGY, INVESTMENT_OBJECTIVE, GENERAL_DISCLOSURE, OTHER
    start_offset: int
    end_offset: int


@dataclass
class DocumentStructure:
    """Structured representation of parsed statutory document."""
    document_filename: str
    raw_text: str
    normalized_text: str
    normalized_text_sha256: str
    sections: List[StructuralSection] = field(default_factory=list)
    series_occurrences: Dict[str, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesBoundary:
    """Closed series container span preventing cross-series text leakage."""
    series_id: str
    start_offset: int
    end_offset: int
    boundary_type: str
    previous_sibling: Optional[str] = None
    next_sibling: Optional[str] = None


@dataclass(frozen=True)
class MandateSection:
    """Extracted statutory mandate section."""
    text: str
    heading_role: str  # PRINCIPAL_STRATEGY, INVESTMENT_OBJECTIVE, NONE
    start_offset: int
    end_offset: int
    completeness_state: str  # COMPLETE, PARTIAL, FALLBACK_OBJECTIVE, NONE
    source_role: str
    raw_source_sha256: str
    normalized_text_sha256: str
    mandate_sha256: str


@dataclass(frozen=True)
class NPORTMetrics:
    """Derived quantitative portfolio holding metrics from Form N-PORT."""
    accession: str
    series_id: str
    report_period: str
    filing_date: str
    acceptance_timestamp: str
    raw_sha256: str
    total_equity_pct: float
    total_govt_pct: float
    corporate_debt_pct: float
    mortgage_backed_pct: float
    distinct_holdings_count: int
    max_security_concentration: float
    derivation_version: str = "NPORT_DERIVATION_V2_0_0"


@dataclass(frozen=True)
class NCENIndexStatus:
    """Regulatory index fund attestation from Form N-CEN Item C.3.b."""
    accession: str
    series_id: str
    is_index_fund: bool
    item_c3b_value: str
    raw_sha256: str
    report_period: Optional[str] = None
    filing_date: Optional[str] = None
    acceptance_timestamp: Optional[str] = None


@dataclass(frozen=True)
class PolicyEvidence:
    """Aggregated multi-source regulatory evidence passed to Policy Classifier."""
    identity: EntityIdentity
    mandate: MandateSection
    nport: Optional[NPORTMetrics] = None
    ncen: Optional[NCENIndexStatus] = None


@dataclass(frozen=True)
class ClassificationDecision:
    """Immutable classification decision under Policy V1.1."""
    policy_version: str
    rule_id: str
    final_classification: str
    decision_trace: List[str]
    rationale: str


@dataclass(frozen=True)
class PopulationRecord:
    """Complete, self-contained certified golden corpus record."""
    symbol: str
    cik: str
    series_id: str
    class_id: str
    legal_name: str
    prospectus_accession: str
    prospectus_form: str
    prospectus_filing_date: str
    prospectus_sec_acceptance_timestamp: str
    prospectus_document_filename: str
    prospectus_document_role: str
    prospectus_raw_sha256: str
    normalization_version: str
    prospectus_normalized_text_sha256: str
    series_section_start_offset: int
    series_section_end_offset: int
    mandate_section_heading_role: str
    mandate_section_start_offset: int
    mandate_section_end_offset: int
    mandate_text_sha256: str
    nport_accession: str
    nport_report_date: str
    nport_filing_date: str
    nport_sec_acceptance_timestamp: str
    total_equity_pct: float
    total_govt_pct: float
    corporate_debt_pct: float
    mortgage_backed_pct: float
    distinct_holdings_count: int
    max_security_concentration: float
    ncen_accession: str
    ncen_filing_date: str
    ncen_sec_acceptance_timestamp: str
    is_index_fund: bool
    prospectus_raw_sha: str
    mandate_section_offsets: List[int]
    mandate_sha: str
    nport_report_period: str
    nport_raw_sha: str
    derived_policy_metrics: Dict[str, Any]
    ncen_report_period: Optional[str]
    ncen_raw_sha: str
    index_flag: bool
    complete_decision_trace: List[str]
    policy_rule_id: str
    policy_version: str
    classification_rule_id: str
    decision_trace: List[str]
    final_classification: str
    adjudication_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
