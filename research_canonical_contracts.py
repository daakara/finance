"""
analyst_dashboard/research/sourced_fundamentals/contracts.py

Data contracts and schemas for Sourced Fundamentals & Undervalued Growth Research.
Enforces field-level provenance, period attribution, metric derivation tracking,
and research dispositions.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any


class MetricStatus(str, Enum):
    REPORTED = "REPORTED"
    DERIVED = "DERIVED"
    ANALYST_ADJUSTED = "ANALYST_ADJUSTED"
    UNAVAILABLE = "UNAVAILABLE"


class ResearchDisposition(str, Enum):
    RESEARCH_FURTHER = "RESEARCH_FURTHER"
    WATCH = "WATCH"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSUITABLE_FOR_THIS_OBJECTIVE = "UNSUITABLE_FOR_THIS_OBJECTIVE"


@dataclass
class FinancialFact:
    """
    Field-level authoritative observation with immutable provenance.
    Every single number in the system must link back to its source concept and accession.
    """
    metric_key: str
    value: Optional[float]
    units: str  # "USD", "shares", "percent", "ratio", "count"
    concept: Optional[str] = None  # e.g. "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax"
    accession_number: Optional[str] = None  # e.g. "0001035983-24-000045"
    form: Optional[str] = None  # "10-K", "10-Q"
    period_start: Optional[str] = None  # YYYY-MM-DD for durations
    period_end: Optional[str] = None  # YYYY-MM-DD for durations and instants
    filing_date: Optional[str] = None  # YYYY-MM-DD
    retrieval_time: Optional[str] = None  # ISO8601
    status: MetricStatus = MetricStatus.REPORTED
    formula: Optional[str] = None  # Explicit formula for derived metrics
    input_references: List[str] = field(default_factory=list)  # References to input metric keys
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class CompanyFundamentals:
    """Container for a company's raw reported and normalized facts."""
    symbol: str
    company_name: str
    cik: str
    primary_exchange: str
    sector_industry: str
    business_model: str
    fiscal_year_end: str
    latest_reported_period: str
    latest_filing_date: str
    facts: Dict[str, FinancialFact] = field(default_factory=dict)
    operating_indicators: Dict[str, Any] = field(default_factory=dict)
    data_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "cik": self.cik,
            "primary_exchange": self.primary_exchange,
            "sector_industry": self.sector_industry,
            "business_model": self.business_model,
            "fiscal_year_end": self.fiscal_year_end,
            "latest_reported_period": self.latest_reported_period,
            "latest_filing_date": self.latest_filing_date,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
            "operating_indicators": self.operating_indicators,
            "data_gaps": self.data_gaps,
        }


@dataclass
class DerivedMetrics:
    """Reproducible calculations derived from authoritative facts."""
    symbol: str
    period_label: str  # e.g. "TTM (Q3 2024 to Q2 2025)"
    
    # Growth & Margins
    revenue_ttm: Optional[float] = None
    revenue_prior_ttm: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None
    operating_income_ttm: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_income_ttm: Optional[float] = None
    net_margin_pct: Optional[float] = None

    # Cash Flow & Conversion
    cfo_ttm: Optional[float] = None
    capex_ttm: Optional[float] = None
    cfo_minus_capex_ttm: Optional[float] = None  # Non-GAAP cash flow measure
    fcf_conversion_ebit_pct: Optional[float] = None  # (CFO - Capex) / Operating Income
    fcf_conversion_net_income_pct: Optional[float] = None

    # Capital Intensity & Returns
    sbc_ttm: Optional[float] = None
    sbc_to_revenue_pct: Optional[float] = None
    invested_capital: Optional[float] = None
    effective_tax_rate_pct: Optional[float] = None
    nopat_ttm: Optional[float] = None
    roic_pct: Optional[float] = None

    # Balance Sheet & Liquidity
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    total_cash_and_investments: Optional[float] = None
    short_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_financial_debt: Optional[float] = None
    net_debt: Optional[float] = None  # Total Debt - Total Cash
    operating_lease_liabilities: Optional[float] = None

    # Share Structure & Per-Share
    diluted_shares_weighted_avg: Optional[float] = None
    common_shares_outstanding: Optional[float] = None
    diluted_eps_ttm: Optional[float] = None  # Conventional TTM diluted EPS (populated only when compatible annual shares match annual net income)
    ttm_net_income_per_latest_quarter_diluted_share: Optional[float] = None  # Research estimate for interim reporting periods
    ttm_eps_estimate_status: Optional[str] = None  # "RESEARCH_ESTIMATE" or "REPORTED_ANNUAL_COMPATIBLE"
    ttm_eps_numerator_period: Optional[str] = None  # e.g. "2025-07-01 to 2026-06-30"
    ttm_eps_denominator_period: Optional[str] = None  # e.g. "2026-04-01 to 2026-06-30 (quarterly diluted shares)"
    cfo_minus_capex_per_share: Optional[float] = None
    share_dilution_1y_pct: Optional[float] = None
    share_dilution_3y_pct: Optional[float] = None

    # Market Reference & Multiples
    market_price: Optional[float] = None
    market_price_date: Optional[str] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    ev_to_ebit: Optional[float] = None
    p_to_cfo_minus_capex: Optional[float] = None
    ev_to_sales: Optional[float] = None

    # Verification and provenance
    calculation_notes: Dict[str, str] = field(default_factory=dict)
    arithmetic_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchAssessmentRecord:
    """Structured qualitative and economic assessment without composite scoring."""
    symbol: str
    disposition: ResearchDisposition
    disposition_rationale: str
    growth_durability: str
    reinvestment_economics: str
    shareholder_outcomes: str
    balance_sheet_resilience: str
    valuation_uncertainty: str
    evidence_sufficiency: str
    thesis_invalidation_conditions: List[str]
    specific_next_evidence_to_monitor: List[str]
    remaining_evidence_gaps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["disposition"] = self.disposition.value
        return d


@dataclass
class ValuationScenario:
    """Explicit multi-scenario valuation with sensitivity analysis."""
    scenario_name: str  # "Conservative", "Base", "Optimistic"
    growth_years_1_to_5_pct: float
    operating_margin_target_pct: float
    wacc_pct: float
    terminal_growth_pct: float
    tax_rate_pct: float
    reinvestment_rate_pct: float
    pv_explicit_cash_flows: float
    pv_terminal_value: float
    enterprise_value_estimate: float
    net_debt_bridge: float  # (Total Debt + Leases - Cash)
    equity_value_estimate: float
    per_share_value_estimate: float
    upside_downside_pct: float
    key_assumptions: str


@dataclass
class DeepValuationAssessment:
    """Comprehensive multi-scenario valuation package for research leads."""
    symbol: str
    framework_used: str  # "FCFF (Enterprise to Equity)" or "FCFE (Direct Equity)"
    current_price: float
    current_price_date: str
    current_market_cap: float
    diluted_shares: float
    scenarios: Dict[str, ValuationScenario]
    reverse_dcf_implied_5y_growth_pct: Optional[float]
    reverse_dcf_fixed_assumptions: Dict[str, Any]
    sensitivity_matrix: Dict[str, Dict[str, Optional[float]]]  # WACC vs Terminal Growth -> Per Share Value
    thesis_invalidation_thresholds: List[str]
    next_quarter_evidence_milestones: List[str]
    valuation_status: str = "AVAILABLE"  # "AVAILABLE" or "SUPPRESSED_UNAVAILABLE"
    suppression_reason: Optional[str] = None
