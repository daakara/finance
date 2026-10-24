"""
analyst_dashboard/research/sourced_fundamentals/contracts.py

Data contracts and schemas for Sourced Fundamentals & Undervalued Growth Research.
Enforces field-level provenance, period attribution, metric derivation tracking,
and research dispositions.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any


class MetricStatus(str, Enum):
    REPORTED = "REPORTED"
    DERIVED = "DERIVED"
    ANALYST_ADJUSTED = "ANALYST_ADJUSTED"
    UNAVAILABLE = "UNAVAILABLE"


class ShareBasis(str, Enum):
    """
    Exhaustive classification of share-count measurement and estimation basis.
    Distinguishes temporal nature and dilution mechanics across 6 orthogonal basis types.
    """
    POINT_IN_TIME_OUTSTANDING = "POINT_IN_TIME_OUTSTANDING"
    WEIGHTED_AVERAGE_BASIC = "WEIGHTED_AVERAGE_BASIC"
    WEIGHTED_AVERAGE_DILUTED = "WEIGHTED_AVERAGE_DILUTED"
    CURRENT_VALUATION_DILUTED = "CURRENT_VALUATION_DILUTED"
    FORWARD_PROJECTED_TERMINAL = "FORWARD_PROJECTED_TERMINAL"
    SECURITY_CLASS_ADR_ADJUSTED = "SECURITY_CLASS_ADR_ADJUSTED"


class DenominatorUseCase(str, Enum):
    """
    Seven distinct financial use cases for share-count denominators in ARX valuation.
    """
    MARKET_CAPITALIZATION = "MARKET_CAPITALIZATION"
    HISTORICAL_ANNUAL_EPS = "HISTORICAL_ANNUAL_EPS"
    HISTORICAL_INTERIM_EPS = "HISTORICAL_INTERIM_EPS"
    TRAILING_TWELVE_MONTHS_EPS = "TRAILING_TWELVE_MONTHS_EPS"
    PRESENT_VALUE_DCF = "PRESENT_VALUE_DCF"
    PROJECTED_TERMINAL_EXIT = "PROJECTED_TERMINAL_EXIT"
    ADR_CROSS_BORDER_CLAIM = "ADR_CROSS_BORDER_CLAIM"


class DenominatorErrorCode(str, Enum):
    """Typed failure states for share denominator and period compatibility."""
    INCOMPATIBLE_SHARE_BASIS = "INCOMPATIBLE_SHARE_BASIS"
    PERIOD_ALIGNMENT_FAILED = "PERIOD_ALIGNMENT_FAILED"
    UNVERIFIED_ADR_RATIO = "UNVERIFIED_ADR_RATIO"
    INCOMPATIBLE_SHARE_CLASS = "INCOMPATIBLE_SHARE_CLASS"
    SHARE_UNIT_MISMATCH = "SHARE_UNIT_MISMATCH"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    INVALID_SHARE_DENOMINATOR = "INVALID_SHARE_DENOMINATOR"


class ShareDenominatorError(ValueError):
    """Base error for share-denominator and period compatibility failures."""
    def __init__(self, code: DenominatorErrorCode, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
        self.context = context or {}


class IncompatibleShareBasisError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS, message, context)


class PeriodAlignmentError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, message, context)


class UnverifiedADRRatioError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.UNVERIFIED_ADR_RATIO, message, context)


class IncompatibleShareClassError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.INCOMPATIBLE_SHARE_CLASS, message, context)


class ShareUnitMismatchError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.SHARE_UNIT_MISMATCH, message, context)


class MissingProvenanceError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.MISSING_PROVENANCE, message, context)


class InvalidShareDenominatorError(ShareDenominatorError):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(DenominatorErrorCode.INVALID_SHARE_DENOMINATOR, message, context)


@dataclass
class ADRConversionMetadata:
    is_adr: bool = False
    adr_ratio: Optional[float] = None
    ratio_source: Optional[str] = None
    depositary_bank: Optional[str] = None
    is_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShareDenominatorMetadata:
    """
    Field-level provenance and semantics for share counts used as denominators.
    Enforces period matching, basis classification, share class integrity, and ADR ratio verification.
    """
    share_count: float
    basis: ShareBasis
    use_case: DenominatorUseCase
    units: str = "shares"
    share_class: str = "Common"
    is_multi_class: bool = False
    economic_claim_ratio: float = 1.0
    as_of_date: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    form: Optional[str] = None
    status: MetricStatus = MetricStatus.REPORTED
    estimate_status: Optional[str] = None
    adr_metadata: Optional[ADRConversionMetadata] = None
    provenance_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["basis"] = self.basis.value
        d["use_case"] = self.use_case.value
        d["status"] = self.status.value
        if self.adr_metadata:
            d["adr_metadata"] = self.adr_metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "ShareDenominatorMetadata":
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or ShareDenominatorMetadata, got {type(d)}")
        adr_data = d.get("adr_metadata")
        adr_meta = None
        if adr_data:
            if isinstance(adr_data, ADRConversionMetadata):
                adr_meta = adr_data
            elif isinstance(adr_data, dict):
                adr_meta = ADRConversionMetadata(**adr_data)

        raw_basis = d["basis"]
        basis = raw_basis if isinstance(raw_basis, ShareBasis) else ShareBasis(raw_basis)
        raw_use_case = d["use_case"]
        use_case = raw_use_case if isinstance(raw_use_case, DenominatorUseCase) else DenominatorUseCase(raw_use_case)
        raw_status = d.get("status", MetricStatus.REPORTED)
        status = raw_status if isinstance(raw_status, MetricStatus) else MetricStatus(raw_status)

        return cls(
            share_count=float(d["share_count"]),
            basis=basis,
            use_case=use_case,
            units=d.get("units", "shares"),
            share_class=d.get("share_class", "Common"),
            is_multi_class=bool(d.get("is_multi_class", False)),
            economic_claim_ratio=float(d.get("economic_claim_ratio", 1.0)),
            as_of_date=d.get("as_of_date"),
            period_start=d.get("period_start"),
            period_end=d.get("period_end"),
            filing_date=d.get("filing_date"),
            accession_number=d.get("accession_number"),
            form=d.get("form"),
            status=status,
            estimate_status=d.get("estimate_status"),
            adr_metadata=adr_meta,
            provenance_notes=d.get("provenance_notes"),
        )


@dataclass
class DenominatorValidationResult:
    is_valid: bool
    error_code: Optional[DenominatorErrorCode] = None
    error_message: Optional[str] = None
    share_count: Optional[float] = None
    basis: Optional[ShareBasis] = None
    use_case: Optional[DenominatorUseCase] = None
    effective_shares: Optional[float] = None
    estimate_status: Optional[str] = None
    notes: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_valid


def validate_share_denominator_compatibility(
    metadata: ShareDenominatorMetadata,
    target_use_case: Optional[DenominatorUseCase] = None,
    numerator_fact: Optional["FinancialFact"] = None,
    valuation_date: Optional[str] = None,
    raise_on_error: bool = True,
) -> DenominatorValidationResult:
    """
    Minimal reusable compatibility validator enforcing P0-A share denominator rules.
    Fails closed on incompatible basis, period misalignment, unverified ADR ratio,
    share class mismatch, missing provenance, unit mismatch, or invalid denominators.
    """
    ctx = {"share_count": metadata.share_count, "basis": metadata.basis, "use_case": metadata.use_case}

    # 1. Zero or negative denominator
    if metadata.share_count is None or metadata.share_count <= 0:
        msg = f"Invalid share denominator value: {metadata.share_count}. Must be strictly positive."
        if raise_on_error:
            raise InvalidShareDenominatorError(msg, ctx)
        return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INVALID_SHARE_DENOMINATOR, error_message=msg)

    # 2. Units mismatch
    if not metadata.units or metadata.units.strip().lower() != "shares":
        msg = f"Share denominator units must be 'shares', got '{metadata.units}'."
        if raise_on_error:
            raise ShareUnitMismatchError(msg, ctx)
        return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.SHARE_UNIT_MISMATCH, error_message=msg)

    # 3. Missing provenance
    has_date = bool(metadata.as_of_date or metadata.period_end or metadata.filing_date)
    has_source = bool(metadata.accession_number or metadata.form or metadata.provenance_notes)
    if not (has_date and has_source):
        msg = "Share denominator lacks required provenance (measurement date and source accession/form/notes required)."
        if raise_on_error:
            raise MissingProvenanceError(msg, ctx)
        return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.MISSING_PROVENANCE, error_message=msg)

    # 4. ADR and cross-border securities
    if metadata.basis == ShareBasis.SECURITY_CLASS_ADR_ADJUSTED or (metadata.adr_metadata and metadata.adr_metadata.is_adr):
        if not metadata.adr_metadata:
            msg = "ADR security requires ADRConversionMetadata with verified conversion ratio."
            if raise_on_error:
                raise UnverifiedADRRatioError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.UNVERIFIED_ADR_RATIO, error_message=msg)

        ratio = metadata.adr_metadata.adr_ratio
        if ratio is None or ratio <= 0:
            msg = f"Invalid ADR-to-ordinary conversion ratio: {ratio}. Ratio must be strictly positive."
            if raise_on_error:
                raise UnverifiedADRRatioError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.UNVERIFIED_ADR_RATIO, error_message=msg)

        if not metadata.adr_metadata.is_verified:
            msg = "ADR conversion ratio is not verified against primary depositary or SEC registration filing."
            if raise_on_error:
                raise UnverifiedADRRatioError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.UNVERIFIED_ADR_RATIO, error_message=msg)

        if not metadata.adr_metadata.ratio_source or not metadata.adr_metadata.ratio_source.strip():
            msg = "ADR conversion ratio lacks required ratio_source provenance."
            if raise_on_error:
                raise UnverifiedADRRatioError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.UNVERIFIED_ADR_RATIO, error_message=msg)

    # 5. Share class compatibility
    if metadata.is_multi_class:
        if metadata.economic_claim_ratio is None or metadata.economic_claim_ratio <= 0:
            msg = f"Multi-class share structure requires strictly positive economic_claim_ratio (got {metadata.economic_claim_ratio})."
            if raise_on_error:
                raise IncompatibleShareClassError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_CLASS, error_message=msg)

        incompatible_classes = ["non-voting preferred", "tracking stock", "class b non-economic", "debt-equivalent preferred"]
        if metadata.share_class.strip().lower() in incompatible_classes:
            msg = f"Incompatible share class '{metadata.share_class}' cannot serve as common equity denominator."
            if raise_on_error:
                raise IncompatibleShareClassError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_CLASS, error_message=msg)

    # 6. Target use case vs share basis compatibility
    eff_use_case = target_use_case or metadata.use_case

    # Rule 6a: Present-Value DCF cannot use forward projected terminal shares
    if eff_use_case == DenominatorUseCase.PRESENT_VALUE_DCF:
        if metadata.basis == ShareBasis.FORWARD_PROJECTED_TERMINAL:
            msg = (
                f"Incompatible share basis '{metadata.basis.value}' for PRESENT_VALUE_DCF. "
                "Present-value DCF divides by current valuation-date diluted shares, never projected terminal shares."
            )
            if raise_on_error:
                raise IncompatibleShareBasisError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS, error_message=msg)

    # Rule 6b: Market capitalization cannot use projected terminal shares
    if eff_use_case == DenominatorUseCase.MARKET_CAPITALIZATION:
        if metadata.basis == ShareBasis.FORWARD_PROJECTED_TERMINAL:
            msg = f"Incompatible share basis '{metadata.basis.value}' for MARKET_CAPITALIZATION."
            if raise_on_error:
                raise IncompatibleShareBasisError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS, error_message=msg)

    # Rule 6c: Historical Annual EPS requires annual weighted-average duration (~365 days)
    if metadata.use_case == DenominatorUseCase.HISTORICAL_ANNUAL_EPS or eff_use_case == DenominatorUseCase.HISTORICAL_ANNUAL_EPS:
        if metadata.basis not in [ShareBasis.WEIGHTED_AVERAGE_DILUTED, ShareBasis.WEIGHTED_AVERAGE_BASIC, ShareBasis.SECURITY_CLASS_ADR_ADJUSTED]:
            msg = f"Historical Annual EPS requires weighted-average shares, got '{metadata.basis.value}'."
            if raise_on_error:
                raise IncompatibleShareBasisError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS, error_message=msg)

        if metadata.period_start and metadata.period_end:
            d_start = datetime.strptime(metadata.period_start, "%Y-%m-%d").date()
            d_end = datetime.strptime(metadata.period_end, "%Y-%m-%d").date()
            duration = (d_end - d_start).days
            if duration < 340 or duration > 385:
                msg = f"Annual EPS denominator duration must be ~365 days (got {duration} days, {metadata.period_start} to {metadata.period_end})."
                if raise_on_error:
                    raise PeriodAlignmentError(msg, ctx)
                return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, error_message=msg)

    # Rule 6d: Historical Interim EPS requires quarterly duration (~90 days)
    if metadata.use_case == DenominatorUseCase.HISTORICAL_INTERIM_EPS or eff_use_case == DenominatorUseCase.HISTORICAL_INTERIM_EPS:
        if metadata.basis not in [ShareBasis.WEIGHTED_AVERAGE_DILUTED, ShareBasis.WEIGHTED_AVERAGE_BASIC, ShareBasis.SECURITY_CLASS_ADR_ADJUSTED]:
            msg = f"Historical Interim EPS requires weighted-average shares, got '{metadata.basis.value}'."
            if raise_on_error:
                raise IncompatibleShareBasisError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS, error_message=msg)

        if metadata.period_start and metadata.period_end:
            d_start = datetime.strptime(metadata.period_start, "%Y-%m-%d").date()
            d_end = datetime.strptime(metadata.period_end, "%Y-%m-%d").date()
            duration = (d_end - d_start).days
            if duration < 70 or duration > 120:
                msg = f"Interim EPS denominator duration must be ~90 days (got {duration} days, {metadata.period_start} to {metadata.period_end})."
                if raise_on_error:
                    raise PeriodAlignmentError(msg, ctx)
                return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, error_message=msg)

    # Rule 6e: TTM EPS interim-to-TTM substitution guard
    if eff_use_case == DenominatorUseCase.TRAILING_TWELVE_MONTHS_EPS:
        if numerator_fact and numerator_fact.period_start and numerator_fact.period_end:
            n_start = datetime.strptime(numerator_fact.period_start, "%Y-%m-%d").date()
            n_end = datetime.strptime(numerator_fact.period_end, "%Y-%m-%d").date()
            num_dur = (n_end - n_start).days
            if num_dur >= 340:  # 12-month net income numerator
                if metadata.period_start and metadata.period_end:
                    d_start = datetime.strptime(metadata.period_start, "%Y-%m-%d").date()
                    d_end = datetime.strptime(metadata.period_end, "%Y-%m-%d").date()
                    den_dur = (d_end - d_start).days
                    if den_dur < 300:  # quarterly or interim denominator
                        if metadata.estimate_status != "RESEARCH_ESTIMATE":
                            msg = (
                                f"Cannot silently divide 12-month net income ({num_dur} days) by interim quarterly shares ({den_dur} days). "
                                "Interim TTM EPS substitution requires explicit estimate_status='RESEARCH_ESTIMATE'."
                            )
                            if raise_on_error:
                                raise PeriodAlignmentError(msg, ctx)
                            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, error_message=msg)

    # 7. Numerator-Denominator Period Alignment Check
    if numerator_fact and numerator_fact.period_end and metadata.period_end:
        num_end = datetime.strptime(numerator_fact.period_end, "%Y-%m-%d").date()
        den_end = datetime.strptime(metadata.period_end, "%Y-%m-%d").date()
        diff_days = abs((num_end - den_end).days)
        if diff_days > 7:
            msg = f"Numerator period end ({numerator_fact.period_end}) does not align with denominator period end ({metadata.period_end}), delta={diff_days} days."
            if raise_on_error:
                raise PeriodAlignmentError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, error_message=msg)

    # 8. Valuation Date Alignment Check (for spot / PV DCF)
    if valuation_date and metadata.as_of_date:
        v_date = datetime.strptime(valuation_date, "%Y-%m-%d").date()
        s_date = datetime.strptime(metadata.as_of_date, "%Y-%m-%d").date()
        if s_date > v_date:
            msg = f"Share denominator date ({metadata.as_of_date}) is after valuation date ({valuation_date}) — lookahead bias detected."
            if raise_on_error:
                raise PeriodAlignmentError(msg, ctx)
            return DenominatorValidationResult(is_valid=False, error_code=DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED, error_message=msg)

    return DenominatorValidationResult(
        is_valid=True,
        share_count=metadata.share_count,
        basis=metadata.basis,
        use_case=eff_use_case,
        effective_shares=metadata.share_count * (metadata.economic_claim_ratio or 1.0),
        estimate_status=metadata.estimate_status,
        notes=metadata.provenance_notes,
    )


# ==============================================================================
# P0-B: Lease Accounting Convention & Provenance Contracts
# ==============================================================================

class LeaseAccountingConvention(str, Enum):
    """
    Governing valuation convention for lease obligations across DCF and multiple models.
    """
    # Convention 1: Operating Expense Model (ARX Canonical Default under US GAAP)
    # Operating leases are expensed in operating income (EBIT) and operating cash flow (CFO).
    # Operating lease liabilities are strictly EXCLUDED from Net Debt Bridge.
    LEASE_AS_OPERATING_EXPENSE = "lease_as_operating_expense"

    # Convention 2: Capitalized Financing Model (Lease-as-Debt)
    # Operating leases are added back to operating income (EBITDAR / Debt-free cash flow).
    # Operating lease liabilities are strictly INCLUDED in Net Debt Bridge.
    LEASE_AS_FINANCING_DEBT = "lease_as_financing_debt"


class AccountingFramework(str, Enum):
    """
    Governing regulatory accounting standard of the reporting entity.
    """
    US_GAAP = "us_gaap"  # FASB ASC 842: Dual model (Operating vs. Finance leases)
    IFRS = "ifrs"        # IASB IFRS 16: Single lessee model (All leases capitalized)


class LeaseIntegrityError(ValueError):
    """Base exception for lease convention and accounting integrity violations."""
    pass


class LeaseDoubleCountingError(LeaseIntegrityError):
    """Raised when operating leases are deducted in both cash flows and net debt bridge."""
    pass


class LeaseConventionMismatchError(LeaseIntegrityError):
    """Raised when valuation components mix incompatible lease conventions."""
    pass


class LeaseFrameworkIncompatibilityError(LeaseIntegrityError):
    """Raised when an unadjusted IFRS 16 issuer is fed into a US GAAP convention."""
    pass


class LeaseAggregationConflictError(LeaseIntegrityError):
    """Raised when reported aggregate lease liabilities conflict with the sum of components."""
    pass


class LeaseIncompatibleComponentsError(LeaseIntegrityError):
    """Raised when current and noncurrent components share different dates, units, or currencies."""
    pass


@dataclass
class LeaseComponentFact:
    """
    Represents an individual, evidenced balance sheet lease fact with full SEC provenance.
    """
    value: Optional[float] = None
    concept: Optional[str] = None
    units: str = "USD"
    period_end: Optional[str] = None
    accession_number: Optional[str] = None
    filing_date: Optional[str] = None
    form: Optional[str] = None
    status: str = "UNAVAILABLE"  # REPORTED or UNAVAILABLE
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "concept": self.concept,
            "units": self.units,
            "period_end": self.period_end,
            "accession_number": self.accession_number,
            "filing_date": self.filing_date,
            "form": self.form,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass
class LeaseLiabilityMetadata:
    """
    Immutable provenance, aggregation, and accounting metadata governing lease treatment in valuation.
    """
    convention: LeaseAccountingConvention = LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE
    framework: AccountingFramework = AccountingFramework.US_GAAP
    certification_status: str = "CERTIFIED_GAAP_COMPATIBLE"  # CERTIFIED_GAAP_COMPATIBLE, RESEARCH_ESTIMATE, UNAVAILABLE

    # Granular balance sheet component facts
    current_liability: LeaseComponentFact = field(default_factory=LeaseComponentFact)
    noncurrent_liability: LeaseComponentFact = field(default_factory=LeaseComponentFact)
    aggregate_total_liability: LeaseComponentFact = field(default_factory=LeaseComponentFact)

    # Resolved total operating lease liabilities
    total_operating_lease_liabilities: Optional[float] = None
    aggregation_method: str = "NONE"  # REPORTED_AGGREGATE, SUMMED_COMPONENTS, RECONCILED_MATCH, PARTIAL_DISCLOSURE_UNRESOLVED, UNAVAILABLE

    # Senior debt interaction components
    finance_lease_liabilities: Optional[float] = None
    finance_leases_bundled_in_total_debt: bool = False

    # Asset recognition
    operating_lease_rou_asset: Optional[float] = None

    # Operating income normalization flags
    is_ebit_adjusted_for_leases: bool = False
    ebit_lease_cost_addback: Optional[float] = None

    # Bridge enforcement flag
    included_in_net_debt_bridge: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "convention": self.convention.value if isinstance(self.convention, LeaseAccountingConvention) else str(self.convention),
            "framework": self.framework.value if isinstance(self.framework, AccountingFramework) else str(self.framework),
            "certification_status": self.certification_status,
            "current_liability": self.current_liability.to_dict() if hasattr(self.current_liability, "to_dict") else self.current_liability,
            "noncurrent_liability": self.noncurrent_liability.to_dict() if hasattr(self.noncurrent_liability, "to_dict") else self.noncurrent_liability,
            "aggregate_total_liability": self.aggregate_total_liability.to_dict() if hasattr(self.aggregate_total_liability, "to_dict") else self.aggregate_total_liability,
            "total_operating_lease_liabilities": self.total_operating_lease_liabilities,
            "aggregation_method": self.aggregation_method,
            "finance_lease_liabilities": self.finance_lease_liabilities,
            "finance_leases_bundled_in_total_debt": self.finance_leases_bundled_in_total_debt,
            "operating_lease_rou_asset": self.operating_lease_rou_asset,
            "is_ebit_adjusted_for_leases": self.is_ebit_adjusted_for_leases,
            "ebit_lease_cost_addback": self.ebit_lease_cost_addback,
            "included_in_net_debt_bridge": self.included_in_net_debt_bridge,
            "notes": self.notes,
        }


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
    share_basis: Optional[str] = None  # e.g. "WEIGHTED_AVERAGE_DILUTED", "POINT_IN_TIME_OUTSTANDING"
    share_denominator_metadata: Optional[Dict[str, Any]] = None
    lease_metadata: Optional[LeaseLiabilityMetadata] = None

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
    net_debt_bridge: float  # Net financial debt bridge: (Total Financial Debt - Total Cash & Investments under Lease-as-Operating-Expense convention)
    equity_value_estimate: float
    per_share_value_estimate: float
    upside_downside_pct: float
    key_assumptions: str
    lease_convention: LeaseAccountingConvention = LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE
    lease_metadata: Optional[LeaseLiabilityMetadata] = None


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
    is_certified: bool = False
    share_basis: Optional[str] = None
    share_denominator_metadata: Optional[Dict[str, Any]] = None
    lease_convention: Optional[str] = None
    lease_metadata: Optional[Dict[str, Any]] = None
