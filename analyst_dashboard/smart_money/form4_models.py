"""
Data models and canonical schemas for SEC Form 4 Smart Money Processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib
import json


class SmartMoneySignalType(str, Enum):
    OPEN_MARKET_PURCHASE = "OPEN_MARKET_PURCHASE"
    OPEN_MARKET_SALE = "OPEN_MARKET_SALE"
    OPTION_EXERCISE = "OPTION_EXERCISE"
    EQUITY_GRANT = "EQUITY_GRANT"
    GIFT = "GIFT"
    TAX_WITHHOLDING = "TAX_WITHHOLDING"
    ISSUER_DISPOSITION = "ISSUER_DISPOSITION"
    OTHER = "OTHER"


class OwnershipNature(str, Enum):
    DIRECT = "D"
    INDIRECT = "I"


@dataclass
class SmartMoneyTransaction:
    """Normalized, deduplicated individual transaction record from SEC Form 4."""
    record_key: str  # Composite dedup key: {accession_number}:{reporting_owner_cik}:{table_type}:{transaction_sequence}
    accession_number: str
    issuer_cik: str
    issuer_symbol: Optional[str]
    issuer_name: Optional[str]
    reporting_owner_cik: str
    reporting_owner_name: str
    is_officer: bool
    officer_title: Optional[str]
    is_director: bool
    is_ten_percent_owner: bool
    filing_date: str  # YYYY-MM-DD
    transaction_date: str  # YYYY-MM-DD
    transaction_code: str  # SEC code e.g. 'P', 'S', 'M', 'A'
    signal_type: SmartMoneySignalType
    shares: float
    price_per_share: float
    total_value_usd: float
    acquired_disposed: str  # 'A' or 'D'
    ownership_nature: OwnershipNature
    is_derivative: bool
    is_open_market_purchase_evidence: bool = False
    is_positive_buying_conviction: bool = False
    acceptance_datetime: Optional[str] = None
    public_availability_at: Optional[str] = None
    source_fetch_time: Optional[str] = None
    normalized_at: Optional[str] = None
    footnotes: List[str] = field(default_factory=list)
    source_url: str = ""
    content_hash: str = ""

    def __post_init__(self):
        # Bi-directional synchronization for backwards compatibility
        if self.is_positive_buying_conviction and not self.is_open_market_purchase_evidence:
            self.is_open_market_purchase_evidence = True
        elif self.is_open_market_purchase_evidence and not self.is_positive_buying_conviction:
            self.is_positive_buying_conviction = True
        # Set public_availability_at only if acceptance_datetime is present
        if not self.public_availability_at:
            if self.acceptance_datetime:
                self.public_availability_at = self.acceptance_datetime
            else:
                self.public_availability_at = None

    @property
    def sec_acceptance_at(self) -> Optional[str]:
        """Explicit SEC/EDGAR acceptance timestamp."""
        return self.acceptance_datetime or self.public_availability_at

    @property
    def conservative_knowledge_time(self) -> Optional[str]:
        """Conservative knowledge timestamp; returns None/UNKNOWN if not evidenced by prospective observation."""
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_key": self.record_key,
            "accession_number": self.accession_number,
            "issuer_cik": self.issuer_cik,
            "issuer_symbol": self.issuer_symbol,
            "issuer_name": self.issuer_name,
            "reporting_owner_cik": self.reporting_owner_cik,
            "reporting_owner_name": self.reporting_owner_name,
            "is_officer": self.is_officer,
            "officer_title": self.officer_title,
            "is_director": self.is_director,
            "is_ten_percent_owner": self.is_ten_percent_owner,
            "filing_date": self.filing_date,
            "transaction_date": self.transaction_date,
            "transaction_code": self.transaction_code,
            "signal_type": self.signal_type.value if isinstance(self.signal_type, SmartMoneySignalType) else self.signal_type,
            "shares": self.shares,
            "price_per_share": self.price_per_share,
            "total_value_usd": self.total_value_usd,
            "acquired_disposed": self.acquired_disposed,
            "ownership_nature": self.ownership_nature.value if isinstance(self.ownership_nature, OwnershipNature) else self.ownership_nature,
            "is_derivative": self.is_derivative,
            "is_open_market_purchase_evidence": self.is_open_market_purchase_evidence,
            "is_positive_buying_conviction": self.is_positive_buying_conviction,
            "acceptance_datetime": self.acceptance_datetime,
            "public_availability_at": self.public_availability_at,
            "source_fetch_time": self.source_fetch_time,
            "normalized_at": self.normalized_at,
            "footnotes": self.footnotes,
            "source_url": self.source_url,
            "content_hash": self.content_hash,
        }


@dataclass
class SmartMoneyFiling:
    """Document-level representation of an SEC Form 4 submission."""
    accession_number: str
    issuer_cik: str
    issuer_symbol: Optional[str]
    filing_date: str
    acceptance_datetime: Optional[str]
    document_type: str  # "4" or "4/A"
    is_amendment: bool
    transactions: List[SmartMoneyTransaction] = field(default_factory=list)
    raw_xml: Optional[str] = None
    source_content_hash: str = ""
    source_url: str = ""
    fetched_at: str = ""
    disposition: str = "FETCHED_AND_PARSED"
    amendment_of_accession: Optional[str] = None


@dataclass
class SmartMoneyIssuerState:
    """Aggregated insider activity state for a single issuer over an evaluation window."""
    symbol: str
    issuer_cik: str
    issuer_name: str
    period_start: str
    period_end: str
    open_market_purchase_count: int
    open_market_sale_count: int
    distinct_buyer_count: int
    gross_purchase_value_usd: float
    net_purchase_value_usd: float
    cluster_buy: bool
    c_suite_buying: bool
    top_insider_titles: List[str]
    conviction_score: float  # 0 to 100 transparent arithmetic score
    conviction_reasons: List[str]
    qualifies_for_radar: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "issuer_cik": self.issuer_cik,
            "issuer_name": self.issuer_name,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "open_market_purchase_count": self.open_market_purchase_count,
            "open_market_sale_count": self.open_market_sale_count,
            "distinct_buyer_count": self.distinct_buyer_count,
            "gross_purchase_value_usd": round(self.gross_purchase_value_usd, 2),
            "net_purchase_value_usd": round(self.net_purchase_value_usd, 2),
            "cluster_buy": self.cluster_buy,
            "c_suite_buying": self.c_suite_buying,
            "top_insider_titles": self.top_insider_titles,
            "conviction_score": round(self.conviction_score, 1),
            "conviction_reasons": self.conviction_reasons,
            "qualifies_for_radar": self.qualifies_for_radar,
        }


@dataclass
class SmartMoneyReadiness:
    """Diagnostic readiness status determining lifecycle state."""
    source_ingestion_operational: bool
    market_wide_universe_governed: bool
    universe_denominator: Optional[int]
    freshness_policy_satisfied: bool
    full_scan_completed: bool
    status: str  # PIPELINE_PENDING, DATA_INCOMPLETE, AVAILABLE, NO_MATCHES, UNAVAILABLE
    blocker_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "checks": {
                "source_ingestion_operational": self.source_ingestion_operational,
                "market_wide_universe_governed": self.market_wide_universe_governed,
                "universe_denominator_verified": bool(self.universe_denominator and self.universe_denominator > 0),
                "freshness_policy_satisfied": self.freshness_policy_satisfied,
                "full_scan_completed": self.full_scan_completed,
            },
            "universe_denominator": self.universe_denominator,
            "blocker_reason": self.blocker_reason,
        }


@dataclass
class SmartMoneyScanSnapshot:
    """Immutable scanner snapshot container matching Radar governed contract."""
    scanner_id: str
    version: str
    status: str
    validation_scope: str
    generated_at: str
    universe_id: str
    scannable_count: int
    matched_count: Optional[int]  # Invariant: null if PIPELINE_PENDING or DATA_INCOMPLETE
    items: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    freshness: Dict[str, Any]
    universe_build_id: Optional[str] = None
    universe_content_hash: Optional[str] = None
    eligibility_policy_version: Optional[str] = None
    eligible_count: Optional[int] = None
    cik_mapped_count: Optional[int] = None
    coverage_status: Optional[str] = None
    semantic_fingerprint: str = ""

    def __post_init__(self):
        if not self.semantic_fingerprint:
            payload = {
                "scanner_id": self.scanner_id,
                "version": self.version,
                "status": self.status,
                "universe_id": self.universe_id,
                "universe_build_id": self.universe_build_id,
                "scannable_count": self.scannable_count,
                "matched_count": self.matched_count,
                "items": self.items,
            }
            dumped = json.dumps(payload, sort_keys=True)
            self.semantic_fingerprint = hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanner_id": self.scanner_id,
            "api_contract_version": self.version,
            "status": self.status,
            "validation_scope": self.validation_scope,
            "snapshot": {
                "generated_at": self.generated_at,
                "universe_size": self.scannable_count,
                "matched_count": self.matched_count,
                "semantic_fingerprint": self.semantic_fingerprint,
            },
            "universe": {
                "universe_id": self.universe_id,
                "universe_build_id": self.universe_build_id,
                "universe_content_hash": self.universe_content_hash,
                "eligibility_policy_version": self.eligibility_policy_version,
                "eligible_count": self.eligible_count or self.scannable_count,
                "cik_mapped_count": self.cik_mapped_count or self.scannable_count,
                "scannable_count": self.scannable_count,
                "coverage_status": self.coverage_status or "COMPLETE",
            },
            "provenance": self.provenance,
            "freshness": self.freshness,
            "results": self.items,
        }
