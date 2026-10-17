"""
analyst_dashboard/universe/admission_mapping.py

Sprint 2 Package 5 (S2-06): Policy-Adjudicated Admission Mappings & Evidence Engine.
Implements the minimum offline capability supporting the Product Owner's qualified directions:

1. Stable Internal Identity with Attributable Listing Mappings:
   - Distinguishes ephemeral listing locators (MIC + Symbol + Date) from durable security identity.
   - Enforces effective-dated venue and symbol associations.
   - Prevents symbol reuse from inheriting identity.
   - Preserves multi-symbol issuer separation (e.g. GOOG vs GOOGL under CIK 0001652044).
   - Idempotent ID lookup/generation without duplicate proliferation.
   - Explicitly records that an internal identifier does not independently prove continuity.

2. Explicit Listing Semantics:
   - Separates listing_country from country of incorporation and domicile.
   - Separates selected_us_assessment_listing from global primary listing.
   - Prohibits automatic silent remapping (e.g. listing_country='USA' does NOT automatically set country='USA').

3. Attributable Field Evidence:
   - Explicit evidence structures for currency, classification, venue, and status.
   - Preserves source identity, record locators, content hashes, observation times, and mapping versions.
   - Distinguishes direct provider observations from policy-derived assertions and unevidenced states.
   - Unknown currency remains unresolved; blanket USD convention is strictly NOT approved.

4. Individually Documented Venue Normalization:
   - Normalizes only documented tape codes (A=AMEX, P=ARCA, Z=BATS, N=NYSE, Q=NASDAQ, V=IEX).
   - Preserves original claims and marks unevidenced/provisional codes (e.g. 'F') as UNRESOLVED.
   - Prohibits blanket parent/subsidiary clearance of venue disagreements.

5. Diagnostic Triage vs Formal Classification:
   - Restricts title/suffix regex to diagnostic hints (DiagnosticTriageHint).
   - Keeps unknown instruments visible in population accounting without silent discard.

6. Consumer-Specific Market Data Readiness:
   - Evaluates actual history requirements by consumer (50, 60, 200, 252 bars).
   - Separates NO_DATA, INSUFFICIENT_HISTORY, STALE, INVALID_OHLCV, and UNRESOLVED_ASSOCIATION.
   - Prohibits blanket 60-bar admission gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# =====================================================================
# Canonical Deterministic Serialization & Hashing Helpers
# =====================================================================

def canonical_hash(obj: Any) -> str:
    """Computes deterministic SHA-256 digest of JSON-serializable structure."""
    serialized = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =====================================================================
# 1. Identity & Listing Locator Models
# =====================================================================

@dataclass(frozen=True)
class ListingLocator:
    """
    Tier A: Time-bounded listing locator.
    Represents a venue-symbol pair for a given observation date.
    Strictly ephemeral; does NOT establish durable security identity or corporate continuity.
    """
    operating_mic: str
    symbol: str
    as_of_date: str

    def to_locator_string(self) -> str:
        clean_mic = self.operating_mic.strip().upper()
        clean_sym = self.symbol.strip().upper()
        return f"LOC_{clean_mic}_{clean_sym}_{self.as_of_date}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locator_string": self.to_locator_string(),
            "operating_mic": self.operating_mic,
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "identity_tier": "TIER_A_TIME_BOUNDED_LOCATOR",
            "proves_continuity": False,
        }


@dataclass(frozen=True)
class InternalSecurityRecord:
    """
    Tier B: Governed internal security master entity.
    Identifies a distinct financial contract of an issuer.
    Existence of this ID does NOT certify corporate continuity or external truthfulness.
    """
    internal_security_id: str
    issuer_cik: Optional[str]
    primary_symbol: str
    share_class: Optional[str]
    created_at_utc: str
    status: str = "ACTIVE"  # ACTIVE, RETIRED, SUSPENDED
    continuity_basis: str = "INTERNAL_LEDGER_REGISTERED"
    certifies_continuity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveDatedListingAssociation:
    """
    Associates an internal security ID with an exchange listing over an effective date window.
    Protects against symbol reuse and unverified continuity across ticker changes.

    Boundary Semantics:
    - effective_from: Inclusive start date (YYYY-MM-DD).
    - effective_to: Inclusive end date (YYYY-MM-DD), or None if currently active.
    - Contradictory date ranges (effective_to < effective_from) fail closed immediately.
    """
    association_id: str
    internal_security_id: str
    symbol: str
    operating_mic: str
    effective_from: str  # YYYY-MM-DD
    effective_to: Optional[str] = None  # None indicates currently active
    source_reference: str = ""
    is_disputed: bool = False
    dispute_reason: Optional[str] = None

    def __post_init__(self):
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError(
                f"Contradictory date range: effective_to ({self.effective_to}) "
                f"cannot be earlier than effective_from ({self.effective_from})."
            )

    def is_active_on(self, target_date: str) -> bool:
        if self.effective_from > target_date:
            return False
        if self.effective_to and self.effective_to < target_date:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. Attributable Field Evidence Models
# =====================================================================

class EvidenceNature(str, Enum):
    DIRECT_PROVIDER_OBSERVATION = "DIRECT_PROVIDER_OBSERVATION"
    PROVENANCE_MAPPED = "PROVENANCE_MAPPED"
    POLICY_DERIVED_ASSERTION = "POLICY_DERIVED_ASSERTION"
    DIAGNOSTIC_TRIAGE_HINT = "DIAGNOSTIC_TRIAGE_HINT"
    UNVERIFIED_ASSERTION = "UNVERIFIED_ASSERTION"


@dataclass(frozen=True)
class AttributableFieldEvidence:
    """
    Immutable audit record for a single field's evidence attribution.
    Enforces that existence of evidence does not imply truthfulness or certification.
    """
    field_name: str
    asserted_value: Any
    evidence_nature: EvidenceNature
    source_identifier: str
    source_record_locator: str
    source_content_hash: Optional[str] = None
    observation_timestamp_utc: Optional[str] = None
    transformation_rule: Optional[str] = None
    mapping_version: str = "1.0.0"
    does_certify_claim: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "asserted_value": self.asserted_value,
            "evidence_nature": self.evidence_nature.value,
            "source_identifier": self.source_identifier,
            "source_record_locator": self.source_record_locator,
            "source_content_hash": self.source_content_hash,
            "observation_timestamp_utc": self.observation_timestamp_utc,
            "transformation_rule": self.transformation_rule,
            "mapping_version": self.mapping_version,
            "does_certify_claim": self.does_certify_claim,
        }


# =====================================================================
# 3. Explicit Listing Semantics Models
# =====================================================================

@dataclass
class ListingSemanticAttributes:
    """
    Container for explicit listing semantics decoupled from statutory contract fields.
    Does NOT silently alter existing contracts.country or contracts.primary_listing.
    """
    # Explicit geographic definitions
    listing_country: Optional[str] = None       # Primary venue market jurisdiction (e.g. "USA")
    incorporation_country: Optional[str] = None # Corporate charter sovereignty (e.g. "USA", "TWN", "NLD")
    domicile_country: Optional[str] = None      # Principal executive office headquarters
    contract_country: Optional[str] = None      # Value passed to existing contracts.SourceSecurity.country

    # Explicit listing scope definitions
    selected_us_assessment_listing: Optional[bool] = None  # True if designated as the US radar assessment line
    primary_us_listing: Optional[bool] = None              # True if primary designated US market quote
    global_primary_listing: Optional[bool] = None          # True if global worldwide primary home market
    contract_primary_listing: Optional[bool] = None        # Value passed to existing contracts.SourceSecurity.primary_listing

    # Evidence bindings
    evidence_references: List[AttributableFieldEvidence] = field(default_factory=list)

    @property
    def country_of_incorporation(self) -> Optional[str]:
        """Canonical alias for incorporation_country."""
        return self.incorporation_country

    def to_source_security_kwargs(self) -> Dict[str, Any]:
        """Produces keyword arguments for SourceSecurity explicit fields."""
        return {
            "listing_country": self.listing_country,
            "incorporation_country": self.incorporation_country,
            "selected_us_assessment_listing": self.selected_us_assessment_listing,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listing_country": self.listing_country,
            "incorporation_country": self.incorporation_country,
            "country_of_incorporation": self.country_of_incorporation,
            "domicile_country": self.domicile_country,
            "contract_country": self.contract_country,
            "selected_us_assessment_listing": self.selected_us_assessment_listing,
            "primary_us_listing": self.primary_us_listing,
            "global_primary_listing": self.global_primary_listing,
            "contract_primary_listing": self.contract_primary_listing,
            "evidence_count": len(self.evidence_references),
        }


# =====================================================================
# 4. Individually Documented Venue Normalizer
# =====================================================================

class VenueNormalizationStatus(str, Enum):
    DOCUMENTED_REPRESENTATIONAL = "DOCUMENTED_REPRESENTATIONAL"  # Exact publisher-documented code expansion
    EXACT_CONCORDANT = "EXACT_CONCORDANT"                      # Direct string agreement across sources
    UNRESOLVED_DISAGREEMENT = "UNRESOLVED_DISAGREEMENT"        # Conflicting assertions without documented resolution
    PROVISIONAL_UNRESOLVED = "PROVISIONAL_UNRESOLVED"          # Uncommenced or unverified exchange code (e.g. 'F')
    UNMAPPED_CODE = "UNMAPPED_CODE"                            # Unknown code in source directory


@dataclass(frozen=True)
class VenueNormalizationResult:
    """Result of normalizing an individual source exchange claim."""
    raw_code: str
    source_field: str
    normalized_venue: Optional[str]
    operating_mic: Optional[str]
    dissemination_tape: Optional[str]
    status: VenueNormalizationStatus
    publisher_citation: str
    is_substantive_disagreement: bool = False
    disagreement_details: Optional[str] = None
    mapping_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_code": self.raw_code,
            "source_field": self.source_field,
            "normalized_venue": self.normalized_venue,
            "operating_mic": self.operating_mic,
            "dissemination_tape": self.dissemination_tape,
            "status": self.status.value,
            "publisher_citation": self.publisher_citation,
            "is_substantive_disagreement": self.is_substantive_disagreement,
            "disagreement_details": self.disagreement_details,
            "mapping_version": self.mapping_version,
        }


class SupportedVenueNormalizer:
    """
    Normalizes only individually documented exchange codes from official publisher documentation.
    Preserves original claims and marks unevidenced or conflicting cases UNRESOLVED.
    Does NOT apply blanket parent/subsidiary clearance.
    """

    # Official documented codes from Nasdaq Trader Symbol Directory Technical Specifications
    _NASDAQ_TRADER_OTHER_LISTED_CODES = {
        "A": {
            "venue": "AMEX",
            "mic": "XASE",
            "tape": "TAPE_B",
            "name": "NYSE American",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'A' = NYSE American (Tape B)",
        },
        "N": {
            "venue": "NYSE",
            "mic": "XNYS",
            "tape": "TAPE_A",
            "name": "New York Stock Exchange",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'N' = New York Stock Exchange (Tape A)",
        },
        "P": {
            "venue": "ARCA",
            "mic": "ARCX",
            "tape": "TAPE_B",
            "name": "NYSE Arca",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'P' = NYSE Arca (Tape B)",
        },
        "Z": {
            "venue": "BATS",
            "mic": "BATS",
            "tape": "TAPE_B",
            "name": "BATS Global Markets / Cboe BZX",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'Z' = BATS Global Markets (Tape B)",
        },
        "V": {
            "venue": "IEX",
            "mic": "IEXG",
            "tape": "TAPE_B",
            "name": "Investors Exchange",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'V' = Investors Exchange (Tape B)",
        },
        "Q": {
            "venue": "NASDAQ",
            "mic": "XNAS",
            "tape": "TAPE_C",
            "name": "NASDAQ Stock Market",
            "citation": "Nasdaq Trader Symbol Directory: Exchange code 'Q' = NASDAQ Global Select / Market (Tape C)",
        },
        # Code 'F': Texas Stock Exchange
        "F": {
            "venue": "TXSE",
            "mic": "TXSE",
            "tape": "TAPE_B",
            "name": "Texas Stock Exchange (Provisional)",
            "citation": "Nasdaq Trader Technical Notice: Exchange code 'F' = Texas Stock Exchange (Assigned)",
        },
    }

    @classmethod
    def normalize_directory_code(cls, raw_code: Optional[str]) -> VenueNormalizationResult:
        if not raw_code:
            return VenueNormalizationResult(
                raw_code="",
                source_field="Exchange",
                normalized_venue=None,
                operating_mic=None,
                dissemination_tape=None,
                status=VenueNormalizationStatus.UNMAPPED_CODE,
                publisher_citation="Missing exchange code in directory observation.",
            )

        code_clean = str(raw_code).strip().upper()

        # Handle provisional code 'F' (Texas Stock Exchange)
        if code_clean == "F":
            spec = cls._NASDAQ_TRADER_OTHER_LISTED_CODES["F"]
            return VenueNormalizationResult(
                raw_code=code_clean,
                source_field="Exchange",
                normalized_venue="TXSE",
                operating_mic="TXSE",
                dissemination_tape="TAPE_B",
                status=VenueNormalizationStatus.PROVISIONAL_UNRESOLVED,
                publisher_citation=spec["citation"],
                is_substantive_disagreement=True,
                disagreement_details="Provisional exchange code 'F' (TXSE) is documented but requires operational trading confirmation.",
            )

        if code_clean in cls._NASDAQ_TRADER_OTHER_LISTED_CODES:
            spec = cls._NASDAQ_TRADER_OTHER_LISTED_CODES[code_clean]
            return VenueNormalizationResult(
                raw_code=code_clean,
                source_field="Exchange",
                normalized_venue=spec["venue"],
                operating_mic=spec["mic"],
                dissemination_tape=spec["tape"],
                status=VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL,
                publisher_citation=spec["citation"],
            )

        if code_clean == "NASDAQ":
            return VenueNormalizationResult(
                raw_code=code_clean,
                source_field="Exchange",
                normalized_venue="NASDAQ",
                operating_mic="XNAS",
                dissemination_tape="TAPE_C",
                status=VenueNormalizationStatus.EXACT_CONCORDANT,
                publisher_citation="Nasdaq Listed Directory: Primary market is NASDAQ Stock Market (Tape C)",
            )

        return VenueNormalizationResult(
            raw_code=code_clean,
            source_field="Exchange",
            normalized_venue=None,
            operating_mic=None,
            dissemination_tape=None,
            status=VenueNormalizationStatus.UNMAPPED_CODE,
            publisher_citation=f"Unrecognized exchange code '{code_clean}' in otherlisted directory.",
            is_substantive_disagreement=True,
        )

    @classmethod
    def reconcile_dual_source_venues(
        cls, sec_exchange: Optional[str], directory_code: Optional[str]
    ) -> VenueNormalizationResult:
        """
        Reconciles SEC reported exchange with directory exchange code without blanket parent assumptions.
        """
        sec_clean = str(sec_exchange).strip().upper() if sec_exchange else None
        dir_res = cls.normalize_directory_code(directory_code)

        if not sec_clean:
            return dir_res

        # Case 0: Provisional directory codes remain provisional unresolved
        if dir_res.status == VenueNormalizationStatus.PROVISIONAL_UNRESOLVED or (sec_clean == "TXSE" and dir_res.raw_code == "F"):
            return VenueNormalizationResult(
                raw_code=f"SEC:{sec_clean}|DIR:{dir_res.raw_code}",
                source_field="Composite",
                normalized_venue=dir_res.normalized_venue or "TXSE",
                operating_mic=dir_res.operating_mic or "TXSE",
                dissemination_tape=dir_res.dissemination_tape or "TAPE_B",
                status=VenueNormalizationStatus.PROVISIONAL_UNRESOLVED,
                publisher_citation=f"SEC reports '{sec_clean}' and Directory reports '{dir_res.raw_code}'; operational trading status remains provisional.",
                is_substantive_disagreement=True,
                disagreement_details="Provisional Texas Stock Exchange mapping requires operational confirmation.",
            )

        # Case 1: Direct concordance
        if dir_res.normalized_venue and sec_clean == dir_res.normalized_venue:
            return VenueNormalizationResult(
                raw_code=f"SEC:{sec_clean}|DIR:{dir_res.raw_code}",
                source_field="Composite",
                normalized_venue=dir_res.normalized_venue,
                operating_mic=dir_res.operating_mic,
                dissemination_tape=dir_res.dissemination_tape,
                status=VenueNormalizationStatus.EXACT_CONCORDANT,
                publisher_citation="Both SEC EDGAR and Directory agree on operating exchange.",
            )

        # Case 2: Documented parent-operating relationship (representational normalization)
        # Group 1: SEC reports NYSE, directory reports A (NYSE American)
        if sec_clean == "NYSE" and dir_res.raw_code == "A":
            return VenueNormalizationResult(
                raw_code="SEC:NYSE|DIR:A",
                source_field="Composite",
                normalized_venue="AMEX",
                operating_mic="XASE",
                dissemination_tape="TAPE_B",
                status=VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL,
                publisher_citation="SEC EDGAR groups NYSE American registrants under 'NYSE'; Directory tape code 'A' identifies operating venue.",
                is_substantive_disagreement=False,
            )

        # Group 2: SEC reports NYSE, directory reports P (NYSE Arca)
        if sec_clean == "NYSE" and dir_res.raw_code == "P":
            return VenueNormalizationResult(
                raw_code="SEC:NYSE|DIR:P",
                source_field="Composite",
                normalized_venue="ARCA",
                operating_mic="ARCX",
                dissemination_tape="TAPE_B",
                status=VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL,
                publisher_citation="SEC EDGAR groups NYSE Arca registrants under 'NYSE'; Directory tape code 'P' identifies operating venue.",
                is_substantive_disagreement=False,
            )

        # Group 3: SEC reports CBOE, directory reports Z (BATS/Cboe BZX)
        if sec_clean == "CBOE" and dir_res.raw_code == "Z":
            return VenueNormalizationResult(
                raw_code="SEC:CBOE|DIR:Z",
                source_field="Composite",
                normalized_venue="BATS",
                operating_mic="BATS",
                dissemination_tape="TAPE_B",
                status=VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL,
                publisher_citation="SEC EDGAR reports parent 'CBOE'; Directory tape code 'Z' identifies operating venue BATS/Cboe BZX.",
                is_substantive_disagreement=False,
            )

        # Substantive unresolved conflict
        return VenueNormalizationResult(
            raw_code=f"SEC:{sec_clean}|DIR:{dir_res.raw_code}",
            source_field="Composite",
            normalized_venue=None,
            operating_mic=None,
            dissemination_tape=None,
            status=VenueNormalizationStatus.UNRESOLVED_DISAGREEMENT,
            publisher_citation="Unreconciled venue conflict between SEC and Directory observations.",
            is_substantive_disagreement=True,
            disagreement_details=f"SEC reports '{sec_clean}' while Directory reports '{dir_res.raw_code}' (normalized '{dir_res.normalized_venue}').",
        )


# =====================================================================
# 5. Diagnostic Triage vs Formal Classification
# =====================================================================

@dataclass(frozen=True)
class DiagnosticTriageHint:
    """
    Result of diagnostic pattern matching on security names/symbols.
    Strictly a triage hint; CANNOT certify common stock or discard unknowns.
    """
    security_name: str
    suspected_instrument_type: Optional[str]
    is_diagnostic_hint: bool = True
    certification_established: bool = False
    matching_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClassificationTriageEngine:
    """
    Evaluates naming heuristics for diagnostic triage while preserving explicit UNKNOWN states.
    """

    _TRIAGE_PATTERNS = [
        (r"\b(UNITS?|UTD?)\b", "UNIT"),
        (r"\b(WARRANTS?|WT|WS)\b", "WARRANT"),
        (r"\b(RIGHTS?|RT)\b", "RIGHT"),
        (r"\b(PREFERRED|PFD|PRF)\b", "PREFERRED"),
        (r"\b(DEPOSITARY SHARES?|ADS|ADR)\b", "ADR"),
        (r"\b(HOLDING CO|CORP|INC|CO|LTD|PLC)\b", "SUSPECTED_CORPORATE_EQUITY"),
    ]

    @classmethod
    def evaluate_triage_hint(cls, security_name: Optional[str], is_etf: Optional[bool]) -> DiagnosticTriageHint:
        if is_etf is True:
            return DiagnosticTriageHint(
                security_name=security_name or "",
                suspected_instrument_type="ETF",
                certification_established=False,
                matching_pattern="DIRECTORY_ETF_FLAG",
            )

        if not security_name:
            return DiagnosticTriageHint(
                security_name="",
                suspected_instrument_type="UNKNOWN",
                certification_established=False,
            )

        clean_name = security_name.strip().upper()
        for pat, suspected in cls._TRIAGE_PATTERNS:
            if re.search(pat, clean_name):
                return DiagnosticTriageHint(
                    security_name=security_name,
                    suspected_instrument_type=suspected,
                    certification_established=False,
                    matching_pattern=pat,
                )

        return DiagnosticTriageHint(
            security_name=security_name,
            suspected_instrument_type="UNKNOWN",
            certification_established=False,
        )


# =====================================================================
# 6. Consumer-Specific Market Data Readiness
# =====================================================================

class MarketDataReadinessState(str, Enum):
    CONSUMER_READY = "CONSUMER_READY"                    # Satisfies specific consumer history requirement
    NO_DATA = "NO_DATA"                                  # Symbol absent from market database
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"        # Candle count < consumer lookback
    STALE_OBSERVATIONS = "STALE_OBSERVATIONS"            # Last bar < evaluation cutoff
    INVALID_OHLCV = "INVALID_OHLCV"                      # Price/volume anomalies detected
    UNRESOLVED_ASSOCIATION = "UNRESOLVED_ASSOCIATION"    # Stored data not linked to verified security_id


@dataclass(frozen=True)
class ConsumerReadinessEvaluation:
    """Evaluates readiness against a specific consumer's operational lookback."""
    consumer_name: str
    required_lookback_bars: int
    available_bars: int
    readiness_state: MarketDataReadinessState
    is_ready: bool
    evaluation_cutoff: Optional[str] = None
    latest_bar_date: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consumer_name": self.consumer_name,
            "required_lookback_bars": self.required_lookback_bars,
            "available_bars": self.available_bars,
            "readiness_state": self.readiness_state.value,
            "is_ready": self.is_ready,
            "evaluation_cutoff": self.evaluation_cutoff,
            "latest_bar_date": self.latest_bar_date,
            "notes": self.notes,
        }


class ConsumerReadinessEvaluator:
    """
    Evaluates market-data readiness per consumer without imposing blanket gates.
    Distinguishes minimum valid-history requirements, query limits, indicator lookbacks,
    and benchmark sample sizes.
    """

    CONSUMER_ROLE_SPECS = {
        "INTERMEDIATE_TREND_50": {
            "required_bars": 50,
            "role": "MINIMUM_VALID_HISTORY",
            "description": "50-day moving average and trend template validation (decision_hierarchy.py:152)",
            "is_acceptance_gate": True,
        },
        "RADAR_QUERY_CHUNK_60": {
            "required_bars": 60,
            "role": "QUERY_LIMIT",
            "description": "Database query limit / chunk retrieval buffer (scanner_runner.py:280; operational buffer, not an admission gate)",
            "is_acceptance_gate": False,
        },
        "STAGE2_LONG_TERM_TREND_200": {
            "required_bars": 200,
            "role": "INDICATOR_LOOKBACK",
            "description": "200-day moving average and trend regime evaluation (market_regime_analyzer.py:145)",
            "is_acceptance_gate": True,
        },
        "STAGE2_YEAR_RANGE_252": {
            "required_bars": 252,
            "role": "BENCHMARK_SAMPLE_SIZE",
            "description": "52-week high/low range and annualization benchmark (volatility_forecaster.py:293, market_regime_analyzer.py:100; not an admission gate)",
            "is_acceptance_gate": False,
        },
        "ANNUAL_BENCHMARK_252": {
            "required_bars": 252,
            "role": "BENCHMARK_SAMPLE_SIZE",
            "description": "Annual trading days benchmark sample size / annualization factor",
            "is_acceptance_gate": False,
        },
    }

    KNOWN_CONSUMERS = {
        "INTERMEDIATE_TREND_50": 50,       # 50-day moving average
        "RADAR_QUERY_CHUNK_60": 60,        # db_engine query limit (operational retrieval chunk)
        "STAGE2_LONG_TERM_TREND_200": 200, # 200-day moving average (Rule R05)
        "STAGE2_YEAR_RANGE_252": 252,      # 52-week high/low range (Rule R06)
        "ANNUAL_BENCHMARK_252": 252,       # Annualized benchmark scalar
    }

    @classmethod
    def evaluate_readiness(
        cls,
        symbol: str,
        candles: List[Dict[str, Any]],
        consumer_name: str = "STAGE2_LONG_TERM_TREND_200",
        required_bars: Optional[int] = None,
        evaluation_cutoff: Optional[str] = None,
        is_association_resolved: bool = True,
    ) -> ConsumerReadinessEvaluation:
        req_bars = required_bars or cls.KNOWN_CONSUMERS.get(consumer_name, 200)

        # Invariant: Unresolved symbol/listing association cannot become CONSUMER_READY
        if not is_association_resolved:
            return ConsumerReadinessEvaluation(
                consumer_name=consumer_name,
                required_lookback_bars=req_bars,
                available_bars=len(candles) if candles else 0,
                readiness_state=MarketDataReadinessState.UNRESOLVED_ASSOCIATION,
                is_ready=False,
                evaluation_cutoff=evaluation_cutoff,
                notes=[f"Symbol '{symbol}' has unverified or unresolved security identity association."],
            )

        if not candles:
            return ConsumerReadinessEvaluation(
                consumer_name=consumer_name,
                required_lookback_bars=req_bars,
                available_bars=0,
                readiness_state=MarketDataReadinessState.NO_DATA,
                is_ready=False,
                evaluation_cutoff=evaluation_cutoff,
                notes=[f"Symbol '{symbol}' has zero stored daily candle records."],
            )

        # Check for invalid OHLCV bars
        for idx, bar in enumerate(candles):
            close_p = bar.get("close") or bar.get("currentPrice") or 0.0
            if close_p <= 0.0:
                return ConsumerReadinessEvaluation(
                    consumer_name=consumer_name,
                    required_lookback_bars=req_bars,
                    available_bars=len(candles),
                    readiness_state=MarketDataReadinessState.INVALID_OHLCV,
                    is_ready=False,
                    evaluation_cutoff=evaluation_cutoff,
                    notes=[f"Non-positive close price ({close_p}) detected at candle index {idx}."],
                )

        available_count = len(candles)
        latest_date = str(candles[-1].get("time") or candles[-1].get("date") or "")

        # Check staleness if cutoff provided
        if evaluation_cutoff and latest_date and latest_date < evaluation_cutoff:
            return ConsumerReadinessEvaluation(
                consumer_name=consumer_name,
                required_lookback_bars=req_bars,
                available_bars=available_count,
                readiness_state=MarketDataReadinessState.STALE_OBSERVATIONS,
                is_ready=False,
                evaluation_cutoff=evaluation_cutoff,
                latest_bar_date=latest_date,
                notes=[f"Latest bar date ({latest_date}) is prior to evaluation cutoff ({evaluation_cutoff})."],
            )

        if available_count < req_bars:
            return ConsumerReadinessEvaluation(
                consumer_name=consumer_name,
                required_lookback_bars=req_bars,
                available_bars=available_count,
                readiness_state=MarketDataReadinessState.INSUFFICIENT_HISTORY,
                is_ready=False,
                evaluation_cutoff=evaluation_cutoff,
                latest_bar_date=latest_date,
                notes=[f"Available candle count ({available_count}) is fewer than consumer requirement ({req_bars})."],
            )

        return ConsumerReadinessEvaluation(
            consumer_name=consumer_name,
            required_lookback_bars=req_bars,
            available_bars=available_count,
            readiness_state=MarketDataReadinessState.CONSUMER_READY,
            is_ready=True,
            evaluation_cutoff=evaluation_cutoff,
            latest_bar_date=latest_date,
            notes=[f"Satisfies {consumer_name} requirement ({available_count} >= {req_bars} bars)."],
        )


# =====================================================================
# 7. In-Memory Internal Identity & Listing Mapping Engine
# =====================================================================

class OfflineAdmissionMappingEngine:
    """
    Offline engine for stable internal identity assignment and listing associations.
    Enforces that:
    - Same symbol reused across different effective dates does NOT inherit identity.
    - Symbol changes linked via explicit alias records preserve continuity.
    - Shared issuer CIK across multi-symbol issuers (GOOG vs GOOGL) generates distinct IDs.
    - Reprocessing the exact same input returns the existing ID (idempotent).
    - Issuing an internal ID does not certify an unverified source match.
    """

    def __init__(self):
        # Master security registry: internal_security_id -> InternalSecurityRecord
        self._security_registry: Dict[str, InternalSecurityRecord] = {}
        # Listing association ledger: association_id -> EffectiveDatedListingAssociation
        self._association_ledger: Dict[str, EffectiveDatedListingAssociation] = {}
        # CIK to security ID index: cik -> Set[internal_security_id]
        self._cik_to_securities: Dict[str, Set[str]] = {}
        # Active symbol+mic index: (symbol, mic, as_of_date) -> internal_security_id
        self._active_listing_index: Dict[Tuple[str, str], List[EffectiveDatedListingAssociation]] = {}

    def register_security(
        self,
        symbol: str,
        operating_mic: str,
        as_of_date: str,
        issuer_cik: Optional[str] = None,
        share_class: Optional[str] = None,
        source_reference: str = "",
        explicit_continuity_alias_id: Optional[str] = None,
    ) -> Tuple[InternalSecurityRecord, EffectiveDatedListingAssociation]:
        """
        Registers or retrieves a stable internal security ID and effective-dated listing association.
        """
        clean_sym = symbol.strip().upper()
        clean_mic = operating_mic.strip().upper()
        clean_cik = str(issuer_cik).zfill(10) if issuer_cik else None

        # Check if an existing association already covers this (symbol, mic, date)
        key = (clean_sym, clean_mic)
        existing_assocs = self._active_listing_index.get(key, [])
        for assoc in existing_assocs:
            if assoc.is_active_on(as_of_date):
                sec_rec = self._security_registry[assoc.internal_security_id]
                # If CIK is provided and matches, return existing idempotent record
                if clean_cik is None or sec_rec.issuer_cik == clean_cik:
                    return sec_rec, assoc
                else:
                    # Invariant: Overlapping active association for a distinct identity fails closed as conflict
                    raise ValueError(
                        f"Overlapping mapping conflict: Listing '{clean_sym}' on '{clean_mic}' is already "
                        f"actively associated with distinct security '{sec_rec.internal_security_id}' "
                        f"(CIK {sec_rec.issuer_cik}) on {as_of_date}. Cannot map to CIK {clean_cik} "
                        f"without explicit corporate action or alias resolution."
                    )

        # Check for explicit continuity alias (e.g. ticker change FB -> META)
        if explicit_continuity_alias_id and explicit_continuity_alias_id in self._security_registry:
            sec_rec = self._security_registry[explicit_continuity_alias_id]
            assoc_id = f"ASSOC_{sec_rec.internal_security_id}_{clean_mic}_{clean_sym}_{as_of_date}"
            new_assoc = EffectiveDatedListingAssociation(
                association_id=assoc_id,
                internal_security_id=sec_rec.internal_security_id,
                symbol=clean_sym,
                operating_mic=clean_mic,
                effective_from=as_of_date,
                effective_to=None,
                source_reference=source_reference,
            )
            self._association_ledger[assoc_id] = new_assoc
            self._active_listing_index.setdefault(key, []).append(new_assoc)
            return sec_rec, new_assoc

        # Multi-symbol issuer handling: distinct symbols under the same CIK receive distinct IDs
        # (e.g. GOOG and GOOGL under CIK 0001652044)
        id_material = f"{clean_cik or 'NOCIK'}:{clean_mic}:{clean_sym}:{as_of_date}"
        sec_id_hash = hashlib.sha256(id_material.encode("utf-8")).hexdigest()[:12].upper()
        new_sec_id = f"ARX_SEC_{sec_id_hash}"

        now_utc = datetime.now(timezone.utc).isoformat()
        sec_rec = InternalSecurityRecord(
            internal_security_id=new_sec_id,
            issuer_cik=clean_cik,
            primary_symbol=clean_sym,
            share_class=share_class,
            created_at_utc=now_utc,
            status="ACTIVE",
        )
        self._security_registry[new_sec_id] = sec_rec

        if clean_cik:
            self._cik_to_securities.setdefault(clean_cik, set()).add(new_sec_id)

        assoc_id = f"ASSOC_{new_sec_id}_{clean_mic}_{clean_sym}_{as_of_date}"
        new_assoc = EffectiveDatedListingAssociation(
            association_id=assoc_id,
            internal_security_id=new_sec_id,
            symbol=clean_sym,
            operating_mic=clean_mic,
            effective_from=as_of_date,
            effective_to=None,
            source_reference=source_reference,
        )
        self._association_ledger[assoc_id] = new_assoc
        self._active_listing_index.setdefault(key, []).append(new_assoc)

        return sec_rec, new_assoc

    def get_securities_for_cik(self, cik: str) -> List[InternalSecurityRecord]:
        clean_cik = str(cik).zfill(10)
        sec_ids = self._cik_to_securities.get(clean_cik, set())
        return [self._security_registry[sid] for sid in sorted(list(sec_ids))]

    def close_association(self, association_id: str, effective_to: str) -> None:
        if association_id in self._association_ledger:
            old = self._association_ledger[association_id]
            if effective_to < old.effective_from:
                raise ValueError(
                    f"Contradictory date range: effective_to ({effective_to}) cannot be earlier "
                    f"than effective_from ({old.effective_from})."
                )
            updated = EffectiveDatedListingAssociation(
                association_id=old.association_id,
                internal_security_id=old.internal_security_id,
                symbol=old.symbol,
                operating_mic=old.operating_mic,
                effective_from=old.effective_from,
                effective_to=effective_to,
                source_reference=old.source_reference,
                is_disputed=old.is_disputed,
                dispute_reason=old.dispute_reason,
            )
            self._association_ledger[association_id] = updated
            key = (old.symbol, old.operating_mic)
            if key in self._active_listing_index:
                self._active_listing_index[key] = [
                    updated if a.association_id == association_id else a
                    for a in self._active_listing_index[key]
                ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete internal engine state for persistent checkpointing."""
        return {
            "security_registry": {k: v.to_dict() for k, v in self._security_registry.items()},
            "association_ledger": {k: v.to_dict() for k, v in self._association_ledger.items()},
            "cik_to_securities": {k: sorted(list(v)) for k, v in self._cik_to_securities.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OfflineAdmissionMappingEngine:
        """Reloads engine state preserving exact identity and associations without regenerating IDs."""
        engine = cls()
        for k, v in data.get("security_registry", {}).items():
            engine._security_registry[k] = InternalSecurityRecord(**v)
        for k, v in data.get("association_ledger", {}).items():
            assoc = EffectiveDatedListingAssociation(**v)
            engine._association_ledger[k] = assoc
            key = (assoc.symbol, assoc.operating_mic)
            engine._active_listing_index.setdefault(key, []).append(assoc)
        for k, v in data.get("cik_to_securities", {}).items():
            engine._cik_to_securities[k] = set(v)
        return engine
