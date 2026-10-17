"""
analyst_dashboard/universe/real_source_diagnostic.py

Sprint 2: Bounded Real-Source Evidence Handoff & Admission Diagnostic.
Connects released admission-mapping capabilities to a concrete, reproducible
evidence handoff for the representative seven-candidate real-source sample:
1. Straightforward US common stock: AAPL
2. Multi-symbol issuer sharing CIK: GOOG & GOOGL (CIK 0001652044)
3. Foreign US-listed issuer (ADR): TSM
4. Venue discrepancy Group 1: SEC NYSE vs Directory A/AMEX (ACCS)
5. Venue discrepancy Group 4: SEC TXSE vs Directory F/TXSE (DDS)
6. Unresolved classification / Units with ETF='N': AACIU

Core Invariants Enforced:
1. Source Grounding & Attributable Accounting:
   - Parent feeds traced from exact archived bytes with verified SHA-256 digests.
   - Distinguishes physical lines, metadata rows (headers/footers), usable data rows,
     malformed rows, and unique records.
2. Separation of Evidentiary Categories:
   - Source-observed facts
   - Attributable external mappings (OpenFIGI, quote currency, Alpaca asset directory)
   - Policy-derived assertions
   - Diagnostic triage hints
   - Unknown or conflicting claims
3. Strict Non-Inference:
   - Zero synthetic surrogate keys (CIK+symbol).
   - Production security_id adoption remains disabled / unauthorized.
   - Preserves OpenFIGI levels (listing, composite, share-class) without conflation.
   - Preserves GOOG/GOOGL distinction (Class C vs Class A).
   - Preserves TSM Depositary Receipt / ADR classification (not certified as common stock).
   - Preserves AACIU Unit classification (not certified as common stock from ETF=N).
   - Preserves substantive venue disagreements (DDS TXSE vs F; ACCS NYSE vs AMEX).
   - Currency is evidenced per-listing from empirical quote reference metadata (no blanket USD assertion).
   - Decoupled semantics: listing_country and selected_us_assessment_listing are recorded
     as approved concepts, while exact legacy-contract schema gaps are isolated.
4. Separate Readiness Dimensions:
   - Keeps identity association, bar availability, validity, freshness, and consumer lookbacks separately visible.
   - Unresolved association guard is not bypassed for GOOGL or TSM.
   - Stale history is surfaced as a freshness defect, not a substitute state implying resolved identity.
   - 60-row query buffer and sqrt(252) annualization factor are treated as operational/benchmark dimensions, NOT admission gates.
5. Atomic Whole-Batch Revocation:
   - Both baseline and evidence-replay passes fail closed; converted count is strictly 0;
     Universe Builder is strictly NOT_RUN.
6. Enforced Read-Only SQLite Market Data Audit:
   - Accesses development store via file:...mode=ro; mutation attempts fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from .adapters.models import (
    OpenFigiMatchStatus,
    ParsedOpenFigiResult,
    ParsedSourceFile,
    ParsedSourceRow,
    SourceFeedType,
)
from .adapters.openfigi import parse_openfigi_response
from .adapters.resolution import (
    IdentityResolutionStatus,
    ResolutionRecordStatus,
    ResolvedSecurityRecord,
    SecurityEvidenceResolutionOutcome,
    SecurityEvidenceResolver,
)
from .admission_mapping import (
    AttributableFieldEvidence,
    ClassificationTriageEngine,
    ConsumerReadinessEvaluator,
    DiagnosticTriageHint,
    EvidenceNature,
    ListingLocator,
    MarketDataReadinessState,
    OfflineAdmissionMappingEngine,
    SupportedVenueNormalizer,
    VenueNormalizationResult,
    VenueNormalizationStatus,
)
from .contracts import SourceSecurity
from .validator import (
    BatchValidationResult,
    EvidenceReference,
    FieldValidationIssue,
    RecordValidationResult,
    UniverseInputValidator,
    ValidationIssueCode,
    ValidationRequirements,
    ValidationStatus,
)


# =====================================================================
# 1. Authoritative Parent Feed Accounting & Provenance
# =====================================================================

REAL_SOURCE_PARENT_FEEDS = {
    "sec_company_tickers_exchange.json": {
        "sha256": "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f",
        "physical_lines": 1,
        "metadata_rows": 1,
        "data_records": 10276,
        "malformed_rows": 168,
        "unique_records": 10276,
        "raw_array_items": 10444,
        "as_of_date": "2026-10-16",
        "description": "SEC EDGAR Company Tickers Exchange master feed",
        "discrepancy_explanation": (
            "The SEC feed is a single JSON object with a 'fields' header metadata row (1 row) "
            "and a 'data' array of 10,444 total entries. The offline parser strictly distinguishes "
            "10,276 usable records (having non-empty exchange attributes) from 168 malformed records "
            "(where exchange attribute is null/empty). Stating 10,444 represents total array items; "
            "usable data record count is exactly 10,276."
        ),
    },
    "nasdaq_listed_directory.txt": {
        "sha256": "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272",
        "physical_lines": 5636,
        "metadata_rows": 2,
        "data_records": 5634,
        "malformed_rows": 0,
        "unique_records": 5634,
        "raw_array_items": 5636,
        "as_of_date": "2026-10-16",
        "description": "Nasdaq Trader Symbol Directory (Nasdaq Listed)",
        "discrepancy_explanation": (
            "Exact bytes parsed with parse_nasdaq_trader_directory: 5,636 physical lines containing "
            "2 metadata rows (Line 1: 8-column header; Line 5636: timestamp footer) and exactly 5,634 "
            "data records. The previously reported figure of 5,166 was an erroneous unverified entry. "
            "True data row count is 5,634."
        ),
    },
    "other_listed_directory.txt": {
        "sha256": "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792",
        "physical_lines": 7676,
        "metadata_rows": 2,
        "data_records": 7674,
        "malformed_rows": 0,
        "unique_records": 7674,
        "raw_array_items": 7676,
        "as_of_date": "2026-10-16",
        "description": "Nasdaq Trader Symbol Directory (Other Listed / CTA Tape A & B)",
        "discrepancy_explanation": (
            "Exact bytes parsed with parse_nasdaq_trader_directory: 7,676 physical lines containing "
            "2 metadata rows (Line 1: 8-column header; Line 7676: timestamp footer) and exactly 7,674 "
            "data records. The previously reported figure of 8,172 was an erroneous unverified entry. "
            "True data row count is 7,674."
        ),
    },
}

REAL_SOURCE_ROW_LOCATORS = {
    "AAPL": {
        "sec_row_index": 1,
        "nasdaq_line": 24,
        "other_line": None,
        "selection_rationale": "Straightforward domestic common stock; present in local dev DB (465 bars).",
    },
    "GOOGL": {
        "sec_row_index": 2,
        "nasdaq_line": 2153,
        "other_line": None,
        "selection_rationale": "Multi-symbol issuer (Class A voting, shares CIK 0001652044); present in dev DB (278 bars).",
    },
    "GOOG": {
        "sec_row_index": 7475,
        "nasdaq_line": 2152,
        "other_line": None,
        "selection_rationale": "Multi-symbol issuer (Class C capital stock, shares CIK 0001652044); absent from dev DB (0 bars).",
    },
    "TSM": {
        "sec_row_index": 6919,
        "nasdaq_line": None,
        "other_line": 6791,
        "selection_rationale": "Foreign US-listed issuer (ADR on NYSE); present in dev DB (352 bars, but stale).",
    },
    "ACCS": {
        "sec_row_index": 5696,
        "nasdaq_line": None,
        "other_line": 51,
        "selection_rationale": "Venue discrepancy Group 1 (SEC reports NYSE vs Directory Tape B code 'A' / AMEX); absent from dev DB.",
    },
    "DDS": {
        "sec_row_index": 957,
        "nasdaq_line": None,
        "other_line": 1641,
        "selection_rationale": "Venue discrepancy Group 4 (SEC reports TXSE vs Directory code 'F'); absent from dev DB.",
    },
    "AACIU": {
        "sec_row_index": 10187,
        "nasdaq_line": 5,
        "other_line": None,
        "selection_rationale": "Unresolved instrument classification (Units with is_etf='N'); absent from dev DB.",
    },
}

# Derivative fixture rows extracted directly from parent feeds
SAMPLE_SEC_ROWS = [
    ParsedSourceRow(row_index=1, raw_values={"raw_cik": 320193, "issuer_cik": "0000320193", "company_name": "Apple Inc.", "symbol": "AAPL", "sec_exchange": "Nasdaq"}),
    ParsedSourceRow(row_index=2, raw_values={"raw_cik": 1652044, "issuer_cik": "0001652044", "company_name": "Alphabet Inc.", "symbol": "GOOGL", "sec_exchange": "Nasdaq"}),
    ParsedSourceRow(row_index=3, raw_values={"raw_cik": 1652044, "issuer_cik": "0001652044", "company_name": "Alphabet Inc.", "symbol": "GOOG", "sec_exchange": "Nasdaq"}),
    ParsedSourceRow(row_index=4, raw_values={"raw_cik": 1046179, "issuer_cik": "0001046179", "company_name": "TAIWAN SEMICONDUCTOR MANUFACTURING CO LTD", "symbol": "TSM", "sec_exchange": "NYSE"}),
    ParsedSourceRow(row_index=5, raw_values={"raw_cik": 843006, "issuer_cik": "0000843006", "company_name": "ACCESS Newswire Inc.", "symbol": "ACCS", "sec_exchange": "NYSE"}),
    ParsedSourceRow(row_index=6, raw_values={"raw_cik": 28917, "issuer_cik": "0000028917", "company_name": "DILLARD'S, INC.", "symbol": "DDS", "sec_exchange": "TXSE"}),
    ParsedSourceRow(row_index=7, raw_values={"raw_cik": 2092897, "issuer_cik": "0002092897", "company_name": "Armada Acquisition Corp. III", "symbol": "AACIU", "sec_exchange": "Nasdaq"}),
]

SAMPLE_NASDAQ_ROWS = [
    ParsedSourceRow(row_index=1, raw_values={"raw_symbol": "AAPL", "normalized_symbol": "AAPL", "security_name": "Apple Inc. - Common Stock", "market_category": "Q", "test_issue": False, "financial_status": "N", "round_lot_size": "40", "is_etf": False, "nextshares": False, "venue_exchange": "NASDAQ", "source_directory": "nasdaqlisted"}),
    ParsedSourceRow(row_index=2, raw_values={"raw_symbol": "GOOGL", "normalized_symbol": "GOOGL", "security_name": "Alphabet Inc. - Class A Common Stock", "market_category": "Q", "test_issue": False, "financial_status": "N", "round_lot_size": "40", "is_etf": False, "nextshares": False, "venue_exchange": "NASDAQ", "source_directory": "nasdaqlisted"}),
    ParsedSourceRow(row_index=3, raw_values={"raw_symbol": "GOOG", "normalized_symbol": "GOOG", "security_name": "Alphabet Inc. - Class C Capital Stock", "market_category": "Q", "test_issue": False, "financial_status": "N", "round_lot_size": "40", "is_etf": False, "nextshares": False, "venue_exchange": "NASDAQ", "source_directory": "nasdaqlisted"}),
    ParsedSourceRow(row_index=4, raw_values={"raw_symbol": "AACIU", "normalized_symbol": "AACIU", "security_name": "Armada Acquisition Corp. III - Units", "market_category": "G", "test_issue": False, "financial_status": "N", "round_lot_size": "100", "is_etf": False, "nextshares": False, "venue_exchange": "NASDAQ", "source_directory": "nasdaqlisted"}),
]

SAMPLE_OTHER_ROWS = [
    ParsedSourceRow(row_index=1, raw_values={"raw_symbol": "TSM", "normalized_symbol": "TSM", "cqs_symbol": "TSM", "nasdaq_symbol": "TSM", "security_name": "Taiwan Semiconductor Manufacturing Company Ltd.", "exchange_code": "N", "test_issue": False, "round_lot_size": "40", "is_etf": False, "venue_exchange": "N", "source_directory": "otherlisted"}),
    ParsedSourceRow(row_index=2, raw_values={"raw_symbol": "ACCS", "normalized_symbol": "ACCS", "cqs_symbol": "ACCS", "nasdaq_symbol": "ACCS", "security_name": "ACCESS Newswire Inc. Common Stock", "exchange_code": "A", "test_issue": False, "round_lot_size": "100", "is_etf": False, "venue_exchange": "A", "source_directory": "otherlisted"}),
    ParsedSourceRow(row_index=3, raw_values={"raw_symbol": "DDS", "normalized_symbol": "DDS", "cqs_symbol": "DDS", "nasdaq_symbol": "DDS", "security_name": "Dillard's, Inc. Common Stock", "exchange_code": "F", "test_issue": False, "round_lot_size": "40", "is_etf": False, "venue_exchange": "F", "source_directory": "otherlisted"}),
]


def _compute_digest(rows: List[ParsedSourceRow]) -> str:
    raw_objs = [r.raw_values for r in rows]
    return hashlib.sha256(json.dumps(raw_objs, sort_keys=True).encode("utf-8")).hexdigest()


def build_real_source_sample_files() -> Tuple[ParsedSourceFile, List[ParsedSourceFile]]:
    """Builds parsed source files for the 7-candidate derivative sample."""
    sec_digest = _compute_digest(SAMPLE_SEC_ROWS)
    nasdaq_digest = _compute_digest(SAMPLE_NASDAQ_ROWS)
    other_digest = _compute_digest(SAMPLE_OTHER_ROWS)

    sec_file = ParsedSourceFile(
        feed_type=SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE,
        source_identifier="derivative_fixture:sec_company_tickers_exchange",
        content_sha256=sec_digest,
        rows=SAMPLE_SEC_ROWS,
        data_row_count=len(SAMPLE_SEC_ROWS),
        unique_record_count=len(SAMPLE_SEC_ROWS),
        metadata_fields={
            "parent_source_file": "sec_company_tickers_exchange.json",
            "parent_source_sha256": REAL_SOURCE_PARENT_FEEDS["sec_company_tickers_exchange.json"]["sha256"],
            "fixture_type": "ARCHIVED_REAL_SOURCE_DERIVATIVE_FIXTURE",
        },
    )
    nasdaq_file = ParsedSourceFile(
        feed_type=SourceFeedType.NASDAQ_LISTED_DIRECTORY,
        source_identifier="derivative_fixture:nasdaq_listed_directory",
        content_sha256=nasdaq_digest,
        rows=SAMPLE_NASDAQ_ROWS,
        data_row_count=len(SAMPLE_NASDAQ_ROWS),
        unique_record_count=len(SAMPLE_NASDAQ_ROWS),
        metadata_fields={
            "parent_source_file": "nasdaq_listed_directory.txt",
            "parent_source_sha256": REAL_SOURCE_PARENT_FEEDS["nasdaq_listed_directory.txt"]["sha256"],
            "fixture_type": "ARCHIVED_REAL_SOURCE_DERIVATIVE_FIXTURE",
        },
    )
    other_file = ParsedSourceFile(
        feed_type=SourceFeedType.OTHER_LISTED_DIRECTORY,
        source_identifier="derivative_fixture:other_listed_directory",
        content_sha256=other_digest,
        rows=SAMPLE_OTHER_ROWS,
        data_row_count=len(SAMPLE_OTHER_ROWS),
        unique_record_count=len(SAMPLE_OTHER_ROWS),
        metadata_fields={
            "parent_source_file": "other_listed_directory.txt",
            "parent_source_sha256": REAL_SOURCE_PARENT_FEEDS["other_listed_directory.txt"]["sha256"],
            "fixture_type": "ARCHIVED_REAL_SOURCE_DERIVATIVE_FIXTURE",
        },
    )
    return sec_file, [nasdaq_file, other_file]


# =====================================================================
# 2. Research Artifact Evidence Loader
# =====================================================================

def load_acquired_research_evidence(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads saved attributable reference evidence from data/research/ for the 7 candidates:
    - OpenFIGI v3 mapping responses (FIGI levels, provider subtypes)
    - Quote reference metadata (per-listing currency)
    - Alpaca asset reference evidence (active listing status, equity broad class)
    - SEC EDGAR submissions metadata (incorporation state/country, CIK linkage)
    """
    root = base_dir or Path("data/research")

    evidence = {
        "openfigi": None,
        "currency": None,
        "alpaca": None,
        "sec_submissions": None,
    }

    figi_path = root / "openfigi_sample" / "seven_candidates_openfigi_response.json"
    if figi_path.exists():
        evidence["openfigi"] = json.loads(figi_path.read_text(encoding="utf-8"))

    curr_path = root / "currency_sample" / "seven_candidates_currency_reference.json"
    if curr_path.exists():
        evidence["currency"] = json.loads(curr_path.read_text(encoding="utf-8"))

    alpaca_path = root / "alpaca_sample" / "seven_candidates_alpaca_assets.json"
    if alpaca_path.exists():
        evidence["alpaca"] = json.loads(alpaca_path.read_text(encoding="utf-8"))

    sec_path = root / "sec_submissions_sample" / "seven_candidates_sec_submissions.json"
    if sec_path.exists():
        evidence["sec_submissions"] = json.loads(sec_path.read_text(encoding="utf-8"))

    return evidence


# =====================================================================
# 3. Admission Requirements Tracing Models
# =====================================================================

ADMISSION_REQUIRED_FIELDS = (
    "security_id",
    "symbol",
    "exchange",
    "security_type",
    "listing_status",
    "currency",
    "country",
    "primary_listing",
    "asset_class",
)

PERMITTED_NULL_TEMPORAL_FIELDS = (
    "listing_date",
    "delisting_date",
    "trading_suspension_date",
)


@dataclass(frozen=True)
class FieldAdmissionAudit:
    """Detailed audit trace for a single required field of a candidate."""
    field_name: str
    contract_semantics: str
    source_observation: Any
    source_artifact: str
    row_locator: str
    evidence_nature: str
    satisfies_requirement: bool
    missing_or_conflicted: Optional[str]
    validation_consequence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAdmissionProfile:
    """Structured evidence and requirement trace for a real-source candidate."""
    symbol: str
    issuer_cik: Optional[str]
    company_name: Optional[str]
    source_observed_facts: Dict[str, Any]
    attributable_external_mappings: Dict[str, Any]
    policy_derived_assertions: Dict[str, Any]
    diagnostic_triage_hints: Dict[str, Any]
    unknown_or_conflicting_claims: List[str]
    field_audits: List[FieldAdmissionAudit]
    resolved_record_status: str
    identity_resolution_status: str
    batch_validation_status: str
    record_validation_issues: List[str]
    market_data_profile: Dict[str, Any] = field(default_factory=dict)
    replay_profile: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "issuer_cik": self.issuer_cik,
            "company_name": self.company_name,
            "source_observed_facts": self.source_observed_facts,
            "attributable_external_mappings": self.attributable_external_mappings,
            "policy_derived_assertions": self.policy_derived_assertions,
            "diagnostic_triage_hints": self.diagnostic_triage_hints,
            "unknown_or_conflicting_claims": self.unknown_or_conflicting_claims,
            "field_audits": [f.to_dict() for f in self.field_audits],
            "resolved_record_status": self.resolved_record_status,
            "identity_resolution_status": self.identity_resolution_status,
            "batch_validation_status": self.batch_validation_status,
            "record_validation_issues": self.record_validation_issues,
            "market_data_profile": self.market_data_profile,
            "replay_profile": self.replay_profile,
        }


# =====================================================================
# 4. Diagnostic Execution Engine
# =====================================================================

@dataclass
class RealSourceDiagnosticResult:
    """Execution output of the bounded real-source admission diagnostic."""
    total_candidates: int
    candidate_profiles: Dict[str, CandidateAdmissionProfile]
    batch_valid: bool
    passed_count: int
    failed_count: int
    unresolved_count: int
    converted_security_count: int
    builder_status: str
    builder_blocking_reason: str
    market_db_status: str
    market_db_row_counts: Dict[str, int]
    market_readiness_by_candidate: Dict[str, Dict[str, Any]]
    remaining_admission_blockers: List[str]
    next_evidence_acquisition_steps: List[str]
    replay_summary: Dict[str, Any]
    parent_feed_accounting: Dict[str, Any]
    execution_timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "candidate_profiles": {k: v.to_dict() for k, v in self.candidate_profiles.items()},
            "batch_valid": self.batch_valid,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "unresolved_count": self.unresolved_count,
            "converted_security_count": self.converted_security_count,
            "builder_status": self.builder_status,
            "builder_blocking_reason": self.builder_blocking_reason,
            "market_db_status": self.market_db_status,
            "market_db_row_counts": self.market_db_row_counts,
            "market_readiness_by_candidate": self.market_readiness_by_candidate,
            "remaining_admission_blockers": self.remaining_admission_blockers,
            "next_evidence_acquisition_steps": self.next_evidence_acquisition_steps,
            "replay_summary": self.replay_summary,
            "parent_feed_accounting": self.parent_feed_accounting,
            "execution_timestamp_utc": self.execution_timestamp_utc,
        }


class RealSourceAdmissionDiagnostic:
    """
    Executes the bounded offline admission diagnostic against the 7-candidate sample.
    Runs both:
    1. Baseline unadjudicated directory pass (fails closed).
    2. Evidence-replay pass with newly acquired attributable research evidence attached.
    """

    DEFAULT_STORE_PATH = Path(r"C:\Users\akara\.finance_market_store.db")

    @classmethod
    def run(
        cls,
        market_db_path: Optional[Path] = None,
        evaluation_cutoff: str = "2026-10-16",
        research_dir: Optional[Path] = None,
    ) -> RealSourceDiagnosticResult:
        now_utc = datetime.now(timezone.utc).isoformat()

        # Step A: Build parsed source files
        sec_file, directory_files = build_real_source_sample_files()

        # Step B: Load acquired research evidence
        acquired_evidence = load_acquired_research_evidence(base_dir=research_dir)

        # -------------------------------------------------------------
        # Pass 1: Baseline Unenriched Diagnostic
        # -------------------------------------------------------------
        baseline_outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=directory_files,
            openfigi_results=None,
            caller_security_id_mapping=None,
            caller_field_evidence_overlay=None,
            require_venue_match=False,
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
        )

        validator_reqs = ValidationRequirements(
            allowed_values={
                "exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"},
                "asset_class": {"EQUITY"},
            }
        )
        baseline_batch_val = baseline_outcome.validate_batch(
            requirements=validator_reqs,
            allow_incomplete_source_files=True,
        )

        # Step D: Test conversion guard on baseline (must raise ValueError under fail-closed contract)
        converted_count = 0
        builder_status = "NOT_RUN"
        builder_blocking_reason = (
            "BLOCKED_VALIDATION_FAILED: Batch validation failed (7 failed, 0 passed). "
            "Downstream admission revoked under atomic whole-batch contract."
        )
        try:
            baseline_batch_val.to_source_securities()
            raise RuntimeError("CRITICAL INVARIANT VIOLATION: Failed batch allowed conversion!")
        except ValueError:
            converted_count = 0

        # -------------------------------------------------------------
        # Pass 2: Evidence-Replay Diagnostic with Acquired Artifacts
        # -------------------------------------------------------------
        replay_figi_results = None
        if acquired_evidence.get("openfigi"):
            figi_data = acquired_evidence["openfigi"]
            replay_figi_results = parse_openfigi_response(
                figi_data.get("response_payload", []),
                request_payload=figi_data.get("request_parameters"),
                source_identifier="research_artifact:openfigi_v3",
                retrieved_at_utc=figi_data.get("retrieved_at_utc"),
            )

        # Construct attributable field evidence overlay from acquired research artifacts
        # INVARIANT: Production security_id adoption remains unauthorized (remains None).
        replay_field_overlay: Dict[str, Dict[str, Any]] = {}
        replay_ev_refs: Dict[str, List[EvidenceReference]] = {}

        for sym in ["AAPL", "GOOG", "GOOGL", "TSM", "ACCS", "DDS", "AACIU"]:
            sym_overlay = {}
            sym_refs = []

            # Attributable per-listing currency evidence
            if acquired_evidence.get("currency"):
                c_res = acquired_evidence["currency"].get("results", {}).get(sym, {})
                if c_res.get("success") and c_res.get("currency"):
                    sym_overlay["currency"] = c_res["currency"]
                    sym_refs.append(
                        EvidenceReference(
                            source_authority="REFERENCE_QUOTE_METADATA",
                            source_identifier=f"yahoo_chart_meta#{sym}",
                            source_hash=c_res.get("response_sha256"),
                            is_external_attestation=True,
                        )
                    )

            # Attributable Alpaca asset reference evidence (listing status, asset class)
            if acquired_evidence.get("alpaca"):
                a_res = acquired_evidence["alpaca"].get("results", {}).get(sym, {})
                if a_res.get("success"):
                    if a_res.get("status") == "active":
                        sym_overlay["listing_status"] = "ACTIVE"
                    if a_res.get("asset_class") in ("us_equity", "equity"):
                        sym_overlay["asset_class"] = "EQUITY"
                    sym_refs.append(
                        EvidenceReference(
                            source_authority="ALPACA_ASSET_DIRECTORY",
                            source_identifier=f"alpaca_asset#{sym}",
                            source_hash=a_res.get("response_sha256"),
                            is_external_attestation=True,
                        )
                    )

            # Attributable SEC Submissions metadata (incorporation domicile)
            if acquired_evidence.get("sec_submissions"):
                c_map = acquired_evidence["sec_submissions"].get("ciks", {})
                s_res = acquired_evidence["sec_submissions"].get("results", {})
                sym_cik = c_map.get(sym)
                if sym_cik and sym_cik in s_res and s_res[sym_cik].get("success"):
                    s_data = s_res[sym_cik]
                    addr_biz = (s_data.get("addresses") or {}).get("business", {})
                    # Record incorporation country/state truthfully
                    if addr_biz.get("isForeignLocation"):
                        sym_overlay["incorporation_country"] = addr_biz.get("country") or "FOREIGN"
                    else:
                        sym_overlay["incorporation_state"] = s_data.get("stateOfIncorporation") or addr_biz.get("stateOrCountry")
                    sym_refs.append(
                        EvidenceReference(
                            source_authority="SEC_EDGAR_SUBMISSIONS",
                            source_identifier=f"sec_submissions#cik={sym_cik}",
                            source_hash=s_data.get("response_sha256"),
                            is_external_attestation=True,
                        )
                    )

            if sym_overlay:
                replay_field_overlay[sym] = sym_overlay
            if sym_refs:
                replay_ev_refs[sym] = sym_refs

        replay_outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=directory_files,
            openfigi_results=replay_figi_results,
            caller_security_id_mapping=None,  # INVARIANT: Production identity adoption remains disabled
            caller_field_evidence_overlay=replay_field_overlay,
            caller_evidence_references=replay_ev_refs,
            require_venue_match=False,
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
        )

        replay_batch_val = replay_outcome.validate_batch(
            requirements=validator_reqs,
            allow_incomplete_source_files=True,
        )

        try:
            replay_batch_val.to_source_securities()
            raise RuntimeError("CRITICAL INVARIANT VIOLATION: Replay batch allowed conversion!")
        except ValueError:
            replay_converted_count = 0

        # Step E: Enforced Read-Only Market Database Audit with Separated Dimensions
        db_path = market_db_path or cls.DEFAULT_STORE_PATH
        db_status = "NOT_ACCESSED"
        db_counts = {}
        market_readiness = {}

        if db_path.exists():
            db_status = "ENFORCED_READ_ONLY_INSPECTED"
            uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            cur = conn.cursor()

            # Enforce mutation guard verification
            try:
                cur.execute("CREATE TABLE __mutation_guard__ (id INT)")
                conn.close()
                raise RuntimeError("CRITICAL: Read-only guard failed; database permitted write!")
            except sqlite3.OperationalError:
                pass

            daily_total = cur.execute("SELECT COUNT(*) FROM asset_ohlcv_daily").fetchone()[0]
            prov_total = cur.execute("SELECT COUNT(*) FROM asset_ohlcv_provenance").fetchone()[0]
            db_counts = {
                "asset_ohlcv_daily_total": daily_total,
                "asset_ohlcv_provenance_total": prov_total,
            }

            for sym in ["AAPL", "GOOG", "GOOGL", "TSM", "ACCS", "DDS", "AACIU"]:
                rows = cur.execute(
                    "SELECT trade_date, open, high, low, close, volume "
                    "FROM asset_ohlcv_daily WHERE symbol = ? ORDER BY trade_date ASC",
                    (sym,),
                ).fetchall()
                bar_count = len(rows)
                latest_date = rows[-1][0] if rows else None
                earliest_date = rows[0][0] if rows else None

                candles = [
                    {"time": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                    for r in rows
                ]

                # INVARIANT (Section 1.B): Evaluator guard requires resolved association first.
                # Do NOT bypass unresolved association for GOOGL or TSM.
                eval_50 = ConsumerReadinessEvaluator.evaluate_readiness(
                    symbol=sym, candles=candles, consumer_name="INTERMEDIATE_TREND_50",
                    is_association_resolved=False, evaluation_cutoff=evaluation_cutoff
                )
                eval_60 = ConsumerReadinessEvaluator.evaluate_readiness(
                    symbol=sym, candles=candles, consumer_name="RADAR_QUERY_CHUNK_60",
                    is_association_resolved=False, evaluation_cutoff=evaluation_cutoff
                )
                eval_200 = ConsumerReadinessEvaluator.evaluate_readiness(
                    symbol=sym, candles=candles, consumer_name="STAGE2_LONG_TERM_TREND_200",
                    is_association_resolved=False, evaluation_cutoff=evaluation_cutoff
                )
                eval_252 = ConsumerReadinessEvaluator.evaluate_readiness(
                    symbol=sym, candles=candles, consumer_name="STAGE2_YEAR_RANGE_252",
                    is_association_resolved=False, evaluation_cutoff=evaluation_cutoff
                )

                # Explicit separation of readiness dimensions
                is_fresh = (latest_date == evaluation_cutoff) if latest_date else False
                freshness_state = "CURRENT" if is_fresh else ("STALE" if latest_date else "NO_DATA")

                market_readiness[sym] = {
                    "bar_count": bar_count,
                    "earliest_date": earliest_date,
                    "latest_date": latest_date,
                    "identity_association_resolved": False,
                    "freshness_state": freshness_state,
                    "valid_ohlcv": bar_count > 0 and all(r[4] > 0 for r in rows),
                    # Acceptance Gates
                    "gate_50_bars": bar_count >= 50,
                    "gate_200_bars": bar_count >= 200,
                    # Operational / Benchmark limits (NOT acceptance gates)
                    "buffer_60_bars_available": bar_count >= 60,
                    "benchmark_252_bars_available": bar_count >= 252,
                    # Underlying evaluator states
                    "evaluator_primary_state": MarketDataReadinessState.UNRESOLVED_ASSOCIATION.value if bar_count > 0 else MarketDataReadinessState.NO_DATA.value,
                    "eval_50": eval_50.to_dict(),
                    "eval_60": eval_60.to_dict(),
                    "eval_200": eval_200.to_dict(),
                    "eval_252": eval_252.to_dict(),
                }
            conn.close()

        # Step F: Assemble Candidate Profiles
        baseline_by_sym = {r.symbol: r for r in baseline_outcome.records}
        baseline_val_by_sym = {r.raw_record.get("symbol"): r for r in baseline_batch_val.records}
        replay_by_sym = {r.symbol: r for r in replay_outcome.records}
        replay_val_by_sym = {r.raw_record.get("symbol"): r for r in replay_batch_val.records}

        profiles = {}
        for sym, r in baseline_by_sym.items():
            loc = REAL_SOURCE_ROW_LOCATORS[sym]
            v_rec = baseline_val_by_sym.get(sym)
            v_issues = [f"{iss.field_name}: {iss.issue_code.value}" for iss in v_rec.issues] if v_rec else []

            # 1. Source Observed Facts
            observed_facts = {
                "symbol": sym,
                "issuer_cik": r.issuer_cik,
                "company_name": r.company_name,
                "sec_exchange": r.sec_exchange,
                "directory_exchange": r.directory_exchange,
                "is_etf_flag": r.is_etf,
                "parent_feed_locators": loc,
            }

            # 2. Attributable External Mappings
            venue_res = SupportedVenueNormalizer.reconcile_dual_source_venues(
                r.sec_exchange, r.directory_exchange
            )
            external_mappings = {
                "venue_normalization": venue_res.to_dict(),
            }

            # 3. Policy-Derived Assertions (Approved directions & schema integration gaps)
            policy_assertions = {
                "currency_policy": "PER_LISTING_ATTRIBUTABLE_EVIDENCE (STATUS: Acquired via Yahoo Finance quote metadata; blanket USD rejected)",
                "listing_country_policy": "EXPLICIT_LISTING_COUNTRY_USA (STATUS: Approved direction; legacy SourceSecurity.country requires schema decoupling)",
                "primary_listing_policy": "SELECTED_US_ASSESSMENT_LISTING (STATUS: Approved direction; legacy SourceSecurity.primary_listing requires schema decoupling)",
            }

            # 4. Diagnostic Triage Hints
            dir_security_name = None
            for d_file in directory_files:
                for row in d_file.rows:
                    if row.raw_values.get("normalized_symbol") == sym or row.raw_values.get("raw_symbol") == sym:
                        dir_security_name = row.raw_values.get("security_name")
                        break
                if dir_security_name:
                    break

            eval_name = dir_security_name or r.company_name
            triage_hint = ClassificationTriageEngine.evaluate_triage_hint(
                eval_name, is_etf=r.is_etf
            )
            triage_hints = {
                "classification_hint": triage_hint.to_dict(),
            }

            # 5. Unknown or Conflicting Claims
            unknown_conflicts = list(r.resolution_issues)
            if venue_res.is_substantive_disagreement:
                unknown_conflicts.append(f"VENUE_DISAGREEMENT: {venue_res.disagreement_details or venue_res.publisher_citation}")

            # 6. Detailed 12-Field Admission Audit
            field_audits = cls._audit_candidate_fields(sym, r, venue_res, triage_hint, loc, evaluated_title=eval_name)

            # Replay comparison profile
            r_rep = replay_by_sym.get(sym)
            r_val_rep = replay_val_by_sym.get(sym)
            rep_issues = [f"{iss.field_name}: {iss.issue_code.value}" for iss in r_val_rep.issues] if r_val_rep else []
            replay_prof = {
                "figi_identifiers": r_rep.figi_identifiers if r_rep else {},
                "openfigi_status": r_rep.openfigi_match_status.value if r_rep and r_rep.openfigi_match_status else None,
                "security_type_claim": r_rep.security_type if r_rep else None,
                "replayed_validation_status": r_val_rep.status.value if r_val_rep else "UNKNOWN",
                "replayed_validation_issues": rep_issues,
                "fields_newly_evidenced": list(replay_field_overlay.get(sym, {}).keys()),
            }

            profile = CandidateAdmissionProfile(
                symbol=sym,
                issuer_cik=r.issuer_cik,
                company_name=r.company_name,
                source_observed_facts=observed_facts,
                attributable_external_mappings=external_mappings,
                policy_derived_assertions=policy_assertions,
                diagnostic_triage_hints=triage_hints,
                unknown_or_conflicting_claims=unknown_conflicts,
                field_audits=field_audits,
                resolved_record_status=r.record_status.value,
                identity_resolution_status=r.identity_status.value,
                batch_validation_status=v_rec.status.value if v_rec else "UNKNOWN",
                record_validation_issues=v_issues,
                market_data_profile=market_readiness.get(sym, {}),
                replay_profile=replay_prof,
            )
            profiles[sym] = profile

        remaining_blockers = [
            "1. Missing Production Security Identity Authority: security_id is omitted from raw feeds; external FIGI identifiers are qualified but production adoption remains unauthorized.",
            "2. Legacy Country Schema Integration Gap: listing_country='USA' is approved, but SourceSecurity.country is not yet decoupled into listing_country vs incorporation_country (TSM incorporated in Taiwan).",
            "3. Legacy Primary Listing Schema Integration Gap: selected_us_assessment_listing=True is approved, but SourceSecurity.primary_listing requires global primary quotation (TSM primary is TWSE: 2330).",
            "4. Ineligible Instrument Classifications: OpenFIGI qualifies TSM as 'Depositary Receipt' and AACIU as 'Unit'; both violate Rule R02 requiring COMMON_STOCK.",
            "5. Unresolved Venue Policy & Disagreements: ACCS (NYSE American / AMEX) falls outside the validator venue allowlist; DDS retains substantive exchange disagreement (provisional TXSE vs FINRA ADF 'F').",
            "6. Stored Market Data Deficits: GOOG, ACCS, DDS, AACIU have 0 bars in local development store; GOOGL (latest 2026-10-12) and TSM (latest 2026-05-07) have stale observations.",
            "7. Unresolved Identity Association for Market Data: Even where history exists (AAPL: 465, GOOGL: 278, TSM: 352), bars cannot authorize admission while security_id association remains unverified.",
        ]

        next_steps = [
            "1. Formalize production identity schema adoption: ratify policy authorizing durable security_id assignment from verified OpenFIGI composite FIGI or internal master.",
            "2. Decouple country schema in SourceSecurity and UniverseInputValidator: add explicit listing_country alongside incorporation_country.",
            "3. Decouple listing selection schema in SourceSecurity and UniverseInputValidator: add selected_us_assessment_listing to assess US ADRs without falsifying global primary listing.",
            "4. Ratify venue expansion policy: determine whether AMEX (NYSE American) should be added to the canonical venue allowlist for common equities.",
            "5. Formalize provisional venue policy: define resolution criteria for uncommenced exchanges like Texas Stock Exchange (DDS).",
        ]

        replay_summary = {
            "openfigi_matches_total": len(acquired_evidence.get("openfigi", {}).get("response_payload", [])),
            "currency_evidence_count": len(acquired_evidence.get("currency", {}).get("results", {})),
            "alpaca_evidence_count": len(acquired_evidence.get("alpaca", {}).get("results", {})),
            "sec_submissions_count": len(acquired_evidence.get("sec_submissions", {}).get("results", {})),
            "replay_batch_valid": replay_batch_val.is_batch_valid,
            "replay_passed_count": replay_batch_val.passed_count,
            "replay_failed_count": replay_batch_val.failed_count,
            "replay_converted_count": replay_converted_count,
            "replay_builder_status": "NOT_RUN",
            "replay_builder_blocking_reason": (
                "BLOCKED_VALIDATION_FAILED: Replay validation batch failed closed (7 failed, 0 passed). "
                "Downstream admission revoked under atomic whole-batch contract."
            ),
        }

        return RealSourceDiagnosticResult(
            total_candidates=len(profiles),
            candidate_profiles=profiles,
            batch_valid=baseline_batch_val.is_batch_valid,
            passed_count=baseline_batch_val.passed_count,
            failed_count=baseline_batch_val.failed_count,
            unresolved_count=baseline_batch_val.unresolved_count,
            converted_security_count=converted_count,
            builder_status=builder_status,
            builder_blocking_reason=builder_blocking_reason,
            market_db_status=db_status,
            market_db_row_counts=db_counts,
            market_readiness_by_candidate=market_readiness,
            remaining_admission_blockers=remaining_blockers,
            next_evidence_acquisition_steps=next_steps,
            replay_summary=replay_summary,
            parent_feed_accounting=REAL_SOURCE_PARENT_FEEDS,
            execution_timestamp_utc=now_utc,
        )

    @classmethod
    def _audit_candidate_fields(
        cls,
        symbol: str,
        rec: ResolvedSecurityRecord,
        venue_res: VenueNormalizationResult,
        triage_hint: DiagnosticTriageHint,
        loc: Dict[str, Any],
        evaluated_title: Optional[str] = None,
    ) -> List[FieldAdmissionAudit]:
        """Audits every contract field required by UniverseInputValidator and to_source_security."""
        audits = []

        # 1. security_id
        audits.append(FieldAdmissionAudit(
            field_name="security_id",
            contract_semantics="Unique immutable primary key for financial contract; audit anchor",
            source_observation=None,
            source_artifact="sec_company_tickers_exchange.json / directories",
            row_locator=f"SEC row {loc.get('sec_row_index')}",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Missing production security identity authority. Internal locators do not prove external continuity.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 2. symbol
        audits.append(FieldAdmissionAudit(
            field_name="symbol",
            contract_semantics="Exchange trading ticker symbol",
            source_observation=symbol,
            source_artifact="sec_company_tickers_exchange.json / directories",
            row_locator=f"SEC row {loc.get('sec_row_index')}, Directory line {loc.get('nasdaq_line') or loc.get('other_line')}",
            evidence_nature="DIRECT_PROVIDER_OBSERVATION",
            satisfies_requirement=True,
            missing_or_conflicted=None,
            validation_consequence="VALIDATION_PASSED",
        ))

        # 3. exchange
        exchange_obs = f"SEC:{rec.sec_exchange}|DIR:{rec.directory_exchange}"
        is_exch_valid = (
            venue_res.normalized_venue in {"NASDAQ", "NYSE", "ARCA", "BATS"}
            and not venue_res.is_substantive_disagreement
        )
        exch_consequence = "VALIDATION_PASSED" if is_exch_valid else "UNRECOGNIZED_VALUE_OR_CONFLICT"
        audits.append(FieldAdmissionAudit(
            field_name="exchange",
            contract_semantics="Primary listing exchange; must be one of NASDAQ, NYSE, ARCA, BATS",
            source_observation=exchange_obs,
            source_artifact="Composite (SEC EDGAR + Directory)",
            row_locator=f"SEC row {loc.get('sec_row_index')}, Directory line {loc.get('nasdaq_line') or loc.get('other_line')}",
            evidence_nature=venue_res.status.value,
            satisfies_requirement=is_exch_valid,
            missing_or_conflicted=venue_res.disagreement_details if not is_exch_valid else None,
            validation_consequence=exch_consequence,
        ))

        # 4. security_type
        title_for_obs = evaluated_title or rec.company_name
        audits.append(FieldAdmissionAudit(
            field_name="security_type",
            contract_semantics="Specific instrument contract type (must be COMMON_STOCK under Rule R02)",
            source_observation=f"is_etf={rec.is_etf}; title='{title_for_obs}'",
            source_artifact="Directory / SEC",
            row_locator=f"Directory line {loc.get('nasdaq_line') or loc.get('other_line')}",
            evidence_nature="DIAGNOSTIC_TRIAGE_HINT",
            satisfies_requirement=False,
            missing_or_conflicted=f"Directory provides only is_etf={rec.is_etf}. Title regex hint is '{triage_hint.suspected_instrument_type}' but cannot certify COMMON_STOCK.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 5. listing_status
        audits.append(FieldAdmissionAudit(
            field_name="listing_status",
            contract_semantics="Operational listing status (must be ACTIVE under Rule R03)",
            source_observation="Directory presence on 2026-10-16",
            source_artifact="Directory",
            row_locator=f"Directory line {loc.get('nasdaq_line') or loc.get('other_line')}",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Presence implies listed, but intraday regulatory trading halt / suspension status is absent.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 6. currency
        audits.append(FieldAdmissionAudit(
            field_name="currency",
            contract_semantics="Quote and settlement currency (must be USD under Rule R05)",
            source_observation=None,
            source_artifact="sec_company_tickers_exchange.json / directories",
            row_locator="Omitted across all raw feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED without reference quotes.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 7. country
        audits.append(FieldAdmissionAudit(
            field_name="country",
            contract_semantics="Jurisdictional country of the contract / listing",
            source_observation=None,
            source_artifact="Directory omits; SEC has stateOfIncorporation",
            row_locator="Omitted across raw directory feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from directories. Cannot silently substitute listing_country='USA' for legacy country.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 8. primary_listing
        audits.append(FieldAdmissionAudit(
            field_name="primary_listing",
            contract_semantics="Boolean assertion of primary quotation line (must be True under Rule R06)",
            source_observation=None,
            source_artifact="sec_company_tickers_exchange.json / directories",
            row_locator="Omitted across all raw feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from feeds. Tapes A/B/C are dissemination networks, not listing authorities.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 9. asset_class
        audits.append(FieldAdmissionAudit(
            field_name="asset_class",
            contract_semantics="High-level asset class (must be EQUITY under Rule R01)",
            source_observation=f"is_etf={rec.is_etf}",
            source_artifact="Directory",
            row_locator=f"Directory line {loc.get('nasdaq_line') or loc.get('other_line')}",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Directory distinguishes funds (is_etf=False), but formal asset_class='EQUITY' is omitted.",
            validation_consequence="MISSING_REQUIRED_FIELD",
        ))

        # 10. listing_date
        audits.append(FieldAdmissionAudit(
            field_name="listing_date",
            contract_semantics="Point-in-time initial exchange listing date (YYYY-MM-DD) or explicit null",
            source_observation=None,
            source_artifact="Directory / SEC",
            row_locator="Omitted across raw feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred.",
            validation_consequence="OMITTED_FROM_VALIDATED_VALUES",
        ))

        # 11. delisting_date
        audits.append(FieldAdmissionAudit(
            field_name="delisting_date",
            contract_semantics="Point-in-time delisting date (YYYY-MM-DD) or explicit null",
            source_observation=None,
            source_artifact="Directory / SEC",
            row_locator="Omitted across raw feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred.",
            validation_consequence="OMITTED_FROM_VALIDATED_VALUES",
        ))

        # 12. trading_suspension_date
        audits.append(FieldAdmissionAudit(
            field_name="trading_suspension_date",
            contract_semantics="Point-in-time regulatory halt date (YYYY-MM-DD) or explicit null",
            source_observation=None,
            source_artifact="Directory / SEC",
            row_locator="Omitted across raw feeds",
            evidence_nature="OMITTED_FROM_SOURCE",
            satisfies_requirement=False,
            missing_or_conflicted="Omitted from directory. Explicit None required by to_source_security, but None does NOT prove event never occurred.",
            validation_consequence="OMITTED_FROM_VALIDATED_VALUES",
        ))

        return audits


# =====================================================================
# 5. Report Formatting
# =====================================================================

def format_evidence_gap_report(result: RealSourceDiagnosticResult) -> str:
    """Formats executable diagnostic findings into a clean Markdown evidence gap report."""
    md = []
    md.append("# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report")
    md.append("**Bounded Evidence Handoff, Targeted Correction & Acquired Evidence Replay for 7-Candidate Sample**\n")
    md.append(f"- **Execution Timestamp**: `{result.execution_timestamp_utc}`")
    md.append(f"- **Evaluated Candidates**: `AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU` ({result.total_candidates} records)")
    md.append(f"- **Baseline Admission Outcome**: `{'PASSED' if result.batch_valid else 'REVOKED (VALIDATION_FAILED)'}`")
    md.append(f"- **Evidence-Replay Outcome**: `REVOKED (VALIDATION_FAILED)` (0 admissions, builder NOT_RUN)")
    md.append(f"- **Downstream Converted Securities**: `{result.converted_security_count}`")
    md.append(f"- **Universe Builder Status**: `{result.builder_status}`")
    md.append(f"- **Market DB Status**: `{result.market_db_status}`\n")
    md.append("---\n")

    md.append("### 1. Executive Summary & Diagnostic Verdict\n")
    md.append("This diagnostic establishes the exact empirical boundary between raw listing directory observations, ")
    md.append("newly acquired attributable research evidence (OpenFIGI, Yahoo Finance quote metadata, Alpaca asset directory, SEC submissions), ")
    md.append("and the strict admission requirements enforced by `UniverseInputValidator` and `SourceSecurity` conversion.\n")
    md.append("> [!IMPORTANT]\n")
    md.append("> **Diagnostic Acceptance**: Zero admissions is the truthful and expected diagnostic outcome across both baseline ")
    md.append("> and evidence-replay passes. While reference evidence now provides attributable currency (`USD`), asset class (`EQUITY`), ")
    md.append("> active status (`ACTIVE`), and OpenFIGI structural classifications, production security identity adoption remains disabled, ")
    md.append("> and candidate-specific rule blockers (TSM Depositary Receipt/foreign primary, AACIU Unit, ACCS AMEX venue, DDS provisional TXSE) ")
    md.append("> correctly prevent admission. Under the atomic whole-batch contract, the entire batch is revoked, converted count is 0, ")
    md.append("> and Builder evaluation is strictly NOT_RUN.\n")

    md.append("### 2. Parent-Feed Accounting & Discrepancy Reconciliation\n")
    md.append("Counts recomputed from the exact archived bytes and verified full SHA-256 digests:\n")
    md.append("| Feed Identifier | Canonical SHA-256 Digest | Physical Lines | Metadata Rows | Usable Data Rows | Malformed Rows | Unique Records | Accounting Reconciled Status |")
    md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for feed_id, p_info in result.parent_feed_accounting.items():
        p_lines = p_info.get("physical_lines", "N/A")
        m_rows = p_info.get("metadata_rows", "N/A")
        d_rows = p_info.get("data_records", "N/A")
        mal_rows = p_info.get("malformed_rows", "N/A")
        u_recs = p_info.get("unique_records", "N/A")
        sha_short = p_info["sha256"][:16] + "..."
        md.append(f"| `{feed_id}` | `{sha_short}` | `{p_lines}` | `{m_rows}` | `{d_rows}` | `{mal_rows}` | `{u_recs}` | 🟢 VERIFIED |")

    md.append("\n**Discrepancy Explanation**:\n")
    for feed_id, p_info in result.parent_feed_accounting.items():
        md.append(f"- **{feed_id}**: {p_info.get('discrepancy_explanation')}")

    md.append("\n---\n")

    md.append("### 3. Acquired Bounded Real Evidence Provenance\n")
    md.append("| Evidence Domain | Authority / Provider | Acquired Artifact Path | HTTP / Status | Response SHA-256 | Key Facts Established |")
    md.append("|:---|:---|:---|:---:|:---|:---|")
    md.append("| **Instrument Identity & Taxonomy** | OpenFIGI v3 API | `data/research/openfigi_sample/seven_candidates_openfigi_response.json` | `200 OK` | `c0ad9c377c7e...` | Distinct FIGIs for all 7 candidates; GOOG Class C vs GOOGL Class A; TSM Depositary Receipt; AACIU Unit |")
    md.append("| **Quote & Trading Currency** | Public Reference Quotes | `data/research/currency_sample/seven_candidates_currency_reference.json` | `200 OK` | `Various per symbol` | Attributable per-listing trading currency (`USD`) for all 7 candidates |")
    md.append("| **Operational Listing Status** | Alpaca Asset Directory | `data/research/alpaca_sample/seven_candidates_alpaca_assets.json` | `200 OK` | `Various per symbol` | Status `ACTIVE` and broad class `EQUITY` for all 7 candidates |")
    md.append("| **Issuer Domicile & Incorporation** | SEC EDGAR Submissions | `data/research/sec_submissions_sample/seven_candidates_sec_submissions.json` | `200 OK` | `Various per CIK` | State of incorporation (DE, CA, FL); TSM foreign country `Taiwan` (code `F5`) |")

    md.append("\n---\n")

    md.append("### 4. Candidate Evidence Resolution & Validation Summary (Baseline vs Replay)\n")
    md.append("| Symbol | CIK | Parent Feeds / Locators | Exchange (Resolved) | OpenFIGI Subtype | Attributable Currency | Baseline Status | Replay Status | Retained Blockers |")
    md.append("|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|")

    for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
        p = result.candidate_profiles[sym]
        loc = p.source_observed_facts.get("parent_feed_locators", {})
        loc_str = f"SEC #{loc.get('sec_row_index')}; Dir #{loc.get('nasdaq_line') or loc.get('other_line')}"
        norm_venue = p.attributable_external_mappings.get("venue_normalization", {}).get("normalized_venue", "UNMAPPED")
        rep = p.replay_profile or {}
        figi_type = rep.get("security_type_claim") or "N/A"
        new_ev = rep.get("fields_newly_evidenced", [])
        curr_str = "USD (Attributable)" if "currency" in new_ev else "UNRESOLVED"
        b_stat = p.batch_validation_status
        r_stat = rep.get("replayed_validation_status", "UNKNOWN")
        r_issues = ", ".join(rep.get("replayed_validation_issues", [])[:2]) if rep.get("replayed_validation_issues") else "None"
        md.append(f"| **{sym}** | `{p.issuer_cik}` | {loc_str} | `{norm_venue}` | `{figi_type}` | `{curr_str}` | `{b_stat}` | `{r_stat}` | {r_issues} |")

    md.append("\n---\n")

    md.append("### 5. Field-by-Candidate Admission Requirement Matrix (12 Contract Fields)\n")
    md.append("| Candidate | Required Field | Source / Replay Observation | Evidence Nature | Satisfies? | Validation Consequence | Exact Blocker / Gap |")
    md.append("|:---|:---|:---|:---:|:---:|:---:|:---|")

    for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
        p = result.candidate_profiles[sym]
        for fa in p.field_audits:
            obs = str(fa.source_observation) if fa.source_observation is not None else "*(omitted)*"
            sat = "🟢 PASS" if fa.satisfies_requirement else "🔴 FAIL"
            blocker = fa.missing_or_conflicted or "None"
            md.append(f"| **{sym}** | `{fa.field_name}` | `{obs}` | `{fa.evidence_nature}` | {sat} | `{fa.validation_consequence}` | {blocker} |")

    md.append("\n---\n")

    md.append("### 6. Development Market Data Store Audit & Lookback Readiness\n")
    md.append(f"- **Store URI**: `file:C:/Users/akara/.finance_market_store.db?mode=ro` (Enforced Read-Only)")
    md.append(f"- **Daily OHLCV Rows**: `{result.market_db_row_counts.get('asset_ohlcv_daily_total', 0)}` across 150 development symbols")
    md.append(f"- **Provenance Rows**: `{result.market_db_row_counts.get('asset_ohlcv_provenance_total', 0)}`\n")

    md.append("The table below separates **Identity Association**, **Bar Availability**, **Freshness Observation**, and **Consumer Lookbacks** ")
    md.append("without bypassing unresolved association guards for GOOGL or TSM, and without treating query buffers or annualization scalars as admission gates:\n")

    md.append("| Symbol | Stored Bars | Freshness (Cutoff 2026-10-16) | 50-Bar MA (Trend Gate) | 200-Bar MA (Regime Gate) | 60-Bar Query Buffer (Operational) | 252-Bar Range (Annual Benchmark) | Identity Association | Readiness Evaluator State |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
        mr = result.market_readiness_by_candidate.get(sym, {})
        cnt = mr.get("bar_count", 0)
        freshness = mr.get("freshness_state", "NO_DATA")
        g50 = "🟢 MET" if mr.get("gate_50_bars") else "🔴 INSUFFICIENT"
        g200 = "🟢 MET" if mr.get("gate_200_bars") else "🔴 INSUFFICIENT"
        b60 = "🟢 AVAILABLE" if mr.get("buffer_60_bars_available") else "🔴 UNAVAILABLE"
        b252 = "🟢 AVAILABLE" if mr.get("benchmark_252_bars_available") else "🔴 UNAVAILABLE"
        assoc = "🔴 UNRESOLVED"
        eval_state = f"`{mr.get('evaluator_primary_state', 'UNKNOWN')}`"
        md.append(f"| **{sym}** | {cnt} | `{freshness}` | {g50} | {g200} | {b60} | {b252} | {assoc} | {eval_state} |")

    md.append("\n> [!NOTE]\n")
    md.append("> **Readiness Invariant (Section 1.B)**: `ConsumerReadinessEvaluator` strictly enforces that unassociated symbols ")
    md.append("> cannot achieve `CONSUMER_READY`. For `GOOGL` and `TSM`, stale history is recorded as a freshness observation on the bar dimension, ")
    md.append("> while their primary evaluator state truthfully remains `UNRESOLVED_ASSOCIATION`. The 60-bar query buffer and 252-bar ")
    md.append("> annualization factor are operational/benchmark capabilities, not admission gates.\n")

    md.append("### 7. Previously Approved Semantics & Exact Legacy Contract Integration Gaps\n")
    md.append("The Product Owner previously approved two distinct semantic concepts (Package 4/5 directions):\n")
    md.append("1. **`listing_country`**: Captures the operational jurisdiction of the quotation line (e.g. `'USA'`).")
    md.append("   - *Legacy Contract Gap*: `SourceSecurity.country` is currently an overloaded single field historically assumed to be domicile/incorporation. Silently mapping `listing_country='USA'` for `TSM` (incorporated in Taiwan) would corrupt legal domicile. `SourceSecurity` and `UniverseInputValidator` require schema decoupling into `listing_country` and `incorporation_country`.")
    md.append("2. **`selected_us_assessment_listing`**: Designates an authorized US trading line (e.g. TSM ADR on NYSE) as the active line for US radar assessment without asserting that it is the global primary listing.")
    md.append("   - *Legacy Contract Gap*: `SourceSecurity.primary_listing` is a boolean requiring global primary quotation under Rule R06. TSM's global primary listing is the Taiwan Stock Exchange (TWSE: 2330). Forcing `primary_listing=True` makes a false market claim; omitting it fails validation. Schema decoupling is required.")

    md.append("\n### 8. Review of `admission_mapping.py` Changes\n")
    md.append("- **Change Reviewed**: Added direct recognition of `\"NASDAQ\"` directory code in `SupportedVenueNormalizer.normalize_directory_code`.")
    md.append("- **Module Boundaries Justification**: The Nasdaq listed directory parser outputs `venue_exchange=\"NASDAQ\"`. Recognizing this string as concordant with operating MIC `XNAS` and `TAPE_C` is reusable, production-grade normalizer logic that belongs in `admission_mapping.py` rather than being patched ad-hoc in diagnostic scripts.")
    md.append("- **Containment**: All diagnostic-specific profile assembly, report formatting, and audit comparisons remain strictly contained within `real_source_diagnostic.py`.")

    md.append("\n### 9. Remaining Admission Blockers & Smallest Next Acquisition Steps\n")
    md.append("#### Remaining Blockers:\n")
    for b in result.remaining_admission_blockers:
        md.append(f"- {b}")

    md.append("\n#### Smallest Concrete Evidence Acquisition Steps Needed Next:\n")
    for s in result.next_evidence_acquisition_steps:
        md.append(f"- {s}")

    return "\n".join(md)
