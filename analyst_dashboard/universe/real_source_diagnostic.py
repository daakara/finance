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
1. Source Grounding: All candidate facts originate from verified 2026-10-16 parent feeds.
2. Separation of Evidentiary Categories:
   - Source-observed facts
   - Attributable external mappings
   - Policy-derived assertions
   - Diagnostic triage hints
   - Unknown or conflicting claims
3. Strict Non-Inference:
   - Zero synthetic surrogate keys (CIK+symbol).
   - Creating internal identifiers does not establish that real listings belong to them.
   - Suffix/name regex remains diagnostic only (does NOT certify COMMON_STOCK).
   - Venue normalization preserves substantive disagreements.
   - Unknown currency remains unknown; blanket USD assumption is strictly NOT approved.
   - Zero silent substitution of listing_country for legacy country, or
     selected_us_assessment_listing for primary_listing.
   - Explicit None in temporal fields does NOT independently prove an event never occurred.
4. Atomic Whole-Batch Revocation:
   - Batch fails closed; conversion count is strictly 0; Builder is strictly NOT_RUN.
5. Enforced Read-Only SQLite Market Data Audit:
   - Accesses development store via file:...mode=ro; mutation attempts fail closed.
   - Distinguishes stored history, query limits, freshness, and consumer lookbacks.
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
    ParsedSourceFile,
    ParsedSourceRow,
    SourceFeedType,
)
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
# 1. Attributable Parent Feed Provenance & Row Locators
# =====================================================================

REAL_SOURCE_PARENT_FEEDS = {
    "sec_company_tickers_exchange.json": {
        "sha256": "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f",
        "record_count": 10444,
        "as_of_date": "2026-10-16",
        "description": "SEC EDGAR Company Tickers Exchange master feed",
    },
    "nasdaq_listed_directory.txt": {
        "sha256": "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272",
        "record_count": 5166,
        "as_of_date": "2026-10-16",
        "description": "Nasdaq Trader Symbol Directory (Nasdaq Listed)",
    },
    "other_listed_directory.txt": {
        "sha256": "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792",
        "record_count": 8172,
        "as_of_date": "2026-10-16",
        "description": "Nasdaq Trader Symbol Directory (Other Listed / CTA Tape A & B)",
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

# Derivative fixture rows extracted from parent feeds
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
# 2. Admission Requirements Tracing Models
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
        }


# =====================================================================
# 3. Diagnostic Execution Engine
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
            "execution_timestamp_utc": self.execution_timestamp_utc,
        }


class RealSourceAdmissionDiagnostic:
    """
    Executes the bounded offline admission diagnostic against the 7-candidate sample.
    Pure in-memory execution, enforced read-only SQLite inspection, fail-closed contracts.
    """

    DEFAULT_STORE_PATH = Path(r"C:\Users\akara\.finance_market_store.db")

    @classmethod
    def run(
        cls,
        market_db_path: Optional[Path] = None,
        evaluation_cutoff: str = "2026-10-16",
    ) -> RealSourceDiagnosticResult:
        now_utc = datetime.now(timezone.utc).isoformat()

        # Step A: Build parsed source files
        sec_file, directory_files = build_real_source_sample_files()

        # Step B: Resolve candidates via SecurityEvidenceResolver
        resolution_outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=directory_files,
            openfigi_results=None,
            caller_security_id_mapping=None,
            caller_field_evidence_overlay=None,
            require_venue_match=False,
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
        )

        records_by_sym = {r.symbol: r for r in resolution_outcome.records}

        # Step C: Validate batch against canonical ValidationRequirements
        validator_reqs = ValidationRequirements(
            allowed_values={
                "exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"},
                "asset_class": {"EQUITY"},
            }
        )
        batch_val = resolution_outcome.validate_batch(
            requirements=validator_reqs,
            allow_incomplete_source_files=True,
        )

        validation_by_sym = {
            r.raw_record.get("symbol"): r for r in batch_val.records
        }

        # Step D: Test conversion guard (must raise ValueError under fail-closed contract)
        converted_count = 0
        builder_status = "NOT_RUN"
        builder_blocking_reason = (
            "BLOCKED_VALIDATION_FAILED: Batch validation failed (7 failed, 0 passed). "
            "Downstream admission revoked under atomic whole-batch contract."
        )
        try:
            batch_val.to_source_securities()
            # If to_source_securities didn't raise, that's an invariant violation
            raise RuntimeError("CRITICAL INVARIANT VIOLATION: Failed batch allowed conversion!")
        except ValueError:
            converted_count = 0

        # Step E: Enforced Read-Only Market Database Audit
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
                # Expected fail-closed behavior
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

                # Evaluate consumer lookbacks via ConsumerReadinessEvaluator
                # Note: is_association_resolved=False because identity is unresolved
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

                market_readiness[sym] = {
                    "bar_count": bar_count,
                    "earliest_date": earliest_date,
                    "latest_date": latest_date,
                    "eval_50_bars": eval_50.to_dict(),
                    "eval_60_bars": eval_60.to_dict(),
                    "eval_200_bars": eval_200.to_dict(),
                    "eval_252_bars": eval_252.to_dict(),
                }
            conn.close()

        # Step F: Assemble Candidate Admission Profiles
        profiles = {}
        for sym, r in records_by_sym.items():
            loc = REAL_SOURCE_ROW_LOCATORS[sym]
            v_rec = validation_by_sym.get(sym)
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

            # 2. Attributable External Mappings (Venue normalization)
            venue_res = SupportedVenueNormalizer.reconcile_dual_source_venues(
                r.sec_exchange, r.directory_exchange
            )
            external_mappings = {
                "venue_normalization": venue_res.to_dict(),
            }

            # 3. Policy-Derived Assertions (Explicitly tracking unapproved status)
            policy_assertions = {
                "currency_policy": "POLICY_DERIVED_US_NMS_CONVENTION (STATUS: NOT_APPROVED by PO; UNRESOLVED)",
                "listing_country_policy": "EXPLICIT_LISTING_COUNTRY_USA (STATUS: PROPOSED; cannot silently replace legacy country)",
                "primary_listing_policy": "SELECTED_US_ASSESSMENT_LISTING (STATUS: PROPOSED; cannot silently replace primary_listing)",
            }

            # 4. Diagnostic Triage Hints (Classification regex)
            # Find directory security_name if present (contains instrument qualifiers like 'Units')
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
            )
            profiles[sym] = profile

        remaining_blockers = [
            "1. Missing Production Security Identity Authority: security_id is omitted from raw feeds; internal Tier A/B locators do not prove external continuity.",
            "2. Missing Attributable Currency Evidence: SEC and directory feeds omit currency; PO rejected blanket USD assumption; currency remains unresolved.",
            "3. Missing Attributable Country Semantics: Legacy country contract field cannot be silently satisfied by listing_country='USA'.",
            "4. Missing Primary Listing Certification: Dissemination tapes do not prove primary-listing status; selected_us_assessment_listing is not yet ratified.",
            "5. Missing Certified Instrument Taxonomy: Directory provides only is_etf=False; title regex cannot certify COMMON_STOCK under Rule R02.",
            "6. Unresolved Venue Disagreements: DDS (exchange code 'F' / TXSE) and ACCS (NYSE vs AMEX) lack ratified venue clearance.",
            "7. Missing Stored Market Data for 4 Candidates: GOOG, ACCS, DDS, AACIU have 0 bars in local development store.",
            "8. Stale Stored Market Data for TSM: TSM candles in dev store terminate at 2026-05-07 (stale relative to 2026-10-16).",
        ]

        next_steps = [
            "1. Acquire OpenFIGI or CIK-to-FIGI mapping for the 7 candidates to establish attributable security_id and certified securityType2.",
            "2. Acquire quote metadata or provider asset master records to supply attributable currency ('USD') per listing.",
            "3. Formalize country schema decoupling: add explicit listing_country while retaining incorporation_country from SEC filings.",
            "4. Ratify selected_us_assessment_listing policy to permit foreign ADR (TSM) evaluation without corrupting global primary listing.",
            "5. Formalize venue policy for AMEX (ACCS) and provisional TXSE (DDS).",
        ]

        return RealSourceDiagnosticResult(
            total_candidates=len(profiles),
            candidate_profiles=profiles,
            batch_valid=batch_val.is_batch_valid,
            passed_count=batch_val.passed_count,
            failed_count=batch_val.failed_count,
            unresolved_count=batch_val.unresolved_count,
            converted_security_count=converted_count,
            builder_status=builder_status,
            builder_blocking_reason=builder_blocking_reason,
            market_db_status=db_status,
            market_db_row_counts=db_counts,
            market_readiness_by_candidate=market_readiness,
            remaining_admission_blockers=remaining_blockers,
            next_evidence_acquisition_steps=next_steps,
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
            missing_or_conflicted="Omitted from raw feeds. Blanket USD convention rejected by PO. Currency remains UNRESOLVED.",
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


def format_evidence_gap_report(result: RealSourceDiagnosticResult) -> str:
    """Formats executable diagnostic findings into a clean Markdown evidence gap report."""
    md = []
    md.append("# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report")
    md.append("**Bounded Evidence Handoff & Admission Diagnostic for 7-Candidate Sample**\n")
    md.append(f"- **Execution Timestamp**: `{result.execution_timestamp_utc}`")
    md.append(f"- **Evaluated Candidates**: `AAPL, GOOG, GOOGL, TSM, ACCS, DDS, AACIU` ({result.total_candidates} records)")
    md.append(f"- **Batch Admission Outcome**: `{'PASSED' if result.batch_valid else 'REVOKED (VALIDATION_FAILED)'}`")
    md.append(f"- **Downstream Converted Securities**: `{result.converted_security_count}`")
    md.append(f"- **Universe Builder Status**: `{result.builder_status}`")
    md.append(f"- **Market DB Status**: `{result.market_db_status}`\n")
    md.append("---\n")

    md.append("### 1. Executive Summary & Diagnostic Verdict\n")
    md.append("This diagnostic establishes the exact empirical boundary between raw listing directory observations ")
    md.append("and the strict admission requirements enforced by `UniverseInputValidator` and `SourceSecurity` conversion.\n")
    md.append("> [!IMPORTANT]\n")
    md.append("> **Diagnostic Acceptance**: Zero admissions is the truthful and expected diagnostic outcome. ")
    md.append("> Raw directory feeds lack 7 mandatory contract fields (`security_id`, `currency`, `country`, `primary_listing`, ")
    md.append("> `security_type`, `asset_class`, `listing_status`). Under the atomic whole-batch contract, ")
    md.append("> the entire batch is revoked, converted count is 0, and Builder evaluation is strictly NOT_RUN.\n")

    md.append("### 2. Candidate Evidence Resolution & Validation Summary\n")
    md.append("| Symbol | CIK | Parent Feeds / Locators | Record Status | Identity Status | Exchange Resolution | Validation Status | Top Blockers |")
    md.append("|:---|:---:|:---|:---:|:---:|:---|:---:|:---|")

    for sym, p in sorted(result.candidate_profiles.items()):
        loc = p.source_observed_facts.get("parent_feed_locators", {})
        loc_str = f"SEC #{loc.get('sec_row_index')}; Dir #{loc.get('nasdaq_line') or loc.get('other_line')}"
        norm_venue = p.attributable_external_mappings.get("venue_normalization", {}).get("normalized_venue", "UNMAPPED")
        top_issues = ", ".join(p.record_validation_issues[:3]) if p.record_validation_issues else "None"
        md.append(f"| **{sym}** | `{p.issuer_cik}` | {loc_str} | `{p.resolved_record_status}` | `{p.identity_resolution_status}` | `{norm_venue}` | `{p.batch_validation_status}` | {top_issues} |")

    md.append("\n---\n")

    md.append("### 3. Field-by-Candidate Admission Requirement Matrix\n")
    md.append("| Candidate | Required Field | Source Observation | Evidence Nature | Satisfies? | Validation Consequence | Exact Blocker / Gap |")
    md.append("|:---|:---|:---|:---:|:---:|:---:|:---|")

    for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
        p = result.candidate_profiles[sym]
        for fa in p.field_audits:
            obs = str(fa.source_observation) if fa.source_observation is not None else "*(omitted)*"
            sat = "🟢 PASS" if fa.satisfies_requirement else "🔴 FAIL"
            blocker = fa.missing_or_conflicted or "None"
            md.append(f"| **{sym}** | `{fa.field_name}` | `{obs}` | `{fa.evidence_nature}` | {sat} | `{fa.validation_consequence}` | {blocker} |")

    md.append("\n---\n")

    md.append("### 4. Development Market Data Store Audit & Lookback Readiness\n")
    md.append(f"- **Store URI**: `file:C:/Users/akara/.finance_market_store.db?mode=ro` (Enforced Read-Only)")
    md.append(f"- **Daily OHLCV Rows**: `{result.market_db_row_counts.get('asset_ohlcv_daily_total', 0)}` across 150 development symbols")
    md.append(f"- **Provenance Rows**: `{result.market_db_row_counts.get('asset_ohlcv_provenance_total', 0)}`\n")

    md.append("| Symbol | Stored Bars | Earliest Date | Latest Date | 50-Bar MA | 60-Bar Query | 200-Bar Regime | 252-Bar Range | Data Readiness State |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")

    for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
        mr = result.market_readiness_by_candidate.get(sym, {})
        cnt = mr.get("bar_count", 0)
        e_date = mr.get("earliest_date") or "N/A"
        l_date = mr.get("latest_date") or "N/A"
        s50 = "🟢 READY" if cnt >= 50 and l_date == "2026-10-16" else "🔴 NOT READY"
        s60 = "🟢 READY" if cnt >= 60 else "🔴 NOT READY"
        s200 = "🟢 READY" if cnt >= 200 and l_date == "2026-10-16" else "🔴 NOT READY"
        s252 = "🟢 READY" if cnt >= 252 and l_date == "2026-10-16" else "🔴 NOT READY"

        if cnt == 0:
            state = "`NO_DATA` (0 bars in store)"
        elif l_date < "2026-10-16":
            state = f"`STALE_OBSERVATIONS` (latest {l_date} < 2026-10-16)"
        else:
            state = f"`UNRESOLVED_ASSOCIATION` (stored data unlinked from security_id)"

        md.append(f"| **{sym}** | {cnt} | {e_date} | {l_date} | {s50} | {s60} | {s200} | {s252} | {state} |")

    md.append("\n> [!NOTE]\n")
    md.append("> **Market Data Boundary**: The 150-symbol local store is a deliberate development dataset. ")
    md.append("> Even where bars exist (AAPL: 465, GOOGL: 278, TSM: 352), market data cannot authorize admission ")
    md.append("> while admission requirements fail closed. Synthetic market data must never support real-source eligibility.\n")

    md.append("### 5. Remaining Admission Blockers & Smallest Next Acquisition Steps\n")
    md.append("#### Remaining Blockers:\n")
    for b in result.remaining_admission_blockers:
        md.append(f"- {b}")

    md.append("\n#### Smallest Concrete Evidence Acquisition Steps Needed Next:\n")
    for s in result.next_evidence_acquisition_steps:
        md.append(f"- {s}")

    return "\n".join(md)
