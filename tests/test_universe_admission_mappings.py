"""
tests/test_universe_admission_mappings.py

Sprint 2 Package 5 (S2-06): Policy-Adjudicated Admission Mappings & Evidence Test Suite.
Verifies both the Synthetic Contract Path and the Archived-Source Path under the
Product Owner's approved directions:

1. Stable Internal Identity with Attributable Listing Mappings.
2. Explicit Listing Semantics (listing_country vs country; selected_us_assessment_listing vs primary_listing).
3. Attributable Field Evidence without blanket USD assertions.
4. Supported Venue Normalization (A=AMEX, P=ARCA, Z=BATS, F=Provisional).
5. Diagnostic Triage (regex does not certify common stock).
6. Consumer-Specific Market Data Readiness (no blanket 60-bar gate).
7. Synthetic Contract Path (passes when evidence is legitimately complete).
8. Archived-Source Path (fails closed when currency/identity evidence remains unresolved;
   builder strictly NOT_RUN, 0 converted securities).
"""

from __future__ import annotations

import hashlib
import json
import pytest

from analyst_dashboard.universe.admission_mapping import (
    AttributableFieldEvidence,
    ClassificationTriageEngine,
    ConsumerReadinessEvaluator,
    EvidenceNature,
    ListingLocator,
    ListingSemanticAttributes,
    MarketDataReadinessState,
    OfflineAdmissionMappingEngine,
    SupportedVenueNormalizer,
    VenueNormalizationStatus,
)
from analyst_dashboard.universe.adapters.models import (
    ParsedSourceFile,
    ParsedSourceRow,
    SourceFeedType,
)
from analyst_dashboard.universe.adapters.resolution import (
    IdentityResolutionStatus,
    ResolutionRecordStatus,
    SecurityEvidenceResolver,
)
from analyst_dashboard.universe.validator import (
    UniverseInputValidator,
    ValidationRequirements,
    ValidationStatus,
)
from analyst_dashboard.universe.contracts import (
    SourcePopulationSnapshot,
    SourceSecurity,
)
from analyst_dashboard.universe.builder import DeterministicUniverseBuilder


# =====================================================================
# Canonical Parent Feed Digests & Original Row Locators (2026-10-16)
# =====================================================================

PARENT_SOURCE_DIGESTS = {
    "sec_company_tickers_exchange.json": "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f",
    "nasdaq_listed_directory.txt": "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272",
    "other_listed_directory.txt": "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792",
}

PARENT_ROW_LOCATORS = {
    "AAPL": {"sec_row_index": 1, "nasdaq_line": 24, "other_line": None},
    "GOOGL": {"sec_row_index": 2, "nasdaq_line": 2153, "other_line": None},
    "DDS": {"sec_row_index": 957, "nasdaq_line": None, "other_line": 1641},
    "ACCS": {"sec_row_index": 5696, "nasdaq_line": None, "other_line": 51},
    "TSM": {"sec_row_index": 6919, "nasdaq_line": None, "other_line": 6791},
    "GOOG": {"sec_row_index": 7475, "nasdaq_line": 2152, "other_line": None},
    "AACIU": {"sec_row_index": 10187, "nasdaq_line": 5, "other_line": None},
}

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


def _compute_digest(rows: list[ParsedSourceRow]) -> str:
    raw_objs = [r.raw_values for r in rows]
    return hashlib.sha256(json.dumps(raw_objs, sort_keys=True).encode("utf-8")).hexdigest()


def _build_real_sample_files() -> tuple[ParsedSourceFile, list[ParsedSourceFile]]:
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
            "parent_source_sha256": PARENT_SOURCE_DIGESTS["sec_company_tickers_exchange.json"],
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
            "parent_source_sha256": PARENT_SOURCE_DIGESTS["nasdaq_listed_directory.txt"],
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
            "parent_source_sha256": PARENT_SOURCE_DIGESTS["other_listed_directory.txt"],
        },
    )
    return sec_file, [nasdaq_file, other_file]


@pytest.mark.tier1
class TestUniverseAdmissionMappings:
    """Tests for Sprint 2 Package 5 admission mappings and qualified policy enforcement."""

    # ── 1. Identity & Listing Locator Invariants ─────────────────────────────

    def test_01_listing_locator_is_ephemeral_and_proves_no_continuity(self):
        locator = ListingLocator(operating_mic="XNAS", symbol="AAPL", as_of_date="2026-10-16")
        assert locator.to_locator_string() == "LOC_XNAS_AAPL_2026-10-16"
        d = locator.to_dict()
        assert d["proves_continuity"] is False
        assert d["identity_tier"] == "TIER_A_TIME_BOUNDED_LOCATOR"

    def test_02_symbol_reuse_does_not_inherit_identity(self):
        engine = OfflineAdmissionMappingEngine()

        # Issuer A registers XYZ in 2020
        sec1, assoc1 = engine.register_security(
            symbol="XYZ", operating_mic="XNAS", as_of_date="2020-01-01", issuer_cik="0000111111"
        )
        engine.close_association(assoc1.association_id, effective_to="2022-12-31")

        # Issuer B uses same symbol XYZ in 2025
        sec2, assoc2 = engine.register_security(
            symbol="XYZ", operating_mic="XNAS", as_of_date="2025-01-01", issuer_cik="0000222222"
        )

        # Invariant: Distinct issuers reusing the same symbol must NOT inherit identity
        assert sec1.internal_security_id != sec2.internal_security_id
        assert assoc1.association_id != assoc2.association_id

    def test_03_symbol_change_with_explicit_alias_preserves_continuity(self):
        engine = OfflineAdmissionMappingEngine()

        # FB listed in 2012
        sec_fb, assoc_fb = engine.register_security(
            symbol="FB", operating_mic="XNAS", as_of_date="2012-05-18", issuer_cik="0001326801"
        )
        engine.close_association(assoc_fb.association_id, effective_to="2022-06-08")

        # META ticker change in 2022 linked via alias
        sec_meta, assoc_meta = engine.register_security(
            symbol="META",
            operating_mic="XNAS",
            as_of_date="2022-06-09",
            issuer_cik="0001326801",
            explicit_continuity_alias_id=sec_fb.internal_security_id,
        )

        # Invariant: Explicit continuity preserves the internal_security_id
        assert sec_fb.internal_security_id == sec_meta.internal_security_id
        assert assoc_fb.association_id != assoc_meta.association_id
        assert assoc_meta.symbol == "META"

    def test_04_multi_symbol_issuer_distinct_identities(self):
        engine = OfflineAdmissionMappingEngine()

        # Alphabet Inc. CIK 0001652044 issues GOOG and GOOGL
        sec_goog, assoc_goog = engine.register_security(
            symbol="GOOG", operating_mic="XNAS", as_of_date="2026-10-16", issuer_cik="0001652044"
        )
        sec_googl, assoc_googl = engine.register_security(
            symbol="GOOGL", operating_mic="XNAS", as_of_date="2026-10-16", issuer_cik="0001652044"
        )

        # Invariant: Distinct share classes of the same issuer receive distinct IDs
        assert sec_goog.internal_security_id != sec_googl.internal_security_id
        assert sec_goog.issuer_cik == "0001652044"
        assert sec_googl.issuer_cik == "0001652044"

        cik_secs = engine.get_securities_for_cik("0001652044")
        assert len(cik_secs) == 2
        assert {s.primary_symbol for s in cik_secs} == {"GOOG", "GOOGL"}

    def test_05_idempotent_registration_prevents_duplicate_proliferation(self):
        engine = OfflineAdmissionMappingEngine()
        sec1, assoc1 = engine.register_security(
            symbol="MSFT", operating_mic="XNAS", as_of_date="2026-10-16", issuer_cik="0000789019"
        )
        sec2, assoc2 = engine.register_security(
            symbol="MSFT", operating_mic="XNAS", as_of_date="2026-10-16", issuer_cik="0000789019"
        )
        assert sec1.internal_security_id == sec2.internal_security_id
        assert assoc1.association_id == assoc2.association_id

    # ── 2. Explicit Listing Semantics (No Silent Remapping) ──────────────────

    def test_06_explicit_semantics_no_automatic_silent_remapping(self):
        attrs = ListingSemanticAttributes(
            listing_country="USA",
            incorporation_country="TWN",
            domicile_country="TWN",
            selected_us_assessment_listing=True,
            primary_us_listing=True,
            global_primary_listing=False,
        )

        # Invariant: Setting listing_country does NOT silently populate contract_country
        assert attrs.listing_country == "USA"
        assert attrs.contract_country is None

        # Invariant: Setting selected_us_assessment_listing does NOT silently populate contract_primary_listing
        assert attrs.selected_us_assessment_listing is True
        assert attrs.contract_primary_listing is None
        assert attrs.global_primary_listing is False

    # ── 3. Supported Venue Normalization ─────────────────────────────────────

    def test_07_documented_venue_normalization_expansions(self):
        # A -> AMEX (Tape B)
        res_a = SupportedVenueNormalizer.normalize_directory_code("A")
        assert res_a.normalized_venue == "AMEX"
        assert res_a.operating_mic == "XASE"
        assert res_a.dissemination_tape == "TAPE_B"
        assert res_a.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL

        # P -> ARCA (Tape B)
        res_p = SupportedVenueNormalizer.normalize_directory_code("P")
        assert res_p.normalized_venue == "ARCA"
        assert res_p.operating_mic == "ARCX"
        assert res_p.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL

        # Z -> BATS (Tape B)
        res_z = SupportedVenueNormalizer.normalize_directory_code("Z")
        assert res_z.normalized_venue == "BATS"
        assert res_z.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL

        # F -> TXSE (Provisional, Substantive Disagreement Preserved)
        res_f = SupportedVenueNormalizer.normalize_directory_code("F")
        assert res_f.normalized_venue == "TXSE"
        assert res_f.status == VenueNormalizationStatus.PROVISIONAL_UNRESOLVED
        assert res_f.is_substantive_disagreement is True

        # Unmapped code returns UNMAPPED_CODE
        res_bad = SupportedVenueNormalizer.normalize_directory_code("QQ")
        assert res_bad.normalized_venue is None
        assert res_bad.status == VenueNormalizationStatus.UNMAPPED_CODE

    def test_08_reconcile_dual_source_venues(self):
        # SEC: NYSE + Directory: A -> AMEX
        rec1 = SupportedVenueNormalizer.reconcile_dual_source_venues("NYSE", "A")
        assert rec1.normalized_venue == "AMEX"
        assert rec1.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL
        assert rec1.is_substantive_disagreement is False

        # SEC: TXSE + Directory: F -> PROVISIONAL_UNRESOLVED
        rec2 = SupportedVenueNormalizer.reconcile_dual_source_venues("TXSE", "F")
        assert rec2.status == VenueNormalizationStatus.PROVISIONAL_UNRESOLVED
        assert rec2.is_substantive_disagreement is True

        # Conflicting assertions: SEC NASDAQ vs Directory N -> UNRESOLVED_DISAGREEMENT
        rec3 = SupportedVenueNormalizer.reconcile_dual_source_venues("NASDAQ", "N")
        assert rec3.normalized_venue is None
        assert rec3.status == VenueNormalizationStatus.UNRESOLVED_DISAGREEMENT
        assert rec3.is_substantive_disagreement is True

    # ── 4. Diagnostic Triage vs Formal Classification ────────────────────────

    def test_09_classification_triage_does_not_certify_common_stock(self):
        # AACIU: Armada Acquisition Corp. III - Units
        hint_unit = ClassificationTriageEngine.evaluate_triage_hint("Armada Acquisition Corp. III - Units", is_etf=False)
        assert hint_unit.suspected_instrument_type == "UNIT"
        assert hint_unit.is_diagnostic_hint is True
        assert hint_unit.certification_established is False

        # AAPL: Apple Inc. - Common Stock
        hint_corp = ClassificationTriageEngine.evaluate_triage_hint("Apple Inc. - Common Stock", is_etf=False)
        assert hint_corp.suspected_instrument_type == "SUSPECTED_CORPORATE_EQUITY"
        # Invariant: Diagnostic hint cannot certify common stock
        assert hint_corp.certification_established is False

        # Unknown name returns UNKNOWN
        hint_unk = ClassificationTriageEngine.evaluate_triage_hint("", is_etf=False)
        assert hint_unk.suspected_instrument_type == "UNKNOWN"
        assert hint_unk.certification_established is False

    # ── 5. Consumer-Specific Market Data Readiness ───────────────────────────

    def test_10_consumer_readiness_distinct_states(self):
        candles_55 = [{"time": f"2026-01-{i:02d}", "close": 150.0} for i in range(1, 56)]

        # Consumer 1: INTERMEDIATE_TREND_50 (50 bars) -> READY
        res_50 = ConsumerReadinessEvaluator.evaluate_readiness("AAPL", candles_55, "INTERMEDIATE_TREND_50")
        assert res_50.readiness_state == MarketDataReadinessState.CONSUMER_READY
        assert res_50.is_ready is True

        # Consumer 2: STAGE2_LONG_TERM_TREND_200 (200 bars) -> INSUFFICIENT_HISTORY
        res_200 = ConsumerReadinessEvaluator.evaluate_readiness("AAPL", candles_55, "STAGE2_LONG_TERM_TREND_200")
        assert res_200.readiness_state == MarketDataReadinessState.INSUFFICIENT_HISTORY
        assert res_200.is_ready is False

        # 0 bars -> NO_DATA
        res_empty = ConsumerReadinessEvaluator.evaluate_readiness("GOOG", [], "INTERMEDIATE_TREND_50")
        assert res_empty.readiness_state == MarketDataReadinessState.NO_DATA

        # Negative price -> INVALID_OHLCV
        bad_candles = [{"time": "2026-01-01", "close": -10.0}]
        res_bad = ConsumerReadinessEvaluator.evaluate_readiness("BAD", bad_candles)
        assert res_bad.readiness_state == MarketDataReadinessState.INVALID_OHLCV

        # Stale date -> STALE_OBSERVATIONS
        stale_candles = [{"time": "2026-01-01", "close": 100.0}]
        res_stale = ConsumerReadinessEvaluator.evaluate_readiness("STALE", stale_candles, evaluation_cutoff="2026-10-16")
        assert res_stale.readiness_state == MarketDataReadinessState.STALE_OBSERVATIONS

    # ── 6. Path A: Synthetic Contract Path ───────────────────────────────────

    def test_11_synthetic_contract_path_passes_only_with_complete_evidence(self):
        """
        Demonstrates that when explicit, attributable synthetic evidence is provided
        for all required fields, the batch passes validation and builder evaluates.
        """
        synthetic_records = [
            {
                "security_id": "SYN_SEC_AAPL",
                "symbol": "AAPL",
                "exchange": "NASDAQ",
                "security_type": "COMMON_STOCK",
                "listing_status": "ACTIVE",
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            },
            {
                "security_id": "SYN_SEC_MSFT",
                "symbol": "MSFT",
                "exchange": "NASDAQ",
                "security_type": "COMMON_STOCK",
                "listing_status": "ACTIVE",
                "currency": "USD",
                "country": "USA",
                "primary_listing": True,
                "asset_class": "EQUITY",
                "listing_date": None,
                "delisting_date": None,
                "trading_suspension_date": None,
            },
        ]

        reqs = ValidationRequirements(
            allowed_values={"exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"}, "asset_class": {"EQUITY"}}
        )
        batch_val = UniverseInputValidator.validate_batch(synthetic_records, requirements=reqs)

        assert batch_val.is_batch_valid is True
        assert batch_val.passed_count == 2
        assert batch_val.failed_count == 0

        # SourceSecurity conversion succeeds
        source_secs = batch_val.to_source_securities()
        assert len(source_secs) == 2

        # DeterministicUniverseBuilder evaluates
        builder = DeterministicUniverseBuilder()
        snapshot = SourcePopulationSnapshot(
            snapshot_id="SYN_SNAP_001",
            source_authority="SYNTHETIC_TEST",
            as_of="2026-10-16T00:00:00Z",
            count=2,
            source_hash="syn_hash",
            securities=source_secs,
        )
        attestation, elig_rows, readiness_rows = builder.build_universe(snapshot)
        assert attestation.construction_status.value == "COMPLETE"
        assert attestation.eligible_count == 2
        assert len(elig_rows) == 2

    # ── 7. Path B: Archived-Source Path (Real Diagnostic Sample) ─────────────

    def test_12_archived_source_path_fails_closed_without_fabricated_evidence(self):
        """
        Evaluates the 7-candidate sample with real archived listing feeds.
        Because currency and production security_id remain unevidenced,
        and blanket USD conventions are rejected, the batch FAILS CLOSED:
        - 0 admitted records.
        - Builder strictly NOT_RUN.
        - Conflicts and unresolved identities preserved.
        """
        sec_file, dir_files = _build_real_sample_files()
        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=dir_files,
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
        )

        assert outcome.total_candidates == 7
        acc = outcome.accounting
        assert acc.identity_resolved_count == 0
        assert acc.identity_unresolved_count == 7

        reqs = ValidationRequirements(
            allowed_values={"exchange": {"NASDAQ", "NYSE", "ARCA", "BATS"}, "asset_class": {"EQUITY"}}
        )
        batch_val = outcome.validate_batch(requirements=reqs, allow_incomplete_source_files=True)

        # Atomic batch validation must fail closed
        assert batch_val.is_batch_valid is False
        assert batch_val.passed_count == 0
        assert batch_val.failed_count == 7

        # Attempting conversion must raise ValueError
        with pytest.raises(ValueError, match="Cannot convert records from a failed batch"):
            batch_val.to_source_securities()

        # Builder must NOT be called for a failed batch
        builder_executed = False
        if batch_val.is_batch_valid:
            builder_executed = True
        assert builder_executed is False

        # Verify that DDS retains venue conflict (TXSE vs F)
        dds_res = next(r for r in batch_val.records if r.raw_record.get("symbol") == "DDS")
        assert any(iss.issue_code.value == "CONFLICTING_MAPPING" for iss in dds_res.issues)

        # Verify that ACCS retains venue conflict (NYSE vs AMEX)
        accs_res = next(r for r in batch_val.records if r.raw_record.get("symbol") == "ACCS")
        assert any(iss.issue_code.value == "CONFLICTING_MAPPING" for iss in accs_res.issues)
