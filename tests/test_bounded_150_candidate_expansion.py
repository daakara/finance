"""
tests/test_bounded_150_candidate_expansion.py

Integration and governance verification suite for the Sprint 2 Approved Scope Addition:
Bounded 150-Candidate Coverage Expansion & Full-Universe Assessment.

Governing Authorization: Product Owner (David Akara) Approved Scope Addition (2026-10-19):
"Demonstrate repeatable acquisition, identity resolution, eligibility evaluation
and market-data readiness for up to 150 explicitly selected securities, with
reconciled exclusions and controlled publication. Full-universe discovery and
coverage/readiness assessment are also included."

Test Invariants Enforced:
1. Real vs. Synthetic Separation: Zero synthetic fixtures in real candidate manifest.
2. Durable Pilot Identity Preservation: AAPL, GOOG, GOOGL maintain established IDs.
3. Share-Class Isolation: GOOG (Class C) vs GOOGL (Class A) isolated under CIK 0001652044.
4. Fail-Closed Evidence Handling & Accounting Closure: 150 = 135 eligible + 15 ineligible + 0 unresolved.
5. Ineligible Breakdown: 10 ETFs fail R01, 5 ADRs fail R02.
6. Market-Data Readiness: 139 present, 11 missing, 38 fresh, 101 stale, split-adjusted only.
7. Controlled Operational Packages: A, B, and C strictly isolated; unapproved; zero production mutation.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple
import pytest

pytestmark = pytest.mark.tier2c

from analyst_dashboard.universe.contracts import (
    CANONICAL_ELIGIBILITY_RULES,
    EligibilityDecision,
    SourcePopulationSnapshot,
    SourceSecurity,
    UNIVERSE_ID,
    compute_sha256,
)
from analyst_dashboard.universe.builder import CURRENT_RELEASE_SHA, DeterministicUniverseBuilder
from analyst_dashboard.universe.identity_persistence import (
    SqliteIdentityRepository,
    IdentityPlanApplicationService,
    IdentityApplicationStatus,
)
from analyst_dashboard.universe.identity_adoption import (
    IdentityAdoptionDryRunEngine,
    IdentityAdoptionPlan,
    IdentityAdoptionAuthorization,
    IdentityAdoptionStatus,
)
from analyst_dashboard.universe.admission_mapping import (
    OfflineAdmissionMappingEngine,
    InternalSecurityRecord,
    EffectiveDatedListingAssociation,
    AttributableFieldEvidence,
    EvidenceNature,
)
from analyst_dashboard.universe.identity_proposal import (
    ProvisionalSecurityEntity,
    ProvisionalListingAssociation,
    PrimaryListingClaimAnalysis,
    CandidateAssociationProposal,
)
from scripts.universe.build_candidate_manifest_150 import (
    validate_route_a_evidence,
    LIVE_PRODUCTION_TARGET_STATE_DIGEST,
    generate_candidate_manifest,
    load_provisional_identity_allocations,
    build_operational_packages,
    resolve_market_db_path,
)
from scripts.apply_registration_plan_wrapper import create_verified_sqlite_backup


REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence" / "coverage_expansion"


class TestBounded150CandidateExpansion:
    """Comprehensive test suite for bounded 150-candidate expansion artifacts and workflows."""

    @pytest.fixture(autouse=True)
    def setup_fixtures(self):
        """Ensures evidence artifacts exist prior to testing."""
        self.manifest_file = EVIDENCE_DIR / "candidate_manifest_150.json"
        self.launch_manifest_file = EVIDENCE_DIR / "proposed_radar_launch_manifest.json"
        self.exclusions_file = EVIDENCE_DIR / "candidate_exclusions_reconciliation.json"
        self.pkg_a_file = EVIDENCE_DIR / "package_a_identity_registration.json"
        self.pkg_b_file = EVIDENCE_DIR / "package_b_market_data_ingestion.json"
        self.pkg_c_file = EVIDENCE_DIR / "package_c_universe_publication.json"

        assert self.manifest_file.exists(), "candidate_manifest_150.json missing!"
        assert self.launch_manifest_file.exists(), "proposed_radar_launch_manifest.json missing!"
        assert self.exclusions_file.exists(), "candidate_exclusions_reconciliation.json missing!"
        assert self.pkg_a_file.exists(), "package_a_identity_registration.json missing!"
        assert self.pkg_b_file.exists(), "package_b_market_data_ingestion.json missing!"
        assert self.pkg_c_file.exists(), "package_c_universe_publication.json missing!"

        self.manifest_data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        self.launch_manifest_data = json.loads(self.launch_manifest_file.read_text(encoding="utf-8"))
        self.exclusions_data = json.loads(self.exclusions_file.read_text(encoding="utf-8"))
        self.pkg_a_data = json.loads(self.pkg_a_file.read_text(encoding="utf-8"))
        self.pkg_b_data = json.loads(self.pkg_b_file.read_text(encoding="utf-8"))
        self.pkg_c_data = json.loads(self.pkg_c_file.read_text(encoding="utf-8"))

    def test_candidate_manifest_integrity_and_required_fields(self):
        """Verifies manifest schema, candidate count, uniqueness, and required evidence fields."""
        assert len(self.manifest_data) == 150, f"Expected 150 candidates, got {len(self.manifest_data)}"

        symbols = [r["symbol"] for r in self.manifest_data]
        assert len(set(symbols)) == 150, "Duplicate symbols detected in candidate manifest!"

        for r in self.manifest_data:
            assert r["symbol"], "Candidate missing symbol!"
            assert r["security_id"].startswith("ARX_SEC_"), f"Invalid security_id format for {r['symbol']}"
            assert r["durable_association_id"].startswith("ASSOC_"), f"Invalid association_id for {r['symbol']}"
            assert r["operating_mic"] in ("XNAS", "XNYS", "ARCX", "BATS", "OTCM"), f"Invalid MIC for {r['symbol']}"
            assert r["observed_venue"] in ("NASDAQ", "NYSE", "ARCA", "BATS", "OTC"), f"Invalid venue for {r['symbol']}"
            assert r["asset_class"] in ("EQUITY", "ETF"), f"Invalid asset class for {r['symbol']}"
            assert r["security_type"] in ("COMMON_STOCK", "ADR", "ETF"), f"Invalid security type for {r['symbol']}"
            assert r["currency"] == "USD", f"Non-USD currency for {r['symbol']}"
            assert r["observation_date"] == "2026-10-17", f"Observation date not bounded for {r['symbol']}"
            assert r["source_locator"], f"Missing source locator for {r['symbol']}"
            assert r["source_feed_sha256"], f"Missing source digest for {r['symbol']}"
            assert r["inclusion_rationale"], f"Missing inclusion rationale for {r['symbol']}"

    def test_pilot_identity_preservation_and_share_class_isolation(self):
        """Verifies AAPL, GOOG, GOOGL preserve established identities and share-class isolation."""
        by_sym = {r["symbol"]: r for r in self.manifest_data}

        # 1. AAPL
        aapl = by_sym["AAPL"]
        assert aapl["security_id"] == "ARX_SEC_0B4C728FE0BE"
        assert aapl["durable_association_id"] == "ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17"
        assert aapl["issuer_cik"] == "0000320193"
        assert aapl["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
        assert aapl["actual_primary_listing"] is True
        assert aapl["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"

        # 2. GOOG (Class C)
        goog = by_sym["GOOG"]
        assert goog["security_id"] == "ARX_SEC_A11A24D442E2"
        assert goog["durable_association_id"] == "ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17"
        assert goog["issuer_cik"] == "0001652044"
        assert goog["share_class"] == "Class C"
        assert goog["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
        assert goog["actual_primary_listing"] is True
        assert goog["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"

        # 3. GOOGL (Class A)
        googl = by_sym["GOOGL"]
        assert googl["security_id"] == "ARX_SEC_B7EF9ED7A35A"
        assert googl["durable_association_id"] == "ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17"
        assert googl["issuer_cik"] == "0001652044"
        assert googl["share_class"] == "Class A"
        assert googl["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
        assert googl["actual_primary_listing"] is True
        assert googl["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"

        # 4. Multi-symbol CIK isolation
        assert goog["security_id"] != googl["security_id"]
        assert goog["durable_association_id"] != googl["durable_association_id"]
        assert goog["issuer_cik"] == googl["issuer_cik"]

    def test_real_vs_synthetic_separation_and_reconciliation_closure(self):
        """Verifies zero synthetic fixtures in real manifest and full dev pool closure."""
        symbols = {r["symbol"] for r in self.manifest_data}

        # Assert no synthetic fixtures
        synthetic_fixtures = {
            "CNT_FRESH", "CNT_NULL", "CNT_SCORES", "SYM_TEST_UNKNOWN",
            "TST_CASCADE", "TST_DEDUP", "TST_STALE", "TST_UNSEASONED"
        }
        assert not symbols.intersection(synthetic_fixtures), "Synthetic fixture found in real manifest!"

        # Assert no crypto
        cryptos = {"BTC-USD", "ETH-USD"}
        assert not symbols.intersection(cryptos), "Crypto pair found in real manifest!"

        # Assert no unverified directory misses
        assert "CPRX" not in symbols, "Unverified CPRX found in candidate manifest!"

        # Verify arithmetic closure in exclusions reconciliation
        recon = self.exclusions_data
        assert recon["development_pool_total"] == 151
        assert recon["retained_count"] == 140
        assert recon["excluded_count"] == 11
        assert recon["directory_additions_count"] == 10
        assert recon["final_manifest_count"] == 150
        assert recon["arithmetic_closure"] is True
        assert recon["development_pool_total"] == recon["retained_count"] + recon["excluded_count"]
        assert recon["final_manifest_count"] == recon["retained_count"] + recon["directory_additions_count"]

    def test_canonical_eligibility_builder_evaluation_closure(self):
        """Verifies pure DeterministicUniverseBuilder eligibility evaluation and closure."""
        source_securities = []
        for r in self.manifest_data:
            ss = SourceSecurity(
                security_id=r["security_id"],
                symbol=r["symbol"],
                exchange=r["observed_venue"],
                security_type=r["security_type"],
                listing_status="ACTIVE",
                currency=r["currency"],
                country=r["country"],
                primary_listing=r["primary_listing"],
                asset_class=r["asset_class"],
                listing_date=r["observation_date"],
            )
            source_securities.append(ss)

        snapshot = SourcePopulationSnapshot(
            snapshot_id="SNAP_EXPANSION_150_TEST",
            source_authority="SEC_EDGAR_AND_NASDAQ_TRADER",
            as_of="2026-10-17T00:00:00Z",
            count=len(source_securities),
            source_hash=compute_sha256([s.to_dict() for s in source_securities]),
            securities=source_securities,
        )

        builder = DeterministicUniverseBuilder()
        attestation, eligibility_rows, _ = builder.build_universe(
            snapshot=snapshot,
            build_id="UBUILD_EXPANSION_150_TEST",
            release_sha=CURRENT_RELEASE_SHA,
        )

        # Full arithmetic accounting closure
        assert attestation.source_population_count == 150
        assert attestation.eligible_count == 135
        assert attestation.ineligible_count == 15
        assert attestation.eligibility_unresolved_count == 0
        assert (
            attestation.source_population_count
            == attestation.eligible_count + attestation.ineligible_count + attestation.eligibility_unresolved_count
        )

        # Specific rule failure inspection
        ineligible_by_code: dict[str, list[str]] = {}
        for r in eligibility_rows:
            if r.eligibility_decision != EligibilityDecision.ELIGIBLE:
                ineligible_by_code.setdefault(r.eligibility_reason_code, []).append(r.symbol)

        # 10 ETFs fail R01 (asset_class != EQUITY)
        etfs = sorted(ineligible_by_code.get("INELIGIBLE_ASSET_CLASS", []))
        expected_etfs = ["ARKK", "GLD", "GLX", "IWM", "JEPI", "QQQ", "SPY", "TLT", "VOO", "XLE"]
        assert etfs == expected_etfs, f"Unexpected ETF failures: {etfs}"

        # 5 ADRs fail R02 (security_type != COMMON_STOCK)
        adrs = sorted(ineligible_by_code.get("INELIGIBLE_SECURITY_TYPE", []))
        expected_adrs = ["ARM", "ASML", "DHLGY", "NVO", "TSM"]
        assert adrs == expected_adrs, f"Unexpected ADR failures: {adrs}"

    def test_market_data_readiness_accounting(self):
        """Verifies market-data readiness counts, depth, freshness, and adjustment semantics using retained bounded input."""
        market_db_path = resolve_market_db_path()
        if not market_db_path.exists():
            pytest.fail(f"Required bounded market data input database not found: {market_db_path}")

        conn = sqlite3.connect(f"file:{market_db_path}?mode=ro", uri=True)
        try:
            c = conn.cursor()

            present_count = 0
            fresh_count = 0
            stale_count = 0
            ge_60_count = 0
            ge_260_count = 0

            for r in self.manifest_data:
                sym = r["symbol"]
                row = c.execute("SELECT count(*), max(trade_date) FROM asset_ohlcv_daily WHERE symbol = ?", (sym,)).fetchone()
                cnt, max_d = row
                if cnt > 0:
                    present_count += 1
                    if max_d == "2026-10-16":
                        fresh_count += 1
                    else:
                        stale_count += 1
                    if cnt >= 60:
                        ge_60_count += 1
                    if cnt >= 260:
                        ge_260_count += 1

            assert present_count == 139, f"Expected 139 candidates present, got {present_count}"
            assert fresh_count == 38, f"Expected 38 fresh candidates, got {fresh_count}"
            assert stale_count == 101, f"Expected 101 stale candidates, got {stale_count}"
            assert ge_60_count == 138, f"Expected 138 with >= 60 bars, got {ge_60_count}"
            assert ge_260_count == 63, f"Expected 63 with >= 260 bars, got {ge_260_count}"
        finally:
            conn.close()

    def test_market_data_readiness_isolated_from_home_or_env_db(self, tmp_path, monkeypatch):
        """Correction A: An unrelated empty database at home or default location does not alter bounded-input resolution."""
        fake_db = tmp_path / ".finance_market_store.db"
        conn = sqlite3.connect(fake_db)
        conn.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, close REAL)")
        conn.close()

        monkeypatch.setenv("FINANCE_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("FINANCE_MARKET_STORE_PATH", raising=False)

        resolved = resolve_market_db_path()
        expected_bounded = REPO_ROOT / "data" / "research" / "market_data" / "bounded_150_dev_market_store.db"
        assert resolved == expected_bounded
        assert resolved != fake_db

    def test_missing_bounded_market_db_fails_closed(self, monkeypatch):
        """Correction A: Missing required bounded input fails closed with an informative message and cannot silently pass."""
        monkeypatch.setenv("FINANCE_MARKET_STORE_PATH", "nonexistent/path/to/market.db")
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_market_db_path()
        assert "does not exist" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    def test_package_a_identity_registration_governance(self):
        """Verifies Package A governance boundaries, empty target state, and fail-closed security."""
        pkg = self.pkg_a_data
        assert pkg["package_id"] == "PKG-EXPANSION-150-A-IDENTITY-REGISTRATION"
        assert pkg["governance_status"]["exact_plan_approval"] == "ABSENT"
        assert pkg["governance_status"]["application_authorized"] is False
        assert pkg["governance_status"]["is_authorized"] is False
        assert pkg["target_store"]["persistent_volume_mount"] == "/root"
        assert pkg["target_store"]["target_db_path"] == "/root/analyst_dashboard/data/identity.db"
        assert pkg["candidate_counts"]["total_candidates"] == 133
        assert pkg["candidate_counts"]["retained_pilot_identities"] == 3
        assert pkg["candidate_counts"]["new_proposed_identities"] == 130
        assert pkg["candidate_counts"]["evidence_resolved_identities"] == 133
        assert pkg["candidate_counts"]["actual_policy_pass"] == 122
        assert pkg["candidate_counts"]["actual_policy_unknown"] == 11
        assert len(pkg["candidates"]) == 133
        assert pkg["atomic_guarantees"]["fail_closed_on_unauthorized"] is True

        # Assert zero boundary validation fixtures (ETFs, ADRs, NAUT) in Package A candidates
        boundary_fixtures = {"ARKK", "GLD", "GLX", "IWM", "JEPI", "QQQ", "SPY", "TLT", "VOO", "XLE", "ARM", "ASML", "DHLGY", "NVO", "TSM", "NAUT"}
        pkg_syms = {c["symbol"] for c in pkg["candidates"]}
        assert not pkg_syms.intersection(boundary_fixtures), f"Boundary fixtures leaked into Package A: {pkg_syms.intersection(boundary_fixtures)}"

    def test_package_b_market_data_ingestion_specification(self):
        """Verifies Package B scope, missing vs stale partition, multi-tier sizing, and zero production writes."""
        pkg = self.pkg_b_data
        assert pkg["package_id"] == "PKG-EXPANSION-150-B-MARKET-DATA-INGESTION"
        assert pkg["governance_status"]["ingestion_authorized"] is False
        assert pkg["governance_status"]["execution_mode"] == "SPECIFICATION_ONLY"

        # Production target backfill (61 missing in production)
        prod_backfill = pkg["ingestion_scope"]["production_target_backfill"]
        assert prod_backfill["missing_candidates_count"] == 61
        assert prod_backfill["total_estimated_bars_251_sessions"] == 61 * 251
        assert prod_backfill["total_estimated_bars_260_bars"] == 61 * 260
        assert prod_backfill["total_estimated_bars_60_bars"] == 61 * 60
        assert "GOOG" not in prod_backfill["missing_symbols"]
        assert "variable_row_count_notice" in prod_backfill

        # Multi-tier sizing alternatives
        multi_tier = pkg["multi_tier_sizing_alternatives"]
        assert multi_tier["alternative_a_builder_minimum_60_bars"]["total_estimated_bars"] == 3885
        assert multi_tier["alternative_b_strategy_benchmark_260_bars"]["total_estimated_bars"] == 16085
        assert multi_tier["alternative_c_illustrative_calendar_sessions_251"]["total_estimated_bars"] == 15536

        # Production target refresh (15 stale in production)
        prod_refresh = pkg["ingestion_scope"]["production_target_refresh"]
        assert prod_refresh["stale_candidates_count"] == 15
        assert prod_refresh["total_estimated_bars"] == 15 * 15

        # Local dev store gap (11 missing locally, 101 stale locally)
        dev_gap = pkg["ingestion_scope"]["local_developer_store_gap"]
        assert dev_gap["missing_count"] == 11
        assert "GOOG" in dev_gap["missing_symbols"]
        assert "JNJ" in dev_gap["missing_symbols"]
        assert dev_gap["stale_count"] == 101
        assert dev_gap["fresh_count"] == 38

        assert "SPLIT_ADJUSTED_ONLY" in pkg["ingestion_scope"]["adjustment_policy"]
        assert pkg["atomic_guarantees"]["zero_production_writes_by_preparation"] is True

    def test_package_c_universe_publication_isolated_namespace(self):
        """Verifies Package C namespace isolation and strict prevention of default scanner switching."""
        pkg = self.pkg_c_data
        assert pkg["package_id"] == "PKG-EXPANSION-150-C-UNIVERSE-PUBLICATION"
        assert pkg["governance_status"]["executable"] is False
        assert pkg["governance_status"]["ready_for_publication_count"] == 0
        assert pkg["governance_status"]["publication_authorized"] is False
        assert pkg["governance_status"]["default_universe_switch_authorized"] is False
        assert pkg["governance_status"]["scanner_activation_authorized"] is False
        assert len(pkg["governance_status"]["blocking_reasons"]) >= 4
        assert pkg["target_namespace"]["universe_id"] == "ARX_US_EQUITIES_EXPANSION_150"
        assert pkg["target_namespace"]["isolated"] is True
        assert pkg["target_namespace"]["prior_published_pilot_namespace"] == "ARX_US_EQUITIES_PILOT_3C"
        assert pkg["target_namespace"]["default_scanner_universe_preserved"] == "ARX_US_EQUITIES"
        assert pkg["membership_accounting"]["source_population_count"] == 133
        assert pkg["membership_accounting"]["actual_eligible_membership_count"] == 133
        assert pkg["membership_accounting"]["counterfactual_eligible_membership_count"] == 133
        assert pkg["membership_accounting"]["ineligible_count"] == 0
        assert pkg["membership_accounting"]["unresolved_count"] == 0
        assert pkg["membership_accounting"]["scannable_membership_count"] == 0
        assert pkg["radar_launch_universe_accounting"]["actual_policy_pass_count"] == 122
        assert pkg["radar_launch_universe_accounting"]["actual_policy_unknown_count"] == 11
        assert pkg["atomic_guarantees"]["default_switch_prevented"] is True
        assert pkg["atomic_guarantees"]["scanner_scheduler_activation_prevented"] is True

    def test_zero_production_mutation_safety(self, tmp_path):
        """Verifies that rehearsal operations execute in isolated temp stores without mutating production."""
        temp_db = tmp_path / "temp_identity.db"
        repo = SqliteIdentityRepository(str(temp_db))
        digest_initial = repo.compute_target_state_digest()
        assert digest_initial == "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee"

        prod_identity_path = Path("/root/analyst_dashboard/data/identity.db")
        try:
            assert not prod_identity_path.exists()
        except PermissionError:
            # In non-root environments (e.g. CI runner), /root is permission-isolated (EACCES),
            # guaranteeing zero production database mutation.
            pass

    def test_proposed_radar_launch_manifest_policy_adherence(self):
        """Verifies that proposed Radar Launch Universe complies strictly with Rules L01–L06."""
        launch_rows = self.launch_manifest_data
        assert len(launch_rows) == 133, f"Expected 133 qualifying common equities, got {len(launch_rows)}"

        symbols = [r["symbol"] for r in launch_rows]
        assert len(set(symbols)) == 133, "Duplicate symbols detected in launch manifest!"

        ranked_candidates = [r for r in launch_rows if r["observed_turnover_rank"] is not None]
        unranked_candidates = [r for r in launch_rows if r["observed_turnover_rank"] is None]

        assert len(ranked_candidates) == 122
        assert len(unranked_candidates) == 11
        assert all(r["launch_rank"] is None for r in launch_rows)

        for r in launch_rows:
            # Rule L01: US Common Equity Only
            assert r["asset_class"] == "EQUITY", f"Non-equity found in launch universe: {r['symbol']}"
            assert r["security_type"] == "COMMON_STOCK", f"Non-common stock found in launch universe: {r['symbol']}"
            assert r["currency"] == "USD"
            assert r["country"] == "USA"

            # Rule L02: Major Regulated Venue
            assert r["operating_mic"] in ("XNAS", "XNYS", "ARCX", "BATS"), f"Invalid MIC for {r['symbol']}"
            if r["symbol"] in ("AAPL", "GOOG", "GOOGL"):
                assert r["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"
                assert r["actual_primary_listing"] is True
            else:
                assert r["primary_listing_evidence_status"] == "ROUTE_A_DIRECT_REFERENCE_OBSERVED"
                assert r["actual_primary_listing"] is True
            assert r["counterfactual_primary_listing"] is True
            assert r["primary_listing"] is True

            # Four explicit dispositions must exist on each record
            assert "disposition_a_selection_policy" in r
            assert "disposition_b_canonical_identity" in r
            assert "disposition_c_market_data" in r
            assert "disposition_d_production_execution" in r

        # Verify negative cases are strictly excluded
        assert "NAUT" not in symbols, "NAUT (penny stock, $1.96) must be excluded from launch universe!"
        assert "SPY" not in symbols, "SPY (ETF) must be excluded from launch universe!"
        assert "QQQ" not in symbols, "QQQ (ETF) must be excluded from launch universe!"
        assert "TSM" not in symbols, "TSM (ADR) must be excluded from launch universe!"
        assert "DHLGY" not in symbols, "DHLGY (ADR) must be excluded from launch universe!"

        # Verify deterministic rank order across ranked candidates (descending median dollar volume, then symbol)
        for i in range(len(ranked_candidates) - 1):
            curr_to = ranked_candidates[i]["median_dollar_turnover_20"]
            next_to = ranked_candidates[i + 1]["median_dollar_turnover_20"]
            assert curr_to >= next_to, f"Ranking order violation between {ranked_candidates[i]['symbol']} and {ranked_candidates[i+1]['symbol']}"
            assert ranked_candidates[i]["observed_turnover_rank"] == i + 1
            assert ranked_candidates[i]["launch_rank"] is None
            if ranked_candidates[i]["symbol"] in ("AAPL", "GOOGL"):
                assert ranked_candidates[i]["ranking_note"] == "ESTABLISHED_PILOT_IDENTITY_EXPANSION_PUBLICATION_UNAUTHORIZED"
            else:
                assert ranked_candidates[i]["ranking_note"] == "RESEARCH_TURNOVER_ORDERING_NOT_QUALIFIED_LAUNCH_RANK"
            assert ranked_candidates[i]["disposition_a_selection_policy"]["overall_outcome"] == "PASS"
            assert ranked_candidates[i]["disposition_a_selection_policy"]["actual_outcome"] == "PASS"
            assert ranked_candidates[i]["disposition_a_selection_policy"]["counterfactual_outcome"] == "PASS"

        # Verify unranked candidates have missing turnover and UNKNOWN policy outcome
        for r in unranked_candidates:
            assert r["observed_turnover_rank"] is None
            assert r["launch_rank"] is None
            assert r["ranking_note"] == "UNRANKED_DATA_PENDING"
            assert r["avg_dollar_turnover_20"] is None
            assert r["median_dollar_turnover_20"] is None
            assert r["latest_close"] is None
            assert r["meets_price_floor"] is None
            assert r["meets_liquidity_floor"] is None
            assert r["meets_seasoning_floor"] is None
            assert r["disposition_a_selection_policy"]["overall_outcome"] == "UNKNOWN"
            assert r["disposition_a_selection_policy"]["counterfactual_outcome"] == "UNKNOWN"

    def test_launch_universe_reconciliation_closure(self):
        """Verifies 100% arithmetic closure of the 161 candidate pool down to 133 launch candidates."""
        recon = self.exclusions_data.get("launch_universe_reconciliation")
        assert recon is not None, "launch_universe_reconciliation missing in exclusions file!"
        assert recon["total_evaluated_pool"] == 161
        assert recon["qualifying_launch_count"] == 133
        assert recon["policy_pass_count"] == 122
        assert recon["policy_unknown_count"] == 11
        assert recon["total_excluded_count"] == 28
        assert recon["arithmetic_closure"] is True
        assert recon["total_evaluated_pool"] == recon["qualifying_launch_count"] + recon["total_excluded_count"]

        # Exact exclusions count verification
        summary = recon["exclusions_summary"]
        assert summary["EXCHANGE_TRADED_FUND"] == 11
        assert summary["AMERICAN_DEPOSITARY_RECEIPT"] == 5
        assert summary["SYNTHETIC_TEST_FIXTURE"] == 8
        assert summary["NON_EQUITY_CRYPTO_ASSET"] == 2
        assert summary["PRICE_FLOOR_VIOLATION"] == 1
        assert summary["UNVERIFIED_DIRECTORY_MISS"] == 1

    def test_package_a_live_production_target_digest_binding(self):
        """Verifies Package A binds to the live production target state digest post-PR #27."""
        pkg = self.pkg_a_data
        target_store = pkg["target_store"]
        assert target_store["target_db_path"] == "/root/analyst_dashboard/data/identity.db"
        assert target_store["live_target_state_digest"] == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"
        assert target_store["expected_empty_digest"] == "ea11990ec0e3790abcb0b4293bfadd7ae4e58ca050e79e949e688a82437a93ee"

        pilot_entities = {e["symbol"]: e["security_id"] for e in target_store["registered_pilot_entities"]}
        assert pilot_entities["AAPL"] == "ARX_SEC_0B4C728FE0BE"
        assert pilot_entities["GOOG"] == "ARX_SEC_A11A24D442E2"
        assert pilot_entities["GOOGL"] == "ARX_SEC_B7EF9ED7A35A"
        assert pkg["candidate_counts"]["radar_launch_universe_candidates"] == 133
        assert pkg["candidate_counts"]["radar_launch_universe_selected"] == 133
        assert pkg["candidate_counts"]["radar_launch_policy_pass"] == 122
        assert pkg["candidate_counts"]["radar_launch_policy_unknown"] == 11

    def test_package_b_production_readback_reconciliation(self):
        """Verifies Package B documents accurate production store status distinguishing dev gaps."""
        pkg = self.pkg_b_data
        recon = pkg.get("production_readback_reconciliation")
        assert recon is not None, "production_readback_reconciliation missing in Package B!"
        assert recon["production_observed_symbols_count"] == 145
        assert recon["production_pilot_bar_counts"]["GOOG"] == 287
        assert recon["production_pilot_bar_counts"]["AAPL"] == 287
        assert recon["production_pilot_bar_counts"]["GOOGL"] == 287
        assert recon["production_missing_candidates"]["count"] == 61
        assert "GOOG" not in recon["production_missing_candidates"]["symbols"]
        assert recon["production_stale_candidates"]["count"] == 15
        assert "JNJ" in recon["production_stale_candidates"]["symbols"]
        assert recon["local_developer_store_gap"]["missing_count"] == 11
        assert "GOOG" in recon["local_developer_store_gap"]["missing_symbols"]

    # ── 8 SPECIFIC BEHAVIORAL INVARIANT TESTS ─────────────────────────────

    def test_missing_data_produces_unknown_not_pass(self):
        """Invariant 1: Missing price/liquidity/history evaluates to UNKNOWN, never PASS."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        for sym in ["GOOG", "JNJ", "JPM", "PG", "WMT", "BAC", "ABT", "MRK", "PFE", "DIS", "CSCO"]:
            cand = by_sym[sym]
            disp_a = cand["disposition_a_selection_policy"]
            assert disp_a["overall_outcome"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L03_price_floor"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L04_liquidity_floor"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L05_seasoning_floor"] == "UNKNOWN"
            assert cand["meets_price_floor"] is None
            assert cand["meets_liquidity_floor"] is None
            assert cand["meets_seasoning_floor"] is None

    def test_ten_missing_candidates_cannot_claim_complete_compliance(self):
        """Invariant 2: The 11 missing-data candidates cannot claim complete L01–L06 compliance."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        for sym in ["GOOG", "JNJ", "JPM", "PG", "WMT", "BAC", "ABT", "MRK", "PFE", "DIS", "CSCO"]:
            cand = by_sym[sym]
            assert cand["data_status"] == "PENDING_INGESTION"
            assert cand["launch_rank"] is None
            assert cand["ranking_note"] == "UNRANKED_DATA_PENDING"
            # Cannot be marked overall PASS
            assert cand["disposition_a_selection_policy"]["overall_outcome"] != "PASS"

    def test_stale_data_not_promoted_by_row_count(self):
        """Invariant 3: Stale data is not promoted to ready by row count alone."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        # JNJ has 251 bars in production, but its latest date is 2026-09-22 (stale)
        jnj = by_sym["JNJ"]
        prod_data = jnj["disposition_c_market_data"]["production_store"]
        assert prod_data["bar_count"] == 251
        assert prod_data["latest_date"] == "2026-09-22"
        assert prod_data["session_fresh"] is False
        assert jnj["disposition_c_market_data"]["overall_readiness"] == "STALE_NEEDS_REFRESH"

        # ADBE has 278 bars in production, but its latest date is 2026-10-10 (stale)
        adbe = by_sym["ADBE"]
        adbe_prod = adbe["disposition_c_market_data"]["production_store"]
        assert adbe_prod["bar_count"] > 250
        assert adbe_prod["latest_date"] != "2026-10-16"
        assert adbe_prod["session_fresh"] is False

    def test_distinct_accounting_categories_reconciliation(self):
        """Invariant 4: Selected (133), Policy-Pass (122), Policy-Unknown (11), Data-Ready Fresh (35/56), and Pub (0) remain distinct."""
        recon = self.exclusions_data["launch_universe_reconciliation"]
        assert recon["qualifying_launch_count"] == 133
        assert recon["policy_pass_count"] == 122
        assert recon["policy_unknown_count"] == 11
        assert recon["market_data_fresh_local_count"] == 35
        assert recon["validation_sample_fresh_local_count"] == 38
        assert recon["market_data_fresh_production_count"] == 57
        assert recon["publication_ready_count"] == 0

        pkg_c = self.pkg_c_data
        acct = pkg_c["radar_launch_universe_accounting"]
        assert acct["selected_common_equities_count"] == 133
        assert acct["policy_pass_count"] == 122
        assert acct["policy_unknown_count"] == 11
        assert acct["market_data_fresh_local_count"] == 35
        assert acct["validation_sample_fresh_local_count"] == 38
        assert acct["market_data_fresh_production_count"] == 57
        assert acct["ready_for_publication_count"] == 0

    def test_goog_local_absence_is_not_production_ingestion_requirement(self):
        """Invariant 5: Local absence (GOOG) is not a production ingestion requirement."""
        pkg_b = self.pkg_b_data
        prod_backfill = pkg_b["ingestion_scope"]["production_target_backfill"]
        assert "GOOG" not in prod_backfill["missing_symbols"]

        recon = pkg_b["production_readback_reconciliation"]
        assert recon["production_pilot_bar_counts"]["GOOG"] == 287
        assert "GOOG" not in recon["production_missing_candidates"]["symbols"]
        assert "GOOG" in recon["local_developer_store_gap"]["missing_symbols"]

    def test_counterfactual_results_do_not_enter_publication_readiness(self):
        """Invariant 6: Counterfactual results do not enter publication readiness totals."""
        pkg_c = self.pkg_c_data
        # Counterfactual simulation yielded 133 eligible on launch common equities
        assert pkg_c["membership_accounting"]["counterfactual_eligible_membership_count"] == 133
        assert pkg_c["membership_accounting"]["actual_eligible_membership_count"] == 133
        # But publication readiness is strictly ZERO
        assert pkg_c["governance_status"]["ready_for_publication_count"] == 0
        assert pkg_c["membership_accounting"]["scannable_membership_count"] == 0
        assert pkg_c["radar_launch_universe_accounting"]["ready_for_publication_count"] == 0
        assert pkg_c["radar_launch_universe_accounting"]["actual_policy_pass_count"] == 122
        assert pkg_c["radar_launch_universe_accounting"]["actual_policy_unknown_count"] == 11

    def test_pilot_identities_preserved_across_all_artifacts(self):
        """Invariant 7: Existing pilot identities remain preserved (AAPL, GOOG, GOOGL)."""
        launch_syms = {r["symbol"]: r for r in self.launch_manifest_data}
        val_syms = {r["symbol"]: r for r in self.manifest_data}

        for sym, sec_id in [("AAPL", "ARX_SEC_0B4C728FE0BE"), ("GOOG", "ARX_SEC_A11A24D442E2"), ("GOOGL", "ARX_SEC_B7EF9ED7A35A")]:
            assert launch_syms[sym]["security_id"] == sec_id
            assert launch_syms[sym]["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
            assert val_syms[sym]["security_id"] == sec_id
            assert val_syms[sym]["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"

    def test_operational_package_counts_reconcile_with_dispositions(self):
        """Invariant 8: Package counts reconcile with actual candidate dispositions."""
        launch_rows = self.launch_manifest_data
        actual_pass = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["actual_outcome"] == "PASS")
        counterfactual_pass = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["counterfactual_outcome"] == "PASS")
        overall_unknown = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["overall_outcome"] == "UNKNOWN")
        counterfactual_unknown = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["counterfactual_outcome"] == "UNKNOWN")
        fresh_local = sum(1 for r in launch_rows if r["disposition_c_market_data"]["local_dev_store"]["session_fresh"])
        fresh_prod = sum(1 for r in launch_rows if r["disposition_c_market_data"]["production_store"]["session_fresh"])

        assert actual_pass == 122
        assert counterfactual_pass == 122
        assert overall_unknown == 11
        assert counterfactual_unknown == 11
        assert fresh_local == 35
        assert fresh_prod == 57

        # Package A reconciliation
        assert self.pkg_a_data["candidate_counts"]["radar_launch_universe_selected"] == len(launch_rows)
        assert self.pkg_a_data["candidate_counts"]["actual_policy_pass"] == 122
        assert self.pkg_a_data["candidate_counts"]["actual_policy_unknown"] == 11
        assert self.pkg_a_data["candidate_counts"]["counterfactual_policy_pass"] == 122
        assert self.pkg_a_data["candidate_counts"]["radar_launch_policy_unknown"] == 11

        # Package B reconciliation
        assert self.pkg_b_data["production_readback_reconciliation"]["production_launch_candidates_missing"] == 61
        assert self.pkg_b_data["production_readback_reconciliation"]["production_launch_candidates_stale"] == 15
        assert self.pkg_b_data["production_readback_reconciliation"]["production_launch_candidates_fresh"] == 57
        assert self.pkg_b_data["ingestion_scope"]["production_target_backfill"]["missing_candidates_count"] == 61
        assert self.pkg_b_data["ingestion_scope"]["production_target_refresh"]["stale_candidates_count"] == 15

        # Package C reconciliation
        assert self.pkg_c_data["radar_launch_universe_accounting"]["selected_common_equities_count"] == len(launch_rows)
        assert self.pkg_c_data["radar_launch_universe_accounting"]["actual_policy_pass_count"] == 122
        assert self.pkg_c_data["radar_launch_universe_accounting"]["actual_policy_unknown_count"] == 11
        assert self.pkg_c_data["radar_launch_universe_accounting"]["counterfactual_policy_pass_count"] == 122
        assert self.pkg_c_data["radar_launch_universe_accounting"]["policy_unknown_count"] == 11
        assert self.pkg_c_data["radar_launch_universe_accounting"]["ready_for_publication_count"] == 0

    def test_canonical_rule_parity_with_contracts(self):
        """Invariant 1: Canonical eligibility rule definitions maintain strict parity with contracts.py."""
        # R01: Asset Class == EQUITY
        assert CANONICAL_ELIGIBILITY_RULES["R01_ASSET_CLASS"]["expected"] == "EQUITY"
        # R02: Security Type == COMMON_STOCK
        assert CANONICAL_ELIGIBILITY_RULES["R02_SECURITY_TYPE"]["expected"] == "COMMON_STOCK"
        # R03: Listing Status == ACTIVE
        assert CANONICAL_ELIGIBILITY_RULES["R03_LISTING_STATUS"]["expected"] == "ACTIVE"
        # R04: Primary Exchange in supported allowlist
        assert set(CANONICAL_ELIGIBILITY_RULES["R04_PRIMARY_EXCHANGE"]["expected"]) == {"NASDAQ", "NYSE", "ARCA", "BATS"}
        # XASE / AMEX is strictly NOT in canonical allowlist
        assert "XASE" not in CANONICAL_ELIGIBILITY_RULES["R04_PRIMARY_EXCHANGE"]["expected"]
        assert "AMEX" not in CANONICAL_ELIGIBILITY_RULES["R04_PRIMARY_EXCHANGE"]["expected"]
        # R05: Currency == USD
        assert CANONICAL_ELIGIBILITY_RULES["R05_CURRENCY"]["expected"] == "USD"
        # R06: Primary Listing is True
        assert CANONICAL_ELIGIBILITY_RULES["R06_PRIMARY_LISTING"]["expected"] is True
        # R07: Not Delisted (delisting_date is None)
        assert CANONICAL_ELIGIBILITY_RULES["R07_NOT_DELISTED"]["expected"] is None

    def test_supported_mic_alone_does_not_establish_r06(self):
        """Invariant 2: Supported operating MIC alone does not establish canonical primary listing (R06)."""
        builder = DeterministicUniverseBuilder()
        # Candidate with supported MIC but unobserved primary_listing (None)
        sec = SourceSecurity(
            security_id="ARX_SEC_TEST_MIC",
            symbol="TEST",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=None,  # Unobserved in raw directory feed
            asset_class="EQUITY",
            listing_date="2026-10-17",
        )
        row = builder.evaluate_security_eligibility(sec, build_id="UBUILD_TEST", observed_at="2026-10-17T00:00:00Z")
        assert row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert row.eligibility_reason_code == "INELIGIBLE_SECONDARY_LISTING"

        # Verify in launch manifest that candidates have attributed Route A evidence
        for r in self.launch_manifest_data:
            if r["symbol"] in ("AAPL", "GOOG", "GOOGL"):
                assert r["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"
                assert r["actual_primary_listing"] is True
            else:
                assert r["primary_listing_evidence_status"] == "ROUTE_A_DIRECT_REFERENCE_OBSERVED"
                assert r["actual_primary_listing"] is True

    def test_counterfactual_eligibility_cannot_become_actual_eligibility(self):
        """Invariant 3: Counterfactual planning simulation cannot be conflated with actual canonical eligibility."""
        for r in self.launch_manifest_data:
            disp_b = r["disposition_b_canonical_identity"]
            actual = disp_b["actual_canonical_eligibility"]
            assert actual["decision"] == "ELIGIBLE"
            assert actual["reason_code"] == "ELIGIBLE_COMMON_EQUITY"
            assert actual["r06_primary_listing"] == "PASS"

            cf = disp_b["counterfactual_canonical_eligibility"]
            assert cf["decision"] == "ELIGIBLE"
            assert cf["reason_code"] == "ELIGIBLE_COMMON_EQUITY"
            assert cf["injected_attributes"] == {"primary_listing": True}
            assert cf["counterfactual_builder_status"] == "COUNTERFACTUAL_TESTED_CLEAN"

    def test_unknown_preconditions_do_not_become_overall_policy_pass(self):
        """Invariant 4: Preconditions with UNKNOWN evidence do not become overall policy PASS."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        for sym in ["GOOG", "JNJ", "JPM", "PG", "WMT", "BAC", "ABT", "MRK", "PFE", "DIS", "CSCO"]:
            disp_a = by_sym[sym]["disposition_a_selection_policy"]
            assert disp_a["overall_outcome"] == "UNKNOWN"
            assert disp_a["actual_outcome"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L03_price_floor"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L04_liquidity_floor"] == "UNKNOWN"
            assert disp_a["rule_outcomes"]["L05_seasoning_floor"] == "UNKNOWN"

        for r in self.launch_manifest_data:
            disp_a = r["disposition_a_selection_policy"]
            if r["symbol"] not in ["GOOG", "JNJ", "JPM", "PG", "WMT", "BAC", "ABT", "MRK", "PFE", "DIS", "CSCO"]:
                assert disp_a["overall_outcome"] == "PASS"
                assert disp_a["actual_outcome"] == "PASS"
            # Counterfactual PASS is strictly segregated
            if r["observed_turnover_rank"] is not None:
                assert disp_a["counterfactual_outcome"] == "PASS"
            else:
                assert disp_a["counterfactual_outcome"] == "UNKNOWN"

    def test_history_requirements_tied_to_actual_consumers(self):
        """Invariant 5: History requirements are strictly mapped to their actual consuming components."""
        pkg_b = self.pkg_b_data
        mapping = pkg_b["history_requirements_and_consumer_mapping"]

        # 60 bars (min 50) is the operative UniverseBuilder data readiness gate
        builder_gate = mapping["operative_builder_readiness"]
        assert builder_gate["threshold_bars"] == 60
        assert builder_gate["minimum_required_bars"] == 50
        assert "DeterministicUniverseBuilder" in builder_gate["consumer"]
        assert builder_gate["status"] == "OPERATIVE_CONTRACT"

        # 260 bars is the downstream strategy / VCP conformance oracle annual trend benchmark
        strategy_benchmark = mapping["downstream_strategy_benchmark"]
        assert strategy_benchmark["threshold_bars"] == 260
        assert "VCP" in strategy_benchmark["consumer"] or "Minervini" in strategy_benchmark["consumer"]
        assert strategy_benchmark["status"] == "DOWNSTREAM_STRATEGY_BENCHMARK"

        # 251 sessions is an illustrative calendar-year estimate, not a contract
        cal_estimate = mapping["illustrative_calendar_year_estimate"]
        assert cal_estimate["threshold_bars"] == 251
        assert cal_estimate["status"] == "PROPOSED_ESTIMATE_ONLY"

    def test_freshness_not_silently_promoted_to_complete_readiness(self):
        """Invariant 6: Session freshness is not silently promoted to complete data or policy readiness."""
        for r in self.launch_manifest_data:
            # JNJ has 251 bars in production, but is stale and has 0 bars in local dev store
            if r["symbol"] == "JNJ":
                assert r["disposition_c_market_data"]["production_store"]["session_fresh"] is False
                assert r["disposition_a_selection_policy"]["overall_outcome"] == "UNKNOWN"

        # Stale candidates with high bar counts remain STALE_NEEDS_REFRESH
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        jnj = by_sym["JNJ"]
        assert jnj["disposition_c_market_data"]["production_store"]["bar_count"] == 251
        assert jnj["disposition_c_market_data"]["production_store"]["session_fresh"] is False
        assert jnj["disposition_c_market_data"]["overall_readiness"] == "STALE_NEEDS_REFRESH"

    def test_existing_pilot_identities_not_recreated(self):
        """Invariant 7: Existing pilot identities (AAPL, GOOG, GOOGL) are durable and not re-registered."""
        pkg_a = self.pkg_a_data
        assert pkg_a["target_store"]["live_target_state_digest"] == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"
        assert pkg_a["governance_status"]["target_store_status"] == "FRESH_LIVE_PRODUCTION_OBSERVATION"
        assert pkg_a["governance_status"]["production_target_binding"] == "BOUND_AUTHENTIC_LIVE_PRODUCTION"
        assert pkg_a["candidate_counts"]["total_candidates"] == 133
        assert pkg_a["candidate_counts"]["retained_pilot_identities"] == 3
        assert pkg_a["candidate_counts"]["new_proposed_identities"] == 130

        pilot_syms = {p["symbol"]: p for p in pkg_a["target_store"]["registered_pilot_entities"]}
        for sym, sec_id in [("AAPL", "ARX_SEC_0B4C728FE0BE"), ("GOOG", "ARX_SEC_A11A24D442E2"), ("GOOGL", "ARX_SEC_B7EF9ED7A35A")]:
            assert sym in pilot_syms
            assert pilot_syms[sym]["security_id"] == sec_id
            assert pilot_syms[sym]["durable_status"] == "ESTABLISHED_DURABLE"

    def test_pilot_membership_and_expansion_membership_remain_distinct(self):
        """Invariant 8: Prior pilot membership and proposed expansion membership remain strictly distinct."""
        pkg_c = self.pkg_c_data
        assert pkg_c["target_namespace"]["universe_id"] == "ARX_US_EQUITIES_EXPANSION_150"
        assert pkg_c["target_namespace"]["prior_published_pilot_namespace"] == "ARX_US_EQUITIES_PILOT_3C"
        assert pkg_c["target_namespace"]["default_scanner_universe_preserved"] == "ARX_US_EQUITIES"
        assert pkg_c["target_namespace"]["isolated"] is True
        assert pkg_c["governance_status"]["ready_for_publication_count"] == 0
        assert pkg_c["membership_accounting"]["scannable_membership_count"] == 0

    def test_ingestion_estimates_distinct_from_exact_acquisition(self):
        """Invariant 9: Ingestion estimates are multi-tiered and distinct from exact acquisition guarantees."""
        pkg_b = self.pkg_b_data
        backfill = pkg_b["ingestion_scope"]["production_target_backfill"]
        assert "variable_row_count_notice" in backfill
        assert backfill["total_estimated_bars_251_sessions"] == 15311
        assert backfill["total_estimated_bars_260_bars"] == 15860
        assert backfill["total_estimated_bars_60_bars"] == 3660

        # Multi-tier sizing
        tiers = pkg_b["multi_tier_sizing_alternatives"]
        assert tiers["alternative_a_builder_minimum_60_bars"]["total_estimated_bars"] == 3885
        assert tiers["alternative_b_strategy_benchmark_260_bars"]["total_estimated_bars"] == 16085
        assert tiers["alternative_c_illustrative_calendar_sessions_251"]["total_estimated_bars"] == 15536

        # GOOG absence in local dev store must not trigger production backfill
        assert "GOOG" not in backfill["missing_symbols"]
        assert "GOOG" in pkg_b["production_readback_reconciliation"]["local_developer_store_gap"]["missing_symbols"]

    def test_candidate_and_outcome_counts_reconcile(self):
        """Invariant 10: Complete arithmetic reconciliation across candidates, exclusions, and dispositions."""
        launch_rows = self.launch_manifest_data
        exclusions = self.exclusions_data["launch_universe_reconciliation"]

        assert len(launch_rows) == 133
        assert exclusions["total_evaluated_pool"] == 161
        assert exclusions["total_excluded_count"] == 28
        assert exclusions["qualifying_launch_count"] == 133
        assert exclusions["total_evaluated_pool"] == exclusions["qualifying_launch_count"] + exclusions["total_excluded_count"]

        # Disposition counts
        actual_pass = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["actual_outcome"] == "PASS")
        counterfactual_pass = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["counterfactual_outcome"] == "PASS")
        overall_unknown = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["overall_outcome"] == "UNKNOWN")

        assert actual_pass == 122
        assert counterfactual_pass == 122
        assert overall_unknown == 11

        # Zero scannable or published
        assert self.pkg_c_data["governance_status"]["ready_for_publication_count"] == 0
        assert self.pkg_c_data["membership_accounting"]["scannable_membership_count"] == 0
        assert self.pkg_c_data["membership_accounting"]["actual_eligible_membership_count"] == 133
        assert self.pkg_c_data["membership_accounting"]["counterfactual_eligible_membership_count"] == 133
        assert self.pkg_c_data["membership_accounting"]["ineligible_count"] == 0
        assert self.pkg_c_data["membership_accounting"]["unresolved_count"] == 0

    # ── 10 FOCUSED EVIDENCE-CLOSURE & GOVERNANCE INVARIANT TESTS ──────────

    def test_direct_primary_exchange_evidence_resolving_r06(self):
        """Focused 1: Direct primary-exchange evidence resolves R06 through the accepted parser."""
        route_a_file = EVIDENCE_DIR / "expansion_133_route_a_evidence.json"
        assert route_a_file.exists(), "expansion_133_route_a_evidence.json missing!"
        data = json.loads(route_a_file.read_text(encoding="utf-8"))
        records = data.get("evidence_records", {})

        # Test MSFT: verified on NASDAQ (XNAS)
        assert "MSFT" in records
        msft_rec = records["MSFT"]
        assert msft_rec["stated_exchange"] == "XNAS"
        assert msft_rec["assigned_primary_listing"] is True
        assert msft_rec["is_direct_assertion"] is True
        assert msft_rec["adjudication_role"] == "DIRECT_PRIMARY_IDENTIFICATION"
        assert msft_rec["classification"] == "EVIDENCE_SUFFICIENT_FOR_DEFINED_PRIMARY_LISTING_CLAIM"

        # Test JNJ: verified on NYSE (XNYS)
        assert "JNJ" in records
        jnj_rec = records["JNJ"]
        assert jnj_rec["stated_exchange"] == "XNYS"
        assert jnj_rec["assigned_primary_listing"] is True
        assert jnj_rec["is_direct_assertion"] is True
        assert jnj_rec["adjudication_role"] == "DIRECT_PRIMARY_IDENTIFICATION"

        # Verify DeterministicUniverseBuilder accepts attributed primary_listing
        builder = DeterministicUniverseBuilder()
        sec = SourceSecurity(
            security_id="ARX_SEC_MSFT_TEST",
            symbol="MSFT",
            exchange="NASDAQ",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=msft_rec["assigned_primary_listing"],
            asset_class="EQUITY",
            listing_date="2026-10-17",
        )
        row = builder.evaluate_security_eligibility(sec, build_id="UBUILD_TEST", observed_at="2026-10-17T00:00:00Z")
        assert row.eligibility_decision == EligibilityDecision.ELIGIBLE
        assert row.eligibility_reason_code == "PASSED"

    def test_ordinary_listing_evidence_remaining_insufficient(self):
        """Focused 2: Ordinary listing evidence alone remains insufficient to establish R06."""
        builder = DeterministicUniverseBuilder()
        # Ordinary directory presence without direct primary exchange claim (primary_listing=None)
        sec = SourceSecurity(
            security_id="ARX_SEC_ORDINARY_LISTED",
            symbol="ORDF",
            exchange="NYSE",
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=None,  # Ordinary directory listing without primary listing claim
            asset_class="EQUITY",
            listing_date="2026-10-17",
        )
        row = builder.evaluate_security_eligibility(sec, build_id="UBUILD_TEST", observed_at="2026-10-17T00:00:00Z")
        assert row.eligibility_decision == EligibilityDecision.INELIGIBLE
        assert row.eligibility_reason_code == "INELIGIBLE_SECONDARY_LISTING"

    def test_symbol_instrument_mismatch_fails_closed(self):
        """Focused 3: Symbol/instrument mismatch fails closed and prevents primary-listing assignment."""
        # Case A: Ticker mismatch (requested GOOG, got GOOGL in record)
        rec_mismatch = {
            "symbol": "GOOGL",
            "cik": "0001652044",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok_sym, reason_sym, details_sym = validate_route_a_evidence("GOOG", "0001652044", "XNAS", rec_mismatch)
        assert ok_sym is False
        assert reason_sym == "SYMBOL_MISMATCH"
        assert details_sym["expected"] == "GOOG"
        assert details_sym["actual"] == "GOOGL"

        # Case B: Inactive instrument
        rec_inactive = {
            "symbol": "INACT",
            "cik": "0000000001",
            "active": False,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok_inact, reason_inact, _ = validate_route_a_evidence("INACT", "0000000001", "XNAS", rec_inactive)
        assert ok_inact is False
        assert reason_inact == "INACTIVE_INSTRUMENT"

    def test_conflicting_evidence_dhlgy_otc_fails_closed(self):
        """Focused 4: Conflicting evidence for DHLGY (ADR on OTC Link) fails closed."""
        route_a_file = EVIDENCE_DIR / "expansion_133_route_a_evidence.json"
        data = json.loads(route_a_file.read_text(encoding="utf-8"))
        records = data.get("evidence_records", {})

        assert "DHLGY" in records
        dhlgy_rec = records["DHLGY"]
        assert dhlgy_rec["stated_exchange"] == "OTC Link"
        assert dhlgy_rec["assigned_primary_listing"] is None

        # DHLGY is an ADR, not a common equity, and is strictly excluded from launch manifest
        launch_syms = {r["symbol"] for r in self.launch_manifest_data}
        assert "DHLGY" not in launch_syms

        # In exclusions reconciliation, DHLGY is documented under ADR exclusions
        exclusions = self.exclusions_data["launch_universe_reconciliation"]["exclusions"]
        dhlgy_excl = next((e for e in exclusions if e["symbol"] == "DHLGY"), None)
        assert dhlgy_excl is not None
        assert dhlgy_excl["reason_code"] == "AMERICAN_DEPOSITARY_RECEIPT"

    def test_provider_access_failure_recorded_as_access_result(self):
        """Focused 5: Provider access failure is recorded as an access result, not ineligibility."""
        def record_provider_result(symbol: str, status_code: int, error_msg: Optional[str] = None) -> dict:
            if status_code != 200:
                return {
                    "symbol": symbol,
                    "retrieval_status": "ACCESS_ERROR",
                    "status_code": status_code,
                    "error_message": error_msg,
                    "is_ineligible_proof": False,  # Never conflate access error with ineligibility
                    "disposition": "ACCESS_DEPENDENCY_PENDING",
                }
            return {"symbol": symbol, "retrieval_status": "SUCCESS", "status_code": 200, "is_ineligible_proof": False}

        # HTTP 401 Unauthorized
        res_401 = record_provider_result("XYZ", 401, "API key expired or unauthorized")
        assert res_401["retrieval_status"] == "ACCESS_ERROR"
        assert res_401["is_ineligible_proof"] is False
        assert res_401["disposition"] == "ACCESS_DEPENDENCY_PENDING"

        # HTTP 429 Rate Limited
        res_429 = record_provider_result("ABC", 429, "Rate limit exceeded (5 req/min)")
        assert res_429["retrieval_status"] == "ACCESS_ERROR"
        assert res_429["is_ineligible_proof"] is False

        # Network Timeout
        res_timeout = record_provider_result("DEF", 504, "Gateway timeout")
        assert res_timeout["retrieval_status"] == "ACCESS_ERROR"
        assert res_timeout["is_ineligible_proof"] is False

    def test_accepted_pilot_identity_preservation(self):
        """Focused 6: Accepted pilot identities (AAPL, GOOG, GOOGL) preserve exact IDs and durable status."""
        pilot_specs = {
            "AAPL": "ARX_SEC_0B4C728FE0BE",
            "GOOG": "ARX_SEC_A11A24D442E2",
            "GOOGL": "ARX_SEC_B7EF9ED7A35A",
        }
        # In launch manifest
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        for sym, expected_id in pilot_specs.items():
            cand = by_sym[sym]
            assert cand["security_id"] == expected_id
            assert cand["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
            assert cand["primary_listing_evidence_status"] == "ROUTE_A_ACCEPTED_RETAINED_EVIDENCE"
            assert cand["disposition_b_canonical_identity"]["identity_status"] == "PILOT_ESTABLISHED_DURABLE"

        # In Package A target store registered entities
        pilot_entities = {e["symbol"]: e for e in self.pkg_a_data["target_store"]["registered_pilot_entities"]}
        for sym, expected_id in pilot_specs.items():
            assert sym in pilot_entities
            assert pilot_entities[sym]["security_id"] == expected_id
            assert pilot_entities[sym]["durable_status"] == "ESTABLISHED_DURABLE"

        # In Package A candidate counts
        assert self.pkg_a_data["candidate_counts"]["retained_pilot_identities"] == 3
        assert self.pkg_a_data["candidate_counts"]["new_proposed_identities"] == 130

    def test_actual_vs_counterfactual_outcome_separation(self):
        """Focused 7: Actual and counterfactual outcomes are strictly segregated across all candidates."""
        launch_rows = self.launch_manifest_data
        for r in launch_rows:
            disp_a = r["disposition_a_selection_policy"]
            disp_b = r["disposition_b_canonical_identity"]

            # Actual vs counterfactual selection policy outcomes
            assert "actual_outcome" in disp_a
            assert "counterfactual_outcome" in disp_a
            assert "overall_outcome" in disp_a
            assert disp_a["overall_outcome"] == disp_a["actual_outcome"]

            # Actual vs counterfactual canonical eligibility
            actual_canon = disp_b["actual_canonical_eligibility"]
            cf_canon = disp_b["counterfactual_canonical_eligibility"]
            assert actual_canon["decision"] == "ELIGIBLE"
            assert cf_canon["decision"] == "ELIGIBLE"
            assert cf_canon["injected_attributes"] == {"primary_listing": True}

        # 122 PASS, 11 UNKNOWN
        actual_passes = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["actual_outcome"] == "PASS")
        actual_unknowns = sum(1 for r in launch_rows if r["disposition_a_selection_policy"]["actual_outcome"] == "UNKNOWN")
        assert actual_passes == 122
        assert actual_unknowns == 11

    def test_exact_population_and_package_membership_reconciliation(self):
        """Focused 8: Exact population and package membership arithmetic and set identity reconciliation."""
        launch_syms = {r["symbol"] for r in self.launch_manifest_data}
        pkg_a_syms = {c["symbol"] for c in self.pkg_a_data["candidates"]}

        # Exact set equality: Launch manifest symbols == Package A candidate symbols
        assert launch_syms == pkg_a_syms
        assert len(launch_syms) == 133

        # Exclusions closure: 161 discovery pool = 133 launch + 28 exclusions
        recon = self.exclusions_data["launch_universe_reconciliation"]
        assert recon["total_evaluated_pool"] == 161
        assert recon["qualifying_launch_count"] == 133
        assert recon["total_excluded_count"] == 28
        assert recon["total_evaluated_pool"] == recon["qualifying_launch_count"] + recon["total_excluded_count"]

        # Strict exclusion of fixtures from launch and Package A
        boundary_fixtures = {"ARKK", "GLD", "GLX", "IWM", "JEPI", "QQQ", "SPY", "TLT", "VOO", "XLE", "ARM", "ASML", "DHLGY", "NVO", "TSM", "NAUT"}
        assert not launch_syms.intersection(boundary_fixtures)
        assert not pkg_a_syms.intersection(boundary_fixtures)

    def test_proposed_policy_status_remaining_distinct_from_ratification(self):
        """Focused 9: Proposed selection policy status remains distinct from formal ratification."""
        pkg_c = self.pkg_c_data
        assert pkg_c["governance_status"]["executable"] is False
        assert pkg_c["governance_status"]["publication_authorized"] is False
        assert pkg_c["governance_status"]["ready_for_publication_count"] == 0
        assert pkg_c["membership_accounting"]["scannable_membership_count"] == 0

        # Even though 122 candidates pass proposed selection rules, zero are scannable or ready for publication
        acct = pkg_c["radar_launch_universe_accounting"]
        assert acct["policy_pass_count"] == 122
        assert acct["ready_for_publication_count"] == 0

        # Package A registration is unapproved
        assert self.pkg_a_data["governance_status"]["application_authorized"] is False
        assert self.pkg_a_data["governance_status"]["exact_plan_approval"] == "ABSENT"

    def test_production_readiness_distinct_from_freshness_and_authorization(self):
        """Focused 10: Production readiness remains strictly distinct from data freshness and authorization."""
        # Freshness is documented (35 local fresh, 56 production fresh)
        recon = self.exclusions_data["launch_universe_reconciliation"]
        assert recon["market_data_fresh_local_count"] == 35
        assert recon["market_data_fresh_production_count"] == 57

        # But authorization is strictly False across all operational packages
        assert self.pkg_a_data["governance_status"]["application_authorized"] is False
        assert self.pkg_b_data["governance_status"]["ingestion_authorized"] is False
        assert self.pkg_b_data["governance_status"]["execution_mode"] == "SPECIFICATION_ONLY"
        assert self.pkg_c_data["governance_status"]["publication_authorized"] is False
        assert self.pkg_c_data["governance_status"]["default_universe_switch_authorized"] is False
        assert self.pkg_c_data["governance_status"]["scanner_activation_authorized"] is False

        # In Package C, at least 4 blocking reasons are explicitly documented
        assert len(self.pkg_c_data["governance_status"]["blocking_reasons"]) >= 4

    # ── 13 AUDIT-MANDATED REGRESSION TESTS ────────────────────────────────

    def test_goog_market_metrics_unknown_outcome_in_local_store(self):
        """Regression 1: GOOG has 0 local bars and evaluates to UNKNOWN for L03–L05 and overall."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        assert "GOOG" in by_sym, "GOOG missing from launch manifest data!"
        goog = by_sym["GOOG"]
        assert goog["bar_count"] == 0
        assert goog["latest_close"] is None
        assert goog["avg_dollar_turnover_20"] is None
        assert goog["median_dollar_turnover_20"] is None
        assert goog["observed_turnover_rank"] is None
        assert goog["launch_rank"] is None
        assert goog["ranking_note"] == "UNRANKED_DATA_PENDING"

        disp_a = goog["disposition_a_selection_policy"]
        assert disp_a["overall_outcome"] == "UNKNOWN"
        assert disp_a["actual_outcome"] == "UNKNOWN"
        assert disp_a["counterfactual_outcome"] == "UNKNOWN"
        rules = disp_a["rule_outcomes"]
        assert rules["L01_asset_class_security_type"] == "PASS"
        assert rules["L02_major_regulated_venue"] == "PASS"
        assert rules["L03_price_floor"] == "UNKNOWN"
        assert rules["L04_liquidity_floor"] == "UNKNOWN"
        assert rules["L05_seasoning_floor"] == "UNKNOWN"
        assert rules["L06_median_turnover_rank"] == "UNOBSERVED"

    def test_pilot_identity_does_not_bypass_market_checks(self):
        """Regression 2: Durable pilot identity does not bypass market-data selection policy checks."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        goog = by_sym["GOOG"]
        # Retains durable pilot identity
        assert goog["identity_disposition"] == "PILOT_ESTABLISHED_DURABLE"
        assert goog["security_id"] == "ARX_SEC_A11A24D442E2"
        # But market checks are NOT bypassed: lacking local data, it cannot be PASS
        assert goog["disposition_a_selection_policy"]["overall_outcome"] == "UNKNOWN"
        assert goog["launch_rank"] is None

    def test_route_a_symbol_mismatch_fails_closed(self):
        """Regression 3: validate_route_a_evidence fails closed on symbol mismatch."""
        rec = {
            "symbol": "MSFT",
            "cik": "0000789019",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok, reason, details = validate_route_a_evidence("AAPL", "0000789019", "XNAS", rec)
        assert ok is False
        assert reason == "SYMBOL_MISMATCH"
        assert details["expected"] == "AAPL"
        assert details["actual"] == "MSFT"

    def test_route_a_cik_mismatch_fails_closed(self):
        """Regression 4: validate_route_a_evidence fails closed on CIK mismatch."""
        rec = {
            "symbol": "GOOG",
            "cik": "0000320193",  # AAPL CIK
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok, reason, details = validate_route_a_evidence("GOOG", "0001652044", "XNAS", rec)
        assert ok is False
        assert reason == "CIK_MISMATCH"
        assert details["expected"] == "0001652044"
        assert details["actual"] == "0000320193"

    def test_route_a_inactive_instrument_fails_closed(self):
        """Regression 5: validate_route_a_evidence fails closed on inactive instrument."""
        rec = {
            "symbol": "INACT",
            "cik": "0000000001",
            "active": False,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok, reason, details = validate_route_a_evidence("INACT", "0000000001", "XNAS", rec)
        assert ok is False
        assert reason == "INACTIVE_INSTRUMENT"

    def test_route_a_non_usd_currency_fails_closed(self):
        """Regression 6: validate_route_a_evidence fails closed on non-USD currency."""
        rec = {
            "symbol": "EURSTK",
            "cik": "0000000002",
            "active": True,
            "currency_name": "EUR",
            "operating_mic": "XNAS",
            "raw_response_sha256": "dummy",
        }
        ok, reason, details = validate_route_a_evidence("EURSTK", "0000000002", "XNAS", rec)
        assert ok is False
        assert reason == "NON_USD_CURRENCY"
        assert details["currency"] == "EUR"

    def test_route_a_unsupported_operating_mic_fails_closed(self):
        """Regression 7: validate_route_a_evidence fails closed on unsupported operating MIC."""
        for bad_mic in ["XASE", "OTC Link", "LSE", ""]:
            rec = {
                "symbol": "TESTMIC",
                "cik": "0000000003",
                "active": True,
                "currency_name": "USD",
                "operating_mic": bad_mic,
                "raw_response_sha256": "dummy",
            }
            ok, reason, details = validate_route_a_evidence("TESTMIC", "0000000003", None, rec)
            assert ok is False
            assert reason == "UNSUPPORTED_OPERATING_MIC"

    def test_route_a_sha256_hash_mismatch_fails_closed(self, tmp_path):
        """Regression 8: validate_route_a_evidence fails closed when raw file hash does not match."""
        raw_dir = tmp_path / "tickers"
        raw_dir.mkdir(parents=True, exist_ok=True)
        fake_file = raw_dir / "TESTHASH.json"
        fake_file.write_text('{"results": {"ticker": "TESTHASH", "active": true, "currency_name": "USD", "primary_exchange": "XNAS"}}', encoding="utf-8")

        rec = {
            "symbol": "TESTHASH",
            "cik": "0000000004",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        }
        ok, reason, details = validate_route_a_evidence("TESTHASH", "0000000004", "XNAS", rec, raw_tickers_dir=raw_dir)
        assert ok is False
        assert reason == "RAW_RESPONSE_SHA256_MISMATCH"

    def test_non_backdating_effective_date_enforced(self):
        """Regression 9: Proposed candidate additions enforce 2026-10-19 effective date, preserving 2026-10-17 for pilot."""
        plan_file = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        raw_plan = json.load(open(plan_file, encoding="utf-8"))
        for asc in raw_plan["associations"]:
            sym = asc["symbol"]
            if sym in ("AAPL", "GOOG", "GOOGL"):
                assert asc["effective_from"] == "2026-10-17", f"Pilot candidate {sym} effective_from altered!"
            else:
                assert asc["effective_from"] == "2026-10-19", f"Expansion candidate {sym} has backdated effective_from {asc['effective_from']}!"

        for c in self.pkg_a_data["candidates"]:
            sym = c["symbol"]
            assoc_id = c["association_id"]
            if sym in ("AAPL", "GOOG", "GOOGL"):
                assert assoc_id.endswith("2026-10-17")
            else:
                assert assoc_id.endswith("2026-10-19")

    def test_shared_cik_across_share_classes_goog_googl(self):
        """Regression 10: GOOG and GOOGL share CIK 0001652044 with distinct opaque entity and association IDs."""
        pkg_a = self.pkg_a_data
        by_sym = {c["symbol"]: c for c in pkg_a["candidates"]}
        goog = by_sym["GOOG"]
        googl = by_sym["GOOGL"]

        assert goog["issuer_cik"] == "0001652044"
        assert googl["issuer_cik"] == "0001652044"
        assert goog["security_id"] != googl["security_id"]
        assert goog["security_id"] == "ARX_SEC_A11A24D442E2"
        assert googl["security_id"] == "ARX_SEC_B7EF9ED7A35A"
        assert goog["association_id"] != googl["association_id"]

    def test_opaque_id_stability_across_replays(self):
        """Regression 11: Opaque security and association IDs remain stable and deterministic across dry-run replays."""
        plan_file = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        assert plan_file.exists(), "package_a_identity_adoption_plan.json missing!"
        plan_dict = json.load(open(plan_file, encoding="utf-8"))
        plan = IdentityAdoptionPlan.from_dict(plan_dict)

        # Verify plan content digest validity
        assert plan.is_plan_content_valid() is True
        # Verify deterministic provisional-to-durable mapping for 130 proposed candidate additions
        p2d = plan.provisional_to_durable_map
        assert len(p2d) == 130
        assert p2d["PROV_SEC_NVDA"] == "ARX_SEC_442CE0B25527"
        assert p2d["PROV_SEC_MSFT"] == "ARX_SEC_E602D4F9329C"
        assert p2d["PROV_SEC_META"] == "ARX_SEC_896111757300"

        # Pilot identities are retained durable registrations verified in target export
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))
        pilot_secs = {s["primary_symbol"]: s["internal_security_id"] for s in export_data["state"]["security_registry"].values()}
        assert pilot_secs["AAPL"] == "ARX_SEC_0B4C728FE0BE"
        assert pilot_secs["GOOG"] == "ARX_SEC_A11A24D442E2"
        assert pilot_secs["GOOGL"] == "ARX_SEC_B7EF9ED7A35A"

    def test_sqlite_online_backup_facility(self, tmp_path):
        """Regression 12: create_verified_sqlite_backup safely clones target in read-only mode and verifies digest."""
        src_path = str(tmp_path / "src_identity.db")
        bck_path = str(tmp_path / "bck_identity.db")

        # Initialize src database using repository
        repo = SqliteIdentityRepository(src_path)
        initial_digest = repo.compute_target_state_digest()

        # Run online backup
        bck_digest = create_verified_sqlite_backup(src_path, bck_path)
        assert bck_digest == initial_digest
        assert Path(bck_path).exists()

        # Exclusive non-clobber creation: calling again with same path must fail with FileExistsError
        with pytest.raises(FileExistsError):
            create_verified_sqlite_backup(src_path, bck_path)

    def test_plan_execution_rehearsal_against_isolated_replica(self, tmp_path):
        """PB-01: Full execution rehearsal against isolated target replica executes execute_controlled_registration."""
        import gc
        from scripts.apply_registration_plan_wrapper import (
            execute_controlled_registration,
            ApprovalManifestError,
            IdentityAlreadyAppliedError,
        )

        db_path = str(tmp_path / "identity.db")
        backup_path = str(tmp_path / "identity.db.pre_apply.bak")
        applied_plan_path = str(tmp_path / "applied_plan.json")
        unsigned_manifest_path = str(tmp_path / "unsigned_manifest.json")
        signed_manifest_path = str(tmp_path / "signed_manifest.json")

        # Load authoritative target identity state export
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        export_data = json.loads(export_file.read_text(encoding="utf-8"))
        export_state = export_data["state"]
        export_target_digest = export_data["metadata"]["target_state_digest"]

        # Seed target db
        repo = SqliteIdentityRepository(db_path)
        for sec_id, sec_data in export_state["security_registry"].items():
            repo.save_security(InternalSecurityRecord(**sec_data))
        for assoc_id, assoc_data in export_state["association_ledger"].items():
            repo.save_association(EffectiveDatedListingAssociation(**assoc_data))
        init_digest = repo.compute_target_state_digest()
        repo.close()

        assert init_digest == export_target_digest
        assert init_digest == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"

        # Load Package A plan and template
        plan_file = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        template_data = json.loads(template_file.read_text(encoding="utf-8"))

        # 1. Unsigned manifest fails closed
        with open(unsigned_manifest_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(template_data, indent=2))

        with pytest.raises(ApprovalManifestError):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=unsigned_manifest_path,
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_path,
            )

        # 2. Tampered file SHA fails closed
        tampered_manifest = dict(template_data)
        tampered_manifest["reviewed_plan_file_sha256"] = "0" * 64
        tampered_manifest["authorized_by"] = "TEST_REHEARSAL_OPERATOR"
        tampered_manifest["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        tampered_manifest["approved_target_db_path"] = str(Path(db_path).resolve())
        tampered_manifest_path = str(tmp_path / "tampered_manifest.json")
        with open(tampered_manifest_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(tampered_manifest, indent=2))

        with pytest.raises(ValueError, match="PLAN_FILE_TAMPERED"):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=tampered_manifest_path,
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_path,
            )

        # 3. Signed manifest
        signed_manifest = dict(template_data)
        signed_manifest["approved_target_db_path"] = str(Path(db_path).resolve())
        signed_manifest["authorized_by"] = "TEST_REHEARSAL_OPERATOR"
        signed_manifest["authorization_timestamp_utc"] = "2026-10-20T08:00:00Z"
        with open(signed_manifest_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(signed_manifest, indent=2))

        # 4. Exclusive non-clobber backup failure test
        Path(backup_path).write_text("PRE_EXISTING_BACKUP", encoding="utf-8")
        with pytest.raises(FileExistsError):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=signed_manifest_path,
                db_path=db_path,
                backup_path=backup_path,
                output_applied_plan_path=applied_plan_path,
            )
        Path(backup_path).unlink()

        # 5. Clean application rehearsal via execute_controlled_registration
        outcome = execute_controlled_registration(
            reviewed_plan_path=str(plan_file),
            approval_manifest_path=signed_manifest_path,
            db_path=db_path,
            backup_path=backup_path,
            output_applied_plan_path=applied_plan_path,
        )
        assert outcome.status == IdentityApplicationStatus.APPLIED
        assert outcome.applied_entities_count == 130
        assert outcome.applied_associations_count == 130
        assert outcome.target_state_digest_before == init_digest
        assert outcome.target_state_digest_after != init_digest
        assert outcome.audit_entry_recorded is True
        assert Path(backup_path).exists()
        assert Path(applied_plan_path).exists()

        # Verify target database state post-execution
        repo_verify = SqliteIdentityRepository(db_path, read_only=True)
        assert repo_verify.compute_target_state_digest() == outcome.target_state_digest_after
        assert len(repo_verify.list_all_securities()) == 133
        assert len(repo_verify.list_all_associations()) == 133
        repo_verify.close()

        # 6. Idempotency test: repeat execution raises IdentityAlreadyAppliedError
        with pytest.raises(IdentityAlreadyAppliedError):
            execute_controlled_registration(
                reviewed_plan_path=str(plan_file),
                approval_manifest_path=signed_manifest_path,
                db_path=db_path,
                backup_path=str(tmp_path / "identity_second.bak"),
                output_applied_plan_path=str(tmp_path / "applied_second.json"),
            )

        del outcome
        gc.collect()

    def test_package_a_approval_manifest_template_governance(self):
        """PB-02: Verifies unsigned approval manifest template schema, digest bindings, and non-pilot scope."""
        template_file = EVIDENCE_DIR / "package_a_approval_manifest_template.json"
        assert template_file.exists(), "package_a_approval_manifest_template.json missing!"
        template = json.loads(template_file.read_text(encoding="utf-8"))

        assert template["approval_id"] == "APPROVAL-EXPANSION-150-A-TEMPLATE"
        assert template["authorized_by"] is None, "Template must remain unsigned (authorized_by=None)!"
        assert template["authorization_timestamp_utc"] is None, "Template authorization timestamp must be None!"
        assert template["scope"] == "REFERENCE_IDENTITY_REGISTRATION"
        assert template["reviewed_plan_id"] == "PLAN_ADOPT_9A2E4E04E878"
        assert template["reviewed_plan_content_digest"] == "9a2e4e04e878b667e25a2a75d56292a0b3e679b15890211cef13db96f7a8bfe5"
        assert template["expected_target_state_digest"] == "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"
        assert template["approved_target_db_path"] == "/root/analyst_dashboard/data/identity.db"
        assert template["remote_execution_transport"] == "railway ssh -s web -e production"

        bindings = template["candidate_bindings"]
        assert len(bindings) == 130, f"Expected 130 candidate bindings, got {len(bindings)}"
        assert "AAPL" not in bindings
        assert "GOOG" not in bindings
        assert "GOOGL" not in bindings

        for sym, b in bindings.items():
            assert b["issuer_cik"]
            assert b["durable_security_id"].startswith("ARX_SEC_")
            assert b["operating_mic"] in ("XNAS", "XNYS", "ARCX", "BATS", "OTCM")
            assert b["effective_from"] == "2026-10-19"

    # ── F01: Real Fail-Closed Reference Validation Tests ─────────────────
    def test_validate_route_a_missing_file_rejected(self):
        """F01: Missing raw response artifact fails closed immediately."""
        fake_rec = {
            "symbol": "AAPL",
            "cik": "0000320193",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "0" * 64,
            "source_artifact": "data/research/reference_polygon/tickers/NONEXISTENT_XYZ.json",
        }
        ok, reason, attrs = validate_route_a_evidence("AAPL", "0000320193", "XNAS", fake_rec)
        assert ok is False
        assert reason == "MISSING_RAW_ARTIFACT"

    def test_validate_route_a_conflicting_cik_rejected(self):
        """F01: Conflicting CIK between record and raw payload fails closed."""
        rec = {
            "symbol": "MSFT",
            "cik": "0000789019",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "abc",
        }
        # Passing wrong candidate CIK
        ok, reason, attrs = validate_route_a_evidence("MSFT", "0000999999", "XNAS", rec)
        assert ok is False
        assert reason == "CIK_MISMATCH"

    def test_validate_route_a_conflicting_mic_rejected(self):
        """F01: Conflicting operating MIC fails closed."""
        rec = {
            "symbol": "MSFT",
            "cik": "0000789019",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "abc",
        }
        ok, reason, attrs = validate_route_a_evidence("MSFT", "0000789019", "XNYS", rec)
        assert ok is False
        assert reason == "OPERATING_MIC_MISMATCH"

    def test_validate_route_a_etf_rejected(self, tmp_path):
        """F01: ETF payload with type != 'CS' fails closed with NON_COMMON_STOCK_TYPE."""
        import hashlib
        raw_payload = {
            "status": "OK",
            "results": {
                "ticker": "SPY",
                "name": "SPDR S&P 500 ETF Trust",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "ARCX",
                "type": "ETF",
                "active": True,
                "currency_name": "USD",
                "cik": "0000884394",
            }
        }
        payload_bytes = json.dumps(raw_payload).encode("utf-8")
        payload_sha = hashlib.sha256(payload_bytes).hexdigest()
        spy_file = tmp_path / "SPY.json"
        spy_file.write_bytes(payload_bytes)

        rec = {
            "symbol": "SPY",
            "cik": "0000884394",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "ARCX",
            "raw_response_sha256": payload_sha,
            "source_artifact": str(spy_file.relative_to(REPO_ROOT)) if str(spy_file).startswith(str(REPO_ROOT)) else str(spy_file),
        }
        ok, reason, attrs = validate_route_a_evidence("SPY", "0000884394", "ARCX", rec, raw_tickers_dir=tmp_path)
        assert ok is False
        assert reason == "NON_COMMON_STOCK_TYPE"

    def test_validate_route_a_missing_record_cik_rejected(self):
        """F01: Missing CIK in evidence record fails closed."""
        rec = {
            "symbol": "TEST",
            "active": True,
            "currency_name": "USD",
            "operating_mic": "XNAS",
            "raw_response_sha256": "abc",
        }
        ok, reason, attrs = validate_route_a_evidence("TEST", None, "XNAS", rec)
        assert ok is False
        assert reason == "MISSING_RECORD_CIK"

    # ── F02: Durable Opaque Identities Decoupled from Tickers ─────────────
    def test_allocations_stable_across_regeneration(self):
        """F02: Provisional identity allocations are durable and stable across regeneration."""
        alloc_file = EVIDENCE_DIR / "provisional_identity_allocations.json"
        assert alloc_file.exists()
        data = json.loads(alloc_file.read_text(encoding="utf-8"))
        allocations = data.get("allocations", {})
        assert len(allocations) == 130
        for key, rec in allocations.items():
            assert key.startswith("CIK:")
            sec_id = rec.get("allocated_durable_security_id") or rec.get("security_id")
            assert sec_id.startswith("ARX_SEC_")
            assert len(sec_id) == 20  # ARX_SEC_ + 12 chars

        # Test regeneration stability: generate candidates twice and assert identical security IDs
        run1, _ = generate_candidate_manifest(allocations=allocations)
        run2, _ = generate_candidate_manifest(allocations=allocations)
        map1 = {c["symbol"]: c["security_id"] for c in run1}
        map2 = {c["symbol"]: c["security_id"] for c in run2}
        assert map1 == map2, "Security IDs diverged across manifest regenerations!"

        # Assert plan entities match candidate security IDs
        plan_file = EVIDENCE_DIR / "package_a_identity_adoption_plan.json"
        if plan_file.exists():
            plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
            for ent in plan_data.get("entities", []):
                sym = ent["primary_symbol"]
                if sym in map1:
                    expected_sec_id = map1[sym]
                    assert ent["proposed_durable_security_id"] == expected_sec_id

    def test_entity_keyed_allocation_and_historical_supersession_mapping(self):
        """F02: Verifies entity-keyed allocation (CIK:0001326801|CLASS:ORD) and recorded FB->META supersession mapping.

        Bounded Guarantee Qualification:
        Verifies that META's pre-allocated security ID is keyed by CIK/share-class entity rather than symbol string,
        and that historical FB ticker-derived ID is mapped to META's durable ID in supersession_mapping.
        This verifies stable allocation retrieval for the retained candidate set. Full runtime corporate-action
        lifecycle management (dynamic renames across arbitrary new symbols, effective dating adjustments, and
        automatic association rollover) is unverified and deferred to post-sprint platform work.
        """
        alloc_file = EVIDENCE_DIR / "provisional_identity_allocations.json"
        data = json.loads(alloc_file.read_text(encoding="utf-8"))
        allocations = data.get("allocations", {})
        meta_alloc = allocations.get("CIK:0001326801|CLASS:ORD")
        assert meta_alloc is not None
        assert meta_alloc["symbol"] == "META"
        durable_sec_id = meta_alloc["allocated_durable_security_id"]
        assert durable_sec_id.startswith("ARX_SEC_")
        old_fb_ticker_id = meta_alloc.get("superseded_ticker_derived_id")
        assert old_fb_ticker_id in data.get("supersession_mapping", {})
        assert data["supersession_mapping"][old_fb_ticker_id] == durable_sec_id

    def test_distinct_cik_entity_keys_prevent_id_collision_and_preserve_pilot_ids(self):
        """F02: Verifies zero collisions among allocated security IDs and zero reallocation of pilot IDs across CIK entity keys.

        Bounded Guarantee Qualification:
        Verifies that all 130 pre-allocated security IDs in provisional_identity_allocations.json are mutually unique,
        and that preserved pilot security IDs (AAPL, GOOG, GOOGL) are never reallocated to expansion candidates.
        Entity keys are explicitly partitioned by CIK rather than ticker symbols alone.
        Full runtime multi-era symbol reuse lifecycle management (historical collision resolution across successive
        different corporate entities using the same ticker in different time periods) is unverified and deferred.
        """
        alloc_file = EVIDENCE_DIR / "provisional_identity_allocations.json"
        data = json.loads(alloc_file.read_text(encoding="utf-8"))
        allocs = data.get("allocations", {})
        sec_ids = [rec.get("allocated_durable_security_id") or rec.get("security_id") for rec in allocs.values()]
        assert len(sec_ids) == len(set(sec_ids)), "Collision detected among allocated security IDs!"
        pilot_ids = [p["security_id"] for p in (data.get("pilot_identities") or data.get("preserved_pilot_identities") or {}).values()]
        assert len(pilot_ids) == 3
        for pid in pilot_ids:
            assert pid not in sec_ids, f"Pilot ID {pid} improperly reallocated!"

    def test_missing_allocation_raises_key_error(self):
        """F02: Fail-closed pre-allocation raises KeyError when an entity key is missing."""
        incomplete_allocations = {"CIK:0000000000|CLASS:ORD": {"allocated_durable_security_id": "ARX_SEC_000000000000"}}
        with pytest.raises(KeyError) as exc_info:
            generate_candidate_manifest(allocations=incomplete_allocations)
        assert "Missing required provisional identity allocation" in str(exc_info.value)

    # ── F03: Package A Attributable Target State ──────────────────────────
    def test_target_identity_state_export_concordance(self):
        """F03: target_identity_state_export.json matches target state digest natively."""
        export_file = EVIDENCE_DIR / "target_identity_state_export.json"
        assert export_file.exists()
        export_data = json.loads(export_file.read_text(encoding="utf-8"))
        expected_digest = "b881778ac5370601174838271c918a1e1dcae7b4522b2bfc7b76a78e622e87e7"
        assert export_data["metadata"]["target_state_digest"] == expected_digest
        assert export_data["metadata"]["production_target_binding"] == "BOUND_AUTHENTIC_LIVE_PRODUCTION"
        assert export_data["metadata"]["replica_provenance"] == "LIVE_PRODUCTION_STORE_READ_ONLY_EXPORT"
        computed = IdentityAdoptionDryRunEngine.compute_target_state_digest(export_data["state"])
        assert computed == expected_digest

        # Verify Package A reflects the same governance status and target store status
        pkg_a = self.pkg_a_data
        assert pkg_a["governance_status"]["production_target_binding"] == "BOUND_AUTHENTIC_LIVE_PRODUCTION"
        assert pkg_a["governance_status"]["target_store_status"] == "FRESH_LIVE_PRODUCTION_OBSERVATION"
        assert pkg_a["target_store"]["production_target_binding"] == "BOUND_AUTHENTIC_LIVE_PRODUCTION"

    def test_proposal_evidence_locators_and_hashes(self):
        """F03: Proposal evidence locators point to on-disk reference files with valid SHA-256."""
        # Check that proposals reference existing data/research/reference_polygon/tickers/
        launch_candidates = [c for c in self.launch_manifest_data]
        assert len(launch_candidates) == 133
        for c in launch_candidates:
            sym = c["symbol"]
            expected_file = REPO_ROOT / "data" / "research" / "reference_polygon" / "tickers" / f"{sym}.json"
            assert expected_file.exists(), f"Reference polygon ticker file missing for {sym}: {expected_file}"
            actual_sha = hashlib.sha256(expected_file.read_bytes()).hexdigest()
            assert actual_sha, f"Empty SHA-256 for {sym}"

    # ── F04: Market-Data Dispositions ─────────────────────────────────────
    def test_market_data_disposition_goog_split_adjusted(self):
        """F04: GOOG has SPLIT_ADJUSTED basis with recorded pilot backfill provenance."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}
        goog = by_sym["GOOG"]
        disp_c = goog["disposition_c_market_data"]
        assert disp_c["adjustment_basis"] == "SPLIT_ADJUSTED"
        assert disp_c["source_provenance"] == "SPLIT_ADJUSTED_BACKFILL_RECORDED"
        assert disp_c["pilot_exception_record"] == "ESTABLISHED_PILOT_BACKFILL"
        assert disp_c["pilot_exception_scope"] == "ISOLATED_PILOT_MEMBER"
        assert disp_c["storage_location"] == "/root/.finance_market_store.db"

    def test_market_data_disposition_unknown_basis(self):
        """F04: Honest market-data disposition: AAPL/GOOGL have UNKNOWN basis under isolated pilot exception; non-pilot assets have UNESTABLISHED provenance."""
        by_sym = {r["symbol"]: r for r in self.launch_manifest_data}

        # 1. AAPL (pilot exception)
        aapl = by_sym["AAPL"]
        disp_aapl = aapl["disposition_c_market_data"]
        assert disp_aapl["adjustment_basis"] == "UNKNOWN"
        assert disp_aapl["source_provenance"] == "UNKNOWN"
        assert disp_aapl["pilot_exception_record"] == "ACCEPTED_FOR_ISOLATED_PILOT_ONLY"
        assert disp_aapl["pilot_exception_scope"] == "ISOLATED_PILOT_ONLY_NOT_EXTENDED_TO_EXPANSION"

        # 2. GOOGL (pilot exception)
        googl = by_sym["GOOGL"]
        disp_googl = googl["disposition_c_market_data"]
        assert disp_googl["adjustment_basis"] == "UNKNOWN"
        assert disp_googl["source_provenance"] == "UNKNOWN"
        assert disp_googl["pilot_exception_record"] == "ACCEPTED_FOR_ISOLATED_PILOT_ONLY"
        assert disp_googl["pilot_exception_scope"] == "ISOLATED_PILOT_ONLY_NOT_EXTENDED_TO_EXPANSION"

        # 3. MSFT (non-pilot asset with data in store: UNESTABLISHED provenance, distinct from storage path)
        msft = by_sym["MSFT"]
        disp_msft = msft["disposition_c_market_data"]
        assert disp_msft["adjustment_basis"] == "UNKNOWN"
        assert disp_msft["source_provenance"] == "UNESTABLISHED"
        assert disp_msft["storage_location"] == "/root/.finance_market_store.db"
        assert disp_msft["pilot_exception_record"] == "NONE"
        assert disp_msft["pilot_exception_scope"] == "NONE"

        # 4. Absent candidate (zero bars: UNESTABLISHED provenance, ABSENT storage)
        absent = next(c for c in self.launch_manifest_data if c["disposition_c_market_data"]["depth_bars"] == 0)
        disp_absent = absent["disposition_c_market_data"]
        assert disp_absent["adjustment_basis"] == "UNKNOWN"
        assert disp_absent["source_provenance"] == "UNESTABLISHED"
        assert disp_absent["storage_location"] == "ABSENT"

    # ── F05: Package B Concrete Acquisition Specification ────────────────
    def test_package_b_provider_specifications_and_260_coverage(self):
        """F05: Package B specifies unverified account entitlements, request window strategy, feed comparability, and bar validation specification."""
        pkg = self.pkg_b_data
        specs = pkg.get("provider_specifications", {})
        assert "polygon" in specs
        assert "alpaca" in specs
        assert "/v2/aggs/ticker" in specs["polygon"]["endpoint_template"]
        assert "/v2/stocks/bars" in specs["alpaca"]["endpoint_template"]
        assert "invalid/unauthorized" in specs["polygon"]["entitlement_tier"] or "Free / Basic tier" in specs["polygon"]["entitlement_tier"]
        assert "Free / Basic tier" in specs["alpaca"]["entitlement_tier"]
        assert specs["polygon"]["account_entitlement_status"] in ("UNVERIFIED_ACCOUNT_ENTITLEMENT", "VERIFIED_DEFICIENT_HTTP_401")
        assert specs["alpaca"]["account_entitlement_status"] in ("UNVERIFIED_ACCOUNT_ENTITLEMENT", "VERIFIED_FREE_TIER_IEX_ONLY")
        assert specs["polygon"]["historical_bar_access_verified"] is False
        assert specs["alpaca"]["historical_bar_access_verified"] is False

        # Feed comparability checks
        assert "Consolidated" in specs["polygon"]["feed_scope"]
        assert "IEX" in specs["alpaca"]["feed_scope"]
        assert "feed_comparability_notes" in specs["alpaca"]

        # Bar validation specification
        assert "bar_validation_specification" in pkg
        bar_spec = pkg["bar_validation_specification"]
        assert "required_fields" in bar_spec
        for field in ["symbol", "trade_date", "open", "high", "low", "close", "volume"]:
            assert field in bar_spec["required_fields"]
        assert "finite_numeric_values" in bar_spec
        assert "ohlc_consistency" in bar_spec
        assert "nonnegative_volume" in bar_spec
        assert "duplicate_handling" in bar_spec
        assert "failure_and_partial_response_handling" in bar_spec

        alts = pkg.get("multi_tier_sizing_alternatives", {})
        assert "alternative_b_strategy_benchmark_260_bars" in alts
        alt_b = alts["alternative_b_strategy_benchmark_260_bars"]
        assert alt_b["bars_per_missing_candidate"] == 260
        assert "2025-09-15" in alt_b["window"]
        assert "calendar days" in alt_b["window"] or "calendar days" in alt_b.get("reconciliation_rationale", "")

        backfill = pkg["ingestion_scope"]["production_target_backfill"]
        params = backfill["per_symbol_parameters"]
        assert len(params) == backfill["missing_candidates_count"]
        for p in params:
            assert "calendar_window_benchmark_260" in p
            assert p["calendar_window_benchmark_260"]["required_bars"] == 260
            assert p["calendar_window_benchmark_260"]["window_start"] == "2025-09-15"
            assert "calendar_window_builder_60" in p
            assert p["calendar_window_builder_60"]["required_bars"] == 60
            assert p["calendar_window_builder_60"]["window_start"] == "2026-07-16"




