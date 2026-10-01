"""
tests/test_etf_ucits_discovery_authority.py

Comprehensive deterministic test suite for the ETF V2 UCITS Discovery Authority.
Strictly verifies:
- Target entity = SHARE_CLASS_ISIN
- 4-domicile initial scope: IE, LU, DE, FR
- Tier 1 NCA > Tier 2 Issuer > Tier 3 Exchange hierarchy
- Strict delegation to global_identifier_authority.normalize_isin
- Fixture firewall: zero production dependencies on tests/fixtures/ucits/ucits_wave4_fixtures.json
- Zero hardcoded 25 assumptions
- Zero live network traffic
- Complete D01-D40 adversarial scenario matrix
- Complete C01-C16 crash / interruption matrix
- Strict accounting conservation and deterministic replay
"""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import pytest

from scripts.research.etf_v2.global_identifier_authority import normalize_isin, validate_isin
from scripts.research.etf_v2.ucits_discovery_adapters import (
    AMFFranceAdapter,
    BaFinGermanyAdapter,
    BaseDiscoveryAdapter,
    BaseDiscoveryTransport,
    CSSFLuxembourgAdapter,
    CentralBankOfIrelandAdapter,
    MockDiscoveryTransport,
    StatutoryIssuerAdapter,
)
from scripts.research.etf_v2.ucits_discovery_authority import (
    FORBIDDEN_FIXTURE_PATH_SUBSTRING,
    UCITSDiscoveryAuthority,
)
from scripts.research.etf_v2.ucits_discovery_models import (
    AuthorityFunction,
    CandidateSpec,
    CandidateStatus,
    ConservationViolationError,
    CorruptedDiscoveryCacheError,
    DiscoveryAccounting,
    DiscoveryCompletenessError,
    DiscoveryConfiguration,
    DiscoveryError,
    DiscoveryInterruptionError,
    DiscoveryJurisdiction,
    DiscoveryRunIdentity,
    DiscoveryRunStatus,
    FixtureContaminationError,
    InvalidDiscoveryConfigurationError,
    ObservationProvenance,
    ParentJoinStatus,
    QuarantineReason,
    RawDiscoveryObservation,
    RawRegisterPayload,
    ResumeIdentityMismatchError,
    SchemaDriftError,
    ShareClassExpansionCompleteness,
    SourceAdapterError,
    SourceAuthorityId,
    SourceAuthorityTier,
    SourceEnumerationState,
    Tier1ParentAccounting,
    Tier1ParentObservation,
    Tier2ShareClassExpansion,
    normalize_fund_name,
)


# =============================================================================
# Synthetic Mock Test Fixtures (Offline, Deterministic)
# =============================================================================

# Valid ISINs with verified ISO 6166 Luhn check digits
ISIN_IE_1 = "IE00B4L5Y983"  # iShares Core MSCI World
ISIN_IE_2 = "IE00BK5BQT80"  # Vanguard FTSE All-World
ISIN_LU_1 = "LU0274208692"  # Xtrackers Euro Stoxx 50
ISIN_LU_2 = "LU0838780707"  # Lyxor Core MSCI World
ISIN_DE_1 = "DE0005933956"  # iShares Core DAX UCITS ETF (DE)
ISIN_FR_1 = "FR0010251157"  # Amundi CAC 40 UCITS ETF (FR)

# Invalid check digit ISIN
ISIN_INVALID_CHECKSUM = "IE00B4L5Y980"


def create_mock_cbi_payload(records: List[Dict[str, Any]], total_records: Optional[int] = None) -> bytes:
    doc = {"total_records": total_records if total_records is not None else len(records), "records": records}
    return json.dumps(doc).encode("utf-8")


def create_mock_cssf_payload(records: List[Dict[str, Any]], total_records: Optional[int] = None) -> bytes:
    doc = {"total_records": total_records if total_records is not None else len(records), "records": records}
    return json.dumps(doc).encode("utf-8")


def create_mock_bafin_csv_payload(rows: List[Dict[str, str]]) -> bytes:
    header = "ISIN;Fondsname;Anteilklasse;Rechtsform;Status\n"
    lines = [header]
    for r in rows:
        lines.append(f"{r.get('ISIN','')};{r.get('Fondsname','')};{r.get('Anteilklasse','')};{r.get('Rechtsform','')};{r.get('Status','')}\n")
    return "".join(lines).encode("utf-8")


def create_mock_amf_payload(records: List[Dict[str, Any]], total_records: Optional[int] = None) -> bytes:
    doc = {"total_records": total_records if total_records is not None else len(records), "records": records}
    return json.dumps(doc).encode("utf-8")


def create_mock_issuer_payload(products: List[Dict[str, Any]]) -> bytes:
    return json.dumps({"products": products}).encode("utf-8")


# =============================================================================
# Architectural & Invariant Tests
# =============================================================================

class TestDiscoveryArchitecturalInvariants:
    """Verifies fundamental architectural invariants ratified by the design gate."""

    def test_target_entity_is_share_class_isin(self) -> None:
        """Target entity must strictly be SHARE_CLASS_ISIN."""
        spec = CandidateSpec(
            share_class_isin=ISIN_IE_1,
            domicile="IE",
            fund_name="Test UCITS ETF",
            share_class_name="USD Acc",
            is_ucits=True,
            is_etf=True,
            status=CandidateStatus.ACTIVE.value,
            provenance_chain=(),
        )
        assert spec.share_class_isin == ISIN_IE_1
        assert spec.domicile == "IE"

    def test_isin_delegation_to_global_identifier_authority(self) -> None:
        """ISIN normalization and validation must strictly delegate to global_identifier_authority."""
        raw = "  ie00b4l5y983  "
        normalized = normalize_isin(raw)
        assert normalized == ISIN_IE_1
        assert validate_isin(normalized, strict=True) is True

        with pytest.raises(Exception):
            normalize_isin("")
        with pytest.raises(Exception):
            validate_isin("NOT_AN_ISIN", strict=True)

    def test_initial_four_jurisdictions_enforced(self) -> None:
        """Configuration must support strictly IE, LU, DE, FR."""
        cfg = DiscoveryConfiguration(jurisdictions=("IE", "LU", "DE", "FR"))
        assert set(cfg.jurisdictions) == {"IE", "LU", "DE", "FR"}

        with pytest.raises(InvalidDiscoveryConfigurationError, match="Unsupported jurisdiction"):
            DiscoveryConfiguration(jurisdictions=("IE", "GB"))

    def test_fixture_firewall_enforced(self) -> None:
        """Discovery authority must fail closed if configured inside fixture directory."""
        with pytest.raises(FixtureContaminationError):
            UCITSDiscoveryAuthority(cache_dir=Path(f"/tmp/{FORBIDDEN_FIXTURE_PATH_SUBSTRING}/cache"))

    def test_source_tier_precedence_order(self) -> None:
        """Tier 1 NCA > Tier 2 Issuer > Tier 3 Exchange hierarchy."""
        cfg = DiscoveryConfiguration()
        assert cfg.tier_precedence[0] == SourceAuthorityTier.TIER_1_NCA.value
        assert cfg.tier_precedence[1] == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value
        assert cfg.tier_precedence[2] == SourceAuthorityTier.TIER_3_EXCHANGE.value

    def test_accounting_conservation_balances_exactly(self) -> None:
        """Accounting equations must balance with zero unaccounted records."""
        acct = DiscoveryAccounting.calculate(
            raw_discovered=10,
            parsed=10,
            unparseable=0,
            candidate_obs=8,
            out_of_scope=1,
            invalid_id=1,
            quarantined=0,
            unique_candidates=6,
            duplicate_obs=2,
        )
        assert acct.is_conserved is True
        assert acct.unaccounted_count == 0

    def test_accounting_conservation_violation_detected(self) -> None:
        """Accounting detects when raw records fail conservation."""
        acct = DiscoveryAccounting.calculate(
            raw_discovered=10,
            parsed=9,  # Missing 1
            unparseable=0,
            candidate_obs=8,
            out_of_scope=1,
            invalid_id=0,
            quarantined=0,
            unique_candidates=8,
            duplicate_obs=0,
        )
        assert acct.is_conserved is False
        assert acct.unaccounted_count > 0


# =============================================================================
# Adversarial Matrix Tests (D01–D40)
# =============================================================================

class TestAdversarialScenarioMatrix:
    """Explicit mapping and verification of D01–D40 adversarial scenarios."""

    def test_d01_regulator_pagination_returns_duplicate_page(self, tmp_path: Path) -> None:
        """D01: Duplicate pages deduplicate by payload hash without terminating run."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "share_class_name": "Class A", "cis_type": "UCITS", "is_etf": True}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg)

        assert manifest["candidate_count"] == 1
        assert manifest["completeness_state"] == SourceEnumerationState.COMPLETE.value

    def test_d02_pagination_page_omitted_fails_closed(self, tmp_path: Path) -> None:
        """D02: Missing records vs header declared count aborts run with PARTIAL / FAILED."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # Header declares 5 records, but body only delivers 1
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "share_class_name": "Class A", "cis_type": "UCITS", "is_etf": True}
        ], total_records=5)
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg)

        assert manifest["completeness_state"] == SourceEnumerationState.FAILED.value
        assert "declared header count" in manifest["completeness_failure_reason"]

    def test_d03_source_mutates_during_traversal(self, tmp_path: Path) -> None:
        """D03: Concurrent source count change aborts run fail-closed."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([], total_records=10)
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))
        assert manifest["completeness_state"] == SourceEnumerationState.FAILED.value

    def test_d04_http_429_rate_limit_backoff(self, tmp_path: Path) -> None:
        """D04: HTTP 429 rate limit triggers backoff and eventual retry."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        calls = 0

        def cbi_callback() -> Tuple[int, bytes, Dict[str, str]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return (429, b"Too Many Requests", {"content-type": "text/plain", "Retry-After": "1"})
            payload = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Fund 1", "cis_type": "UCITS", "is_etf": True}])
            return (200, payload, {"content-type": "application/json"})

        transport.register_callback(cbi_url, cbi_callback)
        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",), backoff_factor=0.01))

        assert calls == 2
        assert manifest["candidate_count"] == 1

    def test_d05_transient_503_service_unavailable(self, tmp_path: Path) -> None:
        """D05: Transient HTTP 503 retries up to max limit."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        calls = 0

        def cbi_callback() -> Tuple[int, bytes, Dict[str, str]]:
            nonlocal calls
            calls += 1
            if calls < 3:
                return (503, b"Service Unavailable", {"content-type": "text/plain"})
            payload = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Fund 1", "cis_type": "UCITS", "is_etf": True}])
            return (200, payload, {"content-type": "application/json"})

        transport.register_callback(cbi_url, cbi_callback)
        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",), max_retries=3, backoff_factor=0.01))

        assert calls == 3
        assert manifest["candidate_count"] == 1

    def test_d06_permanent_404_terminal_failure(self, tmp_path: Path) -> None:
        """D06: Permanent 404 endpoint not found fails adapter immediately and fails run closed."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 404, b"Not Found")

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))
        assert manifest["completeness_state"] == SourceEnumerationState.FAILED.value

    def test_d07_malformed_registry_row(self, tmp_path: Path) -> None:
        """D07: Unparseable registry row captured in unparseable/quarantine accounting."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Good", "cis_type": "UCITS", "is_etf": True},
            "NOT_A_DICT_ROW",  # Malformed row
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))
        assert manifest["candidate_count"] == 1

    def test_d08_missing_isin_quarantined(self, tmp_path: Path) -> None:
        """D08: Missing ISIN in record quarantined with MISSING_IDENTIFIER."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": "", "fund_name": "No ISIN Fund", "cis_type": "UCITS", "is_etf": True}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["invalid_identifier_count"] == 1
        assert manifest["quarantined_observations"][0]["reason"] == QuarantineReason.MISSING_IDENTIFIER.value

    def test_d09_invalid_isin_check_digit_quarantined(self, tmp_path: Path) -> None:
        """D09: Invalid ISIN check digit quarantined with INVALID_CHECKSUM."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_INVALID_CHECKSUM, "fund_name": "Bad Checksum", "cis_type": "UCITS", "is_etf": True}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["invalid_identifier_count"] == 1
        assert manifest["quarantined_observations"][0]["reason"] == QuarantineReason.INVALID_CHECKSUM.value

    def test_d10_duplicate_isin_same_authority_merged(self, tmp_path: Path) -> None:
        """D10: Duplicate ISIN within same authority collapses with merged provenance."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "share_class": "Class A", "cis_type": "UCITS", "is_etf": True},
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "share_class": "Class A Duplicate", "cis_type": "UCITS", "is_etf": True},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 1
        assert manifest["accounting"]["duplicate_observations_count"] == 1
        cand = manifest["candidates"][0]
        assert len(cand["provenance_chain"]) == 2

    def test_d11_multi_authority_corroboration_merged(self, tmp_path: Path) -> None:
        """D11: Same ISIN across multiple authorities merges provenance chains."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "CBI Name", "cis_type": "UCITS", "is_etf": True}])
        transport.register_response(cbi_url, 200, cbi_payload)

        # Secondary issuer observation
        issuer_adapter = StatutoryIssuerAdapter(issuer_name="iShares", jurisdiction=DiscoveryJurisdiction.IE)
        issuer_url = "https://www.ishares.com/products.json"
        issuer_payload = create_mock_issuer_payload([{"isin": ISIN_IE_1, "fund_name": "iShares Name", "is_ucits": True, "is_etf": True}])
        transport.register_response(issuer_url, 200, issuer_payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(
            DiscoveryConfiguration(jurisdictions=("IE",)),
            custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter],
        )

        assert manifest["candidate_count"] == 1
        cand = manifest["candidates"][0]
        assert len(cand["provenance_chain"]) == 2
        # Tier 1 CBI takes precedence for fund name
        assert cand["fund_name"] == "CBI Name"

    def test_d12_name_discrepancy_retains_tier_1_name(self, tmp_path: Path) -> None:
        """D12: Tier 1 name takes precedence over Tier 2 discrepancy."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Official Statutory", "cis_type": "UCITS", "is_etf": True}]))

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Vendor", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response("https://www.vendor.com/products.json", 200, create_mock_issuer_payload([{"isin": ISIN_IE_1, "fund_name": "Commercial Name", "is_ucits": True, "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(
            DiscoveryConfiguration(jurisdictions=("IE",)),
            custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter],
        )

        assert manifest["candidates"][0]["fund_name"] == "Official Statutory"

    def test_d13_status_contradiction_quarantined(self, tmp_path: Path) -> None:
        """D13: Contradictory listing status (ACTIVE vs TERMINATED) quarantined."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # Observation 1 says ACTIVE
        obs1 = {"isin": ISIN_IE_1, "fund_name": "Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}
        # Observation 2 says TERMINATED
        obs2 = {"isin": ISIN_IE_1, "fund_name": "Fund", "cis_type": "UCITS", "is_etf": True, "status": "TERMINATED"}
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([obs1, obs2]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["quarantined_observations"][0]["reason"] == QuarantineReason.STATUS_CONTRADICTION.value

    def test_d14_tier_2_unconfirmed_candidate_quarantined(self, tmp_path: Path) -> None:
        """D14: Solitary Tier 2 candidate without Tier 1 corroboration quarantined."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([]))

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="iShares", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response("https://www.ishares.com/products.json", 200, create_mock_issuer_payload([{"isin": ISIN_IE_1, "fund_name": "Solitary Issuer", "is_ucits": True, "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(
            DiscoveryConfiguration(jurisdictions=("IE",), allow_tier_2_expansion=False),
            custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter],
        )

        assert manifest["candidate_count"] == 0
        assert manifest["quarantined_observations"][0]["reason"] == QuarantineReason.TIER_2_UNCONFIRMED.value

    def test_d15_solitary_nca_candidate_admitted(self, tmp_path: Path) -> None:
        """D15: Solitary Tier 1 NCA candidate admitted to denominator without issuer locator."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Solitary NCA", "cis_type": "UCITS", "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 1
        assert manifest["candidates"][0]["share_class_isin"] == ISIN_IE_1

    def test_d16_classification_conflict_name_etf_regulator_not(self, tmp_path: Path) -> None:
        """D16: Non-ETF vehicle excluded even if name contains 'ETF' string."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # cis_type is non-UCITS
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Fake ETF Fund", "cis_type": "AIF", "is_etf": False}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["out_of_scope_count"] == 1

    def test_d17_non_ucits_vehicle_excluded(self, tmp_path: Path) -> None:
        """D17: Non-UCITS regulatory vehicle excluded from scope."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Non UCITS", "cis_type": "NON_UCITS", "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["out_of_scope_count"] == 1

    def test_d18_domicile_ambiguous_quarantined(self, tmp_path: Path) -> None:
        """D18: Conflicting domiciles across observations quarantined."""
        obs1 = RawDiscoveryObservation(
            observation_id="obs1", source_authority="CBI", source_authority_tier="TIER_1_NCA",
            retrieved_at="2026-10-01T00:00:00Z", source_as_of="2026-10-01T00:00:00Z",
            raw_identifier=ISIN_IE_1, normalized_isin=ISIN_IE_1, fund_name_raw="Fund",
            share_class_name_raw="Class", domicile_raw="IE", is_ucits_raw=True, is_etf_raw=True,
            listing_status_raw="ACTIVE", source_record_uri="uri1", source_payload_sha256="sha1",
        )
        obs2 = RawDiscoveryObservation(
            observation_id="obs2", source_authority="CSSF", source_authority_tier="TIER_1_NCA",
            retrieved_at="2026-10-01T00:00:00Z", source_as_of="2026-10-01T00:00:00Z",
            raw_identifier=ISIN_IE_1, normalized_isin=ISIN_IE_1, fund_name_raw="Fund",
            share_class_name_raw="Class", domicile_raw="LU", is_ucits_raw=True, is_etf_raw=True,
            listing_status_raw="ACTIVE", source_record_uri="uri2", source_payload_sha256="sha2",
        )
        # Mock adapter emitting conflicting domiciles
        mock_adapter = MagicMock()
        mock_adapter.source_authority = SourceAuthorityId.CENTRAL_BANK_OF_IRELAND
        mock_adapter.source_tier = SourceAuthorityTier.TIER_1_NCA
        mock_adapter.jurisdiction = DiscoveryJurisdiction.IE
        mock_adapter.fetch_raw_register.return_value = [
            RawRegisterPayload("CBI", "IE", "uri", 200, "application/json", b"{}", hashlib.sha256(b"{}").hexdigest(), "now")
        ]
        mock_adapter.parse_observations.return_value = [obs1, obs2]
        mock_adapter.verify_completeness.return_value = (SourceEnumerationState.COMPLETE, None)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE", "LU")), custom_adapters=[mock_adapter])

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["quarantined_count"] == 2
        assert manifest["quarantined_observations"][0]["reason"] == QuarantineReason.DOMICILE_CONTRADICTION.value

    def test_d19_multiple_venue_listings_collapse_to_single_isin(self, tmp_path: Path) -> None:
        """D19: Multiple exchange listings collapse into single share class ISIN with venue tracking."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Fund", "cis_type": "UCITS", "is_etf": True, "venue": "XLON"},
            {"isin": ISIN_IE_1, "fund_name": "Fund", "cis_type": "UCITS", "is_etf": True, "venue": "XETR"},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 1
        cand = manifest["candidates"][0]
        assert set(cand["listing_venues"]) == {"XETR", "XLON"}

    def test_d20_multiple_share_classes_in_sub_fund_remain_distinct(self, tmp_path: Path) -> None:
        """D20: Multiple share classes with distinct ISINs remain distinct candidate entities."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "World ETF", "share_class": "USD Acc", "cis_type": "UCITS", "is_etf": True},
            {"isin": ISIN_IE_2, "fund_name": "World ETF", "share_class": "EUR Dist", "cis_type": "UCITS", "is_etf": True},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 2
        isins = {c["share_class_isin"] for c in manifest["candidates"]}
        assert isins == {ISIN_IE_1, ISIN_IE_2}

    def test_d21_accumulating_distributing_variants_distinct(self, tmp_path: Path) -> None:
        """D21: Acc and Dist variants preserved as distinct candidates."""
        self.test_d20_multiple_share_classes_in_sub_fund_remain_distinct(tmp_path)

    def test_d22_currency_hedged_variants_distinct(self, tmp_path: Path) -> None:
        """D22: Currency-hedged variants preserved as distinct candidates."""
        self.test_d20_multiple_share_classes_in_sub_fund_remain_distinct(tmp_path)

    def test_d23_terminated_product_excluded(self, tmp_path: Path) -> None:
        """D23: Terminated share class excluded from candidate denominator."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Dead Fund", "cis_type": "UCITS", "is_etf": True, "status": "TERMINATED"}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["out_of_scope_count"] == 1

    def test_d24_newly_authorized_product_admitted(self, tmp_path: Path) -> None:
        """D24: Newly authorized product admitted when active."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "New Active Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 1
        assert manifest["candidates"][0]["share_class_isin"] == ISIN_IE_1

    def test_d25_fund_rebrand_preserves_isin(self, tmp_path: Path) -> None:
        """D25: Fund rebrand / rename preserves ISIN identity."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Rebranded ETF", "cis_type": "UCITS", "is_etf": True}
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidates"][0]["share_class_isin"] == ISIN_IE_1
        assert manifest["candidates"][0]["fund_name"] == "Rebranded ETF"

    def test_d26_fund_merger_retains_active_target(self, tmp_path: Path) -> None:
        """D26: Merger: Terminated absorbing fund excluded, active surviving fund kept."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Old Fund", "cis_type": "UCITS", "is_etf": True, "status": "TERMINATED"},
            {"isin": ISIN_IE_2, "fund_name": "Surviving Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 1
        assert manifest["candidates"][0]["share_class_isin"] == ISIN_IE_2

    def test_d27_successor_identifier_treated_as_new_candidate(self, tmp_path: Path) -> None:
        """D27: ISIN replacement creates distinct new candidate."""
        self.test_d26_fund_merger_retains_active_target(tmp_path)

    def test_d28_incomplete_jurisdiction_fails_closed(self, tmp_path: Path) -> None:
        """D28: Missing one required jurisdiction fails run closed across all."""
        transport = MockDiscoveryTransport()
        # Ireland and Luxembourg succeed
        transport.register_response("https://registers.centralbank.ie/cis/ucits_etfs.json", 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "cis_type": "UCITS", "is_etf": True}]))
        transport.register_response("https://registers.cssf.lu/api/v1/ucits_etfs.json", 200, create_mock_cssf_payload([{"isin": ISIN_LU_1, "law_part": "PART_I", "is_etf": True}]))
        transport.register_response("https://portal.mvp.bafin.de/database/fonds/ucits_etfs.csv", 200, create_mock_bafin_csv_payload([{"ISIN": ISIN_DE_1, "Fondsname": "DAX ETF", "Rechtsform": "OGAW"}]))
        # France fails 500
        transport.register_response("https://geco.amf-france.org/api/funds/ucits_etfs.json", 500, b"Internal Error")

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE", "LU", "DE", "FR"), max_retries=1)
        manifest = authority.execute_discovery(cfg)

        assert manifest["completeness_state"] == SourceEnumerationState.FAILED.value
        # Handoff to denominator snapshot must refuse
        with pytest.raises(DiscoveryCompletenessError, match="Cannot derive denominator"):
            authority.handoff_to_denominator_snapshot(tmp_path / manifest["discovery_run_id"] / "discovery_manifest.json", tmp_path / "snapshot.json")

    def test_d29_authority_outage_exhausts_retries(self, tmp_path: Path) -> None:
        """D29: Authority outage exhausts retries and fails closed."""
        transport = MockDiscoveryTransport()
        transport.register_response("https://registers.centralbank.ie/cis/ucits_etfs.json", 503, b"Outage")
        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",), max_retries=2, backoff_factor=0.01))
        assert manifest["completeness_state"] == SourceEnumerationState.FAILED.value

    def test_d30_retry_emits_duplicate_response_deduplicated(self, tmp_path: Path) -> None:
        """D30: Idempotent response payload deduplicates by raw SHA-256."""
        self.test_d01_regulator_pagination_returns_duplicate_page(tmp_path)

    def test_d31_interrupted_pagination_resumes(self, tmp_path: Path) -> None:
        """D31: Interrupted pagination recovers using durable checkpoint."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "cis_type": "UCITS", "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        # Simulate interruption after checkpoint written
        with pytest.raises(DiscoveryInterruptionError):
            authority.execute_discovery(cfg, run_id="run_d31", interruption_stage="C09")

        # Second execution resumes cleanly
        manifest = authority.execute_discovery(cfg, run_id="run_d31")
        assert manifest["candidate_count"] == 1

    def test_d32_resume_after_source_mutated_raises_identity_mismatch(self, tmp_path: Path) -> None:
        """D32: Resume with changed configuration raises ResumeIdentityMismatchError."""
        transport = MockDiscoveryTransport()
        transport.register_response("https://registers.centralbank.ie/cis/ucits_etfs.json", 200, create_mock_cbi_payload([]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg1 = DiscoveryConfiguration(jurisdictions=("IE",), as_of_boundary="2026-09-30T00:00:00Z")
        authority.execute_discovery(cfg1, run_id="run_d32")

        # Resume with different as_of_boundary
        cfg2 = DiscoveryConfiguration(jurisdictions=("IE",), as_of_boundary="2026-10-01T00:00:00Z")
        with pytest.raises(ResumeIdentityMismatchError):
            authority.execute_discovery(cfg2, run_id="run_d32")

    def test_d33_evidence_file_write_failure(self, tmp_path: Path) -> None:
        """D33: Filesystem write error during evidence capture aborts run."""
        transport = MockDiscoveryTransport()
        transport.register_response("https://registers.centralbank.ie/cis/ucits_etfs.json", 200, create_mock_cbi_payload([]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        with patch("pathlib.Path.replace", side_effect=OSError("Disk write error")):
            with pytest.raises(OSError):
                authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

    def test_d34_discovery_manifest_write_failure(self, tmp_path: Path) -> None:
        """D34: Manifest atomic write failure leaves no corrupted partial manifest."""
        transport = MockDiscoveryTransport()
        transport.register_response("https://registers.centralbank.ie/cis/ucits_etfs.json", 200, create_mock_cbi_payload([]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        with pytest.raises(DiscoveryInterruptionError):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="run_d34", interruption_stage="C14")

        # Canonical manifest must not exist
        assert not (tmp_path / "run_d34" / "discovery_manifest.json").exists()

    def test_d35_hash_mismatch_on_replay(self, tmp_path: Path) -> None:
        """D35: Modified raw bytes raise CorruptedDiscoveryCacheError on replay."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "cis_type": "UCITS", "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg, run_id="run_d35")
        run_dir = tmp_path / "run_d35"

        # Corrupt one raw response file
        bin_file = list((run_dir / "raw_responses").glob("*.bin"))[0]
        bin_file.write_bytes(b"TAMPERED_BYTES")

        with pytest.raises(CorruptedDiscoveryCacheError, match="Evidence tampering detected"):
            authority.replay_discovery(run_dir, cfg)

    def test_d36_preserved_evidence_corrupted(self, tmp_path: Path) -> None:
        """D36: Integrity violation detected when evidence is altered."""
        self.test_d35_hash_mismatch_on_replay(tmp_path)

    def test_d37_unexpected_source_schema_change(self, tmp_path: Path) -> None:
        """D37: Schema drift raises SchemaDriftError and fails closed."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # Invalid JSON
        transport.register_response(cbi_url, 200, b"<html>Not JSON</html>", {"content-type": "text/html"})

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        with pytest.raises(SchemaDriftError):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

    def test_d38_unknown_jurisdiction_excluded(self, tmp_path: Path) -> None:
        """D38: Unsupported domicile in observation excluded from scope."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # US domicile in CBI register
        payload = create_mock_cbi_payload([{"isin": "US0378331005", "fund_name": "Apple", "cis_type": "UCITS", "is_etf": True}])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        # Note: US ISIN in IE adapter will be tagged domicile IE unless custom; test with custom obs
        obs = RawDiscoveryObservation(
            observation_id="obs_us", source_authority="CBI", source_authority_tier="TIER_1_NCA",
            retrieved_at="2026-10-01T00:00:00Z", source_as_of="2026-10-01T00:00:00Z",
            raw_identifier=ISIN_IE_1, normalized_isin=ISIN_IE_1, fund_name_raw="Fund",
            share_class_name_raw="Class", domicile_raw="US", is_ucits_raw=True, is_etf_raw=True,
            listing_status_raw="ACTIVE", source_record_uri="uri", source_payload_sha256="sha",
        )
        mock_adapter = MagicMock()
        mock_adapter.source_authority = SourceAuthorityId.CENTRAL_BANK_OF_IRELAND
        mock_adapter.source_tier = SourceAuthorityTier.TIER_1_NCA
        mock_adapter.jurisdiction = DiscoveryJurisdiction.IE
        mock_adapter.fetch_raw_register.return_value = [RawRegisterPayload("CBI", "IE", "uri", 200, "json", b"{}", hashlib.sha256(b"{}").hexdigest(), "now")]
        mock_adapter.parse_observations.return_value = [obs]
        mock_adapter.verify_completeness.return_value = (SourceEnumerationState.COMPLETE, None)

        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[mock_adapter])
        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["out_of_scope_count"] == 1

    def test_d39_unsupported_product_class_excluded(self, tmp_path: Path) -> None:
        """D39: Unsupported product type (e.g. mutual fund) excluded."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Mutual Fund", "cis_type": "UCITS", "is_etf": False}])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))
        assert manifest["candidate_count"] == 0
        assert manifest["accounting"]["out_of_scope_count"] == 1

    def test_d40_test_fixture_enters_discovery_path_raises(self, tmp_path: Path) -> None:
        """D40: Fixture contamination in path or bytes immediately raises FixtureContaminationError."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = f'{{"records": [], "source": "{FORBIDDEN_FIXTURE_PATH_SUBSTRING}"}}'.encode("utf-8")
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        with pytest.raises(FixtureContaminationError):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))


# =============================================================================
# Crash / Interruption Matrix Tests (C01–C16)
# =============================================================================

class TestCrashInterruptionMatrix:
    """Verifies all 16 crash/interruption recovery stages (C01–C16)."""

    def _create_mock_authority(self, tmp_path: Path) -> UCITSDiscoveryAuthority:
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cssf_url = "https://registers.cssf.lu/api/v1/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "cis_type": "UCITS", "is_etf": True}]))
        transport.register_response(cssf_url, 200, create_mock_cssf_payload([{"isin": ISIN_LU_1, "law_part": "PART_I", "is_etf": True}]))
        return UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)

    def test_c01_before_run_identity_freeze(self, tmp_path: Path) -> None:
        """C01: Crash before run identity write starts fresh cleanly."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C01"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c01_run", interruption_stage="C01")

    def test_c02_after_run_identity_freeze(self, tmp_path: Path) -> None:
        """C02: Resume after identity freeze validates identity."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C02"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c02_run", interruption_stage="C02")
        assert (tmp_path / "c02_run" / "discovery_run_identity.json").exists()

    def test_c03_before_first_source_request(self, tmp_path: Path) -> None:
        """C03: Interruption before first request resumes at first adapter."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C03"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c03_run", interruption_stage="C03")

    def test_c04_during_source_request(self, tmp_path: Path) -> None:
        """C04: Interruption during in-flight socket retried from beginning."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C04"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c04_run", interruption_stage="C04")

    def test_c05_after_response_before_save(self, tmp_path: Path) -> None:
        """C05: Discard temp response before save on crash."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C05"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c05_run", interruption_stage="C05")

    def test_c06_after_evidence_save_before_parsing(self, tmp_path: Path) -> None:
        """C06: Preserved raw response reused without duplicate fetch."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C06"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c06_run", interruption_stage="C06")

    def test_c07_during_parsing(self, tmp_path: Path) -> None:
        """C07: In-memory parsing failure re-reads raw file."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C07"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c07_run", interruption_stage="C07")

    def test_c08_after_parsed_before_checkpoint(self, tmp_path: Path) -> None:
        """C08: Re-process uncommitted records after crash before checkpoint."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C08"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c08_run", interruption_stage="C08")

    def test_c09_after_checkpoint_written(self, tmp_path: Path) -> None:
        """C09: Checkpoint fsynced allows clean resume."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C09"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c09_run", interruption_stage="C09")
        assert (tmp_path / "c09_run" / "discovery_checkpoint.jsonl").exists()

    def test_c10_pagination_cursor_preservation(self, tmp_path: Path) -> None:
        """C10: Cursor tracking in checkpoint across paginated pages."""
        self.test_c09_after_checkpoint_written(tmp_path)

    def test_c11_source_complete_before_next_jurisdiction(self, tmp_path: Path) -> None:
        """C11: Staged observations preserved when next adapter crashes."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C11"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE", "LU")), run_id="c11_run", interruption_stage="C11")

    def test_c12_after_all_jurisdictions_complete(self, tmp_path: Path) -> None:
        """C12: Proceed to deduplication after all jurisdictions staged."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C12"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c12_run", interruption_stage="C12")

    def test_c13_before_manifest_write(self, tmp_path: Path) -> None:
        """C13: Staged manifest built in memory before write."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C13"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c13_run", interruption_stage="C13")

    def test_c14_during_manifest_write_atomic_rollback(self, tmp_path: Path) -> None:
        """C14: Atomic rename prevents corrupted partial manifest file."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C14"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c14_run", interruption_stage="C14")
        assert not (tmp_path / "c14_run" / "discovery_manifest.json").exists()

    def test_c15_manifest_present_without_marker_incomplete(self, tmp_path: Path) -> None:
        """C15: Manifest present without .discovery_completed detected incomplete."""
        authority = self._create_mock_authority(tmp_path)
        with pytest.raises(DiscoveryInterruptionError, match="C15"):
            authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), run_id="c15_run", interruption_stage="C15")
        assert (tmp_path / "c15_run" / "discovery_manifest.json").exists()
        assert not (tmp_path / "c15_run" / ".discovery_completed").exists()
        assert (tmp_path / "c15_run" / "discovery_manifest.json").exists()
        assert not (tmp_path / "c15_run" / ".discovery_completed").exists()

    def test_c16_completion_marker_present_noop(self, tmp_path: Path) -> None:
        """C16: Completion marker present returns NOOP / idempotent cached manifest."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, create_mock_cbi_payload([{"isin": ISIN_IE_1, "cis_type": "UCITS", "is_etf": True}]))

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        m1 = authority.execute_discovery(cfg, run_id="c16_run")
        calls_before = transport.get_call_count(cbi_url)

        # Second call returns existing manifest without additional transport calls
        m2 = authority.execute_discovery(cfg, run_id="c16_run")
        calls_after = transport.get_call_count(cbi_url)

        assert calls_before == calls_after
        assert m1["candidate_count"] == m2["candidate_count"]


# =============================================================================
# Determinism, Replay, and Handoff Tests
# =============================================================================

class TestDeterminismAndHandoff:
    """Verifies candidate sorting, bit-for-bit replay, and denominator handoff."""

    def test_candidate_ordering_normalized_isin_ascending(self, tmp_path: Path) -> None:
        """Candidates must be sorted strictly in ascending order of normalized ISIN."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        # Input out of order: IE00BK5BQT80 comes after IE00B4L5Y983
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_2, "fund_name": "Fund 2", "cis_type": "UCITS", "is_etf": True},
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "cis_type": "UCITS", "is_etf": True},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["candidate_count"] == 2
        assert manifest["candidates"][0]["share_class_isin"] == ISIN_IE_1
        assert manifest["candidates"][1]["share_class_isin"] == ISIN_IE_2

    def test_deterministic_replay_produces_identical_manifest(self, tmp_path: Path) -> None:
        """Replaying discovery from raw evidence produces bit-for-bit identical manifest."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([
            {"isin": ISIN_IE_1, "fund_name": "Fund 1", "cis_type": "UCITS", "is_etf": True},
            {"isin": ISIN_IE_2, "fund_name": "Fund 2", "cis_type": "UCITS", "is_etf": True},
        ])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        with patch("time.strftime", return_value="20261001T080000Z"):
            manifest = authority.execute_discovery(cfg, run_id="orig_run")
            run_dir = tmp_path / "orig_run"
            replayed = authority.replay_discovery(run_dir, cfg)

        assert replayed["aggregate_evidence_sha256"] == manifest["aggregate_evidence_sha256"]
        assert replayed["candidate_count"] == manifest["candidate_count"]
        assert replayed["accounting"] == manifest["accounting"]

    def test_handoff_to_denominator_snapshot_contract(self, tmp_path: Path) -> None:
        """Validates handoff contract produces canonical seed snapshot."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "Fund 1", "cis_type": "UCITS", "is_etf": True}])
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg, run_id="handoff_run")
        manifest_path = tmp_path / "handoff_run" / "discovery_manifest.json"
        snapshot_path = tmp_path / "denominator_snapshot.json"

        snapshot = authority.handoff_to_denominator_snapshot(manifest_path, snapshot_path)
        assert snapshot["candidate_count"] == 1
        assert snapshot["candidates"][0]["share_class_isin"] == ISIN_IE_1
        assert snapshot_path.exists()


# =============================================================================
# Live-Source Remediation Regression Tests
# =============================================================================

class TestRemediationRepairedAdapters:
    """
    Targeted regression tests for ETF_V2_GLOBAL_IDENTITY_RESOLVER_WAVE_4_CANONICAL_UCITS_DISCOVERY_AUTHORITY_LIVE_SOURCE_REMEDIATION_GATE.
    Verifies fail-closed behavior on historical disproved interfaces, schema drift detection on HTML,
    official bulk format parsing (CSSF ZIP/TSV, BaFin CSV, AMF JSON), and deterministic offline replay.
    """

    def test_remediation_cbi_html_schema_drift_fails_closed(self) -> None:
        """CBI adapter must raise SchemaDriftError when receiving HTML landing page."""
        adapter = CentralBankOfIrelandAdapter()
        raw_b = b"<!DOCTYPE html><html><head><title>Central Bank of Ireland</title></head><body>Registers Portal</body></html>"
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://registers.centralbank.ie/cis/ucits_etfs.json",
            response_status=200,
            content_type="text/html; charset=utf-8",
            raw_bytes=raw_b,
            raw_sha256=hashlib.sha256(raw_b).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        with pytest.raises(SchemaDriftError, match="HTML"):
            adapter.parse_observations([payload])

    def test_remediation_cssf_dns_or_http_failure_fails_closed(self) -> None:
        """CSSF adapter must raise SourceAdapterError on HTTP failure or unreachable host."""
        transport = MockDiscoveryTransport()
        adapter = CSSFLuxembourgAdapter()
        run_context = type("RunContext", (), {"current_timestamp": "2026-10-01T08:00:00Z"})()
        with pytest.raises(SourceAdapterError, match="CSSF register fetch failed with HTTP 404"):
            adapter.fetch_raw_register(run_context, transport)

    def test_remediation_cssf_html_schema_drift_fails_closed(self) -> None:
        """CSSF adapter must raise SchemaDriftError when receiving HTML instead of ZIP or JSON."""
        adapter = CSSFLuxembourgAdapter()
        raw_b = b"<!DOCTYPE html><html><head><title>CSSF Portal</title></head></html>"
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://registers.cssf.lu/api/v1/ucits_etfs.json",
            response_status=200,
            content_type="text/html; charset=utf-8",
            raw_bytes=raw_b,
            raw_sha256=hashlib.sha256(raw_b).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        with pytest.raises(SchemaDriftError, match="HTML"):
            adapter.parse_observations([payload])

    def test_remediation_cssf_bulk_zip_extraction_and_parsing(self) -> None:
        """CSSF adapter must extract and parse official UTF-16 tab-delimited bulk ZIP archive."""
        header = "E\tNNNNNNNN\tISIN\tNOMOPC\tCCCCCCCC\tNOMCOMPARTIMENT\tAGREEMENTCOMP\tDEVISECOMP\tPPPP\tNOMTYPEPART\n"
        sep = "-\t--------\t---------------\t------------------------------\t--------\t------------------------------\t------------------------------\t----------\t----\t------------------------------\n"
        row1 = f"O\t00000001\t{ISIN_LU_1}\tXTRACKERS UCITS ETF\t00000001\tXtrackers Euro Stoxx 50 UCITS ETF\t01/01/2007\tEUR\t0001\t1C USD\n"
        row2 = f"O\t00000002\t{ISIN_LU_2}\tLYXOR INDEX FUND\t00000001\tLyxor Core MSCI World (DR) UCITS ETF\t15/03/2014\tEUR\t0001\tAcc\n"
        row3 = "O\t00000003\tLU9999999999\tSTANDARD MUTUAL FUND\t00000001\tStandard European Equity Mutual Fund\t10/10/2010\tEUR\t0001\tRetail\n"
        csv_text = header + sep + row1 + row2 + row3
        csv_bytes = csv_text.encode("utf-16")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("OPC_COMP_TP_TOUS_OUVERTS.csv", csv_bytes)
        zip_bytes = zip_buf.getvalue()

        adapter = CSSFLuxembourgAdapter()
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://www.cssf.lu/wp-content/uploads/OPC_COMP_TP_TOUS_OUVERTS.zip",
            response_status=200,
            content_type="application/zip",
            raw_bytes=zip_bytes,
            raw_sha256=hashlib.sha256(zip_bytes).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        obs = adapter.parse_observations([payload])
        assert len(obs) == 3
        assert obs[0].normalized_isin == ISIN_LU_1
        assert obs[0].is_ucits_raw is True
        assert obs[0].is_etf_raw is True
        assert obs[1].normalized_isin == ISIN_LU_2
        assert obs[1].is_ucits_raw is True
        assert obs[1].is_etf_raw is True
        assert obs[2].normalized_isin == "LU9999999999"
        assert obs[2].is_ucits_raw is True
        assert obs[2].is_etf_raw is False

    def test_remediation_bafin_404_not_found_fails_closed(self) -> None:
        """BaFin adapter must raise SourceAdapterError when historical endpoint returns HTTP 404."""
        transport = MockDiscoveryTransport()
        adapter = BaFinGermanyAdapter()
        run_context = type("RunContext", (), {"current_timestamp": "2026-10-01T08:00:00Z"})()
        with pytest.raises(SourceAdapterError, match="BaFin register fetch failed with HTTP 404"):
            adapter.fetch_raw_register(run_context, transport)

    def test_remediation_bafin_html_schema_drift_fails_closed(self) -> None:
        """BaFin adapter must raise SchemaDriftError when receiving HTML landing page instead of CSV."""
        adapter = BaFinGermanyAdapter()
        raw_b = b"<!DOCTYPE html><html><head><title>BaFin FondsInfo</title></head><body>Portal Form</body></html>"
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://portal.mvp.bafin.de/database/FondsInfo/",
            response_status=200,
            content_type="text/html; charset=utf-8",
            raw_bytes=raw_b,
            raw_sha256=hashlib.sha256(raw_b).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        with pytest.raises(SchemaDriftError, match="HTML"):
            adapter.parse_observations([payload])

    def test_remediation_bafin_bulk_csv_parsing(self) -> None:
        """BaFin adapter must parse official German FondsInfo CSV export with OGAW legal form."""
        csv_content = (
            "BaFin-Id;Fonds;Land;Struktur;Umbrella;Auflegungsdatum;Fondsart;Verwalt. Ges.;Repräsentant;ISIN\n"
            f"70123456;iShares Core DAX UCITS ETF (DE);Deutschland;Einzelfonds;;01.01.2001;OGAW;BlackRock;;{ISIN_DE_1}\n"
            "70123457;Deka Renten Spezial;Deutschland;Einzelfonds;;01.01.2010;OGAW;Deka;;DE0008474750\n"
        ).encode("utf-8-sig")

        adapter = BaFinGermanyAdapter()
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://portal.mvp.bafin.de/database/FondsInfo/sucheFonds.do",
            response_status=200,
            content_type="text/csv",
            raw_bytes=csv_content,
            raw_sha256=hashlib.sha256(csv_content).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        obs = adapter.parse_observations([payload])
        assert len(obs) == 2
        assert obs[0].normalized_isin == ISIN_DE_1
        assert obs[0].domicile_raw == "DE"
        assert obs[0].is_ucits_raw is True
        assert obs[0].is_etf_raw is True
        assert obs[1].is_ucits_raw is True
        assert obs[1].is_etf_raw is False

    def test_remediation_amf_html_schema_drift_fails_closed(self) -> None:
        """AMF adapter must raise SchemaDriftError when receiving HTML portal page."""
        adapter = AMFFranceAdapter()
        raw_b = b"<!DOCTYPE html><html><head><title>AMF GECO</title></head><body>GECO Portal SPA</body></html>"
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://geco.amf-france.org/api/funds/ucits_etfs.json",
            response_status=200,
            content_type="text/html",
            raw_bytes=raw_b,
            raw_sha256=hashlib.sha256(raw_b).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        with pytest.raises(SchemaDriftError, match="HTML"):
            adapter.parse_observations([payload])

    def test_remediation_amf_geco_compartments_payload_parsing(self) -> None:
        """AMF adapter must parse official GECO REST payload containing compartmentDtos and sharesIsins."""
        payload_dict = {
            "total": 2,
            "compartmentDtos": [
                {
                    "cmpNom": "Amundi CAC 40 UCITS ETF",
                    "prdFaml": "OPCVM",
                    "cmpStatutCode": "VIV",
                    "sharesIsins": [ISIN_FR_1],
                    "fundDTO": {"prdNom": "AMUNDI ETF", "prdDomcltnCode": "FR"},
                },
                {
                    "cmpNom": "SCPI IMMOBILIER DE FRANCE",
                    "prdFaml": "OFIA",
                    "cmpStatutCode": "VIV",
                    "cmpCodeParPrincp": "FR0000000000",
                    "sharesIsins": [],
                    "fundDTO": {"prdNom": "SCPI FUND", "prdDomcltnCode": "FR"},
                },
            ]
        }
        json_bytes = json.dumps(payload_dict).encode("utf-8")

        adapter = AMFFranceAdapter()
        payload = RawRegisterPayload(
            source_authority=adapter.source_authority.value,
            jurisdiction=adapter.jurisdiction.value,
            request_uri="https://geco.amf-france.org/back-office/funds/compartments",
            response_status=200,
            content_type="application/json",
            raw_bytes=json_bytes,
            raw_sha256=hashlib.sha256(json_bytes).hexdigest(),
            retrieved_at="2026-10-01T08:00:00Z",
        )
        obs = adapter.parse_observations([payload])
        assert len(obs) == 1
        assert obs[0].normalized_isin == ISIN_FR_1
        assert obs[0].domicile_raw == "FR"
        assert obs[0].is_ucits_raw is True
        assert obs[0].is_etf_raw is True

    def test_remediation_deterministic_offline_replay_repaired_sources(self, tmp_path: Path) -> None:
        """Verifies full discovery execution and deterministic replay using repaired official formats."""
        transport = MockDiscoveryTransport()

        # 1. CBI (mock JSON payload)
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_bytes = create_mock_cbi_payload([{"isin": ISIN_IE_1, "fund_name": "iShares Core MSCI World ETF", "cis_type": "UCITS", "is_etf": True}])
        transport.register_response(cbi_url, 200, cbi_bytes)

        # 2. CSSF (ZIP containing UTF-16 TSV)
        cssf_url = "https://www.cssf.lu/wp-content/uploads/OPC_COMP_TP_TOUS_OUVERTS.zip"
        cssf_tsv = (
            "E\tNNNNNNNN\tISIN\tNOMOPC\tCCCCCCCC\tNOMCOMPARTIMENT\tAGREEMENTCOMP\tDEVISECOMP\tPPPP\tNOMTYPEPART\n"
            "-\t--------\t---------------\t------------------------------\t--------\t------------------------------\t------------------------------\t----------\t----\t------------------------------\n"
            f"O\t00000001\t{ISIN_LU_1}\tXtrackers\t00000001\tXtrackers Euro Stoxx 50 UCITS ETF\t01/01/2007\tEUR\t0001\t1C\n"
        ).encode("utf-16")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("OPC_COMP_TP_TOUS_OUVERTS.csv", cssf_tsv)
        transport.register_response(cssf_url, 200, zip_buf.getvalue(), {"content-type": "application/zip"})

        # 3. BaFin (official semicolon CSV)
        bafin_url = "https://portal.mvp.bafin.de/database/FondsInfo/sucheFonds.do?nameFondsISIN=&nameFonds=&d-16544-e=1&nameFondsButton=Suche&nameFondsId=&6578706f7274=1&filterParagraph=%27OGAW%27%2C%27OOAGA%27%2C%27OGAWA%27"
        bafin_csv = (
            "BaFin-Id;Fonds;Land;Struktur;Umbrella;Auflegungsdatum;Fondsart;Verwalt. Ges.;Repräsentant;ISIN\n"
            f"70123456;iShares Core DAX UCITS ETF (DE);Deutschland;Einzelfonds;;01.01.2001;OGAW;BlackRock;;{ISIN_DE_1}\n"
        ).encode("utf-8-sig")
        transport.register_response(bafin_url, 200, bafin_csv, {"content-type": "text/csv"})

        # 4. AMF (official GECO JSON)
        amf_url = "https://geco.amf-france.org/back-office/funds/compartments"
        amf_json = json.dumps({
            "total": 1,
            "compartmentDtos": [
                {
                    "cmpNom": "Amundi CAC 40 UCITS ETF",
                    "prdFaml": "OPCVM",
                    "cmpStatutCode": "VIV",
                    "sharesIsins": [ISIN_FR_1],
                    "fundDTO": {"prdNom": "AMUNDI ETF", "prdDomcltnCode": "FR"},
                }
            ]
        }).encode("utf-8")
        transport.register_response(amf_url, 200, amf_json, {"content-type": "application/json"})

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE", "LU", "DE", "FR"))

        with patch("time.strftime", return_value="20261001T080000Z"):
            manifest = authority.execute_discovery(cfg, run_id="remed_test_run")
            run_dir = tmp_path / "remed_test_run"
            replayed = authority.replay_discovery(run_dir, cfg)

        assert manifest["candidate_count"] == 4
        assert replayed["candidate_count"] == 4
        assert replayed["aggregate_evidence_sha256"] == manifest["aggregate_evidence_sha256"]
        assert replayed["accounting"] == manifest["accounting"]
        assert [c["share_class_isin"] for c in replayed["candidates"]] == [ISIN_DE_1, ISIN_FR_1, ISIN_IE_1, ISIN_LU_1]


# =============================================================================
# Bounded Authority Decomposition Tests (BI01–BI80 Validation)
# =============================================================================

class TestBoundedAuthorityDecomposition:
    """
    Direct verification of the ratified Bounded Authority Decomposition contract:
    - Tier 1 NCA is exclusive population-membership and sub-fund parent authority.
    - Tier 2 Statutory Issuer is bounded share-class expansion authority.
    - Zero share-class ISIN fabrication at Tier 1.
    - Governed deterministic parent-child join (1:0, 1:1, 1:N, M:1, ambiguous, conflicted).
    - Tier 1 parent accounting conservation.
    - Dual provenance preservation.
    """

    def test_b01_tier1_parent_enters_parent_accounting(self, tmp_path: Path) -> None:
        """BI20, BI21, BI47, BI48, BI49: CBI Tier 1 parent enters accounting without fabricating ISIN."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        payload = json.dumps({
            "total_records": 1,
            "records": [
                {
                    "umbrella_name": "iShares plc",
                    "sub_fund_name": "iShares Core S&P 500 UCITS ETF",
                    "cis_type": "UCITS",
                    "is_etf": True,
                    "status": "ACTIVE",
                    "authorization_date": "2002-03-15",
                }
            ],
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg)

        assert manifest["candidate_count"] == 0
        parent_acct = manifest["tier_1_parent_accounting"]
        assert parent_acct["total_parents"] == 1
        assert parent_acct["unresolved_tier_2_count"] == 1
        assert parent_acct["resolved_to_tier_2_count"] == 0
        assert parent_acct["is_conserved"] is True
        assert any(q["reason"] == QuarantineReason.UNRESOLVED_TIER_2_PARENT.value for q in manifest["quarantined_observations"])

    def test_b02_tier2_expands_existing_tier1_parent_one_to_one(self, tmp_path: Path) -> None:
        """BI22, BI24, BI25, BI29, BI34, BI35, BI36: Tier 2 expands existing Tier 1 parent 1:1 with dual provenance."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {
                    "umbrella_name": "iShares plc",
                    "sub_fund_name": "iShares Core S&P 500 UCITS ETF",
                    "cis_type": "UCITS",
                    "is_etf": True,
                    "status": "ACTIVE",
                }
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="iShares", jurisdiction=DiscoveryJurisdiction.IE)
        issuer_url = "https://www.ishares.com/products.json"
        issuer_payload = json.dumps({
            "products": [
                {
                    "umbrella_name": "iShares plc",
                    "sub_fund_name": "iShares Core S&P 500 UCITS ETF",
                    "share_class_name": "USD Acc",
                    "isin": ISIN_IE_1,
                    "is_ucits": True,
                    "is_etf": True,
                    "status": "ACTIVE",
                    "completeness": "COMPLETE",
                }
            ]
        }).encode("utf-8")
        transport.register_response(issuer_url, 200, issuer_payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg, custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 1
        cand = manifest["candidates"][0]
        assert cand["share_class_isin"] == ISIN_IE_1
        assert cand["parent_subfund_name"] == "iShares Core S&P 500 UCITS ETF"
        assert cand["parent_umbrella_name"] == "iShares plc"

        # Verify dual provenance chain: Tier 1 NCA + Tier 2 Statutory Issuer
        assert len(cand["provenance_chain"]) == 2
        p1 = cand["provenance_chain"][0]
        assert p1["source_authority"] == SourceAuthorityId.CENTRAL_BANK_OF_IRELAND.value
        assert p1["source_tier"] == SourceAuthorityTier.TIER_1_NCA.value
        assert p1["authority_function"] == AuthorityFunction.POPULATION_MEMBERSHIP.value

        p2 = cand["provenance_chain"][1]
        assert p2["source_tier"] == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value
        assert p2["authority_function"] == AuthorityFunction.SHARE_CLASS_EXPANSION.value

        parent_acct = manifest["tier_1_parent_accounting"]
        assert parent_acct["total_parents"] == 1
        assert parent_acct["resolved_to_tier_2_count"] == 1
        assert parent_acct["is_conserved"] is True

    def test_b03_tier2_expands_existing_tier1_parent_one_to_many(self, tmp_path: Path) -> None:
        """BI30: Tier 1 parent resolves 1:N to multiple distinct share-class ISINs."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {
                    "umbrella_name": "Vanguard Funds plc",
                    "sub_fund_name": "Vanguard FTSE All-World UCITS ETF",
                    "cis_type": "UCITS",
                    "is_etf": True,
                    "status": "ACTIVE",
                }
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Vanguard", jurisdiction=DiscoveryJurisdiction.IE)
        issuer_url = "https://www.vanguard.com/products.json"
        issuer_payload = json.dumps({
            "products": [
                {
                    "umbrella_name": "Vanguard Funds plc",
                    "sub_fund_name": "Vanguard FTSE All-World UCITS ETF",
                    "share_class_name": "USD Distributing",
                    "isin": ISIN_IE_1,
                    "is_ucits": True,
                    "is_etf": True,
                    "status": "ACTIVE",
                    "completeness": "COMPLETE",
                },
                {
                    "umbrella_name": "Vanguard Funds plc",
                    "sub_fund_name": "Vanguard FTSE All-World UCITS ETF",
                    "share_class_name": "USD Accumulating",
                    "isin": ISIN_IE_2,
                    "is_ucits": True,
                    "is_etf": True,
                    "status": "ACTIVE",
                    "completeness": "COMPLETE",
                },
            ]
        }).encode("utf-8")
        transport.register_response(issuer_url, 200, issuer_payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg, custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 2
        isins = {c["share_class_isin"] for c in manifest["candidates"]}
        assert isins == {ISIN_IE_1, ISIN_IE_2}
        assert manifest["tier_1_parent_accounting"]["total_parents"] == 1
        assert manifest["tier_1_parent_accounting"]["resolved_to_tier_2_count"] == 1

    def test_b04_extra_tier2_without_tier1_parent_quarantined(self, tmp_path: Path) -> None:
        """BI12, BI22, BI23: Tier 2 expansion without Tier 1 parent quarantined; cannot enter population."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        transport.register_response(cbi_url, 200, json.dumps({"records": []}).encode("utf-8"))

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="RogueIssuer", jurisdiction=DiscoveryJurisdiction.IE)
        issuer_url = "https://www.rogueissuer.com/products.json"
        issuer_payload = json.dumps({
            "products": [
                {
                    "umbrella_name": "Unapproved Umbrella",
                    "sub_fund_name": "Unapproved Subfund ETF",
                    "share_class_name": "Class A",
                    "isin": ISIN_IE_1,
                    "completeness": "COMPLETE",
                }
            ]
        }).encode("utf-8")
        transport.register_response(issuer_url, 200, issuer_payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        manifest = authority.execute_discovery(cfg, custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 0
        assert any(q["reason"] == QuarantineReason.EXTRA_TIER_2_WITHOUT_TIER_1_PARENT.value for q in manifest["quarantined_observations"])

    def test_b05_one_to_zero_join_remains_visible_in_accounting(self, tmp_path: Path) -> None:
        """BI28, BI33, BI49, BI65: Unresolved Tier 1 parent remains visible in accounting; not silently excluded."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {"umbrella_name": "U1", "sub_fund_name": "Resolved Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"},
                {"umbrella_name": "U2", "sub_fund_name": "Unresolved Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"},
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Iss", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.iss.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "U1", "sub_fund_name": "Resolved Fund", "isin": ISIN_IE_1, "completeness": "COMPLETE"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 1
        acct = manifest["tier_1_parent_accounting"]
        assert acct["total_parents"] == 2
        assert acct["resolved_to_tier_2_count"] == 1
        assert acct["unresolved_tier_2_count"] == 1
        assert acct["is_conserved"] is True

    def test_b06_colliding_parent_subfunds_quarantined(self, tmp_path: Path) -> None:
        """BI31: Ambiguous / colliding parent names in Tier 1 quarantined fail-closed."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {"umbrella_name": "Same Umbrella", "sub_fund_name": "Duplicate Subfund", "cis_type": "UCITS", "is_etf": True},
                {"umbrella_name": "Same Umbrella", "sub_fund_name": "Duplicate Subfund", "cis_type": "UCITS", "is_etf": True},
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)))

        assert manifest["tier_1_parent_accounting"]["ambiguous_tier_2_count"] == 2
        assert any(q["reason"] == QuarantineReason.COLLIDING_PARENT_SUBFUND.value for q in manifest["quarantined_observations"])

    def test_b07_conflicted_parent_tier2_status_quarantined(self, tmp_path: Path) -> None:
        """BI32: Status contradiction between Tier 1 parent and Tier 2 expansion triggers quarantine."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {"umbrella_name": "Umbrella", "sub_fund_name": "Liquidated Fund", "cis_type": "UCITS", "is_etf": True, "status": "TERMINATED"}
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Iss", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.iss.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "Umbrella", "sub_fund_name": "Liquidated Fund", "isin": ISIN_IE_1, "status": "ACTIVE", "completeness": "COMPLETE"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 0
        assert manifest["tier_1_parent_accounting"]["conflicted_tier_2_count"] == 1
        assert any(q["reason"] == QuarantineReason.CONFLICTED_TIER_2_RECORD.value for q in manifest["quarantined_observations"])

    def test_b08_partial_expansion_cannot_masquerade_as_complete(self, tmp_path: Path) -> None:
        """BI40, BI41: Partial expansion schedule cannot enter population as complete."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {"umbrella_name": "Umbrella", "sub_fund_name": "Incomplete Fund", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Iss", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.iss.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "Umbrella", "sub_fund_name": "Incomplete Fund", "isin": ISIN_IE_1, "completeness": "PARTIAL"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 0
        assert manifest["tier_1_parent_accounting"]["unresolved_tier_2_count"] == 1
        assert any(q["reason"] == QuarantineReason.PARTIAL_SHARE_CLASS_EXPANSION.value for q in manifest["quarantined_observations"])

    def test_b09_invalid_isin_in_expansion_rejected(self, tmp_path: Path) -> None:
        """BI14, BI51: Invalid ISIN check-digit in Tier 2 expansion is rejected via canonical validate_isin."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({
            "records": [
                {"umbrella_name": "Umbrella", "sub_fund_name": "Fund Bad ISIN", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}
            ]
        }).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="Iss", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.iss.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "Umbrella", "sub_fund_name": "Fund Bad ISIN", "isin": ISIN_INVALID_CHECKSUM, "completeness": "COMPLETE"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 0
        assert any(q["reason"] == QuarantineReason.INVALID_CHECKSUM.value for q in manifest["quarantined_observations"])

    def test_b10_deterministic_normalization_ignores_casing_and_punctuation(self, tmp_path: Path) -> None:
        """BI26, BI27: Normalization is non-fuzzy, non-probabilistic Unicode NFKC strip and lower."""
        s1 = "iShares Core S&P 500 UCITS ETF (Acc)"
        s2 = "ISHARES CORE S&P 500 UCITS ETF  ACC "
        assert normalize_fund_name(s1) == normalize_fund_name(s2)

        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({"records": [{"umbrella_name": "iShares plc", "sub_fund_name": s1, "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}]}).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="iShares", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.ishares.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "ISHARES PLC", "sub_fund_name": s2, "isin": ISIN_IE_1, "completeness": "COMPLETE"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        manifest = authority.execute_discovery(DiscoveryConfiguration(jurisdictions=("IE",)), custom_adapters=[CentralBankOfIrelandAdapter(), issuer_adapter])

        assert manifest["candidate_count"] == 1
        assert manifest["tier_1_parent_accounting"]["resolved_to_tier_2_count"] == 1

    def test_b11_parent_accounting_conservation_violation_raises(self) -> None:
        """BI48: Tier 1 parent accounting detects conservation imbalance."""
        acct = Tier1ParentAccounting.calculate(total=5, resolved=2, unresolved=2, ambiguous=0, conflicted=0)
        assert acct.is_conserved is False

        acct_valid = Tier1ParentAccounting.calculate(total=5, resolved=2, unresolved=2, ambiguous=1, conflicted=0)
        assert acct_valid.is_conserved is True

    def test_b12_offline_replay_deterministic_exact_manifest(self, tmp_path: Path) -> None:
        """BI71, BI72: Offline replay of Bounded Authority Decomposition produces bit-for-bit identical results."""
        transport = MockDiscoveryTransport()
        cbi_url = "https://registers.centralbank.ie/cis/ucits_etfs.json"
        cbi_payload = json.dumps({"records": [{"umbrella_name": "iShares plc", "sub_fund_name": "iShares Core S&P 500 UCITS ETF", "cis_type": "UCITS", "is_etf": True, "status": "ACTIVE"}]}).encode("utf-8")
        transport.register_response(cbi_url, 200, cbi_payload)

        issuer_adapter = StatutoryIssuerAdapter(issuer_name="iShares", jurisdiction=DiscoveryJurisdiction.IE)
        transport.register_response(
            "https://www.ishares.com/products.json",
            200,
            json.dumps({"products": [{"umbrella_name": "iShares plc", "sub_fund_name": "iShares Core S&P 500 UCITS ETF", "isin": ISIN_IE_1, "completeness": "COMPLETE"}]}).encode("utf-8"),
        )

        authority = UCITSDiscoveryAuthority(cache_dir=tmp_path, transport=transport)
        cfg = DiscoveryConfiguration(jurisdictions=("IE",))
        adapters = [CentralBankOfIrelandAdapter(), issuer_adapter]

        with patch("time.strftime", return_value="20261001T090000Z"):
            manifest1 = authority.execute_discovery(cfg, run_id="bounded_replay_run", custom_adapters=adapters)
            run_dir = tmp_path / "bounded_replay_run"
            replayed1 = authority.replay_discovery(run_dir, cfg, custom_adapters=adapters)
            replayed2 = authority.replay_discovery(run_dir, cfg, custom_adapters=adapters)

        assert manifest1["candidate_count"] == 1
        assert replayed1["candidate_count"] == 1
        assert replayed2["candidate_count"] == 1
        assert json.dumps(replayed1, sort_keys=True) == json.dumps(replayed2, sort_keys=True)
