"""
tests/test_etf_global_identity_resolver_wave3.py

Comprehensive 36-function unit and domain-integration test suite for Wave 3
GlobalETFIdentityResolver, closed input/output schemas, query normalization,
identifier precedence, multi-adapter reconciliation, fail-closed ambiguity and
authority-failure handling, and air-gapped provenance preservation.
"""

from __future__ import annotations

import copy
from dataclasses import fields
import json
from pathlib import Path
from typing import Sequence, Tuple

from scripts.research.etf_v2.global_identifier_authority import WKN_GLOBAL_CANONICAL_ID
from scripts.research.etf_v2.global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentityStatus,
    IdentifierType as Wave1IdentifierType,
    Jurisdiction,
)
from scripts.research.etf_v2.global_identity_resolver import (
    ADAPTER_REGISTRATION_MODEL,
    ADAPTER_SELECTION_MODEL,
    CANDIDATE_INPUT_ORDER_AFFECTS_RESULT,
    CANONICAL_OUTPUT_ORDER_DETERMINISTIC,
    CONFLICT_RECONCILIATION_MODEL,
    MULTI_ADAPTER_QUERY_ALLOWED,
    REGISTRATION_ORDER_AFFECTS_RESULT,
    RESPONSE_ORDER_AFFECTS_RESULT,
    AuthorityAdapterRegistry,
    AuthorityShareClassEntry,
    ETFAuthorityAdapterProtocol,
    GenericBrokerAliasAuthorityAdapter,
    GlobalETFIdentityResolver,
    UCITSResolverAuthorityAdapter,
    USSECResolverAuthorityAdapter,
)
from scripts.research.etf_v2.global_identity_resolver_models import (
    ADAPTER_FAILURE_EQUALS_NOT_FOUND,
    AMBIGUITY_FAILS_CLOSED,
    AMBIGUOUS_RESULT_SELECTS_FIRST_CANDIDATE,
    CANONICAL_GLOBAL_IDENTITY_MODEL,
    DETERMINISTIC_REASON_CODE_REQUIRED,
    FIRST_MATCH_WINS,
    FREE_TEXT_REASON_AS_AUTHORITY,
    INPUT_SCHEMA_CLOSED,
    NOT_FOUND_MEANS_GLOBAL_ABSENCE,
    OUTPUT_SCHEMA_CLOSED,
    PARTIAL_AUTHORITY_FAILURE_FAILS_CLOSED,
    REASON_CODE_SCHEMA_CLOSED,
    RESOLUTION_CONFIDENCE_MODEL,
    RESOLUTION_IDEMPOTENT_FOR_IDENTICAL_AUTHORITY_STATE,
    RESOLVER_MUTATION_MODEL,
    TIMESTAMP_IN_AUTHORITATIVE_RESOLUTION_OUTPUT,
    WAVE_3_SCOPE,
    AdapterCapabilities,
    AdapterExecutionOutcome,
    AdapterResolutionResult,
    BrokerAliasRecord,
    ETFIdentityCandidate,
    ETFIdentityQuery,
    ETFIdentityResolution,
    NormalizedETFIdentityQuery,
    ProvenanceReference,
    ResolutionReason,
    ResolutionStatus,
    ResolverIdentifierType,
    ResolverQueryClass,
    precedence_rank_for_field,
)
from scripts.research.etf_v2.models import EntityIdentity
from scripts.research.etf_v2.ucits_authority_adapter import UCITSAuthorityAdapter


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "resolver"
    / "global_identity_resolver_wave3_fixtures.json"
)


def _load_fixture_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_entry_from_dict(item: dict) -> AuthorityShareClassEntry:
    inst_d = item["instrument"]
    sc_d = item["share_class"]
    prov_d = item["provenance"]

    listings: list[ETFListing] = []
    for l_d in item["listings"]:
        listings.append(
            ETFListing(
                listing_id=l_d["listing_id"],
                share_class_id=l_d["share_class_id"],
                venue_mic=l_d["venue_mic"],
                venue_name=l_d["venue_name"],
                ticker=l_d["ticker"],
                trading_currency=l_d["trading_currency"],
                local_code=l_d.get("local_code"),
                identity_status=IdentityStatus.RESOLVED,
            )
        )

    sc = ETFShareClass(
        share_class_id=sc_d["share_class_id"],
        instrument_id=sc_d["instrument_id"],
        isin=sc_d.get("isin"),
        share_class_name=sc_d["share_class_name"],
        distribution_policy=sc_d.get("distribution_policy", ""),
        base_currency=sc_d.get("base_currency", "USD"),
        hedging_policy=sc_d.get("hedging_policy", ""),
        sec_class_id=sc_d.get("sec_class_id"),
        identity_status=IdentityStatus.RESOLVED,
        listings=tuple(listings),
    )

    inst = ETFInstrument(
        canonical_instrument_id=inst_d["canonical_instrument_id"],
        legal_fund_name=inst_d["legal_fund_name"],
        domicile_iso2=inst_d["domicile_iso2"],
        regulatory_jurisdiction=Jurisdiction(inst_d["regulatory_jurisdiction"]),
        fund_family=inst_d.get("umbrella_name", ""),
        issuer=inst_d.get("issuer_name", ""),
        fund_structure="UCITS_ICAV" if inst_d.get("ucits_compliant") else "1940_ACT_OPEN_END_ETF",
        identity_status=IdentityStatus.RESOLVED,
        share_classes=(sc,),
    )

    prov = ProvenanceReference(
        adapter_id=prov_d["adapter_id"],
        authority_jurisdiction=prov_d["authority_jurisdiction"],
        authority_source=prov_d["authority_source"],
        matched_identifier=sc.share_class_id,
        matched_identifier_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID.value,
        source_record_id=prov_d["source_record_id"],
        source_document_hash=prov_d.get("source_document_hash"),
        outcome="COMPLETED_MATCH",
    )

    return AuthorityShareClassEntry(
        instrument=inst,
        share_class=sc,
        listings=tuple(listings),
        provenance_references=(prov,),
        wkn_codes=tuple(item.get("wkn_codes", [])),
    )


def _build_default_resolver() -> Tuple[
    GlobalETFIdentityResolver,
    Tuple[AuthorityShareClassEntry, ...],
    Tuple[AuthorityShareClassEntry, ...],
    Tuple[BrokerAliasRecord, ...],
]:
    payload = _load_fixture_payload()
    us_entries = tuple(_build_entry_from_dict(x) for x in payload["us_sec_entries"])
    eu_entries = tuple(_build_entry_from_dict(x) for x in payload["eu_ucits_entries"])

    alias_records: list[BrokerAliasRecord] = []
    for a_d in payload["broker_alias_records"]:
        p_d = a_d["provenance"]
        prov = ProvenanceReference(
            adapter_id=p_d["adapter_id"],
            authority_jurisdiction=p_d["authority_jurisdiction"],
            authority_source=p_d["authority_source"],
            matched_identifier=a_d["raw_broker_alias"],
            matched_identifier_type=ResolverIdentifierType.BROKER_ALIAS.value,
            source_record_id=p_d["source_record_id"],
            source_document_hash=p_d.get("source_document_hash"),
            outcome="COMPLETED_MATCH",
        )
        alias_records.append(
            BrokerAliasRecord(
                broker_source_namespace=a_d["broker_source_namespace"],
                raw_broker_alias=a_d["raw_broker_alias"],
                normalized_broker_alias=a_d["normalized_broker_alias"],
                target_share_class_id=a_d["target_share_class_id"],
                target_listing_id=a_d.get("target_listing_id"),
                provenance_reference=prov,
                effective_from=a_d.get("effective_from"),
                effective_to=a_d.get("effective_to"),
                active=bool(a_d.get("active", True)),
            )
        )

    us_adapter = USSECResolverAuthorityAdapter(entries=us_entries)
    eu_adapter = UCITSResolverAuthorityAdapter(entries=eu_entries)
    broker_adapter = GenericBrokerAliasAuthorityAdapter(
        alias_records=alias_records,
        statutory_entries=us_entries + eu_entries,
    )

    resolver = GlobalETFIdentityResolver(adapters=[us_adapter, eu_adapter, broker_adapter])
    return resolver, us_entries, eu_entries, tuple(alias_records)


class _StubStaticAdapter:
    """Helper stub implementing ETFAuthorityAdapterProtocol for adversarial tests."""

    def __init__(
        self,
        adapter_id: str,
        capabilities: AdapterCapabilities,
        result: AdapterResolutionResult | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._adapter_id = adapter_id
        self._capabilities = capabilities
        self._result = result
        self._raise_exc = raise_exc

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def supports(self, normalized_query: NormalizedETFIdentityQuery) -> bool:
        return normalized_query.query_class in self._capabilities.supported_query_classes

    def resolve(self, normalized_query: NormalizedETFIdentityQuery) -> AdapterResolutionResult:
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._result is not None
        return self._result


# ==============================================================================
# 36 MANDATORY WAVE 3 TEST FUNCTIONS
# ==============================================================================


def test_canonical_internal_id_resolution() -> None:
    """1. Resolves canonical internal IDs (etfs:v1:, etfl:v1:, single-share-class etfi:v1:) with EXACT_CANONICAL_ID_MATCH."""
    resolver, _, _, _ = _build_default_resolver()

    # Share-class canonical ID (EU UCITS)
    res_sc = resolver.resolve(ETFIdentityQuery(raw_query="  etfs:v1:isin:ie000xagscy5  "))
    assert res_sc.resolution_status == ResolutionStatus.RESOLVED
    assert res_sc.reason_code == ResolutionReason.EXACT_CANONICAL_ID_MATCH
    assert res_sc.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res_sc.matched_identifier == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res_sc.matched_identifier_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID
    assert res_sc.instrument_identity is not None
    assert res_sc.instrument_identity.canonical_instrument_id == "etfi:v1:EU_UCITS:IE:C468159"
    assert len(res_sc.listing_identities) == 3

    # Listing canonical ID (US SEC)
    res_listing = resolver.resolve(ETFIdentityQuery(raw_query="etfl:v1:ARCX:VTI:USD"))
    assert res_listing.resolution_status == ResolutionStatus.RESOLVED
    assert res_listing.reason_code == ResolutionReason.EXACT_CANONICAL_ID_MATCH
    assert res_listing.canonical_internal_id == "etfs:v1:SEC_CLASS_ID:C000011906"
    assert len(res_listing.listing_identities) == 1
    assert res_listing.listing_identities[0].listing_id == "etfl:v1:ARCX:VTI:USD"

    # Single-share-class instrument canonical ID
    res_inst = resolver.resolve(ETFIdentityQuery(raw_query="etfi:v1:US_SEC:US:S000004310"))
    assert res_inst.resolution_status == ResolutionStatus.RESOLVED
    assert res_inst.reason_code == ResolutionReason.EXACT_CANONICAL_ID_MATCH
    assert res_inst.canonical_internal_id == "etfs:v1:SEC_CLASS_ID:C000011906"


def test_valid_isin_resolution() -> None:
    """2. Resolves valid ISO 6166 ISINs across EU UCITS and US SEC authorities with EXACT_ISIN_MATCH."""
    resolver, _, _, _ = _build_default_resolver()

    # EU UCITS ISIN with lowercase, whitespace, and hyphens
    res_ucits = resolver.resolve(ETFIdentityQuery(raw_query="  ie00-0xag-scy5 "))
    assert res_ucits.resolution_status == ResolutionStatus.RESOLVED
    assert res_ucits.reason_code == ResolutionReason.EXACT_ISIN_MATCH
    assert res_ucits.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res_ucits.matched_identifier == "IE000XAGSCY5"
    assert res_ucits.matched_identifier_type == ResolverIdentifierType.ISIN
    assert res_ucits.share_class_identity is not None
    assert res_ucits.share_class_identity.isin == "IE000XAGSCY5"

    # US SEC ISIN
    res_us = resolver.resolve(ETFIdentityQuery(raw_query="US9229087690"))
    assert res_us.resolution_status == ResolutionStatus.RESOLVED
    assert res_us.reason_code == ResolutionReason.EXACT_ISIN_MATCH
    assert res_us.canonical_internal_id == "etfs:v1:SEC_CLASS_ID:C000011906"


def test_invalid_isin_format() -> None:
    """3. Rejects malformed ISIN strings with INVALID_IDENTIFIER and INVALID_ISIN_FORMAT."""
    resolver, _, _, _ = _build_default_resolver()

    # Too short (11 chars) with ISIN hint
    res_short = resolver.resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY", identifier_type_hint=ResolverIdentifierType.ISIN)
    )
    assert res_short.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_short.reason_code == ResolutionReason.INVALID_ISIN_FORMAT
    assert res_short.canonical_internal_id is None
    assert res_short.ambiguity_candidates == ()

    # Invalid ISO-3166 country prefix
    res_prefix = resolver.resolve(
        ETFIdentityQuery(raw_query="ZZ000XAGSCY5", identifier_type_hint=ResolverIdentifierType.ISIN)
    )
    assert res_prefix.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_prefix.reason_code == ResolutionReason.INVALID_ISIN_FORMAT


def test_invalid_isin_check_digit() -> None:
    """4. Rejects 12-char ISINs with bad ISO 6166 Mod-10 check digits with INVALID_ISIN_CHECK_DIGIT."""
    resolver, _, _, _ = _build_default_resolver()

    # IE000XAGSCY5 has valid check digit 5; IE000XAGSCY4 has invalid check digit 4
    res_bad_digit = resolver.resolve(ETFIdentityQuery(raw_query="IE000XAGSCY4"))
    assert res_bad_digit.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_bad_digit.reason_code == ResolutionReason.INVALID_ISIN_CHECK_DIGIT
    assert res_bad_digit.canonical_internal_id is None
    assert res_bad_digit.share_class_identity is None
    assert res_bad_digit.listing_identities == ()


def test_valid_wkn_resolution() -> None:
    """5. Resolves valid 6-character WKN to canonical share class with EXACT_WKN_MATCH while WKN_GLOBAL_CANONICAL_ID is False."""
    assert WKN_GLOBAL_CANONICAL_ID is False
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(ETFIdentityQuery(raw_query=" a3e-40r "))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.EXACT_WKN_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res.matched_identifier == "A3E40R"
    assert res.matched_identifier_type == ResolverIdentifierType.WKN
    assert not res.canonical_internal_id.endswith("A3E40R")


def test_valid_wkn_no_match() -> None:
    """6. Returns bounded NOT_FOUND with NO_AUTHORITY_MATCH when a syntactically valid WKN is absent from registered authorities."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(
        ETFIdentityQuery(raw_query="A9Z999", identifier_type_hint=ResolverIdentifierType.WKN)
    )
    assert res.resolution_status == ResolutionStatus.NOT_FOUND
    assert res.reason_code == ResolutionReason.NO_AUTHORITY_MATCH
    assert res.canonical_internal_id is None
    assert res.authority_adapter_ids == ("eu_ucits_statutory_adapter",)
    assert len(res.provenance_references) == 1
    assert res.provenance_references[0].outcome == "COMPLETED_NO_MATCH"


def test_invalid_wkn_format() -> None:
    """7. Rejects malformed WKN strings with INVALID_IDENTIFIER and INVALID_WKN_FORMAT."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(
        ETFIdentityQuery(raw_query="A3E40R99", identifier_type_hint=ResolverIdentifierType.WKN)
    )
    assert res.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res.reason_code == ResolutionReason.INVALID_WKN_FORMAT
    assert res.canonical_internal_id is None


def test_ticker_with_mic_resolution() -> None:
    """8. Resolves ticker + MIC (both via mic_hint and explicit TICKER@MIC / MIC:TICKER syntax) with UNIQUE_TICKER_LISTING_MATCH."""
    resolver, _, _, _ = _build_default_resolver()

    # Cross-jurisdiction colliding ticker SHRT disambiguated by MIC=XPAR
    res_hint = resolver.resolve(ETFIdentityQuery(raw_query="SHRT", mic_hint="XPAR"))
    assert res_hint.resolution_status == ResolutionStatus.RESOLVED
    assert res_hint.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res_hint.canonical_internal_id == "etfs:v1:ISIN:IE00B4L5Y983"
    assert len(res_hint.listing_identities) == 1
    assert res_hint.listing_identities[0].listing_id == "etfl:v1:XPAR:SHRT:EUR"

    # Explicit composite syntax BLCH@XETR
    res_at = resolver.resolve(ETFIdentityQuery(raw_query="blch@xetr"))
    assert res_at.resolution_status == ResolutionStatus.RESOLVED
    assert res_at.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res_at.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert len(res_at.listing_identities) == 1
    assert res_at.listing_identities[0].venue_mic == "XETR"


def test_ticker_with_venue_resolution() -> None:
    """9. Resolves ticker + venue_hint with UNIQUE_TICKER_LISTING_MATCH and filters listing_identities to matching venue."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(ETFIdentityQuery(raw_query="BLCH", venue_hint="Tradegate"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert len(res.listing_identities) == 1
    assert res.listing_identities[0].venue_mic == "TGAT"


def test_ticker_with_currency_resolution() -> None:
    """10. Resolves ticker + currency_hint with UNIQUE_TICKER_LISTING_MATCH."""
    resolver, _, _, _ = _build_default_resolver()

    # SHRT exists on ARCX in USD and on XETR in EUR; currency_hint='EUR' uniquely selects IE00B4L5Y983
    res = resolver.resolve(ETFIdentityQuery(raw_query="SHRT", currency_hint="eur"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE00B4L5Y983"
    assert len(res.listing_identities) == 1
    assert res.listing_identities[0].trading_currency == "EUR"


def test_bare_ticker_unique_resolution() -> None:
    """11. Resolves bare ticker when it maps to a single canonical share class across all registered authorities."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(ETFIdentityQuery(raw_query="VTI"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res.canonical_internal_id == "etfs:v1:SEC_CLASS_ID:C000011906"
    assert res.matched_identifier == "VTI"
    assert res.matched_identifier_type == ResolverIdentifierType.TICKER


def test_bare_ticker_ambiguous_across_venues_or_jurisdictions() -> None:
    """12. Fails closed with AMBIGUOUS when a bare ticker collides across multiple share classes / jurisdictions."""
    resolver, _, _, _ = _build_default_resolver()

    # SHRT is listed under US SEC (C000227599) and EU UCITS (IE00B4L5Y983)
    res = resolver.resolve(ETFIdentityQuery(raw_query="SHRT"))
    assert res.resolution_status == ResolutionStatus.AMBIGUOUS
    assert res.reason_code == ResolutionReason.MULTIPLE_JURISDICTIONS
    assert res.canonical_internal_id is None
    assert res.share_class_identity is None
    assert res.listing_identities == ()
    assert len(res.ambiguity_candidates) == 2
    assert [c.share_class_identity.share_class_id for c in res.ambiguity_candidates] == [
        "etfs:v1:ISIN:IE00B4L5Y983",
        "etfs:v1:SEC_CLASS_ID:C000227599",
    ]


def test_legal_name_unique_resolution() -> None:
    """13. Resolves exact legal share-class/fund name when unique across registered authorities with UNIQUE_NAME_MATCH."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="  Global X ETFs ICAV - Global X Blockchain UCITS ETF  ",
            identifier_type_hint=ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
        )
    )
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_NAME_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"


def test_legal_name_ambiguous_resolution() -> None:
    """14. Fails closed with AMBIGUOUS (MULTIPLE_NAME_MATCHES) when a fund legal name matches multiple share classes."""
    resolver, _, _, _ = _build_default_resolver()

    # Umbrella/fund legal name shared by Acc (IE00B4L5Y983) and Dist (IE00B0M62Q58) share classes
    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="iShares III plc - iShares Core MSCI World UCITS ETF",
            identifier_type_hint=ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
        )
    )
    assert res.resolution_status == ResolutionStatus.AMBIGUOUS
    assert res.reason_code == ResolutionReason.MULTIPLE_NAME_MATCHES
    assert res.canonical_internal_id is None
    assert res.share_class_identity is None
    assert res.instrument_identity is not None
    assert res.instrument_identity.canonical_instrument_id == "etfi:v1:EU_UCITS:IE:C398100"
    assert len(res.ambiguity_candidates) == 2


def test_normalized_name_unique_resolution() -> None:
    """15. Resolves normalized share-class name (stripping trademark symbols while preserving tranche qualifiers)."""
    resolver, _, _, _ = _build_default_resolver()

    # Fixture share class name is "Xtrackers® MSCI World Swap UCITS ETF 1C"
    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="xtrackers msci world swap ucits etf 1c",
            identifier_type_hint=ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME,
        )
    )
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_NAME_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:LU0274208692"


def test_broker_alias_unique_through_registered_authority() -> None:
    """16. Resolves a namespace-scoped broker alias (e.g. GLXETFS-BLOCKCH DLA) via registered GenericBrokerAliasAuthorityAdapter."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="  glxetfs-blockch dla ",
            identifier_type_hint=ResolverIdentifierType.BROKER_ALIAS,
            broker_source_hint="DE_BROKER_XETR_FEED",
        )
    )
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_BROKER_ALIAS_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res.matched_identifier == "GLXETFS-BLOCKCH DLA"
    assert res.matched_identifier_type == ResolverIdentifierType.BROKER_ALIAS
    assert len(res.listing_identities) == 1
    assert res.listing_identities[0].listing_id == "etfl:v1:XETR:BLCH:EUR"
    assert len(res.provenance_references) == 2


def test_broker_alias_ambiguous_collision() -> None:
    """17. Fails closed with AMBIGUOUS and MULTIPLE_ALIAS_MATCHES when a broker alias maps to multiple share classes."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="ISHARES-MSCI-WORLD-USD",
            identifier_type_hint=ResolverIdentifierType.BROKER_ALIAS,
            broker_source_hint="DE_BROKER_XETR_FEED",
        )
    )
    assert res.resolution_status == ResolutionStatus.AMBIGUOUS
    assert res.reason_code == ResolutionReason.MULTIPLE_ALIAS_MATCHES
    assert res.canonical_internal_id is None
    assert len(res.ambiguity_candidates) == 2


def test_broker_alias_without_namespace_rejected() -> None:
    """18. Rejects BROKER_ALIAS queries missing broker_source_hint or targeting an unregistered broker namespace."""
    resolver, _, _, _ = _build_default_resolver()

    # Missing broker_source_hint
    res_no_ns = resolver.resolve(
        ETFIdentityQuery(
            raw_query="GLXETFS-BLOCKCH DLA",
            identifier_type_hint=ResolverIdentifierType.BROKER_ALIAS,
        )
    )
    assert res_no_ns.resolution_status == ResolutionStatus.UNSUPPORTED_QUERY_TYPE
    assert res_no_ns.reason_code == ResolutionReason.NO_APPLICABLE_ADAPTER
    assert res_no_ns.canonical_internal_id is None

    # Unregistered broker_source_hint
    res_unknown_ns = resolver.resolve(
        ETFIdentityQuery(
            raw_query="GLXETFS-BLOCKCH DLA",
            identifier_type_hint=ResolverIdentifierType.BROKER_ALIAS,
            broker_source_hint="UNREGISTERED_BROKER_XYZ",
        )
    )
    assert res_unknown_ns.resolution_status == ResolutionStatus.UNSUPPORTED_QUERY_TYPE
    assert res_unknown_ns.reason_code == ResolutionReason.NO_APPLICABLE_ADAPTER


def test_multiple_listings_same_share_class_resolves() -> None:
    """19. Multiple venue/currency listings of the SAME canonical share class resolve to RESOLVED with all listings ordered."""
    resolver, _, _, _ = _build_default_resolver()

    # BLCH is listed on TGAT (EUR), XETR (EUR), and XLON (USD) all under share class IE000XAGSCY5
    res = resolver.resolve(ETFIdentityQuery(raw_query="BLCH"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert [l.listing_id for l in res.listing_identities] == [
        "etfl:v1:TGAT:BLCH:EUR",
        "etfl:v1:XETR:BLCH:EUR",
        "etfl:v1:XLON:BLCH:USD",
    ]


def test_multiple_share_classes_under_same_instrument_ambiguous() -> None:
    """20. Instrument-level query matching an umbrella/sub-fund with >= 2 share classes fails closed with MULTIPLE_SHARE_CLASSES."""
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(ETFIdentityQuery(raw_query="etfi:v1:EU_UCITS:IE:C398100"))
    assert res.resolution_status == ResolutionStatus.AMBIGUOUS
    assert res.reason_code == ResolutionReason.MULTIPLE_SHARE_CLASSES
    assert res.canonical_internal_id is None
    assert res.share_class_identity is None
    assert res.instrument_identity is not None
    assert res.instrument_identity.canonical_instrument_id == "etfi:v1:EU_UCITS:IE:C398100"
    assert [c.share_class_identity.share_class_id for c in res.ambiguity_candidates] == [
        "etfs:v1:ISIN:IE00B0M62Q58",
        "etfs:v1:ISIN:IE00B4L5Y983",
    ]


def test_contradictory_authoritative_identifiers_conflict() -> None:
    """21. Composite queries with contradictory authoritative or lower-precedence identifiers fail closed with AUTHORITY_CONFLICT."""
    resolver, _, _, _ = _build_default_resolver()
    assert FIRST_MATCH_WINS is False

    # Valid ISIN (IE000XAGSCY5) + valid WKN belonging to a different share class (A0RPWH -> IE00B4L5Y983)
    res_auth = resolver.resolve(
        ETFIdentityQuery(raw_query="ISIN=IE000XAGSCY5;WKN=A0RPWH")
    )
    assert res_auth.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT
    assert res_auth.reason_code == ResolutionReason.CONFLICTING_AUTHORITATIVE_IDENTIFIERS
    assert res_auth.canonical_internal_id is None
    assert len(res_auth.ambiguity_candidates) == 2

    # Valid ISIN (IE000XAGSCY5) + incompatible ticker (VTI -> C000011906)
    res_cross = resolver.resolve(
        ETFIdentityQuery(raw_query="ISIN=IE000XAGSCY5;TICKER=VTI")
    )
    assert res_cross.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT
    assert res_cross.reason_code == ResolutionReason.CONFLICTING_AUTHORITATIVE_IDENTIFIERS

    # Corroborating ISIN + matching WKN resolves cleanly
    res_ok = resolver.resolve(
        ETFIdentityQuery(raw_query="ISIN=IE000XAGSCY5;WKN=A3E40R")
    )
    assert res_ok.resolution_status == ResolutionStatus.RESOLVED
    assert res_ok.reason_code == ResolutionReason.EXACT_ISIN_MATCH
    assert res_ok.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"


def test_contradictory_listing_context_fails_closed() -> None:
    """22. Fails closed with AUTHORITY_CONFLICT and CONFLICTING_LISTING_CONTEXT when hints contradict authoritative identity."""
    resolver, _, _, _ = _build_default_resolver()

    # Valid IE UCITS ISIN paired with incompatible US MIC (ARCX)
    res_mic = resolver.resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5", mic_hint="ARCX")
    )
    assert res_mic.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT
    assert res_mic.reason_code == ResolutionReason.CONFLICTING_LISTING_CONTEXT
    assert res_mic.canonical_internal_id is None
    assert len(res_mic.ambiguity_candidates) == 1
    assert res_mic.ambiguity_candidates[0].share_class_identity.share_class_id == "etfs:v1:ISIN:IE000XAGSCY5"

    # Valid IE UCITS ISIN paired with contradictory jurisdiction_hint="US_SEC"
    res_jur = resolver.resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5", jurisdiction_hint="US_SEC")
    )
    assert res_jur.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT
    assert res_jur.reason_code == ResolutionReason.CONFLICTING_LISTING_CONTEXT


def test_invalid_explicit_mic_and_type_hints_rejected() -> None:
    """23. Rejects invalid mic_hint or invalid identifier_type_hint without silently ignoring them."""
    resolver, _, _, _ = _build_default_resolver()

    # Invalid MIC (not 4 alphanumeric chars)
    res_mic = resolver.resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5", mic_hint="INVALID_MIC_99")
    )
    assert res_mic.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_mic.reason_code == ResolutionReason.INVALID_MIC
    assert res_mic.canonical_internal_id is None

    # Invalid identifier_type_hint string
    res_hint = resolver.resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5", identifier_type_hint="NON_EXISTENT_TYPE")
    )
    assert res_hint.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_hint.reason_code == ResolutionReason.INVALID_QUERY

    # Valid ticker passed with explicit ISIN hint must fail as INVALID_ISIN_FORMAT (zero fallthrough)
    res_mismatch = resolver.resolve(
        ETFIdentityQuery(raw_query="VTI", identifier_type_hint=ResolverIdentifierType.ISIN)
    )
    assert res_mismatch.resolution_status == ResolutionStatus.INVALID_IDENTIFIER
    assert res_mismatch.reason_code == ResolutionReason.INVALID_ISIN_FORMAT


def test_multi_adapter_same_share_class_merge() -> None:
    """24. Merges compatible candidates for the same canonical share class across multiple adapters and unions provenance."""
    _, _, eu_entries, _ = _build_default_resolver()
    base_entry = eu_entries[0]  # IE000XAGSCY5

    # Adapter A has XETR listing; Adapter B has XLON listing for the exact same share class
    entry_a = AuthorityShareClassEntry(
        instrument=base_entry.instrument,
        share_class=ETFShareClass(
            share_class_id=base_entry.share_class.share_class_id,
            instrument_id=base_entry.share_class.instrument_id,
            isin=base_entry.share_class.isin,
            share_class_name=base_entry.share_class.share_class_name,
            distribution_policy=base_entry.share_class.distribution_policy,
            base_currency=base_entry.share_class.base_currency,
            hedging_policy=base_entry.share_class.hedging_policy,
            listings=(base_entry.listings[0],),
        ),
        listings=(base_entry.listings[0],),
        provenance_references=(
            ProvenanceReference(
                adapter_id="ucits_adapter_alpha",
                authority_jurisdiction="EU_UCITS",
                authority_source="CBI_REGISTER_ALPHA",
                matched_identifier="IE000XAGSCY5",
                matched_identifier_type="ISIN",
                source_record_id="ALPHA:1",
            ),
        ),
        wkn_codes=("A3E40R",),
    )
    entry_b = AuthorityShareClassEntry(
        instrument=base_entry.instrument,
        share_class=ETFShareClass(
            share_class_id=base_entry.share_class.share_class_id,
            instrument_id=base_entry.share_class.instrument_id,
            isin=base_entry.share_class.isin,
            share_class_name=base_entry.share_class.share_class_name,
            distribution_policy=base_entry.share_class.distribution_policy,
            base_currency=base_entry.share_class.base_currency,
            hedging_policy=base_entry.share_class.hedging_policy,
            listings=(base_entry.listings[2],),
        ),
        listings=(base_entry.listings[2],),
        provenance_references=(
            ProvenanceReference(
                adapter_id="ucits_adapter_beta",
                authority_jurisdiction="EU_UCITS",
                authority_source="ISSUER_SUPPLEMENT_BETA",
                matched_identifier="IE000XAGSCY5",
                matched_identifier_type="ISIN",
                source_record_id="BETA:2",
            ),
        ),
        wkn_codes=("A3E40R",),
    )

    adapter_a = UCITSResolverAuthorityAdapter(adapter_id="ucits_adapter_alpha", entries=(entry_a,))
    adapter_b = UCITSResolverAuthorityAdapter(adapter_id="ucits_adapter_beta", entries=(entry_b,))
    resolver = GlobalETFIdentityResolver(adapters=[adapter_a, adapter_b])

    res = resolver.resolve(ETFIdentityQuery(raw_query="IE000XAGSCY5"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.EXACT_ISIN_MATCH
    assert res.canonical_internal_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert res.authority_adapter_ids == ("ucits_adapter_alpha", "ucits_adapter_beta")
    assert [l.venue_mic for l in res.listing_identities] == ["XETR", "XLON"]
    assert len(res.provenance_references) == 2


def test_multi_adapter_incompatible_authority_conflict() -> None:
    """25. Fails closed with AUTHORITY_CONFLICT and CONFLICTING_ADAPTER_IDENTITIES when adapters return incompatible identities for one ISIN."""
    _, _, eu_entries, _ = _build_default_resolver()
    entry_a = eu_entries[0]  # IE000XAGSCY5 ACCUMULATING

    # Create a conflicting entry for the same share_class_id with DISTRIBUTING policy
    conflicting_sc = ETFShareClass(
        share_class_id=entry_a.share_class.share_class_id,
        instrument_id=entry_a.share_class.instrument_id,
        isin=entry_a.share_class.isin,
        share_class_name="Conflicting Distributing Share Class",
        distribution_policy="DISTRIBUTING",
        base_currency="EUR",
        hedging_policy="HEDGED",
        listings=entry_a.listings,
    )
    entry_b = AuthorityShareClassEntry(
        instrument=entry_a.instrument,
        share_class=conflicting_sc,
        listings=entry_a.listings,
        provenance_references=(
            ProvenanceReference(
                adapter_id="ucits_adapter_conflicting",
                authority_jurisdiction="EU_UCITS",
                authority_source="CONFLICTING_SOURCE",
                matched_identifier="IE000XAGSCY5",
                matched_identifier_type="ISIN",
                source_record_id="CONFLICT:1",
            ),
        ),
    )

    adapter_a = UCITSResolverAuthorityAdapter(adapter_id="ucits_adapter_primary", entries=(entry_a,))
    adapter_b = UCITSResolverAuthorityAdapter(adapter_id="ucits_adapter_conflicting", entries=(entry_b,))
    resolver = GlobalETFIdentityResolver(adapters=[adapter_a, adapter_b])

    res = resolver.resolve(ETFIdentityQuery(raw_query="IE000XAGSCY5"))
    assert res.resolution_status == ResolutionStatus.AUTHORITY_CONFLICT
    assert res.reason_code == ResolutionReason.CONFLICTING_ADAPTER_IDENTITIES
    assert res.canonical_internal_id is None
    assert len(res.ambiguity_candidates) == 2


def test_adapter_registration_order_invariance() -> None:
    """26. Proves REGISTRATION_ORDER_AFFECTS_RESULT == False across all permutations of adapter registration."""
    assert REGISTRATION_ORDER_AFFECTS_RESULT is False
    _, us_entries, eu_entries, alias_records = _build_default_resolver()

    us_adapter = USSECResolverAuthorityAdapter(entries=us_entries)
    eu_adapter = UCITSResolverAuthorityAdapter(entries=eu_entries)
    broker_adapter = GenericBrokerAliasAuthorityAdapter(
        alias_records=alias_records,
        statutory_entries=us_entries + eu_entries,
    )

    resolver_1 = GlobalETFIdentityResolver(adapters=[us_adapter, eu_adapter, broker_adapter])
    resolver_2 = GlobalETFIdentityResolver(adapters=[broker_adapter, eu_adapter, us_adapter])
    resolver_3 = GlobalETFIdentityResolver(adapters=[eu_adapter, broker_adapter, us_adapter])

    queries = [
        ETFIdentityQuery(raw_query="IE000XAGSCY5"),
        ETFIdentityQuery(raw_query="VTI"),
        ETFIdentityQuery(raw_query="SHRT"),
        ETFIdentityQuery(
            raw_query="GLXETFS-BLOCKCH DLA",
            identifier_type_hint=ResolverIdentifierType.BROKER_ALIAS,
            broker_source_hint="DE_BROKER_XETR_FEED",
        ),
    ]
    for q in queries:
        r1 = resolver_1.resolve(q)
        r2 = resolver_2.resolve(q)
        r3 = resolver_3.resolve(q)
        assert r1 == r2 == r3
        assert r1.to_json() == r2.to_json() == r3.to_json()


def test_adapter_response_order_invariance() -> None:
    """27. Proves RESPONSE_ORDER_AFFECTS_RESULT == False when adapters return candidates/provenance in reversed order."""
    assert RESPONSE_ORDER_AFFECTS_RESULT is False
    _, us_entries, eu_entries, _ = _build_default_resolver()

    eu_forward = UCITSResolverAuthorityAdapter(adapter_id="eu_adapter", entries=eu_entries)
    eu_reversed = UCITSResolverAuthorityAdapter(adapter_id="eu_adapter", entries=tuple(reversed(eu_entries)))
    us_forward = USSECResolverAuthorityAdapter(adapter_id="us_adapter", entries=us_entries)
    us_reversed = USSECResolverAuthorityAdapter(adapter_id="us_adapter", entries=tuple(reversed(us_entries)))

    res_fwd = GlobalETFIdentityResolver(adapters=[us_forward, eu_forward]).resolve(
        ETFIdentityQuery(raw_query="iShares III plc - iShares Core MSCI World UCITS ETF")
    )
    res_rev = GlobalETFIdentityResolver(adapters=[us_reversed, eu_reversed]).resolve(
        ETFIdentityQuery(raw_query="iShares III plc - iShares Core MSCI World UCITS ETF")
    )
    assert res_fwd == res_rev
    assert res_fwd.to_json() == res_rev.to_json()


def test_candidate_and_listing_order_invariance() -> None:
    """28. Proves CANDIDATE_INPUT_ORDER_AFFECTS_RESULT == False and verifies exact 6-tuple candidate sort order."""
    assert CANDIDATE_INPUT_ORDER_AFFECTS_RESULT is False
    assert CANONICAL_OUTPUT_ORDER_DETERMINISTIC is True
    _, _, eu_entries, _ = _build_default_resolver()
    base = eu_entries[0]

    # Reverse the listings inside the entry
    reversed_listings_entry = AuthorityShareClassEntry(
        instrument=base.instrument,
        share_class=ETFShareClass(
            share_class_id=base.share_class.share_class_id,
            instrument_id=base.share_class.instrument_id,
            isin=base.share_class.isin,
            share_class_name=base.share_class.share_class_name,
            distribution_policy=base.share_class.distribution_policy,
            base_currency=base.share_class.base_currency,
            hedging_policy=base.share_class.hedging_policy,
            listings=tuple(reversed(base.listings)),
        ),
        listings=tuple(reversed(base.listings)),
        provenance_references=base.provenance_references,
        wkn_codes=base.wkn_codes,
    )

    r_normal = GlobalETFIdentityResolver(
        adapters=[UCITSResolverAuthorityAdapter(entries=(base,))]
    ).resolve(ETFIdentityQuery(raw_query="IE000XAGSCY5"))
    r_reversed = GlobalETFIdentityResolver(
        adapters=[UCITSResolverAuthorityAdapter(entries=(reversed_listings_entry,))]
    ).resolve(ETFIdentityQuery(raw_query="IE000XAGSCY5"))

    assert r_normal == r_reversed
    assert [l.venue_mic for l in r_reversed.listing_identities] == ["TGAT", "XETR", "XLON"]


def test_not_found_bounded_semantics() -> None:
    """29. Verifies NOT_FOUND is bounded to queried authority adapters and never claims global world absence."""
    assert NOT_FOUND_MEANS_GLOBAL_ABSENCE is False
    resolver, _, _, _ = _build_default_resolver()

    # Valid US ISIN check digit that is not in the fixture (US0378331005)
    res = resolver.resolve(ETFIdentityQuery(raw_query="US0378331005"))
    assert res.resolution_status == ResolutionStatus.NOT_FOUND
    assert res.reason_code == ResolutionReason.NO_AUTHORITY_MATCH
    assert res.canonical_internal_id is None
    assert res.instrument_identity is None
    assert res.share_class_identity is None
    assert res.listing_identities == ()
    assert res.ambiguity_candidates == ()
    assert res.authority_adapter_ids == ("us_sec_statutory_adapter",)
    assert len(res.provenance_references) == 1
    assert res.provenance_references[0].outcome == "COMPLETED_NO_MATCH"


def test_unsupported_query_type() -> None:
    """30. Returns UNSUPPORTED_QUERY_TYPE with NO_APPLICABLE_ADAPTER when no registered adapter supports the query class."""
    _, us_entries, _, _ = _build_default_resolver()
    # Register ONLY the US SEC adapter (which does not support WKN)
    us_only_resolver = GlobalETFIdentityResolver(
        adapters=[USSECResolverAuthorityAdapter(entries=us_entries)]
    )

    res = us_only_resolver.resolve(
        ETFIdentityQuery(raw_query="A3E40R", identifier_type_hint=ResolverIdentifierType.WKN)
    )
    assert res.resolution_status == ResolutionStatus.UNSUPPORTED_QUERY_TYPE
    assert res.reason_code == ResolutionReason.NO_APPLICABLE_ADAPTER
    assert res.canonical_internal_id is None
    assert res.authority_adapter_ids == ()


def test_authority_execution_failure_fails_closed() -> None:
    """31. Fails closed with first-class AUTHORITY_FAILURE when an adapter raises or returns UNAVAILABLE / INVALID_RESPONSE."""
    assert ADAPTER_FAILURE_EQUALS_NOT_FOUND is False

    caps = AdapterCapabilities(
        adapter_id="failing_adapter",
        supported_jurisdictions=frozenset({"EU_UCITS", "IE"}),
        supported_query_classes=frozenset({ResolverQueryClass.ISIN}),
        supported_identifier_namespaces=frozenset({"ISIN"}),
    )

    # Case 1: Adapter raises runtime exception
    raising_adapter = _StubStaticAdapter(
        adapter_id="failing_adapter",
        capabilities=caps,
        raise_exc=RuntimeError("Simulated statutory registry read error"),
    )
    res_exc = GlobalETFIdentityResolver(adapters=[raising_adapter]).resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5")
    )
    assert res_exc.resolution_status == ResolutionStatus.AUTHORITY_FAILURE
    assert res_exc.reason_code == ResolutionReason.AUTHORITY_EXECUTION_FAILURE
    assert res_exc.canonical_internal_id is None

    # Case 2: Adapter returns UNAVAILABLE
    unavail_adapter = _StubStaticAdapter(
        adapter_id="failing_adapter",
        capabilities=caps,
        result=AdapterResolutionResult(
            adapter_id="failing_adapter",
            outcome=AdapterExecutionOutcome.UNAVAILABLE,
            failure_reason=ResolutionReason.AUTHORITY_UNAVAILABLE,
        ),
    )
    res_unavail = GlobalETFIdentityResolver(adapters=[unavail_adapter]).resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5")
    )
    assert res_unavail.resolution_status == ResolutionStatus.AUTHORITY_FAILURE
    assert res_unavail.reason_code == ResolutionReason.AUTHORITY_UNAVAILABLE

    # Case 3: Adapter returns COMPLETED_MATCH with empty candidates (malformed response)
    invalid_adapter = _StubStaticAdapter(
        adapter_id="failing_adapter",
        capabilities=caps,
        result=AdapterResolutionResult(
            adapter_id="failing_adapter",
            outcome=AdapterExecutionOutcome.COMPLETED_MATCH,
            candidates=(),
        ),
    )
    res_invalid = GlobalETFIdentityResolver(adapters=[invalid_adapter]).resolve(
        ETFIdentityQuery(raw_query="IE000XAGSCY5")
    )
    assert res_invalid.resolution_status == ResolutionStatus.AUTHORITY_FAILURE
    assert res_invalid.reason_code == ResolutionReason.AUTHORITY_RESPONSE_INVALID


def test_partial_authority_failure_fails_closed() -> None:
    """32. Multi-adapter query where one adapter matches and another required adapter fails returns AUTHORITY_FAILURE (INCOMPLETE_AUTHORITY_SET)."""
    assert PARTIAL_AUTHORITY_FAILURE_FAILS_CLOSED is True
    _, _, eu_entries, _ = _build_default_resolver()

    healthy_ucits = UCITSResolverAuthorityAdapter(entries=eu_entries)
    failing_us_caps = AdapterCapabilities(
        adapter_id="failing_us_adapter",
        supported_jurisdictions=frozenset({"US_SEC", "US"}),
        supported_query_classes=frozenset({ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT}),
        supported_identifier_namespaces=frozenset({"TICKER"}),
    )
    failing_us = _StubStaticAdapter(
        adapter_id="failing_us_adapter",
        capabilities=failing_us_caps,
        result=AdapterResolutionResult(
            adapter_id="failing_us_adapter",
            outcome=AdapterExecutionOutcome.EXECUTION_FAILURE,
            failure_reason=ResolutionReason.AUTHORITY_EXECUTION_FAILURE,
        ),
    )

    # Unhinted ticker query 'BLCH' routes to both healthy_ucits (matches IE000XAGSCY5) and failing_us (fails)
    resolver = GlobalETFIdentityResolver(adapters=[healthy_ucits, failing_us])
    res = resolver.resolve(ETFIdentityQuery(raw_query="BLCH"))
    assert res.resolution_status == ResolutionStatus.AUTHORITY_FAILURE
    assert res.reason_code == ResolutionReason.INCOMPLETE_AUTHORITY_SET
    assert res.canonical_internal_id is None
    assert res.share_class_identity is None


def test_provenance_preservation_invariants() -> None:
    """33. Verifies provenance references are preserved on RESOLVED/AMBIGUOUS/CONFLICT and contain zero timestamps."""
    assert TIMESTAMP_IN_AUTHORITATIVE_RESOLUTION_OUTPUT is False
    resolver, _, _, _ = _build_default_resolver()

    res = resolver.resolve(ETFIdentityQuery(raw_query="IE000XAGSCY5"))
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert len(res.provenance_references) >= 1
    for p in res.provenance_references:
        assert p.adapter_id == "eu_ucits_statutory_adapter"
        assert p.authority_jurisdiction == "EU_UCITS"
        assert p.matched_identifier == "IE000XAGSCY5"
        assert p.matched_identifier_type == "ISIN"
        assert p.source_record_id == "CBI:C468159:IE000XAGSCY5"
        assert p.source_document_hash == "d1a002495057d6e6241d46e8d44f66d114d20f9e532b51cf0cf4b823df44920f"
        p_dict = p.to_dict()
        for k in p_dict:
            assert "timestamp" not in k.lower()
            assert "time" not in k.lower()


def test_read_only_and_idempotent_resolution() -> None:
    """34. Verifies RESOLVER_MUTATION_MODEL == READ_ONLY, closed 7-field/12-field schemas, and bit-identical idempotency."""
    assert RESOLVER_MUTATION_MODEL == "READ_ONLY"
    assert RESOLUTION_IDEMPOTENT_FOR_IDENTICAL_AUTHORITY_STATE is True
    assert INPUT_SCHEMA_CLOSED is True
    assert OUTPUT_SCHEMA_CLOSED is True
    assert RESOLUTION_CONFIDENCE_MODEL == "NOT_USED"
    assert DETERMINISTIC_REASON_CODE_REQUIRED is True
    assert REASON_CODE_SCHEMA_CLOSED is True
    assert FREE_TEXT_REASON_AS_AUTHORITY is False
    assert AMBIGUITY_FAILS_CLOSED is True
    assert AMBIGUOUS_RESULT_SELECTS_FIRST_CANDIDATE is False
    assert WAVE_3_SCOPE == "GLOBAL_IDENTITY_RESOLUTION_DOMAIN_FOUNDATION"
    assert CANONICAL_GLOBAL_IDENTITY_MODEL == "NORMALIZED_THREE_TIER_CORE_WITH_JURISDICTION_ADAPTERS"
    assert ADAPTER_REGISTRATION_MODEL == "EXPLICIT_DECLARATIVE_REGISTRY_WITH_UNIQUE_ADAPTER_IDS"
    assert ADAPTER_SELECTION_MODEL == "REQUEST_SEMANTICS_AND_ADAPTER_CAPABILITIES_ONLY"
    assert MULTI_ADAPTER_QUERY_ALLOWED is True
    assert CONFLICT_RECONCILIATION_MODEL == (
        "COLLECT_ALL_APPLICABLE_RESULTS_THEN_MERGE_EXACT_SHARE_CLASS_MATCHES_"
        "AND_FAIL_CLOSED_ON_INCOMPATIBLE_AUTHORITATIVE_IDENTITIES"
    )

    # Verify closed field counts on ETFIdentityQuery (7 fields) and ETFIdentityResolution (12 fields)
    assert len(fields(ETFIdentityQuery)) == 7
    assert len(fields(ETFIdentityResolution)) == 12
    assert len(ResolutionStatus) == 7
    assert len(ResolutionReason) == 26

    # Verify precedence ladder ranks 1..12
    assert precedence_rank_for_field(ResolverIdentifierType.CANONICAL_INTERNAL_ID) == 1
    assert precedence_rank_for_field(ResolverIdentifierType.ISIN) == 2
    assert precedence_rank_for_field(ResolverIdentifierType.WKN) == 3
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, True, True, True) == 4
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, True, True, False) == 5
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, True, False, False) == 6
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, False, True, False) == 7
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, False, False, True) == 8
    assert precedence_rank_for_field(ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME) == 9
    assert precedence_rank_for_field(ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME) == 10
    assert precedence_rank_for_field(ResolverIdentifierType.BROKER_ALIAS) == 11
    assert precedence_rank_for_field(ResolverIdentifierType.TICKER, False, False, False) == 12

    resolver, us_entries, eu_entries, alias_records = _build_default_resolver()
    snapshot_before = copy.deepcopy((us_entries, eu_entries, alias_records))

    q = ETFIdentityQuery(raw_query="IE000XAGSCY5")
    first_res = resolver.resolve(q)
    second_res = resolver.resolve(q)

    assert first_res == second_res
    assert first_res.to_json() == second_res.to_json()
    assert (us_entries, eu_entries, alias_records) == snapshot_before


def test_no_product_specific_global_x_branches() -> None:
    """35. Scans production Wave 3 domain modules to verify zero hardcoded Global X / BLCH / IE000XAGSCY5 / A3E40R literals."""
    repo_root = Path(__file__).resolve().parents[1]
    domain_files = [
        repo_root / "scripts" / "research" / "etf_v2" / "global_identity_resolver_models.py",
        repo_root / "scripts" / "research" / "etf_v2" / "global_identity_resolver.py",
    ]
    forbidden_literals = (
        "Global X",
        "global x",
        "BLCH",
        "IE000XAGSCY5",
        "A3E40R",
        "GLXETFS-BLOCKCH DLA",
        "C468159",
    )
    for fpath in domain_files:
        text = fpath.read_text(encoding="utf-8")
        for token in forbidden_literals:
            assert token not in text, f"Forbidden product-specific literal '{token}' found in {fpath.name}"


def test_us_sec_and_wave1_wave2_regression_preservation() -> None:
    """36. Verifies USSECResolverAuthorityAdapter and UCITSResolverAuthorityAdapter interoperate with Wave 1 & Wave 2 contracts."""
    sec_entity = EntityIdentity(
        symbol="IVV",
        cik="0001100663",
        series_id="S000004311",
        class_id="C000011907",
        legal_name="iShares Core S&P 500 ETF",
        historical_aliases=[],
    )
    sec_adapter = USSECResolverAuthorityAdapter(
        sec_entities=[(sec_entity, "ARCX", "USD")]
    )
    ucits_adapter = UCITSResolverAuthorityAdapter(ucits_adapter=UCITSAuthorityAdapter())
    assert isinstance(sec_adapter, ETFAuthorityAdapterProtocol)
    assert isinstance(ucits_adapter, ETFAuthorityAdapterProtocol)

    resolver = GlobalETFIdentityResolver(adapters=[sec_adapter, ucits_adapter])
    res = resolver.resolve(
        ETFIdentityQuery(
            raw_query="etfs:v1:SEC_CLASS_ID:C000011907",
            identifier_type_hint=Wave1IdentifierType.INTERNAL_SHARE_CLASS_ID,
        )
    )
    assert res.resolution_status == ResolutionStatus.RESOLVED
    assert res.reason_code == ResolutionReason.EXACT_CANONICAL_ID_MATCH
    assert res.canonical_internal_id == "etfs:v1:SEC_CLASS_ID:C000011907"
    assert res.instrument_identity is not None
    assert res.instrument_identity.regulatory_jurisdiction == Jurisdiction.US_SEC
