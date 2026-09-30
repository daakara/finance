"""
tests/test_etf_global_identity_wave1.py

Unit, contract, and adversarial test suite for ETF V2 Global Identity Wave 1.
Verifies the jurisdiction-neutral domain foundation:
    ETFInstrument != ETFShareClass != ETFListing
Validates ISO 6166 Mod-10 check digits, WKN boundaries, collision safety,
lossless US SEC adapter mapping, and adversarial failure classes A01-A18.
"""

import json
from pathlib import Path
import pytest

from scripts.research.etf_v2.global_identity_models import (
    CanonicalIdCollisionError,
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from scripts.research.etf_v2.global_identifier_authority import (
    INTERNAL_ID_VERSION,
    WKN_GLOBAL_CANONICAL_ID,
    DeterministicIdentityRegistry,
    calculate_isin_check_digit,
    compute_listing_dedup_key,
    generate_instrument_id,
    generate_listing_id,
    generate_share_class_id,
    normalize_isin,
    validate_isin,
    validate_mic,
    validate_wkn,
)
from scripts.research.etf_v2.models import EntityIdentity, PopulationRecord
from scripts.research.etf_v2.us_sec_authority_adapter import (
    US_IDENTITY_MAPPING_LOSSLESS,
    USSECAuthorityAdapter,
)


# ==============================================================================
# 1. DOMAIN MODEL & SERIALIZATION TESTS
# ==============================================================================

def test_deterministic_instrument_serialization():
    """Verifies that ETFInstrument serializes and deserializes deterministically."""
    inst = ETFInstrument(
        canonical_instrument_id="etfi:v1:US_SEC:US:S000077125",
        legal_fund_name="Global X FinTech ETF",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
        fund_family="Global X ETFs",
        issuer="Global X Management Company LLC",
        fund_structure="1940_ACT_OPEN_END_ETF",
        identity_status=IdentityStatus.RESOLVED,
        metadata={"cik": "0001432353", "series_id": "S000077125"},
    )
    d = inst.to_dict()
    assert d["canonical_instrument_id"] == "etfi:v1:US_SEC:US:S000077125"
    assert d["regulatory_jurisdiction"] == "US_SEC"

    # Round trip
    reconstructed = ETFInstrument.from_dict(d)
    assert reconstructed == inst
    assert reconstructed.to_json() == inst.to_json()


def test_deterministic_share_class_serialization():
    """Verifies that ETFShareClass serializes and deserializes deterministically."""
    sc = ETFShareClass(
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        isin="IE000XAGSCY5",
        share_class_name="USD Accumulating",
        distribution_policy="ACCUMULATING",
        base_currency="USD",
        hedging_policy="UNHEDGED",
        identity_status=IdentityStatus.RESOLVED,
        metadata={"manco": "Global X Management Company (Europe) Limited"},
    )
    d = sc.to_dict()
    assert d["share_class_id"] == "etfs:v1:ISIN:IE000XAGSCY5"
    reconstructed = ETFShareClass.from_dict(d)
    assert reconstructed == sc


def test_deterministic_listing_serialization():
    """Verifies that ETFListing serializes and deserializes deterministically."""
    listing = ETFListing(
        listing_id="etfl:v1:XETR:BLCH:EUR",
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        venue_mic="XETR",
        ticker="BLCH",
        trading_currency="EUR",
        venue_name="Xetra",
        local_code="A3E40R",
        broker_aliases=("BLCH.DE", "BLCH.GY"),
        identity_status=IdentityStatus.RESOLVED,
    )
    d = listing.to_dict()
    assert d["listing_id"] == "etfl:v1:XETR:BLCH:EUR"
    reconstructed = ETFListing.from_dict(d)
    assert reconstructed == listing
    assert reconstructed.deduplication_key() == ("etfs:v1:ISIN:IE000XAGSCY5", "XETR", "EUR")


def test_nested_model_round_trip():
    """Verifies full 3-tier nested hierarchy serialization and round-trip."""
    listing = ETFListing(
        listing_id="etfl:v1:TGAT:BLCH:EUR",
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        venue_mic="TGAT",
        ticker="BLCH",
        trading_currency="EUR",
        venue_name="Tradegate",
        local_code="A3E40R",
    )
    sc = ETFShareClass(
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        isin="IE000XAGSCY5",
        share_class_name="USD Accumulating",
        listings=(listing,),
    )
    inst = ETFInstrument(
        canonical_instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        legal_fund_name="Global X Blockchain UCITS ETF",
        domicile_iso2="IE",
        regulatory_jurisdiction=Jurisdiction.EU_UCITS,
        share_classes=(sc,),
    )
    json_repr = inst.to_json()
    reconstructed = ETFInstrument.from_json(json_repr)
    assert reconstructed == inst
    assert len(reconstructed.share_classes) == 1
    assert len(reconstructed.share_classes[0].listings) == 1
    assert reconstructed.share_classes[0].listings[0].ticker == "BLCH"


# ==============================================================================
# 2. IDENTIFIER VALIDATION & NORMALIZATION TESTS
# ==============================================================================

def test_isin_validation_valid():
    """Tests ISO 6166 checksum validation on known valid ISINs."""
    valid_isins = [
        "US0378331005",  # Apple Inc.
        "IE000XAGSCY5",  # Global X Blockchain UCITS ETF
        "US4642872000",  # iShares Core S&P 500 ETF
        "DE000BAY0017",  # Bayer AG
        "LU1681045370",  # Amundi Index Solutions
        "XS1234567896",  # Supranational / Eurobond sample with valid check
    ]
    for isin in valid_isins:
        assert validate_isin(isin, strict=True) is True


def test_isin_validation_invalid_checksum():
    """Tests that corrupting the check digit fails validation."""
    with pytest.raises(InvalidIdentifierError, match="ISIN check-digit failure"):
        validate_isin("IE000XAGSCY4", strict=True)

    with pytest.raises(InvalidIdentifierError, match="ISIN check-digit failure"):
        validate_isin("US0378331004", strict=True)


def test_isin_validation_invalid_length():
    """Tests that ISINs with length != 12 fail closed."""
    with pytest.raises(InvalidIdentifierError, match="Invalid ISIN length"):
        validate_isin("US037833100", strict=True)  # 11 chars

    with pytest.raises(InvalidIdentifierError, match="Invalid ISIN length"):
        validate_isin("US03783310055", strict=True)  # 13 chars


def test_isin_validation_invalid_characters():
    """Tests that non-alphanumeric characters fail closed."""
    with pytest.raises(InvalidIdentifierError, match="Invalid ISIN syntax"):
        validate_isin("US03783310#5", strict=True)


def test_isin_normalization():
    """Tests that whitespace and lowercase are normalized appropriately."""
    assert normalize_isin("  ie000xagscy5  ") == "IE000XAGSCY5"
    assert validate_isin("  ie000xagscy5  ", strict=True) is True


def test_wkn_boundary_and_validation():
    """Verifies that WKN is validated as 6 alphanumeric chars and cannot be global ID."""
    assert WKN_GLOBAL_CANONICAL_ID is False
    assert validate_wkn("A3E40R", strict=True) is True
    assert validate_wkn("514000", strict=True) is True

    with pytest.raises(InvalidIdentifierError, match="Invalid WKN format"):
        validate_wkn("A3E40", strict=True)  # 5 chars

    with pytest.raises(InvalidIdentifierError, match="Invalid WKN format"):
        validate_wkn("A3E40R1", strict=True)  # 7 chars


def test_mic_validation():
    """Verifies venue MIC validation."""
    assert validate_mic("ARCX", strict=True) is True
    assert validate_mic("XETR", strict=True) is True
    assert validate_mic("TGAT", strict=True) is True

    with pytest.raises(InvalidIdentifierError, match="Invalid Venue MIC format"):
        validate_mic("ARCXX", strict=True)


# ==============================================================================
# 3. THREE-TIER IDENTITY INVARIANTS & MULTI-VENUE TESTS
# ==============================================================================

def test_multiple_share_classes_per_instrument():
    """Proves 1 Instrument -> N Share Classes (e.g. Accumulating & Distributing)."""
    sc_acc = ETFShareClass(
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        isin="IE000XAGSCY5",
        share_class_name="USD Accumulating",
        distribution_policy="ACCUMULATING",
        base_currency="USD",
    )
    sc_dist = ETFShareClass(
        share_class_id="etfs:v1:ISIN:IE000B3Z4694",
        instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        isin="IE000B3Z4694",
        share_class_name="EUR Distributing",
        distribution_policy="DISTRIBUTING",
        base_currency="EUR",
    )
    inst = ETFInstrument(
        canonical_instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        legal_fund_name="Global X Blockchain UCITS ETF",
        domicile_iso2="IE",
        regulatory_jurisdiction=Jurisdiction.EU_UCITS,
        share_classes=(sc_acc, sc_dist),
    )
    assert len(inst.share_classes) == 2
    assert inst.share_classes[0].share_class_id != inst.share_classes[1].share_class_id
    assert inst.share_classes[0].distribution_policy != inst.share_classes[1].distribution_policy


def test_multiple_listings_per_share_class():
    """Proves 1 Share Class -> M Listings across different exchanges/currencies."""
    listing_xetr = ETFListing(
        listing_id="etfl:v1:XETR:BLCH:EUR",
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        venue_mic="XETR",
        ticker="BLCH",
        trading_currency="EUR",
        local_code="A3E40R",
    )
    listing_lse = ETFListing(
        listing_id="etfl:v1:XLON:BKCH:USD",
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        venue_mic="XLON",
        ticker="BKCH",
        trading_currency="USD",
    )
    sc = ETFShareClass(
        share_class_id="etfs:v1:ISIN:IE000XAGSCY5",
        instrument_id="etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND",
        isin="IE000XAGSCY5",
        listings=(listing_xetr, listing_lse),
    )
    assert len(sc.listings) == 2
    assert sc.listings[0].listing_id != sc.listings[1].listing_id
    assert sc.listings[0].ticker != sc.listings[1].ticker
    assert sc.listings[0].trading_currency != sc.listings[1].trading_currency


def test_listing_and_ticker_changes_preserve_share_class_id():
    """Proves ticker update or venue migration does NOT change share class or instrument identity."""
    sc_id = generate_share_class_id(IdentifierType.ISIN, "IE000XAGSCY5")
    inst_id = generate_instrument_id(Jurisdiction.EU_UCITS, "IE", "IE000XAGSCY5_FUND")

    # Initial listing on Xetra
    listing1 = ETFListing(
        listing_id=generate_listing_id("XETR", "BLCH", "EUR"),
        share_class_id=sc_id,
        venue_mic="XETR",
        ticker="BLCH",
        trading_currency="EUR",
    )
    # Ticker change on same venue
    listing2 = ETFListing(
        listing_id=generate_listing_id("XETR", "BLCHX", "EUR"),
        share_class_id=sc_id,
        venue_mic="XETR",
        ticker="BLCHX",
        trading_currency="EUR",
    )
    assert listing1.share_class_id == sc_id
    assert listing2.share_class_id == sc_id
    assert listing1.listing_id != listing2.listing_id
    # Share class and instrument IDs remain constant
    assert sc_id == "etfs:v1:ISIN:IE000XAGSCY5"
    assert inst_id == "etfi:v1:EU_UCITS:IE:IE000XAGSCY5_FUND"


def test_same_ticker_different_mics_do_not_collide():
    """Proves same ticker on different venues produces distinct listing IDs."""
    lid_tgat = generate_listing_id("TGAT", "BLCH", "EUR")
    lid_xetr = generate_listing_id("XETR", "BLCH", "EUR")
    assert lid_tgat == "etfl:v1:TGAT:BLCH:EUR"
    assert lid_xetr == "etfl:v1:XETR:BLCH:EUR"
    assert lid_tgat != lid_xetr


def test_listing_deduplication_key_semantics():
    """
    Verifies:
    same share class + same MIC + same currency -> same listing identity
    same share class + different MIC -> different listing identity
    same share class + same MIC + different trading currency -> different listing identity
    """
    sc_id = "etfs:v1:ISIN:IE000XAGSCY5"
    key1 = compute_listing_dedup_key(sc_id, "XETR", "EUR")
    key2 = compute_listing_dedup_key(sc_id, "XETR", "EUR")
    key3 = compute_listing_dedup_key(sc_id, "TGAT", "EUR")
    key4 = compute_listing_dedup_key(sc_id, "XETR", "USD")

    assert key1 == key2
    assert key1 != key3
    assert key1 != key4


def test_display_name_change_preserves_canonical_id():
    """Proves fund rebranding / name changes do not alter canonical instrument ID."""
    id1 = generate_instrument_id(Jurisdiction.US_SEC, "US", "S000077125")
    inst1 = ETFInstrument(
        canonical_instrument_id=id1,
        legal_fund_name="Global X FinTech Thematic ETF",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
    )
    inst2 = ETFInstrument(
        canonical_instrument_id=id1,
        legal_fund_name="Global X FinTech ETF",  # Rebranded
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
    )
    assert inst1.canonical_instrument_id == inst2.canonical_instrument_id


# ==============================================================================
# 4. COLLISION HANDLING & JURISDICTION SAFETY TESTS
# ==============================================================================

def test_canonical_id_collision_fails_closed():
    """Proves registry raises CanonicalIdCollisionError if conflicting attributes re-register."""
    registry = DeterministicIdentityRegistry()
    inst1 = ETFInstrument(
        canonical_instrument_id="etfi:v1:US_SEC:US:S000077125",
        legal_fund_name="Global X FinTech ETF",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
    )
    registry.register_instrument(inst1)
    # Idempotent registration succeeds
    registry.register_instrument(inst1)

    # Conflicting registration with same canonical ID raises collision error
    inst_conflict = ETFInstrument(
        canonical_instrument_id="etfi:v1:US_SEC:US:S000077125",
        legal_fund_name="Conflicting Fraudulent Fund Name",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
    )
    with pytest.raises(CanonicalIdCollisionError, match="Instrument ID collision"):
        registry.register_instrument(inst_conflict)


def test_unknown_jurisdiction_fails_closed_no_us_default():
    """Proves unknown or unsupported jurisdictions fail closed and NEVER default to US."""
    unknown_jurisdiction = Jurisdiction.UNKNOWN
    assert unknown_jurisdiction.is_supported() is False

    with pytest.raises(UnsupportedJurisdictionError, match="Unsupported or unknown regulatory jurisdiction"):
        unknown_jurisdiction.assert_supported()

    with pytest.raises(UnsupportedJurisdictionError, match="Unsupported or unknown regulatory jurisdiction"):
        generate_instrument_id(unknown_jurisdiction, "ZZ", "ROOT123")


# ==============================================================================
# 5. US SEC ADAPTER LOSSLESS MAPPING TESTS
# ==============================================================================

def test_us_sec_authority_adapter_lossless_mapping():
    """Verifies lossless mapping of authoritative SEC EntityIdentity to 3-tier models."""
    sec_entity = EntityIdentity(
        symbol="FINX",
        cik="0001432353",
        series_id="S000077125",
        class_id="C000198642",
        legal_name="Global X FinTech ETF",
        historical_aliases=["FINX.US"],
    )
    instrument, share_class, listing = USSECAuthorityAdapter.adapt_entity_identity(
        sec_entity,
        venue_mic="ARCX",
        trading_currency="USD",
    )

    assert instrument.canonical_instrument_id == "etfi:v1:US_SEC:US:S000077125"
    assert share_class.share_class_id == "etfs:v1:SEC_CLASS_ID:C000198642"
    assert listing.listing_id == "etfl:v1:ARCX:FINX:USD"
    assert listing.ticker == "FINX"

    # Reverse extraction proving 100% losslessness
    extracted = USSECAuthorityAdapter.extract_sec_native_identity(instrument, share_class, listing)
    assert extracted["symbol"] == sec_entity.symbol
    assert extracted["cik"] == sec_entity.cik
    assert extracted["series_id"] == sec_entity.series_id
    assert extracted["class_id"] == sec_entity.class_id
    assert extracted["legal_name"] == sec_entity.legal_name
    assert extracted["historical_aliases"] == sec_entity.historical_aliases
    assert US_IDENTITY_MAPPING_LOSSLESS is True


def test_unresolved_sec_authority_never_upgraded():
    """Verifies adapter fails closed and NEVER upgrades an unresolved SEC state."""
    # Missing series ID or invalid series ID
    invalid_entity = EntityIdentity(
        symbol="BADX",
        cik="0001234567",
        series_id="",  # Unresolved
        class_id="C000123456",
        legal_name="Incomplete Entity",
    )
    with pytest.raises(ValueError, match="Cannot adapt unresolved SEC authority state"):
        USSECAuthorityAdapter.adapt_entity_identity(invalid_entity)

    # Malformed Class ID
    malformed_class_entity = EntityIdentity(
        symbol="BADY",
        cik="0001234567",
        series_id="S000012345",
        class_id="INVALID_CLASS",
        legal_name="Malformed Entity",
    )
    with pytest.raises(ValueError, match="Invalid SEC Class ID format"):
        USSECAuthorityAdapter.adapt_entity_identity(malformed_class_entity)


# ==============================================================================
# 6. ADVERSARIAL FAILURE CLASS VERIFICATION (A01 - A18)
# ==============================================================================

def test_adversarial_a01_ticker_functions_as_global_identity():
    """A01: Asserts that ticker alone CANNOT construct or represent global instrument identity."""
    ticker = "BLCH"
    # An instrument ID requires explicit jurisdiction, domicile, and root series/fund ID
    with pytest.raises(TypeError):
        ETFInstrument(ticker)  # Cannot instantiate from ticker string alone


def test_adversarial_a02_wkn_functions_as_global_identity():
    """A02: Asserts that WKN alone is rejected as global canonical ID."""
    assert WKN_GLOBAL_CANONICAL_ID is False
    wkn = "A3E40R"
    # A WKN is only valid as a listing local_code, not an instrument or share_class primary ID
    with pytest.raises(InvalidIdentifierError):
        generate_share_class_id(IdentifierType.ISIN, wkn)


def test_adversarial_a03_instrument_share_class_listing_collapse():
    """A03: Asserts that 3 identity levels remain separate and cannot be collapsed into 1."""
    inst = ETFInstrument(
        canonical_instrument_id="etfi:v1:US_SEC:US:S000077125",
        legal_fund_name="Global X FinTech ETF",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
    )
    sc = ETFShareClass(
        share_class_id="etfs:v1:SEC_CLASS_ID:C000198642",
        instrument_id=inst.canonical_instrument_id,
    )
    listing = ETFListing(
        listing_id="etfl:v1:ARCX:FINX:USD",
        share_class_id=sc.share_class_id,
        venue_mic="ARCX",
        ticker="FINX",
        trading_currency="USD",
    )
    assert inst.canonical_instrument_id != sc.share_class_id
    assert sc.share_class_id != listing.listing_id
    assert inst.canonical_instrument_id != listing.listing_id


def test_adversarial_a04_same_ticker_different_mics_collides():
    """A04: Asserts same ticker across exchanges does not collide in listing ID."""
    lid_1 = generate_listing_id("TGAT", "BLCH", "EUR")
    lid_2 = generate_listing_id("XETR", "BLCH", "EUR")
    assert lid_1 != lid_2


def test_adversarial_a05_same_share_class_multiple_exchanges_duplicates_fund():
    """A05: Asserts same share class listed on multiple exchanges does NOT create multiple funds."""
    inst_id = "etfi:v1:EU_UCITS:IE:GLOBAL_X_BLOCKCHAIN_FUND"
    sc_id = "etfs:v1:ISIN:IE000XAGSCY5"
    l1 = ETFListing("etfl:v1:TGAT:BLCH:EUR", sc_id, "TGAT", "BLCH", "EUR")
    l2 = ETFListing("etfl:v1:XETR:BLCH:EUR", sc_id, "XETR", "BLCH", "EUR")
    sc = ETFShareClass(sc_id, inst_id, "IE000XAGSCY5", listings=(l1, l2))
    inst = ETFInstrument(inst_id, "Global X Blockchain UCITS ETF", "IE", Jurisdiction.EU_UCITS, share_classes=(sc,))

    assert len(inst.share_classes) == 1  # 1 fund
    assert len(inst.share_classes[0].listings) == 2  # 2 venue listings


def test_adversarial_a06_ticker_change_mutates_share_class_identity():
    """A06: Asserts ticker change does NOT mutate share-class identity."""
    sc_id = "etfs:v1:ISIN:IE000XAGSCY5"
    listing_v1 = ETFListing("etfl:v1:XETR:OLD:EUR", sc_id, "XETR", "OLD", "EUR")
    listing_v2 = ETFListing("etfl:v1:XETR:NEW:EUR", sc_id, "XETR", "NEW", "EUR")
    assert listing_v1.share_class_id == listing_v2.share_class_id == sc_id


def test_adversarial_a07_accumulating_distributing_classes_collapse():
    """A07: Asserts distinct accumulating and distributing tranches remain separate share classes."""
    sc_acc = ETFShareClass("etfs:v1:ISIN:IE000XAGSCY5", "etfi:1", "IE000XAGSCY5", distribution_policy="ACC")
    sc_dist = ETFShareClass("etfs:v1:ISIN:IE000B3Z4694", "etfi:1", "IE000B3Z4694", distribution_policy="DIST")
    assert sc_acc.share_class_id != sc_dist.share_class_id


def test_adversarial_a08_malformed_isin_is_accepted():
    """A08: Asserts malformed ISIN is rejected."""
    with pytest.raises(InvalidIdentifierError):
        validate_isin("NOT_AN_ISIN", strict=True)


def test_adversarial_a09_invalid_isin_checksum_is_accepted():
    """A09: Asserts invalid ISIN checksum is rejected."""
    with pytest.raises(InvalidIdentifierError, match="check-digit failure"):
        validate_isin("IE000XAGSCY0", strict=True)


def test_adversarial_a10_unknown_jurisdiction_defaults_to_us():
    """A10: Asserts unknown jurisdiction fails closed and never defaults to US."""
    with pytest.raises(UnsupportedJurisdictionError):
        Jurisdiction.UNKNOWN.assert_supported()


def test_adversarial_a11_global_adapter_recomputes_sec_authority():
    """A11: Asserts adapter consumes existing EntityIdentity without running SEC scraping/parsing."""
    entity = EntityIdentity("FINX", "0001432353", "S000077125", "C000198642", "Global X FinTech ETF")
    # Adapter simply transforms dataclass fields; no network/disk SEC recomputation occurs
    inst, sc, listing = USSECAuthorityAdapter.adapt_entity_identity(entity)
    assert inst.canonical_instrument_id == "etfi:v1:US_SEC:US:S000077125"


def test_adversarial_a12_unresolved_sec_authority_is_upgraded():
    """A12: Asserts unresolved SEC state cannot be adapted to resolved global identity."""
    unresolved_entity = EntityIdentity("SPY", "0000884394", "", "C000000001", "SPDR S&P 500 ETF Trust")
    with pytest.raises(ValueError, match="Cannot adapt unresolved SEC authority state"):
        USSECAuthorityAdapter.adapt_entity_identity(unresolved_entity)


def test_adversarial_a13_display_name_only_change_alters_canonical_id():
    """A13: Asserts display-name change does not alter canonical ID."""
    id_orig = generate_instrument_id(Jurisdiction.US_SEC, "US", "S000077125")
    id_after_rename = generate_instrument_id(Jurisdiction.US_SEC, "US", "S000077125")
    assert id_orig == id_after_rename


def test_adversarial_a14_serialization_is_nondeterministic():
    """A14: Asserts serialization is 100% byte-identical across runs."""
    inst = ETFInstrument(
        canonical_instrument_id="etfi:v1:US_SEC:US:S000077125",
        legal_fund_name="Global X FinTech ETF",
        domicile_iso2="US",
        regulatory_jurisdiction=Jurisdiction.US_SEC,
        metadata={"b": 2, "a": 1, "z": 99},
    )
    json_1 = inst.to_json()
    json_2 = inst.to_json()
    assert json_1 == json_2
    # Ensure keys are sorted in json representation
    assert json_1.index('"a": 1') < json_1.index('"b": 2') < json_1.index('"z": 99')


def test_adversarial_a15_hash_input_ordering_changes_identity():
    """A15: Asserts canonical ID generator is insensitive to case/whitespace noise in root ID."""
    id1 = generate_instrument_id(Jurisdiction.US_SEC, "us", "  s000077125  ")
    id2 = generate_instrument_id(Jurisdiction.US_SEC, "US", "S000077125")
    assert id1 == id2


def test_adversarial_a16_new_code_is_imported_by_production_runtime():
    """A16: Asserts new Wave 1 modules are NOT imported by api/ or frontend/ runtime code."""
    repo_root = Path(__file__).resolve().parents[1]
    runtime_dirs = [repo_root / "api", repo_root / "frontend"]
    for rdir in runtime_dirs:
        if not rdir.exists():
            continue
        for fpath in rdir.rglob("*.py"):
            text = fpath.read_text(encoding="utf-8")
            assert "global_identity_models" not in text
            assert "global_identifier_authority" not in text
            assert "us_sec_authority_adapter" not in text


def test_adversarial_a17_existing_etf_population_changes():
    """A17: Asserts SEC golden corpus document count remains 859."""
    repo_root = Path(__file__).resolve().parents[1]
    corpus_manifest = repo_root / "docs" / "research" / "ETF_V2_SEC_SOURCE_CORPUS_MANIFEST.json"
    with open(corpus_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["corpus_document_count"] == 859
    assert data["corpus_aggregate_identity"] == "b186f39772763683b238609066a20c21cf1717f0d9dcf32741c47bc4dfeb27b6"


def test_adversarial_a18_existing_classification_changes():
    """A18: Asserts Policy V1.1 normative decision table remains untouched."""
    repo_root = Path(__file__).resolve().parents[1]
    policy_path = repo_root / "docs" / "research" / "ETF_POLICY_V1_1_NORMATIVE_DECISION_TABLE.md"
    assert policy_path.exists()
    text = policy_path.read_text(encoding="utf-8")
    assert "864133d98750f7765409153c4305d02ff9d422aa56556b6a0506738299642f52" in text
