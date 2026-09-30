"""
scripts/research/etf_v2/global_identifier_authority.py

Pure identifier-domain authority module for Pipeline V2.
Enforces versioned deterministic canonical internal IDs:
    etfi:v1:... (Instrument)
    etfs:v1:... (Share Class)
    etfl:v1:... (Listing)
Implements ISO 6166 Modulus-10 Double-Add-Double ISIN validation,
WKN jurisdictional alias boundaries (WKN_GLOBAL_CANONICAL_ID = False),
venue MIC normalization, and collision-safe deterministic registration.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Optional, Tuple

from .identity_authority import IdentityAuthority
from .global_identity_models import (
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

# Canonical Identity Version
INTERNAL_ID_VERSION: str = "v1"

# Explicit Architectural Boundary: WKN is NEVER global canonical identity
WKN_GLOBAL_CANONICAL_ID: bool = False

# Authoritative ISO 3166-1 alpha-2 country codes plus international identifiers (XS, EU)
ISO_3166_1_ALPHA_2_CODES: FrozenSet[str] = frozenset({
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS",
    "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE",
    "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM",
    "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC",
    "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG",
    "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO",
    "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW",
    # Supranational / International securities
    "XS", "EU",
})


def normalize_isin(raw_isin: str) -> str:
    """Canonical text normalization for ISIN strings."""
    if not raw_isin or not isinstance(raw_isin, str):
        raise InvalidIdentifierError(f"ISIN must be a non-empty string, got: {raw_isin!r}")
    return IdentityAuthority.normalize_for_matching(raw_isin).strip().upper()


def calculate_isin_check_digit(payload_11: str) -> int:
    """
    Computes ISO 6166 check digit using Modulus 10 Double-Add-Double (Luhn variant).
    Letters A-Z are converted to 10-35, and digits 0-9 remain unchanged.
    Weights alternate 2, 1, 2, 1... from rightmost digit of the payload.
    """
    digits = []
    for char in payload_11:
        if char.isdigit():
            digits.append(int(char))
        elif "A" <= char <= "Z":
            val = ord(char) - ord("A") + 10
            digits.append(val // 10)
            digits.append(val % 10)
        else:
            raise InvalidIdentifierError(f"Invalid character in ISIN payload: {char!r}")

    # Weights alternate 2, 1, 2, 1... starting with 2 on the rightmost payload digit
    sum_digits = 0
    weight = 2
    for digit in reversed(digits):
        prod = digit * weight
        sum_digits += (prod // 10) + (prod % 10)
        weight = 1 if weight == 2 else 2

    return (10 - (sum_digits % 10)) % 10


def validate_isin(isin: str, strict: bool = True) -> bool:
    """
    Full validation of an ISIN string under ISO 6166:
    1. Normalization (whitespace/case).
    2. Exact 12-character length.
    3. Syntax: 2 letters, 9 alphanumeric, 1 decimal digit.
    4. Country prefix exists in ISO 3166-1 / international table.
    5. Modulus 10 Double-Add-Double check digit verification.
    """
    try:
        clean = normalize_isin(isin)
    except InvalidIdentifierError:
        if strict:
            raise
        return False

    if len(clean) != 12:
        if strict:
            raise InvalidIdentifierError(f"Invalid ISIN length (expected 12 characters): {len(clean)} in {isin!r}")
        return False

    if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", clean):
        if strict:
            raise InvalidIdentifierError(f"Invalid ISIN syntax: {clean}")
        return False

    country_prefix = clean[:2]
    if country_prefix not in ISO_3166_1_ALPHA_2_CODES:
        if strict:
            raise InvalidIdentifierError(f"Invalid or unrecognized ISIN country prefix: {country_prefix}")
        return False

    expected_check = calculate_isin_check_digit(clean[:11])
    actual_check = int(clean[11])
    if actual_check != expected_check:
        if strict:
            raise InvalidIdentifierError(
                f"ISIN check-digit failure for {clean}: expected {expected_check}, found {actual_check}"
            )
        return False

    return True


def validate_wkn(wkn: str, strict: bool = True) -> bool:
    """
    Validates German Wertpapierkennnummer (WKN).
    Must be exactly 6 alphanumeric characters.
    Enforces WKN_GLOBAL_CANONICAL_ID = False.
    """
    if not wkn or not isinstance(wkn, str):
        if strict:
            raise InvalidIdentifierError(f"WKN must be a non-empty string, got: {wkn!r}")
        return False
    clean = IdentityAuthority.normalize_for_matching(wkn).strip().upper()
    if not re.match(r"^[A-Z0-9]{6}$", clean):
        if strict:
            raise InvalidIdentifierError(f"Invalid WKN format (expected 6 alphanumeric chars): {wkn!r}")
        return False
    return True


def validate_mic(mic: str, strict: bool = True) -> bool:
    """
    Validates ISO 10383 Market Identifier Code (MIC).
    Must be exactly 4 alphanumeric characters.
    """
    if not mic or not isinstance(mic, str):
        if strict:
            raise InvalidIdentifierError(f"Venue MIC must be a non-empty string, got: {mic!r}")
        return False
    clean = mic.strip().upper()
    if not re.match(r"^[A-Z0-9]{4}$", clean):
        if strict:
            raise InvalidIdentifierError(f"Invalid Venue MIC format (expected 4 alphanumeric chars): {mic!r}")
        return False
    return True


def generate_instrument_id(jurisdiction: Jurisdiction, domicile_iso2: str, root_id: str) -> str:
    """
    Generates frozen canonical internal instrument ID (v1).
    Format: etfi:v1:{jurisdiction}:{domicile}:{normalized_root_id}
    Fails closed if jurisdiction is unsupported.
    """
    jurisdiction.assert_supported()
    dom = domicile_iso2.strip().upper()
    if dom not in ISO_3166_1_ALPHA_2_CODES:
        raise InvalidIdentifierError(f"Invalid domicile ISO-2 country code: {domicile_iso2!r}")

    clean_root = IdentityAuthority.normalize_for_matching(root_id).strip().upper()
    clean_root = re.sub(r"[^A-Z0-9_-]", "_", clean_root)
    if not clean_root:
        raise InvalidIdentifierError(f"Root identifier cannot be empty for instrument ID: {root_id!r}")

    return f"etfi:{INTERNAL_ID_VERSION}:{jurisdiction.value}:{dom}:{clean_root}"


def generate_share_class_id(scheme: IdentifierType, identifier: str) -> str:
    """
    Generates frozen canonical internal share class ID (v1).
    Format: etfs:v1:{scheme}:{normalized_identifier}
    If scheme is ISIN, validates ISO 6166 checksum before generation.
    """
    if scheme == IdentifierType.ISIN:
        clean_id = normalize_isin(identifier)
        validate_isin(clean_id, strict=True)
    elif scheme in (IdentifierType.SEC_CLASS_ID, IdentifierType.INTERNAL_SHARE_CLASS_ID):
        clean_id = identifier.strip().upper()
        if not re.match(r"^C\d{9}$", clean_id) and scheme == IdentifierType.SEC_CLASS_ID:
            raise InvalidIdentifierError(f"Invalid SEC Class ID format: {identifier!r}")
    else:
        clean_id = IdentityAuthority.normalize_for_matching(identifier).strip().upper()
        clean_id = re.sub(r"[^A-Z0-9_-]", "_", clean_id)

    if not clean_id:
        raise InvalidIdentifierError(f"Share class identifier cannot be empty: {identifier!r}")

    return f"etfs:{INTERNAL_ID_VERSION}:{scheme.value}:{clean_id}"


def generate_listing_id(venue_mic: str, ticker: str, trading_currency: str) -> str:
    """
    Generates frozen canonical internal listing ID (v1).
    Format: etfl:v1:{venue_mic}:{ticker}:{trading_currency}
    """
    validate_mic(venue_mic, strict=True)
    clean_mic = venue_mic.strip().upper()

    clean_ticker = IdentityAuthority.normalize_for_matching(ticker).strip().upper()
    if not clean_ticker or not re.match(r"^[A-Z0-9\.\-_]+$", clean_ticker):
        raise InvalidIdentifierError(f"Invalid ticker format for listing ID: {ticker!r}")

    clean_ccy = trading_currency.strip().upper()
    if not re.match(r"^[A-Z]{3}$", clean_ccy):
        raise InvalidIdentifierError(f"Invalid trading currency ISO-4217 code: {trading_currency!r}")

    return f"etfl:{INTERNAL_ID_VERSION}:{clean_mic}:{clean_ticker}:{clean_ccy}"


def compute_listing_dedup_key(share_class_id: str, venue_mic: str, trading_currency: str) -> Tuple[str, str, str]:
    """
    Computes LISTING_DEDUPLICATION_KEY = (share_class_id, venue_mic, trading_currency).
    """
    validate_mic(venue_mic, strict=True)
    clean_ccy = trading_currency.strip().upper()
    if not re.match(r"^[A-Z]{3}$", clean_ccy):
        raise InvalidIdentifierError(f"Invalid trading currency code: {trading_currency!r}")
    return (share_class_id.strip(), venue_mic.strip().upper(), clean_ccy)


class DeterministicIdentityRegistry:
    """
    In-memory, collision-safe deterministic identity registry primitive.
    Pure domain / test primitive. NOT a live search service.
    Enforces fail-closed behavior on canonical-ID collisions with conflicting attributes.
    """

    def __init__(self) -> None:
        self._instruments: Dict[str, ETFInstrument] = {}
        self._share_classes: Dict[str, ETFShareClass] = {}
        self._listings: Dict[str, ETFListing] = {}
        self._listing_dedup_keys: Dict[Tuple[str, str, str], str] = {}

    def register_instrument(self, instrument: ETFInstrument) -> None:
        """Registers an ETFInstrument, failing closed if a collision with conflicting fields occurs."""
        cid = instrument.canonical_instrument_id
        if cid in self._instruments:
            existing = self._instruments[cid]
            if existing.to_dict() != instrument.to_dict():
                raise CanonicalIdCollisionError(
                    f"Canonical Instrument ID collision detected for {cid} with conflicting attributes.\n"
                    f"Existing: {existing.to_dict()}\nNew: {instrument.to_dict()}"
                )
            return  # Idempotent pass
        self._instruments[cid] = instrument

    def register_share_class(self, share_class: ETFShareClass) -> None:
        """Registers an ETFShareClass, failing closed if a collision occurs."""
        scid = share_class.share_class_id
        if scid in self._share_classes:
            existing = self._share_classes[scid]
            if existing.to_dict() != share_class.to_dict():
                raise CanonicalIdCollisionError(
                    f"Canonical Share Class ID collision detected for {scid} with conflicting attributes.\n"
                    f"Existing: {existing.to_dict()}\nNew: {share_class.to_dict()}"
                )
            return
        self._share_classes[scid] = share_class

    def register_listing(self, listing: ETFListing) -> None:
        """
        Registers an ETFListing, failing closed on:
        1. Canonical listing ID collision with conflicting attributes.
        2. Listing deduplication key collision (share_class_id, venue_mic, trading_currency) pointing to different listing_id.
        """
        lid = listing.listing_id
        dedup_key = listing.deduplication_key()

        if dedup_key in self._listing_dedup_keys:
            existing_lid = self._listing_dedup_keys[dedup_key]
            if existing_lid != lid:
                raise CanonicalIdCollisionError(
                    f"Listing deduplication key collision: key {dedup_key} already registered to {existing_lid}, "
                    f"conflicts with new listing {lid}"
                )

        if lid in self._listings:
            existing = self._listings[lid]
            if existing.to_dict() != listing.to_dict():
                raise CanonicalIdCollisionError(
                    f"Canonical Listing ID collision detected for {lid} with conflicting attributes.\n"
                    f"Existing: {existing.to_dict()}\nNew: {listing.to_dict()}"
                )
            return

        self._listings[lid] = listing
        self._listing_dedup_keys[dedup_key] = lid

    def get_instrument(self, canonical_id: str) -> Optional[ETFInstrument]:
        return self._instruments.get(canonical_id)

    def get_share_class(self, share_class_id: str) -> Optional[ETFShareClass]:
        return self._share_classes.get(share_class_id)

    def get_listing(self, listing_id: str) -> Optional[ETFListing]:
        return self._listings.get(listing_id)
