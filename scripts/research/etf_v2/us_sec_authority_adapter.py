"""
scripts/research/etf_v2/us_sec_authority_adapter.py

US SEC Authority Adapter for Pipeline V2.
Adapts established SEC identity outputs (EntityIdentity and PopulationRecord)
losslessly into the 3-tier global identity model (ETFInstrument, ETFShareClass, ETFListing).

Enforces strict boundaries:
- global identity representation != new SEC authority
- adapter mapping != classification evidence
- ticker != SEC identity authority
- unresolved SEC authority states are NEVER upgraded to resolved global identities
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .models import EntityIdentity, PopulationRecord
from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    Jurisdiction,
)
from .global_identifier_authority import (
    generate_instrument_id,
    generate_listing_id,
    generate_share_class_id,
)

# Explicit contract invariant: US SEC mapping is 100% lossless for valid canonical inputs
US_IDENTITY_MAPPING_LOSSLESS: bool = True


class USSECAuthorityAdapter:
    """
    Consumes existing authoritative SEC identity records and maps them into
    the global 3-tier identity domain foundation without recomputing SEC authority.
    """

    @staticmethod
    def adapt_entity_identity(
        entity: EntityIdentity,
        venue_mic: str = "ARCX",
        trading_currency: str = "USD",
    ) -> Tuple[ETFInstrument, ETFShareClass, ETFListing]:
        """
        Losslessly maps an EntityIdentity into (ETFInstrument, ETFShareClass, ETFListing).
        Fails closed if the entity fails validation or has unresolved CIK/Series/Class IDs.
        NEVER upgrades an unresolved SEC state.
        """
        if not entity or not isinstance(entity, EntityIdentity):
            raise ValueError(f"Expected EntityIdentity instance, got: {type(entity)}")

        # Enforce fail-closed boundary on incomplete or unresolved SEC authority
        if not entity.validate():
            raise ValueError(
                f"Cannot adapt unresolved SEC authority state for symbol '{entity.symbol}'. "
                "CIK, Series ID, and Class ID must all be populated and valid."
            )

        # Validate SEC identifier formats
        clean_sid = entity.series_id.strip()
        clean_cid = entity.class_id.strip()
        clean_cik = str(entity.cik).strip().zfill(10)

        if not re.match(r"^S\d{9}$", clean_sid):
            raise ValueError(f"Invalid SEC Series ID format: {clean_sid}")
        if not re.match(r"^C\d{9}$", clean_cid):
            raise ValueError(f"Invalid SEC Class ID format: {clean_cid}")
        if not re.match(r"^\d{10}$", clean_cik):
            raise ValueError(f"Invalid SEC CIK format: {clean_cik}")

        # 1. Instrument (Fund level)
        instrument_id = generate_instrument_id(
            jurisdiction=Jurisdiction.US_SEC,
            domicile_iso2="US",
            root_id=clean_sid,
        )

        # 2. Share Class (Tranche level)
        share_class_id = generate_share_class_id(
            scheme=IdentifierType.SEC_CLASS_ID,
            identifier=clean_cid,
        )

        # 3. Listing (Venue level)
        listing_id = generate_listing_id(
            venue_mic=venue_mic,
            ticker=entity.symbol,
            trading_currency=trading_currency,
        )

        listing = ETFListing(
            listing_id=listing_id,
            share_class_id=share_class_id,
            venue_mic=venue_mic.strip().upper(),
            ticker=entity.symbol.strip().upper(),
            trading_currency=trading_currency.strip().upper(),
            venue_name="NYSE Arca" if venue_mic.upper() == "ARCX" else venue_mic.upper(),
            broker_aliases=tuple(entity.historical_aliases),
            identity_status=IdentityStatus.RESOLVED,
            metadata={"sec_native_symbol": entity.symbol},
        )

        share_class = ETFShareClass(
            share_class_id=share_class_id,
            instrument_id=instrument_id,
            isin=None,
            share_class_name=entity.symbol.strip().upper(),
            distribution_policy="",
            base_currency=trading_currency.strip().upper(),
            sec_class_id=clean_cid,
            identity_status=IdentityStatus.RESOLVED,
            listings=(listing,),
            metadata={"cik": clean_cik, "series_id": clean_sid, "class_id": clean_cid},
        )

        instrument = ETFInstrument(
            canonical_instrument_id=instrument_id,
            legal_fund_name=entity.legal_name.strip(),
            domicile_iso2="US",
            regulatory_jurisdiction=Jurisdiction.US_SEC,
            fund_structure="1940_ACT_OPEN_END_ETF",
            identity_status=IdentityStatus.RESOLVED,
            share_classes=(share_class,),
            metadata={
                "sec_cik": clean_cik,
                "sec_series_id": clean_sid,
                "sec_primary_class_id": clean_cid,
            },
        )

        return instrument, share_class, listing

    @classmethod
    def adapt_population_record(
        cls,
        record: PopulationRecord,
        venue_mic: str = "ARCX",
        trading_currency: str = "USD",
    ) -> Tuple[ETFInstrument, ETFShareClass, ETFListing]:
        """
        Losslessly maps an authoritative PopulationRecord into the 3-tier global identity model.
        """
        if not record or not isinstance(record, PopulationRecord):
            raise ValueError(f"Expected PopulationRecord instance, got: {type(record)}")

        entity = EntityIdentity(
            symbol=record.symbol,
            cik=record.cik,
            series_id=record.series_id,
            class_id=record.class_id,
            legal_name=record.legal_name,
            historical_aliases=[],
        )
        return cls.adapt_entity_identity(entity, venue_mic=venue_mic, trading_currency=trading_currency)

    @staticmethod
    def extract_sec_native_identity(
        instrument: ETFInstrument,
        share_class: ETFShareClass,
        listing: ETFListing,
    ) -> Dict[str, Any]:
        """
        Extracts native SEC identifiers back from the 3-tier global identity representation,
        proving that US_IDENTITY_MAPPING_LOSSLESS holds.
        """
        cik = instrument.metadata.get("sec_cik") or share_class.metadata.get("cik")
        series_id = instrument.metadata.get("sec_series_id") or share_class.metadata.get("series_id")
        class_id = share_class.sec_class_id or share_class.metadata.get("class_id")
        symbol = listing.ticker
        legal_name = instrument.legal_fund_name

        if not (cik and series_id and class_id and symbol and legal_name):
            raise ValueError("Incomplete SEC identity fields in global identity models.")

        return {
            "symbol": symbol,
            "cik": str(cik).zfill(10),
            "series_id": series_id,
            "class_id": class_id,
            "legal_name": legal_name,
            "historical_aliases": list(listing.broker_aliases),
        }
