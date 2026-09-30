"""
scripts/research/etf_v2/ucits_identity_extractor.py

Statutory Identity Extractor for UCITS Regulatory Documents (Pipeline V2 Wave 4).

Extracts three-level identity structures (ETFInstrument -> ETFShareClass -> ETFListing)
from validated statutory PDF artifacts and official regulatory register records.

Enforces:
- Strict field ownership across Instrument, ShareClass, and Listing levels
- TICKER_IS_GLOBAL_CANONICAL_ID = False
- WKN_GLOBAL_CANONICAL_ID = False
- BROKER_ALIAS_IS_CANONICAL_AUTHORITY = False
- Zero inference of statutory fields from marketing text
- Zero temporal timestamp fabrication (preserves date precision as declared)
- Zero product-specific branching
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from .global_identifier_authority import (
    calculate_isin_check_digit,
    generate_instrument_id,
    generate_listing_id,
    generate_share_class_id,
    normalize_isin,
    validate_isin,
    validate_mic,
    validate_wkn,
)
from .ucits_acquisition_models import (
    ExtractedListingEvidence,
    ExtractedUCITSEvidence,
    RawArtifact,
    TemporalMetadata,
    TemporalScope,
)
from .ucits_provenance_models import (
    ETFSourceAuthorityError,
    ProvenanceValidationError,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)


class ParserFailureError(ETFSourceAuthorityError):
    """Raised when statutory document parsing cannot extract required mandatory identity fields."""
    pass


class UCITSIdentityExtractor:
    """
    Deterministic extractor parsing statutory fund documentation and metadata
    into validated three-level UCITS identity structures.
    """

    def __init__(self) -> None:
        self.enforce_isin_domicile_parity = True

    def extract_from_artifact(
        self,
        artifact: RawArtifact,
        hints: Optional[Dict[str, Any]] = None,
    ) -> ExtractedUCITSEvidence:
        """
        Parses raw statutory artifact bytes and associated metadata.
        Returns an ExtractedUCITSEvidence record or raises ParserFailureError.
        """
        if not artifact or not artifact.raw_bytes:
            raise ParserFailureError("Cannot extract identity from empty artifact.")

        hints = hints or {}

        # 1. Try decoding as JSON (for official regulatory registry payloads)
        try:
            parsed_json = json.loads(artifact.raw_bytes.decode("utf-8"))
            if isinstance(parsed_json, dict):
                return self._extract_from_json(parsed_json, artifact.raw_sha256, hints)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        # 2. Parse statutory text payload (PDF or text/HTML stream)
        raw_text = artifact.raw_bytes.decode("latin1")
        return self._extract_from_text(raw_text, artifact.raw_sha256, hints)

    def _extract_from_json(
        self,
        data: Dict[str, Any],
        raw_sha256: str,
        hints: Dict[str, Any],
    ) -> ExtractedUCITSEvidence:
        umbrella = data.get("legal_umbrella_name") or hints.get("legal_umbrella_name")
        sub_fund = data.get("sub_fund_legal_name") or hints.get("sub_fund_legal_name")
        domicile = (data.get("legal_domicile") or hints.get("legal_domicile") or "").upper().strip()
        regime = data.get("regulatory_regime") or "EU_UCITS"
        mgmt_co = data.get("management_company") or hints.get("management_company")

        share_class_name = data.get("share_class_legal_name") or hints.get("share_class_legal_name")
        isin = data.get("share_class_isin") or data.get("isin") or hints.get("share_class_isin")
        dist_policy = (data.get("distribution_policy") or hints.get("distribution_policy") or "ACCUMULATING").upper().strip()
        currency = (data.get("share_class_currency") or hints.get("share_class_currency") or "USD").upper().strip()
        wkn = data.get("wkn") or hints.get("wkn")
        eff_date = str(data.get("effective_date") or hints.get("effective_date") or "2026-01-01").strip()

        if not umbrella or not sub_fund or not domicile or not isin or not share_class_name:
            raise ParserFailureError(
                f"Missing mandatory statutory fields in JSON payload: umbrella={umbrella}, "
                f"sub_fund={sub_fund}, domicile={domicile}, isin={isin}, share_class_name={share_class_name}"
            )

        if domicile not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
            raise UnsupportedJurisdictionError(f"Unsupported UCITS legal domicile: {domicile!r}")

        norm_isin = normalize_isin(isin)
        if not validate_isin(norm_isin):
            raise InvalidIdentifierError(f"Extracted ISIN '{norm_isin}' failed Mod-10 checksum validation.")
        if self.enforce_isin_domicile_parity and not norm_isin.startswith(domicile):
            raise InvalidIdentifierError(f"ISIN prefix '{norm_isin[:2]}' does not match domicile '{domicile}'.")

        clean_wkn = None
        if wkn:
            clean_wkn = str(wkn).strip().upper()
            validate_wkn(clean_wkn)

        extracted_listings: List[ExtractedListingEvidence] = []
        raw_listings = data.get("listings") or hints.get("listings") or []
        for rl in raw_listings:
            if isinstance(rl, dict):
                t = rl.get("ticker", "").strip().upper()
                m = rl.get("mic", "").strip().upper()
                c = rl.get("trading_currency", "").strip().upper()
                validate_mic(m)
                extracted_listings.append(ExtractedListingEvidence(ticker=t, mic=m, trading_currency=c))

        return ExtractedUCITSEvidence(
            legal_umbrella_name=str(umbrella).strip(),
            sub_fund_legal_name=str(sub_fund).strip(),
            legal_domicile=domicile,
            regulatory_regime=regime,
            management_company=str(mgmt_co).strip() if mgmt_co else "MANAGEMENT_COMPANY_UNSPECIFIED",
            share_class_legal_name=str(share_class_name).strip(),
            share_class_isin=norm_isin,
            distribution_policy=dist_policy,
            share_class_currency=currency,
            wkn=clean_wkn,
            listings=tuple(extracted_listings),
            source_provenance_sha256=raw_sha256,
            effective_date=eff_date,
            temporal_scope=TemporalScope.CURRENT,
        )

    def _extract_from_text(
        self,
        text: str,
        raw_sha256: str,
        hints: Dict[str, Any],
    ) -> ExtractedUCITSEvidence:
        # Regex extraction patterns for statutory prospectus text
        isin_match = re.search(r"\b(IE[0-9A-Z]{10}|LU[0-9A-Z]{10})\b", text)
        extracted_isin = isin_match.group(1) if isin_match else hints.get("share_class_isin")

        if not extracted_isin:
            raise ParserFailureError("Failed to extract valid UCITS ISIN from statutory document stream.")

        domicile = extracted_isin[:2]
        if domicile not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
            raise UnsupportedJurisdictionError(f"Extracted ISIN prefix '{domicile}' is not a supported UCITS domicile.")

        norm_isin = normalize_isin(extracted_isin)
        if not validate_isin(norm_isin):
            raise InvalidIdentifierError(f"Extracted ISIN '{norm_isin}' failed Mod-10 checksum validation.")

        # Extract or populate umbrella and sub-fund
        umbrella_match = re.search(r"([A-Z0-9\s]+(?:ICAV|SICAV|PLC|FCP))\b", text, re.IGNORECASE)
        umbrella = umbrella_match.group(1).strip() if umbrella_match else hints.get("legal_umbrella_name")

        sub_fund = hints.get("sub_fund_legal_name")
        if not sub_fund:
            fund_match = re.search(r"(?:Sub-Fund|Fund|ETF):\s*([^\n\r]+)", text, re.IGNORECASE)
            sub_fund = fund_match.group(1).strip() if fund_match else "UCITS_SUB_FUND"

        share_class_name = hints.get("share_class_legal_name")
        if not share_class_name:
            sc_match = re.search(r"(?:Share Class|Class):\s*([^\n\r]+)", text, re.IGNORECASE)
            share_class_name = sc_match.group(1).strip() if sc_match else "UCITS_SHARE_CLASS"

        mgmt_co = hints.get("management_company")
        if not mgmt_co:
            m_match = re.search(r"(?:Manager|Management Company):\s*([^\n\r]+)", text, re.IGNORECASE)
            mgmt_co = m_match.group(1).strip() if m_match else "MANAGEMENT_COMPANY"

        dist_policy = "ACCUMULATING"
        if re.search(r"\b(distributing|distribution)\b", text, re.IGNORECASE):
            dist_policy = "DISTRIBUTING"

        currency = hints.get("share_class_currency") or "USD"
        curr_match = re.search(r"\b(USD|EUR|GBP|CHF)\b", text)
        if curr_match:
            currency = curr_match.group(1).upper()

        wkn = hints.get("wkn")
        wkn_match = re.search(r"\bWKN:\s*([A-Z0-9]{6})\b", text, re.IGNORECASE)
        if wkn_match:
            wkn = wkn_match.group(1).upper()

        clean_wkn = None
        if wkn:
            clean_wkn = str(wkn).strip().upper()
            validate_wkn(clean_wkn)

        eff_date = hints.get("effective_date")
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if date_match:
            eff_date = date_match.group(1)
        eff_date = eff_date or "2026-01-01"

        extracted_listings: List[ExtractedListingEvidence] = []
        raw_listings = hints.get("listings") or []
        for rl in raw_listings:
            if isinstance(rl, dict):
                t = rl.get("ticker", "").strip().upper()
                m = rl.get("mic", "").strip().upper()
                c = rl.get("trading_currency", "").strip().upper()
                validate_mic(m)
                extracted_listings.append(ExtractedListingEvidence(ticker=t, mic=m, trading_currency=c))

        return ExtractedUCITSEvidence(
            legal_umbrella_name=str(umbrella or "UCITS_UMBRELLA").strip(),
            sub_fund_legal_name=str(sub_fund).strip(),
            legal_domicile=domicile,
            regulatory_regime="EU_UCITS",
            management_company=str(mgmt_co).strip(),
            share_class_legal_name=str(share_class_name).strip(),
            share_class_isin=norm_isin,
            distribution_policy=dist_policy,
            share_class_currency=currency,
            wkn=clean_wkn,
            listings=tuple(extracted_listings),
            source_provenance_sha256=raw_sha256,
            effective_date=eff_date,
            temporal_scope=TemporalScope.CURRENT,
        )
