"""analyst_dashboard/security_master/applicability.py

Canonical Instrument Evidence Applicability Router (Synthesis E Wave 3).
Enforces the formal mathematical contract:
    EVIDENCE_REQUIREMENT = f(canonical_instrument_class, regulatory_structure)

Strictly satisfies:
    NOT_APPLICABLE != MISSING != UNVERIFIED != FAILED
    NO_INSTRUMENT_MAY_BE_FAILED_FOR_EVIDENCE_CLASSIFIED_NOT_APPLICABLE
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Set
from .models import SecurityType, AssetClass, CanonicalInstrument


class InstrumentEvidenceContract:
    """Defines the regulatory and evidence profile for a specific instrument class."""

    def __init__(
        self,
        security_type: SecurityType,
        required_evidence: Set[str],
        not_applicable_evidence: Set[str],
        incomplete_reason: str,
        profile_label: str,
        profile_description: str,
    ):
        self.security_type = security_type
        self.required_evidence = required_evidence
        self.not_applicable_evidence = not_applicable_evidence
        self.incomplete_reason = incomplete_reason
        self.profile_label = profile_label
        self.profile_description = profile_description


# Authoritative regulatory evidence matrix
INSTRUMENT_EVIDENCE_REGISTRY: Dict[SecurityType, InstrumentEvidenceContract] = {
    SecurityType.COMMON_STOCK: InstrumentEvidenceContract(
        security_type=SecurityType.COMMON_STOCK,
        required_evidence={"CORPORATE_FINANCIALS_10K_10Q"},
        not_applicable_evidence=set(),
        incomplete_reason="Audited SEC EDGAR 10-K/10-Q financial filings are unverified.",
        profile_label="Corporate Operating Company",
        profile_description="Evaluated via audited SEC EDGAR 10-K/10-Q corporate financial statements, operating margins, and solvency metrics.",
    ),
    SecurityType.ETF: InstrumentEvidenceContract(
        security_type=SecurityType.ETF,
        required_evidence={"FUND_STRUCTURE_PROFILE"},
        not_applicable_evidence={
            "CORPORATE_FINANCIALS_10K_10Q",
            "FORM_4_CSUITE_INSIDERS",
            "OPERATING_MARGIN",
            "ROIC",
        },
        incomplete_reason="Fund structure profile unverified; AUM and benchmark tracking required.",
        profile_label="Fund / ETF Profile",
        profile_description="Fund / ETF Profile: Evaluated via fund liquidity, net expense ratio, and underlying index momentum. Corporate 10-K financial filings are not applicable.",
    ),
    SecurityType.ADR: InstrumentEvidenceContract(
        security_type=SecurityType.ADR,
        required_evidence={"FOREIGN_ISSUER_DISCLOSURES_20F_6K"},
        not_applicable_evidence={"US_DOMESTIC_FORM_10K_10Q"},
        incomplete_reason="Foreign issuer Form 20-F/6-K disclosures unverified.",
        profile_label="ADR Foreign Issuer Profile",
        profile_description="Evaluated via foreign issuer SEC Form 20-F/6-K disclosures and home-market cross-listing liquidity. US domestic 10-K is not applicable.",
    ),
    SecurityType.REIT: InstrumentEvidenceContract(
        security_type=SecurityType.REIT,
        required_evidence={"REIT_STATUTORY_FILINGS_FFO"},
        not_applicable_evidence={"TRADITIONAL_OPERATING_GROSS_MARGIN"},
        incomplete_reason="REIT statutory filings unverified.",
        profile_label="REIT Capital Profile",
        profile_description="Evaluated via REIT statutory disclosures, Funds From Operations (FFO/AFFO), and property portfolio leverage. Traditional gross margin is not applicable.",
    ),
    SecurityType.OTHER: InstrumentEvidenceContract(
        security_type=SecurityType.OTHER,
        required_evidence={"SPECIALIZED_INSTRUMENT_PROFILE"},
        not_applicable_evidence={"CORPORATE_FINANCIALS_10K_10Q"},
        incomplete_reason="Specialized instrument structure unverified.",
        profile_label="Specialized Security Profile",
        profile_description="Evaluated via specialized security liquidity and structure disclosures.",
    ),
    SecurityType.UNKNOWN: InstrumentEvidenceContract(
        security_type=SecurityType.UNKNOWN,
        required_evidence={"ALL_EVIDENCE"},
        not_applicable_evidence=set(),
        incomplete_reason="Unclassified instrument: Security identity cannot be verified safely in canonical security master.",
        profile_label="Unverified Instrument",
        profile_description="Instrument class unclassified in security master. Fails closed.",
    ),
}


def get_required_evidence_for_instrument(
    security_type: Optional[SecurityType | str] = None,
    asset_class: Optional[AssetClass | str] = None,
) -> InstrumentEvidenceContract:
    """
    Resolves the canonical regulatory evidence contract based on security classification.
    Adheres strictly to server security master taxonomy.
    """
    if security_type is None and asset_class is None:
        return INSTRUMENT_EVIDENCE_REGISTRY[SecurityType.COMMON_STOCK]

    # Resolve SecurityType enum
    sec_type_enum: SecurityType = SecurityType.UNKNOWN
    if isinstance(security_type, SecurityType):
        sec_type_enum = security_type
    elif isinstance(security_type, str):
        clean_st = security_type.strip().upper()
        if clean_st in SecurityType.__members__:
            sec_type_enum = SecurityType[clean_st]
        elif clean_st in {"ETF", "ETN", "ETF_OR_REGISTERED_FUND", "MUTUAL_FUND"}:
            sec_type_enum = SecurityType.ETF
        elif clean_st in {"COMMON", "STOCK", "COMMON_STOCK"}:
            sec_type_enum = SecurityType.COMMON_STOCK
        elif clean_st == "ADR":
            sec_type_enum = SecurityType.ADR
        elif clean_st == "REIT":
            sec_type_enum = SecurityType.REIT

    # Check asset_class fallback if security_type was unspecified or UNKNOWN
    if sec_type_enum == SecurityType.UNKNOWN and asset_class is not None:
        clean_ac = asset_class.value if isinstance(asset_class, AssetClass) else str(asset_class).strip().upper()
        if clean_ac in {"ETF", "FUND"}:
            sec_type_enum = SecurityType.ETF
        elif clean_ac == "EQUITY":
            sec_type_enum = SecurityType.COMMON_STOCK

    return INSTRUMENT_EVIDENCE_REGISTRY.get(sec_type_enum, INSTRUMENT_EVIDENCE_REGISTRY[SecurityType.UNKNOWN])
