"""
analyst_dashboard/security_master/normalization.py

Composite Normalization and Conflict Engine for ARX Terminal Security Master.
Normalizes Alpaca asset directory evidence with OpenFIGI v3 subtype evidence.

Invariants Enforced:
- INV-SECMASTER-15: Alpaca broad asset class cannot independently establish security subtype.
- INV-SECMASTER-16: OpenFIGI subtype evidence and Alpaca listing evidence are normalized server-side.
- INV-SECMASTER-17: Provider conflicts capable of changing eligibility fail closed.
- Material disagreements result in CONFLICTED and FAIL_CLOSED.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timezone

from .models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
    ExecutionEligibility,
    AnalyticsCapability,
    CanonicalInstrument,
)
from .alpaca_adapter import AlpacaAssetEvidence
from .openfigi_adapter import OpenFIGISubtypeEvidence
from .eligibility import evaluate_execution_eligibility

logger = logging.getLogger("arx.security_master.normalization")

# Known exchange normalization aliases (harmless presentation mapping)
EXCHANGE_ALIASES = {
    "XNAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
    "XNYS": "NYSE",
    "NYSE": "NYSE",
    "ARCX": "ARCA",
    "ARCA": "ARCA",
    "BATS": "BATS",
    "IEX": "IEX",
}


def normalize_exchange(raw_exchange: Optional[str]) -> str:
    if not raw_exchange:
        return "UNKNOWN"
    clean = raw_exchange.strip().upper()
    return EXCHANGE_ALIASES.get(clean, clean)


class SecurityMasterNormalizationEngine:
    """
    Field-level composite normalization engine.
    Applies strict precedence, detects material conflicts, and produces CanonicalInstrument.
    """

    def normalize(
        self,
        symbol: str,
        alpaca_evidence: Optional[AlpacaAssetEvidence],
        openfigi_evidence: Optional[OpenFIGISubtypeEvidence],
    ) -> CanonicalInstrument:
        clean_symbol = symbol.strip().upper()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Handle complete provider failure
        if (alpaca_evidence is None or not alpaca_evidence.success) and (openfigi_evidence is None or not openfigi_evidence.success):
            logger.info(f"Both Alpaca and OpenFIGI evidence unavailable for {clean_symbol}; marking UNVERIFIED.")
            return CanonicalInstrument(
                symbol=clean_symbol,
                provider_symbol=clean_symbol,
                asset_class=AssetClass.UNKNOWN,
                security_type=SecurityType.UNKNOWN,
                primary_exchange="UNKNOWN",
                listing_status=ListingStatus.UNVERIFIED,
                classification_status=ClassificationStatus.UNVERIFIED,
                execution_eligibility=ExecutionEligibility.FAIL_CLOSED,
                analytics_capability=AnalyticsCapability.UNKNOWN,
                classification_authority="ARX_SERVER_SECURITY_MASTER",
                source_provenance={
                    "alpaca": alpaca_evidence.to_provenance() if alpaca_evidence else {"success": False, "reason": "NO_EVIDENCE"},
                    "openfigi": openfigi_evidence.to_provenance() if openfigi_evidence else {"success": False, "reason": "NO_EVIDENCE"},
                    "conflict_detected": False,
                },
                classification_timestamp=now_iso,
                stable_identifiers={},
            )

        # 1. Identity & Provider Symbol
        provider_symbol = clean_symbol
        if alpaca_evidence and alpaca_evidence.success:
            provider_symbol = alpaca_evidence.symbol.upper()

        # 2. Primary Exchange
        primary_exchange = "UNKNOWN"
        if alpaca_evidence and alpaca_evidence.success:
            primary_exchange = normalize_exchange(alpaca_evidence.primary_exchange)

        # 3. Listing Status (Alpaca is authoritative)
        listing_status = ListingStatus.UNVERIFIED
        if alpaca_evidence and alpaca_evidence.success:
            listing_status = alpaca_evidence.listing_status

        # 4. Broad Asset Class
        broad_class = AssetClass.UNKNOWN
        if alpaca_evidence and alpaca_evidence.success:
            broad_class = alpaca_evidence.broad_asset_class

        # Check OpenFIGI corroboration of broad class
        if openfigi_evidence and openfigi_evidence.success:
            if openfigi_evidence.security_type == SecurityType.ETF:
                # ETPs under Alpaca are often classified under broad class us_equity; normalize to ETF
                broad_class = AssetClass.ETF
            elif openfigi_evidence.market_sector and openfigi_evidence.market_sector.lower() == "equity" and broad_class == AssetClass.UNKNOWN:
                broad_class = AssetClass.EQUITY

        # 5. Security Subtype (OpenFIGI is authoritative)
        security_type = SecurityType.UNKNOWN
        if openfigi_evidence and openfigi_evidence.success:
            security_type = openfigi_evidence.security_type
        elif broad_class == AssetClass.CRYPTO:
            # Alpaca natively partitions crypto assets; OpenFIGI does not map US crypto pairs
            security_type = SecurityType.CRYPTO

        # 6. Conflict Detection
        conflict_detected = False
        conflict_reasons = []

        # Check symbol identity disagreement
        if (
            alpaca_evidence and alpaca_evidence.success and
            openfigi_evidence and openfigi_evidence.success
        ):
            # Check if providers disagree materially on symbol
            if alpaca_evidence.symbol.upper() != openfigi_evidence.symbol.upper():
                conflict_detected = True
                conflict_reasons.append(f"SYMBOL_MISMATCH: Alpaca={alpaca_evidence.symbol} vs OpenFIGI={openfigi_evidence.symbol}")

            # Check if broad asset class conflicts
            # e.g. Alpaca claims crypto while OpenFIGI claims Equity Common Stock
            if alpaca_evidence.broad_asset_class == AssetClass.CRYPTO and openfigi_evidence.security_type == SecurityType.COMMON_STOCK:
                conflict_detected = True
                conflict_reasons.append("ASSET_CLASS_CONFLICT: Alpaca=CRYPTO vs OpenFIGI=COMMON_STOCK")

            # Check if Alpaca indicates inactive but OpenFIGI has active trading metadata that would conflict
            if alpaca_evidence.listing_status == ListingStatus.INACTIVE and openfigi_evidence.security_type == SecurityType.COMMON_STOCK:
                # Inactive listing status is authoritative for execution failure, not necessarily a data conflict
                pass

        # 7. Classification Status Determination
        if conflict_detected:
            classification_status = ClassificationStatus.CONFLICTED
        elif (
            (alpaca_evidence and alpaca_evidence.success) and
            (openfigi_evidence and openfigi_evidence.success and openfigi_evidence.classification_status == ClassificationStatus.VERIFIED)
        ):
            classification_status = ClassificationStatus.VERIFIED
        elif (
            alpaca_evidence and alpaca_evidence.success and
            alpaca_evidence.broad_asset_class == AssetClass.CRYPTO
        ):
            classification_status = ClassificationStatus.VERIFIED
        elif (
            alpaca_evidence and alpaca_evidence.success and
            (openfigi_evidence is None or not openfigi_evidence.success or openfigi_evidence.security_type == SecurityType.UNKNOWN)
        ):
            # Alpaca proves active us_equity, but OpenFIGI subtype is missing -> UNVERIFIED
            classification_status = ClassificationStatus.UNVERIFIED
        else:
            classification_status = ClassificationStatus.UNVERIFIED

        # 8. Execution Eligibility Policy
        eligibility = evaluate_execution_eligibility(
            asset_class=broad_class,
            security_type=security_type,
            listing_status=listing_status,
            classification_status=classification_status,
        )

        # 9. Stable Identifiers
        stable_ids = {}
        if alpaca_evidence and alpaca_evidence.provider_asset_id:
            stable_ids["alpaca_asset_id"] = alpaca_evidence.provider_asset_id
        if openfigi_evidence:
            if openfigi_evidence.figi:
                stable_ids["figi"] = openfigi_evidence.figi
            if openfigi_evidence.composite_figi:
                stable_ids["composite_figi"] = openfigi_evidence.composite_figi
            if openfigi_evidence.share_class_figi:
                stable_ids["share_class_figi"] = openfigi_evidence.share_class_figi

        # 10. Audit Provenance
        provenance = {
            "alpaca": alpaca_evidence.to_provenance() if alpaca_evidence else {"success": False, "reason": "NO_EVIDENCE"},
            "openfigi": openfigi_evidence.to_provenance() if openfigi_evidence else {"success": False, "reason": "NO_EVIDENCE"},
            "conflict_detected": conflict_detected,
            "conflict_reasons": conflict_reasons,
            "normalized_at": now_iso,
        }

        return CanonicalInstrument(
            symbol=clean_symbol,
            provider_symbol=provider_symbol,
            asset_class=broad_class,
            security_type=security_type,
            primary_exchange=primary_exchange,
            listing_status=listing_status,
            classification_status=classification_status,
            execution_eligibility=eligibility,
            analytics_capability=AnalyticsCapability.SUPPORTED if broad_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO) else AnalyticsCapability.UNSUPPORTED,
            classification_authority="ARX_SERVER_SECURITY_MASTER",
            source_provenance=provenance,
            classification_timestamp=now_iso,
            stable_identifiers=stable_ids,
        )
