"""
analyst_dashboard/security_master/eligibility.py

Execution Eligibility Policy Engine for ARX Terminal Security Master.
Decoupled strictly from asset classification and quantitative analytics.

Invariants Enforced:
- INV-SECMASTER-03: Execution eligibility is strictly decoupled from asset identity.
- INV-SECMASTER-04: Analytics capability does not grant execution eligibility.
- INV-SECMASTER-05: UNKNOWN classification strictly fails closed.
- INV-SECMASTER-06: UNSUPPORTED classification strictly fails closed.
- INV-SECMASTER-07: CONFLICTED classification strictly fails closed.
- INV-SECMASTER-08: ETF routing strictly isolated to ETF execution surface.
- INV-SECMASTER-09: Crypto routing strictly isolated to Crypto execution surface.
- INV-SECMASTER-10: Specialized equity subtypes (ADR, REIT, Preferred, Warrant, Unit, Right) fail closed.
- INV-SECMASTER-20: Zero new execution thresholds invented.
"""

from __future__ import annotations

from .models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
    ExecutionEligibility,
    CanonicalInstrument,
)


def evaluate_execution_eligibility(
    asset_class: AssetClass,
    security_type: SecurityType,
    listing_status: ListingStatus,
    classification_status: ClassificationStatus,
) -> ExecutionEligibility:
    """
    Evaluates authoritative execution eligibility from normalized classification and status.
    Fails closed for all non-verified, inactive, conflicted, or unsupported combinations.
    """
    # Fail closed on any non-verified status (CONFLICTED, UNVERIFIED, etc.)
    if classification_status != ClassificationStatus.VERIFIED:
        return ExecutionEligibility.FAIL_CLOSED

    # Inactive listings strictly fail closed
    if listing_status != ListingStatus.ACTIVE:
        return ExecutionEligibility.FAIL_CLOSED

    # Verified operating Common Stock
    if security_type == SecurityType.COMMON_STOCK and asset_class == AssetClass.EQUITY:
        return ExecutionEligibility.STOCK_EXECUTION

    # Verified Exchange-Traded Fund (ETF / ETP)
    if security_type == SecurityType.ETF or asset_class == AssetClass.ETF:
        return ExecutionEligibility.ETF_EXECUTION

    # Verified Crypto asset
    if security_type == SecurityType.CRYPTO or asset_class == AssetClass.CRYPTO:
        return ExecutionEligibility.CRYPTO_EXECUTION

    # All specialized, unsupported, or derivative subtypes strictly fail closed:
    # ADR, REIT, PREFERRED, WARRANT, UNIT, RIGHT, OTHER, UNKNOWN
    return ExecutionEligibility.FAIL_CLOSED
