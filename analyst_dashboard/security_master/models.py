"""
analyst_dashboard/security_master/models.py

Canonical Domain Model and Enums for ARX Terminal Security Master.
Enforces INV-SECMASTER-01 through INV-SECMASTER-20.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    FUND = "FUND"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class SecurityType(str, Enum):
    COMMON_STOCK = "COMMON_STOCK"
    ADR = "ADR"
    REIT = "REIT"
    PREFERRED = "PREFERRED"
    WARRANT = "WARRANT"
    UNIT = "UNIT"
    RIGHT = "RIGHT"
    ETF = "ETF"
    CRYPTO = "CRYPTO"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ListingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"


class ClassificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTED = "CONFLICTED"


class ExecutionEligibility(str, Enum):
    STOCK_EXECUTION = "STOCK_EXECUTION"
    ETF_EXECUTION = "ETF_EXECUTION"
    CRYPTO_EXECUTION = "CRYPTO_EXECUTION"
    FAIL_CLOSED = "FAIL_CLOSED"


class AnalyticsCapability(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    FULL_ANALYTICS = "FULL_ANALYTICS"
    PARTIAL_ANALYTICS = "PARTIAL_ANALYTICS"
    UNKNOWN = "UNKNOWN"


class CanonicalInstrument(BaseModel):
    """
    Canonical server-owned representation of a financial instrument.
    Decouples identity, security subtype, and execution eligibility.
    Strictly conforms to Section 8 Canonical Contract in ARX_CANONICAL_SECURITY_MASTER_DESIGN.md.
    """
    symbol: str = Field(..., description="Canonical ticker symbol")
    provider_symbol: str = Field(..., description="Provider-native symbol used for market routing")
    asset_class: AssetClass = Field(default=AssetClass.UNKNOWN, description="Broad asset class")
    security_type: SecurityType = Field(default=SecurityType.UNKNOWN, description="Structural security subtype")
    primary_exchange: str = Field(default="UNKNOWN", description="Primary listing exchange or venue")
    listing_status: ListingStatus = Field(default=ListingStatus.UNKNOWN, description="Current listing/activity state")
    classification_status: ClassificationStatus = Field(
        default=ClassificationStatus.UNVERIFIED,
        description="Confidence state of the composite classification"
    )
    execution_eligibility: ExecutionEligibility = Field(
        default=ExecutionEligibility.FAIL_CLOSED,
        description="Authoritative execution routing eligibility"
    )
    analytics_capability: AnalyticsCapability = Field(
        default=AnalyticsCapability.UNKNOWN,
        description="Independent quantitative analytics support state"
    )
    classification_authority: str = Field(
        default="ARX_SERVER_SECURITY_MASTER",
        description="Authoritative subsystem certifying this classification"
    )
    source_provider: str = Field(
        default="COMPOSITE_ALPACA_OPENFIGI",
        description="Authoritative source provider identifier"
    )
    classification_timestamp: str = Field(
        default="",
        description="ISO 8601 UTC timestamp of when classification was resolved"
    )
    stable_identifier: Optional[str] = Field(
        default=None,
        description="Primary provider-stable identifier (FIGI)"
    )
    stable_identifiers: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Provider-stable identifiers (FIGI, composite FIGI, share class FIGI, Alpaca UUID)"
    )
    provider_provenance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Audit trail of provider evidence, raw fields, and timestamps"
    )
    source_provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit trail alias"
    )

    model_config = ConfigDict(use_enum_values=True)

    def model_post_init(self, __context: Any) -> None:
        # Synchronize stable_identifier and stable_identifiers
        if not self.stable_identifier and self.stable_identifiers:
            primary_id = (
                self.stable_identifiers.get("figi")
                or self.stable_identifiers.get("composite_figi")
                or self.stable_identifiers.get("share_class_figi")
                or self.stable_identifiers.get("alpaca_asset_id")
            )
            object.__setattr__(self, "stable_identifier", primary_id)
        elif self.stable_identifier and not self.stable_identifiers:
            object.__setattr__(self, "stable_identifiers", {"figi": self.stable_identifier})

        # Synchronize provider_provenance and source_provenance
        if self.provider_provenance is None and self.source_provenance:
            object.__setattr__(self, "provider_provenance", self.source_provenance)
        elif self.provider_provenance is not None and not self.source_provenance:
            object.__setattr__(self, "source_provenance", self.provider_provenance)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
