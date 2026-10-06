"""
tests/fixtures/security_master_fixtures.py

Deterministic fixtures and mock adapters for ARX Terminal Security Master contract tests.
Requires zero live provider requests.
"""

from typing import Dict, Optional
from analyst_dashboard.security_master.models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
)
from analyst_dashboard.security_master.alpaca_adapter import AlpacaAssetEvidence, AlpacaIdentityAdapter
from analyst_dashboard.security_master.openfigi_adapter import OpenFIGISubtypeEvidence, OpenFIGISubtypeAdapter

# Fixed deterministic evidence registry
FIXTURE_EVIDENCE_REGISTRY = {
    "PLSE": {
        "alpaca": AlpacaAssetEvidence(
            symbol="PLSE",
            success=True,
            provider_asset_id="c919e896-8ac4-4bd3-a6c6-837ef687d7f0",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NASDAQ",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": True, "fractionable": True},
            raw_payload={"id": "c919e896", "class": "us_equity", "exchange": "NASDAQ", "symbol": "PLSE", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="PLSE",
            success=True,
            security_type=SecurityType.COMMON_STOCK,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Common Stock",
            raw_security_type2="Common Stock",
            market_sector="Equity",
            figi="BBG00BRBHVD0",
            composite_figi="BBG00BRBHVD0",
            share_class_figi="BBG00BRBHVG7",
            exch_code="US",
        ),
    },
    "AAPL": {
        "alpaca": AlpacaAssetEvidence(
            symbol="AAPL",
            success=True,
            provider_asset_id="b0b6dd9d-8b9b-48a9-ba46-b9d54906e415",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NASDAQ",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": True},
            raw_payload={"id": "b0b6dd9d", "class": "us_equity", "exchange": "NASDAQ", "symbol": "AAPL", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="AAPL",
            success=True,
            security_type=SecurityType.COMMON_STOCK,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Common Stock",
            raw_security_type2="Common Stock",
            market_sector="Equity",
            figi="BBG000B9XRY4",
            composite_figi="BBG000B9XRY4",
            share_class_figi="BBG001S5N8V8",
            exch_code="US",
        ),
    },
    "SPY": {
        "alpaca": AlpacaAssetEvidence(
            symbol="SPY",
            success=True,
            provider_asset_id="3e0de6f1-bb2e-4b67-96a6-574bf25e3d74",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="ARCA",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": True},
            raw_payload={"id": "3e0de6f1", "class": "us_equity", "exchange": "ARCA", "symbol": "SPY", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="SPY",
            success=True,
            security_type=SecurityType.ETF,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="ETP",
            raw_security_type2="Mutual Fund",
            market_sector="Equity",
            figi="BBG000BDTBL9",
            composite_figi="BBG000BDTBL9",
            share_class_figi="BBG001S5T340",
            exch_code="US",
        ),
    },
    "TSM": {
        "alpaca": AlpacaAssetEvidence(
            symbol="TSM",
            success=True,
            provider_asset_id="841961e5-7489-4971-bb6c-bfd376c2576b",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NYSE",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": True},
            raw_payload={"id": "841961e5", "class": "us_equity", "exchange": "NYSE", "symbol": "TSM", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="TSM",
            success=True,
            security_type=SecurityType.ADR,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="ADR",
            raw_security_type2="Depositary Receipt",
            market_sector="Equity",
            figi="BBG000BDY5M1",
            composite_figi="BBG000BDY5M1",
            share_class_figi="BBG001S6F2W2",
            exch_code="US",
        ),
    },
    "O": {
        "alpaca": AlpacaAssetEvidence(
            symbol="O",
            success=True,
            provider_asset_id="7337a77e-2f54-4ca4-bb37-25e24b0cf611",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NYSE",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": True},
            raw_payload={"id": "7337a77e", "class": "us_equity", "exchange": "NYSE", "symbol": "O", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="O",
            success=True,
            security_type=SecurityType.REIT,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="REIT",
            raw_security_type2="REIT",
            market_sector="Equity",
            figi="BBG000BD7F41",
            composite_figi="BBG000BD7F41",
            share_class_figi="BBG001S5K435",
            exch_code="US",
        ),
    },
    "CORZW": {
        "alpaca": AlpacaAssetEvidence(
            symbol="CORZW",
            success=True,
            provider_asset_id="warrant-corzw-uuid",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NASDAQ",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": False},
            raw_payload={"id": "warrant-corzw", "class": "us_equity", "exchange": "NASDAQ", "symbol": "CORZW", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="CORZW",
            success=True,
            security_type=SecurityType.WARRANT,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Equity WRT",
            raw_security_type2="Warrant",
            market_sector="Equity",
            figi="BBG014XN9G42",
            composite_figi="BBG014XN9G42",
            exch_code="US",
        ),
    },
    "BAC.PRK": {
        "alpaca": AlpacaAssetEvidence(
            symbol="BAC.PRK",
            success=True,
            provider_asset_id="pref-bac-prk-uuid",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NYSE",
            listing_status=ListingStatus.ACTIVE,
            tradability_metadata={"tradable": True, "shortable": False},
            raw_payload={"id": "pref-bac-prk", "class": "us_equity", "exchange": "NYSE", "symbol": "BAC.PRK", "status": "active"},
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="BAC.PRK",
            success=True,
            security_type=SecurityType.PREFERRED,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Preferred Stock",
            raw_security_type2="Preferred Stock",
            market_sector="Equity",
            figi="BBG00R0H57B8",
            composite_figi="BBG00R0H57B8",
            exch_code="US",
        ),
    },
    "INVALID_XYZ_999": {
        "alpaca": AlpacaAssetEvidence(
            symbol="INVALID_XYZ_999",
            success=False,
            error_message="ASSET_NOT_FOUND_404",
            listing_status=ListingStatus.UNKNOWN,
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="INVALID_XYZ_999",
            success=False,
            error_message="NO_IDENTIFIER_FOUND",
            classification_status=ClassificationStatus.UNVERIFIED,
        ),
    },
    "CONFLICTED_MOCK": {
        "alpaca": AlpacaAssetEvidence(
            symbol="CONFLICTED_MOCK",
            success=True,
            provider_asset_id="crypto-uuid",
            broad_asset_class=AssetClass.CRYPTO,
            primary_exchange="CRYPTO",
            listing_status=ListingStatus.ACTIVE,
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="CONFLICTED_MOCK",
            success=True,
            security_type=SecurityType.COMMON_STOCK,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Common Stock",
            market_sector="Equity",
        ),
    },
    "NO_SUBTYPE_MOCK": {
        "alpaca": AlpacaAssetEvidence(
            symbol="NO_SUBTYPE_MOCK",
            success=True,
            provider_asset_id="nosubtype-uuid",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NASDAQ",
            listing_status=ListingStatus.ACTIVE,
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="NO_SUBTYPE_MOCK",
            success=False,
            error_message="NO_IDENTIFIER_FOUND",
            classification_status=ClassificationStatus.UNVERIFIED,
        ),
    },
    "INACTIVE_MOCK": {
        "alpaca": AlpacaAssetEvidence(
            symbol="INACTIVE_MOCK",
            success=True,
            provider_asset_id="inactive-uuid",
            broad_asset_class=AssetClass.EQUITY,
            primary_exchange="NASDAQ",
            listing_status=ListingStatus.INACTIVE,
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="INACTIVE_MOCK",
            success=True,
            security_type=SecurityType.COMMON_STOCK,
            classification_status=ClassificationStatus.VERIFIED,
            raw_security_type="Common Stock",
            market_sector="Equity",
        ),
    },
    "PROVIDER_OUTAGE_MOCK": {
        "alpaca": AlpacaAssetEvidence(
            symbol="PROVIDER_OUTAGE_MOCK",
            success=False,
            error_message="TIMEOUT",
            listing_status=ListingStatus.UNVERIFIED,
        ),
        "openfigi": OpenFIGISubtypeEvidence(
            symbol="PROVIDER_OUTAGE_MOCK",
            success=False,
            error_message="TIMEOUT",
            classification_status=ClassificationStatus.UNVERIFIED,
        ),
    },
}


class MockAlpacaAdapter(AlpacaIdentityAdapter):
    """Deterministic offline mock for Alpaca identity adapter."""

    def __init__(self, override_registry: Optional[Dict[str, AlpacaAssetEvidence]] = None):
        super().__init__(api_key_id="mock_key", api_secret_key="mock_secret")
        self.override_registry = override_registry or {}

    def fetch_asset_evidence(self, symbol: str) -> AlpacaAssetEvidence:
        sym = symbol.strip().upper()
        if sym in self.override_registry:
            return self.override_registry[sym]
        if sym in FIXTURE_EVIDENCE_REGISTRY:
            return FIXTURE_EVIDENCE_REGISTRY[sym]["alpaca"]
        return AlpacaAssetEvidence(
            symbol=sym,
            success=False,
            error_message="ASSET_NOT_FOUND_404",
            listing_status=ListingStatus.UNKNOWN,
        )


class MockOpenFIGIAdapter(OpenFIGISubtypeAdapter):
    """Deterministic offline mock for OpenFIGI subtype adapter."""

    def __init__(self, override_registry: Optional[Dict[str, OpenFIGISubtypeEvidence]] = None):
        super().__init__(api_key="mock_figi_key", rate_limiter=None)
        self.override_registry = override_registry or {}

    def fetch_subtype_evidence(self, symbol: str) -> OpenFIGISubtypeEvidence:
        sym = symbol.strip().upper()
        if sym in self.override_registry:
            return self.override_registry[sym]
        if sym in FIXTURE_EVIDENCE_REGISTRY:
            return FIXTURE_EVIDENCE_REGISTRY[sym]["openfigi"]
        return OpenFIGISubtypeEvidence(
            symbol=sym,
            success=False,
            error_message="NO_IDENTIFIER_FOUND",
            classification_status=ClassificationStatus.UNVERIFIED,
        )
