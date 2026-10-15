"""
analyst_dashboard/universe/fixtures.py

Canonical Deterministic Fixtures for Universe Construction & VCP Regression.
Demotes the hand-maintained 35-symbol list to an explicit historical regression fixture.
"""

from typing import List, Dict, Any
from .contracts import SourceSecurity, SourcePopulationSnapshot, compute_sha256

# ── Demoted Historical 35-Symbol Curated Fixture ─────────────────────────────
# Retained STRICTLY as a deterministic regression fixture and test corpus.
# MUST NOT be used as production membership authority.
HISTORICAL_VCP_35_REGRESSION_FIXTURE: List[str] = [
    # MedTech & Biotech Monopolies
    "LNTH", "CPRX", "MEDP", "TMDX", "ISRG", "VRTX", "LLY", "NVO", "DXCM", "PODD",
    # High-Moat Semiconductors & SiC Ion Implantation
    "ACLS", "POWI", "ON", "MPWR", "KLAC", "LRCX", "ASML", "AVGO",
    # Peter Lynch GARP & Organic Consumer Compounders
    "ELF", "DECK", "LULU", "ONON", "MNST", "ULTA",
    # Clean Tech, Power Infrastructure & Industrials
    "VRT", "ETN", "PWR", "GEV", "FIX", "EME",
    # Disruptive Cloud, EdTech & EDA Infrastructure
    "DUOL", "ANET", "NOW", "SNPS", "CDNS",
]

# ── Canonical Multi-Asset Universe Test Population ───────────────────────────
# Deterministic benchmark containing diverse instrument types to test all eligibility rules.
CANONICAL_FIXTURE_SECURITIES: List[Dict[str, Any]] = [
    # Eligible Common Stocks (US Major Exchanges, Active)
    {"security_id": "SEC_NVDA", "symbol": "NVDA", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_PLTR", "symbol": "PLTR", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_AAPL", "symbol": "AAPL", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_MSFT", "symbol": "MSFT", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_AMD", "symbol": "AMD", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_CRWD", "symbol": "CRWD", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_ACLS", "symbol": "ACLS", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_LNTH", "symbol": "LNTH", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_TMDX", "symbol": "TMDX", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_DUOL", "symbol": "DUOL", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_LLY", "symbol": "LLY", "exchange": "NYSE", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},
    {"security_id": "SEC_VRTX", "symbol": "VRTX", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE"},

    # Ineligible: ETF (Rule R02 Fail)
    {"security_id": "SEC_SPY", "symbol": "SPY", "exchange": "ARCA", "security_type": "ETF", "listing_status": "ACTIVE", "asset_class": "ETF"},
    {"security_id": "SEC_QQQ", "symbol": "QQQ", "exchange": "NASDAQ", "security_type": "ETF", "listing_status": "ACTIVE", "asset_class": "ETF"},

    # Ineligible: ADR (Rule R02 Fail)
    {"security_id": "SEC_TSM", "symbol": "TSM", "exchange": "NYSE", "security_type": "ADR", "listing_status": "ACTIVE"},
    {"security_id": "SEC_NVO", "symbol": "NVO", "exchange": "NYSE", "security_type": "ADR", "listing_status": "ACTIVE"},

    # Ineligible: REIT (Rule R02 Fail)
    {"security_id": "SEC_O", "symbol": "O", "exchange": "NYSE", "security_type": "REIT", "listing_status": "ACTIVE"},

    # Ineligible: Warrant (Rule R02 Fail)
    {"security_id": "SEC_CORZW", "symbol": "CORZW", "exchange": "NASDAQ", "security_type": "WARRANT", "listing_status": "ACTIVE"},

    # Ineligible: Preferred Stock (Rule R02 Fail)
    {"security_id": "SEC_BACPRK", "symbol": "BAC.PRK", "exchange": "NYSE", "security_type": "PREFERRED", "listing_status": "ACTIVE"},

    # Ineligible: Inactive / Delisted (Rule R03 Fail)
    {"security_id": "SEC_INACT1", "symbol": "INACT1", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "INACTIVE"},
    {"security_id": "SEC_DELIST1", "symbol": "DELIST1", "exchange": "NYSE", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "delisting_date": "2026-01-15"},

    # Ineligible: Non-US Exchange (Rule R04 Fail)
    {"security_id": "SEC_LON1", "symbol": "LON1", "exchange": "LSE", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "country": "GBR", "currency": "GBP"},
    {"security_id": "SEC_TOR1", "symbol": "TOR1", "exchange": "TSX", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "country": "CAN", "currency": "CAD"},

    # Ineligible: Non-USD Currency (Rule R05 Fail)
    {"security_id": "SEC_EUR1", "symbol": "EUR1", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "currency": "EUR"},

    # Ineligible: Secondary Listing (Rule R06 Fail)
    {"security_id": "SEC_SEC1", "symbol": "SEC1", "exchange": "NASDAQ", "security_type": "COMMON_STOCK", "listing_status": "ACTIVE", "primary_listing": False},
]


def create_canonical_fixture_snapshot(snapshot_id: str = "fixture-snap-001") -> SourcePopulationSnapshot:
    """Creates a frozen SourcePopulationSnapshot from canonical fixture securities."""
    securities = [
        SourceSecurity(
            security_id=s["security_id"],
            symbol=s["symbol"],
            exchange=s["exchange"],
            security_type=s["security_type"],
            listing_status=s["listing_status"],
            currency=s.get("currency", "USD"),
            country=s.get("country", "USA"),
            primary_listing=s.get("primary_listing", True),
            asset_class=s.get("asset_class", "EQUITY"),
            listing_date=s.get("listing_date"),
            delisting_date=s.get("delisting_date"),
        )
        for s in CANONICAL_FIXTURE_SECURITIES
    ]
    raw_dicts = [s.to_dict() for s in securities]
    source_hash = compute_sha256(raw_dicts)
    return SourcePopulationSnapshot(
        snapshot_id=snapshot_id,
        source_authority="ARX_CANONICAL_FIXTURE_SECURITY_MASTER",
        as_of="2026-10-09T00:00:00Z",
        count=len(securities),
        source_hash=source_hash,
        securities=securities,
    )
