"""
tests/test_form_497k_identity_qualification.py

Comprehensive test suite for:
ETF_V2_FORM_497K_IDENTITY_QUALIFICATION_POLICY_IMPLEMENTATION_GATE

Verifies:
1. Positive qualification across approved variation classes
2. 7-condition enforcement (Form, CIK, Ticker Declaration, Accession, Variation Class, Mandate, Sibling Collision)
3. Mandatory negative tests (non-497K form, CIK mismatch, narrative ticker like SIZE, uncertified accession, hollow supplement)
4. Five fail-closed targets (BFOR, OEFA, OGIG, OUSA, OUSM)
5. STXF hard isolation
"""

import pytest
from pathlib import Path
import json

from scripts.research.etf_v2.models import EntityIdentity, FilingMetadata
from scripts.research.etf_v2.identity_authority import IdentityAuthority
from scripts.research.etf_v2.pipeline import ETFPipelineV2
from scripts.research.etf_v2.prospectus_authority import ProspectusAuthorityResolver


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    return ETFPipelineV2(REPO_ROOT)


@pytest.fixture(scope="module")
def qualification_ledger():
    ledger_path = REPO_ROOT / "docs" / "research" / "ETF_V2_FORM_497K_QUALIFICATION_LEDGER.json"
    with open(ledger_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_positive_qualification_across_approved_variation_classes(pipeline, qualification_ledger):
    """Verifies that approved targets across different variation classes qualify properly."""
    census = {r["symbol"]: r for r in qualification_ledger["census_records"]}
    
    # Sample symbols from various approved variation classes
    sample_symbols = [
        "AIVC",  # REBRANDING_NAME_EVOLUTION
        "BBAX",  # HYPHENATION_COMPOUND_WORD_VARIATION
        "CPNJ",  # SERIES_QUALIFIER_PUNCTUATION_OR_HYPHEN
        "DVYA",  # INDEX_FUND_VS_ETF_SUFFIX
        "GOVZ",  # NUMERIC_PLUS_QUALIFIER_VARIATION
        "NGIF",  # SHARE_CLASS_SUFFIX_VARIATION
        "NITE",  # TICKER_IN_NAME_VARIATION
        "PFM",   # TRADEMARK_OR_SYMBOL_QUALIFIER
        "PSR",   # LEGAL_SUFFIX_OR_STRATEGY_WORDING
        "RSPF",  # PLURAL_WORD_FORM_VARIATION
        "TACN",  # TRUST_BRAND_PREFIX_SEPARATION
        "TIPX",  # CHARACTER_ENCODING_CORRUPTION
    ]
    
    for sym in sample_symbols:
        r = census[sym]
        ident = EntityIdentity(
            symbol=sym,
            cik=r["cik"],
            series_id=r["series_id"],
            class_id=r.get("class_id", ""),
            legal_name=r["target_name"],
            historical_aliases=[]
        )
        candidates = pipeline.filing_universe.get_candidate_prospectuses(ident)
        auth = ProspectusAuthorityResolver.resolve_authority(candidates, ident)
        assert auth is not None, f"Expected {sym} to qualify, but was None"
        assert auth.filing.accession == r["base_accession"]
        assert auth.filing.form == "497K"


def test_mandatory_negative_non_497k_form():
    """Form guard: Form 497K supplement must NOT qualify 485BPOS or 497 filings without baseline match."""
    ident = EntityIdentity(
        symbol="AIVC",
        cik="0001633061",
        series_id="S000082270",
        class_id="C000245556",
        legal_name="Amplify Bloomberg AI Equal Weight ETF",
    )
    filing = FilingMetadata(
        accession="0001213900-26-008336",
        form="485BPOS",  # WRONG FORM
        filing_date="2026-01-01",
        acceptance_timestamp="2026-01-01T12:00:00Z",
        document_filename="doc.htm",
    )
    text = "Summary Prospectus | Amplify Bloomberg AI Value Chain ETF | Ticker: AIVC | Principal Investment Strategy: The Fund invests..."
    res = IdentityAuthority.match_identity(text, ident, filing=filing)
    assert not res["is_qualified"], "Non-497K form must not qualify under supplement"


def test_mandatory_negative_cik_mismatch():
    """CIK guard: Filing from a different registrant CIK must NOT qualify."""
    ident = EntityIdentity(
        symbol="AIVC",
        cik="0001633061",
        series_id="S000082270",
        class_id="C000245556",
        legal_name="Amplify Bloomberg AI Equal Weight ETF",
    )
    # 1. Filing with explicit different CIK
    class MockFilingWithCik:
        form = "497K"
        accession = "0001213900-26-008336"
        cik = "0009999999"  # WRONG CIK

    text = "Summary Prospectus | Amplify Bloomberg AI Value Chain ETF | Ticker: AIVC | Principal Investment Strategy: The Fund invests..."
    res = IdentityAuthority.match_identity(text, ident, filing=MockFilingWithCik())
    assert not res["is_qualified"], "Filing with mismatched CIK must not qualify"

    # 2. Identity with wrong CIK against cohort
    ident_wrong_cik = EntityIdentity(
        symbol="AIVC",
        cik="0009999999",  # WRONG TARGET CIK
        series_id="S000082270",
        class_id="C000245556",
        legal_name="Amplify Bloomberg AI Equal Weight ETF",
    )
    filing = FilingMetadata(
        accession="0001213900-26-008336",
        form="497K",
        filing_date="2026-01-01",
        acceptance_timestamp="2026-01-01T12:00:00Z",
        document_filename="doc.htm",
    )
    res2 = IdentityAuthority.match_identity(text, ident_wrong_cik, filing=filing)
    assert not res2["is_qualified"], "Target identity with mismatched CIK must not qualify"


def test_mandatory_negative_narrative_ticker_collision():
    """Ticker declaration guard: Narrative mentions of ticker must NOT qualify (e.g. SIZE)."""
    assert not IdentityAuthority.has_authoritative_ticker_declaration(
        "The Fund considers market capitalization size and portfolio size when selecting securities.",
        "SIZE"
    ), "Narrative word 'size' must not trigger ticker declaration"
    
    assert IdentityAuthority.has_authoritative_ticker_declaration(
        "Summary Prospectus | iShares MSCI USA Size Factor ETF | Ticker: SIZE | NYSE Arca",
        "SIZE"
    ), "Authoritative declaration 'Ticker: SIZE' must trigger ticker declaration"


def test_mandatory_negative_uncertified_accession():
    """Accession guard: Uncertified accession outside pre-boundary chain must NOT qualify."""
    ident = EntityIdentity(
        symbol="AIVC",
        cik="0001633061",
        series_id="S000082270",
        class_id="C000245556",
        legal_name="Amplify Bloomberg AI Equal Weight ETF",
    )
    filing = FilingMetadata(
        accession="0009999999-26-999999",  # UNCERTIFIED ACCESSION
        form="497K",
        filing_date="2026-01-01",
        acceptance_timestamp="2026-01-01T12:00:00Z",
        document_filename="doc.htm",
    )
    text = "Summary Prospectus | Amplify Bloomberg AI Value Chain ETF | Ticker: AIVC | Principal Investment Strategy: The Fund invests..."
    res = IdentityAuthority.match_identity(text, ident, filing=filing)
    assert not res["is_qualified"], "Uncertified accession must not qualify"


def test_mandatory_negative_hollow_supplement():
    """Mandate completeness guard: Hollow administrative supplement lacking mandate must NOT qualify."""
    ident = EntityIdentity(
        symbol="AIVC",
        cik="0001633061",
        series_id="S000082270",
        class_id="C000245556",
        legal_name="Amplify Bloomberg AI Equal Weight ETF",
    )
    filing = FilingMetadata(
        accession="0001213900-26-008336",
        form="497K",
        filing_date="2026-01-01",
        acceptance_timestamp="2026-01-01T12:00:00Z",
        document_filename="doc.htm",
    )
    # Hollow text lacking Item 4/Item 2 disclosures
    hollow_text = "Summary Prospectus | Amplify Bloomberg AI Value Chain ETF | Ticker: AIVC | Effective immediately, John Doe is added as portfolio manager."
    res = IdentityAuthority.match_identity(hollow_text, ident, filing=filing)
    assert not res["is_qualified"], "Hollow supplement lacking mandate evidence must not qualify"


def test_five_fail_closed_regression_cases(pipeline):
    """Explicitly audits that BFOR, OEFA, OGIG, OUSA, OUSM remain fail-closed."""
    fail_closed_symbols = ["BFOR", "OEFA", "OGIG", "OUSA", "OUSM"]
    ledger_path = REPO_ROOT / "docs" / "research" / "ETF_V2_FORM_497K_QUALIFICATION_LEDGER.json"
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    census = {r["symbol"]: r for r in ledger["census_records"]}
    
    for sym in fail_closed_symbols:
        r = census[sym]
        ident = EntityIdentity(
            symbol=sym,
            cik=r["cik"],
            series_id=r["series_id"],
            class_id=r.get("class_id", ""),
            legal_name=r["target_name"],
            historical_aliases=[]
        )
        candidates = pipeline.filing_universe.get_candidate_prospectuses(ident)
        auth = ProspectusAuthorityResolver.resolve_authority(candidates, ident)
        assert auth is None, f"{sym} must remain fail-closed (got authority: {auth})"


def test_stxf_hard_isolation(pipeline):
    """Verifies that Lane D target STXF remains strictly quarantined and unresolved."""
    ident_stxf = EntityIdentity(
        symbol="STXF",
        cik="0001799757",
        series_id="S000072044",
        class_id="C000227909",
        legal_name="Scanlon Tactical Systematic Alpha Fund",
        historical_aliases=[]
    )
    candidates = pipeline.filing_universe.get_candidate_prospectuses(ident_stxf)
    auth = ProspectusAuthorityResolver.resolve_authority(candidates, ident_stxf)
    assert auth is None, "STXF must remain fail-closed in Lane D"
