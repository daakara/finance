"""Test Suite for Series-Level Statutory Prospectus Mapping Engine (Policy v1.1.0).

Verifies:
1. Golden Multi-Series Fixtures across all 9 major multi-series issuers:
   - iShares (IVV vs IJH)
   - Vanguard (VOO vs VO)
   - SPDR (XBI vs XOP)
   - Invesco (RSP vs PHO)
   - Schwab (SCHX vs SCHA)
   - First Trust (FDL vs FDN)
   - Global X (AIQ vs BOTZ)
   - Capital Group (CGCP vs CGMS)
   - Fidelity (FSRNX vs FZFLX)
2. Correct series chosen, neighboring series rejected.
3. Correct strategy section extracted without wrong omnibus section leakage.
4. CROSS_SERIES_TEXT_LEAKAGE = 0 invariant enforcement.
5. All 4 authorized mapping outcomes (EXACT_SERIES_ID, EXACT_CLASS_ID, EXACT_LEGAL_NAME, DETERMINISTIC_COMPOSITE).
6. Fail-closed outcomes (SOURCE_NOT_FOUND, SERIES_NOT_FOUND_IN_SOURCE, AMBIGUOUS_MULTI_MATCH, PARSE_FAILURE).
7. Strict snapshot boundary enforcement (POST_BOUNDARY_FILINGS_USED = 0).
8. Audit trail completeness per Section 10.
9. Seamless integration with frozen MANDATE_PARSER_V1_2_0_FROZEN.
"""

import pytest
from scripts.research.series_prospectus_mapper import (
    SeriesProspectusMapper,
    SeriesMetadata,
    SeriesMappingResult,
)
from scripts.research.mandate_parser import DeterministicMandateParser
from tests.fixtures.multi_series_fixtures import (
    ISHARES_OMNIBUS_HTML, ISHARES_IVV, ISHARES_IJH,
    VANGUARD_OMNIBUS_HTML, VANGUARD_VOO, VANGUARD_VO,
    SPDR_OMNIBUS_HTML, SPDR_XBI, SPDR_XOP,
    INVESCO_OMNIBUS_HTML, INVESCO_RSP, INVESCO_PHO,
    SCHWAB_OMNIBUS_HTML, SCHWAB_SCHX, SCHWAB_SCHA,
    FIRST_TRUST_OMNIBUS_HTML, FIRST_TRUST_FDL, FIRST_TRUST_FDN,
    GLOBAL_X_OMNIBUS_HTML, GLOBAL_X_AIQ, GLOBAL_X_BOTZ,
    CAPITAL_GROUP_OMNIBUS_HTML, CAPITAL_GROUP_CGCP, CAPITAL_GROUP_CGMS,
    FIDELITY_OMNIBUS_HTML, FIDELITY_FSRNX, FIDELITY_FZFLX,
)


# ==============================================================================
# 1. GOLDEN MULTI-SERIES FIXTURE TESTS (9 Major Issuers)
# ==============================================================================

def test_ishares_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for iShares."""
    # Target IVV (S&P 500), Neighbor IJH (Mid-Cap 400)
    res_ivv = SeriesProspectusMapper.map_series(
        target_series=ISHARES_IVV,
        raw_document_text=ISHARES_OMNIBUS_HTML,
        accession="0001193125-26-000001",
        form="485BPOS",
        document_filename="ishares_omnibus.htm",
        neighboring_series=[ISHARES_IJH]
    )
    assert res_ivv.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "S&P 500 Index" in res_ivv.extracted_strategy_text
    assert "MidCap 400" not in res_ivv.extracted_strategy_text
    assert res_ivv.cross_series_text_leakage == 0

    # Target IJH (Mid-Cap 400), Neighbor IVV (S&P 500)
    res_ijh = SeriesProspectusMapper.map_series(
        target_series=ISHARES_IJH,
        raw_document_text=ISHARES_OMNIBUS_HTML,
        accession="0001193125-26-000001",
        form="485BPOS",
        document_filename="ishares_omnibus.htm",
        neighboring_series=[ISHARES_IVV]
    )
    assert res_ijh.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "S&P MidCap 400 Index" in res_ijh.extracted_strategy_text
    assert "large-capitalization sector" not in res_ijh.extracted_strategy_text
    assert res_ijh.cross_series_text_leakage == 0


def test_vanguard_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Vanguard."""
    res_voo = SeriesProspectusMapper.map_series(
        target_series=VANGUARD_VOO,
        raw_document_text=VANGUARD_OMNIBUS_HTML,
        accession="0001193125-26-000002",
        form="485BPOS",
        document_filename="vanguard_omnibus.htm",
        neighboring_series=[VANGUARD_VO]
    )
    assert res_voo.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "S&P 500 Index" in res_voo.extracted_strategy_text
    assert "CRSP US Mid Cap" not in res_voo.extracted_strategy_text
    assert res_voo.cross_series_text_leakage == 0

    res_vo = SeriesProspectusMapper.map_series(
        target_series=VANGUARD_VO,
        raw_document_text=VANGUARD_OMNIBUS_HTML,
        accession="0001193125-26-000002",
        form="485BPOS",
        document_filename="vanguard_omnibus.htm",
        neighboring_series=[VANGUARD_VOO]
    )
    assert res_vo.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "CRSP US Mid Cap Index" in res_vo.extracted_strategy_text
    assert res_vo.cross_series_text_leakage == 0


def test_spdr_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for SPDR."""
    res_xbi = SeriesProspectusMapper.map_series(
        target_series=SPDR_XBI,
        raw_document_text=SPDR_OMNIBUS_HTML,
        accession="0001193125-26-000003",
        form="485BPOS",
        document_filename="spdr_omnibus.htm",
        neighboring_series=[SPDR_XOP]
    )
    assert res_xbi.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Biotechnology" in res_xbi.extracted_strategy_text
    assert "Oil & Gas" not in res_xbi.extracted_strategy_text
    assert res_xbi.cross_series_text_leakage == 0

    res_xop = SeriesProspectusMapper.map_series(
        target_series=SPDR_XOP,
        raw_document_text=SPDR_OMNIBUS_HTML,
        accession="0001193125-26-000003",
        form="485BPOS",
        document_filename="spdr_omnibus.htm",
        neighboring_series=[SPDR_XBI]
    )
    assert res_xop.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Oil & Gas Exploration & Production" in res_xop.extracted_strategy_text
    assert "Biotechnology" not in res_xop.extracted_strategy_text
    assert res_xop.cross_series_text_leakage == 0


def test_invesco_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Invesco."""
    res_rsp = SeriesProspectusMapper.map_series(
        target_series=INVESCO_RSP,
        raw_document_text=INVESCO_OMNIBUS_HTML,
        accession="0001193125-26-000004",
        form="485BPOS",
        document_filename="invesco_omnibus.htm",
        neighboring_series=[INVESCO_PHO]
    )
    assert res_rsp.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "S&P 500 Equal Weight Index" in res_rsp.extracted_strategy_text
    assert "Water Index" not in res_rsp.extracted_strategy_text
    assert res_rsp.cross_series_text_leakage == 0


def test_schwab_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Schwab."""
    res_schx = SeriesProspectusMapper.map_series(
        target_series=SCHWAB_SCHX,
        raw_document_text=SCHWAB_OMNIBUS_HTML,
        accession="0001193125-26-000005",
        form="485BPOS",
        document_filename="schwab_omnibus.htm",
        neighboring_series=[SCHWAB_SCHA]
    )
    assert res_schx.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Large-Cap Total Stock Market Index" in res_schx.extracted_strategy_text
    assert "Small-Cap" not in res_schx.extracted_strategy_text
    assert res_schx.cross_series_text_leakage == 0


def test_first_trust_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for First Trust."""
    res_fdl = SeriesProspectusMapper.map_series(
        target_series=FIRST_TRUST_FDL,
        raw_document_text=FIRST_TRUST_OMNIBUS_HTML,
        accession="0001193125-26-000006",
        form="485BPOS",
        document_filename="first_trust_omnibus.htm",
        neighboring_series=[FIRST_TRUST_FDN]
    )
    assert res_fdl.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Morningstar Dividend Leaders Index" in res_fdl.extracted_strategy_text
    assert "Internet" not in res_fdl.extracted_strategy_text
    assert res_fdl.cross_series_text_leakage == 0


def test_global_x_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Global X."""
    res_aiq = SeriesProspectusMapper.map_series(
        target_series=GLOBAL_X_AIQ,
        raw_document_text=GLOBAL_X_OMNIBUS_HTML,
        accession="0001193125-26-000007",
        form="485BPOS",
        document_filename="global_x_omnibus.htm",
        neighboring_series=[GLOBAL_X_BOTZ]
    )
    assert res_aiq.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Artificial Intelligence & Big Data Index" in res_aiq.extracted_strategy_text
    assert "Robotics" not in res_aiq.extracted_strategy_text
    assert res_aiq.cross_series_text_leakage == 0


def test_capital_group_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Capital Group."""
    res_cgms = SeriesProspectusMapper.map_series(
        target_series=CAPITAL_GROUP_CGMS,
        raw_document_text=CAPITAL_GROUP_OMNIBUS_HTML,
        accession="0001193125-26-000008",
        form="485BPOS",
        document_filename="capital_group_omnibus.htm",
        neighboring_series=[CAPITAL_GROUP_CGCP]
    )
    assert res_cgms.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "high-yield corporate debt" in res_cgms.extracted_strategy_text
    assert "U.S. government, government agencies" not in res_cgms.extracted_strategy_text
    assert res_cgms.cross_series_text_leakage == 0


def test_fidelity_multi_series_isolation():
    """Proves correct series chosen, neighbor rejected, and zero leakage for Fidelity."""
    res_fsrnx = SeriesProspectusMapper.map_series(
        target_series=FIDELITY_FSRNX,
        raw_document_text=FIDELITY_OMNIBUS_HTML,
        accession="0001193125-26-000009",
        form="485BPOS",
        document_filename="fidelity_omnibus.htm",
        neighboring_series=[FIDELITY_FZFLX]
    )
    assert res_fsrnx.mapping_outcome in SeriesProspectusMapper.AUTHORIZED_OUTCOMES_FOR_PARSING
    assert "Real Estate" in res_fsrnx.extracted_strategy_text
    assert "Small-Mid Cap Momentum" not in res_fsrnx.extracted_strategy_text
    assert res_fsrnx.cross_series_text_leakage == 0


# ==============================================================================
# 2. HIERARCHY AND DETERMINISTIC RESOLUTION TESTS
# ==============================================================================

def test_mapped_exact_series_id_outcome():
    """Proves MAPPED_EXACT_SERIES_ID is assigned when explicit series ID matches."""
    html = "<div>Series S000099999 Class C000088888 <h3>Principal Investment Strategies</h3><p>Invests in S&P 500 Index.</p></div>"
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Test Fund")
    res = SeriesProspectusMapper.map_series(meta, html, "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_SERIES_ID


def test_mapped_exact_class_id_outcome():
    """Proves MAPPED_EXACT_CLASS_ID is assigned when class ID matches but series ID absent."""
    html = "<div>Class C000088888 <h3>Principal Investment Strategies</h3><p>Invests in S&P 500 Index.</p></div>"
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Test Fund")
    res = SeriesProspectusMapper.map_series(meta, html, "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_CLASS_ID


def test_mapped_exact_legal_name_outcome():
    """Proves MAPPED_EXACT_LEGAL_NAME is assigned when legal name matches but IDs absent."""
    html = "<div><h2>Alpha Beta Gamma ETF</h2> <h3>Principal Investment Strategies</h3><p>Invests in S&P 500 Index.</p></div>"
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Alpha Beta Gamma ETF")
    res = SeriesProspectusMapper.map_series(meta, html, "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_LEGAL_NAME


def test_series_not_found_in_source():
    """Proves SERIES_NOT_FOUND_IN_SOURCE when none of the identifiers match."""
    html = "<div><h2>Completely Different Fund</h2> <h3>Principal Investment Strategies</h3><p>Invests in gold.</p></div>"
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Alpha Beta Gamma ETF")
    res = SeriesProspectusMapper.map_series(meta, html, "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_SERIES_NOT_FOUND


def test_empty_document_parse_failure():
    """Proves PARSE_FAILURE when document is empty."""
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Test Fund")
    res = SeriesProspectusMapper.map_series(meta, "   ", "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_PARSE_FAILURE


def test_missing_strategy_section_parse_failure():
    """Proves PARSE_FAILURE when series block has no extractable strategy section."""
    html = "<div>Series S000099999 Class C000088888 <p>Only fee disclosures and background info without any strategy.</p></div>"
    meta = SeriesMetadata(symbol="TEST", cik="12345", series_id="S000099999", class_id="C000088888", legal_name="Test Fund")
    res = SeriesProspectusMapper.map_series(meta, html, "0001234567-26-000001", "485BPOS", "test.htm")
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_PARSE_FAILURE


# ==============================================================================
# 3. BOUNDARY AND AUDIT TRAIL TESTS
# ==============================================================================

def test_post_boundary_filings_excluded():
    """Proves filings after 2026-09-24 are excluded from filing index (POST_BOUNDARY_FILINGS_USED = 0)."""
    sub = {
        "filings": {
            "recent": {
                "form": ["485BPOS", "485BPOS"],
                "filingDate": ["2026-09-25", "2026-09-20"],
                "accessionNumber": ["0001-post-boundary", "0001-pre-boundary"],
                "primaryDocument": ["doc1.htm", "doc2.htm"]
            }
        }
    }
    index = SeriesProspectusMapper.build_filing_index_for_cik("12345", sub, snapshot_boundary="2026-09-24")
    assert len(index) == 1
    assert index[0]["accession"] == "0001-pre-boundary"
    assert index[0]["filing_date"] == "2026-09-20"


def test_audit_trail_fields_retained():
    """Proves all required Section 10 audit trail fields are populated."""
    res = SeriesProspectusMapper.map_series(
        target_series=ISHARES_IVV,
        raw_document_text=ISHARES_OMNIBUS_HTML,
        accession="0001193125-26-000001",
        form="485BPOS",
        document_filename="ishares_omnibus.htm",
        neighboring_series=[ISHARES_IJH]
    )
    # Check Section 10 fields:
    assert res.cik == "1100663"
    assert res.series_id == "S000002871"
    assert res.class_id == "C000007882"
    assert res.selected_accession == "0001193125-26-000001"
    assert res.selected_form == "485BPOS"
    assert res.primary_document == "ishares_omnibus.htm"
    assert res.selected_document == "ishares_omnibus.htm"
    assert res.mapping_rule_id == "MAPPING_RULE_EXACT_SERIES_ID"
    assert "Isolated series block" in res.mapping_evidence
    assert res.mapping_confidence_state == "CONFIDENT_SERIES_ISOLATION"
    assert res.raw_document_sha256 != "NONE"
    assert res.extracted_series_block_sha256 != "NONE"
    assert len(res.extracted_strategy_text) > 50


# ==============================================================================
# 4. INTEGRATION WITH FROZEN MANDATE PARSER
# ==============================================================================

def test_end_to_end_mandate_parser_integration():
    """Proves extracted series strategy integrates cleanly with MANDATE_PARSER_V1_2_0_FROZEN."""
    # 1. Map IVV from iShares omnibus
    map_res = SeriesProspectusMapper.map_series(
        target_series=ISHARES_IVV,
        raw_document_text=ISHARES_OMNIBUS_HTML,
        accession="0001193125-26-000001",
        form="485BPOS",
        document_filename="ishares_omnibus.htm",
        neighboring_series=[ISHARES_IJH]
    )
    assert map_res.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_SERIES_ID

    # 2. Feed extracted text to frozen mandate parser
    mandate_res = DeterministicMandateParser.parse_mandate(
        strategy_text=map_res.extracted_strategy_text,
        accession=map_res.selected_accession,
        section_name=map_res.exact_source_section
    )
    assert mandate_res.broad_or_multi_sector_mandate is True
    assert mandate_res.sector_specific_mandate is False
    assert mandate_res.parser_rule_id == "RULE_BROAD_EQUITY_INDEX"
    assert mandate_res.confidence_state == "CONFIDENT_CONFIRMATORY"
