"""Adversarial verification of SERIES_RESOLVER_V1_4_0 and DOC_INDEX_V1_3_0.

Tests:
1. Version assertions.
2. Financial bounded abbreviation normalization.
3. Negative sibling isolation (FALSE_POSITIVE_NAME_MAPPING = 0, CROSS_SERIES_CONTAMINATION = 0).
4. Real corpus resolution of audited abbreviation variants (e.g. VGT).
"""

import pytest
from pathlib import Path

from scripts.research.document_index_engine import (
    DocumentIndex,
    DocumentIdentity,
    DocumentNormalizer,
    INDEX_ENGINE_VERSION,
    NORMALIZATION_VERSION,
)
from scripts.research.series_prospectus_mapper import (
    SeriesProspectusMapper,
    SeriesMetadata,
    SERIES_RESOLVER_VERSION,
)


def test_versions():
    """Pinned to the current authorized engine versions (V1.3.0 / V1.4.0 remediation gate).

    DOC_INDEX_V1_3_1: adds Windows-1252 apostrophe variant (\u0094/\u0093) to The-Fund-Investment-Goal pattern.
    SERIES_RESOLVER_V1_4_1: syncs _extract_strategy_from_block pattern with DOC_INDEX_V1_3_1 encoding fix.
    """
    assert INDEX_ENGINE_VERSION == "DOC_INDEX_V1_3_1"
    assert NORMALIZATION_VERSION == "NORMALIZATION_V1_3_1"
    assert SERIES_RESOLVER_VERSION == "SERIES_RESOLVER_V1_4_1"


def test_bounded_name_normalization():
    # Equivalence checks
    assert DocumentNormalizer.normalize_name("Vanguard Information Tech ETF") == "vanguard information technology etf"
    assert DocumentNormalizer.normalize_name("Vanguard Consumer Discretion ETF") == "vanguard consumer discretionary etf"
    assert DocumentNormalizer.normalize_name("Vanguard Div Appreciation ETF") == "vanguard dividend appreciation etf"
    assert DocumentNormalizer.normalize_name("Vanguard FTSE All-Wld ex-US SmCp Idx ETF") == "vanguard ftse all world ex us small cap index etf"
    assert DocumentNormalizer.normalize_name("PIMCO Corporate Bond Index Fund") == "pimco corporate bond index fund"
    assert DocumentNormalizer.normalize_name("PIMCO Corporat Bond Index Fund") == "pimco corporate bond index fund"


def test_negative_sibling_isolation_controls():
    """Verify that a near-name or abbreviation variant CANNOT match a sibling fund's section."""
    html_content = """
    <html>
    <body>
    <div id="fund1">
        <h2>Vanguard Large-Cap Growth Index Fund</h2>
        <span>Series S000001111 Class C000002222</span>
        <h3>Principal Investment Strategies</h3>
        <p>The fund invests in large-cap growth stocks.</p>
    </div>
    <hr />
    <div id="fund2">
        <h2>Vanguard Large-Cap Value Index Fund</h2>
        <span>Series S000003333 Class C000004444</span>
        <h3>Principal Investment Strategies</h3>
        <p>The fund invests in large-cap value stocks.</p>
    </div>
    </body>
    </html>
    """.encode("utf-8")

    identity = DocumentIdentity(
        cik="0000036405",
        accession="0000036405-26-000001",
        form="485BPOS",
        filing_date="2026-01-20",
        document_filename="test.htm",
        source_byte_length=len(html_content),
    )

    growth_target = SeriesMetadata(
        symbol="VUG",
        cik="0000036405",
        series_id="S000001111",
        class_id="C000002222",
        legal_name="Vanguard Large-Cap Growth ETF",
    )
    value_sibling = SeriesMetadata(
        symbol="VTV",
        cik="0000036405",
        series_id="S000003333",
        class_id="C000004444",
        legal_name="Vanguard Large-Cap Value ETF",
    )

    doc_index = DocumentIndex(identity, html_content, known_series_metadata=[
        {"symbol": "VUG", "legal_name": growth_target.legal_name, "series_id": growth_target.series_id, "class_id": growth_target.class_id},
        {"symbol": "VTV", "legal_name": value_sibling.legal_name, "series_id": value_sibling.series_id, "class_id": value_sibling.class_id},
    ])

    # Growth target resolution
    res_g = SeriesProspectusMapper.map_series(
        target_series=growth_target,
        document_index_or_text=doc_index,
        neighboring_series=[value_sibling],
    )
    assert res_g.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_SERIES_ID
    assert "large-cap growth stocks" in res_g.extracted_strategy_text
    assert "large-cap value stocks" not in res_g.extracted_strategy_text

    # Value target resolution
    res_v = SeriesProspectusMapper.map_series(
        target_series=value_sibling,
        document_index_or_text=doc_index,
        neighboring_series=[growth_target],
    )
    assert res_v.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_SERIES_ID
    assert "large-cap value stocks" in res_v.extracted_strategy_text
    assert "large-cap growth stocks" not in res_v.extracted_strategy_text


def test_real_corpus_vgt_resolution():
    """Verify that VGT (Vanguard Information Tech ETF) resolves accurately under DOC_INDEX_V1_2_0."""
    vgt_cache = Path("data/research/cache/sec_prospectus/0000052848-26-000651_f45474d1.htm")
    if not vgt_cache.exists():
        pytest.skip("VGT prospectus file not cached locally")

    raw_bytes = vgt_cache.read_bytes()
    identity = DocumentIdentity(
        cik="0000052848",
        accession="0000052848-26-000651",
        form="497K",
        filing_date="2026-01-28",
        document_filename="f45474d1.htm",
        source_byte_length=len(raw_bytes),
    )
    target = SeriesMetadata(
        symbol="VGT",
        cik="0000052848",
        series_id="S000004452",
        class_id="C000012227",
        legal_name="Vanguard Information Tech ETF",
    )
    doc_index = DocumentIndex(identity, raw_bytes, known_series_metadata=[
        {"symbol": "VGT", "legal_name": target.legal_name, "series_id": target.series_id, "class_id": target.class_id}
    ])

    res = SeriesProspectusMapper.map_series(target, doc_index)
    assert res.mapping_outcome == SeriesProspectusMapper.OUTCOME_EXACT_LEGAL_NAME
    assert "information technology" in res.extracted_strategy_text.lower()
