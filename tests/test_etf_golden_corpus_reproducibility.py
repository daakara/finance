"""
tests/test_etf_golden_corpus_reproducibility.py

Automated test suite verifying that docs/research/ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json
is 100% reproducible from raw regulatory evidence under Policy V1.1.
"""

from scripts.research.certify_etf_golden_corpus_v1 import certify_golden_corpus


def test_etf_clean_room_golden_corpus_reproducibility():
    """Verify that all 36 targets reproduce with zero mismatches."""
    success, report = certify_golden_corpus()
    assert success is True, f"Certification failed: {report['discrepancies']}"
    assert report["targets"] == 36
    assert report["field_mismatches"] == 0
    assert report["classification_mismatches"] == 0
    assert report["raw_source_missing"] == 0
    assert report["temporal_leakage"] == 0
    assert report["ncen_mapping_errors"] == 0
    assert report["nport_cross_contamination"] == 0
