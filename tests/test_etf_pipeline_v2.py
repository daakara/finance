"""
tests/test_etf_pipeline_v2.py

Comprehensive acceptance and regression test suite for ARX Terminal ETF Pipeline V2.
Verifies:
1. 36/36 exact replication of docs/research/ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json.
2. Adversarial controls: SEPQ, TUG, FBCG, PFFD, VGT, VCR, IWF, IWV, IWO, EQRR, XB, XCCC.
3. Zero target-specific classification logic.
4. Deterministic re-execution.
"""

import json
from pathlib import Path
import pytest

from scripts.research.etf_v2.models import EntityIdentity
from scripts.research.etf_v2.pipeline import ETFPipelineV2
from scripts.research.etf_v2.identity_authority import IdentityAuthority

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_CORPUS_PATH = REPO_ROOT / "docs" / "research" / "ETF_CLEAN_ROOM_GOLDEN_CORPUS_V1.json"


@pytest.fixture(scope="module")
def golden_records():
    with open(GOLDEN_CORPUS_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["records"]


@pytest.fixture(scope="module")
def pipeline():
    return ETFPipelineV2(REPO_ROOT)


def test_v2_golden_corpus_reproduction(golden_records, pipeline):
    """Asserts that Pipeline V2 achieves 36/36 match on the certified clean-room golden corpus."""
    assert len(golden_records) == 36

    mismatches = []
    processed_count = 0

    for expected in golden_records:
        sym = expected["symbol"]
        identity = IdentityAuthority.create_identity(
            symbol=sym,
            cik=expected["cik"],
            series_id=expected["series_id"],
            class_id=expected["class_id"],
            legal_name=expected["legal_name"],
        )

        record = pipeline.process_target(identity)
        assert record is not None, f"Pipeline V2 returned None for {sym}"
        processed_count += 1

        # 1. Authority Match
        if record.prospectus_accession != expected["prospectus_accession"]:
            mismatches.append(f"{sym}: Prospectus accession {record.prospectus_accession} != {expected['prospectus_accession']}")

        # 2. Metric Matches
        if abs(record.total_equity_pct - expected["total_equity_pct"]) > 1e-4:
            mismatches.append(f"{sym}: total_equity_pct {record.total_equity_pct} != {expected['total_equity_pct']}")
        if abs(record.total_govt_pct - expected["total_govt_pct"]) > 1e-4:
            mismatches.append(f"{sym}: total_govt_pct {record.total_govt_pct} != {expected['total_govt_pct']}")
        if abs(record.corporate_debt_pct - expected["corporate_debt_pct"]) > 1e-4:
            mismatches.append(f"{sym}: corporate_debt_pct {record.corporate_debt_pct} != {expected['corporate_debt_pct']}")
        if record.distinct_holdings_count != expected["distinct_holdings_count"]:
            mismatches.append(f"{sym}: distinct_holdings_count {record.distinct_holdings_count} != {expected['distinct_holdings_count']}")
        if abs(record.max_security_concentration - expected["max_security_concentration"]) > 1e-4:
            mismatches.append(f"{sym}: max_security_concentration {record.max_security_concentration} != {expected['max_security_concentration']}")

        # 3. N-CEN Flag Match
        if record.is_index_fund != expected["is_index_fund"]:
            mismatches.append(f"{sym}: is_index_fund {record.is_index_fund} != {expected['is_index_fund']}")

        # 4. Classification Match
        if record.final_classification != expected["final_classification"]:
            mismatches.append(f"{sym}: Classification {record.final_classification} != {expected['final_classification']}")

    assert processed_count == 36
    assert len(mismatches) == 0, f"Discrepancies found: {mismatches}"


def test_v2_adversarial_controls(pipeline):
    """Explicitly verifies known adversarial inspection controls."""
    with open(GOLDEN_CORPUS_PATH, "r", encoding="utf-8") as f:
        golden_by_sym = {r["symbol"]: r for r in json.load(f)["records"]}

    controls = [
        ("SEPQ", "NON_CONFIRMATORY"),
        ("TUG", "NON_CONFIRMATORY"),
        ("FBCG", "NON_CONFIRMATORY"),
        ("PFFD", "NON_CONFIRMATORY"),
        ("VGT", "CONFIRMATORY_EQUITY_SECTOR"),
        ("VCR", "CONFIRMATORY_EQUITY_SECTOR"),
        ("IWF", "CONFIRMATORY_EQUITY_INDEX"),
        ("IWV", "CONFIRMATORY_EQUITY_INDEX"),
        ("IWO", "CONFIRMATORY_EQUITY_INDEX"),
        ("EQRR", "CONFIRMATORY_EQUITY_INDEX"),
        ("XB", "CONFIRMATORY_FIXED_INCOME_CREDIT"),
        ("XCCC", "CONFIRMATORY_FIXED_INCOME_CREDIT"),
    ]

    for sym, expected_cls in controls:
        gold = golden_by_sym[sym]
        identity = IdentityAuthority.create_identity(
            symbol=sym,
            cik=gold["cik"],
            series_id=gold["series_id"],
            class_id=gold["class_id"],
            legal_name=gold["legal_name"],
        )
        rec = pipeline.process_target(identity)
        assert rec is not None
        assert rec.final_classification == expected_cls, f"Control failure on {sym}: expected {expected_cls}, got {rec.final_classification}"


def test_v2_deterministic_reexecution(golden_records, pipeline):
    """Verifies that sequential runs produce bit-for-bit identical outputs."""
    sample = golden_records[:5]
    for gold in sample:
        identity = IdentityAuthority.create_identity(
            symbol=gold["symbol"],
            cik=gold["cik"],
            series_id=gold["series_id"],
            class_id=gold["class_id"],
            legal_name=gold["legal_name"],
        )
        run1 = pipeline.process_target(identity)
        run2 = pipeline.process_target(identity)
        assert run1.to_dict() == run2.to_dict()


def test_v2_no_target_specific_logic():
    """Scans all V2 codebase to ensure zero symbol/series hardcoded branching."""
    v2_dir = REPO_ROOT / "scripts" / "research" / "etf_v2"
    disallowed_keywords = ["if symbol ==", "if sid ==", "if series_id ==", "if ticker =="]

    for py_file in v2_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for kw in disallowed_keywords:
            assert kw not in content, f"Found prohibited target-specific branch '{kw}' in {py_file.name}"
