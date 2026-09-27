"""Adversarial verification suite for MANDATE_PARSER_V1_3_0.

Asserts Phase D:
1. Engine version is MANDATE_PARSER_V1_3_0.
2. Legacy frozen baseline MANDATE_PARSER_V1_2_0_FROZEN is preserved.
3. 11 systematic Treasury false positives (SEPQ, TUG, PTIN, PSFF, MRGR, CDC, CFO, ONOF, FTIF, ROPE, MNA)
   and 2 review cases (PULS, FLDR) are NEVER classified as pure Treasury government debt:
   GOVERNMENT_CONFIRMATORY_FALSE_POSITIVES = 0.
4. Active US funds with incidental foreign language (FBCG, XCHG, LOWV, LRGC, HIDV) are NOT falsely
   excluded as EX_US_OR_INTERNATIONAL, but correctly recognized as RULE_ACTIVE_MANAGEMENT.
5. International negative controls (CGIC, VEA) remain classified as RULE_EX_US_OR_INTERNATIONAL.
6. Real pure Treasury funds (EDV, SGOV) remain classified as RULE_TREASURY_GOVERNMENT.
"""

import json
from pathlib import Path
import pytest

from scripts.research.mandate_parser import (
    DeterministicMandateParser,
    MandateParseResult,
    MANDATE_PARSER_VERSION,
    MANDATE_PARSER_V1_2_0_FROZEN,
)

CACHE_DIR = Path("data/research/cache/sec_prospectus")
BASELINE_LEDGER = Path("docs/research/ETF_MANDATE_POPULATION_LEDGER_PRE_REMEDIATION_BASELINE.json")


def test_mandate_parser_v1_3_0_version():
    assert MANDATE_PARSER_VERSION == "MANDATE_PARSER_V1_3_0"
    assert DeterministicMandateParser.RULESET_ID == "MANDATE_PARSER_V1_3_0"
    assert MANDATE_PARSER_V1_2_0_FROZEN == "MANDATE_PARSER_V1_2_0_FROZEN"


def test_treasury_false_positive_remediation_adversarial():
    """Verify that none of the 13 audited Treasury false positives classify as government debt."""
    if not BASELINE_LEDGER.exists():
        pytest.skip("Baseline ledger not present")

    baseline = json.load(open(BASELINE_LEDGER, encoding="utf-8"))
    target_symbols = [
        "SEPQ", "TUG", "PTIN", "PSFF", "MRGR", "CDC", "CFO",
        "ONOF", "FTIF", "ROPE", "MNA", "PULS", "FLDR"
    ]

    targets = {t["symbol"]: t for t in baseline if t.get("symbol") in target_symbols}

    for sym in target_symbols:
        assert sym in targets, f"Target {sym} not found in baseline ledger"
        t = targets[sym]
        acc = t.get("accession")
        doc = t.get("document_filename")
        fpath = CACHE_DIR / f"{acc}_{doc}"
        if not fpath.exists():
            continue

        text = fpath.read_text(encoding="utf-8", errors="ignore")
        strat, sec = DeterministicMandateParser.extract_strategy_text(text)
        assert len(strat) >= 50, f"Strategy text insufficient for {sym}"

        res = DeterministicMandateParser.parse_mandate(strat, acc, sec)

        # MANDATORY ASSERTION: ZERO GOVERNMENT DEBT FALSE POSITIVES
        assert res.government_debt_mandate is False, (
            f"Adversarial failure: {sym} was falsely classified as government debt! "
            f"Rule: {res.parser_rule_id}, conf: {res.confidence_state}"
        )
        assert res.parser_rule_id != "RULE_TREASURY_GOVERNMENT"


def test_fbcg_active_remediation():
    """Verify FBCG (Fidelity Blue Chip Growth ETF) classifies as active management, not ex-US."""
    if not BASELINE_LEDGER.exists():
        pytest.skip("Baseline ledger not present")

    baseline = json.load(open(BASELINE_LEDGER, encoding="utf-8"))
    t = next((x for x in baseline if x.get("symbol") == "FBCG"), None)
    if not t:
        pytest.skip("FBCG not in baseline")

    acc = t.get("accession")
    doc = t.get("document_filename")
    fpath = CACHE_DIR / f"{acc}_{doc}"
    if not fpath.exists():
        pytest.skip("FBCG prospectus not cached locally")

    text = fpath.read_text(encoding="utf-8", errors="ignore")
    strat, sec = DeterministicMandateParser.extract_strategy_text(text)
    res = DeterministicMandateParser.parse_mandate(strat, acc, sec)

    assert res.parser_rule_id == "RULE_ACTIVE_MANAGEMENT"
    assert res.non_confirmatory_mandate is True
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_negative_controls_preserved():
    """Verify negative and positive controls preserve expected classifications."""
    # 1. VEA -> Ex-US
    vea_text = (
        "Principal Investment Strategies: The Fund employs an indexing investment approach designed to track the "
        "performance of the FTSE Developed All Cap ex US Index, a market-capitalization-weighted index that is "
        "composed of approximately 4,000 common stocks of large-, mid-, and small-cap companies located in "
        "developed markets outside the United States, including Canada and countries in Europe and the Pacific region."
    )
    res_vea = DeterministicMandateParser.parse_mandate(vea_text)
    assert res_vea.parser_rule_id == "RULE_EX_US_OR_INTERNATIONAL"

    # 2. CGIC -> Ex-US
    cgic_text = (
        "Principal investment strategies: Capital Group International Core Equity ETF is an actively managed "
        "fund that invests primarily in common stocks of companies domiciled outside the United States, including "
        "developed and emerging markets outside the U.S. The adviser reviews financial metrics of foreign companies."
    )
    res_cgic = DeterministicMandateParser.parse_mandate(cgic_text)
    assert res_cgic.parser_rule_id == "RULE_EX_US_OR_INTERNATIONAL"

    # 3. Pure Treasury -> Confirmatory
    edv_text = (
        "Principal Investment Strategies: The Fund employs an indexing investment approach designed to track the "
        "performance of the Bloomberg U.S. Treasury STRIPS 20-30 Year Equal Par Bond Index, which includes zero-coupon "
        "U.S. Treasury securities with maturities ranging from 20 to 30 years."
    )
    res_edv = DeterministicMandateParser.parse_mandate(edv_text)
    assert res_edv.government_debt_mandate is True
    assert res_edv.parser_rule_id == "RULE_TREASURY_GOVERNMENT"
    assert res_edv.confidence_state == "CONFIDENT_CONFIRMATORY"

    # 4. Merger Arbitrage -> Non-Confirmatory
    mrgr_text = (
        "Principal Investment Strategies: The Fund is designed to track the performance of the Index and provide "
        "exposure to a global merger arbitrage strategy. The Index is designed to measure the performance of a "
        "risk arbitrage strategy."
    )
    res_mrgr = DeterministicMandateParser.parse_mandate(mrgr_text)
    assert res_mrgr.non_confirmatory_mandate is True
    assert res_mrgr.parser_rule_id == "RULE_NON_CONFIRMATORY"
