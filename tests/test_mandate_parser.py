"""Unit tests for Deterministic Statutory Mandate Parser under Policy v1.1.0."""

import pytest
from scripts.research.mandate_parser import (
    DeterministicMandateParser,
    MandateParseResult,
    APPROVED_SECTORS
)


def test_broad_equity_index_parsing():
    text = (
        "Principal Investment Strategies: The Fund seeks to track the investment results of the "
        "S&P 500 Index, which measures the performance of the large-capitalization sector of the U.S. "
        "equity market. The Fund invests at least 80% of its assets in the component securities of the Index."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000001")
    assert res.broad_or_multi_sector_mandate is True
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.government_debt_mandate is False
    assert res.corporate_credit_mandate is False
    assert res.non_confirmatory_mandate is False
    assert res.parser_rule_id == "RULE_BROAD_EQUITY_INDEX"
    assert res.confidence_state == "CONFIDENT_CONFIRMATORY"


def test_sector_specific_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund seeks to provide investment results that correspond "
        "generally to the price and yield performance of the Technology Select Sector Index. "
        "The Index includes companies from the following industries: technology hardware, storage & peripherals, "
        "software, semiconductors & semiconductor equipment, and IT services."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000002")
    assert res.sector_specific_mandate is True
    assert res.approved_sector == "TECHNOLOGY"
    assert res.broad_or_multi_sector_mandate is False
    assert res.government_debt_mandate is False
    assert res.corporate_credit_mandate is False
    assert res.non_confirmatory_mandate is False
    assert res.parser_rule_id == "RULE_SECTOR_EQUITY"
    assert res.confidence_state == "CONFIDENT_CONFIRMATORY"


def test_treasury_government_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund seeks to track the investment results of the "
        "ICE U.S. Treasury 7-10 Year Bond Index. The Fund invests at least 80% of its total assets "
        "in U.S. Treasury obligations and securities that are backed by the full faith and credit "
        "of the U.S. government."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000003")
    assert res.government_debt_mandate is True
    assert res.broad_or_multi_sector_mandate is False
    assert res.sector_specific_mandate is False
    assert res.corporate_credit_mandate is False
    assert res.non_confirmatory_mandate is False
    assert res.parser_rule_id == "RULE_TREASURY_GOVERNMENT"
    assert res.confidence_state == "CONFIDENT_CONFIRMATORY"


def test_corporate_credit_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund seeks to track the investment results of the "
        "Markit iBoxx USD Liquid High Yield Index, which is composed of U.S. dollar-denominated, "
        "high yield corporate bonds. The Fund invests at least 80% of its assets in corporate bond securities."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000004")
    assert res.corporate_credit_mandate is True
    assert res.government_debt_mandate is False
    assert res.broad_or_multi_sector_mandate is False
    assert res.sector_specific_mandate is False
    assert res.non_confirmatory_mandate is False
    assert res.parser_rule_id == "RULE_CORPORATE_CREDIT"
    assert res.confidence_state == "CONFIDENT_CONFIRMATORY"


def test_non_confirmatory_active_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund is an actively managed exchange-traded fund that "
        "seeks capital appreciation by utilizing dynamic equity factor rotation and opportunistic asset allocation."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000005")
    assert res.non_confirmatory_mandate is True
    assert res.broad_or_multi_sector_mandate is False
    assert res.sector_specific_mandate is False
    assert res.parser_rule_id == "RULE_NON_CONFIRMATORY"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_non_confirmatory_option_overlay_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund invests in equity securities of the S&P 500 Index "
        "and writes covered call options on the index to generate incremental option overlay income."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000006")
    assert res.non_confirmatory_mandate is True
    assert res.broad_or_multi_sector_mandate is False
    assert res.parser_rule_id == "RULE_NON_CONFIRMATORY"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_mixed_aggregate_bond_mandate_parsing():
    text = (
        "Principal Investment Strategies: The Fund tracks the Bloomberg U.S. Aggregate Bond Index, "
        "investing across U.S. Treasury securities and investment grade corporate bonds as well as mortgage-backed debt."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000007")
    assert res.non_confirmatory_mandate is True
    assert res.government_debt_mandate is False
    assert res.corporate_credit_mandate is False
    assert res.parser_rule_id == "RULE_MIXED_AGGREGATE_BOND"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_empty_or_too_short_text_fails_closed():
    res = DeterministicMandateParser.parse_mandate("Short strategy text.", accession="0001193125-26-000008")
    assert res.confidence_state == "PARSE_FAILURE_EMPTY_SECTION"
    assert res.broad_or_multi_sector_mandate is False
    assert res.non_confirmatory_mandate is False


def test_unclassified_mandate_fails_closed():
    text = (
        "Principal Investment Strategies: The Fund invests in proprietary structured notes and bespoke "
        "synthetic swap baskets to provide synthetic exposure across miscellaneous asset categories."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-000009")
    assert res.confidence_state == "AMBIGUOUS_UNCLASSIFIED"
    assert res.parser_rule_id == "RULE_FAIL_CLOSED_AMBIGUOUS"
    assert res.broad_or_multi_sector_mandate is False
    assert res.non_confirmatory_mandate is False
