"""Unit tests and Golden Adversarial Fixtures for Deterministic Statutory Mandate Parser (Policy v1.1.0).

Validates MANDATE_PARSER_V1_2_0_FROZEN against:
1. Capital Group funds (CGBL, CGUS, CGIC) not triggering false positive Financials sector.
2. VEA (FTSE Developed All Cap ex US) rejected by Geography Invariant.
3. ONEQ (Nasdaq Composite Index) recognized as broad equity index despite passive/active contrast language.
4. BITX non-confirmatory crypto/leveraged recognition.
5. Incidental sector vocabulary rejection across all 11 approved sectors.
6. Affirmative sector mandate recognition across all 11 approved sectors.
7. Pure Treasury, Corporate Credit, Mixed Aggregate Bond, and Active Management rules.
"""

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
        "The Fund invests at least 80% of its total assets in the securities of companies in the Technology Select Sector Index."
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
    assert res.parser_rule_id == "RULE_ACTIVE_MANAGEMENT"
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


# --------------------------------------------------------------------------
# GOLDEN ADVERSARIAL FIXTURES
# --------------------------------------------------------------------------

def test_adversarial_fixture_cgbl_balanced_fof():
    """CGBL (Capital Group Core Balanced ETF) must NOT classify as Financials sector."""
    text = (
        "Principal investment strategies: In seeking to pursue its investment objective, the fund varies its "
        "mix of direct or indirect exposure to equity securities, debt securities and money market instruments. "
        "Under normal market conditions, the fund's investment adviser will maintain the following investment mix: "
        "50%-75% in equity securities, at least 25% in debt securities, and the remainder in money market instruments. "
        "In addition, the fund will achieve its allocation to debt securities through investing in one or more fixed "
        "income exchange-traded funds (ETFs) managed and advised by the fund's investment adviser. Financial condition "
        "of issuers and financial markets volatility are considered by the adviser."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0000051931-26-000262")
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.broad_or_multi_sector_mandate is False
    assert res.parser_rule_id == "RULE_FUND_OF_FUNDS_OR_BALANCED"
    assert res.confidence_state == "UNRESOLVED_POLICY_EXECUTABILITY"


def test_adversarial_fixture_cgus_not_financials_sector():
    """CGUS (Capital Group Core Equity ETF) must NOT classify as Financials sector."""
    text = (
        "Principal investment strategies: Capital Group Core Equity ETF is an actively managed exchange-traded fund "
        "that seeks capital appreciation. The fund invests primarily in common stocks of U.S. companies. "
        "The adviser evaluates financial statements and the financial condition of companies to identify growth opportunities."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0000051931-26-000747")
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.broad_or_multi_sector_mandate is False
    assert res.non_confirmatory_mandate is True
    assert res.parser_rule_id == "RULE_ACTIVE_MANAGEMENT"


def test_adversarial_fixture_cgic_international_ex_us():
    """CGIC (Capital Group International Core Equity ETF) must reject US equity confirmatory status."""
    text = (
        "Principal investment strategies: Capital Group International Core Equity ETF is an actively managed "
        "fund that invests primarily in common stocks of companies domiciled outside the United States, including "
        "developed and emerging markets outside the U.S. The adviser reviews financial metrics of foreign companies."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0000051931-26-000748")
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.broad_or_multi_sector_mandate is False
    assert res.non_confirmatory_mandate is True
    assert res.parser_rule_id == "RULE_EX_US_OR_INTERNATIONAL"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_adversarial_fixture_vea_geography_exclusion():
    """VEA (Vanguard FTSE Developed Markets ETF) tracks ex-US index and must reject US broad equity."""
    text = (
        "Principal Investment Strategies: The Fund employs an indexing investment approach designed to track the "
        "performance of the FTSE Developed All Cap ex US Index, a market-capitalization-weighted index that is "
        "composed of approximately 4,000 common stocks of large-, mid-, and small-cap companies located in "
        "developed markets outside the United States, including Canada and countries in Europe and the Pacific region."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001193125-26-392561")
    assert res.broad_or_multi_sector_mandate is False
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.non_confirmatory_mandate is True
    assert res.parser_rule_id == "RULE_EX_US_OR_INTERNATIONAL"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_adversarial_fixture_oneq_nasdaq_composite_broad():
    """ONEQ (Fidelity Nasdaq Composite Index ETF) must recognize broad index and ignore passive contrast language."""
    text = (
        "Principal Investment Strategies: Using statistical sampling techniques to create a portfolio of "
        "securities listed in the index that have a similar investment profile to the entire index. "
        "Normally investing at least 80% of assets in securities included in the Nasdaq Composite Index. "
        "The Nasdaq Composite Index is a widely recognized, market capitalization-weighted index that is "
        "designed to represent the performance of Nasdaq securities and includes approximately 3,000 stocks. "
        "This differs from an actively managed fund, which typically seeks to outperform. "
        "Technology Industry Concentration risk disclosure describes potential risks."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0000205323-26-000014")
    assert res.broad_or_multi_sector_mandate is True
    assert res.sector_specific_mandate is False
    assert res.approved_sector is None
    assert res.non_confirmatory_mandate is False
    assert res.parser_rule_id == "RULE_BROAD_EQUITY_INDEX"
    assert res.confidence_state == "CONFIDENT_CONFIRMATORY"


def test_adversarial_fixture_bitx_leveraged_crypto():
    """BITX (2x Bitcoin Strategy ETF) must be recognized as non-confirmatory crypto/leveraged."""
    text = (
        "Principal Investment Strategies: The Fund seeks daily investment results, before fees and expenses, "
        "that correspond to 2x the daily performance of the CME Bitcoin Futures Index. The Fund does not "
        "invest directly in bitcoin."
    )
    res = DeterministicMandateParser.parse_mandate(text, accession="0001213900-26-101774")
    assert res.non_confirmatory_mandate is True
    assert res.broad_or_multi_sector_mandate is False
    assert res.sector_specific_mandate is False
    assert res.parser_rule_id == "RULE_NON_CONFIRMATORY"
    assert res.confidence_state == "CONFIDENT_NON_CONFIRMATORY"


def test_incidental_sector_words_rejected():
    """Incidental use of sector words in non-mandate context must not trigger sector classifications."""
    snippets = [
        "Adverse economic developments could impair the financial condition and financial performance of issuers.",
        "The portfolio manager evaluates the financial statements of prospective investments across all industries.",
        "Companies in the portfolio face increased energy costs and volatile electricity utility pricing.",
        "The adviser uses advanced proprietary technology systems to execute portfolio rebalancing trades.",
        "Supply chain disruptions have led to raw materials shortages and delayed industrial manufacturing.",
        "Rising health care and health insurance costs represent an ongoing operational expenditure.",
        "Consumer discretionary spending may decline due to macroeconomic inflation and retail slowdown.",
    ]
    for snippet in snippets:
        text = f"Principal Investment Strategies: The Fund invests broadly across diverse companies. {snippet}"
        res = DeterministicMandateParser.parse_mandate(text, accession="0000000000-26-000000")
        assert res.sector_specific_mandate is False, f"Falsely matched sector on snippet: {snippet}"
        assert res.approved_sector is None


def test_all_11_approved_sectors_affirmative():
    """Affirmative mandate phrasing must correctly trigger each of the 11 approved sectors."""
    sector_fixtures = {
        "TECHNOLOGY": "The Fund invests at least 80% of its net assets in securities of the Technology Select Sector Index.",
        "FINANCIALS": "The Fund seeks to track the performance of the Financial Select Sector Index, investing in banking and insurance.",
        "ENERGY": "Under normal conditions, the Fund invests at least 80% of its assets in securities of the Energy Select Sector Index.",
        "HEALTHCARE": "The Fund invests primarily in securities of companies in the Health Care Select Sector Index.",
        "INDUSTRIALS": "The Fund seeks to track the investment results of the Industrials Select Sector Index.",
        "MATERIALS": "The Fund invests at least 80% of its total assets in component securities of the Materials Select Sector Index.",
        "CONSUMER_STAPLES": "The Fund tracks the price and yield of the Consumer Staples Select Sector Index.",
        "CONSUMER_DISCRETIONARY": "The Fund invests at least 80% of its assets in securities of the Consumer Discretionary Select Sector Index.",
        "UTILITIES": "The Fund invests primarily in equity securities of companies comprising the Utilities Select Sector Index.",
        "REAL_ESTATE": "The Fund seeks to track the performance of the Real Estate Select Sector Index.",
        "COMMUNICATION_SERVICES": "The Fund invests at least 80% of its net assets in the Communication Services Select Sector Index.",
    }
    for sector, fixture in sector_fixtures.items():
        text = f"Principal Investment Strategies: {fixture}"
        res = DeterministicMandateParser.parse_mandate(text, accession="0000000000-26-000001")
        assert res.sector_specific_mandate is True, f"Failed for sector {sector}"
        assert res.approved_sector == sector, f"Expected {sector}, got {res.approved_sector}"
        assert res.parser_rule_id == "RULE_SECTOR_EQUITY"
        assert res.confidence_state == "CONFIDENT_CONFIRMATORY"
