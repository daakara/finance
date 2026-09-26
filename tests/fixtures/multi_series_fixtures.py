"""Golden Multi-Series Fixture Set for ARX Terminal Series Prospectus Mapping.

Includes realistic golden omnibus prospectus fixtures for major multi-series issuers:
1. iShares (iSHARES TRUST)
2. Vanguard (VANGUARD INDEX FUNDS)
3. SPDR (SPDR SERIES TRUST)
4. Invesco (INVESCO EXCHANGE-TRADED FUND TRUST)
5. Schwab (SCHWAB STRATEGIC TRUST)
6. First Trust (FIRST TRUST EXCHANGE-TRADED FUND)
7. Global X (GLOBAL X FUNDS)
8. Capital Group (CAPITAL GROUP FIXED INCOME ETF TRUST)
9. Fidelity (FIDELITY SALEM STREET TRUST)
"""

from typing import Dict, Any, List
from scripts.research.series_prospectus_mapper import SeriesMetadata


# --------------------------------------------------------------------------
# 1. iShares Golden Fixture
# --------------------------------------------------------------------------
ISHARES_OMNIBUS_HTML = """
<html>
<head><title>iShares Trust Prospectus</title></head>
<body>
<div id="series_header">
  <span class="cik">0001100663</span>
</div>

<div class="fund-summary" id="fund_ivv">
  <h2>iShares Core S&P 500 ETF</h2>
  <p>Ticker: IVV &nbsp;|&nbsp; Series ID: S000002871 &nbsp;|&nbsp; Class ID: C000007882</p>
  <h3>Investment Objective</h3>
  <p>The iShares Core S&P 500 ETF seeks to track the investment results of an index composed of large-capitalization U.S. equities.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund seeks to track the investment results of the S&P 500 Index, which measures the performance of the large-capitalization sector of the U.S. equity market. The Fund invests at least 80% of its assets in the component securities of the S&P 500 Index.</p>
  <h3>Principal Risks</h3>
  <p>Equity Securities Risk, Market Risk, Index-Related Risk.</p>
</div>

<hr />

<div class="fund-summary" id="fund_ijh">
  <h2>iShares Core S&P Mid-Cap ETF</h2>
  <p>Ticker: IJH &nbsp;|&nbsp; Series ID: S000002872 &nbsp;|&nbsp; Class ID: C000007883</p>
  <h3>Investment Objective</h3>
  <p>The iShares Core S&P Mid-Cap ETF seeks to track the investment results of an index composed of mid-capitalization U.S. equities.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund seeks to track the investment results of the S&P MidCap 400 Index, which measures the performance of the mid-capitalization sector of the U.S. equity market. The Fund invests at least 80% of its assets in the component securities of the S&P MidCap 400 Index.</p>
  <h3>Principal Risks</h3>
  <p>Mid-Capitalization Companies Risk, Equity Securities Risk.</p>
</div>

<hr />
<div id="general_info">
  <h2>General Information about the Trust</h2>
  <p>Additional trust disclosures and statement of additional information.</p>
</div>
</body>
</html>
"""

ISHARES_IVV = SeriesMetadata(
    symbol="IVV",
    cik="1100663",
    series_id="S000002871",
    class_id="C000007882",
    legal_name="iShares Core S&P 500 ETF",
    trust_name="iSHARES TRUST"
)

ISHARES_IJH = SeriesMetadata(
    symbol="IJH",
    cik="1100663",
    series_id="S000002872",
    class_id="C000007883",
    legal_name="iShares Core S&P Mid-Cap ETF",
    trust_name="iSHARES TRUST"
)


# --------------------------------------------------------------------------
# 2. Vanguard Golden Fixture
# --------------------------------------------------------------------------
VANGUARD_OMNIBUS_HTML = """
<html>
<body>
<div class="header">Vanguard Index Funds CIK 0000036405</div>

<div class="fund-section">
  <h1>Vanguard 500 Index Fund</h1>
  <p>Series S000001001 Class C000002001 (VOO)</p>
  <h2>Investment Objective</h2>
  <p>The Fund seeks to track the performance of a benchmark index that measures the investment return of large-capitalization stocks.</p>
  <h2>Principal Investment Strategies</h2>
  <p>The Fund employs an indexing investment approach designed to track the performance of the S&P 500 Index. The Fund attempts to replicate the target index by investing all, or substantially all, of its assets in the stocks that make up the S&P 500 Index.</p>
</div>

<div class="fund-section">
  <h1>Vanguard Mid-Cap Index Fund</h1>
  <p>Series S000001002 Class C000002002 (VO)</p>
  <h2>Investment Objective</h2>
  <p>The Fund seeks to track the performance of the CRSP US Mid Cap Index.</p>
  <h2>Principal Investment Strategies</h2>
  <p>The Fund employs an indexing investment approach designed to track the performance of the CRSP US Mid Cap Index. The Fund invests by sampling the index, holding a broadly diversified collection of stocks that approximates the full index.</p>
</div>
</body>
</html>
"""

VANGUARD_VOO = SeriesMetadata(
    symbol="VOO",
    cik="36405",
    series_id="S000001001",
    class_id="C000002001",
    legal_name="Vanguard 500 Index Fund",
    trust_name="VANGUARD INDEX FUNDS"
)

VANGUARD_VO = SeriesMetadata(
    symbol="VO",
    cik="36405",
    series_id="S000001002",
    class_id="C000002002",
    legal_name="Vanguard Mid-Cap Index Fund",
    trust_name="VANGUARD INDEX FUNDS"
)


# --------------------------------------------------------------------------
# 3. SPDR Golden Fixture
# --------------------------------------------------------------------------
SPDR_OMNIBUS_HTML = """
<html>
<body>
<div class="trust">SPDR Series Trust CIK 0001064642</div>

<div class="summary">
  <h2>SPDR S&P Biotech ETF</h2>
  <p>Series ID: S000003001 | Class ID: C000003001 (Ticker: XBI)</p>
  <h3>Investment Objective</h3>
  <p>The SPDR S&P Biotech ETF seeks to provide investment results that correspond generally to the total return performance of the S&P Biotechnology Select Industry Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>In seeking to track the performance of the S&P Biotechnology Select Industry Index, the Fund employs a sampling strategy. The Fund invests substantially all, but at least 80%, of its total assets in the securities comprising the Index. The Index represents the biotechnology sub-industry portion of the S&P Total Markets Index.</p>
</div>

<div class="summary">
  <h2>SPDR S&P Oil & Gas Exploration & Production ETF</h2>
  <p>Series ID: S000003002 | Class ID: C000003002 (Ticker: XOP)</p>
  <h3>Investment Objective</h3>
  <p>The SPDR S&P Oil & Gas Exploration & Production ETF seeks to provide investment results that correspond to the S&P Oil & Gas Exploration & Production Select Industry Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund invests at least 80% of its total assets in securities of companies comprising the S&P Oil & Gas Exploration & Production Select Industry Index, representing the oil and gas exploration industry.</p>
</div>
</body>
</html>
"""

SPDR_XBI = SeriesMetadata(
    symbol="XBI",
    cik="1064642",
    series_id="S000003001",
    class_id="C000003001",
    legal_name="SPDR S&P Biotech ETF",
    trust_name="SPDR SERIES TRUST"
)

SPDR_XOP = SeriesMetadata(
    symbol="XOP",
    cik="1064642",
    series_id="S000003002",
    class_id="C000003002",
    legal_name="SPDR S&P Oil & Gas Exploration & Production ETF",
    trust_name="SPDR SERIES TRUST"
)


# --------------------------------------------------------------------------
# 4. Invesco Golden Fixture
# --------------------------------------------------------------------------
INVESCO_OMNIBUS_HTML = """
<html>
<body>
<div class="header">Invesco Exchange-Traded Fund Trust CIK 0001209466</div>

<div class="fund-summary">
  <h2>Invesco S&P 500 Equal Weight ETF</h2>
  <p>Series ID: S000004001 Class ID: C000004001 (RSP)</p>
  <h3>Investment Objective</h3>
  <p>The Fund seeks investment results that correspond generally to the price and yield of the S&P 500 Equal Weight Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund will normally invest at least 90% of its total assets in common stocks that comprise the S&P 500 Equal Weight Index. The Index equally weights the stocks in the S&P 500 Index.</p>
</div>

<div class="fund-summary">
  <h2>Invesco Water Resources ETF</h2>
  <p>Series ID: S000004002 Class ID: C000004002 (PHO)</p>
  <h3>Investment Objective</h3>
  <p>The Fund seeks investment results that correspond to the NASDAQ OMX US Water Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund invests at least 90% of its total assets in securities of companies that conserve and purify water for homes, businesses and industries.</p>
</div>
</body>
</html>
"""

INVESCO_RSP = SeriesMetadata(
    symbol="RSP",
    cik="1209466",
    series_id="S000004001",
    class_id="C000004001",
    legal_name="Invesco S&P 500 Equal Weight ETF",
    trust_name="INVESCO EXCHANGE-TRADED FUND TRUST"
)

INVESCO_PHO = SeriesMetadata(
    symbol="PHO",
    cik="1209466",
    series_id="S000004002",
    class_id="C000004002",
    legal_name="Invesco Water Resources ETF",
    trust_name="INVESCO EXCHANGE-TRADED FUND TRUST"
)


# --------------------------------------------------------------------------
# 5. Schwab Golden Fixture
# --------------------------------------------------------------------------
SCHWAB_OMNIBUS_HTML = """
<html>
<body>
<div>Schwab Strategic Trust CIK 0001454889</div>

<div class="fund">
  <h1>Fund Summary: Schwab U.S. Large-Cap ETF</h1>
  <p>Series ID: S000005001 Class ID: C000005001 (SCHX)</p>
  <h2>Investment Objective</h2>
  <p>The fund's goal is to track as closely as possible, before fees and expenses, the total return of the Dow Jones U.S. Large-Cap Total Stock Market Index.</p>
  <h2>Principal Investment Strategies</h2>
  <p>To pursue its goal, the fund generally invests in stocks that are included in the Dow Jones U.S. Large-Cap Total Stock Market Index. Under normal circumstances, the fund will invest at least 90% of its net assets in these stocks.</p>
</div>

<div class="fund">
  <h1>Fund Summary: Schwab U.S. Small-Cap ETF</h1>
  <p>Series ID: S000005002 Class ID: C000005002 (SCHA)</p>
  <h2>Investment Objective</h2>
  <p>The fund's goal is to track the Dow Jones U.S. Small-Cap Total Stock Market Index.</p>
  <h2>Principal Investment Strategies</h2>
  <p>The fund generally invests in stocks included in the Dow Jones U.S. Small-Cap Total Stock Market Index, investing at least 90% of its net assets in such component stocks.</p>
</div>
</body>
</html>
"""

SCHWAB_SCHX = SeriesMetadata(
    symbol="SCHX",
    cik="1454889",
    series_id="S000005001",
    class_id="C000005001",
    legal_name="Schwab U.S. Large-Cap ETF",
    trust_name="SCHWAB STRATEGIC TRUST"
)

SCHWAB_SCHA = SeriesMetadata(
    symbol="SCHA",
    cik="1454889",
    series_id="S000005002",
    class_id="C000005002",
    legal_name="Schwab U.S. Small-Cap ETF",
    trust_name="SCHWAB STRATEGIC TRUST"
)


# --------------------------------------------------------------------------
# 6. First Trust Golden Fixture
# --------------------------------------------------------------------------
FIRST_TRUST_OMNIBUS_HTML = """
<html>
<body>
<div>First Trust Exchange-Traded Fund CIK 0001329377</div>

<div class="prospectus-section">
  <h2>First Trust Morningstar Dividend Leaders Index Fund</h2>
  <p>Series ID: S000006001 | Class ID: C000006001 | Ticker: FDL</p>
  <h3>Investment Objective</h3>
  <p>The Fund seeks investment results that correspond generally to the price and yield of an equity index called the Morningstar Dividend Leaders Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund normally invests at least 90% of its net assets in common stocks that comprise the Morningstar Dividend Leaders Index. The Index consists of the 100 highest dividend-yielding stocks selected from the Morningstar US Market Index.</p>
</div>

<div class="prospectus-section">
  <h2>First Trust Dow Jones Internet Index Fund</h2>
  <p>Series ID: S000006002 | Class ID: C000006002 | Ticker: FDN</p>
  <h3>Investment Objective</h3>
  <p>The Fund seeks investment results that correspond generally to the Dow Jones Internet Composite Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund normally invests at least 90% of its net assets in common stocks of companies in the Internet industry represented in the Dow Jones Internet Composite Index.</p>
</div>
</body>
</html>
"""

FIRST_TRUST_FDL = SeriesMetadata(
    symbol="FDL",
    cik="1329377",
    series_id="S000006001",
    class_id="C000006001",
    legal_name="First Trust Morningstar Dividend Leaders Index Fund",
    trust_name="FIRST TRUST EXCHANGE-TRADED FUND"
)

FIRST_TRUST_FDN = SeriesMetadata(
    symbol="FDN",
    cik="1329377",
    series_id="S000006002",
    class_id="C000006002",
    legal_name="First Trust Dow Jones Internet Index Fund",
    trust_name="FIRST TRUST EXCHANGE-TRADED FUND"
)


# --------------------------------------------------------------------------
# 7. Global X Golden Fixture
# --------------------------------------------------------------------------
GLOBAL_X_OMNIBUS_HTML = """
<html>
<body>
<div>Global X Funds CIK 0001432353</div>

<div class="summary-section">
  <h2>Global X Artificial Intelligence & Technology ETF</h2>
  <p>Series ID: S000007001 Class ID: C000007001 (AIQ)</p>
  <h3>Investment Objective</h3>
  <p>The Global X Artificial Intelligence & Technology ETF seeks investment results that correspond generally to the price and yield performance of the Indxx Artificial Intelligence & Big Data Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund invests at least 80% of its total assets in the securities of the Indxx Artificial Intelligence & Big Data Index, representing companies involved in artificial intelligence and technology applications.</p>
</div>

<div class="summary-section">
  <h2>Global X Robotics & Artificial Intelligence ETF</h2>
  <p>Series ID: S000007002 Class ID: C000007002 (BOTZ)</p>
  <h3>Investment Objective</h3>
  <p>The Fund seeks to track the Indxx Global Robotics & Artificial Intelligence Thematic Index.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The Fund invests at least 80% of its total assets in the securities of companies engaged in robotics and automation technologies across global markets.</p>
</div>
</body>
</html>
"""

GLOBAL_X_AIQ = SeriesMetadata(
    symbol="AIQ",
    cik="1432353",
    series_id="S000007001",
    class_id="C000007001",
    legal_name="Global X Artificial Intelligence & Technology ETF",
    trust_name="Global X Funds"
)

GLOBAL_X_BOTZ = SeriesMetadata(
    symbol="BOTZ",
    cik="1432353",
    series_id="S000007002",
    class_id="C000007002",
    legal_name="Global X Robotics & Artificial Intelligence ETF",
    trust_name="Global X Funds"
)


# --------------------------------------------------------------------------
# 8. Capital Group Golden Fixture
# --------------------------------------------------------------------------
CAPITAL_GROUP_OMNIBUS_HTML = """
<html>
<body>
<div>Capital Group Fixed Income ETF Trust CIK 0001870117</div>

<div class="fund-summary">
  <h2>Capital Group Core Bond ETF</h2>
  <p>Series ID: S000074251 Class ID: C000231860 (CGCP)</p>
  <h3>Investment Objective</h3>
  <p>The fund's investment objective is to provide as high a level of current income as is consistent with the preservation of capital.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The fund invests primarily in investment grade bonds and other debt instruments issued by the U.S. government, government agencies, and corporations.</p>
</div>

<div class="fund-summary">
  <h2>Capital Group U.S. Multi-Sector Income ETF</h2>
  <p>Series ID: S000077688 Class ID: C000238176 (CGMS)</p>
  <h3>Investment Objective</h3>
  <p>The fund's investment objective is to provide a high level of current income, with capital appreciation as a secondary objective.</p>
  <h3>Principal Investment Strategies</h3>
  <p>The fund normally invests at least 80% of its assets in bonds and other debt instruments across three primary sectors: high-yield corporate debt, investment grade corporate debt and securitized debt.</p>
</div>
</body>
</html>
"""

CAPITAL_GROUP_CGCP = SeriesMetadata(
    symbol="CGCP",
    cik="1870117",
    series_id="S000074251",
    class_id="C000231860",
    legal_name="Capital Group Core Bond ETF",
    trust_name="Capital Group Fixed Income ETF Trust"
)

CAPITAL_GROUP_CGMS = SeriesMetadata(
    symbol="CGMS",
    cik="1870117",
    series_id="S000077688",
    class_id="C000238176",
    legal_name="Capital Group U.S. Multi-Sector Income ETF",
    trust_name="Capital Group Fixed Income ETF Trust"
)


# --------------------------------------------------------------------------
# 9. Fidelity Golden Fixture
# --------------------------------------------------------------------------
FIDELITY_OMNIBUS_HTML = """
<html>
<body>
<div>Fidelity Salem Street Trust CIK 0000035315</div>

<div class="fund-summary">
  <h1>Fund Summary: Fidelity Real Estate Index Fund</h1>
  <p>Series ID: S000033639 Class ID: C000103377 (FSRNX)</p>
  <h2>Investment Objective</h2>
  <p>Fidelity Real Estate Index Fund seeks to provide investment results that correspond to the total return of equity REITs and other real estate-related investments.</p>
  <h2>Principal Investment Strategies</h2>
  <p>Normally investing at least 80% of assets in securities included in the MSCI US IMI Real Estate 25/25 Index, representing the real estate industry.</p>
</div>

<div class="fund-summary">
  <h1>Fund Summary: Fidelity SAI Small-Mid Cap Momentum Index Fund</h1>
  <p>Series ID: S000050321 Class ID: C000158877 (FZFLX)</p>
  <h2>Investment Objective</h2>
  <p>The fund seeks to provide investment results that correspond to the total return of small- to mid-capitalization stocks with high momentum characteristics.</p>
  <h2>Principal Investment Strategies</h2>
  <p>Normally investing at least 80% of assets in securities included in the Fidelity U.S. Small-Mid Cap Momentum Focus Index, representing small-cap and mid-cap momentum equities.</p>
</div>
</body>
</html>
"""

FIDELITY_FSRNX = SeriesMetadata(
    symbol="FSRNX",
    cik="35315",
    series_id="S000033639",
    class_id="C000103377",
    legal_name="Fidelity Real Estate Index Fund",
    trust_name="FIDELITY SALEM STREET TRUST"
)

FIDELITY_FZFLX = SeriesMetadata(
    symbol="FZFLX",
    cik="35315",
    series_id="S000050321",
    class_id="C000158877",
    legal_name="Fidelity SAI Small-Mid Cap Momentum Index Fund",
    trust_name="FIDELITY SALEM STREET TRUST"
)
