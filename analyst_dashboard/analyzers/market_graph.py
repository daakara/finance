"""Market Graph & Contagion Engineering Engine."""

from typing import Dict, Any, List


class MarketGraphEngine:
    """Builds an interconnected knowledge and contagion graph across:
    1. Upstream Suppliers (Hardware, Foundries, Datacenters)
    2. Downstream Enterprise Customers & Distributors
    3. Macro Drivers (FRED Interest Rates, Inflation, Bond Yields)
    4. Sector Competitors & Contagion Beta
    """

    GRAPH_TOPOLOGY = {
        "NVDA": {
            "upstream": [{"name": "TSMC (Foundry)", "link": "90% of advanced GPU wafer production", "impact": "High"}],
            "downstream": [{"name": "MSFT & PLTR (Cloud AI)", "link": "Hyperscale AI infrastructure customers", "impact": "High"}],
            "macro": [{"name": "FRED HY OAS Spread", "link": "Tight credit spreads fuel corporate capex", "impact": "Medium"}],
            "peers": [{"name": "AMD & INTC", "link": "Data center accelerator competition", "impact": "Medium"}],
        },
        "AAPL": {
            "upstream": [{"name": "TSMC & Foxconn", "link": "A-series silicon and device assembly", "impact": "High"}],
            "downstream": [{"name": "Global Retail Consumers", "link": "Hardware replacement cycle and Services subscription", "impact": "High"}],
            "macro": [{"name": "FRED CPI Inflation", "link": "Consumer discretionary purchasing power", "impact": "Medium"}],
            "peers": [{"name": "MSFT & GOOGL", "link": "Ecosystem and digital services competition", "impact": "Medium"}],
        },
        "MSFT": {
            "upstream": [{"name": "NVDA & AMD (Accelerators)", "link": "Datacenter AI GPU supply", "impact": "High"}],
            "downstream": [{"name": "Fortune 500 Enterprises", "link": "Azure Cloud and Office 365 Copilot suite", "impact": "High"}],
            "macro": [{"name": "10Y Treasury Yield", "link": "Discount rate for enterprise software multiples", "impact": "High"}],
            "peers": [{"name": "GOOGL & AMZN", "link": "Hyperscale cloud market share", "impact": "High"}],
        },
        "PLTR": {
            "upstream": [{"name": "Cloud Providers (Azure/AWS)", "link": "Hosting infrastructure for AIP platforms", "impact": "Medium"}],
            "downstream": [{"name": "US DoD & Enterprise AI", "link": "Defense and commercial data operating systems", "impact": "High"}],
            "macro": [{"name": "Federal Defense Budgets", "link": "Government software procurement mandates", "impact": "High"}],
            "peers": [{"name": "SNOW & CRWD", "link": "Enterprise data and security infrastructure", "impact": "Medium"}],
        },
        "BTC-USD": {
            "upstream": [{"name": "Global Energy Grids", "link": "SHA-256 Proof-of-Work mining hash power", "impact": "Medium"}],
            "downstream": [{"name": "Institutional Spot ETFs", "link": "BlackRock (IBIT) & Fidelity custodial inflows", "impact": "High"}],
            "macro": [{"name": "Global M2 Money Supply", "link": "Central bank liquidity and currency debasement hedge", "impact": "High"}],
            "peers": [{"name": "ETH-USD & SOL-USD", "link": "Digital asset market dominance", "impact": "High"}],
        },
        "ETH-USD": {
            "upstream": [{"name": "Ethereum Validator Network", "link": "Proof-of-Stake consensus & staking yield", "impact": "High"}],
            "downstream": [{"name": "Layer-2 Rollups & DeFi", "link": "Arbitrum, Base, and tokenized real-world assets", "impact": "High"}],
            "macro": [{"name": "Fed Funds Rate", "link": "Risk-free yield vs 3.2% staking yield spread", "impact": "High"}],
            "peers": [{"name": "SOL-USD & AVAX-USD", "link": "Smart contract execution layer competition", "impact": "High"}],
        },
        "SPY": {
            "upstream": [{"name": "Top 10 Mega-Cap Equities", "link": "35% of total index weight (AAPL, MSFT, NVDA)", "impact": "High"}],
            "downstream": [{"name": "Global Pension & 401(k) Inflows", "link": "Passive systematic investment flows", "impact": "High"}],
            "macro": [{"name": "FRED 10Y-2Y Yield Curve", "link": "Leading indicator for economic expansion vs recession", "impact": "High"}],
            "peers": [{"name": "QQQ & IWM", "link": "Large-cap tech vs small-cap rotation", "impact": "Medium"}],
        },
    }

    def get_relationship_graph(self, symbol: str) -> Dict[str, Any]:
        """Retrieve relationship topology and contagion linkages for an asset."""
        sym_clean = symbol.upper().replace("-USD", "")
        key = symbol.upper() if symbol.upper() in self.GRAPH_TOPOLOGY else sym_clean

        data = self.GRAPH_TOPOLOGY.get(
            key,
            {
                "upstream": [{"name": "Component & Hardware Suppliers", "link": "Core supply chain inputs", "impact": "Medium"}],
                "downstream": [{"name": "Enterprise & Retail Customers", "link": "Revenue and cash flow sources", "impact": "High"}],
                "macro": [{"name": "FRED Interest Rate Cycle", "link": "Discount rate and capital cost sensitivity", "impact": "High"}],
                "peers": [{"name": "Sector Industry Benchmark", "link": "Relative valuation and multiple contagion", "impact": "Medium"}],
            },
        )

        return {
            "rootNode": symbol.upper(),
            "topology": data,
            "systemicContagionRisk": "Low-to-Moderate (Well-Diversified)",
        }

