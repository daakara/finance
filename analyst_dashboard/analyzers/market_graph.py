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
        "SMCI": {
            "upstream": [
                {"name": "NVIDIA (Blackwell & Hopper GPUs)", "link": "Direct accelerator allocation and NVLink server baseboards", "impact": "Critical"},
                {"name": "Intel & AMD (x86 Server CPUs)", "link": "Xeon and EPYC datacenter host processors", "impact": "High"},
                {"name": "CoolIT / Vertiv (Liquid Cooling)", "link": "Direct-to-chip DLC cooling manifolds and coolant distribution units (CDUs)", "impact": "High"},
            ],
            "downstream": [
                {"name": "xAI (Elon Musk Colossus 100k Cluster)", "link": "Primary liquid-cooled AI cluster rack infrastructure", "impact": "Critical"},
                {"name": "CoreWeave & Neoclouds", "link": "Turnkey GPU cloud datacenter deployments", "impact": "High"},
                {"name": "Meta Platforms & Tesla", "link": "Hyperscale AI model training clusters", "impact": "High"},
            ],
            "macro": [
                {"name": "Hyperscaler AI Capex Budgets", "link": "Big Tech multi-billion-dollar datacenter procurement spending", "impact": "Critical"},
                {"name": "High-Yield Corporate Spreads", "link": "Working capital inventory and components financing costs", "impact": "High"},
            ],
            "peers": [
                {"name": "Dell Technologies (DELL)", "link": "PowerEdge enterprise AI server competition", "impact": "High"},
                {"name": "Hewlett Packard Enterprise (HPE)", "link": "Cray and ProLiant high-density server competition", "impact": "High"},
                {"name": "Wiwynn & Quanta (Taiwan ODMs)", "link": "Direct hyperscaler custom server manufacturing pressure", "impact": "Medium"},
            ],
        },
        "IREN": {
            "upstream": [
                {"name": "NVIDIA (H100/H200/Blackwell GPUs)", "link": "AI cloud GPU server cluster hardware procurement", "impact": "Critical"},
                {"name": "Texas ERCOT & BC Hydro Power Grids", "link": ">1.4GW low-cost renewable power interconnection pipeline", "impact": "Critical"},
                {"name": "Bitmain & MicroBT (ASIC Miners)", "link": "High-efficiency SHA-256 fleet hardware", "impact": "High"},
            ],
            "downstream": [
                {"name": "Enterprise AI Cloud Tenants", "link": "Hyperscaler and AI startup high-density compute leasing", "impact": "High"},
                {"name": "Bitcoin Global Network", "link": "Automated block reward & transaction fee mining revenue", "impact": "High"},
            ],
            "macro": [
                {"name": "Wholesale Electricity Rates ($/MWh)", "link": "Direct power cost floor for mining and datacenter hosting", "impact": "High"},
                {"name": "Bitcoin Hashprice ($/PH/s/Day)", "link": "Global mining revenue realization per unit of compute", "impact": "High"},
            ],
            "peers": [
                {"name": "Core Scientific (CORZ)", "link": "AI datacenter colocation hosting & Bitcoin mining peer", "impact": "High"},
                {"name": "MARA Holdings & Riot Platforms", "link": "Large-scale North American digital infrastructure peers", "impact": "High"},
            ],
        },
        "NVDA": {
            "upstream": [
                {"name": "TSMC (Advanced Packaging & CoWoS)", "link": "Exclusive 3nm/4nm wafer fabrication & 2.5D packaging", "impact": "Critical"},
                {"name": "SK Hynix & Micron (HBM3e Memory)", "link": "Ultra-high-bandwidth memory stacks for AI accelerators", "impact": "Critical"},
            ],
            "downstream": [
                {"name": "Microsoft, Meta & Alphabet", "link": ">40% of hyperscaler AI training and inference cluster revenues", "impact": "Critical"},
                {"name": "SMCI, Dell & HPE (Server OEM/ODMs)", "link": "Distribution channels for enterprise rack systems", "impact": "High"},
            ],
            "macro": [
                {"name": "FRED HY OAS Spread", "link": "Tight corporate credit spreads fuel enterprise capex", "impact": "Medium"},
                {"name": "US-China Semiconductor Export Controls", "link": "Bespoke datacenter silicon compliance mandates", "impact": "High"},
            ],
            "peers": [
                {"name": "AMD (Instinct MI300X/MI350)", "link": "Open-ecosystem ROCm AI accelerator competition", "impact": "High"},
                {"name": "Custom Hyperscaler ASICs (Google TPU, AWS Trainium)", "link": "In-house cloud inference chip substitution", "impact": "High"},
            ],
        },
        "AAPL": {
            "upstream": [
                {"name": "TSMC (N3E Node Silicon)", "link": "A18 & M4 custom Apple Silicon processors", "impact": "Critical"},
                {"name": "Foxconn & Luxshare (Assembly)", "link": "Global iPhone and hardware manufacturing footprint", "impact": "High"},
            ],
            "downstream": [
                {"name": "Global Active Installed Base (2.2B Devices)", "link": "Consumer replacement cycle & Services subscriptions", "impact": "Critical"},
                {"name": "Enterprise & Education Channels", "link": "Mac and iPad enterprise ecosystem adoption", "impact": "Medium"},
            ],
            "macro": [
                {"name": "FRED CPI Inflation & Real Wages", "link": "Consumer discretionary premium purchasing power", "impact": "Medium"},
            ],
            "peers": [
                {"name": "Samsung Electronics", "link": "Premium smartphone hardware competition", "impact": "High"},
                {"name": "Alphabet (Android & Services)", "link": "Mobile OS ecosystem and search revenue sharing", "impact": "High"},
            ],
        },
        "MSFT": {
            "upstream": [
                {"name": "NVIDIA & AMD (AI GPUs)", "link": "Azure hyperscale datacenter AI training clusters", "impact": "Critical"},
                {"name": "OpenAI (Frontier Foundation Models)", "link": "Exclusive commercial API & Copilot model licensing", "impact": "Critical"},
            ],
            "downstream": [
                {"name": "Global Fortune 500 Enterprises", "link": "Azure Cloud, Office 365 Copilot & Teams subscriptions", "impact": "Critical"},
            ],
            "macro": [
                {"name": "10Y Treasury Yield", "link": "Discount rate for enterprise recurring SaaS valuations", "impact": "High"},
            ],
            "peers": [
                {"name": "Amazon AWS & Google Cloud", "link": "Hyperscale cloud market share and AI workload hosting", "impact": "High"},
            ],
        },
        "PLTR": {
            "upstream": [
                {"name": "Cloud Providers (AWS & Azure)", "link": "GovCloud and commercial IL6 hosting infrastructure", "impact": "Medium"},
            ],
            "downstream": [
                {"name": "US Department of Defense & Intelligence", "link": "Mission-critical Maven and Vantage battlefield software", "impact": "Critical"},
                {"name": "Commercial Enterprise Clients", "link": "AIP (Artificial Intelligence Platform) workflow deployment", "impact": "High"},
            ],
            "macro": [
                {"name": "Federal Defense Spending (NDAA)", "link": "Classified software procurement line items", "impact": "High"},
            ],
            "peers": [
                {"name": "Snowflake & Databricks", "link": "Enterprise data warehouse and lakehouse architectures", "impact": "Medium"},
            ],
        },
        "NVO": {
            "upstream": [
                {"name": "Catalent & Lonza (Sterile Fill-Finish)", "link": "Aseptic injectable pen cartridge manufacturing", "impact": "Critical"},
            ],
            "downstream": [
                {"name": "PBMs (CVS Caremark, Express Scripts)", "link": "Formulary placement and commercial insurance rebates", "impact": "Critical"},
                {"name": "Medicare Part D & Commercial Health Plans", "link": "Cardiovascular risk reduction reimbursement coverage", "impact": "High"},
            ],
            "macro": [
                {"name": "CMS Drug Price Negotiation (IRA)", "link": "Federal statutory price ceiling regulations", "impact": "High"},
            ],
            "peers": [
                {"name": "Eli Lilly (LLY)", "link": "Tirzepatide (Zepbound/Mounjaro) obesity competition", "impact": "Critical"},
            ],
        },
        "LLY": {
            "upstream": [
                {"name": "Internal Manufacturing (Indiana & Ireland)", "link": "Multi-billion-dollar dedicated peptide synthesis facilities", "impact": "High"},
            ],
            "downstream": [
                {"name": "Global Pharmacy Benefit Managers", "link": "Commercial obesity and type-2 diabetes insurance coverage", "impact": "Critical"},
            ],
            "macro": [
                {"name": "Federal Treat and Reduce Obesity Act", "link": "Legislative catalyst for universal Medicare GLP-1 reimbursement", "impact": "High"},
            ],
            "peers": [
                {"name": "Novo Nordisk (NVO)", "link": "Semaglutide metabolic therapy competition", "impact": "Critical"},
            ],
        },
        "ANET": {
            "upstream": [
                {"name": "Broadcom (Tomahawk 5 & Jericho 3-AI)", "link": "High-bandwidth merchant switching silicon", "impact": "Critical"},
            ],
            "downstream": [
                {"name": "Microsoft Azure & Meta Platforms", "link": ">40% of hyperscaler AI Ethernet switch cluster deployments", "impact": "Critical"},
            ],
            "macro": [
                {"name": "800G/1.6T Ethernet Transition Cycle", "link": "Hyperscaler backend cluster interconnect replacement", "impact": "High"},
            ],
            "peers": [
                {"name": "Cisco Systems (CSCO)", "link": "Enterprise and cloud networking competition", "impact": "Medium"},
                {"name": "NVIDIA (Mellanox InfiniBand)", "link": "Ultra-low-latency AI cluster fabric competition", "impact": "High"},
            ],
        },
        "CPRX": {
            "upstream": [
                {"name": "Santhera Pharmaceuticals (Licensing)", "link": "Exclusive North American Agamree rights", "impact": "Critical"},
            ],
            "downstream": [
                {"name": "Rare Disease Neuromuscular Patients", "link": "LEMS (Firdapse) & DMD specialty pharmacy distribution", "impact": "Critical"},
            ],
            "macro": [
                {"name": "FDA Orphan Drug Exclusivity (ODE)", "link": "Statutory market exclusivity protection", "impact": "Critical"},
            ],
            "peers": [
                {"name": "BioMarin & Sarepta Therapeutics", "link": "Rare genetic and neuromuscular therapy peers", "impact": "Medium"},
            ],
        },
        "DHLGY": {
            "upstream": [
                {"name": "Boeing & Airbus (Freighter Aircraft)", "link": "Global express air cargo transport fleet", "impact": "High"},
            ],
            "downstream": [
                {"name": "Global B2B Supply Chains & E-Commerce", "link": "Cross-border international time-definite parcel logistics", "impact": "Critical"},
            ],
            "macro": [
                {"name": "Baltic Dry & Global Air Freight Yields", "link": "International trade velocity and ton-mile pricing power", "impact": "High"},
            ],
            "peers": [
                {"name": "FedEx (FDX) & UPS", "link": "Global courier express and contract logistics competition", "impact": "High"},
            ],
        },
        "BTC-USD": {
            "upstream": [{"name": "Global Energy Grids & Miners", "link": "SHA-256 Proof-of-Work mining hash power", "impact": "Medium"}],
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

