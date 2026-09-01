"""Catalyst & Multi-Year Earnings Impact Analyzer Engine."""

from typing import Dict, Any, List
from datetime import datetime

# Knowledge base of clinical trials, product launches, legislative milestones, and multi-year valuation forecasts
ASSET_CATALYST_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "NVDA": {
        "company_name": "NVIDIA Corporation",
        "sector": "Semiconductors & AI Hardware Infrastructure",
        "primary_drug_trial": "Blackwell GB200 NVL72 & Rubin R100 Ultra Platform",
        "trial_phase": "Hyperscaler Mass Deployment Phase",
        "trial_readout_timeline": "Q3 2026 - Q1 2027",
        "efficacy_summary": "4x AI training throughput and 30x real-time inference power efficiency vs H100 Hopper generation.",
        "competitive_edge": "Full-stack proprietary CUDA software moat, NVLink 5.0 interconnect switches, and turnkey AI factory architectures.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Congressional Hearing: US Federal AI Compute Export Regulations & Middle East Sovereign Waivers", "impact": "High Strategic"},
            {"date": "Q4 2026", "event": "GB200 Full High-Density Liquid Cooling Production Ramp", "impact": "High Positive"},
            {"date": "Q1 2027", "event": "GTC 2027: Rubin Next-Gen Architecture Official Architecture Deep Dive", "impact": "Transformational"},
            {"date": "2027-2028", "event": "Sovereign AI Compute Appropriations ($50B+ Global Pipeline)", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 128.5, "net_margin_pct": 54.0, "projected_eps": 2.85, "implied_pe": 45.0, "implied_target": 128.25},
            {"year": 2027, "revenue_billions": 182.0, "net_margin_pct": 52.5, "projected_eps": 4.10, "implied_pe": 38.0, "implied_target": 155.80},
            {"year": 2029, "revenue_billions": 245.0, "net_margin_pct": 50.0, "projected_eps": 5.60, "implied_pe": 32.0, "implied_target": 179.20},
            {"year": 2031, "revenue_billions": 310.0, "net_margin_pct": 48.5, "projected_eps": 7.20, "implied_pe": 28.0, "implied_target": 201.60}
        ]
    },
    "PLTR": {
        "company_name": "Palantir Technologies Inc.",
        "sector": "Defense Software & Enterprise AI",
        "primary_drug_trial": "AIP (Artificial Intelligence Platform) & Defense TITAN",
        "trial_phase": "Commercial Bootcamp Conversion & Military Deployment",
        "trial_readout_timeline": "2026 - 2027",
        "efficacy_summary": "Reduces enterprise LLM workflow prototype-to-production deployment time from months to under 72 hours.",
        "competitive_edge": "IL6/JWCC defense security clearance credentials and ontological data lineage architecture.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "DoD Defense Appropriations Bill (NDAA) Multi-Year Software Allocation Review", "impact": "High Strategic"},
            {"date": "Q4 2026", "event": "Commercial AIP US Customer Count Acceleration (>1,200 Bootcamps Converted)", "impact": "High Positive"},
            {"date": "Q1 2027", "event": "US Army TITAN Ground Station Next-Phase Tactical Procurement Rollout", "impact": "High Positive"},
            {"date": "2028", "event": "Sovereign Defense AI Autonomous Operations Platform Expansion", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 3.4, "net_margin_pct": 28.0, "projected_eps": 0.45, "implied_pe": 80.0, "implied_target": 36.00},
            {"year": 2027, "revenue_billions": 5.2, "net_margin_pct": 32.0, "projected_eps": 0.78, "implied_pe": 65.0, "implied_target": 50.70},
            {"year": 2029, "revenue_billions": 7.8, "net_margin_pct": 35.0, "projected_eps": 1.25, "implied_pe": 50.0, "implied_target": 62.50},
            {"year": 2031, "revenue_billions": 11.5, "net_margin_pct": 38.0, "projected_eps": 1.95, "implied_pe": 40.0, "implied_target": 78.00}
        ]
    },
    "AAPL": {
        "company_name": "Apple Inc.",
        "sector": "Consumer Electronics & Digital Services",
        "primary_drug_trial": "Apple Intelligence On-Device AI & M5 Silicon Architecture",
        "trial_phase": "Global iOS Rollout & Enterprise Services Monetization",
        "trial_readout_timeline": "Fall Product Cycle & Developer Conferences",
        "efficacy_summary": "High-margin recurring services monetization across 2.2B active installed devices and custom neural engine silicon.",
        "competitive_edge": "Hardware-software vertical integration, privacy-first on-device compute moats, and high customer retention.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Fall iPhone Product Cycle with Native Apple Intelligence Features", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Services ARPU Acceleration & App Store Enterprise Subscriptions", "impact": "High Positive"},
            {"date": "2027", "event": "M5 Architecture Ultra-Thin Form Factor Mac Rollout", "impact": "Positive"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 395.0, "net_margin_pct": 26.5, "projected_eps": 6.85, "implied_pe": 33.0, "implied_target": 226.05},
            {"year": 2027, "revenue_billions": 445.0, "net_margin_pct": 28.0, "projected_eps": 8.40, "implied_pe": 30.0, "implied_target": 252.00},
            {"year": 2029, "revenue_billions": 510.0, "net_margin_pct": 29.5, "projected_eps": 10.20, "implied_pe": 28.0, "implied_target": 285.60},
            {"year": 2031, "revenue_billions": 580.0, "net_margin_pct": 30.5, "projected_eps": 12.50, "implied_pe": 26.0, "implied_target": 325.00}
        ]
    },
    "KO": {
        "company_name": "The Coca-Cola Company",
        "sector": "Consumer Defensive / Beverages",
        "primary_drug_trial": "Global Volume Growth, Bottling System Refranchising & Direct-Store-Delivery",
        "trial_phase": "Commercial Market Leadership & Margin Expansion",
        "trial_readout_timeline": "Quarterly Global Unit Case Volume Readouts",
        "efficacy_summary": "World's preeminent beverage brand portfolio with unmatched global distribution bottling network, pricing power, and dividend stability.",
        "competitive_edge": "200+ master bottler network agreements, retail shelf-space monopoly, and strong brand pricing inelasticity.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Global Unit Case Volume & Concentrates Net Sales Readout", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Emerging Market Ready-to-Drink (RTD) Alcohol & Zero-Sugar Expansion", "impact": "Positive"},
            {"date": "2027", "event": "Annual Dividend Aristocrat Payout Increase", "impact": "High Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 47.2, "net_margin_pct": 24.5, "projected_eps": 2.88, "implied_pe": 24.0, "implied_target": 69.12},
            {"year": 2027, "revenue_billions": 52.0, "net_margin_pct": 25.2, "projected_eps": 3.25, "implied_pe": 23.0, "implied_target": 74.75},
            {"year": 2029, "revenue_billions": 57.5, "net_margin_pct": 26.0, "projected_eps": 3.70, "implied_pe": 22.0, "implied_target": 81.40},
            {"year": 2031, "revenue_billions": 63.5, "net_margin_pct": 26.5, "projected_eps": 4.25, "implied_pe": 21.0, "implied_target": 89.25}
        ]
    },
    "SBUX": {
        "company_name": "Starbucks Corporation",
        "sector": "Consumer Cyclical / Specialty Retail & Restaurants",
        "primary_drug_trial": "Triple Shot Reinvention, Store-Level Throughput & Digital Rewards Expansion",
        "trial_phase": "Operational Turnaround & Unit Economics Acceleration",
        "trial_readout_timeline": "Quarterly Same-Store Sales (Comps) Reporting",
        "efficacy_summary": "Premier global specialty coffee brand driving customer throughput with 38,000+ locations and 34M+ active Rewards members.",
        "competitive_edge": "Siren Craft System barista workflow acceleration, mobile order & pay ecosystem, and prime global real estate footprint.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "US Same-Store Sales (Comps) & Store Throughput Acceleration Report", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "China JV Strategic Partnership & Store Format Modernization", "impact": "High Positive"},
            {"date": "2027", "event": "Global Siren Craft System Full Equipment Rollout", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 38.5, "net_margin_pct": 11.5, "projected_eps": 3.80, "implied_pe": 25.0, "implied_target": 95.00},
            {"year": 2027, "revenue_billions": 43.0, "net_margin_pct": 13.0, "projected_eps": 4.85, "implied_pe": 23.0, "implied_target": 111.55},
            {"year": 2029, "revenue_billions": 48.5, "net_margin_pct": 14.2, "projected_eps": 6.10, "implied_pe": 21.0, "implied_target": 128.10},
            {"year": 2031, "revenue_billions": 55.0, "net_margin_pct": 15.0, "projected_eps": 7.40, "implied_pe": 20.0, "implied_target": 148.00}
        ]
    },
    "NVO": {
        "company_name": "Novo Nordisk A/S",
        "sector": "Pharmaceuticals & Biotechnology",
        "primary_drug_trial": "Amycretin (Oral & Subcutaneous Dual GLP-1/Amylin Agonist)",
        "trial_phase": "Phase 2 / Phase 3 Registration Trials",
        "trial_readout_timeline": "Q4 2026 - Q2 2027",
        "efficacy_summary": "13.1% mean body weight reduction at 12 weeks in early trials (vs ~6% for standard Wegovy), with daily oral pill convenience.",
        "competitive_edge": "Bypasses cold-chain injectable supply constraints, dramatically expanding addressable patient population.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Senate HELP Committee Hearing on Medicare Part D GLP-1 Reimbursement Expansion", "impact": "High Strategic"},
            {"date": "Q4 2026", "event": "Amycretin Oral Phase 2 dose-ranging full readout", "impact": "High Positive"},
            {"date": "Q1 2027", "event": "Phase 3 Head-to-Head vs Semaglutide trial launch", "impact": "High Positive"},
            {"date": "Q3 2027", "event": "FDA & EMA regulatory filing for CagriSema", "impact": "High Positive"},
            {"date": "2028-2029", "event": "Commercial launch & manufacturing capacity ramp", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 41.2, "net_margin_pct": 36.5, "projected_eps": 3.85, "implied_pe": 36.0, "implied_target": 138.60},
            {"year": 2027, "revenue_billions": 58.4, "net_margin_pct": 37.8, "projected_eps": 5.60, "implied_pe": 30.0, "implied_target": 168.00},
            {"year": 2029, "revenue_billions": 78.2, "net_margin_pct": 38.5, "projected_eps": 7.85, "implied_pe": 30.0, "implied_target": 235.50},
            {"year": 2031, "revenue_billions": 102.5, "net_margin_pct": 39.2, "projected_eps": 10.90, "implied_pe": 30.0, "implied_target": 327.00}
        ]
    },
    "ARWR": {
        "company_name": "Arrowhead Pharmaceuticals Inc.",
        "sector": "Biotechnology / Targeted RNAi Therapeutics",
        "primary_drug_trial": "Plozasiran (APOC3 RNAi) FCS PDUFA & Phase 3 PALISADE / SHASTA-3 Program",
        "trial_phase": "FDA NDA Priority Review & Phase 3 Pivotal Readouts",
        "trial_readout_timeline": "H2 2026 - Q1 2027",
        "efficacy_summary": "Up to 86% APOC3 knockdown and 74% triglyceride reduction with once-quarterly subcutaneous dosing in pivotal PALISADE trials.",
        "competitive_edge": "Proprietary TRiM (Targeted RNAi Molecule) platform enabling tissue-specific extrahepatic delivery and multi-billion-dollar royalty monetization.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "FDA NDA Acceptance & Priority Review PDUFA Date for Plozasiran (Familial Chylomicronemia Syndrome)", "impact": "High Strategic"},
            {"date": "Q4 2026", "event": "Phase 3 PALISADE Severe Hypertriglyceridemia (SHTG) 52-Week Long-Term Efficacy Readout", "impact": "High Positive"},
            {"date": "Q1 2027", "event": "Zodasiran (ANGPTL3 RNAi) Phase 3 Dyslipidemia Registrational Trial Launch", "impact": "Positive"},
            {"date": "2027", "event": "Commercial Launch of Plozasiran & Royalty Revenue Acceleration", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 0.45, "net_margin_pct": 18.0, "projected_eps": 0.65, "implied_pe": 45.0, "implied_target": 88.50},
            {"year": 2027, "revenue_billions": 0.95, "net_margin_pct": 32.0, "projected_eps": 2.45, "implied_pe": 36.0, "implied_target": 115.00},
            {"year": 2029, "revenue_billions": 1.85, "net_margin_pct": 38.0, "projected_eps": 5.80, "implied_pe": 28.0, "implied_target": 162.40},
            {"year": 2031, "revenue_billions": 3.10, "net_margin_pct": 42.0, "projected_eps": 10.50, "implied_pe": 22.0, "implied_target": 231.00}
        ]
    },
    "LNTH": {
        "company_name": "Lantheus Holdings Inc.",
        "sector": "Radiopharmaceuticals & Precision Diagnostics",
        "primary_drug_trial": "Pylarify (PSMA PET Imaging) & Point Biopharma Radioligand Pipeline",
        "trial_phase": "Commercial Market Monopoly & Radiopharmaceutical Scale",
        "trial_readout_timeline": "Quarterly Pylarify Scan Volume & PSMA Readouts",
        "efficacy_summary": "Gold-standard PSMA PET imaging agent with >80% diagnostic market share across prostate cancer staging.",
        "competitive_edge": "Established nationwide cyclotron distribution network and Medicare CMS reimbursement pass-through stability.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Quarterly Pylarify Net Product Sales & Hospital Adoption Report", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Next-Gen Oncology Radioligand Therapeutic Phase 2/3 Clinical Data", "impact": "High Positive"},
            {"date": "2027", "event": "European EMA Market Expansion & CMS Pass-Through Extension", "impact": "Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 1.65, "net_margin_pct": 34.0, "projected_eps": 7.20, "implied_pe": 16.0, "implied_target": 115.20},
            {"year": 2027, "revenue_billions": 2.10, "net_margin_pct": 36.5, "projected_eps": 9.80, "implied_pe": 15.0, "implied_target": 147.00},
            {"year": 2029, "revenue_billions": 2.65, "net_margin_pct": 38.0, "projected_eps": 12.80, "implied_pe": 14.0, "implied_target": 179.20},
            {"year": 2031, "revenue_billions": 3.30, "net_margin_pct": 39.0, "projected_eps": 16.50, "implied_pe": 13.0, "implied_target": 214.50}
        ]
    },
    "DHLGY": {
        "company_name": "Deutsche Post DHL Group ADR",
        "sector": "Industrials / Global Freight & Logistics Network",
        "primary_drug_trial": "Global Freight Rate Yields, E-Commerce Parcel Pricing & Supply Chain Margin Optimization",
        "trial_phase": "Global Express Leadership & B2B Freight Recovery",
        "trial_readout_timeline": "Quarterly Ocean/Air Freight Volume & German Parcel Readouts",
        "efficacy_summary": "World's preeminent logistics and express cargo network operating across 220+ countries, with leading market share in international time-definite delivery.",
        "competitive_edge": "Unmatched global air/ocean freight fleet, automated sorting hubs, and disciplined dynamic fuel/rate surcharge mechanisms.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Quarterly Global Air & Ocean Freight Volume & Yield Realization", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Peak Holiday E-Commerce Parcel Surcharge & Cross-Border Delivery Report", "impact": "High Positive"},
            {"date": "2027", "event": "Post-Restructuring B2B Supply Chain Operating Margin Expansion", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 84.5, "net_margin_pct": 5.8, "projected_eps": 3.85, "implied_pe": 12.0, "implied_target": 35.50},
            {"year": 2027, "revenue_billions": 92.0, "net_margin_pct": 6.5, "projected_eps": 4.60, "implied_pe": 11.5, "implied_target": 40.25},
            {"year": 2029, "revenue_billions": 100.0, "net_margin_pct": 7.0, "projected_eps": 5.40, "implied_pe": 11.0, "implied_target": 45.60},
            {"year": 2031, "revenue_billions": 109.0, "net_margin_pct": 7.4, "projected_eps": 6.25, "implied_pe": 10.5, "implied_target": 51.20}
        ]
    },
    "XOM": {
        "company_name": "Exxon Mobil Corporation",
        "sector": "Energy / Integrated Oil & Gas",
        "primary_drug_trial": "Permian Basin Pioneer Integration, Guyana Offshore FPSO Scale & Refining Margins",
        "trial_phase": "Upstream Production Acceleration & Capital Discipline",
        "trial_readout_timeline": "Quarterly Upstream Barrel Production & Free Cash Flow Realization",
        "efficacy_summary": "World-class low-breakeven upstream deepwater Guyana and Permian assets with integrated refining and chemical cash flow stability.",
        "competitive_edge": "<$35/bbl breakeven inventory, Pioneer natural resources operational synergies, and pristine balance sheet liquidity.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Quarterly Guyana Yellowtail / Uaru FPSO Production Milestone", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Pioneer Permian Acreage Synergy Realization ($2B+ Annual Run Rate)", "impact": "High Positive"},
            {"date": "2027", "event": "Annual Dividend Aristocrat Payout Escalation & Buyback Tranche", "impact": "Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 355.0, "net_margin_pct": 11.0, "projected_eps": 8.90, "implied_pe": 13.5, "implied_target": 120.15},
            {"year": 2027, "revenue_billions": 385.0, "net_margin_pct": 12.0, "projected_eps": 10.80, "implied_pe": 13.0, "implied_target": 140.40},
            {"year": 2029, "revenue_billions": 415.0, "net_margin_pct": 12.8, "projected_eps": 12.90, "implied_pe": 12.5, "implied_target": 161.25},
            {"year": 2031, "revenue_billions": 445.0, "net_margin_pct": 13.2, "projected_eps": 15.10, "implied_pe": 12.0, "implied_target": 181.20}
        ]
    },
    "JPM": {
        "company_name": "JPMorgan Chase & Co.",
        "sector": "Financial Services / Commercial & Investment Banking",
        "primary_drug_trial": "Net Interest Income (NII) Resilience, Wealth Inflows & Global Corporate Banking",
        "trial_phase": "Fortress Balance Sheet Scale & Market Share Expansion",
        "trial_readout_timeline": "Quarterly Net Interest Margin & Credit Loss Readouts",
        "efficacy_summary": "Preeminent global banking franchise with fortress Tier-1 capital adequacy, diversified investment banking pipeline, and dominant market share.",
        "competitive_edge": "Vast low-cost deposit base, market-leading corporate financing franchise, and annual technology investment scale ($17B+).",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Quarterly Net Interest Income & Investment Banking Fee Recovery Report", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Asset & Wealth Management Net New Inflow Report", "impact": "Positive"},
            {"date": "2027", "event": "Federal Reserve CCAR Comprehensive Capital Analysis & Buyback Authorization", "impact": "Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 168.0, "net_margin_pct": 31.0, "projected_eps": 18.20, "implied_pe": 12.5, "implied_target": 227.50},
            {"year": 2027, "revenue_billions": 182.0, "net_margin_pct": 32.5, "projected_eps": 21.50, "implied_pe": 12.0, "implied_target": 258.00},
            {"year": 2029, "revenue_billions": 198.0, "net_margin_pct": 33.5, "projected_eps": 24.80, "implied_pe": 11.5, "implied_target": 285.20},
            {"year": 2031, "revenue_billions": 215.0, "net_margin_pct": 34.0, "projected_eps": 28.50, "implied_pe": 11.0, "implied_target": 313.50}
        ]
    },
    "LMT": {
        "company_name": "Lockheed Martin Corporation",
        "sector": "Aerospace & Defense / Tactical Systems",
        "primary_drug_trial": "F-35 Lightning II Production Ramp, PAC-3 Missile Defense & Hypersonics",
        "trial_phase": "Multi-Year DoD Procurement & NATO Partner Backlog Conversion",
        "trial_readout_timeline": "Quarterly Program Delivery & Defense Backlog Reporting",
        "efficacy_summary": "Premier aerospace and defense prime contractor holding multi-billion-dollar backlog for 5th-generation stealth fighters and precision munitions.",
        "competitive_edge": "Sole-source provider on critical US and allied strategic defense programs with multi-decade sustainment contracts.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "US Defense Appropriations Bill (NDAA) Multi-Year Procurement Appropriations", "impact": "High Strategic"},
            {"date": "Q4 2026", "event": "F-35 Tech Refresh-3 (TR-3) Full Capability Deployment Milestone", "impact": "High Positive"},
            {"date": "2027", "event": "PAC-3 Missile Interceptor Annual Production Capacity Scale to 650 Units", "impact": "Positive"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 71.5, "net_margin_pct": 9.5, "projected_eps": 28.50, "implied_pe": 18.0, "implied_target": 513.00},
            {"year": 2027, "revenue_billions": 77.0, "net_margin_pct": 10.2, "projected_eps": 33.00, "implied_pe": 17.5, "implied_target": 577.50},
            {"year": 2029, "revenue_billions": 83.5, "net_margin_pct": 10.8, "projected_eps": 38.20, "implied_pe": 17.0, "implied_target": 649.40},
            {"year": 2031, "revenue_billions": 90.0, "net_margin_pct": 11.2, "projected_eps": 43.80, "implied_pe": 16.5, "implied_target": 722.70}
        ]
    },
    "COST": {
        "company_name": "Costco Wholesale Corporation",
        "sector": "Consumer Defensive / Hypermarkets & Wholesale Clubs",
        "primary_drug_trial": "Membership Renewal Rate Dominance, E-Commerce Integration & Global Warehouse Expansion",
        "trial_phase": "Global Unit Expansion & Membership Retention",
        "trial_readout_timeline": "Monthly Comparable Sales (Comps) & Membership Fee Income",
        "efficacy_summary": "World-leading wholesale subscription model with >92% North American membership renewal and unmatched bulk purchasing leverage.",
        "competitive_edge": "Negative working capital cycle, high inventory velocity, Kirkland Signature private-label margin moat, and customer trust.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Monthly Comparable Sales (Comps) & Global Membership Fee Income Readout", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "International Warehouse Expansion (Asia / Europe 30-Unit Buildout)", "impact": "Positive"},
            {"date": "2027", "event": "Special Dividend Capital Return Tranche", "impact": "Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 255.0, "net_margin_pct": 2.8, "projected_eps": 16.80, "implied_pe": 48.0, "implied_target": 806.40},
            {"year": 2027, "revenue_billions": 282.0, "net_margin_pct": 3.0, "projected_eps": 20.20, "implied_pe": 44.0, "implied_target": 888.80},
            {"year": 2029, "revenue_billions": 312.0, "net_margin_pct": 3.2, "projected_eps": 24.50, "implied_pe": 40.0, "implied_target": 980.00},
            {"year": 2031, "revenue_billions": 345.0, "net_margin_pct": 3.3, "projected_eps": 29.20, "implied_pe": 36.0, "implied_target": 1051.20}
        ]
    },
    "CIEN": {
        "company_name": "Ciena Corporation",
        "sector": "Optical Networking & AI Interconnect Infrastructure",
        "primary_drug_trial": "WaveLogic 6 Extreme (WL6e) 1.6Tb/s Coherent Optical Interconnect",
        "trial_phase": "Hyperscaler Datacenter Fabric Deployment Phase",
        "trial_readout_timeline": "Quarterly Cloud Hyperscaler Interconnect Shipments",
        "efficacy_summary": "World-leading 1.6T single-wavelength optical transport delivering 50% power-per-bit reduction across cloud AI clusters.",
        "competitive_edge": "Proprietary 3nm DSP silicon design, optical routing patents, and multi-billion-dollar backlog from Tier-1 hyperscalers.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "WaveLogic 6 Commercial Volume Shipment Acceleration to Cloud Giants", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Coherent Routing & 800ZR/ZR+ Datacenter Interconnect Ramp", "impact": "High Positive"},
            {"date": "2027", "event": "Next-Gen 3.2Tb/s Coherent DSP Optical Architecture Announcement", "impact": "Strategic"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 4.50, "net_margin_pct": 12.5, "projected_eps": 3.60, "implied_pe": 20.0, "implied_target": 72.00},
            {"year": 2027, "revenue_billions": 5.40, "net_margin_pct": 15.0, "projected_eps": 5.20, "implied_pe": 18.0, "implied_target": 93.60},
            {"year": 2029, "revenue_billions": 6.50, "net_margin_pct": 16.5, "projected_eps": 7.10, "implied_pe": 16.5, "implied_target": 117.15},
            {"year": 2031, "revenue_billions": 7.80, "net_margin_pct": 17.5, "projected_eps": 9.20, "implied_pe": 15.0, "implied_target": 138.00}
        ]
    },
    "LLY": {
        "company_name": "Eli Lilly and Company",
        "sector": "Pharmaceuticals & Biotechnology",
        "primary_drug_trial": "Retatrutide (Triple GGG Receptor Agonist) & Orforglipron (Oral)",
        "trial_phase": "Phase 3 TRIUMPH Program",
        "trial_readout_timeline": "2026 - 2027",
        "efficacy_summary": "Up to 24.2% mean body weight loss at 48 weeks in Phase 2; highest recorded efficacy in obesity pharmacotherapy.",
        "competitive_edge": "Triple receptor mechanism (GLP-1, GIP, Glucagon) offering unprecedented metabolic liver & weight loss efficacy.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Federal Medicare & Medicaid Obesity Coverage Legislative Vote", "impact": "High Strategic"},
            {"date": "Q3 2026", "event": "Orforglipron Phase 3 Type-2 Diabetes readout", "impact": "High Positive"},
            {"date": "Q1 2027", "event": "Retatrutide Phase 3 Obesity primary endpoint completion", "impact": "Transformational"},
            {"date": "2028", "event": "Global commercial launch of Retatrutide", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 46.5, "net_margin_pct": 28.0, "projected_eps": 13.80, "implied_pe": 60.0, "implied_target": 828.00},
            {"year": 2027, "revenue_billions": 68.0, "net_margin_pct": 32.5, "projected_eps": 22.50, "implied_pe": 45.0, "implied_target": 1012.50},
            {"year": 2029, "revenue_billions": 94.0, "net_margin_pct": 35.0, "projected_eps": 32.80, "implied_pe": 38.0, "implied_target": 1246.40},
            {"year": 2031, "revenue_billions": 125.0, "net_margin_pct": 36.5, "projected_eps": 44.50, "implied_pe": 32.0, "implied_target": 1424.00}
        ]
    },
    "IREN": {
        "company_name": "IREN (Iris Energy Limited)",
        "sector": "Technology / AI Data Centers & Bitcoin Infrastructure",
        "primary_drug_trial": "AI Cloud GPU Cluster Deployments (NVIDIA H100/H200/Blackwell), Megawatt (MW) Power Scaling & Bitcoin Fleet Hash Rate Efficiency",
        "trial_phase": "HPC Power Interconnection & AI Cloud Capacity Scaling",
        "trial_readout_timeline": "Monthly Operating & Hash Rate Updates + Quarterly AI Cloud ARR Readouts",
        "efficacy_summary": "Next-generation hyperscale data center infrastructure powered by 100% renewable energy, delivering high-density GPU hosting and low-cost Bitcoin mining fleet operations.",
        "competitive_edge": "Secured multi-gigawatt power pipeline (e.g. 1.4GW Childress, TX site), low all-in electricity power costs, proprietary liquid cooling, and Tier 1 GPU colocation architecture.",
        "upcoming_milestones": [
            {"date": "Q3 2026", "event": "Childress 500MW Substation Energization & AI Cloud GPU Cluster Scaling", "impact": "High Positive"},
            {"date": "Q4 2026", "event": "Fleet Hash Rate Expansion to 30+ EH/s & NVIDIA Blackwell Infrastructure Deployment", "impact": "High Positive"},
            {"date": "2027", "event": "Enterprise Hyperscaler Multi-Year AI Compute Hosting Contracts", "impact": "Transformational"}
        ],
        "multi_year_forecast": [
            {"year": 2025, "revenue_billions": 0.48, "net_margin_pct": 35.0, "projected_eps": 1.80, "implied_pe": 25.0, "implied_target": 45.00},
            {"year": 2027, "revenue_billions": 0.95, "net_margin_pct": 38.0, "projected_eps": 3.50, "implied_pe": 22.0, "implied_target": 77.00},
            {"year": 2029, "revenue_billions": 1.65, "net_margin_pct": 40.0, "projected_eps": 6.20, "implied_pe": 20.0, "implied_target": 124.00},
            {"year": 2031, "revenue_billions": 2.50, "net_margin_pct": 42.0, "projected_eps": 9.80, "implied_pe": 18.0, "implied_target": 176.40}
        ]
    }
}


class CatalystEngine:
    """Analyze and generate upcoming product catalysts, clinical trials, and multi-year valuation trajectories."""

    def get_asset_catalyst_report(
        self,
        symbol: str,
        current_price: float = 100.0,
        sector: str = "",
        industry: str = "",
        company_name: str = ""
    ) -> Dict[str, Any]:
        upper = symbol.upper().replace("-USD", "").strip()

        if upper in ASSET_CATALYST_KNOWLEDGE:
            data = ASSET_CATALYST_KNOWLEDGE[upper].copy()
            data["symbol"] = upper
            data["current_price"] = current_price
            return data

        # Sector & Industry Context-Aware Dynamic Catalyst Synthesis
        clean_name = company_name or f"{upper} Corporation"
        sec_lower = sector.lower()
        ind_lower = industry.lower()

        # 1. Biotechnology, Pharmaceuticals & Life Sciences
        if any(w in sec_lower or w in ind_lower for w in ["biotech", "pharma", "health", "therapeutics", "drug", "medicine", "genomics", "biological"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Biotechnology & Pharmaceuticals",
                "primary_drug_trial": f"Pivotal Clinical Pipeline & Registrational Trials for {clean_name}",
                "trial_phase": "Phase 2/3 Registration & FDA NDA/BLA Filing Pipeline",
                "trial_readout_timeline": "Upcoming Clinical Readouts & Regulatory PDUFA Decisions",
                "efficacy_summary": f"Targeted biological mechanism, strong clinical biomarker efficacy, and significant unmet addressable patient population for {clean_name}.",
                "competitive_edge": "Proprietary molecular target IP, patent estate protection, and global strategic pharmaceutical commercial partnerships.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Pivotal Clinical Trial Primary Endpoint Data Readout", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "FDA & EMA Regulatory Filing Submission / Advisory Committee Review", "impact": "High Strategic"},
                    {"date": "2027", "event": "Commercial Product Launch & Global Payer Reimbursement Expansion", "impact": "Transformational"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.20, 1), "net_margin_pct": 22.0, "projected_eps": round(current_price * 0.035, 2), "implied_pe": 40.0, "implied_target": round(current_price * 1.15, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.40, 1), "net_margin_pct": 28.0, "projected_eps": round(current_price * 0.075, 2), "implied_pe": 32.0, "implied_target": round(current_price * 1.45, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.75, 1), "net_margin_pct": 34.0, "projected_eps": round(current_price * 0.130, 2), "implied_pe": 26.0, "implied_target": round(current_price * 1.95, 2)},
                    {"year": 2031, "revenue_billions": 1.25, "net_margin_pct": 38.0, "projected_eps": round(current_price * 0.210, 2), "implied_pe": 22.0, "implied_target": round(current_price * 2.60, 2)}
                ]
            }

        # 1. Logistics, Freight, Courier, Delivery, Shipping & Transportation
        if any(w in sec_lower or w in ind_lower for w in ["logistics", "freight", "transport", "shipping", "parcel", "courier", "delivery", "railroad", "airline", "cargo", "marine"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Industrials / Freight & Logistics Network",
                "primary_drug_trial": "Global Freight Rate Yields, Fleet Utilization & B2B Supply Chain Margin Optimization",
                "trial_phase": "Global Network Route Density & Dynamic Surcharge Execution",
                "trial_readout_timeline": "Quarterly Ocean/Air Freight Volume & Ton-Mile Readouts",
                "efficacy_summary": f"World-class multimodal cargo distribution, high automated sorting throughput, and disciplined fuel/capacity surcharges for {clean_name}.",
                "competitive_edge": "Established intermodal hubs, multi-continental fleet network, and high barriers to entry in global freight routing.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Freight Volume & Net Ton-Mile Yield Report", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Peak E-Commerce Volume Surcharge & Cross-Border Delivery Update", "impact": "High Positive"},
                    {"date": "2027", "event": "Automated Fleet Modernization & Route Optimization Efficiency Gains", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.45, 1), "net_margin_pct": 7.5, "projected_eps": round(current_price * 0.055, 2), "implied_pe": 14.0, "implied_target": round(current_price * 1.08, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.52, 1), "net_margin_pct": 8.2, "projected_eps": round(current_price * 0.068, 2), "implied_pe": 13.5, "implied_target": round(current_price * 1.20, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.60, 1), "net_margin_pct": 8.8, "projected_eps": round(current_price * 0.082, 2), "implied_pe": 13.0, "implied_target": round(current_price * 1.35, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.68, 1), "net_margin_pct": 9.2, "projected_eps": round(current_price * 0.098, 2), "implied_pe": 12.5, "implied_target": round(current_price * 1.50, 2)}
                ]
            }

        # 2. Aerospace, Defense Prime Contractors & Heavy Machinery
        if any(w in sec_lower or w in ind_lower for w in ["defense", "aerospace", "machinery", "industrial equipment", "conglomerate", "military", "aviation", "weapon"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Aerospace & Defense / Industrial Systems",
                "primary_drug_trial": "DoD Strategic Program Procurement, Defense Backlog Conversion & Delivery Milestones",
                "trial_phase": "Multi-Year Government Contract Procurement & Backlog Ramp",
                "trial_readout_timeline": "Quarterly Defense Appropriations & Program Delivery Readouts",
                "efficacy_summary": f"Critical prime contractor capabilities, multi-year sovereign defense order backlogs, and high-margin sustainment contracts for {clean_name}.",
                "competitive_edge": "Sole-source government defense credentials, classified R&D clearance, and long-term fleet maintenance programs.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Defense Appropriations Bill (NDAA) Multi-Year Procurement Review", "impact": "High Strategic"},
                    {"date": "Q4 2026", "event": "Strategic Program Milestone & Delivery Acceleration Report", "impact": "High Positive"},
                    {"date": "2027", "event": "Allied Sovereign Defense Partner Order Backlog Conversion", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.40, 1), "net_margin_pct": 10.5, "projected_eps": round(current_price * 0.052, 2), "implied_pe": 18.0, "implied_target": round(current_price * 1.10, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.46, 1), "net_margin_pct": 11.2, "projected_eps": round(current_price * 0.065, 2), "implied_pe": 17.5, "implied_target": round(current_price * 1.25, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.53, 1), "net_margin_pct": 11.8, "projected_eps": round(current_price * 0.080, 2), "implied_pe": 17.0, "implied_target": round(current_price * 1.42, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.61, 1), "net_margin_pct": 12.2, "projected_eps": round(current_price * 0.098, 2), "implied_pe": 16.5, "implied_target": round(current_price * 1.62, 2)}
                ]
            }

        # 3. Healthcare, Medical Devices, Diagnostics & Health Providers
        if any(w in sec_lower or w in ind_lower for w in ["medical device", "diagnostics", "hospital", "managed care", "health services", "surgical", "healthcare"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Healthcare & Medical Technologies",
                "primary_drug_trial": "FDA 510(k) Clearances, Procedure Volume Acceleration & Commercial Payer Coverage",
                "trial_phase": "Hospital System Rollout & Clinical Procedure Scale",
                "trial_readout_timeline": "Quarterly Surgical Procedure Volume & Commercial Adoption Readouts",
                "efficacy_summary": f"Proprietary medical technology, high procedure utilization rates, and expanding private/Medicare reimbursement coverage for {clean_name}.",
                "competitive_edge": "Surgeon training switching costs, proprietary patented instruments, and broad hospital purchasing agreements.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Procedure Volume & Recurring Consumables Growth", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Next-Gen Medical Device FDA / CE Mark Regulatory Submission", "impact": "High Strategic"},
                    {"date": "2027", "event": "International Hospital Adoption & Reimbursement Rate Expansion", "impact": "Positive"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.30, 1), "net_margin_pct": 20.0, "projected_eps": round(current_price * 0.045, 2), "implied_pe": 26.0, "implied_target": round(current_price * 1.12, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.38, 1), "net_margin_pct": 22.5, "projected_eps": round(current_price * 0.062, 2), "implied_pe": 24.0, "implied_target": round(current_price * 1.35, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.48, 1), "net_margin_pct": 24.5, "projected_eps": round(current_price * 0.082, 2), "implied_pe": 22.0, "implied_target": round(current_price * 1.65, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.60, 1), "net_margin_pct": 26.0, "projected_eps": round(current_price * 0.110, 2), "implied_pe": 20.0, "implied_target": round(current_price * 2.05, 2)}
                ]
            }

        # 4. Semiconductors, Hardware & Advanced Optics
        if any(w in sec_lower or w in ind_lower for w in ["semiconductor", "chip", "hardware", "networking", "optical", "server", "equipment", "wafer"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Semiconductors & AI Hardware",
                "primary_drug_trial": f"Next-Gen Silicon Architecture & Cloud Datacenter Deployment for {clean_name}",
                "trial_phase": "Enterprise Hyperscaler Mass Production Ramp",
                "trial_readout_timeline": "Quarterly Hyperscaler Interconnect & Compute Shipments",
                "efficacy_summary": f"Leading performance-per-watt metrics, proprietary architecture moats, and strong hyperscaler capital expenditure demand for {clean_name}.",
                "competitive_edge": "Advanced packaging IP, high-bandwidth interconnects, and deeply integrated tier-1 customer contracts.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Next-Gen Node Commercial Volume Shipment Acceleration", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Enterprise Hyperscaler Datacenter Capacity Integration", "impact": "High Positive"},
                    {"date": "2027", "event": "Next-Generation Architecture Platform Unveiling", "impact": "Transformational"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.50, 1), "net_margin_pct": 28.0, "projected_eps": round(current_price * 0.055, 2), "implied_pe": 28.0, "implied_target": round(current_price * 1.14, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.70, 1), "net_margin_pct": 30.5, "projected_eps": round(current_price * 0.080, 2), "implied_pe": 25.0, "implied_target": round(current_price * 1.48, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.95, 1), "net_margin_pct": 32.0, "projected_eps": round(current_price * 0.115, 2), "implied_pe": 22.0, "implied_target": round(current_price * 1.95, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 1.30, 1), "net_margin_pct": 33.0, "projected_eps": round(current_price * 0.160, 2), "implied_pe": 20.0, "implied_target": round(current_price * 2.45, 2)}
                ]
            }

        # 5. Enterprise Software, Cloud & Cybersecurity
        if any(w in sec_lower or w in ind_lower for w in ["software", "cloud", "security", "saas", "cyber", "internet", "database", "ai"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Enterprise Software & Cloud Platforms",
                "primary_drug_trial": f"Enterprise AI Monetization & Net Expansion (NRR) Acceleration for {clean_name}",
                "trial_phase": "Enterprise Platform Adoption & High-Margin Recurring ACV Scale",
                "trial_readout_timeline": "Quarterly Annual Recurring Revenue (ARR) & Net Retention Readouts",
                "efficacy_summary": f"High recurring subscription gross margins (>80%), low churn, and expanding enterprise average contract value for {clean_name}.",
                "competitive_edge": "High customer switching costs, workflow data stickiness, and expanding platform integration ecosystem.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly ARR & Enterprise Customer Net Retention Report", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Native Enterprise AI Agent Module General Availability", "impact": "High Positive"},
                    {"date": "2027", "event": "Global Large-Enterprise Expansion & Federal Sector Procurement", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.35, 1), "net_margin_pct": 24.0, "projected_eps": round(current_price * 0.040, 2), "implied_pe": 35.0, "implied_target": round(current_price * 1.16, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.50, 1), "net_margin_pct": 27.5, "projected_eps": round(current_price * 0.065, 2), "implied_pe": 30.0, "implied_target": round(current_price * 1.50, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.70, 1), "net_margin_pct": 30.0, "projected_eps": round(current_price * 0.095, 2), "implied_pe": 26.0, "implied_target": round(current_price * 2.05, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 1.00, 1), "net_margin_pct": 32.0, "projected_eps": round(current_price * 0.140, 2), "implied_pe": 22.0, "implied_target": round(current_price * 2.70, 2)}
                ]
            }

        # 6. Consumer Staples, Food, Beverage & Supermarkets
        if any(w in sec_lower or w in ind_lower for w in ["beverage", "drink", "food", "tobacco", "staple", "consumer defensive", "supermarket", "grocery"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Consumer Defensive / Staples",
                "primary_drug_trial": "Global Volume Growth, Direct-Store-Delivery & Pricing Power Expansion",
                "trial_phase": "Commercial Market Distribution Scaling",
                "trial_readout_timeline": "Quarterly Unit Volume & Pricing Power Readouts",
                "efficacy_summary": f"High operational stability, resilient brand pricing power, and consistent cash dividend conversion for {clean_name}.",
                "competitive_edge": "Established global retail shelf distribution, supply chain logistics, and high customer brand loyalty.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Unit Volume & Price/Mix Net Sales Readout", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Emerging Market Distribution & Product Portfolio Expansion", "impact": "Positive"},
                    {"date": "2027", "event": "Annual Capital Return & Dividend Policy Reaffirmation", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.45, 1), "net_margin_pct": 22.0, "projected_eps": round(current_price * 0.045, 2), "implied_pe": 23.0, "implied_target": round(current_price * 1.08, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.52, 1), "net_margin_pct": 23.0, "projected_eps": round(current_price * 0.055, 2), "implied_pe": 22.0, "implied_target": round(current_price * 1.22, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.60, 1), "net_margin_pct": 24.0, "projected_eps": round(current_price * 0.068, 2), "implied_pe": 21.0, "implied_target": round(current_price * 1.42, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.70, 1), "net_margin_pct": 25.0, "projected_eps": round(current_price * 0.082, 2), "implied_pe": 20.0, "implied_target": round(current_price * 1.64, 2)}
                ]
            }

        # 7. Consumer Discretionary, Luxury, Retail, Restaurants & Auto/EV
        if any(w in sec_lower or w in ind_lower for w in ["restaurant", "coffee", "retail", "consumer cyclical", "apparel", "auto", "vehicle", "luxury", "hotel", "travel", "leisure"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Consumer Discretionary / Retail & Services",
                "primary_drug_trial": "Same-Store Sales (Comps), Store-Level Throughput & Direct-to-Consumer (DTC) Mix",
                "trial_phase": "Unit Economics & Digital Membership Acceleration",
                "trial_readout_timeline": "Quarterly Global Comparable Sales Readouts",
                "efficacy_summary": f"High recurring customer transaction frequency, digital rewards growth, and operational store efficiency for {clean_name}.",
                "competitive_edge": "High-frequency consumer brand affinity, digital ordering ecosystem, and prime retail location footprint.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Same-Store Sales (Comps) & Store Throughput Readout", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "International Market Expansion & Store Format Modernization", "impact": "High Positive"},
                    {"date": "2027", "event": "Digital App & Loyalty Tier Feature Modernization", "impact": "Positive"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.38, 1), "net_margin_pct": 14.0, "projected_eps": round(current_price * 0.042, 2), "implied_pe": 24.0, "implied_target": round(current_price * 1.10, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.45, 1), "net_margin_pct": 15.5, "projected_eps": round(current_price * 0.054, 2), "implied_pe": 22.0, "implied_target": round(current_price * 1.25, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.53, 1), "net_margin_pct": 16.5, "projected_eps": round(current_price * 0.068, 2), "implied_pe": 20.0, "implied_target": round(current_price * 1.45, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.62, 1), "net_margin_pct": 17.5, "projected_eps": round(current_price * 0.085, 2), "implied_pe": 19.0, "implied_target": round(current_price * 1.70, 2)}
                ]
            }

        # 8. Real Estate, Industrial Logistics & Data Center REITs
        if any(w in sec_lower or w in ind_lower for w in ["reit", "real estate", "property", "lease", "mortgage", "housing"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Real Estate Investment Trusts (REITs)",
                "primary_drug_trial": "Adjusted Funds From Operations (AFFO) Growth & Portfolio Occupancy",
                "trial_phase": "Capital Recycling & Net Lease Execution",
                "trial_readout_timeline": "Quarterly AFFO Payout & Lease Renewal Reporting",
                "efficacy_summary": f"High-quality commercial real estate portfolio with long-term tenant leases and inflation-hedged dividend cash flows for {clean_name}.",
                "competitive_edge": "Prime location real estate assets, investment-grade tenant base, and disciplined debt maturity ladder.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly AFFO Per Share & Same-Store NOI Growth Readout", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Strategic Property Acquisition & Capital Recycling Update", "impact": "Positive"},
                    {"date": "2027", "event": "Annual Dividend Distribution & Lease Renewal Review", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.25, 1), "net_margin_pct": 32.0, "projected_eps": round(current_price * 0.052, 2), "implied_pe": 18.0, "implied_target": round(current_price * 1.06, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.29, 1), "net_margin_pct": 33.5, "projected_eps": round(current_price * 0.060, 2), "implied_pe": 17.5, "implied_target": round(current_price * 1.15, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.34, 1), "net_margin_pct": 34.5, "projected_eps": round(current_price * 0.070, 2), "implied_pe": 17.0, "implied_target": round(current_price * 1.28, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.40, 1), "net_margin_pct": 35.0, "projected_eps": round(current_price * 0.082, 2), "implied_pe": 16.5, "implied_target": round(current_price * 1.42, 2)}
                ]
            }

        # 9. Energy, Oil & Gas, Refining & Clean Utilities
        if any(w in sec_lower or w in ind_lower for w in ["energy", "oil", "gas", "petroleum", "solar", "wind", "utility", "power", "pipeline", "refining", "e&p"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Energy & Utilities",
                "primary_drug_trial": "Upstream Production Efficiency, LNG Export Expansion & Free Cash Flow Yield",
                "trial_phase": "Capital Discipline & Infrastructure Utilization",
                "trial_readout_timeline": "Quarterly Barrel Equivalents & Dividend/Buyback Updates",
                "efficacy_summary": f"Disciplined capital allocation, low-cost extraction assets, and resilient commodity cash conversion for {clean_name}.",
                "competitive_edge": "Low breakeven cost per barrel/unit, deep proven reserve base, and integrated midstream export infrastructure.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Upstream Production & Free Cash Flow Realization", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Exploration Drilling Results & Midstream Capacity Ramp", "impact": "Positive"},
                    {"date": "2027", "event": "Shareholder Capital Return & Base Dividend Escalation", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.50, 1), "net_margin_pct": 18.0, "projected_eps": round(current_price * 0.075, 2), "implied_pe": 12.0, "implied_target": round(current_price * 1.08, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.58, 1), "net_margin_pct": 19.5, "projected_eps": round(current_price * 0.090, 2), "implied_pe": 11.5, "implied_target": round(current_price * 1.20, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.67, 1), "net_margin_pct": 20.5, "projected_eps": round(current_price * 0.110, 2), "implied_pe": 11.0, "implied_target": round(current_price * 1.35, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.78, 1), "net_margin_pct": 21.0, "projected_eps": round(current_price * 0.130, 2), "implied_pe": 10.5, "implied_target": round(current_price * 1.50, 2)}
                ]
            }

        # 10. AI Hyperscale Cloud Data Centers, High-Performance Compute (HPC) & Bitcoin Mining Infrastructure
        if upper in {"IREN", "MARA", "RIOT", "CLSK", "CORZ", "WULF", "CIFR", "MSTR", "COIN", "HUT", "BTBT", "BITF"} or any(w in sec_lower or w in ind_lower for w in ["bitcoin", "crypto", "digital asset", "blockchain", "hash rate", "exahash", "hpc", "gpu cloud", "iris energy"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Technology / AI Cloud & Digital Infrastructure",
                "primary_drug_trial": "AI Cloud GPU Cluster Deployment (NVIDIA H100/Blackwell), Megawatt (MW) Data Center Power Capacity & Bitcoin Hash Rate (EH/s) Fleet Efficiency",
                "trial_phase": "Hyperscale AI Cloud Fleet Buildout & Interconnection Capacity Scaling",
                "trial_readout_timeline": "Monthly Operating & Hash Rate Updates + Quarterly AI Cloud ARR Readouts",
                "efficacy_summary": f"Next-generation hyperscale compute infrastructure powered by low-cost renewable energy, delivering high-density GPU hosting and resilient fleet mining margins for {clean_name}.",
                "competitive_edge": "Secured multi-gigawatt power interconnection pipeline, proprietary liquid-cooling architecture, Tier-1 GPU colocation contracts, and low-quartile electricity cost structure.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Hyperscale Substation Energization & AI Cloud GPU Cluster Scaling", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Fleet Hash Rate Expansion Milestone (EH/s) & Enterprise Cloud ARR Acceleration", "impact": "High Positive"},
                    {"date": "2027", "event": "Multi-Year Enterprise Hyperscaler AI Compute Hosting Contracts", "impact": "Transformational"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.32, 1), "net_margin_pct": 32.0, "projected_eps": round(current_price * 0.052, 2), "implied_pe": 24.0, "implied_target": round(current_price * 1.15, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.48, 1), "net_margin_pct": 36.0, "projected_eps": round(current_price * 0.088, 2), "implied_pe": 22.0, "implied_target": round(current_price * 1.48, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.72, 1), "net_margin_pct": 38.5, "projected_eps": round(current_price * 0.140, 2), "implied_pe": 20.0, "implied_target": round(current_price * 2.10, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 1.05, 1), "net_margin_pct": 40.0, "projected_eps": round(current_price * 0.220, 2), "implied_pe": 18.0, "implied_target": round(current_price * 2.95, 2)}
                ]
            }

        # 11. Materials, Physical Mining, Chemicals & Commodities
        if any(w in sec_lower or w in ind_lower for w in ["material", "gold", "copper", "steel", "chemical", "metal", "lithium", "aluminum", "paper", "mineral", "ore", "smelting", "timber"]) or (("mining" in sec_lower or "mining" in ind_lower) and not any(c in ind_lower for c in ["bitcoin", "crypto", "digital"])):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Materials & Mining",
                "primary_drug_trial": "All-In Sustaining Cost (AISC) Margin Optimization & Mineral Reserve Life",
                "trial_phase": "Tier-1 Mine Production & Smelting Operations",
                "trial_readout_timeline": "Quarterly Ore Grade & Ton Yield Reporting",
                "efficacy_summary": f"Tier-1 low-cost extraction assets with multi-decade reserve life and strong commodity cycle leverage for {clean_name}.",
                "competitive_edge": "Low-quartile cost curve positioning, vast proven and probable mineral reserves, and strong balance sheet liquidity.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly All-In Sustaining Cost (AISC) & Production Volume Report", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "New Deposit Feasibility Study & Extraction Permit Readout", "impact": "Positive"},
                    {"date": "2027", "event": "Processing Facility Expansion & Mill Throughput Optimization", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.42, 1), "net_margin_pct": 16.0, "projected_eps": round(current_price * 0.055, 2), "implied_pe": 15.0, "implied_target": round(current_price * 1.07, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.49, 1), "net_margin_pct": 17.5, "projected_eps": round(current_price * 0.068, 2), "implied_pe": 14.5, "implied_target": round(current_price * 1.18, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.57, 1), "net_margin_pct": 18.5, "projected_eps": round(current_price * 0.082, 2), "implied_pe": 14.0, "implied_target": round(current_price * 1.32, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.66, 1), "net_margin_pct": 19.0, "projected_eps": round(current_price * 0.098, 2), "implied_pe": 13.5, "implied_target": round(current_price * 1.48, 2)}
                ]
            }

        # 11. Financials, Commercial Banking, Payments & Asset Management
        if any(w in sec_lower or w in ind_lower for w in ["financial", "bank", "credit", "insurance", "broker", "asset management", "capital markets", "payment", "wealth", "fintech"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Financial Services",
                "primary_drug_trial": "Net Interest Margin (NIM) Expansion & Fee Asset Under Management Growth",
                "trial_phase": "Capital Management & Prime Lending Scale",
                "trial_readout_timeline": "Quarterly Net Interest Income & Credit Loss Readouts",
                "efficacy_summary": f"Fortress capital adequacy, diversified institutional fee revenue, and prudent credit underwriting for {clean_name}.",
                "competitive_edge": "Low-cost core deposit franchise, leading market share across commercial lending, and rigorous risk controls.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Net Interest Income & Tier-1 Capital Ratio Report", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Wealth Management & Fee-Based Asset Inflow Readout", "impact": "Positive"},
                    {"date": "2027", "event": "Annual Stress Test (CCAR) Approval & Buyback Authorization", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.60, 1), "net_margin_pct": 26.0, "projected_eps": round(current_price * 0.085, 2), "implied_pe": 12.5, "implied_target": round(current_price * 1.09, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.69, 1), "net_margin_pct": 27.5, "projected_eps": round(current_price * 0.102, 2), "implied_pe": 12.0, "implied_target": round(current_price * 1.22, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.79, 1), "net_margin_pct": 28.5, "projected_eps": round(current_price * 0.122, 2), "implied_pe": 11.5, "implied_target": round(current_price * 1.38, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.90, 1), "net_margin_pct": 29.0, "projected_eps": round(current_price * 0.145, 2), "implied_pe": 11.0, "implied_target": round(current_price * 1.55, 2)}
                ]
            }

        # 12. Telecommunications, Media, Entertainment & Gaming
        if any(w in sec_lower or w in ind_lower for w in ["telecom", "media", "entertainment", "cable", "streaming", "broadcasting", "gaming"]):
            return {
                "symbol": upper,
                "company_name": clean_name,
                "sector": sector or "Communication Services / Media & Telecom",
                "primary_drug_trial": "Direct-to-Consumer (DTC) Streaming ARPU, Postpaid Subscriber Net Adds & Content Monetization",
                "trial_phase": "Subscriber Monetization & Content Production Discipline",
                "trial_readout_timeline": "Quarterly Postpaid Subscriber Adds & Streaming ARPU Reporting",
                "efficacy_summary": f"Strong proprietary content library, resilient subscriber retention, and expanding direct-to-consumer advertising monetization for {clean_name}.",
                "competitive_edge": "Iconic media franchises, nationwide 5G/fiber network infrastructure, and high recurring subscriber billing.",
                "upcoming_milestones": [
                    {"date": "Q3 2026", "event": "Quarterly Direct-to-Consumer (DTC) Streaming Operating Profit Milestone", "impact": "High Positive"},
                    {"date": "Q4 2026", "event": "Postpaid Wireless & Broadband Subscriber Net Add Acceleration", "impact": "Positive"},
                    {"date": "2027", "event": "Global Content Franchise Distribution & Ad-Tier Expansion", "impact": "Strategic"}
                ],
                "multi_year_forecast": [
                    {"year": 2025, "revenue_billions": round(current_price * 0.40, 1), "net_margin_pct": 15.0, "projected_eps": round(current_price * 0.048, 2), "implied_pe": 16.0, "implied_target": round(current_price * 1.08, 2)},
                    {"year": 2027, "revenue_billions": round(current_price * 0.46, 1), "net_margin_pct": 16.5, "projected_eps": round(current_price * 0.060, 2), "implied_pe": 15.5, "implied_target": round(current_price * 1.20, 2)},
                    {"year": 2029, "revenue_billions": round(current_price * 0.54, 1), "net_margin_pct": 17.5, "projected_eps": round(current_price * 0.075, 2), "implied_pe": 15.0, "implied_target": round(current_price * 1.36, 2)},
                    {"year": 2031, "revenue_billions": round(current_price * 0.62, 1), "net_margin_pct": 18.0, "projected_eps": round(current_price * 0.092, 2), "implied_pe": 14.5, "implied_target": round(current_price * 1.55, 2)}
                ]
            }

        # Generic quantitative catalyst generation for multi-asset stocks
        return {
            "symbol": upper,
            "company_name": clean_name,
            "sector": sector or "Multi-Asset Equities",
            "primary_drug_trial": "Commercial Operating Margin Expansion & Free Cash Flow Compounding",
            "trial_phase": "Operational Execution & Market Scale",
            "trial_readout_timeline": "Quarterly Earnings & Capital Allocation Guidance",
            "efficacy_summary": f"Strong operational execution, revenue compounding, and free cash flow generation for {clean_name}.",
            "competitive_edge": "Ecosystem network effects, operational efficiency, and high customer switching costs.",
            "upcoming_milestones": [
                {"date": "Q3 2026", "event": "Quarterly Earnings & Forward Operating Margin Guidance", "impact": "Medium-to-High"},
                {"date": "Q4 2026", "event": "Core Business Line Expansion & Volume Growth", "impact": "High Positive"},
                {"date": "2027", "event": "International Market Expansion & Capital Return Program", "impact": "Positive"}
            ],
            "multi_year_forecast": [
                {"year": 2025, "revenue_billions": round(current_price * 0.4, 1), "net_margin_pct": 25.0, "projected_eps": round(current_price * 0.04, 2), "implied_pe": 28.0, "implied_target": round(current_price * 1.12, 2)},
                {"year": 2027, "revenue_billions": round(current_price * 0.55, 1), "net_margin_pct": 27.5, "projected_eps": round(current_price * 0.06, 2), "implied_pe": 26.0, "implied_target": round(current_price * 1.45, 2)},
                {"year": 2029, "revenue_billions": round(current_price * 0.75, 1), "net_margin_pct": 29.0, "projected_eps": round(current_price * 0.085, 2), "implied_pe": 24.0, "implied_target": round(current_price * 1.95, 2)},
                {"year": 2031, "revenue_billions": round(current_price * 1.05, 1), "net_margin_pct": 30.0, "projected_eps": round(current_price * 0.12, 2), "implied_pe": 22.0, "implied_target": round(current_price * 2.50, 2)}
            ]
        }
