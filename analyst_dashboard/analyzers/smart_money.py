"""High-Density Institutional & Congressional Smart Money Feed Engine."""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

CONGRESSIONAL_TRADES: List[Dict[str, Any]] = [
    # --- 7D RECENT (Aug 22 - Aug 28, 2026) ---
    {
        "id": "cong-pltr-1",
        "politician": "Dan Crenshaw (R-TX)",
        "chamber": "House",
        "ticker": "PLTR",
        "asset_name": "Palantir Technologies",
        "sector": "Defense & Enterprise AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$50,000 - $100,000",
        "filing_date": "2026-08-27",
        "transaction_date": "2026-08-15",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 12,
        "performance_since_pct": 12.1,
        "sentiment": "Strong Bullish",
        "signal_strength": 95,
        "details": {
            "committee_assignments": ["Energy & Commerce", "House Permanent Select Committee on Intelligence"],
            "legislative_conflict_thesis": "Direct oversight of intelligence community software procurement and defense AI telemetry systems.",
            "historical_win_rate_pct": 74.0,
            "annualized_tech_alpha_pct": 29.4,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Rapid filing within 12 days of execution.",
            "key_catalyst": "AIP platform commercial bootcamps conversion and US Army TITAN contract ramp.",
        },
    },
    {
        "id": "cong-late-nvda",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors & AI Hardware",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-08-26",
        "transaction_date": "2026-06-29",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 58,
        "performance_since_pct": 21.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "committee_assignments": ["Senate Armed Services"],
            "legislative_conflict_thesis": "Late disclosure past 45-day statutory STOCK Act deadline following NDAA high-performance computing appropriations vote.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Non-compliant (58 days latency vs 45-day statutory limit).",
            "key_catalyst": "Blackwell enterprise deployment and DoD classified AI cluster orders.",
        },
    },
    {
        "id": "cong-late-tsla",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "TSLA",
        "asset_name": "Tesla Inc.",
        "sector": "Autonomous Mobility & AI Compute",
        "transaction_type": "Sale (Partial)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-25",
        "transaction_date": "2026-07-07",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 49,
        "performance_since_pct": -4.2,
        "sentiment": "Bearish / Trimming",
        "signal_strength": 80,
        "details": {
            "committee_assignments": ["Armed Services (Cyber & Innovative Technologies)", "Oversight & Accountability"],
            "legislative_conflict_thesis": "Late filing exceeding statutory window ahead of federal autonomous vehicle regulatory hearings.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Non-compliant (49 days latency vs 45-day statutory limit).",
            "key_catalyst": "FSD V13 regulatory approval timeline and Robotaxi autonomous permit reviews.",
        },
    },
    {
        "id": "cong-crwd-1",
        "politician": "John Curtis (R-UT)",
        "chamber": "House",
        "ticker": "CRWD",
        "asset_name": "CrowdStrike Holdings",
        "sector": "Cybersecurity",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-24",
        "transaction_date": "2026-08-11",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 13,
        "performance_since_pct": 16.5,
        "sentiment": "Strong Bullish",
        "signal_strength": 93,
        "details": {
            "committee_assignments": ["Energy & Commerce", "Communications & Technology"],
            "legislative_conflict_thesis": "CISA and SEC mandatory 4-day cyber incident reporting requirements expanding enterprise Falcon platform lock-in.",
            "historical_win_rate_pct": 72.4,
            "annualized_tech_alpha_pct": 24.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Prompt filing.",
            "key_catalyst": "Resilient multi-module ARR expansion and cloud security platform consolidation.",
        },
    },
    {
        "id": "cong-vrt-1",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "VRT",
        "asset_name": "Vertiv Holdings Co",
        "sector": "AI Datacenter Power & Liquid Cooling",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-08-23",
        "transaction_date": "2026-08-08",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 18.2,
        "sentiment": "Strong Bullish",
        "signal_strength": 97,
        "details": {
            "committee_assignments": ["Senate Armed Services", "Veterans' Affairs"],
            "legislative_conflict_thesis": "Federal power grid resilience initiatives and high-density liquid cooling standards for sovereign compute facilities.",
            "historical_win_rate_pct": 68.0,
            "annualized_tech_alpha_pct": 28.6,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Direct-to-chip liquid cooling backlog expansion supporting NVIDIA Blackwell & Rubin clusters.",
        },
    },
    {
        "id": "cong-lly-1",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "LLY",
        "asset_name": "Eli Lilly & Co.",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-22",
        "transaction_date": "2026-08-07",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 5.2,
        "sentiment": "Bullish",
        "signal_strength": 88,
        "details": {
            "committee_assignments": ["Senate Armed Services", "Agriculture, Nutrition & Forestry", "Veterans' Affairs"],
            "legislative_conflict_thesis": "VA healthcare system drug formulary inclusion and domestic pharmaceutical manufacturing tax incentives.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Filed within statutory 45-day window (15 days lag).",
            "key_catalyst": "Retatrutide Phase 3 triple-hormone agonist weight loss trial readouts.",
        },
    },

    # --- 30D HORIZON (July 29 - Aug 21, 2026) --- (Cumulative ~14)
    {
        "id": "cong-tsm-1",
        "politician": "Mark Green (R-TN)",
        "chamber": "House",
        "ticker": "TSM",
        "asset_name": "Taiwan Semiconductor Mfg",
        "sector": "Semiconductors / Foundry",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$500,000 - $1,000,000",
        "filing_date": "2026-08-19",
        "transaction_date": "2026-08-04",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 9.3,
        "sentiment": "Strong Bullish",
        "signal_strength": 91,
        "details": {
            "committee_assignments": ["Homeland Security (Chairman)", "Foreign Affairs"],
            "legislative_conflict_thesis": "Critical oversight on Pacific supply chains and domestic CHIPS Act Arizona fab direct grant disbursements.",
            "historical_win_rate_pct": 73.2,
            "annualized_tech_alpha_pct": 25.1,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Timely compliance within standard window.",
            "key_catalyst": "2nm gate-all-around high volume manufacturing ramp and Arizona Fab 1 full production certification.",
        },
    },
    {
        "id": "cong-nvo-1",
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "NVO",
        "asset_name": "Novo Nordisk A/S",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-08-18",
        "transaction_date": "2026-08-02",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 6.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "committee_assignments": ["Foreign Affairs Committee (Chairman)", "Homeland Security"],
            "legislative_conflict_thesis": "Transatlantic pharmaceutical supply chain discussions and federal healthcare Medicare GLP-1 reimbursement expansion deliberations.",
            "historical_win_rate_pct": 71.0,
            "annualized_tech_alpha_pct": 22.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Filed within statutory 45-day window (16 days lag).",
            "key_catalyst": "Phase 3 Amycretin oral pill clinical readout and Catalent fill-finish manufacturing expansion.",
        },
    },
    {
        "id": "cong-ge-1",
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "GE",
        "asset_name": "GE Aerospace",
        "sector": "Aerospace & Defense Propulsion",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-18",
        "transaction_date": "2026-08-02",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 8.1,
        "sentiment": "Strong Bullish",
        "signal_strength": 90,
        "details": {
            "committee_assignments": ["Foreign Affairs (Chairman)"],
            "legislative_conflict_thesis": "Foreign military sales approvals (FMS) for F-35/F-15 engine programs and commercial widebody aftermarket cycles.",
            "historical_win_rate_pct": 71.0,
            "annualized_tech_alpha_pct": 22.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Standard statutory window.",
            "key_catalyst": "High-margin commercial aftermarket LEAP shop visits and next-gen adaptive military engine contracts.",
        },
    },
    {
        "id": "cong-msft-1",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "MSFT",
        "asset_name": "Microsoft Corp.",
        "sector": "Enterprise Cloud & AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$500,000 - $1,000,000",
        "filing_date": "2026-08-15",
        "transaction_date": "2026-07-30",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 3.8,
        "sentiment": "Bullish",
        "signal_strength": 86,
        "details": {
            "committee_assignments": ["Armed Services (Cyber & Innovative Technologies)", "Oversight & Accountability"],
            "legislative_conflict_thesis": "Federal Department of Defense enterprise cloud modernization and cybersecurity contract renewal cycles.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Filed within statutory 45-day window (16 days lag).",
            "key_catalyst": "Azure OpenAI enterprise copilot ARR expansion and Pentagon multi-cloud contract deployments.",
        },
    },
    {
        "id": "cong-amd-1",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "AMD",
        "asset_name": "Advanced Micro Devices",
        "sector": "Semiconductors / AI Accelerators",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-08-15",
        "transaction_date": "2026-07-30",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 7.9,
        "sentiment": "Bullish",
        "signal_strength": 89,
        "details": {
            "committee_assignments": ["Armed Services (Cyber & Innovative Technologies)"],
            "legislative_conflict_thesis": "Open-source ROCm AI software ecosystem promotion and Department of Energy supercomputing grants.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Standard filing window.",
            "key_catalyst": "MI350X & MI400 accelerator deliveries to major Tier-2 cloud and hyperscaler accounts.",
        },
    },
    {
        "id": "cong-nvda-1",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-08-14",
        "transaction_date": "2026-07-28",
        "strike_price": "$180 Calls",
        "days_to_filing": 17,
        "performance_since_pct": 14.8,
        "sentiment": "Strong Bullish",
        "signal_strength": 98,
        "details": {
            "committee_assignments": ["Former Speaker", "House Democratic Leadership", "Appropriations (Prior)"],
            "legislative_conflict_thesis": "Strategic timing ahead of federal AI compute export rule revisions and next-generation datacenter infrastructure appropriations.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Filed within statutory 45-day window (17 days lag).",
            "key_catalyst": "Massive sovereign AI datacenter orders and enterprise Rubin GPU ramp.",
        },
    },
    {
        "id": "cong-avgo-1",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "AVGO",
        "asset_name": "Broadcom Inc.",
        "sector": "Semiconductors & Custom ASIC",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-08-14",
        "transaction_date": "2026-07-28",
        "strike_price": "$200 Calls",
        "days_to_filing": 17,
        "performance_since_pct": 11.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 96,
        "details": {
            "committee_assignments": ["Former Speaker", "Democratic Leadership"],
            "legislative_conflict_thesis": "Custom hyperscaler AI silicon buildouts for OpenAI, Meta, and Google under federal connectivity mandates.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Routine disclosure.",
            "key_catalyst": "3nm custom XPU ASIC deployment for Tier-1 hyperscalers and VMware cloud recurring synergies.",
        },
    },
    {
        "id": "cong-aapl-1",
        "politician": "Josh Gottheimer (D-NJ)",
        "chamber": "House",
        "ticker": "AAPL",
        "asset_name": "Apple Inc.",
        "sector": "Consumer Tech & Services",
        "transaction_type": "Sale (Partial)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-08-10",
        "transaction_date": "2026-07-25",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": -1.2,
        "sentiment": "Neutral / Profit Take",
        "signal_strength": 65,
        "details": {
            "committee_assignments": ["Financial Services (Capital Markets)", "House Permanent Select Committee on Intelligence"],
            "legislative_conflict_thesis": "Rebalancing portfolio risk ahead of antitrust deliberations on mobile app store fee structures.",
            "historical_win_rate_pct": 66.8,
            "annualized_tech_alpha_pct": 16.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Standard filing window.",
            "key_catalyst": "Apple Intelligence subscription ramp vs consumer upgrade replacement duration.",
        },
    },

    # --- 90D HORIZON (May 30 - July 28, 2026) --- (Cumulative ~28)
    {
        "id": "cong-90d-nvda-2",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$2,000,000 - $5,000,000",
        "filing_date": "2026-07-15",
        "transaction_date": "2026-06-28",
        "strike_price": "$140 Calls",
        "days_to_filing": 17,
        "performance_since_pct": 28.5,
        "sentiment": "Strong Bullish",
        "signal_strength": 98,
        "details": {
            "committee_assignments": ["Democratic Leadership"],
            "legislative_conflict_thesis": "Anticipation of domestic datacenter energy grid interconnection expedited permits.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Punctual filing.",
            "key_catalyst": "Blackwell GB200 NVL72 rack production acceleration.",
        },
    },
    {
        "id": "cong-90d-pltr-2",
        "politician": "Dan Crenshaw (R-TX)",
        "chamber": "House",
        "ticker": "PLTR",
        "asset_name": "Palantir Technologies",
        "sector": "Defense & Enterprise AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-07-08",
        "transaction_date": "2026-06-22",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 24.2,
        "sentiment": "Strong Bullish",
        "signal_strength": 94,
        "details": {
            "committee_assignments": ["Intelligence", "Energy & Commerce"],
            "legislative_conflict_thesis": "DoD CJADC2 interoperability mandate integration.",
            "historical_win_rate_pct": 74.0,
            "annualized_tech_alpha_pct": 29.4,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "TITAN next-generation sensor-to-shooter ground station expansion.",
        },
    },
    {
        "id": "cong-90d-lly-2",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "LLY",
        "asset_name": "Eli Lilly & Co.",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-06-25",
        "transaction_date": "2026-06-10",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 14.8,
        "sentiment": "Strong Bullish",
        "signal_strength": 90,
        "details": {
            "committee_assignments": ["Armed Services", "Veterans' Affairs"],
            "legislative_conflict_thesis": "Medicare Treat and Reduce Obesity Act legislative markups.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Zepbound cardiovascular risk reduction label expansion.",
        },
    },
    {
        "id": "cong-90d-vrt-2",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "VRT",
        "asset_name": "Vertiv Holdings Co",
        "sector": "AI Datacenter Power & Liquid Cooling",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-06-18",
        "transaction_date": "2026-06-04",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 14,
        "performance_since_pct": 32.1,
        "sentiment": "Strong Bullish",
        "signal_strength": 96,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "Federal standards for critical compute infrastructure power efficiency.",
            "historical_win_rate_pct": 68.0,
            "annualized_tech_alpha_pct": 28.6,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Liquid cooling CDU order book growth exceeding 80% YoY.",
        },
    },
    {
        "id": "cong-90d-msft-2",
        "politician": "Josh Gottheimer (D-NJ)",
        "chamber": "House",
        "ticker": "MSFT",
        "asset_name": "Microsoft Corp.",
        "sector": "Enterprise Cloud & AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-06-12",
        "transaction_date": "2026-05-28",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 8.5,
        "sentiment": "Bullish",
        "signal_strength": 88,
        "details": {
            "committee_assignments": ["Financial Services"],
            "legislative_conflict_thesis": "Enterprise security architecture guidelines and OpenAI partnership review.",
            "historical_win_rate_pct": 66.8,
            "annualized_tech_alpha_pct": 16.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Copilot studio enterprise deployment conversion rates.",
        },
    },
    {
        "id": "cong-90d-crwd-2",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "CRWD",
        "asset_name": "CrowdStrike Holdings",
        "sector": "Cybersecurity",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-06-05",
        "transaction_date": "2026-05-21",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 22.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 94,
        "details": {
            "committee_assignments": ["Armed Services (Cyber)"],
            "legislative_conflict_thesis": "Federal zero-trust architecture transition deadlines.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Charlotte AI security analyst ARR milestone.",
        },
    },
    {
        "id": "cong-90d-tsm-2",
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "TSM",
        "asset_name": "Taiwan Semiconductor Mfg",
        "sector": "Semiconductors / Foundry",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-05-30",
        "transaction_date": "2026-05-15",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 18.1,
        "sentiment": "Strong Bullish",
        "signal_strength": 93,
        "details": {
            "committee_assignments": ["Foreign Affairs"],
            "legislative_conflict_thesis": "Pacific maritime security deterrence and semiconductor supply chain continuity.",
            "historical_win_rate_pct": 71.0,
            "annualized_tech_alpha_pct": 22.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "N2 wafer pricing premium locked in with Tier-1 customers.",
        },
    },
    {
        "id": "cong-90d-avgo-2",
        "politician": "John Curtis (R-UT)",
        "chamber": "House",
        "ticker": "AVGO",
        "asset_name": "Broadcom Inc.",
        "sector": "Semiconductors & Custom ASIC",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-05-25",
        "transaction_date": "2026-05-10",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 19.3,
        "sentiment": "Strong Bullish",
        "signal_strength": 95,
        "details": {
            "committee_assignments": ["Energy & Commerce"],
            "legislative_conflict_thesis": "Optical interconnect and Tomahawk 6 networking standards for gigawatt datacenter parks.",
            "historical_win_rate_pct": 72.4,
            "annualized_tech_alpha_pct": 24.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Next-generation 100T co-packaged optics switch volume shipment.",
        },
    },
    {
        "id": "cong-90d-nvo-2",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "NVO",
        "asset_name": "Novo Nordisk A/S",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-05-20",
        "transaction_date": "2026-05-05",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 11.2,
        "sentiment": "Bullish",
        "signal_strength": 89,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "TRICARE preventive health coverage evaluation for metabolic therapies.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Wegovy supply expansion in European and Asian markets.",
        },
    },
    {
        "id": "cong-90d-ge-2",
        "politician": "Mark Green (R-TN)",
        "chamber": "House",
        "ticker": "GE",
        "asset_name": "GE Aerospace",
        "sector": "Aerospace & Defense Propulsion",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-05-18",
        "transaction_date": "2026-05-02",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 13.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 91,
        "details": {
            "committee_assignments": ["Homeland Security"],
            "legislative_conflict_thesis": "Next-generation combat aircraft propulsion R&D funding.",
            "historical_win_rate_pct": 73.2,
            "annualized_tech_alpha_pct": 25.1,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Commercial jet engine aftermarket price realizations.",
        },
    },
    {
        "id": "cong-90d-amd-2",
        "politician": "Josh Gottheimer (D-NJ)",
        "chamber": "House",
        "ticker": "AMD",
        "asset_name": "Advanced Micro Devices",
        "sector": "Semiconductors / AI Accelerators",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-05-14",
        "transaction_date": "2026-04-29",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 15.6,
        "sentiment": "Bullish",
        "signal_strength": 87,
        "details": {
            "committee_assignments": ["Financial Services"],
            "legislative_conflict_thesis": "Hyperscaler multi-vendor compute resilience standards.",
            "historical_win_rate_pct": 66.8,
            "annualized_tech_alpha_pct": 16.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "EPYC Turin server CPU share gains in enterprise datacenters.",
        },
    },
    {
        "id": "cong-90d-nvda-3",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-05-10",
        "transaction_date": "2026-04-25",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 34.8,
        "sentiment": "Strong Bullish",
        "signal_strength": 95,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "National semiconductor strategy and AI workforce training programs.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "CUDA ecosystem enterprise software licensing inflection.",
        },
    },
    {
        "id": "cong-90d-pltr-3",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "PLTR",
        "asset_name": "Palantir Technologies",
        "sector": "Defense & Enterprise AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-05-05",
        "transaction_date": "2026-04-20",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 39.2,
        "sentiment": "Strong Bullish",
        "signal_strength": 93,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "Defense intelligence software integration and battlefield situational awareness.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Maven Smart System operational deployment across all combatant commands.",
        },
    },
    {
        "id": "cong-90d-lly-3",
        "politician": "Dan Crenshaw (R-TX)",
        "chamber": "House",
        "ticker": "LLY",
        "asset_name": "Eli Lilly & Co.",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$50,000 - $100,000",
        "filing_date": "2026-05-02",
        "transaction_date": "2026-04-18",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 14,
        "performance_since_pct": 19.8,
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "committee_assignments": ["Energy & Commerce (Health Sub-committee)"],
            "legislative_conflict_thesis": "FDA accelerated approval pathway reforms for Alzheimer's and metabolic drugs.",
            "historical_win_rate_pct": 74.0,
            "annualized_tech_alpha_pct": 29.4,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Donanemab global regulatory approvals.",
        },
    },

    # --- 180D HORIZON (March 1 - May 1, 2026) --- (Cumulative ~46)
    {
        "id": "cong-180d-nvda-1",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-04-20",
        "transaction_date": "2026-04-02",
        "strike_price": "$120 Calls",
        "days_to_filing": 18,
        "performance_since_pct": 48.2,
        "sentiment": "Strong Bullish",
        "signal_strength": 99,
        "details": {
            "committee_assignments": ["Democratic Leadership"],
            "legislative_conflict_thesis": "Major sovereign AI initiatives in Asia & Middle East with US technology clearance.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "GTC Blackwell architecture reveal and enterprise preorder surge.",
        },
    },
    {
        "id": "cong-180d-tsm-1",
        "politician": "Mark Green (R-TN)",
        "chamber": "House",
        "ticker": "TSM",
        "asset_name": "Taiwan Semiconductor Mfg",
        "sector": "Semiconductors / Foundry",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$500,000 - $1,000,000",
        "filing_date": "2026-04-15",
        "transaction_date": "2026-03-30",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 31.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 93,
        "details": {
            "committee_assignments": ["Homeland Security"],
            "legislative_conflict_thesis": "CHIPS Act funding execution and Japanese Kumamoto Fab 2 expansion.",
            "historical_win_rate_pct": 73.2,
            "annualized_tech_alpha_pct": 25.1,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Full 3nm wafer capacity utilization driven by Apple and AI accelerators.",
        },
    },
    {
        "id": "cong-180d-vrt-1",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "VRT",
        "asset_name": "Vertiv Holdings Co",
        "sector": "AI Datacenter Power & Liquid Cooling",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-04-08",
        "transaction_date": "2026-03-24",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 52.0,
        "sentiment": "Strong Bullish",
        "signal_strength": 97,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "Grid capacity constraints and thermal management requirements for tier-4 datacenters.",
            "historical_win_rate_pct": 68.0,
            "annualized_tech_alpha_pct": 28.6,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Hyperscaler long-term supply agreement for liquid cooling modules.",
        },
    },
    {
        "id": "cong-180d-msft-1",
        "politician": "Ro Khanna (D-CA)",
        "chamber": "House",
        "ticker": "MSFT",
        "asset_name": "Microsoft Corp.",
        "sector": "Enterprise Cloud & AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-03-28",
        "transaction_date": "2026-03-12",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 12.6,
        "sentiment": "Bullish",
        "signal_strength": 89,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "Federal cloud security standards and civilian agency modernization mandates.",
            "historical_win_rate_pct": 69.2,
            "annualized_tech_alpha_pct": 21.5,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Azure cloud acceleration exceeding consensus estimates.",
        },
    },
    {
        "id": "cong-180d-crwd-1",
        "politician": "John Curtis (R-UT)",
        "chamber": "House",
        "ticker": "CRWD",
        "asset_name": "CrowdStrike Holdings",
        "sector": "Cybersecurity",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2026-03-20",
        "transaction_date": "2026-03-05",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 34.0,
        "sentiment": "Strong Bullish",
        "signal_strength": 94,
        "details": {
            "committee_assignments": ["Energy & Commerce"],
            "legislative_conflict_thesis": "Critical infrastructure cyber hardening legislation.",
            "historical_win_rate_pct": 72.4,
            "annualized_tech_alpha_pct": 24.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Falcon platform subscription gross margins reaching record 80%.",
        },
    },
    {
        "id": "cong-180d-avgo-1",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "AVGO",
        "asset_name": "Broadcom Inc.",
        "sector": "Semiconductors & Custom ASIC",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-03-15",
        "transaction_date": "2026-02-28",
        "strike_price": "$160 Calls",
        "days_to_filing": 16,
        "performance_since_pct": 38.5,
        "sentiment": "Strong Bullish",
        "signal_strength": 97,
        "details": {
            "committee_assignments": ["Democratic Leadership"],
            "legislative_conflict_thesis": "Hyperscaler custom ASIC engagement disclosures under SEC material contract guidelines.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Fourth custom ASIC hyperscaler customer contract win.",
        },
    },
    {
        "id": "cong-180d-ge-1",
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "GE",
        "asset_name": "GE Aerospace",
        "sector": "Aerospace & Defense Propulsion",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-03-10",
        "transaction_date": "2026-02-24",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 26.8,
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "committee_assignments": ["Foreign Affairs"],
            "legislative_conflict_thesis": "Global defense export authorizations and commercial aviation parts certification.",
            "historical_win_rate_pct": 71.0,
            "annualized_tech_alpha_pct": 22.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Standalone GE Aerospace spin-off capital return program and dividend boost.",
        },
    },
    {
        "id": "cong-180d-nvo-1",
        "politician": "Michael McCaul (R-TX)",
        "chamber": "House",
        "ticker": "NVO",
        "asset_name": "Novo Nordisk A/S",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2026-03-05",
        "transaction_date": "2026-02-18",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 16,
        "performance_since_pct": 21.0,
        "sentiment": "Strong Bullish",
        "signal_strength": 91,
        "details": {
            "committee_assignments": ["Foreign Affairs"],
            "legislative_conflict_thesis": "European pharmaceutical manufacturing trade agreements.",
            "historical_win_rate_pct": 71.0,
            "annualized_tech_alpha_pct": 22.8,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "CagriSema Phase 3 head-to-head trial initiation.",
        },
    },

    # --- 1Y HORIZON (Sept 2025 - Feb 2026) --- (Cumulative ~72)
    {
        "id": "cong-1y-nvda-1",
        "politician": "Nancy Pelosi (D-CA)",
        "chamber": "House",
        "ticker": "NVDA",
        "asset_name": "NVIDIA Corporation",
        "sector": "Semiconductors / AI",
        "transaction_type": "Purchase (Call Options)",
        "amount_range": "$1,000,000 - $5,000,000",
        "filing_date": "2026-01-15",
        "transaction_date": "2025-12-28",
        "strike_price": "$90 Calls",
        "days_to_filing": 18,
        "performance_since_pct": 78.4,
        "sentiment": "Strong Bullish",
        "signal_strength": 99,
        "details": {
            "committee_assignments": ["Democratic Leadership"],
            "legislative_conflict_thesis": "National AI Research Resource (NAIRR) federal pilot rollout and enterprise tax incentives.",
            "historical_win_rate_pct": 78.5,
            "annualized_tech_alpha_pct": 34.2,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Datacenter revenue exceeding $25B quarterly run rate.",
        },
    },
    {
        "id": "cong-1y-pltr-1",
        "politician": "Dan Crenshaw (R-TX)",
        "chamber": "House",
        "ticker": "PLTR",
        "asset_name": "Palantir Technologies",
        "sector": "Defense & Enterprise AI",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$50,000 - $100,000",
        "filing_date": "2025-12-18",
        "transaction_date": "2025-12-04",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 14,
        "performance_since_pct": 68.2,
        "sentiment": "Strong Bullish",
        "signal_strength": 95,
        "details": {
            "committee_assignments": ["Intelligence"],
            "legislative_conflict_thesis": "Special Operations Command software delivery mandate.",
            "historical_win_rate_pct": 74.0,
            "annualized_tech_alpha_pct": 29.4,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "S&P 500 inclusion anticipation and commercial GAAP profitability.",
        },
    },
    {
        "id": "cong-1y-lly-1",
        "politician": "Tommy Tuberville (R-AL)",
        "chamber": "Senate",
        "ticker": "LLY",
        "asset_name": "Eli Lilly & Co.",
        "sector": "Healthcare / GLP-1",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$100,000 - $250,000",
        "filing_date": "2025-11-20",
        "transaction_date": "2025-11-05",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 42.6,
        "sentiment": "Strong Bullish",
        "signal_strength": 90,
        "details": {
            "committee_assignments": ["Armed Services"],
            "legislative_conflict_thesis": "Federal Medicare drug price negotiation timeline discussions.",
            "historical_win_rate_pct": 64.5,
            "annualized_tech_alpha_pct": 18.4,
            "source_filing_url": "https://efdsearch.senate.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "Mounjaro & Zepbound multi-billion supply manufacturing investments.",
        },
    },
    {
        "id": "cong-1y-tsm-1",
        "politician": "Mark Green (R-TN)",
        "chamber": "House",
        "ticker": "TSM",
        "asset_name": "Taiwan Semiconductor Mfg",
        "sector": "Semiconductors / Foundry",
        "transaction_type": "Purchase (Common Stock)",
        "amount_range": "$250,000 - $500,000",
        "filing_date": "2025-10-15",
        "transaction_date": "2025-09-30",
        "strike_price": "N/A (Equity)",
        "days_to_filing": 15,
        "performance_since_pct": 54.0,
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "committee_assignments": ["Homeland Security"],
            "legislative_conflict_thesis": "Global advanced packaging subsidy allocations.",
            "historical_win_rate_pct": 73.2,
            "annualized_tech_alpha_pct": 25.1,
            "source_filing_url": "https://disclosures-clerk.house.gov",
            "stock_act_compliance": "Compliant.",
            "key_catalyst": "CoWoS packaging capacity expansion 150% YoY.",
        },
    },
]

UNUSUAL_OPTIONS_FLOW: List[Dict[str, Any]] = [
    # --- 7D / Today Flows --- (~$18.4M)
    {
        "id": "opt-nvda-1",
        "time": "10:42:15",
        "date": "2026-08-28",
        "ticker": "NVDA",
        "type": "CALL SWEEP",
        "strike": "$220.00",
        "expiration": "2026-09-18",
        "spot_price": 213.05,
        "premium": "$3,450,000",
        "volume_oi_ratio": 4.85,
        "implied_volatility": "44.2%",
        "order_type": "Ask (Aggressive Buying)",
        "sentiment": "Strong Bullish",
        "signal_strength": 98,
        "details": {
            "execution_urgency": "High (Executed across 6 options exchanges simultaneously)",
            "strike_distance_pct": 3.26,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.42,
            "gamma_pin_level": "$220.00 strike concentration",
            "open_interest_before": 1240,
            "volume_today": 6014,
            "institutional_intent": "Speculative breakout positioning ahead of institutional tech conference and forward guidance revisions.",
            "market_maker_hedging_impact": "Positive Gamma forcing market makers to buy underlying shares on upward momentum.",
        },
    },
    {
        "id": "opt-nvo-1",
        "time": "10:38:40",
        "date": "2026-08-28",
        "ticker": "NVO",
        "type": "CALL BLOCK",
        "strike": "$145.00",
        "expiration": "2026-10-16",
        "spot_price": 138.50,
        "premium": "$1,820,000",
        "volume_oi_ratio": 3.42,
        "implied_volatility": "32.8%",
        "order_type": "Above Ask (High Urgency)",
        "sentiment": "Strong Bullish",
        "signal_strength": 94,
        "details": {
            "execution_urgency": "Extreme (Buyer paid above prevailing ask price to guarantee single-print fill)",
            "strike_distance_pct": 4.69,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.38,
            "gamma_pin_level": "$145.00 major resistance magnet",
            "open_interest_before": 2100,
            "volume_today": 7182,
            "institutional_intent": "High-conviction positioning on Phase 3 oral GLP-1 readout milestones.",
            "market_maker_hedging_impact": "Accelerating delta hedging required if spot crosses $140 threshold.",
        },
    },
    {
        "id": "opt-pltr-1",
        "time": "10:35:12",
        "date": "2026-08-27",
        "ticker": "PLTR",
        "type": "CALL SWEEP",
        "strike": "$190.00",
        "expiration": "2026-09-18",
        "spot_price": 182.40,
        "premium": "$2,890,000",
        "volume_oi_ratio": 6.20,
        "implied_volatility": "52.1%",
        "order_type": "Above Ask (High Urgency)",
        "sentiment": "Strong Bullish",
        "signal_strength": 97,
        "details": {
            "execution_urgency": "Extreme (Multi-exchange sweep lifting ask offers across all venues)",
            "strike_distance_pct": 4.17,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.45,
            "gamma_pin_level": "$190.00 gamma squeeze strike",
            "open_interest_before": 2800,
            "volume_today": 17360,
            "institutional_intent": "Commercial AIP customer expansion and S&P 500 institutional reweighting inflows.",
            "market_maker_hedging_impact": "Dealer short gamma driving violent squeeze dynamics above $185.",
        },
    },
    {
        "id": "opt-tsla-1",
        "time": "10:31:05",
        "date": "2026-08-27",
        "ticker": "TSLA",
        "type": "PUT SWEEP",
        "strike": "$330.00",
        "expiration": "2026-08-28",
        "spot_price": 350.25,
        "premium": "$2,150,000",
        "volume_oi_ratio": 5.12,
        "implied_volatility": "58.4%",
        "order_type": "Bid (Aggressive Hedging)",
        "sentiment": "Bearish / Tail Hedge",
        "signal_strength": 88,
        "details": {
            "execution_urgency": "High (Rapid multi-exchange liquidity sweep)",
            "strike_distance_pct": -5.78,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": -0.31,
            "gamma_pin_level": "$330.00 downside support floor",
            "open_interest_before": 3400,
            "volume_today": 17400,
            "institutional_intent": "Short-term downside protection or volatility arbitrage against delivery figures.",
            "market_maker_hedging_impact": "Negative gamma zone accelerating selling if $345 support breaks.",
        },
    },
    {
        "id": "opt-vrt-1",
        "time": "10:28:18",
        "date": "2026-08-26",
        "ticker": "VRT",
        "type": "CALL SWEEP",
        "strike": "$160.00",
        "expiration": "2026-10-16",
        "spot_price": 149.20,
        "premium": "$4,250,000",
        "volume_oi_ratio": 7.40,
        "implied_volatility": "49.0%",
        "order_type": "Above Ask (High Urgency)",
        "sentiment": "Strong Bullish",
        "signal_strength": 96,
        "details": {
            "execution_urgency": "High (Massive out-of-the-money liquid cooling call buying)",
            "strike_distance_pct": 7.24,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.36,
            "gamma_pin_level": "$160.00 gamma magnet",
            "open_interest_before": 950,
            "volume_today": 7030,
            "institutional_intent": "Datacenter direct-to-chip liquid cooling supply contract expansion.",
            "market_maker_hedging_impact": "Dealer buying underlying shares on any pullbacks to $145.",
        },
    },
    {
        "id": "opt-avgo-1",
        "time": "10:20:10",
        "date": "2026-08-25",
        "ticker": "AVGO",
        "type": "CALL SWEEP",
        "strike": "$310.00",
        "expiration": "2026-09-18",
        "spot_price": 296.80,
        "premium": "$3,840,000",
        "volume_oi_ratio": 4.15,
        "implied_volatility": "39.5%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Strong Bullish",
        "signal_strength": 95,
        "details": {
            "execution_urgency": "High (Custom ASIC hyperscaler order flow)",
            "strike_distance_pct": 4.45,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.41,
            "gamma_pin_level": "$310.00 resistance pivot",
            "open_interest_before": 1400,
            "volume_today": 5810,
            "institutional_intent": "Custom silicon TPU/XPU datacenter revenue inflection.",
            "market_maker_hedging_impact": "Positive gamma ramp above $300.",
        },
    },

    # --- 30D HORIZON FLOWS --- (Cumulative ~$42.8M)
    {
        "id": "opt-lly-1",
        "time": "10:15:22",
        "date": "2026-08-15",
        "ticker": "LLY",
        "type": "CALL SWEEP",
        "strike": "$950.00",
        "expiration": "2026-11-20",
        "spot_price": 920.40,
        "premium": "$5,640,000",
        "volume_oi_ratio": 3.90,
        "implied_volatility": "28.5%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Strong Bullish",
        "signal_strength": 92,
        "details": {
            "execution_urgency": "High (Multi-month LEAP accumulation)",
            "strike_distance_pct": 3.22,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.44,
            "gamma_pin_level": "$950.00 institutional magnet",
            "open_interest_before": 1850,
            "volume_today": 7215,
            "institutional_intent": "Long-term institutional accumulation betting on sustained gross margin expansion.",
            "market_maker_hedging_impact": "Steady upward pressure as spot converges toward $950 strike.",
        },
    },
    {
        "id": "opt-crwd-1",
        "time": "10:11:45",
        "date": "2026-08-12",
        "ticker": "CRWD",
        "type": "CALL SWEEP",
        "strike": "$420.00",
        "expiration": "2026-10-16",
        "spot_price": 398.50,
        "premium": "$4,410,000",
        "volume_oi_ratio": 5.80,
        "implied_volatility": "46.7%",
        "order_type": "Above Ask (High Urgency)",
        "sentiment": "Strong Bullish",
        "signal_strength": 94,
        "details": {
            "execution_urgency": "High (Out-of-the-money cloud cybersecurity call sweep)",
            "strike_distance_pct": 5.40,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.37,
            "gamma_pin_level": "$420.00 breakout target",
            "open_interest_before": 1120,
            "volume_today": 6500,
            "institutional_intent": "Betting on accelerated ARR growth and platform module adoption.",
            "market_maker_hedging_impact": "Dealer buying supporting $390 floor.",
        },
    },
    {
        "id": "opt-tsm-1",
        "time": "10:02:18",
        "date": "2026-08-08",
        "ticker": "TSM",
        "type": "CALL SWEEP",
        "strike": "$210.00",
        "expiration": "2026-10-16",
        "spot_price": 196.30,
        "premium": "$6,620,000",
        "volume_oi_ratio": 4.60,
        "implied_volatility": "37.4%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Strong Bullish",
        "signal_strength": 93,
        "details": {
            "execution_urgency": "High (Multi-venue ADR sweep)",
            "strike_distance_pct": 6.98,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.39,
            "gamma_pin_level": "$210.00 institutional magnet",
            "open_interest_before": 3100,
            "volume_today": 14260,
            "institutional_intent": "Leading foundry pricing power and CoWoS advanced packaging capacity double.",
            "market_maker_hedging_impact": "Delta ramp supporting upward continuation.",
        },
    },
    {
        "id": "opt-amd-1",
        "time": "10:07:30",
        "date": "2026-08-04",
        "ticker": "AMD",
        "type": "CALL BLOCK",
        "strike": "$185.00",
        "expiration": "2026-09-18",
        "spot_price": 174.60,
        "premium": "$4,780,000",
        "volume_oi_ratio": 3.75,
        "implied_volatility": "42.0%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Bullish",
        "signal_strength": 89,
        "details": {
            "execution_urgency": "Moderate-High (Single institutional block print)",
            "strike_distance_pct": 5.95,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.35,
            "gamma_pin_level": "$185.00 resistance pivot",
            "open_interest_before": 2400,
            "volume_today": 9000,
            "institutional_intent": "Positioning ahead of MI350 server deployment benchmarks.",
            "market_maker_hedging_impact": "Hedging active above $175.",
        },
    },
    {
        "id": "opt-spy-30d",
        "time": "09:45:10",
        "date": "2026-08-01",
        "ticker": "SPY",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 762.10,
        "premium": "$3,000,000",
        "volume_oi_ratio": 2.10,
        "implied_volatility": "14.5%",
        "order_type": "Cross Trade (Off-Exchange OTC)",
        "sentiment": "Institutional Inflow",
        "signal_strength": 91,
        "details": {
            "execution_urgency": "Institutional Rebalancing",
            "strike_distance_pct": 0.0,
            "moneyness": "At-the-Market",
            "delta_est": 1.0,
            "gamma_pin_level": "$760.00 pivot floor",
            "open_interest_before": 0,
            "volume_today": 35000,
            "institutional_intent": "Systematic month-start liquidity inflow.",
            "market_maker_hedging_impact": "Neutralizing delta.",
        },
    },

    # --- 90D HORIZON FLOWS --- (Cumulative ~$94.2M)
    {
        "id": "opt-90d-nvda-1",
        "time": "11:15:30",
        "date": "2026-07-20",
        "ticker": "NVDA",
        "type": "CALL SWEEP",
        "strike": "$190.00",
        "expiration": "2026-12-18",
        "spot_price": 178.50,
        "premium": "$12,500,000",
        "volume_oi_ratio": 5.40,
        "implied_volatility": "46.2%",
        "order_type": "Above Ask",
        "sentiment": "Strong Bullish",
        "signal_strength": 99,
        "details": {
            "execution_urgency": "Whale Order ($12.5M LEAP Sweep)",
            "strike_distance_pct": 6.44,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.46,
            "gamma_pin_level": "$190.00 magnet",
            "open_interest_before": 3100,
            "volume_today": 16200,
            "institutional_intent": "Long-term hyperscaler compute buildout thesis.",
            "market_maker_hedging_impact": "Persistent dealer buy pressure.",
        },
    },
    {
        "id": "opt-90d-pltr-1",
        "time": "10:50:18",
        "date": "2026-07-12",
        "ticker": "PLTR",
        "type": "CALL SWEEP",
        "strike": "$160.00",
        "expiration": "2026-11-20",
        "spot_price": 148.20,
        "premium": "$8,900,000",
        "volume_oi_ratio": 6.80,
        "implied_volatility": "54.0%",
        "order_type": "Ask (Aggressive)",
        "sentiment": "Strong Bullish",
        "signal_strength": 96,
        "details": {
            "execution_urgency": "Extreme ($8.9M multi-exchange print)",
            "strike_distance_pct": 7.96,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.41,
            "gamma_pin_level": "$160.00 pin",
            "open_interest_before": 1900,
            "volume_today": 18400,
            "institutional_intent": "Government AI telemetry expansion.",
            "market_maker_hedging_impact": "Delta squeeze active.",
        },
    },
    {
        "id": "opt-90d-qqq-1",
        "time": "09:58:05",
        "date": "2026-06-25",
        "ticker": "QQQ",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 542.10,
        "premium": "$30,000,000",
        "volume_oi_ratio": 2.45,
        "implied_volatility": "18.2%",
        "order_type": "Cross Trade (Off-Exchange OTC)",
        "sentiment": "Institutional Inflow",
        "signal_strength": 92,
        "details": {
            "execution_urgency": "Institutional Print ($30.0M block size)",
            "strike_distance_pct": 0.0,
            "moneyness": "At-the-Market",
            "delta_est": 1.0,
            "gamma_pin_level": "$540.00 pivot floor",
            "open_interest_before": 0,
            "volume_today": 48200,
            "institutional_intent": "Tech sector core allocation by quantitative liquidity providers.",
            "market_maker_hedging_impact": "Neutralizing off-exchange volatility.",
        },
    },

    # --- 180D HORIZON FLOWS --- (Cumulative ~$168.5M)
    {
        "id": "opt-180d-spy-1",
        "time": "10:24:50",
        "date": "2026-04-18",
        "ticker": "SPY",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 745.20,
        "premium": "$48,200,000",
        "volume_oi_ratio": 2.10,
        "implied_volatility": "14.5%",
        "order_type": "Cross Trade (Off-Exchange OTC)",
        "sentiment": "Institutional Inflow",
        "signal_strength": 91,
        "details": {
            "execution_urgency": "Institutional Sizing ($48.2M single dark pool print)",
            "strike_distance_pct": 0.0,
            "moneyness": "At-the-Market",
            "delta_est": 1.0,
            "gamma_pin_level": "$745.00 broad index pivot",
            "open_interest_before": 0,
            "volume_today": 62900,
            "institutional_intent": "Passive benchmark rebalancing and systematic sovereign fund liquidity allocation.",
            "market_maker_hedging_impact": "Neutralizing off-exchange volatility.",
        },
    },
    {
        "id": "opt-180d-nvda-1",
        "time": "11:05:10",
        "date": "2026-03-22",
        "ticker": "NVDA",
        "type": "CALL SWEEP",
        "strike": "$150.00",
        "expiration": "2026-10-16",
        "spot_price": 135.40,
        "premium": "$26,100,000",
        "volume_oi_ratio": 6.10,
        "implied_volatility": "42.0%",
        "order_type": "Above Ask",
        "sentiment": "Strong Bullish",
        "signal_strength": 99,
        "details": {
            "execution_urgency": "Mega Whale ($26.1M Call Sweep)",
            "strike_distance_pct": 10.78,
            "moneyness": "Out-of-the-Money (OTM)",
            "delta_est": 0.44,
            "gamma_pin_level": "$150.00 strike target",
            "open_interest_before": 4200,
            "volume_today": 34000,
            "institutional_intent": "Massive post-GTC compute expansion positioning.",
            "market_maker_hedging_impact": "Dealer buying supporting structural trend.",
        },
    },

    # --- 1Y HORIZON FLOWS --- (Cumulative ~$285.0M)
    {
        "id": "opt-1y-spy-1",
        "time": "14:10:05",
        "date": "2025-11-15",
        "ticker": "SPY",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 685.40,
        "premium": "$65,000,000",
        "volume_oi_ratio": 3.10,
        "implied_volatility": "13.2%",
        "order_type": "Cross Trade (Off-Exchange OTC)",
        "sentiment": "Institutional Inflow",
        "signal_strength": 95,
        "details": {
            "execution_urgency": "Sovereign Wealth Fund Rebalance ($65M print)",
            "strike_distance_pct": 0.0,
            "moneyness": "At-the-Market",
            "delta_est": 1.0,
            "gamma_pin_level": "$685.00 index support",
            "open_interest_before": 0,
            "volume_today": 94000,
            "institutional_intent": "Strategic annual benchmark allocation.",
            "market_maker_hedging_impact": "Neutralizing broad volatility.",
        },
    },
    {
        "id": "opt-1y-qqq-1",
        "time": "13:40:20",
        "date": "2025-10-22",
        "ticker": "QQQ",
        "type": "DARK POOL BLOCK",
        "strike": "Spot Equity",
        "expiration": "N/A",
        "spot_price": 495.20,
        "premium": "$51,500,000",
        "volume_oi_ratio": 2.80,
        "implied_volatility": "16.8%",
        "order_type": "Cross Trade (Off-Exchange OTC)",
        "sentiment": "Institutional Inflow",
        "signal_strength": 93,
        "details": {
            "execution_urgency": "Major Institutional Print ($51.5M)",
            "strike_distance_pct": 0.0,
            "moneyness": "At-the-Market",
            "delta_est": 1.0,
            "gamma_pin_level": "$495.00 anchor",
            "open_interest_before": 0,
            "volume_today": 78000,
            "institutional_intent": "Tech sector systematic accumulation.",
            "market_maker_hedging_impact": "Neutralizing delta.",
        },
    },
]


SEC_FORM_4_TRADES = [
    # --- 7D HORIZON (Aug 22 - Aug 28, 2026) --- (4 trades)
    {
        'id': 'sec-anet-1',
        'insider_name': 'Jayshree Ullal (President & CEO)',
        'ticker': 'ANET',
        'company_name': 'Arista Networks Inc.',
        'role': 'President & Chief Executive Officer',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 12000,
        'price_per_share': 188.40,
        'total_value': '$2,260,800',
        'filing_date': '2026-08-28',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '3.4%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 C-Suite Open Market Buy ($2.3M)',
        'conviction_score': 96,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-lnth-1',
        'insider_name': 'Mary Anne Heino (Director & Ex-CEO)',
        'ticker': 'LNTH',
        'company_name': 'Lantheus Holdings Inc.',
        'role': 'Director',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 15000,
        'price_per_share': 98.50,
        'total_value': '$1,477,500',
        'filing_date': '2026-08-26',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Director Open Market Buy ($1.5M)',
        'conviction_score': 95,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-pltr-1',
        'insider_name': 'Alexander Karp (CEO)',
        'ticker': 'PLTR',
        'company_name': 'Palantir Technologies',
        'role': 'Chief Executive Officer / Director',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 85000,
        'price_per_share': 180.20,
        'total_value': '$15,317,000',
        'filing_date': '2026-08-27',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '5.1%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Open Market Buy ($15.3M)',
        'conviction_score': 98,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-crwd-1',
        'insider_name': 'George Kurtz (CEO)',
        'ticker': 'CRWD',
        'company_name': 'CrowdStrike Holdings',
        'role': 'Chief Executive Officer & Founder',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 25000,
        'price_per_share': 395.00,
        'total_value': '$9,875,000',
        'filing_date': '2026-08-25',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '4.2%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Founder Buy ($9.8M)',
        'conviction_score': 97,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },

    # --- 30D HORIZON (July 29 - Aug 21, 2026) --- (Cumulative 5)
    {
        'id': 'sec-nvda-1',
        'insider_name': 'Jensen Huang (CEO & President)',
        'ticker': 'NVDA',
        'company_name': 'NVIDIA Corporation',
        'role': 'Chief Executive Officer / Director',
        'transaction_type': 'Purchase (Grant / Long Hold)',
        'shares_traded': 120000,
        'price_per_share': 210.50,
        'total_value': '$25,260,000',
        'filing_date': '2026-08-20',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '3.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Executive Conviction ($25.2M)',
        'conviction_score': 99,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-vrt-1',
        'insider_name': 'Giordano Albertazzi (CEO)',
        'ticker': 'VRT',
        'company_name': 'Vertiv Holdings Co',
        'role': 'Chief Executive Officer',
        'transaction_type': 'Purchase (Open Market)',
        'shares_traded': 40000,
        'price_per_share': 146.80,
        'total_value': '$5,872,000',
        'filing_date': '2026-08-19',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.4%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 C-Suite Buy ($5.8M)',
        'conviction_score': 95,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-nvo-1',
        'insider_name': 'Lars Fruergaard Jorgensen (CEO)',
        'ticker': 'NVO',
        'company_name': 'Novo Nordisk A/S',
        'role': 'President & CEO',
        'transaction_type': 'Purchase (Executive Incentive Plan)',
        'shares_traded': 30000,
        'price_per_share': 136.40,
        'total_value': '$4,092,000',
        'filing_date': '2026-08-18',
        'form_type': 'SEC Form 6-K / Form 4',
        'direct_ownership_pct': '0.9%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Executive Accumulation ($4.1M)',
        'conviction_score': 93,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },

    # --- 90D HORIZON (May 30 - July 28, 2026) --- (Cumulative 12)
    {
        'id': 'sec-90d-msft-1',
        'insider_name': 'Satya Nadella (CEO)',
        'ticker': 'MSFT',
        'company_name': 'Microsoft Corp.',
        'role': 'Chairman & CEO',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 35000,
        'price_per_share': 445.20,
        'total_value': '$15,582,000',
        'filing_date': '2026-07-22',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CEO Conviction ($15.5M)',
        'conviction_score': 98,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-avgo-1',
        'insider_name': 'Hock Tan (President & CEO)',
        'ticker': 'AVGO',
        'company_name': 'Broadcom Inc.',
        'role': 'President, CEO & Director',
        'transaction_type': 'Purchase (Grant / Open Market)',
        'shares_traded': 45000,
        'price_per_share': 260.40,
        'total_value': '$11,718,000',
        'filing_date': '2026-07-14',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '2.6%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Founder Tier ($11.7M)',
        'conviction_score': 97,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-lly-1',
        'insider_name': 'David A. Ricks (Chair & CEO)',
        'ticker': 'LLY',
        'company_name': 'Eli Lilly & Co.',
        'role': 'Chairman & CEO',
        'transaction_type': 'Purchase (Direct Holding)',
        'shares_traded': 15000,
        'price_per_share': 880.00,
        'total_value': '$13,200,000',
        'filing_date': '2026-06-28',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.2%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 C-Suite Buy ($13.2M)',
        'conviction_score': 96,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-amd-1',
        'insider_name': 'Lisa Su (Chair & CEO)',
        'ticker': 'AMD',
        'company_name': 'Advanced Micro Devices',
        'role': 'Chair, President & CEO',
        'transaction_type': 'Purchase (Open Market)',
        'shares_traded': 50000,
        'price_per_share': 162.50,
        'total_value': '$8,125,000',
        'filing_date': '2026-06-19',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '3.1%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Executive Buy ($8.1M)',
        'conviction_score': 96,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-ge-1',
        'insider_name': 'H. Lawrence Culp Jr. (CEO)',
        'ticker': 'GE',
        'company_name': 'GE Aerospace',
        'role': 'Chairman and CEO',
        'transaction_type': 'Purchase (Open Market)',
        'shares_traded': 40000,
        'price_per_share': 168.00,
        'total_value': '$6,720,000',
        'filing_date': '2026-06-10',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.9%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CEO Buy ($6.7M)',
        'conviction_score': 94,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-tsm-1',
        'insider_name': 'C.C. Wei (Chairman & CEO)',
        'ticker': 'TSM',
        'company_name': 'Taiwan Semiconductor Mfg',
        'role': 'Chairman & Chief Executive Officer',
        'transaction_type': 'Purchase (ADR Open Market)',
        'shares_traded': 30000,
        'price_per_share': 175.40,
        'total_value': '$5,262,000',
        'filing_date': '2026-06-04',
        'form_type': 'SEC Form 6-K',
        'direct_ownership_pct': '0.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Executive ADR Buy ($5.3M)',
        'conviction_score': 95,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-90d-vrt-2',
        'insider_name': 'David M. Cote (Executive Chairman)',
        'ticker': 'VRT',
        'company_name': 'Vertiv Holdings Co',
        'role': 'Executive Chairman',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 60000,
        'price_per_share': 118.50,
        'total_value': '$7,110,000',
        'filing_date': '2026-05-31',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '3.4%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Chairman Buy ($7.1M)',
        'conviction_score': 96,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },

    # --- 180D HORIZON (March 1 - May 29, 2026) --- (Cumulative 22)
    {
        'id': 'sec-180d-nvda-1',
        'insider_name': 'Colette Kress (EVP & CFO)',
        'ticker': 'NVDA',
        'company_name': 'NVIDIA Corporation',
        'role': 'Chief Financial Officer',
        'transaction_type': 'Purchase (Direct Holding)',
        'shares_traded': 50000,
        'price_per_share': 125.00,
        'total_value': '$6,250,000',
        'filing_date': '2026-04-18',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '0.6%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CFO Conviction ($6.2M)',
        'conviction_score': 94,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-180d-pltr-1',
        'insider_name': 'Peter Thiel (Chairman)',
        'ticker': 'PLTR',
        'company_name': 'Palantir Technologies',
        'role': 'Chairman of the Board / Co-Founder',
        'transaction_type': 'Purchase (Long Term Vehicle)',
        'shares_traded': 100000,
        'price_per_share': 122.00,
        'total_value': '$12,200,000',
        'filing_date': '2026-04-05',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '6.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 Founder Anchor ($12.2M)',
        'conviction_score': 99,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-180d-crwd-1',
        'insider_name': 'Burt Podbere (CFO)',
        'ticker': 'CRWD',
        'company_name': 'CrowdStrike Holdings',
        'role': 'Chief Financial Officer',
        'transaction_type': 'Purchase (Open Market)',
        'shares_traded': 15000,
        'price_per_share': 310.00,
        'total_value': '$4,650,000',
        'filing_date': '2026-03-24',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '0.8%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 C-Suite Buy ($4.6M)',
        'conviction_score': 92,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-180d-msft-2',
        'insider_name': 'Amy Hood (EVP & CFO)',
        'ticker': 'MSFT',
        'company_name': 'Microsoft Corp.',
        'role': 'Chief Financial Officer',
        'transaction_type': 'Purchase (Direct Open Market)',
        'shares_traded': 20000,
        'price_per_share': 415.00,
        'total_value': '$8,300,000',
        'filing_date': '2026-03-15',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '0.7%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CFO Buy ($8.3M)',
        'conviction_score': 95,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },

    # --- 1Y HORIZON (Sept 2025 - Feb 2026) --- (Cumulative 38)
    {
        'id': 'sec-1y-nvda-1',
        'insider_name': 'Jensen Huang (CEO & President)',
        'ticker': 'NVDA',
        'company_name': 'NVIDIA Corporation',
        'role': 'Chief Executive Officer / Director',
        'transaction_type': 'Purchase (Long Term Direct)',
        'shares_traded': 200000,
        'price_per_share': 98.40,
        'total_value': '$19,680,000',
        'filing_date': '2026-01-10',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '3.9%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CEO Buy ($19.7M)',
        'conviction_score': 99,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
    {
        'id': 'sec-1y-lly-1',
        'insider_name': 'David A. Ricks (Chair & CEO)',
        'ticker': 'LLY',
        'company_name': 'Eli Lilly & Co.',
        'role': 'Chairman & CEO',
        'transaction_type': 'Purchase (Direct Holding)',
        'shares_traded': 20000,
        'price_per_share': 720.00,
        'total_value': '$14,400,000',
        'filing_date': '2025-11-12',
        'form_type': 'SEC Form 4',
        'direct_ownership_pct': '1.3%',
        'sentiment': 'Strong Bullish',
        'conviction_tier': '👑 CEO Buy ($14.4M)',
        'conviction_score': 97,
        'sec_url': 'https://www.sec.gov/edgar/searchedgar/companysearch',
    },
]


def calculate_legislative_alignment(trade: Dict[str, Any]) -> int:
    """Calculate quantitative Legislative Alignment & Conflict Index (0 - 100)."""
    score = 45  # Neutral baseline
    details = trade.get("details", {})
    committees = details.get("committee_assignments", [])
    sector = trade.get("sector", "").lower()

    # 1. Jurisdiction Committee Overlap (up to +35 pts)
    for c in committees:
        c_lower = c.lower()
        if any(term in c_lower for term in ["armed services", "defense", "intelligence"]) and any(term in sector for term in ["defense", "ai", "cyber", "software", "foundry"]):
            score += 32
            break
        elif any(term in c_lower for term in ["energy", "commerce", "technology", "innovative"]) and any(term in sector for term in ["semiconductor", "ai", "tech", "cooling", "power", "cybersecurity"]):
            score += 30
            break
        elif any(term in c_lower for term in ["foreign affairs", "homeland"]) and any(term in sector for term in ["pharma", "foundry", "aerospace", "semiconductor", "defense"]):
            score += 26
            break
        elif any(term in c_lower for term in ["veterans", "agriculture", "finance", "banking"]):
            score += 16
            break

    # 2. Dollar Bracket Sizing Tier (up to +15 pts)
    amt = trade.get("amount_range", "")
    if "$1,000,000" in amt or "$5,000,000" in amt:
        score += 15
    elif "$500,000" in amt:
        score += 11
    elif "$250,000" in amt:
        score += 8
    else:
        score += 5

    # 3. Politician Historical Track Record (up to +10 pts)
    win_rate = details.get("historical_win_rate_pct", 65.0)
    if win_rate >= 75.0:
        score += 10
    elif win_rate >= 70.0:
        score += 7
    elif win_rate >= 65.0:
        score += 4

    return min(99, max(35, score))


def compute_filing_staleness(days_to_filing: int, base_strength: int = 90) -> Dict[str, Any]:
    """Compute STOCK Act filing latency, staleness decay penalty, and compliance tier."""
    if days_to_filing <= 15:
        return {
            "staleness_status": "FRESH",
            "staleness_badge": "⚡ Fresh Filing (<15d)",
            "staleness_penalty": 0,
            "staleness_warning": None,
            "effective_signal_strength": base_strength,
            "compliance_tier": "Punctual (Under 15 Days)",
        }
    elif days_to_filing <= 30:
        penalty = 5
        return {
            "staleness_status": "NORMAL",
            "staleness_badge": "⏳ Standard Lag (16-30d)",
            "staleness_penalty": penalty,
            "staleness_warning": None,
            "effective_signal_strength": max(10, base_strength - penalty),
            "compliance_tier": "Compliant (Within 30 Days)",
        }
    elif days_to_filing <= 45:
        penalty = 16
        return {
            "staleness_status": "AGING",
            "staleness_badge": "⚠️ Aging Signal (31-45d)",
            "staleness_penalty": penalty,
            "staleness_warning": f"Disclosed {days_to_filing} days after execution — approaching statutory limit.",
            "effective_signal_strength": max(10, base_strength - penalty),
            "compliance_tier": "Late Approaching (31-45 Days)",
        }
    else:
        penalty = 32
        return {
            "staleness_status": "LATE_FILER",
            "staleness_badge": f"🛑 Late Filer ({days_to_filing}d Lag)",
            "staleness_penalty": penalty,
            "staleness_warning": f"Filed {days_to_filing} days late (exceeds statutory 45d window) — high risk of priced-in mean reversion.",
            "effective_signal_strength": max(10, base_strength - penalty),
            "compliance_tier": "Non-Compliant (>45 Days Late)",
        }


class SmartMoneyDomainStatus(str, Enum):
    VERIFIED_SIGNAL = "VERIFIED_SIGNAL"
    VERIFIED_NO_ACTIVITY = "VERIFIED_NO_ACTIVITY"
    PARTIAL_COVERAGE = "PARTIAL_COVERAGE"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    PIPELINE_HELD = "PIPELINE_HELD"
    CURATED_RESEARCH = "CURATED_RESEARCH"
    INAPPLICABLE = "INAPPLICABLE"


class SmartMoneyDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    UNASSESSED = "UNASSESSED"


def _parse_money(val: Any) -> float:
    """Robustly parse dollar strings like '$3,450,000' or ranges like '$250,000 - $500,000' into float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").replace("USD", "").strip()
        if "-" in cleaned:
            parts = cleaned.split("-")
            try:
                p0 = float(parts[0].replace("$", "").replace(",", "").strip())
                p1 = float(parts[1].replace("$", "").replace(",", "").strip())
                return (p0 + p1) / 2.0
            except Exception:
                return 0.0
        try:
            return float(cleaned)
        except Exception:
            return 0.0
    return 0.0


class SmartMoneyEngine:
    """Quantitative engine for tracking Capitol Hill disclosures, SEC Form 4 & institutional options flow."""

    LOOKBACK_DAYS_OPTIONS = 90
    LOOKBACK_DAYS_INSIDER = 90
    LOOKBACK_DAYS_CONGRESS = 180

    KNOWN_ETFS = {
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
        "XLU", "XLB", "XLRE", "XLC", "SMH", "VTI", "VOO", "VEA", "VWO", "ARKK",
    }

    KNOWN_FOREIGN_ADRS = {
        "DHLGY", "NVO", "ASML", "TSM", "BABA", "AZN", "BP", "SHEL", "SAP", "SNY",
        "TM", "HMC", "VALE", "BTI", "RIO", "UL", "LVMUY", "TCEHY", "NPSNY", "ADDYY",
        "BAYRY", "DDAIF", "VWAGY", "BMWYY", "RHHBY", "SAN", "BBVA", "ING", "BNPQY", "CRH",
    }

    @staticmethod
    def evaluate_smart_money_feeds(
        symbol: str,
        window_days: int = 90,
        include_curated: bool = False,
        asset_class: Optional[str] = None,
        security_type: Optional[str] = None,
        evaluation_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive bounded analytical evaluation of Smart Money evidence across:
        1. Options Flow Sweeps
        2. SEC Form 4 Insider Transactions
        3. Congressional STOCK Act Disclosures

        Maintains independent feed states, distinct timestamps, truthful coverage representation,
        and provides normalized Confluence Engine inputs without synthetic promotion.
        """
        clean_sym = symbol.upper().replace("-USD", "").strip() if symbol else ""
        raw_sym = symbol.upper().strip() if symbol else ""

        is_crypto = (
            raw_sym.endswith("-USD")
            or raw_sym in {"BTC", "ETH", "SOL", "DOGE"}
            or (asset_class and asset_class.upper() == "CRYPTO")
            or (security_type and security_type.upper() == "CRYPTO")
        )

        known_etfs = {
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
            "XLU", "XLB", "XLRE", "XLC", "SMH", "VTI", "VOO", "VEA", "VWO", "ARKK",
        }
        is_etf = (
            raw_sym in known_etfs
            or (asset_class and asset_class.upper() == "ETF")
            or (security_type and security_type.upper() == "ETF")
        )

        known_foreign_adrs = {
            "DHLGY", "NVO", "ASML", "TSM", "BABA", "AZN", "BP", "SHEL", "SAP", "SNY",
            "TM", "HMC", "VALE", "BTI", "RIO", "UL", "LVMUY", "TCEHY", "NPSNY", "ADDYY",
            "BAYRY", "DDAIF", "VWAGY", "BMWYY", "RHHBY", "SAN", "BBVA", "ING", "BNPQY", "CRH",
        }
        is_foreign_adr = (
            (len(clean_sym) == 5 and clean_sym.endswith(("Y", "F")))
            or clean_sym in known_foreign_adrs
        )

        # Establish analytical evaluation window
        try:
            if evaluation_date:
                eval_date = datetime.strptime(str(evaluation_date)[:10], "%Y-%m-%d").date()
            else:
                eval_date = datetime.now(timezone.utc).date()
        except Exception:
            eval_date = datetime.now(timezone.utc).date()

        cutoff_date = eval_date - timedelta(days=window_days)
        cutoff_str = cutoff_date.isoformat()

        # ── FEED 1: SEC FORM 4 INSIDER TRANSACTIONS ───────────────────────────
        sec_observations: List[Dict[str, Any]] = []
        sec_distinct_dates = None
        sec_cik = None

        if is_crypto or is_etf:
            sec_feed = {
                "status": "INAPPLICABLE",
                "status_label": "Exempt (Fund / Digital Asset)",
                "coverage": "EXEMPT_ASSET",
                "cik": None,
                "observations": [],
                "observation_count": 0,
                "direction": "UNVERIFIED",
                "confidence": "NONE",
                "explanation": "Corporate insider transactions inapplicable to funds and digital assets.",
                "provenance": "SEC EDGAR",
                "distinct_dates": None,
            }
        elif is_foreign_adr:
            sec_feed = {
                "status": "INAPPLICABLE",
                "status_label": "Exempt (Foreign Issuer)",
                "coverage": "FOREIGN_FPI_EXEMPT",
                "cik": None,
                "observations": [],
                "observation_count": 0,
                "direction": "UNVERIFIED",
                "confidence": "NONE",
                "explanation": "Foreign Private Issuer (FPI) exempt from US SEC Form 4 filings; local regulatory disclosures apply.",
                "provenance": "Home Country Regulatory Authorities",
                "distinct_dates": None,
            }
        else:
            sec_rows = []
            has_older_in_db = False
            try:
                from analyst_dashboard.smart_money.form4_ingestion import connect_smart_money_db
                conn = connect_smart_money_db()
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                           officer_title, is_director, is_officer, is_ten_percent_owner, filing_date,
                           transaction_date, transaction_code, signal_type, shares, price_per_share, total_value_usd, acquired_disposed,
                           is_positive_buying_conviction, is_open_market_purchase_evidence, acceptance_datetime
                    FROM form4_parsed_transactions
                    WHERE issuer_symbol = ? AND (filing_date >= ? OR transaction_date >= ?)
                    ORDER BY filing_date DESC
                """, (clean_sym, cutoff_str, cutoff_str))
                sec_rows = cur.fetchall()

                cur.execute("SELECT issuer_cik FROM form4_parsed_transactions WHERE issuer_symbol = ? LIMIT 1", (clean_sym,))
                cik_row = cur.fetchone()
                if cik_row:
                    sec_cik = str(cik_row["issuer_cik"]).zfill(10)
                    has_older_in_db = True
                conn.close()
            except Exception as e:
                logger.debug(f"Smart money local DB query for {clean_sym} Form 4 failed: {e}")
                sec_rows = []

            if not sec_cik:
                try:
                    from analyst_dashboard.data.sec_edgar_fetcher import SecEdgarFetcher
                    sec_fetcher = SecEdgarFetcher()
                    sec_cik = sec_fetcher.resolve_cik(clean_sym)
                except Exception:
                    sec_cik = None

            if sec_rows:
                for r in sec_rows:
                    t_code = (r["transaction_code"] or "").upper().strip() if "transaction_code" in r.keys() else ""
                    sig_type = (r["signal_type"] or "").upper().strip() if "signal_type" in r.keys() else ""
                    is_pos_conviction = bool(r["is_positive_buying_conviction"]) or bool(r["is_open_market_purchase_evidence"])
                    
                    is_buy = is_pos_conviction or t_code == "P" or sig_type == "OPEN_MARKET_PURCHASE"
                    is_sell = t_code == "S" or sig_type == "OPEN_MARKET_SALE" or (r["acquired_disposed"] == "D" and not is_buy)

                    if is_buy:
                        tx_type = "Purchase (Direct Open Market)"
                    elif is_sell:
                        tx_type = "Sale (Direct)"
                    elif t_code == "M" or sig_type == "OPTION_EXERCISE":
                        tx_type = "Option Exercise"
                    elif t_code == "A" or sig_type == "EQUITY_GRANT":
                        tx_type = "Equity Grant / Award"
                    elif t_code == "G" or sig_type == "GIFT":
                        tx_type = "Gift"
                    elif t_code == "F" or sig_type == "TAX_WITHHOLDING":
                        tx_type = "Tax Withholding"
                    elif t_code == "D" or sig_type == "ISSUER_DISPOSITION":
                        tx_type = "Disposition to Issuer"
                    else:
                        tx_type = f"Transaction ({t_code})" if t_code else "Other Insider Transaction"

                    sec_observations.append({
                        "id": r["accession_number"],
                        "insider_name": r["reporting_owner_name"],
                        "ticker": clean_sym,
                        "role": r["officer_title"] or ("Director" if r["is_director"] else "Officer"),
                        "transaction_type": tx_type,
                        "transaction_code": t_code,
                        "signal_type": sig_type,
                        "shares_traded": r["shares"] or 0,
                        "price_per_share": r["price_per_share"] or 0.0,
                        "total_value_usd": r["total_value_usd"] or 0.0,
                        "filing_date": r["filing_date"],
                        "transaction_date": r["transaction_date"] or r["filing_date"],
                        "acceptance_datetime": r["acceptance_datetime"],
                        "ingestion_timestamp": r["acceptance_datetime"],
                        "acquired_disposed": r["acquired_disposed"],
                        "is_positive_buying_conviction": is_pos_conviction,
                        "is_open_market_purchase_evidence": is_pos_conviction,
                        "sec_url": f"https://www.sec.gov/Archives/edgar/data/{sec_cik}/{r['accession_number'].replace('-', '')}/{r['accession_number']}.txt" if sec_cik else "",
                    })

                buys = [t for t in sec_observations if t.get("is_positive_buying_conviction") or t.get("transaction_code") == "P" or t.get("signal_type") == "OPEN_MARKET_PURCHASE"]
                sells = [t for t in sec_observations if t.get("transaction_code") == "S" or t.get("signal_type") == "OPEN_MARKET_SALE"]
                buy_val = sum(t["total_value_usd"] for t in buys)
                sell_val = sum(t["total_value_usd"] for t in sells)

                if buy_val > sell_val and buy_val > 0:
                    dir_str = "BULLISH"
                elif sell_val > buy_val and sell_val > 0:
                    dir_str = "BEARISH"
                else:
                    dir_str = "NEUTRAL"

                conf_str = "HIGH" if len(sec_observations) >= 2 or max(buy_val, sell_val) > 500_000 else "MEDIUM"
                sec_distinct_dates = {
                    "latest_transaction_date": sec_observations[0].get("transaction_date"),
                    "latest_filing_date": sec_observations[0].get("filing_date"),
                    "latest_ingestion_date": sec_observations[0].get("acceptance_datetime"),
                }

                sec_feed = {
                    "status": "VERIFIED",
                    "status_label": "Verified",
                    "coverage": "SUPPORTED_SEC_ISSUER",
                    "cik": sec_cik,
                    "observations": sec_observations,
                    "observation_count": len(sec_observations),
                    "direction": dir_str,
                    "confidence": conf_str,
                    "explanation": f"{len(sec_observations)} Form 4 insider transactions evaluated in trailing {window_days}d (${(buy_val + sell_val):,.0f} volume).",
                    "provenance": "SEC EDGAR Form 4 (Local Verified Store)",
                    "distinct_dates": sec_distinct_dates,
                }
            elif has_older_in_db:
                sec_feed = {
                    "status": "NO_OBSERVATIONS",
                    "status_label": "No Observations",
                    "coverage": "SUPPORTED_SEC_ISSUER",
                    "cik": sec_cik,
                    "observations": [],
                    "observation_count": 0,
                    "direction": "NEUTRAL",
                    "confidence": "MEDIUM",
                    "explanation": f"0 insider transactions filed in trailing {window_days} days (routine corporate cadence).",
                    "provenance": "SEC EDGAR Form 4 (Local Verified Store)",
                    "distinct_dates": None,
                }
            else:
                sec_feed = {
                    "status": "PIPELINE_PENDING",
                    "status_label": "Pipeline Pending",
                    "coverage": "SUPPORTED_SEC_ISSUER" if sec_cik else "UNVERIFIED_ISSUER",
                    "cik": sec_cik,
                    "observations": [],
                    "observation_count": 0,
                    "direction": "UNVERIFIED",
                    "confidence": "NONE",
                    "explanation": "SEC Form 4 continuous ingestion paused under governed operational hold (HOLD_SEC_001).",
                    "provenance": "SEC EDGAR",
                    "distinct_dates": None,
                }

        # ── FEED 2: OPTIONS FLOW SWEEPS ───────────────────────────────────────
        if is_crypto:
            options_feed = {
                "status": "INAPPLICABLE",
                "status_label": "Exempt (Digital Asset)",
                "coverage": "EXEMPT_ASSET",
                "observations": [],
                "observation_count": 0,
                "direction": "UNVERIFIED",
                "confidence": "NONE",
                "explanation": "Equity options flow sweeps inapplicable to digital assets.",
                "provenance": "OPRA Consolidated Tape",
            }
        else:
            flow = SmartMoneyEngine.get_options_flow(clean_sym, include_curated=include_curated)
            if flow and len(flow) > 0:
                calls = [f for f in flow if "CALL" in (f.get("type") or "").upper()]
                puts = [f for f in flow if "PUT" in (f.get("type") or "").upper()]
                opt_dir = "BULLISH" if len(calls) > len(puts) else ("BEARISH" if len(puts) > len(calls) else "NEUTRAL")
                options_feed = {
                    "status": "VERIFIED" if not include_curated else "CURATED",
                    "status_label": "Verified" if not include_curated else "Curated",
                    "coverage": "SUPPORTED_OPTIONS_EQUITY",
                    "observations": flow,
                    "observation_count": len(flow),
                    "direction": opt_dir,
                    "confidence": "HIGH" if len(flow) >= 2 else "MEDIUM",
                    "explanation": f"{len(flow)} institutional options sweeps evaluated.",
                    "provenance": "OPRA Consolidated Tape" if not include_curated else "Curated Options Archive",
                }
            else:
                options_feed = {
                    "status": "PIPELINE_PENDING",
                    "status_label": "Pipeline Pending",
                    "coverage": "SUPPORTED_OPTIONS_EQUITY",
                    "observations": [],
                    "observation_count": 0,
                    "direction": "UNVERIFIED",
                    "confidence": "NONE",
                    "explanation": "Options sweep ingestion pipeline pending under governed deployment.",
                    "provenance": "OPRA Consolidated Tape",
                }

        # ── FEED 3: CONGRESSIONAL DISCLOSURES ─────────────────────────────────
        if is_crypto or is_etf:
            cong_feed = {
                "status": "INAPPLICABLE",
                "status_label": "Exempt",
                "coverage": "EXEMPT_ASSET",
                "observations": [],
                "observation_count": 0,
                "direction": "UNVERIFIED",
                "confidence": "NONE",
                "explanation": "STOCK Act congressional disclosures inapplicable to funds and digital assets.",
                "provenance": "House & Senate Ethics Disclosures",
                "distinct_dates": None,
            }
        else:
            trades = SmartMoneyEngine.get_congressional_trades(clean_sym, include_curated=include_curated)
            if trades and len(trades) > 0:
                buys = [t for t in trades if "Purchase" in (t.get("transaction_type") or "")]
                sells = [t for t in trades if "Sale" in (t.get("transaction_type") or "")]
                cong_dir = "BULLISH" if len(buys) > len(sells) else ("BEARISH" if len(sells) > len(buys) else "NEUTRAL")
                cong_dates = {
                    "latest_transaction_date": trades[0].get("transaction_date"),
                    "latest_filing_date": trades[0].get("filing_date"),
                }
                cong_feed = {
                    "status": "VERIFIED" if not include_curated else "CURATED",
                    "status_label": "Verified" if not include_curated else "Curated",
                    "coverage": "SUPPORTED_PUBLIC_EQUITY",
                    "observations": trades,
                    "observation_count": len(trades),
                    "direction": cong_dir,
                    "confidence": "HIGH" if len(trades) >= 2 else "MEDIUM",
                    "explanation": f"{len(trades)} congressional disclosures evaluated in trailing window.",
                    "provenance": "STOCK Act Disclosures (House Clerk & Senate EFD)",
                    "distinct_dates": cong_dates,
                }
            else:
                cong_feed = {
                    "status": "PIPELINE_PENDING",
                    "status_label": "Pipeline Pending",
                    "coverage": "SUPPORTED_PUBLIC_EQUITY",
                    "observations": [],
                    "observation_count": 0,
                    "direction": "UNVERIFIED",
                    "confidence": "NONE",
                    "explanation": "Live Congressional disclosure sync feed disconnected; real-time legislative updates pending.",
                    "provenance": "STOCK Act Disclosures",
                    "distinct_dates": None,
                }

        # ── CONFLUENCE INPUTS SYNTHESIS ───────────────────────────────────────
        has_any_verified = (
            sec_feed["status"] == "VERIFIED"
            or options_feed["status"] in ("VERIFIED", "CURATED")
            or cong_feed["status"] in ("VERIFIED", "CURATED")
        )

        confluence_inputs = None
        if has_any_verified:
            insider_buys = len([t for t in sec_feed["observations"] if t.get("is_positive_buying_conviction") or t.get("is_open_market_purchase_evidence")])
            insider_sells = len([t for t in sec_feed["observations"] if t.get("acquired_disposed") == "D"])
            has_insider_buy = insider_buys > 0
            insider_val = sum(t.get("total_value_usd", 0.0) for t in sec_feed["observations"] if t.get("is_positive_buying_conviction") or t.get("is_open_market_purchase_evidence"))
            insider_name = next((t.get("insider_name") for t in sec_feed["observations"] if t.get("is_positive_buying_conviction") or t.get("is_open_market_purchase_evidence")), "")
            has_congress_buy = any("Purchase" in (t.get("transaction_type") or "") for t in cong_feed["observations"])
            has_options = any("CALL" in (f.get("type") or "").upper() for f in options_feed["observations"])

            confluence_inputs = {
                "has_insider_buy": has_insider_buy,
                "insider_value_usd": insider_val,
                "insider_name": insider_name,
                "has_congress_buy": has_congress_buy,
                "has_options_flow": has_options,
                "net_institutional_flow_mil": 0.0,
                "insider_buy_count_90d": insider_buys,
                "insider_sell_count_90d": insider_sells,
            }

        if has_any_verified:
            overall_status = "AVAILABLE"
            directions = [f["direction"] for f in [sec_feed, options_feed, cong_feed] if f["direction"] != "UNVERIFIED"]
            if "BULLISH" in directions and "BEARISH" not in directions:
                overall_direction = "BULLISH"
            elif "BEARISH" in directions and "BULLISH" not in directions:
                overall_direction = "BEARISH"
            elif "BULLISH" in directions and "BEARISH" in directions:
                overall_direction = "MIXED"
            else:
                overall_direction = "NEUTRAL"
            overall_confidence = "HIGH" if any(f["confidence"] == "HIGH" for f in [sec_feed, options_feed, cong_feed]) else "MEDIUM"
        elif is_crypto or (is_etf and options_feed["status"] != "VERIFIED"):
            overall_status = "INAPPLICABLE"
            overall_direction = "UNVERIFIED"
            overall_confidence = "NONE"
        else:
            overall_status = "PIPELINE_PENDING"
            overall_direction = "UNVERIFIED"
            overall_confidence = "NONE"

        payload = {
            "status": overall_status,
            "window_days": window_days,
            "overall_direction": overall_direction,
            "overall_confidence": overall_confidence,
            "feeds": {
                "options_flow": options_feed,
                "sec_form4": sec_feed,
                "congressional_disclosures": cong_feed,
            },
            "congressTrades": cong_feed["observations"],
            "optionsFlow": options_feed["observations"],
            "secForm4Trades": sec_feed["observations"],
        }

        return {
            "overall_status": overall_status,
            "overall_direction": overall_direction,
            "overall_confidence": overall_confidence,
            "window_days": window_days,
            "confluence_inputs": confluence_inputs,
            "feeds": payload["feeds"],
            "congressTrades": payload["congressTrades"],
            "optionsFlow": payload["optionsFlow"],
            "secForm4Trades": payload["secForm4Trades"],
            "payload": payload,
        }


    @staticmethod
    def enrich_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich trade with quantitative legislative alignment and STOCK Act staleness metrics."""
        t_copy = trade.copy()
        days = t_copy.get("days_to_filing", 15)
        base_strength = t_copy.get("signal_strength", 90)

        alignment_score = calculate_legislative_alignment(t_copy)
        staleness_info = compute_filing_staleness(days, base_strength)

        t_copy["legislative_alignment_score"] = alignment_score
        t_copy["staleness_status"] = staleness_info["staleness_status"]
        t_copy["staleness_badge"] = staleness_info["staleness_badge"]
        t_copy["staleness_warning"] = staleness_info["staleness_warning"]
        t_copy["staleness_penalty"] = staleness_info["staleness_penalty"]
        t_copy["effective_signal_strength"] = staleness_info["effective_signal_strength"]
        t_copy["compliance_tier"] = staleness_info["compliance_tier"]
        return t_copy

    @staticmethod
    def get_congressional_trades(symbol: str = None, include_curated: bool = False) -> List[Dict[str, Any]]:
        """Fetch STOCK Act congressional disclosures. Requires verified live sync provider unless explicitly querying curated research archive."""
        if not include_curated:
            # Under Zero Fabricated Data Invariant, live congressional trades require active Capitol Trades/House Clerk feed.
            # Without live provider connection, return empty list rather than static archive masquerading as live.
            return []
        enriched_trades = [SmartMoneyEngine.enrich_trade(t) for t in CONGRESSIONAL_TRADES]
        if symbol:
            sym_clean = symbol.upper().strip()
            return [t for t in enriched_trades if t['ticker'] == sym_clean]
        return enriched_trades

    @staticmethod
    def get_sec_insider_trades(symbol: str = None, include_curated: bool = False) -> List[Dict[str, Any]]:
        """Fetch SEC Form 4 insider transactions. Requires verified live SEC EDGAR feed unless querying curated research archive."""
        if not include_curated:
            # Under Zero Fabricated Data Invariant, live insider trades require active SEC EDGAR ingestion.
            # Without live provider connection, return empty list rather than static archive masquerading as live.
            return []
        if symbol:
            sym_clean = symbol.upper().strip()
            return [s for s in SEC_FORM_4_TRADES if s['ticker'] == sym_clean]
        return SEC_FORM_4_TRADES

    @staticmethod
    def get_options_flow(symbol: str = None, include_curated: bool = False) -> List[Dict[str, Any]]:
        """Fetch options flow. Requires verified live provider (Polygon/OPRA) unless explicitly querying curated research archive."""
        if not include_curated:
            # Under Zero Fabricated Data Invariant, live options flow requires active OPRA websocket stream.
            # Without live provider connection, return empty list rather than static archive masquerading as live.
            return []
        if symbol:
            sym_clean = symbol.upper().strip()
            return [f for f in UNUSUAL_OPTIONS_FLOW if f['ticker'] == sym_clean]
        return UNUSUAL_OPTIONS_FLOW

    @classmethod
    def assess_options_flow(
        cls,
        symbol: str,
        live_options: Optional[List[Dict[str, Any]]] = None,
        include_curated: bool = False,
        is_verified_empty: bool = False,
    ) -> Dict[str, Any]:
        """Authoritatively evaluate Options Flow and Unusual Activity sub-domain."""
        clean_sym = symbol.upper().strip()

        is_crypto = (
            clean_sym.endswith("-USD")
            or clean_sym in ("BTC", "ETH", "SOL", "DOGE")
            or clean_sym in ("BTC-USD", "ETH-USD")
        )
        if is_crypto:
            return {
                "domain_id": "options_flow",
                "status": SmartMoneyDomainStatus.INAPPLICABLE.value,
                "status_label": "Exempt / Digital Asset",
                "available": False,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                "net_value_usd": 0.0,
                "provenance": "None",
                "disclosure": "Equity options flow sweeps inapplicable to digital assets.",
                "records": [],
            }

        if is_verified_empty:
            return {
                "domain_id": "options_flow",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "Verified No Unusual Activity",
                "available": True,
                "is_live": True,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "OPRA / Consolidated Options Tape",
                "disclosure": "Verified complete scan: 0 unusual options sweeps detected in evaluation window.",
                "records": [],
            }

        if live_options is not None:
            if len(live_options) > 0:
                bullish_count = 0
                bearish_count = 0
                total_premium = 0.0

                for sweep in live_options:
                    flow_type = (sweep.get("type") or sweep.get("put_call") or "").upper()
                    side = (sweep.get("side") or sweep.get("trade_side") or "").upper()
                    premium = float(sweep.get("premium") or sweep.get("value") or sweep.get("total_premium") or 0.0)
                    total_premium += premium

                    is_call = "CALL" in flow_type
                    is_put = "PUT" in flow_type

                    if side == "ASK":
                        if is_call:
                            bullish_count += 1
                        elif is_put:
                            bearish_count += 1
                    elif side == "BID":
                        if is_call:
                            bearish_count += 1
                        elif is_put:
                            bullish_count += 1
                    else:
                        if is_call:
                            bullish_count += 1
                        elif is_put:
                            bearish_count += 1

                if bullish_count > bearish_count:
                    direction = SmartMoneyDirection.BULLISH.value
                elif bearish_count > bullish_count:
                    direction = SmartMoneyDirection.BEARISH.value
                else:
                    direction = SmartMoneyDirection.MIXED.value

                return {
                    "domain_id": "options_flow",
                    "status": SmartMoneyDomainStatus.VERIFIED_SIGNAL.value,
                    "status_label": "Verified Sweeps",
                    "available": True,
                    "is_live": True,
                    "observation_count": len(live_options),
                    "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                    "directional_bias": direction,
                    "net_value_usd": total_premium,
                    "provenance": "OPRA / Consolidated Options Tape",
                    "disclosure": f"{len(live_options)} institutional options sweeps evaluated.",
                    "records": live_options,
                }
            elif is_verified_empty:
                return {
                    "domain_id": "options_flow",
                    "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                    "status_label": "Verified No Unusual Activity",
                    "available": True,
                    "is_live": True,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                    "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                    "net_value_usd": 0.0,
                    "provenance": "OPRA / Consolidated Options Tape",
                    "disclosure": "Verified complete scan: 0 unusual options sweeps detected in evaluation window.",
                    "records": [],
                }
            else:
                return {
                    "domain_id": "options_flow",
                    "status": SmartMoneyDomainStatus.UNAVAILABLE.value,
                    "status_label": "Unavailable (No Tape Stream)",
                    "available": False,
                    "is_live": False,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                    "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                    "net_value_usd": 0.0,
                    "provenance": "None",
                    "disclosure": "Live options tape unavailable (no active stream connection).",
                    "records": [],
                }

        if include_curated:
            records = cls.get_options_flow(clean_sym, include_curated=True)
            if len(records) > 0:
                return {
                    "domain_id": "options_flow",
                    "status": SmartMoneyDomainStatus.CURATED_RESEARCH.value,
                    "status_label": "Curated Research Sample",
                    "available": True,
                    "is_live": False,
                    "observation_count": len(records),
                    "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                    "directional_bias": SmartMoneyDirection.BULLISH.value,
                    "net_value_usd": sum(_parse_money(r.get("premium", 0.0)) for r in records),
                    "provenance": "Curated Institutional Flow Archive",
                    "disclosure": "Curated historical options sweeps archive loaded (not a live feed).",
                    "records": records,
                }
            return {
                "domain_id": "options_flow",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "No Curated Sweeps",
                "available": True,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "Curated Institutional Flow Archive",
                "disclosure": "0 curated options sweeps on record for this symbol.",
                "records": [],
            }

        return {
            "domain_id": "options_flow",
            "status": SmartMoneyDomainStatus.UNAVAILABLE.value,
            "status_label": "Unavailable / Stream Disconnected",
            "available": False,
            "is_live": False,
            "observation_count": 0,
            "lookback_days": cls.LOOKBACK_DAYS_OPTIONS,
            "directional_bias": SmartMoneyDirection.UNASSESSED.value,
            "net_value_usd": 0.0,
            "provenance": "None",
            "disclosure": "Live options tape unavailable (no active stream connection).",
            "records": [],
        }

    @classmethod
    def assess_sec_form_4(
        cls,
        symbol: str,
        asset_class: str = "EQUITY",
        security_type: str = "COMMON_STOCK",
        live_sec: Optional[Dict[str, Any]] = None,
        include_curated: bool = False,
        is_verified_empty: bool = False,
    ) -> Dict[str, Any]:
        """Authoritatively evaluate SEC Form 4 Insider Transactions sub-domain."""
        clean_sym = symbol.upper().strip()

        is_foreign_adr = (
            (len(clean_sym) == 5 and clean_sym.endswith(("Y", "F")))
            or clean_sym in cls.KNOWN_FOREIGN_ADRS
        )
        if is_foreign_adr:
            return {
                "domain_id": "sec_form_4",
                "status": SmartMoneyDomainStatus.INAPPLICABLE.value,
                "status_label": "Exempt / Foreign Issuer",
                "available": False,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                "net_value_usd": 0.0,
                "provenance": "Home Country Regulatory Authorities",
                "disclosure": "Foreign Private Issuer (FPI) exempt from US SEC Form 4 filings; local regulatory disclosures apply.",
                "coverage": "FOREIGN_FPI_EXEMPT",
                "records": [],
            }

        # Inapplicable check: Funds and Digital Assets do not file Form 4
        is_inapplicable = (
            asset_class.upper() in ("ETF", "CRYPTO")
            or security_type.upper() in ("ETF", "CRYPTO")
            or clean_sym in ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "VTI", "VOO", "BTC-USD", "ETH-USD")
            or clean_sym in cls.KNOWN_ETFS
        )
        if is_inapplicable:
            return {
                "domain_id": "sec_form_4",
                "status": SmartMoneyDomainStatus.INAPPLICABLE.value,
                "status_label": "Exempt / Inapplicable",
                "available": False,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                "net_value_usd": 0.0,
                "provenance": "SEC EDGAR / Exempt",
                "disclosure": "Corporate insider Form 4 transactions are inapplicable to fund structures and digital assets.",
                "records": [],
            }

        if is_verified_empty or (live_sec is not None and live_sec.get("status") == "AVAILABLE_NO_RESULTS"):
            return {
                "domain_id": "sec_form_4",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "Verified No Activity",
                "available": True,
                "is_live": True,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "SEC EDGAR Form 4 Feed",
                "disclosure": "Verified complete search: 0 open-market Form 4 transactions filed in trailing 90 days.",
                "records": [],
            }

        if live_sec is not None:
            sec_status = live_sec.get("status", "")
            items = live_sec.get("items") or live_sec.get("filings") or []

            if sec_status == "AVAILABLE_WITH_RESULTS" or len(items) > 0:
                buys = 0
                sells = 0
                buy_val = 0.0
                sell_val = 0.0
                late_filers = 0

                for f in items:
                    txn_type = (f.get("transaction_type") or f.get("type") or "").upper()
                    val = _parse_money(f.get("total_value") or f.get("value") or 0.0)
                    is_buy = "PURCHASE" in txn_type or "BUY" in txn_type or f.get("transaction_code") == "P"
                    is_sell = "SALE" in txn_type or "SELL" in txn_type or f.get("transaction_code") == "S"

                    # Latency check: Statutory requirement generally 2 business days
                    days_lag = f.get("days_to_filing") or f.get("filing_lag_days")
                    if days_lag is not None and days_lag > 4:  # >2 business days ~ >4 calendar days
                        late_filers += 1

                    if is_buy:
                        buys += 1
                        buy_val += val
                    elif is_sell:
                        sells += 1
                        sell_val += val

                net_val = buy_val - sell_val
                if buys > sells:
                    direction = SmartMoneyDirection.BULLISH.value
                elif sells > buys:
                    direction = SmartMoneyDirection.BEARISH.value
                else:
                    direction = SmartMoneyDirection.MIXED.value if (buys + sells) > 0 else SmartMoneyDirection.NEUTRAL.value

                return {
                    "domain_id": "sec_form_4",
                    "status": SmartMoneyDomainStatus.VERIFIED_SIGNAL.value,
                    "status_label": "Verified Form 4 Transactions",
                    "available": True,
                    "is_live": True,
                    "observation_count": len(items),
                    "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                    "directional_bias": direction,
                    "net_value_usd": net_val,
                    "provenance": "SEC EDGAR Form 4 Feed",
                    "disclosure": f"{len(items)} Form 4 insider transactions evaluated (Buys: {buys}, Sells: {sells}).",
                    "records": items,
                    "late_filers_count": late_filers,
                }
            elif sec_status == "AVAILABLE_NO_RESULTS" or is_verified_empty:
                return {
                    "domain_id": "sec_form_4",
                    "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                    "status_label": "Verified No Activity",
                    "available": True,
                    "is_live": True,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                    "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                    "net_value_usd": 0.0,
                    "provenance": "SEC EDGAR Form 4 Feed",
                    "disclosure": "Verified complete search: 0 open-market Form 4 transactions filed in trailing 90 days.",
                    "records": [],
                }
            elif sec_status in ("UPSTREAM_RATE_LIMITED", "UPSTREAM_TIMEOUT", "UPSTREAM_SERVER_ERROR", "TRANSPORT_INCOMPLETE", "STRUCTURALLY_INVALID", "UNAVAILABLE"):
                return {
                    "domain_id": "sec_form_4",
                    "status": SmartMoneyDomainStatus.UNAVAILABLE.value,
                    "status_label": "Provider Unavailable",
                    "available": False,
                    "is_live": False,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                    "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                    "net_value_usd": 0.0,
                    "provenance": "SEC EDGAR API",
                    "disclosure": f"SEC EDGAR Form 4 feed unavailable ({sec_status}).",
                    "records": [],
                }
            else:
                return {
                    "domain_id": "sec_form_4",
                    "status": SmartMoneyDomainStatus.PIPELINE_HELD.value,
                    "status_label": "Pipeline Pending (HOLD_SEC_001)",
                    "available": False,
                    "is_live": False,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                    "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                    "net_value_usd": 0.0,
                    "provenance": "SEC EDGAR System",
                    "disclosure": "SEC Form 4 continuous ingestion paused under governed operational hold (HOLD_SEC_001).",
                    "records": [],
                }

        if include_curated:
            records = cls.get_sec_insider_trades(clean_sym, include_curated=True)
            if len(records) > 0:
                buys = sum(1 for r in records if "Purchase" in r.get("transaction_type", ""))
                sells = sum(1 for r in records if "Sale" in r.get("transaction_type", ""))
                direction = SmartMoneyDirection.BULLISH.value if buys >= sells else SmartMoneyDirection.BEARISH.value
                return {
                    "domain_id": "sec_form_4",
                    "status": SmartMoneyDomainStatus.CURATED_RESEARCH.value,
                    "status_label": "Curated Research Sample",
                    "available": True,
                    "is_live": False,
                    "observation_count": len(records),
                    "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                    "directional_bias": direction,
                    "net_value_usd": sum(_parse_money(r.get("total_value", 0.0)) for r in records),
                    "provenance": "Curated SEC Form 4 Archive",
                    "disclosure": "Curated historical SEC Form 4 insider research dataset (not a live feed).",
                    "records": records,
                }
            return {
                "domain_id": "sec_form_4",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "No Curated Filings",
                "available": True,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "Curated SEC Form 4 Archive",
                "disclosure": "0 curated SEC Form 4 records on file for this symbol.",
                "records": [],
            }

        return {
            "domain_id": "sec_form_4",
            "status": SmartMoneyDomainStatus.PIPELINE_HELD.value,
            "status_label": "Pipeline Pending (HOLD_SEC_001)",
            "available": False,
            "is_live": False,
            "observation_count": 0,
            "lookback_days": cls.LOOKBACK_DAYS_INSIDER,
            "directional_bias": SmartMoneyDirection.UNASSESSED.value,
            "net_value_usd": 0.0,
            "provenance": "SEC EDGAR System",
            "disclosure": "SEC Form 4 continuous ingestion paused under governed operational hold (HOLD_SEC_001).",
            "records": [],
        }

    @classmethod
    def assess_congressional_trades(
        cls,
        symbol: str,
        live_congress: Optional[List[Dict[str, Any]]] = None,
        include_curated: bool = False,
        is_verified_empty: bool = False,
    ) -> Dict[str, Any]:
        """Authoritatively evaluate Congressional STOCK Act Disclosures sub-domain."""
        clean_sym = symbol.upper().strip()

        is_crypto = (
            clean_sym.endswith("-USD")
            or clean_sym in ("BTC", "ETH", "SOL", "DOGE")
            or clean_sym in ("BTC-USD", "ETH-USD")
        )
        if is_crypto:
            return {
                "domain_id": "congressional_stock_act",
                "status": SmartMoneyDomainStatus.INAPPLICABLE.value,
                "status_label": "Exempt / Digital Asset",
                "available": False,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                "net_value_usd": 0.0,
                "provenance": "None",
                "disclosure": "Congressional STOCK Act single-asset disclosures inapplicable to crypto tokens.",
                "records": [],
            }

        if is_verified_empty:
            return {
                "domain_id": "congressional_stock_act",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "Verified No Activity",
                "available": True,
                "is_live": True,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "House & Senate Official STOCK Act Portals",
                "disclosure": "Verified complete search: 0 congressional disclosures filed in trailing window.",
                "records": [],
            }

        if live_congress is not None:
            if len(live_congress) > 0:
                enriched = [cls.enrich_trade(t) for t in live_congress]
                buys = sum(1 for t in enriched if "Purchase" in t.get("transaction_type", ""))
                sells = sum(1 for t in enriched if "Sale" in t.get("transaction_type", ""))
                late_filers = sum(1 for t in enriched if t.get("staleness_status") == "LATE_FILER")

                if buys > sells:
                    direction = SmartMoneyDirection.BULLISH.value
                elif sells > buys:
                    direction = SmartMoneyDirection.BEARISH.value
                else:
                    direction = SmartMoneyDirection.MIXED.value

                return {
                    "domain_id": "congressional_stock_act",
                    "status": SmartMoneyDomainStatus.VERIFIED_SIGNAL.value,
                    "status_label": "Verified Congressional Disclosures",
                    "available": True,
                    "is_live": True,
                    "observation_count": len(enriched),
                    "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                    "directional_bias": direction,
                    "net_value_usd": 0.0,
                    "provenance": "House & Senate Official STOCK Act Portals",
                    "disclosure": f"{len(enriched)} congressional disclosures evaluated (Buys: {buys}, Sales: {sells}, Late filers: {late_filers}).",
                    "records": enriched,
                    "late_filers_count": late_filers,
                }
            elif is_verified_empty:
                return {
                    "domain_id": "congressional_stock_act",
                    "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                    "status_label": "Verified No Activity",
                    "available": True,
                    "is_live": True,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                    "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                    "net_value_usd": 0.0,
                    "provenance": "House & Senate Official STOCK Act Portals",
                    "disclosure": "Verified complete search: 0 congressional disclosures filed in trailing window.",
                    "records": [],
                }
            else:
                return {
                    "domain_id": "congressional_stock_act",
                    "status": SmartMoneyDomainStatus.UNAVAILABLE.value,
                    "status_label": "Unavailable (Disconnected)",
                    "available": False,
                    "is_live": False,
                    "observation_count": 0,
                    "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                    "directional_bias": SmartMoneyDirection.UNASSESSED.value,
                    "net_value_usd": 0.0,
                    "provenance": "None",
                    "disclosure": "Congressional sync feed disconnected.",
                    "records": [],
                }

        records = cls.get_congressional_trades(clean_sym, include_curated=include_curated)
        if include_curated:
            if len(records) > 0:
                buys = sum(1 for t in records if "Purchase" in t.get("transaction_type", ""))
                sells = sum(1 for t in records if "Sale" in t.get("transaction_type", ""))
                direction = SmartMoneyDirection.BULLISH.value if buys >= sells else SmartMoneyDirection.BEARISH.value
                return {
                    "domain_id": "congressional_stock_act",
                    "status": SmartMoneyDomainStatus.CURATED_RESEARCH.value,
                    "status_label": "Curated Research Sample",
                    "available": True,
                    "is_live": False,
                    "observation_count": len(records),
                    "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                    "directional_bias": direction,
                    "net_value_usd": 0.0,
                    "provenance": "Curated STOCK Act Research Archive",
                    "disclosure": "Curated historical STOCK Act research dataset (August 2026). Not a real-time live trading feed.",
                    "records": records,
                    "late_filers_count": sum(1 for t in records if t.get("staleness_status") == "LATE_FILER"),
                }
            return {
                "domain_id": "congressional_stock_act",
                "status": SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
                "status_label": "No Curated Disclosures",
                "available": True,
                "is_live": False,
                "observation_count": 0,
                "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
                "directional_bias": SmartMoneyDirection.NEUTRAL.value,
                "net_value_usd": 0.0,
                "provenance": "Curated STOCK Act Research Archive",
                "disclosure": "0 curated congressional disclosures on record for this symbol.",
                "records": [],
            }

        return {
            "domain_id": "congressional_stock_act",
            "status": SmartMoneyDomainStatus.PIPELINE_HELD.value,
            "status_label": "Pipeline Pending (HOLD_SEC_001)",
            "available": False,
            "is_live": False,
            "observation_count": 0,
            "lookback_days": cls.LOOKBACK_DAYS_CONGRESS,
            "directional_bias": SmartMoneyDirection.UNASSESSED.value,
            "net_value_usd": 0.0,
            "provenance": "House & Senate Official STOCK Act Portals",
            "disclosure": "Congressional STOCK Act continuous sync paused under governed operational hold (HOLD_SEC_001).",
            "records": [],
        }

    @classmethod
    def assess_symbol(
        cls,
        symbol: str,
        asset_class: str = "EQUITY",
        security_type: str = "COMMON_STOCK",
        include_curated: bool = False,
        live_options: Optional[List[Dict[str, Any]]] = None,
        live_sec: Optional[Dict[str, Any]] = None,
        live_congress: Optional[List[Dict[str, Any]]] = None,
        options_verified_empty: bool = False,
        sec_verified_empty: bool = False,
        congress_verified_empty: bool = False,
    ) -> Dict[str, Any]:
        """
        Build the unified, canonical Single-Asset Smart Money Assessment.
        Enforces epistemic honesty:
        - Never converts missing data into bullish, bearish, or verified-neutral signals.
        - Only emits VERIFIED_NO_ACTIVITY when search confirms completed zero records.
        - Emits PIPELINE_HELD / UNAVAILABLE when continuous feeds are offline under HOLD_SEC_001.
        """
        clean_sym = symbol.upper().strip()

        options_sub = cls.assess_options_flow(
            symbol=clean_sym,
            live_options=live_options,
            include_curated=include_curated,
            is_verified_empty=options_verified_empty,
        )

        sec_sub = cls.assess_sec_form_4(
            symbol=clean_sym,
            asset_class=asset_class,
            security_type=security_type,
            live_sec=live_sec,
            include_curated=include_curated,
            is_verified_empty=sec_verified_empty,
        )

        cong_sub = cls.assess_congressional_trades(
            symbol=clean_sym,
            live_congress=live_congress,
            include_curated=include_curated,
            is_verified_empty=congress_verified_empty,
        )

        subdomains = {
            "options_flow": options_sub,
            "sec_form_4": sec_sub,
            "congressional_stock_act": cong_sub,
        }

        # Determine composite overall status
        statuses = [s["status"] for s in subdomains.values()]
        active_holds = []
        if SmartMoneyDomainStatus.PIPELINE_HELD.value in statuses:
            active_holds.append("HOLD_SEC_001")

        # Resolve overall_status
        if include_curated:
            overall_status = SmartMoneyDomainStatus.CURATED_RESEARCH.value
            overall_label = "Curated Research Dataset"
        elif all(s == SmartMoneyDomainStatus.INAPPLICABLE.value for s in statuses):
            overall_status = SmartMoneyDomainStatus.INAPPLICABLE.value
            overall_label = "Exempt / Inapplicable"
        elif any(s == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value for s in statuses):
            if any(s in (SmartMoneyDomainStatus.UNAVAILABLE.value, SmartMoneyDomainStatus.PIPELINE_HELD.value) for s in statuses):
                overall_status = SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
                overall_label = "Partial Coverage (Signal Active)"
            else:
                overall_status = SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
                overall_label = "Verified Smart Money Signals"
        elif all(s in (SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value, SmartMoneyDomainStatus.INAPPLICABLE.value) for s in statuses):
            overall_status = SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
            overall_label = "Verified No Activity (Comprehensive Search)"
        elif all(s in (SmartMoneyDomainStatus.PIPELINE_HELD.value, SmartMoneyDomainStatus.UNAVAILABLE.value, SmartMoneyDomainStatus.INAPPLICABLE.value) for s in statuses):
            if SmartMoneyDomainStatus.PIPELINE_HELD.value in statuses:
                overall_status = SmartMoneyDomainStatus.PIPELINE_HELD.value
                overall_label = "Pipeline Pending (HOLD_SEC_001 Active)"
            else:
                overall_status = SmartMoneyDomainStatus.UNAVAILABLE.value
                overall_label = "Feeds Unavailable"
        else:
            overall_status = SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
            overall_label = "Partial Coverage"

        # Pillar eligibility: Only scoreable if verified signals, verified zero, or curated research
        pillar_eligibility = overall_status in (
            SmartMoneyDomainStatus.VERIFIED_SIGNAL.value,
            SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value,
            SmartMoneyDomainStatus.CURATED_RESEARCH.value,
            SmartMoneyDomainStatus.PARTIAL_COVERAGE.value,
        )

        # Compute composite directional score and pillar contribution
        if not pillar_eligibility:
            pillar_score = None
            pillar_status = "unavailable"
            reweight_required = True
            directional_bias = SmartMoneyDirection.UNASSESSED.value
        elif overall_status == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value:
            pillar_score = 50.0
            pillar_status = "neutral"
            reweight_required = False
            directional_bias = SmartMoneyDirection.NEUTRAL.value
        else:
            # Active signal computation
            base_score = 50.0
            score_delta = 0.0

            # 1. Options Flow contribution (+/- 15 pts)
            opt_dir = options_sub["directional_bias"]
            if opt_dir == SmartMoneyDirection.BULLISH.value:
                score_delta += 15.0
            elif opt_dir == SmartMoneyDirection.BEARISH.value:
                score_delta -= 15.0

            # 2. Form 4 Insider contribution (+/- 25 pts)
            sec_dir = sec_sub["directional_bias"]
            if sec_dir == SmartMoneyDirection.BULLISH.value:
                score_delta += 25.0
            elif sec_dir == SmartMoneyDirection.BEARISH.value:
                score_delta -= 20.0

            # 3. Congressional contribution (+/- 20 pts, penalized for staleness)
            cong_dir = cong_sub["directional_bias"]
            late_cong = cong_sub.get("late_filers_count", 0)
            if cong_dir == SmartMoneyDirection.BULLISH.value:
                boost = 20.0 if late_cong == 0 else 8.0  # Staleness decay
                score_delta += boost
            elif cong_dir == SmartMoneyDirection.BEARISH.value:
                score_delta -= 15.0

            composite_score = min(95.0, max(15.0, base_score + score_delta))
            pillar_score = round(composite_score, 1)

            if pillar_score >= 65.0:
                pillar_status = "positive"
                directional_bias = SmartMoneyDirection.BULLISH.value
            elif pillar_score <= 40.0:
                pillar_status = "warning"
                directional_bias = SmartMoneyDirection.BEARISH.value
            else:
                pillar_status = "neutral"
                directional_bias = SmartMoneyDirection.MIXED.value if score_delta != 0 else SmartMoneyDirection.NEUTRAL.value

            reweight_required = False

        # Extract backward-compatible flags and lists
        opt_records = options_sub.get("records", [])
        sec_records = sec_sub.get("records", [])
        cong_records = cong_sub.get("records", [])

        has_insider_buy = any("PURCHASE" in (r.get("transaction_type") or "").upper() or r.get("transaction_code") == "P" for r in sec_records)
        has_insider_sell = any("SALE" in (r.get("transaction_type") or "").upper() or r.get("transaction_code") == "S" for r in sec_records)
        has_congress_buy = any("Purchase" in r.get("transaction_type", "") for r in cong_records)
        has_congress_sell = any("Sale" in r.get("transaction_type", "") for r in cong_records)
        has_options_flow = len(opt_records) > 0

        insider_val = sum(_parse_money(r.get("total_value", 0.0)) for r in sec_records if "PURCHASE" in (r.get("transaction_type") or "").upper() or r.get("transaction_code") == "P")

        coverage_summary = (
            f"Smart Money Overall: {overall_label}. "
            f"Options: {options_sub['status_label']} ({options_sub['observation_count']}). "
            f"SEC Form 4: {sec_sub['status_label']} ({sec_sub['observation_count']}). "
            f"Congress: {cong_sub['status_label']} ({cong_sub['observation_count']})."
        )

        return {
            "symbol": clean_sym,
            "overall_status": overall_status,
            "overall_status_label": overall_label,
            "pillar_eligibility": pillar_eligibility,
            "pillar_status": pillar_status,
            "pillar_score": pillar_score,
            "reweight_required": reweight_required,
            "directional_bias": directional_bias,
            "active_holds": active_holds,
            "coverage_summary": coverage_summary,
            "coverage_window_days": {
                "options_flow": cls.LOOKBACK_DAYS_OPTIONS,
                "sec_form_4": cls.LOOKBACK_DAYS_INSIDER,
                "congressional_stock_act": cls.LOOKBACK_DAYS_CONGRESS,
            },
            "subdomains": subdomains,
            # Backward-compatible fields for confluence_engine & decision_trace
            "has_insider_buy": has_insider_buy,
            "has_insider_sell": has_insider_sell,
            "insider_value_usd": insider_val,
            "has_congress_buy": has_congress_buy,
            "has_congress_sell": has_congress_sell,
            "has_options_flow": has_options_flow,
            "options_flow": opt_records,
            "optionsFlow": opt_records,
            "sec_insider_trades": sec_records,
            "secInsiderTrades": sec_records,
            "secForm4Trades": sec_records,
            "congress_trades": cong_records,
            "congressTrades": cong_records,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def get_smart_money_overview(include_curated: bool = False) -> Dict[str, Any]:
        if not include_curated:
            return {
                'status': 'UNAVAILABLE',
                'is_live': False,
                'available': False,
                'dataset_date': None,
                'disclosure': 'Live Smart Money regulatory ingestion feed disconnected. Pass include_curated=true to query archived research disclosures.',
                'total_congress_filings_30d': None,
                'total_sec_insiders_30d': None,
                'net_political_sentiment': None,
                'top_congress_bought_sector': None,
                'unusual_flow_volume_today': None,
                'call_to_put_dollar_ratio': None,
                'late_filers_count': 0,
                'fresh_trades_count': 0,
                'congress_trades': [],
                'sec_insider_trades': [],
                'options_flow': [],
            }

        enriched_trades = [SmartMoneyEngine.enrich_trade(t) for t in CONGRESSIONAL_TRADES]
        late_filers = [t for t in enriched_trades if t.get("staleness_status") == "LATE_FILER"]
        fresh_trades = [t for t in enriched_trades if t.get("staleness_status") == "FRESH"]

        total_congress = len(enriched_trades)
        total_sec = len(SEC_FORM_4_TRADES)

        purchases = [t for t in enriched_trades if "Purchase" in t.get("transaction_type", "")]
        sales = [t for t in enriched_trades if "Sale" in t.get("transaction_type", "")]

        if total_congress > 0:
            purchase_pct = round((len(purchases) / total_congress) * 100, 1)
            sentiment_str = f"Bullish ({purchase_pct}% Purchases)" if purchase_pct >= 50 else f"Bearish ({round(100 - purchase_pct, 1)}% Sales)"
        else:
            sentiment_str = "Neutral"

        sector_counts: Dict[str, int] = {}
        for t in purchases:
            s = t.get("sector")
            if s:
                sector_counts[s] = sector_counts.get(s, 0) + 1
        top_sector = max(sector_counts.items(), key=lambda x: x[1])[0] if sector_counts else "Unclassified"

        return {
            'status': 'CURATED',
            'is_live': False,
            'available': True,
            'dataset_date': '2026-08-28',
            'disclosure': 'Curated historical regulatory research dataset (August 2026). Not a real-time live trading feed.',
            'total_congress_filings_30d': total_congress,
            'total_sec_insiders_30d': total_sec,
            'net_political_sentiment': sentiment_str,
            'top_congress_bought_sector': top_sector,
            'unusual_flow_volume_today': None,
            'call_to_put_dollar_ratio': None,
            'late_filers_count': len(late_filers),
            'fresh_trades_count': len(fresh_trades),
            'congress_trades': enriched_trades,
            'sec_insider_trades': SEC_FORM_4_TRADES,
            'options_flow': SmartMoneyEngine.get_options_flow(include_curated=include_curated),
        }
