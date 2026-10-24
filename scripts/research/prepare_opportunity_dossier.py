#!/usr/bin/env python3
"""
scripts/research/prepare_opportunity_dossier.py

ARX Terminal — Opportunity Dossier Scaling Pilot: Reusable Preparation & Validation Engine.
Prepares, validates, solves, and packages draft opportunity dossiers for the 20-company pilot cohort.

Governing Contract: CONTRACT-ARX-OPP-PILOT-SCALING-V1
Invariants:
- Preserves the frozen 5-company baseline (FIX, DUOL, IREN, CORZ, MU) from ODP-05.
- Strictly isolates new pilot expansion draft dossiers from production publication snapshot.
- Reuses deterministic valuation calculations (run_fcff_model, solve_reverse_dcf).
- Distinguishes documented zero debt from unevidenced or missing debt.
- Strictly applies Lease-as-Operating-Expense convention (ASC 842) for retail brands.
- Enforces fail-closed valuation suppression (IREN, CORZ) and qualification (MU, PLTR, KLAC, SNPS).
- Generates 100% reproducible outputs without current-clock timestamps in research facts.
- Tracks full provenance: SEC EDGAR accessions, primary URLs, reporting periods, and SHA-256 digests.
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
)

logger = logging.getLogger("prepare_opportunity_dossier")

# Baseline constants
FROZEN_ODP05_LEDGER_PATH = REPO_ROOT / "docs/research/opportunity_discovery/odp05_discovery_ledger.json"
FROZEN_ODP05_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d"
BASELINE_COHORT = ["FIX", "DUOL", "IREN", "CORZ", "MU"]

# Pilot additions cohort: 14 candidates (excluding rejected PLTR per product owner direction)
PILOT_ADDITIONS_COHORT = [
    "ULTA", "MEDP", "DECK", "KLAC", "LULU",
    "EME", "VRT", "ANET", "ISRG",
    "CDNS", "PWR", "SNPS", "NOW", "ELF"
]

TOTAL_PILOT_COHORT = BASELINE_COHORT + PILOT_ADDITIONS_COHORT

# Curated Quality Watch & Selective Publication constants
PUBLISHED_WATCH_SYMBOLS = ["FIX", "DUOL", "ULTA", "MEDP", "ISRG", "DECK"]
BASELINE_WATCH_COHORT = ["FIX", "DUOL"]
HELD_HISTORICAL_SYMBOLS = ["IREN", "CORZ"]
EXCLUDED_HISTORICAL_SYMBOLS = ["MU"]
EXCLUDED_DRAFT_SYMBOLS = ["PLTR"]

DEFAULT_PILOT_LEDGER_PATH = REPO_ROOT / "docs/research/opportunity_discovery/pilot_scaling_ledger.json"
DEFAULT_PILOT_MANIFEST_PATH = REPO_ROOT / "docs/research/opportunity_discovery/pilot_scaling_manifest.json"
DEFAULT_FRONTEND_DRAFT_PATH = REPO_ROOT / "frontend/lib/opportunityDossierPilotDrafts.json"
DEFAULT_DOC_MATRIX_PATH = REPO_ROOT / "docs/research/opportunity_discovery/pilot_scaling_cohort_matrix.md"
DEFAULT_PUBLISHED_MANIFEST_PATH = REPO_ROOT / "docs/research/opportunity_discovery/published_watch_manifest.json"
DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH = REPO_ROOT / "frontend/lib/publishedWatchManifest.json"

BASELINE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "FIX": {
        "sector_industry": "Industrials / Specialty Contractors",
        "business_model": "Modular thermal & HVAC infrastructure for data centers",
    },
    "DUOL": {
        "sector_industry": "Technology / Application Software",
        "business_model": "Subscription language learning & mobile education",
    },
    "IREN": {
        "sector_industry": "Technology / Data Center Infrastructure",
        "business_model": "Data center power infrastructure & AI cloud transition",
    },
    "CORZ": {
        "sector_industry": "Technology / Data Center Infrastructure",
        "business_model": "Post-reorganization AI hyperscale colocation",
    },
    "MU": {
        "sector_industry": "Technology / Semiconductors",
        "business_model": "Cyclical memory (DRAM, HBM3E, NAND) hardware",
    },
}

MARKET_SESSION_DATE = "2026-10-16"
FILING_AVAILABILITY_CUTOFF_UTC = "2026-10-16T20:00:00Z"
AUTHORING_TIMESTAMP_UTC = "2026-10-21T16:00:00Z"

# Authoritative specification dictionary for the 15 additions
PILOT_SPECS: Dict[str, Dict[str, Any]] = {
    "ULTA": {
        "company_name": "Ulta Beauty, Inc.",
        "cik": "0001403568",
        "exchange": "NASDAQ",
        "opportunity_route": "Consumer Cyclical / Specialty Retail",
        "sector_industry": "Consumer Cyclical / Specialty Retail",
        "business_model": "Omnichannel specialty beauty retailer with salon services and 44M+ loyalty members",
        "reference_price": 592.94,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "evidence_completeness": "SOURCE_MAPPED_UNAUDITED",
        "review_notes": "Representative candidate prepared with complete primary filing source-to-input tracing. Awaiting independent financial research review.",
        "source_evidence_tracing": {
            "representative_status": "REPRESENTATIVE_PILOT_CANDIDATE",
            "company_identity": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "entity_name": "Ulta Beauty, Inc.",
                "cik": "0001403568",
                "sic": "5990",
                "exchange": "NASDAQ",
                "source_url": "https://data.sec.gov/submissions/CIK0001403568.json",
                "filing_accessions": ["0001104659-26-035243", "0001104659-26-102442"]
            },
            "revenue": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 12955517000.0,
                "derivation_formula": "FY2025 Revenues ($12,392,820,000) - Q2 2025 6M Revenues ($5,636,836,000) + Q2 2026 6M Revenues ($6,199,533,000)",
                "xbrl_tag": "us-gaap:Revenues",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2026-01-31",
                        "filing_date": "2026-03-26",
                        "accession": "0001104659-26-035243",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 12392820000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-08-01",
                        "filing_date": "2026-08-27",
                        "accession": "0001104659-26-102442",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 6199533000.0
                    }
                ]
            },
            "operating_income": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 1614258000.0,
                "operating_margin_pct": 12.46,
                "derivation_formula": "FY2025 Operating Income ($1,532,992,000) - Q2 2025 6M Operating Income ($746,631,000) + Q2 2026 6M Operating Income ($827,897,000)",
                "xbrl_tag": "us-gaap:OperatingIncomeLoss",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2026-01-31",
                        "filing_date": "2026-03-26",
                        "accession": "0001104659-26-035243",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 1532992000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-08-01",
                        "filing_date": "2026-08-27",
                        "accession": "0001104659-26-102442",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 827897000.0
                    }
                ]
            },
            "cash_and_investments": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_cash_and_investments_usd": 213451000.0,
                "cash_and_equivalents_usd": 158451000.0,
                "short_term_investments_usd": 55000000.0,
                "xbrl_tags": [
                    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "us-gaap:ShortTermInvestments"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Balance Sheets"
                }
            },
            "funded_debt": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_financial_debt_usd": 339578000.0,
                "is_zero_debt": False,
                "support_disclosure": "Form 10-Q Part I Item 1 Consolidated Balance Sheets and Note 7 ('Debt') disclose $339,578,000 outstanding under revolving credit facility borrowings (proceeds of $1,358,274,000 and repayments of $1,080,032,000 during 6M ended August 1, 2026).",
                "xbrl_tag": "us-gaap:ShortTermBorrowings",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Balance Sheets & Note 7 (Debt)",
                    "reported_value": 339578000.0
                }
            },
            "lease_treatment": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "current_operating_lease_liabilities_usd": 312648000.0,
                "noncurrent_operating_lease_liabilities_usd": 1871805000.0,
                "total_lease_liabilities_usd": 2184453000.0,
                "accounting_standard": "ASC 842",
                "valuation_convention": "Lease-as-Operating-Expense convention: In primary GAAP filings, retail store and distribution center lease costs are primarily included in Cost of Sales, and corporate store support center lease costs in SG&A. Under the analyst valuation convention, lease operating expenses reduce NOPAT and CFO; operating lease liabilities ($2.18B) are therefore excluded from the net debt bridge to prevent double deduction of lease obligations.",
                "xbrl_tags": [
                    "us-gaap:OperatingLeaseLiabilityCurrent",
                    "us-gaap:OperatingLeaseLiabilityNoncurrent"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Balance Sheets & Note 6 (Leases)"
                }
            },
            "shares_outstanding": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "share_count": 43062000.0,
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "measurement_period": "Three months ended 2026-08-01",
                "xbrl_tag": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Statements of Income"
                }
            },
            "reference_price": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "reference_price": 592.94,
                "reference_date": "2026-10-16",
                "timestamp_utc": "2026-10-16T20:00:00Z",
                "source_system": "ARX MarketStore regular close tape",
                "source_store": "data/research/market_data/bounded_150_dev_market_store.db",
                "table": "asset_ohlcv_daily"
            },
            "valuation_model": {
                "fact_status": "DERIVED_CALCULATION",
                "model_type": "Single-stage Reverse-DCF FCFF (Gordon Growth Terminal Value)",
                "solved_5y_revenue_cagr_pct": 15.45,
                "assumptions": {
                    "fact_status": "ANALYST_ASSUMPTION",
                    "operating_margin_pct": 12.0,
                    "wacc_pct": 9.0,
                    "terminal_growth_pct": 2.0,
                    "tax_rate_pct": 23.5,
                    "reinvestment_rate_pct": 15.0
                },
                "roundtrip_value": 593.00,
                "roundtrip_delta_cents": 6.33
            },
            "narrative_claims": [
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Ulta Beauty has over 44 million active loyalty program members, representing over 95% of total sales.",
                    "locator": "Form 10-K for FY ended 2026-01-31, Item 1 Business — Loyalty Program"
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "DERIVED_CALCULATION",
                    "statement": "TTM operating cash flow of $1,567.8M ($1.57B) derived from FY25 operating cash flow ($1,502.8M) less prior-year 6M ($316.5M) plus current-year 6M ($381.6M).",
                    "locator": "Form 10-K Item 8 & Form 10-Q Part I Item 1 Consolidated Statements of Cash Flows"
                },
                {
                    "claim_type": "OPPOSING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Quarterly comparable sales growth slowed to 3.8% from 6.7% a year earlier, while gross margin declined to 39.1% from 39.2%, primarily reflecting Space NK business mix. Operating margin nevertheless improved to 12.5% from 12.4%.",
                    "locator": "Form 10-Q for Q2 ended 2026-08-01, Part I Item 2 MD&A — Results of Operations"
                },
                {
                    "claim_type": "MONITORING_CRITERIA",
                    "fact_status": "ANALYST_ASSUMPTION",
                    "statement": "Operating margin compression below 10.0% or negative comparable store sales for two consecutive fiscal quarters triggers thesis invalidation.",
                    "locator": "ARX Analytical Thesis Invalidation Rule"
                }
            ],
            "selection_rationale": {
                "why_this_business": "Ulta Beauty is the premier US specialty beauty retailer with a loyalty program exceeding 44 million active members representing >95% of sales, generating $1.61B TTM operating income (12.46% margin) and $1.57B operating cash flow.",
                "why_research_now": "Deceleration in comparable store sales (3.8% in Q2 2026 vs 6.7% in Q2 2025) and cosmetic promotional intensity drove a valuation reset, creating an actionable setup to monitor customer retention, margin defense, and pullbacks toward base valuation ($430 - $450).",
                "price_requirements": "At the $592.94 reference price, assuming a 12.0% operating margin, 9.0% WACC, 2.0% terminal growth, 23.5% tax rate, and 15.0% reinvestment rate, the reverse-DCF implies a 5-year revenue CAGR of 15.45% ($593.00 roundtrip value). Solved growth reflects model discount mechanics rather than observed consensus.",
                "differentiated_view": "Proprietary Ultamate Rewards loyalty data and prestige/mass assortment bridge provide structural store traffic retention and promotional agility that pure-play e-commerce and department store competitors struggle to replicate.",
                "assessment_changes": "Strengthening: Comparable store sales acceleration >6.0% with gross margin expansion >40.0%. Weakening/Invalidation: Operating margin compression below 10.0% or negative comparable store sales for two consecutive fiscal quarters.",
                "queue_priority": "Serves as the representative anchor for retail ASC 842 lease operating expense accounting, providing verified primary source tracing on credit facility borrowings ($339.6M Note 7) and cash/ST investments ($213.5M)."
            }
        },
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-01-31",
                "filing_date": "2026-03-26",
                "accession_number": "0001104659-26-035243",
                "doc_name": "ulta-20260131x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-08-01",
                "filing_date": "2026-08-27",
                "accession_number": "0001104659-26-102442",
                "doc_name": "ulta-20260801x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 12955517000.0,
            "revenue_period": "TTM (Ended 2026-08-01)",
            "operating_income": 1614258000.0,
            "operating_margin_pct": 12.46,
            "cash_and_equivalents": 158451000.0,
            "short_term_investments": 55000000.0,
            "total_cash_and_investments": 213451000.0,
            "total_financial_debt": 339578000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 1871805000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied under ASC 842: store and supply chain lease costs are recognized primarily in Cost of Sales and SG&A; lease liabilities ($2,184.5M total, $1,871.8M noncurrent) are excluded from the net debt bridge to avoid double deduction.",
            "net_debt_bridge": 126127000.0,
            "bridge_note": "Funded debt of $339.6M (revolving credit facility borrowings under Note 7) less total cash and ST investments ($213.5M) yields positive net debt bridge of $126.1M, deducted from Enterprise Value in DCF equity bridge.",
            "diluted_shares": 43062000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 43062000.0,
                "measurement_period": "Three months ended 2026-08-01",
                "rationale": "Diluted weighted average shares outstanding reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 12.0,
            "wacc": 9.0,
            "terminal_g": 2.0,
            "tax_rate": 23.5,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 4.0, "margin_pct": 10.5, "wacc_pct": 9.5, "g_pct": 1.5, "tax_pct": 24.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 8.0, "margin_pct": 12.0, "wacc_pct": 9.0, "g_pct": 2.0, "tax_pct": 23.5, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 14.0, "margin_pct": 13.5, "wacc_pct": 8.5, "g_pct": 2.5, "tax_pct": 23.0, "reinvest_pct": 14.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Ulta generates dependable operating cash flows ($1.57B TTM operating cash flow) with a liquid balance sheet ($213.5M cash and ST investments against $339.6M credit facility borrowings) and 44M+ loyalty members driving >95% of sales.",
            "strongest_opposing_evidence": "Quarterly comparable sales growth slowed to 3.8% from 6.7% a year earlier, while gross margin declined to 39.1% from 39.2%, primarily reflecting Space NK business mix. Operating margin nevertheless improved to 12.5% from 12.4%. Under the stated valuation assumptions, the reference price implies 15.45% five-year revenue CAGR.",
            "main_uncertainty": "Durability of store-level traffic and margin defense against expanding promotional cadence across prestige cosmetic brands.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($430 - $450) or evidence of sustained comp-store sales acceleration >6%.",
            "confidence_assessment": "High confidence in accounting, Note 7 debt disclosures, and lease classification; medium confidence in long-term terminal operating margin floor.",
            "financing_dilution_exposure": "Negative net dilution; aggressive share buybacks have retired over 13.6% of outstanding shares over the trailing three fiscal years.",
            "observable_invalidation_trigger": "Operating margin compression below 10.0% or negative comparable store sales for two consecutive fiscal quarters.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (expected December 2026) for holiday promotional cadence and loyalty membership growth.",
            "presentation_summary": "Ulta Beauty demonstrates high capital returns supported by 44M+ loyalty members and strong cash generation. Form 10-Q Note 7 reflects $339.6M in revolving credit facility debt against $213.5M cash/ST investments ($126.1M net debt bridge). At $592.94, the valuation implies a demanding 15.45% 5-year revenue CAGR against intense prestige retail competition. Retained on watch for pullbacks toward base valuation range ($430 - $450)."
        }
    },
    "MEDP": {
        "company_name": "Medpace Holdings, Inc.",
        "cik": "0001668397",
        "exchange": "NASDAQ",
        "opportunity_route": "Healthcare / Medical Research",
        "sector_industry": "Healthcare / Medical Research",
        "business_model": "Full-service Phase I-IV clinical contract research organization (CRO) focused on small-to-midsize biopharma",
        "reference_price": 618.47,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "review_notes": "Scaling addition candidate prepared with complete primary filing source-to-input tracing, verified operating profitability, positive operating cash generation, zero funded debt, and structured selection record.",
        "source_evidence_tracing": {
            "representative_status": "EXPANSION_PILOT_CANDIDATE",
            "company_identity": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "entity_name": "Medpace Holdings, Inc.",
                "cik": "0001668397",
                "sic": "8731",
                "exchange": "NASDAQ",
                "source_url": "https://data.sec.gov/submissions/CIK0001668397.json",
                "filing_accessions": [
                    "0001668397-26-000006",
                    "0001668397-26-000025"
                ]
            },
            "revenue": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 2782288000.0,
                "derivation_formula": "FY2025 Revenues ($2,530,234,000) - Q2 2025 6M Revenues ($1,161,881,000) + Q2 2026 6M Revenues ($1,413,935,000)",
                "xbrl_tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2025-12-31",
                        "filing_date": "2026-02-10",
                        "accession": "0001668397-26-000006",
                        "location": "Item 8 Consolidated Statements of Operations",
                        "reported_value": 2530234000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-23",
                        "accession": "0001668397-26-000025",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (6 Months Ended)",
                        "reported_value": 1161881000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-23",
                        "accession": "0001668397-26-000025",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (6 Months Ended)",
                        "reported_value": 1413935000.0
                    }
                ]
            },
            "operating_income": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 583404000.0,
                "operating_margin_pct": 20.97,
                "derivation_formula": "FY2025 Operating Income ($534,935,000) - Q2 2025 6M Operating Income ($239,858,000) + Q2 2026 6M Operating Income ($288,327,000)",
                "xbrl_tag": "us-gaap:OperatingIncomeLoss",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2025-12-31",
                        "filing_date": "2026-02-10",
                        "accession": "0001668397-26-000006",
                        "location": "Item 8 Consolidated Statements of Operations",
                        "reported_value": 534935000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-23",
                        "accession": "0001668397-26-000025",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (6 Months Ended)",
                        "reported_value": 239858000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-23",
                        "accession": "0001668397-26-000025",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (6 Months Ended)",
                        "reported_value": 288327000.0
                    }
                ]
            },
            "cash_and_investments": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_cash_and_investments_usd": 502688000.0,
                "cash_and_equivalents_usd": 502688000.0,
                "short_term_investments_usd": 0.0,
                "xbrl_tags": [
                    "us-gaap:CashAndCashEquivalentsAtCarryingValue"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-23",
                    "accession": "0001668397-26-000025",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets",
                    "reported_value": 502688000.0
                }
            },
            "funded_debt": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_financial_debt_usd": 0.0,
                "is_zero_debt": True,
                "support_disclosure": "Form 10-Q Part I Item 1 Condensed Consolidated Balance Sheets and Note 6 disclose zero senior notes or funded borrowings outstanding; revolving credit facility ($500M) is fully undrawn as of June 30, 2026.",
                "xbrl_tag": "us-gaap:LongTermDebtNoncurrent",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-23",
                    "accession": "0001668397-26-000025",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets & Note 6 (Debt)",
                    "reported_value": 0.0
                }
            },
            "lease_treatment": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "current_operating_lease_liabilities_usd": 24012000.0,
                "noncurrent_operating_lease_liabilities_usd": 120100000.0,
                "total_lease_liabilities_usd": 144112000.0,
                "accounting_standard": "ASC 842",
                "valuation_convention": "Lease-as-Operating-Expense convention: Operating lease costs for clinical research facilities are recognized in operating costs; lease liabilities ($144.11M total) are excluded from the net debt bridge to avoid double deduction.",
                "xbrl_tags": [
                    "us-gaap:OperatingLeaseLiabilityCurrent",
                    "us-gaap:OperatingLeaseLiabilityNoncurrent"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-23",
                    "accession": "0001668397-26-000025",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets"
                }
            },
            "shares_outstanding": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "share_count": 28554000.0,
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "measurement_period": "Three months ended June 30, 2026",
                "selection_rationale": "Diluted weighted-average common shares outstanding (28,554,000) reported in Form 10-Q Condensed Consolidated Statements of Operations for the three months ended June 30, 2026. Selected over prior-period figures to capture ongoing share repurchases and reflect the current quarterly equity claim on Enterprise Value.",
                "xbrl_tag": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-23",
                    "accession": "0001668397-26-000025",
                    "location": "Part I Item 1 Condensed Consolidated Statements of Operations"
                }
            },
            "reference_price": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "reference_price": 618.47,
                "reference_date": "2026-10-16",
                "timestamp_utc": "2026-10-16T20:00:00Z",
                "source_system": "ARX MarketStore regular close tape",
                "source_store": "data/research/market_data/bounded_150_dev_market_store.db",
                "table": "asset_ohlcv_daily"
            },
            "valuation_model": {
                "fact_status": "DERIVED_CALCULATION",
                "model_type": "Single-stage Reverse-DCF FCFF (Gordon Growth Terminal Value)",
                "solved_5y_revenue_cagr_pct": 29.41,
                "assumptions": {
                    "fact_status": "ANALYST_ASSUMPTION",
                    "operating_margin_pct": 21.0,
                    "wacc_pct": 9.5,
                    "terminal_growth_pct": 2.5,
                    "tax_rate_pct": 21.0,
                    "reinvestment_rate_pct": 12.0
                },
                "roundtrip_value": 618.56,
                "roundtrip_delta_cents": 8.53
            },
            "narrative_claims": [
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "DERIVED_CALCULATION",
                    "statement": "Medpace delivers high operating profitability, positive operating cash generation, and zero funded debt: TTM operating margin is 20.97% ($583.4M operating income on $2,782.3M revenue), supported by $502.7M in cash and cash equivalents against zero funded financial debt.",
                    "locator": "Form 10-K Item 8 Consolidated Statements of Operations and Form 10-Q Part I Item 1 Condensed Consolidated Financial Statements (Balance Sheets & Statements of Operations)."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Medpace carries zero funded senior debt and $502.7M in cash and cash equivalents, with an undrawn $500M revolving credit facility.",
                    "locator": "Form 10-Q Part I Item 1 Condensed Consolidated Balance Sheets & Note 6 (Debt)."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Medpace full-service Phase I-IV CRO model delivers a 20.97% TTM operating margin ($583.4M operating income on $2,782.3M revenue).",
                    "locator": "Form 10-K Item 8 Consolidated Statements of Operations and Form 10-Q Condensed Consolidated Statements of Operations."
                }
            ],
            "selection_rationale": {
                "why_this_business": "Medpace operates a full-service Phase I-IV clinical contract research organization (CRO) focused on small-to-midsize biopharma sponsors. Its unified operational delivery model yields an operating margin of 20.97%, positive operating cash generation, and zero funded debt. It carries a pristine balance sheet with $502.7M in cash and an undrawn $500M credit facility.",
                "why_research_now": "Small-to-midsize biopharma funding stabilized following the 2023-2024 contraction, creating a divergence between steady RFP volumes and conservative equity market expectations. At $618.47, understanding whether the business can sustain >20% EBIT margins through biotech funding cycles justifies active research tracking.",
                "price_requirements": "At a reference price of $618.47, assuming a 21.0% operating margin, 9.5% WACC, 2.5% terminal growth, 21.0% tax rate, and 12.0% reinvestment rate, the single-stage reverse-DCF model implies a 5-year revenue CAGR of 29.41% ($618.56 per-share roundtrip value, 8.53 cents delta). This reflects aggressive growth assumptions rather than observed consensus.",
                "differentiated_view": "No differentiated expectation established yet. The candidate is maintained as a disciplined watch until either a valuation compression toward base scenario levels ($380 - $440) or sustained book-to-bill acceleration >1.25x clarifies upside skew.",
                "assessment_changes": "Strengthening: Net new business awards book-to-bill ratio sustaining >1.25x alongside cancellation rates below historical averages (<10%). Weakening/Invalidation: Net book-to-bill falling below 1.0x for two consecutive quarters or operating margin compressing below 16.0%.",
                "queue_priority": "Among healthcare candidates in the expansion queue, MEDP provides an exceptional capital-light profile with zero debt and superior cash conversion relative to capital-intensive facility operators. Tracking it establishes an authoritative benchmark for CRO capital efficiency."
            }
        },
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-10",
                "accession_number": "0001668397-26-000006",
                "doc_name": "medp-20251231.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-23",
                "accession_number": "0001668397-26-000025",
                "doc_name": "medp-20260630.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 2782288000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 583404000.0,
            "operating_margin_pct": 20.97,
            "cash_and_equivalents": 502688000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 502688000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded senior debt on consolidated balance sheet; revolving credit line fully undrawn.",
            "operating_lease_liabilities_asc842": 144112000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied; clinical facility leases expensed in operating costs; lease liabilities excluded from net debt bridge.",
            "net_debt_bridge": -502688000.0,
            "bridge_note": "Unrestricted cash ($502.7M) against zero funded debt yields $502.7M net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 28554000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 28554000.0,
                "measurement_period": "Three months ended June 30, 2026",
                "rationale": "Diluted weighted-average common shares outstanding (28,554,000) reported in Form 10-Q Condensed Consolidated Statements of Operations for the three months ended June 30, 2026. Selected over prior-period figures to capture ongoing share repurchases and reflect the current quarterly equity claim on Enterprise Value."
            }
        },
        "valuation_spec": {
            "margin": 21.0,
            "wacc": 9.5,
            "terminal_g": 2.5,
            "tax_rate": 21.0,
            "reinvestment_rate": 12.0,
            "scenarios": {
                "conservative": {"growth_pct": 10.0, "margin_pct": 18.0, "wacc_pct": 10.0, "g_pct": 2.0, "tax_pct": 22.0, "reinvest_pct": 15.0},
                "base": {"growth_pct": 16.0, "margin_pct": 21.0, "wacc_pct": 9.5, "g_pct": 2.5, "tax_pct": 21.0, "reinvest_pct": 12.0},
                "optimistic": {"growth_pct": 25.0, "margin_pct": 23.0, "wacc_pct": 9.0, "g_pct": 3.0, "tax_pct": 20.0, "reinvest_pct": 10.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Medpace delivers high operating profitability (20.97% TTM operating margin: $583.4M operating income on $2,782.3M revenue), positive operating cash generation, and zero funded debt, supported by an organic full-service clinical execution model and $502.7M in cash.",
            "strongest_opposing_evidence": "Biopharma venture funding cycles and small-cap clinical trial cancellation rates create volatility in net book-to-bill ratios. At $618.47, market embeds a demanding 29.41% revenue CAGR.",
            "main_uncertainty": "Sustained clinical trial contract backlog conversion amidst fluctuating early-stage biotech liquidity.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($380 - $440) or net new business awards sustaining book-to-bill > 1.25x.",
            "confidence_assessment": "High confidence in operational execution and balance sheet conservatism; moderate valuation headroom.",
            "financing_dilution_exposure": "Share repurchases offset stock compensation, reducing share count by ~9.6% over 3 years.",
            "observable_invalidation_trigger": "Net new business awards book-to-bill ratio declining below 1.0x for consecutive quarters or operating margin breach below 16.0%.",
            "next_monitoring_trigger": "Q3 2026 earnings release and Form 10-Q for trial backlog cancellations and RFP volume.",
            "presentation_summary": "Medpace Holdings delivers high operating profitability (TTM operating margin 20.97%) and zero-debt balance sheet strength ($502.7M cash). At $618.47, the reverse-DCF model implies an aggressive 29.41% 5-year revenue CAGR under base assumptions. Retained on watch pending market pullbacks toward mid-cycle base valuation range ($380 - $440)."
        }
    },
    "DECK": {
        "company_name": "Deckers Outdoor Corp.",
        "cik": "0000910521",
        "exchange": "NYSE",
        "opportunity_route": "Consumer Cyclical / Footwear",
        "sector_industry": "Consumer Cyclical / Footwear & Accessories",
        "business_model": "Premium global footwear and apparel design and wholesale/DTC distribution (HOKA, UGG)",
        "reference_price": 85.72,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "review_notes": "Scaling addition candidate prepared with complete primary filing source-to-input tracing, verified brand/channel growth disclosures, 6:1 split-adjusted capital structure, and structured selection record.",
        "source_evidence_tracing": {
            "representative_status": "EXPANSION_PILOT_CANDIDATE",
            "company_identity": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "entity_name": "Deckers Outdoor Corp.",
                "cik": "0000910521",
                "sic": "3021",
                "exchange": "NYSE",
                "source_url": "https://data.sec.gov/submissions/CIK0000910521.json",
                "filing_accessions": [
                    "0001628280-26-037664",
                    "0000910521-26-000022"
                ]
            },
            "revenue": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 5527289000.0,
                "derivation_formula": "FY2026 Revenues ($5,472,296,000) - Q1 FY2026 Revenues ($964,538,000) + Q1 FY2027 Revenues ($1,019,531,000)",
                "xbrl_tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2026-03-31",
                        "filing_date": "2026-05-22",
                        "accession": "0001628280-26-037664",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 5472296000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-30",
                        "accession": "0000910521-26-000022",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (3 Months Ended)",
                        "reported_value": 964538000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-30",
                        "accession": "0000910521-26-000022",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (3 Months Ended)",
                        "reported_value": 1019531000.0
                    }
                ]
            },
            "operating_income": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 1252917000.0,
                "operating_margin_pct": 22.67,
                "derivation_formula": "FY2026 Operating Income ($1,262,903,000) - Q1 FY2026 Operating Income ($165,287,000) + Q1 FY2027 Operating Income ($155,301,000)",
                "xbrl_tag": "us-gaap:OperatingIncomeLoss",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2026-03-31",
                        "filing_date": "2026-05-22",
                        "accession": "0001628280-26-037664",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 1262903000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-30",
                        "accession": "0000910521-26-000022",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (3 Months Ended)",
                        "reported_value": 165287000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-30",
                        "accession": "0000910521-26-000022",
                        "location": "Part I Item 1 Condensed Consolidated Statements of Operations (3 Months Ended)",
                        "reported_value": 155301000.0
                    }
                ]
            },
            "cash_and_investments": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_cash_and_investments_usd": 1602589000.0,
                "cash_and_equivalents_usd": 1602589000.0,
                "short_term_investments_usd": 0.0,
                "xbrl_tags": [
                    "us-gaap:CashAndCashEquivalentsAtCarryingValue"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-30",
                    "accession": "0000910521-26-000022",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets",
                    "reported_value": 1602589000.0
                }
            },
            "funded_debt": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_financial_debt_usd": 0.0,
                "is_zero_debt": True,
                "support_disclosure": "Form 10-Q Part I Item 1 Condensed Consolidated Balance Sheets and Note 6 disclose zero senior notes or funded credit facility borrowings outstanding as of June 30, 2026; revolving credit facility remains undrawn.",
                "xbrl_tag": "us-gaap:LongTermDebtNoncurrent",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-30",
                    "accession": "0000910521-26-000022",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets & Note 6 (Revolving Credit Facility)",
                    "reported_value": 0.0
                }
            },
            "lease_treatment": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "current_operating_lease_liabilities_usd": 73358000.0,
                "noncurrent_operating_lease_liabilities_usd": 398976000.0,
                "total_lease_liabilities_usd": 472334000.0,
                "accounting_standard": "ASC 842",
                "valuation_convention": "Lease-as-Operating-Expense convention: Store and distribution facility operating leases are recognized in Cost of Sales and SG&A under ASC 842; operating lease liabilities ($472.33M) are excluded from the net debt bridge to avoid double deduction.",
                "xbrl_tags": [
                    "us-gaap:OperatingLeaseLiabilityCurrent",
                    "us-gaap:OperatingLeaseLiabilityNoncurrent"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-30",
                    "accession": "0000910521-26-000022",
                    "location": "Part I Item 1 Condensed Consolidated Balance Sheets"
                }
            },
            "shares_outstanding": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "share_count": 138559000.0,
                "basis_type": "SPLIT_ADJUSTED_DILUTED",
                "measurement_period": "Three months ended June 30, 2026",
                "selection_rationale": "Diluted weighted-average common shares outstanding (138,559 thousand) reported in Form 10-Q Condensed Consolidated Statements of Operations for the three months ended June 30, 2026, reflecting the 6-for-1 forward split executed in September 2024. Selected over prior-period figures to capture the post-split equity claim on Enterprise Value.",
                "xbrl_tag": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-30",
                    "accession": "0000910521-26-000022",
                    "location": "Part I Item 1 Condensed Consolidated Statements of Operations"
                }
            },
            "reference_price": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "reference_price": 85.72,
                "reference_date": "2026-10-16",
                "timestamp_utc": "2026-10-16T20:00:00Z",
                "source_system": "ARX MarketStore regular close tape",
                "source_store": "data/research/market_data/bounded_150_dev_market_store.db",
                "table": "asset_ohlcv_daily"
            },
            "valuation_model": {
                "fact_status": "DERIVED_CALCULATION",
                "model_type": "Single-stage Reverse-DCF FCFF (Gordon Growth Terminal Value)",
                "solved_5y_revenue_cagr_pct": -2.34,
                "assumptions": {
                    "fact_status": "ANALYST_ASSUMPTION",
                    "operating_margin_pct": 22.0,
                    "wacc_pct": 9.0,
                    "terminal_growth_pct": 2.5,
                    "tax_rate_pct": 22.0,
                    "reinvestment_rate_pct": 15.0
                },
                "roundtrip_value": 85.72,
                "roundtrip_delta_cents": 0.05
            },
            "narrative_claims": [
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "In Q1 FY27 (three months ended June 30, 2026), HOKA brand net sales grew 7.7% YoY to $703.5M ($703,538 thousand), with direct-to-consumer (DTC) net sales growing 17.3% YoY to $256.8M ($256,775 thousand) and wholesale net sales growing 2.9% YoY to $446.8M ($446,763 thousand); consolidated net sales reached $1,019.5M ($1,019,531 thousand, +5.7% YoY).",
                    "locator": "Form 10-Q Part I Item 1 Note 7 (Segment Information and Net Sales by Brand and Channel) and Item 2 MD&A (Results of Operations)."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Deckers Outdoor operates with zero funded senior debt and $1.60B in cash and cash equivalents, with an undrawn credit facility.",
                    "locator": "Form 10-Q Part I Item 1 Condensed Consolidated Balance Sheets and Note 6 (Revolving Credit Facility)."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Forward 6-for-1 stock split was executed on September 13, 2024, with all historical share counts, equity incentive plans, and per-share amounts retroactively adjusted.",
                    "locator": "Form 10-K Item 8 Note 1 and Form 10-Q Part I Item 1 Note 8 (Stockholders' Equity)."
                }
            ],
            "selection_rationale": {
                "why_this_business": "Deckers Outdoor designs and distributes premium footwear and apparel, anchored by the high-growth HOKA athletic brand and the seasonally dominant UGG brand. The company operates with zero funded debt, $1.60B in cash reserves, and a 22.67% TTM operating margin under a clean 6:1 split-adjusted capital structure.",
                "why_research_now": "HOKA's expansion from specialist running into mainstream wholesale and DTC channels is reaching an inflection point where wholesale growth has moderated (2.9% YoY in Q1 FY27) while DTC maintains double-digit momentum (17.3% YoY). The valuation embeds low expectations, warranting active tracking of brand durability.",
                "price_requirements": "At a reference price of $85.72, assuming a 22.0% operating margin, 9.0% WACC, 2.5% terminal growth, 22.0% tax rate, and 15.0% reinvestment rate, the reverse-DCF implies a 5-year revenue CAGR of -2.34% ($85.72 per-share roundtrip value). This negative solved growth rate is conditional on the assumed 22% margin and 9% WACC; it does not indicate that the market expects actual contraction or guarantees undervaluation.",
                "differentiated_view": "DTC channel strength (+17.3% YoY) indicates enduring brand affinity that may outlast wholesale channel inventory adjustments, potentially supporting consolidated revenue expansion above the negative reverse-DCF solved baseline.",
                "assessment_changes": "Strengthening: HOKA international DTC net sales growth exceeding 20% YoY alongside consolidated gross margins sustaining above 55.0%. Weakening/Invalidation: Consolidated gross margin breach below 50.0% or HOKA DTC quarterly net sales growth falling below 10.0%.",
                "queue_priority": "Within the consumer discretionary cohort, DECK provides a debt-free balance sheet with $1.60B net cash and substantial operating margins, presenting a lower financial risk profile than leveraged retail peers while offering high brand velocity."
            }
        },
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-03-31",
                "filing_date": "2026-05-22",
                "accession_number": "0001628280-26-037664",
                "doc_name": "deck-20260331.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-30",
                "accession_number": "0000910521-26-000022",
                "doc_name": "deck-20260630.htm",
                "source_role": "Quarterly financial report (Q1 FY27)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 5527289000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 1252917000.0,
            "operating_margin_pct": 22.67,
            "cash_and_equivalents": 1602589000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 1602589000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; credit facility undrawn; audited post 6:1 forward stock split (September 2024).",
            "operating_lease_liabilities_asc842": 472334000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied; store and distribution leases expensed in operating income; lease liabilities ($472.33M) excluded from net debt bridge.",
            "net_debt_bridge": -1602589000.0,
            "bridge_note": "Total cash of $1.60B against zero funded debt yields $1.60B net cash bridge added to enterprise value in DCF equity bridge.",
            "diluted_shares": 138559000.0,
            "selected_share_basis": {
                "basis_type": "SPLIT_ADJUSTED_DILUTED",
                "value": 138559000.0,
                "measurement_period": "Three months ended June 30, 2026",
                "rationale": "Diluted weighted-average common shares outstanding (138,559 thousand) reported in Form 10-Q Condensed Consolidated Statements of Operations for the three months ended June 30, 2026, reflecting the 6-for-1 forward split executed in September 2024. Selected over prior-period figures to capture the post-split equity claim on Enterprise Value."
            }
        },
        "valuation_spec": {
            "margin": 22.0,
            "wacc": 9.0,
            "terminal_g": 2.5,
            "tax_rate": 22.0,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 2.0, "margin_pct": 18.0, "wacc_pct": 9.5, "g_pct": 2.0, "tax_pct": 23.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 8.0, "margin_pct": 22.0, "wacc_pct": 9.0, "g_pct": 2.5, "tax_pct": 22.0, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 14.0, "margin_pct": 24.0, "wacc_pct": 8.5, "g_pct": 3.0, "tax_pct": 21.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "In Q1 FY27 (ended June 30, 2026), HOKA direct-to-consumer (DTC) net sales grew 17.3% YoY to $256.8M and wholesale net sales grew 2.9% YoY to $446.8M (total HOKA net sales +7.7% YoY to $703.5M; consolidated net sales +5.7% YoY to $1,019.5M), demonstrating brand pull in direct channels while wholesale growth normalizes. Supported by $1.60B in cash and zero funded debt.",
            "strongest_opposing_evidence": "High seasonal concentration in UGG winter demand and wholesale channel growth moderation (HOKA wholesale up 2.9% in Q1 FY27 vs 27.9% in FY26). Under the stated valuation assumptions (22% margin, 9% WACC, 2.5% terminal g), the solved 5Y CAGR is -2.34%, reflecting conditional model mechanics rather than a market prediction of contraction or definitive undervaluation.",
            "main_uncertainty": "Sustainability of premium DTC full-price sell-through rates as HOKA expands into broader lifestyle footwear categories.",
            "price_watch_condition": "Confirmation of international DTC growth above 20% while maintaining consolidated gross margin above 55.0%.",
            "confidence_assessment": "High confidence in brand equity and net cash asset backing; moderate concern regarding cyclical consumer discretionary headwinds.",
            "financing_dilution_exposure": "Clean share capital structure; forward 6:1 split successfully resolved historical comparability issues with zero phantom dilution.",
            "observable_invalidation_trigger": "Gross margin decline below 50.0% or HOKA DTC quarterly revenue deceleration below 10.0%.",
            "next_monitoring_trigger": "Q2 FY27 earnings release (expected October/November 2026) for holiday pre-orders and inventory aging.",
            "presentation_summary": "Deckers Outdoor displays strong balance sheet health ($1.60B net cash, zero funded debt). In Q1 FY27, HOKA net sales reached $703.5M (+7.7% YoY), led by DTC (+17.3% to $256.8M) while wholesale moderated (+2.9% to $446.8M); consolidated net sales grew 5.7% to $1,019.5M. Under conditional model assumptions (22% margin, 9% WACC), the reference price of $85.72 yields an implied 5-year revenue CAGR of -2.34%. Retained on watch for international DTC expansion and gross margin defense."
        }
    },
    "KLAC": {
        "company_name": "KLA Corporation",
        "cik": "0000319201",
        "exchange": "NASDAQ",
        "opportunity_route": "Technology / Semiconductor Equipment",
        "sector_industry": "Technology / Semiconductor Equipment",
        "business_model": "Process control, metrology, and yield management systems for semiconductor wafer fabrication",
        "reference_price": 165.74,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_QUALIFIED",
        "valuation_status_label": "Computed with Qualification",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-06-30",
                "filing_date": "2026-08-08",
                "accession_number": "0000319201-26-000035",
                "doc_name": "klac-20260630x10k.htm",
                "source_role": "Annual audited financial disclosures"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 13579476000.0,
            "revenue_period": "FY2026 (Ended 2026-06-30)",
            "operating_income": 5660780000.0,
            "operating_margin_pct": 41.69,
            "cash_and_equivalents": 1649842000.0,
            "short_term_investments": 1468388000.0,
            "total_cash_and_investments": 3118230000.0,
            "total_financial_debt": 5887415000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 264298000.0,
            "lease_convention_treatment": "Operating lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 2769185000.0,
            "bridge_note": "Total financial debt of $5.89B less total cash and marketable investments of $3.12B yields $2.77B net debt bridge subtracted from enterprise value.",
            "diluted_shares": 131963300.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 131963300.0,
                "measurement_period": "FY2026 ended 2026-06-30",
                "rationale": "Diluted weighted average shares reported in 10-K."
            }
        },
        "valuation_spec": {
            "margin": 40.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 14.5,
            "reinvestment_rate": 20.0,
            "qualification_notice": "Qualified by semiconductor capital equipment cyclicality; current ~41.7% operating margins reflect peak advanced-packaging wafer fab equipment (WFE) demand.",
            "scenarios": {
                "conservative": {"growth_pct": -5.0, "margin_pct": 34.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 15.0, "reinvest_pct": 22.0},
                "base": {"growth_pct": 5.0, "margin_pct": 40.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 14.5, "reinvest_pct": 20.0},
                "optimistic": {"growth_pct": 12.0, "margin_pct": 43.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 14.0, "reinvest_pct": 18.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Dominant global monopoly (>55% market share) in optical and e-beam defect inspection and metrology, an indispensable tollgate for GAA and high-bandwidth memory (HBM) yields.",
            "strongest_opposing_evidence": "Geopolitical export controls restricting leading-edge lithography/metrology tool shipments to mainland China, combined with severe semiconductor cycle peaks/troughs.",
            "main_uncertainty": "Impact of tightened US export controls on Chinese trailing-edge mature node spending, which contributed ~35% of FY2026 revenue.",
            "price_watch_condition": "Cyclical pullback toward $130 - $140 or verification that leading-edge 2nm GAA node ramp offsets mature China WFE contraction.",
            "confidence_assessment": "High confidence in technical moat and operating leverage; valuation qualified due to cyclical macro semiconductor exposure.",
            "financing_dilution_exposure": "Active capital return through dividends and repurchases maintains stable share count with minimal net dilution.",
            "observable_invalidation_trigger": "Metrology market share loss exceeding 300 bps to competitors (Applied Materials / ASML) or China revenue decline >50% without leading-edge offset.",
            "next_monitoring_trigger": "Q1 FY27 Form 10-Q filing (October 2026) for WFE orders and advanced packaging adoption rates.",
            "presentation_summary": "KLA Corp holds an unassailable technical monopoly in semiconductor metrology (>55% share) with 41.7% operating margins. Valuation is qualified by semiconductor equipment cyclicality and China export exposure. At $165.74, market prices in cyclical trough normalization (-15.99% CAGR). Retained on watch."
        }
    },
    "LULU": {
        "company_name": "lululemon athletica inc.",
        "cik": "0001397187",
        "exchange": "NASDAQ",
        "opportunity_route": "Consumer Cyclical / Apparel Retail",
        "sector_industry": "Consumer Cyclical / Apparel Retail",
        "business_model": "Technical athletic apparel, footwear, and accessories direct-to-consumer and retail marketing",
        "reference_price": 97.10,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-01-28",
                "filing_date": "2026-03-27",
                "accession_number": "0001397187-26-000018",
                "doc_name": "lulu-20260128x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-08-02",
                "filing_date": "2026-09-05",
                "accession_number": "0001397187-26-000052",
                "doc_name": "lulu-20260802x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 11093955000.0,
            "revenue_period": "TTM (Ended 2026-08-02)",
            "operating_income": 1978775000.0,
            "operating_margin_pct": 17.84,
            "cash_and_equivalents": 1389737000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 1389737000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded short-term or long-term debt; reported zero short-term borrowings under credit agreements.",
            "operating_lease_liabilities_asc842": 1412000000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied under ASC 842; store operating lease payments expensed in SG&A; lease obligations excluded from net debt bridge.",
            "net_debt_bridge": -1389737000.0,
            "bridge_note": "Total cash ($1.39B) less zero funded debt leaves $1.39B net cash bridge added to enterprise value in DCF equity bridge.",
            "diluted_shares": 112919000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 112919000.0,
                "measurement_period": "TTM ended 2026-08-02",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 18.0,
            "wacc": 9.5,
            "terminal_g": 2.5,
            "tax_rate": 24.0,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": -2.0, "margin_pct": 15.0, "wacc_pct": 10.0, "g_pct": 2.0, "tax_pct": 25.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 5.0, "margin_pct": 18.0, "wacc_pct": 9.5, "g_pct": 2.5, "tax_pct": 24.0, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 11.0, "margin_pct": 20.0, "wacc_pct": 9.0, "g_pct": 3.0, "tax_pct": 23.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Robust international momentum (>25% constant currency expansion in China and EMEA) and a fortress balance sheet ($1.39B net cash, zero funded debt).",
            "strongest_opposing_evidence": "North American consumer fatigue, execution missteps in product launches (Breezethrough leggings pause), and rising competitive pressure from Alo Yoga and Vuori.",
            "main_uncertainty": "Stabilization of North American women's comparable store sales and recovery of product innovation velocity.",
            "price_watch_condition": "Evidence of North American comparable sales inflection back above +3% or market pullback toward $80 - $85 base valuation.",
            "confidence_assessment": "High confidence in balance sheet resilience; moderate confidence in domestic apparel competitive moat recovery.",
            "financing_dilution_exposure": "Disciplined share repurchases reduce diluted share count by ~11.3% over trailing three years.",
            "observable_invalidation_trigger": "Consolidated operating margin breach below 14.0% or negative international comparable sales growth.",
            "next_monitoring_trigger": "Q3 2026 earnings call (December 2026) for North America product assortment reset and holiday order books.",
            "presentation_summary": "Lululemon Athletica pairs global brand reach and $1.39B net cash with strong international growth. However, domestic North American stagnation and fashion competition have depressed investor sentiment. At $97.10, market prices in -12.59% growth. Retained on watch for North American sales stabilization."
        }
    },
    "EME": {
        "company_name": "EMCOR Group, Inc.",
        "cik": "0000105634",
        "exchange": "NYSE",
        "opportunity_route": "Specialty Contractors & Critical Infrastructure",
        "sector_industry": "Industrials / Specialty Contractors",
        "business_model": "Electrical, mechanical construction, and facilities services supporting hyperscale data center buildouts",
        "reference_price": 721.66,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-26",
                "accession_number": "0000105634-26-000015",
                "doc_name": "eme-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-30",
                "accession_number": "0000105634-26-000042",
                "doc_name": "eme-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 18597775000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 1930635000.0,
            "operating_margin_pct": 10.38,
            "cash_and_equivalents": 924411000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 924411000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; senior credit facility undrawn; clean capital structure directly comparable to FIX.",
            "operating_lease_liabilities_asc842": 312000000.0,
            "lease_convention_treatment": "Operating lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": -924411000.0,
            "bridge_note": "Total cash of $924.4M against zero funded debt yields $924.4M net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 44552894.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 44552894.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 10.0,
            "wacc": 9.5,
            "terminal_g": 2.5,
            "tax_rate": 23.0,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 6.0, "margin_pct": 8.0, "wacc_pct": 10.0, "g_pct": 2.0, "tax_pct": 24.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 12.0, "margin_pct": 10.0, "wacc_pct": 9.5, "g_pct": 2.5, "tax_pct": 23.0, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 20.0, "margin_pct": 11.5, "wacc_pct": 9.0, "g_pct": 3.0, "tax_pct": 22.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Direct peer to Comfort Systems USA (FIX); record RPO backlog ($9B+) driven by hyperscale AI data center electrical power and mechanical cooling installations.",
            "strongest_opposing_evidence": "Fixed-price contract cost overrun risks and tight skilled labor availability. Market price of $721.66 embeds 16.15% 5Y revenue CAGR at peak 10.0% operating margin.",
            "main_uncertainty": "Labor availability and project execution margins as multi-year mega-project deliveries peak in 2027.",
            "price_watch_condition": "Market pullback toward base valuation range ($560 - $600) or evidence of non-residential project margin expansion above 11.0%.",
            "confidence_assessment": "High confidence in data center secular tailwinds and clean zero-debt balance sheet; valuation fully reflects ongoing cycle expansion.",
            "financing_dilution_exposure": "Stable share count with regular share repurchases offsetting incentive equity grants.",
            "observable_invalidation_trigger": "Backlog sequential decline >10% or material fixed-price contract loss provisioning exceeding $100M.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (October 2026) for institutional backlog updates and mechanical margin trends.",
            "presentation_summary": "EMCOR Group presents premier electrical/mechanical execution capacity for hyperscale data center construction, backed by $924.4M net cash. Reference market price of $721.66 reflects demanding 16.15% priced-in growth at peak 10.0% margins. Retained on watch for pullbacks toward $560 - $600."
        }
    },
    "VRT": {
        "company_name": "Vertiv Holdings Co",
        "cik": "0001674101",
        "exchange": "NYSE",
        "opportunity_route": "Specialty Contractors & Critical Infrastructure",
        "sector_industry": "Industrials / Electrical Equipment",
        "business_model": "Critical digital infrastructure technologies: liquid cooling, power distribution, and thermal management for AI data centers",
        "reference_price": 206.42,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-27",
                "accession_number": "0001674101-26-000018",
                "doc_name": "vrt-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-28",
                "accession_number": "0001674101-26-000052",
                "doc_name": "vrt-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 11479600000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 2174600000.0,
            "operating_margin_pct": 18.94,
            "cash_and_equivalents": 2810600000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 2810600000.0,
            "total_financial_debt": 5879600000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 185000000.0,
            "lease_convention_treatment": "Operating leases excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 3069000000.0,
            "bridge_note": "Carrying debt ($5.88B) less unrestricted cash ($2.81B) yields $3.07B net debt bridge deducted from enterprise value.",
            "diluted_shares": 392746991.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 392746991.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 19.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 22.0,
            "reinvestment_rate": 18.0,
            "scenarios": {
                "conservative": {"growth_pct": 12.0, "margin_pct": 16.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 23.0, "reinvest_pct": 20.0},
                "base": {"growth_pct": 22.0, "margin_pct": 19.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 22.0, "reinvest_pct": 18.0},
                "optimistic": {"growth_pct": 35.0, "margin_pct": 21.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 21.0, "reinvest_pct": 15.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Vertiv is the prime global supplier of direct-to-chip liquid cooling systems and high-density power architecture for Nvidia Blackwell and Rubin hyperscale clusters.",
            "strongest_opposing_evidence": "Supply chain component bottlenecks, high debt leverage ($5.88B debt), and aggressive valuation pricing in 37.70% 5Y revenue CAGR at $206.42.",
            "main_uncertainty": "Adoption speed of liquid-to-air vs liquid-to-liquid heat exchange architectures across second-tier colocation providers.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($125 - $140) or verification of liquid cooling order backlog doubling.",
            "confidence_assessment": "High confidence in technical leadership and hyperscale order visibility; valuation reflects aggressive operational perfection.",
            "financing_dilution_exposure": "Modest warrant dilution post-spac; net leverage declining rapidly via strong free cash flow generation.",
            "observable_invalidation_trigger": "Quarterly organic order intake decline >15% or liquid cooling gross margin deterioration below 30.0%.",
            "next_monitoring_trigger": "Q3 2026 earnings release (October 2026) for direct-to-chip deployment metrics and pipeline backlogs.",
            "presentation_summary": "Vertiv Holdings is the premier beneficiary of AI data center thermal engineering and liquid cooling transitions. However, at $206.42, market expects 37.70% 5-year revenue CAGR holding 19.0% margins with $3.07B net debt. Retained on watch pending valuation pullback toward $125 - $140."
        }
    },
    "ANET": {
        "company_name": "Arista Networks, Inc.",
        "cik": "0001596532",
        "exchange": "NYSE",
        "opportunity_route": "Cloud, Networking & Workflow Software",
        "sector_industry": "Technology / Computer Hardware",
        "business_model": "Cloud and AI data center networking platforms: high-performance Ethernet switching systems and EOS software",
        "reference_price": 176.99,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-14",
                "accession_number": "0001596532-26-000012",
                "doc_name": "anet-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-08-04",
                "accession_number": "0001596532-26-000045",
                "doc_name": "anet-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 10540800000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 4546900000.0,
            "operating_margin_pct": 43.14,
            "cash_and_equivalents": 2290200000.0,
            "short_term_investments": 3694100000.0,
            "total_cash_and_investments": 5984300000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; zero bank borrowings; $5.98B total cash and short-term liquid investments.",
            "operating_lease_liabilities_asc842": 82000000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied; leases excluded from net debt bridge.",
            "net_debt_bridge": -5984300000.0,
            "bridge_note": "Unrestricted cash and marketable securities ($5.98B) against zero funded debt yields $5.98B net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 1276000000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 1276000000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 42.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 17.5,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 14.0, "margin_pct": 38.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 18.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 24.0, "margin_pct": 42.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 17.5, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 38.0, "margin_pct": 44.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 17.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Arista maintains leadership in front-end and AI back-end Ethernet switching fabrics (Ethernet for AI clusters), supported by $5.98B net cash and 43.1% operating margins.",
            "strongest_opposing_evidence": "High customer concentration (Microsoft and Meta account for >35% of total sales) and InfiniBand competition from Nvidia. Reference price embeds 42.45% 5Y CAGR.",
            "main_uncertainty": "Rate of hyperscale Ethernet fabric displacement of InfiniBand in massive multi-rack AI GPU training clusters.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($115 - $125) or customer diversification expanding non-cloud titan revenue.",
            "confidence_assessment": "High confidence in software (EOS) moat and balance sheet fortress; valuation reflects aggressive AI Ethernet transition expectations.",
            "financing_dilution_exposure": "Stable share count; equity compensation fully cushioned by $5.98B net cash and strong operating cash flows.",
            "observable_invalidation_trigger": "Market share loss in 800G/1.6T switches exceeding 200 bps to Cisco or proprietary ASIC internal fabrics.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (November 2026) for 800G optical switch volume ramps and cloud capex guidance.",
            "presentation_summary": "Arista Networks leads cloud and AI data center Ethernet switching with exceptional capital efficiency (43.1% EBIT margin, $5.98B cash, zero debt). However, market pricing of $176.99 embeds an aggressive 42.45% 5Y CAGR. Retained on watch for pullbacks toward $115 - $125."
        }
    },
    "PLTR": {
        "company_name": "Palantir Technologies Inc.",
        "cik": "0001321655",
        "exchange": "NASDAQ",
        "opportunity_route": "Cloud, Networking & Workflow Software",
        "sector_industry": "Technology / Software Infrastructure",
        "business_model": "Enterprise artificial intelligence platform (AIP), defense analytics, and ontology operating systems",
        "reference_price": 186.03,
        "disposition": "NO_SUPPORTED_OPPORTUNITY",
        "disposition_label": "Rejected — No Supported Opportunity",
        "valuation_status": "COMPUTED_QUALIFIED",
        "valuation_status_label": "Computed with Qualification",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-20",
                "accession_number": "0001321655-26-000015",
                "doc_name": "pltr-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-08-06",
                "accession_number": "0001321655-26-000054",
                "doc_name": "pltr-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 6155941000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 2634652000.0,
            "operating_margin_pct": 42.80,
            "cash_and_equivalents": 2030047000.0,
            "short_term_investments": 7379052000.0,
            "total_cash_and_investments": 9409099000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; zero debt obligations; $9.41B total cash, equivalents and short-term US Treasury bills.",
            "operating_lease_liabilities_asc842": 195000000.0,
            "lease_convention_treatment": "Operating lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": -9409099000.0,
            "bridge_note": "Cash, equivalents, and short-term US Treasuries ($9.41B) against zero funded debt yields $9.41B net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 2568694000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 2568694000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 35.0,
            "wacc": 10.0,
            "terminal_g": 3.5,
            "tax_rate": 21.0,
            "reinvestment_rate": 20.0,
            "qualification_notice": "Priced-in growth expectations (98.34% 5Y CAGR) reflect extreme market euphoria and speculative multiple expansion far beyond fundamental cash conversion boundaries.",
            "scenarios": {
                "conservative": {"growth_pct": 20.0, "margin_pct": 30.0, "wacc_pct": 10.5, "g_pct": 3.0, "tax_pct": 22.0, "reinvest_pct": 25.0},
                "base": {"growth_pct": 35.0, "margin_pct": 35.0, "wacc_pct": 10.0, "g_pct": 3.5, "tax_pct": 21.0, "reinvest_pct": 20.0},
                "optimistic": {"growth_pct": 55.0, "margin_pct": 42.0, "wacc_pct": 9.5, "g_pct": 4.0, "tax_pct": 20.0, "reinvest_pct": 18.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Commercial AIP bootcamps accelerating US enterprise adoption (>80% YoY commercial revenue growth) and unmatched defense intelligence integration, backed by $9.41B net cash.",
            "strongest_opposing_evidence": "Extreme valuation multiple (>75x forward revenue, >180x forward EBITDA). Solved reverse DCF requires an impossible 98.34% sustained 5-year revenue CAGR.",
            "main_uncertainty": "Customer contract conversion from AIP pilots to multi-million dollar annual recurring production contracts.",
            "price_watch_condition": "Severe valuation compression below $50 per share before considering fundamentally supported entry.",
            "confidence_assessment": "High confidence in government defense franchise; zero confidence in equity risk/reward profile at current reference price.",
            "financing_dilution_exposure": "Historical high SBC dilution has moderated post-S&P 500 inclusion, but remaining share overhang requires disciplined GAAP audit.",
            "observable_invalidation_trigger": "Deceleration of US commercial revenue growth below 30.0% or government contract renewal friction.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (November 2026) for US commercial customer count and net retention rate.",
            "presentation_summary": "Palantir Technologies demonstrates unmatched enterprise AIP adoption and defense entrenchment, backed by $9.41B cash. However, at $186.03, the stock embeds an extreme 98.34% 5-year CAGR requirement. Rejected as a supported opportunity (NO_SUPPORTED_OPPORTUNITY) due to prohibitive valuation risk."
        }
    },
    "ISRG": {
        "company_name": "Intuitive Surgical, Inc.",
        "cik": "0001035267",
        "exchange": "NASDAQ",
        "opportunity_route": "Healthcare / Medical Research",
        "sector_industry": "Healthcare / Medical Instruments",
        "business_model": "Robotic-assisted minimally invasive surgical systems (da Vinci 5, Ion) and recurring instruments/accessories",
        "reference_price": 426.70,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "review_notes": "Scaling addition candidate prepared with complete primary filing source-to-input tracing, verified recurring revenue disclosures (instruments, services, and operating leases totaling 85%), and structured selection record.",
        "source_evidence_tracing": {
            "representative_status": "EXPANSION_PILOT_CANDIDATE",
            "company_identity": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "entity_name": "Intuitive Surgical, Inc.",
                "cik": "0001035267",
                "sic": "3842",
                "exchange": "NASDAQ",
                "source_url": "https://data.sec.gov/submissions/CIK0001035267.json",
                "filing_accessions": [
                    "0001035267-26-000010",
                    "0001035267-26-000058"
                ]
            },
            "revenue": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 11034400000.0,
                "derivation_formula": "FY2025 Revenues ($10,064,700,000) - Q2 2025 6M Revenues ($4,693,400,000) + Q2 2026 6M Revenues ($5,663,100,000)",
                "xbrl_tag": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2025-12-31",
                        "filing_date": "2026-02-03",
                        "accession": "0001035267-26-000010",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 10064700000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-21",
                        "accession": "0001035267-26-000058",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 4693400000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-21",
                        "accession": "0001035267-26-000058",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 5663100000.0
                    }
                ]
            },
            "operating_income": {
                "fact_status": "DERIVED_CALCULATION",
                "reported_ttm_value_usd": 3451200000.0,
                "operating_margin_pct": 31.28,
                "derivation_formula": "FY2025 Operating Income ($2,945,500,000) - Q2 2025 6M Operating Income ($1,321,500,000) + Q2 2026 6M Operating Income ($1,827,200,000)",
                "xbrl_tag": "us-gaap:OperatingIncomeLoss",
                "unit": "USD",
                "filing_sources": [
                    {
                        "form": "10-K",
                        "period_end": "2025-12-31",
                        "filing_date": "2026-02-03",
                        "accession": "0001035267-26-000010",
                        "location": "Item 8 Consolidated Statements of Income",
                        "reported_value": 2945500000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2025-06-30",
                        "filing_date": "2026-07-21",
                        "accession": "0001035267-26-000058",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 1321500000.0
                    },
                    {
                        "form": "10-Q",
                        "period_end": "2026-06-30",
                        "filing_date": "2026-07-21",
                        "accession": "0001035267-26-000058",
                        "location": "Part I Item 1 Consolidated Statements of Income (6 Months Ended)",
                        "reported_value": 1827200000.0
                    }
                ]
            },
            "cash_and_investments": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_cash_and_investments_usd": 5216200000.0,
                "cash_and_equivalents_usd": 2760400000.0,
                "short_term_investments_usd": 2455800000.0,
                "xbrl_tags": [
                    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
                    "us-gaap:MarketableSecuritiesCurrent"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-21",
                    "accession": "0001035267-26-000058",
                    "location": "Part I Item 1 Consolidated Balance Sheets",
                    "reported_value": 5216200000.0
                }
            },
            "funded_debt": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "total_financial_debt_usd": 0.0,
                "is_zero_debt": True,
                "support_disclosure": "Form 10-Q Part I Item 1 Consolidated Balance Sheets disclose zero funded senior notes or credit borrowings; total cash and investments of $5.22B against zero debt obligations.",
                "xbrl_tag": "us-gaap:LongTermDebtNoncurrent",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-21",
                    "accession": "0001035267-26-000058",
                    "location": "Part I Item 1 Consolidated Balance Sheets",
                    "reported_value": 0.0
                }
            },
            "lease_treatment": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "current_operating_lease_liabilities_usd": 39000000.0,
                "noncurrent_operating_lease_liabilities_usd": 131900000.0,
                "total_lease_liabilities_usd": 170900000.0,
                "accounting_standard": "ASC 842",
                "valuation_convention": "Lease-as-Operating-Expense convention: Operating leases for facilities and equipment are recognized in operating expenses; lease liabilities ($170.9M total reported in Form 10-K Note on Leases; bundled in Other accrued and Other long-term liabilities on interim balance sheet) are excluded from the net debt bridge to prevent double deduction.",
                "xbrl_tags": [
                    "us-gaap:OperatingLeaseLiabilityCurrent",
                    "us-gaap:OperatingLeaseLiabilityNoncurrent"
                ],
                "filing_source": {
                    "form": "10-K",
                    "period_end": "2025-12-31",
                    "filing_date": "2026-02-03",
                    "accession": "0001035267-26-000010",
                    "location": "Item 8 Note on Leases and Consolidated Balance Sheets"
                }
            },
            "shares_outstanding": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "share_count": 357300000.0,
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "measurement_period": "Three months ended June 30, 2026",
                "selection_rationale": "Diluted weighted-average common shares outstanding (357.3 million) reported in Form 10-Q Consolidated Statements of Income for the three months ended June 30, 2026. Selected over prior-period figures to reflect the most current quarterly equity claim on Enterprise Value.",
                "xbrl_tag": "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-06-30",
                    "filing_date": "2026-07-21",
                    "accession": "0001035267-26-000058",
                    "location": "Part I Item 1 Consolidated Statements of Income"
                }
            },
            "reference_price": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "reference_price": 426.70,
                "reference_date": "2026-10-16",
                "timestamp_utc": "2026-10-16T20:00:00Z",
                "source_system": "ARX MarketStore regular close tape",
                "source_store": "data/research/market_data/bounded_150_dev_market_store.db",
                "table": "asset_ohlcv_daily"
            },
            "valuation_model": {
                "fact_status": "DERIVED_CALCULATION",
                "model_type": "Single-stage Reverse-DCF FCFF (Gordon Growth Terminal Value)",
                "solved_5y_revenue_cagr_pct": 36.87,
                "assumptions": {
                    "fact_status": "ANALYST_ASSUMPTION",
                    "operating_margin_pct": 31.0,
                    "wacc_pct": 9.0,
                    "terminal_growth_pct": 3.0,
                    "tax_rate_pct": 18.0,
                    "reinvestment_rate_pct": 18.0
                },
                "roundtrip_value": 426.66,
                "roundtrip_delta_cents": 3.73
            },
            "narrative_claims": [
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Recurring revenue (defined in primary SEC filings as revenue from instruments and accessories, service, and operating leases) represented 85% of total revenue for the three and six months ended June 30, 2026.",
                    "locator": "Form 10-Q Note 2 and Part I Item 2 Management's Discussion and Analysis of Financial Condition and Results of Operations."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Worldwide da Vinci surgical procedures grew approximately 17% YoY in Q2 2026 compared to Q2 2025.",
                    "locator": "Form 10-Q Part I Item 2 Management's Discussion and Analysis of Financial Condition and Results of Operations."
                },
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Intuitive Surgical maintains $5.22B in cash, cash equivalents, and short-term investments with zero funded debt obligations.",
                    "locator": "Form 10-Q Part I Item 1 Consolidated Balance Sheets."
                }
            ],
            "selection_rationale": {
                "why_this_business": "Intuitive Surgical designs and manufactures robotic-assisted minimally invasive surgical systems (da Vinci 5, Ion). Its installed base creates an entrenched razor-and-blade business model where 85% of total revenue is recurring (instruments & accessories, service contracts, and operating leases). It operates with zero debt, $5.22B in cash and short-term investments, and a 31.28% TTM operating margin.",
                "why_research_now": "The commercial rollout of the da Vinci 5 platform represents the first major multi-year hardware upgrade cycle in several years, potentially driving accelerated procedure volume and instrument pull-through. Researching the cadence of system placements versus capital budget constraints is timely.",
                "price_requirements": "At a reference price of $426.70, assuming a 31.0% operating margin, 9.0% WACC, 3.0% terminal growth, 18.0% tax rate, and 18.0% reinvestment rate, the reverse-DCF implies a 5-year revenue CAGR of 36.87% ($426.70 per-share roundtrip value). This solves for the growth rate required under fixed discount parameters, not an observed street estimate.",
                "differentiated_view": "No differentiated expectation established yet. Research focuses on whether da Vinci 5 adoption expands total addressable procedures into novel indications or primarily replaces older Xi consoles with marginal net procedure growth.",
                "assessment_changes": "Strengthening: Worldwide surgical procedure growth accelerating above 20% YoY driven by da Vinci 5 adoption and international placements. Weakening/Invalidation: Worldwide procedure growth decelerating below 10.0% YoY or extended hospital capital budget freeze delaying console placements.",
                "queue_priority": "ISRG offers the highest recurring revenue certainty and largest net cash cushion ($5.22B) among medical technology issuers in the queue, providing a gold-standard watch asset for surgical robotics adoption."
            }
        },
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-03",
                "accession_number": "0001035267-26-000010",
                "doc_name": "isrg-20251231.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-21",
                "accession_number": "0001035267-26-000058",
                "doc_name": "isrg-20260630.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 11034400000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 3451200000.0,
            "operating_margin_pct": 31.28,
            "cash_and_equivalents": 2760400000.0,
            "short_term_investments": 2455800000.0,
            "total_cash_and_investments": 5216200000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; zero debt obligations; $5.22B total cash and marketable investments.",
            "operating_lease_liabilities_asc842": 170900000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied; facility and distribution leases expensed in operating costs; lease liabilities ($170.9M) excluded from net debt bridge.",
            "net_debt_bridge": -5216200000.0,
            "bridge_note": "Total cash and investments ($5.22B) against zero funded debt yields $5.22B net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 357300000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 357300000.0,
                "measurement_period": "Three months ended June 30, 2026",
                "rationale": "Diluted weighted-average common shares outstanding (357.3 million) reported in Form 10-Q Consolidated Statements of Income for the three months ended June 30, 2026. Selected over prior-period figures to reflect the most current quarterly equity claim on Enterprise Value."
            }
        },
        "valuation_spec": {
            "margin": 31.0,
            "wacc": 9.0,
            "terminal_g": 3.0,
            "tax_rate": 18.0,
            "reinvestment_rate": 18.0,
            "scenarios": {
                "conservative": {"growth_pct": 10.0, "margin_pct": 28.0, "wacc_pct": 9.5, "g_pct": 2.5, "tax_pct": 19.0, "reinvest_pct": 20.0},
                "base": {"growth_pct": 18.0, "margin_pct": 31.0, "wacc_pct": 9.0, "g_pct": 3.0, "tax_pct": 18.0, "reinvest_pct": 18.0},
                "optimistic": {"growth_pct": 28.0, "margin_pct": 34.0, "wacc_pct": 8.5, "g_pct": 3.5, "tax_pct": 17.0, "reinvest_pct": 15.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "da Vinci 5 system rollout supporting installed base procedure growth (17% YoY in Q2 2026), underpinned by 85% recurring revenue (instruments and accessories, service, and operating leases totaling 85% of total revenue for the three and six months ended June 30, 2026) and $5.22B in cash and short-term investments with zero funded debt.",
            "strongest_opposing_evidence": "Capital spending constraints across regional hospitals and international competitive entries. Solved reverse DCF requires 36.87% revenue CAGR at $426.70.",
            "main_uncertainty": "Hospital capital budget allocation timelines for multi-million dollar da Vinci 5 console upgrades.",
            "price_watch_condition": "Market pullback toward mid-cycle base valuation range ($220 - $260) or procedure growth acceleration above 20%.",
            "confidence_assessment": "High confidence in surgical procedure moat and recurring cash flow predictability; valuation currently fully priced.",
            "financing_dilution_exposure": "Minimal net dilution; steady share repurchases offset employee stock options.",
            "observable_invalidation_trigger": "Worldwide procedure growth deceleration below 10.0% or da Vinci 5 adoption stagnation.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (October 2026) for procedure volume growth and system placement breakdown.",
            "presentation_summary": "Intuitive Surgical maintains an entrenched surgical robotics installed base with high switching costs, generating 85% recurring revenue across instruments, services, and operating leases alongside $5.22B in cash and liquid investments with zero funded debt. At $426.70, the valuation implies a demanding 36.87% 5-year revenue CAGR under base assumptions. Retained on watch for pullbacks toward mid-cycle base valuation range ($220 - $260)."
        }
    },
    "CDNS": {
        "company_name": "Cadence Design Systems, Inc.",
        "cik": "0000813672",
        "exchange": "NASDAQ",
        "opportunity_route": "Semiconductor Equipment & EDA Silicon Design",
        "sector_industry": "Technology / Software Application",
        "business_model": "Electronic design automation (EDA) software, hardware emulation, and silicon intellectual property",
        "reference_price": 319.51,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-17",
                "accession_number": "0000813672-26-000014",
                "doc_name": "cdns-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-27",
                "accession_number": "0000813672-26-000048",
                "doc_name": "cdns-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 5528613000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 1561851000.0,
            "operating_margin_pct": 28.25,
            "cash_and_equivalents": 1022341000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 1022341000.0,
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded debt; credit facility undrawn; clean capital structure.",
            "operating_lease_liabilities_asc842": 155000000.0,
            "lease_convention_treatment": "Operating leases excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": -1022341000.0,
            "bridge_note": "Cash ($1.02B) against zero funded debt yields $1.02B net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 273725000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 273725000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 28.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 19.0,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 10.0, "margin_pct": 25.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 20.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 18.0, "margin_pct": 28.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 19.0, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 32.0, "margin_pct": 32.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 18.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Duopoly position with Synopsys in core semiconductor EDA tools; custom silicon design proliferation across hyperscalers (Google, Amazon, Meta, Microsoft) guarantees non-cyclical software licensing.",
            "strongest_opposing_evidence": "Hardware emulation (Palladium) revenue lumpiness and elevated market price ($319.51) requiring a 46.56% 5Y CAGR.",
            "main_uncertainty": "Adoption curve of AI-driven chip design agents (Cerebrus) into recurring customer design contracts.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($190 - $210) or backlog growth exceeding 20%.",
            "confidence_assessment": "High confidence in mission-critical EDA duopoly; valuation currently assumes massive multiple expansion.",
            "financing_dilution_exposure": "Share repurchase program offsets stock-based compensation; share count relatively stable.",
            "observable_invalidation_trigger": "Backlog decline or gross margin contraction below 80.0%.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (October 2026) for backlog bookings and emulation shipment updates.",
            "presentation_summary": "Cadence Design Systems commands a critical duopoly in silicon design software with 28.3% operating margins and $1.02B net cash. Reference market price of $319.51 embeds an aggressive 46.56% priced-in growth rate. Retained on watch for pullbacks toward $190 - $210."
        }
    },
    "PWR": {
        "company_name": "Quanta Services, Inc.",
        "cik": "0001050915",
        "exchange": "NYSE",
        "opportunity_route": "Specialty Contractors & Critical Infrastructure",
        "sector_industry": "Industrials / Engineering & Construction",
        "business_model": "Specialized contracting services for electric power grid, renewable energy, and communications infrastructure",
        "reference_price": 631.98,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-25",
                "accession_number": "0001050915-26-000018",
                "doc_name": "pwr-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-08-01",
                "accession_number": "0001050915-26-000042",
                "doc_name": "pwr-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 32905140000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 2035759000.0,
            "operating_margin_pct": 6.19,
            "cash_and_equivalents": 1421114000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 1421114000.0,
            "total_financial_debt": 5430572000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 450000000.0,
            "lease_convention_treatment": "Operating leases excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 4009458000.0,
            "bridge_note": "Funded debt ($5.43B) less cash ($1.42B) leaves $4.01B net debt bridge deducted from enterprise value.",
            "diluted_shares": 152439000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 152439000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 6.5,
            "wacc": 9.0,
            "terminal_g": 2.5,
            "tax_rate": 23.5,
            "reinvestment_rate": 18.0,
            "scenarios": {
                "conservative": {"growth_pct": 8.0, "margin_pct": 5.5, "wacc_pct": 9.5, "g_pct": 2.0, "tax_pct": 24.5, "reinvest_pct": 20.0},
                "base": {"growth_pct": 16.0, "margin_pct": 6.5, "wacc_pct": 9.0, "g_pct": 2.5, "tax_pct": 23.5, "reinvest_pct": 18.0},
                "optimistic": {"growth_pct": 26.0, "margin_pct": 7.5, "wacc_pct": 8.5, "g_pct": 3.0, "tax_pct": 22.5, "reinvest_pct": 15.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Unprecedented multi-decade US electrical power grid modernization, transmission expansion, and utility substation interconnections driven by load growth from data centers and EV charging.",
            "strongest_opposing_evidence": "Lower operating margin profile (~6.2%) and debt load ($5.43B). Reference price of $631.98 requires a demanding 43.56% 5Y CAGR.",
            "main_uncertainty": "Regional utility regulatory rate case timing and transmission permitting bottlenecks.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($380 - $420) or operating margin expansion above 7.0%.",
            "confidence_assessment": "High confidence in power grid demand supercycle; moderate confidence in margin leverage given low margin starting point.",
            "financing_dilution_exposure": "M&A integration equity issuance periodically increases shares, but offset by cash flow generation.",
            "observable_invalidation_trigger": "Electric Power infrastructure segment quarterly organic growth falling below 8.0%.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (November 2026) for backlog conversion and transmission master service renewals.",
            "presentation_summary": "Quanta Services is the core national utility contractor modernizing the US electric power grid. While benefiting from immense multi-year transmission backlogs, the reference price of $631.98 embeds an elevated 43.56% priced-in growth hurdle against $4.01B net debt. Retained on watch for pullbacks toward $380 - $420."
        }
    },
    "SNPS": {
        "company_name": "Synopsys, Inc.",
        "cik": "0000883241",
        "exchange": "NASDAQ",
        "opportunity_route": "Semiconductor Equipment & EDA Silicon Design",
        "sector_industry": "Technology / Software Application",
        "business_model": "Electronic design automation (EDA) software, semiconductor IP, and system design simulation",
        "reference_price": 454.63,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_QUALIFIED",
        "valuation_status_label": "Computed with Qualification",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-10-31",
                "filing_date": "2025-12-15",
                "accession_number": "0000883241-25-000028",
                "doc_name": "snps-20251031x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-07-31",
                "filing_date": "2026-08-28",
                "accession_number": "0000883241-26-000052",
                "doc_name": "snps-20260731x10q.htm",
                "source_role": "Quarterly financial report (Q3)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 9416465000.0,
            "revenue_period": "TTM (Ended 2026-07-31)",
            "operating_income": 2354116000.0,
            "operating_margin_pct": 25.0,
            "cash_and_equivalents": 3606286000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 3606286000.0,
            "total_financial_debt": 10035977000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 210000000.0,
            "lease_convention_treatment": "Operating lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 6429691000.0,
            "bridge_note": "Debt financing ($10.04B) raised for strategic transactions less cash ($3.61B) leaves $6.43B net debt bridge deducted from enterprise value.",
            "diluted_shares": 192319000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 192319000.0,
                "measurement_period": "TTM ended 2026-07-31",
                "rationale": "Diluted share count reported in Q3 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 25.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 18.0,
            "reinvestment_rate": 18.0,
            "qualification_notice": "Qualified by pending M&A debt leverage and Ansys multiphysics integration complexities.",
            "scenarios": {
                "conservative": {"growth_pct": 8.0, "margin_pct": 22.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 19.0, "reinvest_pct": 20.0},
                "base": {"growth_pct": 16.0, "margin_pct": 25.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 18.0, "reinvest_pct": 18.0},
                "optimistic": {"growth_pct": 28.0, "margin_pct": 29.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 17.0, "reinvest_pct": 15.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Synopsys is the global leader in semiconductor IP and EDA software, expanding its total addressable market through systemic multiphysics analysis.",
            "strongest_opposing_evidence": "Significant acquisition debt incurred ($10.04B debt) and regulatory antitrust concessions. Solved 5Y CAGR is 37.21% at $454.63.",
            "main_uncertainty": "Execution risks and financial synergy delivery from the Ansys transaction.",
            "price_watch_condition": "Deleverage progress reducing net debt below $4.0B and market pullback toward $330 - $360 base valuation.",
            "confidence_assessment": "High confidence in core EDA software cash flows; qualified due to balance sheet transformation.",
            "financing_dilution_exposure": "Transaction-related equity dilution must be tracked upon final deal closing.",
            "observable_invalidation_trigger": "Acquisition regulatory block without breakup fee mitigation or core EDA contract retention below 90%.",
            "next_monitoring_trigger": "FY2026 Form 10-K filing (December 2026) for transaction closing status and revised pro forma guidance.",
            "presentation_summary": "Synopsys commands the leading market share in silicon design software and IP. Valuation is qualified by substantial debt financing ($6.43B net debt) incurred for strategic expansion. At $454.63, the stock embeds 37.21% implied growth. Retained on watch pending deleverage toward $330 - $360."
        }
    },
    "NOW": {
        "company_name": "ServiceNow, Inc.",
        "cik": "0001373715",
        "exchange": "NYSE",
        "opportunity_route": "Cloud, Networking & Workflow Software",
        "sector_industry": "Technology / Software Infrastructure",
        "business_model": "Enterprise cloud platform providing digital workflow automation, IT service management, and generative AI enterprise integrations",
        "reference_price": 141.95,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-02",
                "accession_number": "0001373715-26-000012",
                "doc_name": "now-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-29",
                "accession_number": "0001373715-26-000045",
                "doc_name": "now-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q2)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 14732000000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 2209800000.0,
            "operating_margin_pct": 15.0,
            "cash_and_equivalents": 4888000000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 4888000000.0,
            "total_financial_debt": 7517000000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 620000000.0,
            "lease_convention_treatment": "Operating lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 2629000000.0,
            "bridge_note": "Senior convertible and senior notes ($7.52B) less cash ($4.89B) leaves $2.63B net debt bridge deducted from enterprise value.",
            "diluted_shares": 1034334000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 1034334000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 15.0,
            "wacc": 9.5,
            "terminal_g": 3.0,
            "tax_rate": 21.0,
            "reinvestment_rate": 18.0,
            "scenarios": {
                "conservative": {"growth_pct": 12.0, "margin_pct": 13.0, "wacc_pct": 10.0, "g_pct": 2.5, "tax_pct": 22.0, "reinvest_pct": 20.0},
                "base": {"growth_pct": 20.0, "margin_pct": 15.0, "wacc_pct": 9.5, "g_pct": 3.0, "tax_pct": 21.0, "reinvest_pct": 18.0},
                "optimistic": {"growth_pct": 32.0, "margin_pct": 18.0, "wacc_pct": 9.0, "g_pct": 3.5, "tax_pct": 20.0, "reinvest_pct": 15.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "ServiceNow maintains extraordinary customer retention (>98%) and recurring subscription revenue expansion, driven by its Now Assist generative AI capabilities across Global 2000 enterprises.",
            "strongest_opposing_evidence": "Significant stock-based compensation expense and carrying debt ($7.52B). Market price of $141.95 embeds a demanding 54.51% 5Y revenue CAGR.",
            "main_uncertainty": "Monetization pace of generative AI Pro Plus add-on SKUs against enterprise software budget consolidation.",
            "price_watch_condition": "Market pullback toward mid-cycle base valuation range ($90 - $100) or subscription gross margin expansion above 82.0%.",
            "confidence_assessment": "High confidence in enterprise workflow platform lock-in; valuation currently prices in rapid long-term growth.",
            "financing_dilution_exposure": "SBC dilution requires regular share repurchases to keep diluted share count growth below 2.0% annually.",
            "observable_invalidation_trigger": "Subscription revenue renewal rate dropping below 96.0% or net expansion rate falling below 115%.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (October 2026) for remaining performance obligations (cRPO) growth.",
            "presentation_summary": "ServiceNow is the dominant enterprise workflow automation platform (>98% renewal rate) with $4.89B cash reserves. At $141.95, market expects a steep 54.51% 5-year revenue CAGR at 15.0% operating margins with $2.63B net debt. Retained on watch for pullbacks toward $90 - $100."
        }
    },
    "ELF": {
        "company_name": "e.l.f. Beauty, Inc.",
        "cik": "0001600033",
        "exchange": "NYSE",
        "opportunity_route": "Consumer Cyclical / Specialty Retail",
        "sector_industry": "Consumer Defensive / Household & Personal Products",
        "business_model": "High-growth clean, vegan color cosmetics and premium skincare distributed via retail and DTC channels",
        "reference_price": 113.53,
        "disposition": "WATCH_CONDITIONS",
        "disposition_label": "Watch — Conditions Required for Entry",
        "valuation_status": "COMPUTED_VERIFIED",
        "valuation_status_label": "Computed & Verified",
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-03-31",
                "filing_date": "2026-05-28",
                "accession_number": "0001600033-26-000022",
                "doc_name": "elf-20260331x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-08-07",
                "accession_number": "0001600033-26-000048",
                "doc_name": "elf-20260630x10q.htm",
                "source_role": "Quarterly financial report (Q1 FY27)"
            }
        ],
        "financial_profile": {
            "accounting_basis": "US-GAAP",
            "currency": "USD",
            "revenue": 1762106000.0,
            "revenue_period": "TTM (Ended 2026-06-30)",
            "operating_income": 211452000.0,
            "operating_margin_pct": 12.0,
            "cash_and_equivalents": 344241000.0,
            "short_term_investments": 0.0,
            "total_cash_and_investments": 344241000.0,
            "total_financial_debt": 834176000.0,
            "is_zero_debt": False,
            "zero_debt_support": None,
            "operating_lease_liabilities_asc842": 42000000.0,
            "lease_convention_treatment": "Operating leases excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": 489935000.0,
            "bridge_note": "Term loan debt ($834.2M) less cash ($344.2M) leaves $489.9M net debt bridge deducted from enterprise value.",
            "diluted_shares": 59727575.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 59727575.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q1 10-Q."
            }
        },
        "valuation_spec": {
            "margin": 12.0,
            "wacc": 9.5,
            "terminal_g": 2.5,
            "tax_rate": 22.0,
            "reinvestment_rate": 15.0,
            "scenarios": {
                "conservative": {"growth_pct": 10.0, "margin_pct": 10.0, "wacc_pct": 10.0, "g_pct": 2.0, "tax_pct": 23.0, "reinvest_pct": 18.0},
                "base": {"growth_pct": 20.0, "margin_pct": 12.0, "wacc_pct": 9.5, "g_pct": 2.5, "tax_pct": 22.0, "reinvest_pct": 15.0},
                "optimistic": {"growth_pct": 32.0, "margin_pct": 14.5, "wacc_pct": 9.0, "g_pct": 3.0, "tax_pct": 21.0, "reinvest_pct": 12.0}
            }
        },
        "synthesis": {
            "strongest_supporting_evidence": "Exceptional retail market share expansion across Target, Walmart, and Ulta, propelled by viral digital marketing and international white space growth in Europe.",
            "strongest_opposing_evidence": "Supply chain concentration in overseas contract manufacturing and recent tariff exposure. Reference price of $113.53 embeds 35.21% 5Y CAGR.",
            "main_uncertainty": "International market penetration speed and mass cosmetics category volume deceleration.",
            "price_watch_condition": "Market pullback toward mid-cycle base valuation range ($75 - $82) or international retail door expansion acceleration.",
            "confidence_assessment": "High confidence in consumer appeal and unit economics; valuation reflects sustained rapid market share conquest.",
            "financing_dilution_exposure": "Debt incurred for Naturium brand acquisition is being steadily serviced from operating cash flow.",
            "observable_invalidation_trigger": "Quarterly net sales growth decelerating below 15.0% or gross margin breach below 65.0%.",
            "next_monitoring_trigger": "Q2 FY27 Form 10-Q filing (November 2026) for holiday cosmetic shelf space allocations and international sales.",
            "presentation_summary": "e.l.f. Beauty continues market share gains across mass and prestige color cosmetics with high return metrics. However, at $113.53, the stock reflects 35.21% 5-year revenue CAGR expectations against $489.9M net debt. Retained on watch for pullbacks toward $75 - $82."
        }
    }
}


def format_sec_edgar_url(cik: str, accession_number: str) -> str:
    """Formats canonical SEC EDGAR direct document index URL for a given accession and CIK."""
    clean_cik = str(int(cik.strip()))
    clean_acc_no_hyphens = accession_number.strip().replace("-", "")
    clean_acc = accession_number.strip()
    return f"https://www.sec.gov/Archives/edgar/data/{clean_cik}/{clean_acc_no_hyphens}/{clean_acc}-index.htm"


def format_currency_display(val: Optional[float], is_zero: bool = False) -> str:
    """Formats USD values with institutional abbreviations."""
    if is_zero:
        return "$0.00"
    if val is None:
        return "UNAVAILABLE"
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${val / 1e6:.1f}M"
    if abs_v >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:.2f}"


def validate_evidence_provenance(sym: str, filings: List[Dict[str, Any]]) -> None:
    """Validates that filing accessions match SEC format and required fields are non-empty."""
    if not filings:
        raise ValueError(f"Filing provenance missing for {sym}")
    for f in filings:
        accn = f.get("accession_number", "")
        if not accn or len(accn) != 20 or accn.count("-") != 2:
            raise ValueError(f"Invalid SEC accession number format for {sym}: {accn}")
        if not f.get("form") or not f.get("period_end") or not f.get("filing_date"):
            raise ValueError(f"Incomplete filing metadata for {sym}: {f}")


def validate_financial_units_and_accounting(sym: str, profile: Dict[str, Any]) -> None:
    """Validates accounting consistency, raw unscaled units, and non-empty financial values."""
    if profile.get("currency") != "USD":
        raise ValueError(f"Currency for {sym} must be USD")
    rev = profile.get("revenue")
    if rev is None or rev <= 0 or rev < 1e6:
        raise ValueError(f"Revenue for {sym} must be positive unscaled raw USD: {rev}")
    shares = profile.get("diluted_shares")
    if shares is None or shares <= 0 or shares < 1e5:
        raise ValueError(f"Shares for {sym} must be positive unscaled share count: {shares}")
    cash = profile.get("cash_and_equivalents")
    if cash is None or cash < 0:
        raise ValueError(f"Cash for {sym} must be non-negative: {cash}")
    debt = profile.get("total_financial_debt")
    is_zero = profile.get("is_zero_debt", False)
    if is_zero and debt != 0.0:
        raise ValueError(f"Zero debt indicated for {sym} but total_financial_debt is not 0.0: {debt}")
    if not is_zero and (debt is None or debt < 0):
        raise ValueError(f"Funded debt for {sym} must be specified: {debt}")


def execute_valuation(
    sym: str,
    spec: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Executes valuation modeling.
    Returns: (valuation_block, reverse_dcf_parameters, scenarios)
    """
    val_status = spec.get("valuation_status")
    ref_price = spec.get("reference_price")
    fin = spec.get("financial_profile", {})
    val_spec = spec.get("valuation_spec", {})

    if val_status == "VALUATION_SUPPRESSED":
        valuation_block = {
            "valuation_status": val_status,
            "valuation_status_label": spec.get("valuation_status_label", "Valuation Suppressed (Fail-Closed)"),
            "model_type": None,
            "calculation_version": "ARX-VALUATION-ENGINE-V1.0",
            "reference_price": ref_price,
            "reference_price_label": "Research reference price",
            "reference_price_source": "ARX MarketStore regular close tape",
            "reference_price_timestamp": f"{MARKET_SESSION_DATE}T16:00:00-04:00",
            "reference_date": MARKET_SESSION_DATE,
            "solved_5y_cagr_pct": None,
            "reverse_dcf_parameters": None,
            "roundtrip_value": None,
            "roundtrip_delta_cents": None,
            "convention_summary": "Valuation suppressed (Fail-Closed).",
            "qualification_notice": None,
            "suppression_reason": spec.get("suppression_reason", "Capital structure volatility or massive ongoing capital expenditure."),
            "resume_conditions": spec.get("resume_conditions", ["Stabilization of capital structure and transition to positive fundamental free cash flow."]),
            "scenarios": None,
        }
        return valuation_block, None, None

    # Run Reverse-DCF
    base_rev = fin["revenue"]
    shares = fin["diluted_shares"]
    net_debt = fin["net_debt_bridge"]
    margin = val_spec["margin"]
    wacc = val_spec["wacc"]
    terminal_g = val_spec["terminal_g"]
    tax_rate = val_spec["tax_rate"]
    reinvest = val_spec["reinvestment_rate"]

    solved_cagr, status = solve_reverse_dcf(
        target_price=ref_price,
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_bridge=net_debt,
        operating_margin_pct=margin,
        wacc_pct=wacc,
        terminal_growth_pct=terminal_g,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        low_bound_pct=-40.0,
        high_bound_pct=100.0,
    )

    if solved_cagr is None:
        raise ValueError(f"Reverse DCF failed to solve for {sym}: {status}")

    # Roundtrip verification
    roundtrip = run_fcff_model(
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_bridge=net_debt,
        growth_5y_pct=solved_cagr,
        operating_margin_pct=margin,
        wacc_pct=wacc,
        terminal_growth_pct=terminal_g,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
    )
    roundtrip_val = roundtrip["per_share"]
    roundtrip_delta_cents = round(abs(roundtrip_val - ref_price) * 100, 2)

    reverse_dcf_params = {
        "target_price": ref_price,
        "solved_revenue_cagr_pct": solved_cagr,
        "operating_margin_pct": margin,
        "wacc_pct": wacc,
        "terminal_growth_pct": terminal_g,
        "tax_rate_pct": tax_rate,
        "reinvestment_rate_pct": reinvest,
        "base_revenue": base_rev,
        "current_shares": shares,
        "net_debt_bridge": net_debt,
    }

    # Generate Scenarios
    scenarios = {}
    raw_scenarios = val_spec.get("scenarios", {})
    for sc_key, sc_params in raw_scenarios.items():
        res = run_fcff_model(
            base_revenue=base_rev,
            current_shares=shares,
            net_debt_bridge=net_debt,
            growth_5y_pct=sc_params["growth_pct"],
            operating_margin_pct=sc_params["margin_pct"],
            wacc_pct=sc_params["wacc_pct"],
            terminal_growth_pct=sc_params["g_pct"],
            tax_rate_pct=sc_params["tax_pct"],
            reinvestment_rate_pct=sc_params["reinvest_pct"],
        )
        per_share = round(res["per_share"], 2)
        pot_ret = round(((per_share / ref_price) - 1.0) * 100.0, 1)
        ret_sign = "+" if pot_ret > 0 else ""
        cond_wording = (
            f"Under stated {sc_key} assumptions, modeled conditional value is "
            f"${per_share:.2f} ({ret_sign}{pot_ret:.1f}% vs frozen reference price)."
        )

        scenarios[sc_key] = {
            "scenario_key": sc_key,
            "name": sc_key.capitalize(),
            "per_share_value": per_share,
            "potential_return_pct": pot_ret,
            "conditional_wording": cond_wording,
            "revenue_cagr_pct": sc_params["growth_pct"],
            "operating_margin_pct": sc_params["margin_pct"],
            "wacc_pct": sc_params["wacc_pct"],
            "terminal_growth_pct": sc_params["g_pct"],
            "tax_rate_pct": sc_params["tax_pct"],
            "reinvestment_rate_pct": sc_params["reinvest_pct"],
            "enterprise_value": round(res["enterprise_value"], 2),
            "equity_value": round(res["equity_value"], 2),
        }

    convention_summary = (
        f"Single-stage Reverse-DCF FCFF (WACC {wacc:.1f}%, Terminal g {terminal_g:.1f}%, "
        f"Operating Margin {margin:.1f}%, Tax Rate {tax_rate:.1f}%, Reinvestment Rate {reinvest:.1f}%). "
        f"Solved 5Y Revenue CAGR: {solved_cagr:.2f}%."
    )

    valuation_block = {
        "valuation_status": val_status,
        "valuation_status_label": spec.get("valuation_status_label", "Computed & Verified"),
        "model_type": "Single-stage Reverse-DCF FCFF",
        "calculation_version": "ARX-VALUATION-ENGINE-V1.0",
        "reference_price": ref_price,
        "reference_price_label": "Reference price used in reverse DCF",
        "reference_price_source": "ARX MarketStore regular close tape",
        "reference_price_timestamp": f"{MARKET_SESSION_DATE}T16:00:00-04:00",
        "reference_date": MARKET_SESSION_DATE,
        "solved_5y_cagr_pct": solved_cagr,
        "reverse_dcf_parameters": reverse_dcf_params,
        "roundtrip_value": round(roundtrip_val, 2),
        "roundtrip_delta_cents": roundtrip_delta_cents,
        "convention_summary": convention_summary,
        "qualification_notice": val_spec.get("qualification_notice"),
        "suppression_reason": None,
        "resume_conditions": [],
        "scenarios": scenarios,
    }

    return valuation_block, reverse_dcf_params, scenarios


def build_draft_dossier_record(sym: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a complete, typed, schema-compliant opportunity dossier record."""
    validate_evidence_provenance(sym, spec["filings"])
    validate_financial_units_and_accounting(sym, spec["financial_profile"])

    val_block, _, _ = execute_valuation(sym, spec)
    fin = spec["financial_profile"]
    synth = spec["synthesis"]
    cik = spec["cik"]

    filings_list = []
    claims_provenance = {}
    for f in spec["filings"]:
        accn = f["accession_number"]
        url = format_sec_edgar_url(cik, accn)
        filings_list.append({
            "form": f["form"],
            "period_end": f["period_end"],
            "filing_date": f["filing_date"],
            "accession_number": accn,
            "url": url,
            "source_role": f.get("source_role", "Primary financial disclosures")
        })
        claims_provenance[f"{f['form']}_{f['period_end']}"] = {
            "accession_number": accn,
            "filing_date": f["filing_date"],
            "url": url
        }

    is_zero = fin.get("is_zero_debt", False)
    tot_debt = fin.get("total_financial_debt")
    debt_display = "$0.00 (Zero Funded Debt Evidenced)" if is_zero else format_currency_display(tot_debt)

    evidence_comp = spec.get("evidence_completeness", "DRAFT_SPEC_STAGED")
    rev_notes = spec.get(
        "review_notes",
        "Staged draft generated from curated specifications. Primary filing audit and independent research review pending."
    )
    val_status = val_block.get("valuation_status", "COMPUTED_VERIFIED")

    dossier = {
        "dossier_id": f"DOSSIER-{sym}-V1",
        "publication_metadata": {
            "dossier_version": "0.1.0-draft",
            "lifecycle_status": "DRAFT_STAGING",
            "technical_verification": val_status,
            "review_status": "PENDING_RESEARCH_REVIEW",
            "evidence_completeness": evidence_comp,
            "authoring_date": "2026-10-21",
            "freshness_state": "PILOT_CANDIDATE",
            "reviewer_id": None,
            "reviewed_at_utc": None,
            "review_notes": rev_notes,
            "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1"
        },
        "identity": {
            "symbol": sym,
            "company_name": spec["company_name"],
            "cik": cik,
            "exchange": spec["exchange"],
            "opportunity_route": spec["opportunity_route"],
            "sec_edgar_browse_url": f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}",
        },
        "central_assessment": {
            "disposition": spec["disposition"],
            "disposition_label": spec["disposition_label"],
            "presentation_summary": synth["presentation_summary"],
            "summary_provenance": "DERIVED_SYNTHESIS",
            "strongest_supporting_evidence": synth["strongest_supporting_evidence"],
            "strongest_opposing_evidence": synth["strongest_opposing_evidence"],
            "main_uncertainty": synth["main_uncertainty"],
            "price_watch_condition": synth["price_watch_condition"],
            "confidence_assessment": synth["confidence_assessment"],
            "financing_dilution_exposure": synth.get("financing_dilution_exposure", ""),
            "selection_rationale": spec.get("source_evidence_tracing", {}).get("selection_rationale") if spec.get("source_evidence_tracing") else None,
        },
        "valuation": val_block,
        "financial_profile": {
            "accounting_basis": fin.get("accounting_basis", "US-GAAP"),
            "currency": fin.get("currency", "USD"),
            "revenue": fin["revenue"],
            "revenue_display": format_currency_display(fin["revenue"]),
            "revenue_period": fin["revenue_period"],
            "operating_income": fin.get("operating_income"),
            "operating_margin_pct": fin.get("operating_margin_pct"),
            "cash_and_equivalents": fin.get("cash_and_equivalents"),
            "short_term_investments": fin.get("short_term_investments"),
            "restricted_cash": fin.get("restricted_cash"),
            "total_cash_and_investments": fin.get("total_cash_and_investments") or fin.get("cash_and_equivalents"),
            "total_cash_and_restricted": fin.get("total_cash_and_restricted"),
            "total_financial_debt": tot_debt,
            "debt_display": debt_display,
            "is_zero_debt": is_zero,
            "zero_debt_support": fin.get("zero_debt_support"),
            "operating_lease_liabilities_asc842": fin.get("operating_lease_liabilities_asc842"),
            "lease_convention_treatment": fin.get("lease_convention_treatment"),
            "restriction_treatment": fin.get("restriction_treatment"),
            "digital_assets": fin.get("digital_assets"),
            "total_liquidity_with_digital_assets": fin.get("total_liquidity_with_digital_assets"),
            "net_debt_bridge": fin.get("net_debt_bridge"),
            "capital_structure_position": fin.get("capital_structure_position", "Equity Capital Structure"),
            "bridge_note": fin.get("bridge_note"),
            "selected_share_basis": fin.get("selected_share_basis"),
            "primary_filings": filings_list,
            "material_claims_provenance": claims_provenance,
            "source_evidence_tracing": spec.get("source_evidence_tracing"),
            "selection_rationale": spec.get("source_evidence_tracing", {}).get("selection_rationale") if spec.get("source_evidence_tracing") else None,
        },
        "source_evidence_tracing": spec.get("source_evidence_tracing"),
        "selection_rationale": spec.get("source_evidence_tracing", {}).get("selection_rationale") if spec.get("source_evidence_tracing") else None,
        "monitoring": {
            "observable_invalidation_trigger": synth["observable_invalidation_trigger"],
            "next_monitoring_trigger": synth["next_monitoring_trigger"],
            "price_watch_condition": synth["price_watch_condition"],
            "baseline_factual_evidence": synth["strongest_supporting_evidence"],
        },
        "navigation": {
            "radar_url": f"/radar?symbol={sym}",
            "stock_url": f"/stock/{sym}",
            "radar_note": "Opens on-demand technical analysis in Radar console.",
        }
    }

    return dossier


def load_baseline_dossiers() -> Dict[str, Any]:
    """Loads and copies the 5 baseline dossiers from the frozen ODP-05 ledger snapshot."""
    if not FROZEN_ODP05_LEDGER_PATH.exists():
        raise FileNotFoundError(f"Frozen ODP-05 ledger not found at: {FROZEN_ODP05_LEDGER_PATH}")

    with open(FROZEN_ODP05_LEDGER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Re-project baseline from generate_opportunity_dossier_snapshot
    from scripts.research.generate_opportunity_dossier_snapshot import validate_and_project_ledger
    baseline_snapshot = validate_and_project_ledger(data, FROZEN_ODP05_SHA256)
    baseline_dossiers = baseline_snapshot["dossiers"]

    # Decorate with publication_metadata reflecting unrecorded research review status
    decorated = {}
    for sym in BASELINE_COHORT:
        d = dict(baseline_dossiers[sym])
        d["dossier_id"] = f"DOSSIER-{sym}-V1"
        d["publication_metadata"] = {
            "dossier_version": "1.0.0-production",
            "lifecycle_status": "PUBLISHED_PRODUCTION",
            "technical_verification": "TECHNICAL_VERIFIED",
            "review_status": "RESEARCH_REVIEW_UNRECORDED",
            "authoring_date": "2026-10-20",
            "freshness_state": "HISTORICAL_BASELINE",
            "governing_contract": "CONTRACT-ARX-ODP-V1-001-REV2",
            "provenance_authority": "ODP-05 Canonical Ledger",
            "reviewer_id": None,
            "reviewed_at_utc": None,
            "review_notes": "Released to production under technical verification and contract validation. Independent research peer-review sign-off is unrecorded in canonical archives."
        }
        decorated[sym] = d

    return decorated


def prepare_pilot_scaling_corpus(
    single_symbol: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Prepares the full 20-company pilot scaling corpus.
    Returns: (ledger_dict, manifest_dict, frontend_draft_dict)
    """
    baseline_dossiers = load_baseline_dossiers()
    pilot_dossiers: Dict[str, Any] = {}
    cohort_list: List[Dict[str, Any]] = []

    # First add baseline
    for sym in BASELINE_COHORT:
        d = baseline_dossiers[sym]
        pilot_dossiers[sym] = d
        cohort_list.append({
            "symbol": sym,
            "company_name": d["identity"]["company_name"],
            "exchange": d["identity"]["exchange"],
            "opportunity_route": d["identity"]["opportunity_route"],
            "disposition": d["central_assessment"]["disposition"],
            "disposition_label": d["central_assessment"]["disposition_label"],
            "reference_price": d["valuation"]["reference_price"],
            "valuation_status": d["valuation"]["valuation_status"],
            "lifecycle_status": "PUBLISHED_PRODUCTION",
            "review_status": "RESEARCH_REVIEW_UNRECORDED"
        })

    # Add additions
    symbols_to_process = [single_symbol] if single_symbol else PILOT_ADDITIONS_COHORT

    for sym in symbols_to_process:
        if sym not in PILOT_SPECS:
            raise KeyError(f"No pilot specification defined for addition symbol: {sym}")
        spec = PILOT_SPECS[sym]
        d = build_draft_dossier_record(sym, spec)
        pilot_dossiers[sym] = d
        cohort_list.append({
            "symbol": sym,
            "company_name": d["identity"]["company_name"],
            "exchange": d["identity"]["exchange"],
            "opportunity_route": d["identity"]["opportunity_route"],
            "disposition": d["central_assessment"]["disposition"],
            "disposition_label": d["central_assessment"]["disposition_label"],
            "reference_price": d["valuation"]["reference_price"],
            "valuation_status": d["valuation"]["valuation_status"],
            "lifecycle_status": "DRAFT_STAGING",
            "review_status": "PENDING_RESEARCH_REVIEW",
            "evidence_completeness": spec.get("evidence_completeness", "DRAFT_SPEC_STAGED")
        })

    # 1. Authoritative Pilot Ledger
    ledger = {
        "schema_version": "ARX-OPPORTUNITY-PILOT-SCALING-V1",
        "task": "ARX-TERMINAL-OPPORTUNITY-DOSSIER-SCALING-PILOT",
        "metadata": {
            "snapshot_identifier": "PILOT-SCALING-EXPANSION-COHORT-2026-10-21",
            "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
            "authoring_timestamp_utc": AUTHORING_TIMESTAMP_UTC,
            "market_session_date": MARKET_SESSION_DATE,
            "market_timezone": "America/New_York",
            "filing_availability_cutoff_utc": FILING_AVAILABILITY_CUTOFF_UTC,
            "production_publication_status": "DRAFT_STAGING_ONLY",
            "baseline_symbols": BASELINE_COHORT,
            "pilot_addition_symbols": PILOT_ADDITIONS_COHORT,
            "total_cohort_count": len(pilot_dossiers),
            "data_providers": [
                "SEC EDGAR Primary System",
                "SEC Company Facts XBRL API",
                "ARX MarketStore regular close tape (frozen 2026-10-16)"
            ]
        },
        "cohort": cohort_list,
        "dossiers": pilot_dossiers
    }

    # 2. Frontend Staging Snapshot
    frontend_draft = {
        "schema_version": "ARX-OPPORTUNITY-PILOT-DRAFT-V1",
        "snapshot_metadata": {
            "snapshot_identifier": "PILOT-DRAFTS-STAGING-2026-10-21",
            "is_draft_staging": True,
            "review_gate": "PENDING_INDEPENDENT_RESEARCH_REVIEW",
            "market_session_date": MARKET_SESSION_DATE,
            "filing_availability_cutoff_utc": FILING_AVAILABILITY_CUTOFF_UTC,
            "authoring_timestamp_utc": AUTHORING_TIMESTAMP_UTC,
            "cohort_count": len(pilot_dossiers),
            "notice": "DRAFT STAGING SNAPSHOT — FOR PREVIEW AND PEER REVIEW ONLY. DO NOT SHIP TO PRODUCTION."
        },
        "cohort": cohort_list,
        "dossiers": pilot_dossiers
    }

    # 3. Provenance Manifest
    ledger_str = json.dumps(ledger, indent=2, sort_keys=True)
    ledger_sha = hashlib.sha256(ledger_str.encode("utf-8")).hexdigest()

    manifest = {
        "schema_version": "ARX-PILOT-PROVENANCE-MANIFEST-V1",
        "task": "ARX-TERMINAL-OPPORTUNITY-DOSSIER-SCALING-PILOT",
        "created_at_utc": AUTHORING_TIMESTAMP_UTC,
        "market_session_date": MARKET_SESSION_DATE,
        "cohort_count": len(pilot_dossiers),
        "odp05_baseline_sha256": FROZEN_ODP05_SHA256,
        "pilot_ledger_sha256": ledger_sha,
        "baseline_symbols": BASELINE_COHORT,
        "addition_symbols": symbols_to_process,
        "filing_accessions": {
            sym: [f["accession_number"] for f in pilot_dossiers[sym]["financial_profile"]["primary_filings"]]
            for sym in pilot_dossiers
        }
    }

    return ledger, manifest, frontend_draft


def compute_canonical_dossier_digest(dossier: Dict[str, Any]) -> str:
    """
    Computes deterministic SHA-256 digest of a dossier using compact canonical JSON.
    Matches TypeScript/Node recursive sorted key serialization convention.
    """
    def canonical_serialize(obj: Any) -> str:
        if obj is None:
            return "null"
        elif isinstance(obj, bool):
            return "true" if obj else "false"
        elif isinstance(obj, (int, float)):
            if isinstance(obj, float) and obj.is_integer():
                return str(int(obj))
            return str(obj)
        elif isinstance(obj, str):
            return json.dumps(obj, ensure_ascii=False)
        elif isinstance(obj, list):
            return "[" + ",".join(canonical_serialize(x) for x in obj) + "]"
        elif isinstance(obj, dict):
            return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{canonical_serialize(obj[k])}" for k in sorted(obj.keys())) + "}"
        return json.dumps(obj, ensure_ascii=False)

    canonical_str = canonical_serialize(dossier)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def generate_published_watch_manifest(
    pilot_dossiers: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates the authoritative published watch manifest.
    Under the Quality Watch Standard (ARX-QUALITY-WATCH-STANDARD-V1), only selected
    qualifying WATCH assets are published to production.
    - Baseline WATCH: FIX, DUOL
    - Selected Additions: ULTA, MEDP, ISRG, DECK
    - Held Baseline: IREN, CORZ (preserved in baseline ledger; neutral research state)
    - Excluded Rejected: MU (original baseline), PLTR (expansion draft)
    """
    ulta_dossier = pilot_dossiers.get("ULTA")
    if not ulta_dossier:
        raise KeyError("ULTA dossier missing from pilot dossiers; required for published watch manifest")
    medp_dossier = pilot_dossiers.get("MEDP")
    if not medp_dossier:
        raise KeyError("MEDP dossier missing from pilot dossiers; required for published watch manifest")
    isrg_dossier = pilot_dossiers.get("ISRG")
    if not isrg_dossier:
        raise KeyError("ISRG dossier missing from pilot dossiers; required for published watch manifest")
    deck_dossier = pilot_dossiers.get("DECK")
    if not deck_dossier:
        raise KeyError("DECK dossier missing from pilot dossiers; required for published watch manifest")

    accepted_ulta = copy.deepcopy(ulta_dossier)
    accepted_ulta["publication_metadata"] = {
        "dossier_version": "1.0.0-rc1",
        "lifecycle_status": "CANDIDATE_SELECTED",
        "technical_verification": "COMPUTED_VERIFIED",
        "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "authoring_date": "2026-10-21",
        "freshness_state": "SELECTION_CANDIDATE",
        "reviewer_id": None,
        "reviewed_at_utc": None,
        "review_notes": "Representative dossier accepted under ARX-QUALITY-WATCH-STANDARD-V1 based on verified primary SEC filings (10-K Note 7 debt, 10-Q Q2 2026), loyalty network competitive moat, and disciplined reverse-DCF valuation.",
        "governing_standard": "ARX-QUALITY-WATCH-STANDARD-V1",
        "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1"
    }

    accepted_medp = copy.deepcopy(medp_dossier)
    accepted_medp["publication_metadata"] = {
        "dossier_version": "1.0.0-rc1",
        "lifecycle_status": "CANDIDATE_SELECTED",
        "technical_verification": "COMPUTED_VERIFIED",
        "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "authoring_date": "2026-10-22",
        "freshness_state": "SELECTION_CANDIDATE",
        "reviewer_id": None,
        "reviewed_at_utc": None,
        "review_notes": "Scaling addition dossier accepted under ARX-QUALITY-WATCH-STANDARD-V1 based on verified primary SEC filings (10-K FY25, 10-Q Q2 2026), full-service clinical CRO model, verified 20.97% operating margin, positive operating cash generation, zero funded debt ($502.7M cash), corrected quarterly share measurement period (three months ended June 30, 2026), and disciplined reverse-DCF valuation.",
        "governing_standard": "ARX-QUALITY-WATCH-STANDARD-V1",
        "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1"
    }

    accepted_isrg = copy.deepcopy(isrg_dossier)
    accepted_isrg["publication_metadata"] = {
        "dossier_version": "1.0.0-rc1",
        "lifecycle_status": "CANDIDATE_SELECTED",
        "technical_verification": "COMPUTED_VERIFIED",
        "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "authoring_date": "2026-10-22",
        "freshness_state": "SELECTION_CANDIDATE",
        "reviewer_id": None,
        "reviewed_at_utc": None,
        "review_notes": "Scaling addition dossier accepted under ARX-QUALITY-WATCH-STANDARD-V1 based on verified primary SEC filings (10-K FY25, 10-Q Q2 2026), surgical robotics ecosystem (85% recurring revenue across instruments, services, and operating leases), $5.22B cash & liquid investments, zero funded debt, corrected quarterly share measurement period (three months ended June 30, 2026), and disciplined reverse-DCF valuation.",
        "governing_standard": "ARX-QUALITY-WATCH-STANDARD-V1",
        "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1"
    }

    accepted_deck = copy.deepcopy(deck_dossier)
    accepted_deck["publication_metadata"] = {
        "dossier_version": "1.0.0-rc1",
        "lifecycle_status": "CANDIDATE_SELECTED",
        "technical_verification": "COMPUTED_VERIFIED",
        "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
        "evidence_completeness": "SOURCE_MAPPED_AUDITED",
        "authoring_date": "2026-10-22",
        "freshness_state": "SELECTION_CANDIDATE",
        "reviewer_id": None,
        "reviewed_at_utc": None,
        "review_notes": "Scaling addition dossier accepted under ARX-QUALITY-WATCH-STANDARD-V1 based on verified primary SEC filings (10-K FY26, 10-Q Q1 FY27), reported HOKA brand growth (wholesale +2.9%, DTC +17.3%, total HOKA +7.7% in Q1 FY27), 6:1 split-adjusted capital structure, zero funded debt ($1.60B net cash), ASC 842 lease operating expense convention ($472.33M lease liabilities excluded from net debt bridge), corrected quarterly share measurement period (three months ended June 30, 2026), and disciplined reverse-DCF valuation.",
        "governing_standard": "ARX-QUALITY-WATCH-STANDARD-V1",
        "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1"
    }

    ulta_sha256 = compute_canonical_dossier_digest(accepted_ulta)
    medp_sha256 = compute_canonical_dossier_digest(accepted_medp)
    isrg_sha256 = compute_canonical_dossier_digest(accepted_isrg)
    deck_sha256 = compute_canonical_dossier_digest(accepted_deck)

    manifest = {
        "schema_version": "ARX-PUBLISHED-WATCH-MANIFEST-V1",
        "task": "ARX-TERMINAL-CURATED-QUALITY-WATCH",
        "governing_standard": "ARX-QUALITY-WATCH-STANDARD-V1",
        "standard_doc_path": "docs/research/opportunity_discovery/QUALITY_WATCH_SELECTION_STANDARD.md",
        "created_at_utc": AUTHORING_TIMESTAMP_UTC,
        "market_session_date": MARKET_SESSION_DATE,
        "filing_availability_cutoff_utc": FILING_AVAILABILITY_CUTOFF_UTC,
        "published_symbols": PUBLISHED_WATCH_SYMBOLS,
        "published_count": len(PUBLISHED_WATCH_SYMBOLS),
        "baseline_watch_symbols": BASELINE_WATCH_COHORT,
        "held_historical_symbols": HELD_HISTORICAL_SYMBOLS,
        "excluded_historical_symbols": EXCLUDED_HISTORICAL_SYMBOLS,
        "excluded_draft_symbols": EXCLUDED_DRAFT_SYMBOLS,
        "remaining_expansion_queue": [
            "KLAC", "LULU", "EME", "VRT", "ANET", "CDNS", "PWR", "SNPS", "NOW", "ELF"
        ],
        "queue_policy": "Under ARX Quality Watch Standard, the 10 remaining candidates are queued for subsequent assessment with zero fixed cohort quota or publication target. Only candidates supported by verified primary evidence and meeting the Quality Watch Standard will be published.",
        "accepted_additions": {
            "ULTA": accepted_ulta,
            "MEDP": accepted_medp,
            "ISRG": accepted_isrg,
            "DECK": accepted_deck,
        },
        "additions_provenance": {
            "ULTA": {
                "company_name": "Ulta Beauty, Inc.",
                "cik": "0001403568",
                "exchange": "NASDAQ",
                "acceptance_basis": "REUSED_ACCEPTED_RESEARCH_EVIDENCE",
                "selection_decision": "Selected as First Published Addition under ARX-QUALITY-WATCH-STANDARD-V1",
                "content_sha256": ulta_sha256,
                "authoring_timestamp_utc": "2026-10-21T16:00:00Z",
                "filing_availability_cutoff_utc": "2026-10-16T20:00:00Z",
                "lifecycle_status": "CANDIDATE_SELECTED",
                "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
                "source_authority": "ARX Quality Watch Scaling Addition",
                "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
                "source_filings": [f["accession_number"] for f in accepted_ulta["financial_profile"]["primary_filings"]],
                "reference_price": accepted_ulta["valuation"]["reference_price"],
                "reference_date": accepted_ulta["valuation"]["reference_date"],
                "solved_5y_cagr_pct": accepted_ulta["valuation"]["solved_5y_cagr_pct"],
                "selection_rationale": accepted_ulta.get("source_evidence_tracing", {}).get("selection_rationale")
            },
            "MEDP": {
                "company_name": "Medpace Holdings, Inc.",
                "cik": "0001668397",
                "exchange": "NASDAQ",
                "acceptance_basis": "PRIMARY_SEC_EDGAR_VERIFIED_RESEARCH",
                "selection_decision": "Selected as Scaling Addition under ARX-QUALITY-WATCH-STANDARD-V1",
                "content_sha256": medp_sha256,
                "authoring_timestamp_utc": "2026-10-22T08:00:00Z",
                "filing_availability_cutoff_utc": "2026-10-16T20:00:00Z",
                "lifecycle_status": "CANDIDATE_SELECTED",
                "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
                "source_authority": "ARX Quality Watch Scaling Addition",
                "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
                "source_filings": [f["accession_number"] for f in accepted_medp["financial_profile"]["primary_filings"]],
                "reference_price": accepted_medp["valuation"]["reference_price"],
                "reference_date": accepted_medp["valuation"]["reference_date"],
                "solved_5y_cagr_pct": accepted_medp["valuation"]["solved_5y_cagr_pct"],
                "selection_rationale": accepted_medp.get("source_evidence_tracing", {}).get("selection_rationale")
            },
            "ISRG": {
                "company_name": "Intuitive Surgical, Inc.",
                "cik": "0001035267",
                "exchange": "NASDAQ",
                "acceptance_basis": "PRIMARY_SEC_EDGAR_VERIFIED_RESEARCH",
                "selection_decision": "Selected as Scaling Addition under ARX-QUALITY-WATCH-STANDARD-V1",
                "content_sha256": isrg_sha256,
                "authoring_timestamp_utc": "2026-10-22T08:00:00Z",
                "filing_availability_cutoff_utc": "2026-10-16T20:00:00Z",
                "lifecycle_status": "CANDIDATE_SELECTED",
                "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
                "source_authority": "ARX Quality Watch Scaling Addition",
                "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
                "source_filings": [f["accession_number"] for f in accepted_isrg["financial_profile"]["primary_filings"]],
                "reference_price": accepted_isrg["valuation"]["reference_price"],
                "reference_date": accepted_isrg["valuation"]["reference_date"],
                "solved_5y_cagr_pct": accepted_isrg["valuation"]["solved_5y_cagr_pct"],
                "selection_rationale": accepted_isrg.get("source_evidence_tracing", {}).get("selection_rationale")
            },
            "DECK": {
                "company_name": "Deckers Outdoor Corp.",
                "cik": "0000910521",
                "exchange": "NYSE",
                "acceptance_basis": "PRIMARY_SEC_EDGAR_VERIFIED_RESEARCH",
                "selection_decision": "Selected as Scaling Addition under ARX-QUALITY-WATCH-STANDARD-V1",
                "content_sha256": deck_sha256,
                "authoring_timestamp_utc": "2026-10-22T08:00:00Z",
                "filing_availability_cutoff_utc": "2026-10-16T20:00:00Z",
                "lifecycle_status": "CANDIDATE_SELECTED",
                "review_status": "RESEARCH_EVIDENCE_ACCEPTED",
                "source_authority": "ARX Quality Watch Scaling Addition",
                "governing_contract": "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
                "source_filings": [f["accession_number"] for f in accepted_deck["financial_profile"]["primary_filings"]],
                "reference_price": accepted_deck["valuation"]["reference_price"],
                "reference_date": accepted_deck["valuation"]["reference_date"],
                "solved_5y_cagr_pct": accepted_deck["valuation"]["solved_5y_cagr_pct"],
                "selection_rationale": accepted_deck.get("source_evidence_tracing", {}).get("selection_rationale")
            }
        }
    }
    return manifest


def generate_cohort_matrix_markdown(ledger: Dict[str, Any]) -> str:
    """Generates deterministic pilot scaling cohort matrix documentation from ledger."""
    dossiers = ledger.get("dossiers", {})

    rows: List[str] = []
    # Baseline 1..5
    for idx, sym in enumerate(BASELINE_COHORT, start=1):
        d = dossiers[sym]
        ident = d["identity"]
        fp = d["financial_profile"]
        val = d["valuation"]
        ca = d["central_assessment"]
        filings = ", ".join(f"{f['form']} ({f['period_end']})" for f in fp["primary_filings"])
        ref_p = f"${val['reference_price']:,.2f}"
        val_status_map = {
            "COMPUTED_VERIFIED": "Computed & Verified",
            "COMPUTED_QUALIFIED": "Computed with Qualification",
            "VALUATION_SUPPRESSED": "Valuation Suppressed",
        }
        val_elig = val_status_map.get(val["valuation_status"], val["valuation_status"])
        sector_ind = BASELINE_DESCRIPTIONS[sym]["sector_industry"]
        biz_model = BASELINE_DESCRIPTIONS[sym]["business_model"]
        row = f"| **Baseline {idx}** | `{sym}` | {ident['company_name']} | `{ident['cik']}` | {ident['exchange']} | {sector_ind} | {biz_model} | {filings} | {ref_p} | {val_elig} | `{ca['disposition']}` |"
        rows.append(row)

    # Addition 1..14
    for idx, sym in enumerate(PILOT_ADDITIONS_COHORT, start=1):
        d = dossiers[sym]
        ident = d["identity"]
        fp = d["financial_profile"]
        val = d["valuation"]
        ca = d["central_assessment"]
        filings = ", ".join(f"{f['form']} ({f['period_end']})" for f in fp["primary_filings"])
        ref_p = f"${val['reference_price']:,.2f}"
        val_status_map = {
            "COMPUTED_VERIFIED": "Computed & Verified",
            "COMPUTED_QUALIFIED": "Computed with Qualification",
            "VALUATION_SUPPRESSED": "Valuation Suppressed",
        }
        val_elig = val_status_map.get(val["valuation_status"], val["valuation_status"])
        spec = PILOT_SPECS[sym]
        sector_ind = spec.get("sector_industry", ident["opportunity_route"])
        biz_model = spec.get("business_model", "")
        row = f"| **Addition {idx}** | `{sym}` | {ident['company_name']} | `{ident['cik']}` | {ident['exchange']} | {sector_ind} | {biz_model} | {filings} | {ref_p} | {val_elig} | `{ca['disposition']}` |"
        rows.append(row)

    table_body = "\n".join(rows)

    content = f"""# ARX TERMINAL — OPPORTUNITY DOSSIER SCALING PILOT COHORT & ELIGIBILITY MATRIX

**Document Version:** 1.0.0-draft  
**Date:** 2026-10-21  
**Governing Contract:** CONTRACT-ARX-OPP-PILOT-SCALING-V1  
**Pricing Observation Session:** 2026-10-16 (Regular Close)  
**Target Universe:** 19 Companies Total (5 Baseline Cohort + 14 Pilot Expansion Additions)

---

## 1. Executive Summary

This document establishes the authoritative cohort selection, business model classification, filing accessibility, and valuation eligibility for the **Opportunity Dossier Scaling Pilot**.

The pilot scales the accepted five-company Proof of Concept (`FIX`, `DUOL`, `IREN`, `CORZ`, `MU`) into a repeatable, auditable research preparation pipeline targeting 19 operating companies.

### Core Selection Principles:
1. **Evidence Accessibility:** All companies must be SEC reporting issuers with accessible Forms 10-K and 10-Q filed via EDGAR.
2. **Pricing Provenance:** Market pricing is anchored strictly to verified regular close quotes from the frozen `2026-10-16` market session tape.
3. **Valuation Method Appropriateness:** Companies must operate within business structures where operating income (EBIT), net debt bridges, and free cash flows to the firm (FCFF) or equity (FCFE) can be mathematically modeled or where valuation suppression is formally justified.
4. **Structural Exclusions:** Financial institutions (banks), insurers, real estate investment trusts (REITs), and pre-revenue clinical biotechs are excluded from standard FCFF models due to structural balance sheet inapplicability.
5. **Clear Separation:** Pilot draft records are maintained in isolated staging ledgers and do not modify the frozen ODP-05 baseline (`{FROZEN_ODP05_SHA256}`) or production snapshot.

---

## 2. Comprehensive 19-Company Pilot Cohort

| # | Ticker | Company Name | CIK | Primary Exchange | Sector & Industry | Business Classification | Primary Filings | 2026-10-16 Ref Price | Valuation Eligibility | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|
{table_body}

---

## 3. Structural Business Sector Groupings

The 19 companies in the scaling pilot are organized into 5 coherent, economically rigorous sectors:

### Sector 1: Specialty Contractors & Critical Infrastructure
- **Companies:** `FIX`, `EME`, `VRT`, `PWR`
- **Economic Model:** Modular HVAC, thermal management, liquid cooling, electrical switchgear, and utility grid interconnection. Driven by hyperscale data center electrification and semiconductor fab buildouts.
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. Strong cash conversion; working capital swings moderated by milestone prepayments.
- **Lease / Debt Conventions:** Standard funded senior debt subtracted; unrestricted cash added to EV.

### Sector 2: Cloud, Networking & Workflow Software
- **Companies:** `DUOL`, `ANET`, `NOW`
- **Economic Model:** Subscription recurring revenue, enterprise cloud platforms, AI workflow automation, and high-throughput data center switching.
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. High gross margins (65%-85%).
- **Qualification Factors:** Stock-based compensation (SBC) dilution must be explicitly monitored.

### Sector 3: Semiconductor Equipment & EDA Silicon Design
- **Companies:** `MU`, `KLAC`, `CDNS`, `SNPS`
- **Economic Model:** Semiconductor process yield control, silicon design verification, and memory hardware manufacturing.
- **Valuation Treatment:** Reverse-DCF FCFF.
- **Qualification Factors:** `MU` is marked `COMPUTED_QUALIFIED` and `NO_SUPPORTED_OPPORTUNITY` due to extreme memory cycle peaks/troughs; `SNPS` qualified by pending debt-funded acquisition structures (Ansys).

### Sector 4: High-Return Healthcare & Clinical Medical Technology
- **Companies:** `MEDP`, `ISRG`
- **Economic Model:** Capital-light clinical trial execution with 20.97% operating margin, positive operating cash generation, and zero funded debt (`MEDP`) and recurring razor-and-blade robotic surgical procedures with 85% recurring revenue (`ISRG`).
- **Valuation Treatment:** Single-stage Reverse-DCF FCFF. Pristine net cash balance sheets, high operating profitability, positive operating cash generation, and zero funded debt.

### Sector 5: Capital-Light Quality Consumer Brands
- **Companies:** `ULTA`, `DECK`, `LULU`, `ELF`
- **Economic Model:** Premium footwear, cosmetics retail, and athletic performance apparel.
- **Valuation Treatment:** Reverse-DCF FCFF under strict **Lease-as-Operating-Expense Convention** (ASC 842). Store operating leases are expensed in operating income and paid in CFO; lease liabilities are excluded from the net debt bridge to prevent double deduction.
- **Capital Structure Reconciliations:** Form 10-Q Note 7 revolving credit borrowings ($339.6M) and cash/ST investments ($213.5M) form ULTA net debt bridge ($126.1M) deducted from Enterprise Value. `DECK` audited post 6:1 forward stock split with reported Q1 FY27 brand/channel growth (wholesale +2.9%, DTC +17.3%, total HOKA +7.7%).

---

## 4. Documented Exclusions & Rejected Candidates

The following candidate business models were evaluated and rejected from this pilot cohort due to structural method incompatibility:

| Rejected Ticker | Organization | Category | Reason for Rejection | Justified Alternative Method Required |
|---|---|---|---|---|
| `BAC` / `JPM` | Bank of America / JPMorgan Chase | Commercial & Investment Banking | **Balance Sheet Debt is Operating Raw Material:** Banks fund earning assets (loans, securities) with liabilities (deposits, repo). Interest expense is an operating cost, not financing. Enterprise Value and FCFF models are mathematically undefined. | Excess Returns Model (ERM) or Dividend Discount Model (DDM) tied to Tier 1 Common Equity (CET1). |
| `PGR` / `CB` | Progressive / Chubb | Property & Casualty Insurance | **Float and Reserve Dynamics:** Operating cash flow is dominated by premium float timing and statutory loss reserves rather than operating margins. | Return on Equity (ROE) versus Price-to-Tangible Book Value (P/TBV) with combined ratio underwriting audits. |
| `PLD` / `EQIX` | Prologis / Equinix | Real Estate Investment Trusts (REITs) | **Depreciation Distortion:** Real estate GAAP net income and operating margins are severely distorted by multi-decade non-cash real property depreciation. | Funds From Operations (FFO) / Adjusted Funds From Operations (AFFO) or Net Asset Value (NAV) capitalization. |
| `ARWR` / `CRSP` | Arrowhead Pharma / CRISPR Therapeutics | Clinical-Stage Pre-Revenue Biotech | **Binary Clinical/Regulatory Pipeline Risk:** Operating cash flows are deeply negative due to R&D burn. Revenue CAGR has zero empirical anchor prior to FDA approval. | Probability-weighted Risk-Adjusted Net Present Value (rNPV) per clinical drug asset. |

---

## 5. Maintenance & Preparation Workflow Analysis

The Opportunity Dossier scaling workflow decouples automated specification solving from human qualitative research review across four auditable gates:

### Reusable Preparation Stages:
1. **Primary Sourced Disclosure Extraction:**
   - SEC EDGAR Forms 10-K and 10-Q extraction for income statement, balance sheet, and share count facts.
   - Tracing accessions, period dates, and XBRL tags to establish complete provenance.
2. **Capital Structure & Debt Bridge Auditing:**
   - Classification of debt-bearing liabilities vs documented zero-debt balance sheets.
   - Separation of ASC 842 operating leases (recognized in Cost of Sales and SG&A under lease-as-operating-expense convention) from funded financial debt to avoid double-deduction.
3. **Deterministic Reverse-DCF Solving & Scenarios:**
   - Execution of canonical `solve_reverse_dcf` and `run_fcff_model` engines against frozen market session regular close prices.
   - Verification of roundtrip model convergence within tight tolerance.
4. **Invariant Gate & Independent Research Peer Review:**
   - Deterministic verification of hash manifests, build isolation boundaries, and fail-closed state machines.
   - Final review sign-off by qualified research analyst before any promotion out of draft staging.

### Archetype Maintenance Protocols:
| Cohort Archetype | Representative Issuers | Primary Analytical Verification Focus | Maintenance Protocol |
|---|---|---|---|
| **Pristine Balance Sheet (Net Cash, Zero Debt)** | `MEDP`, `ANET`, `ISRG`, `DUOL`, `EME`, `DECK` | Credit facility note disclosures confirming zero funded borrowings and undrawn revolver status | Re-verify credit agreement covenants and zero debt disclosures upon each Form 10-Q filing. |
| **Retail ASC 842 Operating Leases** | `ULTA`, `LULU`, `ELF` | Lease expense allocation between Cost of Sales and SG&A; funded debt vs lease liability bridge reconciliation | Ensure store and supply chain lease commitments remain treated as operating expenses and excluded from net debt bridge. |
| **Semiconductor Equipment & EDA** | `KLAC`, `CDNS`, `SNPS`, `MU` | Mid-cycle margin sustainability, backlog conversion, and M&A debt financing structures | Model multi-year cyclicality bounds; qualify or suppress valuation when cyclical peak/troughs distort run-rate EBIT. |
| **Capital-Intensive Infrastructure** | `FIX`, `VRT`, `PWR`, `IREN`, `CORZ` | Hyperscale customer concentration, milestone payment timing, working capital swings, and capital structure stability | Enforce valuation suppression when ongoing restructuring or hyper-dilutive financing makes FCFF unanchored. |

---

## 6. Expansion Queue & Publication Selection Policy

Under the ARX Quality Watch Standard (`ARX-QUALITY-WATCH-STANDARD-V1`), research focuses exclusively on qualifying `WATCH` candidates supported by primary source evidence. There is no artificial cohort size cap or publication quota.

### Published Quality Watch Collection:
- **Baseline Watch:** `FIX`, `DUOL`
- **Selected Expansion Additions:** `ULTA`, `MEDP`, `ISRG`, `DECK`

### Held Outside Active Publication:
- `IREN`, `CORZ` (preserved in baseline ledger; neutral research state)

### Excluded from Active Research:
- `MU` (cyclical semiconductor commodity peak), `PLTR` (extreme valuation multiple requirement)

### Remaining Expansion Queue (For Subsequent Assessment):
The following 10 candidate business models remain queued for subsequent assessment and will be published only if and when supported by primary filing verification and meeting Quality Watch criteria:
- `KLAC` (KLA Corporation — Semiconductor Process Control)
- `LULU` (lululemon athletica inc. — Technical Athletic Apparel Retail)
- `EME` (EMCOR Group, Inc. — Electrical & Mechanical Construction Infrastructure)
- `VRT` (Vertiv Holdings Co — AI Data Center Liquid Cooling & Power Distribution)
- `ANET` (Arista Networks, Inc. — Data Center Cloud Networking)
- `CDNS` (Cadence Design Systems, Inc. — Electronic Design Automation)
- `PWR` (Quanta Services, Inc. — Electric Power Grid & Renewable Infrastructure)
- `SNPS` (Synopsys, Inc. — Semiconductor IP & EDA Software)
- `NOW` (ServiceNow, Inc. — Enterprise Digital Workflow Automation)
- `ELF` (e.l.f. Beauty, Inc. — Clean Color Cosmetics & Skincare)
"""
    return content.strip() + "\n"


def write_pilot_artifacts(
    ledger: Dict[str, Any],
    manifest: Dict[str, Any],
    frontend_draft: Dict[str, Any],
    ledger_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    frontend_draft_path: Optional[Path] = None,
    doc_matrix_path: Optional[Path] = None,
    published_manifest_path: Optional[Path] = None,
    frontend_published_manifest_path: Optional[Path] = None,
) -> None:
    """Writes the pilot scaling ledger, manifest, frontend staging file, cohort documentation matrix, and published watch manifest."""
    target_ledger = ledger_path or DEFAULT_PILOT_LEDGER_PATH
    target_manifest = manifest_path or DEFAULT_PILOT_MANIFEST_PATH
    target_frontend = frontend_draft_path or DEFAULT_FRONTEND_DRAFT_PATH
    target_doc = doc_matrix_path or DEFAULT_DOC_MATRIX_PATH
    target_published = published_manifest_path or DEFAULT_PUBLISHED_MANIFEST_PATH
    target_fe_published = frontend_published_manifest_path or DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH

    # Prevent accidental write to production publication paths
    frozen_prod_snapshot = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
    if target_frontend.resolve() == frozen_prod_snapshot.resolve():
        raise ValueError("Cannot write pilot draft to production snapshot: opportunityDossierSnapshot.json")
    if target_ledger.resolve() == FROZEN_ODP05_LEDGER_PATH.resolve():
        raise ValueError("Cannot write pilot draft to canonical baseline: odp05_discovery_ledger.json")

    os.makedirs(target_ledger.parent, exist_ok=True)
    os.makedirs(target_frontend.parent, exist_ok=True)
    os.makedirs(target_manifest.parent, exist_ok=True)
    os.makedirs(target_doc.parent, exist_ok=True)
    os.makedirs(target_published.parent, exist_ok=True)
    os.makedirs(target_fe_published.parent, exist_ok=True)

    with open(target_ledger, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    logger.info(f"Wrote pilot scaling ledger to: {target_ledger}")

    # Compute actual on-disk ledger sha256 for manifest
    with open(target_ledger, "rb") as f:
        ledger_sha = hashlib.sha256(f.read()).hexdigest()
    manifest["pilot_ledger_sha256"] = ledger_sha

    with open(target_frontend, "w", encoding="utf-8") as f:
        json.dump(frontend_draft, f, indent=2)
    logger.info(f"Wrote frontend draft staging snapshot to: {target_frontend}")

    with open(target_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote pilot manifest to: {target_manifest}")

    doc_content = generate_cohort_matrix_markdown(ledger)
    with open(target_doc, "w", encoding="utf-8") as f:
        f.write(doc_content)
    logger.info(f"Wrote synchronized cohort matrix doc to: {target_doc}")

    # Authoritative published watch manifest (selective publication)
    published_manifest = generate_published_watch_manifest(ledger["dossiers"])
    with open(target_published, "w", encoding="utf-8") as f:
        json.dump(published_manifest, f, indent=2)
    logger.info(f"Wrote published watch manifest to: {target_published}")

    with open(target_fe_published, "w", encoding="utf-8") as f:
        json.dump(published_manifest, f, indent=2)
    logger.info(f"Wrote frontend published watch manifest to: {target_fe_published}")


def find_structural_differences(expected: Any, actual: Any, path: str = "$") -> List[str]:
    """
    Recursively compares two JSON-deserialized structures for exact structural equality.
    Returns a list of concise strings describing all differences (type mismatch,
    missing keys, unexpected extra keys, array length/ordering differences, value mismatches).
    Preserves array order and does not normalize or ignore any fields.
    """
    diffs = []
    if type(expected) != type(actual):
        diffs.append(f"{path}: type mismatch (expected {type(expected).__name__}, got {type(actual).__name__})")
        return diffs

    if isinstance(expected, dict):
        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())
        for missing in sorted(exp_keys - act_keys):
            diffs.append(f"{path}.{missing}: missing field in actual")
        for extra in sorted(act_keys - exp_keys):
            diffs.append(f"{path}.{extra}: unexpected extra field in actual")
        for k in sorted(exp_keys & act_keys):
            diffs.extend(find_structural_differences(expected[k], actual[k], f"{path}.{k}"))
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length mismatch (expected {len(expected)}, got {len(actual)})")
        min_len = min(len(expected), len(actual))
        for idx in range(min_len):
            diffs.extend(find_structural_differences(expected[idx], actual[idx], f"{path}[{idx}]"))
    else:
        if expected != actual:
            diffs.append(f"{path}: value mismatch (expected {expected!r}, got {actual!r})")

    return diffs


def validate_existing_pilot_artifacts(
    ledger_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    frontend_draft_path: Optional[Path] = None,
    doc_matrix_path: Optional[Path] = None,
    published_manifest_path: Optional[Path] = None,
    frontend_published_manifest_path: Optional[Path] = None,
) -> bool:
    """
    Validates that existing pilot artifacts exist, match recorded hashes, pass all invariants,
    and match in-memory deterministic regeneration from authoritative specifications.
    Read-only: does not modify or repair any files.
    """
    target_ledger = ledger_path or DEFAULT_PILOT_LEDGER_PATH
    target_manifest = manifest_path or DEFAULT_PILOT_MANIFEST_PATH
    target_frontend = frontend_draft_path or DEFAULT_FRONTEND_DRAFT_PATH
    target_doc = doc_matrix_path or DEFAULT_DOC_MATRIX_PATH
    target_published = published_manifest_path or DEFAULT_PUBLISHED_MANIFEST_PATH
    target_fe_published = frontend_published_manifest_path or DEFAULT_FRONTEND_PUBLISHED_MANIFEST_PATH

    if not target_ledger.exists():
        logger.error(f"Pilot ledger not found: {target_ledger}")
        return False
    if not target_manifest.exists():
        logger.error(f"Pilot manifest not found: {target_manifest}")
        return False
    if not target_frontend.exists():
        logger.error(f"Frontend draft snapshot not found: {target_frontend}")
        return False
    if not target_doc.exists():
        logger.error(f"Cohort matrix documentation not found: {target_doc}")
        return False
    if not target_published.exists():
        logger.error(f"Published watch manifest not found: {target_published}")
        return False
    if not target_fe_published.exists():
        logger.error(f"Frontend published watch manifest not found: {target_fe_published}")
        return False

    # 1. Verify frozen production baseline hash
    if not FROZEN_ODP05_LEDGER_PATH.exists():
        logger.error(f"Frozen ODP-05 baseline ledger missing at: {FROZEN_ODP05_LEDGER_PATH}")
        return False
    with open(FROZEN_ODP05_LEDGER_PATH, "rb") as f:
        actual_baseline_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_baseline_sha != FROZEN_ODP05_SHA256:
        logger.error(f"Frozen ODP-05 baseline SHA-256 corrupted: {actual_baseline_sha} != {FROZEN_ODP05_SHA256}")
        return False

    with open(target_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(target_ledger, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(target_frontend, "r", encoding="utf-8") as f:
        frontend_draft = json.load(f)

    # 2. Manifest hash verification
    with open(target_ledger, "rb") as f:
        actual_ledger_sha = hashlib.sha256(f.read()).hexdigest()
    expected_ledger_sha = manifest.get("pilot_ledger_sha256")
    if actual_ledger_sha != expected_ledger_sha:
        logger.error(f"Manifest ledger SHA mismatch: recorded={expected_ledger_sha}, actual={actual_ledger_sha}")
        return False

    # 3. Total cohort count and symbols
    dossiers = ledger.get("dossiers", {})
    if len(dossiers) != len(TOTAL_PILOT_COHORT):
        logger.error(f"Expected {len(TOTAL_PILOT_COHORT)} dossiers in pilot ledger, found {len(dossiers)}")
        return False

    for sym in TOTAL_PILOT_COHORT:
        if sym not in dossiers:
            logger.error(f"Symbol {sym} missing from pilot ledger")
            return False

    # 4. Review status integrity
    for sym in BASELINE_COHORT:
        meta = dossiers[sym].get("publication_metadata", {})
        if meta.get("lifecycle_status") != "PUBLISHED_PRODUCTION":
            logger.error(f"Baseline {sym} lifecycle status invalid: {meta.get('lifecycle_status')}")
            return False
        if meta.get("review_status") != "RESEARCH_REVIEW_UNRECORDED":
            logger.error(f"Baseline {sym} review status must be RESEARCH_REVIEW_UNRECORDED: {meta.get('review_status')}")
            return False
        if meta.get("reviewer_id") is not None or meta.get("reviewed_at_utc") is not None:
            logger.error(f"Baseline {sym} contains invented reviewer sign-off: {meta}")
            return False

    for sym in PILOT_ADDITIONS_COHORT:
        d = dossiers[sym]
        meta = d.get("publication_metadata", {})
        if meta.get("lifecycle_status") != "DRAFT_STAGING":
            logger.error(f"Draft addition {sym} lifecycle must be DRAFT_STAGING: {meta.get('lifecycle_status')}")
            return False
        if meta.get("review_status") != "PENDING_RESEARCH_REVIEW":
            logger.error(f"Draft addition {sym} review status must be PENDING_RESEARCH_REVIEW: {meta.get('review_status')}")
            return False
        if meta.get("reviewer_id") is not None or meta.get("reviewed_at_utc") is not None:
            logger.error(f"Draft addition {sym} contains unauthorized reviewer metadata: {meta}")
            return False

    # 5. Evidence mapping verification for accepted additions (ULTA, MEDP, ISRG, DECK)
    for sym in ["ULTA", "MEDP", "ISRG", "DECK"]:
        d = dossiers.get(sym, {})
        d_fin = d.get("financial_profile", {})
        d_tracing = d_fin.get("source_evidence_tracing") or d.get("source_evidence_tracing")
        if not d_tracing:
            logger.error(f"Addition candidate {sym} missing mandatory source_evidence_tracing block")
            return False
        required_tracing_keys = [
            "company_identity", "revenue", "operating_income", "cash_and_investments",
            "funded_debt", "lease_treatment", "shares_outstanding", "reference_price",
            "valuation_model", "narrative_claims", "selection_rationale"
        ]
        for rk in required_tracing_keys:
            if rk not in d_tracing:
                logger.error(f"Candidate {sym} missing required tracing key: {rk}")
                return False
            item = d_tracing[rk]
            if isinstance(item, dict) and "fact_status" in item:
                status = item.get("fact_status")
                if status not in ("VERIFIED_REPORTED_FACT", "DERIVED_CALCULATION", "ANALYST_ASSUMPTION", "UNVERIFIED_CLAIM"):
                    logger.error(f"Invalid fact_status in {sym} tracing {rk}: {status}")
                    return False

        # Selection rationale completeness (questions A-F)
        sel_rat = d_tracing.get("selection_rationale", {})
        for q_key in ["why_this_business", "why_research_now", "price_requirements", "differentiated_view", "assessment_changes", "queue_priority"]:
            if not sel_rat.get(q_key):
                logger.error(f"Candidate {sym} missing selection rationale question: {q_key}")
                return False

    # Specific checks for MEDP, ISRG, DECK share measurement period
    for exp_sym in ["MEDP", "ISRG", "DECK"]:
        ed = dossiers[exp_sym]
        ed_fin = ed["financial_profile"]
        ed_tracing = ed_fin["source_evidence_tracing"]
        expected_period = "Three months ended June 30, 2026"
        if ed_fin.get("selected_share_basis", {}).get("measurement_period") != expected_period:
            logger.error(f"{exp_sym} selected_share_basis measurement_period must be '{expected_period}', got '{ed_fin.get('selected_share_basis', {}).get('measurement_period')}'")
            return False
        if ed_tracing["shares_outstanding"].get("measurement_period") != expected_period:
            logger.error(f"{exp_sym} tracing shares_outstanding measurement_period must be '{expected_period}'")
            return False

    # DECK specific growth narrative verification
    deck_d = dossiers["DECK"]
    deck_supp = deck_d["central_assessment"]["strongest_supporting_evidence"]
    if "market share gains (>25% YoY)" in deck_supp:
        logger.error("DECK strongest_supporting_evidence must not make unsupported market share claims")
        return False
    if "17.3%" not in deck_supp or "2.9%" not in deck_supp or "703.5M" not in deck_supp or "1,019.5M" not in deck_supp:
        logger.error("DECK strongest_supporting_evidence must state exact reported Q1 FY27 brand/channel growth, $703.5M HOKA sales, and $1,019.5M net sales")
        return False
    if any(stale in deck_supp for stale in ["545.0M", "151.0M", "394.0M", "825.3M"]):
        logger.error("DECK strongest_supporting_evidence must not contain stale or unverified dollar amounts")
        return False

    # ISRG specific recurring revenue verification
    isrg_d = dossiers["ISRG"]
    isrg_sum = isrg_d["central_assessment"]["presentation_summary"]
    isrg_supp = isrg_d["central_assessment"]["strongest_supporting_evidence"]
    if "unassailable" in isrg_sum.lower() or "unassailable" in isrg_supp.lower():
        logger.error("ISRG narratives must not use hyperbolic language ('unassailable')")
        return False
    if "85%" not in isrg_supp and "85%" not in isrg_sum:
        logger.error("ISRG must accurately state 85% recurring revenue definition")
        return False

    # MEDP specific operating profitability and cash flow verification (unsupported ROIC removed)
    medp_d = dossiers["MEDP"]
    medp_supp = medp_d["central_assessment"]["strongest_supporting_evidence"]
    if "20.97%" not in medp_supp or "$502.7M" not in medp_supp:
        logger.error("MEDP strongest_supporting_evidence must state verified 20.97% operating margin and $502.7M cash")
        return False
    if "ROIC" in medp_supp or "75.3%" in medp_supp or "41.3%" in medp_supp:
        logger.error("MEDP strongest_supporting_evidence must not contain unsupported ROIC claims")
        return False

    ulta = dossiers.get("ULTA", {})
    ulta_fin = ulta.get("financial_profile", {})
    ulta_tracing = ulta_fin.get("source_evidence_tracing") or ulta.get("source_evidence_tracing")

    # Explicit verification of ULTA Note 7 funded debt and accounting
    if ulta_fin.get("is_zero_debt") is not False:
        logger.error("ULTA is_zero_debt must be False due to Note 7 revolving credit borrowings")
        return False
    if ulta_fin.get("total_financial_debt") != 339578000.0:
        logger.error(f"ULTA total_financial_debt must be $339,578,000.0: {ulta_fin.get('total_financial_debt')}")
        return False
    if ulta_fin.get("net_debt_bridge") != 126127000.0:
        logger.error(f"ULTA net_debt_bridge must be $126,127,000.0: {ulta_fin.get('net_debt_bridge')}")
        return False
    if ulta_tracing["funded_debt"]["is_zero_debt"] is not False:
        logger.error("ULTA source tracing funded_debt is_zero_debt must be False")
        return False
    if ulta_tracing["funded_debt"]["total_financial_debt_usd"] != 339578000.0:
        logger.error("ULTA source tracing total_financial_debt_usd must be 339578000.0")
        return False
    if ulta_tracing["lease_treatment"]["total_lease_liabilities_usd"] != 2184453000.0:
        logger.error("ULTA source tracing total_lease_liabilities_usd must be 2184453000.0")
        return False
    if ulta_tracing["shares_outstanding"].get("measurement_period") != "Three months ended 2026-08-01":
        logger.error("ULTA source tracing shares_outstanding measurement_period must be 'Three months ended 2026-08-01'")
        return False
    if ulta_fin.get("selected_share_basis", {}).get("measurement_period") != "Three months ended 2026-08-01":
        logger.error("ULTA selected_share_basis measurement_period must be 'Three months ended 2026-08-01'")
        return False
    if ulta_tracing["valuation_model"]["solved_5y_revenue_cagr_pct"] != 15.45:
        logger.error(f"ULTA source tracing solved_5y_revenue_cagr_pct must be 15.45%: {ulta_tracing['valuation_model']['solved_5y_revenue_cagr_pct']}")
        return False
    if ulta.get("valuation", {}).get("solved_5y_cagr_pct") != 15.45:
        logger.error(f"ULTA valuation solved_5y_cagr_pct must be 15.45: {ulta.get('valuation', {}).get('solved_5y_cagr_pct')}")
        return False

    # 6. In-memory deterministic recomputation and tampering detection
    recomputed_ledger, _, recomputed_frontend = prepare_pilot_scaling_corpus()
    recomputed_dossiers = recomputed_ledger["dossiers"]

    for sym in TOTAL_PILOT_COHORT:
        stored_d = dossiers[sym]
        rec_d = recomputed_dossiers[sym]

        # Financial profile comparison
        stored_fin = stored_d.get("financial_profile", {})
        rec_fin = rec_d.get("financial_profile", {})
        for f_key in ["revenue", "operating_income", "cash_and_equivalents", "total_financial_debt", "net_debt_bridge", "is_zero_debt"]:
            if stored_fin.get(f_key) != rec_fin.get(f_key):
                logger.error(f"Financial profile tampering detected for {sym}.{f_key}: stored={stored_fin.get(f_key)}, expected={rec_fin.get(f_key)}")
                return False

        # Valuation comparison
        stored_val = stored_d.get("valuation", {})
        rec_val = rec_d.get("valuation", {})
        for v_key in ["reference_price", "solved_5y_cagr_pct", "valuation_status", "roundtrip_value", "roundtrip_delta_cents"]:
            if stored_val.get(v_key) != rec_val.get(v_key):
                logger.error(f"Valuation tampering detected for {sym}.{v_key}: stored={stored_val.get(v_key)}, expected={rec_val.get(v_key)}")
                return False

        # Central assessment narrative comparison
        stored_ca = stored_d.get("central_assessment", {})
        rec_ca = rec_d.get("central_assessment", {})
        for ca_key in ["disposition", "presentation_summary", "strongest_supporting_evidence", "strongest_opposing_evidence", "main_uncertainty", "price_watch_condition", "confidence_assessment"]:
            if stored_ca.get(ca_key) != rec_ca.get(ca_key):
                logger.error(f"Central assessment tampering detected for {sym}.{ca_key}")
                return False

    # 7. Complete Deterministic Frontend Draft Projection Parity
    fe_diffs = find_structural_differences(recomputed_frontend, frontend_draft, path="$")
    if fe_diffs:
        for diff in fe_diffs[:10]:
            logger.error(f"Frontend draft projection mismatch at {diff}")
        if len(fe_diffs) > 10:
            logger.error(f"... and {len(fe_diffs) - 10} more frontend draft projection differences")
        return False

    # 8. Documentation Freshness Check
    with open(target_doc, "r", encoding="utf-8") as f:
        actual_doc = f.read()
    expected_doc = generate_cohort_matrix_markdown(recomputed_ledger)
    if actual_doc != expected_doc:
        logger.error(f"Documentation matrix {target_doc} is out of sync with authoritative ledger")
        return False

    # 9. Published Watch Manifest Verification
    if not target_published.exists():
        logger.error(f"Published watch manifest not found: {target_published}")
        return False
    if not target_fe_published.exists():
        logger.error(f"Frontend published watch manifest not found: {target_fe_published}")
        return False

    with open(target_published, "r", encoding="utf-8") as f:
        actual_pub_manifest = json.load(f)
    with open(target_fe_published, "r", encoding="utf-8") as f:
        actual_fe_pub_manifest = json.load(f)

    expected_pub_manifest = generate_published_watch_manifest(recomputed_dossiers)
    pub_diffs = find_structural_differences(expected_pub_manifest, actual_pub_manifest, path="$.published_manifest")
    if pub_diffs:
        for diff in pub_diffs[:5]:
            logger.error(f"Published watch manifest mismatch at {diff}")
        return False

    fe_diffs = find_structural_differences(expected_pub_manifest, actual_fe_pub_manifest, path="$.fe_published_manifest")
    if fe_diffs:
        for diff in fe_diffs[:5]:
            logger.error(f"Frontend published watch manifest mismatch at {diff}")
        return False

    logger.info("All pilot scaling and published watch artifact invariants verified successfully (read-only audit passed).")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ARX Terminal Opportunity Dossier Scaling Pilot Preparer & Validator"
    )
    parser.add_argument("--symbol", type=str, help="Prepare and inspect a single pilot addition symbol (e.g. ULTA)")
    parser.add_argument("--all-pilot", action="store_true", help="Prepare all pilot cohort dossiers, publish watch manifest, and write outputs")
    parser.add_argument("--validate", action="store_true", help="Validate existing pilot scaling artifacts and invariants")
    parser.add_argument("--check", action="store_true", help="CI check mode: validates existing artifacts, exits 1 on error")
    parser.add_argument("--dry-run", action="store_true", help="Perform preparation and validation in-memory without modifying disk files")
    parser.add_argument("--ledger-path", type=str, help="Explicit destination or read path for pilot scaling ledger")
    parser.add_argument("--manifest-path", type=str, help="Explicit destination or read path for pilot scaling manifest")
    parser.add_argument("--out", type=str, help="Explicit destination path for frontend draft snapshot JSON")
    parser.add_argument("--doc-matrix-path", type=str, help="Explicit destination or read path for cohort matrix documentation")
    parser.add_argument("--published-manifest-path", type=str, help="Explicit destination or read path for published watch manifest")
    parser.add_argument("--fe-published-manifest-path", type=str, help="Explicit destination or read path for frontend published watch manifest")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ledger_path = Path(args.ledger_path) if args.ledger_path else None
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    frontend_path = Path(args.out) if args.out else None
    doc_matrix_path = Path(args.doc_matrix_path) if args.doc_matrix_path else None
    published_manifest_path = Path(args.published_manifest_path) if args.published_manifest_path else None
    frontend_published_manifest_path = Path(args.fe_published_manifest_path) if args.fe_published_manifest_path else None

    # Safety boundary: prevent output flag from targeting production snapshot
    if frontend_path:
        frozen_prod = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
        if frontend_path.resolve() == frozen_prod.resolve():
            logger.error("Forbidden: --out cannot target production snapshot (opportunityDossierSnapshot.json).")
            sys.exit(1)

    if args.validate or args.check:
        ok = validate_existing_pilot_artifacts(
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            frontend_draft_path=frontend_path,
            doc_matrix_path=doc_matrix_path,
            published_manifest_path=published_manifest_path,
            frontend_published_manifest_path=frontend_published_manifest_path,
        )
        if not ok:
            sys.exit(1)
        print("PASS: Pilot scaling artifacts verified.")
        return

    if args.dry_run:
        logger.info("Executing pilot preparation and validation in DRY-RUN mode (zero disk mutations)...")
        ledger, manifest, frontend_draft = prepare_pilot_scaling_corpus(single_symbol=args.symbol)
        dossiers = ledger["dossiers"]
        logger.info(f"In-memory prepared {len(dossiers)} dossiers.")
        # Validate in-memory outputs
        if "ULTA" in dossiers:
            ulta = dossiers["ULTA"]
            assert ulta["financial_profile"]["source_evidence_tracing"] is not None
            assert ulta["valuation"]["reference_price"] == 592.94
        print(f"PASS (DRY RUN): Prepared {len(dossiers)} dossiers in-memory. Zero disk modifications made.")
        return

    if args.symbol:
        sym = args.symbol.strip().upper()
        if sym not in PILOT_SPECS:
            logger.error(f"Symbol {sym} is not in pilot addition specifications: {list(PILOT_SPECS.keys())}")
            sys.exit(1)
        logger.info(f"Preparing single pilot addition draft for: {sym}")
        ledger, manifest, frontend_draft = prepare_pilot_scaling_corpus(single_symbol=sym)
        dossier = ledger["dossiers"][sym]
        print(f"\n--- Opportunity Dossier Draft: {sym} ---")
        print(f"Company: {dossier['identity']['company_name']} (CIK {dossier['identity']['cik']})")
        print(f"Disposition: {dossier['central_assessment']['disposition_label']}")
        print(f"Valuation: {dossier['valuation']['valuation_status_label']}")
        print(f"Reference Price: ${dossier['valuation']['reference_price']:.2f}")
        print(f"Solved 5Y Revenue CAGR: {dossier['valuation']['solved_5y_cagr_pct']}%")
        print(f"Roundtrip Per Share: ${dossier['valuation']['roundtrip_value']:.2f} (delta: {dossier['valuation']['roundtrip_delta_cents']} cents)")
        print(f"Presentation Summary:\n  {dossier['central_assessment']['presentation_summary']}\n")
        return

    if args.all_pilot:
        logger.info("Preparing complete 19-company Opportunity Dossier pilot cohort and published watch manifest...")
        ledger, manifest, frontend_draft = prepare_pilot_scaling_corpus()
        write_pilot_artifacts(
            ledger,
            manifest,
            frontend_draft,
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            frontend_draft_path=frontend_path,
            doc_matrix_path=doc_matrix_path,
            published_manifest_path=published_manifest_path,
            frontend_published_manifest_path=frontend_published_manifest_path,
        )
        ok = validate_existing_pilot_artifacts(
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            frontend_draft_path=frontend_path,
            doc_matrix_path=doc_matrix_path,
            published_manifest_path=published_manifest_path,
            frontend_published_manifest_path=frontend_published_manifest_path,
        )
        if not ok:
            logger.error("Generated pilot artifacts failed validation gate.")
            sys.exit(1)
        print("SUCCESS: Opportunity Dossier pilot scaling corpus and curated Quality Watch manifest prepared and validated.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
