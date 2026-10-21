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

PILOT_ADDITIONS_COHORT = [
    "ULTA", "MEDP", "DECK", "KLAC", "LULU",
    "EME", "VRT", "ANET", "PLTR", "ISRG",
    "CDNS", "PWR", "SNPS", "NOW", "ELF"
]

TOTAL_PILOT_COHORT = BASELINE_COHORT + PILOT_ADDITIONS_COHORT

DEFAULT_PILOT_LEDGER_PATH = REPO_ROOT / "docs/research/opportunity_discovery/pilot_scaling_ledger.json"
DEFAULT_PILOT_MANIFEST_PATH = REPO_ROOT / "docs/research/opportunity_discovery/pilot_scaling_manifest.json"
DEFAULT_FRONTEND_DRAFT_PATH = REPO_ROOT / "frontend/lib/opportunityDossierPilotDrafts.json"

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
                "total_financial_debt_usd": 0.0,
                "is_zero_debt": True,
                "support_disclosure": "No senior notes or term loans outstanding; $1.0B revolving credit facility completely undrawn at quarter end (Note 8).",
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Balance Sheets & Note 8"
                }
            },
            "lease_treatment": {
                "fact_status": "VERIFIED_REPORTED_FACT",
                "current_operating_lease_liabilities_usd": 312648000.0,
                "noncurrent_operating_lease_liabilities_usd": 1871805000.0,
                "total_lease_liabilities_usd": 2184453000.0,
                "accounting_standard": "ASC 842",
                "valuation_convention": "Lease-as-Operating-Expense convention: retail store lease expense deducted in operating expenses; excluded from debt bridge to avoid double deduction.",
                "xbrl_tags": [
                    "us-gaap:OperatingLeaseLiabilityCurrent",
                    "us-gaap:OperatingLeaseLiabilityNoncurrent"
                ],
                "filing_source": {
                    "form": "10-Q",
                    "period_end": "2026-08-01",
                    "filing_date": "2026-08-27",
                    "accession": "0001104659-26-102442",
                    "location": "Part I Item 1 Consolidated Balance Sheets"
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
                "solved_5y_revenue_cagr_pct": 15.11,
                "assumptions": {
                    "fact_status": "ANALYST_ASSUMPTION",
                    "operating_margin_pct": 12.0,
                    "wacc_pct": 9.0,
                    "terminal_growth_pct": 2.0,
                    "tax_rate_pct": 23.5,
                    "reinvestment_rate_pct": 15.0
                },
                "roundtrip_value": 592.94,
                "roundtrip_delta_cents": 0.45
            },
            "narrative_claims": [
                {
                    "claim_type": "SUPPORTING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Ulta Beauty has over 44 million active loyalty program members, representing over 95% of total sales.",
                    "locator": "Form 10-K for FY ended 2026-01-31, Item 1 Business — Loyalty Program"
                },
                {
                    "claim_type": "OPPOSING_EVIDENCE",
                    "fact_status": "VERIFIED_REPORTED_FACT",
                    "statement": "Prestige category growth has moderated while promotional cadence across prestige beauty and competitive footprint expansion (Sephora at Kohl's) has intensified.",
                    "locator": "Form 10-Q for Q2 ended 2026-08-01, Part I Item 2 MD&A — Operating Results"
                },
                {
                    "claim_type": "MONITORING_CRITERIA",
                    "fact_status": "ANALYST_ASSUMPTION",
                    "statement": "Operating margin compression below 10.0% or negative comparable store sales for two consecutive fiscal quarters triggers thesis invalidation.",
                    "locator": "ARX Analytical Thesis Invalidation Rule"
                }
            ]
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
            "total_financial_debt": 0.0,
            "is_zero_debt": True,
            "zero_debt_support": "Zero funded interest-bearing senior debt under credit facility; revolving facility fully undrawn.",
            "operating_lease_liabilities_asc842": 1871805000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied under ASC 842; store rent deducted in SG&A; lease liabilities excluded from net debt bridge to avoid double deduction.",
            "net_debt_bridge": -213451000.0,
            "bridge_note": "Total cash and investments ($213.5M) less zero funded debt leaves $213.5M net cash bridge added to enterprise value in DCF equity bridge.",
            "diluted_shares": 43062000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 43062000.0,
                "measurement_period": "TTM ended 2026-08-01",
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
            "strongest_supporting_evidence": "Ulta generates dependable operating cash flows ($1.57B TTM CFO) with a pristine balance sheet ($213.5M cash, documented zero funded debt) and 44M+ loyalty members driving >95% of sales.",
            "strongest_opposing_evidence": "Prestige beauty competitive intensity from Sephora at Kohl's and Amazon Beauty has compressed operating margins from 15.0% peak to 12.5%. At $592.94, market prices in 15.11% revenue CAGR.",
            "main_uncertainty": "Durability of store-level traffic and margin defense against expanding promotional cadence across prestige cosmetic brands.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($430 - $460) or evidence of sustained comp-store sales acceleration >6%.",
            "confidence_assessment": "High confidence in accounting and zero-debt capital structure; medium confidence in long-term terminal operating margin floor.",
            "financing_dilution_exposure": "Negative net dilution; aggressive share buybacks have retired over 13.6% of outstanding shares over the trailing three fiscal years.",
            "observable_invalidation_trigger": "Operating margin compression below 10.0% or negative comparable store sales for two consecutive fiscal quarters.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (expected December 2026) for holiday promotional cadence and loyalty membership growth.",
            "presentation_summary": "Ulta Beauty demonstrates high capital returns and a pristine zero-debt balance sheet ($213.5M liquidity) supported by 44M+ loyalty members. However, the reference price of $592.94 embeds a demanding 15.11% 5-year revenue CAGR against intense prestige retail competition. Retained on watch for pullbacks toward $430 - $460."
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
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-18",
                "accession_number": "0001668397-26-000012",
                "doc_name": "medp-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-22",
                "accession_number": "0001668397-26-000048",
                "doc_name": "medp-20260630x10q.htm",
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
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count from Q2 10-Q."
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
            "strongest_supporting_evidence": "Medpace delivers industry-leading capital efficiency (>25% ROIC) and operating margins (>20%), insulated by proprietary full-service clinical execution and $502.7M net cash.",
            "strongest_opposing_evidence": "Biopharma venture funding cycles and small-cap clinical trial cancellation rates create volatility in net book-to-bill ratios. At $618.47, market embeds a demanding 29.41% revenue CAGR.",
            "main_uncertainty": "Sustained clinical trial contract backlog conversion amidst fluctuating early-stage biotech liquidity.",
            "price_watch_condition": "Pullback toward mid-cycle base valuation range ($440 - $480) or net new business awards sustaining book-to-bill > 1.25x.",
            "confidence_assessment": "High confidence in operational execution and balance sheet conservatism; moderate valuation headroom.",
            "financing_dilution_exposure": "Share repurchases offset stock compensation, reducing share count by ~9.6% over 3 years.",
            "observable_invalidation_trigger": "Net new business awards book-to-bill ratio declining below 1.0x for consecutive quarters or operating margin breach below 16.0%.",
            "next_monitoring_trigger": "Q3 2026 earnings release and Form 10-Q for trial backlog cancellations and RFP volume.",
            "presentation_summary": "Medpace Holdings delivers pristine capital returns (>25% ROIC) and zero-debt balance sheet strength ($502.7M cash). However, the frozen market price of $618.47 requires an aggressive 29.41% 5-year revenue CAGR. Retained on watch pending market pullbacks toward $440 - $480."
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
        "filings": [
            {
                "form": "10-K",
                "period_end": "2026-03-31",
                "filing_date": "2026-05-22",
                "accession_number": "0000910521-26-000028",
                "doc_name": "deck-20260331x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-25",
                "accession_number": "0000910521-26-000045",
                "doc_name": "deck-20260630x10q.htm",
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
            "operating_lease_liabilities_asc842": 248900000.0,
            "lease_convention_treatment": "Lease as operating expense convention applied; store and distribution leases expensed in operating income; lease liabilities excluded from net debt bridge.",
            "net_debt_bridge": -1602589000.0,
            "bridge_note": "Total cash of $1.60B against zero funded debt yields $1.60B net cash bridge added to enterprise value in DCF equity bridge.",
            "diluted_shares": 138559000.0,
            "selected_share_basis": {
                "basis_type": "SPLIT_ADJUSTED_DILUTED",
                "value": 138559000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Post-split diluted share count reflecting the 6-for-1 forward split executed in September 2024."
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
            "strongest_supporting_evidence": "HOKA brand momentum continues rapid global wholesale and DTC market share gains (>25% YoY), complemented by $1.60B cash reserves and zero debt.",
            "strongest_opposing_evidence": "High seasonal concentration in UGG winter demand and elevated wholesale inventory across athletic channels. Solved 5Y CAGR is -2.34%, indicating market reflects mature apparel normalization.",
            "main_uncertainty": "Sustainability of premium DTC full-price sell-through rates as HOKA expands into broader lifestyle footwear categories.",
            "price_watch_condition": "Confirmation of international DTC growth above 20% while maintaining consolidated gross margin above 55.0%.",
            "confidence_assessment": "High confidence in brand equity and net cash asset backing; moderate concern regarding cyclical consumer discretionary headwinds.",
            "financing_dilution_exposure": "Clean share capital structure; forward 6:1 split successfully resolved historical comparability issues with zero phantom dilution.",
            "observable_invalidation_trigger": "Gross margin decline below 50.0% or HOKA DTC quarterly revenue deceleration below 10.0%.",
            "next_monitoring_trigger": "Q2 FY27 earnings release (expected October/November 2026) for holiday pre-orders and inventory aging.",
            "presentation_summary": "Deckers Outdoor displays pristine financial health ($1.60B net cash, zero debt) propelled by HOKA's expansion. Market pricing of $85.72 embeds conservative -2.34% implied growth due to wholesale normalization. Retained on watch for international DTC confirmation."
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
        "filings": [
            {
                "form": "10-K",
                "period_end": "2025-12-31",
                "filing_date": "2026-02-05",
                "accession_number": "0001035267-26-000018",
                "doc_name": "isrg-20251231x10k.htm",
                "source_role": "Annual audited financial disclosures"
            },
            {
                "form": "10-Q",
                "period_end": "2026-06-30",
                "filing_date": "2026-07-21",
                "accession_number": "0001035267-26-000045",
                "doc_name": "isrg-20260630x10q.htm",
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
            "operating_lease_liabilities_asc842": 110000000.0,
            "lease_convention_treatment": "Lease liabilities excluded from net debt bridge under standard lease convention.",
            "net_debt_bridge": -5216200000.0,
            "bridge_note": "Total cash and investments ($5.22B) against zero funded debt yields $5.22B net cash added to enterprise value in DCF equity bridge.",
            "diluted_shares": 357300000.0,
            "selected_share_basis": {
                "basis_type": "DILUTED_WEIGHTED_AVERAGE",
                "value": 357300000.0,
                "measurement_period": "TTM ended 2026-06-30",
                "rationale": "Diluted share count reported in Q2 10-Q."
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
            "strongest_supporting_evidence": "da Vinci 5 system replacement cycle accelerating installed base procedure growth (>17% YoY), supported by high recurring revenue (>80% razor-and-blade instruments/services) and $5.22B net cash.",
            "strongest_opposing_evidence": "Capital spending constraints across regional hospitals and international competitive entries. Solved reverse DCF requires 36.87% revenue CAGR at $426.70.",
            "main_uncertainty": "Hospital capital budget allocation timelines for multi-million dollar da Vinci 5 console upgrades.",
            "price_watch_condition": "Market pullback toward mid-cycle base valuation range ($280 - $310) or procedure growth acceleration above 20%.",
            "confidence_assessment": "High confidence in surgical procedure moat and recurring cash flow predictability; valuation currently fully priced.",
            "financing_dilution_exposure": "Minimal net dilution; steady share repurchases offset employee stock options.",
            "observable_invalidation_trigger": "Worldwide procedure growth deceleration below 10.0% or da Vinci 5 adoption stagnation.",
            "next_monitoring_trigger": "Q3 2026 Form 10-Q filing (October 2026) for procedure volume growth and system placement breakdown.",
            "presentation_summary": "Intuitive Surgical maintains an unassailable surgical robotics ecosystem (>80% recurring revenue) and $5.22B net cash. Market pricing of $426.70 embeds 36.87% 5Y CAGR expectations for the da Vinci 5 upgrade cycle. Retained on watch for pullbacks toward $280 - $310."
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
        },
        "source_evidence_tracing": spec.get("source_evidence_tracing"),
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


def write_pilot_artifacts(
    ledger: Dict[str, Any],
    manifest: Dict[str, Any],
    frontend_draft: Dict[str, Any],
    ledger_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    frontend_draft_path: Optional[Path] = None,
) -> None:
    """Writes the pilot scaling ledger, manifest, and frontend staging file."""
    target_ledger = ledger_path or DEFAULT_PILOT_LEDGER_PATH
    target_manifest = manifest_path or DEFAULT_PILOT_MANIFEST_PATH
    target_frontend = frontend_draft_path or DEFAULT_FRONTEND_DRAFT_PATH

    # Prevent accidental write to production publication paths
    frozen_prod_snapshot = REPO_ROOT / "frontend/lib/opportunityDossierSnapshot.json"
    if target_frontend.resolve() == frozen_prod_snapshot.resolve():
        raise ValueError("Cannot write pilot draft to production snapshot: opportunityDossierSnapshot.json")
    if target_ledger.resolve() == FROZEN_ODP05_LEDGER_PATH.resolve():
        raise ValueError("Cannot write pilot draft to canonical baseline: odp05_discovery_ledger.json")

    os.makedirs(target_ledger.parent, exist_ok=True)
    os.makedirs(target_frontend.parent, exist_ok=True)
    os.makedirs(target_manifest.parent, exist_ok=True)

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


def validate_existing_pilot_artifacts(
    ledger_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    frontend_draft_path: Optional[Path] = None,
) -> bool:
    """
    Validates that existing pilot artifacts exist, match recorded hashes, pass all invariants,
    and match in-memory deterministic regeneration from authoritative specifications.
    Read-only: does not modify or repair any files.
    """
    target_ledger = ledger_path or DEFAULT_PILOT_LEDGER_PATH
    target_manifest = manifest_path or DEFAULT_PILOT_MANIFEST_PATH
    target_frontend = frontend_draft_path or DEFAULT_FRONTEND_DRAFT_PATH

    if not target_ledger.exists():
        logger.error(f"Pilot ledger not found: {target_ledger}")
        return False
    if not target_manifest.exists():
        logger.error(f"Pilot manifest not found: {target_manifest}")
        return False
    if not target_frontend.exists():
        logger.error(f"Frontend draft snapshot not found: {target_frontend}")
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
    if len(dossiers) != 20:
        logger.error(f"Expected 20 dossiers in pilot ledger, found {len(dossiers)}")
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

    # 5. Evidence mapping verification for representative dossier (ULTA)
    ulta = dossiers.get("ULTA", {})
    ulta_fin = ulta.get("financial_profile", {})
    ulta_tracing = ulta_fin.get("source_evidence_tracing") or ulta.get("source_evidence_tracing")
    if not ulta_tracing:
        logger.error("Representative candidate ULTA missing mandatory source_evidence_tracing block")
        return False
    required_tracing_keys = [
        "company_identity", "revenue", "operating_income", "cash_and_investments",
        "funded_debt", "lease_treatment", "shares_outstanding", "reference_price",
        "valuation_model", "narrative_claims"
    ]
    for rk in required_tracing_keys:
        if rk not in ulta_tracing:
            logger.error(f"Representative candidate ULTA missing required tracing key: {rk}")
            return False
        item = ulta_tracing[rk]
        if isinstance(item, dict):
            status = item.get("fact_status")
            if status not in ("VERIFIED_REPORTED_FACT", "DERIVED_CALCULATION", "ANALYST_ASSUMPTION", "UNVERIFIED_CLAIM"):
                logger.error(f"Invalid fact_status in ULTA tracing {rk}: {status}")
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
        for f_key in ["revenue", "operating_income", "cash_and_equivalents", "total_financial_debt", "net_debt_bridge"]:
            if stored_fin.get(f_key) != rec_fin.get(f_key):
                logger.error(f"Financial profile tampering detected for {sym}.{f_key}: stored={stored_fin.get(f_key)}, expected={rec_fin.get(f_key)}")
                return False

        # Valuation comparison
        stored_val = stored_d.get("valuation", {})
        rec_val = rec_d.get("valuation", {})
        for v_key in ["reference_price", "solved_5y_cagr_pct", "valuation_status"]:
            if stored_val.get(v_key) != rec_val.get(v_key):
                logger.error(f"Valuation tampering detected for {sym}.{v_key}: stored={stored_val.get(v_key)}, expected={rec_val.get(v_key)}")
                return False

        # Central assessment narrative comparison
        stored_ca = stored_d.get("central_assessment", {})
        rec_ca = rec_d.get("central_assessment", {})
        for ca_key in ["disposition", "presentation_summary", "strongest_supporting_evidence", "strongest_opposing_evidence"]:
            if stored_ca.get(ca_key) != rec_ca.get(ca_key):
                logger.error(f"Central assessment tampering detected for {sym}.{ca_key}")
                return False

        # Frontend draft snapshot parity
        fe_d = frontend_draft.get("dossiers", {}).get(sym)
        if not fe_d:
            logger.error(f"Symbol {sym} missing from frontend draft snapshot")
            return False
        if fe_d.get("valuation", {}).get("reference_price") != stored_val.get("reference_price"):
            logger.error(f"Frontend draft snapshot out of sync with ledger for {sym}")
            return False

    logger.info("All pilot scaling artifact invariants verified successfully (read-only audit passed).")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ARX Terminal Opportunity Dossier Scaling Pilot Preparer & Validator"
    )
    parser.add_argument("--symbol", type=str, help="Prepare and inspect a single pilot addition symbol (e.g. ULTA)")
    parser.add_argument("--all-pilot", action="store_true", help="Prepare all 20 pilot cohort dossiers and write outputs")
    parser.add_argument("--validate", action="store_true", help="Validate existing pilot scaling artifacts and invariants")
    parser.add_argument("--check", action="store_true", help="CI check mode: validates existing artifacts, exits 1 on error")
    parser.add_argument("--dry-run", action="store_true", help="Perform preparation and validation in-memory without modifying disk files")
    parser.add_argument("--ledger-path", type=str, help="Explicit destination or read path for pilot scaling ledger")
    parser.add_argument("--manifest-path", type=str, help="Explicit destination or read path for pilot scaling manifest")
    parser.add_argument("--out", type=str, help="Explicit destination path for frontend draft snapshot JSON")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ledger_path = Path(args.ledger_path) if args.ledger_path else None
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    frontend_path = Path(args.out) if args.out else None

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
        logger.info("Preparing complete 20-company Opportunity Dossier pilot cohort...")
        ledger, manifest, frontend_draft = prepare_pilot_scaling_corpus()
        write_pilot_artifacts(
            ledger,
            manifest,
            frontend_draft,
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            frontend_draft_path=frontend_path,
        )
        ok = validate_existing_pilot_artifacts(
            ledger_path=ledger_path,
            manifest_path=manifest_path,
            frontend_draft_path=frontend_path,
        )
        if not ok:
            logger.error("Generated pilot artifacts failed validation gate.")
            sys.exit(1)
        print("SUCCESS: 20-company Opportunity Dossier pilot scaling corpus prepared and validated.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
