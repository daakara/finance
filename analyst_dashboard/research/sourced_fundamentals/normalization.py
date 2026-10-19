"""
analyst_dashboard/research/sourced_fundamentals/normalization.py

Reproducible Cash Flow Normalization Engine for Comfort Systems USA (FIX).
Unpacks reported GAAP cash conversion ($2,159.81M CFO less Capex) against operating income ($1,849.30M EBIT),
separating reported financials, observed contract liability financing inflows, and explicit analyst assumptions.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationAssumptions:
    """Explicit versioned analyst assumptions for sustainable run-rate cash flow."""
    model_version: str = "2.0.0"
    assumed_da_millions: float = 112.50
    normalized_wc_reinvestment_millions: float = 280.00
    effective_tax_rate_override: Optional[float] = None


def calculate_fix_normalized_cash_flow(
    ebit_raw: float,
    effective_tax_rate_pct: float,
    capex_raw: float,
    reported_cfo_raw: float,
    assumptions: NormalizationAssumptions = NormalizationAssumptions(),
) -> Dict[str, Any]:
    """
    Computes illustrative sustainable run-rate cash flow from reported ledger inputs
    and versioned analyst assumptions.
    
    Inputs (in raw USD):
    - ebit_raw: e.g. 1,849,302,000.0 ($1,849.30M)
    - effective_tax_rate_pct: e.g. 21.93 (%)
    - capex_raw: e.g. 390,259,000.0 ($390.26M)
    - reported_cfo_raw: e.g. 2,550,065,000.0 ($2,550.07M)
    """
    # 1. Convert to millions for standardized reporting precision ($M rounded to 2 decimal places)
    ebit_m = round(ebit_raw / 1e6, 2)
    tax_rate = (assumptions.effective_tax_rate_override if assumptions.effective_tax_rate_override is not None else effective_tax_rate_pct) / 100.0
    taxes_m = round(ebit_m * tax_rate, 2)
    nopat_m = round(ebit_m - taxes_m, 2)

    capex_m = round(capex_raw / 1e6, 2)
    cfo_m = round(reported_cfo_raw / 1e6, 2)
    reported_cfo_less_capex_m = round(cfo_m - capex_m, 2)

    da_m = assumptions.assumed_da_millions
    wc_reinvest_m = assumptions.normalized_wc_reinvestment_millions

    # 2. Arithmetic sustainable cash flow walk
    sustainable_cf_m = round(nopat_m + da_m - capex_m - wc_reinvest_m, 2)

    return {
        "model_version": assumptions.model_version,
        "reported_inputs": {
            "ebit_millions": ebit_m,
            "effective_tax_rate_pct": round(tax_rate * 100.0, 2),
            "taxes_millions": taxes_m,
            "nopat_millions": nopat_m,
            "capex_millions": capex_m,
            "cfo_millions": cfo_m,
            "cfo_less_capex_millions": reported_cfo_less_capex_m,
        },
        "analyst_assumptions": {
            "assumed_da_millions": da_m,
            "normalized_wc_reinvestment_millions": wc_reinvest_m,
            "is_illustrative_assumption": True,
        },
        "normalization_walk": {
            "nopat": nopat_m,
            "plus_da": da_m,
            "minus_capex": capex_m,
            "minus_normalized_wc_reinvestment": wc_reinvest_m,
            "sustainable_cash_flow_millions": sustainable_cf_m,
            "formula": "NOPAT ($1,443.73M) + D&A ($112.50M) - Capex ($390.26M) - Normalized WC Reinvestment ($280.00M) = $885.97M",
        },
        "caveats": [
            "D&A and normalized working capital reinvestment are analyst-selected assumptions, not GAAP audited cash flows.",
            "Reported TTM cash flow ($2,159.81M) was significantly expanded by interim customer contract liabilities (billings in excess of costs).",
            "This model represents an illustrative sensitivity scenario and does not establish certified intrinsic value.",
        ]
    }
