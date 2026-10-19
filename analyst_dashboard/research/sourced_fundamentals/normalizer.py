"""
analyst_dashboard/research/sourced_fundamentals/normalizer.py

Normalizes raw SEC XBRL company facts and submissions into structured,
provenance-tracked CompanyFundamentals instances. Enforces strict period
matching, non-overlapping TTM construction, and metric status attribution.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .contracts import (
    FinancialFact,
    CompanyFundamentals,
    MetricStatus,
)

logger = logging.getLogger(__name__)

# Standard US-GAAP concept priority cascades
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]

OPERATING_INCOME_CONCEPTS = [
    "OperatingIncomeLoss",
]

NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "ProfitLoss",
]

CFO_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]

CAPEX_CONCEPTS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
]

SBC_CONCEPTS = [
    "AllocatedShareBasedCompensationExpense",
    "ShareBasedCompensation",
]

CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
]

ST_INVESTMENT_CONCEPTS = [
    "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesDebtMaturitiesWithinOneYearAmortizedCost",
]

DEBT_CURRENT_CONCEPTS = [
    "DebtCurrent",
    "ShortTermBorrowings",
    "CommercialPaper",
    "LongTermDebtCurrent",
]

DEBT_NONCURRENT_CONCEPTS = [
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermDebt",
]

LEASE_LIABILITY_CONCEPTS = [
    "OperatingLeaseLiability",
    "OperatingLeaseLiabilityNoncurrent",
]

EQUITY_CONCEPTS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

TAX_EXPENSE_CONCEPTS = [
    "IncomeTaxExpenseBenefit",
]

PRETAX_INCOME_CONCEPTS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
]

DILUTED_SHARES_CONCEPTS = [
    "WeightedAverageNumberOfDilutedSharesOutstanding",
]

COMMON_SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
]


class SecFactNormalizer:
    """Normalizes raw XBRL facts and submissions for an individual company."""

    def __init__(self):
        pass

    @staticmethod
    def _filter_filings(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Restrict to official 10-K and 10-Q reports."""
        return [x for x in items if x.get("form") in ["10-K", "10-Q"]]

    def _get_best_concept_items(
        self,
        us_gaap: Dict[str, Any],
        concepts: List[str],
        unit_name: str = "USD",
        target_end: Optional[str] = None,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Find best matching concept. If target_end is provided, prioritizes
        the concept that actually has an observation on that date.
        """
        best_concept = None
        best_items: List[Dict[str, Any]] = []
        best_max_end = ""

        for c in concepts:
            if c in us_gaap and unit_name in us_gaap[c].get("units", {}):
                items = self._filter_filings(us_gaap[c]["units"][unit_name])
                if items:
                    if target_end and any(x.get("end") == target_end for x in items):
                        return c, items
                    max_end = max(x.get("end", "") for x in items)
                    if max_end > best_max_end:
                        best_max_end = max_end
                        best_concept = c
                        best_items = items

        return best_concept, best_items

    def _extract_instant_fact(
        self,
        us_gaap: Dict[str, Any],
        concepts: List[str],
        metric_key: str,
        target_end: str,
        unit_name: str = "USD",
        default_zero_if_absent: bool = False,
    ) -> FinancialFact:
        """Extract a single point-in-time balance sheet fact matching target_end."""
        c, items = self._get_best_concept_items(us_gaap, concepts, unit_name, target_end)
        if not c or not items:
            val = 0.0 if default_zero_if_absent else None
            status = MetricStatus.REPORTED if default_zero_if_absent else MetricStatus.UNAVAILABLE
            return FinancialFact(
                metric_key=metric_key,
                value=val,
                units=unit_name,
                status=status,
                period_end=target_end,
                notes=f"No reported concept found among {concepts}" + (" (assumed 0)" if default_zero_if_absent else "")
            )

        matching = [x for x in items if x.get("end") == target_end]
        if not matching:
            # Check within 95 days if exact match not found (e.g. for common shares cover page)
            val = 0.0 if default_zero_if_absent else None
            status = MetricStatus.REPORTED if default_zero_if_absent else MetricStatus.UNAVAILABLE
            return FinancialFact(
                metric_key=metric_key,
                value=val,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=status,
                period_end=target_end,
                notes=f"Concept found but no instant observation for period {target_end}" + (" (assumed 0)" if default_zero_if_absent else "")
            )

        best_item = sorted(matching, key=lambda x: x.get("filed", ""))[-1]
        return FinancialFact(
            metric_key=metric_key,
            value=float(best_item["val"]),
            units=unit_name,
            concept=f"us-gaap/{c}",
            accession_number=best_item.get("accn"),
            form=best_item.get("form"),
            period_end=best_item.get("end"),
            filing_date=best_item.get("filed"),
            retrieval_time=datetime.utcnow().isoformat() + "Z",
            status=MetricStatus.REPORTED,
            notes=f"Directly reported in {best_item.get('form')} filed {best_item.get('filed')}"
        )

    def _extract_ttm_fact(
        self,
        us_gaap: Dict[str, Any],
        concepts: List[str],
        metric_key: str,
        latest_form: str,
        latest_fp: str,
        latest_end: str,
        unit_name: str = "USD",
        force_positive: bool = False,
    ) -> FinancialFact:
        """
        Extract non-overlapping Trailing Twelve Months (TTM) duration fact.
        - If latest filing is 10-K (FP='FY'): TTM is that full fiscal year.
        - If latest filing is 10-Q (FP in ['Q1', 'Q2', 'Q3']):
            TTM = Latest_FY + Current_YTD - Prior_Comparable_YTD.
        """
        c, items = self._get_best_concept_items(us_gaap, concepts, unit_name, latest_end)
        if not c or not items:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_end,
                notes=f"Concept not found among {concepts}"
            )

        valid = self._filter_filings(items)

        # Case 1: Latest is full fiscal year (10-K)
        if latest_form == "10-K" or latest_fp == "FY":
            matching = [x for x in valid if x.get("form") == "10-K" and x.get("fp") == "FY" and x.get("end") == latest_end]
            if matching:
                item = sorted(matching, key=lambda x: x.get("filed", ""))[-1]
                val = float(item["val"])
                if force_positive:
                    val = abs(val)
                return FinancialFact(
                    metric_key=metric_key,
                    value=val,
                    units=unit_name,
                    concept=f"us-gaap/{c}",
                    accession_number=item.get("accn"),
                    form="10-K",
                    period_start=item.get("start"),
                    period_end=item.get("end"),
                    filing_date=item.get("filed"),
                    retrieval_time=datetime.utcnow().isoformat() + "Z",
                    status=MetricStatus.REPORTED,
                    formula=f"Reported 10-K FY ended {item.get('end')}",
                    notes=f"Full fiscal year 10-K filed on {item.get('filed')}"
                )
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_end,
                notes=f"No 10-K FY item matching period end {latest_end}"
            )

        # Case 2: Latest is interim quarter (10-Q)
        curr_matching = [x for x in valid if x.get("end") == latest_end and x.get("form") == "10-Q"]
        if not curr_matching:
            curr_matching = [x for x in valid if x.get("end") == latest_end]
        if not curr_matching:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_end,
                notes=f"No interim item matching {latest_end}"
            )

        curr_item = sorted(curr_matching, key=lambda x: x.get("filed", ""))[-1]

        # Prior fiscal year full 10-K
        fy_matching = [x for x in valid if x.get("form") == "10-K" and x.get("fp") == "FY" and x.get("end") < latest_end]
        if not fy_matching:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_end,
                notes=f"No prior full-year 10-K found prior to {latest_end}"
            )

        fy_item = sorted(fy_matching, key=lambda x: x.get("end", ""))[-1]

        # Prior comparable YTD in 10-Q
        prior_ytd_matching = [
            x for x in valid
            if x.get("fp") == latest_fp
            and x.get("form") == "10-Q"
            and x.get("end") < latest_end
            and x.get("end") > fy_item.get("start", "")
        ]
        if not prior_ytd_matching:
            target_year = int(latest_end[:4]) - 1
            prior_ytd_matching = [
                x for x in valid
                if x.get("fp") == latest_fp
                and x.get("end", "").startswith(str(target_year))
            ]

        if not prior_ytd_matching:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_end,
                notes=f"Could not locate prior comparable {latest_fp} YTD item to construct non-overlapping TTM"
            )

        prior_ytd_item = sorted(prior_ytd_matching, key=lambda x: x.get("filed", ""))[-1]

        val_fy = float(fy_item["val"])
        val_curr = float(curr_item["val"])
        val_prior = float(prior_ytd_item["val"])

        ttm_val = val_fy + val_curr - val_prior
        if force_positive:
            ttm_val = abs(ttm_val)

        formula_str = (
            f"Latest_FY{fy_item.get('fy')} ({val_fy:,.0f}) + "
            f"Current_{latest_fp}_YTD ({val_curr:,.0f}) - "
            f"Prior_{latest_fp}_YTD ({val_prior:,.0f})"
        )

        return FinancialFact(
            metric_key=metric_key,
            value=ttm_val,
            units=unit_name,
            concept=f"us-gaap/{c}",
            accession_number=curr_item.get("accn"),
            form="10-Q/10-K_TTM",
            period_start=fy_item.get("start"),
            period_end=curr_item.get("end"),
            filing_date=curr_item.get("filed"),
            retrieval_time=datetime.utcnow().isoformat() + "Z",
            status=MetricStatus.DERIVED,
            formula=formula_str,
            input_references=[
                f"{fy_item.get('form')}:{fy_item.get('accn')}",
                f"{curr_item.get('form')}:{curr_item.get('accn')}",
                f"{prior_ytd_item.get('form')}:{prior_ytd_item.get('accn')}",
            ],
            notes="Non-overlapping TTM calculated using authoritative SEC identity: FY + Current YTD - Prior YTD"
        )

    def normalize(
        self,
        symbol: str,
        cik: str,
        facts_dict: Dict[str, Any],
        submissions_dict: Dict[str, Any],
        sector_industry: str = "",
        business_model: str = "",
    ) -> CompanyFundamentals:
        """Full normalization entry point for a company."""
        company_name = facts_dict.get("entityName", symbol)
        us_gaap = facts_dict.get("facts", {}).get("us-gaap", {})

        # Identify latest 10-K and 10-Q from submissions
        recent = submissions_dict.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])

        latest_form = None
        latest_filing_date = None
        latest_report_date = None
        for f, fd, rd in zip(forms, filing_dates, report_dates):
            if f in ["10-K", "10-Q"]:
                latest_form = f
                latest_filing_date = fd
                latest_report_date = rd
                break

        # Fallback anchor detection from us_gaap
        c_anchor, items_anchor = self._get_best_concept_items(
            us_gaap, REVENUE_CONCEPTS + NET_INCOME_CONCEPTS, target_end=latest_report_date
        )
        if not items_anchor:
            raise ValueError(f"No valid revenue or net income observations found for {symbol} (CIK {cik})")

        latest_anchor = sorted(items_anchor, key=lambda x: (x.get("end", ""), x.get("filed", "")))[-1]
        latest_period_end = latest_anchor.get("end")
        latest_fp = latest_anchor.get("fp")
        latest_fy = latest_anchor.get("fy")

        facts: Dict[str, FinancialFact] = {}
        data_gaps: List[str] = []

        # 1. Revenue TTM
        rev_fact = self._extract_ttm_fact(
            us_gaap, REVENUE_CONCEPTS, "revenue_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["revenue_ttm"] = rev_fact
        if rev_fact.value is None:
            data_gaps.append("revenue_ttm unavailable")

        # 2. Operating Income (EBIT) TTM
        ebit_fact = self._extract_ttm_fact(
            us_gaap, OPERATING_INCOME_CONCEPTS, "operating_income_ttm", latest_form, latest_fp, latest_period_end
        )
        # KLAC special handling: KLAC reports Revenues, Cost of Revenue, R&D, and SG&A separately without OperatingIncomeLoss tag
        if ebit_fact.value is None and symbol == "KLAC":
            cost_fact = self._extract_ttm_fact(us_gaap, ["CostOfRevenue"], "cost_of_revenue_ttm", latest_form, latest_fp, latest_period_end)
            rd_fact = self._extract_ttm_fact(us_gaap, ["ResearchAndDevelopmentExpense"], "rd_expense_ttm", latest_form, latest_fp, latest_period_end)
            sga_fact = self._extract_ttm_fact(us_gaap, ["SellingGeneralAndAdministrativeExpense"], "sga_expense_ttm", latest_form, latest_fp, latest_period_end)
            if rev_fact.value and cost_fact.value and rd_fact.value and sga_fact.value:
                derived_ebit = rev_fact.value - cost_fact.value - rd_fact.value - sga_fact.value
                ebit_fact = FinancialFact(
                    metric_key="operating_income_ttm",
                    value=derived_ebit,
                    units="USD",
                    concept="DERIVED(Revenues - CostOfRevenue - R&D - SG&A)",
                    period_end=latest_period_end,
                    filing_date=latest_filing_date,
                    status=MetricStatus.DERIVED,
                    formula=f"Revenues ({rev_fact.value:,.0f}) - CostOfRevenue ({cost_fact.value:,.0f}) - R&D ({rd_fact.value:,.0f}) - SG&A ({sga_fact.value:,.0f})",
                    input_references=["revenue_ttm", "cost_of_revenue_ttm", "rd_expense_ttm", "sga_expense_ttm"],
                    notes="Derived from reported revenue and operating expense components in 10-K"
                )
        facts["operating_income_ttm"] = ebit_fact
        if ebit_fact.value is None:
            data_gaps.append("operating_income_ttm unavailable")

        # 3. Net Income TTM
        ni_fact = self._extract_ttm_fact(
            us_gaap, NET_INCOME_CONCEPTS, "net_income_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["net_income_ttm"] = ni_fact
        if ni_fact.value is None:
            data_gaps.append("net_income_ttm unavailable")

        # 4. CFO TTM
        cfo_fact = self._extract_ttm_fact(
            us_gaap, CFO_CONCEPTS, "cfo_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["cfo_ttm"] = cfo_fact
        if cfo_fact.value is None:
            data_gaps.append("cfo_ttm unavailable")

        # 5. Capex TTM
        capex_fact = self._extract_ttm_fact(
            us_gaap, CAPEX_CONCEPTS, "capex_ttm", latest_form, latest_fp, latest_period_end, force_positive=True
        )
        facts["capex_ttm"] = capex_fact
        if capex_fact.value is None:
            data_gaps.append("capex_ttm unavailable")

        # 6. Non-GAAP Free Cash Flow: CFO minus Capex
        if cfo_fact.value is not None and capex_fact.value is not None:
            fcf_val = cfo_fact.value - capex_fact.value
            facts["cfo_minus_capex_ttm"] = FinancialFact(
                metric_key="cfo_minus_capex_ttm",
                value=fcf_val,
                units="USD",
                status=MetricStatus.DERIVED,
                period_end=latest_period_end,
                filing_date=latest_filing_date,
                formula=f"CFO ({cfo_fact.value:,.0f}) - Capex ({capex_fact.value:,.0f})",
                input_references=["cfo_ttm", "capex_ttm"],
                notes="Non-GAAP derived cash flow measure. Defined strictly as CFO minus Capex."
            )
        else:
            facts["cfo_minus_capex_ttm"] = FinancialFact(
                metric_key="cfo_minus_capex_ttm",
                value=None,
                units="USD",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes="Cannot calculate CFO minus Capex due to missing CFO or Capex"
            )
            data_gaps.append("cfo_minus_capex_ttm unavailable")

        # 7. Stock-Based Compensation (SBC) TTM
        sbc_fact = self._extract_ttm_fact(
            us_gaap, SBC_CONCEPTS, "sbc_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["sbc_ttm"] = sbc_fact

        # 8. Cash & Equivalents (Instant)
        cash_fact = self._extract_instant_fact(
            us_gaap, CASH_CONCEPTS, "cash_and_equivalents", latest_period_end
        )
        facts["cash_and_equivalents"] = cash_fact

        # 9. Short-Term Investments (Instant)
        st_inv_fact = self._extract_instant_fact(
            us_gaap, ST_INVESTMENT_CONCEPTS, "short_term_investments", latest_period_end, default_zero_if_absent=True
        )
        facts["short_term_investments"] = st_inv_fact

        # Total Cash & Investments
        cash_val = cash_fact.value or 0.0
        st_inv_val = st_inv_fact.value or 0.0
        facts["total_cash_and_investments"] = FinancialFact(
            metric_key="total_cash_and_investments",
            value=cash_val + st_inv_val,
            units="USD",
            status=MetricStatus.DERIVED,
            period_end=latest_period_end,
            formula=f"Cash ({cash_val:,.0f}) + Short-Term Investments ({st_inv_val:,.0f})",
            input_references=["cash_and_equivalents", "short_term_investments"],
            notes="Total liquid cash and marketable investment securities"
        )

        # 10. Debt Current & Noncurrent (Instants)
        dcur_fact = self._extract_instant_fact(
            us_gaap, DEBT_CURRENT_CONCEPTS, "short_term_debt", latest_period_end, default_zero_if_absent=True
        )
        facts["short_term_debt"] = dcur_fact

        dnon_fact = self._extract_instant_fact(
            us_gaap, DEBT_NONCURRENT_CONCEPTS, "long_term_debt", latest_period_end, default_zero_if_absent=True
        )
        facts["long_term_debt"] = dnon_fact

        tot_debt = (dcur_fact.value or 0.0) + (dnon_fact.value or 0.0)
        facts["total_financial_debt"] = FinancialFact(
            metric_key="total_financial_debt",
            value=tot_debt,
            units="USD",
            status=MetricStatus.DERIVED,
            period_end=latest_period_end,
            formula=f"Short-Term Debt ({dcur_fact.value or 0:,.0f}) + Long-Term Debt ({dnon_fact.value or 0:,.0f})",
            input_references=["short_term_debt", "long_term_debt"],
            notes="Sum of current and noncurrent financial debt obligations"
        )

        facts["net_debt"] = FinancialFact(
            metric_key="net_debt",
            value=tot_debt - (cash_val + st_inv_val),
            units="USD",
            status=MetricStatus.DERIVED,
            period_end=latest_period_end,
            formula=f"Total Debt ({tot_debt:,.0f}) - Total Cash ({cash_val + st_inv_val:,.0f})",
            input_references=["total_financial_debt", "total_cash_and_investments"],
            notes="Net financial debt (negative indicates net cash balance)"
        )

        # 11. Operating Lease Liabilities
        lease_fact = self._extract_instant_fact(
            us_gaap, LEASE_LIABILITY_CONCEPTS, "operating_lease_liabilities", latest_period_end, default_zero_if_absent=True
        )
        facts["operating_lease_liabilities"] = lease_fact

        # 12. Stockholders' Equity
        equity_fact = self._extract_instant_fact(
            us_gaap, EQUITY_CONCEPTS, "stockholders_equity", latest_period_end
        )
        facts["stockholders_equity"] = equity_fact

        # 13. Income Tax & Pretax Income for Effective Tax Rate
        tax_fact = self._extract_ttm_fact(
            us_gaap, TAX_EXPENSE_CONCEPTS, "tax_expense_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["tax_expense_ttm"] = tax_fact

        pretax_fact = self._extract_ttm_fact(
            us_gaap, PRETAX_INCOME_CONCEPTS, "pretax_income_ttm", latest_form, latest_fp, latest_period_end
        )
        facts["pretax_income_ttm"] = pretax_fact

        # 14. Diluted & Common Shares
        dshares_fact = self._extract_instant_fact(
            us_gaap, DILUTED_SHARES_CONCEPTS, "diluted_shares_weighted_avg", latest_period_end, unit_name="shares"
        )
        facts["diluted_shares_weighted_avg"] = dshares_fact
        if dshares_fact.value is None:
            data_gaps.append("diluted_shares_weighted_avg unavailable")

        cshares_fact = self._extract_instant_fact(
            us_gaap, COMMON_SHARES_CONCEPTS, "common_shares_outstanding", latest_period_end, unit_name="shares"
        )
        facts["common_shares_outstanding"] = cshares_fact

        fiscal_year_end = submissions_dict.get("fiscalYearEnd", "1231")
        primary_exchange = submissions_dict.get("exchanges", ["UNKNOWN"])[0] if submissions_dict.get("exchanges") else "UNKNOWN"

        return CompanyFundamentals(
            symbol=symbol,
            company_name=company_name,
            cik=cik,
            primary_exchange=primary_exchange,
            sector_industry=sector_industry,
            business_model=business_model,
            fiscal_year_end=fiscal_year_end,
            latest_reported_period=latest_period_end,
            latest_filing_date=latest_filing_date or "",
            facts=facts,
            operating_indicators={
                "reporting_form": latest_form,
                "reporting_period": latest_fp,
                "reporting_fiscal_year": latest_fy,
            },
            data_gaps=data_gaps,
        )
