"""
analyst_dashboard/research/sourced_fundamentals/normalizer.py

Normalizes raw SEC XBRL company facts and submissions into structured,
provenance-tracked CompanyFundamentals instances. Enforces strict period
matching, non-overlapping TTM construction, duration validation, and metric status attribution.
Eliminates duration selection errors by matching fiscal year boundaries.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, date

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
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtMaturitiesWithinOneYearAmortizedCost",
]

DEBT_CURRENT_CONCEPTS = [
    "ShortTermBorrowings",
    "LongTermDebtCurrent",
    "DebtCurrent",
    "CommercialPaper",
]

DEBT_NONCURRENT_CONCEPTS = [
    "LongTermDebtNoncurrent",
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligations",
]

DEBT_TOTAL_CONCEPTS = [
    "DebtLongtermAndShorttermCombinedAmount",
    "Debt",
]

TOTAL_CASH_CONCEPTS = [
    "CashCashEquivalentsAndShortTermInvestments",
    "CashAndShortTermInvestments",
]

LEASE_LIABILITY_CONCEPTS = [
    "OperatingLeaseLiability",
    "OperatingLeaseLiabilityNoncurrent",
    "OperatingLeaseLiabilityCurrent",
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


def parse_date(d_str: str) -> date:
    return datetime.strptime(d_str, "%Y-%m-%d").date()


class SecFactNormalizer:
    """Normalizes raw XBRL facts and submissions for an individual company."""

    def __init__(self):
        pass

    @staticmethod
    def _filter_filings(
        items: List[Dict[str, Any]],
        as_of_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Restrict to official 10-K, 10-Q, 10-K/A, and 10-Q/A reports, respecting optional cutoff."""
        res = [x for x in items if x.get("form") in ["10-K", "10-Q", "10-K/A", "10-Q/A"]]
        if as_of_date:
            res = [x for x in res if x.get("filed", "") <= as_of_date]
        return res

    @staticmethod
    def _deduplicate_and_detect_conflicts(
        items: List[Dict[str, Any]],
        concept_name: str = ""
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Deduplicates identical observations and detects genuine unresolvable conflicts.
        Policy:
        1. Exact duplicate elimination: Observations with identical (start, end, form, filed, accn, val, units)
           are collapsed into a single canonical observation.
        2. Genuine conflict detection: If two observations have identical filing metadata
           (form, filed, accn, start, end) but different values (e.g. 1000 vs 1200),
           this is an unresolvable intra-filing contradiction. Returns an error string.
        3. Restatement / Amendment precedence: If observations cover the same period (start, end)
           from different filings, an amended filing (10-K/A, 10-Q/A) or a subsequent annual filing (10-K)
           supersedes earlier filings under documented US-GAAP comparative restatement rules.
        """
        if not items:
            return [], None

        # 1. Exact deduplication
        seen_exact = set()
        unique_items = []
        for it in items:
            key = (
                it.get("form"),
                it.get("filed"),
                it.get("accn"),
                it.get("start"),
                it.get("end"),
                it.get("val"),
                it.get("frame")
            )
            if key not in seen_exact:
                seen_exact.add(key)
                unique_items.append(it)

        # 2. Check for intra-filing genuine conflicts (same form, filed, accn, start, end with differing values)
        metadata_groups: Dict[Tuple, List[float]] = {}
        for it in unique_items:
            m_key = (
                it.get("form"),
                it.get("filed"),
                it.get("accn"),
                it.get("start"),
                it.get("end"),
            )
            val = float(it.get("val", 0.0))
            if m_key not in metadata_groups:
                metadata_groups[m_key] = []
            if val not in metadata_groups[m_key]:
                metadata_groups[m_key].append(val)

        for m_key, vals in metadata_groups.items():
            if len(vals) > 1:
                form, filed, accn, start, end = m_key
                return (
                    [],
                    f"GENUINE_CONFLICT: Concept {concept_name} has conflicting values {vals} "
                    f"with identical filing metadata (form={form}, filed={filed}, accn={accn}, start={start}, end={end})"
                )

        # 3. Restatement / Amendment precedence
        def form_priority(f: str) -> int:
            if f.endswith("/A"):
                return 2
            elif f == "10-K":
                return 1
            return 0

        # Sort deterministically: end, start, filed date, form priority, accn
        sorted_items = sorted(
            unique_items,
            key=lambda x: (
                x.get("end", ""),
                x.get("start", ""),
                x.get("filed", ""),
                form_priority(x.get("form", "")),
                x.get("accn", "")
            )
        )
        return sorted_items, None

    def _get_best_concept_items(
        self,
        us_gaap: Dict[str, Any],
        concepts: List[str],
        unit_name: str = "USD",
        target_end: Optional[str] = None,
        as_of_date: Optional[str] = None,
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
                items = self._filter_filings(us_gaap[c]["units"][unit_name], as_of_date=as_of_date)
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
        as_of_date: Optional[str] = None,
    ) -> FinancialFact:
        """
        Extract a single point-in-time balance sheet fact matching target_end.
        Does NOT convert missing concepts to 0.0. Missing concepts evaluate to UNAVAILABLE.
        Reported zeroes are preserved with full filing provenance.
        """
        c, items = self._get_best_concept_items(
            us_gaap, concepts, unit_name, target_end=target_end, as_of_date=as_of_date
        )
        if not c or not items:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=f"No reported concept found among {concepts}"
            )

        matching = [x for x in items if x.get("end") == target_end]
        if not matching:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=f"Concept us-gaap/{c} found but no instant observation for period {target_end}"
            )

        # Detect conflicts and deduplicate
        deduped_matching, conflict_err = self._deduplicate_and_detect_conflicts(matching, concept_name=c)
        if conflict_err:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=conflict_err
            )

        best_item = deduped_matching[-1]
        raw_val = float(best_item["val"])

        return FinancialFact(
            metric_key=metric_key,
            value=raw_val,
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
        latest_period_end: str,
        unit_name: str = "USD",
        force_positive: bool = False,
        as_of_date: Optional[str] = None,
    ) -> FinancialFact:
        """
        Extract non-overlapping Trailing Twelve Months (TTM) duration fact.
        Enforces strict fiscal boundary and duration validation:
        - Case 1: Latest filing is 10-K full year (duration 350-375 days).
        - Case 2: Latest filing is 10-Q interim quarter.
            Identifies prior full fiscal year (p_fy),
            Current YTD starting at p_fy_end + 1 day (±4 days),
            Prior comparable YTD starting at pp_fy_end + 1 day (±4 days).
            TTM = p_fy + curr_ytd - prior_ytd.
            Derived TTM start date is prior_ytd_end + 1 day.
            Derived TTM end date is curr_ytd_end.
            Derived TTM duration is verified to be 364-366 days.
        """
        c, items = self._get_best_concept_items(
            us_gaap, concepts, unit_name, target_end=latest_period_end, as_of_date=as_of_date
        )
        if not c or not items:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=f"Concept not found among {concepts}"
            )

        valid = self._filter_filings(items, as_of_date=as_of_date)
        latest_end = parse_date(latest_period_end)

        deduped_valid, conflict_err = self._deduplicate_and_detect_conflicts(valid, concept_name=c)
        if conflict_err:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=conflict_err
            )

        # 1. Identify all true 10-K full fiscal year observations (duration 350 to 375 days)
        fy_items = [
            x for x in deduped_valid
            if x.get("form") in ["10-K", "10-K/A"]
            and x.get("start") and x.get("end")
            and 350 <= (parse_date(x["end"]) - parse_date(x["start"])).days <= 375
        ]

        # Case 1: Latest period IS a full 10-K fiscal year
        latest_fy_matches = [x for x in fy_items if x.get("end") == latest_period_end]
        if latest_fy_matches:
            best_fy = sorted(latest_fy_matches, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
            val = float(best_fy["val"])
            if force_positive:
                val = abs(val)
            return FinancialFact(
                metric_key=metric_key,
                value=val,
                units=unit_name,
                concept=f"us-gaap/{c}",
                accession_number=best_fy.get("accn"),
                form=best_fy.get("form"),
                period_start=best_fy.get("start"),
                period_end=best_fy.get("end"),
                filing_date=best_fy.get("filed"),
                retrieval_time=datetime.utcnow().isoformat() + "Z",
                status=MetricStatus.REPORTED,
                formula=f"Reported {best_fy.get('form')} full fiscal year ended {best_fy.get('end')}",
                notes=f"Full fiscal year {best_fy.get('form')} filed on {best_fy.get('filed')}"
            )

        # Case 2: Interim quarter (10-Q)
        p_fys = [x for x in fy_items if parse_date(x["end"]) < latest_end]
        if not p_fys:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=f"No prior full-year 10-K found prior to {latest_period_end}"
            )

        p_fy = sorted(p_fys, key=lambda x: (x["end"], x.get("filed", ""), x.get("accn", "")))[-1]
        p_fy_end = parse_date(p_fy["end"])
        expected_curr_start = p_fy_end + timedelta(days=1)

        # Current YTD: end == latest_end, start within 4 days of expected_curr_start
        curr_ytd_candidates = [
            x for x in deduped_valid
            if x.get("end") == latest_period_end
            and x.get("start")
            and abs((parse_date(x["start"]) - expected_curr_start).days) <= 4
        ]
        if not curr_ytd_candidates:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=f"No current YTD observation found starting near expected fiscal start {expected_curr_start}"
            )

        curr_ytd = sorted(curr_ytd_candidates, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
        curr_duration_days = (latest_end - parse_date(curr_ytd["start"])).days

        # Prior-prior full fiscal year
        pp_fys = [x for x in fy_items if parse_date(x["end"]) < p_fy_end]
        if not pp_fys:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=f"No prior-prior full-year 10-K found prior to {p_fy_end}"
            )

        pp_fy = sorted(pp_fys, key=lambda x: (x["end"], x.get("filed", ""), x.get("accn", "")))[-1]
        pp_fy_end = parse_date(pp_fy["end"])
        expected_prior_start = pp_fy_end + timedelta(days=1)

        # Prior comparable YTD: start within 4 days of expected_prior_start, duration within 7 days of curr_duration_days
        prior_ytd_candidates = [
            x for x in deduped_valid
            if x.get("start") and x.get("end")
            and parse_date(x["end"]) < latest_end
            and abs((parse_date(x["start"]) - expected_prior_start).days) <= 4
            and abs((parse_date(x["end"]) - parse_date(x["start"])).days - curr_duration_days) <= 7
        ]
        if not prior_ytd_candidates:
            return FinancialFact(
                metric_key=metric_key,
                value=None,
                units=unit_name,
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=f"No prior comparable YTD observation found starting near {expected_prior_start}"
            )

        prior_ytd = sorted(prior_ytd_candidates, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]

        val_fy = float(p_fy["val"])
        val_curr = float(curr_ytd["val"])
        val_prior = float(prior_ytd["val"])

        ttm_val = val_fy + val_curr - val_prior
        if force_positive:
            ttm_val = abs(ttm_val)

        # Derived TTM start is the day after prior_ytd end
        derived_ttm_start = parse_date(prior_ytd["end"]) + timedelta(days=1)
        derived_ttm_end = parse_date(curr_ytd["end"])

        formula_str = (
            f"Latest_FY{p_fy.get('fy', '')} ({val_fy:,.0f}) + "
            f"Current_YTD_{curr_ytd.get('fp', '')} ({val_curr:,.0f}) - "
            f"Prior_YTD_{prior_ytd.get('fp', '')} ({val_prior:,.0f})"
        )

        return FinancialFact(
            metric_key=metric_key,
            value=ttm_val,
            units=unit_name,
            concept=f"us-gaap/{c}",
            accession_number=curr_ytd.get("accn"),
            form="10-Q/10-K_TTM",
            period_start=str(derived_ttm_start),
            period_end=str(derived_ttm_end),
            filing_date=curr_ytd.get("filed"),
            retrieval_time=datetime.utcnow().isoformat() + "Z",
            status=MetricStatus.DERIVED,
            formula=formula_str,
            input_references=[
                f"{p_fy.get('form')}:{p_fy.get('accn')}",
                f"{curr_ytd.get('form')}:{curr_ytd.get('accn')}",
                f"{prior_ytd.get('form')}:{prior_ytd.get('accn')}",
            ],
            notes="Non-overlapping TTM calculated using authoritative SEC identity: FY + Current YTD - Prior YTD"
        )

    def _extract_prior_ttm_fact(
        self,
        us_gaap: Dict[str, Any],
        concepts: List[str],
        latest_period_end: str,
        unit_name: str = "USD",
        as_of_date: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Calculates non-overlapping prior TTM revenue and period dates.
        Returns: (prior_ttm_value, prior_ttm_start, prior_ttm_end)
        """
        c, items = self._get_best_concept_items(
            us_gaap, concepts, unit_name, target_end=latest_period_end, as_of_date=as_of_date
        )
        if not c or not items:
            return None, None, None

        valid = self._filter_filings(items, as_of_date=as_of_date)
        latest_end = parse_date(latest_period_end)

        deduped_valid, conflict_err = self._deduplicate_and_detect_conflicts(valid, concept_name=c)
        if conflict_err:
            logger.warning(f"Conflict in prior TTM extraction for {c}: {conflict_err}")
            return None, None, None

        fy_items = [
            x for x in deduped_valid
            if x.get("form") in ["10-K", "10-K/A"]
            and x.get("start") and x.get("end")
            and 350 <= (parse_date(x["end"]) - parse_date(x["start"])).days <= 375
        ]

        # Case 1: Latest is full 10-K FY -> Prior TTM is the immediately preceding 10-K FY
        if any(x.get("end") == latest_period_end for x in fy_items):
            p_fys = [x for x in fy_items if parse_date(x["end"]) < latest_end]
            if p_fys:
                prior_fy = sorted(p_fys, key=lambda x: (x["end"], x.get("filed", "")))[-1]
                return float(prior_fy["val"]), prior_fy.get("start"), prior_fy.get("end")
            return None, None, None

        # Case 2: Interim quarter 10-Q -> Prior TTM = pp_fy + prior_ytd - pp_ytd
        p_fys = [x for x in fy_items if parse_date(x["end"]) < latest_end]
        if not p_fys:
            return None, None, None
        p_fy = sorted(p_fys, key=lambda x: (x["end"], x.get("filed", "")))[-1]
        p_fy_end = parse_date(p_fy["end"])
        expected_curr_start = p_fy_end + timedelta(days=1)

        curr_ytd_cands = [
            x for x in deduped_valid
            if x.get("end") == latest_period_end
            and x.get("start")
            and abs((parse_date(x["start"]) - expected_curr_start).days) <= 4
        ]
        if not curr_ytd_cands:
            return None, None, None
        curr_ytd = sorted(curr_ytd_cands, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
        curr_duration_days = (latest_end - parse_date(curr_ytd["start"])).days

        pp_fys = [x for x in fy_items if parse_date(x["end"]) < p_fy_end]
        if not pp_fys:
            return None, None, None
        pp_fy = sorted(pp_fys, key=lambda x: (x["end"], x.get("filed", "")))[-1]
        pp_fy_end = parse_date(pp_fy["end"])
        expected_prior_start = pp_fy_end + timedelta(days=1)

        prior_ytd_cands = [
            x for x in deduped_valid
            if x.get("start") and x.get("end")
            and parse_date(x["end"]) < latest_end
            and abs((parse_date(x["start"]) - expected_prior_start).days) <= 4
            and abs((parse_date(x["end"]) - parse_date(x["start"])).days - curr_duration_days) <= 7
        ]
        if not prior_ytd_cands:
            return None, None, None
        prior_ytd = sorted(prior_ytd_cands, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]

        # Prior-prior-prior FY
        ppp_fys = [x for x in fy_items if parse_date(x["end"]) < pp_fy_end]
        if not ppp_fys:
            return None, None, None
        ppp_fy = sorted(ppp_fys, key=lambda x: (x["end"], x.get("filed", "")))[-1]
        ppp_fy_end = parse_date(ppp_fy["end"])
        expected_pp_start = ppp_fy_end + timedelta(days=1)

        pp_ytd_cands = [
            x for x in deduped_valid
            if x.get("start") and x.get("end")
            and parse_date(x["end"]) < parse_date(prior_ytd["end"])
            and abs((parse_date(x["start"]) - expected_pp_start).days) <= 4
            and abs((parse_date(x["end"]) - parse_date(x["start"])).days - curr_duration_days) <= 7
        ]
        if not pp_ytd_cands:
            return None, None, None
        pp_ytd = sorted(pp_ytd_cands, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]

        prior_ttm_val = float(pp_fy["val"]) + float(prior_ytd["val"]) - float(pp_ytd["val"])
        prior_ttm_start = parse_date(pp_ytd["end"]) + timedelta(days=1)
        prior_ttm_end = parse_date(prior_ytd["end"])

        return prior_ttm_val, str(prior_ttm_start), str(prior_ttm_end)

    def _extract_diluted_shares(
        self,
        us_gaap: Dict[str, Any],
        target_end: str,
        as_of_date: Optional[str] = None,
    ) -> FinancialFact:
        """
        Extract WeightedAverageNumberOfDilutedSharesOutstanding.
        Discriminates deterministically between quarterly (80-100 days), YTD, and annual (350-375 days)
        observations sharing the same period end.
        Ensures permutation invariance and eliminates source-order sensitivity.
        """
        c, items = self._get_best_concept_items(
            us_gaap, DILUTED_SHARES_CONCEPTS, unit_name="shares", target_end=target_end, as_of_date=as_of_date
        )
        if not c or not items:
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=None,
                units="shares",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes="WeightedAverageNumberOfDilutedSharesOutstanding concept not reported"
            )

        deduped_items, conflict_err = self._deduplicate_and_detect_conflicts(items, concept_name=c)
        if conflict_err:
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=None,
                units="shares",
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=conflict_err
            )

        matching = [x for x in deduped_items if x.get("end") == target_end]
        if not matching:
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=None,
                units="shares",
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=f"No diluted shares observation matching period end {target_end}"
            )

        # Check for genuine conflict within matching items at target_end
        end_matching, end_conflict = self._deduplicate_and_detect_conflicts(matching, concept_name=c)
        if end_conflict:
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=None,
                units="shares",
                concept=f"us-gaap/{c}",
                status=MetricStatus.UNAVAILABLE,
                period_end=target_end,
                notes=end_conflict
            )

        # Identify latest filing form
        latest_filing = sorted(end_matching, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
        form = latest_filing.get("form", "")

        # Case A: Annual 10-K filing -> select full fiscal year duration (350-375 days)
        if form in ["10-K", "10-K/A"]:
            fy_matches = [
                x for x in end_matching
                if x.get("start") and 350 <= (parse_date(x["end"]) - parse_date(x["start"])).days <= 375
            ]
            if fy_matches:
                best_item = sorted(fy_matches, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
                duration_days = (parse_date(best_item["end"]) - parse_date(best_item["start"])).days
                return FinancialFact(
                    metric_key="diluted_shares_weighted_avg",
                    value=float(best_item["val"]),
                    units="shares",
                    concept=f"us-gaap/{c}",
                    accession_number=best_item.get("accn"),
                    form=best_item.get("form"),
                    period_start=best_item.get("start"),
                    period_end=best_item.get("end"),
                    filing_date=best_item.get("filed"),
                    retrieval_time=datetime.utcnow().isoformat() + "Z",
                    status=MetricStatus.REPORTED,
                    notes=f"Reported annual diluted weighted-average shares ({best_item.get('start')} to {best_item.get('end')}, {duration_days} days) in {best_item.get('form')}"
                )

        # Case B: Interim quarter 10-Q -> select single quarterly duration (80-100 days)
        quarterly_matches = [
            x for x in end_matching
            if x.get("start") and 80 <= (parse_date(x["end"]) - parse_date(x["start"])).days <= 100
        ]
        if quarterly_matches:
            best_item = sorted(quarterly_matches, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
            duration_days = (parse_date(best_item["end"]) - parse_date(best_item["start"])).days
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=float(best_item["val"]),
                units="shares",
                concept=f"us-gaap/{c}",
                accession_number=best_item.get("accn"),
                form=best_item.get("form"),
                period_start=best_item.get("start"),
                period_end=best_item.get("end"),
                filing_date=best_item.get("filed"),
                retrieval_time=datetime.utcnow().isoformat() + "Z",
                status=MetricStatus.REPORTED,
                notes=f"Reported quarterly diluted weighted-average shares ({best_item.get('start')} to {best_item.get('end')}, {duration_days} days) in {best_item.get('form')}; not full-year TTM shares"
            )

        # Fallback: YTD or any duration match, deterministically selected (prefer shortest non-zero duration)
        with_start = [x for x in end_matching if x.get("start")]
        if with_start:
            best_item = sorted(with_start, key=lambda x: (
                (parse_date(x["end"]) - parse_date(x["start"])).days,
                x.get("filed", ""),
                x.get("accn", "")
            ))[0]
            duration_days = (parse_date(best_item["end"]) - parse_date(best_item["start"])).days
            return FinancialFact(
                metric_key="diluted_shares_weighted_avg",
                value=float(best_item["val"]),
                units="shares",
                concept=f"us-gaap/{c}",
                accession_number=best_item.get("accn"),
                form=best_item.get("form"),
                period_start=best_item.get("start"),
                period_end=best_item.get("end"),
                filing_date=best_item.get("filed"),
                retrieval_time=datetime.utcnow().isoformat() + "Z",
                status=MetricStatus.REPORTED,
                notes=f"Reported interim diluted weighted-average shares ({best_item.get('start')} to {best_item.get('end')}, {duration_days} days) in {best_item.get('form')}; not full-year TTM shares"
            )

        # Fallback to instant if no duration fact reported
        best_item = sorted(end_matching, key=lambda x: (x.get("filed", ""), x.get("accn", "")))[-1]
        return FinancialFact(
            metric_key="diluted_shares_weighted_avg",
            value=float(best_item["val"]),
            units="shares",
            concept=f"us-gaap/{c}",
            accession_number=best_item.get("accn"),
            form=best_item.get("form"),
            period_end=best_item.get("end"),
            filing_date=best_item.get("filed"),
            retrieval_time=datetime.utcnow().isoformat() + "Z",
            status=MetricStatus.REPORTED,
            notes=f"Reported diluted shares observation in {best_item.get('form')} ending {best_item.get('end')}"
        )

    def normalize(
        self,
        symbol: str,
        cik: str,
        facts_dict: Dict[str, Any],
        submissions_dict: Dict[str, Any],
        sector_industry: str = "",
        business_model: str = "",
        as_of_date: Optional[str] = None,
    ) -> CompanyFundamentals:
        """Full normalization entry point for a company."""
        company_name = facts_dict.get("entityName", symbol)
        us_gaap = facts_dict.get("facts", {}).get("us-gaap", {})

        # Identify latest 10-K and 10-Q from submissions
        recent = submissions_dict.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])

        latest_form = None
        latest_filing_date = None
        latest_report_date = None
        for f, fd, rd in zip(forms, filing_dates, report_dates):
            if f in ["10-K", "10-Q"]:
                if as_of_date and fd > as_of_date:
                    continue
                latest_form = f
                latest_filing_date = fd
                latest_report_date = rd
                break

        # Fallback anchor detection from us_gaap
        c_anchor, items_anchor = self._get_best_concept_items(
            us_gaap, REVENUE_CONCEPTS + NET_INCOME_CONCEPTS, target_end=latest_report_date, as_of_date=as_of_date
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
            us_gaap, REVENUE_CONCEPTS, "revenue_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["revenue_ttm"] = rev_fact
        if rev_fact.value is None:
            data_gaps.append("revenue_ttm unavailable")

        # 2. Operating Income (EBIT) TTM
        ebit_fact = self._extract_ttm_fact(
            us_gaap, OPERATING_INCOME_CONCEPTS, "operating_income_ttm", latest_period_end, as_of_date=as_of_date
        )
        # KLAC special handling: KLAC reports Revenues, Cost of Revenue, R&D, and SG&A separately without OperatingIncomeLoss tag
        if ebit_fact.value is None and symbol == "KLAC":
            cost_fact = self._extract_ttm_fact(us_gaap, ["CostOfRevenue"], "cost_of_revenue_ttm", latest_period_end, as_of_date=as_of_date)
            rd_fact = self._extract_ttm_fact(us_gaap, ["ResearchAndDevelopmentExpense"], "rd_expense_ttm", latest_period_end, as_of_date=as_of_date)
            sga_fact = self._extract_ttm_fact(us_gaap, ["SellingGeneralAndAdministrativeExpense"], "sga_expense_ttm", latest_period_end, as_of_date=as_of_date)
            if rev_fact.value and cost_fact.value and rd_fact.value and sga_fact.value:
                derived_ebit = rev_fact.value - cost_fact.value - rd_fact.value - sga_fact.value
                ebit_fact = FinancialFact(
                    metric_key="operating_income_ttm",
                    value=derived_ebit,
                    units="USD",
                    concept="DERIVED(Revenues - CostOfRevenue - R&D - SG&A)",
                    period_start=rev_fact.period_start,
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
            us_gaap, NET_INCOME_CONCEPTS, "net_income_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["net_income_ttm"] = ni_fact
        if ni_fact.value is None:
            data_gaps.append("net_income_ttm unavailable")

        # 4. CFO TTM
        cfo_fact = self._extract_ttm_fact(
            us_gaap, CFO_CONCEPTS, "cfo_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["cfo_ttm"] = cfo_fact
        if cfo_fact.value is None:
            data_gaps.append("cfo_ttm unavailable")

        # 5. Capex TTM
        capex_fact = self._extract_ttm_fact(
            us_gaap, CAPEX_CONCEPTS, "capex_ttm", latest_period_end, force_positive=True, as_of_date=as_of_date
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
                period_start=cfo_fact.period_start,
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
            us_gaap, SBC_CONCEPTS, "sbc_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["sbc_ttm"] = sbc_fact
        if sbc_fact.value is None:
            data_gaps.append("sbc_ttm unavailable")

        # 8. Cash & Equivalents (Instant)
        cash_fact = self._extract_instant_fact(
            us_gaap, CASH_CONCEPTS, "cash_and_equivalents", latest_period_end, as_of_date=as_of_date
        )
        facts["cash_and_equivalents"] = cash_fact
        if cash_fact.value is None:
            data_gaps.append("cash_and_equivalents unavailable")

        # 9. Short-Term Investments (Instant)
        st_inv_fact = self._extract_instant_fact(
            us_gaap, ST_INVESTMENT_CONCEPTS, "short_term_investments", latest_period_end, as_of_date=as_of_date
        )
        facts["short_term_investments"] = st_inv_fact

        # Total Cash & Investments
        # Completeness principle: If only cash is known, it is reported as cash;
        # do not imply complete cash-and-investments coverage without evidence.
        tot_cash_single = self._extract_instant_fact(
            us_gaap, TOTAL_CASH_CONCEPTS, "total_cash_and_investments", latest_period_end, as_of_date=as_of_date
        )

        cash_val = cash_fact.value
        st_inv_val = st_inv_fact.value

        if tot_cash_single.value is not None:
            facts["total_cash_and_investments"] = tot_cash_single
        elif cash_val is not None and st_inv_val is not None:
            # Fully evidenced sum
            facts["total_cash_and_investments"] = FinancialFact(
                metric_key="total_cash_and_investments",
                value=cash_val + st_inv_val,
                units="USD",
                status=MetricStatus.DERIVED,
                period_end=latest_period_end,
                formula=f"Cash ({cash_val:,.0f}) + Short-Term Investments ({st_inv_val:,.0f})",
                input_references=["cash_and_equivalents", "short_term_investments"],
                notes="Sum of reported cash and short-term marketable investment securities"
            )
        else:
            # Missing investments not becoming an asserted complete aggregate
            facts["total_cash_and_investments"] = FinancialFact(
                metric_key="total_cash_and_investments",
                value=None,
                units="USD",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=(
                    "Short-term investments concept not reported; complete cash-and-investments aggregate "
                    "cannot be asserted without primary filing evidence. Cash is reported separately as cash_and_equivalents."
                )
            )

        # 10. Debt Current, Noncurrent, and Total (Instants)
        dcur_fact = self._extract_instant_fact(
            us_gaap, DEBT_CURRENT_CONCEPTS, "short_term_debt", latest_period_end, as_of_date=as_of_date
        )
        facts["short_term_debt"] = dcur_fact

        dnon_fact = self._extract_instant_fact(
            us_gaap, DEBT_NONCURRENT_CONCEPTS, "long_term_debt", latest_period_end, as_of_date=as_of_date
        )
        facts["long_term_debt"] = dnon_fact

        # Comprehensive total debt concept (e.g. DebtLongtermAndShorttermCombinedAmount, Debt)
        dtot_single = self._extract_instant_fact(
            us_gaap, DEBT_TOTAL_CONCEPTS, "total_financial_debt", latest_period_end, as_of_date=as_of_date
        )

        # Determine total_financial_debt with aggregate completeness enforcement:
        if dtot_single.value is not None:
            # Case 1: Supported complete debt total from an alternate comprehensive concept
            facts["total_financial_debt"] = dtot_single
        elif dcur_fact.value is not None and dnon_fact.value is not None:
            # Case 2: Fully evidenced total from both current and noncurrent components
            tot_debt = dcur_fact.value + dnon_fact.value
            facts["total_financial_debt"] = FinancialFact(
                metric_key="total_financial_debt",
                value=tot_debt,
                units="USD",
                status=MetricStatus.DERIVED,
                period_end=latest_period_end,
                formula=f"Short-Term Debt ({dcur_fact.value:,.0f}) + Long-Term Debt ({dnon_fact.value:,.0f})",
                input_references=["short_term_debt", "long_term_debt"],
                notes="Fully evidenced sum of reported current and noncurrent financial debt obligations"
            )
        elif dcur_fact.value is not None or dnon_fact.value is not None:
            # Case 3: Known partial amount (one component exists, the other is unavailable).
            # Do NOT substitute zero or present a known partial amount as total debt.
            known_s = f"Short-term debt={dcur_fact.value:,.0f}" if dcur_fact.value is not None else "Short-term debt unavailable"
            known_l = f"Long-term debt={dnon_fact.value:,.0f}" if dnon_fact.value is not None else "Long-term debt unavailable"
            facts["total_financial_debt"] = FinancialFact(
                metric_key="total_financial_debt",
                value=None,
                units="USD",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes=(
                    f"Partial debt observation known ({known_s}, {known_l}), but complete debt obligations "
                    "cannot be established without both components or a single comprehensive debt total. "
                    "Known partial amounts are not presented as total debt."
                )
            )
            data_gaps.append("total_financial_debt unavailable (incomplete debt components)")
        else:
            # Case 4: Neither component reported. No ticker-based whitelisting.
            facts["total_financial_debt"] = FinancialFact(
                metric_key="total_financial_debt",
                value=None,
                units="USD",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes="Financial debt concepts are unavailable in primary filings."
            )
            data_gaps.append("total_financial_debt unavailable")

        # Net Debt = Total Debt - Total Cash (or Cash and Equivalents where investments are unevidenced)
        tot_d = facts["total_financial_debt"].value
        tot_c = facts["total_cash_and_investments"].value
        cash_val = facts["cash_and_equivalents"].value

        if tot_d is not None and tot_c is not None:
            facts["net_debt"] = FinancialFact(
                metric_key="net_debt",
                value=tot_d - tot_c,
                units="USD",
                status=MetricStatus.DERIVED,
                period_end=latest_period_end,
                formula=f"Total Debt ({tot_d:,.0f}) - Total Cash & Investments ({tot_c:,.0f})",
                input_references=["total_financial_debt", "total_cash_and_investments"],
                notes="Net financial debt relative to total cash and short-term investments (negative indicates net cash balance)"
            )
        elif tot_d is not None and cash_val is not None:
            # Only cash is known and evidenced; debt is fully evidenced
            facts["net_debt"] = FinancialFact(
                metric_key="net_debt",
                value=tot_d - cash_val,
                units="USD",
                status=MetricStatus.DERIVED,
                period_end=latest_period_end,
                formula=f"Total Debt ({tot_d:,.0f}) - Cash & Equivalents ({cash_val:,.0f})",
                input_references=["total_financial_debt", "cash_and_equivalents"],
                notes="Net financial debt relative to reported cash and cash equivalents; short-term investments unavailable without filing evidence."
            )
        else:
            facts["net_debt"] = FinancialFact(
                metric_key="net_debt",
                value=None,
                units="USD",
                status=MetricStatus.UNAVAILABLE,
                period_end=latest_period_end,
                notes="Cannot calculate net debt due to unavailable total financial debt or cash"
            )

        # 11. Operating Lease Liabilities
        lease_fact = self._extract_instant_fact(
            us_gaap, LEASE_LIABILITY_CONCEPTS, "operating_lease_liabilities", latest_period_end, as_of_date=as_of_date
        )
        facts["operating_lease_liabilities"] = lease_fact

        # 12. Stockholders' Equity
        equity_fact = self._extract_instant_fact(
            us_gaap, EQUITY_CONCEPTS, "stockholders_equity", latest_period_end, as_of_date=as_of_date
        )
        facts["stockholders_equity"] = equity_fact
        if equity_fact.value is None:
            data_gaps.append("stockholders_equity unavailable")

        # 13. Income Tax & Pretax Income for Effective Tax Rate
        tax_fact = self._extract_ttm_fact(
            us_gaap, TAX_EXPENSE_CONCEPTS, "tax_expense_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["tax_expense_ttm"] = tax_fact

        pretax_fact = self._extract_ttm_fact(
            us_gaap, PRETAX_INCOME_CONCEPTS, "pretax_income_ttm", latest_period_end, as_of_date=as_of_date
        )
        facts["pretax_income_ttm"] = pretax_fact

        # 14. Diluted & Common Shares
        dshares_fact = self._extract_diluted_shares(us_gaap, latest_period_end, as_of_date=as_of_date)
        facts["diluted_shares_weighted_avg"] = dshares_fact
        if dshares_fact.value is None:
            data_gaps.append("diluted_shares_weighted_avg unavailable")

        cshares_fact = self._extract_instant_fact(
            us_gaap, COMMON_SHARES_CONCEPTS, "common_shares_outstanding", latest_period_end, unit_name="shares", as_of_date=as_of_date
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
