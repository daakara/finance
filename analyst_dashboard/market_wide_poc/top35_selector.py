"""
ARX Terminal — Directive ARX-MW-001
Top-35 Deterministic Shortlist Selection & Ranking Engine (Section 7).

Evaluates two discovery cohorts:
- Cohort A: Qualified Setups (satisfying mandatory setup, trend, and R:R predicates)
- Cohort B: Developing Discoveries (showing emerging leadership or basing patterns)

Applies deterministic cross-cohort ranking with strict tie-breaking, selects at most
35 unique securities (publishing fewer if fewer qualify), and tests the 35th/36th boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from .contracts import (
    RankingCohort,
    MarketWideCandidate,
    compute_deterministic_hash,
)

MAX_PRESENTATION_CAP: int = 35


class Top35Selector:
    """
    Ranks evaluated candidates deterministically across cohorts,
    enforcing presentation cap <= 35 and explicit tie-breaking.
    """

    @staticmethod
    def rank_and_select(
        candidates: List[MarketWideCandidate],
        presentation_cap: int = MAX_PRESENTATION_CAP,
    ) -> Dict[str, Any]:
        """
        Sorts candidates by:
          1. Cohort priority: Cohort A (Qualified) before Cohort B (Developing)
          2. Composite Confluence Score DESC
          3. Technical Risk-Reward Ratio DESC (None treated as 0.0)
          4. Trend Template Relative Strength Score DESC
          5. Deterministic tie-breaker: Ticker Symbol ASC
        """
        def sort_key(c: MarketWideCandidate) -> Tuple[int, float, float, float, str]:
            cohort_order = 0 if c.ranking_cohort == RankingCohort.COHORT_A_QUALIFIED else 1
            confluence = -float(c.confluence_score)
            rr = -float(c.risk_reward_ratio or 0.0)
            rs = -float(c.relative_strength or 0.0)
            sym = c.symbol.upper()
            return (cohort_order, confluence, rr, rs, sym)

        sorted_candidates = sorted(candidates, key=sort_key)

        total_evaluated = len(sorted_candidates)
        cohort_a_candidates = [c for c in sorted_candidates if c.ranking_cohort == RankingCohort.COHORT_A_QUALIFIED]
        cohort_b_candidates = [c for c in sorted_candidates if c.ranking_cohort == RankingCohort.COHORT_B_DEVELOPING]

        selected_slice = sorted_candidates[:presentation_cap]

        # Assign 1-based ranks
        ranked_top: List[MarketWideCandidate] = []
        for rank_idx, cand in enumerate(selected_slice, start=1):
            cand.rank = rank_idx
            ranked_top.append(cand)

        # 35th / 36th boundary analysis
        boundary_35th: Optional[Dict[str, Any]] = None
        boundary_36th: Optional[Dict[str, Any]] = None
        boundary_diff: Optional[Dict[str, Any]] = None

        if len(sorted_candidates) >= presentation_cap:
            c35 = sorted_candidates[presentation_cap - 1]
            boundary_35th = {
                "rank": 35,
                "symbol": c35.symbol,
                "cohort": c35.ranking_cohort.value,
                "confluence_score": c35.confluence_score,
                "risk_reward_ratio": c35.risk_reward_ratio,
                "relative_strength": c35.relative_strength,
            }

        if len(sorted_candidates) > presentation_cap:
            c36 = sorted_candidates[presentation_cap]
            boundary_36th = {
                "rank": 36,
                "symbol": c36.symbol,
                "cohort": c36.ranking_cohort.value,
                "confluence_score": c36.confluence_score,
                "risk_reward_ratio": c36.risk_reward_ratio,
                "relative_strength": c36.relative_strength,
            }
            boundary_diff = {
                "confluence_margin": round(boundary_35th["confluence_score"] - boundary_36th["confluence_score"], 2),
                "tie_broken_by": "Confluence Score" if boundary_35th["confluence_score"] != boundary_36th["confluence_score"] else (
                    "Risk-Reward Ratio" if boundary_35th["risk_reward_ratio"] != boundary_36th["risk_reward_ratio"] else "Symbol Alphabetical"
                ),
            }

        candidates_dicts = [c.to_dict() for c in ranked_top]
        candidates_digest = compute_deterministic_hash(candidates_dicts)

        return {
            "total_candidates_evaluated": total_evaluated,
            "cohort_a_qualified_count": len(cohort_a_candidates),
            "cohort_b_developing_count": len(cohort_b_candidates),
            "displayed_candidates_count": len(ranked_top),
            "presentation_cap": presentation_cap,
            "candidates": ranked_top,
            "candidates_digest": candidates_digest,
            "boundary_35th": boundary_35th,
            "boundary_36th": boundary_36th,
            "boundary_diff": boundary_diff,
            "deterministic_sort_contract": (
                "COHORT_ORDER_ASC(A=0, B=1) -> CONFLUENCE_DESC -> RR_DESC -> RS_DESC -> SYMBOL_ASC"
            ),
        }
