"""
ARX Terminal — Directive ARX-MW-001
Market-Wide Discovery Orchestrator & End-to-End Pipeline.

Coordinates the complete discovery workflow:
1. Market-Wide Universe Ingestion & 7-Population Accounting
2. Stage 1 Broad-Market Preliminary Screening (Low-cost reproducible filters)
3. Stage 2 Deeper Multi-Engine Evaluation (VCP, Trend Template, Confluence, Risk R:R)
4. Top-35 Deterministic Shortlist Selection (Cohorts A & B, <=35 candidates)
5. Incremental Discovery Comparison Against Historical 35-Symbol Baseline
6. Immutable Shadow Publication Snapshot Generation
"""

from __future__ import annotations

import datetime
import os
import json
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.universe.fixtures import HISTORICAL_VCP_35_REGRESSION_FIXTURE
from .contracts import (
    MarketWideCandidate,
    MarketWideCoverageAccounting,
    MarketWidePublicationSnapshot,
    compute_deterministic_hash,
    POC_UNIVERSE_ID,
    POC_UNIVERSE_VERSION,
    POC_RANKING_RULESET_VERSION,
    POC_CANDIDATE_GENERATION_ID,
)
from .universe_provider import MarketWideUniverseProvider
from .stage1_screener import Stage1BroadScreener
from .stage2_evaluator import Stage2DeeperEvaluator
from .top35_selector import Top35Selector, MAX_PRESENTATION_CAP


class MarketWideDiscoveryOrchestrator:
    """
    End-to-end coordinator for the Market-Wide Discovery and Top-35 Radar POC.
    Operates in strict SHADOW_ONLY posture without production mutation.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        market_db: Optional[MarketDatabaseEngine] = None,
    ):
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.market_db = market_db or MarketDatabaseEngine()
        self.universe_provider = MarketWideUniverseProvider(
            base_dir=self.base_dir,
            market_db=self.market_db,
        )
        self.stage1_screener = Stage1BroadScreener(market_db=self.market_db)
        self.stage2_evaluator = Stage2DeeperEvaluator(market_db=self.market_db)
        self.top35_selector = Top35Selector()

    def run_discovery_pipeline(
        self,
        release_sha: str = "29ead6e4df6cbf4e1e611feb74d61f0370b80e53",
        presentation_cap: int = MAX_PRESENTATION_CAP,
    ) -> Dict[str, Any]:
        """
        Executes full two-stage market discovery pipeline and produces
        immutable publication snapshot and comparative benchmark.
        """
        start_time = datetime.datetime.now(datetime.timezone.utc)
        run_id = f"RUN_MW_POC_{uuid.uuid4().hex[:12]}"

        # Step 1: Universe Ingestion & Accounting
        universe_acc = self.universe_provider.load_and_account_universe()
        scannable_symbols = universe_acc["scannable_symbols"]
        eligible_metadata = universe_acc["eligible_securities"]
        universe_build_id = universe_acc["universe_build_id"]

        # Step 2: Stage 1 Broad-Market Preliminary Screening
        stage1_report = self.stage1_screener.screen_universe(
            symbols=scannable_symbols,
            eligible_metadata=eligible_metadata,
        )
        survivors = stage1_report["survivor_symbols"]

        # Step 3: Stage 2 Deeper Multi-Engine Evaluation
        evaluated_candidates = self.stage2_evaluator.evaluate_all(
            survivors=survivors,
            stage1_results=stage1_report["detailed_results"],
            eligible_metadata=eligible_metadata,
            universe_build_id=universe_build_id,
        )

        # Step 4: Top-35 Selection & Boundary Testing
        selection_report = self.top35_selector.rank_and_select(
            candidates=evaluated_candidates,
            presentation_cap=presentation_cap,
        )
        top35_candidates: List[MarketWideCandidate] = selection_report["candidates"]

        # Step 5: Coverage and Denominator Accounting Closure (Section 10)
        source_pop = universe_acc["source_market_population"]
        identity_pop = universe_acc["identity_resolved_population"]
        eligible_univ = universe_acc["eligible_common_stock_universe"]
        ineligible_pop = universe_acc["excluded_ineligible_population"]
        data_blocked = universe_acc["eligible_data_blocked"]
        scannable = universe_acc["scannable_population"]

        stage1_eval = stage1_report["total_evaluated"]
        stage1_rej = stage1_report["rejected_count"]
        stage1_block = stage1_report["data_blocked_count"]
        stage1_fail = stage1_report["failed_count"]

        stage2_admit = len(survivors)
        stage2_full = sum(1 for c in evaluated_candidates if c.evidence_completeness.get("technical") == "COMPLETE")
        stage2_part = len(evaluated_candidates) - stage2_full

        cohort_a_cnt = selection_report["cohort_a_qualified_count"]
        cohort_b_cnt = selection_report["cohort_b_developing_count"]
        displayed_cnt = selection_report["displayed_candidates_count"]

        univ_completeness = round((eligible_univ / source_pop) * 100.0, 2) if source_pop > 0 else 0.0
        proc_completeness = round((stage1_eval / scannable) * 100.0, 2) if scannable > 0 else 0.0

        accounting = MarketWideCoverageAccounting(
            source_market_population=source_pop,
            identity_resolved_population=identity_pop,
            eligible_universe=eligible_univ,
            ineligible_population=ineligible_pop,
            eligible_data_blocked=data_blocked,
            scannable_securities=scannable,
            stage1_evaluated=stage1_eval,
            stage1_rejected=stage1_rej,
            stage1_data_blocked=stage1_block,
            stage1_failed_or_pending=stage1_fail,
            stage2_admitted=stage2_admit,
            stage2_fully_evaluated=stage2_full,
            stage2_partially_evaluated=stage2_part,
            qualified_candidates_cohort_a=cohort_a_cnt,
            developing_discoveries_cohort_b=cohort_b_cnt,
            displayed_candidates=displayed_cnt,
            universe_completeness_pct=univ_completeness,
            processing_completeness_pct=proc_completeness,
            data_coverage_status="MARKET_WIDE_COVERAGE_UNPROVEN",
            arithmetic_closure=(
                (stage1_rej + stage1_block + stage1_fail + stage2_admit == stage1_eval)
                and (cohort_a_cnt + cohort_b_cnt == stage2_admit)
            ),
        )

        # Step 6: Comparative Benchmark Against Historical 35 Fixture
        historical_35_set = set(HISTORICAL_VCP_35_REGRESSION_FIXTURE)
        top35_symbols = [c.symbol for c in top35_candidates]
        top35_set = set(top35_symbols)

        incremental_discoveries = [c for c in top35_candidates if c.symbol not in historical_35_set]
        historical_overlap = [c for c in top35_candidates if c.symbol in historical_35_set]

        benchmark_comparison = {
            "baseline_historical_fixture_count": len(historical_35_set),
            "poc_top35_count": len(top35_symbols),
            "historical_overlap_count": len(historical_overlap),
            "historical_overlap_symbols": [c.symbol for c in historical_overlap],
            "incremental_discoveries_count": len(incremental_discoveries),
            "incremental_discoveries_symbols": [c.symbol for c in incremental_discoveries],
            "top_incremental_discoveries": [
                {
                    "rank": c.rank,
                    "symbol": c.symbol,
                    "company_name": c.company_name,
                    "cohort": c.ranking_cohort.value,
                    "confluence_score": c.confluence_score,
                    "setup_pattern": c.technical_conditions.get("setup_pattern"),
                    "discovery_reason": c.discovery_reason,
                }
                for c in incremental_discoveries[:10]
            ],
        }

        # Step 7: Immutable Shadow Publication Snapshot (Section 9)
        end_time = datetime.datetime.now(datetime.timezone.utc)
        snapshot = MarketWidePublicationSnapshot(
            snapshot_id=f"SNAP_MW_POC_{run_id[-8:]}",
            universe_build_id=universe_build_id,
            logical_scan_run_id=run_id,
            candidate_generation_id=POC_CANDIDATE_GENERATION_ID,
            ranking_ruleset_version=POC_RANKING_RULESET_VERSION,
            release_sha=release_sha,
            as_of=start_time.strftime("%Y-%m-%d"),
            created_at=end_time.isoformat(),
            coverage_accounting=accounting,
            candidates=top35_candidates,
            candidates_digest=selection_report["candidates_digest"],
            is_production_authoritative=False,
            governance_label="EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC",
        )

        return {
            "run_id": run_id,
            "status": "SUCCESS",
            "duration_seconds": round((end_time - start_time).total_seconds(), 3),
            "snapshot": snapshot,
            "selection_report": selection_report,
            "stage1_report": stage1_report,
            "universe_accounting": universe_acc,
            "benchmark_comparison": benchmark_comparison,
        }
