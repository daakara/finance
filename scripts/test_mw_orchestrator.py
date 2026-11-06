import os
import sys
import json
from analyst_dashboard.market_wide_poc.discovery_orchestrator import MarketWideDiscoveryOrchestrator

def main():
    orchestrator = MarketWideDiscoveryOrchestrator()
    res = orchestrator.run_discovery_pipeline()
    print("Status:", res["status"])
    print("Duration (s):", res["duration_seconds"])

    snap = res["snapshot"]
    acc = snap.coverage_accounting
    print("--- Funnel Accounting ---")
    print(f"Eligible Common Stock Universe: {acc.eligible_universe}")
    print(f"Scannable Population: {acc.scannable_securities}")
    print(f"Stage 1 Evaluated: {acc.stage1_evaluated}")
    print(f"Stage 1 Survived: {acc.stage2_admitted}")
    print(f"Stage 1 Rejected: {acc.stage1_rejected}")
    print(f"Stage 1 Data Blocked: {acc.stage1_data_blocked}")
    print(f"Stage 2 Cohort A (Qualified): {acc.qualified_candidates_cohort_a}")
    print(f"Stage 2 Cohort B (Developing): {acc.developing_discoveries_cohort_b}")
    print(f"Top 35 Candidates Displayed: {acc.displayed_candidates}")
    print(f"Arithmetic Closure Verified: {acc.arithmetic_closure}")

    bench = res["benchmark_comparison"]
    print("\n--- Comparative Benchmark ---")
    print(f"Historical 35 Overlap Count: {bench['historical_overlap_count']}")
    print(f"Incremental Discoveries Count: {bench['incremental_discoveries_count']}")
    print("\nTop 10 Candidates Selected:")
    for c in snap.candidates[:10]:
        print(f"  #{c.rank:02d} [{c.ranking_cohort.value[:8]}] {c.symbol:5s} - Price: ${c.current_price:7.2f} | Conf: {c.confluence_score:4.1f} | R:R: {c.risk_reward_ratio or 0.0:.1f} | Pattern: {c.technical_conditions.get('setup_pattern')}")

    print("\n35th / 36th Boundary:")
    print("  35th:", res["selection_report"]["boundary_35th"])
    print("  36th:", res["selection_report"]["boundary_36th"])
    print("  Diff:", res["selection_report"]["boundary_diff"])

if __name__ == "__main__":
    main()
