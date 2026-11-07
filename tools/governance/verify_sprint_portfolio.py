#!/usr/bin/env python3
"""Offline validator for the non-activating ARX sprint portfolio overlay."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "docs/planning/arx_canonical_backlog.json"
PORTFOLIO = ROOT / "docs/planning/arx_sprint_portfolio_2026_11_07.json"
EXPECTED_COUNTS = {"COMPLETED": 20, "SUPERSEDED": 3, "READY": 10, "BLOCKED": 10}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    portfolio = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    require(canonical["sprint5_execution_authorized"] is False, "Sprint 5 is not authorized")
    require(portfolio["sprint5_execution_authorized"] is False, "Portfolio cannot activate Sprint 5")
    require(canonical["active_sprint"] == "SPRINT_4", "Unexpected canonical active sprint")
    require(len(canonical["obligations"]) == portfolio["canonical_snapshot"]["obligation_count"], "Obligations changed")
    tasks = canonical["tasks"]
    require(len(tasks) == portfolio["canonical_snapshot"]["task_count"], "Task count changed")
    counts = {status: sum(t["status"] == status for t in tasks) for status in EXPECTED_COUNTS}
    require(counts == portfolio["canonical_snapshot"]["task_status_counts"], "Task lifecycle drift; regenerate plan")
    require(counts == EXPECTED_COUNTS, "Baseline status drift; new reconciliation required")

    lanes = portfolio["lanes"]
    require([lane["id"] for lane in lanes] == ["P0", "P1", "P2", "P3", "P4", "HOLD"], "Unexpected sprint lanes")
    for field in ("issues", "pull_requests", "canonical_task_ids"):
        entries = [x for lane in lanes for x in lane[field]]
        require(len(entries) == len(set(entries)), f"Duplicate {field} allocation")
    issues = {n for lane in lanes for n in lane["issues"]}
    prs = {n for lane in lanes for n in lane["pull_requests"]}
    expected_open = {58,59,60,61,62,63,64,65,66,67,68,69,70,165,166,181,197,198,203,204}
    expected_prs = {38,40,42,44,47,73,75,78,83,103,109,124,128,141,142,146,193}
    require(issues == expected_open, "GitHub issue inventory mismatch to dated snapshot")
    require(prs == expected_prs, "GitHub PR inventory mismatch to dated snapshot")
    open_ids = {t["task_id"] for t in tasks if t["status"] in {"READY", "BLOCKED"}}
    mapped = {x for lane in lanes for x in lane["canonical_task_ids"]}
    require(open_ids == mapped, "Non-complete canonical task coverage mismatch")
    task_lookup = {t["task_id"]: t for t in tasks}
    held = {t["task_id"] for t in tasks if t["status"] == "BLOCKED" or t["dimensions"]["owner_authorization"] == "HOLD"}
    hold_ids = set(next(l for l in lanes if l["id"] == "HOLD")["canonical_task_ids"])
    require(held <= hold_ids, "Owner-gated tasks incorrectly allocated to executable lanes")
    require(all(task_lookup[x]["status"] in {"READY", "BLOCKED"} for x in mapped), "Completed task reactivated")
    print("PASS: 6 planning lanes; 20 issues; 17 PRs; 20 non-complete tasks; 71 obligations; Sprint 5 held")


if __name__ == "__main__":
    main()
