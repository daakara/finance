#!/usr/bin/env python3
"""
ARX Terminal — Sprint 4 Closure-Integrity Validator (GOV-001)
Enforces Rules BI-001 through BI-012 across canonical backlog registers.

Rules:
  BI-001: Unique Identity (every task_id is unique)
  BI-002: Accountable Ownership (active executable tasks must have valid owner)
  BI-003: Sprint Assignment (executable tasks assigned to valid sprint; holding queues for gated)
  BI-004: Disposition Completeness (every task has an authoritative valid disposition)
  BI-005: Carry-Forward Continuity (tasks from previous sprints have destination & reason)
  BI-006: Evidence for Completion (completed tasks must provide non-empty evidence references)
  BI-007: PR Traceability (active PRs link to canonical tasks or explicit exceptions)
  BI-008: Duplicate & Alias Detection (no duplicate identities, cyclic supersessions, alias collisions)
  BI-009: Authorization Separation (engineering state != operational authorization)
  BI-010: Closure Consistency (closing sprint cannot contain undispositioned commitments)
  BI-011: Operational Truth (independent representation of deployment, authorization, execution)
  BI-012: Historical Continuity (closed tasks must not be silently removed)
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

VALID_DISPOSITIONS = {
    "CLOSED_VERIFIED",
    "SPRINT4_EXECUTION",
    "SPRINT4_DISCOVERY",
    "SPRINT4_OWNER_GATED",
    "EXISTING_SPRINT_ASSIGNED",
    "SUPERSEDED",
    "TEST_OR_ADMINISTRATIVE",
    "UNVERIFIED",
}

VALID_STATUSES = {
    "IN_PROGRESS",
    "READY",
    "BLOCKED",
    "DEFERRED",
    "COMPLETED",
    "SUPERSEDED",
}

VALID_SPRINTS = {
    "SPRINT_1",
    "SPRINT_2",
    "SPRINT_3",
    "SPRINT_4",
    "HOLDING_QUEUE_OWNER_GATED",
    "DEFERRED_BACKLOG",
    "SPRINT_UNASSIGNED",
}

TASK_ID_REGEX = re.compile(r"^ARX-S[1-4]-[A-Z0-9\-]+$")


class Finding:
    def __init__(self, rule_id: str, task_id: Optional[str], severity: str, message: str):
        self.rule_id = rule_id
        self.task_id = task_id
        self.severity = severity  # 'ERROR' or 'WARNING'
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "task_id": self.task_id,
            "severity": self.severity,
            "message": self.message,
        }

    def __str__(self) -> str:
        tid = f" [{self.task_id}]" if self.task_id else ""
        return f"[{self.severity}] {self.rule_id}{tid}: {self.message}"


class BacklogValidator:
    def __init__(self, backlog_data: Dict[str, Any], prs_inventory: Optional[Dict[str, Any]] = None, baseline_data: Optional[Dict[str, Any]] = None):
        self.data = backlog_data
        self.prs_inventory = prs_inventory or {}
        self.baseline_data = baseline_data
        self.findings: List[Finding] = []

    def validate_all(self) -> Tuple[str, List[Finding]]:
        self.findings = []
        tasks = self.data.get("tasks", [])
        if not isinstance(tasks, list):
            self.findings.append(Finding("BI-001", None, "ERROR", "Backlog 'tasks' field must be a list"))
            return "FAIL", self.findings

        # Execute rules
        self._check_bi_001_unique_identity(tasks)
        self._check_bi_002_accountable_ownership(tasks)
        self._check_bi_003_sprint_assignment(tasks)
        self._check_bi_004_disposition_completeness(tasks)
        self._check_bi_005_carryforward_continuity(tasks)
        self._check_bi_006_evidence_for_completion(tasks)
        self._check_bi_007_pr_traceability(tasks)
        self._check_bi_008_duplicate_and_alias_detection(tasks)
        self._check_bi_009_authorization_separation(tasks)
        self._check_bi_010_closure_consistency(tasks)
        self._check_bi_011_operational_truth(tasks)
        self._check_bi_012_historical_continuity(tasks)

        errors = [f for f in self.findings if f.severity == "ERROR"]
        warnings = [f for f in self.findings if f.severity == "WARNING"]

        if errors:
            status = "FAIL"
        elif warnings:
            status = "PASS_WITH_WARNINGS"
        else:
            status = "PASS"

        return status, self.findings

    def _check_bi_001_unique_identity(self, tasks: List[Dict[str, Any]]):
        seen_ids = set()
        for t in tasks:
            tid = t.get("task_id")
            if not tid:
                self.findings.append(Finding("BI-001", None, "ERROR", "Task missing required 'task_id'"))
                continue
            if not TASK_ID_REGEX.match(tid):
                self.findings.append(Finding("BI-001", tid, "ERROR", f"Task ID '{tid}' violates naming format ^ARX-S[1-4]-[A-Z0-9\\-]+$"))
            if tid in seen_ids:
                self.findings.append(Finding("BI-001", tid, "ERROR", f"Duplicate task_id detected: '{tid}'"))
            seen_ids.add(tid)

    def _check_bi_002_accountable_ownership(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            disp = t.get("disposition")
            owner = t.get("owner")
            # Active executable or discovery tasks must have an accountable owner
            if disp in ("SPRINT4_EXECUTION", "SPRINT4_DISCOVERY"):
                if not owner or owner in ("OWNER_UNASSIGNED", "UNASSIGNED", "None", ""):
                    self.findings.append(Finding("BI-002", tid, "ERROR", f"Active task '{tid}' has no accountable owner assigned (found: '{owner}')"))

    def _check_bi_003_sprint_assignment(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            sprint = t.get("assigned_sprint")
            disp = t.get("disposition")

            if not sprint or sprint not in VALID_SPRINTS:
                self.findings.append(Finding("BI-003", tid, "ERROR", f"Task '{tid}' has invalid or missing assigned_sprint: '{sprint}'"))
                continue

            if disp in ("SPRINT4_EXECUTION", "SPRINT4_DISCOVERY") and sprint != "SPRINT_4":
                self.findings.append(Finding("BI-003", tid, "ERROR", f"Executable Sprint 4 task '{tid}' assigned to '{sprint}' instead of 'SPRINT_4'"))

            if disp == "SPRINT4_OWNER_GATED" and sprint not in ("HOLDING_QUEUE_OWNER_GATED", "DEFERRED_BACKLOG", "SPRINT_4"):
                self.findings.append(Finding("BI-003", tid, "WARNING", f"Owner-gated task '{tid}' should be in HOLDING_QUEUE_OWNER_GATED or SPRINT_4 (assigned: '{sprint}')"))

    def _check_bi_004_disposition_completeness(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            disp = t.get("disposition")
            status = t.get("status")

            if not disp or disp not in VALID_DISPOSITIONS:
                self.findings.append(Finding("BI-004", tid, "ERROR", f"Task '{tid}' missing or invalid disposition: '{disp}'"))
            if not status or status not in VALID_STATUSES:
                self.findings.append(Finding("BI-004", tid, "ERROR", f"Task '{tid}' missing or invalid status: '{status}'"))

    def _check_bi_005_carryforward_continuity(self, tasks: List[Dict[str, Any]]):
        active_sprint = self.data.get("active_sprint", "SPRINT_4")
        for t in tasks:
            tid = t.get("task_id")
            origin = t.get("origin_sprint")
            disp = t.get("disposition")
            reason = t.get("carryforward_reason")
            assigned = t.get("assigned_sprint")

            if origin and origin != active_sprint and origin in ("SPRINT_1", "SPRINT_2", "SPRINT_3"):
                if disp in ("SPRINT4_EXECUTION", "SPRINT4_DISCOVERY", "SPRINT4_OWNER_GATED"):
                    if not assigned or assigned == "SPRINT_UNASSIGNED":
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' from {origin} lacks valid receiving sprint"))
                    if not reason or not str(reason).strip():
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' from {origin} must specify non-empty 'carryforward_reason'"))

    def _check_bi_006_evidence_for_completion(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            disp = t.get("disposition")
            status = t.get("status")
            evidence = t.get("evidence_refs", [])

            if disp == "CLOSED_VERIFIED" or status == "COMPLETED":
                if not evidence or len(evidence) == 0:
                    self.findings.append(Finding("BI-006", tid, "ERROR", f"Task '{tid}' marked {disp}/{status} but has zero evidence references"))
                else:
                    empty_items = [e for e in evidence if not str(e).strip()]
                    if empty_items:
                        self.findings.append(Finding("BI-006", tid, "ERROR", f"Task '{tid}' contains empty string in evidence references"))

    def _check_bi_007_pr_traceability(self, tasks: List[Dict[str, Any]]):
        if not self.prs_inventory:
            return  # No external PR inventory passed, skip

        # Extract all referenced PR numbers from tasks
        tracked_prs = set()
        for t in tasks:
            for pr in t.get("github_prs", []):
                try:
                    tracked_prs.add(int(pr))
                except (ValueError, TypeError):
                    pass

        # PRs inventory can be dict with 'open_prs' or list of ints
        open_prs_list = []
        if isinstance(self.prs_inventory, dict):
            prs_data = self.prs_inventory.get("open_prs") or self.prs_inventory.get("prs") or self.prs_inventory
            if isinstance(prs_data, dict):
                open_prs_list = [int(k) for k in prs_data.keys() if str(k).isdigit()]
            elif isinstance(prs_data, list):
                for p in prs_data:
                    if isinstance(p, dict) and "number" in p:
                        open_prs_list.append(int(p["number"]))
                    elif isinstance(p, int):
                        open_prs_list.append(p)
        elif isinstance(self.prs_inventory, list):
            open_prs_list = [int(x) for x in self.prs_inventory if isinstance(x, int) or str(x).isdigit()]

        for pr_num in open_prs_list:
            if pr_num not in tracked_prs:
                self.findings.append(Finding("BI-007", None, "WARNING", f"Active open GitHub PR #{pr_num} is not linked to any canonical task"))

    def _check_bi_008_duplicate_and_alias_detection(self, tasks: List[Dict[str, Any]]):
        seen_aliases = {}
        superseded_map = {}

        for t in tasks:
            tid = t.get("task_id")
            aliases = t.get("historical_aliases", [])
            for a in aliases:
                if a in seen_aliases:
                    self.findings.append(Finding("BI-008", tid, "ERROR", f"Conflicting historical alias '{a}' shared between '{tid}' and '{seen_aliases[a]}'"))
                seen_aliases[a] = tid

            if t.get("disposition") == "SUPERSEDED" or t.get("status") == "SUPERSEDED":
                superseded_by = t.get("superseded_by")
                if not superseded_by:
                    self.findings.append(Finding("BI-008", tid, "ERROR", f"Task '{tid}' is SUPERSEDED but lacks 'superseded_by' reference"))
                else:
                    superseded_map[tid] = superseded_by

        # Check for cycles in supersession chain
        for start_id, target_id in superseded_map.items():
            curr = target_id
            visited = {start_id}
            while curr in superseded_map:
                if curr in visited:
                    self.findings.append(Finding("BI-008", start_id, "ERROR", f"Cyclic supersession chain detected: {' -> '.join(list(visited) + [curr])}"))
                    break
                visited.add(curr)
                curr = superseded_map[curr]

    def _check_bi_009_authorization_separation(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            dims = t.get("dimensions", {})
            owner_auth = dims.get("owner_authorization")
            op_state = t.get("operational_state", "")
            auth_boundary = t.get("authorization_boundary", "")

            # If operational_state claims ACTIVE_SERVING, owner authorization must be AUTHORIZED
            if op_state == "ACTIVE_SERVING" and owner_auth not in ("AUTHORIZED", "NOT_APPLICABLE"):
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' claims operational_state '{op_state}' while owner_authorization is '{owner_auth}'"))

            # If authorization boundary is OWNER_GATE_REQUIRED or ACTIVATION_HOLD, cannot claim AUTHORIZED unless explicitly granted
            if auth_boundary in ("OWNER_GATE_REQUIRED", "ACTIVATION_HOLD") and owner_auth == "AUTHORIZED":
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' has boundary '{auth_boundary}' but owner_authorization claims 'AUTHORIZED' without grant evidence"))

    def _check_bi_010_closure_consistency(self, tasks: List[Dict[str, Any]]):
        active_sprint = self.data.get("active_sprint", "SPRINT_4")
        # Check closed sprints: SPRINT_1, SPRINT_2, SPRINT_3
        for t in tasks:
            tid = t.get("task_id")
            origin = t.get("origin_sprint")
            assigned = t.get("assigned_sprint")
            status = t.get("status")
            disp = t.get("disposition")

            if origin in ("SPRINT_1", "SPRINT_2", "SPRINT_3") and assigned == origin:
                if status == "IN_PROGRESS":
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {origin} contains task '{tid}' with unresolved status 'IN_PROGRESS'"))
                if disp not in ("CLOSED_VERIFIED", "TEST_OR_ADMINISTRATIVE", "SUPERSEDED"):
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {origin} task '{tid}' has non-terminal disposition '{disp}' without carry-forward"))

    def _check_bi_011_operational_truth(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            dims = t.get("dimensions", {})
            op_exec = dims.get("operational_execution")
            prod_ev = dims.get("production_evidence")
            deployment = dims.get("deployment")
            impl = dims.get("implementation")

            # Merged implementation or deployment cannot automatically claim operational execution
            if op_exec == "EXECUTED" and prod_ev != "VERIFIED":
                self.findings.append(Finding("BI-011", tid, "ERROR", f"Task '{tid}' claims operational_execution 'EXECUTED' but production_evidence is '{prod_ev}' (must be VERIFIED)"))

            # Cannot claim DEPLOYED_ACTIVE if implementation is NOT_STARTED
            if deployment == "DEPLOYED_ACTIVE" and impl in ("NOT_STARTED", "NOT_APPLICABLE"):
                self.findings.append(Finding("BI-011", tid, "ERROR", f"Task '{tid}' claims deployment '{deployment}' while implementation is '{impl}'"))

    def _check_bi_012_historical_continuity(self, tasks: List[Dict[str, Any]]):
        if not self.baseline_data:
            return  # No prior baseline provided

        baseline_tasks = self.baseline_data.get("tasks", [])
        current_tids = {t.get("task_id") for t in tasks if t.get("task_id")}

        for bt in baseline_tasks:
            btid = bt.get("task_id")
            bstatus = bt.get("status")
            if btid and bstatus == "COMPLETED":
                if btid not in current_tids:
                    self.findings.append(Finding("BI-012", btid, "ERROR", f"Previously completed historical task '{btid}' was silently removed from canonical backlog"))


def main():
    parser = argparse.ArgumentParser(description="ARX Terminal Backlog Integrity Validator (GOV-001)")
    parser.add_argument("--backlog", default="docs/planning/arx_canonical_backlog.json", help="Path to canonical backlog JSON")
    parser.add_argument("--prs-inventory", default=None, help="Path to GitHub PR inventory JSON")
    parser.add_argument("--baseline", default=None, help="Path to previous baseline backlog JSON for continuity check")
    parser.add_argument("--strict", action="store_true", help="Fail closed on warnings as well as errors")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args()

    if not os.path.isfile(args.backlog):
        print(f"SOURCE_UNAVAILABLE: Backlog file not found at '{args.backlog}'", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.backlog, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
    except Exception as e:
        print(f"SOURCE_UNAVAILABLE: Failed to parse backlog JSON: {e}", file=sys.stderr)
        sys.exit(2)

    prs_inventory = None
    if args.prs_inventory and os.path.isfile(args.prs_inventory):
        try:
            with open(args.prs_inventory, "r", encoding="utf-8") as f:
                prs_inventory = json.load(f)
        except Exception:
            prs_inventory = None

    baseline_data = None
    if args.baseline and os.path.isfile(args.baseline):
        try:
            with open(args.baseline, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
        except Exception:
            baseline_data = None

    validator = BacklogValidator(backlog_data, prs_inventory=prs_inventory, baseline_data=baseline_data)
    verdict, findings = validator.validate_all()

    if args.strict and verdict == "PASS_WITH_WARNINGS":
        verdict = "FAIL"

    if args.json:
        out = {
            "verdict": verdict,
            "total_tasks": len(backlog_data.get("tasks", [])),
            "total_findings": len(findings),
            "errors": [f.to_dict() for f in findings if f.severity == "ERROR"],
            "warnings": [f.to_dict() for f in findings if f.severity == "WARNING"],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"===========================================================")
        print(f"ARX TERMINAL — BACKLOG INTEGRITY VALIDATOR (GOV-001)")
        print(f"File: {args.backlog} | Tasks: {len(backlog_data.get('tasks', []))}")
        print(f"Verdict: {verdict}")
        print(f"===========================================================")
        for f in findings:
            print(str(f))
        if not findings:
            print("All rules (BI-001 to BI-012) verified with zero errors or warnings.")

    sys.exit(0 if verdict in ("PASS", "PASS_WITH_WARNINGS") else 1)


if __name__ == "__main__":
    main()
