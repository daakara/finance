#!/usr/bin/env python3
"""
ARX Terminal — Cross-Sprint Backlog Integrity Validator (GOV-001)
Enforces Rules BI-000 through BI-012 across canonical backlog registers.

Rules:
  BI-000: Schema Validation (backlog complies with JSON Schema Draft-07)
  BI-001: Unique Identity (every task_id is unique and matches format ^ARX-S[1-4]-[A-Z0-9\\-]+$)
  BI-002: Accountable Ownership (active executable tasks must have valid owner)
  BI-003: Sprint Assignment (executable tasks assigned to valid sprint; holding queues for gated)
  BI-004: Disposition Completeness (every task has an authoritative valid disposition and status)
  BI-005: Carry-Forward Continuity (tasks from previous sprints have destination & reason)
  BI-006: Evidence for Completion (completed tasks must provide non-empty evidence references)
  BI-007: PR Traceability (active PRs link to canonical tasks or explicit exceptions)
  BI-008: Duplicate & Alias Detection (no duplicate identities, cyclic supersessions, alias collisions)
  BI-009: Authorization Separation (engineering state != operational authorization)
  BI-010: Closure Consistency (closing sprint cannot contain undispositioned commitments)
  BI-011: Operational Truth (independent representation of deployment, authorization, execution)
  BI-012: Historical Continuity (closed tasks must not be silently removed)

GitHub Invariant:
  GitHub /issues returns both issues and PRs. Records containing the 'pull_request' key
  are Pull Requests; records lacking 'pull_request' are genuine issues.
  Missing GitHub inventory never means zero issues or PRs.
  Missing historical baseline never proves task continuity.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

VALID_DISPOSITIONS = {
    "CLOSED_VERIFIED",
    "SPRINT4_EXECUTION",
    "SPRINT4_DISCOVERY",
    "SPRINT4_OWNER_GATED",
    "SPRINT5_EXECUTION",
    "SPRINT5_DISCOVERY",
    "SPRINT5_OWNER_GATED",
    "CLOSED_SPRINT_CONDITIONAL_OPEN_TASK",
    "ENGINEERING_ACCEPTED_AWAITING_RELEASE",
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
    "SPRINT_5",
    "HOLDING_QUEUE_OWNER_GATED",
    "DEFERRED_BACKLOG",
    "SPRINT_UNASSIGNED",
}

REQUIRED_IMMUTABLE_HOLDS = {
    "SPRINT3": "CLOSED_WITH_CONTROLLED_CARRYFORWARD",
    "SMART_MONEY_WORKER_ACTIVATION": "NOT_AUTHORIZED",
    "SMART_MONEY_SEC_EGRESS": "DISABLED",
    "SMART_MONEY_RECURRING_INGESTION": "NOT_AUTHORIZED",
    "CORPORATE_EVENTS_TELEMETRY_ACTIVATION": "NOT_AUTHORIZED",
    "HVA00_PROTOCOL_RATIFICATION": "NOT_GRANTED",
    "HVA01_HUMAN_EVALUATION": "NOT_AUTHORIZED",
    "HVA02_ALGORITHM_ACCEPTANCE": "NOT_AUTHORIZED",
    "EXTERNAL_CUSTODY_ACTIVATION": "NOT_AUTHORIZED",
    "UX_PARTICIPANT_RESEARCH": "NOT_AUTHORIZED",
    "MODEL_TUNING": "FROZEN",
}

TASK_ID_REGEX = re.compile(r"^ARX-S[1-5]-[A-Z0-9\-]+$")

GOVERNANCE_PATH_PATTERNS = [
    "docs/planning/",
    "docs/sprints/",
    "tools/governance/",
    "tests/test_backlog_integrity_validator.py",
    "ARX_SPRINT",
    "arx_canonical_backlog",
]


class Finding:
    """Stable diagnostic envelope for governance findings."""
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


def classify_github_issue_record(record: Dict[str, Any]) -> str:
    """
    Classify a GitHub record from /issues.
    Returns 'PULL_REQUEST' if 'pull_request' key is present and truthy, else 'ISSUE'.
    """
    if "pull_request" in record and record["pull_request"]:
        return "PULL_REQUEST"
    return "ISSUE"


def reconcile_github_inventory(
    issues_records: List[Dict[str, Any]],
    pulls_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Reconciles GitHub issues and pulls inventories following the invariant:
    - /issues records containing 'pull_request' are PRs.
    - /issues records without 'pull_request' are genuine issues.
    - /pulls records are PRs.
    Deduplicates by record number.
    """
    genuine_issues = {}
    pull_requests = {}

    for item in issues_records:
        num = item.get("number")
        if num is None:
            continue
        classification = classify_github_issue_record(item)
        if classification == "ISSUE":
            genuine_issues[num] = item
        else:
            pull_requests[num] = item

    for item in pulls_records:
        num = item.get("number")
        if num is not None:
            pull_requests[num] = item

    return {
        "genuine_issues_count": len(genuine_issues),
        "pull_requests_count": len(pull_requests),
        "total_records_count": len(genuine_issues) + len(pull_requests),
        "genuine_issues": genuine_issues,
        "pull_requests": pull_requests,
    }


class BacklogValidator:
    def __init__(
        self,
        backlog_data: Dict[str, Any],
        prs_inventory: Optional[Dict[str, Any]] = None,
        issues_inventory: Optional[Dict[str, Any]] = None,
        baseline_data: Optional[Dict[str, Any]] = None,
        schema_data: Optional[Dict[str, Any]] = None,
        require_baseline: bool = False,
        require_external_inventory: bool = False,
        require_closure: bool = False,
        pre_sprint5_closure: bool = False,
        pre_sprint5_readiness: bool = False,
    ):
        self.data = backlog_data
        self.prs_inventory = prs_inventory
        self.issues_inventory = issues_inventory
        self.baseline_data = baseline_data
        self.schema_data = schema_data
        self.require_baseline = require_baseline
        self.require_external_inventory = require_external_inventory
        self.require_closure = require_closure
        self.pre_sprint5_closure = pre_sprint5_closure
        self.pre_sprint5_readiness = pre_sprint5_readiness
        self.findings: List[Finding] = []

        # Audit statuses
        self.schema_status = "SKIPPED_NO_SCHEMA"
        self.continuity_status = "UNVERIFIED"
        self.reconciliation_status = "NOT_PERFORMED"
        self.reconciliation_counts = {"genuine_issues": 0, "pull_requests": 0}

    def validate_all(self) -> Tuple[str, List[Finding]]:
        self.findings = []

        # 0. Schema Validation (BI-000)
        self._check_bi_000_schema_validation()

        tasks = self.data.get("tasks", [])
        if not isinstance(tasks, list):
            self.findings.append(Finding("BI-001", None, "ERROR", "Backlog 'tasks' field must be a list"))
            return "FAIL", self.findings

        # Execute rules BI-001 to BI-012
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

        # Pre-Sprint 5 Technical Readiness Invariants (Outcome A)
        if self.pre_sprint5_readiness:
            self._check_p5_readiness_invariants(tasks)

        # Pre-Sprint 5 Closure & Execution Gate (Outcome B)
        if self.pre_sprint5_closure:
            self._check_p5_invariants(tasks)

        errors = [f for f in self.findings if f.severity == "ERROR"]
        warnings = [f for f in self.findings if f.severity == "WARNING"]

        if errors:
            status = "FAIL"
        elif warnings:
            status = "PASS_WITH_WARNINGS"
        else:
            status = "PASS"

        return status, self.findings

    def _check_bi_000_schema_validation(self):
        if not self.schema_data:
            self.schema_status = "SKIPPED_NO_SCHEMA"
            return

        if HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=self.data, schema=self.schema_data)
                self.schema_status = "VERIFIED"
            except jsonschema.ValidationError as e:
                self.schema_status = "FAILED"
                loc = "/".join(str(p) for p in e.path) if e.path else "root"
                self.findings.append(Finding("BI-000", None, "ERROR", f"Schema validation error at '{loc}': {e.message}"))
            except Exception as e:
                self.schema_status = "FAILED"
                self.findings.append(Finding("BI-000", None, "ERROR", f"Schema validation failed: {str(e)}"))
        else:
            required_top = ["schema_version", "backlog_name", "tasks"]
            missing_top = [k for k in required_top if k not in self.data]
            if missing_top:
                self.schema_status = "FAILED"
                self.findings.append(Finding("BI-000", None, "ERROR", f"Missing required top-level schema fields: {missing_top}"))
            else:
                self.schema_status = "VERIFIED (STRUCTURAL_FALLBACK)"

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
            if disp in ("SPRINT4_EXECUTION", "SPRINT4_DISCOVERY", "SPRINT5_EXECUTION", "SPRINT5_DISCOVERY"):
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

            if disp in ("SPRINT5_EXECUTION", "SPRINT5_DISCOVERY") and sprint != "SPRINT_5":
                self.findings.append(Finding("BI-003", tid, "ERROR", f"Executable Sprint 5 task '{tid}' assigned to '{sprint}' instead of 'SPRINT_5'"))

            if disp in ("SPRINT4_OWNER_GATED", "SPRINT5_OWNER_GATED") and sprint not in ("HOLDING_QUEUE_OWNER_GATED", "DEFERRED_BACKLOG", "SPRINT_4", "SPRINT_5"):
                self.findings.append(Finding("BI-003", tid, "WARNING", f"Owner-gated task '{tid}' should be in HOLDING_QUEUE_OWNER_GATED or assigned sprint (assigned: '{sprint}')"))

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
        closed_sprints = {"SPRINT_1", "SPRINT_2", "SPRINT_3"}
        if active_sprint == "SPRINT_5":
            closed_sprints.add("SPRINT_4")

        for t in tasks:
            tid = t.get("task_id")
            origin = t.get("origin_sprint")
            disp = t.get("disposition")
            reason = t.get("carryforward_reason")
            assigned = t.get("assigned_sprint")

            if origin and origin != active_sprint and origin in closed_sprints:
                if disp in ("SPRINT4_EXECUTION", "SPRINT4_DISCOVERY", "SPRINT4_OWNER_GATED", "SPRINT5_EXECUTION", "SPRINT5_DISCOVERY", "SPRINT5_OWNER_GATED"):
                    if not assigned or assigned == "SPRINT_UNASSIGNED":
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' from {origin} lacks valid receiving sprint or queue"))
                    if not reason or not str(reason).strip():
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' from {origin} must specify non-empty 'carryforward_reason'"))
                    
                    # BI-005 Hardening (Directive 042): Carryforward tasks must contain valid, non-empty acceptance criteria
                    criteria = t.get("acceptance_criteria")
                    if criteria is None:
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' must specify 'acceptance_criteria'"))
                    elif not isinstance(criteria, list) or len(criteria) == 0:
                        self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' must specify non-empty 'acceptance_criteria' list"))
                    else:
                        for idx, c in enumerate(criteria):
                            if not isinstance(c, str):
                                self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' criterion #{idx+1} must be a string, got {type(c).__name__}"))
                            elif not c.strip():
                                self.findings.append(Finding("BI-005", tid, "ERROR", f"Carried-forward task '{tid}' criterion #{idx+1} cannot be empty or whitespace-only"))

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
        if self.prs_inventory is None:
            self.reconciliation_status = "NOT_PERFORMED"
            if self.require_external_inventory or self.require_closure:
                self.findings.append(Finding("BI-007", None, "ERROR", "External GitHub PR inventory is required for closure certification but missing"))
            return

        self.reconciliation_status = "PERFORMED"
        tracked_prs = set()
        for t in tasks:
            for pr in t.get("github_prs", []):
                try:
                    tracked_prs.add(int(pr))
                except (ValueError, TypeError):
                    pass

        open_prs_list = []
        if isinstance(self.prs_inventory, dict):
            prs_data = (
                self.prs_inventory.get("open_prs")
                or self.prs_inventory.get("pull_requests")
                or self.prs_inventory.get("prs")
                or self.prs_inventory
            )
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

        self.reconciliation_counts["pull_requests"] = len(open_prs_list)

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
            disp = t.get("disposition")
            status = t.get("status")

            if op_state == "ACTIVE_SERVING" and owner_auth not in ("AUTHORIZED", "NOT_APPLICABLE"):
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' claims operational_state '{op_state}' while owner_authorization is '{owner_auth}'"))

            if auth_boundary in ("OWNER_GATE_REQUIRED", "ACTIVATION_HOLD") and owner_auth == "AUTHORIZED":
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' has boundary '{auth_boundary}' but owner_authorization claims 'AUTHORIZED' without grant evidence"))

            # Directive 043: Strict separation of administrative recommendation from operational state
            if "RECOMMENDED" in op_state:
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' operational_state '{op_state}' contains administrative recommendation; recommendations must not pollute operational_state"))

            if "CLOSED_VERIFIED" in op_state and disp != "CLOSED_VERIFIED":
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' claims operational_state '{op_state}' while disposition is '{disp}'"))

            # Directive 043: Tasks cannot be closed without explicit owner authorization
            if (disp == "CLOSED_VERIFIED" or status == "COMPLETED") and owner_auth not in ("AUTHORIZED", "NOT_APPLICABLE"):
                self.findings.append(Finding("BI-009", tid, "ERROR", f"Task '{tid}' marked {disp}/{status} but owner_authorization is '{owner_auth}' (must be AUTHORIZED)"))

    def _check_bi_010_closure_consistency(self, tasks: List[Dict[str, Any]]):
        active_sprint = self.data.get("active_sprint", "SPRINT_4")
        closed_sprints = {"SPRINT_1", "SPRINT_2", "SPRINT_3"}
        if active_sprint == "SPRINT_5":
            closed_sprints.add("SPRINT_4")

        valid_terminal_dispositions = {
            "CLOSED_VERIFIED",
            "TEST_OR_ADMINISTRATIVE",
            "SUPERSEDED",
            "CLOSED_SPRINT_CONDITIONAL_OPEN_TASK",
            "ENGINEERING_ACCEPTED_AWAITING_RELEASE",
        }

        for t in tasks:
            tid = t.get("task_id")
            origin = t.get("origin_sprint")
            assigned = t.get("assigned_sprint")
            status = t.get("status")
            disp = t.get("disposition")

            if origin in closed_sprints and assigned == origin:
                if status == "IN_PROGRESS":
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {origin} contains task '{tid}' with unresolved status 'IN_PROGRESS'"))
                if disp not in valid_terminal_dispositions:
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {origin} task '{tid}' has non-terminal disposition '{disp}' without carry-forward"))

            if assigned in closed_sprints and assigned != active_sprint:
                if status == "IN_PROGRESS":
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {assigned} contains task '{tid}' with unresolved status 'IN_PROGRESS'"))
                if disp not in valid_terminal_dispositions:
                    self.findings.append(Finding("BI-010", tid, "ERROR", f"Closed sprint {assigned} task '{tid}' has non-terminal disposition '{disp}' without carry-forward"))

    def _check_bi_011_operational_truth(self, tasks: List[Dict[str, Any]]):
        for t in tasks:
            tid = t.get("task_id")
            dims = t.get("dimensions", {})
            op_exec = dims.get("operational_execution")
            prod_ev = dims.get("production_evidence")
            deployment = dims.get("deployment")
            impl = dims.get("implementation")

            if op_exec == "EXECUTED" and prod_ev != "VERIFIED":
                self.findings.append(Finding("BI-011", tid, "ERROR", f"Task '{tid}' claims operational_execution 'EXECUTED' but production_evidence is '{prod_ev}' (must be VERIFIED)"))

            if deployment == "DEPLOYED_ACTIVE" and impl in ("NOT_STARTED", "NOT_APPLICABLE"):
                self.findings.append(Finding("BI-011", tid, "ERROR", f"Task '{tid}' claims deployment '{deployment}' while implementation is '{impl}'"))

    def _check_bi_012_historical_continuity(self, tasks: List[Dict[str, Any]]):
        if self.baseline_data is None:
            self.continuity_status = "UNVERIFIED"
            if self.require_baseline or self.require_closure:
                self.findings.append(Finding("BI-012", None, "ERROR", "Historical baseline is required but missing or unparseable; task continuity cannot be verified"))
            return

        baseline_tasks = self.baseline_data.get("tasks", [])
        current_tids = {t.get("task_id") for t in tasks if t.get("task_id")}

        deleted_completed = []
        for bt in baseline_tasks:
            btid = bt.get("task_id")
            bstatus = bt.get("status")
            if btid and bstatus == "COMPLETED":
                if btid not in current_tids:
                    deleted_completed.append(btid)
                    self.findings.append(Finding("BI-012", btid, "ERROR", f"Previously completed historical task '{btid}' was silently removed from canonical backlog"))

        if deleted_completed:
            self.continuity_status = "FAILED"
        else:
            self.continuity_status = "VERIFIED"



    def _check_p5_readiness_invariants(self, tasks: List[Dict[str, Any]]):
        """Technical Closure Readiness Invariants (Outcome A: P5-001 to P5-009, P5-011 to P5-012)."""
        obligations = self.data.get("obligations", [])
        tasks_by_id = {t.get("task_id"): t for t in tasks if t.get("task_id")}

        # P5-001: Zero unscoped obligations
        if not obligations:
            self.findings.append(Finding("P5-001", None, "ERROR", "Mandatory obligation register is empty or missing in canonical backlog"))
        for obl in obligations:
            oid = obl.get("obligation_id")
            scope = obl.get("release_scope")
            if not scope or scope in ("UNSCOPED_BLOCKER", "UNASSIGNED"):
                self.findings.append(Finding("P5-001", oid, "ERROR", f"Obligation '{oid}' lacks an approved release/sprint scope (found: '{scope}')"))

        # P5-002: Zero orphan findings
        for obl in obligations:
            oid = obl.get("obligation_id")
            parent_id = obl.get("canonical_task_id")
            if not parent_id or parent_id not in tasks_by_id:
                self.findings.append(Finding("P5-002", oid, "ERROR", f"Obligation '{oid}' has no valid canonical task parent in backlog (parent: '{parent_id}')"))

        # P5-003: No incomplete mandatory pre-Sprint 5 work
        for obl in obligations:
            oid = obl.get("obligation_id")
            scope = obl.get("release_scope")
            status = obl.get("status")
            disp = obl.get("disposition")
            deploy = obl.get("deployment_state")
            if scope == "PRE_SPRINT5_MANDATORY_CLOSURE":
                if disp == "RELEASE_HELD" or deploy == "RELEASE_HELD" or obl.get("production_acceptance_state") == "RELEASE_HELD":
                    blocker = obl.get("blocking_decision", "PRODUCTION_RELEASE_HELD")
                    self.findings.append(Finding("P5-003", oid, "ERROR", f"Mandatory pre-Sprint 5 obligation '{oid}' is held from production deployment (status='{status}', disp='{disp}', deploy='{deploy}'). Blocker: {blocker}"))
                elif status != "COMPLETED" or disp != "CLOSED_VERIFIED":
                    blocker = obl.get("blocking_decision", "UNRESOLVED_BLOCKER")
                    self.findings.append(Finding("P5-003", oid, "ERROR", f"Mandatory pre-Sprint 5 obligation '{oid}' is incomplete (status='{status}', disp='{disp}'). Blocker: {blocker}"))

        # P5-004: No silent carryforwards
        for obl in obligations:
            oid = obl.get("obligation_id")
            origin = obl.get("origin_sprint")
            scope = obl.get("release_scope")
            auth = obl.get("scope_authorization")
            criteria = obl.get("acceptance_criteria", [])
            owner = obl.get("owner")
            if origin in ("SPRINT_1", "SPRINT_2", "SPRINT_3", "SPRINT_4") and scope in ("ALREADY_APPROVED_SPRINT5_DELIVERY", "SPRINT_5"):
                if not auth or auth not in ("OWNER_AUTHORIZED", "CONDITIONALLY_PREAPPROVED"):
                    self.findings.append(Finding("P5-004", oid, "ERROR", f"Obligation '{oid}' carried forward to Sprint 5 lacks independent owner authorization (auth='{auth}')"))
                if not criteria or all(not str(c).strip() for c in criteria):
                    self.findings.append(Finding("P5-004", oid, "ERROR", f"Obligation '{oid}' carried forward to Sprint 5 lacks concrete acceptance criteria"))
                if not owner or len(owner.strip()) < 2:
                    self.findings.append(Finding("P5-004", oid, "ERROR", f"Obligation '{oid}' carried forward to Sprint 5 lacks an accountable owner"))

        # P5-005: No unauthorized HOLD release
        # 1. Enforce top-level immutable_holds register
        immutable_holds = self.data.get("immutable_holds")
        if not isinstance(immutable_holds, dict):
            self.findings.append(Finding("P5-005", None, "ERROR", "Top-level 'immutable_holds' register is missing or not a dictionary"))
        else:
            for hold_key, expected_val in REQUIRED_IMMUTABLE_HOLDS.items():
                if hold_key not in immutable_holds:
                    self.findings.append(Finding("P5-005", None, "ERROR", f"Mandatory operational hold '{hold_key}' is missing from immutable_holds register"))
                elif immutable_holds[hold_key] != expected_val:
                    self.findings.append(Finding("P5-005", None, "ERROR", f"Mandatory operational hold '{hold_key}' has been relaxed or mutated to '{immutable_holds[hold_key]}' (expected: '{expected_val}')"))

        # 2. Enforce individual held obligations
        for obl in obligations:
            oid = obl.get("obligation_id")
            scope = obl.get("release_scope")
            auth = obl.get("scope_authorization")
            gov_state = obl.get("governance_state")
            status = obl.get("status")
            deploy = obl.get("deployment_state")
            op = obl.get("operational_state")
            is_hold = (scope == "HOLDING_QUEUE_OWNER_GATED" or auth in ("HOLD", "OWNER_HOLD") or gov_state in ("HOLD", "OWNER_HOLD"))
            if is_hold:
                if status == "COMPLETED" or deploy == "DEPLOYED_ACTIVE" or op == "EXECUTED":
                    self.findings.append(Finding("P5-005", oid, "ERROR", f"Governance-held obligation '{oid}' was falsely marked active or completed without owner release (status='{status}', deploy='{deploy}')"))

        # P5-006: Evidence-backed closure
        for obl in obligations:
            oid = obl.get("obligation_id")
            status = obl.get("status")
            disp = obl.get("disposition")
            evidence = obl.get("evidence_refs", [])
            if status == "COMPLETED" or disp == "CLOSED_VERIFIED":
                if not evidence or all(not str(e).strip() for e in evidence):
                    self.findings.append(Finding("P5-006", oid, "ERROR", f"Obligation '{oid}' claims completed/closed but lacks verification evidence references"))

        # P5-007: Release truth
        for obl in obligations:
            oid = obl.get("obligation_id")
            deploy = obl.get("deployment_state")
            deploys = obl.get("deployment_refs", [])
            disp = obl.get("disposition")
            if deploy == "DEPLOYED_ACTIVE":
                if not deploys or all(not str(d).strip() for d in deploys):
                    self.findings.append(Finding("P5-007", oid, "ERROR", f"Obligation '{oid}' claims active deployment but lacks verified deployment references"))
                else:
                    held_signals = ["HELD", "UNPROMOTED", "NOT_DEPLOYED", "STALE", "PENDING"]
                    for d in deploys:
                        if any(sig in str(d).upper() for sig in held_signals):
                            self.findings.append(Finding("P5-007", oid, "ERROR", f"Obligation '{oid}' claims DEPLOYED_ACTIVE but deployment reference '{d}' indicates unpromoted/held status"))
            if deploy in ("NOT_DEPLOYED", "RELEASE_HELD") and disp == "CLOSED_VERIFIED":
                self.findings.append(Finding("P5-007", oid, "ERROR", f"Obligation '{oid}' is {deploy} but claims CLOSED_VERIFIED"))
            elif deploy == "NOT_DEPLOYED" and disp == "CLOSED_VERIFIED" and obl.get("capability_id") in ("MAINLINE_DELIVERY", "CORE_FEATURE"):
                self.findings.append(Finding("P5-007", oid, "ERROR", f"Deliverable obligation '{oid}' is NOT_DEPLOYED but claims CLOSED_VERIFIED"))

        # P5-008: Historical continuity & acyclic lineage
        seen_oids = set()
        for obl in obligations:
            oid = obl.get("obligation_id")
            if oid in seen_oids:
                self.findings.append(Finding("P5-008", oid, "ERROR", f"Duplicate obligation ID '{oid}' found in register"))
            seen_oids.add(oid)

            # Check cyclic supersession
            cur = obl
            chain = [oid]
            while cur and cur.get("superseded_by"):
                nxt_id = cur.get("superseded_by")
                if nxt_id in chain:
                    self.findings.append(Finding("P5-008", oid, "ERROR", f"Cyclic supersession detected in obligation lineage: {' -> '.join(chain + [nxt_id])}"))
                    break
                chain.append(nxt_id)
                cur = next((o for o in obligations if o.get("obligation_id") == nxt_id), None)

        # P5-009: Cross-audit completeness
        required_findings = ["DEF-056-001", "DEF-056-002", "DEF-056-003", "PR-107", "PR-110", "PR-111", "PR-112", "PR-113", "PR-115", "PR-116", "PR-117"]
        found_source_ids = {obl.get("source_finding_id") for obl in obligations}
        for rf in required_findings:
            if rf not in found_source_ids:
                self.findings.append(Finding("P5-009", None, "ERROR", f"Required cross-audit source finding '{rf}' is missing from obligation register"))

        # P5-010 Integrity: Active sprint cannot be SPRINT_5 without affirmative kickoff authorization
        active_sprint = self.data.get("active_sprint")
        s5_exec_authorized = self.data.get("sprint5_execution_authorized")
        if active_sprint == "SPRINT_5" and not s5_exec_authorized:
            self.findings.append(Finding("P5-010", None, "ERROR", "Active sprint set to 'SPRINT_5' while sprint5_execution_authorized is False"))

        # P5-011: No stale baseline substitution
        for obl in obligations:
            oid = obl.get("obligation_id")
            deploys = obl.get("deployment_refs", [])
            for dep in deploys:
                if "5efc2179" in dep:
                    self.findings.append(Finding("P5-011", oid, "ERROR", f"Obligation '{oid}' substitutes stale unverified deployment SHA '5efc2179' instead of reconciled baseline '856170c'"))

        # P5-012: External-source verification integrity
        if self.reconciliation_status == "NOT_PERFORMED" and self.require_external_inventory:
            self.findings.append(Finding("P5-012", None, "ERROR", "External reconciliation was required but execution status is NOT_PERFORMED; cannot verify external PR/issue truthfulness"))

    def _check_p5_execution_authority(self):
        """Enforces affirmative Product Owner kickoff authorization (Outcome B: P5-010)."""
        s5_exec_authorized = self.data.get("sprint5_execution_authorized")
        auth_record = self.data.get("sprint5_execution_authorization_record")

        if s5_exec_authorized is True:
            if not isinstance(auth_record, dict):
                self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution authorized flag is True but 'sprint5_execution_authorization_record' is missing or null"))
            else:
                authorizer = auth_record.get("authorizer")
                decision = auth_record.get("decision")
                ts = auth_record.get("timestamp")
                ref = auth_record.get("decision_reference")
                gov = auth_record.get("governing_directive")

                if authorizer != "PRODUCT_OWNER":
                    self.findings.append(Finding("P5-010", None, "ERROR", f"Sprint 5 execution authorization record invalid authorizer '{authorizer}' (must be 'PRODUCT_OWNER'; autonomous agents cannot authorize kickoff)"))
                if decision != "AFFIRMATIVE_KICKOFF_GRANTED":
                    self.findings.append(Finding("P5-010", None, "ERROR", f"Sprint 5 execution authorization record contradiction: authorized flag is True but record decision is '{decision}'"))
                if not ts or not str(ts).strip():
                    self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution authorization record missing valid timestamp"))
                if not ref or not str(ref).strip():
                    self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution authorization record missing valid decision_reference"))
                if not gov or not str(gov).strip():
                    self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution authorization record missing valid governing_directive"))
        else:
            self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution kickoff is NOT authorized by Product Owner. Execution remains blocked pending affirmative owner gate (reason: OWNER_AUTHORIZATION_EVIDENCE_MISSING)"))
            if isinstance(auth_record, dict):
                decision = auth_record.get("decision")
                if decision == "AFFIRMATIVE_KICKOFF_GRANTED":
                    self.findings.append(Finding("P5-010", None, "ERROR", "Sprint 5 execution authorization record contradiction: authorized flag is False but record claims 'AFFIRMATIVE_KICKOFF_GRANTED'"))

    def _check_p5_invariants(self, tasks: List[Dict[str, Any]]):
        """Complete Pre-Sprint 5 Closure Invariants (Outcome A + Outcome B) under Directives 058/059/060."""
        self._check_p5_readiness_invariants(tasks)
        self._check_p5_execution_authority()


def detect_scoped_governance_changes(
    base_ref: Optional[str] = None
) -> Tuple[bool, List[str], str]:
    """
    Detects whether canonical backlog or governance files have been modified against
    an authoritative Git base reference, not merely uncommitted working-tree changes.

    Returns:
        (is_scoped_change_detected, matching_files_list, detection_method_description)
    """
    candidate_refs = []
    if base_ref:
        candidate_refs.append(base_ref)

    env_base = os.environ.get("GITHUB_BASE_REF")
    if env_base:
        candidate_refs.append(f"origin/{env_base}")
        candidate_refs.append(env_base)

    candidate_refs.extend(["origin/main", "main", "HEAD~1"])

    changed_files = set()
    detection_method = "UNKNOWN"

    for ref in candidate_refs:
        try:
            # 1. Try 3-dot diff: changes introduced on current branch since branching from ref
            cmd = ["git", "diff", "--name-only", f"{ref}...HEAD"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.strip():
                        changed_files.add(line.strip())
                detection_method = f"git diff {ref}...HEAD"
                break

            # 2. Try 2-dot diff
            cmd = ["git", "diff", "--name-only", ref, "HEAD"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.strip():
                        changed_files.add(line.strip())
                detection_method = f"git diff {ref} HEAD"
                break
        except Exception:
            continue

    # Also inspect working tree (staged & unstaged)
    try:
        res_wt = subprocess.run(["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, timeout=3)
        if res_wt.returncode == 0:
            for line in res_wt.stdout.splitlines():
                if line.strip():
                    changed_files.add(line.strip())

        res_cached = subprocess.run(["git", "diff", "--name-only", "--cached"], capture_output=True, text=True, timeout=3)
        if res_cached.returncode == 0:
            for line in res_cached.stdout.splitlines():
                if line.strip():
                    changed_files.add(line.strip())
    except Exception:
        pass

    if not changed_files and detection_method == "UNKNOWN":
        return True, ["GIT_UNAVAILABLE_FAIL_SAFE"], "FAIL_SAFE_FALLBACK"

    matched_governance_files = []
    for f in changed_files:
        if any(pat in f for pat in GOVERNANCE_PATH_PATTERNS):
            matched_governance_files.append(f)

    return (len(matched_governance_files) > 0), matched_governance_files, detection_method


def extract_git_baseline(base_ref: Optional[str] = None, file_path: str = "docs/planning/arx_canonical_backlog.json") -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Attempts to retrieve the authoritative historical baseline directly from Git.
    """
    candidate_refs = []
    if base_ref:
        candidate_refs.append(base_ref)

    env_base = os.environ.get("GITHUB_BASE_REF")
    if env_base:
        candidate_refs.append(f"origin/{env_base}")
        candidate_refs.append(env_base)

    candidate_refs.extend(["origin/main", "main"])

    for ref in candidate_refs:
        try:
            cmd = ["git", "show", f"{ref}:{file_path}"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                return data, f"git show {ref}:{file_path}"
        except Exception:
            continue

    return None, "NO_GIT_BASELINE_FOUND"


def main():
    parser = argparse.ArgumentParser(description="ARX Terminal Backlog Integrity Validator (GOV-001)")
    parser.add_argument("--backlog", default="docs/planning/arx_canonical_backlog.json", help="Path to canonical backlog JSON")
    parser.add_argument("--schema", default="docs/planning/arx_canonical_backlog.schema.json", help="Path to JSON Schema")
    parser.add_argument("--prs-inventory", default=None, help="Path to GitHub PR inventory JSON")
    parser.add_argument("--issues-inventory", default=None, help="Path to GitHub Issues inventory JSON")
    parser.add_argument("--baseline", default=None, help="Path to explicit baseline backlog JSON")
    parser.add_argument("--base-ref", default=None, help="Authoritative Git base ref (e.g. origin/main)")
    parser.add_argument("--stage", choices=["report-only", "scoped", "strict"], default="scoped", help="Progressive enforcement stage")
    parser.add_argument("--strict", action="store_true", help="Fail closed on warnings as well as errors")
    parser.add_argument("--closure-certification", action="store_true", help="Enforce full closure certification requirements (cannot be masked by report-only)")
    parser.add_argument("--pre-sprint5-readiness", action="store_true", help="Enforce Technical Closure Readiness (Outcome A: P5-001 to P5-009, P5-011, P5-012; does not require affirmative PO kickoff authorization)")
    parser.add_argument("--pre-sprint5-closure", action="store_true", help="Enforce Complete Pre-Sprint 5 Closure & Execution Permission (Outcome B: Technical Readiness + Affirmative PO Kickoff Authorization)")
    parser.add_argument("--require-baseline", action="store_true", help="Fail closed if historical baseline is unavailable")
    parser.add_argument("--require-external-inventory", action="store_true", help="Fail closed if external GitHub inventory is missing")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args()

    # 1. Load canonical backlog (Fail closed on missing source)
    if not os.path.isfile(args.backlog):
        print(f"SOURCE_UNAVAILABLE: Backlog file not found at '{args.backlog}'", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.backlog, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
    except Exception as e:
        print(f"SOURCE_UNAVAILABLE: Failed to parse backlog JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # 2. Load Schema
    schema_data = None
    if args.schema:
        if not os.path.isfile(args.schema):
            if args.schema != "docs/planning/arx_canonical_backlog.schema.json":
                print(f"SOURCE_UNAVAILABLE: Specified schema file not found at '{args.schema}'", file=sys.stderr)
                sys.exit(2)
        else:
            try:
                with open(args.schema, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
            except Exception as e:
                print(f"SOURCE_UNAVAILABLE: Failed to parse schema JSON: {e}", file=sys.stderr)
                sys.exit(2)

    # 3. Load Inventories (Fail closed if explicitly requested path is missing)
    prs_inventory = None
    if args.prs_inventory is not None:
        if not os.path.isfile(args.prs_inventory):
            print(f"SOURCE_UNAVAILABLE: Specified PRs inventory file not found at '{args.prs_inventory}'", file=sys.stderr)
            sys.exit(2)
        try:
            with open(args.prs_inventory, "r", encoding="utf-8") as f:
                prs_inventory = json.load(f)
        except Exception as e:
            print(f"SOURCE_UNAVAILABLE: Failed to parse PRs inventory JSON: {e}", file=sys.stderr)
            sys.exit(2)

    issues_inventory = None
    if args.issues_inventory is not None:
        if not os.path.isfile(args.issues_inventory):
            print(f"SOURCE_UNAVAILABLE: Specified issues inventory file not found at '{args.issues_inventory}'", file=sys.stderr)
            sys.exit(2)
        try:
            with open(args.issues_inventory, "r", encoding="utf-8") as f:
                issues_inventory = json.load(f)
        except Exception as e:
            print(f"SOURCE_UNAVAILABLE: Failed to parse issues inventory JSON: {e}", file=sys.stderr)
            sys.exit(2)

    # 4. Load Baseline (Explicit file or Git show)
    baseline_data = None
    baseline_source_desc = "NONE"
    if args.baseline is not None:
        if not os.path.isfile(args.baseline):
            print(f"SOURCE_UNAVAILABLE: Specified baseline file not found at '{args.baseline}'", file=sys.stderr)
            sys.exit(2)
        try:
            with open(args.baseline, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            baseline_source_desc = f"file: {args.baseline}"
        except Exception as e:
            print(f"SOURCE_UNAVAILABLE: Failed to parse baseline JSON: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        git_baseline, desc = extract_git_baseline(args.base_ref, args.backlog)
        if git_baseline:
            baseline_data = git_baseline
            baseline_source_desc = desc
        else:
            baseline_source_desc = "NONE (Initial backlog creation or baseline not present on base branch)"

    # 5. Detect Scoped Governance Changes against Authoritative Git Base
    scoped_detected, matched_files, detection_desc = detect_scoped_governance_changes(args.base_ref)

    # 6. Instantiate & Validate
    validator = BacklogValidator(
        backlog_data,
        prs_inventory=prs_inventory,
        issues_inventory=issues_inventory,
        baseline_data=baseline_data,
        schema_data=schema_data,
        require_baseline=(args.require_baseline or args.closure_certification or args.pre_sprint5_closure or args.pre_sprint5_readiness),
        require_external_inventory=(args.require_external_inventory or args.closure_certification),
        require_closure=args.closure_certification,
        pre_sprint5_closure=args.pre_sprint5_closure,
        pre_sprint5_readiness=args.pre_sprint5_readiness,
    )
    verdict, findings = validator.validate_all()

    # 7. Determine Progressive Enforcement & Exit Code
    has_errors = any(f.severity == "ERROR" for f in findings)
    effective_stage = "strict" if (args.strict or args.closure_certification) else args.stage

    if args.pre_sprint5_closure or args.pre_sprint5_readiness:
        exit_code = 0 if verdict == "PASS" else 1
    elif args.closure_certification:
        exit_code = 0 if verdict == "PASS" else 1
    elif effective_stage == "report-only":
        exit_code = 0
    elif effective_stage == "scoped":
        if scoped_detected:
            exit_code = 0 if verdict in ("PASS", "PASS_WITH_WARNINGS") else 1
        else:
            exit_code = 0
    else:  # strict
        if args.strict:
            exit_code = 0 if verdict == "PASS" else 1
        else:
            exit_code = 0 if verdict in ("PASS", "PASS_WITH_WARNINGS") else 1

    # 8. Diagnostics Emission
    if args.json:
        out = {
            "verdict": verdict,
            "stage": effective_stage,
            "exit_code": exit_code,
            "scoped_governance_change_detected": scoped_detected,
            "matched_governance_files": matched_files,
            "git_diff_method": detection_desc,
            "total_tasks": len(backlog_data.get("tasks", [])),
            "schema_validation": {
                "status": validator.schema_status,
                "schema_path": args.schema if schema_data else None,
            },
            "historical_continuity": {
                "status": validator.continuity_status,
                "baseline_source": baseline_source_desc,
                "note": "Missing historical baseline never proves task continuity" if validator.continuity_status == "UNVERIFIED" else "Verified against baseline"
            },
            "external_reconciliation": {
                "status": validator.reconciliation_status,
                "pull_requests_evaluated": validator.reconciliation_counts["pull_requests"],
                "note": "Deterministic local run; missing inventory never means zero issues/PRs" if validator.reconciliation_status == "NOT_PERFORMED" else "Reconciled against provided inventory"
            },
            "total_findings": len(findings),
            "errors": [f.to_dict() for f in findings if f.severity == "ERROR"],
            "warnings": [f.to_dict() for f in findings if f.severity == "WARNING"],
        }
        print(json.dumps(out, indent=2))
    else:
        print("===========================================================")
        print("ARX TERMINAL — BACKLOG INTEGRITY VALIDATOR (GOV-001)")
        print(f"File: {args.backlog} | Tasks: {len(backlog_data.get('tasks', []))}")
        print(f"Stage: {effective_stage} | Scope Detected: {scoped_detected}")
        print(f"Git Diff Base: {detection_desc}")
        if matched_files:
            print(f"Governance Files Modified ({len(matched_files)}): {', '.join(matched_files[:3])}{'...' if len(matched_files) > 3 else ''}")
        print(f"Schema Validation: {validator.schema_status}")
        print(f"Historical Continuity: {validator.continuity_status} ({baseline_source_desc})")
        print(f"External Reconciliation: {validator.reconciliation_status}")
        print(f"Verdict: {verdict} | Exit Code: {exit_code}")
        print("===========================================================")
        for f in findings:
            print(str(f))
        if not findings:
            print("All rules (BI-000 to BI-012) verified with zero errors or warnings.")

        # Multi-Gate Enforcement Readout
        if not args.pre_sprint5_closure and not args.pre_sprint5_readiness:
            val_a = BacklogValidator(
                backlog_data,
                prs_inventory=prs_inventory,
                issues_inventory=issues_inventory,
                baseline_data=baseline_data,
                schema_data=schema_data,
                require_baseline=True,
                pre_sprint5_readiness=True,
            )
            _, ready_findings = val_a.validate_all()
            ready_errors = [f for f in ready_findings if f.severity == "ERROR"]

            val_b = BacklogValidator(
                backlog_data,
                prs_inventory=prs_inventory,
                issues_inventory=issues_inventory,
                baseline_data=baseline_data,
                schema_data=schema_data,
                require_baseline=True,
                pre_sprint5_closure=True,
            )
            _, exec_findings = val_b.validate_all()
            exec_errors = [f for f in exec_findings if f.severity == "ERROR"]

            print("-----------------------------------------------------------")
            print("MULTI-GATE ENFORCEMENT SUMMARY:")
            print(f"  Gate 1 (Governance Integration Gate): {verdict} (Exit Code: {exit_code}) [PR MERGE ELIGIBLE]")
            if ready_errors:
                print(f"  Gate 2A (Technical Closure Readiness): BLOCKED ({len(ready_errors)} active blockers)")
                for re in ready_errors:
                    print(f"    * {str(re)}")
            else:
                print("  Gate 2A (Technical Closure Readiness): PASS [TECHNICAL CRITERIA SATISFIED]")

            if exec_errors:
                print(f"  Gate 2B (Sprint 5 Execution Permission): BLOCKED ({len(exec_errors)} active blockers)")
                for ee in exec_errors:
                    print(f"    * {str(ee)}")
            else:
                print("  Gate 2B (Sprint 5 Execution Permission): ELIGIBLE [AFFIRMATIVE KICKOFF GRANTED]")
            print("===========================================================")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
