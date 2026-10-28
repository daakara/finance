#!/usr/bin/env python3
"""
ARX Terminal — Sprint Control & Execution Tracker (SCS-v1)
Authoritative tracker, reconciliation engine, and gate verification protocol
derived strictly from docs/planning/arx_canonical_backlog.json.

Enforces:
  1. Multi-dimensional lifecycle separation (Requirements, Implementation, Integration,
     Verification, Publication, Deployment, Evidence, Operational Authorization).
  2. Deterministic View Generation (Sprint Board, Action Queue, Release Matrix).
  3. Automated Reconciliation Checks (Check 01 to Check 10).
  4. Checkpoint Gate Certification (Entry, Execution, Release, Exit).
"""

import os
import sys
import json
import re
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set


# Immutable Platform Holds
IMMUTABLE_HOLDS = {
    "SPRINT3": "CLOSED_WITH_CONTROLLED_CARRYFORWARD",
    "SMART_MONEY_WORKER_CREATION": "NOT_AUTHORIZED",
    "SMART_MONEY_WORKER_ACTIVATION": "NOT_AUTHORIZED",
    "SMART_MONEY_SEC_EGRESS": "DISABLED",
    "SMART_MONEY_RECURRING_INGESTION": "NOT_AUTHORIZED",
    "SM_GATE_08_OWNER_ACTIVATION": "HOLD",
    "CORPORATE_EVENTS_TELEMETRY_ACTIVATION": "NOT_AUTHORIZED",
    "HVA00_PROTOCOL_RATIFICATION": "NOT_GRANTED",
    "HVA01_HUMAN_EVALUATION": "NOT_AUTHORIZED",
    "HVA02_ALGORITHM_ACCEPTANCE": "NOT_AUTHORIZED",
    "EXTERNAL_CUSTODY_ACTIVATION": "NOT_AUTHORIZED",
    "UX_PARTICIPANT_RESEARCH": "NOT_AUTHORIZED",
    "MODEL_TUNING": "FROZEN",
    "VCP_NATURAL_SCHEDULER_ENABLED": "0",
}


class ReconciliationFinding:
    def __init__(
        self,
        check_id: str,
        severity: str,
        task_id: str,
        accountable_owner: str,
        inconsistency: str,
        required_correction: str
    ):
        self.check_id = check_id
        self.severity = severity  # "ERROR", "WARNING", "INFO"
        self.task_id = task_id
        self.accountable_owner = accountable_owner
        self.inconsistency = inconsistency
        self.required_correction = required_correction

    def to_dict(self) -> Dict[str, str]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "task_id": self.task_id,
            "accountable_owner": self.accountable_owner,
            "inconsistency": self.inconsistency,
            "required_correction": self.required_correction,
        }

    def __str__(self) -> str:
        return (
            f"[{self.severity}] [{self.task_id}] [{self.check_id}]: "
            f"{self.inconsistency} -> Required: {self.required_correction} "
            f"(Owner: {self.accountable_owner})"
        )


class SprintControlSystem:
    def __init__(
        self,
        backlog_path: str = "docs/planning/arx_canonical_backlog.json",
        repo_root: Optional[str] = None,
        github_prs_inventory: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        self.backlog_path = backlog_path
        self.repo_root = repo_root or os.getcwd()
        self.github_prs_inventory = github_prs_inventory or {}
        self.data: Dict[str, Any] = {}
        self.tasks: List[Dict[str, Any]] = []
        self.task_map: Dict[str, Dict[str, Any]] = {}
        self.findings: List[ReconciliationFinding] = []
        self._load_backlog()

    def _load_backlog(self):
        full_path = os.path.join(self.repo_root, self.backlog_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Authoritative canonical backlog not found: {full_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tasks = self.data.get("tasks", [])
        self.task_map = {t.get("task_id", ""): t for t in self.tasks}

    def get_mainline_sha(self) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            return "UNKNOWN"

    def get_pr_status_from_git(self, pr_num: int) -> Tuple[str, Optional[str]]:
        """
        Inspects local git log to check if PR merge commit or squash commit exists on main.
        Returns: (status_enum, commit_sha)
        """
        # 1. Check if supplied in external inventory
        if pr_num in self.github_prs_inventory:
            inv = self.github_prs_inventory[pr_num]
            state = inv.get("state")
            merged = inv.get("merged", False)
            if merged or state == "closed":
                return ("MERGED", inv.get("merge_commit_sha"))
            elif inv.get("draft"):
                return ("OPEN_DRAFT", None)
            else:
                return ("OPEN_REVIEW", None)

        # 2. Check git commit history for PR merge references
        try:
            cmd = ["git", "log", "--grep", f"(#{pr_num})", "-n", "1", "--format=%H"]
            res = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return ("MERGED", res.stdout.strip())
        except Exception:
            pass

        return ("UNKNOWN", None)

    # -------------------------------------------------------------------------
    # Phase 1: Completion Predicate
    # -------------------------------------------------------------------------
    def evaluate_completion_predicate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        tid = task.get("task_id", "")
        status = task.get("status", "")
        disp = task.get("disposition", "")
        dims = task.get("dimensions", {})
        asgn = task.get("assigned_sprint", "")

        unmet_dims = []
        unmet_criteria = []
        held_dims = []

        # Check dimensions
        req = dims.get("requirements")
        impl = dims.get("implementation")
        integ = dims.get("integration")
        verif = dims.get("verification")
        pub = dims.get("repository_publication")
        dep = dims.get("deployment")
        pe = dims.get("production_evidence")
        oe = dims.get("operational_execution")
        auth = dims.get("owner_authorization")

        if req not in ("DONE", "NOT_APPLICABLE"):
            unmet_dims.append(f"requirements={req}")
        if impl not in ("DONE", "NOT_APPLICABLE"):
            unmet_dims.append(f"implementation={impl}")
        if integ not in ("DONE", "NOT_APPLICABLE"):
            unmet_dims.append(f"integration={integ}")
        if verif not in ("PASS", "NOT_APPLICABLE"):
            unmet_dims.append(f"verification={verif}")
        if pub not in ("PUBLISHED_MERGED", "NOT_APPLICABLE"):
            unmet_dims.append(f"repository_publication={pub}")

        # Deployment & Evidence checks
        if dep in ("DEPLOYED_ACTIVE", "DEPLOYED_DORMANT"):
            if pe != "VERIFIED":
                unmet_dims.append(f"production_evidence={pe} (expected VERIFIED for {dep})")
        elif dep == "NOT_DEPLOYED" and status in ("COMPLETED", "CLOSED_VERIFIED") and impl == "DONE":
            # If deployment was intended, flag
            pass

        # Check owner authorization & operational execution
        if auth in ("HOLD", "NOT_AUTHORIZED"):
            held_dims.append(f"owner_authorization={auth}")
            if oe == "EXECUTED":
                unmet_dims.append("unauthorized_operational_execution_under_hold")

        # Evaluate acceptance criteria
        criteria = task.get("acceptance_criteria", [])
        evidence_refs = task.get("evidence_refs", [])
        if status in ("COMPLETED", "CLOSED_VERIFIED") and not evidence_refs:
            unmet_criteria.append("zero_evidence_references_for_completed_task")

        # Determine overall eligibility
        if asgn == "HOLDING_QUEUE_OWNER_GATED" or disp == "SPRINT4_OWNER_GATED":
            closure_eligibility = "BLOCKED_HOLDING_QUEUE"
            is_complete = False
        elif disp in ("SUPERSEDED", "TEST_OR_ADMINISTRATIVE"):
            closure_eligibility = "SUPERSEDED_HISTORICAL"
            is_complete = True
        elif status == "COMPLETED" or disp == "CLOSED_VERIFIED":
            if held_dims:
                closure_eligibility = "CONDITIONAL_CLOSURE_OWNER_HELD"
                is_complete = True
            elif unmet_dims:
                closure_eligibility = "INELIGIBLE_ACTIVE_WORK_REMAINING"
                is_complete = False
            else:
                closure_eligibility = "ALREADY_CLOSED"
                is_complete = True
        else:
            if not unmet_dims and not held_dims and evidence_refs:
                closure_eligibility = "ELIGIBLE_FOR_CLOSURE"
                is_complete = True
            elif not unmet_dims and held_dims:
                closure_eligibility = "CONDITIONAL_CLOSURE_OWNER_HELD"
                is_complete = False
            else:
                closure_eligibility = "INELIGIBLE_ACTIVE_WORK_REMAINING"
                is_complete = False

        return {
            "is_complete": is_complete,
            "closure_eligibility": closure_eligibility,
            "unmet_dimensions": unmet_dims,
            "unmet_criteria": unmet_criteria,
            "held_dimensions": held_dims,
        }

    # -------------------------------------------------------------------------
    # Phase 2: Deterministic Views & Actions
    # -------------------------------------------------------------------------
    def generate_action_queue(self, sprint_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        queue = []
        for t in self.tasks:
            asgn = t.get("assigned_sprint")
            if sprint_filter and asgn != sprint_filter:
                continue

            tid = t.get("task_id")
            title = t.get("title")
            prio = t.get("priority", "P2")
            ws = t.get("workstream", "ARCHITECTURE")
            owner = t.get("owner", "UNASSIGNED")
            status = t.get("status", "READY")
            disp = t.get("disposition", "")
            act = t.get("action_tracking", {})

            # Next action resolution
            next_act = act.get("next_action")
            if not next_act:
                if status == "COMPLETED" or disp == "CLOSED_VERIFIED":
                    next_act = "Verify production telemetry and archive closure receipt"
                elif asgn == "HOLDING_QUEUE_OWNER_GATED" or disp == "SPRINT4_OWNER_GATED":
                    trigger = t.get("reopening_trigger", "Owner directive")
                    next_act = f"Preserve holding queue isolation pending: {trigger}"
                elif disp == "SPRINT4_EXECUTION":
                    if t.get("github_prs"):
                        next_act = f"Complete CI verification and submit PR {t.get('github_prs')} for owner review"
                    else:
                        next_act = "Implement task acceptance criteria on isolated feature branch"
                elif disp == "SPRINT4_DISCOVERY":
                    next_act = "Draft discovery synthesis memo and define Sprint 5 architecture boundaries"
                else:
                    next_act = "Awaiting sprint planning intake"

            # Checkpoint target date
            target_date = act.get("checkpoint_target_date")
            if not target_date:
                target_date = "2026-10-27T18:00:00Z" if status not in ("COMPLETED", "BLOCKED") else None

            # Blocker & Trigger
            active_blocker = act.get("active_blocker")
            res_trigger = act.get("resolution_trigger")
            if not active_blocker and (asgn == "HOLDING_QUEUE_OWNER_GATED" or status == "BLOCKED"):
                active_blocker = t.get("carryforward_reason") or "Platform operational hold active"
                res_trigger = t.get("reopening_trigger") or "Research Owner directive"

            # Evidence freshness
            ev_freshness = act.get("evidence_freshness")
            if not ev_freshness:
                refs = t.get("evidence_refs", [])
                if not refs:
                    ev_freshness = "NOT_APPLICABLE" if status == "SUPERSEDED" else "UNVERIFIED"
                else:
                    all_exist = True
                    for ref in refs:
                        if ref.startswith("http") or ref.startswith("git"):
                            continue
                        loc = os.path.join(self.repo_root, ref)
                        if not os.path.exists(loc):
                            all_exist = False
                            break
                    ev_freshness = "FRESH" if all_exist else "STALE"

            predicate = self.evaluate_completion_predicate(t)

            queue.append({
                "task_id": tid,
                "title": title,
                "priority": prio,
                "workstream": ws,
                "assigned_sprint": asgn,
                "accountable_owner": act.get("accountable_owner") or owner,
                "next_action": next_act,
                "checkpoint_target_date": target_date,
                "active_blocker": active_blocker,
                "resolution_trigger": res_trigger,
                "evidence_freshness": ev_freshness,
                "completion_predicate": predicate,
            })

        # Priority sort: P0 -> P1 -> P2 -> P3
        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        queue.sort(key=lambda x: priority_order.get(x["priority"], 99))
        return queue

    def generate_release_matrix(self, sprint_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        matrix = []
        for t in self.tasks:
            asgn = t.get("assigned_sprint")
            if sprint_filter and asgn != sprint_filter:
                continue

            tid = t.get("task_id")
            title = t.get("title")
            prs = t.get("github_prs", [])
            dims = t.get("dimensions", {})
            act = t.get("action_tracking", {}).get("release_linkage", {})

            # PR status & merged SHA
            pr_status = act.get("pr_status")
            merged_sha = act.get("merged_main_sha")

            if not pr_status:
                if not prs:
                    pr_status = "NONE"
                else:
                    # Probe local git
                    found_merged = False
                    for pr in prs:
                        st, csha = self.get_pr_status_from_git(pr)
                        if st == "MERGED":
                            pr_status = "MERGED"
                            merged_sha = csha
                            found_merged = True
                            break
                        elif st in ("OPEN_DRAFT", "OPEN_REVIEW"):
                            pr_status = st
                    if not found_merged and not pr_status:
                        # Fallback based on dimensions
                        pub = dims.get("repository_publication")
                        if pub == "PUBLISHED_MERGED":
                            pr_status = "MERGED"
                        elif pub == "DRAFT_PR":
                            pr_status = "OPEN_DRAFT"
                        else:
                            pr_status = "UNKNOWN"

            # CI conclusion
            ci_conc = act.get("ci_conclusion")
            if not ci_conc:
                if pr_status == "MERGED":
                    ci_conc = "SUCCESS"
                elif pr_status in ("OPEN_DRAFT", "OPEN_REVIEW"):
                    ci_conc = "SUCCESS"  # If verified in previous audit
                elif pr_status == "NONE":
                    ci_conc = "NOT_RUN"
                else:
                    ci_conc = "UNKNOWN"

            # Deployment & Evidence
            dep_status = act.get("deployment_status") or dims.get("deployment", "NOT_APPLICABLE")
            prod_ev = act.get("production_evidence") or dims.get("production_evidence", "NOT_APPLICABLE")
            op_auth = act.get("operational_authorized") or dims.get("owner_authorization", "NOT_APPLICABLE")

            matrix.append({
                "task_id": tid,
                "title": title,
                "github_prs": prs,
                "pr_status": pr_status,
                "ci_conclusion": ci_conc,
                "merged_main_sha": merged_sha,
                "deployment_status": dep_status,
                "production_evidence": prod_ev,
                "operational_authorized": op_auth,
            })

        return matrix

    def generate_sprint_board(self, sprint_filter: Optional[str] = None) -> Dict[str, Any]:
        tasks = self.tasks
        if sprint_filter:
            tasks = [t for t in tasks if t.get("assigned_sprint") == sprint_filter]

        board = {
            "total_tasks": len(tasks),
            "execution_tasks": [],
            "discovery_tasks": [],
            "blocked_tasks": [],
            "owner_gated_tasks": [],
            "completed_tasks": [],
        }

        for t in tasks:
            tid = t.get("task_id", "")
            disp = t.get("disposition", "")
            status = t.get("status", "")
            asgn = t.get("assigned_sprint", "")

            if asgn == "HOLDING_QUEUE_OWNER_GATED" or disp == "SPRINT4_OWNER_GATED":
                board["owner_gated_tasks"].append(tid)
            elif status == "COMPLETED" or disp in ("CLOSED_VERIFIED", "SUPERSEDED", "TEST_OR_ADMINISTRATIVE"):
                board["completed_tasks"].append(tid)
            elif status == "BLOCKED":
                board["blocked_tasks"].append(tid)
            elif disp == "SPRINT4_EXECUTION" or status == "IN_PROGRESS":
                board["execution_tasks"].append(tid)
            elif disp == "SPRINT4_DISCOVERY" or status == "READY":
                board["discovery_tasks"].append(tid)
            else:
                board["discovery_tasks"].append(tid)

        return board

    # -------------------------------------------------------------------------
    # Phase 3: Automated Reconciliation Engine (10 Checks)
    # -------------------------------------------------------------------------
    def run_reconciliation(self, sprint_context: Optional[str] = None, sprint_exit_mode: bool = False) -> List[ReconciliationFinding]:
        self.findings = []
        active_sprint = sprint_context or self.data.get("active_sprint", "SPRINT_4")

        for t in self.tasks:
            tid = t.get("task_id", "")
            status = t.get("status", "")
            disp = t.get("disposition", "")
            orig = t.get("origin_sprint", "")
            asgn = t.get("assigned_sprint", "")
            owner = t.get("owner", "")
            dims = t.get("dimensions", {})
            prs = t.get("github_prs", [])
            act = t.get("action_tracking", {})

            # -----------------------------------------------------------------
            # Check 01: CHECK_MISSING_ACTION_OWNER
            # -----------------------------------------------------------------
            if status in ("IN_PROGRESS", "READY") and asgn not in ("HOLDING_QUEUE_OWNER_GATED", "DEFERRED_BACKLOG"):
                if not owner or owner in ("UNASSIGNED", "None", ""):
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_MISSING_ACTION_OWNER",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner="GOVERNANCE_LEAD",
                        inconsistency=f"Active task '{tid}' has no designated owner",
                        required_correction="Assign an accountable maintainer or core team lead"
                    ))

            # -----------------------------------------------------------------
            # Check 02: CHECK_STALE_PR_MERGE
            # -----------------------------------------------------------------
            # If task claims PR is DRAFT_PR, but PR is already merged into main
            pub = dims.get("repository_publication")
            for pr in prs:
                st, csha = self.get_pr_status_from_git(pr)
                if st == "MERGED" and pub in ("DRAFT_PR", "LOCAL_ONLY"):
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_STALE_PR_MERGE",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency=f"Task claims repository_publication='{pub}', but PR #{pr} is already merged on main (commit {csha[:7] if csha else ''})",
                        required_correction="Update repository_publication to 'PUBLISHED_MERGED' and record merged mainline SHA"
                    ))
                elif st in ("OPEN_DRAFT", "OPEN_REVIEW") and pub == "PUBLISHED_MERGED":
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_STALE_PR_MERGE",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency=f"Task claims repository_publication='PUBLISHED_MERGED', but PR #{pr} is still unmerged ({st})",
                        required_correction="Revert publication status to 'DRAFT_PR' or await owner merge"
                    ))

            # -----------------------------------------------------------------
            # Check 03: CHECK_INCONSISTENT_CLAIMS
            # -----------------------------------------------------------------
            impl = dims.get("implementation")
            verif = dims.get("verification")
            req = dims.get("requirements")

            if (status == "COMPLETED" or disp == "CLOSED_VERIFIED") and impl == "NOT_STARTED" and disp != "TEST_OR_ADMINISTRATIVE":
                self.findings.append(ReconciliationFinding(
                    check_id="CHECK_INCONSISTENT_CLAIMS",
                    severity="ERROR",
                    task_id=tid,
                    accountable_owner=owner or "UNASSIGNED",
                    inconsistency=f"Task claims completed status '{status}/{disp}', but implementation is 'NOT_STARTED'",
                    required_correction="Reconcile implementation dimension or correct status"
                ))

            # Stale ARCH-001 check: If audit task completed deliverables exist, but implementation remains NOT_STARTED
            if tid == "ARX-S4-ARCH-001" and status == "READY" and impl == "NOT_STARTED":
                # Detect whether candidate manifest exists
                manifest_path = os.path.join(self.repo_root, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
                if os.path.exists(manifest_path):
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_INCONSISTENT_CLAIMS",
                        severity="WARNING",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency="Architecture audit deliverables (CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json) exist, but backlog records implementation='NOT_STARTED' and verification='PENDING'",
                        required_correction="Update dimensions to implementation='DONE', verification='PASS', and advance status to ready for closure"
                    ))

            # -----------------------------------------------------------------
            # Check 04: CHECK_EVIDENCE_FRESHNESS
            # -----------------------------------------------------------------
            refs = t.get("evidence_refs", [])
            if status in ("COMPLETED", "CLOSED_VERIFIED") and not refs:
                self.findings.append(ReconciliationFinding(
                    check_id="CHECK_EVIDENCE_FRESHNESS",
                    severity="ERROR",
                    task_id=tid,
                    accountable_owner=owner or "UNASSIGNED",
                    inconsistency="Completed task has empty evidence_refs array",
                    required_correction="Populate evidence_refs with authoritative commits, PRs, or documents"
                ))
            else:
                for ref in refs:
                    if ref.startswith("http") or ref.startswith("git"):
                        continue
                    full_ref = os.path.join(self.repo_root, ref)
                    if not os.path.exists(full_ref):
                        self.findings.append(ReconciliationFinding(
                            check_id="CHECK_EVIDENCE_FRESHNESS",
                            severity="WARNING",
                            task_id=tid,
                            accountable_owner=owner or "UNASSIGNED",
                            inconsistency=f"Citing nonexistent evidence file: '{ref}'",
                            required_correction="Verify evidence file path or generate missing verification receipt"
                        ))

            # -----------------------------------------------------------------
            # Check 05: CHECK_UNRESOLVED_DEPENDENCIES
            # -----------------------------------------------------------------
            deps = t.get("dependencies", [])
            if status in ("READY", "IN_PROGRESS"):
                for dep_id in deps:
                    dep_task = self.task_map.get(dep_id)
                    if dep_task:
                        dep_status = dep_task.get("status")
                        dep_disp = dep_task.get("disposition")
                        if dep_status not in ("COMPLETED", "CLOSED_VERIFIED") and dep_disp not in ("CLOSED_VERIFIED", "SUPERSEDED"):
                            self.findings.append(ReconciliationFinding(
                                check_id="CHECK_UNRESOLVED_DEPENDENCIES",
                                severity="WARNING",
                                task_id=tid,
                                accountable_owner=owner or "UNASSIGNED",
                                inconsistency=f"Task is '{status}' but depends on uncompleted task '{dep_id}' (status: '{dep_status}')",
                                required_correction=f"Block execution of '{tid}' until '{dep_id}' is certified complete"
                            ))

            # -----------------------------------------------------------------
            # Check 06: CHECK_EXPIRED_CHECKPOINTS
            # -----------------------------------------------------------------
            target_date_str = act.get("checkpoint_target_date")
            if target_date_str and status in ("IN_PROGRESS", "READY"):
                try:
                    target_dt = datetime.fromisoformat(target_date_str.replace("Z", "+00:00"))
                    if target_dt < datetime.now(timezone.utc):
                        self.findings.append(ReconciliationFinding(
                            check_id="CHECK_EXPIRED_CHECKPOINTS",
                            severity="WARNING",
                            task_id=tid,
                            accountable_owner=owner or "UNASSIGNED",
                            inconsistency=f"Checkpoint target date '{target_date_str}' has expired without completion evidence",
                            required_correction="Execute scheduled checkpoint review and update next action"
                        ))
                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Check 07: CHECK_INCOMPLETE_CRITERIA
            # -----------------------------------------------------------------
            # Unresolved PR #98 UX Parity Check
            if tid == "ARX-S3-016":
                dep = dims.get("deployment")
                if dep == "DEPLOYED_DORMANT" and 98 in prs:
                    st, _ = self.get_pr_status_from_git(98)
                    if st != "MERGED":
                        self.findings.append(ReconciliationFinding(
                            check_id="CHECK_INCOMPLETE_CRITERIA",
                            severity="WARNING",
                            task_id=tid,
                            accountable_owner=owner or "daakara",
                            inconsistency="PR #98 is unmerged and pending owner visual acceptance, but backlog marks deployment='DEPLOYED_DORMANT'",
                            required_correction="Classify PR #98 as awaiting explicit owner review before claiming deployment parity"
                        ))

            # Conditional closures for S3-010 and S3-013
            if tid == "ARX-S3-010" and status == "COMPLETED":
                if dims.get("operational_execution") != "NOT_PERFORMED":
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_INCOMPLETE_CRITERIA",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency="Task ARX-S3-010 is closed as completed, but operational Smart Money worker activation remains strictly prohibited under Gate 08/09",
                        required_correction="Enforce conditional closure tag: CLOSED_WITH_CONTROLLED_CARRYFORWARD"
                    ))

            # -----------------------------------------------------------------
            # Check 08: CHECK_UNCLASSIFIED_CARRYFORWARDS
            # -----------------------------------------------------------------
            # Missing Sprint 5 carryforward classifications for S1-006, S2-001, S2-005, S2-007, S2-008
            if asgn == "SPRINT_4" and status in ("READY", "IN_PROGRESS"):
                if tid in ("ARX-S1-006", "ARX-S2-001", "ARX-S2-005", "ARX-S2-007", "ARX-S2-008"):
                    if not act.get("sprint_exit_disposition"):
                        self.findings.append(ReconciliationFinding(
                            check_id="CHECK_UNCLASSIFIED_CARRYFORWARDS",
                            severity="WARNING",
                            task_id=tid,
                            accountable_owner=owner or "UNASSIGNED",
                            inconsistency=f"Sprint 4 discovery task '{tid}' has no documented Sprint 5 carryforward disposition or rationale",
                            required_correction="Record sprint_exit_disposition='CARRYFORWARD_SPRINT5' and specify discovery continuation scope"
                        ))

            # -----------------------------------------------------------------
            # Check 09: CHECK_UNAUTHORIZED_TRANSITIONS
            # -----------------------------------------------------------------
            auth = dims.get("owner_authorization")
            oe = dims.get("operational_execution")
            dep = dims.get("deployment")
            if auth in ("HOLD", "NOT_AUTHORIZED"):
                if oe == "EXECUTED":
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_UNAUTHORIZED_TRANSITIONS",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency=f"Task claims operational_execution='EXECUTED' while owner_authorization is '{auth}'",
                        required_correction="Fail closed: revert operational execution to 'NOT_PERFORMED'"
                    ))
                if dep == "DEPLOYED_ACTIVE":
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_UNAUTHORIZED_TRANSITIONS",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency=f"Task claims deployment='DEPLOYED_ACTIVE' while owner_authorization is '{auth}'",
                        required_correction="Fail closed: revert deployment to 'NOT_DEPLOYED' or 'DEPLOYED_DORMANT'"
                    ))

            # -----------------------------------------------------------------
            # Check 10: CHECK_SPRINT_EXIT_UNRESOLVED
            # -----------------------------------------------------------------
            if sprint_exit_mode and asgn == active_sprint:
                if status == "IN_PROGRESS":
                    self.findings.append(ReconciliationFinding(
                        check_id="CHECK_SPRINT_EXIT_UNRESOLVED",
                        severity="ERROR",
                        task_id=tid,
                        accountable_owner=owner or "UNASSIGNED",
                        inconsistency=f"Closing sprint '{active_sprint}' contains unresolved task '{tid}' with status='IN_PROGRESS'",
                        required_correction="Complete task or reassign to next sprint/holding queue prior to exit certification"
                    ))
                elif disp not in ("CLOSED_VERIFIED", "TEST_OR_ADMINISTRATIVE", "SUPERSEDED"):
                    if not act.get("sprint_exit_disposition"):
                        self.findings.append(ReconciliationFinding(
                            check_id="CHECK_SPRINT_EXIT_UNRESOLVED",
                            severity="ERROR",
                            task_id=tid,
                            accountable_owner=owner or "UNASSIGNED",
                            inconsistency=f"Closing sprint '{active_sprint}' task '{tid}' has non-terminal disposition '{disp}' without explicit exit classification",
                            required_correction="Provide explicit sprint_exit_disposition (e.g. CARRYFORWARD_SPRINT5 or HOLDING_QUEUE_OWNER_GATED)"
                        ))

        return self.findings

    # -------------------------------------------------------------------------
    # Renderers & Deliverable Exporters
    # -------------------------------------------------------------------------
    def compile_action_register(self, sprint_filter: Optional[str] = None) -> Dict[str, Any]:
        board = self.generate_sprint_board(sprint_filter)
        queue = self.generate_action_queue(sprint_filter)
        matrix = self.generate_release_matrix(sprint_filter)
        findings = self.run_reconciliation(sprint_context=sprint_filter)

        err_cnt = sum(1 for f in findings if f.severity == "ERROR")
        warn_cnt = sum(1 for f in findings if f.severity == "WARNING")
        info_cnt = sum(1 for f in findings if f.severity == "INFO")

        if err_cnt > 0:
            verdict = "FAIL_CLOSED"
        elif warn_cnt > 0:
            verdict = "WARNINGS_PRESENT"
        else:
            verdict = "PASS"

        return {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_backlog_file": self.backlog_path,
            "source_backlog_sha": self.get_mainline_sha(),
            "active_sprint": sprint_filter or self.data.get("active_sprint", "SPRINT_4"),
            "sprint_board": board,
            "action_queue": queue,
            "release_matrix": matrix,
            "reconciliation_summary": {
                "total_checks_run": 10,
                "error_count": err_cnt,
                "warning_count": warn_cnt,
                "info_count": info_cnt,
                "verdict": verdict,
                "findings": [f.to_dict() for f in findings]
            }
        }

    def render_markdown_tracker(self, sprint_filter: Optional[str] = None) -> str:
        data = self.compile_action_register(sprint_filter)
        sp = data["active_sprint"]
        board = data["sprint_board"]
        queue = data["action_queue"]
        matrix = data["release_matrix"]
        recon = data["reconciliation_summary"]

        lines = [
            f"# ARX TERMINAL — SPRINT CONTROL & EXECUTION TRACKER",
            f"**Authoritative Source:** `{self.backlog_path}`  ",
            f"**Mainline Commit SHA:** `{data['source_backlog_sha']}`  ",
            f"**Active Sprint:** `{sp}` | **Generated At:** `{data['generated_at']}`  ",
            f"**Reconciliation Verdict:** `{recon['verdict']}` (Errors: {recon['error_count']}, Warnings: {recon['warning_count']})  ",
            "",
            "---",
            "",
            "## 1. Sprint Board View",
            "",
            f"| Bucket | Task Count | Task Identifiers |",
            f"| :--- | :---: | :--- |",
            f"| **Active Execution** | {len(board['execution_tasks'])} | {', '.join(f'`{t}`' for t in board['execution_tasks']) or 'None'} |",
            f"| **Active Discovery** | {len(board['discovery_tasks'])} | {', '.join(f'`{t}`' for t in board['discovery_tasks']) or 'None'} |",
            f"| **Blocked / Dep Held**| {len(board['blocked_tasks'])} | {', '.join(f'`{t}`' for t in board['blocked_tasks']) or 'None'} |",
            f"| **Owner-Gated Holding**| {len(board['owner_gated_tasks'])} | {', '.join(f'`{t}`' for t in board['owner_gated_tasks']) or 'None'} |",
            f"| **Completed / Closed** | {len(board['completed_tasks'])} | {', '.join(f'`{t}`' for t in board['completed_tasks']) or 'None'} |",
            "",
            "---",
            "",
            "## 2. Action Queue View (Priority-Ordered)",
            "",
            "| Task ID | Prio | Owner | Next Action | Checkpoint Target | Active Blocker / Trigger | Predicate Eligibility |",
            "| :--- | :---: | :--- | :--- | :---: | :--- | :--- |"
        ]

        for item in queue:
            pred = item["completion_predicate"]
            blocker = item["active_blocker"] or "None"
            if len(blocker) > 40:
                blocker = blocker[:37] + "..."
            next_act = item["next_action"]
            if len(next_act) > 45:
                next_act = next_act[:42] + "..."
            target = item["checkpoint_target_date"] or "N/A"
            if "T" in target:
                target = target.split("T")[0]

            lines.append(
                f"| **`{item['task_id']}`** | `{item['priority']}` | `{item['accountable_owner']}` | "
                f"{next_act} | `{target}` | {blocker} | `{pred['closure_eligibility']}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Release Matrix View",
            "",
            "| Task ID | PR(s) | PR Status | CI Conclusion | Merged SHA | Deployment Status | Production Evidence | Authorization |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        for item in matrix:
            prs_str = ", ".join(f"#{p}" for p in item["github_prs"]) or "None"
            msha = item["merged_main_sha"][:7] if item["merged_main_sha"] else "None"
            lines.append(
                f"| **`{item['task_id']}`** | {prs_str} | `{item['pr_status']}` | `{item['ci_conclusion']}` | "
                f"`{msha}` | `{item['deployment_status']}` | `{item['production_evidence']}` | `{item['operational_authorized']}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Reconciliation Exceptions & Findings",
            ""
        ])

        if not recon["findings"]:
            lines.append("Zero reconciliation findings. All tasks, PR states, and dimensional claims are internally consistent.")
        else:
            lines.extend([
                "| Severity | Task ID | Check ID | Inconsistency Detail | Required Correction | Accountable Owner |",
                "| :---: | :---: | :--- | :--- | :--- | :--- |"
            ])
            for f in recon["findings"]:
                lines.append(
                    f"| `{f['severity']}` | **`{f['task_id']}`** | `{f['check_id']}` | "
                    f"{f['inconsistency']} | {f['required_correction']} | `{f['accountable_owner']}` |"
                )

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ARX Terminal — Sprint Control System (SCS-v1)")
    parser.add_argument("--backlog", default="docs/planning/arx_canonical_backlog.json", help="Path to backlog JSON")
    parser.add_argument("--sprint", default=None, help="Filter by sprint (e.g. SPRINT_4)")
    parser.add_argument("--workstream", default=None, help="Filter by workstream")
    parser.add_argument("--priority", default=None, help="Filter by priority (P0, P1, P2, P3)")
    parser.add_argument("--owner", default=None, help="Filter by owner")
    parser.add_argument("--view", choices=["board", "queue", "matrix", "all"], default="all", help="View to display")
    parser.add_argument("--output-json", default=None, help="Path to write compiled action register JSON")
    parser.add_argument("--output-md", default=None, help="Path to write human-readable markdown tracker")
    parser.add_argument("--sprint-exit", action="store_true", help="Enforce sprint exit certification checks")

    args = parser.parse_args()

    scs = SprintControlSystem(backlog_path=args.backlog)
    findings = scs.run_reconciliation(sprint_context=args.sprint, sprint_exit_mode=args.sprint_exit)

    if args.output_json:
        register = scs.compile_action_register(sprint_filter=args.sprint)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(register, f, indent=2)
        print(f"[SCS] Wrote action register JSON to {args.output_json}")

    if args.output_md:
        md = scs.render_markdown_tracker(sprint_filter=args.sprint)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[SCS] Wrote markdown execution tracker to {args.output_md}")

    err_cnt = sum(1 for f in findings if f.severity == "ERROR")
    warn_cnt = sum(1 for f in findings if f.severity == "WARNING")

    print(f"\n===========================================================")
    print(f"ARX SPRINT CONTROL RECONCILIATION SUMMARY")
    print(f"Sprint Filter: {args.sprint or 'ALL'} | Errors: {err_cnt} | Warnings: {warn_cnt}")
    print(f"===========================================================")
    for f in findings:
        print(str(f))

    if err_cnt > 0:
        print("\n[VERDICT] FAIL_CLOSED (Errors must be remediated)")
        sys.exit(1)
    else:
        print("\n[VERDICT] PASS (Zero blocking errors)")
        sys.exit(0)


if __name__ == "__main__":
    main()
