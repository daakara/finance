"""
ARX Terminal — Sprint Control & Reconciliation Adversarial Test Suite
Tests:
  1. Schema contract conformance (Draft-07).
  2. Canonical backlog zero-task-loss and full coverage.
  3. Holding queue isolation and immutable platform holds preservation.
  4. Stale ARCH-001 audit deliverable reconciliation (Check 03).
  5. Stale PR merge reconciliation (Check 02).
  6. Unresolved PR #98 UX parity audit (Check 07).
  7. Conditional closures for ARX-S3-010 and ARX-S3-013.
  8. Missing Sprint 5 carryforward classifications (Check 08).
  9. Adversarial missing action owner fail-closed (Check 01).
  10. Adversarial unauthorized operational/deployment transitions (Check 09).
  11. Sprint exit gate certification and unresolved task detection (Check 10).
  12. Deterministic view generation and output idempotence.
"""

import copy
import json
import os
import tempfile
import pytest
import jsonschema

from tools.governance.sprint_control import (
    SprintControlSystem,
    ReconciliationFinding,
    IMMUTABLE_HOLDS,
)


@pytest.fixture
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def production_scs(repo_root):
    return SprintControlSystem(
        backlog_path="docs/planning/arx_canonical_backlog.json",
        repo_root=repo_root
    )


def test_01_schema_contract_conformance(repo_root):
    """
    Test 01: Verify docs/planning/arx_sprint_action_register.json conforms strictly
    to docs/planning/arx_sprint_action_register.schema.json.
    """
    schema_path = os.path.join(repo_root, "docs/planning/arx_sprint_action_register.schema.json")
    register_path = os.path.join(repo_root, "docs/planning/arx_sprint_action_register.json")

    assert os.path.exists(schema_path), f"Schema file missing: {schema_path}"
    assert os.path.exists(register_path), f"Register JSON file missing: {register_path}"

    with open(schema_path, "r", encoding="utf-8") as sf:
        schema = json.load(sf)
    with open(register_path, "r", encoding="utf-8") as rf:
        instance = json.load(rf)

    # Must validate cleanly without exception
    jsonschema.validate(instance=instance, schema=schema)
    assert instance["schema_version"] == "1.0.0"
    assert "sprint_board" in instance
    assert "action_queue" in instance
    assert "release_matrix" in instance
    assert "reconciliation_summary" in instance


def test_02_canonical_backlog_zero_task_loss(production_scs):
    """
    Test 02: Verify zero task loss across canonical backlog, sprint board,
    action queue, and release matrix. Total canonical tasks must equal 43.
    """
    assert len(production_scs.tasks) == 43, f"Expected 43 tasks, found {len(production_scs.tasks)}"

    board = production_scs.generate_sprint_board()
    assert board["total_tasks"] == 43

    bucket_sum = (
        len(board["execution_tasks"])
        + len(board["discovery_tasks"])
        + len(board["blocked_tasks"])
        + len(board["owner_gated_tasks"])
        + len(board["completed_tasks"])
    )
    assert bucket_sum == 43, f"Bucket sum {bucket_sum} does not equal total tasks 43"

    queue = production_scs.generate_action_queue()
    assert len(queue) == 43, f"Action queue has {len(queue)} items, expected 43"

    matrix = production_scs.generate_release_matrix()
    assert len(matrix) == 43, f"Release matrix has {len(matrix)} items, expected 43"

    # Verify every canonical task ID appears in queue and matrix
    canonical_ids = {t["task_id"] for t in production_scs.tasks}
    queue_ids = {item["task_id"] for item in queue}
    matrix_ids = {item["task_id"] for item in matrix}

    assert canonical_ids == queue_ids, f"Queue task ID mismatch: {canonical_ids ^ queue_ids}"
    assert canonical_ids == matrix_ids, f"Matrix task ID mismatch: {canonical_ids ^ matrix_ids}"


def test_03_holding_queue_isolation_and_immutable_holds(production_scs):
    """
    Test 03: Verify all 10 holding queue tasks remain isolated in owner_gated_tasks,
    are marked incomplete/blocked, and all 14 immutable holds are preserved.
    """
    board = production_scs.generate_sprint_board()
    holding_tasks = [
        t for t in production_scs.tasks
        if t.get("assigned_sprint") == "HOLDING_QUEUE_OWNER_GATED"
        or t.get("disposition") == "SPRINT4_OWNER_GATED"
    ]
    assert len(holding_tasks) == 10, f"Expected 10 holding queue tasks, found {len(holding_tasks)}"
    assert len(board["owner_gated_tasks"]) == 10

    for ht in holding_tasks:
        pred = production_scs.evaluate_completion_predicate(ht)
        assert pred["is_complete"] is False, f"Holding queue task {ht['task_id']} must not be complete"
        assert pred["closure_eligibility"] == "BLOCKED_HOLDING_QUEUE"
        assert ht.get("dimensions", {}).get("owner_authorization") in ("HOLD", "NOT_AUTHORIZED", "NOT_GRANTED")

    # Verify immutable platform holds
    assert len(IMMUTABLE_HOLDS) >= 14
    expected_holds = {
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
    for k, v in expected_holds.items():
        assert IMMUTABLE_HOLDS.get(k) == v, f"Immutable hold {k} mismatch: {IMMUTABLE_HOLDS.get(k)} != {v}"


def test_04_stale_arch001_reconciliation(production_scs):
    """
    Test 04: Verify Check 03 flags ARX-S4-ARCH-001 when audit deliverables
    exist but backlog dimensions record implementation='NOT_STARTED'.
    """
    findings = production_scs.run_reconciliation()
    arch1_findings = [f for f in findings if f.task_id == "ARX-S4-ARCH-001"]

    assert len(arch1_findings) > 0, "Expected finding for ARX-S4-ARCH-001"
    f = arch1_findings[0]
    assert f.check_id == "CHECK_INCONSISTENT_CLAIMS"
    assert f.severity == "WARNING"
    assert "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json" in f.inconsistency
    assert "implementation='DONE'" in f.required_correction


def test_05_stale_arch002_pr_reconciliation(production_scs):
    """
    Test 05: Verify Check 02 flags tasks where git shows PR is merged into main
    but backlog dimension repository_publication remains DRAFT_PR.
    """
    findings = production_scs.run_reconciliation()
    stale_pr_findings = [f for f in findings if f.check_id == "CHECK_STALE_PR_MERGE"]

    # In production backlog, PR #86 (ARX-S3-006), PR #87/#100 (ARX-S3-010), PR #76 (ARX-S3-013)
    # have merge commits in main but were recorded as DRAFT_PR.
    flagged_tasks = {f.task_id for f in stale_pr_findings}
    assert any(t in flagged_tasks for t in ("ARX-S3-006", "ARX-S3-010", "ARX-S3-013", "ARX-S4-ARCH-002")), (
        f"Expected stale PR merge findings, got: {flagged_tasks}"
    )


def test_06_unresolved_pr98_ux_parity_reconciliation(production_scs):
    """
    Test 06: Verify Check 07 flags ARX-S3-016 when unmerged PR #98 is cited
    while claiming deployment='DEPLOYED_DORMANT' without owner visual sign-off.
    """
    findings = production_scs.run_reconciliation()
    ux_findings = [f for f in findings if f.task_id == "ARX-S3-016" and f.check_id == "CHECK_INCOMPLETE_CRITERIA"]

    assert len(ux_findings) > 0, "Expected Check 07 finding for ARX-S3-016"
    f = ux_findings[0]
    assert "PR #98" in f.inconsistency
    assert "owner visual acceptance" in f.inconsistency
    assert f.severity == "WARNING"


def test_07_conditional_closures_s3_010_and_s3_013(production_scs):
    """
    Test 07: Verify ARX-S3-010 and ARX-S3-013 predicate evaluation and conditional
    closure behavior. In baseline backlog, S3-010 and S3-013 remain INELIGIBLE_ACTIVE_WORK_REMAINING
    due to partial implementation. When engineering is complete but under owner HOLD,
    they must evaluate to CONDITIONAL_CLOSURE_OWNER_HELD, and unauthorized operational
    execution must fail closed with an ERROR finding.
    """
    for tid in ("ARX-S3-010", "ARX-S3-013"):
        task = production_scs.task_map[tid]
        pred = production_scs.evaluate_completion_predicate(task)
        # On mainline, these tasks have partial implementation, so predicate correctly flags work remaining
        assert pred["closure_eligibility"] == "INELIGIBLE_ACTIVE_WORK_REMAINING"
        assert not pred["is_complete"]
        assert any("implementation" in d for d in pred["unmet_dimensions"])

    # Simulate completed engineering with operational hold (Gate 08/09)
    simulated_task = copy.deepcopy(production_scs.task_map["ARX-S3-010"])
    simulated_task["status"] = "COMPLETED"
    simulated_task["disposition"] = "CLOSED_VERIFIED"
    simulated_task["dimensions"]["requirements"] = "DONE"
    simulated_task["dimensions"]["implementation"] = "DONE"
    simulated_task["dimensions"]["integration"] = "DONE"
    simulated_task["dimensions"]["verification"] = "PASS"
    simulated_task["dimensions"]["repository_publication"] = "PUBLISHED_MERGED"
    simulated_task["dimensions"]["owner_authorization"] = "HOLD"
    simulated_task["dimensions"]["operational_execution"] = "NOT_PERFORMED"

    sim_pred = production_scs.evaluate_completion_predicate(simulated_task)
    assert sim_pred["closure_eligibility"] == "CONDITIONAL_CLOSURE_OWNER_HELD"
    assert "owner_authorization=HOLD" in sim_pred["held_dimensions"]

    # Adversarial test: Inject unauthorized operational execution on ARX-S3-010
    adversarial_backlog = copy.deepcopy(production_scs.data)
    for t in adversarial_backlog["tasks"]:
        if t["task_id"] == "ARX-S3-010":
            t["status"] = "COMPLETED"
            t["dimensions"]["operational_execution"] = "EXECUTED"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(adversarial_backlog, tf)
        tmp_name = tf.name

    try:
        adv_scs = SprintControlSystem(
            backlog_path=tmp_name,
            repo_root=production_scs.repo_root
        )
        findings = adv_scs.run_reconciliation()
        errs = [f for f in findings if f.task_id == "ARX-S3-010" and f.severity == "ERROR"]
        assert len(errs) > 0, "Must flag ERROR for unauthorized operational execution under hold"
        check_ids = {e.check_id for e in errs}
        assert "CHECK_INCOMPLETE_CRITERIA" in check_ids or "CHECK_UNAUTHORIZED_TRANSITIONS" in check_ids
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_08_missing_sprint5_carryforward_classifications(production_scs):
    """
    Test 08: Verify Check 08 flags Sprint 4 discovery tasks
    ARX-S1-006, ARX-S2-001, ARX-S2-005, ARX-S2-007, ARX-S2-008
    when sprint_exit_disposition is omitted.
    """
    findings = production_scs.run_reconciliation()
    carryforward_findings = [f for f in findings if f.check_id == "CHECK_UNCLASSIFIED_CARRYFORWARDS"]
    flagged_tasks = {f.task_id for f in carryforward_findings}

    for tid in ("ARX-S1-006", "ARX-S2-001", "ARX-S2-005", "ARX-S2-007", "ARX-S2-008"):
        assert tid in flagged_tasks, f"Expected task {tid} to be flagged for missing sprint exit classification"


def test_09_adversarial_check_missing_action_owner(production_scs):
    """
    Test 09: Verify Check 01 fails closed (ERROR) if an active task has no owner.
    """
    adversarial_backlog = copy.deepcopy(production_scs.data)
    for t in adversarial_backlog["tasks"]:
        if t["task_id"] == "ARX-S4-ARCH-002":
            t["owner"] = "UNASSIGNED"
            t["status"] = "IN_PROGRESS"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(adversarial_backlog, tf)
        tmp_name = tf.name

    try:
        adv_scs = SprintControlSystem(
            backlog_path=tmp_name,
            repo_root=production_scs.repo_root
        )
        findings = adv_scs.run_reconciliation()
        owner_errs = [f for f in findings if f.task_id == "ARX-S4-ARCH-002" and f.check_id == "CHECK_MISSING_ACTION_OWNER"]
        assert len(owner_errs) == 1
        assert owner_errs[0].severity == "ERROR"
        assert owner_errs[0].accountable_owner == "GOVERNANCE_LEAD"
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_10_adversarial_unauthorized_transitions_fail_closed(production_scs):
    """
    Test 10: Verify Check 09 fails closed (ERROR) if a task with owner_authorization='HOLD'
    claims deployment='DEPLOYED_ACTIVE' or operational_execution='EXECUTED'.
    """
    adversarial_backlog = copy.deepcopy(production_scs.data)
    for t in adversarial_backlog["tasks"]:
        if t["task_id"] == "ARX-S1-003":
            t["dimensions"]["deployment"] = "DEPLOYED_ACTIVE"
            t["dimensions"]["operational_execution"] = "EXECUTED"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(adversarial_backlog, tf)
        tmp_name = tf.name

    try:
        adv_scs = SprintControlSystem(
            backlog_path=tmp_name,
            repo_root=production_scs.repo_root
        )
        findings = adv_scs.run_reconciliation()
        trans_errs = [f for f in findings if f.task_id == "ARX-S1-003" and f.check_id == "CHECK_UNAUTHORIZED_TRANSITIONS"]
        assert len(trans_errs) >= 1
        for err in trans_errs:
            assert err.severity == "ERROR"
            assert "Fail closed" in err.required_correction
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_11_sprint_exit_gate_fail_closed_on_unresolved_tasks(production_scs):
    """
    Test 11: Verify Check 10 in sprint_exit_mode fails closed on unresolved tasks
    or tasks without explicit sprint exit disposition.
    """
    findings = production_scs.run_reconciliation(sprint_context="SPRINT_4", sprint_exit_mode=True)
    exit_errs = [f for f in findings if f.check_id == "CHECK_SPRINT_EXIT_UNRESOLVED"]

    assert len(exit_errs) > 0, "Sprint exit gate must detect unclassified or active tasks in closing sprint"
    for err in exit_errs:
        assert err.severity == "ERROR"


def test_12_deterministic_view_generation_and_idempotence(production_scs):
    """
    Test 12: Verify that generating the action queue, sprint board, and release
    matrix multiple times produces 100% deterministic, idempotent outputs.
    """
    reg1 = production_scs.compile_action_register(sprint_filter="SPRINT_4")
    reg2 = production_scs.compile_action_register(sprint_filter="SPRINT_4")

    # JSON representation must be identical (ignoring microsecond timestamps)
    assert reg1["sprint_board"] == reg2["sprint_board"]
    assert reg1["action_queue"] == reg2["action_queue"]
    assert reg1["release_matrix"] == reg2["release_matrix"]
    assert reg1["reconciliation_summary"]["findings"] == reg2["reconciliation_summary"]["findings"]

    md1 = production_scs.render_markdown_tracker(sprint_filter="SPRINT_4")
    md2 = production_scs.render_markdown_tracker(sprint_filter="SPRINT_4")

    # Split lines and compare all lines except timestamp line (line 3)
    lines1 = md1.splitlines()
    lines2 = md2.splitlines()
    assert len(lines1) == len(lines2)
    for i, (l1, l2) in enumerate(zip(lines1, lines2)):
        if "Generated At:" in l1:
            continue
        assert l1 == l2, f"Line {i} mismatch: '{l1}' != '{l2}'"
