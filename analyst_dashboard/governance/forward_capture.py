"""Manual, offline-input forward capture. No analytics, DB or broker invocation.

Imported batches are never certified natural-production opportunity evidence.
Identity helpers are reused without constructing the database capture engine.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from analyst_dashboard.governance.prospective_capture import (
    DECISION_SCHEMA_VERSION, compute_decision_id, compute_evaluation_cycle_id,
)
from analyst_dashboard.governance.storage import (
    is_production_runtime, safe_write_forward_pilot_file, validate_forward_write_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = REPO_ROOT / "docs/governance/ARX_FORWARD_PILOT_UNIVERSE_V1.json"
CONTRACT_PATH = REPO_ROOT / "docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json"
STATES = {"ACTIONABLE_SETUP", "VALID_SETUP", "INSUFFICIENT_DATA", "EVIDENCE_INCOMPLETE",
          "STALE_DATA", "UNVERIFIED"}
PLAN_FIELDS = ("spot", "corridorMin", "corridorMax", "stopLoss", "takeProfit1", "confluenceScore")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO timestamp with timezone")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Naive timestamp rejected")
    return dt.astimezone(timezone.utc)


def session_window(session, start_session, now):
    """Use exchange close/next open, including holidays, DST and early closes."""
    if now.tzinfo is None:
        raise ValueError("Naive capture clock rejected")
    cal = xcals.get_calendar("XNYS")
    if not cal.is_session(session) or not cal.is_session(start_session):
        raise ValueError("Capture/start date must be an XNYS session")
    # Explicit range avoids version-dependent count/offset semantics of sessions_window.
    end = date.fromisoformat(start_session) + timedelta(days=60)
    sessions = cal.sessions_in_range(start_session, end.isoformat())[:20]
    if len(sessions) != 20:
        raise ValueError("Calendar cannot establish the full 20-session horizon")
    labels = [s.date().isoformat() for s in sessions]
    if session not in labels:
        raise ValueError("Session outside frozen 20-session capture horizon")
    close = cal.session_close(session).to_pydatetime()
    next_open = cal.session_open(cal.next_session(session)).to_pydatetime()
    if now.astimezone(ZoneInfo("America/New_York")).date() != date.fromisoformat(session):
        raise ValueError("Historical/future capture date rejected; no backdating")
    if not close + timedelta(minutes=15) <= now <= next_open - timedelta(hours=12):
        raise ValueError("Outside post-close capture window / 12-hour next-open boundary")
    return close, next_open, labels.index(session) + 1


def normalize_evaluation(symbol, item, close, now):
    """Keep every candidate, including absent, failed and unusable evaluations."""
    if item is None:
        return {"decisionState": "INSUFFICIENT_DATA", "evaluationStatus": "MISSING_INPUT",
                "reason": "No retained evaluation supplied", "plan": None}
    try:
        if not isinstance(item, dict):
            raise ValueError("Evaluation must be an object")
        if item.get("error"):
            return {"decisionState": "INSUFFICIENT_DATA", "evaluationStatus": "FAILED_EVALUATION",
                    "reason": "Upstream evaluation failed; inspect retained batch", "plan": None}
        state = item["decisionState"]
        if state not in STATES:
            raise ValueError("Unsupported decision state")
        source = utc(item["sourceTimestampUtc"])
        observed = utc(item["observedAtUtc"])
        if not source <= observed <= now or source < close:
            raise ValueError("Source timing is future, stale or out of order")
        plan = item.get("plan")
        if plan is not None:
            if not isinstance(plan, dict):
                raise ValueError("Plan must be an object")
            for field in PLAN_FIELDS:
                v = plan[field]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError("Plan contains missing/nonfinite values")
            if not (0 < plan["stopLoss"] < plan["corridorMin"] <= plan["corridorMax"] < plan["takeProfit1"]
                    and plan["spot"] > 0 and 0 <= plan["confluenceScore"] <= 100):
                raise ValueError("Plan price/score bounds invalid")
        elif state in {"ACTIONABLE_SETUP", "VALID_SETUP"}:
            raise ValueError("Setup lacks a complete frozen plan")
        return {"decisionState": state, "evaluationStatus": "RECORDED", "plan": plan,
                "sourceTimestampUtc": source.isoformat(), "observedAtUtc": observed.isoformat()}
    except (KeyError, ValueError, TypeError):
        return {"decisionState": "EVIDENCE_INCOMPLETE", "evaluationStatus": "INVALID_INPUT",
                "reason": "Invalid decision, timing or plan; inspect retained batch", "plan": None}


def capture_batch(raw_batch: bytes, *, cohort_id: str, start_session: str,
                  forward_root: Path, now: datetime | None = None, initialize: bool = False):
    """Persist a ten-record cycle atomically; identical retries leave bytes intact.

    Local implementation only. Production activation/integration is a later task.
    `now` is dependency injection for offline tests; the CLI exposes no clock override.
    """
    if is_production_runtime():
        raise ValueError("Production forward capture is not activated")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Naive capture clock rejected")
    now = now.astimezone(timezone.utc)
    if not re.fullmatch(r"COHORT_FORWARD_PILOT_[A-Za-z0-9_-]+", cohort_id):
        raise ValueError("Invalid forward cohort namespace")
    batch = json.loads(raw_batch)
    if not isinstance(batch, dict):
        raise ValueError("Batch must be an object")
    canonical(batch)  # Reject NaN/Infinity before any filesystem access.
    if batch.get("samplingMode") != "SYSTEMATIC" or batch.get("schemaVersion") != "1.0":
        raise ValueError("Expected systematic batch schema 1.0")
    engine_sha = batch.get("engineSha", "")
    if not isinstance(engine_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", engine_sha):
        raise ValueError("Explicit engine SHA required")
    universe = json.loads(UNIVERSE_PATH.read_bytes())
    symbols = universe["symbols"]
    evaluations = batch.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) - set(symbols):
        raise ValueError("Evaluation map contains out-of-universe symbols")
    session = batch["sessionDate"]
    close, next_open, ordinal = session_window(session, start_session, now)
    binding = {"cohortId": cohort_id, "startSession": start_session, "engineSha": engine_sha,
               "universeSha256": digest(universe),
               "contractSha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()}
    # Canonical session identity is independent of invocation time or input row ordering.
    cycle = compute_evaluation_cycle_id("FORWARD_MANUAL_BATCH", close.isoformat(), digest(binding))
    payload_hash = digest(batch)
    records = []
    for symbol in symbols:
        records.append({**binding, "recordType": "FORWARD_CANDIDATE", "schemaVersion": "1.0",
                        "decisionId": compute_decision_id(symbol, cycle, engine_sha, DECISION_SCHEMA_VERSION),
                        "cycleId": cycle, "sessionDate": session, "sessionOrdinal": ordinal,
                        "nextSessionOpenUtc": next_open.isoformat(), "capturedAtUtc": now.isoformat(),
                        "symbol": symbol, "samplingMode": "SYSTEMATIC", "executionMode": "MANUAL_BATCH",
                        "evidenceOrigin": "OFFLINE_INPUT", "eligibleForOpportunityCapture": False,
                        "simulationStatus": "NOT_STARTED", "outcome": None,
                        "inputSha256": hashlib.sha256(raw_batch).hexdigest(), "payloadSha256": payload_hash,
                        **normalize_evaluation(symbol, evaluations.get(symbol), close, now)})
    for record in records:
        record["recordSha256"] = digest(record)
    root = Path(forward_root).absolute()
    if not root.is_dir():
        raise ValueError("Explicit existing forward root required")
    stream = root / cohort_id / "capture_stream.jsonl"
    validated = Path(validate_forward_write_path(str(stream), cohort_id, str(root)))
    # Root lock serializes creation and all cohort read/modify/write operations.
    lock_path = root / ".capture.lock"
    validate_forward_write_path(str(lock_path), cohort_id, str(root))
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(lock_path, flags, 0o600), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = validated.parent / "cohort.json"
        validate_forward_write_path(str(manifest), cohort_id, str(root))
        if manifest.exists():
            if json.loads(manifest.read_text()) != binding:
                raise ValueError("Binding mismatch with frozen cohort manifest")
            if not validated.exists():
                raise ValueError("Missing capture stream; continuation cannot reset cohort")
            if validated.stat().st_size == 0:
                raise ValueError("Empty capture stream; continuation cannot reset cohort")
        else:
            if not initialize or validated.parent.exists():
                raise ValueError("New cohort requires explicit initialization; existing state cannot reset")
            safe_write_forward_pilot_file(str(manifest), cohort_id, canonical(binding) + "\n", str(root))
        previous = validated.read_text() if validated.exists() else ""
        try:
            existing = [json.loads(line) for line in previous.splitlines()]
            if previous and not previous.endswith("\n"):
                raise ValueError("Truncated stream")
            if len(existing) % len(symbols) or len(existing) > 200:
                raise ValueError("Invalid stream record count")
            ids = set()
            for i in range(0, len(existing), len(symbols)):
                group = existing[i:i + len(symbols)]
                if [r["symbol"] for r in group] != symbols or len({r["cycleId"] for r in group}) != 1:
                    raise ValueError("Incomplete cycle")
                input_sha = group[0]["inputSha256"]
                if not isinstance(input_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", input_sha):
                    raise ValueError("Invalid retained input identity")
                source_path = validated.parent / "inputs" / (input_sha + ".json")
                validate_forward_write_path(str(source_path), cohort_id, str(root))
                source_bytes = source_path.read_bytes()
                if hashlib.sha256(source_bytes).hexdigest() != input_sha:
                    raise ValueError("Retained input integrity mismatch")
                source_payload_sha = digest(json.loads(source_bytes))
                for r in group:
                    if any(r[k] != v for k, v in binding.items()) or r["decisionId"] in ids:
                        raise ValueError("Binding mismatch or duplicate decision")
                    expected_id = compute_decision_id(r["symbol"], r["cycleId"], engine_sha,
                                                      DECISION_SCHEMA_VERSION)
                    if r["decisionId"] != expected_id or r["recordSha256"] != digest(
                            {k: v for k, v in r.items() if k != "recordSha256"}):
                        raise ValueError("Capture record integrity mismatch")
                    if r["inputSha256"] != input_sha or r["payloadSha256"] != source_payload_sha:
                        raise ValueError("Capture source binding mismatch")
                    ids.add(r["decisionId"])
            prior = [r for r in existing if r["cycleId"] == cycle]
            if prior:
                if len(prior) != len(symbols) or any(r["payloadSha256"] != payload_hash for r in prior):
                    raise ValueError("Conflicting retry cannot replace frozen decisions")
                return {"cycleId": cycle, "written": 0, "existing": 10, "totalRecords": len(existing)}
            if len(existing) >= 200:
                raise ValueError("Capture ceiling reached")
            if existing and session <= existing[-1]["sessionDate"]:
                raise ValueError("Out-of-order capture rejected")
        except (KeyError, TypeError, json.JSONDecodeError) as e:
            raise ValueError("Corrupt capture stream; manual reconciliation required") from e
        content = previous + "".join(canonical(r) + "\n" for r in records)
        retained_input = validated.parent / "inputs" / (hashlib.sha256(raw_batch).hexdigest() + ".json")
        if retained_input.exists():
            if retained_input.read_bytes() != raw_batch:
                raise ValueError("Retained input hash collision/corruption")
        else:
            safe_write_forward_pilot_file(str(retained_input), cohort_id, raw_batch.decode("utf-8"), str(root))
        safe_write_forward_pilot_file(str(stream), cohort_id, content, str(root))
        return {"cycleId": cycle, "written": 10, "existing": 0, "totalRecords": len(existing) + 10,
                "missingInputs": sum(r["evaluationStatus"] == "MISSING_INPUT" for r in records),
                "invalidOrFailedInputs": sum(r["evaluationStatus"] in {"INVALID_INPUT", "FAILED_EVALUATION"}
                                             for r in records)}
