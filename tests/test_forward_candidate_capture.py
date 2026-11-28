"""Offline PT-02 tests. Run with --noconftest to avoid unrelated market DB seeds."""
import copy
import json
import multiprocessing
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from analyst_dashboard.governance.forward_capture import (
    UNIVERSE_PATH, capture_batch, session_window,
)
from analyst_dashboard.governance.prospective_capture import compute_decision_id, DECISION_SCHEMA_VERSION
from analyst_dashboard.governance.storage import validate_forward_write_path

SYMBOLS = json.loads(UNIVERSE_PATH.read_text())["symbols"]
COHORT = "COHORT_FORWARD_PILOT_002"
NOW = datetime(2026, 11, 23, 21, 20, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Network and database access prohibited in capture tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    for key in ("ARX_FORCE_PRODUCTION_STORAGE_VALIDATION", "RAILWAY_ENVIRONMENT_NAME",
                "RAILWAY_ENVIRONMENT", "ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def batch():
    return {"schemaVersion": "1.0", "sessionDate": "2026-11-23", "engineSha": "a" * 40,
            "samplingMode": "SYSTEMATIC", "evaluations": {
                s: {"decisionState": "VALID_SETUP", "sourceTimestampUtc": "2026-11-23T21:00:00Z",
                    "observedAtUtc": "2026-11-23T21:10:00Z",
                    "plan": {"spot": 100., "corridorMin": 98., "corridorMax": 102.,
                             "stopLoss": 95., "takeProfit1": 110., "confluenceScore": 70.}}
                for s in SYMBOLS}}


def run(root, batch, **kw):
    args = dict(cohort_id=COHORT, start_session="2026-11-23", forward_root=root, now=NOW,
                initialize=True)
    args.update(kw)
    return capture_batch(json.dumps(batch).encode(), **args)


def stream(root):
    return root / COHORT / "capture_stream.jsonl"


def test_complete_batch_frozen_identity_and_retry(tmp_path, batch):
    assert run(tmp_path, batch)["written"] == 10
    original = stream(tmp_path).read_bytes()
    records = [json.loads(r) for r in original.splitlines()]
    assert [r["symbol"] for r in records] == SYMBOLS
    for r in records:
        assert r["decisionId"] == compute_decision_id(r["symbol"], r["cycleId"], "a" * 40,
                                                      DECISION_SCHEMA_VERSION)
        assert r["plan"] == batch["evaluations"][r["symbol"]]["plan"]
        assert not r["eligibleForOpportunityCapture"] and r["outcome"] is None
        assert r["evidenceOrigin"] == "OFFLINE_INPUT"
    batch["evaluations"] = dict(reversed(list(batch["evaluations"].items())))
    assert run(tmp_path, batch, now=NOW.replace(minute=25))["written"] == 0
    assert stream(tmp_path).read_bytes() == original
    batch["evaluations"]["NVDA"]["plan"]["spot"] = 101
    with pytest.raises(ValueError, match="Conflicting"):
        run(tmp_path, batch)
    assert stream(tmp_path).read_bytes() == original


def test_missing_failed_and_invalid_candidates_are_not_dropped(tmp_path, batch):
    del batch["evaluations"]["VOO"]
    batch["evaluations"]["QQQM"] = {"error": "provider timeout"}
    batch["evaluations"]["NVDA"]["sourceTimestampUtc"] = "2026-11-24T21:00:00Z"
    batch["evaluations"]["AAPL"]["plan"]["corridorMax"] = 1
    result = run(tmp_path, batch)
    assert result["written"] == 10 and result["missingInputs"] == 1
    assert result["invalidOrFailedInputs"] == 3
    states = {r["symbol"]: r for r in map(json.loads, stream(tmp_path).read_text().splitlines())}
    assert states["VOO"]["evaluationStatus"] == "MISSING_INPUT"
    assert states["QQQM"]["evaluationStatus"] == "FAILED_EVALUATION"
    assert states["NVDA"]["plan"] is None


@pytest.mark.parametrize("key,value", [("samplingMode", "USER_SELECTED"),
    ("engineSha", "unknown"), ("sessionDate", "2026-11-22"), ("sessionDate", "2026-11-26")])
def test_invalid_batch_never_writes(tmp_path, batch, key, value):
    batch[key] = value
    with pytest.raises(ValueError):
        run(tmp_path, batch)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("now", [NOW.replace(hour=20), NOW.replace(day=24, hour=3),
                                  NOW.replace(day=24), NOW.replace(tzinfo=None)])
def test_lookahead_and_clock_guards(tmp_path, batch, now):
    with pytest.raises(ValueError):
        run(tmp_path, batch, now=now)
    assert list(tmp_path.iterdir()) == []


def test_early_close_and_dst_use_exchange_calendar():
    close, next_open, _ = session_window("2026-11-27", "2026-11-23",
        datetime(2026, 11, 27, 18, 15, tzinfo=timezone.utc))
    assert close.hour == 18 and next_open.hour == 14
    close, _, _ = session_window("2026-03-09", "2026-03-09",
        datetime(2026, 3, 9, 20, 15, tzinfo=timezone.utc))
    assert close.hour == 20


def test_horizon_20_sessions_and_frozen_engine(tmp_path, batch):
    with pytest.raises(ValueError, match="20-session"):
        run(tmp_path, batch, start_session="2026-10-01")
    run(tmp_path, batch)
    old = stream(tmp_path).read_bytes()
    batch["engineSha"] = "b" * 40
    with pytest.raises(ValueError, match="Binding"):
        run(tmp_path, batch)
    assert stream(tmp_path).read_bytes() == old


def test_twenty_sessions_capture_200_records_without_entry_or_settlement(tmp_path, batch):
    import exchange_calendars as xcals
    cal = xcals.get_calendar("XNYS")
    for session in cal.sessions_in_range("2026-11-23", "2026-12-31")[:20]:
        label = session.date().isoformat()
        close = cal.session_close(label).to_pydatetime()
        from datetime import timedelta
        batch["sessionDate"] = label
        for item in batch["evaluations"].values():
            item["sourceTimestampUtc"] = close.isoformat()
            item["observedAtUtc"] = (close + timedelta(minutes=10)).isoformat()
        result = run(tmp_path, batch, now=close + timedelta(minutes=20))
        assert result["written"] == 10
    assert result["totalRecords"] == 200
    rows = list(map(json.loads, stream(tmp_path).read_text().splitlines()))
    assert len({r["decisionId"] for r in rows}) == 200
    assert all(r["simulationStatus"] == "NOT_STARTED" and r["outcome"] is None for r in rows)
    last = stream(tmp_path).read_bytes()
    batch["sessionDate"] = cal.next_session(label).date().isoformat()
    with pytest.raises(ValueError, match="20-session"):
        run(tmp_path, batch, now=cal.session_close(batch["sessionDate"]).to_pydatetime())
    assert stream(tmp_path).read_bytes() == last


def test_corrupt_stream_and_failed_write_do_not_overwrite(tmp_path, batch, monkeypatch):
    run(tmp_path, batch)
    file = stream(tmp_path)
    original = file.read_bytes()
    file.write_bytes(original[:-5])
    with pytest.raises(ValueError):
        run(tmp_path, batch)
    assert file.read_bytes() == original[:-5]
    file.write_bytes(original)
    bad = [json.loads(r) for r in original.splitlines()]
    bad[0]["plan"]["spot"] = 200
    file.write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(ValueError, match="integrity"):
        run(tmp_path, batch)
    file.write_bytes(original)
    next_batch = copy.deepcopy(batch)
    next_batch["sessionDate"] = "2026-11-24"
    def crash(*a):
        raise OSError("Injected pre-replace crash")
    monkeypatch.setattr("analyst_dashboard.governance.storage.os.replace", crash)
    with pytest.raises(OSError, match="Injected"):
        run(tmp_path, next_batch, now=NOW.replace(day=24))
    assert file.read_bytes() == original
    assert not list(file.parent.glob("*.tmp"))


def test_continuation_cannot_reset_missing_stream(tmp_path, batch):
    with pytest.raises(ValueError, match="initialization"):
        run(tmp_path, batch, initialize=False)
    run(tmp_path, batch)
    retained = next((stream(tmp_path).parent / "inputs").glob("*.json"))
    assert json.loads(retained.read_bytes()) == batch
    stream(tmp_path).unlink()
    with pytest.raises(ValueError, match="Missing capture stream"):
        run(tmp_path, batch)
    assert not stream(tmp_path).exists()


def test_empty_stream_or_missing_retained_input_blocks_continuation(tmp_path, batch):
    run(tmp_path, batch)
    original = stream(tmp_path).read_bytes()
    stream(tmp_path).write_bytes(b"")
    with pytest.raises(ValueError, match="Empty capture stream"):
        run(tmp_path, batch)
    stream(tmp_path).write_bytes(original)
    next((stream(tmp_path).parent / "inputs").glob("*.json")).unlink()
    with pytest.raises(OSError):
        run(tmp_path, batch)
    assert stream(tmp_path).read_bytes() == original


def test_production_and_namespace_blocked(tmp_path, batch, monkeypatch):
    with pytest.raises(ValueError):
        run(tmp_path, batch, cohort_id="COHORT_HISTORICAL_PILOT_001")
    outside = tmp_path.parent / "outside-pt02"
    (tmp_path / COHORT).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        run(tmp_path, batch)
    assert not outside.exists()
    monkeypatch.setenv("ARX_FORCE_PRODUCTION_STORAGE_VALIDATION", "1")
    with pytest.raises(ValueError, match="not activated"):
        run(tmp_path, batch)


def test_historical_protection_with_explicit_root(tmp_path, monkeypatch):
    # Regression: validator previously caught its own historical-path rejection.
    monkeypatch.setattr("analyst_dashboard.governance.storage.REPO_ROOT", str(tmp_path))
    protected = tmp_path / "tests/fixtures"
    protected.mkdir(parents=True)
    with pytest.raises(ValueError, match="protected historical"):
        validate_forward_write_path(str(protected / "x.json"), COHORT, str(protected))
    assert list(protected.iterdir()) == []


def _race(root, batch, queue):
    try:
        queue.put(run(Path(root), batch)["written"])
    except Exception as exc:
        queue.put(repr(exc))


def test_concurrent_processes_create_one_cycle(tmp_path, batch):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    workers = [ctx.Process(target=_race, args=(str(tmp_path), batch, queue)) for _ in range(6)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(20)
        assert w.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in workers) == [0, 0, 0, 0, 0, 10]
    assert len(stream(tmp_path).read_text().splitlines()) == 10


def test_cli_help_has_no_activation_switch():
    cli = Path(__file__).resolve().parents[1] / "scripts/capture_forward_candidates.py"
    result = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--input" in result.stdout and "--forward-root" in result.stdout
    assert "--now" not in result.stdout and "--activate" not in result.stdout
