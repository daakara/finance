"""SM-ACT-020 / issue #265: pure, synthetic PostgreSQL backup compatibility checks.

No live database, SEC, R2, or production role access.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from analyst_dashboard.smart_money.storage.backup import pg_client_compat as compat


@pytest.mark.parametrize("version,expected", [
    ("16.2", 16),
    ("16.15 (Debian 16.15-1)", 16),
    ("17.11", 17),
])
def test_postgres_server_major(version, expected):
    assert compat.server_major(version) == expected


@pytest.mark.parametrize("version", ["", "not-a-version", "server=16"])
def test_server_version_rejects_unrecognized(version):
    with pytest.raises(RuntimeError, match="SERVER_VERSION_UNRECOGNIZED"):
        compat.server_major(version)


def test_official_pg16_dump_is_accepted(monkeypatch):
    def fake_run(argv, **kwargs):
        assert argv == ["/opt/pg16/bin/pg_dump", "--version"]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout="pg_dump (PostgreSQL) 16.15\n", stderr="")
    monkeypatch.setattr(compat.subprocess, "run", fake_run)
    assert compat.inspect_client("/opt/pg16/bin/pg_dump", "pg_dump") == "pg_dump (PostgreSQL) 16.15"


def test_pg17_dump_is_fail_closed_even_when_server_is_pg16(monkeypatch):
    monkeypatch.setattr(
        compat.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="pg_dump (PostgreSQL) 17.11", stderr=""),
    )
    with pytest.raises(RuntimeError, match="POSTGRES_CLIENT_MAJOR_MISMATCH"):
        compat.validate_backup_toolchain("16.15", "pg_dump")


@pytest.mark.parametrize("bad_result", [
    SimpleNamespace(returncode=1, stdout="", stderr="postgresql://secret@db/a"),
    SimpleNamespace(returncode=0, stdout="unexpected output", stderr=""),
    SimpleNamespace(returncode=0, stdout="pg_restore (PostgreSQL) 16.15", stderr=""),
])
def test_invalid_client_version_does_not_leak_credentials(monkeypatch, bad_result):
    monkeypatch.setattr(compat.subprocess, "run", lambda *_a, **_k: bad_result)
    with pytest.raises(RuntimeError) as exc:
        compat.inspect_client("pg_dump", "pg_dump")
    assert "secret" not in str(exc.value)


def test_version_probe_timeout_fails_closed(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["pg_dump"], 10)
    monkeypatch.setattr(compat.subprocess, "run", blocked)
    with pytest.raises(RuntimeError, match="VERSION_UNAVAILABLE"):
        compat.inspect_client("pg_dump", "pg_dump")


@pytest.mark.parametrize("header,ok", [
    (b"PGDMP\x01\x0f", True),
    (b"PGDMP\x01\x10", False),
    (b"PGDMP\x01\x0e", False),
    (b"INVALID", False),
    (b"PGDMP", False),
])
def test_archive_header_compatibility(tmp_path: Path, header, ok):
    archive = tmp_path / "dump.custom"
    archive.write_bytes(header + b"synthetic")
    if ok:
        assert compat.inspect_custom_archive(str(archive)) == "1.15"
    else:
        with pytest.raises(RuntimeError, match="ARCHIVE_"):
            compat.inspect_custom_archive(str(archive))


def test_all_tools_must_match_pg16(monkeypatch):
    def fake_probe(path, tool, **kwargs):
        assert kwargs == {}
        if tool == "pg_dumpall":
            return "pg_dumpall (PostgreSQL) 16.15"
        return "pg_dump (PostgreSQL) 16.15"
    monkeypatch.setattr(compat, "inspect_client", fake_probe)
    data = compat.validate_backup_toolchain("16.15", "/usr/bin/pg_dump", globals_binary="/usr/bin/pg_dumpall")
    assert data["pg_dump_client_major"] == 16
    assert data["pg_dumpall_client_version"].endswith("16.15")


def test_wrong_server_major_rejected_before_client_probe(monkeypatch):
    monkeypatch.setattr(compat, "inspect_client", lambda *_a, **_kw: pytest.fail("no probe allowed"))
    with pytest.raises(RuntimeError, match="SERVER_MAJOR_MISMATCH"):
        compat.validate_backup_toolchain("17.11", "pg_dump")
