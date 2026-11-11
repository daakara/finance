"""Fail-closed PostgreSQL client and custom archive compatibility contract.

ARX production PostgreSQL is pinned to major 16. A custom-format dump made by
newer pg_dump can require a newer pg_restore and emit SQL not accepted by PG16.
Use official, unmodified PostgreSQL 16 client binaries for the PG16 recovery
path; do not patch binaries or claim an upload is a tested restore.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from analyst_dashboard.smart_money.storage.config import SMART_MONEY_POSTGRES_MAJOR

_VERSION = re.compile(r"\b(?:pg_dump|pg_dumpall|pg_restore) \(PostgreSQL\) (\d+)\.(\d+)(?:\.\d+)?\b")
_SERVER = re.compile(r"^(\d+)(?:\.|$)")


def server_major(version: str) -> int:
    match = _SERVER.match(version.strip())
    if not match:
        raise RuntimeError("POSTGRES_SERVER_VERSION_UNRECOGNIZED")
    return int(match.group(1))


def inspect_client(binary_path: str, tool: str, *, required_major: int = SMART_MONEY_POSTGRES_MAJOR) -> str:
    """Return validated version string without ever reporting process stderr."""
    if tool not in {"pg_dump", "pg_dumpall", "pg_restore"}:
        raise RuntimeError("POSTGRES_CLIENT_TOOL_UNSUPPORTED")
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("POSTGRES_CLIENT_VERSION_UNAVAILABLE") from None
    if result.returncode != 0:
        raise RuntimeError("POSTGRES_CLIENT_VERSION_UNAVAILABLE")
    version = result.stdout.strip()
    match = _VERSION.search(version)
    if not match or tool not in version:
        raise RuntimeError("POSTGRES_CLIENT_VERSION_UNRECOGNIZED")
    if int(match.group(1)) != required_major:
        raise RuntimeError("POSTGRES_CLIENT_MAJOR_MISMATCH")
    return version


def inspect_custom_archive(path: str, *, required_major: int = SMART_MONEY_POSTGRES_MAJOR) -> str:
    """Confirm PGDMP header version before upload or isolated restoration.

    Archive 1.15 is PostgreSQL 16; 1.16 is PostgreSQL 17. This policy is
    intentionally limited to the currently supported PG16 production contract.
    """
    try:
        header = Path(path).open("rb").read(7)
    except OSError:
        raise RuntimeError("POSTGRES_ARCHIVE_UNREADABLE") from None
    if len(header) < 7 or header[:5] != b"PGDMP":
        raise RuntimeError("POSTGRES_ARCHIVE_HEADER_INVALID")
    major, minor = header[5], header[6]
    if required_major != 16 or (major, minor) != (1, 15):
        raise RuntimeError("POSTGRES_ARCHIVE_FORMAT_UNSUPPORTED")
    return f"{major}.{minor}"


def validate_backup_toolchain(pg_version: str, dump_binary: str, *, globals_binary: str | None = None) -> dict:
    """Fail before doing a database dump if runtime toolchain violates PG16 policy."""
    if server_major(pg_version) != SMART_MONEY_POSTGRES_MAJOR:
        raise RuntimeError("POSTGRES_SERVER_MAJOR_MISMATCH")
    dump_version = inspect_client(dump_binary, "pg_dump")
    globals_version = inspect_client(globals_binary, "pg_dumpall") if globals_binary else None
    return {
        "pg_dump_client_version": dump_version,
        "pg_dump_client_major": SMART_MONEY_POSTGRES_MAJOR,
        "pg_dumpall_client_version": globals_version,
    }
