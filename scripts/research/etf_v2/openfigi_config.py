"""
scripts/research/etf_v2/openfigi_config.py

Authoritative configuration and operational store path authority for OpenFIGI v3 integration.
Guarantees single-authority, CWD-independent SQLite path resolution across all components:
- GlobalSQLiteRateLimiter
- OpenFIGIPersistenceRepository
- OpenFIGIClient
- Controlled live runners and background tasks

Invariants Enforced:
- OFIGI-ACT11: Operational and rate-limit path resolution is deterministic and CWD-independent.
- OFIGI-ACT12: Environment path overrides are unified and deterministically anchored to repository root.
- OFIGI-INV-001: Operational evidence only; cannot modify canonical population.
- OFIGI-INV-014: Operational evidence stored strictly in operational store, never in canonical tables.
- OFIGI-INV-016: Canonical database files remain byte-identical.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# Authoritative project root derived from repository layout
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Logical default subpath relative to project root
DEFAULT_OPERATIONAL_DB_SUBPATH: Path = Path("data/operational/openfigi_operational.db")

# Deterministic default operational DB path anchored to repository root
DEFAULT_OPERATIONAL_DB_PATH: Path = (REPO_ROOT / DEFAULT_OPERATIONAL_DB_SUBPATH).resolve()

# Canonical population database name and markers for firewall protection
CANONICAL_DB_NAME: str = "etf_v2_canonical_population"

# Environment variable override contracts
CANONICAL_OPERATIONAL_DB_ENV_VAR: str = "OPENFIGI_OPERATIONAL_DB"
LEGACY_RATE_LIMIT_ENV_VAR: str = "OPENFIGI_RATE_LIMIT_DB"


def resolve_openfigi_operational_db_path(
    db_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Resolves the authoritative operational SQLite database path for OpenFIGI components.

    Precedence order:
      1. Explicit constructor argument `db_path` (if provided)
      2. Canonical environment override: `OPENFIGI_OPERATIONAL_DB`
      3. Legacy environment alias: `OPENFIGI_RATE_LIMIT_DB`
      4. Repository-root anchored default: `REPO_ROOT / "data" / "operational" / "openfigi_operational.db"`

    All relative paths (whether defaults or overrides) are resolved deterministically relative
    to REPO_ROOT to guarantee complete working-directory independence (CWD-independent).
    Special SQLite in-memory paths (e.g. ":memory:") are preserved as-is.

    Returns:
        Path: Authoritative resolved absolute path (or in-memory Path).
    """
    target: Optional[Union[str, Path]] = None

    if db_path is not None:
        target = db_path
    else:
        env_op = os.environ.get(CANONICAL_OPERATIONAL_DB_ENV_VAR)
        if env_op and env_op.strip():
            target = env_op.strip()
        else:
            env_legacy = os.environ.get(LEGACY_RATE_LIMIT_ENV_VAR)
            if env_legacy and env_legacy.strip():
                target = env_legacy.strip()

    if target is None:
        return DEFAULT_OPERATIONAL_DB_PATH

    raw_str = str(target).strip()
    if raw_str == ":memory:" or raw_str.startswith(":memory:"):
        return Path(raw_str)

    p = Path(raw_str)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()
