"""
analyst_dashboard/security_master/config.py

Path resolver and configuration authority for ARX Terminal Security Master persistence.
Guarantees CWD-independent, deterministic SQLite path resolution and firewall separation.

Invariants Enforced:
- INV-SECMASTER-19: Security Master persistence does not silently reuse ETF v2 operational DB.
- Path resolution is deterministic and CWD-independent.
- Canonical firewall strictly prevents contamination with canonical population DBs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# Authoritative project root derived from repository layout
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Logical default subpath relative to project root
DEFAULT_SECURITY_MASTER_DB_SUBPATH: Path = Path("data/operational/security_master.db")

# Deterministic default operational DB path anchored to repository root
DEFAULT_SECURITY_MASTER_DB_PATH: Path = (REPO_ROOT / DEFAULT_SECURITY_MASTER_DB_SUBPATH).resolve()

# Environment variable override contracts
SECURITY_MASTER_DB_ENV_VAR: str = "SECURITY_MASTER_DB"
SECURITY_MASTER_TTL_ENV_VAR: str = "SECURITY_MASTER_TTL_SECONDS"

# Default freshness duration: 24 hours (86400s) for reference metadata
DEFAULT_SECURITY_MASTER_TTL_SECONDS: float = 86400.0

# Disallowed databases to enforce strict logical separation
FORBIDDEN_DB_MARKERS = [
    "openfigi_operational",
    "etf_v2_canonical_population",
    "data/canonical",
]


class SecurityMasterFirewallError(RuntimeError):
    """Raised when Security Master attempts to connect to an unauthorized or shared database."""
    pass


def resolve_security_master_db_path(
    db_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Resolves the authoritative SQLite database path for Security Master persistence.

    Precedence order:
      1. Explicit constructor argument `db_path`
      2. Environment variable: `SECURITY_MASTER_DB`
      3. Default path anchored to repository root: `REPO_ROOT / "data/operational/security_master.db"`

    All relative paths are anchored to REPO_ROOT to guarantee CWD-independence.
    Special SQLite in-memory paths (":memory:") are preserved as-is.

    Enforces logical firewall: raises SecurityMasterFirewallError if path reuses
    ETF v2 openfigi_operational.db or canonical population database.
    """
    target: Optional[Union[str, Path]] = None

    if db_path is not None:
        target = db_path
    else:
        env_val = os.environ.get(SECURITY_MASTER_DB_ENV_VAR)
        if env_val and env_val.strip():
            target = env_val.strip()

    if target is None:
        target_path = DEFAULT_SECURITY_MASTER_DB_PATH
    else:
        raw_str = str(target).strip()
        if raw_str == ":memory:" or raw_str.startswith(":memory:"):
            return Path(raw_str)
        p = Path(raw_str)
        if p.is_absolute():
            target_path = p.resolve()
        else:
            target_path = (REPO_ROOT / p).resolve()

    # Firewall enforcement: verify no reuse of ETF v2 operational or canonical DB
    norm_path = str(target_path).replace("\\", "/").lower()
    for marker in FORBIDDEN_DB_MARKERS:
        if marker in norm_path:
            raise SecurityMasterFirewallError(
                f"FIREWALL_VIOLATION: Security Master cannot connect to forbidden store {target_path} (matched '{marker}'). "
                f"Security Master persistence must be strictly separate from ETF v2."
            )

    return target_path


def get_security_master_ttl() -> float:
    """Returns configured TTL for Security Master record freshness."""
    env_ttl = os.environ.get(SECURITY_MASTER_TTL_ENV_VAR)
    if env_ttl and env_ttl.strip():
        try:
            return float(env_ttl.strip())
        except ValueError:
            pass
    return DEFAULT_SECURITY_MASTER_TTL_SECONDS
