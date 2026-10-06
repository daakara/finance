"""
scripts/research/etf_v2/openfigi_config.py

Authoritative configuration, single-authority operational store path resolver,
override governance, and startup validation for OpenFIGI v3 integration.

Invariants Enforced:
- INV-OPENFIGI-DB-01: All production OpenFIGI rate-limit coordination uses one physical SQLite file.
- INV-OPENFIGI-DB-02: Operational persistence uses the same physical SQLite file.
- INV-OPENFIGI-DB-03: Process CWD cannot alter production DB identity.
- INV-OPENFIGI-DB-04: Relative production overrides cannot create alternate DB files (REJECTED).
- INV-OPENFIGI-DB-05: Test databases are isolated from production.
- INV-OPENFIGI-DB-06: No fallback path creates a second operational DB.
- INV-OPENFIGI-DB-07: A limiter/persistence path mismatch fails closed at startup.
- INV-OPENFIGI-DB-08: Canonical ETF and security data stores remain unmutated and untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional, Union

# Authoritative project root derived from repository layout
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Logical default subpath relative to project root
DEFAULT_OPERATIONAL_DB_SUBPATH: Path = Path("data/operational/openfigi_operational.db")

# Deterministic default operational DB path anchored to repository root
DEFAULT_OPERATIONAL_DB_PATH: Path = (REPO_ROOT / DEFAULT_OPERATIONAL_DB_SUBPATH).resolve()

# Approved production runtime storage boundary
APPROVED_OPERATIONAL_STORAGE_DIR: Path = (REPO_ROOT / "data" / "operational").resolve()

# Canonical population database name and markers for firewall protection
CANONICAL_DB_NAME: str = "etf_v2_canonical_population"

# Environment variable override contracts
CANONICAL_OPERATIONAL_DB_ENV_VAR: str = "OPENFIGI_OPERATIONAL_DB"
LEGACY_RATE_LIMIT_ENV_VAR: str = "OPENFIGI_RATE_LIMIT_DB"


class OpenFIGIPathValidationError(ValueError):
    """Raised when an operational database path fails validation rules."""
    pass


class OpenFIGIStoreParityError(OpenFIGIPathValidationError):
    """Raised when rate limiter and persistence operational store paths mismatch."""
    pass


class CanonicalStoreContaminationError(RuntimeError):
    """Raised when an operational component attempts to connect to a canonical database."""
    pass


def _is_running_tests() -> bool:
    """Detects whether current execution is within an automated test harness."""
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def validate_openfigi_operational_db_path(
    path: Union[str, Path],
    is_test: Optional[bool] = None,
    allow_memory: bool = False,
) -> Path:
    """
    Validates the operational DB path before production OpenFIGI execution (AC-07-01 to AC-07-06).

    Validation confirms:
    1. The path is absolute (rejects relative production paths).
    2. The path is not in-memory (":memory:") unless running in explicit test context.
    3. The path does not touch canonical population stores (Canonical Firewall).
    4. In production, the path is inside the approved runtime storage boundary and not in temp dirs.
    5. Parent directory exists or can be created safely.
    6. DB file or parent directory is writable.
    7. No fallback operational DB is silently created or substituted.

    Raises:
        OpenFIGIPathValidationError: On any validation failure.
        CanonicalStoreContaminationError: On attempted canonical database access.

    Returns:
        Path: Validated, absolute, resolved operational DB path.
    """
    raw_str = str(path).strip()
    if raw_str == ":memory:" or raw_str.startswith(":memory:"):
        test_mode = is_test if is_test is not None else _is_running_tests()
        if not (test_mode or allow_memory):
            raise OpenFIGIPathValidationError(
                "AC-07-01 VIOLATION: In-memory database (':memory:') is prohibited in production."
            )
        return Path(raw_str)

    p = Path(raw_str)
    if not p.is_absolute():
        raise OpenFIGIPathValidationError(
            f"AC-07-01 VIOLATION: Operational DB path must be absolute. Relative path '{path}' is rejected."
        )

    resolved = p.resolve()

    # Canonical Firewall: Refuse any path touching canonical population store
    resolved_normalized = str(resolved).replace("\\", "/").lower()
    if "data/canonical" in resolved_normalized or CANONICAL_DB_NAME in resolved_normalized:
        raise CanonicalStoreContaminationError(
            f"INV-OPENFIGI-DB-08 / OFIGI-INV-014 VIOLATION: Operational component cannot connect "
            f"to canonical database path: {resolved}"
        )

    test_mode = is_test if is_test is not None else _is_running_tests()
    if not test_mode:
        # In production: must be inside approved runtime storage boundary
        try:
            resolved.relative_to(APPROVED_OPERATIONAL_STORAGE_DIR)
        except ValueError:
            raise OpenFIGIPathValidationError(
                f"AC-07-03 VIOLATION: Production DB path '{resolved}' is outside approved "
                f"runtime storage boundary ({APPROVED_OPERATIONAL_STORAGE_DIR})."
            )

        # Prohibit temp directories in production
        temp_markers = ["temp", "tmp", "appdata/local/temp", "pytest-of-"]
        for marker in temp_markers:
            if marker in resolved_normalized:
                raise OpenFIGIPathValidationError(
                    f"AC-07-03 VIOLATION: Temporary test path '{resolved}' cannot be used in production."
                )

    # Check usability: parent directory and file writability
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OpenFIGIPathValidationError(
            f"AC-07-04 VIOLATION: Cannot create parent directory for operational DB at '{resolved}': {exc}"
        ) from exc

    if resolved.exists():
        if not (os.access(resolved, os.R_OK) and os.access(resolved, os.W_OK)):
            raise OpenFIGIPathValidationError(
                f"AC-07-04 VIOLATION: Operational DB at '{resolved}' exists but is not read/write accessible."
            )
    else:
        if not os.access(resolved.parent, os.W_OK):
            raise OpenFIGIPathValidationError(
                f"AC-07-04 VIOLATION: Parent directory '{resolved.parent}' is not writable for operational DB creation."
            )

    return resolved


def validate_store_path_parity(
    limiter_path: Union[str, Path],
    persistence_path: Union[str, Path]
) -> bool:
    """
    Validates that rate limiter and operational persistence resolve the identical physical SQLite file.
    Rejects any mismatch to fail closed (AC-07-02, INV-OPENFIGI-DB-02, INV-OPENFIGI-DB-07).
    """
    p_lim = Path(limiter_path)
    p_per = Path(persistence_path)

    if str(p_lim) == ":memory:" and str(p_per) == ":memory:":
        return True

    if p_lim.resolve() != p_per.resolve():
        raise OpenFIGIStoreParityError(
            f"AC-07-02 / INV-OPENFIGI-DB-07 VIOLATION: Operational store path mismatch! "
            f"Rate limiter resolved '{p_lim.resolve()}', but persistence resolved '{p_per.resolve()}'."
        )
    return True


def resolve_openfigi_operational_db_path(
    db_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
    is_test: Optional[bool] = None,
    allow_memory: bool = False,
) -> Path:
    """
    Resolves the single authoritative operational SQLite database path for OpenFIGI components.

    Precedence order:
      1. Explicit constructor argument `db_path` (if provided)
      2. Canonical environment override: `OPENFIGI_OPERATIONAL_DB`
      3. Legacy environment alias: `OPENFIGI_RATE_LIMIT_DB`
      4. Repository-root anchored default: `REPO_ROOT / "data" / "operational" / "openfigi_operational.db"`

    Override Governance (Section 6):
      RELATIVE_OVERRIDE_ALLOWED = NO
      Production overrides must be absolute. A relative production override fails startup validation (AC-06-03).
      No production override is anchored to process CWD (AC-06-04).

    Returns:
        Path: Authoritative resolved absolute path.
    """
    target: Optional[Union[str, Path]] = None
    override_source: Optional[str] = None

    if db_path is not None:
        target = db_path
        override_source = "EXPLICIT_CONSTRUCTOR_ARG"
    else:
        env_op = os.environ.get(CANONICAL_OPERATIONAL_DB_ENV_VAR)
        if env_op and env_op.strip():
            target = env_op.strip()
            override_source = CANONICAL_OPERATIONAL_DB_ENV_VAR
        else:
            env_legacy = os.environ.get(LEGACY_RATE_LIMIT_ENV_VAR)
            if env_legacy and env_legacy.strip():
                target = env_legacy.strip()
                override_source = LEGACY_RATE_LIMIT_ENV_VAR

    if target is None:
        resolved = DEFAULT_OPERATIONAL_DB_PATH
    else:
        raw_str = str(target).strip()
        raw_normalized = raw_str.replace("\\", "/").lower()
        if "data/canonical" in raw_normalized or CANONICAL_DB_NAME in raw_normalized:
            raise CanonicalStoreContaminationError(
                f"INV-OPENFIGI-DB-08 / OFIGI-INV-014 VIOLATION: Operational component cannot connect "
                f"to canonical database path: {target}"
            )

        if raw_str == ":memory:" or raw_str.startswith(":memory:"):
            resolved = Path(raw_str)
        else:
            p = Path(raw_str)
            if not p.is_absolute():
                # AC-06-03 & AC-07-01: Relative production overrides are strictly rejected
                raise OpenFIGIPathValidationError(
                    f"AC-06-03 / AC-07-01 VIOLATION: Operational DB override from {override_source} "
                    f"must be absolute. Relative path '{raw_str}' is rejected."
                )
            resolved = p.resolve()

    if validate:
        resolved = validate_openfigi_operational_db_path(
            resolved,
            is_test=is_test,
            allow_memory=allow_memory
        )

    return resolved
