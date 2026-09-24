"""ARX Production Governance & Experiment Storage Authority.

Single canonical authority for persistent volume resolution, physical device attestation,
fail-closed path contracts, and atomic versioned seed initialization.

GOVERNING INVARIANTS:
1. CANONICAL ROOT: Default production data root is /root/analyst_dashboard/data
   on the Railway ext4 persistent volume mount (/root).
2. PHYSICAL ATTESTATION: Validates device IDs (st_dev) against the persistent volume
   mount and proves distinctness from the container root overlayfs.
3. FAIL-CLOSED CONTRACT: In production runtime, any path resolving to container-local
   storage (/app, /tmp, or unmounted overlayfs) strictly raises a fatal exception.
4. ATOMIC SEED INITIALIZATION: If the persistent ledger does not exist, it is initialized
   from the versioned repository seed via atomic replace without overwriting existing data.
"""

import os
import sys
import json
import uuid
import stat
import shutil
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("arx.governance.storage")

# Canonical Defaults
DEFAULT_PRODUCTION_VOLUME_ROOT = "/root"
DEFAULT_PRODUCTION_DATA_DIR = "/root/analyst_dashboard/data"
DEFAULT_GOVERNANCE_DB_FILENAME = "governance.db"
DEFAULT_LEDGER_FILENAME = "paper_trading_ledger.json"

# Repository seed path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANONICAL_SEED_LEDGER_PATH = os.path.join(REPO_ROOT, "analyst_dashboard", "data", DEFAULT_LEDGER_FILENAME)


def is_production_runtime() -> bool:
    """Authoritatively determines if runtime is executing in production.
    
    Pre-conditions:
    - Dedicated production environment in Railway
    - Does NOT classify staging/preview environments as production
    - Honors explicit test forcing flag (ARX_FORCE_PRODUCTION_STORAGE_VALIDATION)
    """
    if os.getenv("ARX_FORCE_PRODUCTION_STORAGE_VALIDATION") == "1":
        return True

    railway_env_name = os.getenv("RAILWAY_ENVIRONMENT_NAME", "").strip().lower()
    railway_env = os.getenv("RAILWAY_ENVIRONMENT", "").strip().lower()
    env_name = os.getenv("ENVIRONMENT", "").strip().lower()

    # Explicit staging/preview exclusion
    for name in (railway_env_name, railway_env):
        if name in ("staging", "preview", "dev", "development", "test"):
            return False

    if railway_env_name == "production" or railway_env == "production":
        return True

    # If general ENVIRONMENT is production and running on Railway infrastructure
    if env_name == "production" and bool(os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_ENVIRONMENT_ID")):
        return True

    return False


def resolve_persistent_volume_root() -> str:
    """Resolves the configured Railway persistent volume mount path."""
    return os.path.abspath(os.getenv("RAILWAY_VOLUME_MOUNT_PATH", DEFAULT_PRODUCTION_VOLUME_ROOT))


def _check_mount_distinct_from_root(mount_path: str) -> Tuple[bool, int, int]:
    """Inspects filesystem device IDs to verify mount_path is physically mounted

    and distinct from the root container overlayfs.
    Returns (is_distinct, mount_dev_id, root_dev_id).
    """
    if not os.path.exists(mount_path):
        return False, -1, -1

    try:
        mount_dev = os.stat(mount_path).st_dev
        root_dev = os.stat("/").st_dev
        is_distinct = (mount_dev != root_dev)
        return is_distinct, mount_dev, root_dev
    except Exception as e:
        logger.error(f"Error checking mount distinctness for {mount_path}: {e}")
        return False, -1, -1


def resolve_data_root() -> str:
    """Resolves the authoritative data root directory.
    
    In production:
    - Strictly defaults to /root/analyst_dashboard/data
    - Any override (ARX_DATA_DIR) must physically reside under the persistent volume
    - Ephemeral fallback is strictly prohibited (fails closed)
    """
    custom_root = os.getenv("ARX_DATA_DIR")
    if custom_root:
        canonical_custom = os.path.abspath(custom_root)
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            # Must be a subpath of the persistent volume
            try:
                rel = os.path.relpath(canonical_custom, vol_root)
                if rel.startswith("..") or rel == ".":
                    raise RuntimeError(
                        f"FAIL_CLOSED: ARX_DATA_DIR='{canonical_custom}' must be a subdirectory of "
                        f"persistent volume mount '{vol_root}' in production."
                    )
            except ValueError:
                raise RuntimeError(
                    f"FAIL_CLOSED: ARX_DATA_DIR='{canonical_custom}' drive mismatch with "
                    f"persistent volume mount '{vol_root}'."
                )
        return canonical_custom

    if is_production_runtime():
        return os.path.join(resolve_persistent_volume_root(), "analyst_dashboard", "data")

    # Local development / test fallback inside repository tree
    return os.path.join(REPO_ROOT, "analyst_dashboard", "data")


def ensure_data_root() -> str:
    """Ensures the authoritative data root directory is created and writable.
    
    In production, verifies that:
    1. Persistent volume mount is present and distinct from container root.
    2. Data root directory is created on the persistent volume.
    3. Directory is writable via an atomic probe.
    Fails closed if any check fails.
    """
    data_root = resolve_data_root()

    if is_production_runtime():
        vol_root = resolve_persistent_volume_root()
        if not os.path.exists(vol_root):
            raise RuntimeError(
                f"FAIL_CLOSED: Production persistent volume mount '{vol_root}' does not exist."
            )
        is_distinct, vol_dev, root_dev = _check_mount_distinct_from_root(vol_root)
        if not is_distinct:
            raise RuntimeError(
                f"FAIL_CLOSED: Persistent volume root '{vol_root}' (dev={vol_dev}) is not "
                f"distinct from container root '/' (dev={root_dev}). Storage is ephemeral!"
            )

    try:
        os.makedirs(data_root, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"FAIL_CLOSED: Could not create data root directory '{data_root}': {e}")

    # Write probe verification
    probe_id = f".write_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    probe_path = os.path.join(data_root, probe_id)
    try:
        with open(probe_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe_path)
    except Exception as e:
        raise RuntimeError(f"FAIL_CLOSED: Data root directory '{data_root}' is not writable: {e}")

    if is_production_runtime():
        dir_dev = os.stat(data_root).st_dev
        vol_dev = os.stat(vol_root).st_dev
        if dir_dev != vol_dev:
            raise RuntimeError(
                f"FAIL_CLOSED: Data root '{data_root}' (dev={dir_dev}) does not reside on "
                f"persistent volume '{vol_root}' (dev={vol_dev})."
            )

    return data_root


def resolve_governance_db_path(custom_path: Optional[str] = None) -> str:
    """Resolves authoritative SQLite governance database path."""
    if custom_path:
        canonical_p = os.path.abspath(custom_path)
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            try:
                rel = os.path.relpath(canonical_p, vol_root)
                if rel.startswith(".."):
                    raise RuntimeError(
                        f"FAIL_CLOSED: Custom governance DB path '{canonical_p}' is outside "
                        f"production persistent volume '{vol_root}'."
                    )
            except ValueError:
                raise RuntimeError(
                    f"FAIL_CLOSED: Custom governance DB path '{canonical_p}' drive mismatch with '{vol_root}'."
                )
        return canonical_p

    env_p = os.getenv("ARX_GOVERNANCE_DB_PATH")
    if env_p:
        canonical_p = os.path.abspath(env_p)
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            try:
                rel = os.path.relpath(canonical_p, vol_root)
                if rel.startswith(".."):
                    raise RuntimeError(
                        f"FAIL_CLOSED: ARX_GOVERNANCE_DB_PATH='{canonical_p}' is outside "
                        f"production persistent volume '{vol_root}'."
                    )
            except ValueError:
                raise RuntimeError(
                    f"FAIL_CLOSED: ARX_GOVERNANCE_DB_PATH='{canonical_p}' drive mismatch with '{vol_root}'."
                )
        return canonical_p

    data_root = resolve_data_root()
    return os.path.join(data_root, DEFAULT_GOVERNANCE_DB_FILENAME)


def ensure_ledger_initialized(target_path: str) -> None:
    """Initializes persistent ledger from canonical seed if not already present.
    
    Guarantees:
    - Never overwrites an existing ledger.
    - Validates seed content integrity before copying.
    - Performs atomic write via temporary file and rename.
    """
    if os.path.exists(target_path):
        return

    parent_dir = os.path.dirname(target_path)
    os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(CANONICAL_SEED_LEDGER_PATH):
        raise FileNotFoundError(
            f"FAIL_CLOSED: Canonical seed ledger not found at '{CANONICAL_SEED_LEDGER_PATH}'"
        )

    with open(CANONICAL_SEED_LEDGER_PATH, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    # Invariant: Seed must contain exactly 10 signals and 0 clean prospective signals
    signals = seed_data.get("signals", [])
    if len(signals) != 10:
        raise ValueError(
            f"FAIL_CLOSED: Seed ledger corrupt: expected 10 signals, got {len(signals)}"
        )

    e1_prospective = [
        s for s in signals
        if s.get("epochId") == "ARX_PROSPECTIVE_VALIDATION_EPOCH_1"
        and s.get("provenanceCohort") == "PROSPECTIVE_CLEAN"
    ]
    e2_prospective = [
        s for s in signals
        if s.get("epochId") == "ARX_PROSPECTIVE_VALIDATION_EPOCH_2"
    ]
    if len(e1_prospective) > 0 or len(e2_prospective) > 0:
        raise ValueError(
            f"FAIL_CLOSED: Seed ledger contains unexpected prospective signals: "
            f"e1={len(e1_prospective)}, e2={len(e2_prospective)}"
        )

    tmp_path = os.path.join(parent_dir, f".ledger_init_{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
        logger.info(f"Initialized persistent prospective ledger at '{target_path}' from canonical seed.")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def resolve_ledger_path(custom_path: Optional[str] = None) -> str:
    """Resolves authoritative prospective paper trading ledger path."""
    if custom_path:
        canonical_p = os.path.abspath(custom_path)
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            try:
                rel = os.path.relpath(canonical_p, vol_root)
                if rel.startswith(".."):
                    raise RuntimeError(
                        f"FAIL_CLOSED: Custom ledger path '{canonical_p}' is outside "
                        f"production persistent volume '{vol_root}'."
                    )
            except ValueError:
                raise RuntimeError(
                    f"FAIL_CLOSED: Custom ledger path '{canonical_p}' drive mismatch with '{vol_root}'."
                )
        return canonical_p

    env_p = os.getenv("ARX_PAPER_TRADING_LEDGER_PATH")
    if env_p:
        canonical_p = os.path.abspath(env_p)
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            try:
                rel = os.path.relpath(canonical_p, vol_root)
                if rel.startswith(".."):
                    raise RuntimeError(
                        f"FAIL_CLOSED: ARX_PAPER_TRADING_LEDGER_PATH='{canonical_p}' is outside "
                        f"production persistent volume '{vol_root}'."
                    )
            except ValueError:
                raise RuntimeError(
                    f"FAIL_CLOSED: ARX_PAPER_TRADING_LEDGER_PATH='{canonical_p}' drive mismatch with '{vol_root}'."
                )
        return canonical_p

    data_root = resolve_data_root()
    target = os.path.join(data_root, DEFAULT_LEDGER_FILENAME)
    if is_production_runtime():
        ensure_ledger_initialized(target)
    return target


def attest_persistent_storage() -> Dict[str, Any]:
    """Attests physical storage persistence for governance database and prospective ledger.
    
    Used by Certification Check #13 and Activation Safety Guard.
    """
    vol_root = resolve_persistent_volume_root()
    is_prod = is_production_runtime()
    vol_present = os.path.exists(vol_root)
    is_distinct, vol_dev, root_dev = _check_mount_distinct_from_root(vol_root)

    gov_db_path = None
    db_resolution_err = None
    try:
        gov_db_path = resolve_governance_db_path()
    except Exception as e:
        db_resolution_err = str(e)

    ledger_path = None
    ledger_resolution_err = None
    try:
        ledger_path = resolve_ledger_path()
    except Exception as e:
        ledger_resolution_err = str(e)

    # Check governance DB physical location
    db_dev = -1
    db_on_vol = False
    if gov_db_path:
        if os.path.exists(gov_db_path):
            db_dev = os.stat(gov_db_path).st_dev
            db_on_vol = (db_dev == vol_dev) and vol_present and is_distinct
        else:
            # Check parent directory
            parent = os.path.dirname(gov_db_path)
            if os.path.exists(parent):
                db_dev = os.stat(parent).st_dev
                db_on_vol = (db_dev == vol_dev) and vol_present and is_distinct

    # Check prospective ledger physical location
    ledger_dev = -1
    ledger_on_vol = False
    if ledger_path:
        if os.path.exists(ledger_path):
            ledger_dev = os.stat(ledger_path).st_dev
            ledger_on_vol = (ledger_dev == vol_dev) and vol_present and is_distinct
        else:
            parent = os.path.dirname(ledger_path)
            if os.path.exists(parent):
                ledger_dev = os.stat(parent).st_dev
                ledger_on_vol = (ledger_dev == vol_dev) and vol_present and is_distinct

    # Evaluate overall validity
    is_valid = True
    error_msg = None

    if is_prod:
        if db_resolution_err:
            is_valid = False
            error_msg = f"Governance DB path invalid: {db_resolution_err}"
        elif ledger_resolution_err:
            is_valid = False
            error_msg = f"Prospective ledger path invalid: {ledger_resolution_err}"
        elif not vol_present:
            is_valid = False
            error_msg = f"Persistent volume mount '{vol_root}' is missing"
        elif not is_distinct:
            is_valid = False
            error_msg = f"Volume root '{vol_root}' is not distinct from container root '/' (dev={vol_dev})"
        elif not db_on_vol:
            is_valid = False
            error_msg = f"Governance DB '{gov_db_path}' (dev={db_dev}) is not on persistent volume (dev={vol_dev})"
        elif not ledger_on_vol:
            is_valid = False
            error_msg = f"Prospective ledger '{ledger_path}' (dev={ledger_dev}) is not on persistent volume (dev={vol_dev})"
    else:
        # Non-production: local development is valid by definition
        is_valid = True

    return {
        "isProduction": is_prod,
        "expectedVolumeRoot": vol_root,
        "volumeMountPresent": vol_present,
        "volumeDeviceId": vol_dev,
        "containerRootDeviceId": root_dev,
        "volumeDistinctFromContainerRoot": is_distinct,
        "governanceDbPath": gov_db_path,
        "governanceDbDeviceId": db_dev,
        "governanceDbOnVolume": db_on_vol,
        "ledgerPath": ledger_path,
        "ledgerDeviceId": ledger_dev,
        "ledgerOnVolume": ledger_on_vol,
        "isValid": is_valid,
        "error": error_msg,
    }


def is_storage_persistent() -> bool:
    """Quick boolean attestation helper."""
    return attest_persistent_storage().get("isValid", False)
