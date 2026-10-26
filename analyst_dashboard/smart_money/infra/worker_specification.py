"""
ARX Terminal — Smart Money Worker Specification & Preflight Validator.

Provides schema validation, environment preflight checks, and memory sizing contracts
for the dedicated background worker service (SM_GATE_01).
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SPEC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "railway-worker.json")
)


def load_worker_specification(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads and validates the Railway worker specification JSON."""
    target_path = path or SPEC_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Worker specification file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Validate essential schema sections
    required_keys = ["serviceName", "deploy", "resources", "network", "fencingAndLifecycle", "environment"]
    for key in required_keys:
        if key not in spec:
            raise ValueError(f"Worker specification missing required top-level key: '{key}'")

    return spec


def validate_worker_preflight_environment(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Validates environment variables against the disabled-by-default production safety policy.
    Returns status receipt: SAFE_DISABLED, AUTHORIZED_LIVE, or INVALID_CONFIGURATION.
    """
    current_env = env if env is not None else os.environ

    role = current_env.get("ARX_SEC_EGRESS_ROLE", "DISABLED").strip().upper()
    schedule_enabled = current_env.get("SMART_MONEY_SCHEDULE_ENABLED", "FALSE").strip().upper() == "TRUE"
    live_ingestion = current_env.get("SMART_MONEY_LIVE_INGESTION", "FALSE").strip().upper() == "TRUE"
    user_agent = current_env.get("SEC_USER_AGENT", "").strip()
    emergency_stop = current_env.get("ARX_EMERGENCY_STOP_ACTIVE", "FALSE").strip().upper() == "TRUE"

    is_disabled_safe = (
        role != "OWNER"
        and not schedule_enabled
        and not live_ingestion
    )

    is_placeholder_user_agent = (
        not user_agent
        or user_agent.upper() == "DISABLED"
        or "sample company" in user_agent.lower()
        or "@sample" in user_agent.lower()
        or "admin@financeterminal.internal" in user_agent.lower()
    )

    if emergency_stop:
        return {
            "status": "EMERGENCY_STOP_ACTIVE",
            "can_start": False,
            "role": role,
            "message": "Worker execution forbidden: ARX_EMERGENCY_STOP_ACTIVE is TRUE.",
        }

    if is_disabled_safe:
        return {
            "status": "SAFE_DISABLED",
            "can_start": False,
            "role": role,
            "schedule_enabled": schedule_enabled,
            "live_ingestion": live_ingestion,
            "is_placeholder_user_agent": is_placeholder_user_agent,
            "message": "Worker is disabled-by-default. No SEC egress or scheduled ingestion permitted.",
        }

    if role == "OWNER":
        if is_placeholder_user_agent:
            return {
                "status": "INVALID_CONFIGURATION",
                "can_start": False,
                "role": role,
                "message": "Worker configured as OWNER but SEC_USER_AGENT is missing, invalid, or a placeholder.",
            }
        return {
            "status": "AUTHORIZED_LIVE",
            "can_start": True,
            "role": role,
            "schedule_enabled": schedule_enabled,
            "live_ingestion": live_ingestion,
            "message": "Worker authorized for live SEC ingestion under exclusive advisory lock.",
        }

    return {
        "status": "INVALID_CONFIGURATION",
        "can_start": False,
        "role": role,
        "message": f"Invalid role configuration: '{role}'.",
    }
